"""Content-addressed cache for Whisper transcriptions.

Whisper on `large-v3` against a 5-minute clip costs ~30-60s on CPU. When
the user re-runs `transcribe_project` (tweaking captions, changing chunk
sizes, swapping styles, etc.) we don't want to re-transcribe.

The cache key combines:

* the source audio file's content hash (so re-encodes invalidate)
* the backend + model + compute_type
* every decoder knob that materially affects output

Lookups are O(1) JSON loads. Writes are atomic (write+rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from smartcut.core.whisper_client import (
    WhisperResult,
    WhisperSegment,
    WhisperWord,
)


CACHE_VERSION = 2  # bump when serialization format changes


def _default_cache_dir() -> Path:
    override = os.environ.get("SMARTCUT_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "smartcut" / "transcripts"


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_cache_key(audio_path: Path, params: dict[str, Any]) -> str:
    """Stable cache key for an (audio, decoder-params) pair."""
    audio_hash = _hash_file(audio_path)
    # Canonical JSON for deterministic keys regardless of dict order.
    params_blob = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.blake2b(digest_size=16)
    h.update(audio_hash.encode())
    h.update(b"|")
    h.update(params_blob.encode())
    return f"{audio_hash}-{h.hexdigest()}"


def _result_to_dict(result: WhisperResult) -> dict:
    return {
        "version": CACHE_VERSION,
        "language": result.language,
        "duration": result.duration,
        "language_probability": result.language_probability,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "words": [asdict(w) for w in s.words],
            }
            for s in result.segments
        ],
    }


def _dict_to_result(blob: dict) -> WhisperResult:
    segments: list[WhisperSegment] = []
    for s in blob.get("segments", []):
        words = [WhisperWord(**w) for w in s.get("words", [])]
        segments.append(WhisperSegment(
            start=s.get("start", 0.0),
            end=s.get("end", 0.0),
            text=s.get("text", ""),
            words=words,
            avg_logprob=s.get("avg_logprob", 0.0),
            no_speech_prob=s.get("no_speech_prob", 0.0),
        ))
    return WhisperResult(
        language=blob.get("language", "unknown"),
        duration=blob.get("duration", 0.0),
        segments=segments,
        language_probability=blob.get("language_probability", 1.0),
    )


def load(key: str, cache_dir: Optional[Path] = None) -> Optional[WhisperResult]:
    cache_dir = cache_dir or _default_cache_dir()
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("version") != CACHE_VERSION:
            return None
        return _dict_to_result(blob)
    except (json.JSONDecodeError, OSError):
        return None


def save(key: str, result: WhisperResult, cache_dir: Optional[Path] = None) -> Path:
    cache_dir = cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob = _result_to_dict(result)
    final = cache_dir / f"{key}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".tx-", dir=str(cache_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp_name, final)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return final


def stats(cache_dir: Optional[Path] = None) -> dict:
    """Cheap inspection helper for the manage tool."""
    cache_dir = cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        return {"path": str(cache_dir), "entries": 0, "bytes": 0}
    files = list(cache_dir.glob("*.json"))
    return {
        "path": str(cache_dir),
        "entries": len(files),
        "bytes": sum(f.stat().st_size for f in files),
    }


def clear(cache_dir: Optional[Path] = None) -> int:
    """Remove all cached transcriptions. Returns count deleted."""
    cache_dir = cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except OSError as e:
            print(f"[smartcut] cache clear: failed to unlink {f}: {e}", file=sys.stderr)
    return count
