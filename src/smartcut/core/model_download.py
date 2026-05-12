"""Shared Whisper model download helper with visible progress.

`faster_whisper.download_model` already auto-downloads on first use, but in
an MCP-server context the user sees nothing happening for many minutes
because progress goes silently to stderr. This module:

1. Resolves the HuggingFace repo for a faster-whisper model size.
2. Detects whether the model is already cached.
3. If not cached, prints a clear status banner + drives the download with
   an explicit tqdm progress bar to stderr (safe — JSON-RPC uses stdout).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


APPROX_SIZE = {
    "tiny": "75 MB",
    "tiny.en": "75 MB",
    "base": "145 MB",
    "base.en": "145 MB",
    "small": "485 MB",
    "small.en": "485 MB",
    "medium": "1.5 GB",
    "medium.en": "1.5 GB",
    "large-v1": "3 GB",
    "large-v2": "3 GB",
    "large-v3": "3 GB",
    "large": "3 GB",
    "large-v3-turbo": "1.6 GB",
    "turbo": "1.6 GB",
    "distil-large-v2": "1.5 GB",
    "distil-large-v3": "1.5 GB",
    "distil-medium.en": "780 MB",
    "distil-small.en": "350 MB",
}


def _resolve_repo_id(size_or_id: str) -> str:
    """Map a faster-whisper model size to its HuggingFace repo id."""
    if "/" in size_or_id:
        return size_or_id
    try:
        from faster_whisper.utils import _MODELS
    except ImportError:
        return f"Systran/faster-whisper-{size_or_id}"
    return _MODELS.get(size_or_id, f"Systran/faster-whisper-{size_or_id}")


def is_model_cached(size_or_id: str, cache_dir: Optional[str] = None) -> bool:
    """Return True if the model's core weight file is already in HF cache.

    We probe `model.bin` (always present in faster-whisper CT2 repos);
    `try_to_load_from_cache` returns the file path on hit or None on miss.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    repo_id = _resolve_repo_id(size_or_id)
    hit = try_to_load_from_cache(repo_id, "model.bin", cache_dir=cache_dir)
    return hit is not None and Path(str(hit)).exists()


def ensure_model_downloaded(
    size_or_id: str,
    cache_dir: Optional[str] = None,
    progress: bool = True,
) -> str:
    """Make sure the model is on disk, downloading with a tqdm bar if needed.

    Returns the local snapshot path so callers can hand it to
    `WhisperModel(path, ...)`. Progress is printed to stderr (never stdout,
    which would corrupt MCP JSON-RPC).
    """
    repo_id = _resolve_repo_id(size_or_id)

    if is_model_cached(size_or_id, cache_dir=cache_dir):
        if progress:
            print(
                f"[smartcut] Whisper model '{size_or_id}' already cached.",
                file=sys.stderr,
                flush=True,
            )
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True)

    size_hint = APPROX_SIZE.get(size_or_id, "?")
    if progress:
        cache_loc = cache_dir or "~/.cache/huggingface/hub/"
        print("", file=sys.stderr, flush=True)
        print("=" * 72, file=sys.stderr, flush=True)
        print(
            f"[smartcut] Downloading Whisper model '{size_or_id}' (~{size_hint})",
            file=sys.stderr,
            flush=True,
        )
        print(f"           from {repo_id}", file=sys.stderr, flush=True)
        print(f"           into {cache_loc}", file=sys.stderr, flush=True)
        print(
            "           (one-time download; future runs read from cache)",
            file=sys.stderr,
            flush=True,
        )
        print("=" * 72, file=sys.stderr, flush=True)

    from huggingface_hub import snapshot_download

    try:
        from tqdm.auto import tqdm as _tqdm

        class _StderrTqdm(_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("file", sys.stderr)
                kwargs.setdefault("leave", True)
                kwargs.setdefault("mininterval", 0.5)
                super().__init__(*args, **kwargs)

        tqdm_class = _StderrTqdm
    except ImportError:
        tqdm_class = None

    path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        tqdm_class=tqdm_class,
    )

    if progress:
        print(
            f"[smartcut] Model '{size_or_id}' ready at {path}",
            file=sys.stderr,
            flush=True,
        )
    return path
