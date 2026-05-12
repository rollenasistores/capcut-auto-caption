"""MCP tools for working with CapCut projects — smart cut via auto-generated subtitles."""

import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from smartcut.config import (
    DUPLICATE_SIMILARITY_THRESHOLD,
    MICROSECONDS_PER_SECOND,
    SILENCE_THRESHOLD_SEC,
    get_settings,
)
from smartcut.core.capcut_finder import (
    find_project_by_name,
    get_capcut_drafts_dir,
    list_projects,
)
from smartcut.core.capcut_reader import CapCutProject, TextStyle, normalize_caption_text
from smartcut.core.caption_style import (
    DEFAULT_FILLERS,
    get_preset,
    is_filler,
    list_presets,
    strip_fillers_from_words,
)
from smartcut.core.models import CapCutSubtitleSegment


# ---------------------------------------------------------------------------
# Tool: list_capcut_projects
# ---------------------------------------------------------------------------

async def list_capcut_projects(
    drafts_dir: Optional[str] = None,
) -> dict:
    """List all CapCut projects in drafts directory."""
    drafts_path = Path(drafts_dir) if drafts_dir else None
    detected_dir = drafts_path or get_capcut_drafts_dir()

    if detected_dir is None:
        return {
            "projects": [],
            "drafts_dir": None,
            "message": "CapCut drafts directory not found. Is CapCut installed?",
        }

    projects = list_projects(detected_dir, require_content=True)
    all_projects = list_projects(detected_dir, require_content=False)
    incomplete_count = len(all_projects) - len(projects)

    message = f"Found {len(projects)} projects"
    if incomplete_count > 0:
        message += f" ({incomplete_count} incomplete — missing draft_info.json)"

    return {
        "projects": [p.model_dump() for p in projects],
        "drafts_dir": str(detected_dir),
        "count": len(projects),
        "message": message if projects else "No complete projects found",
    }


# ---------------------------------------------------------------------------
# Tool: open_capcut_project
# ---------------------------------------------------------------------------

async def open_capcut_project(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
) -> dict:
    """Open existing CapCut project and return its structure."""
    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path  # error dict

    project = CapCutProject.load(path)
    data = project.to_project_data()
    subtitles = project.get_subtitle_segments()

    return {
        "project": data.model_dump(),
        "auto_subtitles_count": len(subtitles),
        "auto_subtitles": [
            {"text": s.text, "start_sec": round(s.timeline_start_sec, 2), "end_sec": round(s.timeline_end_sec, 2)}
            for s in subtitles
        ],
        "message": (
            f"Loaded '{data.project_name}' — "
            f"{len(data.video_segments)} video segments, "
            f"{len(subtitles)} auto-subtitles"
        ),
    }


# ---------------------------------------------------------------------------
# Tool: smart_cut_project
# ---------------------------------------------------------------------------

async def smart_cut_project(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    silence_threshold_sec: float = SILENCE_THRESHOLD_SEC,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    use_openai: bool = False,
) -> dict:
    """
    Smart cut a CapCut project using its auto-generated subtitles.

    Reads CapCut's subtitles to heuristically find gaps and duplicate takes,
    then removes them directly in the project (no backup copy).

    Set use_openai=True for GPT-enhanced duplicate detection (requires OPENAI_API_KEY).
    """
    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path  # error dict

    project = CapCutProject.load(path)

    # Read auto-generated subtitles
    subtitles = project.get_subtitle_segments()
    if not subtitles:
        return {
            "error": "No auto-generated subtitles found in project",
            "suggestion": (
                "Open this project in CapCut, select the video track, "
                "and use Text → Auto Captions to generate subtitles first. "
                "Then run this tool again."
            ),
            "project_path": str(path),
            "project_name": project.project_name,
        }

    threshold_us = int(silence_threshold_sec * MICROSECONDS_PER_SECOND)

    # Step 1: Find gaps (silences between subtitles, including start/end)
    gap_ranges = find_gaps(subtitles, threshold_us, project.duration_us)

    # Step 2: Find duplicate takes
    duplicate_ranges = find_duplicate_takes(subtitles, similarity_threshold)

    # Step 3: Optional OpenAI enhancement
    if use_openai:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError(
                "use_openai=True but OPENAI_API_KEY is not set. "
                "Set it in environment or .env file, or use use_openai=False for heuristic mode."
            )
        llm_ranges = _detect_duplicates_with_llm(subtitles, settings.openai_api_key)
        if llm_ranges:
            duplicate_ranges = llm_ranges

    # Step 4: Merge all ranges
    all_ranges = gap_ranges + duplicate_ranges
    merged_ranges = merge_time_ranges(all_ranges)

    if not merged_ranges:
        return {
            "project_path": str(path),
            "project_name": project.project_name,
            "message": "No cuts needed — no significant gaps or duplicates found",
            "stats": {
                "gaps_found": 0,
                "duplicates_found": 0,
                "time_saved": "0:00",
            },
        }

    # Step 5: Calculate stats before cutting
    total_cut_us = sum(end - start for start, end in merged_ranges)
    original_duration_us = project.duration_us

    # Step 6: Apply cuts
    project.remove_time_ranges(merged_ranges)

    # Step 7: Save directly (no backup)
    project.save()

    return {
        "project_path": str(path),
        "project_name": project.project_name,
        "stats": {
            "original_duration": _format_duration_us(original_duration_us),
            "final_duration": _format_duration_us(original_duration_us - total_cut_us),
            "time_saved": _format_duration_us(total_cut_us),
            "gaps_removed": len(gap_ranges),
            "duplicates_removed": len(duplicate_ranges),
            "total_cuts": len(merged_ranges),
            "subtitles_analyzed": len(subtitles),
            "used_openai": use_openai,
        },
        "cuts_detail": [
            {
                "start_sec": round(s / MICROSECONDS_PER_SECOND, 2),
                "end_sec": round(e / MICROSECONDS_PER_SECOND, 2),
                "duration_sec": round((e - s) / MICROSECONDS_PER_SECOND, 2),
            }
            for s, e in merged_ranges
        ],
        "message": (
            f"Smart cut applied to '{project.project_name}'. "
            f"Removed {len(gap_ranges)} gaps and {len(duplicate_ranges)} duplicate takes, "
            f"saving {_format_duration_us(total_cut_us)}."
        ),
    }


# ---------------------------------------------------------------------------
# Tool: transcribe_project
# ---------------------------------------------------------------------------

async def transcribe_project(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    language: Optional[str] = None,
    backend: Optional[str] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    min_word_probability: Optional[float] = None,
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
    preprocess_audio: bool = True,
    use_cache: bool = True,
    dry_run: bool = False,
    strip_fillers: bool = False,
    extra_fillers: Optional[list[str]] = None,
    caption_preset: Optional[str] = None,
    also_short_captions: bool = True,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_duration_sec: Optional[float] = None,
    prefer_sentences: bool = False,
) -> dict:
    """
    Generate auto-subtitles for a CapCut project using Whisper.

    The pipeline does six things now:

      1. Extract & ASR-preprocess audio (loudness norm + speech filters)
      2. Look up the transcription cache (skip Whisper if hit)
      3. Run Whisper with Tagalog-tuned defaults (see LocalWhisperClient)
      4. Optionally strip Tagalog/English filler words and stutters
      5. Map source-time → timeline-time per clip (every clip captioned)
      6. If ``dry_run`` is False, write the auto-subtitle track + optional
         short-caption track shaped by a style preset

    The cache + dry-run combination is the recommended workflow: first call
    with ``dry_run=True`` produces a *preview* (no project write), then a
    second call with the same params reuses the cache instantly and
    actually writes the captions.
    """
    from smartcut.core.ffmpeg_utils import (
        FFmpegError,
        check_ffmpeg_installed,
        extract_audio,
        extract_audio_for_asr,
    )
    from smartcut.core import transcript_cache

    if also_short_captions:
        if min_words is not None and max_words is not None:
            if min_words < 1 or max_words < min_words:
                return {
                    "error": (
                        f"Invalid short-caption range: min_words={min_words}, "
                        f"max_words={max_words}. Require min_words >= 1 and "
                        "max_words >= min_words."
                    )
                }

    # ---- Resolve preset (if any) and merge chunking defaults --------------
    preset = None
    if caption_preset:
        try:
            preset = get_preset(caption_preset)
        except ValueError as e:
            return {"error": str(e)}
    eff_min_words = min_words if min_words is not None else (preset.min_words if preset else 2)
    eff_max_words = max_words if max_words is not None else (preset.max_words if preset else 4)
    eff_max_chars = max_chars if max_chars is not None else (preset.max_chars if preset else None)
    eff_max_dur = max_duration_sec if max_duration_sec is not None else (preset.max_duration_sec if preset else None)

    settings = get_settings()
    resolved_backend = (backend or settings.whisper_backend or "local").lower()
    if resolved_backend not in ("local", "openai"):
        return {"error": f"Unknown backend '{resolved_backend}' (use 'local' or 'openai')"}

    if not check_ffmpeg_installed():
        return {"error": "ffmpeg not found on PATH — install ffmpeg first"}

    eff_language = language or settings.whisper_language
    eff_initial_prompt = initial_prompt or settings.whisper_initial_prompt
    eff_hotwords = hotwords or settings.whisper_hotwords
    eff_min_word_prob = (
        min_word_probability
        if min_word_probability is not None
        else settings.whisper_min_word_probability
    )
    eff_compute_type = compute_type or settings.whisper_compute_type
    eff_device = device or settings.whisper_device
    chosen_model = model_size or settings.whisper_local_model

    # Lazy client construction — only spin up Whisper if we miss the cache.
    client = None

    def _get_client():
        nonlocal client
        if client is not None:
            return client
        if resolved_backend == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("backend='openai' requires OPENAI_API_KEY")
            from smartcut.core.whisper_client import WhisperClient
            client = WhisperClient(api_key=settings.openai_api_key)
        else:
            from smartcut.core.model_download import is_model_cached
            from smartcut.core.whisper_local import LocalWhisperClient
            if not is_model_cached(chosen_model):
                import sys as _sys
                print(
                    f"[smartcut] First-run: '{chosen_model}' not in HF cache — "
                    f"downloading now (see progress in server stderr).",
                    file=_sys.stderr,
                    flush=True,
                )
            client = LocalWhisperClient(
                model_size=chosen_model,
                device=eff_device,
                compute_type=eff_compute_type,
            )
        return client

    transcribe_kwargs = {
        "language": eff_language,
        "initial_prompt": eff_initial_prompt,
        "hotwords": eff_hotwords,
        "min_word_probability": eff_min_word_prob,
        "beam_size": beam_size,
        "condition_on_previous_text": condition_on_previous_text,
    }
    cache_params = dict(transcribe_kwargs)
    cache_params["__backend__"] = resolved_backend
    cache_params["__model__"] = chosen_model if resolved_backend == "local" else "whisper-1"
    cache_params["__compute_type__"] = eff_compute_type or ("int8" if eff_device == "cpu" else "float16")
    cache_params["__preprocess__"] = bool(preprocess_audio)

    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path

    project = CapCutProject.load(path)
    video_segments = [s for s in project.get_video_segments() if s.source_path]
    if not video_segments:
        return {
            "error": "No video segments found on the timeline",
            "project_path": str(path),
            "project_name": project.project_name,
        }

    unique_sources = {}
    for vs in video_segments:
        p = Path(vs.source_path)
        unique_sources.setdefault(p, p)

    missing = [str(p) for p in unique_sources if not p.exists()]
    if missing:
        return {
            "error": "Source video file(s) not found on disk",
            "missing_paths": missing,
            "suggestion": "Re-link the media in CapCut or restore the files.",
        }

    extra_fillers_set = {w.strip().lower() for w in (extra_fillers or []) if w.strip()}

    all_sentences: list[dict] = []
    per_segment_stats = []
    per_source_lang: dict[str, str] = {}
    cache_hits = 0
    fillers_stripped = 0
    word_probs: list[float] = []
    low_conf_segments: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="smartcut_") as tmpdir:
        tmp = Path(tmpdir)

        # Transcribe each unique source ONCE (cache-first); segments below reuse.
        transcripts: dict[Path, object] = {}
        for idx, src in enumerate(unique_sources):
            audio_file = tmp / f"{idx:03d}_{src.stem}.wav"
            try:
                if preprocess_audio:
                    extract_audio_for_asr(src, audio_file)
                else:
                    extract_audio(src, audio_file)
            except FFmpegError as e:
                return {"error": f"Audio extraction failed for {src}: {e}"}

            cache_key = transcript_cache.compute_cache_key(audio_file, cache_params)
            cached = transcript_cache.load(cache_key) if use_cache else None
            if cached is not None:
                transcripts[src] = cached
                cache_hits += 1
            else:
                try:
                    transcripts[src] = _get_client().transcribe(audio_file, **transcribe_kwargs)
                except RuntimeError as e:
                    return {"error": str(e)}
                if use_cache:
                    try:
                        transcript_cache.save(cache_key, transcripts[src])
                    except OSError:
                        pass  # cache write failure is non-fatal
            per_source_lang[str(src)] = transcripts[src].language

        # Map every clip on the timeline (including repeated/split clips).
        for vs in video_segments:
            src = Path(vs.source_path)
            result = transcripts.get(src)
            if result is None:
                continue

            source_start_sec = vs.source_start
            source_end_sec = vs.source_end
            timeline_start_sec = vs.timeline_start
            timeline_end_sec = vs.timeline_end

            sentences_for_segment = 0
            for ws in result.segments:
                if ws.end <= source_start_sec or ws.start >= source_end_sec:
                    continue

                start_on_timeline = ws.start - source_start_sec + timeline_start_sec
                end_on_timeline = ws.end - source_start_sec + timeline_start_sec
                start_on_timeline = max(start_on_timeline, timeline_start_sec)
                end_on_timeline = min(end_on_timeline, timeline_end_sec)
                if end_on_timeline <= start_on_timeline:
                    continue

                mapped_words = []
                for w in ws.words:
                    if w.end <= source_start_sec or w.start >= source_end_sec:
                        continue
                    w_start = w.start - source_start_sec + timeline_start_sec
                    w_end = w.end - source_start_sec + timeline_start_sec
                    w_start = max(w_start, start_on_timeline)
                    w_end = min(w_end, end_on_timeline)
                    if w_end <= w_start:
                        continue
                    word_probs.append(getattr(w, "probability", 1.0))
                    mapped_words.append({
                        "word": w.word,
                        "start": w_start,
                        "end": w_end,
                        "probability": getattr(w, "probability", 1.0),
                    })

                if strip_fillers and mapped_words:
                    before = len(mapped_words)
                    mapped_words = strip_fillers_from_words(
                        mapped_words, extra=extra_fillers_set,
                    )
                    fillers_stripped += before - len(mapped_words)

                if not mapped_words:
                    continue

                seg_text = normalize_caption_text(
                    " ".join(w["word"] for w in mapped_words)
                )

                avg_logprob = getattr(ws, "avg_logprob", 0.0)
                if avg_logprob < -1.0:
                    low_conf_segments.append({
                        "start": round(start_on_timeline, 2),
                        "end": round(end_on_timeline, 2),
                        "text": seg_text,
                        "avg_logprob": round(avg_logprob, 3),
                    })

                all_sentences.append({
                    "start": start_on_timeline,
                    "end": end_on_timeline,
                    "text": seg_text,
                    "words": [
                        {"word": w["word"], "start": w["start"], "end": w["end"]}
                        for w in mapped_words
                    ],
                })
                sentences_for_segment += 1

            per_segment_stats.append({
                "video": str(src),
                "segment_id": vs.id,
                "timeline_start": round(timeline_start_sec, 2),
                "timeline_end": round(timeline_end_sec, 2),
                "source_start": round(source_start_sec, 2),
                "source_end": round(source_end_sec, 2),
                "language": per_source_lang.get(str(src), ""),
                "sentences": sentences_for_segment,
            })

    if not all_sentences:
        return {
            "error": "Whisper returned no usable segments",
            "project_path": str(path),
            "project_name": project.project_name,
            "cache_hits": cache_hits,
        }

    all_sentences.sort(key=lambda s: s["start"])

    short_caption_chunks: list[dict] = []
    if also_short_captions:
        # Build chunks from the in-memory sentences (works in dry_run too).
        subtitle_view = _sentences_to_subtitle_view(all_sentences)
        short_caption_chunks = build_short_caption_chunks(
            subtitle_view,
            min_words=eff_min_words,
            max_words=eff_max_words,
            max_chars=eff_max_chars,
            max_duration_sec=eff_max_dur,
            prefer_sentences=prefer_sentences,
        )

    quality_stats = {
        "avg_word_probability": round(sum(word_probs) / len(word_probs), 3) if word_probs else None,
        "low_conf_segments": len(low_conf_segments),
        "fillers_stripped": fillers_stripped,
        "cache_hits": cache_hits,
        "cache_misses": len(unique_sources) - cache_hits,
    }

    base_payload = {
        "project_path": str(path),
        "project_name": project.project_name,
        "stats": {
            "backend": resolved_backend,
            "model": cache_params["__model__"],
            "compute_type": cache_params["__compute_type__"],
            "preprocessing": preprocess_audio,
            "unique_sources_transcribed": len(unique_sources),
            "clips_captioned": sum(1 for s in per_segment_stats if s["sentences"] > 0),
            "clips_total": len(per_segment_stats),
            "sentences_added": len(all_sentences),
            "short_captions_added": len(short_caption_chunks),
            "language_hint": eff_language or "auto",
            "initial_prompt_used": bool(eff_initial_prompt),
            "hotwords_used": bool(eff_hotwords),
            "min_word_probability": eff_min_word_prob,
            "caption_preset": preset.name if preset else None,
            **quality_stats,
        },
        "per_segment": per_segment_stats,
        "low_confidence_segments": low_conf_segments[:20],
    }

    if dry_run:
        base_payload["dry_run"] = True
        base_payload["preview"] = {
            "subtitles": [
                {"text": s["text"], "start": round(s["start"], 2), "end": round(s["end"], 2)}
                for s in all_sentences
            ],
            "short_captions": [
                {"text": c["text"], "start": round(c["start"], 2), "end": round(c["end"], 2)}
                for c in short_caption_chunks
            ],
        }
        base_payload["message"] = (
            f"[DRY RUN] Would write {len(all_sentences)} subtitles"
            + (f" + {len(short_caption_chunks)} short captions" if short_caption_chunks else "")
            + f" to '{project.project_name}'. Re-run without dry_run to apply "
            "(cache will make it near-instant)."
        )
        return base_payload

    project.add_auto_subtitle_track(all_sentences)

    if short_caption_chunks:
        style = preset.style if preset else TextStyle(
            font_size=15, bold=True, position_y=0.5,
            background_color=None, background_alpha=0.0,
        )
        project.add_text_track(short_caption_chunks, style=style)

    project.save()

    base_payload["sample"] = [
        {"text": s["text"], "start": round(s["start"], 2), "end": round(s["end"], 2)}
        for s in all_sentences[:5]
    ]
    base_payload["message"] = (
        f"Transcribed {len(unique_sources)} unique source(s) across "
        f"{len(per_segment_stats)} clip(s) → {len(all_sentences)} subtitles"
        + (f" + {len(short_caption_chunks)} short captions" if short_caption_chunks else "")
        + (f" (preset='{preset.name}')" if preset else "")
        + f" for '{project.project_name}'."
    )
    return base_payload


def _sentences_to_subtitle_view(sentences: list[dict]) -> list[CapCutSubtitleSegment]:
    """Adapter so in-memory sentence dicts can feed ``build_short_caption_chunks``.

    The chunker normally consumes :class:`CapCutSubtitleSegment` (read from
    a saved project). For dry-run / cache-hit paths we want to chunk
    without writing first, so we build the same shape from word lists.
    """
    out: list[CapCutSubtitleSegment] = []
    for s in sentences:
        words = s.get("words", [])
        if not words:
            continue
        base_us = int(s["start"] * MICROSECONDS_PER_SECOND)
        starts_ms = [int(max(w["start"] - s["start"], 0) * 1000) for w in words]
        ends_ms = [int(max(w["end"] - s["start"], 0) * 1000) for w in words]
        out.append(CapCutSubtitleSegment(
            segment_id="",
            material_id="",
            text=s.get("text", ""),
            words_text=[w["word"] for w in words],
            words_start_ms=starts_ms,
            words_end_ms=ends_ms,
            timeline_start_us=base_us,
            timeline_duration_us=int((s["end"] - s["start"]) * MICROSECONDS_PER_SECOND),
            recognize_task_id="preview",
        ))
    return out


# ---------------------------------------------------------------------------
# Tool: generate_short_captions
# ---------------------------------------------------------------------------

_PUNCT_BREAK = (".", ",", "!", "?", ";", ":", "—", "–")


def _join_caption_words(words: list[str]) -> str:
    """Join words with single spaces, delegating whitespace normalization
    to the shared :func:`normalize_caption_text` helper."""
    return normalize_caption_text(" ".join(words))


async def generate_short_captions(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_duration_sec: Optional[float] = None,
    prefer_sentences: bool = False,
    caption_preset: Optional[str] = None,
    strip_fillers: bool = False,
    extra_fillers: Optional[list[str]] = None,
    font_size: Optional[int] = None,
    bold: Optional[bool] = None,
    position_y: Optional[float] = None,
) -> dict:
    """
    Generate short-form captions on a new text track from CapCut's auto-subtitles.

    Chunking + styling is driven by a preset (``caption_preset``) and/or
    explicit overrides. Explicit args win over the preset. If no preset is
    given and no overrides, defaults to TikTok-style 2-4 word chunks.
    Original subtitles are preserved.
    """
    preset = None
    if caption_preset:
        try:
            preset = get_preset(caption_preset)
        except ValueError as e:
            return {"error": str(e)}

    eff_min_words = min_words if min_words is not None else (preset.min_words if preset else 2)
    eff_max_words = max_words if max_words is not None else (preset.max_words if preset else 4)
    eff_max_chars = max_chars if max_chars is not None else (preset.max_chars if preset else None)
    eff_max_dur = max_duration_sec if max_duration_sec is not None else (preset.max_duration_sec if preset else None)

    if eff_min_words < 1 or eff_max_words < eff_min_words:
        return {"error": f"Invalid range: min_words={eff_min_words}, max_words={eff_max_words}"}

    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path

    project = CapCutProject.load(path)
    subtitles = project.get_subtitle_segments()
    if not subtitles:
        return {
            "error": "No auto-generated subtitles found in project",
            "suggestion": (
                "Open this project in CapCut and use Text → Auto Captions "
                "to generate subtitles first — or run transcribe_project."
            ),
            "project_path": str(path),
            "project_name": project.project_name,
        }

    fillers_removed = 0
    if strip_fillers:
        extra_set = {w.strip().lower() for w in (extra_fillers or []) if w.strip()}
        filtered: list[CapCutSubtitleSegment] = []
        for sub in subtitles:
            keep_words: list[str] = []
            keep_starts: list[int] = []
            keep_ends: list[int] = []
            prev_clean: Optional[str] = None
            for i, w in enumerate(sub.words_text):
                if is_filler(w, extra=extra_set):
                    fillers_removed += 1
                    prev_clean = None
                    continue
                clean = w.strip().lower().strip(".,!?…—–-")
                if clean and clean == prev_clean:
                    fillers_removed += 1
                    continue
                prev_clean = clean
                keep_words.append(w)
                keep_starts.append(sub.words_start_ms[i])
                keep_ends.append(sub.words_end_ms[i])
            if keep_words:
                filtered.append(CapCutSubtitleSegment(
                    segment_id=sub.segment_id,
                    material_id=sub.material_id,
                    text=" ".join(keep_words),
                    words_text=keep_words,
                    words_start_ms=keep_starts,
                    words_end_ms=keep_ends,
                    timeline_start_us=sub.timeline_start_us,
                    timeline_duration_us=sub.timeline_duration_us,
                    recognize_task_id=sub.recognize_task_id,
                ))
        subtitles = filtered

    chunks = build_short_caption_chunks(
        subtitles,
        min_words=eff_min_words,
        max_words=eff_max_words,
        max_chars=eff_max_chars,
        max_duration_sec=eff_max_dur,
        prefer_sentences=prefer_sentences,
    )
    if not chunks:
        return {
            "error": "Subtitles found, but none had word-level timing to chunk",
            "suggestion": (
                "Regenerate auto-captions in CapCut — older projects may "
                "lack per-word timing data."
            ),
            "project_path": str(path),
            "project_name": project.project_name,
        }

    base_style = preset.style if preset else TextStyle(
        font_size=15, bold=True, position_y=0.5,
        background_color=None, background_alpha=0.0,
    )
    style = TextStyle(
        font_size=font_size if font_size is not None else base_style.font_size,
        font_color=base_style.font_color,
        background_color=base_style.background_color,
        background_alpha=base_style.background_alpha,
        position_y=position_y if position_y is not None else base_style.position_y,
        bold=bold if bold is not None else base_style.bold,
        font_path=base_style.font_path,
    )

    project.add_text_track(chunks, style=style)
    project.save()

    return {
        "project_path": str(path),
        "project_name": project.project_name,
        "stats": {
            "subtitles_analyzed": len(subtitles),
            "chunks_generated": len(chunks),
            "min_words": eff_min_words,
            "max_words": eff_max_words,
            "max_chars": eff_max_chars,
            "max_duration_sec": eff_max_dur,
            "prefer_sentences": prefer_sentences,
            "caption_preset": preset.name if preset else None,
            "fillers_removed": fillers_removed,
        },
        "sample": [
            {"text": c["text"], "start_sec": round(c["start"], 2), "end_sec": round(c["end"], 2)}
            for c in chunks[:8]
        ],
        "message": (
            f"Added {len(chunks)} short captions to '{project.project_name}' "
            f"({eff_min_words}-{eff_max_words} words"
            + (f", preset='{preset.name}'" if preset else "")
            + (f", stripped {fillers_removed} fillers" if fillers_removed else "")
            + "). Original subtitles preserved."
        ),
    }


# ---------------------------------------------------------------------------
# Tool: list_caption_presets
# ---------------------------------------------------------------------------

async def list_caption_presets() -> dict:
    """List available caption style presets with their parameters."""
    return {
        "presets": list_presets(),
        "message": (
            "Pass any 'name' as `caption_preset` to transcribe_project or "
            "generate_short_captions. Explicit args (min_words, max_words, "
            "max_chars, font_size, etc.) override preset defaults."
        ),
    }


# ---------------------------------------------------------------------------
# Tool: manage_transcript_cache
# ---------------------------------------------------------------------------

async def manage_transcript_cache(action: str = "stats") -> dict:
    """Inspect or clear the on-disk transcription cache.

    Actions:
      * ``"stats"`` (default) — return entry count + total bytes
      * ``"clear"`` — delete every cached transcript JSON
    """
    from smartcut.core import transcript_cache

    action = (action or "stats").lower()
    if action == "stats":
        info = transcript_cache.stats()
        info["message"] = (
            f"Transcript cache holds {info['entries']} entries "
            f"({info['bytes'] / 1024:.1f} KB) at {info['path']}."
        )
        return info
    if action == "clear":
        deleted = transcript_cache.clear()
        info = transcript_cache.stats()
        info["deleted"] = deleted
        info["message"] = f"Cleared {deleted} cached transcript(s)."
        return info
    return {"error": f"Unknown action '{action}'. Use 'stats' or 'clear'."}


# ---------------------------------------------------------------------------
# Tool: normalize_project_text
# ---------------------------------------------------------------------------

async def normalize_project_text(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
) -> dict:
    """
    Collapse double-spacing in every text material of an existing CapCut project.

    Walks all text materials (auto-captions and custom text alike), strips
    leading/trailing whitespace from each word token, and collapses runs of
    spaces in the display text to a single space. Saves only if anything
    actually changed.
    """
    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path

    project = CapCutProject.load(path)
    changed = project.normalize_text_whitespace()

    if changed:
        project.save()
        message = f"Normalized {changed} text material(s) in '{project.project_name}'."
    else:
        message = f"No double-spacing found in '{project.project_name}'."

    return {
        "project_path": str(path),
        "project_name": project.project_name,
        "materials_changed": changed,
        "saved": changed > 0,
        "message": message,
    }


_SENTENCE_END = (".", "!", "?", "…")


def build_short_caption_chunks(
    subtitles: list[CapCutSubtitleSegment],
    min_words: int = 2,
    max_words: int = 4,
    max_chars: Optional[int] = None,
    max_duration_sec: Optional[float] = None,
    prefer_sentences: bool = False,
) -> list[dict]:
    """Split each subtitle's word-level timing into readable chunks.

    Break order (first hit wins):

    1. ``prefer_sentences`` is True and we just consumed a token that ends
       a sentence (``.``/``!``/``?``/``…``) — break.
    2. The chunk reached ``max_words``.
    3. The chunk is at least ``min_words`` long *and* the token ends with
       any punctuation (rhythmic break).
    4. ``max_chars`` would be exceeded by the next token.
    5. ``max_duration_sec`` would be exceeded by the next token.

    A trailing chunk shorter than ``min_words`` is merged into the previous
    chunk so no orphan single-word cards appear — unless that would push
    the merged chunk past ``max_chars`` / ``max_duration_sec``, in which
    case the orphan is kept (better orphan than illegible).
    """
    chunks: list[dict] = []

    def _text_chars(tokens: list[str]) -> int:
        return len(_join_caption_words(tokens))

    for sub in subtitles:
        words = sub.words_text
        starts_ms = sub.words_start_ms
        ends_ms = sub.words_end_ms

        if not words or len(words) != len(starts_ms) or len(words) != len(ends_ms):
            continue

        base_us = sub.timeline_start_us
        sub_chunks: list[dict] = []
        buf_words: list[str] = []
        buf_start_ms: int = 0
        buf_end_ms: int = 0

        def _flush() -> None:
            nonlocal buf_words
            if not buf_words:
                return
            sub_chunks.append({
                "start": (base_us + buf_start_ms * 1000) / MICROSECONDS_PER_SECOND,
                "end": (base_us + buf_end_ms * 1000) / MICROSECONDS_PER_SECOND,
                "text": _join_caption_words(buf_words),
            })
            buf_words = []

        for idx, word in enumerate(words):
            if not buf_words:
                buf_start_ms = starts_ms[idx]
            buf_words.append(word)
            buf_end_ms = ends_ms[idx]

            stripped = word.rstrip()
            ends_with_sentence = stripped.endswith(_SENTENCE_END)
            ends_with_punct = stripped.endswith(_PUNCT_BREAK)
            count = len(buf_words)

            chunk_chars = _text_chars(buf_words)
            chunk_dur = (buf_end_ms - buf_start_ms) / 1000.0

            should_break = False
            if prefer_sentences and count >= min_words and ends_with_sentence:
                should_break = True
            elif count >= max_words:
                should_break = True
            elif count >= min_words and ends_with_punct:
                should_break = True
            elif max_chars is not None and chunk_chars >= max_chars and count >= min_words:
                should_break = True
            elif max_duration_sec is not None and chunk_dur >= max_duration_sec and count >= min_words:
                should_break = True

            if should_break:
                _flush()

        if buf_words:
            tail_text = _join_caption_words(buf_words)
            tail = {
                "start": (base_us + buf_start_ms * 1000) / MICROSECONDS_PER_SECOND,
                "end": (base_us + buf_end_ms * 1000) / MICROSECONDS_PER_SECOND,
                "text": tail_text,
            }
            can_merge = len(buf_words) < min_words and sub_chunks
            if can_merge:
                prev = sub_chunks[-1]
                merged_text = _join_caption_words([prev["text"], tail_text])
                merged_dur = tail["end"] - prev["start"]
                over_chars = max_chars is not None and len(merged_text) > max_chars
                over_dur = max_duration_sec is not None and merged_dur > max_duration_sec
                if over_chars or over_dur:
                    sub_chunks.append(tail)
                else:
                    prev["text"] = merged_text
                    prev["end"] = tail["end"]
            else:
                sub_chunks.append(tail)

        chunks.extend(sub_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Heuristic analysis engine
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    Compute similarity between two texts.

    Uses max of:
    - Jaccard word overlap (catches reordered duplicates)
    - SequenceMatcher ratio (catches sequential similarity)
    """
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)

    words_a = set(norm_a.split())
    words_b = set(norm_b.split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union)

    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()

    return max(jaccard, seq_ratio)


def find_gaps(
    subtitles: list[CapCutSubtitleSegment],
    threshold_us: int,
    project_duration_us: int = 0,
) -> list[tuple[int, int]]:
    """
    Find silence gaps that exceed the threshold.

    Checks:
    - Gap from project start (0) to first subtitle
    - Gaps between consecutive subtitles
    - Gap from last subtitle to project end
    """
    gaps = []

    if not subtitles:
        return gaps

    # Gap at the beginning (before first subtitle) — always cut, it's dead air
    first_start = subtitles[0].timeline_start_us
    if first_start > 0:
        gaps.append((0, first_start))

    # Gaps between consecutive subtitles
    for i in range(len(subtitles) - 1):
        current_end = subtitles[i].timeline_end_us
        next_start = subtitles[i + 1].timeline_start_us
        gap = next_start - current_end
        if gap > threshold_us:
            gaps.append((current_end, next_start))

    # Gap at the end (after last subtitle) — always cut, it's dead air
    if project_duration_us > 0:
        last_end = subtitles[-1].timeline_end_us
        if project_duration_us > last_end:
            gaps.append((last_end, project_duration_us))

    return gaps


def find_duplicate_takes(
    subtitles: list[CapCutSubtitleSegment],
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
) -> list[tuple[int, int]]:
    """
    Find duplicate takes by detecting "restart points".

    A person records in takes: they say a sequence of phrases, then start over.
    Example:
        "Hello friends today I'll..."   <- Take 1 (abandoned)
        "Hello friends, today I'll show you..."  <- Take 2 (abandoned)
        "Hello friends, today I'll show you how..."  <- Take 3 (KEEP)

    The takes are NOT consecutive — each take is a GROUP of subtitles.
    We detect restarts by finding subtitle[i] that matches a later subtitle[j],
    meaning the speaker went back to re-record from that point.

    Cuts the ENTIRE span from first removed subtitle to the start of the kept
    version — including all gaps between subtitles within the removed takes.

    Returns time ranges of earlier takes to cut.
    """
    if len(subtitles) < 2:
        return []

    ranges_to_cut = []
    i = 0

    while i < len(subtitles):
        # Look for the LATEST restart of this subtitle's phrase
        last_restart = None
        for j in range(i + 1, len(subtitles)):
            if compute_text_similarity(subtitles[i].text, subtitles[j].text) >= similarity_threshold:
                last_restart = j

        if last_restart is not None:
            # Cut ONE continuous range: from start of first removed
            # to start of the kept version (includes all gaps between removed subs)
            cut_start = subtitles[i].timeline_start_us
            cut_end = subtitles[last_restart].timeline_start_us
            ranges_to_cut.append((cut_start, cut_end))
            i = last_restart
        else:
            i += 1

    return ranges_to_cut


def merge_time_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent time ranges. Returns sorted, non-overlapping list."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


# ---------------------------------------------------------------------------
# Optional OpenAI-enhanced duplicate detection
# ---------------------------------------------------------------------------

def _detect_duplicates_with_llm(
    subtitles: list[CapCutSubtitleSegment],
    api_key: str,
) -> list[tuple[int, int]]:
    """Use OpenAI GPT to detect duplicate takes more accurately."""
    from smartcut.core.llm_client import LLMClient

    paragraphs = [
        {"id": i, "text": s.text}
        for i, s in enumerate(subtitles)
    ]

    client = LLMClient(api_key=api_key)
    result = client.detect_duplicates(paragraphs)

    ranges_to_cut = []
    for group in result.groups:
        for idx in group.remove:
            if 0 <= idx < len(subtitles):
                seg = subtitles[idx]
                ranges_to_cut.append((seg.timeline_start_us, seg.timeline_end_us))

    return ranges_to_cut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_project_path(
    project_path: Optional[str],
    project_name: Optional[str],
) -> Path | dict:
    """Resolve project path from either path or name. Returns Path or error dict."""
    if project_path:
        path = Path(project_path)
    elif project_name:
        path = find_project_by_name(project_name)
        if path is None:
            return {
                "error": f"Project '{project_name}' not found",
                "suggestion": "Use list_capcut_projects to see available projects",
            }
    else:
        return {"error": "Either project_path or project_name must be provided"}

    if not path.exists():
        return {"error": f"Project path not found: {path}"}

    content_file = path / "draft_info.json"
    if not content_file.exists():
        return {
            "error": "Project missing draft_info.json",
            "path": str(path),
            "suggestion": "Open it in CapCut first to regenerate the content file.",
        }

    return path


def _format_duration_us(duration_us: int) -> str:
    """Format microseconds as M:SS."""
    total_sec = duration_us / MICROSECONDS_PER_SECOND
    minutes = int(total_sec // 60)
    seconds = int(total_sec % 60)
    return f"{minutes}:{seconds:02d}"
