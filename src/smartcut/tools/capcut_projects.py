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
from smartcut.core.capcut_reader import CapCutProject, TextStyle
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
    also_short_captions: bool = True,
    min_words: int = 2,
    max_words: int = 4,
) -> dict:
    """
    Generate auto-subtitles for a CapCut project using Whisper.

    Backends:
      - "local"  (default): runs faster-whisper on this machine (no API key,
                 no network — model auto-downloads on first use)
      - "openai": calls OpenAI Whisper API (needs OPENAI_API_KEY, 25MB limit)

    Pipeline: locate source video(s) → ffmpeg extract audio → transcribe with
    word-level timing → inject as CapCut-format auto-subtitle track (so the
    other tools can read it natively) → optional short-caption track.
    """
    from smartcut.core.ffmpeg_utils import (
        FFmpegError,
        check_ffmpeg_installed,
        extract_audio,
    )

    settings = get_settings()
    resolved_backend = (backend or settings.whisper_backend or "local").lower()
    if resolved_backend not in ("local", "openai"):
        return {"error": f"Unknown backend '{resolved_backend}' (use 'local' or 'openai')"}

    if not check_ffmpeg_installed():
        return {"error": "ffmpeg not found on PATH — install ffmpeg first"}

    if resolved_backend == "openai":
        if not settings.openai_api_key:
            return {"error": "backend='openai' requires OPENAI_API_KEY"}
        from smartcut.core.whisper_client import WhisperClient
        client = WhisperClient(api_key=settings.openai_api_key)
    else:
        try:
            from smartcut.core.whisper_local import LocalWhisperClient
            client = LocalWhisperClient(
                model_size=model_size or settings.whisper_local_model,
                device=device or settings.whisper_device,
            )
        except RuntimeError as e:
            return {"error": str(e)}

    path = _resolve_project_path(project_path, project_name)
    if isinstance(path, dict):
        return path

    project = CapCutProject.load(path)
    video_paths = project.get_source_video_paths()
    if not video_paths:
        return {
            "error": "No source videos found in project materials",
            "project_path": str(path),
            "project_name": project.project_name,
        }

    missing = [str(p) for p in video_paths if not p.exists()]
    if missing:
        return {
            "error": "Source video file(s) not found on disk",
            "missing_paths": missing,
            "suggestion": "Re-link the media in CapCut or restore the files.",
        }

    all_sentences: list[dict] = []
    per_source_stats = []

    with tempfile.TemporaryDirectory(prefix="smartcut_") as tmpdir:
        tmp = Path(tmpdir)
        for vpath in video_paths:
            seg = project.get_video_segment_for_source(vpath)
            if seg is None:
                continue

            target = seg.get("target_timerange", {})
            source = seg.get("source_timerange") or {}
            timeline_start_sec = target.get("start", 0) / MICROSECONDS_PER_SECOND
            source_start_sec = source.get("start", 0) / MICROSECONDS_PER_SECOND
            timeline_dur_sec = target.get("duration", 0) / MICROSECONDS_PER_SECOND

            audio_file = tmp / f"{vpath.stem}.wav"
            try:
                extract_audio(vpath, audio_file)
            except FFmpegError as e:
                return {"error": f"Audio extraction failed for {vpath}: {e}"}

            result = client.transcribe(audio_file, language=language)

            sentences_for_source = 0
            for ws in result.segments:
                # Map source-time → absolute timeline-time, clamping to the segment span.
                start_on_timeline = ws.start - source_start_sec + timeline_start_sec
                end_on_timeline = ws.end - source_start_sec + timeline_start_sec
                if end_on_timeline <= timeline_start_sec:
                    continue
                if start_on_timeline >= timeline_start_sec + timeline_dur_sec:
                    continue
                start_on_timeline = max(start_on_timeline, timeline_start_sec)
                end_on_timeline = min(end_on_timeline, timeline_start_sec + timeline_dur_sec)

                mapped_words = []
                for w in ws.words:
                    ws_start = w.start - source_start_sec + timeline_start_sec
                    ws_end = w.end - source_start_sec + timeline_start_sec
                    ws_start = max(ws_start, start_on_timeline)
                    ws_end = min(ws_end, end_on_timeline)
                    if ws_end <= ws_start:
                        continue
                    mapped_words.append({"word": w.word, "start": ws_start, "end": ws_end})

                if not mapped_words:
                    continue

                all_sentences.append({
                    "start": start_on_timeline,
                    "end": end_on_timeline,
                    "text": ws.text,
                    "words": mapped_words,
                })
                sentences_for_source += 1

            per_source_stats.append({
                "video": str(vpath),
                "language": result.language,
                "sentences": sentences_for_source,
            })

    if not all_sentences:
        return {
            "error": "Whisper returned no usable segments",
            "project_path": str(path),
            "project_name": project.project_name,
        }

    all_sentences.sort(key=lambda s: s["start"])
    project.add_auto_subtitle_track(all_sentences)

    short_caption_chunks = []
    if also_short_captions:
        subtitles = project.get_subtitle_segments()
        short_caption_chunks = build_short_caption_chunks(subtitles, min_words, max_words)
        if short_caption_chunks:
            style = TextStyle(font_size=15, bold=True, position_y=0.5,
                              background_color=None, background_alpha=0.0)
            project.add_text_track(short_caption_chunks, style=style)

    project.save()

    return {
        "project_path": str(path),
        "project_name": project.project_name,
        "stats": {
            "backend": resolved_backend,
            "model": (model_size or settings.whisper_local_model) if resolved_backend == "local" else "whisper-1",
            "videos_transcribed": len(per_source_stats),
            "sentences_added": len(all_sentences),
            "short_captions_added": len(short_caption_chunks),
            "language_hint": language or "auto",
        },
        "per_source": per_source_stats,
        "sample": [
            {"text": s["text"], "start": round(s["start"], 2), "end": round(s["end"], 2)}
            for s in all_sentences[:5]
        ],
        "message": (
            f"Transcribed {len(per_source_stats)} video(s) → {len(all_sentences)} subtitles"
            + (f" + {len(short_caption_chunks)} short captions" if short_caption_chunks else "")
            + f" for '{project.project_name}'."
        ),
    }


# ---------------------------------------------------------------------------
# Tool: generate_short_captions
# ---------------------------------------------------------------------------

_PUNCT_BREAK = (".", ",", "!", "?", ";", ":", "—", "–")


async def generate_short_captions(
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    min_words: int = 2,
    max_words: int = 4,
    font_size: int = 15,
    bold: bool = True,
    position_y: float = 0.5,
) -> dict:
    """
    Generate short-form captions (2-4 words per chunk) on a new text track.

    Reads CapCut's auto-generated subtitles (with word-level timing), splits each
    into 2-4 word chunks preferring punctuation breaks, and adds them as a new
    text track. Original subtitles are preserved.
    """
    if min_words < 1 or max_words < min_words:
        return {"error": f"Invalid range: min_words={min_words}, max_words={max_words}"}

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
                "to generate subtitles first."
            ),
            "project_path": str(path),
            "project_name": project.project_name,
        }

    chunks = build_short_caption_chunks(subtitles, min_words, max_words)
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

    style = TextStyle(
        font_size=font_size,
        bold=bold,
        position_y=position_y,
        background_color=None,
        background_alpha=0.0,
    )
    project.add_text_track(chunks, style=style)
    project.save()

    return {
        "project_path": str(path),
        "project_name": project.project_name,
        "stats": {
            "subtitles_analyzed": len(subtitles),
            "chunks_generated": len(chunks),
            "min_words": min_words,
            "max_words": max_words,
        },
        "sample": [
            {"text": c["text"], "start_sec": round(c["start"], 2), "end_sec": round(c["end"], 2)}
            for c in chunks[:8]
        ],
        "message": (
            f"Added {len(chunks)} short captions to '{project.project_name}' "
            f"({min_words}-{max_words} words each). Original subtitles preserved."
        ),
    }


def build_short_caption_chunks(
    subtitles: list[CapCutSubtitleSegment],
    min_words: int = 2,
    max_words: int = 4,
) -> list[dict]:
    """
    Split each subtitle's word-level timing into 2-4 word chunks.

    Break rules: prefer breaking after a word ending in punctuation once the
    chunk has reached min_words; force-break at max_words. A trailing chunk
    of fewer than min_words is merged into the previous chunk so no orphan
    single-word captions appear.

    Returns dicts with 'start' (sec), 'end' (sec), 'text'.
    """
    chunks: list[dict] = []

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

        for idx, word in enumerate(words):
            if not buf_words:
                buf_start_ms = starts_ms[idx]
            buf_words.append(word)
            buf_end_ms = ends_ms[idx]

            ends_with_punct = word.rstrip().endswith(_PUNCT_BREAK)
            count = len(buf_words)
            should_break = count >= max_words or (count >= min_words and ends_with_punct)

            if should_break:
                sub_chunks.append({
                    "start": (base_us + buf_start_ms * 1000) / MICROSECONDS_PER_SECOND,
                    "end": (base_us + buf_end_ms * 1000) / MICROSECONDS_PER_SECOND,
                    "text": " ".join(buf_words),
                })
                buf_words = []

        if buf_words:
            tail = {
                "start": (base_us + buf_start_ms * 1000) / MICROSECONDS_PER_SECOND,
                "end": (base_us + buf_end_ms * 1000) / MICROSECONDS_PER_SECOND,
                "text": " ".join(buf_words),
            }
            if len(buf_words) < min_words and sub_chunks:
                prev = sub_chunks[-1]
                prev["text"] = prev["text"] + " " + tail["text"]
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
