# SmartCut MCP Server — Project Context

## What is this?

MCP server that adds AI-powered transcription + captioning + smart cutting to CapCut. Works with Claude Code to:
- Auto-transcribe source videos with **Whisper large-v3 tuned for Tagalog / Tag-Lish**
- Cache transcriptions per audio hash + params (instant re-runs)
- Support `dry_run=True` previews before writing to the project
- Generate short-form captions with style presets (`tiktok`, `karaoke`, `news`, …)
- Strip Tagalog/English filler words and stutter dedup
- Find silences & duplicate takes from the subtitles and cut them
- Edit CapCut JSON in place (no backups, no re-encoding)

## Project Structure

```
src/smartcut/
├── __init__.py              # Version
├── config.py                # Settings, env vars, constants
├── server.py                # MCP server entry point, 8 tools
├── prefetch.py              # CLI: pre-download Whisper models with progress
├── core/
│   ├── models.py            # Pydantic models
│   ├── whisper_client.py    # OpenAI Whisper API wrapper (lazy import)
│   ├── whisper_local.py     # faster-whisper backend with Tagalog defaults
│   ├── model_download.py    # Visible HF model download (tqdm to stderr)
│   ├── transcript_cache.py  # Content-addressed Whisper result cache
│   ├── caption_style.py     # Style presets + Tagalog filler stripping
│   ├── llm_client.py        # GPT for duplicate detection (optional)
│   ├── ffmpeg_utils.py      # Audio extract + ASR-tuned preprocessing
│   ├── capcut_reader.py     # CapCut JSON read/write + text normalization
│   └── capcut_finder.py     # CapCut project discovery
└── tools/
    └── capcut_projects.py   # All 8 MCP tool implementations
```

## MCP Tools (8 total)

| Tool | Purpose |
|------|---------|
| `list_capcut_projects` | Enumerate drafts |
| `open_capcut_project` | Load and return structure |
| `transcribe_project` | Whisper → CapCut auto-subtitle track + short captions |
| `generate_short_captions` | Re-chunk existing subtitles via style preset |
| `list_caption_presets` | Show available presets and their parameters |
| `normalize_project_text` | Retroactively fix double-spacing in text materials |
| `smart_cut_project` | Remove silences + duplicate takes |
| `manage_transcript_cache` | `stats` / `clear` the on-disk Whisper cache |

## Key Modules

### `core/whisper_local.py`
- `LocalWhisperClient` — faster-whisper wrapper
- `TAGALOG_PRIMER` — built-in Filipino primer auto-applied when `language in {'tl','fil','tgl'}`
- Defaults tuned to **beat CapCut's built-in auto-caption**: `multilingual=True`, `hallucination_silence_threshold=2.0`, `condition_on_previous_text=False`, custom VAD params.
- Accepts: `initial_prompt`, `hotwords`, `min_word_probability`, `beam_size`, `compute_type`, `condition_on_previous_text`.

### `core/transcript_cache.py`
- Content-addressed JSON cache at `~/.cache/smartcut/transcripts/` (or `SMARTCUT_CACHE_DIR`)
- Key = blake2b(audio_content + canonical_json(params))
- Atomic write (temp + rename); version-stamped blob
- `compute_cache_key()`, `load()`, `save()`, `stats()`, `clear()`

### `core/caption_style.py`
- `CaptionPreset` dataclass; `PRESETS` registry
- Presets: `tiktok`, `tiktok-yellow`, `karaoke`, `minimal`, `news`, `podcast`
- `strip_fillers_from_words()` — Tagalog/English filler + stutter dedup
- `DEFAULT_FILLERS` covers "uhm", "ah", "eh", "kasi nga", "di ba", "ano ba", …

### `core/ffmpeg_utils.py`
- `extract_audio()` — basic 16 kHz mono WAV
- `extract_audio_for_asr()` — applies `DEFAULT_SPEECH_FILTER`: high-pass 80, low-pass 8 kHz, dynaudnorm, EBU R128 loudnorm (-16 LUFS) in one ffmpeg pass

### `core/model_download.py`
- `ensure_model_downloaded()` — checks HF cache, prints status banner, drives `snapshot_download` with explicit `tqdm` to stderr (safe for stdio JSON-RPC)
- `is_model_cached()` — `try_to_load_from_cache(repo_id, "model.bin")`
- Used by both `LocalWhisperClient.__init__` and `smartcut.prefetch`

### `core/capcut_reader.py`
- `CapCutProject` class — `load()`, `save()`, `get_subtitle_segments()`, `get_video_segments()`, `remove_time_ranges()`, `add_auto_subtitle_track()`, `add_text_track()`
- `normalize_caption_text()` — single chokepoint that runs inside `_build_text_material()` so every new write is whitespace-clean
- `normalize_text_whitespace()` — retroactive fixer for existing materials

### `tools/capcut_projects.py`
- `transcribe_project()` — orchestrates extract → cache lookup → Whisper → filler strip → per-segment timeline mapping → write or dry-run preview
- `build_short_caption_chunks()` — honours `min_words`, `max_words`, `max_chars`, `max_duration_sec`, `prefer_sentences`
- `_sentences_to_subtitle_view()` — adapter so chunker works on in-memory dry-run sentences

## CapCut Format Notes

- Times are in **microseconds** (1 sec = 1,000,000 μs)
- Video segments have `source_timerange` (where in source) and `target_timerange` (where on timeline)
- Text segments have `target_timerange` only (`source_timerange` is null)
- Auto-generated subtitles: `materials.texts[]` with `recognize_task_id != ""`
- Subtitle word timing: `words.start_time[]` / `words.end_time[]` in **milliseconds**, relative to segment start
- Display text is in `content` JSON field (not top-level `text`)
- CapCut monitors drafts folder via FSEvents and may rename/move folders
- **One source can appear as multiple segments** (splits, repeats) — `transcribe_project` iterates per-segment, transcribes each unique source ONCE, then remaps words per segment

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BACKEND` | `local` | `local` or `openai` |
| `WHISPER_LOCAL_MODEL` | `large-v3` | Any faster-whisper size |
| `WHISPER_DEVICE` | `cpu` | `cpu` / `cuda` |
| `WHISPER_COMPUTE_TYPE` | auto | `int8` / `int8_float16` / `float16` / `float32` |
| `WHISPER_LANGUAGE` | — | Default ISO code (e.g. `tl` for Tagalog primer) |
| `WHISPER_INITIAL_PROMPT` | — | Project-wide decoder context |
| `WHISPER_HOTWORDS` | — | Project-wide vocabulary biasing |
| `WHISPER_MIN_WORD_PROBABILITY` | `0.0` | Confidence floor |
| `SMARTCUT_CACHE_DIR` | `~/.cache/smartcut/transcripts` | Whisper cache location |
| `OPENAI_API_KEY` | — | Only for `backend=openai` or GPT duplicate detection |
| `CAPCUT_DRAFTS_DIR` | auto | Override CapCut drafts folder |

## Running

```bash
cd capcut-ai-editor
python -m venv venv
source venv/bin/activate
pip install -e '.[local]'
python -m smartcut.prefetch large-v3      # one-time, ~3 GB
python -m smartcut.server                 # via MCP config in Claude
```

## Common Tasks

### Add new tool
1. Create function in `tools/capcut_projects.py`
2. Import it in `server.py`
3. Add Tool schema in `list_tools()`
4. Add dispatch in `call_tool()`

### Add a new caption preset
1. Add `CaptionPreset(...)` entry to `PRESETS` dict in `core/caption_style.py`
2. Mention it in the README preset table

### Bump cache format
- Bump `CACHE_VERSION` in `core/transcript_cache.py` — older entries become unreadable and are ignored (not crashes).

### Debug CapCut issues
- Check `.recycle_bin/` folder in drafts dir
- Verify `draft_info.json` exists
- CapCut may need restart to see changes
- Use `normalize_project_text` if older projects have double-spacing
