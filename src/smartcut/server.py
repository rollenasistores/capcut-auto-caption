"""SmartCut MCP Server — simplified entry point."""

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from smartcut.tools.capcut_projects import (
    generate_short_captions,
    list_capcut_projects,
    normalize_project_text,
    open_capcut_project,
    smart_cut_project,
    transcribe_project,
)

server = Server("smartcut")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_capcut_projects",
            description="List all CapCut projects in the drafts directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drafts_dir": {
                        "type": "string",
                        "description": "Custom path to CapCut drafts directory (auto-detected if not set)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="open_capcut_project",
            description=(
                "Open an existing CapCut project and return its structure. "
                "Shows video segments, text tracks, and auto-generated subtitles."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="smart_cut_project",
            description=(
                "Smart cut a CapCut project: remove silences and duplicate takes. "
                "Reads CapCut's auto-generated subtitles to find gaps and duplicates. "
                "User must generate subtitles in CapCut first (Text → Auto Captions). "
                "Modifies the project IN PLACE (no backup). "
                "By default uses heuristic analysis (free, no API keys). "
                "Set use_openai=true for GPT-enhanced duplicate detection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                    "silence_threshold_sec": {
                        "type": "number",
                        "description": "Minimum gap between subtitles to cut (default 1.0 sec)",
                        "default": 1.0,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Text similarity threshold for duplicate detection (0.0-1.0, default 0.6)",
                        "default": 0.6,
                    },
                    "use_openai": {
                        "type": "boolean",
                        "description": "Use OpenAI GPT for enhanced duplicate detection (requires OPENAI_API_KEY)",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="transcribe_project",
            description=(
                "Auto-generate subtitles for a CapCut project using Whisper. "
                "Tuned to BEAT CapCut's built-in auto-caption — especially for Filipino / "
                "Tagalog content. Defaults: anti-hallucination guards, multilingual "
                "decoding for Tag-Lish code-switching, and a Filipino-flavored initial "
                "prompt when language='tl'. Default backend is LOCAL (faster-whisper, no "
                "API key). For best Tagalog accuracy: pass language='tl', plus optionally "
                "hotwords with names/jargon from your video. On a CUDA GPU, also set "
                "device='cuda' and compute_type='float16' (or 'float32' for max quality). "
                "Also adds a short-caption track of 2-4 words per chunk by default; "
                "min_words/max_words are freely commandable. Requires ffmpeg."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                    "language": {
                        "type": "string",
                        "description": (
                            "ISO language code. Use 'tl' for Tagalog (also accepts 'fil'). "
                            "Setting this is the single biggest accuracy win — it stops "
                            "Whisper from guessing and auto-activates the Filipino primer. "
                            "Auto-detect if omitted."
                        ),
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["local", "openai"],
                        "description": "Transcription backend (default 'local' — self-hosted faster-whisper)",
                    },
                    "model_size": {
                        "type": "string",
                        "description": (
                            "Local model: tiny, base, small, medium, large-v3 (default), "
                            "large-v3-turbo, distil-large-v3. For Tagalog accuracy stick "
                            "with large-v3."
                        ),
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda"],
                        "description": "Compute device for local backend (default 'cpu')",
                    },
                    "compute_type": {
                        "type": "string",
                        "enum": ["int8", "int8_float16", "float16", "float32"],
                        "description": (
                            "Numerical precision. CPU default 'int8' (fast, slight accuracy "
                            "loss). On CUDA, use 'float16' for default quality or 'float32' "
                            "for maximum accuracy."
                        ),
                    },
                    "initial_prompt": {
                        "type": "string",
                        "description": (
                            "Free-text context shown to the decoder to bias vocabulary and "
                            "register. Use to introduce proper nouns (names of people, "
                            "brands, products). When language='tl' and this is omitted, a "
                            "built-in Filipino primer is applied automatically."
                        ),
                    },
                    "hotwords": {
                        "type": "string",
                        "description": (
                            "Short comma-separated list of must-recognize words (names, "
                            "brands, technical terms). Gets extra weight during beam "
                            "search — different from initial_prompt. Local backend only "
                            "(merged into prompt for OpenAI backend). "
                            "Example: 'Manila, Cebu, kasi, talaga, brand names'."
                        ),
                    },
                    "min_word_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Drop transcribed words below this confidence (0.0 disables). "
                            "Try 0.3-0.5 to strip low-confidence hallucinations like "
                            "'[Music]' or background-noise misfires."
                        ),
                    },
                    "beam_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "description": (
                            "Beam search width. Higher = more accurate but slower. "
                            "Default 5 is the standard. Try 8-10 for max accuracy."
                        ),
                    },
                    "condition_on_previous_text": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, Whisper carries the previous transcript into the "
                            "next chunk's context. Default false — for talking-head "
                            "content with pauses, true causes hallucination loops."
                        ),
                    },
                    "also_short_captions": {
                        "type": "boolean",
                        "description": "Also add short captions on a separate track (default true).",
                        "default": True,
                    },
                    "min_words": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 2,
                        "description": (
                            "Minimum words per short-caption chunk. Set equal to "
                            "max_words for fixed-size chunks (e.g. 1,1 for one-word cards)."
                        ),
                    },
                    "max_words": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 4,
                        "description": (
                            "Maximum words per short-caption chunk; must be >= min_words."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="normalize_project_text",
            description=(
                "Collapse double-spacing in every text material of an existing CapCut project. "
                "Use this on older projects whose captions were written before whitespace "
                "normalization. Saves only if anything actually changed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="generate_short_captions",
            description=(
                "Generate short-form captions on a new text track from CapCut's auto-subtitles. "
                "Chunk size is fully commandable by the AI via min_words/max_words — e.g. "
                "(1,1) for one-word-per-card karaoke, (2,4) default TikTok style, (5,7) "
                "longer caption blocks. Chunker prefers punctuation breaks once min_words is "
                "reached and force-breaks at max_words. Originals are preserved; the new "
                "track is added on top so the user can toggle/delete. Run auto-captions in "
                "CapCut first (Text → Auto Captions)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                    "min_words": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 2,
                        "description": (
                            "Minimum words per chunk. Set equal to max_words for fixed-size "
                            "chunks. Examples: 1 (one-word cards), 2 (default), 5 (long)."
                        ),
                    },
                    "max_words": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 4,
                        "description": (
                            "Maximum words per chunk; must be >= min_words. The chunker "
                            "force-breaks here even if no punctuation is hit."
                        ),
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "Caption font size (default 15)",
                        "default": 15,
                    },
                    "bold": {
                        "type": "boolean",
                        "description": "Bold text (default true)",
                        "default": True,
                    },
                    "position_y": {
                        "type": "number",
                        "description": "Vertical position 0.0 (top) to 1.0 (bottom), default 0.5 (center)",
                        "default": 0.5,
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "list_capcut_projects":
            result = await list_capcut_projects(**arguments)
        elif name == "open_capcut_project":
            result = await open_capcut_project(**arguments)
        elif name == "smart_cut_project":
            result = await smart_cut_project(**arguments)
        elif name == "generate_short_captions":
            result = await generate_short_captions(**arguments)
        elif name == "normalize_project_text":
            result = await normalize_project_text(**arguments)
        elif name == "transcribe_project":
            result = await transcribe_project(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
