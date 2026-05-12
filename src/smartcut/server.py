"""SmartCut MCP Server — simplified entry point."""

import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from smartcut.tools.capcut_projects import (
    generate_short_captions,
    list_capcut_projects,
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
                "Default backend is LOCAL (faster-whisper, no API key, model auto-downloads). "
                "Set backend='openai' to use the OpenAI Whisper API instead. "
                "Reads source video, extracts audio via ffmpeg, transcribes with word-level "
                "timing, and injects results as a CapCut-format auto-subtitle track so "
                "smart_cut_project and generate_short_captions can read them. "
                "By default also adds a 2-4 word short-caption track. "
                "Requires ffmpeg installed locally."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                    "language": {
                        "type": "string",
                        "description": "ISO language code (e.g. 'tl', 'en'). Auto-detect if omitted.",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["local", "openai"],
                        "description": "Transcription backend (default 'local' — self-hosted faster-whisper)",
                    },
                    "model_size": {
                        "type": "string",
                        "description": "Local model size: tiny, base, small, medium, large-v3 (default 'large-v3')",
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda"],
                        "description": "Compute device for local backend (default 'cpu')",
                    },
                    "also_short_captions": {
                        "type": "boolean",
                        "description": "Also add 2-4 word short captions on a separate track (default true)",
                        "default": True,
                    },
                    "min_words": {"type": "integer", "default": 2},
                    "max_words": {"type": "integer", "default": 4},
                },
                "required": [],
            },
        ),
        Tool(
            name="generate_short_captions",
            description=(
                "Generate short-form captions (2-4 words per chunk) on a new text track. "
                "Uses CapCut's auto-subtitle word-level timing and breaks on punctuation. "
                "Originals are kept; the new track is added on top so the user can toggle/delete it. "
                "Run auto-captions in CapCut first (Text → Auto Captions)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Full path to project folder"},
                    "project_name": {"type": "string", "description": "Project name (partial match)"},
                    "min_words": {
                        "type": "integer",
                        "description": "Minimum words per chunk (default 2)",
                        "default": 2,
                    },
                    "max_words": {
                        "type": "integer",
                        "description": "Maximum words per chunk (default 4)",
                        "default": 4,
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
