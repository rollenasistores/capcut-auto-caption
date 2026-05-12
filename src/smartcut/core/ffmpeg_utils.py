"""Minimal FFmpeg helpers for audio extraction."""

import shutil
import subprocess
from pathlib import Path


class FFmpegError(Exception):
    pass


def check_ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16kHz WAV audio from a video file."""
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Audio extraction failed: {e.stderr.decode(errors='replace')}")
    return output_path


def probe_duration_sec(media_path: Path) -> float:
    """Return media duration in seconds via ffprobe. Returns 0.0 on failure."""
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
            capture_output=True, check=True, text=True,
        )
        return float(out.stdout.strip() or 0.0)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
