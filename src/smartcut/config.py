"""Configuration and settings for SmartCut MCP Server."""

import platform
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    capcut_drafts_dir: Optional[str] = Field(default=None, alias="CAPCUT_DRAFTS_DIR")
    whisper_backend: str = Field(default="local", alias="WHISPER_BACKEND")
    whisper_local_model: str = Field(default="large-v3", alias="WHISPER_LOCAL_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: Optional[str] = Field(default=None, alias="WHISPER_COMPUTE_TYPE")
    whisper_language: Optional[str] = Field(default=None, alias="WHISPER_LANGUAGE")
    whisper_initial_prompt: Optional[str] = Field(default=None, alias="WHISPER_INITIAL_PROMPT")
    whisper_hotwords: Optional[str] = Field(default=None, alias="WHISPER_HOTWORDS")
    whisper_min_word_probability: float = Field(default=0.0, alias="WHISPER_MIN_WORD_PROBABILITY")

    model_config = {"env_file": ".env", "extra": "ignore"}

    def get_capcut_drafts_path(self) -> Path:
        """Get CapCut drafts directory path, auto-detecting if not set."""
        if self.capcut_drafts_dir:
            return Path(self.capcut_drafts_dir)

        system = platform.system()
        if system == "Darwin":  # macOS
            return Path.home() / "Movies" / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
        elif system == "Windows":
            local_app_data = Path.home() / "AppData" / "Local"
            return local_app_data / "CapCut" / "User Data" / "Projects" / "com.lveditor.draft"
        else:
            return Path.cwd() / "capcut_drafts"


SILENCE_THRESHOLD_SEC = 1.0
MIN_SEGMENT_DURATION_SEC = 0.5
DUPLICATE_SIMILARITY_THRESHOLD = 0.6
WHISPER_MODEL = "whisper-1"
LLM_MODEL = "gpt-4.1-mini"
MICROSECONDS_PER_SECOND = 1_000_000


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
