"""OpenAI Whisper API client — word-level transcription."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from smartcut.config import WHISPER_MODEL

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2


@dataclass
class WhisperWord:
    word: str
    start: float
    end: float


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    words: list[WhisperWord] = field(default_factory=list)


@dataclass
class WhisperResult:
    language: str
    duration: float
    segments: list[WhisperSegment] = field(default_factory=list)


class WhisperClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = WHISPER_MODEL

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> WhisperResult:
        if audio_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise RuntimeError(
                f"Audio file is {audio_path.stat().st_size / 1024 / 1024:.1f}MB, "
                f"exceeds Whisper API limit of {MAX_FILE_SIZE_MB}MB. "
                "Use a shorter video or pre-split the audio."
            )

        for attempt in range(MAX_RETRIES):
            try:
                with open(audio_path, "rb") as f:
                    kwargs = {
                        "model": self.model,
                        "file": f,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word", "segment"],
                    }
                    if language:
                        kwargs["language"] = language
                    response = self.client.audio.transcriptions.create(**kwargs)
                return self._parse(response)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
                else:
                    raise RuntimeError(f"Whisper API failed after {MAX_RETRIES} attempts: {e}")

        raise RuntimeError("Unexpected error in transcription")

    def _parse(self, response) -> WhisperResult:
        all_words = getattr(response, "words", []) or []
        segments: list[WhisperSegment] = []

        for seg in response.segments or []:
            seg_words: list[WhisperWord] = []
            for w in all_words:
                ws = getattr(w, "start", None)
                we = getattr(w, "end", None)
                if ws is None or we is None:
                    continue
                if seg.start <= ws < seg.end:
                    seg_words.append(WhisperWord(
                        word=getattr(w, "word", "").strip(),
                        start=ws,
                        end=we,
                    ))
            segments.append(WhisperSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=seg_words,
            ))

        return WhisperResult(
            language=getattr(response, "language", "unknown"),
            duration=segments[-1].end if segments else 0.0,
            segments=segments,
        )
