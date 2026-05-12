"""OpenAI Whisper API client — word-level transcription."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    probability: float = 1.0


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    words: list[WhisperWord] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class WhisperResult:
    language: str
    duration: float
    segments: list[WhisperSegment] = field(default_factory=list)
    language_probability: float = 1.0


class WhisperClient:
    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed. Install with:\n"
                "    pip install openai\n"
                "or:  pip install -e '.[openai]'"
            ) from e
        self.client = OpenAI(api_key=api_key)
        self.model = WHISPER_MODEL

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        min_word_probability: float = 0.0,
        **_ignored,
    ) -> WhisperResult:
        """Transcribe via OpenAI Whisper API.

        The OpenAI endpoint accepts a smaller knob set than faster-whisper:
        only ``language`` and ``prompt`` (we merge ``initial_prompt`` and
        ``hotwords`` into the prompt). Extra kwargs are accepted but
        ignored so the caller can pass the same dict to either backend.
        """
        if audio_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise RuntimeError(
                f"Audio file is {audio_path.stat().st_size / 1024 / 1024:.1f}MB, "
                f"exceeds Whisper API limit of {MAX_FILE_SIZE_MB}MB. "
                "Use a shorter video or pre-split the audio."
            )

        # Apply the Tagalog primer when the caller selects Tagalog and
        # supplies no prompt — mirrors LocalWhisperClient behavior.
        if language and language.lower() in {"tl", "fil", "tgl"} and not initial_prompt:
            from smartcut.core.whisper_local import TAGALOG_PRIMER
            initial_prompt = TAGALOG_PRIMER
            language = "tl"

        merged_prompt = " ".join(p for p in (initial_prompt, hotwords) if p) or None

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
                    if merged_prompt:
                        kwargs["prompt"] = merged_prompt
                    response = self.client.audio.transcriptions.create(**kwargs)
                return self._parse(response, min_word_probability=min_word_probability)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
                else:
                    raise RuntimeError(f"Whisper API failed after {MAX_RETRIES} attempts: {e}")

        raise RuntimeError("Unexpected error in transcription")

    def _parse(self, response, min_word_probability: float = 0.0) -> WhisperResult:
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
                    prob = float(getattr(w, "probability", 1.0) or 1.0)
                    if prob < min_word_probability:
                        continue
                    seg_words.append(WhisperWord(
                        word=getattr(w, "word", "").strip(),
                        start=ws,
                        end=we,
                        probability=prob,
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
