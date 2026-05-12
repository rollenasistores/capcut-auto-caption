"""Local Whisper backend via faster-whisper — no API key required."""

from pathlib import Path
from typing import Optional

from smartcut.core.whisper_client import WhisperResult, WhisperSegment, WhisperWord


class LocalWhisperClient:
    """Runs Whisper inference locally with faster-whisper (CTranslate2).

    First call downloads the model from HuggingFace (~150MB for 'base',
    ~3GB for 'large-v3'). Subsequent calls reuse the cached model.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: Optional[str] = None,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. Install with:\n"
                "    pip install faster-whisper\n"
                "or:  pip install -e '.[local]'"
            ) from e

        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        self.model_size = model_size
        self.device = device

        from smartcut.core.model_download import ensure_model_downloaded

        model_path = ensure_model_downloaded(model_size, progress=True)
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> WhisperResult:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        out_segments: list[WhisperSegment] = []
        for seg in segments_iter:
            words: list[WhisperWord] = []
            for w in (seg.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append(WhisperWord(word=(w.word or "").strip(), start=w.start, end=w.end))
            out_segments.append(WhisperSegment(
                start=seg.start,
                end=seg.end,
                text=(seg.text or "").strip(),
                words=words,
            ))

        return WhisperResult(
            language=info.language,
            duration=info.duration,
            segments=out_segments,
        )
