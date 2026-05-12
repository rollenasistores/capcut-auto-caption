"""Local Whisper backend via faster-whisper — tuned for Tagalog accuracy."""

from pathlib import Path
from typing import Optional, Sequence

from smartcut.core.whisper_client import WhisperResult, WhisperSegment, WhisperWord


# A neutral Filipino primer. Whisper biases its decoder toward the vocabulary
# and register present in `initial_prompt`. We seed common Tagalog function
# words, polite forms, and signal that Tag-Lish code-switching is expected —
# this is the single biggest accuracy win over CapCut's built-in ASR for
# Filipino talking-head content.
TAGALOG_PRIMER = (
    "Magandang araw sa inyong lahat. Salamat sa panonood ng video na ito. "
    "Ang usapan ay halo-halong Tagalog at English (Taglish), kaya may mga "
    "salitang Filipino, mga pangalan ng tao at lugar sa Pilipinas, at mga "
    "technical na termino. Halimbawa: kasi, talaga, sobra, ganun, kahit, "
    "pwede, parang, naman, lang, dito, doon, ako, ikaw, tayo, kayo, sila."
)

TAGALOG_LANGS = frozenset({"tl", "fil", "tgl"})


class LocalWhisperClient:
    """Runs Whisper inference locally with faster-whisper (CTranslate2).

    First call downloads the model (see :mod:`smartcut.core.model_download`
    for visible progress). Subsequent calls reuse the cached weights.

    Defaults are tuned to *beat* CapCut's built-in auto-caption for Filipino
    content: anti-hallucination guards, multilingual decoding (for Tag-Lish
    code-switching), and a Filipino-flavored initial prompt when the caller
    selects Tagalog.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        cpu_threads: int = 0,
        num_workers: int = 1,
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
        self.compute_type = compute_type

        from smartcut.core.model_download import ensure_model_downloaded

        model_path = ensure_model_downloaded(model_size, progress=True)
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: Optional[Sequence[float]] = None,
        condition_on_previous_text: bool = False,
        multilingual: bool = True,
        hallucination_silence_threshold: float = 2.0,
        no_speech_threshold: float = 0.45,
        log_prob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4,
        vad_min_silence_ms: int = 500,
        vad_speech_pad_ms: int = 200,
        min_word_probability: float = 0.0,
    ) -> WhisperResult:
        """Transcribe an audio file with accuracy-tuned decoding.

        Parameters worth tweaking from the MCP caller:

        - ``language``: ISO code. Set to ``"tl"`` for Tagalog; that also
          triggers the built-in :data:`TAGALOG_PRIMER` if no
          ``initial_prompt`` is supplied.
        - ``initial_prompt``: free-text context shown to the decoder. Use it
          to bias toward Tagalog or to introduce proper nouns (names of
          people, brands, products).
        - ``hotwords``: short, comma-separated string of must-recognize
          words. Distinct from ``initial_prompt`` — these get extra weight
          during beam search. Newer faster-whisper feature.
        - ``min_word_probability``: drop words below this confidence (0.0
          disables). Helps strip ASR hallucinations like ``[Music]``.
        - ``condition_on_previous_text``: kept ``False`` by default. For
          talking-head content with pauses, true causes the model to loop
          on the previous (often wrong) transcript.
        - ``multilingual``: ``True`` so Tag-Lish code-switching mid-sentence
          decodes correctly (Whisper otherwise locks to a single language
          per chunk).
        """
        if temperature is None:
            temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

        normalized_lang = language.lower() if language else None
        if normalized_lang in TAGALOG_LANGS and not initial_prompt:
            initial_prompt = TAGALOG_PRIMER
            normalized_lang = "tl"

        kwargs = dict(
            language=normalized_lang,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            best_of=best_of,
            temperature=tuple(temperature),
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": vad_min_silence_ms,
                "speech_pad_ms": vad_speech_pad_ms,
            },
            hallucination_silence_threshold=hallucination_silence_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            multilingual=multilingual,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        if hotwords:
            kwargs["hotwords"] = hotwords

        segments_iter, info = self.model.transcribe(str(audio_path), **kwargs)

        out_segments: list[WhisperSegment] = []
        for seg in segments_iter:
            words: list[WhisperWord] = []
            for w in (seg.words or []):
                if w.start is None or w.end is None:
                    continue
                prob = float(getattr(w, "probability", 1.0) or 1.0)
                if prob < min_word_probability:
                    continue
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append(WhisperWord(
                    word=token,
                    start=float(w.start),
                    end=float(w.end),
                    probability=prob,
                ))

            text = (seg.text or "").strip()
            if not words and not text:
                continue

            out_segments.append(WhisperSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                words=words,
                avg_logprob=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
            ))

        return WhisperResult(
            language=info.language,
            duration=float(info.duration),
            segments=out_segments,
            language_probability=float(getattr(info, "language_probability", 1.0) or 1.0),
        )
