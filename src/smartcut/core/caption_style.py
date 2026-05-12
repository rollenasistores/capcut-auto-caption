"""Caption style presets and Tagalog-aware filler word stripping.

These are the levers that separate "good" auto-captions from "great" ones:

* **Style presets** map a short name (``"tiktok"``, ``"minimal"`` …) onto a
  concrete :class:`~smartcut.core.capcut_reader.TextStyle` plus chunk
  defaults. The MCP caller picks an aesthetic; we materialise it.
* **Filler word stripping** removes common Tagalog/English speech
  disfluencies that Whisper faithfully transcribes (``"uhm"``, ``"ah"``,
  stutter repetitions like ``"kasi kasi"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from smartcut.core.capcut_reader import TextStyle


# Comprehensive (Tag-Lish friendly) filler / disfluency list. Case-folded
# at runtime; punctuation is stripped before comparison.
DEFAULT_FILLERS = frozenset({
    # English disfluencies
    "uh", "uhm", "um", "ahm", "ah", "er", "erm", "hm", "hmm", "mm",
    "like", "you know", "i mean", "sort of", "kind of",
    # Filipino disfluencies
    "ano", "eh", "ano ba", "kasi nga", "di ba", "diba", "yun nga",
    "alam mo", "kuan", "kuwan",
})


@dataclass
class CaptionPreset:
    """Bundle of TextStyle + chunking defaults that defines an aesthetic."""

    name: str
    style: TextStyle
    min_words: int = 2
    max_words: int = 4
    max_chars: Optional[int] = None
    max_duration_sec: Optional[float] = None
    description: str = ""


PRESETS: dict[str, CaptionPreset] = {
    "tiktok": CaptionPreset(
        name="tiktok",
        style=TextStyle(
            font_size=15,
            font_color="#FFFFFF",
            background_color=None,
            background_alpha=0.0,
            position_y=0.5,
            bold=True,
        ),
        min_words=2,
        max_words=4,
        max_chars=24,
        max_duration_sec=2.2,
        description="Bold center captions, 2-4 words — TikTok / Reels style.",
    ),
    "tiktok-yellow": CaptionPreset(
        name="tiktok-yellow",
        style=TextStyle(
            font_size=16,
            font_color="#FFE600",
            background_color=None,
            background_alpha=0.0,
            position_y=0.55,
            bold=True,
        ),
        min_words=1,
        max_words=3,
        max_chars=20,
        max_duration_sec=1.8,
        description="Punchy yellow bold one-to-three-word cards — high-energy creators.",
    ),
    "karaoke": CaptionPreset(
        name="karaoke",
        style=TextStyle(
            font_size=16,
            font_color="#FFFFFF",
            background_color=None,
            background_alpha=0.0,
            position_y=0.5,
            bold=True,
        ),
        min_words=1,
        max_words=1,
        max_chars=16,
        max_duration_sec=1.2,
        description="One-word-per-card karaoke style.",
    ),
    "minimal": CaptionPreset(
        name="minimal",
        style=TextStyle(
            font_size=10,
            font_color="#FFFFFF",
            background_color="#000000",
            background_alpha=0.55,
            position_y=0.88,
            bold=False,
        ),
        min_words=3,
        max_words=7,
        max_chars=42,
        max_duration_sec=3.5,
        description="Lower-third subtitle with translucent bar — classic look.",
    ),
    "news": CaptionPreset(
        name="news",
        style=TextStyle(
            font_size=11,
            font_color="#FFFFFF",
            background_color="#0A0A0A",
            background_alpha=0.85,
            position_y=0.9,
            bold=True,
        ),
        min_words=5,
        max_words=10,
        max_chars=60,
        max_duration_sec=4.0,
        description="Broadcast-style lower third with opaque bar.",
    ),
    "podcast": CaptionPreset(
        name="podcast",
        style=TextStyle(
            font_size=12,
            font_color="#FFFFFF",
            background_color=None,
            background_alpha=0.0,
            position_y=0.78,
            bold=True,
        ),
        min_words=4,
        max_words=8,
        max_chars=50,
        max_duration_sec=3.0,
        description="Roomy captions for talking-head / podcast cuts.",
    ),
}


def get_preset(name: str) -> CaptionPreset:
    key = (name or "").strip().lower()
    if key not in PRESETS:
        raise ValueError(
            f"Unknown caption preset '{name}'. "
            f"Choose from: {', '.join(sorted(PRESETS))}."
        )
    return PRESETS[key]


def list_presets() -> list[dict]:
    """Return a JSON-friendly summary of every preset."""
    return [
        {
            "name": p.name,
            "min_words": p.min_words,
            "max_words": p.max_words,
            "max_chars": p.max_chars,
            "max_duration_sec": p.max_duration_sec,
            "font_size": p.style.font_size,
            "bold": p.style.bold,
            "background": p.style.background_color is not None,
            "description": p.description,
        }
        for p in PRESETS.values()
    ]


def is_filler(word: str, extra: Optional[set[str]] = None) -> bool:
    """Return True if a stripped token is a recognised filler."""
    if not word:
        return True
    clean = word.strip().lower().strip(".,!?…—–-")
    if not clean:
        return True
    if clean in DEFAULT_FILLERS:
        return True
    if extra and clean in extra:
        return True
    return False


def strip_fillers_from_words(
    words: list[dict],
    extra: Optional[set[str]] = None,
    drop_stutter: bool = True,
) -> list[dict]:
    """Remove filler tokens and immediate stutter repetitions from a word list.

    Each ``word`` dict carries at least a ``"word"`` key. The function
    returns a *new* list — input is not mutated.
    """
    out: list[dict] = []
    prev_clean: Optional[str] = None

    for w in words:
        tok = w.get("word", "")
        if is_filler(tok, extra=extra):
            prev_clean = None
            continue

        if drop_stutter:
            clean = tok.strip().lower().strip(".,!?…—–-")
            if clean and clean == prev_clean:
                continue
            prev_clean = clean

        out.append(w)

    return out
