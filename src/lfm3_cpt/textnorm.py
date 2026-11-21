"""Text normalization: one Unicode form, and optional removal of vocalization / cantillation / diacritic marks.

The same normalization is applied to every text the model sees or is scored on (training corpus, eval sets,
minimal pairs, SFT conversations, `rank` candidates), so a word is always tokenized the same way.

Why it matters: in pointed Hebrew or vocalized Arabic every word carries several combining marks. A tokenizer
that does not know them splits words into many pieces, and the "same" word written with and without points
becomes two unrelated token sequences. Pick one convention for the whole project.
"""

from __future__ import annotations

import unicodedata


def _ranges(*spans: tuple[int, int]) -> frozenset[str]:
    return frozenset(chr(c) for lo, hi in spans for c in range(lo, hi + 1))


MARK_SETS: dict[str, frozenset[str]] = {
    # Hebrew accents (te'amim), U+0591-U+05AF
    "hebrew_cantillation": _ranges((0x0591, 0x05AF)),
    # Hebrew vowel points and dagesh/shin/sin dots (niqqud); maqaf, paseq and sof pasuq are kept
    "hebrew_points": _ranges((0x05B0, 0x05BD), (0x05BF, 0x05BF), (0x05C1, 0x05C2), (0x05C4, 0x05C5), (0x05C7, 0x05C7)),
    # Arabic harakat, tanwin, shadda, sukun, superscript alef, Quranic annotation signs
    "arabic_diacritics": _ranges((0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED)),
    # Arabic tatweel (kashida), a purely typographic elongation
    "arabic_tatweel": _ranges((0x0640, 0x0640)),
    # Syriac vowels and diacritics
    "syriac_points": _ranges((0x0730, 0x074A)),
    # Samaritan vowel signs and marks
    "samaritan_marks": _ranges((0x0816, 0x082D)),
    # Latin macrons and breves (ā, ă ...), common in school editions of Latin texts
    "latin_macrons": frozenset({"̄", "̆"}),
}

# Not removals but unifications, listed in `strip_marks` like the mark sets: one spelling of a character that
# texts type in several ways. Maltese writes `ta'`, `għall-`: a typographic apostrophe would make a second word.
REPLACEMENT_SETS: dict[str, dict[str, str]] = {
    "typographic_apostrophes": {"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u0060": "'", "\u00b4": "'"},
    # U+2010 hyphen and U+2011 non-breaking hyphen, as word processors type them: il‐Kummissjoni -> il-Kummissjoni
    "typographic_hyphens": {"\u2010": "-", "\u2011": "-"},
}

UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD", "none")


def validate(form: str, strip: list[str] | tuple[str, ...]) -> None:
    if form not in UNICODE_FORMS:
        raise ValueError(f"unicode_form must be one of {UNICODE_FORMS}, got {form!r}")
    unknown = [s for s in strip if s not in MARK_SETS and s not in REPLACEMENT_SETS]
    if unknown:
        raise ValueError(f"unknown strip_marks {unknown}; choose from {sorted([*MARK_SETS, *REPLACEMENT_SETS])}")


def normalize(text: str, form: str = "NFC", strip: list[str] | tuple[str, ...] = ()) -> str:
    for name in strip:
        if name in REPLACEMENT_SETS:
            text = text.translate(str.maketrans(REPLACEMENT_SETS[name]))
    strip = [s for s in strip if s in MARK_SETS]
    if strip:
        marks = frozenset().union(*(MARK_SETS[s] for s in strip))
        # Decompose first so precomposed letters (e.g. "ā") expose their combining marks.
        text = "".join(ch for ch in unicodedata.normalize("NFD", text) if ch not in marks)
    if form != "none":
        text = unicodedata.normalize(form, text)
    return text


class Normalizer:
    """Callable bound to a config's normalization settings."""

    def __init__(self, form: str = "NFC", strip: list[str] | tuple[str, ...] = ()):
        validate(form, strip)
        self.form, self.strip = form, tuple(strip)

    @classmethod
    def from_config(cls, cfg) -> Normalizer:
        return cls(cfg.unicode_form, cfg.strip_marks)

    def __call__(self, text: str) -> str:
        return normalize(text, self.form, self.strip)

    @property
    def is_identity(self) -> bool:
        return self.form == "none" and not self.strip
