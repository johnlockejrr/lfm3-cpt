"""What counts as a word, for any alphabet: used by the minimal pairs, the gap-filling benchmark and `restore`.

A word is a run of letters (any script), optionally with Hebrew points and geresh/gershayim, and with apostrophes
or hyphens *inside* it or a final apostrophe: Maltese `il-kelma`, `għall-ewwel`, `ta'`; Hebrew `ר׳`, `שֶׁ`. Digits,
punctuation and symbols are not words.
"""

from __future__ import annotations

import re

_LETTER = r"[^\W\d_]"
_MARK = r"[֑-ׇ׳״]"  # Hebrew cantillation, points, geresh, gershayim
_PIECE = rf"(?:{_LETTER}|{_MARK})+"
WORD_RE = re.compile(rf"^{_PIECE}(?:['’\-\u2010\u2011]{_PIECE})*['’]?$")  # - ‐ ‑ are all hyphens
ABBREVIATION_MARKS = "׳״"


def letters(word: str) -> int:
    return sum(c.isalpha() for c in word)


def is_word(token: str, min_letters: int = 2) -> bool:
    return bool(WORD_RE.match(token)) and letters(token) >= min_letters


def is_abbreviation(token: str) -> bool:
    """Hebrew abbreviations and truncations (ר׳, רז״ל): words, but poor gaps to restore."""
    return any(m in token for m in ABBREVIATION_MARKS)
