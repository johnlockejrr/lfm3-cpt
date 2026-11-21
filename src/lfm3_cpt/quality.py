"""Corpus quality filter for any language: OCR garbage and boilerplate (`lfm3-cpt clean`).

Two kinds of noise survive curated corpora:

* **garbage lines**, mostly from OCR of scanned documents: `B FN wAġ Għaxar Snin Sezzjoni ia`. A line is dropped
  when too large a share of its tokens are not words: not letters (after stripping punctuation), mixed case
  inside a word (`wAġ`), or tokens of one or two letters that are not frequent in this corpus (Maltese `u`,
  `li`, `ta'` are; OCR fragments like `FN`, `ia` are not);
* **boilerplate**: the same line in many documents of a file (headers, lists of members, agenda templates).
  Lines are compared lower-cased with digits ignored, so `Laqgħa Nru 26` and `Laqgħa Nru 27` count as one.

A document that loses most of its text is dropped entirely. Everything is counted, with examples, so the filter
can be checked before training on its output.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .words import is_word

_PUNCT = "\"'“”‘’«»()[]{}<>.,;:!?…–—-*•·/\\|"
_NUMBER_RE = re.compile(r"^[\d.,:/%€$£+-]+$")
_MIXED_CASE_RE = re.compile(r"[^\W\d_]*[a-zà-ÿċġħż][A-ZÀ-ÞĊĠĦŻ]")
_QUOTES = '"“”„«»'
_JOINERS = ".…/:;,=+&@_"  # inside a token: an abbreviation (p.m., Ph.D.), URL or joined words, not noise


def _script(ch: str) -> str:
    return unicodedata.name(ch, "").split(" ", 1)[0] if ch.isalpha() else ""


def _mojibake(tok: str) -> bool:
    """Broken encodings: symbols inside words (tieg©u), letters of two scripts in one word (fiΩ-Ωmien), or a
    combining mark standing alone (tieg ̇u)."""
    if any(unicodedata.category(c) in {"So", "Sm"} for c in tok):
        return True
    if tok and unicodedata.category(tok[0]) == "Mn":
        return True
    return len({_script(c) for c in tok} - {""}) > 1


def _key(line: str) -> str:
    norm = re.sub(r"\d+", "0", " ".join(line.lower().split()))
    return hashlib.sha1(norm.encode()).hexdigest()


@dataclass
class Settings:
    max_bad: float = 0.3  # drop a line when more than this share of its tokens are not words
    min_tokens: int = 3  # shorter lines are judged only if they have no word at all
    min_bad: int = 2  # ... and longer ones need at least this many bad tokens: one odd token is no garbage
    boilerplate: bool = True  # off for literature, where repeated lines can be the text (biblical formulas)
    boilerplate_docs: int = 5  # a line in at least this many documents of a file is boilerplate
    boilerplate_min_chars: int = 15  # shorter repeated lines ("Iva.", "Grazzi.") are ordinary text
    min_keep: float = 0.5  # drop a document that keeps less than this share of its characters
    min_chars: int = 200  # ... or less than this many characters
    short_min_share: float = 1e-4  # a 1-2 letter token is a word if it is at least this share of all tokens
    short_min_count: int = 5  # ... and occurs at least this often


@dataclass
class Report:
    docs_in: int = 0
    docs_out: int = 0
    chars_in: int = 0
    chars_out: int = 0
    lines: Counter = field(default_factory=Counter)  # reason -> lines dropped
    examples: dict[str, list[str]] = field(default_factory=dict)


def _token_ok(tok: str, common_short: set[str]) -> bool | None:
    """True = a word, False = noise, None = neutral (a number or punctuation only)."""
    if _mojibake(tok.strip(_PUNCT)):
        return False
    t = "".join(c for c in tok if c not in _QUOTES).strip(_PUNCT)  # Id-"diskors," -> Id-diskors
    if not t or _NUMBER_RE.match(t) or any(c.isdigit() for c in t):
        return None  # numbers, and words with digits (Maltese 11-il, fl-1964) are neither
    if any(c in _JOINERS for c in t):
        return None
    if _MIXED_CASE_RE.match(t) and not t[:1].isupper():
        return False  # wAġ, l0ng: case changes inside a word (but McDonald, ĠUSTIZZJA are fine)
    if len(t) <= 2:
        return t.isalpha() and t.lower() in common_short
    return is_word(t, min_letters=2)


def line_is_garbage(line: str, common_short: set[str], s: Settings) -> bool:
    verdicts = [v for v in (_token_ok(t, common_short) for t in line.split()) if v is not None]
    if not verdicts:
        return False  # numbers / punctuation only: leave (tables, dates); the document rule decides
    bad = verdicts.count(False)
    if len(verdicts) < s.min_tokens:
        return bad == len(verdicts)
    return bad >= s.min_bad and bad / len(verdicts) > s.max_bad


def clean_documents(texts: list[str], s: Settings, show: int = 3) -> tuple[list[str], Report]:
    """Filter one file's documents (two passes: statistics, then filtering)."""
    rep = Report(docs_in=len(texts), chars_in=sum(len(t) for t in texts))
    tokens: Counter = Counter()
    line_docs: Counter = Counter()
    for t in texts:
        seen: set[str] = set()
        for line in t.splitlines():
            if len(line.strip()) >= s.boilerplate_min_chars:
                seen.add(_key(line))
            tokens.update(w.strip(_PUNCT).lower() for w in line.split())
        line_docs.update(seen)
    total = sum(tokens.values()) or 1
    floor = max(s.short_min_count, s.short_min_share * total)
    common_short = {w for w, n in tokens.items() if len(w) <= 2 and w.isalpha() and n >= floor}
    boiler = {k for k, n in line_docs.items() if n >= s.boilerplate_docs} if s.boilerplate else set()

    out: list[str] = []
    for t in texts:
        kept: list[str] = []
        for line in t.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            reason = None
            if len(stripped) >= s.boilerplate_min_chars and _key(line) in boiler:
                reason = "boilerplate"
            elif line_is_garbage(stripped, common_short, s):
                reason = "garbage"
            if reason:
                rep.lines[reason] += 1
                ex = rep.examples.setdefault(reason, [])
                if len(ex) < show and stripped not in ex:
                    ex.append(stripped[:160])
            else:
                kept.append(stripped)
        doc = "\n".join(kept)
        if len(doc) < s.min_chars or len(doc) < s.min_keep * len(t):
            rep.lines["documents dropped"] += 1
            continue
        out.append(doc)
    rep.docs_out = len(out)
    rep.chars_out = sum(len(t) for t in out)
    return out, rep
