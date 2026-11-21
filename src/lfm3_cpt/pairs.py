"""Coarse minimal pairs from held-out text, for any language (`lfm3-cpt pairs`, `sefaria --pairs`).

The original sentence against (a) two neighbouring words swapped, (b) one word replaced by another word of the
same text with a similar frequency. Both corruptions are usually, but not always, wrong: the score measures how
well a model knows the language and register, not one grammar rule. Its value is that it can be made from any
held-out text, for languages that no benchmark covers.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from .words import is_word

_SPLIT_RE = re.compile(r"(?<=[.:;?!])\s+")


def make_pairs(docs: Iterable[dict[str, str]], n: int = 1000, seed: int = 0, words=(6, 30)) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sentences: list[list[str]] = []
    freq: Counter = Counter()
    for d in docs:
        for line in d["text"].splitlines():
            for sent in _SPLIT_RE.split(line):
                toks = sent.split()
                freq.update(t for t in toks if is_word(t))
                if words[0] <= len(toks) <= words[1]:
                    sentences.append(toks)
                elif len(toks) > words[1]:
                    start = rng.randrange(0, len(toks) - words[1] + 1)
                    sentences.append(toks[start : start + words[1]])
    if not sentences:
        return []
    by_band: dict[int, list[str]] = defaultdict(list)
    for w, c in freq.items():
        by_band[min(c.bit_length(), 12)].append(w)
    for band in by_band.values():
        band.sort()
    rng.shuffle(sentences)
    pairs: list[dict[str, str]] = []
    want = {"word_order": n // 2, "word_substitution": n - n // 2}
    seen: set[tuple[str, str]] = set()
    for toks in sentences:
        if not any(want.values()):
            break
        good = " ".join(toks)
        kind = "word_order" if want["word_order"] >= want["word_substitution"] else "word_substitution"
        if want[kind] == 0:
            kind = "word_order" if kind == "word_substitution" else "word_substitution"
        bad_toks = _swap(toks, rng) if kind == "word_order" else _substitute(toks, freq, by_band, rng)
        if bad_toks is None:
            continue
        bad = " ".join(bad_toks)
        if bad == good or (good, bad) in seen:
            continue
        seen.add((good, bad))
        pairs.append({"sentence_good": good, "sentence_bad": bad, "phenomenon": kind})
        want[kind] -= 1
    return pairs


def _swap(toks: list[str], rng: random.Random) -> list[str] | None:
    spots = [i for i in range(len(toks) - 1) if is_word(toks[i]) and is_word(toks[i + 1]) and toks[i] != toks[i + 1]]
    if not spots:
        return None
    i = rng.choice(spots)
    out = list(toks)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _substitute(toks: list[str], freq: Counter, by_band: dict[int, list[str]], rng: random.Random) -> list[str] | None:
    spots = [i for i, t in enumerate(toks) if is_word(t)]
    rng.shuffle(spots)
    for i in spots[:5]:
        band = by_band.get(min(freq[toks[i]].bit_length(), 12), [])
        choices = [w for w in rng.sample(band, min(len(band), 20)) if w != toks[i] and w not in toks]
        if choices:
            out = list(toks)
            out[i] = choices[0]
            return out
    return None
