"""Restore a lacuna: propose fillers for a gap, filter them by what is legible, rank them by the model.

In a manuscript the candidates are rarely given. What is known is the size of the gap (roughly how many letters)
and sometimes a few legible letters or traces. This module

1. collects candidates from the model's own continuations of the text before the gap (one or more words),
   from every word of a corpus that fits the constraints (single-word gaps), and from the user's own list;
2. keeps those that fit: number of letters, and a letter pattern such as `נ??עין` (`?` one letter, `*` any run,
   a space separates words) or a regular expression;
3. ranks them by the log-probability of the whole text with the gap filled, so the words *after* the gap count
   as much as the words before it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

GAP = "___"
_LETTER = "א-ת"
_MARKS_RE = re.compile(r"[֑-ׇ׳״]")


def letters(text: str) -> int:
    """Letters in a filler: spaces, vowel points, geresh and gershayim don't count."""
    return sum(c.isalpha() for c in _MARKS_RE.sub("", text))


def pattern_regex(pattern: str) -> re.Pattern:
    """`נ??עין` -> one letter נ, two unknown letters, then עין. `*` is any run of letters; spaces separate words."""
    out = []
    for ch in pattern:
        if ch == "?":
            out.append(f"[{_LETTER}A-Za-z]")
        elif ch == "*":
            out.append(f"[{_LETTER}A-Za-z]*")
        elif ch.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$")


@dataclass
class Constraints:
    min_letters: int | None = None
    max_letters: int | None = None
    words: int | None = 1  # number of words in the gap; None = any
    pattern: str | None = None
    regex: str | None = None

    def __post_init__(self) -> None:
        self._pattern = pattern_regex(self.pattern) if self.pattern else None
        self._regex = re.compile(self.regex) if self.regex else None

    def fits(self, filler: str) -> bool:
        bare = _MARKS_RE.sub("", filler).strip()
        if not bare:
            return False
        if self.words is not None and len(bare.split()) != self.words:
            return False
        n = letters(bare)
        if self.min_letters is not None and n < self.min_letters:
            return False
        if self.max_letters is not None and n > self.max_letters:
            return False
        if self._pattern and not self._pattern.match(bare):
            return False
        return not (self._regex and not self._regex.search(bare))


def split_gap(text: str, gap: str = GAP) -> tuple[str, str]:
    if text.count(gap) != 1:
        raise ValueError(f"the text must contain the gap marker {gap!r} exactly once")
    left, right = text.split(gap)
    return left.rstrip(), right.lstrip()


def first_words(continuations: Iterable[str], n: int) -> Counter:
    """The first `n` whole words of each continuation; a continuation that doesn't start a new word (it goes on
    with the last word of the context) is skipped."""
    counts: Counter = Counter()
    for cont in continuations:
        if not cont[:1].isspace():
            continue
        words = cont.split()
        if len(words) > n or (len(words) == n and cont.rstrip() != cont):
            counts[" ".join(words[:n])] += 1
    return counts


def vocabulary_candidates(freq: Counter, constraints: Constraints, limit: int) -> list[str]:
    """Corpus words that fit the constraints, most frequent first (single-word gaps only)."""
    if constraints.words not in (1, None):
        return []
    fits = [w for w, _ in freq.most_common() if constraints.fits(w)]
    return fits[:limit]


def restore(
    text: str,
    constraints: Constraints,
    logprob_fn: Callable[[list[str]], list[float]],
    sample_fn: Callable[[str], list[str]] | None = None,
    vocab: Counter | None = None,
    given: Sequence[str] = (),
    max_vocab: int = 50_000,
    gap: str = GAP,
    norm: Callable[[str], str] = lambda s: s,
    pmi: float = 0.0,
    rerank: int = 300,
) -> dict[str, Any]:
    """Collect, filter and rank fillers. Returns the ranked rows and how many candidates each source gave.

    With `pmi`, the correction only reorders the `rerank` best fillers by raw probability (0 = all of them): in an
    open search over a whole corpus, it would otherwise lift words that are merely rare (and the rarest words of a
    corpus include its oddities) above words that actually fit."""
    left, right = split_gap(text, gap)
    sources: dict[str, set[str]] = {}

    def add(filler: str, source: str) -> None:
        filler = norm(" ".join(filler.split()))
        if filler and constraints.fits(filler):
            sources.setdefault(filler, set()).add(source)

    counts: dict[str, int] = {"given": 0, "model": 0, "corpus": 0}
    seen: Counter = Counter()
    for g in given:
        add(g, "given")
    if sample_fn is not None:
        conts = sample_fn(left)
        for n in [constraints.words] if constraints.words else [1, 2, 3]:
            seen.update(first_words(conts, n))
        for filler in seen:
            add(filler, "model")
    if vocab:
        for w in vocabulary_candidates(vocab, constraints, max_vocab):
            add(w, "corpus")
    for srcs in sources.values():
        for s in srcs:
            counts[s] += 1
    if not sources:
        return {"rows": [], "counts": counts, "left": left, "right": right}

    fillers = list(sources)
    texts = [" ".join(x for x in (left, f, right) if x) for f in fillers]
    scores = logprob_fn(texts)
    rows = [
        {
            "filler": f,
            "logprob": s,
            "raw_logprob": s,
            "raw_rank": 0,
            "letters": letters(f),
            "sources": sorted(sources[f]),
            "sampled": seen.get(f, 0),
        }
        for f, s in zip(fillers, scores, strict=True)
    ]
    rows.sort(key=lambda r: r["raw_logprob"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["raw_rank"] = i
    if pmi:
        head = rows[:rerank] if rerank else rows
        priors = logprob_fn([r["filler"] for r in head])
        for r, prior in zip(head, priors, strict=True):
            r["logprob"] = r["raw_logprob"] - pmi * prior
        head.sort(key=lambda r: r["logprob"], reverse=True)
        tail = rows[len(head) :]
        if tail:  # below the reranked head, in raw order, and scored so that they stay below it
            floor = min(r["logprob"] for r in head)
            for r in tail:
                r["logprob"] = floor - (head[-1]["raw_logprob"] - r["raw_logprob"]) - 1e-6
        rows = head + tail
    best = rows[0]["logprob"]
    for i, r in enumerate(rows, 1):
        r["rank"], r["delta"] = i, r["logprob"] - best
    return {"rows": rows, "counts": counts, "left": left, "right": right}
