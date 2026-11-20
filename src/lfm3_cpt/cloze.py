"""Gap-filling (cloze) benchmark: can the model restore a missing word among plausible alternatives?

The minimal pairs ask "which sentence is broken?" and are easy: a random word usually breaks a sentence. A lacuna
asks "which of these plausible words was here?". This module builds such items once, from held-out text, and
scores any number of models on the same file, so the numbers are comparable:

* the masked word is a content word (not one of the most frequent words), with at least a few words of context;
* wrong options from the **corpus** share the word's form class: same first letter (often a prefix: ו ה ב ל מ ש כ),
  same ending (ין, ים, ות, ה, ...), similar length and similar frequency in the training text;
* optional wrong options from a **proposer** model: words it actually predicts in that slot. These are the hard
  ones. The proposer is disadvantaged when it is also scored (the options were chosen against it), so use a
  model that is not compared, or read the corpus-only numbers for it.

Items are JSONL: {"id", "source", "text" (with ___), "answer", "candidates", "kinds", "freq"}. A file of real
variant readings in the same format (e.g. Kaufmann against Parma) can be scored the same way.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

GAP = "___"
_HEB_WORD = re.compile(r"^[א-ת]+$")
_PREFIXES = set("ובהלמשכ")
BUCKETS = (("unseen", 0, 0), ("rare (1-3)", 1, 3), ("medium (4-30)", 4, 30), ("common (31+)", 31, math.inf))


def is_word(w: str) -> bool:
    return bool(_HEB_WORD.match(w))


def bucket(freq: int) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= freq <= hi)


def form_key(word: str) -> tuple[str, str]:
    """(first letter if it can be a prefix, last two letters): words that could fill the same grammatical slot."""
    first = word[0] if word[0] in _PREFIXES else ""
    return first, word[-2:]


def strip_vh(word: str) -> str:
    """The word without a leading ו and/or ה: variants such as והילך/הילך or התשיעי/תשיעי, which manuscripts
    often disagree on."""
    for p in ("וה", "ו", "ה"):
        if word.startswith(p) and len(word) - len(p) >= 2:
            return word[len(p) :]
    return word


def _band(freq: int) -> int:
    return min(freq, 1 << 20).bit_length()


@dataclass
class Vocabulary:
    freq: Counter
    stop: set[str]
    by_form: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    by_end: dict[str, list[str]] = field(default_factory=dict)
    by_first: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_docs(cls, docs: Iterable[str], stop_top: int = 150) -> Vocabulary:
        freq: Counter = Counter()
        for d in docs:
            freq.update(w for w in d.split() if is_word(w))
        stop = {w for w, _ in freq.most_common(stop_top)}
        by_form: dict[tuple[str, str], list[str]] = defaultdict(list)
        by_end: dict[str, list[str]] = defaultdict(list)
        by_first: dict[str, list[str]] = defaultdict(list)
        for w in sorted(freq):
            if len(w) >= 2 and w not in stop:
                by_form[form_key(w)].append(w)
                by_end[w[-2:]].append(w)
                by_first[w[0]].append(w)
        return cls(freq, stop, dict(by_form), dict(by_end), dict(by_first))

    def distractors(self, answer: str, n: int, rng: random.Random, exclude: set[str]) -> list[str]:
        """Same form class; similar length and frequency band first, relaxed step by step if too few: then the
        same ending (ין, ות, ...) with any first letter, then the same first letter."""
        band = _band(self.freq[answer])

        def ok(w: str) -> bool:
            return w != answer and w not in exclude

        def close(w: str) -> bool:
            return abs(len(w) - len(answer)) <= 1

        pool = [w for w in self.by_form.get(form_key(answer), []) if ok(w)]
        ending = [w for w in self.by_end.get(answer[-2:], []) if ok(w)]
        first = [w for w in self.by_first.get(answer[0], []) if ok(w)]
        tiers = [
            [w for w in pool if close(w) and abs(_band(self.freq[w]) - band) <= 1],
            [w for w in pool if close(w)],
            pool,
            [w for w in ending if close(w)],
            ending,
            [w for w in first if close(w)],
        ]
        chosen: list[str] = []
        for tier in tiers:
            options = [w for w in tier if w not in chosen]
            rng.shuffle(options)
            chosen += options[: n - len(chosen)]
            if len(chosen) >= n:
                break
        return chosen


Proposer = Callable[[str, str], list[str]]  # (left context, answer) -> proposed words, best first


def build_items(
    eval_docs: Sequence[str],
    vocab: Vocabulary,
    n_items: int = 400,
    seed: int = 0,
    corpus_distractors: int = 4,
    model_distractors: int = 0,
    proposer: Proposer | None = None,
    min_left: int = 4,
    window: int = 40,
    min_len: int = 3,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    lines: list[tuple[int, list[str]]] = []
    for di, doc in enumerate(eval_docs):
        for line in doc.splitlines():
            toks = line.split()
            if len(toks) >= min_left + 2:
                lines.append((di, toks))
    rng.shuffle(lines)
    items: list[dict[str, Any]] = []
    for di, toks in lines:
        if len(items) >= n_items:
            break
        spots = [
            i for i, w in enumerate(toks) if i >= min_left and is_word(w) and len(w) >= min_len and w not in vocab.stop
        ]
        if not spots:
            continue
        i = rng.choice(spots)
        start = max(0, i - window + 8)  # most of the window before the gap, a little after
        words = toks[start : start + window]
        pos = i - start
        answer = words[pos]
        left = " ".join(words[:pos])
        kinds: dict[str, str] = {}
        if proposer and model_distractors:
            for w in proposer(left, answer):
                if len(kinds) >= model_distractors:
                    break
                if w != answer and is_word(w) and len(w) >= 2 and w not in kinds:
                    kinds[w] = "model"
        for w in vocab.distractors(answer, corpus_distractors, rng, set(kinds) | {answer}):
            kinds[w] = "corpus"
        if sum(k == "corpus" for k in kinds.values()) < min(corpus_distractors, 2):
            continue
        candidates = [answer, *kinds]
        rng.shuffle(candidates)
        items.append(
            {
                "id": len(items) + 1,
                "source": di,
                "text": " ".join([*words[:pos], GAP, *words[pos + 1 :]]),
                "answer": answer,
                "candidates": candidates,
                "kinds": kinds,
                "freq": vocab.freq[answer],
            }
        )
        if progress:
            progress(len(items))
    return items


def score_items(
    logprob_fn: Callable[[list[str]], list[float]], items: Sequence[dict[str, Any]], gap: str = GAP
) -> list[dict[str, Any]]:
    """Rank each item's candidates by the log-probability of the filled text. `logprob_fn` scores a batch."""
    texts: list[str] = []
    for it in items:
        if it["text"].count(gap) != 1:
            raise ValueError(f"item {it.get('id')}: the text must contain {gap!r} exactly once")
        texts += [it["text"].replace(gap, c) for c in it["candidates"]]
    scores = logprob_fn(texts)
    out, k = [], 0
    for it in items:
        cands = it["candidates"]
        s = dict(zip(cands, scores[k : k + len(cands)], strict=True))
        k += len(cands)
        ans = s[it["answer"]]
        rank = 1 + sum(v > ans for c, v in s.items() if c != it["answer"])
        best = max(s, key=s.get)
        wins: dict[str, list[bool]] = defaultdict(list)
        for c, kind in (it.get("kinds") or {}).items():
            if c in s:
                wins[kind].append(ans > s[c])
        out.append(
            {
                "id": it.get("id"),
                "answer": it["answer"],
                "rank": rank,
                "n": len(cands),
                "best": best,
                "lenient": strip_vh(best) == strip_vh(it["answer"]),
                "freq": it.get("freq"),
                "wins": dict(wins),
                "scores": s,
            }
        )
    return out


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {"items": 0}
    ranks = [r["rank"] for r in results]
    summary: dict[str, Any] = {
        "items": n,
        "candidates": sum(r["n"] for r in results) / n,
        "chance_top1": sum(1 / r["n"] for r in results) / n,
        "top1": sum(x == 1 for x in ranks) / n,
        "top1_lenient": sum(r.get("lenient", r["rank"] == 1) for r in results) / n,
        "top3": sum(x <= 3 for x in ranks) / n,
        "mean_rank": sum(ranks) / n,
        "mrr": sum(1 / x for x in ranks) / n,
        "by_bucket": {},
        "pairwise": {},
    }
    for name, _, _ in BUCKETS:
        sub = [r for r in results if r.get("freq") is not None and bucket(r["freq"]) == name]
        if sub:
            summary["by_bucket"][name] = (sum(r["rank"] == 1 for r in sub) / len(sub), len(sub))
    kinds: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for kind, w in r["wins"].items():
            kinds[kind] += w
    for kind, w in kinds.items():
        summary["pairwise"][kind] = (sum(w) / len(w), len(w))
    return summary
