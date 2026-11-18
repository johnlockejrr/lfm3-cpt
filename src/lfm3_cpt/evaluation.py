"""Evaluation: token-weighted perplexity on packed text, minimal-pair grammar tests, candidate ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

ProgressFn = Callable[[int], None]  # called with the number of items just processed


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:  # pragma: no cover - exotic wrappers
        return next(model.parameters()).device


@contextmanager
def eval_mode(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class PerplexityResult:
    loss: float
    perplexity: float
    tokens: int
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pack_token_rows(tokenizer: Any, texts: Sequence[str], max_length: int, batch: int = 256) -> list[list[int]]:
    """Tokenize like TRL (`tokenizer(text=...)`), concatenate and cut into rows of `max_length`.

    This mirrors TRL's "wrapped" packing used for training, so eval perplexity is measured on the same kind of
    rows the model is trained on.
    """
    flat: list[int] = []
    for i in range(0, len(texts), batch):
        for ids in tokenizer(text=list(texts[i : i + batch]))["input_ids"]:
            flat.extend(ids)
    return [flat[i : i + max_length] for i in range(0, len(flat), max_length)]


@torch.no_grad()
def corpus_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    batch_size: int = 2,
    progress: ProgressFn | None = None,
) -> PerplexityResult:
    """exp(total NLL / total predicted tokens) over the whole corpus (token-weighted, not batch-averaged)."""
    rows = [r for r in pack_token_rows(tokenizer, texts, max_length) if len(r) > 1]
    device = _input_device(model)
    total_nll, total_tokens = 0.0, 0
    # Full rows first, in batches; the final shorter row on its own (no padding needed at all).
    full = [r for r in rows if len(r) == max_length]
    tail = [r for r in rows if len(r) != max_length]
    batches = [full[i : i + batch_size] for i in range(0, len(full), batch_size)] + [[r] for r in tail]
    with eval_mode(model):
        for batch in batches:
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            for r in range(input_ids.shape[0]):  # one row at a time keeps the fp32 upcast small
                total_nll += F.cross_entropy(
                    logits[r, :-1].float(), input_ids[r, 1:].to(logits.device), reduction="sum"
                ).item()
                total_tokens += input_ids.shape[1] - 1
            if progress:
                progress(len(batch))
    loss = total_nll / max(total_tokens, 1)
    return PerplexityResult(loss=loss, perplexity=math.exp(min(loss, 50.0)), tokens=total_tokens, rows=len(rows))


def count_packed_rows(tokenizer: Any, texts: Sequence[str], max_length: int) -> int:
    return sum(1 for r in pack_token_rows(tokenizer, texts, max_length) if len(r) > 1)


# ---------------------------------------------------------------------------------------------------------------
# Minimal-pair grammar test (TurBLiMP or your own pairs)
# ---------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def sentence_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: Sequence[str],
    batch_size: int = 64,
    progress: ProgressFn | None = None,
) -> list[float]:
    """Total log-probability of each sentence, conditioned on BOS.

    Sentences are sorted by length so batches carry little padding. Padding is on the right, which matters for
    LFM3: its short convolutions are causal, so right padding never leaks into real tokens, left padding would.
    """
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [bos + tokenizer(s, add_special_tokens=False)["input_ids"] for s in sentences]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)
    scores = [0.0] * len(encoded)
    device = _input_device(model)
    with eval_mode(model):
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids[row, : len(encoded[i])] = torch.tensor(encoded[i])
                mask[row, : len(encoded[i])] = 1
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:, None].to(logits.device))[..., 0]
            totals = (logprobs * mask[:, 1:].to(logprobs.device)).sum(-1).tolist()
            for row, i in enumerate(idx):
                scores[i] = totals[row]
            if progress:
                progress(len(idx))
    return scores


@dataclass
class GrammarResult:
    passed: list[bool]
    phenomena: list[str]
    good_scores: list[float] = field(default_factory=list)
    bad_scores: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(self.passed) / len(self.passed) if self.passed else float("nan")

    def by_phenomenon(self) -> dict[str, float]:
        hits: dict[str, list[bool]] = defaultdict(list)
        for ok, p in zip(self.passed, self.phenomena, strict=True):
            hits[p].append(ok)
        return {p: sum(v) / len(v) for p, v in sorted(hits.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "n_pairs": len(self.passed),
            "by_phenomenon": self.by_phenomenon(),
            "passed": self.passed,
        }


def grammar_test(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: Any,
    batch_size: int = 64,
    progress: ProgressFn | None = None,
) -> GrammarResult:
    """A pair passes when the grammatical sentence gets the higher total log-probability."""
    good = sentence_logprobs(model, tokenizer, list(pairs["sentence_good"]), batch_size, progress)
    bad = sentence_logprobs(model, tokenizer, list(pairs["sentence_bad"]), batch_size, progress)
    passed = [g > b for g, b in zip(good, bad, strict=True)]
    return GrammarResult(passed, list(pairs["linguistic_phenomenon"]), good, bad)


def compare_grammar(pairs: Any, before: Sequence[bool], after: Sequence[bool], n_phenomena: int = 4) -> dict[str, Any]:
    """Pairs that flipped in each direction, plus one example per most-improved phenomenon.

    Unlike the notebook, a phenomenon without any wrong->right flip is simply skipped (no StopIteration).
    """
    phen = list(pairs["linguistic_phenomenon"])
    b = GrammarResult(list(before), phen).by_phenomenon()
    a = GrammarResult(list(after), phen).by_phenomenon()
    fixed = [i for i, (x, y) in enumerate(zip(before, after, strict=True)) if not x and y]
    broken = [i for i, (x, y) in enumerate(zip(before, after, strict=True)) if x and not y]
    ranked = sorted(a, key=lambda p: a[p] - b[p], reverse=True)
    examples = []
    for p in ranked:
        if len(examples) >= n_phenomena or a[p] - b[p] <= 0:
            break
        i = next((i for i in fixed if phen[i] == p), None)
        if i is None:
            continue
        examples.append(
            {
                "phenomenon": p,
                "before": b[p],
                "after": a[p],
                "sentence_good": pairs[i]["sentence_good"],
                "sentence_bad": pairs[i]["sentence_bad"],
            }
        )
    return {"fixed": len(fixed), "broken": len(broken), "before": b, "after": a, "examples": examples}


# ---------------------------------------------------------------------------------------------------------------
# Ranking candidate restorations (lacunae, conjectures, variant readings)
# ---------------------------------------------------------------------------------------------------------------


def predicted_tokens(tokenizer: Any, text: str) -> int:
    """How many tokens of `text` the model actually predicts in `sentence_logprobs` (all of them after BOS)."""
    n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return n if tokenizer.bos_token_id is not None else max(n - 1, 0)


def rank_candidates(
    model: torch.nn.Module,
    tokenizer: Any,
    text: str,
    candidates: Sequence[str],
    gap: str = "___",
    batch_size: int = 32,
    by: str = "total",
) -> list[dict[str, Any]]:
    """Fill the gap with each candidate and rank the full texts by log-probability under the model.

    `total` is the joint log-probability of the whole text: the principled score when the candidates compete for
    the same gap. It favours shorter fillers, so `mean` (per predicted token) is reported too; compare candidates
    of very different lengths with care.
    """
    if text.count(gap) != 1:
        raise ValueError(f"the text must contain the gap marker {gap!r} exactly once")
    if by not in {"total", "mean"}:
        raise ValueError("by must be 'total' or 'mean'")
    unique = list(dict.fromkeys(c for c in candidates if c is not None))
    if not unique:
        raise ValueError("no candidates")
    filled = [text.replace(gap, c) for c in unique]
    totals = sentence_logprobs(model, tokenizer, filled, batch_size)
    rows = []
    for cand, full, total in zip(unique, filled, totals, strict=True):
        n = predicted_tokens(tokenizer, full)
        rows.append({"candidate": cand, "text": full, "logprob": total, "tokens": n, "mean_logprob": total / max(n, 1)})
    key = "logprob" if by == "total" else "mean_logprob"
    rows.sort(key=lambda r: r[key], reverse=True)
    best = rows[0][key]
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["delta"] = r[key] - best
    return rows
