"""How well does the tokenizer cover a language? Measure it before training.

Two numbers, compared with an English reference paragraph:

- characters per token: mostly about *script coverage*. Well-covered text gets ~3-5 characters per token. A
  script the vocabulary barely knows falls back to single characters or even single bytes (below 1: a Hebrew
  or Arabic letter is 2 bytes in UTF-8).
- tokens per word: also reflects morphology (Turkish words are long), so read it together with the first.

A poor result means slower training, a shorter effective context, and a harder job for CPT. `train_embeddings`
then helps the most; beyond that, a byte-level model (ByT5) or vocabulary extension are the alternatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Original text (not quoted from anywhere), used only as a stable English baseline.
ENGLISH_REFERENCE = (
    "The old library stood at the edge of the town, and every morning the keeper opened its heavy doors before "
    "the bells rang. Scholars came from distant cities to read the manuscripts kept there, some of them copied "
    "by hand many centuries ago. They compared the readings of different copies, noted where a scribe had "
    "skipped a line or repeated a word, and tried to recover the text as it was first written. It was slow and "
    "patient work, but each small correction brought them a little closer to the voice of the author. In the "
    "evening the reading room emptied, the lamps were put out one by one, and the books waited quietly for the "
    "next day."
)


def measure(tokenizer: Any, texts: Sequence[str], max_chars: int = 200_000) -> dict[str, float]:
    sample, total = [], 0
    for t in texts:
        if total >= max_chars:
            break
        sample.append(t[: max_chars - total])
        total += len(sample[-1])
    words = sum(len(t.split()) for t in sample)
    chars = sum(len("".join(t.split())) for t in sample)  # non-whitespace characters
    tokens = sum(len(ids) for ids in tokenizer(list(sample), add_special_tokens=False)["input_ids"]) if sample else 0
    unk = getattr(tokenizer, "unk_token_id", None)
    n_unk = 0
    if unk is not None and sample:
        n_unk = sum(ids.count(unk) for ids in tokenizer(list(sample), add_special_tokens=False)["input_ids"])
    return {
        "words": words,
        "tokens": tokens,
        "tokens_per_word": tokens / words if words else float("nan"),
        "chars_per_token": chars / tokens if tokens else float("nan"),
        "unk_rate": n_unk / tokens if tokens else 0.0,
    }


def verdict(chars_per_token: float) -> str:
    if chars_per_token >= 3.0:
        return "good"
    if chars_per_token >= 2.0:
        return "fair: consider --train-embeddings"
    return "poor: use --train-embeddings; consider a byte-level model or vocabulary extension"


def tokenizer_report(tokenizer: Any, texts: Sequence[str]) -> dict[str, Any]:
    target = measure(tokenizer, texts)
    english = measure(tokenizer, [ENGLISH_REFERENCE])
    return {
        "target": target,
        "english": english,
        "tokens_per_word_vs_english": target["tokens_per_word"] / english["tokens_per_word"],
        "verdict": verdict(target["chars_per_token"]),
    }
