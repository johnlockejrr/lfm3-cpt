"""Turn raw instruction and parallel datasets into clean chat examples (`lfm3-cpt sft-source`).

* `taco_messages`: TaCo-style translated Alpaca (e.g. saillab/alpaca_maltese_taco), where the instruction is in the
  target language but the output is a chain: "Instruction in English: … Response in English: … Response in
  Maltese: …". Only the last part is the answer; for Maltese, detached elision apostrophes are repaired.
* `parallel_pairs` / `translation_messages`: sentence pairs from a parallel corpus (e.g. MLRS/OPUS-MT-EN-Fixed),
  as translation requests in both directions.

Every example is `{"messages": [{"role": "user", …}, {"role": "assistant", …}], "source": …}`, which the SFT
loader reads directly (sft_dataset / sft_mix).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from typing import Any

# --- TaCo -------------------------------------------------------------------------------------------------------

_RESPONSE_RE = re.compile(r"Response in (?P<lang>[A-Z][a-z]+)\s*:\s*", re.S)
_END_TOKENS_RE = re.compile(r"<\|[a-z_]+\|>|</?s>|###.*$", re.S)
_PLACEHOLDER_RE = re.compile(r"^<[^>]*(?:response|translated|generated)[^>]*>$", re.I)
_EMPTY = {"", "nan", "none", "null", "<noinput>"}


def _text(v: Any) -> str:
    return "" if v is None else str(v).strip()


def taco_answer(output: str, language: str = "Maltese") -> str:
    """The part of a TaCo output after the last 'Response in <language>:'."""
    parts = [m for m in _RESPONSE_RE.finditer(output) if m["lang"] == language]
    if not parts:
        return ""
    answer = output[parts[-1].end() :]
    answer = _END_TOKENS_RE.sub("", answer)
    nxt = _RESPONSE_RE.search(answer)  # a following 'Response in English:' block, if any
    if nxt:
        answer = answer[: nxt.start()]
    answer = re.sub(r"\n\s*\.\s*$", "", answer.strip()).strip()
    return "" if _PLACEHOLDER_RE.match(answer) else answer


# Machine translation detaches the apostrophe of Maltese elision: "ta 'kontroll", "jista 'jinkludi" for
# "ta' kontroll", "jista' jinkludi". The elided words end in -a (ta', ma', jista', erba', terġa'), so only those
# are joined back: "u 'Kif'" (and + an opening quote) stays as it is.
_DETACHED_ELISION_RE = re.compile(r"\b(\w*[aA]) '(?=\w)")


def repair_elision(text: str) -> str:
    return _DETACHED_ELISION_RE.sub(r"\1' ", text)


def taco_messages(rec: Mapping[str, Any], language: str = "Maltese") -> dict[str, Any] | None:
    fix = repair_elision if language == "Maltese" else (lambda t: t)
    instruction = fix(_text(rec.get("instruction")))
    extra = fix(_text(rec.get("input")))
    answer = fix(taco_answer(_text(rec.get("output")), language))
    if not instruction or not answer:
        return None
    user = instruction if extra.lower() in _EMPTY else f"{instruction}\n\n{extra}"
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], "source": "taco"}


# --- parallel corpora ---------------------------------------------------------------------------------------------


def pair_fields(rec: Mapping[str, Any], src: str, tgt: str) -> tuple[str, str] | None:
    """(src text, tgt text) from the usual layouts: {"translation": {src, tgt}}, {src, tgt}, {"src_text"…},
    or {"source", "target"}."""
    tr = rec.get("translation")
    if isinstance(tr, Mapping) and src in tr and tgt in tr:
        return _text(tr[src]), _text(tr[tgt])
    for a, b in ((src, tgt), (f"{src}_text", f"{tgt}_text"), (f"text_{src}", f"text_{tgt}"), ("source", "target")):
        if a in rec and b in rec:
            return _text(rec[a]), _text(rec[b])
    return None


def parallel_pairs(
    records: Iterable[Mapping[str, Any]],
    src: str,
    tgt: str,
    n: int,
    seed: int = 0,
    min_chars: int = 30,
    max_chars: int = 400,
    pool: int = 20,
) -> list[tuple[str, str]]:
    """Up to `n` clean pairs: both sides of a reasonable length, not copies of each other, mostly letters,
    no duplicates. Reads at most `pool`·n records and samples among the good ones."""
    rng = random.Random(seed)
    good: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if i >= pool * n:
            break
        pair = pair_fields(rec, src, tgt)
        if pair is None:
            if i == 0:
                raise ValueError(f"no {src!r}/{tgt!r} fields in the records (keys: {sorted(rec)})")
            continue
        a, b = (" ".join(x.split()) for x in pair)
        if not (min_chars <= len(a) <= max_chars and min_chars <= len(b) <= max_chars):
            continue
        if a.lower() == b.lower() or a.lower() in seen:
            continue
        letters = sum(c.isalpha() for c in a + b)
        if letters < 0.6 * len(a + b):
            continue
        seen.add(a.lower())
        good.append((a, b))
    rng.shuffle(good)
    return good[:n]


PROMPTS = {
    # request in the source language, answer in the target language
    ("en", "mt"): "Translate into Maltese:\n\n{text}",
    ("mt", "en"): "Ittraduċi għall-Ingliż:\n\n{text}",
}


def translation_messages(
    pairs: Iterable[tuple[str, str]], src: str, tgt: str, template: str | None = None
) -> list[dict]:
    template = template or PROMPTS.get((src, tgt)) or f"Translate from {src} into {tgt}:\n\n{{text}}"
    return [
        {
            "messages": [
                {"role": "user", "content": template.format(text=a)},
                {"role": "assistant", "content": b},
            ],
            "source": f"{src}2{tgt}",
        }
        for a, b in pairs
    ]
