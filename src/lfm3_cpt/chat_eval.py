"""Stage-2 evaluation: how well the model *answers* in Turkish.

- answer perplexity: NLL of the reference answer given the conversation, on the answer tokens only (the SFT loss,
  token-weighted over the whole eval set)
- target-language rate: share of generated answers in `target_language` (catches drifting into English);
  Turkish and Latin by stopwords, Hebrew/Aramaic, Arabic, Syriac and Samaritan by script
- finished rate: share of answers that end with EOS before the token limit (catches rambling / loops)
- chrF vs the reference answer: a rough, reference-based similarity; free-form answers can be good and still
  score low, so read it as a trend, not a grade
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .chatui import final_answer
from .evaluation import _input_device, eval_mode
from .sft_data import Message, chat_ids

ProgressFn = Callable[[int], None]

# ---------------------------------------------------------------------------------------------------------------
# Language detection (heuristic, dependency-free; good enough to tell Turkish from English answers)
# ---------------------------------------------------------------------------------------------------------------

TR_WORDS = frozenset(
    [
        "ve",
        "bir",
        "bu",
        "da",
        "de",
        "için",
        "ile",
        "çok",
        "daha",
        "gibi",
        "olarak",
        "olan",
        "ne",
        "ama",
        "en",
        "mi",
        "mı",
        "mu",
        "mü",
        "ki",
        "her",
        "değil",
        "var",
        "yok",
        "kadar",
        "sonra",
        "şey",
        "ise",
        "veya",
        "ya",
        "şu",
        "ben",
        "sen",
        "biz",
        "siz",
        "onlar",
        "nasıl",
        "neden",
        "çünkü",
        "eğer",
        "ancak",
        "hem",
        "göre",
        "üzere",
        "olduğu",
        "olur",
        "bana",
        "size",
        "onun",
        "bunu",
        "şunu",
        "tarafından",
        "ayrıca",
        "yani",
        "böyle",
        "sadece",
        "bile",
        "önce",
        "artık",
        "hiç",
        "hangi",
        "nerede",
        "zaman",
        "olabilir",
    ]
)
EN_WORDS = frozenset(
    [
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "that",
        "for",
        "it",
        "with",
        "as",
        "was",
        "on",
        "are",
        "be",
        "this",
        "by",
        "you",
        "or",
        "an",
        "not",
        "from",
        "at",
        "which",
        "have",
        "can",
        "will",
        "your",
        "they",
        "what",
        "how",
        "there",
        "their",
        "would",
        "about",
        "also",
        "more",
        "were",
        "has",
        "been",
        "these",
        "than",
        "when",
        "who",
        "into",
        "such",
    ]
)
_TR_LETTERS = set("ğışĞŞİ")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> str:
    """Return "tr", "en", "other" or "empty"."""
    special = sum(ch in _TR_LETTERS for ch in text)
    words = _WORD.findall(text.replace("İ", "i").replace("I", "ı").lower())
    if not words:
        return "empty"
    tr = sum(w in TR_WORDS for w in words) + special / 3
    en = sum(w in EN_WORDS for w in words)
    if tr == 0 and en == 0:
        return "other"
    return "tr" if tr > en else "en"


LA_WORDS = frozenset(
    [
        "et",
        "in",
        "est",
        "non",
        "ad",
        "cum",
        "ut",
        "quod",
        "sed",
        "qui",
        "quae",
        "quam",
        "de",
        "ex",
        "si",
        "enim",
        "esse",
        "sunt",
        "autem",
        "per",
        "nec",
        "atque",
        "etiam",
        "eius",
        "hoc",
        "haec",
        "ab",
        "neque",
        "tamen",
        "ita",
        "quoque",
        "iam",
        "inter",
        "post",
        "ubi",
        "ille",
        "illa",
        "ipse",
        "nos",
        "vos",
        "sic",
        "nunc",
        "erat",
        "fuit",
    ]
)
# Letters of each script (Hebrew-script Aramaic is checked like Hebrew).
_SCRIPTS = {
    "he": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "ar": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "syc": ((0x0700, 0x074F),),
    "sam": ((0x0800, 0x083F),),
}
_SCRIPTS["arc"] = _SCRIPTS["he"]


def script_share(text: str, ranges: tuple[tuple[int, int], ...]) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in letters) / len(letters)


def in_target_language(text: str, target: str) -> bool | None:
    """Whether `text` is in the target language; None when there is no check for it."""
    if not target:
        return None
    if target == "tr":
        return detect_language(text) == "tr"
    if target == "la":
        words = _WORD.findall(text.lower())
        la, en = sum(w in LA_WORDS for w in words), sum(w in EN_WORDS for w in words)
        return la > 0 and la >= en
    if target in _SCRIPTS:
        return script_share(text, _SCRIPTS[target]) >= 0.5
    return None


def answer_language(text: str, target: str) -> str:
    """Label for one answer: the target code if it is in the target language, else en / other / empty / tr."""
    if not text.strip():
        return "empty"
    if in_target_language(text, target):
        return target
    lang = detect_language(text)
    return lang if lang != "tr" or target != "tr" else "other"


# ---------------------------------------------------------------------------------------------------------------
# chrF (character n-gram F-score, Popović 2015), sentence level
# ---------------------------------------------------------------------------------------------------------------


def chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    hyp, ref = "".join(hypothesis.split()), "".join(reference.split())
    if not hyp or not ref:
        return 100.0 if hyp == ref else 0.0
    precisions, recalls = [], []
    for order in range(1, n + 1):
        h = Counter(hyp[i : i + order] for i in range(len(hyp) - order + 1))
        r = Counter(ref[i : i + order] for i in range(len(ref) - order + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


# ---------------------------------------------------------------------------------------------------------------
# Answer perplexity
# ---------------------------------------------------------------------------------------------------------------


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):  # prompt and full sequence differ in length
        if x != y:
            break
        n += 1
    return n


def answer_spans(tokenizer: Any, rows: Sequence[dict[str, Any]], kwargs: dict[str, Any]) -> list[tuple[list[int], int]]:
    """(full ids, index where the answer starts), built exactly like TRL's prompt-completion tokenization."""
    out = []
    for row in rows:
        prompt_ids = chat_ids(tokenizer, list(row["prompt"]), add_generation_prompt=True, **kwargs)
        full = chat_ids(tokenizer, list(row["prompt"]) + list(row["completion"]), **kwargs)
        out.append((full, common_prefix_length(prompt_ids, full)))
    return out


@dataclass
class AnswerPerplexity:
    loss: float
    perplexity: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def answer_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    kwargs: dict[str, Any],
    batch_size: int = 4,
    progress: ProgressFn | None = None,
) -> AnswerPerplexity:
    spans = answer_spans(tokenizer, rows, kwargs)
    order = sorted(range(len(spans)), key=lambda i: len(spans[i][0]), reverse=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = _input_device(model)
    nll, count = 0.0, 0
    with eval_mode(model):
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(spans[i][0]) for i in idx)
            ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for row, i in enumerate(idx):
                full, begin = spans[i]
                ids[row, : len(full)] = torch.tensor(full)
                mask[row, : len(full)] = 1  # right padding: safe for LFM3's causal convolutions
                labels[row, begin : len(full)] = torch.tensor(full[begin:])
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            shift_labels = labels[:, 1:].to(logits.device)
            for r in range(len(idx)):
                nll += F.cross_entropy(
                    logits[r, :-1].float(), shift_labels[r], ignore_index=-100, reduction="sum"
                ).item()
                count += int((shift_labels[r] != -100).sum())
            if progress:
                progress(len(idx))
    loss = nll / max(count, 1)
    return AnswerPerplexity(loss, math.exp(min(loss, 50.0)), count)


# ---------------------------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------------------------


def _eos_ids(model: torch.nn.Module, tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    gen = getattr(model, "generation_config", None)
    eos = getattr(gen, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


@torch.no_grad()
def generate_answers(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[list[Message]],
    kwargs: dict[str, Any],
    max_new_tokens: int = 512,
    batch_size: int = 16,
    repetition_penalty: float = 1.1,
    progress: ProgressFn | None = None,
    no_repeat_ngram: int = 0,
) -> list[dict[str, Any]]:
    """Greedy answers. Prompts are left-padded by hand, so the shared tokenizer's settings are never changed."""
    encoded = [chat_ids(tokenizer, list(p), add_generation_prompt=True, **kwargs) for p in prompts]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = _eos_ids(model, tokenizer)
    device = _input_device(model)
    results: list[dict[str, Any]] = [{}] * len(encoded)
    with eval_mode(model):
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, i in enumerate(idx):
                ids[row, width - len(encoded[i]) :] = torch.tensor(encoded[i])
                mask[row, width - len(encoded[i]) :] = 1
            out = model.generate(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_id,
                **({"no_repeat_ngram_size": no_repeat_ngram} if no_repeat_ngram > 0 else {}),
            )
            for row, i in enumerate(idx):
                new = out[row, width:].tolist()
                finished = any(t in eos for t in new)
                if finished:
                    new = new[: next(k for k, t in enumerate(new) if t in eos)]
                text = final_answer(tokenizer.decode(new, skip_special_tokens=True))  # drop any reasoning
                results[i] = {"text": text, "finished": finished, "tokens": len(new)}
            if progress:
                progress(len(idx))
    return results


def summarize_generations(
    answers: Sequence[dict[str, Any]],
    references: Sequence[str],
    target: str = "tr",
    prompts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generation metrics. With `prompts`, the target-language rate counts only questions asked in the target
    language, and `language_match` is the share of answers in the same language as their question (target or
    English), which catches a model that answers English questions in the new language."""
    n = len(answers)
    langs = [answer_language(a["text"], target) for a in answers]
    out: dict[str, Any] = {"n": n, "target_language": target}
    if target:
        if prompts is not None:
            plangs = [answer_language(p, target) for p in prompts]
            in_target = [a for a, p in zip(langs, plangs, strict=True) if p == target]
            out["target_rate"] = in_target.count(target) / len(in_target) if in_target else float("nan")
            scored = [(a, p) for a, p in zip(langs, plangs, strict=True) if p in {target, "en"}]
            out["language_match"] = sum(a == p for a, p in scored) / len(scored) if scored else float("nan")
            out["prompt_languages"] = plangs
        else:
            out["target_rate"] = langs.count(target) / n if n else float("nan")
    if target == "tr":
        out["turkish_rate"] = out["target_rate"]  # name used by earlier versions
    return {
        **out,
        "english_rate": langs.count("en") / n if n else float("nan"),
        "finished_rate": sum(a["finished"] for a in answers) / n if n else float("nan"),
        "empty_rate": langs.count("empty") / n if n else float("nan"),
        "avg_words": sum(len(a["text"].split()) for a in answers) / n if n else 0.0,
        "chrf": sum(chrf(a["text"], r) for a, r in zip(answers, references, strict=True)) / n if n else float("nan"),
        "languages": langs,
    }
