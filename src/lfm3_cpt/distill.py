"""Self-distillation: let a model (by default the original LFM3) answer prompts, and save the answers as SFT data.

Mixed into stage 2 with `sft_mix`, these examples remind the model how it answered *before* training: in English
when asked in English, with its own knowledge and its own step-by-step style. That counters the two side
effects of an all-Hebrew (or all-Turkish) SFT set: answering every question in the new language, and answering
too briefly to reason.

Only the prompts of the source dataset are used; its reference answers are discarded (except, with
`verify_number`, to check the final number of maths answers).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .chat_eval import detect_language, generate_answers, in_target_language
from .sft_data import _USER_KEYS, Message, to_messages

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def prompts_from_records(records: Iterable[dict[str, Any]], prompt_field: str | None = None) -> list[dict[str, Any]]:
    """Each usable record -> {"prompt": [messages], "reference": str | None}."""
    out = []
    for rec in records:
        if prompt_field:
            text = rec.get(prompt_field)
            if isinstance(text, str) and text.strip():
                out.append({"prompt": [{"role": "user", "content": text.strip()}], "reference": None})
            continue
        msgs = to_messages(rec)
        if msgs is not None:
            out.append({"prompt": msgs[:-1], "reference": msgs[-1]["content"]})
            continue
        lower = {k.lower(): v for k, v in rec.items()}
        key = next(
            (k for k in ("instruction", *_USER_KEYS) if isinstance(lower.get(k), str) and lower[k].strip()), None
        )
        if key:  # a prompts-only dataset
            out.append({"prompt": [{"role": "user", "content": lower[key].strip()}], "reference": None})
    return out


def prompts_from_text(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [{"prompt": [{"role": "user", "content": ln.strip()}], "reference": None} for ln in lines if ln.strip()]


def last_number(text: str | None) -> float | None:
    if not text:
        return None
    found = _NUMBER.findall(text.split("####")[-1])  # GSM8K puts the final answer after ####
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def language_ok(text: str, language: str) -> bool:
    if language in {"", "any"}:
        return True
    if language == "en":
        return detect_language(text) == "en"
    return bool(in_target_language(text, language))


@dataclass
class DistillStats:
    prompts: int = 0
    skipped_done: int = 0
    kept: int = 0
    unfinished: int = 0
    empty: int = 0
    wrong_language: int = 0
    wrong_number: int = 0

    @property
    def processed(self) -> int:
        """Prompts handled so far, including those of an interrupted run that was resumed."""
        return self.kept + self.unfinished + self.empty + self.wrong_language + self.wrong_number

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def distill(
    model: torch.nn.Module,
    tokenizer: Any,
    items: Sequence[dict[str, Any]],
    output: str | Path,
    template_kwargs: dict[str, Any] | None = None,
    batch_size: int = 32,
    chunk: int = 256,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.1,
    language: str = "en",
    verify_number: bool = False,
    resume: bool = False,
    source_name: str = "",
    progress: Callable[[DistillStats], None] | None = None,
) -> DistillStats:
    """Generate answers in chunks, appending kept examples to `output` ({"messages": [...]} per line).

    A sidecar `<output>.progress.json` records how many prompts were processed, so `resume=True` continues an
    interrupted run instead of starting over.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(f"{output}.progress.json")
    stats = DistillStats(prompts=len(items))
    start = 0
    if resume and progress_path.exists():
        saved = json.loads(progress_path.read_text())
        start = int(saved.get("processed", 0))
        for k, v in saved.get("stats", {}).items():
            if k != "prompts" and hasattr(stats, k):
                setattr(stats, k, v)
        stats.skipped_done = start
    elif output.exists() and not resume:
        output.unlink()
    kwargs = template_kwargs or {}
    with output.open("a", encoding="utf-8") as fh:
        for begin in range(start, len(items), chunk):
            part = items[begin : begin + chunk]
            answers = generate_answers(
                model, tokenizer, [it["prompt"] for it in part], kwargs, max_new_tokens, batch_size, repetition_penalty
            )
            for it, ans in zip(part, answers, strict=True):
                text = ans["text"].strip()
                if not ans["finished"]:
                    stats.unfinished += 1
                elif not text:
                    stats.empty += 1
                elif not language_ok(text, language):
                    stats.wrong_language += 1
                elif verify_number and (
                    last_number(it.get("reference")) is None or last_number(text) != last_number(it.get("reference"))
                ):
                    stats.wrong_number += 1
                else:
                    record = {"messages": [*it["prompt"], {"role": "assistant", "content": text}]}
                    if source_name:
                        record["distilled_by"] = source_name
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.kept += 1
            fh.flush()
            processed = begin + len(part)
            progress_path.write_text(json.dumps({"processed": processed, "stats": stats.to_dict()}))
            if progress:
                progress(stats)
    return stats


def messages_preview(messages: list[Message], n: int = 120) -> str:
    text = messages[-1]["content"].replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"
