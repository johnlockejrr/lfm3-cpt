"""Stage-2 data: Turkish instruction/chat examples as TRL prompt-completion conversations.

Most chat datasets on the Hub use one of a handful of layouts. `to_messages` recognises them, so the stage works
with the default (Aya, human-written Turkish) as well as Alpaca-style, ShareGPT-style or `messages` datasets:

- `messages`:            [{"role": ..., "content": ...}, ...]
- `conversations`:       ShareGPT [{"from": "human"|"gpt"|"system", "value": ...}, ...]
- `inputs` / `targets`:  Aya
- `instruction` [+ `input`] / `output`: Alpaca (and its many translations)
- `prompt` | `question` | `input` / `response` | `completion` | `answer` | `output`

Multi-turn conversations become one example whose completion is the *last* assistant turn and whose prompt is
everything before it, so the loss is only on what the assistant says.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from .config import Config, parse_mix_entry
from .corpus_files import expand, fingerprint
from .textnorm import Normalizer

Message = dict[str, str]

SFT_DATA_KEYS = (
    "sft_dataset",
    "sft_dataset_config",
    "sft_train_split",
    "sft_eval_split",
    "sft_filter_column",
    "sft_filter_value",
    "sft_user_column",
    "sft_assistant_column",
    "sft_max_train",
    "sft_max_eval",
    "sft_eval_holdout",
    "sft_system_prompt",
    "sft_max_length",
    "sft_eval_file",
    "sft_mix",
    "sft_mix_eval",
    "unicode_form",
    "strip_marks",
    "seed",
)

LOCAL_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".ndjson": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
}

_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}
_USER_KEYS = ("prompt", "question", "query", "input", "inputs")
_ASSISTANT_KEYS = ("response", "completion", "answer", "output", "targets", "target")


def _clean(text: Any) -> str:
    return text.strip() if isinstance(text, str) else ""


def to_messages(
    example: Mapping[str, Any], user_col: str | None = None, assistant_col: str | None = None
) -> list[Message] | None:
    """Normalise one example to a list of chat messages, or None if it cannot be used."""
    lower = {k.lower(): v for k, v in example.items()}
    if user_col and assistant_col:
        msgs = [
            {"role": "user", "content": _clean(example.get(user_col))},
            {"role": "assistant", "content": _clean(example.get(assistant_col))},
        ]
    elif isinstance(lower.get("messages"), list):
        msgs = [
            {"role": str(m.get("role", "")).lower(), "content": _clean(m.get("content"))} for m in lower["messages"]
        ]
    elif isinstance(lower.get("conversations"), list):
        msgs = [
            {"role": _SHAREGPT_ROLES.get(str(m.get("from", "")).lower(), ""), "content": _clean(m.get("value"))}
            for m in lower["conversations"]
        ]
    elif "instruction" in lower and "output" in lower:
        user = _clean(lower["instruction"])
        extra = _clean(lower.get("input"))
        if extra:
            user = f"{user}\n\n{extra}"
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": _clean(lower["output"])}]
    else:
        u = next((k for k in _USER_KEYS if k in lower), None)
        a = next((k for k in _ASSISTANT_KEYS if k in lower and k != u), None)
        if u is None or a is None:
            return None
        msgs = [{"role": "user", "content": _clean(lower[u])}, {"role": "assistant", "content": _clean(lower[a])}]

    msgs = [m for m in msgs if m["role"] in {"system", "user", "assistant"}]
    # Need at least one user turn before a final, non-empty assistant turn.
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if len(msgs) < 2 or not msgs[-1]["content"] or not any(m["role"] == "user" and m["content"] for m in msgs[:-1]):
        return None
    return msgs


def to_prompt_completion(messages: list[Message], system_prompt: str = "") -> dict[str, list[Message]]:
    prompt = list(messages[:-1])
    if system_prompt and not any(m["role"] == "system" for m in prompt):
        prompt.insert(0, {"role": "system", "content": system_prompt})
    return {"prompt": prompt, "completion": [messages[-1]]}


def chat_ids(
    tokenizer: Any, messages: list[Message], add_generation_prompt: bool = False, **template_kwargs: Any
) -> list[int]:
    """Token ids exactly as TRL builds them (`apply_chat_template(tokenize=True, return_dict=True)`)."""
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, add_generation_prompt=add_generation_prompt, **template_kwargs
    )
    return list(out["input_ids"])


def template_kwargs(cfg: Config) -> dict[str, Any]:
    return {"enable_thinking": cfg.sft_enable_thinking}


def with_template_kwargs(dataset: Dataset, cfg: Config) -> Dataset:
    """Add the `chat_template_kwargs` column TRL passes to the chat template (non-reasoning mode by default)."""
    kwargs = template_kwargs(cfg)
    return dataset.map(lambda _: {"chat_template_kwargs": kwargs})


# ---------------------------------------------------------------------------------------------------------------


@dataclass
class SftBundle:
    train: Dataset
    eval: Dataset
    metadata: dict[str, Any]


def is_local_dataset(name: str) -> bool:
    return Path(name).expanduser().suffix.lower() in LOCAL_BUILDERS or Path(name).expanduser().exists()


def _load_local(spec: str) -> Dataset:
    paths = [str(s.path) for s in expand([spec]) if s.path.stat().st_size > 0]
    if not paths:
        raise ValueError(f"{spec}: the file is empty (if it came from `distill`, no answers were kept)")
    suffix = Path(paths[0]).suffix.lower()
    builder = LOCAL_BUILDERS.get(suffix)
    if builder is None:
        raise ValueError(f"unsupported SFT file type {suffix!r}; use one of {sorted(LOCAL_BUILDERS)}")
    kwargs = {"delimiter": "\t"} if suffix == ".tsv" else {}
    return load_dataset(builder, data_files=paths, split="train", **kwargs)


def _hub_loader(cfg: Config) -> Mapping[str, Dataset]:
    if is_local_dataset(cfg.sft_dataset):
        splits = {cfg.sft_train_split: _load_local(cfg.sft_dataset)}
        if cfg.sft_eval_file:
            splits["__eval_file__"] = _load_local(cfg.sft_eval_file)
        return splits
    splits = [s for s in (cfg.sft_train_split, cfg.sft_eval_split) if s]
    out = {s: load_dataset(cfg.sft_dataset, cfg.sft_dataset_config, split=s) for s in splits}
    if cfg.sft_eval_file:
        out["__eval_file__"] = _load_local(cfg.sft_eval_file)
    return out


def load_mix_source(source: str) -> Dataset:
    if is_local_dataset(source):
        return _load_local(source)
    name, _, subset = source.partition(":")
    return load_dataset(name, subset or None, split="train")


SftLoader = Callable[[Config], Mapping[str, Dataset]]
MixLoader = Callable[[str], Dataset]


def sft_dir(cfg: Config) -> Path:
    return Path(cfg.data_dir) / "sft"


def sft_signature(cfg: Config) -> dict[str, Any]:
    sig = {k: getattr(cfg, k) for k in SFT_DATA_KEYS}
    mix_sources = [parse_mix_entry(e)[0] for e in cfg.sft_mix]
    local = [x for x in (cfg.sft_dataset, cfg.sft_eval_file, *mix_sources) if x and is_local_dataset(x)]
    if local:
        try:
            sig["files"] = fingerprint(local)
        except FileNotFoundError:
            sig["files"] = None
    return sig


def is_sft_prepared(cfg: Config) -> bool:
    root = sft_dir(cfg)
    meta = root / "metadata.json"
    if not meta.exists() or not (root / "train").exists() or not (root / "eval").exists():
        return False
    return json.loads(meta.read_text()).get("signature") == sft_signature(cfg)


def _row_key(row: Mapping[str, Any]) -> str:
    msgs = list(row["prompt"]) + list(row["completion"])
    return json.dumps([[m["role"], m["content"]] for m in msgs], ensure_ascii=False)


def _convert(
    ds: Dataset,
    cfg: Config,
    tokenizer: Any,
    stats: dict[str, int],
    log: Callable[[str], None] = print,
    apply_filter: bool = True,
) -> list[dict[str, Any]]:
    norm = Normalizer.from_config(cfg)
    if not apply_filter:
        pass
    elif cfg.sft_filter_column and cfg.sft_filter_column not in ds.column_names and is_local_dataset(cfg.sft_dataset):
        log(f"[dim]No {cfg.sft_filter_column!r} column in the local SFT file: language filter skipped[/dim]")
    elif cfg.sft_filter_column:
        if cfg.sft_filter_column not in ds.column_names:
            raise ValueError(f"filter column {cfg.sft_filter_column!r} not in {ds.column_names}")
        before = len(ds)
        ds = ds.filter(lambda e: str(e[cfg.sft_filter_column]) == cfg.sft_filter_value)
        stats["filtered_out"] += before - len(ds)
    rows, seen = [], set()
    kwargs = template_kwargs(cfg)
    for example in ds:
        msgs = to_messages(example, cfg.sft_user_column, cfg.sft_assistant_column)
        if msgs is not None and not norm.is_identity:
            msgs = [{"role": m["role"], "content": norm(m["content"])} for m in msgs]
        if msgs is None:
            stats["unusable"] += 1
            continue
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        row = to_prompt_completion(msgs, cfg.sft_system_prompt)
        if (
            tokenizer is not None
            and len(chat_ids(tokenizer, row["prompt"] + row["completion"], **kwargs)) > cfg.sft_max_length
        ):
            stats["too_long"] += 1  # dropped: truncating would cut the answer the model learns from
            continue
        rows.append(row)
    return rows


def prepare_sft_data(
    cfg: Config,
    tokenizer: Any | None = None,
    force: bool = False,
    loader: SftLoader | None = None,
    log: Callable[[str], None] = print,
    mix_loader: MixLoader | None = None,
) -> SftBundle:
    if not force and is_sft_prepared(cfg):
        log(f"[dim]Using cached SFT data in {sft_dir(cfg)}[/dim]")
        return load_sft_data(cfg)
    t0 = time.time()
    log(f"Loading {cfg.sft_dataset}" + (f" [{cfg.sft_dataset_config}]" if cfg.sft_dataset_config else "") + " ...")
    splits = (loader or _hub_loader)(cfg)
    stats = {"filtered_out": 0, "unusable": 0, "duplicates": 0, "too_long": 0}
    train_rows = _convert(splits[cfg.sft_train_split], cfg, tokenizer, stats, log)
    train = Dataset.from_list(train_rows).shuffle(seed=cfg.seed) if train_rows else None
    if train is None:
        raise RuntimeError(f"no usable SFT examples in {cfg.sft_dataset}:{cfg.sft_train_split} ({stats})")
    if "__eval_file__" in splits:
        eval_ds = Dataset.from_list(_convert(splits["__eval_file__"], cfg, tokenizer, stats, log))
    elif cfg.sft_eval_split and cfg.sft_eval_split in splits:
        eval_ds = Dataset.from_list(_convert(splits[cfg.sft_eval_split], cfg, tokenizer, stats, log))
    else:
        if cfg.sft_eval_split and not is_local_dataset(cfg.sft_dataset):
            raise ValueError(f"split {cfg.sft_eval_split!r} not loaded")
        n_eval = max(1, int(len(train) * (cfg.sft_eval_holdout or 0.05)))
        if cfg.sft_max_eval > 0:
            n_eval = min(n_eval, cfg.sft_max_eval)  # rows beyond the eval cap stay in training
        eval_ds, train = train.select(range(n_eval)), train.select(range(n_eval, len(train)))
    # Never evaluate on a training example (a separate test split could overlap with train).
    train_keys = {_row_key(r) for r in train}
    before = len(eval_ds)
    eval_ds = eval_ds.filter(lambda r: _row_key(r) not in train_keys)
    stats["eval_overlap_removed"] = before - len(eval_ds)
    if cfg.sft_max_train > 0:
        train = train.select(range(min(cfg.sft_max_train, len(train))))
    if cfg.sft_max_eval > 0:
        eval_ds = eval_ds.select(range(min(cfg.sft_max_eval, len(eval_ds))))
    if len(eval_ds) == 0:
        raise RuntimeError("the SFT eval set is empty; set sft_eval_split='' to hold out part of train")
    train = train.add_column("source", ["main"] * len(train))
    eval_ds = eval_ds.add_column("source", ["main"] * len(eval_ds))
    if cfg.sft_mix:
        train, eval_ds, mix_stats = _add_mix(cfg, tokenizer, train, eval_ds, mix_loader or load_mix_source, log)
        stats["mix"] = mix_stats

    root = sft_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(root / "train"))
    eval_ds.save_to_disk(str(root / "eval"))
    metadata = {
        "signature": sft_signature(cfg),
        "stats": {"train_examples": len(train), "eval_examples": len(eval_ds), **stats},
        "seconds": round(time.time() - t0, 1),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return SftBundle(train, eval_ds, metadata)


def _add_mix(
    cfg: Config,
    tokenizer: Any,
    train: Dataset,
    eval_ds: Dataset,
    mix_loader: MixLoader,
    log: Callable[[str], None],
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Mix extra sources into train (as shares or repeats) and hold out `sft_mix_eval` of each for evaluation."""
    import random

    rng = random.Random(cfg.seed)
    n_main = len(train)
    entries = [parse_mix_entry(e) for e in cfg.sft_mix]
    share_total = sum(w for _, w in entries if w is not None and w < 1)
    seen = {_row_key(r) for r in train} | {_row_key(r) for r in eval_ds}
    train_parts, eval_parts, out_stats = [train], [eval_ds], {}
    for source, weight in entries:
        log(f"Mixing in {source}" + (f" @{weight:g}" if weight is not None else "") + " ...")
        st = {"filtered_out": 0, "unusable": 0, "duplicates": 0, "too_long": 0}
        rows = [
            r
            for r in _convert(mix_loader(source), cfg, tokenizer, st, log, apply_filter=False)
            if _row_key(r) not in seen
        ]
        rng.shuffle(rows)
        n_eval = min(cfg.sft_mix_eval, max(0, len(rows) - 1))
        held, pool = rows[:n_eval], rows[n_eval:]
        if not pool:
            raise RuntimeError(f"no usable examples in sft_mix source {source!r} ({st})")
        if weight is None:
            chosen = pool
        elif weight >= 1:
            chosen = pool * int(weight)
        else:
            want = round(weight * n_main / (1 - share_total))
            if want > len(pool):
                log(f"[yellow]{source}: wanted {want} examples for @{weight:g}, only {len(pool)} available[/yellow]")
            chosen = rng.sample(pool, min(want, len(pool)))
        for r in chosen + held:
            r["source"] = source
        if chosen:
            train_parts.append(Dataset.from_list(chosen))
        if held:
            eval_parts.append(Dataset.from_list(held))
        out_stats[source] = {"train": len(chosen), "eval": len(held), "available": len(rows), **st}
    from datasets import concatenate_datasets

    train = concatenate_datasets(train_parts).shuffle(seed=cfg.seed)
    eval_ds = concatenate_datasets(eval_parts).shuffle(seed=cfg.seed)  # generation samples cover every source
    return train, eval_ds, out_stats


def load_sft_data(cfg: Config) -> SftBundle:
    root = sft_dir(cfg)
    return SftBundle(
        load_from_disk(str(root / "train")),
        load_from_disk(str(root / "eval")),
        json.loads((root / "metadata.json").read_text()),
    )
