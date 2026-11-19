"""Build a local training corpus from Hub datasets (or raw downloaded files), with a size budget.

Stage 1 reads local files (`train_files`), so a large, mixed corpus is assembled once with `lfm3-cpt export`:
one source at a time, each into `<prefix>.train.jsonl` and `<prefix>.eval.jsonl` (disjoint, the eval documents
come first), with one JSON record `{"text": ...}` per document. Weights between sources are then set in the
config with `pattern@N`, and every source contributes held-out documents to evaluation.

Schemas differ between datasets, so nothing is assumed: `--inspect` prints the structure of the first records,
and `--field` names the text (a dotted path; a list of strings or of records is joined, e.g. the sentences of a
parliamentary protocol).
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def get_path(record: Any, path: str) -> Any:
    value = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def extract_text(
    record: Record, text_field: str, item_field: str | None = None, join: str = "\n", title_field: str | None = None
) -> str | None:
    """The document text of one record; lists are joined (items may be strings or records with `item_field`)."""
    value = get_path(record, text_field)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                item = get_path(item, item_field) if item_field else None
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
        value = join.join(parts) if parts else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if title_field:
        title = get_path(record, title_field)
        if isinstance(title, str) and title.strip():
            text = f"{title.strip()}\n\n{text}"
    return text


def describe(value: Any, depth: int = 0, max_str: int = 80) -> list[str]:
    """Human-readable structure of a record, for --inspect."""
    pad = "  " * depth
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                kind = f"list[{len(v)}]" if isinstance(v, list) else "record"
                lines.append(f"{pad}{k}: {kind}")
                if isinstance(v, dict):
                    lines.extend(describe(v, depth + 1, max_str))
                elif v:
                    lines.append(
                        f"{pad}  [0]:" + ("" if isinstance(v[0], (dict, list)) else f" {_short(v[0], max_str)}")
                    )
                    if isinstance(v[0], (dict, list)):
                        lines.extend(describe(v[0], depth + 2, max_str))
            else:
                lines.append(f"{pad}{k}: {type(v).__name__} = {_short(v, max_str)}")
    return lines


def _short(v: Any, n: int) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Source:
    dataset: str | None = None  # Hub dataset id
    config: str | None = None
    split: str = "train"
    data_files: list[str] = field(default_factory=list)  # local paths/globs or hf:// URLs instead of a dataset id
    builder: str | None = None  # json | csv | parquet | text (for data_files; guessed from the extension)

    def describe(self) -> str:
        if self.dataset:
            return self.dataset + (f" [{self.config}]" if self.config else "") + f" ({self.split})"
        return ", ".join(self.data_files)


BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".ndjson": "json",
    ".gz": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
}


def stream_records(source: Source, shuffle_buffer: int = 0, seed: int = 42) -> Iterable[Record]:
    from datasets import load_dataset

    if source.dataset:
        ds = load_dataset(source.dataset, source.config, split=source.split, streaming=True)
    else:
        if not source.data_files:
            raise ValueError("give a dataset id or data files")
        builder = source.builder or BUILDERS.get(Path(source.data_files[0].split("::")[0]).suffix.lower(), "json")
        kwargs = {"delimiter": "\t"} if source.data_files[0].endswith(".tsv") else {}
        ds = load_dataset(builder, data_files=source.data_files, split="train", streaming=True, **kwargs)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


# ---------------------------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class ExportStats:
    records_seen: int = 0
    skipped_empty: int = 0
    skipped_short: int = 0
    eval_docs: int = 0
    eval_chars: int = 0
    train_docs: int = 0
    train_chars: int = 0
    exhausted: bool = False
    files: dict[str, str] = field(default_factory=dict)


def export_corpus(
    records: Iterable[Record],
    prefix: str | Path,
    text_field: str = "text",
    item_field: str | None = None,
    join: str = "\n",
    title_field: str | None = None,
    min_chars: int = 0,
    eval_docs: int = 0,
    max_docs: int | None = None,
    max_chars: int | None = None,
    skip: int = 0,
    progress: Callable[[ExportStats], None] | None = None,
) -> ExportStats:
    """Write `<prefix>.eval.jsonl` (the first `eval_docs` usable documents) and `<prefix>.train.jsonl` (the rest,
    up to `max_docs` documents or `max_chars` characters, whichever comes first). `max_docs=0` writes only the
    eval file (e.g. English text to measure forgetting)."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = Path(f"{prefix}.train.jsonl"), Path(f"{prefix}.eval.jsonl")
    stats = ExportStats(files={"train": str(train_path), "eval": str(eval_path) if eval_docs else ""})
    it: Iterator[Record] = iter(records)
    with (
        train_path.open("w", encoding="utf-8") as train_fh,
        eval_path.open("w", encoding="utf-8") if eval_docs else io.StringIO() as eval_fh,
    ):
        for record in it:
            stats.records_seen += 1
            if stats.records_seen <= skip:
                continue
            text = extract_text(record, text_field, item_field, join, title_field)
            if text is None:
                stats.skipped_empty += 1
                continue
            if len(text) < min_chars:
                stats.skipped_short += 1
                continue
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            if stats.eval_docs < eval_docs:
                eval_fh.write(line)
                stats.eval_docs += 1
                stats.eval_chars += len(text)
                if max_docs == 0 and stats.eval_docs >= eval_docs:  # eval-only export
                    break
            elif max_docs == 0:
                break
            else:
                train_fh.write(line)
                stats.train_docs += 1
                stats.train_chars += len(text)
            if progress and stats.records_seen % 100 == 0:
                progress(stats)
            if (max_docs and stats.train_docs >= max_docs) or (max_chars and stats.train_chars >= max_chars):
                break
        else:
            stats.exhausted = True
    if not eval_docs and eval_path.exists():
        eval_path.unlink()
    if stats.train_docs == 0 and train_path.exists():
        train_path.unlink()  # e.g. an eval-only export (--max-docs 0): no empty file to match *.train.jsonl
        stats.files["train"] = ""
    if progress:
        progress(stats)
    return stats


# ---------------------------------------------------------------------------------------------------------------
# MultiBLiMP minimal pairs (subject-verb agreement, 101 languages; CC-BY-4.0)
# ---------------------------------------------------------------------------------------------------------------

MULTIBLIMP_REPO = "jumelet/multiblimp"
_PHENOMENON_COLUMNS = ("phenomenon", "grammatical_feature", "feature")


def multiblimp_rows(tsv_text: str) -> list[dict[str, str]]:
    """Convert MultiBLiMP's per-language TSV (`sen` = grammatical, `wrong_sen` = ungrammatical) to our pair format."""
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    if not reader.fieldnames or "sen" not in reader.fieldnames or "wrong_sen" not in reader.fieldnames:
        raise ValueError(f"not a MultiBLiMP file: columns are {reader.fieldnames}")
    phen_col = next((c for c in _PHENOMENON_COLUMNS if c in reader.fieldnames), None)
    rows = []
    for r in reader:
        good, bad = (r.get("sen") or "").strip(), (r.get("wrong_sen") or "").strip()
        if not good or not bad or good == bad:
            continue
        phen = (r.get(phen_col) or "").strip() if phen_col else ""
        rows.append(
            {
                "sentence_good": good,
                "sentence_bad": bad,
                "phenomenon": f"agreement_{phen}" if phen else "subject_verb_agreement",
            }
        )
    return rows


def download_multiblimp(lang: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MULTIBLIMP_REPO, f"{lang}/data.tsv", repo_type="dataset")
    return Path(path).read_text(encoding="utf-8")


def write_pairs(rows: list[dict[str, str]], out: str | Path, limit: int | None = None, seed: int = 0) -> int:
    rows = list(rows)
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
