"""Local corpora: train/eval/ood text from your own files, and minimal pairs from TSV/JSONL.

File specs are glob patterns, optionally with a repeat factor: `corpora/targum/*.txt@3` reads those files three
times (upsampling), which is how you weight a small target corpus against larger related-language corpora.

Documents:
- `.jsonl` / `.json` (JSON lines): one document per record, taken from `text_field`
- anything else is plain UTF-8 text, split according to `doc_split`:
  `file` (one document per file), `paragraph` (blank-line separated), `line` (one per non-empty line)
"""

from __future__ import annotations

import glob
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset

from .config import Config
from .textnorm import Normalizer

_REPEAT = re.compile(r"^(?P<pattern>.+)@(?P<n>\d+)$")
JSON_SUFFIXES = {".jsonl", ".json", ".ndjson"}


@dataclass
class FileSpec:
    path: Path
    repeat: int = 1


def expand(specs: list[str]) -> list[FileSpec]:
    """Expand glob patterns (with optional @N repeat) into files, in a stable order. Missing patterns fail loudly."""
    out: list[FileSpec] = []
    for spec in specs:
        m = _REPEAT.match(spec)
        pattern, repeat = (m["pattern"], int(m["n"])) if m else (spec, 1)
        matches = sorted(Path(p) for p in glob.glob(str(Path(pattern).expanduser()), recursive=True))
        matches = [p for p in matches if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"no files match {pattern!r}")
        out.extend(FileSpec(p, repeat) for p in matches)
    return out


def fingerprint(specs: list[str]) -> list[list[Any]]:
    """Path, size and mtime of every file: the cache is rebuilt when a corpus file changes."""
    if not specs:
        return []
    return [[str(s.path.resolve()), s.repeat, s.path.stat().st_size, s.path.stat().st_mtime_ns] for s in expand(specs)]


def read_documents(path: Path, cfg: Config) -> list[str]:
    fmt = cfg.file_format
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() in JSON_SUFFIXES else "text"
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 ({exc})") from exc
    if fmt == "jsonl":
        docs = []
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
            text = record.get(cfg.text_field) if isinstance(record, dict) else None
            if isinstance(text, str):
                docs.append(text)
        return docs
    if cfg.doc_split == "file":
        return [raw]
    if cfg.doc_split == "paragraph":
        return re.split(r"\n\s*\n", raw)
    return raw.splitlines()


@dataclass
class LocalCorpus:
    train: list[str]
    eval: list[str]
    ood: list[str]
    info: dict[str, Any] = field(default_factory=dict)


def _docs(specs: list[FileSpec], cfg: Config, norm: Normalizer) -> list[str]:
    docs = []
    for spec in specs:
        texts = [norm(t).strip() for t in read_documents(spec.path, cfg)]
        texts = [t for t in texts if t and len(t) >= cfg.file_min_chars]
        docs.extend(texts * spec.repeat)
    return docs


def load_local_corpus(cfg: Config, log: Callable[[str], None] = print) -> LocalCorpus:
    norm = Normalizer.from_config(cfg)
    train_specs = expand(cfg.train_files)
    info: dict[str, Any] = {"train_files": len(train_specs)}
    rng = random.Random(cfg.seed)

    if cfg.eval_files:
        eval_docs = _docs(expand(cfg.eval_files), cfg, norm)
        train_docs = _docs(train_specs, cfg, norm)
        info["eval_source"] = "eval_files"
    elif cfg.eval_holdout == "files" and len({s.path for s in train_specs}) >= 2:
        # Hold out whole files (books, chapters): with small corpora and several epochs, a held-out *document*
        # from the same book still shares names and formulae with training text and flatters perplexity.
        unique = sorted({s.path for s in train_specs})
        n_eval = max(1, round(len(unique) * cfg.eval_fraction))
        held = set(rng.sample(unique, n_eval))
        eval_docs = _docs([FileSpec(p) for p in sorted(held)], cfg, norm)
        train_docs = _docs([s for s in train_specs if s.path not in held], cfg, norm)
        info["eval_source"] = "held-out files"
        info["eval_files"] = [str(p) for p in sorted(held)]
    else:
        if cfg.eval_holdout == "files":
            log("[yellow]Only one training file: holding out documents instead of whole files[/yellow]")
        # Split *before* upsampling, so a repeated document can never be both in train and eval.
        unique_paths = list(dict.fromkeys(s.path for s in train_specs))
        base = _docs([FileSpec(p) for p in unique_paths], cfg, norm)
        order = list(range(len(base)))
        rng.shuffle(order)
        n_eval = max(1, round(len(base) * cfg.eval_fraction))
        eval_idx = set(order[:n_eval])
        eval_docs = [base[i] for i in sorted(eval_idx)]
        eval_set = set(eval_docs)
        train_docs = [d for d in _docs(train_specs, cfg, norm) if d not in eval_set]
        info["eval_source"] = "held-out documents"

    rng.shuffle(train_docs)
    ood_docs = _docs(expand(cfg.ood_files), cfg, norm) if cfg.ood_files else []
    if not train_docs:
        raise RuntimeError("no training documents left (check train_files, file_min_chars and the holdout)")
    if not eval_docs:
        raise RuntimeError("no evaluation documents (add eval_files or more training files)")
    return LocalCorpus(train_docs, eval_docs, ood_docs, info)


# ---------------------------------------------------------------------------------------------------------------
# Minimal pairs from files
# ---------------------------------------------------------------------------------------------------------------


def read_pairs(path: Path, norm: Normalizer) -> list[dict[str, str]]:
    """TSV: `good<TAB>bad[<TAB>phenomenon]` (lines starting with # are comments, an optional header is skipped).
    JSONL: {"sentence_good"|"good": ..., "sentence_bad"|"bad": ..., "phenomenon"|"linguistic_phenomenon": ...}.
    Without a phenomenon, the file name is used.
    """
    default = path.stem
    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    is_json = path.suffix.lower() in JSON_SUFFIXES
    header_checked = False
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if is_json:
            rec = json.loads(line)
            good = rec.get("sentence_good", rec.get("good"))
            bad = rec.get("sentence_bad", rec.get("bad"))
            phen = rec.get("linguistic_phenomenon", rec.get("phenomenon")) or default
        else:
            cols = line.split("\t")
            if not rows and not header_checked:
                header_checked = True  # an optional header, possibly after comment lines
                if cols[0].strip().lower() in {"sentence_good", "good"}:
                    continue
            if len(cols) < 2:
                raise ValueError(f"{path}:{n}: expected good<TAB>bad[<TAB>phenomenon]")
            good, bad = cols[0], cols[1]
            phen = cols[2].strip() if len(cols) > 2 and cols[2].strip() else default
        if not good or not bad:
            raise ValueError(f"{path}:{n}: empty sentence")
        good, bad = norm(good).strip(), norm(bad).strip()
        if good == bad:
            raise ValueError(f"{path}:{n}: the two sentences are identical after normalization")
        rows.append({"sentence_good": good, "sentence_bad": bad, "linguistic_phenomenon": phen})
    return rows


def load_pair_files(cfg: Config) -> Dataset:
    norm = Normalizer.from_config(cfg)
    rows = [r for spec in expand(cfg.grammar_files) for r in read_pairs(spec.path, norm)]
    if not rows:
        raise RuntimeError("the grammar files contain no pairs")
    return Dataset.from_list(rows)
