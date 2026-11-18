"""Corpus preparation, cached on disk.

Two sources for stage 1:
- the Hub (default): Wikipedia in `wiki_config`'s language, FineWeb-2 web pages as out-of-distribution text;
- your own files (`train_files`, see `corpus_files`), for languages and periods the Hub does not cover.

Grammar pairs come from `grammar_files` if given, else TurBLiMP when the target language is Turkish.
All text goes through the same normalization (`unicode_form`, `strip_marks`).

Texts are stored *without* the end-of-sequence token; `with_eos` adds the tokenizer's EOS at use time, so the
cache does not depend on a particular tokenizer.
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from .config import Config
from .corpus_files import fingerprint, load_local_corpus, load_pair_files
from .textnorm import Normalizer
from .tokcheck import tokenizer_report

SPLITS = ("train", "eval", "ood", "grammar")
METADATA = "metadata.json"

# Keys of the config that define the cached data; a change in any of them invalidates the cache.
DATA_KEYS = (
    "wiki_dataset",
    "wiki_config",
    "ood_dataset",
    "ood_config",
    "ood_split",
    "grammar_dataset",
    "train_articles",
    "eval_articles",
    "ood_pages",
    "min_chars",
    "shuffle_buffer",
    "grammar_pairs_per_phenomenon",
    "phenomena",
    "seed",
    "target_language",
    "unicode_form",
    "strip_marks",
    "train_files",
    "eval_files",
    "ood_files",
    "file_format",
    "text_field",
    "doc_split",
    "file_min_chars",
    "eval_holdout",
    "eval_fraction",
    "grammar_files",
)


@dataclass
class DataBundle:
    train: Dataset
    eval: Dataset
    ood: Dataset
    grammar: Dataset | None
    metadata: dict[str, Any]


def format_article(article: dict[str, Any]) -> str:
    return f"{article['title']}\n\n{article['text']}"


def select_articles(
    articles: Iterable[dict[str, Any]], min_chars: int, n_eval: int, n_train: int
) -> tuple[list[str], list[str]]:
    """Take eval articles first, then train articles, from one shuffled stream, so the two never overlap."""
    kept: Iterator[str] = (format_article(a) for a in articles if len(a["text"]) > min_chars)
    eval_texts = list(itertools.islice(kept, n_eval))
    train_texts = list(itertools.islice(kept, n_train))
    if len(eval_texts) < n_eval or len(train_texts) < n_train:
        raise RuntimeError(
            f"the corpus ran out: got {len(eval_texts)}/{n_eval} eval and {len(train_texts)}/{n_train} train articles "
            f"longer than {min_chars} characters"
        )
    return eval_texts, train_texts


def select_pages(pages: Iterable[dict[str, Any]], n: int) -> list[str]:
    texts = [p["text"] for p in itertools.islice((p for p in pages if p.get("text", "").strip()), n)]
    if len(texts) < n:
        raise RuntimeError(f"only {len(texts)}/{n} out-of-distribution pages available")
    return texts


# ---------------------------------------------------------------------------------------------------------------
# Hub loaders (overridable in tests)
# ---------------------------------------------------------------------------------------------------------------


def _wiki_stream(cfg: Config) -> Iterable[dict[str, Any]]:
    stream = load_dataset(cfg.wiki_dataset, cfg.wiki_config, split="train", streaming=True)
    return stream.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)


def _ood_stream(cfg: Config) -> Iterable[dict[str, Any]]:
    return load_dataset(cfg.ood_dataset, name=cfg.ood_config, split=cfg.ood_split, streaming=True)


def _grammar_pairs(cfg: Config) -> Dataset:
    parts = []
    for phenomenon in cfg.phenomena:
        ds = load_dataset(cfg.grammar_dataset, phenomenon, split="train").shuffle(seed=0)
        ds = ds.select(range(min(cfg.grammar_pairs_per_phenomenon, len(ds))))
        if "linguistic_phenomenon" in ds.column_names:
            ds = ds.remove_columns("linguistic_phenomenon")
        ds = ds.add_column("linguistic_phenomenon", [phenomenon] * len(ds))
        parts.append(ds.select_columns(["sentence_good", "sentence_bad", "linguistic_phenomenon"]))
    return concatenate_datasets(parts)


@dataclass
class Loaders:
    wiki: Callable[[Config], Iterable[dict[str, Any]]] = _wiki_stream
    ood: Callable[[Config], Iterable[dict[str, Any]]] = _ood_stream
    grammar: Callable[[Config], Dataset] = _grammar_pairs


# ---------------------------------------------------------------------------------------------------------------


def data_signature(cfg: Config) -> dict[str, Any]:
    sig = {k: getattr(cfg, k) for k in DATA_KEYS}
    files = cfg.train_files + cfg.eval_files + cfg.ood_files + cfg.grammar_files
    if files:
        try:
            sig["files"] = fingerprint(files)
        except FileNotFoundError:
            sig["files"] = None  # cannot match a cache; prepare_data will raise a clear error
    return sig


def is_prepared(cfg: Config) -> bool:
    root = Path(cfg.data_dir)
    meta = root / METADATA
    if not meta.exists() or not all((root / s).exists() for s in ("train", "eval")):
        return False
    stored = json.loads(meta.read_text())
    sig = data_signature(cfg)
    # Grammar pairs are optional: a cache without them is still valid when grammar is disabled.
    if cfg.grammar_enabled and not (root / "grammar").exists():
        return False
    return stored.get("signature") == sig


def count_tokens(tokenizer: Any, texts: list[str], batch: int = 256) -> int:
    total = 0
    for i in range(0, len(texts), batch):
        total += sum(len(ids) for ids in tokenizer(texts[i : i + batch])["input_ids"])
    return total


def prepare_data(
    cfg: Config,
    tokenizer: Any | None = None,
    force: bool = False,
    loaders: Loaders | None = None,
    log: Callable[[str], None] = print,
) -> DataBundle:
    """Download (streaming) and cache the corpora. Re-uses the cache when the data settings match."""
    if not force and is_prepared(cfg):
        log(f"[dim]Using cached data in {cfg.data_dir}[/dim]")
        return load_data(cfg)

    loaders = loaders or Loaders()
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    extra: dict[str, Any] = {}
    if cfg.local_corpus:
        log(f"Reading local corpus ({len(cfg.train_files)} pattern(s)) ...")
        corpus = load_local_corpus(cfg, log)
        train_texts, eval_texts, ood_texts = corpus.train, corpus.eval, corpus.ood
        extra = corpus.info
    else:
        norm = Normalizer.from_config(cfg)
        log(f"Streaming {cfg.wiki_dataset} [{cfg.wiki_config}] ...")
        eval_texts, train_texts = select_articles(
            loaders.wiki(cfg), cfg.min_chars, cfg.eval_articles, cfg.train_articles
        )
        ood_texts = []
        if cfg.ood_pages:
            log(f"Streaming {cfg.ood_dataset} [{cfg.ood_config}/{cfg.ood_split}] ...")
            ood_texts = select_pages(loaders.ood(cfg), cfg.ood_pages)
        if not norm.is_identity:
            train_texts, eval_texts, ood_texts = ([norm(t) for t in xs] for xs in (train_texts, eval_texts, ood_texts))

    datasets = {
        "train": Dataset.from_dict({"text": train_texts}),
        "eval": Dataset.from_dict({"text": eval_texts}),
        "ood": Dataset.from_dict({"text": ood_texts}),
    }
    if cfg.grammar_enabled:
        if cfg.grammar_files:
            log(f"Reading minimal pairs from {len(cfg.grammar_files)} pattern(s) ...")
            datasets["grammar"] = load_pair_files(cfg)
        else:
            log(f"Loading {cfg.grammar_dataset} ({len(cfg.phenomena)} phenomena) ...")
            datasets["grammar"] = loaders.grammar(cfg)
    elif cfg.run_grammar:
        log(f"[dim]No grammar test for target_language={cfg.target_language!r} (add grammar_files for one)[/dim]")

    # Empty splits are not saved (an empty Arrow dataset cannot be reloaded), and a split from an earlier
    # preparation must not survive (e.g. TurBLiMP after switching language).
    for name in SPLITS:
        if (name not in datasets or len(datasets[name]) == 0) and (root / name).exists():
            shutil.rmtree(root / name)
    for name, ds in datasets.items():
        if len(ds):
            ds.save_to_disk(str(root / name))

    stats: dict[str, Any] = {
        "source": "local files" if cfg.local_corpus else f"{cfg.wiki_dataset}:{cfg.wiki_config}",
        "train_articles": len(train_texts),
        "eval_articles": len(eval_texts),
        "ood_pages": len(ood_texts),
        "grammar_pairs": len(datasets["grammar"]) if "grammar" in datasets else 0,
        "train_chars": sum(map(len, train_texts)),
    }
    if tokenizer is not None:
        stats["train_tokens"] = count_tokens(tokenizer, train_texts)
        stats["eval_tokens"] = count_tokens(tokenizer, eval_texts)
        stats["ood_tokens"] = count_tokens(tokenizer, ood_texts) if ood_texts else 0
        stats["tokenizer"] = getattr(tokenizer, "name_or_path", None)
        report = tokenizer_report(tokenizer, train_texts)
        stats["chars_per_token"] = round(report["target"]["chars_per_token"], 2)
        stats["tokens_per_word"] = round(report["target"]["tokens_per_word"], 2)
        stats["tokens_per_word_vs_english"] = round(report["tokens_per_word_vs_english"], 2)
        stats["tokenizer_verdict"] = report["verdict"]
    metadata = {
        "signature": data_signature(cfg),
        "stats": stats,
        "corpus": extra,
        "seconds": round(time.time() - t0, 1),
    }
    (root / METADATA).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return DataBundle(datasets["train"], datasets["eval"], datasets["ood"], datasets.get("grammar"), metadata)


def load_data(cfg: Config) -> DataBundle:
    root = Path(cfg.data_dir)
    metadata = json.loads((root / METADATA).read_text())
    grammar = load_from_disk(str(root / "grammar")) if (root / "grammar").exists() else None
    return DataBundle(
        train=load_from_disk(str(root / "train")),
        eval=load_from_disk(str(root / "eval")),
        ood=load_from_disk(str(root / "ood")) if (root / "ood").exists() else Dataset.from_dict({"text": []}),
        grammar=grammar,
        metadata=metadata,
    )


def with_eos(dataset: Dataset, eos_token: str) -> Dataset:
    """Append EOS so the model learns where documents end (TRL would add it too; being explicit keeps
    training and our own perplexity evaluation identical)."""

    def add(example: dict[str, str]) -> dict[str, str]:
        text = example["text"]
        return {"text": text if text.endswith(eos_token) else text + eos_token}

    return dataset.map(add, desc=None)
