import json

import pytest

from lfm3_cpt.data import is_prepared, load_data, prepare_data, select_articles, select_pages, with_eos

from .conftest import make_wiki


def test_select_articles_disjoint_and_filtered():
    eval_texts, train_texts = select_articles(make_wiki(), min_chars=100, n_eval=5, n_train=20)
    assert len(eval_texts) == 5 and len(train_texts) == 20
    assert not set(eval_texts) & set(train_texts)
    assert all(t.startswith("Madde ") and "\n\n" in t for t in eval_texts + train_texts)
    # stubs (every 4th article) are skipped
    assert "Madde 3\n\n" not in "".join(eval_texts + train_texts)


def test_select_articles_runs_out():
    with pytest.raises(RuntimeError, match="ran out"):
        select_articles(make_wiki(10), min_chars=100, n_eval=5, n_train=20)


def test_select_pages_skips_empty():
    pages = [{"text": "  "}, {"text": "a"}, {"text": "b"}]
    assert select_pages(pages, 2) == ["a", "b"]
    with pytest.raises(RuntimeError):
        select_pages(pages, 3)


def test_prepare_caches_and_reuses(tiny_config, tokenizer, fake_loaders):
    bundle = prepare_data(tiny_config, tokenizer, loaders=fake_loaders, log=lambda *_: None)
    assert (len(bundle.train), len(bundle.eval), len(bundle.ood), len(bundle.grammar)) == (30, 4, 4, 9)
    stats = bundle.metadata["stats"]
    assert stats["train_tokens"] > 0 and stats["grammar_pairs"] == 9
    assert is_prepared(tiny_config)

    # cached: the loaders must not be called again
    def boom(cfg):
        raise AssertionError("should use the cache")

    fake_loaders.wiki = boom
    again = prepare_data(tiny_config, tokenizer, loaders=fake_loaders, log=lambda *_: None)
    assert again.train["text"] == bundle.train["text"]


def test_cache_invalidated_by_data_settings(tiny_config, tokenizer, fake_loaders):
    from lfm3_cpt.config import replace

    prepare_data(tiny_config, tokenizer, loaders=fake_loaders, log=lambda *_: None)
    assert not is_prepared(replace(tiny_config, train_articles=10))
    assert not is_prepared(replace(tiny_config, seed=7))
    # training-only settings do not invalidate the data
    assert is_prepared(replace(tiny_config, learning_rate=1e-5, lora_r=8))
    stored = json.loads((tmp_data_dir(tiny_config) / "metadata.json").read_text())
    assert stored["signature"]["train_articles"] == 30
    assert load_data(tiny_config).metadata == stored


def tmp_data_dir(cfg):
    from pathlib import Path

    return Path(cfg.data_dir)


def test_with_eos_is_idempotent(tokenizer):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": ["a", "b</s>"]})
    once = with_eos(ds, tokenizer.eos_token)
    assert once["text"] == ["a</s>", "b</s>"]
    assert with_eos(once, tokenizer.eos_token)["text"] == once["text"]
