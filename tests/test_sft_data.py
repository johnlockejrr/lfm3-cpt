import json

import pytest

from lfm3_cpt.config import replace
from lfm3_cpt.sft_data import (
    chat_ids,
    is_sft_prepared,
    prepare_sft_data,
    to_messages,
    to_prompt_completion,
    with_template_kwargs,
)

QUIET = {"log": lambda *_: None}


@pytest.mark.parametrize(
    "example, expected_user, expected_answer",
    [
        ({"inputs": "Merhaba?", "targets": "Selam!", "language_code": "tur"}, "Merhaba?", "Selam!"),
        ({"instruction": "Çevir", "input": "hello", "output": "merhaba"}, "Çevir\n\nhello", "merhaba"),
        ({"instruction": "Say hi", "input": "", "output": "hi"}, "Say hi", "hi"),
        ({"Input": "Soru", "Output": "Cevap"}, "Soru", "Cevap"),  # case-insensitive (e.g. InstrucTurca)
        ({"prompt": "p", "response": "r"}, "p", "r"),
        ({"question": "q", "answer": "a"}, "q", "a"),
    ],
)
def test_to_messages_single_turn_formats(example, expected_user, expected_answer):
    msgs = to_messages(example)
    assert msgs == [{"role": "user", "content": expected_user}, {"role": "assistant", "content": expected_answer}]


def test_to_messages_sharegpt_multiturn_keeps_last_assistant():
    ex = {
        "conversations": [
            {"from": "system", "value": "Sen bir asistansın."},
            {"from": "human", "value": "a"},
            {"from": "gpt", "value": "b"},
            {"from": "human", "value": "c"},
            {"from": "gpt", "value": "d"},
            {"from": "human", "value": "dangling question"},
        ]
    }
    msgs = to_messages(ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    row = to_prompt_completion(msgs)
    assert row["completion"] == [{"role": "assistant", "content": "d"}]
    assert row["prompt"][-1] == {"role": "user", "content": "c"}


def test_to_messages_messages_format_and_rejects():
    ok = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
    assert to_messages(ok)[-1]["content"] == "y"
    assert to_messages({"messages": [{"role": "assistant", "content": "only"}]}) is None
    assert to_messages({"inputs": "q", "targets": "   "}) is None
    assert to_messages({"title": "t", "text": "no chat here"}) is None
    assert to_messages({"a": "q", "b": "r"}, user_col="a", assistant_col="b")[1]["content"] == "r"


def test_system_prompt_added_once():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    row = to_prompt_completion(msgs, "Türkçe yanıt ver.")
    assert row["prompt"][0] == {"role": "system", "content": "Türkçe yanıt ver."}
    with_system = [{"role": "system", "content": "own"}, *msgs]
    assert to_prompt_completion(with_system, "Türkçe yanıt ver.")["prompt"][0]["content"] == "own"


def test_prepare_filters_dedupes_and_caches(tiny_config, tokenizer, fake_sft_loader):
    cfg = replace(tiny_config, sft_max_train=-1, sft_max_eval=200)
    bundle = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, **QUIET)
    stats = bundle.metadata["stats"]
    # 40 rows, 10 English filtered out, + 1 duplicate + 1 empty answer (both Turkish)
    assert stats["filtered_out"] >= 10 and stats["duplicates"] == 1 and stats["unusable"] == 1
    assert len(bundle.train) == 30
    assert len(bundle.eval) == 6  # 8 test rows, 2 English
    assert all(r["completion"][0]["role"] == "assistant" for r in bundle.train)
    assert is_sft_prepared(cfg)
    again = prepare_sft_data(cfg, tokenizer, loader=lambda c: (_ for _ in ()).throw(AssertionError), **QUIET)
    assert len(again.train) == 30
    assert not is_sft_prepared(replace(cfg, sft_system_prompt="x"))
    assert is_sft_prepared(replace(cfg, sft_learning_rate=1e-5))  # training knobs don't invalidate the data


def test_prepare_drops_too_long(tiny_config, tokenizer, fake_sft_loader):
    cfg = replace(tiny_config, sft_max_length=20, sft_max_train=-1, sft_max_eval=-1)
    with pytest.raises(RuntimeError):  # everything is longer than 20 tokens
        prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)


def test_holdout_when_no_eval_split(tiny_config, tokenizer, fake_sft_loader):
    cfg = replace(tiny_config, sft_eval_split="", sft_eval_holdout=0.2, sft_max_train=-1, sft_max_eval=-1)
    bundle = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)
    assert len(bundle.eval) == 6 and len(bundle.train) == 24
    train_keys = {json.dumps(r["prompt"]) for r in bundle.train}
    assert not any(json.dumps(r["prompt"]) in train_keys for r in bundle.eval)


def test_no_filter(tiny_config, tokenizer, fake_sft_loader):
    cfg = replace(tiny_config, sft_filter_column="", sft_max_train=-1, sft_max_eval=-1)
    bundle = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)
    assert len(bundle.train) == 40 and len(bundle.eval) == 8


def test_bad_filter_column(tiny_config, tokenizer, fake_sft_loader):
    with pytest.raises(ValueError, match="filter column"):
        prepare_sft_data(
            replace(tiny_config, sft_filter_column="nope"), tokenizer, loader=fake_sft_loader, force=True, **QUIET
        )


def test_template_kwargs_column_and_chat_ids(tiny_config, tokenizer, fake_sft_loader):
    bundle = prepare_sft_data(tiny_config, tokenizer, loader=fake_sft_loader, **QUIET)
    ds = with_template_kwargs(bundle.train, tiny_config)
    assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}
    row = bundle.train[0]
    prompt = chat_ids(tokenizer, row["prompt"], add_generation_prompt=True)
    full = chat_ids(tokenizer, row["prompt"] + row["completion"])
    assert full[: len(prompt)] == prompt and len(full) > len(prompt)
