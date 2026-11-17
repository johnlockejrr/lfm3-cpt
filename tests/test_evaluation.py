import math

import pytest
import torch
import torch.nn.functional as F

from lfm3_cpt.evaluation import (
    GrammarResult,
    compare_grammar,
    corpus_perplexity,
    count_packed_rows,
    grammar_test,
    pack_token_rows,
    sentence_logprobs,
)

from .conftest import make_grammar

TEXTS = [
    "ev evler okul okula gel geldi.</s>",
    "türkiye istanbul ankara tarih nehir.</s>",
    "bir bu çok ile için.</s>",
] * 5


def test_pack_rows_cover_all_tokens(tokenizer):
    rows = pack_token_rows(tokenizer, TEXTS, max_length=16)
    flat = [t for ids in tokenizer(text=TEXTS)["input_ids"] for t in ids]
    assert [t for r in rows for t in r] == flat
    assert all(len(r) == 16 for r in rows[:-1])
    assert count_packed_rows(tokenizer, TEXTS, 16) == len([r for r in rows if len(r) > 1])


def _reference_ppl(model, rows):
    nll, n = 0.0, 0
    for r in rows:
        if len(r) < 2:
            continue
        ids = torch.tensor([r])
        logits = model(input_ids=ids).logits
        nll += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
        n += len(r) - 1
    return nll / n, n


@pytest.mark.parametrize("batch_size", [1, 3])
def test_corpus_perplexity_matches_reference(model, tokenizer, batch_size):
    calls = []
    res = corpus_perplexity(model, tokenizer, TEXTS, max_length=16, batch_size=batch_size, progress=calls.append)
    ref_loss, ref_n = _reference_ppl(model, pack_token_rows(tokenizer, TEXTS, 16))
    assert res.tokens == ref_n
    assert res.loss == pytest.approx(ref_loss, rel=1e-5)
    assert res.perplexity == pytest.approx(math.exp(ref_loss), rel=1e-5)
    assert sum(calls) == res.rows
    assert not model.training


def test_eval_mode_restores_training(model, tokenizer):
    model.train()
    corpus_perplexity(model, tokenizer, TEXTS[:2], max_length=16)
    assert model.training


def test_sentence_logprobs_batched_equals_unbatched(model, tokenizer):
    sentences = ["ev", "bu ev çok güzel ve büyük bir şehir.", "okula gel", "istanbul ankara"]
    batched = sentence_logprobs(model, tokenizer, sentences, batch_size=4)
    single = [sentence_logprobs(model, tokenizer, [s], batch_size=1)[0] for s in sentences]
    assert batched == pytest.approx(single, abs=1e-4)
    # manual reference for one sentence
    ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer("okula gel", add_special_tokens=False)["input_ids"]])
    lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], -1).gather(-1, ids[0, 1:, None]).sum().item()
    assert batched[2] == pytest.approx(lp, abs=1e-4)


def test_sentence_logprobs_does_not_mutate_tokenizer(model, tokenizer):
    tokenizer.padding_side = "left"
    try:
        sentence_logprobs(model, tokenizer, ["ev", "okula gel"], batch_size=2)
        assert tokenizer.padding_side == "left"  # the notebook silently switched it to "right"
    finally:
        tokenizer.padding_side = "right"


def test_grammar_test_and_breakdown(model, tokenizer):
    pairs = make_grammar(per=2)
    res = grammar_test(model, tokenizer, pairs, batch_size=3)
    assert len(res.passed) == 6
    assert set(res.by_phenomenon()) == {"binding", "passives", "scrambling"}
    d = res.to_dict()
    assert d["n_pairs"] == 6 and 0.0 <= d["accuracy"] <= 1.0


def test_grammar_accuracy_by_phenomenon():
    r = GrammarResult([True, False, True, True], ["a", "a", "b", "b"])
    assert r.accuracy == 0.75
    assert r.by_phenomenon() == {"a": 0.5, "b": 1.0}


def test_compare_grammar_never_raises_without_flips():
    pairs = make_grammar(per=2)  # binding, binding, passives, passives, scrambling, scrambling
    before = [False, True, True, True, False, False]
    after = [True, True, True, False, False, False]
    cmp = compare_grammar(pairs, before, after)
    assert cmp["fixed"] == 1 and cmp["broken"] == 1
    assert [e["phenomenon"] for e in cmp["examples"]] == ["binding"]  # passives got worse, scrambling unchanged
    # identical results: no examples, no StopIteration (the notebook's loop would crash here)
    assert compare_grammar(pairs, before, before)["examples"] == []
