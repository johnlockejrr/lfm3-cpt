import pytest
import torch
import torch.nn.functional as F

from lfm3_cpt.chat_eval import (
    answer_perplexity,
    answer_spans,
    chrf,
    detect_language,
    generate_answers,
    summarize_generations,
)
from lfm3_cpt.reporting import sft_summary_table

ROWS = [
    {
        "prompt": [{"role": "user", "content": "bu ev çok güzel mi?"}],
        "completion": [{"role": "assistant", "content": "evet, ev çok güzel."}],
    },
    {
        "prompt": [{"role": "system", "content": "kısa yanıt ver"}, {"role": "user", "content": "istanbul nerede?"}],
        "completion": [{"role": "assistant", "content": "türkiye."}],
    },
    {"prompt": [{"role": "user", "content": "gel"}], "completion": [{"role": "assistant", "content": "geldi"}]},
]
KW = {"enable_thinking": False}


@pytest.mark.parametrize(
    "text, lang",
    [
        ("İstanbul, Türkiye'nin en kalabalık şehridir ve tarihi çok zengindir.", "tr"),
        ("Bu soruya cevap vermek için daha fazla bilgi gerekiyor.", "tr"),
        ("Ağaçlar ışığı sever.", "tr"),
        ("The capital of Turkey is Ankara, and it is in Central Anatolia.", "en"),
        ("Hallo Welt", "other"),
        ("", "empty"),
        ("12345 !!!", "empty"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_chrf():
    assert chrf("aynı cümle", "aynı cümle") == pytest.approx(100.0)
    assert chrf("xyz", "abc") == 0.0
    assert chrf("", "abc") == 0.0
    assert 0 < chrf("evler güzel", "ev güzel") < 100
    assert chrf("ev güzel", "evler güzel") != chrf("evler güzel", "ev güzel")  # recall-weighted (beta=2)


def test_answer_spans_start_at_answer(tokenizer):
    for (full, begin), row in zip(answer_spans(tokenizer, ROWS, KW), ROWS, strict=True):
        answer = tokenizer.decode(full[begin:], skip_special_tokens=True)
        assert answer.strip() == row["completion"][0]["content"]  # exactly the answer, no prompt tokens


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_answer_perplexity_matches_reference(model, tokenizer, batch_size):
    res = answer_perplexity(model, tokenizer, ROWS, KW, batch_size=batch_size)
    nll, n = 0.0, 0
    for full, begin in answer_spans(tokenizer, ROWS, KW):
        ids = torch.tensor([full])
        logits = model(input_ids=ids).logits[0]
        nll += F.cross_entropy(logits[begin - 1 : -1], ids[0, begin:], reduction="sum").item()
        n += len(full) - begin
    assert res.tokens == n
    assert res.loss == pytest.approx(nll / n, rel=1e-4)


def test_generate_batched_equals_single_and_keeps_tokenizer(model, tokenizer):
    tokenizer.padding_side = "right"
    prompts = [r["prompt"] for r in ROWS]
    batched = generate_answers(model, tokenizer, prompts, KW, max_new_tokens=6, batch_size=3, repetition_penalty=1.0)
    single = [
        generate_answers(model, tokenizer, [p], KW, max_new_tokens=6, batch_size=1, repetition_penalty=1.0)[0]
        for p in prompts
    ]
    assert [b["text"] for b in batched] == [s["text"] for s in single]  # left padding is handled correctly
    assert all(0 <= b["tokens"] <= 6 for b in batched)
    assert tokenizer.padding_side == "right"


def test_summarize_generations_and_table():
    answers = [
        {"text": "Bu çok güzel bir şehir.", "finished": True},
        {"text": "This is a city.", "finished": False},
        {"text": "", "finished": True},
    ]
    s = summarize_generations(answers, ["Bu güzel bir şehir.", "Bu bir şehir.", "Evet."])
    assert s["turkish_rate"] == pytest.approx(1 / 3)
    assert s["english_rate"] == pytest.approx(1 / 3)
    assert s["empty_rate"] == pytest.approx(1 / 3)
    assert s["finished_rate"] == pytest.approx(2 / 3)
    before = {"answer": {"perplexity": 9.0}, "generation": s, "lm": {"wiki": {"perplexity": 7.0}}}
    after = {
        "answer": {"perplexity": 5.0},
        "generation": {**s, "turkish_rate": 0.9},
        "lm": {"wiki": {"perplexity": 7.2}},
    }
    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(sft_summary_table(before, after))
    text = console.export_text()
    assert "answers in Turkish" in text and "forgetting check" in text and "-4.00" in text
