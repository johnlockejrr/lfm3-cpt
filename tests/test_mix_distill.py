"""sft_mix (several SFT sources, language-match metric) and distill (model answers → SFT data)."""

import json
import os
import time

import pytest
from click.testing import CliRunner
from rich.console import Console

from lfm3_cpt import distill as distill_mod
from lfm3_cpt.chat_eval import summarize_generations
from lfm3_cpt.cli import cli
from lfm3_cpt.config import build_config, parse_mix_entry, replace
from lfm3_cpt.distill import distill, language_ok, last_number, prompts_from_records, prompts_from_text
from lfm3_cpt.sft_data import is_sft_prepared, prepare_sft_data

QUIET = {"log": lambda *_: None}
EN = [f"The answer to question {i} is that the city is old and the river is long." for i in range(60)]


@pytest.fixture()
def english_mix(tmp_path):
    path = tmp_path / "en-distill.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"What about question {i}?"},
                        {"role": "assistant", "content": EN[i]},
                    ]
                }
            )
            for i in range(60)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# --- config -----------------------------------------------------------------------------------------------------


def test_parse_mix_entry():
    assert parse_mix_entry("a/b.jsonl@0.25") == ("a/b.jsonl", 0.25)
    assert parse_mix_entry("a/b.jsonl@2") == ("a/b.jsonl", 2.0)
    assert parse_mix_entry("HuggingFaceH4/no_robots") == ("HuggingFaceH4/no_robots", None)
    assert parse_mix_entry("user@host/file.jsonl") == ("user@host/file.jsonl", None)  # not a weight
    with pytest.raises(ValueError):
        parse_mix_entry("x@1.5")
    with pytest.raises(ValueError):
        build_config(overrides={"sft_mix": ["a@0.6", "b@0.5"]})  # shares must add up to < 1


# --- mixing -----------------------------------------------------------------------------------------------------


def test_mix_share(tiny_config, tokenizer, fake_sft_loader, english_mix):
    cfg = replace(tiny_config, sft_max_train=-1, sft_max_eval=-1, sft_mix=[f"{english_mix}@0.25"], sft_mix_eval=5)
    b = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, **QUIET)
    sources = list(b.train["source"])
    n_main, n_mix = sources.count("main"), sources.count(str(english_mix))
    assert n_main == 30 and n_mix == 10  # 10 / (30 + 10) = 25 % of the final training set
    assert list(b.eval["source"]).count(str(english_mix)) == 5
    m = b.metadata["stats"]["mix"][str(english_mix)]
    assert (m["train"], m["eval"], m["available"]) == (10, 5, 60)
    # no example is both trained on and evaluated
    keys = lambda ds: {json.dumps(list(r["prompt"]) + list(r["completion"])) for r in ds}  # noqa: E731
    assert not keys(b.train) & keys(b.eval)
    assert is_sft_prepared(cfg)
    time.sleep(0.01)
    english_mix.write_text(
        english_mix.read_text()
        + json.dumps({"messages": [{"role": "user", "content": "new?"}, {"role": "assistant", "content": "yes, new."}]})
        + "\n"
    )
    os.utime(english_mix)
    assert not is_sft_prepared(cfg)  # editing a mix file rebuilds the SFT data


def test_mix_repeat_and_all(tiny_config, tokenizer, fake_sft_loader, english_mix):
    cfg = replace(tiny_config, sft_max_train=-1, sft_max_eval=-1, sft_mix=[f"{english_mix}@2"], sft_mix_eval=10)
    b = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)
    assert list(b.train["source"]).count(str(english_mix)) == 100  # (60 - 10 held out) x 2
    cfg = replace(cfg, sft_mix=[str(english_mix)])
    b = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)
    assert list(b.train["source"]).count(str(english_mix)) == 50


def test_holdout_leftovers_stay_in_training(tiny_config, tokenizer, fake_sft_loader):
    cfg = replace(tiny_config, sft_eval_split="", sft_eval_holdout=0.5, sft_max_eval=3, sft_max_train=-1)
    b = prepare_sft_data(cfg, tokenizer, loader=fake_sft_loader, force=True, **QUIET)
    assert len(b.eval) == 3 and len(b.train) == 30 - 3  # 30 Turkish rows; not 30 - 15


# --- language match ---------------------------------------------------------------------------------------------


def test_language_match_metric():
    answers = [
        {"text": t, "finished": True}
        for t in [
            "בירת צרפת היא פריז.",
            "קנברה היא הבירה.",
            "The capital is Canberra and it is in the south.",
            "19",
            "The dog is sleeping on the sofa.",
        ]
    ]
    prompts = [
        "מהי בירת צרפת?",
        "What is the capital of Australia?",
        "What is the capital of Australia?",
        "12+7?",
        "תרגם לאנגלית: הכלב ישן על הספה.",  # a Hebrew question whose right answer is English
    ]
    refs = [
        "פריז היא בירת צרפת.",
        "The capital is Canberra.",
        "The capital is Canberra.",
        "19",
        "The dog sleeps on the sofa.",
    ]
    s = summarize_generations(answers, refs, "he", prompts=prompts)
    assert s["target_rate"] == 1.0  # the only Hebrew reference got a Hebrew answer
    # judged by the reference: the English question answered in Hebrew is a miss, the translation is a hit
    assert s["language_match"] == pytest.approx(3 / 4)
    assert s["expected_languages"] == ["he", "en", "en", "empty", "en"]
    assert s["prompt_languages"] == ["he", "en", "en", "empty", "he"]


# --- distill ----------------------------------------------------------------------------------------------------


def test_nan_is_shown_as_a_dash():
    from rich.console import Console as C

    from lfm3_cpt.reporting import sft_summary_table

    c = C(record=True, width=100)
    c.print(
        sft_summary_table(
            {"generation": {"language_match": float("nan")}}, {"generation": {"language_match": 0.5}}, language="he"
        )
    )
    text = c.export_text()
    assert "nan" not in text and "—" in text and "50.0%" in text


def test_prompts_from_records_and_text(tmp_path):
    recs = [
        {"instruction": "Say hi", "input": "", "output": "hi"},
        {"question": "What is 2+2?", "answer": "2+2 = 4\n#### 4"},
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello?"},
                {"role": "assistant", "content": "Hi."},
            ]
        },
        {"instruction": "Only a prompt here"},
        {"title": "nothing usable"},
    ]
    items = prompts_from_records(recs)
    assert [it["prompt"][-1]["content"] for it in items] == ["Say hi", "What is 2+2?", "Hello?", "Only a prompt here"]
    assert items[1]["reference"].endswith("#### 4") and items[3]["reference"] is None
    assert items[2]["prompt"][0]["role"] == "system"  # the system prompt is kept
    assert prompts_from_records([{"q": "custom"}], prompt_field="q")[0]["prompt"][0]["content"] == "custom"
    txt = tmp_path / "p.txt"
    txt.write_text("first\n\n second \n")
    assert [it["prompt"][0]["content"] for it in prompts_from_text(txt)] == ["first", "second"]


def test_last_number_and_language():
    assert last_number("so the total is 1,234.5 dollars") == 1234.5
    assert last_number("Natalia sold 48/2 = 24 clips.\n#### 72") == 72
    assert last_number("no digits") is None
    assert language_ok("The capital of Australia is Canberra.", "en")
    assert not language_ok("בירת אוסטרליה היא קנברה.", "en")
    assert language_ok("בירת אוסטרליה היא קנברה.", "he") and language_ok("anything", "any")


def test_distill_filters_and_resume(tmp_path, monkeypatch):
    items = [{"prompt": [{"role": "user", "content": f"q{i}"}], "reference": f"#### {i}"} for i in range(6)]
    canned = {
        "q0": {"text": "The answer is 0.", "finished": True},
        "q1": {"text": "The answer is 7.", "finished": True},  # wrong number
        "q2": {"text": "The answer is 2 but", "finished": False},  # unfinished
        "q3": {"text": "", "finished": True},  # empty
        "q4": {"text": "התשובה היא 4.", "finished": True},  # wrong language
        "q5": {"text": "It is 5, of course.", "finished": True},
    }
    calls = []

    def fake_generate(model, tok, prompts, kwargs, *a, **k):
        calls.append(len(prompts))
        return [canned[p[-1]["content"]] for p in prompts]

    monkeypatch.setattr(distill_mod, "generate_answers", fake_generate)
    out = tmp_path / "d.jsonl"
    st = distill(None, None, items, out, chunk=3, language="en", verify_number=True, source_name="M")
    assert (st.kept, st.wrong_number, st.unfinished, st.empty, st.wrong_language) == (2, 1, 1, 1, 1)
    assert st.processed == 6 and calls == [3, 3]
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["messages"][-1]["content"] for r in rows] == ["The answer is 0.", "It is 5, of course."]
    assert rows[0]["distilled_by"] == "M"

    # resume after the first chunk: only the second chunk is generated again
    progress = json.loads((tmp_path / "d.jsonl.progress.json").read_text())
    assert progress["processed"] == 6
    (tmp_path / "d.jsonl.progress.json").write_text(
        json.dumps({"processed": 3, "stats": {"kept": 1, "wrong_number": 1, "unfinished": 1}})
    )
    out.write_text(out.read_text().splitlines()[0] + "\n")
    calls.clear()
    st = distill(None, None, items, out, chunk=3, language="en", verify_number=True, resume=True)
    assert calls == [3] and st.kept == 2 and st.processed == 6 and st.skipped_done == 3
    assert len(out.read_text().splitlines()) == 2


def test_distill_cli_with_tiny_model(tmp_path, tiny_model_dir):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("bu ev güzel mi?\nokula gel\n")
    out = tmp_path / "out.jsonl"
    args = ["-s", f"model_id={tiny_model_dir}", "-s", "trust_remote_code=false", "-s", "precision=fp32"]
    res = CliRunner().invoke(
        cli,
        [
            *args,
            "distill",
            str(prompts),
            "-o",
            str(out),
            "--language",
            "any",
            "--max-new-tokens",
            "4",
            "--batch-size",
            "2",
        ],
    )
    # the tiny random model never reaches EOS in 4 tokens: nothing kept, and the command says so
    assert res.exit_code == 1, res.output
    assert "kept" in res.output and "no answers were kept" in res.output
    assert out.exists() and out.with_name("out.jsonl.progress.json").exists()
    res = CliRunner().invoke(cli, [*args, "distill", str(prompts), "-o", str(out), "--verify-number"])
    assert res.exit_code == 2 and "reference answers" in res.output


def test_empty_mix_file_is_a_clear_error(tiny_config, tokenizer, fake_sft_loader, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no answers were kept"):
        prepare_sft_data(
            replace(tiny_config, sft_mix=[str(empty)]), tokenizer, loader=fake_sft_loader, force=True, **QUIET
        )


def test_exit_after_streaming_only_for_hub(monkeypatch):
    from lfm3_cpt import cli as cli_mod

    exits = []
    monkeypatch.setattr(os, "_exit", exits.append)
    monkeypatch.delenv("LFM3_CPT_NO_HARD_EXIT", raising=False)
    cli_mod._exit_after_streaming(False)
    assert exits == []
    cli_mod._exit_after_streaming(True)
    assert exits == [0]
    monkeypatch.setenv("LFM3_CPT_NO_HARD_EXIT", "1")
    cli_mod._exit_after_streaming(True)
    assert exits == [0]


# --- end to end: SFT with a distilled English mix ---------------------------------------------------------------


@pytest.mark.slow
def test_sft_with_mix_end_to_end(tiny_config, fake_loaders, fake_sft_loader, english_mix, tmp_path):
    from lfm3_cpt.sft import run_sft
    from lfm3_cpt.train import run_training

    console = Console(file=open(tmp_path / "log.txt", "w"), width=140)  # noqa: SIM115
    run_training(tiny_config, console, merge=False, loaders=fake_loaders)
    cfg = replace(tiny_config, sft_mix=[f"{english_mix}@0.25"], sft_mix_eval=2, sft_max_eval=4, sft_gen_samples=6)
    res = run_sft(cfg, console, start="cpt", loader=fake_sft_loader, lm_eval=False)
    gen = res["final"]["generation"]
    assert "language_match" in gen and len(gen["prompt_languages"]) == 6
    rows = [json.loads(x) for x in (tmp_path / "run" / "sft" / "generations.jsonl").read_text().splitlines()]
    assert {r["source"] for r in rows} == {"main", str(english_mix)}
    log = (tmp_path / "log.txt").read_text()
    assert f"+ {english_mix}" in log and "reference's language" in log
