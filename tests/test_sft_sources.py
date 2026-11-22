"""Raw instruction / parallel data → chat examples: TaCo answers, parallel pairs, the CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.sft_data import to_messages
from lfm3_cpt.sft_sources import pair_fields, parallel_pairs, taco_answer, taco_messages, translation_messages

TACO_OUT = (
    "Instruction in English: Edit this sentence to have a different point of view\n.\n"
    "Response in English: I went to the store.\n.\n"
    "Response in Maltese: Mort fil-maħżen.\n<|end_of_text|>"
)
TACO_PLACEHOLDER = (
    "Instruction in English: Analyze the impact.\n.\nResponse in English: <Generated response in English>\n"
    "Response in Maltese: <Translated response from English to Maltese>"
)


def test_taco_answer():
    assert taco_answer(TACO_OUT) == "Mort fil-maħżen."
    assert taco_answer(TACO_PLACEHOLDER) == ""
    assert taco_answer("Response in Maltese: Iva.\nResponse in English: Yes.") == "Iva."
    assert taco_answer("no chain at all") == ""


def test_taco_messages():
    rec = {
        "instruction": "Editja din is-sentenza biex ikollok perspettiva differenti",
        "input": "John mar fil-maħżen",
        "output": TACO_OUT,
    }
    m = taco_messages(rec)
    assert m["messages"][0]["content"].endswith("\n\nJohn mar fil-maħżen")
    assert m["messages"][1]["content"] == "Mort fil-maħżen."
    assert taco_messages({**rec, "input": "nan"})["messages"][0]["content"] == rec["instruction"]
    assert taco_messages({**rec, "output": TACO_PLACEHOLDER}) is None
    assert to_messages(m) is not None  # the SFT loader reads it as is


def test_pair_fields_and_filtering():
    assert pair_fields({"translation": {"en": "a", "mt": "b"}}, "en", "mt") == ("a", "b")
    assert pair_fields({"en": "a", "mt": "b"}, "en", "mt") == ("a", "b")
    assert pair_fields({"source": "a", "target": "b"}, "en", "mt") == ("a", "b")
    assert pair_fields({"x": 1}, "en", "mt") is None
    recs = [
        {"en": "The government announced a new project in Gozo.", "mt": "Il-Gvern ħabbar proġett ġdid f'Għawdex."},
        {"en": "The government announced a new project in Gozo.", "mt": "duplicate source sentence ignored here"},
        {"en": "Too short.", "mt": "Qasir."},
        {"en": "1234 5678 9012 3456 7890 1234", "mt": "1234 5678 9012 3456 7890 1234"},
        {
            "en": "The council will improve services for the elderly.",
            "mt": "Il-kunsill se jtejjeb is-servizzi għall-anzjani.",
        },
    ]
    pairs = parallel_pairs(recs, "en", "mt", n=10)
    assert len(pairs) == 2 and all(len(a) >= 30 for a, _ in pairs)
    msgs = translation_messages(pairs, "en", "mt")
    assert msgs[0]["messages"][0]["content"].startswith("Translate into Maltese:")
    back = translation_messages([(b, a) for a, b in pairs], "mt", "en")
    assert back[0]["messages"][0]["content"].startswith("Ittraduċi għall-Ingliż:") and back[0]["source"] == "mt2en"


def test_cli_sft_source(tmp_path: Path):
    taco = tmp_path / "taco.jsonl"
    rows = [{"instruction": f"Ikteb sentenza numru {i}", "input": "", "output": TACO_OUT} for i in range(5)]
    rows.append({"instruction": "X", "input": "", "output": TACO_PLACEHOLDER})
    taco.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    out = tmp_path / "mt-alpaca.jsonl"
    res = CliRunner().invoke(cli, ["sft-source", "taco", str(taco), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5 and "1 without a Maltese answer" in res.output

    par = tmp_path / "opus.jsonl"
    words = ["dar", "triq", "knisja", "baħar", "xemx", "ilma", "ħobż", "ktieb", "skola", "belt"]
    lines = [
        {
            "translation": {
                "en": f"This is the sentence about the {w} number {i} today.",
                "mt": f"Din hija s-sentenza dwar il-{w} numru {i} illum.",
            }
        }
        for i, w in enumerate(words * 2)
    ]
    par.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines))
    res = CliRunner().invoke(cli, ["sft-source", "parallel", str(par), "-o", str(tmp_path / "opus"), "-n", "5"])
    assert res.exit_code == 0, res.output
    fwd = (tmp_path / "opus.en2mt.jsonl").read_text(encoding="utf-8").splitlines()
    bwd = (tmp_path / "opus.mt2en.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(fwd) == 5 and len(bwd) == 5
    assert not {json.loads(x)["messages"][1]["content"] for x in fwd} & {
        json.loads(x)["messages"][0]["content"] for x in bwd
    }


def test_repair_elision():
    from lfm3_cpt.sft_sources import repair_elision

    assert repair_elision("sistemi ta 'kontroll") == "sistemi ta' kontroll"
    assert repair_elision("Dan jista 'jinkludi lezzjonijiet") == "Dan jista' jinkludi lezzjonijiet"
    assert repair_elision("Tista 'tgħidli? erba 'snin, terġa 'tibda") == "Tista' tgħidli? erba' snin, terġa' tibda"
    assert repair_elision("u 'Kif inti?' qalli") == "u 'Kif inti?' qalli"  # an opening quote, not an elision
    assert repair_elision("ta' kontroll") == "ta' kontroll"  # already right
    rec = {"instruction": "Spjega", "input": "", "output": "Response in Maltese: Dan jista 'jgħin ta 'kuljum."}
    assert taco_messages(rec)["messages"][1]["content"] == "Dan jista' jgħin ta' kuljum."
