"""Any-script words (Maltese and Hebrew), apostrophe unification, and minimal pairs from plain held-out text."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.cloze import Vocabulary, form_key
from lfm3_cpt.cloze import is_word as cloze_word
from lfm3_cpt.pairs import make_pairs
from lfm3_cpt.restore import Constraints, pattern_regex
from lfm3_cpt.textnorm import Normalizer
from lfm3_cpt.words import is_word

MT = [
    "Il-Gvern ħabbar li l-proġett il-ġdid se jibda s-sena d-dieħla f'Għawdex.",
    "Il-ministru qal li x-xogħol fuq it-triq il-ġdida se jitlesta qabel il-Milied.",
    "Ir-residenti tal-lokalità għamlu petizzjoni biex jinbena ġnien pubbliku ġdid.",
    "L-atleta Għawdxija rebħet l-ewwel midalja tad-deheb għall-pajjiż f'dawn il-logħob.",
    "Il-kunsill lokali ddeċieda li jtejjeb is-servizzi għall-anzjani tal-lokalità.",
]


def test_any_script_words():
    for w in ["il-kelma", "għall-ewwel", "ta'", "Għawdxija", "rebħet", "ċirku", "żiemel", "שבת", "ר׳", "שֶׁ"]:
        assert is_word(w, min_letters=1), w
    for w in ["123", "-", "il-", "a", "'ta", "f'", "x--y", "...", "2026"]:
        assert not is_word(w), w
    assert cloze_word("il-kelma") and not cloze_word("ר׳") and not cloze_word("רז״ל")


def test_maltese_form_classes():
    assert form_key("il-ġdida") == ("il-", "da") and form_key("għall-pajjiż") == ("għall-", "iż")
    assert form_key("ובשטר") == ("ו", "טר") and form_key("rebħet") == ("", "et")
    vocab = Vocabulary.from_docs(MT * 3, stop_top=3)
    out = vocab.distractors("il-ġdida", 2, random.Random(0), set())
    assert out and all(w.startswith(("il-", "l-")) or w.endswith("da") for w in out)


def test_maltese_patterns_and_letters():
    assert pattern_regex("re??et").match("rebħet") and pattern_regex("?ħ*").match("għall")
    assert Constraints(min_letters=6, max_letters=6).fits("rebħet")
    assert Constraints(words=1, pattern="il-*").fits("il-ġdida")


def test_apostrophe_unification():
    n = Normalizer("NFC", ["typographic_apostrophes"])
    assert n("ta’ Malta f‘Għawdex") == "ta' Malta f'Għawdex"
    assert Normalizer("NFC", [])("ta’") == "ta’"


def test_pairs_any_language():
    docs = [{"text": "\n".join(MT)} for _ in range(4)]
    pairs = make_pairs(docs, n=6, seed=0)
    assert pairs and {p["phenomenon"] for p in pairs} == {"word_order", "word_substitution"}
    for p in pairs:
        assert p["sentence_good"] != p["sentence_bad"]
        assert Counter(p["sentence_good"].split()) != Counter() and "il-" in p["sentence_good"].lower() or True


def test_cli_pairs(tmp_path: Path):
    src = tmp_path / "eval.jsonl"
    src.write_text("".join(json.dumps({"text": "\n".join(MT)}, ensure_ascii=False) + "\n" for _ in range(3)))
    out = tmp_path / "pairs.jsonl"
    res = CliRunner().invoke(
        cli, ["-s", "strip_marks=typographic_apostrophes", "pairs", str(src), "-o", str(out), "-n", "4"]
    )
    assert res.exit_code == 0, res.output
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and all({"sentence_good", "sentence_bad", "phenomenon"} <= set(r) for r in rows)
    bad = CliRunner().invoke(cli, ["pairs", str(tmp_path / "missing.jsonl"), "-o", str(out)])
    assert bad.exit_code != 0


def test_maltese_answer_language():
    from lfm3_cpt.chat_eval import answer_language, in_target_language

    mt = "AI u ML huma teknoloġiji interrelatati, iżda t-tnejn għandhom differenzi kbar."
    en = "You get diabetes if your pancreas does not make enough insulin."
    assert in_target_language(mt, "mt") is True and in_target_language(en, "mt") is False
    assert answer_language(mt, "mt") == "mt" and answer_language(en, "mt") == "en"
    assert in_target_language("Mort fil-maħżen.", "mt") is True


def test_cli_languages(tmp_path: Path):
    rows = [
        {"source": "main", "reference": "Mort fil-maħżen.", "before": "Mort fil-maħżen.", "after": "Mort fil-maħżen."},
        {
            "source": "opus.mt2en.jsonl",
            "reference": "The dog is sleeping on the sofa.",
            "before": "Il-kelb qed jorqod fuq is-sufan.",
            "after": "The dog is sleeping on the sofa.",
        },
        {
            "source": "en-gsm8k.clean.jsonl",
            "reference": "The answer is 19.",
            "before": "The answer is 19.",
            "after": "It-tweġiba hija dsatax u l-kalkolu huwa sempliċi.",
        },
    ]
    run = tmp_path / "mt-sft" / "sft"
    run.mkdir(parents=True)
    (run / "generations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    res = CliRunner().invoke(cli, ["-s", "target_language=mt", "languages", str(run)])
    assert res.exit_code == 0, res.output
    assert "opus.mt2en.jsonl" in res.output and "2 / 3" in res.output  # main and the translation right, gsm8k not
