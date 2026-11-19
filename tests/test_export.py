"""`export` (Hub/raw files → budgeted train/eval JSONL) and `multiblimp` (minimal pairs)."""

import json

import pytest
from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.corpus_files import read_pairs
from lfm3_cpt.hub_export import describe, export_corpus, extract_text, get_path, multiblimp_rows
from lfm3_cpt.textnorm import Normalizer

PROTOCOL = {
    "protocol_name": "committee_1999_42",
    "meta": {"knesset": 15, "date": "1999-07-01"},
    "sentences": [
        {"speaker": "A", "text": "שלום לכולם."},
        {"speaker": "B", "text": "  "},
        {"speaker": "B", "text": "נתחיל."},
    ],
}


def test_get_path_and_extract():
    assert get_path(PROTOCOL, "meta.knesset") == 15
    assert get_path(PROTOCOL, "sentences.0.speaker") == "A"
    assert get_path(PROTOCOL, "nope.x") is None
    assert extract_text(PROTOCOL, "sentences", item_field="text") == "שלום לכולם.\nנתחיל."  # blanks dropped
    assert extract_text({"lines": ["a", "", "b"]}, "lines", join=" ") == "a b"
    assert extract_text({"title": "T", "text": "body"}, "text", title_field="title") == "T\n\nbody"
    assert extract_text({"text": "   "}, "text") is None
    assert extract_text(PROTOCOL, "sentences") is None  # records without --item-field


def test_describe_shows_nested_structure():
    lines = describe(PROTOCOL)
    assert "protocol_name: str = 'committee_1999_42'" in lines
    assert "sentences: list[3]" in lines
    assert any(line.strip().startswith("text: str") for line in lines)


def _records(n, size=50):
    return [{"text": f"doc {i} " + "x" * size} for i in range(n)]


def test_export_budgets_and_eval_split(tmp_path):
    seen = []
    st = export_corpus(_records(100), tmp_path / "src", eval_docs=5, max_docs=20, progress=seen.append)
    train = (tmp_path / "src.train.jsonl").read_text().splitlines()
    evals = (tmp_path / "src.eval.jsonl").read_text().splitlines()
    assert (st.train_docs, st.eval_docs, len(train), len(evals)) == (20, 5, 20, 5)
    assert json.loads(evals[0])["text"].startswith("doc 0 ")  # eval first, then train: disjoint
    assert json.loads(train[0])["text"].startswith("doc 5 ")
    assert not st.exhausted and seen
    st = export_corpus(_records(100, size=95), tmp_path / "c", eval_docs=0, max_chars=1000)
    assert st.train_chars >= 1000 and st.train_docs == 10 and not (tmp_path / "c.eval.jsonl").exists()


def test_eval_only_export(tmp_path):
    st = export_corpus(_records(50), tmp_path / "en", eval_docs=7, max_docs=0)
    assert (st.eval_docs, st.train_docs) == (7, 0)
    assert not (tmp_path / "en.train.jsonl").exists() and (tmp_path / "en.eval.jsonl").exists()
    assert st.records_seen == 7  # stops reading as soon as the eval file is full


def test_export_skip_min_chars_and_exhaustion(tmp_path):
    recs = [{"text": "short"}, {"other": 1}, *_records(5)]
    st = export_corpus(recs, tmp_path / "s", min_chars=20, eval_docs=1, skip=1, max_docs=100)
    assert st.skipped_empty == 1 and st.skipped_short == 0  # the short one was skipped by --skip
    assert st.eval_docs == 1 and st.train_docs == 4 and st.exhausted


def test_export_cli_from_local_files(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    (src / "a.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in [PROTOCOL] * 6), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(cli, ["export", "--data-files", str(src / "*.jsonl"), "--inspect"])
    assert res.exit_code == 0, res.output
    assert "sentences: list[3]" in res.output and "record 0" in res.output
    out = tmp_path / "corpora" / "knesset"
    res = runner.invoke(
        cli,
        [
            "export",
            "--data-files",
            str(src / "*.jsonl"),
            "--field",
            "sentences",
            "--item-field",
            "text",
            "--join",
            "\\n",
            "--min-chars",
            "5",
            "--eval-docs",
            "2",
            "-o",
            str(out),
            "--shuffle-buffer",
            "0",
        ],
    )
    assert res.exit_code == 0, res.output
    first = json.loads((tmp_path / "corpora" / "knesset.train.jsonl").read_text().splitlines()[0])
    assert first["text"] == "שלום לכולם.\nנתחיל."
    assert len((tmp_path / "corpora" / "knesset.eval.jsonl").read_text().splitlines()) == 2
    # the result plugs straight into train_files / eval_files
    from lfm3_cpt.config import build_config
    from lfm3_cpt.corpus_files import load_local_corpus

    cfg = build_config(overrides={"train_files": [f"{out}.train.jsonl"], "eval_files": [f"{out}.eval.jsonl"]})
    corpus = load_local_corpus(cfg, log=lambda *_: None)
    assert len(corpus.train) == 4 and len(corpus.eval) == 2


def test_export_cli_errors(tmp_path):
    runner = CliRunner()
    assert runner.invoke(cli, ["export"]).exit_code == 2
    (tmp_path / "x.jsonl").write_text('{"body": "no text field here at all, long enough"}\n')
    res = runner.invoke(
        cli,
        [
            "export",
            "--data-files",
            str(tmp_path / "x.jsonl"),
            "-o",
            str(tmp_path / "o"),
            "--eval-docs",
            "0",
            "--shuffle-buffer",
            "0",
        ],
    )
    assert res.exit_code == 1 and "--inspect" in res.output
    res = runner.invoke(
        cli, ["export", "--data-files", str(tmp_path / "x.jsonl"), "-o", str(tmp_path / "o"), "--max-chars", "lots"]
    )
    assert res.exit_code == 2 and "not a size" in res.output


def test_csv_source(tmp_path):
    (tmp_path / "d.csv").write_text(
        'text,url\n"שורה ראשונה ארוכה מספיק",u1\n"שורה שנייה ארוכה מספיק",u2\n', encoding="utf-8"
    )
    res = CliRunner().invoke(
        cli,
        [
            "export",
            "--data-files",
            str(tmp_path / "d.csv"),
            "-o",
            str(tmp_path / "w"),
            "--eval-docs",
            "0",
            "--min-chars",
            "5",
            "--shuffle-buffer",
            "0",
        ],
    )
    assert res.exit_code == 0, res.output
    assert len((tmp_path / "w.train.jsonl").read_text(encoding="utf-8").splitlines()) == 2


MULTIBLIMP_TSV = (
    "sen\twrong_sen\tlang\tgrammatical_feature\n"
    "הילד הולך\tהילד הולכים\theb\tNumber\n"
    "הילדה הולכת\tהילדה הולך\theb\tGender\n"
    "אני הולך\tאני הולך\theb\tPerson\n"  # identical: dropped
)


def test_multiblimp_rows_and_cli(tmp_path):
    rows = multiblimp_rows(MULTIBLIMP_TSV)
    assert len(rows) == 2 and rows[0]["phenomenon"] == "agreement_Number"
    with pytest.raises(ValueError, match="not a MultiBLiMP"):
        multiblimp_rows("a\tb\n1\t2\n")
    tsv = tmp_path / "data.tsv"
    tsv.write_text(MULTIBLIMP_TSV, encoding="utf-8")
    out = tmp_path / "pairs" / "heb.jsonl"
    res = CliRunner().invoke(cli, ["multiblimp", "heb", "--tsv", str(tsv), "-o", str(out)])
    assert res.exit_code == 0, res.output
    pairs = read_pairs(out, Normalizer())
    assert [p["linguistic_phenomenon"] for p in pairs] == ["agreement_Number", "agreement_Gender"]
    res = CliRunner().invoke(cli, ["multiblimp", "heb", "--tsv", str(tsv), "-o", str(out), "--limit", "1"])
    assert res.exit_code == 0 and len(out.read_text().splitlines()) == 1
