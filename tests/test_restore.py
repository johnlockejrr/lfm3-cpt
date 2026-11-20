"""Lacuna restoration: constraints, candidate collection, ranking with the text after the gap."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.restore import Constraints, first_words, letters, pattern_regex, restore, split_gap


def test_letters_and_pattern():
    assert letters("נִפְרָעִין") == 6 and letters("ר׳ יהודה") == 6
    rx = pattern_regex("נ??עין")
    assert rx.match("נפרעין") and not rx.match("מוציאין") and not rx.match("נפרעים")
    assert pattern_regex("*ין").match("עבודין") and pattern_regex("ר? יהודה").match("רב יהודה")


def test_constraints():
    c = Constraints(min_letters=5, max_letters=6, words=1)
    assert c.fits("נפרעין") and not c.fits("מוציאין") and not c.fits("רבי יהודה")
    assert Constraints(words=2, pattern="ר* *").fits("רבי יהודה")
    assert Constraints(words=None, regex="ין$").fits("שאינן עבודין")
    assert not Constraints().fits("   ")


def test_split_gap_and_first_words():
    assert split_gap("אין ___ מהן") == ("אין", "מהן")
    with pytest.raises(ValueError):
        split_gap("no gap here")
    conts = [" נפרעין מהן", " מוציאין", "ין מהן", " נפרעין מהן לענין"]
    # the 2nd ends without a space (maybe cut off), the 3rd continues the previous word
    assert first_words(conts, 1) == Counter({"נפרעין": 2})
    assert first_words(conts, 2) == Counter({"נפרעין מהן": 1})


def test_restore_ranks_with_right_context():
    seen_texts = []

    def logprob(texts):
        seen_texts.extend(texts)
        return [-1.0 if "נפרעין" in t else -float(len(t)) for t in texts]

    result = restore(
        "אין ___ מהן לענין כתובה",
        Constraints(min_letters=5, max_letters=7),
        logprob,
        sample_fn=lambda left: [" מוציאין מהן", " גובין מהן", " נפרעין מהן", " x"],
        vocab=Counter({"נפרעין": 3, "פורעין": 2, "אב": 9}),
        given=["משתלמין"],
    )
    rows = result["rows"]
    assert rows[0]["filler"] == "נפרעין" and rows[0]["sources"] == ["corpus", "model"] and rows[0]["delta"] == 0
    fillers = {r["filler"] for r in rows}
    assert {"מוציאין", "פורעין", "משתלמין"} <= fillers and "גובין" in fillers and "אב" not in fillers
    assert all(t.endswith("מהן לענין כתובה") for t in seen_texts)  # the words after the gap are scored
    assert result["counts"] == {"given": 1, "model": 3, "corpus": 2}


def test_restore_nothing_fits():
    out = restore("אין ___ מהן", Constraints(pattern="ת???"), lambda t: [0.0] * len(t), given=["נפרעין"])
    assert out["rows"] == []


def test_cli_restore(tmp_path: Path, tiny_model_dir: str):
    vocab = tmp_path / "train.jsonl"
    vocab.write_text(
        json.dumps({"text": "אין נפרעין מהן לענין כתובה אבל גובין מהן מוציאין"}, ensure_ascii=False) + "\n"
    )
    out_file = tmp_path / "restore.json"
    base = ["-p", "cpu-debug", "-s", f"model_id={tiny_model_dir}", "-s", f"output_dir={tmp_path / 'run'}"]
    out = CliRunner().invoke(
        cli,
        [
            *base,
            "restore",
            tiny_model_dir,
            "--text",
            "אין ___ מהן לענין כתובה",
            "--letters",
            "5-7",
            "--vocab-file",
            str(vocab),
            "--candidate",
            "משתלמין",
            "--samples",
            "4",
            "--output",
            str(out_file),
        ],
    )
    assert out.exit_code == 0, out.output
    data = json.loads(out_file.read_text(encoding="utf-8"))
    fillers = {r["filler"] for r in data["rows"]}
    assert {"נפרעין", "מוציאין", "משתלמין", "גובין"} <= fillers and "אין" not in fillers  # 3 letters: too short
    out = CliRunner().invoke(cli, [*base, "restore", tiny_model_dir, "--text", "no gap", "--samples", "0"])
    assert out.exit_code != 0 and "gap marker" in out.output
