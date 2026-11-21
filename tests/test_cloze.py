"""Gap-filling benchmark: vocabulary, item building, scoring, summary, and the CLI end to end."""

from __future__ import annotations

import json
import random
from pathlib import Path

from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.cloze import GAP, Vocabulary, bucket, build_items, form_key, score_items, summarize

TEXT = [
    "החשוד על הבכורות אין לוקחין ממנו בשר צבאין ולא עורות שאינן עבודין",
    "רבי אליעזר אומר לוקחין ממנו עורות של נקבה ואין לוקחין ממנו צמר מלובן",
    "ולא מטונף אבל לוקחין ממנו טוי ובגדים עבודין מכל אדם",
    "האשה נקנית בשלש דרכים וקונה את עצמה בשתי דרכים נקנית בכסף ובשטר ובביאה",
    "בית שמאי אומרים בדינר ובשוה דינר ובית הלל אומרים בפרוטה ובשוה פרוטה",
]


def _docs(n: int = 30) -> list[str]:
    rng = random.Random(0)
    return ["\n".join(rng.sample(TEXT, 3)) for _ in range(n)]


def test_form_key_and_bucket():
    assert form_key("ובשטר") == ("ו", "טר")
    assert form_key("עבודין") == ("", "ין")
    assert bucket(0) == "unseen" and bucket(2) == "rare (1-3)" and bucket(10) == "medium (4-30)"
    assert bucket(500) == "common (31+)"


def test_vocabulary_distractors_share_form_class():
    vocab = Vocabulary.from_docs(_docs(), stop_top=5)
    rng = random.Random(1)
    out = vocab.distractors("עבודין", 1, rng, set())
    assert out and all(form_key(w) == form_key("עבודין") for w in out)  # e.g. צבאין: same class when it exists
    out = vocab.distractors("לוקחין", 3, rng, set())
    assert len(out) == 3 and "לוקחין" not in out
    assert all(w.endswith("ין") for w in out)  # no ל…ין word in this vocabulary: same ending instead


def test_build_items_with_proposer():
    vocab = Vocabulary.from_docs(_docs(), stop_top=5)
    calls = []

    def proposer(left: str, answer: str) -> list[str]:
        calls.append(left)
        return [answer, "מבושלין", "1905", "טהורין"]

    items = build_items(
        _docs(10), vocab, n_items=8, seed=3, corpus_distractors=2, model_distractors=2, proposer=proposer
    )
    assert 0 < len(items) <= 8 and calls
    for it in items:
        assert it["text"].count(GAP) == 1
        assert it["answer"] in it["candidates"] and len(set(it["candidates"])) == len(it["candidates"])
        assert it["answer"] not in it["kinds"]
        model_opts = [w for w, k in it["kinds"].items() if k == "model"]
        assert model_opts == ["מבושלין", "טהורין"]  # the answer and the number are skipped
        assert len(it["text"].split(GAP)[0].split()) >= 4


def test_score_and_summarize():
    items = [
        {
            "id": 1,
            "text": f"a {GAP} c",
            "answer": "b",
            "candidates": ["x", "b", "y"],
            "kinds": {"x": "corpus", "y": "model"},
            "freq": 0,
        },
        {"id": 2, "text": f"a {GAP} c", "answer": "b", "candidates": ["b", "z"], "kinds": {"z": "corpus"}, "freq": 50},
    ]
    good = {"a b c": -1.0, "a x c": -3.0, "a y c": -0.5, "a z c": -2.0}
    results = score_items(lambda texts: [good[t] for t in texts], items)
    assert [r["rank"] for r in results] == [2, 1]
    assert results[0]["best"] == "y"
    s = summarize(results)
    assert s["top1"] == 0.5 and s["top3"] == 1.0 and s["mean_rank"] == 1.5
    assert abs(s["chance_top1"] - (1 / 3 + 1 / 2) / 2) < 1e-9
    assert s["pairwise"]["corpus"] == (1.0, 2) and s["pairwise"]["model"] == (0.0, 1)
    assert s["by_bucket"]["unseen"] == (0.0, 1) and s["by_bucket"]["common (31+)"] == (1.0, 1)


def test_cli_cloze_build_and_score(tmp_path: Path, tiny_model_dir: str):
    eval_file, vocab_file = tmp_path / "eval.jsonl", tmp_path / "train.jsonl"
    for path, docs in ((eval_file, _docs(12)), (vocab_file, _docs(40))):
        path.write_text("".join(json.dumps({"text": d}, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    items, results = tmp_path / "cloze.jsonl", tmp_path / "scores.jsonl"
    runner = CliRunner()
    base = ["-p", "cpu-debug", "-s", f"model_id={tiny_model_dir}", "-s", f"output_dir={tmp_path / 'run'}", "cloze"]
    out = runner.invoke(
        cli,
        [
            *base,
            "build",
            "-o",
            str(items),
            "--eval-file",
            str(eval_file),
            "--vocab-file",
            str(vocab_file),
            "-n",
            "6",
            "--corpus-distractors",
            "2",
            "--stop-top",
            "5",
            "--proposer",
            tiny_model_dir,
            "--samples",
            "4",
        ],
    )
    assert out.exit_code == 0, out.output
    rows = [json.loads(line) for line in items.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["text"].count(GAP) == 1 for r in rows)

    out = runner.invoke(
        cli, [*base, "score", str(items), "-m", tiny_model_dir, "-m", tiny_model_dir, "--output", str(results)]
    )
    assert out.exit_code == 0, out.output
    assert "top-1 accuracy" in out.output and "chance (top-1)" in out.output
    assert len(results.read_text(encoding="utf-8").splitlines()) == 2 * len(rows)


def test_lenient_prefix_variants():
    from lfm3_cpt.cloze import strip_vh

    assert strip_vh("והילך") == strip_vh("הילך") == "ילך"
    assert strip_vh("התשיעי") == "תשיעי" and strip_vh("הן") == "הן"
    items = [
        {"id": 1, "text": f"וקיבל {GAP} וזרק", "answer": "הילך", "candidates": ["הילך", "והילך", "שחט"], "kinds": {}}
    ]
    scores = {"וקיבל הילך וזרק": -2.0, "וקיבל והילך וזרק": -1.0, "וקיבל שחט וזרק": -3.0}
    s = summarize(score_items(lambda t: [scores[x] for x in t], items))
    assert s["top1"] == 0.0 and s["top1_lenient"] == 1.0


def test_pmi_corrects_frequent_words():
    # "על" is likely everywhere; in context "בבית" fits better once its own probability is divided out
    items = [{"id": 1, "text": f"אלא {GAP} הבליעה", "answer": "בבית", "candidates": ["בבית", "על"], "kinds": {}}]
    lp = {"אלא בבית הבליעה": -10.0, "אלא על הבליעה": -9.0, "בבית": -8.0, "על": -3.0}
    raw = summarize(score_items(lambda t: [lp[x] for x in t], items))
    pmi = summarize(score_items(lambda t: [lp[x] for x in t], items, pmi=1.0))
    assert raw["top1"] == 0.0 and pmi["top1"] == 1.0  # -10+8 = -2 beats -9+3 = -6


def test_cli_cloze_score_pmi_columns(tmp_path: Path, tiny_model_dir: str):
    items = tmp_path / "items.jsonl"
    rows = [
        {
            "id": i,
            "text": f"the cat {GAP} on the mat",
            "answer": "sat",
            "candidates": ["sat", "ran", "is"],
            "kinds": {"ran": "corpus"},
            "freq": 3,
        }
        for i in range(3)
    ]
    items.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    base = ["-p", "cpu-debug", "-s", f"model_id={tiny_model_dir}", "-s", f"output_dir={tmp_path / 'run'}"]
    out = CliRunner().invoke(
        cli, [*base, "cloze", "score", str(items), "-m", tiny_model_dir, "--pmi", "0.5", "--pmi", "1"]
    )
    assert out.exit_code == 0, out.output
    assert "α=0.5" in out.output and "α=1" in out.output and "α=0" in out.output
