"""Normalization, local corpora, minimal-pair files, tokenizer check, language detection and ranking."""

import json
import os
import time

import pytest

from lfm3_cpt.chat_eval import answer_language, in_target_language, summarize_generations
from lfm3_cpt.config import build_config, replace
from lfm3_cpt.corpus_files import expand, load_local_corpus, read_documents, read_pairs
from lfm3_cpt.data import is_prepared, load_data, prepare_data
from lfm3_cpt.evaluation import rank_candidates, sentence_logprobs
from lfm3_cpt.textnorm import Normalizer, normalize
from lfm3_cpt.tokcheck import ENGLISH_REFERENCE, measure, tokenizer_report, verdict

QUIET = {"log": lambda *_: None}

POINTED = "בְּרֵאשִׁ֖ית"  # with niqqud, dagesh, shin dot and a cantillation accent
LATIN = {
    "caesar.txt": "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam qui "
    "ipsorum lingua Celtae, nostra Galli appellantur.\n\nHi omnes lingua, institutis, legibus inter se differunt.",
    "cicero.txt": "Quo usque tandem abutere, Catilina, patientia nostra?\n\nQuam diu etiam furor iste tuus nos eludet?",
    "vergil.txt": "Arma virumque cano, Troiae qui primus ab oris\nItaliam, fato profugus, Laviniaque venit\nlitora.",
}


# --- normalization -------------------------------------------------------------------------------------------


def test_strip_hebrew_marks():
    assert normalize(POINTED, strip=["hebrew_cantillation", "hebrew_points"]) == "בראשית"
    only_accents = normalize(POINTED, strip=["hebrew_cantillation"])
    assert "֖" not in only_accents and "ְ" in only_accents  # vowels kept, accent gone
    assert normalize("כ־ה׃", strip=["hebrew_points"]) == "כ־ה׃"  # maqaf and sof pasuq are kept


def test_strip_arabic_and_latin():
    assert normalize("كَتَبَ", strip=["arabic_diacritics"]) == "كتب"
    assert normalize("كـتـب", strip=["arabic_tatweel"]) == "كتب"
    assert normalize("Rōmānī ăb", strip=["latin_macrons"]) == "Romani ab"
    assert normalize("é", "NFC") == "é"
    assert normalize("é", "none") == "é"


def test_normalizer_validation():
    with pytest.raises(ValueError):
        Normalizer("NFX")
    with pytest.raises(ValueError):
        Normalizer("NFC", ["klingon_marks"])
    with pytest.raises(ValueError):
        build_config(overrides={"strip_marks": ["nope"]})
    assert Normalizer("none").is_identity


# --- local corpora -------------------------------------------------------------------------------------------


@pytest.fixture()
def latin_dir(tmp_path):
    d = tmp_path / "latin"
    d.mkdir()
    for name, text in LATIN.items():
        (d / name).write_text(text, encoding="utf-8")
    (d / "extra.jsonl").write_text(
        "\n".join(json.dumps({"text": t, "id": i}) for i, t in enumerate(["Roma aeterna est.", "Senatus populusque."]))
        + "\n\n",
        encoding="utf-8",
    )
    return d


def test_expand_and_repeat(latin_dir):
    specs = expand([f"{latin_dir}/*.txt@3", str(latin_dir / "extra.jsonl")])
    assert [s.path.name for s in specs] == ["caesar.txt", "cicero.txt", "vergil.txt", "extra.jsonl"]
    assert [s.repeat for s in specs] == [3, 3, 3, 1]
    with pytest.raises(FileNotFoundError):
        expand([f"{latin_dir}/*.md"])


@pytest.mark.parametrize("split, n", [("file", 1), ("paragraph", 2), ("line", 3)])
def test_doc_split(latin_dir, split, n):
    cfg = build_config(overrides={"doc_split": split})
    assert (
        len([d for d in read_documents(latin_dir / ("vergil.txt" if split == "line" else "caesar.txt"), cfg) if d]) == n
    )


def test_jsonl_documents(latin_dir):
    assert read_documents(latin_dir / "extra.jsonl", build_config()) == ["Roma aeterna est.", "Senatus populusque."]
    (latin_dir / "bad.jsonl").write_text('{"text": "a"}\n{oops\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_documents(latin_dir / "bad.jsonl", build_config())


def test_holdout_whole_files(latin_dir):
    cfg = build_config(
        overrides={"train_files": [f"{latin_dir}/*.txt@2"], "eval_fraction": 0.3, "doc_split": "paragraph"}
    )
    corpus = load_local_corpus(cfg, **QUIET)
    held = corpus.info["eval_files"]
    assert len(held) == 1
    held_text = (latin_dir / held[0].split("/")[-1]).read_text()
    assert all(d in held_text for d in corpus.eval)
    assert not any(d in held_text for d in corpus.train)  # nothing of the held-out file is trained on
    assert len(corpus.train) == 2 * (5 - len(corpus.eval))  # upsampled x2


def test_holdout_documents_before_upsampling(latin_dir):
    cfg = build_config(
        overrides={"train_files": [f"{latin_dir}/caesar.txt@4"], "doc_split": "paragraph", "eval_fraction": 0.5}
    )
    corpus = load_local_corpus(cfg, **QUIET)  # one file: falls back to document holdout
    assert corpus.info["eval_source"] == "held-out documents"
    assert len(corpus.eval) == 1 and corpus.eval[0] not in corpus.train
    assert len(corpus.train) == 4  # the remaining paragraph, repeated 4 times


def test_eval_files_ood_and_normalization(latin_dir, tmp_path):
    he = tmp_path / "he.txt"
    he.write_text(POINTED + " " + POINTED, encoding="utf-8")
    cfg = build_config(
        overrides={
            "train_files": [f"{latin_dir}/*.txt"],
            "eval_files": [str(latin_dir / "extra.jsonl")],
            "ood_files": [str(he)],
            "strip_marks": ["hebrew_points", "hebrew_cantillation"],
            "file_min_chars": 10,
        }
    )
    corpus = load_local_corpus(cfg, **QUIET)
    assert corpus.eval == ["Roma aeterna est.", "Senatus populusque."]
    assert corpus.ood == ["בראשית בראשית"]
    assert len(corpus.train) == 3


def test_prepare_local_with_pairs_and_cache(latin_dir, tmp_path, tokenizer):
    pairs = tmp_path / "latin_pairs.tsv"
    pairs.write_text(
        "sentence_good\tsentence_bad\tphenomenon\n# comment\n"
        "puella bona cantat\tpuella bonus cantat\tagreement\npueri currunt\tpueri currit\n",
        encoding="utf-8",
    )
    cfg = build_config(
        "cpu-debug",
        overrides={
            "data_dir": str(tmp_path / "data"),
            "target_language": "la",
            "train_files": [f"{latin_dir}/*.txt"],
            "grammar_files": [str(pairs)],
            "eval_fraction": 0.3,
        },
    )
    bundle = prepare_data(cfg, tokenizer, **QUIET)
    assert bundle.metadata["stats"]["source"] == "local files"
    assert "tokenizer_verdict" in bundle.metadata["stats"]
    assert list(bundle.grammar["linguistic_phenomenon"]) == ["agreement", "latin_pairs"]
    assert is_prepared(cfg)
    # editing a corpus file invalidates the cache
    time.sleep(0.01)
    (latin_dir / "cicero.txt").write_text(LATIN["cicero.txt"] + "\n\nO tempora, o mores!", encoding="utf-8")
    os.utime(latin_dir / "cicero.txt")
    assert not is_prepared(cfg)
    # switching to a language without pairs: no grammar split survives from the previous preparation
    cfg2 = replace(cfg, grammar_files=[])
    assert not cfg2.grammar_enabled
    assert prepare_data(cfg2, tokenizer, **QUIET).grammar is None
    assert load_data(cfg2).grammar is None


def test_turblimp_only_for_turkish():
    assert build_config().grammar_enabled
    assert not build_config(overrides={"target_language": "he"}).grammar_enabled
    assert build_config(overrides={"target_language": "arc", "grammar_files": ["x.tsv"]}).grammar_enabled
    with pytest.raises(ValueError):
        build_config(overrides={"target_language": "xx"})


def test_example_latin_pairs_header_after_comments():
    from pathlib import Path

    rows = read_pairs(Path(__file__).parent.parent / "examples" / "latin-pairs.tsv", Normalizer())
    assert len(rows) == 8 and "phenomenon" not in {r["linguistic_phenomenon"] for r in rows}


def test_read_pairs_jsonl_and_errors(tmp_path):
    norm = Normalizer()
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        json.dumps({"good": "a b", "bad": "b a", "phenomenon": "order"})
        + "\n"
        + json.dumps({"sentence_good": "x", "sentence_bad": "y"})
        + "\n"
    )
    rows = read_pairs(p, norm)
    assert [r["linguistic_phenomenon"] for r in rows] == ["order", "pairs"]
    same = tmp_path / "same.tsv"
    same.write_text("ab\tab\n")
    with pytest.raises(ValueError, match="identical"):
        read_pairs(same, norm)
    bad = tmp_path / "bad.tsv"
    bad.write_text("only one column\n")
    with pytest.raises(ValueError, match="good<TAB>bad"):
        read_pairs(bad, norm)


# --- tokenizer check -----------------------------------------------------------------------------------------


def test_tokenizer_report(tokenizer):
    eng = measure(tokenizer, [ENGLISH_REFERENCE])
    assert eng["words"] == len(ENGLISH_REFERENCE.split()) and eng["tokens"] > 0
    # the tiny test tokenizer was trained on Turkish-like words: Hebrew falls back to bytes
    report = tokenizer_report(tokenizer, ["בראשית ברא אלהים את השמים ואת הארץ"] * 5)
    assert report["target"]["chars_per_token"] < 1.0
    assert report["verdict"].startswith("poor")
    assert verdict(4.0) == "good" and verdict(2.5).startswith("fair")


# --- answer language -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, target, expected",
    [
        ("Gallia est omnis divisa in partes tres.", "la", True),
        ("Gaul is divided into three parts.", "la", False),
        ("בקדמין ברא יי ית שמיא וית ארעא", "arc", True),
        ("בראשית ברא אלהים", "he", True),
        ("In the beginning", "he", False),
        ("كتب الطالب الدرس", "ar", True),
        ("ܒܪܫܝܬ ܐܝܬܘܗܝ ܗܘܐ", "syc", True),
        ("ࠁࠓࠔࠉࠕ ࠁࠓࠀ", "sam", True),
        ("anything", "", None),
    ],
)
def test_in_target_language(text, target, expected):
    assert in_target_language(text, target) is expected


def test_summarize_for_aramaic():
    answers = [
        {"text": "וית ארעא", "finished": True},
        {"text": "and the earth", "finished": True},
        {"text": "", "finished": False},
    ]
    s = summarize_generations(answers, ["וית ארעא"] * 3, target="arc")
    assert s["target_rate"] == pytest.approx(1 / 3) and s["english_rate"] == pytest.approx(1 / 3)
    assert "turkish_rate" not in s
    assert s["languages"] == ["arc", "en", "empty"]
    assert answer_language("Bu ev çok güzel.", "tr") == "tr"


# --- ranking -------------------------------------------------------------------------------------------------


def test_rank_candidates(model, tokenizer):
    text = "bu ev ___ güzel."
    cands = ["çok", "daha", "en", "çok"]
    rows = rank_candidates(model, tokenizer, text, cands, batch_size=2)
    assert [r["rank"] for r in rows] == [1, 2, 3]  # duplicate removed
    scores = sentence_logprobs(model, tokenizer, [text.replace("___", c) for c in ["çok", "daha", "en"]])
    assert [r["logprob"] for r in rows] == sorted(scores, reverse=True)
    assert rows[0]["delta"] == 0 and all(r["delta"] <= 0 for r in rows)
    by_mean = rank_candidates(model, tokenizer, text, cands, by="mean")
    assert [r["mean_logprob"] for r in by_mean] == sorted((r["mean_logprob"] for r in by_mean), reverse=True)
    with pytest.raises(ValueError, match="exactly once"):
        rank_candidates(model, tokenizer, "no gap", cands)
    with pytest.raises(ValueError, match="exactly once"):
        rank_candidates(model, tokenizer, "___ ___", cands)


# --- local SFT files -----------------------------------------------------------------------------------------


def test_local_sft_file(tmp_path, tiny_config, tokenizer):
    from lfm3_cpt.sft_data import prepare_sft_data

    rows = [{"instruction": "Targumize:", "input": f"פסוק {i} " + POINTED, "output": f"תרגום {i}"} for i in range(40)]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    cfg = replace(
        tiny_config,
        sft_dataset=str(path),
        sft_max_train=-1,
        sft_max_eval=-1,
        sft_max_length=4096,
        strip_marks=["hebrew_points", "hebrew_cantillation"],
        target_language="arc",
    )
    logs = []
    bundle = prepare_sft_data(cfg, tokenizer, log=logs.append)
    assert len(bundle.eval) == 2 and len(bundle.train) == 38  # 5% held out (no test split in a local file)
    assert "בראשית" in bundle.train[0]["prompt"][0]["content"]  # normalized
    assert any("filter skipped" in m for m in logs)
