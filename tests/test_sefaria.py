"""Sefaria library export: selection, edition choice, cleaning, chapters, tractate holdout, pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.sefaria import (
    Cleaner,
    Selector,
    build_corpus,
    iter_segments,
    library_stats,
    make_pairs,
    split_ref,
    tractate_key,
)

M1 = "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסים לאכול בתרומתן עד סוף האשמורה הראשונה דברי ר' אליעזר"
M2 = "וחכמים אומרים עד חצות רבן גמליאל אומר עד שיעלה עמוד השחר מעשה שבאו בניו מבית המשתה"
M3 = "בשחר מברך שתים לפניה ואחת לאחריה ובערב מברך שתים לפניה ושתים לאחריה אחת ארוכה ואחת קצרה"


def _rows() -> list[dict]:
    def row(ref, text, version="Torat Emet", cat="Mishnah", quality="professional"):
        meta = {"ref": ref, "versionTitle": version, "lang": "he", "docCategory": cat, "dataQuality": quality}
        return {"text": text, "metadata": meta}

    rows = []
    for i, t in enumerate(["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah"]):
        for ch in (1, 2, 10):
            for seg in (2, 1, 10):  # out of order on purpose
                rows.append(row(f"Mishnah {t} {ch}:{seg}", f"{M1} {i} {ch} {seg}:"))
    # a vocalized second edition of one tractate, smaller: must not be chosen, nor duplicated
    rows.append(row("Mishnah Berakhot 1:1", "מֵאֵימָתַי קוֹרִין", version="Vocalized", quality="user"))
    rows.append(row("Tosefta Berakhot (Lieberman) 1:1", M2 + ' רז"ל', version="Lieberman", cat="Tosefta"))
    rows.append(row("Tosefta Berakhot (Lieberman) 1:2", M3, version="Lieberman", cat="Tosefta"))
    rows.append(row("Bartenura on Mishnah Berakhot 1:1:1", "פירוש", cat="Mishnah/Commentary"))
    rows.append(row("Mishnah Berurah 1:1", "הלכה", cat="Halakhah"))
    rows.append(row("Shabbat 2a:1", "גמרא", cat="Talmud"))
    rows.append(row("Shabbat 2a:2", "גמרא", cat="Talmud"))
    rows.append(row("Teshuvot HaRashba 1:1", M3, cat="Responsa"))
    rows.append(row("Mishnah Yoma 5:1", "An English line that slipped into the Hebrew text."))
    return rows


@pytest.fixture
def library(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in _rows():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.write("not json\n")
    return path


def test_split_ref_and_order():
    assert split_ref("Mishnah Berakhot 1:2")[:2] == ("Mishnah Berakhot", ("1",))
    assert split_ref("Shabbat 2a:3")[:2] == ("Shabbat", ("2a",))
    assert split_ref("Igrot Moshe, Orach Chaim 1:5")[0] == "Igrot Moshe, Orach Chaim"
    assert split_ref("Pirkei Avot")[0] == "Pirkei Avot"
    keys = sorted(["2b:1", "10a:1", "2a:2", "2a:1"], key=lambda a: split_ref("X " + a)[2])
    assert keys == ["2a:1", "2a:2", "2b:1", "10a:1"]


def test_tractate_key_joins_parallels():
    assert tractate_key("Mishnah Berakhot") == tractate_key("Tosefta Berakhot (Lieberman)") == "berakhot"


def test_cleaner_native_marks():
    c = Cleaner()
    assert c('אמר ר\' יהודה: "הלכה כרבי"') == "אמר ר׳ יהודה: הלכה כרבי"
    assert c('רז"ל אמרו ה’ בכסלו') == "רז״ל אמרו ה׳ בכסלו"
    assert c("שליח צבור ש“ץ") == "שליח צבור ש״ץ"
    assert c("ﬠל") == "על"  # presentation form
    assert c('אמר<sup>1</sup><i class="footnote">הערה</i> רבי‏ יוסי @11 :') == "אמר רבי יוסי:"
    assert c("An English sentence with עברית") == ""
    assert c.stats["' -> geresh ׳"] == 2 and c.stats["quotation marks removed"] == 2


def test_cleaner_options():
    c = Cleaner(strip_punctuation=True, drop_editorial=True)
    assert c('אמר רבי יוסי (ברכות ב.) [נ"א: רבה], ואם לאו - פטור:') == "אמר רבי יוסי ואם לאו פטור:"
    assert Cleaner(native_marks=False)("ר' יהודה") == "ר' יהודה"


def test_selector_presets():
    s = Selector.from_preset("tannaitic")
    assert s.matches("Mishnah Berakhot", "Mishnah")
    assert s.matches("Tosefta Berakhot (Lieberman)", "Tosefta")
    assert s.matches("Pirkei Avot", "Mishnah")
    assert not s.matches("Bartenura on Mishnah Berakhot", "Mishnah")
    assert not s.matches("Mishnah Berurah", "Halakhah")
    assert not s.matches("Shabbat", "Talmud")
    r = Selector.from_preset("responsa")
    assert r.matches("Teshuvot HaRashba", "Responsa") and not r.matches("Mishnah Berakhot", "Mishnah")
    with pytest.raises(ValueError):
        Selector()


def test_build_corpus(library: Path):
    result = build_corpus(
        iter_segments(library), Selector.from_preset("tannaitic"), min_chars=10, eval_match=["Berakhot"]
    )
    books = {b.book: b for b in result.books}
    assert set(books) == {f"Mishnah {t}" for t in ["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah"]} | {
        "Tosefta Berakhot (Lieberman)"
    }
    assert books["Mishnah Berakhot"].version == "Torat Emet" and books["Mishnah Berakhot"].versions == 2
    # Mishnah and Tosefta Berakhot are held out together
    assert books["Mishnah Berakhot"].split == books["Tosefta Berakhot (Lieberman)"].split == "eval"
    assert books["Mishnah Shabbat"].split == "train"
    # one document per chapter, chapters and segments in reading order
    shabbat = [d for d in result.train if d["source"].startswith("Mishnah Shabbat")]
    assert [d["source"] for d in shabbat] == ["Mishnah Shabbat 1", "Mishnah Shabbat 2", "Mishnah Shabbat 10"]
    assert [ln.split()[-1] for ln in shabbat[0]["text"].splitlines()] == ["1:", "2:", "10:"]
    assert all("ר׳ אליעזר" in d["text"] for d in shabbat)
    assert any("רז״ל" in d["text"] for d in result.eval)
    assert all("English" not in d["text"] for d in result.train + result.eval)


def test_build_corpus_fraction_keeps_training(library: Path):
    result = build_corpus(iter_segments(library), Selector.from_preset("tannaitic"), min_chars=10, eval_fraction=0.2)
    splits = {b.split for b in result.books}
    assert splits == {"train", "eval"}
    assert result.eval and result.train


def test_library_stats(library: Path):
    st = library_stats(iter_segments(library), Selector.from_preset("tannaitic"))
    assert st.lines == len(_rows())
    assert st.categories["Talmud"] > 0
    assert len(st.books["Mishnah Berakhot"]) == 2


def test_make_pairs():
    docs = [{"text": "\n".join([M1, M2, M3] * 3)}]
    pairs = make_pairs(docs, n=10, seed=1)
    kinds = {p["phenomenon"] for p in pairs}
    assert kinds == {"word_order", "word_substitution"}
    for p in pairs:
        assert p["sentence_good"] != p["sentence_bad"]
        assert sorted(p["sentence_good"].split()) == sorted(p["sentence_bad"].split()) or (
            p["phenomenon"] == "word_substitution"
        )


def test_cli_sefaria(library: Path, tmp_path: Path):
    runner = CliRunner()
    out = runner.invoke(cli, ["sefaria", "--file", str(library), "--preset", "tannaitic", "--stats"])
    assert out.exit_code == 0, out.output
    assert "Mishnah Berakhot" in out.output and "Bartenura" not in out.output

    prefix = tmp_path / "corp" / "tannaitic"
    pairs = tmp_path / "corp" / "pairs.jsonl"
    args = ["sefaria", "--file", str(library), "--preset", "tannaitic", "-o", str(prefix), "--min-chars", "10"]
    out = runner.invoke(cli, [*args, "--eval-match", "Berakhot", "--pairs", str(pairs), "--pairs-count", "6"])
    assert out.exit_code == 0, out.output
    assert Path(f"{prefix}.train.jsonl").exists() and Path(f"{prefix}.eval.jsonl").exists()
    manifest = Path(f"{prefix}.books.tsv").read_text(encoding="utf-8")
    assert "Mishnah Berakhot\teval" in manifest
    assert pairs.read_text(encoding="utf-8").count("\n") > 0

    out = runner.invoke(cli, ["sefaria", "--file", str(library), "--title", "^Nothing", "-o", str(prefix)])
    assert out.exit_code != 0 and "nothing selected" in out.output


def test_budget_keeps_whole_books(library: Path):
    full = build_corpus(iter_segments(library), Selector.from_preset("tannaitic"), min_chars=10, eval_fraction=0)
    total = sum(len(d["text"]) for d in full.train)
    part = build_corpus(
        iter_segments(library),
        Selector.from_preset("tannaitic"),
        min_chars=10,
        eval_fraction=0,
        max_train_chars=total // 3,
    )
    used = [b for b in part.books if b.split == "train"]
    assert 0 < len(used) < len(part.books)
    assert {d["source"].rsplit(" ", 1)[0] for d in part.train} <= {b.book for b in used}
    assert sum(len(d["text"]) for d in part.train) <= 1.2 * total // 3 + 1


def test_cli_eval_only(library: Path, tmp_path: Path):
    prefix = tmp_path / "resp"
    args = ["sefaria", "--file", str(library), "--preset", "tannaitic", "-o", str(prefix), "--min-chars", "10"]
    out = CliRunner().invoke(cli, [*args, "--eval-fraction", "0.2", "--max-chars", "0"])
    assert out.exit_code == 0, out.output
    assert not Path(f"{prefix}.train.jsonl").exists() and Path(f"{prefix}.eval.jsonl").exists()
