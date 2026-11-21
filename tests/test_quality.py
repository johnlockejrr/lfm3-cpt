"""Corpus cleaning: OCR garbage, boilerplate, ruined documents; Maltese prose must pass untouched."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from lfm3_cpt.cli import cli
from lfm3_cpt.quality import Settings, clean_documents, line_is_garbage

PROSE = [
    "Il-Gvern ħabbar li l-proġett il-ġdid se jibda s-sena d-dieħla f'Għawdex u fil-Belt.",
    "Fl-1964 Malta kisbet l-indipendenza, u 11-il sena wara saret repubblika.",
    "L-Assoċjazzjoni tal-Kunsilli Lokali ġiet informata b'dan u qalet li se tagħti l-parir tagħha.",
    "Ir-residenti għamlu petizzjoni biex jinbena ġnien pubbliku ġdid ħdejn l-iskola.",
]
COMMON = {"u", "a", "li", "ta", "se", "fi", "il", "ma", "fl", "l"}  # frequent short words of a Maltese corpus
WORDS = ["dar", "triq", "knisja", "baħar", "xemx", "ilma", "ħobż", "ktieb", "skola", "belt", "raħal", "ġnien"]


def test_prose_is_not_garbage():
    s = Settings()
    for line in PROSE:
        assert not line_is_garbage(line, COMMON, s), line
    assert not line_is_garbage("Onor. Edward Zammit Lewis, Ministru għall-Ġustizzja", COMMON, s)
    assert not line_is_garbage("KUMITAT PERMANENTI GĦALL-KONSIDERAZZJONI TA' ABBOZZI", COMMON, s)
    assert not line_is_garbage("11 ta' Marzu 2020", COMMON, s)


def test_ocr_lines_are_garbage():
    s = Settings()
    assert line_is_garbage("B FN wAġ Għaxar Snin Sezzjoni ia", COMMON, s)
    assert line_is_garbage("x | j ] k 7 ; q ~ z", COMMON, s)
    assert line_is_garbage("tA rQ", COMMON, s)


def test_clean_documents_boilerplate_and_dropping():
    header = "KUMITATI TAL-KAMRA TAD-DEPUTATI"
    notice = "Laqgħa Nru {n} tal-Kumitat għall-Konsiderazzjoni"
    # every document has its own prose (repeated prose would be boilerplate too)
    docs = [
        "\n".join([header, notice.format(n=i), *(f"{p} Il-{WORDS[i]} {WORDS[i + 1]}." for p in PROSE)])
        for i in range(6)
    ]
    ocr = [
        "B FN wAġ Għaxar Snin Sezzjoni ia",
        "rQ tA kX iZ jW",
        "Ħ q z ġ v k",
        "nV oT pE qR",
        "ZX cv BN mQ",
        "aB cD eF gH",
        "Jl kZ wQ xY",
        "QQ ZZ XX VV WW",
    ]  # OCR garbage varies from line to line
    docs.append("\n".join([*ocr, PROSE[0]]))  # mostly garbage
    out, rep = clean_documents(docs, Settings(boilerplate_docs=5, min_chars=50))
    assert rep.lines["boilerplate"] == 12  # the header and the numbered notice, 6 times each
    assert rep.lines["documents dropped"] == 1 and rep.docs_out == 6
    assert all(header not in d and "Laqgħa Nru" not in d for d in out)
    assert all(p in out[0] for p in PROSE)
    assert rep.lines["garbage"] == 8
    assert rep.examples["boilerplate"] and rep.examples["garbage"]


def test_cli_clean(tmp_path: Path):
    src = tmp_path / "x.train.jsonl"
    # a realistic share of short words: many documents, one OCR line
    docs = ["\n".join(f"{p} Il-{WORDS[i % 12]} u l-{WORDS[(i + 3) % 12]} {i}." for p in PROSE) for i in range(40)]
    docs[0] += "\nB FN wAġ Għaxar Snin Sezzjoni ia"
    src.write_text("".join(json.dumps({"text": d, "url": "u"}, ensure_ascii=False) + "\n" for d in docs))
    runner = CliRunner()
    res = runner.invoke(cli, ["clean", str(src), "--dry-run"])
    assert res.exit_code == 0, res.output and "garbage" in res.output
    assert not (tmp_path / "out").exists()
    res = runner.invoke(cli, ["clean", str(src), "-o", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    rows = [json.loads(line) for line in (tmp_path / "out" / "x.train.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and all("wAġ" not in r["text"] for r in rows)
    assert PROSE[1] in rows[0]["text"]
    assert runner.invoke(cli, ["clean", str(src)]).exit_code != 0  # neither -o nor --dry-run
