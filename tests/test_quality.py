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


# Lines from the first dry run on Korpus Malti (November 2026).
KEEP = [
    "Għaldaqstant, il‐KA tissuġġerixxi li l‐iżvilupp sostenibbli għandu jiddaħħal fil‐portafoll tal‐prim ministru.",
    "Għalkemm l‐impenn fuq kull livell huwa importanti, il‐Gvern għandu d‐dmir jagħmel il‐parti tiegħu.",
    "Fis-2.30 p.m. kontu hawn.",
    "Fit-3.12 p.m. il-Kumitat aġġorna.",
    'L-emenda "F" m\'għaddietx.',
    "L-Onor.Vella.",
    "Il-warrant…meta tiġi sospiża?",
    'Id-"diskors," l-"istil," l-"eteroglossija," id-"djaloġiżmu" u l-"polifonija" huma fost il-kunċetti li se nittratta.',
    "L-Ewwel Konferenza 'Dun Karm' mill- Prof. Ġużċ Aquilina B.A., LL.D., Ph.D.(Lond.), F.R.A.S.",
    "direttivi EU MCCAA",
    "Nokia PC Suite",
]
DROP = [
    "Gasju f'wib ioeqa So i ta'-Piletir l Mn 9 ie j har 9 k ) ji ; X i xa aa ik fa, kfal-ilti Misilmin FrRiieb.",
    "kk ik jk",
    "k k k",
    "Fl-istudju tieg ̇u, Debono jag ̇ti l-ismijiet ta' Ωew© persona©©i li kienu ©arrew dawn l-o©©etti sagri.",
    "Probabbilment m'hix wa ̇da ori©inali u twa ̇ ̇let wara.",
    "B FN wAġ Għaxar Snin Sezzjoni ia",
]


def test_dry_run_examples():
    s = Settings()
    for line in KEEP:
        assert not line_is_garbage(line, COMMON, s), line
    for line in DROP:
        assert line_is_garbage(line, COMMON, s), line


def test_typographic_hyphens():
    from lfm3_cpt.textnorm import Normalizer
    from lfm3_cpt.words import is_word

    assert Normalizer("NFC", ["typographic_hyphens"])("il‐Kummissjoni l‑ewwel") == "il-Kummissjoni l-ewwel"
    assert is_word("il‐Kummissjoni") and is_word("għall‑ewwel")


def test_no_boilerplate_keeps_repeated_formulas():
    docs = [f"Dan jgħid il-Mulej tal-eżerċti:\nIl-kapitlu tal-ktieb ta' {w}, b'kull vers tiegħu." for w in WORDS[:8]]
    _, with_rule = clean_documents(docs, Settings(boilerplate_docs=5, min_chars=10))
    out, without = clean_documents(docs, Settings(boilerplate_docs=5, min_chars=10, boilerplate=False))
    assert with_rule.lines["boilerplate"] == 8 and without.lines["boilerplate"] == 0
    assert all(d.startswith("Dan jgħid il-Mulej") for d in out)
