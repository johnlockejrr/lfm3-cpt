"""End to end on a local corpus (Latin files + minimal pairs), then the tokenizer-check / rank CLI."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from lfm3_cpt.cli import cli
from lfm3_cpt.config import build_config
from lfm3_cpt.train import run_training

from .test_languages import LATIN

pytestmark = pytest.mark.slow

PAIRS = "puella bona cantat\tpuella bonus cantat\tagreement\npueri currunt\tpueri currit\tagreement\n"


@pytest.fixture()
def latin_run(tmp_path, tiny_model_dir):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in LATIN.items():
        (corpus / name).write_text(text * 3, encoding="utf-8")
    (tmp_path / "pairs.tsv").write_text(PAIRS, encoding="utf-8")
    overrides = {
        "model_id": tiny_model_dir,
        "trust_remote_code": False,
        "target_language": "la",
        "train_files": [f"{corpus}/*.txt@2"],
        "doc_split": "paragraph",
        "eval_fraction": 0.34,
        "grammar_files": [str(tmp_path / "pairs.tsv")],
        "strip_marks": ["latin_macrons"],
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "run"),
        "max_length": 64,
        "max_steps": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "grammar_batch_size": 4,
    }
    cfg = build_config("cpu-debug", overrides=overrides)
    console = Console(file=open(tmp_path / "log.txt", "w"), width=140)
    results = run_training(cfg, console, merge=True)
    args = ["-p", "cpu-debug"]
    for k, v in overrides.items():
        args += ["-s", f"{k}={','.join(v) if isinstance(v, list) else v}"]
    return cfg, results, args, tmp_path


def test_local_run(latin_run):
    cfg, results, _, tmp = latin_run
    assert results["data"]["stats"]["source"] == "local files"
    assert results["data"]["corpus"]["eval_source"] == "held-out files"
    assert results["final"]["grammar"]["n_pairs"] == 2
    assert results["final"]["ood"] is None  # no ood_files
    assert "chars/token" in (tmp / "log.txt").read_text()


def test_tokenizer_check_cli(latin_run):
    _, _, args, tmp = latin_run
    runner = CliRunner()
    res = runner.invoke(cli, [*args, "tokenizer-check"])  # measures train_files
    assert res.exit_code == 0, res.output
    assert "characters per token" in res.output and "verdict" in res.output
    res = runner.invoke(cli, [*args, "tokenizer-check", "--text", "בראשית ברא אלהים את השמים"])
    assert res.exit_code == 0 and "poor" in res.output
    res = runner.invoke(cli, [*args, "tokenizer-check", str(tmp / "nothing*.txt")])
    assert res.exit_code == 1 and "no files match" in res.output


def test_rank_cli(latin_run):
    cfg, _, args, tmp = latin_run
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            *args,
            "rank",
            "--text",
            "Arma virumque ___, Troiae qui primus ab oris",
            "--candidate",
            "cano",
            "--candidate",
            "canō",
            "--candidate",
            "canto",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "merged" in res.output  # defaults to this run's CPT model
    assert res.output.count("cano") >= 1 and "canō" not in res.output.split("candidate")[-1]  # macron stripped

    batch = tmp / "gaps.jsonl"
    batch.write_text(
        json.dumps({"id": "aen-1-1", "text": "Arma ___ cano", "candidates": ["virumque", "virosque"]})
        + "\n"
        + json.dumps(
            {"id": "bg-1-1", "text": "Gallia est omnis ___ in partes tres", "candidates": ["divisa", "divisus"]}
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp / "ranked.jsonl"
    res = runner.invoke(cli, [*args, "rank", "--input", str(batch), "--output", str(out), "--by", "mean"])
    assert res.exit_code == 0, res.output
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["id"] for x in lines] == ["aen-1-1", "bg-1-1"]
    assert lines[0]["by"] == "mean" and lines[0]["ranking"][0]["rank"] == 1

    res = runner.invoke(cli, [*args, "rank", "--text", "no gap here", "--candidate", "x"])
    assert res.exit_code == 1 and "exactly once" in res.output
    res = runner.invoke(cli, [*args, "rank", "--text", "a ___"])
    assert res.exit_code == 2


def test_generate_needs_prompt_for_other_languages(latin_run):
    _, _, args, _ = latin_run
    runner = CliRunner()
    res = runner.invoke(cli, [*args, "generate"])
    assert res.exit_code == 2 and "--prompt" in res.output
    res = runner.invoke(cli, [*args, "generate", "--prompt", "Gallia est", "--max-new-tokens", "3"])
    assert res.exit_code == 0, res.output


def test_report_and_evaluate_local(latin_run):
    cfg, _, args, tmp = latin_run
    runner = CliRunner()
    res = runner.invoke(cli, ["report", cfg.output_dir])
    assert res.exit_code == 0 and "minimal-pair accuracy" in res.output and "agreement" in res.output
    res = runner.invoke(
        cli, [*args, "evaluate", str(Path(cfg.output_dir) / "merged"), "--json-out", str(tmp / "e.json")]
    )
    assert res.exit_code == 0, res.output
    assert json.loads((tmp / "e.json").read_text())["grammar"]["n_pairs"] == 2
