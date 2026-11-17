"""Train the tiny model for a few steps on CPU through the real pipeline, then drive the CLI on the result."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from lfm3_cpt.cli import cli
from lfm3_cpt.train import run_training

pytestmark = pytest.mark.slow


@pytest.fixture()
def trained_run(tiny_config, fake_loaders, tmp_path):
    console = Console(file=open(tmp_path / "console.txt", "w"), width=140)
    results = run_training(tiny_config, console, merge=True, loaders=fake_loaders)
    return tiny_config, results


def test_run_training_outputs(trained_run):
    cfg, results = trained_run
    out = Path(cfg.output_dir)
    for rel in (
        "config.toml",
        "baseline.json",
        "results.json",
        "metrics.jsonl",
        "adapter/adapter_config.json",
        "merged/config.json",
        "merged/cpt_results.json",
    ):
        assert (out / rel).exists(), rel
    assert results["train"]["global_step"] == cfg.max_steps
    assert results["train"]["tokens_seen"] > 0
    for block in ("baseline", "final"):
        assert results[block]["wiki"]["perplexity"] > 1
        assert results[block]["ood"]["tokens"] > 0
        assert results[block]["grammar"]["n_pairs"] == 9
    assert {"fixed", "broken", "examples"} <= set(results["grammar_comparison"])
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert any("loss" in r and "ppl" in r for r in records)
    # both eval sets are tracked during training (the Trainer logs each one as its own record)
    assert any("eval_wiki_loss" in r and "eval_wiki_ppl" in r for r in records)
    assert any("eval_ood_loss" in r for r in records)
    assert any(p.name.startswith("checkpoint-") for p in (out / "checkpoints").iterdir())


def test_resume_reuses_baseline(trained_run, fake_loaders, tmp_path):
    from lfm3_cpt.config import replace

    cfg, _ = trained_run
    cfg = replace(cfg, max_steps=4)
    console = Console(file=open(tmp_path / "console2.txt", "w"), width=140)
    results = run_training(cfg, console, resume=True, loaders=fake_loaders)
    assert results["train"]["global_step"] == 4
    log = (tmp_path / "console2.txt").read_text()
    assert "Re-using cached baseline" in log and "Resuming from" in log


def _base_args(cfg):
    return [
        "--profile",
        "cpu-debug",
        "--set",
        f"model_id={cfg.model_id}",
        "--set",
        "trust_remote_code=false",
        "--set",
        f"data_dir={cfg.data_dir}",
        "--set",
        f"output_dir={cfg.output_dir}",
        "--set",
        "train_articles=30",
        "--set",
        "eval_articles=4",
        "--set",
        "ood_pages=4",
        "--set",
        "min_chars=100",
        "--set",
        "max_length=64",
        "--set",
        "phenomena=binding,passives,scrambling",
        "--set",
        "grammar_batch_size=4",
    ]


def test_cli_on_trained_run(trained_run, tmp_path):
    cfg, _ = trained_run
    runner = CliRunner()
    out = Path(cfg.output_dir)

    res = runner.invoke(cli, ["report", str(out)])
    assert res.exit_code == 0, res.output
    assert "TurBLiMP grammar accuracy" in res.output

    json_out = tmp_path / "eval.json"
    res = runner.invoke(cli, [*_base_args(cfg), "evaluate", str(out / "adapter"), "--json-out", str(json_out)])
    assert res.exit_code == 0, res.output
    data = json.loads(json_out.read_text())
    assert data["adapter"] == str(out / "adapter") and data["grammar"]["n_pairs"] == 9

    res = runner.invoke(cli, [*_base_args(cfg), "evaluate", str(out / "merged"), "--no-grammar"])
    assert res.exit_code == 0, res.output

    res = runner.invoke(
        cli, [*_base_args(cfg), "merge", str(out / "adapter"), "-o", str(tmp_path / "m2"), "--dtype", "fp32"]
    )
    assert res.exit_code == 0, res.output
    assert (tmp_path / "m2" / "config.json").exists()

    res = runner.invoke(
        cli, [*_base_args(cfg), "generate", str(out / "merged"), "--prompt", "bu ev", "--max-new-tokens", "5"]
    )
    assert res.exit_code == 0, res.output
