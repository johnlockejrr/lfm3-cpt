"""Stage 2 end to end on CPU: CPT (no merge) -> SFT from the CPT adapter (auto-merge) -> CLI on the results."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from lfm3_cpt.cli import cli
from lfm3_cpt.config import replace
from lfm3_cpt.sft import resolve_start, run_sft
from lfm3_cpt.train import run_training

from .test_end_to_end import _base_args

pytestmark = pytest.mark.slow


@pytest.fixture()
def cpt_then_sft(tiny_config, fake_loaders, fake_sft_loader, tmp_path):
    console = Console(file=open(tmp_path / "console.txt", "w"), width=140)
    run_training(tiny_config, console, merge=False, loaders=fake_loaders)
    assert not (Path(tiny_config.output_dir) / "merged").exists()
    results = run_sft(tiny_config, console, start="cpt", merge=True, loader=fake_sft_loader)
    return tiny_config, results, tmp_path / "console.txt"


def test_sft_stage_outputs(cpt_then_sft):
    cfg, results, log = cpt_then_sft
    run = Path(cfg.output_dir)
    sft = run / "sft"
    # the CPT adapter was merged automatically so SFT starts from a standalone model
    assert (run / "merged" / "config.json").exists()
    assert results["start_model"] == str((run / "merged").resolve())
    for rel in (
        "config.toml",
        "baseline.json",
        "results.json",
        "metrics.jsonl",
        "generations.jsonl",
        "adapter/adapter_config.json",
        "merged/config.json",
    ):
        assert (sft / rel).exists(), rel
    adapter_cfg = json.loads((sft / "adapter" / "adapter_config.json").read_text())
    assert adapter_cfg["base_model_name_or_path"] == results["start_model"]
    assert adapter_cfg["r"] == cfg.sft_lora_r

    assert results["stage"] == "sft" and results["train"]["global_step"] == cfg.sft_max_steps
    for block in ("baseline", "final"):
        b = results[block]
        assert b["answer"]["tokens"] > 0
        assert b["generation"]["n"] == cfg.sft_gen_samples
        assert "answers" not in b  # raw answers live in generations.jsonl, not results.json
        assert b["lm"]["wiki"]["perplexity"] > 1 and b["lm"]["grammar"]["n_pairs"] == 9  # forgetting check
    gens = [json.loads(x) for x in (sft / "generations.jsonl").read_text().splitlines()]
    assert len(gens) == cfg.sft_gen_samples
    assert {"prompt", "reference", "before", "after", "after_language"} <= set(gens[0])
    records = [json.loads(x) for x in (sft / "metrics.jsonl").read_text().splitlines()]
    assert any("eval_loss" in r for r in records)
    assert "Before SFT" in log.read_text()


def test_resolve_start(cpt_then_sft, tmp_path):
    cfg, results, _ = cpt_then_sft
    assert resolve_start(cfg, "base") == cfg.model_id
    assert resolve_start(cfg, "cpt") == results["start_model"]
    assert resolve_start(cfg, str(Path(cfg.output_dir) / "adapter")) == results["start_model"]
    with pytest.raises(FileNotFoundError):
        resolve_start(replace(cfg, output_dir=str(tmp_path / "empty")), "cpt")


def test_sft_resume_reuses_baseline(cpt_then_sft, fake_sft_loader, tmp_path):
    cfg, _, _ = cpt_then_sft
    log = tmp_path / "resume.txt"
    res = run_sft(
        replace(cfg, sft_max_steps=3), Console(file=open(log, "w"), width=140), resume=True, loader=fake_sft_loader
    )
    assert res["train"]["global_step"] == 3
    text = log.read_text()
    assert "Re-using cached baseline" in text and "Resuming from" in text


def test_sft_cli(cpt_then_sft, tmp_path):
    cfg, _, _ = cpt_then_sft
    runner = CliRunner()
    run = Path(cfg.output_dir)

    # report shows both stages for a run directory
    res = runner.invoke(cli, ["report", str(run)])
    assert res.exit_code == 0, res.output
    assert "minimal-pair accuracy" in res.output and "answers in Turkish" in res.output

    # control experiment from the CLI: SFT on the original model, cached SFT data, no LM eval
    other = tmp_path / "sft-only"
    args = [a if a != f"output_dir={cfg.output_dir}" else f"output_dir={other}" for a in _base_args(cfg)]
    res = runner.invoke(cli, [*args, "sft", "--from", "base", "--no-lm-eval", "--max-steps", "1"])
    assert res.exit_code == 0, res.output
    sft_only = json.loads((other / "sft" / "results.json").read_text())
    assert sft_only["start_model"] == cfg.model_id and "lm" not in sft_only["final"]

    # missing CPT run -> clean error, not a traceback
    res = runner.invoke(cli, [*args[:-2], "--set", f"output_dir={tmp_path / 'nothing'}", "sft"])
    assert res.exit_code == 1 and "no CPT result" in res.output

    # chat REPL picks the SFT model by default
    res = runner.invoke(cli, [*_base_args(cfg), "chat", "--max-new-tokens", "4"], input="merhaba\n/reset\n/exit\n")
    assert res.exit_code == 0, res.output
    assert "sft/merged" in res.output and "conversation cleared" in res.output

    # the SFT adapter merges with the generic merge command (its base is the merged CPT model)
    res = runner.invoke(cli, [*_base_args(cfg), "merge", str(run / "sft" / "adapter"), "-o", str(tmp_path / "m")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "m" / "config.json").exists()
