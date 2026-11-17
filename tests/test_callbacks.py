import json
import math

import pytest
from rich.console import Console
from transformers import TrainerControl, TrainerState, TrainingArguments

from lfm3_cpt.callbacks import JsonlMetricsLogger, TrainingDashboard, eval_split_name, safe_ppl


def test_eval_split_name():
    assert eval_split_name("eval_loss") == "eval"
    assert eval_split_name("eval_wiki_loss") == "wiki"
    assert eval_split_name("eval_ood_loss") == "ood"
    assert eval_split_name("loss") is None
    assert eval_split_name("eval_runtime") is None


def test_safe_ppl():
    assert safe_ppl(0.0) == 1.0
    assert safe_ppl(float("nan")) is None
    assert safe_ppl(None) is None
    assert math.isfinite(safe_ppl(1e9))


def test_dashboard_tracks_metrics_and_best():
    dash = TrainingDashboard(Console(file=open("/dev/null", "w")), tokens_per_step=1000)
    dash.update_from_logs({"loss": 2.0, "learning_rate": 1e-4, "grad_norm": 0.5, "epoch": 0.1}, step=10)
    dash.update_from_logs({"loss": 1.0}, step=20)
    assert dash.metrics["loss"] == 1.0
    assert dash.loss_ema == pytest.approx(1.9)
    assert dash.metrics["tokens_seen"] == 20_000
    dash.update_from_logs({"eval_wiki_loss": 1.5, "eval_ood_loss": 2.5}, step=20)
    dash.update_from_logs({"eval_wiki_loss": 1.7}, step=40)
    assert dash.eval_metrics["wiki"] == {"loss": 1.7, "best": 1.5, "step": 40}
    line = dash.log_line(40, 100)
    assert "step 40/100" in line and "wiki ppl" in line and "ood ppl" in line


def test_dashboard_renders(capsys):
    console = Console(record=True, width=120, force_terminal=False)
    dash = TrainingDashboard(console)
    dash.update_from_logs({"loss": 2.0, "learning_rate": 1e-4}, step=1, tokens_seen=4096)
    dash.update_from_logs({"eval_wiki_loss": 1.0}, step=1)
    console.print(dash)
    text = console.export_text()
    assert "loss" in text and "wiki" in text and "2.0000" in text


def test_dashboard_lifecycle(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    state = TrainerState(max_steps=4)
    state.is_world_process_zero = True
    control = TrainerControl()
    dash = TrainingDashboard(Console(file=open(tmp_path / "out.txt", "w")))
    dash.on_train_begin(args, state, control)
    for step in range(1, 5):
        state.global_step = step
        dash.on_step_end(args, state, control)
    dash.on_prediction_step(args, state, control, eval_dataloader=[1, 2])
    dash.on_prediction_step(args, state, control, eval_dataloader=[1, 2])
    dash.on_evaluate(args, state, control, metrics={})
    assert dash.eval_task is None
    dash.on_log(args, state, control, logs={"loss": 1.0})
    dash.on_train_end(args, state, control)
    assert dash.live is None
    assert dash.progress.tasks[0].completed == 4
    assert "step 4/4" in (tmp_path / "out.txt").read_text()  # non-terminal console gets plain log lines


def test_jsonl_logger(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True)
    state = TrainerState()
    state.global_step = 7
    logger = JsonlMetricsLogger(tmp_path / "m" / "metrics.jsonl")
    logger.on_log(args, state, TrainerControl(), logs={"loss": 1.0, "eval_wiki_loss": 0.0, "learning_rate": 1e-4})
    logger.on_log(args, state, TrainerControl(), logs={"train_runtime": 3.0})
    lines = [json.loads(x) for x in (tmp_path / "m" / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["step"] == 7
    assert lines[0]["ppl"] == math.exp(1.0)
    assert lines[0]["eval_wiki_ppl"] == 1.0
    assert lines[1]["train_runtime"] == 3.0
