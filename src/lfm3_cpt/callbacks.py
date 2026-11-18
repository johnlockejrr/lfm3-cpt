"""Trainer callbacks: a Rich live dashboard (replacing tqdm) and a JSONL metrics logger."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .device import current_memory_gb, peak_memory_gb


def make_progress(console: Console | None = None, transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        expand=True,
    )


def safe_ppl(loss: float | None) -> float | None:
    if loss is None or not math.isfinite(loss):
        return None
    return math.exp(min(loss, 50.0))


def eval_split_name(key: str) -> str | None:
    """`eval_loss` -> "eval", `eval_wiki_loss` -> "wiki", anything else -> None."""
    if not key.startswith("eval_") or not key.endswith("_loss"):
        return None
    middle = key[len("eval_") : -len("_loss")]
    return middle or "eval"


def _fmt(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


class TrainingDashboard(TrainerCallback):
    """Live Rich view: progress bar + training/eval metrics, refreshed while training runs."""

    def __init__(self, console: Console | None = None, tokens_per_step: int | None = None, ema: float = 0.9):
        self.console = console or Console()
        self.tokens_per_step = tokens_per_step
        self.ema_beta = ema
        self.progress = make_progress(self.console)
        self.live: Live | None = None
        self.train_task: TaskID | None = None
        self.eval_task: TaskID | None = None
        self.metrics: dict[str, Any] = {}
        self.eval_metrics: dict[str, dict[str, float]] = {}
        self.loss_ema: float | None = None
        self._last_time: float | None = None
        self._last_tokens: float | None = None
        self._last_step = 0

    # --- rendering -----------------------------------------------------------------------------------------
    def __rich__(self) -> Group:
        return Group(self.progress, self._metrics_table())

    def _metrics_table(self) -> Table:
        m = self.metrics
        table = Table.grid(padding=(0, 3), expand=False)
        for _ in range(4):
            table.add_column(justify="left")
        table.add_row(
            f"[cyan]loss[/] {_fmt(m.get('loss'), '.4f')}",
            f"[cyan]ema[/] {_fmt(self.loss_ema, '.4f')}",
            f"[cyan]ppl[/] {_fmt(safe_ppl(self.loss_ema), '.2f')}",
            f"[cyan]epoch[/] {_fmt(m.get('epoch'), '.3f')}",
        )
        table.add_row(
            f"[cyan]lr[/] {_fmt(m.get('learning_rate'), '.2e')}",
            f"[cyan]grad‖·‖[/] {_fmt(m.get('grad_norm'), '.3f')}",
            f"[cyan]tok/s[/] {_fmt(m.get('tokens_per_second'), ',.0f')}",
            f"[cyan]tokens[/] {_fmt(m.get('tokens_seen'), ',.0f')}",
        )
        mem = current_memory_gb()
        peak = peak_memory_gb()
        if mem is not None:
            table.add_row(
                f"[cyan]mem[/] {mem[0]:.1f} GB",
                f"[cyan]reserved[/] {mem[1]:.1f} GB",
                f"[cyan]peak[/] {_fmt(peak, '.1f')} GB",
                "",
            )
        for split, values in self.eval_metrics.items():
            table.add_row(
                f"[magenta]{split}[/] loss {values['loss']:.4f}",
                f"ppl {safe_ppl(values['loss']):.2f}",
                f"best ppl {safe_ppl(values['best']):.2f}",
                f"@ step {int(values['step'])}",
            )
        return table

    # --- callback hooks -------------------------------------------------------------------------------------
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return
        self.train_task = self.progress.add_task("training", total=state.max_steps, completed=state.global_step)
        self._last_time, self._last_step = time.time(), state.global_step
        self._last_tokens = float(state.num_input_tokens_seen or 0)
        self.live = Live(self, console=self.console, refresh_per_second=4, transient=False)
        self.live.start()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.train_task is not None:
            self.progress.update(self.train_task, completed=state.global_step)

    def on_prediction_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, eval_dataloader=None, **kwargs
    ):
        if not state.is_world_process_zero:
            return
        if self.eval_task is None:
            total = (
                len(eval_dataloader) if eval_dataloader is not None and hasattr(eval_dataloader, "__len__") else None
            )
            self.eval_task = self.progress.add_task("evaluating", total=total)
        self.progress.advance(self.eval_task)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs
    ):
        if self.eval_task is not None:
            self.progress.remove_task(self.eval_task)
            self.eval_task = None

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        self.update_from_logs(logs, state.global_step, getattr(state, "num_input_tokens_seen", None))
        if not self.console.is_terminal:  # e.g. redirected to a file: one plain line per log
            self.console.print(self.log_line(state.global_step, state.max_steps))

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.live is not None:
            self.progress.update(self.train_task, completed=state.global_step)
            self.live.stop()
            self.live = None
            if not self.console.is_terminal:
                self.console.line()  # a redirected Live display ends without a newline

    # --- logic (separate from the hooks so it is easy to test) -------------------------------------------
    def update_from_logs(self, logs: dict[str, float], step: int, tokens_seen: float | None = None) -> None:
        now = time.time()
        if "loss" in logs:
            loss = float(logs["loss"])
            self.metrics["loss"] = loss
            if math.isfinite(loss):
                self.loss_ema = (
                    loss if self.loss_ema is None else self.ema_beta * self.loss_ema + (1 - self.ema_beta) * loss
                )
            for key in ("learning_rate", "grad_norm", "epoch"):
                if key in logs:
                    self.metrics[key] = float(logs[key])
            tokens = logs.get("num_input_tokens_seen", tokens_seen)
            if not tokens and self.tokens_per_step:
                tokens = step * self.tokens_per_step
            if tokens:
                if self._last_time is not None and self._last_tokens is not None and now > self._last_time:
                    self.metrics["tokens_per_second"] = (float(tokens) - self._last_tokens) / (now - self._last_time)
                self.metrics["tokens_seen"] = float(tokens)
                self._last_tokens = float(tokens)
            self._last_time, self._last_step = now, step
        for key, value in logs.items():
            split = eval_split_name(key)
            if split is None:
                continue
            best = self.eval_metrics.get(split, {}).get("best", float("inf"))
            self.eval_metrics[split] = {"loss": float(value), "best": min(best, float(value)), "step": step}

    def log_line(self, step: int, max_steps: int) -> str:
        m = self.metrics
        parts = [
            f"step {step}/{max_steps}",
            f"loss {_fmt(m.get('loss'), '.4f')}",
            f"lr {_fmt(m.get('learning_rate'), '.2e')}",
        ]
        if "tokens_per_second" in m:
            parts.append(f"{m['tokens_per_second']:,.0f} tok/s")
        for split, v in self.eval_metrics.items():
            parts.append(f"{split} ppl {safe_ppl(v['loss']):.2f}")
        return " | ".join(parts)


class JsonlMetricsLogger(TrainerCallback):
    """Append every Trainer log to `metrics.jsonl`, adding perplexities for every `*loss` key."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        record: dict[str, Any] = {"step": state.global_step, "elapsed_s": round(time.time() - self._t0, 2)}
        for key, value in logs.items():
            record[key] = value
            if key.endswith("loss") and isinstance(value, (int, float)):
                ppl = safe_ppl(float(value))
                if ppl is not None:
                    record[key[: -len("loss")] + "ppl"] = ppl
        peak = peak_memory_gb()
        if peak is not None:
            record["peak_mem_gb"] = round(peak, 3)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
