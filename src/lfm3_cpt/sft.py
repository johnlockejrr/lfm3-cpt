"""Stage 2 (optional): a short SFT pass (Turkish conversations by default) on top of the CPT model.

CPT teaches the language; SFT teaches the model to *answer* in it. The stage:

1. starts from the CPT result (`<output_dir>/merged`, merged from `<output_dir>/adapter` automatically if needed),
   or from the original model (`--from base`, the control experiment: SFT without CPT), or any model path;
2. evaluates the starting model: answer perplexity, generated answers (Turkish rate, finished rate, chrF), and,
   to catch forgetting, the stage-1 metrics (Wikipedia/web perplexity, TurBLiMP);
3. trains a fresh LoRA adapter on the answers only (TRL prompt-completion, non-reasoning mode);
4. evaluates again, writes `generations.jsonl` (prompt, reference, before, after) and `results.json`.

Outputs go to `<output_dir>/sft/`. The adapter's base is the merged CPT model, so `lfm3-cpt merge` works as is.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import torch
from rich.console import Console
from rich.table import Table
from transformers import set_seed
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .callbacks import JsonlMetricsLogger, TrainingDashboard, make_progress
from .chat_eval import answer_perplexity, generate_answers, summarize_generations
from .config import Config
from .data import is_prepared, load_data
from .device import DeviceInfo, configure_backend, detect, peak_memory_gb, resolve_precision, torch_dtype
from .modeling import build_lora_config, is_adapter_dir, load_model, load_tokenizer, merge_adapter
from .reporting import config_to_toml, sft_summary_table
from .sft_data import SftBundle, SftLoader, prepare_sft_data, sft_signature, template_kwargs, with_template_kwargs


def sft_output_dir(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "sft"


def stage_config(cfg: Config) -> Config:
    """A copy of the config whose generic training/LoRA fields hold the SFT values (so shared helpers work)."""
    return dataclasses.replace(
        cfg,
        output_dir=str(sft_output_dir(cfg)),
        lora_r=cfg.sft_lora_r,
        lora_alpha=cfg.sft_lora_alpha,
        lora_dropout=cfg.sft_lora_dropout,
        train_embeddings=False,  # embeddings were the CPT stage's job
        num_train_epochs=cfg.sft_num_train_epochs,
        max_steps=cfg.sft_max_steps,
        learning_rate=cfg.sft_learning_rate,
        per_device_train_batch_size=cfg.sft_per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft_gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.sft_per_device_eval_batch_size,
        max_length=cfg.sft_max_length,
        gradient_checkpointing=cfg.sft_gradient_checkpointing,
        logging_steps=cfg.sft_logging_steps,
        eval_steps=cfg.sft_eval_steps,
        save_steps=cfg.sft_save_steps,
    )


def resolve_start(cfg: Config, start: str, log=print) -> str:
    """Where SFT starts: "cpt" (default), "base", or a model / adapter path.

    An adapter is merged (bf16, on CPU) into a standalone model first, so SFT trains a fresh adapter on top of a
    plain model instead of stacking adapters.
    """
    if start == "base":
        return cfg.model_id
    if start == "cpt":
        run = Path(cfg.output_dir)
        if (run / "merged" / "config.json").exists():
            return str((run / "merged").resolve())
        if is_adapter_dir(run / "adapter"):
            start = str(run / "adapter")
        else:
            raise FileNotFoundError(
                f"no CPT result in {run} (expected merged/ or adapter/). Run `lfm3-cpt train` first, "
                "or use --from base / --from PATH."
            )
    path = Path(start)
    if path.exists() and is_adapter_dir(path):
        merged = path.resolve().parent / "merged"
        if not (merged / "config.json").exists():
            log(f"Merging {path} into {merged} (bf16, CPU) so SFT starts from a standalone model ...")
            merge_adapter(path, merged, None, torch.bfloat16, "cpu", cfg.trust_remote_code)
        return str(merged)
    return str(path.resolve()) if path.exists() else start


def build_sft_stage_config(stage: Config, precision: str, info: DeviceInfo) -> SFTConfig:
    return SFTConfig(
        output_dir=str(Path(stage.output_dir) / "checkpoints"),
        num_train_epochs=stage.num_train_epochs,
        max_steps=stage.max_steps,
        per_device_train_batch_size=stage.per_device_train_batch_size,
        gradient_accumulation_steps=stage.gradient_accumulation_steps,
        per_device_eval_batch_size=stage.per_device_eval_batch_size,
        learning_rate=stage.learning_rate,
        lr_scheduler_type=stage.lr_scheduler_type,
        warmup_steps=stage.warmup,
        weight_decay=stage.weight_decay,
        max_grad_norm=stage.max_grad_norm,
        max_length=stage.max_length,
        packing=False,
        completion_only_loss=True,  # loss on the assistant answer only, not on the user's message
        # Batches of similar length: much less padding with short, varied chat examples.
        train_sampling_strategy="group_by_length",
        gradient_checkpointing=stage.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if stage.gradient_checkpointing else None,
        logging_steps=stage.logging_steps,
        eval_strategy="steps",
        eval_steps=stage.eval_steps,
        save_strategy="steps",
        save_steps=stage.save_steps,
        save_total_limit=stage.save_total_limit,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        use_cpu=info.backend == "cpu",
        dataloader_num_workers=stage.dataloader_num_workers,
        dataloader_pin_memory=stage.dataloader_pin_memory and info.backend in {"cuda", "rocm"},
        include_num_input_tokens_seen="non_padding",
        seed=stage.seed,
        data_seed=stage.seed,
        disable_tqdm=True,
        report_to="none",
    )


def evaluate_chat(
    model: torch.nn.Module,
    tokenizer: Any,
    bundle: SftBundle,
    cfg: Config,
    console: Console,
    label: str,
    lm_eval: bool = True,
) -> dict[str, Any]:
    """Chat metrics on the held-out conversations, plus the stage-1 LM metrics when their data is cached."""
    kwargs = template_kwargs(cfg)
    rows = list(bundle.eval)
    gen_rows = rows[: cfg.sft_gen_samples] if cfg.sft_gen_samples > 0 else []
    out: dict[str, Any] = {}
    with make_progress(console, transient=True) as progress:
        task = progress.add_task(f"{label}: answer perplexity", total=len(rows))
        out["answer"] = answer_perplexity(
            model, tokenizer, rows, kwargs, cfg.sft_per_device_eval_batch_size, lambda n: progress.advance(task, n)
        ).to_dict()
        if gen_rows:
            task = progress.add_task(f"{label}: generating answers", total=len(gen_rows))
            answers = generate_answers(
                model,
                tokenizer,
                [r["prompt"] for r in gen_rows],
                kwargs,
                cfg.sft_max_new_tokens,
                cfg.sft_gen_batch_size,
                cfg.sft_repetition_penalty,
                lambda n: progress.advance(task, n),
            )
            out["generation"] = summarize_generations(
                answers, [r["completion"][0]["content"] for r in gen_rows], cfg.target_language
            )
            out["answers"] = answers
    if lm_eval and is_prepared(cfg):
        from .train import evaluate_all  # stage-1 metrics: did SFT undo what CPT taught?

        out["lm"] = evaluate_all(model, tokenizer, load_data(cfg), cfg, console, f"{label} (LM)", cfg.run_grammar)
    return out


def _baseline_key(cfg: Config, start: str, lm_eval: bool) -> dict[str, Any]:
    return {
        "start": start,
        "data": sft_signature(cfg),
        "gen": [cfg.sft_gen_samples, cfg.sft_max_new_tokens, cfg.sft_repetition_penalty, cfg.sft_enable_thinking],
        "lm_eval": lm_eval and is_prepared(cfg),
    }


def _write_generations(path: Path, bundle: SftBundle, cfg: Config, before: dict | None, after: dict) -> None:
    rows = list(bundle.eval)[: cfg.sft_gen_samples]
    b_answers = (before or {}).get("answers") or [None] * len(rows)
    b_langs = (before or {}).get("generation", {}).get("languages") or [None] * len(rows)
    a_langs = after["generation"]["languages"]
    with path.open("w", encoding="utf-8") as fh:
        for row, b, bl, a, al in zip(rows, b_answers, b_langs, after["answers"], a_langs, strict=False):
            fh.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "reference": row["completion"][0]["content"],
                        "before": b["text"] if b else None,
                        "before_language": bl,
                        "after": a["text"],
                        "after_language": al,
                        "after_finished": a["finished"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _strip_answers(block: dict | None) -> dict | None:
    if not block:
        return block
    return {k: v for k, v in block.items() if k != "answers"}


def show_samples(console: Console, bundle: SftBundle, before: dict | None, after: dict, n: int = 3) -> None:
    rows = list(bundle.eval)[:n]
    for i, row in enumerate(rows):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(overflow="fold")
        table.add_row("prompt", row["prompt"][-1]["content"][:400])
        if before and before.get("answers"):
            table.add_row("before", before["answers"][i]["text"][:400])
        table.add_row("after", after["answers"][i]["text"][:400])
        table.add_row("reference", row["completion"][0]["content"][:400])
        console.rule(f"sample {i + 1}")
        console.print(table, highlight=False)


def run_sft(
    cfg: Config,
    console: Console | None = None,
    start: str = "cpt",
    resume: bool = False,
    skip_baseline: bool = False,
    merge: bool = False,
    force_data: bool = False,
    lm_eval: bool = True,
    loader: SftLoader | None = None,
) -> dict[str, Any]:
    console = console or Console()
    stage = stage_config(cfg)
    out_dir = Path(stage.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.toml").write_text(config_to_toml(cfg), encoding="utf-8")

    info = detect()
    configure_backend(info)
    precision = resolve_precision(cfg.precision, info)
    set_seed(cfg.seed)
    start_path = resolve_start(cfg, start, console.print)
    console.rule(f"[bold]LFM3 SFT (stage 2)[/] • {info.name} ({info.backend}) • {precision}")
    console.print(f"🧭 starting from {start_path}")

    tokenizer = load_tokenizer(start_path, cfg.trust_remote_code)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{start_path} has no chat template; SFT needs one")
    bundle = prepare_sft_data(cfg, tokenizer, force=force_data, loader=loader, log=console.print)
    s = bundle.metadata["stats"]
    console.print(
        f"💬 {s['train_examples']} train / {s['eval_examples']} eval conversations from {cfg.sft_dataset} "
        f"(dropped: {s.get('too_long', 0)} too long, {s.get('duplicates', 0)} duplicates, {s.get('unusable', 0)} unusable)"
    )
    hf_datasets.disable_progress_bars()

    with console.status("Loading model ..."):
        device_map = "auto" if info.backend in {"cuda", "rocm"} else None
        model = load_model(
            start_path, torch_dtype(precision), device_map, cfg.trust_remote_code, cfg.attn_implementation
        )

    baseline_path = out_dir / "baseline.json"
    key = _baseline_key(cfg, start_path, lm_eval)
    before: dict[str, Any] | None = None
    if baseline_path.exists():
        cached = json.loads(baseline_path.read_text())
        if cached.get("key") == key:
            before = cached["results"]
            console.print("[dim]Re-using cached baseline evaluation[/dim]")
    if before is None and not skip_baseline:
        before = evaluate_chat(model, tokenizer, bundle, cfg, console, "before", lm_eval)
        baseline_path.write_text(json.dumps({"key": key, "results": before}, indent=2, ensure_ascii=False))
    if before:
        console.print(sft_summary_table(before, None, title="Before SFT", language=cfg.target_language))

    peft_config = build_lora_config(stage, model, warn=console.print)
    trainer = SFTTrainer(
        model=model,
        args=build_sft_stage_config(stage, precision, info),
        train_dataset=with_template_kwargs(bundle.train, cfg),
        eval_dataset=with_template_kwargs(bundle.eval, cfg),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    for cb in (ProgressCallback, PrinterCallback):
        trainer.remove_callback(cb)
    trainer.add_callback(TrainingDashboard(console))
    trainer.add_callback(JsonlMetricsLogger(out_dir / "metrics.jsonl"))
    trainable, total = trainer.model.get_nb_trainable_parameters()
    console.print(
        f"🎛️  LoRA r={stage.lora_r} on {', '.join(peft_config.target_modules)}: {trainable:,} trainable "
        f"({100 * trainable / total:.2f}%) • {stage.effective_batch_rows} conversations/step"
    )

    resume_from = None
    if resume:
        resume_from = get_last_checkpoint(str(out_dir / "checkpoints"))
        console.print(f"↩️  Resuming from {resume_from}" if resume_from else "[yellow]No checkpoint found[/yellow]")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train(resume_from_checkpoint=resume_from)
    runtime = time.time() - t0
    peak = peak_memory_gb()
    console.print(f"🎉 SFT done in {runtime / 60:.1f} min" + (f" • peak GPU memory {peak:.1f} GB" if peak else ""))

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    console.print(f"💾 SFT adapter saved to {adapter_dir} (base: {start_path})")

    after = evaluate_chat(trainer.model, tokenizer, bundle, cfg, console, "after", lm_eval)
    _write_generations(out_dir / "generations.jsonl", bundle, cfg, before, after)
    show_samples(console, bundle, before, after)
    results = {
        "stage": "sft",
        "start_model": start_path,
        "config": cfg.to_dict(),
        "device": info.to_dict(),
        "precision": precision,
        "data": bundle.metadata,
        "baseline": _strip_answers(before),
        "final": _strip_answers(after),
        "train": {
            "runtime_s": round(runtime, 1),
            "global_step": train_output.global_step,
            "train_loss": train_output.training_loss,
            "tokens_seen": trainer.state.num_input_tokens_seen,
            "peak_mem_gb": peak,
            "trainable_params": trainable,
            "total_params": total,
        },
    }
    console.print(sft_summary_table(results["baseline"], results["final"], language=cfg.target_language))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"📝 Results: {out_dir / 'results.json'} • answers: {out_dir / 'generations.jsonl'}")

    if merge:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with console.status("Merging the SFT adapter (bf16, CPU) ..."):
            merged = merge_adapter(
                adapter_dir, out_dir / "merged", start_path, torch.bfloat16, "cpu", cfg.trust_remote_code
            )
        console.print(f"💾 Merged chat model saved to {merged}")
        results["merged_dir"] = str(merged)
    return results
