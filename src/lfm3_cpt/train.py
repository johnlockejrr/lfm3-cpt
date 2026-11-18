"""The CPT pipeline: prepare data -> baseline eval -> LoRA training -> final eval -> report (-> merge)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import datasets as hf_datasets
import torch
from rich.console import Console
from transformers import set_seed
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .callbacks import JsonlMetricsLogger, TrainingDashboard, make_progress
from .config import Config
from .data import DataBundle, Loaders, data_signature, prepare_data, with_eos
from .device import DeviceInfo, configure_backend, detect, peak_memory_gb, resolve_precision, torch_dtype
from .evaluation import compare_grammar, corpus_perplexity, count_packed_rows, grammar_test
from .hub import ensure_model_files, quiet_libraries
from .modeling import build_lora_config, load_model, load_tokenizer, merge_adapter
from .reporting import config_to_toml, print_flip_examples, summary_table


def checkpointing_args(enabled: bool) -> dict[str, Any]:
    """Activation checkpointing settings under the name this transformers version uses.

    transformers 5.23 renamed `gradient_checkpointing*` to `activation_checkpointing*` and warns on the old name;
    older versions only know the old one.
    """
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    flag, kwargs = (
        ("activation_checkpointing", "activation_checkpointing_kwargs")
        if "activation_checkpointing" in names
        else ("gradient_checkpointing", "gradient_checkpointing_kwargs")
    )
    return {flag: enabled, kwargs: {"use_reentrant": False} if enabled else None}


def build_sft_config(cfg: Config, precision: str, info: DeviceInfo) -> SFTConfig:
    return SFTConfig(
        output_dir=str(Path(cfg.output_dir) / "checkpoints"),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup,  # a float < 1 is a fraction of the total steps
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        # Plain-text CPT: every token is trained on; "wrapped" packing concatenates documents (EOS-separated) and
        # cuts the stream into full rows of max_length, as in pre-training.
        max_length=cfg.max_length,
        packing=True,
        packing_strategy="wrapped",
        dataset_text_field="text",
        **checkpointing_args(cfg.gradient_checkpointing),
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=precision == "bf16",
        fp16=precision == "fp16",  # fp16 mixed precision with loss scaling on GPUs without bf16
        use_cpu=info.backend == "cpu",
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory and info.backend in {"cuda", "rocm"},
        include_num_input_tokens_seen="non_padding",
        seed=cfg.seed,
        data_seed=cfg.seed,
        disable_tqdm=True,  # replaced by the Rich dashboard
        report_to="none",
    )


def evaluate_all(
    model: torch.nn.Module,
    tokenizer: Any,
    bundle: DataBundle,
    cfg: Config,
    console: Console,
    label: str,
    grammar: bool = True,
) -> dict[str, Any]:
    """Perplexity on held-out and OOD text, plus the minimal-pair grammar test."""
    eos = tokenizer.eos_token
    wiki = list(with_eos(bundle.eval, eos)["text"])
    ood = list(with_eos(bundle.ood, eos)["text"]) if len(bundle.ood) else []
    out: dict[str, Any] = {}
    with make_progress(console, transient=True) as progress:
        for split, texts in (("wiki", wiki), ("ood", ood)):
            if not texts:
                out[split] = None
                continue
            n_rows = count_packed_rows(tokenizer, texts, cfg.max_length)
            task = progress.add_task(f"{label}: perplexity ({split})", total=n_rows)
            out[split] = corpus_perplexity(
                model, tokenizer, texts, cfg.max_length, cfg.ppl_batch_size, lambda n, t=task: progress.advance(t, n)
            ).to_dict()
        if grammar and bundle.grammar is not None:
            task = progress.add_task(f"{label}: minimal pairs", total=2 * len(bundle.grammar))
            result = grammar_test(
                model, tokenizer, bundle.grammar, cfg.grammar_batch_size, lambda n: progress.advance(task, n)
            )
            out["grammar"] = result.to_dict()
    return out


def _baseline_key(cfg: Config) -> dict[str, Any]:
    return {
        "model_id": cfg.model_id,
        "max_length": cfg.max_length,
        "data": data_signature(cfg),
        "grammar": cfg.run_grammar,
    }


def run_training(
    cfg: Config,
    console: Console | None = None,
    resume: bool = False,
    skip_baseline: bool = False,
    merge: bool = False,
    force_data: bool = False,
    loaders: Loaders | None = None,
) -> dict[str, Any]:
    console = console or Console()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.toml").write_text(config_to_toml(cfg), encoding="utf-8")

    info = detect()
    configure_backend(info)
    precision = resolve_precision(cfg.precision, info)
    set_seed(cfg.seed)
    console.rule(f"[bold]LFM3 CPT[/] • {info.name} ({info.backend}) • {precision}")

    # --- data + model ------------------------------------------------------------------------------------
    quiet_libraries()
    ensure_model_files(cfg.model_id, console)  # before any spinner: one clean download bar
    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    bundle = prepare_data(cfg, tokenizer, force=force_data, loaders=loaders, log=console.print)
    stats = bundle.metadata.get("stats", {})
    console.print(
        f"📚 {stats.get('source')}: train {stats.get('train_articles')} docs "
        f"({stats.get('train_tokens', 0) / 1e6:.1f}M tokens) • eval {stats.get('eval_articles')} • "
        f"OOD {stats.get('ood_pages')} • grammar pairs {stats.get('grammar_pairs')}"
    )
    if "tokenizer_verdict" in stats:
        console.print(
            f"🔤 tokenizer: {stats['chars_per_token']} chars/token, {stats['tokens_per_word']} tokens/word "
            f"({stats['tokens_per_word_vs_english']}× English) • {stats['tokenizer_verdict']}"
        )
    hf_datasets.disable_progress_bars()  # keep the console for the dashboard

    with console.status("Loading model ..."):
        device_map = "auto" if info.backend in {"cuda", "rocm"} else None
        model = load_model(
            cfg.model_id, torch_dtype(precision), device_map, cfg.trust_remote_code, cfg.attn_implementation
        )
    console.print(f"🧠 {cfg.model_id}: {model.num_parameters():,} parameters")

    # --- baseline (cached so a resumed run does not recompute it) -----------------------------------------
    baseline_path = out_dir / "baseline.json"
    baseline: dict[str, Any] | None = None
    if baseline_path.exists():
        cached = json.loads(baseline_path.read_text())
        if cached.get("key") == _baseline_key(cfg):
            baseline = cached["results"]
            console.print("[dim]Re-using cached baseline evaluation[/dim]")
    if baseline is None and not skip_baseline:
        baseline = evaluate_all(model, tokenizer, bundle, cfg, console, "baseline", cfg.run_grammar)
        baseline_path.write_text(json.dumps({"key": _baseline_key(cfg), "results": baseline}, indent=2))
    if baseline:
        console.print(summary_table(baseline, None, title="Before training"))

    # --- trainer --------------------------------------------------------------------------------------------
    peft_config = build_lora_config(cfg, model, warn=console.print)
    eos = tokenizer.eos_token
    eval_sets = {"wiki": with_eos(bundle.eval, eos)}
    if len(bundle.ood):
        eval_sets["ood"] = with_eos(bundle.ood, eos)  # tracked during training too, not only before/after
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg, precision, info),
        train_dataset=with_eos(bundle.train, eos),
        eval_dataset=eval_sets,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    for cb in (ProgressCallback, PrinterCallback):
        trainer.remove_callback(cb)
    trainer.add_callback(TrainingDashboard(console, tokens_per_step=cfg.tokens_per_step))
    trainer.add_callback(JsonlMetricsLogger(out_dir / "metrics.jsonl"))

    trainable, total = trainer.model.get_nb_trainable_parameters()
    n_rows = len(trainer.train_dataset)
    console.print(
        f"🎛️  LoRA r={cfg.lora_r} on {', '.join(peft_config.target_modules)}"
        f"{' + embeddings' if cfg.train_embeddings else ''}: {trainable:,} trainable / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    console.print(
        f"📦 {n_rows} packed rows × {cfg.max_length} tokens • {cfg.effective_batch_rows} rows/step "
        f"({cfg.tokens_per_step:,} tokens/step) • checkpointing={'on' if cfg.gradient_checkpointing else 'off'}"
    )

    resume_from = None
    if resume:
        resume_from = get_last_checkpoint(str(Path(cfg.output_dir) / "checkpoints"))
        if resume_from is None:
            console.print("[yellow]No checkpoint found, starting from scratch[/yellow]")
        else:
            console.print(f"↩️  Resuming from {resume_from}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train(resume_from_checkpoint=resume_from)
    runtime = time.time() - t0
    peak = peak_memory_gb()
    console.print(f"🎉 Training done in {runtime / 60:.1f} min" + (f" • peak GPU memory {peak:.1f} GB" if peak else ""))

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    console.print(f"💾 LoRA adapter saved to {adapter_dir}")

    # --- final evaluation + report ------------------------------------------------------------------------
    final = evaluate_all(trainer.model, tokenizer, bundle, cfg, console, "fine-tuned", cfg.run_grammar)
    results: dict[str, Any] = {
        "config": cfg.to_dict(),
        "device": info.to_dict(),
        "precision": precision,
        "data": bundle.metadata,
        "baseline": baseline,
        "final": final,
        "train": {
            "runtime_s": round(runtime, 1),
            "global_step": train_output.global_step,
            "train_loss": train_output.training_loss,
            "tokens_seen": trainer.state.num_input_tokens_seen,
            "peak_mem_gb": peak,
            "trainable_params": trainable,
            "total_params": total,
            "packed_rows": n_rows,
        },
    }
    if baseline and baseline.get("grammar") and final.get("grammar"):
        comparison = compare_grammar(bundle.grammar, baseline["grammar"]["passed"], final["grammar"]["passed"])
        results["grammar_comparison"] = comparison
        print_flip_examples(console, comparison)
    console.print(summary_table(baseline, final))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"📝 Results: {out_dir / 'results.json'} • metrics: {out_dir / 'metrics.jsonl'}")

    if merge:
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with console.status("Merging LoRA into a fresh bf16 copy of the base model ..."):
            merged = merge_adapter(
                adapter_dir, out_dir / "merged", cfg.model_id, torch.bfloat16, "cpu", cfg.trust_remote_code
            )
        console.print(f"💾 Merged model saved to {merged}")
        results["merged_dir"] = str(merged)
    return results
