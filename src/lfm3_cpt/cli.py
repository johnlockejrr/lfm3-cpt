"""`lfm3-cpt` command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import PROFILES, Config, build_config, parse_set_options

console = Console()


class Ctx:
    def __init__(self, profile: str | None, config_path: str | None, sets: dict[str, Any]):
        self.profile = profile
        self.config_path = config_path
        self.sets = sets

    def config(self, **overrides: Any) -> Config:
        merged = {**self.sets, **{k: v for k, v in overrides.items() if v is not None}}
        try:
            return build_config(self.profile, self.config_path, merged)
        except (KeyError, ValueError) as exc:
            raise click.UsageError(str(exc)) from exc


pass_ctx = click.make_pass_decorator(Ctx)


@click.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
@click.version_option(__version__, prog_name="lfm3-cpt")
@click.option("-p", "--profile", type=click.Choice(sorted(PROFILES)), help="Hardware preset (see `lfm3-cpt info`).")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True, dir_okay=False), help="TOML config file.")
@click.option("-s", "--set", "sets", multiple=True, metavar="KEY=VALUE", help="Override any config key (repeatable).")
@click.pass_context
def cli(ctx: click.Context, profile: str | None, config_path: str | None, sets: tuple[str, ...]) -> None:
    """Continued pre-training (CPT) of LiquidAI LFM3 with TRL + LoRA.

    Precedence: defaults < --profile < --config < --set < command options.
    """
    try:
        parsed = parse_set_options(sets)
    except (KeyError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    ctx.obj = Ctx(profile, config_path, parsed)


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
def info() -> None:
    """Show the GPU / backend (CUDA or ROCm), library versions and the suggested profile."""
    import importlib.metadata as md

    from .device import detect, relevant_env, resolve_precision, suggest_profile

    d = detect()
    table = Table(title="Environment", show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("backend", d.backend.upper())
    table.add_row("device", f"{d.name} ×{d.count}" if d.count else d.name)
    if d.arch:
        table.add_row("arch", d.arch)
    if d.total_memory_gb:
        table.add_row("memory", f"{d.total_memory_gb:.1f} GB")
    table.add_row("bf16 / tf32", f"{d.bf16} / {d.tf32}")
    table.add_row("precision (auto)", resolve_precision("auto", d))
    table.add_row(
        "torch", d.torch_version + (f" (HIP {d.hip_version})" if d.hip_version else f" (CUDA {d.cuda_version})")
    )
    for pkg in ("transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            table.add_row(pkg, md.version(pkg))
        except md.PackageNotFoundError:
            table.add_row(pkg, "[red]missing[/]")
    table.add_row("python", d.python)
    for key, value in relevant_env(d).items():
        table.add_row(key, value)
    table.add_row("suggested profile", f"[bold green]{suggest_profile(d)}[/]")
    console.print(table)


@cli.command("show-config")
@click.option("--toml", "as_toml", is_flag=True, help="Print as TOML (can be saved and passed to --config).")
@pass_ctx
def show_config(obj: Ctx, as_toml: bool) -> None:
    """Print the resolved configuration."""
    from .reporting import config_table, config_to_toml

    cfg = obj.config()
    if as_toml:
        click.echo(config_to_toml(cfg), nl=False)
    else:
        console.print(config_table(cfg))


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.option("--data-dir", type=click.Path(file_okay=False), help="Where to cache the corpora.")
@click.option("--train-articles", type=int)
@click.option("--eval-articles", type=int)
@click.option("--ood-pages", type=int)
@click.option("--no-grammar", is_flag=True, help="Skip downloading TurBLiMP.")
@click.option("--force", is_flag=True, help="Re-download even if a matching cache exists.")
@pass_ctx
def prepare(obj: Ctx, data_dir, train_articles, eval_articles, ood_pages, no_grammar, force) -> None:
    """Stream the corpora from the Hub and cache them on disk (train/eval/ood/grammar)."""
    from .data import prepare_data
    from .modeling import load_tokenizer

    cfg = obj.config(
        data_dir=data_dir,
        train_articles=train_articles,
        eval_articles=eval_articles,
        ood_pages=ood_pages,
        run_grammar=False if no_grammar else None,
    )
    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    bundle = prepare_data(cfg, tokenizer, force=force, log=console.print)
    table = Table(title=f"Data in {cfg.data_dir}", show_header=False)
    for key, value in bundle.metadata.get("stats", {}).items():
        table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)
    console.print(f"\n📝 Sample:\n{bundle.train[0]['text'][:500]}")


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.option("-o", "--output-dir", type=click.Path(file_okay=False))
@click.option("--model-id", help="Base model (Hub id or local path).")
@click.option("--train-articles", type=int)
@click.option("--epochs", "num_train_epochs", type=float)
@click.option("--max-steps", type=int, help="Stop after N optimizer steps (overrides epochs).")
@click.option("--lr", "learning_rate", type=float)
@click.option("--batch-size", "per_device_train_batch_size", type=int)
@click.option("--grad-accum", "gradient_accumulation_steps", type=int)
@click.option("--max-length", type=int, help="Packed row length in tokens.")
@click.option("--lora-r", type=int)
@click.option("--lora-alpha", type=int)
@click.option("--gate-proj/--no-gate-proj", "lora_gate_proj", default=None, help="Also adapt LFM3's attention gate.")
@click.option("--train-embeddings/--no-train-embeddings", default=None, help="Fully train the input embeddings.")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=None)
@click.option("--grammar/--no-grammar", "run_grammar", default=None, help="Run the TurBLiMP grammar test.")
@click.option("--skip-baseline", is_flag=True, help="Do not evaluate the model before training.")
@click.option("--resume", is_flag=True, help="Resume from the last checkpoint in OUTPUT_DIR/checkpoints.")
@click.option("--merge", is_flag=True, help="Merge the adapter into a bf16 standalone model at the end.")
@click.option("--force-data", is_flag=True, help="Re-download the corpora.")
@pass_ctx
def train(obj: Ctx, skip_baseline, resume, merge, force_data, **opts) -> None:
    """Run CPT: baseline eval → LoRA training (live dashboard) → final eval → report."""
    from .train import run_training

    cfg = obj.config(**opts)
    run_training(cfg, console, resume=resume, skip_baseline=skip_baseline, merge=merge, force_data=force_data)


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("model_path", required=False)
@click.option("--adapter", type=click.Path(exists=True, file_okay=False), help="LoRA adapter to apply.")
@click.option("--grammar/--no-grammar", "run_grammar", default=None)
@click.option("--json-out", type=click.Path(dir_okay=False), help="Write the metrics as JSON.")
@pass_ctx
def evaluate(obj: Ctx, model_path, adapter, run_grammar, json_out) -> None:
    """Perplexity (Wikipedia + OOD) and TurBLiMP accuracy of a base, adapter or merged model.

    MODEL_PATH defaults to the configured base model; it may also be an adapter directory.
    """
    from .data import prepare_data
    from .device import configure_backend, detect, resolve_precision, torch_dtype
    from .modeling import adapter_base_model, is_adapter_dir, load_for_inference
    from .reporting import summary_table
    from .train import evaluate_all

    cfg = obj.config(run_grammar=run_grammar)
    info = detect()
    configure_backend(info)
    precision = resolve_precision(cfg.precision, info)
    target = model_path or cfg.model_id
    if adapter is None and is_adapter_dir(target):
        adapter, target = target, adapter_base_model(target)
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(precision), adapter, device_map, cfg.trust_remote_code
        )
    bundle = prepare_data(cfg, tokenizer, log=console.print)
    results = evaluate_all(model, tokenizer, bundle, cfg, console, "eval", cfg.run_grammar)
    console.print(summary_table(None, results, title=f"Evaluation of {adapter or target}"))
    if json_out:
        Path(json_out).write_text(json.dumps({"model": target, "adapter": adapter, **results}, indent=2))
        console.print(f"📝 {json_out}")


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("adapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), help="Default: ADAPTER_DIR/../merged")
@click.option("--base-model", help="Override the base model recorded in the adapter config.")
@click.option("--dtype", type=click.Choice(["bf16", "fp16", "fp32"]), default="bf16")
@click.option("--device", default="cpu", help="cpu (safe, default) or auto (GPU).")
@click.option("--push-to-hub", metavar="REPO_ID", help="Also push the merged model to the Hub.")
@pass_ctx
def merge(obj: Ctx, adapter_dir, output_dir, base_model, dtype, device, push_to_hub) -> None:
    """Merge a LoRA adapter into a fresh copy of the base weights (exact, in the base's dtype)."""
    from .device import torch_dtype
    from .modeling import merge_adapter

    cfg = obj.config()
    out = Path(output_dir) if output_dir else Path(adapter_dir).resolve().parent / "merged"
    with console.status("Merging ..."):
        merged = merge_adapter(adapter_dir, out, base_model, torch_dtype(dtype), device, cfg.trust_remote_code)
    console.print(f"💾 Merged model saved to {merged}")
    if push_to_hub:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with console.status(f"Pushing to {push_to_hub} ..."):
            AutoModelForCausalLM.from_pretrained(merged, trust_remote_code=cfg.trust_remote_code).push_to_hub(
                push_to_hub
            )
            AutoTokenizer.from_pretrained(merged, trust_remote_code=cfg.trust_remote_code).push_to_hub(push_to_hub)
        console.print(f"🚀 https://huggingface.co/{push_to_hub}")


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("model_path", required=False)
@click.option("--adapter", type=click.Path(exists=True, file_okay=False))
@click.option("--prompt", "prompts", multiple=True, help="Text to continue (repeatable).")
@click.option("--chat", is_flag=True, help="Wrap the prompt in the chat template instead of raw continuation.")
@click.option("--max-new-tokens", type=int, default=128)
@click.option("--temperature", type=float, default=0.3, help="0 = greedy.")
@click.option("--repetition-penalty", type=float, default=1.05)
@pass_ctx
def generate(obj: Ctx, model_path, adapter, prompts, chat, max_new_tokens, temperature, repetition_penalty) -> None:
    """Quick qualitative check: continue Turkish text with a base, adapter or merged model."""
    import torch

    from .device import detect, resolve_precision, torch_dtype
    from .modeling import load_for_inference

    cfg = obj.config()
    info = detect()
    target = model_path or cfg.model_id
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), adapter, device_map, cfg.trust_remote_code
        )
    prompts = prompts or ("İstanbul, Türkiye'nin", "Osmanlı İmparatorluğu'nun kuruluşu")
    device = model.get_input_embeddings().weight.device
    for prompt in prompts:
        if chat:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sampling = {"do_sample": temperature > 0}
        if temperature > 0:
            sampling["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )
        completion = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        console.rule()
        sep = "\n" if chat else ""
        console.print(f"[bold cyan]{prompt}[/]{sep}{completion}", markup=True, highlight=False)


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True))
def report(runs: tuple[str, ...]) -> None:
    """Show the before/after summary of one or more finished runs (dir or results.json)."""
    from .reporting import load_results, print_flip_examples, summary_table

    for run in runs:
        results = load_results(run)
        train = results.get("train", {})
        console.rule(f"[bold]{run}")
        if train:
            console.print(
                f"steps {train.get('global_step')} • train loss {train.get('train_loss', float('nan')):.4f} • "
                f"{train.get('runtime_s', 0) / 60:.1f} min • tokens {train.get('tokens_seen') or 0:,} • "
                f"peak mem {train.get('peak_mem_gb') or 0:.1f} GB"
            )
        if results.get("grammar_comparison"):
            print_flip_examples(console, results["grammar_comparison"])
        console.print(summary_table(results.get("baseline"), results.get("final")))


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[red]Interrupted[/red]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
