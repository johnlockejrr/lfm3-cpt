"""`lfm3-cpt` command line interface."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
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
    from .hub import quiet_libraries

    quiet_libraries()  # their tqdm bars turn into one line per redraw under Rich; we show our own progress
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
@click.option("--no-grammar", is_flag=True, help="No grammar test (TurBLiMP / grammar_files).")
@click.option("--force", is_flag=True, help="Re-download even if a matching cache exists.")
@click.option("--sft", "with_sft", is_flag=True, help="Also prepare the stage-2 (SFT) conversations.")
@pass_ctx
def prepare(obj: Ctx, data_dir, train_articles, eval_articles, ood_pages, no_grammar, force, with_sft) -> None:
    """Prepare and cache stage-1 data (Hub or your train_files), minimal pairs [+ SFT conversations].

    Also reports how well the tokenizer covers the training text.
    """
    from .data import prepare_data
    from .modeling import load_tokenizer
    from .sft_data import prepare_sft_data

    cfg = obj.config(
        data_dir=data_dir,
        train_articles=train_articles,
        eval_articles=eval_articles,
        ood_pages=ood_pages,
        run_grammar=False if no_grammar else None,
    )
    from .hub import ensure_model_files

    ensure_model_files(cfg.model_id, console)
    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    bundle = prepare_data(cfg, tokenizer, force=force, log=console.print)
    table = Table(title=f"Data in {cfg.data_dir}", show_header=False)
    for key, value in bundle.metadata.get("stats", {}).items():
        table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)
    corpus = bundle.metadata.get("corpus") or {}
    if corpus.get("eval_source"):
        console.print(f"🧪 evaluation text: {corpus['eval_source']}")
        for path in corpus.get("eval_files", []):
            console.print(f"   held out: {path}", highlight=False)
    console.print(f"\n📝 Sample:\n{bundle.train[0]['text'][:500]}")
    if with_sft:
        sft = prepare_sft_data(cfg, tokenizer, force=force, log=console.print)
        table = Table(title=f"SFT conversations ({cfg.sft_dataset})", show_header=False)
        for key, value in sft.metadata["stats"].items():
            if key == "mix":
                for source, m in value.items():
                    table.add_row(f"+ {source}", f"{m['train']:,} train / {m['eval']:,} eval (of {m['available']:,})")
                continue
            table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
        console.print(table)
        example = sft.train[0]
        console.print(
            f"\n💬 Sample:\n[cyan]user:[/] {example['prompt'][-1]['content'][:300]}\n"
            f"[green]assistant:[/] {example['completion'][0]['content'][:300]}",
            highlight=False,
        )


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
@click.option("--grammar/--no-grammar", "run_grammar", default=None, help="Run the minimal-pair grammar test.")
@click.option("--skip-baseline", is_flag=True, help="Do not evaluate the model before training.")
@click.option("--resume", is_flag=True, help="Resume from the last checkpoint in OUTPUT_DIR/checkpoints.")
@click.option("--merge", is_flag=True, help="Merge the adapter into a bf16 standalone model at the end.")
@click.option("--force-data", is_flag=True, help="Re-download the corpora.")
@click.option("--sft", "then_sft", is_flag=True, help="Then run stage 2 (Turkish SFT) on the CPT result.")
@pass_ctx
def train(obj: Ctx, skip_baseline, resume, merge, force_data, then_sft, **opts) -> None:
    """Run CPT: baseline eval → LoRA training (live dashboard) → final eval → report [→ SFT].

    With --sft the CPT adapter is merged (SFT needs a standalone model to start from) and stage 2 runs right
    after, with the `sft_*` settings. --merge then also merges the SFT adapter.
    """
    import gc

    import torch

    from .train import run_training

    cfg = obj.config(**opts)
    run_training(
        cfg, console, resume=resume, skip_baseline=skip_baseline, merge=merge or then_sft, force_data=force_data
    )
    if then_sft:
        from .sft import run_sft

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_sft(cfg, console, start="cpt", merge=merge, force_data=force_data)


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.option(
    "--from",
    "start",
    default="cpt",
    show_default=True,
    help='"cpt" (this run\'s CPT model), "base" (original model: SFT without CPT), or a model/adapter path.',
)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), help="The CPT run directory (SFT goes in ./sft).")
@click.option("--dataset", "sft_dataset", help="Hub dataset of conversations (default: Turkish Aya).")
@click.option("--dataset-config", "sft_dataset_config")
@click.option("--filter", "filter_", metavar="COLUMN=VALUE", help='Keep rows where COLUMN == VALUE ("" to disable).')
@click.option("--system-prompt", "sft_system_prompt", help="System prompt added to every conversation.")
@click.option("--max-train", "sft_max_train", type=int)
@click.option("--epochs", "sft_num_train_epochs", type=float)
@click.option("--max-steps", "sft_max_steps", type=int)
@click.option("--lr", "sft_learning_rate", type=float)
@click.option("--batch-size", "sft_per_device_train_batch_size", type=int)
@click.option("--grad-accum", "sft_gradient_accumulation_steps", type=int)
@click.option("--lora-r", "sft_lora_r", type=int)
@click.option("--thinking/--no-thinking", "sft_enable_thinking", default=None, help="LFM3 reasoning mode.")
@click.option(
    "--lm-eval/--no-lm-eval",
    default=True,
    show_default=True,
    help="Also re-check stage-1 perplexity and minimal pairs (forgetting).",
)
@click.option("--skip-baseline", is_flag=True)
@click.option("--resume", is_flag=True)
@click.option("--merge", is_flag=True, help="Merge the SFT adapter into a standalone chat model.")
@click.option("--force-data", is_flag=True)
@pass_ctx
def sft(obj: Ctx, start, filter_, lm_eval, skip_baseline, resume, merge, force_data, **opts) -> None:
    """Stage 2 (optional): instruction-tune the CPT model (Turkish conversations by default).

    \b
    Examples:
      lfm3-cpt -p rtx3090 sft                    # on top of this run's CPT model
      lfm3-cpt -p rtx3090 sft --from base -o runs/sft-only   # control: SFT without CPT
      lfm3-cpt sft --dataset turkish-nlp-suite/InstrucTurca --filter "" --max-train 20000
    """
    from .sft import run_sft

    if filter_ is not None:
        if filter_ == "":
            opts.update(sft_filter_column="", sft_filter_value="")
        elif "=" in filter_:
            col, val = filter_.split("=", 1)
            opts.update(sft_filter_column=col, sft_filter_value=val)
        else:
            raise click.UsageError("--filter expects COLUMN=VALUE or an empty string")
    cfg = obj.config(**opts)
    try:
        run_sft(
            cfg,
            console,
            start=start,
            resume=resume,
            skip_baseline=skip_baseline,
            merge=merge,
            force_data=force_data,
            lm_eval=lm_eval,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("model_path", required=False)
@click.option("--adapter", type=click.Path(exists=True, file_okay=False), help="LoRA adapter to apply.")
@click.option("--grammar/--no-grammar", "run_grammar", default=None)
@click.option("--json-out", type=click.Path(dir_okay=False), help="Write the metrics as JSON.")
@pass_ctx
def evaluate(obj: Ctx, model_path, adapter, run_grammar, json_out) -> None:
    """Held-out / OOD perplexity and minimal-pair accuracy of a base, adapter or merged model.

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
    from .hub import ensure_model_files

    ensure_model_files(target, console)
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(precision), adapter, device_map, cfg.trust_remote_code
        )
    bundle = prepare_data(cfg, tokenizer, log=console.print)
    results = evaluate_all(model, tokenizer, bundle, cfg, console, "eval", cfg.run_grammar)
    console.print(summary_table(None, results, title=f"Evaluation of {adapter or target}", single_label="value"))
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
    from .hub import ensure_model_files

    ensure_model_files(base_model or adapter_dir, console)
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
@click.option("--repetition-penalty", type=float, default=1.1)
@click.option("--no-repeat-ngram", type=int, default=0, help="Never repeat a sequence of this many tokens; 0 = off.")
@pass_ctx
def generate(
    obj: Ctx, model_path, adapter, prompts, chat, max_new_tokens, temperature, repetition_penalty, no_repeat_ngram
) -> None:
    """Quick qualitative check: continue text with a base, adapter or merged model.

    Without --prompt, two Turkish prompts are used (target_language=tr); other languages need --prompt.
    """
    import torch

    from .device import detect, resolve_precision, torch_dtype
    from .modeling import load_for_inference

    cfg = obj.config()
    info = detect()
    target = model_path or cfg.model_id
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    from .hub import ensure_model_files

    ensure_model_files(target, console)
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), adapter, device_map, cfg.trust_remote_code
        )
    if not prompts:
        if cfg.target_language != "tr":
            raise click.UsageError("give at least one --prompt in your language")
        prompts = ("İstanbul, Türkiye'nin", "Osmanlı İmparatorluğu'nun kuruluşu")
    from .textnorm import Normalizer

    prompts = tuple(Normalizer.from_config(cfg)(p) for p in prompts)
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
        if no_repeat_ngram > 0:
            sampling["no_repeat_ngram_size"] = no_repeat_ngram
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
    """Show the before/after summary of one or more finished runs (dir or results.json).

    For a run directory, the stage-2 (SFT) results in RUN/sft are shown too when present.
    """
    from .reporting import load_results

    for run in runs:
        path = Path(run)
        if path.is_dir() and not (path / "results.json").exists() and (path / "sft" / "results.json").exists():
            path = path / "sft"
        _report_one(str(path), load_results(path))
        if path.is_dir() and path.name != "sft" and (path / "sft" / "results.json").exists():
            _report_one(str(path / "sft"), load_results(path / "sft"))


def _report_one(name: str, results: dict[str, Any]) -> None:
    from .reporting import print_flip_examples, sft_summary_table, summary_table

    train = results.get("train", {})
    console.rule(f"[bold]{name}" + (" (SFT)" if results.get("stage") == "sft" else ""))
    if results.get("stage") == "sft":
        console.print(f"started from {results.get('start_model')}")
    if train:
        console.print(
            f"steps {train.get('global_step')} • train loss {train.get('train_loss', float('nan')):.4f} • "
            f"{train.get('runtime_s', 0) / 60:.1f} min • tokens {train.get('tokens_seen') or 0:,} • "
            f"peak mem {train.get('peak_mem_gb') or 0:.1f} GB"
        )
    if results.get("grammar_comparison"):
        print_flip_examples(console, results["grammar_comparison"])
    if results.get("stage") == "sft":
        language = (results.get("config") or {}).get("target_language", "tr")
        console.print(sft_summary_table(results.get("baseline"), results.get("final"), language=language))
    else:
        console.print(summary_table(results.get("baseline"), results.get("final")))


# ---------------------------------------------------------------------------------------------------------------
def _exit_after_streaming(streamed_from_hub: bool) -> None:
    """`datasets` streaming leaves background download threads behind when we stop reading early (--inspect, a
    budget), and Python then waits for them, sometimes for minutes. All files are closed at this point, so exit
    directly. Set LFM3_CPT_NO_HARD_EXIT=1 to disable."""
    import os

    if streamed_from_hub and os.environ.get("LFM3_CPT_NO_HARD_EXIT", "") not in {"1", "true", "yes"}:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _parse_size(value: str | None) -> int | None:
    """'250M' -> 250_000_000, '1.5G' -> 1_500_000_000, '20000' -> 20000."""
    if value is None:
        return None
    v = value.strip().upper().replace("_", "")
    mult = {"K": 10**3, "M": 10**6, "G": 10**9, "B": 10**9}.get(v[-1:], 1)
    number = v[:-1] if mult != 1 else v
    try:
        return int(float(number) * mult)
    except ValueError as exc:
        raise click.BadParameter(f"not a size: {value!r} (e.g. 20000, 250M, 1.5G)") from exc


@cli.command()
@click.argument("dataset", required=False)
@click.option("--subset", "ds_config", help="Dataset subset (configuration), e.g. heb_Hebr or 20231101.he.")
@click.option("--split", default="train", show_default=True)
@click.option("--data-files", multiple=True, help="Instead of DATASET: local files/globs or hf:// URLs (repeatable).")
@click.option("--builder", type=click.Choice(["json", "csv", "parquet", "text"]), help="Format of --data-files.")
@click.option("--inspect", is_flag=True, help="Only print the structure of the first records, to find the text field.")
@click.option("-o", "--output", "prefix", help="Output prefix: writes PREFIX.train.jsonl and PREFIX.eval.jsonl.")
@click.option(
    "--field",
    "text_field",
    default="text",
    show_default=True,
    help="Text field (dotted path). A list is joined: strings directly, records via --item-field.",
)
@click.option("--item-field", help="For a list of records: the field holding the text of each item.")
@click.option("--join", default="\\n", show_default=True, help="Separator when joining a list (\\n, \\n\\n, space).")
@click.option("--title-field", help="Prepend this field (e.g. title) to the text, separated by a blank line.")
@click.option("--min-chars", type=int, default=200, show_default=True, help="Drop shorter documents.")
@click.option("--eval-docs", type=int, default=200, show_default=True, help="Documents written to the eval file first.")
@click.option("--max-docs", help="Training document budget (e.g. 50000, 50K); 0 = write only the eval file.")
@click.option("--max-chars", help="Training character budget (e.g. 200M). Whichever budget is hit first wins.")
@click.option("--skip", type=int, default=0, help="Skip the first N records (e.g. to take a disjoint slice).")
@click.option(
    "--shuffle-buffer",
    type=int,
    default=10_000,
    show_default=True,
    help="Streaming shuffle buffer (0 = keep the dataset order).",
)
@click.option("--seed", type=int, default=42, show_default=True)
def export(
    dataset,
    ds_config,
    split,
    data_files,
    builder,
    inspect,
    prefix,
    text_field,
    item_field,
    join,
    title_field,
    min_chars,
    eval_docs,
    max_docs,
    max_chars,
    skip,
    shuffle_buffer,
    seed,
) -> None:
    """Stream a Hub dataset (or raw files) into budgeted train/eval JSONL files for `train_files`.

    \b
    Examples:
      lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr --inspect
      lfm3-cpt export HuggingFaceFW/fineweb-2 --subset heb_Hebr -o corpora/he/fineweb2 --max-chars 200M
      lfm3-cpt export wikimedia/wikipedia --subset 20231101.he --title-field title -o corpora/he/wiki --max-chars 100M
    """
    import itertools

    from .hub_export import ExportStats, Source, describe, export_corpus, stream_records

    if not dataset and not data_files:
        raise click.UsageError("give a DATASET id or --data-files")
    source = Source(dataset, ds_config, split, list(data_files), builder)
    doc_budget, char_budget = _parse_size(max_docs), _parse_size(max_chars)  # validate before streaming
    try:
        records = stream_records(source, 0 if inspect else shuffle_buffer, seed)
        if inspect:
            for i, record in enumerate(itertools.islice(records, 3)):
                console.rule(f"record {i}")
                for line in describe(record):
                    console.print(line, highlight=False, markup=False)
            console.print("\nUse --field (and --item-field for lists of records) to choose the text.")
            _exit_after_streaming(bool(source.dataset))
            return
    except Exception as exc:  # unknown dataset, missing config, no network...
        raise click.ClickException(f"cannot read {source.describe()}: {exc}") from exc
    if not prefix:
        raise click.UsageError("give -o/--output PREFIX (or use --inspect)")
    joiner = join.replace("\\n", "\n").replace("\\t", "\t")

    def show(st: ExportStats) -> None:
        status.update(
            f"{source.describe()}: {st.records_seen:,} records read • train {st.train_docs:,} docs "
            f"/ {st.train_chars / 1e6:,.1f}M chars • eval {st.eval_docs:,}"
        )

    with console.status(f"Streaming {source.describe()} ...") as status:
        try:
            stats = export_corpus(
                records,
                prefix,
                text_field,
                item_field,
                joiner,
                title_field,
                min_chars,
                eval_docs,
                doc_budget,
                char_budget,
                skip,
                show,
            )
        except Exception as exc:
            raise click.ClickException(f"export failed: {exc}") from exc
    table = Table(title=f"Exported {source.describe()}", show_header=False)
    if stats.files.get("train"):
        table.add_row(
            "train", f"{stats.files['train']} • {stats.train_docs:,} docs • {stats.train_chars / 1e6:,.1f}M chars"
        )
    if stats.files.get("eval"):
        table.add_row(
            "eval", f"{stats.files['eval']} • {stats.eval_docs:,} docs • {stats.eval_chars / 1e6:,.1f}M chars"
        )
    table.add_row(
        "records read",
        f"{stats.records_seen:,} ({stats.skipped_empty:,} without text, "
        f"{stats.skipped_short:,} shorter than {min_chars})",
    )
    console.print(table)
    if stats.train_docs + stats.eval_docs == 0:
        raise click.ClickException(f"no text found in field {text_field!r}: run with --inspect to see the records")
    if stats.exhausted and (max_docs or max_chars):
        console.print("[yellow]The source ran out before the budget was reached.[/yellow]")
    _exit_after_streaming(bool(source.dataset))


@cli.command()
@click.argument("lang")
@click.option("-o", "--output", required=True, help="Pairs file to write (JSONL), for grammar_files.")
@click.option("--limit", type=int, help="Keep a random sample of N pairs.")
@click.option(
    "--tsv",
    "tsv_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Use an already downloaded MultiBLiMP data.tsv instead of the Hub.",
)
def multiblimp(lang, output, limit, tsv_path) -> None:
    """Minimal pairs for LANG (ISO 639-3, e.g. heb) from MultiBLiMP 1.0 (subject-verb agreement).

    MultiBLiMP (Jumelet et al. 2025, CC-BY-4.0) derives pairs from Universal Dependencies for 101 languages.
    """
    from .hub_export import download_multiblimp, multiblimp_rows, write_pairs

    try:
        text = Path(tsv_path).read_text(encoding="utf-8") if tsv_path else download_multiblimp(lang)
        rows = multiblimp_rows(text)
    except Exception as exc:
        raise click.ClickException(f"cannot get MultiBLiMP pairs for {lang!r}: {exc}") from exc
    n = write_pairs(rows, output, limit)
    kinds = sorted({r["phenomenon"] for r in rows})
    console.print(f"🧩 {n:,} pairs → {output} ({', '.join(kinds)})")
    console.print(f"   use with: -s grammar_files={output}")


@cli.command()
@click.option(
    "--file",
    "library_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Sefaria/hebrew_library data.jsonl already on disk (default: download it once from the Hub, ~7.7 GB).",
)
@click.option("--preset", type=click.Choice(["tannaitic", "responsa"]), help="A ready-made selection of works.")
@click.option("--title", "titles", multiple=True, help="Select books whose title matches this regex (repeatable).")
@click.option("--category", "categories", multiple=True, help="Select by docCategory substring (repeatable).")
@click.option("--exclude", "excludes", multiple=True, help="Drop books whose title/category matches (repeatable).")
@click.option("--professional-only", is_flag=True, help="Only segments marked dataQuality=professional.")
@click.option(
    "--prefer-version",
    "prefer",
    multiple=True,
    help="Prefer editions (or books) whose title matches, e.g. Kaufmann, 'codex Vienna' (repeatable, in order).",
)
@click.option(
    "--keep-all-editions",
    is_flag=True,
    help="Keep both books when Sefaria has a work twice (e.g. Tosefta X and Tosefta X (Lieberman)).",
)
@click.option("--stats", is_flag=True, help="Only show categories and the books/editions the selection catches.")
@click.option("-o", "--output", "prefix", help="Writes PREFIX.train.jsonl, PREFIX.eval.jsonl, PREFIX.books.tsv.")
@click.option("--eval-fraction", type=float, default=0.05, show_default=True, help="Share held out, by tractate.")
@click.option("--eval-match", multiple=True, help="Hold out exactly these books/tractates (regex, repeatable).")
@click.option("--min-chars", type=int, default=100, show_default=True, help="Drop shorter documents (chapters).")
@click.option("--max-chars", "max_chars", help="Training budget (e.g. 30M), filled with whole books at random.")
@click.option("--max-doc-chars", type=int, default=20_000, show_default=True, help="Split longer chapters.")
@click.option(
    "--native-marks/--keep-quotes",
    default=True,
    show_default=True,
    help="Turn \" and ' in abbreviations into gershayim ״ / geresh ׳ and remove quotation marks.",
)
@click.option(
    "--maqaf-to-space/--keep-maqaf",
    default=True,
    show_default=True,
    help="Replace the maqaf (and a hyphen between Hebrew words) with a space.",
)
@click.option("--strip-punctuation", is_flag=True, help="Also remove modern punctuation (. , ; ! ? dashes).")
@click.option("--drop-editorial", is_flag=True, help="Remove bracketed editorial matter: [...] (...) <...>.")
@click.option("--max-latin", type=float, default=0.02, show_default=True, help="Drop segments with more Latin.")
@click.option("--pairs", "pairs_path", help="Also write minimal pairs from the held-out text (JSONL).")
@click.option("--pairs-count", type=int, default=1000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def sefaria(
    library_file,
    preset,
    titles,
    categories,
    excludes,
    professional_only,
    prefer,
    keep_all_editions,
    stats,
    prefix,
    eval_fraction,
    eval_match,
    min_chars,
    max_chars,
    max_doc_chars,
    native_marks,
    maqaf_to_space,
    strip_punctuation,
    drop_editorial,
    max_latin,
    pairs_path,
    pairs_count,
    seed,
) -> None:
    """Rabbinic Hebrew training text from Sefaria's library (Sefaria/hebrew_library).

    One edition per book, cleaned, rebuilt into chapters, with whole tractates held out for evaluation.

    \b
    Examples:
      lfm3-cpt sefaria --preset tannaitic --stats
      lfm3-cpt sefaria --preset tannaitic --prefer-version Kaufmann --prefer-version 'codex Vienna' \\
          --strip-punctuation --drop-editorial -o corpora/rabbinic/tannaitic --pairs corpora/rabbinic/pairs.jsonl
      lfm3-cpt sefaria --preset responsa --eval-fraction 0.02 --max-chars 0 -o corpora/rabbinic/responsa
    """
    from .sefaria import (
        Cleaner,
        Selector,
        build_corpus,
        download_library,
        iter_segments,
        library_stats,
        make_pairs,
        plan_books,
        work_title,
        write_jsonl,
        write_manifest,
    )

    if not stats and not prefix:
        raise click.UsageError("give -o/--output PREFIX (or --stats)")
    char_budget = _parse_size(max_chars)
    try:
        selector = Selector.from_preset(preset, titles, categories, excludes, professional_only=professional_only)
    except (ValueError, re.error) as exc:
        raise click.UsageError(str(exc)) from exc
    for pattern in (*prefer, *eval_match):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise click.UsageError(f"not a regular expression: {pattern!r} ({exc})") from exc
    if library_file:
        path = Path(library_file)
    else:
        console.print(f"Getting {SEFARIA_LABEL} (cached after the first download) ...")
        try:
            path = download_library()
        except Exception as exc:
            raise click.ClickException(f"cannot download the Sefaria library: {exc}; use --file") from exc

    with console.status(f"Reading {path.name} ...") as status:

        def progress(n: int) -> None:
            status.update(f"Reading {path.name}: {n:,} segments ...")

        segments = iter_segments(path, progress)
        if stats:
            st = library_stats(segments, selector)
        else:
            cleaner = Cleaner(
                native_marks=native_marks,
                strip_punctuation=strip_punctuation,
                drop_editorial=drop_editorial,
                max_latin=max_latin,
                maqaf_to_space=maqaf_to_space,
            )
            result = build_corpus(
                segments,
                selector,
                prefer,
                min_chars,
                max_doc_chars,
                eval_fraction,
                eval_match,
                seed,
                cleaner,
                char_budget,
                keep_all_editions,
            )

    if stats:
        table = Table(title=f"Sefaria library: {st.lines:,} segments, top categories by size")
        table.add_column("docCategory")
        table.add_column("segments", justify="right")
        table.add_column("M chars", justify="right")
        for cat, chars in st.categories.most_common(40):
            table.add_row(cat or "—", f"{st.category_segments[cat]:,}", f"{chars / 1e6:,.1f}")
        console.print(table)
        try:
            plan = plan_books(st.books, [re.compile(p) for p in prefer], keep_all_editions)
        except re.error as exc:
            raise click.UsageError(f"--prefer-version: {exc}") from exc
        works: dict[str, list[str]] = {}
        for book in sorted(plan.version):
            works.setdefault(work_title(book), []).append(book)
        table_w = Table(title=f"Selected: {len(works)} works ({len(plan.version)} books), the edition each would use")
        table_w.add_column("work")
        table_w.add_column("category")
        table_w.add_column("books", justify="right")
        table_w.add_column("edition used (of how many)")
        table_w.add_column("M chars", justify="right")
        total = 0
        for work, members in works.items():
            chosen = Counter(plan.version[b] for b in members)
            name, _ = chosen.most_common(1)[0]
            if len(chosen) > 1:
                name += f"  [+{len(chosen) - 1} other]"
            n_editions = max(len(st.books[b]) for b in members)
            chars = sum(st.books[b][plan.version[b]]["chars"] for b in members)
            total += chars
            table_w.add_row(
                work,
                st.book_category.get(members[0], ""),
                str(len(members)),
                f"{name} ({n_editions})",
                f"{chars / 1e6:.2f}",
            )
        console.print(table_w)
        if plan.dropped:
            shown = ", ".join(f"{b} → {k}" for b, k in list(sorted(plan.dropped.items()))[:4])
            console.print(
                f"{len(plan.dropped)} books left out as other editions of a kept work "
                f"(--prefer-version decides, --keep-all-editions keeps both): {shown} …"
            )
        console.print(f"≈ {total / 1e6:,.1f}M characters (without vowel points), before cleaning.")
        return

    if not result.books:
        raise click.ClickException("nothing selected: run with --stats to see titles and categories")
    # No empty files: an empty PREFIX.train.jsonl would still match train_files globs (--max-chars 0 = eval only).
    for split in ("train", "eval"):
        Path(f"{prefix}.{split}.jsonl").unlink(missing_ok=True)
    n_train = write_jsonl(result.train, f"{prefix}.train.jsonl") if result.train else 0
    n_eval = write_jsonl(result.eval, f"{prefix}.eval.jsonl") if result.eval else 0
    write_manifest(result.books, f"{prefix}.books.tsv")
    chars = {s: sum(len(d["text"]) for d in getattr(result, s)) for s in ("train", "eval")}
    table = Table(title=f"Sefaria export: {len(result.books)} books", show_header=False)
    if n_train:
        table.add_row("train", f"{prefix}.train.jsonl • {n_train:,} docs • {chars['train'] / 1e6:,.2f}M chars")
    if n_eval:
        held = sorted({b.book for b in result.books if b.split == "eval"})
        shown = ", ".join(held[:6]) + (f" … (+{len(held) - 6})" if len(held) > 6 else "")
        table.add_row("eval", f"{prefix}.eval.jsonl • {n_eval:,} docs • {chars['eval'] / 1e6:,.2f}M chars")
        table.add_row("held out", shown)
    unused = sum(b.split == "unused" for b in result.books)
    if unused:
        table.add_row("budget", f"{unused:,} books left out to stay within {max_chars} characters")
    table.add_row("manifest", f"{prefix}.books.tsv (edition chosen per book, duplicates dropped)")
    for what, n in cleaner.stats.items():
        if n:
            table.add_row("cleaning", f"{what}: {n:,}")
    console.print(table)
    if pairs_path:
        if not result.eval:
            console.print("[yellow]No held-out text, so no pairs written.[/yellow]")
        else:
            pairs = make_pairs(result.eval, pairs_count, seed)
            write_jsonl(pairs, pairs_path)
            console.print(f"🧩 {len(pairs):,} minimal pairs from the held-out text → {pairs_path}")


SEFARIA_LABEL = "Sefaria/hebrew_library data.jsonl"


# ---------------------------------------------------------------------------------------------------------------
@cli.command("tokenizer-check")
@click.argument("files", nargs=-1)
@click.option("--text", "texts", multiple=True, help="Text to measure (repeatable), instead of files.")
@click.option("--model", "model_path", help="Tokenizer to check (default: the configured model).")
@click.option("--examples", type=int, default=6, show_default=True, help="Words to show split into tokens.")
@pass_ctx
def tokenizer_check(obj: Ctx, files, texts, model_path, examples) -> None:
    """How well the tokenizer covers your language (compared with English). Run it before training.

    FILES are glob patterns (.txt / .jsonl, like train_files). Without FILES or --text, the configured
    train_files, or else the cached training data, is measured. Normalization settings apply.
    """
    from .corpus_files import expand, read_documents
    from .data import is_prepared, load_data
    from .modeling import load_tokenizer
    from .textnorm import Normalizer
    from .tokcheck import tokenizer_report

    cfg = obj.config()
    norm = Normalizer.from_config(cfg)
    if texts:
        docs = list(texts)
    elif files or cfg.train_files:
        try:
            docs = [d for spec in expand(list(files) or cfg.train_files) for d in read_documents(spec.path, cfg)]
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
    elif is_prepared(cfg):
        docs = list(load_data(cfg).train["text"])
    else:
        raise click.UsageError("give FILES or --text, set train_files, or run `prepare` first")
    docs = [norm(d) for d in docs if d.strip()]
    if not docs:
        raise click.ClickException("no text to measure")
    from .hub import ensure_model_files

    ensure_model_files(model_path or cfg.model_id, console)
    tokenizer = load_tokenizer(model_path or cfg.model_id, cfg.trust_remote_code)
    report = tokenizer_report(tokenizer, docs)
    table = Table(title=f"Tokenizer: {model_path or cfg.model_id}", header_style="bold")
    table.add_column("")
    table.add_column("your text", justify="right")
    table.add_column("English reference", justify="right")
    t, e = report["target"], report["english"]
    table.add_row("words measured", f"{t['words']:,}", f"{e['words']:,}")
    table.add_row("tokens per word", f"{t['tokens_per_word']:.2f}", f"{e['tokens_per_word']:.2f}")
    table.add_row("characters per token", f"{t['chars_per_token']:.2f}", f"{e['chars_per_token']:.2f}")
    table.add_row("unknown-token rate", f"{t['unk_rate']:.2%}", f"{e['unk_rate']:.2%}")
    console.print(table)
    console.print(
        f"tokens per word vs English: [bold]{report['tokens_per_word_vs_english']:.2f}×[/] • "
        f"verdict: [bold]{report['verdict']}[/]"
    )
    words = list(dict.fromkeys(w for d in docs[:50] for w in d.split() if len(w) > 3))[:examples]
    for w in words:
        pieces = [tokenizer.decode([i]) for i in tokenizer(w, add_special_tokens=False)["input_ids"]]
        console.print(f"  {w} → {pieces}", highlight=False)


# ---------------------------------------------------------------------------------------------------------------
@cli.command()
@click.argument("prompts")
@click.option("--model", "model_path", help="Model that answers (default: the configured base model).")
@click.option("-o", "--output", required=True, help='JSONL to write ({"messages": [...]} per line), for sft_mix.')
@click.option("--subset", help="Dataset subset, e.g. main for openai/gsm8k.")
@click.option("--split", default="train", show_default=True)
@click.option("--prompt-field", help="Column holding the prompt (default: detected, as for SFT data).")
@click.option("--max", "max_prompts", type=int, help="Use at most N prompts (a random sample).")
@click.option(
    "--language",
    default="en",
    show_default=True,
    help="Keep only answers in this language: en, he, tr, la, ar, ... or any.",
)
@click.option(
    "--verify-number",
    is_flag=True,
    help="Maths sets with a reference answer (GSM8K): keep only answers with the same final number.",
)
@click.option(
    "--thinking/--no-thinking",
    default=False,
    show_default=True,
    help="Let the model reason first; only its final answer is saved (needs more --max-new-tokens).",
)
@click.option("--max-new-tokens", type=int, default=1024, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--repetition-penalty", type=float, default=1.1, show_default=True)
@click.option("--resume", is_flag=True, help="Continue an interrupted run (appends to OUTPUT).")
@click.option("--seed", type=int, default=42, show_default=True)
@pass_ctx
def distill(
    obj: Ctx,
    prompts,
    model_path,
    output,
    subset,
    split,
    prompt_field,
    max_prompts,
    language,
    verify_number,
    thinking,
    max_new_tokens,
    batch_size,
    repetition_penalty,
    resume,
    seed,
) -> None:
    """Have a model (by default the original LFM3) answer PROMPTS, and save the answers as SFT data.

    PROMPTS is a Hub dataset, a local .jsonl/.json/.csv/.parquet file, or a .txt file with one prompt per line.
    Only the prompts are used. Answers that do not finish, are empty, or are not in --language are dropped.

    \b
    Examples:
      lfm3-cpt distill HuggingFaceH4/no_robots -o corpora/sft/en-no-robots.jsonl --max 3000
      lfm3-cpt distill openai/gsm8k --subset main --thinking --max-new-tokens 2048 \\
          --verify-number -o corpora/sft/en-gsm8k.jsonl --max 1000
    """
    import random

    from datasets import load_dataset

    from .device import configure_backend, detect, resolve_precision, torch_dtype
    from .distill import distill as run_distill
    from .distill import messages_preview, prompts_from_records, prompts_from_text
    from .hub import ensure_model_files
    from .modeling import load_for_inference
    from .sft_data import _load_local, is_local_dataset

    cfg = obj.config()
    try:
        if prompts.endswith(".txt"):
            items = prompts_from_text(prompts)
        elif is_local_dataset(prompts):
            items = prompts_from_records(_load_local(prompts), prompt_field)
        else:
            items = prompts_from_records(load_dataset(prompts, subset, split=split), prompt_field)
    except Exception as exc:
        raise click.ClickException(f"cannot read prompts from {prompts}: {exc}") from exc
    if not items:
        raise click.ClickException("no prompts found (try --prompt-field)")
    if verify_number and not any(it.get("reference") for it in items):
        raise click.UsageError("--verify-number needs reference answers in the dataset")
    if max_prompts and len(items) > max_prompts:
        items = random.Random(seed).sample(items, max_prompts)

    info = detect()
    configure_backend(info)
    target = model_path or cfg.model_id
    ensure_model_files(target, console)
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), None, device_map, cfg.trust_remote_code
        )
    from .callbacks import make_progress

    console.print(
        f"🧪 {len(items):,} prompts from {prompts} • answered by {target} • thinking {'on' if thinking else 'off'}"
    )
    with make_progress(console) as progress:
        task = progress.add_task("distilling", total=len(items))

        def show(st) -> None:
            progress.update(task, completed=st.processed)

        stats = run_distill(
            model,
            tokenizer,
            items,
            output,
            {"enable_thinking": thinking},
            batch_size,
            256,
            max_new_tokens,
            repetition_penalty,
            language,
            verify_number,
            resume,
            target,
            show,
        )
    table = Table(title=f"Distilled → {output}", show_header=False)
    for key, value in stats.to_dict().items():
        table.add_row(key.replace("_", " "), f"{value:,}")
    console.print(table)
    if stats.kept:
        first = json.loads(Path(output).read_text(encoding="utf-8").splitlines()[0])
        console.print(
            f"💬 {messages_preview(first['messages'][:-1])}\n   → {messages_preview(first['messages'])}",
            highlight=False,
        )
        console.print(f"   use with: -s 'sft_mix={output}@0.25'")
    if stats.unfinished > stats.kept:
        console.print(
            "[yellow]Most answers did not finish: raise --max-new-tokens (especially with --thinking).[/yellow]"
        )
    if stats.kept == 0:
        raise click.ClickException(f"no answers were kept, so {output} has nothing to mix in (see the reasons above)")


# ---------------------------------------------------------------------------------------------------------------
def default_lm_model(cfg: Config) -> str:
    run = Path(cfg.output_dir)
    for candidate in (run / "merged", run / "adapter"):
        if (candidate / "config.json").exists() or (candidate / "adapter_config.json").exists():
            return str(candidate)
    return cfg.model_id


@cli.group()
def cloze() -> None:
    """Gap-filling benchmark: restore a masked word among plausible alternatives.

    \b
    Build the items once, then score any number of models on the same file:
      lfm3-cpt -c CONFIG cloze build -o cloze.jsonl --vocab-file corpora/x/train.jsonl [--proposer MODEL]
      lfm3-cpt -c CONFIG cloze score cloze.jsonl -m MODEL_A -m MODEL_B
    """


def _load_lm(cfg: Config, target: str):
    from .device import detect, resolve_precision, torch_dtype
    from .hub import ensure_model_files
    from .modeling import load_for_inference

    info = detect()
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    ensure_model_files(target, console)
    with console.status(f"Loading {target} ..."):
        return load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), None, device_map, cfg.trust_remote_code
        )


def _sampler(model: Any, tokenizer: Any, samples: int, max_new_tokens: int, temperature: float = 1.0):
    """fn(left context) -> `samples` sampled continuations, as the text that follows the context."""
    import torch

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device

    @torch.no_grad()
    def sample(left: str) -> list[str]:
        ids = bos + tokenizer(left, add_special_tokens=False)["input_ids"]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        out = model.generate(
            torch.tensor([ids], device=device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
            do_sample=True,
            temperature=temperature,
            top_k=50,
            max_new_tokens=max_new_tokens,
            num_return_sequences=samples,
            pad_token_id=pad,
        )
        conts = []
        for seq in out:
            full = tokenizer.decode(seq, skip_special_tokens=True)
            conts.append(full[len(prompt) :] if full.startswith(prompt) else "")
        return conts

    return sample


def _read_specs(cfg: Config, specs: list[str]) -> list[str]:
    from .corpus_files import expand, read_documents

    docs: list[str] = []
    for spec in expand(specs):
        docs += read_documents(spec.path, cfg)
    return docs


@cloze.command("build")
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False), help="Items file (JSONL).")
@click.option("--eval-file", "eval_files", multiple=True, help="Held-out text to mask (default: eval_files).")
@click.option(
    "--vocab-file",
    "vocab_files",
    multiple=True,
    help="Text for frequencies and corpus distractors (default: the first train_files entry). Use the target "
    "corpus only, not the replay.",
)
@click.option("-n", "--items", "n_items", type=int, default=400, show_default=True)
@click.option(
    "--corpus-distractors", type=int, default=4, show_default=True, help="Same form class, similar frequency."
)
@click.option("--proposer", help="Model whose own predictions become extra wrong options (the hard ones).")
@click.option("--model-distractors", type=int, default=3, show_default=True, help="Used with --proposer.")
@click.option("--samples", type=int, default=24, show_default=True, help="Proposer samples per item.")
@click.option(
    "--stop-top", type=int, default=150, show_default=True, help="The N most frequent words are never masked."
)
@click.option("--seed", type=int, default=0, show_default=True)
@pass_ctx
def cloze_build(
    obj: Ctx,
    output,
    eval_files,
    vocab_files,
    n_items,
    corpus_distractors,
    proposer,
    model_distractors,
    samples,
    stop_top,
    seed,
) -> None:
    """Build gap-filling items from held-out text."""
    from .cloze import Vocabulary, build_items
    from .textnorm import Normalizer

    cfg = obj.config()
    norm = Normalizer.from_config(cfg)
    eval_specs = list(eval_files) or list(cfg.eval_files)
    vocab_specs = list(vocab_files) or list(cfg.train_files[:1])
    if not eval_specs or not vocab_specs:
        raise click.UsageError("give --eval-file and --vocab-file (or eval_files / train_files in the config)")
    try:
        eval_docs = [norm(d) for d in _read_specs(cfg, eval_specs)]
        vocab_docs = [norm(d) for d in _read_specs(cfg, vocab_specs)]
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"📚 masking {len(eval_docs):,} held-out docs ({', '.join(eval_specs)}); "
        f"vocabulary from {', '.join(vocab_specs)}",
        highlight=False,
    )
    vocab = Vocabulary.from_docs(vocab_docs, stop_top)

    propose = None
    if proposer:
        from .restore import first_words

        model, tokenizer = _load_lm(cfg, proposer)
        sample = _sampler(model, tokenizer, samples, max_new_tokens=8)

        def propose(left: str, answer: str) -> list[str]:
            return [norm(w) for w, _ in first_words(sample(left), 1).most_common()]

    with console.status("Building items ...") as status:
        items = build_items(
            eval_docs,
            vocab,
            n_items,
            seed,
            corpus_distractors,
            model_distractors if proposer else 0,
            propose,
            progress=lambda n: status.update(f"Building items: {n:,}/{n_items:,} ..."),
        )
    if not items:
        raise click.ClickException("no items: the held-out text has no maskable words")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_model = sum(k == "model" for it in items for k in it["kinds"].values())
    avg = sum(len(it["candidates"]) for it in items) / len(items)
    console.print(
        f"🧩 {len(items):,} items, {avg:.1f} candidates each"
        + (f" ({n_model / len(items):.1f} proposed by {proposer})" if proposer else "")
        + f" → {output}",
        highlight=False,
    )
    ex = items[0]
    console.print(f"   e.g. {ex['text']}\n        answer {ex['answer']} • options {' / '.join(ex['candidates'])}")


@cloze.command("score")
@click.argument("items_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-m", "--model", "models", multiple=True, help="Model to score (repeatable; default: this run's model).")
@click.option(
    "--output", type=click.Path(dir_okay=False), help="Per-item results (JSONL), one line per model and item."
)
@click.option("--show", type=int, default=5, show_default=True, help="Print this many misses per model.")
@click.option(
    "--pmi",
    "pmis",
    type=float,
    multiple=True,
    help="Also score with -α·log p(word alone) (repeatable: --pmi 0.5 --pmi 1); 0 = raw probability.",
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@pass_ctx
def cloze_score(obj: Ctx, items_file, models, output, show, pmis, batch_size) -> None:
    """Score models on a gap-filling items file."""
    from .cloze import BUCKETS, score_items, summarize
    from .evaluation import sentence_logprobs

    cfg = obj.config()
    items = [json.loads(line) for line in Path(items_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        raise click.ClickException(f"{items_file} has no items")
    targets = list(models) or [default_lm_model(cfg)]
    alphas = [0.0, *(a for a in pmis if a)]
    summaries: dict[str, dict[str, Any]] = {}  # column label -> summary
    columns: list[str] = []
    out_fh = open(output, "w", encoding="utf-8") if output else None  # noqa: SIM115
    try:
        for target in targets:
            model, tokenizer = _load_lm(cfg, target)
            total = sum(len(it["candidates"]) for it in items)
            if len(alphas) > 1:
                total += len({c for it in items for c in it["candidates"]})
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as bar:
                task = bar.add_task(f"scoring {Path(target).name or target}", total=total)

                cache: dict[str, float] = {}

                def logprob_fn(
                    texts: list[str], model=model, tokenizer=tokenizer, task=task, cache=cache
                ) -> list[float]:
                    todo = sorted({t for t in texts if t not in cache})
                    if todo:
                        got = sentence_logprobs(
                            model, tokenizer, todo, batch_size, progress=lambda k: bar.advance(task, k)
                        )
                        cache.update(zip(todo, got, strict=True))
                    return [cache[t] for t in texts]

                try:
                    by_alpha = {a: score_items(logprob_fn, items, pmi=a) for a in alphas}
                except ValueError as exc:
                    raise click.ClickException(str(exc)) from exc
            name = Path(target).parent.name if Path(target).name in {"merged", "adapter"} else target
            for a, res in by_alpha.items():
                label = name if len(alphas) == 1 else f"{name}\nα={a:g}"
                summaries[label] = summarize(res)
                columns.append(label)
            results = by_alpha[0.0]
            misses = [r for r in results if r["rank"] > 1][:show]
            if misses:
                console.print(f"[bold]Misses of {target}[/bold] (answer → model's choice):", highlight=False)
                by_id = {it.get("id"): it for it in items}
                for r in misses:
                    console.print(
                        f"  {by_id[r['id']]['text']}\n    {r['answer']} → {r['best']} (rank {r['rank']}/{r['n']})",
                        highlight=False,
                    )
            if out_fh:
                for a, res in by_alpha.items():
                    for r in res:
                        out_fh.write(json.dumps({"model": target, "pmi": a, **r}, ensure_ascii=False) + "\n")
            model = tokenizer = None
            _free_memory()
    finally:
        if out_fh:
            out_fh.close()

    table = Table(title=f"Gap filling: {items_file}", header_style="bold")
    table.add_column("metric")
    for col in columns:
        table.add_column(col, justify="right")
    first = summaries[columns[0]]
    rows: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("items", lambda s: f"{s['items']:,}"),
        ("candidates per item", lambda s: f"{s['candidates']:.1f}"),
        ("chance (top-1)", lambda s: f"{s['chance_top1']:.1%}"),
        ("top-1 accuracy", lambda s: f"{s['top1']:.1%}"),
        ("top-1, ignoring a ו/ה prefix", lambda s: f"{s['top1_lenient']:.1%}"),
        ("top-3 accuracy", lambda s: f"{s['top3']:.1%}"),
        ("mean rank of the answer", lambda s: f"{s['mean_rank']:.2f}"),
        ("mean reciprocal rank", lambda s: f"{s['mrr']:.3f}"),
    ]
    for name, _, _ in BUCKETS:
        if name in first["by_bucket"]:
            rows.append(
                (
                    f"  top-1, {name} words (n={first['by_bucket'][name][1]})",
                    lambda s, name=name: f"{s['by_bucket'].get(name, (0, 0))[0]:.1%}",
                )
            )
    for kind in sorted(first["pairwise"]):
        rows.append(
            (f"  beats {kind} options (pairwise)", lambda s, kind=kind: f"{s['pairwise'].get(kind, (0, 0))[0]:.1%}")
        )
    for label, fmt in rows:
        table.add_row(label, *(fmt(summaries[c]) for c in columns))
    console.print(table)
    if output:
        console.print(f"📝 {output}")


@cli.command()
@click.argument("model_path", required=False)
@click.option("--text", required=True, help="Text with the gap marker: 'אין ___ מהן לענין כתובה'.")
@click.option("--words", type=int, default=1, show_default=True, help="Words in the gap (0 = any number).")
@click.option("--letters", "letters_range", help="Letters in the gap: 6, or a range 5-7.")
@click.option("--pattern", help="Legible letters: נ??עין (? one letter, * any run, space between words).")
@click.option("--regex", help="A regular expression the filler must match.")
@click.option("--candidate", "candidates", multiple=True, help="Your own candidate (repeatable).")
@click.option(
    "--vocab-file",
    "vocab_files",
    multiple=True,
    help="Also try every word of this corpus that fits (single-word gaps; default: the first train_files entry).",
)
@click.option("--no-vocab", is_flag=True, help="Don't take candidates from the corpus.")
@click.option(
    "--max-vocab", type=int, default=50_000, show_default=True, help="At most this many corpus words (most frequent)."
)
@click.option(
    "--pmi",
    type=float,
    default=0.0,
    show_default=True,
    help="Subtract α·log p(filler alone): 0 = raw probability, 1 = mutual information; corrects the pull of common words.",
)
@click.option("--samples", type=int, default=64, show_default=True, help="Model continuations to collect words from.")
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top", type=int, default=20, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False), help="All ranked fillers as JSON.")
@click.option("--gap", default="___", show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@pass_ctx
def restore(
    obj: Ctx,
    model_path,
    text,
    words,
    letters_range,
    pattern,
    regex,
    candidates,
    vocab_files,
    no_vocab,
    max_vocab,
    pmi,
    samples,
    temperature,
    top,
    output,
    gap,
    batch_size,
    seed,
) -> None:
    """Propose and rank fillers for a lacuna, constrained by its size and legible letters.

    \b
    Examples:
      lfm3-cpt -c CONFIG restore --text "אין ___ מהן לענין כתובה" --letters 6
      lfm3-cpt -c CONFIG restore --text "אין ___ מהן לענין כתובה" --pattern "נ??עין"
      lfm3-cpt -c CONFIG restore --text "... ___ ..." --words 2 --letters 8-10 --no-vocab
    """
    import torch

    from .cloze import is_word
    from .evaluation import sentence_logprobs
    from .restore import Constraints
    from .restore import restore as run_restore
    from .textnorm import Normalizer

    cfg = obj.config()
    norm = Normalizer.from_config(cfg)
    lo = hi = None
    if letters_range:
        try:
            parts = [int(x) for x in letters_range.replace("–", "-").split("-")]
        except ValueError as exc:
            raise click.UsageError(f"--letters: give a number or a range like 5-7, not {letters_range!r}") from exc
        lo, hi = parts[0], parts[-1]
    try:
        constraints = Constraints(lo, hi, words or None, norm(pattern) if pattern else None, regex)
    except re.error as exc:
        raise click.UsageError(f"--regex: {exc}") from exc
    if text.count(gap) != 1:
        raise click.UsageError(f"the text must contain the gap marker {gap!r} exactly once")

    vocab = None
    if not no_vocab and (words in (0, 1)):
        specs = list(vocab_files) or list(cfg.train_files[:1])
        if specs:
            try:
                docs = [norm(d) for d in _read_specs(cfg, specs)]
            except (OSError, ValueError) as exc:
                raise click.ClickException(str(exc)) from exc
            vocab = Counter(w for d in docs for w in d.split() if is_word(w))

    target = model_path or default_lm_model(cfg)
    model, tokenizer = _load_lm(cfg, target)
    torch.manual_seed(seed)
    sample = _sampler(model, tokenizer, samples, max_new_tokens=6 * ((words or 3) + 1), temperature=temperature)
    with console.status("Proposing and ranking fillers ..."):
        result = run_restore(
            norm(text),
            constraints,
            lambda texts: sentence_logprobs(model, tokenizer, texts, batch_size),
            sample if samples > 0 else None,
            vocab,
            [norm(c) for c in candidates],
            max_vocab,
            gap,
            norm,
            pmi,
        )
    rows = result["rows"]
    c = result["counts"]
    console.print(
        f"🧩 {len(rows)} fillers fit ({c['model']} from {samples} model samples, {c['corpus']} from the corpus, "
        f"{c['given']} given) • scored with {target}" + (f" • pmi α={pmi:g}" if pmi else ""),
        highlight=False,
    )
    if not rows:
        raise click.ClickException("no candidate fits the constraints: loosen --letters/--pattern or add --samples")
    table = Table(title=f"{result['left'][-60:]} ___ {result['right'][:60]}", title_justify="left", header_style="bold")
    for col, justify in (
        ("#", "right"),
        ("filler", "left"),
        ("letters", "right"),
        ("log p", "right"),
        ("Δ", "right"),
        ("source", "left"),
    ):
        table.add_column(col, justify=justify)
    for r in rows[:top]:
        src = ", ".join(r["sources"]) + (f" ×{r['sampled']}" if r["sampled"] else "")
        table.add_row(str(r["rank"]), r["filler"], str(r["letters"]), f"{r['logprob']:.2f}", f"{r['delta']:.2f}", src)
    console.print(table, highlight=False)
    if output:
        Path(output).write_text(
            json.dumps({"text": text, "model": target, **result}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        console.print(f"📝 {output}")


def _free_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@cli.command()
@click.argument("model_path", required=False)
@click.option("--text", help="Text with the gap marker, e.g. 'arma virumque ___ Troiae'.")
@click.option("--candidate", "candidates", multiple=True, help="A possible filler (repeatable).")
@click.option("--candidates-file", type=click.Path(exists=True, dir_okay=False), help="One candidate per line.")
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    help='JSONL batch: {"id": ..., "text": "...___...", "candidates": [...]} per line.',
)
@click.option("--output", type=click.Path(dir_okay=False), help="Write ranked results as JSONL.")
@click.option("--gap", default="___", show_default=True, help="Gap marker in the text.")
@click.option(
    "--by",
    type=click.Choice(["total", "mean"]),
    default="total",
    show_default=True,
    help="Sort by total log-probability, or per token.",
)
@click.option("--top", type=int, default=20, show_default=True, help="Rows to print per text.")
@click.option("--batch-size", type=int, default=32, show_default=True)
@pass_ctx
def rank(obj: Ctx, model_path, text, candidates, candidates_file, input_file, output, gap, by, top, batch_size) -> None:
    """Rank candidate restorations of a gap (lacuna, conjecture, variant reading) by model probability.

    MODEL_PATH defaults to this run's CPT model (merged, else adapter), else the base model. Text and candidates
    get the same normalization as the training data.
    """
    from .device import detect, resolve_precision, torch_dtype
    from .evaluation import rank_candidates
    from .modeling import load_for_inference
    from .textnorm import Normalizer

    cfg = obj.config()
    norm = Normalizer.from_config(cfg)
    jobs: list[dict[str, Any]] = []
    if input_file:
        for n, line in enumerate(Path(input_file).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rec = json.loads(line)
                jobs.append({"id": rec.get("id", n), "text": rec["text"], "candidates": rec["candidates"]})
    else:
        cands = list(candidates)
        if candidates_file:
            cands += [c.strip() for c in Path(candidates_file).read_text(encoding="utf-8").splitlines() if c.strip()]
        if not text or not cands:
            raise click.UsageError("give --text and candidates (--candidate / --candidates-file), or --input")
        jobs.append({"id": 1, "text": text, "candidates": cands})

    info = detect()
    target = model_path or default_lm_model(cfg)
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    from .hub import ensure_model_files

    ensure_model_files(target, console)
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), None, device_map, cfg.trust_remote_code
        )
    console.print(f"🧮 scoring with {target}", highlight=False)
    out_fh = open(output, "w", encoding="utf-8") if output else None  # noqa: SIM115
    try:
        for job in jobs:
            try:
                rows = rank_candidates(
                    model, tokenizer, norm(job["text"]), [norm(c) for c in job["candidates"]], gap, batch_size, by
                )
            except ValueError as exc:
                raise click.ClickException(f"{job['id']}: {exc}") from exc
            table = Table(title=f"{job['id']}: {job['text']}", header_style="bold", title_justify="left")
            for col, justify in (
                ("#", "right"),
                ("candidate", "left"),
                ("log p", "right"),
                ("per token", "right"),
                ("tokens", "right"),
                ("Δ", "right"),
            ):
                table.add_column(col, justify=justify)
            for r in rows[:top]:
                table.add_row(
                    str(r["rank"]),
                    r["candidate"],
                    f"{r['logprob']:.2f}",
                    f"{r['mean_logprob']:.3f}",
                    str(r["tokens"]),
                    f"{r['delta']:.2f}",
                )
            console.print(table, highlight=False)
            if out_fh:
                out_fh.write(
                    json.dumps(
                        {"id": job["id"], "text": job["text"], "model": target, "by": by, "ranking": rows},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if out_fh:
            out_fh.close()
    if output:
        console.print(f"📝 {output}")


# ---------------------------------------------------------------------------------------------------------------
def default_chat_model(cfg: Config) -> str:
    run = Path(cfg.output_dir)
    for candidate in (run / "sft" / "merged", run / "sft" / "adapter", run / "merged", run / "adapter"):
        if (candidate / "config.json").exists() or (candidate / "adapter_config.json").exists():
            return str(candidate)
    return cfg.model_id


@cli.command()
@click.argument("model_path", required=False)
@click.option("--system-prompt", default="", help="Optional system message.")
@click.option("--thinking/--no-thinking", default=False, show_default=True, help="LFM3 reasoning mode.")
@click.option("--max-new-tokens", type=int, default=1024)
@click.option("--temperature", type=float, default=0.3, help="0 = greedy.")
@click.option("--repetition-penalty", type=float, default=1.1, help="1.1 is the LFM3 model card's recommendation.")
@click.option(
    "--no-repeat-ngram",
    type=int,
    default=8,
    show_default=True,
    help="Never repeat a sequence of this many tokens (stops loops); 0 = off.",
)
@pass_ctx
def chat(
    obj: Ctx, model_path, system_prompt, thinking, max_new_tokens, temperature, repetition_penalty, no_repeat_ngram
) -> None:
    """Interactive multi-turn chat. Default model: this run's SFT model, else its CPT model, else the base.

    Type /reset to clear the conversation, /exit (or Ctrl-D) to quit. Reasoning (<think> ... </think>) is hidden
    unless --thinking is given, and only the final answer is kept in the conversation.
    """
    import torch

    from .chatui import AnswerStreamer, final_answer
    from .device import detect, resolve_precision, torch_dtype
    from .modeling import load_for_inference

    cfg = obj.config()
    info = detect()
    target = model_path or default_chat_model(cfg)
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    from .hub import ensure_model_files

    ensure_model_files(target, console)
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), None, device_map, cfg.trust_remote_code
        )
    device = model.get_input_embeddings().weight.device
    console.print(f"💬 [bold]{target}[/] — /reset, /exit")
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}] if system_prompt else []
    streamer = AnswerStreamer(
        tokenizer,
        show_thinking=thinking,
        write=lambda t: click.echo(t, nl=False),
        on_thinking=lambda: click.echo(click.style("(thinking…) ", dim=True), nl=False),
    )
    while True:
        try:
            text = click.prompt(click.style("you", fg="cyan", bold=True), prompt_suffix="> ").strip()
        except (EOFError, click.Abort):
            break
        if text in {"/exit", "/quit"}:
            break
        if text == "/reset":
            history = history[:1] if system_prompt else []
            console.print("[dim]conversation cleared[/dim]")
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        inputs = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=thinking
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sampling = {"do_sample": temperature > 0, **({"temperature": temperature} if temperature > 0 else {})}
        if no_repeat_ngram > 0:
            sampling["no_repeat_ngram_size"] = no_repeat_ngram
        click.echo(click.style("model> ", fg="green", bold=True), nl=False)
        streamer.reset()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
                **sampling,
            )
        answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        history.append({"role": "assistant", "content": final_answer(answer)})


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[red]Interrupted[/red]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
