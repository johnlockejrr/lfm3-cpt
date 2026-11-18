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
def default_lm_model(cfg: Config) -> str:
    run = Path(cfg.output_dir)
    for candidate in (run / "merged", run / "adapter"):
        if (candidate / "config.json").exists() or (candidate / "adapter_config.json").exists():
            return str(candidate)
    return cfg.model_id


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
@click.option("--repetition-penalty", type=float, default=1.05)
@pass_ctx
def chat(obj: Ctx, model_path, system_prompt, thinking, max_new_tokens, temperature, repetition_penalty) -> None:
    """Interactive multi-turn chat. Default model: this run's SFT model, else its CPT model, else the base.

    Type /reset to clear the conversation, /exit (or Ctrl-D) to quit.
    """
    import torch
    from transformers import TextStreamer

    from .device import detect, resolve_precision, torch_dtype
    from .modeling import load_for_inference

    cfg = obj.config()
    info = detect()
    target = model_path or default_chat_model(cfg)
    device_map = "auto" if info.backend in {"cuda", "rocm"} else None
    with console.status(f"Loading {target} ..."):
        model, tokenizer = load_for_inference(
            target, torch_dtype(resolve_precision(cfg.precision, info)), None, device_map, cfg.trust_remote_code
        )
    device = model.get_input_embeddings().weight.device
    console.print(f"💬 [bold]{target}[/] — /reset, /exit")
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}] if system_prompt else []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        try:
            text = click.prompt(click.style("sen", fg="cyan", bold=True), prompt_suffix="> ").strip()
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
        click.echo(click.style("model> ", fg="green", bold=True), nl=False)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
                **sampling,
            )
        answer = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    try:
        cli()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("[red]Interrupted[/red]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
