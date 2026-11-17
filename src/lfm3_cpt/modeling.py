"""Model / tokenizer loading, LoRA configuration and adapter merging."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .config import Config


def load_tokenizer(name_or_path: str, trust_remote_code: bool = True) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    name_or_path: str,
    dtype: torch.dtype,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": trust_remote_code}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)


def module_leaf_names(model: torch.nn.Module) -> set[str]:
    return {name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name}


def input_embedding_name(model: torch.nn.Module) -> str:
    emb = model.get_input_embeddings()
    for name, module in model.named_modules():
        if module is emb:
            return name.rsplit(".", 1)[-1]
    raise RuntimeError("could not locate the input embedding module")


def build_lora_config(cfg: Config, model: torch.nn.Module, warn: Callable[[str], None] = print) -> LoraConfig:
    """LoRA config for CPT, validated against the model's real module names.

    Targets that do not exist in the model are dropped with a warning instead of failing deep inside PEFT; if
    none exist we stop with a clear error.
    """
    available = module_leaf_names(model)
    targets = [t for t in cfg.lora_targets if t in available]
    missing = [t for t in cfg.lora_targets if t not in available]
    if missing:
        warn(f"[yellow]LoRA targets not found in the model and skipped: {', '.join(missing)}[/yellow]")
    if not targets:
        raise ValueError(f"none of the LoRA target modules {cfg.lora_targets} exist in {type(model).__name__}")

    kwargs: dict[str, Any] = {}
    if cfg.train_embeddings:
        # New-language CPT: train the full input embedding (tied to the LM head in LFM models).
        kwargs["modules_to_save"] = [input_embedding_name(model)]
        tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        if tied:
            kwargs["ensure_weight_tying"] = True
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
        **kwargs,
    )


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def adapter_base_model(adapter_dir: str | Path) -> str:
    return json.loads((Path(adapter_dir) / "adapter_config.json").read_text())["base_model_name_or_path"]


def load_for_inference(
    path: str,
    dtype: torch.dtype,
    adapter: str | None = None,
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
) -> tuple[torch.nn.Module, Any]:
    """Load a base/merged model, or a base model + LoRA adapter (`path` may itself be an adapter dir)."""
    if adapter is None and is_adapter_dir(path):
        adapter, path = path, adapter_base_model(path)
    tok_source = adapter if adapter and (Path(adapter) / "tokenizer_config.json").exists() else path
    tokenizer = load_tokenizer(tok_source, trust_remote_code)
    model = load_model(path, dtype, device_map, trust_remote_code)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def merge_adapter(
    adapter_dir: str | Path,
    output_dir: str | Path,
    base_model: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "cpu",
    trust_remote_code: bool = True,
) -> Path:
    """Merge a LoRA adapter into a *freshly loaded* base model and save a standalone checkpoint.

    The notebook merged into the model it trained, which on a T4 was loaded in fp16, so the merged checkpoint
    silently became fp16. Re-loading the base in its native bf16 (on CPU by default, so it never competes with
    a training run for GPU memory) gives an exact merge in the original dtype.
    """
    adapter_dir, output_dir = Path(adapter_dir), Path(output_dir)
    base = base_model or adapter_base_model(adapter_dir)
    model = load_model(base, dtype, device_map, trust_remote_code)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tok_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base
    load_tokenizer(str(tok_source), trust_remote_code).save_pretrained(str(output_dir))
    # Keep the run's provenance next to the weights.
    for extra in ("results.json", "config.toml"):
        src = adapter_dir.parent / extra
        if src.exists():
            shutil.copy(src, output_dir / f"cpt_{extra}")
    return output_dir
