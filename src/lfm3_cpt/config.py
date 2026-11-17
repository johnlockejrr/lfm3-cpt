"""Run configuration: defaults < hardware profile < TOML file < CLI overrides."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]

TURBLIMP_PHENOMENA = [
    "anaphor_agreement",
    "argument_structure_ditransitive",
    "argument_structure_transitive",
    "binding",
    "determiners",
    "ellipsis",
    "irregular_forms",
    "island_effects",
    "nominalization",
    "npi_licensing",
    "passives",
    "quantifiers",
    "relative_clauses",
    "scrambling",
    "subject_agreement",
    "suspended_affixation",
]


@dataclass
class Config:
    # --- model -------------------------------------------------------------
    model_id: str = "LiquidAI/LFM3-1.7B"
    trust_remote_code: bool = True
    attn_implementation: str | None = None  # e.g. "sdpa"; None lets the model decide
    precision: str = "auto"  # auto | bf16 | fp16 | fp32

    # --- paths -------------------------------------------------------------
    output_dir: str = "runs/lfm3-1.7b-cpt-tr"
    data_dir: str = "data/tr"

    # --- data --------------------------------------------------------------
    wiki_dataset: str = "wikimedia/wikipedia"
    wiki_config: str = "20231101.tr"
    ood_dataset: str = "HuggingFaceFW/fineweb-2"
    ood_config: str = "tur_Latn"
    ood_split: str = "test"
    grammar_dataset: str = "juletxara/turblimp"
    train_articles: int = 3_000
    eval_articles: int = 100
    ood_pages: int = 100
    min_chars: int = 2_000  # skip stub articles (mostly lists and templates)
    shuffle_buffer: int = 10_000
    grammar_pairs_per_phenomenon: int = 100
    phenomena: list[str] = field(default_factory=lambda: list(TURBLIMP_PHENOMENA))
    seed: int = 42

    # --- LoRA --------------------------------------------------------------
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    lora_gate_proj: bool = False  # LFM3's per-head attention output gate
    train_embeddings: bool = False  # fully train the (tied) input embeddings via modules_to_save

    # --- training ----------------------------------------------------------
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 2
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup: float = 0.05  # < 1: fraction of total steps, >= 1: number of steps
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_length: int = 2048
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    # --- evaluation --------------------------------------------------------
    grammar_batch_size: int = 64
    ppl_batch_size: int = 2
    run_grammar: bool = True

    # ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def lora_targets(self) -> list[str]:
        targets = list(self.target_modules)
        if self.lora_gate_proj and "gate_proj" not in targets:
            targets.append("gate_proj")
        return targets

    @property
    def effective_batch_rows(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_rows * self.max_length

    def validate(self) -> None:
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(f"precision must be auto|bf16|fp16|fp32, got {self.precision!r}")
        for name in (
            "train_articles",
            "eval_articles",
            "max_length",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "lora_r",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.ood_pages < 0:
            raise ValueError("ood_pages must be >= 0")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        unknown = set(self.phenomena) - set(TURBLIMP_PHENOMENA)
        if unknown:
            raise ValueError(f"unknown TurBLiMP phenomena: {sorted(unknown)}")


# Hardware profiles. Only the keys that differ from the defaults are listed.
PROFILES: dict[str, dict[str, Any]] = {
    # The original notebook: free Colab T4 (16 GB, no bf16).
    "t4": {},
    # RTX 3090 24 GB (Ampere, bf16 + TF32). 4 x 2048 rows fit comfortably with checkpointing.
    "rtx3090": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "per_device_eval_batch_size": 4,
        "ppl_batch_size": 4,
        "dataloader_num_workers": 4,
    },
    # AMD Strix Halo (gfx1151) with 128 GB unified memory under ROCm. Memory is plentiful but compute is
    # modest, so trade memory for speed: bigger batches and no activation checkpointing. Pinned memory is
    # pointless on an APU that shares RAM with the GPU.
    "strix-halo": {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "per_device_eval_batch_size": 8,
        "ppl_batch_size": 8,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": False,
        "grammar_batch_size": 128,
    },
    # Tiny CPU smoke run: proves the pipeline works end to end, not for real training.
    "cpu-debug": {
        "train_articles": 16,
        "eval_articles": 4,
        "ood_pages": 4,
        "grammar_pairs_per_phenomenon": 2,
        "max_length": 256,
        "max_steps": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "per_device_eval_batch_size": 1,
        "ppl_batch_size": 1,
        "logging_steps": 1,
        "eval_steps": 1,
        "dataloader_num_workers": 0,
        "precision": "fp32",
    },
}

_FIELDS = {f.name: f for f in fields(Config)}


def _coerce(name: str, value: Any) -> Any:
    """Coerce a string (from `--set key=value`) or TOML value to the field's type."""
    if name not in _FIELDS:
        raise KeyError(f"unknown config key {name!r}")
    default = getattr(Config(), name)
    if not isinstance(value, str):
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value
    raw = value.strip()
    if isinstance(default, bool):
        low = raw.lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name}: cannot parse {raw!r} as bool")
    if isinstance(default, int):
        return int(float(raw)) if "e" in raw.lower() else int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if default is None and raw.lower() in {"none", "null", ""}:
        return None
    return raw


def load_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    flat: dict[str, Any] = {}
    for key, value in data.items():
        # Sections ([data], [lora], [training], ...) are purely organisational.
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def build_config(
    profile: str | None = None,
    toml_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    values: dict[str, Any] = {}
    file_values = load_toml(toml_path) if toml_path else {}
    profile = profile or file_values.pop("profile", None)
    file_values.pop("profile", None)
    if profile:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        values.update(PROFILES[profile])
    values.update(file_values)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    cfg = Config(**{k: _coerce(k, v) for k, v in values.items()})
    cfg.validate()
    return cfg


def parse_set_options(items: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Parse repeated `--set key=value` options."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        out[key] = _coerce(key, value)
    return out


def replace(cfg: Config, **changes: Any) -> Config:
    new = dataclasses.replace(cfg, **changes)
    new.validate()
    return new
