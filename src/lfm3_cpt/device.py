"""Hardware detection for CUDA (NVIDIA) and ROCm (AMD) backends."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass

import torch


@dataclass
class DeviceInfo:
    backend: str  # cuda | rocm | mps | cpu
    name: str
    count: int
    total_memory_gb: float
    bf16: bool
    tf32: bool
    torch_version: str
    cuda_version: str | None
    hip_version: str | None
    arch: str | None
    python: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect() -> DeviceInfo:
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        backend = "rocm" if hip else "cuda"
        arch = getattr(props, "gcnArchName", None) if hip else f"sm_{props.major}{props.minor}"
        # TF32 is an NVIDIA Ampere+ feature.
        tf32 = backend == "cuda" and props.major >= 8
        return DeviceInfo(
            backend=backend,
            name=props.name,
            count=torch.cuda.device_count(),
            total_memory_gb=props.total_memory / 1024**3,
            bf16=torch.cuda.is_bf16_supported(including_emulation=False),
            tf32=tf32,
            torch_version=torch.__version__,
            cuda_version=None if hip else cuda,
            hip_version=hip,
            arch=arch,
            python=platform.python_version(),
        )
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        backend, name = "mps", "Apple MPS"
    else:
        backend, name = "cpu", platform.processor() or "CPU"
    return DeviceInfo(
        backend=backend,
        name=name,
        count=0,
        total_memory_gb=0.0,
        bf16=False,
        tf32=False,
        torch_version=torch.__version__,
        cuda_version=cuda,
        hip_version=hip,
        arch=None,
        python=platform.python_version(),
    )


def resolve_precision(preference: str, info: DeviceInfo) -> str:
    """Pick bf16 / fp16 / fp32. LFM3 ships in bf16; fp16 is the fallback for GPUs without bf16 (e.g. T4)."""
    if preference != "auto":
        if preference == "bf16" and info.backend in {"cuda", "rocm"} and not info.bf16:
            raise ValueError(f"{info.name} does not support bf16; use --set precision=fp16")
        if preference in {"bf16", "fp16"} and info.backend == "cpu":
            raise ValueError("mixed precision training needs a GPU; use precision=fp32 on CPU")
        return preference
    if info.backend in {"cuda", "rocm"}:
        return "bf16" if info.bf16 else "fp16"
    return "fp32"


def torch_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def suggest_profile(info: DeviceInfo) -> str:
    if info.backend == "rocm" and info.total_memory_gb >= 48:
        return "strix-halo"
    if info.backend in {"cuda", "rocm"} and info.total_memory_gb >= 20:
        return "rtx3090"
    if info.backend in {"cuda", "rocm"}:
        return "t4"
    return "cpu-debug"


def configure_backend(info: DeviceInfo) -> None:
    """Process-wide knobs that are safe on every backend."""
    if info.tf32:
        # LoRA adapters train in fp32; TF32 makes their matmuls much faster on Ampere+.
        torch.set_float32_matmul_precision("high")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def peak_memory_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return None


def current_memory_gb() -> tuple[float, float] | None:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return None


ROCM_ENV_VARS = ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "PYTORCH_HIP_ALLOC_CONF")
CUDA_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")


def relevant_env(info: DeviceInfo) -> dict[str, str]:
    names = ROCM_ENV_VARS if info.backend == "rocm" else CUDA_ENV_VARS
    return {k: os.environ[k] for k in names if k in os.environ}
