import pytest

from lfm3_cpt.config import build_config
from lfm3_cpt.device import DeviceInfo, resolve_precision, suggest_profile
from lfm3_cpt.modeling import build_lora_config, input_embedding_name


def _info(backend="cuda", mem=24.0, bf16=True):
    return DeviceInfo(backend, "gpu", 1, mem, bf16, backend == "cuda", "2.x", "13", None, None, "3.13")


@pytest.mark.parametrize(
    "info, expected",
    [
        (_info("cuda", 24), "rtx3090"),
        (_info("cuda", 16, bf16=False), "t4"),
        (_info("rocm", 96), "strix-halo"),
        (_info("rocm", 24), "rtx3090"),
        (_info("cpu", 0, bf16=False), "cpu-debug"),
    ],
)
def test_suggest_profile(info, expected):
    assert suggest_profile(info) == expected


def test_resolve_precision():
    assert resolve_precision("auto", _info("cuda")) == "bf16"
    assert resolve_precision("auto", _info("rocm")) == "bf16"
    assert resolve_precision("auto", _info("cuda", bf16=False)) == "fp16"  # T4, like the notebook
    assert resolve_precision("auto", _info("cpu", bf16=False)) == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("bf16", _info("cuda", bf16=False))
    with pytest.raises(ValueError):
        resolve_precision("fp16", _info("cpu", bf16=False))


def test_lora_targets_validated(model):
    warnings = []
    cfg = build_config(overrides={"lora_gate_proj": True})
    lora = build_lora_config(cfg, model, warn=warnings.append)
    # Llama has q/k/v/o and gate_proj (its MLP gate), but not LFM's in_proj/out_proj/w1/w2/w3
    assert set(lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"}
    assert "in_proj" in warnings[0] and "w1" in warnings[0]


def test_lora_no_targets(model):
    with pytest.raises(ValueError, match="none of the LoRA target"):
        build_lora_config(build_config(overrides={"target_modules": ["nope"]}), model, warn=lambda _: None)


def test_train_embeddings(model):
    assert input_embedding_name(model) == "embed_tokens"
    lora = build_lora_config(build_config(overrides={"train_embeddings": True}), model, warn=lambda _: None)
    assert lora.modules_to_save == ["embed_tokens"]
    assert lora.ensure_weight_tying is True
