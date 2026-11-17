import pytest

from lfm3_cpt.config import PROFILES, Config, build_config, parse_set_options
from lfm3_cpt.reporting import config_to_toml


def test_defaults_match_notebook():
    cfg = Config()
    assert cfg.model_id == "LiquidAI/LFM3-1.7B"
    assert (cfg.lora_r, cfg.lora_alpha, cfg.learning_rate, cfg.max_length) == (64, 64, 2e-4, 2048)
    assert cfg.effective_batch_rows == 8
    assert cfg.tokens_per_step == 8 * 2048
    assert "gate_proj" not in cfg.lora_targets
    assert len(cfg.phenomena) == 16


def test_gate_proj_toggle():
    assert build_config(overrides={"lora_gate_proj": True}).lora_targets[-1] == "gate_proj"


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profiles_are_valid(name):
    cfg = build_config(name)
    for key, value in PROFILES[name].items():
        assert getattr(cfg, key) == value


def test_precedence_profile_file_override(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('profile = "rtx3090"\n[training]\nlearning_rate = 1e-4\nper_device_train_batch_size = 3\n')
    cfg = build_config(toml_path=path, overrides={"per_device_train_batch_size": 5})
    assert cfg.gradient_accumulation_steps == PROFILES["rtx3090"]["gradient_accumulation_steps"]  # from profile
    assert cfg.learning_rate == 1e-4  # from file
    assert cfg.per_device_train_batch_size == 5  # CLI wins


def test_set_options_coercion():
    parsed = parse_set_options(
        [
            "learning_rate=3e-4",
            "gradient_checkpointing=false",
            "target_modules=q_proj, v_proj",
            "max-steps=10",
            "attn_implementation=sdpa",
            "num_train_epochs=2",
        ]
    )
    assert parsed == {
        "learning_rate": 3e-4,
        "gradient_checkpointing": False,
        "target_modules": ["q_proj", "v_proj"],
        "max_steps": 10,
        "attn_implementation": "sdpa",
        "num_train_epochs": 2.0,
    }


@pytest.mark.parametrize("bad", [["nope=1"], ["learning_rate"], ["gradient_checkpointing=maybe"]])
def test_set_options_errors(bad):
    with pytest.raises((KeyError, ValueError)):
        parse_set_options(bad)


@pytest.mark.parametrize("override", [{"precision": "int8"}, {"train_articles": 0}, {"phenomena": ["made_up"]}])
def test_validation(override):
    with pytest.raises(ValueError):
        build_config(overrides=override)


def test_unknown_profile():
    with pytest.raises(ValueError):
        build_config("a100")


def test_toml_roundtrip(tmp_path):
    cfg = build_config("strix-halo", overrides={"learning_rate": 1.5e-4})
    path = tmp_path / "c.toml"
    path.write_text(config_to_toml(cfg))
    assert build_config(toml_path=path) == cfg
