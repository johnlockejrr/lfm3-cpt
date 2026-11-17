from click.testing import CliRunner

from lfm3_cpt.cli import cli


def test_help_lists_commands():
    res = CliRunner().invoke(cli, ["--help"])
    assert res.exit_code == 0
    for cmd in ("info", "show-config", "prepare", "train", "evaluate", "merge", "generate", "report"):
        assert cmd in res.output


def test_info():
    res = CliRunner().invoke(cli, ["info"])
    assert res.exit_code == 0, res.output
    assert "suggested profile" in res.output


def test_show_config_toml_with_overrides():
    res = CliRunner().invoke(cli, ["-p", "rtx3090", "-s", "learning_rate=1e-4", "show-config", "--toml"])
    assert res.exit_code == 0, res.output
    assert "per_device_train_batch_size = 4" in res.output
    assert "learning_rate = 0.0001" in res.output


def test_bad_set_is_usage_error():
    res = CliRunner().invoke(cli, ["-s", "not_a_key=1", "show-config"])
    assert res.exit_code == 2
    assert "unknown config key" in res.output


def test_config_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('profile = "strix-halo"\n')
    res = CliRunner().invoke(cli, ["-c", str(path), "show-config", "--toml"])
    assert res.exit_code == 0, res.output
    assert "gradient_checkpointing = false" in res.output
