"""Download progress and library noise: one clean bar, no tqdm spam, no library-internal deprecations."""

import dataclasses
import io
import os
import time
import warnings
from types import SimpleNamespace

import pytest
from rich.console import Console
from trl import SFTConfig

from lfm3_cpt import hub
from lfm3_cpt.train import checkpointing_args

REPO = "SomeOrg/Some-Model"


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """A Hub with three files; snapshot_download writes them in chunks into the cache's blobs/ folder."""
    import huggingface_hub
    import huggingface_hub.constants as constants

    cache = tmp_path / "hub-cache"
    blobs = cache / "models--SomeOrg--Some-Model" / "blobs"
    files = {"model.safetensors": 300_000, "config.json": 2_000, "tokenizer.json": 50_000, "model.gguf": 10**9}
    calls = {}

    class FakeApi:
        def model_info(self, repo_id, revision=None, files_metadata=False):
            assert files_metadata
            return SimpleNamespace(
                sha="abc123", siblings=[SimpleNamespace(rfilename=f, size=s) for f, s in files.items()]
            )

    def fake_snapshot_download(repo_id, revision=None, ignore_patterns=None, **_):
        calls["ignore"] = ignore_patterns
        calls["bars_disabled"] = huggingface_hub.utils.are_progress_bars_disabled()
        blobs.mkdir(parents=True, exist_ok=True)
        for name, size in files.items():
            if any(__import__("fnmatch").fnmatch(name, p) for p in ignore_patterns):
                continue
            part = blobs / f"{name}.incomplete"
            for _ in range(3):
                with part.open("ab") as fh:
                    fh.write(b"x" * (size // 3))
                time.sleep(0.02)
            part.rename(blobs / name)
        calls["done"] = True
        return str(cache)

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, f, revision=None: str(blobs / f) if (blobs / f).exists() else None,
    )
    return SimpleNamespace(blobs=blobs, calls=calls, files=files)


def test_download_shows_one_clean_bar(fake_hub):
    out = io.StringIO()
    console = Console(file=out, width=120, force_terminal=False)
    assert hub.ensure_model_files(REPO, console, poll_seconds=0.01) == REPO
    assert fake_hub.calls["done"] and fake_hub.calls["bars_disabled"]  # hub's own bars off during our download
    assert "*.gguf" in fake_hub.calls["ignore"]
    assert not (fake_hub.blobs / "model.gguf").exists()  # other runtimes' files are skipped
    text = out.getvalue()
    assert "SomeOrg/Some-Model (3 files)" in text  # 3 needed files, gguf excluded
    assert "downloading bytes" not in text
    assert text.count("\n") <= 3  # one bar, not one line per refresh


def test_cached_model_is_not_downloaded_again(fake_hub):
    hub.ensure_model_files(REPO, Console(file=io.StringIO()), poll_seconds=0.01)
    fake_hub.calls.clear()
    out = io.StringIO()
    hub.ensure_model_files(REPO, Console(file=out), poll_seconds=0.01)
    assert "done" not in fake_hub.calls and out.getvalue() == ""


def test_download_error_is_raised(fake_hub, monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    with pytest.raises(OSError, match="disk full"):
        hub.ensure_model_files(REPO, Console(file=io.StringIO()), poll_seconds=0.01)


def test_passthrough_local_offline_and_unreachable(tmp_path, monkeypatch, fake_hub):
    import huggingface_hub
    import huggingface_hub.constants as constants

    assert hub.ensure_model_files(str(tmp_path)) == str(tmp_path)  # local directory
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(f'{{"base_model_name_or_path": "{tmp_path}"}}')
    assert hub.ensure_model_files(str(adapter)) == str(tmp_path)  # adapter -> its base

    class Unreachable:
        def model_info(self, *a, **k):
            raise ConnectionError("no network")

    monkeypatch.setattr(huggingface_hub, "HfApi", Unreachable)
    assert hub.ensure_model_files(REPO) == REPO  # transformers will explain the problem itself
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert hub.ensure_model_files(REPO) == REPO


def test_library_deprecations_filtered(monkeypatch):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monkeypatch.delenv("LFM3_CPT_SHOW_WARNINGS", raising=False)
        hub.quiet_libraries()
        warnings.warn(
            "`is_gradient_checkpointing` is deprecated and will be removed in v6.", FutureWarning, stacklevel=2
        )
        warnings.warn("something else entirely", FutureWarning, stacklevel=2)
    assert [str(w.message) for w in caught] == ["something else entirely"]


def test_checkpointing_args_use_this_versions_name():
    names = {f.name for f in dataclasses.fields(SFTConfig)}
    for enabled in (True, False):
        args = checkpointing_args(enabled)
        assert set(args) <= names
        expected = "activation_checkpointing" if "activation_checkpointing" in names else "gradient_checkpointing"
        assert args[expected] is enabled
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)  # no deprecation from our own arguments
            SFTConfig(output_dir="/tmp/x", report_to="none", use_cpu=True, bf16=False, **args)


def test_quiet_libraries_silences_resource_tracker(monkeypatch):
    from lfm3_cpt.hub import quiet_libraries

    monkeypatch.delenv("LFM3_CPT_SHOW_WARNINGS", raising=False)
    monkeypatch.setenv("PYTHONWARNINGS", "default")
    quiet_libraries()
    quiet_libraries()  # idempotent
    assert os.environ["PYTHONWARNINGS"] == "default,ignore:resource_tracker:UserWarning"

    monkeypatch.setenv("LFM3_CPT_SHOW_WARNINGS", "1")
    monkeypatch.delenv("PYTHONWARNINGS")
    quiet_libraries()
    assert "PYTHONWARNINGS" not in os.environ
