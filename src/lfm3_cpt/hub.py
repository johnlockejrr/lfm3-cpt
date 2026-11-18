"""Model downloads with one clean Rich progress bar, and quiet third-party progress bars.

Why: `huggingface_hub`, `transformers` and `datasets` draw their own tqdm bars. Inside a Rich spinner or live
display (which captures the terminal) every redraw of those bars becomes a new line, which turns a model
download into hundreds of "downloading bytes ####" lines. So:

- `quiet_libraries()` turns the libraries' bars off (the CLI calls it once at start-up);
- `ensure_model_files()` downloads a Hub model *before* any spinner starts, and shows its own single progress
  bar. Progress is measured from the size of the cache's `blobs/` folder while `snapshot_download` runs, which
  works the same for Xet and plain HTTP downloads and does not depend on huggingface_hub internals.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import threading
import warnings
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Files a causal-LM checkpoint never needs here (other runtimes and formats).
IGNORE_PATTERNS = [
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "onnx/*",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.tflite",
    "coreml/*",
    "*.mlmodel",
    "*.mlpackage/*",
    "openvino/*",
]


# Deprecation warnings raised *inside* the libraries (TRL/PEFT calling a transformers method that was renamed),
# which nothing in this project can act on. Set LFM3_CPT_SHOW_WARNINGS=1 to see them anyway.
LIBRARY_DEPRECATIONS = (
    r"`is_gradient_checkpointing` is deprecated",
    r"`gradient_checkpointing(_kwargs)?` is deprecated",
    r"`gradient_checkpointing_enable` is deprecated",
)


def quiet_libraries() -> None:
    """Turn off the tqdm bars of huggingface_hub, transformers and datasets (we show our own progress), and the
    library-internal deprecation warnings listed in LIBRARY_DEPRECATIONS."""
    if os.environ.get("LFM3_CPT_SHOW_WARNINGS", "") not in {"1", "true", "yes"}:
        for pattern in LIBRARY_DEPRECATIONS:
            warnings.filterwarnings("ignore", message=pattern, category=FutureWarning)
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover - env var HF_HUB_DISABLE_PROGRESS_BARS=0 has priority
        pass
    try:
        import transformers.utils.logging as tlog

        tlog.disable_progress_bar()
    except Exception:  # pragma: no cover
        pass
    try:
        import datasets

        datasets.disable_progress_bars()
    except Exception:  # pragma: no cover
        pass


def _ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in IGNORE_PATTERNS)


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for entry in os.scandir(path):
            with contextlib.suppress(FileNotFoundError):  # renamed from .incomplete while we scanned
                total += entry.stat(follow_symlinks=False).st_size
    return total


def base_of(path: str) -> str:
    """For an adapter directory, the base model it needs; anything else is returned unchanged."""
    cfg = Path(path) / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["base_model_name_or_path"]
    return path


def ensure_model_files(
    name_or_path: str,
    console: Console | None = None,
    revision: str | None = None,
    poll_seconds: float = 0.25,
) -> str:
    """Make sure a Hub model is in the local cache, downloading it with a single progress bar if needed.

    Local paths, offline mode and anything the Hub cannot answer are passed through untouched: transformers will
    then load (or report the error) as usual.
    """
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE, HF_HUB_OFFLINE
    from huggingface_hub.file_download import repo_folder_name

    name_or_path = base_of(name_or_path)
    if Path(name_or_path).expanduser().exists() or HF_HUB_OFFLINE:
        return name_or_path
    console = console or Console(stderr=True)
    try:
        info = HfApi().model_info(name_or_path, revision=revision, files_metadata=True)
    except Exception:
        return name_or_path  # offline, private without token, typo...: let transformers explain
    files = [s for s in (info.siblings or []) if not _ignored(s.rfilename)]
    missing = [
        s for s in files if not isinstance(try_to_load_from_cache(name_or_path, s.rfilename, revision=info.sha), str)
    ]
    if not missing:
        return name_or_path

    total = sum(s.size or 0 for s in missing)
    blobs = Path(HF_HUB_CACHE) / repo_folder_name(repo_id=name_or_path, repo_type="model") / "blobs"
    baseline = _dir_size(blobs)
    error: list[BaseException] = []

    def download() -> None:
        try:
            snapshot_download(name_or_path, revision=info.sha, ignore_patterns=IGNORE_PATTERNS)
        except BaseException as exc:  # re-raised in the main thread
            error.append(exc)

    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    worker = threading.Thread(target=download, daemon=True)
    columns: list[Any] = [
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ]
    try:
        with Progress(*columns, console=console, transient=False, expand=True) as progress:
            task = progress.add_task(
                f"⬇️  {name_or_path} ({len(missing)} file{'s' if len(missing) != 1 else ''})", total=total or None
            )
            worker.start()
            while worker.is_alive():
                worker.join(poll_seconds)
                done = max(0, _dir_size(blobs) - baseline)
                progress.update(task, completed=min(done, total) if total else done)
            if not error:
                progress.update(task, completed=total or 1, total=total or 1)
    finally:
        if not was_disabled:
            enable_progress_bars()
    if error:
        raise error[0]
    return name_or_path
