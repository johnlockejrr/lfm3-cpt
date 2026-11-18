"""Rich tables for results, and a tiny TOML writer for saving the resolved config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Config


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def config_to_toml(cfg: Config) -> str:
    lines = ["# Resolved lfm3-cpt configuration", ""]
    for key, value in cfg.to_dict().items():
        if value is None:
            lines.append(f"# {key} = (unset)")
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def config_table(cfg: Config) -> Table:
    table = Table(title="Configuration", show_header=True, header_style="bold")
    table.add_column("key", style="cyan")
    table.add_column("value")
    for key, value in cfg.to_dict().items():
        if isinstance(value, list) and len(value) > 6:
            value = f"{len(value)} items: {', '.join(map(str, value[:4]))}, …"
        table.add_row(key, str(value))
    return table


def _ppl(block: dict[str, Any] | None, split: str) -> float | None:
    if not block or split not in block or block[split] is None:
        return None
    return block[split]["perplexity"]


def _acc(block: dict[str, Any] | None) -> float | None:
    if not block or not block.get("grammar"):
        return None
    return block["grammar"]["accuracy"]


def _cell(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def _delta(before: float | None, after: float | None, spec: str, lower_is_better: bool) -> str:
    if before is None or after is None:
        return ""
    d = after - before
    good = d < 0 if lower_is_better else d > 0
    color = "green" if good else ("red" if d != 0 else "dim")
    return f"[{color}]{d:+{spec}}[/]"


def summary_table(baseline: dict[str, Any] | None, final: dict[str, Any] | None, title: str = "Results") -> Table:
    """The notebook's before/after summary, as a Rich table with deltas."""
    table = Table(title=title, header_style="bold")
    table.add_column("metric")
    table.add_column("original", justify="right")
    table.add_column("fine-tuned", justify="right")
    table.add_column("Δ", justify="right")
    for split, label in (("wiki", "perplexity, Wikipedia (held out)"), ("ood", "perplexity, web pages (OOD)")):
        b, a = _ppl(baseline, split), _ppl(final, split)
        if b is None and a is None:
            continue
        table.add_row(label, _cell(b, ".2f"), _cell(a, ".2f"), _delta(b, a, ".2f", True))
    b, a = _acc(baseline), _acc(final)
    if b is not None or a is not None:
        table.add_row(
            "[bold]TurBLiMP grammar accuracy[/]", _cell(b, ".1%"), _cell(a, ".1%"), _delta(b, a, ".1%", False)
        )
        bp = (baseline or {}).get("grammar", {}).get("by_phenomenon", {}) if baseline else {}
        ap = (final or {}).get("grammar", {}).get("by_phenomenon", {}) if final else {}
        for p in sorted(set(bp) | set(ap)):
            table.add_row(
                f"  {p.replace('_', ' ')}",
                _cell(bp.get(p), ".0%"),
                _cell(ap.get(p), ".0%"),
                _delta(bp.get(p), ap.get(p), ".0%", False),
            )
    return table


def print_flip_examples(console: Console, comparison: dict[str, Any]) -> None:
    console.print(
        f"\nGrammar pairs fixed by training: [green]{comparison['fixed']}[/], broken: [red]{comparison['broken']}[/]"
    )
    for ex in comparison["examples"]:
        console.print(f"\n🧩 [bold]{ex['phenomenon'].replace('_', ' ')}[/]: {ex['before']:.0%} → {ex['after']:.0%}")
        console.print(f"   ✅ grammatical:   {ex['sentence_good']}")
        console.print(f"   ❌ ungrammatical: {ex['sentence_bad']}")


def load_results(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text())


def _get(block: dict[str, Any] | None, *keys: str) -> float | None:
    for key in keys:
        if not isinstance(block, dict) or block.get(key) is None:
            return None
        block = block[key]
    return block  # type: ignore[return-value]


SFT_ROWS = (
    # label, path, format, lower_is_better (None: no colour)
    ("answer perplexity (held out)", ("answer", "perplexity"), ".2f", True),
    ("answers in Turkish", ("generation", "turkish_rate"), ".1%", False),
    ("answers in English", ("generation", "english_rate"), ".1%", True),
    ("answers finished (EOS)", ("generation", "finished_rate"), ".1%", False),
    ("chrF vs reference", ("generation", "chrf"), ".1f", False),
    ("avg answer length (words)", ("generation", "avg_words"), ".0f", None),
)

LM_ROWS = (
    ("perplexity, Wikipedia", ("lm", "wiki", "perplexity"), ".2f", True),
    ("perplexity, web pages (OOD)", ("lm", "ood", "perplexity"), ".2f", True),
    ("TurBLiMP grammar accuracy", ("lm", "grammar", "accuracy"), ".1%", False),
)


def sft_summary_table(before: dict[str, Any] | None, after: dict[str, Any] | None, title: str = "SFT results") -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column("metric")
    table.add_column("before SFT", justify="right")
    table.add_column("after SFT", justify="right")
    table.add_column("Δ", justify="right")
    for section, rows in (("chat", SFT_ROWS), ("language model (forgetting check)", LM_ROWS)):
        present = [r for r in rows if _get(before, *r[1]) is not None or _get(after, *r[1]) is not None]
        if not present:
            continue
        table.add_row(f"[dim]{section}[/]", "", "", "")
        for label, path, spec, lower in present:
            b, a = _get(before, *path), _get(after, *path)
            delta = "" if lower is None else _delta(b, a, spec, lower)
            table.add_row(f"  {label}", _cell(b, spec), _cell(a, spec), delta)
    return table
