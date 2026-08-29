"""Rotated-MNIST report wording over the shared router measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json

from apm.continual.artifacts import atomic_write
from apm.experiments.vamp_logt_router_reporting import (
    PLOT_FILES,
    _html,
    write_phase_report as _write_shared_phase_report,
    write_results as _write_shared_results,
)
from apm.experiments.vamp_logt_router_rotated_config import (
    VampLogTRotatedRouterConfig,
)


def write_phase_report(
    directory: Path,
    config: VampLogTRotatedRouterConfig,
    phase: str,
    seed: int,
    bank: object,
    conditions: Mapping[str, object],
    work: object,
    ledger_rows: Sequence[Mapping[str, object]],
    wall_seconds: float,
) -> dict[str, object]:
    """Write shared measurements with task-correct Rotated-MNIST prose."""
    summary = _write_shared_phase_report(
        directory,
        config,
        phase,
        seed,
        bank,
        conditions,
        work,
        ledger_rows,
        wall_seconds,
    )
    markdown = _seed_markdown(summary, config)
    atomic_write(directory / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        directory / "RESULTS.html",
        _html(
            markdown,
            directory,
            "Rotated-MNIST LogT behavioral router",
        ).encode("utf-8"),
    )
    return summary


def write_results(
    run_root: Path,
    config: VampLogTRotatedRouterConfig,
    completed: Sequence[object],
) -> dict[str, object]:
    """Aggregate shared metrics and replace only task-specific report prose."""
    summary = _write_shared_results(run_root, config, completed)
    markdown = _aggregate_markdown(summary, config)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(
            markdown,
            run_root,
            "Rotated-MNIST LogT behavioral router",
        ).encode("utf-8"),
    )
    return summary


def _seed_markdown(
    summary: Mapping[str, object],
    config: VampLogTRotatedRouterConfig,
) -> str:
    metrics = summary["condition_final_metrics"]
    lines = "\n".join(
        f"| `{name}` | {float(row['mean_regret']):.5f} | "
        f"{float(row['selected_accuracy']):.4f} | "
        f"{float(row['oracle_match_rate']):.4f} |"
        for name, row in metrics.items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    outcome = (
        f"This run completed {summary['final_macro_step']} macro-steps from the "
        "blocked VAMP-AF task. The five contexts rotate images by 0, 18, 36, "
        "54, and 72 degrees and shift labels by 0, 2, 4, 6, and 8. The router "
        "observed detached hidden states and output log probabilities from each "
        "active LogT adapter; it never received context metadata or changed the "
        f"hierarchy. The exact protocol identity is `{config.config_hash}`."
    )
    boundary = (
        "This experiment reuses the VAMP-AF data contexts, not its PCA-median "
        "spatial tree. Labels and context IDs exist only behind training, "
        "target-construction, fixed diagnostic, and evaluation boundaries. "
        "Generated checkpoints and ledgers remain local research artifacts."
    )
    title = (
        "Integrated LogT behavioral router: Rotated-MNIST "
        f"{summary['phase']} seed {summary['run_seed']}"
    )
    return f"""# {title}

## Outcome

{outcome}

| Condition | Mean regret | Selected accuracy | Oracle match |
|---|---:|---:|---:|
{lines}

## Implementation checks

```json
{json.dumps(summary['acceptance'], indent=2, sort_keys=True)}
```

## Figures

{figures}

## Interpretation boundary

{boundary}
"""


def _aggregate_markdown(
    summary: Mapping[str, object],
    config: VampLogTRotatedRouterConfig,
) -> str:
    means = summary["condition_high_checkpoint_means"]
    lines = "\n".join(
        f"| `{name}` | {float(row['mean_regret']):.5f} | "
        f"{float(row['selected_accuracy']):.4f} | "
        f"{float(row['selected_mean_cross_entropy']):.5f} |"
        for name, row in means.items()
    ) or "| _No complete primary checkpoint rows_ | — | — | — |"
    criteria = "\n".join(
        f"| {name.replace('_', ' ')} | `{value}` |"
        for name, value in summary["criteria"].items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    schedule = ", ".join(str(value) for value in config.task.primary_context_steps)
    outcome = (
        f"The run status is `{summary['status']}` with "
        f"{summary['completed_primary_seeds']} of {len(config.primary.seeds)} "
        "primary seeds complete. The best measured learned replay condition is "
        f"`{summary['selected_best_replay_condition']}`. Criteria containing "
        "`descriptive_judgment` retain their preregistered qualitative wording; "
        "the roadmap records the evidence-backed judgment after the run."
    )
    interpretation = (
        "The cross-entropy gap-closure estimate is "
        f"`{summary['cross_entropy_gap_closure']}`. Routing regret is measured "
        "against the label-aware best extant node. Matched joint-IID adapters "
        "separately measure hierarchy competence. Example/range comparisons "
        "must be read by target family when their conclusions differ."
    )
    exact_protocol = (
        f"The resolved configuration hash is `{config.config_hash}`. It "
        "authenticates the completed VAMP-AF task, uses blocked context-step "
        f"counts [{schedule}], keeps seed-varying sample order and training "
        "randomness, and assigns 256 historical examples to every eligible "
        "primary replay update. This is a LogT-router experiment on VAMP-AF "
        "data, not a modification of the sealed AF tree."
    )
    return f"""# LogT-VAMP behavioral router on VAMP-AF Rotated-MNIST

## Outcome

{outcome}

| Condition | Mean regret | Selected accuracy | Selected cross-entropy |
|---|---:|---:|---:|
{lines}

## Success criteria

| Criterion | Result |
|---|---|
{criteria}

## Replay and hierarchy interpretation

{interpretation}

## Figures

{figures if means else 'Aggregate figures will appear after primary checkpoint rows exist.'}

## Exact protocol

{exact_protocol}
"""


__all__ = ["write_phase_report", "write_results"]
