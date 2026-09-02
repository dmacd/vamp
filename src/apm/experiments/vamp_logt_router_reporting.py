"""Machine-readable summaries, plots, and plain-language router reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from collections.abc import Mapping, Sequence
import base64
import csv
import io
import json
import math

import numpy as np

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_json,
)
from apm.experiments.vamp_logt_router_config import VampLogTRouterConfig
from apm.experiments.vamp_logt_router_state import ActiveAdapterBank


PLOT_FILES = (
    "01_selected_accuracy.png",
    "02_mean_routing_regret.png",
    "03_oracle_match_rate.png",
    "04_micro_vs_worst_range_regret.png",
    "05_replay_balance_tradeoff.png",
    "06_level_confusion.png",
    "07_best_second_margin_distribution.png",
    "08_router_work_scaling.png",
    "09_routing_vs_hierarchy_gap.png",
)


def write_phase_report(
    directory: Path,
    config: VampLogTRouterConfig,
    phase: str,
    seed: int,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, object],
    work: object,
    ledger_rows: Sequence[Mapping[str, object]],
    wall_seconds: float,
) -> dict[str, object]:
    """Write and validate one completed phase/seed result bundle."""
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    rows = tuple(_without_chain(row) for row in ledger_rows)
    evaluation = tuple(row for row in rows if row.get("row_type") == "evaluation")
    training = tuple(row for row in rows if row.get("row_type") == "training")
    accounting = tuple(row for row in rows if row.get("row_type") == "accounting")
    primary_scope = "full_test" if any(row.get("evaluation_scope") == "full_test" for row in evaluation) else "test_subset"
    final_step = bank.topology.processed_blocks
    final = tuple(
        row
        for row in evaluation
        if row.get("evaluation_scope") == primary_scope
        and row.get("group") == "micro"
        and int(row["macro_step"]) == final_step
    )
    final_by_condition = {str(row["condition"]): row for row in final}
    expected_conditions = {*conditions, "oracle", "most_recent_range", "largest_range", "uniform_active"}
    if set(final_by_condition) != expected_conditions:
        raise RuntimeError("completed seed lacks one or more final routing conditions")
    replay_rows = tuple(
        row
        for row in training
        if row["condition"] != "no_replay_hard" and int(row["macro_step"]) > 1
    )
    exact_replay_budget = all(
        int(row["historical_examples"])
        == (config.smoke if phase == "smoke" else config.primary).historical_budget
        for row in replay_rows
    )
    losses_decreased = float(
        np.mean(
            [
                float(row["mean_last_epoch_loss"])
                <= float(row["mean_first_epoch_loss"])
                for row in training
            ]
        )
    )
    summary: dict[str, object] = {
        "acceptance": {
            "all_primary_metrics_finite": all(
                math.isfinite(float(row["mean_regret"]))
                and math.isfinite(float(row["selected_accuracy"]))
                for row in evaluation
            ),
            "exact_historical_budget": exact_replay_budget,
            "inactive_levels_never_selected": all(
                sum(int(value) for value in row["selection_counts"] if value is not None)
                == int(row["example_count"])
                and all(
                    int(value) == 0
                    for level, value in enumerate(row["selection_counts"])
                    if level not in row["active_levels"]
                )
                for row in evaluation
            ),
            "nonnegative_routing_regret": all(
                float(row["mean_regret"]) >= -1.0e-7 for row in evaluation
            ),
            "router_loss_decrease_fraction": losses_decreased,
            "single_candidate_parity": _single_candidate_parity(evaluation),
        },
        "active_frontier": [
            {
                "first_macro_step": node.first_block + 1,
                "last_macro_step": node.last_block + 1,
                "level": node.level,
                "node_id": node.node_id,
            }
            for node in bank.topology.active_nodes
        ],
        "adapter_example_updates": bank.adapter_example_updates,
        "condition_final_metrics": {
            name: {
                "mean_regret": float(row["mean_regret"]),
                "oracle_match_rate": float(row["oracle_match_rate"]),
                "selected_accuracy": float(row["selected_accuracy"]),
                "selected_mean_cross_entropy": float(row["selected_mean_cross_entropy"]),
            }
            for name, row in sorted(final_by_condition.items())
        },
        "final_macro_step": final_step,
        "metric_rows": len(evaluation),
        "phase": phase,
        "run_seed": seed,
        "schema_version": "vamp-logt-router-seed-summary-v1",
        "wall_seconds": wall_seconds,
        "work": _jsonable_work(work),
    }
    _write_csv(directory / "seed_summary.csv", tuple(summary["condition_final_metrics"].items()))
    _write_plots(directory / "plots", rows, config)
    markdown = _seed_markdown(summary, config)
    atomic_write(directory / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        directory / "RESULTS.html",
        _html(markdown, directory, "LogT behavioral router").encode("utf-8"),
    )
    publish_immutable_json(summary_path, summary)
    required = ("metrics.jsonl", "seed_summary.csv", "RESULTS.md", "RESULTS.html")
    missing = tuple(name for name in required if not (directory / name).is_file())
    missing_plots = tuple(name for name in PLOT_FILES if not (directory / "plots" / name).is_file())
    if missing or missing_plots:
        raise RuntimeError(f"incomplete behavioral-router result: {missing + missing_plots}")
    return load_canonical_json(summary_path)


def write_results(
    run_root: Path,
    config: VampLogTRouterConfig,
    completed: Sequence[object],
) -> dict[str, object]:
    """Aggregate every completed seed and write the run-level scientific result."""
    del completed
    summary_paths = tuple(sorted(run_root.glob("*/seed-*/summary.json")))
    summaries = tuple(load_canonical_json(path) for path in summary_paths)
    primary = tuple(row for row in summaries if row["phase"] == "primary")
    all_rows = tuple(
        _without_chain(row)
        for path in sorted(run_root.glob("primary/seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    high_rows = tuple(
        row
        for row in all_rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "full_test"
        and row.get("group") == "micro"
        and int(row["macro_step"]) in {15, 31, 63}
    )
    condition_means = _condition_means(high_rows)
    replay_conditions = tuple(
        name
        for name in config.primary.conditions
        if name != "no_replay_hard" and name in condition_means
    )
    best_replay = (
        min(replay_conditions, key=lambda name: condition_means[name]["mean_regret"])
        if replay_conditions
        else None
    )
    closure = _oracle_gap_closure(condition_means, best_replay)
    complexity_passed = all(
        bool(row["acceptance"]["exact_historical_budget"])
        and bool(row["acceptance"]["nonnegative_routing_regret"])
        for row in primary
    )
    summary: dict[str, object] = {
        "completed_primary_seeds": len(primary),
        "condition_high_checkpoint_means": condition_means,
        "cross_entropy_gap_closure": closure,
        "criteria": {
            "1_replay_router_substantially_better": "descriptive_judgment",
            "2_closes_at_least_75_percent": closure is not None and closure >= 0.75,
            "3_retention_without_material_current_loss": "descriptive_judgment",
            "4_example_balanced_micro_hypothesis": _balance_hypothesis(
                condition_means, "example", "range", "mean_regret"
            ),
            "5_range_balanced_macro_or_worst_hypothesis": _range_hypothesis(all_rows),
            "6_fixed_budget_t_log_t_accounting": complexity_passed,
            "7_hierarchy_routing_decomposition_present": any(
                row.get("hierarchy_cross_entropy_gap") is not None for row in high_rows
            ),
        },
        "schema_version": "vamp-logt-router-aggregate-summary-v1",
        "selected_best_replay_condition": best_replay,
        "status": "complete" if len(primary) == len(config.primary.seeds) else "partial",
    }
    _write_aggregate_csv(run_root / "seed_summary.csv", summaries)
    if all_rows:
        _write_plots(run_root / "plots", all_rows, config)
    markdown = _aggregate_markdown(summary, config, summaries)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(markdown, run_root, "LogT behavioral router").encode("utf-8"),
    )
    summary_path = run_root / "summary.json"
    if summary_path.is_file():
        existing = load_canonical_json(summary_path)
        if existing != summary:
            atomic_write(summary_path, canonical_json_bytes(summary))
    else:
        publish_immutable_json(summary_path, summary)
    return summary


def _write_plots(
    plot_root: Path,
    rows: Sequence[Mapping[str, object]],
    config: VampLogTRouterConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_root.mkdir(parents=True, exist_ok=True)
    all_evaluation = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("group") == "micro"
    )
    evaluation = _primary_test_rows(all_evaluation)
    conditions = tuple(
        name
        for name in (*config.primary.conditions, "most_recent_range", "largest_range", "oracle")
        if any(row.get("condition") == name for row in evaluation)
    )
    _line_plot(
        plot_root / PLOT_FILES[0], evaluation, conditions, "selected_accuracy",
        "Selected-node accuracy", "Accuracy"
    )
    _line_plot(
        plot_root / PLOT_FILES[1], evaluation, conditions, "mean_regret",
        "Mean routing regret", "Cross-entropy regret (nats)"
    )
    _line_plot(
        plot_root / PLOT_FILES[2], evaluation, conditions, "oracle_match_rate",
        "Hard oracle-match rate", "Match rate"
    )
    _micro_worst_plot(plot_root / PLOT_FILES[3], all_evaluation, config)
    _tradeoff_plot(plot_root / PLOT_FILES[4], all_evaluation)
    _confusion_plot(plot_root / PLOT_FILES[5], evaluation, config)
    _margin_plot(plot_root / PLOT_FILES[6], evaluation)
    _work_plot(plot_root / PLOT_FILES[7], rows)
    _gap_plot(plot_root / PLOT_FILES[8], evaluation, config)
    plt.close("all")


def _line_plot(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for condition in conditions:
        points = _mean_points(rows, condition, field)
        if points:
            axis.plot(*zip(*points), marker="o", markersize=3, label=condition)
    axis.set(title=title, xlabel="Macro-step", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=170)


def _micro_worst_plot(path: Path, rows: Sequence[Mapping[str, object]], config: VampLogTRouterConfig) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for condition in config.primary.conditions:
        selected = tuple(
            row
            for row in rows
            if row.get("condition") == condition
            and row.get("evaluation_scope") == "evaluation_archive"
        )
        for field, style in (("mean_regret", "-"), ("worst_range_mean_regret", "--")):
            points = _mean_points(selected, condition, field)
            if points:
                axis.plot(*zip(*points), linestyle=style, label=f"{condition} {field.split('_')[0]}")
    axis.set(title="Micro-average versus worst-range regret", xlabel="Macro-step", ylabel="Regret (nats)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=170)


def _tradeoff_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    names = ("example_hard", "range_hard", "example_soft", "range_soft")
    values = [
        np.mean(
            [
                float(row["mean_regret"])
                for row in rows
                if row.get("condition") == name
                and row.get("evaluation_scope") == "evaluation_archive"
            ]
        )
        if any(
            row.get("condition") == name
            and row.get("evaluation_scope") == "evaluation_archive"
            for row in rows
        )
        else 0.0
        for name in names
    ]
    axis.bar(names, values, color=("#4C78A8", "#F58518", "#72B7B2", "#E45756"))
    axis.set(title="Example- versus range-balanced replay", ylabel="Mean routing regret (nats)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=170)


def _confusion_plot(path: Path, rows: Sequence[Mapping[str, object]], config: VampLogTRouterConfig) -> None:
    from matplotlib import pyplot as plt

    candidates = tuple(row for row in rows if row.get("condition") in config.primary.conditions)
    matrix = np.zeros((7, 7), dtype=np.int64)
    for row in candidates[-max(1, len(config.primary.conditions)):]:
        matrix += np.asarray(row["target_selection_confusion"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(title="Hard target versus selected LogT level", xlabel="Selected level", ylabel="Target level")
    figure.colorbar(image, ax=axis, label="Examples")
    figure.savefig(path, dpi=170)


def _margin_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib import pyplot as plt

    row = next((value for value in reversed(rows) if value.get("condition") == "oracle"), None)
    histogram = None if row is None else row["best_second_loss_margin_histogram"]
    counts = np.zeros(11) if histogram is None else np.asarray(histogram["counts"])
    edges = np.arange(12) if histogram is None else np.asarray(histogram["edges"])
    labels = [f"{edges[index]:.3g}–{edges[index + 1]:.3g}" for index in range(len(counts))]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(np.arange(len(counts)), counts)
    axis.set(title="Best-versus-second-best node loss margin", xlabel="Margin bin (nats)", ylabel="Examples")
    axis.set_xticks(np.arange(len(counts)), labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=170)


def _work_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    from matplotlib import pyplot as plt

    accounting = tuple(row for row in rows if row.get("row_type") == "accounting")
    steps = sorted(set(int(row["macro_step"]) for row in accounting))
    physical = [
        np.mean(
            [
                int(row["work"]["physical_current_node_evals"])
                + int(row["work"]["physical_historical_node_evals"])
                for row in accounting
                if int(row["macro_step"]) == step
            ]
        )
        for step in steps
    ]
    cumulative_nodes = np.cumsum(
        [
            np.mean(
                [
                    int(row["active_node_count"])
                    for row in accounting
                    if int(row["macro_step"]) == step
                ]
            )
            for step in steps
        ]
    )
    reference = np.asarray([step * math.log2(max(step, 2)) for step in steps])
    if len(reference) and reference[-1] > 0 and physical:
        reference *= physical[-1] / reference[-1]
        cumulative_nodes *= physical[-1] / cumulative_nodes[-1]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(steps, physical, label="measured supervision node-evals")
    axis.plot(
        steps,
        cumulative_nodes,
        linestyle=":",
        label="scaled cumulative sum of active nodes",
    )
    axis.plot(steps, reference, linestyle="--", label="scaled t log2(t)")
    axis.set(title="Fixed-budget router supervision work", xlabel="Macro-step", ylabel="Cumulative node/example evaluations")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=170)


def _gap_plot(path: Path, rows: Sequence[Mapping[str, object]], config: VampLogTRouterConfig) -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for condition in config.primary.conditions:
        selected = tuple(
            row
            for row in rows
            if row.get("condition") == condition
            and row.get("hierarchy_cross_entropy_gap") is not None
        )
        routing = _mean_points(selected, condition, "mean_regret")
        hierarchy = _mean_points(selected, condition, "hierarchy_cross_entropy_gap")
        if routing:
            axis.plot(*zip(*routing), label=f"{condition} routing")
        if hierarchy and condition == config.primary.conditions[0]:
            axis.plot(*zip(*hierarchy), color="black", linestyle="--", label="hierarchy gap")
    axis.set(title="Routing gap versus hierarchy gap", xlabel="Macro-step", ylabel="Cross-entropy gap (nats)")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, fontsize=8, ncol=2)
    else:
        axis.text(
            0.5,
            0.5,
            "Hierarchy decomposition is produced at primary full checkpoints.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.savefig(path, dpi=170)


def _mean_points(
    rows: Sequence[Mapping[str, object]], condition: str, field: str
) -> tuple[tuple[int, float], ...]:
    selected = tuple(
        row
        for row in rows
        if row.get("condition") == condition and row.get(field) is not None
    )
    return tuple(
        (
            step,
            float(np.mean([float(row[field]) for row in selected if int(row["macro_step"]) == step])),
        )
        for step in sorted(set(int(row["macro_step"]) for row in selected))
    )


def _condition_means(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    names = sorted(set(str(row["condition"]) for row in rows))
    fields = ("mean_regret", "selected_accuracy", "selected_mean_cross_entropy", "oracle_mean_cross_entropy")
    return {
        name: {
            field: float(np.mean([float(row[field]) for row in rows if row["condition"] == name]))
            for field in fields
        }
        for name in names
    }


def _oracle_gap_closure(
    means: Mapping[str, Mapping[str, float]], best: str | None
) -> float | None:
    if best is None or "most_recent_range" not in means:
        return None
    oracle = means[best]["oracle_mean_cross_entropy"]
    denominator = means["most_recent_range"]["selected_mean_cross_entropy"] - oracle
    return None if denominator <= 0.0 else 1.0 - (
        means[best]["selected_mean_cross_entropy"] - oracle
    ) / denominator


def _balance_hypothesis(
    means: Mapping[str, Mapping[str, float]], left: str, right: str, field: str
) -> bool | None:
    pairs = tuple(
        (means[f"{left}_{target}"][field], means[f"{right}_{target}"][field])
        for target in ("hard", "soft")
        if f"{left}_{target}" in means and f"{right}_{target}" in means
    )
    return None if not pairs else any(first <= second for first, second in pairs)


def _range_hypothesis(rows: Sequence[Mapping[str, object]]) -> bool | None:
    high = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "evaluation_archive"
        and row.get("group") == "micro"
        and int(row["macro_step"]) in {15, 31, 63}
    )
    comparisons = []
    for target in ("hard", "soft"):
        example = [row for row in high if row["condition"] == f"example_{target}"]
        ranged = [row for row in high if row["condition"] == f"range_{target}"]
        if example and ranged:
            comparisons.append(
                np.mean([float(row["worst_range_mean_regret"]) for row in ranged])
                <= np.mean([float(row["worst_range_mean_regret"]) for row in example])
            )
    return None if not comparisons else any(comparisons)


def _primary_test_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Prefer full-test rows at checkpoints and fixed subsets elsewhere."""
    full_coordinates = {
        (int(row["run_seed"]), int(row["macro_step"]), str(row["condition"]))
        for row in rows
        if row.get("evaluation_scope") == "full_test"
    }
    return tuple(
        row
        for row in rows
        if row.get("evaluation_scope") == "full_test"
        or (
            row.get("evaluation_scope") == "test_subset"
            and (
                int(row["run_seed"]),
                int(row["macro_step"]),
                str(row["condition"]),
            )
            not in full_coordinates
        )
    )


def _single_candidate_parity(rows: Sequence[Mapping[str, object]]) -> bool:
    applicable = tuple(row for row in rows if int(row["active_node_count"]) == 1)
    return all(float(row["mean_regret"]) <= 1.0e-7 for row in applicable)


def _jsonable_work(work: object) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(work)


def _write_csv(path: Path, rows: Sequence[tuple[str, object]]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("condition", "metrics"))
    writer.writeheader()
    for name, metrics in rows:
        writer.writerow({"condition": name, "metrics": json.dumps(metrics, sort_keys=True)})
    atomic_write(path, output.getvalue().encode("utf-8"))


def _write_aggregate_csv(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields = ("phase", "run_seed", "final_macro_step", "wall_seconds", "condition_final_metrics")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for summary in summaries:
        writer.writerow(
            {
                **{field: summary.get(field) for field in fields[:-1]},
                "condition_final_metrics": json.dumps(summary["condition_final_metrics"], sort_keys=True),
            }
        )
    atomic_write(path, output.getvalue().encode("utf-8"))


def _seed_markdown(summary: Mapping[str, object], config: VampLogTRouterConfig) -> str:
    metrics = summary["condition_final_metrics"]
    lines = "\n".join(
        f"| `{name}` | {float(row['mean_regret']):.5f} | {float(row['selected_accuracy']):.4f} | "
        f"{float(row['oracle_match_rate']):.4f} |"
        for name, row in metrics.items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    return f"""# Integrated LogT behavioral router: {summary['phase']} seed {summary['run_seed']}

## Outcome

This run completed {summary['final_macro_step']} Permuted-MNIST macro-steps. The router observed detached hidden states and output log probabilities from every active LogT adapter; it never changed node training or consolidation. The exact protocol identity is `{config.config_hash}`.

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

Labels enter only node training, oracle-target construction, and evaluation. Domain IDs, macro-step metadata, labels, and stored losses never enter the router feature vector. Generated checkpoints and ledgers remain local research artifacts.
"""


def _aggregate_markdown(
    summary: Mapping[str, object],
    config: VampLogTRouterConfig,
    seed_summaries: Sequence[Mapping[str, object]],
) -> str:
    means = summary["condition_high_checkpoint_means"]
    lines = "\n".join(
        f"| `{name}` | {float(row['mean_regret']):.5f} | {float(row['selected_accuracy']):.4f} | "
        f"{float(row['selected_mean_cross_entropy']):.5f} |"
        for name, row in means.items()
    ) or "| _No complete primary checkpoint rows_ | — | — | — |"
    criteria = "\n".join(
        f"| {name.replace('_', ' ')} | `{value}` |"
        for name, value in summary["criteria"].items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    return f"""# LogT-VAMP integrated behavioral router on Permuted-MNIST

## Outcome

The run status is `{summary['status']}` with {summary['completed_primary_seeds']} of {len(config.primary.seeds)} primary seeds complete. The best measured replay condition is `{summary['selected_best_replay_condition']}`. Criteria containing `descriptive_judgment` retain the qualitative wording preregistered by the user and are not converted into post-hoc numeric gates.

| Condition | Mean regret | Selected accuracy | Selected cross-entropy |
|---|---:|---:|---:|
{lines}

## Success criteria

| Criterion | Result |
|---|---|
{criteria}

## Replay and hierarchy interpretation

The cross-entropy gap-closure estimate is `{summary['cross_entropy_gap_closure']}`. Routing regret is always measured against the label-aware best extant node. At full checkpoints, the hierarchy gap separately compares that oracle with a fresh matched joint-IID top-two adapter.

## Figures

{figures if seed_summaries and means else 'Aggregate figures will appear after primary metric rows exist.'}

## Exact protocol

The resolved configuration hash is `{config.config_hash}`. It uses identity plus permutation seeds 1001–1007, the fixed block-order seed 20260827, independent run seeds 0–4, and fixed historical replay of 256 examples per eligible primary step.
"""


def _html(markdown: str, asset_root: Path, title: str) -> str:
    rendered = []
    section_open = False
    code_open = False
    table_lines: list[str] = []

    def flush_table() -> None:
        if table_lines:
            rendered.append(_markdown_table_html(table_lines))
            table_lines.clear()

    for line in markdown.splitlines():
        if not code_open and line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("```"):
            rendered.append("</code></pre>" if code_open else "<pre><code>")
            code_open = not code_open
            continue
        if code_open:
            rendered.append(escape(line))
            continue
        if line.startswith("![") and "](" in line and line.endswith(")"):
            label = line[2 : line.index("](")]
            relative = line[line.index("](") + 2 : -1]
            path = asset_root / relative
            if path.is_file():
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                rendered.append(
                    f'<figure><img alt="{escape(label)}" src="data:image/png;base64,{encoded}">'
                    f"<figcaption>{escape(label)}</figcaption></figure>"
                )
                continue
        if line.startswith("## "):
            if section_open:
                rendered.append("</details>")
            rendered.append(
                f"<details open><summary><h2>{escape(line[3:])}</h2></summary>"
            )
            section_open = True
        elif line.startswith("# "):
            rendered.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("### "):
            rendered.append(f"<h3>{escape(line[4:])}</h3>")
        elif line:
            rendered.append(f"<p>{escape(line)}</p>")
    flush_table()
    if code_open:
        rendered.append("</code></pre>")
    if section_open:
        rendered.append("</details>")
    paragraphs = "\n".join(rendered)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body{{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.5;color:#202124}}
details{{border-top:1px solid #dadce0;padding:.4rem 0 1rem}}
summary{{cursor:pointer;list-style-position:outside}}
summary h2{{display:inline-block;margin:.4rem 0}}
img{{max-width:100%;height:auto}}
pre,.table-row{{font-family:ui-monospace,monospace;white-space:pre;overflow:auto}}
.table-row{{margin:.1rem 0}}
.table-scroll{{margin:.75rem 0 1.25rem;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.95rem}}
th,td{{border:1px solid #dadce0;padding:.5rem .65rem;text-align:left;vertical-align:top}}
thead th{{background:#eef2f7;font-weight:650}}
tbody tr:nth-child(even){{background:#f8fafc}}
.align-right{{font-variant-numeric:tabular-nums;text-align:right}}
.align-center{{text-align:center}}
</style></head><body>{paragraphs}</body></html>"""


def _markdown_table_html(lines: Sequence[str]) -> str:
    """Render one well-formed pipe table, preserving a readable fallback."""
    rows = tuple(_markdown_table_cells(line) for line in lines)
    if len(rows) < 2 or not rows[0]:
        return _markdown_table_fallback(lines)
    alignments = tuple(_markdown_table_alignment(cell) for cell in rows[1])
    column_count = len(rows[0])
    if (
        len(alignments) != column_count
        or any(alignment is None for alignment in alignments)
        or any(len(row) != column_count for row in rows[2:])
    ):
        return _markdown_table_fallback(lines)
    resolved_alignments = tuple(str(alignment) for alignment in alignments)
    header = "".join(
        f'<th scope="col" class="align-{alignment}">{escape(cell)}</th>'
        for cell, alignment in zip(rows[0], resolved_alignments, strict=True)
    )
    body = "\n".join(
        "<tr>"
        + "".join(
            f'<td class="align-{alignment}">{escape(cell)}</td>'
            for cell, alignment in zip(row, resolved_alignments, strict=True)
        )
        + "</tr>"
        for row in rows[2:]
    )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _markdown_table_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return ()
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def _markdown_table_alignment(cell: str) -> str | None:
    marker = cell.strip()
    dashes = marker.strip(":")
    if len(dashes) < 3 or set(dashes) != {"-"}:
        return None
    if marker.startswith(":") and marker.endswith(":"):
        return "center"
    if marker.endswith(":"):
        return "right"
    return "left"


def _markdown_table_fallback(lines: Sequence[str]) -> str:
    return "\n".join(f'<p class="table-row">{escape(line)}</p>' for line in lines)


def _without_chain(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"format", "previous_sha256", "result_sha256", "sequence"}
    }


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


__all__ = ["PLOT_FILES", "write_phase_report", "write_results"]
