"""Reports for the converged full-replay Rotated-MNIST integrator ceiling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
import os
from pathlib import Path

import numpy as np

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_json,
)
from apm.experiments.vamp_logt_integrator_ceiling_rotated_config import (
    VampLogTIntegratorCeilingConfig,
)
from apm.experiments.vamp_logt_router_reporting import _html, _load_jsonl, _without_chain
from apm.experiments.vamp_logt_router_state import ActiveAdapterBank


CEILING_CONDITION = "converged_full_replay_integrator"
REPORT_CONDITIONS = (CEILING_CONDITION, "mean_ensemble", "best_single_node")
CONDITION_LABELS = {
    CEILING_CONDITION: "converged full replay",
    "mean_ensemble": "mean ensemble",
    "best_single_node": "best node (label-aware oracle)",
}
CONDITION_STYLES = {
    CEILING_CONDITION: ("#0072B2", "-", "o", ""),
    "mean_ensemble": ("#E69F00", "--", "s", "//"),
    "best_single_node": ("#B2182B", "-.", "*", "xx"),
}
PLOT_FILES = (
    "01_test_accuracy.png",
    "02_test_cross_entropy.png",
    "03_convergence_epochs.png",
    "04_selected_fit_losses.png",
)


def write_phase_report(
    directory: Path,
    config: VampLogTIntegratorCeilingConfig,
    phase: str,
    seed: int,
    bank: ActiveAdapterBank,
    work: object,
    ledger_rows: Sequence[Mapping[str, object]],
    wall_seconds: float,
) -> dict[str, object]:
    """Write and validate one completed empirical-ceiling seed bundle."""
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    rows = tuple(_without_chain(row) for row in ledger_rows)
    training = tuple(row for row in rows if row.get("row_type") == "training")
    evaluation = tuple(row for row in rows if row.get("row_type") == "evaluation")
    accounting = tuple(row for row in rows if row.get("row_type") == "accounting")
    final_step = bank.topology.processed_blocks
    final_scope = (
        "full_test"
        if any(
            row.get("evaluation_scope") == "full_test"
            and int(row["macro_step"]) == final_step
            for row in evaluation
        )
        else "test_subset"
    )
    final_rows = {
        str(row["condition"]): row
        for row in evaluation
        if row.get("evaluation_scope") == final_scope
        and row.get("group") == "micro"
        and int(row["macro_step"]) == final_step
    }
    if set(final_rows) != set(REPORT_CONDITIONS):
        raise RuntimeError("completed ceiling seed lacks final comparison conditions")
    phase_config = config.smoke if phase == "smoke" else config.primary
    selected = tuple(row for row in training if bool(row["selected"]))
    selected_steps = tuple(sorted(int(row["macro_step"]) for row in selected))
    expected_steps = tuple(range(1, phase_config.macro_steps + 1))
    acceptance = {
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in (*training, *evaluation)
            for field in (
                ("best_validation_loss", "best_training_loss")
                if row.get("row_type") == "training"
                else ("mean_cross_entropy", "accuracy")
            )
        ),
        "complete_cumulative_archives": all(
            int(row["training_examples"])
            == int(row["macro_step"]) * config.benchmark.integrator_batch_size
            and int(row["validation_examples"])
            == int(row["macro_step"]) * config.benchmark.evaluation_batch_size
            for row in training
        ),
        "exact_all_example_presentations": all(
            int(row["training_example_presentations"])
            == int(row["epochs_ran"]) * int(row["training_examples"])
            and int(row["validation_example_presentations"])
            == (int(row["epochs_ran"]) + 2) * int(row["validation_examples"])
            for row in training
        ),
        "exact_feature_work": _exact_feature_work(accounting, config, phase_config.macro_steps),
        "fresh_independent_restarts": all(
            bool(row["fresh_initialization"]) for row in training
        )
        and len({int(row["fit_seed"]) for row in training}) == len(training),
        "parent_mean_ensemble_parity": (
            all(bool(row["parent_mean_ensemble_match"]) for row in evaluation)
            if phase == "primary"
            else True
        ),
        "selected_restart_converged": selected_steps == expected_steps
        and all(bool(row["converged"]) for row in selected),
        "test_and_validation_isolation": all(
            not bool(row["test_labels_used_for_selection"])
            and int(row["validation_updates"]) == 0
            for row in training
        )
        and all(not bool(row["selection_archive"]) for row in evaluation),
    }
    selected_epochs = tuple(int(row["epochs_ran"]) for row in selected)
    summary: dict[str, object] = {
        "acceptance": acceptance,
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
        "all_restart_convergence_fraction": float(
            np.mean([bool(row["converged"]) for row in training])
        ),
        "capped_restart_count": sum(
            str(row["stop_reason"]) == "maximum_epochs" for row in training
        ),
        "condition_final_metrics": {
            condition: {
                "accuracy": float(row["accuracy"]),
                "mean_cross_entropy": float(row["mean_cross_entropy"]),
            }
            for condition, row in sorted(final_rows.items())
        },
        "final_macro_step": final_step,
        "metric_rows": len(evaluation),
        "phase": phase,
        "run_seed": seed,
        "schema_version": "vamp-logt-integrator-ceiling-seed-summary-v1",
        "selected_epoch_maximum": max(selected_epochs),
        "selected_epoch_mean": float(np.mean(selected_epochs)),
        "wall_seconds": wall_seconds,
        "work": _jsonable_work(work),
    }
    _write_convergence_csv(directory / "convergence.csv", training)
    _write_plots(directory / "plots", rows, config)
    markdown = _seed_markdown(summary, config)
    atomic_write(directory / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        directory / "RESULTS.html",
        _html(markdown, directory, "Rotated-MNIST converged integrator ceiling").encode(
            "utf-8"
        ),
    )
    publish_immutable_json(summary_path, summary)
    required = ("metrics.jsonl", "convergence.csv", "RESULTS.md", "RESULTS.html")
    missing = tuple(name for name in required if not (directory / name).is_file())
    missing_plots = tuple(
        name for name in PLOT_FILES if not (directory / "plots" / name).is_file()
    )
    if missing or missing_plots:
        raise RuntimeError(f"incomplete ceiling result: {missing + missing_plots}")
    return load_canonical_json(summary_path)


def write_results(
    run_root: Path,
    config: VampLogTIntegratorCeilingConfig,
    parent_summary: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate completed seeds and quantify the parent-to-ceiling gap."""
    summary_paths = tuple(sorted(run_root.glob("*/seed-*/summary.json")))
    summaries = tuple(load_canonical_json(path) for path in summary_paths)
    primary = tuple(row for row in summaries if row["phase"] == "primary")
    primary_seeds = {int(row["run_seed"]) for row in primary}
    complete = primary_seeds == set(config.primary.seeds)
    rows = tuple(
        _without_chain(row)
        for path in sorted(run_root.glob("primary/seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    high = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "full_test"
        and row.get("group") == "micro"
        and int(row["macro_step"]) in {15, 31, 63}
    )
    ceiling_rows = tuple(row for row in high if row["condition"] == CEILING_CONDITION)
    ceiling_values = _mean_metrics(ceiling_rows) if ceiling_rows else None
    parent_means = parent_summary.get("condition_high_checkpoint_means", {})
    if not isinstance(parent_means, Mapping):
        raise ValueError("parent integrator summary lacks condition means")
    parent_best_name = str(parent_summary["selected_best_replay_condition"])
    parent_offline = _metric_pair(parent_means["offline_cumulative_integrator"])
    parent_online = _metric_pair(parent_means[parent_best_name])
    per_seed = {
        str(seed): _mean_metrics(
            tuple(
                row
                for row in high
                if row["condition"] == CEILING_CONDITION
                and int(row["run_seed"]) == seed
            )
        )
        for seed in sorted(primary_seeds)
        if any(
            row["condition"] == CEILING_CONDITION and int(row["run_seed"]) == seed
            for row in high
        )
    }
    selected_training = tuple(
        row for row in rows if row.get("row_type") == "training" and bool(row["selected"])
    )
    certified = complete and all(
        all(bool(value) for value in summary["acceptance"].values())
        for summary in primary
    )
    summary: dict[str, object] = {
        "ceiling_certified": certified,
        "ceiling_high_checkpoint_means": ceiling_values,
        "ceiling_minus_parent_offline": (
            {
                field: ceiling_values[field] - parent_offline[field]
                for field in ("accuracy", "mean_cross_entropy")
            }
            if ceiling_values is not None
            else None
        ),
        "ceiling_minus_parent_online": (
            {
                field: ceiling_values[field] - parent_online[field]
                for field in ("accuracy", "mean_cross_entropy")
            }
            if ceiling_values is not None
            else None
        ),
        "completed_primary_seeds": len(primary_seeds),
        "parent_best_online_condition": parent_best_name,
        "parent_best_online_high_checkpoint_means": parent_online,
        "parent_four_epoch_offline_high_checkpoint_means": parent_offline,
        "per_seed_ceiling_high_checkpoint_means": per_seed,
        "schema_version": "vamp-logt-integrator-ceiling-aggregate-summary-v1",
        "selected_epoch_maximum": (
            max(int(row["epochs_ran"]) for row in selected_training)
            if selected_training
            else None
        ),
        "selected_epoch_mean": (
            float(np.mean([int(row["epochs_ran"]) for row in selected_training]))
            if selected_training
            else None
        ),
        "status": "complete" if complete else "partial",
    }
    _write_aggregate_csv(run_root / "seed_summary.csv", per_seed)
    if rows:
        _write_plots(run_root / "plots", rows, config)
    if ceiling_values is not None:
        _write_high_comparison_plot(run_root / "plots", summary)
    markdown = _aggregate_markdown(summary, config)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(markdown, run_root, "Rotated-MNIST converged integrator ceiling").encode(
            "utf-8"
        ),
    )
    summary_path = run_root / "summary.json"
    if summary_path.is_file():
        if load_canonical_json(summary_path) != summary:
            atomic_write(summary_path, canonical_json_bytes(summary))
    else:
        publish_immutable_json(summary_path, summary)
    return summary


def _exact_feature_work(
    accounting: Sequence[Mapping[str, object]],
    config: VampLogTIntegratorCeilingConfig,
    macro_steps: int,
) -> bool:
    if not accounting:
        return False
    final = max(accounting, key=lambda row: int(row["macro_step"]))
    expected_training = sum(
        step.bit_count() * step * config.benchmark.integrator_batch_size
        for step in range(1, macro_steps + 1)
    )
    expected_validation = sum(
        step.bit_count() * step * config.benchmark.evaluation_batch_size
        for step in range(1, macro_steps + 1)
    )
    return (
        int(final["work"]["training_node_feature_evals"]) == expected_training
        and int(final["work"]["validation_node_feature_evals"])
        == expected_validation
    )


def _jsonable_work(work: object) -> dict[str, int]:
    names = (
        "training_node_feature_evals",
        "validation_node_feature_evals",
        "test_node_feature_evals",
        "training_example_presentations",
        "validation_example_presentations",
        "optimizer_steps",
        "restart_fits",
    )
    return {name: int(getattr(work, name)) for name in names}


def _mean_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        field: float(np.mean([float(row[field]) for row in rows]))
        for field in ("accuracy", "mean_cross_entropy")
    }


def _metric_pair(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("parent comparison metric is malformed")
    return {
        "accuracy": float(value["accuracy"]),
        "mean_cross_entropy": float(value["mean_cross_entropy"]),
    }


def _write_convergence_csv(
    path: Path, training: Sequence[Mapping[str, object]]
) -> None:
    fields = (
        "macro_step",
        "restart",
        "selected",
        "converged",
        "stop_reason",
        "best_epoch",
        "epochs_ran",
        "best_training_loss",
        "best_training_accuracy",
        "best_validation_loss",
        "best_validation_accuracy",
        "final_learning_rate",
        "training_examples",
        "validation_examples",
        "optimizer_steps",
        "training_example_presentations",
    )
    lines = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in training)
    lines.append(buffer.getvalue())
    atomic_write(path, "".join(lines).encode("utf-8"))


def _write_aggregate_csv(path: Path, rows: Mapping[str, Mapping[str, float]]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("seed", "accuracy", "mean_cross_entropy"))
    writer.writerows(
        (seed, values["accuracy"], values["mean_cross_entropy"])
        for seed, values in rows.items()
    )
    atomic_write(path, buffer.getvalue().encode("utf-8"))


def _write_plots(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    config: VampLogTIntegratorCeilingConfig,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vamp-logt-integrator-ceiling-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    evaluation = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
    )
    context_steps = (
        config.task.primary_context_steps
        if any(row.get("parent_mean_ensemble_match") is not None for row in evaluation)
        else config.task.smoke_context_steps
    )
    for filename, field, title, ylabel in (
        (PLOT_FILES[0], "accuracy", "Untouched test-subset accuracy", "Accuracy"),
        (
            PLOT_FILES[1],
            "mean_cross_entropy",
            "Untouched test-subset cross-entropy",
            "Cross-entropy",
        ),
    ):
        figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
        _add_context_regions(axis, context_steps)
        for condition in REPORT_CONDITIONS:
            points = _mean_points(evaluation, condition, field)
            if points:
                color, linestyle, marker, _hatch = CONDITION_STYLES[condition]
                axis.plot(
                    *zip(*points, strict=True),
                    color=color,
                    label=CONDITION_LABELS[condition],
                    linestyle=linestyle,
                    marker=marker,
                    markevery=max(1, len(points) // 12),
                    linewidth=2.0,
                )
        axis.set(title=title, xlabel="Macro-step", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.savefig(root / filename, dpi=170)
        plt.close(figure)

    training = tuple(row for row in rows if row.get("row_type") == "training")
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    _add_context_regions(axis, context_steps)
    for restart in sorted({int(row["restart"]) for row in training}):
        points = tuple(
            sorted(
                (
                    int(row["macro_step"]),
                    float(row["epochs_ran"]),
                )
                for row in training
                if int(row["restart"]) == restart
            )
        )
        axis.scatter(
            *zip(*points, strict=True),
            color=("#0072B2", "#E69F00", "#009E73")[restart % 3],
            label=f"restart {restart}",
            marker=("o", "s", "^")[restart % 3],
            s=22,
            alpha=0.75,
        )
    selected = tuple(row for row in training if bool(row["selected"]))
    if selected:
        points = tuple(
            sorted((int(row["macro_step"]), float(row["epochs_ran"])) for row in selected)
        )
        axis.plot(
            *zip(*points, strict=True),
            color="#111111",
            label="selected restart",
            linewidth=1.8,
        )
    axis.axhline(config.convergence.maximum_epochs, color="#B2182B", linestyle="--")
    axis.set(title="Convergence cost", xlabel="Macro-step", ylabel="Epochs run")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(root / PLOT_FILES[2], dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    _add_context_regions(axis, context_steps)
    for field, color, linestyle, marker, label in (
        ("best_training_loss", "#009E73", "--", "^", "training CE"),
        ("best_validation_loss", "#6F4EAA", "-", "D", "validation CE"),
    ):
        points = tuple(
            sorted((int(row["macro_step"]), float(row[field])) for row in selected)
        )
        if points:
            axis.plot(
                *zip(*points, strict=True),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(points) // 12),
                label=label,
                linewidth=2.0,
            )
    axis.set(title="Selected fit losses", xlabel="Macro-step", ylabel="Cross-entropy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(root / PLOT_FILES[3], dpi=170)
    plt.close(figure)


def _write_high_comparison_plot(root: Path, summary: Mapping[str, object]) -> None:
    from matplotlib import pyplot as plt

    names = (
        "converged full replay",
        "parent offline (4 epochs)",
        "parent online example replay",
    )
    records = (
        summary["ceiling_high_checkpoint_means"],
        summary["parent_four_epoch_offline_high_checkpoint_means"],
        summary["parent_best_online_high_checkpoint_means"],
    )
    colors = ("#0072B2", "#6F4EAA", "#E69F00")
    hatches = ("", "oo", "//")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for axis, field, title in (
        (axes[0], "mean_cross_entropy", "Cross-entropy (lower is better)"),
        (axes[1], "accuracy", "Accuracy (higher is better)"),
    ):
        values = [float(record[field]) for record in records]
        bars = axis.bar(names, values, color=colors, edgecolor="#111111")
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(root / "05_high_checkpoint_comparison.png", dpi=170)
    plt.close(figure)


def _mean_points(
    rows: Sequence[Mapping[str, object]], condition: str, field: str
) -> tuple[tuple[int, float], ...]:
    steps = sorted(
        {int(row["macro_step"]) for row in rows if row["condition"] == condition}
    )
    return tuple(
        (
            step,
            float(
                np.mean(
                    [
                        float(row[field])
                        for row in rows
                        if row["condition"] == condition
                        and int(row["macro_step"]) == step
                    ]
                )
            ),
        )
        for step in steps
    )


def _add_context_regions(axis: object, context_steps: tuple[int, ...]) -> None:
    starts = np.cumsum((0, *context_steps[:-1])) + 0.5
    colors = ("#E8F1FA", "#FFF0D5", "#E5F5EE", "#F1E9F7", "#FBE6E6")
    for context, (start, count, color) in enumerate(
        zip(starts, context_steps, colors, strict=True)
    ):
        axis.axvspan(start, start + count, color=color, alpha=0.45, zorder=0)
        axis.text(
            start + count / 2,
            1.01,
            f"C{context}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _seed_markdown(
    summary: Mapping[str, object], config: VampLogTIntegratorCeilingConfig
) -> str:
    metrics = summary["condition_final_metrics"]
    ceiling = metrics[CEILING_CONDITION]
    acceptance = summary["acceptance"]
    gate_rows = "".join(
        f"| {name.replace('_', ' ')} | {bool(value)} |\n"
        for name, value in acceptance.items()
    )
    metric_rows = "".join(
        f"| {CONDITION_LABELS[name]} | "
        f"{float(values['mean_cross_entropy']):.5f} | "
        f"{100.0 * float(values['accuracy']):.3f}% |\n"
        for name, values in metrics.items()
    )
    return f"""# Converged full-replay integrator ceiling — seed {summary['run_seed']}

This run rebuilt the frozen LogT hierarchy, trained a fresh integrator on every
cumulative integrator-training example at every macro-step, and used only the
disjoint cumulative evaluation allocation for stopping and restart selection.
The transformed test rows below never selected an epoch or restart.

At step {summary['final_macro_step']}, the selected integrator reached
{float(ceiling['mean_cross_entropy']):.5f}-nat cross-entropy and
{100.0 * float(ceiling['accuracy']):.3f}% accuracy. Selected fits ran
{float(summary['selected_epoch_mean']):.1f} epochs on average and at most
{int(summary['selected_epoch_maximum'])}; {int(summary['capped_restart_count'])}
restart(s) hit the {config.convergence.maximum_epochs}-epoch safety cap.

## Validation gates

| Gate | Passed |
|---|---:|
{gate_rows}

## Final untouched-test metrics

| Condition | Cross-entropy | Accuracy |
|---|---:|---:|
{metric_rows}

## Figures

![Test accuracy](plots/{PLOT_FILES[0]})

![Test cross-entropy](plots/{PLOT_FILES[1]})

![Convergence epochs](plots/{PLOT_FILES[2]})

![Selected training and validation losses](plots/{PLOT_FILES[3]})
"""


def _aggregate_markdown(
    summary: Mapping[str, object], config: VampLogTIntegratorCeilingConfig
) -> str:
    ceiling = summary["ceiling_high_checkpoint_means"]
    offline = summary["parent_four_epoch_offline_high_checkpoint_means"]
    online = summary["parent_best_online_high_checkpoint_means"]
    delta_offline = summary["ceiling_minus_parent_offline"]
    if ceiling is None or delta_offline is None:
        return f"""# Converged full-replay integrator ceiling on Rotated-MNIST

Status: **{summary['status']}**. The smoke bundle is available, but no complete
primary high-checkpoint result exists yet. The parent comparison remains
unchanged and primary can resume under protocol
`{config.protocol_revision}`.
"""
    return f"""# Converged full-replay integrator ceiling on Rotated-MNIST

Status: **{summary['status']}**. Certified empirical ceiling:
**{bool(summary['ceiling_certified'])}**.

Across seeds and full-test checkpoints 15, 31, and 63, fresh converged full
replay reached {float(ceiling['mean_cross_entropy']):.5f}-nat cross-entropy and
{100.0 * float(ceiling['accuracy']):.3f}% accuracy. The completed parent's
fresh four-epoch offline reference reached
{float(offline['mean_cross_entropy']):.5f} nats and
{100.0 * float(offline['accuracy']):.3f}%; the converged fit changed these by
{float(delta_offline['mean_cross_entropy']):+.5f} nats and
{100.0 * float(delta_offline['accuracy']):+.3f} accuracy points.

The best online parent condition, `{summary['parent_best_online_condition']}`,
reached {float(online['mean_cross_entropy']):.5f} nats and
{100.0 * float(online['accuracy']):.3f}% accuracy. This comparison measures the
cost of bounded sequential replay relative to the same fixed features and MLP;
it does not claim a mathematical upper bound over other models.

Selected fits ran {float(summary['selected_epoch_mean']):.1f} epochs on average
and at most {int(summary['selected_epoch_maximum'])} epochs. Test labels were
excluded from all fitting, stopping, learning-rate, and restart choices.

## High-checkpoint comparison

| Condition | Cross-entropy | Accuracy |
|---|---:|---:|
| converged full replay | {float(ceiling['mean_cross_entropy']):.5f} | {100.0 * float(ceiling['accuracy']):.3f}% |
| parent offline, 4 epochs | {float(offline['mean_cross_entropy']):.5f} | {100.0 * float(offline['accuracy']):.3f}% |
| parent online, example replay | {float(online['mean_cross_entropy']):.5f} | {100.0 * float(online['accuracy']):.3f}% |

![High-checkpoint comparison](plots/05_high_checkpoint_comparison.png)

![Test accuracy](plots/{PLOT_FILES[0]})

![Test cross-entropy](plots/{PLOT_FILES[1]})

![Convergence epochs](plots/{PLOT_FILES[2]})

![Selected fit losses](plots/{PLOT_FILES[3]})
"""


__all__ = ["PLOT_FILES", "write_phase_report", "write_results"]
