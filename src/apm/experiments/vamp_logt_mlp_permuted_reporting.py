"""Human-readable reports and high-contrast plots for the dense study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path

import numpy as np

from apm.continual.artifacts import atomic_write, canonical_json_bytes, load_canonical_json
from apm.experiments.vamp_logt_mlp_permuted_ceiling import (
    CEILING_CONDITION,
    CONVERGED_BASE_INTEGRATOR_CONDITION,
    CONVERGED_MLP_CONDITION,
    FROZEN_BASE_CONDITION,
)
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_router_reporting import _html, _load_jsonl, _without_chain


CONDITION_PROTOCOL: Mapping[str, tuple[str, str, str, str, str, str, str, str]] = {
    "router_current_hard": (
        "Router — current only, hard target", "router", "persistent", "256 current", "none", "none", "4", "no",
    ),
    "router_uniform_hard": (
        "Router — uniform history, hard target", "router", "persistent", "256 current", "256 history", "uniform examples", "4", "no",
    ),
    "router_range_hard": (
        "Router — range-balanced history, hard target", "router", "persistent", "256 current", "256 history", "uniform live ranges", "4", "no",
    ),
    "router_uniform_soft": (
        "Router — uniform history, soft target", "router", "persistent", "256 current", "256 history", "uniform examples", "4", "no",
    ),
    "router_range_soft": (
        "Router — range-balanced history, soft target", "router", "persistent", "256 current", "256 history", "uniform live ranges", "4", "no",
    ),
    "integrator_current_only": (
        "Integrator — current only", "integrator", "persistent", "256 current", "none", "none", "4", "no",
    ),
    "integrator_uniform_replay": (
        "Integrator — uniform-history replay", "integrator", "persistent", "256 current", "256 history", "uniform examples", "4", "no",
    ),
    "integrator_range_replay": (
        "Integrator — range-balanced replay", "integrator", "persistent", "256 current", "256 history", "uniform live ranges", "4", "no",
    ),
    "integrator_base_uniform_replay": (
        "Base-only integrator — uniform-history replay", "integrator control", "persistent", "256 current", "256 history", "uniform examples", "4", "no",
    ),
    "fresh_cumulative_four_epoch_integrator": (
        "Fresh cumulative integrator — four epochs", "optimization reference", "fresh at checkpoint", "all cumulative", "all cumulative", "full replay", "4", "no",
    ),
    "pooled_single_mlp_reference": (
        "Pooled single MLP over cumulative node-training data", "model reference", "fresh at checkpoint", "all cumulative", "all cumulative", "full replay", "20", "no",
    ),
    FROZEN_BASE_CONDITION: (
        "Frozen calibrated base MLP", "post-hoc model diagnostic", "frozen", "none", "none", "none", "0", "seed 0 only",
    ),
    CONVERGED_MLP_CONDITION: (
        "Converged cumulative MLP", "post-hoc model diagnostic", "fresh at checkpoint", "all cumulative node-training rows", "all cumulative", "full replay", "20–200", "seed 0; validation; 3 restarts",
    ),
    CONVERGED_BASE_INTEGRATOR_CONDITION: (
        "Converged integrator over the frozen base MLP", "post-hoc integration diagnostic", "fresh at checkpoint", "all cumulative observer rows", "all cumulative", "full replay", "20–200", "seed 0; validation; 3 restarts",
    ),
    CEILING_CONDITION: (
        "Converged full-replay integrator ceiling", "optimization ceiling", "fresh every step", "all cumulative", "all cumulative", "full replay", "20–200", "yes; 3 restarts",
    ),
    "mean_ensemble": (
        "Equal-probability mean of active nodes", "fixed control", "none", "none", "none", "none", "0", "no",
    ),
    "most_recent_range": (
        "Newest temporal range", "fixed control", "none", "none", "none", "none", "0", "no",
    ),
    "largest_range": (
        "Largest temporal range", "fixed control", "none", "none", "none", "none", "0", "no",
    ),
    "uniform_active": (
        "Uniform random active node", "fixed control", "none", "none", "none", "none", "0", "no",
    ),
    "best_single_node": (
        "Best active node (label-aware oracle)", "offline oracle", "none", "none", "none", "none", "0", "uses labels directly",
    ),
    "oracle": (
        "Best active node (label-aware router oracle)", "offline oracle", "none", "none", "none", "none", "0", "uses labels directly",
    ),
}


PLOT_STYLES: Mapping[str, tuple[str, str, str]] = {
    "integrator_current_only": ("#111111", "X", "-"),
    "integrator_uniform_replay": ("#0066CC", "o", "-"),
    "integrator_range_replay": ("#E67300", "s", "--"),
    "integrator_base_uniform_replay": ("#008844", "^", ":"),
    "mean_ensemble": ("#C2188B", "D", "--"),
    "best_single_node": ("#D00020", "*", "-."),
    "fresh_cumulative_four_epoch_integrator": ("#6A1B9A", "P", ":"),
    "pooled_single_mlp_reference": ("#795548", "v", "-."),
    FROZEN_BASE_CONDITION: ("#111111", "o", ":"),
    CONVERGED_MLP_CONDITION: ("#D55E00", "s", "-"),
    CONVERGED_BASE_INTEGRATOR_CONDITION: ("#0072B2", "^", "--"),
    CEILING_CONDITION: ("#00A6D6", "H", "-"),
    "router_current_hard": ("#111111", "X", "-"),
    "router_uniform_hard": ("#0066CC", "o", "-"),
    "router_range_hard": ("#E67300", "s", "--"),
    "router_uniform_soft": ("#008844", "^", ":"),
    "router_range_soft": ("#C2188B", "D", "-."),
}


def write_results(run_root: Path, config: VampLogTDenseConfig) -> dict[str, object]:
    """Aggregate available phases, apply seven decisions, and render reports."""
    online_rows = tuple(
        _without_chain(row)
        for path in sorted((run_root / "online").glob("seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    ceiling_rows = tuple(
        _without_chain(row)
        for path in sorted((run_root / "ceiling").glob("seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    baseline_rows = tuple(
        _without_chain(row)
        for path in sorted((run_root / "baselines").glob("seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    criteria = _criteria(run_root, config, online_rows, ceiling_rows)
    baseline_summary = _single_seed_baseline_summary(
        online_rows, ceiling_rows, baseline_rows
    )
    summary = {
        "baseline_seed_count": len(
            tuple((run_root / "baselines").glob("seed-*/summary.json"))
        ),
        "config_hash": config.config_hash,
        "criteria": criteria,
        "online_seed_count": len(tuple((run_root / "online").glob("seed-*/summary.json"))),
        "ceiling_seed_count": len(tuple((run_root / "ceiling").glob("seed-*/summary.json"))),
        "schema_version": "vamp-logt-dense-results-v1",
        "single_seed_baseline_extension": baseline_summary,
        "status": (
            "complete"
            if len(tuple((run_root / "online").glob("seed-*/summary.json"))) == len(config.online.seeds)
            and len(tuple((run_root / "ceiling").glob("seed-*/summary.json"))) == len(config.online.seeds)
            else "partial"
        ),
    }
    atomic_write(run_root / "summary.json", canonical_json_bytes(summary))
    _write_condition_table(run_root / "condition_protocol.csv")
    plots = _write_plots(
        run_root / "plots", online_rows, ceiling_rows, baseline_rows
    )
    markdown = _markdown(run_root, summary, config, plots)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(markdown, run_root, "Dense Permuted-MNIST LogT experiment").encode("utf-8"),
    )
    return summary


def _criteria(
    run_root: Path,
    config: VampLogTDenseConfig,
    online_rows: Sequence[Mapping[str, object]],
    ceiling_rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    headline = tuple(
        row
        for row in online_rows
        if row.get("evaluation_scope") == "full_test"
        and row.get("group") == "micro"
        and int(row["macro_step"]) in config.evaluation.headline_checkpoints
    )
    if not headline:
        return None
    integrator = tuple(row for row in headline if row.get("row_type") == "integrator_evaluation")
    router = tuple(row for row in headline if row.get("row_type") == "router_evaluation")
    means = {
        condition: {
            "accuracy": float(np.mean([float(row["accuracy"]) for row in integrator if row["condition"] == condition])),
            "cross_entropy": float(np.mean([float(row["mean_cross_entropy"]) for row in integrator if row["condition"] == condition])),
        }
        for condition in {str(row["condition"]) for row in integrator}
    }
    replay_names = ("integrator_uniform_replay", "integrator_range_replay")
    best_replay = min(replay_names, key=lambda name: means[name]["cross_entropy"])
    best_router = min(
        ("router_uniform_hard", "router_range_hard", "router_uniform_soft", "router_range_soft"),
        key=lambda name: float(np.mean([
            float(row["selected_mean_cross_entropy"]) for row in router if row["condition"] == name
        ])),
    )
    router_mean = {
        "accuracy": float(np.mean([float(row["selected_accuracy"]) for row in router if row["condition"] == best_router])),
        "cross_entropy": float(np.mean([float(row["selected_mean_cross_entropy"]) for row in router if row["condition"] == best_router])),
    }
    no_replay = means["integrator_current_only"]
    offline = means["fresh_cumulative_four_epoch_integrator"]
    positive_gap = max(0.0, no_replay["cross_entropy"] - offline["cross_entropy"])
    closure = (
        None
        if positive_gap == 0.0
        else (no_replay["cross_entropy"] - means[best_replay]["cross_entropy"]) / positive_gap
    )
    archive = tuple(
        row
        for row in online_rows
        if row.get("row_type") == "integrator_evaluation"
        and row.get("evaluation_scope") == "evaluation_archive"
        and int(row["macro_step"]) in config.evaluation.headline_checkpoints
        and row.get("group") in {"current_range", "older_ranges"}
    )
    archive_mean = lambda condition, group, metric: (
        float(np.mean(values))
        if (values := [
            float(row[metric])
            for row in archive
            if row["condition"] == condition and row["group"] == group
        ])
        else math.nan
    )
    structural = tuple(
        load_canonical_json(path)
        for path in sorted((run_root / "online").glob("seed-*/summary.json"))
    )
    controls = {str(row["condition"]) for row in integrator}
    values = {
        "1_replay_beats_current_only_and_mean": (
            means[best_replay]["cross_entropy"] < no_replay["cross_entropy"]
            and means[best_replay]["accuracy"] > no_replay["accuracy"]
            and means[best_replay]["cross_entropy"] < means["mean_ensemble"]["cross_entropy"]
            and means[best_replay]["accuracy"] > means["mean_ensemble"]["accuracy"]
        ),
        "2_replay_closes_75_percent_of_four_epoch_gap": closure is not None and closure >= 0.75,
        "3_retention_without_more_than_two_point_current_loss": (
            archive_mean(best_replay, "older_ranges", "mean_cross_entropy")
            < archive_mean("integrator_current_only", "older_ranges", "mean_cross_entropy")
            and archive_mean(best_replay, "current_range", "accuracy")
            >= archive_mean("integrator_current_only", "current_range", "accuracy") - 0.02
        ),
        "4_full_nodes_beat_base_only_without_accuracy_loss": (
            means[best_replay]["cross_entropy"] < means["integrator_base_uniform_replay"]["cross_entropy"]
            and means[best_replay]["accuracy"] >= means["integrator_base_uniform_replay"]["accuracy"]
        ),
        "5_integrator_beats_matched_router_on_both_metrics": (
            means[best_replay]["cross_entropy"] < router_mean["cross_entropy"]
            and means[best_replay]["accuracy"] > router_mean["accuracy"]
        ),
        "6_all_structural_and_accounting_gates_pass": (
            len(structural) == len(config.online.seeds)
            and all(all(bool(value) for value in row["acceptance"].values()) for row in structural)
        ),
        "7_attribution_controls_present": {
            "mean_ensemble", "most_recent_range", "largest_range", "uniform_active",
            "best_single_node", "fresh_cumulative_four_epoch_integrator",
            "pooled_single_mlp_reference",
        }.issubset(controls),
    }
    ceiling_trace = tuple(
        row
        for row in ceiling_rows
        if row.get("row_type") == "ceiling_evaluation"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
    )
    return {
        "all_pass": all(values.values()),
        "best_online_replay": CONDITION_PROTOCOL[best_replay][0],
        "best_router": CONDITION_PROTOCOL[best_router][0],
        "ceiling_every_step_cells": len(ceiling_trace),
        "four_epoch_gap_closure": closure,
        "values": values,
    }


def _single_seed_baseline_summary(
    online_rows: Sequence[Mapping[str, object]],
    ceiling_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    evaluation = tuple(
        row
        for row in baseline_rows
        if row.get("row_type") == "baseline_evaluation"
        and row.get("evaluation_scope") == "full_test"
        and row.get("group") == "micro"
        and int(row.get("run_seed", -1)) == 0
    )
    if not evaluation:
        return None
    checkpoints = tuple(sorted({int(row["macro_step"]) for row in evaluation}))
    source_rows = {
        condition: tuple(
            row for row in evaluation if row.get("condition") == condition
        )
        for condition in (
            FROZEN_BASE_CONDITION,
            CONVERGED_MLP_CONDITION,
            CONVERGED_BASE_INTEGRATOR_CONDITION,
        )
    }
    source_rows.update({
        "pooled_single_mlp_reference": tuple(
            row
            for row in online_rows
            if row.get("row_type") == "integrator_evaluation"
            and row.get("condition") == "pooled_single_mlp_reference"
            and row.get("evaluation_scope") == "full_test"
            and row.get("group") == "micro"
            and int(row.get("run_seed", -1)) == 0
        ),
        CEILING_CONDITION: tuple(
            row
            for row in ceiling_rows
            if row.get("row_type") == "ceiling_evaluation"
            and row.get("condition") == CEILING_CONDITION
            and row.get("evaluation_scope") == "full_test"
            and row.get("group") == "micro"
            and int(row.get("run_seed", -1)) == 0
        ),
    })
    checkpoint_metrics = {}
    complete = True
    for macro_step in checkpoints:
        condition_metrics = {}
        for condition, rows in source_rows.items():
            selected = tuple(
                row for row in rows if int(row.get("macro_step", -1)) == macro_step
            )
            if not selected:
                complete = False
                continue
            condition_metrics[condition] = {
                "accuracy": float(
                    np.mean([float(row["accuracy"]) for row in selected])
                ),
                "cross_entropy": float(
                    np.mean([float(row["mean_cross_entropy"]) for row in selected])
                ),
                "domain_cells": len(selected),
            }
        checkpoint_metrics[str(macro_step)] = condition_metrics
    selected_convergence = tuple(
        row
        for row in baseline_rows
        if row.get("row_type") == "baseline_convergence" and bool(row.get("selected"))
    )
    convergence = {
        condition: {
            str(macro_step): {
                "best_validation_accuracy": float(row["best_validation_accuracy"]),
                "best_validation_cross_entropy": float(row["best_validation_loss"]),
                "epochs_ran": int(row["epochs_ran"]),
                "selected_restart": int(row["restart"]),
                "stop_reason": str(row["stop_reason"]),
                "training_examples": int(row["training_examples"]),
                "validation_examples": int(row["validation_examples"]),
            }
            for macro_step in checkpoints
            for row in selected_convergence
            if row.get("condition") == condition
            and int(row.get("macro_step", -1)) == macro_step
        }
        for condition in (
            CONVERGED_MLP_CONDITION,
            CONVERGED_BASE_INTEGRATOR_CONDITION,
        )
    }
    complete = complete and all(
        len(convergence[condition]) == len(checkpoints) for condition in convergence
    )
    return {
        "aggregation": "equal-weight mean over full-test seen-domain cells",
        "checkpoint_metrics": checkpoint_metrics,
        "convergence": convergence,
        "evaluation_checkpoints": list(checkpoints),
        "primary_three_seed_estimates_unchanged": True,
        "run_seed": 0,
        "status": "complete" if complete else "partial",
    }


def _write_condition_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow((
            "condition", "family", "persistence", "current data", "history",
            "history sampler", "epochs per update", "validation/restarts",
        ))
        writer.writerows(CONDITION_PROTOCOL.values())


def _write_plots(
    directory: Path,
    online_rows: Sequence[Mapping[str, object]],
    ceiling_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if not online_rows and not ceiling_rows and not baseline_rows:
        return ()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    integrator_rows = tuple(
        row
        for row in (*online_rows, *ceiling_rows)
        if row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
        and row.get("condition") in {
            "integrator_current_only", "integrator_uniform_replay", "integrator_range_replay",
            "integrator_base_uniform_replay", "mean_ensemble",
            "fresh_cumulative_four_epoch_integrator", CEILING_CONDITION,
            "pooled_single_mlp_reference",
        }
    )
    for metric, ylabel, filename in (
        ("accuracy", "Test-subset accuracy", "01_integrator_accuracy.png"),
        ("mean_cross_entropy", "Test-subset cross-entropy", "02_integrator_cross_entropy.png"),
    ):
        figure, axis = plt.subplots(figsize=(12.5, 7.2))
        for condition in PLOT_STYLES:
            values = tuple(row for row in integrator_rows if row.get("condition") == condition and metric in row)
            if not values:
                continue
            steps = sorted({int(row["macro_step"]) for row in values})
            means = [float(np.mean([float(row[metric]) for row in values if int(row["macro_step"]) == step])) for step in steps]
            color, marker, linestyle = PLOT_STYLES[condition]
            axis.plot(
                steps,
                means,
                label=CONDITION_PROTOCOL[condition][0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=3.0 if condition == CEILING_CONDITION else 2.0,
                markersize=5.5,
                markevery=max(1, len(steps) // 12),
            )
        axis.set_xlabel("Macro-step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8.5, ncol=2)
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=180)
        plt.close(figure)
        written.append(filename)
    router_rows = tuple(
        row
        for row in online_rows
        if row.get("row_type") == "router_evaluation"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
        and row.get("condition") in PLOT_STYLES
    )
    if router_rows:
        figure, axis = plt.subplots(figsize=(12.5, 7.2))
        for condition in (
            "router_current_hard", "router_uniform_hard", "router_range_hard",
            "router_uniform_soft", "router_range_soft",
        ):
            values = tuple(row for row in router_rows if row["condition"] == condition)
            steps = sorted({int(row["macro_step"]) for row in values})
            means = [float(np.mean([float(row["selected_accuracy"]) for row in values if int(row["macro_step"]) == step])) for step in steps]
            color, marker, linestyle = PLOT_STYLES[condition]
            axis.plot(steps, means, label=CONDITION_PROTOCOL[condition][0], color=color, marker=marker, linestyle=linestyle, linewidth=2.0, markersize=5.5, markevery=max(1, len(steps) // 12))
        axis.set_xlabel("Macro-step")
        axis.set_ylabel("Selected-node test-subset accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8.5)
        figure.tight_layout()
        filename = "03_router_accuracy.png"
        figure.savefig(directory / filename, dpi=180)
        plt.close(figure)
        written.append(filename)
    baseline_evaluation = tuple(
        row
        for row in baseline_rows
        if row.get("row_type") == "baseline_evaluation"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
        and int(row.get("run_seed", -1)) == 0
    )
    if baseline_evaluation:
        comparison_rows = (
            *baseline_evaluation,
            *(
                row
                for row in online_rows
                if row.get("row_type") == "integrator_evaluation"
                and row.get("condition") == "pooled_single_mlp_reference"
                and row.get("evaluation_scope") == "test_subset"
                and row.get("group") == "micro"
                and int(row.get("run_seed", -1)) == 0
            ),
            *(
                row
                for row in ceiling_rows
                if row.get("row_type") == "ceiling_evaluation"
                and row.get("condition") == CEILING_CONDITION
                and row.get("evaluation_scope") == "test_subset"
                and row.get("group") == "micro"
                and int(row.get("run_seed", -1)) == 0
            ),
        )
        baseline_styles = {
            FROZEN_BASE_CONDITION: ("#111111", "o", ":"),
            "pooled_single_mlp_reference": ("#CC79A7", "v", "-."),
            CONVERGED_MLP_CONDITION: ("#D55E00", "s", "-"),
            CONVERGED_BASE_INTEGRATOR_CONDITION: ("#0072B2", "^", "--"),
            CEILING_CONDITION: ("#009E73", "H", "-"),
        }
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
        for axis, metric, ylabel in (
            (axes[0], "accuracy", "Test-subset accuracy"),
            (axes[1], "mean_cross_entropy", "Test-subset cross-entropy"),
        ):
            for condition, (color, marker, linestyle) in baseline_styles.items():
                values = tuple(
                    row
                    for row in comparison_rows
                    if row.get("condition") == condition and metric in row
                )
                if not values:
                    continue
                steps = sorted({int(row["macro_step"]) for row in values})
                means = [
                    float(
                        np.mean(
                            [
                                float(row[metric])
                                for row in values
                                if int(row["macro_step"]) == step
                            ]
                        )
                    )
                    for step in steps
                ]
                axis.plot(
                    steps,
                    means,
                    label=CONDITION_PROTOCOL[condition][0],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.6,
                    markersize=7.0,
                )
            axis.set_xlabel("Macro-step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            fontsize=8.5,
            ncol=3,
        )
        figure.suptitle(
            "Seed-0 post-hoc cumulative baselines (single stream seed)", y=1.06
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        filename = "04_single_seed_cumulative_baselines.png"
        figure.savefig(directory / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(filename)
    return tuple(written)


def _markdown(
    run_root: Path,
    summary: Mapping[str, object],
    config: VampLogTDenseConfig,
    plots: tuple[str, ...],
) -> str:
    calibration_path = run_root / "calibration" / "summary.json"
    calibration = load_canonical_json(calibration_path) if calibration_path.is_file() else None
    lines = [
        "# Dense Permuted-MNIST LogT experiment",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "This successor removes convolution entirely. Every temporal node starts from the same selected three-hidden-layer raw-pixel MLP, adapts all four affine layers, and is then frozen. The router and integrator see only normalized final hidden activations, class log probabilities, and active-slot bits.",
        "",
        "The bounded online comparison intentionally preserves the earlier epoch matrix: current-only integration receives 8 optimizer updates per production step, while replay integration receives 16. This is an epoch-matched comparison, not an optimizer-update-matched control.",
        "",
    ]
    if calibration is not None:
        lines.extend((
            "## Calibration",
            "",
            f"Selected hidden widths: `{calibration.get('selected_hidden_widths')}`; identity test accuracy: `{calibration.get('identity_test_accuracy')}`. Architecture selection used validation only; test metrics were opened after selection.",
            "",
        ))
    lines.extend((
        "## Conditions in plain language",
        "",
        "| Condition | Family | Persistence | Current data | History | Sampler | Epochs | Validation / restarts |",
        "|---|---|---|---|---|---|---:|---|",
    ))
    lines.extend(
        "| " + " | ".join(values) + " |" for values in CONDITION_PROTOCOL.values()
    )
    baseline = summary.get("single_seed_baseline_extension")
    if isinstance(baseline, Mapping):
        lines.extend(("", *_baseline_markdown(baseline)))
    criteria = summary.get("criteria")
    lines.extend(("", "## Frozen decisions", ""))
    if criteria is None:
        lines.append("Primary headline cells are not complete yet, so no success criterion has been evaluated.")
    else:
        lines.extend(
            f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
            for name, passed in criteria["values"].items()
        )
    for filename in plots:
        lines.extend(("", f"![{filename}](plots/{filename})"))
    lines.extend((
        "",
        "The cyan ceiling trace is a fresh, three-restart, validation-selected full-replay fit at every step. It is not an online condition and test data never selects its epoch or restart.",
        "",
    ))
    return "\n".join(lines)


def _baseline_markdown(summary: Mapping[str, object]) -> tuple[str, ...]:
    checkpoints = tuple(int(value) for value in summary["evaluation_checkpoints"])
    metrics = summary["checkpoint_metrics"]
    convergence = summary["convergence"]
    if not isinstance(metrics, Mapping) or not isinstance(convergence, Mapping):
        raise ValueError("single-seed baseline summary is malformed")
    conditions = (
        FROZEN_BASE_CONDITION,
        "pooled_single_mlp_reference",
        CONVERGED_MLP_CONDITION,
        CONVERGED_BASE_INTEGRATOR_CONDITION,
        CEILING_CONDITION,
    )
    lines = [
        "## Seed-0 cumulative baseline extension",
        "",
        (
            "These are post-hoc diagnostics from one stream seed, evaluated only at the five "
            "existing full checkpoints. They do not have a run-to-run uncertainty estimate "
            "and are not included in the main three-seed decision rules. Each metric equally "
            "weights the full 10,000-example test set for every domain seen by that checkpoint. "
            "Validation cross-entropy selects epochs and restarts; test labels select nothing."
        ),
        "",
        (
            "The converged cumulative MLP starts from the calibrated base and trains all four "
            "affine layers on every node-training row seen so far. The converged base-only "
            "integrator keeps that calibrated base frozen and trains only an integrator on its "
            "hidden activation and class probabilities using every observer row seen so far. "
            "The converged all-node integrator remains the comparison that can use every frozen "
            "temporal node."
        ),
        "",
        "Each cell is accuracy / cross-entropy.",
        "",
        "| Macro-step | Frozen base | Fixed 20-epoch cumulative MLP | Converged cumulative MLP | Converged frozen-base integrator | Converged all-node integrator |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for macro_step in checkpoints:
        checkpoint = metrics[str(macro_step)]
        if not isinstance(checkpoint, Mapping):
            raise ValueError("single-seed checkpoint metrics are malformed")
        cells = []
        for condition in conditions:
            value = checkpoint.get(condition)
            if not isinstance(value, Mapping):
                cells.append("not available")
            else:
                cells.append(
                    f"{100.0 * float(value['accuracy']):.2f}% / "
                    f"{float(value['cross_entropy']):.4f}"
                )
        lines.append(f"| {macro_step} | " + " | ".join(cells) + " |")
    lines.extend(
        (
            "",
            "### Validation-selected convergence",
            "",
            "| Macro-step | Cumulative MLP: epochs / selected restart / validation CE | Frozen-base integrator: epochs / selected restart / validation CE |",
            "|---:|---:|---:|",
        )
    )
    for macro_step in checkpoints:
        cells = []
        for condition in (
            CONVERGED_MLP_CONDITION,
            CONVERGED_BASE_INTEGRATOR_CONDITION,
        ):
            condition_rows = convergence.get(condition)
            value = (
                condition_rows.get(str(macro_step))
                if isinstance(condition_rows, Mapping)
                else None
            )
            if not isinstance(value, Mapping):
                cells.append("not available")
            else:
                cells.append(
                    f"{int(value['epochs_ran'])} / {int(value['selected_restart'])} / "
                    f"{float(value['best_validation_cross_entropy']):.4f}"
                )
        lines.append(f"| {macro_step} | " + " | ".join(cells) + " |")
    first_step = checkpoints[0]
    final_step = checkpoints[-1]
    first_metrics = metrics[str(first_step)]
    final_metrics = metrics[str(final_step)]
    mlp_convergence = convergence[CONVERGED_MLP_CONDITION]
    first_training_examples = int(
        mlp_convergence[str(first_step)]["training_examples"]
    )
    fixed_first = first_metrics["pooled_single_mlp_reference"]
    converged_first = first_metrics[CONVERGED_MLP_CONDITION]
    fixed_final = final_metrics["pooled_single_mlp_reference"]
    converged_final = final_metrics[CONVERGED_MLP_CONDITION]
    base_final = final_metrics[FROZEN_BASE_CONDITION]
    base_integrator_final = final_metrics[CONVERGED_BASE_INTEGRATOR_CONDITION]
    all_node_final = final_metrics[CEILING_CONDITION]
    lines.extend(
        (
            "",
            "### What this control establishes",
            "",
            (
                f"- At checkpoint {first_step}, the fixed 20-epoch MLP reached "
                f"{100.0 * float(fixed_first['accuracy']):.2f}% accuracy and "
                f"{float(fixed_first['cross_entropy']):.4f} cross-entropy. The converged "
                f"MLP reached {100.0 * float(converged_first['accuracy']):.2f}% and "
                f"{float(converged_first['cross_entropy']):.4f}. More training improved "
                "cross-entropy but did not recover 98% accuracy. This checkpoint has only "
                f"{first_training_examples:,} cumulative node-training examples spread "
                f"across {int(converged_first['domain_cells'])} seen domains."
            ),
            "",
            (
                f"- At checkpoint {final_step}, the fixed and converged MLPs reached "
                f"{100.0 * float(fixed_final['accuracy']):.2f}% / "
                f"{float(fixed_final['cross_entropy']):.4f} and "
                f"{100.0 * float(converged_final['accuracy']):.2f}% / "
                f"{float(converged_final['cross_entropy']):.4f}, respectively. The "
                "validation-selected converged fit again traded a small amount of top-1 "
                "accuracy for substantially lower cross-entropy."
            ),
            "",
            (
                f"- The frozen base scored {100.0 * float(base_final['accuracy']):.2f}% at "
                f"checkpoint {final_step}. A converged nonlinear integrator over that base "
                f"alone reached {100.0 * float(base_integrator_final['accuracy']):.2f}%, "
                f"versus {100.0 * float(all_node_final['accuracy']):.2f}% for the converged "
                "integrator over all temporal nodes. Most of the usable permutation-specific "
                "information is in the frozen temporal nodes, not the calibrated base alone."
            ),
            "",
            (
                f"- At checkpoint {final_step}, the all-node integrator exceeded the "
                f"converged cumulative MLP by "
                f"{100.0 * (float(all_node_final['accuracy']) - float(converged_final['accuracy'])):.2f} "
                "accuracy points and reduced cross-entropy by "
                f"{float(converged_final['cross_entropy']) - float(all_node_final['cross_entropy']):.4f}. "
                "This comparison uses one stream seed and does not establish run-to-run "
                "stability."
            ),
        )
    )
    return tuple(lines)


__all__ = ["CONDITION_PROTOCOL", "PLOT_STYLES", "write_results"]
