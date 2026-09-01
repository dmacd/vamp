"""Reports, plots, and preregistered criteria for prediction integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.logt_behavioral_integrator import IntegratorConditionState
from apm.experiments.vamp_logt_integrator_metrics import FIXED_CONTROLS
from apm.experiments.vamp_logt_integrator_permuted_config import (
    IntegratorPhaseConfig,
    VampLogTIntegratorConfig,
)
from apm.experiments.vamp_logt_router_reporting import (
    _html,
    _load_jsonl,
    _without_chain,
)
from apm.experiments.vamp_logt_router_state import ActiveAdapterBank

if TYPE_CHECKING:
    from matplotlib.axes import Axes


PLOT_FILES = (
    "01_prediction_accuracy.png",
    "02_prediction_cross_entropy.png",
    "03_current_vs_older_accuracy.png",
    "04_replay_and_controls.png",
    "05_feature_work_scaling.png",
    "06_carry_recovery.png",
)

CEILING_CONDITION = "converged_full_replay_integrator"


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """One redundant visual identity that remains legible without color alone."""

    color: str
    hatch: str
    linestyle: str
    marker: str
    linewidth: float = 2.1
    markersize: float = 4.5
    zorder: int = 3


CONDITION_PLOT_STYLES: Mapping[str, PlotStyle] = {
    "integrator_no_replay": PlotStyle("#111111", "xx", "-.", "X"),
    "integrator_example_replay": PlotStyle("#0072B2", "", "-", "o"),
    "integrator_range_replay": PlotStyle("#E69F00", "//", "--", "s"),
    "base_example_replay": PlotStyle("#009E73", "\\\\", ":", "^"),
    "mean_ensemble": PlotStyle("#CC79A7", "..", "--", "D"),
    "best_single_node": PlotStyle("#B2182B", "++", "-.", "*"),
    "offline_cumulative_integrator": PlotStyle("#6F4EAA", "oo", ":", "P"),
    CEILING_CONDITION: PlotStyle(
        "#00B8D9", "**", "-", "H", linewidth=3.4, markersize=6.0, zorder=6
    ),
}

CONDITION_PLOT_LABELS: Mapping[str, str] = {
    "integrator_no_replay": "integrator: no replay",
    "integrator_example_replay": "integrator: example replay",
    "integrator_range_replay": "integrator: range replay",
    "base_example_replay": "base-only: example replay",
    "mean_ensemble": "mean ensemble",
    "best_single_node": "best node (label-aware oracle)",
    "offline_cumulative_integrator": "offline cumulative integrator",
    CEILING_CONDITION: "integrator: converged full replay (ceiling)",
}

ARCHIVE_PLOT_STYLES: Mapping[tuple[str, str], PlotStyle] = {
    ("integrator_no_replay", "current_range"): PlotStyle(
        "#111111", "", "-", "X"
    ),
    ("integrator_example_replay", "current_range"): PlotStyle(
        "#0072B2", "", "-", "o"
    ),
    ("integrator_range_replay", "current_range"): PlotStyle(
        "#E69F00", "", "-", "s"
    ),
    ("integrator_no_replay", "older_ranges"): PlotStyle(
        "#6F4EAA", "", "--", "v"
    ),
    ("integrator_example_replay", "older_ranges"): PlotStyle(
        "#009E73", "", "--", "^"
    ),
    ("integrator_range_replay", "older_ranges"): PlotStyle(
        "#B2182B", "", "--", "D"
    ),
}


class IntegratorWork(Protocol):
    """Structural type for persisted integrator work counters."""

    node_current_evals: int
    node_historical_evals: int
    base_current_evals: int
    base_historical_evals: int
    evaluation_node_evals: int
    offline_node_evals: int
    offline_example_updates: int
    parent_joint_reference_example_updates: int


def write_phase_report(
    directory: Path,
    config: VampLogTIntegratorConfig,
    phase: str,
    seed: int,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, IntegratorConditionState],
    work: IntegratorWork,
    ledger_rows: Sequence[Mapping[str, object]],
    wall_seconds: float,
    initial_invariants: Mapping[str, bool],
) -> dict[str, object]:
    """Write and validate one completed integrator seed bundle."""
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    rows = tuple(_without_chain(row) for row in ledger_rows)
    evaluation = tuple(row for row in rows if row.get("row_type") == "evaluation")
    training = tuple(row for row in rows if row.get("row_type") == "training")
    primary_scope = (
        "full_test"
        if any(row.get("evaluation_scope") == "full_test" for row in evaluation)
        else "test_subset"
    )
    final_step = bank.topology.processed_blocks
    final = tuple(
        row
        for row in evaluation
        if row.get("evaluation_scope") == primary_scope
        and row.get("group") == "micro"
        and int(row["macro_step"]) == final_step
    )
    final_by_condition = {str(row["condition"]): row for row in final}
    expected = {*conditions, *FIXED_CONTROLS}
    if phase == "primary" and final_step in config.evaluation.full_checkpoints:
        expected.add("offline_cumulative_integrator")
    if set(final_by_condition) != expected:
        raise RuntimeError("completed seed lacks one or more integration conditions")
    phase_config = config.smoke if phase == "smoke" else config.primary
    replay_rows = tuple(
        row
        for row in training
        if row["condition"] != "integrator_no_replay"
        and int(row["macro_step"]) > 1
    )
    acceptance = {
        "all_primary_metrics_finite": all(
            math.isfinite(float(row["mean_cross_entropy"]))
            and math.isfinite(float(row["accuracy"]))
            for row in evaluation
        ),
        "exact_historical_budget": all(
            int(row["historical_examples"]) == phase_config.historical_budget
            for row in replay_rows
        ),
        "future_slot_columns_zero": bool(
            initial_invariants["future_slot_columns_zero"]
        ),
        "fixed_budget_training_work": _fixed_budget_training_work(
            rows, phase_config, config
        ),
        "inactive_slots_zero": all(
            bool(row.get("inactive_slots_zero", True)) for row in training
        ),
        "loss_decrease_fraction": float(
            np.mean(
                [
                    float(row["objective_after"]) <= float(row["objective_before"])
                    for row in training
                ]
            )
        ),
        "node_parameters_unchanged": all(
            bool(row["node_parameters_unchanged"]) for row in training
        ),
        "mean_ensemble_initial_parity": bool(
            initial_invariants["mean_ensemble_initial_parity"]
        ),
        "one_node_initial_parity": bool(
            initial_invariants["one_node_initial_parity"]
        ),
        "zero_residual_output": bool(initial_invariants["zero_residual_output"]),
    }
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
        "condition_final_metrics": {
            name: {
                "accuracy": float(row["accuracy"]),
                "mean_cross_entropy": float(row["mean_cross_entropy"]),
            }
            for name, row in sorted(final_by_condition.items())
        },
        "final_macro_step": final_step,
        "metric_rows": len(evaluation),
        "phase": phase,
        "run_seed": seed,
        "schema_version": "vamp-logt-integrator-seed-summary-v1",
        "wall_seconds": wall_seconds,
        "work": _jsonable_work(work),
    }
    _write_csv(
        directory / "seed_summary.csv",
        tuple(summary["condition_final_metrics"].items()),
    )
    _write_plots(directory / "plots", rows, config)
    markdown = _seed_markdown(summary, config)
    atomic_write(directory / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        directory / "RESULTS.html",
        _html(
            markdown,
            directory,
            "Permuted-MNIST LogT prediction integrator",
        ).encode("utf-8"),
    )
    publish_immutable_json(summary_path, summary)
    required = ("metrics.jsonl", "seed_summary.csv", "RESULTS.md", "RESULTS.html")
    missing = tuple(name for name in required if not (directory / name).is_file())
    missing_plots = tuple(
        name for name in PLOT_FILES if not (directory / "plots" / name).is_file()
    )
    if missing or missing_plots:
        raise RuntimeError(f"incomplete integrator result: {missing + missing_plots}")
    return load_canonical_json(summary_path)


def write_results(
    run_root: Path,
    config: VampLogTIntegratorConfig,
    parent_summary: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate completed seeds and evaluate every frozen success criterion."""
    summary_paths = tuple(sorted(run_root.glob("*/seed-*/summary.json")))
    summaries = tuple(load_canonical_json(path) for path in summary_paths)
    primary = tuple(row for row in summaries if row["phase"] == "primary")
    primary_seeds = {int(row["run_seed"]) for row in primary}
    complete = primary_seeds == set(config.primary.seeds)
    all_rows = tuple(
        _without_chain(row)
        for path in sorted(run_root.glob("primary/seed-*/metrics.jsonl"))
        for row in _load_jsonl(path)
    )
    high = tuple(
        row
        for row in all_rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "full_test"
        and row.get("group") == "micro"
        and int(row["macro_step"]) in {15, 31, 63}
    )
    means = _condition_means(high)
    replay_names = tuple(
        name
        for name in ("integrator_example_replay", "integrator_range_replay")
        if name in means
    )
    best = (
        min(replay_names, key=lambda name: means[name]["mean_cross_entropy"])
        if replay_names
        else None
    )
    closure = _offline_gap_closure(means, best)
    retention = _retention_means(all_rows)
    parent_example = (
        parent_summary.get("condition_high_checkpoint_means", {}).get(
            "example_soft", {}
        )
    )
    criteria = _criteria(
        means, best, closure, retention, parent_example, primary, high
    )
    criteria = {name: complete and value for name, value in criteria.items()}
    summary: dict[str, object] = {
        "completed_primary_seeds": len(primary_seeds),
        "condition_high_checkpoint_means": means,
        "criteria": criteria,
        "offline_gap_closure": closure,
        "parent_example_soft": parent_example,
        "retention_high_checkpoint_means": retention,
        "schema_version": "vamp-logt-integrator-aggregate-summary-v1",
        "selected_best_replay_condition": best,
        "status": "complete" if complete else "partial",
    }
    _write_aggregate_csv(run_root / "seed_summary.csv", summaries)
    if all_rows:
        _write_plots(run_root / "plots", all_rows, config)
    markdown = _aggregate_markdown(summary, config)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(
            markdown,
            run_root,
            "Permuted-MNIST LogT prediction integrator",
        ).encode("utf-8"),
    )
    summary_path = run_root / "summary.json"
    if summary_path.is_file():
        if load_canonical_json(summary_path) != summary:
            atomic_write(summary_path, canonical_json_bytes(summary))
    else:
        publish_immutable_json(summary_path, summary)
    return summary


def rerender_results(
    run_root: Path,
    config: VampLogTIntegratorConfig,
    ceiling_run_root: Path | None = None,
) -> Path:
    """Regenerate derived figures, optionally with an authenticated ceiling overlay."""
    protocol_path = run_root / "protocol.json"
    aggregate_summary_path = run_root / "summary.json"
    if not protocol_path.is_file() or not aggregate_summary_path.is_file():
        raise ValueError("prediction-integrator run is incomplete")
    protocol = load_canonical_json(protocol_path)
    aggregate_summary = load_canonical_json(aggregate_summary_path)
    if (
        protocol.get("config_hash") != config.config_hash
        or aggregate_summary.get("status") != "complete"
    ):
        raise ValueError("prediction-integrator report identity does not match config")

    ceiling_rows = (
        _load_ceiling_overlay_rows(run_root, ceiling_run_root, config)
        if ceiling_run_root is not None
        else {}
    )
    primary_rows: list[Mapping[str, object]] = []
    seed_directories = tuple(sorted(run_root.glob("*/seed-*")))
    if not seed_directories:
        raise ValueError("prediction-integrator run has no seed reports")
    for directory in seed_directories:
        metrics_path = directory / "metrics.jsonl"
        summary_path = directory / "summary.json"
        markdown_path = directory / "RESULTS.md"
        if (
            not metrics_path.is_file()
            or not summary_path.is_file()
            or not markdown_path.is_file()
        ):
            raise ValueError(f"incomplete seed report: {directory}")
        summary = load_canonical_json(summary_path)
        phase = directory.parent.name
        if summary.get("phase") != phase:
            raise ValueError(f"seed report phase mismatch: {directory}")
        seed = int(summary["run_seed"])
        rows = tuple(_without_chain(row) for row in _load_jsonl(metrics_path))
        plotted_rows = (*rows, *ceiling_rows.get((phase, seed), ()))
        _write_plots(directory / "plots", plotted_rows, config)
        _rewrite_embedded_html(
            directory,
            f"Permuted-MNIST LogT prediction integrator: {phase} {directory.name}",
        )
        if phase == "primary":
            primary_rows.extend(plotted_rows)

    if not primary_rows:
        raise ValueError("prediction-integrator run has no primary metric rows")
    _write_plots(run_root / "plots", primary_rows, config)
    _rewrite_embedded_html(run_root, "Permuted-MNIST LogT prediction integrator")
    return run_root


def _load_ceiling_overlay_rows(
    parent_run_root: Path,
    ceiling_run_root: Path,
    config: VampLogTIntegratorConfig,
) -> dict[tuple[str, int], tuple[Mapping[str, object], ...]]:
    """Load every-step ceiling rows only after authenticating both sealed runs."""
    ceiling_protocol_path = ceiling_run_root / "protocol.json"
    ceiling_summary_path = ceiling_run_root / "summary.json"
    parent_protocol_path = parent_run_root / "protocol.json"
    parent_summary_path = parent_run_root / "summary.json"
    required = (
        ceiling_protocol_path,
        ceiling_summary_path,
        parent_protocol_path,
        parent_summary_path,
    )
    if any(not path.is_file() for path in required):
        raise ValueError("ceiling overlay requires complete parent and ceiling reports")

    protocol = load_canonical_json(ceiling_protocol_path)
    summary = load_canonical_json(ceiling_summary_path)
    ceiling_config = protocol.get("config")
    if not isinstance(ceiling_config, Mapping):
        raise ValueError("ceiling overlay protocol lacks its resolved configuration")
    parent = ceiling_config.get("parent_integrator")
    if not isinstance(parent, Mapping):
        raise ValueError("ceiling overlay protocol lacks its parent binding")

    expected_protocol_sha256 = file_sha256(parent_protocol_path)
    expected_summary_sha256 = file_sha256(parent_summary_path)
    if (
        protocol.get("config_hash") != ceiling_run_root.name
        or parent.get("run_id") != parent_run_root.name
        or parent.get("protocol_sha256") != expected_protocol_sha256
        or parent.get("summary_sha256") != expected_summary_sha256
        or protocol.get("parent_integrator_protocol_sha256")
        != expected_protocol_sha256
        or protocol.get("parent_integrator_summary_sha256")
        != expected_summary_sha256
        or summary.get("status") != "complete"
        or summary.get("ceiling_certified") is not True
        or summary.get("completed_primary_seeds") != len(config.primary.seeds)
    ):
        raise ValueError("ceiling overlay does not authenticate the completed parent run")

    protocol_ledgers = protocol.get("parent_integrator_metric_ledger_sha256")
    configured_ledgers = parent.get("primary_metric_ledger_sha256")
    expected_seed_keys = {str(seed) for seed in config.primary.seeds}
    if (
        not isinstance(protocol_ledgers, Mapping)
        or not isinstance(configured_ledgers, Mapping)
        or dict(protocol_ledgers) != dict(configured_ledgers)
        or set(protocol_ledgers) != expected_seed_keys
    ):
        raise ValueError("ceiling overlay parent-ledger binding is incomplete")
    for seed in config.primary.seeds:
        ledger_path = parent_run_root / "primary" / f"seed-{seed}" / "metrics.jsonl"
        if (
            not ledger_path.is_file()
            or protocol_ledgers[str(seed)] != file_sha256(ledger_path)
        ):
            raise ValueError("ceiling overlay parent metric ledgers changed")

    overlay: dict[tuple[str, int], tuple[Mapping[str, object], ...]] = {}
    for phase, phase_config in (("smoke", config.smoke), ("primary", config.primary)):
        expected_steps = set(range(1, phase_config.macro_steps + 1))
        for seed in phase_config.seeds:
            directory = ceiling_run_root / phase / f"seed-{seed}"
            metrics_path = directory / "metrics.jsonl"
            seed_summary_path = directory / "summary.json"
            if not metrics_path.is_file() or not seed_summary_path.is_file():
                raise ValueError(f"ceiling overlay is missing {phase} seed {seed}")
            seed_summary = load_canonical_json(seed_summary_path)
            acceptance = seed_summary.get("acceptance")
            if (
                seed_summary.get("phase") != phase
                or seed_summary.get("run_seed") != seed
                or seed_summary.get("final_macro_step") != phase_config.macro_steps
                or not isinstance(acceptance, Mapping)
                or not acceptance
                or not all(value is True for value in acceptance.values())
            ):
                raise ValueError(f"ceiling overlay seed is not certified: {phase} {seed}")
            candidates = tuple(
                _without_chain(row)
                for row in _load_jsonl(metrics_path)
                if row.get("row_type") == "evaluation"
                and row.get("condition") == CEILING_CONDITION
                and row.get("group") == "micro"
                and row.get("evaluation_scope") in {"test_subset", "full_test"}
            )
            selected = _primary_test_rows(candidates)
            if (
                len(selected) != phase_config.macro_steps
                or {int(row["macro_step"]) for row in selected} != expected_steps
                or any(int(row["run_seed"]) != seed for row in selected)
                or any(
                    not math.isfinite(float(row[field]))
                    for row in selected
                    for field in ("accuracy", "mean_cross_entropy")
                )
            ):
                raise ValueError(
                    f"ceiling overlay lacks one finite test point per step: {phase} {seed}"
                )
            overlay[(phase, seed)] = selected
    return overlay


def _rewrite_embedded_html(directory: Path, title: str) -> None:
    """Refresh standalone HTML after its referenced plots change."""
    markdown = (directory / "RESULTS.md").read_text(encoding="utf-8")
    atomic_write(
        directory / "RESULTS.html",
        _html(markdown, directory, title).encode("utf-8"),
    )


def _criteria(
    means: Mapping[str, Mapping[str, float]],
    best: str | None,
    closure: float | None,
    retention: Mapping[str, Mapping[str, float]],
    parent: Mapping[str, object],
    primary: Sequence[Mapping[str, object]],
    high: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    best_values = {} if best is None else means[best]
    current_loss = (
        math.inf
        if best is None
        else retention.get(best, {}).get("current_accuracy_loss_pp", math.inf)
    )
    old_better = (
        best is not None
        and retention.get(best, {}).get("older_mean_cross_entropy", math.inf)
        < retention.get("integrator_no_replay", {}).get(
            "older_mean_cross_entropy", -math.inf
        )
    )
    structural_gates = (
        "all_primary_metrics_finite",
        "exact_historical_budget",
        "fixed_budget_training_work",
        "future_slot_columns_zero",
        "inactive_slots_zero",
        "mean_ensemble_initial_parity",
        "node_parameters_unchanged",
        "one_node_initial_parity",
        "zero_residual_output",
    )
    acceptance = bool(primary) and all(
        all(bool(row["acceptance"][name]) for name in structural_gates)
        for row in primary
    )
    return {
        "1_replay_beats_no_replay_and_mean_ensemble": (
            best is not None
            and best_values["mean_cross_entropy"]
            < means.get("integrator_no_replay", {}).get("mean_cross_entropy", -math.inf)
            and best_values["mean_cross_entropy"]
            < means.get("mean_ensemble", {}).get("mean_cross_entropy", -math.inf)
        ),
        "2_closes_at_least_75_percent_to_offline": (
            closure is not None and closure >= 0.75
        ),
        "3_retention_with_at_most_2pp_current_loss": old_better and current_loss <= 2.0,
        "4_full_nodes_beat_base_only": (
            best is not None
            and best_values["mean_cross_entropy"]
            < means.get("base_example_replay", {}).get("mean_cross_entropy", -math.inf)
            and best_values["accuracy"]
            >= means.get("base_example_replay", {}).get("accuracy", math.inf)
        ),
        "5_beats_sealed_example_soft_router": (
            best is not None
            and best_values["mean_cross_entropy"]
            < float(parent.get("selected_mean_cross_entropy", -math.inf))
            and best_values["accuracy"]
            > float(parent.get("selected_accuracy", math.inf))
        ),
        "6_fixed_budget_and_structural_gates": acceptance,
        "7_attribution_controls_present": all(
            name in means
            for name in (
                "offline_cumulative_integrator",
                "base_example_replay",
                "best_single_node",
            )
        )
        and any(row.get("joint_iid_mean_cross_entropy") is not None for row in high),
    }


def _fixed_budget_training_work(
    rows: Sequence[Mapping[str, object]],
    phase: IntegratorPhaseConfig,
    config: VampLogTIntegratorConfig,
) -> bool:
    accounting = tuple(row for row in rows if row.get("row_type") == "accounting")
    if not accounting:
        return False
    final = max(accounting, key=lambda row: int(row["macro_step"]))
    macro_steps = phase.macro_steps
    historical_budget = phase.historical_budget
    expected_current = sum(
        step.bit_count() * config.benchmark.integrator_batch_size
        for step in range(1, macro_steps + 1)
    )
    expected_historical = sum(
        step.bit_count() * 2 * historical_budget
        for step in range(2, macro_steps + 1)
    )
    return (
        int(final["work"]["node_current_evals"]) == expected_current
        and int(final["work"]["node_historical_evals"]) == expected_historical
    )


def _condition_means(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    fields = (
        "mean_cross_entropy",
        "accuracy",
        "brier_score",
        "best_single_node_mean_cross_entropy",
        "best_single_node_accuracy",
        "joint_iid_mean_cross_entropy",
        "joint_iid_accuracy",
    )
    return {
        name: {
            field: float(
                np.mean(
                    [
                        float(row[field])
                        for row in rows
                        if row["condition"] == name and row.get(field) is not None
                    ]
                )
            )
            for field in fields
        }
        for name in sorted(set(str(row["condition"]) for row in rows))
    }


def _offline_gap_closure(
    means: Mapping[str, Mapping[str, float]], best: str | None
) -> float | None:
    if (
        best is None
        or "integrator_no_replay" not in means
        or "offline_cumulative_integrator" not in means
    ):
        return None
    no_replay = means["integrator_no_replay"]["mean_cross_entropy"]
    offline = means["offline_cumulative_integrator"]["mean_cross_entropy"]
    denominator = no_replay - offline
    return (
        None
        if denominator <= 0.0
        else 1.0 - (means[best]["mean_cross_entropy"] - offline) / denominator
    )


def _retention_means(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    selected = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "evaluation_archive"
        and int(row["macro_step"]) in {15, 31, 63}
        and row.get("group") in {"current_range", "older_ranges"}
    )
    no_replay_current = np.mean(
        [
            float(row["accuracy"])
            for row in selected
            if row["condition"] == "integrator_no_replay"
            and row["group"] == "current_range"
        ]
    ) if selected else math.nan
    return {
        name: {
            "current_accuracy": float(
                np.mean(
                    [
                        float(row["accuracy"])
                        for row in selected
                        if row["condition"] == name and row["group"] == "current_range"
                    ]
                )
            ),
            "current_accuracy_loss_pp": float(
                100.0
                * (
                    no_replay_current
                    - np.mean(
                        [
                            float(row["accuracy"])
                            for row in selected
                            if row["condition"] == name
                            and row["group"] == "current_range"
                        ]
                    )
                )
            ),
            "older_mean_cross_entropy": float(
                np.mean(
                    [
                        float(row["mean_cross_entropy"])
                        for row in selected
                        if row["condition"] == name and row["group"] == "older_ranges"
                    ]
                )
            ),
        }
        for name in sorted(set(str(row["condition"]) for row in selected))
    }


def _write_plots(
    root: Path,
    rows: Sequence[Mapping[str, object]],
    config: VampLogTIntegratorConfig,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vamp-logt-integrator-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    context_steps = _context_steps(rows, config)
    evaluation = _primary_test_rows(
        tuple(
            row
            for row in rows
            if row.get("row_type") == "evaluation" and row.get("group") == "micro"
        )
    )
    conditions = tuple(
        name
        for name in (
            *config.primary.conditions,
            "mean_ensemble",
            "best_single_node",
            "offline_cumulative_integrator",
            CEILING_CONDITION,
        )
        if any(row.get("condition") == name for row in evaluation)
    )
    ceiling_overlaid = CEILING_CONDITION in conditions
    for filename, field, title, ylabel in (
        (PLOT_FILES[0], "accuracy", "Prediction accuracy", "Accuracy"),
        (PLOT_FILES[1], "mean_cross_entropy", "Prediction cross-entropy", "Cross-entropy"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        _add_context_regions(axis, context_steps)
        for condition in conditions:
            points = _mean_points(evaluation, condition, field)
            if points:
                _plot_series(
                    axis,
                    points,
                    CONDITION_PLOT_LABELS[condition],
                    CONDITION_PLOT_STYLES[condition],
                )
        displayed_title = (
            f"{title} (converged full-replay ceiling overlaid)"
            if ceiling_overlaid
            else title
        )
        axis.set(title=displayed_title, xlabel="Macro-step", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, ncol=2)
        figure.savefig(root / filename, dpi=170)
        plt.close(figure)
    archive = tuple(
        row
        for row in rows
        if row.get("row_type") == "evaluation"
        and row.get("evaluation_scope") == "evaluation_archive"
        and row.get("group") in {"current_range", "older_ranges"}
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _add_context_regions(axis, context_steps)
    for condition in config.primary.conditions[:3]:
        for group in ("current_range", "older_ranges"):
            points = _mean_points(
                tuple(row for row in archive if row.get("group") == group),
                condition,
                "accuracy",
            )
            if points:
                label = f"{CONDITION_PLOT_LABELS[condition]}: {group.replace('_', ' ')}"
                _plot_series(
                    axis,
                    points,
                    label,
                    ARCHIVE_PLOT_STYLES[(condition, group)],
                )
    axis.set(title="Current versus older-range accuracy", xlabel="Macro-step", ylabel="Accuracy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5, ncol=2)
    figure.savefig(root / PLOT_FILES[2], dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    names = tuple(
        name
        for name in (
            *config.primary.conditions,
            "mean_ensemble",
            "offline_cumulative_integrator",
            CEILING_CONDITION,
        )
        if any(row.get("condition") == name for row in evaluation)
    )
    available_steps = sorted(set(int(row["macro_step"]) for row in evaluation))
    comparison_steps = tuple(
        step for step in available_steps if step in {15, 31, 63}
    ) or tuple(available_steps[-1:])
    values = [
        np.mean(
            [
                float(row["mean_cross_entropy"])
                for row in evaluation
                if row.get("condition") == name
                and int(row["macro_step"]) in comparison_steps
            ]
        )
        for name in names
    ]
    styles = tuple(CONDITION_PLOT_STYLES[name] for name in names)
    bars = axis.bar(
        tuple(CONDITION_PLOT_LABELS[name] for name in names),
        values,
        color=tuple(style.color for style in styles),
        edgecolor="#111111",
        linewidth=0.8,
    )
    for bar, style in zip(bars, styles, strict=True):
        bar.set_hatch(style.hatch)
    axis.set(title="High-checkpoint predictor comparison", ylabel="Cross-entropy")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(root / PLOT_FILES[3], dpi=170)
    plt.close(figure)
    accounting = tuple(row for row in rows if row.get("row_type") == "accounting")
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _add_context_regions(axis, context_steps)
    if accounting:
        steps = sorted(set(int(row["macro_step"]) for row in accounting))
        work = [
            np.mean(
                [
                    int(row["work"]["node_current_evals"])
                    + int(row["work"]["node_historical_evals"])
                    for row in accounting
                    if int(row["macro_step"]) == step
                ]
            )
            for step in steps
        ]
        reference = np.asarray([step * math.log2(max(step, 2)) for step in steps])
        if reference[-1] > 0:
            reference *= work[-1] / reference[-1]
        axis.plot(
            steps,
            work,
            color="#0072B2",
            linewidth=2.3,
            marker="o",
            markevery=max(1, len(steps) // 12),
            label="measured node-feature evaluations",
        )
        axis.plot(
            steps,
            reference,
            color="#B2182B",
            linewidth=2.3,
            linestyle="--",
            marker="s",
            markevery=max(1, len(steps) // 12),
            label="scaled t log2(t)",
        )
    axis.set(
        title="Fixed-budget feature work",
        xlabel="Macro-step",
        ylabel="Cumulative evaluations",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(root / PLOT_FILES[4], dpi=170)
    plt.close(figure)
    training = tuple(
        row
        for row in rows
        if row.get("row_type") == "training" and row.get("is_carry")
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    _add_context_regions(axis, context_steps)
    for condition in config.primary.conditions:
        points = _mean_points(
            training,
            condition,
            "carry_accuracy_change_from_baseline",
        )
        if points:
            _plot_series(
                axis,
                points,
                CONDITION_PLOT_LABELS[condition],
                CONDITION_PLOT_STYLES[condition],
            )
    axis.axhline(0.0, color="black", linewidth=1)
    seed_count = len({int(row["run_seed"]) for row in training})
    axis.set(
        title=(
            "Integrator recovery on carry steps"
            if seed_count <= 1
            else f"Mean integrator recovery on carry steps ({seed_count} seeds)"
        ),
        xlabel="Macro-step",
        ylabel="Post-update accuracy minus mean-ensemble baseline",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.savefig(root / PLOT_FILES[5], dpi=170)
    plt.close(figure)


def _plot_series(
    axis: Axes,
    points: Sequence[tuple[int, float]],
    label: str,
    style: PlotStyle,
) -> None:
    """Draw one line with redundant color, dash, and marker encodings."""
    axis.plot(
        *zip(*points, strict=True),
        color=style.color,
        label=label,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        marker=style.marker,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.45,
        markersize=style.markersize,
        zorder=style.zorder,
    )


def _context_steps(
    rows: Sequence[Mapping[str, object]],
    config: VampLogTIntegratorConfig,
) -> tuple[int, ...]:
    """Partition the represented schedule into shuffled eight-domain blocks."""
    maximum_step = max(
        (int(row["macro_step"]) for row in rows if "macro_step" in row),
        default=0,
    )
    if maximum_step > config.primary.macro_steps:
        raise ValueError("plot rows exceed the configured Permuted-MNIST horizon")
    complete, remainder = divmod(maximum_step, config.benchmark.domain_count)
    return (config.benchmark.domain_count,) * complete + (
        ((remainder,) if remainder else ())
    )


def _add_context_regions(axis: Axes, context_steps: Sequence[int]) -> None:
    """Mark independently shuffled eight-domain blocks."""
    start = 0.5
    for context, count in enumerate(context_steps):
        if count <= 0:
            continue
        stop = start + count
        if context % 2:
            axis.axvspan(start, stop, color="#CBD5E1", alpha=0.16, zorder=0)
        axis.text(
            (start + stop) / 2.0,
            0.985,
            f"B{context}",
            color="#475569",
            fontsize=7.5,
            fontweight="bold",
            horizontalalignment="center",
            transform=axis.get_xaxis_transform(),
            verticalalignment="top",
            zorder=4,
        )
        if context + 1 < len(context_steps):
            axis.axvline(
                stop,
                color="#64748B",
                linestyle=(0, (2, 2)),
                linewidth=1.0,
                zorder=1,
            )
        start = stop


def _primary_test_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    full = {
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
            not in full
        )
    )


def _mean_points(
    rows: Sequence[Mapping[str, object]], condition: str, field: str
) -> tuple[tuple[int, float], ...]:
    selected = tuple(row for row in rows if row.get("condition") == condition)
    return tuple(
        (
            step,
            float(
                np.mean(
                    [
                        float(row[field])
                        for row in selected
                        if int(row["macro_step"]) == step
                    ]
                )
            ),
        )
        for step in sorted(set(int(row["macro_step"]) for row in selected))
    )


def _write_csv(
    path: Path, rows: Sequence[tuple[str, Mapping[str, object]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [["condition", "accuracy", "mean_cross_entropy"]]
    lines.extend(
        [name, str(values["accuracy"]), str(values["mean_cross_entropy"])]
        for name, values in rows
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        csv.writer(output).writerows(lines)


def _write_aggregate_csv(
    path: Path, summaries: Sequence[Mapping[str, object]]
) -> None:
    rows = []
    for summary in summaries:
        rows.extend(
            (
                summary["phase"],
                summary["run_seed"],
                name,
                values["accuracy"],
                values["mean_cross_entropy"],
            )
            for name, values in summary["condition_final_metrics"].items()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("phase", "run_seed", "condition", "accuracy", "mean_cross_entropy"))
        writer.writerows(rows)


def _jsonable_work(work: IntegratorWork) -> dict[str, int]:
    return {
        name: int(getattr(work, name))
        for name in (
            "node_current_evals",
            "node_historical_evals",
            "base_current_evals",
            "base_historical_evals",
            "evaluation_node_evals",
            "offline_node_evals",
            "offline_example_updates",
            "parent_joint_reference_example_updates",
        )
    }


def _seed_markdown(
    summary: Mapping[str, object], config: VampLogTIntegratorConfig
) -> str:
    lines = "\n".join(
        f"| `{name}` | {float(values['accuracy']):.4f} | {float(values['mean_cross_entropy']):.5f} |"
        for name, values in summary["condition_final_metrics"].items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    return f"""# Permuted-MNIST LogT prediction integrator: {summary['phase']} seed {summary['run_seed']}

## Final metrics

| Condition | Accuracy | Cross-entropy |
|---|---:|---:|
{lines}

## Smoke and structural gates

`{summary['acceptance']}`

## Figures

{figures}

## Protocol

The exact configuration identity is `{config.config_hash}`. Labels supervise only the final ten-class integrator output; labels and task metadata never enter its 973-value behavior input.
"""


def _aggregate_markdown(
    summary: Mapping[str, object], config: VampLogTIntegratorConfig
) -> str:
    lines = "\n".join(
        f"| `{name}` | {float(values['accuracy']):.4f} | {float(values['mean_cross_entropy']):.5f} |"
        for name, values in summary["condition_high_checkpoint_means"].items()
    ) or "| _No complete primary rows_ | — | — |"
    criteria = "\n".join(
        f"| {name.replace('_', ' ')} | `{value}` |"
        for name, value in summary["criteria"].items()
    )
    figures = "\n\n".join(f"![{name}](plots/{name})" for name in PLOT_FILES)
    return f"""# LogT-VAMP prediction integrator on Permuted-MNIST

## Outcome

The run status is `{summary['status']}` with {summary['completed_primary_seeds']} of {len(config.primary.seeds)} primary seeds complete. The best online replay condition is `{summary['selected_best_replay_condition']}`.

| Condition | Accuracy | Cross-entropy |
|---|---:|---:|
{lines}

The no-replay-to-offline gap closure is `{summary['offline_gap_closure']}`.

## Preregistered criteria

| Criterion | Result |
|---|---|
{criteria}

## Interpretation boundary

The offline cumulative integrator separates representation capacity from online replay optimization. The base-only control tests whether node-specific behaviors add information. Best-single-node selection remains diagnostic and is not an upper bound for a combined predictor.

## Figures

{figures if summary['condition_high_checkpoint_means'] else 'Figures will appear after primary rows exist.'}

## Exact protocol

The resolved configuration hash is `{config.config_hash}` and the immutable plan is `docs/logt_vamp_permuted_mnist_integrator_protocol.md`.
"""


__all__ = [
    "ARCHIVE_PLOT_STYLES",
    "CEILING_CONDITION",
    "CONDITION_PLOT_STYLES",
    "PLOT_FILES",
    "_load_ceiling_overlay_rows",
    "rerender_results",
    "write_phase_report",
    "write_results",
]
