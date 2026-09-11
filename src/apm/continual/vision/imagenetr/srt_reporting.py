"""A standalone, reproducible SRT report with authenticated historical comparisons."""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256
from apm.continual.vision.imagenetr.joint_convergence_reporting import draw_joint_endpoint, joint_report_parts, load_joint_reference
from apm.continual.vision.imagenetr.persistent_affine_reporting import (
    LABEL_H4096, LABEL_H8192, LABEL_MLP, LABEL_ORACLE_H4096, LABEL_ORACLE_H8192,
    LABEL_ORACLE_MLP, LABEL_RANK_JOINT, LABEL_STAGE_JOINT, _series,
)
from apm.continual.vision.imagenetr.srt_analysis import ReplayAnalysis, analyze_replay_job
from apm.continual.vision.imagenetr.srt_evidence import CLOCK_FIELDS, read_sealed, sealed_record, trace_batches


TITLE = "ImageNet-R-50: single-adapter Spaced Repetition Training"
CONDITION_LABELS = {
    **{f"{method}_h{capacity}": f"{'SRT' if method == 'srt' else 'Uniform replay'} rank 16, {capacity:,}-equivalent budget"
       for capacity in (1024, 4096) for method in ("srt", "uniform")},
    "persistent_h4096": LABEL_H4096, "persistent_h8192": LABEL_H8192,
    "persistent_mlp_h4096": LABEL_MLP, "joint_rank16": LABEL_STAGE_JOINT,
    "joint_rank_matched": LABEL_RANK_JOINT,
    **{f"{method}_h4096_{profile}_rho80_unit8":
       f"{'SRT' if method == 'srt' else 'Uniform replay'} rank 16, H=4,096; {profile}, old=0.8, unit=8"
       for profile in ("standard", "strict") for method in ("srt", "uniform")},
}
REFERENCE_STYLES = {
    LABEL_H4096: ("#1f77b4", "-"), LABEL_H8192: ("#d95f02", "-"),
    LABEL_STAGE_JOINT: ("#222222", "--"), LABEL_RANK_JOINT: ("#2ca02c", "-."),
    LABEL_MLP: ("#9467bd", "-"), LABEL_ORACLE_H4096: ("#6baed6", ":"),
    LABEL_ORACLE_H8192: ("#fdae6b", ":"), LABEL_ORACLE_MLP: ("#c5b0d5", ":"),
}
NEW_STYLES = {f"{method}_h{capacity}": ("#c62828" if capacity == 1024 else "#00796b", "-" if method == "srt" else "--")
              for capacity in (1024, 4096) for method in ("srt", "uniform")}
FOLLOWUP_STYLES = {f"{method}_h4096_{profile}_rho80_unit8":
                   ("#b5179e" if profile == "standard" else "#9a6700", "-" if method == "srt" else "--")
                   for profile in ("standard", "strict") for method in ("srt", "uniform")}
ALL_STYLES = {**NEW_STYLES, **FOLLOWUP_STYLES}


@dataclass(frozen=True, slots=True)
class ReportTable:
    """One common table definition shared by the PDF, HTML, and Markdown."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ReportSection:
    """A paper-sized page of context, measurements, and at most two figures."""

    title: str
    paragraphs: tuple[str, ...]
    figures: tuple[Path, ...] = ()
    table: ReportTable | None = None


def _write_table(reports: Path, name: str, rows: tuple[dict[str, object], ...]) -> None:
    clock_columns = CLOCK_FIELDS | {"clock_advance", "final_clock", "next_due_clock", "terminal_overdue_ticks"}
    serialized = tuple({name: str(value) if name in clock_columns and value is not None else value
                        for name, value in row.items()} for row in rows)
    frame = pd.DataFrame(serialized)
    atomic_write(reports / f"{name}.json", canonical_json_bytes(list(serialized)))
    atomic_write(reports / f"{name}.csv", frame.to_csv(index=False).encode())
    buffer = BytesIO()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), buffer, compression="zstd")
    atomic_write(reports / f"{name}.parquet", buffer.getvalue())


def reference_results(run: Path) -> dict[str, object]:
    """Authenticate copied reference evidence without opening any old report writer."""
    protocol = read_sealed(run / "protocol.json", "imagenetr50-srt-protocol-v1")
    for name, expected in protocol["reference_hashes"].items():
        if name != "previous_report.pdf" and file_sha256(run / "references" / name) != expected:
            raise ValueError("copied SRT comparison evidence changed")
    affine = read_sealed(run / "references/affine_result.json", "imagenetr50-persistent-affine-result-v1")
    mlp = read_sealed(run / "references/mlp_result.json", "imagenetr50-persistent-mlp-result-v1")
    if mlp["reference_result_hash"] != affine["content_hash"]:
        raise ValueError("copied MLP and affine comparison identities disagree")
    return {**affine, "mlp": mlp}


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _finish_axis(axis: plt.Axes, ylabel: str, title: str) -> None:
    axis.set(xlabel="Tasks observed", ylabel=ylabel, title=title, xlim=(1, 50))
    axis.grid(axis="y", alpha=.25)


def _legend(axis: plt.Axes, columns: int = 2) -> None:
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, [textwrap.fill(label, 42) for label in labels], loc="upper center",
                bbox_to_anchor=(.5, -.17), ncol=columns, frameon=False, fontsize=10)


def _shared_legend(figure: plt.Figure, axes: tuple[plt.Axes, ...]) -> None:
    entries = {label: handle for axis in axes for handle, label in zip(*axis.get_legend_handles_labels(), strict=True)}
    figure.legend(list(entries.values()), [textwrap.fill(label, 42) for label in entries],
                  loc="outside lower center", ncol=2, frameon=False, fontsize=10)


def plot_accuracy(reports: Path, references: dict[str, object], stages: pd.DataFrame) -> Path:
    """Overlay both SRT budgets against the unchanged task-free reference curves."""
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 11), constrained_layout=True)
    for axis, capacity in zip(axes, (1024, 4096), strict=True):
        for label, values in _series(references).items():
            if "diagnostic" not in label:
                color, style = REFERENCE_STYLES[label]
                axis.plot(range(1, 51), values, color=color, linestyle=style, linewidth=1.5, alpha=.75, label=label)
        for method in ("srt", "uniform"):
            condition = f"{method}_h{capacity}"
            rows = stages[stages.condition == condition]
            color, style = NEW_STYLES[condition]
            axis.plot(rows.stage, rows.accuracy, color=color, linestyle=style, linewidth=2.5, label=CONDITION_LABELS[condition])
        _finish_axis(axis, "Test top-1 accuracy (%)", f"{capacity:,}-equivalent training budget")
        draw_joint_endpoint(axis, references)
        _legend(axis)
    return _save_figure(figure, reports / "stage_accuracy.png")


def plot_followup_accuracy(reports: Path, references: dict[str, object], stages: pd.DataFrame) -> Path:
    """Overlay every fixed-policy rerun with all five carried-forward references."""
    figure, axis = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    for label, values in _series(references).items():
        if "diagnostic" not in label:
            color, style = REFERENCE_STYLES[label]
            axis.plot(range(1, 51), values, color=color, linestyle=style, linewidth=1.5, alpha=.7, label=label)
    for condition, (color, style) in FOLLOWUP_STYLES.items():
        rows = stages[stages.condition == condition]
        axis.plot(rows.stage, rows.accuracy, color=color, linestyle=style, linewidth=2.3, label=CONDITION_LABELS[condition])
    _finish_axis(axis, "Test top-1 accuracy (%)", "Fixed H=4,096, historical target 0.8, interval unit 8")
    draw_joint_endpoint(axis, references)
    _legend(axis)
    return _save_figure(figure, reports / "fixed_policy_stage_accuracy.png")


def plot_policy_comparisons(reports: Path, references: dict[str, object], stages: pd.DataFrame) -> Path:
    """Separate the standard-profile H change from the strict-profile mix/spacing change."""
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 9.5), constrained_layout=True)
    for column, (profile, original_capacity, title) in enumerate((
        ("standard", 1024, "Standard: H=1,024 versus 4,096"),
        ("strict", 4096, "Strict: old/unit=0.5/1 versus 0.8/8"),
    )):
        conditions = tuple(f"{method}_h{original_capacity}" for method in ("srt", "uniform")) + tuple(
            f"{method}_h4096_{profile}_rho80_unit8" for method in ("srt", "uniform"))
        for condition in conditions:
            rows = stages[stages.condition == condition]
            color, style = ALL_STYLES[condition]
            for axis, metric in zip(axes[:, column], ("accuracy", "nll"), strict=True):
                axis.plot(rows.stage, rows[metric], color=color, linestyle=style, linewidth=2, label=CONDITION_LABELS[condition])
        axes[0, column].plot(range(1, 51), [row["accuracy"] for row in references["stage_matched_joint"]],
                             color="#222222", linestyle=":", linewidth=1.5, label=LABEL_STAGE_JOINT)
        _finish_axis(axes[0, column], "Test accuracy (%)", title)
        _finish_axis(axes[1, column], "Test negative log likelihood", "Same conditions, probability quality")
        draw_joint_endpoint(axes[0, column], references)
        draw_joint_endpoint(axes[1, column], references, "nll")
    _shared_legend(figure, tuple(axes.flat))
    return _save_figure(figure, reports / "fixed_policy_comparisons.png")


def plot_optimizer_work(reports: Path, stages: pd.DataFrame) -> Path:
    """Expose update-count differences that a presentation budget alone hides."""
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8), constrained_layout=True)
    for condition, (color, _) in ALL_STYLES.items():
        if not condition.startswith("srt_"):
            continue
        rows = stages[stages.condition == condition]
        if rows.empty:
            continue
        axes[0].plot(rows.stage, rows.cumulative_optimizer_steps, color=color, label=CONDITION_LABELS[condition])
        axes[1].plot(rows.stage, rows.mean_batch_size, color=color, label=CONDITION_LABELS[condition])
    axes[1].axhline(64, color="#555555", linestyle=":", linewidth=1)
    _finish_axis(axes[0], "Cumulative optimizer updates", "SRT and its paired uniform control have identical counts")
    _finish_axis(axes[1], "Mean images per update", "64 is the batch maximum, not a fixed batch size")
    _shared_legend(figure, tuple(axes))
    return _save_figure(figure, reports / "optimizer_work_and_batching.png")


def _plot_nll_and_gaps(reports: Path, references: dict[str, object], stages: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 9), constrained_layout=True)
    joint = np.array([row["accuracy"] for row in references["stage_matched_joint"]])
    for condition, (color, style) in ALL_STYLES.items():
        rows = stages[stages.condition == condition]
        if rows.empty:
            continue
        axes[0].plot(rows.stage, rows.nll, color=color, linestyle=style, label=CONDITION_LABELS[condition])
        axes[1].plot(rows.stage, rows.accuracy.to_numpy() - joint, color=color, linestyle=style, label=CONDITION_LABELS[condition])
    for label, values in (
        (LABEL_H4096, [row["evaluation"]["nll"] for row in references["arms"]["4096"]]),
        (LABEL_MLP, [row["evaluation"]["nll"] for row in references["mlp"]["rows"]]),
        (LABEL_RANK_JOINT, [row["nll"] for row in references["rank_matched_joint"]]),
    ):
        color, style = REFERENCE_STYLES[label]
        axes[0].plot(range(1, 51), values, color=color, linestyle=style, alpha=.7, label=label)
    _finish_axis(axes[0], "Test negative log likelihood", "Probability quality; original rank-16 reference NLL was not retained")
    draw_joint_endpoint(axes[0], references, "nll")
    _finish_axis(axes[1], "Accuracy difference (percentage points)", "Difference from stage-matched joint IID, rank 16")
    axes[1].axhline(0, color="#222222", linewidth=1)
    _shared_legend(figure, tuple(axes))
    return _save_figure(figure, reports / "nll_and_joint_gap.png")


def _resource_rows(run: Path, stages: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    references = json.loads((run / "references/resource_metrics.json").read_text())
    prior = tuple({"condition": row["condition_id"], "stage": row["stage"],
                   "training_forwards": row["training_forward_images"], "training_backwards": row["training_backward_images"],
                   "recompute_forwards": row["recompute_forward_images"], "training_wall_seconds": row["training_wall_seconds"],
                   "cumulative_training_forwards": row["cumulative_training_forward_images"],
                   "cumulative_training_backwards": row["cumulative_training_backward_images"],
                   "cumulative_recompute_forwards": row["cumulative_recompute_forward_images"],
                   "cumulative_training_wall_seconds": row["cumulative_training_wall_seconds"],
                   "evaluation_forwards": row["evaluation_forward_images"], "evaluation_wall_seconds": row["evaluation_wall_seconds"],
                   "cost_source": "authenticated prior resource ledger; hierarchy charged where applicable"}
                  for row in references)
    new = tuple({"condition": row["condition"], "stage": row["stage"],
                 **{name: row[name] for name in ("training_forwards", "training_backwards", "recompute_forwards",
                                                "training_wall_seconds", "cumulative_training_wall_seconds", "evaluation_forwards", "evaluation_wall_seconds")},
                 "cumulative_training_forwards": row["cumulative_presentations"],
                 "cumulative_training_backwards": row["cumulative_presentations"], "cumulative_recompute_forwards": 0,
                 "cost_source": "audited committed SRT presentation and batch ledger"}
                for row in stages)
    return new + prior


def _plot_resources(reports: Path, resources: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 9), constrained_layout=True)
    for condition, rows in resources.groupby("condition", sort=False):
        label = CONDITION_LABELS[condition]
        color, style = ALL_STYLES[condition] if condition in ALL_STYLES else REFERENCE_STYLES[label]
        axes[0].plot(rows.stage, rows.cumulative_training_wall_seconds / 60, color=color, linestyle=style, label=label)
        axes[1].plot(rows.stage, rows.cumulative_training_backwards / 1e6, color=color, linestyle=style, label=label)
    _finish_axis(axes[0], "Cumulative training time (minutes)", "Final-run training time; calibration shown separately")
    _finish_axis(axes[1], "Cumulative forward/backward pairs (millions)", "One pair per trained image path; recomputation listed separately")
    _shared_legend(figure, tuple(axes))
    return _save_figure(figure, reports / "cumulative_work.png")


def _gap_cdf(axis: plt.Axes, rows: pd.DataFrame, color: str, style: str, label: str) -> None:
    """Plot binned completed intervals without hiding the open-ended overflow bin."""
    if not len(rows) or not rows["count"].sum():
        return
    cumulative = rows["count"].cumsum() / rows["count"].sum()
    last_nonzero = np.flatnonzero(rows["count"].to_numpy())[-1]
    finite = rows.upper.notna() & (np.arange(len(rows)) <= last_nonzero)
    overflow = rows.loc[rows.upper.isna(), "count"].sum() / rows["count"].sum()
    if overflow:
        label += f" ({100 * overflow:.2f}% beyond plotted range)"
    boundaries, fractions = rows.loc[finite, "upper"].to_numpy(), cumulative[finite].to_numpy()
    axis.step(np.r_[0, boundaries, boundaries[-1] * 1.1], np.r_[0, fractions, fractions[-1]],
              where="post", color=color, linestyle=style, label=label)


def _plot_replay(reports: Path, histograms: pd.DataFrame, stages: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for condition, (color, style) in ALL_STYLES.items():
        if condition not in set(stages.condition):
            continue
        for metric, axis, title in (("step_gap", axes[0, 0], "Actual historical replay gaps"),
                                     ("lateness", axes[0, 1], "Delay beyond scheduled due time")):
            subset = histograms[(histograms.condition == condition) & (histograms.metric == metric)
                                & (histograms.kind == "historical") & (histograms.previous_quality == "all")]
            _gap_cdf(axis, subset, color, style, CONDITION_LABELS[condition])
            axis.set(xscale="symlog", xlabel="Optimizer steps" if metric == "step_gap" else "Scheduling ticks late",
                     ylabel="Fraction of reviews below x", title=title, ylim=(0, 1.02))
            axis.grid(alpha=.2)
        rows = stages[stages.condition == condition]
        axes[1, 0].plot(rows.stage, rows.actual_historical_fraction, color=color, linestyle=style,
                        label=CONDITION_LABELS[condition])
        if condition.startswith("srt"):
            axes[1, 1].plot(rows.stage, rows.historical_due, color=color, linestyle=style, label=CONDITION_LABELS[condition])
    _finish_axis(axes[1, 0], "Historical / all presentations", "Realized historical exposure, including initial coverage")
    _finish_axis(axes[1, 1], "Historical examples due", "Unserved due examples at each stage boundary")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, [textwrap.fill(label, 42) for label in labels], loc="outside lower center", ncol=2, frameon=False, fontsize=10)
    return _save_figure(figure, reports / "replay_timing.png")


def _plot_requested_intervals(reports: Path, histograms: pd.DataFrame) -> Path:
    conditions = tuple(name for name in ALL_STYLES if name.startswith("srt_") and name in set(histograms.condition))
    figure, axes = plt.subplots(len(conditions) // 2, 2, figsize=(9.5, 4 * (len(conditions) // 2)), constrained_layout=True, squeeze=False)
    for axis, condition in zip(axes.flat, conditions, strict=True):
        for metric, label, style in (("interval_before", "Requested interval", "--"), ("clock_gap", "Actual interval", "-")):
            rows = histograms[(histograms.condition == condition) & (histograms.metric == metric)
                              & (histograms.kind == "historical") & (histograms.previous_quality == "all")]
            _gap_cdf(axis, rows, ALL_STYLES[condition][0], style, label)
        axis.set(xscale="log", xlabel="Scheduling ticks", ylabel="Fraction of reviews below x",
                 title=textwrap.fill(CONDITION_LABELS[condition], 43), ylim=(0, 1.02))
        axis.legend(frameon=False, fontsize=10)
        axis.grid(alpha=.2)
    return _save_figure(figure, reports / "requested_and_actual_intervals.png")


def _plot_sample_coverage(reports: Path, samples: pd.DataFrame, stages: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8), constrained_layout=True)
    for condition, (color, style) in ALL_STYLES.items():
        rows = samples[samples.condition == condition]
        if rows.empty:
            continue
        cohorts = rows.groupby("arrival_stage").historical_reviews
        axes[0].plot(cohorts.mean().index, cohorts.mean(), color=color, linestyle=style, label=CONDITION_LABELS[condition])
        counts = rows.historical_reviews.value_counts().sort_index()
        axes[1].step(counts.index, counts.cumsum() / counts.sum(), where="post", color=color, linestyle=style, label=CONDITION_LABELS[condition])
    axes[0].set(xlabel="Task in which image arrived", ylabel="Mean historical reviews per image", title="Replay allocation by arrival cohort")
    axes[1].set(xlabel="Historical reviews received", ylabel="Fraction of training images", title="Per-image historical replay count distribution", xscale="symlog")
    for axis in axes:
        axis.grid(alpha=.2)
    _legend(axes[1])
    return _save_figure(figure, reports / "sample_coverage.png")


def _plot_timelines(reports: Path, timelines: pd.DataFrame, filename: str = "sample_timelines.png") -> Path:
    conditions = tuple(name for name in ALL_STYLES if name.startswith("srt_") and name in set(timelines.condition))
    figure, axes = plt.subplots(len(conditions), 1, figsize=(9.5, 4 * len(conditions)), constrained_layout=True)
    identities = timelines[["image_id", "arrival_stage"]].drop_duplicates().sort_values(["arrival_stage", "image_id"])
    for axis, srt_condition in zip(axes, conditions, strict=True):
        for index, item in enumerate(identities.itertuples(index=False)):
            for method, offset in (("srt", .15), ("uniform", -.15)):
                condition = srt_condition.replace("srt_", f"{method}_", 1)
                events = timelines[(timelines.condition == condition) & (timelines.image_id == item.image_id)]
                color, _ = ALL_STYLES[condition]
                axis.scatter(events.stage_position, np.full(len(events), index + offset), color=color,
                             marker="|" if method == "srt" else ".", s=18, alpha=.65)
        axis.set(yticks=range(len(identities)), yticklabels=[f"Task {item.arrival_stage}: {item.image_id[:8]}" for item in identities.itertuples(index=False)],
                 xlabel="Task arrival plus fraction of that task's optimizer updates", xlim=(0, 50),
                 title=textwrap.fill(CONDITION_LABELS[srt_condition] + ": SRT above / uniform below", 75))
        axis.grid(axis="x", alpha=.2)
    return _save_figure(figure, reports / filename)


def _calibration_rows(run: Path) -> tuple[dict[str, object], ...]:
    selected = read_sealed(run / "calibration/selected.json")["selected"]
    rows = []
    for path in sorted((run / "calibration").glob("*/result.json")):
        result = read_sealed(path)
        screen = result["rows"][:16]
        rows.append({
            "job": path.parent.name, "capacity": result["capacity"], "profile": result["policy"]["profile"],
            "historical_fraction": result["policy"]["historical_fraction"], "interval_unit": result["policy"]["interval_unit"],
            "completed_stages": len(result["rows"]), "selected": selected[str(result["capacity"])]["job"] == path.parent.name,
            "screen_accuracy": math.fsum(row["evaluation"]["accuracy"] for row in screen) / 16,
            "screen_nll": math.fsum(row["evaluation"]["nll"] for row in screen) / 16,
            "full_mean_accuracy": math.fsum(row["evaluation"]["accuracy"] for row in result["rows"]) / 50 if len(result["rows"]) == 50 else None,
            "full_mean_nll": math.fsum(row["evaluation"]["nll"] for row in result["rows"]) / 50 if len(result["rows"]) == 50 else None,
            "presentations": result["image_presentations"], "optimizer_steps": result["optimizer_steps"],
            "training_wall_seconds": math.fsum(row["fit"]["wall_seconds"] for row in result["rows"]),
            "evaluation_wall_seconds": math.fsum(row["evaluation"]["wall_seconds"] for row in result["rows"]),
            "result_hash": result["content_hash"],
        })
    return tuple(rows)


def _task_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    records = []
    for condition, job in result["conditions"].items():
        best = defaultdict(float)
        initial = {}
        for row in job["rows"]:
            for task, (correct, examples) in enumerate(zip(row["evaluation"]["task_correct"], row["evaluation"]["task_examples"], strict=True), 1):
                accuracy = 100 * correct / examples
                forgetting = best[task] - accuracy if task < row["stage"] else None
                initial.setdefault(task, accuracy)
                best[task] = max(best[task], accuracy)
                records.append({"condition": condition, "stage": row["stage"], "task": task, "accuracy": accuracy,
                                "examples": examples, "correct": correct, "forgetting": forgetting,
                                "backward_transfer": accuracy - initial[task] if task < row["stage"] else None})
    return tuple(records)


def _sections(
    run: Path, reports: Path, result: dict[str, object], references: dict[str, object], analyses: dict[str, ReplayAnalysis],
    calibration: tuple[dict[str, object], ...], resources: tuple[dict[str, object], ...], figures: dict[str, Path],
) -> tuple[ReportSection, ...]:
    joint_final = references["stage_matched_joint"][-1]["accuracy"]
    summaries = tuple(analyses[name].totals for name in NEW_STYLES)
    uniform_wins = sum(
        analyses[f"uniform_h{capacity}"].totals["final_accuracy"] > analyses[f"srt_h{capacity}"].totals["final_accuracy"]
        and analyses[f"uniform_h{capacity}"].totals["final_nll"] < analyses[f"srt_h{capacity}"].totals["final_nll"]
        for capacity in (1024, 4096)
    )
    differences = tuple(
        f"At the {capacity:,}-equivalent budget, SRT finishes at {analyses[f'srt_h{capacity}'].totals['final_accuracy']:.3f}% accuracy "
        f"and {analyses[f'srt_h{capacity}'].totals['final_nll']:.4f} NLL. Relative to its exposure-matched uniform control, "
        f"the differences are {analyses[f'srt_h{capacity}'].totals['final_accuracy'] - analyses[f'uniform_h{capacity}'].totals['final_accuracy']:+.3f} "
        f"accuracy points and {analyses[f'srt_h{capacity}'].totals['final_nll'] - analyses[f'uniform_h{capacity}'].totals['final_nll']:+.4f} NLL. "
        f"Its accuracy difference from joint rank 16 is {analyses[f'srt_h{capacity}'].totals['final_accuracy'] - joint_final:+.3f} points."
        for capacity in (1024, 4096)
    )
    table = ReportTable(("Condition", "Final acc.", "Mean acc.", "Final NLL", "Train min."), tuple(
        (CONDITION_LABELS[row["condition"]], f"{row['final_accuracy']:.3f}%", f"{row['mean_stage_accuracy']:.3f}%",
         f"{row['final_nll']:.4f}", f"{row['training_wall_seconds'] / 60:.2f}") for row in summaries))
    selected = tuple(row for row in calibration if row["selected"])
    selected_table = ReportTable(("Budget", "Threshold profile", "Old fraction", "Time unit", "Mean val. acc."), tuple(
        (f"{row['capacity']:,}", row["profile"], f"{row['historical_fraction']:.1f}", str(row["interval_unit"]), f"{row['full_mean_accuracy']:.3f}%")
        for row in selected))
    resource_table = ReportTable(("Condition", "Forward/backward pairs", "Recompute forwards", "Train min."), tuple(
        (CONDITION_LABELS[row["condition"]], f"{row['cumulative_training_backwards']:,}", f"{row['cumulative_recompute_forwards']:,}",
         f"{row['cumulative_training_wall_seconds'] / 60:.2f}") for row in resources if row["stage"] == 50))
    calibration_table = ReportTable(("Budget", "Profile", "Old", "Unit", "Screen acc.", "Full acc."), tuple(
        (f"{row['capacity']:,}", row["profile"] + (" *" if row["selected"] else ""), f"{row['historical_fraction']:.1f}", str(row["interval_unit"]),
         f"{row['screen_accuracy']:.3f}", "-" if row["full_mean_accuracy"] is None else f"{row['full_mean_accuracy']:.3f}") for row in calibration))
    sections = (
        ReportSection("Original validation-selected results", (
            "Does confidence-based spaced repetition improve a single continuing rank-16 adapter compared with uniform replay under identical realized exposure? "
            "The ImageNet-R split is unchanged: 24,000 training images, 6,000 test images, and fifty four-class tasks in the existing seed-1993 order.",
            f"Uniform replay has higher final accuracy and lower final NLL at {uniform_wins} of the two tested budgets. "
            "This comparison tests the selected SRT recipes, not every possible confidence threshold or spacing rule.",
            *differences,
            "Mean accuracy is the arithmetic mean of the fifty stage test accuracies. NLL is uncalibrated, all-seen-class cross-entropy; lower is better. "
            "These are single-seed results, not estimates of training-run variability. Final-run times below exclude calibration.",
        ), table=table),
        ReportSection("Original full-stream comparison at both work budgets", (
            "Each panel compares one SRT/uniform pair with the five existing task-free conditions. Historical curve names, values, and colors are unchanged. "
            "A budget of H-equivalent work means 4 x (current images + min(H, historical images)) presentations per task, not a memory cap. "
            "Both methods can revisit every arrived training image.",
        ), (figures["accuracy"],)),
        ReportSection("Carried-forward accuracy figure", (
            "This is the original full-stream accuracy figure, copied byte-for-byte from the previous report. The pale dotted curves are label-aware "
            "true-node diagnostics for the frontier models, not deployable methods and not SRT baselines. The prior frontier results and report were not changed.",
            "Joint rank 16 uses the same adapter targets, rank, and affine classification architecture as SRT, but retrains on each full prefix. "
            "Aggregate-rank-matched joint IID matches the total rank of the frontier nodes, not the rank of the single SRT adapter.",
        ), (run / "references/previous_stage_accuracy.png",)),
        ReportSection("Probability quality and the joint-IID gap", (
            "The rank-16 joint source did not retain NLL, so no rank-16 NLL curve is fabricated. The lower panel uses its retained accuracy curve. "
            "Neither joint reference is an execution gate or a mathematical upper bound. Each fresh joint model receives five epochs. "
            "The final rank-16 joint model received 120,000 training presentations and 1,875 updates, whereas continuing uniform H=1,024 received "
            "294,368 presentations and 5,253 updates across its lifetime. Earlier joint-prefix models do not warm-start later ones.",
        ), (figures["nll"],)),
        ReportSection("Measured training work", (
            "Each new training presentation produces one ViT forward/backward pair. Confidence comes from that same forward: there are no quality-only "
            "passes, source hierarchy jobs, or activation recomputation. Prior frontier costs include their source hierarchy. The table separates their extra recomputation forwards.",
            "Training time sums measured batch work and committed checkpoint I/O. Loader and scheduler timings are components of that time, not costs to add again. "
            "Setup, model serialization outside those counters, between-job overhead, and evaluation are separate. Calibration is excluded from this comparison.",
        ), table=resource_table),
        ReportSection("Cumulative work across the stream", (
            "The fixed H-equivalent budget gives linear training-image work for bounded task size and this fixed model. Repeated prefix testing is quadratic. "
            "CPU population scans and report reductions are not covered by the linear model-work claim. These counts are image paths, not profiled FLOPs; "
            "equal counts do not imply equal arithmetic for different adapter ranks or multiple frontier models.",
        ), (figures["resources"],)),
        ReportSection("When reviews actually happened", (
            "The distributions are review-event weighted and separate real optimizer gaps from virtual scheduling-clock delays. Scheduling time advances to "
            "the next due item when neither pool is ready; that does not perform a training update. A large due backlog means requested spacing and actual spacing differ.",
            "The historical fraction plotted includes the mandatory first pass over new images. Its realized value can differ from the configured post-introduction "
            "fraction when one due pool is short. Each uniform control copies the exact resulting old/current counts and partial batch sizes.",
        ), (figures["replay"],)),
        ReportSection("Which samples received replay", (
            "Earlier arrival cohorts have more opportunities for historical replay. Compare methods within a cohort rather than interpreting that age effect as "
            "preferential selection. The lower plot gives each image one vote. Task-50 images cannot receive a historical replay within this experiment.",
            "The per-sample table retains exact mean, median, 90th-percentile, minimum, and maximum optimizer gaps, review counts, and terminal waits. "
            "Every unfinished next-review interval is marked censored; images with no observed second presentation are retained rather than silently omitted.",
        ), (figures["coverage"],)),
        ReportSection("Requested spacing versus realized spacing", (
            "These curves use the same historical review events and the same scheduling-clock unit. Requested intervals come from the previous SM-2 update; "
            "actual intervals run from that previous presentation to the next real review. Any rightward displacement is waiting beyond the requested spacing.",
            "CDF values are evaluated at logarithmic bin boundaries, using completed intervals only. Terminal unfinished intervals are retained separately as censored waits, "
            "not interpreted as completed long gaps. The earlier optimizer-gap plot measures actual learning updates rather than virtual clock advances.",
        ), (figures["intervals"],)),
        ReportSection("Individual review histories", (
            "Eight image identities were selected by a fixed hash rule: two each from arrival tasks 1, 16, 31, and 50. Selection does not use accuracy, confidence, "
            "or whether SRT looks successful. SRT marks sit above each sample row and uniform marks below. The full identities and every event are retained in the analysis tables.",
            "The horizontal coordinate is task arrival plus the fraction of that task's optimizer updates, not elapsed wall time. Exact optimizer-step gaps are in the timing distributions and per-sample tables.",
        ), (figures["timelines"],)),
        ReportSection("Calibration and selected settings", (
            "Eighteen settings per budget were screened through task 16 on the existing 19,200/4,800 training-derived fit/validation split. Two finalists per budget "
            "continued through task 50. Selection maximized mean stage validation accuracy, then minimized mean NLL, then used canonical policy order. No test images entered this process.",
            f"Calibration consumed {sum(row['presentations'] for row in calibration):,} training presentations, "
            f"{sum(row['training_wall_seconds'] for row in calibration) / 3600:.2f} measured training hours, and "
            f"{sum(row['evaluation_wall_seconds'] for row in calibration) / 60:.2f} evaluation minutes. The four final models restarted from cold adapters after selection.",
            "Relaxed thresholds are [.05,.15,.30,.60,.85], standard [.10,.25,.50,.75,.90], and strict [.20,.40,.70,.85,.95]. "
            "Quality counts thresholds met by the pre-update probability of the correct class. The interval unit scales the first/failure interval and the first two successful intervals.",
            "These numerical thresholds were hand-chosen search candidates, not values reported by the paper. Validation selected among them; "
            "it did not establish that a quality score predicts retention after a given delay. Calibration costs above exclude the preserved, superseded development run and preflight.",
        ), table=selected_table),
        ReportSection("Complete calibration matrix", (
            "Screen accuracy averages tasks 1-16; full accuracy averages tasks 1-50 and is present only for continued finalists. An asterisk marks the selected setting. "
            "All values are validation percentages, not test results. NLL, resource counters, hashes, and every stage curve are included in the accompanying tables.",
        ), table=calibration_table),
        ReportSection("Protocol, verification, and interpretation limits", (
            "The frozen ViT-B/16 uses rank/alpha-16 LoRA on QKV and fc1 in all twelve blocks. An ordinary affine head adds four rows per task. "
            "Old rows, adapter parameters, and SGD momentum persist; all seen rows remain trainable. SGD uses batch limit 64, momentum 0.9, weight decay 0.0005, "
            "LoRA learning rate 0.0005, and head learning rate 0.01, without clipping or a learning-rate schedule.",
            "Our implementation follows Atreya et al.'s pre-update scoring and SM-2 equations. Their code is proprietary and their classification thresholds are unspecified. "
            "The guaranteed first pass, calibrated thresholds/interval unit, and explicit empty-queue clock rule are documented local choices, not claims of an exact reproduction.",
            "The real smoke test verified zero-LoRA parity, BF16 batch 64, eight task arrivals, exact interrupted/uninterrupted weights and review events, "
            "and matched uniform exposure. Reporting revalidates all committed events and reconstructs accuracy/NLL from stored predictions. A completed rerun performs zero optimizer steps.",
            "The uniform control is conditioned on SRT's realized allocation; it is not an independently tuned uniform-replay optimum. Quality is measured on random training crops, "
            "so a low score can reflect an uninformative crop as well as forgetting. Hyperparameters were selected offline using a training-derived full-stream validation sweep. "
            "One seed cannot establish reproducibility or a publishable state-of-the-art result.",
            f"For the original validation-selected policies, {analyses['srt_h1024'].totals['terminal_overdue_images']:,} of 24,000 training images are overdue at the final boundary at H=1,024, "
            f"and {analyses['srt_h4096'].totals['terminal_overdue_images']:,} at H=4,096. Requested spacing is therefore not reliably delivered. "
            "This does not isolate the cause of the accuracy difference: crop-dependent confidence, sample selection, and review delays can all contribute.",
            "The next tests should replicate the paired SRT/uniform comparison across seeds and, on training-derived validation data, measure how pre-review confidence predicts later retention "
            "at observed delays. Then test spacing calibrated to the available review budget. H changes both work and the selected recipe here, so it is not a pure work ablation. "
            "The original selection stays frozen. The separately identified, user-requested fixed-policy follow-up is exploratory because it was requested after examining these test results.",
            "Source: Atreya et al. (2026), When to Review: Spaced Repetition for Continual Pre-Training of Language Models, arXiv:2608.17530v1. "
            "https://arxiv.org/html/2608.17530v1",
        )),
    )
    if not set(FOLLOWUP_STYLES) <= set(analyses):
        return sections
    followup_table = ReportTable(("Condition", "Final acc.", "Mean acc.", "Final NLL", "Updates", "Train min."), tuple(
        (CONDITION_LABELS[name], f"{analyses[name].totals['final_accuracy']:.3f}%", f"{analyses[name].totals['mean_stage_accuracy']:.3f}%",
         f"{analyses[name].totals['final_nll']:.4f}", f"{analyses[name].totals['optimizer_steps']:,}",
         f"{analyses[name].totals['training_wall_seconds'] / 60:.2f}")
        for name in FOLLOWUP_STYLES))
    followup_differences = tuple(
        f"With {profile} thresholds, SRT minus its matched uniform control is "
        f"{analyses[f'srt_h4096_{profile}_rho80_unit8'].totals['final_accuracy'] - analyses[f'uniform_h4096_{profile}_rho80_unit8'].totals['final_accuracy']:+.3f} "
        "final accuracy points and "
        f"{analyses[f'srt_h4096_{profile}_rho80_unit8'].totals['final_nll'] - analyses[f'uniform_h4096_{profile}_rho80_unit8'].totals['final_nll']:+.4f} NLL."
        for profile in ("standard", "strict"))
    budget_differences = tuple(
        f"For {'SRT' if method == 'srt' else 'uniform replay'} with standard thresholds, old target 0.8, and unit 8 fixed, "
        "raising H from 1,024 to 4,096 changes final accuracy by "
        f"{analyses[f'{method}_h4096_standard_rho80_unit8'].totals['final_accuracy'] - analyses[f'{method}_h1024'].totals['final_accuracy']:+.3f} "
        "points and NLL by "
        f"{analyses[f'{method}_h4096_standard_rho80_unit8'].totals['final_nll'] - analyses[f'{method}_h1024'].totals['final_nll']:+.4f}."
        for method in ("srt", "uniform"))
    extra = (
        ReportSection("Fixed-policy follow-up: H=4,096, old=0.8, unit=8", (
            "Four additional cold-start 50-task streams hold H=4,096, historical target 0.8, and interval unit 8 fixed. "
            "Standard and strict thresholds each have an exposure-matched uniform control. Every stream uses 844,640 training presentations. "
            "The optimizer, model, data split, seeds, and checkpoint timing are unchanged; no calibration jobs are repeated.",
            "Standard thresholds are [.10,.25,.50,.75,.90]; strict thresholds are [.20,.40,.70,.85,.95]. "
            "The 0.8 target applies after the mandatory current-task first pass. Actual shares can differ when a due pool is short. "
            "A uniform condition's profile name identifies the SRT batch schedule it copies, not a quality rule applied to uniform sampling.",
            "Batch size 64 is a maximum. When few examples are due, partial batches consume more optimizer updates for the same image budget. "
            "The table reports these updates separately; each uniform control matches them exactly.",
            *followup_differences,
            *budget_differences,
            f"The new standard SRT run ends with {analyses['srt_h4096_standard_rho80_unit8'].totals['terminal_overdue_images']:,} overdue images, "
            f"and strict with {analyses['srt_h4096_strict_rho80_unit8'].totals['terminal_overdue_images']:,}, out of 24,000. "
            "A due backlog measures failure to meet a requested schedule, not whether that schedule maximizes accuracy.",
            "These settings were requested after reviewing the original test curves. Treat them as exploratory single-seed ablations, not an independently selected final winner. "
            "Original result hashes and the validation selection remain unchanged.",
        ), table=followup_table),
        ReportSection("Full-stream accuracy with the fixed-policy conditions", (
            "All four new conditions use a single rank-16 adapter. Colors distinguish standard and strict profiles; solid lines denote SRT and dashed lines their matched uniform controls. "
            "All five earlier task-free frontier and joint-IID conditions remain overlaid. The original selected-policy SRT curves remain on the preceding accuracy page.",
        ), (figures["followup_accuracy"],)),
        ReportSection("Direct comparisons with the original SRT recipes", (
            "Left: standard thresholds, old target 0.8, and unit 8 are fixed; only H changes from 1,024 to 4,096. "
            "Right: H=4,096 and strict thresholds are fixed; old target/unit change together from 0.5/1 to 0.8/8. "
            "The right comparison does not isolate mixture from spacing. Each uniform line copies its associated SRT batch schedule. "
            "The lower row shows NLL for the identical conditions; the joint rank-16 source has no retained NLL.",
        ), (figures["policy_comparisons"],)),
        ReportSection("Equal image work does not mean equal optimizer work", (
            "Each line represents an SRT recipe and its exact-count uniform partner. H=4,096 fixes 844,640 image presentations, "
            "but the due-only scheduler can produce batches smaller than 64. Cross-entropy is averaged within each actual batch; "
            "each batch then takes one full-learning-rate SGD update, with momentum and weight decay. More small batches therefore "
            "change optimization as well as overhead. No gradient accumulation is applied. The interval unit measures virtual scheduling ticks, "
            "not a fixed amount of intervening image work; empty-queue clock advances and due backlogs also separate requested from actual replay gaps.",
        ), (figures["optimizer_work"],)),
        ReportSection("What correct-class probability does and does not measure", (
            "The quality signal is p(correct class), equivalently exp(-cross-entropy), measured before the update on the sampled training crop. "
            "Appendix C.2 of Atreya et al. proposes this signal for classifiers. It is cheap and follows the training loss, but is not a measured probability that an example will be forgotten, "
            "nor a measure of the expected accuracy gain from replaying it now.",
            "Two 200-class predictions can both assign 0.50 to the correct class while giving their strongest competitor 0.01 or 0.49. "
            "Both are currently correct and have the same 0.693 cross-entropy, but probability margins of 0.49 versus 0.01. The present scheduler assigns them identical quality. "
            "At exactly 0.50, standard assigns quality 3 (success), while strict assigns quality 2 (failure, resetting the interval to 8 ticks).",
            "Raw confidence also changes under positive temperature rescaling without changing the winning class. New classifier rows can change the softmax denominator, "
            "and a random crop covering as little as 5% of an image can omit the object. Low confidence is therefore not synonymous with forgotten knowledge. "
            "Conversely, a wide margin on today's crop does not prove that later tasks cannot erase the decision.",
            "Our threshold values are hand-chosen candidates. Validation selected recipes by average accuracy, but did not calibrate quality to future retention or replay benefit. "
            "Prioritizing low-confidence examples also changes their effective training weights and can overemphasize ambiguous crops. "
            "Following per-image loss does not establish that this allocation maximizes held-out accuracy. "
            "A useful next study would compare correctness, correct-versus-runner-up margin, and confidence against later prediction failures at measured delays, using training-derived probes. "
            "Then compare a retention-based scheduler under matched work. Margin is a hypothesis to test, not an established replacement.",
            "Source: Atreya et al., https://arxiv.org/html/2608.17530v1#A3.SS2",
        )),
        ReportSection("Individual review histories for the fixed-policy follow-up", (
            "The same eight hash-selected training images appear in both profile panels. SRT marks sit above their paired uniform marks. "
            "Neither the identities nor their inclusion was selected using these outcomes. Timing is task arrival plus the within-task update fraction.",
        ), (figures["followup_timelines"],)),
    )
    return sections[:2] + extra[:4] + sections[2:] + extra[4:]


def render_report(sections: tuple[ReportSection, ...], reports: Path, pdf: Path) -> None:
    """Render shared content to a paper-style PDF and self-contained HTML/Markdown."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("SRTBody", parent=styles["BodyText"], fontSize=10.5, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle("SRTCell", parent=styles["BodyText"], fontSize=8.2, leading=10))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=38, leftMargin=38,
                                 topMargin=38, bottomMargin=38, title=TITLE, author="ImageNet-R experiment", invariant=1)
    story, markdown, html = [], [f"# {TITLE}\n"], [f"<h1>{escape(TITLE)}</h1>"]
    for number, section in enumerate(sections):
        if number:
            story.append(PageBreak())
        section_start = len(story)
        if not number:
            story.extend((Paragraph(TITLE, styles["Title"]), Spacer(1, 12)))
        story.append(Paragraph(section.title, styles["Heading2"]))
        markdown.append(f"\n## {section.title}\n")
        html.append(f"<section><h2>{escape(section.title)}</h2>")
        for paragraph in section.paragraphs:
            story.append(Paragraph(escape(paragraph), styles["SRTBody"]))
            markdown.append(f"\n{paragraph}\n")
            html.append(f"<p>{escape(paragraph)}</p>")
        if section.table is not None:
            rows = (section.table.columns, *section.table.rows)
            cells = [[Paragraph(escape(str(value)), styles["SRTCell"]) for value in row] for row in rows]
            first_width = 205 if section.table.columns[0] == "Condition" else 70
            widths = [first_width] + [(document.width - first_width) / (len(rows[0]) - 1)] * (len(rows[0]) - 1)
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            padding = 3 if len(rows) > 25 else 5
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef2")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#f7f9fb"))),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding), ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#aab6bf")),
            ]))
            story.append(table)
            markdown += ["\n| " + " | ".join(rows[0]) + " |\n", "| " + " | ".join("---" for _ in rows[0]) + " |\n"]
            markdown += ["| " + " | ".join(row) + " |\n" for row in rows[1:]]
            html.append("<table>" + "".join("<tr>" + "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in row) + "</tr>"
                                               for row, tag in ((rows[0], "th"), *((row, "td") for row in rows[1:]))) + "</table>")
        for figure_index, figure in enumerate(section.figures):
            with PILImage.open(figure) as image:
                width, height = image.size
            occupied = sum(item.wrap(document.width, document.height)[1] + item.getSpaceBefore() + item.getSpaceAfter()
                           for item in story[section_start:])
            available = (document.height - 12 - occupied) / (len(section.figures) - figure_index)
            if available <= 0:
                raise ValueError(f"report section has no room for its figure: {section.title}")
            scale = min(document.width / width, available / height)
            story.append(Image(str(figure), width=width * scale, height=height * scale))
            relative = figure.relative_to(reports) if figure.is_relative_to(reports) else Path("../references") / figure.name
            markdown.append(f"\n![{section.title}]({relative.as_posix()})\n")
            html.append(f'<img alt="{escape(section.title)}" src="data:image/png;base64,{base64.b64encode(figure.read_bytes()).decode()}">')
        html.append("</section>")
    def footer(canvas, _document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#586674"))
        canvas.drawString(38, 22, "ImageNet-R-50 | Single-adapter SRT | Same-split comparisons")
        canvas.drawRightString(letter[0] - 38, 22, str(canvas.getPageNumber()))
        canvas.restoreState()
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    atomic_write(pdf, buffer.getvalue())
    atomic_write(reports / "REPORT.md", "".join(markdown).encode())
    style = "body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#202a33}img{max-width:100%}section{margin:48px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce2e7;text-align:left}th{background:#e8eef2}tr:nth-child(even){background:#f7f9fb}"
    atomic_write(reports / "REPORT.html", (f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(TITLE)}</title><style>{style}</style></head><body>"
                                                + "".join(html) + "</body></html>").encode())


def followup_report_jobs(run: Path, source_hash: str) -> tuple[dict[str, Path], dict[str, object] | None]:
    """Authenticate a completed fixed-policy extension without changing its source."""
    pointer_path = run / "reports/fixed_policy_followup.json"
    if not pointer_path.is_file():
        return {}, None
    pointer = read_sealed(pointer_path, "imagenetr50-srt-followup-pointer-v1")
    root = run / "followups" / pointer["run_hash"]
    protocol = read_sealed(root / "protocol.json", "imagenetr50-srt-followup-protocol-v1")
    result = read_sealed(root / "result.json", "imagenetr50-srt-followup-result-v1")
    if (pointer["source_result_hash"] != source_hash or result["source_result_hash"] != source_hash
            or protocol["source_result_hash"] != source_hash or pointer["result_hash"] != result["content_hash"]
            or protocol["content_hash"] != pointer["run_hash"] or result["protocol_hash"] != protocol["content_hash"]
            or not result["zero_step_reuse"] or not result["source_unchanged"]
            or set(result["conditions"]) != set(FOLLOWUP_STYLES) or set(protocol["conditions"]) != set(FOLLOWUP_STYLES)):
        raise ValueError("fixed-policy follow-up identity or source changed")
    profiles = dict(read_sealed(run / "config_resolved.json")["profiles"])
    roots = {name: root / "final" / name for name in FOLLOWUP_STYLES}
    for name, job_root in roots.items():
        job = read_sealed(job_root / "result.json", "imagenetr50-srt-job-result-v1")
        definition = read_sealed(job_root / "job.json", "imagenetr50-srt-job-v1")
        profile = "standard" if "_standard_" in name else "strict"
        method = name.split("_", 1)[0]
        expected = {"profile": profile, "thresholds": profiles[profile], "historical_fraction": .8, "interval_unit": 8}
        reuse = result["reuse"][name]
        if (job != result["conditions"][name] or job["policy"] != expected or definition["policy"] != expected
                or definition["protocol_hash"] != protocol["content_hash"] or job["job_hash"] != definition["content_hash"]
                or job["method"] != method or definition["method"] != method or job["capacity"] != 4096
                or definition["capacity"] != 4096 or len(job["rows"]) != 50 or job["image_presentations"] != 844640
                or reuse["optimizer_steps"] != 0 or reuse["result_hash"] != job["content_hash"]):
            raise ValueError("fixed-policy job differs from its explicit requested recipe")
        if method == "uniform":
            paired = read_sealed(roots[name.replace("uniform_", "srt_", 1)] / "job.json")
            if definition["paired_job_hash"] != paired["content_hash"]:
                raise ValueError("fixed-policy uniform control uses the wrong SRT partner")
    return roots, {"pointer": pointer, "protocol": protocol, "result_hash": result["content_hash"]}


def write_srt_report(run: Path) -> Path:
    """Audit primary evidence and build only the separate SRT report artifacts."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10})
    result = read_sealed(run / "result.json", "imagenetr50-srt-result-v1")
    if set(result["conditions"]) != set(NEW_STYLES) or not result["zero_step_reuse"] or not result["references_unchanged"]:
        raise ValueError("SRT report requires the complete verified four-condition result")
    protocol = read_sealed(run / "protocol.json", "imagenetr50-srt-protocol-v1")
    selection = read_sealed(run / "calibration/selected.json", "imagenetr50-srt-selection-v1")
    if result["protocol_hash"] != protocol["content_hash"] or result["selection_hash"] != selection["content_hash"] or selection["test_used"]:
        raise ValueError("SRT report protocol or validation-only selection changed")
    references = {**reference_results(run), "joint_convergence": load_joint_reference(run, result["content_hash"])}
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    extra_roots, followup = followup_report_jobs(run, result["content_hash"])
    job_roots = {**{name: run / "final" / name for name in NEW_STYLES}, **extra_roots}
    jobs = {name: read_sealed(root / "result.json") for name, root in job_roots.items()}
    analyses = {}
    for condition, job_root in job_roots.items():
        print(f"Report audit: {CONDITION_LABELS[condition]}", flush=True)
        job = jobs[condition]
        if condition in NEW_STYLES:
            if job != result["conditions"][condition]:
                raise ValueError("SRT report job differs from the sealed final matrix")
            definition = read_sealed(job_root / "job.json", "imagenetr50-srt-job-v1")
            if (definition["protocol_hash"] != protocol["content_hash"] or definition["policy"] != selection["selected"][str(job["capacity"])]["policy"]
                    or definition["method"] != job["method"] or definition["capacity"] != job["capacity"]):
                raise ValueError("SRT final model does not use its frozen selected recipe")
        analyses[condition] = analyze_replay_job(job_root, run / "final/image_index.parquet", condition)
    for srt_name in (name for name in jobs if name.startswith("srt_")):
        paired_names = (srt_name, srt_name.replace("srt_", "uniform_", 1))
        first, second = (analyses[name] for name in paired_names)
        for old, new in zip(first.stages, second.stages, strict=True):
            if any(old[name] != new[name] for name in ("introductions", "current_reviews", "historical_reviews", "optimizer_steps", "training_forwards")):
                raise ValueError("SRT and uniform exposure totals no longer match")
        for stage in range(50):
            paired_batches = tuple(
                trace_batches(job_roots[name], jobs[name]["rows"][stage]["trace_chunks"])
                for name in paired_names
            )
            for old, new in zip(*paired_batches, strict=True):
                if any(old[name] != new[name] for name in ("stage", "step", "introduction", "historical_count", "current_count", "presentations")):
                    raise ValueError("SRT and uniform per-update exposure no longer matches")
    stages = tuple(row for analysis in analyses.values() for row in analysis.stages)
    samples = tuple(row for analysis in analyses.values() for row in analysis.samples)
    histograms = tuple(row for analysis in analyses.values() for row in analysis.histograms)
    timelines = tuple(row for analysis in analyses.values() for row in analysis.timelines)
    summaries = tuple(analysis.totals for analysis in analyses.values())
    calibration, resources = _calibration_rows(run), _resource_rows(run, stages)
    joint_sections, joint_figures, joint_tables = joint_report_parts(reports, references["joint_convergence"], summaries)
    tables = {"stage_metrics": stages, "sample_replay": samples, "replay_histograms": histograms,
              "sample_timelines": timelines, "condition_summary": summaries, "calibration": calibration,
              "resource_metrics": resources, "task_metrics": _task_rows({"conditions": jobs}),
              "condition_names": tuple({"condition": name, "label": label,
                                         "profile": jobs[name]["policy"]["profile"] if name in jobs else None,
                                         "historical_fraction": jobs[name]["policy"]["historical_fraction"] if name in jobs else None,
                                         "interval_unit": jobs[name]["policy"]["interval_unit"] if name in jobs else None}
                                        for name, label in CONDITION_LABELS.items() if name not in FOLLOWUP_STYLES or name in jobs),
              **joint_tables}
    tables["condition_names"] += tuple({"condition": row["condition"], "label": row["label"], "profile": None,
                                         "historical_fraction": None, "interval_unit": None}
                                        for row in joint_tables.get("joint_convergence_summary", ()))
    for name, rows in tables.items():
        _write_table(reports, name, rows)
    frame = pd.DataFrame(stages)
    figures = {
        "accuracy": plot_accuracy(reports, references, frame), "nll": _plot_nll_and_gaps(reports, references, frame),
        "resources": _plot_resources(reports, pd.DataFrame(resources)),
        "replay": _plot_replay(reports, pd.DataFrame(histograms), frame),
        "intervals": _plot_requested_intervals(reports, pd.DataFrame(histograms)),
        "coverage": _plot_sample_coverage(reports, pd.DataFrame(samples), frame),
        "timelines": _plot_timelines(reports, pd.DataFrame(tuple(row for row in timelines if row["condition"] in NEW_STYLES))),
        **joint_figures,
    }
    if followup is not None:
        figures = {**figures, "followup_accuracy": plot_followup_accuracy(reports, references, frame),
                   "policy_comparisons": plot_policy_comparisons(reports, references, frame),
                   "optimizer_work": plot_optimizer_work(reports, frame),
                   "followup_timelines": _plot_timelines(reports, pd.DataFrame(tuple(row for row in timelines if row["condition"] in FOLLOWUP_STYLES)),
                                                          "fixed_policy_sample_timelines.png")}
    sections = _sections(run, reports, result, references, analyses, calibration, resources, figures) + joint_sections
    project = run.parents[4]
    pdf = project / "output/pdf/imagenetr50_srt_r16_report.pdf"
    render_report(sections, reports, pdf)
    material = {path.name: file_sha256(path) for path in Path(__file__).parent.glob("srt_*report*.py")}
    material["srt_analysis.py"] = file_sha256(Path(__file__).with_name("srt_analysis.py"))
    material["joint_convergence_reporting.py"] = file_sha256(Path(__file__).with_name("joint_convergence_reporting.py"))
    manifest = sealed_record({
        "schema_version": "imagenetr50-srt-report-v1", "result_hash": result["content_hash"],
        "fixed_policy_followup": followup,
        "joint_convergence": None if references["joint_convergence"] is None else {
            "pointer": references["joint_convergence"]["pointer"], "protocol": references["joint_convergence"]["protocol"],
            "result_hash": references["joint_convergence"]["result"]["content_hash"]},
        "report_code": material, "pdf": str(pdf), "pdf_sha256": file_sha256(pdf),
        "reference_protocol_hash": result["protocol_hash"],
        "condition_names": {row["condition"]: row["label"] for row in tables["condition_names"]},
        "tables": {name: {suffix: file_sha256(reports / f"{name}.{suffix}") for suffix in ("json", "csv", "parquet")} for name in tables},
        "figures": {name: file_sha256(path) for name, path in figures.items()},
        "html_sha256": file_sha256(reports / "REPORT.html"), "markdown_sha256": file_sha256(reports / "REPORT.md"),
        "event_audit_passed": True, "paired_exposure_audit_passed": True,
        "clock_column_encoding": "Exact decimal strings in JSON, CSV, and Parquet; convert to arbitrary-width integers for arithmetic",
    })
    atomic_write(reports / "report_manifest.json", canonical_json_bytes(manifest))
    for command in (("xdg-open", str(pdf)),
                    ("notify-send", "ImageNet-R SRT experiment finished", "The separate SRT report and replay-analysis tables are ready.")):
        if shutil.which(command[0]):
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return pdf
