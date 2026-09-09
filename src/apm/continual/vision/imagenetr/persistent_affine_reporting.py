"""Publication-style reporting for the full-series persistent affine experiment."""

from __future__ import annotations

import base64
import csv
from html import escape
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Final

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.persistent_affine_resources import (
    load_source_training_costs,
    resource_stage_rows,
)
from apm.continual.vision.imagenetr.persistent_mlp_reporting import (
    LABEL_MLP, LABEL_ORACLE_MLP, MLP_CONDITION_ID,
    load_mlp_extension, mlp_detail_cells, mlp_explanation, mlp_resource_rows,
)


LABEL_H4096: Final[str] = "Persistent single-affine + adaptive node LoRAs, H=4,096"
LABEL_H8192: Final[str] = "Persistent single-affine + adaptive node LoRAs, H=8,192"
LABEL_STAGE_JOINT: Final[str] = "Stage-matched joint IID, rank 16"
LABEL_RANK_JOINT: Final[str] = "Aggregate-rank-matched joint IID"
LABEL_ORACLE_H4096: Final[str] = "True-node oracle for persistent H=4,096 (diagnostic)"
LABEL_ORACLE_H8192: Final[str] = "True-node oracle for persistent H=8,192 (diagnostic)"
REPORT_TITLE = "ImageNet-R-50 persistent frontier integrators"
RESOURCE_LABELS = {
    "persistent_h4096": LABEL_H4096,
    "joint_rank16": LABEL_STAGE_JOINT,
    "joint_rank_matched": LABEL_RANK_JOINT,
    MLP_CONDITION_ID: LABEL_MLP,
}


def _validate_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-persistent-affine-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or set(result.get("arms", {})) != {"4096", "8192"}
        or any(len(rows) != 50 for rows in result["arms"].values())
        or len(result.get("stage_matched_joint", ())) != 50
        or len(result.get("rank_matched_joint", ())) != 50
    ):
        raise ValueError("persistent affine result is incomplete or unauthenticated")
    mlp = load_mlp_extension(run, result)
    return result if mlp is None else {**result, "mlp": mlp}


def _series(result: dict[str, object]) -> dict[str, list[float]]:
    arms = result["arms"]
    return {
        LABEL_H4096: [float(row["evaluation"]["accuracy"]) for row in arms["4096"]],
        LABEL_H8192: [float(row["evaluation"]["accuracy"]) for row in arms["8192"]],
        LABEL_STAGE_JOINT: [float(row["accuracy"]) for row in result["stage_matched_joint"]],
        LABEL_RANK_JOINT: [float(row["accuracy"]) for row in result["rank_matched_joint"]],
        LABEL_ORACLE_H4096: [
            float(row["evaluation"]["true_node_oracle_accuracy"])
            for row in arms["4096"]
        ],
        LABEL_ORACLE_H8192: [
            float(row["evaluation"]["true_node_oracle_accuracy"])
            for row in arms["8192"]
        ],
        **({
            LABEL_MLP: [float(row["evaluation"]["accuracy"]) for row in result["mlp"]["rows"]],
            LABEL_ORACLE_MLP: [float(row["evaluation"]["true_node_oracle_accuracy"]) for row in result["mlp"]["rows"]],
        } if "mlp" in result else {}),
    }


def _summary_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    series = _series(result)
    rows = []
    for label, values in series.items():
        final = values[-1]
        rows.append(
            {
                "condition": label,
                "final_accuracy": final,
                "incremental_accuracy": math.fsum(values) / len(values),
                "fragmented_stage_accuracy": math.fsum(
                    value for stage, value in enumerate(values, 1) if stage.bit_count() > 1
                )
                / sum(stage.bit_count() > 1 for stage in range(1, 51)),
                "one_node_stage_accuracy": math.fsum(
                    value for stage, value in enumerate(values, 1) if stage.bit_count() == 1
                )
                / sum(stage.bit_count() == 1 for stage in range(1, 51)),
            }
        )
    return tuple(rows)


def _stage_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    arms = result["arms"]
    stage_joint = result["stage_matched_joint"]
    rank_joint = result["rank_matched_joint"]
    rows = []
    for index in range(50):
        stage = index + 1
        h4, h8 = arms["4096"][index], arms["8192"][index]
        mlp = result["mlp"]["rows"][index]["evaluation"] if "mlp" in result else None
        rows.append(
            {
                "stage": stage,
                "live_nodes": stage.bit_count(),
                "rank_matched_joint_rank": int(rank_joint[index]["rank"]),
                "persistent_affine_h4096_accuracy": float(h4["evaluation"]["accuracy"]),
                "persistent_affine_h4096_nll": float(h4["evaluation"]["nll"]),
                "persistent_affine_h8192_accuracy": float(h8["evaluation"]["accuracy"]),
                "persistent_affine_h8192_nll": float(h8["evaluation"]["nll"]),
                "stage_matched_joint_rank16_accuracy": float(stage_joint[index]["accuracy"]),
                "rank_matched_joint_accuracy": float(rank_joint[index]["accuracy"]),
                "rank_matched_joint_nll": rank_joint[index]["nll"],
                "h4096_gap_to_stage_joint": float(h4["evaluation"]["accuracy"]) - float(stage_joint[index]["accuracy"]),
                "h8192_gap_to_stage_joint": float(h8["evaluation"]["accuracy"]) - float(stage_joint[index]["accuracy"]),
                "h4096_gap_to_rank_joint": float(h4["evaluation"]["accuracy"]) - float(rank_joint[index]["accuracy"]),
                "h8192_gap_to_rank_joint": float(h8["evaluation"]["accuracy"]) - float(rank_joint[index]["accuracy"]),
                "h4096_true_node_oracle_accuracy": float(h4["evaluation"]["true_node_oracle_accuracy"]),
                "h8192_true_node_oracle_accuracy": float(h8["evaluation"]["true_node_oracle_accuracy"]),
                **({
                    "persistent_mlp_h4096_accuracy": float(mlp["accuracy"]),
                    "persistent_mlp_h4096_nll": float(mlp["nll"]),
                    "mlp_h4096_true_node_oracle_accuracy": float(mlp["true_node_oracle_accuracy"]),
                    "mlp_h4096_gap_to_stage_joint": float(mlp["accuracy"]) - float(stage_joint[index]["accuracy"]),
                    "mlp_h4096_gap_to_rank_joint": float(mlp["accuracy"]) - float(rank_joint[index]["accuracy"]),
                } if mlp is not None else {}),
            }
        )
    return tuple(rows)


def _lifecycle_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Project exact node carry/reset evidence into a compact plotting ledger."""
    h4096 = result["arms"]["4096"]
    h8192 = result["arms"]["8192"]
    rows: list[dict[str, object]] = []
    for stage, (h4, h8) in enumerate(zip(h4096, h8192, strict=True), 1):
        topology_fields = ("node_hashes", "slots", "carry")
        if any(h4[field] != h8[field] for field in topology_fields):
            raise ValueError("persistent arms disagree about the source hierarchy")
        if "mlp" in result and any(h4[field] != result["mlp"]["rows"][stage - 1][field] for field in topology_fields):
            raise ValueError("MLP source-node lifetimes differ from the affine arms")
        node_hashes = tuple(str(value) for value in h4["node_hashes"])
        slots = tuple(int(value) for value in h4["slots"])
        continued = frozenset(str(value) for value in h4["carry"]["continued_node_hashes"])
        reset = frozenset(str(value) for value in h4["carry"]["reset_node_hashes"])
        if (
            len(node_hashes) != len(slots)
            or set(node_hashes) != continued | reset
            or continued & reset
        ):
            raise ValueError("persistent node transition evidence is malformed")
        rows.extend(
            {
                "stage": stage,
                "slot": slot,
                "represented_task_capacity": 2**slot,
                "node_hash": node_hash,
                "entry": "continued" if node_hash in continued else "source_reset",
            }
            for node_hash, slot in zip(node_hashes, slots, strict=True)
        )
    return tuple(rows)


def _fragmentation_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Aggregate task-free and oracle accuracy by the number of live nodes."""
    arms = result["arms"]
    rows: list[dict[str, object]] = []
    conditions = ((4096, "affine", arms["4096"]), (8192, "affine", arms["8192"]))
    if "mlp" in result:
        conditions += ((4096, "mlp", result["mlp"]["rows"]),)
    for capacity, kind, condition_rows in conditions:
        for live_nodes in range(1, 6):
            selected = tuple(
                row
                for row in condition_rows
                if int(row["live_nodes"]) == live_nodes
            )
            if not selected:
                raise ValueError("one expected LogT frontier width is absent")
            accuracies = tuple(float(row["evaluation"]["accuracy"]) for row in selected)
            oracles = tuple(
                float(row["evaluation"]["true_node_oracle_accuracy"])
                for row in selected
            )
            rows.append(
                {
                    "historical_capacity": int(capacity),
                    "integrator_kind": kind,
                    "live_nodes": live_nodes,
                    "stages": len(selected),
                    "mean_accuracy": math.fsum(accuracies) / len(selected),
                    "mean_true_node_oracle_accuracy": math.fsum(oracles) / len(selected),
                    "mean_true_node_oracle_gap": math.fsum(
                        oracle - accuracy
                        for oracle, accuracy in zip(oracles, accuracies, strict=True)
                    )
                    / len(selected),
                }
            )
    return tuple(rows)


def _write_tables(
    reports: Path,
    stage_rows: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    lifecycle_rows: tuple[dict[str, object], ...],
    fragmentation_rows: tuple[dict[str, object], ...],
    resource_rows: tuple[dict[str, object], ...],
) -> None:
    for name, rows in (
        ("stage_metrics", stage_rows),
        ("condition_summary", summaries),
        ("adapter_lifecycle", lifecycle_rows),
        ("fragmentation_summary", fragmentation_rows),
        ("resource_metrics", resource_rows),
    ):
        json_path = reports / f"{name}.json"
        atomic_write(json_path, canonical_json_bytes(list(rows)))
        buffer = StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(reports / f"{name}.csv", buffer.getvalue().encode("utf-8"))
        try:
            import pandas as pd

            temporary = reports / f".{name}.parquet.tmp"
            pd.DataFrame(rows).to_parquet(temporary, index=False)
            os.replace(temporary, reports / f"{name}.parquet")
        except ImportError:  # pragma: no cover - environment gate catches this in runs
            pass


def _plot_accuracy(reports: Path, result: dict[str, object]) -> Path:
    import matplotlib.pyplot as plt

    stages = list(range(1, 51))
    series = _series(result)
    colors = {
        LABEL_H4096: "#1f77b4",
        LABEL_H8192: "#d95f02",
        LABEL_STAGE_JOINT: "#222222",
        LABEL_RANK_JOINT: "#2ca02c",
        LABEL_ORACLE_H4096: "#6baed6",
        LABEL_ORACLE_H8192: "#fdae6b",
        LABEL_MLP: "#9467bd",
        LABEL_ORACLE_MLP: "#c5b0d5",
    }
    styles = {
        LABEL_H4096: "-",
        LABEL_H8192: "-",
        LABEL_STAGE_JOINT: "--",
        LABEL_RANK_JOINT: "-.",
        LABEL_ORACLE_H4096: ":",
        LABEL_ORACLE_H8192: ":",
        LABEL_MLP: "-",
        LABEL_ORACLE_MLP: ":",
    }
    figure, axis = plt.subplots(figsize=(11.2, 6.4), constrained_layout=True)
    for label, values in series.items():
        axis.plot(
            stages,
            values,
            label=label,
            color=colors[label],
            linestyle=styles[label],
            linewidth=1.7 if "oracle" in label.lower() else 2.2,
        )
    for stage in (2, 4, 8, 16, 32):
        axis.axvline(stage, color="#aaaaaa", linewidth=0.7, alpha=0.45)
    axis.set(xlabel="Tasks observed", ylabel="Test top-1 accuracy (%)", xlim=(1, 50))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=10.5,
        frameon=False,
        ncol=2,
    )
    axis.set_title("Full-stream accuracy with stage-matched data prefixes")
    path = reports / "stage_accuracy.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_gaps(reports: Path, stage_rows: tuple[dict[str, object], ...]) -> Path:
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in stage_rows]
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), sharex=True, constrained_layout=True)
    conditions = (
        ("Affine H=4,096 minus joint IID", "h4096", "#1f77b4"),
        ("Affine H=8,192 minus joint IID", "h8192", "#d95f02"),
    )
    if "persistent_mlp_h4096_accuracy" in stage_rows[0]:
        conditions += (("MLP H=4,096 minus joint IID", "mlp_h4096", "#9467bd"),)
    for label, prefix, color in conditions:
        axes[0].plot(stages, [float(row[f"{prefix}_gap_to_stage_joint"]) for row in stage_rows], label=label, color=color, linewidth=2)
        axes[1].plot(stages, [float(row[f"{prefix}_gap_to_rank_joint"]) for row in stage_rows], label=label, color=color, linewidth=2)
    for axis, title in zip(
        axes,
        ("Gap to stage-matched joint IID, rank 16", "Gap to aggregate-rank-matched joint IID"),
        strict=True,
    ):
        axis.axhline(0, color="#222222", linewidth=1)
        axis.set_ylabel("Accuracy gap (points)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=10.5)
    axes[-1].set_xlabel("Tasks observed")
    path = reports / "joint_iid_gaps.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_nll(reports: Path, result: dict[str, object]) -> Path:
    import matplotlib.pyplot as plt

    stages = list(range(1, 51))
    arms = result["arms"]
    rank = result["rank_matched_joint"]
    figure, axis = plt.subplots(figsize=(11.2, 5.3), constrained_layout=True)
    axis.plot(stages, [float(row["evaluation"]["nll"]) for row in arms["4096"]], label=LABEL_H4096, color="#1f77b4", linewidth=2)
    axis.plot(stages, [float(row["evaluation"]["nll"]) for row in arms["8192"]], label=LABEL_H8192, color="#d95f02", linewidth=2)
    if "mlp" in result:
        axis.plot(stages, [float(row["evaluation"]["nll"]) for row in result["mlp"]["rows"]], label=LABEL_MLP, color="#9467bd", linewidth=2)
    rank_stages = [stage for stage, row in zip(stages, rank, strict=True) if row["nll"] is not None]
    rank_values = [float(row["nll"]) for row in rank if row["nll"] is not None]
    axis.plot(rank_stages, rank_values, label=LABEL_RANK_JOINT, color="#2ca02c", linestyle="-.", linewidth=2)
    axis.set(xlabel="Tasks observed", ylabel="Test negative log likelihood", xlim=(1, 50))
    axis.set_title("Test NLL (rank-16 source artifacts did not retain logits)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=10.5)
    path = reports / "stage_nll.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_lifecycle(
    reports: Path, lifecycle_rows: tuple[dict[str, object], ...]
) -> Path:
    """Render exact source-entry and adapted-state carry over all 50 stages."""
    import matplotlib.pyplot as plt

    by_node: dict[str, list[dict[str, object]]] = {}
    for row in lifecycle_rows:
        by_node.setdefault(str(row["node_hash"]), []).append(row)
    figure, axis = plt.subplots(figsize=(11.2, 4.9), constrained_layout=True)
    for node_rows in by_node.values():
        stages = [int(row["stage"]) for row in node_rows]
        slots = {int(row["slot"]) for row in node_rows}
        if len(slots) != 1 or stages != list(range(stages[0], stages[-1] + 1)):
            raise ValueError("one source node has a discontinuous online lifetime")
        axis.plot(
            stages,
            [next(iter(slots))] * len(stages),
            color="#1f77b4",
            linewidth=3.2,
            alpha=0.55,
            solid_capstyle="round",
        )
    reset = tuple(row for row in lifecycle_rows if row["entry"] == "source_reset")
    continued = tuple(row for row in lifecycle_rows if row["entry"] == "continued")
    axis.scatter(
        [int(row["stage"]) for row in continued],
        [int(row["slot"]) for row in continued],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.45,
        s=28,
        label="Adapted LoRA and Adam state carried, then updated",
        zorder=3,
    )
    axis.scatter(
        [int(row["stage"]) for row in reset],
        [int(row["slot"]) for row in reset],
        color="#d95f02",
        edgecolor="white",
        linewidth=0.55,
        marker="D",
        s=44,
        label="Sealed source leaf/parent loaded, then updated",
        zorder=4,
    )
    for stage in (2, 4, 8, 16, 32):
        axis.axvline(stage, color="#888888", linewidth=0.7, alpha=0.35)
    axis.set(
        xlabel="Tasks observed / online stage",
        ylabel="LogT hierarchy level",
        xlim=(0.5, 50.5),
        ylim=(-0.45, 5.45),
        yticks=range(6),
        yticklabels=tuple(f"L{level} ({2**level} task{'s' if level else ''})" for level in range(6)),
    )
    axis.set_title("Frontier-node LoRA lifetimes and exact persistence boundaries")
    axis.grid(axis="x", alpha=0.12)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=8.3,
        frameon=False,
        ncol=2,
    )
    path = reports / "adapter_lifecycle.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_fragmentation(
    reports: Path, fragmentation_rows: tuple[dict[str, object], ...]
) -> Path:
    """Plot the task-free loss associated with wider LogT frontiers."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11.2, 5.0), constrained_layout=True)
    for capacity, kind, color, marker in (
        (4_096, "affine", "#1f77b4", "o"),
        (8_192, "affine", "#d95f02", "s"),
        (4_096, "mlp", "#9467bd", "^"),
    ):
        selected = tuple(
            row
            for row in fragmentation_rows
            if int(row["historical_capacity"]) == capacity and row["integrator_kind"] == kind
        )
        if not selected:
            continue
        axis.plot(
            [int(row["live_nodes"]) for row in selected],
            [float(row["mean_true_node_oracle_gap"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=f"{'MLP' if kind == 'mlp' else 'Affine'} H={capacity:,}",
        )
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set(
        xlabel="Live frontier nodes",
        ylabel="True-node oracle minus task-free accuracy (points)",
        xticks=range(1, 6),
    )
    axis.set_title("True-node diagnostic gap by frontier fragmentation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=10.5, frameon=True)
    path = reports / "fragmentation_oracle_gap.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _png_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _plot_resources(reports: Path, rows: tuple[dict[str, object], ...]) -> Path:
    """Compare full algorithm training costs, with recomputation counted explicitly."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11.2, 8.0), sharex=True, constrained_layout=True)
    fields = (
        ("cumulative_training_wall_seconds", 60, "Cumulative recorded training wall-time (minutes)"),
        ("cumulative_training_forward_images_including_recompute", 1_000_000, "Cumulative training forward image-paths (millions)"),
    )
    for condition, color, style in (
        ("persistent_h4096", "#1f77b4", "-"),
        ("joint_rank16", "#222222", "--"),
        ("joint_rank_matched", "#2ca02c", ":"),
        (MLP_CONDITION_ID, "#9467bd", "--"),
    ):
        selected = tuple(row for row in rows if row["condition_id"] == condition)
        if not selected:
            continue
        for axis, (field, scale, title) in zip(axes, fields, strict=True):
            axis.plot(
                [row["stage"] for row in selected],
                [row[field] / scale for row in selected],
                label=RESOURCE_LABELS[condition], color=color, linestyle=style,
                linewidth=2.3,
            )
            axis.set_title(title, fontsize=12)
            axis.grid(axis="y", alpha=0.25)
            axis.set_ylim(bottom=0)
    persistent = tuple(row for row in rows if row["condition_id"] == "persistent_h4096")
    for axis, (field, scale, _title), hierarchy_field in zip(
        axes, fields,
        ("cumulative_hierarchy_wall_seconds", "cumulative_hierarchy_forward_images"),
        strict=True,
    ):
        axis.plot(
            [row["stage"] for row in persistent],
            [(row[field] - row[hierarchy_field]) / scale for row in persistent],
            color="#6baed6", linestyle="-.", linewidth=1.7,
            label="H=4,096 adaptation only (subtotal)",
        )
        axis.set_xlim(1, 50)
    axes[0].set_ylabel("Minutes")
    axes[1].set_ylabel("Million forward image-paths")
    axes[1].set_xlabel("Tasks observed")
    axes[1].text(
        0.03, 0.94, "Both joint curves have identical path counts.\n"
        + ("MLP and affine H=4,096 also have identical ViT path counts.\n" if any(row["condition_id"] == MLP_CONDITION_ID for row in rows) else "")
        + "Frontier totals include activation-checkpoint recomputation.",
        transform=axes[1].transAxes, fontsize=9, va="top",
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=10.5, frameon=False, ncol=2)
    path = reports / "cumulative_resources.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _resource_totals(rows: tuple[dict[str, object], ...]) -> tuple[tuple[object, ...], ...]:
    """Return presentation-ready totals, retaining recomputation and backward counts."""
    final = {row["condition_id"]: row for row in rows if row["stage"] == 50}
    persistent = final["persistent_h4096"]
    totals = tuple(
        (
            label,
            final[condition]["cumulative_training_wall_seconds"],
            final[condition]["cumulative_training_forward_images"],
            final[condition]["cumulative_recompute_forward_images"],
            final[condition]["cumulative_training_backward_images"],
        )
        for condition, label in (
            ("persistent_h4096", "Affine H=4,096 total"),
            ("joint_rank16", "Joint IID, rank 16"),
            ("joint_rank_matched", "Joint IID, rank matched"),
            (MLP_CONDITION_ID, "MLP H=4,096 total"),
        )
        if condition in final
    )
    hierarchy_images = persistent["cumulative_hierarchy_forward_images"]
    adaptation_images = persistent["cumulative_adaptation_forward_images"]
    subtotals = (
        ("  H4 hierarchy subtotal", persistent["cumulative_hierarchy_wall_seconds"], hierarchy_images, 0, hierarchy_images),
        ("  H4 adaptation subtotal", persistent["cumulative_adaptation_wall_seconds"], adaptation_images, persistent["cumulative_recompute_forward_images"], adaptation_images),
    )
    return tuple(
        (label, f"{seconds / 60:.2f}", f"{forward / 1e6:.3f}", f"{recompute / 1e6:.3f}", f"{backward / 1e6:.3f}")
        for label, seconds, forward, recompute, backward in (totals[0], *subtotals, *totals[1:])
    )


RESOURCE_COLUMNS = ("Condition / component", "Train min", "Forward M", "Recompute M", "Backward M")


def _resource_explanation(
    rows: tuple[dict[str, object], ...], sources: dict[str, object]
) -> tuple[tuple[str, str], ...]:
    """Explain exactly which recorded costs support the training complexity claim."""
    final = {row["condition_id"]: row for row in rows if row["stage"] == 50}
    persistent, joint = final["persistent_h4096"], final["joint_rank16"]
    time_ratio = persistent["cumulative_training_wall_seconds"] / joint["cumulative_training_wall_seconds"]
    forward_ratio = persistent["cumulative_training_forward_images_including_recompute"] / joint["cumulative_training_forward_images"]
    offline = sources["joint_final"]["metrics"]
    mlp_cost = (
        f" The MLP H=4,096 total is {final[MLP_CONDITION_ID]['cumulative_training_wall_seconds'] / 60:.2f} "
        "recorded training minutes, with exactly the same ViT path counts as affine H=4,096."
        if MLP_CONDITION_ID in final else ""
    )
    mlp_evaluation = (
        f" MLP H=4,096 adds {final[MLP_CONDITION_ID]['cumulative_evaluation_wall_seconds'] / 60:.2f} "
        "minutes for the same test-path count."
        if MLP_CONDITION_ID in final else ""
    )
    return (
        (
            "Training cost at 50 tasks",
            f"Affine H=4,096 costs {time_ratio:.2f} times the rank-16 joint curve's recorded training time and "
            f"{forward_ratio:.2f} times its forward image-path count including checkpoint recomputation. "
            "The hierarchy subtotal includes all 50 leaves and 47 parents, including intermediate parents "
            "created and retired in the same arrival. The smaller asymptotic bound has not produced a wall-time saving at this horizon." + mlp_cost,
        ),
        (
            "What is measured",
            "One image-path is one image processed by one ViT with one installed adapter. Forward and backward "
            "counts are reconstructed from completed image-presentation counters and actual live-node counts. "
            "Recomputation is one additional forward invocation per image/node under activation checkpointing; it can stop "
            "early. These are path counts, not profiled FLOPs or equal-cost forward/backward operations. "
            "The dense head (affine or MLP), optimizer, and differing LoRA ranks also affect wall time.",
        ),
        (
            "Wall-time and reuse",
            "Curves sum recorded training-job wall times, including data loading and checkpoint writes inside "
            "those timers. They exclude setup, artifact validation, inter-job overhead, and evaluation. Source "
            "training is charged when its subtree becomes available. The imported final rank-16 joint model "
            "is charged its original training cost; each reused reference is counted once within each condition. "
            "These are alternative algorithm totals, not a sum of actual work across the experiment invocations.",
        ),
        (
            "Why ViT training is O(T log T)",
            "Let P_t = 4(m_t + min(4096, M_(t-1))), where m_t is the new task size and M_t is the seen "
            "training prefix. With k_t = popcount(t), adaptive forwards are A_T = sum(k_t P_t). "
            "Each hierarchy example belongs to at most one five-epoch job per level, giving S_T = 5 sum_v(n_v). "
            "Total forward work including checkpoint recomputation is S_T + 2A_T; backward work is S_T + A_T. "
            "For fixed replay capacity, epochs, model size, and bounded task size, both are O(T log T). "
            "The joint curves each use 5 sum_t(M_t) forward/backward pairs, which is O(T squared) in model paths.",
        ),
        (
            "The entire benchmarking workflow has a larger bound",
            f"Repeated full-prefix tests add {persistent['cumulative_evaluation_forward_images']:,} forward image-paths "
            f"and {persistent['cumulative_evaluation_wall_seconds'] / 60:.2f} measured minutes for affine H=4,096. "
            f"Each joint curve adds {joint['cumulative_evaluation_forward_images']:,} test paths. Rank-16 test time "
            f"is {joint['cumulative_evaluation_wall_seconds'] / 60:.2f} minutes; rank-matched test wall-time was not retained. "
            "Testing every growing prefix is O(T squared log T) for the frontier. The replay sampler also scans "
            "all earlier identities at each arrival, giving at least quadratic CPU bookkeeping. The O(T log T) claim "
            "therefore applies to ViT training, not the whole runner. Both head architectures have 200 outputs here; "
            "unbounded class growth would require separate head-cost accounting." + mlp_evaluation,
        ),
        (
            "Joint curves versus one final offline fit",
            "Both joint curves refit at every prefix, so their base-model path counts coincide. The larger "
            "rank changes arithmetic per path, not the number of paths. A single final rank-16 offline fit "
            f"cost {offline['wall_seconds'] / 60:.2f} minutes and {offline['image_presentations']:,} forward/backward "
            "pairs; it provides only the final model and has linear training work in the total image count.",
        ),
    )


def _selected_cells(stage_rows: tuple[dict[str, object], ...]) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Use the same selected-stage columns in PDF, Markdown, and HTML."""
    fields = (
        ("Stage", "stage"), ("Nodes", "live_nodes"),
        ("Affine 4k", "persistent_affine_h4096_accuracy"),
        ("Affine 8k", "persistent_affine_h8192_accuracy"),
    )
    fields += (
        (("MLP 4k", "persistent_mlp_h4096_accuracy"), ("MLP oracle", "mlp_h4096_true_node_oracle_accuracy"))
        if "persistent_mlp_h4096_accuracy" in stage_rows[0]
        else (("Affine 4k oracle", "h4096_true_node_oracle_accuracy"), ("Affine 8k oracle", "h8192_true_node_oracle_accuracy"))
    )
    fields += (("Joint r16", "stage_matched_joint_rank16_accuracy"), ("Rank-matched", "rank_matched_joint_accuracy"), ("Rank", "rank_matched_joint_rank"))
    cells = tuple(
        tuple(f"{row[field]:.2f}%" if field.endswith("accuracy") else str(row[field]) for _label, field in fields)
        for row in stage_rows if row["stage"] in {1, 2, 4, 8, 16, 31, 32, 50}
    )
    return tuple(label for label, _field in fields), cells


def _fragmentation_cells(fragmentation_rows: tuple[dict[str, object], ...]) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Keep architecture names explicit when two conditions use H=4,096."""
    by_key = {
        (row["integrator_kind"], int(row["historical_capacity"]), int(row["live_nodes"])): row
        for row in fragmentation_rows
    }
    conditions = tuple(
        (kind, capacity, label)
        for kind, capacity, label in (("affine", 4096, "Affine 4k"), ("affine", 8192, "Affine 8k"), ("mlp", 4096, "MLP 4k"))
        if (kind, capacity, 1) in by_key
    )
    columns = ("Nodes", "Stages") + tuple(f"{label} {metric}" for _kind, _capacity, label in conditions for metric in ("acc.", "gap"))
    cells = tuple(
        (str(live_nodes), str(by_key[("affine", 4096, live_nodes)]["stages"]))
        + tuple(
            value
            for kind, capacity, _label in conditions
            for value in (f"{by_key[(kind, capacity, live_nodes)]['mean_accuracy']:.2f}%", f"{by_key[(kind, capacity, live_nodes)]['mean_true_node_oracle_gap']:.2f}")
        )
        for live_nodes in range(1, 6)
    )
    return columns, cells


def _markdown_table(columns: tuple[str, ...], cells: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join((
        "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns),
        *("| " + " | ".join(row) + " |" for row in cells),
    ))


def _html_table(columns: tuple[str, ...], cells: tuple[tuple[str, ...], ...]) -> str:
    return (
        "<table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in cells)
        + "</tbody></table>"
    )


def _interpretation(summaries: tuple[dict[str, object], ...]) -> str:
    by_name = {str(row["condition"]): row for row in summaries}
    h4, h8 = by_name[LABEL_H4096], by_name[LABEL_H8192]
    stage, rank = by_name[LABEL_STAGE_JOINT], by_name[LABEL_RANK_JOINT]
    if LABEL_MLP in by_name:
        mlp = by_name[LABEL_MLP]
        return (
            f"At task 50, MLP H=4,096 reaches {mlp['final_accuracy']:.3f}% accuracy, versus "
            f"{h4['final_accuracy']:.3f}% for affine H=4,096 and {h8['final_accuracy']:.3f}% for affine H=8,192. "
            f"Rank-16 and aggregate-rank joint IID reach {stage['final_accuracy']:.3f}% and {rank['final_accuracy']:.3f}%. "
            f"Mean accuracy over all 50 stages is {mlp['incremental_accuracy']:.3f}% for MLP, "
            f"{h4['incremental_accuracy']:.3f}% and {h8['incremental_accuracy']:.3f}% for the affine arms, "
            f"and {stage['incremental_accuracy']:.3f}% and {rank['incremental_accuracy']:.3f}% for the joint curves."
        )
    return (
        f"At task 50, H=4,096 reaches {h4['final_accuracy']:.3f}% and H=8,192 reaches "
        f"{h8['final_accuracy']:.3f}%. The corresponding rank-16 and aggregate-rank joint-IID "
        f"references are {stage['final_accuracy']:.3f}% and {rank['final_accuracy']:.3f}%. "
        f"Across all 50 stages, incremental accuracy is {h4['incremental_accuracy']:.3f}% for H=4,096 "
        f"and {h8['incremental_accuracy']:.3f}% for H=8,192, versus {stage['incremental_accuracy']:.3f}% "
        f"and {rank['incremental_accuracy']:.3f}% for the two joint-IID curves."
    )


def _curve_note(result: dict[str, object]) -> str:
    return (
        "Blue and orange are the single-affine H=4,096 and H=8,192 conditions. "
        + ("Purple is the two-layer ReLU MLP at H=4,096. " if "mlp" in result else "")
        + "All use each live node's own adapted 768-value pre-classifier latent, without macro tokens or metadata. "
        "Dotted matching colors are label-aware true-node diagnostics, not deployable conditions. "
        "Black is stage-matched rank-16 joint IID; green matches total live LoRA rank."
    )


def _write_markdown(
    reports: Path,
    result: dict[str, object],
    stage_rows: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    fragmentation_rows: tuple[dict[str, object], ...],
    resource_rows: tuple[dict[str, object], ...],
    source_costs: dict[str, object],
) -> Path:
    resource_table = _markdown_table(RESOURCE_COLUMNS, _resource_totals(resource_rows))
    resource_text = "\n\n".join(
        f"### {title}\n\n{text}"
        for title, text in _resource_explanation(resource_rows, source_costs)
    )
    mlp_text = "\n\n".join(f"### {title}\n\n{paragraph}" for title, paragraph in mlp_explanation(result))
    if "mlp" in result:
        mlp_text += "\n\n" + _markdown_table(*mlp_detail_cells(result))
        mlp_text += f"\n\nMLP result `{result['mlp']['content_hash']}` under protocol `{result['mlp']['protocol_hash']}`."
    text = f"""# {REPORT_TITLE}

## Result

{_interpretation(summaries)}

![Full-stream test accuracy](stage_accuracy.png)

{_curve_note(result)} Vertical guides mark power-of-two consolidations.

## Selected stages

{_markdown_table(*_selected_cells(stage_rows))}

Here 4k means H=4,096 and 8k means H=8,192; oracle columns are diagnostic only.

![Accuracy gaps to joint-IID controls](joint_iid_gaps.png)

Positive values mean the named persistent integrator is ahead. The top panel
holds joint rank fixed at 16; the bottom panel adjusts joint rank to the number
of live nodes. Neither reference is an execution gate.

![Test NLL](stage_nll.png)

NLL is shown for the persistent integrators and newly trained aggregate-rank models.
The imported one-node rank-16 artifacts retained predictions but not logits,
so those six green NLL points are intentionally absent; accuracy is complete.

## Fragmentation diagnosis

![Oracle gap by live-node count](fragmentation_oracle_gap.png)

{_markdown_table(*_fragmentation_cells(fragmentation_rows))}

The true-node oracle is label-aware and not deployable. Its widening advantage
is consistent with cross-node competition, but the diagnostic also substitutes
the frozen node-local classifier, so it does not isolate routing alone.

## Frontier-node LoRA lifetimes

![Frontier-node LoRA lifetimes](adapter_lifecycle.png)

Every marker is followed by adaptation at that stage. Blue circles continue the
exact final LoRA factors and named AdamW state from the preceding stage. Orange
diamonds load an authenticated source leaf or full-union parent. Consolidation
parents do not inherit the online-adapted LoRAs of their retired children.

## Protocol

- H=4,096 uses four epochs per arrival; H=8,192 uses five.
- Replay is class-stratified and deterministically redrawn at every stage.
- Dense-head parameters and AdamW state persist. A node LoRA and its moments persist
  only while the exact hierarchy-node hash remains live.
- A leaf or consolidation parent enters from its authenticated source model.
- Local classifiers and the ViT base stay frozen. Evaluation is task-free.
- The test split is used only after each stage is sealed and never selects a
  checkpoint, replay population, or condition.

{mlp_text}

## Resource and provenance

![Cumulative training wall-time and model passes](cumulative_resources.png)

{resource_table}

M means million image-paths. Recompute is activation checkpointing, additional to
ordinary forwards. H4 subtotals add to the affine H=4,096 total; do not sum the total again.

{resource_text}

The original affine/joint comparison is result `{result['content_hash']}` under protocol
`{result['protocol_hash']}`. Source hierarchy unchanged:
`{result['hierarchy_source_unchanged']}`. Work in that original completing invocation:
`{json.dumps(result['invocation_work'], sort_keys=True)}`.
"""
    path = reports / "REPORT.md"
    atomic_write(path, text.encode("utf-8"))
    return path


def _write_html(
    reports: Path,
    result: dict[str, object],
    stage_rows: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    fragmentation_rows: tuple[dict[str, object], ...],
    images: tuple[Path, ...],
    resource_rows: tuple[dict[str, object], ...],
    source_costs: dict[str, object],
) -> Path:
    selected_table = _html_table(*_selected_cells(stage_rows))
    embedded = tuple(_png_data(image) for image in images)
    fragmentation_table = _html_table(*_fragmentation_cells(fragmentation_rows))
    resource_table = _html_table(RESOURCE_COLUMNS, _resource_totals(resource_rows))
    resource_text = "".join(
        f"<h3>{escape(title)}</h3><p>{escape(text)}</p>"
        for title, text in _resource_explanation(resource_rows, source_costs)
    )
    mlp_text = "".join(f"<h3>{escape(title)}</h3><p>{escape(paragraph)}</p>" for title, paragraph in mlp_explanation(result))
    if "mlp" in result:
        mlp_text += _html_table(*mlp_detail_cells(result))
        mlp_text += f"<p>MLP result <code>{result['mlp']['content_hash']}</code><br>MLP protocol <code>{result['mlp']['protocol_hash']}</code></p>"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{REPORT_TITLE}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:36px auto;color:#20252a;padding:0 22px}}
h1,h2{{color:#16324f}} img{{max-width:100%;height:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #b8c2cc;padding:7px;text-align:right}} th{{background:#16324f;color:white}}
code{{background:#eef2f5;padding:2px 4px}} .note{{color:#4b5563}}
</style></head><body>
<h1>{REPORT_TITLE}</h1>
<p>{escape(_interpretation(summaries))}</p>
<img src="data:image/png;base64,{embedded[0]}" alt="Full-stream test accuracy">
<p class="note">{escape(_curve_note(result))}</p>
<h2>Selected stages</h2>
{selected_table}
<p>4k means H=4,096 and 8k means H=8,192; oracle columns are diagnostic only.</p>
<img src="data:image/png;base64,{embedded[1]}" alt="Accuracy gaps to joint-IID controls">
<p>Positive values mean the named persistent integrator is ahead of the joint-IID reference. Neither reference is an execution gate.</p>
<img src="data:image/png;base64,{embedded[2]}" alt="Test negative log likelihood">
<h2>Fragmentation diagnosis</h2>
<img src="data:image/png;base64,{embedded[3]}" alt="True-node oracle gap by live-node count">
{fragmentation_table}
<p>The true-node oracle is label-aware and diagnostic only. Its widening advantage is consistent with cross-node competition, but it also substitutes the frozen node-local classifier, so it does not isolate routing alone.</p>
<h2>Frontier-node LoRA lifetimes</h2>
<img src="data:image/png;base64,{embedded[4]}" alt="Frontier-node LoRA carry and reset history">
<p>Every marker is followed by adaptation at that stage. Blue circles carry the exact final LoRA factors and named AdamW state from the preceding stage. Orange diamonds load a sealed source leaf or full-union parent. A consolidation parent does not inherit the online-adapted LoRAs of its retired children.</p>
<h2>Protocol</h2>
<ul><li>H=4,096 uses four epochs per arrival; H=8,192 uses five.</li>
<li>Replay is class-stratified and deterministically redrawn at every stage.</li>
<li>Dense-head parameters and AdamW state persist. Node LoRAs persist exactly while their hierarchy-node hashes remain live.</li>
<li>New leaves and consolidation parents enter from authenticated source models. The base ViT and local classifiers stay frozen.</li>
<li>The test split is used only after each stage is sealed.</li></ul>
{mlp_text}
<h2>Cumulative wall-time and model passes</h2>
<img src="data:image/png;base64,{embedded[5]}" alt="Cumulative training wall-time and forward model-image paths">
{resource_table}
<p>M means million image-paths. Recompute is activation checkpointing, additional to ordinary forwards. H4 subtotals add to the affine H=4,096 total.</p>
{resource_text}
<h2>Provenance</h2>
<p>Original affine/joint protocol <code>{escape(str(result['protocol_hash']))}</code><br>
Result <code>{escape(str(result['content_hash']))}</code><br>
Source hierarchy unchanged: <code>{result['hierarchy_source_unchanged']}</code><br>
Original completing-invocation work: <code>{escape(json.dumps(result['invocation_work'], sort_keys=True))}</code></p>
</body></html>"""
    path = reports / "REPORT.html"
    atomic_write(path, html.encode("utf-8"))
    return path


def _write_pdf(
    output: Path,
    result: dict[str, object],
    summaries: tuple[dict[str, object], ...],
    stage_rows: tuple[dict[str, object], ...],
    fragmentation_rows: tuple[dict[str, object], ...],
    images: tuple[Path, ...],
    resource_rows: tuple[dict[str, object], ...],
    source_costs: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def wrapped(text: str, width: int = 108) -> str:
        return "\n".join(textwrap.wrap(text, width=width))

    def footer(figure, page: int) -> None:
        figure.text(
            0.06,
            0.025,
            REPORT_TITLE,
            fontsize=8,
            color="#666666",
        )
        figure.text(
            0.94,
            0.025,
            f"Page {page}",
            fontsize=8,
            color="#666666",
            horizontalalignment="right",
        )

    def image_axis(figure, image: Path, bounds: tuple[float, float, float, float]) -> None:
        axis = figure.add_axes(bounds)
        axis.imshow(plt.imread(image))
        axis.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".persistent-affine-", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    selected_columns, selected_cells = _selected_cells(stage_rows)
    with PdfPages(
        temporary,
        metadata={
            "Title": REPORT_TITLE,
            "Author": "APM continual-learning experiment",
            "Subject": "Full 50-task adaptive-LoRA affine and two-layer MLP experiment",
        },
    ) as document:
        first = plt.figure(figsize=(8.5, 11), facecolor="white")
        first.text(
            0.5,
            0.95,
            REPORT_TITLE,
            fontsize=17,
            weight="bold",
            color="#16324f",
            horizontalalignment="center",
        )
        first.text(
            0.5,
            0.915,
            "Single-affine H=4,096 / H=8,192 and two-layer MLP H=4,096"
            if "mlp" in result else "Full 50-task H=4,096 and H=8,192 adaptive-LoRA conditions",
            fontsize=11.5,
            horizontalalignment="center",
        )
        first.text(
            0.06,
            0.865,
            wrapped(_interpretation(summaries), 90),
            fontsize=10.2,
            va="top",
        )
        image_axis(first, images[0], (0.055, 0.28, 0.89, 0.45))
        first.text(
            0.06,
            0.235,
            wrapped(_curve_note(result)),
            fontsize=9.2,
            va="top",
            color="#40464d",
        )
        footer(first, 1)
        document.savefig(first)
        plt.close(first)

        second = plt.figure(figsize=(8.5, 11), facecolor="white")
        second.text(0.06, 0.95, "Selected stages and control gaps", fontsize=17, weight="bold", color="#16324f")
        table_axis = second.add_axes((0.055, 0.67, 0.89, 0.22))
        table_axis.axis("off")
        table = table_axis.table(
            cellText=selected_cells,
            colLabels=selected_columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.45)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#b8c2cc")
            if row == 0:
                cell.set_facecolor("#16324f")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f2f5f7")
        image_axis(second, images[1], (0.055, 0.12, 0.89, 0.51))
        second.text(
            0.06,
            0.09,
            wrapped(
                "Positive gap means the named integrator is ahead of the joint-IID reference. Power-of-two stages have one node. Table labels 4k/8k mean H=4,096/8,192; oracle is diagnostic only.",
                105,
            ),
            fontsize=8.8,
            color="#40464d",
            va="top",
        )
        footer(second, 2)
        document.savefig(second)
        plt.close(second)

        third = plt.figure(figsize=(8.5, 11), facecolor="white")
        third.text(0.06, 0.95, "Likelihood and protocol", fontsize=17, weight="bold", color="#16324f")
        image_axis(third, images[2], (0.055, 0.51, 0.89, 0.38))
        third.text(
            0.06,
            0.49,
            wrapped(
                "Rank-16 source artifacts retained hard predictions rather than logits, so rank-matched NLL is absent at stages 1, 2, 4, 8, 16, and 32. Accuracy remains available at all 50 stages."
            ),
            fontsize=8.8,
            va="top",
            color="#40464d",
        )
        third.text(
            0.06,
            0.415,
            "Exact online update rule",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.39,
            wrapped(
                "Every arrival uses all current-task images plus a deterministic stage-keyed class-stratified historical draw. H=4,096 receives four passes and H=8,192 receives five. Dense-head parameters and AdamW state carry forward. A node adapter and its moments carry only if its hierarchy-node hash survives; new leaves and consolidation parents start from authenticated source adapters. The base ViT and local classifiers never train.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        third.text(
            0.06,
            0.275,
            "Comparison boundary",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.25,
            wrapped(
                "Stage-matched joint IID trains one fresh rank-16 adapter for five epochs on every prefix. Aggregate-rank-matched joint IID uses rank and alpha equal to 16 times popcount(stage). It matches total live LoRA rank, but not pretrained node state, multiple ViT paths, dense-head parameters, or deployment compute. Neither curve gates execution.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        third.text(
            0.06,
            0.145,
            "Original affine/joint provenance",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.12,
            f"Protocol {result['protocol_hash']}\nResult {result['content_hash']}\nSource hierarchy unchanged: {result['hierarchy_source_unchanged']}",
            fontsize=7.7,
            va="top",
            family="monospace",
        )
        footer(third, 3)
        document.savefig(third)
        plt.close(third)

        fourth = plt.figure(figsize=(8.5, 11), facecolor="white")
        fourth.text(
            0.06,
            0.95,
            "Fragmentation diagnosis",
            fontsize=17,
            weight="bold",
            color="#16324f",
        )
        image_axis(fourth, images[3], (0.055, 0.43, 0.89, 0.43))
        fragmentation_columns, fragmentation_cells = _fragmentation_cells(fragmentation_rows)
        table_axis = fourth.add_axes((0.055, 0.19, 0.89, 0.17))
        table_axis.axis("off")
        table = table_axis.table(
            cellText=fragmentation_cells,
            colLabels=fragmentation_columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.3)
        table.scale(1.0, 1.4)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#b8c2cc")
            if row == 0:
                cell.set_facecolor("#16324f")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f2f5f7")
        fourth.text(
            0.06,
            0.125,
            wrapped(
                "The label-aware true-node oracle is diagnostic only. Its growing advantage as the number of independently adapted frontier nodes rises is consistent with cross-node competition. Because it also substitutes the frozen node-local classifier, it does not isolate routing alone.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        footer(fourth, 4)
        document.savefig(fourth)
        plt.close(fourth)

        fifth = plt.figure(figsize=(8.5, 11), facecolor="white")
        fifth.text(
            0.06,
            0.95,
            "Frontier-node adaptation lifetimes",
            fontsize=17,
            weight="bold",
            color="#16324f",
        )
        image_axis(fifth, images[4], (0.055, 0.45, 0.89, 0.42))
        fifth.text(
            0.06,
            0.405,
            "What persists",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        fifth.text(
            0.06,
            0.38,
            wrapped(
                "A blue lifetime continues only while the exact source hierarchy-node hash remains live. The next stage begins from that node's final adapted rank-16 LoRA, its 768-column head input block, and its named AdamW moments. All live LoRAs then update jointly on every current-plus-replay example. The affine and MLP conditions share these lifetimes but train their own independent parameters.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        fifth.text(
            0.06,
            0.265,
            "What resets",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        fifth.text(
            0.06,
            0.24,
            wrapped(
                "An orange diamond is a new leaf or consolidation parent loaded from the sealed source hierarchy. A parent was independently retrained on the full union of its represented tasks; it does not inherit the online-adapted LoRAs or optimizer moments of retired children. Unaffected nodes continue normally. Global head biases persist; in the MLP, shared hidden biases and old-class output rows also persist across consolidation.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        footer(fifth, 5)
        document.savefig(fifth)
        plt.close(fifth)

        explanations = _resource_explanation(resource_rows, source_costs)
        for page_number, heading, sections in (
            (6, "Cumulative training cost", explanations[:2]),
            (7, "Resource accounting and scaling", explanations[2:]),
        ):
            page = plt.figure(figsize=(8.5, 11), facecolor="white")
            page.text(0.06, 0.95, heading, fontsize=17, weight="bold", color="#16324f")
            if page_number == 6:
                image_axis(page, images[5], (0.055, 0.325, 0.89, 0.59))
                cursor = 0.29
            else:
                table_axis = page.add_axes((0.055, 0.735, 0.89, 0.17))
                table_axis.axis("off")
                table = table_axis.table(
                    cellText=_resource_totals(resource_rows), colLabels=RESOURCE_COLUMNS,
                    colWidths=(0.36, 0.16, 0.16, 0.16, 0.16), cellLoc="center", loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.0)
                table.scale(1.0, 1.6)
                for (row, _column), cell in table.get_celld().items():
                    cell.set_edgecolor("#b8c2cc")
                    if row == 0:
                        cell.set_facecolor("#16324f")
                        cell.get_text().set_color("white")
                        cell.get_text().set_weight("bold")
                    elif row % 2 == 0:
                        cell.set_facecolor("#f2f5f7")
                page.text(
                    0.06, 0.72,
                    "M = million image-paths. Recompute = activation checkpointing. Affine H4 has two subtotals.",
                    fontsize=8.3, va="top", color="#40464d",
                )
                cursor = 0.675
            for title, paragraph in sections:
                page.text(0.06, cursor, title, fontsize=10.5, weight="bold", color="#16324f")
                lines = textwrap.wrap(paragraph, width=110)
                cursor -= 0.022
                page.text(0.06, cursor, "\n".join(lines), fontsize=9.1, va="top")
                cursor -= len(lines) * 0.014 + 0.03
            if cursor < 0.025:
                raise ValueError("resource report text overflows the page")
            footer(page, page_number)
            document.savefig(page)
            plt.close(page)
        if "mlp" in result:
            page = plt.figure(figsize=(8.5, 11), facecolor="white")
            page.text(0.06, 0.95, "Two-layer MLP: matched H=4,096", fontsize=17, weight="bold", color="#16324f")
            columns, cells = mlp_detail_cells(result)
            table_axis = page.add_axes((0.055, 0.725, 0.89, 0.19))
            table_axis.axis("off")
            table = table_axis.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(7.1)
            table.scale(1, 1.4)
            for (row, _column), cell in table.get_celld().items():
                cell.set_edgecolor("#b8c2cc")
                if row == 0:
                    cell.set_facecolor("#16324f")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f2f5f7")
            cursor = 0.675
            for title, paragraph in mlp_explanation(result):
                page.text(0.06, cursor, title, fontsize=10.5, weight="bold", color="#16324f")
                lines = textwrap.wrap(paragraph, width=110)
                cursor -= 0.022
                page.text(0.06, cursor, "\n".join(lines), fontsize=9.1, va="top")
                cursor -= len(lines) * 0.014 + 0.029
            if cursor < 0.095:
                raise ValueError("MLP report text overflows the page")
            page.text(0.06, 0.08, f"MLP protocol {result['mlp']['protocol_hash']}\nMLP result   {result['mlp']['content_hash']}", fontsize=7.3, family="monospace", va="top")
            footer(page, 8)
            document.savefig(page)
            plt.close(page)
    os.replace(temporary, output)


def write_persistent_affine_report(run: str | Path) -> Path:
    """Build tables, figures, Markdown/HTML projections, and the final PDF."""
    run_path = Path(run).resolve()
    result = _validate_result(run_path)
    reports = run_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summaries = _summary_rows(result)
    stage_rows = _stage_rows(result)
    lifecycle_rows = _lifecycle_rows(result)
    fragmentation_rows = _fragmentation_rows(result)
    source_costs = load_source_training_costs(run_path, result)
    config = load_canonical_json(run_path / "config_resolved.json")
    resource_rows = resource_stage_rows(result, source_costs, config["activation_recomputation"])
    resource_rows += mlp_resource_rows(result, resource_rows)
    _write_tables(
        reports, stage_rows, summaries, lifecycle_rows, fragmentation_rows, resource_rows
    )
    images = (
        _plot_accuracy(reports, result),
        _plot_gaps(reports, stage_rows),
        _plot_nll(reports, result),
        _plot_fragmentation(reports, fragmentation_rows),
        _plot_lifecycle(reports, lifecycle_rows),
        _plot_resources(reports, resource_rows),
    )
    markdown = _write_markdown(
        reports, result, stage_rows, summaries, fragmentation_rows, resource_rows, source_costs
    )
    html = _write_html(
        reports, result, stage_rows, summaries, fragmentation_rows, images, resource_rows, source_costs
    )
    project_root = run_path.parents[4]
    pdf = project_root / "output/pdf/imagenetr50_persistent_affine_v15_report.pdf"
    _write_pdf(pdf, result, summaries, stage_rows, fragmentation_rows, images, resource_rows, source_costs)
    manifest_core = {
        "condition_summary_sha256": file_sha256(reports / "condition_summary.json"),
        "adapter_lifecycle_sha256": file_sha256(reports / "adapter_lifecycle.json"),
        "fragmentation_summary_sha256": file_sha256(reports / "fragmentation_summary.json"),
        "resource_metrics_sha256": file_sha256(reports / "resource_metrics.json"),
        "source_training_costs_sha256": file_sha256(reports / "source_training_costs.json"),
        **({"mlp_result_hash": result["mlp"]["content_hash"], "mlp_extension_sha256": file_sha256(reports / "mlp_extension.json")} if "mlp" in result else {}),
        "figure_sha256": {
            image.name: file_sha256(image)
            for image in images
        },
        "html_sha256": file_sha256(html),
        "markdown_sha256": file_sha256(markdown),
        "pdf": str(pdf),
        "pdf_sha256": file_sha256(pdf),
        "result_hash": result["content_hash"],
        "schema_version": "imagenetr50-persistent-affine-report-manifest-v1",
        "stage_metrics_sha256": file_sha256(reports / "stage_metrics.json"),
    }
    atomic_write(
        reports / "report_manifest.json",
        canonical_json_bytes({**manifest_core, "content_hash": record_sha256(manifest_core)}),
    )
    return pdf


__all__ = [
    "LABEL_H4096",
    "LABEL_H8192",
    "LABEL_ORACLE_H4096",
    "LABEL_ORACLE_H8192",
    "LABEL_RANK_JOINT",
    "LABEL_STAGE_JOINT",
    "write_persistent_affine_report",
]
