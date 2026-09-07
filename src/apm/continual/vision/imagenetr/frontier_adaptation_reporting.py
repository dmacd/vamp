"""Publication-style reporting for the stage-31 frontier-LoRA adaptation audit."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf
from apm.continual.vision.imagenetr.frontier_architecture_replay_reporting import (
    ArchitectureReplayReport,
    load_architecture_replay_report,
)


MACRO_FAMILY_LABEL = "Macro-token frontier, adaptive LoRAs"
LINEAR_FAMILY_LABEL = "Single-affine frontier, adaptive LoRAs"
RANK80_FAMILY_LABEL = "Joint IID, rank 80"
CONDITION_LABELS = {
    1024: f"{MACRO_FAMILY_LABEL} (H=1,024)",
    2048: f"{MACRO_FAMILY_LABEL} (H=2,048)",
    4096: f"{MACRO_FAMILY_LABEL} (H=4,096)",
    8192: f"{MACRO_FAMILY_LABEL} (H=8,192)",
    11827: f"{MACRO_FAMILY_LABEL} (H=11,827; full fit)",
}
FROZEN_ONLINE_LABEL = "Frozen frontier, online full fit"
FROZEN_CACHED_LABEL = "Frozen macro, cached full fit (seed 1993)"
JOINT_LABEL = "Joint IID, rank 16 (full fit; 5 epochs)"
RANK_MATCHED_LABEL = f"{RANK80_FAMILY_LABEL} (H=11,827; full fit; 5 epochs)"
TOTAL_PARAM_MATCHED_LABEL = (
    "Joint IID, rank 224 (full fit; total-active match; 5 epochs)"
)
LINEAR_PRECLASSIFIER_LABEL = (
    f"{LINEAR_FAMILY_LABEL} (H=11,827; full fit)"
)


@dataclass(frozen=True, slots=True)
class JointCapacityControlSpec:
    """Expected artifact identity and plot styling for one joint control."""

    result_filename: str
    schema_version: str
    rank: int
    label: str
    table_name: str
    color: str
    linestyle: str
    marker: str
    architecture_requirements: tuple[tuple[str, object], ...]


JOINT_CAPACITY_CONTROLS = (
    JointCapacityControlSpec(
        "joint_iid_lora_r80.json",
        "imagenetr50-frontier-rank-matched-control-v1",
        80,
        RANK_MATCHED_LABEL,
        "joint_iid_rank80",
        "#1b7837",
        "-.",
        "s",
        (
            ("lora_alpha", 80),
            ("lora_parameters", 6_635_520),
            ("frontier_aggregate_lora_parameters", 6_635_520),
            ("frontier_integrator_parameters_excluded_from_match", 12_055_496),
            ("trainable_parameters", 6_730_876),
        ),
    ),
    JointCapacityControlSpec(
        "joint_iid_lora_r224.json",
        "imagenetr50-frontier-total-param-matched-control-v1",
        224,
        TOTAL_PARAM_MATCHED_LABEL,
        "joint_iid_rank224",
        "#a6611a",
        "--",
        "^",
        (
            ("lora_alpha", 224),
            ("lora_parameters", 18_579_456),
            ("frontier_aggregate_lora_parameters", 6_635_520),
            ("frontier_integrator_parameters_included_in_match", 12_055_496),
            ("frontier_active_parameters", 18_691_016),
            ("parameter_difference", -16_204),
            ("trainable_parameters", 18_674_812),
        ),
    ),
)


def _validated_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-frontier-adaptation-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or result.get("test_evaluations") != 0
    ):
        raise ValueError("frontier adaptation result does not authenticate")
    return result


def _validated_capacity_control(
    run: Path,
    parent_result: Mapping[str, object],
    specification: JointCapacityControlSpec,
) -> dict[str, object] | None:
    path = run / "evaluations" / specification.result_filename
    if not path.is_file():
        return None
    control = load_canonical_json(path)
    core = {key: value for key, value in control.items() if key != "content_hash"}
    architecture = dict(control.get("architecture", {}))
    fit = dict(control.get("fit", {}))
    if (
        control.get("schema_version") != specification.schema_version
        or control.get("content_hash") != record_sha256(core)
        or control.get("parent_result_hash") != parent_result.get("content_hash")
        or control.get("test_evaluations") != 0
        or architecture.get("lora_rank") != specification.rank
        or any(
            architecture.get(key) != expected
            for key, expected in specification.architecture_requirements
        )
        or fit.get("epochs") != 5
        or fit.get("image_presentations") != 60_970
    ):
        raise ValueError(
            f"rank-{specification.rank} joint-IID control does not authenticate"
        )
    return control


def _capacity_summary(
    control: Mapping[str, object], specification: JointCapacityControlSpec
) -> dict[str, object]:
    architecture = dict(control["architecture"])
    fit = dict(control["fit"])
    frontier_integrator_parameters = int(
        architecture.get(
            "frontier_integrator_parameters_included_in_match",
            architecture.get(
                "frontier_integrator_parameters_excluded_from_match", 0
            ),
        )
    )
    frontier_active_parameters = int(
        architecture.get(
            "frontier_active_parameters",
            int(architecture["frontier_aggregate_lora_parameters"])
            + frontier_integrator_parameters,
        )
    )
    return {
        "best_epoch": int(fit["best_epoch"]),
        "best_validation_accuracy": float(fit["best_validation_accuracy"]),
        "best_validation_nll": float(fit["best_validation_nll"]),
        "classifier_parameters": int(architecture["classifier_parameters"]),
        "condition": specification.label,
        "epochs": int(fit["epochs"]),
        "fixed_validation_accuracy": float(fit["fixed_validation_accuracy"]),
        "fixed_validation_nll": float(fit["fixed_validation_nll"]),
        "frontier_active_parameters": frontier_active_parameters,
        "frontier_integrator_parameters": frontier_integrator_parameters,
        "image_presentations": int(fit["image_presentations"]),
        "lora_alpha": int(architecture["lora_alpha"]),
        "lora_parameters": int(architecture["lora_parameters"]),
        "lora_rank": int(architecture["lora_rank"]),
        "parameter_difference": int(
            architecture.get(
                "parameter_difference",
                int(architecture["trainable_parameters"])
                - frontier_active_parameters,
            )
        ),
        "peak_vram_bytes": int(fit["peak_vram_bytes"]),
        "trainable_parameters": int(architecture["trainable_parameters"]),
        "wall_seconds": float(fit["wall_seconds"]),
    }


def _capacity_history(
    run: Path,
    control: Mapping[str, object],
    specification: JointCapacityControlSpec,
) -> tuple[dict[str, object], ...]:
    """Load one authenticated five-epoch joint-capacity trajectory."""
    history_path = (run / str(control["history"])).resolve()
    if run not in history_path.parents or not history_path.is_file():
        raise ValueError("joint-capacity history escapes or is absent from the run")
    if file_sha256(history_path) != control["history_sha256"]:
        raise ValueError("joint-capacity history hash changed")
    rows = tuple(
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(rows) != 5 or tuple(int(row["epoch"]) for row in rows) != tuple(
        range(1, 6)
    ):
        raise ValueError("joint-capacity history is incomplete")
    return tuple({**row, "condition": specification.label} for row in rows)


def _validated_linear_preclassifier_control(
    run: Path, parent_result: Mapping[str, object]
) -> dict[str, object] | None:
    """Authenticate the optional full-fit linear-integrator result."""
    path = run / "evaluations/frontier_linear_preclassifier.json"
    if not path.is_file():
        return None
    control = load_canonical_json(path)
    core = {key: value for key, value in control.items() if key != "content_hash"}
    architecture = dict(control.get("architecture", {}))
    fit = dict(control.get("fit", {}))
    requirements = {
        "active_nodes": 5,
        "feature_dimension": 768,
        "feature_normalization": "none_after_pinned_vit_pre_logits",
        "feature_source": "node_preclassifier",
        "frontier_lora_parameters": 6_635_520,
        "initialization": "exact_local_classifier_union",
        "input_dimension": 3_840,
        "input_order": "ascending_frontier_level",
        "integrator_kind": "single_affine",
        "integrator_parameters": 768_200,
        "output_classes": 200,
        "source_rank": 16,
        "train_base_vit": False,
        "train_frontier_loras": True,
        "train_integrator": True,
        "train_node_classifiers": False,
        "trainable_parameters": 7_403_720,
    }
    if (
        control.get("schema_version")
        != "imagenetr50-frontier-linear-preclassifier-control-v1"
        or control.get("content_hash") != record_sha256(core)
        or control.get("parent_result_hash") != parent_result.get("content_hash")
        or control.get("test_evaluations") != 0
        or any(architecture.get(key) != value for key, value in requirements.items())
        or fit.get("epochs") != 50
        or fit.get("image_presentations") != 609_700
        or fit.get("fixed_epoch") != 5
        or fit.get("fixed_image_presentations") != 60_970
    ):
        raise ValueError("linear pre-classifier control does not authenticate")
    return control


def _linear_preclassifier_history(
    run: Path, control: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Load the authenticated 50-epoch architecture-ablation trajectory."""
    history_path = (run / str(control["history"])).resolve()
    if (
        run not in history_path.parents
        or not history_path.is_file()
        or file_sha256(history_path) != control["history_sha256"]
    ):
        raise ValueError("linear pre-classifier history changed or escaped the run")
    rows = tuple(
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(rows) != 50 or tuple(int(row["epoch"]) for row in rows) != tuple(
        range(1, 51)
    ):
        raise ValueError("linear pre-classifier history is incomplete")
    return tuple({**row, "condition": LINEAR_PRECLASSIFIER_LABEL} for row in rows)


def _linear_preclassifier_summary(
    control: Mapping[str, object]
) -> dict[str, object]:
    """Project the linear-integrator result into the common frontier table."""
    fit = dict(control["fit"])
    return {
        "accuracy_gap_to_joint_at_min_nll_pp": None,
        "adapt_lora": True,
        "best_nll_epoch": int(fit["best_nll_epoch"]),
        "condition": LINEAR_PRECLASSIFIER_LABEL,
        "historical_capacity": 11_827,
        "image_presentations": int(fit["image_presentations"]),
        "max_accuracy_epoch": int(fit["max_accuracy_epoch"]),
        "max_validation_accuracy": float(fit["max_validation_accuracy"]),
        "nll_gap_to_joint": None,
        "peak_vram_bytes": int(fit["peak_vram_bytes"]),
        "simultaneous_joint_match_epoch": None,
        "train_accuracy_at_best": float(fit["train_accuracy_at_best"]),
        "train_nll_at_best": float(fit["train_nll_at_best"]),
        "trainable_parameters": int(fit["trainable_parameters"]),
        "training_examples": 12_194,
        "validation_accuracy_at_best_nll": float(
            fit["validation_accuracy_at_best_nll"]
        ),
        "validation_nll_at_max_accuracy": float(
            fit["validation_nll_at_max_accuracy"]
        ),
        "validation_nll_minimum": float(fit["best_validation_nll"]),
        "wall_seconds": float(fit["wall_seconds"]),
        "fixed_validation_accuracy": float(fit["fixed_validation_accuracy"]),
        "fixed_validation_nll": float(fit["fixed_validation_nll"]),
    }


def _capacity_text(capacity: int) -> str:
    return "11,827; full fit" if capacity == 11_827 else f"{capacity:,}"


def _architecture_condition(family: str, capacity: int) -> str:
    labels = {
        "macro_token": MACRO_FAMILY_LABEL,
        "single_affine": LINEAR_FAMILY_LABEL,
        "joint_iid_rank80": RANK80_FAMILY_LABEL,
    }
    suffix = "; 5 epochs" if family == "joint_iid_rank80" else ""
    return f"{labels[family]} (H={_capacity_text(capacity)}{suffix})"


def _architecture_scaling_rows(
    summaries: Sequence[Mapping[str, object]],
    histories: Sequence[Mapping[str, object]],
    linear_summary: Mapping[str, object] | None,
    linear_history: Sequence[Mapping[str, object]],
    rank80_summary: Mapping[str, object] | None,
    rank80_history: Sequence[Mapping[str, object]],
    replay: ArchitectureReplayReport | None,
) -> tuple[dict[str, object], ...]:
    """Build one common selected/fixed schema across the three architectures."""
    rows = []
    for summary in (row for row in summaries if row["adapt_lora"]):
        capacity = int(summary["historical_capacity"])
        fixed = next(
            row
            for row in histories
            if row["adapt_lora"]
            and int(row["historical_capacity"]) == capacity
            and int(row["epoch"]) == 5
        )
        rows.append(
            {
                "checkpoint_rule": "minimum_validation_nll_over_50_epochs",
                "condition": _architecture_condition("macro_token", capacity),
                "epochs": 50,
                "family": "macro_token",
                "fixed_epoch": 5,
                "fixed_image_presentations": 5 * int(summary["training_examples"]),
                "fixed_validation_accuracy": float(fixed["validation_accuracy"]),
                "fixed_validation_nll": float(fixed["validation_nll"]),
                "historical_capacity": capacity,
                "image_presentations": int(summary["image_presentations"]),
                "max_accuracy_epoch": int(summary["max_accuracy_epoch"]),
                "max_validation_accuracy": float(
                    summary["max_validation_accuracy"]
                ),
                "nll_at_max_accuracy": float(
                    summary["validation_nll_at_max_accuracy"]
                ),
                "peak_vram_bytes": int(summary["peak_vram_bytes"]),
                "selected_accuracy": float(
                    summary["validation_accuracy_at_best_nll"]
                ),
                "selected_epoch": int(summary["best_nll_epoch"]),
                "selected_nll": float(summary["validation_nll_minimum"]),
                "trainable_parameters": int(summary["trainable_parameters"]),
                "training_examples": int(summary["training_examples"]),
                "wall_seconds": float(summary["wall_seconds"]),
            }
        )
    if linear_summary is not None:
        rows.append(
            {
                "checkpoint_rule": "minimum_validation_nll_over_50_epochs",
                "condition": _architecture_condition("single_affine", 11_827),
                "epochs": 50,
                "family": "single_affine",
                "fixed_epoch": 5,
                "fixed_image_presentations": 60_970,
                "fixed_validation_accuracy": float(
                    linear_summary["fixed_validation_accuracy"]
                ),
                "fixed_validation_nll": float(
                    linear_summary["fixed_validation_nll"]
                ),
                "historical_capacity": 11_827,
                "image_presentations": int(linear_summary["image_presentations"]),
                "max_accuracy_epoch": int(linear_summary["max_accuracy_epoch"]),
                "max_validation_accuracy": float(
                    linear_summary["max_validation_accuracy"]
                ),
                "nll_at_max_accuracy": float(
                    linear_summary["validation_nll_at_max_accuracy"]
                ),
                "peak_vram_bytes": int(linear_summary["peak_vram_bytes"]),
                "selected_accuracy": float(
                    linear_summary["validation_accuracy_at_best_nll"]
                ),
                "selected_epoch": int(linear_summary["best_nll_epoch"]),
                "selected_nll": float(linear_summary["validation_nll_minimum"]),
                "trainable_parameters": int(linear_summary["trainable_parameters"]),
                "training_examples": int(linear_summary["training_examples"]),
                "wall_seconds": float(linear_summary["wall_seconds"]),
            }
        )
    if rank80_summary is not None:
        maximum = max(
            rank80_history,
            key=lambda row: (
                float(row["validation_accuracy"]),
                -int(row["epoch"]),
            ),
        )
        rows.append(
            {
                "checkpoint_rule": "fixed_epoch_5_primary;minimum_nll_diagnostic",
                "condition": _architecture_condition("joint_iid_rank80", 11_827),
                "epochs": 5,
                "family": "joint_iid_rank80",
                "fixed_epoch": 5,
                "fixed_image_presentations": 60_970,
                "fixed_validation_accuracy": float(
                    rank80_summary["fixed_validation_accuracy"]
                ),
                "fixed_validation_nll": float(
                    rank80_summary["fixed_validation_nll"]
                ),
                "historical_capacity": 11_827,
                "image_presentations": int(rank80_summary["image_presentations"]),
                "max_accuracy_epoch": int(maximum["epoch"]),
                "max_validation_accuracy": float(maximum["validation_accuracy"]),
                "nll_at_max_accuracy": float(maximum["validation_nll"]),
                "peak_vram_bytes": int(rank80_summary["peak_vram_bytes"]),
                "selected_accuracy": float(
                    rank80_summary["best_validation_accuracy"]
                ),
                "selected_epoch": int(rank80_summary["best_epoch"]),
                "selected_nll": float(rank80_summary["best_validation_nll"]),
                "trainable_parameters": int(rank80_summary["trainable_parameters"]),
                "training_examples": 12_194,
                "wall_seconds": float(rank80_summary["wall_seconds"]),
            }
        )
    if replay is not None:
        rows.extend(
            {
                **summary,
                "condition": _architecture_condition(
                    str(summary["family"]),
                    int(summary["historical_capacity"]),
                ),
            }
            for summary in replay.summaries
        )
    family_order = {
        "macro_token": 0,
        "single_affine": 1,
        "joint_iid_rank80": 2,
    }
    projected = tuple(
        sorted(
            rows,
            key=lambda row: (
                family_order[str(row["family"])],
                int(row["historical_capacity"]),
            ),
        )
    )
    expected_counts = {
        "macro_token": 5,
        "single_affine": 5 if replay is not None else int(linear_summary is not None),
        "joint_iid_rank80": 5 if replay is not None else int(rank80_summary is not None),
    }
    if any(
        sum(row["family"] == family for row in projected) != count
        for family, count in expected_counts.items()
    ):
        raise ValueError("architecture scaling rows are incomplete or duplicated")
    return projected


def _architecture_history_rows(
    histories: Sequence[Mapping[str, object]],
    linear_history: Sequence[Mapping[str, object]],
    rank80_history: Sequence[Mapping[str, object]],
    replay: ArchitectureReplayReport | None,
) -> tuple[dict[str, object], ...]:
    """Label all per-epoch trajectories with one family and H schema."""
    rows = [
        {**row, "family": "macro_token"}
        for row in histories
        if row["adapt_lora"]
    ]
    rows.extend(
        {**row, "family": "single_affine", "historical_capacity": 11_827}
        for row in linear_history
    )
    rows.extend(
        {**row, "family": "joint_iid_rank80", "historical_capacity": 11_827}
        for row in rank80_history
    )
    if replay is not None:
        rows.extend(replay.histories)
    return tuple(rows)


def _architecture_displacement_rows(
    macro_displacements: Sequence[Mapping[str, object]],
    linear_control: Mapping[str, object] | None,
    replay: ArchitectureReplayReport | None,
) -> tuple[dict[str, object], ...]:
    """Return matched macro and affine adapter movements over all H values."""
    rows = [
        {**row, "family": "macro_token"}
        for row in macro_displacements
        if row["adapt_lora"]
    ]
    if linear_control is not None:
        rows.extend(
            {
                **dict(displacement),
                "family": "single_affine",
                "historical_capacity": 11_827,
            }
            for displacement in linear_control["displacements"]
        )
    if replay is not None:
        rows.extend(replay.displacements)
    return tuple(rows)


def _history(run: Path, cell: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line)
        for line in (run / str(cell["history"])).read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(rows) != 50:
        raise ValueError("frontier adaptation history is incomplete")
    return rows


def _summary_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    references = dict(result["references"])
    joint = dict(references["joint_iid"])
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        fit = dict(cell["fit"])
        history = _history(run, cell)
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        image_presentations = int(fit["image_presentations"])
        if image_presentations % len(history):
            raise ValueError("cell image presentations do not encode full epochs")
        simultaneous = tuple(
            row
            for row in history
            if float(row["validation_accuracy"]) >= float(joint["accuracy"])
            and float(row["validation_nll"]) <= float(joint["nll"])
        )
        rows.append(
            {
                "accuracy_gap_to_joint_at_min_nll_pp": float(
                    fit["validation_accuracy_at_best_nll"]
                )
                - float(joint["accuracy"]),
                "adapt_lora": adapted,
                "best_nll_epoch": int(fit["best_nll_epoch"]),
                "condition": (
                    CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
                ),
                "historical_capacity": capacity,
                "image_presentations": image_presentations,
                "max_accuracy_epoch": int(fit["max_accuracy_epoch"]),
                "max_validation_accuracy": float(fit["max_validation_accuracy"]),
                "nll_gap_to_joint": float(fit["best_validation_nll"])
                - float(joint["nll"]),
                "peak_vram_bytes": int(fit["peak_vram_bytes"]),
                "simultaneous_joint_match_epoch": (
                    None if not simultaneous else int(simultaneous[0]["epoch"])
                ),
                "train_accuracy_at_best": float(fit["train_accuracy_at_best"]),
                "train_nll_at_best": float(fit["train_nll_at_best"]),
                "trainable_parameters": int(fit["trainable_parameters"]),
                "training_examples": image_presentations // len(history),
                "validation_accuracy_at_best_nll": float(
                    fit["validation_accuracy_at_best_nll"]
                ),
                "validation_nll_at_max_accuracy": float(
                    fit["validation_nll_at_max_accuracy"]
                ),
                "validation_nll_minimum": float(fit["best_validation_nll"]),
                "wall_seconds": float(fit["wall_seconds"]),
            }
        )
    return tuple(rows)


def _history_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        label = CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
        rows.extend(
            {
                "adapt_lora": adapted,
                "condition": label,
                "epoch": int(row["epoch"]),
                "gradient_norm_mean": float(row["gradient_norm_mean"]),
                "historical_capacity": capacity,
                "image_presentations": int(row["image_presentations"]),
                "lora_learning_rate": row["lora_learning_rate"],
                "macro_learning_rate": float(row["macro_learning_rate"]),
                "optimizer_steps": int(row["optimizer_steps"]),
                "train_objective_accuracy": float(row["train_objective_accuracy"]),
                "train_objective_nll": float(row["train_objective_nll"]),
                "validation_accuracy": float(row["validation_accuracy"]),
                "validation_nll": float(row["validation_nll"]),
                "wall_seconds": float(row["wall_seconds"]),
            }
            for row in _history(run, cell)
        )
    return tuple(rows)


def _displacement_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        label = CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
        rows.extend(
            {
                **dict(displacement),
                "adapt_lora": adapted,
                "condition": label,
                "historical_capacity": capacity,
            }
            for displacement in cell["displacements"]
        )
    return tuple(rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_table_family(
    report_root: Path, name: str, rows: Sequence[Mapping[str, object]]
) -> None:
    projected = tuple(dict(row) for row in rows)
    _write_csv(report_root / f"{name}.csv", projected)
    atomic_write(
        report_root / f"{name}.json", canonical_json_bytes({"rows": projected})
    )
    try:
        import pandas as pd

        pd.DataFrame(projected).to_parquet(
            report_root / f"{name}.parquet", index=False
        )
    except ImportError:  # pragma: no cover - vision environment gate
        pass


def _plot_primary(
    path: Path,
    scaling_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "macro_token": (MACRO_FAMILY_LABEL, "#2166ac", "o", "-"),
        "single_affine": (LINEAR_FAMILY_LABEL, "#008b8b", "*", "--"),
        "joint_iid_rank80": (RANK80_FAMILY_LABEL, "#1b7837", "s", "-."),
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True)
    panels = (
        (axes[0, 0], "selected_accuracy", "Validation accuracy (%)"),
        (axes[0, 1], "selected_nll", "Validation NLL"),
        (axes[1, 0], "fixed_validation_accuracy", "Validation accuracy (%)"),
        (axes[1, 1], "fixed_validation_nll", "Validation NLL"),
    )
    for axis, metric, ylabel in panels:
        families = (
            ("macro_token", "single_affine")
            if metric.startswith("selected")
            else tuple(styles)
        )
        for family in families:
            label, color, marker, linestyle = styles[family]
            rows = tuple(
                row for row in scaling_rows if row["family"] == family
            )
            axis.plot(
                [int(row["historical_capacity"]) for row in rows],
                [float(row[metric]) for row in rows],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=7 if marker == "*" else 4.5,
                linewidth=2.0,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            [1_024, 2_048, 4_096, 8_192, 11_827],
            ["1k", "2k", "4k", "8k", "full\n11,827"],
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0, 0].set_title("Minimum-NLL selection: accuracy")
    axes[0, 1].set_title("Minimum-NLL selection: NLL")
    axes[1, 0].set_title("Fixed epoch 5: accuracy")
    axes[1, 1].set_title("Fixed epoch 5: NLL")
    for axis in axes[1]:
        axis.set_xlabel("Historical replay identities (H)")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    figure.suptitle("Replay scaling under selected and matched five-pass checkpoints")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(
    path: Path,
    histories: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        1024: "#4393c3",
        2048: "#2166ac",
        4096: "#f4a582",
        8192: "#e08214",
        11827: "#b2182b",
    }
    families = (
        ("macro_token", "Macro-token frontier"),
        ("single_affine", "Single-affine frontier"),
        ("joint_iid_rank80", "Joint IID, rank 80"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(11.6, 8.0))
    for row_index, (family, family_label) in enumerate(families):
        for capacity, color in colors.items():
            curve = tuple(
                row
                for row in histories
                if row["family"] == family
                and int(row["historical_capacity"]) == capacity
            )
            if not curve:
                continue
            label = _capacity_text(capacity)
            for axis, metric in zip(
                axes[row_index],
                ("validation_accuracy", "validation_nll"),
                strict=True,
            ):
                axis.plot(
                    [int(row["epoch"]) for row in curve],
                    [float(row[metric]) for row in curve],
                    color=color,
                    linewidth=1.7,
                    label=f"H={label}",
                )
        axes[row_index, 0].set_ylabel(f"{family_label}\naccuracy (%)")
        axes[row_index, 1].set_ylabel(f"{family_label}\nNLL")
        for axis in axes[row_index]:
            axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    axes[0, 0].set_title("Validation accuracy")
    axes[0, 1].set_title("Validation negative log likelihood")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.0,
    )
    figure.suptitle("Per-architecture learning curves across replay populations")
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_displacements(
    path: Path, displacements: Sequence[Mapping[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    families = (
        ("macro_token", "Macro-token frontier"),
        ("single_affine", "Single-affine frontier"),
    )
    colors = ("#2166ac", "#e08214", "#1b7837", "#b2182b", "#7b3294")
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for axis, (family, title) in zip(axes, families, strict=True):
        family_rows = tuple(
            row for row in displacements if row["family"] == family
        )
        for level, color in enumerate(colors):
            rows = tuple(
                sorted(
                    (row for row in family_rows if int(row["level"]) == level),
                    key=lambda row: int(row["historical_capacity"]),
                )
            )
            if not rows:
                continue
            tasks = rows[0]["represented_task_ids"]
            label = (
                f"level {level}, task {int(tasks[0]) + 1}"
                if len(tasks) == 1
                else f"level {level}, tasks {int(tasks[0]) + 1}-{int(tasks[-1]) + 1}"
            )
            axis.plot(
                [int(row["historical_capacity"]) for row in rows],
                [float(row["dense_update_relative_change"]) for row in rows],
                color=color,
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            [1_024, 2_048, 4_096, 8_192, 11_827],
            ["1k", "2k", "4k", "8k", "full\n11,827"],
        )
        axis.set_xlabel("Historical replay identities (H)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Relative dense LoRA-update change")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.8,
    )
    figure.suptitle("How far each sealed frontier adapter moved")
    figure.tight_layout(rect=(0, 0.13, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_frontier(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    intervals = ((31, 31), (29, 30), (25, 28), (17, 24), (1, 16))
    figure, axis = plt.subplots(figsize=(10.2, 4.6))
    for level, (first, last) in enumerate(intervals):
        y = 4.3 - 0.8 * level
        label = f"level {level}: task {first}" if first == last else f"level {level}: tasks {first}-{last}"
        box = FancyBboxPatch(
            (0.4, y - 0.25),
            3.6,
            0.5,
            boxstyle="round,pad=0.04",
            facecolor="#d9eaf4",
            edgecolor="#2166ac",
            linewidth=1.3,
        )
        axis.add_patch(box)
        axis.text(2.2, y, label, ha="center", va="center", fontsize=10)
        axis.add_patch(
            FancyArrowPatch(
                (4.05, y),
                (6.0, 2.7),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#666666",
                linewidth=1.1,
            )
        )
    macro = FancyBboxPatch(
        (6.05, 2.2),
        3.3,
        1.0,
        boxstyle="round,pad=0.06",
        facecolor="#fddbc7",
        edgecolor="#b2182b",
        linewidth=1.5,
    )
    axis.add_patch(macro)
    axis.text(7.7, 2.7, "shared macro-token\nintegrator", ha="center", va="center", fontsize=11)
    axis.text(2.2, 0.45, "Each arrow carries 197 adapted 768-value tokens plus local affine scores.", ha="center", fontsize=9)
    axis.text(7.7, 1.6, "124-way task-free prediction", ha="center", fontsize=9)
    axis.set(xlim=(0, 9.8), ylim=(0.1, 4.9), title="Task-31 fragmented frontier under joint adaptation")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _architecture_tables_html(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    """Render checkpoint-explicit replay tables for the three model families."""
    indexed = {
        (str(row["family"]), int(row["historical_capacity"])): row
        for row in rows
    }
    capacities = tuple(
        sorted({int(row["historical_capacity"]) for row in rows})
    )

    def selected_cells(family: str, capacity: int) -> str:
        row = indexed.get((family, capacity))
        if row is None:
            return "<td>n/a</td><td>n/a</td><td>n/a</td>"
        return (
            f"<td>{int(row['selected_epoch'])}</td>"
            f"<td>{float(row['selected_accuracy']):.3f}%</td>"
            f"<td>{float(row['selected_nll']):.4f}</td>"
        )

    selected_body = "".join(
        "<tr>"
        f"<td>{escape(_capacity_text(capacity))}</td>"
        f"{selected_cells('macro_token', capacity)}"
        f"{selected_cells('single_affine', capacity)}"
        "</tr>"
        for capacity in capacities
    )
    selected = (
        "<table><thead><tr><th rowspan=\"2\">H</th>"
        f"<th colspan=\"3\">{escape(MACRO_FAMILY_LABEL)}</th>"
        f"<th colspan=\"3\">{escape(LINEAR_FAMILY_LABEL)}</th></tr>"
        "<tr><th>Selected epoch</th><th>Accuracy</th><th>NLL</th>"
        "<th>Selected epoch</th><th>Accuracy</th><th>NLL</th></tr></thead>"
        f"<tbody>{selected_body}</tbody></table>"
    )

    def fixed_cells(family: str, capacity: int) -> str:
        row = indexed.get((family, capacity))
        if row is None:
            return "<td>n/a</td><td>n/a</td>"
        return (
            f"<td>{float(row['fixed_validation_accuracy']):.3f}%</td>"
            f"<td>{float(row['fixed_validation_nll']):.4f}</td>"
        )

    fixed_body = "".join(
        "<tr>"
        f"<td>{escape(_capacity_text(capacity))}</td>"
        f"<td>{capacity + 367:,}</td>"
        f"{fixed_cells('macro_token', capacity)}"
        f"{fixed_cells('single_affine', capacity)}"
        f"{fixed_cells('joint_iid_rank80', capacity)}"
        "</tr>"
        for capacity in capacities
    )
    fixed = (
        "<table><thead><tr><th rowspan=\"2\">H</th>"
        "<th rowspan=\"2\">Fit identities</th>"
        f"<th colspan=\"2\">{escape(MACRO_FAMILY_LABEL)}</th>"
        f"<th colspan=\"2\">{escape(LINEAR_FAMILY_LABEL)}</th>"
        f"<th colspan=\"2\">{escape(RANK80_FAMILY_LABEL)}</th></tr>"
        "<tr><th>Accuracy</th><th>NLL</th><th>Accuracy</th><th>NLL</th>"
        "<th>Accuracy</th><th>NLL</th></tr></thead>"
        f"<tbody>{fixed_body}</tbody></table>"
    )
    return selected, fixed


def _architecture_tables_markdown(
    rows: Sequence[Mapping[str, object]],
) -> str:
    """Render the same two checkpoint-explicit tables for Markdown readers."""
    indexed = {
        (str(row["family"]), int(row["historical_capacity"])): row
        for row in rows
    }
    capacities = tuple(
        sorted({int(row["historical_capacity"]) for row in rows})
    )

    def selected(family: str, capacity: int) -> str:
        row = indexed.get((family, capacity))
        if row is None:
            return "n/a"
        return (
            f"e{int(row['selected_epoch'])}: "
            f"{float(row['selected_accuracy']):.3f}% / "
            f"{float(row['selected_nll']):.4f}"
        )

    def fixed(family: str, capacity: int) -> str:
        row = indexed.get((family, capacity))
        if row is None:
            return "n/a"
        return (
            f"{float(row['fixed_validation_accuracy']):.3f}% / "
            f"{float(row['fixed_validation_nll']):.4f}"
        )

    selected_rows = "\n".join(
        f"| {_capacity_text(capacity)} | {selected('macro_token', capacity)} | "
        f"{selected('single_affine', capacity)} |"
        for capacity in capacities
    )
    fixed_rows = "\n".join(
        f"| {_capacity_text(capacity)} | {capacity + 367:,} | "
        f"{fixed('macro_token', capacity)} | "
        f"{fixed('single_affine', capacity)} | "
        f"{fixed('joint_iid_rank80', capacity)} |"
        for capacity in capacities
    )
    return f"""### Minimum-NLL-selected frontier checkpoints

| H | Macro-token frontier | Single-affine frontier |
|---:|:---|:---|
{selected_rows}

Each cell is `epoch: accuracy / NLL`. Rank 80 is omitted here because its
predeclared primary endpoint is epoch five, not a 50-epoch selected checkpoint.

### Fixed epoch-five checkpoints

| H | Fit identities | Macro-token frontier | Single-affine frontier | Joint IID, rank 80 |
|---:|---:|:---|:---|:---|
{fixed_rows}

Each metric cell is `accuracy / NLL`. All three models have completed five
passes over exactly the same identity set within each row.
"""


def write_frontier_adaptation_report(run: str | Path) -> Path:
    """Generate compact ledgers, plots, Markdown, HTML, and a validated PDF."""
    run_path = Path(run).resolve()
    result = _validated_result(run_path)
    report_root = run_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    capacity_controls = tuple(
        (specification, control)
        for specification in JOINT_CAPACITY_CONTROLS
        if (
            control := _validated_capacity_control(
                run_path, result, specification
            )
        )
        is not None
    )
    capacity_summaries = tuple(
        (specification, _capacity_summary(control, specification))
        for specification, control in capacity_controls
    )
    capacity_histories = tuple(
        (specification, _capacity_history(run_path, control, specification))
        for specification, control in capacity_controls
    )
    linear_control = _validated_linear_preclassifier_control(run_path, result)
    linear_summary = (
        None
        if linear_control is None
        else _linear_preclassifier_summary(linear_control)
    )
    linear_history = (
        ()
        if linear_control is None
        else _linear_preclassifier_history(run_path, linear_control)
    )
    summaries = _summary_rows(run_path, result)
    histories = _history_rows(run_path, result)
    displacements = _displacement_rows(result)
    rank80_summary = next(
        (
            summary
            for specification, summary in capacity_summaries
            if specification.rank == 80
        ),
        None,
    )
    rank80_history = next(
        (
            rows
            for specification, rows in capacity_histories
            if specification.rank == 80
        ),
        (),
    )
    architecture_replay = (
        None
        if linear_control is None or rank80_summary is None
        else load_architecture_replay_report(
            run_path,
            str(result["content_hash"]),
            {
                "single_affine": str(linear_control["content_hash"]),
                "joint_iid_rank80": str(
                    next(
                        control["content_hash"]
                        for specification, control in capacity_controls
                        if specification.rank == 80
                    )
                ),
            },
        )
    )
    architecture_scaling = _architecture_scaling_rows(
        summaries,
        histories,
        linear_summary,
        linear_history,
        rank80_summary,
        rank80_history,
        architecture_replay,
    )
    architecture_histories = _architecture_history_rows(
        histories,
        linear_history,
        rank80_history,
        architecture_replay,
    )
    architecture_displacements = _architecture_displacement_rows(
        displacements,
        linear_control,
        architecture_replay,
    )
    for name, rows in (
        ("condition_summary", summaries),
        ("epoch_history", histories),
        ("adapter_displacements", displacements),
    ):
        _write_table_family(report_root, name, rows)
    for specification, summary in capacity_summaries:
        history = next(
            rows
            for history_specification, rows in capacity_histories
            if history_specification == specification
        )
        _write_table_family(
            report_root, f"{specification.table_name}_summary", (summary,)
        )
        _write_table_family(
            report_root, f"{specification.table_name}_history", history
        )
    if linear_summary is not None:
        _write_table_family(
            report_root, "frontier_linear_preclassifier_summary", (linear_summary,)
        )
        _write_table_family(
            report_root, "frontier_linear_preclassifier_history", linear_history
        )
    for name, rows in (
        ("architecture_replay_summary", architecture_scaling),
        ("architecture_replay_history", architecture_histories),
        ("architecture_replay_displacements", architecture_displacements),
    ):
        _write_table_family(report_root, name, rows)
    references = dict(result["references"])
    _plot_primary(report_root / "accuracy_nll_vs_h.png", architecture_scaling)
    _plot_learning_curves(
        report_root / "validation_learning_curves.png",
        architecture_histories,
    )
    _plot_displacements(
        report_root / "adapter_displacements.png", architecture_displacements
    )
    _plot_frontier(report_root / "stage31_frontier.png")
    architecture_selected_table, architecture_fixed_table = (
        _architecture_tables_html(architecture_scaling)
    )
    architecture_markdown = _architecture_tables_markdown(
        architecture_scaling
    )
    if architecture_replay is None:
        architecture_crossover_markdown = ""
        architecture_crossover_html = ""
    else:
        architecture_by_key = {
            (str(row["family"]), int(row["historical_capacity"])): row
            for row in architecture_scaling
        }
        macro_1024 = architecture_by_key[("macro_token", 1_024)]
        macro_2048 = architecture_by_key[("macro_token", 2_048)]
        macro_4096 = architecture_by_key[("macro_token", 4_096)]
        macro_8192 = architecture_by_key[("macro_token", 8_192)]
        macro_full = architecture_by_key[("macro_token", 11_827)]
        affine_1024 = architecture_by_key[("single_affine", 1_024)]
        affine_2048 = architecture_by_key[("single_affine", 2_048)]
        affine_4096 = architecture_by_key[("single_affine", 4_096)]
        affine_8192 = architecture_by_key[("single_affine", 8_192)]
        affine_full = architecture_by_key[("single_affine", 11_827)]
        rank80_full = architecture_by_key[("joint_iid_rank80", 11_827)]
        fixed_affine_lead_1024 = float(
            affine_1024["fixed_validation_accuracy"]
        ) - float(macro_1024["fixed_validation_accuracy"])
        fixed_affine_lead_2048 = float(
            affine_2048["fixed_validation_accuracy"]
        ) - float(macro_2048["fixed_validation_accuracy"])
        fixed_macro_lead_4096 = float(
            macro_4096["fixed_validation_accuracy"]
        ) - float(affine_4096["fixed_validation_accuracy"])
        fixed_macro_lead_8192 = float(
            macro_8192["fixed_validation_accuracy"]
        ) - float(affine_8192["fixed_validation_accuracy"])
        architecture_crossover_markdown = f"""Under both checkpoint views, the
single-affine frontier wins at H=1,024 and H=2,048, while the macro-token
frontier wins from H=4,096 onward. At fixed epoch five, the affine accuracy
lead falls from {fixed_affine_lead_1024:.3f} to
{fixed_affine_lead_2048:.3f} points between H=1,024 and H=2,048; the macro
lead is then {fixed_macro_lead_4096:.3f} points at H=4,096 and
{fixed_macro_lead_8192:.3f} points at H=8,192. Rank-80 joint IID trails both
frontier heads at every truncated H. At full history it passes the affine head
({float(rank80_full['fixed_validation_accuracy']):.3f}% /
{float(rank80_full['fixed_validation_nll']):.4f} versus
{float(affine_full['fixed_validation_accuracy']):.3f}% /
{float(affine_full['fixed_validation_nll']):.4f}), but still trails the macro
head ({float(macro_full['fixed_validation_accuracy']):.3f}% /
{float(macro_full['fixed_validation_nll']):.4f})."""
        architecture_crossover_html = (
            '<div class="callout linear"><strong>Matched-H replay '
            "crossover.</strong> Under both checkpoint views, the single-affine "
            "frontier wins at H=1,024 and H=2,048; the macro-token frontier wins "
            "from H=4,096 onward. At fixed epoch five, the affine accuracy lead "
            f"falls from {fixed_affine_lead_1024:.3f} to "
            f"{fixed_affine_lead_2048:.3f} points, then the macro lead grows from "
            f"{fixed_macro_lead_4096:.3f} points at H=4,096 to "
            f"{fixed_macro_lead_8192:.3f} at H=8,192. Rank 80 trails both at "
            "every truncated H; at full history it passes the affine head but "
            "remains behind the macro head.</div>"
        )
    architecture_protocol_footer = (
        " | matched-H protocol "
        f"{str(architecture_replay.result['protocol_hash'])[:16]}..."
        if architecture_replay is not None
        else ""
    )
    adaptive = tuple(row for row in summaries if row["adapt_lora"])
    best_nll = min(adaptive, key=lambda row: float(row["validation_nll_minimum"]))
    best_accuracy = max(
        adaptive, key=lambda row: float(row["max_validation_accuracy"])
    )
    first_match = next(
        (
            row
            for row in adaptive
            if row["simultaneous_joint_match_epoch"] is not None
        ),
        None,
    )
    joint = dict(references["joint_iid"])
    cached = dict(references["frozen_macro_seed1993"])
    frozen_online = next(row for row in summaries if not row["adapt_lora"])
    full_adaptive = next(
        row
        for row in adaptive
        if int(row["historical_capacity"]) == 11827
    )
    result_cells = tuple(dict(row) for row in result["cells"])
    full_adaptive_cell = next(
        row
        for row in result_cells
        if bool(dict(row["cell"])["adapt_lora"])
        and int(dict(row["cell"])["historical_capacity"]) == 11827
    )
    frozen_online_cell = next(
        row
        for row in result_cells
        if not bool(dict(row["cell"])["adapt_lora"])
    )
    matched_epoch = int(joint["epochs"])
    full_at_matched_epoch = next(
        row
        for row in _history(run_path, full_adaptive_cell)
        if int(row["epoch"]) == matched_epoch
    )
    frozen_at_matched_epoch = next(
        row
        for row in _history(run_path, frozen_online_cell)
        if int(row["epoch"]) == matched_epoch
    )
    if linear_summary is None:
        linear_markdown = (
            "The single-layer pre-classifier condition has not yet been run."
        )
        linear_interpretation_html = f"<p>{escape(linear_markdown)}</p>"
        linear_capacity_row: tuple[
            tuple[str, str, int, int, int, float, float], ...
        ] = ()
    else:
        linear_selected_accuracy_gap = (
            float(linear_summary["validation_accuracy_at_best_nll"])
            - float(full_adaptive["validation_accuracy_at_best_nll"])
        )
        linear_selected_nll_gap = (
            float(linear_summary["validation_nll_minimum"])
            - float(full_adaptive["validation_nll_minimum"])
        )
        linear_fixed_accuracy_gap = (
            float(linear_summary["fixed_validation_accuracy"])
            - float(full_at_matched_epoch["validation_accuracy"])
        )
        linear_fixed_nll_gap = (
            float(linear_summary["fixed_validation_nll"])
            - float(full_at_matched_epoch["validation_nll"])
        )
        integrator_parameter_reduction = 100.0 * (
            1.0 - 768_200 / 12_055_496
        )
        linear_markdown = f"""The single-layer pre-classifier integrator reaches
**{float(linear_summary['validation_accuracy_at_best_nll']):.3f}% accuracy / {float(linear_summary['validation_nll_minimum']):.4f} NLL** at its
minimum-NLL checkpoint, epoch {int(linear_summary['best_nll_epoch'])}. Relative
to the macro-token full-history checkpoint selected by the same rule, this is
{linear_selected_accuracy_gap:+.3f} accuracy points and
{linear_selected_nll_gap:+.4f} NLL. At epoch five, where both have exactly
60,970 image presentations, the linear head reaches
{float(linear_summary['fixed_validation_accuracy']):.3f}% /
{float(linear_summary['fixed_validation_nll']):.4f}, a change of
{linear_fixed_accuracy_gap:+.3f} points and {linear_fixed_nll_gap:+.4f} NLL
from the macro head.

The linear head maps five concatenated 768-value node pre-classifier vectors
directly to 200 logits. Its exact-union initialization copies each frozen local
classifier into its owning input block; every cross-node block starts at zero.
It has 768,200 parameters, {integrator_parameter_reduction:.1f}% fewer than the
12,055,496-parameter macro head. Both conditions train the same five source
LoRAs from the same initial tensors and use the same full-fit data, augmentation,
optimizer schedule, and validation selection rule."""
        linear_interpretation_html = (
            "<p>The single-layer pre-classifier integrator reaches <strong>"
            f"{float(linear_summary['validation_accuracy_at_best_nll']):.3f}% "
            "accuracy / "
            f"{float(linear_summary['validation_nll_minimum']):.4f} NLL</strong> "
            f"at its minimum-NLL checkpoint, epoch "
            f"{int(linear_summary['best_nll_epoch'])}. Relative to the macro-token "
            "full-history checkpoint selected by the same rule, this changes "
            f"accuracy by {linear_selected_accuracy_gap:+.3f} points and NLL by "
            f"{linear_selected_nll_gap:+.4f}. At epoch five it reaches "
            f"{float(linear_summary['fixed_validation_accuracy']):.3f}% / "
            f"{float(linear_summary['fixed_validation_nll']):.4f}, changing accuracy "
            f"by {linear_fixed_accuracy_gap:+.3f} points and NLL by "
            f"{linear_fixed_nll_gap:+.4f} at identical exposure.</p>"
            "<p>The head is one affine map from five concatenated 768-value "
            "pre-classifier vectors to 200 logits. It starts as the exact union of "
            "the frozen local classifiers and can then learn cross-node weights. "
            f"Its 768,200 parameters are {integrator_parameter_reduction:.1f}% fewer "
            "than the macro head's 12,055,496. Both conditions start from the same "
            "five source LoRAs and use the same data, augmentation, optimizer "
            "schedule, and checkpoint rule.</p>"
        )
        linear_capacity_row = (
            (
                f"{_architecture_condition('single_affine', 11_827)} "
                "(epoch 5)",
                "5 x 16",
                6_635_520,
                768_200,
                7_403_720,
                float(linear_summary["fixed_validation_accuracy"]),
                float(linear_summary["fixed_validation_nll"]),
            ),
        )
    fit_examples = int(dict(result["training_seal"])["fit_examples"])
    matched_presentations = matched_epoch * fit_examples
    best_accuracy_gain = (
        float(best_nll["validation_accuracy_at_best_nll"])
        - float(joint["accuracy"])
    )
    best_nll_reduction = (
        float(joint["nll"])
        - float(best_nll["validation_nll_minimum"])
    )
    adaptation_accuracy_gain = (
        float(full_adaptive["validation_accuracy_at_best_nll"])
        - float(frozen_online["validation_accuracy_at_best_nll"])
    )
    adaptation_nll_reduction = (
        float(frozen_online["validation_nll_minimum"])
        - float(full_adaptive["validation_nll_minimum"])
    )
    rank_matched = next(
        (
            summary
            for specification, summary in capacity_summaries
            if specification.rank == 80
        ),
        None,
    )
    total_param_matched = next(
        (
            summary
            for specification, summary in capacity_summaries
            if specification.rank == 224
        ),
        None,
    )
    rank_history = next(
        (
            rows
            for specification, rows in capacity_histories
            if specification.rank == 80
        ),
        (),
    )
    total_param_history = next(
        (
            rows
            for specification, rows in capacity_histories
            if specification.rank == 224
        ),
        (),
    )
    if rank_matched is None:
        rank_markdown = (
            "The aggregate-rank joint-IID control has not yet been run. All existing "
            "same-split joint-IID references use one rank-16 adapter."
        )
        rank_table_html = f"<p>{escape(rank_markdown)}</p>"
        rank_interpretation_html = rank_table_html
        rank_history_html = ""
        rank_callout = ""
    else:
        rank_accuracy_change = (
            float(rank_matched["fixed_validation_accuracy"])
            - float(joint["accuracy"])
        )
        rank_nll_change = (
            float(rank_matched["fixed_validation_nll"]) - float(joint["nll"])
        )
        frontier_rank_accuracy_gap = (
            float(full_at_matched_epoch["validation_accuracy"])
            - float(rank_matched["fixed_validation_accuracy"])
        )
        frontier_rank_nll_gap = (
            float(full_at_matched_epoch["validation_nll"])
            - float(rank_matched["fixed_validation_nll"])
        )
        original_accuracy_gap = (
            float(full_at_matched_epoch["validation_accuracy"])
            - float(joint["accuracy"])
        )
        original_nll_advantage = (
            float(joint["nll"])
            - float(full_at_matched_epoch["validation_nll"])
        )
        rank_accuracy_fraction = 100.0 * rank_accuracy_change / original_accuracy_gap
        rank_nll_fraction = 100.0 * -rank_nll_change / original_nll_advantage
        if total_param_matched is None:
            total_markdown = ""
            total_interpretation_html = ""
            total_rows: tuple[
                tuple[str, str, int, int, int, float, float], ...
            ] = ()
            total_history_columns = ""
            total_history_cells = ("",) * len(rank_history)
            callout_label = "Aggregate-rank result."
            callout_text = (
                f"Rank-80 joint IID reaches <strong>"
                f"{float(rank_matched['fixed_validation_accuracy']):.3f}% / "
                f"{float(rank_matched['fixed_validation_nll']):.4f} NLL</strong> "
                "at epoch five. "
                f"The larger adapter recovers {rank_accuracy_fraction:.1f}% of "
                f"the frontier's accuracy advantage and {rank_nll_fraction:.1f}% "
                "of its NLL advantage over rank-16 joint IID."
            )
        else:
            total_accuracy_change = (
                float(total_param_matched["fixed_validation_accuracy"])
                - float(joint["accuracy"])
            )
            total_nll_change = (
                float(total_param_matched["fixed_validation_nll"])
                - float(joint["nll"])
            )
            rank_to_total_accuracy_change = (
                float(total_param_matched["fixed_validation_accuracy"])
                - float(rank_matched["fixed_validation_accuracy"])
            )
            rank_to_total_nll_change = (
                float(total_param_matched["fixed_validation_nll"])
                - float(rank_matched["fixed_validation_nll"])
            )
            frontier_total_accuracy_gap = (
                float(full_at_matched_epoch["validation_accuracy"])
                - float(total_param_matched["fixed_validation_accuracy"])
            )
            frontier_total_nll_gap = (
                float(full_at_matched_epoch["validation_nll"])
                - float(total_param_matched["fixed_validation_nll"])
            )
            total_accuracy_fraction = (
                100.0 * total_accuracy_change / original_accuracy_gap
            )
            total_nll_fraction = 100.0 * -total_nll_change / original_nll_advantage
            total_peak_accuracy_row = max(
                total_param_history,
                key=lambda row: float(row["validation_accuracy"]),
            )
            total_late_nll_increase = (
                float(total_param_matched["fixed_validation_nll"])
                - float(total_param_matched["best_validation_nll"])
            )
            parameter_difference = int(total_param_matched["parameter_difference"])
            parameter_difference_percent = (
                100.0
                * abs(parameter_difference)
                / int(total_param_matched["frontier_active_parameters"])
            )
            accuracy_comparison = (
                f"the adaptive frontier remains {frontier_total_accuracy_gap:.3f} "
                "accuracy points higher"
                if frontier_total_accuracy_gap >= 0.0
                else (
                    "rank-224 joint IID is "
                    f"{-frontier_total_accuracy_gap:.3f} accuracy points higher"
                )
            )
            nll_comparison = (
                f"the adaptive frontier is {-frontier_total_nll_gap:.4f} NLL lower"
                if frontier_total_nll_gap <= 0.0
                else (
                    "rank-224 joint IID is "
                    f"{frontier_total_nll_gap:.4f} NLL lower"
                )
            )
            total_markdown = f"""

The total-active-parameter control uses rank 224 and reaches
**{float(total_param_matched['fixed_validation_accuracy']):.3f}% accuracy / {float(total_param_matched['fixed_validation_nll']):.4f} NLL**
at epoch five. Its 18,674,812 trainable parameters are 16,204
({parameter_difference_percent:.3f}%) fewer than the adaptive frontier's
18,691,016, the closest possible match at integer rank. Relative to rank 80,
the extra capacity changes accuracy by {rank_to_total_accuracy_change:+.3f}
percentage points and NLL by {rank_to_total_nll_change:+.4f}. Relative to rank
16, rank 224 accounts for {total_accuracy_fraction:.1f}% of the adaptive
frontier's accuracy difference and {total_nll_fraction:.1f}% of its NLL
difference. At the same endpoint, {accuracy_comparison}, while
{nll_comparison}. The rank-224 minimum NLL is
{float(total_param_matched['best_validation_nll']):.4f} at epoch
{int(total_param_matched['best_epoch'])}; accuracy peaks at
{float(total_peak_accuracy_row['validation_accuracy']):.3f}% at epoch
{int(total_peak_accuracy_row['epoch'])}. Validation NLL then rises by
{total_late_nll_increase:.4f} through epoch five while the training objective
continues to improve, which is consistent with late overfitting or
miscalibration under the inherited rank-16 schedule."""
            total_interpretation_html = (
                f"<p>The rank-224 control reaches <strong>"
                f"{float(total_param_matched['fixed_validation_accuracy']):.3f}% "
                "accuracy / "
                f"{float(total_param_matched['fixed_validation_nll']):.4f} NLL"
                "</strong> at epoch five. Its 18,674,812 trainable parameters are "
                f"16,204 ({parameter_difference_percent:.3f}%) fewer than the "
                "adaptive frontier's 18,691,016, which is the closest possible "
                "integer-rank match. Relative to rank 80, it changes accuracy by "
                f"{rank_to_total_accuracy_change:+.3f} percentage points and NLL "
                f"by {rank_to_total_nll_change:+.4f}. At matched exposure, "
                f"{accuracy_comparison}, while {nll_comparison}. Its minimum NLL "
                f"is {float(total_param_matched['best_validation_nll']):.4f} at "
                f"epoch {int(total_param_matched['best_epoch'])}; accuracy peaks "
                f"at {float(total_peak_accuracy_row['validation_accuracy']):.3f}% "
                f"at epoch {int(total_peak_accuracy_row['epoch'])}. Validation NLL "
                f"then rises by {total_late_nll_increase:.4f} through epoch five "
                "while the training objective keeps improving, a pattern "
                "consistent with late overfitting or miscalibration under the "
                "inherited rank-16 schedule.</p>"
                "<p>This removes active parameter count as a large numerical "
                "mismatch, but it does not match factorization, initialization, "
                "training history, or inference compute. Rank 224 is one fresh "
                "adapter in one ViT path; the frontier starts from five pretrained "
                "adapters and combines five ViT paths with a macro transformer.</p>"
            )
            total_rows = (
                (
                    TOTAL_PARAM_MATCHED_LABEL,
                    "1 x 224",
                    int(total_param_matched["lora_parameters"]),
                    int(total_param_matched["classifier_parameters"]),
                    int(total_param_matched["trainable_parameters"]),
                    float(total_param_matched["fixed_validation_accuracy"]),
                    float(total_param_matched["fixed_validation_nll"]),
                ),
            )
            total_history_columns = (
                "<th>Rank-224 accuracy</th><th>Rank-224 NLL</th>"
            )
            total_history_cells = tuple(
                f"<td>{float(row['validation_accuracy']):.3f}%</td>"
                f"<td>{float(row['validation_nll']):.4f}</td>"
                for row in total_param_history
            )
            callout_label = "Total-active-parameter result."
            callout_text = (
                f"Rank-224 joint IID reaches <strong>"
                f"{float(total_param_matched['fixed_validation_accuracy']):.3f}% / "
                f"{float(total_param_matched['fixed_validation_nll']):.4f} NLL"
                "</strong> at epoch five with 0.087% fewer active parameters than "
                f"the adaptive frontier. At matched exposure, {accuracy_comparison} "
                f"and {nll_comparison}."
            )
        rank_markdown = f"""The new rank-80 joint-IID control reaches
**{float(rank_matched['fixed_validation_accuracy']):.3f}% accuracy / {float(rank_matched['fixed_validation_nll']):.4f} NLL**
at the fixed epoch-five endpoint. Increasing the joint adapter from rank 16 to
rank 80 raises accuracy by {rank_accuracy_change:.3f} percentage points and
lowers NLL by {-rank_nll_change:.4f}. This recovers {rank_accuracy_fraction:.1f}%
of the adaptive frontier's accuracy advantage and {rank_nll_fraction:.1f}% of
its NLL advantage over rank-16 joint IID. At the same five-pass checkpoint,
the adaptive frontier remains {frontier_rank_accuracy_gap:.3f} accuracy points
higher and {-frontier_rank_nll_gap:.4f} NLL lower than rank-80 joint IID.

Both sides expose exactly 6,635,520 trainable LoRA parameters and 60,970 image
presentations. That is the full extent of the match. The adaptive frontier also
trains a 12,055,496-parameter macro integrator, starts from five separately
pretrained rank-16 adapters, and executes five ViT paths. Rank-80 joint IID
starts one adapter from the standard zero-effect initialization, trains a
95,356-parameter classifier, and executes one ViT path. Its minimum NLL within
the same five epochs is {float(rank_matched['best_validation_nll']):.4f} with
{float(rank_matched['best_validation_accuracy']):.3f}% accuracy at epoch
{int(rank_matched['best_epoch'])}; this diagnostic does not replace the fixed
epoch-five comparison.{total_markdown}"""
        rank_rows = (
            (
                JOINT_LABEL,
                "1 x 16",
                1_327_104,
                95_356,
                1_422_460,
                float(joint["accuracy"]),
                float(joint["nll"]),
            ),
            (
                RANK_MATCHED_LABEL,
                "1 x 80",
                int(rank_matched["lora_parameters"]),
                int(rank_matched["classifier_parameters"]),
                int(rank_matched["trainable_parameters"]),
                float(rank_matched["fixed_validation_accuracy"]),
                float(rank_matched["fixed_validation_nll"]),
            ),
            *total_rows,
            *linear_capacity_row,
            (
                f"{_architecture_condition('macro_token', 11_827)} "
                "(epoch 5)",
                "5 x 16",
                int(rank_matched["lora_parameters"]),
                int(rank_matched["frontier_integrator_parameters"]),
                int(rank_matched["frontier_active_parameters"]),
                float(full_at_matched_epoch["validation_accuracy"]),
                float(full_at_matched_epoch["validation_nll"]),
            ),
        )
        rank_table_body = "".join(
            "<tr>"
            f"<td>{escape(condition)}</td><td>{layout}</td>"
            f"<td>{lora_parameters:,}</td><td>{other_parameters:,}</td>"
            f"<td>{total_parameters:,}</td>"
            f"<td>{accuracy:.3f}%</td><td>{nll:.4f}</td></tr>"
            for (
                condition,
                layout,
                lora_parameters,
                other_parameters,
                total_parameters,
                accuracy,
                nll,
            ) in rank_rows
        )
        rank_table_html = (
            "<table><thead><tr><th>Condition</th><th>LoRA layout</th>"
            "<th>LoRA parameters</th><th>Other trainable parameters</th>"
            "<th>Total active parameters</th>"
            "<th>Epoch-5 accuracy</th><th>Epoch-5 NLL</th></tr></thead>"
            f"<tbody>{rank_table_body}</tbody></table>"
        )
        if len(total_history_cells) not in {0, len(rank_history)}:
            raise ValueError("joint-capacity histories do not align")
        rank_history_body = "".join(
            "<tr>"
            f"<td>{int(row['epoch'])}</td>"
            f"<td>{float(row['validation_accuracy']):.3f}%</td>"
            f"<td>{float(row['validation_nll']):.4f}</td>"
            f"{total_cells}</tr>"
            for row, total_cells in zip(
                rank_history, total_history_cells, strict=True
            )
        )
        rank_history_html = (
            "<h2>Joint-IID capacity trajectories</h2>"
            "<table><thead><tr><th>Epoch</th><th>Rank-80 accuracy</th>"
            f"<th>Rank-80 NLL</th>{total_history_columns}</tr></thead>"
            f"<tbody>{rank_history_body}</tbody></table>"
            "<p class=\"small\">Epoch five is the predeclared comparison endpoint. "
            "Minimum-NLL epochs are retained only as convergence diagnostics.</p>"
        )
        rank_interpretation_html = (
            f"<p>The rank-80 joint-IID control reaches <strong>"
            f"{float(rank_matched['fixed_validation_accuracy']):.3f}% accuracy / "
            f"{float(rank_matched['fixed_validation_nll']):.4f} NLL</strong> at the "
            f"fixed epoch-five endpoint: {rank_accuracy_change:+.3f} accuracy points "
            f"and {rank_nll_change:+.4f} NLL versus rank 16. The frontier remains "
            f"{frontier_rank_accuracy_gap:.3f} points higher and "
            f"{-frontier_rank_nll_gap:.4f} NLL lower.</p>"
            "<p>Rank 80 and the frontier each expose 6,635,520 LoRA parameters and "
            "60,970 image presentations. The frontier additionally trains a "
            "12,055,496-parameter macro head, starts from five pretrained adapters, "
            "and executes five ViT paths; rank 80 starts one zero-effect adapter and "
            f"uses one path. Its diagnostic minimum NLL is "
            f"{float(rank_matched['best_validation_nll']):.4f} with "
            f"{float(rank_matched['best_validation_accuracy']):.3f}% accuracy at "
            f"epoch {int(rank_matched['best_epoch'])}.</p>"
            f"{total_interpretation_html}"
        )
        rank_callout = (
            f"<div class=\"callout rank\"><strong>{callout_label}</strong> "
            f"{callout_text}</div>"
        )
    if first_match is None:
        match_sentence = (
            "No adaptive checkpoint simultaneously reaches both rank-16 joint-IID metrics."
        )
    elif int(first_match["historical_capacity"]) < 11827:
        match_sentence = (
            f"{first_match['condition']} is the smallest H to reach both rank-16 "
            "joint-IID metrics, first doing so at epoch "
            f"{first_match['simultaneous_joint_match_epoch']}."
        )
    else:
        match_sentence = (
            "Only the full-fit adaptive condition reaches both rank-16 joint-IID metrics, "
            f"first doing so at epoch {first_match['simultaneous_joint_match_epoch']}."
        )
    markdown = f"""# ImageNet-R stage-31 frontier-LoRA adaptation

## Result

The minimum-NLL macro-token condition is **{best_nll['condition']}**, with
**{float(best_nll['validation_accuracy_at_best_nll']):.3f}% validation accuracy**
and **{float(best_nll['validation_nll_minimum']):.4f} NLL** at epoch
{int(best_nll['best_nll_epoch'])}. The largest observed macro-token accuracy is
**{float(best_accuracy['max_validation_accuracy']):.3f}%** from
**{best_accuracy['condition']}** at epoch {int(best_accuracy['max_accuracy_epoch'])},
where NLL is {float(best_accuracy['validation_nll_at_max_accuracy']):.4f}.
{match_sentence}

The rank-16 joint-IID reference is {float(joint['accuracy']):.3f}% / {float(joint['nll']):.4f}
NLL. The previous frozen cached macro result at the same seed is
{float(cached['accuracy']):.3f}% / {float(cached['nll']):.4f}; the new
augmentation-matched frozen control is
{float(frozen_online['validation_accuracy_at_best_nll']):.3f}% /
{float(frozen_online['validation_nll_minimum']):.4f}.

At the selected full-history checkpoint, LoRA adaptation gains
{adaptation_accuracy_gain:.3f} percentage points and reduces NLL by
{adaptation_nll_reduction:.4f} relative to the otherwise matched frozen-LoRA
control. At exactly {matched_epoch} full-fit passes ({matched_presentations:,}
image presentations), the adaptive frontier reaches
{float(full_at_matched_epoch['validation_accuracy']):.3f}% /
{float(full_at_matched_epoch['validation_nll']):.4f}; the frozen frontier reaches
{float(frozen_at_matched_epoch['validation_accuracy']):.3f}% /
{float(frozen_at_matched_epoch['validation_nll']):.4f}; and joint IID reaches
{float(joint['accuracy']):.3f}% / {float(joint['nll']):.4f}. The data exposure is
matched at this checkpoint. Compute is not: the frontier evaluates five
specialized ViTs plus the macro transformer, while joint IID evaluates one
ViT with one shared LoRA and classifier.

## Replay scaling across architectures

{architecture_markdown}

{architecture_crossover_markdown}

![Accuracy and NLL versus H](accuracy_nll_vs_h.png)

## Joint-IID capacity controls

{rank_markdown}

## Single-layer integrator ablation

{linear_markdown}

## What changed

The task-31 frontier contains five sealed rank-16 LoRAs over disjoint task
intervals. Every adaptive frontier cell starts from those exact tensors. The
base ViT and all five node classifiers stay frozen; the five node LoRAs and
either the macro-token or affine head train jointly from task-free inputs. The
affine head reads only the five final pre-classifier vectors. Rank-80 joint IID
instead starts one zero-effect adapter and a 124-way classifier. Every
population includes all 367 current-task images. H is the same nested uniform
hash-order prefix of the 11,827-image historical partition for all three
architectures, so maximum H is exactly the 12,194-image full fit. The 3,049
validation identities remain excluded from optimization.

![Stage-31 frontier](stage31_frontier.png)

## Optimization behavior

![Validation learning curves](validation_learning_curves.png)

![Adapter displacement](adapter_displacements.png)

Both frontier heads use the previous minimum-NLL winner: effective batch 64,
peak AdamW learning rate 3e-5, and 50 warmup-cosine epochs. Newly adaptive node
LoRAs use peak 5e-4 under that schedule. Rank-80 joint IID retains its original
five-epoch SGD recipe. Frontier checkpoint selection is minimum validation NLL;
all three architectures are also reported at fixed epoch five. Maximum
accuracy is a separately labeled diagnostic.

## Interpretation boundaries

This is one seed on a validation split, not a final test estimate. Repeated
epoch evaluation makes the maximum-accuracy statistic exploratory. The
full-fit frozen control isolates online augmentation and image forwarding from
LoRA adaptation. H cells differ in both unique identities and optimizer steps,
but within each H the fixed-epoch comparison uses the same identities and five
complete passes. No test identity was requested. Exact replay authenticated the
six original macro cells, eight new architecture-sweep cells, and all full-fit
controls without a new optimizer step, while leaving the source hierarchy
unchanged.
    """
    atomic_write(
        report_root / "REPORT.md", (markdown.rstrip() + "\n").encode("utf-8")
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R frontier-LoRA adaptation</title><style>
@page {{ size: Letter; margin: 0.55in; }}
* {{ box-sizing: border-box; }} body {{ font-family: Arial, sans-serif; color:#17202a; margin:0; line-height:1.35; font-size:10.5pt; }}
h1 {{ font-size:23pt; margin:0 0 5px; color:#16324f; }} h2 {{ font-size:15pt; color:#16324f; margin:16px 0 7px; }}
.lede {{ font-size:12pt; color:#425466; margin:0 0 12px; }} .callout {{ background:#eef5f9; border-left:4px solid #2166ac; padding:10px 13px; margin:9px 0 12px; }} .rank {{ background:#edf7ef; border-left-color:#1b7837; }} .linear {{ background:#e8f6f6; border-left-color:#008b8b; }}
figure {{ margin:10px 0 12px; break-inside:avoid; }} figure img {{ width:100%; max-height:5.6in; object-fit:contain; }} figcaption {{ color:#59636e; font-size:8.5pt; margin-top:3px; }}
table {{ width:100%; border-collapse:collapse; font-size:7.7pt; margin:8px 0 12px; }} th {{ background:#16324f; color:white; text-align:left; padding:5px; }} td {{ border-bottom:1px solid #ccd5dd; padding:5px; vertical-align:top; }}
.page {{ break-before:page; }} .small {{ font-size:9pt; color:#44515e; }} .dense {{ font-size:9.5pt; line-height:1.28; }} .dense h2 {{ margin:12px 0 6px; }} .dense p {{ margin:7px 0; }} footer {{ margin-top:12px; border-top:1px solid #ccd5dd; padding-top:5px; color:#687580; font-size:8pt; }}
</style></head><body>
<h1>ImageNet-R stage-31 frontier-LoRA adaptation</h1><p class="lede">Can jointly adapting five fragmented node representations close the same-split joint-IID accuracy and NLL gap?</p>
<div class="callout"><strong>Selected macro-token result.</strong> The minimum-NLL macro-token condition is {escape(str(best_nll['condition']))}: <strong>{float(best_nll['validation_accuracy_at_best_nll']):.3f}% accuracy / {float(best_nll['validation_nll_minimum']):.4f} NLL</strong>. That is {best_accuracy_gain:.3f} percentage points higher and {best_nll_reduction:.4f} NLL lower than the five-epoch rank-16 joint-IID reference. {escape(match_sentence)}</div>
{architecture_crossover_html}
<figure><img src="{_image_uri(report_root / 'accuracy_nll_vs_h.png')}" alt="Accuracy and NLL versus replay population"><figcaption>The upper row compares the 50-epoch minimum-NLL selections for the macro-token and single-affine frontiers. The lower row compares macro-token, single-affine, and rank-80 joint IID at fixed epoch five, after five passes over the identical fit identities at each H.</figcaption></figure>
<div class="page"><h2>Replay scaling: selected frontier checkpoints</h2>{architecture_selected_table}<p class="small">Macro-token and single-affine cells share the 50-epoch AdamW schedule and minimum-validation-NLL rule. Rank 80 is excluded because its frozen protocol has only five epochs and a fixed primary endpoint.</p>
<h2>Replay scaling: fixed epoch five</h2>{architecture_fixed_table}<p class="small">Within each H row, all three models see the same current-plus-history identity set for five complete passes. Their optimizer families and compute differ by design.</p></div>
<div class="page"><h2>Full-fit architecture and capacity comparison</h2>{rank_table_html}<p class="small">Every row uses exactly five complete passes over the 12,194-image full-fit population. "Other" means the trainable joint classifier, affine integrator, or macro integrator; frozen parameters are omitted. Rank 80 matches aggregate node-LoRA capacity. Rank 224 is the closest total-active-parameter match.</p>{rank_history_html}</div>
<div class="page"><h2>Architecture and experimental boundary</h2><p>The task-31 frontier has five nodes at levels 0-4, covering task intervals 31, 29-30, 25-28, 17-24, and 1-16. Each node supplies its own final 197 x 768 LoRA-adapted token sequence and immutable local affine scores to the macro transformer. The affine frontier instead concatenates only the five 768-value pre-classifier outputs and maps the resulting 3,840 values directly to 200 logits. Neither frontier head receives task IDs or labels.</p>
<figure><img src="{_image_uri(report_root / 'stage31_frontier.png')}" alt="Five nodes feeding the macro-token integrator"><figcaption>This diagram shows the macro condition: the base ViT and node classifiers are frozen, while five rank-16 LoRAs and the shared 12.06M-parameter macro head can move. The single-layer condition replaces that head with one affine map.</figcaption></figure>
<p>Every cell includes all 367 current-task images. H=1,024, 2,048, 4,096, 8,192, and 11,827 are nested prefixes of one deterministic uniform draw from the historical tasks without replacement. Maximum H therefore gives exactly all 12,194 fit identities. Frontier cells start independently from identical sealed node tensors; rank-80 cells start independently from the same seed and zero-effect LoRA initialization. The validation partition has 3,049 clean identities and never contributes gradients. No test image is opened.</p>
<p><strong>Compute boundary.</strong> Both adaptive frontier heads evaluate the same five node-specific ViTs for every image. The linear condition then runs one 768,200-parameter affine map; the macro condition runs its 12,055,496-parameter transformer. Joint IID evaluates one ViT with one shared LoRA and classifier. Their split and full-fit data exposure match, but deployment compute remains different.</p>
<h2>Why the frozen online control exists</h2><p>The previous frozen macro was trained from cached center-crop tokens. This run loads images online with deterministic random training augmentation. The purple full-fit control follows that new path while freezing the LoRAs, so its difference from the cached gray reference measures the pipeline/augmentation change rather than representation adaptation.</p></div>
<div class="page"><h2>Optimization behavior</h2><figure><img src="{_image_uri(report_root / 'validation_learning_curves.png')}" alt="Validation learning curves"><figcaption>Rows separate the macro-token frontier, single-affine frontier, and rank-80 joint-IID model. Colors mean the same H in every panel; frontier histories contain 50 epochs and rank-80 histories contain the five frozen-protocol epochs.</figcaption></figure>
<figure><img src="{_image_uri(report_root / 'adapter_displacements.png')}" alt="Relative LoRA update displacement"><figcaption>Scale-aware movement of each dense node-LoRA update from its sealed source value. Left is the macro-token frontier and right is the single-affine frontier under the same five H values.</figcaption></figure></div>
<div class="page dense"><h2>Interpretation</h2><p>Allowing the frontier LoRAs to move changes the result by {adaptation_accuracy_gain:.3f} percentage points and {adaptation_nll_reduction:.4f} NLL relative to the online frozen-LoRA control at their minimum-NLL checkpoints. H=4,096 is the smallest tested historical population to cross both rank-16 joint-IID values. At exactly {matched_epoch} full-fit passes ({matched_presentations:,} image presentations), adaptive full history is {float(full_at_matched_epoch['validation_accuracy']):.3f}% / {float(full_at_matched_epoch['validation_nll']):.4f}, frozen full history is {float(frozen_at_matched_epoch['validation_accuracy']):.3f}% / {float(frozen_at_matched_epoch['validation_nll']):.4f}, and rank-16 joint IID is {float(joint['accuracy']):.3f}% / {float(joint['nll']):.4f}. This isolates feature adaptation from training exposure, but not the frontier's extra model and deployment compute.</p>
<p>The replay-scaling tables make two different questions explicit. Minimum-NLL selection asks what each 50-epoch frontier fit can achieve after checkpoint selection. Fixed epoch five asks how the two frontiers and rank-80 joint IID compare after the same number of passes over exactly the same identities. The latter still does not equalize optimizer, initialization, parameter count beyond the LoRA-rank match, or inference compute.</p>
<h2>What the single-layer condition establishes</h2>{linear_interpretation_html}
<h2>What the capacity controls establish</h2>{rank_interpretation_html}
<p>The head schedule is the previous minimum-NLL winner: effective batch 64, peak AdamW LR 3e-5, 50 epochs, 5% warmup, and cosine decay. The LoRA peak LR 5e-4 is imported from the same-split joint recipe but run here in AdamW under the shared schedule. It has not been tuned for this coupled model.</p>
<p><strong>Limits.</strong> This is a one-seed validation screen. The H cells change both unique data and total optimizer updates. Maximum accuracy is exploratory because all 50 validation checkpoints are visible. A promising condition needs replication and a focused LoRA/head learning-rate audit before any locked-test use.</p>
<h2>Integrity and reuse</h2><p>The training seals record zero fit-validation overlap, zero test overlap, zero test evaluations, and unchanged source hierarchy files. Fresh-process replay authenticated all six original macro-frontier cells, all eight architecture-sweep cells, and the full-fit controls without taking an optimizer step. Large checkpoints and model weights remain local; compact histories, tables, plots, and protocol records are the report surface.</p>
<footer>Primary protocol {escape(str(result['content_hash']))[:16]}...{architecture_protocol_footer} | stage 31 | seed 1993 | generated from authenticated result and history ledgers</footer></div>
</body></html>"""
    atomic_write(report_root / "REPORT.html", html.encode("utf-8"))
    return render_integrator_pdf(report_root / "REPORT.html")


__all__ = [
    "CONDITION_LABELS",
    "FROZEN_CACHED_LABEL",
    "FROZEN_ONLINE_LABEL",
    "JOINT_LABEL",
    "JOINT_CAPACITY_CONTROLS",
    "LINEAR_PRECLASSIFIER_LABEL",
    "RANK_MATCHED_LABEL",
    "TOTAL_PARAM_MATCHED_LABEL",
    "write_frontier_adaptation_report",
]
