"""Authenticate and project the optional stage-31 architecture replay sweep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from apm.continual.artifacts import file_sha256, load_canonical_json, record_sha256


SWEEP_RESULT = Path("evaluations/frontier_architecture_replay_sweep.json")
FAMILIES = ("single_affine", "joint_iid_rank80")
CAPACITIES = (1_024, 2_048, 4_096, 8_192)


@dataclass(frozen=True, slots=True)
class ArchitectureReplayReport:
    """Authenticated cell records and reporting projections."""

    result: dict[str, object]
    cells: tuple[dict[str, object], ...]
    summaries: tuple[dict[str, object], ...]
    histories: tuple[dict[str, object], ...]
    displacements: tuple[dict[str, object], ...]


def _history(
    run: Path, cell: Mapping[str, object], family: str
) -> tuple[dict[str, object], ...]:
    path = (run / str(cell["history"])).resolve()
    expected_epochs = 50 if family == "single_affine" else 5
    if (
        run not in path.parents
        or not path.is_file()
        or file_sha256(path) != cell["history_sha256"]
    ):
        raise ValueError("architecture replay history is absent or changed")
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if tuple(int(row["epoch"]) for row in rows) != tuple(
        range(1, expected_epochs + 1)
    ):
        raise ValueError("architecture replay history is incomplete")
    return rows


def _summary(
    cell: Mapping[str, object], history: tuple[dict[str, object], ...]
) -> dict[str, object]:
    specification = dict(cell["cell"])
    population = dict(cell["population"])
    architecture = dict(cell["architecture"])
    fit = dict(cell["fit"])
    family = str(specification["family"])
    maximum = max(
        history,
        key=lambda row: (float(row["validation_accuracy"]), -int(row["epoch"])),
    )
    if family == "single_affine":
        selected_epoch = int(fit["best_nll_epoch"])
        selected_accuracy = float(fit["validation_accuracy_at_best_nll"])
        selected_nll = float(fit["best_validation_nll"])
        fixed_accuracy = float(fit["fixed_validation_accuracy"])
        fixed_nll = float(fit["fixed_validation_nll"])
        fixed_presentations = int(fit["fixed_image_presentations"])
        checkpoint_rule = "minimum_validation_nll_over_50_epochs"
    else:
        selected_epoch = int(fit["best_epoch"])
        selected_accuracy = float(fit["best_validation_accuracy"])
        selected_nll = float(fit["best_validation_nll"])
        fixed_accuracy = float(fit["fixed_validation_accuracy"])
        fixed_nll = float(fit["fixed_validation_nll"])
        fixed_presentations = 5 * int(population["training_examples"])
        checkpoint_rule = "fixed_epoch_5_primary;minimum_nll_diagnostic"
    return {
        "checkpoint_rule": checkpoint_rule,
        "epochs": int(fit["epochs"]),
        "family": family,
        "fixed_epoch": 5,
        "fixed_image_presentations": fixed_presentations,
        "fixed_validation_accuracy": fixed_accuracy,
        "fixed_validation_nll": fixed_nll,
        "historical_capacity": int(specification["historical_capacity"]),
        "image_presentations": int(fit["image_presentations"]),
        "max_accuracy_epoch": int(maximum["epoch"]),
        "max_validation_accuracy": float(maximum["validation_accuracy"]),
        "nll_at_max_accuracy": float(maximum["validation_nll"]),
        "peak_vram_bytes": int(fit["peak_vram_bytes"]),
        "selected_accuracy": selected_accuracy,
        "selected_epoch": selected_epoch,
        "selected_nll": selected_nll,
        "trainable_parameters": int(architecture["trainable_parameters"]),
        "training_examples": int(population["training_examples"]),
        "wall_seconds": float(fit["wall_seconds"]),
    }


def load_architecture_replay_report(
    run: str | Path,
    parent_result_hash: str,
    source_full_result_hashes: Mapping[str, str],
) -> ArchitectureReplayReport | None:
    """Load the optional sweep without requiring its local model tensors."""
    run_path = Path(run).resolve()
    aggregate_path = run_path / SWEEP_RESULT
    if not aggregate_path.is_file():
        return None
    result = load_canonical_json(aggregate_path)
    core = {key: value for key, value in result.items() if key != "content_hash"}
    protocol_path = (run_path / str(result.get("protocol", ""))).resolve()
    protocol = (
        load_canonical_json(protocol_path)
        if run_path in protocol_path.parents and protocol_path.is_file()
        else {}
    )
    protocol_core = {
        key: value for key, value in protocol.items() if key != "content_hash"
    }
    if (
        result.get("schema_version")
        != "imagenetr50-frontier-architecture-replay-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or result.get("parent_result_hash") != parent_result_hash
        or result.get("source_full_results") != dict(source_full_result_hashes)
        or result.get("historical_capacities") != list(CAPACITIES)
        or result.get("families") != list(FAMILIES)
        or result.get("test_evaluations") != 0
        or protocol.get("schema_version")
        != "imagenetr50-frontier-architecture-replay-protocol-v1"
        or protocol.get("content_hash") != record_sha256(protocol_core)
        or protocol.get("content_hash") != result.get("protocol_hash")
    ):
        raise ValueError("architecture replay aggregate does not authenticate")
    expected = {(family, capacity) for capacity in CAPACITIES for family in FAMILIES}
    cells = []
    summaries = []
    histories = []
    displacements = []
    observed: set[tuple[str, int]] = set()
    for reference_value in result["cells"]:
        reference = dict(reference_value)
        path = (run_path / str(reference["result"])).resolve()
        if (
            run_path not in path.parents
            or not path.is_file()
            or file_sha256(path) != reference["result_sha256"]
        ):
            raise ValueError("architecture replay cell reference changed")
        cell = load_canonical_json(path)
        cell_core = {
            key: value for key, value in cell.items() if key != "content_hash"
        }
        specification = dict(cell.get("cell", {}))
        population = dict(cell.get("population", {}))
        family = str(specification.get("family"))
        capacity = int(specification.get("historical_capacity", -1))
        key = (family, capacity)
        if (
            cell.get("schema_version")
            != "imagenetr50-frontier-architecture-replay-cell-v1"
            or cell.get("content_hash") != record_sha256(cell_core)
            or cell.get("content_hash") != reference["result_content_hash"]
            or reference.get("cell") != specification
            or cell.get("protocol_hash") != result["protocol_hash"]
            or cell.get("source_full_result_hash")
            != source_full_result_hashes.get(family)
            or cell.get("test_evaluations") != 0
            or key not in expected
            or key in observed
            or population.get("historical_capacity") != capacity
            or population.get("training_examples") != capacity + 367
        ):
            raise ValueError("architecture replay cell does not authenticate")
        history = _history(run_path, cell, family)
        observed.add(key)
        cells.append(cell)
        summaries.append(_summary(cell, history))
        histories.extend(
            {
                **row,
                "family": family,
                "historical_capacity": capacity,
            }
            for row in history
        )
        displacements.extend(
            {
                **dict(displacement),
                "family": family,
                "historical_capacity": capacity,
            }
            for displacement in cell.get("displacements", [])
        )
    if observed != expected:
        raise ValueError("architecture replay matrix is incomplete")
    order = {family: index for index, family in enumerate(FAMILIES)}
    return ArchitectureReplayReport(
        result,
        tuple(sorted(cells, key=lambda row: (
            int(dict(row["cell"])["historical_capacity"]),
            order[str(dict(row["cell"])["family"])],
        ))),
        tuple(sorted(summaries, key=lambda row: (
            int(row["historical_capacity"]), order[str(row["family"])]
        ))),
        tuple(histories),
        tuple(displacements),
    )


__all__ = [
    "ArchitectureReplayReport",
    "CAPACITIES",
    "FAMILIES",
    "SWEEP_RESULT",
    "load_architecture_replay_report",
]
