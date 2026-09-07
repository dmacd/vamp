from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.frontier_architecture_replay_config import (
    load_frontier_architecture_replay_config,
)
from apm.continual.vision.imagenetr.frontier_architecture_replay_reporting import (
    CAPACITIES,
    FAMILIES,
    load_architecture_replay_report,
)
from apm.continual.vision.imagenetr.frontier_architecture_replay_workflow import (
    ArchitectureReplayCell,
    FrontierArchitectureReplayProtocol,
    _material_paths,
    _replay_populations,
)
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    LINEAR_FAMILY_LABEL,
    MACRO_FAMILY_LABEL,
    RANK80_FAMILY_LABEL,
    _architecture_condition,
    _architecture_tables_html,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_config import (
    load_frontier_linear_preclassifier_config,
)


CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_architecture_replay_sweep_v14.yaml"
)


def _row(index: int) -> ImageRecord:
    identity = f"{index + 1:064x}"
    label = index % 124
    return ImageRecord(
        identity,
        f"class/image-{index}.jpg",
        f"train/class/image-{index}.jpg",
        identity,
        "class",
        label,
        label,
        label // 4,
        "train",
        f"{index + 1001:064x}",
        1,
    )


def test_architecture_replay_config_freezes_matrix_and_source_results() -> None:
    config = load_frontier_architecture_replay_config(CONFIG)
    assert config.historical_capacities == CAPACITIES
    assert config.current_task_examples == 367
    assert config.available_historical_examples == 11_827
    assert config.validation_examples == 3_049
    assert (config.linear_epochs, config.rank80_epochs) == (50, 5)
    assert config.linear_checkpoint_rule == "minimum_validation_nll"
    assert config.rank80_primary_endpoint == "fixed_epoch_5"
    assert file_sha256(config.linear_config) == config.linear_config_sha256
    assert file_sha256(config.rank80_config) == config.rank80_config_sha256
    with pytest.raises(ValueError, match="architecture replay sweep"):
        replace(config, historical_capacities=(1_024,))


def test_architecture_replay_cells_and_protocol_are_content_addressed() -> None:
    assert tuple(
        ArchitectureReplayCell(family, capacity, 1993).condition
        for capacity in CAPACITIES
        for family in FAMILIES
    ) == tuple(
        f"{family}_h{capacity}"
        for capacity in CAPACITIES
        for family in FAMILIES
    )
    with pytest.raises(ValueError, match="frozen matrix"):
        ArchitectureReplayCell("single_affine", 16_384, 1993)
    identities = tuple(f"{index:064x}" for index in range(1, 17))
    protocol = FrontierArchitectureReplayProtocol(*identities)
    assert len(protocol.content_hash) == 64
    assert replace(protocol, config_hash="f" * 64).content_hash != protocol.content_hash


def test_material_surface_includes_sweep_code_but_excludes_reporting() -> None:
    config = load_frontier_architecture_replay_config(CONFIG)
    linear = load_frontier_linear_preclassifier_config(config.linear_config)
    material = _material_paths(
        Path.cwd(),
        CONFIG.resolve(),
        config.linear_config,
        linear.parent_config,
        config.rank80_config,
    )
    names = {path.name for path in material}
    assert {
        "frontier_architecture_replay_config.py",
        "frontier_architecture_replay_workflow.py",
        "imagenetr50_frontier_architecture_replay_sweep_protocol.md",
        "run_frontier_architecture_replay_sweep_local.sh",
    } <= names
    assert "frontier_architecture_replay_reporting.py" not in names
    assert "frontier_adaptation_reporting.py" not in names
    assert all(path.is_file() for path in material)


def test_report_labels_and_checkpoint_tables_keep_architectures_distinct() -> None:
    rows = tuple(
        {
            "family": family,
            "historical_capacity": 1_024,
            "selected_epoch": 3,
            "selected_accuracy": 75.0,
            "selected_nll": 1.0,
            "fixed_validation_accuracy": 74.0,
            "fixed_validation_nll": 1.1,
        }
        for family in ("macro_token", "single_affine", "joint_iid_rank80")
    )
    selected, fixed = _architecture_tables_html(rows)
    assert _architecture_condition("macro_token", 1_024).startswith(
        MACRO_FAMILY_LABEL
    )
    assert _architecture_condition("single_affine", 1_024).startswith(
        LINEAR_FAMILY_LABEL
    )
    assert _architecture_condition("joint_iid_rank80", 1_024).startswith(
        RANK80_FAMILY_LABEL
    )
    assert MACRO_FAMILY_LABEL in selected
    assert LINEAR_FAMILY_LABEL in selected
    assert RANK80_FAMILY_LABEL not in selected
    assert all(label in fixed for label in (
        MACRO_FAMILY_LABEL,
        LINEAR_FAMILY_LABEL,
        RANK80_FAMILY_LABEL,
    ))


def test_population_reconstruction_uses_exact_nested_parent_prefixes() -> None:
    current = tuple(_row(index) for index in range(20, 24))
    historical = tuple(_row(index) for index in range(12))
    entries = []
    for capacity in (4, 8):
        rows = tuple(
            sorted((*current, *historical[:capacity]), key=lambda row: row.image_id)
        )
        entries.append(
            {
                "historical_capacity": capacity,
                "historical_image_ids_hash": record_sha256(
                    [row.image_id for row in historical[:capacity]]
                ),
                "image_ids_hash": record_sha256([row.image_id for row in rows]),
                "training_examples": len(rows),
            }
        )
    replay = {
        "current_image_ids": [row.image_id for row in current],
        "entries": entries,
        "nested_historical_image_ids": [row.image_id for row in historical],
    }
    populations = _replay_populations(
        (*historical, *current), replay, (4, 8), 4, 12
    )
    assert tuple(len(population.rows) for population in populations) == (8, 12)
    assert tuple(row.image_id for row in populations[0].rows) == tuple(
        row.image_id
        for row in sorted((*current, *historical[:4]), key=lambda row: row.image_id)
    )
    assert {row.image_id for row in populations[0].rows} <= {
        row.image_id for row in populations[1].rows
    }


def _write_cell(
    run: Path,
    family: str,
    capacity: int,
    protocol_hash: str,
    source_hash: str,
) -> dict[str, object]:
    cell_root = run / "controls/replay" / family / f"h{capacity:05d}"
    cell_root.mkdir(parents=True)
    epochs = 50 if family == "single_affine" else 5
    history_path = cell_root / "history.jsonl"
    history_path.write_bytes(
        b"".join(
            canonical_json_bytes(
                {
                    "epoch": epoch,
                    "image_presentations": epoch * (capacity + 367),
                    "validation_accuracy": 70.0 + epoch / 100,
                    "validation_nll": 1.5 - epoch / 100,
                }
            )
            + b"\n"
            for epoch in range(1, epochs + 1)
        )
    )
    fit = (
        {
            "best_nll_epoch": 50,
            "epochs": 50,
            "fixed_image_presentations": 5 * (capacity + 367),
            "fixed_validation_accuracy": 70.05,
            "fixed_validation_nll": 1.45,
            "image_presentations": 50 * (capacity + 367),
            "max_accuracy_epoch": 50,
            "max_validation_accuracy": 70.5,
            "peak_vram_bytes": 10,
            "validation_accuracy_at_best_nll": 70.5,
            "validation_nll_at_max_accuracy": 1.0,
            "best_validation_nll": 1.0,
            "wall_seconds": 1.0,
        }
        if family == "single_affine"
        else {
            "best_epoch": 5,
            "best_validation_accuracy": 70.05,
            "best_validation_nll": 1.45,
            "epochs": 5,
            "fixed_validation_accuracy": 70.05,
            "fixed_validation_nll": 1.45,
            "image_presentations": 5 * (capacity + 367),
            "peak_vram_bytes": 10,
            "wall_seconds": 1.0,
        }
    )
    specification = {
        "adapt_lora": True,
        "family": family,
        "historical_capacity": capacity,
        "seed": 1993,
    }
    core = {
        "architecture": {"trainable_parameters": 100},
        "cell": specification,
        "displacements": [],
        "fit": fit,
        "history": str(history_path.relative_to(run)),
        "history_sha256": file_sha256(history_path),
        "population": {
            "historical_capacity": capacity,
            "training_examples": capacity + 367,
        },
        "protocol_hash": protocol_hash,
        "schema_version": "imagenetr50-frontier-architecture-replay-cell-v1",
        "source_full_result_hash": source_hash,
        "test_evaluations": 0,
    }
    record = {**core, "content_hash": record_sha256(core)}
    result_path = cell_root / "result.json"
    result_path.write_bytes(canonical_json_bytes(record))
    return {
        "cell": specification,
        "result": str(result_path.relative_to(run)),
        "result_content_hash": record["content_hash"],
        "result_sha256": file_sha256(result_path),
    }


def test_report_loader_authenticates_all_eight_cells_without_model_weights(
    tmp_path: Path,
) -> None:
    protocol_core = {
        "schema_version": "imagenetr50-frontier-architecture-replay-protocol-v1"
    }
    protocol_hash = record_sha256(protocol_core)
    protocol_path = tmp_path / "controls/replay/protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(
        canonical_json_bytes(
            {**protocol_core, "content_hash": protocol_hash}
        )
    )
    parent_hash = "b" * 64
    sources = {"single_affine": "c" * 64, "joint_iid_rank80": "d" * 64}
    cells = [
        _write_cell(tmp_path, family, capacity, protocol_hash, sources[family])
        for capacity in CAPACITIES
        for family in FAMILIES
    ]
    core = {
        "cells": cells,
        "families": list(FAMILIES),
        "historical_capacities": list(CAPACITIES),
        "parent_result_hash": parent_hash,
        "protocol": str(protocol_path.relative_to(tmp_path)),
        "protocol_hash": protocol_hash,
        "schema_version": "imagenetr50-frontier-architecture-replay-result-v1",
        "source_full_results": sources,
        "test_evaluations": 0,
    }
    aggregate = tmp_path / "evaluations/frontier_architecture_replay_sweep.json"
    aggregate.parent.mkdir(exist_ok=True)
    aggregate.write_bytes(
        canonical_json_bytes({**core, "content_hash": record_sha256(core)})
    )
    report = load_architecture_replay_report(tmp_path, parent_hash, sources)
    assert report is not None
    assert len(report.cells) == 8
    assert len(report.summaries) == 8
    assert len(report.histories) == 220
    assert report.displacements == ()
    assert tuple(
        (row["historical_capacity"], row["family"])
        for row in report.summaries
    ) == tuple(
        (capacity, family) for capacity in CAPACITIES for family in FAMILIES
    )
