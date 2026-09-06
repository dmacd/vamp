from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    JOINT_CAPACITY_CONTROLS,
    TOTAL_PARAM_MATCHED_LABEL,
    _capacity_history,
    _capacity_summary,
    _validated_capacity_control,
)
from apm.continual.vision.imagenetr.frontier_total_param_matched_config import (
    load_frontier_total_param_matched_config,
)
from apm.continual.vision.imagenetr.frontier_total_param_matched_workflow import (
    FrontierTotalParamMatchedProtocol,
    _material_paths,
)


CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_total_param_matched_control_v12.yaml"
)
RANK224_SPECIFICATION = JOINT_CAPACITY_CONTROLS[1]


def test_total_param_config_selects_closest_integer_rank() -> None:
    config = load_frontier_total_param_matched_config(CONFIG)
    assert config.target_rank == config.target_alpha == 224
    assert config.frontier_active_parameters == 18_691_016
    assert config.joint_active_parameters == 18_674_812
    assert config.parameter_difference == -16_204
    rank225_total = 225 * config.lora_parameters_per_rank + config.classifier_parameters
    assert abs(config.parameter_difference) < abs(
        rank225_total - config.frontier_active_parameters
    )
    with pytest.raises(ValueError, match="total-parameter-matched"):
        replace(config, target_rank=225, target_alpha=225)


def test_total_param_protocol_and_material_surface_are_content_addressed() -> None:
    identities = tuple(f"{index:064x}" for index in range(1, 15))
    protocol = FrontierTotalParamMatchedProtocol(*identities)
    assert len(protocol.content_hash) == 64
    assert replace(protocol, config_hash="f" * 64).content_hash != protocol.content_hash
    config = load_frontier_total_param_matched_config(CONFIG)
    material = _material_paths(Path.cwd(), CONFIG.resolve(), config.source_config)
    names = {path.name for path in material}
    assert {
        "frontier_total_param_matched_config.py",
        "frontier_total_param_matched_workflow.py",
        "imagenetr50_frontier_total_param_matched_control_protocol.md",
        "run_frontier_total_param_matched_control_local.sh",
    } <= names
    assert "frontier_adaptation_reporting.py" not in names
    assert all(path.is_file() for path in material)


def test_report_authenticates_rank224_endpoint_and_history(tmp_path: Path) -> None:
    history_path = tmp_path / "controls/rank224/history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        "".join(
            canonical_json_bytes(
                {
                    "epoch": epoch,
                    "validation_accuracy": 75.0 + epoch,
                    "validation_nll": 1.2 - epoch / 10,
                }
            ).decode("utf-8")
            + "\n"
            for epoch in range(1, 6)
        ),
        encoding="utf-8",
    )
    parent = {"content_hash": "a" * 64}
    core = {
        "architecture": {
            "classifier_parameters": 95_356,
            "frontier_active_parameters": 18_691_016,
            "frontier_aggregate_lora_parameters": 6_635_520,
            "frontier_integrator_parameters_included_in_match": 12_055_496,
            "lora_alpha": 224,
            "lora_parameters": 18_579_456,
            "lora_rank": 224,
            "parameter_difference": -16_204,
            "trainable_parameters": 18_674_812,
        },
        "fit": {
            "best_epoch": 5,
            "best_validation_accuracy": 80.0,
            "best_validation_nll": 0.7,
            "epochs": 5,
            "fixed_validation_accuracy": 80.0,
            "fixed_validation_nll": 0.7,
            "image_presentations": 60_970,
            "peak_vram_bytes": 10,
            "wall_seconds": 1.0,
        },
        "history": str(history_path.relative_to(tmp_path)),
        "history_sha256": file_sha256(history_path),
        "parent_result_hash": parent["content_hash"],
        "schema_version": (
            "imagenetr50-frontier-total-param-matched-control-v1"
        ),
        "test_evaluations": 0,
    }
    control_path = tmp_path / "evaluations/joint_iid_lora_r224.json"
    control_path.parent.mkdir()
    control_path.write_bytes(
        canonical_json_bytes({**core, "content_hash": record_sha256(core)})
    )
    control = _validated_capacity_control(
        tmp_path, parent, RANK224_SPECIFICATION
    )
    assert control is not None
    summary = _capacity_summary(control, RANK224_SPECIFICATION)
    assert summary["condition"] == TOTAL_PARAM_MATCHED_LABEL
    assert summary["parameter_difference"] == -16_204
    assert len(_capacity_history(tmp_path, control, RANK224_SPECIFICATION)) == 5
