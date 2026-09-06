from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    RANK_MATCHED_LABEL,
    _rank_matched_history,
    _rank_matched_summary,
    _validated_rank_matched_control,
)
from apm.continual.vision.imagenetr.frontier_rank_matched_config import (
    load_frontier_rank_matched_config,
)
from apm.continual.vision.imagenetr.frontier_rank_matched_workflow import (
    CLASSIFIER_PARAMETERS,
    MACRO_PARAMETERS,
    FrontierRankMatchedProtocol,
    _material_paths,
    vit_lora_parameter_count,
)


CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_rank_matched_control_v11.yaml"
)


def test_rank_matched_config_freezes_capacity_data_and_optimization() -> None:
    config = load_frontier_rank_matched_config(CONFIG)
    assert (config.stage, config.seed) == (31, 1993)
    assert (config.source_rank, config.frontier_adapters) == (16, 5)
    assert (config.target_rank, config.target_alpha, config.dropout) == (80, 80, 0.0)
    assert config.training.epochs == 5
    assert config.training.batch_size == 64
    assert config.training.lora_lr == 0.0005
    assert config.training.head_lr == 0.01


def test_rank80_exactly_matches_five_rank16_lora_parameter_sets() -> None:
    source_parameters = vit_lora_parameter_count(16)
    assert source_parameters == 1_327_104
    assert vit_lora_parameter_count(80) == 5 * source_parameters == 6_635_520
    assert vit_lora_parameter_count(80) + CLASSIFIER_PARAMETERS == 6_730_876
    assert MACRO_PARAMETERS == 12_055_496
    with pytest.raises(ValueError, match="positive"):
        vit_lora_parameter_count(0)


def test_rank_matched_protocol_and_material_surface_are_content_addressed() -> None:
    identities = tuple(f"{index:064x}" for index in range(1, 12))
    protocol = FrontierRankMatchedProtocol(*identities)
    assert len(protocol.content_hash) == 64
    assert replace(protocol, config_hash="f" * 64).content_hash != protocol.content_hash
    material = _material_paths(Path.cwd(), CONFIG.resolve())
    names = {path.name for path in material}
    assert {
        "frontier_rank_matched_config.py",
        "frontier_rank_matched_workflow.py",
        "imagenetr50_frontier_rank_matched_control_protocol.md",
        "run_frontier_rank_matched_control_local.sh",
    } <= names
    assert "frontier_adaptation_reporting.py" not in names
    assert all(path.is_file() for path in material)


def test_report_authenticates_rank80_endpoint_and_history(tmp_path: Path) -> None:
    history_path = tmp_path / "controls/rank80/history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        "".join(
            canonical_json_bytes(
                {
                    "epoch": epoch,
                    "validation_accuracy": 70.0 + epoch,
                    "validation_nll": 1.5 - epoch / 10,
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
            "classifier_parameters": CLASSIFIER_PARAMETERS,
            "frontier_aggregate_lora_parameters": 6_635_520,
            "frontier_integrator_parameters_excluded_from_match": MACRO_PARAMETERS,
            "lora_alpha": 80,
            "lora_parameters": 6_635_520,
            "lora_rank": 80,
            "trainable_parameters": 6_730_876,
        },
        "fit": {
            "best_epoch": 5,
            "best_validation_accuracy": 75.0,
            "best_validation_nll": 1.0,
            "epochs": 5,
            "fixed_validation_accuracy": 75.0,
            "fixed_validation_nll": 1.0,
            "image_presentations": 60_970,
            "peak_vram_bytes": 10,
            "wall_seconds": 1.0,
        },
        "history": str(history_path.relative_to(tmp_path)),
        "history_sha256": file_sha256(history_path),
        "parent_result_hash": parent["content_hash"],
        "schema_version": "imagenetr50-frontier-rank-matched-control-v1",
        "test_evaluations": 0,
    }
    control_path = tmp_path / "evaluations/joint_iid_lora_r80.json"
    control_path.parent.mkdir()
    control_path.write_bytes(
        canonical_json_bytes({**core, "content_hash": record_sha256(core)})
    )
    control = _validated_rank_matched_control(tmp_path, parent)
    assert control is not None
    assert _rank_matched_summary(control)["condition"] == RANK_MATCHED_LABEL
    assert len(_rank_matched_history(tmp_path, control)) == 5
