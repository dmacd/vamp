from __future__ import annotations

import json
from pathlib import Path

import pytest

from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.frontier_adaptation_config import (
    load_frontier_adaptation_config,
)
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    CONDITION_LABELS,
    FROZEN_ONLINE_LABEL,
    _summary_rows,
)
from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    AdaptationCell,
    nested_historical_order,
    warmup_cosine_multiplier,
)
from apm.continual.vision.imagenetr.frontier_adaptation_workflow import (
    FrontierAdaptationProtocol,
    _material_paths,
    _training_rows,
)


CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_lora_adaptation_v10.yaml"
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


def test_frontier_config_freezes_nested_h_and_coupled_optimizer() -> None:
    config = load_frontier_adaptation_config(CONFIG)
    assert config.stage == 31
    assert config.historical_capacities == (1024, 2048, 4096, 8192, 11827)
    assert config.current_task_examples == 367
    assert config.historical_capacities[-1] + config.current_task_examples == 12194
    assert config.effective_batch_size == 64
    assert config.microbatch_size == 64
    assert config.epochs == 50
    assert config.macro_peak_learning_rate == 0.00003
    assert config.lora_peak_learning_rate == 0.0005
    assert config.frozen_full_fit_control


def test_uniform_replay_order_is_deterministic_nested_and_order_independent() -> None:
    rows = tuple(_row(index) for index in range(30))
    forward = nested_historical_order(rows, 1993)
    reverse = nested_historical_order(tuple(reversed(rows)), 1993)
    assert forward == reverse
    assert len(set(row.image_id for row in forward)) == len(rows)
    assert forward[:8] == forward[:16][:8]
    assert nested_historical_order(rows, 1994) != forward


def test_training_population_retains_all_current_rows_and_nests_history() -> None:
    history = nested_historical_order(tuple(_row(index) for index in range(12)), 1993)
    current = tuple(_row(index) for index in range(120, 124))
    small = _training_rows(current, history, 4)
    large = _training_rows(current, history, 8)
    current_ids = {row.image_id for row in current}
    assert len(small) == 8
    assert len(large) == 12
    assert current_ids <= {row.image_id for row in small}
    assert {row.image_id for row in small} <= {row.image_id for row in large}


def test_shared_warmup_cosine_schedule_reaches_peak_and_floor() -> None:
    values = tuple(warmup_cosine_multiplier(step, 100, 0.05, 0.01) for step in range(100))
    assert values[0] == pytest.approx(0.2)
    assert values[4] == pytest.approx(1.0)
    assert values[5] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.01)
    assert all(left >= right for left, right in zip(values[4:-1], values[5:], strict=True))


def test_condition_semantics_are_complete_and_report_labels_are_unique() -> None:
    assert set(CONDITION_LABELS) == {1024, 2048, 4096, 8192, 11827}
    assert len(set(CONDITION_LABELS.values())) == 5
    for capacity, label in CONDITION_LABELS.items():
        assert AdaptationCell(capacity, True, 1993).condition.endswith(str(capacity))
        assert f"{capacity:,}" in label
    assert AdaptationCell(11827, False, 1993).condition == (
        "frozen_frontier_full_fit_control"
    )
    assert FROZEN_ONLINE_LABEL == "Frozen frontier, online full fit"


def test_protocol_identity_is_content_addressed_and_report_code_is_nonmaterial() -> None:
    identity = "a" * 64
    protocol = FrontierAdaptationProtocol(
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
        identity,
    )
    assert len(protocol.content_hash) == 64
    material = _material_paths(Path.cwd(), CONFIG.resolve())
    assert not any(path.name == "frontier_adaptation_reporting.py" for path in material)


def test_report_keeps_minimum_nll_and_maximum_accuracy_distinct(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.jsonl"
    rows = tuple(
        {
            "epoch": epoch,
            "validation_accuracy": 78.0 if epoch == 25 else 75.0,
            "validation_nll": 0.9 if epoch == 25 else 1.0,
        }
        for epoch in range(1, 51)
    )
    history.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "references": {"joint_iid": {"accuracy": 77.0, "nll": 0.95}},
        "cells": [
            {
                "cell": {
                    "adapt_lora": True,
                    "historical_capacity": 1024,
                },
                "fit": {
                    "best_nll_epoch": 25,
                    "best_validation_nll": 0.9,
                    "image_presentations": 51200,
                    "max_accuracy_epoch": 25,
                    "max_validation_accuracy": 78.0,
                    "peak_vram_bytes": 10,
                    "train_accuracy_at_best": 90.0,
                    "train_nll_at_best": 0.2,
                    "trainable_parameters": 100,
                    "validation_accuracy_at_best_nll": 78.0,
                    "validation_nll_at_max_accuracy": 0.9,
                    "wall_seconds": 1.0,
                },
                "history": history.name,
            }
        ],
    }
    summary = _summary_rows(tmp_path, result)[0]
    assert summary["condition"] == CONDITION_LABELS[1024]
    assert summary["simultaneous_joint_match_epoch"] == 25
    assert summary["max_validation_accuracy"] == 78.0
