from __future__ import annotations

from pathlib import Path

import pytest

from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.frontier_adaptation_config import (
    load_frontier_adaptation_config,
)
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    CONDITION_LABELS,
    FROZEN_ONLINE_LABEL,
)
from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    AdaptationCell,
    nested_replay_order,
    warmup_cosine_multiplier,
)
from apm.continual.vision.imagenetr.frontier_adaptation_workflow import (
    FrontierAdaptationProtocol,
    _material_paths,
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
    assert config.historical_capacities == (1024, 2048, 4096, 8192, 12194)
    assert config.effective_batch_size == 64
    assert config.microbatch_size == 64
    assert config.epochs == 50
    assert config.macro_peak_learning_rate == 0.00003
    assert config.lora_peak_learning_rate == 0.0005
    assert config.frozen_full_fit_control


def test_uniform_replay_order_is_deterministic_nested_and_order_independent() -> None:
    rows = tuple(_row(index) for index in range(30))
    forward = nested_replay_order(rows, 1993)
    reverse = nested_replay_order(tuple(reversed(rows)), 1993)
    assert forward == reverse
    assert len(set(row.image_id for row in forward)) == len(rows)
    assert forward[:8] == forward[:16][:8]
    assert nested_replay_order(rows, 1994) != forward


def test_shared_warmup_cosine_schedule_reaches_peak_and_floor() -> None:
    values = tuple(warmup_cosine_multiplier(step, 100, 0.05, 0.01) for step in range(100))
    assert values[0] == pytest.approx(0.2)
    assert values[4] == pytest.approx(1.0)
    assert values[5] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.01)
    assert all(left >= right for left, right in zip(values[4:-1], values[5:], strict=True))


def test_condition_semantics_are_complete_and_report_labels_are_unique() -> None:
    assert set(CONDITION_LABELS) == {1024, 2048, 4096, 8192, 12194}
    assert len(set(CONDITION_LABELS.values())) == 5
    for capacity, label in CONDITION_LABELS.items():
        assert AdaptationCell(capacity, True, 1993).condition.endswith(str(capacity))
        assert f"{capacity:,}" in label
    assert AdaptationCell(12194, False, 1993).condition == (
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
