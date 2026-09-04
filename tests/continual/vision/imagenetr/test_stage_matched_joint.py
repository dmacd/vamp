from __future__ import annotations

import pytest
import torch

from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.stage_matched_joint import validate_stage_rows


def _row(stage: int) -> dict[str, object]:
    return {
        "accuracy": 90.0 - stage / 10,
        "class_count": 4 * stage,
        "stage": stage,
        "task_examples": [3] * stage,
        "test_examples": 3 * stage,
    }


def test_stage_matched_rows_require_a_complete_finite_curve() -> None:
    rows = tuple(_row(stage) for stage in reversed(range(1, 51)))

    ordered = validate_stage_rows(rows)

    assert [row["stage"] for row in ordered] == list(range(1, 51))
    with pytest.raises(ValueError, match="incomplete"):
        validate_stage_rows(rows[:-1])


@pytest.mark.parametrize(
    ("field", "value"),
    (("accuracy", float("nan")), ("class_count", 7), ("test_examples", 1)),
)
def test_stage_matched_rows_reject_invalid_measurements(
    field: str, value: object
) -> None:
    rows = [_row(stage) for stage in range(1, 51)]
    rows[6][field] = value

    with pytest.raises(ValueError, match="invalid result"):
        validate_stage_rows(rows)


def test_prefix_head_initialization_matches_the_same_rows_of_the_full_head() -> None:
    prefix = AffineClassifier(tuple(range(64)), 768, initialization_seed=11_993)
    full = AffineClassifier(tuple(range(200)), 768, initialization_seed=11_993)

    torch.testing.assert_close(prefix.weight, full.weight[:64], rtol=0.0, atol=0.0)
    torch.testing.assert_close(prefix.bias, full.bias[:64], rtol=0.0, atol=0.0)
