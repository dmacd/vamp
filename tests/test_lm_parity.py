from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.parity import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    ParitySnapshot,
    assert_parity,
    compare_parity_snapshots,
    compare_parity_values,
    ordered_capture_spec,
)


def _config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=7,
        max_position_embeddings=4,
        hidden_size=4,
        intermediate_size=8,
        num_layers=2,
        num_heads=2,
        attention_types=("global", "local"),
        local_window_size=2,
    )


def _snapshot() -> ParitySnapshot:
    token_ids = np.asarray(((1, 2, 3),), dtype=np.int32)
    position_ids = np.asarray(((0, 1, 2),), dtype=np.int32)
    global_mask = np.asarray(
        ((True, False, False), (True, True, False), (True, True, True))
    )
    local_mask = np.asarray(
        ((True, False, False), (True, True, False), (False, True, True))
    )
    hidden_shape = (1, 3, 4)
    return ParitySnapshot(
        token_ids=token_ids,
        embedding_output=np.full(hidden_shape, 0.5, dtype=np.float32),
        position_ids=position_ids,
        attention_masks=(global_mask, local_mask),
        captured_hidden=tuple(
            np.full(hidden_shape, capture_index + 1.0, dtype=np.float32)
            for capture_index in range(4)
        ),
        final_hidden=np.full(hidden_shape, 5.0, dtype=np.float32),
        logits=np.full((1, 3, 7), 0.25, dtype=np.float32),
        normalized_nll=np.asarray(1.25, dtype=np.float32),
        greedy_token_ids=np.asarray(((1, 2, 3, 4)), dtype=np.int32),
    )


def test_equal_snapshots_pass_the_complete_ordered_ladder() -> None:
    snapshot = _snapshot()

    report = compare_parity_snapshots(snapshot, snapshot, _config())

    assert report.passed
    assert report.failures == ()
    assert report.rtol == DEFAULT_RTOL == 2e-4
    assert report.atol == DEFAULT_ATOL == 2e-4
    assert tuple((record.layer_index, record.site) for record in report.records) == (
        (None, "token_ids"),
        (None, "embedding_output"),
        (None, "position_ids"),
        (0, "attention_mask.global"),
        (1, "attention_mask.local"),
        (0, "post_attention"),
        (0, "post_mlp"),
        (1, "post_attention"),
        (1, "post_mlp"),
        (None, "final_hidden"),
        (None, "logits"),
        (None, "normalized_nll"),
        (None, "greedy_token_ids"),
    )
    assert all(record.max_absolute_error == 0.0 for record in report.records)
    assert all(record.mean_absolute_error == 0.0 for record in report.records)
    assert_parity(report)


def test_perturbed_capture_reports_layer_site_max_and_mean_error() -> None:
    expected = _snapshot()
    perturbed_capture = np.asarray(expected.captured_hidden[2]).copy()
    perturbed_capture[0, 1, 2] += 0.01
    perturbed_logits = np.asarray(expected.logits).copy()
    perturbed_logits[0, 0, 0] += 1e-5
    actual = replace(
        expected,
        captured_hidden=(
            expected.captured_hidden[0],
            expected.captured_hidden[1],
            perturbed_capture,
            expected.captured_hidden[3],
        ),
        logits=perturbed_logits,
    )

    report = compare_parity_snapshots(expected, actual, _config())

    assert not report.passed
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.layer_index == 1
    assert failure.site == "post_attention"
    assert failure.label == "layer 1 post_attention"
    assert failure.max_absolute_error == pytest.approx(0.01, rel=1e-4)
    assert failure.mean_absolute_error == pytest.approx(0.01 / 12.0, rel=1e-4)
    with pytest.raises(AssertionError, match=r"layer 1 post_attention.*max_abs=.*rtol=0.0002"):
        assert_parity(report)


def test_integer_ids_and_attention_masks_require_exact_equality() -> None:
    expected = _snapshot()
    changed_ids = np.asarray(expected.token_ids).copy()
    changed_ids[0, 0] += 1
    changed_mask = np.asarray(expected.attention_masks[1]).copy()
    changed_mask[2, 0] = True
    actual = replace(
        expected,
        token_ids=changed_ids,
        attention_masks=(expected.attention_masks[0], changed_mask),
    )

    report = compare_parity_snapshots(expected, actual, _config(), rtol=1.0, atol=1.0)

    assert tuple(record.site for record in report.failures) == (
        "token_ids",
        "attention_mask.local",
    )
    assert all(record.exact for record in report.failures)


def test_shape_mismatch_is_a_clear_infinite_error_record() -> None:
    record = compare_parity_values(
        "logits",
        np.zeros((1, 2, 3), dtype=np.float32),
        np.zeros((1, 3, 3), dtype=np.float32),
    )

    assert not record.passed
    assert record.expected_shape == (1, 2, 3)
    assert record.actual_shape == (1, 3, 3)
    assert np.isinf(record.max_absolute_error)
    assert np.isinf(record.mean_absolute_error)


def test_custom_tolerance_is_explicit_and_never_auto_loosened() -> None:
    expected = np.asarray((1.0, 2.0), dtype=np.float32)
    actual = np.asarray((1.01, 2.0), dtype=np.float32)

    default_record = compare_parity_values("capture", expected, actual)
    relaxed_record = compare_parity_values(
        "capture",
        expected,
        actual,
        rtol=0.02,
        atol=0.0,
    )

    assert not default_record.passed
    assert relaxed_record.passed
    assert default_record.max_absolute_error == relaxed_record.max_absolute_error


def test_ordered_capture_spec_contains_both_residuals_per_block() -> None:
    points = ordered_capture_spec(_config()).points

    assert tuple((point.layer_index, point.location) for point in points) == (
        (0, "post_attention"),
        (0, "post_mlp"),
        (1, "post_attention"),
        (1, "post_mlp"),
    )


def test_snapshot_and_error_records_are_frozen() -> None:
    snapshot = _snapshot()
    report = compare_parity_snapshots(snapshot, snapshot, _config())

    with pytest.raises(FrozenInstanceError):
        snapshot.normalized_nll = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.records[0].passed = False  # type: ignore[misc]


def test_snapshot_structure_and_tolerances_are_validated() -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="attention mask per layer"):
        compare_parity_snapshots(
            replace(snapshot, attention_masks=snapshot.attention_masks[:1]),
            snapshot,
            _config(),
        )
    with pytest.raises(ValueError, match="ordered post-attention/post-MLP"):
        compare_parity_snapshots(
            snapshot,
            replace(snapshot, captured_hidden=snapshot.captured_hidden[:2]),
            _config(),
        )
    with pytest.raises(ValueError, match="tolerances"):
        compare_parity_snapshots(snapshot, snapshot, _config(), rtol=-1.0)
