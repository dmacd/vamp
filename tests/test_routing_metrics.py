from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.routing_metrics import (
    RoutingComparisonMetrics,
    summarize_hopfield_routing,
)
from apm.memory.content_addressing import HopfieldAddressResult


def _result() -> HopfieldAddressResult:
    return HopfieldAddressResult(
        selected_indices=jnp.asarray((0, 1, 2, 3), dtype=jnp.int32),
        node_probabilities=jnp.asarray(
            (
                (0.7, 0.2, 0.1, 0.0, 0.0),
                (0.1, 0.6, 0.2, 0.1, 0.0),
                (0.2, 0.1, 0.6, 0.1, 0.0),
                (0.1, 0.2, 0.1, 0.6, 0.0),
            ),
            dtype=jnp.float32,
        ),
        node_scores=jnp.asarray(
            (
                (1.0, 0.5, 0.0, -0.5, -jnp.inf),
                (0.0, 1.0, 0.5, -0.5, -jnp.inf),
                (0.5, 0.0, 1.5, -0.5, -jnp.inf),
                (0.0, 0.5, -0.5, 2.0, -jnp.inf),
            ),
            dtype=jnp.float32,
        ),
        score_margin=jnp.asarray((0.5, 1.0, 1.5, 2.0), dtype=jnp.float32),
        entropy=jnp.asarray((0.1, 0.2, 0.3, 0.4), dtype=jnp.float32),
        top_k_indices=jnp.asarray(
            ((0, 1), (1, 2), (2, 0), (3, 2)),
            dtype=jnp.int32,
        ),
    )


def test_summary_reports_exact_oracle_top_k_and_exhaustive_comparisons() -> None:
    summary = summarize_hopfield_routing(
        _result(),
        oracle_indices=np.asarray((0, 2, 2, 1), dtype=np.int32),
        exhaustive_indices=np.asarray((0, 1, 3, 3), dtype=np.int32),
    )

    expected = RoutingComparisonMetrics(
        example_count=4,
        accuracy_vs_oracle=0.5,
        top_k_recall=0.75,
        agreement_with_exhaustive=0.75,
        mean_margin=1.25,
        mean_entropy=0.25,
    )
    assert summary.example_count == expected.example_count
    assert summary.accuracy_vs_oracle == expected.accuracy_vs_oracle
    assert summary.top_k_recall == expected.top_k_recall
    assert summary.agreement_with_exhaustive == expected.agreement_with_exhaustive
    assert summary.mean_margin == expected.mean_margin
    assert summary.mean_entropy == pytest.approx(expected.mean_entropy)


def test_top_k_recall_is_independent_of_hard_choice_and_exhaustive_agreement() -> None:
    result = _result()._replace(
        selected_indices=jnp.asarray((0, 1, 2, 3), dtype=jnp.int32),
        top_k_indices=jnp.asarray(
            ((0, 2), (1, 2), (2, 3), (3, 0)),
            dtype=jnp.int32,
        ),
    )

    summary = summarize_hopfield_routing(
        result,
        oracle_indices=np.asarray((2, 2, 3, 0), dtype=np.int64),
        exhaustive_indices=np.asarray((3, 3, 3, 0), dtype=np.int64),
    )

    assert summary.accuracy_vs_oracle == 0.0
    assert summary.top_k_recall == 1.0
    assert summary.agreement_with_exhaustive == 0.0


@pytest.mark.parametrize(
    "oracle_indices",
    (
        np.asarray((0, 1, 2), dtype=np.int32),
        np.asarray((0.0, 1.0, 2.0, 3.0), dtype=np.float32),
        np.asarray((0, 1, 2, 5), dtype=np.int32),
        np.asarray((0, 1, 2, 4), dtype=np.int32),
        np.asarray((0, 1, 2, -1), dtype=np.int32),
    ),
)
def test_invalid_oracle_indices_are_rejected(oracle_indices: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError), match="oracle_indices"):
        summarize_hopfield_routing(
            _result(),
            oracle_indices,
            np.asarray((0, 1, 2, 3), dtype=np.int32),
        )


@pytest.mark.parametrize("defect", ("shape", "probability", "selected", "top_k"))
def test_result_shape_and_range_defects_are_rejected(defect: str) -> None:
    result = _result()
    if defect == "shape":
        result = result._replace(entropy=jnp.zeros((3,), dtype=jnp.float32))
    elif defect == "probability":
        result = result._replace(
            node_probabilities=result.node_probabilities.at[0, 0].set(1.5)
        )
    elif defect == "selected":
        result = result._replace(
            selected_indices=jnp.asarray((0, 1, 2, 4), dtype=jnp.int32)
        )
    else:
        result = result._replace(
            top_k_indices=jnp.asarray(
                ((0, 0), (1, 2), (2, 0), (3, 2)),
                dtype=jnp.int32,
            )
        )

    with pytest.raises((TypeError, ValueError)):
        summarize_hopfield_routing(
            result,
            np.asarray((0, 1, 2, 3), dtype=np.int32),
            np.asarray((0, 1, 2, 3), dtype=np.int32),
        )


def test_single_valid_node_preserves_infinite_mean_margin() -> None:
    result = HopfieldAddressResult(
        selected_indices=jnp.asarray((0, 0), dtype=jnp.int32),
        node_probabilities=jnp.asarray(
            ((1.0, 0.0), (1.0, 0.0)),
            dtype=jnp.float32,
        ),
        node_scores=jnp.asarray(
            ((0.4, -jnp.inf), (0.8, -jnp.inf)),
            dtype=jnp.float32,
        ),
        score_margin=jnp.asarray((jnp.inf, jnp.inf), dtype=jnp.float32),
        entropy=jnp.zeros((2,), dtype=jnp.float32),
        top_k_indices=jnp.zeros((2, 1), dtype=jnp.int32),
    )

    summary = summarize_hopfield_routing(
        result,
        np.zeros((2,), dtype=np.int32),
        np.zeros((2,), dtype=np.int32),
    )

    assert summary.accuracy_vs_oracle == 1.0
    assert summary.top_k_recall == 1.0
    assert summary.agreement_with_exhaustive == 1.0
    assert np.isposinf(summary.mean_margin)
    assert summary.mean_entropy == 0.0


def test_comparison_metric_is_frozen_with_exact_fields() -> None:
    metric = RoutingComparisonMetrics(1, 1.0, 1.0, 1.0, 0.5, 0.0)

    assert tuple(field.name for field in fields(metric)) == (
        "example_count",
        "accuracy_vs_oracle",
        "top_k_recall",
        "agreement_with_exhaustive",
        "mean_margin",
        "mean_entropy",
    )
    with pytest.raises(FrozenInstanceError):
        metric.example_count = 2  # type: ignore[misc]
