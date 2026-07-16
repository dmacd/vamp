"""Evaluator-only summaries comparing Hopfield, oracle, and exhaustive routing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import numpy as np

from apm.memory.content_addressing import HopfieldAddressResult


@dataclass(frozen=True)
class RoutingComparisonMetrics:
    """Aggregate Hopfield routing quality against evaluator-only references."""

    example_count: int
    accuracy_vs_oracle: float
    top_k_recall: float
    agreement_with_exhaustive: float
    mean_margin: float
    mean_entropy: float

    def __post_init__(self) -> None:
        if type(self.example_count) is not int or self.example_count <= 0:
            raise ValueError("example_count must be a positive integer")
        rates = {
            "accuracy_vs_oracle": self.accuracy_vs_oracle,
            "top_k_recall": self.top_k_recall,
            "agreement_with_exhaustive": self.agreement_with_exhaustive,
        }
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in rates.values()
        ):
            raise ValueError("routing comparison rates must be finite and in [0, 1]")
        if math.isnan(self.mean_margin) or self.mean_margin < 0.0:
            raise ValueError("mean_margin must be nonnegative and not NaN")
        if not math.isfinite(self.mean_entropy) or self.mean_entropy < 0.0:
            raise ValueError("mean_entropy must be finite and nonnegative")


def summarize_hopfield_routing(
    result: HopfieldAddressResult,
    oracle_indices: jax.Array | np.ndarray,
    exhaustive_indices: jax.Array | np.ndarray,
) -> RoutingComparisonMetrics:
    """Summarize task-free Hopfield choices against evaluator-only node indices."""
    selected = _int32_array(result.selected_indices, "selected_indices", rank=1)
    probabilities = _float32_array(
        result.node_probabilities,
        "node_probabilities",
        rank=2,
    )
    scores = _float32_array(result.node_scores, "node_scores", rank=2)
    margins = _float32_array(result.score_margin, "score_margin", rank=1)
    entropy = _float32_array(result.entropy, "entropy", rank=1)
    top_k = _int32_array(result.top_k_indices, "top_k_indices", rank=2)
    example_count = selected.shape[0]
    if example_count == 0:
        raise ValueError("routing result must contain at least one example")
    node_count = scores.shape[1]
    if node_count == 0:
        raise ValueError("routing result must contain at least one node")
    if probabilities.shape != scores.shape:
        raise ValueError("node probabilities and scores must share shape [batch, nodes]")
    if scores.shape[0] != example_count:
        raise ValueError("node probabilities and scores must match the batch size")
    expected_vector_shape = (example_count,)
    if margins.shape != expected_vector_shape or entropy.shape != expected_vector_shape:
        raise ValueError("selected indices, margins, and entropy must share a batch axis")
    if top_k.shape[0] != example_count or not 1 <= top_k.shape[1] <= node_count:
        raise ValueError("top_k_indices must have shape [batch, nonempty top-k]")
    oracle = _integer_reference_indices(
        oracle_indices,
        "oracle_indices",
        expected_vector_shape,
    )
    exhaustive = _integer_reference_indices(
        exhaustive_indices,
        "exhaustive_indices",
        expected_vector_shape,
    )
    valid_nodes = _validate_result_values(
        probabilities,
        scores,
        margins,
        entropy,
    )
    _validate_index_values(selected, valid_nodes, "selected_indices")
    _validate_index_values(oracle, valid_nodes, "oracle_indices")
    _validate_index_values(exhaustive, valid_nodes, "exhaustive_indices")
    _validate_top_k(top_k, valid_nodes)
    return RoutingComparisonMetrics(
        example_count=example_count,
        accuracy_vs_oracle=float(np.mean(selected == oracle)),
        top_k_recall=float(np.mean(np.any(top_k == oracle[:, None], axis=1))),
        agreement_with_exhaustive=float(np.mean(selected == exhaustive)),
        mean_margin=float(np.mean(margins)),
        mean_entropy=float(np.mean(entropy)),
    )


def _validate_result_values(
    probabilities: np.ndarray,
    scores: np.ndarray,
    margins: np.ndarray,
    entropy: np.ndarray,
) -> np.ndarray:
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("node_scores may contain only finite values or -inf padding")
    valid_nodes = np.isfinite(scores)
    if np.any(np.sum(valid_nodes, axis=1) == 0):
        raise ValueError("every routing row must contain at least one valid node")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("node_probabilities must be finite and in [0, 1]")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
        raise ValueError("node probability rows must sum to one")
    if np.any(probabilities[~valid_nodes] != 0.0):
        raise ValueError("invalid nodes must have zero probability")
    if np.any(np.isnan(margins)) or np.any(margins < 0.0):
        raise ValueError("score margins must be nonnegative and not NaN")
    single_valid_node = np.sum(valid_nodes, axis=1) == 1
    if np.any(np.isposinf(margins) != single_valid_node):
        raise ValueError("infinite margins must identify rows with one valid node")
    if np.any(~np.isfinite(entropy)) or np.any(entropy < 0.0):
        raise ValueError("entropy must be finite and nonnegative")
    return valid_nodes


def _validate_index_values(
    indices: np.ndarray,
    valid_nodes: np.ndarray,
    name: str,
) -> None:
    node_count = valid_nodes.shape[1]
    if np.any((indices < 0) | (indices >= node_count)):
        raise ValueError(f"{name} contains an index outside node capacity")
    rows = np.arange(indices.shape[0])
    if np.any(~valid_nodes[rows, indices]):
        raise ValueError(f"{name} must identify valid nodes")


def _validate_top_k(top_k: np.ndarray, valid_nodes: np.ndarray) -> None:
    node_count = valid_nodes.shape[1]
    if np.any((top_k < 0) | (top_k >= node_count)):
        raise ValueError("top_k_indices contains an index outside node capacity")
    rows = np.arange(top_k.shape[0])[:, None]
    if np.any(~valid_nodes[rows, top_k]):
        raise ValueError("top_k_indices must identify valid nodes")
    if any(len(set(row.tolist())) != len(row) for row in top_k):
        raise ValueError("top_k_indices must be unique within each row")


def _int32_array(value: jax.Array, name: str, *, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if array.dtype != np.dtype("int32"):
        raise TypeError(f"{name} must have dtype int32")
    return array


def _float32_array(value: jax.Array, name: str, *, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if array.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    return array


def _integer_reference_indices(
    value: jax.Array | np.ndarray,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer indices")
    return array.astype(np.int64, copy=False)
