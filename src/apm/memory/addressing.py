"""Address selection and evaluation for dense memory graphs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from apm.data.mnist.task_specs import TaskDataset
from apm.models.vae_losses import flatten_canvases
from apm.training import TrainConfig, evaluate_vae, per_example_observed_energy
from apm.memory.dense import DenseMemoryGraph, effective_params, node_ids


@dataclass(frozen=True)
class AddressedEvaluation:
    """Metrics and selected-address diagnostics for one evaluated task."""

    metrics: dict[str, float]
    selected_node_ids: tuple[str, ...]
    selected_counts: dict[str, int]
    address_accuracy: float
    mean_selected_energy: float


def observed_energy_matrix(
    graph: DenseMemoryGraph,
    canvases: np.ndarray,
    rng_key: jax.Array,
    train_config: TrainConfig,
    candidate_node_ids: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Return [examples, candidates] observed-digit energies for candidate memory nodes."""
    candidates = node_ids(graph) if candidate_node_ids is None else candidate_node_ids
    return np.stack(
        [
            np.asarray(
                per_example_observed_energy(
                    effective_params(graph, candidate_id),
                    canvases,
                    jax.random.fold_in(rng_key, candidate_index),
                    train_config.beta,
                )
            )
            for candidate_index, candidate_id in enumerate(candidates)
        ],
        axis=1,
    ).astype(np.float32)


def select_addresses(
    graph: DenseMemoryGraph,
    canvases: np.ndarray,
    rng_key: jax.Array,
    train_config: TrainConfig,
    candidate_node_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Select the lowest observed-energy address for each example."""
    candidates = node_ids(graph) if candidate_node_ids is None else candidate_node_ids
    score_matrix = observed_energy_matrix(graph, canvases, rng_key, train_config, candidates)
    winner_indices = np.argmin(score_matrix, axis=1)
    return tuple(candidates[int(index)] for index in winner_indices), score_matrix


def best_parent_by_observed_energy(
    graph: DenseMemoryGraph,
    canvases: np.ndarray,
    rng_key: jax.Array,
    train_config: TrainConfig,
    probe_count: int = 1024,
) -> str:
    """Pick the node with lowest mean observed-digit energy on a train probe."""
    probe_canvases = canvases[: min(probe_count, canvases.shape[0])]
    candidates = node_ids(graph)
    score_matrix = observed_energy_matrix(graph, probe_canvases, rng_key, train_config, candidates)
    return candidates[int(np.argmin(np.mean(score_matrix, axis=0)))]


def evaluate_node_on_task(
    graph: DenseMemoryGraph,
    node_id: str,
    task: TaskDataset,
    rng_key: jax.Array,
    train_config: TrainConfig,
) -> dict[str, float]:
    """Evaluate one graph node on one task's test canvases."""
    return evaluate_vae(
        effective_params(graph, node_id),
        flatten_canvases(task.test_canvases()),
        task.test_labels,
        rng_key,
        train_config,
    )


def evaluate_addressed_on_task(
    graph: DenseMemoryGraph,
    task: TaskDataset,
    oracle_node_id: str,
    rng_key: jax.Array,
    train_config: TrainConfig,
    candidate_node_ids: tuple[str, ...] | None = None,
) -> AddressedEvaluation:
    """Evaluate addressed memory on a task by grouping examples under selected nodes."""
    canvases = task.test_canvases()
    selected_node_ids, score_matrix = select_addresses(graph, canvases, rng_key, train_config, candidate_node_ids)
    selected_array = np.asarray(selected_node_ids, dtype=object)
    metrics_by_node = tuple(
        (
            node_id,
            int(np.sum(selected_array == node_id)),
            evaluate_vae(
                effective_params(graph, node_id),
                flatten_canvases(canvases[selected_array == node_id]),
                task.test_labels[selected_array == node_id],
                jax.random.fold_in(rng_key, node_index + 10_000),
                train_config,
            ),
        )
        for node_index, node_id in enumerate(sorted(set(selected_node_ids)))
    )
    metrics = _weighted_metrics(metrics_by_node)
    selected_counts = {node_id: count for node_id, count, _ in metrics_by_node}
    winner_indices = np.argmin(score_matrix, axis=1)
    mean_selected_energy = float(np.mean(score_matrix[np.arange(score_matrix.shape[0]), winner_indices]))
    return AddressedEvaluation(
        metrics=metrics,
        selected_node_ids=selected_node_ids,
        selected_counts=selected_counts,
        address_accuracy=float(np.mean(selected_array == oracle_node_id)),
        mean_selected_energy=mean_selected_energy,
    )


def address_confusion_matrix(
    rows: tuple[tuple[str, str, int], ...],
    task_names: tuple[str, ...],
    node_id_order: tuple[str, ...],
) -> np.ndarray:
    """Build a [task, node] count matrix from selected-address count rows."""
    matrix = np.zeros((len(task_names), len(node_id_order)), dtype=np.float32)
    task_index = {task_name: index for index, task_name in enumerate(task_names)}
    node_index = {node_id: index for index, node_id in enumerate(node_id_order)}
    for task_name, node_id, count in rows:
        matrix[task_index[task_name], node_index[node_id]] += float(count)
    return matrix


def _weighted_metrics(metrics_by_node: tuple[tuple[str, int, dict[str, float]], ...]) -> dict[str, float]:
    total_count = sum(count for _, count, _ in metrics_by_node)
    return {
        metric_name: float(sum(metrics[metric_name] * count for _, count, metrics in metrics_by_node) / total_count)
        for metric_name in metrics_by_node[0][2]
    }
