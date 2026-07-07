"""Deterministic MNIST task streams for continual-learning experiments."""

from __future__ import annotations

import numpy as np

from apm.data.mnist.loader import MnistArrays
from apm.data.mnist.permutations import identity_permutation, random_digit_permutation
from apm.data.mnist.task_specs import TaskDataset, make_permuted_task


def balanced_task_subset(
    task: TaskDataset,
    train_count: int,
    test_count: int,
    seed: int,
) -> TaskDataset:
    """Return a deterministic per-label balanced subset of a task dataset."""
    train_indices = _balanced_indices(task.train_labels, train_count, seed)
    test_indices = _balanced_indices(task.test_labels, test_count, seed + 10_000)
    return TaskDataset(
        spec=task.spec,
        train_images=task.train_images[train_indices],
        train_labels=task.train_labels[train_indices],
        test_images=task.test_images[test_indices],
        test_labels=task.test_labels[test_indices],
    )


def make_permuted_mnist_stream(
    arrays: MnistArrays,
    permutation_seeds: tuple[int, ...] = (0, 1, 2),
    train_count: int = 10_000,
    test_count: int = 2_000,
) -> tuple[TaskDataset, ...]:
    """Build the default all-digit PermutedMNIST stream for Stage 1 APM runs."""
    tasks = tuple(
        make_permuted_task(
            arrays,
            identity_permutation() if seed == 0 else random_digit_permutation(seed),
            f"P{index}",
        )
        for index, seed in enumerate(permutation_seeds)
    )
    return tuple(
        balanced_task_subset(task, train_count=train_count, test_count=test_count, seed=seed + index * 10_000)
        for index, (seed, task) in enumerate(zip(permutation_seeds, tasks))
    )


def _balanced_indices(labels: np.ndarray, requested_count: int, seed: int) -> np.ndarray:
    if requested_count <= 0:
        raise ValueError(f"requested_count must be positive, got {requested_count}")
    label_array = np.asarray(labels, dtype=np.int64)
    unique_labels = tuple(int(label) for label in np.unique(label_array))
    target_count = min(requested_count, label_array.shape[0])
    base_count, remainder = divmod(target_count, len(unique_labels))
    rng = np.random.default_rng(seed)
    selected_by_label = tuple(
        _label_indices(label_array, label, base_count + (1 if label_index < remainder else 0), rng)
        for label_index, label in enumerate(unique_labels)
    )
    return np.sort(np.concatenate(selected_by_label)).astype(np.int64)


def _label_indices(labels: np.ndarray, label: int, requested_count: int, rng: np.random.Generator) -> np.ndarray:
    candidates = np.flatnonzero(labels == label)
    count = min(requested_count, candidates.shape[0])
    return rng.choice(candidates, size=count, replace=False)
