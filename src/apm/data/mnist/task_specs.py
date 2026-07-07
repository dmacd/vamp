"""Immutable task metadata and deterministic task construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyrsistent import PClass, field

from apm.data.mnist.label_canvas import embed_batch_digits_and_labels
from apm.data.mnist.loader import MnistArrays
from apm.data.mnist.permutations import apply_digit_permutation_batch, identity_permutation

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class TaskSpec(PClass):
    """Immutable metadata for one MNIST continual-learning task."""

    name = field(type=str, mandatory=True)
    digit_subset = field(type=tuple, mandatory=True)
    permutation_id = field(type=str, mandatory=True)
    regime_kind = field(type=str, mandatory=True)
    label_mapping = field(type=str, initial="global_10way")


@dataclass(frozen=True)
class TaskDataset:
    """Frozen task arrays plus immutable task metadata."""

    spec: TaskSpec
    train_images: FloatArray
    train_labels: IntArray
    test_images: FloatArray
    test_labels: IntArray

    def train_canvases(self) -> FloatArray:
        """Materialize 32x32 label-embedded canvases for this task's train split."""
        return embed_batch_digits_and_labels(self.train_images, self.train_labels)

    def test_canvases(self) -> FloatArray:
        """Materialize 32x32 label-embedded canvases for this task's test split."""
        return embed_batch_digits_and_labels(self.test_images, self.test_labels)


def _validated_digits(digits: tuple[int, ...]) -> tuple[int, ...]:
    digit_tuple = tuple(int(digit) for digit in digits)
    if not digit_tuple:
        raise ValueError("digit_subset must not be empty")
    if len(set(digit_tuple)) != len(digit_tuple):
        raise ValueError(f"digit_subset contains duplicates: {digit_tuple}")
    if any(digit < 0 or digit > 9 for digit in digit_tuple):
        raise ValueError(f"digit_subset values must be in [0, 9], got {digit_tuple}")
    return digit_tuple


def _split_name(digits: tuple[int, ...]) -> str:
    return "S" + "".join(str(digit) for digit in digits)


def _filtered_arrays(
    images: FloatArray, labels: IntArray, digits: tuple[int, ...]
) -> tuple[FloatArray, IntArray]:
    digit_mask = np.isin(labels, np.asarray(digits, dtype=np.int64))
    return images[digit_mask].astype(np.float32), labels[digit_mask].astype(np.int64)


def _permuted_task_arrays(
    arrays: MnistArrays, digits: tuple[int, ...], permutation: NDArray[np.integer]
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    train_images, train_labels = _filtered_arrays(arrays.train_images, arrays.train_labels, digits)
    test_images, test_labels = _filtered_arrays(arrays.test_images, arrays.test_labels, digits)
    return (
        apply_digit_permutation_batch(train_images, permutation),
        train_labels,
        apply_digit_permutation_batch(test_images, permutation),
        test_labels,
    )


def make_split_task(
    arrays: MnistArrays,
    digits: tuple[int, ...],
    permutation: NDArray[np.integer] | None = None,
    permutation_id: str = "P0",
) -> TaskDataset:
    """Build a SplitMNIST task with global labels and optional digit permutation."""
    digit_tuple = _validated_digits(digits)
    permutation_array = identity_permutation() if permutation is None else permutation
    train_images, train_labels, test_images, test_labels = _permuted_task_arrays(
        arrays, digit_tuple, permutation_array
    )
    spec = TaskSpec(
        name=f"{_split_name(digit_tuple)}_{permutation_id}",
        digit_subset=digit_tuple,
        permutation_id=permutation_id,
        regime_kind="split",
    )
    return TaskDataset(spec, train_images, train_labels, test_images, test_labels)


def make_permuted_task(
    arrays: MnistArrays,
    permutation: NDArray[np.integer],
    permutation_id: str,
) -> TaskDataset:
    """Build an all-digit PermutedMNIST task with global labels."""
    digit_tuple = tuple(range(10))
    train_images, train_labels, test_images, test_labels = _permuted_task_arrays(
        arrays, digit_tuple, permutation
    )
    spec = TaskSpec(
        name=f"ALL_{permutation_id}",
        digit_subset=digit_tuple,
        permutation_id=permutation_id,
        regime_kind="permuted",
    )
    return TaskDataset(spec, train_images, train_labels, test_images, test_labels)


def make_split_permuted_task(
    arrays: MnistArrays,
    digits: tuple[int, ...],
    permutation: NDArray[np.integer],
    permutation_id: str,
) -> TaskDataset:
    """Build a SplitPermutedMNIST task with global labels."""
    digit_tuple = _validated_digits(digits)
    train_images, train_labels, test_images, test_labels = _permuted_task_arrays(
        arrays, digit_tuple, permutation
    )
    spec = TaskSpec(
        name=f"{_split_name(digit_tuple)}_{permutation_id}",
        digit_subset=digit_tuple,
        permutation_id=permutation_id,
        regime_kind="split_permuted",
    )
    return TaskDataset(spec, train_images, train_labels, test_images, test_labels)
