from __future__ import annotations

import numpy as np

from apm.data.mnist import balanced_task_subset, make_digit_mnist_stream, make_permuted_mnist_stream
from apm.data.mnist.permutations import identity_permutation
from apm.data.mnist.task_specs import make_permuted_task


def test_balanced_task_subset_selects_equal_labels(synthetic_mnist_arrays) -> None:
    task = make_permuted_task(synthetic_mnist_arrays, identity_permutation(), "P0")

    subset = balanced_task_subset(task, train_count=10, test_count=10, seed=3)

    assert subset.train_images.shape[0] == 10
    assert subset.test_images.shape[0] == 10
    np.testing.assert_array_equal(np.bincount(subset.train_labels, minlength=10), np.ones(10, dtype=np.int64))
    np.testing.assert_array_equal(np.bincount(subset.test_labels, minlength=10), np.ones(10, dtype=np.int64))


def test_make_permuted_mnist_stream_is_deterministic(synthetic_mnist_arrays) -> None:
    left_stream = make_permuted_mnist_stream(synthetic_mnist_arrays, train_count=10, test_count=10)
    right_stream = make_permuted_mnist_stream(synthetic_mnist_arrays, train_count=10, test_count=10)

    assert tuple(task.spec.name for task in left_stream) == ("ALL_P0", "ALL_P1", "ALL_P2")
    assert tuple(task.spec.name for task in right_stream) == ("ALL_P0", "ALL_P1", "ALL_P2")
    for left_task, right_task in zip(left_stream, right_stream):
        np.testing.assert_array_equal(left_task.train_images, right_task.train_images)
        np.testing.assert_array_equal(left_task.train_labels, right_task.train_labels)


def test_make_digit_mnist_stream_has_one_digit_per_task(synthetic_mnist_arrays) -> None:
    stream = make_digit_mnist_stream(synthetic_mnist_arrays, digits=(0, 1, 2), train_count=2, test_count=1)

    assert tuple(task.spec.name for task in stream) == ("S0_P0", "S1_P0", "S2_P0")
    assert tuple(task.spec.digit_subset for task in stream) == ((0,), (1,), (2,))
    assert tuple(set(task.train_labels.tolist()) for task in stream) == ({0}, {1}, {2})
