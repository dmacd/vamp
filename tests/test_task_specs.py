from __future__ import annotations

import numpy as np

from apm.data.mnist.label_canvas import decode_label_patch
from apm.data.mnist.permutations import identity_permutation, random_digit_permutation
from apm.data.mnist.task_specs import make_permuted_task, make_split_permuted_task, make_split_task


def test_make_split_task_filters_digits_and_uses_global_labels(synthetic_mnist_arrays) -> None:
    task = make_split_task(synthetic_mnist_arrays, digits=(0, 1))

    assert task.spec.name == "S01_P0"
    assert task.spec.digit_subset == (0, 1)
    assert task.spec.regime_kind == "split"
    assert task.spec.label_mapping == "global_10way"
    assert set(task.train_labels.tolist()) == {0, 1}
    assert set(task.test_labels.tolist()) == {0, 1}


def test_make_permuted_task_keeps_all_global_labels(synthetic_mnist_arrays) -> None:
    permutation = random_digit_permutation(17)
    task = make_permuted_task(synthetic_mnist_arrays, permutation=permutation, permutation_id="P1")

    assert task.spec.name == "ALL_P1"
    assert task.spec.digit_subset == tuple(range(10))
    assert task.spec.regime_kind == "permuted"
    assert set(task.train_labels.tolist()) == set(range(10))
    assert not np.array_equal(task.train_images[0], synthetic_mnist_arrays.train_images[0])


def test_make_split_permuted_task_materializes_label_canvases(synthetic_mnist_arrays) -> None:
    task = make_split_permuted_task(
        synthetic_mnist_arrays,
        digits=(0, 2),
        permutation=identity_permutation(),
        permutation_id="P0",
    )
    canvases = task.train_canvases()

    assert task.spec.name == "S02_P0"
    assert task.spec.regime_kind == "split_permuted"
    assert set(task.train_labels.tolist()) == {0, 2}
    assert [decode_label_patch(canvas) for canvas in canvases] == task.train_labels.tolist()
