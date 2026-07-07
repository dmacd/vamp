"""MNIST-family continual-learning datasets and label-canvas utilities."""

from apm.data.mnist.label_canvas import (
    CANVAS_SIZE,
    DIGIT_SIZE,
    LABEL_CELL_WIDTH,
    LABEL_CLASSES,
    LABEL_PATCH_COLS,
    LABEL_PATCH_ROWS,
    candidate_label_canvas,
    decode_label_patch,
    embed_batch_digits_and_labels,
    embed_digit_and_label,
    mask_label_patch,
)
from apm.data.mnist.loader import MnistArrays, load_mnist, load_mnist_npz
from apm.data.mnist.permutations import (
    apply_digit_permutation,
    apply_digit_permutation_batch,
    identity_permutation,
    near_swap_permutation,
    random_digit_permutation,
)
from apm.data.mnist.streams import balanced_task_subset, make_permuted_mnist_stream
from apm.data.mnist.task_specs import (
    TaskDataset,
    TaskSpec,
    make_permuted_task,
    make_split_permuted_task,
    make_split_task,
)

__all__ = [
    "CANVAS_SIZE",
    "DIGIT_SIZE",
    "LABEL_CELL_WIDTH",
    "LABEL_CLASSES",
    "LABEL_PATCH_COLS",
    "LABEL_PATCH_ROWS",
    "MnistArrays",
    "TaskDataset",
    "TaskSpec",
    "apply_digit_permutation",
    "apply_digit_permutation_batch",
    "balanced_task_subset",
    "candidate_label_canvas",
    "decode_label_patch",
    "embed_batch_digits_and_labels",
    "embed_digit_and_label",
    "identity_permutation",
    "load_mnist",
    "load_mnist_npz",
    "make_permuted_task",
    "make_permuted_mnist_stream",
    "make_split_permuted_task",
    "make_split_task",
    "mask_label_patch",
    "near_swap_permutation",
    "random_digit_permutation",
]
