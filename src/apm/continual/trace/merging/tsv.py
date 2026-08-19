"""Task Singular Vector merge matching the pinned Core Space implementation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def tsv_merge(matrices: Sequence[Tensor]) -> Tensor:
    """Merge equal-shaped matrices using disjoint leading singular-vector blocks."""
    if len(matrices) < 2 or len({tuple(matrix.shape) for matrix in matrices}) != 1:
        raise ValueError("TSV requires at least two equal-shaped matrices")
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("TSV inputs must be matrices")
    decomposition = tuple(
        torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
        for matrix in matrices
    )
    left_template, singular_template, right_template = decomposition[0]
    retained_per_task = singular_template.shape[0] // len(matrices)
    if retained_per_task < 1:
        raise ValueError("TSV Core Space is too small for the task count")
    sum_left = torch.zeros_like(left_template)
    sum_singular = torch.zeros_like(singular_template)
    sum_right = torch.zeros_like(right_template)
    for index, (left, singular, right) in enumerate(decomposition):
        block = slice(index * retained_per_task, (index + 1) * retained_per_task)
        sum_left[:, block] = left[:, :retained_per_task]
        sum_singular[block] = singular[:retained_per_task]
        sum_right[block, :] = right[:retained_per_task, :]
    left_left, _, left_right = torch.linalg.svd(sum_left, full_matrices=False)
    right_left, _, right_right = torch.linalg.svd(sum_right, full_matrices=False)
    return torch.linalg.multi_dot(
        (left_left, left_right, torch.diag(sum_singular), right_left, right_right)
    ).to(dtype=matrices[0].dtype)


__all__ = ["tsv_merge"]
