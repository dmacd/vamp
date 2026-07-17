"""Shared validation and slicing for bounded language-model evaluation."""

from __future__ import annotations


def validate_evaluation_microbatch_size(
    evaluation_microbatch_size: int | None,
) -> int | None:
    """Return a validated optional evaluation microbatch size."""
    if evaluation_microbatch_size is not None and (
        type(evaluation_microbatch_size) is not int
        or evaluation_microbatch_size <= 0
    ):
        raise ValueError(
            "evaluation_microbatch_size must be a positive integer when provided"
        )
    return evaluation_microbatch_size


def evaluation_microbatch_slices(
    row_count: int,
    evaluation_microbatch_size: int | None,
) -> tuple[slice, ...]:
    """Partition a nonempty row axis without padding or reordering it."""
    if type(row_count) is not int or row_count <= 0:
        raise ValueError("evaluation row_count must be a positive integer")
    microbatch_size = validate_evaluation_microbatch_size(
        evaluation_microbatch_size
    )
    step = row_count if microbatch_size is None else microbatch_size
    return tuple(
        slice(start, min(start + step, row_count))
        for start in range(0, row_count, step)
    )
