"""Deterministic digit-region permutations for MNIST continual-learning tasks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from apm.data.mnist.label_canvas import CANVAS_SIZE, DIGIT_SIZE

FloatArray = NDArray[np.float32]
Permutation = NDArray[np.int64]


def _validate_permutation(permutation: NDArray[np.integer]) -> Permutation:
    permutation_array = np.asarray(permutation, dtype=np.int64)
    expected_size = DIGIT_SIZE * DIGIT_SIZE
    if permutation_array.shape != (expected_size,):
        raise ValueError(f"permutation must have shape ({expected_size},), got {permutation_array.shape}")
    if not np.array_equal(np.sort(permutation_array), np.arange(expected_size, dtype=np.int64)):
        raise ValueError("permutation must contain each digit-region index exactly once")
    return permutation_array


def identity_permutation() -> Permutation:
    """Return the identity permutation over the 28x28 digit region."""
    return np.arange(DIGIT_SIZE * DIGIT_SIZE, dtype=np.int64)


def random_digit_permutation(seed: int) -> Permutation:
    """Return a deterministic full random permutation over the 28x28 digit region."""
    return np.random.default_rng(seed).permutation(DIGIT_SIZE * DIGIT_SIZE).astype(np.int64)


def near_swap_permutation(seed: int, swap_fraction: float = 0.10) -> Permutation:
    """Return a mild permutation where a fraction of digit pixels are pair-swapped."""
    if swap_fraction < 0.0 or swap_fraction > 1.0:
        raise ValueError(f"swap_fraction must be in [0, 1], got {swap_fraction}")
    rng = np.random.default_rng(seed)
    permutation = identity_permutation()
    participating_count = int(round(permutation.size * swap_fraction))
    participating_count -= participating_count % 2
    if participating_count == 0:
        return permutation
    participating_positions = rng.choice(permutation.size, size=participating_count, replace=False)
    first_positions = participating_positions[0::2]
    second_positions = participating_positions[1::2]
    permutation[first_positions], permutation[second_positions] = (
        permutation[second_positions].copy(),
        permutation[first_positions].copy(),
    )
    return permutation.astype(np.int64)


def apply_digit_permutation(image_or_canvas: FloatArray, permutation: NDArray[np.integer]) -> FloatArray:
    """Apply a permutation to only the 28x28 digit region of an image or canvas."""
    permutation_array = _validate_permutation(permutation)
    input_array = np.asarray(image_or_canvas, dtype=np.float32)
    if input_array.shape == (DIGIT_SIZE, DIGIT_SIZE):
        return input_array.reshape(-1)[permutation_array].reshape(DIGIT_SIZE, DIGIT_SIZE).astype(np.float32)
    if input_array.shape == (CANVAS_SIZE, CANVAS_SIZE):
        output = np.array(input_array, copy=True)
        output[:DIGIT_SIZE, :DIGIT_SIZE] = (
            input_array[:DIGIT_SIZE, :DIGIT_SIZE]
            .reshape(-1)[permutation_array]
            .reshape(DIGIT_SIZE, DIGIT_SIZE)
        )
        return output.astype(np.float32)
    raise ValueError(f"expected 28x28 image or 32x32 canvas; got shape {input_array.shape}")


def apply_digit_permutation_batch(images: FloatArray, permutation: NDArray[np.integer]) -> FloatArray:
    """Apply a digit-region permutation to a batch of 28x28 images or 32x32 canvases."""
    permutation_array = _validate_permutation(permutation)
    input_array = np.asarray(images, dtype=np.float32)
    if input_array.ndim != 3:
        raise ValueError(f"expected a rank-3 image or canvas batch; got shape {input_array.shape}")
    if input_array.shape[1:] == (DIGIT_SIZE, DIGIT_SIZE):
        return input_array.reshape(input_array.shape[0], -1)[:, permutation_array].reshape(
            input_array.shape[0], DIGIT_SIZE, DIGIT_SIZE
        ).astype(np.float32)
    if input_array.shape[1:] == (CANVAS_SIZE, CANVAS_SIZE):
        output = np.array(input_array, copy=True)
        output[:, :DIGIT_SIZE, :DIGIT_SIZE] = (
            input_array[:, :DIGIT_SIZE, :DIGIT_SIZE]
            .reshape(input_array.shape[0], -1)[:, permutation_array]
            .reshape(input_array.shape[0], DIGIT_SIZE, DIGIT_SIZE)
        )
        return output.astype(np.float32)
    raise ValueError(f"expected 28x28 image batch or 32x32 canvas batch; got shape {input_array.shape}")
