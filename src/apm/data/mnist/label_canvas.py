"""Utilities for embedding MNIST digits and labels into one generative canvas."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DIGIT_SIZE = 28
CANVAS_SIZE = 32
LABEL_CLASSES = 10
LABEL_CELL_WIDTH = 2
LABEL_PATCH_ROWS = slice(30, 32)
LABEL_PATCH_COLS = slice(0, LABEL_CLASSES * LABEL_CELL_WIDTH)

FloatArray = NDArray[np.float32]


def _validate_label(label: int) -> int:
    label_int = int(label)
    if label_int < 0 or label_int >= LABEL_CLASSES:
        raise ValueError(f"label must be in [0, {LABEL_CLASSES - 1}], got {label}")
    return label_int


def _digit_image(image: FloatArray) -> FloatArray:
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.shape == (DIGIT_SIZE * DIGIT_SIZE,):
        return image_array.reshape(DIGIT_SIZE, DIGIT_SIZE)
    if image_array.shape == (DIGIT_SIZE, DIGIT_SIZE):
        return image_array
    if image_array.shape == (CANVAS_SIZE, CANVAS_SIZE):
        return image_array[:DIGIT_SIZE, :DIGIT_SIZE]
    raise ValueError(
        "expected a 28x28 digit image, flattened 784-vector, or 32x32 canvas; "
        f"got shape {image_array.shape}"
    )


def embed_digit_and_label(image: FloatArray, label: int) -> FloatArray:
    """Embed a 28x28 digit and a global 10-way label patch into a 32x32 canvas."""
    label_int = _validate_label(label)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    canvas[:DIGIT_SIZE, :DIGIT_SIZE] = _digit_image(image)
    cell_start = label_int * LABEL_CELL_WIDTH
    canvas[LABEL_PATCH_ROWS, cell_start : cell_start + LABEL_CELL_WIDTH] = 1.0
    return canvas


def embed_batch_digits_and_labels(images: FloatArray, labels: NDArray[np.integer]) -> FloatArray:
    """Vectorize label-canvas embedding over matching batches of images and labels."""
    image_array = np.asarray(images, dtype=np.float32)
    label_array = np.asarray(labels)
    if image_array.shape[0] != label_array.shape[0]:
        raise ValueError("images and labels must have the same leading dimension")
    return np.stack(
        [embed_digit_and_label(image, int(label)) for image, label in zip(image_array, label_array)],
        axis=0,
    ).astype(np.float32)


def mask_label_patch(canvas: FloatArray) -> FloatArray:
    """Return a copy of a canvas or canvas batch with the reserved label patch zeroed."""
    canvas_array = np.asarray(canvas, dtype=np.float32)
    masked = np.array(canvas_array, copy=True)
    if masked.shape == (CANVAS_SIZE, CANVAS_SIZE):
        masked[LABEL_PATCH_ROWS, LABEL_PATCH_COLS] = 0.0
        return masked
    if masked.ndim == 3 and masked.shape[1:] == (CANVAS_SIZE, CANVAS_SIZE):
        masked[:, LABEL_PATCH_ROWS, LABEL_PATCH_COLS] = 0.0
        return masked
    raise ValueError(f"expected 32x32 canvas or canvas batch; got shape {masked.shape}")


def decode_label_patch(canvas: FloatArray) -> int:
    """Decode the argmax label from the reserved 10-cell label patch."""
    canvas_array = np.asarray(canvas, dtype=np.float32)
    if canvas_array.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(f"expected a single 32x32 canvas; got shape {canvas_array.shape}")
    patch = canvas_array[LABEL_PATCH_ROWS, LABEL_PATCH_COLS].reshape(2, LABEL_CLASSES, LABEL_CELL_WIDTH)
    return int(patch.mean(axis=(0, 2)).argmax())


def candidate_label_canvas(image: FloatArray, candidate_label: int) -> FloatArray:
    """Create a canvas from an image or canvas digit region with a candidate label patch."""
    return embed_digit_and_label(_digit_image(image), candidate_label)
