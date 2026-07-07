from __future__ import annotations

import numpy as np

from apm.data.mnist.label_canvas import (
    CANVAS_SIZE,
    DIGIT_SIZE,
    LABEL_PATCH_COLS,
    LABEL_PATCH_ROWS,
    candidate_label_canvas,
    decode_label_patch,
    embed_batch_digits_and_labels,
    embed_digit_and_label,
    mask_label_patch,
)


def test_label_canvas_roundtrip() -> None:
    image = np.linspace(0.0, 1.0, DIGIT_SIZE * DIGIT_SIZE, dtype=np.float32).reshape(
        DIGIT_SIZE, DIGIT_SIZE
    )
    canvas = embed_digit_and_label(image, 7)

    assert canvas.shape == (CANVAS_SIZE, CANVAS_SIZE)
    np.testing.assert_allclose(canvas[:DIGIT_SIZE, :DIGIT_SIZE], image)
    assert decode_label_patch(canvas) == 7


def test_mask_label_patch_preserves_digit_region() -> None:
    image = np.ones((DIGIT_SIZE, DIGIT_SIZE), dtype=np.float32)
    canvas = embed_digit_and_label(image, 3)
    masked = mask_label_patch(canvas)

    np.testing.assert_allclose(masked[:DIGIT_SIZE, :DIGIT_SIZE], image)
    assert float(masked[LABEL_PATCH_ROWS, LABEL_PATCH_COLS].sum()) == 0.0


def test_candidate_label_canvas_changes_only_label_patch() -> None:
    image = np.eye(DIGIT_SIZE, dtype=np.float32)
    source = embed_digit_and_label(image, 1)
    candidate = candidate_label_canvas(source, 9)

    np.testing.assert_allclose(candidate[:DIGIT_SIZE, :DIGIT_SIZE], image)
    assert decode_label_patch(candidate) == 9


def test_batch_embedding_decodes_each_label() -> None:
    images = np.stack([np.full((DIGIT_SIZE, DIGIT_SIZE), label / 9.0, dtype=np.float32) for label in range(10)])
    labels = np.arange(10, dtype=np.int64)
    canvases = embed_batch_digits_and_labels(images, labels)

    assert canvases.shape == (10, CANVAS_SIZE, CANVAS_SIZE)
    assert [decode_label_patch(canvas) for canvas in canvases] == list(range(10))
