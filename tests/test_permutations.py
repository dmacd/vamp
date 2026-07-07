from __future__ import annotations

import numpy as np

from apm.data.mnist.label_canvas import DIGIT_SIZE, LABEL_PATCH_COLS, LABEL_PATCH_ROWS, embed_digit_and_label
from apm.data.mnist.permutations import (
    apply_digit_permutation,
    apply_digit_permutation_batch,
    identity_permutation,
    near_swap_permutation,
    random_digit_permutation,
)


def test_identity_permutation_is_exact() -> None:
    image = np.arange(DIGIT_SIZE * DIGIT_SIZE, dtype=np.float32).reshape(DIGIT_SIZE, DIGIT_SIZE)
    np.testing.assert_allclose(apply_digit_permutation(image, identity_permutation()), image)


def test_random_permutation_is_seeded_and_digit_only() -> None:
    image = np.arange(DIGIT_SIZE * DIGIT_SIZE, dtype=np.float32).reshape(DIGIT_SIZE, DIGIT_SIZE)
    canvas = embed_digit_and_label(image, 4)
    permutation = random_digit_permutation(11)
    repeated = random_digit_permutation(11)
    permuted = apply_digit_permutation(canvas, permutation)

    np.testing.assert_array_equal(permutation, repeated)
    assert not np.array_equal(permutation, identity_permutation())
    np.testing.assert_allclose(
        permuted[LABEL_PATCH_ROWS, LABEL_PATCH_COLS],
        canvas[LABEL_PATCH_ROWS, LABEL_PATCH_COLS],
    )
    assert not np.array_equal(permuted[:DIGIT_SIZE, :DIGIT_SIZE], image)


def test_near_swap_permutation_is_deterministic_and_mild() -> None:
    permutation = near_swap_permutation(seed=5, swap_fraction=0.10)
    repeated = near_swap_permutation(seed=5, swap_fraction=0.10)
    changed_positions = int(np.count_nonzero(permutation != identity_permutation()))

    np.testing.assert_array_equal(permutation, repeated)
    assert 0 < changed_positions <= int(round(DIGIT_SIZE * DIGIT_SIZE * 0.10))


def test_batch_permutation_matches_single_image_permutation() -> None:
    images = np.stack(
        [
            np.arange(DIGIT_SIZE * DIGIT_SIZE, dtype=np.float32).reshape(DIGIT_SIZE, DIGIT_SIZE),
            np.ones((DIGIT_SIZE, DIGIT_SIZE), dtype=np.float32),
        ],
        axis=0,
    )
    permutation = random_digit_permutation(3)
    batch = apply_digit_permutation_batch(images, permutation)

    np.testing.assert_allclose(batch[0], apply_digit_permutation(images[0], permutation))
    np.testing.assert_allclose(batch[1], apply_digit_permutation(images[1], permutation))
