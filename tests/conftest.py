from __future__ import annotations

import numpy as np
import pytest

from apm.data.mnist import MnistArrays


@pytest.fixture
def synthetic_mnist_arrays() -> MnistArrays:
    images = np.stack([_digit_pattern(label) for label in range(10) for _ in range(2)], axis=0)
    labels = np.asarray([label for label in range(10) for _ in range(2)], dtype=np.int64)
    return MnistArrays(
        train_images=images.astype(np.float32),
        train_labels=labels,
        test_images=images[:10].astype(np.float32),
        test_labels=np.arange(10, dtype=np.int64),
    )


def _digit_pattern(label: int) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.float32)
    image[label : label + 3, 2:26] = 0.25 + 0.05 * label
    image[2:26, label : label + 3] = 1.0
    return image
