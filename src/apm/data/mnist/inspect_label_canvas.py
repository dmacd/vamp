"""Write a smoke visual grid of 32x32 label-embedded MNIST canvases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apm.config import RESULTS_DIR
from apm.data.mnist.label_canvas import CANVAS_SIZE, embed_batch_digits_and_labels
from apm.data.mnist.loader import load_mnist

FloatArray = NDArray[np.float32]


def _synthetic_digits() -> tuple[FloatArray, NDArray[np.int64]]:
    images = np.zeros((10, 28, 28), dtype=np.float32)
    labels = np.arange(10, dtype=np.int64)
    for label in labels:
        images[label, 3 + label % 10 : 6 + label % 10, 4:24] = 0.55
        images[label, 4:24, 3 + (label * 2) % 18 : 6 + (label * 2) % 18] = 1.0
    return images, labels


def _example_canvases() -> FloatArray:
    try:
        mnist = load_mnist(allow_download=False)
        images, labels = mnist.train_images[:10], mnist.train_labels[:10]
    except (FileNotFoundError, ImportError):
        images, labels = _synthetic_digits()
    return embed_batch_digits_and_labels(images, labels)


def _grid(canvases: FloatArray, columns: int = 5, pad: int = 2) -> FloatArray:
    rows = int(np.ceil(canvases.shape[0] / columns))
    grid = np.zeros(
        (rows * CANVAS_SIZE + (rows - 1) * pad, columns * CANVAS_SIZE + (columns - 1) * pad),
        dtype=np.float32,
    )
    for index, canvas in enumerate(canvases):
        row, col = divmod(index, columns)
        row_start = row * (CANVAS_SIZE + pad)
        col_start = col * (CANVAS_SIZE + pad)
        grid[row_start : row_start + CANVAS_SIZE, col_start : col_start + CANVAS_SIZE] = canvas
    return grid


def _write_pgm(path: Path, image: FloatArray) -> None:
    scaled = np.clip(image, 0.0, 1.0)
    image_bytes = (scaled * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        output_file.write(f"P5\n{image_bytes.shape[1]} {image_bytes.shape[0]}\n255\n".encode("ascii"))
        output_file.write(image_bytes.tobytes())


def main() -> None:
    """Generate the default smoke grid under results/smoke/label_canvas_grid.pgm."""
    output_path = RESULTS_DIR / "smoke" / "label_canvas_grid.pgm"
    _write_pgm(output_path, _grid(_example_canvases()))
    print(output_path)


if __name__ == "__main__":
    main()
