from __future__ import annotations

import gzip
import hashlib
import struct

import numpy as np
import pytest

from apm.data.mnist import load_mnist, load_mnist_npz
from apm.data.mnist.loader import _read_idx_images, _read_idx_labels, _verify_md5


def test_load_mnist_from_local_npz(tmp_path) -> None:
    cache_path = tmp_path / "mnist.npz"
    np.savez(
        cache_path,
        x_train=np.full((2, 28, 28), 255, dtype=np.uint8),
        y_train=np.asarray([0, 9], dtype=np.uint8),
        x_test=np.zeros((1, 28, 28), dtype=np.uint8),
        y_test=np.asarray([5], dtype=np.uint8),
    )

    arrays = load_mnist_npz(cache_path)

    assert arrays.train_images.dtype == np.float32
    assert arrays.train_labels.dtype == np.int64
    assert arrays.train_images.shape == (2, 28, 28)
    assert float(arrays.train_images.max()) == 1.0
    assert arrays.train_labels.tolist() == [0, 9]


def test_missing_mnist_cache_requires_explicit_download(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mnist_npz(tmp_path / "missing.npz")


def test_load_mnist_prefers_explicit_npz_cache(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "mnist.npz"
    np.savez(
        cache_path,
        x_train=np.zeros((1, 28, 28), dtype=np.uint8),
        y_train=np.asarray([1], dtype=np.uint8),
        x_test=np.zeros((1, 28, 28), dtype=np.uint8),
        y_test=np.asarray([1], dtype=np.uint8),
    )
    monkeypatch.setattr(
        "apm.data.mnist.loader._ensure_raw_dir",
        lambda root, allow_download: pytest.fail("IDX files should not load when npz exists"),
    )

    arrays = load_mnist(root=tmp_path / "mnist" / "raw", npz_cache_path=cache_path)

    assert arrays.train_labels.tolist() == [1]


def test_read_idx_images_and_labels(tmp_path) -> None:
    image_path = tmp_path / "images-idx3-ubyte"
    label_path = tmp_path / "labels-idx1-ubyte"
    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    labels = np.asarray([3, 8], dtype=np.uint8)
    _write_idx_images(image_path, images)
    _write_idx_labels(label_path, labels)

    np.testing.assert_array_equal(_read_idx_images(image_path), images.astype(np.float32))
    np.testing.assert_array_equal(_read_idx_labels(label_path), labels.astype(np.int64))


def test_read_idx_gzip_files(tmp_path) -> None:
    image_path = tmp_path / "images-idx3-ubyte.gz"
    label_path = tmp_path / "labels-idx1-ubyte.gz"
    images = np.ones((1, 28, 28), dtype=np.uint8)
    labels = np.asarray([4], dtype=np.uint8)
    _write_gzip(image_path, _idx_image_bytes(images))
    _write_gzip(label_path, _idx_label_bytes(labels))

    np.testing.assert_array_equal(_read_idx_images(image_path), images.astype(np.float32))
    np.testing.assert_array_equal(_read_idx_labels(label_path), labels.astype(np.int64))


def test_verify_md5_rejects_corrupted_file(tmp_path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"not the expected payload")

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        _verify_md5(path, hashlib.md5(b"expected payload").hexdigest())


def test_load_mnist_from_raw_idx_files(tmp_path) -> None:
    raw_dir = tmp_path / "mnist" / "raw"
    _write_raw_mnist(raw_dir)

    arrays = load_mnist(root=raw_dir, npz_cache_path=tmp_path / "missing.npz")

    assert arrays.train_images.shape == (2, 28, 28)
    assert arrays.test_images.shape == (1, 28, 28)
    assert arrays.train_images.dtype == np.float32
    assert arrays.train_labels.tolist() == [0, 9]
    assert float(arrays.train_images.max()) == 1.0


def test_load_mnist_errors_without_npz_or_idx_cache(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="MNIST IDX files not found"):
        load_mnist(
            root=tmp_path / "mnist" / "raw",
            allow_download=False,
            npz_cache_path=tmp_path / "missing.npz",
        )


def test_load_mnist_uses_legacy_torchvision_raw_cache(tmp_path, monkeypatch) -> None:
    legacy_raw_dir = tmp_path / "torchvision" / "MNIST" / "raw"
    _write_raw_mnist(legacy_raw_dir)
    monkeypatch.setattr("apm.data.mnist.loader.LEGACY_TORCHVISION_MNIST_RAW_DIR", legacy_raw_dir)

    arrays = load_mnist(
        allow_download=False,
        npz_cache_path=tmp_path / "missing.npz",
    )

    assert arrays.train_labels.tolist() == [0, 9]


def _idx_image_bytes(images: np.ndarray) -> bytes:
    image_array = np.asarray(images, dtype=np.uint8)
    header = struct.pack(">IIII", 2051, image_array.shape[0], image_array.shape[1], image_array.shape[2])
    return header + image_array.tobytes()


def _idx_label_bytes(labels: np.ndarray) -> bytes:
    label_array = np.asarray(labels, dtype=np.uint8)
    return struct.pack(">II", 2049, label_array.shape[0]) + label_array.tobytes()


def _write_idx_images(path, images: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_idx_image_bytes(images))


def _write_idx_labels(path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_idx_label_bytes(labels))


def _write_gzip(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as output_file:
        output_file.write(payload)


def _write_raw_mnist(raw_dir) -> None:
    train_images = np.stack(
        [np.zeros((28, 28), dtype=np.uint8), np.full((28, 28), 255, dtype=np.uint8)], axis=0
    )
    test_images = np.full((1, 28, 28), 127, dtype=np.uint8)
    _write_idx_images(raw_dir / "train-images-idx3-ubyte", train_images)
    _write_idx_labels(raw_dir / "train-labels-idx1-ubyte", np.asarray([0, 9], dtype=np.uint8))
    _write_idx_images(raw_dir / "t10k-images-idx3-ubyte", test_images)
    _write_idx_labels(raw_dir / "t10k-labels-idx1-ubyte", np.asarray([5], dtype=np.uint8))
