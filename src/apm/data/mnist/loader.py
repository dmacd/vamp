"""Direct IDX MNIST loader used by the continual-learning task builders."""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
from numpy.typing import NDArray

from apm.config import LEGACY_TORCHVISION_MNIST_RAW_DIR, MNIST_CACHE_PATH, MNIST_RAW_DIR
from apm.data.mnist.label_canvas import DIGIT_SIZE

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

MNIST_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
MNIST_RESOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
)
MNIST_RAW_FILES = tuple(filename.removesuffix(".gz") for filename, _ in MNIST_RESOURCES)


@dataclass(frozen=True)
class MnistArrays:
    """Frozen container for normalized MNIST train/test arrays."""

    train_images: FloatArray
    train_labels: IntArray
    test_images: FloatArray
    test_labels: IntArray


def _normalize_images(images: NDArray[np.number]) -> FloatArray:
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim == 2 and image_array.shape[1] == DIGIT_SIZE * DIGIT_SIZE:
        image_array = image_array.reshape(image_array.shape[0], DIGIT_SIZE, DIGIT_SIZE)
    if image_array.ndim != 3 or image_array.shape[1:] != (DIGIT_SIZE, DIGIT_SIZE):
        raise ValueError(f"MNIST images must have shape (n, 28, 28), got {image_array.shape}")
    scale = 255.0 if float(image_array.max(initial=0.0)) > 1.0 else 1.0
    return (image_array / scale).astype(np.float32)


def _normalize_labels(labels: NDArray[np.number]) -> IntArray:
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError(f"MNIST labels must have shape (n,), got {label_array.shape}")
    if label_array.size and (int(label_array.min()) < 0 or int(label_array.max()) > 9):
        raise ValueError("MNIST labels must be in [0, 9]")
    return label_array


def _load_cached_npz(cache_path: Path) -> MnistArrays:
    with np.load(cache_path) as mnist_npz:
        train_images = mnist_npz["x_train"]
        train_labels = mnist_npz["y_train"]
        test_images = mnist_npz["x_test"]
        test_labels = mnist_npz["y_test"]
    return MnistArrays(
        train_images=_normalize_images(train_images),
        train_labels=_normalize_labels(train_labels),
        test_images=_normalize_images(test_images),
        test_labels=_normalize_labels(test_labels),
    )


def _md5_digest(path: Path) -> str:
    hash_state = hashlib.md5()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            hash_state.update(chunk)
    return hash_state.hexdigest()


def _verify_md5(path: Path, expected_md5: str) -> None:
    actual_md5 = _md5_digest(path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}")


def _extract_gzip(gzip_path: Path, output_path: Path) -> None:
    with gzip.open(gzip_path, "rb") as compressed_file, output_path.open("wb") as output_file:
        shutil.copyfileobj(compressed_file, output_file)


def _download_resource(raw_dir: Path, filename: str, expected_md5: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    gzip_path = raw_dir / filename
    if not gzip_path.exists():
        errors: list[Exception] = []
        for mirror in MNIST_MIRRORS:
            try:
                urlretrieve(f"{mirror}{filename}", gzip_path)
                break
            except (RuntimeError, URLError, OSError) as error:
                errors.append(error)
        else:
            raise RuntimeError(
                f"Could not download {filename}; tried {', '.join(MNIST_MIRRORS)}"
            ) from errors[-1]
    _verify_md5(gzip_path, expected_md5)
    _extract_gzip(gzip_path, raw_dir / filename.removesuffix(".gz"))


def _raw_files_exist(raw_dir: Path) -> bool:
    return all((raw_dir / filename).exists() for filename in MNIST_RAW_FILES)


def _candidate_raw_dirs(root: Path) -> tuple[Path, ...]:
    root_candidates = (root, root / "raw", root / "MNIST" / "raw")
    return (
        root_candidates + (LEGACY_TORCHVISION_MNIST_RAW_DIR,)
        if root == MNIST_RAW_DIR
        else root_candidates
    )


def _preferred_raw_dir(root: Path) -> Path:
    return root if root.name == "raw" else root / "raw"


def _ensure_raw_dir(root: Path, allow_download: bool) -> Path:
    existing_raw_dirs = [raw_dir for raw_dir in _candidate_raw_dirs(root) if _raw_files_exist(raw_dir)]
    if existing_raw_dirs:
        return existing_raw_dirs[0]
    if not allow_download:
        candidates = ", ".join(str(raw_dir) for raw_dir in _candidate_raw_dirs(root))
        raise FileNotFoundError(
            f"MNIST IDX files not found. Looked in: {candidates}. "
            "Pass allow_download=True to download them."
        )
    raw_dir = _preferred_raw_dir(root)
    for filename, expected_md5 in MNIST_RESOURCES:
        _download_resource(raw_dir, filename, expected_md5)
    return raw_dir


def _idx_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as input_file:
            return input_file.read()
    return path.read_bytes()


def _read_idx_images(path: Path) -> FloatArray:
    raw_bytes = _idx_bytes(path)
    if len(raw_bytes) < 16:
        raise ValueError(f"IDX image file is too short: {path}")
    magic, image_count, rows, cols = struct.unpack(">IIII", raw_bytes[:16])
    if magic != 2051:
        raise ValueError(f"Expected IDX image magic 2051 in {path}, got {magic}")
    if rows != DIGIT_SIZE or cols != DIGIT_SIZE:
        raise ValueError(f"Expected {DIGIT_SIZE}x{DIGIT_SIZE} MNIST images in {path}, got {rows}x{cols}")
    expected_bytes = image_count * rows * cols
    image_bytes = raw_bytes[16:]
    if len(image_bytes) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} image bytes in {path}, got {len(image_bytes)}")
    return np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_count, rows, cols).astype(np.float32)


def _read_idx_labels(path: Path) -> IntArray:
    raw_bytes = _idx_bytes(path)
    if len(raw_bytes) < 8:
        raise ValueError(f"IDX label file is too short: {path}")
    magic, label_count = struct.unpack(">II", raw_bytes[:8])
    if magic != 2049:
        raise ValueError(f"Expected IDX label magic 2049 in {path}, got {magic}")
    label_bytes = raw_bytes[8:]
    if len(label_bytes) != label_count:
        raise ValueError(f"Expected {label_count} label bytes in {path}, got {len(label_bytes)}")
    return np.frombuffer(label_bytes, dtype=np.uint8).astype(np.int64)


def _load_raw_mnist(raw_dir: Path) -> MnistArrays:
    return MnistArrays(
        train_images=_normalize_images(_read_idx_images(raw_dir / "train-images-idx3-ubyte")),
        train_labels=_normalize_labels(_read_idx_labels(raw_dir / "train-labels-idx1-ubyte")),
        test_images=_normalize_images(_read_idx_images(raw_dir / "t10k-images-idx3-ubyte")),
        test_labels=_normalize_labels(_read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")),
    )


def load_mnist(
    root: str | Path = MNIST_RAW_DIR,
    allow_download: bool = False,
    npz_cache_path: str | Path | None = MNIST_CACHE_PATH,
) -> MnistArrays:
    """Load real MNIST from raw IDX files, optionally falling back to an explicit npz cache."""
    resolved_root = Path(root)
    if npz_cache_path is not None:
        resolved_cache_path = Path(npz_cache_path)
        if resolved_cache_path.exists():
            return _load_cached_npz(resolved_cache_path)
    return _load_raw_mnist(_ensure_raw_dir(resolved_root, allow_download=allow_download))


def load_mnist_npz(cache_path: str | Path = MNIST_CACHE_PATH) -> MnistArrays:
    """Load MNIST arrays from an explicit Keras-style npz cache."""
    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.exists():
        raise FileNotFoundError(f"MNIST npz cache not found at {resolved_cache_path}.")
    return _load_cached_npz(resolved_cache_path)
