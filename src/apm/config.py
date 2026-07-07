"""Shared filesystem defaults for datasets, experiments, and smoke artifacts."""

from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MNIST_CACHE_PATH = DATA_DIR / "mnist.npz"
MNIST_RAW_DIR = DATA_DIR / "mnist" / "raw"
LEGACY_TORCHVISION_MNIST_RAW_DIR = DATA_DIR / "torchvision" / "MNIST" / "raw"
