# Addressed Parameter Memory Prototype

This repository contains research prototypes for addressed parameter memories
on continual-learning tasks. The first data module provides deterministic
SplitMNIST, PermutedMNIST, and SplitPermutedMNIST construction over a 32x32
label-embedded canvas; later modules should reuse `apm.models` and
`apm.training` across datasets.

Create and install the local development environment with:

```bash
python -m venv ve
ve/bin/python -m pip install -e '.[dev]'
```

Run the data-layer tests with:

```bash
ve/bin/python -m pytest
```

Generate a smoke visual grid with:

```bash
ve/bin/python -m apm.data.mnist.inspect_label_canvas
```

Download and load the real MNIST IDX dataset with:

```bash
ve/bin/python -c "from apm.data import load_mnist; arrays = load_mnist(allow_download=True); print(arrays.train_images.shape, arrays.test_images.shape)"
```

Train the stationary label-canvas VAE with:

```bash
ve/bin/python scripts/train_stationary_vae.py
```

The default run writes an HTML report, browser-viewable PNG grids, SVG curves,
raw metrics, and PGM grids under:

```text
results/stationary_vae/default/
```

Run the Stage 1 dense-delta addressed-parameter-memory benchmark with:

```bash
ve/bin/python scripts/run_stage1_apm.py
```

The default run writes metrics, curves, heatmaps, memory graph SVGs, and
reconstruction grids under:

```text
results/stage1_apm/permuted_mnist_dense_delta/
```
