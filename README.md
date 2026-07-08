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

Install the optional FabricPC backend dependency with:

```bash
ve/bin/python -m pip install -e '.[dev,pc]'
```

On CUDA-capable Linux hosts, install the GPU-enabled JAX runtime with:

```bash
ve/bin/python -m pip install -e '.[dev,pc,gpu]'
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

Run the related digit-incremental benchmark, where each task contains one
MNIST digit with global labels, with:

```bash
ve/bin/python scripts/run_stage1_digit_apm.py
```

The default digit-incremental run writes the same report/artifact set under:

```text
results/stage1_apm/digit_mnist_dense_delta/
```

Run the FabricPC MNIST generative spike with:

```bash
ve/bin/python scripts/run_fabricpc_mnist_spike.py
```

The spike uses the same 32x32 digit-plus-label canvas as the VAE experiments,
keeps the top latent node unconstrained, and writes loss, MSE, accuracy, and
reconstruction artifacts under:

```text
results/fabricpc_mnist_spike/
```
