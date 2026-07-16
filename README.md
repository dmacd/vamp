# VAMP Research Prototype

This repository contains research prototypes for **Virtually Addressed Memory
for Parameters (VAMP)**, an architectural continual-learning paradigm built
around virtually addressed parameter memories. The project reference
manuscript was lightly revised and renamed from
`docs/Addressed_Parameter_Memories.pdf` to
[Virtually Addressed Memory for Parameters](docs/Virtually_Addressed_Memory_for_Parameters.pdf).

The first data module provides deterministic SplitMNIST, PermutedMNIST, and
SplitPermutedMNIST construction over a 32x32 label-embedded canvas; later
modules should reuse `apm.models` and `apm.training` across datasets.

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

Run the Stage 1 dense-delta VAMP benchmark with:

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

Run the FabricPC digit benchmark until each task's observed-digit energy
converges with:

```bash
env XLA_PYTHON_CLIENT_PREALLOCATE=false ve/bin/python scripts/run_stage1_digit_apm.py \
  --model-kind fabricpc \
  --training-mode energy-convergence \
  --digits 0 1 2 3 4 5 6 7 8 9 \
  --train-count 6000 \
  --test-count 1000 \
  --replay-count 1000 \
  --parent-probe-count 1024 \
  --report-canvas-count 32
```

The convergence defaults are 10 minimum epochs, 100 maximum epochs, a 0.1%
relative improvement threshold, five stale epochs, and a fixed 1,024-example
training probe. The default output is
`results/stage1_apm/digit_mnist_dense_delta_fabricpc_energy_converged/` and
includes `training_convergence.jsonl` plus convergence curves and summaries in
the HTML report. Use `--help` to override each stopping parameter.

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
