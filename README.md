# VAMP

**Virtually Addressed Memory for Parameters (VAMP)** is an experimental
architectural paradigm for continual learning. Instead of repeatedly mutating
one model, VAMP stores immutable parameter updates in a dependency graph. At
inference time, a task-free router addresses a node from the observed input and
executes the base model plus the parameter deltas on that node's path.

The architecture is described in
[Virtually Addressed Memory for Parameters](docs/Virtually_Addressed_Memory_for_Parameters.pdf).
The repository explores dense parameter memories, predictive-coding networks,
and pathwise LoRA memories for language models.

> [!IMPORTANT]
> **TinyWorlds Nouns-v2 is the only experiment currently in usable shape.**
> The MNIST/predictive-coding work, TinyShakespeare work, TinyStories work, and
> all other TinyWorlds variants are notional designs, negative results, or
> deprecated prototypes. They remain in the repository for research provenance
> and should not be treated as supported benchmarks.

## TinyWorlds Nouns-v2

Nouns-v2 is the completed VAMP experiment. It creates 24 disjoint continual
learning tasks from a pinned TinyStories archive: stories naming no selected
noun family train the base model, stories naming exactly one family train that
task, and stories naming multiple selected families are excluded. A fresh
GPT-Neo base is extended with immutable rank-eight LoRA edges, and task-free
routing is compared with sequential LoRA, full-model fine-tuning, independent
task adapters, and a task-aware VAMP oracle.

The benchmark contains 429,199 task-training stories and 4,440 held-out
validation stories. Every task was evaluated from its introduction through the
final stage. The stored VAMP nodes had exactly zero NLL drift, while the
task-free exhaustive router finished at 73.9% route accuracy and 1.572
true-suffix story NLL. On the same final evaluation, sequential LoRA reached
1.746 NLL with +0.2156 mean forgetting, and sequential full-model fine-tuning
reached 2.048 NLL with +0.5744 mean forgetting. These are results for this
specific benchmark, not a claim of general continual-learning performance.

![TinyWorlds Nouns-v2 continual-learning comparison](results/language_cl/tinyworlds-nouns-v2/continual-nll-comparison.svg)

![Learned TinyWorlds Nouns-v2 VAMP graph](results/language_cl/tinyworlds-nouns-v2/vamp-graph.svg)

- [Readable Markdown report](results/language_cl/tinyworlds-nouns-v2/report.md)
- [Standalone folding HTML report](results/language_cl/tinyworlds-nouns-v2/report.html)
- [Execution and verification record](docs/TINYWORLDS_NOUNS_V2_EXECUTION_REPORT.md)

## Experiment Map

| Experiment family | Current status | Scope and retained evidence |
|---|---|---|
| **TinyWorlds Nouns-v2** | **Usable** | Completed 24-task disjoint language benchmark with immutable VAMP stages, matched controls, routing audits, and a published [report](results/language_cl/tinyworlds-nouns-v2/report.md). |
| TinyShakespeare | Deprecated prototype | Four-task character-level language-model experiments used to establish the pathwise LoRA and routing machinery. A selected [character-permutation report](results/language_cl/tinyshakespeare/character-permutation/standard-seed0-a7bd7d1479ba/report.html) is retained as historical evidence. |
| MNIST and predictive coding | Deprecated prototype | Early label-canvas VAE and FabricPC predictive-coding experiments established dense-delta graphs, energy-based addressing, and report tooling. The selected [digit-incremental FabricPC report](results/stage1_apm/digit_mnist_dense_delta_fabricpc_energy_converged/report.html) is not a current benchmark result. |
| Other TinyWorlds and TinyStories variants | Notional, failed, or deprecated | Dataset constructions, semantic partition attempts, pilots, and negative results are retained under `docs/`, `scripts/`, and versioned source packages for provenance. |

## How VAMP Is Organized

1. Train or load a base model at the graph root.
2. For each new task, score candidate parents and train a parameter delta from
   the selected parent without changing previously committed edges.
3. Store content keys and routing statistics separately from the immutable
   parameter graph.
4. At evaluation, address a node from input-only evidence and compose the root
   with the deltas along that path.

This separation makes parameter retention and task-free addressing distinct
measurements. An oracle path can verify whether old functions were retained;
router metrics then measure whether the correct path can still be found as the
graph grows.

## Development Setup

Python 3.10 or newer is required. Create the local environment and run the
default non-integration test suite with:

```bash
python -m venv ve
ve/bin/python -m pip install -e '.[dev]'
ve/bin/python -m pytest
```

The language-model experiments additionally require the `lm` extra. CUDA hosts
can request the JAX CUDA 12 runtime through the `gpu` extra:

```bash
ve/bin/python -m pip install -e '.[dev,lm,gpu]'
```

The canonical prepared nouns-v2 run is resumable through:

```bash
ve/bin/python scripts/run_tinyworlds_nouns_v2.py
```

It is a full research run, not a quick-start demo: it expects the authenticated
parent corpus artifacts and uses substantial GPU time. The exact data,
checkpoint, evaluation, and resume contracts are recorded in the
[execution report](docs/TINYWORLDS_NOUNS_V2_EXECUTION_REPORT.md).

## Repository Layout

- `src/apm/memory/`: immutable parameter graphs and addressing mechanisms.
- `src/apm/lm/`: plain-JAX GPT-Neo, LoRA, training, and evaluation components.
- `src/apm/data/mnist/`: MNIST task streams used by the early VAE and PC work.
- `src/apm/data/text/`: TinyShakespeare, TinyStories, and TinyWorlds datasets.
- `scripts/`: fixed experiment entry points and artifact builders.
- `docs/`: architecture notes, execution records, plans, and negative results.
- `results/`: selected presentation reports and only their direct media
  dependencies.

Raw corpora, checkpoints, optimizer state, large JSONL evaluation ledgers, and
intermediate run artifacts are intentionally excluded from Git. Published
result directories contain only the human-readable report surface and the
SVG/PNG or small text files directly required to render it.
