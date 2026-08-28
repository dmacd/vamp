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
> **TinyWorlds Nouns-v2 is the only supported benchmark.** TRACE Log-t VAMP is
> a completed, sealed research run with published reviewer evidence, but its
> one-seed/one-order result is not a supported benchmark claim. The
> MNIST/predictive-coding work, TinyShakespeare work, TinyStories work, and all
> other TinyWorlds variants are notional designs, negative results, or
> deprecated prototypes retained for research provenance.

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
| VAMP-AF Rotated MNIST | Completed negative routing result | Full-rank top-two deltas cleared the 98.36% oracle-context capacity gate, but three-seed AF averaged 60.32% versus 62.66% global replay and 99.23% oracle-leaf accuracy. Hard routing agreed with the oracle leaf only 4.97% of the time. |
| LogT NCE/TRE Rotated MNIST | Completed negative routing result | The corrected protocol sampled complete images from the authenticated frozen CNN's 60,000-image training distribution. Calibration passed, but all four TRE schedules failed every static routing gate, so the staged protocol correctly stopped before consolidation and online evaluation. |
| Generative-PC LogT MNIST | Completed negative MAP/GN routing result | The 80-step MAP and exact generalized Gauss–Newton protocols failed all three minimal static controls. Both scores consistently favored the larger history model: the new leaf won 0 of 512 focused comparisons in every replica, including when both nodes represented the same image distribution. No confirmation or partial carry was run. |
| **TRACE Log-t VAMP** | **Completed research run** | Eight-task Llama-3.2-1B continual-learning study with SVD/Core controls, replay repair, four original routers, matched baselines, a CPU-only [task-known provenance follow-up](docs/experiments/trace-logt-vamp/followups/task-known-provenance/report.md), and a complete [reviewer bundle](docs/experiments/trace-logt-vamp/README.md). |
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

Python 3.11 or newer is required. Create the local environment and run the
default non-integration test suite with:

```bash
python3.11 -m venv ve
ve/bin/python -m pip install -e '.[dev]'
ve/bin/python -m pytest
```

The default CPU suite uses four `pytest-xdist` workers with work stealing.
Integration and benchmark tests remain opt-in through their pytest markers.

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

## TRACE Log-t VAMP Runner

The repository also contains a complete implementation of
the TRACE/Llama-3.2-1B-Instruct log-t temporal-consolidation experiment. It
trains 40 immutable LoRA leaves once, builds SVD and Core+TSV capacity-two
hierarchies with optional deterministic replay repair, evaluates two task-free
routers and two diagnostics, and runs the required baselines through a
resumable two-GPU job DAG. The primary run and two leaf-reusing Core-scale
controls completed all 562 jobs. The sealed report, raw candidate generations,
logs, ledgers, manifests, integrity records, and review guide are published in
the [TRACE reviewer bundle](docs/experiments/trace-logt-vamp/README.md).
The bundle also contains a reproducible
[task-known provenance follow-up](docs/experiments/trace-logt-vamp/followups/task-known-provenance/report.md)
that re-scores the sealed generations without loading model weights or running
new inference.

The canonical model remains Meta's exact revision
`9213176726f574b556790deb65791e0c5aa438b6`. Downloading does not require gated
Hugging Face access: TRACE pins public source
`alpindale/Llama-3.2-1B-Instruct` at revision
`f92201d8185818a9d079b3b52efdab4b68bdd17f` and authenticates the model,
tokenizer, and configuration bytes against Meta's repository metadata before
training. In particular, the 2.47-GB BF16 safetensor is byte-identical, with
SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.

The direct deployment uses RunPod's public PyTorch image; it needs neither a
custom registry nor a saved template. RunPod supplies the Pod-scoped API key,
Pod ID, and attached-volume ID to an image entrypoint. For a direct SSH-driven
deployment, place those three values in the launched process environment. An
optional `VAMP_NOTIFY_WEBHOOK_URL` enables notifications.

```bash
export RUNPOD_API_KEY=<runpod-key>
export TRACE_NETWORK_VOLUME_ID=<network-volume-id>
scripts/trace/launch_runpod.sh
```

After copying this checkout to `/workspace/vamp-trace/source/apm`, prepare the
isolated runtime with `scripts/trace/prepare_direct_runpod.sh`. The Dockerfile
remains available as an optional prebuilt-image path, not a deployment
requirement.

All durable state and model/data caches remain below
`/workspace/vamp-trace/`. A replacement two-4090 Pod resumes without repeating
completed work:

```bash
python -m apm.continual.trace.cli status \
  --run /workspace/vamp-trace/runs/<run-contract-hash>
python -m apm.continual.trace.cli resume \
  --run /workspace/vamp-trace/runs/<run-contract-hash>
```

After the primary DAG completes, a new Core scale or repair fraction reuses all
leaf adapters and records the reuse acceptance check:

```bash
python -m apm.continual.trace.cli rebuild-policy \
  --run /workspace/vamp-trace/runs/<run-contract-hash> \
  --policy configs/trace/policy_sweep_example.yaml
```

The source experiment contract is
`docs/TRACE Log-t VAMP Experiment Specification.pdf`; completed implementation
status and durable semantics are recorded in `PLAN.md` and `DESIGN.md`.

## VAMP-AF MNIST Runner

The Addressable Rotated MNIST mechanism POC uses the existing `vision` extra
and local MNIST IDX cache. One command runs or resumes the shared frozen CNN,
representation preflight, real smoke, three paired main seeds, and forced
consolidation stress pass:

```bash
uv run python -m apm.experiments.vamp_af_mnist \
  --config configs/vamp_af_mnist/poc.yaml
```

The `top-two-v3` workflow is complete. Its adapter-capacity preflight passed,
but the aggregate POC did not: three-seed AF averaged 60.32%, below global
replay at 62.66% and far below the 97.06% oracle-context control. Mean
oracle-leaf accuracy was 99.23%, but hard routing selected that leaf only 4.97%
of the time. The forced-consolidation drop gate passed. Rerunning the command
authenticates and reuses the completed checkpoints. Full checkpoints and
generated evidence are kept below the ignored `artifacts/vamp-af-mnist/` tree;
the protocol and output interpretation are summarized in
[the experiment note](docs/experiments/vamp_af_mnist.md).

## LogT NCE/TRE MNIST Runner

The follow-up runner authenticates the completed VAMP-AF base and data, then
runs normalized-ratio calibration, a genuine 63-block LogT static routing
gate, a block-64 consolidation control, and the three-seed 100-block online
comparison in that order. It stops and writes a report when a required gate
fails. Evidence models receive raw quantized images only; adapter features,
labels, and context identifiers are excluded from their API.

```bash
uv run python -m apm.experiments.vamp_logt_evidence_mnist \
  --config configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml
```

The first canonical run completed on 2026-08-26 with a controlled stop after
static selection, but it used independent uniform pixels as the common
reference. That run remains an immutable negative control at
`artifacts/vamp-logt-evidence-mnist/runs/2003268ae73e22544cf9801d58b3fa40e724ff58c70bc31c32b120fdebf38b54/`;
it does not answer the intended base-reference question. The corrected protocol
uses the uniform empirical distribution over all 60,000 original unrotated
MNIST training images used by the sealed frozen CNN. It samples one intact donor
image per adjacent pair, binds the reference tensor hash, and makes complete
replacement exactly that distribution.

The corrected canonical GPU run is
`fa2b8bf7d301b0c096d35cdbd6af1ed9b9369ee7e376d96d74e384019417ef49`.
Calibration passed, but the run stopped at the static gate because none of
K=2, 4, 8, or 16 passed. Direct NCE averaged 51.49% routed classifier accuracy.
K=4 had the best TRE mean at 58.51%, but its worst replica remained 41.65
percentage points behind the label-aware oracle. Across candidates, the most
separable adjacent bridge had 99.10% to 100% balanced accuracy, above the 90%
maximum, and the least stable seed's independent-route agreement was only
6.48% to 67.92%, below the 90% minimum. Consolidation and online evaluation
were therefore deliberately not run.

## Generative-PC Evidence MNIST Runner

The generative predictive-coding experiment trains one normalized image model
and one stopped-gradient digit classifier per active LogT node. Its sole
task-free routing value is the complete log joint of the image and inferred
state after 80 fixed inference steps; higher values win. This MAP joint score
includes the latent prior and all Gaussian normalization constants, but it is
not a marginal likelihood. Labels are available only to train the node-local
classifier and to calculate the diagnostic oracle.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[pc,gpu]'
.venv/bin/python -m apm.experiments.vamp_logt_pc_mnist \
  --config configs/vamp_logt_pc_mnist/minimal.yaml
```

The completed `generative-pc-map-v1` protocol performs no Hessian, Laplace, or
importance-sampling calculation. Its one-node model passed preflight, but MAP
routing failed all three controlled 31-block schedules on the minimal stream
seed. In every replica, the new leaf lost all 512 focused comparisons to the
larger history model. The identical-regime control measured a 724.18-nat median
history advantage even though both compared nodes represented C4; the allowed
offset was 19.88 nats. Here C4 is the 72-degree-rotated, label-shifted context,
not the digit 4. The runner therefore stopped before confirmation and partial
carry. Historical 40-step and 80-step Laplace preflight failures remain
immutable under their original run IDs.

The exact generalized Gauss–Newton continuation can reproduce the authenticated
MAP models and score the same 80-step states with GN0 and GN1:

```bash
.venv/bin/python -m apm.experiments.vamp_logt_pc_gn_mnist \
  --config configs/vamp_logt_pc_mnist/gauss_newton_v2.yaml
```

Its raw G matrix factorized successfully for all 38,016 minimal-condition
states, but neither GN score passed a condition. Only 2 of 11,520 GN route
decisions were close enough for the measured float32-versus-float64 discrepancy
to plausibly change their ordering. The decisive failure was the remaining
approximately 721-nat cross-level bias in the identical-regime control, not
floating-point precision.

## Repository Layout

- `src/apm/memory/`: immutable parameter graphs and addressing mechanisms.
- `src/apm/lm/`: plain-JAX GPT-Neo, LoRA, training, and evaluation components.
- `src/apm/data/mnist/`: MNIST task streams used by the early VAE and PC work.
- `src/apm/data/text/`: TinyShakespeare, TinyStories, and TinyWorlds datasets.
- `scripts/`: fixed experiment entry points and artifact builders.
- `docs/`: architecture notes, execution records, plans, and negative results.
- `results/`: selected presentation reports and only their direct media
  dependencies.

Raw corpora, checkpoints, optimizer state, and intermediate run artifacts are
normally excluded from Git. The sealed TRACE reviewer bundle is the deliberate
exception: its raw candidate-generation JSONL files are stored with Git LFS,
while caches, checkpoints, adapter tensors, and prompt embeddings remain on the
retained evidence volume.
