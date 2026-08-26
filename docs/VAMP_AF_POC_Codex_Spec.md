# VAMP-AF Minimal Proof-of-Concept
## Codex implementation spec and experimental plan

**Repository:** `dmacd/vamp`
**Experiment name:** `vamp-af-mnist`
**Implemented protocol revision:** `top-two-v3`
**Primary objective:** Test whether a single binary tree can jointly define (1) deterministic addressing regions in a frozen embedding space and (2) adapter ancestry, without prompt-NLL, energy scoring, or exhaustive adapter comparison.

---

## 1. What this experiment must answer

1. Can a tree grown only from frozen input embeddings discover useful address regions in an unlabeled non-IID stream?
2. Can adapters attached to the same tree learn useful specializations while preserving ancestor behavior?
3. Does hard tree routing select nearly the best adapter path available in the tree?
4. Can a subtree be replaced by one replay-trained parent adapter without any weight merging?
5. Do routing, splitting, replay, and consolidation costs behave consistently with an amortized \(O(t\log t)\) design?
6. Which parts of the algorithm remain underspecified or unstable?

This is a mechanism proof, not a state-of-the-art continual-learning benchmark.

---

## 2. Explicit non-goals

Do **not** add:

- prompt-NLL, energy-based, Hopfield, kNN, or exhaustive inference-time routing;
- LoRA merging, task arithmetic, SVD consolidation, distillation, functional signatures, or learned split criteria;
- mutable address embeddings;
- soft routing or mixtures of leaves;
- tree rotations or global tree rebuilding;
- reservoir compression in the first implementation;
- ImageNet-R until this MNIST experiment is understood.

The first implementation stores all frozen embeddings and labels. Memory is therefore \(O(t)\). The experiment targets online compute and algorithmic clarity, not sublinear storage.

---

## 3. Benchmark: Addressable Rotated MNIST

Use a controlled domain-incremental MNIST variant designed to require routing.

### 3.1 Base model

Train a small CNN on ordinary MNIST:

```text
Conv(1, 32, 3, padding=1) -> ReLU -> MaxPool(2)
Conv(32, 64, 3, padding=1) -> ReLU -> MaxPool(2)
Flatten
Linear(64*7*7, 128) -> ReLU          # frozen address embedding z
Linear(128, 10)                       # frozen base classifier
```

Train to normal convergence on standard MNIST, save the checkpoint, then freeze the entire model.

Use the normalized 128-dimensional penultimate activation as the address:

\[
z(x)=\frac{\phi(x)}{\|\phi(x)\|_2+\epsilon}.
\]

### 3.2 Continual stream

Create five contexts:

```yaml
rotations_deg: [0, 18, 36, 54, 72]
label_shifts:  [0, 2, 4, 6, 8]
```

For context \(c\):

\[
x_c=\operatorname{rotate}(x,\alpha_c),\qquad
y_c=(y+s_c)\bmod 10.
\]

Present contexts in blocked order, with no context ID available to AF:

```text
context 0 -> context 1 -> context 2 -> context 3 -> context 4
```

Recommended main size:

- 10,000 deterministic training examples per context;
- the complete transformed MNIST test set per context;
- three random seeds after the smoke run.

The context ID may be stored only for diagnostics and the oracle baseline. It must never enter AF routing or adapter training.

### 3.3 Representation preflight

Before the stream experiment, run three frozen-router checks:

1. Train one linear context classifier from \(z\). Report 80% as the original reference, but do not gate AF on it.
2. Freeze both convolutional layers and train one full-rank top-two-layer adapter per context. Mean context test accuracy must be at least 90%.
3. Train one joint top-two-layer adapter on all contexts.

Interpretation:

- Low context-probe accuracy: geometric context recovery may be difficult, but this diagnostic alone does not rule out useful adapter regions.
- Low oracle-adapter accuracy: the frozen convolutional trunk or adapter class is insufficient.
- Joint accuracy equal to oracle accuracy: the benchmark does not require routing.

If the oracle-adapter check fails, stop and ask for guidance. Do not modify AF itself to rescue a bad benchmark.

---

## 4. AF model

Each tree node is simultaneously:

1. an address-space region;
2. an adapter ancestor;
3. a replay/consolidation unit.

### 4.1 Node adapter

Let \(h(x)\in\mathbb{R}^{3136}\) be the frozen output of the two convolutional
blocks. Every node stores full-rank deltas for the embedding and classifier:

\[
\Delta_v=(\Delta E_v,\Delta e_v,\Delta C_v,\Delta c_v).
\]

For a query routed to leaf \(\ell\), accumulate each delta in root-to-leaf
order:

\[
E_\ell=E_0+\sum_{v\in P(\ell)}\Delta E_v,\qquad
C_\ell=C_0+\sum_{v\in P(\ell)}\Delta C_v,
\]

\[
a_\ell(x)=\operatorname{ReLU}(E_\ell h(x)+e_\ell),\qquad
g_\ell(x)=C_\ell a_\ell(x)+c_\ell.
\]

Routing still uses the immutable base address
\(z_0(x)=\operatorname{normalize}(\operatorname{ReLU}(E_0h(x)+e_0))\).
A root-only AF model therefore has the same function class as the global
top-two-layer baseline without making the address mutable.

Initialize every new adapter to zero.

### 4.2 Node record

```python
@dataclass
class AFNode:
    node_id: int
    parent_id: int | None
    depth: int
    adapter: TopTwoLayerDelta
    optimizer_state: Any

    # Internal-node routing rule; None for a leaf.
    split_direction: Tensor | None
    split_threshold: float | None
    left_id: int | None
    right_id: int | None

    # Each example belongs to exactly one current leaf buffer.
    examples: list[StoredExample]

    total_arrivals: int
    arrivals_since_structure_change: int
    last_consolidated_subtree_size: int
    created_at_step: int
```

```python
@dataclass(frozen=True)
class StoredExample:
    embedding: Tensor       # frozen z, preferably CPU float32
    trunk_features: Tensor  # frozen h(x), CPU float32
    base_logits: Tensor     # frozen g0(x), CPU float32
    label: int
    context_id: int         # diagnostics only
    stream_step: int
```

Internal nodes do not need duplicate example storage. Their represented examples are the union of descendant leaf buffers.

---

## 5. Routing

Routing is a pure tree traversal:

```python
def route(root: AFNode, z: Tensor) -> list[AFNode]:
    path = [root]
    node = root
    while not node.is_leaf:
        score = dot(node.split_direction, z)
        node = node.left if score <= node.split_threshold else node.right
        path.append(node)
    return path
```

Requirements:

- no label, context ID, adapter loss, energy, or candidate evaluation;
- exactly one leaf per query;
- all hyperplanes are immutable after installation;
- log the number of split evaluations for every query.

---

## 6. Online adapter update

Process the stream in microbatches, default 64.

1. Compute frozen addresses, trunk features, and base logits.
2. Route each example independently.
3. Group examples by destination leaf.
4. For each touched leaf, perform one optimizer step on:
   - its newly arrived examples; plus
   - an equal-size random replay sample from that leaf's existing buffer.
5. Update only the destination leaf adapter.
6. Freeze every internal-node adapter permanently.
7. Append new examples to the destination leaf buffer.
8. Check the structural trigger.

Default optimizer:

```yaml
optimizer: adamw
adapter_lr: 0.001
weight_decay: 0.0001
replay_ratio: 1.0
```

Ancestor adapters are evaluated but not updated. This makes sibling isolation testable and avoids changing the meaning of every descendant suffix after each arrival.

---

## 7. Splitting

### 7.1 Trigger

Use one deliberately simple rule:

```text
split when:
    leaf.arrivals_since_structure_change >= leaf_capacity
    and leaf.depth < current_depth_cap(t)
```

Defaults:

```yaml
leaf_capacity: 512
depth_cap:
  type: logarithmic
  value: 1 + ceil(log2(1 + t / leaf_capacity))
```

Depth may be smaller than the cap. The cap is an upper bound, not a target.

### 7.2 Hyperplane

Fit a local PCA-median split using only the leaf's frozen embeddings:

1. center the embeddings;
2. compute the leading principal direction \(w\);
3. project each embedding onto \(w\);
4. set threshold \(b\) to the median projection.

This produces a deterministic balanced split of the historical leaf contents without using targets or context IDs.

Use at most `split_fit_samples=2048` uniformly sampled examples to estimate the principal direction, but partition the complete leaf buffer with the resulting hyperplane.

### 7.3 Split operation

```python
def split_leaf(leaf):
    w, b = fit_pca_median_hyperplane(leaf.examples)

    left = new_zero_adapter_child()
    right = new_zero_adapter_child()

    partition every stored example into exactly one child
    install (w, b) on leaf
    freeze leaf.adapter
```

Immediately after installation, both child adapters are zero, so for every example:

\[
g_{\text{before split}}(x)=g_{\text{after split}}(x).
\]

This is a required unit test.

Then replay-train each child adapter on its assigned buffer for a fixed two epochs, with all ancestors frozen. This is child initialization, not consolidation.

---

## 8. Replay-only consolidation

The normal run may not naturally force consolidation. Implement one minimal operator and exercise it in a forced stress run.

### 8.1 Trigger

When a full leaf cannot split because it is at the current depth cap:

1. find the nearest ancestor whose two children are both leaves;
2. permit consolidation only if its current subtree size is at least twice its `last_consolidated_subtree_size`;
3. otherwise continue ordinary leaf replay until the depth cap grows.

No loss, compatibility, adapter norm, or merge heuristic enters the trigger.

### 8.2 Operation: collapse two leaf children into their parent

For parent \(p\):

1. Gather the union of both child buffers.
2. Freeze ancestors above \(p\).
3. Initialize a replacement adapter from the old adapter at \(p\); do not combine child weights.
4. Replay-train that replacement adapter on the union buffer relative to the fixed ancestor deltas.
5. Replace \(p\)'s adapter atomically.
6. Delete both children and their adapters.
7. Make \(p\) a leaf containing the union buffer.
8. Set `arrivals_since_structure_change = 0`.
9. Set `last_consolidated_subtree_size` to the union size.
10. Record accuracy/loss immediately before and after replacement.

Use a fixed three replay epochs. This is deliberately plain behavioral consolidation: the parent relearns the union from examples. There is no LoRA or weight merging.

### 8.3 Forced consolidation run

If no collapse occurs naturally, rerun AF with:

```yaml
depth_cap_override: 3
```

Use one seed. Its purpose is only to determine whether replay collapse is coherent and whether split-collapse oscillation appears.

---

## 9. Required comparison conditions

Keep the matrix small.

| Condition | Addressing | Parameters | Training |
|---|---|---|---|
| Frozen base | none | base only | none after pretraining |
| Global replay | one global adapter | same top-two-layer class as AF | online, same replay ratio |
| AF | embedding partition tree | path-summed parameter deltas | online |
| Oracle context | true context selects one adapter | one top-two adapter per context | online |
| Joint IID reference | none | one top-two adapter | offline shuffled union |

The oracle-context condition is an upper bound on addressability with the chosen adapters. Joint IID is a reference, not an online competitor.

Do not add more routers in the first pass.

---

## 10. Essential diagnostics

### 10.1 Adapter competence versus routing

At evaluation time only, compute a label-aware oracle over all current leaves:

\[
\ell^*(x)=\arg\min_{\ell}\operatorname{CE}(g_\ell(x),y).
\]

Report:

- routed accuracy;
- oracle-leaf accuracy;
- routed loss minus oracle-leaf loss;
- fraction of examples for which routed leaf equals oracle leaf.

This is diagnostic only and may be exhaustive because MNIST and the tree are small. It must not affect training or prediction.

Interpretation:

- oracle-leaf high, routed low: addressing/partition failure;
- oracle-leaf low, oracle-context high: adapter learning or tree ancestry failure;
- both low: representation or adapter-capacity failure.

### 10.2 Tree diagnostics

At every evaluation boundary log:

- number of leaves and total nodes;
- maximum, mean, and percentile route depth;
- examples per leaf;
- context distribution and entropy per leaf;
- adapter norm per node;
- split and consolidation events;
- fraction of traffic per leaf;
- examples processed by online replay, split initialization, and consolidation.

Render a final tree diagram labeled with node ID, depth, sample count, dominant contexts, context entropy, and adapter norm.

### 10.3 Complexity diagnostics

Maintain counters rather than inferring complexity from wall time:

```text
embedding_evaluations
hyperplane_evaluations
adapter_evaluations
online_training_examples
split_replay_examples
consolidation_replay_examples
historical_examples_repartitioned
```

Plot cumulative work against \(t\log_2(t+1)\), and plot:

\[
\frac{\text{cumulative counted work}}{t\log_2(t+1)}.
\]

The proof of concept does not establish an asymptotic theorem from one run, but it should reveal accidental superlinear implementation work.

---

## 11. Evaluation schedule and outputs

Evaluate on every context test set:

- before the stream;
- after every 2,000 stream examples;
- immediately before and after every split;
- immediately before and after every consolidation;
- at the end of each context block;
- at the end of the stream.

Required artifacts:

```text
summary.json
metrics.jsonl
accuracy_matrix.csv
routing_diagnostics.csv
tree_final.json
tree_final.png
accuracy_over_time.png
routed_vs_oracle_leaf.png
tree_size_depth.png
complexity_scaling.png
consolidation_events.csv
config_resolved.yaml
HANDOFF.md
```

`HANDOFF.md` must summarize results, invariant failures, and every ambiguity encountered.

---

## 12. Acceptance criteria

Treat the core AF idea as provisionally supported when all are true:

1. All structural invariants and unit tests pass.
2. AF creates and uses multiple leaves without context labels.
3. AF final average accuracy is within 5 percentage points of oracle-context accuracy.
4. AF beats global replay by at least 5 percentage points on the addressable label-shift stream.
5. Oracle-leaf accuracy exceeds routed accuracy by no more than 3 percentage points after the final context.
6. Maximum route depth never exceeds the configured cap.
7. No obvious upward trend appears in counted-work divided by \(t\log(t+1)\).
8. Replay consolidation, when forced, loses no more than 3 percentage points immediately after its replay pass.

These are POC gates, not publication claims. Record exact outcomes even when a gate fails.

---

## 13. Failure interpretation

Use this decision table before changing the algorithm.

| Observation | Likely conclusion |
|---|---|
| Context probe is poor | Frozen address embedding lacks required context |
| Oracle context is poor | Adapter class or frozen representation is inadequate |
| Global replay matches oracle context | Benchmark does not require routing |
| Oracle leaf is high but AF route is poor | Geometric partition does not track adapter utility |
| Oracle leaf and AF route are both poor | Useful adapters were not learned or ancestry is wrong |
| Leaves are pure but accuracy is poor | Addressing works; adaptation is the failure |
| Leaves mix contexts but oracle-leaf gap is small | Context purity is not required; partition may be functionally adequate |
| Deep chain forms | Capacity/depth rule or split geometry is inadequate |
| Repeated split-collapse cycle | Consolidation cooldown/schedule is underspecified |
| Consolidation causes a large drop | One parent adapter cannot replay-fit the union at current capacity |
| Work ratio rises steadily | Hidden repeated scans/retraining violate the intended amortization |

Do not repair failures with new heuristics until the diagnostics identify which mechanism failed.

---

## 14. Ambiguities this experiment must explicitly report

The implementation report must answer:

1. Does PCA split by context, digit identity, or some mixture?
2. Does geometry-based similarity predict which examples benefit from the same adapter?
3. Is updating only the leaf sufficient, or is path-wide updating eventually needed?
4. How much split-initialization replay is required?
5. What happens to points close to a fixed hyperplane under later stream shifts?
6. Does the logarithmic depth cap merely delay needed refinement?
7. Does replay collapse produce useful shared competence or only erase specialization?
8. Does a collapsed node immediately need to split again?
9. How sensitive are results to leaf capacity?
10. Are disconnected regions learning duplicate adapters that a pure tree cannot share?
11. Does storing all historical embeddings materially simplify behavior that a bounded buffer would break?
12. Is hard single-leaf routing enough, or do boundary cases expose a need for multiple candidates?

Answer these from logged evidence; do not speculate in place of measurements.

---

## 15. Repository deliverables

Preferred layout:

```text
src/apm/continual/addressing_first.py
src/apm/experiments/vamp_af_mnist.py
configs/vamp_af_mnist/poc.yaml
tests/continual/test_addressing_first.py
tests/experiments/test_vamp_af_mnist_smoke.py
docs/experiments/vamp_af_mnist.md
```

Provide a direct module entry point even if the repository CLI is also registered:

```bash
uv run python -m apm.experiments.vamp_af_mnist \
  --config configs/vamp_af_mnist/poc.yaml
```

Follow the repository's current Python, formatting, typing, config, and artifact conventions.

### Required unit tests

1. Deterministic routing.
2. Every example appears in exactly one leaf buffer.
3. Split partitions are exhaustive and disjoint.
4. Predictions are identical immediately before and after a zero-child split.
5. Updating one child does not change a sibling's effective logits.
6. Route depth never exceeds the cap.
7. Collapse buffer equals the union of child buffers.
8. Child nodes become unreachable after collapse.
9. Complexity counters increment exactly once per operation.
10. Tiny end-to-end smoke run completes and writes all required artifacts.

---

## 16. Execution sequence

### Pass 0 — smoke

- 1,000 examples per context;
- one seed;
- `leaf_capacity=128`;
- no forced consolidation;
- run all unit tests and artifact checks.

### Pass 1 — main proof of concept

- 10,000 examples per context;
- three seeds;
- default configuration;
- run all five comparison conditions;
- no parameter sweep.

### Pass 2 — consolidation stress

- AF only;
- one seed;
- `depth_cap_override=3`;
- verify at least one replay collapse;
- diagnose fidelity loss and split-collapse behavior.

Stop after Pass 2. Do not proceed directly to ImageNet-R. Write `HANDOFF.md` with a recommendation:

- abandon or revise AF;
- clarify a named algorithmic ambiguity;
- or port the validated mechanism to the existing ImageNet-R-50 adapter experiment.

---

## 17. Codex completion checklist

- [ ] Implement the benchmark and frozen base checkpoint.
- [ ] Implement the one-tree AF data structure.
- [ ] Implement hard routing and path-summed adapters.
- [ ] Implement capacity-triggered PCA-median splitting.
- [ ] Implement leaf-local replay training.
- [ ] Implement replay-only sibling collapse.
- [ ] Implement all comparison conditions.
- [ ] Implement oracle-leaf diagnostics.
- [ ] Implement complexity counters and tree export.
- [ ] Add unit and smoke tests.
- [ ] Run Passes 0–2.
- [ ] Produce the required artifacts and `HANDOFF.md`.
- [ ] Do not add unrequested routing or consolidation machinery.
