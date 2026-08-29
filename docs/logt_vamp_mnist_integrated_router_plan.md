# LogT-VAMP Integrated Behavioral Router: MNIST Experiment Plan

## 1. Objective

Implement the first proof-of-concept experiment for an **integrated behavioral router** on top of the existing LogT-VAMP MNIST hierarchy.

The LogT hierarchy remains unchanged. At each macro-step, the router observes the behavior of every currently extant LogT node on an input and predicts which node will achieve the lowest supervised loss on that example.

The experiment must answer four questions:

1. Can a learned router approach exhaustive best-node selection at test time?
2. Does fixed-budget historical replay prevent the router from forgetting old routing decisions?
3. How do **example-balanced** and **range-balanced** replay differ?
4. Does the router preserve the intended total training complexity of `O(T log T)` over `T` macro-steps?

The router is an observer of the LogT hierarchy. It must not change node training, consolidation, addressing candidates, or node parameters.

---

## 2. Core hypotheses

### H1: Behavioral routing works

A router that receives hidden-state and output features from all extant nodes will close most of the gap between a simple fixed-node baseline and exhaustive best-node selection.

### H2: Router replay is necessary

A router trained only on the current macro-step will forget old routing distinctions. A fixed number of historical router examples per step will materially reduce old-range routing regret.

### H3: The two replay distributions make different tradeoffs

- **Example-balanced replay** should minimize average regret over historical examples because large temporal ranges receive proportionally more samples.
- **Range-balanced replay** should improve macro-average and worst-range performance because every currently represented temporal range receives equal expected replay mass regardless of its size.

### H4: Soft loss-derived targets reduce costly mistakes

A soft target derived from each node's excess loss should reduce routing regret relative to hard winner labels, even when hard oracle-match accuracy is similar.

### H5: Fixed historical replay removes the extra logarithmic factor

Let `K_t` be the number of active LogT nodes at macro-step `t`. Since `K_t = O(log t)`, evaluating a fixed number of new and historical examples under all active nodes costs `O(log t)` per macro-step and `O(T log T)` over the full run.

---

## 3. Scope and non-goals

### In scope

- The original LogT hierarchy and consolidation schedule.
- A separate feed-forward router.
- Exhaustive evaluation of all extant nodes while generating router supervision.
- Fixed-budget historical replay.
- Example-balanced and range-balanced replay.
- Hard winner targets and soft regret-derived targets.
- MNIST only.

### Explicitly out of scope

- Persistent feature or loss caches.
- Reservoir sampling or bounded historical storage.
- Hierarchical, pairwise, or shared candidate scorers.
- Router-controlled node creation or consolidation.
- Novelty/rejection classes.
- Mixtures or ensembles of nodes at inference time.
- Router gradients flowing into LogT nodes.
- ImageNet or language-domain optimizations.
- Any modification intended to improve the underlying LogT hierarchy.

For this experiment, keeping all historical router examples in memory is acceptable. The objective is to validate routing, not yet to solve router-memory scaling.

---

## 4. Preserve the existing LogT experiment

Use the current MNIST LogT implementation as the source of truth for:

- node architecture;
- leaf training;
- node loss;
- binary carry/merge schedule;
- consolidation and replay inside LogT;
- active-node semantics;
- existing baselines.

The router must be added as a read-only subsystem. With the router disabled, the existing experiment must reproduce its previous results.

The active nodes at a macro-step represent **disjoint temporal ranges**. The router selects directly among those current nodes. It does not traverse parent-child relationships.

If the repository does not yet contain a stable MNIST stream, use the fallback benchmark in Section 5.

---

## 5. Fallback MNIST benchmark

### 5.1 Domains

Use eight fixed Permuted-MNIST domains:

- Domain 0: identity pixel ordering.
- Domains 1 through 7: fixed random permutations of the 784 input pixels.
- Use deterministic permutation seeds `1001` through `1007`.
- Labels remain the original MNIST digit labels.

This produces input-identifiable regimes without introducing the impossible case in which the same observable input requires incompatible labels and no context distinguishes them.

### 5.2 Stream schedule

- Total macro-steps: `T_max = 64`.
- Each block of eight macro-steps contains every domain exactly once.
- Shuffle the domain order independently within each block using stream seed `20260827`.
- Do not provide the domain ID, task ID, macro-step number, or range endpoints to the router.

The checkpoints `t = 7, 15, 31, 63` are especially important because the binary LogT stack has many simultaneously active levels immediately before a power-of-two carry.

### 5.3 Per-step data

For each macro-step, draw disjoint samples from the current domain:

- `B_model`: 256 examples used only for the LogT node update.
- `B_router_new`: 256 examples used for current-step router supervision and later router replay.
- `B_eval`: 128 examples never used to train either the LogT hierarchy or the router; retain them for temporal-range evaluation.

Maintain separate deterministic shuffled index streams for each domain. Do not reuse an example within the same domain until that domain's pool is exhausted.

Use the standard MNIST test split, transformed separately by each domain permutation, for full domain-level evaluation.

### 5.4 Why the router batch is separate

Do not generate the current router target from the exact examples just used to update the newest node. A freshly trained leaf can obtain artificially optimistic in-sample loss and create a trivial or misleading routing target. `B_router_new` is a same-domain held-out calibration batch that measures which current node generalizes best after the LogT update.

---

## 6. Candidate-node slots

Let `L_max` be the number of possible LogT levels over the fixed horizon:

```text
L_max = ceil(log2(T_max)) + 1
```

For `T_max = 64`, use seven slots corresponding to levels 0 through 6.

At macro-step `t`:

- each occupied level contains at most one active node;
- output class `j` means "select the node currently occupying LogT level `j`";
- inactive levels are masked;
- the router may never select an inactive level.

Do not allocate one output class per historical node. Historical node identities disappear during consolidation; level slots remain well-defined.

Use exactly the same candidate set as the existing exhaustive-addressing baseline. If that baseline includes the frozen base model as an additional candidate, preserve that behavior with one extra fixed slot. Otherwise, do not add the base model only for this experiment.

---

## 7. Router input

For every example `x`, run every active node far enough to obtain:

- `h_j(x)`: the penultimate hidden representation produced by the node at level `j`;
- `z_j(x)`: the ten output logits produced by that node;
- `m_j`: an active-slot indicator, equal to 1 for an occupied level and 0 otherwise.

The router feature for level `j` is:

```text
slot_j(x) = concat(
    layer_norm(h_j(x)),
    log_softmax(z_j(x)),
    m_j
)
```

For an inactive level, use an all-zero hidden/output vector and `m_j = 0`.

Concatenate the slots in ascending level order:

```text
router_input(x) = concat(slot_0(x), ..., slot_(L_max-1)(x))
```

Requirements:

- Detach all node-produced features before passing them to the router.
- The router input must depend only on `x` and the current nodes.
- The label `y`, per-node supervised losses, domain ID, and time-step metadata must never enter the router input.
- The same feature extractor must be used by every router condition.

### Router architecture

Use a deliberately over-capacity MLP for the first test:

```text
input
  -> Linear(1024) -> GELU -> LayerNorm -> Dropout(0.1)
  -> Linear(512)  -> GELU -> LayerNorm -> Dropout(0.1)
  -> Linear(256)  -> GELU
  -> Linear(L_max)
```

If the flattened input dimension is smaller or larger than 1024, only the first layer's input width changes.

Apply the active-level mask to the output logits before the final softmax.

Use one independent router and optimizer state per experimental condition.

---

## 8. Oracle supervision

For an example pair `(x, y)`, let `ell_j(x, y)` be the ordinary per-example classification cross-entropy of active node `j` against label `y`.

### 8.1 Hard winner target

Define the hard target as the active node with minimum loss:

```text
winner(x, y) = argmin_j ell_j(x, y)
```

Use ordinary cross-entropy between the router distribution and this class label.

For an exact numerical tie, choose the lower LogT level index to keep the result deterministic. Exact ties should be rare.

### 8.2 Soft regret-derived target

For each active node, define its excess loss relative to the best active node:

```text
regret_j(x, y) = ell_j(x, y) - min_u ell_u(x, y)
```

Convert these regrets into a soft teacher distribution:

```text
q_j(x, y) = softmax(-regret_j(x, y) / tau)
```

Use cross-entropy from `q` to the router distribution. Mask inactive levels before normalization.

Default temperature:

```text
tau = 0.10
```

Do not tune `tau` separately for example-balanced and range-balanced replay. A short one-seed pilot may compare `0.05`, `0.10`, and `0.25`; then freeze one value for the full experiment.

### 8.3 Target recomputation

Historical examples store only:

```text
(x, y, originating_macro_step)
```

Do not store permanent node labels. At every macro-step, recompute hidden features, all current node losses, and the current oracle target under the **post-update, post-consolidation active node set**.

This prevents stale labels after nodes are merged or deleted.

---

## 9. Fixed-budget historical replay

Set the historical router replay budget to:

```text
H = 256 examples per macro-step per replay condition
```

`H` must remain fixed for all macro-steps. It must not grow with the number of active nodes or elapsed time.

The historical archive contains all prior `B_router_new` examples, grouped by originating macro-step. Do not include the current macro-step in historical replay.

### 9.1 Example-balanced replay

Sample `H` examples uniformly from the union of all prior archived router examples.

Because every macro-step contributes the same number of archived examples, an active temporal range receives expected replay mass proportional to the number of macro-steps it contains.

This condition estimates historical risk under a uniform distribution over examples.

### 9.2 Range-balanced replay

Let the current active nodes define disjoint temporal ranges. Intersect each range with macro-steps strictly earlier than the current step and discard any empty historical intersection.

Allocate the fixed budget as evenly as possible across those nonempty ranges:

```text
base_quota = floor(H / number_of_nonempty_ranges)
remainder  = H mod number_of_nonempty_ranges
```

Randomly permute the ranges using the condition's RNG. Give `base_quota + 1` samples to the first `remainder` ranges and `base_quota` samples to the rest. Within each selected range, sample uniformly from all archived examples originating in that range.

If the number of nonempty ranges ever exceeds `H`, choose `H` ranges without replacement and rotate coverage deterministically across macro-steps. This will not occur in the 64-step MNIST run but keeps the sampler well-defined.

This condition estimates historical risk under a uniform distribution over currently represented temporal ranges.

### 9.3 Sampling replacement

Sample without replacement whenever a range or the global archive contains enough examples. Otherwise sample the remaining examples with replacement. Log the number of duplicate replay draws; it should be zero or negligible in the full run.

### 9.4 Current-versus-historical weighting

Train with equal aggregate weight on current and historical router supervision:

```text
router_loss = 0.5 * mean(loss_on_B_router_new)
            + 0.5 * mean(loss_on_historical_replay)
```

When no history exists at the first macro-step, use only the current loss.

Keep this weighting identical in every replay condition.

---

## 10. Online training order

At each macro-step `t`, execute the following order exactly:

1. Draw `B_model`, `B_router_new`, and `B_eval` from the current domain.
2. Update the LogT leaf using `B_model`.
3. Perform every LogT carry, merge, repair, or consolidation triggered by the existing algorithm.
4. Freeze/detach the resulting active nodes for router training.
5. Enumerate the post-update active levels and their temporal ranges.
6. Run all active nodes on `B_router_new` and compute router features and oracle losses.
7. For each router condition:
   - sample exactly `H` historical examples according to that condition;
   - rerun all active nodes on those examples;
   - recompute current oracle losses and targets;
   - update that router using the current and historical supervision.
8. Add `B_router_new` to the historical router archive, tagged with macro-step `t`.
9. Add `B_eval` to the untouched evaluation archive, tagged with macro-step `t`.
10. Evaluate and log the metrics scheduled for this macro-step.

The router must never affect steps 2 or 3.

### Router optimizer

Default configuration:

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
gradient_clip_norm: 1.0
router_minibatch_size: 128
router_epochs_per_macro_step: 4
```

The number of router epochs must remain fixed over time.

---

## 11. Experimental conditions

Run all router conditions against the same LogT node trajectory within each seed.

### Baselines

1. **Oracle**: choose the active node with minimum per-example supervised loss. This is an offline upper bound and is not deployable.
2. **Most-recent-range node**: choose the active node whose temporal range ends at the current macro-step.
3. **Largest-range node**: choose the active node at the highest occupied LogT level.
4. **Uniform active node**: choose an active node uniformly at random.
5. **No-replay hard router**: train the same router only on `B_router_new` using hard targets.

### Main 2 x 2 comparison

| Condition | Historical sampling | Target |
|---|---|---|
| `example_hard` | Example-balanced | Hard winner CE |
| `range_hard` | Range-balanced | Hard winner CE |
| `example_soft` | Example-balanced | Soft regret-derived CE |
| `range_soft` | Range-balanced | Soft regret-derived CE |

Train these four routers simultaneously but independently. They may share newly computed node features for `B_router_new`, but they must have separate parameters, optimizers, replay draws, and RNG streams.

### Secondary feature ablation

Run only after the main experiment works:

- all-node hidden states plus log probabilities;
- all-node log probabilities only;
- one reference-node hidden state only.

Use the best-performing replay/target condition for this ablation. Do not add it to the initial full factorial matrix.

---

## 12. Evaluation protocol

### 12.1 Per-step lightweight evaluation

After every macro-step, evaluate on:

- the current step's untouched `B_eval`;
- all prior `B_eval` batches, grouped by their current active temporal range;
- a fixed 1,000-example subset of the test set for every domain observed so far.

### 12.2 Full evaluation checkpoints

At macro-steps:

```text
7, 15, 31, 63, 64
```

run full evaluation on all 10,000 MNIST test examples for every domain observed so far.

The checkpoints at 7, 15, 31, and 63 are primary because many LogT levels are active. Step 64 is a sanity check: after a complete power-of-two carry there may be only one candidate, so every valid router must match it.

### 12.3 Inference rule

At test time:

1. run all active nodes to obtain their hidden/output features;
2. run the router;
3. mask inactive levels;
4. choose the active level with maximum router probability;
5. use the already-computed prediction from that selected node.

Do not use labels to route at inference time. Labels are used only to compute offline oracle metrics.

---

## 13. Metrics

### 13.1 Primary routing metrics

For each evaluated example, let `selected_loss` be the loss of the router-selected node and `oracle_loss` be the minimum loss among active nodes.

Report:

- **Mean routing regret**: `mean(selected_loss - oracle_loss)`.
- Median and 90th-percentile routing regret.
- Oracle-match rate: fraction of examples for which the router selects the exact hard winner.
- Near-oracle rate at regret thresholds `0.01`, `0.05`, and `0.10`.
- Selected-node classification accuracy.
- Selected-node mean cross-entropy.
- Accuracy gap and cross-entropy gap relative to the oracle.

Mean routing regret is the primary metric. Exact winner accuracy is secondary because two nodes can have nearly identical losses.

### 13.2 Replay-retention metrics

Report all primary metrics as:

- micro-average over historical examples;
- macro-average over current temporal ranges;
- worst current temporal range;
- current range versus all older ranges;
- function of age since the originating macro-step.

These views are required to expose the expected example-balanced versus range-balanced tradeoff.

### 13.3 Router diagnostics

Log:

- hard oracle target frequency by active level;
- router selection frequency by active level;
- target-versus-selection confusion matrix;
- mean router entropy;
- inactive-level selection attempts before masking;
- active node count `K_t`;
- temporal range sizes;
- mean and distribution of the gap between best and second-best node losses.

The last quantity distinguishes genuinely difficult routing examples from arbitrary hard-label disagreements between nearly tied nodes.

### 13.4 Hierarchy-versus-routing decomposition

Keep separate:

- **Routing gap**: selected-node loss minus best-extant-node loss.
- **Hierarchy gap**: best-extant-node performance versus the existing full-replay or joint-training reference.

The router can only reduce the routing gap. It cannot recover a competence already destroyed by consolidation.

### 13.5 Compute metrics

Count training-time node/example forward evaluations used for router supervision:

```text
router_node_evals_t = K_t * (B_router_new_size + H)
```

Count this separately for each router condition and separately from evaluation-only oracle computation.

Also log:

- cumulative router node/example evaluations;
- router optimizer steps;
- router wall-clock time;
- LogT wall-clock time;
- peak GPU memory.

Plot cumulative router node/example evaluations against:

```text
sum_{s=1..t} K_s
```

and against `t log2(t)`. The empirical curve should be consistent with `O(T log T)` rather than `O(T log^2 T)`.

---

## 14. Complexity argument to verify in code

Define:

- `T`: number of macro-steps;
- `K_t`: number of active LogT nodes at step `t`;
- `B`: fixed number of current router examples;
- `H`: fixed number of historical router examples.

The LogT stack guarantees:

```text
K_t <= 1 + floor(log2(t))
```

The router generates supervision by evaluating every active node on `B + H` examples:

```text
cost_t proportional to (B + H) * K_t
```

Because `B` and `H` are constants:

```text
cost_t = O(log t)
```

Therefore:

```text
sum_{t=1..T} cost_t = O(T log T)
```

The four main router conditions add a fixed multiplicative constant, not another asymptotic factor.

Add a runtime assertion that every replay condition receives exactly `H` historical samples whenever at least one historical example exists.

---

## 15. Required implementation checks

### Data and sampling

- [ ] Active temporal ranges are disjoint and cover the represented macro-steps.
- [ ] Current-step examples are not included in historical replay.
- [ ] Example-balanced sampling frequency is proportional to range size in a Monte Carlo unit test.
- [ ] Range-balanced sampling frequency is approximately uniform across active ranges in a Monte Carlo unit test.
- [ ] Every replay condition uses exactly `H` historical draws per eligible step.
- [ ] Sampling is reproducible from explicit seeds.

### Router correctness

- [ ] Router features contain no label, domain ID, time index, or stored oracle loss.
- [ ] Inactive input slots are zero-filled and marked inactive.
- [ ] Inactive output classes are masked before softmax/argmax.
- [ ] Node features are detached.
- [ ] Router backpropagation produces no node-parameter gradients.
- [ ] Historical oracle targets are recomputed after every change to the active node set.
- [ ] All routers see the identical underlying LogT hierarchy within a seed.

### Regression

- [ ] Disabling the router reproduces the existing LogT experiment.
- [ ] At a step with exactly one active node, every router selects it.
- [ ] Oracle loss is never greater than the loss of any fixed-node baseline on the same example.
- [ ] Reported routing regret is never negative beyond numerical tolerance.

---

## 16. Run plan

### Phase A: smoke test

Purpose: validate integration and labels before a full run.

Configuration:

```text
macro_steps: 15
seeds: [0]
historical_budget: 64
router_epochs_per_step: 2
conditions:
  - no_replay_hard
  - example_hard
  - range_hard
```

Required checks:

- Router loss decreases.
- Historical targets change after merges without crashing.
- No inactive level is selected.
- Routing regret is finite and nonnegative.
- Fixed replay count and forward-evaluation accounting are correct.

### Phase B: full primary experiment

Configuration:

```text
macro_steps: 64
seeds: [0, 1, 2, 3, 4]
historical_budget: 256
router_new_batch_size: 256
router_epochs_per_step: 4
conditions:
  - no_replay_hard
  - example_hard
  - range_hard
  - example_soft
  - range_soft
```

Run all router conditions on the same hierarchy trajectory for each seed.

### Phase C: optional follow-up

Only run after Phase B demonstrates a meaningful gap between learned routing and fixed-node baselines.

1. Feature ablation.
2. Historical-budget sensitivity at `H = 64, 128, 256, 512`.
3. Router-capacity sensitivity.
4. Reuse of the exact node-training batch versus the held-out router batch, to quantify teacher-label leakage.

Do not begin persistent caching, reservoirs, or scalable language-oriented router work in this experiment.

---

## 17. Success criteria

The experiment is a successful proof of concept if all of the following hold at the high-active-node checkpoints `15`, `31`, and `63`:

1. At least one replay-trained router has substantially lower mean routing regret than the most-recent-range baseline and the no-replay router.
2. The router closes at least 75% of the cross-entropy gap between the most-recent-range baseline and the exhaustive oracle, averaged across seeds.
3. Replay improves old-range regret without materially degrading current-range accuracy.
4. Example-balanced replay has lower or equal historical micro-average regret than range-balanced replay, or the result clearly falsifies that hypothesis.
5. Range-balanced replay has lower or equal macro-average or worst-range regret than example-balanced replay, or the result clearly falsifies that hypothesis.
6. The cumulative router-supervision forward count follows the fixed-budget `O(T log T)` accounting.
7. The hierarchy-versus-routing decomposition shows that remaining errors can be attributed rather than conflated.

A negative result is still informative if the exhaustive oracle is strong but every router remains far from it. That would show that the current all-node feature representation or MLP is inadequate. If the exhaustive oracle itself is weak, the failure is in the extant-node set or consolidation, not the router.

---

## 18. Pseudocode

```python
for t, domain in enumerate(stream, start=1):
    B_model, B_router_new, B_eval = draw_disjoint_batches(domain)

    # Existing LogT behavior; router must not affect this.
    logt.update_leaf(B_model)
    logt.run_required_consolidations()

    active_nodes = logt.active_nodes_sorted_by_level()
    active_mask = build_active_level_mask(active_nodes, L_max)

    # Shared current-step supervision.
    new_features, new_node_losses = evaluate_all_nodes(
        active_nodes,
        B_router_new,
        detach=True,
    )
    new_hard_targets = argmin_active(new_node_losses)
    new_soft_targets = regret_soft_targets(new_node_losses, tau=0.10)

    for condition in router_conditions:
        if historical_archive:
            old_batch = condition.sampler.sample(
                archive=historical_archive,
                active_ranges=logt.active_ranges(),
                count=H,
                current_step=t,
            )

            old_features, old_node_losses = evaluate_all_nodes(
                active_nodes,
                old_batch,
                detach=True,
            )
            old_hard_targets = argmin_active(old_node_losses)
            old_soft_targets = regret_soft_targets(
                old_node_losses,
                tau=0.10,
            )
        else:
            old_batch = None

        condition.router.train_for_fixed_epochs(
            new_features=new_features,
            new_targets=select_target_type(
                condition.target_type,
                new_hard_targets,
                new_soft_targets,
            ),
            old_features=old_features if old_batch is not None else None,
            old_targets=select_target_type(
                condition.target_type,
                old_hard_targets,
                old_soft_targets,
            ) if old_batch is not None else None,
            active_mask=active_mask,
            current_weight=0.5,
            historical_weight=0.5,
        )

    historical_archive.add(B_router_new, macro_step=t)
    evaluation_archive.add(B_eval, macro_step=t)

    evaluate_and_log(
        t=t,
        active_nodes=active_nodes,
        routers=router_conditions,
        evaluation_archive=evaluation_archive,
        transformed_domain_tests=domain_test_sets,
    )
```

---

## 19. Deliverables

Produce:

1. Router module with masked level-slot outputs.
2. Example-balanced and range-balanced replay samplers.
3. Integration into the current LogT MNIST training loop.
4. Unit tests for sampling, masking, target recomputation, and gradient isolation.
5. A single configuration file reproducing Phase B.
6. Per-step metrics in machine-readable JSONL or Parquet.
7. Seed-level summary table in CSV.
8. Plots:
   - selected-node accuracy over time;
   - mean routing regret over time;
   - oracle-match rate over time;
   - micro-average versus worst-range regret;
   - example-balanced versus range-balanced tradeoff;
   - target/selection confusion by LogT level;
   - best-versus-second-best loss margin distribution;
   - cumulative router node evaluations versus `T log T`;
   - routing gap versus hierarchy gap.
9. `RESULTS.md` containing:
   - exact configuration;
   - pass/fail status for every success criterion;
   - concise interpretation of example-balanced versus range-balanced behavior;
   - any implementation ambiguities or unexpected failure modes surfaced by the run.

Do not report only router classification accuracy. The primary scientific output is routing regret and downstream selected-node performance relative to the exhaustive oracle.
