# LogT-VAMP prediction integrator on VAMP-AF Rotated-MNIST

## Status and question

This document freezes the first direct-prediction successor to the completed
behavioral-router experiment. The implementation checkpoint before this plan is
Git commit `2f69f3e` (`Add LogT behavioral router experiments`). The completed
router run has protocol identity
`97f5f70a91fa3430e244dc4fd91b67b3c8fd28e5bb1eaa0cb3d7d304e3d32896`,
protocol-file SHA-256
`b088d56ce82908562bfb74b8e6105fd7e26aa0303cbb2da4bfdd4b4dfab3744b`,
and aggregate-summary SHA-256
`b598047f7cb044978defb872764e55e6e76022fe2dcc24534c986a928ba05535`.

The experiment asks whether a mutable network can integrate the behavior of all
live frozen LogT nodes more effectively than a router that must choose exactly
one node. It removes dynamic best-node targets. The learned output is a final
ten-class prediction trained against the example's permanent shifted digit
label.

This remains a LogT experiment on VAMP-AF's five data contexts. It does not load
or modify the sealed spatial VAMP-AF tree.

## Frozen task and hierarchy

Carry forward the exact completed Rotated-MNIST router boundary without
retuning it:

- frozen VAMP-AF CNN run
  `c3ad77df09fde94a75e2464450c21486d632bf4f60afe44c9602c6a86acf61af`;
- frozen CNN checkpoint SHA-256
  `45793341113b7a44b397d8781b0590f7dcc54ca05ca2cd7d637b11244033a282`;
- rotations 0, 18, 36, 54, and 72 degrees, bilinear interpolation, no
  expansion, and zero fill;
- label shifts 0, 2, 4, 6, and 8 modulo ten;
- balanced VAMP-AF source-identity SHA-256
  `37179656c4e9b9ba6e8ff82b77941196cb6a87bbc5f02160f319f19254e5d908`;
- blocked primary context schedule 13, 13, 13, 13, and 12 steps;
- disjoint per-step allocations of 256 adapter examples, 256 integrator
  examples, and 128 temporal-evaluation examples;
- de-novo full-rank top-two adapters trained for 20 epochs at every leaf and
  carry; and
- five primary seeds 0 through 4 and the same test subsets and full checkpoints
  7, 15, 31, 63, and 64.

The live frontier remains a standard binary counter. It contains at most one
node per level and at most seven nodes over the 64-step horizon. Carries retire
children only after the replacement parent and enclosing checkpoint are
durable. The integrator cannot change node training, topology, or retirement.

## Fixed integrator input

For every active level, compute the same detached behavior slot used by the
router:

```text
[layer-normalized 128-value node hidden state,
 node 10-way log probabilities,
 active bit]
```

Preallocate seven stable level positions. Each slot has 139 values, giving a
fixed 973-value input. An inactive slot is exactly zero. Labels, context IDs,
rotation angles, label shifts, macro-step indices, temporal endpoints, and
stored targets cannot enter this tensor.

Level positions describe LogT capacity, not persistent node identity. A level
may deactivate during a carry and later contain a different interval. Learned
input columns remain assigned to that level. Columns for levels that have never
been active start at zero, so opening a future level does not inject an
arbitrary residual correction. No runtime network expansion or optimizer-state
surgery is permitted.

## Prediction model

Use the router's existing healthy-size body:

```text
973 -> 1024 -> GELU -> LayerNorm -> dropout(0.1)
    -> 512  -> GELU -> LayerNorm -> dropout(0.1)
    -> 256  -> GELU -> 10 residual class logits
```

The direct output is a residual over the equal-probability ensemble of active
nodes. If `p_l` is an active node's ten-way probability vector, define:

```text
baseline_log_probability = log(mean_l p_l)
final_logits = baseline_log_probability + residual_mlp(all_slots)
```

Initialize the final residual layer to zero. Before training, the model is
therefore the parameter-free mean ensemble. With one active node, it has exact
prediction parity with that node. This protects first activation and complete
carry boundaries from an arbitrary randomly initialized class prediction while
leaving the MLP free to learn nonlinear corrections or implicit soft gating.

Train only the integrator. Node outputs remain detached and node parameters
remain frozen.

## Training and replay

Use AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping at 1.0,
and maximum minibatch size 128. Use two epochs per smoke step and four epochs
per primary step. Preserve one presentation of every current example per epoch.

The learned primary conditions are:

1. `integrator_no_replay`: current examples only;
2. `integrator_example_replay`: 256 examples sampled uniformly from the
   strictly historical integrator archive;
3. `integrator_range_replay`: 256 examples divided equally over nonempty live
   historical LogT ranges; and
4. `base_example_replay`: the same example-balanced replay and MLP capacity,
   but only the frozen base model's hidden state and log probabilities occupy
   slot zero. This control tests whether the live node behaviors add information
   beyond a continually replay-trained central classifier.

For replay conditions, average current-source and historical-source
cross-entropy equally. Historical images are rerun through the current frontier
because node features change after carries, but their shifted digit labels do
not change. Replay sampling affects the query distribution only; there is no
node oracle target and no hard/soft target family.

Smoke uses seed zero, five blocked context steps, a 64-example historical
budget, and all four conditions. Primary uses seeds 0 through 4, 64 steps, and
the fixed 256-example historical budget.

## References and controls

At every evaluation, report these nonlearned node-combination controls:

- equal-probability mean ensemble;
- most-recent active range;
- largest active range;
- deterministic uniform active-node selection; and
- label-aware best single node.

The best-single-node diagnostic is not an upper bound for the integrator. A
combined predictor may correct every individual node and legitimately
outperform it.

At full checkpoints, retain the matched joint-IID top-two adapter from the
router protocol. Also train a fresh `offline_cumulative_integrator` on every
integrator-training example observed through that checkpoint, with features
recomputed against the checkpoint frontier. It uses the same architecture,
four complete epochs, a checkpoint-specific fixed seed, and no evaluation
examples. It is an optimization/replay reference rather than a deployable
continual condition.

Authenticate and quote the completed `example_soft` router's paired high-
checkpoint result, but do not retrain or modify that sealed run.

## Evaluation

The primary learned-prediction metrics are mean ten-class cross-entropy and
classification accuracy. Report both for:

- the historical evaluation archive as a micro-average;
- macro-average and worst live temporal range;
- current range and all older ranges;
- age since presentation;
- fixed transformed-test subsets; and
- complete observed-context test sets at checkpoints 7, 15, 31, 63, and 64.

Report per-step current and historical training loss separately, in addition
to their combined objective. At every carry, report pre-update baseline
accuracy, post-update integrator accuracy, and the change across the carry.
Keep the original per-node oracle and matched joint-IID measurements so node
competence remains distinguishable from integration quality.

The three-way interpretation is fixed:

- online replay near the offline cumulative integrator means fixed-budget
  replay successfully integrated the live frozen nodes;
- a strong offline integrator with weak online replay means the limitation is
  sequential integration or replay optimization; and
- weak online and offline integrators mean the combined node representation is
  insufficient at the tested capacity.

If the full-node integrator does not beat `base_example_replay`, the run cannot
claim that frozen node-specific behavior supplied useful additional
information.

## Preregistered success criteria

Average the five seeds at high-active-node checkpoints 15, 31, and 63. The
experiment supports its main hypothesis only if all of the following hold:

1. At least one replay integrator has lower mean cross-entropy than both
   `integrator_no_replay` and the parameter-free mean ensemble.
2. The best replay integrator closes at least 75% of the positive
   cross-entropy gap from `integrator_no_replay` to
   `offline_cumulative_integrator`. If the offline reference does not improve
   on no replay, this criterion fails rather than using a nonpositive
   denominator.
3. Replay lowers older-range cross-entropy relative to no replay while reducing
   current-range accuracy by no more than 2.0 percentage points.
4. The best full-node replay integrator has lower cross-entropy than
   `base_example_replay` and does not have lower accuracy.
5. The best replay integrator has lower cross-entropy and higher accuracy than
   the sealed Rotated-MNIST `example_soft` router at the same checkpoints.
6. Every replay update uses its exact historical budget, inactive inputs remain
   zero, all outputs are finite, one-node zero-residual parity is exact, and
   cumulative node-feature work follows fixed-budget `O(T log T)` accounting.
7. The offline, base-only, joint-IID, and best-single-node controls make the
   remaining error attributable rather than conflated.

Do not tune architecture, replay weighting, residual initialization, epochs,
historical budget, checkpoints, conditions, or thresholds after smoke or
primary results are visible. Any loss-weight, logits-only, hidden-only,
capacity, or carry-distillation study is a separately versioned successor.

## Smoke and execution gates

Before real smoke, unit tests must establish:

- stable 973-value slot ordering and exact inactive zeros;
- label isolation of integrator features;
- zero-initialized future columns and residual output;
- exact mean-ensemble and one-node parity;
- frozen node tensors during integrator optimization;
- direct-label targets unchanged across a synthetic carry;
- exact example/range replay budgets;
- independent condition parameters and random streams;
- deterministic checkpoint/resume with chained ledgers; and
- correct current/older/range metric aggregation.

The real smoke may open primary only if every metric is finite, all replay
budgets are exact, residual loss decreases in a majority of eligible updates,
inactive slots remain zero, node tensors remain unchanged, and one-node parity
is exact.

Before primary, check host and GPU memory and ensure no other heavy job is
running. Execute all five seeds serially. A completed rerun must restore every
seed without changing any metric-ledger SHA-256.

## Artifact boundary

Use revision `integrated-prediction-v1`, strict config
`configs/vamp_logt_integrator_rotated_mnist/primary.yaml`, CLI module
`apm.experiments.vamp_logt_integrator_rotated_mnist`, and content-addressed root
`artifacts/vamp-logt-integrator-rotated-mnist/`.

Bind the resolved config, this plan, the VAMP-AF parent, the completed router
protocol and summary, raw IDX files, implementation checkpoint, installed
PyTorch version, and every material source file. Each seed must use a chained
JSONL ledger, commit-before-retirement checkpointing, exact optimizer and RNG
resume, machine-readable summaries, CSV, Markdown, standalone HTML, and plots.

The completed router artifacts and their source files remain unchanged.
