# Dense LogT integrator scaling on 100 Permuted-MNIST domains

## Question

Measure how condition-specific training work grows when the dense LogT system
learns 100 pixel permutations. Compare the bounded persistent integrator with a
fresh full-replay integrator without spending the compute required to fit the
full-replay comparator at every step.

This is a successor experiment. It does not alter the completed eight-domain
Permuted-MNIST run or its three-seed report.

## Frozen data and model protocol

- Use one run seed, seed 0.
- Use 100 domains: the identity mapping and pixel-permutation seeds 1001 through
  1099. The fixed stream-order seed is 20260827. Each domain occurs exactly
  once, so a macro-step is also the number of learned permutations.
- Allocate 256 disjoint model-training examples, 256 integrator-observer
  examples, and 128 unused held-out stream examples per domain. Accuracy is
  measured on a deterministic 256-example test subset per learned domain with
  equal domain weight.
- Authenticate and import the completed successor run's seed-zero
  1024/1024/512 identity-trained dense base. Do not recalibrate or select a
  model using this experiment's results.
- Build the same binary-counter LogT hierarchy. Every leaf and carry node is a
  de-novo full dense MLP fit from the shared base for 20 epochs. Frozen node
  features use seven stable level slots, sufficient through step 100.

The resolved choices are in
`configs/vamp_logt_mlp_permuted_100_scaling.yaml`. The run authenticates the
config, this document, material source files, raw MNIST data, all 100
permutations, and the imported base before training.

## The only two conditions

`integrator_uniform_replay` is one persistent integrator. At each step it trains
for four epochs on the 256 current observer examples. Starting at step 2 it also
receives exactly 256 examples sampled uniformly from all strictly earlier
observer examples. Current and historical loss each receive weight 0.5.

`full_replay_integrator_20_epochs` is a fresh integrator fit at learned-domain
counts 1, 2, 4, 8, 10, 16, 26, 41, 66, and 100. It discards all prior
integrator state, recomputes current-frontier features for every observer
example seen so far, and trains for exactly 20 epochs. It has one initialization
per checkpoint and no validation, early stopping, convergence test, or restart
selection. “Ceiling” here means a high-information comparator, not a claim of
mathematical or empirical convergence.

No router, range replay, current-only integrator, base-only integrator, pooled
MLP, or additional seed is run.

## Work accounting

One model example-pass is one example traversing one model. Batch-call counts
are retained separately.

For each condition update, record:

- frozen-node forward example-passes and calls used to construct training
  features;
- optimizing-integrator forward example-passes and calls;
- optimizing-integrator backward example-passes and backward calls;
- state initialization, data preparation, feature construction, optimizer, and
  summed training-only wall time.

The timed total includes only work necessary to produce that condition's
training update. Pre/post diagnostic inference, learned-domain test inference,
reporting, checkpoint I/O, and the shared hierarchy build do not enter it.
Diagnostic and test inference pass counts are nevertheless written under
explicit `excluded_*` names so the exclusion is auditable. Shared hierarchy
forward and backward work is recorded once per step and summarized separately;
it is not charged to both condition curves.

Let `a(k) = popcount(k)` be the number of active binary-counter nodes after
learning permutation `k`. After step 1, uniform replay performs 2,048
integrator forward and backward example-passes per update and `512 a(k)` frozen
node forwards. Its expected condition-specific cost is therefore `O(log k)`
forward and `O(1)` backward per update, or `O(T log T)` forward and `O(T)`
backward cumulatively.

At checkpoint `k`, full replay performs `20 × 256k` integrator forward and
backward example-passes and `256k a(k)` frozen-node forwards. Its expected cost
is `O(k log k)` forward and `O(k)` backward per fit. If fit at every step, the
cumulative cost would be `O(T² log T)` forward and `O(T²)` backward. The actual
experiment fits only the ten declared checkpoints, so its summed runtime must
not be presented as the cost of an every-step ceiling.

## Outputs and interpretation

The resumable runner is:

```bash
.venv-vision/bin/python -m apm.experiments.vamp_logt_mlp_permuted_scaling
```

It writes an authenticated chained metric ledger as each step completes, a
checkpoint for the persistent condition after each step, immutable sampled
full-replay checkpoints, a machine-readable summary and CSV, Markdown and
self-contained HTML reports, a three-panel training time/forward/backward plot,
and an accuracy plot.

Seconds are a single-device observation and may contain warm-up and scheduling
noise. Example-pass counts carry the complexity conclusion. Accuracy is a
single-seed diagnostic of what the two training budgets achieve, not a precise
population estimate.
