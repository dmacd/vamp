# LogT prediction integration on Permuted-MNIST

## Frozen question

This protocol repeats the complete direct-prediction experiment and its
converged full-replay reference on the existing eight-domain Permuted-MNIST
benchmark. It asks whether a residual multilayer perceptron can integrate all
live frozen LogT-node behaviors more effectively than selecting one node, and
whether fixed 256-example logarithmic replay can approach a fully replayed
model trained to convergence.

The parent behavioral-router run is immutable:

- run identity `4b1ed9cf715aa42a951dd71fe2242382ef5f4319d4b10cf0b6e3a4633f7e0b69`;
- protocol SHA-256
  `022dd22f5f225555187f00fd7dd3c7e24fbc097a458e7fb67c3d4f48c3054ab6`;
- summary SHA-256
  `317ccaa6c23ede41d11b53cd40f46c0971700eed52b89d75f04ede3125c9ecc5`;
- frozen CNN checkpoint SHA-256
  `45793341113b7a44b397d8781b0590f7dcc54ca05ca2cd7d637b11244033a282`.

Neither successor may change the hierarchy, parent router, source data, or
joint-IID checkpoint references.

## Task and allocation

Use the identity pixel ordering plus fixed permutation seeds 1001 through
1007. The original stream seed `20260827` independently shuffles all eight
domains within each eight-step block. Labels remain ordinary MNIST digit
labels. Domain identity, permutation identity, macro-step, and temporal-range
endpoints never enter the integrator.

At each of 64 macro-steps allocate disjoint batches of 256 adapter examples,
256 integrator examples, and 128 untouched temporal-evaluation examples. Use
the same five primary seeds 0 through 4, fixed 1,000-example test subset per
observed domain, and full checkpoints 7, 15, 31, 63, and 64. Node adapters use
the parent's de-novo 20-epoch AdamW training and binary-counter carry schedule.

## Integrator and online conditions

Each of seven persistent level positions holds a detached 128-value normalized
node hidden state, ten log probabilities, and an active bit. Inactive positions
are exactly zero, giving a 973-value input. The model is:

```text
973 -> 1024 -> 512 -> 256 -> 10 residual logits
```

GELU, LayerNorm, and dropout 0.1 match the Rotated-MNIST integrator. The
residual corrects the equal-probability mean of active-node predictions; its
output layer and never-activated input columns start at zero. With one node,
the untrained model exactly reproduces that node.

Train four independent conditions with AdamW at 0.001, weight decay 0.0001,
gradient clipping 1.0, and minibatches no larger than 128:

1. current examples only;
2. current plus 256 example-balanced historical examples;
3. current plus 256 live-range-balanced historical examples; and
4. the example-balanced schedule using only frozen-base behavior in slot zero.

Primary uses four complete epochs per step. Replay conditions weight current
and historical cross-entropy equally. Historical images are reevaluated
against the current frontier; permanent digit labels are the only targets.
Smoke uses seed zero, 15 steps, a 64-example historical budget, and two epochs.

At every evaluation retain the mean ensemble, most-recent range, largest
range, deterministic uniform-node choice, and label-aware best node. At full
checkpoints train a fresh four-epoch cumulative integrator and load the sealed
parent joint-IID adapter. The four-epoch cumulative model is an optimization
reference, not the ceiling.

## Frozen evaluation criteria

Average seeds at checkpoints 15, 31, and 63. Report cross-entropy and accuracy
for fixed/full tests, the temporal archive, current versus older ranges, and
per-range macro and worst cases. Apply the same seven decisions as the
Rotated-MNIST protocol: replay must beat no replay and the mean ensemble; close
at least 75% of the positive no-replay-to-four-epoch-offline gap; improve older
ranges with at most a 2-point current-range accuracy loss; full-node replay
must beat the base-only control without lower accuracy; the best integrator
must beat the sealed `example_soft` router on both metrics; all accounting and
structural gates must pass; and attribution controls must be present.

No architecture, replay, epoch, weighting, checkpoint, or threshold choice may
change after smoke or primary results become visible.

## Converged full-replay reference

After the online parent is complete, run a separately identified ceiling. At
every step rebuild the exact frontier, recompute features for every cumulative
integrator-training example, and initialize three independent integrators from
scratch. Every epoch presents every training example exactly once. The
cumulative disjoint evaluation allocation controls learning rate, stopping,
best-epoch restoration, and restart selection; it never updates weights and is
not reported as unbiased evaluation.

Start at learning rate 0.001. After five epochs without at least 0.0001-nat
validation improvement, halve it down to 0.00001. Declare convergence only
after ten further non-improving epochs at the minimum learning rate. The
200-epoch bound is a failure cap. Select only converged restarts, preferring
lower validation cross-entropy, then higher validation accuracy, then lower
restart index. Test scores never affect selection.

The ceiling is certified only if at least one restart converges at every step,
all example-presentation and feature-work counts are exact, test/validation
isolation holds, and the reconstructed mean ensemble matches the online parent
at every authenticated coordinate. Compare it with both the parent's best
online replay condition and four-epoch cumulative reference at checkpoints
15, 31, and 63. Overlay its five-seed every-step trace on the online accuracy
and cross-entropy plots.

## Artifact and execution boundary

The online protocol uses config
`configs/vamp_logt_integrator_permuted_mnist/primary.yaml`, CLI
`apm.experiments.vamp_logt_integrator_permuted_mnist`, and artifact root
`artifacts/vamp-logt-integrator-permuted-mnist/`. The ceiling uses the matched
`ceiling_permuted` config/CLI modules and
`artifacts/vamp-logt-integrator-ceiling-permuted-mnist/`.

Both protocols bind their resolved configuration, this document, all material
source files, authenticated parents, raw MNIST hashes, PyTorch version,
chained ledgers, checkpoints, exact optimizer state, and reports. Execute seeds
serially after host/GPU memory checks. Checkpoint before retiring inactive node
or restart state, and verify a completed rerun leaves sealed ledgers unchanged.
