# Dense-base LogT routing and integration on Permuted-MNIST

## Frozen question

The completed Permuted-MNIST router/integrator experiments used a frozen
convolutional base trained only on ordinary MNIST. Arbitrary global pixel
permutations destroy the locality that this base assumes, so poor later-domain
node behavior can be caused by the representation rather than the LogT memory,
replay, router, or integrator.

This successor removes convolution completely. It asks whether a sufficiently
large raw-pixel multilayer perceptron can support competent frozen temporal
nodes across all eight permutations, and whether logarithmic replay can then
integrate their behaviors stably. The prior CNN runs are narrative context,
not paired conditions: their base, nodes, routers, integrators, checkpoints,
and artifacts remain sealed and unchanged.

## Dense architecture calibration

“Three-layer MLP” means three nonlinear hidden affine layers followed by a
separate ten-class affine output. Every hidden block is `Linear -> ReLU ->
Dropout(0.2)`. There is no convolution, augmentation, ensemble, or
normalization layer inside the classifier. Inputs are flattened 28-by-28
pixels in `[0, 1]`.

The candidate widths are, in increasing parameter count:

| Hidden widths | Affine parameters | Seven-slot observer input |
|---|---:|---:|
| 1024, 1024, 512 | 2,383,370 | 3,661 |
| 1536, 1536, 768 | 4,754,698 | 5,453 |
| 2048, 2048, 1024 | 7,912,458 | 7,245 |

MNIST's 60,000 training identities receive one fixed, deterministic,
class-stratified 50,000/10,000 calibration split. The pooled training and
validation populations apply all eight pixel orderings to their respective
source identities, so no underlying image crosses that boundary. Calibration
seeds 0, 1, and 2 are run for every width. Each seed first trains on identity
MNIST, then initializes an otherwise independent task-ID-free pooled fit from
that identity checkpoint. Pooled weights are architecture-selection evidence
only and never enter the main experiment.

All calibration layers train with AdamW, learning rate 0.001, weight decay
0.0001, batches of 256, and gradient clipping at 1.0. After at least 20 epochs,
the rate halves after five epochs without a 0.0001-nat validation improvement,
down to 0.00001. Training stops after ten further non-improving epochs at that
minimum or at 100 epochs, and restores the lowest-validation-cross-entropy
checkpoint.

A width is eligible only when its mean identity validation accuracy is at
least 99.0%, its seed-zero identity validation accuracy is at least 99.0%, and
its mean pooled validation accuracy is no more than 0.25 percentage points
below the widest candidate. The smallest eligible width is selected. If none
is eligible, the protocol stops without retuning. Only after selection are
identity and pooled test metrics opened. The selected seed-zero identity
checkpoint becomes the one shared frozen base.

## Stream and immutable hierarchy tape

The task is exactly the earlier Permuted-MNIST stream: identity ordering plus
permutation seeds 1001 through 1007, 64 macro-steps, stream seed `20260827`,
and experiment seeds 0 through 4. Each step allocates disjoint batches of 256
node-training examples, 256 observer-training examples, and 128 temporal
validation/evaluation examples. Fixed test subsets contain 1,000 examples per
seen domain. Full test checkpoints are 7, 15, 31, 63, and 64; headline
decisions average checkpoints 15, 31, and 63.

The temporal graph is a standard binary counter with seven stable level
positions. A leaf trains on its 256-example interval. Every carry trains a new
parent from the shared frozen base on the exact union; it never inherits child
weights. Each fit clones the full effective base and trains all four affine
weights and biases for 20 epochs with AdamW at 0.001, weight decay 0.0001,
batches of 64, clipping at 1.0, and deterministic dropout. AdamW acts on the
effective cloned parameters, including weight decay; only afterward is
`trained - base` stored as the node delta.

All 127 nodes created by a 64-step seed are retained. Every node delta has an
immutable checkpoint and hash, and every macro-step has a frontier manifest.
Router, online integrator, four-epoch reference, pooled single-MLP reference,
and converged ceiling all load the same authenticated hierarchy tape. They do
not rebuild condition-specific nodes.

## Observer and matched router

For every active node and example, dropout is disabled and the observer takes
the final hidden ReLU activation, its ten class log probabilities, and one
active bit. The hidden activation is normalized per example before
concatenation. Seven level slots keep stable positions; every inactive or
future slot is exactly zero. Labels, losses, domain/permutation identity,
macro-step, range endpoints, and task metadata cannot enter observer features.

The matched router retains the earlier 1024/512/256 hidden widths, dropout
0.1, four epochs per step, 256-example historical budget, and soft-target
temperature 0.10. Five persistent conditions are trained independently:

1. current examples only with hard best-node targets;
2. current plus uniformly sampled historical examples with hard targets;
3. current plus live-range-balanced history with hard targets;
4. current plus uniformly sampled history with soft excess-loss targets; and
5. current plus range-balanced history with soft targets.

Historical examples are always reevaluated against the current frontier;
their hard or soft target is recomputed after every carry.

## Prediction integration and references

The residual integrator has hidden widths 1024/512/256 and dropout 0.1. It
adds ten learned residual logits to the log of the equal-probability mean of
active-node predictions. Future input columns and the output layer start at
zero, preserving exact one-node and mean-ensemble parity before training.

Four persistent conditions use AdamW at 0.001, weight decay 0.0001, clipping
at 1.0, batches of at most 128, and four epochs per step:

1. current examples only;
2. current plus 256 uniformly sampled historical examples;
3. current plus 256 live-range-balanced historical examples; and
4. the uniform-history schedule using only frozen-base behavior in slot zero.

Replay gives current and historical sources equal total weight. The old epoch
matrix is intentionally preserved: with production batch sizes, current-only
receives 8 optimizer updates per step and replay receives 16. There is no
optimizer-update-matched control, and every report must disclose this.

At full checkpoints a fresh integrator trains for four epochs on every
cumulative observer example. A separate full dense MLP, initialized from the
shared base and trained on the cumulative node-training split, is the pooled
single-model reference. Fixed controls are the mean ensemble, newest range,
largest range, deterministic uniform active node, and label-aware best active
node.

## Converged full-replay ceiling

At every macro-step, not just the five checkpoints, the ceiling recomputes all
cumulative observer-training and held-out temporal-validation features against
that step's authenticated frontier. Three integrators initialize freshly and
independently. Every epoch presents every training row exactly once. The same
0.001-to-0.00001 validation convergence rule applies, with a 20-epoch minimum
and 200-epoch failure cap.

Only converged restarts are eligible. Selection minimizes restored validation
cross-entropy, then maximizes validation accuracy, with restart index as the
final deterministic tie break. Test
data never selects an epoch, checkpoint, or restart. The selected ceiling is
evaluated on a fixed test subset at every step and on the complete seen-domain
test population at full checkpoints. It is an empirical optimization ceiling
for these frozen features and this integrator family, not a mathematical upper
bound over every possible predictor.

## Frozen decisions and artifacts

The same seven decisions as the earlier prediction-integrator protocol apply:
replay must beat current-only and the mean ensemble; close at least 75% of the
positive current-only-to-four-epoch-reference cross-entropy gap; improve older
ranges while losing at most two current-range accuracy points; beat the
base-only replay control without lower accuracy; beat the matched new router
on both cross-entropy and accuracy; pass every structural/accounting gate; and
include all attribution controls.

The resolved config, raw IDX hashes, permutation definitions, protocol,
material implementation hashes, calibration split identities, model and
frontier hashes, chained ledgers, exact optimizer state, convergence histories,
and reports are bound below the ignored
`artifacts/vamp-logt-mlp-permuted-mnist/` root. One resumable CLI owns all
phases:

```bash
uv run python -m apm.experiments.vamp_logt_mlp_permuted_mnist \
  --config configs/vamp_logt_mlp_permuted_mnist/primary.yaml \
  --phase all
```

The other phase values are `calibration`, `hierarchy`, `online`, and
`ceiling`; prerequisites are authenticated and resumed automatically.
