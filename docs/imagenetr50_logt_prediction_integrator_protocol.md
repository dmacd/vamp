# ImageNet-R-50 LogT Prediction Integrator Protocol

## Question

Can the direct prediction-integrator architecture that succeeded on the
Permuted-MNIST LogT hierarchy recover ImageNet-R class-incremental accuracy
without predicting a task or selecting one node?

The primary scalable condition is a bounded-replay method. A full-union
hierarchy and fresh full-replay integrators are empirical ceilings, not members
of the scalable method family. The claim boundary is a fixed ViT-B/16-IN21K,
rank-16 LoRA, 50 four-class tasks, and the repository's frozen seed-1993
24,000/6,000 ImageNet-R split.

## Immutable inputs and leakage boundary

The experiment authenticates primary run
`08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db`,
its U100 policy, the pinned timm checkpoint, and the router study's immutable
19,200/4,800 split of the 24,000 training images. The existing U100 diagnostic
is explicitly optimistic because its nodes trained on all 24,000 images. It is
only a capacity screen. The exact prior result summary, router validation
ledger, and real batch-64/BF16 preflight record are byte-hash-bound reference
inputs rather than moving report aliases.

Clean development retrains every leaf, parent, and integrator using the 19,200
fit identities. The 4,800 validation identities cannot enter node training,
consolidation reservoirs, integrator history, or optimizer state. After all
feature and capacity choices are frozen, the locked run retrains on all 24,000
training identities and opens the 6,000 official test identities.

## Capacity-one hierarchy

One node may occupy each binary-counter level. An arriving leaf repeatedly
carries through occupied levels. Fifty tasks create 47 parents and end with
three live intervals: tasks 49–50 at level 1, tasks 33–48 at level 4, and tasks
1–32 at level 5. The order of computation is independent of the stable
six-slot observer layout.

Every leaf is a fresh rank-16 adapter and four affine classifier rows. A parent
unions disjoint classifier rows exactly, initializes a fresh zero-effect
rank-16 adapter, and trains for five epochs. Bounded parents use at most K
training identities. K is selected from 512, 1,024, and 2,048 through task 16.
Reservoirs are class-stratified, retain at least one identity per represented
class, and use permanent per-image hash priorities. A parent is obtained by
bottom-K selection from its two child reservoirs; retired full source sets need
not be reopened.

The full-union ceiling trains the same architecture and initialization on all
represented fit identities. The smallest K whose true-node validation oracle
stays within 2 percentage points of this ceiling at tasks 2, 4, 8, and 16 is
frozen.

## Task-free integrator

Each active level slot contains a declared subset of:

- layer-normalized adapted 768-dimensional pre-logits;
- adapted raw affine scores scattered into 200 global class positions;
- adapted local log probabilities scattered into 200 positions;
- frozen-base pre-logits scored by the node's classifier rows;
- a 200-way class-ownership mask; and
- one active bit.

Inactive slots are exactly zero. Labels, task IDs, node truth, arrival index,
and interval endpoints do not occur in the observation. The three feature
families are `scores` (401 values per slot), `behavior` (1,369), and
`behavior_base` (1,569). The latter gives the preregistered 9,414-value
six-slot input. The smallest family within 0.25 percentage point of the best
sealed diagnostic result is selected.

The parameter-free baseline is the raw affine union: each live node supplies
the logits for exactly the classes it owns. A 9,414/8,214/2,406-to-1,024-to-512-
to-256-to-200 MLP predicts a residual. Its output layer is initialized to exact
zero, so initialization has bit-exact raw-union parity, including sole-node
frontiers. Unseen classes are masked.

Fresh ceilings use three deterministic restarts, AdamW at 1e-3 with 1e-4
weight decay, batch size 512, dropout 0.1, and validation-controlled stopping
between 20 and 100 epochs. Persistent conditions retain optimizer state and
train four epochs per arrival on every current-task fit image plus a
class-stratified historical reservoir H in {512, 1,024, 2,048}. Historical
identities are re-forwarded against the current frontier; stale behavior
vectors are never replayed.

Feature-family selection and every fresh-ceiling comparison use the arithmetic
mean validation accuracy over the three fixed restarts. The best individual
restart is reported but is never the gate statistic.

Behavior tensors use immutable row-addressed safetensors shards keyed by
model, node, transform, and image identity. A changed stage subset assembles
already-cached rows in its requested order and forwards only missing rows; it
does not precompute future or test identities. Training and evaluation have
separate hash-chained ledgers so later evaluation coverage can be added from
frozen stage checkpoints without repeating optimizer work.

## Gates

1. The sealed U100 diagnostic must reach 85% validation accuracy and beat the
   best static task-free control by 3 points. If not, one all-leaf capacity
   ceiling is run and the hierarchy study stops.
2. An eight-task real-data smoke must pass topology, finite-training, leakage,
   cache, and checkpoint checks.
3. Through task 16, K must pass the 2-point hierarchy-oracle gate. The smallest
   H must stay within 2 points of fresh integration at tasks 2, 4, 8, and 16,
   within 1 point at task 16, and beat the best static control by 3 points.
4. At task 50, clean development must reach 79%, beat its best static control
   by 5 points, keep the bounded hierarchy oracle within 2 points of full
   union, and keep persistent integration within 2 points of fresh integration.
5. Only then is the locked test matrix evaluated. Additional paired
   replications are required if both local E2-LoRA metrics are exceeded.

For the locked run, the all-24,000 hierarchy and all 50 persistent checkpoints
must exist before a training-seal record is written. That seal verifies zero
test behavior requests and binds the final frontier and checkpoint bytes. Only
then may the evaluation pass request any of the 6,000 test identities.

## Complexity and reporting

An arrival performs at most `bit_length(t)` parent consolidations. Persistent
observation work is exactly `popcount(t) * (current + H)` node/example forwards
before cache reuse, hence O(log t) work per arrival for fixed budgets and
O(T log T) cumulative work. Leaf work is constant per task. Dense affine-head
arithmetic grows with represented classes and is reported separately from ViT
model passes.

The workflow writes atomic node and integrator checkpoints, immutable
safetensors artifacts, hash-chained stage ledgers, CSV/Parquet/JSON result
tables, Markdown and standalone HTML reports, lineage and accuracy plots, disk
and model-work accounting, and the exact selected matrix. The one supported
entry point is:

```bash
scripts/vision/imagenetr/run_integrator_local.sh
```
