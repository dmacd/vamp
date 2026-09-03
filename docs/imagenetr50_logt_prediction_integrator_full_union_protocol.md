# ImageNet-R-50 Full-Union LogT Prediction Integrator Protocol

## Question and protocol change

Can the direct prediction-integrator architecture that succeeded on
Permuted-MNIST recover ImageNet-R class-incremental accuracy without predicting
a task or selecting one node?

This protocol makes full-union parent consolidation the primary condition.
Every parent receives every training example represented by its two children,
matching the successful Permuted-MNIST methodology. The completed v1 run
remains an immutable negative ablation of fixed-K parent replay: its best
K=2,048 hierarchy was 4.263 percentage points below full union at task 16.
That failure no longer blocks the integrator experiment.

The claim boundary is a fixed ViT-B/16-IN21K backbone, rank-16 LoRA, 50
four-class tasks, and the repository's frozen seed-1993 24,000/6,000
ImageNet-R split. This protocol does not claim fixed-memory consolidation.

## Immutable inputs and leakage boundary

The experiment authenticates primary run
`08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db`,
its U100 policy, the pinned timm checkpoint, and the router study's immutable
19,200/4,800 split of the 24,000 training images. The existing U100 diagnostic
is optimistic because its nodes trained on all 24,000 images; it is only a
capacity screen. Prior result summaries, router validation metrics, and the
real batch-64/BF16 preflight are byte-hash-bound inputs.

Clean development trains every leaf, parent, and integrator using only the
19,200 fit identities. The 4,800 validation identities cannot enter node
training, parent source identities, integrator history, or optimizer state.
After all feature and history choices are frozen, the locked run retrains on
all 24,000 training identities and opens the 6,000 test identities.

## Full-union capacity-one hierarchy

One node may occupy each binary-counter level. An arriving leaf repeatedly
carries through occupied levels. Fifty tasks create 47 parents and finish with
three live intervals: tasks 49-50 at level 1, tasks 33-48 at level 4, and tasks
1-32 at level 5.

Every leaf is a fresh rank-16 adapter and four affine classifier rows. A parent
unions disjoint classifier rows exactly, initializes a fresh zero-effect
rank-16 adapter, and trains for five epochs on the complete represented source
union. Node artifacts retain the exact training image identities. No fixed-K
parent condition is selected or used as a gate in this protocol.

## Task-free integrator

Each active level slot contains a declared subset of normalized adapted
pre-logits, adapted affine scores, adapted local log probabilities, frozen-base
features scored by the node classifier, class ownership, and one active bit.
Inactive slots are exactly zero. Labels, task IDs, node truth, arrival index,
and interval endpoints do not occur in observations.

The three feature families remain `scores`, `behavior`, and `behavior_base`.
The smallest family within 0.25 percentage point of the best sealed diagnostic
mean is selected. The parameter-free baseline is the raw affine union. A
residual MLP initialized with a zero output layer has exact raw-union parity.

Fresh ceilings use three deterministic restarts and validation-controlled
stopping. Persistent conditions retain optimizer state and train four epochs
per arrival on every current-task fit image plus a class-stratified historical
reservoir H in {512, 1,024, 2,048}. Historical identities are re-forwarded
against the current frontier; stale behavior tensors are never replayed. Thus
parent training is full union while persistent integrator replay remains
bounded, as in the Permuted-MNIST experiment.

## Gates

1. The sealed U100 diagnostic must reach 85% validation accuracy and beat the
   best static task-free control by 3 points. Otherwise the declared all-leaf
   ceiling runs and the hierarchy study stops.
2. An eight-task real-data smoke must pass topology, finite-training, leakage,
   cache, exact full-source membership, and checkpoint-reuse checks.
3. Through task 16, the smallest H must keep persistent validation accuracy
   within 2 points of the fresh mean at tasks 2, 4, and 8; within 1 point at
   task 16; and beat the strongest static task-16 control by 3 points.
4. At task 50, clean development must reach 79%, beat its strongest static
   control by 5 points, and remain within 2 points of fresh integration.
5. Only then is the locked test matrix evaluated. Paired replications are
   required if both local E2-LoRA metrics are exceeded.

For the locked run, the all-24,000 hierarchy and all 50 persistent checkpoints
must exist before the training seal is written. The seal verifies zero test
behavior requests and binds the final frontier and checkpoint bytes. Only then
may evaluation request any test identity.

## Complexity and reporting

The hierarchy retains at most `popcount(t)` live adapters and performs at most
`bit_length(t)` carries per arrival. A carry's data work grows with its interval:
for N examples per task, cumulative parent example presentations are
O(N T log T), worst-case arrival work is O(N T), and amortized arrival work is
O(N log T), excluding fixed epoch factors. Persistent observation work remains
`popcount(t) * (current + H)` node/example forwards per arrival for fixed H.

The workflow writes atomic checkpoints, immutable safetensors artifacts,
hash-chained ledgers, CSV/Parquet/JSON tables, Markdown and standalone HTML
reports, lineage and accuracy plots, resource accounting, and the exact frozen
matrix. The one supported entry point is:

```bash
scripts/vision/imagenetr/run_integrator_local.sh
```
