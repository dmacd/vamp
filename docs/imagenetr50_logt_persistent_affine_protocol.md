# ImageNet-R-50 persistent single-affine frontier

## Question

This experiment asks whether the stage-31 gains from jointly adapting a LogT
frontier's rank-16 LoRAs and a single affine pre-classifier integrator persist
through the full 50-task stream. It promotes only the H=4,096 and H=8,192
single-affine conditions. No macro-token features or model are used.

## Online state

At stage `t`, the capacity-one binary-counter frontier contains `popcount(t)`
nodes in ascending hierarchy-level order. Each node independently runs the
pinned ViT-B/16 with its own rank-16 LoRA and frozen local classifier. Its
768-value final pre-classifier vector is an input block to one affine 200-way
integrator. There are no score inputs, metadata bits, hidden layers,
activations, dropout, task IDs, or label-derived routing inputs.

The affine head is resized when the live-node count changes. Blocks belonging
to a node whose immutable source hash survives the transition are copied
exactly, including their AdamW moments. A new leaf or consolidation parent
starts from its sealed source adapter and an exact local-classifier block.
Biases and blocks for surviving nodes continue. Thus an existing node's LoRA
and optimizer state continue across arrivals; only nodes replaced by a LogT
carry return to the separately trained consolidation artifact. The base ViT
and every local classifier remain frozen.

## Replay and optimization

Every stage uses all examples from the newly arrived task plus at most H
training examples from earlier tasks. Historical identities are selected by a
deterministic, class-stratified priority namespace that includes the stage and
experiment seed, so an exact resume reproduces a draw while later stages see
new draws. Loss is example-uniform. The H=4,096 arm receives four epochs per
arrival and H=8,192 receives five, matching the minimum-NLL epochs selected by
the locked stage-31 architecture sweep. Both use batch 64 AdamW, affine peak
learning rate 3e-5, LoRA peak learning rate 5e-4, weight decay 1e-4, gradient
clip 1, and the first four or five epochs of the frozen 50-epoch warmup-cosine
schedule. Optimizer state is carried for the affine head and surviving nodes.

The total online presentation count is bounded by
`sum_t epochs(H) * (current_t + min(H, history_t))`; model-path work multiplies
this by at most `floor(log2(50)) + 1` live nodes. This is O(T log T) for fixed H
and fixed examples per task, with the larger H represented as a measured
constant.

## Joint-IID comparisons

The existing stage-matched joint-IID curve is the rank-16, alpha-16 reference:
at every stage a fresh adapter and prefix-wide affine classifier train for five
epochs on all training images seen so far. It is imported byte-for-byte from
the authenticated promoted run.

The new aggregate-rank-matched joint-IID curve uses the same data, five-epoch
SGD recipe, initialization seeds, augmentation, and evaluation, but sets rank
and alpha to `16 * popcount(stage)`. This matches the sum of live frontier LoRA
ranks, not the affine head's parameters, the frontier's pretrained state, or
its multiple ViT forward paths. At one-node stages it is exactly the existing
rank-16 stage-matched condition and is reused rather than retrained.

Both controls are scientific references, never execution gates. The fixed
test prefix is evaluated only after each stage's training state is sealed; no
test identity affects replay, optimization, checkpoint choice, or protocol
selection.

## Persistence and acceptance

The run binds the selected stage-31 H results, source hierarchy, joint-IID
curve, dataset, model, split, software environment, configuration, and material
training code. Atomic epoch checkpoints retain named optimizer state. Every
stage artifact records continued and reset node hashes, replay membership,
work, task metrics, and tensor hashes. A completed replay must require zero new
optimizer steps and leave source hierarchy bytes unchanged.

Acceptance requires exact-union initialization, finite gradients, a correct
eight-task transition trace, 50 sealed stages for both H arms, 50 entries for
both joint-IID curves, no train/test overlap, unchanged source artifacts, and
a report containing stage accuracy, gaps, NLL where available, work, and
frontier-size context.
