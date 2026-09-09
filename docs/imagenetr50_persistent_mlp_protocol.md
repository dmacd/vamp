# ImageNet-R-50 persistent two-layer MLP extension

This adds exactly one H=4,096 condition to the completed v15 report. Both
affine arms and both joint-IID curves remain immutable comparison results.
Only the new head and its own continuing frontier LoRAs train; all source
leaves and full-union parents are authenticated and reused.

## Architecture and initialization

The inputs are the concatenated final 768-value pre-classifier representations
from every live node's base ViT plus that node's own trainable rank-16 LoRA.
The head is `Linear(768 * live_nodes, 1024) -> ReLU -> Linear(1024, 200)`.
There is no affine skip path, dropout, normalization, macro token, score input,
metadata, additional latent layer, or label-aware training/routing input.

The fixed 1,024 hidden units provide a bottleneck smaller than the 3,840-value
input at the five-node frontier, while retaining more coordinates than one
node's 768-value latent. Head size is `786432 * live_nodes + 206024`, reaching
4,138,184 parameters at five nodes, versus 768,200 for the affine head.
Fixed hidden and output widths keep head arithmetic linear in live-node count.

Let W and b be the exact union of the current source classifiers. The first
200 hidden units initialize to `Wx+b` and the next 200 to its negative. The
output starts with `[I, -I, 0]`, so `ReLU(z)-ReLU(-z)=z` reproduces the source
union. The remaining 624 hidden units use seeded random latent projections
and initially zero output weights. Every first- and second-layer weight
trains; this initialization does not restrict the network to score features.
The head seed is `1993 + 10000 * stage`, in an isolated RNG context.

## What persists

At each arrival, first-layer columns belonging to surviving source-node hashes
carry exactly, including AdamW moments. Their node LoRAs and moments carry too.
A replaced node loads its sealed source adapter; its new first-layer block is
initialized from the new source classifier plus seeded random projections.
Retired columns and node moments are removed.

The hidden coordinate system stays fixed. Hidden biases for existing signed
class coordinates and the 624 extra units carry. Output rows and biases for
all previously observed classes also carry, with their optimizer moments.
New-class rows retain source-union initialization and zero moments. Thus the
MLP remains persistent even at a full consolidation; after carry, its output
need not equal the newly loaded parent classifier. Same-frontier resume loads
every head/LoRA tensor and named optimizer state exactly.

## Matched optimization and work

Replay identities, augmentation/order seeds, four epochs per stage, batch 64,
AdamW, head learning rate 3e-5, LoRA learning rate 5e-4, weight decay 1e-4,
gradient clip 1, and the first four epochs of the 50-epoch warmup-cosine
schedule are identical to the sealed affine H=4,096 arm. Replay membership,
frontier hashes, and epoch budgets are checked before every stage trains.
There is no test-selected checkpoint, sweep, or accuracy gate.

The ViT image-path counts therefore match the affine H=4,096 condition exactly;
the larger MLP changes arithmetic and measured time per path. Full-union source
training is charged once to each alternative algorithm, even though this
invocation reuses it. The O(T log T) claim is confined to fixed-budget model
training, not full-prefix benchmarking or the current CPU replay-selection scan.

## Verification and reporting

Require exact-union initialization, finite real BF16 gradients, trainable
parameter boundaries, exact head/LoRA/moment carry and reset, atomic resume,
zero split overlap, and the correct eight-stage topology. Finish all 50
stages, verify identical source/replay identities, and demonstrate a zero-step
second invocation. Update the existing report with MLP accuracy, NLL, its
true-node diagnostic, gaps to the unchanged controls, parameter counts, and
cumulative training wall-time/model passes. Record the distinction between
rank matching and total trainable-parameter matching; no new joint fit is run.
