# ImageNet-R stage-31 linear pre-classifier integrator

## Question

The adaptive full-history frontier exceeds every same-split joint-IID control,
but its 12.06-million-parameter macro-token transformer is both large and
architecturally complex. This condition asks whether jointly adapted node
representations need that transformer at all. It replaces the macro head with
the smallest direct learned combination of the five node representations and
changes nothing else in the full-history training protocol.

## Architecture

For each image, the five live frontier nodes independently run the pinned
ViT-B/16 with their own rank-16 LoRA installed. From each path, retain only the
768-value output immediately before that node's frozen affine classifier.
Concatenate those vectors in ascending hierarchy-level order to form one
3,840-value input. A single affine layer maps it directly to 200 global logits,
after which the 76 unseen classes are masked. There is no hidden layer,
activation, dropout, residual union, local-score input, META input, task ID, or
label-dependent route.

The affine layer starts as the exact union of the five local classifiers. For
each represented class, its source classifier row is copied into the block for
the owning node and every cross-node block starts at zero. This makes the
untrained model equal to raw classifier union while allowing training to learn
weights from every node. The local classifiers themselves remain frozen.

The affine integrator has 768,200 trainable parameters. Together with the five
rank-16 LoRAs, the condition has 7,403,720 active parameters, versus 18,691,016
for the macro-token adaptive frontier. The pinned base ViT remains frozen in
both conditions.

## Data and optimization

The condition starts from the same five sealed stage-31 nodes as every v10
adaptation cell. It uses all 367 current-task images and all 11,827 historical
fit images, for exactly 12,194 fit identities. The same 3,049 clean validation
identities remain excluded from optimization, and no test image is opened.

Training inherits the macro condition's deterministic augmentation and epoch
order, effective batch 64, 50 epochs, AdamW, 5% warmup, cosine decay, peak
integrator learning rate 3e-5, peak LoRA learning rate 5e-4, weight decay 1e-4,
and gradient clipping at 1.0. Non-reentrant activation recomputation preserves
the five differentiable LoRA paths within local RTX 4090 memory. Minimum
validation negative log likelihood selects the primary checkpoint. Epoch five
is also reported because it gives exactly 60,970 image presentations, matching
the existing joint-IID and macro-frontier fixed-exposure comparison.

## Isolation and persistence

The additive protocol binds the parent result, source hierarchy, exact fit and
validation identities, model, split, configuration, material code, and
installed environment. Epoch history and checkpoints are written
incrementally. A completed invocation authenticates the result, history, and
model artifact without taking another optimizer step. Compact ledgers and
plots are integrated into the existing stage-31 report; large model tensors
remain local.
