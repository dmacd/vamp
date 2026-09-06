# ImageNet-R stage-31 frontier-LoRA adaptation

This clean diagnostic asks whether the five node-specific representations at
the maximally fragmented task-31 frontier can be made easier for the
macro-token integrator to combine. It starts every condition from the same five
sealed hierarchy nodes and the same seed-1993 one-block macro head. The node
classifiers and pinned ViT base remain frozen; all 24 rank-16 QKV/fc1 LoRA
factor pairs in each live node and the complete macro head may update.

The fitting populations are nested deterministic uniform draws from the exact
12,194-image clean fit partition. H is 1,024, 2,048, 4,096, 8,192, or the full
12,194 images. Thus a larger H contains every identity in all smaller H
conditions. Every condition uses 50 epochs, effective batch 64, a one-block
macro head with peak AdamW learning rate 3e-5, and LoRA groups with peak rate
5e-4. Both rates share a 5% warmup and cosine decay to 1% of peak. The LoRA
rate is imported from the matched joint-IID recipe; it was not selected by the
preceding macro-head-only optimizer screen.

For every image, each live node runs its own adapted ViT. Its complete final
197-by-768 token sequence is normalized per token exactly as in the frozen
macro-token experiment. Its immutable affine classifier converts its own
pre-classifier class token into local scores. Those attached tensors construct
the same six-slot META fields and enter the unchanged macro transformer. No
label or task identity enters the model. Non-reentrant activation
recomputation bounds memory by rebuilding one node forward during backward
instead of retaining five full ViT activation graphs.

The fifth fitting condition is exactly the fresh full-fit population. A
frozen-LoRA full-fit control uses the same online image loader, augmentation,
initial macro parameters, schedule, and population, isolating adapter updates
from the switch away from cached center-crop tensors. The existing authenticated
frozen macro result and same-split joint-IID result are external horizontal
references, not gates.

The primary checkpoint for each condition is the epoch with minimum clean
validation NLL; maximum validation accuracy is reported separately. The fixed
3,049-image validation partition never trains the macro head or the adapted
LoRAs, and no test image is opened. The report compares validation accuracy and
NLL, learning curves, optimizer work, LoRA displacement, and whether any one
epoch simultaneously reaches the joint-IID accuracy and NLL references. This
single-seed screen motivates later replication and learning-rate tuning only if
frontier adaptation is useful.
