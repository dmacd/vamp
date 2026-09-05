# ImageNet-R-50 Macro-Token Integrator Ceiling Protocol

This follow-up asks whether preserving all 197 final, node-adapted ViT tokens
raises the integration ceiling at fragmented LogT frontiers. It does not run a
persistent online replay sweep. The immutable fresh-parent hierarchy, ImageNet-R
split, class order, backbone, LoRA architecture, and stage-matched joint-IID
curve remain fixed.

## Clean selection

The clean hierarchy trains upstream leaves and parents only on the 19,200
router-fit images. At stages 31 and 50, integrator fitting uses the arrived
fit images and early stopping uses the disjoint 4,800-image validation
partition. Seed 1993 evaluates transformer depths 1 and 2 crossed with AdamW
learning rates 0.0001, 0.0003, and 0.001. The selected cell has the lowest mean
best validation NLL over the two stages; an exact tie prefers fewer blocks and
then the lower learning rate. The winning cell is repeated with seeds 1994 and
1995. Test images cannot be requested during this phase.

For each active hierarchy level, the input retains the full final adapted
197-by-768 token sequence. Every token receives parameter-free LayerNorm over
its 768 features. Corresponding positions from six fixed level slots feed one
shared 4608-to-768 projection; inactive slots are zeros and live nodes are never
repacked. A learned 197-position embedding is added. Raw affine logits,
within-node log probabilities, classifier ownership, and active bits form the
exact 3,606-value behavior vector, encoded as a 768-value META token. One or two
pre-normalized, 12-head, width-768 transformer blocks with a four-times GELU
feed-forward layer integrate the 198-token sequence. A direct affine 200-way
classifier reads macro-CLS. There is no raw-union residual skip.

The macro head uses microbatches of 64 accumulated eight at a time, AdamW
weight decay 0.0001, gradient clipping at 1.0, BF16 compute, at least 20 and at
most 100 epochs, and patience 10 on validation NLL. All shared components use
name-derived initialization seeds so the one- and two-block cells begin with
identical shared parameters.

## Controls and owner diagnostics

The v6 final-CLS behavior MLP is retrained on exactly the same clean fitting
population and with its established architecture and optimizer. A frozen
linear owner probe reads the selected classifier's macro-CLS representation.
A separate end-to-end owner model has the selected macro architecture but is
trained directly to predict the active level owning the class. Owner labels
remain exclusively in supervision. Both diagnostics report owner accuracy and
the deployable accuracy obtained by choosing a node without labels and then
classifying within that node's raw affine rows. The label-aware true-node oracle
is reported only as a diagnostic.

## Locked refit and test

After architecture and per-seed stopping epochs are frozen, the selected macro
classifier, the data-matched v6 MLP, and both owner diagnostics are initialized
fresh and refit on all arrived images in the 24,000-image training split for
their corresponding clean-selected epoch counts. A training seal proves that
zero test token requests occurred. Only then are the 6,000 test images opened.
Rejected architecture cells are never evaluated on test. Results at stages 31
and 50 are compared descriptively with true-node routing and the existing
stage-matched joint-IID rank-16 QKV-plus-fc1 LoRA; neither comparison gates the
run.

Full token tensors are reproducible scratch, not portable results. They are
stored as bounded 64-image safetensors shards, capped at 64 GiB, and removed
only after the corresponding models and evaluations are sealed. Models,
selection records, hash-chained requests, compact metrics, resource accounting,
figures, and Markdown/HTML/PDF reports are retained. An immediate completed-run
replay must perform zero node training, zero integrator optimization, and zero
adapted-token forwards.
