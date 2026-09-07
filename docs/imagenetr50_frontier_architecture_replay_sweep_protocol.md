# ImageNet-R stage-31 architecture replay sweep

## Question

The full-history stage-31 comparison leaves two unanswered scaling questions.
First, does the single-affine frontier fail only when historical replay is
small, or does its gap from the macro-token frontier persist throughout the
same nested replay ladder? Second, how does a one-path rank-80 joint-IID model
scale when it sees the identical current-plus-history populations rather than
only the full fit partition?

## Matrix

The additive matrix evaluates historical capacities 1,024, 2,048, 4,096, and
8,192. Every cell also contains all 367 current-task images. The historical
identities are the exact prefixes frozen by the parent v10 replay manifest;
the experiment does not draw a second sample. The 3,049 clean validation
identities remain excluded from optimization, and no locked-test image may be
opened.

The single-affine cells preserve the v13 architecture and training protocol.
Five node-specific rank-16 LoRAs remain trainable. Their five final 768-value
pre-classifier vectors are concatenated and mapped directly to 200 logits by
one affine layer initialized as the exact local-classifier union. Every cell
trains for 50 warmup-cosine AdamW epochs, and minimum validation negative log
likelihood selects its primary checkpoint. Epoch five remains a separately
reported exposure-matched checkpoint.

The joint-IID cells preserve the v11 aggregate-rank control. One fresh rank-80,
alpha-80 adapter and one 124-way affine classifier train from zero-effect LoRA
initialization for five SGD epochs. Epoch five is the primary endpoint; the
minimum-NLL epoch within those five epochs remains a diagnostic. Thus the
rank-80 curve tests replay-population scaling under its original recipe rather
than silently granting it the affine frontier's 50-epoch checkpoint search.

## Reporting comparisons

The report must keep checkpoint semantics visible. Its primary scaling figure
has separate rows for minimum-NLL-selected frontier results and for the fixed
fifth epoch. The selected row compares the macro-token and single-affine
frontiers. The fixed row compares macro-token, single-affine, and rank-80 joint
IID after the same number of passes over each matching population. Full-fit
points come from the already authenticated v10, v13, and v11 artifacts.

Learning-curve panels separate architectures so the fifteen trajectories do
not collapse into one unreadable legend. Adapter-displacement panels compare
the macro-token and affine frontier under the same H values. Tables retain the
exact fit count, image presentations, selected epoch, accuracy, NLL, runtime,
and active parameter count for every cell.

## Isolation and persistence

The protocol binds both source configurations and completed full-fit results,
the parent replay manifest, exact population hashes, dataset, model, split,
material code, and installed environment. Each cell writes a hash-chained
history and atomic checkpoint before publishing a compact result and local
model artifact. Rerunning the default workflow must authenticate all eight
cells without an optimizer step. The source hierarchy and existing full-fit
controls remain byte-identical.

