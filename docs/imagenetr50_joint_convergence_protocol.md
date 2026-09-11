# ImageNet-R-50 rank-16 joint-IID convergence study

This follow-up measures a validation-converged training recipe, not a
mathematical accuracy ceiling. The earlier 78.867% reference is a fixed
five-epoch endpoint and remains unchanged. Only task 50 is studied; no
converged stage-matched curve is implied.

## Fixed model, populations, and optimization

Use the existing pinned ViT-B/16, rank/alpha-16 QKV and fc1 LoRAs in all
twelve blocks, and an ordinary 200-way affine head with bias. All 200 classes
are available jointly from the first update. No hierarchy, integrator,
distillation, replay selection, or test-dependent routing is involved.

The immutable 24,000/6,000 train/test split and seed-1993 class permutation
remain unchanged. Development uses the existing training-derived 19,200 fit
and 4,800 validation identities. Test images must not enter development or
training. Three fresh final fits use all 24,000 training images, with seeds
1993, 1994, and 1995. These reuse the development-selected schedule; they are
replications of the fitted recipe, not independent hyperparameter searches.

Use the original SGD settings: batch size 64, momentum 0.9, weight decay
0.0005, initial LoRA learning rate 0.0005, and head learning rate 0.01.
Every epoch visits every fitting image exactly once, with the original
deterministic epoch permutation and augmentation seed (model seed + 50,000).
The augmentation epoch starts at zero, as in the original joint control.
BF16 and frozen backbone boundaries are unchanged. Partial final batches
are retained; the 19,200 and 24,000 populations divide exactly by 64.

## Predeclared stopping and selection

After each development epoch, evaluate the complete validation population
with the deterministic evaluation transform. Retain augmented pre-update
training loss/accuracy and deterministic clean-fit metrics on 2,048
hash-selected fitting images. Fit-probe metrics are diagnostic, not used for
stopping or selection. Report all model forwards and optimizer work.

Maintain separate significant-improvement anchors for validation accuracy
and NLL. An accuracy gain of at least 0.1 percentage point or NLL reduction
of at least 0.002 resets the shared plateau counter; smaller improvements
accumulate against the last significant anchor. After eight consecutive
plateau epochs, multiply both learning rates by 0.2 and reset the counter.
Allow three such reductions. After the third reduction, require twelve
plateau epochs and at least thirty total epochs before declaring a
validation plateau. Improvements in either metric postpone stopping.

The 160-epoch safety limit is not evidence of convergence. If reached without
the plateau rule, record that fact explicitly; do not rename it convergence.
This rule establishes no stationary-point or global-optimum guarantee.

Select the primary checkpoint by maximum validation accuracy, breaking ties
by lower NLL and then earlier epoch. Separately retain the minimum-NLL
checkpoint, breaking ties by greater accuracy and then earlier epoch.
Selection uses raw observed metrics, not the plateau tolerances.

Seal the complete per-epoch learning-rate schedule, stopping reason, primary
epoch, and minimum-NLL epoch before final test evaluation. Replay that exact
epoch schedule on each full-data seed through the terminal development
epoch. Save epoch five, both selected epochs, and the terminal checkpoint.
Evaluate these predeclared checkpoints on the 6,000 test images only after
all final training is complete. Report every seed, mean and sample standard
deviation, NLL, and accuracy. Never select a seed or checkpoint using test
results. The primary reference is the three-seed mean at the
accuracy-selected epoch, explicitly labeled validation-selected.

The full-data fits have 25% more optimizer updates per epoch than development.
Transferring the schedule by epochs holds passes over the available training
population fixed; it is not exact update matching or proof that each full-data
run independently reached its optimum. Terminal versus selected endpoints
make the effect of further training visible without test-based selection.

## Integrity, runtime, and reporting

Freeze dataset, model, environment, source result, configuration, and material
training-code hashes. Keep development/final image memberships explicit.
Use one nice-10 GPU worker and bounded image-loader parallelism. Checkpoint
weights, SGD momentum, RNGs, partial-epoch counters, committed epoch records,
and plateau state atomically. Orphan epoch artifacts are not adopted after a
crash. Publish one immutable result and demonstrate zero-step reuse.

Keep the current SRT and earlier joint results unchanged. Add the new
task-50 endpoint and its across-seed spread to the existing report, with
development curves, learning-rate changes, selection, full-data epoch
histories, and work accounting. Do not draw a new 50-stage curve from one
task-50 measurement. Retain compact histories, per-image final predictions,
protocols, audits, figures, and logs for handoff. Weights and optimizer
checkpoints remain local.
