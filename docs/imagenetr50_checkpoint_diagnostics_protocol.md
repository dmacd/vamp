# Fixed-checkpoint probability and replay-retention diagnostics

This is a post-hoc diagnostic, not another training run, model selection,
or replacement benchmark score. No optimizer is constructed. Frozen task-50
checkpoints, the original data split, model architecture, and clean evaluation
transform remain unchanged.

## Matrix and populations

Collect all 200 raw logits on all 6,000 test images for each of three
replay-schedule-matched offline seeds and the H=4,096 standard/strict
old=0.8/unit=8 SRT and paired uniform endpoints: seven checkpoints total.
Also collect clean-view logits on all 24,000 training images for each of the
four SRT/uniform endpoints. Total planned model forwards: 138,000 image paths.
Training logits are fit/retention diagnostics, never generalization estimates.

Use BF16 model inference, batch size 64, and the original evaluation ordering.
Serialize logits and identities in immutable 1,024-image chunks as collected;
resume only after authenticating each committed chunk. Reconstruct NLL in
float64 independently with NumPy/SciPy log-sum-exp. Require exactly matching
predicted classes and absolute per-image NLL differences below 0.00003 against
the original predictions. Report the measured differences, not merely a pass.
Model tensors and original source files must be unchanged after inference.
No calibration output can feed model selection, replay scheduling, or training.

## Cross-fitted temperature diagnostic

Before inference, assign one immutable, common five-fold test partition by
class-stratified image-ID hash order with rotating fold offsets. Each class
appears in every fold, global folds contain 1,200 examples, and memberships
never depend on predictions, accuracy, NLL, replay history, or fitted values.
For each checkpoint and each fold, minimize mean NLL on the other 4,800 images
using one positive scalar temperature; score only that fold's 1,200 images.
Each image therefore has one out-of-fold calibrated score. Fit in float64 by
the convex inverse-temperature derivative, with numerical temperature bounds
[0.01, 100]. Any bound optimum is retained and explicitly flagged. No offset,
class-specific temperature, fold selection, or calibration-recipe search occurs.

Report raw and out-of-fold NLL, top-1 accuracy, multiclass Brier score, and
fixed 15-bin top-label reliability. Positive temperature must leave every
predicted class unchanged. Retain per-fold temperatures, calibration/evaluation
identity hashes, and every per-image raw/calibrated score. Folds measure
calibration variation, not independent training seeds. Offline seed variation
is reported separately; SRT/uniform remain single-seed results per profile.

The same test set has been inspected in earlier studies. Cross-fitting keeps
an image's label out of the temperature used to score it; it does not turn
this post-hoc analysis into an untouched confirmatory test. Original raw
benchmark scores and main accuracy curves stay unchanged.

## Common-image training diagnostics

Use each SRT arm's recorded history to define cohorts, then evaluate exactly
those identities under both its own checkpoint and its paired uniform model.
Primary descriptive cohorts are all training images; images last reviewed
before task 40; that group with last correct-label probability at least 0.9;
images reviewed during tasks 40-50; and the top 1% by SRT presentation count,
breaking ties by image ID. Also retain fixed last-review-stage bins
1-10, 11-20, 21-30, 31-39, 40-49, and 50.

Report cohort counts, accuracy, NLL, correct-class probability, and current
error counts. Preserve per-image last-stage, last-confidence, presentation,
and paired current clean-view scores. These cohorts were motivated by observed
earlier records and are not randomized. Comparing the last augmented pre-update
probability with a final clean-view prediction confounds elapsed learning,
new classes, crop changes, and the last update itself. It is not a matched-view
measurement of forgetting. Unequal sample difficulty and class/arrival mixtures
also limit causal interpretation. No new replay-policy ablation is included.

## Integration and verification

Use one nice-10 GPU process after checking memory and other heavy jobs. Keep
the existing SRT runner lock, bounded loading, progress and measured ETA.
Authenticate dataset bytes, base weights, seven endpoints, source code,
installed environment, saved predictions, and replay-history tables. Test
fold isolation, numerical NLL parity, temperature-rank invariance, paired
cohort identity, interrupted collection, completed zero-forward reuse, and
report layout. Regenerate the existing SRT report with clearly separated
diagnostic pages; never replace original results with calibrated numbers.
