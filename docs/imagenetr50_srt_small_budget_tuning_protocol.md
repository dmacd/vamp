# ImageNet-R-50 small-budget comparison and H=128 SRT tuning

The user requested ordinary SRT versus uniform at H=256 and H=128, followed
by hyperparameter optimization of SRT's final H=128 accuracy. Commit the H=512
result and these experiment definitions before launching new work. Preserve
the existing split, rank-16 model and core training algorithm.

## Fixed-policy comparisons first

Use the existing follow-up runner with the new H=256 and H=128 configurations:
standard thresholds, old target 0.8, interval unit 8, seed 1993. Each stream
starts cold and continues through all 50 tasks. Uniform exactly copies its
SRT partner's batch sizes, old/current counts, introduction boundaries and
update count. Adapter, affine head and optimizer state persist within a stream.

Each H=256 stream uses 146,176 training image forward/backward pairs; each
H=128 stream uses 121,088. H is a presentation-work parameter, not a bound on
stored history. All arrived training images remain eligible. No quality,
coverage or accuracy gate is introduced.

## H=128 search objective and held-out data

Optimize **task-50 validation accuracy**, not mean-stage accuracy, an early-task
proxy, NLL, or the test set. Break exact accuracy ties by lower task-50 validation
NLL, then canonical candidate identity. Every candidate trains all 50 tasks;
there is no early-task elimination or best-epoch selection. Report all candidates.

Reuse the immutable training-derived 19,200/4,800 fit/validation split from the
original SRT calibration. No test image may enter fitting, validation, confidence
feedback or candidate selection. Keep seed 1993 and the augmentation seed common
across candidates. Search is single-seed and a finite coordinate search, not a
claim of a global optimum. Validation has been reused in earlier studies; final
results are exploratory, not untouched confirmation or a SOTA claim.

## Frozen finite search

1. **Policy grid: 18 full streams.** Cross the original relaxed, standard and
   strict threshold profiles with old fractions 0.5, 0.8 and 0.95, and interval
   units 1 and 8. Keep LoRA/head learning rates at 0.0005/0.01. This includes
   the original standard/0.8/8 recipe as a validation reference.
2. **Learning rates: up to eight new full streams.** For the validation winner
   of phase 1, cross LoRA rate multipliers 0.5/1/2 with head rate multipliers
   0.5/1/2 relative to 0.0005/0.01. Reuse the unchanged candidate rather than
   retraining it. Compare the new candidates with all previous candidates.
3. **Local policy refinement: up to six new full streams.** Around the best
   candidate after phase 2, change one coordinate at a time: raise each threshold
   to power 0.8 or 1.25; adjust the old fraction by -0.1 or +0.1, clipped to
   [0,1]; halve or double the interval unit, rounded to a positive integer.
   Keep the selected learning rates. Deduplicate exact recipes by content hash.

Threshold powers preserve ordering and all five quality levels. They are
candidate hyperparameters to test, not calibrated probabilities or assumed
optima. Momentum 0.9, weight decay 0.0005, maximum batch size 64, presentation
formula and all SM-2 update equations remain unchanged. Changing quality,
mixture, spacing or learning rates may change realized batch sizes and update
counts even at fixed H; report actual work and do not hide this difference.

The search includes at most 32 distinct complete fitting streams, each with
101,888 presentations, plus validation inference. Persist each phase's winner
before deriving its next candidates. Completed candidates and sealed decisions
are immutable and resumable. A numerical divergence is a recorded failed
candidate, not a reason to alter the search after seeing test results.

## Final evaluation and reporting

Seal the final choice from validation before its final full-data refit. Train
the chosen SRT recipe cold on all 24,000 training images for the full 50-task
series, together with a uniform control matching its exact schedule and
optimizer settings. This control is not independently tuned uniform replay.
If the chosen recipe exactly equals the already completed H=128 baseline,
reuse that pair and explicitly report that no new recipe was selected.

Extend the same SRT report. Compare fixed H=128/256/512/1,024/4,096 accuracy and
raw NLL curves with consistent labels. Add the tuned H=128 pair, the full
validation candidate ledger and search cost separately from final-run cost.
Carry the existing stage-matched curves and newer offline task-50 endpoints
forward without interpreting them as gates or H=128 work matches. Preserve
the immutable earlier replay-history input to the checkpoint diagnostics.

Record training presentations, optimizer updates, batch-size distributions,
wall time, per-image replay counts and timing distributions. Authenticate
membership and prediction/checkpoint hashes, verify exact paired schedules,
prove zero-step completed-job reuse and inspect the final PDF. Use one nice-10
GPU worker, bounded loading, resumable checkpoints and active health checks.
