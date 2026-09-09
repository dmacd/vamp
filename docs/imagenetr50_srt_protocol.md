# ImageNet-R-50: single-adapter Spaced Repetition Training

This is a local vision adaptation of Atreya et al., *When to Review: Spaced
Repetition for Continual Pre-Training of Language Models*, arXiv:2608.17530v1,
18 August 2026 (https://arxiv.org/html/2608.17530v1). The authors' code is not
released. Their classification confidence thresholds are unspecified. Section
3.2 uses the pre-update training forward for quality, despite contradictory
extra-forward wording elsewhere; this implementation follows section 3.2.

## Frozen experiment

The existing 24,000/6,000 image split and seed-1993 class order remain fixed.
One rank/alpha-16 adapter covers attention QKV and MLP fc1 in all 12 ViT-B/16
blocks. The base is frozen. An ordinary affine head grows by four classes per
task; cross-entropy and prediction cover all seen classes. Adapter weights,
old classifier rows, and SGD momentum persist without consolidation or resets.
New head rows come from one seed-1993 cold initialization with zero momentum.

SGD matches the existing joint reference: batch 64, momentum 0.9, weight decay
0.0005, adapter learning rate 0.0005, head learning rate 0.01, constant learning
rates, and BF16. There is no gradient clipping or activation recomputation.

At each task, P=4*(current_count+min(H,history_count)) presentations are allowed.
H=1024 and H=4096 yield 294,368 and 844,640 final-run presentations. H limits
work, not stored history. All arrived training images remain eligible.
One shuffled first pass over every current image consumes part of P; afterward
both old and current examples are reviewed. Every view is keyed by seed, image
identity, and exposure ordinal. The initial pass is an explicit local addition.

## Review scheduler

Correct-label probability is exp(-per-image cross-entropy), computed in FP32
from the augmented pre-update forward. Quality is the number of calibrated
confidence thresholds met. No additional scoring forward is used.

SM-2 uses ease E=2.5 initially, minimum 1.3, and the paper's equations 8-10.
Ease is updated before interval growth. Quality below 3 resets successful
reviews to zero and interval to one time unit. First and second successes use
one and six units; later successes use ceil(previous_interval*updated_ease).
Ease updates use exact integer hundredths, including integer ceiling for
interval multiplication. Virtual times are arbitrary-size integers; Parquet
stores their exact decimal strings, decoded by the event reader. This avoids
overflow when an entirely retained population advances to distant reviews.

An immutable leftist heap holds future reviews. Persistent indexed old/current
sets permit uniform draws without scanning history. Old/current unused batch
slots are released to the other due pool. No image appears twice in a batch.
If both pools are empty, advance a separate scheduling clock to the next due
time without an optimizer step. Real optimizer and presentation gaps are
recorded separately. Outstanding reviews are not drained after task 50.

The exposure-matched uniform control shares initial coverage, architecture,
optimization, and each realized SRT batch's old/current counts, but samples
freely from the corresponding arrived pools. It has no SM-2 due state.

## Calibration and selection

Reuse the existing 19,200/4,800 training-derived fitting/validation partition.
Only arrived fitting data enters training; validation never enters scheduler
state. Calibrate each H separately, using P on the fitting population:

- Old fractions: 0.2, 0.5, 0.8 after initial coverage.
- Quality thresholds: relaxed [.05,.15,.30,.60,.85], standard
  [.10,.25,.50,.75,.90], strict [.20,.40,.70,.85,.95].
- Interval units: 1 or 8 scheduling ticks.

Screen eighteen settings per budget on tasks 1-16. Continue the best two per
budget through task 50. Rank by mean stage validation accuracy, then lower
mean NLL, then canonical policy bytes. Freeze both winners before any test
evaluation, restart each final adapter from the pinned base, and run each
paired uniform control. All checkpoints are end-of-budget, not best-test.
There are no accuracy execution gates and only one training seed.

## Persistence, evidence, and reporting

Weights, momentum, indexed-pool order, review state, RNG state, counters, and
trace-chunk references commit atomically every 50 updates and at stage ends.
Chunks publish before checkpoints; uncommitted chunks are ignored on resume.
Retain per-image confidence, quality, requested interval, due time, lateness,
and actual optimizer/presentation/task gaps. Report terminal outstanding
intervals as censored and show replay counts by arrival cohort.

The separate SRT report imports authenticated prior affine H4/H8, MLP H4,
rank-16 joint, aggregate-rank joint, and diagnostic oracle results. The prior
PDF and figure remain byte-identical. The copied figure is accompanied by SRT
overlays, NLL/forgetting, measured training and evaluation work, replay timing,
and calibration evidence. Missing reference NLL remains missing.

Training forwards/backwards equal actual presentations: there are no source
hierarchy costs, quality forwards, or recomputation. Fixed H and bounded task
size give linear training-image work for this fixed architecture. Repeated
prefix evaluation remains quadratic; CPU summaries and population scans are
not included in that linear claim. Timers distinguish training batches, loader
and scheduler work, checkpoint I/O, evaluation, and calibration.

Default execution is `bash scripts/vision/imagenetr/run_srt_local.sh`, one GPU
worker at nice level 10 and four bounded image workers. `status` is read-only;
`report` rebuilds only the separate report from sealed evidence. No older
experiment bootstrap or report writer is invoked.
