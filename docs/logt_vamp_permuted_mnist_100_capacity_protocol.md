# Dense LogT capacity and sample-count study on 100 Permuted-MNIST domains

## Question and comparison arms

Test whether model capacity limits the accuracy of the one-node-per-level LogT
integrator, then test whether twice as many task examples help once the larger
models are in place. Use only run seed 0 and the fixed 100-domain order from the
completed scaling study.

The three report arms are:

1. **Reference model, standard samples.** Reuse the authenticated seed-0,
   one-node-per-level results from the completed five-seed run. Its base widths
   are 1024/1024/512 and its integrator widths are 1024/512/256.
2. **Large model, standard samples.** Train a new 2272/2272/1136 base and a
   1912/956/478 integrator, while retaining the reference allocation of 256
   node-training and 256 observer examples per domain.
3. **Large model, doubled samples.** Use the same larger architectures and
   paired initialization, with 512 node-training and 512 observer examples per
   domain.

The reference base has 2,383,370 trainable parameters; the large base has
9,541,274, or 4.0033 times as many. With seven fixed input slots, the reference
integrator has 4,411,658 parameters and the large integrator has 17,650,160,
or 4.0008 times as many. These are the actual module parameter counts, not a
fourfold hidden-width multiplier.

Reference versus large-model/standard-samples isolates the architecture change
within the limits of a single run seed. Large-model/standard-samples versus
large-model/doubled-samples isolates the additional training examples. The
reference versus large-model/doubled-samples contrast changes two factors and
must not be described as a capacity-only effect.

## Fixed training protocol

- Domains are identity plus fixed pixel-permutation seeds 1001 through 1099,
  in stream order 20260827. Run seed is exactly 0.
- Only the one-node-per-level binary-counter hierarchy is used. The prior
  two-node-per-level condition is not run.
- The large base architecture is fixed before the run by the parameter-count
  target. Train it from a deterministic initialization on the same stratified
  50,000 identity-MNIST training rows and select epochs using the same 10,000
  identity-MNIST validation rows as the reference calibration. The schedule is
  validation-loss convergence between 20 and 100 epochs. There is no width
  search and no test-set selection.
- Both large-model arms share that base checkpoint. Corresponding temporal
  nodes use the same semantic initialization seed (run seed, level, first
  block, last block), although the doubled-sample node sees a strict superset
  of training rows.
- In the doubled-sample allocation, all original model and observer rows keep
  their roles and disjoint extra rows are appended. The 128 unused held-out
  stream rows and the fixed 256-example test subset per learned domain are
  identical across sample arms.
- The persistent uniform-replay integrator trains four epochs at every step.
  Its current batch is 256 or 512 observer examples and, after step 1, its
  historical replay budget is respectively 256 or 512 examples. Current and
  historical losses retain equal 0.5 source weights.
- The fresh full-replay integrator is initialized from scratch and trained for
  exactly 20 epochs at learned-domain counts 1, 2, 4, 8, 10, 16, 26, 41, 66,
  and 100. It uses every observer example available in its arm. Twenty epochs
  is a fixed budget, not a convergence claim or a best-possible ceiling.
- The large integrator initialization is identical across the two sample arms.
  Evaluation never controls training.

The resolved configuration is
`configs/vamp_logt_mlp_permuted_100_capacity.yaml`. The single existing scaling
runner remains the implementation and accepts this config explicitly; this
study does not add another runner or artifact hierarchy beyond one run root
with two arm subdirectories.

## Training-work accounting and empirical fits

Retain the prior definition of training-only work: state initialization, data
preparation, frozen-node feature forwards, and integrator optimizer
forwards/backwards are counted. Evaluation, report generation, checkpoint I/O,
and shared temporal-node training are excluded from condition-specific totals;
shared hierarchy work is reported separately.

For the full-replay checkpoints, let

`g(t) = t log2(t + 1)`.

For each arm and each measured work series, fit both:

- a through-origin curve `work = c g(t)`; and
- an empirical power curve `work = c t^p`, fit by linear regression in log
  coordinates.

Use checkpoints `t >= 4` for both fits and report the coefficient or exponent
and ordinary R-squared on the original work scale. Plot the observed series and
both fitted curves. Also plot training wall time divided by `g(t)`, and frozen
feature-forward example-passes divided by `N t log2(t + 1)`, where `N` is the
observer examples per domain in that arm. A flat normalized curve is the visual
signature of exact proportional growth, while a bounded, declining, or
oscillating curve is also consistent with the `O(t log t)` upper bound. The
frontier size is `popcount(t)`, not a smooth logarithm. Plot integrator backward
passes divided by `20 N t` as a countercheck; that ratio should be exactly one.

Model-pass counts are the hardware-independent scaling evidence. Wall-clock
fits are descriptive because the reference was measured in an earlier process
and GPU timing contains warm-up and scheduling noise.

## Outputs and limits

The report must show all six accuracy traces with unambiguous colors and line
styles, checkpoint tables with literal arm definitions, parameter ratios,
capacity and sample contrasts, empirical fit results, normalized runtime/work
plots, CSV ledgers, a machine-readable summary, Markdown, and self-contained
HTML.

One seed cannot estimate run-seed variance or permutation-order sensitivity.
The reference is authenticated rather than rerun, so its seconds are not a
same-session timing control. Doubling samples also doubles the node-training
and replay work by design. Accuracy differences can answer whether the tested
capacity or data increase helped this fixed order and seed; they cannot by
themselves identify a universal capacity limit.

## Post-run reporting correction — 2026-09-02

The original empirical-fit section above mistakenly scopes fitted curves to
full replay even though the experiment request asked to compare runtime growth
for both conditions. This is a reporting omission, not a change to training or
recorded measurements.

The corrected report must retain the per-update wall-time and forward/backward
plots for both conditions and add all of the following:

- cumulative persistent-replay wall-time and total-forward-pass fits against
  both `T log2(T+1)` and an empirical power law, using every step from 4 to 100;
- persistent-replay cumulative wall time divided by `T log2(T+1)`;
- persistent cumulative frozen-feature forwards divided by
  `N T log2(T+1)`;
- persistent cumulative integrator backwards divided by the exact linear
  count `E N (2T-1)`, where `E=4`; and
- the existing full-replay fit and normalized diagnostics, presented alongside
  the persistent diagnostics rather than as the sole fitted condition.

The report must state that persistent values are cumulative through `T`, while
full-replay values are the cost of one fresh fit at sampled checkpoint `t`.
Derived reports, figures, and summaries may change under this amendment;
training ledgers and checkpoints must not.
