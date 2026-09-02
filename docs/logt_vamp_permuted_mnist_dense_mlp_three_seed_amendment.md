# Three-seed amendment for the ungated dense Permuted-MNIST study

## Decision

The primary exploratory analysis uses run seeds 0, 1, and 2. The originally
declared seeds 3 and 4 are no longer required for the primary report. They may
be completed later if the three-seed result warrants a higher-confidence
estimate.

This changes only the number of independent run seeds. It does not change the
data stream, hierarchy construction, model widths, replay budgets, optimizer
settings, evaluation checkpoints, ceiling convergence rule, or three restarts
per ceiling fit.

## Timing and provenance

The user directed this reduction while ceiling seed 0 was still running at
macro step 63 and before any ceiling seed had completed. All five bounded
online seeds had already completed. Online seeds 3 and 4 therefore remain in
the source artifact but are excluded from every primary three-seed aggregate,
criterion, and plot.

The immutable generation protocol remains attached to the source run. The
artifact-local `analysis_amendment.json` records the included and excluded
seeds, the decision stage, and the reason for the change. The amended report
is rendered in a separate analysis view so that excluded seed files remain
recoverable and cannot be selected accidentally by filename globbing.

## Interpretation

Three paired seeds are sufficient for the current exploratory question:
whether logarithmic replay looks promising enough to justify more compute.
They do not provide a precise estimate of run-to-run variance. If the result is
promising, completing seeds 3 and 4 will extend the same source run without
changing the already generated seed-0 through seed-2 evidence.

## Completed outcome

Seeds 0, 1, and 2 completed all 64 online steps and all 64 converged-ceiling
steps under source-run hash
`d38c612562699eb55c578a48a8ea94639596c4009a539c1092b63b76eb4f26c0`.
Every ceiling step used three independent restarts and validation-only epoch
and restart selection. The amended primary report contains 192 learned-ceiling
test-subset cells. Ceiling seed 3 entered its first two steps before the
interrupt completed; that incomplete artifact is retained and explicitly
excluded.

Across macro-steps 15, 31, and 63, with equal weight over the eight test
permutations within each seed, uniform-history replay reached
87.84% ± 0.18% accuracy and 0.4563 ± 0.0090 cross-entropy. Current-only
integration reached 74.59% ± 1.19% and 0.8113 ± 0.0312, while the converged
full-replay integrator reached 89.94% ± 0.27% and 0.3662 ± 0.0056. The online
replay learner therefore closed 81.8% of the current-only-to-four-epoch
cross-entropy gap and remained 2.10 accuracy points below the converged
ceiling. Uniform and range-balanced replay were effectively tied. All seven
frozen decision rules passed.

This is promising exploratory evidence, not a precise three-seed population
estimate. No additional seed is required for the amended report. Extending to
seeds 3 and 4 remains an optional, predeclared confirmation run; the existing
conditions and headline cells must not change if that extension is launched.

## Post-hoc cumulative-baseline extension

The primary report did not separate two possible causes of the fixed 20-epoch
pooled MLP's low early accuracy: limited cumulative training data and incomplete
optimization. It also lacked a converged control that asks how much label
information a learned head can recover from the calibrated base MLP alone,
without any temporal-node features.

A post-hoc diagnostic extension therefore runs the following conditions for
stream seed 0 at the existing full checkpoints 7, 15, 31, 63, and 64:

- the unchanged calibrated base MLP, with no additional training;
- a fresh cumulative MLP initialized from that base, trained on every
  node-training row available by the checkpoint; and
- a fresh integrator over only the frozen calibrated base MLP's normalized
  final hidden activation and class log probabilities, trained on every
  observer row available by the checkpoint.

The two learned conditions use every held-out evaluation row available by the
checkpoint for validation-driven learning-rate reduction, stopping, and
restart selection. They use the existing ceiling convergence rule and three
independent optimizer restarts. Test labels never select an epoch or restart.
The cumulative MLP and base-only integrator retain their original disjoint data
roles rather than pooling node-training and observer allocations.

Only seed 0 is run. These measurements are single-stream diagnostics with no
run-to-run uncertainty estimate. They remain separate from the three-seed
headline aggregates and seven frozen decisions. Their purpose is to compare
the fixed 20-epoch pooled MLP with a converged version, and to compare a
converged frozen-base integrator with the existing converged integrator over all
active temporal nodes.

The extension completed in 570 seconds on the local RTX 4090. All three
restarts of both learned conditions converged at every checkpoint, and every
data-count, presentation-count, validation-selection, seed-scope, and resume
gate passed. At checkpoints 7/15/31/63/64, the converged cumulative MLP reached
79.45/85.10/89.08/91.92/92.48% accuracy with cross-entropy
0.6682/0.4758/0.3704/0.2784/0.2663. The fixed 20-epoch pooled MLP was more
accurate at those checkpoints (82.67/87.01/90.92/92.42/93.06%) but had worse
cross-entropy (0.9663/0.6486/0.5030/0.4343/0.3820). More optimization therefore
did not recover ~98% early accuracy; checkpoint 7 contains only 1,792
node-training examples distributed across seven domains.

The unchanged calibrated base scored 21.11% once all eight domains were
present. The converged frozen-base integrator reached only 57.33% at checkpoint
64, while the converged all-node integrator reached 93.41%. This single-seed
comparison attributes most of the recoverable permutation-specific information
to the frozen temporal nodes rather than to a more thoroughly trained nonlinear
head over the identity-trained base alone.
