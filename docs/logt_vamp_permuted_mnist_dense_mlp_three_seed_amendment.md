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
