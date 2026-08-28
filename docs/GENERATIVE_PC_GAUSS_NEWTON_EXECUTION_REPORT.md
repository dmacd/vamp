# Exact generalized Gauss–Newton PC evidence: execution report

The Gauss–Newton experiment is complete. The original
`generative-pc-gn-v1` run stopped at its float-precision prerequisite. The
separately versioned `generative-pc-gn-v2` continuation kept that comparison as
a diagnostic and ran the full minimal routing test. GN0 and GN1 were
numerically valid, but neither routed well enough to pass any of the three
minimal conditions. The final v2 verdict is
`not_supported_by_this_implementation`.

## Why v2 continued past the precision warning

The v1 audit found a largest float32-versus-float64 GN1 difference of 0.027267
nats. That is a real discrepancy, so v2 does not call the two computations
equivalent. It records the same audit result and changes only its role from a
hard prerequisite to a diagnostic.

The reason this is defensible is scale. A route chooses the node with the
largest score. If each of the two leading scores could move by the largest
observed amount in opposite directions, their ordering is potentially
precision-sensitive when their separation is at most 0.054533 nats. Only 2 of
11,520 GN route decisions met that conservative condition: one GN0 decision
and one GN1 decision. This is not a mathematical bound on all float error,
because the precision audit used eight fixed images. It is direct evidence that
the observed precision discrepancy cannot explain score deficits of hundreds
of nats or classifier-accuracy gaps of tens of percentage points.

## Final v2 result

The v2 run used identity
`9abf13060bfb972d2aec535ff74e9c06d9e28a01668030f2fb907abaac8f3ad5`.
It authenticated the imported MAP model tree, reproduced its sealed MAP mean
with zero error, passed the analytic formula checks, produced finite GN0 and
GN1 scores, and factorized every unmodified G matrix. Across the three minimal
conditions, raw G Cholesky factorization succeeded for 38,016 of 38,016 scored
states. The exact Hessian was positive definite on 37,018 of those states; its
998 failures remained diagnostic and did not invalidate G.

The novel-leaf condition asks whether a new model trained on the new C4 context
can beat the longer C0 history on held-out C4 images. Here C4 means the
72-degree image rotation with an eight-position label shift; it does not mean
the digit 4. For all three independently initialized replicas, the C4 leaf won
zero of 512 focused images under GN0 and zero of 512 under GN1. GN1's median
leaf-minus-history score was -522.90, -515.17, and -523.93 nats. Task-free GN1
routing classified 24.06%--25.00% of the general held-out images correctly,
whereas a label-aware oracle that chose the best node for each labeled image
reached 89.06%--90.94%. The worst gap was 66.09 percentage points.

The recurrent-leaf condition asks whether a new C4 leaf can beat a history that
has already seen two C4 blocks. Again, the leaf won zero of 512 focused images
in every replica under both GN scores. GN1's median leaf-minus-history score was
-644.20, -645.79, and -658.42 nats. Task-free GN1 accuracy was
28.91%--30.63%, the oracle reached 88.28%--90.31%, and the worst gap was 61.25
percentage points.

The identical-regime condition removes novelty from the comparison: both the
one-block leaf and the sixteen-block history represent C4. A comparable score
should not strongly prefer one solely because it covers a different training
interval. GN1 nevertheless gave the leaf median deficits of -717.07, -722.04,
and -730.89 nats, and the leaf again won zero of 512 images in every replica.
The measured GN1 cross-level offset was 720.95 nats. Ordinary variation between
same-level replicas permitted only 19.84 nats. Task-free GN1 accuracy was
50.94%--51.88%, the oracle reached 90.47%--92.66%, and the worst gap was 41.09
percentage points.

Minimum independent-replica route agreement ranged from 95.16% to 98.91%
across MAP, GN0, and GN1. Thus different random initializations mostly made the
same bad choices. GN0 and GN1 sometimes improved task-free accuracy over MAP by
fractions of a percentage point and reduced some focused deficits by roughly
15--23 nats, but those changes were negligible beside the remaining
512--731-nat deficits.

Neither GN0 nor GN1 passed the minimal stage, so the preregistered workflow did
not run confirmation seeds or partial carry. More settling, more seeds, or a
looser precision tolerance does not address the demonstrated problem: this GN
evidence score remains dominated by model-history interval size. A successor
estimator must fix cross-model score comparability before reopening the later
phases.

The machine-readable summaries, raw arrays, Markdown and HTML reports, and four
plots are under
`artifacts/vamp-logt-pc-mnist/runs/9abf13060bfb972d2aec535ff74e9c06d9e28a01668030f2fb907abaac8f3ad5/`.

## What was computed

For each image, the predictive-coding model inferred a 32-value latent vector
and a 128-value hidden vector for exactly 80 steps. The workflow then computed
four scores at that same state.

- MAP is the complete normalized log joint at the 80-step state. It does not
  include latent-state volume.

- Raw-Hessian Laplace adds a volume term derived from the exact second
  derivatives of the negative log joint. It is undefined when that exact
  Hessian is not positive definite and is diagnostic only in this protocol.

- GN0 uses `G=A^T A` in place of the exact Hessian, where `A` measures how all
  whitened model residuals change when any of the 160 inferred values changes.

- GN1 adds `0.5 g^T G^-1 g` to GN0. This compensates, to quadratic order, for
  the fact that the state can retain a nonzero gradient after the fixed 80
  inference steps. GN1 is the primary experimental score.

No score clips an eigenvalue, takes an absolute determinant, or adds a diagonal
shift. The exact Hessian is computed for diagnosis on every query but cannot
block a valid G score.

## Authenticated source

The run copied exactly the consumed subset of MAP run
`c4643cd904ae9802c6a427868b954e6ff54b960a6c589231ccd9b3ddfb4e06a7`.
The subset contains 106 files and 19,680,412 bytes: the source protocol, the
selected preflight model, three active banks, all 45 active static model
replicas, and their stored MAP scores. Its digest is
`ae124f978a6ca6074567853ace6a6596ee87afaff8525e48bf56a408613b6ae9`.
Recomputed MAP values matched the sealed values exactly in the preflight check.

## Original v1 precision-gate result

Raw Cholesky factorization of G succeeded on 64 of 64 images. Its smallest
eigenvalue ranged from 0.794221 to 0.990030, safely above zero. The exact
Hessian was positive definite on 61 of 64 images. Images 25, 46, and 58 retained
one negative exact-Hessian direction, but their smallest G eigenvalues were
0.935435, 0.965475, and 0.951688. Directly moving the latent state along those
exact-Hessian directions reduced the negative log joint in one direction at
probe distances 0.01, 0.05, and 0.10. Thus the earlier negative-Hessian problem
does not infect G.

GN1 added between 2.3871 and 16.3962 nats to GN0 across the 64 images, with a
median addition of 4.6476 nats. This addition is the measured effect of the
remaining gradient at the 80-step state; it is not an arbitrary correction
constant.

The required numerical check compared GN1 computed in float32 and float64 at
the same inferred states on eight fixed images. The allowed absolute difference
was 0.001 nats. All eight missed that tolerance: the differences ranged from
0.004215 to 0.027267 nats, with a median of 0.012236 nats. A component audit
found that most of the difference already existed in the 784-pixel joint score,
before the G factorization.

Because this numerical prerequisite failed, the v1 workflow did not run the
three minimal routing conditions, confirmation seeds, or partial carry. That
v1 result alone did not say whether GN0 or GN1 routed well or badly. The v2
continuation reported above subsequently answered that question while
preserving v1 unchanged.

The complete machine-readable result, Markdown report, HTML report, raw audit
arrays, and plot are under
`artifacts/vamp-logt-pc-mnist/runs/6ba7bbc1ed5d0e1c5bbd6f7615b3af1e75c93b92004005c34df6d32bd588eede/`.
