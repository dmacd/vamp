# Development Plan

## Public Repository Surface

- `README.md` now presents VAMP as a public research project and identifies
  TinyWorlds Nouns-v2 as the only experiment currently in usable shape.
  MNIST/FabricPC, TinyShakespeare, TinyStories, and all other TinyWorlds work is
  explicitly labeled as a research proof of concept, negative result,
  notional design, or deprecated prototype rather than a supported benchmark.
- Selected nouns-v2, TinyShakespeare, and MNIST/FabricPC reports are published
  with only their directly referenced SVG/PNG and small text dependencies.
  Raw corpora, checkpoints, manifests, CSV/JSON/JSONL ledgers, optimizer state,
  and other generated artifacts remain excluded.

## Completed Outcome — 100-permutation Capacity and Sample-count Diagnostic

The successor protocol is frozen in
`docs/logt_vamp_permuted_mnist_100_capacity_protocol.md` and resolves under
config identity
`26adf88c61114a8cd32e6aa25dcc2c4aa8bc62b7c020db9878a9d42281492dd2`.
It reuses the authenticated seed-0 one-node results from the completed
five-seed study as the 1x-model/1x-sample reference, then runs a
4.003x-parameter 2272/2272/1136 base and a 4.001x-parameter
1912/956/478 integrator with (a) the original 256 node and observer examples
per domain and (b) paired doubled 512-example allocations. This isolates the
capacity contrast before measuring the extra-sample contrast. It uses seed 0,
one node per level, the persistent uniform-replay integrator at every step,
and fresh 20-epoch full replay at the ten declared checkpoints. The prior
two-node policy is excluded.

The production run is complete. The doubled allocation retains all original
model/observer role assignments, appends disjoint rows, preserves evaluation
rows, and doubles replay through paired base and extra draws.
Hierarchy/checkpoint coordinates authenticate the sample multiplier. The
shared scaling runner accepts this frozen config; no second runner was added.

At step 100, persistent uniform replay reached 75.64% for the reference,
75.02% for the 4x-parameter model with standard samples, and 83.30% for the
4x-parameter model with doubled samples. Thus the isolated capacity contrast
was -0.62 percentage points, while the isolated sample-count contrast was
+8.29 points. The corresponding fresh 20-epoch full-replay accuracies were
77.86%, 79.41%, and 85.15%: +1.55 points from capacity and +5.74 points from
samples. In this single run, more parameters alone therefore did not repair
the persistent integrator; additional node/observer examples did. The latter
is the effect of the entire doubled training-data budget, including doubled
node and replay work, not a free accuracy improvement.

The first generated report mistakenly fitted only full replay even though the
request asked for both conditions. The 2026-09-02 report-only amendment fixes
that omission without changing training. The corrected fit figure gives
persistent cumulative runtime the full top row and full-replay checkpoint
runtime the bottom row. The normalized figure gives each condition separate
wall-time, frozen-forward, and integrator-backward panels, with exact overlaps
called out explicitly.

For persistent replay, cumulative wall time through all 100 updates has power
exponents 1.111, 1.103, and 1.123 for the reference, large/standard, and
large/doubled arms, with R-squared 0.999, 1.000, and 0.999. The corresponding
`T log2(T+1)` fits have R-squared 0.997, 0.995, and 0.997. Cumulative forward
example-passes have power exponent 1.119 and R-squared 1.000, versus
`T log2(T+1)` R-squared 0.992. At step 100 the cumulative wall totals are
12.876, 23.237, and 36.913 seconds. Cumulative frozen-feature work divided by
`N T log2(T+1)` is 0.957 for all arms, while integrator backward work divided
by its exact linear count is 1.000.

Full-replay wall time over steps 4--100 remains better described by an
approximately linear power fit: exponent 0.965 and R-squared 0.999 for the
large standard-sample arm, and exponent 1.000 and R-squared 1.000 for the
doubled-sample arm. Its corresponding `t log t` fits have R-squared 0.973 and
0.976. This difference is expected: persistent integrator work is constant per
update and accumulates linearly, while its frozen-feature work accumulates as
`Theta(T log T)`; one full-replay fit has linear 20-epoch integrator work plus
an oscillating `t * popcount(t)` frozen-feature term.

All 11 acceptance checks pass. The serial focused gates pass 39 tests, the
vision-environment experiment suite passes 20 tests, and the explicit RTX 4090
smoke passes for the 9,541,274-parameter base and 17,650,160-parameter
integrator. All five real plots were visually inspected, and the standalone
HTML uses semantic tables and collapsible sections. An exact resume before the
reporting amendment restored every phase without retraining. Regenerating the
corrected report changed only derived summaries, reports, and figures; both
new-arm metric ledgers remain byte-identical. The authenticated artifacts are
under
`artifacts/vamp-logt-mlp-permuted-mnist-100-capacity/runs/26adf88c61114a8cd32e6aa25dcc2c4aa8bc62b7c020db9878a9d42281492dd2/`.
No required work remains. Replication across seeds would be a new follow-up;
this one-seed result is not a variance estimate.

## Completed Outcome — Five-seed 100-permutation Consolidation Comparison

The five-seed scaling successor is complete under protocol identity
`81f6d7f10d752f6b68058a254945c10db281d852e13066a49f3d29d7787f344f`.
Its amendment is
`docs/logt_vamp_permuted_mnist_100_scaling_five_seed_amendment.md`, its live
configuration is `configs/vamp_logt_mlp_permuted_100_scaling.yaml`, and it uses
the same single resumable scaling entry point as the predecessor. Seeds 0
through 4 share the fixed 100-domain order and vary example allocation,
held-out subsets, node training, replay draws, and integrator optimization.
The experiment therefore estimates run-seed variance but not permutation-order
sensitivity.

The original one-node-per-level policy is paired against a two-node-per-level
policy. On a two-node overflow, the two older residents merge and the newest
resident remains. The first seven input slots retain their predecessor
positions and weights; seven secondary slots are appended with exactly zero
input weights. Both policies run the persistent four-epoch uniform-replay
integrator at every step and a fresh 20-epoch full-replay integrator at steps
1, 2, 4, 8, 10, 16, 26, 41, 66, and 100. The latter is a fixed-compute
high-information comparator, not a converged ceiling.

At step 100, persistent uniform replay reached 74.91% +/- 0.63% with one node
per level and 66.29% +/- 0.76% with two. The paired two-minus-one difference
was -8.62 +/- 1.29 percentage points, and every seed was negative (-6.94 to
-10.16 points). The degradation appears by step 2 and remains broad rather
than being driven by a single seed. In contrast, fresh full replay reached
77.60% +/- 0.53% and 77.93% +/- 0.46%, for a paired +0.33 +/- 0.68-point
difference. The retained two-node features therefore still support the same
accuracy when the integrator receives substantially more cumulative training;
the failure is in adapting the bounded persistent integrator to the larger,
more fragmented frontier, not an evident loss of representational information
in the frozen hierarchy.

The two-node policy makes 19.33% less shared hierarchy-training work per seed
(2,693,120 versus 3,338,240 forward passes, with the same counts backward)
because it consolidates less often. That saving moves work downstream: across
all uniform updates it requires 54.29% more frozen-feature forward
example-passes (566,016 versus 366,848), while integrator forward/backward pass
counts remain fixed by the replay budget. Its 14-slot integrator has 8,160,522
parameters versus 4,411,658 for seven slots, an 84.98% increase in work per
integrator pass that example-pass counts intentionally do not normalize away.
At step 100 the two-node frontier has nine active nodes versus three, uniform
wall time is 0.230 versus 0.145 seconds, and full-replay wall time is 15.369
versus 10.728 seconds.

All ten acceptance checks pass, including exact seed/cell counts, fixed replay
budgets, exact 20-epoch full fits, evaluation exclusion, and exact seed-zero
reproduction of the predecessor under the one-node policy. The focused serial
CPU suite and explicit CUDA structural smoke pass. All four plots use distinct
high-contrast encodings, and the standalone HTML contains semantic tables and
embedded figures. A complete rerun authenticated and resumed every hierarchy
and condition without retraining; all 42 checked reports, plots, summaries,
and ledgers remained byte-identical. The report and machine-readable tables
are under
`artifacts/vamp-logt-mlp-permuted-mnist-100-scaling-five-seed/runs/81f6d7f10d752f6b68058a254945c10db281d852e13066a49f3d29d7787f344f/`.
No required work remains. A separate order-randomized study or a larger
persistent-integrator replay/optimization budget would be a new follow-up,
not an extension of this completed fixed-order protocol.

## Completed Predecessor — Single-seed 100-permutation Integrator Scaling

The single-seed scaling predecessor is complete under protocol identity
`6d4f13fdf7d3aad964b1d8becae3fa7130e05422548576912e6e830506a3710c`.
Its frozen specification is
`docs/logt_vamp_permuted_mnist_100_scaling_protocol.md`; the current config at
`configs/vamp_logt_mlp_permuted_100_scaling.yaml` now selects the completed
five-seed successor above. Its one resumable entry point is
`src/apm/experiments/vamp_logt_mlp_permuted_scaling.py`. The predecessor stream
contains the identity transform followed by 99 distinct pixel permutations.
It compares only the persistent integrator trained for four epochs on 256
current and 256 uniformly replayed examples with a fresh integrator trained for
20 epochs on every example seen so far. The full-replay fit is measured at
steps 1, 2, 4, 8, 10, 16, 26, 41, 66, and 100; the uniform-replay condition is
measured at all 100 steps.

Training work now separates frozen-node and integrator forward example-passes
from integrator backward example-passes and batch calls. Timed training
includes condition initialization, training-data preparation, frozen-node
feature extraction, and optimizer work. It excludes pre/post diagnostics,
test evaluation, reporting, and the shared hierarchy construction, because
neither condition depends on those evaluations and the hierarchy is common to
both. The shared hierarchy's separately reported construction cost was
3,338,240 forward and 3,338,240 backward example-passes.

At step 100, uniform replay used 3,584 forward and 2,048 backward
example-passes in 0.146 seconds; full replay used 588,800 forward and 512,000
backward example-passes in 11.341 seconds. Full-replay backward work increased
exactly 100-fold from step 1 to 100, and its measured update time increased
80.8-fold, corresponding to a two-endpoint empirical exponent of 0.954.
Uniform replay's backward work is constant after step 1. Its forward work is
`2,048 + 512 * popcount(k)` because the fixed replay batch is evaluated by the
active LogT frontier, so it is `O(log k)` in the worst case and sawtoothed under
binary carries. Its mean update time increased only 1.61-fold from steps 1–10
to steps 91–100. Across all 100 updates, uniform replay used 366,848 forward
and 203,776 backward example-passes.

The accuracy result is less decisive than the work result. At step 100,
uniform replay reached 75.64% accuracy and 0.9160 cross-entropy; the 20-epoch
full-replay fit reached 77.86% and 1.0773. Across its ten sampled checkpoints,
full replay won six top-1 comparisons and five cross-entropy comparisons. A
fixed 20-epoch run is therefore a high-information comparator, not a converged
ceiling or a best-possible integrator. The experiment establishes the expected
work separation but does not show that full replay stably solves integration
over 100 one-off permutations.

All seven acceptance checks pass. The focused serial suite passes all 15
tests, the two plots use distinct blue-solid-circle and pink-dashed-square
encodings, and the standalone HTML contains semantic tables and embedded
figures. A complete rerun restored the run without changing the chained
metric-ledger, CSV, or summary hashes. The report, plots, and machine-readable
work table are under
`artifacts/vamp-logt-mlp-permuted-mnist-100-scaling/runs/6d4f13fdf7d3aad964b1d8becae3fa7130e05422548576912e6e830506a3710c/`.

This run is a valid standalone work-scaling measurement, but it is not a
prefix-controlled extension of the earlier eight-domain experiment. Applying
the same `stream_seed` to a 100-domain permutation changed the full schedule:
the old prefix was `[6, 0, 3, 4, 5, 1, 2, 7]`, while the new prefix was
`[5, 90, 10, 63, 35, 71, 85, 43]`. In particular, the identity domain moved
from step 2 to step 44. The scaling runner also changed the uniform replay RNG
coordinate and reduced the test subset from 1,000 to 256 examples per domain.
An eight-step diagnostic exactly reproduced the old reported curve with the
old hierarchy and RNG. On that fixed path, the smaller test subset changed
accuracy by at most 1.0 percentage point and the replay RNG changed it by at
most 2.9 points. At steps 2 and 3 those two changes together accounted for
only +0.22 and -1.83 points, versus total new-minus-old gaps of -6.03 and
-10.43 points; the changed tasks and hierarchy accounted for the remainder.
The new early accuracy curve must therefore not be interpreted as a controlled
scaling continuation of the prior curve.

A comparison-correct successor should preserve the old eight-step domain
prefix explicitly, retain the legacy uniform-replay RNG coordinate, and use a
matched evaluation subset. Repeating with more seeds or replacing the fixed
20-epoch comparator with a validation-selected converged fit would be separate
follow-ups.

## Completed Exploratory Outcome — Ungated Dense-base LogT on Permuted-MNIST

The post-hoc ungated successor to the strict calibration stop is complete. Its
config is `configs/vamp_logt_mlp_permuted_mnist_ungated/primary.yaml`, its
amendment is recorded in
`docs/logt_vamp_permuted_mnist_dense_mlp_three_seed_amendment.md`, and its
source-run identity is
`d38c612562699eb55c578a48a8ea94639596c4009a539c1092b63b76eb4f26c0`.
It authenticated the strict run's full 18-fit calibration ledger, selected the
smallest 1024/1024/512 MLP under an explicit no-gate successor rule, and used
its seed-zero identity checkpoint as the shared 2,383,370-parameter base. The
identity test accuracy was 98.24%; the pooled calibration test accuracy was
98.27%. These test values describe the already-selected successor and did not
drive width selection.

All hierarchy tapes and bounded online runs completed. The generation config
originally declared five seeds, and all five online artifacts had completed
before the user reduced the exploratory analysis to seeds 0, 1, and 2. The
artifact-local immutable amendment excludes online seeds 3 and 4 from every
primary number, criterion, and plot. Converged-ceiling seeds 0 through 2 each
completed all 64 steps with three independent restarts and validation-only
epoch/restart selection. The runner entered ceiling seed 3 before interruption;
its two incomplete steps remain recoverable and explicitly excluded. The
three-seed primary report contains the expected 192 learned-ceiling cells.

At headline macro-steps 15, 31, and 63, uniform-history replay averaged
87.84% ± 0.18% accuracy and 0.4563 ± 0.0090 cross-entropy across the three
seed-level means. Current-only integration reached 74.59% ± 1.19% and
0.8113 ± 0.0312; the equal-probability mean reached 82.82% ± 0.58% and
0.7816 ± 0.0134; the fresh four-epoch cumulative integrator reached
89.55% ± 0.25% and 0.3775 ± 0.0083; and the converged full-replay integrator
reached 89.94% ± 0.27% and 0.3662 ± 0.0056. Replay therefore gained 13.25
accuracy points over current-only training, closed 81.8% of the
current-only-to-four-epoch cross-entropy gap, and remained 2.10 accuracy points
and 0.0901 cross-entropy behind the empirical ceiling. Range-balanced replay
was 0.10 points more accurate but 0.0084 cross-entropy worse than uniform
replay, which is a practical tie at n=3 rather than evidence for either
sampler.

Replay improved rather than sacrificed the current archive: uniform replay
reached 90.45% on the current range versus 89.76% for current-only training,
while improving older-range accuracy from 74.55% to 87.50%. The full-node
integrator beat the base-only replay ablation by 53.86 accuracy points. The
best replay integrator also beat the cross-entropy-selected soft range router
on both metrics, though the hard uniform router was 0.26 points more accurate
and far worse calibrated. All seven frozen decisions and all structural,
budget, zero-slot, frozen-node, and decreasing-loss checks pass. The focused
serial regression suite passes all 11 selected tests; its benchmark-marked
CUDA smoke remains separately deselected in the sandbox, and the explicit RTX
4090 structural smoke passed before the production run.

The post-hoc cumulative-baseline extension is also complete for stream seed 0
at full checkpoints 7, 15, 31, 63, and 64. It leaves every primary three-seed
aggregate and decision unchanged. Each learned diagnostic used three fresh
restarts, cumulative held-out validation selection, and the same converged
stopping rule as the ceiling; all 30 fits converged. The selected cumulative
MLP fits ran 48–50 epochs, and the selected frozen-base-integrator fits ran
55–69 epochs.

At checkpoint 7, after only 1,792 node-training examples spread across seven
seen permutations, the fixed 20-epoch pooled MLP reached 82.67% accuracy and
0.9663 cross-entropy. Training that same base-initialized MLP to the validation
stopping rule reached 79.45% and 0.6682: lower top-1 accuracy but substantially
better cross-entropy. At checkpoint 64, the fixed fit reached 93.06% and
0.3820, while the converged fit reached 92.48% and 0.2663. Convergence therefore
does not explain the missing ~98% early accuracy; the early cumulative learner
has far less and much more heterogeneous training evidence than the 400,000
transformed examples per pooled-calibration epoch.

The frozen calibrated MLP itself reached only 21.11% across all eight domains.
A converged integrator over that base alone improved to 57.33% and 1.3124 at
checkpoint 64, whereas the converged integrator over all frozen temporal nodes
reached 93.41% and 0.2307. The node hierarchy supplies most of the usable
permutation-specific information; a nonlinear head cannot recover it from the
identity-trained base representation alone. At checkpoint 64 the all-node
integrator also exceeded the converged cumulative MLP by 0.93 accuracy points
and reduced cross-entropy by 0.0356, although this comparison has only one
stream seed.

This is promising but exploratory evidence. Three seeds do not give a precise
uncertainty estimate, and the converged fit is an empirical ceiling for the
fixed features, architecture, and search rather than a mathematical optimum.
The complete Markdown/HTML report, condition table, machine-readable summary,
and high-contrast plots are under
`artifacts/vamp-logt-mlp-permuted-mnist-ungated/runs/d38c612562699eb55c578a48a8ea94639596c4009a539c1092b63b76eb4f26c0/analysis/3-seed-primary/`.
Its HTML now renders all six report tables as semantic, striped tables with
numeric alignment and horizontal overflow instead of escaped Markdown rows.
No required experiment work remains. Completing ceiling seeds 3 and 4 is an
optional predeclared confirmation extension if tighter uncertainty is later
wanted.

## Completed Calibration Stop — Dense-base LogT on Permuted-MNIST

The CNN-confound successor is implemented and frozen in
`docs/logt_vamp_permuted_mnist_dense_mlp_protocol.md`. Its isolated config is
`configs/vamp_logt_mlp_permuted_mnist/primary.yaml`, and its one resumable CLI
owns `calibration`, `hierarchy`, `online`, `ceiling`, and `all` phases. The
implementation includes the three-hidden-layer raw-pixel MLP width sweep,
validation-only selection, full-affine de-novo node deltas, immutable 127-node
hierarchy tapes, the complete matched router and integrator matrices, pooled
MLP and four-epoch cumulative references, an every-step three-restart
converged ceiling, chained ledgers, exact checkpoint resume, high-contrast
plots, literal condition descriptions, and the seven frozen decisions.

The focused CPU suite and explicit RTX 4090 structural smoke pass. The complete
18-fit calibration sweep then finished under config hash
`067f7dd57bf9501b745fdcee3ceb9e6be39090525bf17f563e7187d2635fce26`.
Mean identity validation accuracy was 98.130%, 98.053%, and 98.087% for widths
1024/1024/512, 1536/1536/768, and 2048/2048/1024 respectively; the matching
seed-zero accuracies were 98.190%, 98.110%, and 98.020%. All six values missed
their frozen 99.0% gates. Mean pooled validation accuracy was 98.145%, 97.943%,
and 98.001%; every width satisfied the separate requirement of being within
0.25 percentage points of the widest model.

The authoritative calibration summary therefore records `status: ineligible`
and no selected width. Test metrics were not evaluated, no shared base was
published, and the hierarchy, online, and ceiling phases did not run. This is
the intended preregistered stop, not a training or infrastructure failure. The
durable calibration evidence is under
`artifacts/vamp-logt-mlp-permuted-mnist/runs/067f7dd57bf9501b745fdcee3ceb9e6be39090525bf17f563e7187d2635fce26/`.
No work remains under this frozen protocol. Any attempt to improve the dense
base training recipe or change the 99% threshold must be specified as a new
successor protocol; the completed ungated successor above does exactly that
without altering this negative result. The old CNN-based Permuted-MNIST
artifacts remain immutable context and are not a formal paired comparator.

## Completed Outcome — Integrated LogT Behavioral Router on Permuted-MNIST

The experiment specified by
`docs/logt_vamp_mnist_integrated_router_plan.md` is implemented as the isolated
`integrated-router-v4` protocol. The strict configuration is
`configs/vamp_logt_router_mnist/primary.yaml`; the implementation lives in
`src/apm/continual/logt_behavioral_router.py` and the
`src/apm/experiments/vamp_logt_router_*` modules. It authenticates the existing
frozen MNIST CNN, allocates disjoint model/router/evaluation batches, builds a
de-novo top-two adapter at every LogT carry, and trains five independent router
conditions against the same hierarchy trajectory. Router inputs contain only
detached, normalized hidden states, output log probabilities, and active-level
bits. Historical hard or soft targets are recomputed against the current
frontier; example-balanced and range-balanced replay each receive exactly 256
historical examples at every eligible primary step.

The direct run-or-resume command is:

```bash
uv run python -m apm.experiments.vamp_logt_router_mnist \
  --config configs/vamp_logt_router_mnist/primary.yaml
```

The one-seed 15-step smoke gate and all five 64-step primary seeds completed on
the local RTX 4090 under protocol identity
`4b1ed9cf715aa42a951dd71fe2242382ef5f4319d4b10cf0b6e3a4633f7e0b69`.
At the high-active-node checkpoints 15, 31, and 63, `example_soft` was the best
learned replay condition. Averaged over 15 seed/checkpoint cells, it reached
0.42689-nat mean routing regret and 78.877% selected accuracy. The no-replay
router reached 1.89049 nats and 58.535%; the most-recent-range baseline reached
3.13208 nats and 26.550%. Thus the replay router reduced regret by 77.42%
relative to no replay and closed 86.37% of the most-recent baseline's
cross-entropy gap to the exhaustive oracle. Every seed independently closed
85.32% to 87.60%, above the preregistered 75% gate.

Replay also satisfies the qualitative retention criterion. On the untouched
evaluation archive at the same checkpoints, `example_soft` reduced older-range
regret from the no-replay router's 1.95453 nats to 0.44961 nats while
current-range accuracy rose from 82.396% to 83.021%. The replay-distribution
hypotheses are target-dependent rather than universal: with hard targets,
range-balanced replay lowered macro regret from 0.49788 to 0.48196 nats and
worst-range regret from 0.66984 to 0.62706; with soft targets,
example-balanced replay was slightly better on micro, macro, and worst-range
regret. Soft targets improved both replay samplers relative to their hard-target
counterparts.

The hierarchy-versus-routing decomposition rules out missing extant-node
competence as the main limitation. The exhaustive extant-node oracle averaged
0.32360-nat cross-entropy and 89.742% accuracy, compared with 0.60015 nats and
83.007% for the matched checkpoint joint-IID adapter. Routing erased that
hierarchy advantage: `example_soft` remained 0.42689 nats behind the oracle and
0.15034 nats worse than joint IID. The fixed largest-range policy also retained
lower primary regret than the best learned router, 0.38445 versus 0.42689 nats,
although its selected accuracy was 0.272 percentage points lower. The result is
therefore a successful proof of fixed-budget behavioral routing and replay, not
evidence that the learned router is already the best task-free policy.

Every seed passed finite-metric, exact-budget, inactive-mask, nonnegative-
regret, decreasing-loss, and one-candidate parity checks. The measured logical
and physical counters match fixed-budget `O(T log T)` accounting. Twenty-six
serial focused/regression tests pass with one sandboxed CUDA skip, and that
CUDA-specific soft-target device test passes separately on the GPU. A completed
rerun restored all seeds at 64/64 without changing any metric-ledger hash. The
aggregate Markdown, standalone HTML, CSV, nine plots, chained ledgers,
checkpoints, and matched joint-IID references remain under the ignored run
directory. No required implementation work remains. The plan's capacity,
feature, and replay-budget studies remain optional follow-ups and have not been
launched.

## Completed Outcome — Integrated LogT Router on VAMP-AF Rotated-MNIST

The no-retuning successor in
`docs/logt_vamp_rotated_mnist_integrated_router_protocol.md` is complete. It
uses the exact five VAMP-AF contexts: rotations of 0, 18, 36, 54, and 72 degrees
with matching label shifts of 0, 2, 4, 6, and 8 modulo ten. The authenticated
balanced source identities, blocked context order, frozen CNN, adapter/router
settings, five primary seeds, and evaluation checkpoints are fixed before the
run. This is a LogT behavioral-router experiment on VAMP-AF's data contexts;
it neither loads nor changes the sealed spatial AF tree. Its strict config is
`configs/vamp_logt_router_rotated_mnist/primary.yaml`, and its isolated runner
is:

```bash
uv run python -m apm.experiments.vamp_logt_router_rotated_mnist \
  --config configs/vamp_logt_router_rotated_mnist/primary.yaml
```

The smoke gates and all five 64-step primary seeds completed on the local RTX
4090 under protocol identity
`97f5f70a91fa3430e244dc4fd91b67b3c8fd28e5bb1eaa0cb3d7d304e3d32896`.
At checkpoints 15, 31, and 63, `example_soft` was the best learned and best
fixed-or-learned task-free condition. It averaged 2.41164-nat routing regret
and 68.787% selected accuracy, compared with 4.35502 nats and 39.852% for no
replay and 3.64267 nats and 41.691% for the most-recent-range policy. Its
paired per-seed mean regret was lower than both comparators in all five seeds.
This is substantial improvement for criterion 1, but it closes only 33.79% of
the most-recent policy's cross-entropy gap to the exhaustive oracle. Individual
seed closure ranged from 26.80% to 43.49%, so criterion 2's 75% gate fails
without a borderline interpretation.

Replay does not meet the qualitative retention criterion on this task.
`example_soft` reduced older-range regret from 5.47486 to 2.38083 nats, but
current-range accuracy fell from 91.302% to 80.729% relative to no replay.
Every seed lost current accuracy, with paired losses from 7.55 to 16.15
percentage points. The other replay conditions also lost at least 7.9 points
on the aggregate current-range view, so criterion 3 fails rather than being
rescued by another replay sampler or target family.

The replay-balance result is consistent across target families. Example
balancing has lower micro-average regret than range balancing for both hard
and soft targets. Contrary to the preregistered range-balance hypothesis,
range balancing is also worse on both macro-average and worst-range regret for
both target families; the machine-readable hypothesis flag is therefore
false, and the required target-specific reporting clearly falsifies the
hypothesis rather than hiding the disagreement.

The hierarchy decomposition locates the remaining error in routing. The
exhaustive extant-node oracle reaches 95.639% accuracy and 0.13747-nat
cross-entropy, substantially better than the matched joint-IID adapter at
78.109% and 0.71308 nats. `example_soft` remains 2.41164 nats behind that
oracle and 1.83603 nats worse than joint IID. Specialist competence is present
in the live hierarchy; the current observer and router fail to select it.

All smoke invariants pass, fixed-budget `O(T log T)` accounting holds, and the
hierarchy/routing fields are complete. The focused serial regression slice has
31 sandbox passes and one expected CUDA skip; that CUDA-specific test passes
separately with the local GPU visible. A completed rerun restored every seed at
64/64 and left all five chained metric ledgers byte-identical. Reports, plots,
CSV, summaries, checkpoints, and ledgers remain below the ignored
`artifacts/vamp-logt-router-rotated-mnist/` tree. No implementation work remains
for this protocol. If another run is authorized, the next bounded test should
be the preregistered router-capacity sensitivity under a new protocol identity;
it should not retune or reinterpret this completed result.

## Completed Outcome — LogT Prediction Integrator on VAMP-AF Rotated-MNIST

The direct-prediction successor is frozen in
`docs/logt_vamp_rotated_mnist_integrator_plan.md`. It keeps the completed
Rotated-MNIST hierarchy protocol but replaces single-node routing with a
residual ten-class MLP over all seven stable level slots. Its strict config is
`configs/vamp_logt_integrator_rotated_mnist/primary.yaml`, its artifact root is
`artifacts/vamp-logt-integrator-rotated-mnist/`, and its only runner is:

```bash
uv run python -m apm.experiments.vamp_logt_integrator_rotated_mnist \
  --config configs/vamp_logt_integrator_rotated_mnist/primary.yaml
```

The smoke gates and all five 64-step primary seeds completed on the local RTX
4090 under protocol identity
`9b5f70bf484cd19c7624142e80118e32857452d44f45fbb97d0b15df29a6689a`.
At high-active-node checkpoints 15, 31, and 63,
`integrator_example_replay` was the best online replay condition. It averaged
0.81866-nat cross-entropy and 67.437% accuracy, compared with 2.57234 nats and
39.139% without replay and 0.99861 nats and 50.448% for the parameter-free mean
ensemble. It beat both references in every seed. The fresh offline cumulative
integrator reached 0.72279 nats and 69.118%, so example replay closed 94.82% of
the positive no-replay-to-offline cross-entropy gap. Full-node replay also beat
the matched base-only replay control, 0.81866 versus 1.39717 nats and 67.437%
versus 57.305%, showing that the frozen node-specific behaviors supplied useful
information.

The all-criteria main hypothesis nevertheless fails. Example replay reduced
older-range cross-entropy from 3.21211 to 0.85531 nats, but current-range
accuracy fell from 91.458% to 82.135%, a 9.323-point loss against the frozen
2-point limit. Every seed showed the same tradeoff, with current-range losses
from 7.29 to 11.98 points. Relative to the sealed `example_soft` router, the
integrator greatly improved cross-entropy, 0.81866 versus 2.54911 nats, but its
67.437% accuracy was 1.350 points lower than the router's 68.787%; the criterion
required improvement on both metrics. Criteria 1, 2, 4, 6, and 7 pass, while
criteria 3 and 5 fail. The result therefore supports stable fixed-budget
integration and useful cross-node information, but not the preregistered claim
that this integrator dominates routing without sacrificing current-range
plasticity.

All ten smoke and primary structural checks pass in every seed: metrics are
finite, replay and feature-work budgets are exact, inactive and future slots
remain zero, initial parity is exact, node parameters remain frozen, and every
eligible integrator update decreases its loss. The focused integrator and
sealed-router regression slice passes 27 tests with one expected sandboxed
CUDA skip. A completed rerun restored all seeds at 64/64 and left all five
chained metric ledgers and the protocol byte-identical. Reports, plots, CSV,
summaries, checkpoints, and ledgers remain below the ignored
`artifacts/vamp-logt-integrator-rotated-mnist/` tree. A visualization-only
follow-up assigns every condition a fixed high-contrast color, marker, dash,
and bar hatch; marks the five blocked context regions; and averages carry
recovery by macro-step instead of connecting different seeds. The derived PNG
and standalone HTML reports were regenerated with `--render-only`; protocol,
summary, and metric-ledger hashes remained unchanged.

The first visible collapse occurs between steps 13 and 14, not at step 13.
Step 13 is the last C0 presentation and still reaches 99.04% example-replay
accuracy on the 1,000-example C0 test subset. Step 14 introduces C1, changes
the labels from `y` to `y + 2 mod 10`, and expands the plotted test subset to
1,000 C0 plus 1,000 C1 examples. The mean ensemble remains 99.08% accurate on
C0 but reaches only 0.80% on C1, producing the apparent 49.94% aggregate cliff.
The largest old range is confidently wrong on C1: 0.60% accuracy and
18.74762-nat cross-entropy.

The hierarchy transition aggravates that task boundary. Step 14 is also a
carry: the C0 leaf for step 13 and the first C1 leaf are immediately replaced
by one mixed range covering steps 13–14. That mixed most-recent node reaches
69.56% on C1 and 76.48% on C0, while the two larger C0 ranges remain live and
dominate the equal-probability ensemble on C1. At step 15, a pure C1 leaf raises
the label-aware best-node diagnostic to 93.10% on C1, but the mean ensemble is
still only 3.64% there. Specialist competence is therefore present after one
additional step; unweighted combination remains the immediate failure. The
no-replay integrator then adapts to C1 at step 15 (89.04%) by forgetting C0
(12.62%), while example replay keeps a less extreme 66.86%/72.50% C0/C1
balance. This is the same plasticity-versus-retention conflict measured by the
preregistered failure.

No required implementation work remains for this protocol. A successor, if
authorized, should counterbalance whether a context boundary lands on a carry
and should compare immediate cross-context carry with a boundary-preserving
leaf. The present schedule confounds carry parity with context difficulty, so
it cannot quantify how much of the step-14 loss comes from mixed-node creation
versus the new rotated, relabeled task itself.

## Completed Outcome — Converged Full-Replay Integrator Ceiling

The missing optimization ceiling is frozen in
`docs/logt_vamp_rotated_mnist_converged_integrator_ceiling_plan.md`. The parent
`offline_cumulative_integrator` was fresh and cumulative, but it trained for
only four epochs at checkpoints 7, 15, 31, 63, and 64. The successor uses the
same frozen hierarchy features and 973-to-1024-to-512-to-256-to-10 residual
MLP, but initializes three independent fits from scratch at every primary
macro-step, presents every cumulative integrator-training example once per
epoch, and trains until a held-out convergence rule is satisfied. Its strict
config is `configs/vamp_logt_integrator_ceiling_rotated_mnist/primary.yaml`,
its content identity is
`b7a55dbe9bbc563e4cc39f2253ba39e56bf90a578bf2a42b6b42f8f0d17dda98`,
and its runner is:

```bash
uv run python -m apm.experiments.vamp_logt_integrator_ceiling_rotated_mnist \
  --config configs/vamp_logt_integrator_ceiling_rotated_mnist/primary.yaml \
  --phase primary
```

The real smoke and all five 64-step primary seeds completed on the local RTX
4090. All 960 primary restarts converged before the 200-epoch failure cap;
selected fits ran 58.481 epochs on average and at most 97. Every seed passed
all eight gates for finite metrics, cumulative archives, exact example
presentations, exact feature work, independent restarts, parent mean-ensemble
parity, converged selection, and validation/test isolation. The completed
primary rerun restored all seeds at 64/64 without training and left all five
metric ledgers plus the aggregate summary byte-identical.

At the preregistered full-test checkpoints 15, 31, and 63, the certified
`converged_full_replay_integrator` averaged 0.59862-nat cross-entropy and
77.287% accuracy. The parent's four-epoch offline reference reached 0.72279
nats and 69.118%, so the realistic ceiling improves cross-entropy by 0.12418
nats and accuracy by 8.169 points. The best online example-replay integrator
reached 0.81866 nats and 67.437%, leaving a 0.22004-nat and 9.850-point gap to
the ceiling. At the final fully merged checkpoint the ceiling averaged 0.59373
nats and 78.980%, versus 0.71931 nats and 77.120% for the sole frozen node.

This result changes the diagnosis: the fixed features and healthy-sized MLP
can integrate the temporal nodes substantially better than either the bounded
online learner or the four-epoch offline reference suggested. Logarithmic,
fixed-budget replay and sequential optimization are therefore major sources
of the remaining loss. The result is still only an empirical ceiling for the
fixed features, architecture, data allocation, AdamW family, and validation
search; it is not the best function that could exist in an unrestricted
mathematical sense. No implementation work remains for this protocol. Reports,
plots, restart histories, checkpoints, summaries, and chained ledgers remain
under the ignored `artifacts/vamp-logt-integrator-ceiling-rotated-mnist/` tree.
The parent comparison report now authenticates that ceiling against the exact
parent protocol, aggregate summary, and all five primary metric ledgers, then
overlays its five-seed mean at every one of the 64 macro-steps. The accuracy
and cross-entropy plots use a thick cyan hexagon trace and explicitly identify
the converged full-replay ceiling in both the title and legend; the
high-checkpoint control plot includes the same condition. `--render-only`
auto-discovers the latest matching certified ceiling, so a later rerender does
not silently remove the trace. The focused integrator/ceiling regression slice
passes 15 tests, and the derived rerender left the sealed parent protocol,
summary, and five primary ledgers byte-identical.

## Completed Outcome — LogT Prediction Integrator on Permuted-MNIST

The no-retuning cross-task replication is frozen in
`docs/logt_vamp_permuted_mnist_integrator_protocol.md`. It uses the completed
eight-domain Permuted-MNIST hierarchy: the identity ordering plus permutation
seeds 1001 through 1007, independently shuffled within every eight-step block.
It keeps the same 973-to-1024-to-512-to-256-to-10 residual MLP, stable level
slots, fixed replay budgets, disjoint 256/256/128 allocations, five seeds, and
64-step horizon as the Rotated-MNIST integrator. Its strict config is
`configs/vamp_logt_integrator_permuted_mnist/primary.yaml`, its protocol
identity is
`78215f1a411accb0cf1eb4dfbde89fa05b80fcbcefc415756ae58684f2901bd6`,
and its runner is:

```bash
uv run python -m apm.experiments.vamp_logt_integrator_permuted_mnist \
  --config configs/vamp_logt_integrator_permuted_mnist/primary.yaml
```

The smoke gate and all five primary seeds completed on the local RTX 4090. At
checkpoints 15, 31, and 63, `integrator_example_replay` was the best online
replay condition. It averaged 0.78170-nat cross-entropy and 75.801% accuracy,
compared with 1.09155 nats and 64.284% without replay and 0.97631 nats and
75.145% for the parameter-free mean ensemble. Full-node replay also beat the
matched base-only replay control, 0.78170 versus 1.82054 nats and 75.801%
versus 38.195%, so the frozen node-specific behaviors supplied useful
information beyond the base classifier.

The online main hypothesis remains a mixed negative result. Example replay
closed 71.30% of the positive no-replay-to-four-epoch-offline cross-entropy
gap, below the frozen 75% gate. It did satisfy the retention criterion:
older-range cross-entropy fell from 1.12743 to 0.80974 nats while current-range
accuracy changed from 80.469% to 80.365%, a loss of only 0.104 points. Against
the sealed Permuted-MNIST `example_soft` router, however, the integrator was
worse on both required metrics: 0.78170 versus 0.75049 nats and 75.801% versus
78.877% accuracy. Criteria 1, 3, 4, 6, and 7 pass; criteria 2 and 5 fail.

## Completed Outcome — Converged Permuted-MNIST Integrator Ceiling

The same frozen protocol also specifies a separately authenticated full-replay
ceiling. Its config is
`configs/vamp_logt_integrator_ceiling_permuted_mnist/primary.yaml`, its content
identity is
`5ba377fef6cdf430f357fb61732f83130b061d0cbebd8fbc9de0cd3621a73ccd`,
and its runner is:

```bash
uv run python -m apm.experiments.vamp_logt_integrator_ceiling_permuted_mnist \
  --config configs/vamp_logt_integrator_ceiling_permuted_mnist/primary.yaml \
  --phase primary
```

All five 64-step seeds and all 960 primary restarts completed in 3 hours and 8
seconds. Every restart converged before the 200-epoch failure cap; selected
fits ran 59.444 epochs on average and at most 81. Every seed passed all eight
finite-metric, cumulative-archive, exact-presentation, exact-feature-work,
fresh-restart, parent-parity, converged-selection, and test/validation-isolation
gates.

Across checkpoints 15, 31, and 63, the certified ceiling reached 0.57618-nat
cross-entropy and 81.779% accuracy. It beat online example replay by 0.20552
nats and 5.979 accuracy points and beat the four-epoch cumulative reference by
0.08077 nats and 2.594 points. Both improvements held in all five seeds; the
per-seed ceiling means ranged from 0.55636 to 0.58834 nats and from 81.398% to
82.425% accuracy. The ceiling also beat the sealed soft router by 0.17431 nats
and 2.902 points, although it remained 0.25258 nats and 7.963 points behind the
label-aware best-node oracle.

The comparison separates two material deficits without assigning a unique
cause to either. Moving from bounded online replay to a fresh four-epoch
full-replay fit improves accuracy by 3.385 points and cross-entropy by 0.12475
nats; training that same full cumulative allocation to validation convergence
with three restarts adds another 2.594 points and 0.08077 nats. Full replay,
fresh optimization, and additional convergence work are therefore all part of
the observed gap. This protocol does not isolate replay coverage, warm-start
path dependence, epoch count, and restart selection into separate causal
effects.

At final step 64, where the frontier contains only one level-6 node, the
ceiling still improves the five-seed mean from 86.566% and 0.53658 nats for
that node to 87.248% and 0.43817 nats. That final gain is recalibration and
nonlinear remapping of one node's frozen behavior, not cross-node integration.
At the high-activity checkpoints, by contrast, the larger gains establish that
the fixed multi-node features and MLP support substantially better task-free
prediction than the online learner recovered.

Completed reruns restored both online and ceiling seeds at 64/64 without
changing either protocol, either aggregate summary, or any of the ten primary
metric-ledger hashes. The focused shared, Rotated, and Permuted integrator
regression slice passes 21 tests serially. The parent accuracy and
cross-entropy reports now overlay the certified every-step ceiling as a thick
cyan hexagon trace with distinct color, marker, and dash encodings for all
conditions. No required implementation work remains. A causal follow-up would
hold full replay fixed while separately varying epoch budget, warm start, and
restart count; the present ceiling should not be relabeled as that ablation.

## Completed Outcome — VAMP-AF Top-Two Adapter And Routing Failure

The mechanism POC specified by `docs/VAMP_AF_POC_Codex_Spec.md` is implemented
under `src/apm/continual/addressing_first.py` and `src/apm/experiments/` with a
strict protocol at `configs/vamp_af_mnist/poc.yaml`. The implementation uses a
shared frozen seed-0 CNN, immutable functional AF tree transitions, input-only
PCA-median routing, leaf-local AdamW replay, zero-child split parity, and
replay-only two-leaf collapse. Every node now owns full-rank deltas for the
CNN's `3136→128` embedding and `128→10` classifier. The convolutional trunk and
normalized base address remain frozen. Frozen-base, global-replay,
oracle-context, and matched-presentation joint-IID controls use the same
top-two-layer adapter class and frozen feature tables.

The direct command is:

```bash
uv run python -m apm.experiments.vamp_af_mnist \
  --config configs/vamp_af_mnist/poc.yaml
```

Focused tests cover routing determinism, unique leaf ownership, exact
zero-child parity with realistic suffix dimensions, sibling isolation, cap
enforcement, collapse deletion, exact work counters, strict configuration,
deterministic blocked streams, explicit AdamW parity, and a full synthetic
artifact-writing/resume smoke. All 16 focused tests pass in the vision
environment.

The original affine-logit protocol remains recorded under run
`7e1ef0f3899bb5c4888d91ae28a29d7a328c58b76a446b04ca344f10853b017b`;
its 78.788% oracle-context mean motivated the declared adapter-capacity change.
The implemented `top-two-v3` protocol is isolated under run
`c3ad77df09fde94a75e2464450c21486d632bf4f60afe44c9602c6a86acf61af`.
Its five full-rank oracle adapters reached 98.96%, 98.16%, 98.20%, 98.22%, and
98.24%, for a 98.356% mean above the 90% gate. The context probe reached
50.066% and is diagnostic-only. The joint top-two adapter reached 81.426%.

The real 5,000-example smoke completed on the RTX 4090 in 59.96 seconds with
30 splits, 31 leaves, 61 nodes, and no consolidation event. Final AF routed
accuracy was 45.878%, compared with 35.016% global replay, 54.584% joint IID,
60.044% online oracle context, and 24.880% frozen base. The exhaustive
oracle-leaf diagnostic reached 88.008%, but hard routing agreed with its chosen
leaf on only 15.560% of test examples. Adapter capacity is therefore repaired;
the smoke failure is predominantly routing/address alignment.

The authenticated main and consolidation passes are also complete. Across the
three 50,000-example main seeds, AF routed accuracy was 61.170%, 59.132%, and
60.672% (60.325% mean). The matched means were 62.661% global replay, 76.278%
joint IID, 97.059% online oracle context, and 24.880% frozen base. Mean
oracle-leaf accuracy was 99.228%, while mean hard-route/oracle-leaf agreement
was only 4.973%. The final main trees retained 69--72 leaves at the depth-eight
cap after 70--73 splits and one or two consolidations per seed. AF therefore
missed the oracle-context gate by 36.735 points, trailed global replay by 2.336
points, and missed the oracle-leaf gap gate by 38.903 points. Its measured work
ratio also rose in every seed, from first-quartile medians of 2.01--2.08 to
last-quartile medians of 2.86--2.90.

The forced depth-three consolidation stress pass completed 19 splits and 13
collapses, ending with seven leaves and 69.240% AF accuracy. Its worst immediate
collapse change was -0.392 points, so the consolidation-drop gate passed. The
aggregate outcome passes structural invariants, multiple-leaf use, the depth
cap, and consolidation fidelity; it fails the two comparative accuracy gates,
the oracle-leaf routing gap, and the flat-work-trend gate. The full result is a
negative POC for this hard PCA-median address, not an adapter-capacity failure.
Generated reports and ledgers remain under the ignored authenticated run
directory. A 7.5-second rerun reused every sealed phase and pass. Any follow-up
should treat input-address alignment or multi-candidate routing as a new,
explicit protocol rather than tuning this completed result in place.

## Completed Outcome — LogT NCE/TRE Evidence Routing on MNIST

The phase-gated implementation requested by
`docs/Codex Handoff_ NCE-TRE Evidence Routing for LogT-VAMP on MNIST.md` is
complete. It is isolated from the sealed VAMP-AF code and artifacts. The new
runner authenticates run
`c3ad77df09fde94a75e2464450c21486d632bf4f60afe44c9602c6a86acf61af`,
reuses its exact frozen CNN and aligned adapter feature table, reconstructs an
authenticated uint8 raw-image table for evidence only, and never rewrites the
baseline.

Implemented surfaces include a bridge-conditioned full-capacity evidence CNN,
balanced coordinate-replacement NCE/TRE training, a pure standard binary
counter, immutable active-bank state, de-novo top-two adapter replay, atomic
per-node artifacts, checkpoint-before-child-deletion carries, exact work
counters and ceilings, held-out latent-source routing metrics, label-aware
oracle diagnostics, static K selection, consolidation twins, full online
direct/TRE/oracle evaluation, and complete-sentence Markdown/standalone-HTML
reports. The strict protocol is
`configs/vamp_logt_evidence_mnist/nce_tre.yaml`; the only runner command is:

```bash
uv run python -m apm.experiments.vamp_logt_evidence_mnist \
  --config configs/vamp_logt_evidence_mnist/nce_tre.yaml
```

The normalized implementation calibration remains 64-dimensional and
bimodal. Its fixed Bernoulli components are 0.30 and 0.70, with eight TRE
bridges and 2,500 optimizer steps per bridge batch. This calibration choice was
sealed before any MNIST evidence run: the more extreme 0.05/0.95 construction
also saturated the first TRE bridges and therefore could not test offset
recovery. On the local RTX 4090, the final three-replica calibration reached
0.22836-nat TRE RMSE versus 0.66339 for direct NCE, 0.11727-nat maximum
absolute signed bias, and 0.27302-nat maximum inter-replica disagreement; all
five implementation gates passed.

Fourteen focused tests pass in the vision environment. They cover the
normalized ratio and triangle-bound test, exact AddressCNN-width evidence
backbone, uint8 preservation, deterministic training, disjoint LogT intervals,
deterministic carries, fixed-multiple work bounds, scoped child deletion,
strict config rejection, label-isolated routing, complete condition
definitions, deterministic blocked streams, and a two-block artifact/resume
smoke. The broader vision-environment suite passed 807 tests and skipped 275
resource tests; its 23 failures were all missing-`tokenizers` dependency errors
in unrelated TinyWorlds fixtures. All 32 affected fixture tests pass in the
semantic environment where that dependency is installed.

The canonical GPU run completed its preregistered calibration and static phases
on 2026-08-26 under run identity
`2003268ae73e22544cf9801d58b3fa40e724ff58c70bc31c32b120fdebf38b54`.
Calibration passed all five gates: TRE reached 0.22836-nat mean ratio RMSE,
versus 0.66339 nats for direct NCE, with 0.11726-nat maximum absolute signed
bias and 0.27302-nat maximum disagreement between independent replicas.

The static phase then produced the prescribed controlled stop. None of K=2,
4, 8, or 16 passed. Every candidate had a maximum held-out adjacent-waymark
balanced accuracy of 1.0000, above the allowed 0.90, so at least one bridge in
every schedule remained completely separable. K=8 and K=16 did pass the 0.90
independent-route-agreement gate, at 0.9998 and 0.9817 respectively, but every
candidate also failed the routed-classifier gate: the worst seed/replica gap
from the label-aware oracle was 0.7225 for K=2 and K=4 and 0.7242 for K=8 and
K=16, rather than at most 0.10. The runner therefore selected no K and correctly
did not run the block-64 consolidation or 100-block online phases. The readable
Markdown and standalone-HTML reports are retained under the ignored run
directory. Any attempt to use more than 16 bridges, alter the waymark schedule,
or change the shared reference is a new protocol; it must not tune or overwrite
this completed negative result.

### Completed correction — frozen-base training-image reference

The user clarified on 2026-08-26 that the intended shared reference was the
frozen CNN's training-image distribution, not independent uniform pixels. The
versioned correction is implemented without modifying the completed uniform-
reference artifacts. The new strict config is
`configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml`, and
`docs/NCE_TRE_BASE_REFERENCE.md` freezes the interpretation.

The corrected \(Q\) samples complete images uniformly with replacement from
all 60,000 original unrotated MNIST training images used for the sealed base.
The reference cache binds the parent protocol, base checkpoint, source IDX,
example count, and quantized tensor hash. Every adjacent pair shares one donor
image; the final replacement endpoint is exactly that intact donor. Focused
tests cover the empirical endpoint, deterministic model training, strict
configuration, raw-only API, and the resumable two-block bank. Fifteen focused
tests and all 23 affected regression tests pass.

The corrected canonical GPU run completed on 2026-08-26 under run identity
`fa2b8bf7d301b0c096d35cdbd6af1ed9b9369ee7e376d96d74e384019417ef49`.
The authenticated reference contains exactly 60,000 images and has content
hash `c3f355699f910376f8f8376956d367ff5df3883cc339decd2bcd28fb2e61f3a4`.
The unchanged normalized calibration passed all five gates. The genuine MNIST
static phase then failed all three routing gates for every candidate K, so no K
was selected and the runner correctly stopped before consolidation or online
evaluation.

Direct NCE averaged 51.49% routed classifier accuracy over the three stream
seeds and three independent replicas, compared with 98.29% for the
label-aware node oracle. Its mean gap from the oracle was 46.80 percentage
points and its worst gap was 62.62 points. TRE's best mean routed classifier
accuracy was 58.51% at K=4. The maximum adjacent-waymark balanced accuracy was
100.00%, 100.00%, 99.95%, and 99.10% for K=2, 4, 8, and 16 respectively, all
above the allowed 90%. The minimum agreement between independently trained
routers was 6.48%, 67.92%, 9.75%, and 8.68% respectively, all below the required
90%. Their worst oracle gaps were 51.62, 41.65, 49.80, and 47.45 percentage
points respectively, all above the allowed 10 points. This is the completed
negative result for the intended reference; any new schedule, training budget,
or gate is a separate protocol rather than unfinished work here.

### Completed implementation and gated outcome — normalized generative-PC evidence

The implementation handoff for a separate generative predictive-coding
experiment is preserved verbatim in
`docs/CODEX_HANDOFF_LOGT_GENERATIVE_PC_EVIDENCE.md`. It proposes node-local
normalized PC density models, complete MAP and Laplace evidence scores, three
controlled 31-block LogT schedules, exact analytic score tests, a one-node
quality preflight, and mandatory static routing gates before any partial carry.
The repository now targets Python 3.11, JAX/JAXlib 0.7.0, and FabricPC 0.4.0 at
commit `138941ef5763ab202c7df07879d3f21678e6cc0a`. The new image-only density
backend, strict protocol, authenticated raw-data boundary, PC-specific LogT
bank lifecycle, work counters, phase-gated workflow, reporting, and command
entry point are implemented. Dense-Hessian batches are limited to four, one
compiled backend is reused per replica during bank construction, and JAX caches
are released between independent preflight candidates and static conditions.
All 18 focused backend, protocol, artifact/resume, and legacy FabricPC
regression tests pass serially in a clean Python 3.11 environment. Heavy JAX or
PyTorch suites must remain serial because the repository's default four pytest
workers can exhaust host RAM by loading four independent runtimes.

The canonical GPU workflow completed its analytic and preflight phases on
2026-08-27 under run identity
`2045bf96a406251ae9fa8825a93c9abe1933df28becfa0032939b5274879626b`.
The analytic implementation passed: the maximum linear-Gaussian Laplace error
was `4.44e-16` nats, and omitting the latent prior caused at least `2.5233` nats
of error. All eight one-node training candidates passed the learning gates.
The selected training candidate used image precision 100, hidden precision 1,
and inference step size 0.01; it reached 83.20% held-out classifier accuracy,
reduced the median latent-gradient norm by a factor of 236.27, and improved the
mean complete joint score over the untrained model by 7,344.14 nats.

The preflight nevertheless failed its curvature gate. Each permitted global
diagonal Hessian shift (`1e-8`, `1e-6`, and `1e-4`) yielded finite regularized
Laplace scores for 60 of 64 audit images, or 93.75%, below the required 99%.
The runner therefore selected no global shift and correctly did not execute
static routing or partial carry. The preregistered verdict is inconclusive: the
model learned and the fixed 40 inference steps substantially reduced its
latent-state gradients, but four resulting states were not positive-curvature
Laplace expansion points. The implementation plan is complete. Any attempt to
expand the shift range, change the mode-finding rule, or continue with MAP alone
is a new protocol rather than unfinished work in this run.

A post-hoc audit of the exact 64 images localizes the failure. Images 21, 46,
53, and 58 each retained exactly one negative-curvature Hessian eigenvector
after the 40 fixed inference steps. Diagonal Hessian shifts slightly greater
than 0.4102, 0.1281, 0.7910, and 1.5470 respectively would have been needed to
make those step-40 Hessians positive definite, compared with the largest
permitted shift of 0.0001. All four digits were classified correctly, so the
failure is not explained by digit recognition. The ground-up 64-image visual
report is `output/pdf/vamp-logt-pc-64-image-curvature-report.pdf`.

A bounded GPU follow-up then reran those same four images from the same zero
initialization for 80 total inference steps at the unchanged step size of 0.01.
All four moved from one negative Hessian eigenvalue at step 40 to zero negative
eigenvalues at step 80. Their minimum eigenvalues became 0.4239, 0.0625, 0.6452,
and 0.6968; their complete negative log joints fell by 41.41, 30.52, 43.00, and
42.22 nats; and their gradient norms fell from 12.08–16.38 to 4.42–6.88. Thus
80 steps repair the observed curvature failures, but the remaining nonzero
gradients do not establish full stationary-point convergence. The reproducible
probe is `scripts/diagnose_vamp_logt_pc_80_steps.py`, and its report and raw
measurements are in `output/vamp-logt-pc-80-step-diagnostic/`. The canonical
40-step failure remains unchanged. The authorized `generative-pc-v2` successor
fixes 80 steps consistently for training and scoring, which forced a new config
identity and new model fits rather than reusing v1 checkpoints.

That v2 workflow completed its analytic and full 64-image preflight on
2026-08-27 under run identity
`e9f1d732b04a230cce243b3d70cd336c44bfc4fabf95b1ee2301af45fd85af7b`.
The analytic check passed, and all eight newly trained candidates passed the
learning gates. The selected training candidate again used image precision 100,
hidden precision 1, and inference step size 0.01. It reached 81.25% held-out
classifier accuracy, reduced the median latent-gradient norm by a factor of
603.54, improved the mean complete joint score over the untrained model by
7,351.64 nats, and reached reconstruction MSE 0.005111.

The newly trained 80-step model still failed the curvature gate. Exactly 61 of
64 audit states had a positive-definite latent Hessian, or 95.3125%; the
preregistered 99% threshold requires all 64 images at this sample size. The
permitted global diagonal shifts of `1e-8`, `1e-6`, and `1e-4` did not repair
any of the three failures. Images 25, 46, and 58 each had exactly one negative
Hessian eigenvalue, with minimum eigenvalues -0.280943, -0.143258, and
-0.687548. Their final latent-gradient norms were 5.776, 7.020, and 10.098, so
the run does not claim that these states reached stationary points. Images 46
and 58 were classified correctly; image 25, a digit 2, was classified as 1.

The v2 result improves the raw pass count only from 60/64 to 61/64. Because v2
retrained the density model, the identities also changed: v1 failures 21 and 53
passed, v1 failures 46 and 58 remained, and image 25 became a new failure.
The runner therefore selected no complete protocol and correctly stopped before
static routing or consolidation. This closes the authorized 80-step protocol as
another negative preflight result, rather than evidence that 80 fixed steps are
a general solution. A readable visual audit, complete-sentence report, and raw
measurements are in `output/vamp-logt-pc-v2-curvature-audit/`; the workflow log
is `output/vamp-logt-pc-v2-run.log`. Any further attempt should be a new protocol
with a changed mode-finding rule, such as a longer or convergence-controlled
settling schedule, and must pass the same full preflight before routing.

The separately versioned `generative-pc-map-v1` branch is now implemented and
complete. It retained v2's 80-step training and scoring schedule but used only
the complete normalized joint score at the resulting inferred state. It did
not compute a Hessian, a Laplace correction, importance weights, or a
multi-start curvature audit. The canonical GPU run completed on 2026-08-27
under run identity
`c4643cd904ae9802c6a427868b954e6ff54b960a6c589231ccd9b3ddfb4e06a7`.
The analytic formula check and all eight one-node learning candidates passed.
The selected candidate used image precision 100, hidden precision 1, and
inference step size 0.01. It reached 81.25% held-out classifier accuracy,
reduced the median latent-gradient norm by a factor of 603.54, improved the
mean complete joint score by 7,351.64 nats, and reached reconstruction MSE
0.005111.

Here `C0` means the unrotated MNIST context with unchanged labels, while `C4`
means the 72-degree-rotated context whose labels are shifted by eight modulo
ten; `C4` does not mean the digit 4. MAP routing failed every required minimal
static condition. In the novel leaf condition, the new C4 leaf lost to the
sixteen-block C0 history on all 512 focused C4 images in each of three
independently initialized replicas; its median score deficits were 537.78,
535.20, and 539.97 nats. In the recurrent
leaf condition, a new C4 leaf also lost all 512 comparisons to a history that
already contained two C4 blocks, with median deficits of 666.34, 663.02, and
675.42 nats. In the identical-regime control, both compared nodes represented
C4, but the one-block leaf again lost all 512 comparisons to the sixteen-block
history. Its median deficits were 719.95, 724.18, and 730.08 nats. The measured
cross-level offset was 724.18 nats, versus an allowed 19.88 nats based on
ordinary replica variation.

The three-replica route agreements of 96.41% to 98.91% show that this was a
stable preference for the larger history models, not unstable randomness.
Task-free routed classifier accuracy was 23.75%--30.47% in the two leaf tests
and 50.63%--51.88% in the identical-regime control, while the diagnostic
label-aware oracle reached 88.28%--92.66%. Every persisted Hessian and
importance-sampling counter is exactly zero. Because no MAP score passed the
minimal seed, the runner correctly skipped confirmation seeds and partial
carry. The final verdict is `not_supported_by_this_implementation`; this closes
the MAP-only branch rather than leaving the main experiment unfinished.

Curvature-aware or other posterior-volume estimators are parked for later work.
They must return under a new protocol and may not reinterpret the completed MAP
artifacts. The next such protocol should explicitly test whether its score is
comparable across independently trained nodes with different interval sizes
before it is allowed to run confirmation or partial carry.

### Implemented gated outcome — exact generalized Gauss–Newton evidence

The separately versioned `generative-pc-gn-v1` workflow is implemented. It
defines the whitened residual vector explicitly, constructs the dense
`G=A^T A` matrix over all 160 inferred values, and evaluates MAP,
raw-Hessian Laplace, GN0, and GN1 at one shared 80-step latent state. GN1 adds
the nonzero-gradient term `0.5 g^T G^-1 g` and is the primary score. The
implementation never clips eigenvalues, takes absolute determinants, or adds
damping. The exact Hessian is retained for every scored query but is only a
diagnostic; it cannot block or approve a GN route. Negative-Hessian states are
probed in both directions at distances 0.01, 0.05, and 0.10 using actual
changes in the negative log joint.

The minimal phase authenticates and copies a 106-file, 19,680,412-byte subset
of sealed MAP run
`c4643cd904ae9802c6a427868b954e6ff54b960a6c589231ccd9b3ddfb4e06a7`.
That subset includes the MAP protocol, selected preflight model, three active
banks, all 45 active model replicas, and the raw MAP score files. Its tree
digest is
`ae124f978a6ca6074567853ace6a6596ee87afaff8525e48bf56a408613b6ae9`.
The workflow verifies the tree before scoring, copies it into the new run, and
requires recomputed MAP scores to agree within 0.0001 nats. Static scoring is
checkpointed after each replica and condition. If a GN score passes minimal,
the implemented downstream path trains fresh confirmation models for stream
seeds 1 and 2 and then runs the existing block-27 to block-28 partial-carry
comparison. One GPU process, batches of four, and an external 8 GiB host-memory
hard limit bound the canonical execution.

All 26 focused backend, analytic, source-authentication, topology, work-counter,
conditional-Hessian-routing, resume, and legacy-backend tests pass serially.
The canonical bounded GPU run used identity
`6ba7bbc1ed5d0e1c5bbd6f7615b3af1e75c93b92004005c34df6d32bd588eede`.
The analytic checks passed. In the fixed 64-image real-model audit, raw G
Cholesky factorization succeeded for all 64 images, with smallest G eigenvalues
from 0.794221 to 0.990030. The exact Hessian remained indefinite for images 25,
46, and 58, with smallest eigenvalues -0.280943, -0.143258, and -0.687548;
their corresponding smallest G eigenvalues were 0.935435, 0.965475, and
0.951688. Actual negative-direction probes decreased the negative log joint in
one direction at all three distances for each of those images. This confirms
the intended structural property: G stays positive definite even when the raw
Hessian is not.

The workflow nevertheless stopped before static routing because the sealed
numerical-precision check failed. Recomputed MAP parity was exact, all GN scores
were finite, and G succeeded 64/64, but none of the fixed eight images met the
required 0.001-nat float32-versus-float64 GN1 tolerance. Absolute differences
ranged from 0.004215 to 0.027267 nats, with a median of 0.012236 nats. A
component audit localized most of the discrepancy to float32 evaluation of the
784-pixel joint score rather than the 160-by-160 factorization. The gated
verdict is therefore `inconclusive`; this run says nothing about GN routing
accuracy because it did not execute the minimal static conditions. Moving GN
scoring to float64 or changing the tolerance requires a new protocol revision,
not reinterpretation of this run.

The explicitly authorized `generative-pc-gn-v2` continuation is now complete.
It preserves every v1 formula, source model, 80-step latent state, route gate,
and downstream stopping rule. Its only protocol change is to retain the same
eight-image float32-versus-float64 comparison as a diagnostic instead of a
prerequisite. The v1 run remains immutable and inconclusive. The v2 identity is
`9abf13060bfb972d2aec535ff74e9c06d9e28a01668030f2fb907abaac8f3ad5`.

The bounded one-GPU v2 run passed source authentication, exact MAP parity, all
analytic identities, and every required GN numerical check. Raw G Cholesky
factorization succeeded for all 38,016 scored states across the three minimal
conditions, and every GN0 and GN1 score was finite. The exact Hessian remained
diagnostic and was positive definite for 12,191/12,672 novel, 12,385/12,672
recurrent, and 12,442/12,672 identical-regime states.

Neither GN score passed any minimal condition. On the focused leaf queries,
the new leaf won zero of 512 images in every replica under both GN0 and GN1.
GN1's median leaf-minus-history score was -523.93 to -515.17 nats in the novel
condition, -658.42 to -644.20 nats in the recurrent condition, and -730.89 to
-717.07 nats even when leaf and history represented the identical data regime.
The identical-regime GN1 cross-level offset was 720.95 nats against a 19.84-nat
allowance. GN1 task-free routed accuracy was 24.06%--25.00% for novel data,
28.91%--30.63% for recurrent data, and 50.94%--51.88% for the identical regime,
while the label-aware oracle reached 88.28%--92.66%. Replica route agreement was
96.25%--98.75%, so the failure is stable rather than random.

The observed precision discrepancy is too small to explain this result. Using
twice the largest audited float difference, 0.054533 nats, as a conservative
route-sensitivity threshold flags only 2 of 11,520 GN route decisions: one GN0
decision and one GN1 decision. This comparison is not a global error bound, but
it separates the measured precision risk from the hundreds-of-nats score bias
and 38.59--66.09 percentage-point oracle gaps. The final v2 verdict is
`not_supported_by_this_implementation`. Because neither GN0 nor GN1 passed the
minimal seed, fresh confirmation and partial carry were correctly skipped.
Further work should address the score's strong dependence on model-history
interval size rather than adjust float tolerance or spend compute on later
seeds under this estimator. The final Python 3.11/FabricPC 0.4.0 focused suite
passes all 21 backend, protocol, source, topology, serialization, and resume
tests serially, and the regenerated Markdown, HTML, JSON, and four PNG plots
were inspected successfully.

## Completed Outcome — ImageNet-R-50 Log-t VAMP Local Experiment

The isolated PyTorch vision experiment under
`src/apm/continual/vision/imagenetr/` is implemented and complete on the local
RTX 4090. Its resolved seed-1993 protocol freezes all 30,000 ImageNet-R images
into a byte-verified, hard-linked 24,000/6,000 train/test split, fixes 50
four-class tasks, and binds the exact
`vit_base_patch16_224.augreg_in21k` checkpoint revision and SHA-256. The
dedicated `.venv-vision` environment, `vision` project extra, resolved primary
configuration, and local bootstrap/run scripts leave TRACE and its environment
untouched.

Run contract
`08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db`
completed all 90 registered jobs with no failures. It sealed 50 immutable
rank-16 leaves once, all frozen/sequential/joint controls, the complete
42-merge union-retrained hierarchy, SVD/Core+TSV/output-drift hierarchies at
zero and five-percent repair, and all three one-percent repair follow-ups after
five-percent repair materially helped. Both mandatory rebuild records preserve
all leaf hashes and report zero new leaf optimizer steps. The final report and
complete CSV/Parquet/JSON ledgers live under
`artifacts/imagenetr50/runs/08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db/reports/`.

The paired local results do not support a task-free Log-t VAMP advantage. Joint
IID reaches 78.867% final affine accuracy; the pinned, unchanged E²-LoRA
reproduction reaches 78.10% final and 82.9952% incremental-average accuracy.
Union-retrained Log-t reaches 59.533% task-free affine accuracy but 79.350%
with the true-node oracle, exposing a 19.817-point addressing/calibration gap.
The all-leaf bank reaches 93.117% with the true-task oracle but only 52.267%
affine task-free. Five-percent repair raises the compact trees by roughly five
points, with Core+TSV best at 47.517% affine and 69.183% true-node oracle, but
SVD, Core+TSV, and output drift are effectively tied. Published E²-LoRA
78.58/83.96 values remain external context because its publication split is
not recoverable; no SOTA claim is made.

Acceptance is complete: real batch-64 BF16 preflight and zero-LoRA parity pass;
all source, proxy, repair, and calibration identities are training-only; the
pinned external checkout remains clean; focused vision tests, bytecode
compilation, and dependency checks pass; and the complete repository suite
passes 782 tests with 275 resource skips. Any follow-up should start from the
sealed primary report and isolate the dominant addressing/calibration loss
before broader proxy/rank/scale or CtM sweeps.

## Completed Outcome — ImageNet-R-50 Recursive Learned Router Capacity Failure

Protocol
`e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0`
completed the preregistered negative branch on 2026-08-21. The A0-A3 validation
capacity gate closed: the I-U100 true-node oracle reached 97.646%, while the
centroid baseline reached 64.562%, R0 reached 59.083%, descriptor-only R1
reached 58.729%, and adapter-response R3 reached 57.750% routed accuracy. R1
and R3 missed the required at-most-one-point oracle gap by 38.917 and 39.896
points, respectively. The paired R3-minus-R1 difference was -0.979 percentage
points on 4,800 router-validation images (paired bootstrap 95% interval
-1.812 to -0.167); R3 did not recover the addressing gap.

The declared nonlinear A4 diagnostic also failed, reaching 59.458% routed
accuracy with a 38.187-point oracle gap. The outcome is therefore a failure of
the tested frozen-query/score family at flat full-data capacity, before
recursive maintenance becomes the limiting mechanism. In accordance with the
frozen protocol, the B/C recursive matrix, the test split, and seeds 1994/1995
were not run. There are intentionally no final-test results and no
test-selected follow-ups.

The completed run authenticated sealed inference authority
`08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db`,
froze 19,200 router-fit and 4,800 router-validation identities, and recorded
zero test images used. Its eight-hook BF16 R3 cache occupies 294,912,880 bytes
and was built on the RTX 4090 in 25.8 seconds at about 930 images/second. The
artifact-backed eight-task smoke passed all paired R1/R3 exact, U100, SVD0,
and SVD5 policies. The terminal reuse proof recreated zero nodes, reused all
12 B4 and all 12 B9 smoke nodes, executed zero new router optimizer steps, and
confirmed byte-identical before/after inference inventories with zero leaf or
inference-parent optimizer steps.

The Markdown/HTML report and machine-readable ledgers live under
`artifacts/imagenetr50/router_v1/runs/e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0/reports/`.
All 13 scheduled jobs completed with no failures; 40 focused vision tests and
the broader non-vision regression suite pass. A complete workflow rerun reused
the same immutable protocol and returned in 2.6 seconds. Any new router work
should be a separately frozen representation/loss experiment aimed at closing
the flat capacity gap; broader recursive merge sweeps are not justified by
this result.

This router-only follow-up used
`docs/imagenetr50_recursive_router_oracle_recovery_plan.pdf` with one deliberate
protocol change: R3 adapter-response routing was promoted from a deferred
rescue condition to a mandatory, paired member of the main experimental
matrix. The result compares R1 and R3 on the same images, teachers, inference
nodes, optimizer settings, and seed without activating R3 after observing
results. Because 15 of the 45 material source files differed from the sealed
run's code manifest at handoff commit `233a615`, the router protocol binds the
authenticated artifact bytes rather than claiming that the current checkout
exactly reproduces the old training source.

### Protocol and data boundary

- Add a separate content-addressed namespace under
  `artifacts/imagenetr50/router_v1/runs/<router-protocol-hash>/`. A frozen
  `protocol_link.json` binds the sealed run hash, the exact required inference
  policy and node hashes, the original model/dataset/transform identities, the
  handoff commit and source-drift audit, the current router-only material-code
  and environment manifests, the resolved router config, and every random
  seed. Do not call the original bootstrap or rewrite its protocol.
- Resolve I-U100 as `logt_retrain_union_r16`, I-SVD0 as
  `logt_svd_r16_repair000`, and I-SVD5 as
  `logt_svd_r16_repair005`. Phase 0 must authenticate every required artifact
  and snapshot, record missing inputs, and stop before a long job if any
  dependency is corrupt or would require inference retraining.
- Freeze one class-stratified, deterministic hash split of the 24,000 training
  identities into 19,200 router-fit and 4,800 router-validation images.
  Historical stage views may expose only arrived tasks. The 6,000 test images
  remain unavailable to configuration, architecture, epoch, rank, repair, or
  seed decisions and are evaluated only after the matrix is sealed.
- Keep label-aware logic inside `RouterTeacher` and diagnostic evaluation.
  The deployed `RouterQuery` contains image/feature identities only; it cannot
  accept labels, classes, task IDs, true nodes, or benchmark-specific utility.
  The ImageNet teacher maps a training label to the unique live node containing
  that class, while the interface also admits a future utility/NLL teacher.
- Reuse existing frozen-feature or evaluation-logit caches only when their full
  semantic key matches. Otherwise create router-specific caches. Cache
  construction may batch all images, but every training view must enforce its
  split and stage membership before returning rows.

### Router and promoted R3 design

- Recompute the fixed 128-float candidate descriptor from each actual inference
  node. Use seeded compact products over all 24 scaled QKV/fc1 LoRA updates,
  per-matrix energy statistics, and fixed generic level/count/span metadata.
  Never materialize a model-wide dense update and never concatenate descendant
  descriptors.
- Retain R0 as the node-local linear capacity floor. R1 is the PDF's rank-eight
  bilinear compatibility scorer over the normalized 768-float frozen ViT
  prelogit and 128-float node descriptor, with its query, descriptor, and bias
  residuals. R2 remains a conditional nonlinear capacity diagnostic only.
- Define R3 as a strict extension of R1, not an unrelated larger router. At the
  QKV and fc1 inputs of zero-based ViT blocks 0, 4, 7, and 11, cache only the
  frozen-base CLS activation. For each node/module, compact-SVD its scaled LoRA
  update as `U diag(s) V^T` and compute the normalized response
  `log1p(||diag(s) V^T h|| / (||h|| + eps))`. This equals the full update-response
  norm while being invariant to LoRA factor gauge and requiring only a
  rank-sized projection. The resulting eight-float response vector enters an
  additional node-local linear branch on top of R1.
- Bind the selected blocks, pooling rule, normalization, response dimension,
  and compact-SVD algorithm in the config before any router validation. The
  compact activation cache is expected to be about 352 MiB in BF16 for all
  30,000 images, rather than storing full token activations. Record its measured
  size and throughput in Phase 0.
- Store the canonical response kernels beside the inference-node descriptor and
  bind them to the inference adapter hash. Recompute them from the actual parent
  adapter after every inference merge/repair. Do not merge child response
  vectors. Count kernel bytes and candidate low-rank response projections as
  routing cost even though they are not full candidate-adapter forwards.
- Use the frozen old-frontier insertion rule from the PDF: fit all new-leaf
  positives against the old frontier log-sum-exp with margin one, plus exactly
  64 deterministic negative examples per live old node. Old router hashes must
  remain byte-identical.
- Implement P-EXACT as a nondeployable `logaddexp` functional oracle; P-U100 as
  a fresh fixed-size parent fitted on all stage-available router-fit rows with
  route cross-entropy plus unit-weight LSE distillation; P-SVD0 as compact
  rank-eight parameter merge with zero examples and optimizer steps; and
  P-SVD5 as that merge followed by deterministic five-percent positive repair
  plus balanced negatives. For R3, compact-merge the R1 interaction and
  source-mass-merge its response/residual vectors, but compute features from
  the actual parent response kernels.
- Resolve the PDF's ambiguous `flat causal` label as an explicit
  `flat_seen_data` control: independently refit the whole current frontier from
  scratch using only data visible by that stage. It is a causal-data capacity
  ceiling, not a state-preserving or scaling-compliant algorithm. All recursive
  P-* conditions use immutable old scorers and causal insertion.

### Main experimental matrix

The matrix is phase-gated rather than a broad Cartesian sweep. R0 has no
recursive slice, R2 remains conditional, Core+TSV/output-drift inference trees
remain excluded, and one-percent/fixed-budget repair remains follow-up work.
R1 and R3 are the two predeclared main architecture slices.

| ID | Inference | Router | Maintenance | Role |
|---|---|---|---|---|
| A0 | I-U100 final | centroid | existing | routed baseline |
| A1 | I-U100 final | R0 | flat full-data | capacity floor and smoke |
| A2 | I-U100 final | R1 | flat full-data | descriptor-only capacity |
| A3 | I-U100 final | R3 | flat full-data | mandatory adapter-response capacity |
| A4 | I-U100 final | R2 | flat full-data | only if neither A2 nor A3 reaches the validation gate |
| A5 | I-SVD5 final | R1 | flat full-data | cheap-node brittleness control |
| A6 | I-SVD5 final | R3 | flat full-data | paired adapter-response brittleness control |
| B0 | I-U100 all stages | R1 | flat_seen_data | non-scaling capacity ceiling |
| B1 | I-U100 all stages | R1 | P-EXACT | functional arithmetic oracle |
| B2 | I-U100 all stages | R1 | P-U100 | full-replay parent ceiling |
| B3 | I-U100 all stages | R1 | P-SVD0 | zero-example merge |
| B4 | I-U100 all stages | R1 | P-SVD5 | descriptor-only scalable condition |
| B5 | I-U100 all stages | R3 | flat_seen_data | adapter-response capacity ceiling |
| B6 | I-U100 all stages | R3 | P-EXACT | adapter-response functional oracle |
| B7 | I-U100 all stages | R3 | P-U100 | adapter-response parent ceiling |
| B8 | I-U100 all stages | R3 | P-SVD0 | adapter-response zero-example merge |
| B9 | I-U100 all stages | R3 | P-SVD5 | adapter-response scalable headline |
| C1 | I-SVD0 all stages | R1 | P-SVD5 | descriptor-only cheap-inference transfer |
| C2 | I-SVD5 all stages | R1 | P-SVD5 | descriptor-only repaired-inference transfer |
| C3 | I-SVD0 all stages | R3 | P-SVD5 | adapter-response cheap-inference transfer |
| C4 | I-SVD5 all stages | R3 | P-SVD5 | full cheap inference/router condition |

A0-A3 are mandatory. If neither R1 nor R3 comes within 1.0 percentage point of
the I-U100 validation oracle, run A4, emit the representation/capacity failure
report, and do not launch the recursive matrix. If either main architecture
opens the recursive gate, execute both complete R1 and R3 B slices; R3 is not
skipped merely because R1 passed first. Transfer rows C1-C4 run only after the
B results are internally coherent. Start with router seed 1993; if either
scalable condition reaches the preregistered target, repeat the paired R1/R3
headline and transfer conditions with two more frozen seeds rather than
replicating only the winner.

### Implementation sequence

1. **Audit and freeze.** Add the strict router config and immutable protocol,
   policy, node, snapshot, cache, and job records. Implement a read-only sealed
   run loader, authenticate the three inference hierarchies, freeze the router
   fit/validation manifest, inventory inference hashes, and emit
   `phase0_audit.json` plus `PHASE0.md`. Target 10-15 minutes; hard stop at 20.
2. **Build task-free inputs.** Add the generic teacher, descriptor builder,
   canonical response-kernel builder, prelogit/CLS cache, and leakage-checked
   historical data views. Extend shared compact math only where the existing
   QR/SVD implementation is genuinely reusable.
3. **Build scorer training and persistence.** Implement R0, R1, R2, and nested
   R3 behind one scorer interface; deterministic flat fitting; causal leaf
   insertion; P-EXACT/P-U100/P-SVD0/P-SVD5 parent creation; immutable
   safetensors state; atomic checkpoints; resume; and a router-only scheduler.
   Use a dedicated `router_cli` with config-driven `run`, `status`, and `report`
   commands so the completed primary workflow is not overloaded.
4. **Verify before real execution.** Run unit tests plus a synthetic eight-task
   linearly separable fixture, then an eight-task real-artifact smoke containing
   R0, paired R1/R3, all parent policies, one R3 response merge, and one repair.
   No matrix job may start until cache identity, causal immutability, exact LSE
   mass preservation, task-free API structure, finite R3 features, and
   zero-inference-step reuse all pass.
5. **Run the capacity gate.** Cache the full deterministic base features once,
   run A0-A3, use router validation only for capacity decisions, and run A4 only
   under its declared gate. Freeze selected epochs/settings before one sealed
   test pass. Then run A5-A6 without architecture retuning.
6. **Run recursive I-U100.** Execute B0-B9 in paired R1/R3 order with durable
   checkpoints and partial reports at stages 8, 16, 32, and 50. P-U100 must
   approach its same-architecture flat ceiling; P-SVD5 must recover at least
   95% of the P-U100 gain or finish within 1.0 point before transfer proceeds.
7. **Transfer and replicate.** Run C1-C4 with exactly the frozen architecture,
   feature, optimizer, and repair settings. If the first seed reaches target,
   add the two paired seeds. Add one-percent or fixed B=64/128 repair only after
   the primary matrix is complete and only as explicitly labeled follow-up.
8. **Report and prove reuse.** Emit the complete router manifests, stage/task
   metrics, selection and merge diagnostics, resources, lineage, Markdown and
   self-contained HTML reports, plus before/after inference hashes and zero
   inference optimizer-step evidence. Rerun one R1 and one R3 policy to prove
   content-addressed reuse and no duplicate completed stages.

### Tests, metrics, and acceptance

- Unit tests cover deterministic descriptors, no dense model-sized deltas,
  compact/dense sketch parity, compact SVD parity, R1 rank-16 exact child sums,
  R3 response parity with direct dense `delta @ h`, gauge invariance, selected
  module/pooling enforcement, response-zero equivalence to R1, safetensors
  round trips, and deterministic cache order/content identities.
- Structural tests prove `RouterQuery` has no label/task/class/oracle surface;
  only teachers and diagnostic evaluators receive truth; stage views reject
  future, validation-for-fit, and test identities; parent features come from
  the actual parent adapter; and R3 routing performs zero candidate adapted
  model forwards.
- Recursive tests cover unique class-to-live-node targets at every stage,
  byte-identical old scorers after leaf insertion, exact pre/post P-EXACT
  frontier probabilities, P-SVD0 zero examples/steps, exact P-SVD5 repair IDs,
  P-U100 historical-only replay, retired-node archival, final eight-node
  parity, resume idempotence, and unchanged inference hashes.
- Every condition reports routed classification, true-node oracle and gap,
  centroid and best-existing gain recovery, overall/conditional/top-two node
  selection, level/count recall, mean/p95 probability-mass error, collapsed
  frontier KL, LSE MSE, merge regret, paired per-image R3-minus-R1 correctness,
  and validation/test separation. Confidence intervals use paired image-level
  resampling; they do not drive configuration choices.
- Resource accounting separates learned router parameters, R3 response-kernel
  bytes, activation-cache bytes, examples/presentations, optimizer steps,
  merge/repair time, score FLOPs, response projections, one frozen-base forward,
  zero candidate adapted forwards, and the one selected-node adapted forward.
  Live state and score work must remain `O(log T)` with fixed layer count.
- The primary I-U100 mechanism pass remains at least 78.5% routed accuracy,
  at most a 1.0-point oracle gap, and at least 95% recovery from the strongest
  existing task-free result; a strong pass is at least 78.85%. I-SVD5 targets
  at least 68.3% and I-SVD0 about 63.85%, each within 1.0 point of its own
  oracle. Evaluate these thresholds separately for B4 and B9 rather than
  reporting a test-selected winner.
- Interpret the paired result explicitly: R3-only success means adapter-state
  response is necessary; success by both favors R1 unless R3 gives consistent
  robustness worth its measured cost; R1-only success is a negative result for
  adapter-dependent routing; failure by both is a query/score-family failure,
  not evidence against recursive consolidation. The experiment is incomplete
  without the full R3 slice, even if R1 reaches the oracle first.

## Completed Outcome — TRACE Log-t VAMP

The Revision-2 TRACE implementation and remote scientific session are complete.
The code lives in `src/apm/continual/trace/` and ran on RunPod against its fully
downloaded and authenticated public model snapshot. The
code pins the TreeLoRA 500-example archive and source revision,
reproduces manifest
`19fe258e74f5dba6408e9b498fb1b5e4c4dac16d4840363d142bd89a19e47ba2`,
retains Meta's immutable Llama model identity, and downloads it from a pinned
ungated public mirror whose required model, tokenizer, and configuration
objects match the canonical repository identities. TRACE verifies each
downloaded byte stream against a pinned size and SHA-256 manifest and
incorporates that manifest, the dataset, source, canonical model/tokenizer,
dependency lock, and complete code-tree identity into the run contract. The
dependency identity also binds the actual material package, Python, PyTorch,
and CUDA runtime versions. Every DAG job also carries that run-contract hash
directly.

Implemented experiment surfaces are:

- exact TreeLoRA prompt/answer collation, answer-only loss, fixed task epochs,
  40 deterministic 100-example arrivals, fresh base-relative rank-eight
  leaves, matched sequential/reference, sequential-40, joint-IID, and
  taskwise presentation plans;
- a pure capacity-two oldest-first hierarchy with 33 historical merges and the
  exact seven-node final topology; permanent leaf priority records and
  composable policy-sized repair reservoirs totaling 610 replay presentations
  at five percent;
- compact FP32 weighted SVD merging and pinned Core-Space/TSV algebra, direct
  rank-bounded reconstruction with each child's scale read from its immutable
  PEFT rank/alpha configuration, reusable
  replay-independent merge-cache identities, retained spectra/bases/cores,
  selected rank-up-to-16 precompression diagnostics, validation-NLL
  post-compression and post-repair diagnostics for every merge, and four
  fresh-parent calibration intervals;
- atomic immutable artifact directories, hash-chained work/job/evaluation
  ledgers, exact optimizer/scheduler/RNG/cursor checkpoints every 50 steps or
  two minutes, phase snapshots, and idempotent discovery that returns an
  already-published adapter without loading a model;
- answer-isolated prompt-NLL and frozen-base centroid routing, validation-only
  task-aware selection, target-aware diagnostic oracle selection, stable
  per-example generation seeds, row-local RNG streams inside fixed GPU
  generation batches, and resumable candidate caches reused by all routers;
- the 422-job primary DAG, two independent GPU subprocess workers, SVD/Core
  affinity, retry/restart recovery, measured job-family ETA, the 23h30 soft
  pause and 24-hour hard boundary, durable interim reporting, nonfatal webhook
  events, marker-guarded Pod deletion, and an independent marker-aware
  watchdog that never addresses a Network Volume deletion API;
- a pinned RunPod image/launcher for two RTX 4090s, a 150-GB persistent volume
  mounted at `/workspace`, a 50-GB container disk, persistent Hugging Face
  caches, a build-time zero-LoRA/PEFT round-trip smoke test, and public `run`,
  `resume`, `status`, `report`, and leaf-reusing `rebuild-policy` commands; and
- deterministic Markdown and self-contained HTML reports, score CSV/Parquet,
  the six registered merge-diagnostic comparisons, authenticated lineage SVG,
  primary cost/memory and sequential/joint-IID gap tables, archive-versus-live
  accounting, runtime/throughput/ETA state, a validation-only final-policy
  comparison that never reads test scores, and the required leaf-reuse
  acceptance record.

Verification currently passes 39 focused default tests, including mirror
identity and drift rejection, and the complete repository regression gate
passes 757 tests with 275 resource skips. It also passes the real pinned-data
integration test
(4,000 train, 741 validation, 781 test examples), exact
interrupted/uninterrupted adapter parity, dense/compact merge parity, Core/TSV
reference algebra, artifact and ledger recovery, report determinism, and the
focused legacy TinyWorlds regressions affected by the shared artifact-module
migration. Policy-only rebuilds use the same 24-hour lifecycle as the primary
run and publish leaf-reuse acceptance only after their added DAG is complete.
Both registered Core-scale controls, their reuse gates, all plots, and the final
authenticated report are complete. Any new implementation change must rerun the
appropriate local regression gate; additional science is follow-up replication
or routing work, not unfinished work in this run.

Deployment on 2026-08-18 used direct SSH setup on secure-cloud Pod
`yyqmyhmzei2k0z`: two RTX 4090s in `EU-RO-1` at $1.48/hour total, the public
RunPod PyTorch 2.8/CUDA 12.8 image, a 50-GB container disk, and persistent
150-GB Network Volume `oybty7q8vt` mounted at `/workspace`. No custom registry,
saved template, or Hugging Face token is required. Pod deletion is exclusively
authorized by the durable safe-to-terminate marker; an independent wall-clock
deletion is not part of the launcher.

The initial Pod `mwog19gmdsbtox` authenticated the public snapshot, created run
contract `0b5c71f9623f757e3198a7b0b70a02269fb9df4853c7a2e353fe8cdc819a6388`,
and reached 13 complete jobs with no failures. It was intentionally terminated
after the full regression gate exposed a creation-time wall-clock deletion flag
that violated the marker-only cleanup contract. RunPod does not expose an
in-place cancellation operation for that field. Its durable diagnostic state
remains on the volume, but that superseded run must not supply scientific
results. The corrected, fully tested source on the replacement Pod passed its
real PyTorch/PEFT adapter round-trip, reauthenticated the persistent model
cache, and created fresh run contract
`c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5`.
Its primary 422-job ledger completed with no remaining failures, pauses, or
pending work. The original Pod then published its authoritative termination
marker and deleted itself; after attaching the same Network Volume to secure
replacement Pod `csujshwtro5i0f`, an authenticated status check and independent
SHA-256 verification confirmed the complete ledger and both durable primary
reports. During the controls, fifteen-minute health checks covered
Pod/coordinator liveness, durable ledger progress, GPU utilization/temperature,
storage, and new errors. No marker-aware watchdog was used while new jobs were
being registered because the preceding marker became stale as soon as the
ledger expanded; each coordinator overwrote it only after its expanded ledger
quiesced and then performed marker-authorized self-termination.
The repair-free Core scale-0.5 control completed all 70 jobs without failure.
Its fresh marker records 492 complete jobs and zero jobs in every other state;
the marker-bound Markdown and HTML report hashes independently match the
durable files. Pod `csujshwtro5i0f` then disappeared from RunPod exactly as its
marker-authorized self-termination contract required. That marker is archived
as `state/sessions/core05-repair000-SAFE_TO_TERMINATE.json`. Secure replacement
Pod `zsopdi1mcogokw` then completed the preregistered Core scale-0.5/10%-repair
control without a failure. The final marker records 562 complete jobs and zero
jobs in every other state; its SHA-256 is
`934d9f4d9f9b3b5269194c4c58f98d8527e1c01ea1facde9099c6b4633396357`.
The marker-bound Markdown and HTML reports independently match hashes
`f43d35570993e5c40773165ca16147014343581777ce38d5555e26e765dcc3c3`
and `526aab21c7d39f7cc9da9d2ff23ce4004d0801c041b139fc19e7b940e1ba5589`.
Both experiment Pods self-terminated after publishing their reports. A
short-lived L4 verifier rechecked and retrieved the final bundle and was then
deleted; no RunPod compute remains. The 150-GB evidence volume is retained at
$0.015/hour, and the account balance after cleanup was $9.40. A hash-verified,
ignored 1.7-MB copy of the report bundle and all safety markers is preserved at
`results/trace-logt-vamp/c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5/final/`.
The only recovered runtime fault was the pinned `datasets==2.21.0` SARI loader's
need for explicit custom-code trust. One exhausted evaluation was moved through
an authenticated `FAILED -> PAUSED` transition for a single operator-authorized
retry; the orphaned worker checkpointed at a safe boundary, and the coordinator
resumed from durable candidate state with
`HF_DATASETS_TRUST_REMOTE_CODE=1`. The retry and every subsequent job completed.
The working-tree fix now passes `trust_remote_code=True` at both SARI load sites
and has focused coverage for aggregate and per-example scoring.

The final science resolves the earlier Core-scale question without evidence of
an implementation bug. Five independent FP64 comparisons against the pinned
upstream Core-Space algebra already matched within 2--7 ppm. The repair-free
scale-0.5 control now supplies the causal protocol check: relative to scale 0.3,
mean merge damage fell from 0.754 to 0.267 answer-NLL, prompt-NLL OP rose from
19.355 to 19.906, forgetting fell from 5.726 to 1.276, and negative-only BWT
improved from -6.101 to -2.151. Thus recursive `0.3^depth` attenuation was a
real source of damage, but not the whole SVD/Core gap. Both controls reused 100%
of leaf training steps with unchanged leaf hashes, so this comparison changes
only consolidation policy.

Rank-eight SVD remains intrinsically gentler. Without repair it retains 93.0%
mean weighted spectral energy versus Core's 89.0%, has -0.001 mean merge damage
versus scale-0.5 Core's 0.267, and reaches 21.236 prompt-NLL OP versus 19.906.
It beats scale-0.5 Core on five of eight final tasks, ties two, and loses one;
its answer-oracle advantage is much larger (41.410 versus 29.227). The deepest
unrepaired Core node still adds 2.707 NLL while the corresponding SVD level is
slightly beneficial on average. SVD's result is therefore mathematically and
empirically coherent: its rank-eight reconstruction preserves the weighted
mean far better than recursively scaled Core parents.

Core scale-0.5/10%-repair demonstrates that replay can repair the merge itself.
Across 33 merges, 1,220 replay examples and 162 optimizer steps reduce mean
post-repair damage to -0.010 NLL, including negative mean damage at levels three
and four. Relative to repair-free scale 0.5, answer-oracle OP rises by 8.799,
task-aware OP by 8.780, and frozen-centroid OP by 8.150. Prompt-NLL OP rises only
0.736 to 20.642, however, and forgetting worsens from 1.276 to 1.918. Repair is
creating usable competence that the primary router largely fails to select.

Routing is the dominant deployment bottleneck. For scale-0.5/10%-repair, the
final answer oracle exceeds prompt NLL by 17.384 OP; the corresponding gaps are
20.174 for repair-free SVD and 23.110 for repaired SVD. Under the registered
primary prompt-NLL router, the best VAMP condition is scale-0.3/5%-repair at
23.430 OP, still below sequential LoRA at 34.096 and joint-IID at 47.253. The
secondary but task-free frozen-centroid router reaches 34.217 with repaired SVD,
0.120 above the sequential reference. This makes router calibration and route
selection, rather than further merge tuning alone, the highest-value next line
of work.

These are one-seed, one-order results across eight heterogeneous tasks. The
leaf-reuse controls isolate merge policy cleanly, but they are not independent
stochastic replications, and the repair budgets are unmatched (Core 10% versus
SVD 5%). Next work should instrument route selections and confusion, test a
calibrated/frozen-centroid deployment policy, replicate scale 0.3 versus 0.5 and
SVD across multiple seeds/orders, and compare Core and SVD at matched repair
budgets before promoting a new default.

The scientific handoff is versioned under
`docs/experiments/trace-logt-vamp/`. It contains the hash-verified final report
bundle, run contract and job ledger, all readable coordinator/job logs, state,
non-embedding manifests, registered control policies, and every raw candidate
generation. A deterministic index, one-record-per-file sample, sampling helper,
and independent-review prompt make the evidence navigable without scanning the
entire corpus. The 532 raw generation JSONL files are stored through Git LFS;
checkpoints, adapter/merge tensors, embedding caches, package/model caches,
source copies, and the superseded run remain only on the retained evidence
volume because they are resumability inputs rather than reviewer evidence.

The CPU-only task-known provenance follow-up is also complete. At every stage,
its fixed lookup selects the active hierarchy node with the greatest coverage
of the known task's five arrivals, breaking ties by node purity and then
recency; no prompt, answer, validation metric, or test metric participates in
the decision. The analysis hash-verifies all 1,798 evidence files and the 432
relevant candidate files, then exactly reconstructs the existing prompt-NLL,
task-aware, and answer-oracle aggregates in 1,296 checks with zero maximum
error. Across all six VAMP conditions, repaired SVD is the strongest provenance
result at 38.340 OP. It is 0.159 points above the existing validation-selected
task-aware lookup and selects the same final node for six of eight tasks. Its
lower 1.878 forgetting value is not evidence of cleaner retention: provenance
routing also lowers the mean diagonal score by 3.284 points, so cross-router
comparison should use final OP and per-task final scores. The deterministic
Markdown, self-contained HTML, plots, complete route audit, score tables, and
manifest live under
`docs/experiments/trace-logt-vamp/followups/task-known-provenance/`. This closes
the simple task-known control; the next high-value science remains calibrated
task-free routing, matched repair budgets, and multi-seed/order replication.

## Completed Outcome — Standalone TinyWorlds Benchmark Repository

TinyWorlds distribution has been extracted into an independent Git repository
at `../tinyworlds`, root commit `32723b8`. Its public catalog contains only the
concrete `nouns-v2` contract; the catalog, resource layout, and versioning
policy support future benchmark families and representation flavors without
advertising empty configurations.

The repository is now a deterministic recipe over immutable, checksummed
TinyStories and TinyStories-8M files rather than a second data host. It tracks
no Parquet, Arrow, or Git-LFS objects. The old 96-shard, 1.17-GiB materialized
root was amended away, its reflog expired, and all 96 local LFS objects pruned;
the complete checkout is 2.9 MiB. Consumers use `tinyworlds.load_dataset()` to
download the pinned upstream bytes once and prepare a content-addressed local
cache. An optional Hub mirror may later accelerate this without becoming the
benchmark's authority.

The scientific identity is serialization-independent logical manifest
`ad690aeca99c87b60b5719c130e18d78b1beb1f2d5c28f5c679a5b8ab77f1ea5`.
It binds all 2,644,609 rows in six logical splits through canonical story-ID,
token-boundary, and token-stream hashes. A clean test starting only from the
pinned upstream corpus and tokenizer processed all 2,745,125 raw documents,
reproduced the authenticated 2,745,124-story store, token store, and every
partition index, emitted a fresh 96-file cache, and passed full byte, schema,
row, assignment, and logical validation. The new cache checksum differed from
the historical Parquet release as intended while its logical identity matched.

The standalone code covers clean upstream reconstruction and the authenticated
RPA fast-export path. It preserves source/tokenizer revisions and digests,
reviewed noun decisions, construction provenance, normalization, deduplication,
matching, assignment, deterministic holdout/probe selection, tokenization,
cache materialization, and fail-closed validation. The repository also includes
a parseable Hugging Face dataset card, zero-issue Croissant dependency metadata,
benchmark protocol, data card, reproduction/versioning docs, dual licenses,
citation metadata, CI, and package/CLI APIs. Sixteen lightweight tests, the full
release/streaming test, Ruff, lock validation, sdist/wheel builds, and a wheel
import smoke test pass.

The public repository is live at
[`dmacd/tinyworlds`](https://github.com/dmacd/tinyworlds), with `master` as its
only and default branch. No Hub mirror, DOI, or package-index entry has been
created. Existing RPA TinyWorlds research code and authenticated artifacts
remain in place so the dirty active worktree and completed experiment replays
are not broken. Once RPA is made to consume the standalone recipe, redundant
generation-only surfaces can be removed in a separate intentional migration.

## Completed Outcome — TinyWorlds Nouns-v2 Bounded Addressing

The frozen final-checkpoint addressing study is implemented and complete. Its
no-options GPU-0 runner, independent v1 contracts, load-only canonical
authentication, five frozen key schemes, physically compact batched LoRA edge
banks, compact EBT-H, strict resumable ledgers, deterministic analysis, CSV/SVG
exports, Graphviz renders, and separate Markdown/self-contained HTML reports
live under `results/language_cl/tinyworlds-nouns-v2/addressing-study/`. The
preregistered method and release gates are documented in
`docs/TINYWORLDS_NOUNS_V2_ADDRESSING_EXPERIMENT_PLAN.md`; durable compact and
residual semantics are recorded in `DESIGN.md`.

The study authenticated partition
`210c4e2d067077fe774782024a594ade7e7472a986d554f186453549cf910f1b`,
base parameters
`fff309bfbfcee8d59c5c3fc04152cc37be2142201f3bf9116b7b024e81a24f3c`,
final VAMP tensors
`97414ac3d8656ab083b2e570a4162dc69b024f90cf819b80b1cab94213553e63`,
all 900 registered probes, all 4,440 official validation cases, and every
canonical nouns-v2 result ledger before model work. Frozen key artifact
`9036d673332303a6f949b853b3f4e4d90e89b2340901b27db9a8c4d90e53ce43`
binds tensor
`d7d63d097e70471452b72d89b96ccf031c979701e18ec72de6f1d3d33ec4b51f`.
Retrieval and EBT contracts are respectively
`0b9481d422175fd5492bd2d0098237e4ab0b544d04da2eba387931c6c8201cdb`
and
`040eff168d2e6a9b02d359ed7eca43107ea3a075bcaec972bcb09d102213a4bc`.
The final ledgers contain exactly 22,200 retrieval, 48,840 EBT, and 69
shape-timing rows; their file hashes are respectively
`ae63beed46a02f361f00c1e5d35e27ac30bbb899914b213a9b001895dda7bf08`,
`d15b51d4c83fddb49d54c51fa540a7e5e9374e6ce284ea92494a929b81ecb9dd`,
and
`ee10f1f977f1d41130cf1a1a6524a5735b85896ce4aa77fc235acf80302715d6`.

Compact top-8 passed both preregistered non-inferiority gates. Versus dense
all-node EBT-H, its suffix story NLL was 1.58215 versus 1.58184 (+0.00030,
within the +0.02 margin) and route accuracy was 64.64% versus 64.50% (+0.14
percentage point rather than a loss). Top-4 reached 1.58965 NLL and 56.76%
accuracy. Canonical retrieval recall@4/@8 was 64.08%/78.09%. Mean gathered edge
counts were 6.19 for top-4 and 10.47 for top-8 versus 24 dense edges; mean active
LoRA-edge evaluations fell from 58,350 dense to 15,114/25,531. This operation
reduction did not translate into a material kernel-speed win: synchronized warm
latency per eight rows was 0.3304 s dense, 0.3269 s top-4, and 0.3382 s top-8
(24.21, 24.47, and 23.65 examples/s).

Among frozen keys, midpoint content centroids ranked first on recall@8 at
79.55%, a paired +1.46 percentage-point improvement over canonical keys (seed-0
10,000-bootstrap 95% interval [+0.81, +2.12]); midpoint content/residual
centroids reached 79.32% ([+0.11, +2.36]). Content prototypes fell to 75.07%
and fused residual prototypes to 73.65%. At compact top-8, the corresponding
story NLL values were 1.58161, 1.58204, 1.58296, and 1.58417 versus 1.58215
canonical; every paired NLL interval included zero. The report therefore shows
that midpoint matching helps centroid retrieval, while nearest-prototype and
residual fusion do not provide a reliable suffix-quality improvement here.

The clean evaluation took 4,308.3 seconds end to end inside the evaluation
phase and 4,429.7 seconds for the complete runner. Allocator peak was 5.23 GiB
under the frozen 12 GiB gate. Real-checkpoint compact/dense parity had maximum
absolute drift 0.000513 under the 0.001 FP32 tolerance with identical selected
nodes. All 69 observed prefix/capacity shapes have one cold compile and five
synchronized warm samples. The visually audited plots have readable labels,
separate incomparable cost units, accessible SVG metadata, and complete
25-node/24-edge top-4/top-8 graphs.

Publication manifest identity is
`aab391c9cb1612018a551c5b097dcf98d9cdbe9a53430c086f187a652f63660c`;
Markdown and HTML file hashes are
`8a0638d4e9f517f84954294b4c45f942747adefbccbd296d2ec91d0906e602e5`
and
`4843053f40c97a1beb80f676e1d67436273e40627d56adacb7245236c5116bd4`.
After the authorized canonical stagewise-report extension, a current-source
no-compute replay strict-loaded every row, preserved both immutable contracts
and every raw ledger hash, reproduced the refreshed report tree twice
byte-for-byte, rechecked the protected nouns-v1/v2 hashes, and completed in
122.6 seconds. The derived analysis/report/manifest identities changed only to
record the new canonical report and run-manifest provenance. The focused CPU
suite passes 9 tests, the real-GPU parity/allocator smoke passes, all 6 opt-in
real-source tests pass, and the clean default suite passes 676 tests with 274
resource skips and 18 marker deselections. At original study publication, no
canonical nouns-v1/v2 artifact changed.

## Completed Outcome — Canonical-Centroid Compact Stagewise Curve

The nouns-v2 report now includes physically compact top-eight EBT-H using the
canonical stored full-probe centroids in both the true-suffix-loss comparison
and task-free accuracy-versus-graph-growth plot. Only this added router was
evaluated: no base, adapter, or VAMP checkpoint was retrained, and the original
`stagewise-cl.jsonl` remains unchanged. Its independent v1 contract is
`b24ba9f6189c5dc7d87948e7385e43fefbeb41e8e6a1b0f0a6fea2d9b01a564b`;
the complete 72,256-row ledger hash is
`a6aefd8b1c0ff82b01515e50f92fb3197fc18de28257cca77937a8474dc0f584`.

At the final 25-node stage, compact top-eight reached 1.58212 story-weighted
suffix NLL, 1.61788 token-weighted suffix NLL, and 64.66% route accuracy. Dense
all-node EBT-H reached 1.58179, 1.61845, and 64.55%, respectively. The compact
curve's mean/max routing forgetting is +0.0115/+0.0272, versus
+0.0118/+0.0278 dense. Across all stages it gathered 8.80 edges per case on
average and used only the 4/8/12/16 physical-capacity buckets; the final-stage
mean was 10.47 gathered edges. The regenerated Markdown/HTML report and both
SVGs passed visual inspection. Their hashes are respectively
`b88fcbb1c6720c690dca36823c9766ba907db82be379b8f1abd9a66516551daf`,
`2c504cb698ae47509fa075fa06a895c14f5d81c4530aa55f992d67672a54a76d`,
`5a38e2191577dabc5268a0009363ea987884f6dac6812b82e0176ab70fd4680a`,
and `d34faf08e3e8ad3598f9f453361ceb71676b76364db2d6526731807a0c6bbd8d`.
A cross-run audit against the separately batched final-checkpoint study
preserved oracle NLL exactly; 3 of 4,440 hard routes differed at FP32
near-boundary decisions, so exact row equality across different padding and
compact-capacity batch shapes is not claimed. The no-compute exact-resume run
strict-loaded every checkpoint and ledger, enforced a 1.60 GiB live allocator
peak against the 12 GiB limit, and regenerated the report successfully. All 27
focused CPU tests pass, the updated opt-in real-artifact replay passes, and the
clean default suite passes 677 tests with 274 resource skips and 18 marker
deselections.

## Active Outcome — TinyWorlds Nouns-v2 Log-t Temporal Consolidation

The fixed temporal-consolidation experiment is implemented under the
content-addressed contract
`3f4ef4a10fd471b418a32a8f7b45431602c1f6abc080c19a7822ea2c2dd839b4`.
It selects eight immutable 512-story shards for every noun (192 arrivals),
evaluates blocked and round-robin orders, performs the prescribed capacity-two
oldest-first carry schedule, and finishes each order with 183 merges and nine
live chunks. Four finite epochs, standalone base-relative LoRAs, sequential
LoRA, the independent noun bank, and IID LoRA/full-model controls are all
encoded in the immutable contract. Canonical authentication currently covers
all 4,440 validation cases, all 900 root/task probes, the selected base, all 24
VAMP stages, every existing result ledger, and 168 protected nouns-v1/v2 files.

The implementation includes exact optimizer/RNG/update resume, finite
story-bounded padded batches, prefix-isolated exhaustive routing, evaluator-only
suffix oracles, bounded per-stage evaluation ledgers, merge/source/sentinel
distortion and lineage telescoping, shape-bucket timing, the 12 GiB allocator
gate, and canonical-artifact before/after hashes. The no-options GPU-zero runner
prints phased progress and ETAs, starts a GET-only loopback dashboard, and
publishes Markdown, self-contained HTML, a frozen dashboard, CSVs, accessible
Matplotlib SVGs, and compact/full Graphviz lineage diagrams. Raw resumable
ledgers remain in the contract's persistent `.work-v1` tree.

The focused CPU suite passes, including exact LoRA and full-model interruption
parity, schedule/coverage/isolation, malformed and cross-contract ledger
rejection, dashboard/report structure, timing validation, and byte-identical
plot/report generation. Read-only authentication against the real artifacts has
also passed and reproduced the contract above without changing the protected
hash set. The loopback-only HTTP gate and clean default suite pass. The bounded
real-data GPU smoke compiled the production 32-by-256 training and routing
kernels, produced byte-identical adapter tensors across direct and interrupted/
resumed trajectories, evaluated an isolated midpoint case, and measured a
6,664,041,728-byte peak under the 12 GiB gate. Equivalent short training jobs
now reuse one pure compiled executable per model architecture; after the single
18.46-second cold path, the interrupted/resumed eight-update trajectory took
0.41 seconds.

The complete no-options run has atomically published all shared, independent,
IID, sequential, and merge training artifacts for both orders: all 366 merge
jobs and both final nine-chunk live stacks are complete. Its first timing pass
stopped safely after 89 of 208 shapes when the process retained every distinct
JIT executable and exhausted the GPU allocator. Compile-cache eviction alone
did not release the default allocator's retained blocks, so production timing
now launches each pending shape in a fresh process from a deterministic,
content-hashed 102.89 MiB input bundle. Every worker verifies the bundle and
base, records one cold call, five synchronized warm calls, and its allocator
peak, appends exactly one chained row, and exits to release the device pool.
The resume path also treats callbacks replayed while strict-loading already
published sub-jobs as idempotent without weakening the recorder's default
fail-closed monotonicity check.

Eight rows by eight nodes at prefix width 512 does not fit the frozen 12 GiB
gate even in a fresh process. Exhaustive routing therefore selects physical
rows from `{8, 4, 2, 1}` using a fixed 20,736 row-node-token work budget; this
execution-only microbatching preserves every logical prefix score and suffix
result and reports physical rows explicitly. The formerly failing width-512
shape completed with four rows and a 6,887,178,240-byte peak. Isolated shapes
102–104 also completed, including an eight-row width-480 shape at
12,651,053,056 bytes under the gate. The focused CPU suite now passes 22 tests
with one loopback-dependent skip.

The first isolated resume reached timing shape 114 before the eight-row,
nine-node, width-320 shape crossed the same transient-autotuning boundary. The
work budget was consequently tightened from 24,576 to 20,736: the immediately
preceding eight-row shape has product 20,736 and a measured 8,889,827,328-byte
peak. The exact previously failing width-320 shape now completes with four rows
and a 4,931,577,856-byte peak. The focused CPU suite still passes 22 tests with
one loopback-dependent skip. The corrected no-options run resumes from the same
persistent `.work-v1` contract directory and serves its verified loopback
dashboard at `http://127.0.0.1:8765/`; after restart it advanced through shape
117 of 208. The run subsequently completed all 208 timing shapes, both
distortion audits, both sentinel/macro evaluations, final controls,
publication/regeneration, and the final immutability gate.

The foreground runner was subsequently reaped with its tool-managed terminal,
which also removed the in-process HTTP server while timing shape 120 was in
flight. A no-options, read-only dashboard entry point now serves the existing
authenticated disk projection independently of the GPU runner. Both processes
are launched as restartable user services, keeping port 8765 available across
runner exits without giving the dashboard ownership of experiment state.

The independent service exposed a previously unexecuted live-HTML defect: a
Python string escape emitted a literal newline inside the JavaScript event-log
separator, causing the browser to stop before rendering the first authenticated
snapshot even though the API returned in under two milliseconds. The escape is
now emitted as valid JavaScript and covered by the dashboard surface test. The
runner had meanwhile advanced into blocked evaluation, but its single parent
process retained a 16.9 GiB GPU pool. That device footprint is larger than the
12 GiB operational target, although it is not interchangeable with JAX's
active-allocation peak used by the formal gate. It was stopped at a resumable
ledger boundary; the evaluation-memory isolation described below resolved the
issue before compute resumed.

The bounded resume now selects JAX's telemetry-preserving CUDA asynchronous
allocator for the parent process while leaving isolated timing workers on the
default allocator. Distortion and evaluation callbacks enforce the 12 GiB peak
at every streamed progress boundary rather than waiting until publication. The
production resume crossed the prior macro-blocked high-water mark from 102,168
to at least 102,933 rows while remaining active. Its process footprint rose to
2.74 GiB during the first cold resumed batches and then fell to 1.58 GiB while
sustaining evaluation, rather than retaining the prior 16.9 GiB
default-allocator pool.

The live client now labels snapshots as live or paused/stale from their latest
authenticated event time, so an externally stopped runner cannot look active
indefinitely. Initial event hydration is bounded to the newest 500 rows instead
of downloading the full 34,000-row progress history. The separate ResNet tuning
process released GPU 0, and the restartable temporal service is active again;
the dashboard health endpoint reports `blocked evaluation` and its snapshot is
current. The manual pause did reveal that no GPU-availability watcher had been
installed, so transitions out of an intentional resource pause currently still
require an explicit service start.

Resume callbacks at an unchanged running high-water mark are also idempotent
now. This prevents strict replay from appending hundreds of duplicate progress
events or temporarily erasing the last real metrics before it reaches pending
work; failed-to-running and genuinely advancing transitions remain visible.

The experiment completed at 2026-08-17 14:53 PDT. Every dashboard job is
complete, the runner exited cleanly, and the independent read-only dashboard
continues to serve the frozen state. The published manifest is
`15f3ee2a5a2c5054b158ba62d7a0d1b9fcaa22e40634a73c9cbffceca5888bcb`;
an independent check reproduces that identity and all 32 artifact hashes. The
run took 42.21 hours end to end and peaked at 11.78 GiB of JAX allocator use
under the fixed 12 GiB gate. The focused suite passes 22 tests with one expected
loopback skip, and the clean default suite passes 699 tests with 275 registered
resource skips and 20 marker deselections.

At arrival 192 on all 4,440 validation stories, routed log-t improves frozen-
base story NLL from 1.63759 to 1.57980 in blocked order and 1.58858 in
round-robin order. Token NLL is 1.61432/1.62206 versus 1.66357 for base, while
token accuracy moves only from 60.63% to 60.89%/60.79%. Sequential LoRA is
worse than base in both orders at 1.73303/1.69788 story NLL. Mean final
forgetting is 0.00606/0.00270 for log-t versus 0.14980/0.05978 for sequential
LoRA. This supports the retention hypothesis under both stream structures.

Round-robin minus blocked routed log-t is +0.008785 story NLL with a
deterministic noun-stratified 95% interval of [+0.006776, +0.010677]; token
accuracy changes by -0.122 percentage points. The final suffix oracles remain
close at 1.53113 blocked and 1.53345 round robin, so most order sensitivity lies
in prefix addressing rather than destroyed bank quality. Prefix/oracle
agreement is 38.6% blocked and 17.6% round robin, and prefix entropy remains
near the ten-way maximum. Routing captures 54.3% and 47.1% of the respective
base-to-oracle gains. Frozen midpoint-prefix NLL is therefore the clear
bottleneck.

The nine-adapter deployed log-t bank is effectively tied with the practical
24-adapter independent-noun bank: its story NLL is 0.00458 better in blocked
order and 0.00421 worse in round-robin order. It remains behind joint IID LoRA
(1.55432 story NLL) and substantially behind the offline IID full model
(1.39903). The final result is a positive bounded-memory/retention result, not
an absolute-quality win over offline replay: temporal consolidation preserves
useful specialization with logarithmic live state, but improved content
addressing is needed to realize the stored quality.

### Completed addendum — full-story final-bank routing

The read-only full-story routing diagnostic is complete under independent
contract
`67657f3a6baf0e529b6ac668e3e2876269ce18c53023c4f31682627c4fb1b253`.
It authenticates the temporal parent manifest and all 32 parent artifact
hashes, strict-loads the two final nine-interval log-t banks and the 24-adapter
independent-noun bank, and joins their exact 13,320 final validation rows. No
training or merge was rerun, and the parent report, analysis, manifest, and all
canonical nouns artifacts retained their prior hashes.

Using the entire story raises blocked log-t noun-support accuracy from 71.71%
to 84.03% (+12.32 pp, paired noun-stratified 95% interval +11.08 to +13.56 pp)
and lowers story-weighted suffix NLL from 1.57980 to 1.54457 (-0.03523,
95% interval -0.03767 to -0.03286). Round-robin support rises from 81.76% to
94.39% (+12.64 pp, +11.51 to +13.78 pp), while suffix NLL falls from 1.58858
to 1.54588 (-0.04270, -0.04490 to -0.04063). Independent-bank exact noun
routing rises from 70.68% to 79.75% (+9.08 pp, +7.82 to +10.34 pp), while
suffix NLL falls from 1.58438 to 1.54004 (-0.04434, -0.04713 to -0.04163).

These full-story selections recover 72.4%, 77.4%, and 66.5% of the respective
midpoint-to-suffix-oracle story-NLL gaps. Their self-selected whole-story NLLs
are 1.49926, 1.50921, and 1.48287; selecting the same candidates from only the
midpoint yields 1.51297, 1.52478, and 1.49939. The evidence therefore supports
the proposed explanation: weak midpoint routing cues account for a large,
statistically clear portion of apparent suffix loss. This is diagnostic rather
than deployable quality because full-story selection reads the suffix whose
loss is reported; it remains above the evaluator-only suffix oracle and does
not show that the held-out routing problem is solved.

All 111 stories whose midpoint prefix crosses the canonical 256-transition
window boundary were directly rescored. The deterministic audit added near
ties and one minimum-margin short story per noun, for 190 unique stories and
570 bank/story rows. Direct and reconstructed short scores differ by at most
`4.01e-7`, with zero route mismatches; the smallest unaudited top-two margin is
`0.000200103`, above twice the fixed `1e-4` tolerance. The direct and derived
ledger hashes are
`dfcfb669efb87cb04f65c7ff549a22cd3e80cb38c19addb55ee2c5e37481a5c4`
and
`68f8f6f127593071f03220b9b6efc6bb7686d90f93092b8a76b056e39f84c1df`.
Peak JAX allocator use was 3.35 GiB under the 12 GiB gate. The no-options
GPU-zero runner resumed its evidence after a deliberately failed deterministic
publication check, then regenerated the corrected Markdown/HTML/SVG bundle
byte-identically. The publication manifest identity is
`3fa2931e448812d72f8c118065f030b46385121736db3f1cb5cb9f83b89c469d`;
the Markdown, HTML, and SVG file hashes are respectively
`74ee79e34896dc126a747c2159cd322595e46a07ba79990e16f2c9dd1402a0b0`,
`ebde5ba62a6c010a0fdb161090b15ed65112857ae02c19639d81634f27f2556b`,
and `abfe6e3a2094a6929d8dc757592b7588ce15c50eeb55aa121d7f8bb70d66ce04`.
The final exact-resume replay passed without changing the publication snapshot,
the opt-in real-source GPU parity/allocator test passed, and the clean default
suite passes 704 tests with 275 resource skips and 21 marker deselections.

### Completed addendum — joint-IID LoRA rank sweep

The fixed rank-4/8/16/32 joint-IID sweep completed under independent contract
`e87a835334a64c22b634a5e51f300cf5ad5fd529bd9fdcdf2268842fbd3df301`.
It strict-loaded the original rank-8 adapter and 4,440-row ledger plus the
original joint-IID full-model ledger, then trained only ranks 4, 16, and 32.
Every new adapter inherited the canonical rank-8 batch and random namespaces,
four-epoch 15,024-update schedule, and unit LoRA scale (`alpha = rank`). Exact
story order, suffix masks, and all 476,035 suffix targets match the parent
evaluation; rank-shaped base-path NLL drift is zero.

The full model remains substantially better: its story/token NLL is
`1.399026/1.452044`, versus `1.553880/1.590484` at rank 4,
`1.554322/1.590877` at rank 8, `1.559201/1.595611` at rank 16, and
`1.569790/1.605972` at rank 32. Rank 4 differs from rank 8 by only
`-0.000443` story NLL (95% paired noun-stratified bootstrap interval
`[-0.001118, +0.000225]`), while ranks 16 and 32 are significantly worse by
`+0.004878 [+0.004174, +0.005592]` and
`+0.015467 [+0.014554, +0.016380]`. Thus extra rank does not explain or close
the full-model gap; quality degrades above rank 8 under the matched schedule.

The three new 4,440-row ledgers and the reused rank-8 ledger total 17,760 exact
evaluation rows. Peak allocation was 7.78 GiB under the 12 GiB gate, and the
authenticated end-to-end run took 201.1 minutes. Exact-resume replay,
byte-identical report regeneration, and parent immutability checks passed. The
publication manifest is
`bf8b74cdb996679adf501234aaf4f540ba92cf599ac44590960d47ffc83676bb`;
the result identity is
`df775fed77517fc3002c1f215758729c52393a72903e201c3ac5e1a7057fc121`.
The final focused suite passes 34 tests with one resource skip, the opt-in
real-GPU rank/parity/allocator gate passes, and the clean default suite passes
711 tests with 275 resource skips and 21 marker deselections.

### Completed addendum — joint-IID LoRA with a trainable tied embedding

The fixed rank-8/rank-32 projection-LoRA plus tied-embedding study completed
under independent contract
`b5e1d49866bcaa06fd840fd055cf6d658ace1bacb4433b04333549ac543372ae`.
It uses the rank-sweep's exact 98,304-story population, batch/random namespace,
four-epoch schedule, and 4,440-story/476,035-target suffix evaluation. All six
LoRA projections in every transformer block and the single tied token
input/output matrix are trainable; position embeddings, layer norms, biases,
and original attention/MLP kernels remain frozen. The joint globally clipped
loss uses AdamW at `1e-3` for LoRA and `5e-5` for the embedding.

Training the tied embedding removes the projection-only LoRA gap. Rank 8
achieves story/token NLL `1.382183/1.438839`, versus
`1.554322/1.590877` for projection-only rank 8 and
`1.399026/1.452044` for the joint-IID full model. Its story-NLL improvement
over projection-only LoRA is `-0.172139` (paired noun-stratified 95% bootstrap
interval `[-0.177414, -0.166913]`), and it beats the full model by `-0.016843`
`[-0.018336, -0.015400]`. Rank 32 achieves
`1.399205/1.455794`, improving over projection-only rank 32 by `-0.170585`
`[-0.175792, -0.165459]`; its story NLL is statistically tied with the full
model at `+0.000179 [-0.001360, +0.001672]`, although its token NLL is worse by
`+0.003750 [+0.002214, +0.005266]`. Rank 32 remains worse than rank 8 by
`+0.017022 [+0.016220, +0.017826]` story NLL. The jointly learned embedding
without its LoRA scores `1.470652` and `1.472401` story NLL for the rank-8 and
rank-32 runs, so the result is not an embedding-only replacement.

Rank 8 trains 13,160,704 values (66.80% of base parameter count) and rank 32
trains 14,045,440 (71.29%); the tied embedding alone contributes 12,865,792.
Both completed 15,024 optimizer updates in 67.8 and 68.3 minutes. Peak
allocation was 8.16 GiB under the 12 GiB gate, and the authenticated end-to-end
run took 140.2 minutes. The publication manifest is
`ecdefc0e61f67e85f49ca8e15e8c50dadf930ab560fd2a6a3b475fd42301b013`.

The first completed-state replay exposed and fixed an analysis-boundary bug:
replay had included the allocator measurement's authentication envelope where
the initial run used only its raw payload. Metrics and rendered reports were
unchanged, but `analysis.json` differed. The runner now normalizes that payload,
a regression test covers both representations, and two subsequent completed
runs strict-loaded all checkpoints and ledgers with zero retraining or
reevaluation; the final replay regenerated the complete publication
byte-identically. The focused embedding/rank-sweep/temporal CPU suite passes
with one resource skip, and the real-GPU joint-update/parity/allocator gate
passes. The clean default CPU suite passes 718 tests with 275 resource skips in
89.2 seconds using the new four-worker `pytest-xdist` work-stealing default;
`pytest-xdist>=3.6` is now an explicit development dependency.

## Active Outcome — TinyWorlds Nouns-v2 Disjoint Benchmark

The isolated `tinyworlds-nouns-v2` contracts, partitioner, shared-engine
version bindings, canonical runner, bounded evaluation wrappers, optional
judging wrapper, standalone audit/report builders, and CPU/real-source gates are
implemented. Nouns-v1 remains unchanged on disk and strict-loads with its
completed Markdown, HTML, and run-manifest hashes intact.

The v2 manifest authenticates the complete nouns-v1 parent partition, reviewed
breakdown, decisions, sources, tokenizer, and all 2,745,124 story/token records.
Production partition
`210c4e2d067077fe774782024a594ade7e7472a986d554f186453549cf910f1b`
was independently rebuilt byte-for-byte. It contains the frozen 24 tasks in
descending purified mass, 2,210,934 clean base-universe stories, 429,199 pure
task-training stories, 4,440 pure validation pairs, and permanent ledgers for
77,361/776 excluded multi-task train/validation stories. The 2% internal base
holdout is 44,286 stories, leaving 2,166,648 optimizer-visible stories (79.73%
of original training versus 81.36% universe coverage). Every task still clears
the 256/64 threshold; 36 context-fitting probes per task are excluded only from
that task's update stream.

The baseline-focused CPU suite passes 29 tests. RTX 4090 preflight
`c4fda525322e00f2b271d351aa263fffda50dd238acd36b85b9709ceba36cc70`
passed at 6.47 GiB against the frozen 12 GiB gate. Fresh seed-zero base training
is complete under identity
`94831c31c8f11a594534c2989182d378fc2e022382b61168b73e7400f9648e21`:
held-in NLL improved from 1.35878 to 1.27038, the selected parameter checksum is
`fff309bfbfcee8d59c5c3fc04152cc37be2142201f3bf9116b7b024e81a24f3c`,
and measured allocator peak was 8.49 GiB. All 24 immutable VAMP stages and
48,000 adapter updates are complete; the final tensor checksum is
`97414ac3d8656ab083b2e570a4162dc69b024f90cf819b80b1cab94213553e63`.
The matched controls are also complete: one sequential LoRA received 48,000
updates across the 24-task stream, while 24 independent root LoRAs received
2,000 updates apiece. Their authenticated cumulative tensor checksum is
`24257b20257e99fa99e74c7cae6a133232c601f1407ac04ed3b96893bd73e464`.
The sequential full-model control also completed 48,000 matched updates with no
LoRA and no task identity at evaluation. It carried every GPT-Neo parameter
through the 24-task stream, reset AdamW at each task boundary, stayed within the
allocator gate at 8.58 GiB, and finished in run
`ea80b00eab0ee31bf93678e1e0612b82fc32fa459147af47eccbcac45306bba5`
with parameter checksum
`c177ba1080fb8fc989aaf5705ebc2b05529e5d5bbe30c02f0be02f423b37720a`.
All 26,640 whole-story NLL/routing rows and all 4,440 midpoint generation rows
are complete. The follow-up continual-learning audit published 72,256
VAMP task/story/stage cases spanning each task's introduction through stage 24,
plus 72,256 matched sequential/independent adapter cases and 72,256 sequential
full-model cases.

Stored-oracle NLL drift is exactly zero, confirming that immutable VAMP task
functions are retained, and independent-adapter drift is exactly zero. The
sequential LoRA finishes at 1.746 story NLL with +0.2156 mean forgetting and
+0.3661 worst-task forgetting. Full-model fine-tuning is substantially worse:
it finishes at 2.048 story NLL with +0.5744 mean forgetting and +0.9411
worst-task forgetting. The task-aware independent ceiling finishes at 1.523
story NLL; VAMP's task-aware stored oracle reaches 1.539. Among task-free
methods, exhaustive routing finishes at 1.572 NLL, 73.9% accuracy, and +0.0029
mean forgetting, versus 1.615/37.4%/+0.0160 for Hopfield,
1.581/70.4%/+0.0128 for EBT uniform, and 1.582/64.5%/+0.0118 for EBT Hopfield.
The final-stage suffix results match the existing generation ledger exactly.

The refreshed Markdown and self-contained folding HTML reports include the
comparative continual-NLL plot, routing plot, and rendered 25-node/24-edge VAMP
dependency graph. Report identity is
`f15725c35bc459d55f87c8078c4a99829de207d16aafb5c08291418b704c8be7`;
the VAMP, adapter-baseline, and full-model stagewise ledger SHA-256 values are
respectively
`e8ee50c59b04a1656fec440e8dec801b556df85577d57252eee0e51cf6d88def`,
`6821b6d04d52dd9c6fd5445e5cef7ad0a6491b6a6a2165d6138caceaf054fb13`, and
`cfcc4639a34b2a5865b0c19ef20b5f70aa27294a5831ffeaede5b75f32ebed68`.
The canonical command reached `local_complete`, notified successfully, and a
final invocation independently reconstructed the partition and strict-loaded
the base, all 24 VAMP stages, all 24 cumulative adapter stages, all 24
full-model stages, and every evaluation ledger without repeating model work;
all publication hashes reproduced byte-for-byte. The full-model extension took
55 minutes on the RTX 4090. The complete default suite passes 667 tests with
274 resource skips and 16 marker deselections; the opt-in production gate
passes all 5 tests, including full partition reconstruction, strict
VAMP/adapter/full-model loading, comparison-ledger authentication, and frozen
nouns-v1 hashes. Nouns-v1's report and run hashes remain unchanged. The local
nouns-v2 benchmark, all three stagewise audits, controls, plots, and graph
report are complete; OpenRouter judgment remains an optional resumable
extension behind `--judge`.

## Active Outcome — TinyWorlds Noun-Overlap v1

The isolated `tinyworlds-nouns-v1` engine is implemented through partitioning,
fresh-base training, VAMP-only adaptation, both evaluations, OpenRouter
judging, and artifact-derived Markdown/interactive-HTML reporting. The generic
language parent scorer now accepts an immutable eligibility mask while retaining
every raw score, language tasks can consume lazy indexed batch sequences, and
adaptation artifacts can explicitly represent a VAMP-only run without
fabricating independent or sequential baselines.

The pinned TinyStories scan completed on 2026-08-05 and published strict review
packet
`df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35`.
It authenticated 2,717,495 training records and 27,630 validation records,
yielding 2,717,494 unique training stories. The proposed greedy base order is
`mother, home, bird`, covering 51.970% of unique training stories and 54.470%
of their tokens. The current decisions project 42 adapter tasks, ten included
families below the 256/64 task threshold, and two explicit exclusions (`saw`
and `friend`). The 42 task validation sets contain 27,866 nonexclusive
task/story memberships; this is also the projected number of generation cases
and external judge requests.

The user manually approved that exact noun breakdown on 2026-08-05. Approval
artifact `2d923cb596d0c01d51a3f0848fb8332a02006d791e218a8163a74505efc92bd5`
binds the reviewed breakdown, decision snapshot, source identity, and statement
of approval. Canonical execution published partition
`04ca2acf85f9505f0b7568b1696fbf290a8d2cbf78387dcfd6e815258fcc28b8`
and passed GPU preflight `e15b669aca33ca9a200244e7742589532d3b4b0d2baa3bb3eddd727d0a5cd026`
at 7.99 GiB. The fresh base improved held-in NLL from `1.41908` to
`1.31084`, passed every gate, and published selection
`c900a4fc47fcb8317900c83c53e61be33e0c0c856e624a8713e7348a57e27788`.
All 42 VAMP stages and 84,000 adapter updates are complete; the last immutable
stage is `c56516ed22a9e5ee89868330fc61031c90edf6ea1d2d5d2dcf75e191e9fd0156`.

Whole-story evaluation is complete: all 27,866 task/story memberships produced
167,196 canonical condition rows in the atomically published 68 MiB ledger.
The initial one-story implementation projected roughly 25 hours, so the final
evaluator shares 32 story windows across node scoring and caps differentiable
EBT sub-batches at eight rows. Its safe run matches every byte of the completed
32-row EBT attempt.

Midpoint generation is running in its immediately persisted resumable ledger.
The generic greedy decoder pre-fills once, uses per-layer KV caches, and
advances tokens in one compiled device loop. The noun evaluator shares one
generation whenever conditions select the same story/node pair and first-fit
packs similar frozen budgets across bounded 128-window host chunks. Device
calls contain at most 72 distinct addressed rows; the measured live footprint
is approximately 9.7 GiB, below the frozen 12 GiB limit. A 192-row attempt was
rejected after reaching 17.9 GiB, and its rows remain diagnostic-only.

The first canonical generation process durably completed 6,091 of 27,866 cases
before XLA's accumulated CUDA command graphs exhausted device memory. Every
completed case remained intact. The canonical launcher now disables XLA GPU
command buffers before importing JAX, as recommended by the backend failure,
and resumed at case 6,092 without repeating completed work. The final Markdown
and HTML reports do not exist until this local pass completes.
`OPENROUTER_API_KEY` is absent, so completion will publish both local reports,
notify the desktop, and stop at `awaiting_judge_credentials`; a later canonical
rerun can begin directly with external judging. The focused
generation/noun/VAMP selection passes 25 tests.

## Completed TinyWorlds-Q Outcome

The generic `tinyworlds-q-semantic-v1` query benchmark is implemented and its
CPU gates pass. It is a separate package and artifact family; no semantic-v6
file, checkpoint, result, or compatibility path changed. Query-v1 measures
reviewed four-choice fact knowledge directly rather than using whole-story loss.

Implemented surfaces now include immutable concept/fact/template/catalog,
partition, experiment, and result contracts; fixed rabbit/horse and five-world
manifests; namespaced 5% construction selection; exact same-sentence predicate
discovery and provenance review packets; mandatory human review gates; sealed
catalog publication; fact-withholding partitioning with exact story/token
ledgers; memory-mapped indexed batching; exact scratch-base training/resume;
strict accepted-base publication; validation-only parent/router probe
preparation; real independent, sequential, and VAMP tensor stages; compilation
into the shared `KnowledgeQuery` scorer; deterministic fact-level bootstraps;
exact-trigger generation inspection; dynamic 1--100 world capacities and
schedules; bounded scoring; atomic JSONL ledgers; resource preflight; one-time
sealed transactions; and schedule-complete descriptive Markdown/standalone-HTML
reporting. Experiment identities include the complete GPT-Neo architecture,
all six LoRA target switches, and the derived adapter optimizer contract.
Selected bases bind the full catalog partition and base-training contract
rather than an active adapter prefix, allowing one large-catalog base to serve
the registered nested prefixes.

The CPU fixtures cover construction exclusion, story-level fact
leakage, same-sentence evidence accounting, multi-concept exclusion, exact
candidate balance and tokenizer boundaries, byte-identical partition rebuild,
catalog/partition/tensor tampering, parent-prefix preservation, sealed-query
rejection and completion, fact-level statistics, a real tiny GPT-Neo
uninterrupted-versus-resumed parameter/trace parity check, and resumable
independent, sequential, and VAMP stage identities. The pilot-specific fixture
also covers the compact 24-primary/8-backup review queue, evidence support,
token balance, and publication. Synthetic manifests at 1,
5, 10, 20, and 100 worlds cover derived capacities, tensor masks, full and
milestone schedules, chunking, dynamic reports, and explicit preflight limits.
The focused TinyWorlds-Q suite passes all 11 tests. A broader 80-test query,
knowledge, scoring, training, statistics, routing, artifact, and semantic-v6
compatibility run also passes in the pinned semantic environment. The complete
default non-opt-in collection passes with 626 passed, 274 registered resource
skips, and 11 marker-deselected tests.

The real pilot proposal packet is now published at
`data/tinyworlds-q-semantic/review/5b01c86812593681133b46effd786d5647dcb3e8cf0308e8482bb54f01b7775b`.
A fresh 24-worker replay authenticated all 4,967,871 archive records, scanned
4,967,647 nonempty duplicate groups, and selected 248,051 construction groups.
The strict-reloaded packet contains 200 ranked candidates for rabbit and 200
for horse; all 400 have at least sixteen supporting construction groups. Its
14 GiB replay workspace remains at
`data/tinyworlds-q-semantic/work/pilot-review-primary`.

The raw 400-candidate packet is now explicitly an audit appendix, not a human
work queue. A targeted replay over the retained index published the 29-predicate
evidence packet
`1603f089988125c2a0782d5bb41ebb0ce113ec466ed6248b14ad4a8e0040d071`
and compact shortlist
`ad00bafb6bc5adef50a76f2b1ff7230bce02e46b04526d7bf81753a01dc5dd65`.
The concise `review.md` is 66 lines: twelve primary proposals per concept,
four backups per concept, one representative sentence and support count per
primary, reviewed-form placeholders, exact trigger closure, and tokenizer-
balanced false choices. Detailed evidence, exact token IDs, HTML, canonical
JSON, and an editable TSV remain alongside it for drill-down.

The interactive user approved all 24 primary proposals at
`2026-07-25T04:30:07Z`. Approval artifact
`fbe0db124a77ce0215b2632d12cc97320e7eeda60de77b3fe8d48384eaef539b`
records affirmative truth, evidence, trigger-closure, answer-form, and forward-
distractor gates for every primary row. No backup was promoted.

The interactive user approved all 24 fact-specific reverse choices at
`2026-07-25T04:40:14Z`. Reverse approval
`bc184647bfec6f33c04a0e527d1c70e4c1415555695fedbf5d09d4066a41bbb8`
binds corrected review
`32f206833ce828fb954628d9063821c853579b96ffbf567d5dd0a2fc5e0ce9c0`.
Official catalog
`5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f`
contains 24 facts, 72 validation queries, and 120 physically separated sealed
queries. Validation-only strict reload passed; sealed prompts were not
deserialized.

Pilot partition
`419e6c8b6362add9af081885066559cc34b18f5c7044894f343c7caf0091ad0c`
passed every construction, leakage, fact-support, and lexical-exposure gate.
Its weakest fact has 320 non-construction training groups; rabbit and horse
retain 11,344 and 3,859 ordinary lexical base groups. A fresh 24-worker archive
replay with 37,000-record sort runs reproduced the complete 12 GiB tree
byte-for-byte; both trees hash to
`7b8c50a68cfcde41dc1579836ab7bb431fd85a4652c0fd036ab8986adae87f9f`.

GPU preflight
`6519ee1a5820a039c7b3f8e016b149fd7a90bb23fd5c0cb468a430cd6ed84eb8`
passed on the RTX 4090. Two disposable updates had finite NLLs `10.859072643`
and `10.853386662`; the warm update took `0.488339` seconds and allocator peak
was 7,417,784,832 bytes, below 12 GiB. It projects 18,530 updates per epoch and
about 5:01:38 for the registered two-epoch base. The real seed-zero run
completed under
`checkpoints/tinyworlds-q-semantic-v1/work/pilot-base-6fbf5f5e5a7ab4cd3c862884a8b64f08e931d4fe209d57376ebda10c9c5f4bac`.
Held-in validation NLL improved from `1.231696441` after epoch one to
`1.157588485` after epoch two, a `0.074107956` improvement. The allocator peak
was 7,557,684,224 bytes. All base gates passed, and selected base
`91b1dd7cf314fcdf81509d6421a3a33621f7106a54161d0aa080911dc1db4961`
is published. That base authorized the fixed pilot sweep.

The registered independent sweep completed one exact 2,000-update trajectory
per world and preserved its 500/1,000/2,000 snapshots. Under the original
preregistered policy, no budget passed both the 60% accuracy and 15-point
acquisition gates for both worlds. Authenticated failure
`aad4811425c10b0faf5f6f452067e35a58d6cee397970711951e50bfad2247f5`
remains immutable evidence of that stop.

After reviewing the ceiling-sensitive rabbit baseline, the interactive user
authorized explicit amendment
`2855b647928700a119ea6e95365379719ad733d45c6ede20cafcd1593a64458c`.
It keeps 60% absolute validation accuracy as the learnability gate and makes
acquisition a mandatory descriptive statistic. It neither rewrites the
failure nor opens sealed test data. Under the amended policy, 2,000 updates is
the first budget where both worlds pass: rabbit reaches `0.638889` from a
`0.555556` base (`+0.083333`), and horse reaches `0.611111` from `0.250000`
(`+0.361111`).

The selected two-world independent, sequential, and VAMP exercise completed,
all nine validation methods were written to the bounded ledger, and an exact
completed-stage no-op resume passed. Pilot result
`55c97f2a649ea434f79e729b2eaff01753a254ce0a5c26e53a1095d4df0364c7`
binds policy, amendment, tensors, ledgers, runtime, and memory. It is an
operational pilot only and gives VAMP no scientific verdict. The sealed test
remains closed.

The five-world main configuration is now frozen at
`82d0d3258e0e723588d151387c0151156b408770df1f84bcb5450ac72f9327ff`.
It fixes order `cat, dog, bird, robot, dragon`, a fresh seed-zero base, 2,000
adapter updates, the exact query/scoring protocol, all nine methods, 10,000
fact-resampled bootstrap replicates, validation-only routing, and a single
sealed opening after all artifacts are frozen.

Main construction review is complete. Raw discovery packet
`7164cd2cc18be5ba29d7106a44f23dbec5bf39a9a962b9c441ccf07501a8132f`
and targeted evidence packet
`ce1b06c7f7a325cedded9970ac008329c93d97d29c84344b93d22b450db14374`
remain non-authoritative audits. Compact shortlist
`fe2f78e92e1c4e0d26280f2741beea728ea3125c932c3126b770da6cd90104cc`
contains 12 primaries and 4 backups per world, exact token-balanced choices,
reviewed trigger proposals, and construction support; the smallest primary
support is 17 groups. The interactive user approved all 60 primaries at
`2026-07-25T22:38:51Z`; approval
`8b0f2868b216b837f2b2c90c0f7faaa141874fe87b2387c6fecd62faed8f616b`
records all five affirmative gates and promotes no backup. Fact-specific
reverse review
`c805da6c075920f85a58b0c4ed25ee4aa6dac2e5763e2578648efd0c0800e1f0`
contains 60 one-token four-choice rows. The interactive user approved every
row at `2026-07-25T22:54:49Z`; reverse approval
`c643731930ae9721ea4c4420f14a830c04ca8179bee8caccb8a73756ec0c1067`
strict-loads with the complete primary authority.

Official catalog
`0ffd78e81d1da4a4fbd20b49bc02f3dec94560085f4490a357c7f73239f9e8ba`
contains 60 facts, 180 validation queries, and 300 physically separated sealed
queries. Its independent catalog trees are byte-identical at
`d6c13a83bf1c614115b2a246bf93b33cd12d9e6ab1b9730a4a43f0ba19cef75f`;
the pilot catalog hash is unchanged. Main partition
`d8536d0295af4fa56174369430b2e615008e28fb239d7d66a428b36988fa7d6b`
passed every primary construction, leakage, support, and lexical gate plus a
strict full-tree reload. It retains 3,509,177 base groups and 669,256,202 base
tokens. Its weakest fact has 248 training groups; robot is the tightest
ordinary lexical exposure at 372 base groups. A fresh 24-worker archive replay
with 37,000-record sort runs independently reproduced the complete partition;
the two trees are byte-identical at
`566700c59c9c05e87525806a2fd54ff48d283b57b4212884153a6808b12a9828`.
Validation-only sample report
`a677d66b572610229a52d4d46b20b30d206f665afeb1c8fc3a82fd5e6c170143`
strict-loads with six exact stories and all 180 validation queries. The 11-test
query suite, broader 80-test compatibility selection, and complete default
collection pass; the latter reports 626 passed, 274 resource-gated skips, and
11 marker deselections. These sources form the main-partition checkpoint;
GPU preflight
`28380737a808e4288c9b8b51cd6a97e9c64c60e23a59b51e10fd2ea565e14641`
also passes. It measured an 8.412 GiB allocator peak and projects 9.040 GiB,
both below 12 GiB; a warm update took 0.418 seconds and projects the two-epoch
base at 3:21:12. The first attempt correctly stopped before an update because
`ve-semantic` has CPU-only JAX; the accepted run used the existing CUDA JAX
0.6.2 `ve` environment on the RTX 4090. Shared pilot/main base orchestration
and the registered main launcher are implemented and focused tests pass. The
fresh base completed all 28,912 updates under training identity
`001e16d8908ae593ffc23b423a1a672e005c3cf7b35dacbb09636d1807a96d93`.
Held-in NLL improved from `1.266449873` to `1.189207350`, a
`0.077242523` reduction, over 13,288,046 active validation tokens. Allocator
peak was 9,032,018,176 bytes, below 12 GiB, so every base gate passed. Strict
selected base
`0777adef5291c416d53af23ac6694bcfd308f0f6534883e4cc7cede2254783a2`
is published; its launcher wall interval was approximately 3:49. A fresh
launcher invocation authenticated the full partition and preflight, strict-
loaded that selection, and exited without another optimizer step. The exact
main adapter/validation runner and a separately guarded sealed runner are
implemented. Final-analysis protocol
`489042464fd4243e1780d585c4ba7ed6cd1134c9f7a5bf3d7e6f2fb4aaa8712a`
fixes all nine methods, acquisition and retention for every non-base method,
independent cross-world specificity, accuracy and margin effects, router node
accuracy/regret, 10,000 fact bootstraps, and 96-token greedy generation from
the matching final independent adapter. Headline condition summaries use only
the primary matching adapter/path rows; the forced cross-world independent
matrix enters the specificity effect and ledger instead of being pooled into
independent accuracy. The protocol enters the validation freeze before test
access.

Resource accounting now explicitly includes every forced independent-adapter
specificity cell. The five-world sealed ledger is 9,900 rows and a conservative
10,137,600 bytes, not the accepted v1 preflight's 8,100-primary-row / 8,294,400-
byte approximation. The immutable preflight still authenticates; its measured
training/memory evidence is unchanged, and the corrected projection is far
below the frozen 4 GiB limit. New preflights use the complete count. The final
report and validation loaders now reconstruct every canonical result row,
recheck schedule/routing coverage, and recompute registered fact effects and
derived final renderings during recovery; byte hashes alone are not treated as
semantic parity.

All 30,000 main adapter updates are complete: 2,000 updates for each of the
sequential, independent, and VAMP systems in each of the five ordered worlds.
All five immutable tensor stages strict-load, and a completed-stage no-op
resume reproduces final tensor checksum
`f2e744d2ebb1d182e74ada95970ce126bb93cc0fec370b535e07d6e889241878`.
VAMP attached cat, dog, robot, and dragon to root and attached bird to dog.
The measured stage-persistence wall interval was `1839.455669` seconds; the
recovery invocation itself spent `9.685603` seconds in adaptation/no-op resume.

The first complete validation evaluation exposed a publication-check bug: the
shared knowledge scorer correctly records the task-oracle node for every
direct-query row, while the new strict validator had incorrectly treated that
field as router-only. The validator and final-report recovery now require
oracle-node evidence for every method and reserve selected-node/regret fields
for the five routed methods. The failed attempt published no validation
artifact and opened no test data. A fresh deterministic-schedule evaluation
then published 5,940-row validation freeze
`a32636c0a467cf4e95253cf19792d5eda0ede556e059376988cc410b7ba6dfb0`
with result-ledger SHA-256
`8baa895b9f234c79eabcbd140e503c2c7a7609f4af37f79a6c39b8f2a704e9b9`.
Its aggregate validation accuracies are base `0.372222`, independent
`0.564815`, sequential `0.438889`, VAMP oracle `0.531481`, exhaustive
`0.468519`, Hopfield `0.414815`, uniform EBT `0.464815`, Hopfield EBT
`0.464815`, and deterministic random `0.411111`. Validation took `602.319236`
seconds and allocator peak remained 9,032,018,176 bytes.

Closed sealed transaction
`ce92e165fcc3f58b449253a628e7616ef254c700c225db45ab88708f8f8de946`
binds the catalog, partition, selected base, preset, and exact validation
freeze. The separately guarded launcher durably opened that sole transaction,
evaluated all 9,900 scheduled rows, ran the frozen generation inspection and
10,000-replicate fact analysis, atomically published result
`8f34f8fe9f791ae822b2cdde35ebb1cb24b9a4f7efab0c68e0cf600f694a9986`,
and wrote its matching completion marker. Strict file hashes and semantic
reconstruction pass for the 5,025,203-byte JSONL ledger and all derived report
views.

The final five-world test accuracy is `0.360000` for base, `0.603333` for
independent adapters, `0.446667` for the overwritten sequential adapter,
`0.603333` for VAMP oracle paths, and `0.443333`--`0.450000` for the three
exhaustive/EBT task-free routes. Independent acquisition is `+0.243333`
(95% fact-resampled interval `[0.193333, 0.293333]`), with zero measured final
accuracy loss and `+0.256667` node specificity. Sequential acquisition is
`+0.236667`, followed by `-0.150000` retention loss. VAMP oracle also has zero
measured accuracy loss; its best final task-free router-node accuracy is
`0.600000`, identifying routing as the principal gap to the oracle result.
All five secondary greedy generations have zero exact registered-trigger
recall. Sealed evaluation, generation, and fact analysis took `638.502606`,
`32.231832`, and `1.059834` seconds; peak allocation remained
9,032,018,176 bytes. This is descriptive evidence and carries no scientific
pass/fail label. A post-publication strict semantic reload reproduced the exact
report identity, and the final 11-test focused plus 80-test compatibility gates
pass. See
[`docs/TINYWORLDS_Q_SEMANTIC_V1_EXECUTION_REPORT.md`](docs/TINYWORLDS_Q_SEMANTIC_V1_EXECUTION_REPORT.md).

A post-result presentation view is now published at
[`docs/TINYWORLDS_Q_SEMANTIC_V1_INTERACTIVE_REPORT.html`](docs/TINYWORLDS_Q_SEMANTIC_V1_INTERACTIVE_REPORT.html).
It is a dependency-free interactive HTML page derived from the authenticated
completed report and transaction-published opened audit; it does not rerun a
model or invoke the sealed catalog loader. It now applies a uniform
forward-only scope: all displayed comparisons are recomputed from the 180
sealed prompts that explicitly name cat, dog, bird, robot, or dragon, while all
120 reverse prompts are excluded and disclosed. Fact paraphrases are averaged
before deterministic 10,000-replicate equal-world bootstrapping. Forward
accuracy is `0.427778` for base, `0.533333` for independent adapters,
`0.455556` for the overwritten sequential adapter, `0.527778` for VAMP with
the right node, and `0.516667` for content-start EBT routing. Independent
acquisition is `+0.105556` (95% interval `[0.033333, 0.177778]`). The EBT
routers select the named world with `0.772222` accuracy (95% interval
`[0.700000, 0.838889]`), separating routing quality from their `0.516667`
answer accuracy.

The non-cherry-picked explorer fixes forward test paraphrase 00 for every
fact, giving 60 cases balanced across all 60 facts and five worlds. Folded case
cards expose the four choices, within-question preference, all nine method
answers, routing choice, and reviewed construction evidence. Presentation
presets cover 7 base misses repaired by the matching adapter, 27 misses that
remain, 7 cases lost by sequential overwriting, and 2 cases where stored VAMP
knowledge is correct but the content-start router selects a wrong node. The
page also contains a plain-language guided tour, a routing-versus-answer table,
forward-only per-world results, a percentage-point explanation, all 60
reviewed facts, a method glossary, post-result disclosure, and exact source
identities. Its deterministic HTML SHA-256 is
`5db2ba80d592aabce69b5b18f14e54fc4a4a0f7fd59572269a7fd419030abe2d`.
Focused statistics/parser/rendering/tamper tests pass, and a headless-browser
interaction check verified five headline cards, five router rows, the absence
of a direction filter, 60 total and 12 per-world cases, the two routing-miss
examples, four choices per case, and explicit 300-source/180-included/
120-excluded provenance.

Semantic-v6 remains immutable negative evidence. Its seed-zero calibration
stopped at `semantic_grid_failure`; its sealed test was never opened, and it
must not be rerun or reinterpreted as query-v1 evidence.

## Completed Milestone: Semantic-v6 Base Gate Stop; VAMP Not Opened

- **Strict v6 base machinery is complete.** Version-native training, resume,
  per-group evaluation, empirical-null validation, selected-base publication,
  and strict loaders reject archive-v1 and semantic-v1 identities. The two-
  epoch gate remained separate from test access and correctly prevented the
  epochs-three-through-five continuation. No epoch satisfied the unchanged
  semantic gap gate, so no selected checkpoint was published.
- **The VAMP study is frozen.** Experiment config
  `ca16318486600745e8a49903f495819741082f120fa7b95b3f9277efa83ada73`
  fixes A-to-E order, three rank-eight adapter systems, 2,000 updates per
  system and world, validation-only parent/key probes, the four stored methods,
  five task-free routers, all prefix/cue conditions, timing, memory, and the
  diagnostic paired-control specificity audit.
- **Resume and sealed boundaries are implemented.** Base checkpoints retain
  optimizer, random, cursor, and schedule state every 1,000 updates and at
  epoch boundaries. The runner resumes the newest strict checkpoint and trims
  only the later, uncheckpointed loss-log tail. Adapter publications persist
  all three random streams and one immutable stage artifact per completed
  world. Evaluation ledgers are atomic, while incomplete validation or sealed
  attempts are preserved under recovery directories. The final evaluator
  writes a durable binding only after the base and all adapters are frozen;
  test indexes cannot be counted or read before that transaction.
- **Reporting is implemented.** Sequential progress is written live under the
  printed temporary artifact directory. The final content-addressed bundle
  includes Markdown and standalone HTML reports, canonical JSON/JSONL,
  per-group base and forced-adapter ledgers, exact test provenance, the full
  nine-method matrix, forgetting, transfer, routing cost, memory, and 10,000-
  replicate specificity intervals. The VAMP result is explicitly exploratory
  and has no new pass/fail threshold. Adapter and result artifacts persist and
  enforce the 12 GiB allocator ceiling.
- **The focused implementation gate passes.** All 61 pinned-environment tests
  pass across strict partitioning, empirical statistics, training and resume,
  adaptation persistence, nine-method evaluation, routing/memory accounting,
  sealed authorization, and paired specificity.
- **The real validation-only anchor audit passed.** The canonical root and all
  five worlds each supply 128 deterministic full-length validation spans. All
  768 sequence hashes are unique, and no test index was read.
- **The disposable GPU preflight passed.** Preflight
  `b7f49909368685a5494a3033e0df7df69cf2e8c1064092c541013b873671988d`
  completed exactly two isolated updates with losses `10.8570` and `10.8511`,
  measured `0.467411` seconds per warm update and `0.015083` seconds per warm
  validation batch, and peaked at 9,030,551,296 bytes. Its checkpoints use a
  separate non-reusable identity. The sealed test remained closed.
- **The registered calibration stop is complete.** The real run published two
  validation ledgers and the immutable `semantic_grid_failure` decision. It
  preserved strict base states but published no selected checkpoint, adapter
  tensor, VAMP result, or sealed-test report. See
  [`docs/TW-P_SEMANTIC_V6_BASE_VAMP_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V6_BASE_VAMP_EXECUTION_REPORT.md).

## Completed Milestone: TinyWorlds-P Semantic-v6 Exact Comparison Feasibility

- **The intervention is preregistered (2026-07-23).** `DESIGN.md` binds the
  semantic-v4 catalog and semantic-v5 failure identities. All 22
  balance-eligible layouts receive the real split allocator and complete
  ten-control construction before the unchanged semantic ranking is applied.
- **The separate implementation is complete.** Version-specific contracts,
  full-candidate feasibility evidence, strict success and failure loading,
  publication, validation-only sample reporting, focused tests, and a fixed
  primary/rebuild runner are implemented. Shared archive code exposes the
  existing exact split/control preparation step without changing its normal
  publication behavior or providing a compatibility alias. The final focused
  CPU suite passes 57 tests, and the new modules compile in the pinned
  environment.
- **The real feasibility screen succeeded.** Both archive runs retained
  2,520,317 unique stories and 479,183,203 scored tokens and reproduced all
  28,224 parent topology measurements. Seventeen of the 22 balanced layouts
  completed all validation/test comparisons. Ranks 0, 8, 9, 14, and 21 failed
  world B's validation column comparison because too few distinct stories
  remained after the fixed split and global non-reuse rules.
- **Semantic rank 1 is the registered winner.** Its A-through-E cells are
  `(2,4), (7,4), (7,6), (2,6), (3,2)`, with scored-token masses `6,136,097`,
  `5,873,159`, `5,921,676`, `6,114,634`, and `5,440,146`. Every mass lies
  within 10% of the median. The final allocation has complete comparisons and
  31,117 deterministic one-to-one pairings.
- **The partition is strict and independently reproduced.** Partition
  `3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`
  was built with 50,000-record sort batches and rebuilt from the archive with
  37,000-record batches. Both complete strict reloads passed. A direct recursive
  comparison found no difference across their 167 files, and both trees hash to
  `b5ba1ce33d1cad7eb00bba0b6eec35e2b94c3a6b997a20149081cc61c862279d`.
- **The pre-training sample report is published.** Report
  `b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579`
  covers the held-in validation set, all five worlds, and both comparison arms
  for every world. It includes cluster inventories and exact archive
  provenance, parses as self-contained HTML, and records that the sealed test
  was not opened. See
  [`docs/TW-P_SEMANTIC_V6_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V6_EXECUTION_REPORT.md)
  and the generated
  [`sample report`](data/tinyworlds-p-semantic/sample-reports/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa/b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579/sample-report.md).
- **No downstream boundary was crossed.** Semantic-v6 has no GPU preflight,
  optimizer update, checkpoint, model-loss result, semantic-gap decision, or
  sealed-test evaluation. The successful construction does not by itself
  authorize training.

## Completed Milestone: TinyWorlds-P Semantic-v5 Control Stop

Semantic-v5 makes one change to the completed v4 partition attempt. It treats
the unchanged 10% cell-mass rule as an eligibility requirement before semantic
ranking. The exact v4 catalog and partition failure are immutable parent
evidence.

### Semantic-v5 Status

- **The intervention is preregistered (2026-07-23).** `DESIGN.md` binds the v4
  catalog and failure identities and states the new selection order. V5 keeps
  every word, cluster, source, threshold, nuisance measure, split, control,
  pairing, and sealed-test rule unchanged. It cannot use model loss or promote
  a diagnostic candidate under the v4 name.
- **The separate v5 implementation is complete.** Dedicated contracts,
  balance-first selection, strict parent/source/settings checks, partition and
  sample-report formats, structured control-failure evidence, a fixed runner,
  and focused tests are implemented. The builder independently reproduces all
  parent topology records before it can select a v5 layout. No v4 artifact is
  loaded through a compatibility alias.
- **The balance-first intervention worked.** The real archive replay retained
  2,520,317 groups and 479,183,203 active tokens and reproduced all 28,224 v4
  topology measurements. Twenty-two layouts were balance-eligible. V5 selected
  `(3,4), (4,4), (4,6), (3,6), (2,0)` with masses `9,899,869`, `8,829,612`,
  `8,742,369`, `10,104,204`, and `9,357,468`; all five pass the unchanged 10%
  rule.
- **Exact control allocation stopped the partition.** World B's validation
  column arm required 2,314 distinct groups, but only 1,511 remained after the
  fixed split and global no-reuse rules. The shortage occurs before fine
  nuisance or token matching. V5 therefore did not choose another balanced
  layout or loosen a tolerance.
- **The stop is strict and independently reproduced.** Failure
  `090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`
  embeds the exact v4 catalog and parent failure and binds the complete
  assignment-ledger SHA-256. A fresh 37,000-record-run rebuild reproduced the
  50,000-record-run assignment ledger and all 54 MB of failure evidence byte
  for byte. Both strict loaders pass, a recursive comparison finds no
  difference, and the final focused CPU suite passes 51 tests. See
  [`docs/TW-P_SEMANTIC_V5_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V5_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11/audit.md).
- **No downstream boundary was crossed.** V5 has no success partition, sample
  report, GPU preflight, calibration, checkpoint, or sealed-test result. A
  different candidate, exact split-level control-feasibility prefilter, changed
  split size, altered matching design, or control reuse belongs to a later
  version and cannot reinterpret this result.

## Completed Milestone: TinyWorlds-P Semantic-v4 Partition Stop

`tinyworlds-p-semantic-v4` tests the single fixed-reference intervention
motivated by the v3 deletion/reseeding cascade. The scientific contract is
preregistered in `DESIGN.md` before the real construction is run.

### Semantic-v4 Status

- **Contract frozen and implemented (2026-07-22).** V4 binds the
  canonical v3 failure
  `ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`
  and must exactly replay v3's pass-zero, unweighted eight-cluster fit in its
  original hash namespace. It then applies the unchanged `0.03` margin once
  against those frozen centroids. It never deletes and reclusters, moves a
  survivor, or updates a centroid after the boundary screen. Separate v4
  config, catalog, failure, audit, strict-loader, runner, and test contracts
  implement this rule without compatibility aliases.
- **The real fixed-centroid grid passes.** The one-shot screen excludes the
  exact v3 pass-zero sets: 188 of 978 noun candidates and 81 of 365 verb
  candidates. The eight clusters retain 790 nouns and 284 verbs; the minimum
  cluster sizes are 39 nouns and 18 verbs. Maximum noun/verb fit-centroid pair
  cosines are `0.8735721184` and `0.8916218581`, and retained joint archive
  token mass is 479,183,203 of 898,327,086 (`53.341729362%`). Catalog
  `ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`
  therefore clears every frozen gate.
- **Strict evidence semantics are verified.** Boundary-excluded candidates
  retain their vector, fit assignment, and measured margin in the v4 ledger;
  retained cluster inventories use the same fit assignments, while published
  centroids remain the authenticated all-candidate fit centroids. The loader
  reconstructs the fit, margins, dispositions, and gates under both pinned and
  current project numeric environments. An independent real rebuild reproduced
  all 11 files byte for byte; the self-contained HTML parses without external
  resources, five catalog fixtures pass, and the catalog-stage focused group
  passed 44 tests before partition work. See
  [`docs/TW-P_SEMANTIC_V4_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V4_EXECUTION_REPORT.md)
  and the generated
  [`catalog audit`](data/tinyworlds-p-semantic/catalog/v4/ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee/audit.md).
- **V4-native partition machinery is implemented and verified.** Separate
  partition/tree/preset/failure/sample-report contracts reject v1--v3
  artifacts. The CPU 3-by-3 fixture covers strict reconstruction,
  construction leakage, global one-to-one pairing, complete validation
  sampling, cross-worker/run-size byte identity, legacy rejection, and shard
  tampering. Topology failures now retain every ranked candidate, exact score
  fractions, source/seed bindings, a strict loader, Markdown/HTML audits, and
  byte-rebuild enforcement; synthetic tests cover repeat publication and
  tamper rejection.
- **The real partition stopped at the frozen topology gate (2026-07-22).** The
  archive replay retained 2,520,317 groups and exactly 479,183,203 tokens. All
  28,224 physical topologies passed nonempty, component-visibility, and
  control-capacity filters. The preregistered semantic-first winner used cells
  `(1,2), (3,2), (3,4), (1,4), (6,1)` and masses `2,559,355`, `5,440,146`,
  `9,899,869`, `4,699,583`, and `1,428,732`, violating the fixed 10%-around-
  median gate. The authenticated failure is
  `37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`;
  an independent replay reproduced its identity and every byte. The complete
  focused archive/semantic/partition/training CPU gate now passes 47 tests.
- **V4 is terminal unless a new version is preregistered.** Twenty-two other
  candidates satisfy the median gate, but selecting one after observing the
  failure would change the objective order. V4 therefore has no success
  partition, split allocation, paired controls, sample report, GPU runtime
  preflight, calibration, checkpoint, or sealed-test result. Any
  balance-feasibility prefilter, reordered objective, changed tolerance, or
  diagnostic-candidate choice belongs to semantic-v5 and cannot reinterpret
  this stop. See the generated
  [`partition failure audit`](data/tinyworlds-p-semantic/v4/failures/37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07/audit.md).

## Completed Milestone: TinyWorlds-P Semantic-v3 Construction Stop

`tinyworlds-p-semantic-v3` isolates the next intervention suggested by the v2
stop: semantic words are assigned only to their nearest spherical centroid,
while token and nuisance balance is deferred to story allocation after the
catalog is fixed.

### Semantic-v3 Status

- **Semantic-first contract implemented and frozen (2026-07-22).** V3 reuses
  the exact v1 MiniLM evidence and binds v2 failure artifact
  `23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
  Every real word's raw role score, fold, reference count, conformal value, and
  cutoff must exactly replay that ledger. V3 removes token-mass capacities and
  their repair from word clustering, uses unweighted farthest-first spherical
  k-means with nearest-centroid assignment, and records token mass only after
  assignment. The eight clusters, `0.03` true-nearest margin, five exclusion
  passes, word-count floors, `0.90` centroid-pair ceiling, and 40% joint
  retained-mass gate remained fixed.
- **Numeric construction provenance is explicit.** A first general-environment
  preflight preserved every decision and the complete boundary trace but
  exposed several-billionth serialization differences from the v2 ledger
  between NumPy 2.5.1 and 1.26.4. Its content-addressed bundle
  `94614921b5386653f92ee8dc372fc45b566502f9706723df57b257ab4a1252f2`
  remains preserved but is noncanonical. Before canonical publication, the v3
  config bound NumPy 1.26.4 and exact v2 score replay. No scientific threshold,
  cluster assignment, pass trace, or stop decision changed.
- **The real semantic-first grid reached a narrower automated stop.** The
  unchanged role/sense screens left 978 noun and 365 verb candidates. Noun
  boundary failures were `188, 17, 6, 1, 12, 1`; verb failures were `81, 7,
  0`. Verbs converged after two reclusters. On noun pass five, `crayon` remained
  below the fixed margin at `0.0296120345`, so v3 stopped. The canonical strict
  failure is
  `ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`.
- **The apparent one-word miss masks a cascade.** A post-stop diagnostic
  removal and sixth recluster exposed 22 new noun boundary words and a 25-word
  noun cluster, below the independent 32-word floor. Coverage would remain
  above 48% and centroid-pair cosines remain below `0.90`; the remaining issue
  is instability from hard deletion plus full reseeding, not mass balance or
  corpus coverage. Waiving the last word or extending v3's pass budget is not
  a valid repair.
- **Strict implementation and verification are complete.** Separate v3
  config/catalog/failure formats, semantic-only clustering, nearest-assignment
  replay, content-addressed Markdown/HTML audits, the fixed cached-evidence
  runner, and synthetic mass-independence/success/failure/rebuild/tamper tests
  are implemented. An independent real rebuild reproduced all nine canonical
  files byte for byte; the cached loader returns the same failure identity.
  Three focused regression groups pass 153 tests, and all semantic modules,
  runners, and tests compile. See
  [`docs/TW-P_SEMANTIC_V3_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V3_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v3/failures/ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146/audit.md).
- **No downstream artifact is authorized.** Semantic-v3 has no catalog,
  partition, sample report, GPU training run, checkpoint, or sealed-test
  result. Any fixed-centroid boundary screen, stable-core construction, robust
  objective, changed threshold, or changed pass budget must be preregistered
  as semantic-v4 rather than used to reinterpret v3.

## Completed Milestone: TinyWorlds-P Semantic-v2 Calibrated Construction Stop

`tinyworlds-p-semantic-v2` is the role-calibrated successor requested after
semantic-v1 exposed a systematic MiniLM anchor offset. It reuses the exact
authenticated semantic-v1 encoder evidence, leaves both archive-v1 and
semantic-v1 immutable, and changes no language-model, partition, semantic
vector, sense, cluster, retained-mass, or evaluation gate.

### Semantic-v2 Status

- **Cross-fitted role calibration implemented and frozen (2026-07-22).** The
  raw statistic remains each word's 10th-percentile
  `context·target-anchor - context·opposite-anchor` margin. Words are assigned
  to five SHA-256 folds in the `tinyworlds-p-semantic-v2` namespace. For each
  declared role and held-out fold, the other four folds form a word-level
  reference distribution; the held-out word receives the finite-sample
  lower-tail conformal value
  `(1 + count(reference <= score)) / (reference_count + 1)`. A word is a role
  outlier only when that value is at most `0.05`. Thus every decision is
  out-of-fold, role-specific, construction-only, and independent of model
  loss, partitions, and sealed test. No calibration-panel words are
  automatically discarded.
- **Separate v2 contracts and strict artifacts implemented.** The versioned
  config records the reused v1 evidence contract, fold namespace/count,
  conformal method and alpha, reference-size floor, all unchanged semantic
  thresholds, and the deterministic single-prior-word repair used only when
  indivisible token masses create a greedy packing dead end. The repair
  preserves descending-mass processing and the exact 90--110% capacity
  bounds. Content-addressed success/failure builders, exhaustive Markdown and
  standalone HTML audits, strict calibration replay, tree authentication,
  tamper rejection, and synthetic byte-rebuild fixtures are implemented.
- **The real role screen now behaves as calibrated.** Reusing encoder evidence
  `efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`,
  the five noun reference sets contain 837--883 words and the verb sets
  294--326. The calibrated lower tail excluded 51 of 1,066 nouns (4.78%) and
  19 of 394 verbs (4.82%). The unchanged two-sense gate excluded another 37
  nouns and 10 verbs, leaving 978 noun and 365 verb candidates for the fixed
  8-by-8 construction.
- **The unchanged boundary gate produced the new automated stop.** Noun
  boundary failures across the initial clustering plus five permitted
  exclusion/recluster passes were `259, 137, 103, 54, 55, 47`; verb failures
  were `106, 39, 23, 21, 18, 18`. Both roles therefore still had words below
  the assigned-cluster margin `0.03` after the pass budget. The authenticated
  failure bundle is
  `23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
  It contains all word scores, fold/reference evidence, conformal values,
  sense metrics, pass-level cluster masses and margin distributions, exact
  story contexts, candidate PCA, and every disposition.
- **No downstream artifact is authorized.** If the 47 noun and 18 verb
  failures observed on the terminal pass were also removed, the diagnostic
  remainder would contain 323 nouns, 140 verbs, and only 98,322,186 joint
  tokens (10.945% of the 898,327,086-token non-construction archive), well
  below the independently frozen 40% retained-mass floor. That counterfactual
  does not extend the pass budget; it shows that more exclusion passes would
  not rescue this version. Semantic-v2 therefore has no catalog, partition,
  sample report, GPU training run, checkpoint, or sealed-test result. Any
  change to the cluster representation, capacity objective, boundary margin,
  pass count, or retained-mass floor is a separately designed semantic-v3,
  not a reinterpretation of v2.
- **Execution and verification are reproducible.** The fixed v2 runner
  authenticates and reuses the 441 MiB evidence cache, completes construction
  in about 16 seconds without GPU inference, and subsequently reloads the same
  failure identity in about four seconds. An independent temporary real-data
  rebuild reproduced the failure SHA and all nine published files byte for
  byte. All 142 focused semantic/archive and shared GPT-Neo/checkpoint tests
  pass. The v2 fixtures cover fold isolation,
  conformal order invariance, content-addressed 3-by-3 success and failure
  fixtures, byte-identical rebuilds, calibration replay, and tamper rejection.
  See
  [`docs/TW-P_SEMANTIC_V2_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V2_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v2/failures/23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25/audit.md).

## Completed Milestone: TinyWorlds-P Semantic-v1 Construction Stop

`tinyworlds-p-semantic-v1` is implemented as the semantic-conjunction
successor to archive-v1. It preserves the noun-by-verb factorial experiment,
but gives noun and verb groups an independently constructed semantic meaning.
Archive-v1 remains immutable negative evidence; semantic-v1 has separate
evidence, catalog, partition, sample-report, training, evaluation, and
checkpoint contracts and provides no compatibility alias.

### Semantic-v1 Status

- **Contracts and implementation complete (2026-07-22).** The new
  `apm.data.text.tinyworlds_p_semantic` package pins the archive, tokenizer,
  complete MiniLM snapshot, construction/config identity, role anchors,
  context sampling, float32 mean-pooled normalized inference, deterministic
  semantic screens, capacity-constrained spherical clustering, audits,
  semantic topology, exact archive replay, paired controls, strict loading,
  validation-only sample reporting, group-loss ledgers, SHA-seeded paired
  bootstrap/placebo statistics, Holm correction, calibration, resume,
  selection, sealed-test, and publication boundaries. Fixed runners prepare
  evidence, build and independently reproduce a partition, and train a fresh
  seed-zero base only when the preceding artifact gate exists.
- **Pinned encoder evidence published.** The complete 11-file
  `sentence-transformers/all-MiniLM-L6-v2` snapshot at revision
  `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0` has encoder identity
  `1101bb824cee453866d6dcd2b489b29ad2c55b20de5bbaceda67f38206a21502`.
  The real CUDA/24-worker preparation published evidence
  `efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`:
  247,629 construction groups and 47,172,075 construction tokens are
  permanently reserved, 898,327,086 eligible tokens remain outside that
  slice, and 195,492 context/anchor texts were embedded. The evidence cache is
  independent of later clustering thresholds.
- **Frozen eight-cluster screen reached its declared automated stop.** Of
  1,066 nouns, 1,060 failed the strictly positive 10th-percentile role-margin
  rule. Only `pirate`, `present`, `ship`, `train`, `treat`, and `witch`
  survived. Of 394 verbs, 305 failed role margin and four failed the
  multi-sense silhouette gate, leaving 85. Six nouns cannot seed the required
  eight noun clusters and are far short of the later 32-nouns-per-cluster
  requirement, so construction failed before clustering. No threshold was
  relaxed, cluster count changed, word relabelled, or archive-v1 artifact
  consulted.
- **Failure evidence is immutable and exhaustive.** The strict failure bundle
  `ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a`
  contains all 1,460 role words, token masses, exact context counts, measured
  role margins, measured silhouettes where applicable, dispositions,
  representative exact archive contexts, and candidate-vector PCA in
  Markdown and self-contained HTML. See
  [`docs/TW-P_SEMANTIC_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v1/failures/ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a/audit.md).
- **Downstream work is intentionally inapplicable.** Because no valid
  semantic-v1 catalog exists, a partition, byte-identical rebuild, sample
  report, GPU runtime estimate, calibration, checkpoint selection, sealed-test
  opening, and base publication are not authorized. Both downstream runners
  authenticate the failure bundle and exit with controlled status 2 before
  doing work. Any change to anchors, role metric, threshold, vocabulary, or
  cluster count is a new `tinyworlds-p-semantic-v2` experiment and cannot
  reinterpret semantic-v1.
- **Verification remains focused and reproducible.** All 110 collected tests
  in the semantic, archive-native TinyWorlds-P, GPT-Neo/LoRA, checkpoint, and
  training-state scope pass; the long real-source module remains opt-in. CPU fixtures cover
  semantic screen and clustering determinism, construction exclusion,
  content-addressed success/failure audits, archive-v1 rejection, leakage,
  exact-byte reconstruction, paired-control coverage, cross-worker/run-size
  byte identity, tamper rejection, group-loss persistence, empirical-null
  gates, sample-report isolation, and interrupted/resumed training parity.
  Real archive/GPU gates remain opt-in; their measured result is retained and
  is not part of routine tests.

## Completed Milestone: TinyWorlds-P Archive-Only Calibration Stop

The completed roadmap is `tinyworlds-p-archive-v1`, tracked in
[`docs/TW-P_PLAN.md`](docs/TW-P_PLAN.md). It replaces generated TinyWorlds prose
with unmodified stories taken directly from released records in the pinned
`TinyStories_all_data.tar.gz` archive. Five noun-bucket by verb-bucket cells are
withheld from a freshly initialized eight-layer GPT-Neo base; only story text
reaches the model. All base, world, control, validation, and sealed-test sets
are derived from eligible archive entities. The original TinyStories train,
validation, and GPT-4-only text aggregates are irrelevant to this benchmark
and are not inputs. TinyWorlds-v2 external generation is parked as
non-qualifying historical evidence. LoRA/VAMP continual episodes remain
unstarted because the archive-only scratch base did not pass its publication
gates.

### TinyWorlds-P Status

- **Authoritative archive-only source decision (2026-07-21).** The pinned
  `TinyStories_all_data.tar.gz` records are the complete source universe.
  Archive entities with mechanically recoverable noun, verb, and adjective
  roles are grouped, bucketed, and assigned directly. Base train, held-in
  validation/test, five world train/validation/test splits, and matched
  controls all come exclusively from that eligible archive universe. There is
  no corpus/archive join and no dependency on any published TinyStories text
  aggregate.
- **Prior partition and calibration are superseded.** The prior 8x8 partition,
  scratch run, and 6x6 conclusion were produced from the obsolete
  corpus-intersection universe. They remain immutable historical diagnostics,
  but they are not TinyWorlds-P publication candidates and must not be resumed,
  selected, or used to set the new partition. Exact identifiers remain only in
  the historical audit documents linked below.
- **Corpus-intersection implementation purged.** The obsolete join and all
  corpus-backed paths, identities, offsets, gates, loaders, runners, and tests
  are gone. The purge-only checkpoint remains in history; every subsequently
  restored surface is archive-native and rejects old artifact identities.
- **Archive-native ingestion implemented.** The TinyWorlds-P-owned parser
  authenticates and streams the tarball once, writes exact story bytes to an
  archive-order spool, classifies bounded batches in physical workers,
  externally sorts by normalized story identity and record ID, groups every
  occurrence with its provenance and multiplicity, audits all exclusions, and
  enforces only the 95% token-weighted role-classification gate.
- **Archive-only partition artifacts restored.** Partition construction now
  derives every bucket, cell, split, and control from eligible archive groups;
  publishes exact story and uint16 token shards under
  `data/tinyworlds-p-archive/v1/`; binds documents to member, member-local
  index, record hash, and story hash; and strictly rejects old source keys.
  The CPU 3x3 fixture reconstructs source/token bytes, checks leakage and
  globally unique controls, rejects tampering, and rebuilds byte-identically
  across worker/run settings. The fixed preparation runner and opt-in real
  archive replay have been restored.
- **Canonical archive partition built and reproduced.** The strict 8x8 artifact
  is
  `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`.
  It contains all 4,966,067 eligible archive records (945,499,161 active
  tokens), passes the 99.968% token-weighted role gate, retains every agreeing
  duplicate occurrence, excludes six conflicting duplicate groups, and passes
  topology, component-visibility, split-marginal, globally unique-control,
  exact-byte reconstruction, and sealed-test isolation checks. A second build
  with 24 workers and a different external-sort run size authenticated to the
  same identity and exact `tree.json`, proving byte equality for every strict
  tree entry. The opt-in acceptance test took 39m08s end to end; the build took
  35m11s and its final parallel strict reload took 3m56s. Long real-source
  gates remain excluded from the default test suite.
- **Archive-only scratch training restored.** Memory-mapped batching,
  token-weighted accumulation, immutable complete resume states, streaming
  validation, one-shot sealed test, milestone publication, and the fixed GPU
  runner now consume only strict archive-v1 artifacts. Low-gap fallback uses a
  fresh 6x6 partition with 94/3/3 held-in splits; excessive-gap fallback uses
  10x10 with 96/2/2. Training and every validation/sealed-test batch report
  detached-safe, sparsely refreshed measured phase and pass-path ETAs. CPU
  tests prove interrupted/resumed state and trace parity, schedule and selection
  boundaries, old-resume rejection, finite evaluation, and exact evaluation
  progress.
- **Focused CPU/shared checks pass.** The 82-test TinyWorlds-P, GPT-Neo,
  checkpoint, and training-state suite passes in four concurrent groups
  (9.8s wall time); parked TinyWorlds-v2 tests are still collection-skipped.
- **GPU smoke passes.** The opt-in RTX 4090 smoke strictly loaded the real tree,
  compiled production training, wrote an interrupted update-one state, resumed
  through update two, and measured an 8.695 GiB JAX allocator peak against the
  12 GiB gate. Splitting strict semantics into assignment, provenance, and
  shard/index proof passes reduced the full smoke from 5m20s to 4m20s.
- **Archive-v1 calibration ended with the declared scientific stop
  (2026-07-22).** The fresh seed-zero 8x8 run completed epochs one and two at
  updates 18,832 and 37,664. Held-in NLL improved from 1.261707 to 1.201706,
  but mean gap was only 0.008017, so the fixed policy built the one allowed
  fresh 6x6 partition with 94/3/3 held-in splits. That fallback completed at
  updates 17,200 and 34,400; held-in NLL improved from 1.267558 to 1.206720
  and peak allocation was 8.772 GiB, but mean gap remained only 0.002802 and
  every world gap was below 0.05. The runner therefore exited with its
  controlled status 2. It did not train epochs three through five, select a
  checkpoint, open sealed test, or publish a base. The exact identities,
  metrics, output hashes, gate audit, practical gap analysis, and provenance of
  the engineering thresholds are recorded in
  [`docs/TW-P_ARCHIVE_CALIBRATION_REPORT.md`](docs/TW-P_ARCHIVE_CALIBRATION_REPORT.md).
  A deterministic
  [`validation sample appendix`](docs/TW-P_ARCHIVE_VALIDATION_SAMPLES.md)
  covers held-in base, all five worlds, and both arms of all five controls on
  both grids: 32 exact hash-verified stories selected without semantic review.
  Its two-grid generator runs concurrently in about 1.2 seconds and does not
  read sealed-test indexes.
- **Terminal policy consequence.** The archive-only implementation, 8x8
  partition, independent byte-identical rebuild, fresh 6x6 fallback partition,
  and both scratch calibration attempts are complete. The frozen archive-v1
  conjunction hypothesis did not produce the required representation gap.
  No additional regrid, gate change, historical comparison, or test-set
  inspection is authorized under this benchmark version; a different
  hypothesis requires a new versioned benchmark.
- **Historical audits retained for provenance only.** The original train/archive
  mismatch analysis is preserved in
  [`docs/TW-P_SOURCE_AUDIT.md`](docs/TW-P_SOURCE_AUDIT.md), and the obsolete
  intersection-based calibration is preserved in
  [`docs/TW-P_CALIBRATION_AUDIT.md`](docs/TW-P_CALIBRATION_AUDIT.md). Neither
  audit defines a current source, coverage gate, split, or stopping decision.
- **Test scope remains focused.** Every parked TinyWorlds-v2 test is
  collection-skipped; do not run those legacy bodies. Final verification uses
  the focused TinyWorlds-P and shared GPT-Neo/checkpoint scope in concurrent
  CPU groups. The completed opt-in archive rebuild and RTX 4090 evidence are
  retained rather than rerun as default tests. No continual LoRA or VAMP
  stream work begins from this stopped milestone.

## Parked Milestone: TinyWorlds-v2 External-Generation Benchmark

The parked roadmap is `tinyworlds-v2-gpt`, tracked in detail in
[`docs/TW-v2_PLAN.md`](docs/TW-v2_PLAN.md). The symbolic world ledger remains
authoritative for truth and scoring, while pinned external language models
produce variable-length natural TinyStories-style text through immutable,
content-addressed request/response caches. V2 does not reuse the v1
deterministic renderer or exact-token prose fitting.

### TinyWorlds-v2 Status

- **Phase 1 — direct Qwen/GPT-5.4-Mini author bakeoff: complete with an
  automated scientific stop (2026-07-19).**
  The active experiment now compares exactly two full author routes on the same
  200 already-profiled neutral briefs: Qwen 3.5 35B-A3B and GPT-5.4 Mini. Both
  models generate 200 stories and both are evaluated as possible later corpus
  authors. There is no 50-story screen, finalist expansion, or third-model
  verifier, so the paid plan is exactly 400 author requests. V4 responses are
  one-field `{"story": ...}` objects; whole-word spans, requested-feature
  realization, forbidden forms, and length evidence are derived locally. Only
  mechanically observable requirements are hard gates; semantic plot features
  remain report and blinded-audit evidence. The two routes are compared using
  the frozen TinyStories vocabulary/token
  distribution, TinyStories-8M NLL, surface statistics, cost, and a blinded
  100-reference/100-generated audit split exactly 50/50 by author. The new
  versioned path is `data/tinyworlds-v2/reference-two-route-v2/`, with its own
  raw cache and a `$15` hard cap. Focused tests for the V4 request, local
  validator, explicit two-route catalog plan, direct quality selector, exact
  2×200 preflight, and cost-cap no-secret stop pass. The already completed
  reference artifact is reused, avoiding the prior corpus scan/profile delay.
  The live preflight resolved Qwen to Alibaba and GPT-5.4 Mini to Azure. Its
  expected cost was `$0.305313`, its conservative two-attempt exposure was
  `$1.683770`, and the run therefore remained below the `$15` cap. All 400
  author calls completed for `$0.1906939625`. The corrected V2 artifact reused
  those exact cached responses without a second paid call and strictly
  validates at `data/tinyworlds-v2/reference-two-route-v2/`, manifest
  `6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9`.
  Its terminal status is `no_quality_qualified_route`; the exact balanced audit
  digest is
  `a5d9da91fe9636bda942e1f4532620e7761d4c722358f5cb0e1443fa042fff3a`.

  GPT-5.4 Mini accepted 192/200 briefs (96%) versus Qwen's 123/200 (61.5%),
  and its aggregate alignment distance was better (1.665 versus 2.777). Both
  routes passed vocabulary coverage and token-unigram JSD, with no
  alphanumeric identifier contamination. Both failed the frozen-base NLL,
  story-length, paragraph-format, and dialogue-distribution gates; GPT's
  median NLL delta was 0.905 and its median story length was 40.9% below the
  matched reference median, while Qwen's were 0.927 and 50.7%. The paragraph
  result is potentially a representation mismatch because the matched released
  references contain no counted double-newline breaks while the prompt asks
  for paragraphs, so the blinded audit must be inspected before changing that
  criterion. Phase 2 remains blocked. The 73-test core bakeoff suite passes and
  the active artifact passes strict validation, including raw request/route and
  cost-journal evidence, cost arithmetic, direct-quality/status consistency,
  and audit packet/key/HTML balance. A broader legacy replay run was
  stopped on entry to its known 20--30-minute cache fixture because the failed
  automated gate already prevents phase advancement; the complete default
  suite and zero-network V2 derived replay remain required before any future
  Phase 1 pass can be promoted.

  A deliberately small prompt-tuning review completed on 2026-07-19 without
  changing that stop. It reused 20 namespaced development briefs and their 40
  cached V4 controls, then generated exactly 20 new V6 stories per model. V6
  asks for 130--170 words and adds ordinary TinyStories cadence, opening,
  sentence, English-vocabulary, ending, and single-line paragraph guidance.
  The exact live preflight was `$0.039824` expected / `$0.174217`
  conservative under a separate `$1` cap; all 40 calls cost `$0.0244546375`.
  The promoted diagnostic is `data/tinyworlds-v2/prompt-tuning-v1/`, manifest
  `074cdacbc38e311a85de988801a8c5d2cef561fd88b19daa43640176162836f3`,
  with all outputs in `review.html`. Qwen acceptance moved from 13/20 to 14/20,
  median accepted length from 75 to 110.5 words, and median TinyStories-8M NLL
  from 2.498 to 2.133. GPT remained 20/20, moved from 90.5 to 116.5 words, and
  from 2.296 to 2.262 NLL. Only 1/20 Qwen and 4/20 GPT outputs actually reached
  the requested 130--170-word interval, and both still emitted blank-line
  paragraph breaks. Thus V6 improved alignment distance for both routes but
  did not fix the length, paragraph, or NLL gates. The 20 matched references
  happen to be long-skewed (median 172 words), so this set is explicitly
  development/review evidence and cannot qualify a route or be reused as a
  clean final holdout. The 78-test focused generation/artifact suite passes,
  as does a hardened offline reload; the loader rebuilds requests, locally derived
  evidence, measurement coverage, quality ranking, cost arithmetic, raw-cache
  attempts, and the settled cost journal. The 34-minute complete default suite
  was not repeated for this non-qualifying diagnostic and remains mandatory
  before any future Phase 1 promotion.

  A second 20-brief prompt-shape diagnostic tested V7 on 2026-07-19. V7 moves
  the concrete length/shape requirements to the end of the user message,
  removes the wrapper's redundant compression cues, requires one newline-free
  story block, and asks for 18--20 sentences, at least six connected events,
  and a soft 155--190-word target. It reused the exact V6 controls and purchased
  only 20 new stories per model. The live preflight was `$0.041028` expected /
  `$0.179473` conservative under the separate `$1` cap; all 40 calls completed
  in about 15 seconds for `$0.0296057500`. The raw diagnostic is
  `data/tinyworlds-v2/prompt-tuning-v2/`, manifest
  `838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a`.
  V7 eliminated newlines and moved median accepted length to 154.5 words for
  Qwen and 153.5 for GPT, but Qwen remained 14/20 accepted and GPT fell from
  20/20 to 18/20. More importantly, median TinyStories-8M NLL worsened from
  2.133 to 2.568 for Qwen and from 2.262 to 2.781 for GPT. The concrete
  checklist repaired surface shape while making the stories less like the
  distribution learned by TinyStories-8M.

  The first V7 quality report also exposed a comparator error: 3,393 of its
  10,000 selected GPT-4 validation stories occur in the pinned original
  TinyStories training file under NFKC/case-folded/whitespace-collapsed exact
  identity. The paid V7 output remains immutable, but that report is retained
  only as contaminated-comparator evidence. A zero-call V3 reevaluation filters
  those overlaps, rebuilds the reference profile from the remaining 6,607
  validation stories, and reuses all 80 cached V6/V7 stories and all 66 accepted
  NLL measurements. It is at `data/tinyworlds-v2/prompt-tuning-v3/`, manifest
  `50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74`,
  with every sample in `review.html`. Nine of the 20 small paired archive
  references also occur in original training, but generated-to-reference NLL
  gaps were nearly unchanged across the seen and unseen subsets. Contamination
  therefore mattered to evaluation hygiene, but does not explain the main
  mismatch. The current composite distance nominally ranks V7 first because
  its length/format match offsets other errors; no V7 route passes the hard
  acceptance, NLL, distribution, and language-feature gates, so that rank is
  not a production-prompt selection. The current 94-test focused generation,
  decontamination, replay, and artifact suite passes. The complete default
  suite was not repeated for this non-qualifying development diagnostic.

  A third 20-brief diagnostic tested the bare released prompt on 2026-07-19.
  V8 sends exactly one user message containing the archived TinyStories prompt
  followed by `Possible story:`. It has no system message, repeated
  instructions, JSON request, response schema, or added length/shape rule; the
  complete plain assistant reply is the story and all evidence is derived
  locally. The only other request fields are transport controls such as the
  pinned route, deterministic seed, output ceiling, and no-fallback policy.
  V8 reused the exact V7 stories as controls and used the decontaminated
  6,607-story validation profile directly. The preflight was `$0.034422`
  expected / `$0.152463` conservative under the `$1` cap. All 40 calls finished
  in 17 seconds and cost `$0.0155166000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v4/`, manifest
  `362a0c85c7722fbaf36120eaa5479285edb798bc067d8f7c7fd41631571e2bb0`.

  Both bare-prompt routes accepted 20/20 and realized all three required word
  roles. GPT median NLL improved from V7's 2.781 to 2.185, while Qwen improved
  from 2.568 to 2.475; the decontaminated validation median is 1.347. The bare
  prompt also restored the released prompt's compression and paragraph
  behavior: GPT fell to an 80-word median and Qwen to 113.5 words, versus 138
  in validation, and every new story used paragraph breaks. GPT's NLL gain is
  strong evidence that the wrapper/checklist caused part of its mismatch, but
  neither bare route passes: both still fail NLL and token-distribution gates,
  GPT is 42.0% short, and Qwen is 17.8% short with longer pooled sentences.
  The composite score retains V7 for both routes because its length/shape fit
  outweighs V8's acceptance and NLL gains. That descriptive choice is not a
  production selection. The active artifact validates from persisted evidence
  and the complete focused V2 generation/comparator suite passes; the long
  default suite was not repeated because this diagnostic cannot promote Phase
  1.

  A fourth diagnostic added exactly one sentence to V8:
  `Aim for about 130 to 150 words.` V9 otherwise preserves the same single user
  message, provider seed, route, technical controls, plain-text response, local
  validation, V8 control outputs, and decontaminated comparator. Its preflight
  was `$0.034641` expected / `$0.153339` conservative; all 40 calls generated
  in 15 seconds and cost `$0.0220921000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v5/`, manifest
  `1605d21acff2647fe4be456a627653f606b7e4e90c7241d3d552ebe513430c73`.

  The cue repaired length for both models but exposed a route-specific
  tradeoff. Qwen moved from 113.5 to 147 median words, improved median NLL from
  2.475 to 2.339 and token JSD from 0.324 to 0.293, and reduced composite
  distance from 2.644 to 2.550; it is the better Qwen prompt despite falling
  from 20/20 to 18/20 when two stories omitted required word forms. GPT moved
  from 80 to 128 words and improved token JSD from 0.352 to 0.314, but median
  NLL worsened from 2.185 to 2.381. Its composite distances are effectively
  tied, with bare V8 retaining the nominal lead by 0.0008. V9 therefore passes
  the story-length band for both routes, but neither route passes the NLL,
  token-distribution, sentence-length, paragraph-serialization, or dialogue
  distribution gates. The strict artifact reload and 106-test focused suite
  pass; the long default suite was not repeated for a non-promotable diagnostic.

  A matched LoRA learnability sidebar completed on 2026-07-20 without changing
  the Phase 1 stop. It trained rank-8 adapters for 512 updates on the same eight
  child-to-badge facts and four badge-to-place rules in 24 documents per arm;
  only the prose after each canonical leading evidence sentence differed. The
  arms were a decontaminated official-TinyStories control, Qwen 3.5 35B-A3B,
  and GPT-5.4 Mini. The 72 author calls finished for `$0.0392434000`; the
  promoted artifact is `data/tinyworlds-v2/reasoning-sidebar-v1/`, manifest
  `59200a624dcc8e2afe4cfcdb720d22724184eb97797d7da8208cf0b527d797fe`.
  All three adapters reduced their own training-corpus NLL to at most 0.027,
  and a zero-training exact-clause follow-up scored 100% on all eight literal
  facts and all four literal rules for every adapted arm. Nevertheless,
  held-out two-wording test recall was 25.0% for the TinyStories and Qwen arms
  and 31.2% for GPT, while every arm scored exactly 25.0% on one-hop
  fact-plus-rule questions (four-choice chance is 25%). The clause diagnostic
  is `data/tinyworlds-v2/reasoning-sidebar-v1-clause-probe/`, manifest
  `1d1d8a7921e4ab74b4b23d57266da776d06bf01b3effe5fecc6a92ed5a318b6f`.
  Thus the LoRAs stored literal continuations but did not expose stable
  paraphrase-invariant bindings or compositional knowledge. Because the
  in-distribution control fails the same transfer test, this sidebar cannot
  attribute that failure to Qwen/GPT distribution mismatch. It is exploratory
  evidence only and does not advance Phase 1. Both artifacts strictly reload,
  and the 17-test sidebar/shared-workflow/candidate-scoring suite passes. The
  34-minute complete default suite was not repeated for this non-promotable
  diagnostic.

  The first completed direct artifact at
  `data/tinyworlds-v2/reference-two-route-v1/` is preserved as an over-strict
  validator diagnostic. It spent `$0.1906939625` for all 400 responses but
  incorrectly hard-gated semantic labels such as moral, conflict,
  foreshadowing, and twist using lexical patterns. V2 reinterprets the same
  immutable cached responses without new paid generation: only whole-word
  constraints, safety/length, and quoted dialogue are hard-local evidence;
  semantic plot labels remain reported human-audit judgments.

  The earlier seven-route implementation remains immutable historical evidence.
  It covers exact normalized-content
  source cohorts, deterministic 16-process surface profiles and persisted GPU
  NLL measurements, semantic route identity separated from exact catalog
  provenance, versioned behavior-changing transport headers, public catalog
  revalidation before every paid batch (with a passing public-only live
  resolver smoke on 2026-07-18), append-only completion/stats caching,
  and independent billing observations. Its inclusive runtime cap uses both a
  nonblocking cross-process lease and an fsynced write-ahead reservation/
  settlement journal, persists the complete route lock and sanitized zero-BYOK
  authorization with each reservation, records concurrent pre-POST
  cancellations without charging them, reconciles historical locks after a
  crash, and stops without reposting on ambiguous billing. BYOK also fails
  closed: the inference key cannot prove account state, so production requires
  either a distinct `OPENROUTER_MANAGEMENT_API_KEY` zero-key check or an
  explicit canonical manual attestation valid for at most 24 hours; every
  returned completion/stats record must still prove `is_byok=false`. The
  artifact boundary includes exact generator/verifier cost attribution, route,
  audit, and raw-cache evidence, strict cross-artifact semantic validation, and
  a zero-network byte-for-byte derived replay. Targeted offline unit and
  integration gates, the pinned-source integration, the real-GPU NLL smoke,
  the public-catalog resolver smoke, and focused cost/recovery/replay checks
  pass. The complete offline default suite passes 753 tests with one optional
  skip and eight resource-marked deselections in 2,045.40 seconds; peak RSS was
  4,085,656 KiB. After explicit zero-BYOK confirmation, the production runner
  authenticated and profiled the pinned sources, measured them on the GPU,
  resolved the live routes, and reached its exact preflight. Expected spend was
  `$3.439507`, but the required two-attempt exposure was `$20.020653`, above
  the fixed `$15.000000` cap; the 800-request GPT-5.4 verifier reserve alone was
  `$17.544000`. Per contract, the inference key was not read and zero completion
  POSTs, charges, or generated samples occurred. The promoted stopped artifact
  is `data/tinyworlds-v2/reference/`, with manifest
  `28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9`.
  Strict semantic validation passes and zero-network replay reproduces all 31
  derived files (101,081 bytes). No human audit exists and the Phase 1 gate did
  not pass. A redundant post-artifact default-suite rerun was intentionally
  interrupted at 85% before repeating the known 20--25-minute cache fixture;
  it had no failures, no code changed after the complete pre-run pass, and the
  artifact validation/replay gates passed independently.
  Diagnostic previews treat the external models as synthetic-story authors,
  not as teachers whose behavior is being distilled. The original v1 preview
  at `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v1/` (manifest
  `1ddba6e0862de3e416b4ce21538f5471723e823d6c39c5a32da27a0ea72596b6`)
  is retained as an archived request-contract experiment because its
  `enforce_distillable_text` restriction prevented a meaningful comparison.
  A corrected v2 attempt removed that restriction but was interrupted after 13
  paid requests: its first Qwen response emitted 5,138 hidden reasoning tokens
  despite the 512-token visible-output bound, and exact provider spend reached
  `$0.008248631`. The completed v3 preview explicitly disables optional
  reasoning for Qwen and Gemini and uses the remaining `$0.041751369` of the
  separately authorized `$0.05` cumulative cap. Its promoted artifact is
  `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v3/`, manifest
  `6e1aa9697d8e62263a49c6bc8d22aa22bcb568ca4e551e68b75c727ab063d9f0`.
  All 21 outcomes replay with zero network; 5 passed the current deterministic
  gate (Mistral 1/3, Gemini 3/3, GPT-5.4 Mini 1/3). Provider-reported v3 cost is
  `$0.0061824395`; one unknown-cost timeout is conservatively charged
  `$0.00063045`, for v3 ledger exposure of `$0.0068128895`. This small preview
  remains diagnostic-only and ineligible for route selection. Preliminary inspection
  also shows that the current mechanical gate confounds story quality with the
  model's self-reported evidence schema: Qwen returned three coherent stories
  in a different JSON layout, while two strong GPT stories failed exact
  evidence-quote matching and weaker Gemini prose passed. Before any full
  funnel, separate locally derived story checks from response-metadata validity.
  That correction is now implemented by the active V4 direct comparison. A
  post-run safety audit also closed partial-stats, cache-only recovery,
  no-replace promotion, and cost-evidence validation gaps without making any
  further provider requests. The focused preview/generation suite now passes
  107 tests, and both archived v1 and promoted v3 still validate and replay
  byte-identically under the hardened validator.
- **Phases 2–7: blocked by Phase 1.** Counterbalanced world bibles, natural
  training stories, probes, calibration, the eight-task pilot, and scaling are
  specified in the v2 tracker but are not authorized early. Phase 3 has a
  second mandatory human stop before full-corpus generation.

### TinyWorlds-v1 Gate History

- **Phase 0 — TinyStories post-mortem: complete (2026-07-17).** One exact
  retrain published reusable adaptation artifact
  `0866c521d7accc2576150b5a2cc9b1e4bb9067bcb6403c8da5262f8419b09eef`.
  Reload-only evaluation produced all three paired conditions and the report
  at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-postmortem-seed0-18fcf925db5f`.
  All 1,173 tensor checksums remained identical, the report reproduced
  byte-for-byte from the completed result, and the historical report remained
  byte-identical. The gate suite passed 473 tests with one optional skip and
  five resource-marked deselections.
- **Phase 1 — symbolic TinyWorlds generator: complete (2026-07-17).** The
  calibration and pilot bundles strictly load and rebuild byte-identically at
  digests `ae532f527f9cb35702734aaa819453127f5c30faaf4994436e69a43d2c023c27`
  and `0f24a708301f77b1af8a798869a5c76f8ae7f47205caa7c8cb66b6447d73ca32`.
  The pinned 1,924,281,556-byte original corpus matched SHA-256
  `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`;
  two production re-streams found zero hits among all 144 generated lexical
  forms. All 48 queries have unique graph answers and canonical proofs, bridge
  support requires both branches, all six holdout axes are disjoint, and the
  hardened loader rejects consistently rehashed provenance, topology, proof,
  candidate, revision, story, and capacity tampering. The gate suite passed
  504 tests with one optional skip and five resource-marked deselections; both
  marked tokenizer/novelty integrations passed.
- **Phase 2 — deterministic rendering: complete (2026-07-17).** The pinned
  tokenizer materialized calibration and pilot bundles with exact counts
  5,248 stories/3,072 semantic groups and 10,368 stories/6,144 semantic
  groups. Their rendered digests are
  `ad4a69713060f7d661e92f7415fc4a9ddaffc4852e000cb7c8ba49c3872e4750`
  and `3ad30374480cbeff2587d722b632702de5b097f97f4f9dd651195e3844015bc3`.
  A second invocation strictly reconstructed both trees as `verified`, kept
  the combined tree digest
  `387f4fe2d03b5c868446b9a1884e9134d064b52f4e5e2130796eda1ed53faca0`,
  and reproduced the canonical materialization result byte-for-byte.
  Accepted immutable query-group plans, split-specific templates, exact token
  boundaries and masks, cues, candidate leakage, symbolic alignment, and
  deterministic fallback provenance all passed. The focused gate passed 30
  tests, the marked real-tokenizer gate covered all eight query kinds, and the
  complete default suite passed 516 tests with one optional skip and five
  resource-marked deselections.
- **Phase 3 — candidate scoring and knowledge evaluation: complete
  (2026-07-17).** Frozen, hard-node, and arbitrary-coefficient scorers share
  one four-candidate path with active-token normalization and exact
  microbatch equivalence. The shared hard tensor, both single-run EBT
  refinements and their soft variants, every metric/aggregation axis,
  validation-only parent search, four-role counterfactuals, selected-only
  commit, and immutable resume chunks passed audit. Executed gates cover
  synthetic answer selection, one-hot soft/hard equality, incompatible
  revision answers, incomplete cross-branch hard support, suffix-blind
  routing, and all 11 methods over a real bounded two-edge knowledge stage.
  Final-budget reloads revalidate full execution identity, checkpoint
  histories are exact schedule prefixes, and full-state chunks reject stale
  initial states and dangling symlink targets. The combined focused CPU gate
  passed 77 tests; the complete default suite passed 518 tests with one
  optional skip and five resource-marked deselections.
- **Phase 4 — four-task calibration: complete with controlled scientific stop
  (2026-07-17).** The canonical RTX 4090 runner evaluated exactly three
  validation configurations: the 24-fact baseline, 12 facts, and 36 facts,
  each with 32 exposures/fact, 1,000 updates, rank 8, and hard distractors.
  None passed the complete validation gate, so the fixed ladder stopped
  mechanically at `facts_axis_has_no_passing_configuration`. The immutable
  stopped result is at
  `results/language_cl/tinyworlds-v1/knowledge-graph/calibration-stopped-seed0-e314a9704528`
  with result SHA-256
  `e314a9704528bb8a8133bb4a1465b8be10922df390cbecfb23d33933411ab4e3`.
  All trials retained a 1,024/1,024 exact-KG ceiling, zero committed-node
  drift, and 128/128 old-context consistency. However, frozen novel-binding
  was 64/64 rather than the required 20–30%, leaving zero direct-recall lift;
  revision-node and paired-revision consistency were 0/128 in every trial.
  Independent one-hop accuracy was 0/64 for the 24- and 12-fact trials and
  64/64 for 36 facts, which did not repair the other failed gates. The maximum
  recorded allocator peak was 6,738,299,904 bytes (6.276 GiB), below the
  enforced 12 GiB target. Strict result reload and complete promoted-bundle
  validation pass. The stopped bundle intentionally contains no calibration
  profile and no locked-test artifact. The final complete default suite passes
  537 tests with one optional skip and five resource-marked deselections.
- **Phase 5 — eight-task pilot/report: not launched.** Phase 4 produced no
  passing `calibration_profile.json`; `pilot_authorized` is false and the
  held-out calibration test remained unopened. The TinyWorlds-v1 contract
  therefore forbids launching the pilot. This is the prescribed visible
  scientific outcome, not an implementation failure.
- **Interactive calibration playground: implemented (2026-07-18).**
  `notebooks/tinyworlds_playground.ipynb` provides a read-only view of the
  symbolic world, rendered stories and alignments, proof depth, cue variants,
  hard and continuous support, saved candidate NLLs, gate evidence, and parent
  transfer. Its supporting module strictly reloads the promoted Phase 4 result
  and can generate small worlds through the production symbolic generator,
  tokenizer, and renderer. Fresh samples never inherit saved model scores;
  only the exact canonical seed-0 demo may be joined to validation evidence.
  The notebook does not train, tune, open the held-out test split, or imply
  that Phase 5 is authorized. A clean-kernel execution passed end-to-end, and
  the complete default suite passed 550 tests with one optional skip and five
  resource-marked deselections.
- **TinyWorlds-v1 language-distribution audit: failed (2026-07-18).** The
  renderer is mechanically tokenizer-valid but far outside the frozen
  TinyStories model's training distribution. Entity surfaces are `N` plus 12
  hexadecimal characters and average 8.42 BPE pieces, versus 1.21 for names in
  a matched corpus sample. Generated stories have token-distribution
  Jensen-Shannon divergence 0.702 from TinyStories, while two independent
  TinyStories samples differ by 0.014; an estimated 68.5% of story tokens are
  exact-length padding. Frozen-model scoring on 128 matched 256-token examples
  produced mean NLL 6.780 for TinyWorlds and 1.460 for original TinyStories,
  about a 204-fold perplexity ratio. Raw task IDs, hash marks, symbolic
  predicate labels, and rule-variable labels also enter visible prose, while
  candidate-specific filler contaminates answer NLL. The exact whole-word
  novelty audit did not measure these properties. Preserve v1 as historical
  evidence, but do not interpret its learned-model results as a clean test of
  knowledge-graph acquisition.

### Parked TinyWorlds-v2 Follow-up

1. Preserve the promoted stopped calibration bundle as the terminal
   TinyWorlds-v1 result; do not modify it or launch its Phase 5 pilot.
2. Preserve the valid `blocked_by_cost_cap` Phase 1 artifact and its exact
   `$3.439507` expected / `$20.020653` conservative evidence; do not overwrite
   or reinterpret it.
3. Preserve the completed V2 direct-bakeoff artifact, its 400 raw responses,
   exact `$0.1906939625` bill, failed quality metrics, and balanced audit. Do not
   overwrite the scientific stop or silently relax its thresholds.
4. Inspect `data/tinyworlds-v2/prompt-tuning-v5/review.html`; it shows all 20 V9
   outputs per model beside the exact V8 controls and archive stories, and
   exposes the exact one-sentence request difference.
5. Preserve the route-specific result rather than averaging it away: V9 is
   better for Qwen on length, NLL, token JSD, and composite distance, while GPT
   trades its V8 NLL advantage for correct length and nearly identical
   composite distance. Neither prompt/route cell passes the complete gate.
6. Do not buy another prompt cell until the V5 review is inspected. Any further
   change must state which remaining failure it isolates; do not reintroduce a
   system message, JSON, or a narrative checklist as a bundled intervention.
7. Do not generate world bibles. Phase 2 remains blocked because neither route
   passed the automated Phase 1 gate; human inspection cannot retroactively
   turn the current artifact into a passing one.
8. Add zero-network derived replay for a future passing two-route result before
   treating the Phase 1 gate as passed; the current strict validator already
   authenticates configuration, planned V4 requests, result partitions, local
   evidence, and measurement coverage.
9. Run the complete default suite after any corrective implementation and
   before a future Phase 1 promotion; retain the fast focused suite as the paid
   boundary preflight.

### Deferred Alternative

- A domain-pretrained, TinyStories-scale base model is recorded in
  `docs/TINYWORLDS_DOMAIN_PRETRAINING_NOTE.md`. It would learn KG-oriented
  language from many disposable worlds before continual evaluation on wholly
  held-out worlds. This is a preserved research option, not active work or
  authorization to modify the v1 artifacts.

## Completed Foundation: Language-Model VAMP Proof of Concept

- The completed language-model VAMP foundation and its phase gates are
  recorded in `docs/LM_VAMP_EXECUTION_PLAN.md`.
- Phases 0-10 are implemented: the architecture/build contract is recorded,
  generic immutable graph topology backs the migrated dense-MNIST memory,
  the typed plain-JAX GPT-Neo core passes its CPU correctness and overfit
  gate, and fixed-capacity pathwise LoRA passes its isolation, masking, and
  candidate-gradient gate.
  The verified TinyShakespeare path now includes pinned text preparation,
  deterministic character batches, immutable clipped-AdamW training,
  schema-v1 safetensors checkpoints, and uncached greedy generation.
  Strict TinyStories-8M conversion and the complete offline Hugging Face
  residual/logit/NLL/generation parity ladder pass at the pinned revision.
  Prefix/suffix language contracts, exhaustive normalized-prefix routing,
  and frozen-base content-key derivation pass their masking and task-identity
  exclusion gates.
  The immutable language transition passes a real two-task character-
  permutation run with stable base, prior-run state, and old-node logits.
  Normalized Hopfield retrieval passes real derived-key, capacity masking,
  independent-batch, temperature, top-k, and evaluator-metric gates. EBT
  refinement now optimizes independent per-example node logits, supports all
  four prescribed starts, returns both soft and hard results, and passes its
  decreasing-objective, masking, equivalence, and immutability gates.
  Phase 10 adds deterministic character-permutation, corpus-region,
  stable-hash, and pinned TinyStories topic curricula; document-safe packing;
  the complete four stored/five routed baseline matrix; stored/routing
  forgetting, regret, transfer, logical/runtime memory, synchronized timing,
  random-control confidence intervals, and enforced peak-memory targets; and
  deterministic standalone language reports with all prescribed artifacts.
  TinyStories corpus loading now verifies the pinned files before bounded
  streaming selection, retaining only selected stories plus compact content
  identities. Evaluation and routing support explicit microbatches, reuse
  suffix scores across routers, and cache the shape-stable EBT optimizer
  executable. The real pinned validation aggregate cannot supply the original
  1,000 examples for every topic under the fixed two-concept-plus-margin rule,
  so the measured single-GPU preset uses 10,000/128/128 train/validation/test
  stories and 128 probes/examples while preserving the source, split, topic,
  and deterministic hash-selection contracts.
  Report samples are now completed before the final allocator peak is enforced
  and are reused by the report-only projection. EBT refinement now retains
  aligned node-probability, path-edge-coefficient, and objective trajectories;
  reports preserve a deterministic final-task representative trace as JSONL
  and render four coefficient heatmaps plus an objective curve.
- The target is a shared plain-JAX GPT-Neo base with immutable pathwise LoRA
  memory, a TinyShakespeare smoke path, converted TinyStories-8M weights,
  exhaustive/Hopfield/EBT task-free addressing, and reproducible continual-
  learning reports.

## Current Phase and Immediate Gate

The engineering implementation and resource-backed Phase 10 validation are
complete on the local RTX 4090. Scientific follow-up is now focused on transfer
and routing quality rather than feasibility.

Measured status on 2026-07-16:

- the pinned TinyShakespeare corpus is present and its locally trained
  5,000-step base checkpoint improved validation NLL from 4.20647 to 1.61060;
- both pinned TinyStories V2/GPT-4 aggregates match their prescribed sizes and
  SHA-256 digests, and the TinyStories-8M conversion is present with parameter
  checksum `cdb66d6fe8377d09c43db0631fecb7265216d4383232bff6d0d5f7d0047bf5de`;
- strict prepared-source conversion and the full offline Hugging Face/JAX
  tokenization, residual, logit, NLL, and greedy-generation parity ladder pass;
- the complete TinyShakespeare character-permutation report is at
  `results/language_cl/tinyshakespeare/character-permutation/standard-seed0-a7bd7d1479ba`.
  At the final stage and primary 64-token prefix, exhaustive and both EBT
  routers reach 100% task-node accuracy, Hopfield reaches 86.33%, and the
  deterministic random control reaches 23.44%. The measured allocator peak is
  2,203,208,704 bytes;
- a real 2.2 GB TinyStories streaming preflight filled all four task splits at
  10,000/128/128, retained 13,815 root-validation stories, completed in 4m36s,
  and used no swap; and
- the full stable-hash negative-control report is at
  `results/language_cl/tinyshakespeare/stable-hash/standard-seed0-5f8f82a979a3`.
  Aggregating all four tasks at each prefix, exhaustive, both EBT routers, and
  the deterministic random router have 95% intervals containing 25% chance at
  every prefix. Hopfield contains chance at prefix 32 and is below chance at
  prefixes 64 and 128. The five above-chance task-slice audit flags disappear
  under the prescribed four-task aggregation; two also occur in the
  deterministic random control. The audit found no cross-split macro-document
  overlap and no above-chance aggregate leakage signal; and
- the full TinyStories topic report is at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-seed0-9f715620e7c2`.
  It completed all four 2,000-step tasks and the benchmark/sample workload in
  about 55 minutes. The final allocator peak was 8,349,717,248 bytes (7.776
  GiB), below the enforced 12 GiB target. At the primary 64-token prefix,
  independent root LoRAs were best at 1.33617 NLL; sequential LoRA reached
  1.35269 with 0.02031 forgetting; and VAMP oracle reached 1.36035 with zero
  stored forgetting. Exhaustive and Hopfield routing reached 46.48% and 45.90%
  exact task-node accuracy, while Hopfield was 8.9x faster warm. Both inherited
  children finished about 0.05 NLL behind independent training, and EBT did not
  improve on plain Hopfield. The process loaded the schema-1 report surface
  before commit `95e0cf4`, so its aggregate metrics are complete but the newer
  stepwise EBT coefficient traces require a rerun if they are needed.

Both canonical runners now emit the manifest, seven JSONL metric families,
address confusion, three aggregate metric charts, graph, five EBT routing-
dynamics charts, samples, and standalone HTML under a content-addressed run
directory. Offline bounded tests exercise the same training, all nine methods,
measurement, trace capture, sample generation, and report writer without
substituting for the full-resource measurements. The latest default CPU gate
passes 376 tests with one expected optional-dependency-boundary skip and two
resource-marked tests deselected. Running those two integration tests
explicitly also passes both against the prepared local artifacts.

## Deferred Stage-1 FabricPC Work

Stage-1 dense-delta VAMP over PermutedMNIST and digit-incremental MNIST remains
available with VAE and FabricPC backends. Its reports, fixed-epoch schedule,
and observed-energy-convergence schedule are retained, but further FabricPC
benchmark development is parked until the LM VAMP milestone changes priority.

Deferred gaps are:

- pad and mask FabricPC evaluation, observed-energy, reconstruction, tail, and
  addressed-winner batches to prevent shape-driven JAX recompilation;
- run the full ten-digit convergence benchmark (current verification covers a
  deterministic two-digit VAE run and one-digit FabricPC smoke run);
- inspect stopping epochs, reconstruction quality, and train/test address
  confusion; and
- compare that run with the existing fixed ten-epoch FabricPC checkpoint,
  especially early-digit routing to later parameter nodes.

Observed energy remains model-specific and supports within-model stopping and
addressing only. Reaching the epoch limit retains the best state and reports
`max_epochs`; it is not convergence.

## Known LM VAMP Gaps

No required engineering surface or resource-backed gate from the execution
plan remains incomplete. Research priorities are to replace prefix-NLL-only
parent selection or add a root fallback for negative transfer; measure whether
whole-story topic cues are visible in each evaluated prefix; correct the
fantasy-biased content keys; and explain why EBT sharpens its addressing while
losing suffix quality, especially at prefix 128. Add incremental progress
events before another long resource run. Rerun the full TinyStories report only
if the post-`95e0cf4` stepwise EBT coefficient artifacts are required. The
validated public routing wrapper performs host-side postcondition checks and is
intentionally timed directly; extracting a separate outer-JIT-compatible
validated factory remains optional optimization.
