# ImageNet-R-50 Full-Union LogT Prediction Integrator — Ungated Protocol

## Question

How close can the direct LogT prediction-integrator architecture come to the
offline joint-IID LoRA ceiling on ImageNet-R-50 when every hierarchy parent is
retrained on the complete represented training union?

This protocol is a prospective continuation of the completed full-union v2
development run. It removes every accuracy threshold and control-margin stop.
Accuracy can describe the outcome but cannot prevent later phases from running.
The offline joint-IID LoRA result is the primary comparison and the pinned local
E2-LoRA reproduction is secondary; neither is an acceptance gate.

## Frozen continuation choices

The v2 run
`7f8ac3ef574fe7ec3a2097c3a4b8a8ed13c5c1e4f34a856d69b6c32108a6a946`
kept the 6,000-image test partition sealed and stopped after task-16
development. Its exact clean-development record is bound by SHA-256
`89dabe2a46d1aff985c22ad56897de0839d39585c52e195013e38b633c8cf616`.
Before opening the test partition, this continuation freezes:

- `scores` as the feature family already selected by v2;
- H=2,048 as the historical reservoir, because it had the highest task-16
  persistent validation accuracy among H in {512, 1,024, 2,048}; and
- full-union parent retraining, rank-16 LoRA, the optimizer schedules, seed
  1993, and the existing deterministic 19,200/4,800 development split.

The prior three-point task-16 margin, 79% task-50 floor, five-point task-50
margin, and automatic replication trigger are deleted rather than set to zero.
Fresh integration and static unions remain useful diagnostics, but they do not
select the final comparison or stop execution.

## Hierarchy and integrator

One node may occupy each binary-counter level. Fifty four-class tasks create
47 full-union parents and end with live intervals 49–50, 33–48, and 1–32.
Every parent is initialized as a fresh zero-effect rank-16 adapter, unions its
children's disjoint affine classifier rows exactly, and trains for five epochs
on every fit identity represented by its children. No fixed-K parent sample
limit exists.

The persistent residual MLP observes the stable six-slot `scores` tensor and
trains for four epochs per arrival on every current-task image plus the frozen
H=2,048 class-stratified historical reservoir. Labels, task IDs, true nodes,
and interval metadata cannot enter task-free prediction. The task-2/4/8/16
fresh fits, feature-family diagnostic, hierarchy controls, and task-50 fresh
fit are retained only to explain behavior.

## Execution boundary

The workflow may stop only for an operational or integrity failure: invalid
manifests, split leakage, non-finite training, incorrect topology, failed
artifact reuse, checkpoint corruption, or unavailable required hardware/data.
It does not stop for accuracy.

Clean development extends through all 50 tasks. The locked run then rebuilds
the hierarchy and persistent integrator from all 24,000 training images. All
50 persistent checkpoints and the final frontier are sealed, and the request
ledger must prove that zero test identities were opened, before evaluating the
6,000 test images.

## Comparisons and reporting

The primary table reports Last and Incremental Accuracy for:

1. the full-union persistent LogT integrator; and
2. the common-split offline joint-IID rank-16 LoRA ceiling.

The common-split local E2-LoRA reproduction is reported secondarily, with its
published result clearly separated as external context. Differences are plain
descriptive measurements, not pass/fail decisions. Static unions, fresh fits,
task-level matrices, lineage, resource accounting, and exact cache/model work
remain diagnostic appendices.

The one supported entry point remains:

```bash
scripts/vision/imagenetr/run_integrator_local.sh
```
