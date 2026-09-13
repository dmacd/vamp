# Task-50 checkpoint diagnostics

These are inference-only, post-hoc diagnostics of the existing ImageNet-R-50
rank-16 checkpoints. No model was retrained, selected, or replaced. The
original SRT report and its raw benchmark curves remain the reporting surface;
four appended sections examine probability calibration and replay retention.

Protocol/run:
`6f23cffe01196eab7b2e36451d7ea3e2ab40c5360a5864aa857fdcbd3225d03c`.
Sealed result:
`5fbcf4ea2b16ad4b70396524a4468d54400b30e8bc117aa5215d5131dbbdef50`.
Implementation/protocol commit before inference: `79a760b`.

## Findings

| Task-50 condition | Raw test accuracy | Raw NLL | Out-of-fold NLL |
| --- | ---: | ---: | ---: |
| Offline joint IID, replay-schedule matched; three-seed mean | 80.828% | 1.0463 | 0.8386 |
| Uniform, standard | 80.917% | 0.8951 | 0.8201 |
| SRT, standard | 73.967% | 1.2682 | 1.1493 |
| Uniform, strict | 80.750% | 0.8974 | 0.8319 |
| SRT, strict | 75.717% | 1.2217 | 1.0740 |

All replay rows use H=4,096, old fraction 0.8, interval unit 8, and seed 1993.
Offline seeds are 1993, 1994, and 1995. NLL uses natural logarithms. A positive
temperature leaves every class prediction and accuracy unchanged. The mean
offline-minus-standard-uniform NLL gap drops from 0.15122 to 0.01845: an 87.8%
reduction, largely explained by a common confidence scale. The calibrated
offline NLLs are 0.82066, 0.84641, and 0.84864; seed variation remains relevant.

For standard SRT, 6,239 training images last had correct-class probability at
least 0.9 but were not reviewed during tasks 40-50. The final clean-view
accuracy is 90.880% under SRT and 99.359% under uniform on the same identities
(569 versus 40 errors). The broader 7,563-image unrevisited group accounts for
795 of SRT's net 1,020 additional training errors. Its 240 most-presented
images score 85.00% under SRT and 84.58% under uniform.

Strict SRT is a counterexample to an explanation based only on poor overall
training fit. Its full clean-training accuracy is 96.913%, close to uniform's
97.288%, while test accuracy still differs by 5.033 points. Strict SRT improves
its 240 most-presented images to 92.08% versus uniform's 82.50%. Nevertheless,
its 4,221 stale/high-confidence images score 94.835% versus 99.597%. The evidence
supports testing replay allocation and stale quality estimates, but does not
isolate their effects on held-out accuracy.

## Boundaries

- Five fixed class-stratified folds contain 1,200 test images each. Each scalar
  temperature fits 4,800 other images. A scored image's label never fits its
  own temperature. All 35 fits are interior to the numerical bounds 0.01-100.
- The test set was inspected previously. These are cross-fitted diagnostic
  numbers, not new benchmark scores, training seeds, or untouched confirmation.
- Cohorts use SRT history, then apply the same identities to both checkpoints.
  Final clean-training scores are not generalization estimates. Comparing them
  with prior augmented/pre-update confidence does not isolate forgetting.
- `true_logit_margin` is in the original, unscaled logit units. Both raw and
  calibrated prediction records retain that same reference margin; only the
  probability-derived quantities change with temperature.
- The next causal control is unrun: preserve realized batches and old/current
  counts while separately testing maximum review gaps and capped repeated-image
  allocation. Strict-profile gains on difficult training cases must not be
  ignored when interpreting that test.

## Evidence and rebuilding

The resolved configuration and method are in
`configs/vision/imagenetr/checkpoint_diagnostics.yaml` and
`docs/imagenetr50_checkpoint_diagnostics_protocol.md`.
Default workflow:

```bash
bash scripts/vision/imagenetr/run_checkpoint_diagnostics_local.sh
```

`status` is read-only; `report` rebuilds the existing SRT report from completed
evidence without inference. Full local reruns require the original dataset,
base weights, source checkpoints, and source run artifacts.

Published material includes the protocol, code/environment/source manifests,
fold identities, replay histories, eight compact analysis Parquet tables,
collection result/job manifests, verification records, and runner logs.
The existing SRT report directory contains matching figures, CSV/Parquet
exports, and compact JSON summaries. Per-image exports retain all 42,000 test
predictions and 48,000 paired training records; no selected cases are dropped.

Large raw logit chunks, model weights, and redundant per-image JSON exports
remain local. Analysis Parquet tables, not the report's convenience CSV/JSON
copies, are the result's hash authority. The source snapshot and verification
record also hash the local raw chunks so their reuse can be audited.

Measured collection work is 138,000 forward image paths, zero optimizer
updates, and 587.759 seconds inside collection chunks. That timing excludes
model restoration, setup, report generation, and extra audits. All 42,000
original test predictions match exactly; the maximum independent float64 NLL
difference from saved FP32 loss is 0.000001104. All model parameters and buffers
remain unchanged.
