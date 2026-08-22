# ImageNet-R-50 recursive-router handoff

Run: `e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0`

Status: complete on the preregistered capacity-failure branch. The test split
was not opened, and the B/C recursive matrix and extra seeds were not run.

## Main result

| Row | Router | Routed validation accuracy | True-node oracle | Oracle gap |
| --- | --- | ---: | ---: | ---: |
| A0 | centroid | 64.562% | 97.646% | 33.083 points |
| A1 | R0 | 59.083% | 97.646% | 38.563 points |
| A2 | R1 | 58.729% | 97.646% | 38.917 points |
| A3 | R3 | 57.750% | 97.646% | 39.896 points |
| A4 | R2 | 59.458% | 97.646% | 38.187 points |

R3 minus R1 was -0.979 percentage points over the same 4,800 validation
images. The paired image-level bootstrap 95% interval was -1.812 to -0.167
points (224 R1-only correct images and 177 R3-only correct images).

Neither main architecture came within the required 1.0 point of the oracle.
The negative result therefore localizes the failure before recursive router
maintenance: the tested frozen query representations and score families do not
have enough flat full-data routing capacity. Adapter-response features did not
help under the predeclared R3 construction.

## Audit result

- 19,200 router-fit and 4,800 router-validation identities were frozen.
- `test_images_used` is zero.
- All 13 scheduled jobs completed; none failed.
- B4 and B9 each reused all 12 smoke nodes, created zero nodes, and executed
  zero new optimizer steps in the terminal reuse proof.
- The sealed inference inventory is byte-identical before and after.
- Leaf and inference-parent optimizer steps are both zero.
- The final workflow rerun returned the same immutable protocol without
  repeating completed work.

## Files for analysis

- `reports/REPORT.md` and `reports/REPORT.html`: human-readable outcome.
- `reports/stage_metrics.*`: aggregate validation metrics and confidence
  intervals.
- `reports/task_accuracy_matrix.*`: per-task aggregate accuracy.
- `reports/paired_r3_minus_r1.*`: paired R3-versus-R1 comparison.
- `reports/resource_accounting.*`: router work and memory accounting.
- `diagnostics/capacity_gate.json`: exact preregistered gate inputs.
- `diagnostics/reuse_proof.json`: zero-work reuse evidence.
- `diagnostics/smoke.json`: eight-task algebra and policy smoke outcome.
- `protocol/`: run linkage, code/environment identities, preflight, matrix, and
  before/after sealed inference inventories.

Large caches, model/scorer tensors, checkpoints, and per-image evaluation
evidence remain local and intentionally uncommitted. They are reproducible from
the authenticated local run and are not needed to interpret the reported
capacity failure.
