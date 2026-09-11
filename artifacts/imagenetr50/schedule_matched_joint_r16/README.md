# Offline rank-16 control with the replay optimizer schedule

The new all-data control nearly closes the observed accuracy gap to uniform
replay, but not the NLL gap. It averages **80.828% accuracy / 1.0463 NLL**
across three cold seeds. Uniform replay scores **80.917% / 0.8951** in the
single matched source run. The previous validation-accuracy-selected joint
reference averages **79.172% / 1.0490** across three seeds.

## What changed

All 24,000 training images and all 200 classifier rows are available from the
first update. Each batch draws uniformly over training images, without
replacement within that batch and independently between batches. The model
is the same frozen ViT-B/16 with rank/alpha-16 QKV/fc1 LoRA and a 200-way affine
head: 1,480,904 trainable parameters.

The control copies the exact 56,243 batch sizes, 844,640 image presentations,
constant SGD learning rates, momentum, and weight decay from
`uniform_h4096_standard_rho80_unit8`. The mean realized batch size is 15.018,
not 64; there are 449 single-image updates. Unlike replay, no source task
boundary, image identity, or old/current quota constrains the samples.

The protocol and training implementation were committed as `0a644c3` before
training. Seeds 1993-1995 each train to the fixed copied endpoint, with no
new hyperparameter search or validation/test-based selection. Final test
evaluation waits until all three fits finish. The fifty intermediate
checkpoints are training-work boundaries with training-only fit probes,
not continual-learning stages or candidate test-selected checkpoints.

## Results on the same 6,000 test images

| Condition | Runs | Accuracy (%) | NLL |
|---|---:|---:|---:|
| Offline control, seed 1993 | 1 | 81.183 | 1.01384 |
| Offline control, seed 1994 | 1 | 80.717 | 1.04687 |
| Offline control, seed 1995 | 1 | 80.583 | 1.07816 |
| Offline control, mean (sample SD) | 3 | 80.828 (0.315) | 1.04629 (0.03216) |
| Uniform H=4,096, standard/old=0.8/unit=8 | 1 | 80.917 | 0.89507 |
| Previous validation-accuracy-selected joint reference | 3 | 79.172 (0.167) | 1.04897 (0.02019) |

The new offline mean gains 1.656 accuracy percentage points over the previous
joint recipe and is 0.089 points below the matched replay source. Thus the
earlier 79.17% result was not an architectural ceiling. This comparison does
not isolate batch size from learning-rate history, number of updates, and
sampling changes relative to that earlier joint recipe.

The remaining NLL difference is 0.1512. Averaged over offline seeds,
misclassified images contribute 1.0033 to whole-test NLL, versus 0.8318 for
replay; correct predictions contribute 0.0430 versus 0.0633. The two
contributions sum to total NLL. Error sets differ, so this is not a
paired-error test or a full calibration measurement.

Seed 1995 had a finite early loss/gradient spike: one single-image update
reached NLL 122.752. Its training-probe accuracy was 2.197% after 1,512
updates, recovered to 68.018% after 4,006, and finished at 98.975%. Nothing
was reset or retuned, and this seed remains in all summaries. The report
includes every seed and the post-hoc training-only stability diagnostics.

Three offline seeds do not replicate the single replay arm or its chosen
schedule. These are observed differences, not evidence of statistical
equivalence or a globally optimal joint-IID ceiling. The follow-up was
requested after earlier test results were inspected. A useful next control
would replicate replay under the same fixed schedule before interpreting
the small accuracy difference, then separate curriculum from cumulative
sample weighting if the NLL advantage persists. Neither is launched here.

## Work and verification

The three fits total 168,729 optimizer updates and 2,533,920 training
forward/backward image pairs. The fifty 2,048-image probes per seed add
307,200 forward image paths, and three final tests add 18,000: 2,859,120
forward paths overall. Preflight work is excluded. Measured training batches
take 190.61 minutes in total; checkpoint and evaluation/artifact times are
recorded separately. One nice-10 GPU process ran at a time.

Preflight and post-training audits authenticate all 30,000 image bytes.
Every draw and augmentation ordinal is reconstructed from the recorded
hashes; all 24,000 training images are used in each fit and no test image
enters training or probes. Interrupted mixed-size-batch training reproduces
uninterrupted tensors exactly. The completed workflow checks zero-step
reuse; `audit/` additionally records a fresh-process file/hash/mtime audit.
The fresh default invocation exits successfully with zero optimizer updates,
all 3,884 scientific/selected source files unchanged, and byte-identical
regenerated report/PDF outputs.
The final focused vision slice passes 181 tests (14 integration/benchmark
tests deselected), with six explicit report/evidence/layout tests passing.

## Evidence retained here

`LATEST_RUN.json` selects run
`7a027ff732874edd48617b430fc5812149990913edbf103b4fe653cc32744766`.
Its sealed result hash is
`ba351626e45f935c399357dd51d3e4982d53de2f4c1b36930bb7df749d2b1f47`.

- Protocol, software/code/model/dataset hashes, image memberships, the exact
  ordered schedule, preflight and final data audits, and agent health logs.
- Per-seed results, fifty block result records, per-image exposure counts,
  and one compact `update_metrics.parquet` containing all 56,243 measured
  updates. `update_export.json` binds these tables to the original chunks.
- All final per-example predictions and test metric records, runner logs,
  test logs/XML, and fresh-process reuse evidence.
- The existing SRT report contains seven new JSON/CSV/Parquet table families,
  two new figures, and a task-50-only mean/SD marker on the comparison plots.

Model weights, optimizer checkpoints, the original individual update chunks,
raw images, and caches stay local. Their content hashes remain in the
retained evidence. The compact tables support analysis without reconstructing
training; a full authenticated workflow rebuild requires those local files.
No separate publication bundle is generated.

Run/reuse: `bash scripts/vision/imagenetr/run_schedule_matched_joint_local.sh`.
Report only: append `report`; read-only status: append `status`.
The protocol is in `docs/imagenetr50_schedule_matched_joint_protocol.md` and
scientific choices are fixed in the matching YAML configuration.

See the [updated SRT report](../srt_r16_v1/runs/b3de2a819efe0707b85cf105cf8d4c53370c807b48108bf996dc4f4883a213ea/reports/REPORT.md)
and its [PDF](../../../output/pdf/imagenetr50_srt_r16_report.pdf).
