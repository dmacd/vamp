# H=512 SRT versus uniform: completed 50-task comparison

The existing report is `output/pdf/imagenetr50_srt_r16_report.pdf`. Its new
opening sections compare all standard-profile H=512, H=1,024 and H=4,096 curves
on common accuracy/NLL axes, while retaining the earlier comparisons and
checkpoint diagnostics. The source run's `reports/REPORT.md` and self-contained
`reports/REPORT.html` contain the same analysis.

## Fixed protocol and measured result

Both streams start fresh with seed 1993, standard thresholds, old target 0.8
and interval unit 8. Only H changes; architecture, data, optimizer, augmentation,
quality thresholds and the core training implementation are unchanged. No
review-age protection or repetition cap is introduced. The exact protocol is
`docs/imagenetr50_srt_h512_protocol.md`.

| H=512 condition | Task-50 accuracy | Raw NLL | Mean stage accuracy | Training minutes | Evaluation minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| SRT | 78.933333% | 0.969016 | 83.676893% | 10.208 | 6.117 |
| Matched uniform | 78.366667% | 0.958859 | 83.942644% | 9.356 | 5.750 |

Each stream uses **196,204 training image forward/backward pairs and 3,358
optimizer updates**. H limits work, not the stored history: all 24,000 arrived
training images remain available by task 50. The held-out test set contains
6,000 images. Training/evaluation times exclude bootstrap, report generation,
checkpoint serialization and the separate verification work.

SRT has 34 more correct task-50 predictions, a 0.566667-point advantage. Uniform
has slightly better NLL and mean stage accuracy. One seed does not establish
which method generally wins. These are exploratory results on a test set
already inspected during earlier experiments.

Uniform's task-50 accuracy is now below both newer offline rank-16 references:
79.172222% for the validation-selected epoch and 80.827778% for the
H=4,096-schedule-matched control (each a three-seed mean). The latter exceeds
H=512 uniform by 2.461111 points but uses 844,640 training presentations and
56,243 updates per seed, so it is **not an H=512 work-matched control**. This
comparison does not isolate why earlier uniform runs exceeded older joint-IID
recipes. Lowering H also changes the realized batch/update/review schedule.
No new temperature fits were performed; NLL here remains raw.

## Identity and retained evidence

Source run:
`artifacts/imagenetr50/srt_r16_v1/runs/b3de2a819efe0707b85cf105cf8d4c53370c807b48108bf996dc4f4883a213ea`

New run under that source's `followups/`:
`c2ea171a6b060ebbce0fcb3e5f5d74a7fe4a2e9e6377be055791328c201bfc5c`

Sealed result:
`15edeec8c9ca6c2a0868d5708395349db372c5290de98d40b6fef4a6c948ba36`

- Both job definitions, all 100 stage result records and the two final
  per-image prediction Parquet files are retained, together with the run's
  protocol, code and software manifests, result and completion status.
- The source report's small CSV/JSON/Parquet tables cover all stages, tasks,
  conditions, measured work and replay timing distributions. All figures and
  the PDF/HTML/Markdown report are retained.
- `reports/replay_samples.parquet` is the complete 240,000-row, ten-condition
  per-image summary. Its 49 MB CSV and 143 MB JSON duplicates remain local.
  The earlier `sample_replay` tables remain frozen inputs to the checkpoint
  diagnostics; their bytes have not changed.
- This directory retains the original run log, fresh zero-training reuse log,
  focused/integration test records, the pre-extension condition summary, and
  hashes protecting earlier results and the persistent-frontier report.
- Model/optimizer checkpoints, raw event shards, image/model caches and
  intermediate PDF renders stay local. Tracked evidence supports analysis
  without a GPU; rebuilding the full event audit requires the local store.

## Reproduction and checks

```bash
bash scripts/vision/imagenetr/run_srt_followup_local.sh run --config configs/vision/imagenetr/srt_h512_rho80_unit8.yaml
.venv-vision/bin/python -m pytest -n 0 -q -m integration tests/test_imagenetr_srt_small_budget_evidence.py
```

The run command uses one nice-10 GPU worker and reuses completed jobs with
zero optimizer steps. `status` selects this configuration without writing;
`report` regenerates the existing report without training. The core training
files and source scientific results remain unchanged. The first two task
checkpoint/prediction hashes match the H=1,024 runs exactly, as expected while
fewer than 512 historical images are available.

The explicit integration audit validates all 30,000 image bytes and split
membership, every new checkpoint/prediction hash, task-50 accuracy/NLL from
per-image predictions, the combined report, unchanged older summaries and the
frozen diagnostic input. The report also audits the full replay-event history
and exact SRT/uniform exposure schedules.

All 78 focused and explicit integration tests passed in one `-n 0` process;
see `verification/final_tests.xml` and `.log`. No tests failed or were skipped.
The fresh complete workflow reused both streams without training, and all 32
PDF pages passed visual QA. `verification/result.json` records these checks
and the exact final report hash. The entire unrelated repository test suite
was not rerun for this isolated extension.
