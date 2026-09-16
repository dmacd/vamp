# Small-budget replay and H=128 SRT tuning

This extends the existing ImageNet-R SRT report at
`output/pdf/imagenetr50_srt_r16_report.pdf`; it does not create another report.
The full protocol is `docs/imagenetr50_srt_small_budget_tuning_protocol.md`.

## Fixed-policy comparisons

The new H=256 and H=128 comparisons each train fresh SRT and uniform streams
through all 50 tasks. They retain standard thresholds `[0.1, 0.25, 0.5, 0.75,
0.9]`, historical target 0.8, interval unit 8, seed 1993 and the original
rank-16 adapter and optimizer. Each uniform stream copies its corresponding
SRT stream's exact per-update historical/current counts and batch sizes.

H changes the available training-presentation budget, not stored history.
All 24,000 training images remain available after they arrive. Each H=256
stream uses 146,176 training presentations; each H=128 stream uses 121,088.
The task-50 test set contains the same 6,000 images as earlier experiments.
The unchanged per-stage budget is `4 * (new_images + min(H, old_images))`.
After one mandatory introduction of every new image, the scheduler divides
the remaining work between old and current images. H does not itself fix the
number of unique old images or actual historical presentations.

| Completed fixed-policy condition | Task-50 accuracy | Raw NLL | Mean-stage accuracy | Updates |
| --- | ---: | ---: | ---: | ---: |
| SRT H=256 | 77.166667% | 1.052572 | 82.493824% | 2,462 |
| Matched uniform H=256 | 76.816667% | 1.067371 | 82.459527% | 2,462 |
| SRT H=128 | 75.100000% | 1.139954 | 81.164315% | 2,046 |
| Matched uniform H=128 | 74.333333% | 1.177052 | 80.994767% | 2,046 |

The H=256 pair is sealed under source-relative
`followups/6d52053eebd27cdf477e7b953238860119538dcb0c40c9a0cbae5528811ed11d`,
result hash `8fa4438633377a2faee1d6cf2beec95d008c889e503065f1782999adab895395`.
The H=128 pair is sealed under source-relative
`followups/32160fed886a7a59a0dabb0a213142884b1617c08e408d920ff4f683736509a6`,
result hash `212a76e702b0c66a5070a1d8a9fa72b50ee99ea84492254b346f01f5b0c0612b`.
All four completed jobs were reused with zero optimizer steps. The H=128
tuning search remains in progress; no final tuning result is claimed here.
Its frozen source-relative run is
`tuning/e193c7cefcf995fe9f0e1ccb84147e820d9438e282555abe17ed76bfb1364429`.

Under the unchanged recipe, SRT leads uniform by 0.35 percentage points at
H=256 and 0.766667 points at H=128, with lower raw NLL in each pair. Both
methods lose accuracy as H falls from 512 to 256 to 128. These one-seed
differences do not establish a general SRT advantage. Neither smaller-budget
uniform run exceeds the newer offline rank-16 references; those references
use substantially more work and are not matched controls for these H values.

## Selecting H=128 hyperparameters

Selection uses the existing 19,200/4,800 training-derived fit/validation
partition. Every candidate trains all 50 tasks. The objective is **final
task-50 validation accuracy**, with lower validation NLL and then the
candidate content hash breaking ties. Test scores and mean-stage accuracy do
not enter selection.

The frozen search has at most 32 distinct recipes:

1. Eighteen combinations of three confidence profiles, historical targets
   0.5/0.8/0.95, and interval units 1/8, at the original learning rates.
2. The winning policy with independently halved, unchanged or doubled LoRA
   and classifier learning rates. The already evaluated recipe is reused.
3. One-coordinate changes around the current winner: two threshold powers,
   two historical-target offsets and two interval multipliers.

Each phase decision is saved before the next phase's recipes are derived.
Numerically failed candidates are recorded and excluded; other errors stop
the runner for recovery. Successful candidates each consume 101,888 training
presentations on the smaller fitting population. Search work is charged
separately from the final full-data streams.

After the choice is sealed, the selected SRT recipe is refitted from scratch
on all training images. A new uniform control receives the same selected
optimizer settings and exact realized SRT schedule. Uniform is not separately
tuned. If the original recipe wins unchanged, its existing pair is reused.

This is a finite, single-seed coordinate search, not proof of a global optimum.
The validation split and test set have been inspected in earlier studies.
Offline joint-IID results remain comparisons, not acceptance gates, and are
not work-matched to H=128 or H=256.

## Execution and evidence

The preceding H=512 work was committed as `c6de142`. New baseline definitions
were committed as `eb8c7ce`, and tuning implementation as `5491e5c`, before
their respective GPU runs. All were pushed to master. One nice-10 worker runs
the jobs sequentially with immutable stage boundaries and resumable optimizer
and replay state. The original training implementation remains unchanged.

```bash
bash scripts/vision/imagenetr/run_srt_followup_local.sh run --config configs/vision/imagenetr/srt_h256_rho80_unit8.yaml
bash scripts/vision/imagenetr/run_srt_followup_local.sh run --config configs/vision/imagenetr/srt_h128_rho80_unit8.yaml
bash scripts/vision/imagenetr/run_srt_tuning_local.sh
```

Run logs and explicit verification records live here. Scientific job records
live under the original source run's `followups/` and `tuning/` directories.
The report retains every candidate's settings, validation scores and work,
full final-stream curves, and per-image replay summaries. Final prediction
Parquet files support independent accuracy/NLL reconstruction. Raw event
shards, model/optimizer checkpoints, caches and redundant large per-image
CSV/JSON exports remain local.

All three explicit small-budget evidence tests pass. They authenticate every
dataset image, stage model and prediction file, reconstruct final test scores,
check completed-job reuse, and preserve older scientific summaries. The
report independently audits all committed replay events and paired schedules.
The new comparison pages and changed work/replay panels have been visually
checked. Search and final-selected-pair verification remain pending.
