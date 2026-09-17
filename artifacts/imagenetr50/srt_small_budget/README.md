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
validation search and both selected full-data refits are complete.
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

## Completed validation search

All 32 distinct candidates finished all 50 tasks. Selection was sealed as
`3909a1344db5af7f28004557bab4a66aa791ef93654e0baea88045cc670b5e83`
before either new full-data refit. The selected candidate is
`c84fa3d51903a2cfff9746d3fb7d87e58f18853a92b2be9675d3171c2ff0836e`:
relaxed thresholds `[0.05, 0.15, 0.3, 0.6, 0.85]`, historical target 0.95,
interval unit 4, LoRA learning rate 0.001 and classifier learning rate 0.005.
Rank, presentation budget, optimizer family and core SRT algorithm are unchanged.
The historical fraction is a target: a shortage of due old images can be
filled with due current images, and vice versa, after mandatory introductions.

| Validation recipe | Task-50 accuracy | Raw NLL |
| --- | ---: | ---: |
| Original standard / old 0.8 / unit 8 | 72.645833% | 1.214815 |
| Policy-phase winner: relaxed / old 0.95 / unit 8 | 74.333333% | 1.184862 |
| Rate-phase winner: same policy, LoRA/head 0.001/0.005 | 75.041667% | 1.104739 |
| Final selection: same rates and policy, unit 4 | 76.083333% | 1.078863 |

The runner-up used historical target 1 and unit 8 with the same relaxed
thresholds and selected rates. It scored 76.062500% / 1.064438 NLL. The winner
therefore leads by one of 4,800 validation images (0.020833 percentage points),
despite higher NLL. That is the predeclared accuracy objective, not evidence
that this near-tie is reproducible. No test score changed the choice.

Search work totals 3,260,416 training presentations, 54,883 optimizer updates
and 4,015,552 validation forwards. Measured training and validation took
161.55 and 141.63 minutes, respectively; these exclude other overhead and are
separate from the cost of running one selected stream.

No candidate raised a numerical exception. Two nevertheless collapsed:
LoRA/head rates 0.001/0.01 and 0.001/0.02 reached only 2.458333% and 1.541667%
final validation accuracy. Their task-2 mean training NLL was already 28.025
and 208.370, respectively. This is consistent with early optimization
instability, not merely forgetting late in the stream. Both are retained in
the full candidate table and plot. Finishing with finite numbers is not a
quality gate, and the search was not altered in response to these results.

## Selected recipe on the full training population

Both fresh runs use 121,088 training presentations and 1,974 optimizer updates.
Uniform receives the selected SRT optimizer settings and exact per-update
batch sizes and old/current counts. It is not independently tuned.

| Selected H=128 condition | Task-50 accuracy | Raw NLL | Mean-stage accuracy |
| --- | ---: | ---: | ---: |
| SRT | 76.666667% | 1.037512 | 82.371977% |
| Matched uniform | 77.016667% | 1.031127 | 82.085947% |

SRT gains 1.566667 percentage points over its original H=128 recipe, with
NLL lower by 0.102442. Uniform gains 2.683333 points over its original H=128
recipe and leads the new SRT pair by 0.350000 points at task 50. SRT has
higher mean-stage accuracy, but that was not the tuning objective. These
results do not establish an SRT advantage over uniform. Both final accuracies
remain below the longer-trained rank-16 joint-IID accuracy-selected mean
(79.172222%, three seeds); that reference uses more training work and is not
an H=128 matched control.

Measured training/evaluation times were 358.56/346.77 seconds for SRT and
346.49/343.43 seconds for uniform. The selected pair has fewer updates than
the original H=128 pair (1,974 versus 2,046), despite equal presentations.

The final tuning result is sealed as
`3712f1d3c19fe6db3b4278912fcd70901cb17469e8746f4b673b39b8c8b510d4`.
The selected SRT job hash is
`a170c2b8afae0509b47ca06d3efa7980535a385cfbfd4465dc1235262e0862fb`;
uniform is `15b4e125e5a48807ef6dc26499386326ccacc4b6a7556d59a1faa87b5ee3ef6c`.
All 32 completed validation jobs and both final jobs were reused with zero
optimizer steps and unchanged result hashes. Original scientific results
and the older frozen replay-history input remain unchanged.

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

Final verification passes in explicit sequential `-n 0` slices: 83 focused
short tests, four real-artifact evidence audits and both synthetic tuning
layout variants. The evidence audits authenticate every dataset image and
every stage model/prediction file, reconstruct final scores and all three
search decisions, and preserve older scientific summaries. They also verify
identical task-1 models when only the historical target changes, since there
are no historical images at task 1. The report separately audits all committed
final-stream replay events and exact paired schedules.

A fresh completed-workflow invocation again required zero optimizer steps
and reproduced the report byte-for-byte. A subsequent report-only correction
separated the expanded replay-coverage legend from its axis label and added
the selected uniform result to the opening summary. The layout regression
fails on the old figure and passes after the fix. The final 39-page PDF,
including all six new tuning pages and changed work/replay panels, has been
visually checked. Its SHA-256 is
`a6e3b17b24ba61b9ac186d458a0564d391853f5b9ef5931e2aa9b41dc76b04a9`.

The remaining uncertainty is scientific, not an unfinished run: replicate
the selected recipe and its matched uniform control before interpreting the
0.35-point method difference or the one-image validation ranking. The search
does not establish a globally optimal SRT recipe or a separately tuned
uniform optimum.
