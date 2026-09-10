# ImageNet-R-50: single-adapter Spaced Repetition Training

## Original validation-selected results

Does confidence-based spaced repetition improve a single continuing rank-16 adapter compared with uniform replay under identical realized exposure? The ImageNet-R split is unchanged: 24,000 training images, 6,000 test images, and fifty four-class tasks in the existing seed-1993 order.

Uniform replay has higher final accuracy and lower final NLL at 2 of the two tested budgets. This comparison tests the selected SRT recipes, not every possible confidence threshold or spacing rule.

At the 1,024-equivalent budget, SRT finishes at 79.267% accuracy and 0.9585 NLL. Relative to its exposure-matched uniform control, the differences are -1.267 accuracy points and +0.0993 NLL. Its accuracy difference from joint rank 16 is +0.400 points.

At the 4,096-equivalent budget, SRT finishes at 77.333% accuracy and 1.1100 NLL. Relative to its exposure-matched uniform control, the differences are -3.067 accuracy points and +0.1977 NLL. Its accuracy difference from joint rank 16 is -1.533 points.

Mean accuracy is the arithmetic mean of the fifty stage test accuracies. NLL is uncalibrated, all-seen-class cross-entropy; lower is better. These are single-seed results, not estimates of training-run variability. Final-run times below exclude calibration.

| Condition | Final acc. | Mean acc. | Final NLL | Train min. |
| --- | --- | --- | --- | --- |
| SRT rank 16, 1,024-equivalent budget | 79.267% | 83.916% | 0.9585 | 14.52 |
| Uniform replay rank 16, 1,024-equivalent budget | 80.533% | 85.059% | 0.8592 | 14.05 |
| SRT rank 16, 4,096-equivalent budget | 77.333% | 81.908% | 1.1100 | 40.11 |
| Uniform replay rank 16, 4,096-equivalent budget | 80.400% | 84.197% | 0.9123 | 40.46 |

## Original full-stream comparison at both work budgets

Each panel compares one SRT/uniform pair with the five existing task-free conditions. Historical curve names, values, and colors are unchanged. A budget of H-equivalent work means 4 x (current images + min(H, historical images)) presentations per task, not a memory cap. Both methods can revisit every arrived training image.

![Original full-stream comparison at both work budgets](stage_accuracy.png)

## Fixed-policy follow-up: H=4,096, old=0.8, unit=8

Four additional cold-start 50-task streams hold H=4,096, historical target 0.8, and interval unit 8 fixed. Standard and strict thresholds each have an exposure-matched uniform control. Every stream uses 844,640 training presentations. The optimizer, model, data split, seeds, and checkpoint timing are unchanged; no calibration jobs are repeated.

Standard thresholds are [.10,.25,.50,.75,.90]; strict thresholds are [.20,.40,.70,.85,.95]. The 0.8 target applies after the mandatory current-task first pass. Actual shares can differ when a due pool is short. A uniform condition's profile name identifies the SRT batch schedule it copies, not a quality rule applied to uniform sampling.

Batch size 64 is a maximum. When few examples are due, partial batches consume more optimizer updates for the same image budget. The table reports these updates separately; each uniform control matches them exactly.

With standard thresholds, SRT minus its matched uniform control is -6.950 final accuracy points and +0.3731 NLL.

With strict thresholds, SRT minus its matched uniform control is -5.033 final accuracy points and +0.3243 NLL.

For SRT with standard thresholds, old target 0.8, and unit 8 fixed, raising H from 1,024 to 4,096 changes final accuracy by -5.300 points and NLL by +0.3097.

For uniform replay with standard thresholds, old target 0.8, and unit 8 fixed, raising H from 1,024 to 4,096 changes final accuracy by +0.383 points and NLL by +0.0359.

The new standard SRT run ends with 10 overdue images, and strict with 6,078, out of 24,000. A due backlog measures failure to meet a requested schedule, not whether that schedule maximizes accuracy.

These settings were requested after reviewing the original test curves. Treat them as exploratory single-seed ablations, not an independently selected final winner. Original result hashes and the validation selection remain unchanged.

| Condition | Final acc. | Mean acc. | Final NLL | Updates | Train min. |
| --- | --- | --- | --- | --- | --- |
| SRT rank 16, H=4,096; standard, old=0.8, unit=8 | 73.967% | 79.490% | 1.2682 | 56,243 | 60.41 |
| Uniform replay rank 16, H=4,096; standard, old=0.8, unit=8 | 80.917% | 84.820% | 0.8951 | 56,243 | 62.77 |
| SRT rank 16, H=4,096; strict, old=0.8, unit=8 | 75.717% | 81.014% | 1.2217 | 21,926 | 43.32 |
| Uniform replay rank 16, H=4,096; strict, old=0.8, unit=8 | 80.750% | 84.816% | 0.8974 | 21,926 | 43.46 |

## Full-stream accuracy with the fixed-policy conditions

All four new conditions use a single rank-16 adapter. Colors distinguish standard and strict profiles; solid lines denote SRT and dashed lines their matched uniform controls. All five earlier task-free frontier and joint-IID conditions remain overlaid. The original selected-policy SRT curves remain on the preceding accuracy page.

![Full-stream accuracy with the fixed-policy conditions](fixed_policy_stage_accuracy.png)

## Direct comparisons with the original SRT recipes

Left: standard thresholds, old target 0.8, and unit 8 are fixed; only H changes from 1,024 to 4,096. Right: H=4,096 and strict thresholds are fixed; old target/unit change together from 0.5/1 to 0.8/8. The right comparison does not isolate mixture from spacing. Each uniform line copies its associated SRT batch schedule. The lower row shows NLL for the identical conditions; the joint rank-16 source has no retained NLL.

![Direct comparisons with the original SRT recipes](fixed_policy_comparisons.png)

## Equal image work does not mean equal optimizer work

Each line represents an SRT recipe and its exact-count uniform partner. H=4,096 fixes 844,640 image presentations, but the due-only scheduler can produce batches smaller than 64. Cross-entropy is averaged within each actual batch; each batch then takes one full-learning-rate SGD update, with momentum and weight decay. More small batches therefore change optimization as well as overhead. No gradient accumulation is applied. The interval unit measures virtual scheduling ticks, not a fixed amount of intervening image work; empty-queue clock advances and due backlogs also separate requested from actual replay gaps.

![Equal image work does not mean equal optimizer work](optimizer_work_and_batching.png)

## Carried-forward accuracy figure

This is the original full-stream accuracy figure, copied byte-for-byte from the previous report. The pale dotted curves are label-aware true-node diagnostics for the frontier models, not deployable methods and not SRT baselines. The prior frontier results and report were not changed.

Joint rank 16 uses the same adapter targets, rank, and affine classification architecture as SRT, but retrains on each full prefix. Aggregate-rank-matched joint IID matches the total rank of the frontier nodes, not the rank of the single SRT adapter.

![Carried-forward accuracy figure](../references/previous_stage_accuracy.png)

## Probability quality and the joint-IID gap

The rank-16 joint source did not retain NLL, so no rank-16 NLL curve is fabricated. The lower panel uses its retained accuracy curve. Neither joint reference is an execution gate or a mathematical upper bound. Each fresh joint model receives five epochs. The final rank-16 joint model received 120,000 training presentations and 1,875 updates, whereas continuing uniform H=1,024 received 294,368 presentations and 5,253 updates across its lifetime. Earlier joint-prefix models do not warm-start later ones.

![Probability quality and the joint-IID gap](nll_and_joint_gap.png)

## Measured training work

Each new training presentation produces one ViT forward/backward pair. Confidence comes from that same forward: there are no quality-only passes, source hierarchy jobs, or activation recomputation. Prior frontier costs include their source hierarchy. The table separates their extra recomputation forwards.

Training time sums measured batch work and committed checkpoint I/O. Loader and scheduler timings are components of that time, not costs to add again. Setup, model serialization outside those counters, between-job overhead, and evaluation are separate. Calibration is excluded from this comparison.

| Condition | Forward/backward pairs | Recompute forwards | Train min. |
| --- | --- | --- | --- |
| SRT rank 16, 1,024-equivalent budget | 294,368 | 0 | 14.52 |
| Uniform replay rank 16, 1,024-equivalent budget | 294,368 | 0 | 14.05 |
| SRT rank 16, 4,096-equivalent budget | 844,640 | 0 | 40.11 |
| Uniform replay rank 16, 4,096-equivalent budget | 844,640 | 0 | 40.46 |
| SRT rank 16, H=4,096; standard, old=0.8, unit=8 | 844,640 | 0 | 60.41 |
| Uniform replay rank 16, H=4,096; standard, old=0.8, unit=8 | 844,640 | 0 | 62.77 |
| SRT rank 16, H=4,096; strict, old=0.8, unit=8 | 844,640 | 0 | 43.32 |
| Uniform replay rank 16, H=4,096; strict, old=0.8, unit=8 | 844,640 | 0 | 43.46 |
| Persistent single-affine + adaptive node LoRAs, H=4,096 | 3,050,765 | 2,385,440 | 104.23 |
| Stage-matched joint IID, rank 16 | 3,140,210 | 0 | 80.72 |
| Aggregate-rank-matched joint IID | 3,140,210 | 0 | 85.88 |
| Persistent two-layer MLP + adaptive node LoRAs, H=4,096 | 3,050,765 | 2,385,440 | 107.19 |

## Cumulative work across the stream

The fixed H-equivalent budget gives linear training-image work for bounded task size and this fixed model. Repeated prefix testing is quadratic. CPU population scans and report reductions are not covered by the linear model-work claim. These counts are image paths, not profiled FLOPs; equal counts do not imply equal arithmetic for different adapter ranks or multiple frontier models.

![Cumulative work across the stream](cumulative_work.png)

## When reviews actually happened

The distributions are review-event weighted and separate real optimizer gaps from virtual scheduling-clock delays. Scheduling time advances to the next due item when neither pool is ready; that does not perform a training update. A large due backlog means requested spacing and actual spacing differ.

The historical fraction plotted includes the mandatory first pass over new images. Its realized value can differ from the configured post-introduction fraction when one due pool is short. Each uniform control copies the exact resulting old/current counts and partial batch sizes.

![When reviews actually happened](replay_timing.png)

## Which samples received replay

Earlier arrival cohorts have more opportunities for historical replay. Compare methods within a cohort rather than interpreting that age effect as preferential selection. The lower plot gives each image one vote. Task-50 images cannot receive a historical replay within this experiment.

The per-sample table retains exact mean, median, 90th-percentile, minimum, and maximum optimizer gaps, review counts, and terminal waits. Every unfinished next-review interval is marked censored; images with no observed second presentation are retained rather than silently omitted.

![Which samples received replay](sample_coverage.png)

## Requested spacing versus realized spacing

These curves use the same historical review events and the same scheduling-clock unit. Requested intervals come from the previous SM-2 update; actual intervals run from that previous presentation to the next real review. Any rightward displacement is waiting beyond the requested spacing.

CDF values are evaluated at logarithmic bin boundaries, using completed intervals only. Terminal unfinished intervals are retained separately as censored waits, not interpreted as completed long gaps. The earlier optimizer-gap plot measures actual learning updates rather than virtual clock advances.

![Requested spacing versus realized spacing](requested_and_actual_intervals.png)

## Individual review histories

Eight image identities were selected by a fixed hash rule: two each from arrival tasks 1, 16, 31, and 50. Selection does not use accuracy, confidence, or whether SRT looks successful. SRT marks sit above each sample row and uniform marks below. The full identities and every event are retained in the analysis tables.

The horizontal coordinate is task arrival plus the fraction of that task's optimizer updates, not elapsed wall time. Exact optimizer-step gaps are in the timing distributions and per-sample tables.

![Individual review histories](sample_timelines.png)

## Calibration and selected settings

Eighteen settings per budget were screened through task 16 on the existing 19,200/4,800 training-derived fit/validation split. Two finalists per budget continued through task 50. Selection maximized mean stage validation accuracy, then minimized mean NLL, then used canonical policy order. No test images entered this process.

Calibration consumed 6,752,992 training presentations, 7.49 measured training hours, and 34.94 evaluation minutes. The four final models restarted from cold adapters after selection.

Relaxed thresholds are [.05,.15,.30,.60,.85], standard [.10,.25,.50,.75,.90], and strict [.20,.40,.70,.85,.95]. Quality counts thresholds met by the pre-update probability of the correct class. The interval unit scales the first/failure interval and the first two successful intervals.

These numerical thresholds were hand-chosen search candidates, not values reported by the paper. Validation selected among them; it did not establish that a quality score predicts retention after a given delay. Calibration costs above exclude the preserved, superseded development run and preflight.

| Budget | Threshold profile | Old fraction | Time unit | Mean val. acc. |
| --- | --- | --- | --- | --- |
| 1,024 | standard | 0.8 | 8 | 82.593% |
| 4,096 | strict | 0.5 | 1 | 80.512% |

## Complete calibration matrix

Screen accuracy averages tasks 1-16; full accuracy averages tasks 1-50 and is present only for continued finalists. An asterisk marks the selected setting. All values are validation percentages, not test results. NLL, resource counters, hashes, and every stage curve are included in the accompanying tables.

| Budget | Profile | Old | Unit | Screen acc. | Full acc. |
| --- | --- | --- | --- | --- | --- |
| 1,024 | relaxed | 0.2 | 1 | 85.757 | - |
| 1,024 | relaxed | 0.2 | 8 | 86.323 | - |
| 1,024 | relaxed | 0.5 | 1 | 85.627 | - |
| 1,024 | relaxed | 0.5 | 8 | 85.929 | - |
| 1,024 | relaxed | 0.8 | 1 | 86.295 | - |
| 1,024 | relaxed | 0.8 | 8 | 86.207 | - |
| 1,024 | standard | 0.2 | 1 | 84.973 | - |
| 1,024 | standard | 0.2 | 8 | 86.578 | - |
| 1,024 | standard | 0.5 | 1 | 85.604 | - |
| 1,024 | standard | 0.5 | 8 | 87.325 | 82.475 |
| 1,024 | standard | 0.8 | 1 | 86.776 | - |
| 1,024 | standard * | 0.8 | 8 | 87.442 | 82.593 |
| 1,024 | strict | 0.2 | 1 | 84.721 | - |
| 1,024 | strict | 0.2 | 8 | 86.768 | - |
| 1,024 | strict | 0.5 | 1 | 85.732 | - |
| 1,024 | strict | 0.5 | 8 | 86.183 | - |
| 1,024 | strict | 0.8 | 1 | 87.090 | - |
| 1,024 | strict | 0.8 | 8 | 86.743 | - |
| 4,096 | relaxed | 0.2 | 1 | 84.917 | - |
| 4,096 | relaxed | 0.2 | 8 | 82.927 | - |
| 4,096 | relaxed | 0.5 | 1 | 84.462 | - |
| 4,096 | relaxed | 0.5 | 8 | 82.950 | - |
| 4,096 | relaxed | 0.8 | 1 | 85.071 | - |
| 4,096 | relaxed | 0.8 | 8 | 82.946 | - |
| 4,096 | standard | 0.2 | 1 | 84.760 | - |
| 4,096 | standard | 0.2 | 8 | 84.455 | - |
| 4,096 | standard | 0.5 | 1 | 85.222 | - |
| 4,096 | standard | 0.5 | 8 | 85.382 | - |
| 4,096 | standard | 0.8 | 1 | 85.642 | - |
| 4,096 | standard | 0.8 | 8 | 84.835 | - |
| 4,096 | strict | 0.2 | 1 | 85.702 | - |
| 4,096 | strict | 0.2 | 8 | 85.910 | - |
| 4,096 | strict * | 0.5 | 1 | 86.296 | 80.512 |
| 4,096 | strict | 0.5 | 8 | 85.638 | - |
| 4,096 | strict | 0.8 | 1 | 86.328 | 80.226 |
| 4,096 | strict | 0.8 | 8 | 85.546 | - |

## Protocol, verification, and interpretation limits

The frozen ViT-B/16 uses rank/alpha-16 LoRA on QKV and fc1 in all twelve blocks. An ordinary affine head adds four rows per task. Old rows, adapter parameters, and SGD momentum persist; all seen rows remain trainable. SGD uses batch limit 64, momentum 0.9, weight decay 0.0005, LoRA learning rate 0.0005, and head learning rate 0.01, without clipping or a learning-rate schedule.

Our implementation follows Atreya et al.'s pre-update scoring and SM-2 equations. Their code is proprietary and their classification thresholds are unspecified. The guaranteed first pass, calibrated thresholds/interval unit, and explicit empty-queue clock rule are documented local choices, not claims of an exact reproduction.

The real smoke test verified zero-LoRA parity, BF16 batch 64, eight task arrivals, exact interrupted/uninterrupted weights and review events, and matched uniform exposure. Reporting revalidates all committed events and reconstructs accuracy/NLL from stored predictions. A completed rerun performs zero optimizer steps.

The uniform control is conditioned on SRT's realized allocation; it is not an independently tuned uniform-replay optimum. Quality is measured on random training crops, so a low score can reflect an uninformative crop as well as forgetting. Hyperparameters were selected offline using a training-derived full-stream validation sweep. One seed cannot establish reproducibility or a publishable state-of-the-art result.

For the original validation-selected policies, 14,498 of 24,000 training images are overdue at the final boundary at H=1,024, and 12,560 at H=4,096. Requested spacing is therefore not reliably delivered. This does not isolate the cause of the accuracy difference: crop-dependent confidence, sample selection, and review delays can all contribute.

The next tests should replicate the paired SRT/uniform comparison across seeds and, on training-derived validation data, measure how pre-review confidence predicts later retention at observed delays. Then test spacing calibrated to the available review budget. H changes both work and the selected recipe here, so it is not a pure work ablation. The original selection stays frozen. The separately identified, user-requested fixed-policy follow-up is exploratory because it was requested after examining these test results.

Source: Atreya et al. (2026), When to Review: Spaced Repetition for Continual Pre-Training of Language Models, arXiv:2608.17530v1. https://arxiv.org/html/2608.17530v1

## What correct-class probability does and does not measure

The quality signal is p(correct class), equivalently exp(-cross-entropy), measured before the update on the sampled training crop. Appendix C.2 of Atreya et al. proposes this signal for classifiers. It is cheap and follows the training loss, but is not a measured probability that an example will be forgotten, nor a measure of the expected accuracy gain from replaying it now.

Two 200-class predictions can both assign 0.50 to the correct class while giving their strongest competitor 0.01 or 0.49. Both are currently correct and have the same 0.693 cross-entropy, but probability margins of 0.49 versus 0.01. The present scheduler assigns them identical quality. At exactly 0.50, standard assigns quality 3 (success), while strict assigns quality 2 (failure, resetting the interval to 8 ticks).

Raw confidence also changes under positive temperature rescaling without changing the winning class. New classifier rows can change the softmax denominator, and a random crop covering as little as 5% of an image can omit the object. Low confidence is therefore not synonymous with forgotten knowledge. Conversely, a wide margin on today's crop does not prove that later tasks cannot erase the decision.

Our threshold values are hand-chosen candidates. Validation selected recipes by average accuracy, but did not calibrate quality to future retention or replay benefit. Prioritizing low-confidence examples also changes their effective training weights and can overemphasize ambiguous crops. Following per-image loss does not establish that this allocation maximizes held-out accuracy. A useful next study would compare correctness, correct-versus-runner-up margin, and confidence against later prediction failures at measured delays, using training-derived probes. Then compare a retention-based scheduler under matched work. Margin is a hypothesis to test, not an established replacement.

Source: Atreya et al., https://arxiv.org/html/2608.17530v1#A3.SS2

## Individual review histories for the fixed-policy follow-up

The same eight hash-selected training images appear in both profile panels. SRT marks sit above their paired uniform marks. Neither the identities nor their inclusion was selected using these outcomes. Timing is task arrival plus the within-task update fraction.

![Individual review histories for the fixed-policy follow-up](fixed_policy_sample_timelines.png)
