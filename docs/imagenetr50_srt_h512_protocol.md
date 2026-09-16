# ImageNet-R-50 SRT versus uniform replay at H=512

This user-requested follow-up changes only the work-equivalent budget from
H=1,024 to H=512. It is not the proposed review-age/repetition-cap intervention
study. Those interventions remain deferred.

## Fixed comparison

- Two cold-start, complete 50-task streams, seed 1993: ordinary SRT and its
  exact-per-update matched uniform control.
- Standard quality thresholds `[0.10, 0.25, 0.50, 0.75, 0.90]`, old target 0.8,
  interval unit 8. No new threshold search, validation selection, LR search,
  early stopping, or test-dependent checkpoint selection.
- Same immutable 24,000/6,000 split, class order, frozen ViT-B/16 backbone,
  rank/alpha-16 QKV/fc1 LoRA, growing affine head, optimizer and augmentation
  recipe as the original SRT experiment. Adapter, head and SGD momentum
  persist within each stream; the streams do not warm-start from old runs.
- At stage t, training presentations are
  `4 * (new_images + min(512, historical_images))`. H is a work parameter,
  not a limit on the stored replay population. Every arrived training image
  remains eligible. Mandatory new-task coverage is unchanged.
- Uniform copies SRT's realized batch sizes, old/current counts, introduction
  boundaries and optimizer-update count, but samples replay identities uniformly.
  Confidence measurement uses the same pre-update forward, with no extra pass.

## Evidence and interpretation

Freeze the configuration and content-addressed protocol before training. Keep
the original result, the H=4,096 fixed-policy follow-up, the joint-IID controls,
and the inference-only checkpoint diagnostics immutable. Append this result to
the same SRT report through its own sealed follow-up pointer.

The earlier `sample_replay.parquet` is a hash-pinned input to the completed
checkpoint diagnostic and remains frozen. The current complete per-image
export is `replay_samples.parquet` (with CSV/JSON projections); it includes
the new H=512 pair. The report manifest identifies the historical input
separately from its current analysis tables.

Report every stage's accuracy and raw NLL, per-task scores, actual image work,
optimizer updates, wall time, replay counts, coverage and timing distributions.
Authenticate image bytes and membership, checkpoint/prediction hashes, paired
batch schedules and zero-step reuse. Exclude test identities from every training
input. Preserve original raw benchmark scores and diagnostic temperature fits.

Compare H=512 directly with the existing standard/old=0.8/unit=8 H=1,024 and
H=4,096 pairs. Carry forward the historical stage-matched joint-IID curves and
the newer three-seed offline rank-16 endpoints. The replay-schedule-matched
offline endpoint matches H=4,096 standard work, not this smaller budget; show
it only at task 50, not as a new stage-matched curve or a pass/fail gate.

This is a single-seed exploratory follow-up requested after inspecting the test
set. Differences across budgets also change realized batch schedules and may
change replay behavior. Do not attribute an accuracy change solely to memory
size, infer statistical significance from one seed, or call the older joint-IID
training recipe an architectural ceiling.

## Execution

```bash
bash scripts/vision/imagenetr/run_srt_followup_local.sh run --config configs/vision/imagenetr/srt_h512_rho80_unit8.yaml
```

Use the isolated vision environment and one nice-10 GPU worker. Check host and
GPU memory before launch; do not overlap heavy jobs. Atomic existing checkpoints
allow the same command to resume safely. `status` is read-only and `report`
rebuilds the existing report without training.
