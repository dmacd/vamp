# TinyWorlds-P Semantic-v6 Base and VAMP Execution Report

Date: 2026-07-23
Updated: 2026-07-24 after the registered base-calibration stop

## Current outcome

The complete downstream experiment is implemented. The real CUDA preflight
passed, and the fresh seed-zero base calibration then completed epochs one and
two. It stopped exactly where required with decision
`semantic_grid_failure`: the model learned the held-in stories normally, but
the five intended semantic worlds were not reliably harder than their matched
controls. No selected checkpoint, adapter experiment, or sealed-test result
exists.

The one supported command is:

```bash
ve/bin/python scripts/run_tinyworlds_p_semantic_v6_vamp.py
```

Its first invocation performed only the disposable preflight and stopped. The
reviewed second invocation accepted that estimate and ran the registered
calibration. The failed two-epoch base gate stopped before adapter training, as
designed. The frozen v6 configuration must not be rerun as a new experiment.

## Frozen sources and experiment

The runner strictly binds:

- partition
  `3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`;
- semantic catalog
  `ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`;
- validation-only sample report
  `b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579`;
  and
- VAMP experiment config
  `ca16318486600745e8a49903f495819741082f120fa7b95b3f9277efa83ada73`.

The base remains the registered seed-zero GPT-Neo model, optimizer, schedule,
two-epoch calibration, five-epoch maximum, empirical semantic gate, and 12 GiB
allocator limit. Version-6-native resume and selected-base formats prevent an
archive-v1 or semantic-v1 checkpoint from entering the run.

The base writes strict state every 1,000 optimizer updates and at each epoch.
After an interruption, the runner restores the newest complete optimizer,
random, schedule, and next-batch cursor and removes only log records newer than
that checkpoint. Completed loss ledgers are published atomically; incomplete
validation attempts are preserved under a recovery directory before replay.

If the base passes, the continual order is exactly A, B, C, D, E. The study
trains a continually overwritten LoRA, independent root LoRAs, and VAMP under
the same rank-eight, alpha-eight, 2,000-update budget per world. Parent search
and content keys use only deterministic validation spans.

The final test comparison contains four stored methods and five task-free
routers. It reports all four prefix lengths, visible-cue strata, forgetting,
parent transfer, memory, synchronized routing cost, and forced-adapter
specificity against both persisted comparison arms. Specificity is a
diagnostic 10,000-replicate paired bootstrap, not a new scientific gate.

## Implemented boundaries

Base calibration writes per-group loss sums and active-token counts for the
held-in set, every world, and every comparison. Epoch two must pass the same
mean/world bootstrap, label-swap, Holm, held-in quality, and memory rules that
will define checkpoint eligibility at later epochs. Test data is absent from
this selection path.

Adapter training stores a complete immutable artifact at every world boundary.
The artifact contains sequential, independent, and VAMP adapters; their three
separate random streams; the graph and address book; parent scores; and update
loss traces. A resumed run accepts only a contiguous prefix of these stage
artifacts.

After all model choices are frozen, the runner creates one durable sealed
transaction. Only then can it read or even count test indexes. The final
content-addressed publication authenticates the base test, complete nine-
method measurement JSONL, paired-control ledgers, test-suite provenance,
metrics, Markdown report, and standalone HTML report. Re-entering the same
transaction can finish interrupted reporting but cannot change its bound base,
adapters, partition, or config.

An interrupted base test is preserved and restarted inside that same durable
transaction. Adapter training and the final result both record the largest
observed allocator peak and fail above the frozen 12 GiB limit.

## Validation-only real-source check

The complete canonical partition passed a strict reload before the new anchor
selector was exercised. It selected 128 full 256-token spans for the held-in
root and 128 for each of worlds A through E. All 768 sequence hashes were
unique, each came from a distinct duplicate-story group within its source, and
every selected span was marked `validation`. No test index was read.

## Measured GPU preflight

The RTX 4090 preflight publication is
`b7f49909368685a5494a3033e0df7df69cf2e8c1064092c541013b873671988d`.
It is stored under
`checkpoints/tinyworlds-p-semantic-v6/preflight/` and strictly reloads.

The preflight used a separate identity and resume format, performed exactly two
disposable optimizer updates, and evaluated one warm validation batch. It did
not compute a semantic gap and did not read the sealed test.

| Measurement | Result |
| --- | ---: |
| Update 1 NLL | 10.856969 |
| Update 2 NLL | 10.851126 |
| Warm optimizer update | 0.467411 seconds |
| Warm validation batch | 0.015083 seconds |
| Allocator peak | 9,030,551,296 bytes (8.41 GiB) |
| Updates per epoch | 9,730 |
| Validation batches per epoch | 2,645 |
| Estimated two-epoch calibration | 2:32:56 |
| Estimated five-epoch validation-only base path | 6:22:19 |
| Adapter-training proxy | 3:53:42 |

The adapter estimate deliberately uses the measured base-update time as a
proxy because no selected base exists yet. It is useful for scheduling, not a
claimed adapter benchmark.

## Registered base-calibration result

The reviewed real run authenticated the fixed partition, sample report,
experiment config, and preflight before performing 19,460 seed-zero optimizer
updates. Epochs one and two took about 2 hours 17 minutes. Held-in validation
NLL fell from `1.318863` to `1.241619`, a decrease of `0.077244`, and the peak
allocator use was 9,160,916,224 bytes (8.53 GiB). These results passed the
registered training-quality and memory checks.

The epoch-two scientific contrast failed every aggregate evidence requirement:

| Measurement | Observed | Required |
| --- | ---: | ---: |
| Mean world-minus-control gap | -0.001364 nats/token | at least 0.048790 |
| 95% paired-bootstrap interval | [-0.009339, 0.006350] | lower bound above 0 |
| One-sided mean placebo probability | 0.631937 | at most 0.01 |

The five individual point estimates were `0.010117` for A, `0.009586` for B,
`-0.005320` for C, `-0.005461` for D, and `-0.015741` for E. Every individual
95% interval crossed zero, and none of the one-sided placebo tests passed the
registered familywise requirement. A and B therefore provided only small,
uncertain positive differences; C, D, and E went in the opposite direction.

The authenticated result is
`checkpoints/tinyworlds-p-semantic-v6/work/base-calibration/calibration.json`.
Its decision is `semantic_grid_failure`. The runner consequently did not train
epochs three through five, select a base checkpoint, construct any adapter, or
create the sealed-test transaction. The sealed test remains unopened.

## Verification

Focused tests cover the existing partition reconstruction, leakage, pairing,
tamper, per-group evaluation, language adaptation, and empirical-null paths,
plus the new deterministic specificity statistic, frozen router contract,
empty-prefix cue handling, live measurement sink, task-boundary resume parity,
strict sealed returns, and periodic-checkpoint log repair. The final pinned-
environment group passes all 61 tests. The new modules
and fixed runner compile in both the development and pinned semantic
environments. The canonical validation anchors also passed the real-source
check described above.

## Recorded stop and next decision

Semantic-v6 is complete negative evidence at its registered base gate. Running
the same frozen configuration again cannot authorize the blocked downstream
stages. The next useful work is to diagnose, using only training and validation
evidence, why the held-out semantic cells were no harder than the matched
controls. Any changed benchmark mechanism or gate must be preregistered as a
new version. The semantic-v6 test split remains sealed throughout that work.
