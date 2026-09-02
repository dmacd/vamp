# Sample-calibrated 100-permutation integrator protocol

## Question

Can the reference-sized frozen-node hierarchy support at least 95% accuracy
when node and integrator supervision are increased, and how does persistent
uniform replay compare with fresh full replay at that calibrated sample count?

This is a seed-0 successor to the completed 100-permutation capacity study. It
keeps that study's identity-plus-99-permutation stream, task order, one-node-
per-level consolidation policy, base architecture, integrator architecture,
optimizers, and evaluation subsets. The only calibrated resources are paired
model/observer examples per task and the fresh full-replay epoch count.

## Fixed model and stream

- Base MLP: 784-1024-1024-512-10, imported byte-for-byte from the authenticated
  dense-base run.
- Integrator MLP: 3661-1024-512-256-10, with seven stable temporal slots.
- Tasks: the existing fixed identity-plus-99-permutation order.
- Run seed: 0. Consolidation: at most one node per temporal level.
- Node training: 20 AdamW epochs. Persistent integrator training: four AdamW
  epochs per task with equal current and historical loss weights.

The earlier capacity experiment found only a 1.55-percentage-point full-replay
gain from roughly quadrupling model parameters at task 100, versus a
5.74-point gain from doubling examples. This successor therefore starts with
the smaller model. A larger model is not run if the smaller model passes the
accuracy requirement.

## Validation-only sample calibration

Candidate multipliers are tested in ascending order: 1, 2, 4, 8, 16, and 32.
Each multiplier supplies `256 * multiplier` model examples and the same number
of disjoint observer examples per task. The original 128 held-out training
examples per task remain fixed across multipliers; added rows never replace
base model, observer, or held-out assignments.

For each candidate, build the hierarchy through task 10. At each prefix from
1 through 10, initialize a fresh integrator, train on all observer examples in
that prefix for 20 epochs, and measure after every epoch on the concatenated
held-out training examples for the learned domains. Final MNIST test examples
are not read during selection.

The candidate passes at epoch `e` only if learned-domain mean held-out accuracy
is at least 95% at every one of the ten prefixes at that same epoch. Select the
first candidate multiplier with any passing epoch, then select its earliest
passing epoch. Stop the sample search immediately. This is deliberately more
stringent than requiring only the mean of the ten prefix accuracies to reach
95%.

If no candidate passes by 32x samples and 20 epochs, stop with a visible failed
calibration result. Do not run the 100-task conditions or silently enlarge the
model.

## Endpoint-first execution and time limit

After selection, extend the selected hierarchy to all 100 tasks and fit the
fresh full-replay task-100 endpoint before the persistent condition or other
full-replay checkpoints. Record its final test accuracy, wall time, frozen-node
forward work, integrator forward work, and integrator backward work.

Use the endpoint's measured seconds per counted model pass to project the
remaining fixed full-replay fits and persistent updates. Include elapsed
calibration/hierarchy time and a five-minute reporting reserve. Always run
fresh full replay at tasks 1, 2, 4, 8, 10, and 100. Run tasks 16, 26, 41, and 66
only if the projected total remains at or below 3,600 seconds. Persist this
decision before further training so resume cannot change it.

### Bounded-memory endpoint correction

Two task-100 endpoint attempts were killed before publishing any result because
the initial implementation materialized the complete transformed image archive
and its roughly 12 GB fixed-slot feature matrix at the same time. The corrected
implementation preserves the scientific condition but changes storage: it
computes each frozen-node feature exactly once, writes feature, ensemble, and
label rows to one temporary float32 memory map, and trains from that map in the
same seeded global minibatch order. At most one per-task feature block and one
optimizer minibatch are held in anonymous memory. The cache is deleted after
the full-replay checkpoint is published.

The original immutable `protocol.json`, completed validation-only calibration,
and completed 100-task hierarchy remain authoritative. A separate immutable
`protocol-amendment-oom-streaming.json` authenticates the storage-only code and
document changes and records that both failed attempts were unpublished. Work
counts remain one frozen-feature pass per example plus the selected number of
integrator forward and backward passes. Disk writes and shuffled reads are
included in data-preparation wall time.

On resume, elapsed time through the already-completed hierarchy is reconstructed
from the persisted selection duration and artifact completion times. This
prevents a fresh process clock from incorrectly restoring the optional
checkpoints after the one-hour budget has already been spent.

## Final conditions

| Condition | Training at task `t` |
|---|---|
| Persistent uniform replay | Continue one integrator for four epochs on `N` current examples and, after task 1, `N` examples sampled uniformly from all earlier observer examples. Here `N` is the selected per-task observer count. |
| Fresh full replay | Initialize a new integrator and train it for the selected fixed epoch count on all `N * t` observer examples seen through task `t`. |

Both conditions use the same selected hierarchy, task order, observer archive,
and final test subsets. Test evaluation never changes training. Calibration
validation passes are reported separately because the selection depends on
them; final-condition work excludes all post-selection evaluation.

## Required artifacts and checks

The existing scaling entry point must remain the only runner. It writes
resumable calibration rows as they are produced, hierarchy and integrator
checkpoints for crash recovery, a persisted schedule decision, separate work
counters, compact CSV/JSON evidence, distinct high-contrast figures, and
self-contained Markdown/HTML reports. Acceptance requires the exact selected
sample/epoch rule, sealed test selection, task-100-first full replay, exact
replay budgets, exact work accounting, the persisted time decision, a complete
100-task persistent trace, the scheduled full-replay cells, deterministic
resume, focused serial tests, and an explicit CUDA run.
