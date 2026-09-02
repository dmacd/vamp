# Five-seed and two-node-level amendment for 100-permutation scaling

## Purpose

Extend the completed single-seed 100-permutation scaling study to measure
run-seed variance and compare two temporal consolidation capacities. This is a
new authenticated successor. It does not modify the predecessor run at
`6d4f13fdf7d3aad964b1d8becae3fa7130e05422548576912e6e830506a3710c`.

## Fixed factors

- Run seeds are exactly 0, 1, 2, 3, and 4.
- The 100-domain definitions, stream-order seed, per-step data allocations,
  calibrated 1024/1024/512 base, node optimizer, integrator optimizer, replay
  budget, and evaluation subset size remain those in
  `logt_vamp_permuted_mnist_100_scaling_protocol.md`.
- The permutation order is fixed across run seeds. The five seeds vary source
  examples, test subsets, node fits, integrator initialization, dropout,
  minibatches, and replay draws. They do not measure permutation-order
  variance.
- The two integrator training conditions remain the persistent four-epoch
  uniform-replay integrator at every step and the fresh 20-epoch full-replay
  integrator at steps 1, 2, 4, 8, 10, 16, 26, 41, 66, and 100.
- The 20-epoch full-replay condition is a fixed-budget comparator, not a
  converged ceiling.

## Consolidation policies

`one-node-per-level` is the predecessor binary-counter policy. A second node
at an occupied level immediately merges with the existing node and carries to
the next level.

`two-nodes-per-level` retains as many as two nodes at each level. When a third
node would occupy a level, the two older resident nodes merge and carry to the
next level; the newest node remains at its current level. Carries apply the
same rule recursively. Every node still covers a contiguous power-of-two
interval, and every frontier exactly partitions the learned stream prefix.

Each level has two persistent input positions. All seven primary positions
come first and exactly match the predecessor's seven input positions. The
seven secondary positions are appended. A level's first resident uses its
primary position and its second resident uses its secondary position. On
overflow both residents disappear, so no surviving node changes position.
Secondary input weights start at exact zero. All shared input weights and all
downstream weights are copied from the paired one-node initialization, making
the two policy initializations identical on their shared inputs.

## Comparisons and accounting

Both integrator training conditions run under both consolidation policies for
all five seeds. Replay examples, optimizer randomness, and evaluation examples
are paired by run seed across policies. Each policy has its own matched frozen
hierarchy and full-replay fit because its active features differ.

Training-only forward and backward example-passes retain the predecessor's
accounting boundary. For capacity `c`, let `a_c(k)` be the measured active-node
count after step `k`; `a_c(k) = O(log k)` for fixed `c`. After step 1, uniform
replay uses `512 a_c(k) + 2,048` forward example-passes and 2,048 backward
example-passes per update. A full-replay fit uses
`256k a_c(k) + 20 * 256k` forward example-passes and `20 * 256k` backward
example-passes. Shared hierarchy construction is reported separately for each
policy and seed.

The aggregate report gives means and sample standard deviations across five
seeds, preserves every seed-level metric in CSV, and reports paired
two-node-minus-one-node accuracy differences. Policy comparisons are paired by
seed. Wall time is descriptive; example-pass counts determine the complexity
claim.

Seed 0 under `one-node-per-level` must reproduce every predecessor accuracy,
cross-entropy, and model-pass count exactly. Wall-clock values are excluded
from this check. This guards the comparison against an accidental change to
the original condition while adding capacity.

## Execution and outputs

The single entry point remains:

```bash
.venv-vision/bin/python -m apm.experiments.vamp_logt_mlp_permuted_scaling
```

Each policy/seed hierarchy and integrator run has its own resumable boundary
and chained ledger. The aggregate Markdown, self-contained HTML, plots, CSV,
and JSON summary are regenerated only from completed authenticated ledgers.
