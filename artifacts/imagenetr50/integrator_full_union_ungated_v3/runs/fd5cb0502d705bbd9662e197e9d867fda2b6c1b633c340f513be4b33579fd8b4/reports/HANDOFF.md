# ImageNet-R ungated full-union integrator handoff

Run `fd5cb0502d705bbd9662e197e9d867fda2b6c1b633c340f513be4b33579fd8b4`
reached `COMPLETE` in 1 hour 26 minutes 18 seconds. Implementation commit
`8328b85` removed all accuracy gates and froze the feature family (`scores`),
history capacity (H=2,048), and full-union parent training before test access.
The offline joint-IID result is the primary ceiling and local E2-LoRA is a
secondary reference; neither controlled execution.

## Locked benchmark

| condition | final accuracy | incremental accuracy | difference from integrator, final | difference from integrator, incremental |
| --- | ---: | ---: | ---: | ---: |
| Full-union prediction integrator | 65.567% | 73.015% | — | — |
| Offline joint-IID rank-16 LoRA | 78.867% | 84.795% | +13.300 pp | +11.780 pp |
| Local E2-LoRA | 78.100% | 82.995% | +12.533 pp | +9.980 pp |
| Published E2-LoRA | 78.580% | 83.960% | external context | external context |

The method is not competitive with either local reference. This is one fixed
class order and one persistent-integrator seed, so it is not a variance
estimate. The local comparisons use the same immutable ImageNet-R split; the
published values do not establish protocol identity.

The locked task-50 frontier provides a more specific diagnosis:

| task-50 condition | test accuracy |
| --- | ---: |
| Persistent direct integrator | 65.567% |
| Raw union | 69.267% |
| Affine-calibrated union | 69.567% |
| True-node oracle (diagnostic) | 76.733% |

The direct integrator lost 3.700 points to raw union and 11.167 points to the
label-aware node oracle. Across all 50 stages it averaged 0.784 point below raw
union, improving raw union at 16 stages and losing at 33. At the power-of-two
stages, where the capacity-one frontier contains a single consolidated node,
it tracked or modestly improved raw union: task 2 was +1.210 points, task 4
+1.930, task 8 +1.032, task 16 -0.049, and task 32 +0.102. The large endpoint
gap therefore appears when the task-free model must combine several live
nodes, not when it merely recalibrates one root.

## Clean-development diagnosis

At task 50 on the untouched development validation partition, fresh full
replay averaged 68.340% over seeds 1993/1994/1995. Persistent H=2,048 training
reached 63.146%, a 5.194-point deficit. Fresh replay beat the best static union
by only 0.986 point and remained 6.181 points below the 74.521% true-node
oracle. This separates two problems:

1. Bounded persistent replay and/or its continual optimization path loses about
   five points relative to fitting the same architecture from scratch.
2. Even fresh direct integration recovers little of the information available
   to label-aware node selection.

On locked test, the 76.733% node oracle is itself 2.133 points below the
78.867% joint-IID ceiling. Better task-free integration alone therefore cannot
close the full endpoint gap with the present consolidated parents.

## Highest-information next experiments

Keep method selection on the 19,200/4,800 fit/validation split and do not tune
against this run's opened test results.

1. **Separate replay capacity from optimization path.** At task 50, compare
   fresh fitting, the current H=2,048 stream, full cumulative-history replay,
   and a step-matched full-history stream on exactly the same frozen frontier.
   This determines whether the 5.194-point loss comes from discarded observer
   examples or from warm-started sequential optimization.
2. **Make adapter-dependent R3 routing a main condition.** At tasks 16 and 32,
   where a carry collapses the frontier to one root, expose authenticated child
   responses to an R3 mixture teacher and distill that behavior into bounded
   parent state. Compare the current root, root plus retired children, the R3
   child mixture, and its deployable distilled parent. This tests whether a
   carry discards the alternatives needed by a direct integrator.
3. **Measure the parent ceiling explicitly.** Add an all-leaf true-task oracle
   and interval joint-IID parent controls on validation. Their differences from
   the current parent-node oracle quantify consolidation loss separately from
   routing loss. If the parent ceiling remains below joint IID, change parent
   training or distillation before tuning another router.
4. **Lock a successor only after the two losses shrink on validation.** Freeze
   its choices, then perform one paired test evaluation. Replicate across class
   orders or seeds before making a publication-level comparison.

## Integrity, reuse, and verification

- All 24,000 training identities were consumed before the training seal; zero
  of the 6,000 test identities appeared in training, proxy, or calibration
  requests before that seal.
- The locked result contains 50 stage metrics and all 1,275 stage/task cells.
- An identical rerun completed in 4.535 seconds, reused all six phases, and
  performed zero additional leaf, parent, or integrator optimizer steps.
- Content and nanosecond mtimes were unchanged for 100 leaf checkpoints, 94
  parent checkpoints, 100 persistent-integrator checkpoints, 1,576 hierarchy
  metadata files, 15 scientific JSON records, and the behavior ledger. See
  `../evaluations/reuse_proof.json`.
- All 52 focused serial ImageNet-R tests and both explicitly enabled real-model
  integrator GPU tests passed. The Markdown/HTML report and both plots were
  inspected.
