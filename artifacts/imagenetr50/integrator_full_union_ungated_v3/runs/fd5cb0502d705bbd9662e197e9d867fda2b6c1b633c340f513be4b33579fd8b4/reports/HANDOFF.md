# ImageNet-R ungated full-union integrator handoff

Run `fd5cb0502d705bbd9662e197e9d867fda2b6c1b633c340f513be4b33579fd8b4`
reached `COMPLETE` in 1 hour 26 minutes 18 seconds. Implementation commit
`8328b85` removed all accuracy gates and froze the feature family (`scores`),
history capacity (H=2,048), and full-union parent training before test access.
The offline joint-IID result is the primary ceiling and local E2-LoRA is a
secondary reference; neither controlled execution.

## Stage-matched joint-IID follow-up

The post-hoc available-data control is complete for all 50 stages. At each
stage it fits a fresh rank-16 LoRA adapter and affine classifier for five
epochs using only the immutable training examples from tasks available at
that stage, then evaluates the matching test prefix. It uses the same ViT,
LoRA targets, rank, optimizer, initialization seed, augmentation scheme, and
prefix-wide softmax recipe as the offline joint-IID control. Stage 50 reuses
and re-evaluates the authenticated offline model rather than fitting a second
nominally equivalent endpoint.

| tasks | true-node oracle | stage-matched joint | full-50 joint on prefix | future-data association | joint minus oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.763% | 94.915% | 97.458% | +2.542 pp | -0.847 pp |
| 2 | 88.710% | 88.710% | 97.177% | +8.468 pp | +0.000 pp |
| 4 | 84.035% | 85.614% | 93.860% | +8.246 pp | +1.579 pp |
| 8 | 81.801% | 84.803% | 91.651% | +6.848 pp | +3.002 pp |
| 16 | 77.789% | 82.396% | 86.324% | +3.928 pp | +4.607 pp |
| 32 | 74.917% | 80.276% | 81.577% | +1.301 pp | +5.359 pp |
| 50 | 76.733% | 78.867% | 78.867% | +0.000 pp | +2.133 pp |

The stage-matched curve reaches 78.867% final and 81.630% incremental
accuracy. The retrospective full-50 model reaches 84.795% incremental
accuracy, 3.164 points higher on average; over stages 1--49 its mean advantage
is 3.229 points (range -0.441 to +9.169). Future-task examples are therefore
associated with meaningful positive transfer at many prefixes, but they do
not explain the endpoint: at task 50 no future benchmark tasks exist and the
joint model still leads the true-node oracle by 2.133 points.

The power-of-two stages isolate node quality particularly cleanly. Their
frontier is one consolidated node, so correct-node routing and fragmentation
are absent, while the joint control sees the same task horizon and image
union. Across tasks 2/4/8/16/32, stage-matched joint exceeds the true-node
oracle by 2.283 points on average. At task 32 specifically, the oracle node is
the exact tasks-1--32 root retained at task 50: the full-50 model gains only
1.301 points from later tasks on that prefix, while the fresh stage-32 model
leads the same-data parent by 5.359 points. This directly identifies parent
training/head construction or optimization as a larger issue than missing
future information for the oldest final node.

Likely contributors, not yet separately identified, are the hierarchy's
union of child classifier rows instead of a freshly initialized global head;
zero parent weight decay versus 5e-4 for joint IID; different initialization,
sample order, and stateless augmentation seeds; repeated carry optimization;
and a five-epoch, single-seed budget that may leave either recipe
under-converged. Outside power-of-two stages, interval-local adapters also
lack cross-interval representation sharing and global negative-class
gradients. These are node-quality hypotheses: task-free routing cannot repair
the true owned node itself, although an R3 response integrator could exploit
complementary behavior from other nodes.

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

1. **Run a one-node parent factorial at tasks 16 and 32.** Hold the exact image
   union, task horizon, rank, epochs, and evaluation fixed while crossing fresh
   versus inherited classifier rows, zero versus 5e-4 weight decay, and several
   initialization/order seeds. Add a longer-budget arm only if the five-epoch
   curves have not stabilized. This is the shortest experiment that can
   attribute the 4.607/5.359-point same-data node deficits.
2. **Complete an interval-local decomposition of task 50.** The stage-32 model
   already supplies a fresh tasks-1--32 control. Fit only fresh tasks-33--48
   and tasks-49--50 models, evaluate each against its corresponding final live
   node, and also mask the full-50 joint model to the same intervals. This
   separates consolidation quality, cross-interval transfer, and global-head
   effects with two new fits instead of another full sweep.
3. **Measure hierarchy capacity and routing independently.** Compare the
   capacity-one and existing capacity-two true-node curves at matched stages,
   then test the R3 adapter-response condition as a main deployable condition.
   Preserve a true-node diagnostic alongside it: routing gains and stronger
   owned-node representations answer different questions.
4. **Only then revisit persistent integrator replay.** On the strongest frozen
   frontier, compare H=2,048 with full cumulative-history replay and a
   step-matched full-history stream. This isolates the already measured
   5.194-point persistent-versus-fresh task-50 gap without confounding it with
   an avoidably weak parent frontier.
5. **Replicate before a publication claim.** Freeze choices on development,
   run paired test evaluation once, and repeat across class orders or seeds.
   Treat joint IID as a descriptive offline ceiling and local E2-LoRA as a
   secondary comparator, never as execution gates.

## Integrity, reuse, and verification

- All 24,000 training identities were consumed before the training seal; zero
  of the 6,000 test identities appeared in training, proxy, or calibration
  requests before that seal.
- The locked result contains 50 stage metrics and all 1,275 stage/task cells.
- The accuracy figure separates selected clean-validation full-replay
  checkpoints from locked-test curves and includes both the authenticated
  50-stage offline joint-IID reference and the independently trained
  stage-matched joint curve; `stage_metrics.*` carries both joint curves.
- The stage-matched control trained 49 fresh prefix models and reused the
  authenticated task-50 model. It performed 3,020,210 image presentations and
  47,330 optimizer steps in 4,656.731 training seconds plus 152.784 evaluation
  seconds, with peak allocated VRAM of 4,144,351,232 bytes.
- Its immediate authenticated resume completed in 2.95 seconds with zero
  optimizer work. Protocol, 50-row ledger, and summary SHA-256 values remained
  `bd4e903c...`, `5e6f5f5b...`, and `ede45ac3...`, respectively.
- An identical rerun completed in 4.535 seconds, reused all six phases, and
  performed zero additional leaf, parent, or integrator optimizer steps.
- Content and nanosecond mtimes were unchanged for 100 leaf checkpoints, 94
  parent checkpoints, 100 persistent-integrator checkpoints, 1,576 hierarchy
  metadata files, 15 scientific JSON records, and the behavior ledger. See
  `../evaluations/reuse_proof.json`.
- All 61 focused ImageNet-R tests pass. The repository-wide suite has only the
  31 already documented optional-environment failures: eight require FabricPC
  and 23 require `tokenizers`; no unrelated assertion failed. The Markdown,
  self-contained HTML, seven-page A4 PDF, and both plots were inspected. The
  PDF's first render exposed a split-table header defect; the published rerender
  shortens human-facing content hashes while preserving exact IDs in
  `resource_accounting.*`, fits the table on one page, and has no clipping.
