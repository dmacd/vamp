# ImageNet-R full-union integrator handoff

Run `7f8ac3ef574fe7ec3a2097c3a4b8a8ed13c5c1e4f34a856d69b6c32108a6a946`
ended in `COMPLETE_HISTORY_SELECTION_FAILURE` after 15 minutes 5 seconds. This
was a predeclared scientific stop, not a crash. Source commit `1d00249` removed
fixed-K parent consolidation from the primary condition. Every parent trained
on the complete fit-image union represented by its children.

## Clean validation result

| tasks | fresh full replay | raw full-union control | H=512 | H=1024 | H=2048 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 88.265% | 85.714% | 88.265% | 88.265% | 88.265% |
| 4 | 83.555% | 83.186% | 83.628% | 82.743% | 84.071% |
| 8 | 80.958% | 79.505% | 79.270% | 79.976% | 80.801% |
| 16 | 77.994% | 77.284% | 76.857% | 76.857% | 77.162% |

H=2,048 met the replay-proximity gates at every checkpoint. At task 16 it was
0.832 percentage points below fresh replay, inside the allowed 1.0-point gap.
It failed the independent control-margin gate: the protocol required at least
80.284%, three points above the 77.284% static control. H=2,048 was 3.122
points short. The fresh integrator was also 2.289 points short, so increasing H
cannot make the tested architecture pass unless it exceeds its observed fresh
ceiling.

## Interpretation

The fixed-K bottleneck is gone. The remaining failure is not primarily bounded
integrator replay. At task 16, the capacity-one binary-counter frontier has one
live full-union root. The integrator can recalibrate that root's prediction but
cannot combine child-adapter behaviors that no longer exist in its input. The
successful sealed diagnostic was optimistic because it exposed eight retained
U100 nodes instead of this one-node power-of-two frontier.

This run does not measure task-50 or test accuracy. Its gate correctly kept the
6,000 test images and the downstream local E2-LoRA comparison unopened.

## Highest-information next experiment

Stay on the clean fit/validation split and measure the fresh task-16 ceiling
under matched parent checkpoints while varying only information retained across
a carry:

1. Current single root, as the frozen reference.
2. Root plus its two retired children.
3. The four grandchildren and all 16 leaves as diagnostic ceilings.
4. A single deployable parent trained to distill an adapter-dependent child
   mixture, including the R3 router teacher, on the same full union.

This distinguishes missing child information from insufficient parent
distillation. Only after one bounded/deployable representation beats the
full-union root by the required margin should persistent H be tuned again. Do
not lower the margin or enlarge H merely to continue to task 50; neither action
tests the bottleneck observed here.

## Verification and reuse

- 52 focused serial ImageNet-R tests passed.
- 6 authenticated real-data/model/GPU tests passed on the RTX 4090.
- The broad serial suite had only the known missing optional dependencies: 8
  FabricPC failures and 23 TinyWorlds `tokenizers` failures.
- An identical rerun completed in 3.994 seconds, reused all six phases, and
  performed zero additional leaf or parent optimizer steps. Content and mtimes
  were unchanged for all 16 leaf checkpoints, 15 parent checkpoints, 254
  hierarchy metadata records, 11 scientific JSON records, and the behavior
  ledger. See `../evaluations/reuse_proof.json`.
