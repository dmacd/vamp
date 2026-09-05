# ImageNet-R-50 Macro-Token Integrator Ceiling Study

## Abstract

This experiment tests whether the fragmented-frontier deficit comes from discarding patch-level information before integration. Every live node supplies its full final 197-token, LoRA-adapted ViT representation. Corresponding token positions are fused across six stable hierarchy levels, then integrated by a small transformer. Architecture choice used an end-to-end clean 19,200/4,800 fit/validation hierarchy. Only the selected model and a data-matched v6 final-CLS MLP were refit on all training images and evaluated on locked test.

**Main result.** The selected macro-token model changed locked-test accuracy relative to the data-matched v6 MLP by +2.123 points at stage 31 and +2.061 points at stage 50. Its signed differences from stage-matched joint IID were -5.171 and -0.978 points. At stage 50 it differed from local E²-LoRA by -5.106 points.

## Locked-test comparison

| Stage | Macro-token mean | v6 MLP mean | Raw union | True-node oracle | Joint IID | Macro − joint |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 75.122 | 73.000 | 68.632 | 83.700 | 80.294 | -5.171 |
| 50 | 77.889 | 75.828 | 72.800 | 81.117 | 78.867 | -0.978 |

Seed ranges are shown in `accuracy_comparison.png`; they are observed three-seed ranges, not confidence intervals. Stage-matched joint IID uses the same pinned rank-16 QKV-plus-fc1 LoRA and affine classifier architecture as each fresh consolidation node, trained jointly on the exact prefix training set for five epochs. It is an offline ceiling, not a gate.

The local E²-LoRA reference is 82.995% final incremental accuracy on the same frozen split. The published 83.960% is external protocol context, not a matched threshold. Both appear only at stage 50 because no stage-31 E²-LoRA measurement exists.

## Clean architecture selection

The predeclared seed-1993 sweep crossed one or two transformer blocks with learning rates 0.0001, 0.0003, and 0.001 at stages 31 and 50. The winner was depth 1, learning rate 0.0003, selected by mean clean validation NLL 1.090314. Ties would have preferred fewer blocks and then lower learning rate. Seeds 1994 and 1995 repeated only this winner. Rejected cells never saw test.

## Owner information

Owner labels were never part of inference inputs. A frozen linear probe asks whether class-trained macro-CLS already makes the owning slot linearly available. A separate end-to-end owner transformer asks how much the same input can expose when optimized directly for ownership. At stage 31, frozen linear owner probe reached 74.161% owner accuracy and 65.068% routed class accuracy. At stage 31, end-to-end owner model reached 81.027% owner accuracy and 72.956% routed class accuracy. At stage 50, frozen linear owner probe reached 84.450% owner accuracy and 72.067% routed class accuracy. At stage 50, end-to-end owner model reached 86.600% owner accuracy and 74.300% routed class accuracy. The true-node oracle remains label-aware and diagnostic; predicted-owner routing is task-free.

## Architecture and data boundary

Each active node produces a normalized 197 × 768 sequence with its own LoRA installed. The shared cross-slot projection is exactly equivalent to `Linear(4608, 768)` over six zero-padded stable slots, without materializing the zeros. A 3,606-value behavior vector—raw logits, local log probabilities, ownership, and active bits—becomes one META token. The selected one- or two-block transformer processes 198 width-768 tokens and classifies directly from macro-CLS. It has no raw-union residual skip. Exact trainable parameter counts are 12,055,496 for one block and 19,143,368 for two.

The v6 control consumes the same cached node computations, but retains only final CLS plus behavior fields in its 8,214-value input. Clean upstream nodes never trained on validation images. Locked refits used all prefix training images for each model's own clean-selected epoch count. The training seal recorded zero test-token requests.

## Resource and reproducibility boundary

Full token sequences were BF16 stage-local scratch in immutable 64-image shards with a 64-GiB cap. They were removed after model/evaluation seals. Request and model manifests retain exact cache bytes, adapted-node forward counts, optimizer work, wall time, and peak VRAM. The immediate reuse proof required zero new hierarchy optimizer steps, zero new adapted-token requests, byte-identical model artifacts, and empty token scratch.

`inference_cost.csv` reports head-only dense multiply-accumulates and the common number of node-adapted ViT forwards per image. It excludes LayerNorm, nonlinearities, softmax, and data movement, so it is an architectural workload estimate rather than a latency measurement.

## Interpretation

The macro-versus-v6 contrast isolates patch retention and spatial cross-node integration under matched data, source nodes, and optimizer-selection protocol. A positive contrast supports the claim that pooled final CLS omitted useful cross-node evidence; a small or negative contrast shifts attention toward node quality, regularization, or the learning target. The owner probes distinguish information availability from class-head exploitation, but do not by themselves establish a causal routing mechanism. Two stages and three classifier seeds are enough for a ceiling study, not a final SOTA claim.

## Artifacts

`stage_summary`, `clean_candidates`, `clean_summary`, `owner_diagnostics`, `task_accuracy_matrix`, and `resource_accounting` are emitted as CSV, JSON, and Parquet. `REPORT.html` is self-contained. `protocol/architecture_selection.json`, `protocol/training_seal.json`, and `protocol/reuse_proof.json` preserve the selection, leakage, and resume boundaries.
