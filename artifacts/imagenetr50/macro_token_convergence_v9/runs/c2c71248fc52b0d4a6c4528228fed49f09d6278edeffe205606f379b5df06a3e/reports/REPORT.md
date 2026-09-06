# ImageNet-R Stage-31 Macro-Token Convergence Audit

## Main result

The predeclared selection rule minimized clean validation NLL. It chose effective batch 64 and peak learning rate 3e-05. At each seed's minimum-NLL checkpoint, this schedule reached 74.341% mean validation accuracy and 1.0929 mean NLL (1993: 74.483%, 1994: 74.549%, 1995: 73.991%). The same-split joint-IID control reached 77.862% accuracy and 0.9425 NLL after its fixed fifth epoch. The selected macro mean therefore differed from joint IID by -3.520 accuracy points and +0.1504 NLL.

Checkpoint policy changes the accuracy diagnosis. The selected schedule's per-seed maximum accuracy averaged 76.233% (1993: 76.451%, 1994: 75.894%, 1995: 76.353%), still -1.629 points from joint IID. In the exploratory nine-cell screen, batch 128 with peak learning rate 0.0003 reached 78.091% at epoch 33, +0.230 points relative to joint IID, but its NLL was 1.1902. That one-seed, post-screen maximum was not replicated and is not the selected estimate.

The exact legacy rerun reached 73.762% at its minimum-NLL checkpoint (epoch 2). The selected screening cell changed seed-1993 accuracy at minimum NLL by +0.722 points. Every selected run peaked before epoch 50; more epochs under the same schedule are not supported by validation NLL.

## What was tested

All macro models used the same one-block 12,055,496-parameter architecture and the same frozen stage-31 hierarchy representations. Seed 1993 crossed effective batches 64, 128, and 512 with peak AdamW learning rates 3e-5, 1e-4, and 3e-4. Each cell used a five-percent linear warmup followed by cosine decay through epoch 50. The legacy control used constant 3e-4 for 20 epochs. The selected schedule was repeated with seeds 1994 and 1995.

The joint-IID control used the identical 12,194 fit and 3,049 validation identities, but it trained a fresh rank-16 QKV-plus-fc1 LoRA and affine 124-class head for five epochs. It therefore measures what joint feature adaptation can achieve on this clean split; it is not a gate.

## Interpretation

The macro models reached near-perfect fit accuracy, while every selected seed minimized validation NLL by epoch 9. More optimizer work therefore does not close the probabilistic gap: after those checkpoints, validation NLL rises even when top-1 accuracy sometimes improves. The exploratory 78.091% maximum shows that the fixed macro inputs can support a competitive decision boundary on this development split, but its much worse NLL and lack of replication point to calibration and generalization instability rather than simple non-convergence.

The comparison cannot isolate classifier architecture from representation quality. Joint IID adapts one LoRA jointly over all 124 classes, while the macro classifier receives frozen node-specific features. The joint model's 0.9425 validation NLL, achieved before its fit accuracy saturated, is evidence that joint feature adaptation produces a cleaner representation than merely fitting the macro classifier longer.

The experiment never requested a test image. These are development measurements only and do not revise the locked-test result from v8.

## Reproducibility

The source v8 run, fit-only hierarchy, split, environment, code, and resolved configuration are content-addressed. Epoch rows were hash-chained and fsynced before the next epoch. An immediate cache replay used 76,215 cached node-example rows, performed zero adapted-token forwards, and preserved both population identities.
