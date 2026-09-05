# ImageNet-R Stage-31 Macro-Token Convergence Audit

This clean-only follow-up tests whether the stage-31 macro-token deficit comes
from the original optimizer schedule rather than from the fixed hierarchy
representations. It authenticates the completed v8 macro-token study and reuses
the same stage-31 clean hierarchy, 12,194 fit identities, and 3,049 validation
identities. It never requests a test image and does not produce a locked-test
number.

## Macro-token matrix

The architecture remains the v8 one-block, 12,055,496-parameter macro-token
classifier. Seed 1993 crosses effective batch sizes 64, 128, and 512 with peak
AdamW learning rates 0.00003, 0.0001, and 0.0003. Every cell trains for 50
epochs. Its per-update learning rate warms up linearly over the first five
percent of updates, then decays on a cosine curve to one percent of the peak.
Dropout 0.1, weight decay 0.0001, BF16 compute, and gradient clipping at 1.0
remain fixed. A separate control exactly repeats the old constant-0.0003,
effective-batch-512 recipe for 20 epochs.

The screening winner has the lowest validation negative log likelihood (NLL)
at any epoch. Ties prefer the smaller effective batch and then lower learning
rate. The winning schedule is initialized independently and repeated with
seeds 1994 and 1995. Full epoch histories are fsynced to hash-chained JSONL as
training proceeds. Each history reports the training pass objective, clean
validation NLL and accuracy, current learning rate, image presentations, and
optimizer updates. Best checkpoints are selected only from validation NLL.

## Matched joint-IID control

A fresh rank-16 QKV-plus-fc1 LoRA and affine 124-class head trains on exactly
the same 12,194 fit identities. It uses the sealed stage-matched joint-IID
recipe: five epochs, batch size 64, SGD with momentum 0.9, LoRA learning rate
0.0005, head learning rate 0.01, and weight decay 0.0005. Every epoch is
evaluated on the same 3,049 validation identities. The primary comparison uses
the fixed fifth epoch; the best validation epoch is reported as a diagnostic.

## Interpretation boundary

If a macro schedule approaches the matched joint-IID validation accuracy, the
old optimization protocol was the main limitation. If macro training accuracy
approaches 100 percent while validation remains materially lower than joint
IID, then additional fitting alone does not explain the gap. That outcome
would prioritize regularization, auxiliary soft ownership supervision, or a
different mechanism for combining the fixed node-specific representations.
The audit cannot support a new test claim because the locked test remains
sealed.
