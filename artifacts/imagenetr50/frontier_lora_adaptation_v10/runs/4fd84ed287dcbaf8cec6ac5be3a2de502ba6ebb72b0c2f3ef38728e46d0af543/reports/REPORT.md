# ImageNet-R stage-31 frontier-LoRA adaptation

## Result

The minimum-NLL macro-token condition is **Macro-token frontier, adaptive LoRAs (H=11,827; full fit)**, with
**85.864% validation accuracy**
and **0.5710 NLL** at epoch
13. The largest observed macro-token accuracy is
**85.864%** from
**Macro-token frontier, adaptive LoRAs (H=11,827; full fit)** at epoch 13,
where NLL is 0.5710.
Macro-token frontier, adaptive LoRAs (H=4,096) is the smallest H to reach both rank-16 joint-IID metrics, first doing so at epoch 6.

The rank-16 joint-IID reference is 77.862% / 0.9425
NLL. The previous frozen cached macro result at the same seed is
74.483% / 1.0903; the new
augmentation-matched frozen control is
75.533% /
0.9967.

At the selected full-history checkpoint, LoRA adaptation gains
10.331 percentage points and reduces NLL by
0.4257 relative to the otherwise matched frozen-LoRA
control. At exactly 5 full-fit passes (60,970
image presentations), the adaptive frontier reaches
84.880% /
0.5897; the frozen frontier reaches
72.155% /
1.1788; and joint IID reaches
77.862% / 0.9425. The data exposure is
matched at this checkpoint. Compute is not: the frontier evaluates five
specialized ViTs plus the macro transformer, while joint IID evaluates one
ViT with one shared LoRA and classifier.

## Replay scaling across architectures

### Minimum-NLL-selected frontier checkpoints

| H | Macro-token frontier | Single-affine frontier |
|---:|:---|:---|
| 1,024 | e31: 57.330% / 1.9572 | e2: 71.925% / 1.2792 |
| 2,048 | e25: 71.564% / 1.2565 | e3: 73.434% / 1.2058 |
| 4,096 | e12: 80.125% / 0.8360 | e4: 75.992% / 1.0586 |
| 8,192 | e8: 83.601% / 0.6430 | e5: 78.944% / 0.9737 |
| 11,827; full fit | e13: 85.864% / 0.5710 | e3: 79.239% / 0.8907 |

Each cell is `epoch: accuracy / NLL`. Rank 80 is omitted here because its
predeclared primary endpoint is epoch five, not a 50-epoch selected checkpoint.

### Fixed epoch-five checkpoints

| H | Fit identities | Macro-token frontier | Single-affine frontier | Joint IID, rank 80 |
|---:|---:|:---|:---|:---|
| 1,024 | 1,391 | 50.738% / 2.2827 | 71.368% / 1.3873 | 48.049% / 2.7526 |
| 2,048 | 2,415 | 67.727% / 1.3817 | 72.680% / 1.2629 | 60.512% / 1.8930 |
| 4,096 | 4,463 | 77.796% / 0.9260 | 76.320% / 1.1238 | 70.548% / 1.3572 |
| 8,192 | 8,559 | 83.732% / 0.6621 | 78.944% / 0.9737 | 76.812% / 1.0353 |
| 11,827; full fit | 12,194 | 84.880% / 0.5897 | 78.649% / 0.9560 | 80.125% / 0.8339 |

Each metric cell is `accuracy / NLL`. All three models have completed five
passes over exactly the same identity set within each row.


Under both checkpoint views, the
single-affine frontier wins at H=1,024 and H=2,048, while the macro-token
frontier wins from H=4,096 onward. At fixed epoch five, the affine accuracy
lead falls from 20.630 to
4.952 points between H=1,024 and H=2,048; the macro
lead is then 1.476 points at H=4,096 and
4.788 points at H=8,192. Rank-80 joint IID trails both
frontier heads at every truncated H. At full history it passes the affine head
(80.125% /
0.8339 versus
78.649% /
0.9560), but still trails the macro
head (84.880% /
0.5897).

![Accuracy and NLL versus H](accuracy_nll_vs_h.png)

## Joint-IID capacity controls

The new rank-80 joint-IID control reaches
**80.125% accuracy / 0.8339 NLL**
at the fixed epoch-five endpoint. Increasing the joint adapter from rank 16 to
rank 80 raises accuracy by 2.263 percentage points and
lowers NLL by 0.1085. This recovers 32.2%
of the adaptive frontier's accuracy advantage and 30.8% of
its NLL advantage over rank-16 joint IID. At the same five-pass checkpoint,
the adaptive frontier remains 4.756 accuracy points
higher and 0.2443 NLL lower than rank-80 joint IID.

Both sides expose exactly 6,635,520 trainable LoRA parameters and 60,970 image
presentations. That is the full extent of the match. The adaptive frontier also
trains a 12,055,496-parameter macro integrator, starts from five separately
pretrained rank-16 adapters, and executes five ViT paths. Rank-80 joint IID
starts one adapter from the standard zero-effect initialization, trains a
95,356-parameter classifier, and executes one ViT path. Its minimum NLL within
the same five epochs is 0.7954 with
80.026% accuracy at epoch
3; this diagnostic does not replace the fixed
epoch-five comparison.

The total-active-parameter control uses rank 224 and reaches
**81.109% accuracy / 0.7772 NLL**
at epoch five. Its 18,674,812 trainable parameters are 16,204
(0.087%) fewer than the adaptive frontier's
18,691,016, the closest possible match at integer rank. Relative to rank 80,
the extra capacity changes accuracy by +0.984
percentage points and NLL by -0.0568. Relative to rank
16, rank 224 accounts for 46.3% of the adaptive
frontier's accuracy difference and 46.9% of its NLL
difference. At the same endpoint, the adaptive frontier remains 3.772 accuracy points higher, while
the adaptive frontier is 0.1875 NLL lower. The rank-224 minimum NLL is
0.7199 at epoch
3; accuracy peaks at
82.060% at epoch
4. Validation NLL then rises by
0.0573 through epoch five while the training objective
continues to improve, which is consistent with late overfitting or
miscalibration under the inherited rank-16 schedule.

## Single-layer integrator ablation

The single-layer pre-classifier integrator reaches
**79.239% accuracy / 0.8907 NLL** at its
minimum-NLL checkpoint, epoch 3. Relative
to the macro-token full-history checkpoint selected by the same rule, this is
-6.625 accuracy points and
+0.3197 NLL. At epoch five, where both have exactly
60,970 image presentations, the linear head reaches
78.649% /
0.9560, a change of
-6.232 points and +0.3664 NLL
from the macro head.

The linear head maps five concatenated 768-value node pre-classifier vectors
directly to 200 logits. Its exact-union initialization copies each frozen local
classifier into its owning input block; every cross-node block starts at zero.
It has 768,200 parameters, 93.6% fewer than the
12,055,496-parameter macro head. Both conditions train the same five source
LoRAs from the same initial tensors and use the same full-fit data, augmentation,
optimizer schedule, and validation selection rule.

## What changed

The task-31 frontier contains five sealed rank-16 LoRAs over disjoint task
intervals. Every adaptive frontier cell starts from those exact tensors. The
base ViT and all five node classifiers stay frozen; the five node LoRAs and
either the macro-token or affine head train jointly from task-free inputs. The
affine head reads only the five final pre-classifier vectors. Rank-80 joint IID
instead starts one zero-effect adapter and a 124-way classifier. Every
population includes all 367 current-task images. H is the same nested uniform
hash-order prefix of the 11,827-image historical partition for all three
architectures, so maximum H is exactly the 12,194-image full fit. The 3,049
validation identities remain excluded from optimization.

![Stage-31 frontier](stage31_frontier.png)

## Optimization behavior

![Validation learning curves](validation_learning_curves.png)

![Adapter displacement](adapter_displacements.png)

Both frontier heads use the previous minimum-NLL winner: effective batch 64,
peak AdamW learning rate 3e-5, and 50 warmup-cosine epochs. Newly adaptive node
LoRAs use peak 5e-4 under that schedule. Rank-80 joint IID retains its original
five-epoch SGD recipe. Frontier checkpoint selection is minimum validation NLL;
all three architectures are also reported at fixed epoch five. Maximum
accuracy is a separately labeled diagnostic.

## Interpretation boundaries

This is one seed on a validation split, not a final test estimate. Repeated
epoch evaluation makes the maximum-accuracy statistic exploratory. The
full-fit frozen control isolates online augmentation and image forwarding from
LoRA adaptation. H cells differ in both unique identities and optimizer steps,
but within each H the fixed-epoch comparison uses the same identities and five
complete passes. No test identity was requested. Exact replay authenticated the
six original macro cells, eight new architecture-sweep cells, and all full-fit
controls without a new optimizer step, while leaving the source hierarchy
unchanged.
