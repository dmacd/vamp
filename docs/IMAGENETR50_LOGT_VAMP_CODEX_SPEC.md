# ImageNet-R-50 Log-t VAMP Vision Experiment
## Implementation and Experimental Specification for Codex

**Status:** implementation handoff

**Primary goal:** test whether logarithmic temporal consolidation of independently trained LoRA memories can approach joint-IID performance on a recognized class-incremental vision benchmark, and determine how much of the result depends on replay, parameter-space merging, and inference-time addressing.

---

## 1. Research question

Use ImageNet-R in the 50-task class-incremental protocol with a ViT-B/16 pretrained on ImageNet-21K. Each task introduces 4 new classes. Train one immutable task-local LoRA leaf per task. Maintain a log-t temporal bank with at most two live nodes per level. When a level overflows, consolidate the two oldest equal-level nodes and carry the result upward.

The main questions are:

1. Does the logarithmic hierarchy itself retain enough information to approach a jointly trained rank-matched model?
2. Can expensive union retraining at parent nodes be replaced by cheap LoRA merging?
3. If pure merging is imperfect, how much small replay repair is needed to close the gap?
4. How much of any residual error is due to consolidation versus task-free addressing?
5. Can all later merge/replay/router sweeps reuse the same 50 trained leaves with zero leaf retraining?

The experiment should report both benchmark accuracy and the resource tradeoff among training compute, replay compute, live parameter memory, archived experimental artifacts, and inference-time addressing cost.

---

## 2. External benchmark target and protocol anchor

Use the ImageNet-R class-incremental protocol released with E²-LoRA:

- 200 ImageNet-R classes.
- 50 sequential tasks.
- 4 new classes per task.
- class order generated from the PyCIL-style seeded permutation with seed `1993`.
- ViT-B/16 pretrained on ImageNet-21K (`vit_base_patch16_224_in21k` in the released code).
- 224x224 inputs.
- 5 training epochs per task in the released 50-task E²-LoRA config.
- batch size 64 in the released config.

Use the released E²-LoRA repository only as a protocol and external-baseline reference. Do not silently import its custom continual-learning machinery into VAMP.

Important: E²-LoRA is not a simple frozen-base fixed-rank LoRA baseline. Its released implementation dynamically allocates/compresses LoRA capacity, performs output-drift restructuring, uses distillation and classifier alignment, and gives the backbone a small non-zero learning rate in the released ImageNet-R config. Therefore:

- reproduce E²-LoRA separately as an external reference if practical;
- use a simpler controlled fixed-rank frozen-backbone setup for all internal VAMP causal comparisons;
- never claim that an internal VAMP-vs-fixed-LoRA comparison is directly apples-to-apples with E²-LoRA unless every relevant protocol detail actually matches.

Current literature target at the time this plan was written:

- E²-LoRA ImageNet-R-50: about **78.58% final accuracy** and **83.96% incremental average accuracy**.
- reported joint-training reference: about **82.76% final accuracy**.

Before any SOTA claim, rerun a literature check and verify exact protocol compatibility.

---

## 3. Mandatory implementation recommendation

**Train the 50 task-local LoRA leaves exactly once and keep them forever as immutable experimental inputs.**

This is a hard architectural requirement.

Changing any of the following must not retrain leaves:

- merge method;
- merge scaling coefficient;
- output rank;
- replay/repair fraction;
- repair optimizer settings;
- proxy set size;
- routing method;
- logit calibration method;
- evaluation implementation;
- report generation.

Likewise, never overwrite an intermediate parent produced under a different policy. Every derived node must be content/config addressed from:

- child node hashes;
- consolidation method;
- method hyperparameters;
- output rank;
- repair policy;
- repair sample IDs;
- software/git revision where semantically relevant.

The desired workflow is:

```
train 50 leaves once
        |
        +--> build SVD tree
        +--> build Core+TSV tree
        +--> build output-drift tree
        +--> build replay-retrained tree
        +--> rerun any tree with repair fraction / rank / scale changes
```

Leaf training is the only stage that must never be repeated for a merge-policy sweep.

---

## 4. Code organization

Put ImageNet-R experiment-specific code under its own package:

```
src/apm/continual/vision/imagenetr/
```

Recommended layout:

```
src/apm/continual/vision/
    __init__.py

    imagenetr/
        __init__.py
        protocol.py
        data.py
        artifacts.py
        manifests.py

        model.py
        lora.py
        heads.py
        leaf_training.py

        bank.py
        lineage.py

        merging/
            __init__.py
            common.py
            exact_sum.py
            svd.py
            core_space.py
            tsv.py
            output_drift.py
            ctm.py

        proxy_memory.py
        repair.py

        routing.py
        calibration.py

        baselines.py
        evaluation.py
        metrics.py
        diagnostics.py

        scheduler.py
        checkpoints.py
        reporting.py
        cli.py
```

Tests:

```
tests/continual/vision/imagenetr/
```

Configs:

```
configs/vision/imagenetr/
```

Run scripts:

```
scripts/vision/imagenetr/
```

Generic low-rank merge utilities that are clearly reusable by TRACE may live in a generic shared module, but experiment orchestration belongs under `continual/vision/imagenetr`.

---

## 5. Backend and dependency choice

Use:

- PyTorch;
- timm;
- torchvision;
- safetensors;
- numpy/scipy as needed;
- pandas/pyarrow for diagnostics;
- matplotlib for report plots.

Do not port this benchmark into JAX merely for consistency with other VAMP code. Reproducing a standard ViT/ImageNet-R protocol with minimal implementation risk is more important.

Pin all material dependency versions in the run manifest.

---

## 6. Dataset preparation

Use the same ImageFolder layout expected by the released E²-LoRA implementation:

```
$CIL_DATA_ROOT/
    imagenet-r/
        train/
            <class directories>
        test/
            <class directories>
```

Create a permanent dataset manifest containing:

- canonical file paths or stable image IDs;
- original ImageNet-R class label;
- remapped class index after the seeded class permutation;
- task index 0..49;
- train/test split;
- image hash if practical.

Freeze this manifest before any model training.

### Class order

Use the same seeded PyCIL-style permutation as the E²-LoRA config:

```
seed = 1993
shuffle = true
init_cls = 4
increment = 4
```

Unit test that:

- there are 50 tasks;
- every task contains exactly 4 classes;
- all 200 classes appear exactly once;
- train and test membership never overlaps.

---

## 7. Image transforms

Match the released ImageNet-R preprocessing unless a separate explicitly named ablation is run.

Training:

```
RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3/4, 4/3))
RandomHorizontalFlip(p=0.5)
ToTensor()
```

Test:

```
Resize(256, bicubic)
CenterCrop(224)
ToTensor()
```

Do not silently add ImageNet normalization if the pinned protocol implementation being reproduced does not use it. If normalization is deliberately changed, create a different protocol name and do not compare the resulting numbers as exact reproductions.

---

## 8. Backbone

Primary backbone:

```
vit_base_patch16_224_in21k
```

Use the same pretrained checkpoint family as the reference implementation and record the exact timm model/checkpoint revision/hash.

For all internal VAMP comparisons:

- pretrained backbone weights are frozen;
- trainable representation parameters are LoRA only;
- classifier parameters are handled separately as described below.

Use BF16 autocast where numerically stable. Low-rank decompositions and spectral diagnostics should use FP32 or FP64 where appropriate.

---

## 9. Controlled VAMP LoRA parameterization

Use a conventional fixed-rank LoRA rather than the dynamically reallocated E²-LoRA mechanism.

Primary configuration:

```yaml
lora:
  rank: 16
  alpha: 16
  scale: 1.0
  dropout: 0.0
  targets:
    - attention.qkv
    - mlp.fc1
```

The target locations intentionally mirror the adaptation sites used in the released E²-LoRA vision implementation: attention QKV and the first MLP projection.

Implement LoRA factors explicitly enough that the following operations are easy and exact:

- export each adapted matrix as low-rank factors;
- form an exact weighted sum of two LoRAs without materializing a full dense matrix;
- compute compact QR/SVD compression;
- apply Core-Space alignment;
- perform output-drift projection;
- save/load the resulting fixed-rank parent.

A zero-effect LoRA must reproduce frozen-backbone logits within numerical tolerance.

---

## 10. Classifier design

The classifier must not become an uncontrolled source of continual interference.

Use one logical 200-class namespace. A leaf task owns only the four classifier rows for its four classes. A node representing several tasks owns the union of their classifier rows.

### Leaf

For leaf task `t`:

- instantiate/train four classifier rows;
- train cross entropy only over those four classes;
- save those rows with the leaf artifact.

### Parameter-only merge

When children represent disjoint class sets, **union classifier rows exactly**. Do not average rows for different classes.

### Replay-retrained parent

For a union-retrained parent, initialize the unioned child rows and permit all represented rows to train on the parent union data.

### Repair

During repair, permit the represented classifier rows to update alongside the parent LoRA.

### Score comparability

Independent nodes may have poorly calibrated logit scales. That is treated as an addressing/calibration problem, not hidden inside the merge algorithm. Implement explicit calibration modes later in the spec.

Keep raw and calibrated results separately.

---

## 11. Leaf training

For every task `t = 0..49`:

1. start from the exact same frozen pretrained ViT;
2. create a fresh zero-effect rank-16 LoRA;
3. create/train only that task's four classifier rows;
4. train only on images from the current task;
5. save the leaf adapter, head rows, optimizer/training manifest, and source IDs;
6. never warm-start a leaf from another leaf.

Thus each leaf is base-relative:

\[
\theta_t = \theta_0 + \Delta_t.
\]

### Starting optimizer

```yaml
leaf_training:
  epochs: 5
  batch_size: 64
  optimizer: SGD
  momentum: 0.9
  weight_decay: 5.0e-4
  lora_lr: 5.0e-4
  head_lr: 1.0e-2
```

These learning-rate magnitudes are chosen to stay near the released E²-LoRA training scale while keeping the pretrained backbone frozen.

Do not perform extensive leaf hyperparameter tuning before the first complete end-to-end run. If isolated leaf accuracy is obviously poor, tune leaf training first because all later conclusions depend on healthy leaves.

---

## 12. Artifact model

Every node, leaf or parent, is immutable and contains at least:

```text
node_id
content_hash
level
first_task
last_task
represented_task_ids
represented_class_ids
represented_train_image_count

parent_a
parent_b
consolidation_method
consolidation_config_hash
repair_config_hash

lora_path
classifier_rows_path

proxy_image_ids
repair_image_ids

creation_timestamp
git_commit
software_manifest_hash
```

Every saved adapter must be hashable and reproducible from its recorded inputs.

---

## 13. Log-t bank policy

Use the existing VAMP policy:

> Keep at most two live nodes at each level. When a third node appears at a level, consolidate the two oldest equal-level nodes and carry the parent upward.

Pseudocode:

```python
def insert(bank, node, level=0):
    bank[level].append(node)
    while len(bank[level]) > 2:
        a = bank[level].pop_oldest()
        b = bank[level].pop_oldest()
        parent = consolidate(a, b)
        retire_from_live_bank(a, b)
        level += 1
        bank[level].append(parent)
```

Children are removed only from the **live bank**. They remain in the experimental artifact archive forever.

---

## 14. Expected final topology

After 50 arrivals, the live bank must be exactly:

```text
L0: [49]       [50]
L1: [45-46]    [47-48]
L2: [41-44]
L3: [33-40]
L4: [1-16]     [17-32]
```

Using one-based task numbering.

There must be exactly:

- 50 immutable leaves;
- 42 historical merge events;
- 8 final live nodes.

Including the frozen base gives at most 9 candidate parameterizations at final exhaustive addressing time.

Unit test this exact topology.

---

## 15. Consolidation condition C0: no consolidation

Name:

```
leaf_bank_50
```

Keep all 50 independently trained leaves active.

Evaluate with:

- task-free exhaustive class prediction;
- task-aware true-node oracle.

This is not a scalable final method. It is a key diagnostic upper reference for determining whether the problem is:

- weak leaves;
- addressing/calibration;
- or actual consolidation loss.

---

## 16. Consolidation condition C1: full union retraining

Name:

```
logt_retrain_union_r16
```

This is the gold-standard implementation of the original log-t temporal-consolidation idea.

For every merge `A + B -> P`:

1. create a fresh rank-16 parent LoRA from the frozen base;
2. initialize its classifier with the exact union of child rows;
3. train on the full union of all original training images represented by A and B;
4. train for 5 epochs using the same basic optimizer scale as leaves;
5. save P permanently.

Do not initialize the parent LoRA from a child or merged LoRA; this condition is intended to answer how well a fresh rank-16 model can learn the union at every temporal scale.

### Cost

For 50 equal-sized task units, the 42 merge parents collectively cover 164 task-units of source data. Therefore full parent retraining adds approximately:

\[
164/50 = 3.28
\]

times the total leaf training image presentations.

Leaf training plus all parent retraining therefore costs about:

\[
4.28\times
\]

the one-pass leaf-training budget.

This is intentionally more expensive. It is the clean control that separates topology/rank limitations from cheap-merge limitations.

---

## 17. Consolidation condition C2: weighted low-rank sum + truncated SVD

Name:

```
logt_svd_r16
```

For children with rank-16 updates:

\[
\Delta_A = B_A A_A,
\qquad
\Delta_B = B_B A_B,
\]

form a weighted combined update:

\[
\Delta_{raw} = \lambda (w_A\Delta_A + w_B\Delta_B).
\]

Default weights are proportional to represented source-image count. For equal-level nodes with equal data mass this is normally:

\[
w_A=w_B=1/2.
\]

Default overall scale:

\[
\lambda=1.
\]

Represent the exact combined update by concatenating factors so its rank is at most 32.

Then compute the best rank-16 Frobenius approximation without materializing the full dense weight matrix:

1. QR factorization of the stacked left factor;
2. QR factorization of the transpose of the stacked right factor;
3. SVD of the resulting small 32x32 core;
4. retain the top 16 singular directions;
5. convert back into rank-16 LoRA factors.

Do decomposition work in FP32.

Never average `A` factors and `B` factors separately.

Record for every adapted matrix:

\[
\rho_{16}=
\frac{\sum_{i=1}^{16}\sigma_i^2}
     {\sum_i\sigma_i^2}.
\]

Also record relative reconstruction error and child-update cosine similarity.

---

## 18. Consolidation condition C3: Core Space + TSV

Name:

```
logt_core_tsv_r16
```

Implement Core Space alignment for the two child LoRAs, then merge the aligned small core matrices with Task Singular Vectors (TSV), then return the result to a fixed rank-16 LoRA.

High-level algorithm for each adapted matrix:

1. derive a shared left basis from the two child `B` factors;
2. derive a shared right basis from the two child `A` factors;
3. express each child update in the shared compact coordinate system;
4. apply TSV to the two compact aligned updates;
5. SVD/compress the merged compact update if needed;
6. emit a conventional rank-16 LoRA parent.

Do not recursively increase rank.

Cache:

- shared bases;
- aligned child core matrices;
- TSV intermediate state;
- singular spectra;
- pre-rank-truncation merged core.

Support a later merge-scale sweep without retraining or re-forwarding the backbone.

---

## 19. Consolidation condition C4: output-drift-preserving merge

Name:

```
logt_drift_r16
```

This is the most vision-specific merge condition and should be treated as a first-class primary method.

The motivation is the empirical observation underlying E²-LoRA: the feature/output drift caused by a task update can be much more spectrally concentrated than the raw parameter update itself.

For each adapted matrix, let the provisional child-combined update be:

\[
\Delta = \lambda(w_A\Delta_A+w_B\Delta_B).
\]

Collect representative input activations `X` to that adapted linear transformation using the node proxy images.

Compute output drift:

\[
Y = X\Delta^\top
\]

or the equivalent orientation used by the implementation.

Compute:

\[
Y = U\Sigma V^\top.
\]

Keep the leading 16 output directions `U_16`.

Project the combined update onto those directions. In a consistent column-vector convention:

\[
\Delta_{16}=U_{16}U_{16}^\top\Delta.
\]

Factor the projected update directly into rank-16 LoRA form.

Do not materialize model-wide dense parameter deltas.

Record:

- full output-drift singular spectrum;
- rank-16 retained output energy;
- parameter-space retained energy;
- validation/proxy loss before and after projection.

This method should let us test whether preserving function-space change is more useful than preserving weight-space change.

---

## 20. Proxy memory for merge diagnostics and output drift

Every node carries a deterministic fixed-size proxy set.

Primary setting:

```yaml
proxy_images_per_node: 16
```

Selection rule:

- assign every training image a permanent deterministic hash priority;
- a leaf stores its 16 lowest-priority images;
- a parent stores the 16 lowest-priority images from the union of its children;
- during a merge, the method may access the union of the two child proxy sets, at most 32 images.

For output-drift merging, forward these images through the frozen base and cache the needed layer input activations.

Cache proxy activations by:

- node/child hashes;
- model revision;
- layer identifier;
- image IDs;
- transform/eval-preprocessing hash.

Support later proxy-size ablations:

```
8, 16, 32, 64
```

without leaf retraining.

Do not use test images as proxies.

---

## 21. Optional modern pure-merging baseline: Compress-then-Merge

Name:

```
ctm_r16
```

Implement after the three primary cheap merge families are functioning.

The goal is to compare against a modern fixed-rank adapter-merging strategy that learns/computes a shared low-rank representation before applying the merge, rather than merging at full effective rank and truncating afterward.

Do not block the first complete benchmark run on this condition.

---

## 22. Repair replay

All cheap merge methods must support an optional bounded repair stage.

Primary repair fractions:

```
0.00
0.01
0.05
```

Support later:

```
0.02
0.10
```

A repair fraction `f` means a node representing `N` source training images may use approximately:

\[
K=\lceil fN\rceil
\]

historical images for one repair update phase.

Use deterministic bottom-K hash sampling over the node's represented source IDs so the same policy produces exactly the same repair set every run.

The experimental archive may retain the full dataset, but the algorithmic-memory report must count only the samples allowed by the selected replay policy.

---

## 23. Repair optimizer

Start from the already merged rank-16 parent.

Primary repair config:

```yaml
repair:
  epochs: 1
  optimizer: SGD
  momentum: 0.9
  weight_decay: 5.0e-4
  lora_lr: 2.5e-4
  head_lr: 5.0e-3
```

During repair:

- pretrained ViT remains frozen;
- parent LoRA is trainable;
- all classifier rows represented by the parent are trainable.

Save repair as a separate derived artifact from the un-repaired merge parent so the same merge can be reused with many repair settings.

Primary repaired trees:

```
svd_r16 + 5% repair
core_tsv_r16 + 5% repair
drift_r16 + 5% repair
```

If 5% is clearly helpful, run 1% immediately because the more important result is the minimum replay needed to close the merge gap.

---

## 24. Merge scaling

Support:

\[
\Delta_P=\lambda(w_A\Delta_A+w_B\Delta_B).
\]

Initial scale sweep support:

```
0.50
0.75
1.00
```

Primary fixed result:

```
lambda = 1.0
```

Do not tune scale using the test set.

A later adaptive-scale method may choose lambda using only node proxies or a held-out validation subset derived from training data. Report adaptive selection separately from fixed-scale results.

---

## 25. Exact-rank diagnostic

For selected merges, evaluate the exact child sum before rank compression.

Name:

```
exact_sum_r32
```

For rank-16 children the exact weighted sum has rank at most 32.

Compare on the same represented classes:

- children individually;
- exact rank-<=32 sum;
- SVD rank-16 parent;
- Core+TSV rank-16 parent;
- output-drift rank-16 parent;
- fresh union-retrained rank-16 parent.

This decomposition answers whether degradation comes from:

- combining the child functions at all;
- compressing back to rank 16;
- or using a particular merge rule.

Sample at least:

- two low-level same-region merges;
- two mid-level merges;
- two high-level heterogeneous merges.

---

## 26. Task-free inference: exhaustive addressed classification

No task identity may be supplied to the headline VAMP result.

At a given stage, the candidate set consists of the current live temporal nodes. Their represented class sets are disjoint and union to all classes seen so far.

For image `x`:

1. evaluate each live LoRA node on `x`;
2. obtain logits only for classes represented by that node;
3. calibrate scores into a common scale;
4. concatenate all node-class scores;
5. predict the global maximum.

Formally:

\[
(\hat n,\hat c)
=
\arg\max_{n,\;c\in C_n} s_{n,c}(x).
\]

At task 50 this requires only 8 live adapter evaluations.

This is the primary task-free addressed-memory result.

---

## 27. Score calibration

Because nodes are independently trained, raw logits may have different scale/offset distributions.

Implement and report both raw and calibrated results.

### Calibration A: fixed normalized score

Preferred initial no-extra-memory approach:

- L2 normalize penultimate features;
- L2 normalize classifier row vectors;
- use cosine logits with one global fixed scale.

If this classifier design is adopted from the beginning, use it consistently for every internal baseline.

### Calibration B: per-node affine calibration

Maintain per-node parameters:

\[
T_n>0,\qquad b_n
\]

and transform:

\[
s'_{n,c}=s_{n,c}/T_n+b_n.
\]

Fit only from training-derived proxy/replay images belonging to current live nodes.

Count these parameters as live routing metadata.

Never fit calibration using test labels.

### Required reporting

Always include:

- raw task-free result;
- calibrated task-free result;
- true-node oracle result.

This makes it clear whether improvements come from memory quality or calibration.

---

## 28. Alternative one-pass router

Implement a cheaper router using the frozen base.

For each class/task, compute a frozen-base feature centroid from training images. Node routing metadata is the set or aggregate of descendant class centroids.

At inference:

1. run the frozen base once;
2. compare query feature against node/class centroids;
3. choose one candidate node;
4. run only that node's adapter and classify inside it.

Report:

- routing accuracy;
- final classification accuracy;
- one-base-forward + one-adapter-forward cost.

This is not required to beat exhaustive routing. It is an addressing-cost ablation.

---

## 29. Addressing oracles

### True-node oracle

Given the ground-truth class label, select the current live node whose represented class set contains that label. Then classify only within that node.

This removes routing error while preserving consolidation error.

### All-leaf true-task oracle

Keep all 50 leaves and select the exact task leaf using ground truth.

This is the approximate isolated-adaptation ceiling for the chosen leaf architecture.

Never label either oracle as task-free continual-learning performance.

---

## 30. Internal baselines

Required controlled baselines:

### B0 — frozen backbone reference

Measure the pretrained representation with an appropriately trained classifier reference. Clearly specify what training the head receives.

### B1 — sequential fixed-rank LoRA

One rank-16 LoRA and one expanding classifier, updated sequentially across all 50 tasks with no replay.

Name:

```
seq_lora_r16
```

This is the direct catastrophic-forgetting baseline.

### B2 — joint IID fixed-rank LoRA

One fresh rank-16 LoRA and a 200-class classifier trained jointly on all ImageNet-R training data for 5 epochs.

Name:

```
joint_iid_lora_r16
```

This is the central architectural upper reference.

### B3 — all immutable leaves

Name:

```
leaf_bank_50
```

Evaluate task-free and true-task oracle.

### B4 — full replay-retrained log-t hierarchy

Name:

```
logt_retrain_union_r16
```

This is the gold-standard hierarchy.

---

## 31. External baseline reproduction: E²-LoRA

Where practical, clone/pin the official E²-LoRA implementation and run its released ImageNet-R-50 configuration separately.

Record:

- repository URL;
- exact commit SHA;
- config file hash;
- dependency environment;
- observed final accuracy;
- observed incremental average accuracy;
- wall time and hardware.

Do not modify the E²-LoRA code to share VAMP internals. Treat it as an independent external baseline.

If reproduction fails or differs materially from the paper, report both the published number and the locally reproduced number with the discrepancy clearly stated.

---

## 32. HAM/GLAM relevance

HAM/GLAM is a strong mechanistic related-work comparison because it also manages task LoRAs through hierarchical/grouped merging. However its published 50-task suite is not exactly the same ImageNet-R protocol.

Do not delay the ImageNet-R run waiting for a perfect HAM reproduction.

Later secondary work can run VAMP on one of HAM/GLAM's published datasets for a direct algorithmic comparison.

---

## 33. Evaluation schedule

After every task `t = 1..50`:

1. snapshot the current live bank;
2. evaluate on all classes seen so far with no task identity;
3. record global class-incremental accuracy;
4. record per-task accuracy for every prior task;
5. record live-node count and addressing cost.

For merge-policy trees that are constructed after all leaves are available, reconstruct the exact historical tree state after each task from the deterministic lineage and evaluate those snapshots.

Do not evaluate only the final tree; incremental average accuracy requires every stage.

---

## 34. Primary metrics

### Final accuracy

\[
LastAcc=A_{50}.
\]

Primary external benchmark number.

### Incremental average accuracy

\[
IncAcc=\frac{1}{50}\sum_{t=1}^{50} A_t.
\]

### Mean forgetting

For task `i`:

\[
F_i=\max_{t\ge i} A_{i,t}-A_{i,50}.
\]

Then:

\[
F=\frac{1}{50}\sum_i F_i.
\]

### Joint-IID gap

\[
G_{IID}=LastAcc_{joint}-LastAcc_{VAMP}.
\]

This should be emphasized alongside raw benchmark accuracy.

### Routing regret

At the example or aggregate level compare task-free routing to true-node oracle.

At aggregate level:

\[
R_{route}=Acc_{oracle-node}-Acc_{task-free}.
\]

### Consolidation gap

Compare true-node oracle of a compressed tree to true-node oracle of the all-leaf bank.

### Merge approximation gap

Compare cheap-merge tree to full union-retrained log-t tree.

---

## 35. Resource metrics

Record for every condition:

- training image presentations;
- repair/replay image presentations;
- optimizer steps;
- proxy images used only for forward statistics;
- proxy forward passes;
- leaf-training wall time;
- consolidation wall time;
- repair wall time;
- evaluation wall time;
- peak VRAM;
- final live LoRA parameter count;
- archived LoRA parameter count;
- live proxy image count;
- live repair image count;
- final live node count;
- average live node count over the stream;
- average candidate forwards per query;
- final candidate forwards per query;
- frozen-router forward count where applicable.

Keep forward-only proxy use separate from gradient replay.

---

## 36. Merge diagnostics

Every merge event must produce a row containing at least:

```text
merge_id
policy_hash
level
child_a_interval
child_b_interval
parent_interval
represented_tasks
represented_classes
represented_images

merge_method
output_rank
merge_scale

child_a_update_norm
child_b_update_norm
child_update_cosine

parameter_singular_spectrum
rank16_parameter_energy
relative_parameter_reconstruction_error

output_drift_spectrum                if applicable
rank16_output_energy                 if applicable

raw_parent_proxy_accuracy
repaired_parent_proxy_accuracy
retrained_parent_proxy_accuracy      if available

proxy_image_count
repair_image_count
repair_optimizer_steps
merge_wall_seconds
repair_wall_seconds
```

For Core+TSV additionally store:

- shared basis dimensions;
- aligned core matrices or hashes;
- TSV singular values/state;
- precompression merged-core rank.

---

## 37. Required plots

At minimum generate:

1. global class-incremental accuracy vs task index;
2. final accuracy by method;
3. incremental average accuracy by method;
4. mean forgetting by method;
5. joint-IID gap by method;
6. task-free vs true-node oracle performance;
7. all-leaf vs log-t consolidation gap;
8. cheap merge vs full union-retrained tree;
9. SVD vs Core+TSV vs output-drift merge;
10. 0% vs 1% vs 5% repair;
11. merge damage vs tree level;
12. merge damage vs retained parameter energy;
13. merge damage vs retained output-drift energy;
14. live node count vs task index;
15. accuracy vs historical images revisited with gradient;
16. accuracy vs live parameter memory;
17. addressing cost vs accuracy.

The most important efficiency plot is:

```
final / incremental accuracy
versus
historical images revisited with gradient
```

because it directly shows whether cheap merging + tiny repair approaches full replay consolidation.

---

## 38. Primary first-seed experiment matrix

Use seed `1993`.

Required:

```text
frozen_reference
seq_lora_r16
joint_iid_lora_r16
leaf_bank_50

logt_retrain_union_r16

logt_svd_r16_repair000
logt_svd_r16_repair005

logt_core_tsv_r16_repair000
logt_core_tsv_r16_repair005

logt_drift_r16_repair000
logt_drift_r16_repair005

e2lora_official_reference
```

All cheap VAMP conditions must use the exact same 50 immutable leaf hashes.

Do not retrain leaves when moving from one tree policy to another.

---

## 39. Second-pass sweep

Only after the complete first matrix exists, run reusable-artifact sweeps.

### Repair fraction

```
0
0.01
0.02
0.05
0.10
```

### Merge scale

```
0.50
0.75
1.00
```

### Proxy size

```
8
16
32
64
```

### Output rank, only if rank 16 is clearly a bottleneck

```
8
16
24
32
```

Do not start broad hyperparameter sweeps before the first interpretable end-to-end result exists.

---

## 40. Runtime and hardware

Primary local target:

```
1 x RTX 4090 24GB
```

Optional remote target:

```
2 x RTX 4090
```

For two GPUs, use them as independent workers rather than data-parallelizing ViT-B/16 unless profiling proves otherwise.

Good independent work units include:

- separate leaf tasks;
- sequential baseline;
- joint baseline;
- parent retraining jobs at independent branches;
- different merge-policy trees;
- stage/method evaluation jobs;
- repair sweeps.

### Initial engineering estimate

One 4090:

- data/model/protocol smoke: 0.2–0.5 h;
- all 50 immutable leaves: 0.3–0.8 h;
- sequential + joint internal baselines: 0.3–0.8 h;
- full replay-retrained log-t tree: 0.8–2.0 h;
- cheap merge trees: 0.1–0.4 h;
- full incremental evaluation: 0.5–1.2 h;
- diagnostics/report: 0.2–0.5 h.

Expected first complete seed:

```
3–6 hours on one RTX 4090
```

Two independent 4090 workers:

```
1.5–3.5 hours
```

These are pre-run engineering estimates only. After the first 5 tasks, replace them with measured ETA from actual throughput.

---

## 41. Runtime instrumentation

Measure continuously:

- train images/sec;
- eval images/sec;
- optimizer steps/sec;
- proxy forward images/sec;
- mean merge seconds/node;
- mean repair seconds/image and seconds/node;
- GPU utilization;
- dataloader stall fraction.

After at least five leaf jobs, print a live ETA for:

- remaining leaves;
- remaining parent retraining;
- remaining evaluations;
- whole experiment.

If evaluation dominates, increase parallelism/granularity before changing scientific content.

---

## 42. Resumability and interruption

Every expensive job must be independently resumable.

Job types:

```text
train_leaf_<task>
train_seq_baseline
train_joint_baseline
train_retrain_parent_<node>
build_merge_parent_<policy>_<node>
repair_parent_<policy>_<node>
eval_<policy>_<stage>
report
```

States:

```text
PENDING
RUNNING
COMPLETE
FAILED
PAUSED
```

A completed immutable artifact is never recomputed unless explicitly forced.

Persist scheduler state after every completed job.

For leaf/parent training, checkpoint sufficiently often that interruption costs at most a small fraction of one short job.

---

## 43. Optional RunPod packaging

The code should be runnable locally first, but also package cleanly for RunPod.

Provide:

```
docker/vision/imagenetr/Dockerfile
scripts/vision/imagenetr/runpod_entrypoint.sh
```

Use a persistent network volume for:

- dataset;
- pretrained model cache;
- leaves;
- all parent artifacts;
- evaluations;
- reports;
- scheduler state.

Never rely on container-local storage for anything expensive to reproduce.

If a remote run exceeds 24 hours, use the same operational rule as the TRACE experiment:

1. stop dispatching new jobs before the limit;
2. finish/checkpoint active atomic jobs;
3. write a durable interim report;
4. persist scheduler state and all manifests;
5. notify through the configured notification hook if available;
6. terminate compute while retaining the persistent volume;
7. resume later without retraining completed leaves or parents.

No experiment conditions should be silently dropped merely to hit the wall-clock limit.

---

## 44. Fast 8-task smoke test

Before the full 50-task run, execute tasks 1–8 only.

This is for correctness, not hyperparameter optimization.

Validate:

- leaves learn substantially above chance;
- leaf save/load is exact;
- head union is exact;
- log-t bank topology is correct;
- SVD merge algebra is correct;
- Core+TSV produces finite stable outputs;
- output-drift proxy extraction works;
- repair works;
- task-free routing produces sensible predictions;
- true-node oracle is substantially better than chance;
- artifact reuse works.

Once these basic properties pass, run all 50 tasks without extensive smoke-subset tuning.

---

## 45. Failure diagnosis order

When a result is poor, diagnose in this order.

### 1. Are isolated leaves good?

If no: leaf training/model/head problem.

### 2. Is all-leaf true-task oracle good?

If no: insufficient leaf/head capacity or bad leaf training.

### 3. Is all-leaf task-free addressing good?

If oracle is good but task-free is bad: routing/logit-calibration problem.

### 4. Is full replay-retrained log-t true-node oracle good?

If leaf bank is good but this is bad: fixed rank or hierarchical consolidation is intrinsically losing important information.

### 5. Does cheap merge true-node oracle fall below full retraining?

If yes: merge approximation problem.

### 6. Does 1–5% repair close the gap?

If yes: cheap merging captures most structure and only sparse functional correction is needed.

### 7. Is task-free still below true-node oracle after good consolidation?

If yes: addressing remains the dominant issue.

The generated report must use this causal structure rather than presenting only aggregate accuracies.

---

## 46. Success thresholds

Use literature numbers only as external context, not as automatic correctness thresholds.

Approximate current target:

```
E²-LoRA LastAcc: 78.58
E²-LoRA IncAcc:  83.96
Joint LastAcc:   82.76
```

Interpretation for the first seed:

- `<70`: likely major implementation/routing weakness;
- `70–76`: functioning but not headline competitive;
- `76–78.5`: strong long-sequence CL result;
- `>78.58`: potential current-SOTA territory, subject to protocol verification;
- `80–82`: extremely strong, materially closing the sequential/joint gap;
- `~82+`: approaching joint-training performance.

The scientifically stronger result is not merely a small score win. It is a high score combined with low replay, logarithmic live memory, and a small joint-IID gap.

---

## 47. The result we are really looking for

A particularly compelling outcome would look roughly like:

```text
joint IID rank-16                    82.x
full union-retrained log-t           81.x
output-drift merge-only              79–80.x
output-drift + 1% repair             80–81.x
Core+TSV + 1% repair                 similar or better
external E²-LoRA                     78.58 published target
```

This would support all of the important claims at once:

1. the logarithmic addressed-memory topology nearly preserves the joint solution;
2. cheap parameter consolidation retains most of that performance;
3. only tiny replay is needed to repair residual error;
4. the live memory remains `O(log T)`;
5. exhaustive addressing remains practical because there are only 8 live nodes after 50 tasks.

Do not assume this result. The experiment is designed to determine whether it is true.

---

## 48. Required artifact reuse demonstration

After the first complete merge tree exists:

1. change repair fraction from `0.05` to `0.01`;
2. rebuild one full cheap-merge tree;
3. reevaluate it;
4. verify that no leaf training job executed.

The report must include:

```text
leaf hashes unchanged: true
leaf optimizer steps during rerun: 0
new gradient work: repair only
```

Also perform one merge-method rerun, e.g. SVD -> output-drift, with zero leaf optimizer steps.

---

## 49. Artifact store

Recommended persistent structure:

```text
artifacts/imagenetr50/
    protocol/
        dataset_manifest.json
        class_order.json
        model_manifest.json
        software_manifest.json

    leaves/
        task_000/
        ...
        task_049/

    merge_cache/
        <childA>__<childB>/
            exact_factors/
            svd/
            core_space/
            output_drift/
            proxy_activations/

    trees/
        <policy_hash>/
            nodes/
            snapshots/
            scheduler_state/

    baselines/
        sequential/
        joint_iid/
        e2lora/

    evaluations/
        <policy_hash>/

    diagnostics/
    reports/
    logs/
```

The experimental archive may retain everything. Deployment-memory accounting must count only live algorithmic state.

---

## 50. Experimental archive vs live algorithm memory

Keep these concepts separate in every report.

### Experimental archive

May include:

- all 50 leaves;
- every retired parent/child;
- full source dataset;
- proxy activation caches;
- all evaluation outputs;
- all alternative merge trees.

This exists to make scientific reruns cheap.

### Live VAMP state

For final 50-task deployment under a selected policy, count only:

- 8 active LoRA nodes;
- represented classifier rows/metadata;
- allowed routing calibration state;
- allowed proxy/replay memory for that policy.

Do not count archived retired nodes as live memory when reporting algorithmic asymptotics.

---

## 51. Correctness tests

Implement at least the following tests before accepting benchmark results.

1. Seed 1993 produces exactly 50 tasks of 4 classes.
2. All 200 classes occur exactly once.
3. Training/test splits do not overlap.
4. All leaves start from identical frozen backbone weights.
5. Zero LoRA reproduces the frozen backbone.
6. Only LoRA and active classifier rows receive gradients during leaf training.
7. Leaf save/load reproduces logits.
8. Classifier-row union is exact for disjoint classes.
9. Exact stacked rank-<=32 LoRA sum matches dense matrix addition on test matrices.
10. Compact rank-16 SVD matches dense truncated SVD on test matrices.
11. Core-Space reconstruction matches its dense reference on small matrices.
12. Core+TSV output rank is <=16 after the final compression step.
13. Output-drift projection output rank is <=16.
14. Output-drift compact factors reproduce the projected dense update on test matrices.
15. Changing merge policy does not change any leaf hash.
16. Changing repair fraction does not change any leaf hash.
17. Changing router does not change any adapter hash.
18. Fifty tasks produce exactly 42 merge events.
19. The final bank is exactly the 8-node topology specified above.
20. Active node task intervals are disjoint.
21. Active node class sets are disjoint.
22. Active node class sets union to all classes seen so far.
23. Proxy selection is deterministic.
24. Repair selection is deterministic.
25. No test image is used for training, repair, proxy selection, calibration, or hyperparameter selection.
26. Headline inference receives no task ID.
27. True-node oracle is clearly marked diagnostic.
28. Rerunning a merge method performs zero leaf optimizer steps.
29. Resuming a partially completed run does not duplicate completed jobs.
30. Deployment-memory accounting excludes archived retired artifacts.

---

## 52. Required output artifacts

Every complete run should emit:

```text
config_resolved.yaml
protocol_manifest.json
model_manifest.json
software_manifest.json
leaf_manifest.jsonl
node_manifest.jsonl
job_manifest.jsonl

stage_accuracy.csv
task_accuracy_matrix.csv
merge_diagnostics.parquet
routing_diagnostics.parquet
resource_metrics.json
summary.json

REPORT.md
REPORT.html
```

`REPORT.md` must be understandable without reading training logs.

---

## 53. Required report structure

`REPORT.md` should contain:

1. exact protocol and software revisions;
2. leaf quality summary;
3. baseline reproduction summary;
4. primary final/average accuracy table;
5. joint-IID gap;
6. task-free vs oracle routing gap;
7. full-retrained-tree vs cheap-merge-tree gap;
8. SVD vs Core+TSV vs output-drift comparison;
9. repair fraction comparison;
10. merge diagnostics by level;
11. compute/memory/addressing accounting;
12. artifact reuse demonstration;
13. failure analysis following the diagnostic hierarchy;
14. conclusions about which assumption succeeded or failed;
15. prioritized next experiments.

Do not bury negative results. If the hierarchy fails even under full retraining, say so clearly. If the hierarchy works but routing fails, say that instead. If routing works but cheap merging fails, isolate that conclusion.

---

## 54. Resolved initial config

Use this as the starting point, adjusting only where implementation details require an explicitly documented equivalent.

```yaml
experiment:
  name: imagenetr50_logt_vamp
  seed: 1993

paths:
  artifact_root: artifacts/imagenetr50

dataset:
  name: imagenet-r
  classes: 200
  tasks: 50
  classes_per_task: 4
  class_order: pycil_seeded_permutation
  class_order_seed: 1993
  input_size: 224

backbone:
  model: vit_base_patch16_224_in21k
  frozen: true
  precision: bfloat16

lora:
  rank: 16
  alpha: 16
  dropout: 0.0
  targets:
    - attention.qkv
    - mlp.fc1

leaf_training:
  epochs: 5
  batch_size: 64
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5.0e-4
  lora_lr: 5.0e-4
  head_lr: 1.0e-2

bank:
  max_live_nodes_per_level: 2
  overflow_policy: merge_two_oldest
  adapters_base_relative: true

merge:
  output_rank: 16
  weighting: source_image_count
  scale: 1.0
  decomposition_dtype: float32

proxy:
  images_per_node: 16
  selection: deterministic_bottom_k_hash
  cache_layer_inputs: true

repair:
  fractions:
    - 0.0
    - 0.01
    - 0.05
  primary_fraction: 0.05
  epochs: 1
  optimizer: sgd
  momentum: 0.9
  weight_decay: 5.0e-4
  lora_lr: 2.5e-4
  head_lr: 5.0e-3

routing:
  primary: exhaustive_live_nodes
  score_modes:
    - raw
    - normalized
    - affine_calibrated
  diagnostic_oracles:
    - true_node
    - all_leaf_true_task
  alternative:
    - frozen_feature_router

evaluation:
  after_every_task: true
  report_last_accuracy: true
  report_incremental_average_accuracy: true
  report_forgetting: true
  report_joint_gap: true
  report_routing_gap: true

runtime:
  preferred_gpu: RTX_4090
  allow_two_independent_gpu_workers: true
  checkpoint_jobs: true
  resume_completed_artifacts: true
```

---

## 55. Execution order

Run in this order unless a dependency forces a minor reordering.

### Phase 0 — protocol correctness

- dataset manifest;
- exact seeded class order;
- transforms;
- pretrained checkpoint pin;
- zero-LoRA tests;
- head tests;
- compact SVD tests;
- bank-topology simulation.

### Phase 1 — 8-task smoke

- train 8 leaves;
- evaluate leaf oracle/task-free;
- build at least one SVD parent;
- build at least one Core+TSV parent;
- build at least one output-drift parent;
- run one repair;
- verify artifact reuse.

### Phase 2 — immutable leaves

Train and save all 50 leaves.

### Phase 3 — controlled internal baselines

- sequential rank-16 LoRA;
- joint IID rank-16 LoRA;
- all-leaf bank evaluation.

### Phase 4 — gold-standard log-t tree

Build the complete union-retrained rank-16 tree.

### Phase 5 — cheap trees

Build from the same immutable leaves:

- SVD;
- Core+TSV;
- output drift.

### Phase 6 — primary repair

Build 5%-repair variants from cached un-repaired merge parents.

If clearly beneficial, add 1% repair immediately.

### Phase 7 — external E²-LoRA reference

Run separately if not already reproduced.

### Phase 8 — reports and targeted sweeps

Only after the complete primary matrix exists.

---

## 56. Definition of done

The first-seed experiment is complete when all of the following exist:

1. exact ImageNet-R-50 protocol manifest using seed 1993;
2. 50 immutable independently trained rank-16 LoRA leaves;
3. controlled sequential rank-16 baseline;
4. controlled joint-IID rank-16 baseline;
5. all-leaf task-free and oracle evaluation;
6. complete full-union-retrained log-t tree;
7. complete SVD merge tree;
8. complete Core+TSV merge tree;
9. complete output-drift merge tree;
10. 5%-repair variants for all three cheap merge trees;
11. 1%-repair variants if 5% materially helps;
12. task-free, calibrated, and true-node oracle results;
13. per-task/stage accuracy matrices;
14. merge diagnostics by level;
15. compute, replay, memory, and addressing accounting;
16. external E²-LoRA published/reproduced reference clearly separated from internal baselines;
17. artifact-reuse test proving a full alternate merge/repair run with zero leaf optimizer steps;
18. a report that explicitly answers:

\[
\boxed{\text{Does log-}t\text{ consolidation itself work?}}
\]

\[
\boxed{\text{How close can it get to joint IID?}}
\]

\[
\boxed{\text{Can cheap LoRA merging replace union retraining?}}
\]

\[
\boxed{\text{Does output-drift compression outperform weight-space compression?}}
\]

\[
\boxed{\text{How much replay is actually required?}}
\]

\[
\boxed{\text{How much residual error is addressing rather than storage?}}
\]

\[
\boxed{\text{Can merge policy experiments be rerun without retraining leaves?}}
\]

That is the complete first experiment. Do not add speculative routing machinery, semantic indexing, or unrelated continual-learning heuristics until these core questions have clean empirical answers.

---

## 57. Reference sources to pin during implementation

Codex should record/pin, but not blindly copy from, the following references:

- **E²-LoRA** official repository: `kiddo127/E2-LoRA`, especially the class-incremental ImageNet-R-50 config, data loader, and adaptation implementation.
- E²-LoRA paper, for reported ImageNet-R-50 and joint-training results and the output-drift motivation.
- Core Space Merging paper / official implementation for the Core-Space + TSV merge algorithm.
- HAM/GLAM paper for related hierarchical LoRA consolidation comparisons.
- Compress-then-Merge paper for the optional fixed-rank pure-merging baseline.

The benchmark protocol, internal VAMP architecture, and causal comparisons in this document remain the controlling specification.
