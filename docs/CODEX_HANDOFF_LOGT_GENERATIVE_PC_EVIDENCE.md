# Codex handoff: Minimal Generative-PC Evidence Experiment for LogT-VAMP

## 1. Objective

Test the following claim directly:

> Independently trained, normalized generative predictive-coding models can produce input-evidence scores that remain meaningfully comparable across active LogT nodes, including a small recent leaf and a much larger historical node whose training distribution may be disjoint, partially overlapping, or effectively identical.

This experiment should answer three separate questions:

1. Does completing the PC model as an actual normalized generative model remove arbitrary cross-node energy offsets?
2. Is settled maximum-a-posteriori PC energy sufficient, or is a posterior-volume correction needed?
3. Does the resulting score route well enough to recover most of the task performance available under an oracle node choice?

Base the implementation on `dmacd/vamp` commit `55b4276da8332d9c0004b5c4ce0397cfe090764e`, which introduced the completed NCE/TRE experiment. Do not modify or overwrite that experiment or its artifacts. fileciteturn35file0L2-L2

The current NCE/TRE workflow already supplies the right experimental skeleton: authenticated data, a true binary-counter LogT frontier, phase gates, independent replicas, held-out temporal-source routing, oracle routing regret, durable checkpoints, and exact work accounting. Its static phase correctly stopped before consolidation and online evaluation when no TRE schedule passed. Reuse that discipline, not the NCE/TRE model. fileciteturn25file0L2-L2 fileciteturn31file0L2-L2

The experiment instantiates the predictive-coding version of addressed inference already contemplated in VAMP, but keeps addressing evaluation distinct from task-model retention. fileciteturn1file0

---

## 2. Central design decision

Use **one generative PC model per active temporal node**. Do not retain the frozen CNN plus top-two adapter as the node's underlying inference model. The selected node's prediction must come from that node's PC representation and a node-local classifier head.

Do retain the current exhaustive evidence scan:

\[
\hat v(x)=\arg\max_{v\in A_t} e_v(x),
\]

where:

- \(x\) is the query image;
- \(A_t\) is the set of active LogT nodes after \(t\) stream blocks;
- \(e_v(x)\) is the evidence score produced by node \(v\);
- \(\hat v(x)\) is the selected node.

The active nodes remain disjoint contiguous temporal intervals, with at most one node at each level. The current `LogTState` and `insert_block` implementation already enforce exactly this and bound the frontier by \(\lceil\log_2(t+1)\rceil\). Reuse them unchanged. fileciteturn24file0L2-L2

Do **not** introduce a Hopfield router, learned router, PCA index, or candidate-pruning mechanism. Exhaustive scoring of all active nodes is already logarithmic in stream length and isolates the question we care about. fileciteturn1file1

---

## 3. FabricPC recommendation

Use FabricPC, but create a new probabilistic backend rather than routing on the existing `per_example_observed_energy` output.

The existing VAMP FabricPC backend is valuable because it already provides:

- graph construction;
- iterative latent inference;
- PC weight training;
- per-example reconstruction;
- node energy extraction;
- JAX batching and compilation.

However, its current graph has an unregularized root latent, and `_state_energy` sums energies only for nodes with incoming edges. Its exported observed energy is therefore a settled prediction-error sum, not a complete negative log joint density. fileciteturn15file0L2-L2 fileciteturn16file0L2-L2 fileciteturn17file0L2-L2

FabricPC's `GaussianEnergy` likewise contains only

\[
\frac{\tau}{2}\lVert z-\mu\rVert^2,
\]

where \(\tau\) is precision, meaning inverse variance. It deliberately omits the Gaussian normalization term. That omission is harmless for ordinary latent inference with globally fixed precision, but the exact term must exist in the evidence implementation and analytic tests. fileciteturn26file0L2-L2

Use the stable standard `InferenceSGD` path for this first experiment. Do not use the experimental ePC solver yet. The ePC branch optimizes a mixture of internal prediction-error variables and source latents through a global differentiable program; adding it now would confound evidence semantics with a second inference algorithm. It should become a later solver ablation if the basic result is positive. fileciteturn27file0L2-L2

Pin FabricPC to release `0.4.0`, corresponding to main commit `138941ef5763ab202c7df07879d3f21678e6cc0a`. The present VAMP `pc` extra points to mutable Git `main`; do not use that mutable dependency for a sealed experiment. FabricPC 0.4.0 also requires Python 3.11 or newer, so run this experiment in a Python 3.11 environment. fileciteturn18file0L2-L2 fileciteturn36file0L2-L2

---

## 4. The node model

Use the smallest FabricPC graph that is still a nonlinear generative neural model:

\[
0 \longrightarrow z \longrightarrow h \longrightarrow x.
\]

Here:

- \(z\) is a 32-dimensional top latent state;
- \(h\) is a 128-dimensional hidden latent state;
- \(x\) is the 784-dimensional flattened image;
- the first arrow defines a standard Gaussian prior on \(z\);
- the second arrow uses a trainable linear map followed by `tanh`;
- the third arrow uses a trainable linear map followed by a sigmoid image mean.

The model is

\[
p_v(x,h,z)
=
p(z)\,p_v(h\mid z)\,p_v(x\mid h).
\]

Use:

\[
p(z)=\mathcal N(0,I),
\]

\[
p_v(h\mid z)
=
\mathcal N\!\left(
h;\tanh(W_{h,v}z+b_{h,v}),
\tau_h^{-1}I
\right),
\]

\[
p_v(x\mid h)
=
\mathcal N\!\left(
x;\operatorname{sigmoid}(W_{x,v}h+b_{x,v}),
\tau_x^{-1}I
\right).
\]

In these expressions:

- \(W_{h,v}\), \(b_{h,v}\), \(W_{x,v}\), and \(b_{x,v}\) are trainable parameters belonging to node \(v\);
- \(I\) is the identity matrix of the required size;
- \(\tau_h\) is the fixed hidden-state precision;
- \(\tau_x\) is the fixed image precision.

Every node must have exactly the same architecture, latent dimensions, and fixed precisions. Do not learn a separate observation variance or hidden variance per node in this experiment.

The fixed image standard deviation

\[
\sigma_x=\frac{1}{\sqrt{\tau_x}}
\]

defines the neighborhood scale at which the model measures probability around the learned manifold. This replaces the explicit small corruption used in NCE/TRE.

### Implementing the prior in FabricPC

Create:

1. a zero-valued `IdentityNode` named `prior`, always clamped to zero;
2. an `IdentityNode` named `latent`, with Gaussian precision one;
3. an edge from `prior` to `latent`.

The latent node's prediction is then zero and its energy is the standard-normal prior energy. FabricPC's identity node has no trainable parameters and can carry its own Gaussian energy, so no FabricPC core modification should be necessary. fileciteturn37file0L2-L2

### Node-local task head

Attach a node-local linear softmax classifier to the settled hidden state \(h^*(x)\):

\[
p_v(y\mid h^*)=\operatorname{softmax}(W_{y,v}h^*+b_{y,v}),
\]

where \(y\) is the digit label.

Train this head with cross-entropy using a stopped-gradient copy of the settled hidden state. Do not propagate classifier gradients into the generative PC model in this experiment. This keeps the image density target exactly \(p_v(x)\) and guarantees that labels cannot alter evidence.

The frozen CNN, frozen trunk features, top-two adapter, context identifier, and transformed label must not enter PC evidence inference. Context identifiers remain available only for constructing the controlled stream and calculating evaluation provenance.

---

## 5. The three evidence scores

For one node \(v\), let \(u\) denote the concatenation of all free PC states:

\[
u=(z,h).
\]

Let \(J_v(x,u)\) be the complete negative log joint density

\[
J_v(x,u)=-\log p_v(x,h,z).
\]

For the model above,

\[
\begin{aligned}
J_v(x,z,h)
={}&
\frac12\lVert z\rVert^2
+\frac{32}{2}\log(2\pi)
\\
&+
\frac{\tau_h}{2}
\left\lVert
h-\tanh(W_{h,v}z+b_{h,v})
\right\rVert^2
+\frac{128}{2}\log\left(\frac{2\pi}{\tau_h}\right)
\\
&+
\frac{\tau_x}{2}
\left\lVert
x-\operatorname{sigmoid}(W_{x,v}h+b_{x,v})
\right\rVert^2
+\frac{784}{2}\log\left(\frac{2\pi}{\tau_x}\right).
\end{aligned}
\]

PC inference produces a settled state

\[
u_v^*(x)=\arg\min_u J_v(x,u).
\]

Implement and compare these scores.

### A. Legacy residual score

\[
e_v^{\mathrm{res}}(x)
=
-\left[
E_{h,v}(x)+E_{x,v}(x)
\right].
\]

This is essentially the familiar sum of settled prediction errors. It excludes the latent prior and posterior-volume correction. It is the negative control closest to current FabricPC energy routing.

### B. Complete MAP score

\[
e_v^{\mathrm{MAP}}(x)
=
-J_v\!\left(x,u_v^*(x)\right).
\]

This score has a fixed probabilistic gauge because it is a complete normalized joint density. It is still not marginal evidence because it retains only the best latent explanation.

### C. Laplace evidence score

Let

\[
H_v(x)
=
\nabla_u^2 J_v(x,u)
\big|_{u=u_v^*(x)}
\]

be the Hessian of the complete negative log joint with respect to the 160 free latent values. The Laplace score is

\[
e_v^{\mathrm{Lap}}(x)
=
-J_v(x,u_v^*)
+\frac{160}{2}\log(2\pi)
-\frac12\log\det H_v(x).
\]

This is the primary score under test.

Compute the full 160-by-160 Hessian with `jax.hessian`. This is intentionally small enough that the first experiment does not need diagonal, block-diagonal, or stochastic log-determinant approximations.

Use one globally fixed Hessian regularizer:

\[
H_v^{\mathrm{reg}}(x)=H_v(x)+\lambda I,
\]

where \(\lambda\) is selected in the analytic preflight and then frozen. Do not adapt \(\lambda\) by node or level. Record the smallest raw Hessian eigenvalue and the fraction of examples that would fail Cholesky factorization without regularization.

### D. Importance-weighted audit

For a small fixed subset only, use the Laplace Gaussian as an importance-sampling proposal and estimate marginal evidence with 32 or 64 samples. Record:

- the difference from the Laplace score;
- effective sample size;
- whether the difference depends on node level;
- whether different deterministic inference starts find different modes.

Do not use this estimator as the primary router in the first experiment.

### No score calibration

The following are prohibited:

- subtracting a node's mean training energy;
- dividing by a node's score standard deviation;
- fitting per-node or per-level biases;
- temperature scaling nodes separately;
- adding \(\log N_v\), where \(N_v\) is the node's training-set size;
- choosing a prior probability proportional to temporal interval length.

The target is the conditional density represented by each node, compared under equal node priors. Any of the above would evade rather than test commensurability.

---

## 6. Controlled LogT stream

Reuse the authenticated rotated/label-shifted MNIST transformations and raw uint8 data from the NCE/TRE experiment. That code already separates raw images, labels, context identifiers, and temporal provenance, and constructs held-out samples from the node's context mixture. fileciteturn33file0L2-L2

For the first useful run, reduce the snapshot from 63 blocks to **31 blocks** and reduce the block size from 500 to **250 examples**. This preserves a genuine five-node LogT frontier while reducing PC training and Hessian evaluation substantially.

After 31 blocks, the live temporal frontier is:

| Level | Temporal interval | Blocks | Examples |
|---|---:|---:|---:|
| 4 | 0–15 | 16 | 4,000 |
| 3 | 16–23 | 8 | 2,000 |
| 2 | 24–27 | 4 | 1,000 |
| 1 | 28–29 | 2 | 500 |
| 0 | 30 | 1 | 250 |

Use the existing five transformed MNIST contexts. Call them \(C_0,C_1,C_2,C_3,C_4\). These names mean only the five fixed rotation-plus-label-shift conditions; they are never model inputs.

Run three stream conditions.

### Condition 1: novel leaf

| Active interval | Assigned context blocks |
|---|---|
| Level 4 | \(16\times C_0\) |
| Level 3 | \(8\times C_1\) |
| Level 2 | \(4\times C_2\) |
| Level 1 | \(2\times C_3\) |
| Level 0 leaf | \(1\times C_4\) |

This tests whether the recent leaf gets high evidence for a genuinely new regime and historical nodes do not.

### Condition 2: recurrent leaf

| Active interval | Assigned context blocks |
|---|---|
| Level 4 | \(14\times C_0+2\times C_4\) |
| Level 3 | \(8\times C_1\) |
| Level 2 | \(4\times C_2\) |
| Level 1 | \(2\times C_3\) |
| Level 0 leaf | \(1\times C_4\) |

Here the long historical node devotes one eighth of its training distribution to the regime present in the leaf.

Let \(\alpha=1/8\) denote that historical mixture weight. If the other historical component has negligible density near a typical \(C_4\) image, the ideal leaf-minus-history log-density margin is approximately

\[
-\log\alpha=\log 8\approx2.079\text{ nats}.
\]

Treat this as a diagnostic target, not a hard equality: the transformed MNIST distributions overlap and both learned models are approximate.

### Condition 3: identical regime at different levels

| Active interval | Assigned context blocks |
|---|---|
| Level 4 | \(16\times C_4\) |
| Level 3 | \(8\times C_1\) |
| Level 2 | \(4\times C_2\) |
| Level 1 | \(2\times C_3\) |
| Level 0 leaf | \(1\times C_4\) |

This is the crucial cross-level null control. The level-4 node and level-0 leaf represent the same underlying context but differ by a factor of 16 in training-set size. A commensurate density score should not carry a large automatic level or sample-count offset.

Use disjoint training examples whenever a context recurs.

---

## 7. Evaluation sets

For each condition construct:

1. **General active-bank holdout:** 128 held-out examples per active node, sampled according to that node's context mixture, using the existing `NodeHoldout` semantics.
2. **Focused leaf-history holdout:** 512 held-out \(C_4\) images scored only by the level-0 leaf and level-4 historical node.
3. **Curvature audit subset:** 64 focused examples used for multi-start inference and importance-weighted evidence.
4. **Independent-fit controls:** three independently initialized PC model replicas for every active node.

Use one stream-data seed in the minimal experiment and three independent model-training seeds. If the protocol passes, repeat with three stream seeds as confirmation.

Every evidence score must receive only the image tensor.

---

## 8. Phase-gated workflow

## Phase 0 — isolate and pin

Create a new experiment identity, for example:

```text
vamp-logt-generative-pc-evidence-mnist-v1
```

Record:

- VAMP source commit;
- FabricPC version and source commit;
- JAX and JAXlib versions;
- CUDA device;
- complete resolved configuration;
- MNIST and inherited VAMP-AF data hashes;
- all source files participating in the experiment.

Use JAX for model training and inference. Keep PyTorch on CPU for authenticated data loading only so JAX and PyTorch do not compete for GPU memory.

Set JAX allocation flags before importing or initializing the backend, including disabling full-memory preallocation.

## Phase 1 — exact score implementation tests

### Linear-Gaussian exactness test

Construct a fixed model

\[
z\sim\mathcal N(0,I),
\qquad
x\mid z\sim\mathcal N(Wz+b,\sigma^2I).
\]

The exact marginal is

\[
x\sim\mathcal N\!\left(b,WW^\mathsf T+\sigma^2I\right).
\]

For randomly generated test points:

- calculate exact log likelihood analytically;
- calculate the PC MAP score;
- calculate the Laplace score;
- verify that Laplace equals the exact log likelihood.

Required gate:

```text
maximum absolute Laplace error < 1e-4 nats in float64
```

This test must catch:

- a missing latent prior;
- omitted Gaussian constants;
- incorrect score sign;
- incorrect Hessian variables;
- an incorrect log-determinant sign;
- accidental inclusion of classifier parameters or labels.

### Curved-manifold test

Use a one-dimensional latent variable with a fixed nonlinear decoder, for example:

\[
z\sim\mathcal N(0,1),
\]

\[
x_1=z+\epsilon_1,
\qquad
x_2=\frac12z^2+\epsilon_2,
\]

where \(\epsilon_1\) and \(\epsilon_2\) are independent fixed-variance Gaussian noises.

Calculate near-exact marginal likelihood by one-dimensional numerical quadrature. Compare MAP and Laplace scores over:

- ordinary single-mode regions;
- high-curvature regions;
- regions with multiple plausible latent explanations.

This is not expected to make Laplace exact everywhere. Its purpose is to verify the expected approximation behavior before any learned MNIST result is interpreted.

Do not proceed if the linear-Gaussian test fails.

## Phase 2 — one-node PC learning preflight

Train one PC model on 2,000 examples from \(C_0\), using one independent held-out set.

A tiny preflight may compare at most:

- two globally shared image precisions;
- two globally shared hidden precisions;
- two fixed inference step sizes.

Select using only:

- held-out image joint score;
- reconstruction error;
- inference convergence;
- classifier-head accuracy.

Do not inspect LogT routing metrics during this selection.

Freeze the selected architecture, precisions, optimizer budget, inference step count, and Hessian regularizer before Phase 3.

Minimum quality gates:

- held-out classifier accuracy at least 80%;
- held-out complete joint score materially better than the untrained model;
- median latent-gradient norm reduced by at least a factor of ten during inference;
- no NaNs or divergent latent trajectories;
- at least 99% of audit examples admit the globally fixed regularized Cholesky factorization.

Failure here is a model/training failure, not evidence against PC addressing.

## Phase 3 — static LogT experiment

For each of the three schedules:

1. Build the real 31-block binary-counter bank.
2. Train every created leaf and merged PC node on its exact interval replay.
3. Train its classifier head from settled hidden states.
4. Delete child model artifacts only after the parent and bank checkpoint have been committed.
5. Evaluate the five active models with all three score variants.
6. Repeat model training under three independent initialization seeds.

Run both:

- the focused level-0 versus level-4 comparison;
- routing over all five active nodes.

### Primary metrics

Record:

- routed classifier accuracy;
- label-aware oracle-node accuracy;
- routed-minus-oracle accuracy gap;
- routing regret in cross-entropy nats;
- exact temporal-source routing accuracy;
- context-mixture-equivalent source accuracy;
- independent-replica route agreement;
- per-level mean and variance of each raw score;
- leaf-minus-history score distribution;
- leaf win rate against the level-4 node;
- classifier accuracy conditional on routed node;
- inference energy decrease and final gradient norm;
- Hessian minimum eigenvalue and regularization diagnostics;
- MAP-versus-Laplace route disagreement;
- Laplace-versus-importance-audit disagreement.

The label-aware oracle is diagnostic only. For each query it selects the active node whose classifier head gives the lowest true-label cross-entropy.

### Static acceptance gates

Retain the NCE/TRE gates where they remain meaningful:

```text
independent replica route agreement >= 0.90
routed classifier accuracy within 0.10 of oracle accuracy
```

Add:

```text
mean oracle classifier accuracy >= 0.85
novel-leaf focused win-rate lower 95% bootstrap bound > 0.80
recurrent-leaf focused win-rate lower 95% bootstrap bound > 0.50
recurrent-leaf median leaf-minus-history margin > 0
```

For the identical-regime condition, do not impose that the leaf or history node must win. Instead compare the observed level-0/level-4 median score offset with the offset between independent replicas trained on the same interval. Require:

```text
absolute cross-level median offset
<= 2 * typical same-interval replica offset + 0.25 nats
```

This makes the null gate relative to the irreducible variation of independently trained models rather than pretending two finite learned densities should be numerically identical.

Evaluate these gates separately for:

- residual score;
- complete MAP score;
- Laplace score.

The central hypothesis passes if the Laplace score passes all static gates without any post-hoc calibration. A strong result would be MAP failing while Laplace passes. MAP passing as well would still support the normalized-generative-model intuition, while suggesting that posterior volume is not essential at this scale.

If no score passes, stop. Do not run consolidation or a 100-block online stream.

## Phase 4 — one partial carry

Run only if Phase 3 passes.

Use a binary-counter transition from 27 processed blocks to 28 processed blocks. This creates a level-2 parent from four recent blocks while leaving older level-3 and level-4 nodes active, so the newly consolidated model can immediately be compared against other live temporal intervals.

For the new parent:

- train a fresh PC model from the exact four-block union;
- train an independent de novo twin from the same examples, schedule, and architecture;
- checkpoint the committed parent before deleting its two children;
- score both parent replicas against the same held-out data and against the remaining active nodes.

Required gates:

```text
parent/twin routed-node agreement >= 0.90
parent/twin classifier accuracy difference <= 0.02
no systematic raw Laplace-score offset beyond same-data replica variation
children absent from live bank and live model directory after commit
work counters remain inside the fixed LogT bound
```

Do not perform weight averaging, child-logit averaging, energy-offset inheritance, or distillation. The new parent is trained de novo from union replay.

---

## 9. Core scoring pseudocode

The new backend should expose a function with approximately this contract:

```python
@dataclass(frozen=True)
class PcEvidenceScores:
    residual: np.ndarray
    map_log_evidence: np.ndarray
    laplace_log_evidence: np.ndarray
    final_gradient_norm: np.ndarray
    minimum_hessian_eigenvalue: np.ndarray
    hessian_was_regularized: np.ndarray
```

The core calculation should follow this structure:

```python
def score_images(params, images, inference_initial_state):
    # Fixed-cost PC inference. The same initialization protocol is used for
    # every node and query.
    final_state = infer_image_latents(
        params=params,
        images=images,
        initial_state=inference_initial_state,
    )

    u_star = pack_free_latents(final_state)  # concatenated z and h

    complete_joint_nll = vmap(
        lambda image, u: image_joint_nll(params, image, u)
    )(images, u_star)

    map_score = -complete_joint_nll

    hessians = vmap(
        lambda image, u: jax.hessian(
            lambda free_u: image_joint_nll(params, image, free_u)
        )(u)
    )(images, u_star)

    regularized_hessians = hessians + laplace_floor * identity
    _, log_determinants = vmap(jnp.linalg.slogdet)(regularized_hessians)

    free_dimension = u_star.shape[-1]
    laplace_score = (
        map_score
        + 0.5 * free_dimension * log(2 * pi)
        - 0.5 * log_determinants
    )

    return ...
```

Important implementation rules:

- Do not differentiate through the iterative inference trajectory when calculating the Hessian. The Hessian is with respect to the settled free latent state at fixed model parameters.
- Use identical query-dependent initial states across candidate nodes.
- Use a fixed number of inference iterations.
- Route using higher score as better.
- Keep float64 analytic tests.
- Cross-check GPU float32 log determinants against float64 results on a small fixed subset.
- Store raw scores before aggregation or routing.

---

## 10. Files to add

Do not modify the completed NCE/TRE experiment except for genuinely generic imports that are proven backward-compatible.

Add:

```text
src/apm/models/fabricpc_density_backend.py

src/apm/experiments/vamp_logt_pc_config.py
src/apm/experiments/vamp_logt_pc_data.py
src/apm/experiments/vamp_logt_pc_training.py
src/apm/experiments/vamp_logt_pc_workflow.py
src/apm/experiments/vamp_logt_pc_reporting.py
src/apm/experiments/vamp_logt_pc_mnist.py

configs/vamp_logt_pc_mnist/minimal.yaml

tests/test_fabricpc_density_backend.py
tests/test_vamp_logt_pc_protocol.py
tests/test_vamp_logt_pc_smoke.py
```

Reuse unchanged:

```text
src/apm/continual/logt_evidence_bank.py
src/apm/continual/artifacts.py
```

Reuse or carefully wrap the authenticated raw-data functions from:

```text
src/apm/experiments/vamp_logt_evidence_data.py
```

The current NCE/TRE configuration uses 500-example blocks, a 63-block static snapshot, three stream seeds, independent replicas, a 10-point classifier-oracle gap, and exact work counters. The new experiment should have a separate strict schema rather than weakening those existing values in place. fileciteturn32file0L2-L2

---

## 11. Required tests

At minimum:

1. **Exact Gaussian evidence**
   - Laplace equals analytic marginal likelihood.
   - Removing the latent prior makes the test fail.

2. **Normalization constants**
   - Gaussian constants are present with the correct precision sign.
   - Changing a globally shared precision changes scores exactly as expected.

3. **Score sign**
   - Higher model likelihood produces a larger evidence score.

4. **Label isolation**
   - Permuting labels changes classifier training but not image evidence.
   - Context identifiers are rejected by the evidence API.

5. **Classifier isolation**
   - Changing classifier-head parameters leaves all three evidence scores unchanged.

6. **Deterministic scoring**
   - Fixed parameters, images, and initializer produce matching scores within declared numerical tolerance.

7. **Hessian correctness**
   - Autodiff Hessian agrees with finite differences on a tiny model.
   - Linear-Gaussian log determinant matches the analytic value.

8. **LogT topology**
   - Active intervals are disjoint and exhaustive.
   - At most one node exists at each level.

9. **Controlled schedules**
   - Novel, one-eighth recurrence, and identical-regime schedules contain exactly the declared context counts.
   - Recurrent occurrences use disjoint training rows.

10. **No node-size prior**
    - Routing code never reads node example count except for metrics and work accounting.

11. **Merge lifecycle**
    - Parent checkpoint precedes child deletion.
    - Retired children are absent from the active model bank.

12. **Work ceiling**
    - The fixed per-example PC training schedule obeys the declared \(O(t\log t)\) ceiling.

The existing NCE/TRE implementation already has focused tests for disjoint intervals, carries, child deletion, determinism, label isolation, and work bounds; follow the same style rather than inventing a second artifact protocol. fileciteturn35file0L2-L2

---

## 12. Work accounting and asymptotic requirement

Let:

- \(t\) be the number of processed stream blocks;
- \(B\) be the fixed number of examples per block;
- \(E\) be the fixed number of training epochs;
- \(I\) be the fixed number of PC inference steps per presentation;
- \(d\) be the fixed total free-latent dimension, here \(d=160\).

Every example participates in one leaf fit and at most one merge fit at each occupied LogT level. Therefore the number of PC example presentations is bounded by

\[
O(EBt\log t).
\]

Including latent settling gives

\[
O(EIBt\log t).
\]

At inference, there are \(O(\log t)\) active models. MAP routing costs

\[
O(I\log t)
\]

fixed-model evaluations. Full Laplace routing costs

\[
O\!\left((I+d^3)\log t\right)
\]

under dense Hessian factorization. Since \(I\) and \(d\) are fixed independently of stream age, this remains \(O(\log t)\) in lifetime, although its constant may be substantial.

Track:

```text
pc_leaf_example_presentations
pc_merge_example_presentations
pc_inference_state_updates
pc_route_model_evals
pc_laplace_hessian_evals
pc_importance_audit_samples
active_pc_models
```

Programmatically assert:

\[
\texttt{pc\_example\_presentations}
\leq
R\,E\,B\,t\,\lceil\log_2(t+1)\rceil,
\]

where \(R\) is the fixed number of independent model replicas.

No adaptive convergence loop may grow with node age. Convergence diagnostics may invalidate a run, but may not silently buy more iterations for an older or harder node.

---

## 13. Reporting

Produce:

```text
artifacts/vamp-logt-pc-mnist/runs/<config-hash>/
    protocol.json
    config_resolved.yaml
    calibration/
    preflight/
    static/
        novel_leaf/
        recurrent_leaf_1_8/
        identical_regime/
    consolidation/        # only if static passes
    metrics.jsonl
    report.md
    report.html
```

The report should include:

- the exact probabilistic model and score formulas;
- analytic score-validation results;
- one-node learning quality;
- score matrices for each static condition;
- leaf-minus-history margin histograms;
- MAP-versus-Laplace route changes;
- routed and oracle classifier accuracies;
- routing regret;
- inter-replica route agreement;
- raw score differences between independent fits;
- Hessian and posterior-mode diagnostics;
- measured training and routing work;
- explicit reason for any gated stop.

Also include the completed NCE/TRE result as historical context, not as a numerically matched control. That experiment passed its calibration but failed every static routing schedule and therefore did not proceed to consolidation or online evaluation. fileciteturn30file0L2-L2

---

## 14. Verdict rules

The report must classify the outcome using these categories.

### Supported

- PC models themselves have adequate oracle task performance;
- complete MAP or Laplace scores are stable across independent fits;
- Laplace passes both novel and overlapping-regime routing gates;
- routed accuracy is within ten percentage points of oracle;
- no systematic level offset appears in the identical-regime control.

### Partially supported

Examples:

- focused leaf-versus-history comparisons pass but full-bank routing does not;
- MAP passes and Laplace adds little;
- Laplace works but is highly sensitive to multimodal initialization;
- evidence is commensurate but the PC task model has insufficient capacity.

### Not supported by this implementation

- PC oracle performance is good;
- latent inference converges;
- the analytic score implementation passes;
- but MAP and Laplace evidence still fail the cross-node routing gates.

### Inconclusive

- PC models fail to learn the individual node distributions;
- latent inference does not settle;
- Hessian diagnostics are pathological;
- classifier oracle performance is too low;
- the exact score tests fail.

This distinction is essential. A weak PC model is not evidence that normalized generative PC scores are intrinsically non-comparable.

---

## 15. Non-negotiable scope constraints

Codex should not:

- route on the current raw FabricPC energy unchanged;
- use the old frozen CNN or top-two adapter as the selected node's inference model;
- add a learned router;
- introduce NCE, TRE, KDE, PCA, or frozen embeddings;
- fit node-specific score offsets;
- learn separate node precisions;
- include node sample count in routing;
- use context or label information in evidence;
- initialize merged models by weight averaging;
- use ePC in the first run;
- run the full 100-block online experiment before the static gates pass;
- alter the completed NCE/TRE protocol or its artifacts.

The canonical command should be:

```bash
uv run --python 3.11 python -m apm.experiments.vamp_logt_pc_mnist \
  --config configs/vamp_logt_pc_mnist/minimal.yaml
```

The experiment is complete when the analytic tests, one-node preflight, all three 31-block static conditions, phase-gated report, and work-bound assertions have run. The partial carry is required only if at least one principled PC score passes the static gates.
