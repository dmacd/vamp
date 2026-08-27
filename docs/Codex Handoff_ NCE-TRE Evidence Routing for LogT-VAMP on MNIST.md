# Codex Handoff: NCE/TRE Evidence Routing for LogT-VAMP on MNIST

## Objective

Replace embedding-plus-PCA addressing with one learned evidence model per active LogT temporal node.

Reuse the current MNIST contexts, base CNN, adapter implementation, replay machinery, oracle-routing diagnostic, routing-regret metric, determinism checks, and work counters. Do **not** reuse `AddressingFirstTree` as the node structure: it maintains a spatial tree with many active leaves. This experiment needs the LogT binary-counter structure, with disjoint temporal nodes and at most one active node per level.

Let \(t\) denote the number of fixed-size stream blocks processed. There must be \(O(\log t)\) active nodes.

## Evidence model

For every active temporal node \(v\), maintain a trainable conditional discriminator \(R_v(x,k)\):

- \(x\) is the raw image, not a frozen embedding.
- \(k\) identifies one step in a fixed sequence of \(K\) intermediate distributions.
- The network must have the same trainable backbone architecture, width, and depth as `AddressCNNOriginal`. Replace the ten-class output with a scalar output and add bridge-level conditioning. All convolutional and dense layers remain trainable.
- Do not use PCA, feature tables, context labels, digit labels, tiny MLP heads, or frozen-base features in the evidence path.

Define:

- \(P_{v,0}\): node \(v\)'s image distribution after one fixed, small corruption that defines “near the data manifold.”
- \(P_{v,K}=Q\): one tractable reference distribution shared by every node at every time.
- \(P_{v,1},\ldots,P_{v,K-1}\): intermediate “waymark” distributions between them.

Use progressive random replacement of image coordinates by values from \(Q\). The final step replaces every coordinate, making the endpoint exactly common and normalized. This construction has a direct language-model analogue: progressively replace tokens rather than pixels. Do not introduce MNIST-specific rotations, handcrafted features, or context-dependent references.

For each adjacent pair, train \(R_v(x,k)\) with balanced binary noise-contrastive estimation:

\[
R_v(x,k)
\approx
\log P_{v,k}(x)-\log P_{v,k+1}(x).
\]

Balanced sampling is mandatory; otherwise apply the exact known class-prior correction. Noise-contrastive estimation identifies the additive energy offset rather than leaving each node with an arbitrary energy zero.

The node evidence is

\[
e_v(x)=\sum_{k=0}^{K-1}R_v(x,k)
\approx
\log P_{v,0}(x)-\log Q(x).
\]

Because \(Q\) is identical for every node, route by

\[
v^*(x)=\arg\max_v e_v(x).
\]

Direct NCE is the \(K=1\) baseline. TRE is the same construction with enough intermediate distributions to prevent the data-versus-reference classifier from saturating across a large density gap.

Keep \(K\) fixed independently of stream length. A conditional full-capacity network may be evaluated for all \(k\) values in one batch.

## De-risking order

### 1. Verify the estimator before involving continual learning

Add a small automated test using known normalized multimodal distributions. Train direct NCE and TRE independently several times and verify:

- recovered log-density ratios have the correct additive offset;
- independently trained models agree within measured error;
- TRE succeeds in a case where direct NCE saturates;
- summed ratio error follows the basic bound

\[
|e_v-\hat e_v|
\leq
\sum_k |R_v(\cdot,k)-\hat R_v(\cdot,k)|.
\]

This is an implementation and calibration test, not an experimental result.

### 2. Static MNIST routing test

Construct a fixed snapshot of genuine LogT active nodes from the existing temporal stream. Freeze their already-trained adapters. Train evidence models only from each node's own replay data plus on-the-fly samples from the common reference.

Compare:

1. direct full-capacity NCE;
2. full-capacity TRE;
3. oracle node selection by minimum classification loss.

Measure:

- held-out temporal-source routing accuracy;
- final classifier accuracy under evidence routing;
- routing regret against oracle selection;
- adjacent-waymark classifier saturation;
- evidence-score variation across independent training seeds;
- performance by LogT level, to detect level-dependent score offsets.

Select the smallest fixed \(K\) for which adjacent bridge classifiers retain meaningful overlap. Freeze that schedule before any online experiment.

Do not proceed unless TRE produces stable cross-node score comparisons and brings routed classifier accuracy to within **10 percentage points** of the oracle-node result on all three existing seeds. A weaker result means the central addressing premise has not yet been demonstrated.

### 3. Consolidation test

Implement the binary-counter carry:

- a new block creates a level-zero node;
- two equal-level nodes are replaced by one node at the next level;
- the new adapter and new evidence model are trained from the same union replay pass;
- the two old adapters and evidence models are then deleted.

Train the merged evidence model as a full-capacity model on the merged replay distribution. Do not average child logits, merge energies algebraically, or distill into a smaller network in this experiment.

For selected merges, train a separate de novo control on the same union replay. Compare its evidence scores, routing decisions, and held-out loss against the normally consolidated model. Reject the method if evidence develops a systematic offset with node level or repeated consolidation.

### 4. Full online comparison

Run the original five-context MNIST stream with:

- the existing PCA-addressed VAMP-AF result as the recorded baseline;
- true LogT plus direct NCE;
- true LogT plus TRE;
- oracle-node routing as the upper bound.

Keep the existing adapter-capacity controls and routed-versus-oracle evaluation unchanged.

## Complexity requirements

Each active node owns one full-capacity conditional evidence model. There are \(O(\log t)\) active models.

Routing evaluates every active model, so it costs

\[
O(K\log t)=O(\log t)
\]

for fixed \(K\).

During learning, every stream example participates in one leaf training event and at most one consolidation at each LogT level. Use:

- a fixed number of evidence updates per replay example;
- one sampled bridge index and a constant number of reference examples per positive example during training;
- no all-pairs node discrimination;
- no negatives retrieved from every historical node;
- no rescanning of the full stream;
- no convergence loop whose budget grows with node age.

The cumulative evidence-training work is therefore \(O(t\log t)\), with a constant factor determined by the fixed optimizer schedule and fixed \(K\).

Add and report:

- `evidence_train_example_updates`;
- `evidence_merge_example_updates`;
- `evidence_reference_examples`;
- `evidence_route_model_evals`;
- `active_evidence_models`.

Assert programmatically that cumulative evidence example-updates are bounded by a fixed constant times \(t\lceil\log_2(t+1)\rceil\).

## Deliverables

Add:

- `src/apm/continual/nce_tre_evidence.py`;
- a minimal true-LogT temporal-node manager or experiment-local equivalent;
- `src/apm/experiments/vamp_logt_evidence_mnist.py`;
- tests for ratio calibration, disjoint active intervals, merge deletion, routing, determinism, and the work bound;
- one result document containing static de-risking results, merge results, full-stream accuracy, routing regret, bridge saturation, and measured work.

Keep the existing VAMP-AF implementation and recorded baseline untouched.