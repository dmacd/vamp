# TinyWorlds Nouns-v2 bounded addressing experiment plan

## Purpose and frozen boundary

This is a preregistered, final-checkpoint study of bounded task-free addressing.
It does not train the base model, retrain a LoRA edge, alter the VAMP graph, or
replace a canonical nouns-v1/nouns-v2 result. The only executable entry point is
`scripts/run_tinyworlds_nouns_v2_addressing_study.py`; it has no options and uses
GPU 0.

The run must authenticate these existing nouns-v2 identities before it evaluates
anything:

- partition `210c4e2d067077fe774782024a594ade7e7472a986d554f186453549cf910f1b`;
- selected-base training identity
  `94831c31c8f11a594534c2989182d378fc2e022382b61168b73e7400f9648e21`;
- selected-base parameter checksum
  `fff309bfbfcee8d59c5c3fc04152cc37be2142201f3bf9116b7b024e81a24f3c`;
- final 25-node/24-edge VAMP tensor checksum
  `97414ac3d8656ab083b2e570a4162dc69b024f90cf819b80b1cab94213553e63`;
- canonical local-complete run
  `f758b8aeef6f06f4992b245d656dd5d99034a13a07c23376f2f62c3e643fa177`;
- all 900 registered probes (36 root probes and 36 probes for each of 24
  task nodes), all 4,440 official midpoint validation stories, and every
  existing whole-story, generation, VAMP-stagewise, adapter-control, and
  full-model-control ledger row.

The runner snapshots the canonical nouns-v1 and nouns-v2 checkpoints, ledgers,
reports, manifests, and partitions by SHA-256 before evaluation and requires the
same snapshot after publication. Authentication uses strict load-only APIs: an
absent or mismatched selected base, preflight, stage, or ledger is a hard failure,
not permission to recreate it.

## Experiment 1: dense versus physically compact EBT-H

The first comparison uses the canonical stored content keys and the same EBT-H
objective in three execution modes:

1. dense all-node EBT-H over all 25 nodes and all 24 resident edge factors;
2. physically compact top-4 EBT-H;
3. physically compact top-8 EBT-H.

All methods use 20 Adam steps, learning rate 0.1, temperature 1, entropy penalty
0.01, and Hopfield inverse temperature 10. Compact execution still computes all
25 cheap content-key dot products. It then retains only the selected four or
eight node paths, gathers each example's insertion-ordered union of path edges,
and optimizes only four or eight logits. The transformer receives a batched edge
bank containing only those gathered factors. Physical capacity is padded to the
smallest sufficient bucket in `4, 8, 12, 16, 20, 24`; prefix length retains the
existing 32-token buckets and evaluation uses eight-row microbatches.

This differs from logical masking. A dense masked implementation leaves every
edge factor resident and merely assigns zero coefficients outside a shortlist.
Physical compaction changes the tensors executed by the projection kernels. A
dense masked path remains solely as a numerical reference. Before the production
ledger starts, compact and dense masking must agree on real final-checkpoint data
within `1e-3` for candidate probabilities, expanded edge coefficients, soft NLL,
hard NLL, and the complete optimization trace, with exactly equal selected nodes.

The primary outputs are route accuracy, suffix story-weighted and token-weighted
NLL, regret versus the stored oracle path, canonical true-node recall@4/recall@8,
selected path-edge count, gathered active edge count, synchronized warm GPU
latency, throughput, cold compilation, and end-to-end wall time. Compact top-8 is
preregistered as non-inferior only if both of these conditions hold:

- story-weighted suffix NLL increases by no more than 0.02 versus dense all-node;
- route accuracy falls by no more than 2 percentage points.

The report is published regardless of the verdict.

## Experiment 2: frozen content and error keys

The second comparison keeps the base and all VAMP tensors frozen and evaluates
five deterministic key schemes under compact top-4 and top-8 EBT-H:

- `canonical_full_centroid`: the existing centroid of full-probe final hidden
  encodings;
- `midpoint_content_centroid`: a centroid of probe encodings cut by the exact
  deployment midpoint rule;
- `midpoint_content_prototype`: maximum cosine similarity over all 36 midpoint
  probe encodings for a node;
- `midpoint_content_residual_centroid`: a centroid of fused midpoint content and
  analytic error encodings;
- `midpoint_content_residual_prototype`: maximum cosine similarity over all 36
  fused probe encodings for a node.

For active prefix transition \(t\), the error signature is the analytic gradient
of cross-entropy with respect to the final hidden vector:

\[
g_t = \operatorname{softmax}(\mathrm{logits}_t)E - E[\mathrm{target}_t],
\]

where \(E\) is the tied token-embedding matrix. Active gradients are mean-pooled
under the transition mask and L2-normalized. Unit content and residual vectors
are fused as `[content / sqrt(2), residual / sqrt(2)]`. The 36 registered probes
for every node, including the root, are used. No validation story contributes to
a key. A router query contains only the story's first-half causal transitions;
the task noun and second-half tokens remain evaluator-only metadata.

The primary Experiment 2 ordering is recall@8 first and compact top-8 suffix NLL
second. The report also includes recall@1/4/8, retrieval entropy and score margin,
compact route accuracy, final EBT entropy/margin, and per-task selected-node
confusion. Every alternative is paired by task/story with the canonical key.
Differences use exactly 10,000 bootstrap resamples from NumPy's seed-zero
generator and report the 2.5% and 97.5% quantiles.

## Exact coverage and operation accounting

The study streams two independent, self-hashing v1 JSONL ledgers:

- exactly 22,200 retrieval rows: five schemes times 4,440 cases;
- exactly 48,840 EBT rows: five schemes times two compact widths times 4,440
  cases, plus one canonical dense-all control times 4,440 cases.

Contracts bind the selected base, final VAMP tensor checksum, graph, partition,
probe set, validation set, canonical run and ledger hashes, and frozen key
artifact. Result rows bind their own contract and have independent format and
content hashes. Resume accepts only a canonical unique prefix of expected rows,
truncates at most an incomplete last line, and rejects malformed, duplicate,
unexpected, or tampered records.

Every observed `(mode, candidate width, prefix-width bucket, physical-edge
bucket)` shape receives one separately reported cold compile and five synchronized
warm repetitions. Timing and cost output keeps these units separate:

- GPU kernel latency and examples per second;
- end-to-end wall time;
- model-forward-equivalent prefix tokens;
- all-node Hopfield dot products;
- active LoRA-edge evaluations.

The cumulative stage-sequence cost figure compares canonical exhaustive,
Hopfield, EBT-uniform, and EBT-H routing in separate operation-unit panels. A
separate final-checkpoint figure compares measured compact latency and quality.

## Publication and gates

All new artifacts live under
`results/language_cl/tinyworlds-nouns-v2/addressing-study/`. The default runner
prints its resumable temporary ledger directory, phase lines, phase and overall
ETAs, and sparse evaluation coverage. It enforces the existing 12 GiB allocator
gate and attempts a desktop completion notification.

Publication contains independent Markdown and self-contained HTML reports;
aggregate, per-task, timing, and cost CSV exports; accessible Matplotlib SVG
plots; and Graphviz top-4/top-8 VAMP renders. Graph nodes encode candidate
inclusion frequency and graph edges encode compact union-path activation.
Method, bootstrap, per-task/confusion, timing/cost, graph, and provenance detail
is collapsible. The report explains why frozen-base representations may carry
useful lexical signals, why full-probe centroids are mismatched to midpoint
queries here, the exact residual formula, and physical versus logical compaction.
The report tree must regenerate byte-for-byte from the authenticated ledgers.

Release requires focused CPU tests, a bounded real-GPU smoke, the complete run,
an exact-resume replay, the opt-in real-source gate, and the clean default suite.
Tests cover analytic residual/autodiff agreement, normalization and padding,
midpoint/root/suffix isolation, prototype scoring and stable ties, compact
structure plus eager/JIT/gradient parity at top-4/top-8, interruption and
tamper rejection, exact coverage, accessible standalone reports, complete graph
coverage, and deterministic regeneration.
