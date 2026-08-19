# TinyWorlds nouns-v2 joint-IID LoRA plus tied-embedding experiment

## Question

The matched joint-IID rank sweep showed that projection-only LoRA does not
approach the full-model suffix NLL when its rank rises from 8 to 32. This
addendum tests the leading architectural explanation: projection LoRA freezes
the token embedding, even though this GPT-Neo implementation also uses that
same matrix as its output language-model classifier.

The primary comparison is projection-only LoRA versus projection LoRA plus a
jointly trainable tied token embedding at ranks 8 and 32. The authenticated
joint-IID full model remains the quality reference.

## Fixed training protocol

Both new conditions start from the exact selected nouns-v2 base and use all six
LoRA projection targets in each of its eight transformer blocks. The token
embedding starts byte-identically from the base and remains tied: the same
trainable matrix supplies input token vectors and output logits. Position
embeddings, layer-normalization parameters, linear biases, and all original
transformer kernels remain frozen.

Each condition sees the same 98,304 joint-IID stories, four finite epochs,
15,024 minibatches of 32 reset-at-256 windows, and the canonical rank-8 batch
and random namespace used by the completed rank sweep. Rank and alpha are both
8 or both 32, so LoRA scale remains one.

A single joint loss updates two AdamW parameter groups after one combined
global-norm clip at 1.0:

- LoRA factors: learning rate 1e-3 and weight decay 0.01;
- tied embedding/head: learning rate 5e-5 and weight decay 0.01.

These are the already published LoRA and full-model learning rates. Using a
separate embedding group avoids applying the adapter's twenty-times-larger
step directly to the 12.9-million-parameter vocabulary matrix.

## Evaluation and analysis

Each trained model is forced for all 4,440 official final validation stories.
The evaluator uses the same midpoint boundary, evaluator-only suffix masks,
reset-at-256 windows, story order, and 476,035 suffix targets as the parent
temporal report and rank sweep. Routing is not involved.

Primary metrics are story-weighted and token-weighted suffix NLL. Teacher-forced
suffix top-one token accuracy is secondary. Rows also preserve the learned
embedding-only candidate NLL before its jointly trained LoRA is applied; this
is a diagnostic decomposition, not a separately trained condition.

Paired differences use a deterministic seed-zero 10,000-sample bootstrap
stratified by noun. Required comparisons are each joint condition against its
same-rank projection-only control and against the full model, plus rank 32
against rank 8 within the joint method. The report includes per-noun results,
training loss, runtime, parameter counts, embedding displacement, allocator
peak, and exact ledger provenance.

## Execution and integrity

The no-options runner fixes GPU 0, disables JAX preallocation, enforces the
existing 12 GiB allocator gate, prints its persistent work directory, and
provides phase, rank, and overall progress estimates. Training state, optimizer
moments, RNG, update position, loss ledgers, and evaluation ledgers resume at
authenticated boundaries. Only the latest large optimizer checkpoint is kept
after a newer checkpoint has strict-loaded successfully.

The addendum has an independent contract, checkpoint tree, chained JSONL
ledgers, Markdown report, self-contained HTML report, accessible Matplotlib SVG,
CSV exports, completion record, and publication manifest. It binds and protects
the canonical temporal publication, the completed rank-sweep publication,
their reference checkpoints, and their evaluation ledgers. Replaying a
completed run must perform no optimizer work and reproduce every publication
byte.

Focused tests cover tied use of the trainable embedding, frozen non-target
parameters, optimizer-group learning rates, rank-shaped training, interruption
and exact resume, strict artifact loading, malformed ledgers, deterministic
bootstrap, standalone accessible reports, and the no-options GPU runner. A
bounded GPU smoke precedes the complete experiment.
