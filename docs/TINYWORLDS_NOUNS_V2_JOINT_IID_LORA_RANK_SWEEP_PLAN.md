# TinyWorlds nouns-v2 joint-IID LoRA rank sweep

## Question

The temporal-consolidation study's joint-IID rank-8 LoRA reaches 1.554322
story-weighted and 1.590877 token-weighted suffix NLL, substantially worse than
the joint-IID full-model control at 1.399026 and 1.452044. This addendum tests
whether that gap is primarily a rank-8 capacity limitation by evaluating LoRA
ranks 4, 8, 16, and 32.

The experiment changes only LoRA rank. It does not change the selected base,
training population, epochs, example order, optimizer settings, validation
stories, midpoint split, suffix masks, or NLL aggregation.

## Authenticated parent evidence

The runner must strict-load temporal contract
`3f4ef4a10fd471b418a32a8f7b45431602c1f6abc080c19a7822ea2c2dd839b4`
and publication manifest
`15f3ee2a5a2c5054b158ba62d7a0d1b9fcaa22e40634a73c9cbffceca5888bcb`.
It authenticates every parent publication file plus:

- rank-8 job `cd4605c8240b459058c5a916ac6747edfd7712e99fcfd3710bd80cad1470a3cb`,
  its adapter, 15,024-row loss trace, and manifest;
- full-model job `61376ee6e474516ab6471d74ca97dfe2737586863c2d5b3a50c123147120bc80`,
  its checkpoint, loss trace, and manifest;
- the original 4,440-row rank-8 evaluation ledger, hash
  `a0a5308b77bfc632dc91fb1b027e9f2fa1b9e7d51a6f913f5808733d3685692b`;
- the original 4,440-row full-model ledger, hash
  `46b9fef540af40897e08461a123839ee39f98c5b4566e22bd0583bf5a7ecbbe8`.

The published aggregate is recomputed from those source ledgers and must match
exactly before new training begins. Canonical nouns-v1/v2 and all bound parent
files are hashed before the run and required to remain unchanged afterward.

## Training specification

All ranks use the exact 98,304 joint-IID stories selected by the temporal
contract: 4,096 per noun, divided into the same 192 source shards. Each model
gets four finite epochs. Stories are permuted and windowed with the canonical
rank-8 job identity as the batch namespace, producing the same 15,024 physical
minibatches of 32 reset-at-256 causal windows.

The LoRA target set remains all six projections in every transformer block.
AdamW uses learning rate `1e-3`, weight decay `0.01`, gradient clipping `1.0`,
and the existing deterministic training key. Alpha equals rank, so
`alpha / rank = 1` for every condition. This preserves the canonical rank-8
scale while varying only factor capacity. The common random namespace is the
canonical rank-8 job identity; differently shaped random arrays are still not
claimed to be nested initializations.

Rank 8 is not retrained. Its authenticated adapter and training trace are the
comparison anchor. Ranks 4, 16, and 32 train into contract-addressed independent
checkpoint directories. Each loss ledger and optimizer state is hash-bound and
resumable at an exact update boundary. Interrupted runs truncate only ledger
rows beyond the latest authenticated state. Published adapters strict-load
their tensor names, shapes, dtype, values, job record, loss trace, and manifest.

## Evaluation and metrics

Each adapter is forced for all 4,440 official final validation stories. This is
the same `joint_iid_lora` condition as the parent report: no routing decision is
being studied. The router midpoint is retained in each row solely to preserve
the parent schema and exact case identity. Evaluation scores the evaluator-only
story suffix using the same reset-at-256 windows and token masks.

The primary results are:

1. story-weighted suffix NLL (each story contributes equally), matching the
   parent final-quality figure;
2. token-weighted suffix NLL (all suffix losses divided by all suffix tokens).

Suffix top-one token accuracy is secondary and is explicitly labeled as
teacher-forced next-token accuracy, not routing accuracy. The report also gives
per-noun NLL, trainable parameter count, adapter size, final training loss,
optimizer updates, runtime, and allocator peak.

All new ledgers must match the rank-8 ledger's exact `(task, story)` order,
prefix count, suffix token count, and total 476,035 suffix targets. The base
candidate is scored alongside every adapter as a numerical control; maximum
per-story base-path NLL drift across rank-shaped compilations must be at most
`2e-5`.

Paired condition-minus-reference differences use a deterministic seed-zero,
10,000-sample bootstrap stratified by noun. Each rank is compared with the
joint-IID full model; ranks 4, 16, and 32 are also compared with canonical rank
8. Intervals are reported for both story-weighted and token-weighted NLL.

## Execution and publication

`scripts/run_tinyworlds_nouns_v2_joint_iid_rank_sweep.py` is the only runner and
has no CLI options. It fixes GPU 0, disables JAX preallocation, uses the async
allocator, and enforces the existing 12 GiB peak gate. It prints the persistent
temporary directory and five human-readable phases. Training and evaluation
have exact per-rank progress bars with live phase ETAs, metrics, and an overall
ETA. A desktop notification is attempted after completion.

The independent result bundle lives at
`results/language_cl/tinyworlds-nouns-v2/temporal-consolidation/<parent>/joint-iid-lora-rank-sweep-v1/`.
Resumable ledgers stay below the parent `.work-v1` tree; new adapters stay below
the parent checkpoint tree. The report bundle contains:

- canonical contract, analysis, execution, allocator, and manifest JSON;
- aggregate, per-task, bootstrap, training, and ledger-provenance CSV files;
- an accessible Matplotlib SVG with separate story- and token-weighted panels;
- separate Markdown and self-contained HTML reports with collapsible method,
  uncertainty, per-task, and provenance sections.

Report construction runs twice and must be byte-identical. An exact-resume
replay must strict-load all three new adapters and ledgers without changing any
published byte.

## Verification gates

Focused tests cover canonical rank-8 identity preservation, scale-one rank
configs, rank-bound job identities, arbitrary-rank adapter packing/scoring,
training resume, evaluation row order, parent-row aggregate parity, paired
bootstrap determinism, malformed/tampered ledgers, report accessibility,
standalone HTML, CSV coverage, and byte-identical regeneration. A bounded GPU
smoke checks rank-shaped training/evaluation and allocator access before the
full run. After completion, run the exact replay, the opt-in real-source check,
and the clean default suite.
