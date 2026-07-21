# TinyWorlds-P v1 Execution Tracker

TinyWorlds-P replaces generated benchmark prose with genuine stories from the
pinned TinyStories training corpus. It withholds five noun-bucket × verb-bucket
conjunctions while keeping their individual noun and verb components visible in
the reduced base corpus. The model receives story tokens only; recipes, bucket
labels, cells, and split metadata are partition-construction records.

The supplied planning document is archived alongside this tracker as
[TinyStories - partitioned.pdf](TinyStories%20-%20partitioned.pdf). `DESIGN.md`
is the durable implementation contract and `PLAN.md` is the live repository
roadmap.

## Immutable identities

- Benchmark: `tinyworlds-p-v1`.
- Corpus: `TinyStories-train.txt`, revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, 1,924,281,556 bytes,
  SHA-256 `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`.
- Metadata: the pinned local `TinyStories_all_data.tar.gz` archive.
- Tokenizer: the existing hashed 50,257-token GPT-2 BPE files.
- Public deterministic seed: `0`.
- Base model: a fresh eight-layer, width-256 GPT-Neo; no old checkpoint is a
  baseline or initialization source.

## Partition contract

Identity normalization is NFKC, case folding, whitespace collapse, and
canonical straight quotes. Training shards preserve the original corpus bytes.
Corpus and metadata records are externally sorted and merged by normalized
SHA-256, so neither source is held in memory and worker completion order cannot
affect the result. Exact normalized duplicates form indivisible assignment
groups, with every raw occurrence retained.

Only stories with a uniquely recoverable noun, verb, and adjective recipe are
eligible. Conflicting metadata, unclassifiable metadata, and unmatched corpus
stories are excluded from both base and worlds. Construction stops below 95%
hash-match coverage, 95% role coverage among matched token mass, or 90% combined
eligible coverage.

Nouns, verbs, and adjectives are independently assigned by token mass to
deterministic greedy buckets. Adjectives stratify splits but do not define held
out worlds. Five balanced cells are selected and canonically relabelled:

```text
A = N0 × V0    B = N1 × V0
D = N0 × V1    C = N1 × V1    E = N2 × V2
```

Duplicate groups are split by active token mass. Worlds use 80/10/10 and the
held-in complement uses 96/2/2 train/validation/test. The allocator balances
source model, feature signature, adjective bucket, and token-length bin.
Validation and test worlds receive no-replacement held-in controls split
between the same noun row and same verb column, under the predeclared strict
matching tolerances. Test is sealed before model training or selection.

The published partition contains canonical assignments, original-text shards,
little-endian `uint16` token shards, document indexes, controls, audits,
manifests, and a strict file-size/SHA-256 tree. Rebuilding from the same inputs
must reproduce every byte.

## Base-training contract

Training uses 256-token windows, 32 sequences per microbatch, and eight-way
gradient accumulation. Gradients are normalized by total active tokens. The
optimizer is AdamW with peak learning rate `5e-4`, betas `(0.9, 0.95)`, epsilon
`1e-8`, weight decay `0.1`, global gradient clip `1.0`, 1% linear warmup, then
cosine decay to `5e-5`. Parameters and optimizer state remain float32.

The streaming trainer memory-maps token shards and records epoch, block,
microbatch, optimizer update, RNG, and schedule position. It emits progress and
loss JSONL continuously and immutable complete training states every 1,000
updates and at each epoch. Interrupted/resumed execution must be bit-identical
to uninterrupted execution.

An 8×8 partition is calibrated for two epochs. A low gap triggers one 6×6
fallback; an excessive gap triggers one 10×10 fallback. Training-quality
failures do not trigger regridding. A passing run continues through epoch five,
and selection uses the lowest held-in validation NLL among gap-eligible epoch
2–5 checkpoints, with earlier epochs winning ties. Test is opened exactly once
after selection. Publication also requires held-in validation NLL at most 2.0
and an observed allocator peak below 12 GiB.

## Execution tracker

- [x] Implement immutable partition, source, shard, control, and manifest
  contracts.
- [x] Implement bounded external source join, duplicate grouping, mechanical
  role recovery, coverage gates, deterministic buckets, cell selection,
  stratified allocation, and strict controls.
- [x] Implement strict artifact persistence/loading, original-text and token
  shards, memory-mapped deterministic batching, and tamper checks.
- [x] Implement token-weighted streaming training, complete resume states,
  validation, calibration fallback, selection, sealed test, and publication.
- [x] Add fixed preparation and training runners with temporary-directory,
  phase, progress, and ETA reporting.
- [x] Add a deterministic CPU end-to-end fixture, including interrupted/resumed
  parity.
- [x] Pass the complete default test suite before canonical preparation (828
  passed, one skipped, 11 deselected; warning-free 33m46s exact-revision run on
  2026-07-20).
- [x] Attempt and audit the canonical build. It stopped before promotion at
  78.9663% hash-match and 78.9425% eligible coverage, below 95% and 90%; see
  `TW-P_SOURCE_AUDIT.md`. No partial partition was published.
- [ ] Pass focused TinyWorlds-P plus shared LM/checkpoint tests after partition
  promotion and immediately before accelerator training. Do not rerun the
  parked legacy TinyWorlds-v2 suite; its dominant cache/reseal case measured
  24m54s and was explicitly excluded from subsequent milestone verification on
  2026-07-20.
- [ ] Run the RTX 4090 calibration, any single permitted grid fallback, and the
  five-epoch final training path. Blocked by the mandatory source gate; the GPU
  was verified but training was correctly not started.
- [ ] Publish the selected checkpoint, sealed-test results, samples, strict
  tree, and final measured report. Blocked by the mandatory source gate.

Continual LoRA/VAMP streams, replay, consolidation, regeneration, semantic
clustering, and near-deduplication are deliberately outside this milestone.
