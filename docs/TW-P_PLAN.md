# TinyWorlds-P Archive v1 Execution Tracker

TinyWorlds-P replaces generated benchmark prose with genuine stories taken
directly from released entities in the pinned TinyStories metadata archive. It
withholds five noun-bucket × verb-bucket conjunctions while keeping their
individual noun and verb components visible in the held-in base partition. The
model receives story tokens only; prompts, recipes, bucket labels, cells, and
split metadata are partition-construction records.

The supplied planning document is archived alongside this tracker as
[TinyStories - partitioned.pdf](TinyStories%20-%20partitioned.pdf). `DESIGN.md`
is the durable implementation contract and `PLAN.md` is the live repository
roadmap.

The archive-v1 execution is complete with a controlled low-gap stop. The
current result and evidence hashes are in the
[archive-only calibration report](TW-P_ARCHIVE_CALIBRATION_REPORT.md).

## Immutable identities

- Benchmark: `tinyworlds-p-archive-v1`.
- Sole story source: `TinyStories_all_data.tar.gz`, revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, 1,608,001,638 bytes,
  SHA-256 `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
- Explicit non-inputs: `TinyStories-train.txt`, `TinyStories-valid.txt`, and
  both GPT-4-only text aggregates. They define no source coverage, eligibility,
  split, or baseline for TinyWorlds-P.
- Tokenizer: the existing hashed 50,257-token GPT-2 BPE files.
- Public deterministic seed: `0`.
- Base model: a fresh eight-layer, width-256 GPT-Neo; no old checkpoint is a
  baseline or initialization source.

## Partition contract

Identity normalization is NFKC, case folding, whitespace collapse, and
canonical straight quotes. It is used only to group duplicate archive stories.
Training shards preserve the exact UTF-8 bytes of each accepted archive
`story`. Archive entities are externally sorted by normalized story SHA-256,
so the source is never held in memory and worker completion order cannot affect
the result. Exact normalized duplicates form indivisible assignment groups,
with every released archive record, provenance location, and multiplicity
retained.

Only stories with a uniquely recoverable noun, verb, and adjective recipe are
eligible. Duplicate records with conflicting recipes and records with
unclassifiable released metadata are excluded from both base and worlds. There
is no corpus/archive matching step, unmatched-corpus category, hash-match gate,
or combined corpus-coverage gate. All base and world split assignments, and
every matched-control selection from the held-in splits, come exclusively from
the eligible archive groups.

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
fallback with held-in 94/3/3 splits, preserving approximately 12% shared-row /
shared-column validation-and-test control capacity for 10% world demand. An
excessive gap triggers one 10×10 fallback with held-in 96/2/2 splits.
Training-quality failures do not trigger regridding. A passing run continues
through epoch five, and selection uses the lowest held-in validation NLL among
gap-eligible epoch 2–5 checkpoints, with earlier epochs winning ties. Test is
opened exactly once after selection. Publication also requires held-in
validation NLL at most 2.0 and an observed allocator peak below 12 GiB.

## Execution tracker

- [x] Record the archive-only source decision. The pinned archive is the sole
  story universe, and all published TinyStories text aggregates are irrelevant
  to TinyWorlds-P construction and training.
- [x] Preserve the prior train/archive source and calibration audits as
  historical diagnostics only. The intersection partition and its scratch run
  are explicitly ineligible for publication or resume under this contract.
- [x] Retain the source-independent implementation for deterministic
  buckets/topology, stratified splits, matched controls, calibration decisions,
  generic contracts, and fixed scratch-training policy.
- [x] Remove `TinyStories-train.txt` and corpus identity from TinyWorlds-P input
  and artifact contracts. Do not retain the corpus join as a compatibility
  alias or optional mode.
- [x] Replace source joining with one bounded archive pass that verifies the
  pinned tarball, binds every record to member/index/content identity,
  tokenizes exact released story text, groups normalized duplicates, recovers
  recipes, and audits exclusions and archive token mass.
- [x] Update partition shards, document indexes, manifests, strict loading, and
  rebuild identity so every source reference points to an archive entity and no
  field presupposes a flat text-corpus offset.
- [x] Rewrite normalization/source/duplicate/property tests and the CPU 3x3
  end-to-end partition fixture around archive-only entities. It preserves
  worker-count/order independence, exact reconstruction, leakage rejection,
  control non-reuse, tamper rejection, and old-artifact rejection.
- [x] Restore memory-mapped batching, token-weighted scratch training,
  immutable resume, validation, calibration, sealed-test evaluation,
  publication, and the fixed archive-only GPU runner. CPU tests cover schedule
  and calibration boundaries, the 6x6 94/3/3 and 10x10 96/2/2 fallbacks,
  interrupted/resumed parity, checkpoint identity, and best eligible epoch.
- [x] Run the focused TinyWorlds-P and shared GPT-Neo/checkpoint suite. The
  final 82-test archive/core/partition/training/GPT-Neo/checkpoint/training-state
  scope passes as four concurrent jobs in 9.8 seconds; parked TinyWorlds-v2
  tests remain collection-skipped. The same focused scope also passed before
  the GPU run.
- [x] Build the real archive-only 8x8 partition, rebuild it byte-identically,
  strictly reload it, and audit cell balance, component visibility, split
  marginals, controls, and sealed-test isolation. The canonical partition is
  `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`:
  4,966,067 eligible records and 945,499,161 active tokens. Its independent
  24-worker rebuild used a different run size and produced the identical strict
  tree in 39m08s including both loads. This acceptance gate is opt-in and the
  normal 81-test scope runs as four concurrent jobs in about 10 seconds.
- [x] Run the opt-in RTX 4090 compile/resume smoke against the strict real
  partition. Interrupted update one resumed through update two, and JAX's
  allocator peak was 8.695 GiB against the fixed 12 GiB gate. Three independent
  semantic validators now authenticate assignment topology, archive provenance,
  and shard/index reconstruction concurrently.
- [x] Start a fresh seed-zero RTX 4090 calibration from the archive-only
  partition. The 8x8 attempt completed at 18,832 updates per epoch without
  inspecting or resuming the superseded intersection run. Its epoch-two
  held-in NLL was 1.201706 and mean gap was 0.008017, which required the one
  declared fresh 6x6 fallback.
- [x] Apply the complete one-fallback policy. The fresh 6x6 partition is
  `7bf90c70ca7207d8b0fdd7896eed7a2ae019bbcbd74126cfcc2115ae0759b4fb`
  with 94/3/3 held-in splits. Its scratch attempt completed at 17,200 updates
  per epoch; epoch-two held-in NLL was 1.206720, allocator peak was 8.772 GiB,
  and mean gap was 0.002802. The second low-gap result ended the milestone as
  specified. Epochs three through five, checkpoint selection, sealed test, and
  base publication were correctly not run. See the current
  [archive-only calibration report](TW-P_ARCHIVE_CALIBRATION_REPORT.md) for
  exact identities, per-world metrics, and evidence hashes.

Continual LoRA/VAMP streams, replay, consolidation, regeneration, semantic
clustering, and near-deduplication are deliberately outside this milestone.
