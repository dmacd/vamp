# TinyWorlds-P Semantic-v1 Execution Report

Date: 2026-07-22

Benchmark: `tinyworlds-p-semantic-v1`

Result: **automated semantic-construction stop; no catalog or downstream model artifact**

## Outcome

The semantic successor is implemented, and its pinned real-source construction
run completed. The run produced authenticated MiniLM evidence, then stopped at
the first frozen catalog invariant: only six nouns survived the strictly
positive 10th-percentile target-role margin, so eight noun clusters cannot be
initialized. This is a scientific construction result, not a runtime failure.

No threshold was changed, no cluster count was relaxed, and no archive-v1
partition or checkpoint was loaded. There is no semantic-v1 partition, sample
report, training run, selected checkpoint, base publication, or sealed-test
evaluation. The partition and training runners authenticate the failure audit
and exit with controlled status 2 before starting downstream work.

The complete generated audit is available as
[`audit.md`](../data/tinyworlds-p-semantic/catalog/v1/failures/ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a/audit.md)
and self-contained
[`audit.html`](../data/tinyworlds-p-semantic/catalog/v1/failures/ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a/audit.html).

## Frozen identities

The sole story source was the 1,608,001,638-byte
`TinyStories_all_data.tar.gz` archive at dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, SHA-256
`26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
The conditional partition contract binds the existing 50,257-token GPT-2 BPE
tokenizer at TinyStories-8M revision
`8612e3b15c66ffa94eaa6ee0de5c96edd2d630af`; that tokenizer was not used for
semantic encoder inference.

The semantic encoder was
`sentence-transformers/all-MiniLM-L6-v2` at revision
`b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`, with 384-dimensional float32
attention-mask mean pooling and L2 normalization. Its complete identity is
`1101bb824cee453866d6dcd2b489b29ad2c55b20de5bbaceda67f38206a21502`.
All 11 locally selected model/tokenizer files were authenticated:

| File | Bytes | SHA-256 |
|---|---:|---|
| `1_Pooling/config.json` | 190 | `4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23` |
| `config.json` | 612 | `953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41` |
| `config_sentence_transformers.json` | 116 | `061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3` |
| `model.safetensors` | 90,868,376 | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| `modules.json` | 349 | `84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf` |
| `pytorch_model.bin` | 90,888,945 | `c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256` |
| `sentence_bert_config.json` | 53 | `fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7` |
| `special_tokens_map.json` | 112 | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |
| `tokenizer.json` | 466,247 | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |
| `tokenizer_config.json` | 350 | `acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b` |
| `vocab.txt` | 231,508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

## Implementation delivered

The new package is
[`src/apm/data/text/tinyworlds_p_semantic/`](../src/apm/data/text/tinyworlds_p_semantic/).
It separates contracts for encoder evidence, semantic catalogs, partitions,
sample reports, empirical-null evaluation, training/resume, and publication.
The principal boundaries are:

- pinned-snapshot discovery, target-centered context cropping, construction
  group selection, CUDA MiniLM inference, and reusable evidence publication;
- deterministic role-margin and two-sense screens, equal anchor/context word
  vectors, mass-constrained spherical clustering, boundary exclusion passes,
  strict success catalogs, and content-addressed failure audits;
- archive-native non-construction filtering, semantic topology selection,
  80/10/10 world and 96/2/2 held-in allocation, globally unique matched
  controls, persisted one-to-one group pairings, exact-byte shards, and strict
  source/catalog authentication;
- validation-only pre-training sample reports with all cluster inventories and
  exact archive provenance, without access to sealed-test indexes;
- sorted per-group loss ledgers, 10,000 paired bootstrap replicates, 10,000
  within-pair label-swap placebos, Holm correction, and the fixed
  `ln(1.05)` semantic gap gate;
- fresh seed-zero two-epoch calibration, fail-without-regrid behavior,
  five-epoch continuation, eligible-epoch held-in-NLL selection, one-shot
  sealed test, semantic-only resume identity, and complete checkpoint
  publication.

The fixed entry points are:

- [`scripts/prepare_tinyworlds_p_semantic.py`](../scripts/prepare_tinyworlds_p_semantic.py)
- [`scripts/build_tinyworlds_p_semantic_partition.py`](../scripts/build_tinyworlds_p_semantic_partition.py)
- [`scripts/train_tinyworlds_p_semantic_base.py`](../scripts/train_tinyworlds_p_semantic_base.py)

The latter two now report the real construction stop directly. They cannot
mistake the failure audit for a missing preparation step or begin archive/GPU
work.

## Real construction execution

The real preparation used 24 archive/context workers and one NVIDIA RTX 4090.
The isolated construction environment used CUDA PyTorch 2.1.0+cu121,
Transformers 4.46.3, Tokenizers 0.20.3, Safetensors 0.4.5, and
Hugging Face Hub 0.26.5. Repeated deterministic preflights measured roughly
24,800--26,200 texts/second for 1,024-text float32 batches and passed an exact
bit-for-bit replay on the final run. The first full
archive/evidence publication took about 8 minutes 38 seconds from working-
directory creation to evidence publication. The subsequent screen reuses the
441 MiB evidence artifact and takes about eight to nine seconds.

The archive ingest observed:

| Measure | Value |
|---|---:|
| Archive members | 50 |
| Archive records | 4,967,871 |
| Normalized duplicate groups | 4,967,648 |
| Eligible records/groups | 4,966,067 |
| Eligible active tokens | 945,499,161 |
| Token-weighted role-classification coverage | 99.968473% |
| Conflicting groups/records | 6 / 20 |
| Unclassifiable records | 1,574 |

The namespaced modulo-20 rule then reserved 247,629 construction groups with
47,172,075 active tokens. The permanently non-construction remainder contains
898,327,086 active tokens. Construction prepared 195,492 anchor/context texts.

The reusable evidence artifact is:

- directory:
  `data/tinyworlds-p-semantic/evidence/v1/efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434/`
- evidence SHA-256:
  `efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`
- embeddings SHA-256:
  `d02c5e0c0fcc921e6290e47590ef8eae4059b99082ccba0a4a01fe727e8854da`
- context SHA-256:
  `108d5f6ad7bb05c575687a668509d4ee1f77edab249a01b99d3a6c2327ad700f`

## Frozen screen result

All 1,460 role words exceeded the required 32 exact construction contexts;
1,457 reached the 128-context cap and the remaining three had 74--86. The
decisive loss therefore occurred at semantic role separation rather than
context scarcity:

| Role | Input words | Nonpositive role q10 | Multi-sense silhouette | Pre-cluster survivors |
|---|---:|---:|---:|---:|
| Noun | 1,066 | 1,060 | 0 | 6 |
| Verb | 394 | 305 | 4 | 85 |

The six surviving nouns were:

| Noun | Non-construction token mass | Role-margin q10 | Context silhouette |
|---|---:|---:|---:|
| `pirate` | 836,269 | 0.002524 | 0.095442 |
| `present` | 844,918 | 0.008832 | 0.184655 |
| `ship` | 813,357 | 0.006050 | 0.062910 |
| `train` | 848,656 | 0.006329 | 0.102038 |
| `treat` | 870,924 | 0.001541 | 0.098114 |
| `witch` | 881,719 | 0.013312 | 0.113561 |

The fixed algorithm needs at least eight words merely to seed eight centroids,
then requires at least 32 nouns in each cluster. It therefore stopped before
any capacity assignment, boundary exclusion, centroid-pair gate, retained-mass
gate, cell heatmap, or topology selection. Passing the role screen is recorded
as `semantic_grid_failure` in the failure-only word ledger because no final
cluster disposition exists.

The strict failure artifact is:

- directory:
  `data/tinyworlds-p-semantic/catalog/v1/failures/ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a/`
- failure SHA-256:
  `ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a`
- reason: `fewer role words than requested semantic clusters`
- contents: all 1,460 word vectors and dispositions, all role-pair masses,
  representative exact contexts, frozen config/encoder/evidence identities,
  Markdown audit, self-contained HTML audit, and authenticated tree
- size: approximately 24 MiB

## Verification

The semantic scripts and every semantic module compile with `py_compile`.
Seventeen direct semantic/training tests pass. The broader focused shared suite
passes all 110 collected tests across GPT-Neo, LoRA, checkpoint,
training-state, archive ingest, archive partition/training, semantic catalog,
semantic partition, semantic statistics, and semantic resume/evaluation code.
The real-archive integration module remains opt-in and collection-skipped by
default, as required for long-source gates.

The fixtures cover:

- construction-slice determinism, exact whole-word contexts, role and
  multi-sense exclusion, capacity clustering, assigned-cluster boundary
  margins, exhaustive success/failure audits, and tamper rejection;
- a CPU 3-by-3 archive fixture with construction leakage checks, exact source
  and token reconstruction, globally unique paired controls, strict
  archive-v1 rejection, different worker/run-size builds, and complete
  byte-identical trees;
- sorted per-group losses, empirical bootstrap/placebo determinism, Holm
  correction, all fixed calibration gates, validation-only sample-report
  isolation, semantic resume identity, and interrupted/resumed parameter and
  trace parity.

Cached-evidence preparation exits 2 after authenticating and republishing the
same failure identity. Both partition and training runners also exit 2 with
`partition/sample report: not authorized` and
`GPU preflight/training/sealed test: not authorized`, respectively. Filesystem
inspection confirms there is no semantic partition directory and no semantic
checkpoint below the pinned encoder snapshot.

## Version consequence

Semantic-v1 is frozen at this construction result. The failure cannot be
repaired in place by weakening the 10th-percentile role test, changing anchor
templates, using a different encoder, accepting a smaller grid, or manually
labelling clusters. Any such hypothesis is `tinyworlds-p-semantic-v2`; it may
reuse the authenticated encoder evidence where its evidence contract is
identical and compare audits, but it cannot reinterpret this v1 stop.
