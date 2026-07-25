# TinyWorlds-Q Semantic-v1 Engineering Report

## Outcome

The query-native benchmark engine and CPU verification fixtures are complete.
The interactive user approved all 24 primary pilot facts and all 24
fact-specific reverse choices. The official catalog and partition are
published, a fresh archive replay reproduced both byte-for-byte, and the GPU
preflight passed. The real seed-zero base completed, passed every quality and
memory gate, and published a strict selected base. The independent pilot sweep
then completed, but no registered budget passed both learnability conditions
for both worlds. The mandatory pilot gate stopped sequential/VAMP and main
execution. No sealed test was opened. This is an operational stop, not a
scientific VAMP result.

`tinyworlds-q-semantic-v1` is isolated under its own Python package and the
registered roots `data/tinyworlds-q-semantic/`,
`checkpoints/tinyworlds-q-semantic-v1/`, and
`results/language_cl/tinyworlds-q-semantic-v1/`. Semantic-v6 remains unchanged.

## Implemented boundaries

- The pilot and main concept manifests use the exact registered surface forms.
- Construction selection is a namespaced SHA-256 bucket-zero-modulo-twenty
  decision over complete duplicate groups.
- Automated extraction proposes ranked exact sentence-level predicate n-grams
  with complete story provenance. It cannot create semantic authority.
- Catalog construction requires twelve accepted facts per world, four relation
  categories, sixteen construction groups per fact, complete affirmative human
  review, three validation and five test paraphrases, balanced answer positions,
  three false distractors, and equal answer-token lengths.
- Validation and sealed-test templates are physically separate authenticated
  files. Test loading requires a durable frozen transaction; an interrupted
  transaction may resume, and a completed transaction cannot reopen.
- Partitioning withholds registered facts rather than all concept mentions,
  excludes construction and multi-concept fact-bearing stories, counts evidence
  only within one sentence, and enforces 32 node-training groups per fact plus
  256 ordinary lexical base groups per concept.
- Exact archive story bytes, token sequences, duplicate groups, assignment
  reasons, actual tokenizer special IDs, and provenance are streamable and
  authenticated. Long archive/review/partition passes emit phase progress and
  retain their reported temporary work locations.
- Memory-mapped indexed batches feed a fresh seed-zero GPT-Neo trainer. Its
  complete AdamW/random/cursor state resumes exactly, and only a finished
  two-epoch run that passes held-in NLL and allocator gates can publish a
  query-v1 selected base. That base is partition-bound rather than
  active-prefix-bound, so a large catalog can reuse it at 5, 10, 20, and 100
  worlds.
- Validation-only question prefixes prepare the fixed parent and router probes.
  The staged runner trains real rank-eight tensors for independent, sequential,
  and VAMP systems and persists graph, address, trace, RNG, and tensor state at
  every completed world.
- The real pilot runner snapshots one 2,000-update independent trajectory per
  world at all three registered budgets, reconstructs its declared accuracy
  from exact validation rows, and selects only from matching independent
  adapters. It then trains sequential and VAMP only at that selected budget,
  verifies a strict no-op resume, and evaluates specificity, VAMP oracle, and
  all task-free routers without deserializing test prompts. The worst case is
  the preflighted 12,000 adapter updates.
- Reviewed prompts compile into the existing answer-only four-candidate scorer,
  preserving hard-node, routing, VAMP, support, and regret behavior.
- Statistical units are facts. The production default is 10,000 deterministic
  fact-resampled replicates with equal world weights.
- Manifest-driven capacity and schedules cover one through one hundred worlds;
  full evaluation stops above twenty and milestone evaluation retains every
  acquisition plus complete final coverage.
- Stage artifacts, result ledgers, reports, and sealed transactions publish
  atomically. Final reports enforce the exact dynamic evaluation schedule and
  bind the complete result-ledger digest. Preflight estimates training, parent
  search, routing, result size, and memory from the active manifest.

## Verification

The new CPU fixtures pass seven tests. They cover construction and
leakage rules, candidate/token boundaries, exact rebuild and tamper rejection,
catalog parent nesting, sealed access, deterministic statistics, dynamic
capacity/schedules, bounded scoring, strict report coverage, and resumable
stage identity. The pilot-specific fixture also verifies the 24-row primary
review queue, 8-row backup set, evidence support, equal answer-token lengths,
compact publication, strict primary/reverse-approval reload, and approved
catalog compilation. The fixtures run a tiny GPT-Neo base both uninterrupted and
through an interruption and verify identical parameters and progress bytes.
A focused pilot-publication path additionally checks prefix-only router
stacking, all-budget accuracy reconstruction, dynamic validation schedule
coverage, canonical ledgers, exact safetensors sweep reload, and result-file
tamper rejection. It now also covers authenticated all-budget failure
publication and failure-ledger tamper rejection.
A focused compatibility run passes 36 existing tests across knowledge tasks,
candidate scoring, knowledge training/evaluation, semantic statistics,
semantic-v6 partitioning, and semantic-v6 VAMP.

The repository-wide default suite reports 618 passed, 274 skipped, 11
deselected, and one unrelated pre-existing failure. The failure is the strict
zero-atol assertion in
`test_every_router_returns_valid_task_free_decisions_and_suffix_metrics`, where
two float32 probability rows sum to `1.0000001`. That test does not import the
query-v1 package; shared routing behavior was left unchanged to preserve the
semantic-v6 evidence boundary.

The tests run in `ve-semantic`; the system Python lacks the repository's optional
`optax` dependency and therefore cannot collect the broader LM modules.

## Real pilot review packet

A fresh 24-worker replay of the pinned archive completed in 8.3 minutes and
published review packet
`5b01c86812593681133b46effd786d5647dcb3e8cf0308e8482bb54f01b7775b`.
The archive integrity and role gate passed for all 4,967,871 records. The
query-native pass scanned 4,967,647 nonempty duplicate groups and selected
248,051 namespaced construction groups. Strict reload passed.

The packet contains 400 ranked candidates: 200 for rabbit and 200 for horse.
Every candidate has at least sixteen distinct construction groups, so human
review has a support-eligible pool for both concepts. The ranking deliberately
contains unjudged narrative co-occurrences as well as plausible relations;
high support does not make a candidate true. The complete JSON, Markdown, and
standalone HTML with exact sentence provenance are in the
[`review packet`](../data/tinyworlds-q-semantic/review/5b01c86812593681133b46effd786d5647dcb3e8cf0308e8482bb54f01b7775b/review.md).
The 14 GiB streamed replay workspace is retained under
`data/tinyworlds-q-semantic/work/pilot-review-primary`.

The 400-candidate file is retained only as the raw discovery audit. It is not a
document a reviewer is expected to read sequentially. A second targeted pass
over the retained duplicate-group index published evidence packet
`1603f089988125c2a0782d5bb41ebb0ce113ec466ed6248b14ad4a8e0040d071`
for the 29 exact predicates used by a reviewer-ready shortlist. Compact
shortlist `ad00bafb6bc5adef50a76f2b1ff7230bce02e46b04526d7bf81753a01dc5dd65`
contains twelve primary proposals and four backups for each pilot concept. Its
smallest rabbit and horse proposal supports are respectively 27 and 18
construction groups. Every proposed answer set has equal GPT-2 suffix length.
The concise 66-line decision sheet is the intended human surface:
[`pilot approval sheet`](../data/tinyworlds-q-semantic/review-shortlists/ad00bafb6bc5adef50a76f2b1ff7230bce02e46b04526d7bf81753a01dc5dd65/review.md).
The same directory contains detailed evidence, exact token IDs, standalone
HTML, canonical JSON, and an editable TSV form.

The user instruction `Approve all primaries` was recorded at
`2026-07-25T04:30:07Z` as approval artifact
`fbe0db124a77ce0215b2632d12cc97320e7eeda60de77b3fe8d48384eaef539b`.
It binds compact shortlist
`ad00bafb6bc5adef50a76f2b1ff7230bce02e46b04526d7bf81753a01dc5dd65`
and records all five fact gates as affirmative for every primary proposal. No
backup proposal was promoted.

## Approved catalog, partition, and preflight

The user instruction `Approve all reverse choices` was recorded at
`2026-07-25T04:40:14Z` as approval
`bc184647bfec6f33c04a0e527d1c70e4c1415555695fedbf5d09d4066a41bbb8`.
It binds the corrected fact-specific reverse sheet and records all falsity,
grammatical-type, and tokenizer-length gates as affirmative.

Official pilot catalog
`5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f`
contains exactly 12 facts per world, 72 validation templates, and 120 sealed
test templates. An independent catalog publication has the same recursive tree
hash. Normal loading authenticated but did not deserialize the sealed file.

Partition
`419e6c8b6362add9af081885066559cc34b18f5c7044894f343c7caf0091ad0c`
retains 4,502,964 base groups and 857,498,081 base tokens. Rabbit and horse have
171,823 and 22,227 node-training groups. The weakest individual fact still has
320 authoritative training groups, and ordinary lexical exposure is 11,344
rabbit groups and 3,859 horse groups. All 248,051 construction groups and 826
multi-concept fact-bearing groups remain outside model inputs. A fresh replay
using different sort-run boundaries reproduced all files; both complete trees
hash to `7b8c50a68cfcde41dc1579836ab7bb431fd85a4652c0fd036ab8986adae87f9f`.

GPU preflight
`6519ee1a5820a039c7b3f8e016b149fd7a90bb23fd5c0cb468a430cd6ed84eb8`
ran on an RTX 4090 with JAX 0.6.2. Its two disposable update losses were
`10.859072643` and `10.853386662`; warm update time was `0.488339` seconds,
warm validation-batch time was `0.020196` seconds, and allocator peak was
7,417,784,832 bytes. The projected result ledger is 1,658,880 bytes. All frozen
limits passed, and the disposable state is not reusable.

## Completed base gate

The fresh two-epoch base completed at
`checkpoints/tinyworlds-q-semantic-v1/work/pilot-base-6fbf5f5e5a7ab4cd3c862884a8b64f08e931d4fe209d57376ebda10c9c5f4bac`.
Authenticated held-in validation covers 17,043,802 active tokens. NLL improved
from `1.231696441` after epoch one to `1.157588485` after epoch two, a
`0.074107956` improvement. Allocator peak was 7,557,684,224 bytes. The run was
finite, stayed below 12 GiB, and passed every registered base gate. Selected
base `91b1dd7cf314fcdf81509d6421a3a33621f7106a54161d0aa080911dc1db4961`
is published with the complete training trace and source bindings.

The completed base authorized the fixed 500/1,000/2,000-update pilot selection.

## Pilot learnability stop

The runner trained one deterministic 2,000-update independent trajectory for
each pilot world and persisted exact snapshots at all three registered budgets.
Validation produced:

| Updates | Rabbit base | Rabbit adapter | Rabbit gain | Horse base | Horse adapter | Horse gain | Passed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 500 | 0.555556 | 0.611111 | +0.055556 | 0.250000 | 0.583333 | +0.333333 | False |
| 1,000 | 0.555556 | 0.611111 | +0.055556 | 0.250000 | 0.555556 | +0.305556 | False |
| 2,000 | 0.555556 | 0.638889 | +0.083333 | 0.250000 | 0.611111 | +0.361111 | False |

Horse passed both conditions only at 2,000 updates. Rabbit exceeded 60%
accuracy at every budget but never reached the required 15-percentage-point
gain over its comparatively strong base. Therefore no budget passed for both
worlds. Failure artifact
`aad4811425c10b0faf5f6f452067e35a58d6cee397970711951e50bfad2247f5`
binds the source identities, selected base, preflight, sweep tensors, exact
validation JSONL ledgers, report, and allocator evidence. The sealed test was
not opened. Selected-budget sequential/VAMP and the main catalog remain
unauthorized under the frozen plan.
