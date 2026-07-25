# TinyWorlds-Q Semantic-v1 Engineering Report

## Outcome

The query-native benchmark engine and CPU verification fixtures are complete.
No real semantic fact was approved, no GPU work was started, and no sealed test
was opened. This is an implementation milestone, not a scientific result.

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

The new CPU fixtures pass six tests. They cover construction and
leakage rules, candidate/token boundaries, exact rebuild and tamper rejection,
catalog parent nesting, sealed access, deterministic statistics, dynamic
capacity/schedules, bounded scoring, strict report coverage, and resumable
stage identity. The pilot-specific fixture also verifies the 24-row primary
review queue, 8-row backup set, evidence support, equal answer-token lengths,
and compact publication. The fixtures run a tiny GPT-Neo base both uninterrupted
and through an interruption and verify identical parameters and progress bytes.
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

## Open gate

Human review is intentionally not automated. A reviewer can approve all 24
primary rows or name rejections, edits, and promoted backups from the compact
sheet. Those decisions must then be recorded in the full semantic ledger.
Until that happens, publishing a pilot catalog or starting a GPU optimizer
would violate the benchmark design.

After review, the required sequence is an independent byte rebuild of the pilot
partition, GPU resource preflight, fresh seed-zero base training, 500/1,000/2,000
update pilot selection, and the two-world sequential/VAMP persistence exercise.
Only a passing pilot freezes and opens the five-world main construction.
