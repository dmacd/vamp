# Development Plan

## Current Outcome

The generic `tinyworlds-q-semantic-v1` query benchmark is implemented and its
CPU gates pass. It is a separate package and artifact family; no semantic-v6
file, checkpoint, result, or compatibility path changed. Query-v1 measures
reviewed four-choice fact knowledge directly rather than using whole-story loss.

Implemented surfaces now include immutable concept/fact/template/catalog,
partition, experiment, and result contracts; fixed rabbit/horse and five-world
manifests; namespaced 5% construction selection; exact same-sentence predicate
discovery and provenance review packets; mandatory human review gates; sealed
catalog publication; fact-withholding partitioning with exact story/token
ledgers; memory-mapped indexed batching; exact scratch-base training/resume;
strict accepted-base publication; validation-only parent/router probe
preparation; real independent, sequential, and VAMP tensor stages; compilation
into the shared `KnowledgeQuery` scorer; deterministic fact-level bootstraps;
exact-trigger generation inspection; dynamic 1--100 world capacities and
schedules; bounded scoring; atomic JSONL ledgers; resource preflight; one-time
sealed transactions; and schedule-complete descriptive Markdown/standalone-HTML
reporting. Experiment identities include the complete GPT-Neo architecture,
all six LoRA target switches, and the derived adapter optimizer contract.
Selected bases bind the full catalog partition and base-training contract
rather than an active adapter prefix, allowing one large-catalog base to serve
the registered nested prefixes.

The CPU fixtures cover construction exclusion, story-level fact
leakage, same-sentence evidence accounting, multi-concept exclusion, exact
candidate balance and tokenizer boundaries, byte-identical partition rebuild,
catalog/partition/tensor tampering, parent-prefix preservation, sealed-query
rejection and completion, fact-level statistics, a real tiny GPT-Neo
uninterrupted-versus-resumed parameter/trace parity check, and resumable
independent, sequential, and VAMP stage identities. The pilot-specific fixture
also covers the compact 24-primary/8-backup review queue, evidence support,
token balance, and publication. Synthetic manifests at 1,
5, 10, 20, and 100 worlds cover derived capacities, tensor masks, full and
milestone schedules, chunking, dynamic reports, and explicit preflight limits.
The focused TinyWorlds-Q suite passes all 11 tests. A broader 80-test query,
knowledge, scoring, training, statistics, routing, artifact, and semantic-v6
compatibility run also passes in the pinned semantic environment. The complete
default non-opt-in collection passes with 626 passed, 274 registered resource
skips, and 11 marker-deselected tests.

The real pilot proposal packet is now published at
`data/tinyworlds-q-semantic/review/5b01c86812593681133b46effd786d5647dcb3e8cf0308e8482bb54f01b7775b`.
A fresh 24-worker replay authenticated all 4,967,871 archive records, scanned
4,967,647 nonempty duplicate groups, and selected 248,051 construction groups.
The strict-reloaded packet contains 200 ranked candidates for rabbit and 200
for horse; all 400 have at least sixteen supporting construction groups. Its
14 GiB replay workspace remains at
`data/tinyworlds-q-semantic/work/pilot-review-primary`.

The raw 400-candidate packet is now explicitly an audit appendix, not a human
work queue. A targeted replay over the retained index published the 29-predicate
evidence packet
`1603f089988125c2a0782d5bb41ebb0ce113ec466ed6248b14ad4a8e0040d071`
and compact shortlist
`ad00bafb6bc5adef50a76f2b1ff7230bce02e46b04526d7bf81753a01dc5dd65`.
The concise `review.md` is 66 lines: twelve primary proposals per concept,
four backups per concept, one representative sentence and support count per
primary, reviewed-form placeholders, exact trigger closure, and tokenizer-
balanced false choices. Detailed evidence, exact token IDs, HTML, canonical
JSON, and an editable TSV remain alongside it for drill-down.

The interactive user approved all 24 primary proposals at
`2026-07-25T04:30:07Z`. Approval artifact
`fbe0db124a77ce0215b2632d12cc97320e7eeda60de77b3fe8d48384eaef539b`
records affirmative truth, evidence, trigger-closure, answer-form, and forward-
distractor gates for every primary row. No backup was promoted.

The interactive user approved all 24 fact-specific reverse choices at
`2026-07-25T04:40:14Z`. Reverse approval
`bc184647bfec6f33c04a0e527d1c70e4c1415555695fedbf5d09d4066a41bbb8`
binds corrected review
`32f206833ce828fb954628d9063821c853579b96ffbf567d5dd0a2fc5e0ce9c0`.
Official catalog
`5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f`
contains 24 facts, 72 validation queries, and 120 physically separated sealed
queries. Validation-only strict reload passed; sealed prompts were not
deserialized.

Pilot partition
`419e6c8b6362add9af081885066559cc34b18f5c7044894f343c7caf0091ad0c`
passed every construction, leakage, fact-support, and lexical-exposure gate.
Its weakest fact has 320 non-construction training groups; rabbit and horse
retain 11,344 and 3,859 ordinary lexical base groups. A fresh 24-worker archive
replay with 37,000-record sort runs reproduced the complete 12 GiB tree
byte-for-byte; both trees hash to
`7b8c50a68cfcde41dc1579836ab7bb431fd85a4652c0fd036ab8986adae87f9f`.

GPU preflight
`6519ee1a5820a039c7b3f8e016b149fd7a90bb23fd5c0cb468a430cd6ed84eb8`
passed on the RTX 4090. Two disposable updates had finite NLLs `10.859072643`
and `10.853386662`; the warm update took `0.488339` seconds and allocator peak
was 7,417,784,832 bytes, below 12 GiB. It projects 18,530 updates per epoch and
about 5:01:38 for the registered two-epoch base. The real seed-zero run
completed under
`checkpoints/tinyworlds-q-semantic-v1/work/pilot-base-6fbf5f5e5a7ab4cd3c862884a8b64f08e931d4fe209d57376ebda10c9c5f4bac`.
Held-in validation NLL improved from `1.231696441` after epoch one to
`1.157588485` after epoch two, a `0.074107956` improvement. The allocator peak
was 7,557,684,224 bytes. All base gates passed, and selected base
`91b1dd7cf314fcdf81509d6421a3a33621f7106a54161d0aa080911dc1db4961`
is published. That base authorized the fixed pilot sweep.

The registered independent sweep completed one exact 2,000-update trajectory
per world and preserved its 500/1,000/2,000 snapshots. Under the original
preregistered policy, no budget passed both the 60% accuracy and 15-point
acquisition gates for both worlds. Authenticated failure
`aad4811425c10b0faf5f6f452067e35a58d6cee397970711951e50bfad2247f5`
remains immutable evidence of that stop.

After reviewing the ceiling-sensitive rabbit baseline, the interactive user
authorized explicit amendment
`2855b647928700a119ea6e95365379719ad733d45c6ede20cafcd1593a64458c`.
It keeps 60% absolute validation accuracy as the learnability gate and makes
acquisition a mandatory descriptive statistic. It neither rewrites the
failure nor opens sealed test data. Under the amended policy, 2,000 updates is
the first budget where both worlds pass: rabbit reaches `0.638889` from a
`0.555556` base (`+0.083333`), and horse reaches `0.611111` from `0.250000`
(`+0.361111`).

The selected two-world independent, sequential, and VAMP exercise completed,
all nine validation methods were written to the bounded ledger, and an exact
completed-stage no-op resume passed. Pilot result
`55c97f2a649ea434f79e729b2eaff01753a254ce0a5c26e53a1095d4df0364c7`
binds policy, amendment, tensors, ledgers, runtime, and memory. It is an
operational pilot only and gives VAMP no scientific verdict. The sealed test
remains closed.

The five-world main configuration is now frozen at
`82d0d3258e0e723588d151387c0151156b408770df1f84bcb5450ac72f9327ff`.
It fixes order `cat, dog, bird, robot, dragon`, a fresh seed-zero base, 2,000
adapter updates, the exact query/scoring protocol, all nine methods, 10,000
fact-resampled bootstrap replicates, validation-only routing, and a single
sealed opening after all artifacts are frozen.

Main construction review is complete. Raw discovery packet
`7164cd2cc18be5ba29d7106a44f23dbec5bf39a9a962b9c441ccf07501a8132f`
and targeted evidence packet
`ce1b06c7f7a325cedded9970ac008329c93d97d29c84344b93d22b450db14374`
remain non-authoritative audits. Compact shortlist
`fe2f78e92e1c4e0d26280f2741beea728ea3125c932c3126b770da6cd90104cc`
contains 12 primaries and 4 backups per world, exact token-balanced choices,
reviewed trigger proposals, and construction support; the smallest primary
support is 17 groups. The interactive user approved all 60 primaries at
`2026-07-25T22:38:51Z`; approval
`8b0f2868b216b837f2b2c90c0f7faaa141874fe87b2387c6fecd62faed8f616b`
records all five affirmative gates and promotes no backup. Fact-specific
reverse review
`c805da6c075920f85a58b0c4ed25ee4aa6dac2e5763e2578648efd0c0800e1f0`
contains 60 one-token four-choice rows. The interactive user approved every
row at `2026-07-25T22:54:49Z`; reverse approval
`c643731930ae9721ea4c4420f14a830c04ca8179bee8caccb8a73756ec0c1067`
strict-loads with the complete primary authority.

Official catalog
`0ffd78e81d1da4a4fbd20b49bc02f3dec94560085f4490a357c7f73239f9e8ba`
contains 60 facts, 180 validation queries, and 300 physically separated sealed
queries. Its independent catalog trees are byte-identical at
`d6c13a83bf1c614115b2a246bf93b33cd12d9e6ab1b9730a4a43f0ba19cef75f`;
the pilot catalog hash is unchanged. Main partition
`d8536d0295af4fa56174369430b2e615008e28fb239d7d66a428b36988fa7d6b`
passed every primary construction, leakage, support, and lexical gate plus a
strict full-tree reload. It retains 3,509,177 base groups and 669,256,202 base
tokens. Its weakest fact has 248 training groups; robot is the tightest
ordinary lexical exposure at 372 base groups. A fresh 24-worker archive replay
with 37,000-record sort runs independently reproduced the complete partition;
the two trees are byte-identical at
`566700c59c9c05e87525806a2fd54ff48d283b57b4212884153a6808b12a9828`.
Validation-only sample report
`a677d66b572610229a52d4d46b20b30d206f665afeb1c8fc3a82fd5e6c170143`
strict-loads with six exact stories and all 180 validation queries. The 11-test
query suite, broader 80-test compatibility selection, and complete default
collection pass; the latter reports 626 passed, 274 resource-gated skips, and
11 marker deselections. These sources form the main-partition checkpoint;
GPU preflight
`28380737a808e4288c9b8b51cd6a97e9c64c60e23a59b51e10fd2ea565e14641`
also passes. It measured an 8.412 GiB allocator peak and projects 9.040 GiB,
both below 12 GiB; a warm update took 0.418 seconds and projects the two-epoch
base at 3:21:12. The first attempt correctly stopped before an update because
`ve-semantic` has CPU-only JAX; the accepted run used the existing CUDA JAX
0.6.2 `ve` environment on the RTX 4090. Shared pilot/main base orchestration
and the registered main launcher are implemented and focused tests pass. The
fresh resumable base is actively training under identity
`001e16d8908ae593ffc23b423a1a672e005c3cf7b35dacbb09636d1807a96d93`;
complete 1,000-update checkpoints are being persisted and the run remains
finite with no memory stop. The exact main adapter/validation runner and a
separately guarded sealed runner are implemented. Final-analysis protocol
`489042464fd4243e1780d585c4ba7ed6cd1134c9f7a5bf3d7e6f2fb4aaa8712a`
fixes all nine methods, acquisition and retention for every non-base method,
independent cross-world specificity, accuracy and margin effects, router node
accuracy/regret, 10,000 fact bootstraps, and 96-token greedy generation from
the matching final independent adapter. Headline condition summaries use only
the primary matching adapter/path rows; the forced cross-world independent
matrix enters the specificity effect and ledger instead of being pooled into
independent accuracy. The protocol enters the validation freeze before test
access.

Resource accounting now explicitly includes every forced independent-adapter
specificity cell. The five-world sealed ledger is 9,900 rows and a conservative
10,137,600 bytes, not the accepted v1 preflight's 8,100-primary-row / 8,294,400-
byte approximation. The immutable preflight still authenticates; its measured
training/memory evidence is unchanged, and the corrected projection is far
below the frozen 4 GiB limit. New preflights use the complete count. The final
report and validation loaders now reconstruct every canonical result row,
recheck schedule/routing coverage, and recompute registered fact effects and
derived final renderings during recovery; byte hashes alone are not treated as
semantic parity. The sealed test remains closed. See
[`docs/TINYWORLDS_Q_SEMANTIC_V1_EXECUTION_REPORT.md`](docs/TINYWORLDS_Q_SEMANTIC_V1_EXECUTION_REPORT.md).

Semantic-v6 remains immutable negative evidence. Its seed-zero calibration
stopped at `semantic_grid_failure`; its sealed test was never opened, and it
must not be rerun or reinterpreted as query-v1 evidence.

## Completed Milestone: Semantic-v6 Base Gate Stop; VAMP Not Opened

- **Strict v6 base machinery is complete.** Version-native training, resume,
  per-group evaluation, empirical-null validation, selected-base publication,
  and strict loaders reject archive-v1 and semantic-v1 identities. The two-
  epoch gate remained separate from test access and correctly prevented the
  epochs-three-through-five continuation. No epoch satisfied the unchanged
  semantic gap gate, so no selected checkpoint was published.
- **The VAMP study is frozen.** Experiment config
  `ca16318486600745e8a49903f495819741082f120fa7b95b3f9277efa83ada73`
  fixes A-to-E order, three rank-eight adapter systems, 2,000 updates per
  system and world, validation-only parent/key probes, the four stored methods,
  five task-free routers, all prefix/cue conditions, timing, memory, and the
  diagnostic paired-control specificity audit.
- **Resume and sealed boundaries are implemented.** Base checkpoints retain
  optimizer, random, cursor, and schedule state every 1,000 updates and at
  epoch boundaries. The runner resumes the newest strict checkpoint and trims
  only the later, uncheckpointed loss-log tail. Adapter publications persist
  all three random streams and one immutable stage artifact per completed
  world. Evaluation ledgers are atomic, while incomplete validation or sealed
  attempts are preserved under recovery directories. The final evaluator
  writes a durable binding only after the base and all adapters are frozen;
  test indexes cannot be counted or read before that transaction.
- **Reporting is implemented.** Sequential progress is written live under the
  printed temporary artifact directory. The final content-addressed bundle
  includes Markdown and standalone HTML reports, canonical JSON/JSONL,
  per-group base and forced-adapter ledgers, exact test provenance, the full
  nine-method matrix, forgetting, transfer, routing cost, memory, and 10,000-
  replicate specificity intervals. The VAMP result is explicitly exploratory
  and has no new pass/fail threshold. Adapter and result artifacts persist and
  enforce the 12 GiB allocator ceiling.
- **The focused implementation gate passes.** All 61 pinned-environment tests
  pass across strict partitioning, empirical statistics, training and resume,
  adaptation persistence, nine-method evaluation, routing/memory accounting,
  sealed authorization, and paired specificity.
- **The real validation-only anchor audit passed.** The canonical root and all
  five worlds each supply 128 deterministic full-length validation spans. All
  768 sequence hashes are unique, and no test index was read.
- **The disposable GPU preflight passed.** Preflight
  `b7f49909368685a5494a3033e0df7df69cf2e8c1064092c541013b873671988d`
  completed exactly two isolated updates with losses `10.8570` and `10.8511`,
  measured `0.467411` seconds per warm update and `0.015083` seconds per warm
  validation batch, and peaked at 9,030,551,296 bytes. Its checkpoints use a
  separate non-reusable identity. The sealed test remained closed.
- **The registered calibration stop is complete.** The real run published two
  validation ledgers and the immutable `semantic_grid_failure` decision. It
  preserved strict base states but published no selected checkpoint, adapter
  tensor, VAMP result, or sealed-test report. See
  [`docs/TW-P_SEMANTIC_V6_BASE_VAMP_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V6_BASE_VAMP_EXECUTION_REPORT.md).

## Completed Milestone: TinyWorlds-P Semantic-v6 Exact Comparison Feasibility

- **The intervention is preregistered (2026-07-23).** `DESIGN.md` binds the
  semantic-v4 catalog and semantic-v5 failure identities. All 22
  balance-eligible layouts receive the real split allocator and complete
  ten-control construction before the unchanged semantic ranking is applied.
- **The separate implementation is complete.** Version-specific contracts,
  full-candidate feasibility evidence, strict success and failure loading,
  publication, validation-only sample reporting, focused tests, and a fixed
  primary/rebuild runner are implemented. Shared archive code exposes the
  existing exact split/control preparation step without changing its normal
  publication behavior or providing a compatibility alias. The final focused
  CPU suite passes 57 tests, and the new modules compile in the pinned
  environment.
- **The real feasibility screen succeeded.** Both archive runs retained
  2,520,317 unique stories and 479,183,203 scored tokens and reproduced all
  28,224 parent topology measurements. Seventeen of the 22 balanced layouts
  completed all validation/test comparisons. Ranks 0, 8, 9, 14, and 21 failed
  world B's validation column comparison because too few distinct stories
  remained after the fixed split and global non-reuse rules.
- **Semantic rank 1 is the registered winner.** Its A-through-E cells are
  `(2,4), (7,4), (7,6), (2,6), (3,2)`, with scored-token masses `6,136,097`,
  `5,873,159`, `5,921,676`, `6,114,634`, and `5,440,146`. Every mass lies
  within 10% of the median. The final allocation has complete comparisons and
  31,117 deterministic one-to-one pairings.
- **The partition is strict and independently reproduced.** Partition
  `3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`
  was built with 50,000-record sort batches and rebuilt from the archive with
  37,000-record batches. Both complete strict reloads passed. A direct recursive
  comparison found no difference across their 167 files, and both trees hash to
  `b5ba1ce33d1cad7eb00bba0b6eec35e2b94c3a6b997a20149081cc61c862279d`.
- **The pre-training sample report is published.** Report
  `b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579`
  covers the held-in validation set, all five worlds, and both comparison arms
  for every world. It includes cluster inventories and exact archive
  provenance, parses as self-contained HTML, and records that the sealed test
  was not opened. See
  [`docs/TW-P_SEMANTIC_V6_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V6_EXECUTION_REPORT.md)
  and the generated
  [`sample report`](data/tinyworlds-p-semantic/sample-reports/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa/b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579/sample-report.md).
- **No downstream boundary was crossed.** Semantic-v6 has no GPU preflight,
  optimizer update, checkpoint, model-loss result, semantic-gap decision, or
  sealed-test evaluation. The successful construction does not by itself
  authorize training.

## Completed Milestone: TinyWorlds-P Semantic-v5 Control Stop

Semantic-v5 makes one change to the completed v4 partition attempt. It treats
the unchanged 10% cell-mass rule as an eligibility requirement before semantic
ranking. The exact v4 catalog and partition failure are immutable parent
evidence.

### Semantic-v5 Status

- **The intervention is preregistered (2026-07-23).** `DESIGN.md` binds the v4
  catalog and failure identities and states the new selection order. V5 keeps
  every word, cluster, source, threshold, nuisance measure, split, control,
  pairing, and sealed-test rule unchanged. It cannot use model loss or promote
  a diagnostic candidate under the v4 name.
- **The separate v5 implementation is complete.** Dedicated contracts,
  balance-first selection, strict parent/source/settings checks, partition and
  sample-report formats, structured control-failure evidence, a fixed runner,
  and focused tests are implemented. The builder independently reproduces all
  parent topology records before it can select a v5 layout. No v4 artifact is
  loaded through a compatibility alias.
- **The balance-first intervention worked.** The real archive replay retained
  2,520,317 groups and 479,183,203 active tokens and reproduced all 28,224 v4
  topology measurements. Twenty-two layouts were balance-eligible. V5 selected
  `(3,4), (4,4), (4,6), (3,6), (2,0)` with masses `9,899,869`, `8,829,612`,
  `8,742,369`, `10,104,204`, and `9,357,468`; all five pass the unchanged 10%
  rule.
- **Exact control allocation stopped the partition.** World B's validation
  column arm required 2,314 distinct groups, but only 1,511 remained after the
  fixed split and global no-reuse rules. The shortage occurs before fine
  nuisance or token matching. V5 therefore did not choose another balanced
  layout or loosen a tolerance.
- **The stop is strict and independently reproduced.** Failure
  `090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`
  embeds the exact v4 catalog and parent failure and binds the complete
  assignment-ledger SHA-256. A fresh 37,000-record-run rebuild reproduced the
  50,000-record-run assignment ledger and all 54 MB of failure evidence byte
  for byte. Both strict loaders pass, a recursive comparison finds no
  difference, and the final focused CPU suite passes 51 tests. See
  [`docs/TW-P_SEMANTIC_V5_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V5_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11/audit.md).
- **No downstream boundary was crossed.** V5 has no success partition, sample
  report, GPU preflight, calibration, checkpoint, or sealed-test result. A
  different candidate, exact split-level control-feasibility prefilter, changed
  split size, altered matching design, or control reuse belongs to a later
  version and cannot reinterpret this result.

## Completed Milestone: TinyWorlds-P Semantic-v4 Partition Stop

`tinyworlds-p-semantic-v4` tests the single fixed-reference intervention
motivated by the v3 deletion/reseeding cascade. The scientific contract is
preregistered in `DESIGN.md` before the real construction is run.

### Semantic-v4 Status

- **Contract frozen and implemented (2026-07-22).** V4 binds the
  canonical v3 failure
  `ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`
  and must exactly replay v3's pass-zero, unweighted eight-cluster fit in its
  original hash namespace. It then applies the unchanged `0.03` margin once
  against those frozen centroids. It never deletes and reclusters, moves a
  survivor, or updates a centroid after the boundary screen. Separate v4
  config, catalog, failure, audit, strict-loader, runner, and test contracts
  implement this rule without compatibility aliases.
- **The real fixed-centroid grid passes.** The one-shot screen excludes the
  exact v3 pass-zero sets: 188 of 978 noun candidates and 81 of 365 verb
  candidates. The eight clusters retain 790 nouns and 284 verbs; the minimum
  cluster sizes are 39 nouns and 18 verbs. Maximum noun/verb fit-centroid pair
  cosines are `0.8735721184` and `0.8916218581`, and retained joint archive
  token mass is 479,183,203 of 898,327,086 (`53.341729362%`). Catalog
  `ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`
  therefore clears every frozen gate.
- **Strict evidence semantics are verified.** Boundary-excluded candidates
  retain their vector, fit assignment, and measured margin in the v4 ledger;
  retained cluster inventories use the same fit assignments, while published
  centroids remain the authenticated all-candidate fit centroids. The loader
  reconstructs the fit, margins, dispositions, and gates under both pinned and
  current project numeric environments. An independent real rebuild reproduced
  all 11 files byte for byte; the self-contained HTML parses without external
  resources, five catalog fixtures pass, and the catalog-stage focused group
  passed 44 tests before partition work. See
  [`docs/TW-P_SEMANTIC_V4_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V4_EXECUTION_REPORT.md)
  and the generated
  [`catalog audit`](data/tinyworlds-p-semantic/catalog/v4/ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee/audit.md).
- **V4-native partition machinery is implemented and verified.** Separate
  partition/tree/preset/failure/sample-report contracts reject v1--v3
  artifacts. The CPU 3-by-3 fixture covers strict reconstruction,
  construction leakage, global one-to-one pairing, complete validation
  sampling, cross-worker/run-size byte identity, legacy rejection, and shard
  tampering. Topology failures now retain every ranked candidate, exact score
  fractions, source/seed bindings, a strict loader, Markdown/HTML audits, and
  byte-rebuild enforcement; synthetic tests cover repeat publication and
  tamper rejection.
- **The real partition stopped at the frozen topology gate (2026-07-22).** The
  archive replay retained 2,520,317 groups and exactly 479,183,203 tokens. All
  28,224 physical topologies passed nonempty, component-visibility, and
  control-capacity filters. The preregistered semantic-first winner used cells
  `(1,2), (3,2), (3,4), (1,4), (6,1)` and masses `2,559,355`, `5,440,146`,
  `9,899,869`, `4,699,583`, and `1,428,732`, violating the fixed 10%-around-
  median gate. The authenticated failure is
  `37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`;
  an independent replay reproduced its identity and every byte. The complete
  focused archive/semantic/partition/training CPU gate now passes 47 tests.
- **V4 is terminal unless a new version is preregistered.** Twenty-two other
  candidates satisfy the median gate, but selecting one after observing the
  failure would change the objective order. V4 therefore has no success
  partition, split allocation, paired controls, sample report, GPU runtime
  preflight, calibration, checkpoint, or sealed-test result. Any
  balance-feasibility prefilter, reordered objective, changed tolerance, or
  diagnostic-candidate choice belongs to semantic-v5 and cannot reinterpret
  this stop. See the generated
  [`partition failure audit`](data/tinyworlds-p-semantic/v4/failures/37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07/audit.md).

## Completed Milestone: TinyWorlds-P Semantic-v3 Construction Stop

`tinyworlds-p-semantic-v3` isolates the next intervention suggested by the v2
stop: semantic words are assigned only to their nearest spherical centroid,
while token and nuisance balance is deferred to story allocation after the
catalog is fixed.

### Semantic-v3 Status

- **Semantic-first contract implemented and frozen (2026-07-22).** V3 reuses
  the exact v1 MiniLM evidence and binds v2 failure artifact
  `23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
  Every real word's raw role score, fold, reference count, conformal value, and
  cutoff must exactly replay that ledger. V3 removes token-mass capacities and
  their repair from word clustering, uses unweighted farthest-first spherical
  k-means with nearest-centroid assignment, and records token mass only after
  assignment. The eight clusters, `0.03` true-nearest margin, five exclusion
  passes, word-count floors, `0.90` centroid-pair ceiling, and 40% joint
  retained-mass gate remained fixed.
- **Numeric construction provenance is explicit.** A first general-environment
  preflight preserved every decision and the complete boundary trace but
  exposed several-billionth serialization differences from the v2 ledger
  between NumPy 2.5.1 and 1.26.4. Its content-addressed bundle
  `94614921b5386653f92ee8dc372fc45b566502f9706723df57b257ab4a1252f2`
  remains preserved but is noncanonical. Before canonical publication, the v3
  config bound NumPy 1.26.4 and exact v2 score replay. No scientific threshold,
  cluster assignment, pass trace, or stop decision changed.
- **The real semantic-first grid reached a narrower automated stop.** The
  unchanged role/sense screens left 978 noun and 365 verb candidates. Noun
  boundary failures were `188, 17, 6, 1, 12, 1`; verb failures were `81, 7,
  0`. Verbs converged after two reclusters. On noun pass five, `crayon` remained
  below the fixed margin at `0.0296120345`, so v3 stopped. The canonical strict
  failure is
  `ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`.
- **The apparent one-word miss masks a cascade.** A post-stop diagnostic
  removal and sixth recluster exposed 22 new noun boundary words and a 25-word
  noun cluster, below the independent 32-word floor. Coverage would remain
  above 48% and centroid-pair cosines remain below `0.90`; the remaining issue
  is instability from hard deletion plus full reseeding, not mass balance or
  corpus coverage. Waiving the last word or extending v3's pass budget is not
  a valid repair.
- **Strict implementation and verification are complete.** Separate v3
  config/catalog/failure formats, semantic-only clustering, nearest-assignment
  replay, content-addressed Markdown/HTML audits, the fixed cached-evidence
  runner, and synthetic mass-independence/success/failure/rebuild/tamper tests
  are implemented. An independent real rebuild reproduced all nine canonical
  files byte for byte; the cached loader returns the same failure identity.
  Three focused regression groups pass 153 tests, and all semantic modules,
  runners, and tests compile. See
  [`docs/TW-P_SEMANTIC_V3_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V3_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v3/failures/ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146/audit.md).
- **No downstream artifact is authorized.** Semantic-v3 has no catalog,
  partition, sample report, GPU training run, checkpoint, or sealed-test
  result. Any fixed-centroid boundary screen, stable-core construction, robust
  objective, changed threshold, or changed pass budget must be preregistered
  as semantic-v4 rather than used to reinterpret v3.

## Completed Milestone: TinyWorlds-P Semantic-v2 Calibrated Construction Stop

`tinyworlds-p-semantic-v2` is the role-calibrated successor requested after
semantic-v1 exposed a systematic MiniLM anchor offset. It reuses the exact
authenticated semantic-v1 encoder evidence, leaves both archive-v1 and
semantic-v1 immutable, and changes no language-model, partition, semantic
vector, sense, cluster, retained-mass, or evaluation gate.

### Semantic-v2 Status

- **Cross-fitted role calibration implemented and frozen (2026-07-22).** The
  raw statistic remains each word's 10th-percentile
  `context·target-anchor - context·opposite-anchor` margin. Words are assigned
  to five SHA-256 folds in the `tinyworlds-p-semantic-v2` namespace. For each
  declared role and held-out fold, the other four folds form a word-level
  reference distribution; the held-out word receives the finite-sample
  lower-tail conformal value
  `(1 + count(reference <= score)) / (reference_count + 1)`. A word is a role
  outlier only when that value is at most `0.05`. Thus every decision is
  out-of-fold, role-specific, construction-only, and independent of model
  loss, partitions, and sealed test. No calibration-panel words are
  automatically discarded.
- **Separate v2 contracts and strict artifacts implemented.** The versioned
  config records the reused v1 evidence contract, fold namespace/count,
  conformal method and alpha, reference-size floor, all unchanged semantic
  thresholds, and the deterministic single-prior-word repair used only when
  indivisible token masses create a greedy packing dead end. The repair
  preserves descending-mass processing and the exact 90--110% capacity
  bounds. Content-addressed success/failure builders, exhaustive Markdown and
  standalone HTML audits, strict calibration replay, tree authentication,
  tamper rejection, and synthetic byte-rebuild fixtures are implemented.
- **The real role screen now behaves as calibrated.** Reusing encoder evidence
  `efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`,
  the five noun reference sets contain 837--883 words and the verb sets
  294--326. The calibrated lower tail excluded 51 of 1,066 nouns (4.78%) and
  19 of 394 verbs (4.82%). The unchanged two-sense gate excluded another 37
  nouns and 10 verbs, leaving 978 noun and 365 verb candidates for the fixed
  8-by-8 construction.
- **The unchanged boundary gate produced the new automated stop.** Noun
  boundary failures across the initial clustering plus five permitted
  exclusion/recluster passes were `259, 137, 103, 54, 55, 47`; verb failures
  were `106, 39, 23, 21, 18, 18`. Both roles therefore still had words below
  the assigned-cluster margin `0.03` after the pass budget. The authenticated
  failure bundle is
  `23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
  It contains all word scores, fold/reference evidence, conformal values,
  sense metrics, pass-level cluster masses and margin distributions, exact
  story contexts, candidate PCA, and every disposition.
- **No downstream artifact is authorized.** If the 47 noun and 18 verb
  failures observed on the terminal pass were also removed, the diagnostic
  remainder would contain 323 nouns, 140 verbs, and only 98,322,186 joint
  tokens (10.945% of the 898,327,086-token non-construction archive), well
  below the independently frozen 40% retained-mass floor. That counterfactual
  does not extend the pass budget; it shows that more exclusion passes would
  not rescue this version. Semantic-v2 therefore has no catalog, partition,
  sample report, GPU training run, checkpoint, or sealed-test result. Any
  change to the cluster representation, capacity objective, boundary margin,
  pass count, or retained-mass floor is a separately designed semantic-v3,
  not a reinterpretation of v2.
- **Execution and verification are reproducible.** The fixed v2 runner
  authenticates and reuses the 441 MiB evidence cache, completes construction
  in about 16 seconds without GPU inference, and subsequently reloads the same
  failure identity in about four seconds. An independent temporary real-data
  rebuild reproduced the failure SHA and all nine published files byte for
  byte. All 142 focused semantic/archive and shared GPT-Neo/checkpoint tests
  pass. The v2 fixtures cover fold isolation,
  conformal order invariance, content-addressed 3-by-3 success and failure
  fixtures, byte-identical rebuilds, calibration replay, and tamper rejection.
  See
  [`docs/TW-P_SEMANTIC_V2_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_V2_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v2/failures/23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25/audit.md).

## Completed Milestone: TinyWorlds-P Semantic-v1 Construction Stop

`tinyworlds-p-semantic-v1` is implemented as the semantic-conjunction
successor to archive-v1. It preserves the noun-by-verb factorial experiment,
but gives noun and verb groups an independently constructed semantic meaning.
Archive-v1 remains immutable negative evidence; semantic-v1 has separate
evidence, catalog, partition, sample-report, training, evaluation, and
checkpoint contracts and provides no compatibility alias.

### Semantic-v1 Status

- **Contracts and implementation complete (2026-07-22).** The new
  `apm.data.text.tinyworlds_p_semantic` package pins the archive, tokenizer,
  complete MiniLM snapshot, construction/config identity, role anchors,
  context sampling, float32 mean-pooled normalized inference, deterministic
  semantic screens, capacity-constrained spherical clustering, audits,
  semantic topology, exact archive replay, paired controls, strict loading,
  validation-only sample reporting, group-loss ledgers, SHA-seeded paired
  bootstrap/placebo statistics, Holm correction, calibration, resume,
  selection, sealed-test, and publication boundaries. Fixed runners prepare
  evidence, build and independently reproduce a partition, and train a fresh
  seed-zero base only when the preceding artifact gate exists.
- **Pinned encoder evidence published.** The complete 11-file
  `sentence-transformers/all-MiniLM-L6-v2` snapshot at revision
  `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0` has encoder identity
  `1101bb824cee453866d6dcd2b489b29ad2c55b20de5bbaceda67f38206a21502`.
  The real CUDA/24-worker preparation published evidence
  `efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`:
  247,629 construction groups and 47,172,075 construction tokens are
  permanently reserved, 898,327,086 eligible tokens remain outside that
  slice, and 195,492 context/anchor texts were embedded. The evidence cache is
  independent of later clustering thresholds.
- **Frozen eight-cluster screen reached its declared automated stop.** Of
  1,066 nouns, 1,060 failed the strictly positive 10th-percentile role-margin
  rule. Only `pirate`, `present`, `ship`, `train`, `treat`, and `witch`
  survived. Of 394 verbs, 305 failed role margin and four failed the
  multi-sense silhouette gate, leaving 85. Six nouns cannot seed the required
  eight noun clusters and are far short of the later 32-nouns-per-cluster
  requirement, so construction failed before clustering. No threshold was
  relaxed, cluster count changed, word relabelled, or archive-v1 artifact
  consulted.
- **Failure evidence is immutable and exhaustive.** The strict failure bundle
  `ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a`
  contains all 1,460 role words, token masses, exact context counts, measured
  role margins, measured silhouettes where applicable, dispositions,
  representative exact archive contexts, and candidate-vector PCA in
  Markdown and self-contained HTML. See
  [`docs/TW-P_SEMANTIC_EXECUTION_REPORT.md`](docs/TW-P_SEMANTIC_EXECUTION_REPORT.md)
  and the generated
  [`failure audit`](data/tinyworlds-p-semantic/catalog/v1/failures/ba0c6d40f54522ac74e6f4d1813d997c19b5c21d081b038b0c0357f875d01c8a/audit.md).
- **Downstream work is intentionally inapplicable.** Because no valid
  semantic-v1 catalog exists, a partition, byte-identical rebuild, sample
  report, GPU runtime estimate, calibration, checkpoint selection, sealed-test
  opening, and base publication are not authorized. Both downstream runners
  authenticate the failure bundle and exit with controlled status 2 before
  doing work. Any change to anchors, role metric, threshold, vocabulary, or
  cluster count is a new `tinyworlds-p-semantic-v2` experiment and cannot
  reinterpret semantic-v1.
- **Verification remains focused and reproducible.** All 110 collected tests
  in the semantic, archive-native TinyWorlds-P, GPT-Neo/LoRA, checkpoint, and
  training-state scope pass; the long real-source module remains opt-in. CPU fixtures cover
  semantic screen and clustering determinism, construction exclusion,
  content-addressed success/failure audits, archive-v1 rejection, leakage,
  exact-byte reconstruction, paired-control coverage, cross-worker/run-size
  byte identity, tamper rejection, group-loss persistence, empirical-null
  gates, sample-report isolation, and interrupted/resumed training parity.
  Real archive/GPU gates remain opt-in; their measured result is retained and
  is not part of routine tests.

## Completed Milestone: TinyWorlds-P Archive-Only Calibration Stop

The completed roadmap is `tinyworlds-p-archive-v1`, tracked in
[`docs/TW-P_PLAN.md`](docs/TW-P_PLAN.md). It replaces generated TinyWorlds prose
with unmodified stories taken directly from released records in the pinned
`TinyStories_all_data.tar.gz` archive. Five noun-bucket by verb-bucket cells are
withheld from a freshly initialized eight-layer GPT-Neo base; only story text
reaches the model. All base, world, control, validation, and sealed-test sets
are derived from eligible archive entities. The original TinyStories train,
validation, and GPT-4-only text aggregates are irrelevant to this benchmark
and are not inputs. TinyWorlds-v2 external generation is parked as
non-qualifying historical evidence. LoRA/VAMP continual episodes remain
unstarted because the archive-only scratch base did not pass its publication
gates.

### TinyWorlds-P Status

- **Authoritative archive-only source decision (2026-07-21).** The pinned
  `TinyStories_all_data.tar.gz` records are the complete source universe.
  Archive entities with mechanically recoverable noun, verb, and adjective
  roles are grouped, bucketed, and assigned directly. Base train, held-in
  validation/test, five world train/validation/test splits, and matched
  controls all come exclusively from that eligible archive universe. There is
  no corpus/archive join and no dependency on any published TinyStories text
  aggregate.
- **Prior partition and calibration are superseded.** The prior 8x8 partition,
  scratch run, and 6x6 conclusion were produced from the obsolete
  corpus-intersection universe. They remain immutable historical diagnostics,
  but they are not TinyWorlds-P publication candidates and must not be resumed,
  selected, or used to set the new partition. Exact identifiers remain only in
  the historical audit documents linked below.
- **Corpus-intersection implementation purged.** The obsolete join and all
  corpus-backed paths, identities, offsets, gates, loaders, runners, and tests
  are gone. The purge-only checkpoint remains in history; every subsequently
  restored surface is archive-native and rejects old artifact identities.
- **Archive-native ingestion implemented.** The TinyWorlds-P-owned parser
  authenticates and streams the tarball once, writes exact story bytes to an
  archive-order spool, classifies bounded batches in physical workers,
  externally sorts by normalized story identity and record ID, groups every
  occurrence with its provenance and multiplicity, audits all exclusions, and
  enforces only the 95% token-weighted role-classification gate.
- **Archive-only partition artifacts restored.** Partition construction now
  derives every bucket, cell, split, and control from eligible archive groups;
  publishes exact story and uint16 token shards under
  `data/tinyworlds-p-archive/v1/`; binds documents to member, member-local
  index, record hash, and story hash; and strictly rejects old source keys.
  The CPU 3x3 fixture reconstructs source/token bytes, checks leakage and
  globally unique controls, rejects tampering, and rebuilds byte-identically
  across worker/run settings. The fixed preparation runner and opt-in real
  archive replay have been restored.
- **Canonical archive partition built and reproduced.** The strict 8x8 artifact
  is
  `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`.
  It contains all 4,966,067 eligible archive records (945,499,161 active
  tokens), passes the 99.968% token-weighted role gate, retains every agreeing
  duplicate occurrence, excludes six conflicting duplicate groups, and passes
  topology, component-visibility, split-marginal, globally unique-control,
  exact-byte reconstruction, and sealed-test isolation checks. A second build
  with 24 workers and a different external-sort run size authenticated to the
  same identity and exact `tree.json`, proving byte equality for every strict
  tree entry. The opt-in acceptance test took 39m08s end to end; the build took
  35m11s and its final parallel strict reload took 3m56s. Long real-source
  gates remain excluded from the default test suite.
- **Archive-only scratch training restored.** Memory-mapped batching,
  token-weighted accumulation, immutable complete resume states, streaming
  validation, one-shot sealed test, milestone publication, and the fixed GPU
  runner now consume only strict archive-v1 artifacts. Low-gap fallback uses a
  fresh 6x6 partition with 94/3/3 held-in splits; excessive-gap fallback uses
  10x10 with 96/2/2. Training and every validation/sealed-test batch report
  detached-safe, sparsely refreshed measured phase and pass-path ETAs. CPU
  tests prove interrupted/resumed state and trace parity, schedule and selection
  boundaries, old-resume rejection, finite evaluation, and exact evaluation
  progress.
- **Focused CPU/shared checks pass.** The 82-test TinyWorlds-P, GPT-Neo,
  checkpoint, and training-state suite passes in four concurrent groups
  (9.8s wall time); parked TinyWorlds-v2 tests are still collection-skipped.
- **GPU smoke passes.** The opt-in RTX 4090 smoke strictly loaded the real tree,
  compiled production training, wrote an interrupted update-one state, resumed
  through update two, and measured an 8.695 GiB JAX allocator peak against the
  12 GiB gate. Splitting strict semantics into assignment, provenance, and
  shard/index proof passes reduced the full smoke from 5m20s to 4m20s.
- **Archive-v1 calibration ended with the declared scientific stop
  (2026-07-22).** The fresh seed-zero 8x8 run completed epochs one and two at
  updates 18,832 and 37,664. Held-in NLL improved from 1.261707 to 1.201706,
  but mean gap was only 0.008017, so the fixed policy built the one allowed
  fresh 6x6 partition with 94/3/3 held-in splits. That fallback completed at
  updates 17,200 and 34,400; held-in NLL improved from 1.267558 to 1.206720
  and peak allocation was 8.772 GiB, but mean gap remained only 0.002802 and
  every world gap was below 0.05. The runner therefore exited with its
  controlled status 2. It did not train epochs three through five, select a
  checkpoint, open sealed test, or publish a base. The exact identities,
  metrics, output hashes, gate audit, practical gap analysis, and provenance of
  the engineering thresholds are recorded in
  [`docs/TW-P_ARCHIVE_CALIBRATION_REPORT.md`](docs/TW-P_ARCHIVE_CALIBRATION_REPORT.md).
  A deterministic
  [`validation sample appendix`](docs/TW-P_ARCHIVE_VALIDATION_SAMPLES.md)
  covers held-in base, all five worlds, and both arms of all five controls on
  both grids: 32 exact hash-verified stories selected without semantic review.
  Its two-grid generator runs concurrently in about 1.2 seconds and does not
  read sealed-test indexes.
- **Terminal policy consequence.** The archive-only implementation, 8x8
  partition, independent byte-identical rebuild, fresh 6x6 fallback partition,
  and both scratch calibration attempts are complete. The frozen archive-v1
  conjunction hypothesis did not produce the required representation gap.
  No additional regrid, gate change, historical comparison, or test-set
  inspection is authorized under this benchmark version; a different
  hypothesis requires a new versioned benchmark.
- **Historical audits retained for provenance only.** The original train/archive
  mismatch analysis is preserved in
  [`docs/TW-P_SOURCE_AUDIT.md`](docs/TW-P_SOURCE_AUDIT.md), and the obsolete
  intersection-based calibration is preserved in
  [`docs/TW-P_CALIBRATION_AUDIT.md`](docs/TW-P_CALIBRATION_AUDIT.md). Neither
  audit defines a current source, coverage gate, split, or stopping decision.
- **Test scope remains focused.** Every parked TinyWorlds-v2 test is
  collection-skipped; do not run those legacy bodies. Final verification uses
  the focused TinyWorlds-P and shared GPT-Neo/checkpoint scope in concurrent
  CPU groups. The completed opt-in archive rebuild and RTX 4090 evidence are
  retained rather than rerun as default tests. No continual LoRA or VAMP
  stream work begins from this stopped milestone.

## Parked Milestone: TinyWorlds-v2 External-Generation Benchmark

The parked roadmap is `tinyworlds-v2-gpt`, tracked in detail in
[`docs/TW-v2_PLAN.md`](docs/TW-v2_PLAN.md). The symbolic world ledger remains
authoritative for truth and scoring, while pinned external language models
produce variable-length natural TinyStories-style text through immutable,
content-addressed request/response caches. V2 does not reuse the v1
deterministic renderer or exact-token prose fitting.

### TinyWorlds-v2 Status

- **Phase 1 — direct Qwen/GPT-5.4-Mini author bakeoff: complete with an
  automated scientific stop (2026-07-19).**
  The active experiment now compares exactly two full author routes on the same
  200 already-profiled neutral briefs: Qwen 3.5 35B-A3B and GPT-5.4 Mini. Both
  models generate 200 stories and both are evaluated as possible later corpus
  authors. There is no 50-story screen, finalist expansion, or third-model
  verifier, so the paid plan is exactly 400 author requests. V4 responses are
  one-field `{"story": ...}` objects; whole-word spans, requested-feature
  realization, forbidden forms, and length evidence are derived locally. Only
  mechanically observable requirements are hard gates; semantic plot features
  remain report and blinded-audit evidence. The two routes are compared using
  the frozen TinyStories vocabulary/token
  distribution, TinyStories-8M NLL, surface statistics, cost, and a blinded
  100-reference/100-generated audit split exactly 50/50 by author. The new
  versioned path is `data/tinyworlds-v2/reference-two-route-v2/`, with its own
  raw cache and a `$15` hard cap. Focused tests for the V4 request, local
  validator, explicit two-route catalog plan, direct quality selector, exact
  2×200 preflight, and cost-cap no-secret stop pass. The already completed
  reference artifact is reused, avoiding the prior corpus scan/profile delay.
  The live preflight resolved Qwen to Alibaba and GPT-5.4 Mini to Azure. Its
  expected cost was `$0.305313`, its conservative two-attempt exposure was
  `$1.683770`, and the run therefore remained below the `$15` cap. All 400
  author calls completed for `$0.1906939625`. The corrected V2 artifact reused
  those exact cached responses without a second paid call and strictly
  validates at `data/tinyworlds-v2/reference-two-route-v2/`, manifest
  `6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9`.
  Its terminal status is `no_quality_qualified_route`; the exact balanced audit
  digest is
  `a5d9da91fe9636bda942e1f4532620e7761d4c722358f5cb0e1443fa042fff3a`.

  GPT-5.4 Mini accepted 192/200 briefs (96%) versus Qwen's 123/200 (61.5%),
  and its aggregate alignment distance was better (1.665 versus 2.777). Both
  routes passed vocabulary coverage and token-unigram JSD, with no
  alphanumeric identifier contamination. Both failed the frozen-base NLL,
  story-length, paragraph-format, and dialogue-distribution gates; GPT's
  median NLL delta was 0.905 and its median story length was 40.9% below the
  matched reference median, while Qwen's were 0.927 and 50.7%. The paragraph
  result is potentially a representation mismatch because the matched released
  references contain no counted double-newline breaks while the prompt asks
  for paragraphs, so the blinded audit must be inspected before changing that
  criterion. Phase 2 remains blocked. The 73-test core bakeoff suite passes and
  the active artifact passes strict validation, including raw request/route and
  cost-journal evidence, cost arithmetic, direct-quality/status consistency,
  and audit packet/key/HTML balance. A broader legacy replay run was
  stopped on entry to its known 20--30-minute cache fixture because the failed
  automated gate already prevents phase advancement; the complete default
  suite and zero-network V2 derived replay remain required before any future
  Phase 1 pass can be promoted.

  A deliberately small prompt-tuning review completed on 2026-07-19 without
  changing that stop. It reused 20 namespaced development briefs and their 40
  cached V4 controls, then generated exactly 20 new V6 stories per model. V6
  asks for 130--170 words and adds ordinary TinyStories cadence, opening,
  sentence, English-vocabulary, ending, and single-line paragraph guidance.
  The exact live preflight was `$0.039824` expected / `$0.174217`
  conservative under a separate `$1` cap; all 40 calls cost `$0.0244546375`.
  The promoted diagnostic is `data/tinyworlds-v2/prompt-tuning-v1/`, manifest
  `074cdacbc38e311a85de988801a8c5d2cef561fd88b19daa43640176162836f3`,
  with all outputs in `review.html`. Qwen acceptance moved from 13/20 to 14/20,
  median accepted length from 75 to 110.5 words, and median TinyStories-8M NLL
  from 2.498 to 2.133. GPT remained 20/20, moved from 90.5 to 116.5 words, and
  from 2.296 to 2.262 NLL. Only 1/20 Qwen and 4/20 GPT outputs actually reached
  the requested 130--170-word interval, and both still emitted blank-line
  paragraph breaks. Thus V6 improved alignment distance for both routes but
  did not fix the length, paragraph, or NLL gates. The 20 matched references
  happen to be long-skewed (median 172 words), so this set is explicitly
  development/review evidence and cannot qualify a route or be reused as a
  clean final holdout. The 78-test focused generation/artifact suite passes,
  as does a hardened offline reload; the loader rebuilds requests, locally derived
  evidence, measurement coverage, quality ranking, cost arithmetic, raw-cache
  attempts, and the settled cost journal. The 34-minute complete default suite
  was not repeated for this non-qualifying diagnostic and remains mandatory
  before any future Phase 1 promotion.

  A second 20-brief prompt-shape diagnostic tested V7 on 2026-07-19. V7 moves
  the concrete length/shape requirements to the end of the user message,
  removes the wrapper's redundant compression cues, requires one newline-free
  story block, and asks for 18--20 sentences, at least six connected events,
  and a soft 155--190-word target. It reused the exact V6 controls and purchased
  only 20 new stories per model. The live preflight was `$0.041028` expected /
  `$0.179473` conservative under the separate `$1` cap; all 40 calls completed
  in about 15 seconds for `$0.0296057500`. The raw diagnostic is
  `data/tinyworlds-v2/prompt-tuning-v2/`, manifest
  `838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a`.
  V7 eliminated newlines and moved median accepted length to 154.5 words for
  Qwen and 153.5 for GPT, but Qwen remained 14/20 accepted and GPT fell from
  20/20 to 18/20. More importantly, median TinyStories-8M NLL worsened from
  2.133 to 2.568 for Qwen and from 2.262 to 2.781 for GPT. The concrete
  checklist repaired surface shape while making the stories less like the
  distribution learned by TinyStories-8M.

  The first V7 quality report also exposed a comparator error: 3,393 of its
  10,000 selected GPT-4 validation stories occur in the pinned original
  TinyStories training file under NFKC/case-folded/whitespace-collapsed exact
  identity. The paid V7 output remains immutable, but that report is retained
  only as contaminated-comparator evidence. A zero-call V3 reevaluation filters
  those overlaps, rebuilds the reference profile from the remaining 6,607
  validation stories, and reuses all 80 cached V6/V7 stories and all 66 accepted
  NLL measurements. It is at `data/tinyworlds-v2/prompt-tuning-v3/`, manifest
  `50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74`,
  with every sample in `review.html`. Nine of the 20 small paired archive
  references also occur in original training, but generated-to-reference NLL
  gaps were nearly unchanged across the seen and unseen subsets. Contamination
  therefore mattered to evaluation hygiene, but does not explain the main
  mismatch. The current composite distance nominally ranks V7 first because
  its length/format match offsets other errors; no V7 route passes the hard
  acceptance, NLL, distribution, and language-feature gates, so that rank is
  not a production-prompt selection. The current 94-test focused generation,
  decontamination, replay, and artifact suite passes. The complete default
  suite was not repeated for this non-qualifying development diagnostic.

  A third 20-brief diagnostic tested the bare released prompt on 2026-07-19.
  V8 sends exactly one user message containing the archived TinyStories prompt
  followed by `Possible story:`. It has no system message, repeated
  instructions, JSON request, response schema, or added length/shape rule; the
  complete plain assistant reply is the story and all evidence is derived
  locally. The only other request fields are transport controls such as the
  pinned route, deterministic seed, output ceiling, and no-fallback policy.
  V8 reused the exact V7 stories as controls and used the decontaminated
  6,607-story validation profile directly. The preflight was `$0.034422`
  expected / `$0.152463` conservative under the `$1` cap. All 40 calls finished
  in 17 seconds and cost `$0.0155166000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v4/`, manifest
  `362a0c85c7722fbaf36120eaa5479285edb798bc067d8f7c7fd41631571e2bb0`.

  Both bare-prompt routes accepted 20/20 and realized all three required word
  roles. GPT median NLL improved from V7's 2.781 to 2.185, while Qwen improved
  from 2.568 to 2.475; the decontaminated validation median is 1.347. The bare
  prompt also restored the released prompt's compression and paragraph
  behavior: GPT fell to an 80-word median and Qwen to 113.5 words, versus 138
  in validation, and every new story used paragraph breaks. GPT's NLL gain is
  strong evidence that the wrapper/checklist caused part of its mismatch, but
  neither bare route passes: both still fail NLL and token-distribution gates,
  GPT is 42.0% short, and Qwen is 17.8% short with longer pooled sentences.
  The composite score retains V7 for both routes because its length/shape fit
  outweighs V8's acceptance and NLL gains. That descriptive choice is not a
  production selection. The active artifact validates from persisted evidence
  and the complete focused V2 generation/comparator suite passes; the long
  default suite was not repeated because this diagnostic cannot promote Phase
  1.

  A fourth diagnostic added exactly one sentence to V8:
  `Aim for about 130 to 150 words.` V9 otherwise preserves the same single user
  message, provider seed, route, technical controls, plain-text response, local
  validation, V8 control outputs, and decontaminated comparator. Its preflight
  was `$0.034641` expected / `$0.153339` conservative; all 40 calls generated
  in 15 seconds and cost `$0.0220921000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v5/`, manifest
  `1605d21acff2647fe4be456a627653f606b7e4e90c7241d3d552ebe513430c73`.

  The cue repaired length for both models but exposed a route-specific
  tradeoff. Qwen moved from 113.5 to 147 median words, improved median NLL from
  2.475 to 2.339 and token JSD from 0.324 to 0.293, and reduced composite
  distance from 2.644 to 2.550; it is the better Qwen prompt despite falling
  from 20/20 to 18/20 when two stories omitted required word forms. GPT moved
  from 80 to 128 words and improved token JSD from 0.352 to 0.314, but median
  NLL worsened from 2.185 to 2.381. Its composite distances are effectively
  tied, with bare V8 retaining the nominal lead by 0.0008. V9 therefore passes
  the story-length band for both routes, but neither route passes the NLL,
  token-distribution, sentence-length, paragraph-serialization, or dialogue
  distribution gates. The strict artifact reload and 106-test focused suite
  pass; the long default suite was not repeated for a non-promotable diagnostic.

  A matched LoRA learnability sidebar completed on 2026-07-20 without changing
  the Phase 1 stop. It trained rank-8 adapters for 512 updates on the same eight
  child-to-badge facts and four badge-to-place rules in 24 documents per arm;
  only the prose after each canonical leading evidence sentence differed. The
  arms were a decontaminated official-TinyStories control, Qwen 3.5 35B-A3B,
  and GPT-5.4 Mini. The 72 author calls finished for `$0.0392434000`; the
  promoted artifact is `data/tinyworlds-v2/reasoning-sidebar-v1/`, manifest
  `59200a624dcc8e2afe4cfcdb720d22724184eb97797d7da8208cf0b527d797fe`.
  All three adapters reduced their own training-corpus NLL to at most 0.027,
  and a zero-training exact-clause follow-up scored 100% on all eight literal
  facts and all four literal rules for every adapted arm. Nevertheless,
  held-out two-wording test recall was 25.0% for the TinyStories and Qwen arms
  and 31.2% for GPT, while every arm scored exactly 25.0% on one-hop
  fact-plus-rule questions (four-choice chance is 25%). The clause diagnostic
  is `data/tinyworlds-v2/reasoning-sidebar-v1-clause-probe/`, manifest
  `1d1d8a7921e4ab74b4b23d57266da776d06bf01b3effe5fecc6a92ed5a318b6f`.
  Thus the LoRAs stored literal continuations but did not expose stable
  paraphrase-invariant bindings or compositional knowledge. Because the
  in-distribution control fails the same transfer test, this sidebar cannot
  attribute that failure to Qwen/GPT distribution mismatch. It is exploratory
  evidence only and does not advance Phase 1. Both artifacts strictly reload,
  and the 17-test sidebar/shared-workflow/candidate-scoring suite passes. The
  34-minute complete default suite was not repeated for this non-promotable
  diagnostic.

  The first completed direct artifact at
  `data/tinyworlds-v2/reference-two-route-v1/` is preserved as an over-strict
  validator diagnostic. It spent `$0.1906939625` for all 400 responses but
  incorrectly hard-gated semantic labels such as moral, conflict,
  foreshadowing, and twist using lexical patterns. V2 reinterprets the same
  immutable cached responses without new paid generation: only whole-word
  constraints, safety/length, and quoted dialogue are hard-local evidence;
  semantic plot labels remain reported human-audit judgments.

  The earlier seven-route implementation remains immutable historical evidence.
  It covers exact normalized-content
  source cohorts, deterministic 16-process surface profiles and persisted GPU
  NLL measurements, semantic route identity separated from exact catalog
  provenance, versioned behavior-changing transport headers, public catalog
  revalidation before every paid batch (with a passing public-only live
  resolver smoke on 2026-07-18), append-only completion/stats caching,
  and independent billing observations. Its inclusive runtime cap uses both a
  nonblocking cross-process lease and an fsynced write-ahead reservation/
  settlement journal, persists the complete route lock and sanitized zero-BYOK
  authorization with each reservation, records concurrent pre-POST
  cancellations without charging them, reconciles historical locks after a
  crash, and stops without reposting on ambiguous billing. BYOK also fails
  closed: the inference key cannot prove account state, so production requires
  either a distinct `OPENROUTER_MANAGEMENT_API_KEY` zero-key check or an
  explicit canonical manual attestation valid for at most 24 hours; every
  returned completion/stats record must still prove `is_byok=false`. The
  artifact boundary includes exact generator/verifier cost attribution, route,
  audit, and raw-cache evidence, strict cross-artifact semantic validation, and
  a zero-network byte-for-byte derived replay. Targeted offline unit and
  integration gates, the pinned-source integration, the real-GPU NLL smoke,
  the public-catalog resolver smoke, and focused cost/recovery/replay checks
  pass. The complete offline default suite passes 753 tests with one optional
  skip and eight resource-marked deselections in 2,045.40 seconds; peak RSS was
  4,085,656 KiB. After explicit zero-BYOK confirmation, the production runner
  authenticated and profiled the pinned sources, measured them on the GPU,
  resolved the live routes, and reached its exact preflight. Expected spend was
  `$3.439507`, but the required two-attempt exposure was `$20.020653`, above
  the fixed `$15.000000` cap; the 800-request GPT-5.4 verifier reserve alone was
  `$17.544000`. Per contract, the inference key was not read and zero completion
  POSTs, charges, or generated samples occurred. The promoted stopped artifact
  is `data/tinyworlds-v2/reference/`, with manifest
  `28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9`.
  Strict semantic validation passes and zero-network replay reproduces all 31
  derived files (101,081 bytes). No human audit exists and the Phase 1 gate did
  not pass. A redundant post-artifact default-suite rerun was intentionally
  interrupted at 85% before repeating the known 20--25-minute cache fixture;
  it had no failures, no code changed after the complete pre-run pass, and the
  artifact validation/replay gates passed independently.
  Diagnostic previews treat the external models as synthetic-story authors,
  not as teachers whose behavior is being distilled. The original v1 preview
  at `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v1/` (manifest
  `1ddba6e0862de3e416b4ce21538f5471723e823d6c39c5a32da27a0ea72596b6`)
  is retained as an archived request-contract experiment because its
  `enforce_distillable_text` restriction prevented a meaningful comparison.
  A corrected v2 attempt removed that restriction but was interrupted after 13
  paid requests: its first Qwen response emitted 5,138 hidden reasoning tokens
  despite the 512-token visible-output bound, and exact provider spend reached
  `$0.008248631`. The completed v3 preview explicitly disables optional
  reasoning for Qwen and Gemini and uses the remaining `$0.041751369` of the
  separately authorized `$0.05` cumulative cap. Its promoted artifact is
  `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v3/`, manifest
  `6e1aa9697d8e62263a49c6bc8d22aa22bcb568ca4e551e68b75c727ab063d9f0`.
  All 21 outcomes replay with zero network; 5 passed the current deterministic
  gate (Mistral 1/3, Gemini 3/3, GPT-5.4 Mini 1/3). Provider-reported v3 cost is
  `$0.0061824395`; one unknown-cost timeout is conservatively charged
  `$0.00063045`, for v3 ledger exposure of `$0.0068128895`. This small preview
  remains diagnostic-only and ineligible for route selection. Preliminary inspection
  also shows that the current mechanical gate confounds story quality with the
  model's self-reported evidence schema: Qwen returned three coherent stories
  in a different JSON layout, while two strong GPT stories failed exact
  evidence-quote matching and weaker Gemini prose passed. Before any full
  funnel, separate locally derived story checks from response-metadata validity.
  That correction is now implemented by the active V4 direct comparison. A
  post-run safety audit also closed partial-stats, cache-only recovery,
  no-replace promotion, and cost-evidence validation gaps without making any
  further provider requests. The focused preview/generation suite now passes
  107 tests, and both archived v1 and promoted v3 still validate and replay
  byte-identically under the hardened validator.
- **Phases 2–7: blocked by Phase 1.** Counterbalanced world bibles, natural
  training stories, probes, calibration, the eight-task pilot, and scaling are
  specified in the v2 tracker but are not authorized early. Phase 3 has a
  second mandatory human stop before full-corpus generation.

### TinyWorlds-v1 Gate History

- **Phase 0 — TinyStories post-mortem: complete (2026-07-17).** One exact
  retrain published reusable adaptation artifact
  `0866c521d7accc2576150b5a2cc9b1e4bb9067bcb6403c8da5262f8419b09eef`.
  Reload-only evaluation produced all three paired conditions and the report
  at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-postmortem-seed0-18fcf925db5f`.
  All 1,173 tensor checksums remained identical, the report reproduced
  byte-for-byte from the completed result, and the historical report remained
  byte-identical. The gate suite passed 473 tests with one optional skip and
  five resource-marked deselections.
- **Phase 1 — symbolic TinyWorlds generator: complete (2026-07-17).** The
  calibration and pilot bundles strictly load and rebuild byte-identically at
  digests `ae532f527f9cb35702734aaa819453127f5c30faaf4994436e69a43d2c023c27`
  and `0f24a708301f77b1af8a798869a5c76f8ae7f47205caa7c8cb66b6447d73ca32`.
  The pinned 1,924,281,556-byte original corpus matched SHA-256
  `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`;
  two production re-streams found zero hits among all 144 generated lexical
  forms. All 48 queries have unique graph answers and canonical proofs, bridge
  support requires both branches, all six holdout axes are disjoint, and the
  hardened loader rejects consistently rehashed provenance, topology, proof,
  candidate, revision, story, and capacity tampering. The gate suite passed
  504 tests with one optional skip and five resource-marked deselections; both
  marked tokenizer/novelty integrations passed.
- **Phase 2 — deterministic rendering: complete (2026-07-17).** The pinned
  tokenizer materialized calibration and pilot bundles with exact counts
  5,248 stories/3,072 semantic groups and 10,368 stories/6,144 semantic
  groups. Their rendered digests are
  `ad4a69713060f7d661e92f7415fc4a9ddaffc4852e000cb7c8ba49c3872e4750`
  and `3ad30374480cbeff2587d722b632702de5b097f97f4f9dd651195e3844015bc3`.
  A second invocation strictly reconstructed both trees as `verified`, kept
  the combined tree digest
  `387f4fe2d03b5c868446b9a1884e9134d064b52f4e5e2130796eda1ed53faca0`,
  and reproduced the canonical materialization result byte-for-byte.
  Accepted immutable query-group plans, split-specific templates, exact token
  boundaries and masks, cues, candidate leakage, symbolic alignment, and
  deterministic fallback provenance all passed. The focused gate passed 30
  tests, the marked real-tokenizer gate covered all eight query kinds, and the
  complete default suite passed 516 tests with one optional skip and five
  resource-marked deselections.
- **Phase 3 — candidate scoring and knowledge evaluation: complete
  (2026-07-17).** Frozen, hard-node, and arbitrary-coefficient scorers share
  one four-candidate path with active-token normalization and exact
  microbatch equivalence. The shared hard tensor, both single-run EBT
  refinements and their soft variants, every metric/aggregation axis,
  validation-only parent search, four-role counterfactuals, selected-only
  commit, and immutable resume chunks passed audit. Executed gates cover
  synthetic answer selection, one-hot soft/hard equality, incompatible
  revision answers, incomplete cross-branch hard support, suffix-blind
  routing, and all 11 methods over a real bounded two-edge knowledge stage.
  Final-budget reloads revalidate full execution identity, checkpoint
  histories are exact schedule prefixes, and full-state chunks reject stale
  initial states and dangling symlink targets. The combined focused CPU gate
  passed 77 tests; the complete default suite passed 518 tests with one
  optional skip and five resource-marked deselections.
- **Phase 4 — four-task calibration: complete with controlled scientific stop
  (2026-07-17).** The canonical RTX 4090 runner evaluated exactly three
  validation configurations: the 24-fact baseline, 12 facts, and 36 facts,
  each with 32 exposures/fact, 1,000 updates, rank 8, and hard distractors.
  None passed the complete validation gate, so the fixed ladder stopped
  mechanically at `facts_axis_has_no_passing_configuration`. The immutable
  stopped result is at
  `results/language_cl/tinyworlds-v1/knowledge-graph/calibration-stopped-seed0-e314a9704528`
  with result SHA-256
  `e314a9704528bb8a8133bb4a1465b8be10922df390cbecfb23d33933411ab4e3`.
  All trials retained a 1,024/1,024 exact-KG ceiling, zero committed-node
  drift, and 128/128 old-context consistency. However, frozen novel-binding
  was 64/64 rather than the required 20–30%, leaving zero direct-recall lift;
  revision-node and paired-revision consistency were 0/128 in every trial.
  Independent one-hop accuracy was 0/64 for the 24- and 12-fact trials and
  64/64 for 36 facts, which did not repair the other failed gates. The maximum
  recorded allocator peak was 6,738,299,904 bytes (6.276 GiB), below the
  enforced 12 GiB target. Strict result reload and complete promoted-bundle
  validation pass. The stopped bundle intentionally contains no calibration
  profile and no locked-test artifact. The final complete default suite passes
  537 tests with one optional skip and five resource-marked deselections.
- **Phase 5 — eight-task pilot/report: not launched.** Phase 4 produced no
  passing `calibration_profile.json`; `pilot_authorized` is false and the
  held-out calibration test remained unopened. The TinyWorlds-v1 contract
  therefore forbids launching the pilot. This is the prescribed visible
  scientific outcome, not an implementation failure.
- **Interactive calibration playground: implemented (2026-07-18).**
  `notebooks/tinyworlds_playground.ipynb` provides a read-only view of the
  symbolic world, rendered stories and alignments, proof depth, cue variants,
  hard and continuous support, saved candidate NLLs, gate evidence, and parent
  transfer. Its supporting module strictly reloads the promoted Phase 4 result
  and can generate small worlds through the production symbolic generator,
  tokenizer, and renderer. Fresh samples never inherit saved model scores;
  only the exact canonical seed-0 demo may be joined to validation evidence.
  The notebook does not train, tune, open the held-out test split, or imply
  that Phase 5 is authorized. A clean-kernel execution passed end-to-end, and
  the complete default suite passed 550 tests with one optional skip and five
  resource-marked deselections.
- **TinyWorlds-v1 language-distribution audit: failed (2026-07-18).** The
  renderer is mechanically tokenizer-valid but far outside the frozen
  TinyStories model's training distribution. Entity surfaces are `N` plus 12
  hexadecimal characters and average 8.42 BPE pieces, versus 1.21 for names in
  a matched corpus sample. Generated stories have token-distribution
  Jensen-Shannon divergence 0.702 from TinyStories, while two independent
  TinyStories samples differ by 0.014; an estimated 68.5% of story tokens are
  exact-length padding. Frozen-model scoring on 128 matched 256-token examples
  produced mean NLL 6.780 for TinyWorlds and 1.460 for original TinyStories,
  about a 204-fold perplexity ratio. Raw task IDs, hash marks, symbolic
  predicate labels, and rule-variable labels also enter visible prose, while
  candidate-specific filler contaminates answer NLL. The exact whole-word
  novelty audit did not measure these properties. Preserve v1 as historical
  evidence, but do not interpret its learned-model results as a clean test of
  knowledge-graph acquisition.

### Parked TinyWorlds-v2 Follow-up

1. Preserve the promoted stopped calibration bundle as the terminal
   TinyWorlds-v1 result; do not modify it or launch its Phase 5 pilot.
2. Preserve the valid `blocked_by_cost_cap` Phase 1 artifact and its exact
   `$3.439507` expected / `$20.020653` conservative evidence; do not overwrite
   or reinterpret it.
3. Preserve the completed V2 direct-bakeoff artifact, its 400 raw responses,
   exact `$0.1906939625` bill, failed quality metrics, and balanced audit. Do not
   overwrite the scientific stop or silently relax its thresholds.
4. Inspect `data/tinyworlds-v2/prompt-tuning-v5/review.html`; it shows all 20 V9
   outputs per model beside the exact V8 controls and archive stories, and
   exposes the exact one-sentence request difference.
5. Preserve the route-specific result rather than averaging it away: V9 is
   better for Qwen on length, NLL, token JSD, and composite distance, while GPT
   trades its V8 NLL advantage for correct length and nearly identical
   composite distance. Neither prompt/route cell passes the complete gate.
6. Do not buy another prompt cell until the V5 review is inspected. Any further
   change must state which remaining failure it isolates; do not reintroduce a
   system message, JSON, or a narrative checklist as a bundled intervention.
7. Do not generate world bibles. Phase 2 remains blocked because neither route
   passed the automated Phase 1 gate; human inspection cannot retroactively
   turn the current artifact into a passing one.
8. Add zero-network derived replay for a future passing two-route result before
   treating the Phase 1 gate as passed; the current strict validator already
   authenticates configuration, planned V4 requests, result partitions, local
   evidence, and measurement coverage.
9. Run the complete default suite after any corrective implementation and
   before a future Phase 1 promotion; retain the fast focused suite as the paid
   boundary preflight.

### Deferred Alternative

- A domain-pretrained, TinyStories-scale base model is recorded in
  `docs/TINYWORLDS_DOMAIN_PRETRAINING_NOTE.md`. It would learn KG-oriented
  language from many disposable worlds before continual evaluation on wholly
  held-out worlds. This is a preserved research option, not active work or
  authorization to modify the v1 artifacts.

## Completed Foundation: Language-Model VAMP Proof of Concept

- The completed language-model VAMP foundation and its phase gates are
  recorded in `docs/LM_VAMP_EXECUTION_PLAN.md`.
- Phases 0-10 are implemented: the architecture/build contract is recorded,
  generic immutable graph topology backs the migrated dense-MNIST memory,
  the typed plain-JAX GPT-Neo core passes its CPU correctness and overfit
  gate, and fixed-capacity pathwise LoRA passes its isolation, masking, and
  candidate-gradient gate.
  The verified TinyShakespeare path now includes pinned text preparation,
  deterministic character batches, immutable clipped-AdamW training,
  schema-v1 safetensors checkpoints, and uncached greedy generation.
  Strict TinyStories-8M conversion and the complete offline Hugging Face
  residual/logit/NLL/generation parity ladder pass at the pinned revision.
  Prefix/suffix language contracts, exhaustive normalized-prefix routing,
  and frozen-base content-key derivation pass their masking and task-identity
  exclusion gates.
  The immutable language transition passes a real two-task character-
  permutation run with stable base, prior-run state, and old-node logits.
  Normalized Hopfield retrieval passes real derived-key, capacity masking,
  independent-batch, temperature, top-k, and evaluator-metric gates. EBT
  refinement now optimizes independent per-example node logits, supports all
  four prescribed starts, returns both soft and hard results, and passes its
  decreasing-objective, masking, equivalence, and immutability gates.
  Phase 10 adds deterministic character-permutation, corpus-region,
  stable-hash, and pinned TinyStories topic curricula; document-safe packing;
  the complete four stored/five routed baseline matrix; stored/routing
  forgetting, regret, transfer, logical/runtime memory, synchronized timing,
  random-control confidence intervals, and enforced peak-memory targets; and
  deterministic standalone language reports with all prescribed artifacts.
  TinyStories corpus loading now verifies the pinned files before bounded
  streaming selection, retaining only selected stories plus compact content
  identities. Evaluation and routing support explicit microbatches, reuse
  suffix scores across routers, and cache the shape-stable EBT optimizer
  executable. The real pinned validation aggregate cannot supply the original
  1,000 examples for every topic under the fixed two-concept-plus-margin rule,
  so the measured single-GPU preset uses 10,000/128/128 train/validation/test
  stories and 128 probes/examples while preserving the source, split, topic,
  and deterministic hash-selection contracts.
  Report samples are now completed before the final allocator peak is enforced
  and are reused by the report-only projection. EBT refinement now retains
  aligned node-probability, path-edge-coefficient, and objective trajectories;
  reports preserve a deterministic final-task representative trace as JSONL
  and render four coefficient heatmaps plus an objective curve.
- The target is a shared plain-JAX GPT-Neo base with immutable pathwise LoRA
  memory, a TinyShakespeare smoke path, converted TinyStories-8M weights,
  exhaustive/Hopfield/EBT task-free addressing, and reproducible continual-
  learning reports.

## Current Phase and Immediate Gate

The engineering implementation and resource-backed Phase 10 validation are
complete on the local RTX 4090. Scientific follow-up is now focused on transfer
and routing quality rather than feasibility.

Measured status on 2026-07-16:

- the pinned TinyShakespeare corpus is present and its locally trained
  5,000-step base checkpoint improved validation NLL from 4.20647 to 1.61060;
- both pinned TinyStories V2/GPT-4 aggregates match their prescribed sizes and
  SHA-256 digests, and the TinyStories-8M conversion is present with parameter
  checksum `cdb66d6fe8377d09c43db0631fecb7265216d4383232bff6d0d5f7d0047bf5de`;
- strict prepared-source conversion and the full offline Hugging Face/JAX
  tokenization, residual, logit, NLL, and greedy-generation parity ladder pass;
- the complete TinyShakespeare character-permutation report is at
  `results/language_cl/tinyshakespeare/character-permutation/standard-seed0-a7bd7d1479ba`.
  At the final stage and primary 64-token prefix, exhaustive and both EBT
  routers reach 100% task-node accuracy, Hopfield reaches 86.33%, and the
  deterministic random control reaches 23.44%. The measured allocator peak is
  2,203,208,704 bytes;
- a real 2.2 GB TinyStories streaming preflight filled all four task splits at
  10,000/128/128, retained 13,815 root-validation stories, completed in 4m36s,
  and used no swap; and
- the full stable-hash negative-control report is at
  `results/language_cl/tinyshakespeare/stable-hash/standard-seed0-5f8f82a979a3`.
  Aggregating all four tasks at each prefix, exhaustive, both EBT routers, and
  the deterministic random router have 95% intervals containing 25% chance at
  every prefix. Hopfield contains chance at prefix 32 and is below chance at
  prefixes 64 and 128. The five above-chance task-slice audit flags disappear
  under the prescribed four-task aggregation; two also occur in the
  deterministic random control. The audit found no cross-split macro-document
  overlap and no above-chance aggregate leakage signal; and
- the full TinyStories topic report is at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-seed0-9f715620e7c2`.
  It completed all four 2,000-step tasks and the benchmark/sample workload in
  about 55 minutes. The final allocator peak was 8,349,717,248 bytes (7.776
  GiB), below the enforced 12 GiB target. At the primary 64-token prefix,
  independent root LoRAs were best at 1.33617 NLL; sequential LoRA reached
  1.35269 with 0.02031 forgetting; and VAMP oracle reached 1.36035 with zero
  stored forgetting. Exhaustive and Hopfield routing reached 46.48% and 45.90%
  exact task-node accuracy, while Hopfield was 8.9x faster warm. Both inherited
  children finished about 0.05 NLL behind independent training, and EBT did not
  improve on plain Hopfield. The process loaded the schema-1 report surface
  before commit `95e0cf4`, so its aggregate metrics are complete but the newer
  stepwise EBT coefficient traces require a rerun if they are needed.

Both canonical runners now emit the manifest, seven JSONL metric families,
address confusion, three aggregate metric charts, graph, five EBT routing-
dynamics charts, samples, and standalone HTML under a content-addressed run
directory. Offline bounded tests exercise the same training, all nine methods,
measurement, trace capture, sample generation, and report writer without
substituting for the full-resource measurements. The latest default CPU gate
passes 376 tests with one expected optional-dependency-boundary skip and two
resource-marked tests deselected. Running those two integration tests
explicitly also passes both against the prepared local artifacts.

## Deferred Stage-1 FabricPC Work

Stage-1 dense-delta VAMP over PermutedMNIST and digit-incremental MNIST remains
available with VAE and FabricPC backends. Its reports, fixed-epoch schedule,
and observed-energy-convergence schedule are retained, but further FabricPC
benchmark development is parked until the LM VAMP milestone changes priority.

Deferred gaps are:

- pad and mask FabricPC evaluation, observed-energy, reconstruction, tail, and
  addressed-winner batches to prevent shape-driven JAX recompilation;
- run the full ten-digit convergence benchmark (current verification covers a
  deterministic two-digit VAE run and one-digit FabricPC smoke run);
- inspect stopping epochs, reconstruction quality, and train/test address
  confusion; and
- compare that run with the existing fixed ten-epoch FabricPC checkpoint,
  especially early-digit routing to later parameter nodes.

Observed energy remains model-specific and supports within-model stopping and
addressing only. Reaching the epoch limit retains the best state and reports
`max_epochs`; it is not convergence.

## Known LM VAMP Gaps

No required engineering surface or resource-backed gate from the execution
plan remains incomplete. Research priorities are to replace prefix-NLL-only
parent selection or add a root fallback for negative transfer; measure whether
whole-story topic cues are visible in each evaluated prefix; correct the
fantasy-biased content keys; and explain why EBT sharpens its addressing while
losing suffix quality, especially at prefix 128. Add incremental progress
events before another long resource run. Rerun the full TinyStories report only
if the post-`95e0cf4` stepwise EBT coefficient artifacts are required. The
validated public routing wrapper performs host-side postcondition checks and is
intentionally timed directly; extracting a separate outer-JIT-compatible
validated factory remains optional optimization.
