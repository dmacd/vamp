# TinyWorlds-P Semantic-v4 Construction and Partition Report

Date: 2026-07-22

Benchmark: `tinyworlds-p-semantic-v4`

Result: **the one-shot fixed-centroid catalog passed, then the preregistered
semantic-first topology failed its 10% median-mass gate; v4 stopped before
partition publication or training**

## Outcome

Semantic-v4 tested the instability isolated by semantic-v3. It reproduced
v3's all-candidate pass-zero noun and verb fits exactly, froze those centroids
and assignments, and screened every candidate once at the unchanged `0.03`
nearest-versus-second-nearest cosine margin. It did not delete and recluster,
reseed a centroid, move a retained word, or update a centroid after screening.

The one-shot screen excluded 188 of 978 noun candidates and 81 of 365 verb
candidates. All eight fixed noun clusters still retain at least 39 words,
above the 32-word floor. All eight fixed verb clusters retain at least 18
words, above the 12-word floor. Fit-centroid separation and joint archive
coverage also pass. V4 therefore publishes the first valid semantic word grid
in this sequence:

- catalog SHA-256:
  `ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`;
- retained non-construction tokens: `479,183,203` of `898,327,086`;
- retained fraction: `53.341729362%`;
- generated [`audit.md`](../data/tinyworlds-p-semantic/catalog/v4/ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee/audit.md)
  and standalone
  [`audit.html`](../data/tinyworlds-p-semantic/catalog/v4/ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee/audit.html).

This pass authorized the separately frozen v4-native partition attempt below.
That attempt did not pass, so it never authorized GPU training, checkpoint
selection, or sealed-test opening.

## Frozen intervention

V4 binds semantic-v3 failure artifact
`ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`
and reuses encoder evidence
`efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`.
Construction authenticates the v3 source ledger and requires exact replay of
every word's token mass, context count, calibrated role score, fold, reference
count, conformal value, cutoff, sense metric, and candidate vector. It also
requires the new pass-zero noun and verb trace records to equal v3 exactly.

For each role, the only clustering fit is:

- eight fixed clusters;
- unweighted farthest-first spherical k-means;
- v3's canonical hash namespace and tie authority;
- nearest-cosine assignment;
- float32 normalized centroids;
- at most 100 centroid iterations.

After that fit, the centroids and assignments are immutable. Each candidate's
margin is measured against those centroids exactly once. A margin below
`0.03` excludes the word from the factorial grid. The unchanged post-screen
gates require at least 32 nouns or 12 verbs per fixed cluster, fit-centroid
pair cosine below `0.90`, and at least 40% joint archive token coverage.

The catalog distinguishes the two relevant views:

- `fit-clusters.json` contains all 978 noun and 365 verb candidates used to
  estimate the frozen centroids;
- `clusters.json` contains the 790 nouns and 284 verbs retained for the
  factorial grid, using the same centroids and assignments;
- `words.jsonl` retains vectors, fit assignments, and fixed margins even for
  all 269 boundary-excluded words.

A strict loader reconstructs the complete one-time fit from the word vectors,
checks every assignment and margin, checks both cluster views, recomputes all
gates, and rejects changed identities or bytes.

## Exact source replay and screen

The v2-calibrated role and sense screens are unchanged:

| Role | Archive role words | Role outliers | Multi-sense | Fit candidates | Boundary-excluded | Retained |
|---|---:|---:|---:|---:|---:|---:|
| Noun | 1,066 | 51 | 37 | 978 | 188 | 790 |
| Verb | 394 | 19 | 10 | 365 | 81 | 284 |

The two v4 screen records are byte-for-value equal to v3's pass-zero records:

| Role | Fit words | Below 0.03 | Minimum margin | Margin q10 | Median margin |
|---|---:|---:|---:|---:|---:|
| Noun | 978 | 188 | 0.000126 | 0.018312 | 0.066057 |
| Verb | 365 | 81 | 0.000604 | 0.013316 | 0.062116 |

There are no later passes. In particular, v4 does not produce v3's
`790 -> 773 -> 767 -> 766 -> 754` noun cascade or its changing centroid
geometries. The boundary decision is relative to one predeclared reference
fit.

## Cluster gates

| Role | Cluster | Fit words | Retained words | Excluded | Fit token mass | Retained token mass |
|---|---:|---:|---:|---:|---:|---:|
| Noun | 0 | 182 | 142 | 40 | 154,794,518 | 120,830,887 |
| Noun | 1 | 73 | 59 | 14 | 61,432,198 | 49,695,806 |
| Noun | 2 | 97 | 77 | 20 | 81,441,273 | 64,478,810 |
| Noun | 3 | 141 | 126 | 15 | 118,429,117 | 105,733,683 |
| Noun | 4 | 132 | 111 | 21 | 109,675,885 | 92,055,125 |
| Noun | 5 | 210 | 162 | 48 | 178,790,650 | 138,057,365 |
| Noun | 6 | 42 | 39 | 3 | 34,843,331 | 32,329,604 |
| Noun | 7 | 101 | 74 | 27 | 84,830,013 | 62,125,725 |
| Verb | 0 | 86 | 57 | 29 | 195,896,353 | 129,488,598 |
| Verb | 1 | 20 | 18 | 2 | 45,472,227 | 40,968,460 |
| Verb | 2 | 23 | 20 | 3 | 53,215,477 | 46,219,508 |
| Verb | 3 | 60 | 48 | 12 | 137,565,002 | 110,109,346 |
| Verb | 4 | 49 | 37 | 12 | 112,373,977 | 85,053,952 |
| Verb | 5 | 38 | 30 | 8 | 86,546,558 | 68,259,829 |
| Verb | 6 | 40 | 38 | 2 | 89,779,262 | 85,261,163 |
| Verb | 7 | 49 | 36 | 13 | 110,967,203 | 81,599,215 |

The smallest retained noun cluster contains 39 words and the smallest verb
cluster 18. The maximum noun fit-centroid pair cosine is `0.8735721184`; the
maximum verb value is `0.8916218581`. Both are below `0.90`. Joint retained
token coverage is `53.341729362%`, 13.34 percentage points above the 40%
floor.

## Why the excluded fraction does not invalidate the test

The experimental factors are the eight noun clusters and eight verb clusters,
not a requirement that every candidate word enter the benchmark. Excluding a
boundary word is valid only if the decision is construction-only, fixed before
language-model training, and leaves each factor sufficiently populated and
observable. V4 now establishes those conditions mechanically:

- no loss, partition outcome, checkpoint, or test observation affects an
  exclusion;
- the same immutable reference geometry decides every word;
- every fixed cluster remains well above its independent inventory floor;
- more than half of eligible archive token mass remains jointly observable;
- every excluded word and its measured evidence remain auditable.

The exclusions would have invalidated this version if they emptied or starved
a cluster, collapsed centroid separation, or reduced coverage below 40%.
Those were automated stops, and none fired. The result does not claim that
boundary words are meaningless; it claims they are too ambiguous to define a
clean cluster-specific treatment under the frozen representation.

## Partition topology outcome

The partition contract was frozen before the archive run. It accepts only the
canonical archive, tokenizer, and v4 catalog and uses no model loss. Every
non-construction group whose noun and verb survived enters the cell audit. The
topology remains the A/B/C/D 2-by-2 corner plus unrelated E. Candidates first
must be nonempty, expose every selected component in at least 64 outside
groups, and provide row/column control capacity. Remaining candidates are
ranked lexicographically by semantic dispersion, token imbalance, nuisance
imbalance, negative control capacity, and v4-namespaced hash. Only the winner
then faces the fixed 10%-around-median cell-mass gate.

The canonical replay produced:

| Disposition | Duplicate groups | Active tokens |
|---|---:|---:|
| Retained for topology | 2,520,317 | 479,183,203 |
| Permanent construction slice | 247,629 | 47,172,075 |
| Catalog-excluded noun or verb | 2,198,121 | 419,143,883 |

The retained token total exactly equals the catalog's authenticated mass. All
28,224 physical candidates were nonempty and passed both component visibility
and control capacity. The preregistered winner was:

| World | Noun cluster | Verb cluster | Groups | Active tokens |
|---|---:|---:|---:|---:|
| A | 1 | 2 | 13,297 | 2,559,355 |
| B | 3 | 2 | 28,201 | 5,440,146 |
| C | 3 | 4 | 52,311 | 9,899,869 |
| D | 1 | 4 | 24,492 | 4,699,583 |
| E | 6 | 1 | 7,688 | 1,428,732 |

The five-cell median is `4,699,583`, giving an allowed interval of
`[4,229,624.7, 5,169,541.3]`. Only D lies inside it; v4 therefore stopped. The
winner's semantic dispersion is `0.32355184049620855`.

Exactly 22 candidates satisfy the median gate as diagnostic evidence. The
highest-ranked of those uses cells `(3,4), (4,4), (4,6), (3,6), (2,0)`, with
masses `9,899,869`, `8,829,612`, `8,742,369`, `10,104,204`, and `9,357,468`.
Its semantic dispersion is `0.329787101192787`, so it loses at the first
lexicographic objective. Promoting it after observing the failure would change
v4's selection rule from semantic-first to median-feasible-first. The run does
not make that post-hoc substitution.

The content-addressed failure SHA-256 is
`37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`.
Its generated [`audit.md`](../data/tinyworlds-p-semantic/v4/failures/37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07/audit.md),
self-contained
[`audit.html`](../data/tinyworlds-p-semantic/v4/failures/37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07/audit.html),
and 28,224-record `topology-candidates.jsonl` bind the complete stop. A second
replay rebuilt all failure files byte-for-byte.

No success partition directory was published. Consequently there are no
world/base splits, paired control allocations, validation sample report,
runtime measurement, calibration run, checkpoint, or sealed-test result.

## Published catalog artifact

The canonical directory is
`data/tinyworlds-p-semantic/catalog/v4/ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee/`.
Its bound payload identities are:

- calibration:
  `64df24c65dc73e9b8de06bc2cd3e3106b73b3f9e9da12f5fd27997e3ed89af6c`;
- one-shot trace:
  `f23ae570cdc90c6d9581761b05464c9ddaab5fc04c81d218117fc7e889955765`;
- exhaustive word ledger:
  `c218192436509b3afde97f3f512663ec2537f5c70b52e20f050a5c8a2e93ac04`;
- all-candidate fit clusters:
  `1a83d01407cdab871b1b197dcbdfa5762f667abac5c7f6ce8ed21a9239b306b1`;
- retained clusters:
  `5d24476f22ca2cfb77c544a53a26d077095bfe34af810192c817a26899a6be21`;
- exact contexts:
  `e35d83bf41ab57b988a3256cc9a29e6862fdb444fdade5ba2d586349a7cfa43a`;
- role-pair mass ledger:
  `6493d6bc1e349b46fd3759941c6f8a675d84c923a13fb454e2a4cca9b5166025`.

The tree has 11 files totaling 35,915,963 bytes. The HTML audit parses cleanly
and contains no external scripts, stylesheets, images, or network references.

## Implementation and verification

V4 adds separate config/catalog/failure formats, a one-shot builder, strict
fit replay, fit-versus-retained audits, a cached-evidence runner, v4-native
partition/sample-report contracts, a complete topology-failure publisher and
loader, and CPU fixtures. The synthetic tests cover:

- frozen v3 namespace and no-update configuration;
- content-addressed success and byte-identical rebuild;
- explicit low-margin exclusions whose vectors and fit assignments persist;
- proof that published centroids are not recomputed from retained members;
- failed cluster-count publication and strict replay;
- failed role-calibration publication and strict evidence loading;
- success and failure tamper rejection;
- exact construction exclusion, archive-byte reconstruction, global control
  non-reuse, one-to-one pair coverage, and all 16 validation-report conditions;
- v4/v1 format rejection and cross-worker/run-size partition rebuilds; and
- ranked topology-failure publication, byte-identical repeat publication, and
  audit tamper rejection.

The canonical run used NumPy 1.26.4 and completed in about 15 seconds without
GPU inference. A loader replay also passes in the project's newer NumPy
environment. An independent temporary construction reproduced the catalog SHA
and all 11 files byte for byte. The real partition scan took about 12 minutes
through the topology stop; two bounded recovery/rebuild passes independently
reproduced the 20 MB failure bundle. The focused
archive/semantic/partition/training regression group passes all 47 tests in
the complete semantic environment; all new modules, runner code, and tests
compile.

## Terminal v4 boundary

V4 has no authorized downstream action. Its catalog remains valid construction
evidence, but its fixed topology policy did not produce a partition. The
runner now detects and strictly reloads the canonical failure, reports the
stop, and exits before repeating archive work.

A future attempt may reasonably test a balance-feasibility prefilter or a
joint/nonlexicographic semantic-balance objective, because the audit proves
that 22 balanced topologies exist. That is a new intervention and must be
declared as semantic-v5 before inspecting any new result. It may compare this
audit, but it cannot promote a diagnostic v4 candidate, change v4's 10%
tolerance, manufacture a v4 sample report, or start v4 GPU calibration.
