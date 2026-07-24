# TinyWorlds-P Semantic-v3 Semantic-First Construction Report

Date: 2026-07-22

Benchmark: `tinyworlds-p-semantic-v3`

Result: **semantic-first clustering removed the mass-assignment conflict, but
the frozen boundary fixed point still failed; no catalog or downstream model
artifact**

## Outcome

Semantic-v3 tested the specific intervention motivated by semantic-v2: words
were clustered only by semantic geometry, and archive token balance was
deferred to later story allocation. No token mass was used to initialize a
centroid, weight a word, constrain an assignment, or move a word away from its
nearest centroid.

The change substantially improved the boundary trace. Verbs converged after
two exclusion/recluster passes with 277 retained candidates. Nouns went from
188 failures on the initial clustering to one failure on the fifth and final
permitted recluster. That last word, `crayon`, had a true nearest-versus-second
nearest margin of `0.0296120345`, below the frozen `0.03` minimum. Therefore
the exact v3 contract stopped and published failure evidence rather than
silently taking a sixth pass.

The generated canonical evidence is available as
[`audit.md`](../data/tinyworlds-p-semantic/catalog/v3/failures/ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146/audit.md)
and standalone
[`audit.html`](../data/tinyworlds-p-semantic/catalog/v3/failures/ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146/audit.html).

## Frozen intervention

V3 reuses MiniLM evidence
`efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`
and the exact authenticated v2 role-score ledger in failure artifact
`23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
The role fold, raw q10 score, reference count, conformal p-value, empirical
cutoff, and pass/fail decision for every sufficient-context word are therefore
identical to v2. The context minimum, sense gate, and semantic word-vector
construction are also unchanged.

For each role, v3 uses:

- eight fixed clusters;
- deterministic farthest-first seeds in the v3 hash namespace;
- unweighted float32 spherical centroids;
- assignment to the highest-cosine centroid only;
- at most 100 centroid iterations;
- a true nearest-minus-second-nearest margin minimum of `0.03`;
- the initial clustering plus five exclusion/recluster passes;
- at least 32 nouns or 12 verbs per final cluster;
- centroid-pair cosine below `0.90`;
- at least 40% joint non-construction token retention.

The v3 config contains no cluster-mass bounds and no packing repair. Cluster
and cell token masses are audit measurements only. Had the semantic catalog
passed, balancing would have occurred over complete duplicate-story groups in
a separately frozen partition contract.

## Numeric provenance preflight

The first implementation invocation used the general project environment with
NumPy 2.5.1. It produced the same role decisions, boundary trace, terminal
word, and stop reason, but a few recorded float values differed from v2 by
several billionths because v2 had been constructed under the semantic
environment's NumPy 1.26.4. That preflight bundle is preserved as
`94614921b5386653f92ee8dc372fc45b566502f9706723df57b257ab4a1252f2`
but is not the canonical v3 contract.

Before canonical publication, the implementation added two provenance checks:
the config binds NumPy 1.26.4 and the v2 failure-artifact identity, and the
real-evidence builder requires every recomputed calibrated word score to equal
the authenticated v2 ledger exactly. No threshold, clustering method, pass
budget, word disposition, boundary trace, or stop decision changed. The
canonical calibration payload SHA-256 is consequently the same as v2:
`64df24c65dc73e9b8de06bc2cd3e3106b73b3f9e9da12f5fd27997e3ed89af6c`.

## Screen result

All 1,460 role words again had at least 32 exact contexts. Exact v2 replay
gave the same pre-clustering result:

| Role | Input words | Calibrated role outliers | Multi-sense exclusions | Cluster candidates |
|---|---:|---:|---:|---:|
| Noun | 1,066 | 51 | 37 | 978 |
| Verb | 394 | 19 | 10 | 365 |

## Semantic-first boundary result

| Role | Pass | Input words | Margin failures | Minimum margin | Margin q10 | Median margin |
|---|---:|---:|---:|---:|---:|---:|
| Noun | 0 | 978 | 188 | 0.000126 | 0.018312 | 0.066057 |
| Noun | 1 | 790 | 17 | 0.017673 | 0.044523 | 0.087217 |
| Noun | 2 | 773 | 6 | 0.002943 | 0.046789 | 0.090001 |
| Noun | 3 | 767 | 1 | 0.019970 | 0.047740 | 0.090853 |
| Noun | 4 | 766 | 12 | 0.005571 | 0.046408 | 0.090593 |
| Noun | 5 | 754 | 1 | 0.029612 | 0.048261 | 0.092612 |
| Verb | 0 | 365 | 81 | 0.000604 | 0.013316 | 0.062116 |
| Verb | 1 | 284 | 7 | 0.010055 | 0.046412 | 0.087323 |
| Verb | 2 | 277 | 0 | 0.032401 | 0.049035 | 0.090930 |

Across permitted passes, 225 distinct nouns and 88 distinct verbs were
flagged. This is far less destructive than v2's capacity-constrained 655 noun
and 225 verb exclusions, and all assigned margins are now nonnegative because
assignment is truly nearest-centroid. It nevertheless does not establish a
valid noun fixed point within the preregistered budget.

## Why the apparent one-word miss is not harmless

The terminal count alone understates the failure. A post-stop diagnostic,
which is not used to reinterpret v3, removed `crayon` and ran one additional
nearest-centroid recluster. That pass exposed 22 new noun failures:
`barn`, `bee`, `bush`, `clown`, `fence`, `fish`, `grass`, `green`, `hedge`,
`hero`, `leaf`, `nature`, `pine`, `seal`, `season`, `shelter`, `tank`, `tree`,
`tuna`, `well`, `whale`, and `yard`. Before removing those words, the eight
noun cluster counts were `138, 52, 102, 76, 123, 158, 25, 79`; the 25-word
cluster already violated the independent 32-noun floor. Thus accepting or
deleting only the terminal word would hide a cascade rather than repair a
nearly complete catalog.

Coverage is not the blocker. Accepting terminal `crayon` would retain
49.666249% of non-construction token mass; removing it would retain 49.595779%;
and removing the next 22 diagnostic failures would still retain 48.162340%.
The diagnostic maximum centroid-pair cosines were `0.832504` for nouns and
`0.851936` for verbs, both below `0.90`. V3 therefore isolated a different
problem: repeated hard deletion plus complete farthest-first reclustering does
not yield a stable eight-cluster noun core under the current margin/count
rules.

## Published failure evidence

The canonical immutable failure artifact is:

- directory:
  `data/tinyworlds-p-semantic/catalog/v3/failures/ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146/`
- failure SHA-256:
  `ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`
- calibration payload SHA-256:
  `64df24c65dc73e9b8de06bc2cd3e3106b73b3f9e9da12f5fd27997e3ed89af6c`
- boundary trace SHA-256:
  `9e9b78ab6308a77362050f406ab1f1fefd97cc836c84ef6a90a8a4aa6b15b6f1`
- word ledger SHA-256:
  `94d5624d8995e84f0cefd96bc46cdff100b917a595e734b4b915fed2a249a76f`
- exact-context payload SHA-256:
  `e35d83bf41ab57b988a3256cc9a29e6862fdb444fdade5ba2d586349a7cfa43a`
- role-pair mass SHA-256:
  `6493d6bc1e349b46fd3759941c6f8a675d84c923a13fb454e2a4cca9b5166025`
- total size: 35,643,464 bytes.

The Markdown and self-contained HTML audits include every role word and
disposition, exact role calibration, sense metrics, candidate vectors,
pass-level masses and margin summaries, PCA views, and representative exact
archive contexts. The HTML parses without external scripts, stylesheets,
images, or network references.

## Implementation and verification

The v3 implementation adds:

- [`v3_contracts.py`](../src/apm/data/text/tinyworlds_p_semantic/v3_contracts.py)
  for the semantic-first config and strict artifact identities;
- [`v3_catalog.py`](../src/apm/data/text/tinyworlds_p_semantic/v3_catalog.py)
  for exact v2 calibration replay, semantic-only clustering, publication, and
  nearest-assignment validation;
- [`prepare_tinyworlds_p_semantic_v3.py`](../scripts/prepare_tinyworlds_p_semantic_v3.py)
  for the fixed cached-evidence run;
- [`test_tinyworlds_p_semantic_v3_catalog.py`](../tests/test_tinyworlds_p_semantic_v3_catalog.py)
  for mass-independence, exact fold reuse, CPU 3-by-3 success/failure,
  byte-identical rebuild, audit, and tamper coverage.

Three focused semantic/archive and shared GPT-Neo/checkpoint/training
regression groups pass all 153 tests. Every semantic module, runner, and test
file compiles.

The canonical construction took about 18 seconds including strict v2 source
authentication and publication. A cached invocation strictly reloaded the
same failure in about four seconds. An independent build under an
automatically removed temporary root reproduced the failure identity and all
nine files byte for byte. No GPU inference, archive partition build, language
model, checkpoint, or sealed-test file was used.

## Version consequence

Semantic-v3 answered the v2 question: mass-constrained word assignment was a
major source of negative and ambiguous margins, and semantic-first assignment
preserves enough archive coverage. It also showed that hard boundary deletion
with complete reinitialization is itself unstable for the noun geometry.
Semantic-v3 therefore stops without a catalog, partition, sample report,
training run, checkpoint, or sealed-test result.

A future attempt must be a separately preregistered semantic-v4. Plausible
interventions include fitting clusters once and treating low-margin words as
out-of-grid without reseeding, learning a stable core before applying the
margin, or replacing iterative deletion with a single globally specified
robust clustering objective. It must not waive v3's terminal word, extend its
pass budget post hoc, or call the v3 failure a successful confirmatory grid.
