# TinyWorlds-P Semantic-v2 Role-Calibration Report

Date: 2026-07-22

Benchmark: `tinyworlds-p-semantic-v2`

Result: **role calibration succeeded; fixed semantic clustering stopped; no catalog or downstream model artifact**

## Outcome

Semantic-v2 corrected the role gate that eliminated almost every noun in
semantic-v1. It reused the exact v1 MiniLM evidence and replaced the absolute
zero cutoff with a five-fold, role-specific, cross-conformal lower-tail test.
The new test rejected 4.78% of nouns and 4.82% of verbs, close to its declared
5% tail, rather than rejecting 99.44% of nouns as v1 did.

The experiment nevertheless stopped at a later unchanged invariant. The
capacity-constrained eight-cluster construction continued to find words with
assigned-centroid versus best-alternative cosine margin below `0.03` after all
five permitted exclusion/recluster passes. Forty-seven nouns and eighteen
verbs still failed on the terminal pass. The benchmark therefore has a strict
failure audit but no semantic catalog, partition, sample report, GPU training
run, checkpoint, publication, or sealed-test result.

The generated evidence is available as
[`audit.md`](../data/tinyworlds-p-semantic/catalog/v2/failures/23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25/audit.md)
and standalone
[`audit.html`](../data/tinyworlds-p-semantic/catalog/v2/failures/23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25/audit.html).

## Why the role gate changed

Semantic-v1 computed, for each word and construction context,

`margin = context·declared-role-anchor - context·opposite-role-anchor`

and required the 10th percentile of those margins to exceed zero. The zero
boundary was principled only if MiniLM and the six sentence templates were
already centered between noun and verb evidence. They were not: noun q10
scores had median `-0.045440` and only six of 1,066 nouns exceeded zero. The
encoder is a sentence-semantic model rather than a calibrated part-of-speech
classifier, so the result primarily measured a role/template offset.

V2 retains the raw margin, all six templates, exact contexts, and q10 summary,
but calibrates the offset against other words carrying the same declared role.
For role `r` and word `w`, the fold is

`SHA256("tinyworlds-p-semantic-v2" || 0x00 || "role-calibration-fold-v1" || 0x00 || r || 0x00 || w) mod 5`.

For a word in fold `f`, the reference set is every sufficient-context word of
the same role outside `f`. Its lower-tail conformal value is

`p = (1 + count(reference_score <= word_score)) / (reference_count + 1)`.

The word passes exactly when `p > 0.05`. Under exchangeability of same-role
word scores, the added one gives the usual finite-sample conformal
calibration; inclusive ties are conservative, every word is assessed out of
fold, and no 20% calibration panel is thrown away. References are unweighted because
the screened unit is a vocabulary word; archive token mass is used later by
the capacity constraints. The procedure sees only permanent construction
evidence and released role metadata. It cannot read model loss, a partition,
a checkpoint, or sealed test.

## Frozen identities and choices

V2 reused encoder evidence
`efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`
without rerunning MiniLM. That artifact binds:

- the 1,608,001,638-byte pinned TinyStories archive, SHA-256
  `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`;
- the permanent modulo-20 construction slice of 247,629 duplicate groups and
  47,172,075 active tokens;
- 898,327,086 non-construction active tokens;
- 195,492 float32, mean-pooled, L2-normalized anchor/context embeddings;
- `sentence-transformers/all-MiniLM-L6-v2` revision
  `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`, encoder identity
  `1101bb824cee453866d6dcd2b489b29ad2c55b20de5bbaceda67f38206a21502`.

The v2 intervention is limited to role calibration and a deterministic
feasibility repair for indivisible mass assignment. V1 never reached
clustering, so it did not expose that its continuous remaining-mass check
could leave two underweight clusters for one final word. At such a dead end,
v2 considers moving exactly one prior word while placing the current word. It
chooses the feasible combination with the best fixed-centroid cosine objective
and v2 hash ties. The 90--110% bounds are unchanged; absence of a one-move
repair remains a hard failure.

The semantic vector remains the equal normalized combination of the declared-
role anchor centroid and archive-context centroid. The context minimum `32`,
two-means silhouette maximum `0.20`, cluster count `8`, centroid iteration
limit `100`, cluster margin minimum `0.03`, exclusion/recluster limit `5`,
minimum cluster sizes `32` nouns and `12` verbs, centroid-pair cosine maximum
`0.90`, and joint retained-token minimum `40%` are all unchanged.

## Calibrated role result

All 1,460 words had at least 32 exact contexts, so all received a calibrated
score. Held-out-fold reference sizes and empirical rejection cutoffs were:

| Role | Fold | References | Raw-q10 rejection cutoff |
|---|---:|---:|---:|
| Noun | 0 | 841 | -0.079342 |
| Noun | 1 | 883 | -0.079342 |
| Noun | 2 | 848 | -0.078293 |
| Noun | 3 | 837 | -0.078293 |
| Noun | 4 | 855 | -0.079907 |
| Verb | 0 | 315 | -0.044263 |
| Verb | 1 | 294 | -0.049396 |
| Verb | 2 | 326 | -0.044986 |
| Verb | 3 | 315 | -0.044784 |
| Verb | 4 | 326 | -0.044263 |

The pre-clustering screens were:

| Role | Input words | Calibrated role outliers | Multi-sense exclusions | Cluster candidates |
|---|---:|---:|---:|---:|
| Noun | 1,066 | 51 | 37 | 978 |
| Verb | 394 | 19 | 10 | 365 |

This is the intended calibration outcome: the declared-role reference tail is
removed without treating a shared negative encoder offset as evidence against
the entire noun vocabulary.

## Fixed-grid result

The complete boundary trace was:

| Role | Pass | Input words | Margin failures | Minimum margin | Margin q10 | Median margin |
|---|---:|---:|---:|---:|---:|---:|
| Noun | 0 | 978 | 259 | -0.119019 | 0.016030 | 0.057796 |
| Noun | 1 | 719 | 137 | -0.235075 | -0.001321 | 0.070130 |
| Noun | 2 | 582 | 103 | -0.193556 | 0.014708 | 0.074137 |
| Noun | 3 | 479 | 54 | -0.171637 | 0.026136 | 0.083475 |
| Noun | 4 | 425 | 55 | -0.229771 | 0.019976 | 0.096328 |
| Noun | 5 | 370 | 47 | -0.229158 | 0.015653 | 0.096746 |
| Verb | 0 | 365 | 106 | -0.106542 | 0.009449 | 0.052815 |
| Verb | 1 | 259 | 39 | -0.135017 | 0.021577 | 0.075801 |
| Verb | 2 | 220 | 23 | -0.149802 | 0.026796 | 0.082961 |
| Verb | 3 | 197 | 21 | -0.145813 | 0.028912 | 0.096121 |
| Verb | 4 | 176 | 18 | -0.129740 | 0.027093 | 0.089011 |
| Verb | 5 | 158 | 18 | -0.149717 | 0.027432 | 0.108304 |

Pass zero is the initial clustering; passes one through five follow the five
allowed exclusions. A valid result requires zero failures on the final row for
each role. Instead, failures persisted and sometimes increased after
rebalancing, which is expected when fixed mass capacities force assignments
away from an unconstrained nearest centroid.

Across all passes, 655 distinct nouns and 225 distinct verbs were flagged.
The 323 nouns and 140 verbs not flagged by the end are not a valid catalog:
they were never reclustered after removing the terminal failures. As a
diagnostic, their noun-by-verb intersection covers 98,322,186 active tokens,
only 10.945032% of the 898,327,086-token non-construction archive. Therefore
extending the pass count would still fail the independent 40% retained-mass
gate even before topology and control feasibility.

## Published failure evidence

The immutable failure artifact is:

- directory:
  `data/tinyworlds-p-semantic/catalog/v2/failures/23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25/`
- failure SHA-256:
  `23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`
- calibration payload SHA-256:
  `64df24c65dc73e9b8de06bc2cd3e3106b73b3f9e9da12f5fd27997e3ed89af6c`
- boundary trace SHA-256:
  `d53b9e20ac18b49aef989c59d985b883e6269304467d2e7210e5e64f69e8b0f9`
- word ledger SHA-256:
  `3bd6870048492b3ab814608d403da737a7cf300026e282eab943535dc4113409`
- exact-context payload SHA-256:
  `e35d83bf41ab57b988a3256cc9a29e6862fdb444fdade5ba2d586349a7cfa43a`
- total size: approximately 35 MiB.

The bundle contains all 1,460 word records, raw margins, folds, reference
counts, conformal values, fold cutoffs, silhouettes, final failure
dispositions, candidate vectors, all pass-level cluster masses and margin
summaries, representative exact archive contexts, complete role-pair masses,
Markdown, standalone HTML with PCA, and a strict file tree. The loader
recomputes fold assignment and conformal values from the ledger rather than
trusting the persisted values alone.

## Implementation and verification

The v2 implementation adds:

- [`v2_contracts.py`](../src/apm/data/text/tinyworlds_p_semantic/v2_contracts.py)
  for the isolated benchmark/config/artifact contracts;
- [`role_calibration.py`](../src/apm/data/text/tinyworlds_p_semantic/role_calibration.py)
  for deterministic folds and cross-conformal scores;
- [`v2_catalog.py`](../src/apm/data/text/tinyworlds_p_semantic/v2_catalog.py)
  for screening, repaired capacity assignment, traced clustering,
  content-addressed publication, and strict replay;
- [`v2_audit.py`](../src/apm/data/text/tinyworlds_p_semantic/v2_audit.py)
  for exhaustive Markdown and standalone HTML;
- [`prepare_tinyworlds_p_semantic_v2.py`](../scripts/prepare_tinyworlds_p_semantic_v2.py)
  for strict cached-evidence execution.

Synthetic tests use a CPU 3-by-3 grid to cover held-out-fold isolation,
conformal order invariance, content-addressed success and failure publication,
byte-identical rebuilds, exhaustive audit presence, strict calibration replay,
and tamper rejection. Existing v1 catalog tests continue to pass, confirming
that v2 tie namespaces and assignment repair do not reinterpret v1 artifacts.
The final focused semantic/archive and shared GPT-Neo/checkpoint regression
groups pass all 142 tests, and every semantic module and runner compiles.

The real run took approximately 16 seconds for cached loading, calibration,
screening, all twelve role/pass clusterings, and publication. A second runner
invocation strictly authenticated and returned the same failure identity in
approximately four seconds. A separate build under an automatically removed
temporary root reproduced the same failure SHA and all nine bundle files byte
for byte. No CUDA work was needed, and no partition, language-model input,
checkpoint, or sealed-test file was opened.

## Version consequence

Semantic-v2 has answered the calibration question: an absolute zero was the
wrong role baseline, and cross-conformal calibration restores nearly the
entire vocabulary. It has also isolated the next obstacle: an assigned-cluster
margin of `0.03` is incompatible with repeated capacity balancing at the
required retained mass. This version stops. Any experiment with a different
semantic vector, globally optimized capacity objective, unconstrained
nearest/second-nearest margin, softer boundary evidence, changed pass budget,
or changed retained-mass floor must be preregistered as
`tinyworlds-p-semantic-v3`; it cannot reinterpret v2 or open v2's nonexistent
downstream stages.
