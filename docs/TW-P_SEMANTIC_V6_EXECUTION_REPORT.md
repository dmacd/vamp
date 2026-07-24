# TinyWorlds-P Semantic-v6 Execution Report

Date: 2026-07-23

## Outcome

Semantic-v6 succeeded. It found a semantically strong five-world layout that
also has enough closely matched comparison stories in every validation and
test condition. The partition is therefore complete and can support the
planned language-model experiment.

The selected layout was the second-best layout under the frozen semantic
ranking. The best-ranked layout could not supply one required comparison set,
while the second-ranked layout completed every comparison. Seventeen of the
22 balanced layouts were feasible, so the successful result does not depend
on a single unusually lucky alternative.

The partition identity is
`3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`.
A complete second build, using a different external-sort batch size,
reproduced all 167 authenticated files byte for byte. The sealed test was not
opened, and no model training was started.

## What version 6 changed

Semantic-v5 first selected the best balanced layout and only then attempted to
find its comparison stories. That layout failed because world B did not have
enough validation stories for its column comparison.

Version 6 makes one registered change. It runs the real validation/test split
and complete comparison-story allocation for every balanced layout before it
applies the unchanged semantic ranking. A layout is eligible only when all
five worlds complete both comparison arms in validation and test. This check
retains the fixed comparison order, global story non-reuse, token matching,
and source, feature, adjective, and length limits.

Everything else remains fixed. Version 6 uses:

- the exact semantic-v4 catalog, containing 790 nouns and 284 verbs in eight
  clusters per role;
- semantic-v5's authenticated terminal failure as parent evidence;
- the pinned TinyStories archive and TinyStories-8M tokenizer;
- 80/10/10 world splits and 96/2/2 held-in splits;
- complete duplicate-story groups, with no splitting or replacement;
- the same row and column comparison definitions and matching limits;
- one-to-one world/comparison pairings; and
- the sealed-test boundary.

The catalog identity is
`ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`.
The parent semantic-v5 failure is
`090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`.
No model loss, checkpoint, or sealed-test result entered layout selection.

## Reproduced source data

Both real builds read the pinned 1,608,001,638-byte archive and reconstructed
4,967,871 source records. They reproduced the semantic-v4 exclusions exactly:

- 2,520,317 retained unique-story groups, containing 479,183,203 scored
  tokens;
- 247,629 groups reserved for semantic construction, containing 47,172,075
  scored tokens; and
- 2,198,121 groups containing at least one excluded noun or verb, containing
  419,143,883 scored tokens.

A unique-story group keeps identical normalized stories together. A scored
token is one tokenizer piece on which model loss would later be measured.
These counts describe data volume; they are not model results.

The builds also reproduced all 28,224 parent layout measurements. Twenty-two
layouts passed the unchanged rule requiring all five worlds to lie within 10%
of that layout's median text amount.

## Exact feasibility results

All 22 balanced layouts received a real split and full comparison allocation.
Seventeen passed. Five failed because world B's validation column arm did not
have enough distinct stories after the fixed split and global non-reuse rules:

| Semantic rank | Available stories | Required stories | Shortage |
| ---: | ---: | ---: | ---: |
| 0 | 1,649 | 2,314 | 665 |
| 8 | 1,584 | 2,314 | 730 |
| 9 | 986 | 2,615 | 1,629 |
| 14 | 889 | 2,615 | 1,726 |
| 21 | 1,065 | 2,691 | 1,626 |

The other ranks—1 through 7, 10 through 13, and 15 through 20—completed every
comparison. Each candidate record stores the hash of its complete split stream
and either the hash of its successful comparison family or its exact failure
reason. The selected rank-1 split hash is
`b090e78fbfa80a26ea4dd4403612c199e06c634c1b31d68a4edec0055aa55999`,
and its comparison-family hash is
`583a9cff169564d6043ea51cc3a4ab92bee6b5a5d14a5e74297ad3845fdec976`.

## Selected layout

The selected worlds use these fixed noun-cluster and verb-cluster coordinates:

| World | Noun cluster | Verb cluster | Scored tokens | Unique-story groups |
| --- | ---: | ---: | ---: | ---: |
| A | 2 | 4 | 6,136,097 | 32,210 |
| B | 7 | 4 | 5,873,159 | 30,715 |
| C | 7 | 6 | 5,921,676 | 31,656 |
| D | 2 | 6 | 6,114,634 | 32,871 |
| E | 3 | 2 | 5,440,146 | 28,201 |

The median is 5,921,676 scored tokens. The permitted interval is
5,329,508.4 through 6,513,843.6, so every world passes the frozen 10% rule.
The selected layout's semantic-dispersion score is `0.33029539315764556`.
The infeasible rank-0 layout scored `0.329787101192787`, so the feasibility
screen moved only one position down the registered semantic ranking.

## Splits, comparisons, and pairing

The held-in training set contains 2,270,077 unique stories and 431,711,495
scored tokens. Its validation set contains 47,294 stories and 8,992,992 tokens;
its sealed test contains 47,293 stories and 8,993,004 tokens.

The five world splits are:

| World | Training stories | Validation stories | Test stories |
| --- | ---: | ---: | ---: |
| A | 25,771 | 3,220 | 3,219 |
| B | 24,573 | 3,070 | 3,072 |
| C | 25,328 | 3,164 | 3,164 |
| D | 26,300 | 3,285 | 3,286 |
| E | 22,564 | 2,818 | 2,819 |

All row and column comparisons were constructed for every world in validation
and test. The allocation then produced 31,117 deterministic one-to-one
world/comparison pairings. A fresh publication-path allocation reproduced the
selected candidate's recorded split and comparison hashes before any shard was
written.

## Independent reproduction

The primary build used 50,000-record external-sort batches. The independent
build started again from the pinned archive and used 37,000-record batches. It
reproduced:

- all archive and exclusion counts;
- all 28,224 parent layout measurements;
- all 22 exact feasibility outcomes and failure reasons;
- rank 1 as the selected layout;
- the selected split, comparison, and pairing evidence;
- the partition identity; and
- every byte of the 11 GB, 167-file partition directory.

Both builds strictly reloaded the complete result. Their authenticated tree
SHA-256 is
`b5ba1ce33d1cad7eb00bba0b6eec35e2b94c3a6b997a20149081cc61c862279d`.
A direct recursive directory comparison found no differing file.

The primary partition is stored at
[`data/tinyworlds-p-semantic/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`](../data/tinyworlds-p-semantic/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa).
The independent copy is stored at
[`data/tinyworlds-p-semantic/rebuild-verification/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`](../data/tinyworlds-p-semantic/rebuild-verification/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa).

## Validation-only sample report

The authenticated pre-training report covers exactly 16 conditions: the
held-in validation set, all five validation worlds, and both validation
comparison arms for all five worlds. Every example includes its exact story
bytes, archive provenance, tokenizer length, cluster coordinates, and hashes.
It also includes the complete noun and verb cluster inventories.

The report identity is
`b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579`.
Its Markdown version is
[`sample-report.md`](../data/tinyworlds-p-semantic/sample-reports/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa/b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579/sample-report.md),
and its self-contained HTML version is
[`sample-report.html`](../data/tinyworlds-p-semantic/sample-reports/v6/3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa/b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579/sample-report.html).
The HTML parses successfully and has no external resource reference. The
report explicitly records `sealed_test_opened: false`.

## Verification and scientific boundary

The final focused CPU suite passed 57 tests. It covers semantic selection,
complete-candidate measurement, all-candidate failure, frozen parent and
partition settings, actual split-level comparison success and shortage, strict
version rejection, archive reconstruction, leakage, pairing, tampering, and
prior semantic versions. The new modules and fixed runner compile in the pinned
environment, and the semantic-v6 files pass the scoped whitespace check.

Semantic-v6 is a successful data construction, not yet a language-model result.
No GPU preflight, optimizer update, checkpoint, validation loss, semantic-gap
decision, or sealed-test evaluation exists for this version. The next stage
must first bind training and resume artifacts strictly to this partition, then
measure the real GPU runtime before a fresh seed-zero calibration is started.
