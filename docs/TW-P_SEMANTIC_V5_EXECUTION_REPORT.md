# TinyWorlds-P Semantic-v5 Execution Report

Date: 2026-07-23

## Outcome

Semantic-v5 successfully fixed the imbalance that stopped semantic-v4. It
selected five experimental conditions containing similar amounts of text, and
all five passed the unchanged 10% balance rule.

The build then stopped for a different reason. Condition B's validation set
needed 2,314 distinct stories for its column comparison, but only 1,511 valid
stories were available after the split and global no-reuse rules were applied.
The shortage was 803 stories. Because that comparison could not be constructed
as promised, semantic-v5 has no usable partition and cannot proceed to
training.

The authenticated failure is
`090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`.
Its primary and independent rebuilds are byte-identical.

## What version 5 changed

Version 5 made one change to the frozen semantic-v4 partition rule. Version 4
ranked all layouts by semantic coherence and checked text balance only after it
had selected a winner. Version 5 first removed any layout whose five conditions
did not all lie within 10% of that layout's median text amount. It then applied
the same semantic ranking to the remaining layouts.

Everything else remained fixed. In particular, version 5 reused:

- the exact semantic-v4 catalog, containing 790 nouns and 284 verbs in eight
  clusters per role;
- the exact semantic-v4 partition failure as parent evidence;
- the TinyStories archive and TinyStories-8M tokenizer;
- the 80/10/10 experimental-condition splits and 96/2/2 held-in splits;
- complete duplicate-story groups, with no splitting or replacement;
- the same source, feature, adjective, and length controls;
- global non-reuse of comparison stories; and
- the sealed-test boundary.

The catalog identity was
`ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`.
The parent partition-failure identity was
`37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`.

## Reproduced input data

The real run independently read the pinned 1,608,001,638-byte archive. It
reconstructed 4,967,871 source records and exactly reproduced the version-4
exclusions:

- 2,520,317 retained duplicate groups, containing 479,183,203 active tokens;
- 247,629 reserved semantic-construction groups, containing 47,172,075 active
  tokens; and
- 2,198,121 groups containing at least one excluded noun or verb, containing
  419,143,883 active tokens.

Here, an active token is one model-scored piece of text. The token totals are
used to compare the amount of training and evaluation text; they are not model
losses and did not depend on a trained checkpoint.

The complete version-4 topology audit was also recomputed from the archive.
All 28,224 layout measurements matched the saved parent evidence exactly.
Twenty-two layouts passed the version-5 balance requirement.

## Selected balanced layout

The strongest balanced layout used the following noun-cluster and verb-cluster
pairs for conditions A through E:

| Condition | Noun cluster | Verb cluster | Active tokens | Duplicate groups |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 4 | 9,899,869 | 52,311 |
| B | 4 | 4 | 8,829,612 | 46,281 |
| C | 4 | 6 | 8,742,369 | 47,344 |
| D | 3 | 6 | 10,104,204 | 53,842 |
| E | 2 | 0 | 9,357,468 | 49,605 |

The median was 9,357,468 active tokens. The permitted interval was
8,421,721.2 through 10,293,214.8, so all five conditions passed. The selected
layout's semantic-dispersion score was `0.329787101192787`. This is the same
balanced diagnostic layout recorded by semantic-v4, but version 5 selected it
under a rule registered before the version-5 run. Version 4's different winner
and failure remain unchanged.

## Why control construction stopped

Each experimental condition needs two comparison arms. A row comparison keeps
the noun cluster fixed and changes the verb cluster. A column comparison keeps
the verb cluster fixed and changes the noun cluster. These comparisons let the
evaluation separate the effects of the two word roles.

Condition B uses noun cluster 4 and verb cluster 4. Its validation split
contained 4,628 story groups, so its comparison set needed 2,314 row stories
and 2,314 column stories. The column stories had to satisfy all of the
following basic requirements:

- they belonged to the held-in validation split;
- they used verb cluster 4 but not noun cluster 4;
- they were not one of the five experimental cells; and
- no earlier condition had already claimed them as a comparison story.

After the registered comparison-allocation order had reserved stories for E,
A, and C, only 1,511 column candidates remained for B. That is 65.3% of the
2,314 required stories. The code stopped at this raw count check, before it
could attempt the finer source, feature, adjective, length, or token matching.
Therefore, loosening one of those matching tolerances would not repair this
particular shortage.

The earlier topology screen's control-capacity check was a coarse check over
the unsplit archive. It correctly established that comparison stories existed,
but it did not guarantee enough stories after the held-in split and global
no-reuse reservations. The full allocation exposed that stricter limitation.

## Authenticated failure evidence

The primary failure is stored at
[`data/tinyworlds-p-semantic/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`](../data/tinyworlds-p-semantic/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11).
Its human-readable audit is
[`audit.md`](../data/tinyworlds-p-semantic/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11/audit.md).

The failure binds:

- the archive, tokenizer, semantic catalog, and parent failure identities;
- every frozen partition setting and the version-5 seed identity
  `b94b454cec00500539ec2655dc382b52c4ee30c287559df68c2d36924395bcff`;
- the complete exclusion counts and balance-first topology selection;
- the structured 1,511-versus-2,314 control shortage; and
- the complete assignment-ledger SHA-256
  `92e9166c57915acfb983cdecad030c7a17a805b18999fab173edd4125d32b6bd`.

The artifact embeds and strictly authenticates the exact semantic-v4 catalog
and semantic-v4 partition failure. Its authenticated tree SHA-256 is
`3898637e610ca8a3ed1c061f175011cad7e828163527b6c092dd1e7e50969966`.

## Independent reproduction

The primary assignment ledger used 50,000-record external-sort batches. A
second run started again from the pinned archive and used 37,000-record batches.
The second run independently reproduced:

- the archive and exclusion counts;
- every version-4 layout measurement;
- the same version-5 balanced layout;
- the complete 3.95 GB assignment ledger, including its SHA-256;
- the same condition-B column shortage; and
- every byte of the 54 MB authenticated failure directory.

The independent failure is stored at
[`data/tinyworlds-p-semantic/rebuild-verification/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`](../data/tinyworlds-p-semantic/rebuild-verification/v5/failures/090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11).
A recursive directory comparison found no differing file. Both strict loaders
authenticated the same failure and tree identities, and direct hashing of the
two external assignment ledgers reproduced the bound assignment identity.

## Verification and scientific boundary

The final focused CPU suite passed 51 tests. It covers the archive code, prior
semantic catalog versions, old and new partition behavior, paired statistics,
training helpers, version separation, balance-first selection, frozen parent
settings, and structured control-shortage parsing. All new version-5 modules
and the fixed runner also compile in the pinned NumPy 1.26.4 environment.
The generated HTML audit parses successfully and contains no external resource
reference.

Semantic-v5 is a terminal failed construction, not a failed language-model
experiment. It demonstrates that balance-first selection solves the specific
version-4 imbalance, but that this selected layout cannot support the promised
comparison design at the frozen split sizes and no-reuse rule. Because the
comparison is incomplete, training would not produce the registered test, so
training was not started.

No semantic-v5 partition, sample report, GPU preflight, optimizer update,
checkpoint, or sealed-test result exists. Choosing another balanced layout,
checking exact split-level comparison feasibility before semantic ranking,
reducing the evaluation split, or permitting story reuse would each be a new
scientific intervention. Any such change must be registered as a later version
and may not reinterpret semantic-v5.
