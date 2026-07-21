# TinyWorlds-P v1 Canonical Source Audit

The canonical 8×8 preparation stopped at the immutable source-coverage gate on
2026-07-20. No partition tree was published and no calibration or base training
was started. The machine-readable result is
[`TW-P_SOURCE_AUDIT.json`](TW-P_SOURCE_AUDIT.json).

## Pinned inputs

- `TinyStories-train.txt`: 1,924,281,556 bytes, SHA-256
  `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`.
- `TinyStories_all_data.tar.gz`: 1,608,001,638 bytes, SHA-256
  `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
- Both source contracts name upstream revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- GPT-2 BPE tokenizer: the pinned 50,257-token local artifact.

The fixed runner used 16 isolated `spawn` workers. Corpus normalization and
tokenization finished in 5.4 minutes, metadata normalization in another 3.4
minutes, and the external merge stopped at 9.6 minutes total. Its retained
local evidence directory is
`data/tinyworlds-p/work/prepare-v1-it_f88wo/`; it includes all sorted runs, the
1.3 GiB joined-groups stream, and `join-audit.json`.

## Gate result

| Measure | Numerator | Denominator | Measured | Minimum | Result |
|---|---:|---:|---:|---:|:---:|
| Hash-matched token mass | 370,690,767 | 469,429,276 | 78.966265% | 95% | Fail |
| Role-classified matched token mass | 370,578,982 | 370,690,767 | 99.969844% | 95% | Pass |
| Eligible corpus token mass | 370,578,973 | 469,429,276 | 78.942450% | 90% | Fail |

The corpus contains 2,119,489 raw occurrences in 1,799,248 normalized duplicate
groups. Assignment eligibility broke down as follows:

| Status | Groups | Raw occurrences | Active tokens |
|---|---:|---:|---:|
| Eligible | 1,371,109 | 1,609,076 | 370,578,973 |
| Unmatched metadata | 427,714 | 509,915 | 98,738,509 |
| Unclassifiable metadata | 424 | 497 | 111,785 |
| Conflicting metadata | 1 | 1 | 9 |

Role recovery is therefore not the limiting factor. Exact story identity is.

## Mismatch diagnosis

Unmatched occurrences are spread throughout the corpus rather than isolated to
one appended segment. In every complete 100,000-occurrence source-index block,
roughly 19.8%–27.0% of occurrences were unmatched.

An exhaustive diagnostic scan read all 4,967,871 released metadata records. A
representative unmatched corpus occurrence had source index `1,866,356`, raw
SHA-256
`3e4e42f51f01aaeaf056df1069f512aaa80c574a594d08066959d4cdab78ee6a`,
and normalized SHA-256
`000001545c829f0986b46c23ac2a5afeb12f2d70ca7ebf7da64c7f854f6c62a6`.
Its opening, “Once upon a time there was a bald man,” occurs in released record
`archive:./data22.json:18405:e371cff1205b09f4527fe96f016547ff1823789b8b74fcb606bc4dc310ccc42e`,
but that record immediately continues with a different generated story. The
exact corpus story is absent from the archive. Normalization tests, exact
matched controls, and the uniform source-index distribution rule out a quote,
whitespace, casing, boundary, or isolated-version-tail explanation.

## Decision

The predeclared gates are unchanged. Excluding unmatched text prevents
untracked held-out conjunctions from leaking into the base, but the resulting
78.94% eligible mass does not satisfy the benchmark contract. Lowering the
threshold, joining stories approximately, substituting another corpus, or
inferring recipes from prose would change the approved experiment and was not
done. TinyWorlds-P v1 therefore stops at source audit until a metadata source
covering the pinned corpus is supplied or the benchmark contract is explicitly
revised.
