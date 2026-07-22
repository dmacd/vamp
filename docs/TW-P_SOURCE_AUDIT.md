# Historical TinyWorlds-P v1 Canonical Source Audit

## Status: historical and superseded

Effective 2026-07-21, TinyWorlds-P no longer uses
`TinyStories-train.txt` or any other published TinyStories text aggregate. The
pinned `TinyStories_all_data.tar.gz` archive is the sole story universe, and
all base, world, control, validation, and sealed-test assignments are drawn
directly from eligible archive entities. There is no current corpus/archive
join, unmatched-corpus category, hash-match coverage gate, or combined
corpus-coverage gate.

Everything below documents why the abandoned corpus-intersection design was
rejected. Its counts remain useful release-provenance evidence, but they do not
define the current TinyWorlds-P source, partition, eligibility, calibration, or
publication contract.

The original-contract canonical 8×8 preparation stopped at the immutable
source-coverage gate on 2026-07-20. At that point no partition tree had been
published and no calibration had started. This document preserves that source
diagnosis and the later explicitly approved 75%/95%/75% contract revision. The
revised build subsequently published a partition; its training outcome is
recorded separately in
[`TW-P_CALIBRATION_AUDIT.md`](TW-P_CALIBRATION_AUDIT.md). The machine-readable
original gate result is [`TW-P_SOURCE_AUDIT.json`](TW-P_SOURCE_AUDIT.json).

## Why the abandoned source join existed

`TinyStories-train.txt` contains story text and document separators only. It
does not contain a prompt, ingredient words, feature labels, source model, or a
record ID. `TinyStories_all_data.tar.gz` contains JSON records that pair a
generated story with those fields. Under the abandoned design, TinyWorlds-P
needed the released noun and verb for each corpus story in order to withhold
noun-bucket × verb-bucket cells, so it attempted to attach an archive record to
each corpus story.

The release provides no shared identifier between these two files. A match
therefore means only:

```text
SHA256(normalize(complete corpus story))
    == SHA256(normalize(complete archive record["story"]))
```

Normalization applies NFKC, case folding, straight-quote canonicalization,
Unicode whitespace collapse, and edge trim. It does not compare prompts,
prefixes, words, summaries, or semantic similarity. A matched archive record
supplies the prompt, words, features, source model, and provenance for exactly
the same story. An unmatched corpus story is one whose complete normalized
identity occurs in none of the 4,967,871 archive story fields. No metadata is
assigned to such a story.

This join was expected to work because the
[official dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
describes the archive as a superset of the training stories with metadata and
prompts. The measured release artifacts do not satisfy that description under
complete-story identity.

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

## Original gate result

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

Role recovery is therefore not the limiting factor. Complete story identity is.

## Source-model and held-out diagnosis

The archive contains 2,222,541 GPT-3.5 records and 2,745,330 GPT-4 records. The
exactly joined corpus occurrences divide into these source cohorts:

| Archive provenance | Groups | Corpus occurrences | Active tokens |
|---|---:|---:|---:|
| GPT-4 | 938,003 | 1,121,191 | 284,495,159 |
| GPT-3.5 | 433,530 | 488,382 | 86,195,599 |
| Both labels | 1 | 1 | 9 |
| No exact archive story | 427,714 | 509,915 | 98,738,509 |

The official dataset card says that the GPT-4-only V2 train aggregate contains
all GPT-4 examples from the original train aggregate. An exact normalized-hash
scan tested every unmatched group against all locally pinned aggregate files:

| Comparison source | Documents scanned | Unmatched groups found | Unmatched tokens explained |
|---|---:|---:|---:|
| `TinyStoriesV2-GPT4-train.txt` | 2,717,495 | 0 | 0 |
| `TinyStoriesV2-GPT4-valid.txt` | 27,630 | 0 | 0 |
| `TinyStories-valid.txt` | 21,990 | 1,500 | 274,327 |

There is a real original-train/original-validation overlap, but it explains
only 0.278% of unmatched token mass. Held-out mixing is not the cause of the
21.034% unmatched corpus-token mass. The zero overlap with both GPT-4-only
aggregates, together with the release's stated GPT-4 subset relationship,
strongly places the missing mass on the original GPT-3.5 side. That source
attribution is an inference from the published subset contract because the
unmatched text file itself carries no source labels.

The source-index structure makes the release defect clearer. The 509,915
unmatched occurrences form 1,696 contiguous corpus runs. Of those runs, 1,689
have lengths exactly divisible by 100 and contain 507,900 occurrences, or
99.605% of all unmatched occurrences. The dominant run lengths are:

| Run length | Number of runs | Occurrences |
|---:|---:|---:|
| 300 | 1,349 | 404,700 |
| 100 | 194 | 19,400 |
| 600 | 102 | 61,200 |
| 400 | 30 | 12,000 |

This is generation-batch or assembly-shard structure. It is incompatible with
a quote-normalization bug, random story-level loss, or an isolated held-out
tail. The evidence establishes that the published train aggregate contains
whole source batches whose exact generated stories are absent from the
published metadata archive. It does not establish whether those batches were
omitted, overwritten by alternate completions, or assembled from an unreleased
generation ledger.

The release history is also not a synchronized single snapshot. The archive
was uploaded on 2023-05-17 while the train object was
`fcf1ba64e3cfb88e0fae7f24edcac0a868ccb63559995058a7f6be2fa523b5f0`
and 1,959,863,806 bytes. The train and validation objects were replaced on
2023-05-18 in
[commit `5485261`](https://huggingface.co/datasets/roneneldan/TinyStories/commit/5485261731eaac25dd8e5ebbc3839d0a9870b185)
under “Fix encoding problem”; the archive was not replaced. The current train
object is 35,582,250 bytes smaller. However, the representative unmatched story
below has the same normalized SHA-256 in the pre-fix train object, so that
replacement did not create this mismatch. At least this missing batch predates
the encoding repair.

## Representative complete-story evidence

The corpus occurrence is `train:001866356`, source index `1,866,356`, byte
offset `1,693,478,347`, and byte length `1,010`. Its raw SHA-256 is
`3e4e42f51f01aaeaf056df1069f512aaa80c574a594d08066959d4cdab78ee6a`;
its normalized SHA-256 is
`000001545c829f0986b46c23ac2a5afeb12f2d70ca7ebf7da64c7f854f6c62a6`.
It is the 299th story in the 300-story unmatched run from source indices
`1,866,058` through `1,866,357`, between an archive-matched GPT-3.5 run and an
archive-matched GPT-4 run.

There is no released metadata record for this complete story. Its prose uses
surfaces such as “box,” “sigh,” and “bald,” but inferring a prompt recipe from
story prose would violate the benchmark contract and need not recover the
actual prompt.

Corpus story:

> Once upon a time there was a bald man. He had a big, green box. The box was
> shiny and it had a big lock on it. He looked at the box and sighed. Inside
> the box he kept all his most precious things. Every day, he would look at the
> box, open it, and sigh.
>
> One day, a little girl came along. She saw the bald man with the box. She
> asked him what was in it. He said nothing and just sighed again. The girl
> asked again but he just kept sighing.
>
> The girl was curious. She wanted to see what was inside the box. She asked
> the bald man over and over again, but he still said nothing. Finally, he
> opened the box and showed the girl the contents. Inside was a big furry teddy
> bear. The girl was so excited!
>
> The man smiled at the girl and the girl thanked him. The bald man was glad he
> was able to make the girl happy. From that day on, he opened his box every day
> to show the girl something new. It was always special. Every time he opened
> his box, he would look at the girl and smile. He no longer sighed, but smiled.

An exhaustive archive scan found one record with the same normalized opening
sentence, not the same story. This is a same-opening, different-completion
candidate, never a join match. Its released metadata is:

```json
{
  "record_id": "archive:./data22.json:18405:e371cff1205b09f4527fe96f016547ff1823789b8b74fcb606bc4dc310ccc42e",
  "source": "GPT-4",
  "source_member": "./data22.json",
  "source_index": 18405,
  "content_sha256": "e371cff1205b09f4527fe96f016547ff1823789b8b74fcb606bc4dc310ccc42e",
  "normalized_story_sha256": "ce694e510f67565299233b31651b89f8fa4bbf2aa5ea1f853f7741bbb29046be",
  "instruction": {
    "prompt": "Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb \"believe\", the noun \"parade\" and the adjective \"bald\". Remember to only use simple words!\n\nPossible story:",
    "words": ["believe", "parade", "bald"],
    "features": []
  },
  "summary": "A bald man organizes a parade with the help of his friends and is cheered by everyone, making him believe he can do anything."
}
```

Candidate archive story:

> Once upon a time there was a bald man. He was very happy, and he believed he
> could do anything.
>
> One day, the bald man wanted to have a parade. He asked all his friends to
> help him. They made big signs and balloons to decorate the parade.
>
> The day of the parade came. Everyone was so excited. The bald man waved to
> all the people and smiled. Everyone cheered and clapped for him.
>
> At the end of the parade, the bald man was so happy. He hugged all his
> friends and thanked them for helping him.
>
> The bald man believed that he could do anything, and now everyone believed it
> too.

The stories have different normalized SHA-256 values and diverge immediately
after the first sentence. The candidate's `believe × parade × bald` recipe
cannot be attached to the corpus's box story.

## Superseded interim contract revision

The original 95% hash-match and 90% eligible-coverage gates correctly stopped
the first build and made the release mismatch visible. After reviewing the
complete audit, the user explicitly approved ignoring archive-missing stories
and proceeding with the exact metadata-bearing subset. Under that interim
contract, the minimum hash-match and eligible-coverage gates were 75% and the
role-coverage minimum was 95%. The measured 78.966% match, 99.970% role
classification, and 78.942% eligible mass passed those interim gates.

Unmatched, unclassifiable, and conflicting stories remain excluded from both
base and worlds. Prefix joining, assigning recipes from prose, treating
alternate completions as identical, and silently admitting untracked text
remain prohibited. That run proceeded with 370,578,973 eligible tokens through
bucket, cell, split, control, and shard construction.

That interim 75%/95%/75% corpus-intersection contract was itself superseded on
2026-07-21 by the archive-only decision at the top of this document. It must
not be implemented, resumed, or treated as a current gate.
