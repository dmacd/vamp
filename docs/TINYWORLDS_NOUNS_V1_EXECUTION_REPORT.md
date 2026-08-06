# TinyWorlds Noun-Overlap v1 — Implementation and Review-Gate Report

Date: 2026-08-05

## Outcome

The noun-overlap experiment is implemented and passed its required manual
boundary. The pinned archive scan and review-packet publication completed, and
the runner initially exited with status 3 at the noun-approval gate without
constructing the model partition, discovering a GPU, training parameters,
evaluating stories, or calling OpenRouter. The user subsequently approved the
exact reviewed breakdown on 2026-08-05, authorizing canonical execution.

Review packet:

`data/tinyworlds-nouns-v1/noun-breakdowns/df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35/`

Editable decisions:

`data/tinyworlds-nouns-v1/noun-decisions.json`

The packet contains canonical JSON, Markdown, a standalone HTML page with 57
noun folds, the exact decision snapshot, and a file-hash manifest. Strict reload
and byte verification passed.

## Proposed noun breakdown

- Source records authenticated: 2,717,495 training and 27,630 validation.
- Unique normalized stories: 2,717,494 training and 27,630 validation.
- EOS-terminated tokens: 530,660,896 training and 5,359,770 validation.
- Greedy base order: `mother`, `home`, `bird`.
- Final base union: 1,412,270 unique training stories, or 51.9696%.
- Final base token coverage: 54.4704%.
- Retained task proposals: 42, ordered by training mass beginning with `dog`,
  `house`, `cat`, `father`, and `car`.
- Included but below task threshold: `tractor`, `hammer`, `wrench`,
  `screwdriver`, `engine`, `prince`, `wizard`, `kingdom`, `unicorn`, `witch`.
- Proposed exclusions requiring human review: `saw` because its corpus use is
  overwhelmingly the past tense of “see,” and `friend` because it is an
  unbounded relationship noun rather than a narrow topic class.
- Nonexclusive retained-task memberships: 2,755,417 training and 27,866
  validation. The latter implies 27,866 completion cases and, if judging is
  run exactly as planned, 27,866 OpenRouter requests.

## Implemented surfaces

- Exact whole-word/alternate-form matching over the ordered existing topic
  catalog, global normalized deduplication, validation precedence, overlapping
  memberships, greedy 50% base selection, deterministic probes, compact ledgers,
  and content-addressed partition publication.
- A hash-bound manual decision and approval boundary that is evaluated before
  partition construction or experiment imports.
- Fresh seed-zero GPT-Neo base training with the registered two-epoch optimizer,
  exact optimizer/RNG/cursor resume, 1,000-update and epoch checkpoints, finite
  and improving validation-loss gates, and the 12 GiB allocator gate.
- VAMP-only rank-eight task training with raw all-node prefix-NLL scores, root
  eligibility only for task one, non-root eligibility thereafter, inherited
  parent paths, prior-edge checksum checks, and immutable resumable stages.
- Whole-story loss/routing under all six conditions, midpoint-only routing,
  true-suffix loss, batched equal-budget greedy completions, immediate JSONL
  persistence, and exact resume coverage.
- Deterministically anonymized seven-way OpenRouter judging with strict scores,
  complete rankings, immediate request/response persistence, and credential-free
  local completion.
- Artifact-only Markdown and standalone interactive HTML reporting with graph
  topology, acquisition, routing, confusion matrices, base and pairwise overlap,
  descriptive overlap trends, judge summaries, and folded representative story
  successes and failures.

## Verification

- Focused noun and generic VAMP suite: 16 passed.
- Complete default non-GPU collection: 640 passed, 274 skipped, 11 deselected in
  254.92 seconds.
- Python compilation and scoped whitespace checks: passed.
- Strict packet reload: passed, including all 57 HTML folds and the manual-gate
  notice.
- Gate-state audit: no noun partition, checkpoint directory, or result directory
  exists.

## Manual approval

The approval command was run against review hash
`df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35`.
It published the immutable approval artifact:

`data/tinyworlds-nouns-v1/noun-approvals/2d923cb596d0c01d51a3f0848fb8332a02006d791e218a8163a74505efc92bd5.json`

The artifact also binds decision hash
`96f5c41cf6acf7ba4e5acd8bdedcc0b7bf5cbb254786cc9b1481ac3554efb325`
and source hash
`fa12754af0edc204065af692ec9da0ea83cd059425c48a70763fc83660e8fbfc`.
The required next action is the canonical no-argument run, beginning with exact
partition construction and then GPU preflight. Any future decision edit
invalidates this approval and requires a rebuilt packet and new manual approval.
