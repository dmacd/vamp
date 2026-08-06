# TinyWorlds Noun-Overlap v1 — Implementation and Review-Gate Report

Date: 2026-08-05 to 2026-08-06

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

## Review-gate verification

- Focused noun and generic VAMP suite at the approval boundary: 16 passed.
- Complete default non-GPU collection: 640 passed, 274 skipped, 11 deselected in
  254.92 seconds.
- Python compilation and scoped whitespace checks: passed.
- Strict packet reload: passed, including all 57 HTML folds and the manual-gate
  notice.
- Gate-state audit at that checkpoint: no noun partition, checkpoint directory,
  or result directory existed.

## Manual approval

The approval command was run against review hash
`df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35`.
It published the immutable approval artifact:

`data/tinyworlds-nouns-v1/noun-approvals/2d923cb596d0c01d51a3f0848fb8332a02006d791e218a8163a74505efc92bd5.json`

The artifact also binds decision hash
`96f5c41cf6acf7ba4e5acd8bdedcc0b7bf5cbb254786cc9b1481ac3554efb325`
and source hash
`fa12754af0edc204065af692ec9da0ea83cd059425c48a70763fc83660e8fbfc`.
The approval authorized the canonical no-argument run, beginning with exact
partition construction and then GPU preflight. Any future decision edit still
invalidates this approval and requires a rebuilt packet and new manual approval.

## Approved execution progress

The canonical run published partition
`04ca2acf85f9505f0b7568b1696fbf290a8d2cbf78387dcfd6e815258fcc28b8`
with 42 retained noun tasks. CUDA preflight
`e15b669aca33ca9a200244e7742589532d3b4b0d2baa3bb3eddd727d0a5cd026`
measured 7.99 GiB. The fresh seed-zero base passed its held-in gate: validation
NLL moved from `1.4190817762` after epoch one to `1.3108355385` after epoch two,
an improvement of `0.1082462377`. Selected-base identity is
`c900a4fc47fcb8317900c83c53e61be33e0c0c856e624a8713e7348a57e27788`.

All 42 VAMP nodes completed their fixed 2,000 updates. Final stage
`c56516ed22a9e5ee89868330fc61031c90edf6ea1d2d5d2dcf75e191e9fd0156`
binds adaptation manifest
`3384c80f51f23ffad72d8b1261bd447ffaa7423ee619b6690b3d38e1a679c8bf`.
Whole-story evaluation completed every one of the 27,866 task/story
memberships. The atomically published `whole-story-nll.jsonl` contains exactly
167,196 canonical rows: all six conditions for every membership.

The first one-story-at-a-time evaluation measurement projected approximately
25 hours. A bounded batching correction now scores up to 32 story windows
together while limiting differentiable EBT to eight rows after a 32-row EBT
attempt exceeded memory. The safe and 32-row attempts are byte-identical on
all 186 common completed rows, and the safe path projects roughly two hours.
Generation required a second bounded performance correction. The original
decoder recomputed and recompiled the complete prefix after each token; even a
five-story batch projected close to a day. The generic decoder now performs
one prompt prefill, retains per-layer global/local attention KV caches, and
advances the continuation in one compiled device loop. A direct CPU oracle test
matches cached multi-step output to full-prefix recomputation with unequal
prompt lengths, mixed attention layers, and different LoRA nodes per row.

The noun evaluator generates a story/node continuation only once when several
conditions choose that same node, then reconstructs all six labeled results.
It sorts cases by their frozen reference-length budget and uses deterministic
first-fit packing over host chunks of at most 128 suffix windows. Device calls
contain at most 72 distinct story/node rows. This path preserves every prompt,
condition, selected path, suffix loss, and per-story output budget. Its live
process footprint is approximately 9.7 GiB. A measured 192-row experiment rose
to 17.9 GiB and was rejected rather than weakening the frozen 12 GiB gate; all
rows from performance experiments are named diagnostic files and are excluded
from canonical loading and reports.

The final generation ledger is currently running from row zero and persists
each completed bounded chunk immediately. Its observed projection is roughly
five to six hours. The current focused generation/noun/VAMP selection passes
25 tests; the broader generation, GPT-Neo, LoRA, noun, and language-run
selection passes 81 tests. Python compilation and scoped whitespace checks
pass.
`OPENROUTER_API_KEY` is not configured. After local generation finishes, the
runner will still reconstruct and publish Markdown and standalone interactive
HTML, set the manifest phase to `awaiting_judge_credentials`, and send the
requested desktop notification. A later identical invocation will skip every
local artifact and resume at external judging.
