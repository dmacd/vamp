# TinyWorlds Nouns-v2 Log-t Temporal Consolidation Study

## Purpose

This study tests the synchronous, base-relative temporal consolidation scheme in
`docs/log_t_temporal_consolidation.pdf`.  It asks whether an LSM-tree-like bank
of at most two LoRA chunks per level can preserve useful language-model
specialization while keeping the live adapter set and exhaustive address cost
logarithmic in the number of arrivals.

The study is deliberately separate from the canonical nouns-v2 run.  It may
read and authenticate the selected base, partition, probes, checkpoints,
ledgers, and reports, but it writes only beneath the temporal-consolidation
checkpoint and result directories.  Every result is published, with no
quality-based merge rejection and no pass/fail quality threshold.

## Frozen data stream

- Use seed zero and the canonical order of the 24 nouns.
- Exclude all 36 registered training probes for every noun and every official
  validation story.
- Deterministically rank the remaining pure training stories and retain 4,096
  per noun.
- Divide each noun into eight consecutive, immutable shards of 512 whole
  stories.  All causal windows from a story remain in the same shard.
- Evaluate two orders over the same 192 shard identities:
  - **blocked:** all eight shards for one noun before the next noun;
  - **round robin:** shard zero for every noun, then shard one, and so on.
- Select the 16 lowest-hash official validation stories per noun as the live
  sentinel.  The final evaluation uses all 4,440 official validation stories.

The contract records every selected story ID, shard membership, arrival order,
probe and validation identity, source artifact hash, seed, and training setting.
It is immutable once published.

## Models and training

All LoRA models use the authenticated frozen base, rank and alpha eight,
physical batches of 32, context length 256, clipped AdamW, learning rate
`1e-3`, weight decay `0.01`, and exactly four finite epochs.  Incomplete final
batches are padded and masked; no example is dropped or cycled.  Every fresh
adapter begins as a deterministic zero-effect LoRA.

The compared systems are:

1. **Frozen base.**  No trainable state.
2. **Log-t temporal consolidation.**  Train one standalone level-zero adapter
   per arrival.  Keep at most two chunks per level.  When a third arrives,
   retrain the two oldest equal-level chunks from the base on their exact source
   union, retire the children from the live set, and recursively carry upward.
3. **Sequential LoRA.**  Continue one adapter through each stream without
   replay.  Retain the adapter across arrivals and reset Adam state at each
   arrival, matching the existing nouns-v2 sequential control.
4. **Independent noun bank.**  Train 24 fresh adapters, one on each noun's
   eight-shard union, and exhaustively route over base plus those adapters.
5. **Joint IID LoRA.**  Train one rank-eight adapter on the shuffled union of
   all 98,304 selected stories.
6. **Joint IID full model.**  Fine-tune a copy of the selected base for four
   epochs on the same union using learning rate `5e-5` and weight decay `0.01`.

The level-zero adapters, independent noun bank, and IID controls are shared
between the two ordering comparisons where their inputs are identical.  The
merges and sequential controls are order-specific.  The independent noun bank
is a practical requested baseline, not the exact 192-adapter no-consolidation
ablation; reports must keep that limitation explicit.

At 192 arrivals, each ordering must have made 183 merges and finish with the
nine intervals `[1-64]`, `[65-128]`, `[129-160]`, `[161-176]`, `[177-184]`,
`[185-188]`, `[189-190]`, `[191]`, and `[192]`.  The deployed temporal address
set is those nine adapters plus the base.

## Addressing and evaluation

Log-t and the independent bank score only the exact midpoint prefix.  For each
candidate, compute mean prefix token NLL and select the first minimum in a
stable order: base first, then temporal interval/level/artifact order or the
canonical noun order.  The router receives neither suffix tokens nor task
identity.  The chosen model then scores the held-back suffix.

An evaluator-only suffix oracle scores every candidate on the true suffix.  It
is used only to measure oracle NLL, regret, and prefix/suffix selection
agreement.  Mixed temporal chunks do not have a unique true task label, so the
report calls selection of a chunk containing any data for the query noun a
**noun-support hit**, not route accuracy.  Exact noun-route accuracy and
confusion are reported only for the independent noun bank.

After every arrival, evaluate the 16-story sentinel for every noun encountered
so far under base, sequential LoRA, routed log-t, and log-t suffix oracle.  This
is 38,400 blocked and 69,312 round-robin story-stage cases.  Every eight
arrivals, evaluate those bounded conditions on every official validation story
for encountered nouns: 72,256 blocked and 104,035 round-robin cases under the
canonical per-noun validation counts.  At the
final stage, reuse those rows and add both IID controls and the exhaustively
routed independent bank for all 4,440 stories.

Primary measurements are suffix story and token NLL, suffix top-one token
accuracy, forgetting, backward transfer, oracle regret, address/oracle
agreement, temporal interval/age/level/task mixture, and paired ordering
effects.  Final paired differences use deterministic, noun-stratified,
seed-zero 10,000-sample bootstrap intervals.  They are descriptive summaries,
not population claims or pass/fail tests.

For every merge, measure the signed parent-minus-child loss on both exact child
datasets and define

`delta = max(L_Da(parent) - L_Da(child_a), L_Db(parent) - L_Db(child_b))`.

Measure the analogous sentinel-validation distortion, retain each original
arrival's signed and positive lineage increments, and verify that the final
direct level-zero-to-active-ancestor drift telescopes to the signed increments
and is bounded by their positive parts.

Report active adapter memory, archived adapter storage, source references,
candidate evaluations, forward-equivalent tokens, optimizer work, carry depth,
worst insertion latency, amortized work, cold compilation, five synchronized
warm timing repetitions, end-to-end wall time, and the 12 GiB allocator peak.
Do not combine incomparable operation units in one axis or total.

## Live dashboard and artifacts

The no-options GPU-zero runner starts a standard-library HTTP server bound only
to `127.0.0.1`, choosing the first free port from 8765 through 8775.  It prints
the URL and persistent temporary work directory, then runs authentication,
data preparation, shared controls, both ordering studies, evaluation, timing,
and publication.  The server reads immutable disk snapshots and never owns GPU
or model state.

GET-only endpoints are `/`, `/healthz`, `/api/v1/snapshot`,
`/api/v1/events?after=<seq>`, and an allowlisted `/artifacts/` subtree.  The
client polls every two seconds with ETags.  CSS and JavaScript are inline, the
page has a restrictive CSP, and no external dependency is required.

The dashboard shows the complete job matrix; overall, phase, and nested job
progress; both live temporal stacks; the current carry; active/archive memory;
training loss; provisional evaluation estimates; live plots; provenance; and
an append-only event log.  Every job projected to exceed five minutes gets its
own exact-unit progress bar, elapsed time, measured rate, and ETA.  Long-running
evaluation metrics show stratified provisional estimates, sample count, and
coverage until the full result and final bootstrap are available.  ETAs begin
from authenticated canonical timing records and update in operation/shape
buckets.  Incomparable work units remain separate; overall ETA is obtained by
summing bucket-specific remaining-time estimates.

Progress and result ledgers are canonical, append-only, self-hashed JSONL.
`status.json` is an atomic projection rebuilt from those ledgers on restart.
Training state includes parameters, optimizer moments, RNG, epoch, batch
cursor, and ledger position and is saved at least every 128 updates or two
minutes.  Completed adapters and full-model checkpoints are atomically
published with their source, lineage, settings, tensor checksum, runtime, loss
trace, and allocator evidence.

Final outputs include separate Markdown and self-contained HTML reports, a
self-contained frozen dashboard, aggregate/per-task/stage/arrival/merge/timing/
cost/bootstrap CSVs, accessible Matplotlib SVGs, and Graphviz lineage diagrams
for both orderings.  The compact diagrams emphasize the nine final chunks; the
full audit diagrams cover all 192 leaves and 183 merges.

## Verification gates

- Unit tests cover deterministic selection, probe/validation isolation,
  padding, epoch coverage, carry schedules, oldest-first merges, exact
  intervals, stable routing, prefix-only structure, mixed-chunk metrics,
  distortion, telescoping errors, cost accounting, and bootstrap determinism.
- Resume tests interrupt level-zero, merge, sequential, IID LoRA, and full-model
  training and require the same completed tensor identities and reports.
- Ledger tests reject malformed, reordered, duplicate, cross-contract, and
  tampered rows while repairing only an incomplete final line.
- Dashboard tests cover loopback binding, GET-only behavior, CSP, traversal
  rejection, ETags, event replay, ETA calculations, provisional estimates,
  resume reconstruction, and a progress bar for every job above five minutes.
- Report tests require standalone Markdown/HTML, accessible embedded SVGs,
  readable labels and units, collapsible detail, complete Graphviz coverage,
  CSV identity, and byte-identical regeneration.
- Run focused CPU tests, a bounded real-data GPU smoke, interruption/resume
  smoke, the complete default experiment, exact report replay, the opt-in real
  source test, and the clean full suite.
- Hash protected nouns-v1 and nouns-v2 checkpoints, ledgers, manifests, and
  reports before and after execution and reject any change.

Implementation lives in a dedicated temporal-consolidation package and
no-options runner.  Checkpoints and results are isolated below
`checkpoints/tinyworlds-nouns-v2/temporal-consolidation/<contract-sha>/` and
`results/language_cl/tinyworlds-nouns-v2/temporal-consolidation/<contract-sha>/`.
After verified execution, durable semantics belong in `DESIGN.md` and measured
status and follow-ups belong in `PLAN.md`.
