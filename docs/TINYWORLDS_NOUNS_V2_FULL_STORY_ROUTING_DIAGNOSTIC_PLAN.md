# TinyWorlds nouns-v2 full-story routing diagnostic

## Question and interpretation

The temporal-consolidation report routes each validation story from its exact
midpoint prefix and reports loss on the held-back suffix. This diagnostic asks
whether the full story contains materially stronger evidence about which final
memory to use. The hypothesis is that TinyWorlds noun identity is only weakly
identified at the midpoint, so addressing error inflates the apparent suffix
loss of otherwise useful adapters.

Full-story selection deliberately sees the evaluation suffix. Its suffix NLL is
therefore a diagnostic upper bound on what stronger story evidence could recover,
not a deployable held-out result. A second whole-story NLL view reports each
candidate on the same tokens used for selection and is labeled self-selected.

## Frozen population and methods

Use the existing 4,440 official nouns-v2 validation stories and exactly three
already-trained final banks:

- blocked-order log-t: base plus nine live temporal intervals;
- round-robin log-t: base plus nine live temporal intervals;
- independent noun: base plus 24 independently trained noun adapters.

Do not train, merge, or modify a model. Candidate order is copied from each
authenticated parent ledger, base is first, and stable first minimum resolves a
tie. For log-t banks, “routing accuracy” means that the selected interval's
training support contains the story noun. For the independent bank it means the
exact `noun-<task>` adapter. Base is a miss in both definitions.

For each candidate `j`, select by mean NLL over every causal transition in the
story. Report route accuracy, story- and token-weighted suffix NLL after
selection, story- and token-weighted whole-story NLL, agreement with midpoint
selection and the suffix oracle, top-two margin, per-task results, and
task-to-candidate confusion. Compare paired full-story minus midpoint changes
with a deterministic seed-zero 10,000-sample bootstrap stratified by the 24
nouns. Also report the fraction of the midpoint-to-suffix-oracle NLL gap removed.

## Exact reconstruction and bounded GPU audit

The parent row stores every candidate's midpoint-prefix mean, suffix mean, and
both token counts. When the prefix has at most 256 transitions, the canonical
whole-story score is exactly the token-weighted combination

```text
(prefix_mean[j] * prefix_tokens + suffix_mean[j] * suffix_tokens)
/ (prefix_tokens + suffix_tokens)
```

because it uses the same reset-at-256 causal windows as the parent evaluation.
Prefixes longer than 256 were originally scored as one router sequence, so all
such stories must instead be scored directly using canonical story windows.

Guard reconstruction with a preregistered deterministic GPU audit containing:

- every prefix longer than 256 transitions;
- every story whose smallest reconstructed top-two margin in any bank is at
  most `0.0002` NLL;
- the minimum-margin short story for each noun.

The authenticated parent evidence selects exactly 190 unique stories, producing
570 bank/story direct rows. For short audited stories, require maximum absolute
candidate-score error at most `1e-4` and identical stable selection. Require
every unaudited margin to exceed twice that tolerance. Fail without publication
if any gate changes or fails.

## Authentication, resume, and publication

Bind an independent
`tinyworlds-nouns-v2-temporal-full-story-routing-contract-v1` to the parent
contract and manifest, every parent publication hash, the three exact source
ledger hashes and 13,320 rows, the selected base, all final adapter artifacts,
the audit story-set hash, scoring/window semantics, and bootstrap settings.

Write the 570 direct rows and 13,320 derived rows immediately to separate
hash-chained JSONL ledgers under
`temporal-consolidation/.work-v1/<parent>/full-story-routing-v1/`. Resume only an
authenticated canonical prefix and reject malformed, reordered, duplicated, or
tampered rows. Publish a standalone addendum under
`temporal-consolidation/<parent>/full-story-routing-v1/`; never rewrite the
parent report's 32 artifacts.

Publish separate Markdown and self-contained HTML reports, accessible
Matplotlib SVG, aggregate/per-task/bootstrap/confusion CSV files, canonical
analysis JSON, execution and allocator measurements, and a manifest hashing
every addendum artifact. Use collapsible method, audit, uncertainty, and
provenance sections. Regenerate the complete report byte-identically and verify
the protected nouns-v1/v2 artifacts and parent publication remain unchanged.

## Runner and gates

`scripts/run_tinyworlds_nouns_v2_full_story_routing.py` is the sole no-options
runner. It fixes GPU 0, disables JAX preallocation, prints the persistent work
directory, provides phase and overall ETA bars plus exact direct-row progress,
enforces the existing 12 GiB allocator limit, resumes interrupted ledgers, and
sends a desktop completion notification. This diagnostic is expected to finish
in under five minutes once kernels compile, so it does not create another web
dashboard.

CPU tests cover weighted reconstruction, stable ties, audit selection, metric
semantics, bootstrap determinism, ledger tamper/resume rejection, report
self-containment/accessibility, and byte-identical regeneration. A bounded GPU
test exercises canonical direct scoring and the allocator gate on real sources.
