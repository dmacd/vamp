# TinyWorlds-v2-GPT Execution Plan

## Purpose and Authority

This is the canonical living execution tracker for `tinyworlds-v2-gpt`. The
research rationale and concrete examples are preserved in
`TinyWorlds-v2 Benchmark planning.pdf`; this file records implementation
status, fixed gates, and the work authorized next.

TinyWorlds-v1 is a completed negative scientific result. Its symbolic bundles,
reports, and stopped calibration remain immutable. V2 retains the world ledger,
VAMP graph, candidate scoring, parent-transfer machinery, and continual-learning
metrics, but replaces the complete prose-generation contract.

The governing rule is:

> The world ledger determines what is true and what is scored. A capable
> external language model determines how that truth is expressed as a natural
> TinyStories-style story.

Work advances one phase at a time. A phase is promoted only after its artifact
gate and the complete default test suite pass. Phase 1 and the Phase 3 sample
generation end at mandatory human-approval stops.

## Status

| Phase | State | Advancement gate |
|---|---|---|
| V1 stopped result | Preserved | No further v1 tuning or pilot is authorized |
| 1. Reference profile and generator bakeoff | **Active — production gates pending** | Reference artifacts validate, one generator route passes automated review, and the exact blinded audit digest receives explicit human approval |
| 2. Counterbalanced world bibles | Blocked by Phase 1 | World audit and deterministic counterbalance checks pass |
| 3. Natural training corpus | Blocked by Phase 2 | The 150-story sample audit receives explicit human approval, then the complete corpus validates |
| 4. Natural evaluation data | Blocked by Phase 3 | Frozen-base sensitivity, balance, semantic, leakage, and 100-probe human gates pass before adapter training |
| 5. Four-task calibration | Blocked by Phase 4 | Validation ladder passes and locked test is opened exactly once |
| 6. Eight-task pilot | Blocked by Phase 5 | Complete pilot bundle passes integrity, coverage, drift, and report-reproduction gates |
| 7. Scaling and consolidation studies | Deferred | Phase 6 passes with interpretable learning and routing behavior |

Current checkpoint (2026-07-19): the offline Phase 1 implementation is in
place through the artifact-integrity boundary. It covers the exact
pinned-source cohorts and profiles, semantic route locks with separately
preserved catalog provenance, fixed request construction, crash/process-safe
cost accounting, historical route and per-reservation authorization evidence,
explicit pre-POST cancellation, the seven-route funnel, quality selection, a
balanced blinded audit, strict cross-artifact validation, secret-reflection
stops, and exhaustive zero-network replay from raw attempts. Targeted offline
unit and integration gates, the real-source extraction/profile integration,
the real-GPU NLL smoke, the GET-only public-catalog resolver smoke for all
planned generator and verifier routes, and focused cost/recovery/replay checks
pass. The complete offline default suite passes 753 tests with one optional
skip and eight resource-marked deselections in 2,045.40 seconds; peak RSS was
4,085,656 KiB.

The production bakeoff has not run. No live cost preflight has authorized a
completion request, no paid OpenRouter completion has been submitted, no
generated bakeoff sample exists, and no production audit artifact or Phase 1
gate result exists. The remaining Phase 1 work is zero-BYOK authorization,
live cost preflight, the paid seven-route bakeoff, automated selection, the
blinded human audit, and explicit approval of its exact digest. Phase 2
therefore remains blocked.

The supplied `openrouter-tinyworlds-key.txt` is the inference credential, not
a Management API key. Paid execution is presently waiting for either a
distinct `OPENROUTER_MANAGEMENT_API_KEY` or explicit user confirmation recorded
in the short-lived manual attestation described below. No manual attestation
has been created.

## Non-negotiable V2 Contract

- Implement V2 in a parallel `tinyworlds_v2` package and versioned artifact
  tree. Do not alter, reinterpret, resume, or import prose from v1.
- Never use `TinyWorldsTemplateRegistry`, `_fact_statement`,
  `_rule_statement`, `_query_statement`, `_fit_exact_tokens`, padding
  fragments, artificial cue blocks, formal relation questions, or exact-token
  English in the v2 data path.
- Every training story and natural knowledge probe comes from a pinned external
  model and provider route through a content-addressed, resumable cache.
- Generated text is variable length. EOS, tensor padding, and loss masks handle
  batching; filler English never does.
- Text may not expose task/family IDs, relation names, query terminology,
  answer markers, page numbers, prompts, schemas, or other benchmark internals.
- The ledger is authoritative. Generated prose is accepted only after
  deterministic, semantic, stylistic, statistical, and required human checks.
- Normal tests are offline. Remote generation is an explicit integration or
  production action, and reloading a completed artifact never calls a network.

## Phase 1 — Reference Profile and Generator Bakeoff

Phase 1 proves that the generation pipeline can reproduce ordinary
TinyStories-like prose before it is allowed to express world knowledge. It
must not generate any world, task, training, or probe data.

### 1.1 Pinned reference distribution

- Stream only `TinyStories_all_data.tar.gz` from the TinyStories dataset at
  revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, expected size
  `1,608,001,638` bytes and SHA-256
  `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
- Deterministically select 10,000 GPT-4 records with original prompt metadata,
  10,000 disjoint GPT-4 reference stories, 10,000 records from the locally
  verified GPT-4-only V2 validation aggregate, and 200 paired prompt/reference
  records for the bakeoff. Enforce disjointness with a separately recorded
  Unicode-normalized story hash, not only the whole released-record hash.
- Reconstruct the empirical noun, verb, adjective, and narrative-feature
  distributions from released prompt metadata. Do not substitute a single
  hand-written approximation of the TinyStories recipe.
- Profile GPT-2 token and word vocabularies, token frequencies, story and
  sentence lengths, paragraph counts, dialogue, openings, endings, repetition,
  requested-feature realization, and TinyStories-8M per-story NLL.
- Compute text statistics in deterministic 16-process shards and NLL on the
  RTX 4090. Merge by stable record identity so process completion order cannot
  affect any artifact.
- Authenticate the complete archive before parsing it, stream strict JSON
  records without extracting the tar tree, and reject malformed members,
  duplicate JSON fields, non-finite values, and oversized records. Rank unique
  story content rather than released-record locations so duplicate source
  records do not receive extra sampling weight.
- Define story-content identity as SHA-256 over NFKC-normalized, case-folded,
  whitespace-collapsed text. Enforce uniqueness and cross-cohort disjointness
  with that identity across all archive cohorts and the pinned validation
  cohort while retaining location- and raw-content-bound source IDs for
  provenance.

### 1.2 OpenRouter generation boundary

- Add the focused modules `generation_schema.py`, `openrouter.py`,
  `generation_cache.py`, and `reference_profile.py` beneath
  `src/apm/data/text/tinyworlds_v2/`, plus one thin fixed-preset Phase 1 runner.
  Do not introduce research-choice CLI switches or a general provider
  abstraction.
- Use direct HTTP behind an optional `generation` dependency. The production
  key is read from `OPENROUTER_API_KEY`, falling back locally to
  `openrouter-tinyworlds-key.txt`; enforce mode `0600` on that file and never
  log, hash, cache, copy, or serialize its contents.
- Resolve and persist the exact OpenRouter model catalog plus one endpoint
  response per planned model before generation. Prefer a first-party endpoint,
  then unquantized FP16/BF16, then FP8; reject advertised 4-bit routes.
- Separate volatile catalog provenance from billable route identity. The full
  catalog-response digest remains in the route record and artifact, while the
  route-lock/request hash covers the local route ID, requested and canonical
  model identities, provider selector and returned-provider identity,
  quantization, and input/output prices. A byte-only catalog refresh therefore
  reuses the same request/cache entry; any change to a semantic field creates
  a different lock or fails validation.
- Disable model/provider fallbacks, automatic routing, response healing, and
  plugins. Bind each request to its expected canonical model and provider, and
  fail on catalog or route drift.
- Refresh the public catalog immediately before every paid route batch and
  verifier batch, then compare its semantic lock with the original one. Halt
  before that batch's first POST and publish `catalog_route_drift` if the
  model, provider, quantization, or price changed. Record the fresh catalog
  digest on every submitted raw attempt even when its semantic cache identity
  is unchanged.
- Content-address the exact model, route, prompts, JSON schema, settings, and
  request body. Preserve exact raw response bytes, relevant response headers,
  request ID, returned model/provider metadata, token usage, and billed cost.
- Include a versioned transport-protocol identity in every request hash. The
  current protocol requires `X-OpenRouter-Metadata: enabled` and
  `X-OpenRouter-Cache: false`; changing a behavior-affecting header or its
  interpretation requires a protocol/version change and therefore a new
  request identity. Authorization material is never part of that identity.
- Request OpenRouter routing metadata explicitly. Resolve the serving provider
  from its unique selected endpoint, and use the authenticated generation-stats
  record as a persisted fallback when completion usage omits billed cost or
  serving provenance.
- Retry only transport, rate-limit, and server failures. Schema-invalid or
  semantically invalid responses remain immutable failed observations and
  count against route reliability.

### 1.3 Seven-model funnel

Screen the same first 50 neutral briefs through all seven routes. Expand three
finalists over the remaining 150 briefs, yielding 200 paired observations per
finalist. Canonical route names are resolved to pinned OpenRouter model and
provider identities immediately before the cost preflight.

| Candidate route | Input / output price per 1M tokens at plan time |
|---|---:|
| Ling 2.6 Flash | `$0.01 / $0.03` |
| Gemma 4 26B | `$0.07 / $0.34` |
| DeepSeek V4 Flash | `$0.098 / $0.196` |
| Mistral Small 4 | `$0.15 / $0.60` |
| Qwen3.5 35B-A3B | `$0.14 / $1.00` |
| Gemini 3.1 Flash Lite | `$0.25 / $1.50` |
| GPT-5.4 Mini | `$0.75 / $4.50` |

Use pinned `gpt-5.4-2026-03-05` as the independent blind style verifier, not as
another generator candidate. Bind the GPT-5.4 Mini route to
`gpt-5.4-mini-2026-03-17` when that snapshot is available through the selected
provider. Snapshot every observed catalog price and re-evaluate all costs
before any billable submission.

A screening route must achieve at least 98% schema-valid responses, at least
90% deterministic acceptance, and zero forbidden forms in any response.
Expand the cheapest passing route, the closest-to-reference route, and
the best remaining normalized cost/alignment trade-off. Resolve ties in table
order. If fewer than three routes pass, expand every passer; if none pass,
publish a stopped result.

### 1.4 Cost preflight and ceilings

- Before any billable request, write and print `cost_estimates.json` from the
  completed request bodies, current pinned-route prices, matched-reference
  output lengths, verifier traffic, retry allowance, and worst-case in-flight
  reservations.
- Expected total Phase 1 OpenRouter spend is approximately `$3–$7`; the hard
  inclusive ceiling is `$15`. If the conservative estimate crosses the cap,
  submit nothing.
- Include direct OpenAI Batch comparisons for
  `gpt-5.4-mini-2026-03-17` and `gpt-5.4-2026-03-05`. Batch is an
  estimate-only alternative in this phase; no second production backend is
  introduced. Its token charges are approximately half normal API pricing
  under the documented 24-hour Batch contract.
- After observed usage and rejection rates are known, project full-corpus
  economy, balanced, and quality-ceiling envelopes. Economy minimizes projected
  cost among fully qualified routes; quality-ceiling minimizes alignment
  distance; balanced equally weights min-max-normalized cost and alignment.
  Estimates, provider-reported actual costs, and conservative unknown-cost
  charges remain separate fields.
- Enforce the inclusive cap again at runtime with one ledger shared by all
  workers. Reserve each exact request's byte/max-output/locked-price ceiling
  before POST, reconcile every current or historical cached attempt on resume,
  and halt if an HTTP response or ambiguous completion transport failure lacks
  provider-reported cost. Generation-stats retries are append-only and never
  repeat the originating completion POST. Encode maximum-price JSON numbers
  conservatively upward and reserve the transmitted cap. A denied new
  reservation does not revoke prior authorized reservations; a later provider
  billing/contract failure on one supersedes that admission-denial reason.
- Hold one nonblocking cross-process lease for the entire paid raw-cache
  lifecycle, in addition to the eight-worker in-process lock. Before each POST,
  atomically write and fsync an immutable upper-bound reservation; atomically
  settle it only after persisting the HTTP observation. On restart, reconcile
  the journal, canonical request, its complete persisted route lock, raw
  attempt, and provider-reported cost. Recheck the reservation immediately
  before transport; if another worker has halted the ledger, append an explicit
  `cancelled_before_post` state that is neither charged nor classified as a
  submission. A reservation with no recoverable response is charged at its
  full bound and stops as `orphaned_cost_reservation`; a known billed settlement
  whose raw response is missing stops as `billed_attempt_response_missing`.
  Neither case may be reposted. Stopped artifacts retain exact generator,
  verifier, and route attribution for both actual and conservative charges.
- Treat BYOK as a separate billing boundary. The inference credential is not a
  Management API credential and cannot prove the absence of workspace BYOK
  keys. Before any completion POST, require either a zero-key `/api/v1/byok`
  result obtained with a distinct `OPENROUTER_MANAGEMENT_API_KEY`, or the
  repository-root `openrouter-tinyworlds-no-byok-attestation.json` created only
  after explicit user confirmation. The manual record has a strict canonical
  schema and the exact statement “I attest that this OpenRouter workspace has
  zero configured BYOK keys.”, plus an active UTC interval with a maximum
  24-hour lifetime. Persist only sanitized `byok_preflight.json` evidence with the
  proof source, digest/count, and times; never persist the management response
  or BYOK key metadata. Embed that same canonical allowed evidence and its
  digest in every durable paid reservation, so historical authorization remains
  independently provable after a failed later preflight. Reconcile historical
  paid state before deciding whether the current invocation may submit new
  work. If neither proof is available, publish a fail-closed provider-billing
  stop with zero new completion POSTs. Independently require every successful
  completion or generation-stats fallback to prove `is_byok=false`; observed or
  ambiguous BYOK billing consumes the conservative request bound and halts the
  run.

### 1.5 Automated quality and alignment gates

Score all 200 observations for each finalist against matched genuine
references. A route qualifies only if every gate passes:

- Schema-valid responses at least 99%; deterministic acceptance at least 95%.
- Requested noun, verb, adjective, and narrative-feature adherence at least
  95%. Feature realization is measured from story text by the deterministic
  checker, separately from requested prevalence and model-copied evidence.
- No response contains internal IDs, relation labels, answer markers,
  meta-language, or padding fragments.
- At least 98% of word tokens are covered by the reference vocabulary profile.
- GPT-2 token-unigram Jensen-Shannon divergence is no greater than
  `max(0.10, 5 * matched_reference_split_JSD)`.
- Median TinyStories-8M NLL differs from matched references by no more than
  `0.30`; normalized NLL Wasserstein distance is no greater than `0.35` of the
  reference interquartile range.
- Median story length is within 15%; median sentence length is within two
  words; dialogue, paragraph, and realized-feature rates are within ten
  percentage points of matched references. Requested-feature prevalence is
  within ten points of the separate 10,000-record released prompt-metadata
  profile. Median repeated 3--5-gram incidence differs by at most 0.05.
- Digit-bearing and numeric-only lexical-token rates differ from matched
  references by at most 0.01. These tokens remain in every lexical denominator;
  mixed letter-digit IDs, snake-case IDs, and page/chapter numbers are blocked,
  while ordinary standalone numbers remain valid children's prose.
- Blind verifier rubric means are no more than 0.5 points below genuine
  references, and at least 90% of stories have no grammar, coherence,
  repetition, or meta-language hard failure.

If the gates cannot be met within the cost ceiling, publish either
`no_quality_qualified_route` or `blocked_by_cost_cap`, including every failed
metric and the remaining-cost estimate. Do not reduce sample counts, omit
verification, or relax thresholds.

### 1.6 Mandatory human audit

- Produce a static interactive audit with 100 matched genuine GPT-4 references
  and 100 generated controls. Balance generated controls 34/33/33 across three
  finalists (or as evenly as possible across fewer finalists) and randomize
  them deterministically. Preserve all screened finalists in this comparison;
  only the automated-qualified subset remains eligible for final selection.
- The first-pass `audit.html` contains only opaque IDs, blinded text, source
  prompts, token counts, base NLL, and automated style scores. Keep source and
  route identities in a separately revealed `audit_key.json`.
- Require a TS-like accept/reject decision, simplicity and coherence ratings,
  and a genuine/generated guess for every item before export.
- Generated acceptance must reach 85% and be no more than ten percentage
  points below genuine references; every selectable route must reach 80%;
  overall source discrimination must not exceed 65%.
- Choose the cheapest automated passer whose human acceptance is within ten
  points of the best route.
- Phase 2 remains locked even when all metrics pass. Advancement requires the
  user to explicitly approve the exact audit artifact digest in
  `audit_approval.json`.

### 1.7 Phase 1 artifacts and verification

Publish beneath `data/tinyworlds-v2/reference/`:

- `source_manifest.json`, `neutral_story_briefs.jsonl`,
  `prompt_metadata_sample.jsonl`, `reference_story_sample.jsonl`,
  `validation_source_sample.jsonl`, `reference_annotations.jsonl`,
  `reference_observations.jsonl`, `paired_reference_observations.jsonl`, and
  `reference_statistics.json`;
- `configuration.json`, exact raw `catalog/` responses and normalized route
  locks, all persisted measurement batches and runtimes,
  `generator_bakeoff.jsonl`, `verifier_results.jsonl`,
  `sequential_results.jsonl`, `cost_estimates.json`, `cost_actuals.json`,
  `cost_observations.json`, `runtime_cost_ledger.json` when paid execution is
  reached, `quality_comparisons.json`, `quality_details.json`,
  `finalist_decision.json`, `status.json`, `audit.html`,
  `audit_packet.json`, and `audit_key.json` as applicable to the terminal
  status;
- the human-authored overlays `audit_decisions.json`,
  `audit_approval_request.json`, and `audit_approval.json`, which remain
  outside the immutable base manifest;
- one directory per canonical route containing `requests.jsonl`,
  `batch_submission.json`, `plan.json`, `raw_responses.jsonl`,
  `accepted.jsonl`, `rejected.jsonl`, and `manifest.json`, plus the verifier
  manifest and an authenticated copy of every submitted raw request, response,
  stats observation, full historical route lock, per-reservation sanitized BYOK
  authorization, and runtime-cost journal entry under `raw_cache/`.

Every manifest binds source pins, request and response hashes, semantic route
identity and full catalog provenance, prompt/schema and transport-protocol
versions, validator/measurement versions, prices, costs, and all file digests.
The strict semantic loader cross-checks source counts and identities, funnel
selection, quality reports, planned/submitted/completed requests, accepted and
rejected splits, raw attempts, cost-journal settlements, audit allocation, and
all permitted terminal statuses rather than trusting the root digest alone.
Those terminal statuses are exactly `blocked_by_cost_cap`,
`blocked_by_runtime_cost_cap`, `provider_billing_unknown`,
`catalog_route_drift`, `no_quality_qualified_route`,
`audit_insufficient_accepted_samples`, and `awaiting_human_audit`.

Derived replay starts only from persisted briefs, route locks, raw HTTP/cache
evidence, and accelerator-derived measurements. With a transport that raises
on any network access, it reconstructs canonical requests, reparses every
attempted response discovered in the raw cache, including interrupted terminal
attempts absent from committed route/verifier result streams, reruns source
joins, deterministic validators and quality selection, rebuilds route/verifier
streams and the blinded audit, and byte-compares every replayed derived file
(apart from the replay tree's own root manifest). Raw provider, BYOK, and
response-contract evidence must agree with the published terminal cause and
status. It never reopens the TinyStories sources, loads the
tokenizer/checkpoint or GPU, reads an API key, or performs a network call.
Public completed-artifact validation and human approval both perform this
complete semantic and replay validation; passing an overlay-only digest check
is insufficient.

Audit construction uses an exact deterministic balanced assignment over
distinct pair IDs, not a greedy sampler. All screen finalists remain in the
blinded generated controls; only automated-qualified routes can be selected
after human scoring. If accepted samples cannot satisfy the exact per-route
quotas, publish `audit_insufficient_accepted_samples` with feasibility evidence
instead of changing counts or silently substituting examples.

Tests cover source verification and streaming selection, deterministic shard
merging, request hashes, route/catalog drift, immutable cache behavior,
retry classification, schema rejection, cost arithmetic and cap enforcement,
crash and cross-process recovery, BYOK fail-closed behavior, quality metrics,
finalist selection, exact blinded-audit allocation, semantic tamper rejection,
zero-network replay, and approval-digest enforcement. Default tests use fake
transports and small fixtures. Marked integrations cover the real archive,
tokenizer/checkpoint NLL, and minimal OpenRouter smoke requests. Run the
complete default suite before billable generation and after validating the
completed artifact.

**Phase 1 gate:** all reference and bakeoff artifacts strictly load; at least
one route passes automated and human thresholds within the cap; raw-cache
rebuild is byte-identical; the full default suite passes; and the exact audit
digest has explicit human approval. Stop before Phase 2 otherwise.

## Phase 2 — Ordinary Counterbalanced World Bibles

- Have the selected LLM propose coherent settings, casts, places, and natural
  associations from a constrained slot schema. Deterministic code assigns the
  final candidate values with a Latin square after the frozen checkpoint is
  fixed.
- Use ordinary child names, animals, toys, food, homes, parks, shops, and
  weather. Each task introduces about 4–8 direct facts, 1–2 conventions, at
  most one contextual revision, 2–4 recurring characters, and 2–3 places.
- Use four counterbalanced families with topology `seed -> ordinary extension`
  and `seed -> contextual revision -> revision extension`. Interleave families
  in the eventual presentation order.
- Require every candidate value to be correct equally often, assignments to be
  arbitrary but natural and post-checkpoint, facts to require no glossary, and
  the world bible to contain no prose later used as training data.
- Publish world bibles, tasks, fact slots, counterbalance matrix, lineage, raw
  generation provenance, validation results, and a compact human world audit.

**Phase 2 gate:** deterministic replay verifies every assignment, topology,
ledger fact, and balance count; semantic/style checks pass; the world audit is
approved; and the default suite passes.

## Phase 3 — TinyStories-Process Training Corpus

- For each brief, choose one or two target facts; empirically sample a noun,
  verb, adjective, zero to three narrative features, and a simple plot shape;
  then ask the pinned generator to make the facts matter naturally in the
  plot. Supply only relevant cast/places, target facts, contradictions to
  avoid, and candidate alternatives that must not be established.
- Require structured evidence metadata and verify it independently. The
  generation model is never its own sole judge.
- Generate a maximal immutable corpus supporting deterministic exposure
  subsets: 32 accepted exposures per direct fact, 24 demonstrations per
  convention, and 16 stories per contextual revision. Record both exposures
  when one story realizes two related facts.
- Target approximately 100–220 GPT-2 tokens. Use EOS, masks, and tensor
  padding; never lengthen prose to a fixed capacity.
- Separate source, request, response, accepted/rejected, exposure, validation,
  cost, and manifest records. Each accepted story binds request, model/route,
  prompt/schema, raw response, task/facts, evidence, ingredients, token count,
  and base NLL.

Before full generation, produce exactly 50 accepted stories for one seed, 50
for one ordinary extension, and 50 for one revision. The audit shows every
request, raw output, accepted text, ledger fact, verifier result, token
statistics, and rejection reason.

**Phase 3 gate:** stop for explicit approval of the 150-story audit digest.
Only then generate and validate the maximal corpus; require fact entailment
above 95%, contradiction below 1%, acceptable reference-versus-generated
distribution drift, complete exposure counts, immutable cache replay, and the
default suite.

## Phase 4 — Natural Evaluation Stories and World Probes

- Use evaluation prompt families and request namespaces disjoint from
  training. Generate complete held-out natural stories for NLL, perplexity,
  narrative quality, and forgetting, plus unfinished natural vignettes with
  controlled candidate continuations for exact world competence.
- Derive distractors from the counterbalance matrix. Prefer candidate pools
  differing by at most one GPT-2 token; normalize primary NLL over active
  candidate tokens and report unnormalized total NLL as sensitivity analysis.
  Balance candidate order independently and require all insertions to be
  grammatical.
- Generate each pre-answer story to at least 192 tokens. The 64-, 128-, and
  192-token router contexts are nested final slices ending at the same answer
  boundary with identical candidates and no synthetic padding.
- Compute cue strata from visible natural text and evidence spans:
  `cue_sufficient`, `cue_present`, `cue_hidden_or_ambiguous`, and independently
  generated `cue_free_control`. Literal task and family IDs never appear.
- Initially support direct arbitrary recall, convention application in a new
  plot, contextual revision, new-character convention use, and open-book
  control. Defer two-hop and cross-branch probes.

**Phase 4 gate:** before any adapter training, frozen closed-book direct
accuracy is 20–35% and statistically compatible with 25%; answer and position
biases are immaterial; open-book accuracy is at least 65% and substantially
higher; a stronger verifier finds exactly one ledger answer; deterministic
checks find no leak or split overlap; 100 random probes receive human approval;
and the default suite passes.

## Phase 5 — Four-Task Natural-Data Calibration

- Use two disjoint counterbalanced families and the interleaved order
  `willow_seed`, `sunny_seed`, `willow_winter`, `sunny_winter`.
- Initial configuration: 8 direct facts/task, 2 conventions/seed, 1 contextual
  revision/revision task, 16 exposures/fact, 500 accepted stories/task, 128
  validation probes/task, 256 locked test probes/task, rank 8, and 1,000
  updates.
- Generate prose once, then vary one axis at a time using deterministic subsets:
  exposures `{8, 16, 32}`, updates `{500, 1,000, 2,000}`, and rank
  `{4, 8, 16}`. Never regenerate data between trials.
- Require independent direct recall to improve at least 25 percentage points
  over frozen and reach at least 60%; VAMP oracle within five points of
  independent; exact seed-node preservation; winter and ordinary seed
  competence above 60%; measurable sequential interference; held-out natural
  NLL degradation no more than 0.10 versus independent; entailment above 95%;
  and contradiction below 1%.

If the dataset sensitivity gate fails, return to the data contract rather than
increasing LoRA capacity. Open the locked test exactly once after validation
passes.

**Phase 5 gate:** publish a hashed passing calibration profile and one locked
test result with zero committed-node drift. A stopped result contains no
profile and cannot authorize Phase 6.

## Phase 6 — Interleaved Eight-Task VAMP Pilot

- Use two families with `seed`, `spring`, `winter`, and `winter_extension`
  stages in the prescribed family-interleaved order. Do not add cross-family
  bridges yet.
- Evaluate frozen base, sequential LoRA, independent root LoRA, VAMP
  true-parent, selected-parent and oracle, exhaustive routing, Hopfield,
  uniform-start EBT, Hopfield-start EBT, random valid node, and wrong-family
  parent counterfactuals.
- Preserve stored/routed competence and forgetting, transfer, persistent
  memory, addressing cost, parent selection, and graph-shape metrics. Add
  reference/generated NLL, fact entailment, contradiction, candidate balance,
  cue visibility, probe-type/context-length slices, and click-through access
  to underlying prompts, stories, ledger facts, verifier records, and probes.
- Keep the dataset artifact fixed across methods and enforce the existing
  RTX 4090 12 GiB allocator-peak gate.

**Phase 6 gate:** every proof and ledger answer remains valid; every required
task, stage, method, prefix, cue stratum, and probe class is present; metrics
are finite; committed-node drift is zero; test data never participated in
selection; and the report rebuilds byte-identically from its completed result.
Scientific hypothesis failures remain visible results.

## Phase 7 — Long Streams and Consolidation-Ready Worlds

- Scale fixed-dataset experiments through 8, 16, 32, and 64 tasks over four
  recurring world families. Evaluate a fixed sentinel set after every task and
  the full held-out set at powers of two and final.
- Hold out new plots around known facts, new characters using conventions, new
  combinations on one path, contextual revisions, and cross-family visits or
  bridges.
- Introduce bridge probes only after the eight-task pilot demonstrates basic
  learnability. Use them to measure hard-path limitations, Hopfield retrieval,
  EBT mixtures, and best-node regret.
- Without changing the dataset, compare the unconsolidated graph, losslessly
  folded paths, family macro-nodes, and macro-node-plus-task residuals on
  memory, address cost, retained competence, generalization, and transfer.

**Phase 7 gate:** each scale and consolidation comparison has identical data
identity, complete sentinel/full-set coverage, validated ledger semantics,
finite metrics, zero forbidden mutation, reproducible reports, and explicit
resource/cost evidence.

## Cross-Phase Validation and Artifact Rules

- Semantic verification uses a separately pinned, preferably stronger model
  and returns required-fact entailment, contradictions, new controlled claims,
  exact evidence, and leakage. TinyStories-style verification is a separate
  blinded rubric over vocabulary, simplicity, grammar, coherence, plot,
  repetition, and meta-language.
- Deterministic validation checks words/names, exact evidence substrings,
  forbidden alternatives, balance, ledger assignments, split/request hashes,
  token capacity, shared answer boundaries, EOS/masks, internal labels, and
  long n-gram copying.
- Compare genuine GPT-4 references, neutral selected-model stories, and
  world-conditioned stories. The neutral-to-world shift is the critical test
  of whether world constraints damage the chosen generator distribution.
- Every long-running script immediately prints its temporary artifact
  directory, emits human-readable phase lines plus phase/overall ETA progress,
  appends sequential JSONL in batches, validates before publication, and
  atomically promotes only a complete artifact.
- Each job writes content-addressed requests, batch/submission metadata, raw
  responses, accepted/rejected observations, and a manifest. Missing-only
  resume never overwrites raw data; changed models, routes, prompts, schemas,
  settings, or validators create a new version or derived artifact.
- Update this tracker and `PLAN.md` after every phase; put durable boundaries
  in `DESIGN.md`. Run the complete default suite after every phase and the
  relevant marked integrations before promotion.
