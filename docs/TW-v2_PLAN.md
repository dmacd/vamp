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
| 1. Reference profile and generator bakeoff | **Stopped — direct bakeoff complete; neither route qualified** | A new versioned run must produce a strictly valid Qwen/GPT-5.4-Mini 2×200 artifact with at least one automated passer, byte-identical derived replay, a complete default-suite pass, and explicit approval of its exact 50/50 blinded-audit digest |
| 2. Counterbalanced world bibles | Blocked by Phase 1 | World audit and deterministic counterbalance checks pass |
| 3. Natural training corpus | Blocked by Phase 2 | The 150-story sample audit receives explicit human approval, then the complete corpus validates |
| 4. Natural evaluation data | Blocked by Phase 3 | Frozen-base sensitivity, balance, semantic, leakage, and 100-probe human gates pass before adapter training |
| 5. Four-task calibration | Blocked by Phase 4 | Validation ladder passes and locked test is opened exactly once |
| 6. Eight-task pilot | Blocked by Phase 5 | Complete pilot bundle passes integrity, coverage, drift, and report-reproduction gates |
| 7. Scaling and consolidation studies | Deferred | Phase 6 passes with interpretable learning and routing behavior |

Current checkpoint (2026-07-19): Phase 1 has been simplified to a direct paired
comparison of Qwen 3.5 35B-A3B and GPT-5.4 Mini. Both are full corpus-author
candidates and each receives the same 200 neutral briefs. The active V4 request
returns exactly one `story` field; required-word spans, visible feature
realization, forbidden forms, length, and token-form evidence are derived
locally. Semantic narrative-feature heuristics are report-only; only exact
mechanical requirements can reject a story. There is no 50-story screen,
finalist expansion, or third-model verifier. Consequently the complete paid
plan is 400 author requests, followed
by frozen TinyStories vocabulary/token/NLL measurements and a blinded audit of
100 genuine and 100 generated stories, exactly 50 per author. The active output
path is `data/tinyworlds-v2/reference-two-route-v2/` and it uses a separate
content-addressed cache. The already authenticated reference artifact is reused
directly, so the original source archive and 30-minute profiling path are not
repeated. The live catalog pinned Qwen to Alibaba and GPT-5.4 Mini to Azure.
The exact preflight estimated `$0.305313`, reserved `$1.683770` for the
two-attempt policy, and passed the `$15` cap. All 400 author calls then completed
for `$0.1906939625`; no provider-reported cost is missing. Rebuilding V2 from
the same raw cache made no new completion POSTs.

The terminal artifact is
`data/tinyworlds-v2/reference-two-route-v2/`, manifest
`6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9`,
with status `no_quality_qualified_route`. Its exact balanced audit digest is
`a5d9da91fe9636bda942e1f4532620e7761d4c722358f5cb0e1443fa042fff3a`.
GPT-5.4 Mini accepted 192/200 briefs (96%) and Qwen accepted 123/200 (61.5%).
Both passed the vocabulary-coverage and token-unigram-JSD gates and produced no
alphanumeric identifier contamination. Both failed TinyStories-8M NLL,
story-length, paragraph-format, and dialogue-distribution gates. GPT was the
clearer candidate—alignment distance 1.665 versus Qwen's 2.777—but still did
not qualify. The paragraph metric may partly measure a formatting mismatch:
the paired released references have no counted double-newline breaks even
though the released prompts request paragraphs. Inspect the blinded audit
before revising that measurement or paying for another run.

The 73-test core bakeoff suite passes and the completed V2 artifact passes its
strict loader, including raw request/route and cost-journal evidence, cost
arithmetic, direct-quality/status consistency, and audit packet/key/HTML
balance. The broader legacy replay tests were stopped when they entered
their known 20--30-minute cache fixture; because no route passed, the phase
cannot advance regardless. Zero-network V2 derived replay and a post-change
complete default-suite run remain mandatory before any future passing artifact
can be promoted. Phase 2 remains blocked.

The first post-stop prompt review is now complete at
`data/tinyworlds-v2/prompt-tuning-v1/`, manifest
`074cdacbc38e311a85de988801a8c5d2cef561fd88b19daa43640176162836f3`.
It is intentionally a 20-brief development diagnostic, not a smaller Phase 1
gate: the same 20 briefs bind 40 cached V4 controls and exactly 40 new V6 calls
(20 per author), and `review.html` displays every tuned output beside its
control and matched reference. The live expected/conservative costs were
`$0.039824` / `$0.174217` under a separate `$1` cap; exact billed cost was
`$0.0244546375`. V6 requested 130--170 words plus reference-shaped cadence and
formatting. Qwen improved from 13/20 to 14/20 accepted, 75 to 110.5 median
accepted words, and 2.498 to 2.133 median TinyStories-8M NLL. GPT stayed 20/20
accepted, improved from 90.5 to 116.5 words, and moved from 2.296 to 2.262 NLL.
Only one Qwen and four GPT outputs reached the requested word interval, and
both routes continued to serialize blank lines between paragraphs. No route
qualifies and Phase 2 remains blocked. These 20 examples, whose matched
references are unusually long (median 172 words), are now development data;
any future gate run requires a disjoint held-out cohort. The 78-test focused
generation/artifact suite and hardened semantic/raw-evidence reload pass. The
complete default suite was not repeated for this diagnostic and is still
required before a future Phase 1 promotion.

The next isolated prompt-shape diagnostic is complete. V7 removes the
wrapper's duplicate compression cues and places concrete output requirements
last: one newline-free story block, exact required-word forms, quoted dialogue
when requested, 18--20 complete sentences, at least six connected events, and
a soft 155--190-word target. It reused the 40 cached V6 controls and made
exactly 40 new calls, 20 per author. The preflight was `$0.041028` expected /
`$0.179473` conservative under the separate `$1` cap; the calls completed in
about 15 seconds for `$0.0296057500`. The immutable paid-output artifact is
`data/tinyworlds-v2/prompt-tuning-v2/`, manifest
`838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a`.
It eliminated newlines and brought median accepted length to 154.5 words for
Qwen and 153.5 for GPT. Qwen acceptance did not improve (14/20), GPT fell from
20/20 to 18/20, and median TinyStories-8M NLL worsened from 2.133 to 2.568 and
from 2.262 to 2.781 respectively. The experiment therefore supports a concrete
failure mode: a detailed shape checklist can improve visible compliance while
moving prose away from the base model's learned TinyStories distribution.

The initial V7 comparison is not the authoritative quality interpretation.
An exact normalized-content audit found that 3,393 of the selected 10,000
GPT-4 validation records also occur in the pinned original training file.
Nine of the 20 small paired archive references likewise overlap training.
`prompt-tuning-v2` is preserved unchanged as paid-generation evidence, while
the zero-call corrected interpretation is
`data/tinyworlds-v2/prompt-tuning-v3/`, manifest
`50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74`.
V3 filters the overlap with NFKC, case-folding, whitespace collapse, SHA-256
prefiltering, and full normalized-text confirmation; its reference profile is
built from the remaining 6,607 validation stories. It reuses all 80 cached
stories and 66 accepted NLL measurements and incurs zero new calls, cost, or
GPU measurements. Seen-versus-unseen reference medians and generation gaps are
nearly the same, so overlap invalidated the clean-holdout claim but does not
explain the central NLL mismatch. The nominal composite score still ranks V7
first because length and newline conformity offset other errors; that rank is
descriptive only because every V7 route still fails hard gates. The corrected
40-sample review is `prompt-tuning-v3/review.html`. The current focused suite
passes 94 tests; the complete default suite was not repeated because this
development diagnostic cannot promote Phase 1.

That prompt-envelope hypothesis has now been tested directly in V8. Each
request contains one user message formed as the exact archived prompt plus
`Possible story:`. There is no system message, repeated instruction, JSON
request, response schema, or added length/shape language. Pinned route, seed,
512-token ceiling, disabled reasoning, data-denial, and no-fallback fields are
transport controls rather than natural-language instructions. The complete
plain assistant message is retained unchanged as the story, and required-word,
feature, safety, length, and token-form evidence remains locally derived. V8
uses V7 as its exact cached control and the train-decontaminated 6,607-story
validation profile as its comparator.

The exact preflight was `$0.034422` expected / `$0.152463` conservative under
the separate `$1` cap. All 40 calls completed in 17 seconds for
`$0.0155166000`. The result is
`data/tinyworlds-v2/prompt-tuning-v4/`, manifest
`362a0c85c7722fbaf36120eaa5479285edb798bc067d8f7c7fd41631571e2bb0`,
and all samples are in `review.html`. Both routes moved to 20/20 mechanically
accepted samples. GPT median NLL improved sharply from 2.781 to 2.185, and Qwen
improved from 2.568 to 2.475, against a 1.347 validation median. At the same
time, the bare released prompt restored short-story behavior: GPT produced a
median 80 words and Qwen 113.5, versus 138 in validation, and all new stories
contained paragraph breaks. Both still fail NLL and token-unigram gates; GPT
also fails story length badly, while Qwen narrowly misses the 15% length band
and has overlong pooled sentences. Composite distance therefore retains V7
for both routes despite V8's acceptance/NLL gains. The result establishes that
the wrapper caused part of GPT's mismatch but was not the only cause. No route
qualifies and Phase 2 remains blocked. The strict persisted-evidence reload and
complete focused generation/comparator suite pass; the long default suite was
not repeated for a development-only non-passer.

The isolated minimal-length experiment is also complete. V9 differs from V8
by exactly one inserted sentence: `Aim for about 130 to 150 words.` Briefs,
model/provider routes, deterministic provider seeds, token ceiling, response
handling, local validator, decontaminated comparator, and every other technical
field remain identical. V8's 40 generated controls and NLLs are loaded from
the strict V4 artifact without rescoring. The V9 preflight was `$0.034641`
expected / `$0.153339` conservative under the `$1` cap; generation completed
in 15 seconds and the 40 calls cost `$0.0220921000`. The result is
`data/tinyworlds-v2/prompt-tuning-v5/`, manifest
`1605d21acff2647fe4be456a627653f606b7e4e90c7241d3d552ebe513430c73`,
with the exact prompt and every output in `review.html`.

V9 fixed median length for both models, but not identically. Qwen moved from
113.5 to 147 words, improved median NLL from 2.475 to 2.339, token JSD from
0.324 to 0.293, and composite distance from 2.644 to 2.550. It fell from 20/20
to 18/20 accepted because two stories omitted required word forms; nevertheless
V9 is the better Qwen cell. GPT moved from 80 to 128 words and improved token
JSD from 0.352 to 0.314, but median NLL worsened from 2.185 to 2.381. Its V8
and V9 composite distances are 2.2645 and 2.2653, so bare V8 retains a nominal
0.0008 lead despite failing length. Both V9 routes pass the 15% median-length
band, but neither passes NLL, token-unigram, sentence-length,
paragraph-serialization, or dialogue-distribution gates. Qwen also fails the
85% acceptance threshold. No route qualifies and Phase 2 remains blocked.
The artifact strictly reloads and the 106-test focused suite passes; the long
default suite was not repeated for a development-only non-passer.

A separate learnability sidebar now tests whether the measured prose mismatch
actually prevents this TinyStories-8M/LoRA stack from storing and using a tiny
world. It is deliberately not a Phase 1 gate or prompt-selection cell. Each of
three matched arms contains the same eight arbitrary child-to-badge facts and
four badge-to-meeting-place rules, 24 documents, canonical leading evidence,
rank-8 initialization, RNG state, batch, and 512-update schedule. Only the
continuation author changes: decontaminated official TinyStories prose, Qwen
3.5 35B-A3B, or GPT-5.4 Mini. The paid boundary was 72 calls (three candidates
per evidence per external author, selecting the first two mechanically valid
stories) under a `$0.50` cap; exact spend was `$0.0392434000`.

The strict result is `data/tinyworlds-v2/reasoning-sidebar-v1/`, manifest
`59200a624dcc8e2afe4cfcdb720d22724184eb97797d7da8208cf0b527d797fe`.
Frozen corpus NLL was 1.621 for the TinyStories control, 2.436 for Qwen, and
2.806 for GPT, confirming a surface-distribution gap. Every adapted corpus NLL
fell below 0.028. Yet held-out test direct recall was 25.0%, 25.0%, and 31.2%
respectively, and all three arms scored 25.0% on one-hop fact-plus-rule probes,
where 25% is chance. A no-training follow-up reused the saved adapters and
asked for the next word after each exact evidence prefix. Every arm scored
100% on all eight fact clauses and all four rule clauses, with large positive
margins. That diagnostic is
`data/tinyworlds-v2/reasoning-sidebar-v1-clause-probe/`, manifest
`1d1d8a7921e4ab74b4b23d57266da776d06bf01b3effe5fecc6a92ed5a318b6f`.

The mechanism conclusion is narrower and more useful than an author ranking:
the current next-token LoRA setup memorizes literal continuations but does not
turn them into stable paraphrase-invariant bindings or one-hop compositional
knowledge. Since the official-TinyStories control fails too, this experiment
does not identify an incremental Qwen/GPT author penalty despite their higher
frozen NLLs. Before using distribution gates as corpus-author vetoes, a future
learnability comparison must first establish a passing in-distribution control,
most likely by adding explicit query-style supervision or changing the
adaptation objective. This sidebar leaves Phase 1 stopped and Phase 2 blocked.
Both artifacts strictly reload, and the 17-test sidebar, shared LoRA workflow,
and candidate-scoring suite passes. The complete default suite was not repeated
because this diagnostic cannot promote Phase 1.

Do not purchase another prompt cell before inspecting V5. Preserve the
route-specific conclusion rather than averaging it away, and require any next
change to isolate one named remaining failure. Reintroducing a system message,
JSON, and narrative checklist together would lose the causal isolation gained
here.

The completed `reference-two-route-v1` artifact is retained as a validator
diagnostic. All 400 requests completed for `$0.1906939625`, but V1 incorrectly
treated lexical patterns for moral, conflict, foreshadowing, twist, and ending
valence as hard semantic proof. The active V2 artifact reuses those exact raw
responses without new completion POSTs and hard-gates only whole-word
ingredients, safety/length, and structurally quoted dialogue. Other narrative
features remain report/audit fields.

The historical seven-route Phase 1 implementation remains in place through the artifact-integrity boundary. It covers the exact
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

The live production attempt reached a controlled `blocked_by_cost_cap` stop.
After explicit zero-BYOK confirmation, it authenticated and profiled the pinned
sources, ran the real-GPU measurements, resolved all live routes, and computed
an expected spend of `$3.439507`. The mandatory two-attempt exposure was
`$20.020653`, above the fixed `$15.000000` cap; the 800-request GPT-5.4
verifier reserve contributed `$17.544000`. The inference key was therefore not
read and no completion POST, charge, or paid sample exists.

The stopped artifact is `data/tinyworlds-v2/reference/`, with manifest
`28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9`.
Strict semantic validation passes, and zero-network replay reproduces all 31
derived files (101,081 bytes). No human audit or Phase 1 gate result exists. A
redundant post-artifact default-suite rerun was intentionally interrupted at
85% before repeating the known 20--25-minute production-cache fixture; it had
no failures, no code changed after the complete pre-run pass, and the artifact
validation and replay gates passed independently.

Three fixed 3-brief-by-7-route previews now record the request-contract
corrections. The v1 artifact at
`data/tinyworlds-v2/previews/phase1-route-preview-3x7-v1/`, manifest
`1ddba6e0862de3e416b4ce21538f5471723e823d6c39c5a32da27a0ea72596b6`,
is archived evidence only: it enabled OpenRouter's `enforce_distillable_text`
switch and consequently blocked four candidate models before a useful quality
comparison. That framing was wrong for this work. The models are being asked to
author constrained synthetic stories, not to transfer their general behavior
to a student through serious model distillation.

The corrected v2 request contract removed that routing restriction, used signed
31-bit seeds, and made the structured-output instruction explicit. Its live run
was interrupted after 13 paid attempts. The thirteenth response, the first Qwen
sample, accepted a nominal 512-token visible-output limit but returned 5,236
output tokens, of which 5,138 were hidden reasoning tokens, and cost
`$0.006837025`. Exact provider spend across all 13 attempts was `$0.008248631`.
This exceeded that request's `$0.000987025` precomputed bound and exposed a
provider-specific reasoning-control gap; the interrupted temporary tree and raw
cache are retained as billing/protocol evidence, not promoted as a preview.

The v3 request contract explicitly sets `reasoning.effort=none` for Qwen and
Gemini, the two preview routes with relevant optional-reasoning support. It was
authorized only for the `$0.041751369` residual of a `$0.05` cumulative cap
after debiting the exact v2 spend. The completed artifact is
`data/tinyworlds-v2/previews/phase1-route-preview-3x7-v3/`, manifest
`6e1aa9697d8e62263a49c6bc8d22aa22bcb568ca4e551e68b75c727ab063d9f0`.
All 21 outcomes strictly validate and replay without network access. Five pass
the current deterministic screen: Mistral 1/3, Gemini 3/3, and GPT-5.4 Mini
1/3; Ling, Gemma, DeepSeek, and Qwen are 0/3. Provider-reported v3 spend is
`$0.0061824395`; one Gemma gateway timeout has no reported cost and is charged
at its full `$0.00063045` bound, making v3 conservative ledger exposure
`$0.0068128895`. Exact cumulative provider spend for v2 plus v3 is
`$0.0144310705`, and cumulative conservative exposure including that unknown
charge is `$0.0150615205`.

Phase 2 remains blocked. The v3 artifact has `scientific_role=diagnostic_only`,
is ineligible for route selection, and awaits human review of both accepted and
rejected stories. It is now superseded operationally by the explicitly
authorized two-author V4 comparison, which has a new artifact identity and cost
preflight and cannot overwrite the stopped reference tree or any preview.

Preliminary story inspection exposes a flaw in the current acceptance metric.
All three coherent Qwen stories failed only because Qwen returned different
evidence field names; two strong GPT stories failed because their self-reported
feature quotes did not exactly match the prose; meanwhile weaker Gemini prose
passed all three records. V4 fixes this by deriving observable evidence locally
from the story and treating the one-field response envelope as transport.

A post-run harness audit found and fixed four non-scientific safety edges:
bounded missing-cost responses now require all four generation-stat lookups;
complete-cache publication recovery loads neither a fresh API key nor a stale
BYOK attestation; promotion reserves the destination before renaming; and the
validator rederives runtime, actual, unknown, per-route, and cumulative costs
from raw responses and the immutable journal. The focused 107-test suite and
zero-network replay of both v1 and v3 pass after these changes; no additional
paid request was made and both artifact manifests remain unchanged.

## Non-negotiable V2 Contract

- Implement V2 in a parallel `tinyworlds_v2` package and versioned artifact
  tree. Do not alter, reinterpret, resume, or import prose from v1.
- Never use `TinyWorldsTemplateRegistry`, `_fact_statement`,
  `_rule_statement`, `_query_statement`, `_fit_exact_tokens`, padding
  fragments, artificial cue blocks, formal relation questions, or exact-token
  English in the v2 data path.
- Every training story and natural knowledge probe comes from a pinned external
  model and provider route through a content-addressed, resumable cache.
- External models author synthetic data; this is not a claim that the benchmark
  reproduces their general capabilities. Do not use a platform routing label as
  a proxy for story quality. Qwen 3.5 35B-A3B and GPT-5.4 Mini are both active
  corpus-author candidates for this open research benchmark.
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
- Refresh the public catalog immediately before every paid author-route batch,
  then compare its semantic lock with the original one. Halt
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
- Use signed 31-bit deterministic seeds and say `JSON` explicitly in every
  structured-output prompt. Do not enable `enforce_distillable_text`. V4 asks
  Qwen and GPT-5.4 Mini for exactly `{"story": ...}` and pins optional reasoning
  to `none`; any change creates a new request-contract identity.
- Retry only transport, rate-limit, and server failures. Schema-invalid or
  semantically invalid responses remain immutable failed observations and
  count against route reliability.

### 1.3 Direct two-author comparison

Send all 200 paired neutral briefs to both pinned author routes, for exactly 400
generation requests. There is no screen, finalist stage, or external verifier.
Resolve and persist the exact OpenRouter model/provider route and live price
immediately before preflight.

| Author route | Plan-time input / output price per 1M tokens |
|---|---:|
| Qwen 3.5 35B-A3B | `$0.14 / $1.00` |
| GPT-5.4 Mini | `$0.75 / $4.50` |

The historical seven-route table and 3-by-7 previews are immutable diagnostic
evidence only. They do not select an author and none of their prose enters the
corpus. Both active authors proceed through the complete 200-brief comparison;
selection occurs only after automated metrics and the balanced human audit.

### 1.4 Cost preflight and ceilings

- Before any billable request, write and print `cost_estimates.json` from the
  completed request bodies, current pinned-route prices, matched-reference
  output lengths, one full retry allowance, and worst-case in-flight
  reservations. The request counts must be exactly 200 Qwen and 200 GPT-5.4
  Mini; there is no verifier reserve.
- Keep the hard inclusive ceiling at `$15`. Print the current expected and
  conservative totals before reading the inference key, and submit nothing if
  the conservative total crosses the cap.
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
  Neither case may be reposted. Stopped artifacts retain exact author-route
  attribution for both actual and conservative charges.
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

Score all 200 observations for each author against matched genuine
references. A route qualifies only if every gate passes:

- Schema-valid responses at least 99%; deterministic acceptance at least 95%.
- Requested noun, verb, and adjective whole-word adherence at least 95%.
  Structurally quoted dialogue is a hard check when dialogue is requested.
  Moral, conflict, foreshadowing, twist, and ending valence are reported from
  deterministic heuristics but are not hard semantic proof or rejection gates;
  the blinded human audit judges them.
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
- The independent human audit supplies simplicity/coherence and TS-like
  judgment; no model judges its own prose or the other author's prose.

If the gates cannot be met within the cost ceiling, publish either
`no_quality_qualified_route` or `blocked_by_cost_cap`, including every failed
metric and the remaining-cost estimate. Do not reduce sample counts or relax
thresholds.

### 1.6 Mandatory human audit

- Produce a static interactive audit with 100 matched genuine GPT-4 references
  and 100 generated controls. Balance generated controls exactly 50 Qwen and 50
  GPT-5.4 Mini and randomize them deterministically. Both authors remain visible
  to the hidden key; only automated-qualified routes are eligible for final
  selection.
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

Publish the active comparison beneath
`data/tinyworlds-v2/reference-two-route-v2/`; preserve
`data/tinyworlds-v2/reference/` as its immutable reference-profile parent:

- `neutral_story_briefs.jsonl` plus a digest-bound `reference_binding.json` to
  the already validated parent, avoiding duplication or recomputation of its
  source/profile files;
- `reference_binding.json`, `configuration.json`, exact raw `catalog/`
  responses and normalized route locks, one generation measurement batch and
  runtime, `generator_bakeoff.jsonl`, `cost_estimates.json`,
  `cost_actuals.json`, `quality_comparisons.json`, `quality_details.json`,
  `status.json`, `audit.html`, `audit_packet.json`, and `audit_key.json` as
  applicable to the terminal status; there is no verifier or finalist artifact;
- the human-authored overlays `audit_decisions.json`,
  `audit_approval_request.json`, and `audit_approval.json`, which remain
  outside the immutable base manifest;
- one directory per author containing `requests.jsonl`, `plan.json`,
  `accepted.jsonl`, and `rejected.jsonl`, plus an authenticated copy of every
  submitted raw request, response, stats observation, full route lock,
  per-reservation sanitized BYOK authorization, and runtime-cost journal entry
  under `raw_cache/`.

Every manifest binds source pins, request and response hashes, semantic route
identity and full catalog provenance, prompt/schema and transport-protocol
versions, validator/measurement versions, prices, costs, and all file digests.
The active strict loader cross-checks the exact parent digest, 2×200 V4 request
plan, route order, completed result identities, locally rederived story
evidence, accepted/rejected partitions, and measurement coverage rather than
trusting the root digest alone. Its terminal statuses are
`blocked_by_cost_cap`, `no_quality_qualified_route`,
`audit_insufficient_accepted_samples`, and `awaiting_human_audit`. Runtime
billing/contract failures retain their dedicated raw cache and temporary tree
for safe resume instead of being converted into a scientific result.

Before the Phase 1 gate can pass, derived replay must start only from persisted briefs, route locks, raw HTTP/cache
evidence, and accelerator-derived measurements. With a transport that raises
on any network access, it reconstructs canonical requests, reparses every
attempted response discovered in the raw cache, including interrupted terminal
attempts absent from committed result streams, reruns source
joins, deterministic validators and quality selection, rebuilds route
streams and the blinded audit, and byte-compares every replayed derived file
(apart from the replay tree's own root manifest). Raw provider, BYOK, and
response-contract evidence must agree with the published terminal cause and
status. It never reopens the TinyStories sources, loads the
tokenizer/checkpoint or GPU, reads an API key, or performs a network call.
Before a future artifact can pass the phase gate, public completed-artifact
validation and human approval must both invoke this complete semantic and
replay validation; passing an overlay-only digest check is insufficient. The
current V2 strict loader does not yet implement the derived replay requirement.

The completed V2 artifact has manifest
`6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9`,
audit digest
`a5d9da91fe9636bda942e1f4532620e7761d4c722358f5cb0e1443fa042fff3a`,
and terminal status `no_quality_qualified_route`. It is valid evidence, not a
passing Phase 1 artifact. Its audit remains useful for diagnosis, but no
approval overlay can make an automated non-passer eligible for Phase 2.

Audit construction uses an exact deterministic balanced assignment over
distinct pair IDs, not a greedy sampler. Both authors remain in the
blinded generated controls; only automated-qualified routes can be selected
after human scoring. If accepted samples cannot satisfy the exact per-route
quotas, publish `audit_insufficient_accepted_samples` with feasibility evidence
instead of changing counts or silently substituting examples.

Tests cover source verification and streaming selection, deterministic shard
merging, request hashes, route/catalog drift, immutable cache behavior,
retry classification, schema rejection, cost arithmetic and cap enforcement,
crash and cross-process recovery, BYOK fail-closed behavior, quality metrics,
direct route selection, exact blinded-audit allocation, semantic tamper rejection,
zero-network replay, and approval-digest enforcement. Default tests use fake
transports and small fixtures. Marked integrations cover the real archive,
tokenizer/checkpoint NLL, and minimal OpenRouter smoke requests. Run the focused
paid-boundary suite before generation and the complete default suite after the
completed artifact, before Phase 1 can advance.

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
- Request story-only responses and derive observable evidence locally. The
  ledger checker and human gate remain independent of the author model.
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
