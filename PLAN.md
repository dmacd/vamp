# Development Plan

## Active Milestone: TinyWorlds-P Partitioned TinyStories Base

The active roadmap is `tinyworlds-p-archive-v1`, tracked in
[`docs/TW-P_PLAN.md`](docs/TW-P_PLAN.md). It replaces generated TinyWorlds prose
with unmodified stories taken directly from released records in the pinned
`TinyStories_all_data.tar.gz` archive. Five noun-bucket by verb-bucket cells are
withheld from a freshly initialized eight-layer GPT-Neo base; only story text
reaches the model. All base, world, control, validation, and sealed-test sets
are derived from eligible archive entities. The original TinyStories train,
validation, and GPT-4-only text aggregates are irrelevant to this benchmark
and are not inputs. TinyWorlds-v2 external generation is parked as
non-qualifying historical evidence. LoRA/VAMP continual episodes are deferred
until the archive-only partition and scratch base pass their publication gates.

### TinyWorlds-P Status

- **Authoritative archive-only source decision (2026-07-21).** The pinned
  `TinyStories_all_data.tar.gz` records are the complete source universe.
  Archive entities with mechanically recoverable noun, verb, and adjective
  roles are grouped, bucketed, and assigned directly. Base train, held-in
  validation/test, five world train/validation/test splits, and matched
  controls all come exclusively from that eligible archive universe. There is
  no corpus/archive join and no dependency on any published TinyStories text
  aggregate.
- **Prior partition and calibration are superseded.** The prior 8x8 partition,
  scratch run, and 6x6 conclusion were produced from the obsolete
  corpus-intersection universe. They remain immutable historical diagnostics,
  but they are not TinyWorlds-P publication candidates and must not be resumed,
  selected, or used to set the new partition. Exact identifiers remain only in
  the historical audit documents linked below.
- **Corpus-intersection implementation purged.** The obsolete join and all
  corpus-backed paths, identities, offsets, gates, loaders, runners, and tests
  are gone. The purge-only checkpoint remains in history; every subsequently
  restored surface is archive-native and rejects old artifact identities.
- **Archive-native ingestion implemented.** The TinyWorlds-P-owned parser
  authenticates and streams the tarball once, writes exact story bytes to an
  archive-order spool, classifies bounded batches in physical workers,
  externally sorts by normalized story identity and record ID, groups every
  occurrence with its provenance and multiplicity, audits all exclusions, and
  enforces only the 95% token-weighted role-classification gate.
- **Archive-only partition artifacts restored.** Partition construction now
  derives every bucket, cell, split, and control from eligible archive groups;
  publishes exact story and uint16 token shards under
  `data/tinyworlds-p-archive/v1/`; binds documents to member, member-local
  index, record hash, and story hash; and strictly rejects old source keys.
  The CPU 3x3 fixture reconstructs source/token bytes, checks leakage and
  globally unique controls, rejects tampering, and rebuilds byte-identically
  across worker/run settings. The fixed preparation runner and opt-in real
  archive replay have been restored.
- **Canonical archive partition built and reproduced.** The strict 8x8 artifact
  is
  `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`.
  It contains all 4,966,067 eligible archive records (945,499,161 active
  tokens), passes the 99.968% token-weighted role gate, retains every agreeing
  duplicate occurrence, excludes six conflicting duplicate groups, and passes
  topology, component-visibility, split-marginal, globally unique-control,
  exact-byte reconstruction, and sealed-test isolation checks. A second build
  with 24 workers and a different external-sort run size authenticated to the
  same identity and exact `tree.json`, proving byte equality for every strict
  tree entry. The opt-in acceptance test took 39m08s end to end; the build took
  35m11s and its final parallel strict reload took 3m56s. Long real-source
  gates remain excluded from the default test suite.
- **Archive-only scratch training restored.** Memory-mapped batching,
  token-weighted accumulation, immutable complete resume states, streaming
  validation, one-shot sealed test, milestone publication, and the fixed GPU
  runner now consume only strict archive-v1 artifacts. Low-gap fallback uses a
  fresh 6x6 partition with 94/3/3 held-in splits; excessive-gap fallback uses
  10x10 with 96/2/2. Training and every validation/sealed-test batch report
  throttled, measured phase and pass-path ETAs. CPU tests prove
  interrupted/resumed state and trace parity, schedule and selection boundaries,
  old-resume rejection, finite evaluation, and exact evaluation progress.
- **Focused CPU/shared checks pass.** The 81-test TinyWorlds-P, GPT-Neo,
  checkpoint, and training-state suite passes in four concurrent groups
  (10.3s wall time); parked TinyWorlds-v2 tests are still collection-skipped.
- **GPU smoke passes.** The opt-in RTX 4090 smoke strictly loaded the real tree,
  compiled production training, wrote an interrupted update-one state, resumed
  through update two, and measured an 8.695 GiB JAX allocator peak against the
  12 GiB gate. Splitting strict semantics into assignment, provenance, and
  shard/index proof passes reduced the full smoke from 5m20s to 4m20s.
- **Current execution sequence.** Complete the fresh seed-zero RTX 4090
  calibration, apply its predeclared one-fallback policy, then train and publish
  the selected archive-only base if the gap and quality gates pass. The fixed
  runner derives exact update counts and phase/overall estimates from
  archive-only active-token mass and reports adaptive ETAs.
- **Archive-v1 calibration active (2026-07-21).** A fresh seed-zero 8x8 run is
  training in
  `checkpoints/tinyworlds-p-archive-v1/work/base-archive-v1-mu03o__f` after a
  3m08s strict load. The bound schedule is 150,653 microbatches and 18,832
  optimizer updates per epoch; no historical checkpoint was inspected or
  resumed.
- **Historical audits retained for provenance only.** The original train/archive
  mismatch analysis is preserved in
  [`docs/TW-P_SOURCE_AUDIT.md`](docs/TW-P_SOURCE_AUDIT.md), and the obsolete
  intersection-based calibration is preserved in
  [`docs/TW-P_CALIBRATION_AUDIT.md`](docs/TW-P_CALIBRATION_AUDIT.md). Neither
  audit defines a current source, coverage gate, split, or stopping decision.
- **Test scope direction remains focused.** Every parked TinyWorlds-v2 test is
  collection-skipped; do not run those legacy bodies. After the archive-only
  refactor, run focused TinyWorlds-P and shared GPT-Neo/checkpoint tests before
  and after partition promotion, followed by the host RTX 4090 workflow. No
  continual LoRA or VAMP stream work begins in this milestone.

## Parked Milestone: TinyWorlds-v2 External-Generation Benchmark

The parked roadmap is `tinyworlds-v2-gpt`, tracked in detail in
[`docs/TW-v2_PLAN.md`](docs/TW-v2_PLAN.md). The symbolic world ledger remains
authoritative for truth and scoring, while pinned external language models
produce variable-length natural TinyStories-style text through immutable,
content-addressed request/response caches. V2 does not reuse the v1
deterministic renderer or exact-token prose fitting.

### TinyWorlds-v2 Status

- **Phase 1 — direct Qwen/GPT-5.4-Mini author bakeoff: complete with an
  automated scientific stop (2026-07-19).**
  The active experiment now compares exactly two full author routes on the same
  200 already-profiled neutral briefs: Qwen 3.5 35B-A3B and GPT-5.4 Mini. Both
  models generate 200 stories and both are evaluated as possible later corpus
  authors. There is no 50-story screen, finalist expansion, or third-model
  verifier, so the paid plan is exactly 400 author requests. V4 responses are
  one-field `{"story": ...}` objects; whole-word spans, requested-feature
  realization, forbidden forms, and length evidence are derived locally. Only
  mechanically observable requirements are hard gates; semantic plot features
  remain report and blinded-audit evidence. The two routes are compared using
  the frozen TinyStories vocabulary/token
  distribution, TinyStories-8M NLL, surface statistics, cost, and a blinded
  100-reference/100-generated audit split exactly 50/50 by author. The new
  versioned path is `data/tinyworlds-v2/reference-two-route-v2/`, with its own
  raw cache and a `$15` hard cap. Focused tests for the V4 request, local
  validator, explicit two-route catalog plan, direct quality selector, exact
  2×200 preflight, and cost-cap no-secret stop pass. The already completed
  reference artifact is reused, avoiding the prior corpus scan/profile delay.
  The live preflight resolved Qwen to Alibaba and GPT-5.4 Mini to Azure. Its
  expected cost was `$0.305313`, its conservative two-attempt exposure was
  `$1.683770`, and the run therefore remained below the `$15` cap. All 400
  author calls completed for `$0.1906939625`. The corrected V2 artifact reused
  those exact cached responses without a second paid call and strictly
  validates at `data/tinyworlds-v2/reference-two-route-v2/`, manifest
  `6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9`.
  Its terminal status is `no_quality_qualified_route`; the exact balanced audit
  digest is
  `a5d9da91fe9636bda942e1f4532620e7761d4c722358f5cb0e1443fa042fff3a`.

  GPT-5.4 Mini accepted 192/200 briefs (96%) versus Qwen's 123/200 (61.5%),
  and its aggregate alignment distance was better (1.665 versus 2.777). Both
  routes passed vocabulary coverage and token-unigram JSD, with no
  alphanumeric identifier contamination. Both failed the frozen-base NLL,
  story-length, paragraph-format, and dialogue-distribution gates; GPT's
  median NLL delta was 0.905 and its median story length was 40.9% below the
  matched reference median, while Qwen's were 0.927 and 50.7%. The paragraph
  result is potentially a representation mismatch because the matched released
  references contain no counted double-newline breaks while the prompt asks
  for paragraphs, so the blinded audit must be inspected before changing that
  criterion. Phase 2 remains blocked. The 73-test core bakeoff suite passes and
  the active artifact passes strict validation, including raw request/route and
  cost-journal evidence, cost arithmetic, direct-quality/status consistency,
  and audit packet/key/HTML balance. A broader legacy replay run was
  stopped on entry to its known 20--30-minute cache fixture because the failed
  automated gate already prevents phase advancement; the complete default
  suite and zero-network V2 derived replay remain required before any future
  Phase 1 pass can be promoted.

  A deliberately small prompt-tuning review completed on 2026-07-19 without
  changing that stop. It reused 20 namespaced development briefs and their 40
  cached V4 controls, then generated exactly 20 new V6 stories per model. V6
  asks for 130--170 words and adds ordinary TinyStories cadence, opening,
  sentence, English-vocabulary, ending, and single-line paragraph guidance.
  The exact live preflight was `$0.039824` expected / `$0.174217`
  conservative under a separate `$1` cap; all 40 calls cost `$0.0244546375`.
  The promoted diagnostic is `data/tinyworlds-v2/prompt-tuning-v1/`, manifest
  `074cdacbc38e311a85de988801a8c5d2cef561fd88b19daa43640176162836f3`,
  with all outputs in `review.html`. Qwen acceptance moved from 13/20 to 14/20,
  median accepted length from 75 to 110.5 words, and median TinyStories-8M NLL
  from 2.498 to 2.133. GPT remained 20/20, moved from 90.5 to 116.5 words, and
  from 2.296 to 2.262 NLL. Only 1/20 Qwen and 4/20 GPT outputs actually reached
  the requested 130--170-word interval, and both still emitted blank-line
  paragraph breaks. Thus V6 improved alignment distance for both routes but
  did not fix the length, paragraph, or NLL gates. The 20 matched references
  happen to be long-skewed (median 172 words), so this set is explicitly
  development/review evidence and cannot qualify a route or be reused as a
  clean final holdout. The 78-test focused generation/artifact suite passes,
  as does a hardened offline reload; the loader rebuilds requests, locally derived
  evidence, measurement coverage, quality ranking, cost arithmetic, raw-cache
  attempts, and the settled cost journal. The 34-minute complete default suite
  was not repeated for this non-qualifying diagnostic and remains mandatory
  before any future Phase 1 promotion.

  A second 20-brief prompt-shape diagnostic tested V7 on 2026-07-19. V7 moves
  the concrete length/shape requirements to the end of the user message,
  removes the wrapper's redundant compression cues, requires one newline-free
  story block, and asks for 18--20 sentences, at least six connected events,
  and a soft 155--190-word target. It reused the exact V6 controls and purchased
  only 20 new stories per model. The live preflight was `$0.041028` expected /
  `$0.179473` conservative under the separate `$1` cap; all 40 calls completed
  in about 15 seconds for `$0.0296057500`. The raw diagnostic is
  `data/tinyworlds-v2/prompt-tuning-v2/`, manifest
  `838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a`.
  V7 eliminated newlines and moved median accepted length to 154.5 words for
  Qwen and 153.5 for GPT, but Qwen remained 14/20 accepted and GPT fell from
  20/20 to 18/20. More importantly, median TinyStories-8M NLL worsened from
  2.133 to 2.568 for Qwen and from 2.262 to 2.781 for GPT. The concrete
  checklist repaired surface shape while making the stories less like the
  distribution learned by TinyStories-8M.

  The first V7 quality report also exposed a comparator error: 3,393 of its
  10,000 selected GPT-4 validation stories occur in the pinned original
  TinyStories training file under NFKC/case-folded/whitespace-collapsed exact
  identity. The paid V7 output remains immutable, but that report is retained
  only as contaminated-comparator evidence. A zero-call V3 reevaluation filters
  those overlaps, rebuilds the reference profile from the remaining 6,607
  validation stories, and reuses all 80 cached V6/V7 stories and all 66 accepted
  NLL measurements. It is at `data/tinyworlds-v2/prompt-tuning-v3/`, manifest
  `50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74`,
  with every sample in `review.html`. Nine of the 20 small paired archive
  references also occur in original training, but generated-to-reference NLL
  gaps were nearly unchanged across the seen and unseen subsets. Contamination
  therefore mattered to evaluation hygiene, but does not explain the main
  mismatch. The current composite distance nominally ranks V7 first because
  its length/format match offsets other errors; no V7 route passes the hard
  acceptance, NLL, distribution, and language-feature gates, so that rank is
  not a production-prompt selection. The current 94-test focused generation,
  decontamination, replay, and artifact suite passes. The complete default
  suite was not repeated for this non-qualifying development diagnostic.

  A third 20-brief diagnostic tested the bare released prompt on 2026-07-19.
  V8 sends exactly one user message containing the archived TinyStories prompt
  followed by `Possible story:`. It has no system message, repeated
  instructions, JSON request, response schema, or added length/shape rule; the
  complete plain assistant reply is the story and all evidence is derived
  locally. The only other request fields are transport controls such as the
  pinned route, deterministic seed, output ceiling, and no-fallback policy.
  V8 reused the exact V7 stories as controls and used the decontaminated
  6,607-story validation profile directly. The preflight was `$0.034422`
  expected / `$0.152463` conservative under the `$1` cap. All 40 calls finished
  in 17 seconds and cost `$0.0155166000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v4/`, manifest
  `362a0c85c7722fbaf36120eaa5479285edb798bc067d8f7c7fd41631571e2bb0`.

  Both bare-prompt routes accepted 20/20 and realized all three required word
  roles. GPT median NLL improved from V7's 2.781 to 2.185, while Qwen improved
  from 2.568 to 2.475; the decontaminated validation median is 1.347. The bare
  prompt also restored the released prompt's compression and paragraph
  behavior: GPT fell to an 80-word median and Qwen to 113.5 words, versus 138
  in validation, and every new story used paragraph breaks. GPT's NLL gain is
  strong evidence that the wrapper/checklist caused part of its mismatch, but
  neither bare route passes: both still fail NLL and token-distribution gates,
  GPT is 42.0% short, and Qwen is 17.8% short with longer pooled sentences.
  The composite score retains V7 for both routes because its length/shape fit
  outweighs V8's acceptance and NLL gains. That descriptive choice is not a
  production selection. The active artifact validates from persisted evidence
  and the complete focused V2 generation/comparator suite passes; the long
  default suite was not repeated because this diagnostic cannot promote Phase
  1.

  A fourth diagnostic added exactly one sentence to V8:
  `Aim for about 130 to 150 words.` V9 otherwise preserves the same single user
  message, provider seed, route, technical controls, plain-text response, local
  validation, V8 control outputs, and decontaminated comparator. Its preflight
  was `$0.034641` expected / `$0.153339` conservative; all 40 calls generated
  in 15 seconds and cost `$0.0220921000`. The strict artifact is
  `data/tinyworlds-v2/prompt-tuning-v5/`, manifest
  `1605d21acff2647fe4be456a627653f606b7e4e90c7241d3d552ebe513430c73`.

  The cue repaired length for both models but exposed a route-specific
  tradeoff. Qwen moved from 113.5 to 147 median words, improved median NLL from
  2.475 to 2.339 and token JSD from 0.324 to 0.293, and reduced composite
  distance from 2.644 to 2.550; it is the better Qwen prompt despite falling
  from 20/20 to 18/20 when two stories omitted required word forms. GPT moved
  from 80 to 128 words and improved token JSD from 0.352 to 0.314, but median
  NLL worsened from 2.185 to 2.381. Its composite distances are effectively
  tied, with bare V8 retaining the nominal lead by 0.0008. V9 therefore passes
  the story-length band for both routes, but neither route passes the NLL,
  token-distribution, sentence-length, paragraph-serialization, or dialogue
  distribution gates. The strict artifact reload and 106-test focused suite
  pass; the long default suite was not repeated for a non-promotable diagnostic.

  A matched LoRA learnability sidebar completed on 2026-07-20 without changing
  the Phase 1 stop. It trained rank-8 adapters for 512 updates on the same eight
  child-to-badge facts and four badge-to-place rules in 24 documents per arm;
  only the prose after each canonical leading evidence sentence differed. The
  arms were a decontaminated official-TinyStories control, Qwen 3.5 35B-A3B,
  and GPT-5.4 Mini. The 72 author calls finished for `$0.0392434000`; the
  promoted artifact is `data/tinyworlds-v2/reasoning-sidebar-v1/`, manifest
  `59200a624dcc8e2afe4cfcdb720d22724184eb97797d7da8208cf0b527d797fe`.
  All three adapters reduced their own training-corpus NLL to at most 0.027,
  and a zero-training exact-clause follow-up scored 100% on all eight literal
  facts and all four literal rules for every adapted arm. Nevertheless,
  held-out two-wording test recall was 25.0% for the TinyStories and Qwen arms
  and 31.2% for GPT, while every arm scored exactly 25.0% on one-hop
  fact-plus-rule questions (four-choice chance is 25%). The clause diagnostic
  is `data/tinyworlds-v2/reasoning-sidebar-v1-clause-probe/`, manifest
  `1d1d8a7921e4ab74b4b23d57266da776d06bf01b3effe5fecc6a92ed5a318b6f`.
  Thus the LoRAs stored literal continuations but did not expose stable
  paraphrase-invariant bindings or compositional knowledge. Because the
  in-distribution control fails the same transfer test, this sidebar cannot
  attribute that failure to Qwen/GPT distribution mismatch. It is exploratory
  evidence only and does not advance Phase 1. Both artifacts strictly reload,
  and the 17-test sidebar/shared-workflow/candidate-scoring suite passes. The
  34-minute complete default suite was not repeated for this non-promotable
  diagnostic.

  The first completed direct artifact at
  `data/tinyworlds-v2/reference-two-route-v1/` is preserved as an over-strict
  validator diagnostic. It spent `$0.1906939625` for all 400 responses but
  incorrectly hard-gated semantic labels such as moral, conflict,
  foreshadowing, and twist using lexical patterns. V2 reinterprets the same
  immutable cached responses without new paid generation: only whole-word
  constraints, safety/length, and quoted dialogue are hard-local evidence;
  semantic plot labels remain reported human-audit judgments.

  The earlier seven-route implementation remains immutable historical evidence.
  It covers exact normalized-content
  source cohorts, deterministic 16-process surface profiles and persisted GPU
  NLL measurements, semantic route identity separated from exact catalog
  provenance, versioned behavior-changing transport headers, public catalog
  revalidation before every paid batch (with a passing public-only live
  resolver smoke on 2026-07-18), append-only completion/stats caching,
  and independent billing observations. Its inclusive runtime cap uses both a
  nonblocking cross-process lease and an fsynced write-ahead reservation/
  settlement journal, persists the complete route lock and sanitized zero-BYOK
  authorization with each reservation, records concurrent pre-POST
  cancellations without charging them, reconciles historical locks after a
  crash, and stops without reposting on ambiguous billing. BYOK also fails
  closed: the inference key cannot prove account state, so production requires
  either a distinct `OPENROUTER_MANAGEMENT_API_KEY` zero-key check or an
  explicit canonical manual attestation valid for at most 24 hours; every
  returned completion/stats record must still prove `is_byok=false`. The
  artifact boundary includes exact generator/verifier cost attribution, route,
  audit, and raw-cache evidence, strict cross-artifact semantic validation, and
  a zero-network byte-for-byte derived replay. Targeted offline unit and
  integration gates, the pinned-source integration, the real-GPU NLL smoke,
  the public-catalog resolver smoke, and focused cost/recovery/replay checks
  pass. The complete offline default suite passes 753 tests with one optional
  skip and eight resource-marked deselections in 2,045.40 seconds; peak RSS was
  4,085,656 KiB. After explicit zero-BYOK confirmation, the production runner
  authenticated and profiled the pinned sources, measured them on the GPU,
  resolved the live routes, and reached its exact preflight. Expected spend was
  `$3.439507`, but the required two-attempt exposure was `$20.020653`, above
  the fixed `$15.000000` cap; the 800-request GPT-5.4 verifier reserve alone was
  `$17.544000`. Per contract, the inference key was not read and zero completion
  POSTs, charges, or generated samples occurred. The promoted stopped artifact
  is `data/tinyworlds-v2/reference/`, with manifest
  `28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9`.
  Strict semantic validation passes and zero-network replay reproduces all 31
  derived files (101,081 bytes). No human audit exists and the Phase 1 gate did
  not pass. A redundant post-artifact default-suite rerun was intentionally
  interrupted at 85% before repeating the known 20--25-minute cache fixture;
  it had no failures, no code changed after the complete pre-run pass, and the
  artifact validation/replay gates passed independently.
  Diagnostic previews treat the external models as synthetic-story authors,
  not as teachers whose behavior is being distilled. The original v1 preview
  at `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v1/` (manifest
  `1ddba6e0862de3e416b4ce21538f5471723e823d6c39c5a32da27a0ea72596b6`)
  is retained as an archived request-contract experiment because its
  `enforce_distillable_text` restriction prevented a meaningful comparison.
  A corrected v2 attempt removed that restriction but was interrupted after 13
  paid requests: its first Qwen response emitted 5,138 hidden reasoning tokens
  despite the 512-token visible-output bound, and exact provider spend reached
  `$0.008248631`. The completed v3 preview explicitly disables optional
  reasoning for Qwen and Gemini and uses the remaining `$0.041751369` of the
  separately authorized `$0.05` cumulative cap. Its promoted artifact is
  `data/tinyworlds-v2/previews/phase1-route-preview-3x7-v3/`, manifest
  `6e1aa9697d8e62263a49c6bc8d22aa22bcb568ca4e551e68b75c727ab063d9f0`.
  All 21 outcomes replay with zero network; 5 passed the current deterministic
  gate (Mistral 1/3, Gemini 3/3, GPT-5.4 Mini 1/3). Provider-reported v3 cost is
  `$0.0061824395`; one unknown-cost timeout is conservatively charged
  `$0.00063045`, for v3 ledger exposure of `$0.0068128895`. This small preview
  remains diagnostic-only and ineligible for route selection. Preliminary inspection
  also shows that the current mechanical gate confounds story quality with the
  model's self-reported evidence schema: Qwen returned three coherent stories
  in a different JSON layout, while two strong GPT stories failed exact
  evidence-quote matching and weaker Gemini prose passed. Before any full
  funnel, separate locally derived story checks from response-metadata validity.
  That correction is now implemented by the active V4 direct comparison. A
  post-run safety audit also closed partial-stats, cache-only recovery,
  no-replace promotion, and cost-evidence validation gaps without making any
  further provider requests. The focused preview/generation suite now passes
  107 tests, and both archived v1 and promoted v3 still validate and replay
  byte-identically under the hardened validator.
- **Phases 2–7: blocked by Phase 1.** Counterbalanced world bibles, natural
  training stories, probes, calibration, the eight-task pilot, and scaling are
  specified in the v2 tracker but are not authorized early. Phase 3 has a
  second mandatory human stop before full-corpus generation.

### TinyWorlds-v1 Gate History

- **Phase 0 — TinyStories post-mortem: complete (2026-07-17).** One exact
  retrain published reusable adaptation artifact
  `0866c521d7accc2576150b5a2cc9b1e4bb9067bcb6403c8da5262f8419b09eef`.
  Reload-only evaluation produced all three paired conditions and the report
  at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-postmortem-seed0-18fcf925db5f`.
  All 1,173 tensor checksums remained identical, the report reproduced
  byte-for-byte from the completed result, and the historical report remained
  byte-identical. The gate suite passed 473 tests with one optional skip and
  five resource-marked deselections.
- **Phase 1 — symbolic TinyWorlds generator: complete (2026-07-17).** The
  calibration and pilot bundles strictly load and rebuild byte-identically at
  digests `ae532f527f9cb35702734aaa819453127f5c30faaf4994436e69a43d2c023c27`
  and `0f24a708301f77b1af8a798869a5c76f8ae7f47205caa7c8cb66b6447d73ca32`.
  The pinned 1,924,281,556-byte original corpus matched SHA-256
  `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`;
  two production re-streams found zero hits among all 144 generated lexical
  forms. All 48 queries have unique graph answers and canonical proofs, bridge
  support requires both branches, all six holdout axes are disjoint, and the
  hardened loader rejects consistently rehashed provenance, topology, proof,
  candidate, revision, story, and capacity tampering. The gate suite passed
  504 tests with one optional skip and five resource-marked deselections; both
  marked tokenizer/novelty integrations passed.
- **Phase 2 — deterministic rendering: complete (2026-07-17).** The pinned
  tokenizer materialized calibration and pilot bundles with exact counts
  5,248 stories/3,072 semantic groups and 10,368 stories/6,144 semantic
  groups. Their rendered digests are
  `ad4a69713060f7d661e92f7415fc4a9ddaffc4852e000cb7c8ba49c3872e4750`
  and `3ad30374480cbeff2587d722b632702de5b097f97f4f9dd651195e3844015bc3`.
  A second invocation strictly reconstructed both trees as `verified`, kept
  the combined tree digest
  `387f4fe2d03b5c868446b9a1884e9134d064b52f4e5e2130796eda1ed53faca0`,
  and reproduced the canonical materialization result byte-for-byte.
  Accepted immutable query-group plans, split-specific templates, exact token
  boundaries and masks, cues, candidate leakage, symbolic alignment, and
  deterministic fallback provenance all passed. The focused gate passed 30
  tests, the marked real-tokenizer gate covered all eight query kinds, and the
  complete default suite passed 516 tests with one optional skip and five
  resource-marked deselections.
- **Phase 3 — candidate scoring and knowledge evaluation: complete
  (2026-07-17).** Frozen, hard-node, and arbitrary-coefficient scorers share
  one four-candidate path with active-token normalization and exact
  microbatch equivalence. The shared hard tensor, both single-run EBT
  refinements and their soft variants, every metric/aggregation axis,
  validation-only parent search, four-role counterfactuals, selected-only
  commit, and immutable resume chunks passed audit. Executed gates cover
  synthetic answer selection, one-hot soft/hard equality, incompatible
  revision answers, incomplete cross-branch hard support, suffix-blind
  routing, and all 11 methods over a real bounded two-edge knowledge stage.
  Final-budget reloads revalidate full execution identity, checkpoint
  histories are exact schedule prefixes, and full-state chunks reject stale
  initial states and dangling symlink targets. The combined focused CPU gate
  passed 77 tests; the complete default suite passed 518 tests with one
  optional skip and five resource-marked deselections.
- **Phase 4 — four-task calibration: complete with controlled scientific stop
  (2026-07-17).** The canonical RTX 4090 runner evaluated exactly three
  validation configurations: the 24-fact baseline, 12 facts, and 36 facts,
  each with 32 exposures/fact, 1,000 updates, rank 8, and hard distractors.
  None passed the complete validation gate, so the fixed ladder stopped
  mechanically at `facts_axis_has_no_passing_configuration`. The immutable
  stopped result is at
  `results/language_cl/tinyworlds-v1/knowledge-graph/calibration-stopped-seed0-e314a9704528`
  with result SHA-256
  `e314a9704528bb8a8133bb4a1465b8be10922df390cbecfb23d33933411ab4e3`.
  All trials retained a 1,024/1,024 exact-KG ceiling, zero committed-node
  drift, and 128/128 old-context consistency. However, frozen novel-binding
  was 64/64 rather than the required 20–30%, leaving zero direct-recall lift;
  revision-node and paired-revision consistency were 0/128 in every trial.
  Independent one-hop accuracy was 0/64 for the 24- and 12-fact trials and
  64/64 for 36 facts, which did not repair the other failed gates. The maximum
  recorded allocator peak was 6,738,299,904 bytes (6.276 GiB), below the
  enforced 12 GiB target. Strict result reload and complete promoted-bundle
  validation pass. The stopped bundle intentionally contains no calibration
  profile and no locked-test artifact. The final complete default suite passes
  537 tests with one optional skip and five resource-marked deselections.
- **Phase 5 — eight-task pilot/report: not launched.** Phase 4 produced no
  passing `calibration_profile.json`; `pilot_authorized` is false and the
  held-out calibration test remained unopened. The TinyWorlds-v1 contract
  therefore forbids launching the pilot. This is the prescribed visible
  scientific outcome, not an implementation failure.
- **Interactive calibration playground: implemented (2026-07-18).**
  `notebooks/tinyworlds_playground.ipynb` provides a read-only view of the
  symbolic world, rendered stories and alignments, proof depth, cue variants,
  hard and continuous support, saved candidate NLLs, gate evidence, and parent
  transfer. Its supporting module strictly reloads the promoted Phase 4 result
  and can generate small worlds through the production symbolic generator,
  tokenizer, and renderer. Fresh samples never inherit saved model scores;
  only the exact canonical seed-0 demo may be joined to validation evidence.
  The notebook does not train, tune, open the held-out test split, or imply
  that Phase 5 is authorized. A clean-kernel execution passed end-to-end, and
  the complete default suite passed 550 tests with one optional skip and five
  resource-marked deselections.
- **TinyWorlds-v1 language-distribution audit: failed (2026-07-18).** The
  renderer is mechanically tokenizer-valid but far outside the frozen
  TinyStories model's training distribution. Entity surfaces are `N` plus 12
  hexadecimal characters and average 8.42 BPE pieces, versus 1.21 for names in
  a matched corpus sample. Generated stories have token-distribution
  Jensen-Shannon divergence 0.702 from TinyStories, while two independent
  TinyStories samples differ by 0.014; an estimated 68.5% of story tokens are
  exact-length padding. Frozen-model scoring on 128 matched 256-token examples
  produced mean NLL 6.780 for TinyWorlds and 1.460 for original TinyStories,
  about a 204-fold perplexity ratio. Raw task IDs, hash marks, symbolic
  predicate labels, and rule-variable labels also enter visible prose, while
  candidate-specific filler contaminates answer NLL. The exact whole-word
  novelty audit did not measure these properties. Preserve v1 as historical
  evidence, but do not interpret its learned-model results as a clean test of
  knowledge-graph acquisition.

### Parked TinyWorlds-v2 Follow-up

1. Preserve the promoted stopped calibration bundle as the terminal
   TinyWorlds-v1 result; do not modify it or launch its Phase 5 pilot.
2. Preserve the valid `blocked_by_cost_cap` Phase 1 artifact and its exact
   `$3.439507` expected / `$20.020653` conservative evidence; do not overwrite
   or reinterpret it.
3. Preserve the completed V2 direct-bakeoff artifact, its 400 raw responses,
   exact `$0.1906939625` bill, failed quality metrics, and balanced audit. Do not
   overwrite the scientific stop or silently relax its thresholds.
4. Inspect `data/tinyworlds-v2/prompt-tuning-v5/review.html`; it shows all 20 V9
   outputs per model beside the exact V8 controls and archive stories, and
   exposes the exact one-sentence request difference.
5. Preserve the route-specific result rather than averaging it away: V9 is
   better for Qwen on length, NLL, token JSD, and composite distance, while GPT
   trades its V8 NLL advantage for correct length and nearly identical
   composite distance. Neither prompt/route cell passes the complete gate.
6. Do not buy another prompt cell until the V5 review is inspected. Any further
   change must state which remaining failure it isolates; do not reintroduce a
   system message, JSON, or a narrative checklist as a bundled intervention.
7. Do not generate world bibles. Phase 2 remains blocked because neither route
   passed the automated Phase 1 gate; human inspection cannot retroactively
   turn the current artifact into a passing one.
8. Add zero-network derived replay for a future passing two-route result before
   treating the Phase 1 gate as passed; the current strict validator already
   authenticates configuration, planned V4 requests, result partitions, local
   evidence, and measurement coverage.
9. Run the complete default suite after any corrective implementation and
   before a future Phase 1 promotion; retain the fast focused suite as the paid
   boundary preflight.

### Deferred Alternative

- A domain-pretrained, TinyStories-scale base model is recorded in
  `docs/TINYWORLDS_DOMAIN_PRETRAINING_NOTE.md`. It would learn KG-oriented
  language from many disposable worlds before continual evaluation on wholly
  held-out worlds. This is a preserved research option, not active work or
  authorization to modify the v1 artifacts.

## Completed Foundation: Language-Model VAMP Proof of Concept

- The completed language-model VAMP foundation and its phase gates are
  recorded in `docs/LM_VAMP_EXECUTION_PLAN.md`.
- Phases 0-10 are implemented: the architecture/build contract is recorded,
  generic immutable graph topology backs the migrated dense-MNIST memory,
  the typed plain-JAX GPT-Neo core passes its CPU correctness and overfit
  gate, and fixed-capacity pathwise LoRA passes its isolation, masking, and
  candidate-gradient gate.
  The verified TinyShakespeare path now includes pinned text preparation,
  deterministic character batches, immutable clipped-AdamW training,
  schema-v1 safetensors checkpoints, and uncached greedy generation.
  Strict TinyStories-8M conversion and the complete offline Hugging Face
  residual/logit/NLL/generation parity ladder pass at the pinned revision.
  Prefix/suffix language contracts, exhaustive normalized-prefix routing,
  and frozen-base content-key derivation pass their masking and task-identity
  exclusion gates.
  The immutable language transition passes a real two-task character-
  permutation run with stable base, prior-run state, and old-node logits.
  Normalized Hopfield retrieval passes real derived-key, capacity masking,
  independent-batch, temperature, top-k, and evaluator-metric gates. EBT
  refinement now optimizes independent per-example node logits, supports all
  four prescribed starts, returns both soft and hard results, and passes its
  decreasing-objective, masking, equivalence, and immutability gates.
  Phase 10 adds deterministic character-permutation, corpus-region,
  stable-hash, and pinned TinyStories topic curricula; document-safe packing;
  the complete four stored/five routed baseline matrix; stored/routing
  forgetting, regret, transfer, logical/runtime memory, synchronized timing,
  random-control confidence intervals, and enforced peak-memory targets; and
  deterministic standalone language reports with all prescribed artifacts.
  TinyStories corpus loading now verifies the pinned files before bounded
  streaming selection, retaining only selected stories plus compact content
  identities. Evaluation and routing support explicit microbatches, reuse
  suffix scores across routers, and cache the shape-stable EBT optimizer
  executable. The real pinned validation aggregate cannot supply the original
  1,000 examples for every topic under the fixed two-concept-plus-margin rule,
  so the measured single-GPU preset uses 10,000/128/128 train/validation/test
  stories and 128 probes/examples while preserving the source, split, topic,
  and deterministic hash-selection contracts.
  Report samples are now completed before the final allocator peak is enforced
  and are reused by the report-only projection. EBT refinement now retains
  aligned node-probability, path-edge-coefficient, and objective trajectories;
  reports preserve a deterministic final-task representative trace as JSONL
  and render four coefficient heatmaps plus an objective curve.
- The target is a shared plain-JAX GPT-Neo base with immutable pathwise LoRA
  memory, a TinyShakespeare smoke path, converted TinyStories-8M weights,
  exhaustive/Hopfield/EBT task-free addressing, and reproducible continual-
  learning reports.

## Current Phase and Immediate Gate

The engineering implementation and resource-backed Phase 10 validation are
complete on the local RTX 4090. Scientific follow-up is now focused on transfer
and routing quality rather than feasibility.

Measured status on 2026-07-16:

- the pinned TinyShakespeare corpus is present and its locally trained
  5,000-step base checkpoint improved validation NLL from 4.20647 to 1.61060;
- both pinned TinyStories V2/GPT-4 aggregates match their prescribed sizes and
  SHA-256 digests, and the TinyStories-8M conversion is present with parameter
  checksum `cdb66d6fe8377d09c43db0631fecb7265216d4383232bff6d0d5f7d0047bf5de`;
- strict prepared-source conversion and the full offline Hugging Face/JAX
  tokenization, residual, logit, NLL, and greedy-generation parity ladder pass;
- the complete TinyShakespeare character-permutation report is at
  `results/language_cl/tinyshakespeare/character-permutation/standard-seed0-a7bd7d1479ba`.
  At the final stage and primary 64-token prefix, exhaustive and both EBT
  routers reach 100% task-node accuracy, Hopfield reaches 86.33%, and the
  deterministic random control reaches 23.44%. The measured allocator peak is
  2,203,208,704 bytes;
- a real 2.2 GB TinyStories streaming preflight filled all four task splits at
  10,000/128/128, retained 13,815 root-validation stories, completed in 4m36s,
  and used no swap; and
- the full stable-hash negative-control report is at
  `results/language_cl/tinyshakespeare/stable-hash/standard-seed0-5f8f82a979a3`.
  Aggregating all four tasks at each prefix, exhaustive, both EBT routers, and
  the deterministic random router have 95% intervals containing 25% chance at
  every prefix. Hopfield contains chance at prefix 32 and is below chance at
  prefixes 64 and 128. The five above-chance task-slice audit flags disappear
  under the prescribed four-task aggregation; two also occur in the
  deterministic random control. The audit found no cross-split macro-document
  overlap and no above-chance aggregate leakage signal; and
- the full TinyStories topic report is at
  `results/language_cl/tinystories-v2-gpt4/topic/single-gpu-seed0-9f715620e7c2`.
  It completed all four 2,000-step tasks and the benchmark/sample workload in
  about 55 minutes. The final allocator peak was 8,349,717,248 bytes (7.776
  GiB), below the enforced 12 GiB target. At the primary 64-token prefix,
  independent root LoRAs were best at 1.33617 NLL; sequential LoRA reached
  1.35269 with 0.02031 forgetting; and VAMP oracle reached 1.36035 with zero
  stored forgetting. Exhaustive and Hopfield routing reached 46.48% and 45.90%
  exact task-node accuracy, while Hopfield was 8.9x faster warm. Both inherited
  children finished about 0.05 NLL behind independent training, and EBT did not
  improve on plain Hopfield. The process loaded the schema-1 report surface
  before commit `95e0cf4`, so its aggregate metrics are complete but the newer
  stepwise EBT coefficient traces require a rerun if they are needed.

Both canonical runners now emit the manifest, seven JSONL metric families,
address confusion, three aggregate metric charts, graph, five EBT routing-
dynamics charts, samples, and standalone HTML under a content-addressed run
directory. Offline bounded tests exercise the same training, all nine methods,
measurement, trace capture, sample generation, and report writer without
substituting for the full-resource measurements. The latest default CPU gate
passes 376 tests with one expected optional-dependency-boundary skip and two
resource-marked tests deselected. Running those two integration tests
explicitly also passes both against the prepared local artifacts.

## Deferred Stage-1 FabricPC Work

Stage-1 dense-delta VAMP over PermutedMNIST and digit-incremental MNIST remains
available with VAE and FabricPC backends. Its reports, fixed-epoch schedule,
and observed-energy-convergence schedule are retained, but further FabricPC
benchmark development is parked until the LM VAMP milestone changes priority.

Deferred gaps are:

- pad and mask FabricPC evaluation, observed-energy, reconstruction, tail, and
  addressed-winner batches to prevent shape-driven JAX recompilation;
- run the full ten-digit convergence benchmark (current verification covers a
  deterministic two-digit VAE run and one-digit FabricPC smoke run);
- inspect stopping epochs, reconstruction quality, and train/test address
  confusion; and
- compare that run with the existing fixed ten-epoch FabricPC checkpoint,
  especially early-digit routing to later parameter nodes.

Observed energy remains model-specific and supports within-model stopping and
addressing only. Reaching the epoch limit retains the best state and reports
`max_epochs`; it is not convergence.

## Known LM VAMP Gaps

No required engineering surface or resource-backed gate from the execution
plan remains incomplete. Research priorities are to replace prefix-NLL-only
parent selection or add a root fallback for negative transfer; measure whether
whole-story topic cues are visible in each evaluated prefix; correct the
fantasy-biased content keys; and explain why EBT sharpens its addressing while
losing suffix quality, especially at prefix 128. Add incremental progress
events before another long resource run. Rerun the full TinyStories report only
if the post-`95e0cf4` stepwise EBT coefficient artifacts are required. The
validated public routing wrapper performs host-side postcondition checks and is
intentionally timed directly; extracting a separate outer-JIT-compatible
validated factory remains optional optimization.
