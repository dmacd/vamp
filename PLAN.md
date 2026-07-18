# Development Plan

## Active Milestone: TinyWorlds Knowledge-Graph Benchmark

The active roadmap is the six-phase TinyWorlds benchmark. Phases are promoted
only after their production artifact gate and the complete default test suite
pass. Implementations for later phases may be present in the worktree, but
they are not considered complete until the preceding gates have passed.

### Gate Status

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

### Immediate Work

1. Preserve the promoted stopped calibration bundle as the terminal
   TinyWorlds-v1 result; do not launch or tune Phase 5 from it.
2. Analyze the visible calibration failure—especially frozen direct accuracy
   at 100%, zero direct lift, and zero revision recovery—without altering the
   completed evidence.
3. Introduce any changed gate, query construction, candidate policy, or
   calibration hypothesis as a new versioned benchmark contract and rerun
   calibration from scratch before a pilot can be authorized.

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
