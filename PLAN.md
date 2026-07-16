# Development Plan

## Active Milestone: Language-Model VAMP Proof of Concept

- Language-model VAMP is the active development priority. The decision-complete
  roadmap and phase gates are recorded in `docs/LM_VAMP_EXECUTION_PLAN.md`.
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
- The target is a shared plain-JAX GPT-Neo base with immutable pathwise LoRA
  memory, a TinyShakespeare smoke path, converted TinyStories-8M weights,
  exhaustive/Hopfield/EBT task-free addressing, and reproducible continual-
  learning reports.

## Current Phase and Immediate Gate

The engineering implementation is complete. The active work is resource-backed
Phase 10 validation using the pinned local datasets and checkpoints.

The remaining operational gate is:

- materialize or verify the pinned TinyShakespeare corpus, TinyShakespeare base
  checkpoint, TinyStories V2/GPT-4 aggregates, and converted TinyStories-8M
  artifact;
- run `scripts/run_tinyshakespeare_cl.py` and inspect the complete four-task
  character-permutation report;
- run `scripts/run_tinystories_cl.py` on one GPU and inspect the bounded topic
  report, including the enforced 12 GiB allocator peak; and
- run the stable-hash negative control and confirm that its four-task routing
  interval contains 25% chance, or perform the recorded leakage audit if it
  does not.

Both canonical runners now emit the manifest, six JSONL metric families,
address confusion, three metric charts, graph, samples, and standalone HTML
under a content-addressed run directory. Offline bounded tests exercise the
same training, all nine methods, measurement, sample generation, and report
writer without claiming the absent full-resource measurements. The latest
default CPU gate passes 357 tests with one optional-tokenizer skip and two
resource-marked tests deselected.

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

No known engineering surface from the execution plan remains unimplemented.
The workspace does not contain the required language datasets or checkpoints,
so the two canonical real-data reports and the single-GPU 12 GiB measurement
have not been executed here. The validated public routing wrapper performs
host-side postcondition checks and is intentionally timed directly; extracting
a separate outer-JIT-compatible validated factory is optional future
optimization, not a Phase 10 requirement.
