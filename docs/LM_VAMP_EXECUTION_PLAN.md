# Language-Model VAMP Proof-of-Concept Execution Plan

## Purpose and working contract

This roadmap implements the proof of concept specified in
`docs/TinyStories_PoC_Plan.pdf`: a shared plain-JAX GPT-Neo base, immutable
pathwise LoRA memory, a TinyShakespeare smoke model, converted
TinyStories-8M weights, three task-free routers, and reproducible
continual-learning reports.

Work through the phases in order. Before each phase, inspect `AGENTS.md`,
`STYLE.md`, `DESIGN.md`, `PLAN.md`, the current graph implementation, and its
tests. Update `PLAN.md` whenever implementation status or priorities change,
and record durable architectural decisions in `DESIGN.md`. Parameters,
optimizer state, graph state, address state, and random keys remain explicit
immutable values. A phase is complete only after its gate passes; do not begin
the next phase early.

The ordering here intentionally constructs exhaustive prefix scoring and
content keys before the continual runner that consumes them.

## Phase 0: Freeze the architecture and build boundary

- Record this execution plan, the durable LM/VAMP design contract, and LM VAMP
  as the active milestone.
- Consolidate packaging in `pyproject.toml`; remove duplicate legacy
  setuptools metadata.
- Add optional `lm` dependencies (`huggingface_hub`, `tokenizers`, and
  `safetensors`) and the conversion-only dependencies (`torch` and
  `transformers==4.28.1`) under `hf-convert`.
- Register `integration` and `benchmark` pytest markers and exclude both from
  the default suite. Ignore local `/checkpoints/` artifacts.

Gate: the pre-existing default test suite passes unchanged.

## Phase 1: Extract model-independent graph topology

- Introduce immutable `NodeId`, `TaskId`, `MemoryNode[PayloadT]`, and
  `MemoryGraph[PayloadT]` values in `apm.memory.graph`.
- Provide pure operations for initialization, child insertion, lookup, child
  traversal, root-to-node paths, insertion-ordered node/edge identifiers, and
  path-incidence construction.
- Define incidence rows in node insertion order and columns in non-root
  insertion order; the root row is all zeroes.
- Replace `DenseMemoryGraph` with a typed `DenseParameterMemory[ParamsT]`
  containing root parameters plus `MemoryGraph[ParamsT]`. Keep dense
  subtraction, reconstruction, and accounting in `dense.py`; eliminate
  `ParamTree = Any`.
- Migrate every MNIST addressing, visualization, export, runner, and test use
  in one breaking refactor. Do not keep compatibility aliases.

Gate: linear and branching graph behavior, invalid insertions, incidence,
dense reconstruction/statistics, the complete existing suite, and the
synthetic Stage-1 smoke run all pass.

## Phase 2: Implement pure-JAX GPT-Neo

- Add frozen configuration values and typed `NamedTuple` PyTrees for linear,
  layer-normalization, attention, MLP, block, and complete model parameters.
- Implement global and local causal attention, pre-layer normalization,
  `gelu_new`, tied token/output embeddings, bias-free Q/K/V, biased attention
  output and MLP projections, and float32 attention scores.
- Forward APIs accept explicit attention and position masks, static capture
  instructions, `training`, and an explicit PRNG key whenever configured
  dropout is nonzero. Phase 3 extends these same functions with packed LoRA
  coefficients after the base-only parity gate is established.
- `CaptureSpec` is an ordered tuple of `(layer_index,
  post_attention|post_mlp)` points. `ForwardResult.captured_hidden` uses that
  fixed order.
- Support right-padded batches; padding IDs are inert through attention and
  loss masks.

Gate: shape, mask-boundary, causality, embedding tying, bias, capture,
eager/JIT agreement, finite-gradient, and tiny fixed-batch overfit tests pass.
The fixed-batch NLL falls below `0.05`.

## Phase 3: Add pathwise LoRA and fixed-capacity packing

- Define `LoraConfig`, `LoraProjection`, `LoraBlock`, `LoraEdge`, per-site
  `LoraEdgeBank`, and `PackedLoraMemory`.
- Support Q, K, V, attention-output, MLP-input, and MLP-output projections.
  Presets enable all six; a static target mask may disable sites without
  changing PyTree shape.
- Use one rank per run and scale `alpha / rank`, initially with `alpha=rank`.
  Initialize left factors with fan-in-scaled random values and right factors
  to exact zero.
- Stack corresponding factors with a shared leading edge axis. Never sum A or
  B coordinates across edges; sum each completed edge's low-rank output
  contribution.
- Pack graph state to curriculum capacities with `node_path_matrix`,
  `valid_node_mask`, and `valid_edge_mask`. Packed memory is derived, never a
  second authoritative state.
- Train a candidate by placing it in the next padded slot inside the loss.
  Differentiate only the candidate and apply `stop_gradient` to the frozen
  base and committed edge bank.

Gate: exact zero-effect equality, direct one-edge calculation, two-edge
addition, sibling isolation, one-hot and continuous coefficients,
candidate-only gradients, padded-slot neutrality, exact base hashes, and
bitwise-stable logits for old nodes all pass.

## Phase 4: Build the TinyShakespeare text and training path

- Add a text-tokenizer protocol, immutable `TokenBatch`, deterministic dataset
  references/loaders, LM train state, checkpoint I/O, and uncached greedy
  generation.
- Pin TinyShakespeare to `karpathy/char-rnn` commit
  `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`. Download only through an
  explicit preparation command and verify SHA-256.
- Build the character vocabulary from training text only in deterministic
  sorted order, reserving PAD and EOS IDs. Split the corpus contiguously
  90/5/5 and align boundaries before producing windows.
- Standard preset: context 256, width 128, four layers, four heads, MLP width
  512, alternating global/local attention, local window 64, zero dropout,
  batch 32, and 5,000 AdamW steps at `3e-4` with gradient clipping at 1.0.
- LoRA preset: rank/alpha 4, batch 32, and 1,000 steps per edge at `1e-3`.
- Save schema-v1 safetensors checkpoints with canonical flattened parameter
  names, config, tokenizer/source hashes, and a manifest. Optimizer-resume
  checkpoints are outside the PoC.
- Keep scripts and notebooks thin: they call tested library functions rather
  than reimplementing training behavior.

Gate: validation NLL decreases, checkpoint round-trip is exact, greedy token
IDs repeat, a single-task edge improves task NLL, and the base checksum does
not change.

## Phase 5: Convert and validate TinyStories-8M

- Pin `roneneldan/TinyStories-8M` revision
  `8612e3b15c66ffa94eaa6ee0de5c96edd2d630af`. Its published configuration is
  the eight-layer, width-256 GPT-Neo parity target.
- Convert PyTorch linear weights by transposing them into JAX kernels,
  normalize a null intermediate size to `4 * hidden_size`, verify tied LM-head
  and token embeddings, and reject every missing or unexpected source key.
- Emit safetensors, a schema-v1 manifest, and pinned tokenizer files. Record
  source revisions and hashes, converter version, canonical names, shapes,
  dtypes, and environment.
- Confine Torch and Transformers imports to the converter and marked parity
  tests.
- Validate tokenization, embeddings, positions, global/local masks, both
  residual points in every block, final hidden state, logits, NLL, and greedy
  token IDs. Begin with `rtol=atol=2e-4`; report per-layer maximum and mean
  errors instead of loosening the global tolerance.

Gate: strict conversion and the complete parity ladder pass from locally
prepared artifacts with network access disabled.

## Phase 6: Introduce language contracts and routing foundations

- Add immutable `LanguageTask`, `LanguageCurriculum`, `AddressBook`,
  `AddressResult`, `BaseCheckpointRef`, and evaluation-example records.
- Make evaluation spans disjoint:
  - Router batches contain only prefix inputs and targets.
  - Competence batches contain prefix plus suffix, with loss enabled only on
    suffix targets.
  - Task/oracle identity remains evaluator metadata and is absent from router
    signatures.
- Implement normalized exhaustive prefix NLL and convert node scores to
  probabilities with temperature-one `softmax(-NLL)`.
- Implement the shared frozen-base content encoder: masked-mean final hidden
  states followed by L2 normalization. Build node keys from 256 deterministic
  probes using a 64-token/character prefix.
- Give the root a key derived from the base validation probe, so it remains a
  valid Hopfield candidate.

Gate: known-best-node selection, valid-token normalization, padding
neutrality, suffix non-leakage, fixed-capacity masking, deterministic keys,
and structural prohibition of task identity in router APIs all pass.

## Phase 7: Implement the immutable continual-learning transition

- Add `LanguageVampRun` and a pure transition:
  `task -> exhaustive parent probe -> candidate-edge training -> immutable
  commit -> content-key derivation -> stored and task-free evaluation`.
- Keep the frozen base outside the graph. Store its checkpoint reference,
  graph, address book, RNG key, completed tasks, and immutable metrics in the
  run value. Packed arrays remain deterministic derived data.
- Separate orchestration from artifact writing. Do not extend the
  canvas-oriented MNIST `ModelBackend`/`TaskDataset` abstractions or its
  mutable/global runner for language workloads.

Gate: a two-task TinyShakespeare character-permutation run completes mixed
task-free exhaustive routing while all previously committed logits remain
unchanged.

## Phase 8: Add Hopfield routing

- Retrieve with normalized dot products and masked
  `softmax(beta * similarity)`, initially `beta=10`.
- Support independent batched queries and `top_k=min(4, valid_nodes)`.
- Report selected node, probabilities, entropy, top-k recall, margin, and
  agreement with exhaustive routing.

Gate: exact-key self-retrieval, masking, temperature/top-k behavior,
independent batch routing, and frozen-base/no-adapter content embeddings all
pass.

## Phase 9: Add EBT address refinement

- Optimize per-example node logits only and map node probabilities to edge
  coefficients through the path-incidence matrix.
- Default to 20 Adam steps at learning rate `0.1`, fixed `tau=1`, and entropy
  penalty `0.01`. Support uniform, Hopfield, full-node, and Hopfield-top-k
  initialization.
- Report soft-mixture NLL and the primary hard `argmax` node result.

Gate: one-hot equivalence to discrete execution, finite and decreasing
constructed objectives, invalid-node zero probability, per-example
independence, accepted Hopfield initialization, and unchanged base/edge hashes
all pass.

## Phase 10: Add curricula, baselines, metrics, and reports

- Implement four-task TinyShakespeare character-permutation, corpus-region,
  and stable-hash curricula. Character permutations use seeds 0-3 over the 26
  letters, preserve case, and leave nonletters untouched.
- Use the pinned TinyStories V2/GPT-4 files at dataset revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; describe results as in-domain
  continual adaptation.
- Normalize stories with Unicode NFC and collapsed whitespace, hash with
  SHA-256, remove train/evaluation duplicates, and split official V2
  validation deterministically 50/50 into validation and test.
- Build topic curricula from case-folded whole-word matches:
  - Animals: dog, cat, bird, rabbit, bear, lion, horse, cow, sheep, duck, fish,
    frog, mouse/mice, elephant, monkey, pet.
  - Vehicles/tools: car, truck, train, bus, bicycle/bike, boat, plane, tractor,
    hammer, saw, wrench, screwdriver, wheel, engine, garage.
  - Family/home: mother/mom, father/dad, sister, brother, family, friend, home,
    house, grandma, grandpa, parent, neighbor.
  - Fantasy/royalty: king, queen, prince, princess, castle, dragon, fairy,
    wizard, magic, knight, crown, kingdom, unicorn, witch.
- Include listed plurals, require at least two distinct hits and a one-hit
  margin over the runner-up, reject ties/overlaps, and equalize buckets by
  lowest content hashes.
- Single-GPU TinyStories preset: four tasks; 10,000/1,000/1,000
  train/validation/test stories per task; context 256; rank/alpha 8; batch 32;
  2,000 adapter steps per task; 256 parent/key probes; 256 evaluation examples
  per task and prefix length; five-node/four-edge capacity; 12 GiB peak-device
  memory target.
- Evaluate TinyShakespeare prefixes of 32/64/128 characters with suffix 128,
  and TinyStories prefixes of 16/32/64/128 tokens with suffix 128.
- Run frozen-base, sequential-single-LoRA, independent-root-LoRA, VAMP oracle,
  exhaustive, Hopfield, EBT-uniform, EBT-Hopfield, and deterministic-random-node
  baselines with identical adapter budgets.
- Report stored versus routing forgetting, routing regret, transfer,
  persistent versus runtime memory, and synchronized cold/warm addressing
  cost.
- Write reports beneath
  `results/language_cl/<dataset>/<curriculum>/<preset>-seed0-<config_hash>/`,
  reusing low-level JSON/JSONL/SVG/lightbox utilities rather than the
  reconstruction-oriented MNIST report surface.

Gate: one complete TinyShakespeare character-permutation report and one
bounded TinyStories V2 topic report contain every baseline, router, metric
family, graph visualization, and generated samples.

## Test and completion policy

- Default tests are CPU-only, deterministic, fast, network-free, and
  checkpoint-free.
- Integration tests cover standard TinyShakespeare training and local
  TinyStories parity. Benchmark tests cover full reports, GPU memory,
  throughput, and cold/warm timing with `block_until_ready()`.
- Required end-to-end coverage includes zero-effect and path-sum LoRA,
  independent siblings, candidate-only gradients, stable base and committed
  logits, exact path incidence, padding neutrality, all router invariants,
  disjoint prefix/suffix evaluation, two-task TinyShakespeare continual
  learning, exact checkpoint round-tripping, and TinyStories logits/generation
  parity.
- Engineering completion requires structural, parity, immutability,
  split-integrity, and end-to-end gates. Weak transfer or routing is a research
  result, not an engineering failure.
- For the four-task negative control, the 95% confidence interval should
  contain the 25% chance rate. A materially higher result triggers a leakage
  audit before scientific claims are made.

## Explicit non-goals

Do not add Equinox or NNX model representations, a KV cache before parity,
variable rank within one graph, adapter compression, embedding or layer-norm
LoRA, a graph-rewrite IR, task IDs in routing, base weights inside graph nodes,
image/text union backends, ordinary-test downloads, compatibility aliases,
mixed-precision parity, or full TinyStories base pretraining.
