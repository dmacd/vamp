# VAMP Technical Design

## Scope and Terminology

Virtually Addressed Memory for Parameters (VAMP) is the name of the
architectural continual-learning paradigm developed in this repository. Its
canonical reference is the lightly revised and renamed
[Virtually Addressed Memory for Parameters](docs/Virtually_Addressed_Memory_for_Parameters.pdf)
manuscript, which replaces `docs/Addressed_Parameter_Memories.pdf`.

The language-model proof of concept applies VAMP to GPT-Neo language models
with immutable pathwise LoRA edges. Its implementation contract is
`docs/LM_VAMP_EXECUTION_PLAN.md`. Dense MNIST and language VAMP share graph
topology, but they do not share model, task, training, or evaluation
interfaces.

## Model and State Representation

The canonical language model is plain JAX. Frozen configuration dataclasses,
typed `NamedTuple` parameter PyTrees, tuples of layers, and pure initialization,
application, loss, and training functions define the model. Optax state and
random keys are explicit inputs and outputs. Nonzero dropout requires an
explicit key. Neither Equinox nor NNX model objects, stateful trainers, or
mutable sessions belong in the graph, checkpoint schema, or public language
VAMP interfaces.

The initial GPT-Neo implementation uses pre-layer normalization, tied token
and output embeddings, global and local causal attention, `gelu_new`,
bias-free Q/K/V projections, and biased attention-output and MLP projections.
Attention scores are computed in float32. Right-padding is supported through
explicit attention, position, and loss masks.

For Hugging Face GPT-Neo 4.28.1 parity, query/key dot products are not scaled
by the square root of the head size. A local window of width `w` includes the
current key and the preceding `w - 1` keys, using the strict lower bound
`key_position > query_position - w`. Padding masks keys, while the separate
loss mask determines which query positions contribute to reported NLL.
Intermediate capture is an ordered immutable tuple of explicitly requested
post-attention or post-MLP residual points; it is not a mutable hook registry.

## Generic Graph and Frozen Base Boundary

The reusable memory graph owns topology only: immutable nodes, parent/child
relationships, insertion order, depth, root-to-node paths, and path-incidence
matrices. Incoming edge payloads are generic. Payload interpretation and byte
accounting remain in model-specific layers.

Dense MNIST memory stores its root parameters beside a generic graph whose
edge payloads are dense parameter deltas. Language VAMP keeps the frozen base
model entirely outside the graph and stores one independent LoRA payload on
each non-root edge. A language run refers to the base by an immutable
checkpoint reference; it never copies base weights into nodes.

Persistent memory is reported as one base model plus committed LoRA edges,
address keys, and graph metadata. Optimizer state, compilation artifacts,
runtime caches, and candidate or packed working arrays are reported
separately.

## Pathwise LoRA Semantics

For edge `e`, projection input `x`, frozen kernel `W`, optional bias `b`,
factors `A_e` and `B_e`, scale `s_e`, and the root-to-node edge path `P(n)`,
the effective projection is:

```text
y = xW + b + sum(s_e * (x A_e) B_e for e in P(n))
```

Each edge is a completed independent low-rank residual. A and B coordinates
are never summed separately, and child factors are not coordinate differences
from their parent. Adding a node therefore cannot modify any previously
committed path. The initial implementation supports all six transformer
linear projections, one rank per run, and fixed scaling `alpha / rank`; it
does not adapt embeddings or layer normalization.

## Fixed-Capacity Device Representation

The immutable host graph compiles into fixed-capacity arrays whose capacities
come from the curriculum. `PackedLoraMemory` contains the stacked edge bank, a
`[max_nodes, max_edges]` path-incidence matrix, and valid-node and valid-edge
masks. Unused entries are zero-padded and masked. Packed arrays are derived
from authoritative graph state rather than persisted as a second state.

Candidate training inserts the new edge into the next padded slot within the
loss. Gradients flow only through the candidate; the base and committed bank
are stopped. Fixed shapes prevent graph growth, tail batches, or changing
candidate sets from causing avoidable JAX recompilation.

Every supported projection retains the same LoRA PyTree fields even when a
static target mask disables it. Disabled sites initialize both factors to
zero and use the exact base projection branch. At model entry, valid-edge
masks are applied once to shared or per-example edge coefficients. Supplying
LoRA state is all-or-nothing: the packed memory, coefficients, and static
LoRA configuration must be present together. With no LoRA state, the model
executes the Phase-2 base projection path unchanged.

## Language Addressing and Evaluation

Every evaluation example is split into an address prefix and a competence
suffix. Routers see only prefix inputs and targets. Competence loss sees the
prefix as context but enables loss only on suffix targets. Task or oracle
identity is evaluator metadata and must not appear in a router signature.

All task-free routers use the same insertion-ordered nodes and path-incidence
matrix:

- Exhaustive addressing scores every valid node by normalized prefix NLL and
  converts `-NLL` to probabilities with a temperature-one softmax.
- Hopfield addressing compares an L2-normalized query to L2-normalized node
  content keys, masks invalid nodes, and applies `softmax(beta * similarity)`.
  Queries and keys come from masked-mean final hidden states of the frozen base
  with no adapters active.
- EBT refinement optimizes independent per-example node logits against prefix
  NLL, then maps node probabilities to continuous edge coefficients through
  the path-incidence matrix. The reported primary choice is the hard argmax
  node; soft-mixture NLL is diagnostic.

The prefix/suffix boundary is represented structurally, not by convention.
`RouterBatch` has only prefix transitions and activates every valid prefix
loss. `CompetenceBatch` repeats those transitions as context, activates no
loss over them, and enables one contiguous suffix-loss span. Task IDs and
oracle node IDs live only in the surrounding evaluation record. Exhaustive
routing returns fixed-capacity per-example NLL scores, masked probabilities,
hard indices, margins, and entropy.

Content addresses never execute adapters. They are L2-normalized masked means
of frozen-base final hidden states; node keys are normalized centroids of a
fixed, preset-defined deterministic probe set. The canonical TinyStories
preset uses 128 probes and TinyShakespeare uses 64. The root uses the same
derivation path on a base validation probe and occupies address-book slot zero.

Hopfield routing operates only on those normalized values. Invalid address
rows receive `-inf` similarity and exactly zero probability; valid rows use
`softmax(beta * dot(query, key))`. The router returns independent batched
choices, probabilities, entropy, similarity margin, and the effective top-k.
Oracle accuracy, top-k recall, and agreement with exhaustive routing are
computed only in evaluator code.

The root remains addressable and receives a content key derived from a frozen
base validation probe. Hopfield output may initialize EBT, including a masked
top-k variant. Routing never uses competence suffix tokens.

Evaluation microbatching is an execution boundary, not a change to routing or
competence semantics. General library paths retain their vectorized default;
the canonical TinyStories runner records and uses a row microbatch size of
eight independently of training batches. It preserves input order and exact
per-example values, scores valid nodes sequentially within each chunk, and
applies the same bound to parent probes, frozen content keys, stored
competence, all task-free routers, report samples, and final timing. Suffix
competence is computed once for each stage/task/prefix sweep and reused across
the five router evaluations because it is router-independent. Timing still
covers the complete logical evaluation batch and reports operation counts for
that complete batch, while the microbatch setting participates in report
identity.

EBT owns only a `[batch, max_nodes]` float32 logit value. A masked,
temperature-one softmax produces node probabilities, and multiplication by the
packed path-incidence matrix produces continuous edge coefficients. Invalid
nodes have `-inf` final logits, exactly zero probability, and zero logit
gradients. Adam minimizes each example's normalized prefix NLL plus a positive
entropy penalty; the batch objective is their sum, so batched refinement is
identical to refining each row independently. Uniform and Hopfield starts
refine every valid node, full-node starts strongly favor an explicitly supplied
node while retaining the full candidate set, and Hopfield-top-k permanently
masks candidates outside the retrieved top-k. Base parameters and the packed
edge bank are stopped values. The result records the full per-example
`steps + 1` objective trace, soft-mixture prefix NLL, and the primary hard
argmax node NLL.

## Continual Transition and Reporting

A language stage is a pure transition from an immutable run plus one task to a
new run: exhaustively select a parent, train one zero-effect candidate edge,
commit it, derive its content key, and evaluate stored and task-free behavior.
The run stores its base checkpoint reference, graph, address book, random key,
completed tasks, and metrics. Artifact writing is outside this transition.

The authoritative run never stores packed memory. Each transition derives it,
averages exhaustive validation-prefix scores to select a parent, trains only
the new candidate edge, commits one child, derives its frozen-base content
key, and evaluates every completed task. The explicit RNG stream advances in
the returned value; the prior run, base, and committed edge values remain
unchanged.

Stored forgetting and routing forgetting are distinct measurements. Reports
also separate persistent memory from runtime memory and cold compilation from
warm execution; synchronized JAX timings call `block_until_ready()`.

The canonical adaptation matrix contains four stored-competence methods. The
frozen base has no adapter; sequential-single-LoRA reuses and updates one root
adapter across tasks; independent-root-LoRA trains one fresh root adapter per
task; and VAMP-oracle executes the committed node associated with evaluator
task identity. The canonical task-free routing matrix contains exhaustive,
Hopfield, EBT-uniform, EBT-Hopfield, and deterministic-random-node methods.
All five receive the same `RouterBatch`; only evaluator code combines their
decisions with suffix competence or task-oracle metadata. The deterministic
random control hashes the seed and valid prefix tokens, so padding contents and
evaluation order cannot change its choice. It samples non-root task nodes when
any exist and uses the root only for a root-only graph. Thus its final-stage
chance accuracy is 25% in a four-task curriculum even though the learned
routers remain free to address the root.

Stored and routing measurements are indexed by stage, introduced task, and
prefix length. Stored forgetting is the increase from a stored method's best
earlier suffix NLL. Routing forgetting applies the same construction to routed
suffix NLL and is never substituted for adapter drift. Routing regret is
reported against both the task-oracle node and the best suffix-competence node.
Transfer records the selected parent's initial advantage over the root, the
first update improvement, the fixed-budget improvement, and the final deficit
relative to an independently trained root adapter. Base and committed-path
checksums accompany stored rows. The deterministic-random router always, and
every router on a declared negative-control curriculum, includes a two-sided
95% Wilson interval, the stage-wise `1 / task_count` chance rate, and a leakage
audit flag when the observed rate is above chance and the interval excludes
chance. A router's root choice counts as incorrect rather than being discarded.

Logical persistent memory consists of the base, committed edges, valid content
keys, and serialized graph metadata. Packed edge banks, path matrices, validity
masks, padding, and optimizer state are runtime or training storage and are
reported separately. Address timing measures the first synchronized call and
then repeated synchronized calls at a fixed shape; reports include both wall
time and router-specific logical operation counts rather than treating a
Hopfield dot product and a full model candidate evaluation as equivalent.
When a peak-device-memory target is configured, the runner requires allocator
peak statistics from the active JAX backend and fails the run if the observed
peak exceeds the target. The completed benchmark owns its generated samples:
all sample routing and generation finishes before the final allocator read, so
the enforced and reported peak covers that work as well as training,
evaluation, and timing. Successful reports record the observed peak, backend
limit, target, platform, and device kind; CPU smoke runs without a target may
report those allocator fields as unavailable.

Language reports are deterministic projections of a completed benchmark,
including its already-generated samples; report building performs no sample
routing or generation. Reports are not authoritative training state. Their
identity is the canonical configuration JSON hash beneath
`results/language_cl/<dataset>/<curriculum>/<preset>-seed0-<hash>/`. Rewriting
the same bundle replaces JSONL files rather than appending and produces the
same manifest, charts, graph, samples, and HTML bytes. Every report must cover
all nine canonical methods and remain language-specific; it does not reuse the
MNIST reconstruction presentation. The configuration hash covers model,
checkpoint, tokenizer, curriculum capacity, data packing and evaluation,
adapter targets, optimizer budget, router/EBT settings, seeds, timing repeats,
and any device-memory target, as well as dataset-specific selection rules.

## Text Curriculum Semantics

TinyShakespeare is split contiguously before any task transform. The
character-permutation curriculum applies seeds 0 through 3 to the 26 ASCII
letters, preserves case, and leaves every nonletter unchanged. Corpus-region
tasks divide each raw source split into four exact contiguous spans. The
TinyShakespeare stable-hash control first divides each split into fixed
1,024-character raw macro-documents, preserving whitespace and punctuation,
then assigns each document by its raw UTF-8 SHA-256. The common
TinyShakespeare evaluation budget is 64 examples for each task and prefix;
this fits the smallest disjoint validation region. Training and evaluation
windows are constructed within one document and one split; no packed sequence
may cross a document, task, or train/validation/test boundary.

The supported TinyStories curriculum consumes only the pinned V2/GPT-4 train
and validation aggregates. Stories are Unicode-NFC normalized, internal
whitespace is collapsed, and normalized SHA-256 is their identity. Duplicate
content is removed globally, training content is excluded from evaluation,
and the official validation aggregate is divided into equal validation and
test halves by content-hash order. Topic matching is case-folded and
whole-word. Aliases and plurals map to one semantic concept; eligibility
requires at least two distinct concepts for one topic and a margin of at least
one concept over every other topic. Ties and overlaps are rejected, and each
topic/split bucket takes the lowest content hashes to obtain exact equal
counts. These choices make the bounded topic benchmark in-domain continual
adaptation rather than an out-of-domain generalization claim.

Pinned-file verification precedes TinyStories decoding. The official
validation aggregate is small enough to retain as normalized documents, while
the train aggregate is parsed through separator-safe text chunks and is never
materialized as one Python string. During that pass the loader excludes
official-validation identities, retains a compact set of seen SHA-256
identities so the first normalized occurrence wins globally, and keeps only
each topic's requested lowest-content-hash story payloads. For the pinned
source, content hashes are authoritative identities; unlike the small
in-memory helper, the streaming path does not retain every story merely to
diagnose a hypothetical SHA-256 collision. This boundary reproduces selection
semantics without retaining the 2.2 GB raw train aggregate or every normalized
train story.

Tokenization happens after document selection. TinyStories uses the tokenizer
artifact bound to the converted checkpoint, while TinyShakespeare constructs
its character vocabulary from training text only. Evaluation span selection
is deterministic and shared across prefix-length sweeps: a selected maximum
span is shortened at the prefix/suffix boundary, so routers at different
prefix lengths are compared on the same underlying document material. The
lowest required span identities are selected from a lazy candidate stream with
bounded storage; original candidate ordinals preserve the former stable-sort
behavior when identities tie. The configured suffix length is a fixed capacity:
an eligible sequence must contain the maximum prefix plus at least one suffix
target, and any unused suffix capacity is right padded and masked out of NLL.

## Checkpoints and the PyTorch Boundary

Language checkpoints use schema-versioned safetensors plus a manifest with
canonical flattened names, configuration, shapes, dtypes, source and
tokenizer hashes, and provenance. Optimizer-resume state is not part of the
initial schema.

PyTorch and Transformers are conversion-only dependencies. They may be
imported by the TinyStories converter and marked parity tests, but not by JAX
training, generation, routing, or benchmark modules. Conversion rejects
missing and unexpected keys, verifies tied embeddings, and produces local
artifacts from pinned source revisions. Ordinary tests never download data or
checkpoints.

The supported TinyStories source is one immutable aggregate contract: model
revision, Transformers 4.28.1 semantics, config/model byte hashes and sizes,
and all five tokenizer artifacts. Conversion consumes the full 125-entry
state dictionary, validates global/local causal buffers, transposes each
PyTorch linear kernel into input-major form, rejects all missing/unexpected
entries, and verifies the tied LM head before discarding its duplicate. The
converted outer artifact atomically binds the checkpoint, tokenizer files,
converter/library/environment provenance, and source contract.

Language checkpoints use a dependency-free, standards-compatible
safetensors container and an atomically published schema-v1 manifest.
Canonical parameter names are input-major and include no duplicate LM head.
Loading is strict over tensor names, shapes, float32 dtypes, configuration,
content hashes, tokenizer/source identity, and manifest metadata. Train-state
and optimizer-resume serialization are intentionally outside this schema.

TinyShakespeare character vocabularies reserve PAD 0 and EOS 1, then assign
training-only characters in sorted order. The raw corpus is split contiguously
before encoding or causal window construction, so no window crosses a
train/validation/test boundary. Tail examples and batches are right padded to
their fixed context and batch capacities; attention and loss masks make those
slots inert.

## Task Training Schedules

Stage 1 accepts either a fixed-epoch schedule or an observed-energy-convergence
schedule. The same schedule contract is used by the VAE and FabricPC backends.

Energy convergence is measured after each epoch on a deterministic subset of
the current task's training arrays. The subset and inference random key remain
fixed across epochs. Test examples and labels do not participate in stopping.
The monitored value is the same digit-only energy used by memory addressing:
digit-region BCE plus beta-weighted KL for the VAE, and inferred graph energy
with only the digit node clamped for FabricPC.

Patience resets after cumulative improvement from the reference energy reaches
the configured relative threshold. Absolute best energy is tracked separately,
so sub-threshold improvements can still supply the selected checkpoint. On
convergence or the maximum epoch limit, the best parameters and corresponding
optimizer state are restored while retaining the final random key to prevent
random-number reuse in later tasks.

Non-finite monitored energy aborts the run. Reaching the maximum epoch limit
continues the benchmark with an explicit `max_epochs` status and must not be
reported as convergence.
