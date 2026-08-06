# VAMP Technical Design

## TinyWorlds Noun-Overlap v1

`tinyworlds-nouns-v1` is an isolated, descriptive continual-learning
experiment over the pinned TinyStories V2/GPT-4 archive. It does not reuse or
modify TinyWorlds-Q, semantic-v6, or their checkpoints. Its artifacts live only
under `data/tinyworlds-nouns-v1/`, `checkpoints/tinyworlds-nouns-v1/`, and
`results/language_cl/tinyworlds-nouns-v1/`.

### Noun authority and manual boundary

The noun manifest preserves the complete ordered `TINYSTORIES_TOPICS` catalog.
Each family owns explicit case-folded whole-word forms; there is no parser,
embedding classifier, WordNet expansion, or model-authored label. An editable
decision document records inclusion, exact forms, category, and a review
reason for every family. The default proposal calls out known ambiguous
families instead of silently treating their surface counts as semantic truth.

The pinned train and official-validation files are verified while being
streamed. Unicode-normalized stories are globally deduplicated by SHA-256, with
official-validation provenance taking precedence over an identical training
story. The disk-backed scan records the normalized bytes, EOS-terminated GPT-2
tokens, all matched families, per-form counts, and deterministic complete-story
examples. Its review packet contains the full noun list, projected role,
threshold evidence, greedy base-selection trace, exact forms, prevalence, and
story provenance in canonical JSON, Markdown, and standalone folding HTML.

Partition construction, JAX device discovery, preflight, training, evaluation,
and judging are unavailable until a manual approval artifact names the exact
breakdown hash. The approval also binds the decision document and pinned source
identity. Editing a form or decision invalidates the scan binding and approval;
changing any review-packet byte is detected before the packet can be reused.

### Overlapping partition semantics

Included noun families are ordered by descending unique training-story count
and noun ID. The base concept prefix stops at the first addition whose union
covers at least 50% of all unique training stories. Base membership remains a
union: a story is not removed because it also names a later task noun. A
deterministic 2% hash slice of this union is held inside the training source for
epoch-level base validation, keeping the official validation aggregate
untouched for final task evaluation.

Every remaining family with at least 256 training and 64 official-validation
stories becomes a task, ordered by descending training mass and noun ID. Task
membership is intentionally nonexclusive. Thirty-six context-fitting,
namespaced lowest-hash training stories are the task's parent/content-key
probes and are removed only from that task's update stream. Thirty-six
deterministic non-held-out base stories form the root key. The partition ledger
persists exact matched surface forms as well as all family memberships; task
summaries retain base overlap and every pairwise training overlap.

All official-validation task/story memberships are used for whole-story loss.
Midpoint generation uses every validation membership whose exact first half
contains at least one causal transition and leaves at least one position in the
2,048-position model. This is a mechanical model-capacity eligibility rule,
not a fixed-size sample. A story shared by two task nouns is evaluated twice
with different oracle nodes.

### Fresh base and VAMP-only graph

The base is a fresh seed-zero eight-layer GPT-Neo model trained for two epochs
with 256-token, 32-story microbatches and eight-way gradient accumulation.
Resume artifacts bind the full model configuration, tokenizer-bearing
partition, optimizer schedule, RNG, exact epoch/batch cursor, and preceding GPU
preflight. Optimizer state is committed every 1,000 updates and at epoch
boundaries. Publication requires finite training and validation loss, lower
validation NLL after epoch two, and a measured allocator peak no greater than
12 GiB; it imposes no semantic or absolute-NLL cutoff.

Adaptation trains only rank-eight, alpha-eight VAMP LoRA edges for 2,000 updates
per noun. The generic parent scorer always records raw mean prefix NLL for root
and every existing task node. Stage one can select only root; later stages use
the shared eligibility mask to select the insertion-order NLL argmin among
non-root nodes. Candidate training applies the selected node's complete path,
then commits a fresh edge. Every prior edge checksum is rechecked. Each stage
strictly persists the graph, tensors, address book, VAMP RNG, raw parent scores,
eligibility, content keys, loss trace, source-consumption counts, and exact
partition/base/config bindings. VAMP-only adaptation artifacts explicitly
declare their mode rather than fabricating empty independent or sequential
baselines.

### Evaluation and presentation

Whole-story evaluation scores `base`, the named task's `oracle` node, and the
four existing exhaustive/Hopfield/EBT task-free routes. Long stories are split
into non-overlapping causal windows so every target, including EOS, contributes
exactly once. The complete story supplies routing evidence; result rows stream
atomically with selected paths, story- and token-normalized loss, perplexity,
oracle match, and regret. Node loss and frozen-content encoding share bounded
chunks of at most 32 story windows. Differentiable EBT routing uses shape-stable
sub-batches of at most eight rows; this is a resource bound only, and its
per-row logits remain independent. A measured 32-row EBT attempt exhausted GPU
memory, while eight-row results matched its completed rows exactly.

Completion evaluation splits EOS-terminated tokens at the exact midpoint. A
router-facing value contains only the first half. All six conditions receive
that same prompt, true-suffix scoring mask, and greedy output budget; the saved
second half is evaluator/reference data only. OpenRouter judging presents the
prefix once and deterministically anonymizes the six completions plus the
reference. Prefix routing and suffix loss use a bounded multi-story path;
differentiable EBT remains capped at eight rows even when the host groups up to
128 suffix windows for length-aware generation packing.

Greedy decoding pre-fills each right-padded prompt once, retains fixed-width
per-layer key/value caches, and advances all active rows in one compiled device
loop. Cached global and local attention use each row's absolute insertion
position and valid-key mask, so unequal prompt lengths, EOS stopping, and LoRA
paths retain direct autoregressive semantics. A focused oracle test compares
multiple cached steps with full-prefix recomputation under mixed global/local
layers and per-row LoRA nodes. Conditions selecting the same node for the same
story share one generated row because their model, prompt, and deterministic
decoder are identical; all six labeled condition records are reconstructed
afterward. Cases are sorted by frozen output budget and first-fit packed by
distinct story/node rows and context compatibility. Each call has at most 72
addressed rows, and the observed process footprint is about 9.7 GiB, below the
12 GiB gate. Extraction still uses each story's true prefix length and frozen
budget. Strict 1--5 scores and a complete seven-way ranking are persisted
request by request and can resume without repeating completed calls.

The final Markdown and standalone interactive HTML are projections of strict
partition, adaptation, NLL, generation, and optional judge ledgers. They report
story- and token-weighted loss, per-noun acquisition, all routing accuracies and
regrets, confusion matrices, graph topology, overlap counts, descriptive
overlap correlations, judge means/ranks/pairwise wins, and a deterministic mix
of strong, weak, correctly routed, and misrouted story examples. No scientific
pass/fail threshold is attached to this experiment.

## TinyWorlds-Q Semantic-v1 Query-Native Knowledge Benchmark

`tinyworlds-q-semantic-v1` is a separate archive-grounded benchmark for direct
semantic knowledge. It does not reinterpret, import, alias, or modify
`tinyworlds-p-semantic-v6`; the v6 story-loss stop remains immutable negative
evidence. Query-v1 binds the same pinned TinyStories archive and GPT-2
tokenizer identities, but builds its own duplicate-group assignment ledger and
publishes only under `data/tinyworlds-q-semantic/`,
`checkpoints/tinyworlds-q-semantic-v1/`, and
`results/language_cl/tinyworlds-q-semantic-v1/`.

### Semantic authority and sealing

The benchmark's semantic authority is an explicit reviewed catalog, never an
automated extractor or model judgment. A concept definition owns exact
normalized surface forms. Each concept owns exactly twelve facts spanning at
least four relation categories. A fact records its canonical and accepted
answers, complete trigger closure, a grammatical answer type, and exact
construction-slice sentence provenance from at least sixteen duplicate groups.
Every accepted fact binds one extraction candidate and an affirmative human
decision for truth, answer forms, trigger closure, false distractors, and
evidence. Rejected reviewed candidates remain in the audit. These decisions,
all source identities, prompt text, answers, token IDs, evidence, and ordering
participate in the catalog SHA-256.

The extraction candidate's exact predicate must occur with the concept in
every authoritative construction-evidence sentence. Human-approved trigger
closure may additionally contain inflections or synonymous surface forms that
do not each appear in the construction slice. Those reviewed additions enter
the catalog hash and conservative story-level withholding, but they do not
retroactively manufacture construction evidence.

The pilot concept prefix is `rabbit, horse`, with surfaces
`rabbit/rabbits/bunny/bunnies` and `horse/horses/pony/ponies`. The first main
prefix is `cat, dog, bird, robot, dragon`, with the registered singular,
plural, kitten/puppy forms. Larger official catalogs must name a parent and
preserve every parent concept, fact, review, rejected candidate, and query as
an ordered byte-equivalent prefix.

Duplicate groups are assigned to construction when
`SHA256("tinyworlds-q-semantic-v1:construction" NUL group_sha256) mod 20 == 0`.
Construction groups are visible to extraction and human review only. They are
permanently excluded from every base or adapter model input. Discovery ranks
exact same-sentence concept/predicate n-grams and retains complete archive
record, story, group, member, index, and sentence provenance. Discovery output
is proposal evidence and cannot publish a catalog by itself.

The complete discovery packet is an audit appendix, not the human decision
queue. A compact shortlist binds one targeted evidence packet, the pinned
tokenizer identity, proposed facts, accepted and trigger forms, proposed false
choices, exact answer-token suffixes, support counts, and deterministic
representative sentences. Its primary surface contains exactly twelve rows per
concept and keeps four alternatives per concept in a separate backup section.
The same manifest-driven implementation publishes the two-world pilot and
five-world main surfaces. The full evidence remains addressable by candidate
hash when a row needs closer inspection. Neither ranking, shortlisting, nor a
checked-looking render creates semantic authority; only an explicit recorded
human decision can promote a proposal into the catalog.

Each fact has three validation paraphrases (two forward and one reverse) and
five sealed-test paraphrases (three forward and two reverse). Across those
eight templates, each answer position occurs exactly twice. All four answers
share a reviewed grammatical type and have equal tokenizer suffix length.
Reverse distractors are reviewed per fact rather than reused as one concept
pool: another concept that also satisfies the exposed fact is not a false
distractor. Primary fact approval and reverse-choice approval are distinct,
content-addressed human decisions, so approving facts cannot implicitly approve
unseen or semantically invalid reverse choices.

Approved catalog compilation is manifest-driven rather than pilot- or
five-world-specific. The compiler requires the complete ordered shortlist,
the exact ordered primary IDs, the exact per-fact reverse IDs and concepts,
and one reviewer identity across both approval layers. This keeps the
rabbit/horse and five-world catalogs on the same codepath while making a
missing, reordered, or cross-manifest approval a hard error.
Catalog publication physically separates metadata, validation templates, and
sealed test templates. A normal loader authenticates but never deserializes
the sealed payload. Test deserialization requires one durable transaction that
binds the catalog, partition, selected base, all adapters, and full experiment
configuration. An opened transaction may resume after interruption; once its
result is marked complete it cannot open or deserialize the test again.

### Fact-withholding partition semantics

Matching and assignment operate on complete normalized duplicate groups, so a
group is assigned once and all exact archive occurrences move together. A
story-level match determines leakage:

- a non-construction story with no registered trigger for any concept it
  mentions is eligible for the 96/2/2 base split, including ordinary lexical
  mentions of target words;
- a story with exactly one target concept and one or more of that concept's
  registered triggers is assigned to that concept's 90/10 node split;
- a fact-bearing story that mentions more than one target concept is excluded;
  and
- construction and excluded stories remain in the authenticated assignment
  and exact-byte ledgers but are not yielded by model-input iterators.

The trigger and concept may occur in different sentences for conservative
story-level withholding. A group counts as authoritative support for a fact
only when a registered concept surface and trigger occur in the same sentence.
Publication requires at least 32 distinct authoritative non-construction node-
training groups per fact and at least 256 non-fact base-training groups that
mention each concept. Exact story bytes and little-endian token sequences are
persisted with archive provenance and offsets. A strict loader rehashes every
file, reconstructs assignment counts, checks source bindings and directory
identity, and proves that no construction group entered a model role.

Before GPU work, a separate validation-only sample artifact selects the
lexicographically first exact story from base validation and from each node's
validation index. It records archive provenance, exact story bytes, token IDs,
and every validation query record. Its API accepts only a
`ValidationCatalogView`, binds the complete partition and catalog identities,
and rejects changed files or any attempt to substitute sealed templates.

### Training and adaptation persistence

The base trainer reads authenticated split indexes and memory-maps the token
payload. It shuffles bounded 1,024-document blocks deterministically, preserves
order inside a block, and creates fixed 32-by-256 token batches without loading
the corpus into memory. Its resume identity binds the partition, complete
GPT-Neo architecture, optimizer schedule, seed, accumulation, and planned
update count. Checkpoints contain parameters, AdamW state, random state, the
next epoch/block/microbatch cursor, and schedule position; loss rows are
appended to the printed working directory as they are produced. Resuming trims
only a post-checkpoint trace tail and must reproduce uninterrupted parameters
and trace bytes.

GPU entrypoints disable JAX pool preallocation before importing accelerator
code. The operational memory gate uses the device allocator's measured peak
bytes in use, merged with the registered preflight peak across resume
boundaries, rather than treating an implementation's reserved pool as model
memory. Base orchestration is shared across pilot and main manifests: it
strict-loads the matching preflight, resumes the exact training identity,
persists validation evidence after each epoch, and publishes only a complete
passing two-epoch base.

A base can cross into adapter work only through the query-v1 selected-base
publisher. That boundary accepts a complete two-epoch query-native run only
after both held-in NLL measurements and the measured allocator peak pass the
registered gate. It writes a strict source-bound GPT-Neo checkpoint and rejects
semantic-v6 or arbitrary checkpoints. The experiment hash expands and records
the full model fields, rank-eight all-projection target mask, and derived
adapter optimizer contract rather than relying on code defaults.
The selected base binds the full catalog partition and base-training contract,
not an active adapter prefix, so the same authenticated checkpoint can serve
preserved 5-, 10-, 20-, and 100-world prefixes.

Adapter preparation accepts the validation-only catalog view. It compiles
exactly 36 question prefixes per world and deterministically selects the root,
parent-search, and content-key probes; no answer suffix or sealed template is
available on this path. Each world materializes at most the registered update
budget of node batches, then advances independent, continually overwritten,
and VAMP systems through the existing shared trainers. Every completed world
publishes a strict real-tensor adaptation artifact containing task order,
graph, address keys, adapters, training traces, and all three random streams.
An interrupted run resumes the latest complete prefix and deterministically
replays only an incomplete world.

The pilot trains one deterministic 2,000-update independent trajectory per
world and persists exact adapter/trace snapshots at absolute updates 500,
1,000, and 2,000. This makes the three learnability measurements genuine
prefixes of the same seed-zero optimizer trajectory and keeps worst-case pilot
training within the preflight's 12,000-update projection. Only matching
independent snapshots contribute to selection. After all three budgets are
measured, the smallest passing budget trains sequential and VAMP stages while
reusing the selected independent snapshots. Those full stage artifacts bind
the independent-sweep hash, are strict-reloaded through the completed-stage
resume path, and are evaluated at both world stages. This exercise cannot
consult sealed prompts or tune the main configuration against a VAMP score.

### Query evaluation and statistics

Reviewed templates compile into the existing four-candidate `KnowledgeQuery`
contract. The query wrapper adds catalog, concept, fact, direction, split, and
template identities while retaining the shared answer-only candidate scoring,
hard-node scoring, VAMP routing, support, and regret computations. Forward
questions expose a concept and ask for a property/action; reverse questions
expose the fact and ask for the concept. VAMP parents and router keys consume
validation question prefixes only.

Stage evaluation right-pads and stacks only router prefixes. Candidate answers
remain in separate competence batches and only their suffix tokens receive
loss. Reference scores are produced in bounded query chunks. Each stored
adapter, VAMP oracle path, and task-free router is projected through the shared
knowledge evaluator, so candidate margins, node accuracy, and routed regret do
not acquire a query-v1-specific numerical definition. Independent adapters are
also scored against other learned worlds to retain node-specificity evidence.

The pilot result is a content-addressed validation-only bundle. It binds the
catalog, partition, preflight, selected base, all three budget configs and
tensor/manifest identities, exact per-query JSONL ledgers, selected-budget
resume parity, runtime, and allocator peak. Its Markdown and standalone HTML
are derived views. Publishing this bundle proves pilot learnability and
operational integrity only; it never creates a VAMP scientific verdict or test
authorization.

When no budget passes, the runner instead publishes a content-addressed
validation-only failure bundle before raising the mandatory stop. That bundle
binds the same frozen sources and independent-sweep tensors, every budget's
base accuracy, adapter accuracy, acquisition delta, exact per-query JSONL, and
allocator evidence. It contains no selected budget or selected adaptation.
Strict retry loading treats that failure as terminal under its exact policy,
so it cannot be silently re-evaluated into main authorization.

The original pilot did stop this way: both worlds needed 60% absolute accuracy
and a 15-percentage-point acquisition delta, and rabbit missed the latter at
all budgets despite exceeding the absolute threshold. That failure remains
immutable. A later user-authorized protocol-amendment artifact may reference
the exact failure and sweep, preserve their bytes, name a new policy, and
explain the change. It cannot mutate or relabel the original result. The
registered amendment keeps 60% absolute accuracy as the pilot gate and makes
acquisition mandatory descriptive evidence because permitted lexical exposure
made the relative threshold ceiling-sensitive. Under that amendment, the
smallest passing budget is 2,000 updates. Sequential/VAMP completion and exact
resume remain operational requirements before main authorization, and sealed
prompts remain unavailable throughout.

Main execution is separately frozen by a content-addressed authorization that
binds the failure, amendment, completed pilot result, concept order, 2,000
updates, fresh seed-zero base, exact query protocol, answer-only scoring, nine
methods, fact-level bootstrap contract, validation-only parent/router inputs,
and one sealed opening after all artifacts are frozen. Main fact extraction
and human review may proceed after that freeze, but no proposal becomes a fact
until the primary and reverse approval artifacts are recorded.

Every primary statistic first averages all paraphrases within a fact. Accuracy,
correct-answer margin, acquisition, specificity, retention, router accuracy,
and routed regret are then aggregated with 10,000 deterministic fact-resampled
bootstrap replicates. Facts are resampled within each world and world means
receive equal weight; templates and tokens are never treated as independent
observations. Greedy generation is secondary and reports only exact registered-
trigger recall plus raw outputs. Final reports are descriptive and carry no
VAMP scientific pass/fail verdict.

### Dynamic execution and resource boundaries

All query-v1 training and evaluation surfaces consume an ordered concept
manifest. For `N` active worlds, graph capacity is derived as `N + 1` nodes and
`N` edges; stage prefixes, tensor masks, progress totals, report labels, parent
search, and result estimates use the same manifest. A large catalog and one
base partition can serve preserved prefixes of 5, 10, 20, or 100 worlds.

The `full` schedule evaluates every learned world after every stage and is
bounded to twenty worlds. The `milestone` schedule always records each world's
acquisition, evaluates all learned worlds at configured milestones, and
performs a complete final evaluation. Query/node scoring is bounded by a
configured chunk size, and JSONL ledgers stream to a reported temporary
directory before atomic publication. Preflight separately estimates training,
parent search, routing, result storage, and peak memory. The 100-world path is
the same implementation and fails explicitly when measured allocator or
projected result bytes exceed frozen limits.

Result-size accounting includes the base rows once per world, every scheduled
row for each non-base non-independent method, and the complete forced
independent-adapter by query-world matrix at each evaluated stage. The earliest
v1 preflight artifacts counted only matching independent rows; strict loading
preserves those immutable measurements but also applies the corrected
projection before reuse. Newly constructed projections use the complete
matrix, and final publication binds and checks the actual JSONL byte count
against the experiment limit.

Final publication requires the exact scheduled sealed-test cells for all nine
registered methods, sixty unique queries per world/cell, 10,000 bootstrap
replicates, acquisition/specificity/retention effects, ordered generation
inspection, runtime, and memory evidence. The content identity binds the
streamed result-ledger SHA-256; strict reload rejects changed renderings or any
altered result byte.

Condition accuracy, margin, router accuracy, and routed regret summarize only
the primary matching adapter or oracle/path row at each stage and method. The
forced independent-adapter by query-world matrix remains in the authenticated
ledger and feeds the specificity effect, but is not pooled into the headline
matching-independent accuracy.

Strict validation/report recovery reconstructs each canonical result record
through the public result contract, reruns exact schedule and routing coverage,
and recomputes registered fact-level analyses. Final report reload additionally
regenerates Markdown, HTML, and the dynamic schedule. Matching file hashes are
necessary but are not by themselves sufficient recovery evidence.

The shared knowledge scorer supplies a task-oracle node for every direct
semantic query, including base, independent, sequential, and VAMP-oracle rows.
That field is hard-node provenance, not proof that routing occurred. Strict
query recovery therefore requires an oracle node for every method, while only
the five task-free routed methods may carry a selected node and routed regret.
This distinction preserves the shared scorer's result semantics instead of
normalizing away non-router oracle evidence in the query wrapper.

Before a sealed transaction can open, a content-addressed validation freeze
must bind the selected base, preflight, validation-only probe preparation,
every immutable stage tensor/manifest, exact validation ledger, no-op resume
parity, corrected result-size projection, and complete final-analysis protocol.
The sealed runner authenticates that chain before writing the durable open
marker. A retry may resume the same opened transaction or idempotently close an
already published matching report, but a completed transaction cannot evaluate
or deserialize test prompts again.

The canonical five-world execution exercised that boundary exactly once.
Transaction
`ce92e165fcc3f58b449253a628e7616ef254c700c225db45ab88708f8f8de946`
is durably opened and completed by descriptive report
`8f34f8fe9f791ae822b2cdde35ebb1cb24b9a4f7efab0c68e0cf600f694a9986`.
Its 9,900-row ledger, 10,000-replicate fact analysis, generation records,
runtime/memory evidence, Markdown, HTML, and dynamic schedule all strict-load
from their frozen bindings. This completion is terminal for that transaction:
future invocations may authenticate and return the report but cannot obtain a
second test view.

### Post-result presentation views

An explanatory presentation may be derived after a sealed transaction is
complete, but it is not a new evaluation. Its only test-bearing inputs are the
transaction-published opened audit and the strictly authenticated final result
ledger. Building it must not call the sealed catalog loader, score a model,
choose a checkpoint, alter a router, or create another opening. The page binds
and displays the final report, catalog, transaction, ledger, and opened-audit
identities.

The active five-world presentation is a forward-only post-result diagnostic.
It recomputes every displayed accuracy, effect, interval, world breakdown, and
router comparison from the 180 test prompts whose direction is `forward` and
which explicitly name their concept. The 120 reverse prompts are excluded by
one direction-level rule because some do not uniquely identify a routable
world. This derived scope must be stated prominently and cannot replace or
mutate the registered all-direction report. Directional paraphrases are still
averaged within each fact before 10,000-replicate equal-world bootstrapping.

Example selection is fixed structurally rather than chosen by outcome. The
forward-only presentation uses test paraphrase 00 for every reviewed fact, so
its 60 cases preserve all 60 facts and equal world weight while ensuring that
every displayed prompt names its world. Success, persistent-miss,
sequential-loss, and routing-miss views are filters over that fixed set; they
do not change membership. Each case may expose the published choices, all
final method outcomes, selected node, and reviewed construction evidence.

Because the page reveals completed test prompts and outcomes, it must visibly
identify itself as post-result material and must not become training, tuning,
parent-selection, router-key, or checkpoint-selection input for a later run.
A later experiment needs new sealed questions. Presentation HTML is standalone
and dependency-free so its explanation, folding, filters, examples, and exact
source bindings remain available as one portable file.

Runtime evidence distinguishes the current `adaptation_or_resume` invocation
from `adaptation_stage_wall_interval`. The latter is the elapsed filesystem
interval from creation of the dedicated adaptation workspace to publication of
the final immutable stage manifest. It survives a publication-layer retry and
must be described as a wall interval, not summed accelerator compute time,
because a genuinely interrupted run may include idle time between stages.

The retained seed-zero GPT-Neo architecture, two-epoch base optimizer schedule,
rank-eight all-projection LoRA, and 12 GiB allocator ceiling are fixed in the
experiment identity. A base proceeds only when held-in epoch-two NLL is at most
2.2, improves by at least 0.02 from epoch one, remains finite, and fits memory.
Before reusable training, a content-addressed GPU preflight runs exactly two
disposable base updates and warm validation, parent-search, and task-free
routing probes. It binds the catalog, partition, full dynamic preset, measured
allocator peak, result-size estimate, and runtime projections; its parameters
cannot be promoted into the real run.

The canonical five-world scratch base is selection
`0777adef5291c416d53af23ac6694bcfd308f0f6534883e4cc7cede2254783a2`,
trained under identity
`001e16d8908ae593ffc23b423a1a672e005c3cf7b35dacbb09636d1807a96d93`.
Its held-in NLLs are `1.266449873` and `1.189207350`; the `0.077242523`
improvement and 9,032,018,176-byte allocator peak satisfy the frozen base
gates. This selection, rather than any semantic-v6 parameter artifact, is the
only base authorized for the five-world adapters. A completed-launcher replay
must strict-load this selection and return without changing optimizer state;
that no-op replay passed before adapter launch.

The original rabbit/horse policy tested adapter budgets 500, 1,000, and 2,000
in order and required both worlds to reach 60% validation accuracy and gain 15
percentage points over base. Its failure is retained. The registered amendment
selects the first budget where both worlds reach 60%, reports acquisition
without thresholding it, and selected 2,000. Main execution uses the separately
frozen order `cat, dog, bird, robot, dragon` for independent, sequential, and
VAMP systems.

## TinyWorlds-P Semantic-v6 Exact Comparison Feasibility

`tinyworlds-p-semantic-v6` is a new benchmark version that addresses the
comparison-story shortage observed in semantic-v5. Semantic-v5 remains an
immutable failed construction. Version 6 binds the successful semantic-v4
catalog
`ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`
and the semantic-v5 partition failure
`090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`.
The parent failure, in turn, authenticates the complete semantic-v4 topology
audit and its 22 balance-eligible candidates.

### Single isolated intervention

Version 6 changes only the point at which exact comparison feasibility is
checked. It retains the version-5 requirement that all five selected cells lie
within 10% of their median active-token mass. For every candidate that passes
that balance rule, version 6 runs the actual deterministic archive allocation:
the five world cells use the frozen 80/10/10 split, the held-in remainder uses
the frozen 96/2/2 split, and complete duplicate groups remain intact.

A candidate is comparison-feasible only if the existing allocator completes
all row and column controls for all five worlds in both validation and test.
This is the complete ten-control allocation, not the earlier whole-archive
capacity estimate. It includes the fixed `E, A, C, B, D` reservation order,
global non-reuse within each evaluation split, exact nuisance matching, token
matching, and every existing control tolerance. A normal registered
partition-gate failure makes that candidate ineligible. An unexpected error is
an implementation failure and stops the run rather than being counted as
scientific evidence.

All balance-eligible candidates are measured before selection. The retained
comparison-feasible candidates are then ranked by the unchanged objective:
semantic dispersion, token imbalance, nuisance imbalance, negative control
capacity, and finally a semantic-v6-namespaced canonical hash. Model losses,
training observations, and sealed test data never enter this decision. If no
candidate is feasible, version 6 publishes a content-addressed failure and
stops. It does not reduce the evaluation sample, reuse controls, relax a
matching tolerance, or change the 10% balance rule.

Each feasibility record binds the candidate, its semantic rank, the SHA-256 of
its complete split-assignment stream, and either the SHA-256 of its successful
control family or its exact registered failure reason. The selected candidate
is allocated again for publication and must reproduce the recorded split and
control identities exactly. Candidate parallelism and external-sort batch
sizes are execution details and may not affect any record. A complete second
archive build with a different sort batch size must reproduce the selected
partition or failure byte for byte.

Every other semantic-v5 boundary remains fixed: the 790 nouns, 284 verbs,
eight clusters per role, construction exclusion, exact archive bytes,
tokenizer, adjective/source/feature/length measurements, one-to-one
world/control pairing, validation-only sample report, and sealed-test boundary.
No semantic-v4 or semantic-v5 artifact may be loaded through a version-6
compatibility alias.

### Canonical semantic-v6 partition

The canonical archive replay retained 2,520,317 unique-story groups and
479,183,203 scored tokens and reproduced all 28,224 parent topology records.
Twenty-two layouts passed the unchanged 10%-around-median balance rule. The
builder ran the real split and complete validation/test comparison allocation
for every one of them before selection.

Seventeen layouts completed all comparisons. Semantic ranks 0, 8, 9, 14, and
21 failed because world B's validation column comparison lacked enough
distinct stories after the fixed split and global non-reuse order. Rank 0 had
1,649 candidates for 2,314 required stories. Rank 1 was the first feasible
layout and therefore won under the preregistered semantic ordering. Its cells
for A through E are `(2,4), (7,4), (7,6), (2,6), (3,2)`, with scored-token
masses `6,136,097`, `5,873,159`, `5,921,676`, `6,114,634`, and `5,440,146`.
All five lie inside the fixed interval around the median of `5,921,676`.

The selected layout's split-assignment identity is
`b090e78fbfa80a26ea4dd4403612c199e06c634c1b31d68a4edec0055aa55999`,
and its complete comparison-family identity is
`583a9cff169564d6043ea51cc3a4ab92bee6b5a5d14a5e74297ad3845fdec976`.
A fresh publication-path allocation reproduced both identities and produced
31,117 one-to-one world/comparison pairings.

The successful partition is
`3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa`.
A second archive build using 37,000-record external-sort batches reproduced the
50,000-record primary build byte for byte across all 167 files. Both strict
loaders passed, and their common tree SHA-256 is
`b5ba1ce33d1cad7eb00bba0b6eec35e2b94c3a6b997a20149081cc61c862279d`.
The validation-only sample report is
`b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579`.

This result closes semantic-v6 construction only. No GPU training, checkpoint
selection, model loss, semantic-gap decision, or sealed-test access occurred.
Training and resume artifacts must bind this exact partition through a
version-6-native strict boundary before calibration may begin.

## Semantic-v6 Base Gate and First VAMP Experiment

The first downstream experiment is
`tinyworlds-p-semantic-v6-vamp-chain-v1`. It is a fixed, exploratory
continual-adaptation study over the canonical semantic-v6 partition. It does
not change the catalog, partition, comparison stories, base-model gate, or
checkpoint-selection rule.

### Base training and selection boundary

Semantic-v6 has its own training, resume, validation, selected-checkpoint, and
publication formats. They reject semantic-v1 and archive-v1 artifacts even
though the optimizer and GPT-Neo architecture remain scientifically
unchanged. The run starts from seed zero, uses the registered five-epoch
schedule, and first trains exactly two epochs. Epoch two must satisfy the
registered empirical semantic gate and held-in quality requirements before
epochs three through five can run. Among epochs two through five that satisfy
the same semantic gate, the checkpoint with the lowest held-in validation NLL
is selected, with the earlier epoch breaking an exact tie.

Selection publishes the base checkpoint, tokenizer, validation ledgers,
sample report, and resume state without reading a test index. Adapter training
begins only from this strict selected-base artifact. The sealed test is not
part of calibration, continuation, checkpoint selection, parent selection,
adapter optimization, or router-key construction.

The base runner writes an authenticated optimizer, RNG, schedule, and exact
next-batch cursor every 1,000 updates and at every epoch boundary. After an
interruption it selects the newest strict checkpoint, removes only loss-log
records beyond that checkpoint, and resumes deterministically. Completed
group-loss ledgers are atomically renamed into place. An incomplete validation
directory is moved intact under the run's recovery directory before that epoch
is reevaluated, so a partial ledger is never mistaken for completed evidence.

Before any real update, a separate
`tinyworlds-p-semantic-v6-gpu-preflight` identity runs exactly two disposable
updates and one warm validation batch. Its resume format cannot be loaded by
the real run. It checks finite loss and the 12 GiB allocator limit and
publishes measured runtime estimates. The single fixed runner stops after a
new preflight so that proceeding requires a later invocation after review.

### Frozen five-world adaptation study

The task order is exactly `A, B, C, D, E`, using cells `(2,4)`, `(7,4)`,
`(7,6)`, `(2,6)`, and `(3,2)`. The experiment preset has SHA-256
`ca16318486600745e8a49903f495819741082f120fa7b95b3f9277efa83ada73`.
Each of the three trained adapter systems receives exactly 2,000 updates per
world with rank-eight, alpha-eight LoRA on every supported projection, AdamW
learning rate `0.001`, weight decay `0.01`, and gradient clipping at `1.0`.
The systems are one continually overwritten LoRA, five independent root
LoRAs, and VAMP's immutable pathwise graph. VAMP may contain six nodes and
five edges. Parent search uses only 128 deterministic validation spans from
each new world; the root key uses 128 held-in validation spans. Every selected
span is 256 tokens and at most one span comes from a duplicate-story group.

Task-boundary artifacts persist the three independent random streams, all
adapter tensors, the VAMP graph, address keys, parent scores, and complete
update-loss traces. A resumed run reconstructs immutable progress at the last
complete world and must be tensor-identical to uninterrupted execution. It
never reconstructs an old semantic or archive compatibility path.

### Evaluation and one-time test transaction

After all adapter tensors are frozen, the runner durably writes one sealed
transaction binding the partition, selected base, adapter publication, and
complete experiment config. Only then may any test index be read. A fixed test
suite selects 128 deterministic 256-token spans per world and evaluates nested
prefix lengths 16, 32, 64, and 128 against a disjoint 128-token suffix. The
primary condition is the 64-token prefix. Exact whole-word occurrences from
the target noun and verb clusters classify each visible prefix as
cue-sufficient, cue-present, or cue-hidden/ambiguous; this label is an
evaluation stratum and is never supplied to a router.

The full comparison has four stored methods—frozen base, sequential LoRA,
independent root LoRA, and VAMP oracle—and five task-free routers—exhaustive,
Hopfield, uniform-initialized EBT, Hopfield-initialized EBT, and deterministic
random node. Hopfield uses beta `10` and top-k `4`. EBT uses 20 steps, learning
rate `0.1`, temperature `1.0`, and entropy penalty `0.01`. Evaluation stores
every stage, prior task, prefix, and nonempty cue stratum, plus acquisition and
best-so-far forgetting, parent-transfer diagnostics, persistent and padded
memory, and synchronized cold/warm routing cost.

Adapter specificity is diagnostic rather than a gate. For each world, the
final sequential adapter, that world's independent adapter, and that world's
VAMP oracle path are forced on the world stories and on both already-paired
comparison arms. The statistic is the adapter's NLL improvement on the world
minus its improvement on the comparison. Row and column arms remain separate,
and each receives a deterministic 10,000-replicate paired bootstrap interval.
Routers are not assigned a control accuracy because comparison stories have no
task-oracle node.

The base sealed evaluation retains its registered bootstrap, label-swap, and
Holm statistics, but the VAMP study has no new pass/fail threshold. Its final
content-addressed Markdown, standalone HTML, JSON, JSONL, ledgers, timings,
and provenance report is explicitly exploratory and cannot alter the selected
checkpoint or reinterpret semantic-v6 construction.

The sealed authorization names one durable transaction rather than assuming a
process cannot fail. If its base evaluation is interrupted, the incomplete
directory is preserved under that transaction's recovery directory and the
same bound transaction may finish it. Adapter and final-result publications
record the largest observed JAX allocator peak and reject a peak above 12 GiB.

## TinyWorlds-P Semantic-v5 Balance-Eligible Topology

`tinyworlds-p-semantic-v5` is a new benchmark version that addresses the
specific partition failure observed in semantic-v4. Semantic-v4 remains an
immutable failed result. Version 5 may use the v4 audit as parent evidence, but
it may not relabel a v4 candidate as the v4 winner or otherwise reinterpret
that result.

### Single isolated intervention

Version 5 binds the successful v4 semantic catalog
`ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`
and the v4 partition failure
`37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`.
It reuses the same 790 nouns, 284 verbs, eight clusters per role, frozen
centroids, construction exclusion, canonical archive, tokenizer, and
adjective/source/feature/length measurements.

The only change is when the existing 10% five-cell mass rule is applied. In
v4, all visible and control-capable candidates were ranked for semantic
quality, and the single winner was checked for balance afterward. In v5, a
candidate is eligible only when all five cell masses lie within 10% of that
candidate's median. Candidates that fail this requirement are removed before
ranking. If no candidate remains, v5 stops.

Among eligible candidates, the ranking is otherwise unchanged. It minimizes,
in order, semantic dispersion, token imbalance, and nuisance imbalance; it
then maximizes control capacity and uses the v5-namespaced canonical hash only
to break an exact tie. Component visibility and row/column control capacity
remain eligibility requirements before the mass check. Model losses, training
results, and sealed test data are never inputs to selection.

### Independent replay and downstream boundary

The v5 builder must recompute the complete topology audit from the archive and
must reproduce the parent v4 candidate measurements before selecting a v5
winner. The parent failure is embedded in the partition as authenticated
construction evidence. Separate v5 tree, partition, sample-report, resume, and
checkpoint formats are required; there are no v4 compatibility aliases.

If a balanced topology is selected, all later mechanics remain unchanged:
80/10/10 world splits, 96/2/2 held-in splits, complete duplicate groups, exact
archive bytes, global control non-reuse, one-to-one world/control pairings, and
the sealed-test boundary. A validation-only sample report and an independent
byte-for-byte partition rebuild are required before any GPU preflight or
optimizer update. A failure during allocation or control matching stops v5
without changing a tolerance or choosing another topology.

### Canonical semantic-v5 partition stop

The canonical archive replay reproduced all 28,224 version-4 topology records
and the exact retained mass of 2,520,317 duplicate groups and 479,183,203
active tokens. Twenty-two candidates passed the version-5 balance eligibility
rule. The semantic leader among them used cells `(3,4), (4,4), (4,6), (3,6),
(2,0)` for A through E, with active-token masses `9,899,869`, `8,829,612`,
`8,742,369`, `10,104,204`, and `9,357,468`. Their median was `9,357,468` and
all five lay inside the fixed 10% interval, so version 5 resolved the version-4
topology imbalance without changing a word, cluster, or threshold.

The full split allocation then exposed a stricter control limitation. World B
had 4,628 validation groups and therefore required 2,314 groups in each control
arm. After the registered global non-reuse order reserved controls for E, A,
and C, B's column arm had only 1,511 candidates. This raw count check occurs
before nuisance or token matching, so relaxing a matching tolerance would not
repair the 803-group shortage. The coarse pre-split control-capacity screen is
therefore not a guarantee of exact split-level, globally non-reused control
feasibility.

Version 5 stopped as preregistered and did not substitute another of the 22
balanced candidates. The authenticated failure is
`090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11`.
It binds the exact parents, frozen settings, selection, exclusions, structured
shortfall, and assignment-ledger identity. A fresh archive rebuild with a
different external-sort run size reproduced the 3.95 GB assignment ledger's
SHA-256 and all 54 MB of the failure directory byte for byte.

Semantic-v5 has no partition, sample report, GPU run, checkpoint, or sealed-test
result. Exact split-level control feasibility as a pre-ranking requirement, a
different control allocation, a smaller evaluation sample, story reuse, or a
different candidate choice would each require a newly registered version.
None may be used to reinterpret the version-5 stop.

## TinyWorlds-P Semantic-v4 Frozen-Centroid Boundary

`tinyworlds-p-semantic-v4` is a new, preregistered construction contract that
tests the instability isolated by semantic-v3. Archive-v1 and semantic-v1--v3
remain immutable evidence. V4 cannot waive a v3 failure, use model loss, inspect
a partition or checkpoint, or open sealed test data.

### Single isolated intervention

V4 binds semantic-v3 failure artifact
`ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`
and reuses its authenticated v1 encoder evidence, v2-calibrated role decisions,
sense decisions, and candidate vectors. For each role it must reproduce v3's
pass-zero fit exactly: eight unweighted spherical clusters, the v3 farthest-first
hash authority, nearest-cosine assignment, float32 centroids, and at most 100
iterations. The starting assignments, centroids, margins, and pass-zero summary
must equal a fresh replay from the bound vectors.

The only intervention is that the fit is then frozen. Every candidate is
screened once using its assigned-centroid minus best-alternative-centroid cosine
margin from that fit. A word below `0.03` is out of grid. V4 does not delete and
reseed, reassign a survivor, or recompute a centroid after this screen. This is
the fixed-reference question suggested by v3's deletion/reseeding cascade; it
is not a sixth v3 pass or a changed margin threshold.

### Catalog meaning and gates

Published centroids are **fit centroids**, estimated before boundary exclusion.
Published cluster word inventories and token masses contain only retained
members. The exhaustive word ledger retains the vector, fit cluster, and fixed
margin for every role/sense-qualified word, including boundary exclusions, so a
strict loader can reconstruct the one-time fit and every disposition. A
retained word's final cluster must equal its fit cluster. Pre-cluster role and
sense exclusions retain no vector or fit assignment.

After the one-shot screen, every fixed cluster must retain at least 32 nouns or
12 verbs. Fit-centroid pair cosines must remain below `0.90`, and nouns and verbs
together must retain at least 40% of non-construction archive token mass. The
cluster count, role calibration, sense threshold, semantic vectors, word-count
floors, centroid ceiling, coverage floor, and later story-allocation balance
are unchanged. Failure of any gate publishes an exhaustive content-addressed
v4 failure and stops before partition work. A pass authorizes only catalog
publication and subsequent independently frozen partition work, never training
by itself.

### Canonical semantic-v4 result

The canonical catalog is
`ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee`.
Its one-shot screen exactly reproduced v3 pass zero: 188 of 978 noun
candidates and 81 of 365 verb candidates fell below `0.03`. The fixed clusters
retain 790 nouns and 284 verbs. Their minimum cluster inventories are 39 nouns
and 18 verbs, the maximum fit-centroid pair cosines are `0.8735721184` and
`0.8916218581`, and joint token coverage is `53.341729362%`. All frozen gates
therefore pass.

This result establishes a valid semantic word grid, not a trained benchmark.
It authorized the independently frozen partition attempt recorded below; it
did not guarantee that attempt would pass. The v4 catalog may not be silently
loaded through a v1--v3 partition compatibility path.

### Semantic-v4 partition boundary

The v4 partition is a separate `tinyworlds-p-semantic-v4` artifact and accepts
only the strict v4 catalog, canonical archive, and canonical tokenizer. Its
tree and partition formats are version-specific; v1--v3 catalog or partition
artifacts are rejected rather than adapted. The embedded catalog is the exact
content-addressed catalog, including both frozen-fit and retained views.

All eligible non-construction duplicate groups whose noun and verb survive the
catalog enter deterministic allocation without replacement. V4 preserves the
80/10/10 world and 96/2/2 held-in split weights, complete duplicate groups,
exact archive story bytes, adjective/source/feature/length nuisance handling,
global control non-reuse, and the sealed-test boundary. Execution-only worker
and external-sort run sizes are not identity inputs and must produce identical
bytes.

Topology is selected without model observations. Candidate topologies are the
same A/B/C/D 2-by-2 corner plus unrelated E. Candidates first must pass
component visibility and row/column control-capacity filters. The remaining
lexicographic objective minimizes semantic dispersion, selected-cell token
imbalance, and nuisance imbalance; then maximizes control capacity and uses a
v4-namespaced canonical hash tie. The selected five cell masses must each lie
within 10% of their median. Every selected component must occur in at least 64
outside groups.

After 80/10/10 and 96/2/2 allocation, validation and test world groups receive
globally non-reused controls matched under the frozen token, source/feature,
adjective/length, and mean-length tolerances. Each world group is then paired
one-to-one with a control group, ordered first by nuisance stratum and then by
token length, active mass, and v4-namespaced canonical hash. Pairings are
persisted for bootstrap and label-swap statistics without changing aggregate
NLL.

Before any optimizer update, a content-addressed v4 sample report must cover
held-in validation, all five semantic-world validations, and both row and
column control arms for every world. It contains complete retained cluster
inventories and exact story bytes plus archive provenance. Its reader rejects
test indexes by filename and records `sealed_test_opened: false`.

### Canonical semantic-v4 partition stop

The canonical archive replay retained 2,520,317 non-construction duplicate
groups and exactly 479,183,203 active tokens, equal to the catalog mass. It
permanently excluded 247,629 construction groups containing 47,172,075 tokens
and 2,198,121 groups containing at least one catalog-excluded noun or verb,
containing 419,143,883 tokens. No group was sampled or replaced.

All 28,224 physical A/B/C/D-plus-E candidates were nonempty, passed the
64-outside-group component-visibility filter, and had sufficient row/column
control capacity. Under the preregistered lexicographic objective, the winner
used cells `(1,2), (3,2), (3,4), (1,4), (6,1)` for A through E. Its active-token
masses were `2,559,355`, `5,440,146`, `9,899,869`, `4,699,583`, and `1,428,732`.
Their median is `4,699,583`, so the fixed 10% interval is
`[4,229,624.7, 5,169,541.3]`; four of the five masses lie outside it. The
partition therefore stopped before split allocation.

There are 22 candidates satisfying the median gate as a diagnostic fact. The
best of them uses `(3,4), (4,4), (4,6), (3,6), (2,0)` and has masses
`9,899,869`, `8,829,612`, `8,742,369`, `10,104,204`, and `9,357,468`. Its
semantic-dispersion score is `0.3297871012`, behind the selected candidate's
`0.3235518405`. Substituting it after observing the stop would change v4 from
semantic-first lexicographic selection to balance-feasibility-first selection;
v4 does not make that post-hoc change.

The strict content-addressed failure is
`37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07`.
It binds the archive, tokenizer, catalog, partition preset, seed identity,
semantic exclusions, adjective buckets, exact score fractions, all ranked
candidates, and the failed median calculation. An independent replay produced
the same identity and every failure-artifact byte. Because no success
partition exists, v4 has no split allocation, paired controls, sample report,
runtime preflight, calibration, checkpoint, or sealed-test result. Altering
the objective order, treating median balance as a prefilter, changing 10%, or
choosing one of the 22 diagnostic candidates requires a new benchmark version.

## TinyWorlds-P Semantic-v3 Semantic-First Boundary

`tinyworlds-p-semantic-v3` is a new, preregistered construction contract. It
changes the semantic-v2 cluster objective after v2 showed that forcing word
assignments into 90--110% token-mass capacities made cluster membership and
semantic margin conflict. Archive-v1, semantic-v1, and semantic-v2 remain
immutable negative evidence. V3 may reuse their public construction evidence,
but it cannot reinterpret an earlier stop or use a language-model loss,
partition, checkpoint, or sealed-test observation.

### Isolated intervention and reused evidence

V3 reuses encoder evidence
`efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`
without rerunning MiniLM. It also replays semantic-v2's exact five-fold role
calibration, including the `tinyworlds-p-semantic-v2` fold authority and
`role-calibration-fold-v1` namespace. Thus the sufficient-context screen, raw
10th-percentile role statistic, cross-conformal p-values, 0.05 rejection rule,
two-sense silhouette gate, and equal anchor/context word vector are identical
to v2. A word's v3 fold and role disposition must equal its v2 result.

The calibrated word scores are not merely recomputed from an equivalent
formula. V3 binds semantic-v2 failure artifact
`23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`,
strictly authenticates its ledger, and requires exact equality for every raw
margin, fold, reference count, conformal value, and cutoff. Canonical real
construction uses NumPy 1.26.4, the numerical environment that produced v2;
the version is part of the v3 config and a mismatch fails before publication.
This closes a preflight finding that NumPy 2.5.1 preserved all decisions but
changed a few serialized values by several billionths.

The only construction intervention is where balance occurs. Nouns and verbs
are independently clustered with deterministic farthest-first, unweighted
spherical k-means. Every assignment is to the word vector's highest-cosine
centroid; canonical SHA-256 ties use the semantic-v3 benchmark namespace.
Archive token mass is neither a word weight nor an assignment constraint. It
is computed after clustering solely for coverage and audit. Consequently the
90--110% cluster-mass bounds and discrete packing repair are absent from the
v3 config rather than silently disabled.

### Frozen semantic gates and later balance

The grid remains exactly eight noun clusters by eight verb clusters, with at
most 100 centroid iterations and five boundary-exclusion/recluster passes. On
each pass, the margin is the cosine to the word's actual nearest centroid
minus the cosine to its second-nearest centroid. Words below `0.03` are
excluded. A successful fixed point still requires at least 32 nouns or 12
verbs in every cluster, all centroid-pair cosines below `0.90`, and at least
40% of non-construction archive token mass after both-role exclusions. These
gates fail closed and will not be changed after inspecting the real v3 audit.

Token and nuisance balance moves to partition story allocation. The semantic
catalog reports unconstrained cluster and 8-by-8 cell masses but does not make
a less-similar word carry a cluster's mass quota. A later v3 partition must
select or weight complete duplicate-story groups within their already-fixed
semantic cells, preserve exact archive bytes and no replacement, and prove
world/control feasibility before training. The exact allocation contract will
be frozen and tested before a real partition is built; it cannot alter word
clusters or use model loss. No semantic-v1/v2 partition loader or checkpoint
is a v3 compatibility path.

### Canonical semantic-v3 result

The canonical construction failure is
`ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146`.
Verbs reached a zero-failure fixed point after two reclusters. Nouns had one
word below the margin after the fifth permitted recluster: `crayon` at
`0.0296120345`. V3 therefore has no catalog or downstream artifact.

The one-word terminal count is not permission to waive the rule. A diagnostic
sixth recluster after removing `crayon` exposed 22 new noun failures and a
25-word noun cluster, below the independent 32-word minimum. Joint coverage
would still be above 48%, and centroid-pair cosines remain below 0.90, so v3
isolates hard deletion plus full reseeding as the remaining instability. Any
fixed-centroid screen, stable-core method, robust objective, changed margin,
or changed pass budget is semantic-v4 and cannot reinterpret v3.

## TinyWorlds-P Semantic-v2 Role-Calibration Boundary

`tinyworlds-p-semantic-v2` is a new construction contract created from the
semantic-v1 role-margin failure. Semantic-v1 and archive-v1 remain immutable
negative evidence. V2 reuses the exact authenticated MiniLM evidence artifact
`efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434`;
it does not rerun the encoder, change a story byte, move a duplicate group out
of the permanent construction slice, or use any partition, checkpoint, model
loss, or sealed-test observation.

### Cross-conformal role evidence

The raw per-word statistic is unchanged from v1. For each exact construction
context, the score is the cosine to the normalized declared-role anchor
centroid minus the cosine to the normalized opposite-role anchor centroid.
The word statistic is the linear 10th percentile across its contexts. V1
compared this value to an absolute zero, which conflated role evidence with a
large encoder/template offset that differed between nouns and verbs.

V2 calibrates that offset without using the word being judged. Each declared
role word is assigned to one of five folds by SHA-256 over the benchmark ID,
the frozen `role-calibration-fold-v1` namespace, role, and word. For a word in
fold `f`, all sufficient-context words of the same declared role outside `f`
form its reference distribution. Its lower-tail conformal value is

`p = (1 + #{reference scores <= word score}) / (reference_count + 1)`.

The word passes when `p > 0.05`. Under exchangeability of same-role word
scores, the added one gives the usual finite-sample conformal calibration;
inclusive comparison handles ties conservatively, and holding out the complete
word fold prevents a word's contexts from setting its own cutoff.
Calibration is unweighted at word level because the screened unit is a word;
token mass remains reserved for capacity-constrained clustering. No fold is
removed wholesale, so the procedure uses cross-fitting rather than spending
20% of the vocabulary as a permanent panel. The artifact persists every
fold, reference count, empirical cutoff, raw margin, and p-value, and the
strict loader recomputes all of them from the word ledger.

All other semantic screens remain fixed. At least 32 contexts are required;
the deterministic two-means silhouette maximum is 0.20; and the semantic word
vector remains the normalized equal-weight combination of the declared-role
anchor centroid and archive-context centroid. V2 therefore isolates the role
calibration intervention rather than simultaneously changing the semantic
representation.

### Capacity feasibility and unchanged grid gates

V1 never reached mass-constrained clustering and consequently did not expose
a discrete greedy-packing corner: continuous remaining-mass checks can leave
two underweight clusters for one indivisible final word. V2 retains
descending-token-mass assignment and the exact 90--110% mass bounds, but, only
at such a dead end, deterministically considers moving one already assigned
word. It chooses the feasible current assignment plus prior-word move with
the best fixed-centroid cosine objective, using the v2 hash namespace for the
final tie. If no single move restores feasibility, construction still fails.
This is an algorithmic feasibility repair, not a capacity relaxation.

The cluster count (eight per role), farthest-first seeds, 100-iteration
budget, five boundary exclusion/recluster passes, assigned-centroid margin
minimum 0.03, per-cluster word-count minima, centroid-pair cosine maximum
0.90, and joint retained-token minimum 40% are unchanged. Success and failure
artifacts use v2-only formats and paths under
`data/tinyworlds-p-semantic/catalog/v2/`; there is no v1 compatibility alias.

### Real semantic-v2 construction result

The calibrated lower-tail rule excluded 51/1,066 nouns and 19/394 verbs; the
unchanged sense screen excluded 37 more nouns and 10 more verbs. This left 978
noun and 365 verb candidates, demonstrating that the v1 mass rejection was a
calibration problem rather than evidence that almost every noun had the wrong
role.

The fixed grid nevertheless failed later. Across the initial clustering and
five permitted exclusion/recluster passes, 655 distinct nouns and 225
distinct verbs fell below the assigned-cluster margin. On the terminal pass,
47 of 370 nouns and 18 of 158 verbs still failed, so neither role converged.
The failure artifact is
`23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25`.
As a diagnostic only, removing those terminal failures too would leave 323
nouns and 140 verbs whose intersection covers 98,322,186 tokens, or 10.945%
of non-construction mass; that is far below the unchanged 40% gate. V2 stops
without a catalog, partition, training run, or sealed-test opening. A future
change to the cluster objective or boundary semantics requires semantic-v3.

## TinyWorlds-P Semantic-v1 Benchmark Boundary

`tinyworlds-p-semantic-v1` is a new benchmark contract, not a reinterpretation
or continuation of `tinyworlds-p-archive-v1`. It reuses the pinned archive and
GPT-2 tokenizer as source material while introducing a separately authenticated
semantic-construction authority. Its artifacts live under
`data/tinyworlds-p-semantic/` and
`checkpoints/tinyworlds-p-semantic-v1/`. Loaders accept only the strict semantic
archive, tokenizer, encoder, config, catalog, partition, sample-report, and
training identities; no archive-v1 partition, checkpoint, resume state, or
compatibility alias is accepted.

The semantic encoder is
`sentence-transformers/all-MiniLM-L6-v2` revision
`b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`. Every recursively selected model
and tokenizer file is hashed. Inference is deterministic float32 attention-mask
mean pooling followed by L2 normalization and produces 384-dimensional
construction evidence. Encoder text is never a model input. The eventual
GPT-Neo base still sees only exact archive story bytes tokenized by the pinned
GPT-2 BPE.

### Construction slice and evidence cache

Normalized duplicate groups whose namespaced SHA-256 rank has residue zero
modulo 20 form the permanent semantic-construction slice. The rule is applied
to the group, so duplicate occurrences cannot straddle construction and model
data. Construction groups never enter base, world, or control splits.

For each exact normalized whole-word noun and verb occurrence, construction
contexts are ordered by a canonical context hash. At most the first 128 are
retained and at least 32 are required. Sentences beyond the encoder limit are
cropped to a target-centered 128-wordpiece window. Each word also receives the
three frozen role anchors recorded in the versioned construction config. The
evidence publication contains the complete encoder identity, construction
contexts, embeddings, role-pair token masses, and hashes. It is content
addressed separately from catalog clustering, allowing a later benchmark
version or threshold study to reuse immutable encoder observations without
rerunning MiniLM.

### Semantic screens and fixed clusters

Nouns and verbs are screened independently. For every word, the target-role
anchor centroid and opposite-role anchor centroid are L2 normalized. Each
normalized archive context receives the cosine difference
`context·target - context·opposite`; its 10th percentile must be strictly
positive. Deterministic farthest-first spherical two-means over contexts must
have mean cosine silhouette at most 0.20. A surviving word vector is the
normalized equal-weight average of its normalized target-anchor centroid and
normalized archive-context centroid.

Each role must form exactly eight capacity-constrained spherical clusters.
Initialization is farthest-first; assignments process descending non-
construction token mass and use benchmark-namespaced hashes for all ties.
Every cluster must remain between 90% and 110% of its role's mean token mass.
Centroids iterate at most 100 times. Words with assigned-centroid versus best-
alternative cosine margin below 0.03 are excluded and the complete role is
reclustered for at most five exclusion passes. The result then requires at
least 32 nouns or 12 verbs per cluster, every centroid-pair cosine below 0.90,
and at least 40% of non-construction archive token mass after joint role
exclusions. These are fail-closed constraints: cluster count and bounds are
never relaxed, and there is no 8-to-6 or 8-to-10 regrid.

Successful catalogs and failed-grid audits are both content addressed and
never overwrite earlier evidence. A successful audit lists all retained and
excluded words, cluster metrics, nearest/median/boundary words, exact archive
contexts, PCA geometry, cell masses, and optional parent differences in
Markdown and self-contained HTML. A failure audit records every pre-clustering
word and metric, the exact failed invariant, candidate PCA, and why cluster or
cell views do not exist. Cluster names are not generated and human review is
not an input to construction.

The real semantic-v1 evidence passed source and encoder authentication but
stopped at the frozen word screen: only six nouns survived the role-margin and
silhouette rules, fewer than the required eight initial centroids. Therefore
semantic-v1 has an immutable failure audit but no catalog, partition, model, or
sealed-test result. Changing an anchor, metric, threshold, vocabulary, or
cluster count requires `tinyworlds-p-semantic-v2`.

### Conditional partition and paired-control semantics

When a future fixed semantic catalog passes, all clean non-construction
duplicate groups are replayed from exact archive bytes. Physical noun and verb
clusters define the same A/B/C/D 2-by-2 corner and row/column-unrelated E cell.
Topology selection may use semantic cluster quality, active-token balance,
nuisance balance, component visibility, and control feasibility only; it
cannot inspect model loss. Worlds use 80/10/10 and the held-in complement uses
96/2/2. Source, feature, adjective, and length balancing; global no-replacement
controls; and the sealed-test boundary follow the archive-native semantics.

After allocation, each world group is paired one-to-one with one control group
inside the appropriate arm. Pairing first respects nuisance strata, then
minimizes token-length/mass differences with canonical hashes as final ties.
The pairing is persisted and strictly revalidated; it does not change the
aggregate world/control NLL. Before any training, a content-addressed report
must sample held-in validation, all five semantic worlds, and both arms of all
five validation controls, including complete cluster inventories and exact
archive provenance. Report code cannot read a test index.

### Conditional empirical-null calibration

Evaluation persists canonical per-duplicate-group loss sums and active-token
counts. For each world's fixed group pairs, 10,000 SHA-seeded paired bootstrap
replicates produce a 95% interval and 10,000 within-pair label swaps produce a
one-sided empirical-null probability. The mean statistic is the unweighted
mean of five token-weighted world gaps with independently seeded stratified
replicates. A validation epoch passes only when the observed mean gap is at
least `ln(1.05) = 0.048790164` nats/token, its bootstrap lower bound is
positive, and its one-sided placebo probability is at most 0.01; every world
must have positive observed and bootstrap-lower gaps; and all five world
placebo tests must reject under Holm step-down correction at familywise 0.05.
There is no upper-gap rejection.

The model, optimizer, seed zero, two-epoch calibration, five-epoch maximum,
resume coordinates, 12 GiB allocator limit, and held-in NLL/improvement gates
remain the archive-native GPT-Neo contracts, but update counts and ETAs derive
from the semantic partition's retained mass. A failed semantic gap ends the
run without regridding. A pass resumes through epoch five; among epochs that
satisfy the same semantic-gap gate, the lowest held-in NLL wins with earlier
epoch as the exact-tie rule. The selected checkpoint must also have held-in NLL
at most 2.0. Only then is sealed test opened once and reported without changing
selection.

## TinyWorlds-P Archive-v1 Historical Boundary

TinyWorlds-P is the benchmark contract `tinyworlds-p-archive-v1`. Its sole story
universe is the set of released entities in the pinned
`TinyStories_all_data.tar.gz` archive. Base training, held-out worlds, matched
controls, validation, and sealed test are derived only from eligible archive
entities. The original `TinyStories-train.txt`,
`TinyStories-valid.txt`, and GPT-4-only text aggregates are not inputs,
reference corpora, coverage targets, or split authorities for TinyWorlds-P.
They must not be read by partition construction or base training.

The canonical source is the pinned 1,608,001,638-byte
`TinyStories_all_data.tar.gz` from dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, SHA-256
`26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
The tokenizer is the existing hashed 50,257-token GPT-2 BPE artifact. These
identities, normalization, all research choices, assignments, controls, and
tree digests are artifact contracts rather than runner defaults. Released
prompt metadata determines partitions and audits but is never included in
model input. External generation, story embeddings, learned routers, semantic
clustering, semantic near-deduplication, replay, consolidation, LoRA, and VAMP
episodes are outside the base-publication milestone.

### Archive identity and duplicate semantics

Each archive entity is bound to its tar member, member-local index, and a hash
of its complete released JSON record. Duplicate-story identity is SHA-256 over
the entity's UTF-8 `story` after NFKC, case folding, canonical straight
single/double quotation marks, Unicode whitespace collapse, and edge trim.
Normalization exists only for duplicate identity. Training and original-text
shards retain the exact UTF-8 bytes of each accepted archive `story`; the
tokenizer appends the corresponding EOS token after each occurrence.

The archive is streamed into bounded sorted runs. It is never materialized in
memory and no mutable database is part of the build. Normalization and
tokenization use 16 physical processes in the production preset. Published
order is canonical, so worker completion order and temporary run size cannot
change output bytes.

Every normalized duplicate group is one indivisible assignment unit. All raw
archive record occurrences and their multiplicity remain attached to that
assignment. Prompt roles are accepted only when explicit released noun, verb,
and adjective labels mechanically and uniquely identify the three released
words. Every duplicate record must classify and all complete normalized
recipes must agree; otherwise the group is excluded as unclassifiable or
conflicting.

There is no corpus-to-archive join, unmatched-corpus category, hash-match
coverage gate, or combined corpus-coverage gate. Eligibility is determined
solely from the pinned archive record and its released metadata. Every base,
world, control, validation, and test assignment is a subset of eligible archive
groups. The audit retains archive record/group counts, duplicate multiplicity,
token mass, classification exclusions, and every accepted provenance record.

### Buckets and five-cell topology

Ingredient surfaces use NFKC, case folding, and trim. Noun, verb, and adjective
vocabularies are balanced independently. Words are processed largest eligible
token mass first and assigned to the currently lightest of eight buckets.
Equal word masses and equal bucket loads use SHA-256 namespaces derived from
the benchmark version, complete source/tokenizer identities, public seed zero,
role namespace, word, and bucket. Changing a verb vocabulary therefore cannot
perturb noun ties. Adjectives affect stratification only.

Cell statistics retain active tokens, groups, and token-weighted source,
feature-signature, adjective-bucket, and length-bin marginals. The selected
topology is a 2x2 noun/verb corner plus a row- and column-unrelated cell. All
valid physical bucket tuples are scored by exact rational token imbalance and
nuisance-distribution imbalance; the minimum combined score wins, with a
canonical namespaced tuple hash as final tie break. Physical buckets are then
relabelled:

- A = N0 x V0
- B = N1 x V0
- C = N1 x V1
- D = N0 x V1
- E = N2 x V2

Every selected cell must lie within 10% of the five-cell median active-token
count. Every actual noun and verb used in a selected-world group must occur in
at least 64 eligible groups outside all five worlds. Counts refer to duplicate
groups, not raw occurrences.

### Split and control semantics

Duplicate groups are allocated largest-token-first with deterministic hash
ties. Each candidate split is scored against its active-token target and four
token-weighted marginals: released source model, canonical feature signature,
adjective bucket, and canonical-story token-length bin (`<=64`, `65-128`,
`129-192`, `>192`). World cells use 80/10/10 train/validation/sealed-test
weights. The held-in complement uses 96/2/2. Test assignments are made in the
same immutable pass as every other split and are never consulted by grid,
epoch, or checkpoint selection.

Each world validation and test split owns a no-replacement control drawn from
the corresponding held-in split. Half its groups come from the same noun row
and other verb columns, and half from the same verb column and other noun rows.
Joint nuisance strata seed a deterministic proportional preference; exact
joint-stratum reproduction is not required because the contract constrains
marginals. Cross-stratum swaps first satisfy the source, feature, adjective,
and length-bin marginal bounds, then within-stratum swaps match active token
mass and mean length without changing those marginals. Controls fail rather
than relax when active token mass differs by more than 0.25%, source or feature
prevalence by more than two percentage points, adjective-bucket or length-bin
prevalence by more than three points, or mean canonical length by more than 5%.
A held-in group cannot serve two controls in the same evaluation split.

### Partition persistence and loading

Partition publications live at
`data/tinyworlds-p-archive/v1/<partition-sha256>/`. They contain source and
normalization identities, bucket word lists, topology, one assignment per
duplicate group, provenance, controls, audit, base/world/control manifests,
original-byte shards, little-endian uint16 token shards, and document indexes.
Both shard kinds roll only between stories near 32 MiB. Indexes bind raw
archive record IDs, tar members and member-local indexes, content hashes,
recipe and bucket evidence, partition/split, shard offsets, and token counts.

`tree.json` lists every other file with exact size and SHA-256. Strict loading
rejects unknown/missing paths, symlinks, noncanonical JSON, checksum changes,
assignment/count mismatches, control reuse, an unknown control group, duplicate
assignment identities, malformed topology, and any held-out physical cell in
a base assignment. Rebuilding identical inputs produces the same partition
identity and every published byte even when process completion, worker count,
or external-sort run size changes.

`iter_partition_batches` accepts canonical selectors such as `base/train`,
`world/A/validation`, or `control/A/test`. It memory maps uint16 shards, never
joins two stories into a causal window, pads only tensor rows, and groups a
fixed number of documents into source blocks. Epoch-specific SHA-256 ranks
shuffle those blocks while preserving document and window order within each
block. The block coordinate is the resume boundary.

### Scratch base training and selection

The production base initializes every float32 parameter from seed zero. It is
the existing tied-embedding GPT-Neo shape: vocabulary 50,257; positions 2,048;
width 256; MLP 1,024; eight alternating global/local layers; 16 heads; local
window 256; no dropout. Old TinyStories checkpoints provide engineering
continuity only and are not initialization or baselines.

Training uses 256-token windows, microbatches of 32, and eight-microbatch
gradient accumulation. Each microbatch differentiates summed masked NLL.
Gradient PyTrees are summed and divided once by the total active token count,
so padded or short microbatches cannot receive equal weight. Global-norm clip
1.0 precedes AdamW with maximum LR `5e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
and weight decay 0.1. The schedule linearly warms through the first ceiling of
1% of planned updates, reaches the maximum at the warmup boundary, then
cosine-decays to exactly `5e-5` on the final planned update.

Resume state includes model, optimizer, RNG, update, next epoch, shuffled
block, microbatch, and schedule position. States are immutable at every 1,000
updates and epoch. Checkpoints occur only after a complete optimizer update;
an interrupted run resumes from the prior complete state and is bit-identical
to uninterrupted execution. Progress and loss JSONL is flushed continuously.

An 8x8 partition trains for two calibration epochs. At both epoch boundaries,
validation measures held-in NLL and
`gap(world) = NLL(world) - NLL(matched control)`. Epoch two passes when mean
gap is 0.08-0.30, every gap is at least 0.05, held-in NLL is at most 2.2,
held-in NLL improved by at least 0.02 since epoch one, partition/leakage gates
passed, and JAX reports an allocator peak no greater than 12 GiB. Too-small or
per-world-low gaps trigger exactly one fresh 6x6 build with held-in 94/3/3
train/validation/test weights. That leaves approximately 12% shared-row or
shared-column held-in control capacity for 10% world demand while preserving
global no-replacement. Mean gap above 0.30 triggers exactly one fresh 10x10
build with held-in 96/2/2 weights. Held-in quality, improvement, or memory
failure never triggers regridding. A failing second grid ends the milestone
without final training.

A passing two-epoch state resumes through epoch five. Among epochs two through
five that still pass gap gates, lowest held-in validation NLL wins and an exact
tie selects the earlier epoch. Publication additionally requires held-in
validation NLL at most 2.0. Only after selection are held-in, five-world, and
five-control test splits opened once. Test hypothesis failures are reported
results and cannot change the selected partition or checkpoint.

The base publication lives at
`checkpoints/tinyworlds-p-archive-v1/<training-sha256>/` and contains the strict selected
base checkpoint, copied hashed tokenizer, every complete resume state, trace,
validation records, sealed test, fixed-prompt samples, identities, strict tree,
and report. The report includes coverage, calibration, learning curves,
world/control gaps, selected epoch, runtime, active-token throughput, and the
measured allocator peak.

## TinyWorlds-v2 Benchmark Boundary

TinyWorlds-v2 is a new, parallel benchmark contract named
`tinyworlds-v2-gpt`. The symbolic world ledger is authoritative for entity and
task lineage, controlled assignments, truth, support provenance, candidate
answers, and scoring. A pinned external language model expresses ledger
constraints as ordinary, variable-length TinyStories-style narratives; the
ledger never serves as a prose grammar.

V2 retains the plain-JAX GPT-Neo base, immutable pathwise-LoRA VAMP graph,
candidate scorer, hard and soft addressing, Hopfield and EBT evaluators,
parent counterfactuals, forgetting/transfer decomposition, and report/artifact
infrastructure. It does not use the v1 template registry, generic fact/rule
statements, exact-token text fitting, padding fragments, artificial cue blocks,
formal relation questions, or visible internal identifiers. Text padding
exists only in token arrays under explicit masks.

TinyWorlds-v1 and its promoted stopped calibration are immutable negative
scientific evidence. V2 lives in a separate package and artifact namespace; no
v2 result changes, resumes, repairs, or reinterprets a v1 artifact. Two-hop,
cross-branch, and consolidation questions remain deferred until natural direct,
convention, and contextual-revision probes demonstrate learnability.

## TinyWorlds-v2 External Generation and Reproducibility

External generation is an artifact-production boundary, never an implicit data
loader. A generation request is content-addressed over the exact model and
provider route, system and user prompts, JSON schema, settings, and body. The
complete raw response is immutable and records its request ID, response bytes,
returned model/provider identity, relevant headers, usage, and billed cost.
Accepted and rejected interpretations are derived artifacts, so validators may
be rerun without another model call. A request is never silently regenerated
to replace a schema or semantic failure.

The external models are commissioned authors of constrained synthetic stories;
the benchmark is not trying to reproduce their general behavior in a student
model. OpenRouter's `enforce_distillable_text` routing switch is therefore not a
scientific quality or route-eligibility gate. Because accepted stories may
later become training data, corpus eligibility is an explicit experiment
decision rather than an inference from a router metadata label. For the active
two-author run, Qwen 3.5 35B-A3B and GPT-5.4 Mini are both eligible authors;
diagnostic preview records still do not flow into a training corpus.

The active Phase 1 comparison is direct and paired: both authors receive all
200 neutral briefs. There is no cheap-route screen, finalist expansion, or
external verifier pass. This makes the paid boundary exactly 400 author
requests and avoids confounding author quality with a third model's ratings.
Automated evidence instead comes from the frozen TinyStories vocabulary/token
profile, TinyStories-8M NLL, surface statistics, and deterministic brief checks;
the 100-reference/100-generated blinded audit supplies the independent semantic
and stylistic judgment, with generated controls split 50/50 by author.

The V4 generator response contains only `story`. Required-word occurrences are
whole-word spans with exact character offsets derived locally, and requested
features, forbidden forms, length, dialogue, and token-form evidence are also
computed from the story text. Model-authored evidence or quotes are neither
requested nor trusted. V1--V3 and the seven-route table remain frozen solely to
replay their historical artifacts; the active V4 run has a separate artifact
version, destination, and raw cache.

Prompt changes are separately versioned request semantics. V5 is the frozen
length-only contract (`Write 130 to 170 words.`); V6 adds a declared
reference-shape profile while retaining the one-field response and locally
derived evidence boundary. The first V6 run is a 20-brief development review,
not a reduced Phase 1 gate. Because the user requested only 20 new outputs per
model, this first run compares bundled V6 against V4 and does not claim to
attribute changes to length versus cadence; V5 was defined but not purchased.
It binds each new story to the same brief's cached
V4 control and matched reference, uses a dedicated `$1` runtime ledger and raw
cache, and publishes all pairs in one authenticated review page. Once used for
prompt choice, those briefs are tuning data and cannot be called a clean
holdout. Prompt compliance is measured, never assumed: V6's prose instruction
to use single newlines did not prevent either route from emitting blank-line
paragraphs, so a future serialization constraint must receive its own request
contract rather than being silently normalized after generation.

V7 is a separate development-only prompt-shape contract; it preserves the
exact V4--V6 request bytes and does not alter or satisfy a Phase 1 gate. It
removes the redundant `short` and `3- to 4-year-old` compression cues from the
system message while retaining young-child safety, the released-instruction
boundary, ordinary-prose output, and the strict story-only JSON response. Its
final user-message block requires unchanged required-word forms, actual
double-quoted speech when dialogue is requested, one story-field text block
without newlines, 18--20 complete sentences, at least six connected events, a
soft 155--190-word target, and a little natural simple repetition. A dedicated
brief-ID hash namespace assigns openings independently at 60% `Once upon a
time`, 20% `One day`, and 20% another simple opening. Versioning this bundle as
V7 isolates the hypothesis that instruction placement and concrete narrative
shape—not merely a word-count request—drive length and TinyStories alignment.
V8 tests the complementary bare-envelope condition without changing or
reinterpreting V7.

V8 is the bare released-prompt ablation. Its chat envelope contains exactly one
user message: the archived TinyStories instruction, two newline characters,
and `Possible story:`. It contains no system role, repeated ingredient list,
JSON instruction, response schema, or added prose constraint. Model/provider
pinning, deterministic seed, reasoning disablement, output ceiling,
data-collection denial, and fallback denial remain request transport controls;
they do not add natural-language tokens. The complete assistant content is
preserved as the story without JSON parsing, fence removal, prefix stripping,
or whitespace normalization. A nonempty UTF-8 reply is structurally valid, and
the existing local story validator derives all spans, visible features, safety
flags, and acceptance evidence from that exact text.

Plain-prompt experiments use the V3 train-decontaminated validation profile
directly, never the historical 10,000-record comparator or the small matched
archive set. Each prior control remains immutable and is loaded with its saved
NLL rather than rescored. V9 is the one-factor length intervention: it inserts
exactly `Aim for about 130 to 150 words.` between the archived prompt and
`Possible story:`. V8 and V9 otherwise share the same one-message envelope,
provider seed, route controls, and plain-response interpretation. Any later
change is another request contract and must name the single factor it adds; it
must not silently restore a system prompt, JSON, or the V7 narrative checklist.

Local derivation does not make every derived field a hard validator. Exact
schema, whole-word ingredients, safety/forbidden forms, length, and structurally
quoted dialogue are mechanically decidable and may reject a sample. Moral,
conflict, foreshadowing, twist, and ending valence are semantic narrative
labels: deterministic heuristics may report them, but they cannot prove
realization or reject prose. Their authoritative assessment belongs to the
blinded human audit. Changing this boundary requires a validator and artifact
version change; cached raw responses may be reinterpreted only into a new
artifact, never silently in place.

## TinyWorlds-v2 Author Learnability Diagnostics

Distribution resemblance and downstream learnability are different claims.
An author cannot be rejected as LoRA-incompatible merely because frozen-base
NLL or surface-distribution gates fail. A learnability comparison must hold the
symbolic facts, exact leading evidence, document count, adapter initialization,
RNG state, optimizer schedule, and probes fixed, and must include a
decontaminated in-distribution TinyStories control. If that control does not
learn the semantic task, external-author differences are inconclusive rather
than evidence of an author penalty.

Probe difficulty is interpreted as a ladder. Exact-clause completion tests
whether a seen literal continuation was stored. Held-out paraphrases test
whether the corresponding entity binding is queryable independently of its
training wording. One-hop probes test whether one stored entity-to-value fact
can be composed with one stored value-to-conclusion rule when the named
entity-to-conclusion sentence was never shown. Near-zero training NLL or
perfect exact-clause completion is not called reasoning when paraphrase or
one-hop transfer stays at chance. Author suitability is assessed only at the
highest rung passed by a matched in-distribution control.

Route identity distinguishes billable semantics from catalog provenance. The
versioned semantic lock hashes the local route name, requested model alias,
expected dated canonical model, provider selector and returned-provider name,
quantization, and input/output prices. It deliberately excludes the digest of
the complete public catalog responses. Those exact response bytes and their
digest remain manifested provenance, but harmless catalog byte changes neither
invalidate an immutable response nor manufacture a new request identity. Any
change to a semantic lock field changes the lock/request hash or fails closed.

The initial production backend is direct OpenRouter HTTP behind an optional
generation dependency. Both the model and one concrete provider endpoint are
locked before submission; automatic model/provider fallback, response healing,
plugins, and unrecorded routing are forbidden. Catalog or returned-identity
drift fails closed. This is intentionally a narrow backend boundary, not a
general provider abstraction. Direct OpenAI Batch may be costed for comparison
but is not a second Phase 1 execution path.

OpenRouter serving provenance is taken from the opt-in router-metadata record,
not from a nonstandard top-level response field. Exactly one endpoint must be
marked selected and its provider must match the catalog lock. Token usage and
billing are independent observations: a failed generation can have a reported
cost without complete token counts. When completion metadata is incomplete,
the client polls generation stats by generation ID under a separate bounded
retry policy. Every stats response and transport failure is append-only and
preserved byte-for-byte; resume continues the stats lookup without repeating
the completion POST. The final record must agree on provider, model, identity,
and cost before the attempt is accepted.

The original catalog lock is revalidated from the public model and endpoint
catalogs immediately before every paid generator-route batch and every
50-request verifier batch. Revalidation compares semantic fields rather than
raw catalog bytes. Drift halts the shared ledger before the batch's first POST
and becomes an explicit `catalog_route_drift` result; a live alias is never
trusted without rechecking its expected dated canonical model. Each raw
attempt separately records the fresh catalog digest that authorized its
submission, preserving exact provenance across harmless catalog-byte changes.

Behavior-affecting HTTP conventions are part of a versioned transport protocol
included in the canonical request hash. The current completion protocol opts
into router metadata and explicitly disables OpenRouter response caching with
`X-OpenRouter-Metadata: enabled` and `X-OpenRouter-Cache: false`. A change to
either behavior requires a protocol/version bump and a new request identity.
Authorization bytes are deliberately excluded.

Optional model reasoning is also an explicit request-contract field. Short
story routes known to produce separately billed hidden reasoning are sent
`reasoning.effort=none` when their pinned endpoint supports it; routes without
that capability omit the field. Hidden reasoning usage is billable output even
when it is absent from the visible story, so it must either be disabled or be
included in the reservation bound. Changing this policy changes the canonical
request body and requires a new request-contract version.

The completion secret enters only from `OPENROUTER_API_KEY` or the local
`openrouter-tinyworlds-key.txt` fallback, whose mode must be `0600`. It is an
inference credential and is never treated as proof of workspace BYOK state.
An optional, distinct `OPENROUTER_MANAGEMENT_API_KEY` may perform only the
authenticated `/api/v1/byok` preflight; it is never used for completions. The
alternative repository-root
`openrouter-tinyworlds-no-byok-attestation.json` is a canonical manual claim
created only after explicit user confirmation, binds the exact statement “I
attest that this OpenRouter workspace has zero configured BYOK keys.” and a UTC
interval, and expires within 24 hours. Only sanitized preflight evidence (proof
source, response/attestation digest, count when available, and times) is
manifested; raw management responses and BYOK key metadata are not persisted.
All secret values remain absent from logs, hashes, manifests, caches, reports,
exceptions, and copied artifacts.

BYOK is checked twice because it is an external-cost escape hatch. Paid work is
disabled unless the management preflight proves zero configured keys or a valid
manual attestation is present. Every successful completion must additionally
carry explicit `is_byok=false` evidence; a generation-stats fallback must do the
same before its cost or provider is trusted. A positive or ambiguous BYOK
runtime observation is charged conservatively to the request bound and halts
further POSTs.

Authorization is attached to the durable paid boundary, not inferred from the
latest run-level preflight. Every write-ahead reservation embeds one canonical,
sanitized zero-BYOK authorization record and its digest. This lets replay prove
that historical POSTs were authorized even if a later resume has no current
proof or records a new failed preflight. The ledger first reconciles all prior
requests, responses, and reservations, then evaluates whether new work is
authorized. A run-level preflight record describes only the current invocation;
it never invalidates or retroactively authorizes earlier paid work.

Reproducibility does not mean calling a remote model twice and expecting the
same text. It means pinned source/model/route/prompt/schema/settings identities,
content-addressed requests, complete raw-response preservation, immutable
accepted datasets, and byte-identical reconstruction of derived files from the
cache. Loading and reconstruction never call a network. Changing any pinned
input creates a new dataset version; resume submits only absent requests and
never overwrites a raw response.

Phase 1 replay is a stricter zero-network boundary than ordinary artifact
loading. It starts from persisted briefs, semantic route locks, raw cache and
cost-journal evidence, plus already persisted tokenizer/NLL measurements. A
network-forbidden client reconstructs canonical jobs and discovers attempts
from the raw cache rather than trusting committed generator or verifier result
streams. It reparses immutable HTTP bytes for every attempted request,
including interrupted terminal attempts that never reached those streams,
reruns source joins, deterministic validation and quality selection, and
rebuilds generator, verifier, execution-manifest, and audit files. Provider,
BYOK, and response-contract terminal causes must agree with the underlying raw
attempt status. Every rebuilt derived byte is compared to the source tree
except the replay tree's own root manifest. Replay never reads the source
corpora, accelerator, tokenizer/checkpoint, API credentials, or network. Every
public completed-artifact validation and human-approval entry point invokes
this full semantic and replay gate before accepting a tree or overlay; an
internally consistent manifest or approval digest cannot bless corrupted base
evidence.

Remote bytes are screened for the active completion and management secrets
before hashing, caching, error formatting, or artifact copying. A reflected
secret becomes a canonical secret-free terminal marker so replay can prove why
the run stopped without retaining the reflected material or retrying the
request. Final publication validates the temporary tree, uses an atomic
no-replace rename, and validates the promoted tree again. A destination created
by another process during promotion is preserved rather than overwritten.

Billable work is guarded by a serialized preflight computed from exact request
bodies, current route prices, expected output lengths, verification traffic,
retry allowance, and worst-case in-flight reservations. Phase 1 has a `$15`
inclusive hard ceiling. Estimated and provider-billed costs are distinct
provenance fields. Normal tests use fixtures and fake transports and perform no
downloads or remote generation.

Small route previews use separate, explicitly authorized cumulative caps and
are labeled `diagnostic_only`. If a preview is interrupted after paid requests,
its exact provider-reported spend is debited before calculating a successor
run's residual cap. A preview may inform a human decision but cannot select a
production route, authorize a full funnel, or advance a phase gate.
Missing generation statistics count as exhausted only after the complete fixed
lookup schedule. A complete raw cache can be republished without loading a
credential or refreshing its short-lived BYOK attestation, promotion first
reserves the destination with no-replace creation, and bundle validation
rederives actual, unknown, per-route, runtime, and cumulative exposure directly
from the immutable response and cost journals.

The same ceiling is enforced during execution by one thread-safe ledger shared
by all eight workers while one nonblocking filesystem lease excludes a second
paid process from the complete raw-cache lifecycle. Before each completion POST
the ledger derives an upper bound from exact UTF-8 request bytes, maximum output
tokens, and locked maximum prices, then atomically persists and fsyncs an
immutable write-ahead reservation. The corresponding cache entry stores the
complete canonical route lock rather than reconstructing historical route
semantics from a current catalog or request body. Immediately before transport,
the ledger verifies that the reservation remains postable. If another worker
has already halted the run, it appends an explicit `cancelled_before_post`
journal state; cancellation is neither a submission nor a charge. The HTTP
attempt is durably cached before an uncancelled reservation is settled.
Provider-reported actual cost replaces the bound;
an HTTP response with unknown cost or a transport failure that may have occurred
after POST consumes the full bound as a separately labeled conservative unknown
charge and halts further POSTs.

OpenRouter's maximum-price fields are per-million-token JSON numbers. Exact
decimal catalog prices are encoded conservatively upward, and reservation
arithmetic uses the actual transmitted ceiling. Denying a new reservation does
not revoke earlier reservations already counted under the cap; a later
ambiguous-billing or provider-contract failure on an authorized POST supersedes
that benign admission-denial reason.

Restart reconciles the journal against canonical requests, their persisted
route locks and BYOK authorizations, and immutable raw attempts from both
current and historical catalog locks, using each frozen request's own prices.
A response already present may safely settle an orphaned reservation. A
reservation with no recoverable response is charged at its bound and stops as
`orphaned_cost_reservation`; a provider-billed settlement missing its raw
response stops as `billed_attempt_response_missing`. Neither condition is
reposted. Cancelled reservations require no response and are excluded from
submitted, interrupted, actual, and unknown-cost counts. Stopped artifacts
attribute every actual and conservative charge exactly to its persisted
generator or verifier route rather than assigning all historical cost to the
current stage. Exactly `$15` is permitted; a reservation above it is rejected
before transport.

Phase 1 authenticates and streams the pinned release, then ranks unique story
content by namespaced hashes. Content identity uses NFKC normalization,
case-folding, whitespace collapse, and SHA-256, and is disjoint within and
across the prompt, reference, paired, and validation cohorts; source-location
and raw-content identities remain separately available for provenance. Surface
measurements run in deterministic 16-process shards and merge by stable record
identity, while TinyStories-8M NLL is computed once on the GPU and persisted for
replay.

Phase 1 compares three reference views for different scientific purposes. The
global 20,000-story profile defines vocabulary coverage. The 200 genuine
stories paired to the generation briefs define token, NLL, length, paragraph,
dialogue, repetition, token-form, and realized-feature comparisons; a stable
split of that paired profile calibrates token-unigram divergence. The separate
10,000-record released prompt-metadata cohort defines requested-feature rates,
so the unannotated validation half cannot dilute the released TinyStories
recipe. Full-corpus cost projections mean 4,000 accepted stories: 500 accepted
stories for each of the eight planned tasks. The economy envelope chooses the
lowest projected cost among fully
qualified routes, quality-ceiling chooses the lowest alignment distance, and
balanced gives equal weight to min-max-normalized projected cost and alignment.
If no route fully qualifies, all three are explicitly unavailable.

An official split label is provenance, not proof of train/evaluation
disjointness. Every TinyStories cohort used as a distributional comparator must
be checked against the authenticated original training file before it is called
a clean holdout. Comparator identity is Unicode NFKC normalization followed by
case-folding and whitespace collapse; SHA-256 is a prefilter, and equality of
the full normalized texts confirms a collision-safe match. Records overlapping
training are excluded from distribution profiles. Small paired archive stories
may remain visible as human review examples, but they are not authoritative
distribution evidence unless they pass the same audit.

Paid outputs and their original interpretations are immutable. If a comparator
audit later changes which references are eligible, publish a new derived
reevaluation that authenticates the source artifact, reuses its exact generated
stories and measurements, embeds the filtered comparator and audit, and makes
zero model calls. Never overwrite the paid artifact or silently relabel a
contaminated report. A composite alignment distance is descriptive ranking
evidence, not a gate override: improvements in length or serialization cannot
compensate for failed acceptance, language-distribution, or NLL gates when
selecting a production author prompt.

The blinded audit packet and key belong to the immutable manifested Phase 1
tree. Browser decisions, the explicit approval request, and the derived final
approval are the only permitted post-manifest overlays. Each overlay is strict,
canonical JSON bound to the exact audit and decision digests; passing automated
or human thresholds never creates approval implicitly. Audit construction
solves the complete deterministic balanced assignment over distinct pair IDs,
keeping both fixed authors represented while making only automated-qualified
routes selectable after human scoring. If the requested 100 generated controls
cannot be allocated at the
fixed per-route quotas, the run publishes
`audit_insufficient_accepted_samples` and exact feasibility evidence rather
than reducing the audit, using duplicate pairs, or applying a greedy fallback.

## TinyWorlds-v2 Validation and Human Gates

The generation model is not its own sole judge. The Phase 1 neutral-author
bakeoff deliberately has no model verifier: deterministic distribution checks
and the blinded human audit are independent of both authors. For later
world-conditioned generation, semantic verification is a separate pinned
request that classifies required-fact entailment, contradictions, new
controlled claims, exact evidence, and answer leakage. TinyStories quality is
assessed separately with a source-blinded rubric over
preschool vocabulary, sentence simplicity, grammar, plot coherence,
repetition, and meta-language. Deterministic validators enforce exact evidence
substrings, ledger assignments, forbidden alternatives and identifiers,
candidate balance, split/request isolation, token boundaries, EOS/masks, and
copying limits.

Distributional validation compares real GPT-4 TinyStories, neutral stories
from the selected generator, and world-conditioned stories from that same
generator. The neutral-to-world shift is the primary measure of whether ledger
constraints damage language quality. Natural evaluation prefixes are nested
64/128/192-token suffixes of one unfinished story and end at one shared answer
boundary; candidate NLL is normalized only over active answer tokens, with
unnormalized total NLL retained as a sensitivity measure.

Phase 1 and the Phase 3 sample corpus end at artifact-bound human gates. An
approval names the exact audit digest; code cannot infer approval from passing
automated metrics or from an unbound acknowledgement. No world generation may
start before the Phase 1 audit is approved, and no full training corpus may be
generated before the Phase 3 sample audit is approved.

## TinyWorlds-v1 Benchmark Boundary (Preserved)

TinyWorlds-v1 is a deterministic, knowledge-graph-first continual-learning
benchmark. Its symbolic graph, typed facts, positive safe Horn rules, proofs,
task dependencies, and candidate answers are authoritative. Natural-language
stories and queries are deterministic projections of those records. External
text-generation models are outside the benchmark through Phase 5.

TinyStories language specialization remains a separate historical benchmark.
Its post-mortem is labeled exactly `in-domain topic specialization`; that
interpretation is never reused by a TinyWorlds report.

## Training Artifacts and Report-Time Evaluation

Language training and report-time evaluation are separate interfaces.
Training tasks own batches plus fixed parent and content-key probes.
`LanguageEvaluationSuite` owns explicit report conditions, paired examples,
source provenance, and visible-prefix cue metadata. Adding or changing a suite
cannot change training probes, parent choices, content keys, RNG evolution, or
adapter tensors. `evaluate_language_benchmark` accepts already-trained
adaptations and a suite and performs no training transition.

The reusable adaptation boundary is a checksummed safetensors
`LanguageAdaptationArtifact`. It stores sequential, independent, and VAMP
LoRAs; immutable graph topology; address keys; RNG state; parent scores;
training traces and task order; and model, LoRA, training, and run hashes. The
frozen base is referenced by its checkpoint identity and is never copied into
the artifact. Loading is strict over schema, names, shapes, dtypes, hashes, and
the frozen-base reference.

The TinyStories post-mortem uses 128 deterministic 256-token anchor spans per
task, selected with stride 32 and source-story round robin from the complete
classified official test half. The 64/128, 128/128, and 192/64 conditions are
nested views of the same ordered span-level pair identities. Provenance records
the source document, token offset, pair hash, and unique-story count; the
benchmark does not interpret the spans as independent stories.

Cue strata are derived from visible prefix text only. A prefix is
`cue_sufficient` when the existing topic classifier returns its assigned task,
`cue_present` when an assigned-topic concept is visible without satisfying the
classifier, and `cue_hidden_or_ambiguous` otherwise. Examples are never
filtered by cue stratum, and `all` is a derived aggregate. A completed report
is a deterministic projection of its completed evaluation object and must
reproduce byte-for-byte without training or mutating the adaptation artifact.

## TinyWorlds-v1 Symbolic Generation

The master seed is SHA-256-derived from the canonical benchmark version,
public seed, frozen-base manifest SHA-256, and frozen-base parameter checksum.
Every stochastic choice uses a named SHA-256 namespace plus stable record
identifiers. Adding a record or an unrelated namespace therefore cannot
reshuffle existing entities, facts, rules, proofs, queries, or split choices.
Calibration and pilot worlds share the derivation contract but occupy disjoint
namespaces and generated vocabularies.

The symbolic layer uses immutable typed IDs, typed entities, a predicate
registry, ground atoms, positive-safe Horn rules, canonical query ASTs, and
proof records. Ordinary predicates are unary or binary; contextual revision
predicates use an explicit third context argument. Predicate dependencies must
be acyclic. Closure is deterministic and limited to depth two; canonical proof
selection is independent of input ordering. An inferred conclusion may not
also be a direct training statement.

The fixed calibration topology is seed, extension, revision, bridge. The
pilot contains willow and sunny families in interleaved seed, extension,
revision, and bridge stages. Extensions and revisions are seed children;
bridges descend from revisions but their queries require seed, extension,
revision, and bridge edge support. Cross-branch queries therefore have an
empty hard-node oracle and explicit required edges. No single hard path may
claim complete bridge support.

Every semantic query has one graph answer, one canonical proof, four unique
same-type candidates, and explicit task/edge support. Hard distractors are
selected deterministically in incompatible-revision, competing-task,
partial-proof, then filler priority. The predefined standard calibration mix
is the only alternative policy. A bundle must globally replay exactly one of
those policies; arbitrary per-query mixing or role relabeling is invalid.

Symbolic train, validation, and test plans are split before rendering. Template
family, plot, query phrasing, entity combination, proof-chain, and symbolic
text-hash axes are mutually disjoint. Story slices are canonical per task:
training contains all direct facts and task rules, validation contains the
first eight direct facts, and test contains the next eight, with no evaluation
rules stated directly.

## TinyWorlds-v1 Persistence and Novelty

Symbolic bundles are canonical JSON/JSONL plus a dependency-free
MeTTa/Atomese text projection. Manifests bind every artifact's path, count,
size, digest, bundle/world identity, version, and master seed. Loaders reject
unknown fields or files, noncanonical encoding, dangling references, digest or
count mismatches, split overlap, noncanonical proofs, story slices, topology,
edge ownership, revision provenance, bridge quantities, mixed fact capacity,
or fabricated candidate provenance. The text export is regenerated from the
authoritative records and compared byte-for-byte.

The original pretraining novelty boundary is exactly
`TinyStories-train.txt` at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, size 1,924,281,556, and SHA-256
`c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`.
Only the marked novelty audit streams this file. It audits every generated
name, class, role, and inflection by case-folded exact lexical words. Ordinary
tests use offline fixtures.

## TinyWorlds-v1 Deterministic Rendering

Natural-language data is a deterministic projection of the symbolic bundle
through `tinyworlds-templates-v1`. Every predicate, rule, query kind, and plot
target has train-, validation-, and test-specific template identities. Story
records retain the realized template occurrence order plus exact character
spans aligned to authoritative fact and rule IDs. Root-validation stories are
task-neutral and claim no symbolic knowledge.

The 256 validation and 512 test semantic groups per task are immutable
`QueryGroupPlan` instances. A source symbolic query or proof may repeat because
the finite world cannot supply hundreds of distinct proof graphs; repetition
is explicit through source-query, source-proof, occurrence, and replication
provenance. Each instance has a unique group and holdout identity, fixed four-
candidate order and correct position, and one accepted template family. It is
rendered exactly once into paired 64-, 128-, and 192-token variants. Ordered
fallback attempts are expanded before rendering, only expected rendering
rejections advance to the next attempt, and the accepted plan is stored on the
rendered group rather than inferred from modulo cycling.

The final 64 tokens are one shared query core. Preceding 64-token blocks carry
hash-balanced task, family, ambiguous, or cue-free-control text without answer
facts. Cue metadata is replayed from the visible prefix: sufficient cues admit
exactly the intended task, present cues admit its family, and hidden/control
cues admit all tasks. Closed-book prefixes are rejected when they contain a
candidate string or candidate-token subsequence.

Prefix-plus-candidate tokenization is authoritative. A record is accepted only
when standalone prefix tokens are an exact prefix of all four combined
sequences, all candidate suffixes have equal positive token counts, and every
sequence fits the fixed 256-token context. Neutral fragments are searched with
a deterministic closest-under-target traversal and an exhaustive deterministic
fallback; words are never truncated. Training adapters consume whole rendered
stories with exact fact-exposure accounting, while knowledge evaluation uses
the explicit semantic candidate boundary and never a sliding-span selector.

Rendered persistence schema 2 stores accepted-plan provenance and canonical
story/query JSONL while reconstructing token arrays on load. Strict loading
replays every story and query from the symbolic bundle and pinned tokenizer,
including text, templates, plots, alignment spans and IDs, cues, candidates,
proof/support metadata, masks, and accepted fallback choice. Plot IDs and text
hashes are split-isolated. Canonical materialization publishes only after this
validation and a reuse invocation must reproduce the completed content-only
result byte-for-byte without changing the artifact tree.

## TinyWorlds-v1 Candidate Scoring and Knowledge Evaluation

Knowledge competence is an exact four-candidate comparison. Each candidate
stores its own prefix-plus-answer `CompetenceBatch`, while one separate
`RouterBatch` contains only the common visible prefix. Candidate text may
change neither that router batch nor any routing result. NLL is normalized
only over active answer tokens, so context length and padding do not alter the
candidate comparison.

Frozen-base, hard-node, and arbitrary per-example edge-coefficient scoring
share one execution contract. Queries are flattened as `query x 4`, grouped
only when tensor shapes require it, microbatched through the same model path,
and restored in original order. Hard evaluation produces one
`[query, candidate, node-capacity]` tensor; VAMP oracle and every hard router
reuse it. A one-hot continuous edge vector is numerically equivalent to its
corresponding hard path.

Uniform- and Hopfield-initialized EBT each refine the prefix once per execution
shape. The final trace supplies both the hard argmax decision and the
continuous edge coefficients used by its paired soft candidate scorer. Soft
regret is measured against the best hard-node correct-answer NLL and may be
negative. Hard required-edge support is the fraction present on the selected
path; soft support is the mean final coefficient over required edges. A future
edge contributes zero until it exists. Cross-branch queries deliberately have
no hard-node oracle, so task-oracle and node-accuracy values are undefined
rather than fabricated.

Every method reports candidate accuracy, wrong-minus-correct margin,
correct-answer NLL, routed/task-oracle/best-hard regrets, routing uncertainty,
and applicable node, top-k, and support metrics. Aggregation is deterministic
over stage, method, task, family, query kind, prefix length, cue regime,
reasoning type and depth, novelty regime, and open/closed-book mode.

## TinyWorlds-v1 Parent Selection and Resumable Transfer

TinyWorlds parent selection uses only the explicit validation suite. Every
current hard node is scored by mean correct-candidate NLL; equal means resolve
by graph insertion order. Each stage trains same-state counterfactuals from the
root, true parent, selected parent, and strongest other-family parent, with
coincident roles sharing one physical trial. Only the selected-parent edge is
committed; all other trained states remain transfer diagnostics.

Counterfactual checkpoints contain the complete trainable, optimizer, random,
and update state. Their diagnostic schedule is exactly update zero, powers of
two, and the final budget. Immutable chunks are atomically published, reject
existing paths including dangling symlinks, and must form a prefix of that
schedule. Reload binds the stored state to the recomputed plan and validation
suite, frozen parameters, packed memory, training batches, model/LoRA/optimizer
configuration, and current update-zero state. This identity check is mandatory
even when a final-budget chunk needs no additional optimizer update.

## TinyWorlds-v1 Calibration Boundary

Calibration uses its own four-task world and the fixed one-axis ladder. Trial
selection, optimizer randomness, and cache identity are derived only from the
topology, training records, validation stories, and validation queries for
both prescribed distractor policies. Held-out test records have a separate
locked-test identity and cannot change validation execution, selected values,
or training randomness. A stopped calibration never opens the test split. On
the successful path, the test split is opened exactly once and only after the
locked scratch configuration passes every validation gate.

Every calibration resume chain must be the exact prefix of updates zero, one,
two, four, and so on through the requested final budget. Loading validates all
earlier chunks rather than trusting only the latest one, binds independent
training to its original full initial state, and rejects symlinked chunk roots
or targets. Committed-node stability hashes every fixed validation query ID and
its complete logit tensor in deterministic microbatches; a change outside the
first probe is therefore still drift.

Canonical trials require the JAX GPU platform, device kind exactly
`NVIDIA GeForce RTX 4090`, and a 12 GiB allocator-peak target. Device and peak
evidence is captured after trial scoring and validated before the immutable
trial artifact is published. Cache reuse revalidates the same runtime identity,
and promotion strictly loads this evidence for every validation artifact and
the success-only locked-test artifact.

The raw trial-tree digest and execution digest are part of each observation,
calibration result, and passing profile. A completed ladder always promotes one
canonical, hashed `calibration_result.json`: either a successful result with a
locked profile and one test observation, or a stopped result with the complete
validation evidence and mechanical stop reason but no profile or test data.
Only the successful form can authorize the eight-task pilot.

A mechanically valid early stop is a completed Phase 4 outcome, not an
implementation failure. The ladder stops at the first axis for which none of
the fixed candidates passes, promotes every completed validation artifact and
its mechanical stop reason, and leaves both the profile and test artifact
absent. Such a result cannot authorize Phase 5. Gates must not be relaxed and
test data must not be consulted to rescue the run; changing the calibration
hypothesis requires a new versioned contract and a fresh calibration.

## TinyWorlds-v1 Interactive Inspection Boundary

The TinyWorlds notebook is a read-only analysis surface over immutable
artifacts. Its support layer strictly reloads the promoted calibration result,
validates trial-tree and execution identities, and uses the production
symbolic generator, tokenizer, renderer, closure, and proof checker for small
in-memory demonstrations. It does not duplicate benchmark semantics in
notebook cells, trigger training, perform parameter selection, or open the
held-out calibration test split.

Saved candidate scores are meaningful only for the exact canonical seed-0
world and rendered prefix/candidate strings that produced them. The notebook
therefore refuses to attach those scores to a newly seeded or otherwise
changed demonstration. New worlds remain useful for inspecting stories,
proofs, cues, candidates, and exact-KG behavior without suggesting that a
model evaluated them.

The addressing view illustrates hard-path recall and continuous required-edge
support algebra. Phase 4 did not persist EBT-soft candidate scores, so the
notebook must not fabricate them. Widget and Jupyter dependencies remain in an
optional package extra, and the core inspection module does not eagerly import
the UI stack.

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
Each EBT refinement result also retains aligned step-zero-through-final traces
of its objective, node-address probabilities, and induced path-edge
coefficients. The canonical benchmark records both EBT initializations for test
example zero of the final task at the primary prefix length. This deterministic
representative trace is diagnostic rather than an aggregate performance metric;
reports preserve it as JSONL and render node/edge heatmaps plus an objective
curve so initialization and routing convergence can be inspected directly.
When a peak-device-memory target is configured, the runner requires allocator
peak statistics from the active JAX backend and fails the run if the observed
peak exceeds the target. The completed benchmark owns its addressing traces
and generated samples: trace capture, sample routing, and generation all finish
before the final allocator read, so the enforced and reported peak covers that
work as well as training, evaluation, and timing. Successful reports record the
observed peak, backend limit, target, platform, and device kind; CPU smoke runs
without a target may report those allocator fields as unavailable.

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
