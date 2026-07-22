# VAMP Technical Design

## TinyWorlds-P Benchmark Boundary

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
