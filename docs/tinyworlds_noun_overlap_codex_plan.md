# TinyWorlds Noun-Overlap Experiment — Codex Execution Plan

## Goal

Implement the next TinyWorlds experiment as a simple noun-based continual-learning run:

1. Build a fresh TinyStories base set from common noun classes whose union covers about 50% of the training stories.
2. Treat each remaining noun class as a VAMP task.
3. Allow task datasets to overlap: a story containing both cats and dogs belongs to both the `cat` and `dog` tasks.
4. Train a fresh base model, then one VAMP LoRA edge per noun task.
5. Use the existing cheap NLL-based parent search, not adaptation-aware addressing.
6. Run the two requested evaluations:
   - whole-story NLL under each addressing mode;
   - first-half addressing, second-half generation, and external story-quality judging.

Keep this isolated from the completed TinyWorlds-Q and semantic-v6 experiments.

## Explicit non-goals

Do not add:

- exclusive story ownership;
- adaptation-aware parent search or temporary probe-LoRA training;
- fact extraction, semantic review, or query construction;
- class balancing;
- consolidation;
- independent or sequential-LoRA baselines;
- elaborate preregistration or sealed-test transaction machinery.

Use deterministic artifacts and resumability, but do not build another benchmark bureaucracy.

---

## 1. New experiment package

Create one small package and one canonical runner:

```text
src/apm/data/text/tinyworlds_nouns_v1/
    __init__.py
    contracts.py
    partition.py
    experiment.py
    evaluation.py
    judging.py
    report.py

scripts/run_tinyworlds_nouns_v1.py
tests/test_tinyworlds_nouns_v1.py
```

Artifact roots:

```text
data/tinyworlds-nouns-v1/
checkpoints/tinyworlds-nouns-v1/
results/language_cl/tinyworlds-nouns-v1/
```

The single runner should resume through these phases:

```text
1. Partition
2. Base training
3. VAMP task training
4. Whole-story NLL evaluation
5. Half-story generation
6. OpenRouter judging
7. Final report
```

Reuse generic code already present in the repository rather than copying TinyWorlds-Q-specific catalog or query code.

---

## 2. Noun matching and overlapping task membership

### Noun catalog

Start from the existing `TINYSTORIES_TOPICS` / `TopicConcept` catalog in:

```text
src/apm/data/text/curricula.py
```

Use each concept's explicit forms, including existing alternates such as:

```text
cat / cats
mouse / mice
mother / mothers / mom / moms
bicycle / bicycles / bike / bikes
```

Match case-insensitive whole words. Do not add spaCy, WordNet, or an LLM classifier. A small editable override table is acceptable for obvious missing forms.

### Story membership

Deduplicate stories by normalized content hash, then compute every noun class matched by each story.

Membership is deliberately nonexclusive:

```text
"The cat and dog played together."
    -> cat task
    -> dog task
```

A story may occur in any number of noun tasks. This is intended, not leakage to repair.

Persist one compact story ledger containing:

```text
story_id
source split
matched noun classes
matched surface forms
token count
```

---

## 3. Select the base noun classes

Use the pinned TinyStories training file.

1. Count how many unique training stories match each noun class.
2. Sort noun classes by descending matching-story count, then by noun name.
3. Add noun classes in that order.
4. After each addition, recompute the union of training stories matching at least one selected noun.
5. Stop at the first addition for which this union covers at least 50% of all unique TinyStories training stories.

The base training set is that union.

Do not remove a story merely because it also mentions a future task noun. For example, if `cat` is a base noun and `dog` is a later task noun, a cat-and-dog story may appear both in base training and in the dog task. Record the fraction of each task's stories that were also present in the base, but do not filter them.

Publish a short base-selection table:

```text
noun added
noun story count
new stories added to union
cumulative unique-story coverage
cumulative token coverage
```

---

## 4. Build the noun tasks

The task noun classes are the noun classes not selected for the base.

Keep every remaining noun class with at least:

```text
256 matching training stories
64 matching official-validation stories
```

For each retained noun:

```text
training set   = every TinyStories training story matching that noun
validation set = every official TinyStories validation story matching that noun
```

Again, these sets may overlap across nouns.

Order tasks by descending training-story count, then noun name.

For each task, reserve the 36 lowest-hash training stories as addressing probes and remove only those 36 stories from that task's LoRA-update stream. Use the same 36 stories for parent scoring and the node content key. All official-validation stories remain untouched for final testing.

Use 36 deterministic base stories as the root content-key probes.

The task/story pair is the evaluation unit. A cat-and-dog validation story is evaluated once as a cat example and once as a dog example, with different oracle nodes.

---

## 5. Train the fresh base model

Use the same GPT-Neo architecture, tokenizer, optimizer, and basic training settings already used by the current TinyWorlds experiments:

```text
fresh seed-zero initialization
2 base epochs
context length 256
microbatch size 32
8 accumulated microbatches
maximum learning rate 5e-4
minimum learning rate 5e-5
1% warmup
Adam beta1 0.9
Adam beta2 0.95
Adam epsilon 1e-8
weight decay 0.1
gradient clipping 1.0
```

Do not load the fully trained TinyStories-8M parameters, because they have already seen the whole corpus.

Requirements:

- exact resume of parameters, optimizer, RNG, epoch, and data cursor;
- checkpoint at least every 1,000 optimizer updates and at epoch boundaries;
- finite train and validation NLL;
- epoch-two validation NLL lower than epoch one;
- retain the existing 12 GiB allocator limit.

The selected base artifact must bind the partition, tokenizer, architecture, and training configuration.

---

## 6. Train the VAMP graph

Train only VAMP. Do not call the wrapper that also trains independent and sequential baselines.

Reuse:

```text
init_language_vamp_run
score_parent_nodes
advance_language_vamp_run
```

Use:

```text
LoRA rank 8
LoRA alpha 8
2,000 updates per noun task
learning rate 1e-3
weight decay 0.01
gradient clipping 1.0
```

### Parent policy

Do not implement adaptation-aware addressing.

Use the existing mean prefix-NLL parent scores:

```text
Task 1:
    attach to root.

Task 2 onward:
    score root and every existing task node using the 36 task probes;
    log every raw score;
    attach to the lowest-NLL non-root node.
```

This is the only topology constraint. It prevents the trivial all-root star without paying for temporary LoRA adaptation trials. Keep the root score in the artifact so the report shows how often the forced non-root parent was worse than root.

Implement this as a small optional eligibility mask in the existing parent-selection path rather than duplicating the training code.

For each task:

1. Score eligible parents.
2. Select the parent.
3. Train a fresh LoRA edge while applying the selected parent's complete root-to-parent path.
4. Freeze the new edge.
5. Verify that every older edge checksum is unchanged.
6. Save an immutable resumable stage containing the graph, edge tensors, parent scores, content key, RNG state, and loss trace.

---

## 7. Test 1 — whole-story NLL

Evaluate every task/validation-story pair under exactly these conditions:

```text
base
oracle
vamp_exhaustive
vamp_hopfield
vamp_ebt_uniform
vamp_ebt_hopfield
```

Definitions:

- `base`: frozen base with no LoRA edges.
- `oracle`: the VAMP node belonging to the noun task being evaluated.
- The other four are the existing task-free addressing modes.

For task-free modes, use the complete story as the addressing query. Then compute NLL over the complete story under the selected node.

Reuse the repository's causal token windowing for stories longer than one training context. Sum token NLL so every causal target, including EOS, is counted exactly once.

Stream one JSONL row per task/story/condition:

```text
task_noun
story_id
condition
selected_node
selected_path
oracle_node
oracle_match
total_nll
token_count
mean_nll
perplexity
regret_vs_oracle
```

Report:

```text
per-task mean NLL and perplexity
overall story-weighted mean
overall token-weighted mean
task-node routing accuracy
mean regret versus oracle
confusion matrix
```

---

## 8. Test 2 — first-half addressing and completion

For every task/validation-story pair:

1. Tokenize the story with EOS.
2. Split at the exact token midpoint.
3. Use the first half as the addressing query and generation prompt.
4. Use the original second half as the reference continuation.
5. Compute true-second-half NLL under each condition.
6. Generate a continuation under each condition using deterministic greedy decoding.
7. Give every generated condition the same maximum output length: the reference continuation's token count, capped only by the model position limit.

Generate for:

```text
base
oracle
vamp_exhaustive
vamp_hopfield
vamp_ebt_uniform
vamp_ebt_hopfield
```

Save the original reference as:

```text
reference
```

Write one streamed JSONL row per task/story pair containing:

```text
task_noun
story_id
prefix
reference continuation
full original story

for each condition:
    selected node
    selected path
    true-second-half NLL
    generated continuation
    generated token count
    EOS reached
```

The router must receive only the first half. Add a test that makes continuation access impossible through the API.

---

## 9. OpenRouter judging

Default judge:

```text
z-ai/glm-5.2
```

Optional alternative:

```text
openai/gpt-5.3-chat
```

Read the model slug from one environment variable:

```text
OPENROUTER_JUDGE_MODEL
```

Default it to GLM 5.2. Read credentials from:

```text
OPENROUTER_API_KEY
```

Use one request per task/story pair. Present the prefix once, then present these anonymized and deterministically shuffled continuations:

```text
base
oracle
vamp_exhaustive
vamp_hopfield
vamp_ebt_uniform
vamp_ebt_hopfield
reference
```

Do not reveal the source condition or task noun.

Require a small JSON result:

```json
{
  "scores": [
    {
      "candidate": "A",
      "coherence": 1,
      "writing_quality": 1,
      "ending_quality": 1,
      "overall": 1,
      "reason": "brief explanation"
    }
  ],
  "ranking": ["A", "B", "C"]
}
```

Use integer scores from 1 to 5. Require every candidate exactly once.

Persist each request and result immediately. On resume, skip completed task/story IDs. Retry transient failures, but do not build a complicated provider-management layer.

If the API key is absent, finish all local work and stop with a clear message. Rerunning the same command should resume at judging.

Summarize:

```text
mean score by condition
mean rank by condition
win rate against base
win rate against oracle
win rate against reference
per-task results
```

---

## 10. Final report and artifacts

Publish:

```text
partition.json
base-selection.csv
task-counts.csv
vamp-graph.json
vamp-graph.dot
parent-scores.csv
whole-story-nll.jsonl
half-story-generations.jsonl
judge-results.jsonl
report.md
report.html
run-manifest.json
```

The report should answer:

1. Which nouns entered the base, and what fraction of TinyStories did they cover?
2. Which noun tasks were trained, and how much did they overlap with each other and the base?
3. What graph topology resulted under the non-root parent rule?
4. How much did oracle VAMP improve NLL over the base for each noun?
5. How well did each task-free addressing mode recover the correct node?
6. How much NLL regret came from routing?
7. Which generated completions were judged best?
8. Did task overlap help transfer or merely weaken task specificity?

Include several representative stories and completions, but do not add statistical machinery beyond basic means, counts, and pairwise win rates.

---

## 11. Minimal tests

Add focused tests for:

1. Whole-word and alternate-form noun matching.
2. One story belonging to multiple noun classes.
3. Base noun union crossing 50% coverage.
4. Base/task overlap being retained and reported.
5. Deterministic task order and probe selection.
6. First task attaching to root and later tasks attaching to the best eligible non-root node.
7. Child training applying the selected parent path.
8. Older edge checksums remaining unchanged.
9. Whole-story NLL counting each target once.
10. First-half routing having no continuation access.
11. Equal generation budgets across conditions.
12. Deterministic judge anonymization, parsing, and resume behavior.

Keep GPU and network tests skipped by default.

---

## 12. Canonical command and completion condition

Run and resume the complete experiment with:

```bash
python scripts/run_tinyworlds_nouns_v1.py
```

The work is complete when:

- the noun partition and approximately 50% base coverage are published;
- the fresh base has completed two epochs;
- every retained noun task has a committed VAMP node;
- every validation task/story pair has all six NLL rows;
- every eligible task/story pair has all six generations plus the reference;
- judging is complete when credentials are available;
- the report reconstructs entirely from persisted artifacts without retraining or regenerating.
