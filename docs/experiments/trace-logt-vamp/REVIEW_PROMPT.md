# Independent review prompt

Review this TRACE Log-t VAMP experiment as a skeptical continual-learning
researcher. Your goal is to identify the strongest warranted conclusions,
challenge implementation or statistical assumptions, and recommend one next
experiment with the highest expected information value.

## What to inspect

Read, in order:

1. [`final/reports/primary-report.md`](final/reports/primary-report.md)
2. [`final/reports/primary-scores.csv`](final/reports/primary-scores.csv)
3. [`final/reports/primary-merge-diagnostics.csv`](final/reports/primary-merge-diagnostics.csv)
4. [`final/reports/primary-retrained-parent-calibration.csv`](final/reports/primary-retrained-parent-calibration.csv)
5. [`candidate-index.csv`](candidate-index.csv) and
   [`candidate-sample.jsonl`](candidate-sample.jsonl)
6. Relevant raw `result.json` and `*-candidates.jsonl` files under
   `evidence-volume/runs/<run-id>/evaluations/`
7. The run contract and policies under `evidence-volume/runs/<run-id>/manifests/`
8. Implementation code under `src/apm/continual/trace/`, especially
   `merging/`, `routing.py`, `evaluation.py`, `metrics.py`, and `reporting.py`

Use `sample_candidates.py` when example-level inspection would help. Do not
assume the report's interpretation is correct merely because its integrity is
verified.

## Registered interpretation to audit

- The registered primary task-free router is prompt NLL. Answer-oracle,
  task-aware, and frozen-centroid results are diagnostics or secondary routers,
  not substitutes for the primary result.
- Core scale 0.5 substantially reduces the recursive attenuation seen at scale
  0.3, while independently checked Core-Space algebra agrees with the upstream
  implementation within 2--7 ppm. This currently argues for a protocol-scale
  problem rather than a coding bug.
- Rank-eight SVD remains gentler than repair-free Core scale 0.5: it retains
  more weighted spectral energy and produces lower merge damage.
- Ten-percent Core repair nearly removes measured merge damage and greatly
  improves answer-oracle/task-aware/centroid scores, but prompt-NLL OP improves
  little. This suggests route selection is the dominant bottleneck.
- The best registered primary VAMP condition reaches 23.430 OP, versus 34.096
  for sequential LoRA and 47.253 for joint-IID. Repaired SVD with the secondary
  frozen-centroid router reaches 34.217.

## Questions the review must answer

1. Are OP, forgetting, signed/negative-only BWT, SARI, and routing comparisons
   calculated and interpreted correctly for this heterogeneous task sequence?
2. Is there any remaining plausible implementation artifact that could explain
   SVD's advantage over Core after the scale-0.5 causal control?
3. Does the repair evidence really isolate consolidation recovery, or could its
   apparent oracle gains arise from evaluation, reservoir, or budget mismatch?
4. What do raw route selections and candidate predictions reveal about why
   prompt NLL fails—calibration, task inference, candidate-set growth, prompt
   format, or something else?
5. Which conclusion is robust enough to guide the next experiment, and which
   conclusions remain underidentified by the single seed/order and unmatched
   repair budgets?

## Required recommendation format

Recommend exactly one next experiment and specify:

- hypothesis and decisive alternative;
- minimal intervention and all held-fixed components;
- conditions, seeds/orders, and matched compute/repair budgets;
- primary and secondary metrics, including route-selection diagnostics;
- expected outcomes under competing explanations;
- stopping rule and approximate GPU budget;
- what result would change the recommended research direction.

Prefer a compact experiment that separates routing/calibration from merge
quality before proposing a broad hyperparameter sweep.
