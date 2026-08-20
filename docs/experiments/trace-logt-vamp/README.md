# TRACE Log-t VAMP reviewer bundle

This directory is the complete human-reviewable evidence bundle for the sealed
TRACE Log-t VAMP run
`c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5`.
It is intended for an independent scientific review, including selection of the
next experiment. Start with [`REVIEW_PROMPT.md`](REVIEW_PROMPT.md), then read the
final Markdown report and use the compact index/sample before opening raw
candidate generations.

## Fast review path

1. [`final/reports/primary-report.md`](final/reports/primary-report.md) gives the
   registered aggregate results and interpretation.
2. [`final/reports/primary-scores.csv`](final/reports/primary-scores.csv) and
   [`final/reports/primary-merge-diagnostics.csv`](final/reports/primary-merge-diagnostics.csv)
   contain the main numeric evidence.
3. [`candidate-index.csv`](candidate-index.csv) maps all 532 raw generation
   files to condition, stage, task, split, row count, byte count, and SHA-256.
4. [`candidate-sample.jsonl`](candidate-sample.jsonl) contains one
   deterministically selected record from every raw generation file, stored in
   ordinary Git for lightweight browsing.
5. `evidence-volume/runs/<run-id>/evaluations/` contains the full example-level
   results and raw candidate generations. The JSONL generations are Git LFS
   objects.

## Evidence map

| Evidence | Files | Logical bytes | Location |
| --- | ---: | ---: | --- |
| Raw evaluations and result records | 1,046 | 405,486,158 | `evidence-volume/runs/<run-id>/evaluations/` |
| Candidate-generation JSONL (315,397 rows) | 532 | 404,918,095 | same; Git LFS |
| Per-job human-readable logs | 156 | 5,808,105 | `evidence-volume/runs/<run-id>/logs/` |
| Coordinator/bootstrap/control logs | 10 | 11,585,102 | `evidence-volume/logs/` |
| Ledger, outputs, and session state | 575 | 391,288 | `evidence-volume/runs/<run-id>/state/` |
| Run/data/job manifests | 9 | 21,073,660 | `evidence-volume/runs/<run-id>/manifests/` |
| Registered control policies | 2 | 179 | `evidence-volume/policies/` |
| Final/interim reports and termination markers | 23 | 1,642,372 | `final/` |

`evidence-volume/SOURCE_SHA256SUMS` contains hashes computed on the mounted
RunPod volume before teardown. All 1,798 transferred source files were checked
against it after download. The final marker has SHA-256
`934d9f4d9f9b3b5269194c4c58f98d8527e1c01ea1facde9099c6b4633396357`;
its bound Markdown and HTML hashes are
`f43d35570993e5c40773165ca16147014343581777ce38d5555e26e765dcc3c3`
and `526aab21c7d39f7cc9da9d2ff23ce4004d0801c041b139fc19e7b940e1ba5589`.

## Conditions

| Directory / policy hash | Report condition |
| --- | --- |
| `1e51d297…` | `vamp_svd_r8_repair000` |
| `ae101c7f…` | `vamp_svd_r8_repair005` |
| `546c828a…` | `vamp_core_tsv_r8_scale03_repair000` |
| `f6efd136…` | `vamp_core_tsv_r8_scale03_repair005` |
| `b2be4a77…` | `vamp_core_tsv_r8_scale05_repair000` |
| `828c74a7…` | `vamp_core_tsv_r8_scale05_repair010` |
| named directories | `frozen_base`, `seq_lora_reference`, `seq_lora_40`, `joint_iid_lora`, and `taskwise_lora` |

The complete hashes and condition names are also present in
`primary-scores.csv`; do not infer conditions from abbreviated hashes alone.

## Working with Git LFS data

A normal clone with Git LFS installed materializes the candidate JSONL files.
To fetch only this experiment's raw generations:

```bash
git lfs pull --include="docs/experiments/trace-logt-vamp/evidence-volume/runs/c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5/evaluations/**/*.jsonl"
```

Browse without reading the full corpus:

```bash
python docs/experiments/trace-logt-vamp/sample_candidates.py \
  --condition vamp_svd_r8_repair000 --task Py150 --stage 8 \
  --split test --limit 5
```

The sampler streams files, uses a fixed hash priority, and does not depend on
filesystem order. Run `python .../sample_candidates.py --help` for filters, or
pass `--rebuild-derived` to reproduce the committed index and one-per-file
sample at their fixed paths.

## Scope and deliberate exclusions

This bundle includes every item classified during volume audit as reviewer
evidence: raw evaluations, readable job/coordinator logs, state/ledgers,
non-embedding manifests, registered control policies, and the complete final
result directory. It deliberately excludes resumability or cache material:
checkpoints, LoRA/merge tensors, prompt-embedding tensor and 243 MB embedding
JSONL dump, model/package caches, source checkout copies, and the superseded
13-job run. The implementation and tests are already versioned under
`src/apm/continual/trace/` and `tests/continual/trace/`.

The corpus was scanned before publication for the exact RunPod credential and
common Hugging Face, GitHub, AWS, bearer-token, and API-key secret forms. No
credential match was found. Words such as `password`, `secret`, or `api_key`
inside benchmark prompts/predictions are task data, not deployment secrets.

## Integrity checks

After LFS objects are materialized:

```bash
python docs/experiments/trace-logt-vamp/verify_bundle.py
```

The hashes in `final/reports/primary-manifest.json` and
`final/SAFE_TO_TERMINATE.json` are authoritative for the sealed report.
