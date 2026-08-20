# ImageNet-R-50 Results Handoff

This directory contains the compact evidence bundle for protocol
`08d22d66a713f9d3d45454935af6043a716a71888a169946ef5c2244af0809db`.
It is intended for analysis of the completed primary matrix and selection of
the next experiments. It is not a portable training archive.

## Start here

1. `reports/REPORT.md` gives the protocol summary and headline table.
2. `reports/summary.json` contains exact aggregate values for every condition.
3. `reports/stage_accuracy.csv` and `reports/task_accuracy_matrix.csv` contain
   the complete stage-level and stage-by-task measurements.
4. `reports/merge_diagnostics.parquet` and
   `reports/routing_diagnostics.parquet` separate consolidation damage from
   task-free routing error.
5. `reports/resource_metrics.json` contains training, consolidation, proxy,
   deployment-memory, and addressing-cost accounting.
6. `baselines/e2lora/result.json` and `baselines/e2lora/official.log` preserve
   the local official E2-LoRA reproduction result and its raw execution log.

`reports/REPORT.html` is a self-contained visual report. The PNG files and
`reports/lineage.svg` are also retained so the Markdown report renders with
its original figures.

## Included evidence

- The complete generated `reports/` directory: narrative reports, plots,
  summary JSON, CSV matrices, Parquet diagnostics, lineage, job ledger, leaf
  manifest, node manifest, and resource accounting.
- `config_resolved.yaml` and compact protocol manifests for the class order,
  model checkpoint, pinned sources, installed software, material-code hashes,
  and GPU preflight.
- `state/scheduler_state.json`, which records 90 completed jobs with no error.
- `diagnostics/artifact_reuse.json` and
  `diagnostics/exact_rank_diagnostics.json`.
- The official E2-LoRA result record and raw log.
- `artifacts/imagenetr50/LATEST_RUN.json`, which selects this protocol.

## Deliberately excluded

- Dataset images, the source archive, and the 30,000-row dataset manifest. The
  full manifest is useful for reconstructing the immutable split, but not for
  interpreting results. Its content hash is
  `76ca289711f37323e256a7b62d19e29c414ee4d513b31e3e6c23adc5eb768448`;
  the split contains 24,000 train and 6,000 test images with seed 1993.
- Per-condition `evaluations/*/evaluation.json` files. They duplicate the
  stage and task records in the two report CSV files and add about 11 MB.
- Adapter/classifier tensors, optimizer checkpoints, leaf artifacts, tree
  artifacts, baseline model states, merge intermediates, and repair states.
  These are needed to resume or rebuild models, not to analyze the outcome.
- Frozen-feature, proxy-activation, and evaluation-logit caches. They are
  performance caches rather than scientific evidence.
- The local virtual environment, downloaded backbone, prepared ImageFolder
  tree, and pinned external source checkouts.
- `logs/`, which is empty. The only substantive raw log is the retained
  E2-LoRA `official.log`.

The omitted resumable/cached run archive is about 21 GB; the committed handoff
is about 16 MB.

## Provenance notes

- `reports/job_manifest.jsonl` was snapshotted from inside the report job, so
  its own row says `RUNNING`. The later and authoritative
  `state/scheduler_state.json` records all 90 jobs, including that report job,
  as `COMPLETE` with no errors.
- Node artifacts record Git HEAD `5ad73726a49704f30de0b48f052930b323f8db46`
  because the run occurred before the vision implementation was integrated in
  commit `24ada1e548208b5c90910b4f8b84a78ddd40f4a5`. The run-specific
  `protocol/code_manifest.json` is the authority for material source hashes.
- At handoff creation, 30 of the 45 material files still match that code
  manifest byte-for-byte. Fifteen integrated files differ, including LoRA,
  calibration, merge, checkpoint, manifest, and repair modules. Treat exact
  reproduction from the current checkout as unverified until those differences
  are audited or the run is repeated from a newly frozen protocol.
- Generated ledgers contain absolute local paths under
  `/home/daniel/projects/snet/rpa`; those paths are provenance only and are not
  portable.
- A credential-pattern scan of the included files found no secrets. The
  E2-LoRA log SHA-256 is
  `c9beaf91d7786c6fc55483248ebe353947e728045eb8d8884bb446afbea2662d`.
