# Single-adapter SRT: completed experiment

The corrected run is
`runs/b3de2a819efe0707b85cf105cf8d4c53370c807b48108bf996dc4f4883a213ea`.
Its sealed result hash is
`a7834602b2f9b7a00401f7a34fb3396ae3cf31c43a6d7b6ce528eb48bcc2f098`.

Read that run's `reports/REPORT.md` or self-contained `reports/REPORT.html`.
The rendered report is `output/pdf/imagenetr50_srt_r16_report.pdf` at the
repository root. It now includes the fixed-policy H=4,096 and H=512 extensions,
the offline joint controls and checkpoint diagnostics. The earlier persistent
frontier report and its results were not rewritten.

## Latest extension: H=512

Standard thresholds, old target 0.8, interval unit 8, seed 1993: SRT finishes at
78.933% accuracy / 0.9690 raw NLL; uniform at 78.367% / 0.9589. Mean stage
accuracy favors uniform, 83.943% versus 83.677%. Both use 196,204 training
presentations and 3,358 updates. These are single-seed exploratory results,
not evidence that SRT generally wins. Uniform is below the newer 80.828%
three-seed offline mean, which matches the H=4,096 schedule rather than H=512.

See `artifacts/imagenetr50/srt_h512/README.md` for run identities, verification,
resource measurements and rerun instructions. The main report adds a common
accuracy/NLL figure for the standard H=512/1,024/4,096 pairs. Older comparison
panels remain available.

## Original validation-selected comparison

Uniform replay outperformed the selected SRT recipe at both original work budgets:

| Condition | Final accuracy | Mean stage accuracy | Final NLL |
| --- | ---: | ---: | ---: |
| SRT, H=1,024 | 79.267% | 83.916% | 0.9585 |
| Uniform, H=1,024 | 80.533% | 85.059% | 0.8592 |
| SRT, H=4,096 | 77.333% | 81.908% | 1.1100 |
| Uniform, H=4,096 | 80.400% | 84.197% | 0.9123 |

H is a presentation-work budget, not a memory limit. These are single-seed,
same-split results. The two recipes were selected using training-derived
validation data, before final test evaluation. The numerical confidence
thresholds were hand-chosen candidates, not reported paper values.

## Evidence retained for analysis

- `reports/`: all figures, stage/task metrics, measured work, replay timing
  distributions, per-image replay summaries, sample timelines, calibration
  comparisons, and consistent condition names. Small tables use CSV, JSON and
  Parquet. The current complete per-image projection, `replay_samples.parquet`,
  contains 240,000 rows across ten conditions; its redundant CSV/JSON forms
  stay local. The earlier `sample_replay` tables are frozen inputs to the
  checkpoint diagnostics, not current exports. Their hash remains unchanged.
- `result.json` and `final/*/result.json`: all 50 stages for each final model,
  including per-task counts, metrics, work, and checkpoint/prediction hashes.
- `final/*/stages/050/predictions.parquet`: final per-image predictions and
  NLL for paired comparisons. Earlier raw predictions remain local; their
  aggregates and hashes are in the stage records.
- `calibration/*/result.json`: every completed calibration stage, not only
  the screening averages. `screen_selection.json` and `selected.json` record
  the frozen choices and selection criterion.
- `protocol.json`, configuration/software/code manifests, the calibration
  split, and image indices: scientific identity and membership provenance.
- `references/`: authenticated earlier result curves, work ledger, and the
  exact original full-stream figure.
- `preflight/result.json`, `reuse.json`, and `verification.json`: real-model
  preflight, zero-step reuse, and final evidence/rendering checks.
- Root logs include the corrected full run, fresh completed-workflow reuse,
  and focused tests. The preliminary run log and its supersession annotation
  are retained separately; its results did not enter selection or reporting.

Raw replay events, model/optimizer checkpoints, model caches, image data,
and operational health-monitor files remain local. This tracked evidence
supports analysis without a GPU, but intentionally is not enough to rerun the
full raw-event audit on a clean checkout without restoring the local store.
The report manifest includes hashes of local-only outputs as well as tracked
ones, and all table formats were checked against one another before handoff.

There were no crashes requiring recovery during active five-minute agent
monitoring. The independent systemd watcher was disabled. Calibration used
6,752,992 training presentations and the final runs used 2,278,016; preflight
and the superseded development run are excluded from those totals.
