# VAMP-AF Addressable Rotated MNIST

This experiment tests whether one binary tree can provide both deterministic
input addressing and top-two-layer adapter ancestry. It is a mechanism proof,
not a state-of-the-art continual-learning benchmark.

The frozen seed-0 CNN maps MNIST images to normalized 128-dimensional
addresses. Five blocked contexts rotate the same balanced training identities
by 0, 18, 36, 54, and 72 degrees and shift labels by 0, 2, 4, 6, and 8. AF may
use only the frozen base embedding to route. Labels train full-rank deltas for
the CNN's `3136→128` embedding and `128→10` classifier, while both convolutional
layers remain frozen. Context IDs are reserved for the oracle control and
diagnostics.

Before AF, the runner reports the linear context probe and requires at least
90% mean oracle-context top-two-layer accuracy. The context score is diagnostic
only. A failed oracle gate stops execution without changing AF. A passing gate
opens three fixed passes: a 1,000-example/context smoke, a
10,000-example/context three-seed main comparison, and one depth-three
forced-collapse stress run.

Run or resume the complete workflow with:

```bash
uv run python -m apm.experiments.vamp_af_mnist \
  --config configs/vamp_af_mnist/poc.yaml
```

Each pass/seed directory writes the required metrics ledger, accuracy and
routing CSVs, final tree JSON/PNG, four plots, consolidation ledger, resolved
configuration, and handoff. The aggregate run directory adds a machine-readable
acceptance summary and standalone Markdown/HTML reports. Raw state, checkpoints,
and generated evidence remain beneath `artifacts/vamp-af-mnist/` and are not
committed.

The authoritative implementation protocol is
[`VAMP_AF_POC_Codex_Spec.md`](../VAMP_AF_POC_Codex_Spec.md). Durable tree and
counter semantics live in [`DESIGN.md`](../../DESIGN.md); live execution status
lives in [`PLAN.md`](../../PLAN.md).

## Measured top-two-v3 result

Run `c3ad77df09fde94a75e2464450c21486d632bf4f60afe44c9602c6a86acf61af`
completed the shared CNN, revised preflight, smoke, three main seeds, and
forced-consolidation stress pass on the local RTX 4090. The CNN selected four
epochs and reached 99.11% ordinary-MNIST test accuracy. The context probe
reached 50.066%. The five top-two-layer oracle adapters reached 98.96%, 98.16%,
98.20%, 98.22%, and 98.24%, for a 98.356% mean that clears the 90% capacity
gate. The joint top-two adapter reached 81.426%.

The 5,000-example smoke completed in 59.96 seconds with 30 splits, 31 leaves,
and 61 nodes. Final AF routed accuracy was 45.878%, versus 35.016% global
replay, 54.584% joint IID, and 60.044% online oracle-context accuracy. The
exhaustive oracle-leaf diagnostic reached 88.008%, while hard routing agreed
with that leaf on only 15.560% of test examples. The revised adapter therefore
fixes the capacity gate and exposes routing as the dominant smoke-stage gap.

The three 50,000-example main seeds reached 61.170%, 59.132%, and 60.672% AF
routed accuracy, for a 60.325% mean. Mean controls were 62.661% global replay,
76.278% joint IID, 97.059% online oracle context, and 24.880% frozen base. The
exhaustive oracle-leaf mean was 99.228%, but hard routing agreed with its chosen
leaf on only 4.973% of examples. Final main trees contained 72, 71, and 69
leaves, reached the depth-eight cap, and recorded five consolidations in total.
All three work-ratio checks detected upward trends: first-quartile medians were
2.01--2.08 and last-quartile medians were 2.86--2.90.

The depth-three stress pass performed 19 splits and 13 forced collapses, ending
with seven leaves and 69.240% routed accuracy. The worst immediate collapse
changed accuracy by -0.392 points, passing the three-point drop bound. The
aggregate run therefore passes structural invariants, multiple-leaf use, depth,
and consolidation fidelity, but fails AF proximity to oracle context, AF's
five-point advantage over global replay, the oracle-leaf gap, and the flat-work
trend. This is a completed negative routing result: the top-two adapters retain
high-quality functions, while the frozen PCA-median address does not reliably
select them. A final authenticated resume reused all artifacts in 7.5 seconds.
