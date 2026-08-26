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
completed the shared CNN, revised preflight, and real smoke pass on the local
RTX 4090. The CNN again selected four epochs and reached 99.11% ordinary-MNIST
test accuracy. The context probe reached 50.066%. The five top-two-layer oracle
adapters reached 98.96%, 98.16%, 98.20%, 98.22%, and 98.24%, for a 98.356%
mean that clears the 90% capacity gate. The joint top-two adapter reached
81.426%.

The 5,000-example smoke completed in 59.96 seconds with 30 splits, 31 leaves,
and 61 nodes. Final AF routed accuracy was 45.878%, versus 35.016% global
replay, 54.584% joint IID, and 60.044% online oracle-context accuracy. The
exhaustive oracle-leaf diagnostic reached 88.008%, while hard routing agreed
with that leaf on only 15.560% of test examples. The revised adapter therefore
fixes the capacity gate and exposes routing as the dominant smoke-stage gap;
the three-seed main and forced-consolidation passes have not been run yet.
