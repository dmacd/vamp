# Development Plan

## Current Milestone

- Stage 1 supports dense-delta addressed memory over PermutedMNIST and
  digit-incremental MNIST with VAE and FabricPC model backends.
- Benchmark reports include train/test retention, address confusion, raw
  observed-energy diagnostics, memory graphs, reconstructions, and clickable
  Matplotlib plots. The shared report viewer supports fit, 25%-800% zoom,
  scroll and drag panning, keyboard/wheel zoom, and opening the original asset.
- Fixed-epoch and observed-energy-convergence schedules are implemented for
  both backends. Convergence traces record every monitored epoch and the state
  selected for the committed memory node.

## Next Run

1. Make FabricPC evaluation shape-stable by padding and masking evaluation,
   observed-energy, and reconstruction batches before any further full run.
2. Run the full ten-digit FabricPC benchmark with the default convergence
   schedule and memory-only benchmark mode.
3. Inspect task stopping epochs, reconstruction quality, and train/test address
   confusion before changing model regularization or addressing semantics.
4. Compare the convergence run against the existing fixed ten-epoch FabricPC
   checkpoint, especially early-digit routing to later parameter nodes.

## Known Gaps

- The full ten-digit convergence benchmark has not yet been run; current
  verification uses deterministic two-digit VAE and one-digit FabricPC smoke
  runs.
- Observed energy is model-specific and is valid for within-model stopping and
  addressing, not direct VAE-to-FabricPC scale comparisons.
- Reaching the maximum epoch limit is reported and the best state is retained,
  but it does not count as convergence.
- FabricPC inference calls are JIT-compiled but currently receive variable tail
  and addressed winner-group shapes. JAX compiles a separate executable for
  every distinct shape, causing periodic all-core XLA compilation spikes and
  GPU stalls during later evaluation phases.
