# Technical Design

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
