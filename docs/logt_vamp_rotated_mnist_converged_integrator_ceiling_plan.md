# Converged full-replay integrator ceiling on VAMP-AF Rotated-MNIST

## Status and purpose

This document freezes the optimization-ceiling follow-up to the completed
`integrated-prediction-v1` experiment. The parent run has protocol identity
`9b5f70bf484cd19c7624142e80118e32857452d44f45fbb97d0b15df29a6689a`,
protocol-file SHA-256
`4308ac3ffd281656483f3f6db9ed17a7acdb82021cebc232a564664f117ac924`,
and aggregate-summary SHA-256
`abf2633df5e3409a9b30fef865d3257f8b7589d059d0d675f83e235fdd59e40b`.

The parent experiment does not contain a converged integrator ceiling. Its
`offline_cumulative_integrator` is fresh and sees all cumulative integrator
examples, but it is trained for exactly four epochs and only at checkpoints
7, 15, 31, 63, and 64. It is an optimization reference, not evidence of the
best performance available from the fixed node features and MLP. The
label-aware best-single-node control is an oracle selector and answers a
different question.

This successor measures the strongest integrator found under a deliberately
large, leakage-free optimization budget at every macro-step. It is an
empirical ceiling for the fixed feature representation, fixed MLP architecture,
available training allocation, optimizer family, and preregistered search. It
is not a mathematical upper bound over all functions or architectures.

## Frozen parent boundary

Reuse without alteration:

- the frozen VAMP-AF CNN and authenticated MNIST files;
- rotations 0, 18, 36, 54, and 72 degrees and label shifts 0, 2, 4, 6, and 8;
- the blocked 13, 13, 13, 13, and 12 step primary schedule;
- disjoint 256-example adapter, 256-example integrator, and 128-example
  evaluation allocations at each step;
- de-novo top-two adapter training and the binary-counter LogT frontier;
- five primary seeds 0 through 4;
- the seven 139-value level slots and the 973-to-1024-to-512-to-256-to-10
  residual integrator; and
- the parent's fixed transformed-test subsets and full-test checkpoints.

The new run rebuilds the hierarchy deterministically. The parent deliberately
retired inactive adapter files after durable checkpoints, so intermediate
frontiers cannot be reconstructed from the final checkpoint alone. Parent
metric ledgers are read-only, individually authenticated comparison inputs.

## Full-replay ceiling

At every macro-step, instantiate each restart from scratch. Recompute all
active-node slots against the current frontier for every integrator-training
example observed so far. Every epoch presents every cumulative training
example exactly once, uniformly shuffled; there is no logarithmic replay
sampling, current-source upweighting, or warm-start path dependence.

The cumulative evaluation allocation is the convergence-validation archive.
It can select epochs and restarts but cannot update model weights. Because it
is now a tuning input, do not report its performance as evaluation evidence.
The transformed test subsets and complete transformed test sets never
participate in fitting, stopping, learning-rate control, or restart selection.

Use AdamW with the parent's learning rate 0.001, weight decay 0.0001,
minibatches of at most 128, and gradient clipping at 1.0. After five epochs
without a validation cross-entropy improvement of at least 0.0001 nat, halve
the learning rate. Continue down to 0.00001. Declare convergence only after the
minimum learning rate has then completed ten more non-improving epochs. Stop at
200 epochs as a safety cap; a restart that reaches the cap is recorded as not
converged and is ineligible for selection. Restore the checkpoint with the
lowest raw validation cross-entropy. Use one restart in smoke and three
independent restarts per primary step; select the converged restart with the
lowest validation cross-entropy, breaking exact ties by higher validation
accuracy and then lower restart index.

The selected condition is named `converged_full_replay_integrator`. The phrase
"ceiling" is valid only when at least one restart converges at every step. The
reports must expose every restart, best epoch, epochs run, final learning rate,
training and validation metrics, and example-presentation count rather than
hiding a safety-cap failure.

## Evaluation and interpretation

At every step, evaluate the selected integrator on the fixed test subset for
all observed contexts. At checkpoints 7, 15, 31, 63, and 64, also evaluate on
the complete observed-context test sets. Retain the parameter-free mean
ensemble and label-aware best-single-node diagnostics. Exact mean-ensemble
parity with the authenticated parent ledger proves that the rebuilt frontier,
test identities, and fixed-node behavior match the completed experiment.

The primary comparison averages seeds at checkpoints 15, 31, and 63 and
reports:

- the converged ceiling versus the parent's four-epoch offline reference;
- the converged ceiling versus the parent's best online replay integrator;
- the remaining cross-entropy and accuracy gaps;
- per-seed and per-checkpoint variation; and
- convergence cost in epochs, optimizer steps, feature evaluations, and
  example presentations.

Interpret outcomes directly:

- a large ceiling gain over the four-epoch offline reference means the parent
  underestimates what its fixed features and MLP can achieve;
- a strong ceiling with a large online gap isolates replay or sequential
  optimization as the main limitation;
- a weak converged ceiling implicates feature sufficiency, model capacity, or
  train/test generalization rather than the logarithmic replay sampler alone;
  and
- failure to reach the frozen convergence rule means no ceiling claim is
  available, regardless of the best test score observed.

Test scores are never used to choose a restart, epoch, hyperparameter, or
whether the result is called successful.

## Smoke, resume, and artifact boundary

Unit tests must establish deterministic all-example epochs, best-validation
restoration, minimum-learning-rate convergence, cap failure, fresh independent
restarts, validation/test isolation, exact checkpoint/resume, and parent
mean-ensemble parity. Real smoke uses seed zero and five macro-steps. It opens
primary only if every selected restart converges, every cumulative example is
presented once per epoch, all metrics are finite, inactive slots are zero, and
the rebuilt fixed controls match the parent coordinates where available.

Commit the chained metric ledger and run checkpoint before retiring inactive
node files or transient restart weights. Preserve compact convergence histories
for every restart. Retain selected model weights at full checkpoints and remove
other completed-step weight payloads after their enclosing checkpoint is
durable. Execute primary seeds serially after host and GPU memory checks.

Use protocol revision `integrated-prediction-ceiling-v1`, strict config
`configs/vamp_logt_integrator_ceiling_rotated_mnist/primary.yaml`, CLI module
`apm.experiments.vamp_logt_integrator_ceiling_rotated_mnist`, and
content-addressed root
`artifacts/vamp-logt-integrator-ceiling-rotated-mnist/`. Bind the complete
resolved parent config, this plan, all material source files, parent protocol,
summary, and primary metric-ledger hashes, frozen classifier and data hashes,
installed PyTorch version, convergence histories, reports, and exact work
accounting.
