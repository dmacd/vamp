# ImageNet-R-50 Node-Adapted Latent Replay Protocol

This is an exploratory full-50 follow-up to the immutable fresh-parent run
`5fc6e1a57076f06a31da8b638207644109289a9b8888daf6b44934ade3c89320`.
It reuses that run's 50 leaves, 47 full-union parents, 50 historical
frontiers, 24,000/6,000 split, and stage-matched joint-IID curve without any
leaf or parent optimizer step.

The task-free integrator receives six stable binary-counter level slots. Each
active slot concatenates the following values for the corresponding live
node:

1. the 768-dimensional top-level pre-classifier representation produced
   **after that node's own rank-16 LoRA is installed**, normalized within the
   example;
2. 200 affine classifier scores;
3. 200 within-node log probabilities;
4. a 200-dimensional classifier-row ownership mask; and
5. one active-slot bit.

Inactive slots are zero. The resulting input has `6 * 1,369 = 8,214`
dimensions. Labels and task identities are absent from inference inputs. The
residual MLP keeps the existing 1,024/512/256 hidden widths, output head,
dropout, AdamW settings, four epochs per arrival, deterministic seed 1993,
and warm-started persistent state.

The primary matrix changes only the class-stratified uniform historical replay
capacity: `H = 2,048`, `4,096`, or `8,192`. Each arm receives the current
four-class task plus at most `H` deterministic historical training examples.
All three arms use the same initial parameter and random-number schedule; arm
identity does not perturb initialization. They train completely before any
test behavior is materialized in the follow-up run.

For fixed `H`, arrival `t` observes at most the current task plus `H` historical
examples at `popcount(t) <= ceil(log2(t + 1))` live nodes. Total behavior work
is therefore bounded by a capacity-dependent constant times `T log T`.
Increasing `H` changes that constant, not the asymptotic dependence on stream
length. The report records exact image presentations, node/example observation
bounds, cache misses, optimizer steps, and wall time for each arm.

The same locked test identities have already informed earlier project
diagnosis. Results from this follow-up are therefore descriptive and
hypothesis-generating, not a fresh confirmatory or publishable test claim.
The stage-matched joint-IID rank-16 LoRA curve and true-node oracle are
comparators, never gates.
