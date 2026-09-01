# Post-hoc ungated dense-base successor on Permuted-MNIST

## Status and scope

This document is a successor amendment to
`logt_vamp_permuted_mnist_dense_mlp_protocol.md`. It was written after the
original three-width, three-seed calibration completed and therefore is not a
preregistered change. The original protocol and its `ineligible` artifact stay
unchanged. Results from this successor must be described as post-hoc or
exploratory when compared with the original 99% gate.

Every architecture, dataset, split, seed, optimizer, convergence rule,
temporal allocation, hierarchy rule, condition, reference, evaluation view,
acceptance criterion, and accounting requirement from the original protocol
remains fixed except for architecture-selection eligibility below.

## Authorized calibration change

Calibration accuracy is no longer a blocking gate. Identity and pooled
validation accuracy remain recorded as descriptive capacity measurements, but
they cannot stop the successor or select among candidates. Selection takes
the smallest calibrated candidate, 1024/1024/512, without consulting a test
metric. The retained threshold fields are non-operative historical context.

The successor does not repeat deterministic training. It imports the original
18 calibration fits only after authenticating the original protocol identity,
dataset and permutation manifest, complete calibration configuration apart
from the non-operative selection fields, every per-fit result coordinate,
every restored best-epoch metric, and every checkpoint hash. The smallest
width, 1024/1024/512, is then selected. Its original
seed-zero identity checkpoint becomes the successor's shared base. Test
metrics may be evaluated only after that selection is reproduced.

## Downstream experiment

The complete downstream plan is unchanged. Five seeds build independent
64-step immutable hierarchy tapes, retaining all 127 nodes per seed. The five
matched router conditions, four online integrator conditions, fixed controls,
fresh four-epoch cumulative integrator, pooled single-MLP reference, and
three-restart validation-converged full-replay ceiling consume those tapes.
The ceiling remains fresh at every macro-step. The seven original decisions
and the disclosed 8-versus-16 online optimizer-update asymmetry remain in
force.

The successor uses config revision `dense-full-model-v2-posthoc-ungated` and
writes only below `artifacts/vamp-logt-mlp-permuted-mnist-ungated/`. The
original artifact root is read-only calibration evidence.
