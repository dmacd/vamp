# TinyWorlds-P Archive v1 Calibration Report

Status: controlled scientific stop on 2026-07-22.

This is the current archive-only result for benchmark
`tinyworlds-p-archive-v1`. It is not the historical corpus-intersection audit.
The run used only eligible entities from the pinned
`TinyStories_all_data.tar.gz` archive and initialized both model attempts from
seed-zero parameters. No historical TinyWorlds-P checkpoint was loaded,
resumed, selected, or used as a baseline.

## Outcome

The initial 8x8 calibration learned the archive distribution but produced a
mean world/control gap below the declared pass interval. That outcome triggered
the one allowed fresh 6x6 fallback. The fallback again learned the archive
distribution, passed its quality and memory gates, and again produced gaps that
were too small. The predeclared one-fallback policy therefore ended the
milestone without epochs 3-5, checkpoint selection, sealed-test access, or a
base publication.

This is a valid negative result, not an interrupted run. The fixed runner
returned status 2 after writing both fallback validation records and the final
calibration decision.

## Source and partition identities

- Archive SHA-256:
  `26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699`.
- Initial 8x8 partition:
  `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`.
- Initial strict `tree.json` SHA-256:
  `ce0b76d6275f695977be5eb365c20c498efacd390d14b61a4282e644a293f829`.
- Fresh 6x6 fallback partition:
  `7bf90c70ca7207d8b0fdd7896eed7a2ae019bbcbd74126cfcc2115ae0759b4fb`.
- Fallback strict `tree.json` SHA-256:
  `0d9818a4949a4d46f2598a79877420d792d592f35edb51427817b8479322c373`.
- Fallback held-in split weights: 94/3/3.
- Fallback held-in active tokens: 764,363,966 train; 24,393,576
  validation; 24,393,567 sealed test.
- Eligible source mass on both grids: 4,966,067 records and 945,499,161
  active tokens.

The 8x8 partition was built and independently reproduced byte-for-byte before
GPU work. The 6x6 partition was freshly allocated from the same authenticated
archive universe after the 8x8 decision; it did not mutate the 8x8 artifact.

## Initial 8x8 calibration

- Training identity:
  `165580211b3aea895564d3c5d58ceee82254dcb055ef2566c987c70a3ddd629d`.
- Exact epoch states: epoch 1 at update 18,832 and epoch 2 at update 37,664.
- Peak JAX allocation: 9,189,799,168 bytes (8.559 GiB).
- Decision: `fallback_6x6`.

| Epoch | Held-in NLL | Mean gap | A | B | C | D | E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.261707 | 0.008421 | 0.017214 | 0.008893 | 0.009470 | 0.000787 | 0.005740 |
| 2 | 1.201706 | 0.008017 | 0.016777 | 0.008202 | 0.008427 | 0.002115 | 0.004565 |

Held-in NLL improved by 0.060001, and the allocator stayed below the 12 GiB
limit. The mean gap was below 0.08 and every per-world gap was below 0.05, so
the declared low-gap fallback was mandatory.

Persisted evidence hashes:

- `validation.jsonl`:
  `6f9097eeda838b91fedebba6b332560b1feea424499915a3f0f5887410b64505`.
- `calibration.json`:
  `c4d58f91fcd534990871df09005f4bb1b3505d1451f443884a1dd1dc44de0c0e`.

## Fresh 6x6 fallback calibration

- Training identity:
  `fca14275bc154f8e498f30acb4b37e30d52aebd05a01ed3e01f9b23bf4511427`.
- Exact epoch states: epoch 1 at update 17,200 and epoch 2 at update 34,400.
- Peak JAX allocation: 9,418,418,432 bytes (8.772 GiB).
- Decision: `fallback_6x6`, which on the already-consumed single fallback means
  stop without publication.

| Epoch | Held-in NLL | Mean gap | A | B | C | D | E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.267558 | 0.002252 | 0.009129 | 0.007281 | -0.002891 | 0.002993 | -0.005250 |
| 2 | 1.206720 | 0.002802 | 0.008364 | 0.009118 | -0.003093 | 0.004483 | -0.004860 |

Gate audit at epoch 2:

- Held-in NLL at most 2.2: pass (`1.206720`).
- Improvement of at least 0.02: pass (`0.060838`).
- Allocator peak at most 12 GiB: pass (`8.772 GiB`).
- Mean gap in `[0.08, 0.30]`: fail (`0.002802`).
- Every per-world gap at least 0.05: fail (all five are below 0.05).

Persisted evidence hashes:

- `validation.jsonl`:
  `4f95a31705a57fb217679fe1263dd574949c260498141a45bc8eb56ab258e313`.
- `calibration.json`:
  `da2c7650546156f90e9964de72b4a2fb3a62aedce30203ab1397d46723b4e77b`.

## Sealed-test and publication audit

The fallback work directory contains `progress.jsonl`, `validation.jsonl`,
`calibration.json`, immutable periodic states, and the two epoch states. It
contains no `sealed-test.json` and no publication staging directory. The
checkpoint namespace contains only the `work/` directory; there is no
`checkpoints/tinyworlds-p-archive-v1/<training-sha256>/` publication.

Because the second grid failed the representation-gap gate, epochs 3-5,
best-eligible-epoch selection, fixed-prompt sampling, and sealed-test evaluation
were ineligible. Opening test or publishing a base would have violated the
benchmark contract.

## Interpretation

Both grids fit held-in archive stories well, but withholding a noun-bucket by
verb-bucket conjunction did not create the required validation loss separation
from matched held-in controls. Tightening from 8x8 to 6x6 reduced rather than
increased the mean gap. Under the frozen archive-v1 hypothesis, this is evidence
that the selected bucket-level conjunction task is not sufficiently separated
for the intended base/world experiment. It is not evidence for changing the
gate after observing validation, adding another fallback, or consulting sealed
test. Any new topology or gap hypothesis requires a new versioned benchmark.

## Final verification

- The focused 82-test archive/core/partition/training/GPT-Neo/checkpoint/state
  scope passed as four concurrent CPU jobs in 9.8 seconds. The slowest group
  took 9.6 seconds.
- Every collected parked TinyWorlds-v2 test was collection-marked skipped in
  0.7 seconds; no V2 test body ran.
- Package import resolves `TinyStories_all_data.tar.gz` as the canonical source
  and the canonical preset as 8x8.
- A scoped search over TinyWorlds-P source, runners, focused tests, and real
  integration code finds no legacy corpus identity, path, offset, occurrence,
  join, unmatched, hash-match, corpus-coverage, or aggregate filename symbol.
- Replaying both persisted validation pairs through
  `calibration_grid_decision` reproduces `fallback_6x6` exactly.
- The intended documentation and V2-skip diff passes `git diff --check`.
- The completed opt-in archive acceptance run previously rebuilt the full 8x8
  tree byte-identically with 24 workers and a different external-sort run
  size. Its 39-minute gate remains opt-in and was not converted into a default
  test.
- The completed real RTX 4090 resume smoke previously measured an 8.695 GiB
  allocator peak and proved interrupted/resumed update parity. It remains an
  explicit GPU gate rather than a default CPU test.
