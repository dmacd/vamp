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

## What the worlds actually are

A TinyWorlds-P "world" is not a location, topic, cast, or coherent semantic
domain. It is one cell in a mechanically balanced grid over recovered recipe
metadata: one noun bucket crossed with one verb bucket. The buckets are
deliberately balanced by archive token mass, not clustered by meaning. In the
8x8 grid, a selected noun bucket contains 133 or 134 unrelated words and a
selected verb bucket contains 49 unrelated words. In the 6x6 fallback, those
counts rise to 178 nouns and 66 verbs. For example, the same 8x8 noun bucket
contains `apple`, `costume`, `gym`, `police`, `spirit`, and `work`; one verb
bucket contains `act`, `enter`, `joke`, `ride`, `steal`, and `zip`.

The five cells form a 2x2 corner, A/B/C/D, plus an unrelated cell E:

| Grid | A | B | C | D | E |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8x8 | `N6 x V2` | `N7 x V2` | `N7 x V3` | `N6 x V3` | `N2 x V4` |
| 6x6 | `N1 x V2` | `N2 x V2` | `N2 x V3` | `N1 x V3` | `N3 x V0` |

A world split therefore consists of ordinary released TinyStories whose
recovered noun and verb happen to fall in that cell. The base is trained on
the other cells. It still sees every selected noun bucket paired with other
verb buckets and every selected verb bucket paired with other noun buckets;
only the selected conjunctions are withheld. The model receives story text
alone. It never receives the recipe, bucket number, world label, source label,
or archive coordinate.

Each matched control is also ordinary held-in validation text. Half of its
groups keep the world's noun bucket but use another verb bucket; half keep the
world's verb bucket but use another noun bucket. Controls are globally
no-replacement and matched to their world on source, requested feature
signature, adjective bucket, length bin, active-token mass, and mean length.
The intended comparison isolates the missing noun/verb conjunction rather
than a broad genre or difficulty shift.

## Representative validation samples

The complete [validation sample appendix](TW-P_ARCHIVE_VALIDATION_SAMPLES.md)
contains 32 full stories covering every condition used in calibration on both
grids: held-in base; worlds A-E; and, because every control is a two-arm
mixture, both the same-noun-row and same-verb-column arm of controls A-E. It
does not read or display sealed-test data.

Selection was fixed without reading story semantics. For each condition, the
generator selected a canonical story at the lower-median token length; for
each control it selected one such story from each arm. Every displayed story
was reconstructed directly from its persisted text-shard coordinates and
verified against its exact-story SHA-256 before invisible trailing line
whitespace was removed for Markdown. The two held-in examples are `helpless /
goat / dare` on 8x8 and `peaceful / attic / choose` on 6x6, where each triple
is adjective / noun / verb. The generated appendix SHA-256 is
`d12464979c3b3a8e41777b18d324e8ed77ab1103a5b24d521abf5beaba3ad537`.

The following compact index shows the world and control recipes. The appendix
contains their full text, provenance, hashes, condition sizes, complete world
topologies, and deterministic samples from every selected bucket.

| Grid/world | World sample | Same-noun-row control | Same-verb-column control |
| :--- | :--- | :--- | :--- |
| 8x8 A | `gloomy / touch / bow` | `fake / comet / bake` | `ancient / folder / mail` |
| 8x8 B | `long / present / belong` | `dizzy / pin / sing` | `dependable / view / observe` |
| 8x8 C | `great / scarf / stir` | `perfect / pedal / have` | `huge / task / zoom` |
| 8x8 D | `modern / pastel / stir` | `safe / rainbow / scream` | `tough / log / receive` |
| 8x8 E | `original / tap / promise` | `thin / infant / bow` | `unknown / glue / accept` |
| 6x6 A | `rude / teach / pour` | `bright / towel / accept` | `cheap / pit / run` |
| 6x6 B | `pale / song / raise` | `tidy / log / yawn` | `bright / chalk / examine` |
| 6x6 C | `stubborn / order / suffer` | `careless / bee / reveal` | `bald / pencil / spin` |
| 6x6 D | `scared / circle / open` | `charming / cooler / please` | `elderly / onion / scare` |
| 6x6 E | `thin / diary / organize` | `soft / village / shine` | `yummy / toy / laugh` |

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

## Why the measured gap is so small in practical terms

The epoch-two gap has a direct perplexity interpretation because
`gap = NLL(world) - NLL(control)`: `exp(gap)` is the ratio of world perplexity
to control perplexity. The result was not merely below the gate; the practical
penalty was tiny:

| Result | Mean gap (nats/token) | World/control perplexity ratio | Relative penalty |
| :--- | ---: | ---: | ---: |
| 8x8 epoch 2 | 0.008017 | 1.00805 | 0.805% |
| 6x6 epoch 2 | 0.002802 | 1.00281 | 0.281% |
| Minimum accepted mean | 0.080000 | 1.08329 | 8.329% |

Four concrete properties explain why the intended signal can be this weak:

1. **The worlds are heterogeneous lexical bins.** A cell combines hundreds of
   unrelated noun/verb possibilities; it does not establish a recurring
   setting, relation, plot, or task that would make its stories cohere as a
   distinct distribution.
2. **The components are not unseen.** Base training removes only selected
   noun-bucket/verb-bucket conjunctions. It supplies abundant stories using
   the same noun bucket with other verbs and the same verb bucket with other
   nouns, so ordinary language-model composition can bridge the held-out cell.
3. **The recipe signal is not a clean semantic operator in the text.** The
   appendix's 8x8-A exemplar labels `bow` as its recovered verb, but the story
   principally realizes Bow as a dog's name. The 6x6-A exemplar uses `teach`
   as the recovered noun and realizes it as “a teach.” These are valid released
   records under the mechanical recipe contract, but they show why withholding
   metadata cells need not withhold a consistent textual relation.
4. **The controls intentionally remove easier differences.** Validation
   world/control token mass is exact in seven of ten comparisons and differs
   by at most the declared 0.25% tolerance in the other three. The allocator
   also matches source, requested features, adjective bucket, and length. World
   and control samples consequently share the same short TinyStories cadence
   and broad narrative patterns; the conjunction is almost the only designed
   difference.

The fallback reinforces this reading. Moving from 8x8 to 6x6 made each held-out
cell larger, yet reduced the mean penalty, and worlds C and E became slightly
easier than their controls. That is inconsistent with a strong, uniform
missing-conjunction effect. The samples make the mechanism plausible, but they
are not a token-level causal attribution; the run did not persist per-token
variance, confidence intervals, or ablations that could apportion the residual
gap among lexical composition and imperfect nuisance matching.

## Why the acceptance range was 0.08-0.30

The honest provenance is that this interval was a pre-run engineering
heuristic, not an empirically calibrated confidence interval. Section 9 of
the [original planning PDF](<TinyStories - partitioned.pdf>), imported with the
TinyWorlds-P implementation in commit `52b609d`, calls 0.08-0.30 a reasonable
initial go/no-go heuristic and explicitly says the proposed thresholds are
engineering thresholds rather than established benchmark constants. The
repository contains no preceding pilot distribution, statistical-power
calculation, standard-error estimate, or external benchmark that derives the
exact 0.08, 0.30, or 0.05 values.

Their practical design interpretation is:

| Gate | Perplexity interpretation | Intended decision role |
| :--- | :--- | :--- |
| Mean gap at least 0.08 | Worlds at least 8.33% harder than matched controls on average | Reject a negligible conjunction effect before investing in the continual-learning experiment. |
| Every world at least 0.05 | Every world at least 5.13% harder than its control | Prevent one or two unusually hard cells from hiding ineffective worlds in the mean. |
| Mean gap at most 0.30 | Worlds no more than 34.99% harder on average | Avoid a grossly over-separated task; the frozen policy responds with a finer 10x10 grid. |

The lower-bound interpretation follows directly from the stated goal of
detecting a meaningful held-out conjunction. The upper-bound rationale is an
inference from the prescribed 10x10 regrid for an “excessively large” gap: it
was meant to keep the task challenging without turning it into a severe
distribution shift. The source plan does not justify 0.30 more precisely than
that.

Precommitting these imperfect heuristics still served one important purpose:
it prevented validation results from moving the goalposts. It does not make
the interval a universal scientific constant. A successor benchmark should
derive its range before training from pilot/null gap distributions, uncertainty
estimates, and a declared minimum effect of practical interest. That work would
define a new benchmark version; it cannot retroactively change this result.

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
- The deterministic sample generator reads the two grids concurrently in about
  1.2 seconds, verifies all 32 displayed stories against their persisted
  exact-story hashes, and reproduces the appendix byte-for-byte without reading
  a sealed-test index.
- The intended report, appendix, generator, and roadmap diff passes scoped
  whitespace checks.
- The completed opt-in archive acceptance run previously rebuilt the full 8x8
  tree byte-identically with 24 workers and a different external-sort run
  size. Its 39-minute gate remains opt-in and was not converted into a default
  test.
- The completed real RTX 4090 resume smoke previously measured an 8.695 GiB
  allocator peak and proved interrupted/resumed update parity. It remains an
  explicit GPU gate rather than a default CPU test.
