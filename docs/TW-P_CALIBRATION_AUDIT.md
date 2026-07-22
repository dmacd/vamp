# Historical TinyWorlds-P v1 Calibration Stop Audit

## Status: historical and ineligible for resume

Effective 2026-07-21, TinyWorlds-P draws every split directly from eligible
entities in `TinyStories_all_data.tar.gz`. The partition and training run below
used the now-abandoned intersection of `TinyStories-train.txt` with the archive.
They remain immutable diagnostic evidence, but they are not publication
candidates, do not constrain the archive-only grid decision, and must not be
resumed or selected. Archive-only calibration starts from fresh seed-zero
parameters after a new partition is built and validated.

The superseded corpus-intersection run produced an internally valid 8×8
partition and then stopped under its predeclared second-grid failure rule. It
did not publish a base checkpoint and did not open the sealed test split.

## 8×8 calibration

- Partition SHA-256:
  `8925869844f4641ecab0aa377165fcff189625e789be5a8cbf7c5a939067d4ab`
- Training SHA-256:
  `46050b6323b054928e29a60cd395b7325174552c3578390d476e06a864a31b67`
- Working evidence:
  `checkpoints/tinyworlds-p-v1/work/base-v1-_pc1qu81/grid-8-training/`
- Two-epoch training runtime: approximately 1 hour 40 minutes.
- Allocator peak: 7,491,143,424 bytes (6.976 GiB), below the 12 GiB gate.

| Epoch | Held-in NLL | Mean gap |
|---:|---:|---:|
| 1 | 1.596230 | 0.000545 |
| 2 | 1.493838 | -0.000128 |

Held-in NLL improved by 0.102392, so quality and learning-progress gates
passed. The epoch-2 per-world results were:

| World | World NLL | Control NLL | Gap |
|:---:|---:|---:|---:|
| A | 1.489593 | 1.515530 | -0.025936 |
| B | 1.516733 | 1.455456 | 0.061277 |
| C | 1.504103 | 1.536035 | -0.031932 |
| D | 1.497004 | 1.464204 | 0.032801 |
| E | 1.491891 | 1.528739 | -0.036848 |

The required mean gap was 0.08–0.30 and every world needed at least 0.05.
The low-gap rule therefore selected the one allowed fresh 6×6 fallback. The
8×8 parameters were not reused.

## 6×6 partition failure

The fallback authenticated and joined the same sources, passed coverage,
selected its topology, and completed deterministic split assignments. It then
failed before shard publication because globally no-replacement controls are
mathematically infeasible with the fixed 96/2/2 held-in split.

For the balanced 2×2 corner, each of two worlds sharing a noun row needs half
of its 10% world-validation groups from that row. Their combined row demand is
therefore approximately 10% of one cell's group mass. A 6×6 corner leaves four
held-in cells in the shared row, whose fixed 2% validation allocations provide
only approximately 8% of one cell's mass. The same deficit occurs for the two
shared verb columns. An 8×8 grid leaves six held-in cells, providing
approximately 12%, which is why its controls were feasible.

The exact group counts prove the deficit before nuisance matching or token
matching is considered:

| Split | Shared arm | Worlds | Available held-in groups | Required groups | Deficit |
|:---|:---|:---:|---:|---:|---:|
| Validation | noun row 0 | A/D | 3,027 | 3,814 | 787 |
| Validation | noun row 1 | B/C | 3,013 | 3,828 | 815 |
| Validation | verb column 1 | A/B | 3,057 | 3,828 | 771 |
| Validation | verb column 3 | C/D | 3,099 | 3,816 | 717 |
| Test | noun row 0 | A/D | 3,058 | 3,813 | 755 |
| Test | noun row 1 | B/C | 2,950 | 3,828 | 878 |
| Test | verb column 1 | A/B | 3,069 | 3,828 | 759 |
| Test | verb column 3 | C/D | 3,189 | 3,814 | 625 |

The concrete allocator stopped on world B validation after earlier controls
had reserved overlapping groups: its row arm required 1,919 groups and only
715 unused candidates remained. Different ordering cannot solve the aggregate
capacity inequalities above.

## Contract consequence

The frozen policy says a failing second partition ends the milestone without
final training. Consequently:

- no epoch 3–5 training was run;
- no final checkpoint was selected or published;
- sealed test was never opened;
- the valid 8×8 partition remains published;
- the complete 8×8 calibration states and failed 6×6 preparation evidence
  remain in the working directory above.

Possible recovery policies—changing 96/2/2, permitting control reuse, reducing
control size, or defining a different fallback grid—would each alter the
benchmark contract and require an explicit new decision. None was applied.
