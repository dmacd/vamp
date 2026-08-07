# TinyWorlds Nouns-v2 Execution Report

## Phase 1 — Engine and disjoint partition complete

The isolated nouns-v2 implementation is complete through its CPU and real-source
partition gates. It adds independent v2 manifest, partition, audit, preset,
training-stage, result-row, run, judge, and report identities while retaining
the stabilized nouns-v1 training/evaluation mechanics through explicit
version-aware store and format bindings.

Authenticated parent inputs:

- nouns-v1 partition: `04ca2acf85f9505f0b7568b1696fbf290a8d2cbf78387dcfd6e815258fcc28b8`
- reviewed breakdown: `df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35`
- v2 manifest: `90d28b7cb9d34b4db23ab068019fb8c8923e8d8fabe64d301944e53c444df233`
- v2 partition: `210c4e2d067077fe774782024a594ade7e7472a986d554f186453549cf910f1b`

Measured partition counts:

- 2,210,934 clean base-universe stories (81.36% of original training)
- 44,286 deterministic internal base-validation stories
- 2,166,648 optimizer-visible base stories (79.73% of original training)
- 429,199 pure task-training stories
- 4,440 pure official-validation task/story pairs
- 77,361 excluded multi-task training stories
- 776 excluded multi-task validation stories
- 36 context-fitting probes for each of 24 tasks

The published partition occupies 505 MiB because it references the authenticated
parent story/token byte stores instead of duplicating them. Its two exclusion
ledgers and standalone JSON/Markdown/folding-HTML audit occupy 30 MiB. A second
complete pass over all 2,745,124 parent records reproduced the partition hash
byte-for-byte before GPU execution.

Verification so far:

- focused nouns-v1/v2 CPU suite: 25 passed
- opt-in exact real-source/reconstruction/v1-preservation gate: 2 passed
- nouns-v1 report Markdown SHA-256 remains
  `74f0035c755f95ff57624f8270615f3c9171f7b792d7b5ee22bae147ac15c4ae`
- nouns-v1 report HTML SHA-256 remains
  `9ef9cfea2c827836da8999c3d032fa63ff3a109664ee65b250066577a09d1526`
- nouns-v1 run manifest SHA-256 remains
  `fffa0e0f64f1c63ae3efb16363042907169737dc6858944cbcd0f46b178cc628`

## Phase 2 — GPU preflight complete

GPU preflight
`c4fda525322e00f2b271d351aa263fffda50dd238acd36b85b9709ceba36cc70`
passed on GPU 0, an RTX 4090. One representative seed-zero base update measured
6,945,854,720 allocator bytes (6.47 GiB), below the frozen 12 GiB limit. The
resource record projects 73,864 base microbatches per epoch, 18,466 accumulated
optimizer updates across two epochs, 48,000 adapter updates, 10,800 parent
node/story scores, 26,640 whole-story result rows, and 4,440 generation rows.

## Phase 3 — Fresh seed-zero base selected

The isolated nouns-v2 base trained from fresh seed-zero GPT-Neo parameters for
exactly 18,466 accumulated optimizer updates across two epochs. It did not load
or derive parameters from nouns-v1. The authenticated held-in NLL improved from
`1.358781703409687` after epoch one to `1.2703836058418045` after epoch two, so
the frozen quality gate passed with finite evidence and a strict second-epoch
improvement.

Selected-base identities and measurements:

- training: `94831c31c8f11a594534c2989182d378fc2e022382b61168b73e7400f9648e21`
- selection: `44ce3b0c420bb32df274e0e8ee45ddfa9244d9c7d8f959159927bfdd67ba4230`
- parameter checksum: `fff309bfbfcee8d59c5c3fc04152cc37be2142201f3bf9116b7b024e81a24f3c`
- measured allocator peak: 9,111,707,392 bytes (8.49 GiB), below 12 GiB
- authenticated resume coverage: full optimizer, RNG, epoch/batch cursor, and
  byte-stable loss trace under `tinyworlds-nouns-base-training-v2`

Next phase: ordered 24-stage VAMP training with 2,000 updates per task.
