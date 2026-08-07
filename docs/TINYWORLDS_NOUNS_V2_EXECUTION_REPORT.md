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

## Phase 4 — Ordered VAMP graph complete

VAMP completed all 24 stages and exactly 48,000 rank-eight/alpha-eight LoRA
updates in 49.3 minutes. Every stage was atomically committed under the v2-only
stage format after verifying that the new commit left every older edge tensor
unchanged. The completed artifact strict-loaded from stage 24 with:

- VAMP run: `c8c7eaa3560fdca19dd789b767f1b3dea829f95190b0a130983d15661965e2fa`
- final stage: `a050bb51141bb37002de10ba29d448f83cfc2ab864c480ddf594c682b727f0d8`
- adaptation manifest: `c762b966a5a23561e16b0bee5bb9d92b205153f9b89784ac06596d2ca0416887`
- final tensor checksum: `97414ac3d8656ab083b2e570a4162dc69b024f90cf819b80b1cab94213553e63`

The learned non-root parent graph is:

`mouse→root`, `rabbit→mouse`, `boat→mouse`, `brother→mouse`,
`parent→brother`, `duck→boat`, `sister→brother`, `pet→brother`,
`bicycle→boat`, `grandma→parent`, `lion→rabbit`, `fairy→mouse`,
`train→boat`, `cow→duck`, `wheel→duck`, `monkey→mouse`,
`princess→fairy`, `plane→boat`, `elephant→rabbit`, `neighbor→brother`,
`dragon→rabbit`, `queen→princess`, `horse→mouse`, and `bus→wheel`.

## Phase 5 — Whole-story NLL and routing complete

The bounded evaluator published exactly 26,640 canonical rows: all 4,440
validation pairs under all six conditions. Ledger SHA-256 is
`0a39bc33006d67f03054824e488767153a75e4c86e8224a3d157afa6f05459d2`.
Its story-weighted/token-weighted NLL and routing summaries are:

| condition | story NLL | token NLL | route accuracy | oracle regret |
|---|---:|---:|---:|---:|
| base | 1.636 | 1.636 | 0.0% | +0.159 |
| oracle | 1.477 | 1.499 | 100.0% | +0.000 |
| VAMP exhaustive | 1.472 | 1.492 | 83.6% | -0.006 |
| VAMP Hopfield | 1.560 | 1.575 | 38.8% | +0.083 |
| VAMP EBT uniform | 1.480 | 1.502 | 77.4% | +0.002 |
| VAMP EBT Hopfield | 1.487 | 1.508 | 70.7% | +0.010 |

## Phase 6 — Midpoint routing and generation complete

The prefix-isolated evaluator published exactly 4,440 canonical rows using the
stabilized KV-cache decoder and immediate bounded-chunk persistence. Ledger
SHA-256 is
`9ca842f394222d441b705c0cdf661e97a6edbd787cba095e49ed7270dc27eacf`.
The router saw only the first token half; the held-back suffix was consumed
after routing for NLL and as the equal-budget generation reference.

Midpoint task-free route accuracy was 73.9% exhaustive, 37.4% Hopfield, 70.4%
EBT uniform, and 64.5% EBT Hopfield. Corresponding true-suffix story NLL was
1.572, 1.615, 1.581, and 1.582, compared with 1.638 for base and 1.539 for the
oracle adapter.

## Phase 7 — Local reports and completion gates complete

The canonical default command reached `local_complete`, exited successfully,
and sent its desktop notification without calling OpenRouter. Published report
identities are:

- report identity: `1dbb44e821562079f946d21826278265890008b8d6979c96a9d2150aa940cea8`
- Markdown SHA-256: `c6b01feea266db74e5da04c870c36bb7c09eb23f9f30161a55e1be0e673ae86b`
- folding HTML SHA-256: `967acbabbc3bfe6b2936951d37358f67993aa7bf33b9b92c4644d9fb262246ac`
- local run identity: `9f614a2d018d7df51e49650a6a52507b88ddb963c151d011af915ac3f22677b2`

The standalone HTML contains strong, weak, correctly routed, and misrouted
examples for the v2 tasks and has no external script, stylesheet, or network
dependency. A second canonical invocation repeated parent authentication and
partition reconstruction, strict-loaded every completed model/result artifact,
performed no optimizer, NLL, or generation work, and reproduced all five
published ledger/report/run-manifest file hashes byte-for-byte.

Final verification:

- focused/shared CPU regression suite: 46 passed
- opt-in exact real-source/reconstruction/v1-preservation suite: 2 passed
- nouns-v1 Markdown, HTML, and run-manifest hashes remain exactly unchanged
- OpenRouter judgment remains optional behind `--judge` and can resume solely
  from the completed local generation ledger
