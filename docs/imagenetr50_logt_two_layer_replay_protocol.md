# ImageNet-R-50 two-layer node-latent ablation

This experiment changes only the representation available to the persistent
integrator. It retains the replay-adaptation v6 hierarchy, split, eight-cell
sampler/weighting/optimizer matrix, H=8,192 budget, four epochs per arrival,
diagnostic stages 31 and 50, and three-restart fresh full-history fits.

Each active level slot retains the existing LoRA-adapted final 768-dimensional
pre-classifier class token. It adds the same node's 768-dimensional class token
captured after the penultimate transformer block, before the final transformer
block and the backbone's final normalization. Thus the added representation
contains the base computation plus that node's LoRA updates through the first
11 of 12 transformer blocks. Both latent vectors receive independent per-image
layer normalization.

The existing per-slot feature prefix remains unchanged: final latent, raw
affine scores, within-node log probabilities, classifier ownership, and active
bit. The penultimate latent is appended, increasing each slot from 1,369 to
2,137 values and the six-slot input from 8,214 to 12,822. The expanded input
layer copies every v6-compatible weight into the corresponding old-feature
column and initializes every new penultimate-latent column to zero. All middle
and output layers copy the v6 initialization exactly. This makes the expanded
initial model a nested extension rather than an unrelated random restart.

The completed replay-adaptation v6 result is an authenticated, immutable
single-layer comparison. Conditions are selected by validation evidence only;
the locked test remains unavailable until all online and fresh full-history
models are sealed. Large behavior caches and checkpoints remain local. The
committed evidence consists of manifests, compact training/evaluation ledgers,
tables, plots, and Markdown, standalone HTML, and PDF reports.
