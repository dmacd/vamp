# LogT-VAMP behavioral router on the VAMP-AF Rotated-MNIST task

## Question and interpretation

This successor asks whether the integrated behavioral router that was tested on
Permuted-MNIST also works on the observable, label-changing contexts from the
completed VAMP-AF experiment. It keeps the LogT binary-counter adapter hierarchy
and the complete router/replay matrix. It replaces only the task stream.

This is not a learned router over the VAMP-AF spatial tree. The completed AF
tree, its PCA-median decisions, and its artifacts remain unchanged. “VAMP-AF
task” means the exact five transformed data contexts used by that experiment.

## Frozen task boundary

- Authenticate VAMP-AF run
  `c3ad77df09fde94a75e2464450c21486d632bf4f60afe44c9602c6a86acf61af`
  and frozen CNN checkpoint
  `45793341113b7a44b397d8781b0590f7dcc54ca05ca2cd7d637b11244033a282`.
- Use rotations 0, 18, 36, 54, and 72 degrees with bilinear interpolation,
  no expansion, and zero fill.
- Shift labels by 0, 2, 4, 6, and 8 modulo ten in the matching contexts.
- Recompute the exact balanced 10,000 source identities selected by VAMP-AF
  with seed zero and require their little-endian int64 SHA-256 to equal
  `37179656c4e9b9ba6e8ff82b77941196cb6a87bbc5f02160f319f19254e5d908`.
- Preserve VAMP-AF’s blocked context order. For the 64-step primary run, use
  13, 13, 13, 13, and 12 consecutive steps. This uses at most 8,320 of the
  10,000 authenticated identities in any context.
- For smoke, reproduce VAMP-AF’s balanced 1,000-identity/context subset and use
  one 640-example macro-step per context. The five steps still exercise carries
  at steps two and four while opening every transform.

Within a context, use VAMP-AF’s seed/context shuffle. Allocate each macro-step’s
first 256 examples to adapter training, the next 256 to current router
supervision and future replay, and the final 128 to the untouched temporal
evaluation archive. The three allocations are disjoint. Test examples never
train either component.

## Controlled comparison

Do not retune the adapter, router, target temperature, replay budget, router
architecture, optimizer, seed set, test-subset size, or full checkpoints after
seeing Rotated-MNIST results. Carry forward the Permuted-MNIST primary settings:

- de-novo full-rank top-two adapters, 20 epochs per created node;
- five independent seeds 0 through 4;
- no-replay hard, example/range hard, and example/range soft routers;
- 256 historical examples for every eligible replay update;
- full evaluation and matched joint-IID references at steps 7, 15, 31, 63,
  and 64; and
- the same fixed most-recent, largest-range, uniform-active, and exhaustive
  label-aware oracle policies.

Labels, context IDs, rotation angles, label shifts, range endpoints, and time
indices remain outside the router input. The router observes only detached
per-node hidden states, output log probabilities, and active level bits.

## Gates and reporting

The smoke must pass finite metrics, exact replay budget, inactive-level masking,
nonnegative regret, decreasing router loss in a majority of updates, and exact
one-candidate parity before primary execution.

Use the original seven success criteria at high-active-node checkpoints 15,
31, and 63. Keep “substantially better” and “without materially degrading” as
descriptive judgments. Report target-specific disagreement rather than hiding
it behind the existential example/range hypothesis checks. In particular,
compare the best learned replay router with both fixed range policies, not only
with no replay.

Write a new content-addressed artifact tree below
`artifacts/vamp-logt-router-rotated-mnist/`. Bind the resolved configuration,
parent protocol and checkpoint, raw IDX files, this protocol, all material
source files, and the exact source-identity hash. Preserve chained metric
ledgers, commit-before-retirement checkpoints, exact resume, the complete
machine-readable summary, CSV, nine plots, Markdown, and standalone folding
HTML.
