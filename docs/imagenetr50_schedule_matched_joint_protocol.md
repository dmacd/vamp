# ImageNet-R-50 offline joint control with replay-matched optimization

This control asks whether ordinary all-data training can match the observed
uniform-replay endpoint when its optimizer schedule is held identical. It is
an exploratory follow-up to inspected test results, not an independently
validation-selected winner or a globally tuned joint-IID ceiling.

## Fixed intervention

The schedule source is the completed uniform H=4,096, standard thresholds,
historical target 0.8, interval unit 8 follow-up. Authenticate all fifty stage
results and their committed batch records, then freeze the ordered list of
56,243 batch sizes, totaling 844,640 training-image presentations. Source
stage boundaries become work checkpoints only; they do not restrict data or
the classifier. The control does not use source example identities, losses,
confidence values, old/new counts, or due times to choose its examples.

From the first update, every draw can select any of the same 24,000 training
images. Sample uniformly without replacement within each batch, independently
between batches. Use a counter-derived random generator keyed by seed and
global optimizer update. There is no task-conditioned mixture or mandatory
new-task first pass. Track each image's exposure ordinal to reproduce its
augmentation across resume; use the same augmentation family as replay.

Use the pinned frozen ViT-B/16, rank/alpha-16 QKV and fc1 adapters in all twelve
blocks, and the same ordinary affine classifier, with all 200 rows active
from the start. This is 1,327,104 LoRA plus 153,800 classifier parameters.
Cold seeds are 1993, 1994, and 1995. The seed-1993 initialization matches the
source's adapter and cold head rows, but exposes every row immediately.

Copy the source optimizer exactly: SGD momentum 0.9, weight decay 0.0005,
constant LoRA learning rate 0.0005 and head learning rate 0.01. Each batch
gets one update on its mean cross-entropy, including batches smaller than 64;
there is no gradient accumulation, clipping, warmup, or rate reduction.
The fixed stopping point is update 56,243, not a selected checkpoint.

## Measurements and limits

Keep a clean, hash-selected 2,048-image training probe at the fifty source work
boundaries. These diagnostics cannot stop training or select a checkpoint.
All models see the complete training set as their sampling population; there
is no new validation split or tuning phase. After all three fits finish,
evaluate only their final models on the same 6,000 test images, with the
existing all-200-class evaluator. Report accuracy and raw NLL per seed and
their means and sample standard deviations.

Compare the result with the source uniform endpoint, the previous joint
accuracy-selected endpoint, its five-epoch and terminal controls, and the
other existing replay conditions. Three new seeds share one source schedule
obtained from a single seed's SRT partner; this does not replicate the replay
arm or remove uncertainty in that schedule. The intervention changes both
the data/class curriculum and cumulative sample weighting while holding the
optimizer schedule fixed. It cannot isolate those remaining causes from one
another, and does not establish the best optimizer for either method.

Append the condition to the current SRT report, retaining every earlier curve
and condition. Show a distinct task-50-only mean/SD marker on accuracy/NLL
figures. Any diagnostic learning curves use updates or image presentations as
their horizontal coordinate; they are not causal continual-learning curves.

## Evidence and execution

Freeze source, schedule, data/model, software, code, and configuration hashes
before training. Save compact per-update sample/draw hashes and batch metrics
in checkpoint-committed chunks, plus per-image final exposure counts. Track
training forward/backward pairs, diagnostic/test forwards, optimizer updates,
and measured batch/checkpoint/evaluation work separately. Old models and
results remain untouched; no TRACE files change.

Use one nice-10 GPU worker and two bounded data-loader workers to limit host
RAM. Persist model, optimizer/momentum, exposure counters, RNG states, partial
work-checkpoint statistics, and committed chunk references atomically. Ignore
uncommitted orphan artifacts after interruption. A real mixed-batch preflight
must verify exact interrupted/uninterrupted tensors and zero-step reuse.
The completed workflow must also demonstrate fresh-process zero-step reuse.
Publish compact evidence and updated report artifacts, not model/optimizer
checkpoints or raw image data. Keep active-agent health checks during the run.
