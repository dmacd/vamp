Question: With the artifacts from our initial run, can we easily construct a comparable task-known result by doing something simple like looking at the relative fraction of each task's represenation in each of the level nodes and choosing the best one based on that? Would this be a valid/fair comparison?

----------------------------------------------------------------------------------------------------------------------------------
Answer:

Yes. **We can construct this from the existing artifacts with essentially zero GPU work**, and I think it is a valid task-known control—provided we define the mapping rule before looking at its performance.

The evidence bundle already contains the complete candidate generations for every active adapter × evaluation example, and the hierarchy lineage tells us exactly which arrivals each live node represents. Adapter tensors are not needed for this.  The hierarchy nodes explicitly store their arrival intervals, and each arrival is 100 examples.

### I would define the router slightly differently from your wording

For known task (t) and live node (n), let

[
C(t,n)=
\frac{#{\text{arrivals of task }t\text{ represented by }n}}
{#{\text{arrivals belonging to task }t}}.
]

Then select

[
n^*(t)=\arg\max_n C(t,n).
]

In other words, select the node containing the **largest fraction of that task**, rather than the node whose contents have the largest fraction belonging to that task.

That distinction matters. A one-arrival leaf from task 8 is 100% pure task 8, but it represents only 20% of task 8's training data. A two-arrival node is also 100% pure but contains 40% of the task. The latter is clearly the more sensible task-known choice.

For ties I'd predeclare something mechanical like:

1. greatest task coverage;
2. greatest node purity, (P(t|n));
3. greatest `end_arrival` — the most recent node.

No scores, validation labels, prompt contents, or test labels enter that decision.

At the final stage, for example, our seven live nodes deterministically represent:

[
[1!:!16], [17!:!24], [25!:!32],
[33!:!36], [37!:!38], [39], [40].
]

Since the TRACE tasks occupy arrivals 1–5, 6–10, 11–15, etc., that rule would map approximately:

| Known task  | Selected node | Task arrivals inside |
| ----------- | ------------- | -------------------: |
| C-STANCE    | 1–16          |                  5/5 |
| FOMC        | 1–16          |                  5/5 |
| MeetingBank | 1–16          |                  5/5 |
| Py150       | 17–24         |                  4/5 |
| ScienceQA   | 17–24         |                  4/5 |
| NumGLUE-cm  | 25–32         |                  5/5 |
| NumGLUE-ds  | 33–36         |                  3/5 |
| 20Minuten   | 37–38         |                  2/5 |

That is actually a nice illustration of what Log-t consolidation does: task identity does **not** trivially give us a dedicated task adapter, because the temporal hierarchy has cut across task boundaries.

### Is it fair against CRAFT?

**Yes, information-wise. In fact it gives us less routing sophistication than CRAFT.**

CRAFT explicitly stores a **task-to-intervention table** during training. At inference, when it is given a prompt from a previously seen task, it retrieves the intervention assigned to that task. ([arXiv][1]) Its task→intervention assignment was itself learned from a warm-up representation and output-distribution divergence. ([arXiv][1])

Our proposed control would instead say:

> “You tell me the task ID, and I mechanically choose the live VAMP node containing the largest amount of training history from that task.”

We're not using the answer, prompt, validation accuracy, or an oracle. So I would be comfortable calling that a **task-known provenance router** and comparing it directly to CRAFT's task-known results.

There is one caveat: this task→node lookup metadata is an (O(#\text{tasks})) convenience. We should **not** use this condition when making the task-free/O(log T) system claim. It's explicitly a control where task identity is provided, just as it is for CRAFT.

### In fact, we should report two task-known VAMP numbers

We already have a somewhat stronger condition called `task_aware`. It looks at the validation set and chooses the candidate that actually scores best for each task; the implementation does exactly that.

I'd distinguish:

**`task_known_provenance`**
Task ID → node with greatest represented-task coverage. No performance data involved.

**`task_known_validation`** — our existing `task_aware`
Task ID → validation-best live node.

CRAFT's mechanism is conceptually between those: it knows the task and establishes a task→intervention mapping using task data during training, but it doesn't post-hoc search every live intervention on the validation set. So the provenance result is the especially conservative apples-to-apples comparison; the validation-best result tells us what the existing hierarchy can do with a well-chosen task-specific lookup.

And this is **very easy to calculate now**: reread the cached candidate JSONLs, reconstruct the active intervals at each of the eight stages, apply the deterministic mapping above, and rescore the already-generated test predictions. No model weights or inference are required.

I would compute it for **all six VAMP conditions**. It could be quite informative whether the repaired SVD hierarchy's provenance mapping lands close to its existing 38.18 validation-selected task-aware result or substantially below it. If close, then task identity nearly solves addressing by itself; if far apart, it tells us that “which node contains the task” and “which node actually retained that task best” diverge significantly after consolidation.

[1]: https://arxiv.org/abs/2605.05732 "CRAFT: Forgetting-Aware Intervention-Based Adaptation for Continual Learning"
