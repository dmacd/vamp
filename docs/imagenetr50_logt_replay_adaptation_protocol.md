# ImageNet-R-50 replay-adaptation diagnosis

This experiment determines whether the persistent node-adapted integrator is
learning the old-task distribution or only memorizing its retained historical
examples. It is a post-hoc diagnosis on the established ImageNet-R split, not a
new publication claim.

The prior ImageNet-R run used a permanent-priority historical reservoir. That
made each stage exactly reproducible but unnecessarily reused nearly the same
examples. The Permuted-MNIST integrator instead seeded its uniform replay draw
with the macro-step, producing a fresh deterministic random subset at every
arrival. Deterministic rotation preserves exact resume behavior and the same
fixed-H, cumulative O(T log T) work bound.

The online matrix crosses two historical samplers (static and stage-keyed
rotating), two objectives (ordinary example-uniform cross-entropy and equal
total loss weight per seen task), and two AdamW policies (carry moments or reset
moments at every arrival). Every condition uses H=8,192, the same per-node
LoRA-adapted 768-dimensional pre-classifier latents, the same initialization,
the same four epochs per arrival, and the same fresh-parent hierarchy.

Online integrators train only on the 19,200-image fit partition. At stages 31
and 50, the workflow records accuracy by task on the selected replay rows, the
complete fit population, the 4,800-image integrator validation partition, and
the locked test partition. The validation partition was seen by the frozen
LoRA nodes during their earlier all-train fitting, but never by these integrator
optimizers.

Fresh full-history integrators at stages 31 and 50 use every fit row, three
independent restarts, and validation-loss checkpoint selection. They are
diagnostic architecture/optimization ceilings and do not satisfy the online
work constraint. Locked test features remain unavailable until every online
condition and full-history selection is sealed.

Large feature caches, optimizer checkpoints, and model weights remain local.
The committed evidence consists of protocol manifests, selection and behavior
ledgers, fit/validation/test tables, plots, and Markdown, standalone HTML, and
PDF reports.
