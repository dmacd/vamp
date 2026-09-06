# ImageNet-R stage-31 rank-matched joint-IID control

## Question

The stage-31 adaptive frontier executes five rank-16 node adapters. This
control asks whether its advantage over the existing rank-16 joint-IID model
comes from having five times as many LoRA parameters, before accounting for
the macro-token integrator or the cost of five ViT forwards.

No prior ImageNet-R artifact contains the required condition. The existing
same-split joint-IID references all use one rank-16 adapter.

## Capacity match

Each ViT-B/16 adapter changes 24 projections: attention QKV and MLP fc1 in all
12 transformer blocks. It has 82,944 trainable LoRA parameters per unit rank,
so one rank-16 adapter has 1,327,104 and five have 6,635,520. The control uses
one rank-80 adapter with exactly 6,635,520 LoRA parameters. Alpha is 80, which
keeps the LoRA multiplier `alpha / rank` equal to one, as in the rank-16
models. The 124-way affine classifier adds 95,356 trainable parameters.

This matches aggregate adapter rank and LoRA parameter count. It does not make
the models functionally or computationally equivalent. The adaptive frontier
applies five separately initialized and previously trained rank-16 updates in
five independent ViT passes, then uses a 12,055,496-parameter macro-token
integrator. The joint model applies one rank-80 update in one ViT pass.

## Data and optimization

The control reuses the parent run's exact 12,194 fit and 3,049 validation
identities for tasks 1-31. It changes only rank and alpha relative to the
existing joint-IID reference: five epochs, physical batch 64, SGD momentum
0.9, weight decay 5e-4, LoRA learning rate 5e-4, classifier learning rate
1e-2, deterministic epoch order, online training augmentation, and center-crop
evaluation all remain fixed. The primary value is the fixed epoch-five
endpoint. The minimum-NLL epoch is retained as a diagnostic.

The matched comparison is the adaptive full-history frontier at epoch five:
both models have seen 60,970 image presentations and expose 6,635,520
trainable LoRA parameters. Their other trainable parameters, initialization,
number of ViT forwards, and prior node-training work differ and must remain
visible in the report.

## Isolation and persistence

The rank-matched protocol binds the immutable parent result, split, model,
dataset, exact image identities, configuration, material code, and installed
environment. It never opens a test image. Epoch history and checkpoints are
written incrementally. A completed invocation validates the model directory,
history, protocol, and result without constructing a ViT or taking an
optimizer step. The new control extends the existing stage-31 report; it does
not create a replacement experiment report or rewrite the parent result.

