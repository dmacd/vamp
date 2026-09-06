# ImageNet-R stage-31 total-parameter-matched joint-IID control

## Question

The stage-31 adaptive frontier trains five rank-16 node adapters and a
macro-token integrator. The earlier rank-80 joint-IID control matched only the
five adapters' aggregate LoRA parameter count. This control asks how much of
the remaining accuracy and negative-log-likelihood difference persists when a
single joint-IID adapter receives nearly the same total number of trainable
parameters as the complete adaptive frontier.

No completed artifact subsumes this condition. The rank-16 and rank-80 joint
controls contain 1,422,460 and 6,730,876 active parameters, respectively; the
adaptive frontier contains 18,691,016.

## Capacity match

Each unit of LoRA rank contributes 82,944 parameters across attention QKV and
MLP fc1 in all 12 ViT-B/16 blocks. The joint classifier necessarily contributes
95,356 more. The nearest integer rank to the adaptive frontier's 18,691,016
active parameters is therefore 224:

- joint rank-224 LoRA: 18,579,456 parameters;
- joint affine classifier: 95,356 parameters;
- joint total: 18,674,812 parameters;
- adaptive five-node LoRAs plus macro integrator: 18,691,016 parameters.

The joint control is 16,204 parameters smaller, a 0.087% shortfall. Rank 225
would be 66,740 parameters larger, a 0.357% overshoot, so rank 224 is the
closest realizable total-active-parameter match. Alpha is 224, retaining the
unit `alpha / rank` multiplier used by every preceding control.

Parameter count does not imply functional or compute equivalence. The
adaptive frontier starts from five separately pretrained rank-16 adapters,
runs five ViT paths, and combines their token sequences with a macro
transformer. The joint control starts one rank-224 adapter from the standard
zero-effect initialization, runs one ViT path, and applies one affine
classifier.

## Data and optimization

The control inherits the authenticated rank-80 control's exact 12,194 fit and
3,049 validation identities for tasks 1-31. Only rank and alpha change: five
epochs, physical batch 64, SGD momentum 0.9, weight decay 5e-4, LoRA learning
rate 5e-4, classifier learning rate 1e-2, deterministic epoch order, online
training augmentation, and center-crop evaluation remain fixed. The fixed
epoch-five endpoint is primary; the minimum-validation-NLL epoch is a
convergence diagnostic.

The adaptive full-history frontier at epoch five is the direct comparator.
Both conditions see 60,970 image presentations. The rank-16 and rank-80 joint
controls remain in the report to separate the effect of matching aggregate
node-LoRA capacity from the additional parameter budget occupied by the macro
integrator.

## Isolation and persistence

The new protocol binds the immutable parent frontier result, completed rank-80
control, split, model, dataset, image identities, configuration, material code,
and installed environment. It never opens a test image. It writes epoch
history and resumable checkpoints incrementally. A completed invocation
authenticates the model directory, history, protocol, rank-80 source, and
result without constructing a ViT or taking an optimizer step. It updates the
existing stage-31 report rather than creating a separate report.
