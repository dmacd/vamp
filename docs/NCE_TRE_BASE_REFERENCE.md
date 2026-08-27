# NCE/TRE Base-Reference Protocol Correction

## Status and scope

This document defines the corrected reference distribution for the LogT
NCE/TRE MNIST experiment. It supersedes only the definition of the shared
reference distribution in
`Codex Handoff_ NCE-TRE Evidence Routing for LogT-VAMP on MNIST.md`. The LogT
topology, adapter capacity, raw-image evidence architecture, calibration,
candidate bridge counts, optimizer budgets, static gates, consolidation
controls, online comparisons, and work bounds remain unchanged.

The earlier run
`2003268ae73e22544cf9801d58b3fa40e724ff58c70bc31c32b120fdebf38b54`
used independent uniform pixel noise. It is retained as a completed negative
control, but it does not answer the intended base-reference question and must
not be presented as doing so.

## Correct shared reference

The corrected common reference \(Q\) is the uniform empirical distribution
over all 60,000 original, unrotated MNIST training images used to train the
authenticated frozen CNN. The frozen CNN was selected on a deterministic
50,000/10,000 split and then retrained on all 60,000 training images. The new
protocol binds the sealed base checkpoint, the sealed VAMP-AF protocol, the
original training-image IDX hash, the exact 60,000-image quantized tensor hash,
and the reference example count.

The frozen CNN is a discriminative classifier and is not itself treated as a
normalized image density. “Base reference” therefore means the empirical
image distribution on which that exact frozen classifier was trained. No
labels, context identifiers, CNN features, logits, or adapter responses enter
the evidence model.

## Waymark sampling

For each source image \(x\), sample one complete donor image \(z\) uniformly
with replacement from the 60,000-image reference bank. At waymark \(k\), draw
an independent Bernoulli mask for every coordinate using the fixed linear
replacement probability and return each source pixel where the mask is false
and the corresponding pixel from \(z\) where the mask is true. The paired
adjacent-waymark samples share the same donor image, preserving the protocol's
one-reference-example-per-source accounting while leaving both required
marginal distributions unchanged.

At the final endpoint the replacement probability is one, so the waymark is
exactly the complete sampled donor image. The endpoint is therefore exactly
\(Q\), retains the real joint pixel structure of base-training images, and is
identical for every temporal node and every stream time.

## Execution identity

The corrected strict configuration is
`configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml`, with protocol
revision `nce-tre-base-reference-v2` and reference identifier
`frozen_base_training_images_uint8`. It creates a new content-addressed run and
must never reuse or overwrite artifacts from the uniform-reference run.
