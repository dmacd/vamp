"""Dataset-agnostic model architectures for addressed-memory experiments."""

from apm.models.mlp_vae import VaeConfig, VaeParams, init_mlp_vae_params, vae_forward
from apm.models.vae_losses import (
    bernoulli_bce_with_logits,
    digit_region_bce,
    digit_region_flat_mask,
    flatten_canvases,
    label_patch_flat_mask,
    negative_elbo,
    per_example_negative_elbo,
    standard_normal_kl,
)

__all__ = [
    "VaeConfig",
    "VaeParams",
    "bernoulli_bce_with_logits",
    "digit_region_bce",
    "digit_region_flat_mask",
    "flatten_canvases",
    "init_mlp_vae_params",
    "label_patch_flat_mask",
    "negative_elbo",
    "per_example_negative_elbo",
    "standard_normal_kl",
    "vae_forward",
]
