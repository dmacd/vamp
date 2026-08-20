"""Pinned timm ViT construction, LoRA model composition, and activation capture."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from apm.continual.artifacts import file_sha256
from apm.continual.vision.imagenetr.constants import (
    TIMM_MODEL_FILENAME,
    TIMM_MODEL_NAME,
    TIMM_MODEL_REPOSITORY,
    TIMM_MODEL_REVISION,
    TIMM_MODEL_SHA256,
)
from apm.continual.vision.imagenetr.heads import AffineClassifier, ClassifierRows
from apm.continual.vision.imagenetr.lora import LoRALinear, inject_vit_lora


def download_pinned_checkpoint(cache_directory: str | Path) -> Path:
    """Download the public immutable timm checkpoint and verify its exact bytes."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("huggingface_hub is required by the vision environment") from error
    path = Path(
        hf_hub_download(
            repo_id=TIMM_MODEL_REPOSITORY,
            filename=TIMM_MODEL_FILENAME,
            revision=TIMM_MODEL_REVISION,
            cache_dir=Path(cache_directory),
        )
    )
    if file_sha256(path) != TIMM_MODEL_SHA256:
        raise ValueError("downloaded timm checkpoint differs from the pinned SHA-256")
    return path


def create_pinned_backbone(checkpoint_path: str | Path) -> nn.Module:
    """Create the explicit ViT-B/16 architecture and strictly load pinned weights."""
    try:
        import timm
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("timm and safetensors are required by the vision environment") from error
    checkpoint = Path(checkpoint_path)
    if file_sha256(checkpoint) != TIMM_MODEL_SHA256:
        raise ValueError("local backbone checkpoint SHA-256 changed")
    backbone = timm.create_model(TIMM_MODEL_NAME, pretrained=False)
    state = load_file(checkpoint, device="cpu")
    incompatible = backbone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("pinned checkpoint does not strictly match the timm architecture")
    backbone.reset_classifier(0)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


class AdapterVisionModel(nn.Module):
    """Frozen pinned ViT, 24 explicit LoRA matrices, and represented affine rows."""

    def __init__(
        self,
        backbone: nn.Module,
        class_ids: Sequence[int],
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        initialization_seed: int = 0,
        initial_rows: ClassifierRows | None = None,
    ) -> None:
        super().__init__()
        self.backbone = inject_vit_lora(
            backbone, rank, alpha, dropout, initialization_seed
        )
        feature_dim = int(getattr(backbone, "num_features", 0))
        if feature_dim != 768:
            raise ValueError("the pinned ViT-B/16 feature dimension must be 768")
        self.classifier = AffineClassifier(
            class_ids, feature_dim, initialization_seed + 10_000, initial_rows
        )

    def features(self, images: Tensor) -> Tensor:
        """Return the pinned timm pre-logit class-token representation."""
        tokens = self.backbone.forward_features(images)
        return self.backbone.forward_head(tokens, pre_logits=True)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw affine represented-class logits."""
        return self.classifier(self.features(images))

    def cosine_scores(self, images: Tensor, scale: float) -> Tensor:
        """Return bias-free normalized scores for task-free calibration control."""
        return self.classifier.cosine_scores(self.features(images), scale)


def capture_adapted_inputs(
    model: AdapterVisionModel,
    images: Tensor,
) -> dict[str, Tensor]:
    """Capture one forward's QKV/fc1 input activations in stable module order."""
    if any(
        torch.count_nonzero(layer.lora_b.detach()).item() != 0
        for _name, layer in model.named_modules()
        if isinstance(layer, LoRALinear)
    ):
        raise ValueError("output-drift proxy inputs must come from the zero-adapter frozen base")
    captured: dict[str, Tensor] = {}
    handles = tuple(
        layer.register_forward_pre_hook(
            lambda _module, inputs, name=name: captured.__setitem__(
                name, inputs[0].detach().to(device="cpu", dtype=torch.float32)
            )
        )
        for name, layer in model.named_modules()
        if isinstance(layer, LoRALinear)
    )
    try:
        with torch.inference_mode():
            model.features(images)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 24:
        raise ValueError("activation capture did not observe all 24 adapted matrices")
    return dict(sorted(captured.items()))


def require_trainable_boundary(model: AdapterVisionModel) -> None:
    """Fail unless exactly LoRA and represented classifier parameters are trainable."""
    permitted = {
        name
        for name, _ in model.named_parameters()
        if name.endswith("lora_a")
        or name.endswith("lora_b")
        or name in {"classifier.weight", "classifier.bias"}
    }
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != permitted or len(permitted) != 50:
        raise ValueError("trainable parameter boundary includes frozen ViT state")


__all__ = [
    "AdapterVisionModel",
    "capture_adapted_inputs",
    "create_pinned_backbone",
    "download_pinned_checkpoint",
    "require_trainable_boundary",
]
