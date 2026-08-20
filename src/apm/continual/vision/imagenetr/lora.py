"""Explicit timm linear LoRA injection and safetensors serialization."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator, Mapping
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import file_sha256
from apm.continual.vision.imagenetr.merging.common import LoRAFactors


class LoRALinear(nn.Module):
    """Frozen affine base plus a conventional scaled low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.0,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid LoRA rank, alpha, or dropout")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.reset_adapter(initialization_seed)

    @property
    def in_features(self) -> int:
        """Return the frozen affine input dimension."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Return the frozen affine output dimension."""
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        """Expose the frozen base weight for timm compatibility."""
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        """Expose the frozen base bias for timm compatibility."""
        return self.base.bias

    def reset_adapter(self, seed: int) -> None:
        """Create a deterministic fresh A factor and a zero-effect B factor."""
        if seed < 0:
            raise ValueError("adapter initialization seed must be nonnegative")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the frozen affine operation and scaled low-rank residual."""
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + self.scaling * residual

    def factors(self) -> LoRAFactors:
        """Return detached cloned factors suitable for immutable artifacts."""
        return LoRAFactors(
            self.lora_a.detach().clone(),
            self.lora_b.detach().clone(),
            self.scaling,
        )

    def load_factors(self, factors: LoRAFactors) -> None:
        """Load an equivalent update while preserving this module's fixed scale."""
        if factors.shape != (self.out_features, self.in_features) or factors.rank != self.rank:
            raise ValueError("adapter factors do not match the injected LoRA module")
        rescale = math.sqrt(factors.scale / self.scaling)
        with torch.no_grad():
            self.lora_a.copy_(factors.a.to(self.lora_a) * rescale)
            self.lora_b.copy_(factors.b.to(self.lora_b) * rescale)


def inject_vit_lora(
    backbone: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    initialization_seed: int = 0,
) -> nn.Module:
    """Inject LoRA into every ViT attention QKV and MLP fc1 projection."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 12:
        raise ValueError("the pinned ViT-B/16 must expose exactly 12 blocks")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for index, block in enumerate(blocks):
        if not isinstance(block.attn.qkv, nn.Linear) or not isinstance(block.mlp.fc1, nn.Linear):
            raise ValueError("unexpected timm ViT projection surfaces")
        block.attn.qkv = LoRALinear(
            block.attn.qkv,
            rank,
            alpha,
            dropout,
            initialization_seed + 2 * index,
        )
        block.mlp.fc1 = LoRALinear(
            block.mlp.fc1,
            rank,
            alpha,
            dropout,
            initialization_seed + 2 * index + 1,
        )
    return backbone


def iter_lora_layers(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield injected modules in stable state-dict name order."""
    yield from sorted(
        (
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        ),
        key=lambda item: item[0],
    )


def adapter_factors(model: nn.Module) -> dict[str, LoRAFactors]:
    """Export every injected module as detached scaled factors."""
    result = {name: layer.factors() for name, layer in iter_lora_layers(model)}
    if len(result) != 24:
        raise ValueError("the primary ViT adapter must contain exactly 24 matrices")
    return result


def load_adapter_factors(model: nn.Module, factors: Mapping[str, LoRAFactors]) -> None:
    """Load a complete exact-key adapter state into the injected ViT."""
    layers = dict(iter_lora_layers(model))
    if set(layers) != set(factors):
        raise ValueError("adapter artifact modules differ from the injected model")
    for name in sorted(layers):
        layers[name].load_factors(factors[name])


def trainable_lora_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return only A/B parameters, with no frozen base parameters."""
    return tuple(
        parameter
        for _, layer in iter_lora_layers(model)
        for parameter in (layer.lora_a, layer.lora_b)
    )


def save_adapter(path: str | Path, factors: Mapping[str, LoRAFactors]) -> str:
    """Serialize an exact complete adapter to safetensors and return its SHA-256."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    if not factors:
        raise ValueError("cannot serialize an empty adapter")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: value.detach().to(device="cpu").contiguous()
        for module, factor in sorted(factors.items())
        for key, value in (
            (f"{module}.lora_a", factor.a),
            (f"{module}.lora_b", factor.b),
        )
    }
    metadata = {
        "scales": json.dumps(
            {module: factor.scale for module, factor in sorted(factors.items())},
            separators=(",", ":"),
            sort_keys=True,
        ),
        "schema_version": "imagenetr50-lora-safetensors-v1",
    }
    save_file(tensors, target, metadata=metadata)
    return file_sha256(target)


def load_adapter(path: str | Path) -> dict[str, LoRAFactors]:
    """Load and validate a complete ImageNet-R LoRA safetensors artifact."""
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    source = Path(path)
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema_version") != "imagenetr50-lora-safetensors-v1":
            raise ValueError("unknown adapter safetensors schema")
        scales = json.loads(metadata.get("scales", "{}"))
        keys = tuple(handle.keys())
        modules = tuple(sorted(key[: -len(".lora_a")] for key in keys if key.endswith(".lora_a")))
        if not modules or set(keys) != {
            key for module in modules for key in (f"{module}.lora_a", f"{module}.lora_b")
        } or set(scales) != set(modules):
            raise ValueError("adapter safetensors keys are incomplete")
        return {
            module: LoRAFactors(
                handle.get_tensor(f"{module}.lora_a"),
                handle.get_tensor(f"{module}.lora_b"),
                float(scales[module]),
            )
            for module in modules
        }


__all__ = [
    "LoRALinear",
    "adapter_factors",
    "inject_vit_lora",
    "iter_lora_layers",
    "load_adapter",
    "load_adapter_factors",
    "save_adapter",
    "trainable_lora_parameters",
]
