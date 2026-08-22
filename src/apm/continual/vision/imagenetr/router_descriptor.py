"""Compact adapter descriptors and gauge-invariant R3 response kernels."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
import math

import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.vision.imagenetr.merging.common import LoRAFactors, compact_svd
from apm.continual.vision.imagenetr.proxy_memory import TensorCache
from apm.continual.vision.imagenetr.router_artifacts import InferenceNodeRef, RouterStore
from apm.continual.vision.imagenetr.router_config import RouterConfig


def selected_response_modules(config: RouterConfig) -> tuple[str, ...]:
    """Return the fixed numeric-depth order of R3 adapted projections."""
    suffix = {"attn.qkv": "attn.qkv", "mlp.fc1": "mlp.fc1"}
    return tuple(
        f"backbone.blocks.{block}.{suffix[target]}"
        for block in config.response_blocks
        for target in config.response_targets
    )


def descriptor_config_record(config: RouterConfig) -> dict[str, object]:
    """Return all material descriptor choices."""
    return {
        "dim": config.descriptor_dim,
        "metadata": [
            "level",
            "log_images",
            "log_classes",
            "first_task",
            "last_task",
            "span",
            "creation_stage",
            "leaf",
        ],
        "probe_distribution": "rademacher_normalized",
        "probe_dim": config.descriptor_probe_dim,
        "schema_version": "imagenetr50-router-descriptor-config-v1",
        "seed": config.descriptor_seed,
    }


def response_config_record(config: RouterConfig) -> dict[str, object]:
    """Return all material R3 response choices."""
    return {
        "blocks": list(config.response_blocks),
        "compact_svd": "fp32_reduced_qr_core_svd_v1",
        "dtype": config.response_dtype,
        "normalization": "log1p_delta_norm_over_input_norm_then_layer_norm",
        "pooling": "cls",
        "response_dim": len(config.response_blocks) * len(config.response_targets),
        "schema_version": "imagenetr50-router-response-config-v1",
        "targets": list(config.response_targets),
    }


def _rademacher(rows: int, columns: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(0, 2, (rows, columns), generator=generator, dtype=torch.int8)
    return (2.0 * values.to(torch.float32) - 1.0) / math.sqrt(rows)


def _module_seed(seed: int, module: str) -> int:
    return int(sha256(f"router-descriptor-v1\0{seed}\0{module}".encode()).hexdigest()[:16], 16)


def compact_update_energy(factors: LoRAFactors) -> Tensor:
    """Return the scaled dense Frobenius norm without materializing the update."""
    left = factors.b.to(torch.float64).T @ factors.b.to(torch.float64)
    right = factors.a.to(torch.float64) @ factors.a.to(torch.float64).T
    squared = factors.scale**2 * torch.sum(left * right.T)
    return torch.sqrt(torch.clamp(squared, min=0.0)).to(torch.float32)


def build_descriptor(
    adapter: Mapping[str, LoRAFactors],
    node: InferenceNodeRef,
    config: RouterConfig,
) -> Tensor:
    """Build the exact 96-sketch + 24-energy + 8-metadata descriptor."""
    if len(adapter) != 24 or set(adapter) != {
        f"backbone.blocks.{block}.{target}"
        for block in range(12)
        for target in ("attn.qkv", "mlp.fc1")
    }:
        raise ValueError("descriptor requires all 24 primary adapted matrices")
    values: list[Tensor] = []
    energies: list[Tensor] = []
    for module in sorted(adapter):
        factors = adapter[module]
        seed = _module_seed(config.descriptor_seed, module)
        output_probe = _rademacher(factors.shape[0], config.descriptor_probe_dim, seed)
        input_probe = _rademacher(factors.shape[1], config.descriptor_probe_dim, seed ^ 0x5DEECE66D)
        sketch = (
            output_probe.T
            @ factors.b.to(torch.float32)
            @ factors.a.to(torch.float32)
            @ input_probe
        ) * factors.scale
        values.append(sketch.reshape(-1))
        normalized = compact_update_energy(factors) / math.sqrt(
            factors.shape[0] * factors.shape[1]
        )
        energies.append(torch.log1p(normalized).reshape(1))
    artifact = node.artifact
    first, last = artifact.first_task, artifact.last_task
    task_span = last - first + 1
    creation_stage = last + 1 if artifact.level == 0 else last + 2**artifact.level
    metadata = torch.tensor(
        (
            artifact.level / 4.0,
            math.log1p(artifact.represented_train_image_count) / math.log1p(24_000),
            math.log1p(len(artifact.represented_class_ids)) / math.log1p(200),
            first / 49.0,
            last / 49.0,
            task_span / 50.0,
            creation_stage / 50.0,
            float(not artifact.parent_hashes),
        ),
        dtype=torch.float32,
    )
    descriptor = torch.cat((*values, *energies, metadata.reshape(-1)))
    if descriptor.shape != (config.descriptor_dim,) or not torch.isfinite(descriptor).all():
        raise ValueError("node descriptor has the wrong shape or non-finite values")
    return descriptor


def build_response_kernels(
    adapter: Mapping[str, LoRAFactors], config: RouterConfig
) -> dict[str, Tensor]:
    """Build canonical ``diag(s) V^T`` kernels for selected R3 modules."""
    result: dict[str, Tensor] = {}
    for module in selected_response_modules(config):
        if module not in adapter:
            raise ValueError(f"R3 response module is absent from the adapter: {module}")
        _left, singular, right = compact_svd(adapter[module])
        kernel = singular[:, None] * right
        if kernel.ndim != 2 or kernel.shape[1] != 768 or not torch.isfinite(kernel).all():
            raise ValueError("invalid canonical R3 response kernel")
        result[module] = kernel.contiguous()
    return result


def response_features(
    cls_activations: Mapping[str, Tensor],
    kernels: Mapping[str, Tensor],
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Return gauge-invariant normalized update-response features per image."""
    if tuple(cls_activations) != tuple(kernels):
        raise ValueError("R3 activations and kernels cover different modules or order")
    columns = []
    for module in kernels:
        activation = cls_activations[module].to(torch.float32)
        kernel = kernels[module].to(device=activation.device, dtype=torch.float32)
        update_norm = torch.linalg.vector_norm(activation @ kernel.T, dim=-1)
        input_norm = torch.linalg.vector_norm(activation, dim=-1)
        columns.append(torch.log1p(update_norm / torch.clamp(input_norm, min=epsilon)))
    result = torch.stack(columns, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("R3 response features are non-finite")
    return result


@dataclass(frozen=True, slots=True)
class NodeRouterFeatures:
    """Cached fixed node descriptor and optional R3 response kernels."""

    descriptor: Tensor
    response_kernels: dict[str, Tensor]
    descriptor_sha256: str
    response_kernel_sha256: str | None

    def to(self, device: torch.device) -> NodeRouterFeatures:
        """Move runtime tensors while preserving their immutable file identities."""
        return NodeRouterFeatures(
            self.descriptor.to(device),
            {name: value.to(device) for name, value in self.response_kernels.items()},
            self.descriptor_sha256,
            self.response_kernel_sha256,
        )


def _cache_tensor_sha(cache: TensorCache, values: Mapping[str, object]) -> str:
    path = cache.root / cache.key(values) / "tensors.safetensors"
    return file_sha256(path)


def load_or_build_node_features(
    store: RouterStore,
    node: InferenceNodeRef,
    config: RouterConfig,
    include_response: bool,
) -> NodeRouterFeatures:
    """Content-address and reuse descriptor/response state by inference hash."""
    bundle = node.load()
    descriptor_values = {
        "config_hash": record_sha256(descriptor_config_record(config)),
        "inference_node_hash": node.node_hash,
    }
    descriptor_cache = TensorCache(
        store.run / "descriptors", "imagenetr50-router-descriptor-cache-v1"
    )
    descriptor_tensors, _ = descriptor_cache.get_or_compute(
        descriptor_values,
        lambda: {"descriptor": build_descriptor(bundle.adapter, node, config)},
    )
    kernels: dict[str, Tensor] = {}
    kernel_sha: str | None = None
    if include_response:
        response_values = {
            "config_hash": record_sha256(response_config_record(config)),
            "inference_node_hash": node.node_hash,
        }
        response_cache = TensorCache(
            store.run / "response_kernels",
            "imagenetr50-router-response-kernel-cache-v1",
        )
        tensors, _ = response_cache.get_or_compute(
            response_values,
            lambda: {
                f"kernel_{index:02d}": value
                for index, value in enumerate(
                    build_response_kernels(bundle.adapter, config).values()
                )
            },
        )
        kernels = dict(zip(selected_response_modules(config), tensors.values()))
        kernel_sha = _cache_tensor_sha(response_cache, response_values)
    return NodeRouterFeatures(
        descriptor_tensors["descriptor"],
        kernels,
        _cache_tensor_sha(descriptor_cache, descriptor_values),
        kernel_sha,
    )


__all__ = [
    "NodeRouterFeatures",
    "build_descriptor",
    "build_response_kernels",
    "compact_update_energy",
    "descriptor_config_record",
    "load_or_build_node_features",
    "response_config_record",
    "response_features",
    "selected_response_modules",
]
