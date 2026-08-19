"""Artifact-level SVD and Core+TSV consolidation jobs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, publish_immutable_json
from apm.continual.trace.adapter_io import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_FILENAME,
    load_adapter_tensors,
)
from apm.continual.trace.merging.common import (
    LoRAFactors,
    factors_from_peft_state,
    peft_state_from_factors,
    weighted_child_weights,
)
from apm.continual.trace.merging.core_tsv import CoreTsvResult, merge_module_states as core_merge_modules
from apm.continual.trace.merging.svd_mean import merge_module_states as svd_merge_modules
from apm.continual.trace.protocol import MergePolicy


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Paths and content hashes emitted by one parameter-space merge."""

    adapter_path: Path
    adapter_sha256: str
    diagnostics_path: Path
    core_cache_path: Path | None


def consolidate_adapters(
    left_adapter: str | Path,
    right_adapter: str | Path,
    left_example_count: int,
    right_example_count: int,
    policy: MergePolicy,
    output_directory: str | Path,
    *,
    retain_precompress: bool = False,
    device: str | torch.device = "cpu",
) -> ConsolidationResult:
    """Merge two immutable PEFT adapter files into one rank-bounded parent."""
    left_path, right_path = Path(left_adapter), Path(right_adapter)
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError("both child adapter artifacts are required")
    weights = weighted_child_weights((left_example_count, right_example_count))
    started = time.monotonic()
    child_scales = tuple(_adapter_scale(path) for path in (left_path, right_path))
    children = tuple(
        factors_from_peft_state(
            {
                name: tensor.to(device)
                for name, tensor in load_adapter_tensors(path).items()
            },
            scale=scale,
        )
        for path, scale in zip((left_path, right_path), child_scales)
    )
    if set(children[0]) != set(children[1]):
        raise ValueError("child adapters expose different target modules")
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    core_cache_path: Path | None = None
    if policy.method == "svd_mean_r8":
        merged, diagnostics = svd_merge_modules(
            children,
            weights,
            policy.output_rank,
            policy.parent_scale,
        )
        diagnostic_records = {
            module: diagnostic.as_record() for module, diagnostic in diagnostics.items()
        }
    else:
        if policy.core_scale is None:
            raise ValueError("Core TSV policy omitted its scale")
        results = core_merge_modules(
            children,
            policy.core_scale,
            policy.output_rank,
            policy.parent_scale,
            frozenset(children[0]) if retain_precompress else frozenset(),
        )
        merged = {module: result.factors for module, result in results.items()}
        diagnostic_records = {
            module: result.diagnostics.as_record() for module, result in results.items()
        }
        core_cache_path = target / "core_cache.safetensors"
        _save_core_cache(results, core_cache_path)
    adapter_path = target / ADAPTER_FILENAME
    _save_tensor_file(peft_state_from_factors(merged), adapter_path)
    diagnostics_path = publish_immutable_json(
        target / "merge_metrics.json",
        {
            "child_adapter_sha256": [file_sha256(left_path), file_sha256(right_path)],
            "child_example_counts": [left_example_count, right_example_count],
            "child_lora_scales": list(child_scales),
            "child_weights": list(weights),
            "format": "trace-merge-metrics-v1",
            "module_diagnostics": diagnostic_records,
            "input_singular_spectra": {
                module: [
                    _factor_singular_values(child[module]) for child in children
                ]
                for module in sorted(children[0])
            },
            "module_similarity": {
                module: _similarity_record(children[0][module], children[1][module])
                for module in sorted(children[0])
            },
            "merge_configuration": policy.merge_record(),
            "merge_config_hash": policy.merge_config_hash,
            "merge_wall_seconds": time.monotonic() - started,
            "precompress_retained": retain_precompress,
            "validation_metrics": {
                "post_compression": None,
                "post_repair": None,
                "pre_compression": None,
                "precompression_selected": retain_precompress,
                "status": "pending-external-validation",
            },
        },
    )
    publish_immutable_json(
        target / "adapter_config.json",
        {
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "bias": "none",
            "inference_mode": True,
            "lora_alpha": policy.parent_alpha,
            "lora_dropout": 0.1,
            "peft_type": "LORA",
            "r": policy.output_rank,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        },
    )
    return ConsolidationResult(
        adapter_path=adapter_path,
        adapter_sha256=file_sha256(adapter_path),
        diagnostics_path=diagnostics_path,
        core_cache_path=core_cache_path,
    )


def _save_core_cache(results: Mapping[str, CoreTsvResult], path: Path) -> None:
    tensors = {
        key: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for module, result in sorted(results.items())
        for key, tensor in (
            (f"{module}.left_basis", result.left_basis),
            (f"{module}.merged_core", result.merged_core),
            (f"{module}.right_basis", result.right_basis),
            *(
                (
                    (f"{module}.precompress_A", result.precompress_factors.a),
                    (f"{module}.precompress_B", result.precompress_factors.b),
                )
                if result.precompress_factors is not None
                else ()
            ),
        )
    }
    _save_tensor_file(tensors, path)


def materialize_precompress_adapter(
    core_cache_path: str | Path,
    output_directory: str | Path,
    policy: MergePolicy,
) -> tuple[Path, int, int]:
    """Materialize a selected rank-up-to-16 Core diagnostic as a temporary adapter."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("TRACE Core diagnostics require safetensors") from error
    cache = load_file(Path(core_cache_path), device="cpu")
    a_tensors = {
        key.removesuffix(".precompress_A"): value
        for key, value in cache.items()
        if key.endswith(".precompress_A")
    }
    b_tensors = {
        key.removesuffix(".precompress_B"): value
        for key, value in cache.items()
        if key.endswith(".precompress_B")
    }
    if not a_tensors or set(a_tensors) != set(b_tensors):
        raise ValueError("Core cache does not contain a complete precompression diagnostic")
    ranks = {tensor.shape[0] for tensor in a_tensors.values()}
    if len(ranks) != 1:
        raise ValueError("Core precompression ranks differ between modules")
    rank = next(iter(ranks))
    alpha = int(round(policy.parent_scale * rank))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _save_tensor_file(
        {
            key: tensor
            for module in sorted(a_tensors)
            for key, tensor in (
                (f"{module}.lora_A.weight", a_tensors[module]),
                (f"{module}.lora_B.weight", b_tensors[module]),
            )
        },
        output / ADAPTER_FILENAME,
    )
    publish_immutable_json(
        output / "adapter_config.json",
        {
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "bias": "none",
            "inference_mode": True,
            "lora_alpha": alpha,
            "lora_dropout": 0.1,
            "peft_type": "LORA",
            "r": rank,
            "target_modules": ["q_proj", "v_proj"],
            "task_type": "CAUSAL_LM",
        },
    )
    return output / ADAPTER_FILENAME, rank, alpha


def _save_tensor_file(tensors: Mapping[str, Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("TRACE consolidation requires safetensors") from error
    save_file(
        {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in tensors.items()
        },
        path,
        metadata={"format": "pt"},
    )


def _adapter_scale(adapter_path: Path) -> float:
    """Read the exact PEFT LoRA scale carried by an immutable child adapter."""
    config_path = adapter_path.parent / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"child adapter config is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = int(config["r"])
        alpha = float(config["lora_alpha"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid child adapter config: {config_path}") from error
    if rank <= 0 or alpha <= 0.0:
        raise ValueError(f"invalid LoRA rank/alpha in {config_path}")
    if bool(config.get("use_rslora", False)):
        raise ValueError("TRACE consolidation does not support rank-stabilized LoRA")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("TRACE consolidation requires uniform LoRA rank and alpha")
    return alpha / rank


def _similarity_record(left: LoRAFactors, right: LoRAFactors) -> dict[str, float]:
    left_norm = _factor_norm(left)
    right_norm = _factor_norm(right)
    inner = float(
        (
            left.scale
            * right.scale
            * torch.trace(
                left.a.to(torch.float32).T
                @ (left.b.to(torch.float32).T @ right.b.to(torch.float32))
                @ right.a.to(torch.float32)
            )
        )
        .cpu()
        .item()
    )
    denominator = left_norm * right_norm
    return {
        "cosine": 0.0 if denominator == 0.0 else inner / denominator,
        "left_frobenius_norm": left_norm,
        "right_frobenius_norm": right_norm,
    }


def _factor_norm(factors: LoRAFactors) -> float:
    gram_b = factors.b.to(torch.float32).T @ factors.b.to(torch.float32)
    gram_a = factors.a.to(torch.float32) @ factors.a.to(torch.float32).T
    squared = factors.scale**2 * torch.trace(gram_b @ gram_a)
    return float(torch.sqrt(torch.clamp(squared, min=0.0)).cpu().item())


def _factor_singular_values(factors: LoRAFactors) -> list[float]:
    left_basis, left_triangular = torch.linalg.qr(
        factors.b.to(torch.float32), mode="reduced"
    )
    right_basis, right_triangular = torch.linalg.qr(
        factors.a.to(torch.float32).T, mode="reduced"
    )
    del left_basis, right_basis
    values = torch.linalg.svdvals(left_triangular @ right_triangular.T) * factors.scale
    return [float(value) for value in values.to(device="cpu", dtype=torch.float64)]


__all__ = [
    "ConsolidationResult",
    "consolidate_adapters",
    "materialize_precompress_adapter",
]
