from pathlib import Path

import pytest
import torch

from apm.continual.vision.imagenetr.protocol import NodeArtifact
from apm.continual.vision.imagenetr.bank import LogicalNode
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.router_artifacts import InferenceNodeRef
from apm.continual.vision.imagenetr.router_config import load_router_config
from apm.continual.vision.imagenetr.router_descriptor import (
    build_descriptor,
    build_response_kernels,
    response_features,
    selected_response_modules,
)


PROJECT_ROOT = Path(__file__).parents[4]


def _config():
    return load_router_config(
        PROJECT_ROOT
        / "configs/vision/imagenetr/recursive_router_oracle_recovery_v1.yaml"
    )


def _node() -> InferenceNodeRef:
    artifact = NodeArtifact(
        run_hash="a" * 64,
        software_manifest_hash="b" * 64,
        git_commit="unit",
        creation_timestamp_utc="2026-08-21T00:00:00Z",
        level=0,
        first_task=0,
        last_task=0,
        represented_task_ids=(0,),
        represented_class_ids=(0, 1, 2, 3),
        represented_train_image_count=120,
        parent_hashes=(),
        unrepaired_parent_hash=None,
        consolidation_method="leaf",
        consolidation_config_hash="c" * 64,
        repair_config_hash="d" * 64,
        lora_sha256="e" * 64,
        classifier_sha256="f" * 64,
        proxy_image_ids=(),
        repair_image_ids=(),
        source_priority_hash="1" * 64,
        training_optimizer_steps=1,
    )
    return InferenceNodeRef(LogicalNode(0, 0, 0), artifact.content_hash, Path("."), artifact, "2" * 64)


def _adapter(in_features: int = 11) -> dict[str, LoRAFactors]:
    generator = torch.Generator().manual_seed(4)
    return {
        f"backbone.blocks.{block}.{target}": LoRAFactors(
            torch.randn(3, in_features, generator=generator),
            torch.randn(7 if target == "attn.qkv" else 5, 3, generator=generator),
            0.5,
        )
        for block in range(12)
        for target in ("attn.qkv", "mlp.fc1")
    }


def test_descriptor_is_exactly_128_deterministic_and_never_calls_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, adapter, node = _config(), _adapter(), _node()
    monkeypatch.setattr(
        LoRAFactors,
        "dense",
        lambda _self: (_ for _ in ()).throw(AssertionError("dense delta forbidden")),
    )
    first = build_descriptor(adapter, node, config)
    second = build_descriptor(adapter, node, config)
    assert first.shape == (128,)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_r3_response_matches_dense_delta_and_is_gauge_invariant() -> None:
    config = _config()
    generator = torch.Generator().manual_seed(12)
    adapter, equivalent = {}, {}
    for module in selected_response_modules(config):
        a = torch.randn(4, 768, generator=generator)
        b = torch.randn(9, 4, generator=generator)
        q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
        adapter[module] = LoRAFactors(a, b, 0.75)
        equivalent[module] = LoRAFactors(q @ a, b @ q.T, 0.75)
    activations = {
        module: torch.randn(6, 768, generator=generator)
        for module in selected_response_modules(config)
    }
    kernels = build_response_kernels(adapter, config)
    transformed = build_response_kernels(equivalent, config)
    first = response_features(activations, kernels)
    second = response_features(activations, transformed)
    torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-5)
    for column, module in enumerate(selected_response_modules(config)):
        dense = adapter[module].dense()
        expected = torch.log1p(
            torch.linalg.vector_norm(activations[module] @ dense.T, dim=-1)
            / torch.linalg.vector_norm(activations[module], dim=-1)
        )
        torch.testing.assert_close(first[:, column], expected, rtol=2e-5, atol=2e-5)
