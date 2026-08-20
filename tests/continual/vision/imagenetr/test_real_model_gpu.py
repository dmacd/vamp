"""Opt-in gates for the pinned checkpoint and local accelerator."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import os

import pytest
import torch
from torch.nn import functional as F

from apm.continual.vision.imagenetr.lora import iter_lora_layers
from apm.continual.vision.imagenetr.config import load_config
from apm.continual.vision.imagenetr.data import (
    image_transforms,
    load_dataset_manifest,
    validate_prepared_dataset,
)
from apm.continual.vision.imagenetr.model import (
    AdapterVisionModel,
    create_pinned_backbone,
    require_trainable_boundary,
)
from apm.continual.vision.imagenetr.preflight import run_preflight


def _checkpoint() -> Path:
    configured = os.environ.get("IMAGENETR50_CHECKPOINT")
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
    cache = Path("data/imagenetr50/model_cache")
    matches = tuple(cache.rglob("model.safetensors")) if cache.is_dir() else ()
    if len(matches) != 1:
        pytest.skip("the revision-pinned ImageNet-R checkpoint is not cached")
    return matches[0]


@pytest.mark.integration
def test_pinned_checkpoint_strict_load_and_exact_adapter_surface() -> None:
    model = AdapterVisionModel(create_pinned_backbone(_checkpoint()), (0, 1, 2, 3))
    require_trainable_boundary(model)
    layers = tuple(iter_lora_layers(model))
    assert len(layers) == 24
    assert tuple(name for name, _layer in layers) == tuple(sorted(name for name, _layer in layers))
    assert sum(name.endswith("attn.qkv") for name, _layer in layers) == 12
    assert sum(name.endswith("mlp.fc1") for name, _layer in layers) == 12


@pytest.mark.integration
def test_authenticated_dataset_has_exact_isolated_hardlinked_split() -> None:
    prepared = Path("data/imagenetr50/imagenet-r")
    manifest_path = prepared / "dataset_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("the authenticated ImageNet-R split is not prepared")
    manifest = load_dataset_manifest(manifest_path)
    validate_prepared_dataset(prepared, manifest)
    train = manifest.select("train")
    test = manifest.select("test")
    assert len(train) == 24_000 and len(test) == 6_000
    assert {row.image_id for row in train}.isdisjoint(row.image_id for row in test)
    source = Path("data/imagenetr50/source/imagenet-r")
    assert all(
        (source / row.source_relative_path).stat().st_ino
        == (prepared / row.prepared_relative_path).stat().st_ino
        for row in manifest.images
    )


@pytest.mark.benchmark
def test_local_gpu_bf16_batch64_forward_backward_gate() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible; run this benchmark outside the filesystem sandbox")
    if not torch.cuda.is_bf16_supported():
        pytest.fail("the visible CUDA device does not support BF16")
    device = torch.device("cuda:0")
    model = AdapterVisionModel(
        create_pinned_backbone(_checkpoint()),
        (0, 1, 2, 3),
        initialization_seed=1993,
    ).to(device)
    require_trainable_boundary(model)
    images = torch.rand(64, 3, 224, 224, device=device)
    labels = torch.arange(64, device=device) % 4
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5.0e-4,
        momentum=0.9,
    )
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert torch.cuda.max_memory_allocated(device) < torch.cuda.get_device_properties(device).total_memory


@pytest.mark.benchmark
def test_real_dataset_model_serialization_and_throughput_preflight() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible; run this benchmark outside the filesystem sandbox")
    config = load_config("configs/vision/imagenetr/primary.yaml")
    prepared = Path("data/imagenetr50/imagenet-r")
    if not (prepared / "dataset_manifest.json").is_file():
        pytest.skip("the authenticated ImageNet-R split is not prepared")
    manifest = load_dataset_manifest(prepared / "dataset_manifest.json")
    template = create_pinned_backbone(_checkpoint())
    train_transform, _test_transform = image_transforms(config.input_size)
    result = run_preflight(
        config,
        manifest,
        prepared,
        lambda: deepcopy(template),
        train_transform,
    )
    assert result.batch_size == 64
    assert result.bf16_supported
    assert result.dataset_images_per_second > 0.0
    assert result.zero_lora_max_absolute_error <= 2.0e-3
    assert result.peak_vram_bytes > 0
