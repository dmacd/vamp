"""Real dataset/model/GPU hard gates before smoke or full training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
import tempfile
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ManifestDataset,
    validate_prepared_dataset,
)
from apm.continual.vision.imagenetr.heads import load_classifier, save_classifier
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter,
    load_adapter_factors,
    save_adapter,
)
from apm.continual.vision.imagenetr.model import AdapterVisionModel, require_trainable_boundary


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Measured evidence that the resolved batch/model/dataset protocol is runnable."""

    device_name: str
    bf16_supported: bool
    zero_lora_max_absolute_error: float
    one_step_loss: float
    dataset_images_per_second: float
    peak_vram_bytes: int
    batch_size: int

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible hard-gate record."""
        return {
            "batch_size": self.batch_size,
            "bf16_supported": self.bf16_supported,
            "dataset_images_per_second": self.dataset_images_per_second,
            "device_name": self.device_name,
            "one_step_loss": self.one_step_loss,
            "peak_vram_bytes": self.peak_vram_bytes,
            "schema_version": "imagenetr50-preflight-v1",
            "zero_lora_max_absolute_error": self.zero_lora_max_absolute_error,
        }


def run_preflight(
    config: ImageNetRConfig,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
) -> PreflightResult:
    """Exercise BF16 parity, throughput, serialization, and one batch-64 optimizer step."""
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet-R primary preflight requires the local CUDA GPU")
    device = torch.device("cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support BF16")
    validate_prepared_dataset(prepared_root, manifest)
    rows = manifest.select("train", (0,))
    dataset = ManifestDataset(prepared_root, rows, train_transform, config.seed, 0)
    loader = DataLoader(
        dataset,
        batch_size=config.leaf_training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    started = time.monotonic()
    batch = next(iter(loader))
    throughput = batch[0].shape[0] / max(time.monotonic() - started, 1.0e-9)

    frozen = backbone_factory().to(device).eval()
    adapted = AdapterVisionModel(
        backbone_factory(),
        (0, 1, 2, 3),
        config.lora_rank,
        config.lora_alpha,
        config.lora_dropout,
        config.seed,
    ).to(device).eval()
    require_trainable_boundary(adapted)
    images = batch[0].to(device, non_blocking=True)
    labels = batch[1].to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        frozen_features = frozen.forward_head(frozen.forward_features(images[:2]), pre_logits=True)
        adapted_features = adapted.features(images[:2])
    parity_error = float(
        torch.max(torch.abs(frozen_features.float() - adapted_features.float())).item()
    )
    if parity_error > 2.0e-3:
        raise RuntimeError(f"zero-LoRA BF16 parity failed at {parity_error:.6g}")
    del frozen

    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.SGD(
        (
            {
                "params": [
                    parameter
                    for name, parameter in adapted.named_parameters()
                    if name.endswith("lora_a") or name.endswith("lora_b")
                ],
                "lr": config.leaf_training.lora_lr,
            },
            {
                "params": (adapted.classifier.weight, adapted.classifier.bias),
                "lr": config.leaf_training.head_lr,
            },
        ),
        momentum=config.leaf_training.momentum,
        weight_decay=config.leaf_training.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = adapted(images)
        loss = F.cross_entropy(logits, adapted.classifier.local_targets(labels))
    loss.backward()
    optimizer.step()
    peak = torch.cuda.max_memory_allocated(device)

    with tempfile.TemporaryDirectory(prefix="imagenetr50-preflight-") as temporary:
        root = Path(temporary)
        adapter_path, classifier_path = root / "adapter.safetensors", root / "head.safetensors"
        save_adapter(adapter_path, adapter_factors(adapted))
        save_classifier(classifier_path, adapted.classifier.rows())
        loaded_adapter, loaded_head = load_adapter(adapter_path), load_classifier(classifier_path)
        clone = AdapterVisionModel(
            backbone_factory(),
            loaded_head.class_ids,
            config.lora_rank,
            config.lora_alpha,
            config.lora_dropout,
            config.seed,
            loaded_head,
        ).to(device)
        load_adapter_factors(clone, loaded_adapter)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(
                clone(images[:2]).float(), adapted(images[:2]).float(), rtol=2e-3, atol=2e-3
            )
        del clone
    del adapted, images, labels
    torch.cuda.empty_cache()
    return PreflightResult(
        device_name=torch.cuda.get_device_name(device),
        bf16_supported=True,
        zero_lora_max_absolute_error=parity_error,
        one_step_loss=float(loss.detach().item()),
        dataset_images_per_second=throughput,
        peak_vram_bytes=int(peak),
        batch_size=config.leaf_training.batch_size,
    )


__all__ = ["PreflightResult", "run_preflight"]
