"""Deterministic finite-presentation LoRA and classifier training core."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Iterable, Sequence
import math
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.vision.imagenetr.checkpoints import (
    atomic_torch_save,
    load_training_checkpoint,
)
from apm.continual.vision.imagenetr.config import TrainingConfig
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter_factors,
    trainable_lora_parameters,
)
from apm.continual.vision.imagenetr.model import AdapterVisionModel, require_trainable_boundary
from apm.continual.vision.imagenetr.merging.common import LoRAFactors


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Final immutable trainables plus measured finite-work resource totals."""

    adapter: dict[str, LoRAFactors]
    classifier: ClassifierRows
    optimizer_steps: int
    image_presentations: int
    wall_seconds: float
    final_loss: float
    peak_vram_bytes: int


def deterministic_epoch_order(length: int, seed: int, epoch: int) -> tuple[int, ...]:
    """Return a stable full permutation independent of worker scheduling."""
    if length < 1 or seed < 0 or epoch < 0:
        raise ValueError("epoch order inputs must be nonnegative and nonempty")
    derived = int(
        sha256(f"imagenetr50-epoch-v1\0{seed}\0{epoch}".encode()).hexdigest()[:8], 16
    )
    return tuple(int(value) for value in np.random.RandomState(derived).permutation(length))


def _checkpoint_record(
    model: AdapterVisionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    optimizer_steps: int,
    image_presentations: int,
    final_loss: float,
) -> dict[str, object]:
    return {
        "adapter": {
            key: tensor
            for module, factors in adapter_factors(model).items()
            for key, tensor in (
                (f"{module}.a", factors.a.cpu()),
                (f"{module}.b", factors.b.cpu()),
                (f"{module}.scale", torch.tensor(factors.scale)),
            )
        },
        "classifier_bias": model.classifier.bias.detach().cpu(),
        "classifier_class_ids": list(model.classifier.class_ids),
        "classifier_weight": model.classifier.weight.detach().cpu(),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "epoch": epoch,
        "final_loss": final_loss,
        "image_presentations": image_presentations,
        "next_batch": next_batch,
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "schema_version": "imagenetr50-training-checkpoint-v1",
    }


def _restore_checkpoint(
    model: AdapterVisionModel,
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, object],
) -> tuple[int, int, int, int, float]:
    from apm.continual.vision.imagenetr.merging.common import LoRAFactors

    raw = checkpoint["adapter"]
    if not isinstance(raw, dict):
        raise ValueError("checkpoint adapter state is malformed")
    modules = tuple(sorted(key[: -len(".a")] for key in raw if key.endswith(".a")))
    factors = {
        module: LoRAFactors(
            raw[f"{module}.a"], raw[f"{module}.b"], float(raw[f"{module}.scale"].item())
        )
        for module in modules
    }
    load_adapter_factors(model, factors)
    rows = ClassifierRows(
        tuple(int(value) for value in checkpoint["classifier_class_ids"]),
        checkpoint["classifier_weight"],
        checkpoint["classifier_bias"],
    )
    model.classifier.load_rows(rows)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(checkpoint["cpu_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["next_batch"]),
        int(checkpoint["optimizer_steps"]),
        int(checkpoint["image_presentations"]),
        float(checkpoint["final_loss"]),
    )


def train_adapter_model(
    model: AdapterVisionModel,
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    train_transform: object,
    config: TrainingConfig,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    active_class_ids: Iterable[int] | None = None,
    train_lora: bool = True,
    num_workers: int = 0,
    checkpoint_steps: int = 50,
    show_progress: bool = True,
) -> TrainingResult:
    """Train one resumable deterministic adapter/head job at optimizer boundaries."""
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("training jobs may consume only frozen training rows")
    if checkpoint_steps < 1:
        raise ValueError("checkpoint step interval must be positive")
    model.to(device)
    lora_parameters = trainable_lora_parameters(model)
    # A frozen-reference task intentionally disables these parameters.  Restore
    # the canonical boundary before validating the next independently trained
    # head, then apply the current job's train_lora choice below.
    for parameter in lora_parameters:
        parameter.requires_grad_(True)
    require_trainable_boundary(model)
    for parameter in lora_parameters:
        parameter.requires_grad_(train_lora)
    parameter_groups = [
        {"params": lora_parameters, "lr": config.lora_lr},
        {
            "params": (model.classifier.weight, model.classifier.bias),
            "lr": config.head_lr,
        },
    ]
    if not train_lora:
        parameter_groups = parameter_groups[1:]
    optimizer = torch.optim.SGD(
        parameter_groups,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    checkpoint = Path(checkpoint_path)
    epoch, next_batch, optimizer_steps, presentations, final_loss = 0, 0, 0, 0, math.nan
    if checkpoint.is_file():
        epoch, next_batch, optimizer_steps, presentations, final_loss = _restore_checkpoint(
            model, optimizer, load_training_checkpoint(checkpoint)
        )
    active = tuple(sorted(model.classifier.class_ids if active_class_ids is None else active_class_ids))
    if not active or not set(active) <= set(model.classifier.class_ids):
        raise ValueError("active training classes are outside the represented head")
    full_row = {class_id: index for index, class_id in enumerate(model.classifier.class_ids)}
    active_indices = torch.tensor([full_row[class_id] for class_id in active], device=device)
    inactive = tuple(
        class_id for class_id in model.classifier.class_ids if class_id not in frozenset(active)
    )
    frozen_inactive = model.classifier.selected_rows(inactive) if inactive else None
    active_lookup = torch.full((200,), -1, dtype=torch.long, device=device)
    active_lookup[torch.tensor(active, device=device)] = torch.arange(len(active), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    total_batches = config.epochs * math.ceil(len(rows) / config.batch_size)
    progress = tqdm(
        total=total_batches,
        initial=optimizer_steps,
        disable=not show_progress,
        desc="optimizer steps",
        unit="batch",
    )
    for current_epoch in range(epoch, config.epochs):
        order = deterministic_epoch_order(len(rows), seed, current_epoch)
        ordered_rows = tuple(rows[index] for index in order)
        dataset = ManifestDataset(
            prepared_root, ordered_rows, train_transform, seed, current_epoch
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0 if len(rows) < config.batch_size else min(num_workers, os_cpu_workers()),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        starting_batch = next_batch if current_epoch == epoch else 0
        model.train()
        for batch_index, (images, labels, _image_ids) in enumerate(loader):
            if batch_index < starting_batch:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)[:, active_indices]
                local_targets = active_lookup[labels]
                if torch.any(local_targets < 0):
                    raise ValueError("training batch contains a class outside the active rows")
                loss = F.cross_entropy(logits, local_targets)
            loss.backward()
            model.classifier.mask_inactive_gradients(active)
            optimizer.step()
            # Zero gradients alone are insufficient under SGD weight decay (and
            # momentum).  Restore old sequential-head rows bit-for-bit after
            # every step so only the current four affine rows can change.
            if frozen_inactive is not None:
                model.classifier.restore_rows(frozen_inactive)
            optimizer_steps += 1
            presentations += images.shape[0]
            final_loss = float(loss.detach().item())
            progress.update(1)
            next_position = batch_index + 1
            if optimizer_steps % checkpoint_steps == 0:
                atomic_torch_save(
                    checkpoint,
                    _checkpoint_record(
                        model,
                        optimizer,
                        current_epoch,
                        next_position,
                        optimizer_steps,
                        presentations,
                        final_loss,
                    ),
                )
        next_batch = 0
        atomic_torch_save(
            checkpoint,
            _checkpoint_record(
                model,
                optimizer,
                current_epoch + 1,
                0,
                optimizer_steps,
                presentations,
                final_loss,
            ),
        )
    progress.close()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    final_adapter = {
        module: LoRAFactors(factors.a.cpu(), factors.b.cpu(), factors.scale)
        for module, factors in adapter_factors(model).items()
    }
    rows = model.classifier.rows()
    final_classifier = ClassifierRows(
        rows.class_ids, rows.weight.cpu(), rows.bias.cpu()
    )
    return TrainingResult(
        adapter=final_adapter,
        classifier=final_classifier,
        optimizer_steps=optimizer_steps,
        image_presentations=presentations,
        wall_seconds=time.monotonic() - started,
        final_loss=final_loss,
        peak_vram_bytes=int(peak),
    )


def os_cpu_workers() -> int:
    """Bound data loading to at most 75% of visible CPU cores."""
    import os

    return max(1, int((os.cpu_count() or 1) * 0.75))


__all__ = [
    "TrainingResult",
    "deterministic_epoch_order",
    "os_cpu_workers",
    "train_adapter_model",
]
