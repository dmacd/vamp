"""Residual 200-way prediction integration over fixed LogT level slots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.integrator_config import (
    IntegratorOptimizationConfig,
)


CLASS_COUNT = 200
PRELOGIT_DIM = 768
CORE_VARIANTS = ("scores", "behavior", "behavior_base")
VARIANT_SLOT_DIMS: Mapping[str, int] = {
    "scores": 2 * CLASS_COUNT + 1,
    "behavior": PRELOGIT_DIM + 3 * CLASS_COUNT + 1,
    "behavior_base": PRELOGIT_DIM + 4 * CLASS_COUNT + 1,
    "behavior_two_layer": 2 * PRELOGIT_DIM + 3 * CLASS_COUNT + 1,
}


@dataclass(frozen=True, slots=True)
class IntegratorObservations:
    """Detached label-free inputs and their parameter-free raw-union baseline."""

    features: Tensor
    baseline_logits: Tensor
    seen_class_mask: Tensor
    active_slot_mask: Tensor
    variant: str
    slot_dim: int

    def __post_init__(self) -> None:
        rows = len(self.features)
        if (
            self.variant not in VARIANT_SLOT_DIMS
            or self.slot_dim != VARIANT_SLOT_DIMS[self.variant]
            or self.features.ndim != 2
            or self.features.shape[1] % self.slot_dim
            or self.baseline_logits.shape != (rows, CLASS_COUNT)
            or self.seen_class_mask.shape != (CLASS_COUNT,)
            or self.seen_class_mask.dtype != torch.bool
            or self.active_slot_mask.shape != (self.features.shape[1] // self.slot_dim,)
            or self.active_slot_mask.dtype != torch.bool
            or not bool(self.active_slot_mask.any())
            or not bool(self.seen_class_mask.any())
            or any(tensor.requires_grad for tensor in (self.features, self.baseline_logits))
            or not bool(torch.isfinite(self.features).all())
            or not bool(torch.isfinite(self.baseline_logits[:, self.seen_class_mask]).all())
            or not bool(torch.isneginf(self.baseline_logits[:, ~self.seen_class_mask]).all())
        ):
            raise ValueError("integrator observations are malformed or attached")

    @property
    def maximum_slots(self) -> int:
        """Return the fixed number of semantic input slots."""
        return self.features.shape[1] // self.slot_dim

    def select(self, indices: Tensor) -> "IntegratorObservations":
        """Project cached observations to a deterministic image subset."""
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError("observation row indices must be a one-dimensional int64 tensor")
        return IntegratorObservations(
            self.features[indices],
            self.baseline_logits[indices],
            self.seen_class_mask,
            self.active_slot_mask,
            self.variant,
            self.slot_dim,
        )


@dataclass(frozen=True, slots=True)
class IntegratorSupervision:
    """Direct class labels paired with a separately constructed observation batch."""

    observations: IntegratorObservations
    labels: Tensor

    def __post_init__(self) -> None:
        if (
            self.labels.shape != (len(self.observations.features),)
            or self.labels.dtype != torch.int64
            or self.labels.requires_grad
            or bool(torch.any(self.labels < 0))
            or bool(torch.any(self.labels >= CLASS_COUNT))
            or not bool(self.observations.seen_class_mask[self.labels].all())
        ):
            raise ValueError("integrator labels are invalid or outside the seen classes")


class ImageNetResidualIntegrator(nn.Module):
    """Fixed-width residual classifier over stable ImageNet-R LogT slots."""

    def __init__(
        self,
        maximum_slots: int,
        slot_dim: int,
        hidden_widths: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if maximum_slots < 1 or slot_dim < 1 or len(hidden_widths) != 3:
            raise ValueError("integrator dimensions must be positive with three hidden widths")
        self.maximum_slots = maximum_slots
        self.slot_dim = slot_dim
        self.input_dim = maximum_slots * slot_dim
        first, second, third = hidden_widths
        self.input_layer = nn.Linear(self.input_dim, first)
        self.middle = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(first),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.GELU(),
            nn.LayerNorm(second),
            nn.Dropout(dropout),
            nn.Linear(second, third),
            nn.GELU(),
        )
        self.output_layer = nn.Linear(third, CLASS_COUNT)
        with torch.no_grad():
            self.input_layer.weight[:, slot_dim:].zero_()
            self.output_layer.weight.zero_()
            self.output_layer.bias.zero_()

    def forward(
        self, features: Tensor, baseline_logits: Tensor, seen_class_mask: Tensor
    ) -> Tensor:
        """Add a learned correction and mask classes absent from the frontier."""
        if (
            features.ndim != 2
            or features.shape[1] != self.input_dim
            or baseline_logits.shape != (len(features), CLASS_COUNT)
            or seen_class_mask.shape != (CLASS_COUNT,)
        ):
            raise ValueError("integrator inputs do not match its fixed architecture")
        residual = self.output_layer(self.middle(self.input_layer(features)))
        logits = baseline_logits.masked_fill(~seen_class_mask, 0.0) + residual
        return logits.masked_fill(~seen_class_mask, -torch.inf)


@dataclass(slots=True)
class IntegratorState:
    """One trainable integrator and its independently owned AdamW state."""

    name: str
    model: ImageNetResidualIntegrator
    optimizer: torch.optim.AdamW
    optimizer_steps: int = 0


@dataclass(frozen=True, slots=True)
class IntegratorFitResult:
    """Best restored metrics and exact optimizer-work accounting."""

    epochs: int
    best_epoch: int
    optimizer_steps: int
    train_loss: float
    train_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None
    image_presentations: int
    training_forward_example_passes: int
    training_backward_example_passes: int
    validation_forward_example_passes: int
    peak_vram_bytes: int
    wall_seconds: float
    converged: bool

    def __post_init__(self) -> None:
        finite = (self.train_loss, self.train_accuracy, self.wall_seconds)
        optional = (self.validation_loss, self.validation_accuracy)
        if (
            self.epochs < 1
            or not 1 <= self.best_epoch <= self.epochs
            or self.optimizer_steps < 1
            or self.image_presentations < 1
            or self.training_forward_example_passes < self.image_presentations
            or self.training_backward_example_passes != self.image_presentations
            or self.validation_forward_example_passes < 0
            or self.peak_vram_bytes < 0
            or not all(math.isfinite(value) for value in finite)
            or any(value is not None and not math.isfinite(value) for value in optional)
            or (self.validation_loss is None) != (self.validation_accuracy is None)
        ):
            raise ValueError("integrator fit result is incomplete")


def _derived_seed(seed: int, *parts: object) -> int:
    return int(
        sha256("\0".join(("imagenetr50-integrator-v1", str(seed), *(str(part) for part in parts))).encode()).hexdigest()[:15],
        16,
    )


def create_integrator_state(
    name: str,
    maximum_slots: int,
    variant: str,
    config: IntegratorOptimizationConfig,
    seed: int,
    device: torch.device,
) -> IntegratorState:
    """Create one reproducibly initialized residual predictor and optimizer."""
    if variant not in VARIANT_SLOT_DIMS or not name or seed < 0:
        raise ValueError("unknown feature variant or invalid integrator identity")
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(_derived_seed(seed, name, variant, "initialization"))
        model = ImageNetResidualIntegrator(
            maximum_slots,
            VARIANT_SLOT_DIMS[variant],
            config.hidden_widths,
            config.dropout,
        ).to(device)
    return IntegratorState(
        name,
        model,
        torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        ),
    )


def _metrics(
    model: ImageNetResidualIntegrator,
    data: IntegratorSupervision,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.inference_mode():
        for offset in range(0, len(data.labels), batch_size):
            features = data.observations.features[offset : offset + batch_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            baseline = data.observations.baseline_logits[offset : offset + batch_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels = data.labels[offset : offset + batch_size].to(device, non_blocking=True)
            logits = model(features, baseline, data.observations.seen_class_mask.to(device))
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
    return loss_sum / len(data.labels), 100.0 * correct / len(data.labels)


def _train_epoch(
    state: IntegratorState,
    data: IntegratorSupervision,
    config: IntegratorOptimizationConfig,
    seed: int,
    epoch: int,
    device: torch.device,
) -> int:
    state.model.train()
    generator = torch.Generator().manual_seed(_derived_seed(seed, state.name, "epoch", epoch))
    order = torch.randperm(len(data.labels), generator=generator)
    batches = 0
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(_derived_seed(seed, state.name, "dropout", epoch))
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset : offset + config.batch_size]
            features = data.observations.features[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            baseline = data.observations.baseline_logits[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels = data.labels[indices].to(device, non_blocking=True)
            state.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = state.model(
                    features, baseline, data.observations.seen_class_mask.to(device)
                )
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), config.gradient_clip_norm)
            state.optimizer.step()
            state.optimizer_steps += 1
            batches += 1
    return batches


def fit_integrator_epochs(
    state: IntegratorState,
    training: IntegratorSupervision,
    epochs: int,
    config: IntegratorOptimizationConfig,
    seed: int,
    stage: int,
    device: torch.device,
) -> IntegratorFitResult:
    """Advance a persistent integrator for a fixed number of complete epochs."""
    if epochs < 1 or stage < 1:
        raise ValueError("persistent integration requires positive epochs and stage")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_steps = state.optimizer_steps
    for epoch in range(epochs):
        _train_epoch(
            state,
            training,
            config,
            _derived_seed(seed, "stage", stage),
            epoch,
            device,
        )
    train_loss, train_accuracy = _metrics(state.model, training, config.batch_size, device)
    return IntegratorFitResult(
        epochs,
        epochs,
        state.optimizer_steps - initial_steps,
        train_loss,
        train_accuracy,
        None,
        None,
        epochs * len(training.labels),
        (epochs + 1) * len(training.labels),
        epochs * len(training.labels),
        0,
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        time.monotonic() - started,
        False,
    )


def fit_fresh_integrator(
    state: IntegratorState,
    training: IntegratorSupervision,
    validation: IntegratorSupervision,
    config: IntegratorOptimizationConfig,
    seed: int,
    stage: int,
    device: torch.device,
    progress: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_key: str | None = None,
) -> IntegratorFitResult:
    """Fit a fresh restart to deterministic held-out convergence and restore its best epoch."""
    if stage < 1 or not len(training.labels) or not len(validation.labels):
        raise ValueError("fresh integration requires nonempty train and validation data")
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    if (checkpoint_path is None) != (checkpoint_key is None):
        raise ValueError("fresh checkpoint path and identity must be supplied together")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_loss = math.inf
    best_accuracy = 0.0
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    stale = 0
    start_epoch = 0
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        if (
            type(checkpoint) is not dict
            or checkpoint.get("schema_version") != "imagenetr50-integrator-fresh-checkpoint-v1"
            or checkpoint.get("checkpoint_key") != checkpoint_key
        ):
            raise ValueError("fresh integrator checkpoint identity changed")
        state.model.load_state_dict(checkpoint["model"], strict=True)
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.optimizer_steps = int(checkpoint["optimizer_steps"])
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint["best_loss"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = dict(checkpoint["best_state"])
        stale = int(checkpoint["stale"])
    already_converged = (
        start_epoch >= config.fresh_minimum_epochs and stale >= config.fresh_patience
    )
    history = tqdm(
        range(
            config.fresh_maximum_epochs + 1 if already_converged else start_epoch + 1,
            config.fresh_maximum_epochs + 1,
        ),
        desc=f"fresh integrator stage {stage}",
        disable=not progress,
        unit="epoch",
        leave=False,
    )
    epochs_ran = start_epoch
    for epoch in history:
        _train_epoch(state, training, config, _derived_seed(seed, "fresh", stage), epoch, device)
        validation_loss, validation_accuracy = _metrics(
            state.model, validation, config.batch_size, device
        )
        improved = validation_loss <= best_loss - config.improvement_delta
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in state.model.state_dict().items()
            }
        stale = 0 if improved else stale + 1
        epochs_ran = epoch
        if checkpoint_path is not None:
            atomic_torch_save(
                checkpoint_path,
                {
                    "best_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "checkpoint_key": checkpoint_key,
                    "epoch": epoch,
                    "model": state.model.state_dict(),
                    "optimizer": state.optimizer.state_dict(),
                    "optimizer_steps": state.optimizer_steps,
                    "schema_version": "imagenetr50-integrator-fresh-checkpoint-v1",
                    "stale": stale,
                },
            )
        history.set_postfix(best=f"{best_accuracy:.2f}%", loss=f"{best_loss:.4f}")
        if epoch >= config.fresh_minimum_epochs and stale >= config.fresh_patience:
            break
    if not best_state:
        raise RuntimeError("fresh integrator produced no finite validation checkpoint")
    state.model.load_state_dict(best_state, strict=True)
    train_loss, train_accuracy = _metrics(state.model, training, config.batch_size, device)
    validation_loss, validation_accuracy = _metrics(
        state.model, validation, config.batch_size, device
    )
    return IntegratorFitResult(
        epochs_ran,
        best_epoch,
        state.optimizer_steps,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy,
        epochs_ran * len(training.labels),
        (epochs_ran + 1) * len(training.labels),
        epochs_ran * len(training.labels),
        (epochs_ran + 1) * len(validation.labels),
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        time.monotonic() - started,
        epochs_ran < config.fresh_maximum_epochs,
    )


def predict_integrator(
    model: ImageNetResidualIntegrator,
    observations: IntegratorObservations,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Return task-free global predictions for an immutable observation batch."""
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(observations.features), batch_size):
            predictions.append(
                model(
                    observations.features[offset : offset + batch_size].to(
                        device=device, dtype=torch.float32, non_blocking=True
                    ),
                    observations.baseline_logits[offset : offset + batch_size].to(
                        device=device, dtype=torch.float32, non_blocking=True
                    ),
                    observations.seen_class_mask.to(device),
                ).argmax(dim=1).cpu()
            )
    return torch.cat(tuple(predictions))


def parameter_count(model: ImageNetResidualIntegrator) -> int:
    """Return exact trainable integrator parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_smallest_near_best(
    accuracies: Mapping[str, float], tolerance_points: float
) -> str:
    """Choose the lowest-dimensional feature family within tolerance of the best."""
    if set(accuracies) != set(CORE_VARIANTS) or tolerance_points < 0.0:
        raise ValueError("feature selection requires every predeclared variant")
    best = max(accuracies.values())
    eligible = tuple(
        variant
        for variant in CORE_VARIANTS
        if accuracies[variant] >= best - tolerance_points
    )
    return min(eligible, key=lambda variant: VARIANT_SLOT_DIMS[variant])


__all__ = [
    "CLASS_COUNT",
    "CORE_VARIANTS",
    "PRELOGIT_DIM",
    "VARIANT_SLOT_DIMS",
    "ImageNetResidualIntegrator",
    "IntegratorFitResult",
    "IntegratorObservations",
    "IntegratorState",
    "IntegratorSupervision",
    "create_integrator_state",
    "fit_fresh_integrator",
    "fit_integrator_epochs",
    "parameter_count",
    "predict_integrator",
    "select_smallest_near_best",
]
