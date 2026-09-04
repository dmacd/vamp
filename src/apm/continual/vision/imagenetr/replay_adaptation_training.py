"""Paired optimization rules for the ImageNet-R replay diagnosis."""

from __future__ import annotations

import math
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.vision.imagenetr.integrator_config import (
    IntegratorOptimizationConfig,
)
from apm.continual.vision.imagenetr.integrator_model import (
    IntegratorFitResult,
    IntegratorState,
    IntegratorSupervision,
    _derived_seed,
    _metrics,
    fit_integrator_epochs,
)


def task_uniform_weights(labels: Tensor) -> Tensor:
    """Assign equal total cross-entropy weight to every represented four-class task."""
    if labels.ndim != 1 or labels.dtype != torch.int64 or not len(labels):
        raise ValueError("task-uniform weighting requires nonempty int64 labels")
    task_ids = labels // 4
    represented = torch.unique(task_ids, sorted=True)
    counts = torch.bincount(task_ids, minlength=int(task_ids.max()) + 1).float()
    weights = len(labels) / (len(represented) * counts[task_ids])
    if (
        not bool(torch.isfinite(weights).all())
        or not bool(torch.all(weights > 0))
        or not math.isclose(float(weights.mean()), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("task-uniform weights are not finite and normalized")
    return weights


def reset_adamw(
    state: IntegratorState, config: IntegratorOptimizationConfig
) -> None:
    """Reset AdamW moments while preserving model parameters and cumulative step count."""
    state.optimizer = torch.optim.AdamW(
        state.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _train_weighted_epoch(
    state: IntegratorState,
    data: IntegratorSupervision,
    weights: Tensor,
    config: IntegratorOptimizationConfig,
    seed: int,
    epoch: int,
    device: torch.device,
) -> None:
    state.model.train()
    generator = torch.Generator().manual_seed(
        _derived_seed(seed, state.name, "epoch", epoch)
    )
    order = torch.randperm(len(data.labels), generator=generator)
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
            batch_weights = weights[indices].to(device, non_blocking=True)
            state.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = state.model(
                    features,
                    baseline,
                    data.observations.seen_class_mask.to(device),
                )
                loss = (
                    F.cross_entropy(logits, labels, reduction="none") * batch_weights
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), config.gradient_clip_norm
            )
            state.optimizer.step()
            state.optimizer_steps += 1


def fit_replay_epochs(
    state: IntegratorState,
    training: IntegratorSupervision,
    epochs: int,
    config: IntegratorOptimizationConfig,
    seed: int,
    stage: int,
    device: torch.device,
    weighting: str,
) -> IntegratorFitResult:
    """Fit one paired replay condition with ordinary or task-uniform loss."""
    if weighting == "example_uniform":
        return fit_integrator_epochs(
            state, training, epochs, config, seed, stage, device
        )
    if weighting != "task_uniform" or epochs < 1 or stage < 1:
        raise ValueError("unknown replay weighting or invalid training boundary")
    weights = task_uniform_weights(training.labels)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_steps = state.optimizer_steps
    for epoch in range(epochs):
        _train_weighted_epoch(
            state,
            training,
            weights,
            config,
            _derived_seed(seed, "stage", stage),
            epoch,
            device,
        )
    train_loss, train_accuracy = _metrics(
        state.model, training, config.batch_size, device
    )
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


__all__ = [
    "fit_replay_epochs",
    "reset_adamw",
    "task_uniform_weights",
]
