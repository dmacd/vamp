"""Frozen-base, full-model deltas for the Permuted-MNIST dense hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


AFFINE_LAYER_COUNT = 4
CLASS_COUNT = 10
INPUT_DIMENSION = 28 * 28


@dataclass(frozen=True, slots=True)
class DenseMlpState:
    """Detached tensors for three hidden affine layers and one digit head."""

    tensors: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.tensors) != 2 * AFFINE_LAYER_COUNT:
            raise ValueError("dense MLP state must contain four weight/bias pairs")
        weights = self.tensors[0::2]
        biases = self.tensors[1::2]
        expected_inputs = (INPUT_DIMENSION, *(weight.shape[0] for weight in weights[:-1]))
        if (
            any(weight.ndim != 2 for weight in weights)
            or any(bias.shape != (weight.shape[0],) for weight, bias in zip(weights, biases, strict=True))
            or any(weight.shape[1] != expected for weight, expected in zip(weights, expected_inputs, strict=True))
            or weights[-1].shape[0] != CLASS_COUNT
            or any(tensor.requires_grad for tensor in self.tensors)
            or not all(torch.isfinite(tensor).all() for tensor in self.tensors)
        ):
            raise ValueError("dense MLP state shapes or values are incompatible")

    @property
    def hidden_widths(self) -> tuple[int, int, int]:
        """Return the three hidden-layer widths."""
        return tuple(int(weight.shape[0]) for weight in self.tensors[0:6:2])  # type: ignore[return-value]

    @property
    def embedding_dim(self) -> int:
        """Return the final hidden activation width exposed to observers."""
        return self.hidden_widths[-1]

    @property
    def parameter_count(self) -> int:
        """Return the exact number of scalar affine parameters."""
        return sum(tensor.numel() for tensor in self.tensors)


@dataclass(frozen=True, slots=True)
class DenseExamples:
    """A lazy Cartesian product of source images and fixed pixel transforms."""

    images: Tensor
    labels: Tensor
    permutations: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        rows = len(self.labels)
        if (
            self.images.shape != (rows, 1, 28, 28)
            or self.images.dtype != torch.float32
            or self.labels.shape != (rows,)
            or self.labels.dtype != torch.int64
            or not self.permutations
            or any(
                permutation.shape != (INPUT_DIMENSION,)
                or permutation.dtype != torch.int64
                or not torch.equal(
                    torch.sort(permutation).values,
                    torch.arange(INPUT_DIMENSION, dtype=torch.int64),
                )
                for permutation in self.permutations
            )
        ):
            raise ValueError("dense examples are malformed")

    def __len__(self) -> int:
        return len(self.labels) * len(self.permutations)

    def batch(self, row_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Materialize arbitrary Cartesian-product rows in requested order."""
        if row_ids.ndim != 1 or row_ids.dtype != torch.int64:
            raise ValueError("dense example row IDs must be an int64 vector")
        source_count = len(self.labels)
        transform_ids = torch.div(row_ids, source_count, rounding_mode="floor")
        source_ids = row_ids.remainder(source_count)
        flattened = self.images[source_ids].flatten(1)
        transformed = torch.empty_like(flattened)
        for transform_id in torch.unique(transform_ids, sorted=True).tolist():
            selected = transform_ids == int(transform_id)
            transformed[selected] = flattened[selected][:, self.permutations[int(transform_id)]]
        return transformed, self.labels[source_ids]


@dataclass(frozen=True, slots=True)
class DenseOptimizerConfig:
    """AdamW and batching constants for one dense-model fit."""

    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.batch_size < 1
            or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("invalid dense optimizer configuration")


@dataclass(frozen=True, slots=True)
class DenseConvergenceConfig:
    """Validation-driven learning-rate and stopping rules."""

    minimum_epochs: int
    maximum_epochs: int
    improvement_delta: float
    learning_rate_patience: int
    learning_rate_factor: float
    minimum_learning_rate: float
    convergence_patience: int

    def __post_init__(self) -> None:
        if (
            self.minimum_epochs < 1
            or self.maximum_epochs < self.minimum_epochs
            or self.improvement_delta <= 0.0
            or self.learning_rate_patience < 1
            or not 0.0 < self.learning_rate_factor < 1.0
            or self.minimum_learning_rate <= 0.0
            or self.convergence_patience < 1
        ):
            raise ValueError("invalid dense convergence configuration")


@dataclass(frozen=True, slots=True)
class DenseEpochResult:
    """One epoch of train and optional validation evidence."""

    epoch: int
    learning_rate: float
    training_loss: float
    training_accuracy: float
    validation_loss: float | None
    validation_accuracy: float | None


@dataclass(frozen=True, slots=True)
class DenseFitResult:
    """Restored best state and complete deterministic training accounting."""

    state: DenseMlpState
    best_epoch: int
    epochs_ran: int
    optimizer_steps: int
    training_example_presentations: int
    validation_example_presentations: int
    stop_reason: str
    history: tuple[DenseEpochResult, ...]

    def __post_init__(self) -> None:
        if (
            self.best_epoch < 1
            or self.epochs_ran < self.best_epoch
            or self.optimizer_steps < 1
            or self.training_example_presentations < 1
            or self.validation_example_presentations < 0
            or len(self.history) != self.epochs_ran
            or self.stop_reason not in {"fixed_epochs", "minimum_learning_rate_plateau", "maximum_epochs"}
        ):
            raise ValueError("invalid dense fit result")


class DenseMnistMLP(nn.Module):
    """Three ReLU/dropout hidden layers plus a separate ten-class output."""

    def __init__(self, hidden_widths: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        if any(width < 1 for width in hidden_widths) or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid dense MLP architecture")
        first, second, third = hidden_widths
        self.hidden_widths = hidden_widths
        self.dropout_probability = dropout
        self.hidden_layers = nn.ModuleList(
            (
                nn.Linear(INPUT_DIMENSION, first),
                nn.Linear(first, second),
                nn.Linear(second, third),
            )
        )
        self.classifier = nn.Linear(third, CLASS_COUNT)

    def hidden_logits(self, flattened_images: Tensor) -> tuple[Tensor, Tensor]:
        """Return the last ReLU activation and logits, with dropout only in train mode."""
        if flattened_images.ndim != 2 or flattened_images.shape[1] != INPUT_DIMENSION:
            raise ValueError("dense MLP expects flattened 28x28 images")
        hidden = flattened_images
        for layer in self.hidden_layers:
            hidden = F.dropout(
                F.relu(layer(hidden)),
                p=self.dropout_probability,
                training=self.training,
            )
        return hidden, self.classifier(hidden)

    def forward(self, flattened_images: Tensor) -> Tensor:
        """Return ten-class logits."""
        return self.hidden_logits(flattened_images)[1]


def dense_state(model: DenseMnistMLP, device: torch.device | str = "cpu") -> DenseMlpState:
    """Copy all four affine layers into one detached state."""
    layers = (*model.hidden_layers, model.classifier)
    return DenseMlpState(
        tuple(
            tensor.detach().to(device).clone()
            for layer in layers
            for tensor in (layer.weight, layer.bias)
        )
    )


def load_dense_state(model: DenseMnistMLP, state: DenseMlpState) -> None:
    """Load a compatible detached state into an existing module."""
    if model.hidden_widths != state.hidden_widths:
        raise ValueError("dense state width differs from the target model")
    with torch.no_grad():
        for layer, weight, bias in zip(
            (*model.hidden_layers, model.classifier),
            state.tensors[0::2],
            state.tensors[1::2],
            strict=True,
        ):
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)


def zero_dense_delta(base: DenseMlpState) -> DenseMlpState:
    """Return exact zero deltas matching every frozen affine tensor."""
    return DenseMlpState(tuple(torch.zeros_like(tensor) for tensor in base.tensors))


def dense_delta(base: DenseMlpState, effective: DenseMlpState) -> DenseMlpState:
    """Return `effective - base` for every affine weight and bias."""
    if base.hidden_widths != effective.hidden_widths:
        raise ValueError("dense states have different architectures")
    return DenseMlpState(
        tuple(
            (trained - frozen.to(trained.device)).detach()
            for frozen, trained in zip(base.tensors, effective.tensors, strict=True)
        )
    )


def apply_dense_delta(base: DenseMlpState, delta: DenseMlpState) -> DenseMlpState:
    """Return effective affine parameters for one frozen-base delta."""
    if base.hidden_widths != delta.hidden_widths:
        raise ValueError("dense base and delta have different architectures")
    return DenseMlpState(
        tuple(
            (frozen.to(update.device) + update).detach()
            for frozen, update in zip(base.tensors, delta.tensors, strict=True)
        )
    )


def dense_hidden_logits(
    flattened_images: Tensor,
    base: DenseMlpState,
    delta: DenseMlpState,
) -> tuple[Tensor, Tensor]:
    """Evaluate a frozen base plus one node delta with dropout disabled."""
    if flattened_images.ndim != 2 or flattened_images.shape[1] != INPUT_DIMENSION:
        raise ValueError("dense inference expects flattened 28x28 images")
    parameters = apply_dense_delta(base, delta).tensors
    hidden = flattened_images
    for weight, bias in zip(parameters[0:6:2], parameters[1:6:2], strict=True):
        hidden = F.relu(F.linear(hidden, weight, bias))
    return hidden, F.linear(hidden, parameters[6], parameters[7])


def fit_dense_model(
    training: DenseExamples,
    initial_state: DenseMlpState,
    optimizer_config: DenseOptimizerConfig,
    seed: int,
    device: torch.device,
    *,
    fixed_epochs: int | None = None,
    validation: DenseExamples | None = None,
    convergence: DenseConvergenceConfig | None = None,
    dropout: float = 0.2,
    progress_label: str = "dense MLP",
    progress: bool = False,
) -> DenseFitResult:
    """Fit all affine layers with deterministic AdamW and restore the best epoch."""
    if (fixed_epochs is None) == (convergence is None):
        raise ValueError("choose exactly one fixed or converged dense schedule")
    if convergence is not None and validation is None:
        raise ValueError("converged dense training requires held-out validation")
    maximum_epochs = fixed_epochs if fixed_epochs is not None else convergence.maximum_epochs
    if maximum_epochs is None or maximum_epochs < 1:
        raise ValueError("dense training requires positive epochs")
    model = DenseMnistMLP(initial_state.hidden_widths, dropout).to(device)
    load_dense_state(model, initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    epochs = tqdm(
        range(1, maximum_epochs + 1),
        desc=progress_label,
        disable=not progress,
        leave=False,
        unit="epoch",
    )
    best_state = initial_state
    best_epoch = 0
    best_validation_loss = math.inf
    significant_reference = math.inf
    epochs_without_significant_improvement = 0
    minimum_rate_plateau_epochs = 0
    optimizer_steps = 0
    history: list[DenseEpochResult] = []
    stop_reason = "fixed_epochs" if fixed_epochs is not None else "maximum_epochs"
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(seed)
        for epoch in epochs:
            model.train()
            order = torch.randperm(
                len(training),
                generator=torch.Generator().manual_seed(seed + epoch),
            )
            total_loss = 0.0
            correct = 0
            for offset in range(0, len(order), optimizer_config.batch_size):
                images, labels = training.batch(order[offset : offset + optimizer_config.batch_size])
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optimizer_config.gradient_clip_norm
                )
                optimizer.step()
                optimizer_steps += 1
                total_loss += float(loss.detach().item()) * len(labels)
                correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            validation_metrics = (
                None
                if validation is None
                else dense_metrics(model, validation, device, optimizer_config.batch_size)
            )
            training_loss = total_loss / len(training)
            training_accuracy = correct / len(training)
            validation_loss = None if validation_metrics is None else validation_metrics[0]
            validation_accuracy = None if validation_metrics is None else validation_metrics[1]
            history.append(
                DenseEpochResult(
                    epoch,
                    float(optimizer.param_groups[0]["lr"]),
                    training_loss,
                    training_accuracy,
                    validation_loss,
                    validation_accuracy,
                )
            )
            selection_loss = training_loss if validation_loss is None else validation_loss
            if fixed_epochs is not None or selection_loss < best_validation_loss:
                best_validation_loss = selection_loss
                best_epoch = epoch
                best_state = dense_state(model)
            if convergence is not None:
                significant = selection_loss <= significant_reference - convergence.improvement_delta
                if significant:
                    significant_reference = selection_loss
                    epochs_without_significant_improvement = 0
                    minimum_rate_plateau_epochs = 0
                else:
                    epochs_without_significant_improvement += 1
                    if float(optimizer.param_groups[0]["lr"]) <= convergence.minimum_learning_rate + 1.0e-15:
                        minimum_rate_plateau_epochs += 1
                if (
                    float(optimizer.param_groups[0]["lr"]) > convergence.minimum_learning_rate
                    and epochs_without_significant_improvement >= convergence.learning_rate_patience
                ):
                    next_rate = max(
                        convergence.minimum_learning_rate,
                        float(optimizer.param_groups[0]["lr"]) * convergence.learning_rate_factor,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = next_rate
                    epochs_without_significant_improvement = 0
                    minimum_rate_plateau_epochs = 0
                if (
                    epoch >= convergence.minimum_epochs
                    and float(optimizer.param_groups[0]["lr"]) <= convergence.minimum_learning_rate + 1.0e-15
                    and minimum_rate_plateau_epochs >= convergence.convergence_patience
                ):
                    stop_reason = "minimum_learning_rate_plateau"
                    break
            epochs.set_postfix(loss=f"{selection_loss:.4f}")
    if best_epoch < 1:
        raise RuntimeError("dense fit never produced a finite checkpoint")
    return DenseFitResult(
        best_state,
        best_epoch,
        len(history),
        optimizer_steps,
        len(history) * len(training),
        0 if validation is None else len(history) * len(validation),
        stop_reason,
        tuple(history),
    )


def dense_metrics(
    model: DenseMnistMLP,
    examples: DenseExamples,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy with dropout disabled."""
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            ids = torch.arange(offset, min(offset + batch_size, len(examples)))
            images, labels = examples.batch(ids)
            logits = model(images.to(device)).cpu()
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
    return total_loss / len(examples), correct / len(examples)


__all__ = [
    "AFFINE_LAYER_COUNT",
    "CLASS_COUNT",
    "DenseConvergenceConfig",
    "DenseEpochResult",
    "DenseExamples",
    "DenseFitResult",
    "DenseMlpState",
    "DenseMnistMLP",
    "DenseOptimizerConfig",
    "apply_dense_delta",
    "dense_delta",
    "dense_hidden_logits",
    "dense_metrics",
    "dense_state",
    "fit_dense_model",
    "load_dense_state",
    "zero_dense_delta",
]
