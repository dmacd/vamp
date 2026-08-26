"""Immutable full-rank deltas for a frozen two-linear-layer classifier suffix."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TopTwoBaseState:
    """Frozen embedding and classifier tensors shared by every adapter path."""

    embedding_weight: Tensor
    embedding_bias: Tensor
    classifier_weight: Tensor
    classifier_bias: Tensor

    def __post_init__(self) -> None:
        hidden_dim, trunk_dim = self.embedding_weight.shape
        classes, classifier_hidden_dim = self.classifier_weight.shape
        if (
            self.embedding_weight.ndim != 2
            or self.embedding_bias.shape != (hidden_dim,)
            or self.classifier_weight.ndim != 2
            or classifier_hidden_dim != hidden_dim
            or self.classifier_bias.shape != (classes,)
            or any(tensor.requires_grad for tensor in self.tensors)
            or not all(torch.isfinite(tensor).all() for tensor in self.tensors)
            or trunk_dim < 1
            or hidden_dim < 1
            or classes < 1
        ):
            raise ValueError("top-two base tensor shapes or values are incompatible")

    @property
    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.embedding_weight,
            self.embedding_bias,
            self.classifier_weight,
            self.classifier_bias,
        )

    @property
    def trunk_dim(self) -> int:
        return int(self.embedding_weight.shape[1])

    @property
    def hidden_dim(self) -> int:
        return int(self.embedding_weight.shape[0])

    @property
    def classes(self) -> int:
        return int(self.classifier_weight.shape[0])


@dataclass(frozen=True, slots=True)
class TopTwoAdapterState:
    """One node's committed deltas for both frozen suffix layers."""

    embedding_weight: Tensor
    embedding_bias: Tensor
    classifier_weight: Tensor
    classifier_bias: Tensor

    def __post_init__(self) -> None:
        if (
            self.embedding_weight.ndim != 2
            or self.embedding_bias.shape != (self.embedding_weight.shape[0],)
            or self.classifier_weight.ndim != 2
            or self.classifier_weight.shape[1] != self.embedding_weight.shape[0]
            or self.classifier_bias.shape != (self.classifier_weight.shape[0],)
            or any(tensor.requires_grad for tensor in self.tensors)
            or not all(torch.isfinite(tensor).all() for tensor in self.tensors)
        ):
            raise ValueError("top-two adapter tensor shapes or values are incompatible")

    @property
    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.embedding_weight,
            self.embedding_bias,
            self.classifier_weight,
            self.classifier_bias,
        )


@dataclass(frozen=True, slots=True)
class TopTwoAdamWState:
    """Explicit AdamW moments for the four top-two-layer delta tensors."""

    step: int
    first_moments: tuple[Tensor, Tensor, Tensor, Tensor]
    second_moments: tuple[Tensor, Tensor, Tensor, Tensor]

    def __post_init__(self) -> None:
        if (
            self.step < 0
            or len(self.first_moments) != 4
            or len(self.second_moments) != 4
            or any(tensor.requires_grad for tensor in (*self.first_moments, *self.second_moments))
            or not all(torch.isfinite(tensor).all() for tensor in (*self.first_moments, *self.second_moments))
        ):
            raise ValueError("invalid top-two AdamW state")


@dataclass(frozen=True, slots=True)
class TopTwoOptimizerConfig:
    """Optimizer constants shared by oracle and AF top-two-layer updates."""

    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float

    def __post_init__(self) -> None:
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.beta1 < 1.0
            or not 0.0 <= self.beta2 < 1.0
            or self.epsilon <= 0.0
        ):
            raise ValueError("invalid top-two optimizer configuration")


def top_two_base_state(
    embedding_weight: Tensor,
    embedding_bias: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    device: torch.device | str = "cpu",
) -> TopTwoBaseState:
    """Copy model parameters into one detached immutable base state."""
    return TopTwoBaseState(
        embedding_weight.detach().to(device).clone(),
        embedding_bias.detach().to(device).clone(),
        classifier_weight.detach().to(device).clone(),
        classifier_bias.detach().to(device).clone(),
    )


def zero_top_two_adapter(
    base: TopTwoBaseState,
    device: torch.device | str | None = None,
) -> TopTwoAdapterState:
    """Create exactly-zero deltas with shapes matching the frozen base."""
    target = base.embedding_weight.device if device is None else torch.device(device)
    return TopTwoAdapterState(*(torch.zeros_like(tensor, device=target) for tensor in base.tensors))


def zero_top_two_adamw(adapter: TopTwoAdapterState) -> TopTwoAdamWState:
    """Create zero AdamW moments matching one top-two-layer adapter."""
    return TopTwoAdamWState(
        0,
        tuple(torch.zeros_like(tensor) for tensor in adapter.tensors),
        tuple(torch.zeros_like(tensor) for tensor in adapter.tensors),
    )


def sum_top_two_adapters(
    adapters: tuple[TopTwoAdapterState, ...],
    base: TopTwoBaseState,
) -> TopTwoAdapterState:
    """Return cumulative path deltas, or an exact zero adapter for an empty path."""
    totals = list(zero_top_two_adapter(base).tensors)
    for adapter in adapters:
        totals = [
            (total + update).detach()
            for total, update in zip(totals, adapter.tensors)
        ]
    return TopTwoAdapterState(*totals)


def top_two_logits(
    trunk_features: Tensor,
    base: TopTwoBaseState,
    adapter: TopTwoAdapterState,
) -> Tensor:
    """Apply frozen base parameters plus one cumulative adapter through the ReLU suffix."""
    if trunk_features.ndim != 2 or trunk_features.shape[1] != base.trunk_dim:
        raise ValueError("top-two forward received incompatible trunk features")
    return _top_two_logits_from_tensors(trunk_features, base, adapter.tensors)


def train_top_two_adapter_step(
    trunk_features: Tensor,
    labels: Tensor,
    base: TopTwoBaseState,
    fixed_adapter: TopTwoAdapterState,
    adapter: TopTwoAdapterState,
    optimizer: TopTwoAdamWState,
    config: TopTwoOptimizerConfig,
) -> tuple[TopTwoAdapterState, TopTwoAdamWState, float]:
    """Apply one pure AdamW step to the target deltas after fixed ancestor deltas."""
    if labels.shape != (trunk_features.shape[0],):
        raise ValueError("top-two training batch shapes are incompatible")
    parameters = tuple(tensor.detach().clone().requires_grad_(True) for tensor in adapter.tensors)
    effective_tensors = tuple(
        fixed + parameter for fixed, parameter in zip(fixed_adapter.tensors, parameters)
    )
    loss = F.cross_entropy(
        _top_two_logits_from_tensors(trunk_features, base, effective_tensors), labels
    )
    gradients = torch.autograd.grad(loss, parameters)
    step = optimizer.step + 1
    first_moments = tuple(
        config.beta1 * moment + (1.0 - config.beta1) * gradient
        for moment, gradient in zip(optimizer.first_moments, gradients)
    )
    second_moments = tuple(
        config.beta2 * moment + (1.0 - config.beta2) * gradient.square()
        for moment, gradient in zip(optimizer.second_moments, gradients)
    )
    committed = TopTwoAdapterState(
        *(
            (
                parameter.detach() * (1.0 - config.learning_rate * config.weight_decay)
                - config.learning_rate
                * (first / (1.0 - config.beta1**step))
                / (
                    torch.sqrt(second / (1.0 - config.beta2**step))
                    + config.epsilon
                )
            ).detach()
            for parameter, first, second in zip(parameters, first_moments, second_moments)
        )
    )
    committed_optimizer = TopTwoAdamWState(
        step,
        tuple(moment.detach() for moment in first_moments),
        tuple(moment.detach() for moment in second_moments),
    )
    return committed, committed_optimizer, float(loss.detach().item())


def _top_two_logits_from_tensors(
    trunk_features: Tensor,
    base: TopTwoBaseState,
    adapter_tensors: tuple[Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    embedding_weight, embedding_bias, classifier_weight, classifier_bias = adapter_tensors
    hidden = F.relu(
        F.linear(
            trunk_features,
            base.embedding_weight + embedding_weight,
            base.embedding_bias + embedding_bias,
        )
    )
    return F.linear(
        hidden,
        base.classifier_weight + classifier_weight,
        base.classifier_bias + classifier_bias,
    )


__all__ = [
    "TopTwoAdapterState",
    "TopTwoAdamWState",
    "TopTwoBaseState",
    "TopTwoOptimizerConfig",
    "sum_top_two_adapters",
    "top_two_base_state",
    "top_two_logits",
    "train_top_two_adapter_step",
    "zero_top_two_adapter",
    "zero_top_two_adamw",
]
