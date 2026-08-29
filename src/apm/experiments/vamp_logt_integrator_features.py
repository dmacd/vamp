"""Shared detached feature construction for LogT prediction integrators."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from apm.continual.logt_behavioral_integrator import (
    IntegratorSupervision,
    build_base_observations,
    build_node_observations,
    inactive_slots_are_zero,
)
from apm.experiments.vamp_logt_integrator_rotated_config import (
    IntegratorConfig,
    IntegratorEvaluationConfig,
)
from apm.experiments.vamp_logt_router_data import ExampleBatch, FrozenClassifierDependency
from apm.experiments.vamp_logt_router_state import ActiveAdapterBank


class IntegratorFeatureConfig(Protocol):
    """Structural configuration needed to construct fixed integrator slots."""

    integrator: IntegratorConfig
    evaluation: IntegratorEvaluationConfig


def frozen_integrator_trunk_features(
    config: IntegratorFeatureConfig,
    dependency: FrozenClassifierDependency,
    images: Tensor,
    device: torch.device,
) -> Tensor:
    """Return detached frozen-CNN trunk features in bounded inference batches."""
    from apm.continual.logt_behavioral_router import frozen_trunk_features

    return frozen_trunk_features(
        dependency.model,
        images,
        device,
        config.evaluation.inference_batch_size,
    )


def integrator_supervision(
    config: IntegratorFeatureConfig,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    examples: ExampleBatch,
    device: torch.device,
    *,
    base_only: bool = False,
) -> IntegratorSupervision:
    """Build detached fixed-slot supervision from one immutable image batch."""
    return integrator_supervision_from_trunk(
        config,
        dependency,
        bank,
        examples,
        frozen_integrator_trunk_features(config, dependency, examples.images, device),
        device,
        base_only=base_only,
    )


def integrator_supervision_from_trunk(
    config: IntegratorFeatureConfig,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    examples: ExampleBatch,
    trunk: Tensor,
    device: torch.device,
    *,
    base_only: bool = False,
) -> IntegratorSupervision:
    """Build detached fixed slots while reusing one already computed trunk pass."""
    observations = (
        build_base_observations(
            trunk,
            dependency.base,
            config.integrator.maximum_levels,
            device,
            config.evaluation.inference_batch_size,
        )
        if base_only
        else build_node_observations(
            bank.topology.active_nodes,
            bank.adapters,
            trunk,
            dependency.base,
            config.integrator.maximum_levels,
            device,
            config.evaluation.inference_batch_size,
        )
    )
    slot_dim = dependency.base.hidden_dim + 10 + 1
    if not inactive_slots_are_zero(observations, slot_dim):
        raise RuntimeError("integrator observation contains a nonzero inactive slot")
    return IntegratorSupervision(observations, examples.labels.detach())


__all__ = [
    "IntegratorFeatureConfig",
    "frozen_integrator_trunk_features",
    "integrator_supervision",
    "integrator_supervision_from_trunk",
]
