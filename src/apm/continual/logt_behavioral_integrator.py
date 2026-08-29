"""Direct label integration over fixed LogT level-slot behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.logt_evidence_bank import TemporalNode
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoBaseState,
    top_two_base_state,
    top_two_hidden_logits,
    zero_top_two_adapter,
)
from apm.experiments.vamp_logt_integrator_rotated_config import IntegratorConfig
from apm.experiments.vamp_logt_router_data import named_seed


CLASS_COUNT = 10


@dataclass(frozen=True, slots=True)
class IntegratorObservations:
    """Detached fixed-slot features and their parameter-free ensemble."""

    features: Tensor
    node_log_probabilities: Tensor
    active_mask: Tensor
    baseline_log_probabilities: Tensor

    def __post_init__(self) -> None:
        if self.node_log_probabilities.ndim != 3:
            raise ValueError("integrator node probabilities must have three axes")
        rows, levels, classes = self.node_log_probabilities.shape
        if (
            self.features.ndim != 2
            or levels < 1
            or classes != CLASS_COUNT
            or self.features.shape[0] != rows
            or self.features.shape[1] % levels != 0
            or self.active_mask.shape != (levels,)
            or self.active_mask.dtype != torch.bool
            or not bool(self.active_mask.any())
            or self.baseline_log_probabilities.shape != (rows, CLASS_COUNT)
            or any(tensor.requires_grad for tensor in self.tensors)
            or not all(torch.isfinite(tensor).all() for tensor in self.tensors)
        ):
            raise ValueError("integrator observations are malformed or attached")

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        """Return every tensor covered by the detached/finite invariant."""
        return (
            self.features,
            self.node_log_probabilities,
            self.active_mask,
            self.baseline_log_probabilities,
        )


@dataclass(frozen=True, slots=True)
class IntegratorSupervision:
    """One immutable feature/label batch for direct prediction."""

    observations: IntegratorObservations
    labels: Tensor

    def __post_init__(self) -> None:
        if (
            self.labels.shape != (len(self.observations.features),)
            or self.labels.dtype != torch.int64
            or self.labels.requires_grad
            or bool(torch.any(self.labels < 0))
            or bool(torch.any(self.labels >= CLASS_COUNT))
        ):
            raise ValueError("integrator labels are invalid or misaligned")


@dataclass(slots=True)
class IntegratorConditionState:
    """One independent residual integrator and AdamW state."""

    name: str
    integrator: "LevelSlotIntegrator"
    optimizer: torch.optim.AdamW
    optimizer_steps: int = 0


@dataclass(frozen=True, slots=True)
class IntegratorTrainingResult:
    """Pre/post source losses and accuracy for one bounded update."""

    objective_before: float
    objective_after: float
    current_loss_before: float
    current_loss_after: float
    historical_loss_before: float | None
    historical_loss_after: float | None
    current_accuracy_before: float
    current_accuracy_after: float
    baseline_current_accuracy: float
    optimizer_steps: int

    def __post_init__(self) -> None:
        values = (
            self.objective_before,
            self.objective_after,
            self.current_loss_before,
            self.current_loss_after,
            self.current_accuracy_before,
            self.current_accuracy_after,
            self.baseline_current_accuracy,
        )
        optional = (self.historical_loss_before, self.historical_loss_after)
        if (
            not all(math.isfinite(value) for value in values)
            or any(value is not None and not math.isfinite(value) for value in optional)
            or (self.historical_loss_before is None) != (self.historical_loss_after is None)
            or self.optimizer_steps < 1
        ):
            raise ValueError("integrator training did not produce finite bounded evidence")


class LevelSlotIntegrator(nn.Module):
    """Residual ten-class predictor over persistent LogT level positions."""

    def __init__(
        self,
        input_dim: int,
        maximum_levels: int,
        slot_dim: int,
        hidden_widths: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            input_dim != maximum_levels * slot_dim
            or maximum_levels < 1
            or slot_dim < 1
            or len(hidden_widths) != 3
        ):
            raise ValueError("integrator dimensions require three fixed hidden widths")
        self.input_dim = input_dim
        self.maximum_levels = maximum_levels
        self.slot_dim = slot_dim
        first, second, third = hidden_widths
        self.input_layer = nn.Linear(input_dim, first)
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

    def residual_logits(self, features: Tensor) -> Tensor:
        """Return learned class-logit corrections for one fixed-slot matrix."""
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("integrator received an incompatible feature matrix")
        return self.output_layer(self.middle(self.input_layer(features)))

    def forward(self, features: Tensor, baseline_log_probabilities: Tensor) -> Tensor:
        """Return final logits as mean-ensemble log probabilities plus a residual."""
        if baseline_log_probabilities.shape != (len(features), CLASS_COUNT):
            raise ValueError("integrator baseline is misaligned")
        return baseline_log_probabilities + self.residual_logits(features)


def create_condition_state(
    name: str,
    input_dim: int,
    slot_dim: int,
    config: IntegratorConfig,
    seed: int,
    device: torch.device,
) -> IntegratorConditionState:
    """Create one independently initialized integrator and optimizer."""
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, "integrator-init", name))
        integrator = LevelSlotIntegrator(
            input_dim,
            config.maximum_levels,
            slot_dim,
            config.hidden_widths,
            config.dropout,
        ).to(device)
    return IntegratorConditionState(
        name,
        integrator,
        torch.optim.AdamW(
            integrator.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )


def build_node_observations(
    nodes: tuple[TemporalNode, ...],
    adapters: Mapping[str, TopTwoAdapterState],
    trunk_features: Tensor,
    base: TopTwoBaseState,
    maximum_levels: int,
    device: torch.device,
    batch_size: int,
) -> IntegratorObservations:
    """Build label-free behavior slots for every active frozen node."""
    if (
        not nodes
        or len(trunk_features) < 1
        or set(adapters) != {node.node_id for node in nodes}
        or len({node.level for node in nodes}) != len(nodes)
        or any(not 0 <= node.level < maximum_levels for node in nodes)
        or batch_size < 1
    ):
        raise ValueError("active bank cannot generate integrator observations")
    target_base = top_two_base_state(*base.tensors, device=device)
    slot_dim = target_base.hidden_dim + CLASS_COUNT + 1
    features = torch.zeros(
        (len(trunk_features), maximum_levels, slot_dim), dtype=torch.float32
    )
    node_log_probabilities = torch.zeros(
        (len(trunk_features), maximum_levels, CLASS_COUNT), dtype=torch.float32
    )
    active_mask = torch.zeros(maximum_levels, dtype=torch.bool)
    for node in sorted(nodes, key=lambda value: value.level):
        adapter = TopTwoAdapterState(
            *(tensor.detach().to(device).clone() for tensor in adapters[node.node_id].tensors)
        )
        hidden_rows, logit_rows = [], []
        with torch.inference_mode():
            for offset in range(0, len(trunk_features), batch_size):
                hidden, logits = top_two_hidden_logits(
                    trunk_features[offset : offset + batch_size].to(device),
                    target_base,
                    adapter,
                )
                hidden_rows.append(hidden.cpu())
                logit_rows.append(logits.cpu())
        hidden = torch.cat(hidden_rows)
        log_probabilities = F.log_softmax(torch.cat(logit_rows), dim=1)
        features[:, node.level, :-1] = torch.cat(
            (F.layer_norm(hidden, (target_base.hidden_dim,)), log_probabilities),
            dim=1,
        )
        features[:, node.level, -1] = 1.0
        node_log_probabilities[:, node.level] = log_probabilities
        active_mask[node.level] = True
    active = node_log_probabilities[:, active_mask]
    baseline = torch.logsumexp(active, dim=1) - math.log(active.shape[1])
    return IntegratorObservations(
        features.flatten(1).detach(),
        node_log_probabilities.detach(),
        active_mask.detach(),
        baseline.detach(),
    )


def build_base_observations(
    trunk_features: Tensor,
    base: TopTwoBaseState,
    maximum_levels: int,
    device: torch.device,
    batch_size: int,
) -> IntegratorObservations:
    """Build a matched-capacity control with only frozen-base behavior in slot zero."""
    if len(trunk_features) < 1 or maximum_levels < 1 or batch_size < 1:
        raise ValueError("base control requires positive levels and batch size")
    target_base = top_two_base_state(*base.tensors, device=device)
    target_adapter = zero_top_two_adapter(target_base)
    hidden_rows, logit_rows = [], []
    with torch.inference_mode():
        for offset in range(0, len(trunk_features), batch_size):
            hidden, logits = top_two_hidden_logits(
                trunk_features[offset : offset + batch_size].to(device),
                target_base,
                target_adapter,
            )
            hidden_rows.append(hidden.cpu())
            logit_rows.append(logits.cpu())
    hidden = torch.cat(hidden_rows)
    log_probabilities = F.log_softmax(torch.cat(logit_rows), dim=1)
    slot_dim = target_base.hidden_dim + CLASS_COUNT + 1
    features = torch.zeros(
        (len(trunk_features), maximum_levels, slot_dim), dtype=torch.float32
    )
    node_log_probabilities = torch.zeros(
        (len(trunk_features), maximum_levels, CLASS_COUNT), dtype=torch.float32
    )
    features[:, 0, :-1] = torch.cat(
        (F.layer_norm(hidden, (target_base.hidden_dim,)), log_probabilities), dim=1
    )
    features[:, 0, -1] = 1.0
    node_log_probabilities[:, 0] = log_probabilities
    active_mask = torch.zeros(maximum_levels, dtype=torch.bool)
    active_mask[0] = True
    return IntegratorObservations(
        features.flatten(1).detach(),
        node_log_probabilities.detach(),
        active_mask,
        log_probabilities.detach(),
    )


def train_condition(
    state: IntegratorConditionState,
    current: IntegratorSupervision,
    historical: IntegratorSupervision | None,
    epochs: int,
    config: IntegratorConfig,
    seed: int,
    macro_step: int,
    device: torch.device,
) -> IntegratorTrainingResult:
    """Train one condition with fixed current/historical source weighting."""
    if epochs < 1:
        raise ValueError("integrator training requires at least one epoch")
    before_current = _source_metrics(state.integrator, current, device, config.minibatch_size)
    before_historical = (
        None
        if historical is None
        else _source_metrics(state.integrator, historical, device, config.minibatch_size)
    )
    state.integrator.train()
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, state.name, macro_step, "dropout"))
        for epoch in range(epochs):
            chunks = _training_chunks(
                len(current.labels),
                0 if historical is None else len(historical.labels),
                config.minibatch_size,
                named_seed(seed, state.name, macro_step, "minibatches", epoch),
            )
            for current_ids, historical_ids in chunks:
                current_scale = len(current_ids) * len(chunks) / len(current.labels)
                current_loss = current_scale * _batch_loss(
                    state.integrator, current, current_ids, device
                )
                loss = current_loss
                if historical is not None:
                    historical_scale = (
                        len(historical_ids) * len(chunks) / len(historical.labels)
                    )
                    historical_loss = historical_scale * _batch_loss(
                        state.integrator, historical, historical_ids, device
                    )
                    loss = (
                        config.current_source_weight * current_loss
                        + (1.0 - config.current_source_weight) * historical_loss
                    )
                state.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    state.integrator.parameters(), config.gradient_clip_norm
                )
                state.optimizer.step()
                state.optimizer_steps += 1
    after_current = _source_metrics(state.integrator, current, device, config.minibatch_size)
    after_historical = (
        None
        if historical is None
        else _source_metrics(state.integrator, historical, device, config.minibatch_size)
    )
    objective_before = _combined_loss(before_current[0], before_historical, config)
    objective_after = _combined_loss(after_current[0], after_historical, config)
    baseline_accuracy = float(
        (
            current.observations.baseline_log_probabilities.argmax(dim=1)
            == current.labels
        )
        .float()
        .mean()
        .item()
    )
    return IntegratorTrainingResult(
        objective_before,
        objective_after,
        before_current[0],
        after_current[0],
        None if before_historical is None else before_historical[0],
        None if after_historical is None else after_historical[0],
        before_current[1],
        after_current[1],
        baseline_accuracy,
        state.optimizer_steps,
    )


def prediction_logits(
    integrator: LevelSlotIntegrator,
    observations: IntegratorObservations,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Return detached final class logits for one complete observation batch."""
    integrator.eval()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(observations.features), batch_size):
            rows.append(
                integrator(
                    observations.features[offset : offset + batch_size].to(device),
                    observations.baseline_log_probabilities[
                        offset : offset + batch_size
                    ].to(device),
                ).cpu()
            )
    return torch.cat(rows)


def inactive_slots_are_zero(observations: IntegratorObservations, slot_dim: int) -> bool:
    """Return whether every inactive fixed slot is exactly zero."""
    slots = observations.features.reshape(
        len(observations.features), len(observations.active_mask), slot_dim
    )
    inactive = slots[:, ~observations.active_mask]
    return bool(torch.equal(inactive, torch.zeros_like(inactive)))


def _batch_loss(
    integrator: LevelSlotIntegrator,
    supervision: IntegratorSupervision,
    indices: Tensor,
    device: torch.device,
) -> Tensor:
    observations = supervision.observations
    return F.cross_entropy(
        integrator(
            observations.features[indices].to(device),
            observations.baseline_log_probabilities[indices].to(device),
        ),
        supervision.labels[indices].to(device),
    )


def _source_metrics(
    integrator: LevelSlotIntegrator,
    supervision: IntegratorSupervision,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    logits = prediction_logits(integrator, supervision.observations, device, batch_size)
    return (
        float(F.cross_entropy(logits, supervision.labels).item()),
        float((logits.argmax(dim=1) == supervision.labels).float().mean().item()),
    )


def _combined_loss(
    current_loss: float,
    historical_metrics: tuple[float, float] | None,
    config: IntegratorConfig,
) -> float:
    return (
        current_loss
        if historical_metrics is None
        else config.current_source_weight * current_loss
        + (1.0 - config.current_source_weight) * historical_metrics[0]
    )


def _training_chunks(
    current_count: int,
    historical_count: int,
    maximum_batch_size: int,
    seed: int,
) -> tuple[tuple[Tensor, Tensor], ...]:
    total = current_count + historical_count
    chunk_count = math.ceil(total / maximum_batch_size)
    generator = torch.Generator().manual_seed(seed)
    current = torch.tensor_split(
        torch.randperm(current_count, generator=generator), chunk_count
    )
    historical = (
        torch.tensor_split(
            torch.randperm(historical_count, generator=generator), chunk_count
        )
        if historical_count
        else tuple(torch.empty(0, dtype=torch.int64) for _ in range(chunk_count))
    )
    if any(
        len(left) + len(right) > maximum_batch_size
        for left, right in zip(current, historical, strict=True)
    ):
        raise RuntimeError("source-balanced integrator minibatch exceeded its bound")
    return tuple(zip(current, historical, strict=True))


__all__ = [
    "CLASS_COUNT",
    "IntegratorConditionState",
    "IntegratorObservations",
    "IntegratorSupervision",
    "IntegratorTrainingResult",
    "LevelSlotIntegrator",
    "build_base_observations",
    "build_node_observations",
    "create_condition_state",
    "inactive_slots_are_zero",
    "prediction_logits",
    "train_condition",
]
