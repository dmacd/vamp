"""Direct label integration over fixed LogT level-slot behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from time import perf_counter

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
    """Pre/post metrics and training-only model work for one bounded update."""

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
    training_forward_example_passes: int
    training_backward_example_passes: int
    training_forward_calls: int
    training_backward_calls: int
    training_wall_seconds: float

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
            or min(
                self.training_forward_example_passes,
                self.training_backward_example_passes,
                self.training_forward_calls,
                self.training_backward_calls,
            )
            < 1
            or not math.isfinite(self.training_wall_seconds)
            or self.training_wall_seconds < 0.0
        ):
            raise ValueError("integrator training did not produce finite bounded evidence")


@dataclass(frozen=True, slots=True)
class FullReplayConvergenceConfig:
    """Frozen stopping and learning-rate rules for cumulative full replay."""

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
            raise ValueError("invalid full-replay convergence settings")


@dataclass(frozen=True, slots=True)
class ConvergenceEpochResult:
    """One complete all-example epoch and its held-out validation evidence."""

    epoch: int
    learning_rate: float
    next_learning_rate: float
    training_loss: float
    training_accuracy: float
    validation_loss: float
    validation_accuracy: float
    best_validation_loss: float
    improved_best: bool
    significant_improvement: bool
    optimizer_steps: int

    def __post_init__(self) -> None:
        numeric = (
            self.learning_rate,
            self.next_learning_rate,
            self.training_loss,
            self.training_accuracy,
            self.validation_loss,
            self.validation_accuracy,
            self.best_validation_loss,
        )
        if (
            self.epoch < 0
            or self.optimizer_steps < 0
            or not all(math.isfinite(value) for value in numeric)
            or min(self.learning_rate, self.next_learning_rate) <= 0.0
        ):
            raise ValueError("invalid full-replay epoch evidence")

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible epoch record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConvergedFullReplayResult:
    """Best restored checkpoint and complete convergence accounting."""

    converged: bool
    stop_reason: str
    best_epoch: int
    epochs_ran: int
    best_training_loss: float
    best_training_accuracy: float
    best_validation_loss: float
    best_validation_accuracy: float
    final_learning_rate: float
    optimizer_steps: int
    training_example_presentations: int
    validation_example_presentations: int
    history: tuple[ConvergenceEpochResult, ...]

    def __post_init__(self) -> None:
        numeric = (
            self.best_training_loss,
            self.best_training_accuracy,
            self.best_validation_loss,
            self.best_validation_accuracy,
            self.final_learning_rate,
        )
        if (
            self.stop_reason not in {"minimum_learning_rate_plateau", "maximum_epochs"}
            or self.converged != (self.stop_reason == "minimum_learning_rate_plateau")
            or self.epochs_ran < 1
            or not 0 <= self.best_epoch <= self.epochs_ran
            or self.optimizer_steps < 1
            or self.training_example_presentations < 1
            or self.validation_example_presentations < 1
            or len(self.history) != self.epochs_ran + 1
            or not all(math.isfinite(value) for value in numeric)
        ):
            raise ValueError("invalid converged full-replay result")

    def as_record(self, *, include_history: bool = True) -> dict[str, object]:
        """Return a JSON-compatible convergence record."""
        record = {
            key: value
            for key, value in asdict(self).items()
            if key != "history"
        }
        if include_history:
            record["history"] = [row.as_record() for row in self.history]
        return record


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
    *,
    maximum_slots: int | None = None,
) -> IntegratorConditionState:
    """Create one independently initialized integrator and optimizer."""
    slot_count = config.maximum_levels if maximum_slots is None else maximum_slots
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, "integrator-init", name))
        integrator = LevelSlotIntegrator(
            input_dim,
            slot_count,
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
    """Train one condition and measure only loss-construction/gradient work."""
    if epochs < 1:
        raise ValueError("integrator training requires at least one epoch")
    before_current = _source_metrics(state.integrator, current, device, config.minibatch_size)
    before_historical = (
        None
        if historical is None
        else _source_metrics(state.integrator, historical, device, config.minibatch_size)
    )
    state.integrator.train()
    historical_count = 0 if historical is None else len(historical.labels)
    chunk_count = math.ceil((len(current.labels) + historical_count) / config.minibatch_size)
    training_forward_example_passes = epochs * (len(current.labels) + historical_count)
    training_forward_calls = epochs * chunk_count * (1 + int(historical is not None))
    training_backward_calls = epochs * chunk_count
    device_indices = [device.index or 0] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = perf_counter()
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_seconds = perf_counter() - training_start
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
        training_forward_example_passes,
        training_forward_example_passes,
        training_forward_calls,
        training_backward_calls,
        training_wall_seconds,
    )


def train_converged_full_replay(
    state: IntegratorConditionState,
    training: IntegratorSupervision,
    validation: IntegratorSupervision,
    config: IntegratorConfig,
    convergence: FullReplayConvergenceConfig,
    seed: int,
    macro_step: int,
    device: torch.device,
    *,
    progress: bool = False,
) -> ConvergedFullReplayResult:
    """Fit on every training row until the frozen validation rule converges."""
    if min(len(training.labels), len(validation.labels)) < 1:
        raise ValueError("full replay requires nonempty training and validation archives")
    if config.learning_rate < convergence.minimum_learning_rate:
        raise ValueError("initial learning rate is below the convergence floor")
    if any(
        abs(float(group["lr"]) - config.learning_rate) > 1.0e-15
        for group in state.optimizer.param_groups
    ):
        raise ValueError("full-replay optimizer did not start at the frozen learning rate")

    initial_training = _source_metrics(
        state.integrator, training, device, config.minibatch_size
    )
    initial_validation = _source_metrics(
        state.integrator, validation, device, config.minibatch_size
    )
    best_parameters = _cpu_parameter_snapshot(state.integrator)
    best_epoch = 0
    best_validation_loss = initial_validation[0]
    significant_reference = initial_validation[0]
    epochs_without_significant_improvement = 0
    minimum_rate_plateau_epochs = 0
    initial_rate = float(state.optimizer.param_groups[0]["lr"])
    history = [
        ConvergenceEpochResult(
            0,
            initial_rate,
            initial_rate,
            initial_training[0],
            initial_training[1],
            initial_validation[0],
            initial_validation[1],
            initial_validation[0],
            True,
            True,
            state.optimizer_steps,
        )
    ]
    stop_reason = "maximum_epochs"
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    epochs = tqdm(
        range(1, convergence.maximum_epochs + 1),
        desc=f"{state.name} convergence",
        disable=not progress,
        leave=False,
        unit="epoch",
    )
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, state.name, macro_step, "dropout"))
        for epoch in epochs:
            learning_rate = float(state.optimizer.param_groups[0]["lr"])
            state.integrator.train()
            generator = torch.Generator().manual_seed(
                named_seed(seed, state.name, macro_step, "full-replay", epoch)
            )
            order = torch.randperm(len(training.labels), generator=generator)
            for offset in range(0, len(order), config.minibatch_size):
                indices = order[offset : offset + config.minibatch_size]
                loss = _batch_loss(state.integrator, training, indices, device)
                state.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    state.integrator.parameters(), config.gradient_clip_norm
                )
                state.optimizer.step()
                state.optimizer_steps += 1

            training_metrics = _source_metrics(
                state.integrator, training, device, config.minibatch_size
            )
            validation_metrics = _source_metrics(
                state.integrator, validation, device, config.minibatch_size
            )
            improved_best = validation_metrics[0] < best_validation_loss
            if improved_best:
                best_validation_loss = validation_metrics[0]
                best_epoch = epoch
                best_parameters = _cpu_parameter_snapshot(state.integrator)
            significant_improvement = (
                validation_metrics[0]
                <= significant_reference - convergence.improvement_delta
            )
            if significant_improvement:
                significant_reference = validation_metrics[0]
                epochs_without_significant_improvement = 0
                minimum_rate_plateau_epochs = 0
            else:
                epochs_without_significant_improvement += 1

            next_learning_rate = learning_rate
            at_minimum_rate = (
                learning_rate <= convergence.minimum_learning_rate + 1.0e-15
            )
            if at_minimum_rate:
                if not significant_improvement:
                    minimum_rate_plateau_epochs += 1
            elif (
                epochs_without_significant_improvement
                >= convergence.learning_rate_patience
            ):
                next_learning_rate = max(
                    convergence.minimum_learning_rate,
                    learning_rate * convergence.learning_rate_factor,
                )
                for group in state.optimizer.param_groups:
                    group["lr"] = next_learning_rate
                epochs_without_significant_improvement = 0
                minimum_rate_plateau_epochs = 0

            history.append(
                ConvergenceEpochResult(
                    epoch,
                    learning_rate,
                    next_learning_rate,
                    training_metrics[0],
                    training_metrics[1],
                    validation_metrics[0],
                    validation_metrics[1],
                    best_validation_loss,
                    improved_best,
                    significant_improvement,
                    state.optimizer_steps,
                )
            )
            epochs.set_postfix(
                best=f"{best_validation_loss:.4f}",
                lr=f"{next_learning_rate:.1e}",
                validation=f"{validation_metrics[0]:.4f}",
            )
            if (
                epoch >= convergence.minimum_epochs
                and at_minimum_rate
                and minimum_rate_plateau_epochs >= convergence.convergence_patience
            ):
                stop_reason = "minimum_learning_rate_plateau"
                break

    state.integrator.load_state_dict(best_parameters, strict=True)
    restored_training = _source_metrics(
        state.integrator, training, device, config.minibatch_size
    )
    restored_validation = _source_metrics(
        state.integrator, validation, device, config.minibatch_size
    )
    epochs_ran = history[-1].epoch
    return ConvergedFullReplayResult(
        stop_reason == "minimum_learning_rate_plateau",
        stop_reason,
        best_epoch,
        epochs_ran,
        restored_training[0],
        restored_training[1],
        restored_validation[0],
        restored_validation[1],
        float(state.optimizer.param_groups[0]["lr"]),
        state.optimizer_steps,
        epochs_ran * len(training.labels),
        (epochs_ran + 2) * len(validation.labels),
        tuple(history),
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


def _cpu_parameter_snapshot(integrator: LevelSlotIntegrator) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in integrator.state_dict().items()
    }


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
    "ConvergedFullReplayResult",
    "ConvergenceEpochResult",
    "FullReplayConvergenceConfig",
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
    "train_converged_full_replay",
    "train_condition",
]
