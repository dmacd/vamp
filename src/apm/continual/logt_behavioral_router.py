"""Masked level-slot behavioral routing and fixed-budget replay."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.logt_evidence_bank import TemporalNode
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoBaseState,
    top_two_base_state,
    top_two_hidden_logits,
)
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_router_config import RouterConfig
from apm.experiments.vamp_logt_router_data import ExampleBatch, named_seed


SLOT_OUTPUT_DIM = 10


@dataclass(frozen=True, slots=True)
class RouterSupervision:
    """Detached all-node behavior and oracle targets for one example batch."""

    features: Tensor
    node_logits: Tensor
    node_losses: Tensor
    hard_targets: Tensor
    soft_targets: Tensor
    active_mask: Tensor

    def __post_init__(self) -> None:
        rows, levels, classes = self.node_logits.shape
        if (
            self.features.shape[0] != rows
            or classes != SLOT_OUTPUT_DIM
            or self.node_losses.shape != (rows, levels)
            or self.hard_targets.shape != (rows,)
            or self.soft_targets.shape != (rows, levels)
            or self.active_mask.shape != (levels,)
            or self.active_mask.dtype != torch.bool
            or any(value.requires_grad for value in self.tensors)
            or not torch.isfinite(self.features).all()
            or not torch.isfinite(self.node_logits).all()
            or not torch.isfinite(self.soft_targets).all()
        ):
            raise ValueError("router supervision tensors are malformed or attached")
        if torch.any(self.hard_targets < 0) or torch.any(self.hard_targets >= levels):
            raise ValueError("hard targets are outside the fixed level slots")

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.features,
            self.node_logits,
            self.node_losses,
            self.hard_targets,
            self.soft_targets,
            self.active_mask,
        )


@dataclass(frozen=True, slots=True)
class ReplayDraw:
    """One exact historical selection with sampling diagnostics."""

    batch: ExampleBatch
    archive_indices: tuple[int, ...]
    duplicate_draws: int
    range_draw_counts: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if (
            len(self.archive_indices) != len(self.batch.labels)
            or self.duplicate_draws
            != len(self.archive_indices) - len(set(self.archive_indices))
            or self.duplicate_draws < 0
        ):
            raise ValueError("historical replay diagnostics do not match the draw")


@dataclass(slots=True)
class RouterConditionState:
    """One independent learned router and optimizer."""

    name: str
    router: "LevelSlotRouter"
    optimizer: torch.optim.AdamW
    optimizer_steps: int = 0


@dataclass(frozen=True, slots=True)
class RouterTrainingResult:
    """Bounded optimization evidence for one macro-step."""

    mean_first_epoch_loss: float
    mean_last_epoch_loss: float
    optimizer_steps: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.mean_first_epoch_loss)
            or not math.isfinite(self.mean_last_epoch_loss)
            or self.optimizer_steps < 1
        ):
            raise ValueError("router training did not produce finite bounded work")


class LevelSlotRouter(nn.Module):
    """Over-capacity MLP whose output classes are persistent LogT levels."""

    def __init__(
        self,
        input_dim: int,
        maximum_levels: int,
        hidden_widths: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or maximum_levels < 1 or len(hidden_widths) != 3:
            raise ValueError("router dimensions require three positive hidden widths")
        self.input_dim = input_dim
        self.maximum_levels = maximum_levels
        first, second, third = hidden_widths
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.GELU(),
            nn.LayerNorm(first),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.GELU(),
            nn.LayerNorm(second),
            nn.Dropout(dropout),
            nn.Linear(second, third),
            nn.GELU(),
            nn.Linear(third, maximum_levels),
        )

    def unmasked_logits(self, features: Tensor) -> Tensor:
        """Return raw fixed-slot logits for diagnostics."""
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("router received an incompatible feature matrix")
        return self.network(features)

    def forward(self, features: Tensor, active_mask: Tensor) -> Tensor:
        """Return logits with inactive level classes made impossible."""
        if (
            active_mask.shape != (self.maximum_levels,)
            or active_mask.dtype != torch.bool
            or not bool(active_mask.any())
        ):
            raise ValueError("router requires one nonempty fixed-level mask")
        return self.unmasked_logits(features).masked_fill(~active_mask[None, :], -torch.inf)


def create_condition_state(
    name: str,
    input_dim: int,
    config: RouterConfig,
    seed: int,
    device: torch.device,
) -> RouterConditionState:
    """Create one independently initialized router and AdamW state."""
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, "router-init", name))
        router = LevelSlotRouter(
            input_dim,
            config.maximum_levels,
            config.hidden_widths,
            config.dropout,
        ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    return RouterConditionState(name, router, optimizer)


def frozen_trunk_features(
    model: AddressCNN,
    images: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Run the frozen convolutional trunk without retaining an autograd graph."""
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28) or batch_size < 1:
        raise ValueError("frozen trunk received invalid images or batch size")
    rows = []
    model.to(device)
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            rows.append(model.trunk_features(images[offset : offset + batch_size].to(device)).cpu())
    return torch.cat(rows).detach()


def build_router_supervision(
    nodes: tuple[TemporalNode, ...],
    adapters: Mapping[str, TopTwoAdapterState],
    trunk_features: Tensor,
    labels: Tensor,
    base: TopTwoBaseState,
    maximum_levels: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> RouterSupervision:
    """Recompute detached level-slot behavior and current-frontier oracle targets."""
    if (
        not nodes
        or set(adapters) != {node.node_id for node in nodes}
        or len({node.level for node in nodes}) != len(nodes)
        or any(not 0 <= node.level < maximum_levels for node in nodes)
        or labels.shape != (len(trunk_features),)
        or temperature <= 0.0
        or batch_size < 1
    ):
        raise ValueError("active bank cannot generate aligned router supervision")
    target_base = top_two_base_state(*base.tensors, device=device)
    hidden_dim = target_base.hidden_dim
    features = torch.zeros(
        (len(labels), maximum_levels, hidden_dim + SLOT_OUTPUT_DIM + 1),
        dtype=torch.float32,
    )
    node_logits = torch.zeros(
        (len(labels), maximum_levels, SLOT_OUTPUT_DIM), dtype=torch.float32
    )
    active_mask = torch.zeros(maximum_levels, dtype=torch.bool)
    for node in sorted(nodes, key=lambda value: value.level):
        adapter = TopTwoAdapterState(
            *(tensor.detach().to(device).clone() for tensor in adapters[node.node_id].tensors)
        )
        hidden_rows, logit_rows = [], []
        with torch.inference_mode():
            for offset in range(0, len(labels), batch_size):
                hidden, logits = top_two_hidden_logits(
                    trunk_features[offset : offset + batch_size].to(device),
                    target_base,
                    adapter,
                )
                hidden_rows.append(hidden.cpu())
                logit_rows.append(logits.cpu())
        hidden = torch.cat(hidden_rows)
        logits = torch.cat(logit_rows)
        features[:, node.level, :-1] = torch.cat(
            (
                F.layer_norm(hidden, (hidden_dim,)),
                F.log_softmax(logits, dim=1),
            ),
            dim=1,
        )
        features[:, node.level, -1] = 1.0
        node_logits[:, node.level] = logits
        active_mask[node.level] = True
    expanded_labels = labels[:, None, None].expand(-1, maximum_levels, 1)
    losses = -F.log_softmax(node_logits, dim=2).gather(2, expanded_labels).squeeze(2)
    losses[:, ~active_mask] = torch.inf
    hard_targets = losses.argmin(dim=1)
    minimum = losses.min(dim=1, keepdim=True).values
    teacher_logits = -(losses - minimum) / temperature
    teacher_logits[:, ~active_mask] = -torch.inf
    return RouterSupervision(
        features.flatten(1).detach(),
        node_logits.detach(),
        losses.detach(),
        hard_targets.detach(),
        F.softmax(teacher_logits, dim=1).detach(),
        active_mask.detach(),
    )


def sample_example_balanced(
    archive: ExampleBatch,
    count: int,
    seed: int,
    current_macro_step: int,
) -> ReplayDraw:
    """Draw uniformly from all strictly historical router examples."""
    _validate_archive(archive, count, current_macro_step)
    indices = _sample_pool(tuple(range(len(archive.labels))), count, seed)
    return _replay_draw(archive, indices, ())


def sample_range_balanced(
    archive: ExampleBatch,
    nodes: tuple[TemporalNode, ...],
    count: int,
    seed: int,
    current_macro_step: int,
) -> ReplayDraw:
    """Allocate a fixed replay budget uniformly over nonempty active ranges."""
    _validate_archive(archive, count, current_macro_step)
    historical_steps = archive.macro_steps.numpy() - 1
    ranges = tuple(
        (node.first_block, min(node.last_block, current_macro_step - 2))
        for node in sorted(nodes, key=lambda value: value.first_block)
        if node.first_block <= current_macro_step - 2
    )
    pools = tuple(
        tuple(
            int(index)
            for index in np.flatnonzero(
                (historical_steps >= first) & (historical_steps <= last)
            )
        )
        for first, last in ranges
    )
    if not pools or any(not pool for pool in pools):
        raise ValueError("active historical ranges do not cover the replay archive")
    range_indices = list(range(len(ranges)))
    rng = np.random.default_rng(seed)
    if len(range_indices) > count:
        start = (current_macro_step * count) % len(range_indices)
        range_indices = [range_indices[(start + offset) % len(range_indices)] for offset in range(count)]
    else:
        range_indices = [int(value) for value in rng.permutation(range_indices)]
    base_quota, remainder = divmod(count, len(range_indices))
    quotas = {
        range_index: base_quota + (1 if offset < remainder else 0)
        for offset, range_index in enumerate(range_indices)
    }
    selected = tuple(
        archive_index
        for range_index in range_indices
        for archive_index in _sample_pool(
            pools[range_index],
            quotas[range_index],
            named_seed(seed, "range", range_index),
        )
    )
    diagnostics = tuple(
        (ranges[index][0], ranges[index][1], quotas.get(index, 0))
        for index in range(len(ranges))
    )
    return _replay_draw(archive, selected, diagnostics)


def train_condition(
    state: RouterConditionState,
    current: RouterSupervision,
    historical: RouterSupervision | None,
    target_type: str,
    epochs: int,
    config: RouterConfig,
    seed: int,
    macro_step: int,
    device: torch.device,
) -> RouterTrainingResult:
    """Optimize one router for a fixed number of source-balanced epochs."""
    if target_type not in {"hard", "soft"} or epochs < 1:
        raise ValueError("unknown router target type or empty training schedule")
    if historical is not None and not torch.equal(current.active_mask, historical.active_mask):
        raise ValueError("current and historical targets use different active frontiers")
    state.router.train()
    epoch_means = []
    device_indices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_indices):
        torch.manual_seed(named_seed(seed, state.name, macro_step, "dropout"))
        for epoch in range(epochs):
            chunks = _training_chunks(
                len(current.features),
                0 if historical is None else len(historical.features),
                config.minibatch_size,
                named_seed(seed, state.name, macro_step, "minibatches", epoch),
            )
            losses = []
            for current_ids, historical_ids in chunks:
                current_scale = len(current_ids) * len(chunks) / len(current.features)
                source_losses = [
                    current_scale * _target_loss(
                        state.router(
                            current.features[current_ids].to(device),
                            current.active_mask.to(device),
                        ),
                        current,
                        current_ids,
                        target_type,
                        device,
                    )
                ]
                if historical is not None:
                    historical_scale = (
                        len(historical_ids) * len(chunks) / len(historical.features)
                    )
                    source_losses.append(
                        historical_scale * _target_loss(
                            state.router(
                                historical.features[historical_ids].to(device),
                                historical.active_mask.to(device),
                            ),
                            historical,
                            historical_ids,
                            target_type,
                            device,
                        )
                    )
                loss = torch.stack(source_losses).mean()
                state.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    state.router.parameters(), config.gradient_clip_norm
                )
                state.optimizer.step()
                state.optimizer_steps += 1
                losses.append(float(loss.detach().item()))
            epoch_means.append(float(np.mean(losses)))
    return RouterTrainingResult(epoch_means[0], epoch_means[-1], state.optimizer_steps)


def router_selections(
    router: LevelSlotRouter,
    supervision: RouterSupervision,
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return masked selections, distributions, and unmasked inactive attempts."""
    router.eval()
    selections, probabilities, inactive = [], [], []
    mask = supervision.active_mask.to(device)
    with torch.inference_mode():
        for offset in range(0, len(supervision.features), batch_size):
            features = supervision.features[offset : offset + batch_size].to(device)
            raw = router.unmasked_logits(features)
            masked = raw.masked_fill(~mask[None, :], -torch.inf)
            raw_choices = raw.argmax(dim=1)
            selections.append(masked.argmax(dim=1).cpu())
            probabilities.append(F.softmax(masked, dim=1).cpu())
            inactive.append((~mask[raw_choices]).cpu())
    return torch.cat(selections), torch.cat(probabilities), torch.cat(inactive)


def _validate_archive(archive: ExampleBatch, count: int, current_macro_step: int) -> None:
    if (
        count < 1
        or not len(archive.labels)
        or bool(torch.any(archive.macro_steps >= current_macro_step))
    ):
        raise ValueError("replay requires a nonempty strictly historical archive")


def _sample_pool(pool: tuple[int, ...], count: int, seed: int) -> tuple[int, ...]:
    if not pool or count < 0:
        raise ValueError("cannot sample an invalid replay pool")
    if count == 0:
        return ()
    generator = np.random.default_rng(seed)
    order = tuple(int(value) for value in generator.permutation(pool))
    if count <= len(order):
        return order[:count]
    extra = generator.choice(np.asarray(pool), size=count - len(order), replace=True)
    return (*order, *(int(value) for value in extra))


def _replay_draw(
    archive: ExampleBatch,
    indices: tuple[int, ...],
    range_counts: tuple[tuple[int, int, int], ...],
) -> ReplayDraw:
    selected = torch.tensor(indices, dtype=torch.int64)
    return ReplayDraw(
        archive.select(selected),
        indices,
        len(indices) - len(set(indices)),
        range_counts,
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
    current = torch.tensor_split(torch.randperm(current_count, generator=generator), chunk_count)
    if historical_count:
        historical = torch.tensor_split(
            torch.randperm(historical_count, generator=generator), chunk_count
        )
    else:
        historical = tuple(torch.empty(0, dtype=torch.int64) for _ in range(chunk_count))
    if any(len(left) + len(right) > maximum_batch_size for left, right in zip(current, historical)):
        raise RuntimeError("source-balanced router minibatch exceeded its fixed bound")
    return tuple(zip(current, historical, strict=True))


def _target_loss(
    logits: Tensor,
    supervision: RouterSupervision,
    indices: Tensor,
    target_type: str,
    device: torch.device,
) -> Tensor:
    if target_type == "hard":
        return F.cross_entropy(logits, supervision.hard_targets[indices].to(device))
    active = supervision.active_mask
    targets = supervision.soft_targets[indices][:, active].to(device)
    return -(
        targets * F.log_softmax(logits[:, active.to(device)], dim=1)
    ).sum(dim=1).mean()


__all__ = [
    "LevelSlotRouter",
    "ReplayDraw",
    "RouterConditionState",
    "RouterSupervision",
    "RouterTrainingResult",
    "build_router_supervision",
    "create_condition_state",
    "frozen_trunk_features",
    "router_selections",
    "sample_example_balanced",
    "sample_range_balanced",
    "train_condition",
]
