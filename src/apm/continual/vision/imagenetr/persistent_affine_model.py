"""Persistent dense integration over a changing adapter-dependent LogT frontier."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter_factors,
    trainable_lora_parameters,
)
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.persistent_integrator_head import PersistentIntegratorHead


CLASS_COUNT = 200
FEATURE_DIMENSION = 768


@dataclass(frozen=True, slots=True)
class CarrySummary:
    """Exact node identities continued or reset at one arrival boundary."""

    continued_node_hashes: tuple[str, ...]
    reset_node_hashes: tuple[str, ...]
    retired_node_hashes: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return canonical JSON-compatible transition evidence."""
        return {
            "continued_node_hashes": list(self.continued_node_hashes),
            "reset_node_hashes": list(self.reset_node_hashes),
            "retired_node_hashes": list(self.retired_node_hashes),
        }


class PersistentFrontierIntegrator(nn.Module):
    """Variable-width node features feeding a persistent affine or two-layer head."""

    def __init__(
        self,
        nodes: Sequence[BehaviorNode],
        slot_indices: Sequence[int],
        backbone_factory: Callable[[], nn.Module],
        rank: int,
        alpha: int,
        device: torch.device,
        hidden_dimension: int = 0,
        initialization_seed: int = 1993,
    ) -> None:
        super().__init__()
        ordered = tuple(
            sorted(zip(nodes, slot_indices, strict=True), key=lambda pair: pair[1])
        )
        represented_tasks = tuple(
            task for node, _slot in ordered for task in node.represented_task_ids
        )
        if (
            not ordered
            or len(ordered) > 6
            or len({slot for _node, slot in ordered}) != len(ordered)
            or any(not 0 <= slot < 6 for _node, slot in ordered)
            or tuple(sorted(represented_tasks)) != tuple(range(len(represented_tasks)))
        ):
            raise ValueError("persistent affine frontier is not a task-prefix LogT bank")
        models: list[AdapterVisionModel] = []
        for node, _slot in ordered:
            model = AdapterVisionModel(
                backbone_factory(),
                node.classifier.class_ids,
                rank,
                alpha,
                0.0,
                0,
                node.classifier,
            )
            load_adapter_factors(model, node.adapter)
            model.classifier.weight.requires_grad_(False)
            model.classifier.bias.requires_grad_(False)
            model.eval()
            models.append(model)
        self.node_models = nn.ModuleList(models)
        self.node_hashes = tuple(node.node_hash for node, _slot in ordered)
        self.slot_indices = tuple(slot for _node, slot in ordered)
        self.class_ids = tuple(node.classifier.class_ids for node, _slot in ordered)
        self.seen_class_ids = tuple(sorted(class_id for ids in self.class_ids for class_id in ids))
        if self.seen_class_ids != tuple(range(4 * len(represented_tasks))):
            raise ValueError("persistent affine frontier classes are not a complete prefix")
        self.register_buffer(
            "seen_class_mask",
            torch.zeros(CLASS_COUNT, dtype=torch.bool).index_fill(
                0, torch.tensor(self.seen_class_ids, dtype=torch.int64), True
            ),
            persistent=True,
        )
        weight = torch.zeros(CLASS_COUNT, len(ordered) * FEATURE_DIMENSION)
        bias = torch.zeros(CLASS_COUNT)
        for index, node_model in enumerate(self.node_models):
            rows = node_model.classifier.rows()
            owned = torch.tensor(rows.class_ids, dtype=torch.int64)
            first = index * FEATURE_DIMENSION
            weight[:, first:first + FEATURE_DIMENSION].index_copy_(0, owned, rows.weight)
            bias.index_copy_(0, owned, rows.bias)
        self.integrator = PersistentIntegratorHead(weight, bias, hidden_dimension, initialization_seed)
        self.to(device)
        require_persistent_trainable_boundary(self)

    @property
    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return all and only live frontier LoRA tensors."""
        return tuple(
            parameter
            for node_model in self.node_models
            for parameter in trainable_lora_parameters(node_model)
        )

    @property
    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return affine parameters followed by all live LoRA factors."""
        return tuple(self.integrator.parameters()) + self.lora_parameters

    def set_training_mode(self) -> None:
        """Train the affine map while retaining deterministic ViT inference state."""
        self.integrator.train()
        for node_model in self.node_models:
            node_model.eval()

    def set_evaluation_mode(self) -> None:
        """Put every active component into deterministic evaluation mode."""
        self.integrator.eval()
        for node_model in self.node_models:
            node_model.eval()

    @staticmethod
    def _frozen_features(node_model: AdapterVisionModel, images: Tensor) -> Tensor:
        with torch.no_grad():
            return node_model.features(images)

    def node_features(
        self,
        images: Tensor,
        adapt_lora: bool,
        activation_recomputation: bool,
    ) -> tuple[Tensor, ...]:
        """Return each live node's own adapter-dependent pre-classifier vector."""
        return tuple(
            checkpoint(
                lambda current, selected=node_model: selected.features(current),
                images,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if adapt_lora and activation_recomputation
            else node_model.features(images)
            if adapt_lora
            else self._frozen_features(node_model, images)
            for node_model in self.node_models
        )

    def forward_components(
        self,
        images: Tensor,
        adapt_lora: bool,
        activation_recomputation: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Return integrated logits plus reused node features and local scores."""
        features = self.node_features(images, adapt_lora, activation_recomputation)
        logits = self.integrator(torch.cat(features, dim=1)).masked_fill(
            ~self.seen_class_mask, -torch.inf
        )
        local_scores = tuple(
            node_model.classifier(node_features).float()
            for node_model, node_features in zip(self.node_models, features, strict=True)
        )
        return logits, features, local_scores

    def forward(
        self,
        images: Tensor,
        adapt_lora: bool,
        activation_recomputation: bool,
    ) -> Tensor:
        """Return global task-free logits from the single affine integrator."""
        return self.forward_components(
            images, adapt_lora, activation_recomputation
        )[0]


def require_persistent_trainable_boundary(model: PersistentFrontierIntegrator) -> None:
    """Fail unless only the dense integrator and every live LoRA can update."""
    head_names = {f"integrator.{name}" for name, _parameter in model.integrator.named_parameters()}
    expected = head_names | {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("node_models.")
        and (name.endswith("lora_a") or name.endswith("lora_b"))
    }
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != expected or len(expected) != len(head_names) + 48 * len(model.node_models):
        raise ValueError("persistent affine trainable parameters crossed the boundary")


def export_model_state(model: PersistentFrontierIntegrator) -> dict[str, object]:
    """Capture the complete continuing state without duplicating frozen bases."""
    return {
        "hidden_dimension": model.integrator.hidden_dimension,
        "integrator_state": {
            name: value.detach().cpu().clone()
            for name, value in model.integrator.state_dict().items()
        },
        "node_adapters": {
            node_hash: {
                module: LoRAFactors(
                    factors.a.detach().cpu().clone(),
                    factors.b.detach().cpu().clone(),
                    factors.scale,
                )
                for module, factors in adapter_factors(node_model).items()
            }
            for node_hash, node_model in zip(
                model.node_hashes, model.node_models, strict=True
            )
        },
        "node_hashes": model.node_hashes,
        "seen_class_ids": model.seen_class_ids,
        "slot_indices": model.slot_indices,
    }


def carry_model_state(
    model: PersistentFrontierIntegrator,
    previous: Mapping[str, object] | None,
) -> CarrySummary:
    """Carry surviving node/head state and retain source initialization for replacements."""
    if previous is None:
        return CarrySummary((), model.node_hashes, ())
    old_hashes = tuple(str(value) for value in previous["node_hashes"])
    old_seen = tuple(int(value) for value in previous["seen_class_ids"])
    head_state = previous["integrator_state"]
    old_adapters = previous["node_adapters"]
    if (
        not isinstance(head_state, Mapping)
        or previous.get("hidden_dimension") != model.integrator.hidden_dimension
        or not all(isinstance(value, Tensor) for value in head_state.values())
        or not isinstance(old_adapters, Mapping)
        or len(set(old_hashes)) != len(old_hashes)
    ):
        raise ValueError("previous persistent affine state is malformed")
    model.integrator.carry(head_state, old_hashes, model.node_hashes, old_seen)
    with torch.no_grad():
        for new_index, node_hash in enumerate(model.node_hashes):
            if node_hash not in old_hashes:
                continue
            raw = old_adapters.get(node_hash)
            if not isinstance(raw, Mapping) or not all(
                isinstance(value, LoRAFactors) for value in raw.values()
            ):
                raise ValueError("continued node lacks authenticated adapter factors")
            load_adapter_factors(model.node_models[new_index], raw)
    return node_transition(old_hashes, model.node_hashes)


def node_transition(
    previous_hashes: Sequence[str], current_hashes: Sequence[str]
) -> CarrySummary:
    """Classify a frontier change into exact continuations, resets, and retirements."""
    previous = tuple(previous_hashes)
    current = tuple(current_hashes)
    if len(previous) != len(set(previous)) or len(current) != len(set(current)):
        raise ValueError("frontier node hashes must be unique")
    previous_set = frozenset(previous)
    current_set = frozenset(current)
    return CarrySummary(
        tuple(node_hash for node_hash in current if node_hash in previous_set),
        tuple(node_hash for node_hash in current if node_hash not in previous_set),
        tuple(node_hash for node_hash in previous if node_hash not in current_set),
    )


def load_exact_model_state(
    model: PersistentFrontierIntegrator, state: Mapping[str, object]
) -> None:
    """Restore one same-frontier checkpoint without applying transition rules."""
    if (
        tuple(state.get("node_hashes", ())) != model.node_hashes
        or tuple(state.get("slot_indices", ())) != model.slot_indices
        or tuple(state.get("seen_class_ids", ())) != model.seen_class_ids
        or state.get("hidden_dimension") != model.integrator.hidden_dimension
    ):
        raise ValueError("checkpoint frontier differs from the requested model")
    head_state, adapters = state.get("integrator_state"), state.get("node_adapters")
    if not isinstance(head_state, Mapping) or not isinstance(adapters, Mapping):
        raise ValueError("checkpoint model state is malformed")
    model.integrator.load_state_dict(head_state, strict=True)
    for node_hash, node_model in zip(model.node_hashes, model.node_models, strict=True):
        raw = adapters.get(node_hash)
        if not isinstance(raw, Mapping) or not all(
            isinstance(value, LoRAFactors) for value in raw.values()
        ):
            raise ValueError("checkpoint adapter state is malformed")
        load_adapter_factors(node_model, raw)


def raw_union_logits(
    model: PersistentFrontierIntegrator,
    local_scores: Sequence[Tensor],
) -> Tensor:
    """Return the parameter-free union of frozen local classifier rows."""
    if len(local_scores) != len(model.class_ids):
        raise ValueError("local-score count differs from the live frontier")
    output = torch.full(
        (len(local_scores[0]), CLASS_COUNT),
        -torch.inf,
        dtype=torch.float32,
        device=local_scores[0].device,
    )
    for class_ids, scores in zip(model.class_ids, local_scores, strict=True):
        output[:, torch.tensor(class_ids, device=output.device)] = scores
    return output


def true_node_logits(
    model: PersistentFrontierIntegrator,
    local_scores: Sequence[Tensor],
    labels: Tensor,
) -> Tensor:
    """Return label-aware node logits for diagnostic evaluation only."""
    output = torch.full(
        (len(labels), CLASS_COUNT),
        -torch.inf,
        dtype=torch.float32,
        device=labels.device,
    )
    for class_ids, scores in zip(model.class_ids, local_scores, strict=True):
        owned = torch.zeros(CLASS_COUNT, dtype=torch.bool, device=labels.device)
        indices = torch.tensor(class_ids, dtype=torch.int64, device=labels.device)
        owned[indices] = True
        selected = owned[labels]
        output[selected[:, None] & owned[None, :]] = scores[selected].flatten()
    return output


__all__ = [
    "CarrySummary",
    "PersistentFrontierIntegrator",
    "carry_model_state",
    "export_model_state",
    "load_exact_model_state",
    "node_transition",
    "raw_union_logits",
    "require_persistent_trainable_boundary",
    "true_node_logits",
]
