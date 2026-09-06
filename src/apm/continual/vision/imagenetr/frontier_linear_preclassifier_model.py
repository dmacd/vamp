"""Single-affine integration over node-specific pre-classifier representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    AdaptiveFrontierModel,
    require_frontier_trainable_boundary,
)
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.macro_token_model import CLASS_COUNT, TOKEN_DIMENSION
from apm.continual.vision.imagenetr.model import AdapterVisionModel


class SingleLayerPreclassifierIntegrator(nn.Module):
    """Map concatenated node features directly to global class logits."""

    def __init__(self, classifiers: Sequence[ClassifierRows]) -> None:
        super().__init__()
        rows = tuple(classifiers)
        class_ids = tuple(class_id for classifier in rows for class_id in classifier.class_ids)
        if (
            len(rows) != 5
            or len(class_ids) != 124
            or len(class_ids) != len(set(class_ids))
            or any(classifier.weight.shape[1] != TOKEN_DIMENSION for classifier in rows)
        ):
            raise ValueError("linear integrator requires the exact stage-31 frontier")
        self.active_nodes = len(rows)
        self.input_dimension = self.active_nodes * TOKEN_DIMENSION
        self.affine = nn.Linear(self.input_dimension, CLASS_COUNT).to(rows[0].weight)
        with torch.no_grad():
            self.affine.weight.zero_()
            self.affine.bias.zero_()
            for node_index, classifier in enumerate(rows):
                owned = torch.tensor(
                    classifier.class_ids,
                    dtype=torch.int64,
                    device=self.affine.weight.device,
                )
                first = node_index * TOKEN_DIMENSION
                last = first + TOKEN_DIMENSION
                self.affine.weight[:, first:last].index_copy_(
                    0, owned, classifier.weight
                )
                self.affine.bias.index_copy_(0, owned, classifier.bias)

    def forward(self, features: Tensor, seen_class_mask: Tensor) -> Tensor:
        """Return masked logits from one direct affine feature combination."""
        if (
            features.ndim != 2
            or features.shape[1] != self.input_dimension
            or seen_class_mask.shape != (CLASS_COUNT,)
            or seen_class_mask.dtype != torch.bool
        ):
            raise ValueError("linear integrator inputs do not match its architecture")
        return self.affine(features).masked_fill(~seen_class_mask, -torch.inf)


class LinearPreclassifierFrontierModel(AdaptiveFrontierModel):
    """Five independently adapted ViTs feeding one direct affine classifier."""

    def __init__(
        self,
        nodes: Sequence[BehaviorNode],
        slot_indices: Sequence[int],
        backbone_factory: Callable[[], nn.Module],
        rank: int,
        alpha: int,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            nodes,
            slot_indices,
            backbone_factory,
            rank,
            alpha,
            1,
            0.0,
            seed,
            device,
        )
        self.macro = SingleLayerPreclassifierIntegrator(
            tuple(node_model.classifier.rows() for node_model in self.node_models)
        ).to(device)
        require_frontier_trainable_boundary(self, adapt_lora=True)

    def preclassifier_features(
        self,
        images: Tensor,
        adapt_lora: bool,
        activation_recomputation: bool,
    ) -> tuple[Tensor, ...]:
        """Return each live node's own LoRA-dependent pre-classifier features."""
        if adapt_lora and not all(
            parameter.requires_grad for parameter in self.lora_parameters
        ):
            raise ValueError("requested adaptation differs from the trainable boundary")
        return tuple(
            checkpoint(
                lambda current_images, selected=node_model: selected.features(
                    current_images
                ),
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

    @staticmethod
    def _frozen_features(
        node_model: AdapterVisionModel, images: Tensor
    ) -> Tensor:
        with torch.no_grad():
            return node_model.features(images)

    def forward(
        self, images: Tensor, adapt_lora: bool, activation_recomputation: bool
    ) -> Tensor:
        """Combine five node representations without scores or hidden layers."""
        features = self.preclassifier_features(
            images, adapt_lora, activation_recomputation
        )
        return self.macro(torch.cat(features, dim=1), self.seen_class_mask)


__all__ = [
    "LinearPreclassifierFrontierModel",
    "SingleLayerPreclassifierIntegrator",
]
