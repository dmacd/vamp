"""Ordered-manifest curriculum plans and fixed-capacity tensor masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.scaling import evaluation_schedule


@dataclass(frozen=True, slots=True)
class ConceptStage:
    """One dynamic continual-learning transition and its active prefix."""

    stage: int
    concept_id: str
    learned_concept_ids: tuple[str, ...]
    node_index: int
    edge_index: int

    def __post_init__(self) -> None:
        if (
            type(self.stage) is not int
            or self.stage <= 0
            or len(self.learned_concept_ids) != self.stage
            or self.learned_concept_ids[-1] != self.concept_id
            or self.node_index != self.stage
            or self.edge_index != self.stage - 1
        ):
            raise ValueError("concept stage does not match its ordered prefix")


@dataclass(frozen=True, eq=False, slots=True)
class CapacityMasks:
    """Read-only max-capacity node/edge masks for one active stage."""

    node_mask: np.ndarray
    edge_mask: np.ndarray

    def __post_init__(self) -> None:
        node_mask = np.asarray(self.node_mask, dtype=np.bool_).copy()
        edge_mask = np.asarray(self.edge_mask, dtype=np.bool_).copy()
        if node_mask.ndim != 1 or edge_mask.ndim != 1:
            raise ValueError("capacity masks must be one-dimensional")
        node_mask.flags.writeable = False
        edge_mask.flags.writeable = False
        object.__setattr__(self, "node_mask", node_mask)
        object.__setattr__(self, "edge_mask", edge_mask)


def concept_stages(preset: QueryExperimentPreset) -> tuple[ConceptStage, ...]:
    """Derive every training stage from the ordered active concept manifest."""
    return tuple(
        ConceptStage(
            stage=stage,
            concept_id=concept_id,
            learned_concept_ids=preset.concept_ids[:stage],
            node_index=stage,
            edge_index=stage - 1,
        )
        for stage, concept_id in enumerate(preset.concept_ids, start=1)
    )


def capacity_masks(
    preset: QueryExperimentPreset,
    stage: int,
) -> CapacityMasks:
    """Build padded node/edge masks without five-world constants."""
    if type(stage) is not int or not 0 <= stage <= preset.active_world_count:
        raise ValueError("capacity-mask stage lies outside the active manifest")
    node_mask = np.arange(preset.max_nodes) <= stage
    edge_mask = np.arange(preset.max_edges) < stage
    return CapacityMasks(node_mask, edge_mask)


def validate_active_catalog_prefix(
    catalog: SemanticQueryCatalog | ValidationCatalogView,
    preset: QueryExperimentPreset,
) -> None:
    """Allow one large catalog/partition to run any preserved active prefix."""
    if catalog.concept_ids[: preset.active_world_count] != preset.concept_ids:
        raise ValueError("experiment concepts are not an ordered catalog prefix")


def progress_totals(
    preset: QueryExperimentPreset,
    *,
    queries_per_world: int,
    method_count: int,
) -> dict[str, int]:
    """Derive training and evaluation progress totals from world count."""
    if any(
        type(value) is not int or value <= 0
        for value in (queries_per_world, method_count)
    ):
        raise ValueError("progress query and method counts must be positive")
    return {
        "adapter_updates": 3 * preset.active_world_count * preset.adapter_updates,
        "evaluation_rows": len(evaluation_schedule(preset))
        * queries_per_world
        * method_count,
        "parent_probes": sum(
            stage * preset.parent_probe_count
            for stage in range(1, preset.active_world_count + 1)
        ),
    }


__all__ = [
    "CapacityMasks",
    "ConceptStage",
    "capacity_masks",
    "concept_stages",
    "progress_totals",
    "validate_active_catalog_prefix",
]
