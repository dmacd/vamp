"""Frozen fact-level effects for the descriptive sealed-test report."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryResult,
)
from apm.data.text.tinyworlds_q_semantic.final_protocol import (
    REGISTERED_FINAL_EVALUATION_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.statistics import (
    BootstrapEstimate,
    FactObservation,
    acquisition_effect,
    average_paraphrases,
    retention_effect,
    specificity_effect,
)


def compute_registered_final_effects(
    results: tuple[SemanticQueryResult, ...],
    preset: QueryExperimentPreset,
) -> tuple[BootstrapEstimate, ...]:
    """Compute every preregistered acquisition, retention, and specificity effect."""
    protocol = REGISTERED_FINAL_EVALUATION_PROTOCOL
    observations = average_paraphrases(results)
    base = tuple(
        item
        for item in observations
        if item.stage == 0
        and item.method == "base"
        and item.adapter_concept_id is None
    )
    if len(base) != 12 * preset.active_world_count:
        raise ValueError("final analysis base facts are incomplete")
    effects = tuple(
        renamed
        for method in protocol.acquisition_methods
        for acquisition_rows, final_rows in (
            (
                _acquisition_observations(observations, method, preset),
                _final_observations(observations, method, preset),
            ),
        )
        for metric in protocol.effect_metrics
        for estimate in (
            acquisition_effect(
                base,
                acquisition_rows,
                metric,  # type: ignore[arg-type]
                replicates=protocol.bootstrap_replicates,
            ),
            retention_effect(
                acquisition_rows,
                final_rows,
                metric,  # type: ignore[arg-type]
                replicates=protocol.bootstrap_replicates,
            ),
        )
        for renamed in (_rename_effect(estimate, method),)
    )
    specificity_rows = tuple(
        item
        for item in observations
        if item.stage == preset.active_world_count
        and item.method == protocol.specificity_method
    )
    specificity = tuple(
        _rename_effect(
            specificity_effect(
                specificity_rows,
                metric,  # type: ignore[arg-type]
                replicates=protocol.bootstrap_replicates,
            ),
            protocol.specificity_method,
        )
        for metric in protocol.effect_metrics
    )
    return effects + specificity


def _acquisition_observations(
    observations: tuple[FactObservation, ...],
    method: str,
    preset: QueryExperimentPreset,
) -> tuple[FactObservation, ...]:
    selected = tuple(
        item
        for concept_index, concept_id in enumerate(preset.concept_ids, start=1)
        for item in observations
        if item.stage == concept_index
        and item.method == method
        and item.concept_id == concept_id
        and _matching_adapter(item, method)
    )
    if len(selected) != 12 * preset.active_world_count:
        raise ValueError(f"final acquisition facts are incomplete for {method}")
    return selected


def _final_observations(
    observations: tuple[FactObservation, ...],
    method: str,
    preset: QueryExperimentPreset,
) -> tuple[FactObservation, ...]:
    selected = tuple(
        item
        for item in observations
        if item.stage == preset.active_world_count
        and item.method == method
        and _matching_adapter(item, method)
    )
    if len(selected) != 12 * preset.active_world_count:
        raise ValueError(f"final retention facts are incomplete for {method}")
    return selected


def _matching_adapter(observation: FactObservation, method: str) -> bool:
    return (
        observation.adapter_concept_id == observation.concept_id
        if method == "independent"
        else observation.adapter_concept_id is None
    )


def _rename_effect(
    estimate: BootstrapEstimate,
    method: str,
) -> BootstrapEstimate:
    return BootstrapEstimate(
        metric=f"{estimate.metric}:{method}",
        point=estimate.point,
        lower=estimate.lower,
        upper=estimate.upper,
        replicate_count=estimate.replicate_count,
        fact_count=estimate.fact_count,
        world_count=estimate.world_count,
    )


__all__ = ["compute_registered_final_effects"]
