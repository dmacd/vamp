"""Fact-level semantic summaries and deterministic equal-world bootstraps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import re
from typing import Literal

import numpy as np

from apm.data.text.tinyworlds_p.normalization import normalize_story_identity
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    ConceptDefinition,
    QueryDirection,
    SemanticFact,
    SemanticQueryCatalog,
    SemanticQueryResult,
)


CANONICAL_BOOTSTRAP_REPLICATES = 10_000
FactMetricName = Literal["accuracy", "margin", "router_accuracy", "routed_regret"]


@dataclass(frozen=True, slots=True)
class FactObservation:
    """All paraphrases averaged into one fact-level statistical unit."""

    stage: int
    method: str
    split: str
    adapter_concept_id: str | None
    concept_id: str
    fact_id: str
    template_count: int
    accuracy: float
    margin: float
    router_accuracy: float | None
    routed_regret: float | None

    def __post_init__(self) -> None:
        if type(self.template_count) is not int or self.template_count <= 0:
            raise ValueError("fact observations require at least one paraphrase")
        if not 0.0 <= self.accuracy <= 1.0 or not isfinite(self.margin):
            raise ValueError("fact accuracy or margin is invalid")
        if self.router_accuracy is not None and not 0.0 <= self.router_accuracy <= 1.0:
            raise ValueError("fact router accuracy must lie in [0, 1]")
        if self.routed_regret is not None and not isfinite(self.routed_regret):
            raise ValueError("fact routed regret must be finite")

    def metric(self, name: FactMetricName) -> float:
        """Return one named fact-level metric, rejecting unavailable routing values."""
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"fact metric {name} is unavailable")
        return float(value)


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    """One equal-world fact-resampled point estimate and percentile interval."""

    metric: str
    point: float
    lower: float
    upper: float
    replicate_count: int
    fact_count: int
    world_count: int

    def __post_init__(self) -> None:
        if any(not isfinite(value) for value in (self.point, self.lower, self.upper)):
            raise ValueError("bootstrap estimates must be finite")
        if self.lower > self.upper:
            raise ValueError("bootstrap interval bounds are reversed")
        if any(
            type(value) is not int or value <= 0
            for value in (self.replicate_count, self.fact_count, self.world_count)
        ):
            raise ValueError("bootstrap dimensions must be positive")

    def as_record(self) -> dict[str, object]:
        """Return one canonical report-ready estimate."""
        return {
            "fact_count": self.fact_count,
            "lower": self.lower,
            "metric": self.metric,
            "point": self.point,
            "replicate_count": self.replicate_count,
            "upper": self.upper,
            "world_count": self.world_count,
        }


@dataclass(frozen=True, slots=True)
class GenerationInspection:
    """One exact-trigger generation diagnostic without an external judge."""

    concept_id: str
    prompt: str
    output: str
    recalled_fact_ids: tuple[str, ...]
    recall: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError("generation trigger recall must lie in [0, 1]")

    def as_record(self) -> dict[str, object]:
        """Return the complete prompt, output, and exact-trigger result."""
        return {
            "concept_id": self.concept_id,
            "output": self.output,
            "prompt": self.prompt,
            "recall": self.recall,
            "recalled_fact_ids": list(self.recalled_fact_ids),
        }


def average_paraphrases(
    results: tuple[SemanticQueryResult, ...],
) -> tuple[FactObservation, ...]:
    """Collapse templates into facts before any statistical aggregation."""
    return _average_paraphrases(
        results,
        validation_template_count=3,
        test_template_count=5,
    )


def average_direction_paraphrases(
    results: tuple[SemanticQueryResult, ...],
    direction: QueryDirection,
) -> tuple[FactObservation, ...]:
    """Collapse only one direction while retaining fact-level statistical units."""
    if direction not in ("forward", "reverse"):
        raise ValueError("directional fact aggregation requires forward or reverse")
    selected = tuple(row for row in results if row.direction == direction)
    return _average_paraphrases(
        selected,
        validation_template_count=2 if direction == "forward" else 1,
        test_template_count=3 if direction == "forward" else 2,
    )


def _average_paraphrases(
    results: tuple[SemanticQueryResult, ...],
    *,
    validation_template_count: int,
    test_template_count: int,
) -> tuple[FactObservation, ...]:
    if not results:
        raise ValueError("fact aggregation requires semantic query results")
    keys = tuple(
        sorted(
            {
                (
                    row.stage,
                    row.method,
                    row.split,
                    row.adapter_concept_id,
                    row.concept_id,
                    row.fact_id,
                )
                for row in results
            },
            key=lambda item: (
                item[0], item[1], item[2], item[3] or "", item[4], item[5]
            ),
        )
    )
    observations = tuple(
        _fact_observation(
            tuple(
                row
                for row in results
                if (
                    row.stage,
                    row.method,
                    row.split,
                    row.adapter_concept_id,
                    row.concept_id,
                    row.fact_id,
                )
                == key
            ),
            expected_count=(
                validation_template_count
                if key[2] == "validation"
                else test_template_count
            ),
        )
        for key in keys
    )
    return observations


def _fact_observation(
    rows: tuple[SemanticQueryResult, ...],
    *,
    expected_count: int,
) -> FactObservation:
    first = rows[0]
    if (
        len(rows) != expected_count
        or len({row.template_id for row in rows}) != len(rows)
    ):
        raise ValueError("each fact statistic requires every unique split paraphrase")
    router_rows = tuple(
        row
        for row in rows
        if row.selected_node_index is not None and row.oracle_node_index is not None
    )
    regret_rows = tuple(row for row in rows if row.routed_regret is not None)
    if router_rows and len(router_rows) != len(rows):
        raise ValueError("fact routing metadata must cover every paraphrase or none")
    if regret_rows and len(regret_rows) != len(rows):
        raise ValueError("fact routed regret must cover every paraphrase or none")
    return FactObservation(
        stage=first.stage,
        method=first.method,
        split=first.split,
        adapter_concept_id=first.adapter_concept_id,
        concept_id=first.concept_id,
        fact_id=first.fact_id,
        template_count=len(rows),
        accuracy=float(np.mean([row.answer_correct for row in rows])),
        margin=float(np.mean([row.correct_answer_margin for row in rows])),
        router_accuracy=(
            None
            if not router_rows
            else float(
                np.mean(
                    [
                        row.selected_node_index == row.oracle_node_index
                        for row in router_rows
                    ]
                )
            )
        ),
        routed_regret=(
            None
            if not regret_rows
            else float(np.mean([row.routed_regret for row in regret_rows]))
        ),
    )


def bootstrap_fact_metric(
    observations: tuple[FactObservation, ...],
    metric: FactMetricName,
    *,
    replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
    identity: str = "primary",
) -> BootstrapEstimate:
    """Resample facts within worlds, then give every world exactly equal weight."""
    return _bootstrap_values(
        observations,
        metric,
        tuple(observation.metric(metric) for observation in observations),
        replicates=replicates,
        identity=identity,
    )


def paired_fact_effect(
    baseline: tuple[FactObservation, ...],
    treatment: tuple[FactObservation, ...],
    metric: FactMetricName,
    *,
    replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
    identity: str,
) -> BootstrapEstimate:
    """Estimate treatment-minus-baseline on aligned facts before resampling."""
    baseline_keys = tuple((item.concept_id, item.fact_id) for item in baseline)
    treatment_keys = tuple((item.concept_id, item.fact_id) for item in treatment)
    if len(set(baseline_keys)) != len(baseline_keys) or len(
        set(treatment_keys)
    ) != len(treatment_keys):
        raise ValueError("paired semantic effects require one observation per fact")
    baseline_by_fact = {
        (item.concept_id, item.fact_id): item for item in baseline
    }
    treatment_by_fact = {
        (item.concept_id, item.fact_id): item for item in treatment
    }
    if set(baseline_by_fact) != set(treatment_by_fact):
        raise ValueError("paired semantic effects require identical facts")
    ordered_keys = tuple(sorted(baseline_by_fact))
    aligned = tuple(treatment_by_fact[key] for key in ordered_keys)
    differences = tuple(
        treatment_by_fact[key].metric(metric) - baseline_by_fact[key].metric(metric)
        for key in ordered_keys
    )
    return _bootstrap_values(
        aligned,
        f"{identity}:{metric}",
        differences,
        replicates=replicates,
        identity=identity,
    )


def acquisition_effect(
    base: tuple[FactObservation, ...],
    acquired: tuple[FactObservation, ...],
    metric: FactMetricName,
    *,
    replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
) -> BootstrapEstimate:
    """Report base-to-adapter acquisition as a paired fact effect."""
    return paired_fact_effect(
        base,
        acquired,
        metric,
        replicates=replicates,
        identity="base-to-adapter-acquisition",
    )


def retention_effect(
    acquisition: tuple[FactObservation, ...],
    final: tuple[FactObservation, ...],
    metric: FactMetricName,
    *,
    replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
) -> BootstrapEstimate:
    """Report final-minus-acquisition retention on the same facts."""
    return paired_fact_effect(
        acquisition,
        final,
        metric,
        replicates=replicates,
        identity="acquisition-to-final-retention",
    )


def specificity_effect(
    observations: tuple[FactObservation, ...],
    metric: FactMetricName,
    *,
    replicates: int = CANONICAL_BOOTSTRAP_REPLICATES,
    identity: str = "node-specificity",
) -> BootstrapEstimate:
    """Bootstrap each forced adapter's own-world minus equal-weight other worlds."""
    if not observations or any(
        observation.adapter_concept_id is None for observation in observations
    ):
        raise ValueError("specificity requires forced-adapter fact observations")
    adapter_ids = tuple(
        sorted({observation.adapter_concept_id for observation in observations})
    )
    query_worlds = tuple(sorted({observation.concept_id for observation in observations}))
    if len(adapter_ids) < 2 or set(adapter_ids) != set(query_worlds):
        raise ValueError("specificity requires every adapter on every query world")
    values_by_cell = {
        (adapter_id, query_world): np.asarray(
            [
                observation.metric(metric)
                for observation in observations
                if observation.adapter_concept_id == adapter_id
                and observation.concept_id == query_world
            ],
            dtype=np.float64,
        )
        for adapter_id in adapter_ids
        for query_world in query_worlds
    }
    if any(values.size == 0 for values in values_by_cell.values()):
        raise ValueError("specificity matrix is missing an adapter/world fact cell")
    point = float(
        np.mean(
            [
                np.mean(values_by_cell[(adapter_id, adapter_id)])
                - np.mean(
                    [
                        np.mean(values_by_cell[(adapter_id, query_world)])
                        for query_world in query_worlds
                        if query_world != adapter_id
                    ]
                )
                for adapter_id in adapter_ids
            ]
        )
    )
    seed = int.from_bytes(
        sha256(
            f"{BENCHMARK_ID}\0{identity}\0{metric}\0{replicates}".encode(
                "utf-8"
            )
        ).digest()[:16],
        "big",
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    replicate_values = np.zeros(replicates, dtype=np.float64)
    for adapter_id in adapter_ids:
        cell_replicates = {
            query_world: np.mean(
                values[
                    generator.integers(
                        0,
                        len(values),
                        size=(replicates, len(values)),
                    )
                ],
                axis=1,
            )
            for query_world in query_worlds
            for values in (values_by_cell[(adapter_id, query_world)],)
        }
        other_mean = np.mean(
            [
                values
                for query_world, values in cell_replicates.items()
                if query_world != adapter_id
            ],
            axis=0,
        )
        replicate_values += (
            cell_replicates[adapter_id] - other_mean
        ) / len(adapter_ids)
    lower, upper = np.quantile(
        replicate_values,
        (0.025, 0.975),
        method="linear",
    )
    return BootstrapEstimate(
        metric=f"{identity}:{metric}",
        point=point,
        lower=float(lower),
        upper=float(upper),
        replicate_count=replicates,
        fact_count=len(observations),
        world_count=len(query_worlds),
    )


def _bootstrap_values(
    observations: tuple[FactObservation, ...],
    metric_label: str,
    values: tuple[float, ...],
    *,
    replicates: int,
    identity: str,
) -> BootstrapEstimate:
    if (
        not observations
        or len(observations) != len(values)
        or type(replicates) is not int
        or replicates <= 0
        or any(not isfinite(value) for value in values)
    ):
        raise ValueError("bootstrap inputs must contain aligned finite fact values")
    worlds = tuple(sorted({item.concept_id for item in observations}))
    indexes_by_world = tuple(
        np.asarray(
            [index for index, item in enumerate(observations) if item.concept_id == world],
            dtype=np.int64,
        )
        for world in worlds
    )
    if any(indexes.size == 0 for indexes in indexes_by_world):
        raise AssertionError("bootstrap world indexing lost observations")
    value_array = np.asarray(values, dtype=np.float64)
    point = float(
        np.mean([np.mean(value_array[indexes]) for indexes in indexes_by_world])
    )
    seed = int.from_bytes(
        sha256(
            f"{BENCHMARK_ID}\0{identity}\0{metric_label}\0{replicates}".encode("utf-8")
        ).digest()[:16],
        "big",
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    replicate_values = np.zeros(replicates, dtype=np.float64)
    for indexes in indexes_by_world:
        sampled_positions = generator.integers(
            0,
            len(indexes),
            size=(replicates, len(indexes)),
        )
        replicate_values += np.mean(
            value_array[indexes[sampled_positions]],
            axis=1,
        ) / len(worlds)
    lower, upper = np.quantile(
        replicate_values,
        (0.025, 0.975),
        method="linear",
    )
    return BootstrapEstimate(
        metric=metric_label,
        point=point,
        lower=float(lower),
        upper=float(upper),
        replicate_count=replicates,
        fact_count=len(observations),
        world_count=len(worlds),
    )


def generation_prompts(
    concepts: tuple[ConceptDefinition, ...],
) -> tuple[tuple[str, str], ...]:
    """Return the registered secondary free-generation prompts."""
    return tuple(
        (
            concept.concept_id,
            f"Write three short facts about {concept.surface_forms[1] if len(concept.surface_forms) > 1 else concept.surface_forms[0]}.",
        )
        for concept in concepts
    )


def inspect_generation(
    catalog: SemanticQueryCatalog,
    outputs: tuple[tuple[str, str, str], ...],
    *,
    concept_ids: tuple[str, ...] | None = None,
) -> tuple[GenerationInspection, ...]:
    """Measure exact registered-trigger recall while retaining every raw output."""
    active_concepts = catalog.concept_ids if concept_ids is None else concept_ids
    if (
        type(active_concepts) is not tuple
        or not active_concepts
        or active_concepts != catalog.concept_ids[: len(active_concepts)]
    ):
        raise ValueError("generation concepts must be an ordered catalog prefix")
    fact_order = {fact.fact_id: index for index, fact in enumerate(catalog.facts)}
    facts_by_concept = {
        concept_id: tuple(
            fact for fact in catalog.facts if fact.concept_id == concept_id
        )
        for concept_id in active_concepts
    }
    if (
        len(outputs) != len(facts_by_concept)
        or tuple(concept_id for concept_id, _, _ in outputs) != active_concepts
    ):
        raise ValueError("generation outputs must cover the ordered active concepts")
    inspections = tuple(
        GenerationInspection(
            concept_id=concept_id,
            prompt=prompt,
            output=output,
            recalled_fact_ids=tuple(
                sorted(
                    (
                        fact.fact_id
                        for fact in facts_by_concept[concept_id]
                        if any(_contains_trigger(output, trigger) for trigger in fact.trigger_forms)
                    ),
                    key=fact_order.__getitem__,
                )
            ),
            recall=sum(
                any(_contains_trigger(output, trigger) for trigger in fact.trigger_forms)
                for fact in facts_by_concept[concept_id]
            )
            / len(facts_by_concept[concept_id]),
        )
        for concept_id, prompt, output in outputs
    )
    return inspections


def _contains_trigger(output: str, trigger: str) -> bool:
    normalized = normalize_story_identity(output)
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", normalized) is not None


__all__ = [
    "BootstrapEstimate",
    "CANONICAL_BOOTSTRAP_REPLICATES",
    "FactObservation",
    "GenerationInspection",
    "acquisition_effect",
    "average_direction_paraphrases",
    "average_paraphrases",
    "bootstrap_fact_metric",
    "generation_prompts",
    "inspect_generation",
    "paired_fact_effect",
    "retention_effect",
    "specificity_effect",
]
