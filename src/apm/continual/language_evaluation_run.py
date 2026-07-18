"""Evaluation-only execution over already-trained language adaptations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import struct
from typing import Literal

import jax
import numpy as np

from apm.continual.language_baseline_training import (
    LanguageAdaptationBaselines,
    pack_root_adapter,
)
from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_benchmark_run import LanguageBenchmarkSettings
from apm.continual.language_benchmarks import (
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.continual.language_evaluation import (
    LanguageEvaluationSuite,
    LanguageSuiteExample,
)
from apm.continual.language_routing import (
    competence_nll_by_node,
    evaluate_language_router,
)
from apm.continual.language_tasks import AddressBook, CompetenceBatch, TaskId
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.memory.graph import MemoryGraph, NodeId


EvaluationMethodCategory = Literal["stored", "router"]


@dataclass(frozen=True)
class LanguageConditionMeasurement:
    """One nonempty task/condition/cue aggregate from a frozen adaptation set."""

    stage: int
    method: str
    category: EvaluationMethodCategory
    task_id: TaskId
    condition_id: str
    prefix_tokens: int
    suffix_tokens: int
    cue_regime: str
    example_count: int
    pair_count: int
    source_story_count: int
    suffix_nll: float
    perplexity: float
    routing_accuracy: float | None
    routing_regret: float | None
    entropy: float | None

    def __post_init__(self) -> None:
        if type(self.stage) is not int or self.stage <= 0:
            raise ValueError("language evaluation stage must be positive")
        expected_category = "stored" if self.method in STORED_BASELINE_NAMES else "router"
        if self.category != expected_category:
            raise ValueError("language evaluation method category is inconsistent")
        if not self.task_id or not self.condition_id:
            raise ValueError("language condition measurement identity must not be empty")
        if self.cue_regime not in (
            "cue_sufficient",
            "cue_present",
            "cue_hidden_or_ambiguous",
            "all",
        ):
            raise ValueError("language condition measurement has an unknown cue regime")
        dimensions = (
            self.prefix_tokens,
            self.suffix_tokens,
            self.example_count,
            self.pair_count,
            self.source_story_count,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("language condition dimensions and counts must be positive")
        if not math.isfinite(self.suffix_nll) or self.suffix_nll < 0.0:
            raise ValueError("language condition suffix NLL must be finite and nonnegative")
        if not math.isfinite(self.perplexity) or self.perplexity <= 0.0:
            raise ValueError("language condition perplexity must be finite and positive")
        optional_values = (self.routing_accuracy, self.routing_regret, self.entropy)
        if self.category == "stored" and any(value is not None for value in optional_values):
            raise ValueError("stored measurements cannot contain routing-only metrics")
        if self.category == "router" and any(value is None for value in optional_values):
            raise ValueError("router measurements require routing metrics")
        if self.routing_accuracy is not None and not 0.0 <= self.routing_accuracy <= 1.0:
            raise ValueError("routing accuracy must lie in [0, 1]")
        if any(value is not None and not math.isfinite(value) for value in optional_values):
            raise ValueError("routing metrics must be finite when present")


@dataclass(frozen=True)
class LanguageEvaluationBenchmark:
    """Completed evaluation projection with immutable adapter checksum evidence."""

    suite: LanguageEvaluationSuite
    measurements: tuple[LanguageConditionMeasurement, ...]
    adaptation_checksum_before: str
    adaptation_checksum_after: str

    def __post_init__(self) -> None:
        if not isinstance(self.suite, LanguageEvaluationSuite):
            raise TypeError("suite must be a LanguageEvaluationSuite")
        if not self.measurements or any(
            not isinstance(value, LanguageConditionMeasurement)
            for value in self.measurements
        ):
            raise ValueError("language evaluation benchmark requires measurements")
        if self.adaptation_checksum_before != self.adaptation_checksum_after:
            raise ValueError("language evaluation changed a persisted adaptation tensor")


def evaluate_language_benchmark(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
    suite: LanguageEvaluationSuite,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    settings: LanguageBenchmarkSettings = LanguageBenchmarkSettings(),
) -> LanguageEvaluationBenchmark:
    """Evaluate trained adapters on an explicit suite without invoking training."""
    if not isinstance(
        adaptations,
        (LanguageAdaptationBaselines, LanguageAdaptationArtifact),
    ):
        raise TypeError("adaptations must be trained baselines or a loaded artifact")
    if not isinstance(suite, LanguageEvaluationSuite):
        raise TypeError("suite must be a LanguageEvaluationSuite")
    task_order = _task_order(adaptations)
    suite_tasks = tuple(dict.fromkeys(example.task_id for example in suite.examples))
    if any(task_id not in task_order for task_id in suite_tasks):
        raise ValueError("evaluation suite contains a task absent from adaptations")
    if tuple(task_id for task_id in task_order if task_id in suite_tasks) != suite_tasks:
        raise ValueError("evaluation suite task order must follow adaptation order")

    checksum_before = _adaptation_checksum(adaptations)
    measurements: list[LanguageConditionMeasurement] = []
    independent_by_task = _independent_adapters(adaptations)
    condition_by_id = {
        condition.condition_id: condition for condition in suite.conditions
    }
    final_graph = _vamp_graph(adaptations)

    for stage in range(1, len(task_order) + 1):
        graph = MemoryGraph[LoraEdge](nodes=final_graph.nodes[: stage + 1])
        address_book = _stage_address_book(_address_book(adaptations), stage + 1)
        packed_memory = pack_lora_memory(
            graph,
            model_config,
            lora_config,
            _max_nodes(adaptations),
            _max_edges(adaptations),
        )
        _, sequential_memory = pack_root_adapter(
            _sequential_adapter(adaptations, stage),
            model_config,
            lora_config,
        )
        for task_id in task_order[:stage]:
            _, independent_memory = pack_root_adapter(
                independent_by_task[task_id],
                model_config,
                lora_config,
            )
            oracle_index = next(
                index
                for index, node in enumerate(graph.nodes)
                if node.node_id == NodeId(str(task_id))
            )
            for condition_id, condition in condition_by_id.items():
                suite_examples = suite.examples_for(task_id, condition_id)
                if not suite_examples:
                    continue
                examples = tuple(value.example for value in suite_examples)
                competence_batch = _stack_competence(suite_examples)
                token_weights = np.sum(competence_batch.loss_mask, axis=1)
                vamp_nll = competence_nll_by_node(
                    base_params,
                    model_config,
                    packed_memory,
                    lora_config,
                    competence_batch,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                )
                sequential_nll = competence_nll_by_node(
                    base_params,
                    model_config,
                    sequential_memory,
                    lora_config,
                    competence_batch,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                )[:, 1]
                independent_nll = competence_nll_by_node(
                    base_params,
                    model_config,
                    independent_memory,
                    lora_config,
                    competence_batch,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                )[:, 1]
                stored_values = {
                    "frozen_base": vamp_nll[:, 0],
                    "sequential_single_lora": sequential_nll,
                    "independent_root_lora": independent_nll,
                    "vamp_oracle": vamp_nll[:, oracle_index],
                }
                route_values = tuple(
                    evaluate_language_router(
                        router,
                        base_params,
                        model_config,
                        graph,
                        packed_memory,
                        lora_config,
                        address_book,
                        examples,
                        random_seed=settings.random_router_seed,
                        hopfield_config=settings.hopfield,
                        ebt_config=settings.ebt,
                        evaluation_microbatch_size=settings.evaluation_microbatch_size,
                        suffix_nll_by_node=vamp_nll,
                    )
                    for router in ROUTER_BASELINE_NAMES
                )
                strata = _stratum_indices(suite_examples)
                for cue_regime, indices in strata:
                    selected_examples = tuple(suite_examples[index] for index in indices)
                    selected_weights = token_weights[np.asarray(indices)]
                    measurements.extend(
                        LanguageConditionMeasurement(
                            stage=stage,
                            method=method,
                            category="stored",
                            task_id=task_id,
                            condition_id=condition_id,
                            prefix_tokens=condition.prefix_tokens,
                            suffix_tokens=condition.suffix_tokens,
                            cue_regime=cue_regime,
                            example_count=len(indices),
                            pair_count=len({value.pair_id for value in selected_examples}),
                            source_story_count=len(
                                {
                                    value.provenance.source_document_id
                                    for value in selected_examples
                                }
                            ),
                            suffix_nll=(mean_nll := _weighted_mean(
                                values[np.asarray(indices)], selected_weights
                            )),
                            perplexity=math.exp(min(mean_nll, 80.0)),
                            routing_accuracy=None,
                            routing_regret=None,
                            entropy=None,
                        )
                        for method, values in stored_values.items()
                    )
                    measurements.extend(
                        _router_measurement(
                            stage,
                            task_id,
                            condition_id,
                            condition.prefix_tokens,
                            condition.suffix_tokens,
                            cue_regime,
                            selected_examples,
                            indices,
                            selected_weights,
                            route_value,
                        )
                        for route_value in route_values
                    )

    checksum_after = _adaptation_checksum(adaptations)
    return LanguageEvaluationBenchmark(
        suite=suite,
        measurements=tuple(measurements),
        adaptation_checksum_before=checksum_before,
        adaptation_checksum_after=checksum_after,
    )


def _router_measurement(
    stage: int,
    task_id: TaskId,
    condition_id: str,
    prefix_tokens: int,
    suffix_tokens: int,
    cue_regime: str,
    suite_examples: tuple[LanguageSuiteExample, ...],
    indices: tuple[int, ...],
    token_weights: np.ndarray,
    route_value,
) -> LanguageConditionMeasurement:
    evaluations = tuple(route_value.examples[index] for index in indices)
    suffix_nll = _weighted_mean(
        np.asarray(tuple(value.selected_suffix_nll for value in evaluations)),
        token_weights,
    )
    return LanguageConditionMeasurement(
        stage=stage,
        method=route_value.router,
        category="router",
        task_id=task_id,
        condition_id=condition_id,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        cue_regime=cue_regime,
        example_count=len(indices),
        pair_count=len({value.pair_id for value in suite_examples}),
        source_story_count=len(
            {value.provenance.source_document_id for value in suite_examples}
        ),
        suffix_nll=suffix_nll,
        perplexity=math.exp(min(suffix_nll, 80.0)),
        routing_accuracy=float(
            np.mean(tuple(value.task_oracle_correct for value in evaluations))
        ),
        routing_regret=_weighted_mean(
            np.asarray(tuple(value.task_oracle_regret for value in evaluations)),
            token_weights,
        ),
        entropy=float(
            np.mean(np.asarray(route_value.decision.entropy)[np.asarray(indices)])
        ),
    )


def _stage_address_book(address_book: AddressBook, valid_count: int) -> AddressBook:
    valid_mask = np.arange(address_book.max_nodes) < valid_count
    return AddressBook(
        node_ids=tuple(
            node_id if index < valid_count else None
            for index, node_id in enumerate(address_book.node_ids)
        ),
        keys=np.where(valid_mask[:, None], address_book.keys, 0.0),
        valid_node_mask=valid_mask,
    )


def _stack_competence(
    examples: tuple[LanguageSuiteExample, ...],
) -> CompetenceBatch:
    batches = tuple(value.example.competence_batch for value in examples)
    return CompetenceBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches)),
        attention_mask=np.concatenate(tuple(batch.attention_mask for batch in batches)),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches)),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches)),
    )


def _stratum_indices(
    examples: tuple[LanguageSuiteExample, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    specific = tuple(
        (
            cue_regime,
            tuple(
                index
                for index, example in enumerate(examples)
                if example.cue_regime == cue_regime
            ),
        )
        for cue_regime in (
            "cue_sufficient",
            "cue_present",
            "cue_hidden_or_ambiguous",
        )
    )
    return tuple(value for value in specific if value[1]) + (
        ("all", tuple(range(len(examples)))),
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _adaptation_checksum(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> str:
    if isinstance(adaptations, LanguageAdaptationArtifact):
        return adaptations.tensor_checksum
    digest = sha256()
    values = (
        tuple(stage.adapter for stage in adaptations.sequential_single_lora.stages)
        + tuple(adapter.adapter for adapter in adaptations.independent_root_lora.adapters)
        + tuple(
            node.incoming_edge
            for node in adaptations.vamp.graph.nodes
            if node.incoming_edge is not None
        )
    )
    for value in values:
        for leaf in jax.tree_util.tree_leaves(value):
            array = np.ascontiguousarray(np.asarray(leaf))
            descriptor = f"{array.dtype.str}:{array.shape}".encode("ascii")
            digest.update(struct.pack("<Q", len(descriptor)))
            digest.update(descriptor)
            digest.update(array.tobytes())
    for array in (
        adaptations.vamp.address_book.keys,
        adaptations.vamp.address_book.valid_node_mask,
        adaptations.vamp.rng_key,
    ):
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _task_order(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> tuple[TaskId, ...]:
    return (
        adaptations.task_order
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else tuple(
            adapter.task_id
            for adapter in adaptations.independent_root_lora.adapters
        )
    )


def _independent_adapters(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> dict[TaskId, LoraEdge]:
    records = (
        adaptations.independent_adapters
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.independent_root_lora.adapters
    )
    return {record.task_id: record.adapter for record in records}


def _sequential_adapter(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
    stage: int,
) -> LoraEdge:
    stages = (
        adaptations.sequential_stages
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.sequential_single_lora.stages
    )
    return stages[stage - 1].adapter


def _vamp_graph(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> MemoryGraph[LoraEdge]:
    return (
        adaptations.vamp_graph
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.vamp.graph
    )


def _address_book(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> AddressBook:
    return (
        adaptations.address_book
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.vamp.address_book
    )


def _max_nodes(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> int:
    return (
        adaptations.max_nodes
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.vamp.max_nodes
    )


def _max_edges(
    adaptations: LanguageAdaptationBaselines | LanguageAdaptationArtifact,
) -> int:
    return (
        adaptations.max_edges
        if isinstance(adaptations, LanguageAdaptationArtifact)
        else adaptations.vamp.max_edges
    )


__all__ = [
    "LanguageConditionMeasurement",
    "LanguageEvaluationBenchmark",
    "evaluate_language_benchmark",
]
