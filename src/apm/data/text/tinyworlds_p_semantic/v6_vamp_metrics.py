"""Post-training measurements for the fixed semantic-v6 VAMP experiment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_benchmark_metrics import (
    LanguageMemoryAccounting,
    account_language_memory,
)
from apm.continual.language_benchmark_run import LanguageBenchmarkSettings
from apm.continual.language_benchmarks import (
    AddressingOperationCounts,
    AddressingTiming,
    ROUTER_BASELINE_NAMES,
    time_synchronized_addressing,
)
from apm.continual.language_evaluation import LanguageEvaluationSuite
from apm.continual.language_evaluation_run import (
    LanguageConditionMeasurement,
    LanguageEvaluationBenchmark,
    evaluate_language_benchmark,
)
from apm.continual.language_routing import route_language_prefix
from apm.continual.language_tasks import RouterBatch, TaskId
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import GptNeoParams


@dataclass(frozen=True, slots=True)
class V6ForgettingMeasurement:
    """Final-stage NLL increase relative to acquisition and best prior NLL."""

    method: str
    task_id: str
    acquisition_stage: int
    acquisition_nll: float
    best_nll: float
    final_nll: float
    forgetting_from_acquisition: float
    forgetting_from_best: float

    def __post_init__(self) -> None:
        values = (self.acquisition_nll, self.best_nll, self.final_nll)
        if not self.method or not self.task_id or any(
            not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("VAMP forgetting measurement is malformed")
        if not math.isclose(
            self.forgetting_from_acquisition,
            self.final_nll - self.acquisition_nll,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.forgetting_from_best,
            self.final_nll - self.best_nll,
            abs_tol=1e-12,
        ):
            raise ValueError("VAMP forgetting differences are inconsistent")


@dataclass(frozen=True, slots=True)
class V6TransferMeasurement:
    """Parent choice, fixed-budget learning, and oracle deficit for one world."""

    stage: int
    task_id: str
    selected_parent_id: str
    root_initial_nll: float
    selected_parent_initial_nll: float
    parent_advantage: float
    first_step_improvement: float
    fixed_budget_improvement: float
    update_budget: int
    final_vamp_nll: float
    final_independent_nll: float
    final_deficit_vs_independent: float


@dataclass(frozen=True, slots=True)
class V6RouterTiming:
    """One synchronized final-stage task-free routing measurement."""

    method: str
    timing: AddressingTiming


@dataclass(frozen=True, slots=True)
class V6VampPosthocMetrics:
    """All frozen-suite VAMP metrics that do not require paired controls."""

    benchmark: LanguageEvaluationBenchmark
    forgetting: tuple[V6ForgettingMeasurement, ...]
    transfer: tuple[V6TransferMeasurement, ...]
    memory: LanguageMemoryAccounting
    routing_timing: tuple[V6RouterTiming, ...]


def measure_v6_vamp_posthoc(
    adaptation: LanguageAdaptationArtifact,
    suite: LanguageEvaluationSuite,
    base_params: GptNeoParams,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
    *,
    measurement_sink: Callable[[LanguageConditionMeasurement], None] | None = None,
    condition_progress: Callable[[int, TaskId, str, int, int], None] | None = None,
    phase_sink: Callable[[str], None] | None = None,
) -> V6VampPosthocMetrics:
    """Evaluate the nine methods, forgetting, transfer, memory, and router cost."""
    _require_inputs(adaptation, preset)
    settings = _benchmark_settings(preset)
    benchmark = evaluate_language_benchmark(
        adaptation,
        suite,
        base_params,
        adaptation.model_config,
        adaptation.lora_config,
        settings,
        measurement_sink=measurement_sink,
        condition_progress=condition_progress,
    )
    forgetting = _forgetting_measurements(benchmark, preset)
    transfer = _transfer_measurements(adaptation, benchmark, preset)
    packed_memory = pack_lora_memory(
        adaptation.vamp_graph,
        adaptation.model_config,
        adaptation.lora_config,
        adaptation.max_nodes,
        adaptation.max_edges,
    )
    primary_rows = tuple(
        row
        for row in benchmark.measurements
        if row.stage == len(preset.task_order)
        and row.prefix_tokens == preset.primary_prefix_length
        and row.cue_regime == "all"
    )
    frozen_mean = _method_mean(primary_rows, "frozen_base")
    vamp_mean = _method_mean(primary_rows, "vamp_oracle")
    memory = account_language_memory(
        base_params,
        adaptation.vamp_graph,
        adaptation.address_book,
        packed_memory,
        adaptation.lora_config,
        nll_improvement=frozen_mean - vamp_mean,
    )
    if phase_sink is not None:
        phase_sink("Timing the five final-stage task-free routers.")
    timing = _time_routers(
        adaptation,
        suite,
        base_params,
        packed_memory,
        preset,
    )
    return V6VampPosthocMetrics(benchmark, forgetting, transfer, memory, timing)


def _forgetting_measurements(
    benchmark: LanguageEvaluationBenchmark,
    preset: V6VampExperimentPreset,
) -> tuple[V6ForgettingMeasurement, ...]:
    rows = tuple(
        row
        for row in benchmark.measurements
        if row.prefix_tokens == preset.primary_prefix_length
        and row.cue_regime == "all"
    )
    methods = tuple(dict.fromkeys(row.method for row in rows))
    return tuple(
        _forgetting_for(method, task, stage, rows, len(preset.task_order))
        for stage, task in enumerate(preset.task_order, start=1)
        for method in methods
    )


def _forgetting_for(
    method: str,
    task: str,
    acquisition_stage: int,
    rows: tuple[LanguageConditionMeasurement, ...],
    final_stage: int,
) -> V6ForgettingMeasurement:
    relevant = tuple(
        row
        for row in rows
        if row.method == method
        and str(row.task_id) == task
        and row.stage >= acquisition_stage
    )
    acquisition = next(row.suffix_nll for row in relevant if row.stage == acquisition_stage)
    final = next(row.suffix_nll for row in relevant if row.stage == final_stage)
    best = min(row.suffix_nll for row in relevant)
    return V6ForgettingMeasurement(
        method=method,
        task_id=task,
        acquisition_stage=acquisition_stage,
        acquisition_nll=acquisition,
        best_nll=best,
        final_nll=final,
        forgetting_from_acquisition=final - acquisition,
        forgetting_from_best=final - best,
    )


def _transfer_measurements(
    adaptation: LanguageAdaptationArtifact,
    benchmark: LanguageEvaluationBenchmark,
    preset: V6VampExperimentPreset,
) -> tuple[V6TransferMeasurement, ...]:
    primary = tuple(
        row
        for row in benchmark.measurements
        if row.prefix_tokens == preset.primary_prefix_length
        and row.cue_regime == "all"
    )
    return tuple(
        V6TransferMeasurement(
            stage=record.stage_index,
            task_id=str(record.task_id),
            selected_parent_id=str(record.parent_node_id),
            root_initial_nll=record.parent_mean_node_nll[0],
            selected_parent_initial_nll=record.parent_mean_node_nll[
                record.parent_node_index
            ],
            parent_advantage=(
                record.parent_mean_node_nll[0]
                - record.parent_mean_node_nll[record.parent_node_index]
            ),
            first_step_improvement=(
                0.0
                if len(record.training_trace) == 1
                else record.training_trace[0] - record.training_trace[1]
            ),
            fixed_budget_improvement=(
                record.training_trace[0] - record.training_trace[-1]
            ),
            update_budget=len(record.training_trace),
            final_vamp_nll=_one_nll(
                primary,
                record.stage_index,
                str(record.task_id),
                "vamp_oracle",
            ),
            final_independent_nll=_one_nll(
                primary,
                record.stage_index,
                str(record.task_id),
                "independent_root_lora",
            ),
            final_deficit_vs_independent=(
                _one_nll(
                    primary,
                    record.stage_index,
                    str(record.task_id),
                    "vamp_oracle",
                )
                - _one_nll(
                    primary,
                    record.stage_index,
                    str(record.task_id),
                    "independent_root_lora",
                )
            ),
        )
        for record in adaptation.vamp_stages
    )


def _time_routers(
    adaptation: LanguageAdaptationArtifact,
    suite: LanguageEvaluationSuite,
    base_params: GptNeoParams,
    packed_memory,
    preset: V6VampExperimentPreset,
) -> tuple[V6RouterTiming, ...]:
    examples = tuple(
        item.example
        for item in suite.examples
        if item.condition_id == suite.primary_condition_id
    )
    prefix_batch = RouterBatch(
        input_ids=np.concatenate(tuple(item.router_batch.input_ids for item in examples)),
        attention_mask=np.concatenate(
            tuple(item.router_batch.attention_mask for item in examples)
        ),
        target_ids=np.concatenate(tuple(item.router_batch.target_ids for item in examples)),
        loss_mask=np.concatenate(tuple(item.router_batch.loss_mask for item in examples)),
    )
    settings = _benchmark_settings(preset)
    return tuple(
        V6RouterTiming(
            method=method,
            timing=time_synchronized_addressing(
                lambda method=method: route_language_prefix(
                    method,
                    base_params,
                    adaptation.model_config,
                    packed_memory,
                    adaptation.lora_config,
                    adaptation.address_book,
                    prefix_batch,
                    random_seed=settings.random_router_seed,
                    hopfield_config=settings.hopfield,
                    ebt_config=settings.ebt,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                ),
                _operation_counts(method, prefix_batch, packed_memory, preset.ebt_steps),
                batch_size=prefix_batch.input_ids.shape[0],
                warm_repetitions=preset.timing_warm_repetitions,
            ),
        )
        for method in ROUTER_BASELINE_NAMES
    )


def _operation_counts(method, prefix_batch, packed_memory, ebt_steps: int):
    prefix_tokens = int(np.sum(prefix_batch.loss_mask))
    valid_nodes = int(np.sum(packed_memory.valid_node_mask))
    valid_edges = int(np.sum(packed_memory.valid_edge_mask))
    is_ebt = method in ("vamp_ebt_uniform", "vamp_ebt_hopfield")
    base_forwards = (
        valid_nodes
        if method == "vamp_exhaustive"
        else ebt_steps + 3 + int(method == "vamp_ebt_hopfield")
        if is_ebt
        else 1
        if method == "vamp_hopfield"
        else 0
    )
    return AddressingOperationCounts(
        prefix_tokens=prefix_tokens,
        candidates_available=valid_nodes,
        candidates_scored=(
            0 if method == "deterministic_random_node" else valid_nodes
        ),
        full_model_forward_equivalent_tokens=prefix_tokens * base_forwards,
        base_forwards=base_forwards,
        edge_evaluations=prefix_tokens * valid_edges * base_forwards,
        hopfield_dot_products=(
            prefix_batch.input_ids.shape[0] * valid_nodes
            if method in ("vamp_hopfield", "vamp_ebt_hopfield")
            else 0
        ),
        ebt_steps=ebt_steps if is_ebt else 0,
        ebt_mask_size=valid_nodes if is_ebt else 0,
        selected_execution_cost=1,
    )


def _benchmark_settings(preset: V6VampExperimentPreset) -> LanguageBenchmarkSettings:
    return LanguageBenchmarkSettings(
        seed=preset.seed,
        random_router_seed=preset.random_router_seed,
        hopfield=preset.hopfield_config,
        ebt=preset.ebt_config,
        evaluation_microbatch_size=preset.evaluation_microbatch_size,
        timing_warm_repetitions=preset.timing_warm_repetitions,
        sample_new_tokens=preset.sample_new_tokens,
        peak_device_memory_target_bytes=preset.allocator_peak_limit_bytes,
    )


def _one_nll(
    rows: tuple[LanguageConditionMeasurement, ...],
    stage: int,
    task: str,
    method: str,
) -> float:
    matches = tuple(
        row.suffix_nll
        for row in rows
        if row.stage == stage and str(row.task_id) == task and row.method == method
    )
    if len(matches) != 1:
        raise ValueError("semantic-v6 VAMP measurement identity is not unique")
    return matches[0]


def _method_mean(
    rows: tuple[LanguageConditionMeasurement, ...],
    method: str,
) -> float:
    values = tuple(row.suffix_nll for row in rows if row.method == method)
    if len(values) != len(V6_VAMP_EXPERIMENT_PRESET.task_order):
        raise ValueError("semantic-v6 VAMP final measurement set is incomplete")
    return math.fsum(values) / len(values)


def _require_inputs(
    adaptation: LanguageAdaptationArtifact,
    preset: V6VampExperimentPreset,
) -> None:
    if not isinstance(adaptation, LanguageAdaptationArtifact):
        raise TypeError("semantic-v6 VAMP metrics require an adaptation artifact")
    if type(preset) is not V6VampExperimentPreset:
        raise TypeError("semantic-v6 VAMP metrics require their strict preset")
    if tuple(str(task) for task in adaptation.task_order) != preset.task_order:
        raise ValueError("semantic-v6 VAMP metric task order changed")


__all__ = [
    "V6ForgettingMeasurement",
    "V6RouterTiming",
    "V6TransferMeasurement",
    "V6VampPosthocMetrics",
    "measure_v6_vamp_posthoc",
]
