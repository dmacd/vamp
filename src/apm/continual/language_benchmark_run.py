"""End-to-end training and measurement for language continual-learning baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import math

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_baseline_training import (
    LanguageAdaptationBaselines,
    pack_root_adapter,
    train_language_adaptation_baselines,
)
from apm.continual.language_benchmark_metrics import (
    LanguageMemoryAccounting,
    account_language_memory,
)
from apm.continual.language_benchmarks import (
    AddressingCoefficientTrace,
    AddressingOperationCounts,
    AddressingTiming,
    GeneratedLanguageSample,
    ROUTER_BASELINE_NAMES,
    RouterBaselineName,
    StoredBaselineName,
    summarize_negative_control,
    time_synchronized_addressing,
)
from apm.continual.language_routing import (
    competence_nll_by_node,
    evaluate_language_router,
    route_language_prefix,
    trace_ebt_language_prefix,
)
from apm.continual.language_tasks import (
    AddressBook,
    BaseCheckpointRef,
    CompetenceBatch,
    LanguageEvaluationExample,
    NodeId,
    RouterBatch,
    TaskId,
)
from apm.data.text.language_tasks import PreparedLanguageCurriculum
from apm.lm.config import GptNeoConfig
from apm.lm.evaluation import (
    evaluation_microbatch_slices,
    validate_evaluation_microbatch_size,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.generation import greedy_generate
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text import TextTokenizer
from apm.lm.training import LmTrainConfig, init_candidate_lora_train_state
from apm.memory.address_refinement import EbtConfig
from apm.memory.content_addressing import HopfieldConfig
from apm.memory.graph import MemoryGraph, memory_node_path


@dataclass(frozen=True)
class StoredCompetenceMeasurement:
    """One stored baseline's suffix competence at a stage/task/prefix."""

    stage: int
    baseline: StoredBaselineName
    task_id: TaskId
    prefix_length: int
    suffix_nll: float
    perplexity: float
    frozen_base_nll: float
    independent_root_nll: float
    improvement_over_frozen: float
    deficit_vs_independent: float
    stored_forgetting: float
    base_checksum_stable: bool
    committed_checksum_stable: bool


@dataclass(frozen=True)
class RoutingMeasurement:
    """Aggregated task-free route quality for one stage/task/prefix/router."""

    stage: int
    router: RouterBaselineName
    task_id: TaskId
    prefix_length: int
    example_count: int
    routing_accuracy: float
    top_k_recall: float
    exhaustive_agreement: float
    routed_suffix_nll: float
    task_oracle_suffix_nll: float
    best_node_suffix_nll: float
    task_oracle_regret: float
    best_node_regret: float
    entropy: float
    margin: float
    routing_forgetting: float
    negative_control_correct_count: int | None
    negative_control_chance_accuracy: float | None
    negative_control_ci95_lower: float | None
    negative_control_ci95_upper: float | None
    negative_control_chance_in_ci95: bool | None
    leakage_audit_required: bool | None


@dataclass(frozen=True)
class TransferMeasurement:
    """Parent advantage and fixed-budget learning behavior for one VAMP stage."""

    stage: int
    task_id: TaskId
    selected_parent_id: NodeId
    root_initial_nll: float
    selected_parent_initial_nll: float
    parent_advantage: float
    first_step_improvement: float
    fixed_budget_improvement: float
    update_budget: int
    tokens_per_update: int
    final_vamp_nll: float
    final_independent_nll: float
    final_deficit_vs_independent: float


@dataclass(frozen=True)
class StageMemoryMeasurement:
    """Persistent/runtime accounting after one committed VAMP stage."""

    stage: int
    accounting: LanguageMemoryAccounting


@dataclass(frozen=True)
class RouterTimingMeasurement:
    """Synchronized final-stage cost for one task-free router."""

    stage: int
    router: RouterBaselineName
    timing: AddressingTiming


@dataclass(frozen=True)
class PeakDeviceMemoryMeasurement:
    """Observed allocator peak and optional enforced device-memory target."""

    platform: str
    device_kind: str
    peak_bytes_in_use: int | None
    bytes_limit: int | None
    target_bytes: int | None
    within_target: bool | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.platform, str)
            or not self.platform
            or not isinstance(self.device_kind, str)
            or not self.device_kind
        ):
            raise ValueError("device platform and kind must not be empty")
        for field_name in (
            "peak_bytes_in_use",
            "bytes_limit",
            "target_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a nonnegative integer")
        expected = (
            None
            if self.target_bytes is None or self.peak_bytes_in_use is None
            else self.peak_bytes_in_use <= self.target_bytes
        )
        if self.within_target is not None and type(self.within_target) is not bool:
            raise TypeError("within_target must be a boolean when provided")
        if self.within_target != expected:
            raise ValueError("within_target must compare observed peak with target")


@dataclass(frozen=True, eq=False)
class LanguageBenchmarkResult:
    """Trained baselines plus every report-facing measurement family."""

    adaptations: LanguageAdaptationBaselines
    settings: LanguageBenchmarkSettings
    stored_competence: tuple[StoredCompetenceMeasurement, ...]
    routing: tuple[RoutingMeasurement, ...]
    transfer: tuple[TransferMeasurement, ...]
    memory: tuple[StageMemoryMeasurement, ...]
    addressing_cost: tuple[RouterTimingMeasurement, ...]
    addressing_traces: tuple[AddressingCoefficientTrace, ...]
    samples: tuple[GeneratedLanguageSample, ...]
    peak_device_memory: PeakDeviceMemoryMeasurement
    final_confusion: np.ndarray

    def __post_init__(self) -> None:
        if (
            len(self.addressing_traces) != 2
            or any(
                not isinstance(trace, AddressingCoefficientTrace)
                for trace in self.addressing_traces
            )
            or {trace.router for trace in self.addressing_traces}
            != {"vamp_ebt_uniform", "vamp_ebt_hopfield"}
        ):
            raise ValueError("benchmark results require both EBT addressing traces")
        confusion = np.array(self.final_confusion, dtype=np.int64, copy=True)
        if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
            raise ValueError("final_confusion must be square")
        if np.any(confusion < 0):
            raise ValueError("final_confusion must be nonnegative")
        confusion.flags.writeable = False
        object.__setattr__(self, "final_confusion", confusion)


@dataclass(frozen=True)
class LanguageBenchmarkSettings:
    """Fixed router and synchronized timing settings for one benchmark run."""

    seed: int = 0
    random_router_seed: int = 0
    hopfield: HopfieldConfig = HopfieldConfig()
    ebt: EbtConfig = EbtConfig()
    evaluation_microbatch_size: int | None = None
    timing_warm_repetitions: int = 5
    sample_new_tokens: int = 32
    peak_device_memory_target_bytes: int | None = None
    negative_control_curriculum: bool = False

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("benchmark seed must be nonnegative")
        if type(self.random_router_seed) is not int or self.random_router_seed < 0:
            raise ValueError("random router seed must be nonnegative")
        if type(self.timing_warm_repetitions) is not int or self.timing_warm_repetitions <= 0:
            raise ValueError("timing_warm_repetitions must be positive")
        if type(self.sample_new_tokens) is not int or self.sample_new_tokens <= 0:
            raise ValueError("sample_new_tokens must be positive")
        validate_evaluation_microbatch_size(self.evaluation_microbatch_size)
        if self.peak_device_memory_target_bytes is not None and (
            type(self.peak_device_memory_target_bytes) is not int
            or self.peak_device_memory_target_bytes <= 0
        ):
            raise ValueError("peak device-memory target must be a positive byte count")
        if type(self.negative_control_curriculum) is not bool:
            raise TypeError("negative_control_curriculum must be a boolean")


def run_language_benchmark(
    prepared: PreparedLanguageCurriculum,
    base_checkpoint: BaseCheckpointRef,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    tokenizer: TextTokenizer,
    settings: LanguageBenchmarkSettings = LanguageBenchmarkSettings(),
) -> LanguageBenchmarkResult:
    """Train, measure, and sample every language continual-learning baseline."""
    if not isinstance(prepared, PreparedLanguageCurriculum):
        raise TypeError("prepared must be a PreparedLanguageCurriculum")
    adaptations = train_language_adaptation_baselines(
        prepared.curriculum,
        prepared.root_validation_probes,
        base_checkpoint,
        base_params,
        model_config,
        lora_config,
        train_config,
        jax.random.PRNGKey(settings.seed),
        evaluation_microbatch_size=settings.evaluation_microbatch_size,
    )
    base_checksum = _tree_checksum(base_params)
    raw_stored: list[StoredCompetenceMeasurement] = []
    raw_routing: list[RoutingMeasurement] = []
    memory_rows: list[StageMemoryMeasurement] = []
    final_confusion = np.zeros(
        (prepared.curriculum.max_nodes, prepared.curriculum.max_nodes),
        dtype=np.int64,
    )
    independent_by_task = {
        adapter.task_id: adapter.adapter
        for adapter in adaptations.independent_root_lora.adapters
    }

    for stage in range(1, len(prepared.curriculum.tasks) + 1):
        graph, address_book, packed_memory = _vamp_stage_values(
            adaptations,
            model_config,
            lora_config,
            stage,
        )
        sequential_adapter = adaptations.sequential_single_lora.stages[
            stage - 1
        ].adapter
        _, sequential_memory = pack_root_adapter(
            sequential_adapter,
            model_config,
            lora_config,
        )
        for task in prepared.curriculum.tasks[:stage]:
            _, independent_memory = pack_root_adapter(
                independent_by_task[task.task_id],
                model_config,
                lora_config,
            )
            for prefix_length in prepared.build_config.prefix_lengths:
                sweep = _evaluation_sweep(prepared, task.task_id, prefix_length)
                competence_batch = _stack_competence_examples(sweep.test_examples)
                token_weights = np.sum(competence_batch.loss_mask, axis=1)
                frozen_nll = _weighted_mean(
                    _frozen_competence_nll(
                        base_params,
                        model_config,
                        competence_batch,
                        evaluation_microbatch_size=(
                            settings.evaluation_microbatch_size
                        ),
                    ),
                    token_weights,
                )
                sequential_nll = _weighted_mean(
                    competence_nll_by_node(
                        base_params,
                        model_config,
                        sequential_memory,
                        lora_config,
                        competence_batch,
                        evaluation_microbatch_size=(
                            settings.evaluation_microbatch_size
                        ),
                    )[:, 1],
                    token_weights,
                )
                independent_nll = _weighted_mean(
                    competence_nll_by_node(
                        base_params,
                        model_config,
                        independent_memory,
                        lora_config,
                        competence_batch,
                        evaluation_microbatch_size=(
                            settings.evaluation_microbatch_size
                        ),
                    )[:, 1],
                    token_weights,
                )
                vamp_nll_by_node = competence_nll_by_node(
                    base_params,
                    model_config,
                    packed_memory,
                    lora_config,
                    competence_batch,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                )
                oracle_index = next(
                    index
                    for index, node in enumerate(graph.nodes)
                    if node.node_id == NodeId(str(task.task_id))
                )
                vamp_nll = _weighted_mean(
                    vamp_nll_by_node[:, oracle_index],
                    token_weights,
                )
                checksum_stable = _path_checksum(graph, oracle_index) == _path_checksum(
                    adaptations.vamp.graph,
                    oracle_index,
                )
                stored_values = (
                    ("frozen_base", frozen_nll, True),
                    ("sequential_single_lora", sequential_nll, True),
                    ("independent_root_lora", independent_nll, True),
                    ("vamp_oracle", vamp_nll, checksum_stable),
                )
                raw_stored.extend(
                    StoredCompetenceMeasurement(
                        stage=stage,
                        baseline=baseline,
                        task_id=task.task_id,
                        prefix_length=prefix_length,
                        suffix_nll=value,
                        perplexity=math.exp(min(value, 80.0)),
                        frozen_base_nll=frozen_nll,
                        independent_root_nll=independent_nll,
                        improvement_over_frozen=frozen_nll - value,
                        deficit_vs_independent=value - independent_nll,
                        stored_forgetting=0.0,
                        base_checksum_stable=_tree_checksum(base_params) == base_checksum,
                        committed_checksum_stable=committed_stable,
                    )
                    for baseline, value, committed_stable in stored_values
                )

                route_results = tuple(
                    evaluate_language_router(
                        router,
                        base_params,
                        model_config,
                        graph,
                        packed_memory,
                        lora_config,
                        address_book,
                        sweep.test_examples,
                        random_seed=settings.random_router_seed,
                        hopfield_config=settings.hopfield,
                        ebt_config=settings.ebt,
                        evaluation_microbatch_size=(
                            settings.evaluation_microbatch_size
                        ),
                        suffix_nll_by_node=vamp_nll_by_node,
                    )
                    for router in ROUTER_BASELINE_NAMES
                )
                exhaustive_indices = np.asarray(
                    route_results[0].decision.selected_indices
                )
                raw_routing.extend(
                    _routing_measurement(
                        stage,
                        task.task_id,
                        prefix_length,
                        route_result,
                        exhaustive_indices,
                        settings.negative_control_curriculum,
                    )
                    for route_result in route_results
                )
                if (
                    stage == len(prepared.curriculum.tasks)
                    and prefix_length == prepared.build_config.primary_prefix_length
                ):
                    hopfield_result = next(
                        result
                        for result in route_results
                        if result.router == "vamp_hopfield"
                    )
                    final_confusion += hopfield_result.confusion_counts

        primary_vamp_rows = tuple(
            row
            for row in raw_stored
            if row.stage == stage
            and row.baseline == "vamp_oracle"
            and row.prefix_length == prepared.build_config.primary_prefix_length
        )
        frozen_mean = float(np.mean(tuple(row.frozen_base_nll for row in primary_vamp_rows)))
        vamp_mean = float(np.mean(tuple(row.suffix_nll for row in primary_vamp_rows)))
        optimizer_snapshot = init_candidate_lora_train_state(
            graph.nodes[-1].incoming_edge,
            jax.random.PRNGKey(settings.seed + stage),
            train_config,
        ).opt_state
        memory_rows.append(
            StageMemoryMeasurement(
                stage,
                account_language_memory(
                    base_params,
                    graph,
                    address_book,
                    packed_memory,
                    lora_config,
                    optimizer_state_snapshots=(optimizer_snapshot,),
                    nll_improvement=frozen_mean - vamp_mean,
                ),
            )
        )

    stored = _with_stored_forgetting(tuple(raw_stored))
    routing = _with_routing_forgetting(tuple(raw_routing))
    transfer = _transfer_measurements(
        prepared,
        adaptations,
        stored,
    )
    addressing_cost = _time_final_stage_routers(
        prepared,
        adaptations,
        base_params,
        model_config,
        lora_config,
        settings,
    )
    addressing_traces = _measure_addressing_coefficient_traces(
        prepared,
        adaptations,
        settings,
        base_params,
        model_config,
        lora_config,
    )
    samples = _generate_language_samples(
        prepared,
        adaptations,
        settings,
        base_params,
        model_config,
        lora_config,
        tokenizer,
    )
    peak_device_memory = measure_peak_device_memory(
        settings.peak_device_memory_target_bytes
    )
    return LanguageBenchmarkResult(
        adaptations=adaptations,
        settings=settings,
        stored_competence=stored,
        routing=routing,
        transfer=transfer,
        memory=tuple(memory_rows),
        addressing_cost=addressing_cost,
        addressing_traces=addressing_traces,
        samples=samples,
        peak_device_memory=peak_device_memory,
        final_confusion=final_confusion,
    )


def _measure_addressing_coefficient_traces(
    prepared: PreparedLanguageCurriculum,
    adaptations: LanguageAdaptationBaselines,
    settings: LanguageBenchmarkSettings,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> tuple[AddressingCoefficientTrace, ...]:
    """Trace both EBT initializations on one deterministic final-task example."""
    final_graph = adaptations.vamp.graph
    final_memory = pack_lora_memory(
        final_graph,
        model_config,
        lora_config,
        adaptations.vamp.max_nodes,
        adaptations.vamp.max_edges,
    )
    final_task = prepared.curriculum.tasks[-1]
    prefix_length = prepared.build_config.primary_prefix_length
    sweep = next(
        sweep
        for sweep in prepared.evaluation_sweeps
        if sweep.task_id == final_task.task_id
        and sweep.prefix_length == prefix_length
    )
    example_index = 0
    prefix_batch = sweep.test_examples[example_index].router_batch
    node_labels = tuple(str(node.node_id) for node in final_graph.nodes)
    edge_labels = tuple(
        f"{node.parent_id} → {node.node_id}"
        for node in final_graph.nodes
        if node.parent_id is not None
    )
    refinements = tuple(
        (
            router,
            trace_ebt_language_prefix(
                router,
                base_params,
                model_config,
                final_memory,
                lora_config,
                adaptations.vamp.address_book,
                prefix_batch,
                hopfield_config=settings.hopfield,
                ebt_config=settings.ebt,
            ),
        )
        for router in ("vamp_ebt_uniform", "vamp_ebt_hopfield")
    )
    return tuple(
        AddressingCoefficientTrace(
            router=router,
            task_id=str(final_task.task_id),
            prefix_length=prefix_length,
            example_index=example_index,
            node_labels=node_labels,
            edge_labels=edge_labels,
            objective_trace=np.asarray(refinement.objective_trace)[:, 0],
            node_probabilities=np.asarray(refinement.node_probability_trace)[
                :, 0, : len(node_labels)
            ],
            edge_coefficients=np.asarray(refinement.edge_coefficient_trace)[
                :, 0, : len(edge_labels)
            ],
        )
        for router, refinement in refinements
    )


def _generate_language_samples(
    prepared: PreparedLanguageCurriculum,
    adaptations: LanguageAdaptationBaselines,
    settings: LanguageBenchmarkSettings,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    tokenizer: TextTokenizer,
) -> tuple[GeneratedLanguageSample, ...]:
    final_graph = adaptations.vamp.graph
    final_memory = pack_lora_memory(
        final_graph,
        model_config,
        lora_config,
        adaptations.vamp.max_nodes,
        adaptations.vamp.max_edges,
    )
    final_task = prepared.curriculum.tasks[-1]
    sweep = next(
        sweep
        for sweep in prepared.evaluation_sweeps
        if sweep.task_id == final_task.task_id
        and sweep.prefix_length == prepared.build_config.primary_prefix_length
    )
    example = sweep.test_examples[0]
    prefix_length = prepared.build_config.primary_prefix_length
    sample_new_tokens = min(
        settings.sample_new_tokens,
        model_config.max_position_embeddings - prefix_length,
    )
    if sample_new_tokens <= 0:
        raise ValueError("model context leaves no room for generated samples")
    prompt_ids = jnp.asarray(
        example.competence_batch.input_ids[:, :prefix_length],
        dtype=jnp.int32,
    )
    prompt_mask = jnp.ones_like(prompt_ids, dtype=jnp.bool_)
    sequential_adapter = adaptations.sequential_single_lora.stages[-1].adapter
    _, sequential_memory = pack_root_adapter(
        sequential_adapter,
        model_config,
        lora_config,
    )
    independent_adapter = next(
        adapter.adapter
        for adapter in adaptations.independent_root_lora.adapters
        if adapter.task_id == final_task.task_id
    )
    _, independent_memory = pack_root_adapter(
        independent_adapter,
        model_config,
        lora_config,
    )
    oracle_index = next(
        index
        for index, node in enumerate(final_graph.nodes)
        if node.node_id == NodeId(str(final_task.task_id))
    )
    router_indices = {
        router: int(
            route_language_prefix(
                router,
                base_params,
                model_config,
                final_memory,
                lora_config,
                adaptations.vamp.address_book,
                example.router_batch,
                random_seed=settings.random_router_seed,
                hopfield_config=settings.hopfield,
                ebt_config=settings.ebt,
                evaluation_microbatch_size=settings.evaluation_microbatch_size,
            ).selected_indices[0]
        )
        for router in ROUTER_BASELINE_NAMES
    }
    generation_specs = (
        ("frozen_base", None, None),
        ("sequential_single_lora", sequential_memory, 1),
        ("independent_root_lora", independent_memory, 1),
        ("vamp_oracle", final_memory, oracle_index),
        *tuple(
            (router, final_memory, router_indices[router])
            for router in ROUTER_BASELINE_NAMES
        ),
    )
    prefix_text = tokenizer.decode(tuple(int(value) for value in prompt_ids[0]))
    return tuple(
        GeneratedLanguageSample(
            baseline=name,
            task_id=str(final_task.task_id),
            prefix=prefix_text,
            continuation=_generate_continuation(
                base_params,
                model_config,
                tokenizer,
                prompt_ids,
                prompt_mask,
                sample_new_tokens,
                lora_config,
                memory,
                node_index,
            ),
        )
        for name, memory, node_index in generation_specs
    )


def _generate_continuation(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    tokenizer: TextTokenizer,
    prompt_ids: jax.Array,
    prompt_mask: jax.Array,
    sample_new_tokens: int,
    lora_config: LoraConfig,
    memory: PackedLoraMemory | None,
    node_index: int | None,
) -> str:
    generated = greedy_generate(
        base_params,
        model_config,
        prompt_ids,
        prompt_mask,
        sample_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        lora_memory=memory,
        lora_config=None if memory is None else lora_config,
        node_index=node_index,
    )
    continuation_ids = tuple(
        int(value) for value in generated[0, prompt_ids.shape[1] :]
    )
    decoded = tokenizer.decode(continuation_ids)
    return decoded or tokenizer.decode(
        continuation_ids,
        skip_special_tokens=False,
    ) or "<EOS>"


def measure_peak_device_memory(
    target_bytes: int | None = None,
) -> PeakDeviceMemoryMeasurement:
    """Read allocator peak statistics and enforce an explicitly supplied target."""
    if target_bytes is not None and (type(target_bytes) is not int or target_bytes <= 0):
        raise ValueError("target_bytes must be a positive integer when provided")
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("JAX reported no local devices")
    device = devices[0]
    return _peak_device_memory_from_stats(
        platform=device.platform,
        device_kind=device.device_kind,
        memory_stats=device.memory_stats(),
        target_bytes=target_bytes,
    )


def _peak_device_memory_from_stats(
    *,
    platform: str,
    device_kind: str,
    memory_stats: Mapping[str, int] | None,
    target_bytes: int | None,
) -> PeakDeviceMemoryMeasurement:
    if target_bytes is not None and (type(target_bytes) is not int or target_bytes <= 0):
        raise ValueError("target_bytes must be a positive integer when provided")
    peak = _optional_memory_stat(memory_stats, "peak_bytes_in_use")
    limit = _optional_memory_stat(memory_stats, "bytes_limit")
    if target_bytes is not None and peak is None:
        raise RuntimeError(
            "the selected JAX backend does not expose peak device-memory statistics"
        )
    if target_bytes is not None and peak is not None and peak > target_bytes:
        raise MemoryError(
            f"peak device memory {peak} bytes exceeded target {target_bytes} bytes"
        )
    return PeakDeviceMemoryMeasurement(
        platform=platform,
        device_kind=device_kind,
        peak_bytes_in_use=peak,
        bytes_limit=limit,
        target_bytes=target_bytes,
        within_target=(None if target_bytes is None else True),
    )


def _optional_memory_stat(
    memory_stats: Mapping[str, int] | None,
    field_name: str,
) -> int | None:
    if memory_stats is None or field_name not in memory_stats:
        return None
    value = int(memory_stats[field_name])
    if value < 0:
        raise ValueError(f"device memory statistic {field_name} must be nonnegative")
    return value


def _vamp_stage_values(
    adaptations: LanguageAdaptationBaselines,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    stage: int,
) -> tuple[MemoryGraph[LoraEdge], AddressBook, PackedLoraMemory]:
    final_run = adaptations.vamp
    graph = MemoryGraph(nodes=final_run.graph.nodes[: stage + 1])
    keys = np.zeros_like(final_run.address_book.keys)
    keys[: stage + 1] = final_run.address_book.keys[: stage + 1]
    valid_mask = np.arange(final_run.max_nodes) < stage + 1
    address_book = AddressBook(
        node_ids=final_run.address_book.node_ids[: stage + 1]
        + (None,) * (final_run.max_nodes - stage - 1),
        keys=keys,
        valid_node_mask=valid_mask,
    )
    return graph, address_book, pack_lora_memory(
        graph,
        model_config,
        lora_config,
        final_run.max_nodes,
        final_run.max_edges,
    )


def _evaluation_sweep(
    prepared: PreparedLanguageCurriculum,
    task_id: TaskId,
    prefix_length: int,
):
    return next(
        sweep
        for sweep in prepared.evaluation_sweeps
        if sweep.task_id == task_id and sweep.prefix_length == prefix_length
    )


def _stack_competence_examples(
    examples: tuple[LanguageEvaluationExample, ...],
) -> CompetenceBatch:
    return CompetenceBatch(
        input_ids=np.concatenate(tuple(example.competence_batch.input_ids for example in examples)),
        attention_mask=np.concatenate(
            tuple(example.competence_batch.attention_mask for example in examples)
        ),
        target_ids=np.concatenate(tuple(example.competence_batch.target_ids for example in examples)),
        loss_mask=np.concatenate(tuple(example.competence_batch.loss_mask for example in examples)),
    )


def _stack_router_examples(
    examples: tuple[LanguageEvaluationExample, ...],
) -> RouterBatch:
    return RouterBatch(
        input_ids=np.concatenate(tuple(example.router_batch.input_ids for example in examples)),
        attention_mask=np.concatenate(
            tuple(example.router_batch.attention_mask for example in examples)
        ),
        target_ids=np.concatenate(tuple(example.router_batch.target_ids for example in examples)),
        loss_mask=np.concatenate(tuple(example.router_batch.loss_mask for example in examples)),
    )


def _frozen_competence_nll(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    batch: CompetenceBatch,
    *,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for row_slice in evaluation_microbatch_slices(
        batch.input_ids.shape[0],
        evaluation_microbatch_size,
    ):
        logits = apply_gpt_neo(
            base_params,
            model_config,
            jnp.asarray(batch.input_ids[row_slice]),
            jnp.asarray(batch.attention_mask[row_slice]),
        ).logits
        losses = per_token_nll(
            logits,
            jnp.asarray(batch.target_ids[row_slice]),
        )
        mask = jnp.asarray(batch.loss_mask[row_slice], dtype=jnp.float32)
        chunks.append(
            np.asarray(
                jnp.sum(losses * mask, axis=1) / jnp.sum(mask, axis=1),
                dtype=np.float32,
            )
        )
    result = np.concatenate(chunks, axis=0)
    result.flags.writeable = False
    return result


def _weighted_mean(values: np.ndarray, token_weights: np.ndarray) -> float:
    return float(np.sum(values * token_weights) / np.sum(token_weights))


def _routing_measurement(
    stage: int,
    task_id: TaskId,
    prefix_length: int,
    result,
    exhaustive_indices: np.ndarray,
    negative_control_curriculum: bool,
) -> RoutingMeasurement:
    rows = result.examples
    negative_control = (
        summarize_negative_control(
            np.asarray(tuple(row.selected_index for row in rows), dtype=np.int32),
            np.asarray(tuple(row.task_oracle_index for row in rows), dtype=np.int32),
            stage,
        )
        if negative_control_curriculum
        or result.router == "deterministic_random_node"
        else None
    )
    return RoutingMeasurement(
        stage=stage,
        router=result.router,
        task_id=task_id,
        prefix_length=prefix_length,
        example_count=len(rows),
        routing_accuracy=float(np.mean(tuple(row.task_oracle_correct for row in rows))),
        top_k_recall=float(
            np.mean(tuple(bool(row.top_k_task_oracle_hit) for row in rows))
        ),
        exhaustive_agreement=float(
            np.mean(np.asarray(result.decision.selected_indices) == exhaustive_indices)
        ),
        routed_suffix_nll=float(np.mean(tuple(row.selected_suffix_nll for row in rows))),
        task_oracle_suffix_nll=float(
            np.mean(tuple(row.task_oracle_suffix_nll for row in rows))
        ),
        best_node_suffix_nll=float(
            np.mean(tuple(row.best_node_suffix_nll for row in rows))
        ),
        task_oracle_regret=float(np.mean(tuple(row.task_oracle_regret for row in rows))),
        best_node_regret=float(np.mean(tuple(row.best_node_regret for row in rows))),
        entropy=float(np.mean(tuple(float(row.address_entropy) for row in rows))),
        margin=float(
            np.mean(tuple(float(row.top_two_probability_margin) for row in rows))
        ),
        routing_forgetting=0.0,
        negative_control_correct_count=(
            None if negative_control is None else negative_control.correct_count
        ),
        negative_control_chance_accuracy=(
            None if negative_control is None else negative_control.chance_accuracy
        ),
        negative_control_ci95_lower=(
            None
            if negative_control is None
            else negative_control.confidence_interval.lower
        ),
        negative_control_ci95_upper=(
            None
            if negative_control is None
            else negative_control.confidence_interval.upper
        ),
        negative_control_chance_in_ci95=(
            None
            if negative_control is None
            else negative_control.chance_rate_in_interval
        ),
        leakage_audit_required=(
            None
            if negative_control is None
            else negative_control.leakage_audit_required
        ),
    )


def _with_stored_forgetting(
    rows: tuple[StoredCompetenceMeasurement, ...],
) -> tuple[StoredCompetenceMeasurement, ...]:
    best: dict[tuple[str, TaskId, int], float] = {}
    updated: list[StoredCompetenceMeasurement] = []
    for row in rows:
        key = (row.baseline, row.task_id, row.prefix_length)
        current_best = min(best.get(key, math.inf), row.suffix_nll)
        best[key] = current_best
        updated.append(
            StoredCompetenceMeasurement(
                **{
                    **row.__dict__,
                    "stored_forgetting": row.suffix_nll - current_best,
                }
            )
        )
    return tuple(updated)


def _with_routing_forgetting(
    rows: tuple[RoutingMeasurement, ...],
) -> tuple[RoutingMeasurement, ...]:
    best: dict[tuple[str, TaskId, int], float] = {}
    updated: list[RoutingMeasurement] = []
    for row in rows:
        key = (row.router, row.task_id, row.prefix_length)
        current_best = min(best.get(key, math.inf), row.routed_suffix_nll)
        best[key] = current_best
        updated.append(
            RoutingMeasurement(
                **{
                    **row.__dict__,
                    "routing_forgetting": row.routed_suffix_nll - current_best,
                }
            )
        )
    return tuple(updated)


def _transfer_measurements(
    prepared: PreparedLanguageCurriculum,
    adaptations: LanguageAdaptationBaselines,
    stored: tuple[StoredCompetenceMeasurement, ...],
) -> tuple[TransferMeasurement, ...]:
    primary_prefix = prepared.build_config.primary_prefix_length
    return tuple(
        _transfer_measurement(
            stage_metrics,
            prepared.curriculum.tasks[stage_metrics.stage_index - 1],
            stored,
            primary_prefix,
        )
        for stage_metrics in adaptations.vamp.stage_metrics
    )


def _transfer_measurement(
    stage_metrics,
    task,
    stored: tuple[StoredCompetenceMeasurement, ...],
    primary_prefix: int,
) -> TransferMeasurement:
    losses = stage_metrics.candidate_step_losses
    vamp_nll = next(
        row.suffix_nll
        for row in stored
        if row.stage == stage_metrics.stage_index
        and row.task_id == task.task_id
        and row.prefix_length == primary_prefix
        and row.baseline == "vamp_oracle"
    )
    independent_nll = next(
        row.suffix_nll
        for row in stored
        if row.stage == stage_metrics.stage_index
        and row.task_id == task.task_id
        and row.prefix_length == primary_prefix
        and row.baseline == "independent_root_lora"
    )
    tokens_per_update = int(np.sum(task.train_batches[0].loss_mask))
    selected_initial = stage_metrics.parent_mean_node_nll[
        stage_metrics.parent_node_index
    ]
    return TransferMeasurement(
        stage=stage_metrics.stage_index,
        task_id=task.task_id,
        selected_parent_id=stage_metrics.parent_node_id,
        root_initial_nll=stage_metrics.parent_mean_node_nll[0],
        selected_parent_initial_nll=selected_initial,
        parent_advantage=stage_metrics.parent_mean_node_nll[0] - selected_initial,
        first_step_improvement=(
            0.0 if len(losses) == 1 else losses[0] - losses[1]
        ),
        fixed_budget_improvement=losses[0] - losses[-1],
        update_budget=len(losses),
        tokens_per_update=tokens_per_update,
        final_vamp_nll=vamp_nll,
        final_independent_nll=independent_nll,
        final_deficit_vs_independent=vamp_nll - independent_nll,
    )


def _time_final_stage_routers(
    prepared: PreparedLanguageCurriculum,
    adaptations: LanguageAdaptationBaselines,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    settings: LanguageBenchmarkSettings,
) -> tuple[RouterTimingMeasurement, ...]:
    final_stage = len(prepared.curriculum.tasks)
    _, address_book, packed_memory = _vamp_stage_values(
        adaptations,
        model_config,
        lora_config,
        final_stage,
    )
    prefix_batch = _stack_router_examples(
        tuple(
            example
            for task in prepared.curriculum.tasks
            for example in _evaluation_sweep(
                prepared,
                task.task_id,
                prepared.build_config.primary_prefix_length,
            ).test_examples
        )
    )
    return tuple(
        RouterTimingMeasurement(
            final_stage,
            router,
            time_synchronized_addressing(
                lambda router=router: route_language_prefix(
                    router,
                    base_params,
                    model_config,
                    packed_memory,
                    lora_config,
                    address_book,
                    prefix_batch,
                    random_seed=settings.random_router_seed,
                    hopfield_config=settings.hopfield,
                    ebt_config=settings.ebt,
                    evaluation_microbatch_size=settings.evaluation_microbatch_size,
                ),
                _operation_counts(
                    router,
                    prefix_batch,
                    packed_memory,
                    settings.ebt.steps,
                ),
                batch_size=prefix_batch.input_ids.shape[0],
                warm_repetitions=settings.timing_warm_repetitions,
            ),
        )
        for router in ROUTER_BASELINE_NAMES
    )


def _operation_counts(
    router: RouterBaselineName,
    prefix_batch: RouterBatch,
    packed_memory: PackedLoraMemory,
    ebt_steps: int,
) -> AddressingOperationCounts:
    prefix_tokens = int(np.sum(prefix_batch.loss_mask))
    valid_nodes = int(np.sum(packed_memory.valid_node_mask))
    valid_edges = int(np.sum(packed_memory.valid_edge_mask))
    is_ebt = router in ("vamp_ebt_uniform", "vamp_ebt_hopfield")
    base_forwards = (
        valid_nodes
        if router == "vamp_exhaustive"
        else ebt_steps + 3 + int(router == "vamp_ebt_hopfield")
        if is_ebt
        else 1
        if router == "vamp_hopfield"
        else 0
    )
    candidates_scored = (
        valid_nodes if router != "deterministic_random_node" else 0
    )
    return AddressingOperationCounts(
        prefix_tokens=prefix_tokens,
        candidates_available=valid_nodes,
        candidates_scored=candidates_scored,
        full_model_forward_equivalent_tokens=prefix_tokens * base_forwards,
        base_forwards=base_forwards,
        edge_evaluations=prefix_tokens * valid_edges * base_forwards,
        hopfield_dot_products=(
            prefix_batch.input_ids.shape[0] * valid_nodes
            if router in ("vamp_hopfield", "vamp_ebt_hopfield")
            else 0
        ),
        ebt_steps=ebt_steps if is_ebt else 0,
        ebt_mask_size=valid_nodes if is_ebt else 0,
        selected_execution_cost=1,
    )


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _path_checksum(graph: MemoryGraph[LoraEdge], node_index: int) -> str:
    return _tree_checksum(
        tuple(
            node.incoming_edge
            for node in memory_node_path(graph, graph.nodes[node_index].node_id)
            if node.incoming_edge is not None
        )
    )
