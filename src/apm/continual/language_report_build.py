"""Convert a measured language benchmark into the standard report bundle."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import jax
import numpy as np

from apm.continual.language_benchmark_run import LanguageBenchmarkResult
from apm.continual.language_benchmarks import (
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.continual.language_report import (
    AddressConfusion,
    LanguageReportBundle,
    LanguageReportManifest,
    ReportRecord,
    write_language_report,
)
from apm.data.text.language_tasks import PreparedLanguageCurriculum
from apm.lm.parameters import GptNeoParams
from apm.memory.dense import tree_nbytes
from apm.memory.graph import MemoryGraph
from apm.memory.visualization import EdgeVisualStats, NodeVisualStats


def build_language_report_bundle(
    manifest: LanguageReportManifest,
    prepared: PreparedLanguageCurriculum,
    benchmark: LanguageBenchmarkResult,
    base_params: GptNeoParams,
) -> LanguageReportBundle:
    """Project one completed benchmark into every standard report input."""
    final_graph = benchmark.adaptations.vamp.graph
    node_stats, edge_stats = _graph_visual_stats(
        prepared,
        benchmark,
        base_params,
    )
    return LanguageReportBundle(
        manifest=manifest,
        stage_metrics=tuple(
            _record(
                stage=metrics.stage_index,
                task_id=str(metrics.task_id),
                parent=str(metrics.parent_node_id),
                parent_node_index=metrics.parent_node_index,
                parent_prefix_nll=metrics.parent_mean_node_nll[
                    metrics.parent_node_index
                ],
                final_candidate_loss=metrics.candidate_step_losses[-1],
            )
            for metrics in benchmark.adaptations.vamp.stage_metrics
        ),
        stored_competence=tuple(
            _record(
                stage=row.stage,
                baseline=row.baseline,
                task_id=str(row.task_id),
                prefix_length=row.prefix_length,
                suffix_nll=row.suffix_nll,
                perplexity=row.perplexity,
                frozen_base_nll=row.frozen_base_nll,
                independent_root_nll=row.independent_root_nll,
                improvement_over_frozen=row.improvement_over_frozen,
                deficit_vs_independent=row.deficit_vs_independent,
                stored_forgetting=row.stored_forgetting,
                base_checksum_stable=row.base_checksum_stable,
                committed_checksum_stable=row.committed_checksum_stable,
            )
            for row in benchmark.stored_competence
        ),
        routing_metrics=tuple(
            _record(
                stage=row.stage,
                router=row.router,
                task_id=str(row.task_id),
                prefix_length=row.prefix_length,
                example_count=row.example_count,
                routing_accuracy=row.routing_accuracy,
                top_k_recall=row.top_k_recall,
                exhaustive_agreement=row.exhaustive_agreement,
                routed_suffix_nll=row.routed_suffix_nll,
                task_oracle_suffix_nll=row.task_oracle_suffix_nll,
                best_node_suffix_nll=row.best_node_suffix_nll,
                task_oracle_regret=row.task_oracle_regret,
                best_node_regret=row.best_node_regret,
                entropy=row.entropy,
                margin=row.margin,
                routing_forgetting=row.routing_forgetting,
                negative_control_correct_count=(
                    row.negative_control_correct_count
                ),
                negative_control_chance_accuracy=(
                    row.negative_control_chance_accuracy
                ),
                negative_control_ci95_lower=row.negative_control_ci95_lower,
                negative_control_ci95_upper=row.negative_control_ci95_upper,
                negative_control_chance_in_ci95=(
                    row.negative_control_chance_in_ci95
                ),
                leakage_audit_required=row.leakage_audit_required,
            )
            for row in benchmark.routing
        ),
        transfer_metrics=tuple(
            _record(
                stage=row.stage,
                task_id=str(row.task_id),
                selected_parent_id=str(row.selected_parent_id),
                transfer=row.parent_advantage,
                root_initial_nll=row.root_initial_nll,
                selected_parent_initial_nll=row.selected_parent_initial_nll,
                first_step_improvement=row.first_step_improvement,
                fixed_budget_improvement=row.fixed_budget_improvement,
                update_budget=row.update_budget,
                tokens_per_update=row.tokens_per_update,
                final_vamp_nll=row.final_vamp_nll,
                final_independent_nll=row.final_independent_nll,
                final_deficit_vs_independent=row.final_deficit_vs_independent,
            )
            for row in benchmark.transfer
        ),
        memory_metrics=tuple(
            _record(
                stage=row.stage,
                persistent_bytes=row.accounting.persistent_bytes,
                runtime_bytes=(
                    row.accounting.packed_runtime_bytes
                    + row.accounting.optimizer_peak_bytes
                ),
                base_parameter_count=row.accounting.base_parameter_count,
                base_bytes=row.accounting.base_bytes,
                committed_lora_bytes=row.accounting.committed_lora_bytes,
                address_key_bytes=row.accounting.address_key_bytes,
                graph_metadata_bytes=row.accounting.graph_metadata_bytes,
                packed_runtime_bytes=row.accounting.packed_runtime_bytes,
                packed_padding_bytes=row.accounting.packed_padding_bytes,
                optimizer_peak_bytes=row.accounting.optimizer_peak_bytes,
                bytes_per_task=row.accounting.bytes_per_task,
                nll_improvement=row.accounting.nll_improvement,
                bytes_per_nll_improvement=(
                    row.accounting.bytes_per_nll_improvement
                ),
                peak_device_memory_bytes=(
                    benchmark.peak_device_memory.peak_bytes_in_use
                    if row.stage == len(benchmark.memory)
                    else None
                ),
                peak_device_memory_target_bytes=(
                    benchmark.peak_device_memory.target_bytes
                    if row.stage == len(benchmark.memory)
                    else None
                ),
                device_memory_limit_bytes=(
                    benchmark.peak_device_memory.bytes_limit
                    if row.stage == len(benchmark.memory)
                    else None
                ),
                peak_device_memory_within_target=(
                    benchmark.peak_device_memory.within_target
                    if row.stage == len(benchmark.memory)
                    else None
                ),
                device_platform=(
                    benchmark.peak_device_memory.platform
                    if row.stage == len(benchmark.memory)
                    else None
                ),
                device_kind=(
                    benchmark.peak_device_memory.device_kind
                    if row.stage == len(benchmark.memory)
                    else None
                ),
            )
            for row in benchmark.memory
        ),
        addressing_cost=tuple(
            _addressing_cost_record(row.stage, row.router, row.timing)
            for row in benchmark.addressing_cost
        ),
        competence_curve=_competence_curve(prepared, benchmark),
        routing_curve=_routing_curve(prepared, benchmark),
        memory_curve=tuple(
            _record(
                stage=row.stage,
                persistent_bytes=row.accounting.persistent_bytes,
                packed_runtime_bytes=row.accounting.packed_runtime_bytes,
                optimizer_peak_bytes=row.accounting.optimizer_peak_bytes,
            )
            for row in benchmark.memory
        ),
        addressing_traces=benchmark.addressing_traces,
        address_confusion=AddressConfusion(
            tuple(str(node.node_id) for node in final_graph.nodes),
            benchmark.final_confusion,
        ),
        graph=cast(MemoryGraph[object], final_graph),
        node_stats=node_stats,
        edge_stats=edge_stats,
        samples=benchmark.samples,
    )


def write_language_benchmark_report(
    results_root: str | Path,
    manifest: LanguageReportManifest,
    prepared: PreparedLanguageCurriculum,
    benchmark: LanguageBenchmarkResult,
    base_params: GptNeoParams,
) -> Path:
    """Emit the deterministic report projection of one completed benchmark."""
    return write_language_report(
        results_root,
        build_language_report_bundle(
            manifest,
            prepared,
            benchmark,
            base_params,
        ),
    )


def _record(**values: str | int | float | bool | None) -> ReportRecord:
    return ReportRecord(tuple(values.items()))


def _addressing_cost_record(stage, router, timing) -> ReportRecord:
    operations = timing.operations
    return _record(
        stage=stage,
        router=router,
        cold_seconds=timing.cold_compile_seconds,
        warm_seconds=timing.warm_latency_seconds,
        warm_throughput=timing.warm_throughput_examples_per_second,
        batch_size=timing.batch_size,
        prefix_tokens=operations.prefix_tokens,
        candidates_available=operations.candidates_available,
        candidates_scored=operations.candidates_scored,
        forward_equivalent_tokens=operations.full_model_forward_equivalent_tokens,
        base_forwards=operations.base_forwards,
        edge_evaluations=operations.edge_evaluations,
        hopfield_dot_products=operations.hopfield_dot_products,
        ebt_steps=operations.ebt_steps,
        ebt_mask_size=operations.ebt_mask_size,
        selected_execution_cost=operations.selected_execution_cost,
    )


def _competence_curve(
    prepared: PreparedLanguageCurriculum,
    benchmark: LanguageBenchmarkResult,
) -> tuple[ReportRecord, ...]:
    primary = prepared.build_config.primary_prefix_length
    return tuple(
        _record(
            stage=stage,
            **{
                baseline: float(
                    np.mean(
                        tuple(
                            row.suffix_nll
                            for row in benchmark.stored_competence
                            if row.stage == stage
                            and row.prefix_length == primary
                            and row.baseline == baseline
                        )
                    )
                )
                for baseline in STORED_BASELINE_NAMES
            },
        )
        for stage in range(1, len(prepared.curriculum.tasks) + 1)
    )


def _routing_curve(
    prepared: PreparedLanguageCurriculum,
    benchmark: LanguageBenchmarkResult,
) -> tuple[ReportRecord, ...]:
    primary = prepared.build_config.primary_prefix_length
    return tuple(
        _record(
            stage=stage,
            **{
                router: float(
                    np.mean(
                        tuple(
                            row.routing_accuracy
                            for row in benchmark.routing
                            if row.stage == stage
                            and row.prefix_length == primary
                            and row.router == router
                        )
                    )
                )
                for router in ROUTER_BASELINE_NAMES
            },
        )
        for stage in range(1, len(prepared.curriculum.tasks) + 1)
    )


def _graph_visual_stats(
    prepared: PreparedLanguageCurriculum,
    benchmark: LanguageBenchmarkResult,
    base_params: GptNeoParams,
) -> tuple[tuple[NodeVisualStats, ...], tuple[EdgeVisualStats, ...]]:
    graph = benchmark.adaptations.vamp.graph
    primary = prepared.build_config.primary_prefix_length
    final_stage = len(prepared.curriculum.tasks)
    final_hopfield = tuple(
        row
        for row in benchmark.routing
        if row.stage == final_stage
        and row.prefix_length == primary
        and row.router == "vamp_hopfield"
    )
    node_ids = tuple(str(node.node_id) for node in graph.nodes)
    confusion = benchmark.final_confusion
    winner_by_task = tuple(int(np.argmax(row)) for row in confusion)
    node_stats = tuple(
        NodeVisualStats(
            node_id=str(node.node_id),
            trained_task=("root" if node.trained_task is None else str(node.trained_task)),
            depth=node.depth,
            memory_bytes=(
                tree_nbytes(base_params)
                if node.incoming_edge is None
                else tree_nbytes(node.incoming_edge)
            ),
            eval_wins=tuple(
                node_ids[task_index]
                for task_index, winner_index in enumerate(winner_by_task)
                if winner_index == node_index and task_index > 0
            ),
            best_task_accuracy=(
                0.0
                if node_index == 0
                else next(
                    (
                        row.routing_accuracy
                        for row in final_hopfield
                        if str(row.task_id) == str(node.trained_task)
                    ),
                    0.0,
                )
            ),
        )
        for node_index, node in enumerate(graph.nodes)
    )
    edge_stats = tuple(
        EdgeVisualStats(
            parent_id=str(node.parent_id),
            child_id=str(node.node_id),
            child_task=str(node.trained_task),
            delta_l2_norm=_tree_l2_norm(node.incoming_edge),
            delta_bytes=tree_nbytes(node.incoming_edge),
            eval_gain=next(
                row.improvement_over_frozen
                for row in benchmark.stored_competence
                if row.stage == node.train_stage
                and row.task_id == node.trained_task
                and row.prefix_length == primary
                and row.baseline == "vamp_oracle"
            ),
        )
        for node in graph.nodes
        if node.incoming_edge is not None
    )
    return node_stats, edge_stats


def _tree_l2_norm(tree) -> float:
    squared = sum(
        float(np.sum(np.square(np.asarray(leaf, dtype=np.float64))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return float(np.sqrt(squared))
