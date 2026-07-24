"""Validation-only probe preparation and staged query adapter training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import (
    LanguageAdaptationArtifact,
    extract_language_adaptation_artifact,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_baseline_training import (
    IndependentRootAdapter,
    IndependentRootLoraProgress,
    LanguageAdaptationBaselines,
    SequentialLoraProgress,
    SequentialLoraStage,
    advance_independent_root_lora_progress,
    advance_sequential_lora_progress,
    complete_independent_root_lora_progress,
    complete_sequential_lora_progress,
    init_independent_root_lora_progress,
    init_sequential_lora_progress,
)
from apm.continual.language_run import (
    LanguageStageMetrics,
    LanguageVampRun,
    advance_language_vamp_run,
    init_language_vamp_run,
    score_parent_nodes,
)
from apm.continual.language_tasks import LanguageTask, RouterBatch
from apm.data.text.tinyworlds_q_semantic.batching import (
    count_query_partition_microbatches,
    iter_query_partition_batches,
)
from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    record_sha256,
    require_identifier,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.curriculum import (
    validate_active_catalog_prefix,
)
from apm.data.text.tinyworlds_q_semantic.queries import (
    CompiledSemanticQuery,
    compile_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.selected_base import (
    QuerySelectedBase,
    load_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.training import allocator_peak_bytes
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.text import TextTokenizer
from apm.memory.graph import TaskId


AdaptationProgress = Callable[[str, str, int, float, int], None]
PreparationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, eq=False, slots=True)
class QueryTaskProbeSet:
    """One world's parent/key prefixes, with no sealed or answer tokens."""

    concept_id: str
    parent_query_ids: tuple[str, ...]
    content_key_query_ids: tuple[str, ...]
    parent_probes: tuple[RouterBatch, ...]
    content_key_probes: tuple[RouterBatch, ...]

    def __post_init__(self) -> None:
        require_identifier(self.concept_id, "query probe concept")
        pairs = (
            (self.parent_query_ids, self.parent_probes, "parent"),
            (self.content_key_query_ids, self.content_key_probes, "content-key"),
        )
        for query_ids, probes, label in pairs:
            if (
                type(query_ids) is not tuple
                or not query_ids
                or len(set(query_ids)) != len(query_ids)
                or type(probes) is not tuple
                or len(probes) != len(query_ids)
                or any(type(probe) is not RouterBatch for probe in probes)
            ):
                raise ValueError(f"query {label} probes are incomplete or duplicated")


@dataclass(frozen=True, eq=False, slots=True)
class PreparedQueryAdaptation:
    """Training-safe task metadata and validation-only routing probes."""

    catalog_sha256: str
    partition_sha256: str
    config_sha256: str
    concept_ids: tuple[str, ...]
    root_query_ids: tuple[str, ...]
    root_validation_probes: tuple[RouterBatch, ...]
    task_probes: tuple[QueryTaskProbeSet, ...]
    preparation_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.catalog_sha256, "prepared catalog"),
            (self.partition_sha256, "prepared partition"),
            (self.config_sha256, "prepared config"),
            (self.preparation_sha256, "prepared adaptation"),
        ):
            require_sha256(value, label)
        if (
            type(self.concept_ids) is not tuple
            or not self.concept_ids
            or len(set(self.concept_ids)) != len(self.concept_ids)
        ):
            raise ValueError("prepared concepts must be a unique ordered manifest")
        if (
            type(self.root_query_ids) is not tuple
            or not self.root_query_ids
            or len(self.root_query_ids) != len(self.root_validation_probes)
            or any(type(probe) is not RouterBatch for probe in self.root_validation_probes)
        ):
            raise ValueError("prepared root probes are incomplete")
        if tuple(item.concept_id for item in self.task_probes) != self.concept_ids:
            raise ValueError("prepared task probes changed ordered concepts")
        widths = {
            probe.input_ids.shape[1]
            for probe in self.root_validation_probes
            for _ in (0,)
        } | {
            probe.input_ids.shape[1]
            for task in self.task_probes
            for probe in task.parent_probes + task.content_key_probes
        }
        if len(widths) != 1:
            raise ValueError("all prepared validation probes must share one width")

    @property
    def max_nodes(self) -> int:
        """Return manifest-derived root-plus-world capacity."""
        return len(self.concept_ids) + 1

    @property
    def max_edges(self) -> int:
        """Return manifest-derived adapter-edge capacity."""
        return len(self.concept_ids)


@dataclass(frozen=True, slots=True)
class MaterializedQueryTask:
    """One bounded training stage plus exact source batch accounting."""

    task: LanguageTask
    available_batch_count: int
    materialized_batch_count: int
    active_tokens: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.available_batch_count,
                self.materialized_batch_count,
                self.active_tokens,
            )
        ):
            raise ValueError("materialized query task counts must be positive")
        if self.materialized_batch_count != len(self.task.train_batches):
            raise ValueError("materialized query task batch count changed")
        if self.materialized_batch_count > self.available_batch_count:
            raise ValueError("materialized more query batches than the partition supplies")


@dataclass(frozen=True, slots=True)
class QueryAdaptationRun:
    """Newest complete real tensor stage for independent, sequential, and VAMP."""

    stage_directory: Path
    completed_concept_ids: tuple[str, ...]
    preparation_sha256: str
    config_sha256: str
    allocator_peak_bytes: int
    adaptation: LanguageAdaptationArtifact

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_directory", Path(self.stage_directory))
        if not self.stage_directory.is_dir():
            raise FileNotFoundError(self.stage_directory)
        require_sha256(self.preparation_sha256, "adaptation run preparation")
        require_sha256(self.config_sha256, "adaptation run config")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("adaptation allocator peak must be nonnegative")
        if self.adaptation.task_order != tuple(
            TaskId(concept_id) for concept_id in self.completed_concept_ids
        ):
            raise ValueError("adaptation tensor task order changed")


def prepare_query_adaptation(
    catalog: ValidationCatalogView,
    artifact: QueryPartitionArtifact,
    tokenizer: TextTokenizer,
    preset: QueryExperimentPreset,
) -> PreparedQueryAdaptation:
    """Compile only validation prefixes and deterministically select VAMP probes."""
    if type(catalog) is not ValidationCatalogView:
        raise TypeError("query adaptation preparation requires a validation catalog view")
    validate_active_catalog_prefix(catalog, preset)
    if (
        artifact.catalog_sha256 != catalog.catalog_sha256
        or artifact.concept_ids[: preset.active_world_count] != preset.concept_ids
        or artifact.tokenizer_identity != catalog.tokenizer_identity
    ):
        raise ValueError("query adaptation catalog and partition bindings changed")
    compiled = tuple(
        query
        for query in compile_semantic_queries(
            catalog,
            tokenizer,
            split="validation",
            maximum_context_tokens=preset.context_length,
        )
        if query.concept_id in preset.concept_ids
    )
    by_concept = {
        concept_id: tuple(query for query in compiled if query.concept_id == concept_id)
        for concept_id in preset.concept_ids
    }
    if any(len(queries) != 36 for queries in by_concept.values()):
        raise ValueError("every active world must supply exactly 36 validation queries")
    root_selected = _select_queries(
        compiled,
        preset.root_probe_count,
        "root-key",
        catalog.catalog_sha256,
    )
    selected_by_concept = tuple(
        (
            concept_id,
            _select_queries(
                by_concept[concept_id],
                preset.parent_probe_count,
                f"parent:{concept_id}",
                catalog.catalog_sha256,
            ),
            _select_queries(
                by_concept[concept_id],
                preset.content_key_probe_count,
                f"content-key:{concept_id}",
                catalog.catalog_sha256,
            ),
        )
        for concept_id in preset.concept_ids
    )
    all_selected = root_selected + tuple(
        query
        for _, parent, content in selected_by_concept
        for query in parent + content
    )
    maximum_width = max(
        query.knowledge_query.router_batch.input_ids.shape[1]
        for query in all_selected
    )

    def probes(queries: tuple[CompiledSemanticQuery, ...]) -> tuple[RouterBatch, ...]:
        return tuple(
            _right_pad_router_batch(
                query.knowledge_query.router_batch,
                maximum_width,
                tokenizer.pad_token_id,
            )
            for query in queries
        )

    task_probes = tuple(
        QueryTaskProbeSet(
            concept_id=concept_id,
            parent_query_ids=tuple(query.template_id for query in parent),
            content_key_query_ids=tuple(query.template_id for query in content),
            parent_probes=probes(parent),
            content_key_probes=probes(content),
        )
        for concept_id, parent, content in selected_by_concept
    )
    root_query_ids = tuple(query.template_id for query in root_selected)
    identity_record = {
        "catalog_sha256": catalog.catalog_sha256,
        "config_sha256": preset.config_sha256,
        "concept_ids": list(preset.concept_ids),
        "partition_sha256": artifact.partition_sha256,
        "probe_width": maximum_width,
        "root_query_ids": list(root_query_ids),
        "tasks": [
            {
                "concept_id": task.concept_id,
                "content_key_query_ids": list(task.content_key_query_ids),
                "parent_query_ids": list(task.parent_query_ids),
            }
            for task in task_probes
        ],
    }
    return PreparedQueryAdaptation(
        catalog_sha256=catalog.catalog_sha256,
        partition_sha256=artifact.partition_sha256,
        config_sha256=preset.config_sha256,
        concept_ids=preset.concept_ids,
        root_query_ids=root_query_ids,
        root_validation_probes=probes(root_selected),
        task_probes=task_probes,
        preparation_sha256=record_sha256(identity_record),
    )


def materialize_query_language_task(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    concept_id: str,
    *,
    epoch: int = 0,
    maximum_batches: int | None = None,
    progress: PreparationProgress | None = None,
) -> MaterializedQueryTask:
    """Load one stage's bounded node batches and keep all test indexes closed."""
    _require_preparation_bindings(prepared, artifact, preset)
    if concept_id not in prepared.concept_ids:
        raise ValueError("query language task concept is outside the active manifest")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("query language task epoch must be nonnegative")
    limit = preset.adapter_updates if maximum_batches is None else maximum_batches
    if type(limit) is not int or limit <= 0:
        raise ValueError("query language task maximum_batches must be positive")
    available = count_query_partition_microbatches(
        artifact,
        preset,
        role="node",
        concept_id=concept_id,
        split="train",
    )
    materialize_count = min(available, limit)
    materialized_batches = []
    for completed, batch in enumerate(
        islice(
            iter_query_partition_batches(
                artifact,
                preset,
                role="node",
                concept_id=concept_id,
                split="train",
                epoch=epoch,
            ),
            materialize_count,
        ),
        start=1,
    ):
        materialized_batches.append(batch)
        if progress is not None:
            progress(f"prepare {concept_id}", completed, materialize_count)
    batches = tuple(materialized_batches)
    if len(batches) != materialize_count:
        raise ValueError("query node batch stream ended before its authenticated count")
    probe_set = prepared.task_probes[prepared.concept_ids.index(concept_id)]
    task = LanguageTask(
        task_id=TaskId(concept_id),
        train_batches=batches,
        validation_examples=(),
        test_examples=(),
        parent_probes=probe_set.parent_probes,
        content_key_probes=probe_set.content_key_probes,
    )
    return MaterializedQueryTask(
        task=task,
        available_batch_count=available,
        materialized_batch_count=len(batches),
        active_tokens=sum(int(np.sum(batch.loss_mask)) for batch in batches),
    )


def train_or_resume_query_adaptations(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    working_directory: str | Path,
    preset: QueryExperimentPreset,
    *,
    progress: AdaptationProgress | None = None,
) -> QueryAdaptationRun:
    """Train all three systems and persist real immutable tensors after each world."""
    _require_preparation_bindings(prepared, artifact, preset)
    if type(selected_base) is not QuerySelectedBase:
        raise TypeError("query adaptation requires a strict q-native selected base")
    selected_base = load_query_selected_base(
        selected_base.directory,
        artifact,
        preset,
    )
    base_checkpoint = selected_base.checkpoint
    loaded_base = load_gpt_neo_checkpoint(base_checkpoint)
    if loaded_base.config != preset.model_config:
        raise ValueError("query adaptation base architecture changed")
    base_params = loaded_base.params
    working = Path(working_directory)
    print(f"TinyWorlds-Q adaptation artifacts: {working.resolve()}", flush=True)
    stages_root = working / "stages"
    stages_root.mkdir(parents=True, exist_ok=True)
    completed_stages = _completed_stage_directories(
        stages_root,
        preset.active_world_count,
    )
    if completed_stages:
        saved = load_language_adaptation_artifact(completed_stages[-1])
        sequential, independent, vamp = _restore_progress(
            saved,
            prepared,
            artifact,
            selected_base,
            preset,
        )
    else:
        sequential_key, independent_key, vamp_key = jax.random.split(
            jax.random.PRNGKey(preset.seed),
            3,
        )
        sequential = init_sequential_lora_progress(
            base_params,
            loaded_base.config,
            preset.lora_config,
            preset.adapter_train_config,
            sequential_key,
        )
        independent = init_independent_root_lora_progress(
            base_params,
            loaded_base.config,
            preset.adapter_train_config,
            independent_key,
        )
        vamp = init_language_vamp_run(
            base_checkpoint,
            base_params,
            loaded_base.config,
            prepared.root_validation_probes,
            vamp_key,
            max_nodes=preset.max_nodes,
            max_edges=preset.max_edges,
            key_probe_count=preset.root_probe_count,
            evaluation_microbatch_size=preset.query_chunk_size,
        )
    completed_count = len(vamp.completed_tasks)
    for stage_index, concept_id in enumerate(
        preset.concept_ids[completed_count:],
        start=completed_count + 1,
    ):
        materialized = materialize_query_language_task(
            prepared,
            artifact,
            preset,
            concept_id,
            epoch=0,
        )
        task = materialized.task
        sequential = advance_sequential_lora_progress(
            sequential,
            task,
            base_params,
            loaded_base.config,
            preset.lora_config,
            training_progress=_method_progress(progress, "sequential", concept_id),
        )
        independent = advance_independent_root_lora_progress(
            independent,
            task,
            base_params,
            loaded_base.config,
            preset.lora_config,
            training_progress=_method_progress(progress, "independent", concept_id),
        )
        parent = score_parent_nodes(
            vamp,
            task.parent_probes,
            base_params,
            loaded_base.config,
            preset.lora_config,
            evaluation_microbatch_size=preset.query_chunk_size,
        )
        vamp = advance_language_vamp_run(
            vamp,
            task,
            base_params,
            loaded_base.config,
            preset.lora_config,
            preset.adapter_train_config,
            parent,
            key_probe_count=preset.content_key_probe_count,
            evaluation_microbatch_size=preset.query_chunk_size,
            training_progress=_method_progress(progress, "vamp", concept_id),
        )
        vamp = replace(
            vamp,
            completed_tasks=tuple(
                _compact_task(completed) for completed in vamp.completed_tasks
            ),
        )
        stage_artifact = extract_language_adaptation_artifact(
            _completed_baselines(sequential, independent, vamp, preset),
            loaded_base.config,
            preset.lora_config,
            config_hashes=_config_hashes(prepared, artifact, selected_base, preset),
        )
        save_language_adaptation_artifact(
            stages_root / f"stage-{stage_index:03d}",
            stage_artifact,
        )
        peak = allocator_peak_bytes()
        if peak > preset.allocator_peak_limit_bytes:
            raise MemoryError(
                f"query adaptation peak {peak:,} exceeds "
                f"{preset.allocator_peak_limit_bytes:,} bytes"
            )
    final_stage = stages_root / f"stage-{preset.active_world_count:03d}"
    final_artifact = load_language_adaptation_artifact(final_stage)
    return QueryAdaptationRun(
        stage_directory=final_stage.resolve(),
        completed_concept_ids=preset.concept_ids,
        preparation_sha256=prepared.preparation_sha256,
        config_sha256=preset.config_sha256,
        allocator_peak_bytes=allocator_peak_bytes(),
        adaptation=final_artifact,
    )


def _select_queries(
    queries: tuple[CompiledSemanticQuery, ...],
    count: int,
    purpose: str,
    catalog_sha256: str,
) -> tuple[CompiledSemanticQuery, ...]:
    if len(queries) < count:
        raise ValueError(
            f"validation query pool supplies {len(queries)} prefixes; requires {count}"
        )
    return tuple(
        sorted(
            queries,
            key=lambda query: (
                sha256(
                    (
                        f"{BENCHMARK_ID}\0probe\0{catalog_sha256}\0{purpose}\0"
                        f"{query.template_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                query.template_id,
            ),
        )[:count]
    )


def _right_pad_router_batch(
    batch: RouterBatch,
    width: int,
    pad_token_id: int,
) -> RouterBatch:
    current = batch.input_ids.shape[1]
    if current > width:
        raise ValueError("query router probe exceeds the shared width")
    if current == width:
        return batch
    padding = width - current
    return RouterBatch(
        input_ids=np.pad(
            batch.input_ids,
            ((0, 0), (0, padding)),
            constant_values=pad_token_id,
        ),
        attention_mask=np.pad(
            batch.attention_mask,
            ((0, 0), (0, padding)),
            constant_values=False,
        ),
        target_ids=np.pad(
            batch.target_ids,
            ((0, 0), (0, padding)),
            constant_values=pad_token_id,
        ),
        loss_mask=np.pad(
            batch.loss_mask,
            ((0, 0), (0, padding)),
            constant_values=False,
        ),
    )


def _compact_task(task: LanguageTask) -> LanguageTask:
    return LanguageTask(
        task_id=task.task_id,
        train_batches=task.train_batches[:1],
        validation_examples=(),
        test_examples=(),
        parent_probes=task.parent_probes,
        content_key_probes=task.content_key_probes,
    )


def _completed_baselines(
    sequential: SequentialLoraProgress,
    independent: IndependentRootLoraProgress,
    vamp: LanguageVampRun,
    preset: QueryExperimentPreset,
) -> LanguageAdaptationBaselines:
    return LanguageAdaptationBaselines(
        sequential_single_lora=complete_sequential_lora_progress(sequential),
        independent_root_lora=complete_independent_root_lora_progress(independent),
        vamp=vamp,
        train_config=preset.adapter_train_config,
        base_parameter_checksum=vamp.base_checkpoint.parameter_checksum,
    )


def _restore_progress(
    artifact_state: LanguageAdaptationArtifact,
    prepared: PreparedQueryAdaptation,
    partition: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    preset: QueryExperimentPreset,
) -> tuple[SequentialLoraProgress, IndependentRootLoraProgress, LanguageVampRun]:
    base_checkpoint = selected_base.checkpoint
    loaded_base = load_gpt_neo_checkpoint(base_checkpoint)
    expected_hashes = _config_hashes(prepared, partition, selected_base, preset)
    if (
        artifact_state.base_checkpoint.manifest_sha256
        != base_checkpoint.manifest_sha256
        or artifact_state.base_checkpoint.parameter_checksum
        != base_checkpoint.parameter_checksum
        or artifact_state.model_config != loaded_base.config
        or artifact_state.lora_config != preset.lora_config
        or artifact_state.train_config != preset.adapter_train_config
        or dict(artifact_state.config_hashes) != expected_hashes
    ):
        raise ValueError("query adaptation resume identity changed")
    task_count = len(artifact_state.task_order)
    expected_order = tuple(TaskId(item) for item in preset.concept_ids[:task_count])
    if artifact_state.task_order != expected_order:
        raise ValueError("query adaptation resume task order changed")
    compact_tasks = tuple(
        _compact_task(
            materialize_query_language_task(
                prepared,
                partition,
                preset,
                concept_id,
                maximum_batches=1,
            ).task
        )
        for concept_id in preset.concept_ids[:task_count]
    )
    sequential = SequentialLoraProgress(
        stages=tuple(
            SequentialLoraStage(
                stage_index=record.stage_index,
                task_id=record.task_id,
                adapter=record.adapter,
                step_losses=record.training_trace,
            )
            for record in artifact_state.sequential_stages
        ),
        current_adapter=artifact_state.sequential_stages[-1].adapter,
        rng_key=jnp.asarray(artifact_state.rng_state.sequential_single_lora),
        train_config=artifact_state.train_config,
        base_parameter_checksum=artifact_state.base_checkpoint.parameter_checksum,
    )
    independent = IndependentRootLoraProgress(
        adapters=tuple(
            IndependentRootAdapter(
                task_id=record.task_id,
                adapter=record.adapter,
                step_losses=record.training_trace,
            )
            for record in artifact_state.independent_adapters
        ),
        rng_key=jnp.asarray(artifact_state.rng_state.independent_root_lora),
        train_config=artifact_state.train_config,
        base_parameter_checksum=artifact_state.base_checkpoint.parameter_checksum,
    )
    vamp = LanguageVampRun(
        base_checkpoint=base_checkpoint,
        graph=artifact_state.vamp_graph,
        address_book=artifact_state.address_book,
        rng_key=jnp.asarray(artifact_state.rng_state.vamp),
        completed_tasks=compact_tasks,
        stage_metrics=tuple(
            LanguageStageMetrics(
                stage_index=record.stage_index,
                task_id=record.task_id,
                parent_node_index=record.parent_node_index,
                parent_node_id=record.parent_node_id,
                parent_mean_node_nll=record.parent_mean_node_nll,
                candidate_step_losses=record.training_trace,
                task_metrics=(),
            )
            for record in artifact_state.vamp_stages
        ),
        max_nodes=artifact_state.max_nodes,
        max_edges=artifact_state.max_edges,
    )
    return sequential, independent, vamp


def _config_hashes(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    preset: QueryExperimentPreset,
) -> dict[str, str]:
    return {
        "query-base-manifest": selected_base.checkpoint.manifest_sha256,
        "query-base-parameters": selected_base.checkpoint.parameter_checksum,
        "query-base-selection": selected_base.selection_sha256,
        "query-catalog": prepared.catalog_sha256,
        "query-experiment": preset.config_sha256,
        "query-partition": artifact.partition_sha256,
        "query-preparation": prepared.preparation_sha256,
    }


def _require_preparation_bindings(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
) -> None:
    if (
        prepared.partition_sha256 != artifact.partition_sha256
        or prepared.catalog_sha256 != artifact.catalog_sha256
        or prepared.config_sha256 != preset.config_sha256
        or prepared.concept_ids != preset.concept_ids
        or artifact.concept_ids[: preset.active_world_count] != preset.concept_ids
        or prepared.max_nodes != preset.max_nodes
        or prepared.max_edges != preset.max_edges
    ):
        raise ValueError("prepared query adaptation identity changed")


def _completed_stage_directories(root: Path, maximum: int) -> tuple[Path, ...]:
    candidates = tuple(sorted(root.glob("stage-[0-9][0-9][0-9]")))
    expected = tuple(
        root / f"stage-{index:03d}" for index in range(1, len(candidates) + 1)
    )
    if candidates != expected or len(candidates) > maximum:
        raise ValueError("query adaptation stages are not one contiguous prefix")
    return candidates


def _method_progress(
    progress: AdaptationProgress | None,
    method: str,
    concept_id: str,
) -> Callable[[int, float, int], None] | None:
    if progress is None:
        return None
    return lambda step, loss, total: progress(method, concept_id, step, loss, total)


__all__ = [
    "AdaptationProgress",
    "MaterializedQueryTask",
    "PreparationProgress",
    "PreparedQueryAdaptation",
    "QueryAdaptationRun",
    "QueryTaskProbeSet",
    "materialize_query_language_task",
    "prepare_query_adaptation",
    "train_or_resume_query_adaptations",
]
