"""Resumable sequential and independent-adapter training for nouns-v2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import (
    LanguageAdaptationArtifact,
    attach_language_baseline_runs,
    flatten_lora_edge,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_baseline_training import (
    IndependentRootAdapter,
    IndependentRootLoraProgress,
    SequentialLoraProgress,
    SequentialLoraStage,
    advance_independent_root_lora_progress,
    advance_sequential_lora_progress,
    complete_independent_root_lora_progress,
    complete_sequential_lora_progress,
    init_independent_root_lora_progress,
    init_sequential_lora_progress,
)
from apm.continual.language_tasks import LanguageTask
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedTokenBatchSequence,
    NounSelectedBase,
    allocator_peak_bytes,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASELINE_STAGE_FORMAT,
    CHECKPOINT_ROOT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.lm.checkpoint import LoadedGptNeoCheckpoint, load_gpt_neo_checkpoint
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.training import LmTrainConfig
from apm.memory.graph import TaskId


BaselineProgress = Callable[[str, int, int, float], None]
_SEQUENTIAL_SEED_OFFSET = 3
_INDEPENDENT_SEED_OFFSET = 4


def run_or_resume_nouns_v2_baselines(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress: BaselineProgress | None = None,
) -> LanguageAdaptationArtifact:
    """Train or task-boundary-resume both stored adapter controls."""
    loaded_base, lora_config, train_config, run_sha256 = _baseline_execution(
        partition,
        preset,
        selected_base,
        vamp_stages,
    )
    root = Path(checkpoint_root) / "baselines" / run_sha256
    root.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns-v2 baseline checkpoint directory: {root.resolve()}", flush=True)
    tasks = tuple(_baseline_task(partition, task_id, preset) for task_id in partition.task_ids)
    latest = _latest_baseline_stage(root, partition.task_ids)
    if latest is None:
        sequential = init_sequential_lora_progress(
            loaded_base.params,
            loaded_base.config,
            lora_config,
            train_config,
            jax.random.PRNGKey(preset.seed + _SEQUENTIAL_SEED_OFFSET),
        )
        independent = init_independent_root_lora_progress(
            loaded_base.params,
            loaded_base.config,
            train_config,
            jax.random.PRNGKey(preset.seed + _INDEPENDENT_SEED_OFFSET),
        )
    else:
        stage_index = int(latest.name.split("-", 2)[1])
        persisted = load_language_adaptation_artifact(latest / "adaptation")
        _require_baseline_stage(
            _load_stage_record(latest),
            persisted,
            vamp_stages[stage_index - 1],
            partition,
            preset,
            selected_base,
            run_sha256,
            latest,
        )
        sequential, independent = _progress_from_artifact(persisted)

    for stage_index in range(len(sequential.stages), len(tasks)):
        task = tasks[stage_index]
        sequential_started = time.monotonic()
        sequential = advance_sequential_lora_progress(
            sequential,
            task,
            loaded_base.params,
            loaded_base.config,
            lora_config,
            training_progress=_method_progress(
                progress,
                f"sequential-{task.task_id}",
            ),
        )
        sequential_seconds = time.monotonic() - sequential_started
        independent_started = time.monotonic()
        independent = advance_independent_root_lora_progress(
            independent,
            task,
            loaded_base.params,
            loaded_base.config,
            lora_config,
            training_progress=_method_progress(
                progress,
                f"independent-{task.task_id}",
            ),
        )
        independent_seconds = time.monotonic() - independent_started
        peak_bytes = allocator_peak_bytes()
        if peak_bytes > preset.allocator_peak_limit_bytes:
            raise RuntimeError("noun baseline training exceeded the 12 GiB allocator limit")
        combined = attach_language_baseline_runs(
            vamp_stages[stage_index],
            complete_sequential_lora_progress(sequential),
            complete_independent_root_lora_progress(independent),
            config_hashes={"baseline_run": run_sha256},
        )
        target = root / f"stage-{stage_index + 1:03d}-{task.task_id}"
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root))
        try:
            save_language_adaptation_artifact(temporary / "adaptation", combined)
            if not isinstance(task.train_batches, IndexedTokenBatchSequence):
                raise TypeError("noun baseline task lost its indexed training sequence")
            consumed = task.train_batches.consumed_story_ids(preset.adapter_updates)
            stage_core = {
                "adaptation_manifest_sha256": _file_sha256(
                    temporary / "adaptation" / "manifest.json"
                ),
                "adaptation_tensor_checksum": combined.tensor_checksum,
                "baseline_run_sha256": run_sha256,
                "base_training_sha256": selected_base.training_sha256,
                "elapsed_seconds": {
                    "independent_root_lora": independent_seconds,
                    "sequential_single_lora": sequential_seconds,
                },
                "format": BASELINE_STAGE_FORMAT,
                "independent_adapter_checksum": _adapter_checksum(
                    independent.adapters[-1].adapter,
                    loaded_base.config,
                    lora_config,
                ),
                "partition_sha256": partition.partition_sha256,
                "peak_allocator_bytes": peak_bytes,
                "preset_sha256": preset.config_sha256,
                "sequential_adapter_checksum": _adapter_checksum(
                    sequential.stages[-1].adapter,
                    loaded_base.config,
                    lora_config,
                ),
                "source_story_count": task.train_batches.story_count,
                "source_vamp_tensor_checksum": vamp_stages[stage_index].tensor_checksum,
                "source_window_count": task.train_batches.window_count,
                "stage_index": stage_index + 1,
                "task_id": str(task.task_id),
                "unique_consumed_story_count": len(consumed),
            }
            _atomic_write(
                temporary / "stage.json",
                canonical_json_bytes(
                    {**stage_core, "stage_sha256": record_sha256(stage_core)}
                ),
            )
            os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return load_nouns_v2_baseline_stages(
        partition,
        preset,
        selected_base,
        vamp_stages,
        checkpoint_root,
    )[-1]


def load_nouns_v2_baseline_stages(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> tuple[LanguageAdaptationArtifact, ...]:
    """Strict-load every sequential/independent task-boundary checkpoint."""
    _, _, _, run_sha256 = _baseline_execution(
        partition,
        preset,
        selected_base,
        vamp_stages,
    )
    root = Path(checkpoint_root) / "baselines" / run_sha256
    expected_paths = tuple(
        root / f"stage-{index:03d}-{task_id}"
        for index, task_id in enumerate(partition.task_ids, start=1)
    )
    latest = _latest_baseline_stage(root, partition.task_ids)
    if latest != expected_paths[-1] or any(not path.is_dir() for path in expected_paths):
        raise ValueError("noun baseline audit requires all canonical stages")
    stages: list[LanguageAdaptationArtifact] = []
    prior_sequential: tuple[str, ...] = ()
    prior_independent: tuple[str, ...] = ()
    for stage_index, (path, vamp) in enumerate(zip(expected_paths, vamp_stages), start=1):
        persisted = load_language_adaptation_artifact(path / "adaptation")
        _require_baseline_stage(
            _load_stage_record(path),
            persisted,
            vamp,
            partition,
            preset,
            selected_base,
            run_sha256,
            path,
        )
        sequential_checksums = tuple(
            _adapter_checksum(record.adapter, persisted.model_config, persisted.lora_config)
            for record in persisted.sequential_stages
        )
        independent_checksums = tuple(
            _adapter_checksum(record.adapter, persisted.model_config, persisted.lora_config)
            for record in persisted.independent_adapters
        )
        if (
            sequential_checksums[: len(prior_sequential)] != prior_sequential
            or independent_checksums[: len(prior_independent)] != prior_independent
        ):
            raise ValueError("persisted noun baseline stage changed an earlier snapshot")
        prior_sequential = sequential_checksums
        prior_independent = independent_checksums
        if len(persisted.task_order) != stage_index:
            raise ValueError("noun baseline stage prefix length changed")
        stages.append(persisted)
    return tuple(stages)


def _baseline_execution(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
) -> tuple[LoadedGptNeoCheckpoint, LoraConfig, LmTrainConfig, str]:
    if len(vamp_stages) != len(partition.task_ids):
        raise ValueError("baseline training requires every strict VAMP stage")
    loaded_base = load_gpt_neo_checkpoint(selected_base.reference)
    lora_config = LoraConfig(rank=preset.lora_rank, alpha=preset.lora_alpha)
    train_config = LmTrainConfig(
        learning_rate=preset.adapter_learning_rate,
        steps=preset.adapter_updates,
        batch_size=preset.microbatch_size,
        weight_decay=preset.adapter_weight_decay,
        gradient_clip_norm=preset.gradient_clip_norm,
    )
    if any(
        stage.base_checkpoint.parameter_checksum
        != selected_base.reference.parameter_checksum
        or stage.model_config != loaded_base.config
        or stage.lora_config != lora_config
        or stage.train_config != train_config
        or tuple(str(task) for task in stage.task_order)
        != partition.task_ids[:stage_index]
        for stage_index, stage in enumerate(vamp_stages, start=1)
    ):
        raise ValueError("baseline training VAMP source bindings changed")
    run_sha256 = record_sha256(
        {
            "base_training_sha256": selected_base.training_sha256,
            "baseline_families": [
                "sequential_single_lora",
                "independent_root_lora",
            ],
            "format": BASELINE_STAGE_FORMAT,
            "independent_seed": preset.seed + _INDEPENDENT_SEED_OFFSET,
            "lora_config": _dataclass_record(lora_config),
            "partition_sha256": partition.partition_sha256,
            "preset_sha256": preset.config_sha256,
            "sequential_seed": preset.seed + _SEQUENTIAL_SEED_OFFSET,
            "train_config": _dataclass_record(train_config),
            "training_order_namespace": "adapter-{task_id}",
            "vamp_final_tensor_checksum": vamp_stages[-1].tensor_checksum,
        }
    )
    return loaded_base, lora_config, train_config, run_sha256


def _baseline_task(
    partition: NounsV2PartitionArtifact,
    task_id: str,
    preset: NounsV2ExperimentPreset,
) -> LanguageTask:
    return LanguageTask(
        task_id=TaskId(task_id),
        train_batches=IndexedTokenBatchSequence(
            partition,
            f"task-{task_id}-train",
            context_length=preset.context_length,
            batch_size=preset.microbatch_size,
            order_namespace=f"adapter-{task_id}",
        ),
        validation_examples=(),
        test_examples=(),
    )


def _progress_from_artifact(
    artifact: LanguageAdaptationArtifact,
) -> tuple[SequentialLoraProgress, IndependentRootLoraProgress]:
    sequential_stages = tuple(
        SequentialLoraStage(
            record.stage_index,
            record.task_id,
            record.adapter,
            record.training_trace,
        )
        for record in artifact.sequential_stages
    )
    independent_adapters = tuple(
        IndependentRootAdapter(
            record.task_id,
            record.adapter,
            record.training_trace,
        )
        for record in artifact.independent_adapters
    )
    return (
        SequentialLoraProgress(
            sequential_stages,
            sequential_stages[-1].adapter,
            jnp.asarray(artifact.rng_state.sequential_single_lora, dtype=jnp.uint32),
            artifact.train_config,
            artifact.base_checkpoint.parameter_checksum,
        ),
        IndependentRootLoraProgress(
            independent_adapters,
            jnp.asarray(artifact.rng_state.independent_root_lora, dtype=jnp.uint32),
            artifact.train_config,
            artifact.base_checkpoint.parameter_checksum,
        ),
    )


def _require_baseline_stage(
    record: dict[str, object],
    artifact: LanguageAdaptationArtifact,
    vamp: LanguageAdaptationArtifact,
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    run_sha256: str,
    path: Path,
) -> None:
    expected_fields = {
        "adaptation_manifest_sha256",
        "adaptation_tensor_checksum",
        "baseline_run_sha256",
        "base_training_sha256",
        "elapsed_seconds",
        "format",
        "independent_adapter_checksum",
        "partition_sha256",
        "peak_allocator_bytes",
        "preset_sha256",
        "sequential_adapter_checksum",
        "source_story_count",
        "source_vamp_tensor_checksum",
        "source_window_count",
        "stage_index",
        "task_id",
        "unique_consumed_story_count",
    }
    stage_index = len(artifact.task_order)
    if not 1 <= stage_index <= len(partition.task_ids):
        raise ValueError("noun baseline stage prefix length changed")
    elapsed = record.get("elapsed_seconds")
    elapsed_values = tuple(elapsed.values()) if type(elapsed) is dict else ()
    elapsed_keys_match = (
        type(elapsed) is dict
        and set(elapsed) == {"independent_root_lora", "sequential_single_lora"}
    )
    expected_batches = IndexedTokenBatchSequence(
        partition,
        f"task-{partition.task_ids[stage_index - 1]}-train",
        context_length=preset.context_length,
        batch_size=preset.microbatch_size,
        order_namespace=f"adapter-{partition.task_ids[stage_index - 1]}",
    )
    expected_consumed_count = len(
        expected_batches.consumed_story_ids(preset.adapter_updates)
    )
    numeric_counts = tuple(
        record.get(name)
        for name in (
            "peak_allocator_bytes",
            "source_story_count",
            "source_window_count",
            "unique_consumed_story_count",
        )
    )
    if (
        set(record) != expected_fields
        or record.get("format") != BASELINE_STAGE_FORMAT
        or record.get("baseline_run_sha256") != run_sha256
        or record.get("base_training_sha256") != selected_base.training_sha256
        or record.get("partition_sha256") != partition.partition_sha256
        or record.get("preset_sha256") != preset.config_sha256
        or record.get("stage_index") != stage_index
        or record.get("task_id") != partition.task_ids[stage_index - 1]
        or path.name != f"stage-{stage_index:03d}-{partition.task_ids[stage_index - 1]}"
        or record.get("source_vamp_tensor_checksum") != vamp.tensor_checksum
        or record.get("source_story_count") != expected_batches.story_count
        or record.get("source_window_count") != expected_batches.window_count
        or record.get("unique_consumed_story_count") != expected_consumed_count
        or record.get("adaptation_tensor_checksum") != artifact.tensor_checksum
        or record.get("adaptation_manifest_sha256")
        != _file_sha256(path / "adaptation" / "manifest.json")
        or record.get("sequential_adapter_checksum")
        != _adapter_checksum(
            artifact.sequential_stages[-1].adapter,
            artifact.model_config,
            artifact.lora_config,
        )
        or record.get("independent_adapter_checksum")
        != _adapter_checksum(
            artifact.independent_adapters[-1].adapter,
            artifact.model_config,
            artifact.lora_config,
        )
        or not elapsed_keys_match
    ):
        raise ValueError("noun baseline stage metadata changed")
    if (
        any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in elapsed_values
        )
        or any(type(value) is not int or value <= 0 for value in numeric_counts)
        or int(numeric_counts[-1]) > int(numeric_counts[1])
        or int(numeric_counts[0]) > preset.allocator_peak_limit_bytes
        or tuple(str(task) for task in artifact.task_order)
        != partition.task_ids[:stage_index]
        or len(artifact.sequential_stages) != stage_index
        or len(artifact.independent_adapters) != stage_index
        or dict(artifact.config_hashes).get("baseline_run") != run_sha256
        or not _matches_vamp_source(artifact, vamp)
    ):
        raise ValueError("noun baseline stage bindings changed")


def _matches_vamp_source(
    combined: LanguageAdaptationArtifact,
    vamp: LanguageAdaptationArtifact,
) -> bool:
    excluded = (
        "sequential.",
        "independent.",
        "rng.sequential_single_lora",
        "rng.independent_root_lora",
    )
    tensors = lambda artifact: tuple(
        (name, digest)
        for name, digest in artifact.tensor_checksums
        if not name.startswith(excluded)
    )
    node_metadata = lambda artifact: tuple(
        (
            str(node.node_id),
            None if node.parent_id is None else str(node.parent_id),
            None if node.trained_task is None else str(node.trained_task),
            node.train_stage,
            node.depth,
        )
        for node in artifact.vamp_graph.nodes
    )
    return (
        combined.base_checkpoint == vamp.base_checkpoint
        and combined.model_config == vamp.model_config
        and combined.lora_config == vamp.lora_config
        and combined.train_config == vamp.train_config
        and combined.task_order == vamp.task_order
        and combined.vamp_stages == vamp.vamp_stages
        and combined.address_book.node_ids == vamp.address_book.node_ids
        and node_metadata(combined) == node_metadata(vamp)
        and tensors(combined) == tensors(vamp)
    )


def _latest_baseline_stage(root: Path, task_ids: tuple[str, ...]) -> Path | None:
    stages = tuple(sorted(path for path in root.glob("stage-*-*") if path.is_dir()))
    expected = tuple(
        f"stage-{index:03d}-{task_id}"
        for index, task_id in enumerate(task_ids[: len(stages)], start=1)
    )
    if tuple(path.name for path in stages) != expected:
        raise ValueError("noun baseline stage directories are not one canonical prefix")
    return stages[-1] if stages else None


def _load_stage_record(path: Path) -> dict[str, object]:
    payload = (path / "stage.json").read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or payload != canonical_json_bytes(record):
        raise ValueError("noun baseline stage JSON is not canonical")
    supplied = record.pop("stage_sha256", None)
    if supplied != record_sha256(record):
        raise ValueError("noun baseline stage identity changed")
    return record


def _adapter_checksum(
    adapter: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> str:
    digest = sha256()
    for name, leaf in sorted(
        flatten_lora_edge(adapter, model_config, lora_config).items()
    ):
        array = np.asarray(leaf)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dataclass_record(value: object) -> dict[str, object]:
    return {
        field.name: (
            _dataclass_record(field_value)
            if hasattr(field_value, "__dataclass_fields__")
            else field_value
        )
        for field in fields(value)
        for field_value in (getattr(value, field.name),)
    }


def _method_progress(
    progress: BaselineProgress | None,
    phase: str,
) -> Callable[[int, float, int], None] | None:
    return (
        None
        if progress is None
        else lambda update, loss, total: progress(phase, update, total, loss)
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


__all__ = [
    "load_nouns_v2_baseline_stages",
    "run_or_resume_nouns_v2_baselines",
]
