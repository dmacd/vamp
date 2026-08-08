"""Resumable sequential full-parameter fine-tuning for nouns-v2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
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

from apm.continual.language_full_finetune_training import (
    run_full_parameter_updates,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedTokenBatchSequence,
    NounSelectedBase,
    allocator_peak_bytes,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    CHECKPOINT_ROOT,
    FULL_FINETUNE_LOSS_FORMAT,
    FULL_FINETUNE_STAGE_FORMAT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.lm.checkpoint import (
    BaseCheckpointRef,
    LoadedGptNeoCheckpoint,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainConfig, LmTrainState, init_base_train_state
from apm.lm.training_state_artifact import (
    lm_train_state_checksum,
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)


FullFinetuneProgress = Callable[[str, int, int, float], None]
_SEED_OFFSET = 5


@dataclass(frozen=True, slots=True)
class FullFinetuneStage:
    """Authenticated full-model parameters after one task boundary."""

    stage_index: int
    task_id: str
    task_order: tuple[str, ...]
    checkpoint: BaseCheckpointRef
    prior_parameter_checksum: str
    rng_state: tuple[int, int]
    run_sha256: str
    loss_trace_sha256: str
    peak_allocator_bytes: int

    def __post_init__(self) -> None:
        if self.stage_index <= 0 or len(self.task_order) != self.stage_index:
            raise ValueError("full-finetune stage prefix is invalid")
        if self.task_id != self.task_order[-1]:
            raise ValueError("full-finetune task does not end its stage prefix")
        for value in (
            self.prior_parameter_checksum,
            self.run_sha256,
            self.loss_trace_sha256,
        ):
            _require_sha256(value, "full-finetune stage identity")
        if (
            len(self.rng_state) != 2
            or any(type(value) is not int or not 0 <= value < 2**32 for value in self.rng_state)
            or self.peak_allocator_bytes <= 0
        ):
            raise ValueError("full-finetune stage RNG or peak is invalid")

    @property
    def parameter_checksum(self) -> str:
        """Return the full model's parameter checksum at this boundary."""
        return self.checkpoint.parameter_checksum


def nouns_v2_full_finetune_train_config(
    preset: NounsV2ExperimentPreset,
) -> LmTrainConfig:
    """Return the frozen full-model control optimizer and per-task budget."""
    return LmTrainConfig(
        learning_rate=preset.minimum_learning_rate,
        steps=preset.adapter_updates,
        batch_size=preset.microbatch_size,
        weight_decay=preset.adapter_weight_decay,
        gradient_clip_norm=preset.gradient_clip_norm,
    )


def run_or_resume_nouns_v2_full_finetune(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress: FullFinetuneProgress | None = None,
) -> FullFinetuneStage:
    """Train or exact-resume the sequential full-model control."""
    loaded_base, train_config, run_sha256 = _execution(
        partition,
        preset,
        selected_base,
    )
    root = Path(checkpoint_root) / "full-finetune" / run_sha256
    root.mkdir(parents=True, exist_ok=True)
    work_root = root / "work"
    work_root.mkdir(exist_ok=True)
    print(f"TinyWorlds nouns-v2 full-finetune work directory: {work_root.resolve()}", flush=True)
    completed = _load_completed_prefix(
        root,
        partition,
        preset,
        selected_base,
        loaded_base,
        train_config,
        run_sha256,
    )
    if completed:
        latest = load_gpt_neo_checkpoint(completed[-1].checkpoint)
        current_params = latest.params
        current_rng = jnp.asarray(completed[-1].rng_state, dtype=jnp.uint32)
        prior_checksum = completed[-1].parameter_checksum
    else:
        current_params = loaded_base.params
        current_rng = jax.random.PRNGKey(preset.seed + _SEED_OFFSET)
        prior_checksum = selected_base.reference.parameter_checksum
    checkpoint_interval = max(1, preset.adapter_updates // 4)
    for stage_index in range(len(completed) + 1, len(partition.task_ids) + 1):
        task_id = partition.task_ids[stage_index - 1]
        batches = IndexedTokenBatchSequence(
            partition,
            f"task-{task_id}-train",
            context_length=preset.context_length,
            batch_size=preset.microbatch_size,
            order_namespace=f"adapter-{task_id}",
        )
        work = work_root / f"stage-{stage_index:03d}-{task_id}"
        work.mkdir(parents=True, exist_ok=True)
        task_identity = _task_identity(run_sha256, stage_index, task_id, prior_checksum)
        template = init_base_train_state(current_params, current_rng, train_config)
        resumed = _load_latest_work_state(work, task_identity, template)
        state = template if resumed is None else resumed
        loss_path = work / "losses.jsonl"
        _trim_loss_trace(loss_path, int(state.step), run_sha256, stage_index, task_id)
        started = time.monotonic()
        while int(state.step) < train_config.steps:
            stop_update = min(
                train_config.steps,
                ((int(state.step) // checkpoint_interval) + 1) * checkpoint_interval,
            )
            state, losses = run_full_parameter_updates(
                state,
                batches,
                loaded_base.config,
                train_config,
                stop_update=stop_update,
                progress=_method_progress(progress, f"full-finetune-{task_id}"),
            )
            _append_losses(
                loss_path,
                run_sha256,
                stage_index,
                task_id,
                stop_update - len(losses),
                losses,
            )
            _write_work_state(work, task_identity, state, run_sha256, stage_index, task_id)
        elapsed = time.monotonic() - started
        loss_summary = _validate_loss_trace(
            loss_path,
            train_config.steps,
            run_sha256,
            stage_index,
            task_id,
        )
        peak_bytes = allocator_peak_bytes()
        if peak_bytes > preset.allocator_peak_limit_bytes:
            raise RuntimeError("noun full fine-tuning exceeded the 12 GiB allocator limit")
        target = root / f"stage-{stage_index:03d}-{task_id}"
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root))
        try:
            checkpoint = save_gpt_neo_checkpoint(
                temporary / "checkpoint",
                state.trainable,
                loaded_base.config,
                tokenizer=loaded_base.tokenizer,
                source=loaded_base.source,
            )
            shutil.copy2(loss_path, temporary / "losses.jsonl")
            consumed = batches.consumed_story_ids(train_config.steps)
            stage_core = {
                "base_training_sha256": selected_base.training_sha256,
                "checkpoint_interval": checkpoint_interval,
                "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
                "elapsed_seconds": elapsed,
                "format": FULL_FINETUNE_STAGE_FORMAT,
                "loss_summary": loss_summary,
                "loss_trace_sha256": _file_sha256(temporary / "losses.jsonl"),
                "parameter_checksum": checkpoint.parameter_checksum,
                "partition_sha256": partition.partition_sha256,
                "peak_allocator_bytes": peak_bytes,
                "preset_sha256": preset.config_sha256,
                "prior_parameter_checksum": prior_checksum,
                "rng_state": [int(value) for value in np.asarray(state.rng_key)],
                "run_sha256": run_sha256,
                "source_story_count": batches.story_count,
                "source_window_count": batches.window_count,
                "stage_index": stage_index,
                "task_id": task_id,
                "task_order": list(partition.task_ids[:stage_index]),
                "train_config": _dataclass_record(train_config),
                "unique_consumed_story_count": len(consumed),
                "update_count": train_config.steps,
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
        stage = _load_stage(
            target,
            stage_index,
            prior_checksum,
            partition,
            preset,
            selected_base,
            loaded_base,
            train_config,
            run_sha256,
        )
        completed = completed + (stage,)
        current_params = state.trainable
        current_rng = state.rng_key
        prior_checksum = stage.parameter_checksum
        shutil.rmtree(work)
    if len(completed) != len(partition.task_ids):
        raise RuntimeError("noun full fine-tuning did not complete every stage")
    return completed[-1]


def load_nouns_v2_full_finetune_stages(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> tuple[FullFinetuneStage, ...]:
    """Strict-load all 24 full-model task-boundary checkpoints."""
    loaded_base, train_config, run_sha256 = _execution(
        partition,
        preset,
        selected_base,
    )
    root = Path(checkpoint_root) / "full-finetune" / run_sha256
    stages = _load_completed_prefix(
        root,
        partition,
        preset,
        selected_base,
        loaded_base,
        train_config,
        run_sha256,
    )
    if len(stages) != len(partition.task_ids):
        raise ValueError("noun full-finetune audit requires all canonical stages")
    return stages


def _execution(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
) -> tuple[LoadedGptNeoCheckpoint, LmTrainConfig, str]:
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    train_config = nouns_v2_full_finetune_train_config(preset)
    checkpoint_interval = max(1, preset.adapter_updates // 4)
    run_sha256 = record_sha256(
        {
            "base_parameter_checksum": selected_base.reference.parameter_checksum,
            "base_training_sha256": selected_base.training_sha256,
            "checkpoint_interval": checkpoint_interval,
            "format": FULL_FINETUNE_STAGE_FORMAT,
            "optimizer_reset_each_task": True,
            "partition_sha256": partition.partition_sha256,
            "preset_sha256": preset.config_sha256,
            "seed": preset.seed + _SEED_OFFSET,
            "task_order": list(partition.task_ids),
            "train_config": _dataclass_record(train_config),
            "training_order_namespace": "adapter-{task_id}",
            "trainable": "all_gpt_neo_parameters",
        }
    )
    return loaded, train_config, run_sha256


def _load_completed_prefix(
    root: Path,
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    loaded_base: LoadedGptNeoCheckpoint,
    train_config: LmTrainConfig,
    run_sha256: str,
) -> tuple[FullFinetuneStage, ...]:
    if not root.is_dir():
        return ()
    stage_paths = tuple(sorted(path for path in root.glob("stage-*") if path.is_dir()))
    expected_prefix = tuple(
        root / f"stage-{index:03d}-{task_id}"
        for index, task_id in enumerate(partition.task_ids[: len(stage_paths)], start=1)
    )
    if stage_paths != expected_prefix:
        raise ValueError("full-finetune checkpoint stages are not one canonical prefix")
    stages: list[FullFinetuneStage] = []
    prior = selected_base.reference.parameter_checksum
    for stage_index, path in enumerate(stage_paths, start=1):
        stage = _load_stage(
            path,
            stage_index,
            prior,
            partition,
            preset,
            selected_base,
            loaded_base,
            train_config,
            run_sha256,
        )
        stages.append(stage)
        prior = stage.parameter_checksum
    return tuple(stages)


def _load_stage(
    path: Path,
    stage_index: int,
    prior_checksum: str,
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    loaded_base: LoadedGptNeoCheckpoint,
    train_config: LmTrainConfig,
    run_sha256: str,
) -> FullFinetuneStage:
    expected_entries = {"checkpoint", "losses.jsonl", "stage.json"}
    if path.is_symlink() or {entry.name for entry in path.iterdir()} != expected_entries:
        raise ValueError("full-finetune stage entries are not canonical")
    payload = (path / "stage.json").read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError("full-finetune stage metadata is not canonical JSON")
    supplied = record.pop("stage_sha256", None)
    task_id = partition.task_ids[stage_index - 1]
    batches = IndexedTokenBatchSequence(
        partition,
        f"task-{task_id}-train",
        context_length=preset.context_length,
        batch_size=preset.microbatch_size,
        order_namespace=f"adapter-{task_id}",
    )
    expected_fields = {
        "base_training_sha256",
        "checkpoint_interval",
        "checkpoint_manifest_sha256",
        "elapsed_seconds",
        "format",
        "loss_summary",
        "loss_trace_sha256",
        "parameter_checksum",
        "partition_sha256",
        "peak_allocator_bytes",
        "preset_sha256",
        "prior_parameter_checksum",
        "rng_state",
        "run_sha256",
        "source_story_count",
        "source_window_count",
        "stage_index",
        "task_id",
        "task_order",
        "train_config",
        "unique_consumed_story_count",
        "update_count",
    }
    loss_summary = _validate_loss_trace(
        path / "losses.jsonl",
        train_config.steps,
        run_sha256,
        stage_index,
        task_id,
    )
    checkpoint = load_gpt_neo_checkpoint(path / "checkpoint")
    rng_state = record.get("rng_state")
    numeric = tuple(
        record.get(name)
        for name in (
            "checkpoint_interval",
            "peak_allocator_bytes",
            "source_story_count",
            "source_window_count",
            "unique_consumed_story_count",
            "update_count",
        )
    )
    if (
        set(record) != expected_fields
        or supplied != record_sha256(record)
        or record.get("format") != FULL_FINETUNE_STAGE_FORMAT
        or record.get("run_sha256") != run_sha256
        or record.get("partition_sha256") != partition.partition_sha256
        or record.get("preset_sha256") != preset.config_sha256
        or record.get("base_training_sha256") != selected_base.training_sha256
        or record.get("stage_index") != stage_index
        or record.get("task_id") != task_id
        or record.get("task_order") != list(partition.task_ids[:stage_index])
        or record.get("prior_parameter_checksum") != prior_checksum
        or record.get("train_config") != _dataclass_record(train_config)
        or record.get("checkpoint_interval") != max(1, preset.adapter_updates // 4)
        or record.get("update_count") != train_config.steps
        or record.get("source_story_count") != batches.story_count
        or record.get("source_window_count") != batches.window_count
        or record.get("unique_consumed_story_count")
        != len(batches.consumed_story_ids(train_config.steps))
        or record.get("loss_summary") != loss_summary
        or record.get("loss_trace_sha256") != _file_sha256(path / "losses.jsonl")
        or record.get("checkpoint_manifest_sha256") != checkpoint.reference.manifest_sha256
        or record.get("parameter_checksum") != checkpoint.reference.parameter_checksum
        or checkpoint.config != loaded_base.config
        or checkpoint.tokenizer != loaded_base.tokenizer
        or checkpoint.source != loaded_base.source
        or type(rng_state) is not list
        or len(rng_state) != 2
        or any(type(value) is not int or not 0 <= value < 2**32 for value in rng_state)
        or any(type(value) is not int or value <= 0 for value in numeric)
        or int(record["peak_allocator_bytes"]) > preset.allocator_peak_limit_bytes
        or type(record.get("elapsed_seconds")) not in (int, float)
        or not math.isfinite(float(record["elapsed_seconds"]))
        or float(record["elapsed_seconds"]) <= 0.0
    ):
        raise ValueError("full-finetune stage metadata or bindings changed")
    return FullFinetuneStage(
        stage_index=stage_index,
        task_id=task_id,
        task_order=partition.task_ids[:stage_index],
        checkpoint=checkpoint.reference,
        prior_parameter_checksum=prior_checksum,
        rng_state=(int(rng_state[0]), int(rng_state[1])),
        run_sha256=run_sha256,
        loss_trace_sha256=str(record["loss_trace_sha256"]),
        peak_allocator_bytes=int(record["peak_allocator_bytes"]),
    )


def _task_identity(
    run_sha256: str,
    stage_index: int,
    task_id: str,
    prior_checksum: str,
) -> str:
    return record_sha256(
        {
            "prior_parameter_checksum": prior_checksum,
            "run_sha256": run_sha256,
            "stage_index": stage_index,
            "task_id": task_id,
        }
    )


def _load_latest_work_state(
    work: Path,
    task_identity: str,
    template: LmTrainState[GptNeoParams],
) -> LmTrainState[GptNeoParams] | None:
    candidates = tuple(sorted(path for path in work.glob("update-*") if path.is_dir()))
    if not candidates:
        return None
    latest = candidates[-1]
    record = _load_canonical_record(latest / "resume.json", "full-finetune resume")
    supplied = record.pop("resume_sha256", None)
    if (
        supplied != record_sha256(record)
        or record.get("format") != FULL_FINETUNE_STAGE_FORMAT
        or record.get("task_identity") != task_identity
        or record.get("update") != int(latest.name.split("-", 1)[1])
    ):
        raise ValueError("full-finetune work-state identity changed")
    state = load_lm_train_state_artifact(
        latest / "state",
        task_identity,
        (template,),
    )[0]
    if (
        int(state.step) != record["update"]
        or lm_train_state_checksum(state) != record.get("state_sha256")
    ):
        raise ValueError("full-finetune work state and resume record differ")
    return state


def _write_work_state(
    work: Path,
    task_identity: str,
    state: LmTrainState[GptNeoParams],
    run_sha256: str,
    stage_index: int,
    task_id: str,
) -> None:
    target = work / f"update-{int(state.step):04d}"
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=work))
        try:
            write_lm_train_state_artifact(temporary / "state", task_identity, (state,))
            core = {
                "format": FULL_FINETUNE_STAGE_FORMAT,
                "run_sha256": run_sha256,
                "stage_index": stage_index,
                "state_sha256": lm_train_state_checksum(state),
                "task_id": task_id,
                "task_identity": task_identity,
                "update": int(state.step),
            }
            _atomic_write(
                temporary / "resume.json",
                canonical_json_bytes({**core, "resume_sha256": record_sha256(core)}),
            )
            os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    for prior in tuple(path for path in work.glob("update-*") if path != target):
        shutil.rmtree(prior)


def _append_losses(
    path: Path,
    run_sha256: str,
    stage_index: int,
    task_id: str,
    start_update: int,
    losses: tuple[float, ...],
) -> None:
    payload = b"".join(
        canonical_json_bytes(
            {
                **core,
                "loss_sha256": record_sha256(core),
            }
        )
        for update, loss in enumerate(losses, start=start_update + 1)
        for core in (
            {
                "format": FULL_FINETUNE_LOSS_FORMAT,
                "loss": loss,
                "run_sha256": run_sha256,
                "stage_index": stage_index,
                "task_id": task_id,
                "update": update,
            },
        )
    )
    with path.open("ab") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _trim_loss_trace(
    path: Path,
    update_count: int,
    run_sha256: str,
    stage_index: int,
    task_id: str,
) -> None:
    if not path.is_file():
        if update_count != 0:
            raise ValueError("full-finetune work state has no loss trace")
        return
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < update_count:
        raise ValueError("full-finetune loss trace is behind its work state")
    kept = lines[:update_count]
    _validate_loss_lines(kept, run_sha256, stage_index, task_id)
    if len(lines) != update_count:
        _atomic_write(path, b"".join(kept))


def _validate_loss_trace(
    path: Path,
    expected_count: int,
    run_sha256: str,
    stage_index: int,
    task_id: str,
) -> dict[str, object]:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) != expected_count:
        raise ValueError("full-finetune loss trace length changed")
    losses = _validate_loss_lines(lines, run_sha256, stage_index, task_id)
    return {
        "final": losses[-1],
        "first": losses[0],
        "maximum": max(losses),
        "minimum": min(losses),
        "update_count": len(losses),
    }


def _validate_loss_lines(
    lines: list[bytes],
    run_sha256: str,
    stage_index: int,
    task_id: str,
) -> tuple[float, ...]:
    losses: list[float] = []
    for expected_update, line in enumerate(lines, start=1):
        if not line.endswith(b"\n"):
            raise ValueError("full-finetune loss trace has an interrupted tail")
        row = json.loads(line)
        if type(row) is not dict or canonical_json_bytes(row) != line:
            raise ValueError("full-finetune loss trace is not canonical JSONL")
        supplied = row.pop("loss_sha256", None)
        loss = row.get("loss")
        if (
            set(row) != {"format", "loss", "run_sha256", "stage_index", "task_id", "update"}
            or supplied != record_sha256(row)
            or row.get("format") != FULL_FINETUNE_LOSS_FORMAT
            or row.get("run_sha256") != run_sha256
            or row.get("stage_index") != stage_index
            or row.get("task_id") != task_id
            or row.get("update") != expected_update
            or type(loss) not in (int, float)
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise ValueError("full-finetune loss trace row changed")
        losses.append(float(loss))
    return tuple(losses)


def _method_progress(
    progress: FullFinetuneProgress | None,
    phase: str,
) -> Callable[[int, float, int], None] | None:
    if progress is None:
        return None
    return lambda completed, loss, total: progress(phase, completed, total, loss)


def _dataclass_record(value: object) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _load_canonical_record(path: Path, label: str) -> dict[str, object]:
    payload = path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return record


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
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
    "FullFinetuneStage",
    "load_nouns_v2_full_finetune_stages",
    "nouns_v2_full_finetune_train_config",
    "run_or_resume_nouns_v2_full_finetune",
]
