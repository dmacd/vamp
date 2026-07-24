"""Seed-zero semantic-v6 training with version-native resume identities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile

import jax

from apm.data.text.tinyworlds_p import training as archive_training
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.partition_runtime import semantic_runtime_view
from apm.data.text.tinyworlds_p_semantic.training import StreamingTrainingConfig
from apm.data.text.tinyworlds_p_semantic.v6_batching import (
    count_v6_partition_microbatches,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6_SEMANTIC_TRAINING_PRESET,
    V6SemanticPartitionArtifact,
    V6SemanticTrainingPreset,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainState
from apm.lm.training_state_artifact import lm_train_state_checksum


V6_RESUME_FORMAT = "tinyworlds-p-semantic-v6-training-resume"
_CHECKPOINT_NAME = re.compile(
    r"(?:update|interrupted)-[0-9]{9}|epoch-[0-9]{2}-update-[0-9]{9}"
)
TrainingCursor = archive_training.TrainingCursor
TrainingCheckpoint = archive_training.TrainingCheckpoint


@dataclass(frozen=True)
class V6StreamingTrainingConfig(StreamingTrainingConfig):
    """Executable settings derived from the frozen semantic-v6 preset."""

    @classmethod
    def from_preset(
        cls,
        preset: V6SemanticTrainingPreset = V6_SEMANTIC_TRAINING_PRESET,
    ) -> V6StreamingTrainingConfig:
        """Expand the registered v6 preset into the shared optimizer shape."""
        return cls(
            model_config=preset.model_config,
            epochs=preset.epochs,
            calibration_epochs=preset.calibration_epochs,
            context_length=preset.context_length,
            microbatch_size=preset.microbatch_size,
            accumulation_microbatches=preset.accumulation_microbatches,
            maximum_learning_rate=preset.maximum_learning_rate,
            minimum_learning_rate=preset.minimum_learning_rate,
            warmup_fraction=preset.warmup_fraction,
            adam_beta1=preset.adam_beta1,
            adam_beta2=preset.adam_beta2,
            adam_epsilon=preset.adam_epsilon,
            weight_decay=preset.weight_decay,
            gradient_clip_norm=preset.gradient_clip_norm,
            parameter_seed=preset.parameter_seed,
            state_interval_updates=preset.state_interval_updates,
            allocator_peak_limit_bytes=preset.allocator_peak_limit_bytes,
        )


@dataclass(frozen=True, slots=True)
class V6StreamingTrainingResult:
    """Semantic-v6 state, cursor, checkpoints, trace, and exact update plan."""

    state: LmTrainState[GptNeoParams]
    cursor: TrainingCursor
    checkpoints: tuple[TrainingCheckpoint, ...]
    trace_path: Path
    training_sha256: str
    planned_optimizer_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_path", Path(self.trace_path))
        if not self.trace_path.is_file():
            raise FileNotFoundError(self.trace_path)


def run_v6_streaming_base_training(
    artifact: V6SemanticPartitionArtifact,
    working_directory: str | Path,
    config: V6StreamingTrainingConfig | None = None,
    *,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    stop_after_update: int | None = None,
    progress: Callable[[TrainingCursor, float, int], None] | None = None,
) -> V6StreamingTrainingResult:
    """Train or resume only from a strictly authenticated semantic-v6 partition."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 training requires its strict partition")
    effective = config or V6StreamingTrainingConfig.from_preset()
    if type(effective) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 training requires V6StreamingTrainingConfig")
    result = archive_training.run_streaming_base_training(
        semantic_runtime_view(artifact, V6SemanticPartitionArtifact),
        working_directory,
        effective.archive_config,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
        stop_after_update=stop_after_update,
        progress=progress,
        identity_namespace=V6_BENCHMARK_ID,
        resume_format=V6_RESUME_FORMAT,
    )
    return V6StreamingTrainingResult(
        state=result.state,
        cursor=result.cursor,
        checkpoints=result.checkpoints,
        trace_path=result.trace_path,
        training_sha256=result.training_sha256,
        planned_optimizer_updates=result.planned_optimizer_updates,
    )


def init_v6_streaming_train_state(
    config: V6StreamingTrainingConfig,
    planned_optimizer_updates: int,
) -> LmTrainState[GptNeoParams]:
    """Initialize the semantic-v6 model, optimizer, and random stream."""
    if type(config) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 state requires V6StreamingTrainingConfig")
    return archive_training.init_streaming_train_state(
        config.archive_config,
        planned_optimizer_updates,
    )


def load_v6_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    """Load only a complete semantic-v6 resume checkpoint."""
    return archive_training.load_streaming_checkpoint(
        directory,
        training_sha256,
        template,
        resume_format=V6_RESUME_FORMAT,
    )


def load_latest_v6_streaming_result(
    artifact: V6SemanticPartitionArtifact,
    working_directory: str | Path,
    config: V6StreamingTrainingConfig,
) -> V6StreamingTrainingResult | None:
    """Load the newest strict checkpoint and trim its uncheckpointed trace tail."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 resume discovery requires its strict partition")
    if type(config) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 resume discovery requires its strict config")
    working = Path(working_directory)
    states = working / "states"
    if not states.is_dir():
        return None
    candidates = tuple(sorted(states.iterdir()))
    if any(
        path.is_symlink()
        or not path.is_dir()
        or _CHECKPOINT_NAME.fullmatch(path.name) is None
        for path in candidates
    ):
        raise ValueError("semantic-v6 resume checkpoint names changed")
    if not candidates:
        return None
    training_sha256, planned_updates = _training_identity(artifact, config)
    manifests = tuple(
        (path, _resume_cursor(path, training_sha256)) for path in candidates
    )
    selected_path, selected_cursor = max(
        manifests,
        key=lambda item: (
            item[1].optimizer_update,
            item[1].epoch,
            item[1].block,
            item[1].microbatch,
            item[0].name.startswith("epoch-"),
        ),
    )
    template = init_v6_streaming_train_state(config, planned_updates)
    state, loaded_cursor = load_v6_streaming_checkpoint(
        selected_path,
        training_sha256,
        template,
    )
    if loaded_cursor != selected_cursor:
        raise ValueError("semantic-v6 resume cursor changed during strict loading")
    trace_path = working / "progress.jsonl"
    _truncate_trace_at_checkpoint(trace_path, selected_cursor.optimizer_update)
    checkpoint = TrainingCheckpoint(
        selected_path.resolve(),
        selected_cursor,
        lm_train_state_checksum(state),
    )
    return V6StreamingTrainingResult(
        state=state,
        cursor=selected_cursor,
        checkpoints=(checkpoint,),
        trace_path=trace_path,
        training_sha256=training_sha256,
        planned_optimizer_updates=planned_updates,
    )


def write_v6_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    state: LmTrainState[GptNeoParams],
    cursor: TrainingCursor,
) -> TrainingCheckpoint:
    """Write an immutable semantic-v6 resume checkpoint."""
    return archive_training.write_streaming_checkpoint(
        directory,
        training_sha256,
        state,
        cursor,
        resume_format=V6_RESUME_FORMAT,
    )


def v6_cosine_learning_rate(
    update: int | jax.Array,
    total_updates: int,
    config: V6StreamingTrainingConfig,
) -> jax.Array:
    """Apply the registered warmup and cosine schedule."""
    return archive_training.cosine_learning_rate(
        update,
        total_updates,
        config.archive_config,
    )


def _training_identity(
    artifact: V6SemanticPartitionArtifact,
    config: V6StreamingTrainingConfig,
) -> tuple[str, int]:
    updates_per_epoch = math.ceil(
        count_v6_partition_microbatches(artifact, "base/train")
        / config.accumulation_microbatches
    )
    identity = record_sha256(
        {
            "identity_namespace": V6_BENCHMARK_ID,
            "partition_sha256": artifact.partition_sha256,
            "training": config.archive_config.as_record(),
            "updates_per_epoch": updates_per_epoch,
        }
    )
    return identity, updates_per_epoch * config.epochs


def _resume_cursor(directory: Path, training_sha256: str) -> TrainingCursor:
    path = directory / "resume.json"
    raw = path.read_bytes()
    try:
        record = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic-v6 resume manifest: {path}") from error
    if (
        type(record) is not dict
        or canonical_json_bytes(record) != raw
        or set(record)
        != {"cursor", "format", "state_sha256", "training_sha256", "version"}
        or record.get("format") != V6_RESUME_FORMAT
        or record.get("training_sha256") != training_sha256
        or record.get("version") != 1
    ):
        raise ValueError(f"semantic-v6 resume manifest changed: {path}")
    cursor = record.get("cursor")
    if type(cursor) is not dict or set(cursor) != {
        "block",
        "epoch",
        "microbatch",
        "optimizer_update",
        "schedule_position",
    }:
        raise ValueError(f"semantic-v6 resume cursor changed: {path}")
    return TrainingCursor(
        epoch=_nonnegative_integer(cursor, "epoch"),
        block=_nonnegative_integer(cursor, "block"),
        microbatch=_nonnegative_integer(cursor, "microbatch"),
        optimizer_update=_nonnegative_integer(cursor, "optimizer_update"),
        schedule_position=_nonnegative_integer(cursor, "schedule_position"),
    )


def _truncate_trace_at_checkpoint(path: Path, optimizer_update: int) -> None:
    retained: list[bytes] = []
    previous_update = 0
    with path.open("rb") as source:
        for line in source:
            if previous_update == optimizer_update:
                break
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("semantic-v6 training trace changed") from error
            update = (
                _nonnegative_integer(record, "optimizer_update")
                if type(record) is dict and canonical_json_bytes(record) == line
                else -1
            )
            if update != previous_update + 1 or update > optimizer_update:
                raise ValueError("semantic-v6 training trace order changed")
            retained.append(line)
            previous_update = update
    if previous_update != optimizer_update:
        raise ValueError("semantic-v6 checkpoint is ahead of its training trace")
    payload = b"".join(retained)
    if path.read_bytes() == payload:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="resume-trace-",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _nonnegative_integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 resume field {field!r} must be nonnegative")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "TrainingCheckpoint",
    "TrainingCursor",
    "V6_RESUME_FORMAT",
    "V6StreamingTrainingConfig",
    "V6StreamingTrainingResult",
    "init_v6_streaming_train_state",
    "load_latest_v6_streaming_result",
    "load_v6_streaming_checkpoint",
    "run_v6_streaming_base_training",
    "v6_cosine_learning_rate",
    "write_v6_streaming_checkpoint",
]
