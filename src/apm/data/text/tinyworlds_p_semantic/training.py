"""Seed-zero semantic-v1 training with benchmark-specific resume identities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax

from apm.data.text.tinyworlds_p import training as archive_training
from apm.data.text.tinyworlds_p_semantic.batching import _runtime_view
from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    SEMANTIC_TRAINING_PRESET,
    SemanticPartitionArtifact,
    SemanticTrainingPreset,
)
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainState


RESUME_FORMAT = "tinyworlds-p-semantic-training-resume"
TrainingCursor = archive_training.TrainingCursor
TrainingCheckpoint = archive_training.TrainingCheckpoint


@dataclass(frozen=True, slots=True)
class StreamingTrainingConfig:
    """Executable semantic trainer settings derived from the frozen preset."""

    model_config: GptNeoConfig
    epochs: int
    calibration_epochs: int
    context_length: int
    microbatch_size: int
    accumulation_microbatches: int
    maximum_learning_rate: float
    minimum_learning_rate: float
    warmup_fraction: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    parameter_seed: int
    state_interval_updates: int
    allocator_peak_limit_bytes: int

    def __post_init__(self) -> None:
        self.archive_config

    @classmethod
    def from_preset(
        cls,
        preset: SemanticTrainingPreset = SEMANTIC_TRAINING_PRESET,
    ) -> StreamingTrainingConfig:
        """Expand the semantic-v1 preset into an executable configuration."""
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

    @property
    def archive_config(self) -> archive_training.StreamingTrainingConfig:
        """Return the source-neutral optimizer core consumed by the shared loop."""
        return archive_training.StreamingTrainingConfig(
            model_config=self.model_config,
            epochs=self.epochs,
            calibration_epochs=self.calibration_epochs,
            context_length=self.context_length,
            microbatch_size=self.microbatch_size,
            accumulation_microbatches=self.accumulation_microbatches,
            maximum_learning_rate=self.maximum_learning_rate,
            minimum_learning_rate=self.minimum_learning_rate,
            warmup_fraction=self.warmup_fraction,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            parameter_seed=self.parameter_seed,
            state_interval_updates=self.state_interval_updates,
            allocator_peak_limit_bytes=self.allocator_peak_limit_bytes,
        )

    def as_record(self) -> dict[str, object]:
        """Return the complete behavior-changing training record."""
        return self.archive_config.as_record()


@dataclass(frozen=True, slots=True)
class StreamingTrainingResult:
    """Semantic training state, cursor, immutable checkpoints, and trace."""

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


def run_streaming_base_training(
    artifact: SemanticPartitionArtifact,
    working_directory: str | Path,
    config: StreamingTrainingConfig | None = None,
    *,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    stop_after_update: int | None = None,
    progress: Callable[[TrainingCursor, float, int], None] | None = None,
) -> StreamingTrainingResult:
    """Train or resume semantic-v1 while rejecting archive-v1 resume manifests."""
    if type(artifact) is not SemanticPartitionArtifact:
        raise TypeError("semantic training requires SemanticPartitionArtifact")
    effective = config or StreamingTrainingConfig.from_preset()
    result = archive_training.run_streaming_base_training(
        _runtime_view(artifact),
        working_directory,
        effective.archive_config,
        resume_from=resume_from,
        stop_after_epoch=stop_after_epoch,
        stop_after_update=stop_after_update,
        progress=progress,
        identity_namespace=BENCHMARK_ID,
        resume_format=RESUME_FORMAT,
    )
    return StreamingTrainingResult(
        state=result.state,
        cursor=result.cursor,
        checkpoints=result.checkpoints,
        trace_path=result.trace_path,
        training_sha256=result.training_sha256,
        planned_optimizer_updates=result.planned_optimizer_updates,
    )


def init_streaming_train_state(
    config: StreamingTrainingConfig,
    planned_optimizer_updates: int,
) -> LmTrainState[GptNeoParams]:
    """Initialize semantic model/optimizer/RNG state from seed zero."""
    return archive_training.init_streaming_train_state(
        config.archive_config,
        planned_optimizer_updates,
    )


def load_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    """Load only a semantic-v1 resume identity and complete state."""
    return archive_training.load_streaming_checkpoint(
        directory,
        training_sha256,
        template,
        resume_format=RESUME_FORMAT,
    )


def write_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    state: LmTrainState[GptNeoParams],
    cursor: TrainingCursor,
) -> TrainingCheckpoint:
    """Write a complete checkpoint under the semantic-v1 resume contract."""
    return archive_training.write_streaming_checkpoint(
        directory,
        training_sha256,
        state,
        cursor,
        resume_format=RESUME_FORMAT,
    )


def cosine_learning_rate(
    update: int | jax.Array,
    total_updates: int,
    config: StreamingTrainingConfig,
) -> jax.Array:
    """Apply the unchanged warmup and cosine schedule."""
    return archive_training.cosine_learning_rate(
        update,
        total_updates,
        config.archive_config,
    )


def allocator_peak_bytes() -> int:
    """Return the largest JAX allocator peak currently reported."""
    return archive_training.allocator_peak_bytes()


__all__ = [
    "RESUME_FORMAT",
    "StreamingTrainingConfig",
    "StreamingTrainingResult",
    "TrainingCheckpoint",
    "TrainingCursor",
    "allocator_peak_bytes",
    "cosine_learning_rate",
    "init_streaming_train_state",
    "load_streaming_checkpoint",
    "run_streaming_base_training",
    "write_streaming_checkpoint",
]
