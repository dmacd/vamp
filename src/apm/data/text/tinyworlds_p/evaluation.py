"""Streaming validation and one-shot sealed-test evaluation for TinyWorlds-P."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from apm.data.text.tinyworlds_p.batching import (
    count_partition_microbatches,
    iter_partition_batches,
)
from apm.data.text.tinyworlds_p.contracts import PartitionArtifact, WORLD_LABELS
from apm.data.text.tinyworlds_p.training import (
    EpochValidation,
    WorldGap,
    allocator_peak_bytes,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.config import GptNeoConfig
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch


SplitEvaluationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class SplitNll:
    """Total active tokens and their normalized NLL for one persisted split."""

    split: str
    active_tokens: int
    nll: float

    def __post_init__(self) -> None:
        if type(self.split) is not str or not self.split:
            raise ValueError("evaluated split must be nonempty")
        if type(self.active_tokens) is not int or self.active_tokens <= 0:
            raise ValueError("evaluated split must contain active tokens")
        if not math.isfinite(self.nll) or self.nll < 0.0:
            raise ValueError("evaluated NLL must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SealedTestResults:
    """The single held-in/world/control test opening for a selected checkpoint."""

    selected_epoch: int
    held_in: SplitNll
    worlds: tuple[SplitNll, ...]
    controls: tuple[SplitNll, ...]

    def __post_init__(self) -> None:
        if type(self.selected_epoch) is not int or not 2 <= self.selected_epoch <= 5:
            raise ValueError("selected test epoch must lie in 2-5")
        expected_worlds = tuple(f"world/{world}/test" for world in WORLD_LABELS)
        expected_controls = tuple(f"control/{world}/test" for world in WORLD_LABELS)
        if tuple(result.split for result in self.worlds) != expected_worlds:
            raise ValueError("sealed test worlds are incomplete or out of order")
        if tuple(result.split for result in self.controls) != expected_controls:
            raise ValueError("sealed test controls are incomplete or out of order")


def evaluate_partition_split(
    params: GptNeoParams,
    artifact: PartitionArtifact,
    split: str,
    model_config: GptNeoConfig | None = None,
    *,
    progress: SplitEvaluationProgress | None = None,
) -> SplitNll:
    """Stream total float32 NLL and normalize once across all active tokens."""
    effective_model_config = model_config or _model_config_for_artifact(artifact)
    if effective_model_config.vocab_size != artifact.tokenizer_identity.vocab_size:
        raise ValueError("evaluation model vocabulary differs from partition tokenizer")

    def evaluate_batch(batch: TokenBatch) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            params,
            effective_model_config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
        )
        mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask), jnp.sum(mask)

    compiled = jax.jit(evaluate_batch)
    total_loss = 0.0
    active_tokens = 0
    planned_batches = (
        count_partition_microbatches(artifact, split) if progress is not None else 0
    )
    for completed_batches, batch in enumerate(
        iter_partition_batches(artifact, split, epoch=0),
        start=1,
    ):
        loss_sum, token_count = compiled(batch)
        total_loss += float(loss_sum)
        active_tokens += int(token_count)
        if progress is not None:
            progress(split, completed_batches, planned_batches)
    if active_tokens == 0:
        raise ValueError(f"evaluation split contains no active tokens: {split}")
    return SplitNll(split, active_tokens, total_loss / active_tokens)


def evaluate_epoch_validation(
    params: GptNeoParams,
    artifact: PartitionArtifact,
    epoch: int,
    model_config: GptNeoConfig | None = None,
    *,
    progress: SplitEvaluationProgress | None = None,
) -> EpochValidation:
    """Evaluate held-in validation and all five world/control matched gaps."""
    held_in = evaluate_partition_split(
        params,
        artifact,
        "base/validation",
        model_config,
        progress=progress,
    )
    world_results = tuple(
        (
            evaluate_partition_split(
                params,
                artifact,
                f"world/{world}/validation",
                model_config,
                progress=progress,
            ),
            evaluate_partition_split(
                params,
                artifact,
                f"control/{world}/validation",
                model_config,
                progress=progress,
            ),
        )
        for world in WORLD_LABELS
    )
    return EpochValidation(
        epoch=epoch,
        held_in_nll=held_in.nll,
        world_gaps=tuple(
            WorldGap(world, world_result.nll, control_result.nll)
            for world, (world_result, control_result) in zip(
                WORLD_LABELS,
                world_results,
                strict=True,
            )
        ),
        allocator_peak_bytes=allocator_peak_bytes(),
    )


def evaluate_sealed_test_once(
    params: GptNeoParams,
    artifact: PartitionArtifact,
    selected_epoch: int,
    model_config: GptNeoConfig | None = None,
    *,
    progress: SplitEvaluationProgress | None = None,
) -> SealedTestResults:
    """Open every sealed test split once for the already selected checkpoint."""
    return SealedTestResults(
        selected_epoch=selected_epoch,
        held_in=evaluate_partition_split(
            params,
            artifact,
            "base/test",
            model_config,
            progress=progress,
        ),
        worlds=tuple(
            evaluate_partition_split(
                params,
                artifact,
                f"world/{world}/test",
                model_config,
                progress=progress,
            )
            for world in WORLD_LABELS
        ),
        controls=tuple(
            evaluate_partition_split(
                params,
                artifact,
                f"control/{world}/test",
                model_config,
                progress=progress,
            )
            for world in WORLD_LABELS
        ),
    )


def _model_config_for_artifact(artifact: PartitionArtifact) -> GptNeoConfig:
    # Partition artifacts intentionally bind data, not architecture.  Validation
    # uses the v1 architecture, while tiny tests monkeypatch this private boundary.
    from apm.data.text.tinyworlds_p.contracts import BASE_TRAINING_PRESET

    config = BASE_TRAINING_PRESET.model_config
    if config.vocab_size != artifact.tokenizer_identity.vocab_size:
        raise ValueError("v1 evaluation model vocabulary differs from partition tokenizer")
    return config


__all__ = [
    "SealedTestResults",
    "SplitNll",
    "evaluate_epoch_validation",
    "evaluate_partition_split",
    "evaluate_sealed_test_once",
]
