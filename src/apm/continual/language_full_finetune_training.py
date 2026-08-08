"""Pure resumable updates for sequential full-parameter language fine-tuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import jax

from apm.continual.language_baseline_training import HomogeneousTokenBatchSequence
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig, LmTrainState, base_train_step


TrainingProgress = Callable[[int, float, int], None]
FullParameterStep = Callable[
    [LmTrainState[GptNeoParams], TokenBatch],
    tuple[LmTrainState[GptNeoParams], jax.Array],
]


def run_full_parameter_updates(
    state: LmTrainState[GptNeoParams],
    batches: Sequence[TokenBatch],
    model_config: GptNeoConfig,
    train_config: LmTrainConfig,
    *,
    stop_update: int | None = None,
    progress: TrainingProgress | None = None,
) -> tuple[LmTrainState[GptNeoParams], tuple[float, ...]]:
    """Advance one task-local full-model state to an absolute update boundary."""
    if not isinstance(state, LmTrainState):
        raise TypeError("full fine-tuning state must be an LmTrainState")
    _validate_batches(batches, train_config)
    start_update = int(state.step)
    target_update = train_config.steps if stop_update is None else stop_update
    if (
        type(target_update) is not int
        or not start_update <= target_update <= train_config.steps
    ):
        raise ValueError("full fine-tuning stop_update is outside the task budget")
    compiled_step = _compiled_base_step(model_config, train_config)
    current = state
    losses: list[float] = []
    for update in range(start_update, target_update):
        current, loss = compiled_step(current, batches[update % len(batches)])
        scalar_loss = float(loss)
        losses.append(scalar_loss)
        if progress is not None:
            progress(update + 1, scalar_loss, train_config.steps)
    return current, tuple(losses)


@lru_cache(maxsize=None)
def _compiled_base_step(
    model_config: GptNeoConfig,
    train_config: LmTrainConfig,
) -> FullParameterStep:
    return jax.jit(
        lambda current_state, batch: base_train_step(
            current_state,
            batch,
            model_config,
            train_config,
        )
    )


def _validate_batches(
    batches: Sequence[TokenBatch],
    train_config: LmTrainConfig,
) -> None:
    if not batches:
        raise ValueError("full fine-tuning requires at least one training batch")
    if isinstance(batches, HomogeneousTokenBatchSequence):
        shapes = {(batches.batch_size, batches.sequence_width)}
    else:
        shapes = {tuple(batch.input_ids.shape) for batch in batches}
    if any(rows != train_config.batch_size for rows, _ in shapes):
        raise ValueError("full fine-tuning batches must match the configured row count")
    if len({width for _, width in shapes}) != 1:
        raise ValueError("full fine-tuning batches must share one sequence width")


__all__ = ["TrainingProgress", "run_full_parameter_updates"]
