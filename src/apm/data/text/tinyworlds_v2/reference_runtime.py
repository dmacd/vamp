"""TinyStories checkpoint scoring for the TinyWorlds-v2 reference profile.

The pure batching layer in this module is intentionally independent of JAX.
Default tests can therefore exercise every token boundary and aggregation rule
without loading a checkpoint or selecting an accelerator.  The production
entry point imports JAX lazily and requires the pinned TinyStories-8M model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class NllStory:
    """One stable story identity and its unmodified natural-language text."""

    record_id: str
    text: str

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("NLL story record_id must be nonempty")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("NLL story text must be nonempty")


@dataclass(frozen=True, slots=True)
class StoryTokenWindow:
    """One fixed-width causal window whose targets are scored exactly once."""

    story_index: int
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    active_tokens: int

    def __post_init__(self) -> None:
        if type(self.story_index) is not int or self.story_index < 0:
            raise ValueError("story window index must be nonnegative")
        if (
            type(self.input_ids) is not tuple
            or type(self.target_ids) is not tuple
            or not self.input_ids
            or len(self.input_ids) != len(self.target_ids)
        ):
            raise ValueError("story window input and target IDs must have equal width")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.input_ids):
            raise ValueError("story window input IDs must be nonnegative integers")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.target_ids):
            raise ValueError("story window target IDs must be nonnegative integers")
        if (
            type(self.active_tokens) is not int
            or self.active_tokens <= 0
            or self.active_tokens > len(self.input_ids)
        ):
            raise ValueError("story window active token count is outside its width")


@dataclass(frozen=True, slots=True)
class PerStoryNll:
    """Total and normalized causal loss for one complete reference story."""

    record_id: str
    total_nll: float
    token_count: int
    normalized_nll: float

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("per-story NLL record_id must be nonempty")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("per-story NLL token_count must be positive")
        if not np.isfinite(self.total_nll) or self.total_nll < 0.0:
            raise ValueError("per-story total NLL must be finite and nonnegative")
        if not np.isfinite(self.normalized_nll) or self.normalized_nll < 0.0:
            raise ValueError("per-story normalized NLL must be finite and nonnegative")
        expected = self.total_nll / self.token_count
        if not np.isclose(self.normalized_nll, expected, rtol=1e-7, atol=1e-8):
            raise ValueError("normalized NLL does not match total NLL / token_count")


@dataclass(frozen=True, slots=True)
class ReferenceNllRun:
    """Pinned runtime identity and ordered per-story checkpoint scores."""

    checkpoint_parameter_checksum: str
    checkpoint_manifest_sha256: str
    tokenizer_sha256: str
    jax_platform: str
    device_kind: str
    sequence_length: int
    batch_size: int
    scores: tuple[PerStoryNll, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("checkpoint parameter checksum", self.checkpoint_parameter_checksum),
            ("checkpoint manifest SHA-256", self.checkpoint_manifest_sha256),
            ("tokenizer SHA-256", self.tokenizer_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{label} must be lowercase SHA-256")
        if type(self.jax_platform) is not str or not self.jax_platform:
            raise ValueError("JAX platform must be nonempty")
        if type(self.device_kind) is not str or not self.device_kind:
            raise ValueError("JAX device kind must be nonempty")
        if type(self.sequence_length) is not int or self.sequence_length <= 0:
            raise ValueError("NLL sequence_length must be positive")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("NLL batch_size must be positive")
        if type(self.scores) is not tuple or not self.scores:
            raise ValueError("reference NLL run must contain scores")
        if len({score.record_id for score in self.scores}) != len(self.scores):
            raise ValueError("reference NLL score IDs must be unique")


class TextEncoder(Protocol):
    """Structural tokenizer boundary used by the pure window builder."""

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        """Return token IDs for one story."""


class WindowBatchScorer(Protocol):
    """Return per-row total NLL and active-token counts for one fixed batch."""

    def __call__(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray,
        attention_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score one batch with arrays of shape ``[batch, sequence]``."""


def build_story_token_windows(
    stories: Sequence[NllStory],
    tokenizer: TextEncoder,
    *,
    sequence_length: int = 256,
    pad_token_id: int,
) -> tuple[StoryTokenWindow, ...]:
    """Tokenize stories and cover every next-token target exactly once.

    Windows do not cross story boundaries.  For stories longer than one
    window, the final target of the preceding window becomes the first input
    of the next one, preserving coverage without double-counting a target.
    """
    if not stories:
        raise ValueError("reference NLL requires at least one story")
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("NLL sequence_length must be positive")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("NLL pad_token_id must be nonnegative")

    windows: list[StoryTokenWindow] = []
    for story_index, story in enumerate(stories):
        if type(story) is not NllStory:
            raise TypeError("stories must contain NllStory values")
        token_ids = tokenizer.encode(story.text, add_eos=True)
        if len(token_ids) < 2:
            raise ValueError(
                f"story {story.record_id!r} must encode to at least two tokens including EOS"
            )
        target_count = len(token_ids) - 1
        for start in range(0, target_count, sequence_length):
            active_tokens = min(sequence_length, target_count - start)
            inputs = token_ids[start : start + active_tokens]
            targets = token_ids[start + 1 : start + active_tokens + 1]
            padding = (pad_token_id,) * (sequence_length - active_tokens)
            windows.append(
                StoryTokenWindow(
                    story_index=story_index,
                    input_ids=tuple(inputs) + padding,
                    target_ids=tuple(targets) + padding,
                    active_tokens=active_tokens,
                )
            )
    return tuple(windows)


def aggregate_story_window_nll(
    stories: Sequence[NllStory],
    windows: Sequence[StoryTokenWindow],
    *,
    batch_size: int,
    scorer: WindowBatchScorer,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[PerStoryNll, ...]:
    """Microbatch fixed windows and merge losses in stable story order."""
    if not stories or not windows:
        raise ValueError("NLL aggregation requires stories and windows")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("NLL batch_size must be positive")
    width = len(windows[0].input_ids)
    if any(len(window.input_ids) != width for window in windows):
        raise ValueError("all NLL windows must have identical width")
    total_losses = np.zeros(len(stories), dtype=np.float64)
    total_tokens = np.zeros(len(stories), dtype=np.int64)

    for batch_start in range(0, len(windows), batch_size):
        real_batch = tuple(windows[batch_start : batch_start + batch_size])
        padded_batch = real_batch + (real_batch[-1],) * (batch_size - len(real_batch))
        input_ids = np.asarray([window.input_ids for window in padded_batch], dtype=np.int32)
        target_ids = np.asarray([window.target_ids for window in padded_batch], dtype=np.int32)
        positions = np.arange(width, dtype=np.int32)[None, :]
        active = np.asarray(
            [window.active_tokens for window in padded_batch], dtype=np.int32
        )[:, None]
        attention_mask = positions < active
        loss_mask = attention_mask.astype(np.float32)
        batch_losses, batch_tokens = scorer(
            input_ids,
            target_ids,
            attention_mask,
            loss_mask,
        )
        losses = np.asarray(batch_losses, dtype=np.float64)
        counts = np.asarray(batch_tokens, dtype=np.int64)
        if losses.shape != (batch_size,) or counts.shape != (batch_size,):
            raise ValueError("NLL scorer outputs must have shape [batch]")
        if np.any(~np.isfinite(losses)) or np.any(losses < 0.0):
            raise ValueError("NLL scorer returned invalid losses")
        for row, window in enumerate(real_batch):
            if counts[row] != window.active_tokens:
                raise ValueError("NLL scorer token count does not match the loss mask")
            total_losses[window.story_index] += losses[row]
            total_tokens[window.story_index] += counts[row]
        if progress_callback is not None:
            progress_callback(len(real_batch))

    if np.any(total_tokens <= 0):
        raise ValueError("every NLL story must receive at least one active token")
    return tuple(
        PerStoryNll(
            record_id=story.record_id,
            total_nll=float(total_losses[index]),
            token_count=int(total_tokens[index]),
            normalized_nll=float(total_losses[index] / total_tokens[index]),
        )
        for index, story in enumerate(stories)
    )


def score_tinystories_checkpoint_nll(
    stories: Sequence[NllStory],
    checkpoint_directory: str | Path,
    tokenizer_path: str | Path,
    *,
    sequence_length: int = 256,
    batch_size: int = 32,
    require_gpu: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> ReferenceNllRun:
    """Score reference stories with the immutable TinyStories-8M checkpoint."""
    import hashlib

    import jax
    import jax.numpy as jnp

    from apm.lm.checkpoint import load_gpt_neo_checkpoint
    from apm.lm.gpt_neo import apply_gpt_neo
    from apm.lm.losses import per_token_nll
    from apm.lm.text import TokenizersTextTokenizer

    devices = jax.local_devices()
    if len(devices) != 1:
        raise RuntimeError(
            f"reference NLL requires exactly one local JAX device, found {len(devices)}"
        )
    device = devices[0]
    platform = str(device.platform)
    if require_gpu and platform != "gpu":
        raise RuntimeError(
            f"reference NLL production run requires the RTX GPU, got {platform!r}"
        )
    checkpoint = load_gpt_neo_checkpoint(Path(checkpoint_directory))
    if sequence_length > checkpoint.config.max_position_embeddings:
        raise ValueError("NLL sequence_length exceeds checkpoint context capacity")
    tokenizer_file = Path(tokenizer_path)
    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_file)
    windows = build_story_token_windows(
        stories,
        tokenizer,
        sequence_length=sequence_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    def evaluate_batch(
        input_ids: jax.Array,
        target_ids: jax.Array,
        attention_mask: jax.Array,
        loss_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            checkpoint.params,
            checkpoint.config,
            input_ids.astype(jnp.int32),
            attention_mask.astype(jnp.bool_),
        )
        mask = loss_mask.astype(jnp.float32)
        losses = per_token_nll(
            result.logits,
            target_ids.astype(jnp.int32),
        )
        return jnp.sum(losses * mask, axis=1), jnp.sum(mask, axis=1)

    compiled_evaluation = jax.jit(evaluate_batch)

    def compiled_scorer(
        input_ids: np.ndarray,
        target_ids: np.ndarray,
        attention_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        losses, counts = compiled_evaluation(
            jnp.asarray(input_ids),
            jnp.asarray(target_ids),
            jnp.asarray(attention_mask),
            jnp.asarray(loss_mask),
        )
        return np.asarray(losses), np.asarray(counts, dtype=np.int64)

    scores = aggregate_story_window_nll(
        stories,
        windows,
        batch_size=batch_size,
        scorer=compiled_scorer,
        progress_callback=progress_callback,
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    return ReferenceNllRun(
        checkpoint_parameter_checksum=checkpoint.reference.parameter_checksum,
        checkpoint_manifest_sha256=checkpoint.reference.manifest_sha256,
        tokenizer_sha256=tokenizer_sha256,
        jax_platform=platform,
        device_kind=str(device.device_kind),
        sequence_length=sequence_length,
        batch_size=batch_size,
        scores=scores,
    )


__all__ = [
    "NllStory",
    "PerStoryNll",
    "ReferenceNllRun",
    "StoryTokenWindow",
    "aggregate_story_window_nll",
    "build_story_token_windows",
    "score_tinystories_checkpoint_nll",
]
