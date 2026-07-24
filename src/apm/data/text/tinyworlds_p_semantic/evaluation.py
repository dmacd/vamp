"""Per-duplicate-group semantic evaluation and empirical-null persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apm.data.text.tinyworlds_p.training import allocator_peak_bytes
from apm.data.text.tinyworlds_p_semantic.contracts import (
    SEMANTIC_TRAINING_PRESET,
    WORLD_LABELS,
    SemanticPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.statistics import (
    CANONICAL_REPLICATES,
    GroupLoss,
    PairedLoss,
    SemanticEpochValidation,
    summarize_empirical_gaps,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch


EvaluationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class SplitGroupEvaluation:
    """One aggregate split NLL backed by a sorted immutable group ledger."""

    split: str
    active_tokens: int
    loss_sum: float
    ledger_path: Path
    ledger_sha256: str
    group_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ledger_path", Path(self.ledger_path))
        if not self.ledger_path.is_file():
            raise FileNotFoundError(self.ledger_path)
        if type(self.active_tokens) is not int or self.active_tokens <= 0:
            raise ValueError("evaluated semantic split must contain active tokens")
        if not math.isfinite(self.loss_sum) or self.loss_sum < 0.0:
            raise ValueError("semantic split loss sum must be finite and nonnegative")
        if type(self.group_count) is not int or self.group_count <= 0:
            raise ValueError("semantic split must contain duplicate groups")

    @property
    def nll(self) -> float:
        """Return aggregate token-normalized NLL without averaging group NLLs."""
        return self.loss_sum / self.active_tokens


@dataclass(frozen=True, slots=True)
class SemanticSealedTest:
    """One and only one selected-checkpoint sealed-test opening."""

    selected_epoch: int
    held_in: SplitGroupEvaluation
    validation: SemanticEpochValidation
    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not 2 <= self.selected_epoch <= 5:
            raise ValueError("sealed semantic test epoch must lie in 2-5")
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)


def evaluate_partition_split(
    params: GptNeoParams,
    artifact: SemanticPartitionArtifact,
    split: str,
    ledger_path: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    progress: EvaluationProgress | None = None,
) -> SplitGroupEvaluation:
    """Stream model losses and persist sorted loss sums/counts per duplicate group."""
    config = model_config or SEMANTIC_TRAINING_PRESET.model_config
    if config.vocab_size != artifact.tokenizer_identity.vocab_size:
        raise ValueError("evaluation model vocabulary differs from semantic tokenizer")
    path = Path(ledger_path)
    if path.exists():
        raise FileExistsError(f"semantic group-loss ledger already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_batch(batch: TokenBatch) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            params,
            config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
        )
        mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask, axis=1), jnp.sum(mask, axis=1)

    compiled = jax.jit(evaluate_batch)
    planned_batches = _count_evaluation_batches(artifact, split) if progress is not None else 0
    total_loss = 0.0
    total_tokens = 0
    group_count = 0
    current_group: str | None = None
    current_loss = 0.0
    current_tokens = 0
    previous_group = ""
    with path.open("wb") as output:
        for completed, (batch, groups) in enumerate(
            _iter_group_batches(artifact, split),
            start=1,
        ):
            row_losses, row_tokens = compiled(batch)
            losses = np.asarray(row_losses, dtype=np.float64)
            tokens = np.asarray(row_tokens, dtype=np.int64)
            for group, loss_sum, active_tokens in zip(groups, losses, tokens, strict=True):
                if group is None:
                    if int(active_tokens) != 0:
                        raise ValueError("padded semantic evaluation row acquired active tokens")
                    continue
                if current_group is not None and group != current_group:
                    if group <= previous_group:
                        raise ValueError("semantic group-loss stream is not sorted")
                    _write_group_loss(output, current_group, current_loss, current_tokens)
                    previous_group = current_group
                    group_count += 1
                    current_loss, current_tokens = 0.0, 0
                current_group = group
                current_loss += float(loss_sum)
                current_tokens += int(active_tokens)
                total_loss += float(loss_sum)
                total_tokens += int(active_tokens)
            if progress is not None:
                progress(split, completed, planned_batches)
        if current_group is not None:
            _write_group_loss(output, current_group, current_loss, current_tokens)
            group_count += 1
        output.flush()
        os.fsync(output.fileno())
    if total_tokens == 0 or group_count == 0:
        raise ValueError(f"semantic evaluation split contains no tokens: {split}")
    return SplitGroupEvaluation(
        split=split,
        active_tokens=total_tokens,
        loss_sum=total_loss,
        ledger_path=path,
        ledger_sha256=_file_sha256(path),
        group_count=group_count,
    )


def evaluate_epoch_validation(
    params: GptNeoParams,
    artifact: SemanticPartitionArtifact,
    epoch: int,
    output_directory: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    replicates: int = CANONICAL_REPLICATES,
    progress: EvaluationProgress | None = None,
) -> SemanticEpochValidation:
    """Evaluate held-in/world/control validation and persist empirical-null evidence."""
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(f"semantic epoch evaluation already exists: {directory}")
    directory.mkdir(parents=True)
    held = evaluate_partition_split(
        params,
        artifact,
        "base/validation",
        directory / "base-validation.groups.jsonl",
        model_config,
        progress=progress,
    )
    evaluations = {
        (role, world): evaluate_partition_split(
            params,
            artifact,
            f"{role}/{world}/validation",
            directory / f"{role}-{world}-validation.groups.jsonl",
            model_config,
            progress=progress,
        )
        for world in WORLD_LABELS
        for role in ("world", "control")
    }
    pairs_by_world = tuple(
        _paired_losses(
            artifact,
            world,
            "validation",
            evaluations[("world", world)].ledger_path,
            evaluations[("control", world)].ledger_path,
        )
        for world in WORLD_LABELS
    )
    identity = record_sha256(
        {
            "epoch": epoch,
            "ledgers": {
                f"{role}/{world}": evaluations[(role, world)].ledger_sha256
                for world in WORLD_LABELS
                for role in ("world", "control")
            },
            "partition_sha256": artifact.partition_sha256,
            "split": "validation",
        }
    )
    worlds, mean = summarize_empirical_gaps(
        pairs_by_world,
        identity,
        replicates=replicates,
    )
    validation = SemanticEpochValidation(
        epoch=epoch,
        held_in_nll=held.nll,
        worlds=worlds,
        mean_empirical=mean,
        allocator_peak_bytes=allocator_peak_bytes(),
    )
    _write_json(
        directory / "validation.json",
        {
            "evaluation_identity_sha256": identity,
            "held_in": _split_record(held),
            "partition_sha256": artifact.partition_sha256,
            "validation": semantic_validation_record(validation),
        },
    )
    _write_tree(directory, identity, "tinyworlds-p-semantic-validation-tree")
    return validation


def evaluate_sealed_test_once(
    params: GptNeoParams,
    artifact: SemanticPartitionArtifact,
    selected_epoch: int,
    output_directory: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    replicates: int = CANONICAL_REPLICATES,
    progress: EvaluationProgress | None = None,
) -> SemanticSealedTest:
    """Open sealed test exactly once and report, without changing checkpoint selection."""
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(f"sealed semantic test was already opened: {directory}")
    directory.mkdir(parents=True)
    _write_json(
        directory / "sealed-open.json",
        {
            "partition_sha256": artifact.partition_sha256,
            "selected_epoch": selected_epoch,
        },
    )
    held = evaluate_partition_split(
        params,
        artifact,
        "base/test",
        directory / "base-test.groups.jsonl",
        model_config,
        progress=progress,
    )
    evaluations = {
        (role, world): evaluate_partition_split(
            params,
            artifact,
            f"{role}/{world}/test",
            directory / f"{role}-{world}-test.groups.jsonl",
            model_config,
            progress=progress,
        )
        for world in WORLD_LABELS
        for role in ("world", "control")
    }
    pairs_by_world = tuple(
        _paired_losses(
            artifact,
            world,
            "test",
            evaluations[("world", world)].ledger_path,
            evaluations[("control", world)].ledger_path,
        )
        for world in WORLD_LABELS
    )
    identity = record_sha256(
        {
            "ledgers": {
                f"{role}/{world}": evaluations[(role, world)].ledger_sha256
                for world in WORLD_LABELS
                for role in ("world", "control")
            },
            "partition_sha256": artifact.partition_sha256,
            "selected_epoch": selected_epoch,
            "split": "test",
        }
    )
    worlds, mean = summarize_empirical_gaps(
        pairs_by_world,
        identity,
        replicates=replicates,
    )
    result = SemanticEpochValidation(
        epoch=selected_epoch,
        held_in_nll=held.nll,
        worlds=worlds,
        mean_empirical=mean,
        allocator_peak_bytes=allocator_peak_bytes(),
    )
    _write_json(
        directory / "sealed-test.json",
        {
            "evaluation_identity_sha256": identity,
            "held_in": _split_record(held),
            "partition_sha256": artifact.partition_sha256,
            "selected_epoch": selected_epoch,
            "test": semantic_validation_record(result),
        },
    )
    _write_tree(directory, identity, "tinyworlds-p-semantic-sealed-test-tree")
    return SemanticSealedTest(selected_epoch, held, result, directory)


def load_group_losses(path: str | Path) -> tuple[GroupLoss, ...]:
    """Strictly load a sorted per-duplicate-group loss ledger."""
    previous = ""
    losses = []
    with Path(path).open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid group-loss ledger line {line_number}") from error
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError("group-loss ledger must be canonical JSONL")
            group = _text(record, "normalized_story_sha256")
            if group <= previous:
                raise ValueError("group-loss ledger is not unique and sorted")
            previous = group
            loss = record.get("loss_sum")
            tokens = record.get("active_tokens")
            if type(loss) not in (int, float) or type(tokens) is not int:
                raise ValueError("group-loss ledger values are malformed")
            losses.append(GroupLoss(group, float(loss), tokens))
    return tuple(losses)


def semantic_validation_record(validation: SemanticEpochValidation) -> dict[str, object]:
    """Return canonical persisted effect sizes, intervals, and placebo probabilities."""
    empirical = lambda item: {
        "bootstrap_lower": item.bootstrap_lower,
        "bootstrap_upper": item.bootstrap_upper,
        "observed_gap": item.observed_gap,
        "placebo_probability": item.placebo_probability,
        "replicate_count": item.replicate_count,
    }
    return {
        "allocator_peak_bytes": validation.allocator_peak_bytes,
        "epoch": validation.epoch,
        "held_in_nll": validation.held_in_nll,
        "mean": empirical(validation.mean_empirical),
        "worlds": [
            {
                "control_nll": item.control_nll,
                "empirical": empirical(item.empirical),
                "world": item.world,
                "world_nll": item.world_nll,
            }
            for item in validation.worlds
        ],
    }


def _paired_losses(
    artifact: SemanticPartitionArtifact,
    world: str,
    split: str,
    world_ledger: Path,
    control_ledger: Path,
) -> tuple[PairedLoss, ...]:
    world_losses = {item.normalized_story_sha256: item for item in load_group_losses(world_ledger)}
    control_losses = {item.normalized_story_sha256: item for item in load_group_losses(control_ledger)}
    pairs = tuple(
        PairedLoss(
            world=world,
            world_loss=world_losses[item.world_group_sha256],
            control_loss=control_losses[item.control_group_sha256],
        )
        for item in artifact.pairings
        if item.world == world and item.split == split
    )
    if (
        {item.world_loss.normalized_story_sha256 for item in pairs} != set(world_losses)
        or {item.control_loss.normalized_story_sha256 for item in pairs} != set(control_losses)
    ):
        raise ValueError("persisted pairings do not cover evaluated group-loss ledgers")
    return pairs


def _iter_group_batches(
    artifact: SemanticPartitionArtifact,
    split: str,
) -> Iterator[tuple[TokenBatch, tuple[str | None, ...]]]:
    path = artifact.root / "indexes" / _index_filename(split)
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    groups: list[str | None] = []
    shards: dict[int, np.memmap] = {}
    try:
        for record in _iter_index(path):
            shard_id, offset, token_count = (
                _integer(record, field)
                for field in ("token_shard", "token_offset", "token_count")
            )
            shard = shards.get(shard_id)
            if shard is None:
                shard = np.memmap(
                    artifact.root / "shards" / f"tokens-{shard_id:06d}.uint16",
                    dtype="<u2",
                    mode="r",
                )
                shards[shard_id] = shard
            tokens = np.asarray(shard[offset : offset + token_count], dtype=np.int32)
            if len(tokens) != token_count:
                raise ValueError("semantic evaluation index exceeds its token shard")
            group = _text(record, "normalized_story_sha256")
            for start in range(0, max(token_count - 1, 0), artifact.preset.context_length):
                rows.append(_window(tokens, start, artifact))
                groups.append(group)
                if len(rows) == artifact.preset.batch_size:
                    yield _stack(rows), tuple(groups)
                    rows, groups = [], []
        if rows:
            for _ in range(artifact.preset.batch_size - len(rows)):
                rows.append(_empty_window(artifact))
                groups.append(None)
            yield _stack(rows), tuple(groups)
    finally:
        shards.clear()


def _window(
    tokens: np.ndarray,
    start: int,
    artifact: SemanticPartitionArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = artifact.preset.context_length
    chunk = tokens[start : start + length + 1]
    transitions = len(chunk) - 1
    input_ids = np.full(length, artifact.pad_token_id, dtype=np.int32)
    target_ids = np.full(length, artifact.pad_token_id, dtype=np.int32)
    attention = np.zeros(length, dtype=np.bool_)
    loss_mask = np.zeros(length, dtype=np.bool_)
    input_ids[:transitions] = chunk[:-1]
    target_ids[:transitions] = chunk[1:]
    attention[:transitions] = True
    loss_mask[:transitions] = True
    return input_ids, attention, target_ids, loss_mask


def _empty_window(
    artifact: SemanticPartitionArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = artifact.preset.context_length
    return (
        np.full(length, artifact.pad_token_id, dtype=np.int32),
        np.zeros(length, dtype=np.bool_),
        np.full(length, artifact.pad_token_id, dtype=np.int32),
        np.zeros(length, dtype=np.bool_),
    )


def _stack(rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> TokenBatch:
    return TokenBatch(
        input_ids=np.stack([item[0] for item in rows]),
        attention_mask=np.stack([item[1] for item in rows]),
        target_ids=np.stack([item[2] for item in rows]),
        loss_mask=np.stack([item[3] for item in rows]),
    )


def _count_evaluation_batches(artifact: SemanticPartitionArtifact, split: str) -> int:
    windows = sum(
        math.ceil(max(0, _integer(record, "token_count") - 1) / artifact.preset.context_length)
        for record in _iter_index(artifact.root / "indexes" / _index_filename(split))
    )
    return math.ceil(windows / artifact.preset.batch_size)


def _index_filename(split: str) -> str:
    parts = split.split("/")
    if len(parts) == 2 and parts[0] == "base" and parts[1] in ("validation", "test"):
        return f"base-{parts[1]}.jsonl"
    if (
        len(parts) == 3
        and parts[0] in ("world", "control")
        and parts[1] in WORLD_LABELS
        and parts[2] in ("validation", "test")
    ):
        return f"{parts[0]}-{parts[1]}-{parts[2]}.jsonl"
    raise ValueError("semantic evaluation may open only validation or sealed-test splits")


def _iter_index(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line in source:
            record = json.loads(line)
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError(f"semantic evaluation index is not canonical: {path}")
            yield record


def _write_group_loss(
    output,
    group: str,
    loss_sum: float,
    active_tokens: int,
) -> None:
    output.write(
        canonical_json_bytes(
            {
                "active_tokens": active_tokens,
                "loss_sum": loss_sum,
                "normalized_story_sha256": group,
            }
        )
    )


def _split_record(result: SplitGroupEvaluation) -> dict[str, object]:
    return {
        "active_tokens": result.active_tokens,
        "group_count": result.group_count,
        "ledger": result.ledger_path.name,
        "ledger_sha256": result.ledger_sha256,
        "loss_sum": result.loss_sum,
        "nll": result.nll,
        "split": result.split,
    }


def _write_tree(directory: Path, identity: str, format_name: str) -> None:
    files = tuple(
        {
            "name": path.name,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        directory / "tree.json",
        {
            "evaluation_identity_sha256": identity,
            "files": list(files),
            "format": format_name,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"field {field!r} must be a nonnegative integer")
    return value


__all__ = [
    "EvaluationProgress",
    "SemanticSealedTest",
    "SplitGroupEvaluation",
    "evaluate_epoch_validation",
    "evaluate_partition_split",
    "evaluate_sealed_test_once",
    "load_group_losses",
    "semantic_validation_record",
]
