"""Immutable contracts and pure schedules for the nouns-v2 temporal study."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Literal, TypeAlias

from apm.data.text.tinyworlds_nouns_v1.experiment import StoryIndexEntry
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)


STUDY_ID = "tinyworlds-nouns-v2-temporal-consolidation"
CONTRACT_FORMAT = f"{STUDY_ID}-contract-v1"
ADAPTER_FORMAT = f"{STUDY_ID}-adapter-v1"
FULL_MODEL_FORMAT = f"{STUDY_ID}-full-model-v1"
TRAINING_ROW_FORMAT = f"{STUDY_ID}-training-row-v1"
EVALUATION_ROW_FORMAT = f"{STUDY_ID}-evaluation-row-v1"
MERGE_ROW_FORMAT = f"{STUDY_ID}-merge-row-v1"
TIMING_ROW_FORMAT = f"{STUDY_ID}-timing-row-v1"
PROGRESS_ROW_FORMAT = f"{STUDY_ID}-progress-row-v1"
REPORT_FORMAT = f"{STUDY_ID}-report-v1"

SEED = 0
TASK_STORY_COUNT = 4_096
SHARDS_PER_TASK = 8
STORIES_PER_SHARD = 512
ARRIVAL_COUNT = len(TASK_IDS) * SHARDS_PER_TASK
FIXED_EPOCHS = 4
LEVEL_CAPACITY = 2
CONTEXT_LENGTH = 256
PHYSICAL_BATCH_SIZE = 32
EVALUATION_BATCH_SIZE = 32
LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_LEARNING_RATE = 1e-3
FULL_MODEL_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
SENTINEL_STORIES_PER_TASK = 16
MACRO_CHECKPOINT_INTERVAL = SHARDS_PER_TASK
BOOTSTRAP_REPETITIONS = 10_000
CHECKPOINT_UPDATE_INTERVAL = 128
CHECKPOINT_SECONDS = 120.0
ALLOCATOR_LIMIT_BYTES = 12 * 1024**3
WARM_TIMING_REPETITIONS = 5

TemporalOrder: TypeAlias = Literal["blocked", "round_robin"]
TEMPORAL_ORDERS: tuple[TemporalOrder, ...] = ("blocked", "round_robin")


@dataclass(frozen=True, slots=True)
class TemporalShard:
    """One immutable 512-story level-zero training dataset."""

    shard_id: str
    task_id: str
    shard_index: int
    story_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.shard_id, "temporal shard")
        if self.task_id not in TASK_IDS:
            raise ValueError("temporal shard task is not canonical")
        if not 0 <= self.shard_index < SHARDS_PER_TASK:
            raise ValueError("temporal shard index is outside the fixed range")
        if (
            len(self.story_ids) != STORIES_PER_SHARD
            or len(set(self.story_ids)) != STORIES_PER_SHARD
        ):
            raise ValueError("temporal shards require 512 unique stories")
        for story_id in self.story_ids:
            require_sha256(story_id, "temporal shard story")

    def as_record(self) -> dict[str, object]:
        """Return the canonical contract representation of this shard."""
        return {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "story_ids": list(self.story_ids),
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class TemporalChunk:
    """One active or retired interval in the deterministic b=2 hierarchy."""

    chunk_id: str
    order: TemporalOrder
    level: int
    start_arrival: int
    end_arrival: int
    shard_ids: tuple[str, ...]
    task_counts: tuple[tuple[str, int], ...]
    parent_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.chunk_id, "temporal chunk")
        if self.order not in TEMPORAL_ORDERS:
            raise ValueError("temporal chunk order is invalid")
        expected_size = 2**self.level
        if (
            type(self.level) is not int
            or self.level < 0
            or self.start_arrival < 1
            or self.end_arrival - self.start_arrival + 1 != expected_size
            or len(self.shard_ids) != expected_size
            or len(set(self.shard_ids)) != expected_size
        ):
            raise ValueError("temporal chunk interval does not match its level")
        if sum(count for _, count in self.task_counts) != expected_size:
            raise ValueError("temporal chunk task counts do not cover its shards")
        if tuple(task for task, _ in self.task_counts) != tuple(
            task for task in TASK_IDS if task in dict(self.task_counts)
        ):
            raise ValueError("temporal chunk task counts are not canonical")
        if any(count <= 0 for _, count in self.task_counts):
            raise ValueError("temporal chunk task counts must be positive")
        if (self.level == 0) != (len(self.parent_chunk_ids) == 0):
            raise ValueError("only level-zero chunks omit parent chunks")
        if self.level > 0 and len(self.parent_chunk_ids) != 2:
            raise ValueError("merged temporal chunks require exactly two parents")
        for shard_id in self.shard_ids:
            require_sha256(shard_id, "temporal chunk shard")
        for parent_id in self.parent_chunk_ids:
            require_sha256(parent_id, "temporal chunk parent")

    @property
    def size(self) -> int:
        """Return the represented arrival count."""
        return len(self.shard_ids)

    @property
    def noun_entropy(self) -> float:
        """Return normalized noun entropy in nats for descriptive analysis."""
        counts = tuple(count for _, count in self.task_counts)
        total = sum(counts)
        return -sum(
            (count / total) * math.log(count / total) for count in counts
        )

    def as_record(self) -> dict[str, object]:
        """Return a canonical chunk descriptor."""
        return {
            "chunk_id": self.chunk_id,
            "end_arrival": self.end_arrival,
            "level": self.level,
            "noun_entropy": self.noun_entropy,
            "order": self.order,
            "parent_chunk_ids": list(self.parent_chunk_ids),
            "shard_ids": list(self.shard_ids),
            "start_arrival": self.start_arrival,
            "task_counts": [[task, count] for task, count in self.task_counts],
        }


@dataclass(frozen=True, slots=True)
class TemporalMerge:
    """One deterministic oldest-first carry operation."""

    left: TemporalChunk
    right: TemporalChunk
    parent: TemporalChunk

    def __post_init__(self) -> None:
        if (
            self.left.order != self.right.order
            or self.left.order != self.parent.order
            or self.left.level != self.right.level
            or self.parent.level != self.left.level + 1
            or self.left.end_arrival + 1 != self.right.start_arrival
            or self.parent.start_arrival != self.left.start_arrival
            or self.parent.end_arrival != self.right.end_arrival
            or self.parent.parent_chunk_ids
            != (self.left.chunk_id, self.right.chunk_id)
        ):
            raise ValueError("temporal merge lineage or intervals are invalid")


@dataclass(frozen=True, slots=True)
class TemporalHierarchyState:
    """Immutable live levels after a contiguous stream prefix."""

    order: TemporalOrder
    arrival_count: int
    levels: tuple[tuple[TemporalChunk, ...], ...]

    def __post_init__(self) -> None:
        if self.order not in TEMPORAL_ORDERS or self.arrival_count < 0:
            raise ValueError("temporal hierarchy identity is invalid")
        chunks = self.active_chunks
        if any(
            chunk.order != self.order
            or chunk.level != level
            or len(level_chunks) > LEVEL_CAPACITY
            for level, level_chunks in enumerate(self.levels)
            for chunk in level_chunks
        ):
            raise ValueError("temporal hierarchy violates its level capacity")
        covered = tuple(
            arrival
            for chunk in sorted(chunks, key=lambda value: value.start_arrival)
            for arrival in range(chunk.start_arrival, chunk.end_arrival + 1)
        )
        if covered != tuple(range(1, self.arrival_count + 1)):
            raise ValueError("active temporal chunks do not partition the stream prefix")

    @property
    def active_chunks(self) -> tuple[TemporalChunk, ...]:
        """Return active chunks in chronological order."""
        return tuple(
            sorted(
                (chunk for level in self.levels for chunk in level),
                key=lambda value: (value.start_arrival, value.end_arrival),
            )
        )


def empty_hierarchy(order: TemporalOrder) -> TemporalHierarchyState:
    """Return the empty hierarchy for one fixed ordering."""
    if order not in TEMPORAL_ORDERS:
        raise ValueError("unknown temporal order")
    return TemporalHierarchyState(order=order, arrival_count=0, levels=())


def select_temporal_shards(
    entries_by_task: Mapping[str, Sequence[StoryIndexEntry]],
    probe_ids_by_task: Mapping[str, Sequence[str]],
    validation_story_ids: Sequence[str],
) -> tuple[TemporalShard, ...]:
    """Select and shard the fixed pure-story training population."""
    if tuple(entries_by_task) != TASK_IDS or tuple(probe_ids_by_task) != TASK_IDS:
        raise ValueError("temporal selection requires canonical task mapping order")
    validation_ids = frozenset(validation_story_ids)
    selected: list[TemporalShard] = []
    seen: set[str] = set()
    for task_id in TASK_IDS:
        probe_ids = frozenset(probe_ids_by_task[task_id])
        if len(probe_ids) != 36:
            raise ValueError("temporal selection requires all 36 task probes")
        eligible = tuple(
            sorted(
                (
                    entry
                    for entry in entries_by_task[task_id]
                    if entry.story_id not in probe_ids
                    and entry.story_id not in validation_ids
                ),
                key=lambda entry: (
                    _selection_digest(task_id, entry.story_id),
                    entry.story_id,
                ),
            )
        )
        if len(eligible) < TASK_STORY_COUNT:
            raise ValueError(f"task {task_id} has too few eligible temporal stories")
        story_ids = tuple(entry.story_id for entry in eligible[:TASK_STORY_COUNT])
        if seen & set(story_ids):
            raise ValueError("temporal task selections overlap")
        seen.update(story_ids)
        for shard_index in range(SHARDS_PER_TASK):
            start = shard_index * STORIES_PER_SHARD
            shard_story_ids = story_ids[start : start + STORIES_PER_SHARD]
            shard_core = {
                "format": f"{STUDY_ID}-shard-v1",
                "seed": SEED,
                "shard_index": shard_index,
                "story_ids": list(shard_story_ids),
                "task_id": task_id,
            }
            selected.append(
                TemporalShard(
                    shard_id=record_sha256(shard_core),
                    task_id=task_id,
                    shard_index=shard_index,
                    story_ids=shard_story_ids,
                )
            )
    if len(selected) != ARRIVAL_COUNT or len(seen) != ARRIVAL_COUNT * STORIES_PER_SHARD:
        raise RuntimeError("temporal shard coverage differs from the fixed contract")
    return tuple(selected)


def temporal_arrivals(
    shards: Sequence[TemporalShard],
    order: TemporalOrder,
) -> tuple[TemporalShard, ...]:
    """Order the same shard identities into one temporal stream."""
    if order not in TEMPORAL_ORDERS:
        raise ValueError("unknown temporal order")
    if len(shards) != ARRIVAL_COUNT or len({shard.shard_id for shard in shards}) != ARRIVAL_COUNT:
        raise ValueError("temporal arrival ordering requires all fixed shards")
    lookup = {(shard.task_id, shard.shard_index): shard for shard in shards}
    expected_keys = {
        (task_id, shard_index)
        for task_id in TASK_IDS
        for shard_index in range(SHARDS_PER_TASK)
    }
    if set(lookup) != expected_keys:
        raise ValueError("temporal shards do not cover every noun/index pair")
    keys = (
        tuple(
            (task_id, shard_index)
            for task_id in TASK_IDS
            for shard_index in range(SHARDS_PER_TASK)
        )
        if order == "blocked"
        else tuple(
            (task_id, shard_index)
            for shard_index in range(SHARDS_PER_TASK)
            for task_id in TASK_IDS
        )
    )
    return tuple(lookup[key] for key in keys)


def insert_arrival(
    state: TemporalHierarchyState,
    shard: TemporalShard,
) -> tuple[TemporalHierarchyState, tuple[TemporalMerge, ...]]:
    """Insert one arrival and perform every synchronous oldest-first carry."""
    arrival = state.arrival_count + 1
    current = _level_zero_chunk(state.order, arrival, shard)
    levels = [list(level) for level in state.levels]
    merges: list[TemporalMerge] = []
    level = 0
    while True:
        while len(levels) <= level:
            levels.append([])
        levels[level].append(current)
        levels[level].sort(key=lambda chunk: chunk.start_arrival)
        if len(levels[level]) <= LEVEL_CAPACITY:
            break
        left, right, newest = levels[level]
        levels[level] = [newest]
        parent = _merged_chunk(left, right)
        merges.append(TemporalMerge(left, right, parent))
        current = parent
        level += 1
    while levels and not levels[-1]:
        levels.pop()
    return (
        TemporalHierarchyState(
            order=state.order,
            arrival_count=arrival,
            levels=tuple(tuple(level_chunks) for level_chunks in levels),
        ),
        tuple(merges),
    )


def simulate_hierarchy(
    shards: Sequence[TemporalShard],
    order: TemporalOrder,
) -> tuple[TemporalHierarchyState, tuple[TemporalMerge, ...]]:
    """Return the final state and complete merge schedule for one ordering."""
    state = empty_hierarchy(order)
    history: list[TemporalMerge] = []
    for shard in temporal_arrivals(shards, order):
        state, merges = insert_arrival(state, shard)
        history.extend(merges)
    return state, tuple(history)


def select_validation_sentinel(
    entries_by_task: Mapping[str, Sequence[StoryIndexEntry]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Select the fixed 16-story validation sentinel for each noun."""
    if tuple(entries_by_task) != TASK_IDS:
        raise ValueError("sentinel selection requires canonical task mapping order")
    result = tuple(
        (
            task_id,
            tuple(
                entry.story_id
                for entry in sorted(
                    entries_by_task[task_id],
                    key=lambda entry: (
                        _sentinel_digest(task_id, entry.story_id),
                        entry.story_id,
                    ),
                )[:SENTINEL_STORIES_PER_TASK]
            ),
        )
        for task_id in TASK_IDS
    )
    if any(len(ids) != SENTINEL_STORIES_PER_TASK for _, ids in result):
        raise ValueError("a noun has too few validation stories for its sentinel")
    return result


def build_contract_record(
    *,
    bindings: Mapping[str, object],
    shards: Sequence[TemporalShard],
    sentinel: Sequence[tuple[str, Sequence[str]]],
) -> dict[str, object]:
    """Build the independently hashed executable v1 study contract."""
    blocked = temporal_arrivals(shards, "blocked")
    round_robin = temporal_arrivals(shards, "round_robin")
    core = {
        "addressing": {
            "candidate_order": "base_then_interval_end_level_artifact",
            "router_input": "exact_midpoint_prefix_only",
            "score": "mean_prefix_token_nll",
            "tie_break": "first_minimum",
        },
        "bindings": dict(sorted(bindings.items())),
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": SEED},
        "format": CONTRACT_FORMAT,
        "hierarchy": {
            "level_capacity": LEVEL_CAPACITY,
            "merge_policy": "two_oldest_equal_level_always_merge",
            "representation": "standalone_base_relative_lora",
        },
        "orders": {
            "blocked": [shard.shard_id for shard in blocked],
            "round_robin": [shard.shard_id for shard in round_robin],
        },
        "schema_version": 1,
        "sentinel": [[task, list(ids)] for task, ids in sentinel],
        "shards": [shard.as_record() for shard in shards],
        "study_id": STUDY_ID,
        "training": {
            "batch_size": PHYSICAL_BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "epochs": FIXED_EPOCHS,
            "full_model_learning_rate": FULL_MODEL_LEARNING_RATE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "lora_alpha": LORA_ALPHA,
            "lora_learning_rate": LORA_LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    return {**core, "contract_sha256": record_sha256(core)}


def validate_contract_record(record: Mapping[str, object]) -> dict[str, object]:
    """Strictly validate a canonical temporal-study contract value."""
    value = dict(record)
    supplied = value.pop("contract_sha256", None)
    if (
        supplied != record_sha256(value)
        or value.get("format") != CONTRACT_FORMAT
        or value.get("schema_version") != 1
        or value.get("study_id") != STUDY_ID
    ):
        raise ValueError("temporal-consolidation contract identity changed")
    return {**value, "contract_sha256": supplied}


def contract_bytes(record: Mapping[str, object]) -> bytes:
    """Return canonical bytes only after validating the contract hash."""
    return canonical_json_bytes(validate_contract_record(record))


def expected_final_intervals() -> tuple[tuple[int, int], ...]:
    """Return the preregistered final b=2 intervals for 192 arrivals."""
    return (
        (1, 64),
        (65, 128),
        (129, 160),
        (161, 176),
        (177, 184),
        (185, 188),
        (189, 190),
        (191, 191),
        (192, 192),
    )


def _selection_digest(task_id: str, story_id: str) -> str:
    return sha256(
        f"{STUDY_ID}\0selection\0{SEED}\0{task_id}\0{story_id}".encode("utf-8")
    ).hexdigest()


def _sentinel_digest(task_id: str, story_id: str) -> str:
    return sha256(
        f"{STUDY_ID}\0sentinel\0{SEED}\0{task_id}\0{story_id}".encode("utf-8")
    ).hexdigest()


def _level_zero_chunk(
    order: TemporalOrder,
    arrival: int,
    shard: TemporalShard,
) -> TemporalChunk:
    core = {
        "end_arrival": arrival,
        "format": f"{STUDY_ID}-chunk-v1",
        "level": 0,
        "order": order,
        "parent_chunk_ids": [],
        "shard_ids": [shard.shard_id],
        "start_arrival": arrival,
        "task_counts": [[shard.task_id, 1]],
    }
    return TemporalChunk(
        chunk_id=record_sha256(core),
        order=order,
        level=0,
        start_arrival=arrival,
        end_arrival=arrival,
        shard_ids=(shard.shard_id,),
        task_counts=((shard.task_id, 1),),
    )


def _merged_chunk(left: TemporalChunk, right: TemporalChunk) -> TemporalChunk:
    counts = Counter(dict(left.task_counts))
    counts.update(dict(right.task_counts))
    task_counts = tuple((task, counts[task]) for task in TASK_IDS if counts[task])
    core = {
        "end_arrival": right.end_arrival,
        "format": f"{STUDY_ID}-chunk-v1",
        "level": left.level + 1,
        "order": left.order,
        "parent_chunk_ids": [left.chunk_id, right.chunk_id],
        "shard_ids": list(left.shard_ids + right.shard_ids),
        "start_arrival": left.start_arrival,
        "task_counts": [[task, count] for task, count in task_counts],
    }
    return TemporalChunk(
        chunk_id=record_sha256(core),
        order=left.order,
        level=left.level + 1,
        start_arrival=left.start_arrival,
        end_arrival=right.end_arrival,
        shard_ids=left.shard_ids + right.shard_ids,
        task_counts=task_counts,
        parent_chunk_ids=(left.chunk_id, right.chunk_id),
    )


__all__ = [
    "ADAPTER_FORMAT",
    "ALLOCATOR_LIMIT_BYTES",
    "ARRIVAL_COUNT",
    "BOOTSTRAP_REPETITIONS",
    "CHECKPOINT_SECONDS",
    "CHECKPOINT_UPDATE_INTERVAL",
    "CONTRACT_FORMAT",
    "CONTEXT_LENGTH",
    "EVALUATION_BATCH_SIZE",
    "EVALUATION_ROW_FORMAT",
    "FIXED_EPOCHS",
    "FULL_MODEL_FORMAT",
    "FULL_MODEL_LEARNING_RATE",
    "GRADIENT_CLIP_NORM",
    "LEVEL_CAPACITY",
    "LORA_ALPHA",
    "LORA_LEARNING_RATE",
    "LORA_RANK",
    "MACRO_CHECKPOINT_INTERVAL",
    "MERGE_ROW_FORMAT",
    "PHYSICAL_BATCH_SIZE",
    "PROGRESS_ROW_FORMAT",
    "REPORT_FORMAT",
    "SEED",
    "SENTINEL_STORIES_PER_TASK",
    "SHARDS_PER_TASK",
    "STORIES_PER_SHARD",
    "STUDY_ID",
    "TASK_STORY_COUNT",
    "TEMPORAL_ORDERS",
    "TIMING_ROW_FORMAT",
    "TRAINING_ROW_FORMAT",
    "TemporalChunk",
    "TemporalHierarchyState",
    "TemporalMerge",
    "TemporalOrder",
    "TemporalShard",
    "WARM_TIMING_REPETITIONS",
    "WEIGHT_DECAY",
    "build_contract_record",
    "contract_bytes",
    "empty_hierarchy",
    "expected_final_intervals",
    "insert_arrival",
    "select_temporal_shards",
    "select_validation_sentinel",
    "simulate_hierarchy",
    "temporal_arrivals",
    "validate_contract_record",
]
