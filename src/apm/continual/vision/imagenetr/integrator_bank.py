"""Capacity-one LogT topology and mergeable class-stratified reservoirs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.bank import LogicalNode, MergeEvent, StageSnapshot
from apm.continual.vision.imagenetr.data import ImageRecord, largest_remainder_counts


@dataclass(frozen=True, slots=True)
class BinaryCounterState:
    """Immutable capacity-one bank with at most one interval at each level."""

    levels: tuple[LogicalNode | None, ...] = ()

    def __post_init__(self) -> None:
        if any(node is not None and node.level != level for level, node in enumerate(self.levels)):
            raise ValueError("binary-counter node occupies the wrong level")
        represented = tuple(task for node in self.live_nodes for task in node.task_ids)
        if len(represented) != len(set(represented)):
            raise ValueError("binary-counter live intervals overlap")

    @property
    def live_nodes(self) -> tuple[LogicalNode, ...]:
        """Return active nodes in stable ascending level-slot order."""
        return tuple(node for node in self.levels if node is not None)

    def at(self, level: int) -> LogicalNode | None:
        """Return the node occupying a level, if one exists."""
        return self.levels[level] if level < len(self.levels) else None


def insert_binary_leaf(
    state: BinaryCounterState,
    task_index: int,
    prior_event_count: int,
) -> tuple[BinaryCounterState, tuple[MergeEvent, ...]]:
    """Insert the next task and carry occupied levels like a binary counter."""
    represented = {task for node in state.live_nodes for task in node.task_ids}
    if represented != set(range(task_index)) or prior_event_count < 0:
        raise ValueError("binary-counter leaves must arrive once in task order")
    levels = state.levels
    carry = LogicalNode(0, task_index, task_index)
    events: tuple[MergeEvent, ...] = ()
    while True:
        level = carry.level
        levels += (None,) * max(0, level + 1 - len(levels))
        occupied = levels[level]
        if occupied is None:
            levels = levels[:level] + (carry,) + levels[level + 1 :]
            break
        parent = LogicalNode(level + 1, occupied.first_task, carry.last_task)
        events += (
            MergeEvent(
                sequence=prior_event_count + len(events),
                stage=task_index + 1,
                left=occupied,
                right=carry,
                parent=parent,
            ),
        )
        levels = levels[:level] + (None,) + levels[level + 1 :]
        carry = parent
    while levels and levels[-1] is None:
        levels = levels[:-1]
    return BinaryCounterState(levels), events


def simulate_binary_topology(
    task_count: int = 50,
) -> tuple[tuple[MergeEvent, ...], tuple[StageSnapshot, ...]]:
    """Return all capacity-one carries and exact post-arrival frontiers."""
    if task_count < 1:
        raise ValueError("task count must be positive")
    state = BinaryCounterState()
    events: tuple[MergeEvent, ...] = ()
    snapshots: tuple[StageSnapshot, ...] = ()
    for task_index in range(task_count):
        state, created = insert_binary_leaf(state, task_index, len(events))
        events += created
        snapshots += (StageSnapshot(task_index + 1, state.live_nodes),)
    return events, snapshots


@dataclass(frozen=True, slots=True)
class StratifiedReservoir:
    """Permanent-priority bottom-K identities with explicit source coverage."""

    image_ids: tuple[str, ...]
    selected_class_counts: tuple[tuple[int, int], ...]
    source_class_counts: tuple[tuple[int, int], ...]
    represented_source_count: int
    capacity: int
    namespace: str

    def __post_init__(self) -> None:
        ids = tuple(self.image_ids)
        selected_counts = dict(self.selected_class_counts)
        source_counts = dict(self.source_class_counts)
        if (
            not self.namespace
            or self.capacity < 1
            or self.represented_source_count < len(ids)
            or len(ids) != min(self.capacity, self.represented_source_count)
            or len(set(ids)) != len(ids)
            or any(len(image_id) != 64 for image_id in ids)
            or not selected_counts
            or tuple(sorted(selected_counts.items())) != self.selected_class_counts
            or tuple(sorted(source_counts.items())) != self.source_class_counts
            or set(selected_counts) != set(source_counts)
            or any(not 0 <= class_id < 200 or count < 1 for class_id, count in self.selected_class_counts)
            or any(count < selected_counts[class_id] for class_id, count in self.source_class_counts)
            or sum(selected_counts.values()) != len(ids)
            or sum(source_counts.values()) != self.represented_source_count
        ):
            raise ValueError("invalid class-stratified reservoir")

    @property
    def content_hash(self) -> str:
        """Return the exact membership and selection-policy identity."""
        return record_sha256(
            {
                "capacity": self.capacity,
                "selected_class_counts": [list(value) for value in self.selected_class_counts],
                "source_class_counts": [list(value) for value in self.source_class_counts],
                "image_ids": list(self.image_ids),
                "namespace": self.namespace,
                "represented_source_count": self.represented_source_count,
                "schema_version": "imagenetr50-integrator-stratified-reservoir-v1",
            }
        )


def _priority(row: ImageRecord, namespace: str) -> tuple[str, str]:
    return (
        sha256(f"{namespace}\0{row.image_id}\0{row.priority}".encode()).hexdigest(),
        row.image_id,
    )


def _minimum_one_allocation(
    source_counts: Mapping[int, int], selected_total: int
) -> dict[int, int]:
    if selected_total < len(source_counts):
        raise ValueError("reservoir capacity must retain at least one image per class")
    base = {str(class_id): count - 1 for class_id, count in source_counts.items()}
    remaining = selected_total - len(source_counts)
    if remaining == 0:
        return {class_id: 1 for class_id in source_counts}
    positive = {name: count for name, count in base.items() if count > 0}
    additions = (
        largest_remainder_counts(positive, remaining)
        if positive
        else {str(class_id): 0 for class_id in source_counts}
    )
    return {
        class_id: 1 + additions.get(str(class_id), 0)
        for class_id in sorted(source_counts)
    }


def class_stratified_reservoir(
    rows: Sequence[ImageRecord], capacity: int, namespace: str
) -> StratifiedReservoir:
    """Select an order-independent approximately balanced bottom-K reservoir."""
    if not rows or capacity < 1 or not namespace or any(row.split != "train" for row in rows):
        raise ValueError("reservoirs require nonempty training rows and a namespace")
    if len({row.image_id for row in rows}) != len(rows):
        raise ValueError("reservoir source rows contain duplicate identities")
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.remapped_class_index].append(row)
    selected_total = min(capacity, len(rows))
    allocation = _minimum_one_allocation(
        {class_id: len(values) for class_id, values in grouped.items()}, selected_total
    )
    selected = tuple(
        row
        for class_id in sorted(grouped)
        for row in sorted(grouped[class_id], key=lambda value: _priority(value, namespace))[
            : allocation[class_id]
        ]
    )
    return StratifiedReservoir(
        tuple(row.image_id for row in selected),
        tuple((class_id, allocation[class_id]) for class_id in sorted(allocation)),
        tuple((class_id, len(grouped[class_id])) for class_id in sorted(grouped)),
        len(rows),
        capacity,
        namespace,
    )


def resize_stratified_reservoir(
    reservoir: StratifiedReservoir,
    rows_by_id: Mapping[str, ImageRecord],
    capacity: int,
) -> StratifiedReservoir:
    """Project a retained reservoir to a smaller capacity without reopening its source."""
    if capacity < 1 or capacity > reservoir.capacity:
        raise ValueError("reservoir projection must use a positive nonincreasing capacity")
    if capacity == reservoir.capacity:
        return reservoir
    if any(image_id not in rows_by_id for image_id in reservoir.image_ids):
        raise ValueError("retained reservoir identities cannot be resolved")
    source_counts = dict(reservoir.source_class_counts)
    selected_total = min(capacity, reservoir.represented_source_count)
    allocation = _minimum_one_allocation(source_counts, selected_total)
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for image_id in reservoir.image_ids:
        row = rows_by_id[image_id]
        grouped[row.remapped_class_index].append(row)
    if any(len(grouped[class_id]) < count for class_id, count in allocation.items()):
        raise ValueError("retained reservoir lacks a required permanent-priority row")
    selected = tuple(
        row
        for class_id in sorted(grouped)
        for row in sorted(
            grouped[class_id], key=lambda value: _priority(value, reservoir.namespace)
        )[: allocation[class_id]]
    )
    return StratifiedReservoir(
        tuple(row.image_id for row in selected),
        tuple((class_id, allocation[class_id]) for class_id in sorted(allocation)),
        reservoir.source_class_counts,
        reservoir.represented_source_count,
        capacity,
        reservoir.namespace,
    )


def merge_stratified_reservoirs(
    children: Sequence[StratifiedReservoir],
    rows_by_id: Mapping[str, ImageRecord],
    capacity: int,
    namespace: str,
) -> StratifiedReservoir:
    """Merge child samples without reopening retired full source populations."""
    if len(children) != 2 or any(child.namespace != namespace for child in children):
        raise ValueError("exactly two same-policy child reservoirs are required")
    ids = tuple(image_id for child in children for image_id in child.image_ids)
    if len(ids) != len(set(ids)) or any(image_id not in rows_by_id for image_id in ids):
        raise ValueError("child reservoir identities overlap or cannot be resolved")
    candidates = tuple(rows_by_id[image_id] for image_id in ids)
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for row in candidates:
        grouped[row.remapped_class_index].append(row)
    source_counts = {
        class_id: count
        for child in children
        for class_id, count in child.source_class_counts
    }
    if len(source_counts) != sum(len(child.source_class_counts) for child in children):
        raise ValueError("child source classes overlap")
    selected_total = min(capacity, sum(source_counts.values()))
    allocation = _minimum_one_allocation(source_counts, selected_total)
    if any(len(grouped[class_id]) < count for class_id, count in allocation.items()):
        raise ValueError("child reservoirs discarded a required permanent-priority row")
    selected = tuple(
        row
        for class_id in sorted(grouped)
        for row in sorted(grouped[class_id], key=lambda value: _priority(value, namespace))[
            : allocation[class_id]
        ]
    )
    return StratifiedReservoir(
        tuple(row.image_id for row in selected),
        tuple((class_id, allocation[class_id]) for class_id in sorted(allocation)),
        tuple(sorted(source_counts.items())),
        sum(child.represented_source_count for child in children),
        capacity,
        namespace,
    )


def require_binary_work_bound(
    snapshots: Iterable[StageSnapshot], events: Iterable[MergeEvent]
) -> None:
    """Assert the exact per-arrival and cumulative LogT structural bounds."""
    by_stage: dict[int, int] = defaultdict(int)
    for event in events:
        by_stage[event.stage] += 1
    for snapshot in snapshots:
        if len(snapshot.live_nodes) != snapshot.stage.bit_count():
            raise ValueError("capacity-one frontier does not equal popcount(stage)")
        if by_stage[snapshot.stage] > (snapshot.stage - 1).bit_length():
            raise ValueError("one arrival performed more than logarithmically many carries")


__all__ = [
    "BinaryCounterState",
    "StratifiedReservoir",
    "class_stratified_reservoir",
    "insert_binary_leaf",
    "merge_stratified_reservoirs",
    "require_binary_work_bound",
    "resize_stratified_reservoir",
    "simulate_binary_topology",
]
