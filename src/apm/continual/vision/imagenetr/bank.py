"""Pure capacity-two oldest-first logarithmic hierarchy transitions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from apm.continual.artifacts import record_sha256


@dataclass(frozen=True, slots=True)
class LogicalNode:
    """Policy-independent contiguous task interval in the temporal hierarchy."""

    level: int
    first_task: int
    last_task: int

    def __post_init__(self) -> None:
        task_count = self.last_task - self.first_task + 1
        if (
            self.level < 0
            or self.first_task < 0
            or self.last_task < self.first_task
            or task_count != 2**self.level
        ):
            raise ValueError("logical nodes must cover one aligned power-of-two interval")

    @property
    def task_ids(self) -> tuple[int, ...]:
        """Return the represented zero-based task IDs."""
        return tuple(range(self.first_task, self.last_task + 1))

    @property
    def node_id(self) -> str:
        """Return the policy-independent logical node identity."""
        return record_sha256(
            {
                "first_task": self.first_task,
                "last_task": self.last_task,
                "level": self.level,
                "schema_version": "imagenetr50-logical-node-v1",
            }
        )

    @property
    def one_based_interval(self) -> str:
        """Return the report interval using conventional one-based task numbers."""
        start, end = self.first_task + 1, self.last_task + 1
        return str(start) if start == end else f"{start}-{end}"


@dataclass(frozen=True, slots=True)
class MergeEvent:
    """One deterministic overflow merge in historical arrival order."""

    sequence: int
    stage: int
    left: LogicalNode
    right: LogicalNode
    parent: LogicalNode

    def __post_init__(self) -> None:
        if (
            self.sequence < 0
            or self.stage < 1
            or self.left.level != self.right.level
            or self.right.first_task != self.left.last_task + 1
            or self.parent.level != self.left.level + 1
            or self.parent.first_task != self.left.first_task
            or self.parent.last_task != self.right.last_task
        ):
            raise ValueError("invalid oldest-first merge event")

    @property
    def merge_id(self) -> str:
        """Return the stable logical merge-event identity."""
        return record_sha256(
            {
                "left": self.left.node_id,
                "parent": self.parent.node_id,
                "right": self.right.node_id,
                "schema_version": "imagenetr50-merge-event-v1",
                "sequence": self.sequence,
                "stage": self.stage,
            }
        )


@dataclass(frozen=True, slots=True)
class BankState:
    """Immutable oldest-to-newest live nodes at every represented level."""

    levels: tuple[tuple[LogicalNode, ...], ...] = ()

    def __post_init__(self) -> None:
        if any(
            len(nodes) > 2
            or any(node.level != level for node in nodes)
            or tuple(sorted(nodes, key=lambda node: node.first_task)) != nodes
            for level, nodes in enumerate(self.levels)
        ):
            raise ValueError("invalid capacity-two bank state")
        tasks = tuple(task for node in self.live_nodes for task in node.task_ids)
        if len(set(tasks)) != len(tasks):
            raise ValueError("live temporal nodes overlap")

    @property
    def live_nodes(self) -> tuple[LogicalNode, ...]:
        """Return all live nodes in level then age order."""
        return tuple(node for nodes in self.levels for node in nodes)

    def nodes_at(self, level: int) -> tuple[LogicalNode, ...]:
        """Return the oldest-to-newest queue at one level."""
        return self.levels[level] if level < len(self.levels) else ()


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Exact live logical bank immediately after one task arrival."""

    stage: int
    live_nodes: tuple[LogicalNode, ...]

    def __post_init__(self) -> None:
        represented = tuple(task for node in self.live_nodes for task in node.task_ids)
        if self.stage < 1 or tuple(sorted(represented)) != tuple(range(self.stage)):
            raise ValueError("stage snapshot does not partition all seen tasks")


def _set_level(
    levels: tuple[tuple[LogicalNode, ...], ...],
    level: int,
    nodes: tuple[LogicalNode, ...],
) -> tuple[tuple[LogicalNode, ...], ...]:
    extended = levels + ((),) * max(0, level + 1 - len(levels))
    replaced = extended[:level] + (nodes,) + extended[level + 1 :]
    return replaced[:-1] if replaced and not replaced[-1] else replaced


def insert_leaf(
    state: BankState,
    task_index: int,
    prior_event_count: int,
) -> tuple[BankState, tuple[MergeEvent, ...]]:
    """Insert one next leaf and perform every induced oldest-first carry."""
    represented = {task for node in state.live_nodes for task in node.task_ids}
    if represented != set(range(task_index)):
        raise ValueError("leaves must arrive once in strict task order")
    levels = _set_level(
        state.levels,
        0,
        state.nodes_at(0) + (LogicalNode(0, task_index, task_index),),
    )
    events: tuple[MergeEvent, ...] = ()
    level = 0
    while len(levels[level]) > 2:
        left, right, newest = levels[level]
        parent = LogicalNode(level + 1, left.first_task, right.last_task)
        levels = _set_level(levels, level, (newest,))
        next_nodes = levels[level + 1] if level + 1 < len(levels) else ()
        levels = _set_level(levels, level + 1, next_nodes + (parent,))
        events += (
            MergeEvent(
                sequence=prior_event_count + len(events),
                stage=task_index + 1,
                left=left,
                right=right,
                parent=parent,
            ),
        )
        level += 1
    return BankState(levels), events


def simulate_topology(task_count: int = 50) -> tuple[tuple[MergeEvent, ...], tuple[StageSnapshot, ...]]:
    """Construct the complete deterministic lineage and every historical snapshot."""
    if task_count < 1:
        raise ValueError("task count must be positive")
    state, events, snapshots = BankState(), (), ()
    for task_index in range(task_count):
        state, new_events = insert_leaf(state, task_index, len(events))
        events += new_events
        snapshots += (StageSnapshot(task_index + 1, state.live_nodes),)
    return events, snapshots


def require_partition(nodes: Iterable[LogicalNode], seen_tasks: int) -> None:
    """Require disjoint intervals whose union is every task seen so far."""
    tasks = tuple(task for node in nodes for task in node.task_ids)
    if len(tasks) != len(set(tasks)) or tuple(sorted(tasks)) != tuple(range(seen_tasks)):
        raise ValueError("live nodes do not form the required task partition")


__all__ = [
    "BankState",
    "LogicalNode",
    "MergeEvent",
    "StageSnapshot",
    "insert_leaf",
    "require_partition",
    "simulate_topology",
]
