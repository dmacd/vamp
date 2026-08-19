"""Pure capacity-two temporal hierarchy and immutable lineage identities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Sequence

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.trace.protocol import LEVEL_CAPACITY


@dataclass(frozen=True, slots=True)
class HierarchyNode:
    """One base-relative leaf or merged interval in the TRACE hierarchy."""

    node_id: str
    level: int
    start_arrival: int
    end_arrival: int
    arrival_ids: tuple[str, ...]
    parent_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.node_id, "hierarchy node")
        expected_count = 2**self.level
        if (
            type(self.level) is not int
            or self.level < 0
            or self.start_arrival < 1
            or self.end_arrival - self.start_arrival + 1 != expected_count
            or len(self.arrival_ids) != expected_count
            or len(set(self.arrival_ids)) != expected_count
        ):
            raise ValueError("hierarchy interval does not match its level")
        if (self.level == 0) != (not self.parent_node_ids):
            raise ValueError("only leaves omit parent node identities")
        if self.level > 0 and len(self.parent_node_ids) != 2:
            raise ValueError("merged nodes require exactly two parents")
        for identity in self.arrival_ids + self.parent_node_ids:
            require_sha256(identity, "hierarchy lineage")

    @property
    def represented_arrivals(self) -> int:
        """Return the number of level-zero arrivals represented by this node."""
        return len(self.arrival_ids)

    @property
    def represented_examples(self) -> int:
        """Return the number of training examples represented by this node."""
        return 100 * self.represented_arrivals

    def as_record(self) -> dict[str, object]:
        """Return the canonical logical lineage record."""
        return {
            "arrival_ids": list(self.arrival_ids),
            "end_arrival": self.end_arrival,
            "level": self.level,
            "node_id": self.node_id,
            "parent_node_ids": list(self.parent_node_ids),
            "start_arrival": self.start_arrival,
        }


@dataclass(frozen=True, slots=True)
class MergeEvent:
    """One deterministic oldest-first carry operation."""

    left: HierarchyNode
    right: HierarchyNode
    parent: HierarchyNode

    def __post_init__(self) -> None:
        if (
            self.left.level != self.right.level
            or self.parent.level != self.left.level + 1
            or self.left.end_arrival + 1 != self.right.start_arrival
            or self.parent.start_arrival != self.left.start_arrival
            or self.parent.end_arrival != self.right.end_arrival
            or self.parent.parent_node_ids != (self.left.node_id, self.right.node_id)
        ):
            raise ValueError("merge event has invalid lineage")


@dataclass(frozen=True, slots=True)
class HierarchyState:
    """Immutable live levels after a contiguous stream prefix."""

    arrival_count: int
    levels: tuple[tuple[HierarchyNode, ...], ...]

    def __post_init__(self) -> None:
        if self.arrival_count < 0:
            raise ValueError("arrival count must be nonnegative")
        if any(
            node.level != level or len(nodes) > LEVEL_CAPACITY
            for level, nodes in enumerate(self.levels)
            for node in nodes
        ):
            raise ValueError("hierarchy violates level identity or capacity")
        covered = tuple(
            chain.from_iterable(
                range(node.start_arrival, node.end_arrival + 1)
                for node in self.active_nodes
            )
        )
        if covered != tuple(range(1, self.arrival_count + 1)):
            raise ValueError("active nodes do not partition the stream prefix")

    @property
    def active_nodes(self) -> tuple[HierarchyNode, ...]:
        """Return live nodes in chronological order."""
        return tuple(
            sorted(
                (node for level in self.levels for node in level),
                key=lambda node: node.start_arrival,
            )
        )

    def topology(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return interval endpoints grouped by level."""
        return tuple(
            tuple((node.start_arrival, node.end_arrival) for node in nodes)
            for nodes in self.levels
        )


def empty_hierarchy() -> HierarchyState:
    """Return an empty TRACE hierarchy."""
    return HierarchyState(arrival_count=0, levels=())


def leaf_node(arrival: int, arrival_id: str) -> HierarchyNode:
    """Create a level-zero logical node for one immutable arrival."""
    require_sha256(arrival_id, "arrival")
    core = {
        "arrival_ids": [arrival_id],
        "end_arrival": arrival,
        "format": "trace-hierarchy-node-v1",
        "level": 0,
        "parent_node_ids": [],
        "start_arrival": arrival,
    }
    return HierarchyNode(
        node_id=record_sha256(core),
        level=0,
        start_arrival=arrival,
        end_arrival=arrival,
        arrival_ids=(arrival_id,),
    )


def merged_node(left: HierarchyNode, right: HierarchyNode) -> HierarchyNode:
    """Create the unique parent of two contiguous equal-level nodes."""
    if left.level != right.level or left.end_arrival + 1 != right.start_arrival:
        raise ValueError("only contiguous equal-level nodes may merge")
    core = {
        "arrival_ids": list(left.arrival_ids + right.arrival_ids),
        "end_arrival": right.end_arrival,
        "format": "trace-hierarchy-node-v1",
        "level": left.level + 1,
        "parent_node_ids": [left.node_id, right.node_id],
        "start_arrival": left.start_arrival,
    }
    return HierarchyNode(
        node_id=record_sha256(core),
        level=left.level + 1,
        start_arrival=left.start_arrival,
        end_arrival=right.end_arrival,
        arrival_ids=left.arrival_ids + right.arrival_ids,
        parent_node_ids=(left.node_id, right.node_id),
    )


def insert_arrival(
    state: HierarchyState,
    arrival_id: str,
) -> tuple[HierarchyState, tuple[MergeEvent, ...]]:
    """Insert an arrival and synchronously perform every required carry."""
    current = leaf_node(state.arrival_count + 1, arrival_id)
    levels = [list(nodes) for nodes in state.levels]
    merges: list[MergeEvent] = []
    level = 0
    while True:
        while len(levels) <= level:
            levels.append([])
        levels[level] = sorted((*levels[level], current), key=lambda node: node.start_arrival)
        if len(levels[level]) <= LEVEL_CAPACITY:
            break
        left, right, newest = levels[level]
        levels[level] = [newest]
        parent = merged_node(left, right)
        merges.append(MergeEvent(left=left, right=right, parent=parent))
        current = parent
        level += 1
    while levels and not levels[-1]:
        levels.pop()
    return (
        HierarchyState(
            arrival_count=state.arrival_count + 1,
            levels=tuple(tuple(nodes) for nodes in levels),
        ),
        tuple(merges),
    )


def build_hierarchy(
    arrival_ids: Sequence[str],
) -> tuple[HierarchyState, tuple[MergeEvent, ...]]:
    """Build the hierarchy and complete ordered merge history for arrivals."""
    state = empty_hierarchy()
    history: list[MergeEvent] = []
    for arrival_id in arrival_ids:
        state, merges = insert_arrival(state, arrival_id)
        history.extend(merges)
    return state, tuple(history)


__all__ = [
    "HierarchyNode",
    "HierarchyState",
    "MergeEvent",
    "build_hierarchy",
    "empty_hierarchy",
    "insert_arrival",
    "leaf_node",
    "merged_node",
]
