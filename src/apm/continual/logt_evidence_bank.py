"""Pure standard binary-counter topology for temporal evidence banks."""

from __future__ import annotations

from dataclasses import dataclass
import math

from pyrsistent import PMap, pmap

from apm.continual.artifacts import record_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class EvidenceWorkCounters:
    """Exact logical evidence-training and routing work for one live bank."""

    evidence_train_example_updates: int = 0
    evidence_merge_example_updates: int = 0
    evidence_reference_examples: int = 0
    evidence_route_model_evals: int = 0
    active_evidence_models: int = 0

    def __post_init__(self) -> None:
        values = (
            self.evidence_train_example_updates,
            self.evidence_merge_example_updates,
            self.evidence_reference_examples,
            self.evidence_route_model_evals,
            self.active_evidence_models,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("evidence work counters must be nonnegative integers")
        if self.evidence_reference_examples != self.evidence_example_updates:
            raise ValueError("the fixed balanced protocol requires one reference per update")

    @property
    def evidence_example_updates(self) -> int:
        """Return leaf plus merge evidence presentations."""
        return self.evidence_train_example_updates + self.evidence_merge_example_updates

    def with_training(self, example_updates: int, *, merge: bool) -> "EvidenceWorkCounters":
        """Return counters after one fixed-budget leaf or merge model fit."""
        if example_updates < 1:
            raise ValueError("completed evidence training must consume examples")
        return EvidenceWorkCounters(
            self.evidence_train_example_updates + (0 if merge else example_updates),
            self.evidence_merge_example_updates + (example_updates if merge else 0),
            self.evidence_reference_examples + example_updates,
            self.evidence_route_model_evals,
            self.active_evidence_models,
        )

    def with_routing(self, model_evaluations: int, active_models: int) -> "EvidenceWorkCounters":
        """Return counters after a routed evaluation and update the live-model gauge."""
        if model_evaluations < 0 or active_models < 0:
            raise ValueError("routing work and active model counts cannot be negative")
        return EvidenceWorkCounters(
            self.evidence_train_example_updates,
            self.evidence_merge_example_updates,
            self.evidence_reference_examples,
            self.evidence_route_model_evals + model_evaluations,
            active_models,
        )


def require_evidence_work_bound(
    counters: EvidenceWorkCounters,
    processed_blocks: int,
    block_size: int,
    epochs: int,
    fixed_model_families: int,
) -> None:
    """Assert the declared fixed-multiple O(t log t) training ceiling."""
    if fixed_model_families < 1:
        raise ValueError("the work bound requires at least one fixed model family")
    ceiling = fixed_model_families * evidence_update_bound(processed_blocks, block_size, epochs)
    if counters.evidence_example_updates > ceiling:
        raise RuntimeError(
            f"evidence work {counters.evidence_example_updates} exceeded fixed ceiling {ceiling}"
        )


@dataclass(frozen=True, slots=True)
class TemporalNode:
    """One contiguous power-of-two interval represented by an active LogT node."""

    node_id: str
    level: int
    first_block: int
    last_block: int
    example_ids: tuple[int, ...]
    parent_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.node_id, "temporal node ID")
        for parent_node_id in self.parent_node_ids:
            require_sha256(parent_node_id, "temporal parent node ID")
        block_count = self.last_block - self.first_block + 1
        if (
            self.level < 0
            or self.first_block < 0
            or block_count != 2**self.level
            or not self.example_ids
            or len(set(self.example_ids)) != len(self.example_ids)
            or (self.level == 0) != (not self.parent_node_ids)
            or (self.level > 0 and len(self.parent_node_ids) != 2)
        ):
            raise ValueError("invalid temporal-node interval or lineage")

    @property
    def block_count(self) -> int:
        """Return the represented number of fixed stream blocks."""
        return self.last_block - self.first_block + 1


@dataclass(frozen=True, slots=True)
class TemporalMerge:
    """One deterministic equal-level binary-counter carry."""

    left: TemporalNode
    right: TemporalNode
    parent: TemporalNode

    def __post_init__(self) -> None:
        if (
            self.left.level != self.right.level
            or self.right.first_block != self.left.last_block + 1
            or self.parent.level != self.left.level + 1
            or self.parent.first_block != self.left.first_block
            or self.parent.last_block != self.right.last_block
            or self.parent.example_ids != self.left.example_ids + self.right.example_ids
            or self.parent.parent_node_ids != (self.left.node_id, self.right.node_id)
        ):
            raise ValueError("invalid temporal merge")


@dataclass(frozen=True, slots=True)
class LogTState:
    """Immutable live binary-counter levels after one contiguous stream prefix."""

    block_size: int
    processed_blocks: int
    active_by_level: PMap[int, TemporalNode]

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.processed_blocks < 0:
            raise ValueError("invalid LogT state dimensions")
        if any(level != node.level for level, node in self.active_by_level.items()):
            raise ValueError("active LogT levels do not match their nodes")
        chronological = self.active_nodes
        covered_blocks = tuple(
            block
            for node in chronological
            for block in range(node.first_block, node.last_block + 1)
        )
        covered_examples = tuple(example for node in chronological for example in node.example_ids)
        if (
            covered_blocks != tuple(range(self.processed_blocks))
            or covered_examples != tuple(range(self.processed_blocks * self.block_size))
            or len(self.active_by_level) > math.ceil(math.log2(self.processed_blocks + 1))
        ):
            raise ValueError("active LogT nodes do not partition the processed stream")

    @property
    def active_nodes(self) -> tuple[TemporalNode, ...]:
        """Return current nodes in chronological interval order."""
        return tuple(sorted(self.active_by_level.values(), key=lambda node: node.first_block))


def empty_logt_state(block_size: int) -> LogTState:
    """Return an empty true binary-counter state for one fixed block size."""
    return LogTState(block_size, 0, pmap())


def insert_block(
    state: LogTState,
    example_ids: tuple[int, ...],
) -> tuple[LogTState, TemporalNode, tuple[TemporalMerge, ...]]:
    """Insert one next block and return the leaf plus every induced carry."""
    expected = tuple(
        range(state.processed_blocks * state.block_size, (state.processed_blocks + 1) * state.block_size)
    )
    if example_ids != expected:
        raise ValueError("LogT blocks must be fixed, complete, and arrive in stream order")
    leaf = temporal_leaf(state.processed_blocks, example_ids)
    active = state.active_by_level
    current = leaf
    merges = []
    while current.level in active:
        left = active[current.level]
        active = active.remove(current.level)
        merge = merge_temporal_nodes(left, current)
        merges.append(merge)
        current = merge.parent
    active = active.set(current.level, current)
    return LogTState(state.block_size, state.processed_blocks + 1, active), leaf, tuple(merges)


def temporal_leaf(block: int, example_ids: tuple[int, ...]) -> TemporalNode:
    """Create one content-addressed level-zero node for a complete stream block."""
    return _node(0, block, block, example_ids, ())


def merge_temporal_nodes(left: TemporalNode, right: TemporalNode) -> TemporalMerge:
    """Merge two contiguous equal-level nodes into their content-addressed parent."""
    if left.level != right.level or left.last_block + 1 != right.first_block:
        raise ValueError("temporal consolidation requires contiguous equal-level nodes")
    parent = _node(
        left.level + 1,
        left.first_block,
        right.last_block,
        left.example_ids + right.example_ids,
        (left.node_id, right.node_id),
    )
    return TemporalMerge(left, right, parent)


def evidence_update_bound(processed_blocks: int, block_size: int, epochs: int) -> int:
    """Return the fixed O(t log t) example-update ceiling from the protocol."""
    if processed_blocks < 0 or block_size < 1 or epochs < 1:
        raise ValueError("invalid evidence work-bound arguments")
    if processed_blocks == 0:
        return 0
    return epochs * block_size * processed_blocks * math.ceil(math.log2(processed_blocks + 1))


def _node(
    level: int,
    first_block: int,
    last_block: int,
    example_ids: tuple[int, ...],
    parent_node_ids: tuple[str, ...],
) -> TemporalNode:
    core = {
        "example_count": len(example_ids),
        "example_ids_sha256": record_sha256(list(example_ids)),
        "first_block": first_block,
        "last_block": last_block,
        "level": level,
        "parent_node_ids": list(parent_node_ids),
        "schema_version": "vamp-logt-temporal-node-v1",
    }
    return TemporalNode(
        record_sha256(core),
        level,
        first_block,
        last_block,
        example_ids,
        parent_node_ids,
    )


__all__ = [
    "EvidenceWorkCounters",
    "LogTState",
    "TemporalMerge",
    "TemporalNode",
    "empty_logt_state",
    "evidence_update_bound",
    "insert_block",
    "merge_temporal_nodes",
    "require_evidence_work_bound",
    "temporal_leaf",
]
