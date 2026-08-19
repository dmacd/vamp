"""Immutable TRACE adapter-bank lookup and replay-reservoir loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from apm.continual.trace.artifacts import validate_artifact_directory
from apm.continual.trace.lineage import HierarchyNode
from apm.continual.trace.protocol import MergePolicy
from apm.continual.trace.reservoirs import ReservoirEntry, select_reservoir


def node_from_record(value: Mapping[str, object]) -> HierarchyNode:
    """Strictly reconstruct one logical hierarchy node from canonical JSON fields."""
    return HierarchyNode(
        node_id=str(value["node_id"]),
        level=int(value["level"]),
        start_arrival=int(value["start_arrival"]),
        end_arrival=int(value["end_arrival"]),
        arrival_ids=tuple(str(item) for item in _sequence(value["arrival_ids"])),
        parent_node_ids=tuple(
            str(item) for item in _sequence(value["parent_node_ids"])
        ),
    )


def node_artifact_directory(
    run_directory: str | Path,
    node: HierarchyNode,
    policy_hash: str,
) -> Path:
    """Resolve and authenticate a leaf or policy-derived node artifact."""
    run = Path(run_directory)
    target = (
        run / "leaves" / node.arrival_ids[0]
        if node.level == 0
        else run / "derived" / policy_hash / "nodes" / node.node_id
    )
    validate_artifact_directory(target)
    return target


def node_reservoir(
    run_directory: str | Path,
    node: HierarchyNode,
    policy: MergePolicy,
) -> tuple[ReservoirEntry, ...]:
    """Load the policy-sized reservoir for a leaf or merged child."""
    directory = node_artifact_directory(run_directory, node, policy.policy_hash)
    value = json.loads(
        (directory / "reservoir_priorities.json").read_text(encoding="utf-8")
    )
    if type(value) is not dict or type(value.get("entries")) is not list:
        raise ValueError("node reservoir record is malformed")
    entries = tuple(
        ReservoirEntry(priority=str(row["priority"]), example_id=str(row["example_id"]))
        for row in value["entries"]
        if type(row) is dict
    )
    if len(entries) != len(value["entries"]):
        raise ValueError("node reservoir contains a malformed entry")
    if node.level == 0:
        return select_reservoir(
            entries,
            node.represented_examples,
            policy.repair_fraction,
        )
    expected = select_reservoir(
        entries,
        node.represented_examples,
        policy.repair_fraction,
    )
    if entries != expected:
        raise ValueError("derived node reservoir differs from its policy")
    return entries


def reservoir_record(entries: Sequence[ReservoirEntry]) -> dict[str, object]:
    """Return the canonical persisted form of a selected node reservoir."""
    return {
        "entries": [
            {"example_id": entry.example_id, "priority": entry.priority}
            for entry in entries
        ],
        "format": "trace-node-reservoir-v1",
    }


def _sequence(value: object) -> Sequence[object]:
    if type(value) is not list:
        raise ValueError("hierarchy node sequence field is malformed")
    return value


__all__ = [
    "node_artifact_directory",
    "node_from_record",
    "node_reservoir",
    "reservoir_record",
]
