"""Policy tree construction over one immutable shared leaf set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import math

import torch
from torch import Tensor

from apm.continual.artifacts import publish_immutable_json
from apm.continual.vision.imagenetr.artifacts import NodeBundle, VisionStore
from apm.continual.vision.imagenetr.bank import (
    LogicalNode,
    MergeEvent,
    StageSnapshot,
    require_partition,
    simulate_topology,
)
from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.data import DatasetManifest
from apm.continual.vision.imagenetr.merging.parents import build_parent
from apm.continual.vision.imagenetr.protocol import MergePolicy


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    """One historical logical snapshot projected to exact policy artifact hashes."""

    stage: int
    logical_node_ids: tuple[str, ...]
    node_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TreeBuildResult:
    """Complete or smoke hierarchy and explicit leaf-reuse accounting."""

    policy: MergePolicy
    nodes: tuple[NodeBundle, ...]
    snapshots: tuple[MaterializedSnapshot, ...]
    merge_events: int
    new_parent_optimizer_steps: int
    leaf_optimizer_steps: int
    leaf_hashes_unchanged: bool


def build_tree(
    store: VisionStore,
    leaves: Sequence[NodeBundle],
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    policy: MergePolicy,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    proxy_transform: object,
    device: torch.device,
    software_manifest_hash: str,
    git_commit: str,
    activation_provider: Callable[[tuple[str, ...]], Mapping[str, Tensor]] | None = None,
    task_count: int = 50,
    show_progress: bool = True,
) -> TreeBuildResult:
    """Build a deterministic historical hierarchy without ever training a leaf."""
    if len(leaves) < task_count or task_count < 1 or task_count > config.tasks:
        raise ValueError("tree construction lacks the required immutable leaves")
    selected = tuple(sorted(leaves[:task_count], key=lambda leaf: leaf.artifact.first_task))
    if tuple(leaf.artifact.first_task for leaf in selected) != tuple(range(task_count)):
        raise ValueError("leaf set does not contain each task exactly once in order")
    original_leaf_hashes = tuple(leaf.artifact.content_hash for leaf in selected)
    events, logical_snapshots = simulate_topology(task_count)
    events_by_stage = {
        stage: tuple(event for event in events if event.stage == stage)
        for stage in range(1, task_count + 1)
    }
    nodes_by_logical_id: dict[str, NodeBundle] = {}
    all_nodes: list[NodeBundle] = []
    materialized: list[MaterializedSnapshot] = []
    optimizer_steps = 0
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        total=len(events),
        disable=not show_progress,
        desc=f"{policy.method} tree",
        unit="merge",
    )
    for stage in range(1, task_count + 1):
        leaf = selected[stage - 1]
        leaf_logical = LogicalNode(0, stage - 1, stage - 1)
        nodes_by_logical_id[leaf_logical.node_id] = leaf
        all_nodes.append(leaf)
        for event in events_by_stage[stage]:
            execution = build_parent(
                store,
                event,
                nodes_by_logical_id[event.left.node_id],
                nodes_by_logical_id[event.right.node_id],
                manifest,
                prepared_root,
                config,
                policy,
                backbone_factory,
                train_transform,
                proxy_transform,
                device,
                software_manifest_hash,
                git_commit,
                activation_provider,
                show_progress=False,
            )
            nodes_by_logical_id[event.parent.node_id] = execution.bundle
            all_nodes.append(execution.bundle)
            optimizer_steps += execution.optimizer_steps_this_execution
            progress.update(1)
        snapshot = logical_snapshots[stage - 1]
        require_partition(snapshot.live_nodes, stage)
        live = tuple(nodes_by_logical_id[node.node_id] for node in snapshot.live_nodes)
        class_ids = tuple(
            class_id for bundle in live for class_id in bundle.artifact.represented_class_ids
        )
        if len(class_ids) != len(set(class_ids)) or set(class_ids) != set(range(4 * stage)):
            raise ValueError("materialized live-node class namespaces do not partition seen classes")
        materialized.append(
            MaterializedSnapshot(
                stage,
                tuple(node.node_id for node in snapshot.live_nodes),
                tuple(bundle.artifact.content_hash for bundle in live),
            )
        )
    progress.close()
    tree_root = store.run / "trees" / policy.content_hash
    publish_immutable_json(tree_root / "policy.json", policy.as_record())
    snapshot_path = tree_root / (
        "snapshots.json" if task_count == config.tasks else f"snapshots_{task_count:03d}.json"
    )
    publish_immutable_json(
        snapshot_path,
        {
            "policy_hash": policy.content_hash,
            "schema_version": "imagenetr50-materialized-snapshots-v1",
            "snapshots": [
                {
                    "logical_node_ids": list(snapshot.logical_node_ids),
                    "node_hashes": list(snapshot.node_hashes),
                    "stage": snapshot.stage,
                }
                for snapshot in materialized
            ],
        },
    )
    final_leaf_hashes = tuple(leaf.artifact.content_hash for leaf in selected)
    return TreeBuildResult(
        policy=policy,
        nodes=tuple(all_nodes),
        snapshots=tuple(materialized),
        merge_events=len(events),
        new_parent_optimizer_steps=optimizer_steps,
        leaf_optimizer_steps=0,
        leaf_hashes_unchanged=original_leaf_hashes == final_leaf_hashes,
    )


__all__ = ["MaterializedSnapshot", "TreeBuildResult", "build_tree"]
