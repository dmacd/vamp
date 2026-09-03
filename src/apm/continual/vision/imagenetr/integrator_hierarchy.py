"""Resumable capacity-one ImageNet-R hierarchy training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil

import torch

from apm.continual.artifacts import load_canonical_json, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.artifacts import (
    NodeBundle,
    load_node_bundle,
    publish_artifact_directory,
    write_node_work_directory,
)
from apm.continual.vision.imagenetr.bank import LogicalNode, MergeEvent
from apm.continual.vision.imagenetr.data import DatasetManifest, ImageRecord
from apm.continual.vision.imagenetr.heads import ClassifierRows, union_classifier_rows
from apm.continual.vision.imagenetr.integrator_artifacts import (
    HierarchyPolicy,
    IntegratorBootstrap,
    IntegratorStageSnapshot,
)
from apm.continual.vision.imagenetr.integrator_bank import (
    StratifiedReservoir,
    class_stratified_reservoir,
    merge_stratified_reservoirs,
    require_binary_work_bound,
    resize_stratified_reservoir,
    simulate_binary_topology,
)
from apm.continual.vision.imagenetr.manifests import git_commit_or_unknown
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.training import train_adapter_model


RESERVOIR_NAMESPACE = "imagenetr50-integrator-consolidation-bottom-k-v1"


@dataclass(frozen=True, slots=True)
class HierarchyWork:
    """Exact training work separated from cheap classifier-row arithmetic."""

    leaf_optimizer_steps: int = 0
    parent_optimizer_steps: int = 0
    leaf_image_presentations: int = 0
    parent_image_presentations: int = 0
    classifier_union_scalar_copies: int = 0
    reused_leaves: int = 0
    reused_parents: int = 0

    def __add__(self, other: "HierarchyWork") -> "HierarchyWork":
        return HierarchyWork(
            leaf_optimizer_steps=self.leaf_optimizer_steps + other.leaf_optimizer_steps,
            parent_optimizer_steps=self.parent_optimizer_steps + other.parent_optimizer_steps,
            leaf_image_presentations=(
                self.leaf_image_presentations + other.leaf_image_presentations
            ),
            parent_image_presentations=(
                self.parent_image_presentations + other.parent_image_presentations
            ),
            classifier_union_scalar_copies=(
                self.classifier_union_scalar_copies
                + other.classifier_union_scalar_copies
            ),
            reused_leaves=self.reused_leaves + other.reused_leaves,
            reused_parents=self.reused_parents + other.reused_parents,
        )


@dataclass(frozen=True, slots=True)
class HierarchyBuildResult:
    """One authenticated policy, its nodes, snapshots, and measured work."""

    policy: HierarchyPolicy
    nodes: tuple[NodeBundle, ...]
    snapshots: tuple[IntegratorStageSnapshot, ...]
    work: HierarchyWork

    @property
    def node_map(self) -> dict[str, NodeBundle]:
        """Map logical node identities to their validated tensor bundles."""
        return {
            logical_id: bundle
            for logical_id, bundle in (
                (str(load_canonical_json(bundle.directory / "logical.json")["logical_node_id"]), bundle)
                for bundle in self.nodes
            )
        }

    def frontier(self, stage: int) -> tuple[NodeBundle, ...]:
        """Return bundles in stable level-slot order for one completed stage."""
        if not 1 <= stage <= len(self.snapshots):
            raise ValueError("requested hierarchy stage is unavailable")
        snapshot = self.snapshots[stage - 1]
        mapping = self.node_map
        return tuple(mapping[logical_id] for logical_id in snapshot.logical_node_ids)


def _partition_rows(
    bootstrap: IntegratorBootstrap,
    partition: str,
    tasks: Sequence[int],
) -> tuple[ImageRecord, ...]:
    allowed = (
        frozenset(bootstrap.split.fit_image_ids)
        if partition == "fit"
        else frozenset(row.image_id for row in bootstrap.manifest.images if row.split == "train")
    )
    rows = bootstrap.manifest.select("train", tasks, allowed)
    if not rows or (partition == "fit" and set(row.image_id for row in rows) & set(bootstrap.split.validation_image_ids)):
        raise ValueError("hierarchy partition is empty or leaks validation identities")
    return rows


def _priority_hash(rows: Sequence[ImageRecord]) -> str:
    return record_sha256(
        [
            {"image_id": row.image_id, "priority": row.priority}
            for row in sorted(rows, key=lambda value: value.image_id)
        ]
    )


def _leaf_cache_path(
    bootstrap: IntegratorBootstrap, policy: HierarchyPolicy, task_index: int, job_hash: str
) -> Path:
    return (
        bootstrap.store.run
        / "hierarchies"
        / "leaf_cache"
        / policy.partition
        / f"seed_{policy.seed}"
        / f"task_{task_index:03d}"
        / job_hash
    )


def _node_metadata(
    bootstrap: IntegratorBootstrap,
    logical: LogicalNode,
    parent_hashes: tuple[str, ...],
    method: str,
    job_hash: str,
    reservoir: StratifiedReservoir,
    represented_rows: Sequence[ImageRecord],
    optimizer_steps: int,
) -> dict[str, object]:
    return {
        "run_hash": bootstrap.protocol.content_hash,
        "software_manifest_hash": bootstrap.protocol.environment_manifest_hash,
        "git_commit": git_commit_or_unknown(bootstrap.project_root),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "level": logical.level,
        "first_task": logical.first_task,
        "last_task": logical.last_task,
        "represented_task_ids": logical.task_ids,
        "represented_class_ids": tuple(range(4 * logical.first_task, 4 * (logical.last_task + 1))),
        "represented_train_image_count": len(represented_rows),
        "parent_hashes": parent_hashes,
        "unrepaired_parent_hash": None,
        "consolidation_method": method,
        "consolidation_config_hash": job_hash,
        "repair_config_hash": record_sha256(
            {"fraction": 0.0, "schema_version": "imagenetr50-integrator-no-repair-v1"}
        ),
        "proxy_image_ids": reservoir.image_ids,
        "repair_image_ids": (),
        "source_priority_hash": _priority_hash(represented_rows),
        "training_optimizer_steps": optimizer_steps,
    }


def _publish_trained_node(
    bootstrap: IntegratorBootstrap,
    target: Path,
    logical: LogicalNode,
    adapter: dict[str, LoRAFactors],
    classifier: ClassifierRows,
    metadata: dict[str, object],
    training_metrics: Mapping[str, object],
    reservoir: StratifiedReservoir,
) -> NodeBundle:
    if not all(
        isinstance(value, LoRAFactors) for value in adapter.values()
    ):
        raise TypeError("trained hierarchy node tensors have unexpected types")
    work = bootstrap.store.run / "work" / f"node_{logical.node_id}_{target.name}"
    if work.exists():
        shutil.rmtree(work)
    write_node_work_directory(work, adapter, classifier, metadata)
    publish_immutable_json(
        work / "logical.json",
        {
            "level": logical.level,
            "logical_node_id": logical.node_id,
            "represented_tasks": list(logical.task_ids),
            "schema_version": "imagenetr50-integrator-logical-node-link-v1",
        },
    )
    publish_immutable_json(
        work / "reservoir.json",
        {
            "capacity": reservoir.capacity,
            "content_hash": reservoir.content_hash,
            "image_ids": list(reservoir.image_ids),
            "namespace": reservoir.namespace,
            "represented_source_count": reservoir.represented_source_count,
            "selected_class_counts": [list(value) for value in reservoir.selected_class_counts],
            "source_class_counts": [list(value) for value in reservoir.source_class_counts],
            "schema_version": "imagenetr50-integrator-stratified-reservoir-v1",
        },
    )
    publish_immutable_json(
        work / "training_metrics.json",
        {**dict(training_metrics), "schema_version": "imagenetr50-integrator-node-training-v1"},
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return load_node_bundle(target)


def _train_leaf(
    bootstrap: IntegratorBootstrap,
    policy: HierarchyPolicy,
    task_index: int,
    backbone_factory: Callable[[], torch.nn.Module],
    device: torch.device,
    progress: bool,
) -> tuple[NodeBundle, HierarchyWork]:
    logical = LogicalNode(0, task_index, task_index)
    rows = _partition_rows(bootstrap, policy.partition, logical.task_ids)
    # Leaves are shared across every replay policy.  Retain the largest declared
    # reservoir so the first policy built cannot discard candidates needed later.
    leaf_reservoir_capacity = max(bootstrap.config.consolidation_reservoir_sizes)
    reservoir = class_stratified_reservoir(
        rows, leaf_reservoir_capacity, RESERVOIR_NAMESPACE
    )
    seed = policy.seed + 1000 * task_index
    job_hash = record_sha256(
        {
            "initialization_seed": seed,
            "partition": policy.partition,
            "source_ids": [row.image_id for row in rows],
            "task": task_index,
            "training_config_hash": policy.training_config_hash,
            "schema_version": "imagenetr50-integrator-leaf-job-v1",
        }
    )
    cache = _leaf_cache_path(bootstrap, policy, task_index, job_hash)
    if cache.is_dir():
        return load_node_bundle(cache), HierarchyWork(reused_leaves=1)
    model = AdapterVisionModel(
        backbone_factory(),
        tuple(range(4 * task_index, 4 * task_index + 4)),
        bootstrap.primary_config.lora_rank,
        bootstrap.primary_config.lora_alpha,
        0.0,
        seed,
    )
    result = train_adapter_model(
        model,
        bootstrap.config.data_root / "imagenet-r",
        rows,
        bootstrap.train_transform,
        bootstrap.config.consolidation_training,
        seed,
        device,
        bootstrap.store.run / "checkpoints" / f"leaf_{job_hash}.pt",
        num_workers=bootstrap.config.num_workers,
        checkpoint_steps=bootstrap.config.checkpoint_steps,
        show_progress=progress,
    )
    bundle = _publish_trained_node(
        bootstrap,
        cache,
        logical,
        result.adapter,
        result.classifier,
        _node_metadata(
            bootstrap, logical, (), "leaf", job_hash, reservoir, rows, result.optimizer_steps
        ),
        {
            "final_loss": result.final_loss,
            "image_presentations": result.image_presentations,
            "optimizer_steps": result.optimizer_steps,
            "peak_vram_bytes": result.peak_vram_bytes,
            "wall_seconds": result.wall_seconds,
        },
        reservoir,
    )
    return bundle, HierarchyWork(
        leaf_optimizer_steps=result.optimizer_steps,
        leaf_image_presentations=result.image_presentations,
    )


def _reservoir_from_bundle(
    bundle: NodeBundle,
    represented_rows: Sequence[ImageRecord],
    capacity: int,
) -> StratifiedReservoir:
    record = load_canonical_json(bundle.directory / "reservoir.json")
    retained = StratifiedReservoir(
        tuple(str(value) for value in record["image_ids"]),
        tuple((int(class_id), int(count)) for class_id, count in record["selected_class_counts"]),
        tuple((int(class_id), int(count)) for class_id, count in record["source_class_counts"]),
        int(record["represented_source_count"]),
        int(record["capacity"]),
        str(record["namespace"]),
    )
    source_counts = tuple(
        sorted(Counter(row.remapped_class_index for row in represented_rows).items())
    )
    if (
        retained.content_hash != record.get("content_hash")
        or retained.image_ids != bundle.artifact.proxy_image_ids
        or retained.source_class_counts != source_counts
        or retained.represented_source_count != len(represented_rows)
        or retained.namespace != RESERVOIR_NAMESPACE
    ):
        raise ValueError("node reservoir metadata differs from its represented source")
    rows_by_id = {row.image_id: row for row in represented_rows}
    return resize_stratified_reservoir(retained, rows_by_id, capacity)


def _train_parent(
    bootstrap: IntegratorBootstrap,
    policy: HierarchyPolicy,
    event: MergeEvent,
    left: NodeBundle,
    right: NodeBundle,
    rows_by_id: Mapping[str, ImageRecord],
    backbone_factory: Callable[[], torch.nn.Module],
    device: torch.device,
    progress: bool,
) -> tuple[NodeBundle, HierarchyWork]:
    target = bootstrap.store.hierarchy_node(policy.content_hash, event.parent.node_id)
    if target.is_dir():
        return load_node_bundle(target), HierarchyWork(reused_parents=1)
    represented_rows = _partition_rows(bootstrap, policy.partition, event.parent.task_ids)
    left_rows = _partition_rows(bootstrap, policy.partition, event.left.task_ids)
    right_rows = _partition_rows(bootstrap, policy.partition, event.right.task_ids)
    reservoir = merge_stratified_reservoirs(
        (
            _reservoir_from_bundle(left, left_rows, policy.reservoir_capacity),
            _reservoir_from_bundle(right, right_rows, policy.reservoir_capacity),
        ),
        rows_by_id,
        policy.reservoir_capacity,
        RESERVOIR_NAMESPACE,
    )
    training_rows = (
        represented_rows
        if policy.replay_mode == "full_union"
        else tuple(rows_by_id[image_id] for image_id in reservoir.image_ids)
    )
    classifier = union_classifier_rows((left.classifier, right.classifier))
    seed = policy.seed + 300_000 + event.sequence
    job_hash = record_sha256(
        {
            "children": [left.artifact.content_hash, right.artifact.content_hash],
            "initialization_seed": seed,
            "policy": policy.as_record(),
            "source_ids": [row.image_id for row in training_rows],
            "schema_version": "imagenetr50-integrator-parent-job-v1",
        }
    )
    model = AdapterVisionModel(
        backbone_factory(),
        classifier.class_ids,
        bootstrap.primary_config.lora_rank,
        bootstrap.primary_config.lora_alpha,
        0.0,
        seed,
        classifier,
    )
    result = train_adapter_model(
        model,
        bootstrap.config.data_root / "imagenet-r",
        training_rows,
        bootstrap.train_transform,
        bootstrap.config.consolidation_training,
        seed,
        device,
        bootstrap.store.run / "checkpoints" / f"parent_{job_hash}.pt",
        num_workers=bootstrap.config.num_workers,
        checkpoint_steps=bootstrap.config.checkpoint_steps,
        show_progress=progress,
    )
    bundle = _publish_trained_node(
        bootstrap,
        target,
        event.parent,
        result.adapter,
        result.classifier,
        _node_metadata(
            bootstrap,
            event.parent,
            (left.artifact.content_hash, right.artifact.content_hash),
            f"{policy.replay_mode}_retrain_union",
            job_hash,
            reservoir,
            represented_rows,
            result.optimizer_steps,
        ),
        {
            "final_loss": result.final_loss,
            "image_presentations": result.image_presentations,
            "optimizer_steps": result.optimizer_steps,
            "peak_vram_bytes": result.peak_vram_bytes,
            "training_image_count": len(training_rows),
            "wall_seconds": result.wall_seconds,
        },
        reservoir,
    )
    union_scalar_copies = len(classifier.class_ids) * (
        classifier.weight.shape[1] + 1
    )
    return bundle, HierarchyWork(
        parent_optimizer_steps=result.optimizer_steps,
        parent_image_presentations=result.image_presentations,
        classifier_union_scalar_copies=union_scalar_copies,
    )


def _snapshot_from_record(record: dict[str, object]) -> IntegratorStageSnapshot:
    supplied = str(record.pop("content_hash", ""))
    snapshot = IntegratorStageSnapshot(
        policy_hash=str(record["policy_hash"]),
        stage=int(record["stage"]),
        logical_node_ids=tuple(str(value) for value in record["logical_node_ids"]),
        node_hashes=tuple(str(value) for value in record["node_hashes"]),
        levels=tuple(int(value) for value in record["levels"]),
        schema_version=str(record["schema_version"]),
    )
    if snapshot.content_hash != supplied:
        raise ValueError("persisted hierarchy snapshot changed")
    return snapshot


def build_hierarchy(
    bootstrap: IntegratorBootstrap,
    policy: HierarchyPolicy,
    task_count: int,
    backbone_factory: Callable[[], torch.nn.Module],
    device: torch.device,
    progress: bool = True,
) -> HierarchyBuildResult:
    """Train or reuse one prefix of the capacity-one hierarchy and seal every frontier."""
    if not 1 <= task_count <= bootstrap.config.tasks:
        raise ValueError("hierarchy task count is outside 1..50")
    root = bootstrap.store.hierarchy(policy.content_hash)
    publish_immutable_json(root / "policy.json", policy.as_record())
    events, logical_snapshots = simulate_binary_topology(task_count)
    require_binary_work_bound(logical_snapshots, events)
    rows_by_id = {
        row.image_id: row
        for row in _partition_rows(bootstrap, policy.partition, tuple(range(task_count)))
    }
    bundles: dict[str, NodeBundle] = {}
    work = HierarchyWork()
    from tqdm.auto import tqdm

    overall = tqdm(
        total=task_count + len(events),
        desc=f"{policy.replay_mode} hierarchy through task {task_count}",
        disable=not progress,
        unit="node",
    )
    try:
        for task_index in range(task_count):
            logical = LogicalNode(0, task_index, task_index)
            leaf, leaf_work = _train_leaf(
                bootstrap, policy, task_index, backbone_factory, device, progress
            )
            target = bootstrap.store.hierarchy_node(policy.content_hash, logical.node_id)
            if not target.is_dir():
                publish_artifact_directory(leaf.directory, target)
            bundles[logical.node_id] = load_node_bundle(target)
            work += leaf_work
            overall.update(1)
        for event in events:
            parent, parent_work = _train_parent(
                bootstrap,
                policy,
                event,
                bundles[event.left.node_id],
                bundles[event.right.node_id],
                rows_by_id,
                backbone_factory,
                device,
                progress,
            )
            bundles[event.parent.node_id] = parent
            work += parent_work
            overall.update(1)
    finally:
        overall.close()
    snapshots = tuple(
        IntegratorStageSnapshot(
            policy.content_hash,
            logical.stage,
            tuple(node.node_id for node in logical.live_nodes),
            tuple(bundles[node.node_id].artifact.content_hash for node in logical.live_nodes),
            tuple(node.level for node in logical.live_nodes),
        )
        for logical in logical_snapshots
    )
    for snapshot in snapshots:
        path = bootstrap.store.snapshot(policy.content_hash, snapshot.stage)
        if path.is_file():
            if _snapshot_from_record(load_canonical_json(path)) != snapshot:
                raise ValueError("durable hierarchy frontier differs from reconstruction")
        else:
            publish_immutable_json(path, snapshot.as_record())
    # Completion identity must not depend on whether this invocation trained or
    # reused a node; otherwise resuming after evaluation failure changes immutable
    # bytes even though the scientific hierarchy is identical.
    publish_immutable_json(
        root / f"complete_{task_count:03d}.json",
        {
            "events": len(events),
            "final_snapshot_hash": snapshots[-1].content_hash,
            "node_hashes": sorted(bundle.artifact.content_hash for bundle in bundles.values()),
            "policy_hash": policy.content_hash,
            "schema_version": "imagenetr50-integrator-hierarchy-complete-v2",
            "task_count": task_count,
        },
    )
    return HierarchyBuildResult(
        policy,
        tuple(bundles[key] for key in sorted(bundles)),
        snapshots,
        work,
    )


__all__ = [
    "HierarchyBuildResult",
    "HierarchyWork",
    "RESERVOIR_NAMESPACE",
    "build_hierarchy",
]
