"""Historical evaluation projections for baselines and the all-leaf bank."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.artifacts import NodeBundle
from apm.continual.vision.imagenetr.baselines import BaselineExecution
from apm.continual.vision.imagenetr.data import DatasetManifest, deterministic_bottom_k
from apm.continual.vision.imagenetr.heads import ClassifierRows, load_classifier
from apm.continual.vision.imagenetr.lineage import MaterializedSnapshot, TreeBuildResult
from apm.continual.vision.imagenetr.lora import load_adapter
from apm.continual.vision.imagenetr.protocol import MergePolicy, NodeArtifact


def _dummy_policy(rank: int) -> MergePolicy:
    return MergePolicy(
        method="retrain_union",
        output_rank=rank,
        scale=1.0,
        weighting="source_image_count",
        repair_fraction=0.0,
        repair_config_hash=record_sha256(
            {"fraction": 0.0, "schema_version": "imagenetr50-control-no-repair-v1"}
        ),
        proxy_size=16,
    )


def leaf_bank_tree(leaves: Sequence[NodeBundle], rank: int = 16) -> TreeBuildResult:
    """Represent all immutable leaves as the non-scalable accumulating diagnostic bank."""
    ordered = tuple(sorted(leaves, key=lambda leaf: leaf.artifact.first_task))
    if tuple(leaf.artifact.first_task for leaf in ordered) != tuple(range(50)):
        raise ValueError("all-leaf control requires exactly tasks 0..49")
    snapshots = tuple(
        MaterializedSnapshot(
            stage,
            tuple(leaf.artifact.content_hash for leaf in ordered[:stage]),
            tuple(leaf.artifact.content_hash for leaf in ordered[:stage]),
        )
        for stage in range(1, 51)
    )
    return TreeBuildResult(
        _dummy_policy(rank), ordered, snapshots, 0, 0, 0, True
    )


def _tensor_hash(rows: ClassifierRows) -> str:
    return record_sha256(
        {
            "bias": rows.bias.detach().cpu().tolist(),
            "class_ids": list(rows.class_ids),
            "schema_version": "imagenetr50-derived-control-head-v1",
            "weight": rows.weight.detach().cpu().tolist(),
        }
    )


def baseline_history_tree(
    execution: BaselineExecution,
    manifest: DatasetManifest,
    rank: int = 16,
) -> TreeBuildResult:
    """Load/derive 50 immutable stage states for frozen, sequential, or joint controls."""
    root = execution.bundle.directory / "stage_snapshots"
    is_joint = execution.name == "joint_iid_lora_r16"
    if is_joint:
        source = root / "stage_001"
        adapter = load_adapter(source / "adapter.safetensors")
        full_head = load_classifier(source / "classifier.safetensors")
        sources = tuple(source for _ in range(50))
    else:
        sources = tuple(root / f"stage_{stage:03d}" for stage in range(1, 51))
        if any(not source.is_dir() for source in sources):
            raise ValueError("baseline artifact lacks all 50 historical stage states")
    bundles = []
    snapshots = []
    for stage, source in enumerate(sources, start=1):
        if is_joint:
            class_count = 4 * stage
            head = ClassifierRows(
                tuple(range(class_count)),
                full_head.weight[:class_count].clone(),
                full_head.bias[:class_count].clone(),
            )
            stage_adapter = adapter
            lora_sha = execution.bundle.artifact.lora_sha256
        else:
            stage_adapter = load_adapter(source / "adapter.safetensors")
            head = load_classifier(source / "classifier.safetensors")
            from apm.continual.artifacts import file_sha256

            lora_sha = file_sha256(source / "adapter.safetensors")
        rows = manifest.select("train", range(stage))
        artifact = NodeArtifact(
            run_hash=execution.bundle.artifact.run_hash,
            software_manifest_hash=execution.bundle.artifact.software_manifest_hash,
            git_commit=execution.bundle.artifact.git_commit,
            creation_timestamp_utc=execution.bundle.artifact.creation_timestamp_utc,
            level=0,
            first_task=0,
            last_task=stage - 1,
            represented_task_ids=tuple(range(stage)),
            represented_class_ids=tuple(range(4 * stage)),
            represented_train_image_count=len(rows),
            parent_hashes=(),
            unrepaired_parent_hash=None,
            consolidation_method="baseline",
            consolidation_config_hash=record_sha256(
                {
                    "baseline": execution.name,
                    "schema_version": "imagenetr50-baseline-stage-v1",
                    "stage": stage,
                }
            ),
            repair_config_hash=execution.bundle.artifact.repair_config_hash,
            lora_sha256=lora_sha,
            classifier_sha256=_tensor_hash(head),
            proxy_image_ids=deterministic_bottom_k(
                rows, 16, "imagenetr50-proxy-v1"
            ),
            repair_image_ids=(),
            source_priority_hash=record_sha256(
                [{"image_id": row.image_id, "priority": row.priority} for row in rows]
            ),
            training_optimizer_steps=0,
        )
        bundle = NodeBundle(artifact, stage_adapter, head, source)
        bundles.append(bundle)
        snapshots.append(
            MaterializedSnapshot(stage, (artifact.content_hash,), (artifact.content_hash,))
        )
    return TreeBuildResult(
        _dummy_policy(rank), tuple(bundles), tuple(snapshots), 0, 0, 0, True
    )


__all__ = ["baseline_history_tree", "leaf_bank_tree"]
