"""Train, seal, discover, and reuse immutable independent ImageNet-R leaves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
import shutil

import torch

from apm.continual.artifacts import publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.artifacts import (
    NodeBundle,
    VisionStore,
    load_node_bundle,
    publish_artifact_directory,
    write_node_work_directory,
)
from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.data import DatasetManifest, deterministic_bottom_k
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.protocol import JobSpec
from apm.continual.vision.imagenetr.training import train_adapter_model


@dataclass(frozen=True, slots=True)
class LeafExecution:
    """One leaf artifact plus whether this invocation performed optimizer work."""

    bundle: NodeBundle
    reused: bool
    optimizer_steps_this_execution: int


def leaf_job_spec(
    run_hash: str,
    task_index: int,
    manifest: DatasetManifest,
    config: ImageNetRConfig,
) -> JobSpec:
    """Bind one leaf job to its exact source identities and training policy."""
    rows = manifest.select("train", (task_index,))
    return JobSpec.create(
        run_hash,
        "train_leaf",
        payload={
            "class_ids": list(range(4 * task_index, 4 * task_index + 4)),
            "initialization_seed": config.seed + 1000 * task_index,
            "source_ids_hash": record_sha256([row.image_id for row in rows]),
            "task_index": task_index,
            "training": asdict(config.leaf_training),
        },
    )


def train_leaf(
    store: VisionStore,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    task_index: int,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    device: torch.device,
    software_manifest_hash: str,
    git_commit: str,
    show_progress: bool = True,
) -> LeafExecution:
    """Train one fresh base-relative task leaf or validate and reuse it unchanged."""
    if not 0 <= task_index < config.tasks:
        raise ValueError("leaf task index is outside the resolved protocol")
    job = leaf_job_spec(store.run_hash, task_index, manifest, config)
    target = store.leaf_job(task_index, job.content_hash)
    if target.is_dir():
        return LeafExecution(load_node_bundle(target), True, 0)
    rows = manifest.select("train", (task_index,))
    class_ids = tuple(range(4 * task_index, 4 * task_index + 4))
    initialization_seed = config.seed + 1000 * task_index
    model = AdapterVisionModel(
        backbone_factory(),
        class_ids,
        config.lora_rank,
        config.lora_alpha,
        config.lora_dropout,
        initialization_seed,
    )
    checkpoint = store.run / "checkpoints" / f"leaf_{task_index:03d}_{job.content_hash}.pt"
    result = train_adapter_model(
        model,
        prepared_root,
        rows,
        train_transform,
        config.leaf_training,
        initialization_seed,
        device,
        checkpoint,
        num_workers=config.num_workers,
        checkpoint_steps=config.checkpoint_steps,
        show_progress=show_progress,
    )
    proxy_ids = deterministic_bottom_k(
        rows, config.proxy_images_per_node, "imagenetr50-proxy-v1"
    )
    source_priority_hash = record_sha256(
        [
            {"image_id": row.image_id, "priority": row.priority}
            for row in sorted(rows, key=lambda value: value.image_id)
        ]
    )
    work = store.run / "work" / f"leaf_{task_index:03d}_{job.content_hash}"
    shutil.rmtree(work, ignore_errors=True)
    artifact = write_node_work_directory(
        work,
        result.adapter,
        result.classifier,
        {
            "run_hash": store.run_hash,
            "software_manifest_hash": software_manifest_hash,
            "git_commit": git_commit,
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "level": 0,
            "first_task": task_index,
            "last_task": task_index,
            "represented_task_ids": (task_index,),
            "represented_class_ids": class_ids,
            "represented_train_image_count": len(rows),
            "parent_hashes": (),
            "unrepaired_parent_hash": None,
            "consolidation_method": "leaf",
            "consolidation_config_hash": job.content_hash,
            "repair_config_hash": record_sha256(
                {"fraction": 0.0, "schema_version": "imagenetr50-leaf-no-repair-v1"}
            ),
            "proxy_image_ids": proxy_ids,
            "repair_image_ids": (),
            "source_priority_hash": source_priority_hash,
            "training_optimizer_steps": result.optimizer_steps,
        },
    )
    publish_immutable_json(
        work / "source_image_ids.json",
        {
            "image_ids": [row.image_id for row in rows],
            "schema_version": "imagenetr50-leaf-source-images-v1",
            "source_ids_hash": record_sha256([row.image_id for row in rows]),
        },
    )
    publish_immutable_json(
        work / "training_metrics.json",
        {
            "final_loss": result.final_loss,
            "image_presentations": result.image_presentations,
            "optimizer_steps": result.optimizer_steps,
            "peak_vram_bytes": result.peak_vram_bytes,
            "schema_version": "imagenetr50-leaf-training-metrics-v1",
            "wall_seconds": result.wall_seconds,
        },
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    bundle = load_node_bundle(target)
    if bundle.artifact.content_hash != artifact.content_hash:
        raise ValueError("published leaf semantic identity changed")
    return LeafExecution(bundle, False, result.optimizer_steps)


__all__ = ["LeafExecution", "leaf_job_spec", "train_leaf"]
