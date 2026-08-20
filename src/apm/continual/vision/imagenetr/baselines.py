"""Precisely controlled frozen-head, sequential-LoRA, and joint-IID baselines."""

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
from apm.continual.vision.imagenetr.heads import (
    AffineClassifier,
    ClassifierRows,
    save_classifier,
    union_classifier_rows,
)
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.protocol import JobSpec
from apm.continual.vision.imagenetr.training import TrainingResult, train_adapter_model


@dataclass(frozen=True, slots=True)
class BaselineExecution:
    """One controlled baseline artifact and optimizer-work reuse evidence."""

    name: str
    bundle: NodeBundle
    reused: bool
    optimizer_steps_this_execution: int


def baseline_job_spec(
    name: str,
    store: VisionStore,
    manifest: DatasetManifest,
    config: ImageNetRConfig,
) -> JobSpec:
    return JobSpec.create(
        store.run_hash,
        f"train_{name}",
        payload={
            "dataset_hash": manifest.content_hash,
            "lora_alpha": config.lora_alpha,
            "lora_rank": config.lora_rank,
            "name": name,
            "training": asdict(
                config.joint_training if name == "joint_iid_lora_r16" else config.leaf_training
            ),
        },
    )


def _baseline_metadata(
    store: VisionStore,
    manifest: DatasetManifest,
    config: ImageNetRConfig,
    job_hash: str,
    name: str,
    optimizer_steps: int,
    software_manifest_hash: str,
    git_commit: str,
) -> dict[str, object]:
    rows = manifest.select("train")
    return {
        "run_hash": store.run_hash,
        "software_manifest_hash": software_manifest_hash,
        "git_commit": git_commit,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "level": 0,
        "first_task": 0,
        "last_task": 49,
        "represented_task_ids": tuple(range(50)),
        "represented_class_ids": tuple(range(200)),
        "represented_train_image_count": len(rows),
        "parent_hashes": (),
        "unrepaired_parent_hash": None,
        "consolidation_method": "baseline",
        "consolidation_config_hash": job_hash,
        "repair_config_hash": record_sha256(
            {"fraction": 0.0, "schema_version": "imagenetr50-baseline-no-repair-v1"}
        ),
        "proxy_image_ids": deterministic_bottom_k(
            rows, config.proxy_images_per_node, "imagenetr50-proxy-v1"
        ),
        "repair_image_ids": (),
        "source_priority_hash": record_sha256(
            [
                {"image_id": row.image_id, "priority": row.priority}
                for row in sorted(rows, key=lambda value: value.image_id)
            ]
        ),
        "training_optimizer_steps": optimizer_steps,
    }


def _publish_baseline(
    name: str,
    store: VisionStore,
    manifest: DatasetManifest,
    config: ImageNetRConfig,
    job: JobSpec,
    result: TrainingResult,
    stage_states: tuple[tuple[dict[str, LoRAFactors], ClassifierRows], ...],
    software_manifest_hash: str,
    git_commit: str,
) -> BaselineExecution:
    target = store.baseline(name, job.content_hash)
    work = store.run / "work" / f"baseline_{name}_{job.content_hash}"
    shutil.rmtree(work, ignore_errors=True)
    artifact = write_node_work_directory(
        work,
        result.adapter,
        result.classifier,
        _baseline_metadata(
            store,
            manifest,
            config,
            job.content_hash,
            name,
            result.optimizer_steps,
            software_manifest_hash,
            git_commit,
        ),
    )
    for stage, (adapter, classifier) in enumerate(stage_states, start=1):
        snapshot = work / "stage_snapshots" / f"stage_{stage:03d}"
        save_adapter(snapshot / "adapter.safetensors", adapter)
        save_classifier(snapshot / "classifier.safetensors", classifier)
    publish_immutable_json(
        work / "training_metrics.json",
        {
            "final_loss": result.final_loss,
            "image_presentations": result.image_presentations,
            "optimizer_steps": result.optimizer_steps,
            "peak_vram_bytes": result.peak_vram_bytes,
            "schema_version": "imagenetr50-baseline-training-metrics-v1",
            "wall_seconds": result.wall_seconds,
        },
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    bundle = load_node_bundle(target)
    if bundle.artifact.content_hash != artifact.content_hash:
        raise ValueError("published controlled baseline identity changed")
    return BaselineExecution(name, bundle, False, result.optimizer_steps)


def train_frozen_reference(
    store: VisionStore,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    device: torch.device,
    software_manifest_hash: str,
    git_commit: str,
    show_progress: bool = True,
) -> BaselineExecution:
    """Train 50 independent four-row heads on the same frozen representation and union."""
    name = "frozen_reference"
    job = baseline_job_spec(name, store, manifest, config)
    target = store.baseline(name, job.content_hash)
    if target.is_dir():
        return BaselineExecution(name, load_node_bundle(target), True, 0)
    model = AdapterVisionModel(
        backbone_factory(), (0, 1, 2, 3), config.lora_rank, config.lora_alpha, 0.0, config.seed
    )
    heads: list[ClassifierRows] = []
    stages: list[tuple[dict[str, LoRAFactors], ClassifierRows]] = []
    total_steps = total_presentations = 0
    total_wall = 0.0
    peak = 0
    final_loss = float("nan")
    for task in range(config.tasks):
        class_ids = tuple(range(4 * task, 4 * task + 4))
        model.classifier = AffineClassifier(
            class_ids, 768, config.seed + 10_000 + task
        )
        task_result = train_adapter_model(
            model,
            prepared_root,
            manifest.select("train", (task,)),
            train_transform,
            config.leaf_training,
            config.seed + 20_000 + task,
            device,
            store.run / "checkpoints" / f"frozen_head_{task:03d}_{job.content_hash}.pt",
            train_lora=False,
            num_workers=config.num_workers,
            checkpoint_steps=config.checkpoint_steps,
            show_progress=show_progress,
        )
        heads.append(task_result.classifier)
        union = heads[0] if len(heads) == 1 else union_classifier_rows(tuple(heads))
        stages.append((task_result.adapter, union))
        total_steps += task_result.optimizer_steps
        total_presentations += task_result.image_presentations
        total_wall += task_result.wall_seconds
        peak = max(peak, task_result.peak_vram_bytes)
        final_loss = task_result.final_loss
    final = TrainingResult(
        stages[-1][0],
        stages[-1][1],
        total_steps,
        total_presentations,
        total_wall,
        final_loss,
        peak,
    )
    return _publish_baseline(
        name,
        store,
        manifest,
        config,
        job,
        final,
        tuple(stages),
        software_manifest_hash,
        git_commit,
    )


def train_sequential_lora(
    store: VisionStore,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    device: torch.device,
    software_manifest_hash: str,
    git_commit: str,
    show_progress: bool = True,
) -> BaselineExecution:
    """Warm-start one adapter while only each arriving task's four head rows train."""
    name = "seq_lora_r16"
    job = baseline_job_spec(name, store, manifest, config)
    target = store.baseline(name, job.content_hash)
    if target.is_dir():
        return BaselineExecution(name, load_node_bundle(target), True, 0)
    model = AdapterVisionModel(
        backbone_factory(), (0, 1, 2, 3), config.lora_rank, config.lora_alpha, 0.0, config.seed
    )
    stages: list[tuple[dict[str, LoRAFactors], ClassifierRows]] = []
    total_steps = total_presentations = 0
    total_wall = 0.0
    peak = 0
    final_loss = float("nan")
    for task in range(config.tasks):
        current_classes = tuple(range(4 * task, 4 * task + 4))
        if task:
            new_rows = AffineClassifier(
                current_classes, 768, config.seed + 30_000 + task
            ).to(device).rows()
            expanded = union_classifier_rows((model.classifier.rows(), new_rows))
            model.classifier = AffineClassifier(
                expanded.class_ids, 768, initial_rows=expanded
            )
        task_result = train_adapter_model(
            model,
            prepared_root,
            manifest.select("train", (task,)),
            train_transform,
            config.leaf_training,
            config.seed + 40_000 + task,
            device,
            store.run / "checkpoints" / f"sequential_{task:03d}_{job.content_hash}.pt",
            active_class_ids=current_classes,
            num_workers=config.num_workers,
            checkpoint_steps=config.checkpoint_steps,
            show_progress=show_progress,
        )
        stages.append((task_result.adapter, task_result.classifier))
        total_steps += task_result.optimizer_steps
        total_presentations += task_result.image_presentations
        total_wall += task_result.wall_seconds
        peak = max(peak, task_result.peak_vram_bytes)
        final_loss = task_result.final_loss
    final = TrainingResult(
        stages[-1][0],
        stages[-1][1],
        total_steps,
        total_presentations,
        total_wall,
        final_loss,
        peak,
    )
    return _publish_baseline(
        name,
        store,
        manifest,
        config,
        job,
        final,
        tuple(stages),
        software_manifest_hash,
        git_commit,
    )


def train_joint_iid_lora(
    store: VisionStore,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    device: torch.device,
    software_manifest_hash: str,
    git_commit: str,
    show_progress: bool = True,
) -> BaselineExecution:
    """Train one fresh rank-16 adapter and 200-way head jointly for five epochs."""
    name = "joint_iid_lora_r16"
    job = baseline_job_spec(name, store, manifest, config)
    target = store.baseline(name, job.content_hash)
    if target.is_dir():
        return BaselineExecution(name, load_node_bundle(target), True, 0)
    model = AdapterVisionModel(
        backbone_factory(), tuple(range(200)), config.lora_rank, config.lora_alpha, 0.0, config.seed
    )
    result = train_adapter_model(
        model,
        prepared_root,
        manifest.select("train"),
        train_transform,
        config.joint_training,
        config.seed + 50_000,
        device,
        store.run / "checkpoints" / f"joint_{job.content_hash}.pt",
        num_workers=config.num_workers,
        checkpoint_steps=config.checkpoint_steps,
        show_progress=show_progress,
    )
    return _publish_baseline(
        name,
        store,
        manifest,
        config,
        job,
        result,
        ((result.adapter, result.classifier),),
        software_manifest_hash,
        git_commit,
    )


__all__ = [
    "BaselineExecution",
    "baseline_job_spec",
    "train_frozen_reference",
    "train_joint_iid_lora",
    "train_sequential_lora",
]
