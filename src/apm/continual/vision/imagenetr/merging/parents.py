"""Immutable pairwise parent construction for all primary merge families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
import math
import shutil
import time

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.artifacts import (
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    NodeBundle,
    VisionStore,
    load_node_bundle,
    publish_artifact_directory,
    write_node_work_directory,
)
from apm.continual.vision.imagenetr.bank import MergeEvent
from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    ManifestDataset,
    deterministic_bottom_k,
)
from apm.continual.vision.imagenetr.heads import ClassifierRows, union_classifier_rows
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    frobenius_inner,
    exact_weighted_factors,
    normalized_weights,
    update_cosine,
)
from apm.continual.vision.imagenetr.merging.core_tsv import (
    CoreTsvResult,
    merge_module_states as core_tsv_module_states,
)
from apm.continual.vision.imagenetr.merging.output_drift import (
    OutputDriftResult,
    merge_module_states as output_drift_module_states,
)
from apm.continual.vision.imagenetr.merging.svd import merge_module_states as svd_module_states
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.protocol import MergePolicy
from apm.continual.vision.imagenetr.repair import repair_parent
from apm.continual.vision.imagenetr.training import train_adapter_model


@dataclass(frozen=True, slots=True)
class ParentExecution:
    """One policy parent plus cache/reuse and new optimizer-work evidence."""

    bundle: NodeBundle
    merge_cache_reused: bool
    policy_node_reused: bool
    optimizer_steps_this_execution: int


def _require_children(event: MergeEvent, left: NodeBundle, right: NodeBundle) -> None:
    if (
        left.artifact.first_task != event.left.first_task
        or left.artifact.last_task != event.left.last_task
        or right.artifact.first_task != event.right.first_task
        or right.artifact.last_task != event.right.last_task
        or left.artifact.level != right.artifact.level
        or set(left.artifact.represented_class_ids) & set(right.artifact.represented_class_ids)
    ):
        raise ValueError("merge artifacts differ from the logical oldest-first event")


def _source_metadata(
    manifest: DatasetManifest,
    event: MergeEvent,
    proxy_size: int,
) -> tuple[tuple[ImageRecord, ...], tuple[str, ...], str]:
    rows = manifest.select("train", event.parent.task_ids)
    proxy_ids = deterministic_bottom_k(rows, proxy_size, "imagenetr50-proxy-v1")
    priority_hash = record_sha256(
        [
            {"image_id": row.image_id, "priority": row.priority}
            for row in sorted(rows, key=lambda value: value.image_id)
        ]
    )
    return rows, proxy_ids, priority_hash


def _aggregate_diagnostics(
    method: str,
    left: NodeBundle,
    right: NodeBundle,
    results: Mapping[str, object],
    weights: tuple[float, ...],
    merge_scale: float,
    wall_seconds: float,
) -> dict[str, object]:
    module_rows = {}
    for module, result in sorted(results.items()):
        diagnostic = result[1] if isinstance(result, tuple) else result.diagnostics
        row = diagnostic.as_record()
        if isinstance(result, OutputDriftResult):
            row.update(
                {
                    "output_drift_spectrum": list(result.output_singular_values),
                    "retained_output_energy": result.retained_output_energy,
                }
            )
        if isinstance(result, CoreTsvResult):
            row.update(
                {
                    "core_dimensions": [
                        result.left_basis.shape[1],
                        result.right_basis.shape[1],
                    ],
                    "precompression_core_rank": int(
                        torch.linalg.matrix_rank(result.merged_core).item()
                    ),
                    "tsv_singular_values": [
                        float(value)
                        for value in result.merged_core_singular_values.detach().cpu().tolist()
                    ],
                }
            )
        module_rows[module] = row
    left_norm_squared = math.fsum(
        float(frobenius_inner(value, value).item()) for value in left.adapter.values()
    )
    right_norm_squared = math.fsum(
        float(frobenius_inner(value, value).item()) for value in right.adapter.values()
    )
    mean_cosine = math.fsum(
        update_cosine(left.adapter[module], right.adapter[module])
        for module in left.adapter
    ) / len(left.adapter)
    return {
        "child_a_update_norm": math.sqrt(max(0.0, left_norm_squared)),
        "child_b_update_norm": math.sqrt(max(0.0, right_norm_squared)),
        "child_update_cosine": mean_cosine,
        "merge_method": method,
        "merge_scale": merge_scale,
        "merge_wall_seconds": wall_seconds,
        "module_diagnostics": module_rows,
        "output_rank": next(iter(results.values()))[0].rank
        if isinstance(next(iter(results.values())), tuple)
        else next(iter(results.values())).factors.rank,
        "weights": list(weights),
    }


def _save_intermediates(path: Path, results: Mapping[str, object]) -> None:
    tensors: dict[str, Tensor] = {}
    for module, result in sorted(results.items()):
        if isinstance(result, CoreTsvResult):
            tensors.update(
                {
                    f"{module}.left_basis": result.left_basis,
                    f"{module}.right_basis": result.right_basis,
                    f"{module}.merged_core": result.merged_core,
                    **{
                        f"{module}.aligned_core_{index}": core
                        for index, core in enumerate(result.aligned_cores)
                    },
                }
            )
    if tensors:
        try:
            from safetensors.torch import save_file
        except ImportError as error:  # pragma: no cover - vision environment gate
            raise RuntimeError("safetensors is required by the vision environment") from error
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
            path / "merge_intermediates.safetensors",
            metadata={"schema_version": "imagenetr50-merge-intermediates-v1"},
        )


def _parameter_merge(
    method: str,
    left: NodeBundle,
    right: NodeBundle,
    weights: tuple[float, float],
    config: ImageNetRConfig,
    policy: MergePolicy,
    activations: Mapping[str, Tensor] | None,
) -> tuple[dict[str, LoRAFactors], dict[str, object], dict[str, object]]:
    children = (left.adapter, right.adapter)
    started = time.monotonic()
    if method == "svd":
        factors, diagnostics = svd_module_states(
            children, weights, policy.output_rank, 1.0, policy.scale
        )
        results: dict[str, object] = {
            module: (factors[module], diagnostics[module]) for module in factors
        }
    elif method == "core_tsv":
        results = core_tsv_module_states(
            children, weights, policy.output_rank, 1.0, policy.scale
        )
        factors = {module: result.factors for module, result in results.items()}
    elif method == "output_drift":
        if activations is None:
            raise ValueError("output-drift merge requires frozen proxy activations")
        results = output_drift_module_states(
            children, weights, activations, policy.output_rank, 1.0, policy.scale
        )
        factors = {module: result.factors for module, result in results.items()}
    else:
        raise ValueError("parameter merge method is not supported")
    diagnostics_record = _aggregate_diagnostics(
        method,
        left,
        right,
        results,
        weights,
        policy.scale,
        time.monotonic() - started,
    )
    return factors, results, diagnostics_record


def _proxy_metrics(
    adapter: Mapping[str, LoRAFactors],
    classifier: ClassifierRows,
    rank: int,
    alpha: int,
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: tuple[ImageRecord, ...],
    transform: object,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Measure deterministic training-proxy accuracy/loss for one parent function."""
    if not rows:
        raise ValueError("proxy metrics require classifier rows and training proxies")
    model = AdapterVisionModel(
        backbone_factory(),
        classifier.class_ids,
        rank,
        alpha,
        0.0,
        0,
        classifier,
    )
    load_adapter_factors(model, adapter)
    model.to(device).eval()
    dataset = ManifestDataset(prepared_root, rows, transform, 0, 0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = examples = 0
    loss_sum = 0.0
    classes = torch.tensor(classifier.class_ids, device=device, dtype=torch.long)
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = F.cross_entropy(
                    logits, model.classifier.local_targets(labels), reduction="sum"
                )
            correct += int(torch.sum(classes[torch.argmax(logits, dim=1)] == labels).item())
            examples += labels.numel()
            loss_sum += float(loss.item())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 100.0 * correct / examples, loss_sum / examples


def build_parent(
    store: VisionStore,
    event: MergeEvent,
    left: NodeBundle,
    right: NodeBundle,
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
    show_progress: bool = True,
) -> ParentExecution:
    """Build/reuse one un-repaired merge then optionally derive bounded repair."""
    _require_children(event, left, right)
    if policy.method == "core_tsv" and policy.core_space_revision is None:
        raise ValueError("Core+TSV policy lacks its pinned source revision")
    weights = normalized_weights(
        (
            left.artifact.represented_train_image_count,
            right.artifact.represented_train_image_count,
        )
    )
    merge_identity = record_sha256(
        {
            "children": [left.artifact.content_hash, right.artifact.content_hash],
            "merge_cache_policy": policy.merge_cache_hash,
            "schema_version": "imagenetr50-parameter-merge-job-v1",
        }
    )
    cache_target = store.merge_pair(
        left.artifact.content_hash, right.artifact.content_hash, merge_identity
    )
    rows, proxy_ids, priority_hash = _source_metadata(
        manifest, event, policy.proxy_size
    )
    activation_ids = tuple(
        sorted(set(left.artifact.proxy_image_ids + right.artifact.proxy_image_ids))
    )
    proxy_rows = manifest.select("train", image_ids=activation_ids)
    cache_reused = cache_target.is_dir()
    if cache_reused:
        un_repaired = load_node_bundle(cache_target)
    else:
        classifier = union_classifier_rows((left.classifier, right.classifier))
        diagnostics_record: dict[str, object]
        intermediates: dict[str, object] = {}
        if policy.method == "retrain_union":
            model = AdapterVisionModel(
                backbone_factory(),
                classifier.class_ids,
                config.lora_rank,
                config.lora_alpha,
                config.lora_dropout,
                config.seed + 300_000 + event.sequence,
                classifier,
            )
            result = train_adapter_model(
                model,
                prepared_root,
                rows,
                train_transform,
                config.leaf_training,
                config.seed + 400_000 + event.sequence,
                device,
                store.run / "checkpoints" / f"retrain_{merge_identity}.pt",
                num_workers=config.num_workers,
                checkpoint_steps=config.checkpoint_steps,
                show_progress=show_progress,
            )
            adapter, classifier, optimizer_steps = (
                result.adapter,
                result.classifier,
                result.optimizer_steps,
            )
            diagnostics_record = {
                "merge_image_presentations": result.image_presentations,
                "merge_method": policy.method,
                "merge_scale": policy.scale,
                "merge_wall_seconds": result.wall_seconds,
                "optimizer_steps": result.optimizer_steps,
                "output_rank": policy.output_rank,
                "peak_vram_bytes": result.peak_vram_bytes,
                "weights": list(weights),
            }
        else:
            activations = (
                activation_provider(activation_ids)
                if policy.method == "output_drift" and activation_provider is not None
                else None
            )
            adapter, intermediates, diagnostics_record = _parameter_merge(
                policy.method, left, right, weights, config, policy, activations
            )
            optimizer_steps = 0
        raw_proxy_accuracy, raw_proxy_loss = _proxy_metrics(
            adapter,
            classifier,
            config.lora_rank,
            config.lora_alpha,
            backbone_factory,
            prepared_root,
            proxy_rows,
            proxy_transform,
            config.leaf_training.batch_size,
            device,
        )
        diagnostics_record.update(
            {
                "raw_parent_proxy_accuracy": raw_proxy_accuracy,
                "raw_parent_proxy_loss": raw_proxy_loss,
            }
        )
        if policy.method == "retrain_union":
            diagnostics_record["retrained_parent_proxy_accuracy"] = raw_proxy_accuracy
        if policy.method == "output_drift":
            exact_adapter = {
                module: exact_weighted_factors(
                    (left.adapter[module], right.adapter[module]), weights, policy.scale
                )
                for module in sorted(left.adapter)
            }
            pre_accuracy, pre_loss = _proxy_metrics(
                exact_adapter,
                union_classifier_rows((left.classifier, right.classifier)),
                2 * config.lora_rank,
                2 * config.lora_rank,
                backbone_factory,
                prepared_root,
                proxy_rows,
                proxy_transform,
                config.leaf_training.batch_size,
                device,
            )
            diagnostics_record.update(
                {
                    "preprojection_proxy_accuracy": pre_accuracy,
                    "preprojection_proxy_loss": pre_loss,
                    "postprojection_proxy_loss": raw_proxy_loss,
                }
            )
        work = store.run / "work" / f"merge_{merge_identity}"
        shutil.rmtree(work, ignore_errors=True)
        artifact = write_node_work_directory(
            work,
            adapter,
            classifier,
            {
                "run_hash": store.run_hash,
                "software_manifest_hash": software_manifest_hash,
                "git_commit": git_commit,
                "creation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "level": event.parent.level,
                "first_task": event.parent.first_task,
                "last_task": event.parent.last_task,
                "represented_task_ids": event.parent.task_ids,
                "represented_class_ids": tuple(sorted(classifier.class_ids)),
                "represented_train_image_count": len(rows),
                "parent_hashes": (
                    left.artifact.content_hash,
                    right.artifact.content_hash,
                ),
                "unrepaired_parent_hash": None,
                "consolidation_method": policy.method,
                "consolidation_config_hash": merge_identity,
                "repair_config_hash": record_sha256(
                    {"fraction": 0.0, "schema_version": "imagenetr50-no-repair-v1"}
                ),
                "proxy_image_ids": proxy_ids,
                "repair_image_ids": (),
                "source_priority_hash": priority_hash,
                "training_optimizer_steps": optimizer_steps,
            },
        )
        publish_immutable_json(
            work / "merge_diagnostics.json",
            {
                **diagnostics_record,
                "child_a_interval": event.left.one_based_interval,
                "child_b_interval": event.right.one_based_interval,
                "level": event.parent.level,
                "merge_id": event.merge_id,
                "parent_interval": event.parent.one_based_interval,
                "parent_content_hash": artifact.content_hash,
                "policy_hash": policy.merge_cache_hash,
                "proxy_image_count": len(
                    set(left.artifact.proxy_image_ids + right.artifact.proxy_image_ids)
                ),
                "represented_classes": len(classifier.class_ids),
                "represented_images": len(rows),
                "represented_tasks": len(event.parent.task_ids),
                "schema_version": "imagenetr50-merge-diagnostics-v1",
            },
        )
        _save_intermediates(work, intermediates)
        publish_artifact_directory(work, cache_target)
        shutil.rmtree(work)
        un_repaired = load_node_bundle(cache_target)

    tree_target = store.tree_node(policy.content_hash, event.parent.node_id)
    if tree_target.is_dir():
        return ParentExecution(load_node_bundle(tree_target), cache_reused, True, 0)
    if policy.repair_fraction == 0.0 or policy.method == "retrain_union":
        publish_artifact_directory(cache_target, tree_target)
        return ParentExecution(
            load_node_bundle(tree_target),
            cache_reused,
            False,
            0 if cache_reused else un_repaired.artifact.training_optimizer_steps,
        )

    repaired, repair_ids = repair_parent(
        un_repaired,
        manifest,
        prepared_root,
        config,
        policy.repair_fraction,
        backbone_factory,
        train_transform,
        device,
        store.run / "checkpoints" / f"repair_{policy.content_hash}_{event.parent.node_id}.pt",
        show_progress,
    )
    repaired_proxy_accuracy, repaired_proxy_loss = _proxy_metrics(
        repaired.adapter,
        repaired.classifier,
        config.lora_rank,
        config.lora_alpha,
        backbone_factory,
        prepared_root,
        proxy_rows,
        proxy_transform,
        config.leaf_training.batch_size,
        device,
    )
    work = store.run / "work" / f"repair_{policy.content_hash}_{event.parent.node_id}"
    shutil.rmtree(work, ignore_errors=True)
    artifact = write_node_work_directory(
        work,
        repaired.adapter,
        repaired.classifier,
        {
            "run_hash": store.run_hash,
            "software_manifest_hash": software_manifest_hash,
            "git_commit": git_commit,
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "level": event.parent.level,
            "first_task": event.parent.first_task,
            "last_task": event.parent.last_task,
            "represented_task_ids": event.parent.task_ids,
            "represented_class_ids": un_repaired.artifact.represented_class_ids,
            "represented_train_image_count": un_repaired.artifact.represented_train_image_count,
            "parent_hashes": un_repaired.artifact.parent_hashes,
            "unrepaired_parent_hash": un_repaired.artifact.content_hash,
            "consolidation_method": f"{policy.method}_repair",
            "consolidation_config_hash": un_repaired.artifact.consolidation_config_hash,
            "repair_config_hash": policy.repair_config_hash,
            "proxy_image_ids": un_repaired.artifact.proxy_image_ids,
            "repair_image_ids": repair_ids,
            "source_priority_hash": un_repaired.artifact.source_priority_hash,
            "training_optimizer_steps": repaired.optimizer_steps,
        },
    )
    publish_immutable_json(
        work / "repair_metrics.json",
        {
            "final_loss": repaired.final_loss,
            "image_presentations": repaired.image_presentations,
            "optimizer_steps": repaired.optimizer_steps,
            "parent_content_hash": artifact.content_hash,
            "peak_vram_bytes": repaired.peak_vram_bytes,
            "raw_parent_proxy_accuracy": load_canonical_json(
                cache_target / "merge_diagnostics.json"
            )["raw_parent_proxy_accuracy"],
            "repair_fraction": policy.repair_fraction,
            "repaired_parent_proxy_accuracy": repaired_proxy_accuracy,
            "repaired_parent_proxy_loss": repaired_proxy_loss,
            "repair_wall_seconds": repaired.wall_seconds,
            "schema_version": "imagenetr50-repair-metrics-v1",
        },
    )
    publish_artifact_directory(work, tree_target)
    shutil.rmtree(work)
    return ParentExecution(
        load_node_bundle(tree_target), cache_reused, False, repaired.optimizer_steps
    )


__all__ = ["ParentExecution", "build_parent"]
