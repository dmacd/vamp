"""Phase-gated local workflow for the ImageNet-R LogT prediction integrator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
import math
import time

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.artifacts import NodeBundle, load_node_bundle
from apm.continual.vision.imagenetr.calibration import (
    CalibrationExamples,
    fit_affine_calibration,
)
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.integrator_artifacts import (
    HierarchyPolicy,
    IntegratorBootstrap,
    bootstrap_integrator,
    hierarchy_policy,
)
from apm.continual.vision.imagenetr.integrator_bank import class_stratified_reservoir
from apm.continual.vision.imagenetr.integrator_hierarchy import (
    HierarchyBuildResult,
    build_hierarchy,
)
from apm.continual.vision.imagenetr.integrator_model import (
    IntegratorFitResult,
    IntegratorState,
    IntegratorSupervision,
    create_integrator_state,
    fit_fresh_integrator,
    fit_integrator_epochs,
    parameter_count,
    predict_integrator,
    select_smallest_near_best,
)
from apm.continual.vision.imagenetr.integrator_observations import (
    BehaviorNode,
    FrontierTensors,
    accuracy,
    build_frontier_tensors,
)
from apm.continual.vision.imagenetr.integrator_persistence import (
    load_integrator_fit,
    publish_integrator_fit,
    restore_integrator_checkpoint,
    save_integrator_checkpoint,
)
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.router_features import test_transform_hash


DEFAULT_INTEGRATOR_CONFIG = Path(
    "configs/vision/imagenetr/logt_prediction_integrator_v1.yaml"
)
HISTORICAL_NAMESPACE = "imagenetr50-integrator-history-bottom-k-v1"


def _write_state(bootstrap: IntegratorBootstrap, phase: str, **values: object) -> None:
    atomic_write(
        bootstrap.store.run / "state" / "workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-integrator-workflow-state-v1",
                **values,
            }
        ),
    )


def _backbone_factory(bootstrap: IntegratorBootstrap) -> Callable[[], torch.nn.Module]:
    return lambda: create_pinned_backbone(bootstrap.checkpoint)


def _partition_rows(
    bootstrap: IntegratorBootstrap,
    partition: str,
    tasks: Sequence[int],
) -> tuple[ImageRecord, ...]:
    rows = bootstrap.manifest.select("train", tasks)
    if partition == "fit":
        allowed = frozenset(bootstrap.split.fit_image_ids)
        rows = tuple(row for row in rows if row.image_id in allowed)
    elif partition == "validation":
        allowed = frozenset(bootstrap.split.validation_image_ids)
        rows = tuple(row for row in rows if row.image_id in allowed)
    elif partition != "all_train":
        raise ValueError("unknown integrator data partition")
    if not rows:
        raise ValueError("integrator data partition is empty")
    return rows


def _test_rows(
    bootstrap: IntegratorBootstrap, tasks: Sequence[int]
) -> tuple[ImageRecord, ...]:
    rows = bootstrap.manifest.select("test", tasks)
    if not rows:
        raise ValueError("integrator test partition is empty")
    return rows


def _frontier_tensors(
    bootstrap: IntegratorBootstrap,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    maximum_slots: int,
    rows: Sequence[ImageRecord],
    device: torch.device,
    cache_family: str,
) -> FrontierTensors:
    started = time.monotonic()
    tensors = build_frontier_tensors(
        nodes,
        slots,
        maximum_slots,
        bootstrap.config.data_root / "imagenet-r",
        rows,
        bootstrap.test_transform,
        test_transform_hash(bootstrap.primary_config.input_size),
        bootstrap.protocol.model_manifest_hash,
        _backbone_factory(bootstrap),
        bootstrap.store.run / "cache" / "behaviors",
        bootstrap.primary_config.lora_rank,
        bootstrap.primary_config.lora_alpha,
        bootstrap.primary_config.cosine_scale,
        bootstrap.config.feature_batch_size,
        bootstrap.config.num_workers,
        device,
    )
    ChainedJsonlLedger(
        bootstrap.store.run / "ledgers" / "behavior_requests.jsonl",
        "imagenetr50-integrator-behavior-request-v1",
    ).append(
        {
            "base_example_forwards": tensors.base_example_forwards,
            "cache_family": cache_family,
            "cache_hits": tensors.cache_hits,
            "cache_misses": tensors.cache_misses,
            "elapsed_seconds": time.monotonic() - started,
            "examples": len(rows),
            "image_ids_hash": record_sha256([row.image_id for row in rows]),
            "node_example_forwards": tensors.node_example_forwards,
            "node_hashes": [node.node_hash for node in nodes],
            "slots": list(slots),
            "splits": sorted({row.split for row in rows}),
        }
    )
    return tensors


def _hierarchy_frontier(
    hierarchy: HierarchyBuildResult, stage: int
) -> tuple[tuple[BehaviorNode, ...], tuple[int, ...], str]:
    snapshot = hierarchy.snapshots[stage - 1]
    bundles = hierarchy.frontier(stage)
    return (
        tuple(BehaviorNode.from_bundle(bundle) for bundle in bundles),
        snapshot.levels,
        snapshot.content_hash,
    )


def _supervision(
    tensors: FrontierTensors,
    variant: str,
    selected_ids: Sequence[str] | None = None,
) -> IntegratorSupervision:
    observations = tensors.observations(variant)
    if selected_ids is None:
        return IntegratorSupervision(observations, tensors.labels)
    index = {image_id: position for position, image_id in enumerate(tensors.image_ids)}
    if len(set(selected_ids)) != len(selected_ids) or any(value not in index for value in selected_ids):
        raise ValueError("integrator supervision selects unknown or duplicate image IDs")
    indices = torch.tensor([index[value] for value in selected_ids], dtype=torch.int64)
    return IntegratorSupervision(observations.select(indices), tensors.labels[indices])


def _fit_or_load_fresh(
    bootstrap: IntegratorBootstrap,
    family: str,
    frontier_hash: str,
    tensors: FrontierTensors,
    variant: str,
    maximum_slots: int,
    fit_ids: Sequence[str],
    validation_ids: Sequence[str],
    seed: int,
    stage: int,
    device: torch.device,
) -> tuple[IntegratorState, IntegratorFitResult, Path]:
    config = bootstrap.config.optimization
    observations = tensors.observations(variant)
    index = {image_id: position for position, image_id in enumerate(tensors.image_ids)}
    fit_indices = torch.tensor([index[value] for value in fit_ids], dtype=torch.int64)
    validation_indices = torch.tensor(
        [index[value] for value in validation_ids], dtype=torch.int64
    )
    training = IntegratorSupervision(
        observations.select(fit_indices), tensors.labels[fit_indices]
    )
    validation = IntegratorSupervision(
        observations.select(validation_indices), tensors.labels[validation_indices]
    )
    del observations
    job_hash = record_sha256(
        {
            "family": family,
            "fit_ids_hash": record_sha256(list(fit_ids)),
            "frontier_hash": frontier_hash,
            "maximum_slots": maximum_slots,
            "optimization": asdict(config),
            "seed": seed,
            "stage": stage,
            "validation_ids_hash": record_sha256(list(validation_ids)),
            "variant": variant,
            "schema_version": "imagenetr50-integrator-fresh-job-v1",
        }
    )
    name = f"{family}-{variant}-stage{stage}-seed{seed}"
    state = create_integrator_state(name, maximum_slots, variant, config, seed, device)
    target = bootstrap.store.run / "integrators" / family / job_hash
    if target.is_dir():
        return state, load_integrator_fit(target, state), target
    result = fit_fresh_integrator(
        state,
        training,
        validation,
        config,
        seed,
        stage,
        device,
        progress=True,
        checkpoint_path=bootstrap.store.run / "checkpoints" / f"fresh_{job_hash}.pt",
        checkpoint_key=job_hash,
    )
    target = publish_integrator_fit(
        bootstrap.store,
        family,
        job_hash,
        state,
        result,
        {
            "fit_ids_hash": record_sha256(list(fit_ids)),
            "frontier_hash": frontier_hash,
            "maximum_slots": maximum_slots,
            "seed": seed,
            "stage": stage,
            "validation_ids_hash": record_sha256(list(validation_ids)),
            "variant": variant,
        },
    )
    return state, result, target


def _control_metrics(tensors: FrontierTensors) -> dict[str, float]:
    controls = {
        name: accuracy(predictions, tensors.labels)
        for name, predictions in tensors.control_predictions().items()
    }
    controls["true_node_oracle"] = accuracy(
        tensors.true_node_oracle_predictions(), tensors.labels
    )
    return controls


def _affine_control_accuracy(
    fitting: FrontierTensors,
    evaluation: FrontierTensors,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
) -> float:
    """Fit training-derived node scale/offsets and score an unlabeled evaluation view."""
    if fitting.active_slot_mask.tolist() != evaluation.active_slot_mask.tolist():
        raise ValueError("affine control frontiers differ between fitting and evaluation")
    raw_by_node = {
        node.node_hash: fitting.raw_scores[:, slot, fitting.ownership[slot]]
        for node, slot in zip(nodes, slots, strict=True)
    }
    class_ids = {node.node_hash: node.classifier.class_ids for node in nodes}
    calibration = fit_affine_calibration(
        CalibrationExamples(fitting.image_ids, fitting.labels),
        raw_by_node,
        class_ids,
    )
    scores = torch.full((len(evaluation.labels), 200), -torch.inf)
    for node, slot in zip(nodes, slots, strict=True):
        owned = evaluation.ownership[slot]
        temperature, offset = calibration.parameters_for(node.node_hash)
        scores[:, owned] = evaluation.raw_scores[:, slot, owned] / temperature + offset
    return accuracy(scores.argmax(dim=1), evaluation.labels)


def _sealed_static_validation(bootstrap: IntegratorBootstrap) -> dict[str, float]:
    source = (
        bootstrap.config.router_artifact_root
        / "runs"
        / bootstrap.config.sealed_router_run_hash
        / "reports"
        / "stage_metrics.json"
    )
    record = load_canonical_json(source)
    rows = tuple(
        row
        for row in record.get("rows", ())
        if row.get("split") == "validation"
        and row.get("inference_condition") == "I-U100"
        and int(row.get("stage", 0)) == 50
    )
    if not rows:
        raise ValueError("sealed router report lacks the validation controls")
    values = {
        f"sealed_{row['condition_id']}": float(row["routed_accuracy"])
        for row in rows
    }
    values["sealed_existing_task_free"] = max(
        float(row["existing_task_free_accuracy"]) for row in rows
    )
    return values


def _local_reference_results(bootstrap: IntegratorBootstrap) -> dict[str, float]:
    """Load the exact prior local controls bound into this run's protocol."""
    source = (
        bootstrap.config.inference_artifact_root
        / "runs"
        / bootstrap.config.sealed_run_hash
        / "reports"
        / "summary.json"
    )
    summary = load_canonical_json(source)
    if summary.get("schema_version") != "imagenetr50-summary-v1":
        raise ValueError("sealed ImageNet-R summary has an unknown schema")
    by_condition = {
        str(row["condition"]): row for row in summary.get("conditions", ())
    }
    required = {
        "frozen_reference",
        "seq_lora_r16",
        "joint_iid_lora_r16",
        "logt_retrain_union_r16",
    }
    if not required <= set(by_condition):
        raise ValueError("sealed ImageNet-R summary lacks a required local control")
    external = dict(summary["external_e2lora"])
    local_last = float(external["local_final_accuracy"])
    local_incremental = float(external["local_incremental_average_accuracy"])
    if (
        not bool(external.get("succeeded"))
        or abs(local_last - bootstrap.config.gates.local_e2_last) > 1.0e-9
        or abs(local_incremental - bootstrap.config.gates.local_e2_incremental) > 1.0e-9
    ):
        raise ValueError("configured E2-LoRA comparator differs from its sealed local run")
    return {
        "frozen_reference_last": float(
            by_condition["frozen_reference"]["raw_last_accuracy"]
        ),
        "joint_iid_last": float(
            by_condition["joint_iid_lora_r16"]["raw_last_accuracy"]
        ),
        "local_e2_incremental": local_incremental,
        "local_e2_last": local_last,
        "published_e2_incremental": float(
            external["published_incremental_average_accuracy"]
        ),
        "published_e2_last": float(external["published_final_accuracy"]),
        "sealed_logt_affine_last": float(
            by_condition["logt_retrain_union_r16"][
                "affine_calibrated_last_accuracy"
            ]
        ),
        "sealed_logt_oracle_last": float(
            by_condition["logt_retrain_union_r16"][
                "true_node_oracle_last_accuracy"
            ]
        ),
        "sealed_logt_raw_last": float(
            by_condition["logt_retrain_union_r16"]["raw_last_accuracy"]
        ),
        "sequential_last": float(
            by_condition["seq_lora_r16"]["raw_last_accuracy"]
        ),
    }


def run_preflight(bootstrap: IntegratorBootstrap, device: torch.device) -> dict[str, object]:
    """Check immutable boundaries, split isolation, GPU support, and zero parity."""
    target = bootstrap.store.run / "protocol" / "preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    fit, validation = set(bootstrap.split.fit_image_ids), set(bootstrap.split.validation_image_ids)
    train = {row.image_id for row in bootstrap.manifest.images if row.split == "train"}
    test = {row.image_id for row in bootstrap.manifest.images if row.split == "test"}
    inherited = load_canonical_json(
        bootstrap.config.inference_artifact_root
        / "runs"
        / bootstrap.config.sealed_run_hash
        / "protocol"
        / "preflight.json"
    )
    if (
        inherited.get("schema_version") != "imagenetr50-preflight-v1"
        or inherited.get("batch_size") != 64
        or not inherited.get("bf16_supported")
        or inherited.get("zero_lora_max_absolute_error") != 0
    ):
        raise RuntimeError("sealed ViT/ImageNet-R preflight evidence is invalid")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    state = create_integrator_state(
        "preflight",
        bootstrap.config.maximum_levels,
        "scores",
        bootstrap.config.optimization,
        bootstrap.config.seed,
        device,
    )
    feature_rows = torch.randn(3, state.model.input_dim, device=device)
    baseline = torch.randn(3, 200, device=device)
    mask = torch.ones(200, dtype=torch.bool, device=device)
    with torch.inference_mode():
        parity_error = float((state.model(feature_rows, baseline, mask) - baseline).abs().max().item())
    state.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        one_step_loss = torch.nn.functional.cross_entropy(
            state.model(feature_rows, baseline, mask),
            torch.tensor((0, 1, 2), device=device),
        )
    one_step_loss.backward()
    state.optimizer.step()
    core: dict[str, object] = {
        "bf16_supported": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "fit_images": len(fit),
        "fit_validation_overlap": len(fit & validation),
        "fit_validation_union_matches_train": fit | validation == train,
        "local_references": _local_reference_results(bootstrap),
        "one_step_loss": float(one_step_loss.detach().item()),
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "sealed_primary_preflight": inherited,
        "schema_version": "imagenetr50-integrator-preflight-v1",
        "test_training_overlap": len(test & train),
        "validation_images": len(validation),
        "zero_residual_max_error": parity_error,
    }
    if (
        device.type != "cuda"
        or not core["bf16_supported"]
        or core["fit_validation_overlap"] != 0
        or not core["fit_validation_union_matches_train"]
        or core["test_training_overlap"] != 0
        or parity_error != 0.0
        or not math.isfinite(float(core["one_step_loss"]))
    ):
        raise RuntimeError("integrator preflight failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_sealed_diagnostic(
    bootstrap: IntegratorBootstrap, device: torch.device
) -> dict[str, object]:
    """Fit all feature variants on the optimistic sealed U100 validation diagnostic."""
    target = bootstrap.store.run / "diagnostic" / "result.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[phase 1/6] Sealed U100 prediction-integrator capacity diagnostic", flush=True)
    ordered_refs = tuple(
        sorted(
            bootstrap.sealed_tree.final.nodes,
            key=lambda node: (node.logical_node.first_task, node.logical_node.last_task),
        )
    )
    nodes = tuple(BehaviorNode.from_sealed(node) for node in ordered_refs)
    rows = bootstrap.manifest.select("train")
    tensors = _frontier_tensors(
        bootstrap,
        nodes,
        tuple(range(len(nodes))),
        bootstrap.config.diagnostic_maximum_slots,
        rows,
        device,
        "sealed_u100",
    )
    fit_ids = tuple(image_id for image_id in tensors.image_ids if image_id in frozenset(bootstrap.split.fit_image_ids))
    validation_ids = tuple(
        image_id for image_id in tensors.image_ids if image_id in frozenset(bootstrap.split.validation_image_ids)
    )
    validation_tensors = tensors.select(validation_ids)
    controls = {**_sealed_static_validation(bootstrap), **_control_metrics(validation_tensors)}
    controls["affine_calibrated_union"] = _affine_control_accuracy(
        tensors.select(fit_ids),
        validation_tensors,
        nodes,
        tuple(range(len(nodes))),
    )
    run_rows: list[dict[str, object]] = []
    selected_accuracies: dict[str, float] = {}
    frontier_hash = record_sha256(
        {
            "node_hashes": [node.node_hash for node in nodes],
            "slots": list(range(len(nodes))),
            "schema_version": "imagenetr50-integrator-sealed-frontier-v1",
        }
    )
    for variant in bootstrap.config.feature_variants:
        variant_results = []
        for seed in bootstrap.config.replication_seeds:
            _state, result, artifact = _fit_or_load_fresh(
                bootstrap,
                "sealed_diagnostic",
                frontier_hash,
                tensors,
                variant,
                bootstrap.config.diagnostic_maximum_slots,
                fit_ids,
                validation_ids,
                seed,
                50,
                device,
            )
            row = {
                "artifact": str(artifact),
                "best_epoch": result.best_epoch,
                "epochs": result.epochs,
                "seed": seed,
                "validation_accuracy": result.validation_accuracy,
                "validation_loss": result.validation_loss,
                "variant": variant,
            }
            run_rows.append(row)
            variant_results.append(float(result.validation_accuracy or 0.0))
        selected_accuracies[variant] = sum(variant_results) / len(variant_results)
    selected_variant = select_smallest_near_best(
        selected_accuracies, bootstrap.config.gates.feature_tolerance
    )
    best_control = max(
        value for name, value in controls.items() if name != "true_node_oracle"
    )
    best_integrator = max(selected_accuracies.values())
    gate_open = (
        best_integrator >= bootstrap.config.gates.diagnostic_accuracy
        and best_integrator >= best_control + bootstrap.config.gates.diagnostic_control_margin
    )
    core: dict[str, object] = {
        "best_control_accuracy": best_control,
        "best_integrator_accuracy": best_integrator,
        "base_example_forwards": tensors.base_example_forwards,
        "cache_hits": tensors.cache_hits,
        "cache_misses": tensors.cache_misses,
        "controls": controls,
        "feature_accuracies": selected_accuracies,
        "feature_restart_aggregation": "arithmetic_mean",
        "gate_open": gate_open,
        "node_example_forwards": tensors.node_example_forwards,
        "optimistic": True,
        "runs": run_rows,
        "schema_version": "imagenetr50-integrator-sealed-diagnostic-v1",
        "selected_variant": selected_variant,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_all_leaf_ceiling(
    bootstrap: IntegratorBootstrap,
    variant: str,
    device: torch.device,
) -> dict[str, object]:
    """Run the declared static all-leaf ceiling after a failed U100 diagnostic."""
    target = bootstrap.store.run / "diagnostic" / "all_leaf_ceiling.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[diagnostic fallback] All-50-leaf prediction ceiling", flush=True)
    leaf_directories = tuple(
        next(path for path in task_root.iterdir() if path.is_dir())
        for task_root in sorted((bootstrap.sealed_tree.snapshots[0].nodes[0].directory.parents[2] / "leaves").glob("task_*"))
    )
    if len(leaf_directories) != 50:
        raise ValueError("sealed primary run does not expose exactly 50 leaf artifacts")
    bundles = tuple(load_node_bundle(path) for path in leaf_directories)
    nodes = tuple(BehaviorNode.from_bundle(bundle) for bundle in bundles)
    rows = bootstrap.manifest.select("train")
    tensors = _frontier_tensors(
        bootstrap,
        nodes,
        tuple(range(50)),
        50,
        rows,
        device,
        "sealed_all_leaves",
    )
    fit = frozenset(bootstrap.split.fit_image_ids)
    validation = frozenset(bootstrap.split.validation_image_ids)
    fit_ids = tuple(image_id for image_id in tensors.image_ids if image_id in fit)
    validation_ids = tuple(image_id for image_id in tensors.image_ids if image_id in validation)
    frontier_hash = record_sha256(
        {
            "node_hashes": [node.node_hash for node in nodes],
            "schema_version": "imagenetr50-integrator-all-leaf-frontier-v1",
        }
    )
    _state, result, artifact = _fit_or_load_fresh(
        bootstrap,
        "all_leaf_ceiling",
        frontier_hash,
        tensors,
        variant,
        50,
        fit_ids,
        validation_ids,
        bootstrap.config.seed,
        50,
        device,
    )
    core = {
        "artifact": str(artifact),
        "controls": _control_metrics(tensors.select(validation_ids)),
        "schema_version": "imagenetr50-integrator-all-leaf-ceiling-v1",
        "validation_accuracy": result.validation_accuracy,
        "validation_loss": result.validation_loss,
        "variant": variant,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_smoke(
    bootstrap: IntegratorBootstrap,
    selected_variant: str,
    device: torch.device,
) -> dict[str, object]:
    """Exercise eight real tasks, all carry levels, caching, and persistent integration."""
    target = bootstrap.store.run / "evaluations" / "smoke.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[phase 2/6] Eight-task real-data smoke", flush=True)
    capacity = min(bootstrap.config.consolidation_reservoir_sizes)
    hierarchy = build_hierarchy(
        bootstrap,
        hierarchy_policy(bootstrap.config, "fit", "bounded", capacity),
        bootstrap.config.smoke_tasks,
        _backbone_factory(bootstrap),
        device,
    )
    _state, rows = _persistent_run(
        bootstrap,
        hierarchy,
        selected_variant,
        min(bootstrap.config.historical_reservoir_sizes),
        "fit",
        "validation",
        bootstrap.config.smoke_tasks,
        (bootstrap.config.smoke_tasks,),
        bootstrap.config.seed,
        device,
    )
    final = rows[-1]
    replayed = build_hierarchy(
        bootstrap,
        hierarchy.policy,
        bootstrap.config.smoke_tasks,
        _backbone_factory(bootstrap),
        device,
        progress=False,
    )
    _replayed_state, replayed_rows = _persistent_run(
        bootstrap,
        replayed,
        selected_variant,
        min(bootstrap.config.historical_reservoir_sizes),
        "fit",
        "validation",
        bootstrap.config.smoke_tasks,
        (bootstrap.config.smoke_tasks,),
        bootstrap.config.seed,
        device,
    )
    checks = {
        "artifact_reuse": (
            replayed.work.leaf_optimizer_steps == 0
            and replayed.work.parent_optimizer_steps == 0
            and tuple(bundle.artifact.content_hash for bundle in replayed.nodes)
            == tuple(bundle.artifact.content_hash for bundle in hierarchy.nodes)
            and replayed_rows == rows
        ),
        "finite_accuracy": math.isfinite(float(final["accuracy"])),
        "one_live_root": int(final["live_nodes"]) == 1,
        "seven_carries": len(hierarchy.nodes) - bootstrap.config.smoke_tasks == 7,
        "validation_excluded": not bool(
            set(bootstrap.split.validation_image_ids)
            & {
                image_id
                for bundle in hierarchy.nodes
                for image_id in bundle.artifact.proxy_image_ids
            }
        ),
    }
    core = {
        "acceptance": checks,
        "gate_open": all(checks.values()),
        "hierarchy_work": asdict(hierarchy.work),
        "metric": final,
        "reuse_work": asdict(replayed.work),
        "schema_version": "imagenetr50-integrator-smoke-v1",
        "selected_variant": selected_variant,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _hierarchy_controls(
    bootstrap: IntegratorBootstrap,
    hierarchy: HierarchyBuildResult,
    stages: Sequence[int],
    partition: str,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    rows = []
    for stage in stages:
        nodes, slots, _frontier_hash = _hierarchy_frontier(hierarchy, stage)
        tasks = tuple(range(stage))
        universe_rows = (
            _partition_rows(bootstrap, "all_train", tasks)
            if partition == "validation"
            else _test_rows(bootstrap, tasks)
            if partition == "test"
            else _partition_rows(bootstrap, partition, tasks)
        )
        tensors = _frontier_tensors(
            bootstrap,
            nodes,
            slots,
            bootstrap.config.maximum_levels,
            universe_rows,
            device,
            f"hierarchy_{hierarchy.policy.content_hash}",
        )
        if partition == "validation":
            fit_ids = tuple(
                row.image_id
                for row in universe_rows
                if row.image_id in frozenset(bootstrap.split.fit_image_ids)
            )
            evaluation_ids = tuple(
                row.image_id
                for row in universe_rows
                if row.image_id in frozenset(bootstrap.split.validation_image_ids)
            )
            fitting_tensors = tensors.select(fit_ids)
            evaluation_tensors = tensors.select(evaluation_ids)
        else:
            fitting_tensors = (
                _frontier_tensors(
                    bootstrap,
                    nodes,
                    slots,
                    bootstrap.config.maximum_levels,
                    _partition_rows(bootstrap, "all_train", tasks),
                    device,
                    f"hierarchy_{hierarchy.policy.content_hash}",
                )
                if partition == "test"
                else None
            )
            evaluation_tensors = tensors
        controls = _control_metrics(evaluation_tensors)
        if fitting_tensors is not None:
            controls["affine_calibrated_union"] = _affine_control_accuracy(
                fitting_tensors, evaluation_tensors, nodes, slots
            )
        rows.append(
            {
                "controls": controls,
                "examples": len(evaluation_tensors.labels),
                "live_nodes": len(nodes),
                "stage": stage,
            }
        )
    return tuple(rows)


def _fresh_checkpoints(
    bootstrap: IntegratorBootstrap,
    hierarchy: HierarchyBuildResult,
    variant: str,
    stages: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    results = []
    for stage in stages:
        print(f"  fresh integration ceiling at task {stage}", flush=True)
        nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
        all_rows = _partition_rows(bootstrap, "all_train", tuple(range(stage)))
        tensors = _frontier_tensors(
            bootstrap,
            nodes,
            slots,
            bootstrap.config.maximum_levels,
            all_rows,
            device,
            f"hierarchy_{hierarchy.policy.content_hash}",
        )
        fit = frozenset(bootstrap.split.fit_image_ids)
        validation = frozenset(bootstrap.split.validation_image_ids)
        fit_ids = tuple(image_id for image_id in tensors.image_ids if image_id in fit)
        validation_ids = tuple(image_id for image_id in tensors.image_ids if image_id in validation)
        restarts = []
        for seed in bootstrap.config.replication_seeds:
            _state, result, artifact = _fit_or_load_fresh(
                bootstrap,
                "clean_fresh",
                frontier_hash,
                tensors,
                variant,
                bootstrap.config.maximum_levels,
                fit_ids,
                validation_ids,
                seed,
                stage,
                device,
            )
            restarts.append(
                {
                    "artifact": str(artifact),
                    "best_epoch": result.best_epoch,
                    "epochs": result.epochs,
                    "seed": seed,
                    "validation_accuracy": result.validation_accuracy,
                }
            )
        results.append(
            {
                "best_validation_accuracy": max(
                    float(row["validation_accuracy"]) for row in restarts
                ),
                "mean_validation_accuracy": sum(
                    float(row["validation_accuracy"]) for row in restarts
                )
                / len(restarts),
                "restarts": restarts,
                "stage": stage,
            }
        )
    return tuple(results)


def _persistent_training_rows(
    bootstrap: IntegratorBootstrap,
    partition: str,
    stage: int,
    historical_capacity: int,
) -> tuple[ImageRecord, ...]:
    current = _partition_rows(bootstrap, partition, (stage - 1,))
    if stage == 1:
        return current
    history = _partition_rows(bootstrap, partition, tuple(range(stage - 1)))
    reservoir = class_stratified_reservoir(
        history, historical_capacity, HISTORICAL_NAMESPACE
    )
    by_id = {row.image_id: row for row in history}
    historical = tuple(by_id[image_id] for image_id in reservoir.image_ids)
    if set(row.image_id for row in current) & set(reservoir.image_ids):
        raise ValueError("current and historical integrator sources overlap")
    return current + historical


def _persistent_family_name(
    hierarchy: HierarchyBuildResult,
    variant: str,
    historical_capacity: int,
    training_partition: str,
    seed: int,
) -> str:
    return (
        f"{hierarchy.policy.content_hash}_{variant}_history{historical_capacity}_"
        f"{training_partition}_seed{seed}"
    )


def _persistent_run(
    bootstrap: IntegratorBootstrap,
    hierarchy: HierarchyBuildResult,
    variant: str,
    historical_capacity: int,
    training_partition: str,
    evaluation_partition: str,
    task_count: int,
    evaluation_stages: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[IntegratorState, tuple[dict[str, object], ...]]:
    family = _persistent_family_name(
        hierarchy,
        variant,
        historical_capacity,
        training_partition,
        seed,
    )
    state = create_integrator_state(
        f"persistent-{family}",
        bootstrap.config.maximum_levels,
        variant,
        bootstrap.config.optimization,
        seed,
        device,
    )
    root = bootstrap.store.run / "integrators" / "persistent" / family
    training_ledger = ChainedJsonlLedger(
        root / "training_metrics.jsonl",
        "imagenetr50-integrator-stage-training-v1",
    )
    evaluation_ledger = ChainedJsonlLedger(
        root / f"{evaluation_partition}_metrics.jsonl",
        "imagenetr50-integrator-stage-evaluation-v1",
    )
    completed_training = {
        int(row["stage"]): row for row in training_ledger.rows
    }
    completed_evaluations = {
        int(row["stage"]): row for row in evaluation_ledger.rows
    }
    requested_evaluations = frozenset(evaluation_stages)
    from tqdm.auto import tqdm

    stages = tqdm(
        range(1, task_count + 1),
        desc=f"persistent H={historical_capacity} through task {task_count}",
        unit="task",
    )
    for stage in stages:
        nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
        training_rows = _persistent_training_rows(
            bootstrap, training_partition, stage, historical_capacity
        )
        current_rows = _partition_rows(
            bootstrap, training_partition, (stage - 1,)
        )
        if (
            len(nodes) != stage.bit_count()
            or len(slots) != len(nodes)
            or tuple(sorted(slots)) != slots
            or any(not 0 <= slot < bootstrap.config.maximum_levels for slot in slots)
            or len(training_rows) > len(current_rows) + historical_capacity
        ):
            raise RuntimeError("persistent integrator violated its LogT/replay work bound")
        training_ids_hash = record_sha256([row.image_id for row in training_rows])
        checkpoint = root / "checkpoints" / f"stage_{stage:03d}.pt"
        if checkpoint.is_file():
            restore_integrator_checkpoint(
                checkpoint,
                state,
                stage,
                variant,
                frontier_hash,
                training_ids_hash,
            )
            fit_result = None
        else:
            tensors = _frontier_tensors(
                bootstrap,
                nodes,
                slots,
                bootstrap.config.maximum_levels,
                training_rows,
                device,
                f"hierarchy_{hierarchy.policy.content_hash}",
            )
            fit_result = fit_integrator_epochs(
                state,
                _supervision(tensors, variant),
                bootstrap.config.optimization.persistent_epochs,
                bootstrap.config.optimization,
                seed,
                stage,
                device,
            )
            save_integrator_checkpoint(
                checkpoint,
                state,
                stage,
                variant,
                frontier_hash,
                training_ids_hash,
            )
        if stage not in completed_training:
            training_ledger.append(
                {
                    "frontier_hash": frontier_hash,
                    "historical_capacity": historical_capacity,
                    "integrator_optimizer_steps": state.optimizer_steps,
                    "live_nodes": len(nodes),
                    "node_example_forwards_bound": len(training_rows) * len(nodes),
                    "parameter_count": parameter_count(state.model),
                    "stage": stage,
                    "training_examples": len(training_rows),
                    "training_fit": None if fit_result is None else asdict(fit_result),
                    "training_ids_hash": training_ids_hash,
                    "variant": variant,
                }
            )
        if stage in requested_evaluations and stage not in completed_evaluations:
            tasks = tuple(range(stage))
            evaluation_rows = (
                _test_rows(bootstrap, tasks)
                if evaluation_partition == "test"
                else _partition_rows(bootstrap, evaluation_partition, tasks)
            )
            tensors = _frontier_tensors(
                bootstrap,
                nodes,
                slots,
                bootstrap.config.maximum_levels,
                evaluation_rows,
                device,
                f"hierarchy_{hierarchy.policy.content_hash}",
            )
            observations = tensors.observations(variant)
            predictions = predict_integrator(
                state.model,
                observations,
                bootstrap.config.optimization.batch_size,
                device,
            )
            task_accuracies = {
                str(task + 1): accuracy(
                    predictions[tensors.labels // 4 == task],
                    tensors.labels[tensors.labels // 4 == task],
                )
                for task in range(stage)
            }
            evaluation_ledger.append(
                {
                    "accuracy": accuracy(predictions, tensors.labels),
                    "controls": _control_metrics(tensors),
                    "evaluation_examples": len(evaluation_rows),
                    "evaluation_partition": evaluation_partition,
                    "frontier_hash": frontier_hash,
                    "live_nodes": len(nodes),
                    "stage": stage,
                    "task_accuracies": task_accuracies,
                    "variant": variant,
                }
            )
        stages.set_postfix(live=len(nodes), steps=state.optimizer_steps)
    training_ledger.require_unique_keys(("stage",))
    evaluation_ledger.require_unique_keys(("stage",))
    evaluations = {int(row["stage"]): row for row in evaluation_ledger.rows}
    combined = tuple(
        {**row, **evaluations.get(int(row["stage"]), {})}
        for row in training_ledger.rows
    )
    return state, combined


def _select_consolidation_capacity(
    bounded: Mapping[int, tuple[dict[str, object], ...]],
    full: Sequence[dict[str, object]],
    tolerance: float,
) -> int | None:
    full_by_stage = {int(row["stage"]): row for row in full}
    return next(
        (
            capacity
            for capacity in sorted(bounded)
            if all(
                float(full_by_stage[int(row["stage"])]["controls"]["true_node_oracle"])
                - float(row["controls"]["true_node_oracle"])
                <= tolerance
                for row in bounded[capacity]
            )
        ),
        None,
    )


def _select_historical_capacity(
    runs: Mapping[int, Sequence[dict[str, object]]],
    fresh: Sequence[dict[str, object]],
    control_floor: float,
    bootstrap: IntegratorBootstrap,
) -> int | None:
    fresh_by_stage = {
        int(row["stage"]): float(row["mean_validation_accuracy"]) for row in fresh
    }
    checkpoints = frozenset(bootstrap.config.calibration_checkpoints)
    for capacity in sorted(runs):
        evaluated = {
            int(row["stage"]): row
            for row in runs[capacity]
            if "accuracy" in row and int(row["stage"]) in checkpoints
        }
        if set(evaluated) != checkpoints:
            continue
        within = all(
            fresh_by_stage[stage] - float(evaluated[stage]["accuracy"])
            <= (
                bootstrap.config.gates.task16_persistent_fresh_tolerance
                if stage == 16
                else bootstrap.config.gates.persistent_fresh_tolerance
            )
            for stage in checkpoints
        )
        margin = float(evaluated[16]["accuracy"]) >= (
            control_floor + bootstrap.config.gates.persistent_control_margin
        )
        if within and margin:
            return capacity
    return None


def run_clean_development(
    bootstrap: IntegratorBootstrap,
    selected_variant: str,
    device: torch.device,
) -> dict[str, object]:
    """Select bounded hierarchy and history capacities without touching test identities."""
    target = bootstrap.store.run / "evaluations" / "clean_development.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[phase 3/6] Clean fit/validation hierarchy and capacity selection", flush=True)
    training_hash_capacity = max(bootstrap.config.consolidation_reservoir_sizes)
    full_policy = hierarchy_policy(
        bootstrap.config, "fit", "full_union", training_hash_capacity
    )
    full = build_hierarchy(
        bootstrap, full_policy, 16, _backbone_factory(bootstrap), device
    )
    bounded_trees = {
        capacity: build_hierarchy(
            bootstrap,
            hierarchy_policy(bootstrap.config, "fit", "bounded", capacity),
            16,
            _backbone_factory(bootstrap),
            device,
        )
        for capacity in bootstrap.config.consolidation_reservoir_sizes
    }
    full_controls = _hierarchy_controls(
        bootstrap, full, bootstrap.config.calibration_checkpoints, "validation", device
    )
    bounded_controls = {
        capacity: _hierarchy_controls(
            bootstrap,
            hierarchy,
            bootstrap.config.calibration_checkpoints,
            "validation",
            device,
        )
        for capacity, hierarchy in bounded_trees.items()
    }
    selected_consolidation = _select_consolidation_capacity(
        bounded_controls,
        full_controls,
        bootstrap.config.gates.hierarchy_oracle_tolerance,
    )
    if selected_consolidation is None:
        core = {
            "bounded_controls": bounded_controls,
            "full_controls": full_controls,
            "gate_open": False,
            "reason": "no bounded consolidation reservoir stayed within the hierarchy-oracle tolerance",
            "schema_version": "imagenetr50-integrator-clean-development-v1",
            "selected_variant": selected_variant,
        }
        record = {**core, "content_hash": record_sha256(core)}
        publish_immutable_json(target, record)
        return record
    selected_tree = bounded_trees[selected_consolidation]
    fresh = _fresh_checkpoints(
        bootstrap,
        selected_tree,
        selected_variant,
        bootstrap.config.calibration_checkpoints,
        device,
    )
    persistent: dict[int, tuple[dict[str, object], ...]] = {}
    for capacity in bootstrap.config.historical_reservoir_sizes:
        print(f"  persistent history calibration H={capacity}", flush=True)
        _state, rows = _persistent_run(
            bootstrap,
            selected_tree,
            selected_variant,
            capacity,
            "fit",
            "validation",
            16,
            bootstrap.config.calibration_checkpoints,
            bootstrap.config.seed,
            device,
        )
        persistent[capacity] = rows
    task16_controls = next(row for row in bounded_controls[selected_consolidation] if row["stage"] == 16)
    control_floor = max(
        float(value)
        for name, value in task16_controls["controls"].items()
        if name != "true_node_oracle"
    )
    selected_history = _select_historical_capacity(
        persistent, fresh, control_floor, bootstrap
    )
    selection_open = selected_history is not None
    core = {
        "bounded_controls": bounded_controls,
        "control_floor": control_floor,
        "fresh": list(fresh),
        "full_controls": list(full_controls),
        "gate_open": selection_open,
        "persistent": {str(key): list(value) for key, value in persistent.items()},
        "reason": None if selection_open else "no bounded historical reservoir met the fresh/control gates",
        "schema_version": "imagenetr50-integrator-clean-development-v1",
        "selected_consolidation_capacity": selected_consolidation,
        "selected_historical_capacity": selected_history,
        "selected_variant": selected_variant,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_development_extension(
    bootstrap: IntegratorBootstrap,
    selection: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    """Extend the frozen clean selection through task 50 and apply the dev gate."""
    target = bootstrap.store.run / "evaluations" / "development_task50.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[phase 4/6] Frozen clean development extension through task 50", flush=True)
    consolidation = int(selection["selected_consolidation_capacity"])
    history = int(selection["selected_historical_capacity"])
    variant = str(selection["selected_variant"])
    bounded = build_hierarchy(
        bootstrap,
        hierarchy_policy(bootstrap.config, "fit", "bounded", consolidation),
        50,
        _backbone_factory(bootstrap),
        device,
    )
    full = build_hierarchy(
        bootstrap,
        hierarchy_policy(
            bootstrap.config,
            "fit",
            "full_union",
            max(bootstrap.config.consolidation_reservoir_sizes),
        ),
        50,
        _backbone_factory(bootstrap),
        device,
    )
    _state, persistent = _persistent_run(
        bootstrap,
        bounded,
        variant,
        history,
        "fit",
        "validation",
        50,
        tuple(range(1, 51)),
        bootstrap.config.seed,
        device,
    )
    fresh = _fresh_checkpoints(bootstrap, bounded, variant, (50,), device)
    bounded_controls = _hierarchy_controls(bootstrap, bounded, (50,), "validation", device)[0]
    full_controls = _hierarchy_controls(bootstrap, full, (50,), "validation", device)[0]
    final = persistent[-1]
    static_floor = max(
        float(value)
        for name, value in bounded_controls["controls"].items()
        if name != "true_node_oracle"
    )
    checks = {
        "accuracy": float(final["accuracy"]) >= bootstrap.config.gates.development_accuracy,
        "hierarchy": float(full_controls["controls"]["true_node_oracle"])
        - float(bounded_controls["controls"]["true_node_oracle"])
        <= bootstrap.config.gates.hierarchy_oracle_tolerance,
        "persistent_fresh": float(fresh[0]["mean_validation_accuracy"])
        - float(final["accuracy"])
        <= bootstrap.config.gates.persistent_fresh_tolerance,
        "static_margin": float(final["accuracy"])
        >= static_floor + bootstrap.config.gates.development_control_margin,
    }
    core = {
        "acceptance": checks,
        "bounded_controls": bounded_controls,
        "fresh": list(fresh),
        "full_controls": full_controls,
        "gate_open": all(checks.values()),
        "persistent": list(persistent),
        "schema_version": "imagenetr50-integrator-development-task50-v1",
        "selection": dict(selection),
        "static_floor": static_floor,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_locked_benchmark(
    bootstrap: IntegratorBootstrap,
    selection: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    """Retrain on all 24k training identities, seal choices, then open the 6k test set."""
    target = bootstrap.store.run / "evaluations" / "locked_test.json"
    if target.is_file():
        return load_canonical_json(target)
    print("[phase 5/6] Locked 24k-train / 6k-test benchmark", flush=True)
    consolidation = int(selection["selected_consolidation_capacity"])
    history = int(selection["selected_historical_capacity"])
    variant = str(selection["selected_variant"])
    matrix_core = {
        "consolidation_capacity": consolidation,
        "historical_capacity": history,
        "protocol_hash": bootstrap.protocol.content_hash,
        "schema_version": "imagenetr50-integrator-locked-matrix-v1",
        "test_rows_opened_before_seal": 0,
        "variant": variant,
    }
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "locked_matrix.json",
        {**matrix_core, "content_hash": record_sha256(matrix_core)},
    )
    bounded = build_hierarchy(
        bootstrap,
        hierarchy_policy(bootstrap.config, "all_train", "bounded", consolidation),
        50,
        _backbone_factory(bootstrap),
        device,
    )
    _state, _training_only = _persistent_run(
        bootstrap,
        bounded,
        variant,
        history,
        "all_train",
        "test",
        50,
        (),
        bootstrap.config.seed,
        device,
    )
    behavior_ledger = ChainedJsonlLedger(
        bootstrap.store.run / "ledgers" / "behavior_requests.jsonl",
        "imagenetr50-integrator-behavior-request-v1",
    )
    training_seal_path = bootstrap.store.run / "protocol" / "locked_training_seal.json"
    if not training_seal_path.is_file():
        test_requests_before_seal = sum(
            "test" in row["splits"] for row in behavior_ledger.rows
        )
        if test_requests_before_seal:
            raise RuntimeError("test behavior was opened before locked training completed")
    training_seal_core = {
        "final_frontier_hash": bounded.snapshots[-1].content_hash,
        "persistent_checkpoint_sha256": file_sha256(
            bootstrap.store.run
            / "integrators"
            / "persistent"
            / _persistent_family_name(
                bounded,
                variant,
                history,
                "all_train",
                bootstrap.config.seed,
            )
            / "checkpoints"
            / "stage_050.pt"
        ),
        "schema_version": "imagenetr50-integrator-locked-training-seal-v1",
        "test_requests_before_seal": 0,
    }
    publish_immutable_json(
        training_seal_path,
        {
            **training_seal_core,
            "content_hash": record_sha256(training_seal_core),
        },
    )
    _state, persistent = _persistent_run(
        bootstrap,
        bounded,
        variant,
        history,
        "all_train",
        "test",
        50,
        tuple(range(1, 51)),
        bootstrap.config.seed,
        device,
    )
    stage_rows = tuple(row for row in persistent if "accuracy" in row)
    last = float(stage_rows[-1]["accuracy"])
    incremental = sum(float(row["accuracy"]) for row in stage_rows) / len(stage_rows)
    final_controls = _hierarchy_controls(
        bootstrap, bounded, (50,), "test", device
    )[0]
    task_matrix = tuple(
        {
            "accuracy": value,
            "stage": int(row["stage"]),
            "task": int(task),
        }
        for row in stage_rows
        for task, value in row["task_accuracies"].items()
    )
    beats_local = (
        last > bootstrap.config.gates.local_e2_last
        and incremental > bootstrap.config.gates.local_e2_incremental
    )
    local_references = _local_reference_results(bootstrap)
    core = {
        "beats_local_e2_both_metrics": beats_local,
        "comparisons": {
            "last_minus_joint_iid": last - local_references["joint_iid_last"],
            "last_minus_local_e2": last - local_references["local_e2_last"],
            "incremental_minus_local_e2": (
                incremental - local_references["local_e2_incremental"]
            ),
        },
        "incremental_accuracy": incremental,
        "last_accuracy": last,
        "local_e2_incremental": bootstrap.config.gates.local_e2_incremental,
        "local_e2_last": bootstrap.config.gates.local_e2_last,
        "local_references": local_references,
        "final_static_controls": final_controls,
        "replications_required": beats_local,
        "schema_version": "imagenetr50-integrator-locked-test-v1",
        "selection": dict(selection),
        "stage_metrics": list(stage_rows),
        "task_accuracy_matrix": list(task_matrix),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_integrator_workflow(
    config_path: str | Path = DEFAULT_INTEGRATOR_CONFIG,
) -> Path:
    """Run the single phase-gated experiment, resuming at immutable boundaries."""
    bootstrap = bootstrap_integrator(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("the local ImageNet-R integrator workflow requires CUDA")
    from tqdm.auto import tqdm

    from apm.continual.vision.imagenetr.integrator_reporting import write_integrator_report

    device = torch.device("cuda:0")
    overall = tqdm(total=6, desc="ImageNet-R integrator workflow", unit="phase")
    print(f"Temporary/resumable artifact directory: {bootstrap.store.run}", flush=True)
    _write_state(bootstrap, "PREFLIGHT")
    run_preflight(bootstrap, device)
    diagnostic = run_sealed_diagnostic(bootstrap, device)
    overall.update(1)
    if not bool(diagnostic["gate_open"]):
        ceiling = run_all_leaf_ceiling(
            bootstrap, str(diagnostic["selected_variant"]), device
        )
        _write_state(
            bootstrap,
            "COMPLETE_DIAGNOSTIC_FAILURE",
            all_leaf_ceiling=ceiling,
            diagnostic=diagnostic,
        )
        write_integrator_report(bootstrap.store.run)
        overall.update(5)
        overall.close()
        return bootstrap.store.run
    _write_state(bootstrap, "SMOKE", selected_variant=diagnostic["selected_variant"])
    smoke = run_smoke(bootstrap, str(diagnostic["selected_variant"]), device)
    overall.update(1)
    if not bool(smoke["gate_open"]):
        _write_state(bootstrap, "COMPLETE_SMOKE_FAILURE", smoke=smoke)
        write_integrator_report(bootstrap.store.run)
        overall.update(4)
        overall.close()
        return bootstrap.store.run
    _write_state(bootstrap, "CLEAN_SELECTION", selected_variant=diagnostic["selected_variant"])
    selection = run_clean_development(
        bootstrap, str(diagnostic["selected_variant"]), device
    )
    overall.update(1)
    if not bool(selection["gate_open"]):
        _write_state(bootstrap, "COMPLETE_CLEAN_SELECTION_FAILURE", selection=selection)
        write_integrator_report(bootstrap.store.run)
        overall.update(3)
        overall.close()
        return bootstrap.store.run
    _write_state(bootstrap, "DEVELOPMENT_EXTENSION", selection=selection)
    development = run_development_extension(bootstrap, selection, device)
    overall.update(1)
    if not bool(development["gate_open"]):
        _write_state(bootstrap, "COMPLETE_DEVELOPMENT_FAILURE", development=development)
        write_integrator_report(bootstrap.store.run)
        overall.update(2)
        overall.close()
        return bootstrap.store.run
    _write_state(bootstrap, "LOCKED_TEST", selection=selection)
    locked = run_locked_benchmark(bootstrap, selection, device)
    overall.update(1)
    phase = "COMPLETE_REPLICATION_REQUIRED" if locked["replications_required"] else "COMPLETE"
    _write_state(bootstrap, phase, locked_test=locked)
    write_integrator_report(bootstrap.store.run)
    overall.update(1)
    overall.close()
    print("[phase 6/6] Reports complete", flush=True)
    return bootstrap.store.run


__all__ = [
    "DEFAULT_INTEGRATOR_CONFIG",
    "run_clean_development",
    "run_all_leaf_ceiling",
    "run_development_extension",
    "run_integrator_workflow",
    "run_locked_benchmark",
    "run_preflight",
    "run_sealed_diagnostic",
    "run_smoke",
]
