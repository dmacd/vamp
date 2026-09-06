"""Run the clean stage-31 nested-H frontier-LoRA adaptation audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import tempfile
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.frontier_adaptation_config import (
    DEFAULT_FRONTIER_ADAPTATION_CONFIG,
    FrontierAdaptationConfig,
    load_frontier_adaptation_config,
)
from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    AdaptationCell,
    AdaptiveFrontierModel,
    fit_adaptation_cell,
    nested_replay_order,
)
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.integrator_workflow import (
    _hierarchy_frontier,
    _partition_rows,
)
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.macro_token_workflow import (
    MacroTokenBootstrap,
    MacroTokenProtocol,
    bootstrap_macro_token,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    _build,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.training import os_cpu_workers


@dataclass(frozen=True, slots=True)
class FrontierAdaptationProtocol:
    """Content identity binding clean source authority, code, and configuration."""

    source_macro_run_hash: str
    source_macro_protocol_sha256: str
    source_convergence_run_hash: str
    source_convergence_protocol_sha256: str
    source_convergence_result_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-frontier-adaptation-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("source macro run", self.source_macro_run_hash),
            ("source macro protocol", self.source_macro_protocol_sha256),
            ("source convergence run", self.source_convergence_run_hash),
            ("source convergence protocol", self.source_convergence_protocol_sha256),
            ("source convergence result", self.source_convergence_result_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-frontier-adaptation-protocol-v1":
            raise ValueError("frontier adaptation protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the complete experiment namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class FrontierAdaptationBootstrap:
    """Authenticated source experiment and isolated adaptation run paths."""

    project_root: Path
    config: FrontierAdaptationConfig
    source: MacroTokenBootstrap
    convergence_result: dict[str, object]
    protocol: FrontierAdaptationProtocol
    run: Path


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/imagenetr50_frontier_lora_adaptation_protocol.md",
        project_root
        / "scripts/vision/imagenetr/run_frontier_lora_adaptation_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *(package / name for name in (
            "artifacts.py",
            "checkpoints.py",
            "data.py",
            "frontier_adaptation_config.py",
            "frontier_adaptation_training.py",
            "frontier_adaptation_workflow.py",
            "heads.py",
            "integrator_artifacts.py",
            "integrator_hierarchy.py",
            "integrator_observations.py",
            "integrator_workflow.py",
            "lora.py",
            "macro_token_config.py",
            "macro_token_model.py",
            "macro_token_training.py",
            "macro_token_workflow.py",
            "manifests.py",
            "model.py",
            "promoted_integrator_config.py",
            "promoted_integrator_workflow.py",
            "protocol.py",
            "router_features.py",
            "training.py",
        )),
        package / "merging/common.py",
    )


def _prepare_run(run: Path, protocol: FrontierAdaptationProtocol) -> None:
    for relative in (
        "protocol",
        "models",
        "evaluations/cells",
        "histories",
        "checkpoints",
        "ledgers",
        "reports",
        "state",
        "work",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(run / "protocol/protocol.json", protocol.as_record())


def _stored_macro_protocol(record: Mapping[str, object]) -> MacroTokenProtocol:
    protocol = MacroTokenProtocol(
        source_run_hash=str(record["source_run_hash"]),
        source_protocol_sha256=str(record["source_protocol_sha256"]),
        dataset_manifest_hash=str(record["dataset_manifest_hash"]),
        model_manifest_hash=str(record["model_manifest_hash"]),
        split_hash=str(record["split_hash"]),
        config_hash=str(record["config_hash"]),
        code_manifest_hash=str(record["code_manifest_hash"]),
        environment_manifest_hash=str(record["environment_manifest_hash"]),
        schema_version=str(record["schema_version"]),
    )
    if protocol.content_hash != record.get("content_hash"):
        raise ValueError("stored macro-token protocol content changed")
    return protocol


def _validated_convergence_result(path: Path) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version") != "imagenetr50-macro-convergence-result-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("test_evaluations") != 0
    ):
        raise ValueError("source convergence result does not authenticate")
    return record


def bootstrap_frontier_adaptation(
    config_path: str | Path = DEFAULT_FRONTIER_ADAPTATION_CONFIG,
) -> FrontierAdaptationBootstrap:
    """Authenticate v8/v9 sources and prepare the isolated adaptation namespace."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_frontier_adaptation_config(resolved)
    current_source = bootstrap_macro_token(config.source_macro_config)
    source_run = config.source_macro_artifact_root / "runs" / config.source_macro_run_hash
    source_protocol_path = source_run / "protocol/protocol.json"
    source_hierarchy_path = source_run / "protocol/clean_hierarchy.json"
    source_stage_path = source_run / "evaluations/clean/stage_031.json"
    stored_macro = _stored_macro_protocol(load_canonical_json(source_protocol_path))
    integrator = current_source.source.integrator
    if (
        stored_macro.content_hash != config.source_macro_run_hash
        or current_source.config.config_hash != stored_macro.config_hash
        or integrator.manifest.content_hash != stored_macro.dataset_manifest_hash
        or integrator.protocol.model_manifest_hash != stored_macro.model_manifest_hash
        or integrator.split.content_hash != stored_macro.split_hash
        or file_sha256(source_protocol_path) != config.source_macro_protocol_sha256
        or file_sha256(source_hierarchy_path) != config.source_clean_hierarchy_sha256
        or file_sha256(source_stage_path) != config.source_clean_stage31_sha256
    ):
        raise ValueError("configured macro-token source artifacts changed")
    source = replace(current_source, protocol=stored_macro, run=source_run)
    atomic_write(
        config.source_macro_artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": config.source_macro_run_hash,
                "schema_version": "imagenetr50-macro-token-latest-v1",
            }
        ),
    )
    convergence_run = (
        config.source_convergence_artifact_root
        / "runs"
        / config.source_convergence_run_hash
    )
    convergence_protocol_path = convergence_run / "protocol/protocol.json"
    convergence_result_path = convergence_run / "evaluations/result.json"
    convergence_protocol = load_canonical_json(convergence_protocol_path)
    convergence_result = _validated_convergence_result(convergence_result_path)
    if (
        file_sha256(convergence_protocol_path)
        != config.source_convergence_protocol_sha256
        or file_sha256(convergence_result_path)
        != config.source_convergence_result_sha256
        or convergence_protocol.get("content_hash")
        != config.source_convergence_run_hash
        or convergence_protocol.get("source_macro_run_hash")
        != config.source_macro_run_hash
    ):
        raise ValueError("configured convergence reference artifacts changed")
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = FrontierAdaptationProtocol(
        config.source_macro_run_hash,
        config.source_macro_protocol_sha256,
        config.source_convergence_run_hash,
        config.source_convergence_protocol_sha256,
        config.source_convergence_result_sha256,
        integrator.manifest.content_hash,
        integrator.protocol.model_manifest_hash,
        integrator.split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    run = config.artifact_root / "runs" / protocol.content_hash
    _prepare_run(run, protocol)
    publish_immutable_bytes(
        run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("router_split.json", integrator.split.as_record()),
        ("source_macro_protocol.json", load_canonical_json(source_protocol_path)),
        ("source_convergence_protocol.json", convergence_protocol),
        ("source_convergence_result.json", convergence_result),
    ):
        publish_immutable_json(run / "protocol" / filename, record)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-frontier-adaptation-latest-v1",
            }
        ),
    )
    return FrontierAdaptationBootstrap(
        project_root, config, source, convergence_result, protocol, run
    )


def _phase(
    bootstrap: FrontierAdaptationBootstrap,
    number: int,
    total: int,
    message: str,
) -> None:
    """Print and persist one human-readable workflow phase transition."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.run / "ledgers/workflow_events.jsonl",
        "imagenetr50-frontier-adaptation-workflow-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": "imagenetr50-frontier-adaptation-workflow-event-v1",
            "wall_time_unix": time.time(),
        }
    )
    atomic_write(
        bootstrap.run / "state/workflow.json",
        canonical_json_bytes(
            {
                "message": message,
                "phase": number,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-frontier-adaptation-workflow-state-v1",
            }
        ),
    )


def _clean_rows(
    bootstrap: FrontierAdaptationBootstrap, partition: str
) -> tuple[ImageRecord, ...]:
    if partition not in {"fit", "validation"}:
        raise ValueError("frontier adaptation may request only fit or validation rows")
    return _partition_rows(
        bootstrap.source.source.integrator,
        partition,
        tuple(range(bootstrap.config.stage)),
    )


def _replay_manifest(
    bootstrap: FrontierAdaptationBootstrap, fitting_rows: Sequence[ImageRecord]
) -> tuple[ImageRecord, ...]:
    """Publish the exact nested population order and per-budget diagnostics."""
    ordered = nested_replay_order(fitting_rows, bootstrap.config.seed)
    entries = []
    for capacity in bootstrap.config.historical_capacities:
        selected = ordered[:capacity]
        class_counts = {
            str(class_id): sum(row.remapped_class_index == class_id for row in selected)
            for class_id in range(4 * bootstrap.config.stage)
        }
        task_counts = {
            str(task): sum(row.task_index == task for row in selected)
            for task in range(bootstrap.config.stage)
        }
        entries.append(
            {
                "class_counts": class_counts,
                "historical_capacity": capacity,
                "image_ids_hash": record_sha256(
                    [row.image_id for row in selected]
                ),
                "task_counts": task_counts,
            }
        )
    core: dict[str, object] = {
        "entries": entries,
        "full_fit_examples": len(ordered),
        "nested_order_image_ids": [row.image_id for row in ordered],
        "nested_prefixes": all(
            tuple(ordered[:smaller])
            == tuple(ordered[:larger])[:smaller]
            for smaller, larger in zip(
                bootstrap.config.historical_capacities[:-1],
                bootstrap.config.historical_capacities[1:],
                strict=True,
            )
        ),
        "sampler": "uniform_hash_order_without_replacement",
        "schema_version": "imagenetr50-frontier-adaptation-replay-v1",
        "seed": bootstrap.config.seed,
    }
    if (
        len(ordered) != bootstrap.config.full_fit_examples
        or bootstrap.config.historical_capacities[-1] != len(ordered)
        or not core["nested_prefixes"]
    ):
        raise RuntimeError("nested replay manifest differs from the declared matrix")
    publish_immutable_json(
        bootstrap.run / "protocol/replay_populations.json",
        {**core, "content_hash": record_sha256(core)},
    )
    return ordered


def _frontier(
    hierarchy: HierarchyBuildResult, stage: int
) -> tuple[tuple[BehaviorNode, ...], tuple[int, ...], str]:
    nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
    if (
        tuple(node.represented_task_ids for node in nodes)
        != ((30,), (28, 29), (24, 25, 26, 27), tuple(range(16, 24)), tuple(range(16)))
        or slots != (0, 1, 2, 3, 4)
    ):
        raise RuntimeError("stage-31 frontier topology changed")
    return nodes, slots, frontier_hash


def _new_model(
    bootstrap: FrontierAdaptationBootstrap,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    device: torch.device,
) -> AdaptiveFrontierModel:
    integrator = bootstrap.source.source.integrator
    return AdaptiveFrontierModel(
        nodes,
        slots,
        lambda: create_pinned_backbone(integrator.checkpoint),
        integrator.primary_config.lora_rank,
        integrator.primary_config.lora_alpha,
        1,
        bootstrap.config.dropout,
        bootstrap.config.seed,
        device,
    )


def _preflight(
    bootstrap: FrontierAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    ordered_fit: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    device: torch.device,
) -> dict[str, object]:
    """Validate clean isolation, real attached gradients, parity, and GPU memory."""
    target = bootstrap.run / "protocol/preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    nodes, slots, frontier_hash = _frontier(hierarchy, bootstrap.config.stage)
    integrator = bootstrap.source.source.integrator
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    fit_ids = frozenset(row.image_id for row in ordered_fit)
    validation_ids = frozenset(row.image_id for row in validation_rows)
    model = _new_model(bootstrap, nodes, slots, device)
    model.set_lora_trainable(True)
    real_rows = tuple(ordered_fit[: bootstrap.config.microbatch_size])
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            real_rows,
            integrator.train_transform,
            bootstrap.config.seed,
            0,
        ),
        batch_size=bootstrap.config.microbatch_size,
        shuffle=False,
        num_workers=min(bootstrap.config.num_workers, os_cpu_workers()),
        pin_memory=True,
    )
    images, labels, _image_ids = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.set_evaluation_mode()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        direct = model(images, adapt_lora=False, activation_recomputation=False)
    model.macro.zero_grad(set_to_none=True)
    for node_model in model.node_models:
        node_model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        recomputed = model(images, adapt_lora=True, activation_recomputation=True)
        loss = F.cross_entropy(recomputed, labels)
    loss.backward()
    trainables = tuple(model.macro.parameters()) + model.lora_parameters
    gradient_tensors = tuple(
        parameter.grad for parameter in trainables if parameter.requires_grad
    )
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    core: dict[str, object] = {
        "activation_recomputation_max_logit_error": float(
            torch.max(
                torch.abs(
                    direct[:, model.seen_class_mask].float()
                    - recomputed.detach()[:, model.seen_class_mask].float()
                )
            )
        ),
        "attached_gradient_tensors": sum(value is not None for value in gradient_tensors),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "finite_gradient_tensors": sum(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradient_tensors
        ),
        "fit_examples": len(ordered_fit),
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "frontier_hash": frontier_hash,
        "frontier_node_hashes": [node.node_hash for node in nodes],
        "frontier_slots": list(slots),
        "live_nodes": len(nodes),
        "loss": float(loss.detach()),
        "microbatch_size": bootstrap.config.microbatch_size,
        "output_shape": list(recomputed.shape),
        "peak_vram_bytes": int(peak),
        "schema_version": "imagenetr50-frontier-adaptation-preflight-v1",
        "test_fit_overlap": len(test_ids & fit_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "trainable_parameters": sum(parameter.numel() for parameter in trainables),
        "validation_examples": len(validation_rows),
    }
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["fit_examples"] != 12194
        or core["validation_examples"] != 3049
        or core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or core["live_nodes"] != 5
        or core["trainable_parameters"] != 18_691_016
        or core["output_shape"] != [bootstrap.config.microbatch_size, 200]
        or core["activation_recomputation_max_logit_error"] != 0.0
        or core["attached_gradient_tensors"] != len(trainables)
        or core["finite_gradient_tensors"] != len(trainables)
    ):
        raise RuntimeError(f"frontier adaptation preflight failed: {core}")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    del model, images, labels, direct, recomputed, loss
    torch.cuda.empty_cache()
    return record


def _validated_cell(path: Path) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version") != "imagenetr50-frontier-adaptation-cell-v1"
        or record.get("content_hash") != record_sha256(core)
    ):
        raise ValueError(f"completed frontier adaptation cell changed: {path}")
    return record


def _job_hash(
    bootstrap: FrontierAdaptationBootstrap,
    cell: AdaptationCell,
    frontier_hash: str,
    training_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
) -> str:
    return record_sha256(
        {
            "cell": cell.as_record(),
            "fit_image_ids": [row.image_id for row in training_rows],
            "frontier_hash": frontier_hash,
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-frontier-adaptation-job-v1",
            "validation_image_ids": [row.image_id for row in validation_rows],
        }
    )


def _publish_model(
    bootstrap: FrontierAdaptationBootstrap,
    model: AdaptiveFrontierModel,
    cell: AdaptationCell,
    fit_record: Mapping[str, object],
    train_metrics: Mapping[str, object],
    validation_metrics: Mapping[str, object],
    displacements: Sequence[Mapping[str, object]],
    history_path: Path,
    job_hash: str,
    nodes: Sequence[BehaviorNode],
) -> tuple[Path, str]:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = bootstrap.run / "models" / job_hash
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(
        tempfile.mkdtemp(prefix="frontier-adaptation-", dir=bootstrap.run / "work")
    )
    try:
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in sorted(model.macro.state_dict().items())
            },
            work / "macro.safetensors",
            metadata={"schema_version": "imagenetr50-frontier-macro-model-v1"},
        )
        adapter_rows = []
        for index, (node_model, node) in enumerate(
            zip(model.node_models, nodes, strict=True)
        ):
            filename = f"node_{index:02d}_adapter.safetensors"
            digest = save_adapter(work / filename, adapter_factors(node_model))
            adapter_rows.append(
                {
                    "filename": filename,
                    "node_hash": node.node_hash,
                    "sha256": digest,
                }
            )
        publish_immutable_json(
            work / "fit.json",
            {
                "adapters": adapter_rows,
                "cell": cell.as_record(),
                "displacements": [dict(row) for row in displacements],
                "fit": dict(fit_record),
                "history_sha256": file_sha256(history_path),
                "job_hash": job_hash,
                "schema_version": "imagenetr50-frontier-adaptation-model-v1",
                "train_metrics": dict(train_metrics),
                "validation_metrics": dict(validation_metrics),
            },
        )
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _fit_or_load(
    bootstrap: FrontierAdaptationBootstrap,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    frontier_hash: str,
    cell: AdaptationCell,
    training_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    role: str,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Fit one condition or authenticate it without constructing its five ViTs."""
    job_hash = _job_hash(
        bootstrap, cell, frontier_hash, training_rows, validation_rows
    )
    target = bootstrap.run / "evaluations/cells" / f"{job_hash}.json"
    if target.is_file():
        record = _validated_cell(target)
        validate_artifact_directory(bootstrap.run / str(record["artifact"]))
        return record, True
    model = _new_model(bootstrap, nodes, slots, device)
    history_path = bootstrap.run / "histories" / f"{job_hash}.jsonl"
    checkpoint_path = bootstrap.run / "checkpoints" / f"{job_hash}.pt"
    integrator = bootstrap.source.source.integrator
    fit, train_metrics, validation_metrics, displacements = fit_adaptation_cell(
        model=model,
        nodes=nodes,
        cell=cell,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=training_rows,
        validation_rows=validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=bootstrap.config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    model_path, artifact_hash = _publish_model(
        bootstrap,
        model,
        cell,
        fit.as_record(),
        train_metrics.as_record(),
        validation_metrics.as_record(),
        displacements,
        history_path,
        job_hash,
        nodes,
    )
    core: dict[str, object] = {
        "artifact": str(model_path.relative_to(bootstrap.run)),
        "artifact_sha256": artifact_hash,
        "cell": cell.as_record(),
        "displacements": [dict(row) for row in displacements],
        "fit": fit.as_record(),
        "history": str(history_path.relative_to(bootstrap.run)),
        "history_sha256": file_sha256(history_path),
        "job_hash": job_hash,
        "role": role,
        "schema_version": "imagenetr50-frontier-adaptation-cell-v1",
        "train_metrics": train_metrics.as_record(),
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False


def _references(bootstrap: FrontierAdaptationBootstrap) -> dict[str, object]:
    result = bootstrap.convergence_result
    replications = tuple(dict(row) for row in result["replications"])
    selected_seed = next(
        row for row in replications if int(dict(row["cell"])["seed"]) == 1993
    )
    selected_fits = tuple(dict(row["fit"]) for row in replications)
    joint = dict(dict(result["clean_joint"])["fit"])
    return {
        "frozen_macro_replication_mean": {
            "accuracy": sum(float(row["validation_accuracy"]) for row in selected_fits)
            / len(selected_fits),
            "nll": sum(float(row["validation_nll"]) for row in selected_fits)
            / len(selected_fits),
            "seeds": [1993, 1994, 1995],
        },
        "frozen_macro_seed1993": {
            "accuracy": float(dict(selected_seed["fit"])["validation_accuracy"]),
            "nll": float(dict(selected_seed["fit"])["validation_nll"]),
        },
        "joint_iid": {
            "accuracy": float(joint["fixed_validation_accuracy"]),
            "epochs": int(joint["epochs"]),
            "nll": float(joint["fixed_validation_nll"]),
        },
        "schema_version": "imagenetr50-frontier-adaptation-references-v1",
    }


def _seal(
    bootstrap: FrontierAdaptationBootstrap,
    ordered_fit: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    source_hashes_before: Mapping[str, str],
) -> dict[str, object]:
    integrator = bootstrap.source.source.integrator
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    fitting_ids = frozenset(row.image_id for row in ordered_fit)
    validation_ids = frozenset(row.image_id for row in validation_rows)
    source_paths = {
        "clean_hierarchy": bootstrap.source.run / "protocol/clean_hierarchy.json",
        "source_protocol": bootstrap.source.run / "protocol/protocol.json",
    }
    source_hashes_after = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    core: dict[str, object] = {
        "fit_examples": len(fitting_ids),
        "fit_validation_overlap": len(fitting_ids & validation_ids),
        "schema_version": "imagenetr50-frontier-adaptation-training-seal-v1",
        "source_files_unchanged": dict(source_hashes_before) == source_hashes_after,
        "source_hashes_after": source_hashes_after,
        "test_evaluations": 0,
        "test_fit_overlap": len(test_ids & fitting_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "training_derived_only": all(
            row.split == "train" for row in (*ordered_fit, *validation_rows)
        ),
        "validation_examples": len(validation_ids),
    }
    if (
        core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or not core["source_files_unchanged"]
        or not core["training_derived_only"]
    ):
        raise RuntimeError("frontier adaptation training seal failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(bootstrap.run / "protocol/training_seal.json", record)
    return record


def _publish_result(
    bootstrap: FrontierAdaptationBootstrap,
    preflight: Mapping[str, object],
    cells: Sequence[Mapping[str, object]],
    references: Mapping[str, object],
    seal: Mapping[str, object],
    reuse: Mapping[str, object],
) -> dict[str, object]:
    core: dict[str, object] = {
        "cells": [dict(row) for row in cells],
        "preflight": dict(preflight),
        "references": dict(references),
        "reuse_proof": dict(reuse),
        "schema_version": "imagenetr50-frontier-adaptation-result-v1",
        "stage": bootstrap.config.stage,
        "test_evaluations": 0,
        "training_seal": dict(seal),
    }
    record = {**core, "content_hash": record_sha256(core)}
    path = bootstrap.run / "evaluations/result.json"
    if path.is_file() and load_canonical_json(path) != record:
        raise ValueError("frontier adaptation result changed")
    publish_immutable_json(path, record)
    return record


def run_frontier_adaptation(
    config_path: str | Path = DEFAULT_FRONTIER_ADAPTATION_CONFIG,
) -> Path:
    """Run or exactly resume the clean stage-31 frontier-LoRA adaptation sweep."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the frontier adaptation audit requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_frontier_adaptation(config_path)
    print(f"Temporary/resumable artifact directory: {bootstrap.run}", flush=True)
    completed = bootstrap.run / "evaluations/result.json"
    if completed.is_file():
        result = load_canonical_json(completed)
        core = {key: value for key, value in result.items() if key != "content_hash"}
        if (
            result.get("schema_version")
            != "imagenetr50-frontier-adaptation-result-v1"
            or result.get("content_hash") != record_sha256(core)
        ):
            raise ValueError("completed frontier adaptation result changed")
        from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
            write_frontier_adaptation_report,
        )

        write_frontier_adaptation_report(bootstrap.run)
        print("Completed sweep authenticated; no optimizer work was repeated.", flush=True)
        return bootstrap.run
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=8, desc="ImageNet-R frontier adaptation", unit="phase")
    _phase(bootstrap, 1, 8, "Authenticate sources and rebuild the fit hierarchy")
    hierarchy = _build(bootstrap.source.source, "fit", 50, device, progress=False)
    if (
        hierarchy.policy.content_hash
        != bootstrap.source.config.fit_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("clean hierarchy was not reused exactly")
    nodes, slots, frontier_hash = _frontier(hierarchy, bootstrap.config.stage)
    source_paths = {
        "clean_hierarchy": bootstrap.source.run / "protocol/clean_hierarchy.json",
        "source_protocol": bootstrap.source.run / "protocol/protocol.json",
    }
    source_hashes_before = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    overall.update(1)

    _phase(bootstrap, 2, 8, "Freeze nested fit populations and the validation boundary")
    fit_rows = _clean_rows(bootstrap, "fit")
    validation_rows = _clean_rows(bootstrap, "validation")
    ordered_fit = _replay_manifest(bootstrap, fit_rows)
    overall.update(1)

    _phase(bootstrap, 3, 8, "Run real BF16 attached-gradient and memory preflight")
    preflight = _preflight(
        bootstrap, hierarchy, ordered_fit, validation_rows, device
    )
    overall.update(1)

    _phase(bootstrap, 4, 8, "Sweep five nested H values with all frontier LoRAs adaptive")
    cells: list[dict[str, object]] = []
    sweep_started = time.monotonic()
    for index, capacity in enumerate(bootstrap.config.historical_capacities, start=1):
        print(
            f"Adaptation condition {index}/{len(bootstrap.config.historical_capacities)}: "
            f"H={capacity:,}",
            flush=True,
        )
        record, reused = _fit_or_load(
            bootstrap,
            nodes,
            slots,
            frontier_hash,
            AdaptationCell(capacity, True, bootstrap.config.seed),
            ordered_fit[:capacity],
            validation_rows,
            "adaptive_h_sweep",
            device,
        )
        cells.append(record)
        if index == 1 and not reused:
            elapsed = time.monotonic() - sweep_started
            relative_work = sum(
                capacity_value + len(validation_rows)
                for capacity_value in bootstrap.config.historical_capacities[1:]
            ) / (capacity + len(validation_rows))
            print(
                f"Measured remaining adaptive-sweep ETA: {elapsed * relative_work / 3600:.2f} h",
                flush=True,
            )
    overall.update(1)

    _phase(bootstrap, 5, 8, "Fit the augmentation-matched frozen-LoRA full-fit control")
    frozen, _reused = _fit_or_load(
        bootstrap,
        nodes,
        slots,
        frontier_hash,
        AdaptationCell(bootstrap.config.full_fit_examples, False, bootstrap.config.seed),
        ordered_fit,
        validation_rows,
        "frozen_online_control",
        device,
    )
    cells.append(frozen)
    overall.update(1)

    _phase(bootstrap, 6, 8, "Seal training inputs and authenticate zero-work cell reuse")
    repeated = tuple(
        _fit_or_load(
            bootstrap,
            nodes,
            slots,
            frontier_hash,
            AdaptationCell(
                int(dict(row["cell"])["historical_capacity"]),
                bool(dict(row["cell"])["adapt_lora"]),
                bootstrap.config.seed,
            ),
            ordered_fit[: int(dict(row["cell"])["historical_capacity"])],
            validation_rows,
            str(row["role"]),
            device,
        )[1]
        for row in cells
    )
    reuse_core: dict[str, object] = {
        "all_cells_reused": all(repeated),
        "cell_count": len(repeated),
        "new_optimizer_steps": 0,
        "schema_version": "imagenetr50-frontier-adaptation-reuse-v1",
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": hierarchy.work.parent_optimizer_steps,
    }
    if not reuse_core["all_cells_reused"]:
        raise RuntimeError("frontier adaptation cells did not reuse exactly")
    reuse = {**reuse_core, "content_hash": record_sha256(reuse_core)}
    publish_immutable_json(bootstrap.run / "protocol/reuse_proof.json", reuse)
    seal = _seal(
        bootstrap, ordered_fit, validation_rows, source_hashes_before
    )
    overall.update(1)

    _phase(bootstrap, 7, 8, "Publish the accuracy/NLL result and compact evidence")
    references = _references(bootstrap)
    result = _publish_result(
        bootstrap, preflight, cells, references, seal, reuse
    )
    overall.update(1)

    _phase(bootstrap, 8, 8, "Generate the publication-style clean diagnostic report")
    from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
        write_frontier_adaptation_report,
    )

    write_frontier_adaptation_report(bootstrap.run)
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.run / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "result_hash": result["content_hash"],
                "schema_version": "imagenetr50-frontier-adaptation-invocation-v1",
            }
        ),
    )
    atomic_write(
        bootstrap.run / "state/workflow.json",
        canonical_json_bytes(
            {
                "phase": 8,
                "result_hash": result["content_hash"],
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-frontier-adaptation-complete-v1",
                "status": "COMPLETE",
            }
        ),
    )
    overall.update(1)
    overall.close()
    print(f"Frontier adaptation report complete in {elapsed / 60:.1f} minutes.", flush=True)
    return bootstrap.run


if __name__ == "__main__":
    print(run_frontier_adaptation())


__all__ = [
    "FrontierAdaptationBootstrap",
    "FrontierAdaptationProtocol",
    "bootstrap_frontier_adaptation",
    "run_frontier_adaptation",
]
