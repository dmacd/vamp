"""Run the full-fit stage-31 linear pre-classifier integrator ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
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
from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    AdaptationFit,
    fit_adaptation_cell,
)
from apm.continual.vision.imagenetr.frontier_adaptation_workflow import (
    FrontierAdaptationBootstrap,
    _clean_rows,
    _frontier,
    _material_paths as _parent_material_paths,
    bootstrap_frontier_adaptation,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_config import (
    DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG,
    FrontierLinearPreclassifierConfig,
    load_frontier_linear_preclassifier_config,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_model import (
    LinearPreclassifierFrontierModel,
)
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    _build,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.training import os_cpu_workers


CONTROL_RESULT = Path("evaluations/frontier_linear_preclassifier.json")


@dataclass(frozen=True, slots=True)
class LinearPreclassifierCell:
    """The single full-history architecture condition passed to shared training."""

    historical_capacity: int
    adapt_lora: bool
    seed: int

    @property
    def condition(self) -> str:
        """Return the stable artifact and reporting label."""
        return "frontier_linear_preclassifier_full_fit"

    def as_record(self) -> dict[str, object]:
        """Return canonical cell fields without implicit architecture state."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrontierLinearPreclassifierProtocol:
    """Content identity for the additive linear-integrator condition."""

    parent_run_hash: str
    parent_protocol_sha256: str
    parent_result_sha256: str
    parent_result_hash: str
    parent_replay_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    fit_image_ids_hash: str
    validation_image_ids_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-frontier-linear-preclassifier-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent protocol", self.parent_protocol_sha256),
            ("parent result file", self.parent_result_sha256),
            ("parent result", self.parent_result_hash),
            ("parent replay", self.parent_replay_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("fit identities", self.fit_image_ids_hash),
            ("validation identities", self.validation_image_ids_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if (
            self.schema_version
            != "imagenetr50-frontier-linear-preclassifier-protocol-v1"
        ):
            raise ValueError("linear pre-classifier protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the immutable control namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class FrontierLinearPreclassifierBootstrap:
    """Authenticated parent, exact populations, protocol, and output paths."""

    project_root: Path
    config: FrontierLinearPreclassifierConfig
    parent: FrontierAdaptationBootstrap
    parent_result: dict[str, object]
    fit_rows: tuple[ImageRecord, ...]
    validation_rows: tuple[ImageRecord, ...]
    protocol: FrontierLinearPreclassifierProtocol
    control_root: Path


def _material_paths(
    project_root: Path, config_path: Path, parent_config_path: Path
) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    additions = (
        config_path,
        project_root / "docs/imagenetr50_frontier_linear_preclassifier_protocol.md",
        project_root
        / "scripts/vision/imagenetr/run_frontier_linear_preclassifier_local.sh",
        package / "frontier_linear_preclassifier_config.py",
        package / "frontier_linear_preclassifier_model.py",
        package / "frontier_linear_preclassifier_workflow.py",
    )
    return tuple(
        dict.fromkeys(
            (*_parent_material_paths(project_root, parent_config_path), *additions)
        )
    )


def _validated_parent_result(path: Path) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-adaptation-result-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("test_evaluations") != 0
    ):
        raise ValueError("parent frontier result does not authenticate")
    return record


def bootstrap_frontier_linear_preclassifier(
    config_path: str | Path = DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG,
) -> FrontierLinearPreclassifierBootstrap:
    """Authenticate the completed parent and prepare the additive condition."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_frontier_linear_preclassifier_config(resolved)
    parent = bootstrap_frontier_adaptation(config.parent_config)
    parent_run = config.parent_artifact_root / "runs" / config.parent_run_hash
    protocol_path = parent_run / "protocol/protocol.json"
    result_path = parent_run / "evaluations/result.json"
    replay_path = parent_run / "protocol/replay_populations.json"
    parent_result = _validated_parent_result(result_path)
    fit_rows = _clean_rows(parent, "fit")
    validation_rows = _clean_rows(parent, "validation")
    fit_ids = tuple(row.image_id for row in fit_rows)
    validation_ids = tuple(row.image_id for row in validation_rows)
    if (
        parent.protocol.content_hash != config.parent_run_hash
        or parent.run != parent_run
        or file_sha256(protocol_path) != config.parent_protocol_sha256
        or file_sha256(result_path) != config.parent_result_sha256
        or parent_result["content_hash"] != config.parent_result_content_hash
        or file_sha256(replay_path) != config.parent_replay_sha256
        or parent.config.stage != config.stage
        or parent.config.seed != config.seed
        or len(fit_rows) != config.full_fit_examples
        or len(validation_rows) != config.validation_examples
        or set(fit_ids) & set(validation_ids)
        or any(row.split != "train" for row in (*fit_rows, *validation_rows))
    ):
        raise ValueError("linear integrator differs from its authenticated parent")
    code = material_tree_manifest(
        _material_paths(project_root, resolved, config.parent_config)
    )
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = FrontierLinearPreclassifierProtocol(
        config.parent_run_hash,
        config.parent_protocol_sha256,
        config.parent_result_sha256,
        config.parent_result_content_hash,
        config.parent_replay_sha256,
        parent.protocol.dataset_manifest_hash,
        parent.protocol.model_manifest_hash,
        parent.protocol.split_hash,
        record_sha256(list(fit_ids)),
        record_sha256(list(validation_ids)),
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    control_root = (
        parent_run / "controls/frontier_linear_preclassifier" / protocol.content_hash
    )
    for relative in ("work", "state"):
        (control_root / relative).mkdir(parents=True, exist_ok=True)
    for filename, record in (
        ("protocol.json", protocol.as_record()),
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
    ):
        publish_immutable_json(control_root / filename, record)
    publish_immutable_bytes(
        control_root / "config_resolved.json",
        canonical_json_bytes(config.as_record()),
    )
    return FrontierLinearPreclassifierBootstrap(
        project_root,
        config,
        parent,
        parent_result,
        fit_rows,
        validation_rows,
        protocol,
        control_root,
    )


def _phase(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    number: int,
    total: int,
    message: str,
) -> None:
    """Print and persist one human-readable workflow phase."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.control_root / "workflow_events.jsonl",
        "imagenetr50-frontier-linear-preclassifier-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": "imagenetr50-frontier-linear-preclassifier-event-v1",
            "wall_time_unix": time.time(),
        }
    )


def _new_model(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    device: torch.device,
) -> LinearPreclassifierFrontierModel:
    integrator = bootstrap.parent.source.source.integrator
    return LinearPreclassifierFrontierModel(
        nodes,
        slots,
        lambda: create_pinned_backbone(integrator.checkpoint),
        bootstrap.config.source_rank,
        bootstrap.config.source_alpha,
        bootstrap.config.seed,
        device,
    )


def _validated_preflight(
    path: Path, bootstrap: FrontierLinearPreclassifierBootstrap
) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-linear-preclassifier-preflight-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("control_protocol_hash") != bootstrap.protocol.content_hash
        or record.get("test_evaluations") != 0
    ):
        raise ValueError("linear pre-classifier preflight does not authenticate")
    return record


def _preflight(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    model: LinearPreclassifierFrontierModel,
    nodes: Sequence[BehaviorNode],
    frontier_hash: str,
    device: torch.device,
) -> dict[str, object]:
    """Check exact-union initialization, gradient boundaries, data, and memory."""
    target = bootstrap.control_root / "preflight.json"
    if target.is_file():
        return _validated_preflight(target, bootstrap)
    integrator = bootstrap.parent.source.source.integrator
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            bootstrap.fit_rows[: bootstrap.config.microbatch_size],
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
    model.set_evaluation_mode()
    with torch.inference_mode():
        features = model.preclassifier_features(images, False, False)
        combined = model.macro(torch.cat(features, dim=1), model.seen_class_mask)
        union = torch.full_like(combined, -torch.inf)
        for node_model, node_features in zip(
            model.node_models, features, strict=True
        ):
            class_ids = torch.tensor(
                node_model.classifier.class_ids,
                dtype=torch.int64,
                device=device,
            )
            union[:, class_ids] = node_model.classifier(node_features)
        union_error = float(
            torch.max(
                torch.abs(
                    combined[:, model.seen_class_mask]
                    - union[:, model.seen_class_mask]
                )
            )
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        direct = model(images, adapt_lora=False, activation_recomputation=False)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        recomputed = model(images, adapt_lora=True, activation_recomputation=True)
        loss = F.cross_entropy(recomputed, labels)
    loss.backward()
    trainables = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = tuple(parameter.grad for parameter in trainables)
    fit_ids = frozenset(row.image_id for row in bootstrap.fit_rows)
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id
        for row in integrator.manifest.images
        if row.split == "test"
    )
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    integrator_parameters = sum(parameter.numel() for parameter in model.macro.parameters())
    lora_parameters = sum(parameter.numel() for parameter in model.lora_parameters)
    core: dict[str, object] = {
        "activation_recomputation_max_logit_error": float(
            torch.max(
                torch.abs(
                    direct[:, model.seen_class_mask].float()
                    - recomputed[:, model.seen_class_mask].float()
                )
            )
        ),
        "attached_gradient_tensors": sum(value is not None for value in gradients),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "control_protocol_hash": bootstrap.protocol.content_hash,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "finite_gradient_tensors": sum(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        ),
        "fit_examples": len(fit_ids),
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "frontier_hash": frontier_hash,
        "frontier_node_hashes": [node.node_hash for node in nodes],
        "initial_union_max_logit_error": union_error,
        "input_dimension": bootstrap.config.input_dimension,
        "integrator_parameters": integrator_parameters,
        "live_nodes": len(nodes),
        "loss": float(loss.detach()),
        "output_shape": list(recomputed.shape),
        "peak_vram_bytes": int(peak),
        "schema_version": "imagenetr50-frontier-linear-preclassifier-preflight-v1",
        "test_evaluations": 0,
        "test_fit_overlap": len(test_ids & fit_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "trainable_parameters": sum(parameter.numel() for parameter in trainables),
        "trainable_tensors": len(trainables),
        "frontier_lora_parameters": lora_parameters,
        "validation_examples": len(validation_ids),
    }
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["fit_examples"] != bootstrap.config.full_fit_examples
        or core["validation_examples"] != bootstrap.config.validation_examples
        or core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or core["live_nodes"] != bootstrap.config.active_nodes
        or core["input_dimension"] != 3_840
        or integrator_parameters != bootstrap.config.integrator_parameters
        or lora_parameters != bootstrap.config.frontier_lora_parameters
        or core["trainable_parameters"] != bootstrap.config.active_parameters
        or core["trainable_tensors"] != 242
        or core["attached_gradient_tensors"] != len(trainables)
        or core["finite_gradient_tensors"] != len(trainables)
        or core["output_shape"] != [bootstrap.config.microbatch_size, 200]
        or core["activation_recomputation_max_logit_error"] != 0.0
        or union_error > 1e-4
    ):
        raise RuntimeError(f"linear pre-classifier preflight failed: {core}")
    model.zero_grad(set_to_none=True)
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _training_seal(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    hierarchy: HierarchyBuildResult,
) -> dict[str, object]:
    """Prove exact populations, no test access, and unchanged parent bytes."""
    integrator = bootstrap.parent.source.source.integrator
    fit_ids = frozenset(row.image_id for row in bootstrap.fit_rows)
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    parent_run = bootstrap.parent.run
    parent_hashes = {
        "protocol": file_sha256(parent_run / "protocol/protocol.json"),
        "replay": file_sha256(parent_run / "protocol/replay_populations.json"),
        "result": file_sha256(parent_run / "evaluations/result.json"),
    }
    expected = {
        "protocol": bootstrap.config.parent_protocol_sha256,
        "replay": bootstrap.config.parent_replay_sha256,
        "result": bootstrap.config.parent_result_sha256,
    }
    core: dict[str, object] = {
        "fit_examples": len(fit_ids),
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "parent_files_unchanged": parent_hashes == expected,
        "parent_hashes_after": parent_hashes,
        "schema_version": "imagenetr50-frontier-linear-preclassifier-training-seal-v1",
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": hierarchy.work.parent_optimizer_steps,
        "test_evaluations": 0,
        "test_fit_overlap": len(test_ids & fit_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "training_derived_only": all(
            row.split == "train"
            for row in (*bootstrap.fit_rows, *bootstrap.validation_rows)
        ),
        "validation_examples": len(validation_ids),
    }
    if (
        core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or not core["parent_files_unchanged"]
        or not core["training_derived_only"]
        or core["source_hierarchy_leaf_optimizer_steps"] != 0
        or core["source_hierarchy_parent_optimizer_steps"] != 0
    ):
        raise RuntimeError("linear pre-classifier training seal failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(bootstrap.control_root / "training_seal.json", record)
    return record


def _validated_control_result(
    path: Path, bootstrap: FrontierLinearPreclassifierBootstrap
) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    history = (bootstrap.parent.run / str(record.get("history", ""))).resolve()
    artifact = (bootstrap.parent.run / str(record.get("artifact", ""))).resolve()
    architecture = dict(record.get("architecture", {}))
    fit = dict(record.get("fit", {}))
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-linear-preclassifier-control-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("control_protocol_hash") != bootstrap.protocol.content_hash
        or record.get("parent_result_hash")
        != bootstrap.config.parent_result_content_hash
        or record.get("test_evaluations") != 0
        or bootstrap.parent.run not in history.parents
        or bootstrap.parent.run not in artifact.parents
        or file_sha256(history) != record.get("history_sha256")
        or validate_artifact_directory(artifact) != record.get("artifact_sha256")
        or architecture.get("integrator_kind") != bootstrap.config.integrator_kind
        or architecture.get("input_dimension") != bootstrap.config.input_dimension
        or architecture.get("integrator_parameters")
        != bootstrap.config.integrator_parameters
        or architecture.get("trainable_parameters")
        != bootstrap.config.active_parameters
        or fit.get("epochs") != bootstrap.config.epochs
        or fit.get("image_presentations")
        != bootstrap.config.full_fit_examples * bootstrap.config.epochs
    ):
        raise ValueError("linear pre-classifier result does not authenticate")
    return record


def _publish_model(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    model: LinearPreclassifierFrontierModel,
    nodes: Sequence[BehaviorNode],
    fit: AdaptationFit,
    history_path: Path,
    job_hash: str,
) -> tuple[Path, str]:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = bootstrap.control_root / "model"
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(
        tempfile.mkdtemp(prefix="linear-preclassifier-", dir=bootstrap.control_root / "work")
    )
    try:
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in sorted(model.macro.state_dict().items())
            },
            work / "integrator.safetensors",
            metadata={
                "schema_version": "imagenetr50-frontier-linear-preclassifier-model-v1"
            },
        )
        adapters = tuple(
            {
                "filename": filename,
                "node_hash": node.node_hash,
                "sha256": save_adapter(
                    work / filename, adapter_factors(node_model)
                ),
            }
            for index, (node_model, node) in enumerate(
                zip(model.node_models, nodes, strict=True)
            )
            for filename in (f"node_{index:02d}_adapter.safetensors",)
        )
        publish_immutable_json(
            work / "fit.json",
            {
                "adapters": list(adapters),
                "fit": fit.as_record(),
                "history_sha256": file_sha256(history_path),
                "job_hash": job_hash,
                "protocol_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-frontier-linear-preclassifier-fit-v1",
            },
        )
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _fit_or_load(
    bootstrap: FrontierLinearPreclassifierBootstrap,
    hierarchy: HierarchyBuildResult,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    frontier_hash: str,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Train once or authenticate the completed condition without a new model."""
    target = bootstrap.parent.run / CONTROL_RESULT
    if target.is_file():
        return _validated_control_result(target, bootstrap), True
    model = _new_model(bootstrap, nodes, slots, device)
    preflight = _preflight(bootstrap, model, nodes, frontier_hash, device)
    cell = LinearPreclassifierCell(
        bootstrap.config.historical_capacity,
        bootstrap.config.train_frontier_loras,
        bootstrap.config.seed,
    )
    history_path = bootstrap.control_root / "history.jsonl"
    checkpoint_path = bootstrap.control_root / "checkpoint.pt"
    job_hash = record_sha256(
        {
            "cell": cell.as_record(),
            "fit_image_ids": [row.image_id for row in bootstrap.fit_rows],
            "frontier_hash": frontier_hash,
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-frontier-linear-preclassifier-job-v1",
            "validation_image_ids": [
                row.image_id for row in bootstrap.validation_rows
            ],
        }
    )
    integrator = bootstrap.parent.source.source.integrator
    fit, train_metrics, validation_metrics, displacements = fit_adaptation_cell(
        model=model,
        nodes=nodes,
        cell=cell,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=bootstrap.fit_rows,
        validation_rows=bootstrap.validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=bootstrap.config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    model_path, artifact_hash = _publish_model(
        bootstrap, model, nodes, fit, history_path, job_hash
    )
    history = tuple(
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    fixed = next(row for row in history if int(row["epoch"]) == 5)
    seal = _training_seal(bootstrap, hierarchy)
    core: dict[str, object] = {
        "architecture": {
            "active_nodes": bootstrap.config.active_nodes,
            "feature_dimension": bootstrap.config.feature_dimension,
            "feature_normalization": bootstrap.config.feature_normalization,
            "feature_source": bootstrap.config.feature_source,
            "frontier_lora_parameters": bootstrap.config.frontier_lora_parameters,
            "initialization": bootstrap.config.initialization,
            "input_dimension": bootstrap.config.input_dimension,
            "input_order": bootstrap.config.input_order,
            "integrator_kind": bootstrap.config.integrator_kind,
            "integrator_parameters": bootstrap.config.integrator_parameters,
            "output_classes": bootstrap.config.output_classes,
            "source_alpha": bootstrap.config.source_alpha,
            "source_rank": bootstrap.config.source_rank,
            "train_base_vit": bootstrap.config.train_base_vit,
            "train_frontier_loras": bootstrap.config.train_frontier_loras,
            "train_integrator": bootstrap.config.train_integrator,
            "train_node_classifiers": bootstrap.config.train_node_classifiers,
            "trainable_parameters": bootstrap.config.active_parameters,
        },
        "artifact": str(model_path.relative_to(bootstrap.parent.run)),
        "artifact_sha256": artifact_hash,
        "control_protocol": str(
            (bootstrap.control_root / "protocol.json").relative_to(
                bootstrap.parent.run
            )
        ),
        "control_protocol_hash": bootstrap.protocol.content_hash,
        "displacements": [dict(row) for row in displacements],
        "fit": {
            **fit.as_record(),
            "fixed_epoch": 5,
            "fixed_image_presentations": int(fixed["image_presentations"]),
            "fixed_validation_accuracy": float(fixed["validation_accuracy"]),
            "fixed_validation_nll": float(fixed["validation_nll"]),
        },
        "history": str(history_path.relative_to(bootstrap.parent.run)),
        "history_sha256": file_sha256(history_path),
        "parent_result_hash": bootstrap.config.parent_result_content_hash,
        "preflight": dict(preflight),
        "schema_version": "imagenetr50-frontier-linear-preclassifier-control-v1",
        "test_evaluations": 0,
        "train_metrics": train_metrics.as_record(),
        "training": {
            "activation_recomputation": bootstrap.config.activation_recomputation,
            "effective_batch_size": bootstrap.config.effective_batch_size,
            "epochs": bootstrap.config.epochs,
            "gradient_clip_norm": bootstrap.config.gradient_clip_norm,
            "integrator_peak_learning_rate": (
                bootstrap.config.integrator_peak_learning_rate
            ),
            "lora_peak_learning_rate": bootstrap.config.lora_peak_learning_rate,
            "minimum_learning_rate_ratio": (
                bootstrap.config.minimum_learning_rate_ratio
            ),
            "schedule": bootstrap.config.schedule,
            "warmup_fraction": bootstrap.config.warmup_fraction,
            "weight_decay": bootstrap.config.weight_decay,
        },
        "training_seal": dict(seal),
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False


def run_frontier_linear_preclassifier(
    config_path: str | Path = DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG,
) -> Path:
    """Run or resume the linear integrator and update the stage-31 report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the linear pre-classifier ablation requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_frontier_linear_preclassifier(config_path)
    print(
        f"Temporary/resumable artifact directory: {bootstrap.control_root}",
        flush=True,
    )
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(
        total=5,
        desc="ImageNet-R linear frontier integrator",
        unit="phase",
    )
    _phase(bootstrap, 1, 5, "Authenticate the parent and rebuild the fit hierarchy")
    hierarchy = _build(
        bootstrap.parent.source.source, "fit", 50, device, progress=False
    )
    if (
        hierarchy.policy.content_hash
        != bootstrap.parent.source.config.fit_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("source hierarchy was not reused exactly")
    nodes, slots, frontier_hash = _frontier(hierarchy, bootstrap.config.stage)
    overall.update(1)
    _phase(bootstrap, 2, 5, "Verify the exact full-fit and validation populations")
    if (
        len(bootstrap.fit_rows) != bootstrap.config.full_fit_examples
        or len(bootstrap.validation_rows) != bootstrap.config.validation_examples
    ):
        raise RuntimeError("linear pre-classifier populations changed")
    overall.update(1)
    _phase(bootstrap, 3, 5, "Fit or authenticate the adaptive linear integrator")
    result, reused = _fit_or_load(
        bootstrap, hierarchy, nodes, slots, frontier_hash, device
    )
    overall.update(1)
    _phase(bootstrap, 4, 5, "Prove artifact reuse without another optimizer step")
    repeated, was_reused = _fit_or_load(
        bootstrap, hierarchy, nodes, slots, frontier_hash, device
    )
    if repeated != result or not was_reused:
        raise RuntimeError("linear pre-classifier condition did not reuse exactly")
    reuse_core: dict[str, object] = {
        "all_controls_reused": was_reused,
        "new_optimizer_steps": 0,
        "schema_version": "imagenetr50-frontier-linear-preclassifier-reuse-v1",
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": hierarchy.work.parent_optimizer_steps,
    }
    publish_immutable_json(
        bootstrap.control_root / "reuse_proof.json",
        {**reuse_core, "content_hash": record_sha256(reuse_core)},
    )
    overall.update(1)
    _phase(bootstrap, 5, 5, "Integrate the architecture ablation into the report")
    from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
        write_frontier_adaptation_report,
    )

    report = write_frontier_adaptation_report(bootstrap.parent.run)
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.control_root / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "result_hash": result["content_hash"],
                "reused_initially": reused,
                "schema_version": "imagenetr50-frontier-linear-preclassifier-invocation-v1",
            }
        ),
    )
    overall.update(1)
    overall.close()
    action = "authenticated" if reused else "trained"
    print(
        f"Linear pre-classifier integrator {action}; existing report updated in "
        f"{elapsed / 60:.1f} minutes: {report}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    print(run_frontier_linear_preclassifier())


__all__ = [
    "CONTROL_RESULT",
    "FrontierLinearPreclassifierBootstrap",
    "FrontierLinearPreclassifierProtocol",
    "LinearPreclassifierCell",
    "bootstrap_frontier_linear_preclassifier",
    "run_frontier_linear_preclassifier",
]
