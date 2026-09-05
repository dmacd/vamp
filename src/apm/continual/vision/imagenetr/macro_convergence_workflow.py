"""Clean stage-31 convergence audit for the ImageNet-R macro-token integrator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import tempfile
import time

import torch

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
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.integrator_workflow import (
    _hierarchy_frontier,
    _partition_rows,
)
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.heads import save_classifier
from apm.continual.vision.imagenetr.macro_convergence_config import (
    DEFAULT_MACRO_CONVERGENCE_CONFIG,
    MacroConvergenceConfig,
    load_macro_convergence_config,
)
from apm.continual.vision.imagenetr.macro_convergence_training import (
    ConvergenceFit,
    MacroConvergenceCell,
    fit_clean_joint_control,
    fit_macro_convergence_cell,
)
from apm.continual.vision.imagenetr.macro_token_cache import (
    MacroTokenPopulation,
    clear_macro_population,
    materialize_macro_population,
)
from apm.continual.vision.imagenetr.macro_token_model import (
    MacroTokenClassifier,
    parameter_count,
)
from apm.continual.vision.imagenetr.macro_token_workflow import (
    MacroTokenBootstrap,
    MacroTokenProtocol,
    bootstrap_macro_token,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    _build,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_features import test_transform_hash


@dataclass(frozen=True, slots=True)
class MacroConvergenceProtocol:
    """Content identity for a clean-only convergence audit."""

    source_macro_run_hash: str
    source_macro_protocol_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-macro-convergence-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("source macro run", self.source_macro_run_hash),
            ("source macro protocol", self.source_macro_protocol_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-macro-convergence-protocol-v1":
            raise ValueError("macro convergence protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the complete experiment namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class MacroConvergenceBootstrap:
    """Authenticated source experiment and isolated convergence run paths."""

    project_root: Path
    config: MacroConvergenceConfig
    source: MacroTokenBootstrap
    protocol: MacroConvergenceProtocol
    run: Path

    @property
    def scratch_root(self) -> Path:
        """Return the isolated reproducible macro-token cache root."""
        return self.run / "scratch/macro_tokens"


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "configs/vision/imagenetr/primary.yaml",
        project_root / "docs/imagenetr50_macro_token_convergence_protocol.md",
        project_root / "scripts/vision/imagenetr/run_macro_convergence_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *(sorted(package.glob("macro_convergence_*.py"))),
        *(package / name for name in (
            "artifacts.py",
            "checkpoints.py",
            "config.py",
            "data.py",
            "heads.py",
            "integrator_hierarchy.py",
            "integrator_workflow.py",
            "lora.py",
            "macro_token_cache.py",
            "macro_token_config.py",
            "macro_token_model.py",
            "macro_token_training.py",
            "macro_token_workflow.py",
            "manifests.py",
            "model.py",
            "promoted_integrator_workflow.py",
            "router_features.py",
            "training.py",
        )),
    )


def _prepare_run(run: Path, protocol: MacroConvergenceProtocol) -> None:
    for relative in (
        "protocol",
        "models/macro",
        "models/joint",
        "evaluations/macro_cells",
        "histories",
        "checkpoints",
        "ledgers",
        "reports",
        "scratch/macro_tokens",
        "state",
        "work",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(run / "protocol/protocol.json", protocol.as_record())


def bootstrap_macro_convergence(
    config_path: str | Path = DEFAULT_MACRO_CONVERGENCE_CONFIG,
) -> MacroConvergenceBootstrap:
    """Authenticate v8 and prepare the isolated clean-only experiment."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_macro_convergence_config(resolved)
    current_source = bootstrap_macro_token(config.source_macro_config)
    configured_source = (
        config.source_macro_artifact_root / "runs" / config.source_macro_run_hash
    )
    source_protocol = configured_source / "protocol/protocol.json"
    source_hierarchy = configured_source / "protocol/clean_hierarchy.json"
    source_clean_stage = configured_source / "evaluations/clean/stage_031.json"
    stored_source_record = load_canonical_json(source_protocol)
    stored_source = MacroTokenProtocol(
        **{
            key: value
            for key, value in stored_source_record.items()
            if key != "content_hash"
        }
    )
    if (
        stored_source.content_hash != config.source_macro_run_hash
        or stored_source_record.get("content_hash") != stored_source.content_hash
        or current_source.config.config_hash != stored_source.config_hash
        or current_source.source.integrator.manifest.content_hash
        != stored_source.dataset_manifest_hash
        or current_source.source.integrator.protocol.model_manifest_hash
        != stored_source.model_manifest_hash
        or current_source.source.integrator.split.content_hash != stored_source.split_hash
        or file_sha256(source_protocol) != config.source_macro_protocol_sha256
        or file_sha256(source_hierarchy) != config.source_clean_hierarchy_sha256
        or file_sha256(source_clean_stage) != config.source_clean_stage31_sha256
        or load_canonical_json(source_clean_stage).get("fitting_population")
        != config.shuffle_population_hash
    ):
        raise ValueError("configured v8 source artifacts changed")
    source = replace(current_source, protocol=stored_source, run=configured_source)
    atomic_write(
        config.source_macro_artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": config.source_macro_run_hash,
                "schema_version": "imagenetr50-macro-token-latest-v1",
            }
        ),
    )
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    integrator = source.source.integrator
    protocol = MacroConvergenceProtocol(
        config.source_macro_run_hash,
        config.source_macro_protocol_sha256,
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
        ("source_macro_protocol.json", load_canonical_json(source_protocol)),
        ("router_split.json", integrator.split.as_record()),
    ):
        publish_immutable_json(run / "protocol" / filename, record)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-macro-convergence-latest-v1",
            }
        ),
    )
    return MacroConvergenceBootstrap(project_root, config, source, protocol, run)


def _phase(
    bootstrap: MacroConvergenceBootstrap, number: int, total: int, message: str
) -> None:
    """Print and durably append one human-readable workflow phase transition."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.run / "ledgers/workflow_events.jsonl",
        "imagenetr50-macro-convergence-workflow-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": "imagenetr50-macro-convergence-workflow-event-v1",
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
                "schema_version": "imagenetr50-macro-convergence-workflow-state-v1",
            }
        ),
    )


def _clean_rows(
    bootstrap: MacroConvergenceBootstrap, partition: str
) -> tuple[ImageRecord, ...]:
    if partition not in {"fit", "validation"}:
        raise ValueError("convergence audit may request only fit or validation rows")
    return _partition_rows(
        bootstrap.source.source.integrator,
        partition,
        tuple(range(bootstrap.config.stage)),
    )


def _preflight(bootstrap: MacroConvergenceBootstrap, device: torch.device) -> dict[str, object]:
    """Validate the GPU, clean split, architecture, and sealed test boundary."""
    target = bootstrap.run / "protocol/preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    fit_rows = _clean_rows(bootstrap, "fit")
    validation_rows = _clean_rows(bootstrap, "validation")
    fit_ids = frozenset(row.image_id for row in fit_rows)
    validation_ids = frozenset(row.image_id for row in validation_rows)
    test_ids = frozenset(
        row.image_id
        for row in bootstrap.source.source.integrator.manifest.images
        if row.split == "test"
    )
    model = MacroTokenClassifier(
        bootstrap.config.depth,
        bootstrap.config.dropout,
        bootstrap.config.screening_seed,
    )
    core: dict[str, object] = {
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "fit_examples": len(fit_rows),
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "macro_parameters": parameter_count(model),
        "matrix_cells": len(bootstrap.config.matrix),
        "schema_version": "imagenetr50-macro-convergence-preflight-v1",
        "stage": bootstrap.config.stage,
        "test_fit_overlap": len(test_ids & fit_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "validation_examples": len(validation_rows),
    }
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["fit_examples"] != 12_194
        or core["validation_examples"] != 3_049
        or core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or core["macro_parameters"] != 12_055_496
        or core["matrix_cells"] != 9
    ):
        raise RuntimeError("macro convergence preflight failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _population(
    bootstrap: MacroConvergenceBootstrap,
    hierarchy: HierarchyBuildResult,
    partition: str,
    device: torch.device,
) -> MacroTokenPopulation:
    """Materialize one isolated clean population without accepting test rows."""
    rows = _clean_rows(bootstrap, partition)
    nodes, slots, frontier_hash = _hierarchy_frontier(
        hierarchy, bootstrap.config.stage
    )
    integrator = bootstrap.source.source.integrator
    return materialize_macro_population(
        protocol_hash=bootstrap.config.source_macro_run_hash,
        frontier_hash=frontier_hash,
        partition=partition,
        nodes=nodes,
        slot_indices=slots,
        rows=rows,
        prepared_root=integrator.config.data_root / "imagenet-r",
        transform=integrator.test_transform,
        transform_hash=test_transform_hash(integrator.primary_config.input_size),
        model_hash=integrator.protocol.model_manifest_hash,
        backbone_factory=lambda: create_pinned_backbone(integrator.checkpoint),
        scratch_root=bootstrap.scratch_root,
        request_ledger=ChainedJsonlLedger(
            bootstrap.run / "ledgers/macro_token_requests.jsonl",
            "imagenetr50-macro-token-request-v1",
        ),
        rank=integrator.primary_config.lora_rank,
        alpha=integrator.primary_config.lora_alpha,
        shard_size=bootstrap.config.cache_shard_size,
        batch_size=bootstrap.config.feature_batch_size,
        num_workers=bootstrap.config.num_workers,
        cache_limit_bytes=bootstrap.config.cache_limit_bytes,
        device=device,
    )


def _validated_record(path: Path, schema_version: str) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version") != schema_version
        or record.get("content_hash") != record_sha256(core)
    ):
        raise ValueError(f"completed convergence record changed: {path}")
    return record


def _macro_job_hash(
    bootstrap: MacroConvergenceBootstrap,
    cell: MacroConvergenceCell,
    fitting: MacroTokenPopulation,
    validation: MacroTokenPopulation,
) -> str:
    return record_sha256(
        {
            "cell": cell.as_record(),
            "fit_population": fitting.identity,
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-macro-convergence-job-v1",
            "validation_population": validation.identity,
        }
    )


def _publish_macro_model(
    bootstrap: MacroConvergenceBootstrap,
    model: MacroTokenClassifier,
    cell: MacroConvergenceCell,
    fit: ConvergenceFit,
    job_hash: str,
    fitting: MacroTokenPopulation,
    validation: MacroTokenPopulation,
    history_path: Path,
) -> tuple[Path, str]:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = bootstrap.run / "models/macro" / job_hash
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(tempfile.mkdtemp(prefix="macro-cell-", dir=bootstrap.run / "work"))
    try:
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in sorted(model.state_dict().items())
            },
            work / "model.safetensors",
            metadata={"schema_version": "imagenetr50-macro-convergence-model-v1"},
        )
        publish_immutable_json(
            work / "fit.json",
            {
                "cell": cell.as_record(),
                "fit": fit.as_record(),
                "fit_population": fitting.identity,
                "history_sha256": file_sha256(history_path),
                "job_hash": job_hash,
                "parameter_count": parameter_count(model),
                "schema_version": "imagenetr50-macro-convergence-fit-v1",
                "validation_population": validation.identity,
            },
        )
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _fit_or_load_macro(
    bootstrap: MacroConvergenceBootstrap,
    cell: MacroConvergenceCell,
    fitting: MacroTokenPopulation,
    validation: MacroTokenPopulation,
    role: str,
    device: torch.device,
) -> dict[str, object]:
    """Fit or authenticate one complete macro-token schedule."""
    job_hash = _macro_job_hash(bootstrap, cell, fitting, validation)
    target = bootstrap.run / "evaluations/macro_cells" / f"{job_hash}.json"
    if target.is_file():
        record = _validated_record(target, "imagenetr50-macro-convergence-cell-v1")
        validate_artifact_directory(bootstrap.run / str(record["artifact"]))
        return record
    model = MacroTokenClassifier(
        cell.depth, bootstrap.config.dropout, cell.seed
    ).to(device)
    history_path = bootstrap.run / "histories" / f"{job_hash}.jsonl"
    checkpoint_path = bootstrap.run / "checkpoints" / f"{job_hash}.pt"
    fit = fit_macro_convergence_cell(
        model=model,
        cell=cell,
        training=fitting,
        validation=validation,
        dropout=bootstrap.config.dropout,
        weight_decay=bootstrap.config.weight_decay,
        gradient_clip_norm=bootstrap.config.gradient_clip_norm,
        warmup_fraction=(
            bootstrap.config.warmup_fraction
            if cell.schedule == "warmup_cosine"
            else 0.0
        ),
        minimum_learning_rate_ratio=bootstrap.config.minimum_learning_rate_ratio,
        shuffle_population_hash=bootstrap.config.shuffle_population_hash,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    model_path, artifact_hash = _publish_macro_model(
        bootstrap,
        model,
        cell,
        fit,
        job_hash,
        fitting,
        validation,
        history_path,
    )
    core: dict[str, object] = {
        "artifact": str(model_path.relative_to(bootstrap.run)),
        "artifact_sha256": artifact_hash,
        "cell": cell.as_record(),
        "fit": fit.as_record(),
        "history": str(history_path.relative_to(bootstrap.run)),
        "history_sha256": file_sha256(history_path),
        "job_hash": job_hash,
        "role": role,
        "schema_version": "imagenetr50-macro-convergence-cell-v1",
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record


def _select_winner(
    bootstrap: MacroConvergenceBootstrap,
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select the lowest clean validation NLL with frozen deterministic ties."""
    winner = min(
        candidates,
        key=lambda row: (
            float(dict(row["fit"])["validation_nll"]),
            int(dict(row["cell"])["effective_batch_size"]),
            float(dict(row["cell"])["peak_learning_rate"]),
        ),
    )
    core: dict[str, object] = {
        "criterion": "lowest_best_validation_nll_then_batch_then_learning_rate",
        "schema_version": "imagenetr50-macro-convergence-selection-v1",
        "screening_seed": bootstrap.config.screening_seed,
        "winner": {
            "effective_batch_size": int(
                dict(winner["cell"])["effective_batch_size"]
            ),
            "peak_learning_rate": float(
                dict(winner["cell"])["peak_learning_rate"]
            ),
            "schedule": str(dict(winner["cell"])["schedule"]),
            "validation_accuracy": float(dict(winner["fit"])["validation_accuracy"]),
            "validation_nll": float(dict(winner["fit"])["validation_nll"]),
        },
    }
    record = {**core, "content_hash": record_sha256(core)}
    path = bootstrap.run / "protocol/selection.json"
    if path.is_file() and load_canonical_json(path) != record:
        raise ValueError("macro convergence selection changed")
    publish_immutable_json(path, record)
    return record


def _joint_job_hash(
    bootstrap: MacroConvergenceBootstrap,
    fit_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
) -> str:
    integrator = bootstrap.source.source.integrator
    return record_sha256(
        {
            "fit_image_ids": [row.image_id for row in fit_rows],
            "model_seed": integrator.primary_config.seed,
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-clean-joint-job-v1",
            "training": asdict(integrator.primary_config.joint_training),
            "training_seed": integrator.primary_config.seed + 50_000,
            "validation_image_ids": [row.image_id for row in validation_rows],
        }
    )


def _run_clean_joint(
    bootstrap: MacroConvergenceBootstrap, device: torch.device
) -> dict[str, object]:
    """Fit or authenticate the same-split five-epoch joint-IID control."""
    target = bootstrap.run / "evaluations/clean_joint.json"
    if target.is_file():
        record = _validated_record(target, "imagenetr50-macro-convergence-joint-v1")
        validate_artifact_directory(bootstrap.run / str(record["artifact"]))
        return record
    integrator = bootstrap.source.source.integrator
    fit_rows = _clean_rows(bootstrap, "fit")
    validation_rows = _clean_rows(bootstrap, "validation")
    job_hash = _joint_job_hash(bootstrap, fit_rows, validation_rows)
    model = AdapterVisionModel(
        create_pinned_backbone(integrator.checkpoint),
        tuple(range(4 * bootstrap.config.stage)),
        integrator.primary_config.lora_rank,
        integrator.primary_config.lora_alpha,
        integrator.primary_config.lora_dropout,
        integrator.primary_config.seed,
    ).to(device)
    history_path = bootstrap.run / "histories/clean_joint_iid.jsonl"
    checkpoint_path = bootstrap.run / "checkpoints/clean_joint_iid.pt"
    fit, train_metrics, validation_metrics = fit_clean_joint_control(
        model=model,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=fit_rows,
        validation_rows=validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=integrator.primary_config.joint_training,
        training_seed=integrator.primary_config.seed + 50_000,
        num_workers=bootstrap.config.num_workers,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    model_target = bootstrap.run / "models/joint" / job_hash
    if model_target.is_dir():
        artifact_hash = validate_artifact_directory(model_target)
    else:
        work = Path(
            tempfile.mkdtemp(prefix="clean-joint-", dir=bootstrap.run / "work")
        )
        try:
            adapter_sha256 = save_adapter(work / "adapter.safetensors", adapter_factors(model))
            classifier_sha256 = save_classifier(
                work / "classifier.safetensors", model.classifier.rows()
            )
            publish_immutable_json(
                work / "fit.json",
                {
                    "adapter_sha256": adapter_sha256,
                    "classifier_sha256": classifier_sha256,
                    "fit": fit.as_record(),
                    "fit_image_ids_hash": record_sha256(
                        [row.image_id for row in fit_rows]
                    ),
                    "history_sha256": file_sha256(history_path),
                    "job_hash": job_hash,
                    "schema_version": "imagenetr50-clean-joint-fit-v1",
                    "validation_image_ids_hash": record_sha256(
                        [row.image_id for row in validation_rows]
                    ),
                },
            )
            artifact_hash = publish_artifact_directory(work, model_target)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    core: dict[str, object] = {
        "artifact": str(model_target.relative_to(bootstrap.run)),
        "artifact_sha256": artifact_hash,
        "fit": fit.as_record(),
        "fit_metrics": train_metrics.as_record(),
        "history": str(history_path.relative_to(bootstrap.run)),
        "history_sha256": file_sha256(history_path),
        "job_hash": job_hash,
        "schema_version": "imagenetr50-macro-convergence-joint-v1",
        "training": asdict(integrator.primary_config.joint_training),
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record


def _request_seal(bootstrap: MacroConvergenceBootstrap) -> dict[str, object]:
    """Prove every adapted-token request used only training-derived identities."""
    ledger = ChainedJsonlLedger(
        bootstrap.run / "ledgers/macro_token_requests.jsonl",
        "imagenetr50-macro-token-request-v1",
    )
    partitions = tuple(str(row["partition"]) for row in ledger.rows)
    split_values = tuple(
        split for row in ledger.rows for split in tuple(row.get("splits", ()))
    )
    core: dict[str, object] = {
        "partitions": list(partitions),
        "request_rows": len(ledger.rows),
        "schema_version": "imagenetr50-macro-convergence-training-seal-v1",
        "test_image_requests": sum(split == "test" for split in split_values),
        "training_derived_only": set(partitions) <= {"fit", "validation"}
        and set(split_values) <= {"train"},
    }
    if not core["training_derived_only"] or core["test_image_requests"] != 0:
        raise RuntimeError("macro convergence audit crossed the test boundary")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(bootstrap.run / "protocol/training_seal.json", record)
    return record


def _publish_result(
    bootstrap: MacroConvergenceBootstrap,
    preflight: Mapping[str, object],
    legacy: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    replications: Sequence[Mapping[str, object]],
    joint: Mapping[str, object],
    seal: Mapping[str, object],
    reuse_proof: Mapping[str, object],
) -> dict[str, object]:
    core: dict[str, object] = {
        "clean_joint": dict(joint),
        "legacy_control": dict(legacy),
        "preflight": dict(preflight),
        "replications": [dict(row) for row in replications],
        "reuse_proof": dict(reuse_proof),
        "schema_version": "imagenetr50-macro-convergence-result-v1",
        "screening_candidates": [dict(row) for row in candidates],
        "selection": dict(selection),
        "stage": bootstrap.config.stage,
        "test_evaluations": 0,
        "training_seal": dict(seal),
    }
    record = {**core, "content_hash": record_sha256(core)}
    path = bootstrap.run / "evaluations/result.json"
    if path.is_file() and load_canonical_json(path) != record:
        raise ValueError("macro convergence result changed")
    publish_immutable_json(path, record)
    return record


def run_macro_convergence(
    config_path: str | Path = DEFAULT_MACRO_CONVERGENCE_CONFIG,
) -> Path:
    """Run or exactly resume the clean stage-31 macro-token convergence audit."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the macro convergence audit requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_macro_convergence(config_path)
    print(f"Temporary/resumable artifact directory: {bootstrap.run}", flush=True)
    completed_result = bootstrap.run / "evaluations/result.json"
    if completed_result.is_file():
        _validated_record(
            completed_result, "imagenetr50-macro-convergence-result-v1"
        )
        from apm.continual.vision.imagenetr.macro_convergence_reporting import (
            write_macro_convergence_report,
        )

        write_macro_convergence_report(bootstrap.run)
        print("Completed audit authenticated; no optimizer work was repeated.", flush=True)
        return bootstrap.run

    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=9, desc="ImageNet-R macro convergence", unit="phase")
    _phase(bootstrap, 1, 9, "Authenticate the clean split and real BF16 GPU")
    preflight = _preflight(bootstrap, device)
    overall.update(1)

    _phase(bootstrap, 2, 9, "Reuse the complete fit-only hierarchy with zero training")
    hierarchy = _build(bootstrap.source.source, "fit", 50, device, progress=False)
    if (
        hierarchy.policy.content_hash
        != bootstrap.source.config.fit_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("clean hierarchy was not reused exactly")
    overall.update(1)

    _phase(bootstrap, 3, 9, "Materialize stage-31 fit and validation macro tokens")
    fitting, validation = tuple(
        _population(bootstrap, hierarchy, partition, device)
        for partition in ("fit", "validation")
    )
    overall.update(1)

    fit_progress = tqdm(total=12, desc="Convergence fits", unit="fit")
    _phase(bootstrap, 4, 9, "Repeat the original constant-learning-rate control")
    legacy_cell = MacroConvergenceCell(
        "legacy_constant",
        bootstrap.config.legacy_effective_batch_size,
        bootstrap.config.legacy_learning_rate,
        bootstrap.config.legacy_epochs,
        bootstrap.config.screening_seed,
        bootstrap.config.depth,
    )
    legacy = _fit_or_load_macro(
        bootstrap, legacy_cell, fitting, validation, "legacy_control", device
    )
    fit_progress.update(1)
    overall.update(1)

    _phase(bootstrap, 5, 9, "Run the nine-cell warmup-cosine schedule matrix")
    candidates = []
    for index, (batch_size, learning_rate) in enumerate(
        bootstrap.config.matrix, start=1
    ):
        print(
            f"Schedule candidate {index}/{len(bootstrap.config.matrix)}: "
            f"batch={batch_size}, peak_lr={learning_rate:g}",
            flush=True,
        )
        candidates.append(
            _fit_or_load_macro(
                bootstrap,
                MacroConvergenceCell(
                    bootstrap.config.schedule,
                    batch_size,
                    learning_rate,
                    bootstrap.config.epochs,
                    bootstrap.config.screening_seed,
                    bootstrap.config.depth,
                ),
                fitting,
                validation,
                "screening",
                device,
            )
        )
        fit_progress.update(1)
    overall.update(1)

    _phase(bootstrap, 6, 9, "Select by clean NLL and repeat the winner over three seeds")
    selection = _select_winner(bootstrap, candidates)
    winner = dict(selection["winner"])
    screening_winner = next(
        row
        for row in candidates
        if int(dict(row["cell"])["effective_batch_size"])
        == int(winner["effective_batch_size"])
        and float(dict(row["cell"])["peak_learning_rate"])
        == float(winner["peak_learning_rate"])
    )
    replications = [screening_winner]
    for seed in bootstrap.config.replication_seeds[1:]:
        replications.append(
            _fit_or_load_macro(
                bootstrap,
                MacroConvergenceCell(
                    str(winner["schedule"]),
                    int(winner["effective_batch_size"]),
                    float(winner["peak_learning_rate"]),
                    bootstrap.config.epochs,
                    seed,
                    bootstrap.config.depth,
                ),
                fitting,
                validation,
                "replication",
                device,
            )
        )
        fit_progress.update(1)
    overall.update(1)

    _phase(bootstrap, 7, 9, "Train the same-split five-epoch joint-IID control")
    joint = _run_clean_joint(bootstrap, device)
    fit_progress.update(1)
    fit_progress.close()
    overall.update(1)

    _phase(bootstrap, 8, 9, "Seal the test boundary and prove token-cache reuse")
    repeated_fit, repeated_validation = tuple(
        _population(bootstrap, hierarchy, partition, device)
        for partition in ("fit", "validation")
    )
    reuse_core: dict[str, object] = {
        "cache_hits": repeated_fit.cache_hits + repeated_validation.cache_hits,
        "cache_misses": repeated_fit.cache_misses + repeated_validation.cache_misses,
        "fit_population_unchanged": repeated_fit.identity == fitting.identity,
        "node_example_forwards": (
            repeated_fit.node_example_forwards
            + repeated_validation.node_example_forwards
        ),
        "schema_version": "imagenetr50-macro-convergence-reuse-v1",
        "validation_population_unchanged": repeated_validation.identity
        == validation.identity,
    }
    if (
        reuse_core["cache_misses"] != 0
        or reuse_core["node_example_forwards"] != 0
        or not reuse_core["fit_population_unchanged"]
        or not reuse_core["validation_population_unchanged"]
    ):
        raise RuntimeError("macro convergence token cache was not reused exactly")
    reuse_proof = {**reuse_core, "content_hash": record_sha256(reuse_core)}
    publish_immutable_json(bootstrap.run / "protocol/reuse_proof.json", reuse_proof)
    seal = _request_seal(bootstrap)
    result = _publish_result(
        bootstrap,
        preflight,
        legacy,
        candidates,
        selection,
        replications,
        joint,
        seal,
        reuse_proof,
    )
    overall.update(1)

    _phase(bootstrap, 9, 9, "Generate the publication-style clean audit report")
    from apm.continual.vision.imagenetr.macro_convergence_reporting import (
        write_macro_convergence_report,
    )

    write_macro_convergence_report(bootstrap.run)
    for population in (fitting, validation):
        clear_macro_population(population, bootstrap.scratch_root)
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.run / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "result_hash": result["content_hash"],
                "schema_version": "imagenetr50-macro-convergence-invocation-v1",
            }
        ),
    )
    atomic_write(
        bootstrap.run / "state/workflow.json",
        canonical_json_bytes(
            {
                "phase": 9,
                "result_hash": result["content_hash"],
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-macro-convergence-complete-v1",
                "status": "COMPLETE",
            }
        ),
    )
    overall.update(1)
    overall.close()
    print(f"Clean convergence report complete in {elapsed / 60:.1f} minutes.", flush=True)
    return bootstrap.run


if __name__ == "__main__":
    print(run_macro_convergence())


__all__ = [
    "MacroConvergenceBootstrap",
    "MacroConvergenceProtocol",
    "bootstrap_macro_convergence",
    "run_macro_convergence",
]
