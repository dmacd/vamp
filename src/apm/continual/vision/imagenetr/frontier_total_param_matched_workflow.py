"""Run the rank-224 total-parameter-matched stage-31 joint-IID control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from apm.continual.vision.imagenetr.frontier_rank_matched_workflow import (
    FrontierRankMatchedBootstrap,
    _material_paths as _rank_matched_material_paths,
    _validated_control_result as _validated_rank_matched_result,
    bootstrap_frontier_rank_matched,
)
from apm.continual.vision.imagenetr.frontier_total_param_matched_config import (
    DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG,
    FrontierTotalParamMatchedConfig,
    load_frontier_total_param_matched_config,
)
from apm.continual.vision.imagenetr.heads import save_classifier
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    save_adapter,
    trainable_lora_parameters,
)
from apm.continual.vision.imagenetr.macro_convergence_training import (
    JOINT_HISTORY_FORMAT,
    fit_clean_joint_control,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import (
    AdapterVisionModel,
    create_pinned_backbone,
    require_trainable_boundary,
)
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.training import os_cpu_workers


CONTROL_RESULT = Path("evaluations/joint_iid_lora_r224.json")


@dataclass(frozen=True, slots=True)
class FrontierTotalParamMatchedProtocol:
    """Content identity for the additive total-parameter-matched control."""

    parent_run_hash: str
    parent_result_sha256: str
    parent_result_hash: str
    source_result_sha256: str
    source_result_hash: str
    source_protocol_hash: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    fit_image_ids_hash: str
    validation_image_ids_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = (
        "imagenetr50-frontier-total-param-matched-protocol-v1"
    )

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent result file", self.parent_result_sha256),
            ("parent result", self.parent_result_hash),
            ("source result file", self.source_result_sha256),
            ("source result", self.source_result_hash),
            ("source protocol", self.source_protocol_hash),
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
            != "imagenetr50-frontier-total-param-matched-protocol-v1"
        ):
            raise ValueError("total-parameter-matched protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the immutable control namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class FrontierTotalParamMatchedBootstrap:
    """Authenticated source controls, populations, protocol, and output paths."""

    project_root: Path
    config: FrontierTotalParamMatchedConfig
    source: FrontierRankMatchedBootstrap
    source_result: dict[str, object]
    fit_rows: tuple[ImageRecord, ...]
    validation_rows: tuple[ImageRecord, ...]
    protocol: FrontierTotalParamMatchedProtocol
    control_root: Path


def _material_paths(
    project_root: Path,
    config_path: Path,
    source_config_path: Path,
) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    additions = (
        config_path,
        project_root
        / "docs/imagenetr50_frontier_total_param_matched_control_protocol.md",
        project_root
        / "scripts/vision/imagenetr/run_frontier_total_param_matched_control_local.sh",
        package / "frontier_total_param_matched_config.py",
        package / "frontier_total_param_matched_workflow.py",
    )
    return tuple(
        dict.fromkeys(
            (*_rank_matched_material_paths(project_root, source_config_path), *additions)
        )
    )


def bootstrap_frontier_total_param_matched(
    config_path: str | Path = DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG,
) -> FrontierTotalParamMatchedBootstrap:
    """Authenticate both prior controls and prepare the additive rank-224 run."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_frontier_total_param_matched_config(resolved)
    source = bootstrap_frontier_rank_matched(config.source_config)
    source_result_path = source.parent.run / config.source_result
    source_result = _validated_rank_matched_result(source_result_path, source)
    architecture = dict(source_result["architecture"])
    parent_preflight = dict(source.parent_result["preflight"])
    if (
        file_sha256(source_result_path) != config.source_result_sha256
        or source_result["content_hash"] != config.source_result_content_hash
        or source_result["control_protocol_hash"] != config.source_protocol_hash
        or architecture["lora_parameters"] != config.frontier_lora_parameters
        or architecture["classifier_parameters"] != config.classifier_parameters
        or architecture["frontier_integrator_parameters_excluded_from_match"]
        != config.frontier_integrator_parameters
        or parent_preflight["trainable_parameters"]
        != config.frontier_active_parameters
        or source.config.training.epochs != 5
        or source.config.training.batch_size != 64
        or source.config.target_rank != 80
        or source.config.target_alpha != 80
    ):
        raise ValueError(
            "total-parameter control differs from its authenticated sources"
        )
    fit_rows = source.fit_rows
    validation_rows = source.validation_rows
    fit_ids = tuple(row.image_id for row in fit_rows)
    validation_ids = tuple(row.image_id for row in validation_rows)
    code = material_tree_manifest(
        _material_paths(project_root, resolved, config.source_config)
    )
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = FrontierTotalParamMatchedProtocol(
        source.config.parent_run_hash,
        source.config.parent_result_sha256,
        str(source.parent_result["content_hash"]),
        config.source_result_sha256,
        config.source_result_content_hash,
        config.source_protocol_hash,
        source.parent.protocol.dataset_manifest_hash,
        source.parent.protocol.model_manifest_hash,
        source.parent.protocol.split_hash,
        record_sha256(list(fit_ids)),
        record_sha256(list(validation_ids)),
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    control_root = (
        source.parent.run
        / "controls/joint_iid_lora_r224"
        / protocol.content_hash
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
    return FrontierTotalParamMatchedBootstrap(
        project_root,
        config,
        source,
        source_result,
        fit_rows,
        validation_rows,
        protocol,
        control_root,
    )


def _phase(
    bootstrap: FrontierTotalParamMatchedBootstrap,
    number: int,
    total: int,
    message: str,
) -> None:
    """Print and persist one human-readable control phase."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.control_root / "workflow_events.jsonl",
        "imagenetr50-frontier-total-param-matched-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": (
                "imagenetr50-frontier-total-param-matched-event-v1"
            ),
            "wall_time_unix": time.time(),
        }
    )


def _new_model(
    bootstrap: FrontierTotalParamMatchedBootstrap,
    rank: int,
    alpha: int,
) -> AdapterVisionModel:
    integrator = bootstrap.source.parent.source.source.integrator
    return AdapterVisionModel(
        create_pinned_backbone(integrator.checkpoint),
        tuple(range(4 * bootstrap.source.config.stage)),
        rank,
        alpha,
        bootstrap.config.dropout,
        bootstrap.source.config.seed,
    )


def _validated_preflight(
    path: Path,
    bootstrap: FrontierTotalParamMatchedBootstrap,
) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-total-param-matched-preflight-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("control_protocol_hash") != bootstrap.protocol.content_hash
        or record.get("test_evaluations") != 0
    ):
        raise ValueError("total-parameter-matched preflight does not authenticate")
    return record


def _preflight(
    bootstrap: FrontierTotalParamMatchedBootstrap,
    model: AdapterVisionModel,
    device: torch.device,
) -> dict[str, object]:
    """Check closest-rank arithmetic, zero-effect parity, and gradient isolation."""
    target = bootstrap.control_root / "preflight.json"
    if target.is_file():
        return _validated_preflight(target, bootstrap)
    reference = _new_model(
        bootstrap,
        bootstrap.source.config.source_rank,
        bootstrap.source.config.source_rank,
    ).to(device)
    integrator = bootstrap.source.parent.source.source.integrator
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            bootstrap.fit_rows[: bootstrap.source.config.training.batch_size],
            integrator.train_transform,
            bootstrap.source.config.seed,
            0,
        ),
        batch_size=bootstrap.source.config.training.batch_size,
        shuffle=False,
        num_workers=min(bootstrap.config.num_workers, os_cpu_workers()),
        pin_memory=True,
    )
    images, labels, _image_ids = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    model.eval()
    reference.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        target_logits = model(images)
        reference_logits = reference(images)
    parity_error = float(torch.max(torch.abs(target_logits - reference_logits)))
    del reference
    torch.cuda.empty_cache()
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        loss = F.cross_entropy(model(images), labels)
    loss.backward()
    trainables = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    gradients = tuple(parameter.grad for parameter in trainables)
    actual_lora_parameters = sum(
        parameter.numel() for parameter in trainable_lora_parameters(model)
    )
    joint_active_parameters = sum(parameter.numel() for parameter in trainables)
    difference = joint_active_parameters - bootstrap.config.frontier_active_parameters
    core: dict[str, object] = {
        "attached_gradient_tensors": sum(
            value is not None for value in gradients
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "classifier_parameters": (
            model.classifier.weight.numel() + model.classifier.bias.numel()
        ),
        "closest_integer_rank": bootstrap.config.target_rank,
        "control_protocol_hash": bootstrap.protocol.content_hash,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "finite_gradient_tensors": sum(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        ),
        "fit_examples": len(bootstrap.fit_rows),
        "frontier_active_parameters": bootstrap.config.frontier_active_parameters,
        "frontier_aggregate_lora_parameters": (
            bootstrap.config.frontier_lora_parameters
        ),
        "frontier_integrator_parameters": (
            bootstrap.config.frontier_integrator_parameters
        ),
        "lora_alpha": bootstrap.config.target_alpha,
        "lora_parameters": actual_lora_parameters,
        "lora_rank": bootstrap.config.target_rank,
        "lora_scale": (
            bootstrap.config.target_alpha / bootstrap.config.target_rank
        ),
        "loss": float(loss.detach()),
        "parameter_difference": difference,
        "relative_parameter_difference": (
            difference / bootstrap.config.frontier_active_parameters
        ),
        "schema_version": (
            "imagenetr50-frontier-total-param-matched-preflight-v1"
        ),
        "test_evaluations": 0,
        "trainable_parameters": joint_active_parameters,
        "validation_examples": len(bootstrap.validation_rows),
        "zero_lora_max_logit_error_vs_rank16": parity_error,
    }
    expected_lora_parameters = (
        bootstrap.config.target_rank
        * bootstrap.config.lora_parameters_per_rank
    )
    next_rank_difference = abs(
        expected_lora_parameters
        + bootstrap.config.lora_parameters_per_rank
        + bootstrap.config.classifier_parameters
        - bootstrap.config.frontier_active_parameters
    )
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["attached_gradient_tensors"] != 50
        or core["finite_gradient_tensors"] != 50
        or actual_lora_parameters != expected_lora_parameters
        or core["classifier_parameters"] != bootstrap.config.classifier_parameters
        or joint_active_parameters != bootstrap.config.joint_active_parameters
        or difference != bootstrap.config.parameter_difference
        or abs(difference) >= next_rank_difference
        or parity_error != 0.0
    ):
        raise RuntimeError("total-parameter-matched joint-IID preflight failed")
    model.zero_grad(set_to_none=True)
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _training_seal(
    bootstrap: FrontierTotalParamMatchedBootstrap,
) -> dict[str, object]:
    """Prove exact populations, no test access, and unchanged source bytes."""
    integrator = bootstrap.source.parent.source.source.integrator
    fit_ids = frozenset(row.image_id for row in bootstrap.fit_rows)
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    parent_run = bootstrap.source.parent.run
    parent_hashes_after = {
        "protocol": file_sha256(parent_run / "protocol/protocol.json"),
        "replay": file_sha256(parent_run / "protocol/replay_populations.json"),
        "result": file_sha256(parent_run / "evaluations/result.json"),
    }
    expected_parent_hashes = {
        "protocol": bootstrap.source.config.parent_protocol_sha256,
        "replay": bootstrap.source.config.parent_replay_sha256,
        "result": bootstrap.source.config.parent_result_sha256,
    }
    source_result_path = parent_run / bootstrap.config.source_result
    source_result_hash_after = file_sha256(source_result_path)
    core: dict[str, object] = {
        "fit_examples": len(fit_ids),
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "parent_files_unchanged": parent_hashes_after == expected_parent_hashes,
        "parent_hashes_after": parent_hashes_after,
        "schema_version": (
            "imagenetr50-frontier-total-param-matched-training-seal-v1"
        ),
        "source_control_result_sha256_after": source_result_hash_after,
        "source_control_unchanged": (
            source_result_hash_after == bootstrap.config.source_result_sha256
        ),
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
        or not core["source_control_unchanged"]
        or not core["training_derived_only"]
    ):
        raise RuntimeError("total-parameter-matched training seal failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(bootstrap.control_root / "training_seal.json", record)
    return record


def _validated_control_result(
    path: Path,
    bootstrap: FrontierTotalParamMatchedBootstrap,
) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    history = (bootstrap.source.parent.run / str(record.get("history", ""))).resolve()
    artifact = (
        bootstrap.source.parent.run / str(record.get("artifact", ""))
    ).resolve()
    architecture = dict(record.get("architecture", {}))
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-total-param-matched-control-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("control_protocol_hash") != bootstrap.protocol.content_hash
        or record.get("parent_result_hash")
        != bootstrap.source.parent_result["content_hash"]
        or record.get("source_result_hash")
        != bootstrap.config.source_result_content_hash
        or record.get("test_evaluations") != 0
        or bootstrap.source.parent.run not in history.parents
        or bootstrap.source.parent.run not in artifact.parents
        or not history.is_file()
        or record.get("history_sha256") != file_sha256(history)
        or len(ChainedJsonlLedger(history, JOINT_HISTORY_FORMAT).rows) != 5
        or validate_artifact_directory(artifact) != record.get("artifact_sha256")
        or architecture.get("lora_rank") != bootstrap.config.target_rank
        or architecture.get("lora_alpha") != bootstrap.config.target_alpha
        or architecture.get("trainable_parameters")
        != bootstrap.config.joint_active_parameters
        or architecture.get("parameter_difference")
        != bootstrap.config.parameter_difference
    ):
        raise ValueError(
            "total-parameter-matched joint-IID result does not authenticate"
        )
    return record


def _fit_or_load(
    bootstrap: FrontierTotalParamMatchedBootstrap,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Train rank 224 once or authenticate the completed result without a model."""
    target = bootstrap.source.parent.run / CONTROL_RESULT
    if target.is_file():
        return _validated_control_result(target, bootstrap), True
    model = _new_model(
        bootstrap,
        bootstrap.config.target_rank,
        bootstrap.config.target_alpha,
    ).to(device)
    require_trainable_boundary(model)
    preflight = _preflight(bootstrap, model, device)
    integrator = bootstrap.source.parent.source.source.integrator
    history_path = bootstrap.control_root / "history.jsonl"
    checkpoint_path = bootstrap.control_root / "checkpoint.pt"
    fit, fit_metrics, validation_metrics = fit_clean_joint_control(
        model=model,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=bootstrap.fit_rows,
        validation_rows=bootstrap.validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=bootstrap.source.config.training,
        training_seed=bootstrap.source.config.seed + 50_000,
        num_workers=bootstrap.config.num_workers,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=bootstrap.protocol.content_hash,
        device=device,
    )
    model_target = bootstrap.control_root / "model"
    work = Path(
        tempfile.mkdtemp(prefix="joint-r224-", dir=bootstrap.control_root / "work")
    )
    try:
        adapter_sha256 = save_adapter(
            work / "adapter.safetensors", adapter_factors(model)
        )
        classifier_sha256 = save_classifier(
            work / "classifier.safetensors", model.classifier.rows()
        )
        publish_immutable_json(
            work / "fit.json",
            {
                "adapter_sha256": adapter_sha256,
                "classifier_sha256": classifier_sha256,
                "fit": fit.as_record(),
                "history_sha256": file_sha256(history_path),
                "protocol_hash": bootstrap.protocol.content_hash,
                "schema_version": (
                    "imagenetr50-frontier-total-param-matched-fit-v1"
                ),
            },
        )
        artifact_sha256 = publish_artifact_directory(work, model_target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    seal = _training_seal(bootstrap)
    source_architecture = dict(bootstrap.source_result["architecture"])
    lora_parameters = (
        bootstrap.config.target_rank
        * bootstrap.config.lora_parameters_per_rank
    )
    core: dict[str, object] = {
        "architecture": {
            "classifier_parameters": bootstrap.config.classifier_parameters,
            "frontier_active_parameters": (
                bootstrap.config.frontier_active_parameters
            ),
            "frontier_adapter_count": int(
                source_architecture["frontier_adapter_count"]
            ),
            "frontier_aggregate_lora_parameters": (
                bootstrap.config.frontier_lora_parameters
            ),
            "frontier_integrator_parameters_included_in_match": (
                bootstrap.config.frontier_integrator_parameters
            ),
            "lora_alpha": bootstrap.config.target_alpha,
            "lora_parameters": lora_parameters,
            "lora_rank": bootstrap.config.target_rank,
            "lora_scale": (
                bootstrap.config.target_alpha / bootstrap.config.target_rank
            ),
            "match_scope": bootstrap.config.match_scope,
            "parameter_difference": bootstrap.config.parameter_difference,
            "relative_parameter_difference": (
                bootstrap.config.parameter_difference
                / bootstrap.config.frontier_active_parameters
            ),
            "source_rank": bootstrap.source.config.source_rank,
            "trainable_parameters": bootstrap.config.joint_active_parameters,
        },
        "artifact": str(model_target.relative_to(bootstrap.source.parent.run)),
        "artifact_sha256": artifact_sha256,
        "control_protocol": str(
            (bootstrap.control_root / "protocol.json").relative_to(
                bootstrap.source.parent.run
            )
        ),
        "control_protocol_hash": bootstrap.protocol.content_hash,
        "fit": fit.as_record(),
        "fit_metrics": fit_metrics.as_record(),
        "history": str(history_path.relative_to(bootstrap.source.parent.run)),
        "history_sha256": file_sha256(history_path),
        "parent_result_hash": bootstrap.source.parent_result["content_hash"],
        "preflight": dict(preflight),
        "schema_version": (
            "imagenetr50-frontier-total-param-matched-control-v1"
        ),
        "source_result_hash": bootstrap.config.source_result_content_hash,
        "test_evaluations": 0,
        "training": asdict(bootstrap.source.config.training),
        "training_seal": dict(seal),
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False
def run_frontier_total_param_matched_control(
    config_path: str | Path = DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG,
) -> Path:
    """Run or resume rank-224 joint IID and update the existing stage-31 report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the total-parameter-matched control requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_frontier_total_param_matched(config_path)
    print(
        f"Temporary/resumable artifact directory: {bootstrap.control_root}",
        flush=True,
    )
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(
        total=4,
        desc="ImageNet-R total-parameter control",
        unit="phase",
    )
    _phase(bootstrap, 1, 4, "Authenticate the frontier and rank-80 source control")
    overall.update(1)
    _phase(bootstrap, 2, 4, "Fit or authenticate one rank-224 joint-IID adapter")
    result, reused = _fit_or_load(bootstrap, device)
    overall.update(1)
    _phase(bootstrap, 3, 4, "Prove completed-control reuse without optimizer work")
    repeated_result, repeated = _fit_or_load(bootstrap, device)
    if repeated_result != result or not repeated:
        raise RuntimeError("total-parameter-matched control did not reuse exactly")
    reuse_core: dict[str, object] = {
        "all_controls_reused": repeated,
        "new_optimizer_steps": 0,
        "schema_version": (
            "imagenetr50-frontier-total-param-matched-reuse-v1"
        ),
    }
    publish_immutable_json(
        bootstrap.control_root / "reuse_proof.json",
        {**reuse_core, "content_hash": record_sha256(reuse_core)},
    )
    overall.update(1)
    _phase(bootstrap, 4, 4, "Integrate rank 224 into the existing stage-31 report")
    from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
        write_frontier_adaptation_report,
    )

    report = write_frontier_adaptation_report(bootstrap.source.parent.run)
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.control_root / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "result_hash": result["content_hash"],
                "reused_initially": reused,
                "schema_version": (
                    "imagenetr50-frontier-total-param-matched-invocation-v1"
                ),
            }
        ),
    )
    overall.update(1)
    overall.close()
    action = "authenticated" if reused else "trained"
    print(
        f"Total-parameter control {action}; existing report updated in "
        f"{elapsed / 60:.1f} minutes: {report}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    print(run_frontier_total_param_matched_control())


__all__ = [
    "CONTROL_RESULT",
    "FrontierTotalParamMatchedBootstrap",
    "FrontierTotalParamMatchedProtocol",
    "bootstrap_frontier_total_param_matched",
    "run_frontier_total_param_matched_control",
]
