"""Run the rank-80 joint-IID control attached to the stage-31 report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from apm.continual.vision.imagenetr.frontier_adaptation_workflow import (
    FrontierAdaptationBootstrap,
    _clean_rows,
    bootstrap_frontier_adaptation,
)
from apm.continual.vision.imagenetr.frontier_rank_matched_config import (
    DEFAULT_FRONTIER_RANK_MATCHED_CONFIG,
    FrontierRankMatchedConfig,
    load_frontier_rank_matched_config,
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


CONTROL_RESULT = Path("evaluations/joint_iid_lora_r80.json")
LORA_PARAMETERS_PER_RANK = 82_944
CLASSIFIER_PARAMETERS = 95_356
MACRO_PARAMETERS = 12_055_496


def vit_lora_parameter_count(rank: int) -> int:
    """Return QKV-plus-fc1 LoRA parameters for all 12 ViT-B/16 blocks."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    return LORA_PARAMETERS_PER_RANK * rank


@dataclass(frozen=True, slots=True)
class FrontierRankMatchedProtocol:
    """Content identity for the additive rank-matched joint-IID control."""

    parent_run_hash: str
    parent_result_sha256: str
    parent_result_hash: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    fit_image_ids_hash: str
    validation_image_ids_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-frontier-rank-matched-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent result file", self.parent_result_sha256),
            ("parent result", self.parent_result_hash),
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
        if self.schema_version != "imagenetr50-frontier-rank-matched-protocol-v1":
            raise ValueError("rank-matched protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the immutable control namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class FrontierRankMatchedBootstrap:
    """Authenticated parent run, exact populations, and control paths."""

    project_root: Path
    config: FrontierRankMatchedConfig
    parent: FrontierAdaptationBootstrap
    parent_result: dict[str, object]
    fit_rows: tuple[ImageRecord, ...]
    validation_rows: tuple[ImageRecord, ...]
    protocol: FrontierRankMatchedProtocol
    control_root: Path


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/imagenetr50_frontier_rank_matched_control_protocol.md",
        project_root
        / "scripts/vision/imagenetr/run_frontier_rank_matched_control_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *(package / name for name in (
            "artifacts.py",
            "checkpoints.py",
            "config.py",
            "data.py",
            "frontier_adaptation_config.py",
            "frontier_adaptation_workflow.py",
            "frontier_rank_matched_config.py",
            "frontier_rank_matched_workflow.py",
            "heads.py",
            "integrator_workflow.py",
            "lora.py",
            "macro_convergence_training.py",
            "manifests.py",
            "model.py",
            "protocol.py",
            "training.py",
        )),
        package / "merging/common.py",
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


def bootstrap_frontier_rank_matched(
    config_path: str | Path = DEFAULT_FRONTIER_RANK_MATCHED_CONFIG,
) -> FrontierRankMatchedBootstrap:
    """Authenticate the completed parent and prepare its additive control."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_frontier_rank_matched_config(resolved)
    parent = bootstrap_frontier_adaptation(config.parent_config)
    parent_run = config.parent_artifact_root / "runs" / config.parent_run_hash
    parent_protocol_path = parent_run / "protocol/protocol.json"
    parent_result_path = parent_run / "evaluations/result.json"
    parent_replay_path = parent_run / "protocol/replay_populations.json"
    parent_result = _validated_parent_result(parent_result_path)
    integrator = parent.source.source.integrator
    fit_rows = _clean_rows(parent, "fit")
    validation_rows = _clean_rows(parent, "validation")
    fit_ids = tuple(row.image_id for row in fit_rows)
    validation_ids = tuple(row.image_id for row in validation_rows)
    if (
        parent.protocol.content_hash != config.parent_run_hash
        or parent.run != parent_run
        or file_sha256(parent_protocol_path) != config.parent_protocol_sha256
        or file_sha256(parent_result_path) != config.parent_result_sha256
        or file_sha256(parent_replay_path) != config.parent_replay_sha256
        or parent.config.stage != config.stage
        or parent.config.seed != config.seed
        or integrator.primary_config.lora_rank != config.source_rank
        or integrator.primary_config.lora_alpha != config.source_rank
        or integrator.primary_config.joint_training != config.training
        or len(fit_rows) != 12_194
        or len(validation_rows) != 3_049
        or set(fit_ids) & set(validation_ids)
        or any(row.split != "train" for row in (*fit_rows, *validation_rows))
    ):
        raise ValueError("rank-matched control differs from its authenticated parent")
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = FrontierRankMatchedProtocol(
        config.parent_run_hash,
        config.parent_result_sha256,
        str(parent_result["content_hash"]),
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
        parent_run / "controls/joint_iid_lora_r80" / protocol.content_hash
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
    return FrontierRankMatchedBootstrap(
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
    bootstrap: FrontierRankMatchedBootstrap,
    number: int,
    total: int,
    message: str,
) -> None:
    """Print and persist one human-readable control phase."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.control_root / "workflow_events.jsonl",
        "imagenetr50-frontier-rank-matched-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": "imagenetr50-frontier-rank-matched-event-v1",
            "wall_time_unix": time.time(),
        }
    )


def _new_model(
    bootstrap: FrontierRankMatchedBootstrap, rank: int, alpha: int
) -> AdapterVisionModel:
    integrator = bootstrap.parent.source.source.integrator
    return AdapterVisionModel(
        create_pinned_backbone(integrator.checkpoint),
        tuple(range(4 * bootstrap.config.stage)),
        rank,
        alpha,
        bootstrap.config.dropout,
        bootstrap.config.seed,
    )


def _preflight(
    bootstrap: FrontierRankMatchedBootstrap,
    model: AdapterVisionModel,
    device: torch.device,
) -> dict[str, object]:
    """Check exact aggregate rank, zero-effect parity, gradients, and isolation."""
    target = bootstrap.control_root / "preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    reference = _new_model(
        bootstrap, bootstrap.config.source_rank, bootstrap.config.source_rank
    ).to(device)
    integrator = bootstrap.parent.source.source.integrator
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            bootstrap.fit_rows[: bootstrap.config.training.batch_size],
            integrator.train_transform,
            bootstrap.config.seed,
            0,
        ),
        batch_size=bootstrap.config.training.batch_size,
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
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        target_logits = model(images)
        reference_logits = reference(images)
    parity_error = float(torch.max(torch.abs(target_logits - reference_logits)))
    del reference
    torch.cuda.empty_cache()
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = F.cross_entropy(model(images), labels)
    loss.backward()
    trainables = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = tuple(parameter.grad for parameter in trainables)
    actual_lora_parameters = sum(
        parameter.numel() for parameter in trainable_lora_parameters(model)
    )
    expected_lora_parameters = vit_lora_parameter_count(bootstrap.config.target_rank)
    parent_preflight = dict(bootstrap.parent_result["preflight"])
    parent_lora_parameters = vit_lora_parameter_count(
        bootstrap.config.source_rank
    ) * bootstrap.config.frontier_adapters
    core: dict[str, object] = {
        "attached_gradient_tensors": sum(value is not None for value in gradients),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "classifier_parameters": model.classifier.weight.numel()
        + model.classifier.bias.numel(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "finite_gradient_tensors": sum(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        ),
        "fit_examples": len(bootstrap.fit_rows),
        "frontier_adapter_count": bootstrap.config.frontier_adapters,
        "frontier_aggregate_lora_parameters": parent_lora_parameters,
        "frontier_integrator_parameters": int(parent_preflight["trainable_parameters"])
        - parent_lora_parameters,
        "lora_alpha": bootstrap.config.target_alpha,
        "lora_parameters": actual_lora_parameters,
        "lora_rank": bootstrap.config.target_rank,
        "lora_scale": bootstrap.config.target_alpha / bootstrap.config.target_rank,
        "loss": float(loss.detach()),
        "schema_version": "imagenetr50-frontier-rank-matched-preflight-v1",
        "test_evaluations": 0,
        "trainable_parameters": sum(parameter.numel() for parameter in trainables),
        "validation_examples": len(bootstrap.validation_rows),
        "zero_lora_max_logit_error_vs_rank16": parity_error,
    }
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["attached_gradient_tensors"] != 50
        or core["finite_gradient_tensors"] != 50
        or actual_lora_parameters != expected_lora_parameters
        or actual_lora_parameters != parent_lora_parameters
        or core["classifier_parameters"] != CLASSIFIER_PARAMETERS
        or core["frontier_integrator_parameters"] != MACRO_PARAMETERS
        or core["trainable_parameters"]
        != expected_lora_parameters + CLASSIFIER_PARAMETERS
        or parity_error != 0.0
    ):
        raise RuntimeError("rank-matched joint-IID preflight failed")
    model.zero_grad(set_to_none=True)
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _training_seal(bootstrap: FrontierRankMatchedBootstrap) -> dict[str, object]:
    """Prove exact fit/validation use, no test access, and unchanged parent bytes."""
    integrator = bootstrap.parent.source.source.integrator
    fit_ids = frozenset(row.image_id for row in bootstrap.fit_rows)
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    parent_run = bootstrap.parent.run
    parent_hashes_after = {
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
        "parent_files_unchanged": parent_hashes_after == expected,
        "parent_hashes_after": parent_hashes_after,
        "schema_version": "imagenetr50-frontier-rank-matched-training-seal-v1",
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
    ):
        raise RuntimeError("rank-matched training seal failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(bootstrap.control_root / "training_seal.json", record)
    return record


def _validated_control_result(
    path: Path, bootstrap: FrontierRankMatchedBootstrap
) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    history = bootstrap.parent.run / str(record.get("history", ""))
    artifact = bootstrap.parent.run / str(record.get("artifact", ""))
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-rank-matched-control-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("control_protocol_hash") != bootstrap.protocol.content_hash
        or record.get("parent_result_hash")
        != bootstrap.parent_result["content_hash"]
        or record.get("test_evaluations") != 0
        or not history.is_file()
        or record.get("history_sha256") != file_sha256(history)
        or len(ChainedJsonlLedger(history, JOINT_HISTORY_FORMAT).rows) != 5
        or validate_artifact_directory(artifact) != record.get("artifact_sha256")
    ):
        raise ValueError("rank-matched joint-IID result does not authenticate")
    return record


def _fit_or_load(
    bootstrap: FrontierRankMatchedBootstrap, device: torch.device
) -> tuple[dict[str, object], bool]:
    """Train the rank-80 control once or authenticate it without a model."""
    target = bootstrap.parent.run / CONTROL_RESULT
    if target.is_file():
        return _validated_control_result(target, bootstrap), True
    model = _new_model(
        bootstrap, bootstrap.config.target_rank, bootstrap.config.target_alpha
    ).to(device)
    require_trainable_boundary(model)
    preflight = _preflight(bootstrap, model, device)
    integrator = bootstrap.parent.source.source.integrator
    history_path = bootstrap.control_root / "history.jsonl"
    checkpoint_path = bootstrap.control_root / "checkpoint.pt"
    fit, fit_metrics, validation_metrics = fit_clean_joint_control(
        model=model,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=bootstrap.fit_rows,
        validation_rows=bootstrap.validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=bootstrap.config.training,
        training_seed=bootstrap.config.seed + 50_000,
        num_workers=bootstrap.config.num_workers,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=bootstrap.protocol.content_hash,
        device=device,
    )
    model_target = bootstrap.control_root / "model"
    work = Path(
        tempfile.mkdtemp(prefix="joint-r80-", dir=bootstrap.control_root / "work")
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
                "schema_version": "imagenetr50-frontier-rank-matched-fit-v1",
            },
        )
        artifact_sha256 = publish_artifact_directory(work, model_target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    seal = _training_seal(bootstrap)
    core: dict[str, object] = {
        "architecture": {
            "classifier_parameters": CLASSIFIER_PARAMETERS,
            "frontier_adapter_count": bootstrap.config.frontier_adapters,
            "frontier_aggregate_lora_parameters": vit_lora_parameter_count(
                bootstrap.config.source_rank
            )
            * bootstrap.config.frontier_adapters,
            "frontier_integrator_parameters_excluded_from_match": MACRO_PARAMETERS,
            "lora_alpha": bootstrap.config.target_alpha,
            "lora_parameters": vit_lora_parameter_count(
                bootstrap.config.target_rank
            ),
            "lora_rank": bootstrap.config.target_rank,
            "lora_scale": bootstrap.config.target_alpha
            / bootstrap.config.target_rank,
            "source_rank": bootstrap.config.source_rank,
            "trainable_parameters": vit_lora_parameter_count(
                bootstrap.config.target_rank
            )
            + CLASSIFIER_PARAMETERS,
        },
        "artifact": str(model_target.relative_to(bootstrap.parent.run)),
        "artifact_sha256": artifact_sha256,
        "control_protocol": str(
            (bootstrap.control_root / "protocol.json").relative_to(
                bootstrap.parent.run
            )
        ),
        "control_protocol_hash": bootstrap.protocol.content_hash,
        "fit": fit.as_record(),
        "fit_metrics": fit_metrics.as_record(),
        "history": str(history_path.relative_to(bootstrap.parent.run)),
        "history_sha256": file_sha256(history_path),
        "parent_result_hash": bootstrap.parent_result["content_hash"],
        "preflight": dict(preflight),
        "schema_version": "imagenetr50-frontier-rank-matched-control-v1",
        "test_evaluations": 0,
        "training": asdict(bootstrap.config.training),
        "training_seal": dict(seal),
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False


def run_frontier_rank_matched_control(
    config_path: str | Path = DEFAULT_FRONTIER_RANK_MATCHED_CONFIG,
) -> Path:
    """Run or exactly resume rank-80 joint IID and update the parent report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the rank-matched control requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_frontier_rank_matched(config_path)
    print(f"Temporary/resumable artifact directory: {bootstrap.control_root}", flush=True)
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=4, desc="ImageNet-R rank-matched control", unit="phase")
    _phase(bootstrap, 1, 4, "Authenticate the parent result and exact clean split")
    overall.update(1)
    _phase(bootstrap, 2, 4, "Fit or authenticate one rank-80 joint-IID adapter")
    result, reused = _fit_or_load(bootstrap, device)
    overall.update(1)
    _phase(bootstrap, 3, 4, "Prove completed-control reuse without optimizer work")
    repeated_result, repeated = _fit_or_load(bootstrap, device)
    if repeated_result != result or not repeated:
        raise RuntimeError("rank-matched control did not reuse exactly")
    reuse_core: dict[str, object] = {
        "all_controls_reused": repeated,
        "new_optimizer_steps": 0,
        "schema_version": "imagenetr50-frontier-rank-matched-reuse-v1",
    }
    publish_immutable_json(
        bootstrap.control_root / "reuse_proof.json",
        {**reuse_core, "content_hash": record_sha256(reuse_core)},
    )
    overall.update(1)
    _phase(bootstrap, 4, 4, "Update the existing stage-31 report")
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
                "schema_version": "imagenetr50-frontier-rank-matched-invocation-v1",
            }
        ),
    )
    overall.update(1)
    overall.close()
    action = "authenticated" if reused else "trained"
    print(
        f"Rank-matched control {action}; existing report updated in "
        f"{elapsed / 60:.1f} minutes: {report}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    print(run_frontier_rank_matched_control())


__all__ = [
    "CLASSIFIER_PARAMETERS",
    "CONTROL_RESULT",
    "FrontierRankMatchedBootstrap",
    "FrontierRankMatchedProtocol",
    "LORA_PARAMETERS_PER_RANK",
    "MACRO_PARAMETERS",
    "bootstrap_frontier_rank_matched",
    "run_frontier_rank_matched_control",
    "vit_lora_parameter_count",
]
