"""Post-hoc stage-matched joint-IID control for the completed integrator run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
import math
import shutil
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    NodeBundle,
    load_node_bundle,
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    ManifestDataset,
    image_transforms,
    load_dataset_manifest,
)
from apm.continual.vision.imagenetr.heads import save_classifier
from apm.continual.vision.imagenetr.integrator_artifacts import latest_integrator_run
from apm.continual.vision.imagenetr.integrator_config import ImageNetRIntegratorConfig
from apm.continual.vision.imagenetr.integrator_workflow import DEFAULT_INTEGRATOR_CONFIG
from apm.continual.vision.imagenetr.lora import load_adapter_factors, save_adapter
from apm.continual.vision.imagenetr.metrics import accuracy
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.routing import GroundTruth
from apm.continual.vision.imagenetr.training import (
    TrainingResult,
    os_cpu_workers,
    train_adapter_model,
)


CONTROL_SCHEMA = "imagenetr50-stage-matched-joint-iid-v1"
LEDGER_SCHEMA = "imagenetr50-stage-matched-joint-iid-stage-v1"


@dataclass(frozen=True, slots=True)
class StageMatchedContext:
    """Authenticated inputs and local paths for the stage-matched control."""

    project_root: Path
    integrator_config: ImageNetRIntegratorConfig
    integrator_run: Path
    primary_config: ImageNetRConfig
    primary_run: Path
    manifest: DatasetManifest
    checkpoint: Path
    source_joint: NodeBundle
    protocol: dict[str, object]
    control_root: Path
    train_transform: object
    test_transform: object


@dataclass(frozen=True, slots=True)
class StageEvaluation:
    """Predictions and exact aggregate measurements for one prefix model."""

    accuracy: float
    labels: Tensor
    predictions: Tensor
    task_correct: tuple[int, ...]
    task_examples: tuple[int, ...]
    wall_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _find_checkpoint(data_root: Path) -> Path:
    candidates = tuple(
        path for path in (data_root / "model_cache").rglob(TIMM_MODEL_SHA256) if path.is_file()
    )
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise FileNotFoundError("the pinned ViT checkpoint is not locally authenticated")
    return candidates[0]


def _find_source_joint(primary_run: Path) -> NodeBundle:
    candidates = tuple(
        path.parent
        for path in sorted(
            (primary_run / "baselines" / "joint_iid_lora_r16").glob("*/artifact.json")
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError("the sealed run must contain exactly one joint-IID artifact")
    return load_node_bundle(candidates[0])


def _protocol_record(
    project_root: Path,
    integrator_run: Path,
    integrator_config: ImageNetRIntegratorConfig,
    primary_config: ImageNetRConfig,
    manifest: DatasetManifest,
    source_joint: NodeBundle,
) -> dict[str, object]:
    material = material_tree_manifest(
        (
            project_root / "configs/vision/imagenetr/primary.yaml",
            project_root / "src/apm/continual/artifacts.py",
            project_root / "src/apm/continual/vision/imagenetr/stage_matched_joint.py",
            project_root / "src/apm/continual/vision/imagenetr/config.py",
            project_root / "src/apm/continual/vision/imagenetr/data.py",
            project_root / "src/apm/continual/vision/imagenetr/heads.py",
            project_root / "src/apm/continual/vision/imagenetr/lora.py",
            project_root / "src/apm/continual/vision/imagenetr/model.py",
            project_root / "src/apm/continual/vision/imagenetr/training.py",
        )
    )
    locked_path = integrator_run / "evaluations" / "locked_test.json"
    locked = load_canonical_json(locked_path)
    core: dict[str, object] = {
        "architecture": {
            "backbone": primary_config.model_name,
            "classifier": "prefix-wide affine head with bias",
            "lora_alpha": primary_config.lora_alpha,
            "lora_dropout": primary_config.lora_dropout,
            "lora_rank": primary_config.lora_rank,
            "lora_targets": list(primary_config.lora_targets),
        },
        "augmentation_seed": primary_config.seed + 50_000,
        "class_prefix": "global remapped classes [0, 4 * stage)",
        "code_manifest_hash": material["content_hash"],
        "dataset_hash": manifest.content_hash,
        "evaluation": "raw affine top-1 on test tasks [0, stage), after fixed training",
        "integrator_locked_test_sha256": file_sha256(locked_path),
        "integrator_run_hash": integrator_run.name,
        "model_initialization_seed": primary_config.seed,
        "model_sha256": TIMM_MODEL_SHA256,
        "primary_run_hash": integrator_config.sealed_run_hash,
        "role": "post-hoc diagnostic; no selection or execution gate",
        "schema_version": CONTROL_SCHEMA,
        "source_joint_node_hash": source_joint.artifact.content_hash,
        "source_joint_task50_accuracy": float(
            dict(locked["local_references"])["joint_iid_last"]
        ),
        "stages": list(range(1, primary_config.tasks + 1)),
        "task50_policy": "reuse and re-evaluate the authenticated offline joint-IID model",
        "test_identity_use": "evaluation only after each prefix model is fully trained",
        "training": asdict(primary_config.joint_training),
        "training_rows": "only immutable train rows from tasks [0, stage)",
        "material_code_manifest": material,
    }
    return {**core, "content_hash": record_sha256(core)}


def bootstrap_stage_matched_control(
    config_path: str | Path = DEFAULT_INTEGRATOR_CONFIG,
) -> StageMatchedContext:
    """Authenticate the completed run and create one content-addressed control root."""
    source = Path(config_path).resolve()
    project_root = source.parents[3]
    integrator_config, integrator_run = latest_integrator_run(source)
    primary_config = load_config(project_root / "configs/vision/imagenetr/primary.yaml")
    primary_run = (
        integrator_config.inference_artifact_root
        / "runs"
        / integrator_config.sealed_run_hash
    )
    primary_protocol = load_canonical_json(primary_run / "protocol" / "protocol_manifest.json")
    if primary_protocol.get("content_hash") != integrator_config.sealed_run_hash:
        raise ValueError("the primary run protocol no longer authenticates its directory")
    if primary_protocol.get("config_hash") != primary_config.config_hash:
        raise ValueError("the current joint-IID configuration differs from the sealed run")
    manifest = load_dataset_manifest(
        integrator_config.data_root / "imagenet-r" / "dataset_manifest.json"
    )
    if manifest.content_hash != primary_protocol.get("dataset_manifest_hash"):
        raise ValueError("the local dataset differs from the sealed joint-IID run")
    source_joint = _find_source_joint(primary_run)
    protocol = _protocol_record(
        project_root,
        integrator_run,
        integrator_config,
        primary_config,
        manifest,
        source_joint,
    )
    control_root = (
        integrator_run
        / "controls"
        / "stage_matched_joint_iid"
        / str(protocol["content_hash"])
    )
    for relative in ("checkpoints", "models", "work"):
        (control_root / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(control_root / "protocol.json", protocol)
    train_transform, test_transform = image_transforms(primary_config.input_size)
    return StageMatchedContext(
        project_root,
        integrator_config,
        integrator_run,
        primary_config,
        primary_run,
        manifest,
        _find_checkpoint(integrator_config.data_root),
        source_joint,
        protocol,
        control_root,
        train_transform,
        test_transform,
    )


def evaluate_prefix_model(
    model: AdapterVisionModel,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    stage: int,
) -> StageEvaluation:
    """Evaluate a sealed prefix model and retain per-example predictions locally."""
    if not rows or any(row.split != "test" or row.task_index >= stage for row in rows):
        raise ValueError("prefix evaluation requires only test rows from represented tasks")
    model.to(device).eval()
    loader = DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os_cpu_workers()),
        pin_memory=device.type == "cuda",
    )
    class_ids = torch.tensor(model.classifier.class_ids, dtype=torch.long)
    prediction_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    started = time.monotonic()
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                local_predictions = model(images).argmax(dim=1).cpu()
            prediction_batches.append(class_ids[local_predictions])
            label_batches.append(labels.cpu())
    predictions = torch.cat(tuple(prediction_batches))
    labels = torch.cat(tuple(label_batches))
    truth = GroundTruth(tuple(row.image_id for row in rows), labels)
    task_examples = tuple(int(torch.sum(labels // 4 == task).item()) for task in range(stage))
    task_correct = tuple(
        int(torch.sum((predictions == labels) & (labels // 4 == task)).item())
        for task in range(stage)
    )
    return StageEvaluation(
        accuracy(predictions, truth),
        labels,
        predictions,
        task_correct,
        task_examples,
        time.monotonic() - started,
    )


def _stage_result(
    context: StageMatchedContext,
    stage: int,
    training: TrainingResult | None,
    evaluation: StageEvaluation,
    reused_source_model: bool,
) -> dict[str, object]:
    train_rows = context.manifest.select("train", range(stage))
    test_rows = context.manifest.select("test", range(stage))
    expected_presentations = len(train_rows) * context.primary_config.joint_training.epochs
    if training is not None and training.image_presentations != expected_presentations:
        raise ValueError("stage training did not consume the fixed presentation budget")
    core: dict[str, object] = {
        "accuracy": evaluation.accuracy,
        "class_count": 4 * stage,
        "evaluation_seconds": evaluation.wall_seconds,
        "image_presentations": 0 if training is None else training.image_presentations,
        "optimizer_steps": 0 if training is None else training.optimizer_steps,
        "peak_vram_bytes": 0 if training is None else training.peak_vram_bytes,
        "reused_source_model": reused_source_model,
        "schema_version": LEDGER_SCHEMA,
        "stage": stage,
        "task_correct": list(evaluation.task_correct),
        "task_examples": list(evaluation.task_examples),
        "test_examples": len(test_rows),
        "test_image_ids_hash": record_sha256([row.image_id for row in test_rows]),
        "train_examples": len(train_rows),
        "train_image_ids_hash": record_sha256([row.image_id for row in train_rows]),
        "training_final_loss": None if training is None else training.final_loss,
        "training_seconds": 0.0 if training is None else training.wall_seconds,
    }
    return {**core, "stage_result_hash": record_sha256(core)}


def _publish_stage(
    context: StageMatchedContext,
    stage: int,
    training: TrainingResult | None,
    evaluation: StageEvaluation,
    reused_source_model: bool,
) -> dict[str, object]:
    work = context.control_root / "work" / f"stage_{stage:03d}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if training is None:
        adapter, classifier = context.source_joint.adapter, context.source_joint.classifier
    else:
        adapter, classifier = training.adapter, training.classifier
    adapter_sha256 = save_adapter(work / "adapter.safetensors", adapter)
    classifier_sha256 = save_classifier(work / "classifier.safetensors", classifier)
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    save_file(
        {
            "labels": evaluation.labels.contiguous(),
            "predictions": evaluation.predictions.contiguous(),
        },
        work / "predictions.safetensors",
        metadata={"schema_version": "imagenetr50-stage-predictions-v1"},
    )
    result = {
        **_stage_result(
            context, stage, training, evaluation, reused_source_model
        ),
        "adapter_sha256": adapter_sha256,
        "classifier_sha256": classifier_sha256,
        "predictions_sha256": file_sha256(work / "predictions.safetensors"),
        "source_joint_node_hash": (
            context.source_joint.artifact.content_hash if reused_source_model else None
        ),
    }
    publish_immutable_json(work / "stage_result.json", result)
    target = context.control_root / "models" / f"stage_{stage:03d}"
    artifact_sha256 = publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return {**result, "stage_artifact_sha256": artifact_sha256}


def _load_stage(context: StageMatchedContext, stage: int) -> dict[str, object] | None:
    target = context.control_root / "models" / f"stage_{stage:03d}"
    if not target.is_dir():
        return None
    artifact_sha256 = validate_artifact_directory(target)
    result = load_canonical_json(target / "stage_result.json")
    if result.get("stage") != stage or result.get("schema_version") != LEDGER_SCHEMA:
        raise ValueError("stored stage result has the wrong identity")
    return {**result, "stage_artifact_sha256": artifact_sha256}


def validate_stage_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Require exactly one finite, internally consistent result for every stage."""
    ordered = tuple(sorted((dict(row) for row in rows), key=lambda row: int(row["stage"])))
    if [int(row["stage"]) for row in ordered] != list(range(1, 51)):
        raise ValueError("stage-matched joint control is incomplete")
    if any(
        not math.isfinite(float(row["accuracy"]))
        or int(row["class_count"]) != 4 * int(row["stage"])
        or int(row["test_examples"]) != sum(int(value) for value in row["task_examples"])
        for row in ordered
    ):
        raise ValueError("stage-matched joint control contains an invalid result")
    return ordered


def _train_or_reuse_stage(
    context: StageMatchedContext,
    stage: int,
    device: torch.device,
) -> dict[str, object]:
    existing = _load_stage(context, stage)
    if existing is not None:
        return existing
    class_ids = tuple(range(4 * stage))
    model = AdapterVisionModel(
        create_pinned_backbone(context.checkpoint),
        class_ids,
        context.primary_config.lora_rank,
        context.primary_config.lora_alpha,
        context.primary_config.lora_dropout,
        context.primary_config.seed,
    )
    training: TrainingResult | None = None
    reused_source_model = stage == context.primary_config.tasks
    if reused_source_model:
        load_adapter_factors(model, context.source_joint.adapter)
        model.classifier.load_rows(context.source_joint.classifier)
    else:
        training = train_adapter_model(
            model,
            context.integrator_config.data_root / "imagenet-r",
            context.manifest.select("train", range(stage)),
            context.train_transform,
            context.primary_config.joint_training,
            context.primary_config.seed + 50_000,
            device,
            context.control_root / "checkpoints" / f"stage_{stage:03d}.pt",
            num_workers=context.primary_config.num_workers,
            checkpoint_steps=context.primary_config.checkpoint_steps,
            show_progress=True,
        )
    evaluation = evaluate_prefix_model(
        model,
        context.integrator_config.data_root / "imagenet-r",
        context.manifest.select("test", range(stage)),
        context.test_transform,
        context.primary_config.joint_training.batch_size,
        context.primary_config.num_workers,
        device,
        stage,
    )
    result = _publish_stage(context, stage, training, evaluation, reused_source_model)
    del model
    torch.cuda.empty_cache()
    return result


def run_stage_matched_joint_control(
    config_path: str | Path = DEFAULT_INTEGRATOR_CONFIG,
) -> Path:
    """Train or resume all prefix controls and publish their compact report projection."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the stage-matched ImageNet-R control requires BF16 CUDA")
    print("Phase 1/3: authenticating the completed integrator and joint-IID run.", flush=True)
    context = bootstrap_stage_matched_control(config_path)
    print(f"Resumable artifact directory: {context.control_root}", flush=True)
    ledger = ChainedJsonlLedger(context.control_root / "stage_metrics.jsonl", LEDGER_SCHEMA)
    ledger.require_unique_keys(("stage",))
    ledger_by_stage = {int(row["stage"]): row for row in ledger.rows}
    for stage, row in ledger_by_stage.items():
        stored = _load_stage(context, stage)
        if stored is None or stored["stage_artifact_sha256"] != row["stage_artifact_sha256"]:
            raise ValueError("stage ledger refers to missing or changed model evidence")
    train_presentations = {
        stage: len(context.manifest.select("train", range(stage)))
        * context.primary_config.joint_training.epochs
        for stage in range(1, context.primary_config.tasks)
    }
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    completed_presentations = sum(
        train_presentations[stage]
        for stage in ledger_by_stage
        if stage in train_presentations
    )
    overall = tqdm(
        total=sum(train_presentations.values()),
        initial=completed_presentations,
        desc="stage-matched joint training",
        unit="image",
    )
    print("Phase 2/3: fitting fresh joint prefix models and evaluating fixed test prefixes.", flush=True)
    device = torch.device("cuda:0")
    for stage in range(1, context.primary_config.tasks + 1):
        if stage in ledger_by_stage:
            continue
        print(
            f"Stage {stage:02d}/50: "
            + ("re-evaluating sealed task-50 joint model" if stage == 50 else "fresh prefix fit"),
            flush=True,
        )
        result = _train_or_reuse_stage(context, stage, device)
        ledger_row = ledger.append(result)
        ledger_by_stage[stage] = ledger_row
        if stage in train_presentations:
            overall.update(train_presentations[stage])
        overall.set_postfix_str(f"stage {stage}/50, accuracy {float(result['accuracy']):.2f}%")
    overall.close()
    print("Phase 3/3: validating the complete curve and sealing its report projection.", flush=True)
    rows = validate_stage_rows(tuple(ledger_by_stage.values()))
    source_accuracy = float(context.protocol["source_joint_task50_accuracy"])
    if not math.isclose(float(rows[-1]["accuracy"]), source_accuracy, abs_tol=1e-6):
        raise ValueError("task-50 stage control does not reproduce the sealed joint-IID endpoint")
    public_rows = tuple(
        {
            key: value
            for key, value in row.items()
            if key not in {"format", "previous_sha256", "result_sha256", "sequence"}
        }
        for row in rows
    )
    existing_summary = (
        load_canonical_json(context.control_root / "summary.json")
        if (context.control_root / "summary.json").is_file()
        else None
    )
    summary_core: dict[str, object] = {
        "completed_at_utc": (
            _utc_now()
            if existing_summary is None
            else str(existing_summary["completed_at_utc"])
        ),
        "incremental_accuracy": math.fsum(
            float(row["accuracy"]) for row in public_rows
        )
        / len(public_rows),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in public_rows),
        "protocol_hash": context.protocol["content_hash"],
        "rows": list(public_rows),
        "schema_version": "imagenetr50-stage-matched-joint-iid-summary-v1",
        "source_joint_task50_accuracy": source_accuracy,
        "training_image_presentations": sum(
            int(row["image_presentations"]) for row in public_rows
        ),
    }
    summary = {**summary_core, "content_hash": record_sha256(summary_core)}
    summary_path = context.control_root / "summary.json"
    publish_immutable_json(summary_path, summary)
    projection = {
        **summary,
        "control_relative_path": str(context.control_root.relative_to(context.integrator_run)),
    }
    atomic_write(
        context.integrator_run / "evaluations" / "stage_matched_joint_iid.json",
        canonical_json_bytes(projection),
    )
    return summary_path


__all__ = [
    "StageEvaluation",
    "StageMatchedContext",
    "bootstrap_stage_matched_control",
    "evaluate_prefix_model",
    "run_stage_matched_joint_control",
    "validate_stage_rows",
]
