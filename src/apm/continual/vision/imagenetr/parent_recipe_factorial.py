"""Development-only factorial isolating ImageNet-R parent-training recipe effects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
import math
import os
import shutil
import time

import torch
from torch import Tensor
from torch.nn import functional as F
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
from apm.continual.vision.imagenetr.config import TrainingConfig
from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    ManifestDataset,
    image_transforms,
    load_dataset_manifest,
)
from apm.continual.vision.imagenetr.heads import (
    ClassifierRows,
    save_classifier,
    union_classifier_rows,
)
from apm.continual.vision.imagenetr.integrator_config import (
    ImageNetRIntegratorConfig,
    load_integrator_config,
)
from apm.continual.vision.imagenetr.integrator_bank import simulate_binary_topology
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter_factors,
    save_adapter,
)
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import router_split_from_record
from apm.continual.vision.imagenetr.router_protocol import RouterSplit
from apm.continual.vision.imagenetr.training import TrainingResult, train_adapter_model


DEFAULT_CONFIG = Path("configs/vision/imagenetr/parent_recipe_factorial_v1.yaml")
PROTOCOL_SCHEMA = "imagenetr50-parent-recipe-factorial-protocol-v1"
EPOCH_SCHEMA = "imagenetr50-parent-recipe-factorial-epoch-v1"
JOB_SCHEMA = "imagenetr50-parent-recipe-factorial-job-v1"
SUMMARY_SCHEMA = "imagenetr50-parent-recipe-factorial-summary-v1"


@dataclass(frozen=True, slots=True)
class ParentRecipeConfig:
    """Complete immutable scientific surface for the parent-recipe factorial."""

    name: str
    protocol_revision: str
    replication_seeds: tuple[int, ...]
    artifact_root: Path
    source_integrator_config: Path
    source_integrator_run_hash: str
    source_fit_policy_hash: str
    stages: tuple[int, ...]
    head_initializations: tuple[str, ...]
    weight_decays: tuple[float, ...]
    seed_schedules: tuple[str, ...]
    training: TrainingConfig
    evaluation_partition: str
    evaluation_checkpoints: tuple[int, ...]
    selection_stages: tuple[int, ...]
    substantial_gap_closure_fraction: float
    num_workers: int
    checkpoint_steps: int

    def __post_init__(self) -> None:
        expected = (
            self.name == "imagenetr50_parent_recipe_factorial_v1"
            and self.protocol_revision == "imagenetr50-parent-recipe-factorial-v1"
            and self.replication_seeds == (1993,)
            and len(self.source_integrator_run_hash) == 64
            and len(self.source_fit_policy_hash) == 64
            and self.stages == (8, 16, 32)
            and self.head_initializations == ("fresh", "inherited_union")
            and self.weight_decays == (0.0005, 0.0)
            and self.seed_schedules == ("joint", "parent")
            and self.training.epochs == 5
            and self.training.batch_size == 64
            and self.training.momentum == 0.9
            and self.training.lora_lr == 0.0005
            and self.training.head_lr == 0.01
            and self.evaluation_partition == "router_validation"
            and self.evaluation_checkpoints == (0, 5)
            and self.selection_stages == (16, 32)
            and self.substantial_gap_closure_fraction == 0.5
            and self.num_workers >= 0
            and self.checkpoint_steps >= 1
        )
        if not expected:
            raise ValueError("configuration differs from the frozen parent-recipe factorial")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of every experiment choice."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["artifact_root"] = str(self.artifact_root)
        record["source_integrator_config"] = str(self.source_integrator_config)
        return record


@dataclass(frozen=True, slots=True)
class ParentRecipeCondition:
    """One cell in the complete head-by-decay-by-schedule factorial."""

    head_initialization: str
    weight_decay: float
    seed_schedule: str

    def __post_init__(self) -> None:
        if (
            self.head_initialization not in {"fresh", "inherited_union"}
            or self.weight_decay not in {0.0, 0.0005}
            or self.seed_schedule not in {"joint", "parent"}
        ):
            raise ValueError("invalid parent-recipe factorial cell")

    @property
    def key(self) -> str:
        """Return the stable compact condition name used in every artifact."""
        decay = "wd0" if self.weight_decay == 0.0 else "wd5e4"
        return f"{self.head_initialization}__{decay}__{self.seed_schedule}"

    @property
    def label(self) -> str:
        """Return the single human-facing label shared by tables and plots."""
        head = "Inherited union head" if self.head_initialization == "inherited_union" else "Fresh head"
        decay = "0" if self.weight_decay == 0.0 else "5e-4"
        schedule = "parent" if self.seed_schedule == "parent" else "joint"
        return f"{head} | wd={decay} | {schedule} seed/order"

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible condition record."""
        return {**asdict(self), "condition_key": self.key, "condition_label": self.label}


@dataclass(frozen=True, slots=True)
class StageSource:
    """Authenticated source root and its exact pre-parent classifier union."""

    stage: int
    event_sequence: int
    root: NodeBundle
    children: tuple[NodeBundle, NodeBundle]
    inherited_rows: ClassifierRows
    original_training: dict[str, object]

    def as_record(self) -> dict[str, object]:
        """Return the source identities needed to reproduce this stage."""
        return {
            "child_node_hashes": [child.artifact.content_hash for child in self.children],
            "event_sequence": self.event_sequence,
            "inherited_classifier_hashes": [
                child.artifact.classifier_sha256 for child in self.children
            ],
            "root_classifier_sha256": self.root.artifact.classifier_sha256,
            "root_lora_sha256": self.root.artifact.lora_sha256,
            "root_node_hash": self.root.artifact.content_hash,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class ParentRecipeContext:
    """Authenticated source state and paths for one resumable factorial run."""

    project_root: Path
    config: ParentRecipeConfig
    source: "ParentFactorialSource"
    fit_rows: tuple[ImageRecord, ...]
    validation_rows: tuple[ImageRecord, ...]
    stage_sources: tuple[StageSource, ...]
    protocol: dict[str, object]
    run_root: Path


@dataclass(frozen=True, slots=True)
class ParentFactorialSource:
    """Read-only authenticated view of the completed integrator hierarchy."""

    project_root: Path
    config: ImageNetRIntegratorConfig
    primary_config: ImageNetRConfig
    manifest: DatasetManifest
    split: RouterSplit
    run_root: Path
    protocol: dict[str, object]
    checkpoint: Path
    train_transform: object
    test_transform: object
    environment_manifest_hash: str
    model_manifest_hash: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Aggregate clean-validation measurements for one model checkpoint."""

    accuracy: float
    cross_entropy: float
    examples: int
    task_correct: tuple[int, ...]
    task_examples: tuple[int, ...]
    wall_seconds: float


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the frozen factorial")
    return value


def _resolved_path(value: object, project_root: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_parent_recipe_config(path: str | Path = DEFAULT_CONFIG) -> ParentRecipeConfig:
    """Load the strict single experiment configuration without CLI overrides."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {
            "experiment",
            "paths",
            "sealed_inputs",
            "matrix",
            "training",
            "evaluation",
            "runtime",
        },
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "replication_seeds"},
    )
    paths = _mapping(
        root["paths"], "paths", {"artifact_root", "source_integrator_config"}
    )
    sealed = _mapping(
        root["sealed_inputs"],
        "sealed_inputs",
        {"source_integrator_run_hash", "source_fit_policy_hash"},
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {"stages", "head_initializations", "weight_decays", "seed_schedules"},
    )
    training = _mapping(
        root["training"],
        "training",
        {"epochs", "batch_size", "optimizer", "momentum", "lora_lr", "head_lr"},
    )
    evaluation = _mapping(
        root["evaluation"],
        "evaluation",
        {
            "partition",
            "checkpoints",
            "selection_stages",
            "substantial_gap_closure_fraction",
        },
    )
    runtime = _mapping(
        root["runtime"], "runtime", {"num_workers", "checkpoint_steps"}
    )
    if str(training["optimizer"]).lower() != "sgd":
        raise ValueError("the parent-recipe factorial requires SGD")
    return ParentRecipeConfig(
        name=str(experiment["name"]),
        protocol_revision=str(experiment["protocol_revision"]),
        replication_seeds=tuple(int(value) for value in experiment["replication_seeds"]),
        artifact_root=_resolved_path(paths["artifact_root"], project_root),
        source_integrator_config=_resolved_path(
            paths["source_integrator_config"], project_root
        ),
        source_integrator_run_hash=str(sealed["source_integrator_run_hash"]),
        source_fit_policy_hash=str(sealed["source_fit_policy_hash"]),
        stages=tuple(int(value) for value in matrix["stages"]),
        head_initializations=tuple(str(value) for value in matrix["head_initializations"]),
        weight_decays=tuple(float(value) for value in matrix["weight_decays"]),
        seed_schedules=tuple(str(value) for value in matrix["seed_schedules"]),
        training=TrainingConfig(
            int(training["epochs"]),
            int(training["batch_size"]),
            float(training["momentum"]),
            0.0,
            float(training["lora_lr"]),
            float(training["head_lr"]),
        ),
        evaluation_partition=str(evaluation["partition"]),
        evaluation_checkpoints=tuple(int(value) for value in evaluation["checkpoints"]),
        selection_stages=tuple(int(value) for value in evaluation["selection_stages"]),
        substantial_gap_closure_fraction=float(
            evaluation["substantial_gap_closure_fraction"]
        ),
        num_workers=int(runtime["num_workers"]),
        checkpoint_steps=int(runtime["checkpoint_steps"]),
    )


def condition_matrix(
    config: ParentRecipeConfig,
) -> tuple[ParentRecipeCondition, ...]:
    """Return all eight factorial cells in deterministic display order."""
    return tuple(
        ParentRecipeCondition(head, decay, schedule)
        for head, decay, schedule in product(
            config.head_initializations,
            config.weight_decays,
            config.seed_schedules,
        )
    )


def seed_recipe(
    condition: ParentRecipeCondition,
    replication_seed: int,
    event_sequence: int,
) -> tuple[int, int]:
    """Return model and data-order seeds for one controlled recipe cell."""
    if replication_seed < 0 or event_sequence < 0:
        raise ValueError("seed recipe inputs must be nonnegative")
    if condition.seed_schedule == "joint":
        return replication_seed, replication_seed + 50_000
    parent_seed = replication_seed + 300_000 + event_sequence
    return parent_seed, parent_seed


def _stage_sources(
    source: ParentFactorialSource,
    source_fit_policy_hash: str,
    stages: Sequence[int],
) -> tuple[StageSource, ...]:
    hierarchy_root = source.run_root / "hierarchies" / source_fit_policy_hash
    node_paths = tuple(sorted((hierarchy_root / "nodes").glob("*/node.json")))
    node_path_by_hash = {
        str(load_canonical_json(path)["content_hash"]): path.parent for path in node_paths
    }
    events, _snapshots = simulate_binary_topology(max(stages))
    event_by_stage = {
        event.stage: event
        for event in events
        if event.parent.first_task == 0 and event.parent.last_task + 1 == event.stage
    }
    result: list[StageSource] = []
    for stage in stages:
        snapshot = load_canonical_json(
            hierarchy_root / "snapshots" / f"stage_{stage:03d}.json"
        )
        if (
            snapshot.get("policy_hash") != source_fit_policy_hash
            or snapshot.get("stage") != stage
            or len(snapshot.get("logical_node_ids", ())) != 1
        ):
            raise ValueError("factorial stages require one authenticated fit root")
        logical_id = str(snapshot["logical_node_ids"][0])
        root = load_node_bundle(hierarchy_root / "nodes" / logical_id)
        if len(root.artifact.parent_hashes) != 2 or stage not in event_by_stage:
            raise ValueError("factorial source root lacks its final binary carry")
        children = tuple(
            load_node_bundle(node_path_by_hash[identity])
            for identity in root.artifact.parent_hashes
        )
        if len(children) != 2:
            raise AssertionError("the final carry must have exactly two children")
        inherited = union_classifier_rows(tuple(child.classifier for child in children))
        if inherited.class_ids != tuple(range(4 * stage)):
            raise ValueError("inherited parent rows do not cover the exact class prefix")
        result.append(
            StageSource(
                stage,
                event_by_stage[stage].sequence,
                root,
                (children[0], children[1]),
                inherited,
                load_canonical_json(root.directory / "training_metrics.json"),
            )
        )
    return tuple(result)


def _protocol_record(
    project_root: Path,
    config_path: Path,
    config: ParentRecipeConfig,
    source: ParentFactorialSource,
    fit_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    stages: Sequence[StageSource],
) -> dict[str, object]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    material = material_tree_manifest(
        (
            config_path,
            project_root / "scripts/vision/imagenetr/run_parent_recipe_factorial_local.sh",
            project_root / "src/apm/continual/artifacts.py",
            *(package / filename for filename in (
                "artifacts.py",
                "config.py",
                "data.py",
                "heads.py",
                "lora.py",
                "model.py",
                "parent_recipe_factorial.py",
                "parent_recipe_reporting.py",
                "training.py",
            )),
        )
    )
    core: dict[str, object] = {
        "code_manifest": material,
        "config": config.as_record(),
        "config_hash": config.config_hash,
        "dataset_manifest_hash": source.manifest.content_hash,
        "environment_manifest_hash": source.environment_manifest_hash,
        "fit_image_ids_hash": record_sha256([row.image_id for row in fit_rows]),
        "fit_images": len(fit_rows),
        "model_manifest_hash": source.model_manifest_hash,
        "no_test_use": True,
        "schema_version": PROTOCOL_SCHEMA,
        "source_fit_policy_hash": config.source_fit_policy_hash,
        "source_integrator_run_hash": config.source_integrator_run_hash,
        "stage_sources": [stage.as_record() for stage in stages],
        "validation_image_ids_hash": record_sha256(
            [row.image_id for row in validation_rows]
        ),
        "validation_images": len(validation_rows),
        "validation_split_hash": source.split.content_hash,
    }
    return {**core, "content_hash": record_sha256(core)}


def load_parent_factorial_source(
    config: ParentRecipeConfig, project_root: Path
) -> ParentFactorialSource:
    """Load the explicitly pinned completed integrator without moving its latest pointer."""
    integrator_config = load_integrator_config(config.source_integrator_config)
    source_run = (
        integrator_config.artifact_root / "runs" / config.source_integrator_run_hash
    )
    source_protocol = load_canonical_json(source_run / "protocol" / "protocol.json")
    if source_protocol.get("content_hash") != config.source_integrator_run_hash:
        raise ValueError("configured source integrator run does not authenticate")
    if source_protocol.get("config_hash") != integrator_config.config_hash:
        raise ValueError("source integrator configuration changed")
    primary_config = load_config(project_root / "configs/vision/imagenetr/primary.yaml")
    manifest = load_dataset_manifest(
        integrator_config.data_root / "imagenet-r" / "dataset_manifest.json"
    )
    if manifest.content_hash != source_protocol.get("dataset_manifest_hash"):
        raise ValueError("source hierarchy dataset changed")
    split = router_split_from_record(
        load_canonical_json(source_run / "protocol" / "router_split.json")
    )
    if split.content_hash != source_protocol.get("split_hash"):
        raise ValueError("source hierarchy development split changed")
    model_manifest = load_canonical_json(source_run / "protocol" / "model_manifest.json")
    checkpoint_candidates = tuple(
        path
        for path in (integrator_config.data_root / "model_cache").rglob(
            TIMM_MODEL_SHA256
        )
        if path.is_file()
    )
    if (
        len(checkpoint_candidates) != 1
        or file_sha256(checkpoint_candidates[0]) != TIMM_MODEL_SHA256
        or model_manifest.get("sha256") != TIMM_MODEL_SHA256
    ):
        raise FileNotFoundError("the pinned source backbone is unavailable or changed")
    train_transform, test_transform = image_transforms(primary_config.input_size)
    return ParentFactorialSource(
        project_root,
        integrator_config,
        primary_config,
        manifest,
        split,
        source_run,
        source_protocol,
        checkpoint_candidates[0],
        train_transform,
        test_transform,
        str(source_protocol["environment_manifest_hash"]),
        str(source_protocol["model_manifest_hash"]),
    )


def bootstrap_parent_recipe_factorial(
    config_path: str | Path = DEFAULT_CONFIG,
) -> ParentRecipeContext:
    """Authenticate the source hierarchy and prepare one isolated run namespace."""
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parents[3]
    config = load_parent_recipe_config(resolved_config_path)
    source = load_parent_factorial_source(config, project_root)
    fit_ids = frozenset(source.split.fit_image_ids)
    validation_ids = frozenset(source.split.validation_image_ids)
    test_ids = frozenset(
        row.image_id for row in source.manifest.images if row.split == "test"
    )
    fit_rows = source.manifest.select("train", range(max(config.stages)), fit_ids)
    validation_rows = source.manifest.select(
        "train", range(max(config.stages)), validation_ids
    )
    used_ids = {row.image_id for row in (*fit_rows, *validation_rows)}
    if (
        not fit_rows
        or not validation_rows
        or fit_ids & validation_ids
        or used_ids & test_ids
        or len(used_ids) != len(fit_rows) + len(validation_rows)
    ):
        raise ValueError("factorial fit/validation identities overlap or touch locked test")
    stages = _stage_sources(source, config.source_fit_policy_hash, config.stages)
    protocol = _protocol_record(
        project_root,
        resolved_config_path,
        config,
        source,
        fit_rows,
        validation_rows,
        stages,
    )
    run_root = config.artifact_root / "runs" / str(protocol["content_hash"])
    for relative in ("checkpoints", "jobs", "job_specs", "ledgers", "protocol", "reports", "work"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(run_root / "protocol" / "protocol.json", protocol)
    publish_immutable_json(run_root / "config_resolved.json", config.as_record())
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol["content_hash"],
                "schema_version": "imagenetr50-parent-recipe-factorial-latest-v1",
            }
        ),
    )
    return ParentRecipeContext(
        project_root,
        config,
        source,
        fit_rows,
        validation_rows,
        stages,
        protocol,
        run_root,
    )


def _prefix_rows(
    rows: Sequence[ImageRecord], stage: int
) -> tuple[ImageRecord, ...]:
    return tuple(row for row in rows if row.task_index < stage)


def evaluate_validation(
    model: AdapterVisionModel,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    stage: int,
) -> ValidationResult:
    """Evaluate raw affine scores on untouched router-validation identities only."""
    if not rows or any(row.split != "train" or row.task_index >= stage for row in rows):
        raise ValueError("development evaluation requires only represented train-split rows")
    model.to(device).eval()
    loader = DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, max(1, int((os.cpu_count() or 1) * 0.75))),
        pin_memory=device.type == "cuda",
    )
    class_ids = torch.tensor(model.classifier.class_ids, dtype=torch.long)
    correct = 0
    loss_sum = 0.0
    examples = 0
    task_correct = [0] * stage
    task_examples = [0] * stage
    started = time.monotonic()
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                local_targets = model.classifier.local_targets(labels_device)
                batch_loss = F.cross_entropy(logits, local_targets, reduction="sum")
            predictions = class_ids[logits.argmax(dim=1).cpu()]
            matches = predictions == labels
            correct += int(matches.sum().item())
            loss_sum += float(batch_loss.item())
            examples += labels.shape[0]
            for task in range(stage):
                mask = labels // 4 == task
                task_examples[task] += int(mask.sum().item())
                task_correct[task] += int((matches & mask).sum().item())
    return ValidationResult(
        100.0 * correct / examples,
        loss_sum / examples,
        examples,
        tuple(task_correct),
        tuple(task_examples),
        time.monotonic() - started,
    )


def _head_statistics(rows: ClassifierRows) -> dict[str, float]:
    norms = torch.linalg.vector_norm(rows.weight.float(), dim=1)
    bias = rows.bias.float()
    return {
        "bias_mean": float(bias.mean().item()),
        "bias_std": float(bias.std(unbiased=False).item()),
        "weight_norm_mean": float(norms.mean().item()),
        "weight_norm_std": float(norms.std(unbiased=False).item()),
    }


def _evaluation_row(
    job_spec: Mapping[str, object],
    epoch: int,
    evaluation: ValidationResult,
    model: AdapterVisionModel,
    optimizer_steps: int,
    image_presentations: int,
    training_seconds: float,
    peak_vram_bytes: int,
) -> dict[str, object]:
    return {
        "condition_key": job_spec["condition_key"],
        "condition_label": job_spec["condition_label"],
        "cross_entropy": evaluation.cross_entropy,
        "epoch": epoch,
        "evaluation_seconds": evaluation.wall_seconds,
        "head_statistics": _head_statistics(model.classifier.rows()),
        "image_presentations": image_presentations,
        "job_hash": job_spec["content_hash"],
        "optimizer_steps": optimizer_steps,
        "peak_vram_bytes": peak_vram_bytes,
        "replication_seed": job_spec["replication_seed"],
        "stage": job_spec["stage"],
        "task_correct": list(evaluation.task_correct),
        "task_examples": list(evaluation.task_examples),
        "training_seconds": training_seconds,
        "validation_accuracy": evaluation.accuracy,
        "validation_examples": evaluation.examples,
    }


def _job_spec(
    context: ParentRecipeContext,
    stage_source: StageSource,
    condition: ParentRecipeCondition,
    replication_seed: int,
) -> dict[str, object]:
    model_seed, training_seed = seed_recipe(
        condition, replication_seed, stage_source.event_sequence
    )
    train_rows = _prefix_rows(context.fit_rows, stage_source.stage)
    validation_rows = _prefix_rows(context.validation_rows, stage_source.stage)
    core: dict[str, object] = {
        **condition.as_record(),
        "model_seed": model_seed,
        "protocol_hash": context.protocol["content_hash"],
        "replication_seed": replication_seed,
        "schema_version": "imagenetr50-parent-recipe-factorial-job-spec-v1",
        "source_child_node_hashes": [
            child.artifact.content_hash for child in stage_source.children
        ],
        "source_root_node_hash": stage_source.root.artifact.content_hash,
        "stage": stage_source.stage,
        "train_image_ids_hash": record_sha256([row.image_id for row in train_rows]),
        "training": asdict(replace(context.config.training, weight_decay=condition.weight_decay)),
        "training_seed": training_seed,
        "validation_image_ids_hash": record_sha256(
            [row.image_id for row in validation_rows]
        ),
    }
    return {**core, "content_hash": record_sha256(core)}


def _load_completed_job(
    context: ParentRecipeContext, job_spec: Mapping[str, object]
) -> dict[str, object] | None:
    target = context.run_root / "jobs" / str(job_spec["content_hash"])
    if not target.is_dir():
        return None
    artifact_sha256 = validate_artifact_directory(target)
    result = load_canonical_json(target / "job_result.json")
    result_core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != JOB_SCHEMA
        or result.get("job_hash") != job_spec["content_hash"]
        or result.get("content_hash") != record_sha256(result_core)
        or result.get("epoch_ledger_sha256")
        != file_sha256(context.run_root / "ledgers" / f"{job_spec['content_hash']}.jsonl")
    ):
        raise ValueError("completed factorial job no longer authenticates")
    return {**result, "artifact_sha256": artifact_sha256}


def _is_source_parent_cell(
    condition: ParentRecipeCondition, replication_seed: int
) -> bool:
    return (
        condition.head_initialization == "inherited_union"
        and condition.weight_decay == 0.0
        and condition.seed_schedule == "parent"
        and replication_seed == 1993
    )


def _publish_job(
    context: ParentRecipeContext,
    job_spec: Mapping[str, object],
    model: AdapterVisionModel,
    final_row: Mapping[str, object],
    original_optimizer_steps: int,
    original_image_presentations: int,
    reused_source_model: bool,
) -> dict[str, object]:
    work = context.run_root / "work" / str(job_spec["content_hash"])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    adapter_sha256 = save_adapter(work / "adapter.safetensors", adapter_factors(model))
    classifier_sha256 = save_classifier(
        work / "classifier.safetensors", model.classifier.rows()
    )
    ledger_path = context.run_root / "ledgers" / f"{job_spec['content_hash']}.jsonl"
    core: dict[str, object] = {
        "adapter_sha256": adapter_sha256,
        "classifier_sha256": classifier_sha256,
        "condition_key": job_spec["condition_key"],
        "condition_label": job_spec["condition_label"],
        "epoch_ledger_sha256": file_sha256(ledger_path),
        "final_cross_entropy": final_row["cross_entropy"],
        "final_validation_accuracy": final_row["validation_accuracy"],
        "job_hash": job_spec["content_hash"],
        "original_image_presentations": original_image_presentations,
        "original_optimizer_steps": original_optimizer_steps,
        "replication_seed": job_spec["replication_seed"],
        "reused_source_model": reused_source_model,
        "schema_version": JOB_SCHEMA,
        "stage": job_spec["stage"],
    }
    result_without_artifact = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(work / "job_result.json", result_without_artifact)
    target = context.run_root / "jobs" / str(job_spec["content_hash"])
    artifact_sha256 = publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return {**result_without_artifact, "artifact_sha256": artifact_sha256}


def _run_job(
    context: ParentRecipeContext,
    stage_source: StageSource,
    condition: ParentRecipeCondition,
    replication_seed: int,
    device: torch.device,
) -> dict[str, object]:
    spec = _job_spec(context, stage_source, condition, replication_seed)
    publish_immutable_json(
        context.run_root / "job_specs" / f"{spec['content_hash']}.json", spec
    )
    completed = _load_completed_job(context, spec)
    if completed is not None:
        return completed
    model_seed, training_seed = int(spec["model_seed"]), int(spec["training_seed"])
    initial_rows = (
        stage_source.inherited_rows
        if condition.head_initialization == "inherited_union"
        else None
    )
    model = AdapterVisionModel(
        create_pinned_backbone(context.source.checkpoint),
        tuple(range(4 * stage_source.stage)),
        context.source.primary_config.lora_rank,
        context.source.primary_config.lora_alpha,
        0.0,
        model_seed,
        initial_rows,
    )
    train_rows = _prefix_rows(context.fit_rows, stage_source.stage)
    validation_rows = _prefix_rows(context.validation_rows, stage_source.stage)
    ledger_path = context.run_root / "ledgers" / f"{spec['content_hash']}.jsonl"
    ledger = ChainedJsonlLedger(ledger_path, EPOCH_SCHEMA)
    ledger.require_unique_keys(("job_hash", "epoch"))
    epochs = tuple(int(row["epoch"]) for row in ledger.rows)
    if epochs not in {(), (0,), (0, context.config.training.epochs)}:
        raise ValueError("factorial epoch ledger has unsupported checkpoint coverage")
    if not epochs:
        initial = evaluate_validation(
            model,
            context.source.config.data_root / "imagenet-r",
            validation_rows,
            context.source.test_transform,
            context.config.training.batch_size,
            context.config.num_workers,
            device,
            stage_source.stage,
        )
        ledger.append(_evaluation_row(spec, 0, initial, model, 0, 0, 0.0, 0))
    reused = _is_source_parent_cell(condition, replication_seed)
    training_result: TrainingResult | None = None
    training_seconds = 0.0
    if reused:
        load_adapter_factors(model, stage_source.root.adapter)
        model.classifier.load_rows(stage_source.root.classifier)
        original_steps = int(stage_source.original_training["optimizer_steps"])
        original_presentations = int(stage_source.original_training["image_presentations"])
        peak_vram = int(stage_source.original_training["peak_vram_bytes"])
    else:
        training_result = train_adapter_model(
            model,
            context.source.config.data_root / "imagenet-r",
            train_rows,
            context.source.train_transform,
            replace(context.config.training, weight_decay=condition.weight_decay),
            training_seed,
            device,
            context.run_root / "checkpoints" / f"{spec['content_hash']}.pt",
            num_workers=context.config.num_workers,
            checkpoint_steps=context.config.checkpoint_steps,
            show_progress=False,
        )
        if training_result.image_presentations != len(train_rows) * context.config.training.epochs:
            raise ValueError("factorial job consumed the wrong finite presentation budget")
        original_steps = training_result.optimizer_steps
        original_presentations = training_result.image_presentations
        peak_vram = training_result.peak_vram_bytes
        training_seconds = training_result.wall_seconds
    if epochs != (0, context.config.training.epochs):
        final = evaluate_validation(
            model,
            context.source.config.data_root / "imagenet-r",
            validation_rows,
            context.source.test_transform,
            context.config.training.batch_size,
            context.config.num_workers,
            device,
            stage_source.stage,
        )
        ledger.append(
            _evaluation_row(
                spec,
                context.config.training.epochs,
                final,
                model,
                original_steps,
                original_presentations,
                training_seconds,
                peak_vram,
            )
        )
    final_row = ledger.rows[-1]
    result = _publish_job(
        context,
        spec,
        model,
        final_row,
        original_steps,
        original_presentations,
        reused,
    )
    del model, training_result
    torch.cuda.empty_cache()
    return result


def select_followup(rows: Sequence[Mapping[str, object]], threshold: float) -> dict[str, object]:
    """Select the strongest development recipe and decide the frozen full-run trigger."""
    final_rows = tuple(dict(row) for row in rows)
    by_key_stage = {
        (str(row["condition_key"]), int(row["stage"])): float(
            row["final_validation_accuracy"]
        )
        for row in final_rows
    }
    joint_key = "fresh__wd5e4__joint"
    parent_key = "inherited_union__wd0__parent"
    stages = (16, 32)
    condition_keys = tuple(sorted({str(row["condition_key"]) for row in final_rows}))
    selected_key = max(
        condition_keys,
        key=lambda key: (
            math.fsum(by_key_stage[(key, stage)] for stage in stages) / len(stages),
            key,
        ),
    )
    closure_rows = []
    for stage in stages:
        joint = by_key_stage[(joint_key, stage)]
        parent = by_key_stage[(parent_key, stage)]
        selected = by_key_stage[(selected_key, stage)]
        gap = joint - parent
        closure = (selected - parent) / gap if gap > 0.0 else float("-inf")
        closure_rows.append(
            {
                "gap_closed_fraction": closure,
                "joint_reference_accuracy": joint,
                "original_parent_accuracy": parent,
                "selected_accuracy": selected,
                "stage": stage,
            }
        )
    trigger = all(
        math.isfinite(float(row["gap_closed_fraction"]))
        and float(row["gap_closed_fraction"]) >= threshold
        for row in closure_rows
    )
    return {
        "full50_triggered": trigger,
        "joint_reference_condition": joint_key,
        "original_parent_condition": parent_key,
        "selection_metric": "mean validation accuracy at stages 16 and 32",
        "selected_condition": selected_key,
        "stage_gap_closure": closure_rows,
        "substantial_gap_closure_fraction": threshold,
    }


def validate_complete_results(
    context: ParentRecipeContext, rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], ...]:
    """Require one finite authenticated result for every frozen factorial job."""
    expected = {
        (condition.key, stage, seed)
        for condition in condition_matrix(context.config)
        for stage in context.config.stages
        for seed in context.config.replication_seeds
    }
    keyed = {
        (str(row["condition_key"]), int(row["stage"]), int(row["replication_seed"])): dict(row)
        for row in rows
    }
    if set(keyed) != expected or len(keyed) != len(rows):
        raise ValueError("parent-recipe factorial is incomplete or duplicated")
    if any(
        not math.isfinite(float(row["final_validation_accuracy"]))
        or not math.isfinite(float(row["final_cross_entropy"]))
        for row in keyed.values()
    ):
        raise ValueError("parent-recipe factorial contains nonfinite measurements")
    return tuple(keyed[key] for key in sorted(keyed, key=lambda item: (item[1], item[0], item[2])))


def run_parent_recipe_factorial(
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    """Run or resume the complete development-only factorial and seal its summary."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the parent-recipe factorial requires BF16 CUDA")
    print("Phase 1/3: authenticating source hierarchy and clean development split.", flush=True)
    context = bootstrap_parent_recipe_factorial(config_path)
    print(f"Resumable artifact directory: {context.run_root}", flush=True)
    conditions = condition_matrix(context.config)
    jobs = tuple(
        (stage_source, condition, seed)
        for stage_source in context.stage_sources
        for condition in conditions
        for seed in context.config.replication_seeds
    )
    completed_rows = tuple(
        loaded
        for stage_source, condition, seed in jobs
        if (
            loaded := _load_completed_job(
                context, _job_spec(context, stage_source, condition, seed)
            )
        )
        is not None
    )
    total_presentations = sum(
        len(_prefix_rows(context.fit_rows, stage_source.stage))
        * context.config.training.epochs
        for stage_source, condition, seed in jobs
        if not _is_source_parent_cell(condition, seed)
    )
    completed_presentations = sum(
        int(row["original_image_presentations"])
        for row in completed_rows
        if not bool(row["reused_source_model"])
    )
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    overall = tqdm(
        total=total_presentations,
        initial=completed_presentations,
        desc="parent-recipe factorial",
        unit="image",
    )
    print("Phase 2/3: fitting the eight controlled recipes at stages 8, 16, and 32.", flush=True)
    device = torch.device("cuda:0")
    results: list[dict[str, object]] = []
    for index, (stage_source, condition, seed) in enumerate(jobs, start=1):
        spec = _job_spec(context, stage_source, condition, seed)
        existing = _load_completed_job(context, spec)
        print(
            f"Job {index:02d}/{len(jobs)}: stage {stage_source.stage}, {condition.label}"
            + (" [reuse]" if existing is not None else ""),
            flush=True,
        )
        result = existing or _run_job(
            context, stage_source, condition, seed, device
        )
        results.append(result)
        if existing is None and not bool(result["reused_source_model"]):
            overall.update(int(result["original_image_presentations"]))
        overall.set_postfix_str(
            f"stage {stage_source.stage}, accuracy {float(result['final_validation_accuracy']):.2f}%"
        )
    overall.close()
    print("Phase 3/3: validating the matrix and freezing the full-50 trigger decision.", flush=True)
    ordered = validate_complete_results(context, results)
    selection = select_followup(
        ordered, context.config.substantial_gap_closure_fraction
    )
    summary_core: dict[str, object] = {
        "conditions": [condition.as_record() for condition in conditions],
        "protocol_hash": context.protocol["content_hash"],
        "rows": list(ordered),
        "schema_version": SUMMARY_SCHEMA,
        "selection": selection,
    }
    summary = {**summary_core, "content_hash": record_sha256(summary_core)}
    summary_path = context.run_root / "summary.json"
    publish_immutable_json(summary_path, summary)
    print(
        f"Selected {selection['selected_condition']}; full-50 trigger="
        f"{selection['full50_triggered']}.",
        flush=True,
    )
    return summary_path


if __name__ == "__main__":
    run_parent_recipe_factorial()


__all__ = [
    "ParentRecipeCondition",
    "ParentRecipeConfig",
    "bootstrap_parent_recipe_factorial",
    "condition_matrix",
    "evaluate_validation",
    "load_parent_recipe_config",
    "load_parent_factorial_source",
    "run_parent_recipe_factorial",
    "seed_recipe",
    "select_followup",
    "validate_complete_results",
]
