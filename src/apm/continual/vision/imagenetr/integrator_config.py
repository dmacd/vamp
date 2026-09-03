"""Strict configuration for the ungated ImageNet-R LogT prediction integrator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.vision.imagenetr.config import TrainingConfig


FEATURE_VARIANTS = ("scores", "behavior", "behavior_base")


@dataclass(frozen=True, slots=True)
class IntegratorOptimizationConfig:
    """Fixed residual-MLP optimization and convergence choices."""

    hidden_widths: tuple[int, int, int]
    dropout: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    batch_size: int
    persistent_epochs: int
    fresh_minimum_epochs: int
    fresh_maximum_epochs: int
    fresh_patience: int
    improvement_delta: float

    def __post_init__(self) -> None:
        if (
            self.hidden_widths != (1024, 512, 256)
            or not 0.0 <= self.dropout < 1.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.batch_size < 2
            or self.persistent_epochs != 4
            or self.fresh_minimum_epochs < 1
            or self.fresh_maximum_epochs < self.fresh_minimum_epochs
            or not 1 <= self.fresh_patience <= self.fresh_maximum_epochs
            or self.improvement_delta <= 0.0
        ):
            raise ValueError("invalid ImageNet-R integrator optimization settings")


@dataclass(frozen=True, slots=True)
class IntegratorSelectionConfig:
    """Choices frozen from the completed v2 development-only evidence."""

    feature_variant: str
    historical_capacity: int

    def __post_init__(self) -> None:
        if self.feature_variant not in FEATURE_VARIANTS or self.historical_capacity < 1:
            raise ValueError("invalid frozen integrator selection")


@dataclass(frozen=True, slots=True)
class IntegratorReferenceConfig:
    """Authenticated report-only E2-LoRA reference values."""

    local_e2_last: float
    local_e2_incremental: float

    def __post_init__(self) -> None:
        percentages = (self.local_e2_last, self.local_e2_incremental)
        if any(not 0.0 <= value <= 100.0 for value in percentages):
            raise ValueError("integrator references must be percentages")


@dataclass(frozen=True, slots=True)
class ImageNetRIntegratorConfig:
    """Complete resolved identity of the LogT prediction-integrator study."""

    name: str
    protocol_revision: str
    seed: int
    replication_seeds: tuple[int, ...]
    artifact_root: Path
    inference_artifact_root: Path
    router_artifact_root: Path
    predecessor_artifact_root: Path
    data_root: Path
    sealed_run_hash: str
    sealed_u100_policy_hash: str
    sealed_router_run_hash: str
    sealed_router_split_hash: str
    predecessor_run_hash: str
    predecessor_clean_development_sha256: str
    handoff_commit: str
    classes: int
    tasks: int
    classes_per_task: int
    fit_fraction: float
    maximum_levels: int
    diagnostic_maximum_slots: int
    parent_training: str
    source_identity_capacity: int
    reporting_checkpoints: tuple[int, ...]
    consolidation_training: TrainingConfig
    feature_variants: tuple[str, ...]
    optimization: IntegratorOptimizationConfig
    selection: IntegratorSelectionConfig
    references: IntegratorReferenceConfig
    feature_batch_size: int
    num_workers: int
    checkpoint_steps: int
    smoke_tasks: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("sealed run", self.sealed_run_hash),
            ("sealed U100 policy", self.sealed_u100_policy_hash),
            ("sealed router run", self.sealed_router_run_hash),
            ("sealed router split", self.sealed_router_split_hash),
            ("predecessor run", self.predecessor_run_hash),
            ("predecessor clean development", self.predecessor_clean_development_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_prediction_integrator_full_union_ungated_v3"
            or self.protocol_revision != "imagenetr50-logt-integrator-full-union-ungated-v3"
            or self.seed < 0
            or not self.replication_seeds
            or self.replication_seeds[0] != self.seed
            or len(set(self.replication_seeds)) != len(self.replication_seeds)
            or any(seed < 0 for seed in self.replication_seeds)
            or len(self.handoff_commit) < 7
            or self.classes != 200
            or self.tasks != 50
            or self.classes_per_task != 4
            or self.classes != self.tasks * self.classes_per_task
            or self.fit_fraction != 0.8
            or self.maximum_levels != 6
            or self.diagnostic_maximum_slots != 8
            or self.parent_training != "full_union"
            or self.source_identity_capacity != 24_000
            or self.reporting_checkpoints != (2, 4, 8, 16)
            or self.feature_variants != FEATURE_VARIANTS
            or self.selection.feature_variant != "scores"
            or self.selection.historical_capacity != 2048
            or self.feature_batch_size < 1
            or self.num_workers < 0
            or self.checkpoint_steps < 1
            or self.smoke_tasks != 8
        ):
            raise ValueError("configuration differs from the frozen integrator protocol")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of every resolved scientific choice."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in (
            "artifact_root",
            "inference_artifact_root",
            "router_artifact_root",
            "predecessor_artifact_root",
            "data_root",
        ):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the integrator protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _training(value: object) -> TrainingConfig:
    record = _mapping(
        value,
        "hierarchy.consolidation_training",
        {"epochs", "batch_size", "optimizer", "momentum", "weight_decay", "lora_lr", "head_lr"},
    )
    if str(record["optimizer"]).lower() != "sgd":
        raise ValueError("hierarchy consolidation must use SGD")
    return TrainingConfig(
        epochs=int(record["epochs"]),
        batch_size=int(record["batch_size"]),
        momentum=float(record["momentum"]),
        weight_decay=float(record["weight_decay"]),
        lora_lr=float(record["lora_lr"]),
        head_lr=float(record["head_lr"]),
    )


def load_integrator_config(path: str | Path) -> ImageNetRIntegratorConfig:
    """Load the sole strict YAML surface and reject undeclared choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {
            "experiment",
            "paths",
            "sealed_inputs",
            "dataset",
            "hierarchy",
            "integrator",
            "selection",
            "references",
            "runtime",
        },
    )
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "protocol_revision", "seed", "replication_seeds"}
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {
            "artifact_root",
            "inference_artifact_root",
            "router_artifact_root",
            "predecessor_artifact_root",
            "data_root",
        },
    )
    sealed = _mapping(
        root["sealed_inputs"],
        "sealed_inputs",
        {
            "run_hash",
            "u100_policy_hash",
            "router_run_hash",
            "router_split_hash",
            "predecessor_run_hash",
            "predecessor_clean_development_sha256",
            "handoff_commit",
        },
    )
    dataset = _mapping(
        root["dataset"], "dataset", {"classes", "tasks", "classes_per_task", "fit_fraction"}
    )
    hierarchy = _mapping(
        root["hierarchy"],
        "hierarchy",
        {
            "maximum_levels",
            "diagnostic_maximum_slots",
            "parent_training",
            "source_identity_capacity",
            "reporting_checkpoints",
            "consolidation_training",
        },
    )
    integrator = _mapping(
        root["integrator"],
        "integrator",
        {
            "feature_variants",
            "hidden_widths",
            "dropout",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "batch_size",
            "persistent_epochs",
            "fresh_minimum_epochs",
            "fresh_maximum_epochs",
            "fresh_patience",
            "improvement_delta",
        },
    )
    if str(integrator["optimizer"]).lower() != "adamw":
        raise ValueError("the prediction integrator must use AdamW")
    selection = _mapping(
        root["selection"],
        "selection",
        {"feature_variant", "historical_capacity"},
    )
    references = _mapping(
        root["references"],
        "references",
        {"local_e2_last", "local_e2_incremental"},
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {"feature_batch_size", "num_workers", "checkpoint_steps", "smoke_tasks"},
    )
    project_root = source.parents[3]
    return ImageNetRIntegratorConfig(
        name=str(experiment["name"]),
        protocol_revision=str(experiment["protocol_revision"]),
        seed=int(experiment["seed"]),
        replication_seeds=tuple(int(value) for value in experiment["replication_seeds"]),
        artifact_root=_path(paths["artifact_root"], project_root),
        inference_artifact_root=_path(paths["inference_artifact_root"], project_root),
        router_artifact_root=_path(paths["router_artifact_root"], project_root),
        predecessor_artifact_root=_path(paths["predecessor_artifact_root"], project_root),
        data_root=_path(os.environ.get("CIL_DATA_ROOT", str(paths["data_root"])), project_root),
        sealed_run_hash=str(sealed["run_hash"]),
        sealed_u100_policy_hash=str(sealed["u100_policy_hash"]),
        sealed_router_run_hash=str(sealed["router_run_hash"]),
        sealed_router_split_hash=str(sealed["router_split_hash"]),
        predecessor_run_hash=str(sealed["predecessor_run_hash"]),
        predecessor_clean_development_sha256=str(
            sealed["predecessor_clean_development_sha256"]
        ),
        handoff_commit=str(sealed["handoff_commit"]),
        classes=int(dataset["classes"]),
        tasks=int(dataset["tasks"]),
        classes_per_task=int(dataset["classes_per_task"]),
        fit_fraction=float(dataset["fit_fraction"]),
        maximum_levels=int(hierarchy["maximum_levels"]),
        diagnostic_maximum_slots=int(hierarchy["diagnostic_maximum_slots"]),
        parent_training=str(hierarchy["parent_training"]),
        source_identity_capacity=int(hierarchy["source_identity_capacity"]),
        reporting_checkpoints=tuple(int(value) for value in hierarchy["reporting_checkpoints"]),
        consolidation_training=_training(hierarchy["consolidation_training"]),
        feature_variants=tuple(str(value) for value in integrator["feature_variants"]),
        optimization=IntegratorOptimizationConfig(
            hidden_widths=tuple(int(value) for value in integrator["hidden_widths"]),
            dropout=float(integrator["dropout"]),
            learning_rate=float(integrator["learning_rate"]),
            weight_decay=float(integrator["weight_decay"]),
            gradient_clip_norm=float(integrator["gradient_clip_norm"]),
            batch_size=int(integrator["batch_size"]),
            persistent_epochs=int(integrator["persistent_epochs"]),
            fresh_minimum_epochs=int(integrator["fresh_minimum_epochs"]),
            fresh_maximum_epochs=int(integrator["fresh_maximum_epochs"]),
            fresh_patience=int(integrator["fresh_patience"]),
            improvement_delta=float(integrator["improvement_delta"]),
        ),
        selection=IntegratorSelectionConfig(
            feature_variant=str(selection["feature_variant"]),
            historical_capacity=int(selection["historical_capacity"]),
        ),
        references=IntegratorReferenceConfig(
            **{key: float(value) for key, value in references.items()}
        ),
        feature_batch_size=int(runtime["feature_batch_size"]),
        num_workers=int(runtime["num_workers"]),
        checkpoint_steps=int(runtime["checkpoint_steps"]),
        smoke_tasks=int(runtime["smoke_tasks"]),
    )


__all__ = [
    "FEATURE_VARIANTS",
    "ImageNetRIntegratorConfig",
    "IntegratorReferenceConfig",
    "IntegratorSelectionConfig",
    "IntegratorOptimizationConfig",
    "load_integrator_config",
]
