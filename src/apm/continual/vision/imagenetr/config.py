"""Strict configuration loading for the ImageNet-R-50 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
import os

from apm.continual.artifacts import record_sha256


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer and finite-presentation settings for one training family."""

    epochs: int
    batch_size: int
    momentum: float
    weight_decay: float
    lora_lr: float
    head_lr: float

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or not 0.0 <= self.momentum < 1.0
            or self.weight_decay < 0.0
            or self.lora_lr <= 0.0
            or self.head_lr <= 0.0
        ):
            raise ValueError("invalid training configuration")


@dataclass(frozen=True, slots=True)
class ImageNetRConfig:
    """Complete resolved scientific and runtime configuration."""

    name: str
    seed: int
    artifact_root: Path
    data_root: Path
    classes: int
    tasks: int
    classes_per_task: int
    train_fraction: float
    input_size: int
    model_name: str
    precision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    leaf_training: TrainingConfig
    joint_training: TrainingConfig
    repair_training: TrainingConfig
    max_live_nodes_per_level: int
    output_rank: int
    merge_scale: float
    proxy_images_per_node: int
    repair_fractions: tuple[float, ...]
    primary_repair_fraction: float
    repair_material_improvement_points: float
    score_modes: tuple[str, ...]
    num_workers: int
    checkpoint_steps: int
    smoke_tasks: int
    cosine_scale: float

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.seed < 0
            or self.classes != self.tasks * self.classes_per_task
            or self.classes != 200
            or self.tasks != 50
            or not 0.0 < self.train_fraction < 1.0
            or self.input_size != 224
            or self.lora_rank < 1
            or self.lora_alpha < 1
            or self.lora_dropout != 0.0
            or self.output_rank < 1
            or self.max_live_nodes_per_level != 2
            or self.proxy_images_per_node < 1
            or self.num_workers < 0
            or self.checkpoint_steps < 1
            or not 1 <= self.smoke_tasks <= self.tasks
            or self.primary_repair_fraction not in self.repair_fractions
            or self.repair_material_improvement_points < 0.0
            or any(not 0.0 <= fraction <= 1.0 for fraction in self.repair_fractions)
            or set(self.score_modes) != {"raw", "cosine", "affine_calibrated"}
        ):
            raise ValueError("invalid ImageNet-R protocol configuration")
        if self.lora_targets != ("attention.qkv", "mlp.fc1"):
            raise ValueError("the primary protocol adapts only QKV and MLP fc1")

    @property
    def config_hash(self) -> str:
        """Return the canonical content identity of the fully resolved config."""
        return record_sha256(_json_record(self))

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible resolved configuration record."""
        return _json_record(self)


def _json_record(config: ImageNetRConfig) -> dict[str, object]:
    record = asdict(config)
    record["artifact_root"] = str(config.artifact_root)
    record["data_root"] = str(config.data_root)
    return record


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_keys(
    value: object,
    label: str,
    expected: set[str],
) -> Mapping[str, object]:
    """Require one exact mapping surface, including no silent nested extras."""
    record = _require_mapping(value, label)
    if set(record) != expected:
        raise ValueError(f"{label} keys differ from the resolved protocol")
    return record


def _training(value: object, label: str) -> TrainingConfig:
    record = _require_mapping(value, label)
    required = {
        "epochs",
        "batch_size",
        "optimizer",
        "momentum",
        "weight_decay",
        "lora_lr",
        "head_lr",
    }
    if set(record) != required or str(record["optimizer"]).lower() != "sgd":
        raise ValueError(f"{label} must use the exact SGD configuration surface")
    return TrainingConfig(
        epochs=int(record["epochs"]),
        batch_size=int(record["batch_size"]),
        momentum=float(record["momentum"]),
        weight_decay=float(record["weight_decay"]),
        lora_lr=float(record["lora_lr"]),
        head_lr=float(record["head_lr"]),
    )


def _resolved_path(raw: object, project_root: Path) -> Path:
    value = Path(os.path.expandvars(str(raw))).expanduser()
    return value if value.is_absolute() else (project_root / value).resolve()


def load_config(path: str | Path) -> ImageNetRConfig:
    """Load the one supported YAML configuration and reject unknown keys."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - isolated environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _require_mapping(raw, "configuration")
    expected_sections = {
        "experiment",
        "paths",
        "dataset",
        "backbone",
        "lora",
        "leaf_training",
        "joint_training",
        "bank",
        "merge",
        "proxy",
        "repair",
        "routing",
        "runtime",
    }
    if set(root) != expected_sections:
        raise ValueError("configuration sections differ from the resolved protocol")
    experiment = _require_keys(root["experiment"], "experiment", {"name", "seed"})
    paths = _require_keys(root["paths"], "paths", {"artifact_root", "data_root"})
    dataset = _require_keys(
        root["dataset"],
        "dataset",
        {"classes", "tasks", "classes_per_task", "train_fraction", "input_size"},
    )
    backbone = _require_keys(root["backbone"], "backbone", {"model", "precision"})
    lora = _require_keys(root["lora"], "lora", {"rank", "alpha", "dropout", "targets"})
    bank = _require_keys(root["bank"], "bank", {"max_live_nodes_per_level"})
    merge = _require_keys(root["merge"], "merge", {"output_rank", "scale"})
    proxy = _require_keys(root["proxy"], "proxy", {"images_per_node"})
    repair = _require_keys(
        root["repair"],
        "repair",
        {"fractions", "primary_fraction", "material_improvement_points", "training"},
    )
    routing = _require_keys(
        root["routing"], "routing", {"score_modes", "cosine_scale"}
    )
    runtime = _require_keys(
        root["runtime"],
        "runtime",
        {"num_workers", "checkpoint_steps", "smoke_tasks"},
    )
    project_root = source.parents[3]
    leaf = _training(root["leaf_training"], "leaf_training")
    joint = _training(root["joint_training"], "joint_training")
    repair_training = _training(repair["training"], "repair.training")
    config = ImageNetRConfig(
        name=str(experiment["name"]),
        seed=int(experiment["seed"]),
        artifact_root=_resolved_path(paths["artifact_root"], project_root),
        data_root=_resolved_path(
            os.environ.get("CIL_DATA_ROOT", str(paths["data_root"])), project_root
        ),
        classes=int(dataset["classes"]),
        tasks=int(dataset["tasks"]),
        classes_per_task=int(dataset["classes_per_task"]),
        train_fraction=float(dataset["train_fraction"]),
        input_size=int(dataset["input_size"]),
        model_name=str(backbone["model"]),
        precision=str(backbone["precision"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        lora_targets=tuple(str(item) for item in lora["targets"]),
        leaf_training=leaf,
        joint_training=joint,
        repair_training=repair_training,
        max_live_nodes_per_level=int(bank["max_live_nodes_per_level"]),
        output_rank=int(merge["output_rank"]),
        merge_scale=float(merge["scale"]),
        proxy_images_per_node=int(proxy["images_per_node"]),
        repair_fractions=tuple(float(item) for item in repair["fractions"]),
        primary_repair_fraction=float(repair["primary_fraction"]),
        repair_material_improvement_points=float(repair["material_improvement_points"]),
        score_modes=tuple(str(item) for item in routing["score_modes"]),
        num_workers=int(runtime["num_workers"]),
        checkpoint_steps=int(runtime["checkpoint_steps"]),
        smoke_tasks=int(runtime["smoke_tasks"]),
        cosine_scale=float(routing["cosine_scale"]),
    )
    if config.model_name != "vit_base_patch16_224.augreg_in21k":
        raise ValueError("the resolved protocol requires the explicit pinned timm name")
    if config.precision != "bfloat16":
        raise ValueError("the primary protocol precision must be bfloat16")
    return config


__all__ = ["ImageNetRConfig", "TrainingConfig", "load_config"]
