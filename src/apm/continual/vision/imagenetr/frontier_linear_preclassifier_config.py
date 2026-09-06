"""Strict configuration for the stage-31 linear pre-classifier integrator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_linear_preclassifier_v13.yaml"
)


@dataclass(frozen=True, slots=True)
class FrontierLinearPreclassifierConfig:
    """Complete immutable configuration for one full-fit architecture ablation."""

    name: str
    protocol_revision: str
    stage: int
    seed: int
    parent_config: Path
    parent_artifact_root: Path
    parent_run_hash: str
    parent_protocol_sha256: str
    parent_result_sha256: str
    parent_result_content_hash: str
    parent_replay_sha256: str
    integrator_kind: str
    feature_source: str
    feature_normalization: str
    input_order: str
    initialization: str
    active_nodes: int
    feature_dimension: int
    output_classes: int
    source_rank: int
    source_alpha: int
    lora_parameters_per_node: int
    train_frontier_loras: bool
    train_integrator: bool
    train_node_classifiers: bool
    train_base_vit: bool
    historical_capacity: int
    current_task_examples: int
    full_fit_examples: int
    validation_examples: int
    effective_batch_size: int
    microbatch_size: int
    epochs: int
    schedule: str
    integrator_peak_learning_rate: float
    lora_peak_learning_rate: float
    warmup_fraction: float
    minimum_learning_rate_ratio: float
    weight_decay: float
    gradient_clip_norm: float
    activation_recomputation: bool
    evaluation_batch_size: int
    num_workers: int
    checkpoint_every_epochs: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent protocol", self.parent_protocol_sha256),
            ("parent result file", self.parent_result_sha256),
            ("parent result", self.parent_result_content_hash),
            ("parent replay", self.parent_replay_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name
            != "imagenetr50_stage31_frontier_linear_preclassifier_v13"
            or self.protocol_revision
            != "imagenetr50-stage31-frontier-linear-preclassifier-v13"
            or self.stage != 31
            or self.seed != 1993
            or self.integrator_kind != "single_affine"
            or self.feature_source != "node_preclassifier"
            or self.feature_normalization
            != "none_after_pinned_vit_pre_logits"
            or self.input_order != "ascending_frontier_level"
            or self.initialization != "exact_local_classifier_union"
            or self.active_nodes != 5
            or self.feature_dimension != 768
            or self.output_classes != 200
            or self.source_rank != 16
            or self.source_alpha != 16
            or self.lora_parameters_per_node != 1_327_104
            or not self.train_frontier_loras
            or not self.train_integrator
            or self.train_node_classifiers
            or self.train_base_vit
            or self.historical_capacity != 11_827
            or self.current_task_examples != 367
            or self.full_fit_examples != 12_194
            or self.historical_capacity + self.current_task_examples
            != self.full_fit_examples
            or self.validation_examples != 3_049
            or self.effective_batch_size != 64
            or self.microbatch_size != 64
            or self.epochs != 50
            or self.schedule != "warmup_cosine"
            or self.integrator_peak_learning_rate != 0.00003
            or self.lora_peak_learning_rate != 0.0005
            or self.warmup_fraction != 0.05
            or self.minimum_learning_rate_ratio != 0.01
            or self.weight_decay != 0.0001
            or self.gradient_clip_norm != 1.0
            or not self.activation_recomputation
            or self.evaluation_batch_size != 64
            or self.num_workers < 0
            or self.checkpoint_every_epochs != 1
            or self.effective_batch_size % self.microbatch_size
        ):
            raise ValueError(
                "configuration differs from the linear pre-classifier ablation"
            )

    @property
    def input_dimension(self) -> int:
        """Return the concatenated dimension of the five node representations."""
        return self.active_nodes * self.feature_dimension

    @property
    def integrator_parameters(self) -> int:
        """Return the direct affine readout's weight and bias count."""
        return self.input_dimension * self.output_classes + self.output_classes

    @property
    def frontier_lora_parameters(self) -> int:
        """Return the active parameters in all five rank-16 adapters."""
        return self.active_nodes * self.lora_parameters_per_node

    @property
    def active_parameters(self) -> int:
        """Return the trainable integrator plus frontier-LoRA parameter count."""
        return self.integrator_parameters + self.frontier_lora_parameters

    @property
    def macro_peak_learning_rate(self) -> float:
        """Expose the common training engine's generic integration-head rate."""
        return self.integrator_peak_learning_rate

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["parent_config"] = str(self.parent_config)
        record["parent_artifact_root"] = str(self.parent_artifact_root)
        record.update(
            {
                "active_parameters": self.active_parameters,
                "frontier_lora_parameters": self.frontier_lora_parameters,
                "input_dimension": self.input_dimension,
                "integrator_parameters": self.integrator_parameters,
            }
        )
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"{label} keys differ from the linear pre-classifier protocol"
        )
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (project_root / expanded).resolve()
    )


def load_frontier_linear_preclassifier_config(
    path: str | Path = DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG,
) -> FrontierLinearPreclassifierConfig:
    """Load the single config-driven linear pre-classifier ablation."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "parent", "architecture", "data", "optimization", "runtime"},
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "stage", "seed"},
    )
    parent = _mapping(
        root["parent"],
        "parent",
        {
            "config",
            "artifact_root",
            "run_hash",
            "protocol_sha256",
            "result_sha256",
            "result_content_hash",
            "replay_sha256",
        },
    )
    architecture_keys = {
        "integrator_kind",
        "feature_source",
        "feature_normalization",
        "input_order",
        "initialization",
        "active_nodes",
        "feature_dimension",
        "output_classes",
        "source_rank",
        "source_alpha",
        "lora_parameters_per_node",
        "train_frontier_loras",
        "train_integrator",
        "train_node_classifiers",
        "train_base_vit",
    }
    architecture = _mapping(root["architecture"], "architecture", architecture_keys)
    data = _mapping(
        root["data"],
        "data",
        {
            "historical_capacity",
            "current_task_examples",
            "full_fit_examples",
            "validation_examples",
        },
    )
    optimization = _mapping(
        root["optimization"],
        "optimization",
        {
            "effective_batch_size",
            "microbatch_size",
            "epochs",
            "schedule",
            "integrator_peak_learning_rate",
            "lora_peak_learning_rate",
            "warmup_fraction",
            "minimum_learning_rate_ratio",
            "weight_decay",
            "gradient_clip_norm",
            "activation_recomputation",
        },
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {"evaluation_batch_size", "num_workers", "checkpoint_every_epochs"},
    )
    return FrontierLinearPreclassifierConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["stage"]),
        int(experiment["seed"]),
        _path(parent["config"], project_root),
        _path(parent["artifact_root"], project_root),
        str(parent["run_hash"]),
        str(parent["protocol_sha256"]),
        str(parent["result_sha256"]),
        str(parent["result_content_hash"]),
        str(parent["replay_sha256"]),
        str(architecture["integrator_kind"]),
        str(architecture["feature_source"]),
        str(architecture["feature_normalization"]),
        str(architecture["input_order"]),
        str(architecture["initialization"]),
        int(architecture["active_nodes"]),
        int(architecture["feature_dimension"]),
        int(architecture["output_classes"]),
        int(architecture["source_rank"]),
        int(architecture["source_alpha"]),
        int(architecture["lora_parameters_per_node"]),
        bool(architecture["train_frontier_loras"]),
        bool(architecture["train_integrator"]),
        bool(architecture["train_node_classifiers"]),
        bool(architecture["train_base_vit"]),
        int(data["historical_capacity"]),
        int(data["current_task_examples"]),
        int(data["full_fit_examples"]),
        int(data["validation_examples"]),
        int(optimization["effective_batch_size"]),
        int(optimization["microbatch_size"]),
        int(optimization["epochs"]),
        str(optimization["schedule"]),
        float(optimization["integrator_peak_learning_rate"]),
        float(optimization["lora_peak_learning_rate"]),
        float(optimization["warmup_fraction"]),
        float(optimization["minimum_learning_rate_ratio"]),
        float(optimization["weight_decay"]),
        float(optimization["gradient_clip_norm"]),
        bool(optimization["activation_recomputation"]),
        int(runtime["evaluation_batch_size"]),
        int(runtime["num_workers"]),
        int(runtime["checkpoint_every_epochs"]),
    )


__all__ = [
    "DEFAULT_FRONTIER_LINEAR_PRECLASSIFIER_CONFIG",
    "FrontierLinearPreclassifierConfig",
    "load_frontier_linear_preclassifier_config",
]
