"""Frozen configuration for the full-series persistent affine frontier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_PERSISTENT_AFFINE_CONFIG = Path(
    "configs/vision/imagenetr/logt_persistent_affine_v15.yaml"
)


@dataclass(frozen=True, slots=True)
class PersistentAffineConfig:
    """Complete immutable protocol for two online affine-frontier arms."""

    name: str
    protocol_revision: str
    seed: int
    tasks: int
    artifact_root: Path
    source_config: Path
    source_config_sha256: str
    source_run_hash: str
    source_protocol_sha256: str
    source_hierarchy_policy_hash: str
    source_hierarchy_complete_sha256: str
    stage_matched_joint_sha256: str
    stage_matched_joint_content_hash: str
    selection_run: Path
    selection_result_sha256: str
    h4096_result_sha256: str
    h4096_result_content_hash: str
    h8192_result_sha256: str
    h8192_result_content_hash: str
    historical_capacities: tuple[int, ...]
    epochs_per_stage: tuple[tuple[int, int], ...]
    replay_sampler: str
    replay_weighting: str
    optimizer_policy: str
    integrator_kind: str
    feature_source: str
    feature_dimension: int
    output_classes: int
    source_rank: int
    source_alpha: int
    train_frontier_loras: bool
    train_integrator: bool
    train_node_classifiers: bool
    train_base_vit: bool
    effective_batch_size: int
    microbatch_size: int
    schedule: str
    schedule_horizon_epochs: int
    integrator_peak_learning_rate: float
    lora_peak_learning_rate: float
    warmup_fraction: float
    minimum_learning_rate_ratio: float
    weight_decay: float
    gradient_clip_norm: float
    activation_recomputation: bool
    joint_rank_policy: str
    joint_rank_per_live_node: int
    joint_alpha_policy: str
    joint_epochs: int
    evaluation_batch_size: int
    num_workers: int
    checkpoint_every_epochs: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("source configuration", self.source_config_sha256),
            ("source run", self.source_run_hash),
            ("source protocol", self.source_protocol_sha256),
            ("source hierarchy policy", self.source_hierarchy_policy_hash),
            ("source hierarchy completion", self.source_hierarchy_complete_sha256),
            ("stage-matched joint file", self.stage_matched_joint_sha256),
            ("stage-matched joint result", self.stage_matched_joint_content_hash),
            ("selection result file", self.selection_result_sha256),
            ("H=4096 result file", self.h4096_result_sha256),
            ("H=4096 result", self.h4096_result_content_hash),
            ("H=8192 result file", self.h8192_result_sha256),
            ("H=8192 result", self.h8192_result_content_hash),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_persistent_affine_v15"
            or self.protocol_revision != "imagenetr50-logt-persistent-affine-v15"
            or self.seed != 1993
            or self.tasks != 50
            or self.historical_capacities != (4_096, 8_192)
            or self.epochs_per_stage != ((4_096, 4), (8_192, 5))
            or self.replay_sampler != "stage_keyed_rotating_class_stratified"
            or self.replay_weighting != "example_uniform"
            or self.optimizer_policy != "carry_surviving_named_state"
            or self.integrator_kind != "single_affine"
            or self.feature_source != "node_preclassifier"
            or self.feature_dimension != 768
            or self.output_classes != 200
            or self.source_rank != 16
            or self.source_alpha != 16
            or not self.train_frontier_loras
            or not self.train_integrator
            or self.train_node_classifiers
            or self.train_base_vit
            or self.effective_batch_size != 64
            or self.microbatch_size != 64
            or self.schedule != "warmup_cosine"
            or self.schedule_horizon_epochs != 50
            or self.integrator_peak_learning_rate != 0.00003
            or self.lora_peak_learning_rate != 0.0005
            or self.warmup_fraction != 0.05
            or self.minimum_learning_rate_ratio != 0.01
            or self.weight_decay != 0.0001
            or self.gradient_clip_norm != 1.0
            or not self.activation_recomputation
            or self.joint_rank_policy != "source_rank_times_live_nodes"
            or self.joint_rank_per_live_node != 16
            or self.joint_alpha_policy != "equal_rank"
            or self.joint_epochs != 5
            or self.evaluation_batch_size != 64
            or self.num_workers < 0
            or self.checkpoint_every_epochs != 1
            or self.effective_batch_size % self.microbatch_size
        ):
            raise ValueError("configuration differs from the persistent affine protocol")

    @property
    def epoch_map(self) -> dict[int, int]:
        """Return the fixed per-arrival epoch budget for each replay arm."""
        return dict(self.epochs_per_stage)

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in ("artifact_root", "source_config", "selection_run"):
            record[name] = str(getattr(self, name))
        record["historical_capacities"] = list(self.historical_capacities)
        record["epochs_per_stage"] = [list(values) for values in self.epochs_per_stage]
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the persistent affine protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def load_persistent_affine_config(
    path: str | Path = DEFAULT_PERSISTENT_AFFINE_CONFIG,
) -> PersistentAffineConfig:
    """Load the only supported full-series persistent affine matrix."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source_path = Path(path).resolve()
    project_root = source_path.parents[3]
    root = _mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "sources", "matrix", "architecture", "optimization", "joint_control", "runtime"},
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision", "seed", "tasks"})
    paths = _mapping(root["paths"], "paths", {"artifact_root", "source_config", "selection_run"})
    sources = _mapping(
        root["sources"],
        "sources",
        {
            "source_config_sha256", "source_run_hash", "source_protocol_sha256",
            "source_hierarchy_policy_hash", "source_hierarchy_complete_sha256",
            "stage_matched_joint_sha256", "stage_matched_joint_content_hash",
            "selection_result_sha256", "h4096_result_sha256", "h4096_result_content_hash",
            "h8192_result_sha256", "h8192_result_content_hash",
        },
    )
    matrix = _mapping(
        root["matrix"], "matrix",
        {"historical_capacities", "epochs_per_stage", "replay_sampler", "replay_weighting", "optimizer_policy"},
    )
    architecture = _mapping(
        root["architecture"], "architecture",
        {"integrator_kind", "feature_source", "feature_dimension", "output_classes", "source_rank", "source_alpha", "train_frontier_loras", "train_integrator", "train_node_classifiers", "train_base_vit"},
    )
    optimization = _mapping(
        root["optimization"], "optimization",
        {"effective_batch_size", "microbatch_size", "schedule", "schedule_horizon_epochs", "integrator_peak_learning_rate", "lora_peak_learning_rate", "warmup_fraction", "minimum_learning_rate_ratio", "weight_decay", "gradient_clip_norm", "activation_recomputation"},
    )
    joint = _mapping(
        root["joint_control"], "joint_control",
        {"rank_policy", "rank_per_live_node", "alpha_policy", "epochs"},
    )
    runtime = _mapping(root["runtime"], "runtime", {"evaluation_batch_size", "num_workers", "checkpoint_every_epochs"})
    raw_epochs = matrix["epochs_per_stage"]
    if not isinstance(raw_epochs, Mapping):
        raise ValueError("epochs_per_stage must map replay capacity to epochs")
    return PersistentAffineConfig(
        str(experiment["name"]), str(experiment["protocol_revision"]), int(experiment["seed"]), int(experiment["tasks"]),
        _path(paths["artifact_root"], project_root), _path(paths["source_config"], project_root),
        str(sources["source_config_sha256"]), str(sources["source_run_hash"]), str(sources["source_protocol_sha256"]),
        str(sources["source_hierarchy_policy_hash"]), str(sources["source_hierarchy_complete_sha256"]),
        str(sources["stage_matched_joint_sha256"]), str(sources["stage_matched_joint_content_hash"]),
        _path(paths["selection_run"], project_root), str(sources["selection_result_sha256"]),
        str(sources["h4096_result_sha256"]), str(sources["h4096_result_content_hash"]),
        str(sources["h8192_result_sha256"]), str(sources["h8192_result_content_hash"]),
        tuple(int(value) for value in matrix["historical_capacities"]),
        tuple(sorted((int(key), int(value)) for key, value in raw_epochs.items())),
        str(matrix["replay_sampler"]), str(matrix["replay_weighting"]), str(matrix["optimizer_policy"]),
        str(architecture["integrator_kind"]), str(architecture["feature_source"]), int(architecture["feature_dimension"]),
        int(architecture["output_classes"]), int(architecture["source_rank"]), int(architecture["source_alpha"]),
        bool(architecture["train_frontier_loras"]), bool(architecture["train_integrator"]),
        bool(architecture["train_node_classifiers"]), bool(architecture["train_base_vit"]),
        int(optimization["effective_batch_size"]), int(optimization["microbatch_size"]), str(optimization["schedule"]),
        int(optimization["schedule_horizon_epochs"]), float(optimization["integrator_peak_learning_rate"]),
        float(optimization["lora_peak_learning_rate"]), float(optimization["warmup_fraction"]),
        float(optimization["minimum_learning_rate_ratio"]), float(optimization["weight_decay"]),
        float(optimization["gradient_clip_norm"]), bool(optimization["activation_recomputation"]),
        str(joint["rank_policy"]), int(joint["rank_per_live_node"]), str(joint["alpha_policy"]), int(joint["epochs"]),
        int(runtime["evaluation_batch_size"]), int(runtime["num_workers"]), int(runtime["checkpoint_every_epochs"]),
    )


__all__ = ["DEFAULT_PERSISTENT_AFFINE_CONFIG", "PersistentAffineConfig", "load_persistent_affine_config"]
