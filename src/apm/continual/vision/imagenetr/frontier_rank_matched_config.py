"""Strict configuration for the stage-31 rank-matched joint-IID control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.vision.imagenetr.config import TrainingConfig


DEFAULT_FRONTIER_RANK_MATCHED_CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_rank_matched_control_v11.yaml"
)


@dataclass(frozen=True, slots=True)
class FrontierRankMatchedConfig:
    """Complete immutable configuration for one aggregate-rank control."""

    name: str
    protocol_revision: str
    stage: int
    seed: int
    parent_config: Path
    parent_artifact_root: Path
    parent_run_hash: str
    parent_protocol_sha256: str
    parent_result_sha256: str
    parent_replay_sha256: str
    source_rank: int
    frontier_adapters: int
    target_rank: int
    target_alpha: int
    dropout: float
    training: TrainingConfig
    num_workers: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent protocol", self.parent_protocol_sha256),
            ("parent result", self.parent_result_sha256),
            ("parent replay", self.parent_replay_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_stage31_joint_iid_rank_matched_v11"
            or self.protocol_revision
            != "imagenetr50-stage31-joint-iid-rank-matched-v11"
            or self.stage != 31
            or self.seed != 1993
            or self.source_rank != 16
            or self.frontier_adapters != 5
            or self.target_rank != self.source_rank * self.frontier_adapters
            or self.target_alpha != self.target_rank
            or self.dropout != 0.0
            or self.training
            != TrainingConfig(5, 64, 0.9, 0.0005, 0.0005, 0.01)
            or self.num_workers < 0
        ):
            raise ValueError("configuration differs from the rank-matched control")

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["parent_config"] = str(self.parent_config)
        record["parent_artifact_root"] = str(self.parent_artifact_root)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the rank-matched protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (project_root / expanded).resolve()
    )


def load_frontier_rank_matched_config(
    path: str | Path = DEFAULT_FRONTIER_RANK_MATCHED_CONFIG,
) -> FrontierRankMatchedConfig:
    """Load the single config-driven rank-matched joint-IID control."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "parent", "capacity", "training", "runtime"},
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
            "replay_sha256",
        },
    )
    capacity = _mapping(
        root["capacity"],
        "capacity",
        {"source_rank", "frontier_adapters", "target_rank", "target_alpha", "dropout"},
    )
    training = _mapping(
        root["training"],
        "training",
        {
            "epochs",
            "batch_size",
            "optimizer",
            "momentum",
            "weight_decay",
            "lora_lr",
            "head_lr",
        },
    )
    runtime = _mapping(root["runtime"], "runtime", {"num_workers"})
    if str(training["optimizer"]).lower() != "sgd":
        raise ValueError("rank-matched joint IID must use SGD")
    return FrontierRankMatchedConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["stage"]),
        int(experiment["seed"]),
        _path(parent["config"], project_root),
        _path(parent["artifact_root"], project_root),
        str(parent["run_hash"]),
        str(parent["protocol_sha256"]),
        str(parent["result_sha256"]),
        str(parent["replay_sha256"]),
        int(capacity["source_rank"]),
        int(capacity["frontier_adapters"]),
        int(capacity["target_rank"]),
        int(capacity["target_alpha"]),
        float(capacity["dropout"]),
        TrainingConfig(
            int(training["epochs"]),
            int(training["batch_size"]),
            float(training["momentum"]),
            float(training["weight_decay"]),
            float(training["lora_lr"]),
            float(training["head_lr"]),
        ),
        int(runtime["num_workers"]),
    )


__all__ = [
    "DEFAULT_FRONTIER_RANK_MATCHED_CONFIG",
    "FrontierRankMatchedConfig",
    "load_frontier_rank_matched_config",
]

