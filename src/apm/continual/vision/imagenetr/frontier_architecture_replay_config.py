"""Strict configuration for the stage-31 architecture replay sweep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_architecture_replay_sweep_v14.yaml"
)


@dataclass(frozen=True, slots=True)
class FrontierArchitectureReplayConfig:
    """Immutable sources and matrix for two matched replay-capacity sweeps."""

    name: str
    protocol_revision: str
    stage: int
    seed: int
    linear_config: Path
    linear_config_sha256: str
    rank80_config: Path
    rank80_config_sha256: str
    parent_artifact_root: Path
    parent_run_hash: str
    parent_replay_sha256: str
    parent_replay_content_hash: str
    linear_full_result_sha256: str
    linear_full_result_content_hash: str
    rank80_full_result_sha256: str
    rank80_full_result_content_hash: str
    historical_capacities: tuple[int, ...]
    current_task_examples: int
    available_historical_examples: int
    validation_examples: int
    linear_epochs: int
    rank80_epochs: int
    linear_checkpoint_rule: str
    rank80_primary_endpoint: str

    def __post_init__(self) -> None:
        for label, identity in (
            ("linear configuration", self.linear_config_sha256),
            ("rank-80 configuration", self.rank80_config_sha256),
            ("parent run", self.parent_run_hash),
            ("parent replay file", self.parent_replay_sha256),
            ("parent replay", self.parent_replay_content_hash),
            ("linear full result file", self.linear_full_result_sha256),
            ("linear full result", self.linear_full_result_content_hash),
            ("rank-80 full result file", self.rank80_full_result_sha256),
            ("rank-80 full result", self.rank80_full_result_content_hash),
        ):
            require_sha256(identity, label)
        if (
            self.name
            != "imagenetr50_stage31_frontier_architecture_replay_sweep_v14"
            or self.protocol_revision
            != "imagenetr50-stage31-frontier-architecture-replay-sweep-v14"
            or self.stage != 31
            or self.seed != 1993
            or self.historical_capacities != (1_024, 2_048, 4_096, 8_192)
            or self.current_task_examples != 367
            or self.available_historical_examples != 11_827
            or self.validation_examples != 3_049
            or self.linear_epochs != 50
            or self.rank80_epochs != 5
            or self.linear_checkpoint_rule != "minimum_validation_nll"
            or self.rank80_primary_endpoint != "fixed_epoch_5"
        ):
            raise ValueError("configuration differs from the architecture replay sweep")

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["linear_config"] = str(self.linear_config)
        record["rank80_config"] = str(self.rank80_config)
        record["parent_artifact_root"] = str(self.parent_artifact_root)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the architecture replay protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (project_root / expanded).resolve()
    )


def load_frontier_architecture_replay_config(
    path: str | Path = DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG,
) -> FrontierArchitectureReplayConfig:
    """Load the only supported stage-31 architecture replay matrix."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "sources", "matrix"},
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "stage", "seed"},
    )
    sources = _mapping(
        root["sources"],
        "sources",
        {
            "linear_config",
            "linear_config_sha256",
            "rank80_config",
            "rank80_config_sha256",
            "parent_artifact_root",
            "parent_run_hash",
            "parent_replay_sha256",
            "parent_replay_content_hash",
            "linear_full_result_sha256",
            "linear_full_result_content_hash",
            "rank80_full_result_sha256",
            "rank80_full_result_content_hash",
        },
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {
            "historical_capacities",
            "current_task_examples",
            "available_historical_examples",
            "validation_examples",
            "linear_epochs",
            "rank80_epochs",
            "linear_checkpoint_rule",
            "rank80_primary_endpoint",
        },
    )
    capacities = matrix["historical_capacities"]
    if not isinstance(capacities, list):
        raise ValueError("historical capacities must be an ordered list")
    return FrontierArchitectureReplayConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["stage"]),
        int(experiment["seed"]),
        _path(sources["linear_config"], project_root),
        str(sources["linear_config_sha256"]),
        _path(sources["rank80_config"], project_root),
        str(sources["rank80_config_sha256"]),
        _path(sources["parent_artifact_root"], project_root),
        str(sources["parent_run_hash"]),
        str(sources["parent_replay_sha256"]),
        str(sources["parent_replay_content_hash"]),
        str(sources["linear_full_result_sha256"]),
        str(sources["linear_full_result_content_hash"]),
        str(sources["rank80_full_result_sha256"]),
        str(sources["rank80_full_result_content_hash"]),
        tuple(int(value) for value in capacities),
        int(matrix["current_task_examples"]),
        int(matrix["available_historical_examples"]),
        int(matrix["validation_examples"]),
        int(matrix["linear_epochs"]),
        int(matrix["rank80_epochs"]),
        str(matrix["linear_checkpoint_rule"]),
        str(matrix["rank80_primary_endpoint"]),
    )


__all__ = [
    "DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG",
    "FrontierArchitectureReplayConfig",
    "load_frontier_architecture_replay_config",
]
