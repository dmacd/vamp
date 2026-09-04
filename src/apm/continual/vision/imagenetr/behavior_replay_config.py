"""Strict configuration for the node-adapted behavior replay experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_BEHAVIOR_REPLAY_CONFIG = Path(
    "configs/vision/imagenetr/logt_behavior_replay_v5.yaml"
)


@dataclass(frozen=True, slots=True)
class BehaviorReplayConfig:
    """Resolved identity of the full-50 adapted-latent replay sweep."""

    name: str
    protocol_revision: str
    seed: int
    artifact_root: Path
    source_config: Path
    source_run_hash: str
    source_protocol_sha256: str
    source_hierarchy_policy_hash: str
    source_hierarchy_complete_sha256: str
    source_locked_test_sha256: str
    feature_variant: str
    historical_capacities: tuple[int, ...]
    tasks: int
    cache_seed_mode: str

    def __post_init__(self) -> None:
        for label, identity in (
            ("source run", self.source_run_hash),
            ("source protocol file", self.source_protocol_sha256),
            ("source hierarchy policy", self.source_hierarchy_policy_hash),
            ("source hierarchy completion", self.source_hierarchy_complete_sha256),
            ("source locked test", self.source_locked_test_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_behavior_replay_v5"
            or self.protocol_revision != "imagenetr50-logt-behavior-replay-v5"
            or self.seed != 1993
            or self.feature_variant != "behavior"
            or self.historical_capacities != (2048, 4096, 8192)
            or self.tasks != 50
            or self.cache_seed_mode != "hardlink_train_then_test"
        ):
            raise ValueError("configuration differs from the adapted-latent replay protocol")

    @property
    def config_hash(self) -> str:
        """Return the complete scientific configuration identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["artifact_root"] = str(self.artifact_root)
        record["source_config"] = str(self.source_config)
        record["historical_capacities"] = list(self.historical_capacities)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the adapted-latent replay protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    source = Path(os.path.expandvars(str(value))).expanduser()
    return source.resolve() if source.is_absolute() else (project_root / source).resolve()


def load_behavior_replay_config(
    path: str | Path = DEFAULT_BEHAVIOR_REPLAY_CONFIG,
) -> BehaviorReplayConfig:
    """Load the sole config-driven node-adapted behavior replay matrix."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "source", "matrix", "runtime"},
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "seed"},
    )
    paths = _mapping(
        root["paths"], "paths", {"artifact_root", "source_config"}
    )
    source_record = _mapping(
        root["source"],
        "source",
        {
            "run_hash",
            "protocol_sha256",
            "hierarchy_policy_hash",
            "hierarchy_complete_sha256",
            "locked_test_sha256",
        },
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {"feature_variant", "historical_capacities", "tasks"},
    )
    runtime = _mapping(root["runtime"], "runtime", {"cache_seed_mode"})
    return BehaviorReplayConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["seed"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_config"], project_root),
        str(source_record["run_hash"]),
        str(source_record["protocol_sha256"]),
        str(source_record["hierarchy_policy_hash"]),
        str(source_record["hierarchy_complete_sha256"]),
        str(source_record["locked_test_sha256"]),
        str(matrix["feature_variant"]),
        tuple(int(value) for value in matrix["historical_capacities"]),
        int(matrix["tasks"]),
        str(runtime["cache_seed_mode"]),
    )


__all__ = [
    "BehaviorReplayConfig",
    "DEFAULT_BEHAVIOR_REPLAY_CONFIG",
    "load_behavior_replay_config",
]
