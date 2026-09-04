"""Strict configuration for the ImageNet-R replay-adaptation diagnosis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_REPLAY_ADAPTATION_CONFIG = Path(
    "configs/vision/imagenetr/logt_replay_adaptation_v6.yaml"
)


@dataclass(frozen=True, slots=True)
class ReplayAdaptationConfig:
    """Resolved identity of the paired replay-sampling diagnostic matrix."""

    name: str
    protocol_revision: str
    seed: int
    artifact_root: Path
    source_config: Path
    behavior_artifact_root: Path
    promoted_run_hash: str
    hierarchy_policy_hash: str
    hierarchy_complete_sha256: str
    behavior_run_hash: str
    behavior_protocol_sha256: str
    behavior_result_sha256: str
    behavior_training_seal_sha256: str
    diagnostic_stages: tuple[int, ...]
    historical_capacity: int
    sampler_modes: tuple[str, ...]
    weighting_modes: tuple[str, ...]
    optimizer_modes: tuple[str, ...]
    online_epochs: int
    full_history_restarts: int
    cache_seed_mode: str

    def __post_init__(self) -> None:
        for label, identity in (
            ("promoted run", self.promoted_run_hash),
            ("hierarchy policy", self.hierarchy_policy_hash),
            ("hierarchy completion", self.hierarchy_complete_sha256),
            ("behavior run", self.behavior_run_hash),
            ("behavior protocol", self.behavior_protocol_sha256),
            ("behavior result", self.behavior_result_sha256),
            ("behavior training seal", self.behavior_training_seal_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_replay_adaptation_v6"
            or self.protocol_revision != "imagenetr50-replay-adaptation-v6"
            or self.seed != 1993
            or self.diagnostic_stages != (31, 50)
            or self.historical_capacity != 8192
            or self.sampler_modes != ("static", "rotating")
            or self.weighting_modes != ("example_uniform", "task_uniform")
            or self.optimizer_modes != ("carry", "reset_each_stage")
            or self.online_epochs != 4
            or self.full_history_restarts != 3
            or self.cache_seed_mode != "hardlink_train_then_test"
        ):
            raise ValueError("configuration differs from the replay-adaptation protocol")

    @property
    def conditions(self) -> tuple[tuple[str, str, str], ...]:
        """Return the fixed paired factorial in stable report order."""
        return tuple(
            (sampler, weighting, optimizer)
            for sampler in self.sampler_modes
            for weighting in self.weighting_modes
            for optimizer in self.optimizer_modes
        )

    @property
    def config_hash(self) -> str:
        """Return the complete scientific configuration identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in ("artifact_root", "source_config", "behavior_artifact_root"):
            record[name] = str(record[name])
        for name in (
            "diagnostic_stages",
            "sampler_modes",
            "weighting_modes",
            "optimizer_modes",
        ):
            record[name] = list(record[name])
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the replay-adaptation protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    source = Path(os.path.expandvars(str(value))).expanduser()
    return source.resolve() if source.is_absolute() else (project_root / source).resolve()


def load_replay_adaptation_config(
    path: str | Path = DEFAULT_REPLAY_ADAPTATION_CONFIG,
) -> ReplayAdaptationConfig:
    """Load the sole config-driven replay-adaptation diagnostic."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source_path = Path(path).resolve()
    project_root = source_path.parents[3]
    root = _mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "source", "matrix", "runtime"},
    )
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "protocol_revision", "seed"}
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "source_config", "behavior_artifact_root"},
    )
    source = _mapping(
        root["source"],
        "source",
        {
            "promoted_run_hash",
            "hierarchy_policy_hash",
            "hierarchy_complete_sha256",
            "behavior_run_hash",
            "behavior_protocol_sha256",
            "behavior_result_sha256",
            "behavior_training_seal_sha256",
        },
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {
            "diagnostic_stages",
            "historical_capacity",
            "sampler_modes",
            "weighting_modes",
            "optimizer_modes",
            "online_epochs",
            "full_history_restarts",
        },
    )
    runtime = _mapping(root["runtime"], "runtime", {"cache_seed_mode"})
    return ReplayAdaptationConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["seed"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_config"], project_root),
        _path(paths["behavior_artifact_root"], project_root),
        str(source["promoted_run_hash"]),
        str(source["hierarchy_policy_hash"]),
        str(source["hierarchy_complete_sha256"]),
        str(source["behavior_run_hash"]),
        str(source["behavior_protocol_sha256"]),
        str(source["behavior_result_sha256"]),
        str(source["behavior_training_seal_sha256"]),
        tuple(int(value) for value in matrix["diagnostic_stages"]),
        int(matrix["historical_capacity"]),
        tuple(str(value) for value in matrix["sampler_modes"]),
        tuple(str(value) for value in matrix["weighting_modes"]),
        tuple(str(value) for value in matrix["optimizer_modes"]),
        int(matrix["online_epochs"]),
        int(matrix["full_history_restarts"]),
        str(runtime["cache_seed_mode"]),
    )


__all__ = [
    "DEFAULT_REPLAY_ADAPTATION_CONFIG",
    "ReplayAdaptationConfig",
    "load_replay_adaptation_config",
]
