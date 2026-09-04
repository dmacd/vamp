"""Strict configuration for the ImageNet-R two-layer latent replay ablation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_TWO_LAYER_REPLAY_CONFIG = Path(
    "configs/vision/imagenetr/logt_two_layer_replay_v7.yaml"
)


@dataclass(frozen=True, slots=True)
class TwoLayerReplayConfig:
    """Resolved identity of the nested two-layer representation ablation."""

    name: str
    protocol_revision: str
    seed: int
    artifact_root: Path
    source_config: Path
    behavior_artifact_root: Path
    comparison_artifact_root: Path
    promoted_run_hash: str
    hierarchy_policy_hash: str
    hierarchy_complete_sha256: str
    behavior_run_hash: str
    behavior_protocol_sha256: str
    behavior_result_sha256: str
    behavior_training_seal_sha256: str
    comparison_run_hash: str
    comparison_result_sha256: str
    comparison_training_seal_sha256: str
    diagnostic_stages: tuple[int, ...]
    historical_capacity: int
    sampler_modes: tuple[str, ...]
    weighting_modes: tuple[str, ...]
    optimizer_modes: tuple[str, ...]
    online_epochs: int
    full_history_restarts: int
    existing_latent: str
    added_latent: str
    normalization: str
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
            ("comparison run", self.comparison_run_hash),
            ("comparison result", self.comparison_result_sha256),
            ("comparison training seal", self.comparison_training_seal_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_two_layer_replay_v7"
            or self.protocol_revision != "imagenetr50-two-layer-replay-v7"
            or self.seed != 1993
            or self.diagnostic_stages != (31, 50)
            or self.historical_capacity != 8192
            or self.sampler_modes != ("static", "rotating")
            or self.weighting_modes != ("example_uniform", "task_uniform")
            or self.optimizer_modes != ("carry", "reset_each_stage")
            or self.online_epochs != 4
            or self.full_history_restarts != 3
            or self.existing_latent != "final_preclassifier_class_token"
            or self.added_latent
            != "penultimate_block_output_class_token_before_final_block_and_norm"
            or self.normalization != "independent_per_image_layer_norm"
            or self.cache_seed_mode != "hardlink_train_then_test"
        ):
            raise ValueError("configuration differs from the two-layer replay protocol")

    @property
    def feature_variant(self) -> str:
        """Return the nested two-layer integrator input family."""
        return "behavior_two_layer"

    @property
    def common_state_name(self) -> str:
        """Reuse the v6 initialization identity for all shared parameter blocks."""
        return "persistent-node-adapted-behavior-common-seed-v1"

    @property
    def conditions(self) -> tuple[tuple[str, str, str], ...]:
        """Return the unchanged eight-cell replay factorial."""
        return tuple(
            (sampler, weighting, optimizer)
            for sampler in self.sampler_modes
            for weighting in self.weighting_modes
            for optimizer in self.optimizer_modes
        )

    @property
    def config_hash(self) -> str:
        """Return every frozen scientific and comparison choice."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in (
            "artifact_root",
            "source_config",
            "behavior_artifact_root",
            "comparison_artifact_root",
        ):
            record[name] = str(record[name])
        for name in (
            "diagnostic_stages",
            "sampler_modes",
            "weighting_modes",
            "optimizer_modes",
        ):
            record[name] = list(record[name])
        record["feature_variant"] = self.feature_variant
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the two-layer replay protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    source = Path(os.path.expandvars(str(value))).expanduser()
    return source.resolve() if source.is_absolute() else (project_root / source).resolve()


def load_two_layer_replay_config(
    path: str | Path = DEFAULT_TWO_LAYER_REPLAY_CONFIG,
) -> TwoLayerReplayConfig:
    """Load the sole config-driven two-layer replay ablation."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source_path = Path(path).resolve()
    project_root = source_path.parents[3]
    root = _mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        "configuration",
        {
            "comparison",
            "experiment",
            "matrix",
            "paths",
            "representation",
            "runtime",
            "source",
        },
    )
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "protocol_revision", "seed"}
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {
            "artifact_root",
            "behavior_artifact_root",
            "comparison_artifact_root",
            "source_config",
        },
    )
    source = _mapping(
        root["source"],
        "source",
        {
            "behavior_protocol_sha256",
            "behavior_result_sha256",
            "behavior_run_hash",
            "behavior_training_seal_sha256",
            "hierarchy_complete_sha256",
            "hierarchy_policy_hash",
            "promoted_run_hash",
        },
    )
    comparison = _mapping(
        root["comparison"],
        "comparison",
        {"result_sha256", "run_hash", "training_seal_sha256"},
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {
            "diagnostic_stages",
            "full_history_restarts",
            "historical_capacity",
            "online_epochs",
            "optimizer_modes",
            "sampler_modes",
            "weighting_modes",
        },
    )
    representation = _mapping(
        root["representation"],
        "representation",
        {"added_latent", "existing_latent", "normalization"},
    )
    runtime = _mapping(root["runtime"], "runtime", {"cache_seed_mode"})
    return TwoLayerReplayConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["seed"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_config"], project_root),
        _path(paths["behavior_artifact_root"], project_root),
        _path(paths["comparison_artifact_root"], project_root),
        str(source["promoted_run_hash"]),
        str(source["hierarchy_policy_hash"]),
        str(source["hierarchy_complete_sha256"]),
        str(source["behavior_run_hash"]),
        str(source["behavior_protocol_sha256"]),
        str(source["behavior_result_sha256"]),
        str(source["behavior_training_seal_sha256"]),
        str(comparison["run_hash"]),
        str(comparison["result_sha256"]),
        str(comparison["training_seal_sha256"]),
        tuple(int(value) for value in matrix["diagnostic_stages"]),
        int(matrix["historical_capacity"]),
        tuple(str(value) for value in matrix["sampler_modes"]),
        tuple(str(value) for value in matrix["weighting_modes"]),
        tuple(str(value) for value in matrix["optimizer_modes"]),
        int(matrix["online_epochs"]),
        int(matrix["full_history_restarts"]),
        str(representation["existing_latent"]),
        str(representation["added_latent"]),
        str(representation["normalization"]),
        str(runtime["cache_seed_mode"]),
    )


__all__ = [
    "DEFAULT_TWO_LAYER_REPLAY_CONFIG",
    "TwoLayerReplayConfig",
    "load_two_layer_replay_config",
]
