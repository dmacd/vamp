"""Strict configuration for the ImageNet-R macro-token ceiling study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_MACRO_TOKEN_CONFIG = Path(
    "configs/vision/imagenetr/logt_macro_token_integrator_v8.yaml"
)


@dataclass(frozen=True, slots=True)
class MacroTokenOptimization:
    """Frozen optimization and early-stopping contract for macro-token heads."""

    dropout: float
    weight_decay: float
    gradient_clip_norm: float
    microbatch_size: int
    accumulation_steps: int
    minimum_epochs: int
    maximum_epochs: int
    patience: int
    improvement_delta: float

    def __post_init__(self) -> None:
        if (
            self.dropout != 0.1
            or self.weight_decay != 0.0001
            or self.gradient_clip_norm != 1.0
            or self.microbatch_size != 64
            or self.accumulation_steps != 8
            or self.minimum_epochs != 20
            or self.maximum_epochs != 100
            or self.patience != 10
            or self.improvement_delta != 0.0001
        ):
            raise ValueError("macro-token optimization differs from the frozen study")

    @property
    def effective_batch_size(self) -> int:
        """Return the nominal number of examples contributing to one update."""
        return self.microbatch_size * self.accumulation_steps


@dataclass(frozen=True, slots=True)
class MacroTokenConfig:
    """Complete scientific identity of the clean ceiling and locked refit study."""

    name: str
    protocol_revision: str
    seed: int
    replication_seeds: tuple[int, ...]
    owner_probe_seed: int
    artifact_root: Path
    source_config: Path
    source_artifact_root: Path
    source_run_hash: str
    source_protocol_sha256: str
    fit_hierarchy_policy_hash: str
    fit_hierarchy_008_sha256: str
    all_train_hierarchy_policy_hash: str
    all_train_hierarchy_050_sha256: str
    stage_matched_joint_sha256: str
    source_locked_test_sha256: str
    stages: tuple[int, ...]
    depths: tuple[int, ...]
    learning_rates: tuple[float, ...]
    macro_optimization: MacroTokenOptimization
    control_learning_rate: float
    control_weight_decay: float
    control_batch_size: int
    control_hidden_widths: tuple[int, ...]
    control_dropout: float
    control_gradient_clip_norm: float
    cache_dtype: str
    cache_shard_size: int
    cache_limit_bytes: int
    feature_batch_size: int
    num_workers: int
    checkpoint_every_epochs: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("source run", self.source_run_hash),
            ("source protocol", self.source_protocol_sha256),
            ("fit hierarchy policy", self.fit_hierarchy_policy_hash),
            ("fit hierarchy smoke", self.fit_hierarchy_008_sha256),
            ("all-training hierarchy policy", self.all_train_hierarchy_policy_hash),
            ("all-training hierarchy", self.all_train_hierarchy_050_sha256),
            ("stage-matched joint curve", self.stage_matched_joint_sha256),
            ("source locked test", self.source_locked_test_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_macro_token_integrator_v8"
            or self.protocol_revision != "imagenetr50-logt-macro-token-integrator-v8"
            or self.seed != 1993
            or self.replication_seeds != (1993, 1994, 1995)
            or self.owner_probe_seed != 1993
            or self.stages != (31, 50)
            or self.depths != (1, 2)
            or self.learning_rates != (0.0001, 0.0003, 0.001)
            or self.control_learning_rate != 0.001
            or self.control_weight_decay != 0.0001
            or self.control_batch_size != 512
            or self.control_hidden_widths != (1024, 512, 256)
            or self.control_dropout != 0.1
            or self.control_gradient_clip_norm != 1.0
            or self.cache_dtype != "bfloat16"
            or self.cache_shard_size != 64
            or self.cache_limit_bytes != 64 * 1024**3
            or self.feature_batch_size != 64
            or self.num_workers < 0
            or self.checkpoint_every_epochs != 1
        ):
            raise ValueError("configuration differs from the macro-token ceiling study")

    @property
    def config_hash(self) -> str:
        """Return every frozen scientific and runtime choice."""
        return record_sha256(self.as_record())

    @property
    def candidates(self) -> tuple[tuple[int, float], ...]:
        """Return the predeclared architecture/learning-rate selection cells."""
        return tuple(
            (depth, learning_rate)
            for depth in self.depths
            for learning_rate in self.learning_rates
        )

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in ("artifact_root", "source_config", "source_artifact_root"):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the macro-token protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def load_macro_token_config(
    path: str | Path = DEFAULT_MACRO_TOKEN_CONFIG,
) -> MacroTokenConfig:
    """Load the single config-driven macro-token ceiling experiment."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source_path = Path(path).resolve()
    project_root = source_path.parents[3]
    root = _mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "source", "matrix", "optimization", "control", "runtime"},
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "seed", "replication_seeds", "owner_probe_seed"},
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "source_config", "source_artifact_root"},
    )
    source = _mapping(
        root["source"],
        "source",
        {
            "run_hash",
            "protocol_sha256",
            "fit_hierarchy_policy_hash",
            "fit_hierarchy_008_sha256",
            "all_train_hierarchy_policy_hash",
            "all_train_hierarchy_050_sha256",
            "stage_matched_joint_sha256",
            "locked_test_sha256",
        },
    )
    matrix = _mapping(
        root["matrix"], "matrix", {"stages", "depths", "learning_rates"}
    )
    optimization = _mapping(
        root["optimization"],
        "optimization",
        {
            "dropout",
            "weight_decay",
            "gradient_clip_norm",
            "microbatch_size",
            "accumulation_steps",
            "minimum_epochs",
            "maximum_epochs",
            "patience",
            "improvement_delta",
        },
    )
    control = _mapping(
        root["control"],
        "control",
        {
            "learning_rate",
            "weight_decay",
            "batch_size",
            "hidden_widths",
            "dropout",
            "gradient_clip_norm",
        },
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {
            "cache_dtype",
            "cache_shard_size",
            "cache_limit_bytes",
            "feature_batch_size",
            "num_workers",
            "checkpoint_every_epochs",
        },
    )
    return MacroTokenConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["seed"]),
        tuple(int(value) for value in experiment["replication_seeds"]),
        int(experiment["owner_probe_seed"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_config"], project_root),
        _path(paths["source_artifact_root"], project_root),
        str(source["run_hash"]),
        str(source["protocol_sha256"]),
        str(source["fit_hierarchy_policy_hash"]),
        str(source["fit_hierarchy_008_sha256"]),
        str(source["all_train_hierarchy_policy_hash"]),
        str(source["all_train_hierarchy_050_sha256"]),
        str(source["stage_matched_joint_sha256"]),
        str(source["locked_test_sha256"]),
        tuple(int(value) for value in matrix["stages"]),
        tuple(int(value) for value in matrix["depths"]),
        tuple(float(value) for value in matrix["learning_rates"]),
        MacroTokenOptimization(**{key: optimization[key] for key in optimization}),
        float(control["learning_rate"]),
        float(control["weight_decay"]),
        int(control["batch_size"]),
        tuple(int(value) for value in control["hidden_widths"]),
        float(control["dropout"]),
        float(control["gradient_clip_norm"]),
        str(runtime["cache_dtype"]),
        int(runtime["cache_shard_size"]),
        int(runtime["cache_limit_bytes"]),
        int(runtime["feature_batch_size"]),
        int(runtime["num_workers"]),
        int(runtime["checkpoint_every_epochs"]),
    )


__all__ = [
    "DEFAULT_MACRO_TOKEN_CONFIG",
    "MacroTokenConfig",
    "MacroTokenOptimization",
    "load_macro_token_config",
]
