"""Strict configuration for the clean stage-31 macro-token convergence audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_MACRO_CONVERGENCE_CONFIG = Path(
    "configs/vision/imagenetr/logt_macro_token_convergence_v9.yaml"
)


@dataclass(frozen=True, slots=True)
class MacroConvergenceConfig:
    """Complete immutable configuration for the clean convergence audit."""

    name: str
    protocol_revision: str
    stage: int
    depth: int
    screening_seed: int
    replication_seeds: tuple[int, ...]
    artifact_root: Path
    source_macro_config: Path
    source_macro_artifact_root: Path
    source_macro_run_hash: str
    source_macro_protocol_sha256: str
    source_clean_hierarchy_sha256: str
    effective_batch_sizes: tuple[int, ...]
    learning_rates: tuple[float, ...]
    schedule: str
    epochs: int
    warmup_fraction: float
    minimum_learning_rate_ratio: float
    legacy_learning_rate: float
    legacy_effective_batch_size: int
    legacy_epochs: int
    microbatch_size: int
    dropout: float
    weight_decay: float
    gradient_clip_norm: float
    joint_fixed_epochs: int
    cache_dtype: str
    cache_shard_size: int
    cache_limit_bytes: int
    feature_batch_size: int
    num_workers: int
    checkpoint_every_epochs: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("source macro run", self.source_macro_run_hash),
            ("source macro protocol", self.source_macro_protocol_sha256),
            ("source clean hierarchy", self.source_clean_hierarchy_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_macro_token_convergence_v9"
            or self.protocol_revision
            != "imagenetr50-logt-macro-token-convergence-v9"
            or self.stage != 31
            or self.depth != 1
            or self.screening_seed != 1993
            or self.replication_seeds != (1993, 1994, 1995)
            or self.effective_batch_sizes != (64, 128, 512)
            or self.learning_rates != (0.00003, 0.0001, 0.0003)
            or self.schedule != "warmup_cosine"
            or self.epochs != 50
            or self.warmup_fraction != 0.05
            or self.minimum_learning_rate_ratio != 0.01
            or self.legacy_learning_rate != 0.0003
            or self.legacy_effective_batch_size != 512
            or self.legacy_epochs != 20
            or self.microbatch_size != 64
            or self.dropout != 0.1
            or self.weight_decay != 0.0001
            or self.gradient_clip_norm != 1.0
            or self.joint_fixed_epochs != 5
            or self.cache_dtype != "bfloat16"
            or self.cache_shard_size != 64
            or self.cache_limit_bytes != 64 * 1024**3
            or self.feature_batch_size != 64
            or self.num_workers < 0
            or self.checkpoint_every_epochs != 1
            or any(batch % self.microbatch_size for batch in self.effective_batch_sizes)
        ):
            raise ValueError("configuration differs from the convergence audit")

    @property
    def matrix(self) -> tuple[tuple[int, float], ...]:
        """Return the deterministic effective-batch and learning-rate crossing."""
        return tuple(
            (batch_size, learning_rate)
            for batch_size in self.effective_batch_sizes
            for learning_rate in self.learning_rates
        )

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible record."""
        record = asdict(self)
        for name in (
            "artifact_root",
            "source_macro_config",
            "source_macro_artifact_root",
        ):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the convergence protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def load_macro_convergence_config(
    path: str | Path = DEFAULT_MACRO_CONVERGENCE_CONFIG,
) -> MacroConvergenceConfig:
    """Load the single config-driven stage-31 convergence audit."""
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
            "experiment",
            "paths",
            "source",
            "matrix",
            "legacy_control",
            "optimization",
            "joint_iid_control",
            "runtime",
        },
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "stage", "depth", "screening_seed", "replication_seeds"},
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "source_macro_config", "source_macro_artifact_root"},
    )
    source = _mapping(
        root["source"],
        "source",
        {"macro_run_hash", "macro_protocol_sha256", "clean_hierarchy_sha256"},
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {
            "effective_batch_sizes",
            "learning_rates",
            "schedule",
            "epochs",
            "warmup_fraction",
            "minimum_learning_rate_ratio",
        },
    )
    legacy = _mapping(
        root["legacy_control"],
        "legacy control",
        {"learning_rate", "effective_batch_size", "epochs"},
    )
    optimization = _mapping(
        root["optimization"],
        "optimization",
        {"microbatch_size", "dropout", "weight_decay", "gradient_clip_norm"},
    )
    joint = _mapping(
        root["joint_iid_control"], "joint-IID control", {"fixed_epochs"}
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
    return MacroConvergenceConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["stage"]),
        int(experiment["depth"]),
        int(experiment["screening_seed"]),
        tuple(int(value) for value in experiment["replication_seeds"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_macro_config"], project_root),
        _path(paths["source_macro_artifact_root"], project_root),
        str(source["macro_run_hash"]),
        str(source["macro_protocol_sha256"]),
        str(source["clean_hierarchy_sha256"]),
        tuple(int(value) for value in matrix["effective_batch_sizes"]),
        tuple(float(value) for value in matrix["learning_rates"]),
        str(matrix["schedule"]),
        int(matrix["epochs"]),
        float(matrix["warmup_fraction"]),
        float(matrix["minimum_learning_rate_ratio"]),
        float(legacy["learning_rate"]),
        int(legacy["effective_batch_size"]),
        int(legacy["epochs"]),
        int(optimization["microbatch_size"]),
        float(optimization["dropout"]),
        float(optimization["weight_decay"]),
        float(optimization["gradient_clip_norm"]),
        int(joint["fixed_epochs"]),
        str(runtime["cache_dtype"]),
        int(runtime["cache_shard_size"]),
        int(runtime["cache_limit_bytes"]),
        int(runtime["feature_batch_size"]),
        int(runtime["num_workers"]),
        int(runtime["checkpoint_every_epochs"]),
    )


__all__ = [
    "DEFAULT_MACRO_CONVERGENCE_CONFIG",
    "MacroConvergenceConfig",
    "load_macro_convergence_config",
]
