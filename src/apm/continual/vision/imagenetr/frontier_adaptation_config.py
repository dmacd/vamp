"""Strict configuration for clean stage-31 frontier-LoRA adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_FRONTIER_ADAPTATION_CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_lora_adaptation_v10.yaml"
)


@dataclass(frozen=True, slots=True)
class FrontierAdaptationConfig:
    """Complete immutable configuration for the nested-H adaptation sweep."""

    name: str
    protocol_revision: str
    stage: int
    seed: int
    artifact_root: Path
    source_macro_config: Path
    source_macro_artifact_root: Path
    source_convergence_artifact_root: Path
    source_macro_run_hash: str
    source_macro_protocol_sha256: str
    source_clean_hierarchy_sha256: str
    source_clean_stage31_sha256: str
    source_convergence_run_hash: str
    source_convergence_protocol_sha256: str
    source_convergence_result_sha256: str
    historical_capacities: tuple[int, ...]
    current_task_examples: int
    full_fit_examples: int
    frozen_full_fit_control: bool
    effective_batch_size: int
    microbatch_size: int
    epochs: int
    schedule: str
    macro_peak_learning_rate: float
    lora_peak_learning_rate: float
    warmup_fraction: float
    minimum_learning_rate_ratio: float
    dropout: float
    weight_decay: float
    gradient_clip_norm: float
    activation_recomputation: bool
    evaluation_batch_size: int
    num_workers: int
    checkpoint_every_epochs: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("source macro run", self.source_macro_run_hash),
            ("source macro protocol", self.source_macro_protocol_sha256),
            ("source clean hierarchy", self.source_clean_hierarchy_sha256),
            ("source clean stage 31", self.source_clean_stage31_sha256),
            ("source convergence run", self.source_convergence_run_hash),
            ("source convergence protocol", self.source_convergence_protocol_sha256),
            ("source convergence result", self.source_convergence_result_sha256),
        ):
            require_sha256(identity, label)
        if (
            self.name != "imagenetr50_logt_frontier_lora_adaptation_v10"
            or self.protocol_revision
            != "imagenetr50-logt-frontier-lora-adaptation-v10"
            or self.stage != 31
            or self.seed != 1993
            or self.historical_capacities != (1024, 2048, 4096, 8192, 11827)
            or self.current_task_examples != 367
            or self.full_fit_examples != 12194
            or self.historical_capacities[-1] + self.current_task_examples
            != self.full_fit_examples
            or not self.frozen_full_fit_control
            or self.effective_batch_size != 64
            or self.microbatch_size != 64
            or self.epochs != 50
            or self.schedule != "warmup_cosine"
            or self.macro_peak_learning_rate != 0.00003
            or self.lora_peak_learning_rate != 0.0005
            or self.warmup_fraction != 0.05
            or self.minimum_learning_rate_ratio != 0.01
            or self.dropout != 0.1
            or self.weight_decay != 0.0001
            or self.gradient_clip_norm != 1.0
            or not self.activation_recomputation
            or self.evaluation_batch_size != 64
            or self.num_workers < 0
            or self.checkpoint_every_epochs != 1
            or self.effective_batch_size % self.microbatch_size
        ):
            raise ValueError("configuration differs from the frontier adaptation audit")

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in (
            "artifact_root",
            "source_macro_config",
            "source_macro_artifact_root",
            "source_convergence_artifact_root",
        ):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the frontier adaptation protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (project_root / expanded).resolve()
    )


def load_frontier_adaptation_config(
    path: str | Path = DEFAULT_FRONTIER_ADAPTATION_CONFIG,
) -> FrontierAdaptationConfig:
    """Load the single config-driven stage-31 frontier adaptation sweep."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source_path = Path(path).resolve()
    project_root = source_path.parents[3]
    root = _mapping(
        yaml.safe_load(source_path.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "source", "matrix", "optimization", "runtime"},
    )
    experiment = _mapping(
        root["experiment"],
        "experiment",
        {"name", "protocol_revision", "stage", "seed"},
    )
    paths = _mapping(
        root["paths"],
        "paths",
        {
            "artifact_root",
            "source_macro_config",
            "source_macro_artifact_root",
            "source_convergence_artifact_root",
        },
    )
    source = _mapping(
        root["source"],
        "source",
        {
            "macro_run_hash",
            "macro_protocol_sha256",
            "clean_hierarchy_sha256",
            "clean_stage31_sha256",
            "convergence_run_hash",
            "convergence_protocol_sha256",
            "convergence_result_sha256",
        },
    )
    matrix = _mapping(
        root["matrix"],
        "matrix",
        {
            "historical_capacities",
            "current_task_examples",
            "full_fit_examples",
            "frozen_full_fit_control",
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
            "macro_peak_learning_rate",
            "lora_peak_learning_rate",
            "warmup_fraction",
            "minimum_learning_rate_ratio",
            "dropout",
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
    return FrontierAdaptationConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        int(experiment["stage"]),
        int(experiment["seed"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["source_macro_config"], project_root),
        _path(paths["source_macro_artifact_root"], project_root),
        _path(paths["source_convergence_artifact_root"], project_root),
        str(source["macro_run_hash"]),
        str(source["macro_protocol_sha256"]),
        str(source["clean_hierarchy_sha256"]),
        str(source["clean_stage31_sha256"]),
        str(source["convergence_run_hash"]),
        str(source["convergence_protocol_sha256"]),
        str(source["convergence_result_sha256"]),
        tuple(int(value) for value in matrix["historical_capacities"]),
        int(matrix["current_task_examples"]),
        int(matrix["full_fit_examples"]),
        bool(matrix["frozen_full_fit_control"]),
        int(optimization["effective_batch_size"]),
        int(optimization["microbatch_size"]),
        int(optimization["epochs"]),
        str(optimization["schedule"]),
        float(optimization["macro_peak_learning_rate"]),
        float(optimization["lora_peak_learning_rate"]),
        float(optimization["warmup_fraction"]),
        float(optimization["minimum_learning_rate_ratio"]),
        float(optimization["dropout"]),
        float(optimization["weight_decay"]),
        float(optimization["gradient_clip_norm"]),
        bool(optimization["activation_recomputation"]),
        int(runtime["evaluation_batch_size"]),
        int(runtime["num_workers"]),
        int(runtime["checkpoint_every_epochs"]),
    )


__all__ = [
    "DEFAULT_FRONTIER_ADAPTATION_CONFIG",
    "FrontierAdaptationConfig",
    "load_frontier_adaptation_config",
]
