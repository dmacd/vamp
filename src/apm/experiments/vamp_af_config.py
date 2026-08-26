"""Strict resolved configuration for the VAMP-AF MNIST proof of concept."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256


@dataclass(frozen=True, slots=True)
class BaseTrainingConfig:
    """Frozen CNN convergence-selection settings."""

    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    maximum_epochs: int
    patience: int
    minimum_improvement: float
    validation_examples: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Addressable Rotated MNIST construction settings."""

    rotations_deg: tuple[float, ...]
    label_shifts: tuple[int, ...]
    main_examples_per_context: int
    smoke_examples_per_context: int
    interpolation: str


@dataclass(frozen=True, slots=True)
class AdapterTrainingConfig:
    """Shared top-two-layer adapter optimizer settings."""

    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    batch_size: int


@dataclass(frozen=True, slots=True)
class PreflightConfig:
    """Frozen-router diagnostic and adapter-capacity gate settings."""

    epochs: int
    batch_size: int
    context_accuracy_reference: float
    oracle_accuracy_minimum: float
    joint_oracle_tolerance: float


@dataclass(frozen=True, slots=True)
class StructureConfig:
    """Tree structure and replay initialization settings."""

    leaf_capacity: int
    split_fit_samples: int
    split_epochs: int
    consolidation_epochs: int


@dataclass(frozen=True, slots=True)
class PassConfig:
    """One fixed workflow pass and its allowed conditions."""

    name: str
    seeds: tuple[int, ...]
    examples_per_context: int
    leaf_capacity: int
    conditions: tuple[str, ...]
    depth_cap_override: int | None
    require_consolidation: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution, persistence, and evaluation cadence settings."""

    device: str
    evaluation_interval: int
    deterministic_algorithms: bool
    progress: bool


@dataclass(frozen=True, slots=True)
class VampAFConfig:
    """Complete resolved scientific and runtime protocol."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    base: BaseTrainingConfig
    data: DataConfig
    adapter: AdapterTrainingConfig
    preflight: PreflightConfig
    structure: StructureConfig
    passes: tuple[PassConfig, ...]
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        expected_conditions = {
            "af",
            "frozen_base",
            "global_replay",
            "joint_iid",
            "oracle_context",
        }
        if (
            not self.name
            or not self.protocol_revision
            or self.base.seed < 0
            or self.base.batch_size < 1
            or self.base.learning_rate <= 0.0
            or self.base.weight_decay < 0.0
            or self.base.maximum_epochs < 1
            or self.base.patience < 1
            or self.base.minimum_improvement < 0.0
            or not 0 < self.base.validation_examples < 60_000
            or len(self.data.rotations_deg) != 5
            or len(self.data.label_shifts) != 5
            or self.data.interpolation != "bilinear"
            or self.adapter.batch_size < 1
            or self.preflight.epochs < 1
            or self.preflight.batch_size < 1
            or not 0.0 <= self.preflight.context_accuracy_reference <= 1.0
            or not 0.0 <= self.preflight.oracle_accuracy_minimum <= 1.0
            or self.runtime.device not in {"auto", "cpu", "cuda"}
            or self.runtime.evaluation_interval < 1
            or tuple(row.name for row in self.passes) != ("smoke", "main", "consolidation_stress")
        ):
            raise ValueError("invalid VAMP-AF protocol configuration")
        if any(
            row.examples_per_context < 1
            or row.leaf_capacity < 2
            or not row.seeds
            or not set(row.conditions) <= expected_conditions
            or "af" not in row.conditions
            for row in self.passes
        ):
            raise ValueError("invalid VAMP-AF pass configuration")
        smoke, main, stress = self.passes
        if (
            set(smoke.conditions) != expected_conditions
            or set(main.conditions) != expected_conditions
            or stress.conditions != ("af",)
            or stress.depth_cap_override != 3
            or not stress.require_consolidation
            or len(main.seeds) != 3
        ):
            raise ValueError("VAMP-AF passes differ from the required experiment matrix")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of the fully resolved YAML protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible resolved configuration record."""
        record = asdict(self)
        record["artifact_root"] = str(self.artifact_root)
        record["data_root"] = str(self.data_root)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the resolved VAMP-AF protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> VampAFConfig:
    """Load the single strict YAML surface and reject silent scientific choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(
        raw,
        "configuration",
        {"experiment", "paths", "base", "data", "adapter", "preflight", "structure", "passes", "runtime"},
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    paths = _mapping(root["paths"], "paths", {"artifact_root", "data_root"})
    base = _mapping(
        root["base"],
        "base",
        {
            "seed",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "maximum_epochs",
            "patience",
            "minimum_improvement",
            "validation_examples",
        },
    )
    data = _mapping(
        root["data"],
        "data",
        {
            "rotations_deg",
            "label_shifts",
            "main_examples_per_context",
            "smoke_examples_per_context",
            "interpolation",
        },
    )
    adapter = _mapping(
        root["adapter"],
        "adapter",
        {"learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "batch_size"},
    )
    preflight = _mapping(
        root["preflight"],
        "preflight",
        {"epochs", "batch_size", "context_accuracy_reference", "oracle_accuracy_minimum", "joint_oracle_tolerance"},
    )
    structure = _mapping(
        root["structure"],
        "structure",
        {"leaf_capacity", "split_fit_samples", "split_epochs", "consolidation_epochs"},
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {"device", "evaluation_interval", "deterministic_algorithms", "progress"},
    )
    raw_passes = root["passes"]
    if not isinstance(raw_passes, list):
        raise ValueError("passes must be a list")
    passes = tuple(
        PassConfig(
            name=str(record["name"]),
            seeds=tuple(int(seed) for seed in record["seeds"]),
            examples_per_context=int(record["examples_per_context"]),
            leaf_capacity=int(record["leaf_capacity"]),
            conditions=tuple(str(condition) for condition in record["conditions"]),
            depth_cap_override=(
                None if record["depth_cap_override"] is None else int(record["depth_cap_override"])
            ),
            require_consolidation=bool(record["require_consolidation"]),
        )
        for raw_record in raw_passes
        for record in (
            _mapping(
                raw_record,
                "pass",
                {
                    "name",
                    "seeds",
                    "examples_per_context",
                    "leaf_capacity",
                    "conditions",
                    "depth_cap_override",
                    "require_consolidation",
                },
            ),
        )
    )
    project_root = source.parents[2]
    return VampAFConfig(
        name=str(experiment["name"]),
        protocol_revision=str(experiment["protocol_revision"]),
        artifact_root=_path(paths["artifact_root"], project_root),
        data_root=_path(paths["data_root"], project_root),
        base=BaseTrainingConfig(
            seed=int(base["seed"]),
            batch_size=int(base["batch_size"]),
            learning_rate=float(base["learning_rate"]),
            weight_decay=float(base["weight_decay"]),
            maximum_epochs=int(base["maximum_epochs"]),
            patience=int(base["patience"]),
            minimum_improvement=float(base["minimum_improvement"]),
            validation_examples=int(base["validation_examples"]),
        ),
        data=DataConfig(
            rotations_deg=tuple(float(value) for value in data["rotations_deg"]),
            label_shifts=tuple(int(value) for value in data["label_shifts"]),
            main_examples_per_context=int(data["main_examples_per_context"]),
            smoke_examples_per_context=int(data["smoke_examples_per_context"]),
            interpolation=str(data["interpolation"]),
        ),
        adapter=AdapterTrainingConfig(
            learning_rate=float(adapter["learning_rate"]),
            weight_decay=float(adapter["weight_decay"]),
            beta1=float(adapter["beta1"]),
            beta2=float(adapter["beta2"]),
            epsilon=float(adapter["epsilon"]),
            batch_size=int(adapter["batch_size"]),
        ),
        preflight=PreflightConfig(
            epochs=int(preflight["epochs"]),
            batch_size=int(preflight["batch_size"]),
            context_accuracy_reference=float(preflight["context_accuracy_reference"]),
            oracle_accuracy_minimum=float(preflight["oracle_accuracy_minimum"]),
            joint_oracle_tolerance=float(preflight["joint_oracle_tolerance"]),
        ),
        structure=StructureConfig(
            leaf_capacity=int(structure["leaf_capacity"]),
            split_fit_samples=int(structure["split_fit_samples"]),
            split_epochs=int(structure["split_epochs"]),
            consolidation_epochs=int(structure["consolidation_epochs"]),
        ),
        passes=passes,
        runtime=RuntimeConfig(
            device=str(runtime["device"]),
            evaluation_interval=int(runtime["evaluation_interval"]),
            deterministic_algorithms=bool(runtime["deterministic_algorithms"]),
            progress=bool(runtime["progress"]),
        ),
    )


__all__ = [
    "BaseTrainingConfig",
    "DataConfig",
    "AdapterTrainingConfig",
    "PassConfig",
    "PreflightConfig",
    "RuntimeConfig",
    "StructureConfig",
    "VampAFConfig",
    "load_config",
]
