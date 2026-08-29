"""Strict protocol configuration for behavioral routing on VAMP-AF contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256
from apm.experiments.vamp_logt_router_config import (
    AdapterConfig,
    BaselineConfig,
    EvaluationConfig,
    PRIMARY_CONDITIONS,
    PhaseConfig,
    RouterConfig,
    RuntimeConfig,
)


@dataclass(frozen=True, slots=True)
class RotatedTaskConfig:
    """Authenticated VAMP-AF task definition and blocked phase schedules."""

    vamp_af_config_sha256: str
    rotations_deg: tuple[float, ...]
    label_shifts: tuple[int, ...]
    interpolation: str
    source_selection_seed: int
    primary_source_examples_per_context: int
    smoke_source_examples_per_context: int
    source_indices_sha256: str
    smoke_context_steps: tuple[int, ...]
    primary_context_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        require_sha256(self.vamp_af_config_sha256, "VAMP-AF configuration")
        require_sha256(self.source_indices_sha256, "VAMP-AF source identities")
        if (
            self.rotations_deg != (0.0, 18.0, 36.0, 54.0, 72.0)
            or self.label_shifts != (0, 2, 4, 6, 8)
            or self.interpolation != "bilinear"
            or self.source_selection_seed != 0
            or self.primary_source_examples_per_context < 1
            or not 0
            < self.smoke_source_examples_per_context
            <= self.primary_source_examples_per_context
            or len(self.smoke_context_steps) != 5
            or len(self.primary_context_steps) != 5
            or any(value < 0 for value in (*self.smoke_context_steps, *self.primary_context_steps))
            or sum(self.smoke_context_steps) < 1
            or sum(self.primary_context_steps) < 1
        ):
            raise ValueError("invalid VAMP-AF Rotated-MNIST task configuration")

    @property
    def domain_count(self) -> int:
        """Return the number of authenticated VAMP-AF contexts."""
        return len(self.rotations_deg)


@dataclass(frozen=True, slots=True)
class RotatedBenchmarkConfig:
    """Fixed block size and 64-step primary horizon."""

    macro_steps: int
    model_batch_size: int
    router_batch_size: int
    evaluation_batch_size: int

    def __post_init__(self) -> None:
        if (
            self.macro_steps < 1
            or min(
                self.model_batch_size,
                self.router_batch_size,
                self.evaluation_batch_size,
            )
            < 1
        ):
            raise ValueError("invalid Rotated-MNIST benchmark configuration")

    @property
    def examples_per_step(self) -> int:
        """Return the complete disjoint training allocation per macro-step."""
        return (
            self.model_batch_size
            + self.router_batch_size
            + self.evaluation_batch_size
        )


@dataclass(frozen=True, slots=True)
class VampLogTRotatedRouterConfig:
    """Complete successor protocol for the VAMP-AF Rotated-MNIST task."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    baseline_run_root: Path
    baseline: BaselineConfig
    task: RotatedTaskConfig
    benchmark: RotatedBenchmarkConfig
    adapter: AdapterConfig
    router: RouterConfig
    smoke: PhaseConfig
    primary: PhaseConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            self.name != "vamp-logt-behavioral-router-rotated-mnist"
            or not self.protocol_revision
            or self.baseline_run_root.name != self.baseline.run_id
            or self.task.domain_count != 5
            or sum(self.task.smoke_context_steps) != self.smoke.macro_steps
            or sum(self.task.primary_context_steps) != self.primary.macro_steps
            or self.benchmark.macro_steps != self.primary.macro_steps
            or self.router.maximum_levels != 7
            or self.smoke.conditions
            != ("no_replay_hard", "example_hard", "range_hard")
            or self.primary.conditions != PRIMARY_CONDITIONS
            or max(self.evaluation.full_checkpoints) > self.primary.macro_steps
        ):
            raise ValueError("resolved Rotated-MNIST router protocol violates its design")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of this complete resolved protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in ("artifact_root", "data_root", "baseline_run_root"):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the Rotated-MNIST router protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> VampLogTRotatedRouterConfig:
    """Load the sole strict YAML surface and reject undeclared choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {
            "experiment",
            "paths",
            "baseline",
            "task",
            "benchmark",
            "adapter",
            "router",
            "phases",
            "evaluation",
            "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    paths = _mapping(root["paths"], "paths", {"artifact_root", "data_root", "baseline_run_root"})
    baseline = _mapping(root["baseline"], "baseline", {"run_id", "checkpoint_sha256"})
    task = _mapping(
        root["task"],
        "task",
        {
            "vamp_af_config_sha256",
            "rotations_deg",
            "label_shifts",
            "interpolation",
            "source_selection_seed",
            "primary_source_examples_per_context",
            "smoke_source_examples_per_context",
            "source_indices_sha256",
            "smoke_context_steps",
            "primary_context_steps",
        },
    )
    benchmark = _mapping(
        root["benchmark"],
        "benchmark",
        {"macro_steps", "model_batch_size", "router_batch_size", "evaluation_batch_size"},
    )
    adapter = _mapping(
        root["adapter"],
        "adapter",
        {
            "epochs",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
        },
    )
    router = _mapping(
        root["router"],
        "router",
        {
            "maximum_levels",
            "hidden_widths",
            "dropout",
            "temperature",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "minibatch_size",
        },
    )
    phases = _mapping(root["phases"], "phases", {"smoke", "primary"})
    evaluation = _mapping(
        root["evaluation"],
        "evaluation",
        {
            "test_subset_per_domain",
            "inference_batch_size",
            "full_checkpoints",
            "near_oracle_thresholds",
        },
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {"device", "deterministic_algorithms", "progress"},
    )
    if (
        str(adapter["optimizer"]).lower() != "adamw"
        or str(router["optimizer"]).lower() != "adamw"
    ):
        raise ValueError("node and router optimizers must both be AdamW")
    project_root = source.parents[2]

    def phase(name: str) -> PhaseConfig:
        row = _mapping(
            phases[name],
            f"phases.{name}",
            {"seeds", "macro_steps", "historical_budget", "router_epochs_per_step", "conditions"},
        )
        return PhaseConfig(
            tuple(int(value) for value in row["seeds"]),
            int(row["macro_steps"]),
            int(row["historical_budget"]),
            int(row["router_epochs_per_step"]),
            tuple(str(value) for value in row["conditions"]),
        )

    return VampLogTRotatedRouterConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["data_root"], project_root),
        _path(paths["baseline_run_root"], project_root),
        BaselineConfig(str(baseline["run_id"]), str(baseline["checkpoint_sha256"])),
        RotatedTaskConfig(
            str(task["vamp_af_config_sha256"]),
            tuple(float(value) for value in task["rotations_deg"]),
            tuple(int(value) for value in task["label_shifts"]),
            str(task["interpolation"]),
            int(task["source_selection_seed"]),
            int(task["primary_source_examples_per_context"]),
            int(task["smoke_source_examples_per_context"]),
            str(task["source_indices_sha256"]),
            tuple(int(value) for value in task["smoke_context_steps"]),
            tuple(int(value) for value in task["primary_context_steps"]),
        ),
        RotatedBenchmarkConfig(
            int(benchmark["macro_steps"]),
            int(benchmark["model_batch_size"]),
            int(benchmark["router_batch_size"]),
            int(benchmark["evaluation_batch_size"]),
        ),
        AdapterConfig(
            int(adapter["epochs"]),
            int(adapter["batch_size"]),
            float(adapter["learning_rate"]),
            float(adapter["weight_decay"]),
            float(adapter["beta1"]),
            float(adapter["beta2"]),
            float(adapter["epsilon"]),
        ),
        RouterConfig(
            int(router["maximum_levels"]),
            tuple(int(value) for value in router["hidden_widths"]),
            float(router["dropout"]),
            float(router["temperature"]),
            float(router["learning_rate"]),
            float(router["weight_decay"]),
            float(router["gradient_clip_norm"]),
            int(router["minibatch_size"]),
        ),
        phase("smoke"),
        phase("primary"),
        EvaluationConfig(
            int(evaluation["test_subset_per_domain"]),
            int(evaluation["inference_batch_size"]),
            tuple(int(value) for value in evaluation["full_checkpoints"]),
            tuple(float(value) for value in evaluation["near_oracle_thresholds"]),
        ),
        RuntimeConfig(
            str(runtime["device"]),
            bool(runtime["deterministic_algorithms"]),
            bool(runtime["progress"]),
        ),
    )


__all__ = [
    "RotatedBenchmarkConfig",
    "RotatedTaskConfig",
    "VampLogTRotatedRouterConfig",
    "load_config",
]
