"""Strict protocol configuration for direct LogT prediction integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256
from apm.experiments.vamp_logt_router_config import (
    AdapterConfig,
    BaselineConfig,
    RuntimeConfig,
)


INTEGRATOR_CONDITIONS = (
    "integrator_no_replay",
    "integrator_example_replay",
    "integrator_range_replay",
    "base_example_replay",
)


@dataclass(frozen=True, slots=True)
class ParentRouterConfig:
    """Authenticated completed router result used for paired interpretation."""

    run_id: str
    protocol_sha256: str
    summary_sha256: str
    implementation_commit: str

    def __post_init__(self) -> None:
        require_sha256(self.run_id, "parent router run")
        require_sha256(self.protocol_sha256, "parent router protocol")
        require_sha256(self.summary_sha256, "parent router summary")
        if len(self.implementation_commit) < 7:
            raise ValueError("parent router commit must be an identifiable Git revision")


@dataclass(frozen=True, slots=True)
class IntegratorBenchmarkConfig:
    """Fixed Permuted-MNIST allocation and primary horizon."""

    macro_steps: int
    permutation_seeds: tuple[int, ...]
    stream_seed: int
    model_batch_size: int
    integrator_batch_size: int
    evaluation_batch_size: int

    def __post_init__(self) -> None:
        if (
            self.macro_steps < 1
            or len(self.permutation_seeds) != 7
            or len(set(self.permutation_seeds)) != 7
            or any(seed < 0 for seed in self.permutation_seeds)
            or self.stream_seed < 0
            or min(
                self.model_batch_size,
                self.integrator_batch_size,
                self.evaluation_batch_size,
            )
            < 1
        ):
            raise ValueError("invalid prediction-integrator benchmark")

    @property
    def domain_count(self) -> int:
        """Return the identity domain plus seven fixed permutations."""
        return len(self.permutation_seeds) + 1

    @property
    def router_batch_size(self) -> int:
        """Expose the semantically equivalent allocation used by shared data code."""
        return self.integrator_batch_size

    @property
    def examples_per_step(self) -> int:
        """Return the complete disjoint allocation per macro-step."""
        return (
            self.model_batch_size
            + self.integrator_batch_size
            + self.evaluation_batch_size
        )


@dataclass(frozen=True, slots=True)
class IntegratorConfig:
    """Fixed residual MLP, optimizer, and source-balancing choices."""

    maximum_levels: int
    hidden_widths: tuple[int, ...]
    dropout: float
    residual_baseline: str
    future_slot_initialization: str
    residual_output_initialization: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minibatch_size: int
    current_source_weight: float
    offline_epochs: int

    def __post_init__(self) -> None:
        if (
            self.maximum_levels != 7
            or self.hidden_widths != (1024, 512, 256)
            or not 0.0 <= self.dropout < 1.0
            or self.residual_baseline != "mean_probability"
            or self.future_slot_initialization != "zero"
            or self.residual_output_initialization != "zero"
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.minibatch_size < 2
            or self.current_source_weight != 0.5
            or self.offline_epochs != 4
        ):
            raise ValueError("integrator settings differ from the frozen protocol")


@dataclass(frozen=True, slots=True)
class IntegratorPhaseConfig:
    """One resumable smoke or primary phase."""

    seeds: tuple[int, ...]
    macro_steps: int
    historical_budget: int
    integrator_epochs_per_step: int
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or any(seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
            or self.macro_steps < 1
            or self.historical_budget < 1
            or self.integrator_epochs_per_step < 1
            or self.conditions != INTEGRATOR_CONDITIONS
        ):
            raise ValueError("invalid prediction-integrator phase")


@dataclass(frozen=True, slots=True)
class IntegratorEvaluationConfig:
    """Fixed evaluation subsets and full-checkpoint cadence."""

    test_subset_per_domain: int
    inference_batch_size: int
    full_checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.test_subset_per_domain < 1
            or self.inference_batch_size < 1
            or tuple(sorted(set(self.full_checkpoints))) != self.full_checkpoints
            or not self.full_checkpoints
        ):
            raise ValueError("invalid prediction-integrator evaluation")


@dataclass(frozen=True, slots=True)
class VampLogTIntegratorConfig:
    """Complete direct-prediction protocol on Permuted-MNIST."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    baseline_run_root: Path
    parent_router_run_root: Path
    baseline: BaselineConfig
    parent_router: ParentRouterConfig
    benchmark: IntegratorBenchmarkConfig
    adapter: AdapterConfig
    integrator: IntegratorConfig
    smoke: IntegratorPhaseConfig
    primary: IntegratorPhaseConfig
    evaluation: IntegratorEvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            self.name != "vamp-logt-prediction-integrator-permuted-mnist"
            or self.protocol_revision != "integrated-prediction-permuted-v1"
            or self.baseline_run_root.name != self.baseline.run_id
            or self.parent_router_run_root.name != self.parent_router.run_id
            or self.benchmark.domain_count != 8
            or self.smoke.macro_steps > self.benchmark.macro_steps
            or self.benchmark.macro_steps != self.primary.macro_steps
            or max(self.evaluation.full_checkpoints) > self.primary.macro_steps
        ):
            raise ValueError("resolved prediction-integrator protocol violates its plan")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of the complete resolved protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in (
            "artifact_root",
            "data_root",
            "baseline_run_root",
            "parent_router_run_root",
        ):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the prediction-integrator protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> VampLogTIntegratorConfig:
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
            "parent_router",
            "benchmark",
            "adapter",
            "integrator",
            "phases",
            "evaluation",
            "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "data_root", "baseline_run_root", "parent_router_run_root"},
    )
    baseline = _mapping(root["baseline"], "baseline", {"run_id", "checkpoint_sha256"})
    parent = _mapping(
        root["parent_router"],
        "parent_router",
        {"run_id", "protocol_sha256", "summary_sha256", "implementation_commit"},
    )
    benchmark = _mapping(
        root["benchmark"],
        "benchmark",
        {
            "macro_steps",
            "permutation_seeds",
            "stream_seed",
            "model_batch_size",
            "integrator_batch_size",
            "evaluation_batch_size",
        },
    )
    adapter = _mapping(
        root["adapter"],
        "adapter",
        {
            "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
            "beta1", "beta2", "epsilon",
        },
    )
    integrator = _mapping(
        root["integrator"],
        "integrator",
        {
            "maximum_levels", "hidden_widths", "dropout", "residual_baseline",
            "future_slot_initialization", "residual_output_initialization", "optimizer",
            "learning_rate", "weight_decay", "gradient_clip_norm", "minibatch_size",
            "current_source_weight", "offline_epochs",
        },
    )
    phases = _mapping(root["phases"], "phases", {"smoke", "primary"})
    evaluation = _mapping(
        root["evaluation"],
        "evaluation",
        {"test_subset_per_domain", "inference_batch_size", "full_checkpoints"},
    )
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {"device", "deterministic_algorithms", "progress"},
    )
    if (
        str(adapter["optimizer"]).lower() != "adamw"
        or str(integrator["optimizer"]).lower() != "adamw"
    ):
        raise ValueError("adapter and integrator optimizers must be AdamW")
    project_root = source.parents[2]

    def phase(name: str) -> IntegratorPhaseConfig:
        row = _mapping(
            phases[name],
            f"phases.{name}",
            {
                "seeds",
                "macro_steps",
                "historical_budget",
                "integrator_epochs_per_step",
                "conditions",
            },
        )
        return IntegratorPhaseConfig(
            tuple(int(value) for value in row["seeds"]),
            int(row["macro_steps"]),
            int(row["historical_budget"]),
            int(row["integrator_epochs_per_step"]),
            tuple(str(value) for value in row["conditions"]),
        )

    return VampLogTIntegratorConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["data_root"], project_root),
        _path(paths["baseline_run_root"], project_root),
        _path(paths["parent_router_run_root"], project_root),
        BaselineConfig(str(baseline["run_id"]), str(baseline["checkpoint_sha256"])),
        ParentRouterConfig(
            str(parent["run_id"]),
            str(parent["protocol_sha256"]),
            str(parent["summary_sha256"]),
            str(parent["implementation_commit"]),
        ),
        IntegratorBenchmarkConfig(
            int(benchmark["macro_steps"]),
            tuple(int(value) for value in benchmark["permutation_seeds"]),
            int(benchmark["stream_seed"]),
            int(benchmark["model_batch_size"]),
            int(benchmark["integrator_batch_size"]),
            int(benchmark["evaluation_batch_size"]),
        ),
        AdapterConfig(
            int(adapter["epochs"]), int(adapter["batch_size"]),
            float(adapter["learning_rate"]), float(adapter["weight_decay"]),
            float(adapter["beta1"]), float(adapter["beta2"]), float(adapter["epsilon"]),
        ),
        IntegratorConfig(
            int(integrator["maximum_levels"]),
            tuple(int(value) for value in integrator["hidden_widths"]),
            float(integrator["dropout"]),
            str(integrator["residual_baseline"]),
            str(integrator["future_slot_initialization"]),
            str(integrator["residual_output_initialization"]),
            float(integrator["learning_rate"]),
            float(integrator["weight_decay"]),
            float(integrator["gradient_clip_norm"]),
            int(integrator["minibatch_size"]),
            float(integrator["current_source_weight"]),
            int(integrator["offline_epochs"]),
        ),
        phase("smoke"),
        phase("primary"),
        IntegratorEvaluationConfig(
            int(evaluation["test_subset_per_domain"]),
            int(evaluation["inference_batch_size"]),
            tuple(int(value) for value in evaluation["full_checkpoints"]),
        ),
        RuntimeConfig(
            str(runtime["device"]),
            bool(runtime["deterministic_algorithms"]),
            bool(runtime["progress"]),
        ),
    )


__all__ = [
    "INTEGRATOR_CONDITIONS",
    "IntegratorConfig",
    "IntegratorEvaluationConfig",
    "IntegratorPhaseConfig",
    "VampLogTIntegratorConfig",
    "load_config",
]
