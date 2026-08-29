"""Strict protocol configuration for integrated LogT behavioral routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


REPLAY_CONDITIONS = (
    "example_hard",
    "range_hard",
    "example_soft",
    "range_soft",
)
PRIMARY_CONDITIONS = ("no_replay_hard", *REPLAY_CONDITIONS)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Pinned frozen classifier dependency."""

    run_id: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.run_id, "baseline run ID")
        require_sha256(self.checkpoint_sha256, "baseline checkpoint")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Fixed Permuted-MNIST stream and disjoint batch allocation."""

    macro_steps: int
    permutation_seeds: tuple[int, ...]
    stream_seed: int
    model_batch_size: int
    router_batch_size: int
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
                self.router_batch_size,
                self.evaluation_batch_size,
            )
            < 1
        ):
            raise ValueError("invalid Permuted-MNIST benchmark configuration")

    @property
    def domain_count(self) -> int:
        """Return identity plus the configured random permutations."""
        return len(self.permutation_seeds) + 1

    @property
    def examples_per_step(self) -> int:
        """Return the complete disjoint training allocation per macro-step."""
        return (
            self.model_batch_size
            + self.router_batch_size
            + self.evaluation_batch_size
        )


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """De-novo active-node classifier training schedule."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.beta1 < 1.0
            or not 0.0 <= self.beta2 < 1.0
            or self.epsilon <= 0.0
        ):
            raise ValueError("invalid node-adapter optimizer configuration")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Level-slot router architecture and optimizer schedule."""

    maximum_levels: int
    hidden_widths: tuple[int, ...]
    dropout: float
    temperature: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minibatch_size: int

    def __post_init__(self) -> None:
        if (
            self.maximum_levels < 1
            or not self.hidden_widths
            or any(width < 1 for width in self.hidden_widths)
            or not 0.0 <= self.dropout < 1.0
            or self.temperature <= 0.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or self.minibatch_size < 2
        ):
            raise ValueError("invalid behavioral-router configuration")


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    """One resumable smoke or primary experimental phase."""

    seeds: tuple[int, ...]
    macro_steps: int
    historical_budget: int
    router_epochs_per_step: int
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or any(seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
            or self.macro_steps < 1
            or self.historical_budget < 1
            or self.router_epochs_per_step < 1
            or not self.conditions
            or not set(self.conditions) <= set(PRIMARY_CONDITIONS)
        ):
            raise ValueError("invalid behavioral-router phase")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation cadence and bounded per-step test allocation."""

    test_subset_per_domain: int
    inference_batch_size: int
    full_checkpoints: tuple[int, ...]
    near_oracle_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.test_subset_per_domain < 1
            or self.inference_batch_size < 1
            or not self.full_checkpoints
            or tuple(sorted(set(self.full_checkpoints))) != self.full_checkpoints
            or not self.near_oracle_thresholds
            or any(value < 0.0 for value in self.near_oracle_thresholds)
        ):
            raise ValueError("invalid router evaluation configuration")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution-only controls included in the run identity."""

    device: str
    deterministic_algorithms: bool
    progress: bool

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("invalid router runtime device")


@dataclass(frozen=True, slots=True)
class VampLogTRouterConfig:
    """Complete resolved protocol for one integrated-router experiment."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    baseline_run_root: Path
    baseline: BaselineConfig
    benchmark: BenchmarkConfig
    adapter: AdapterConfig
    router: RouterConfig
    smoke: PhaseConfig
    primary: PhaseConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            self.name != "vamp-logt-behavioral-router-mnist"
            or not self.protocol_revision
            or self.baseline_run_root.name != self.baseline.run_id
            or self.benchmark.domain_count != 8
            or self.benchmark.macro_steps != self.primary.macro_steps
            or self.router.maximum_levels != 7
            or self.smoke.conditions
            != ("no_replay_hard", "example_hard", "range_hard")
            or self.primary.conditions != PRIMARY_CONDITIONS
            or max(self.evaluation.full_checkpoints) > self.primary.macro_steps
        ):
            raise ValueError("resolved behavioral-router protocol violates its design")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of this resolved protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        for name in ("artifact_root", "data_root", "baseline_run_root"):
            record[name] = str(getattr(self, name))
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the behavioral-router protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> VampLogTRouterConfig:
    """Load the single strict YAML protocol without accepting hidden choices."""
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
    benchmark = _mapping(
        root["benchmark"],
        "benchmark",
        {
            "macro_steps",
            "permutation_seeds",
            "stream_seed",
            "model_batch_size",
            "router_batch_size",
            "evaluation_batch_size",
        },
    )
    adapter = _mapping(
        root["adapter"],
        "adapter",
        {"epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "beta1", "beta2", "epsilon"},
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
        {"test_subset_per_domain", "inference_batch_size", "full_checkpoints", "near_oracle_thresholds"},
    )
    runtime = _mapping(root["runtime"], "runtime", {"device", "deterministic_algorithms", "progress"})
    if str(adapter["optimizer"]).lower() != "adamw" or str(router["optimizer"]).lower() != "adamw":
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

    return VampLogTRouterConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["data_root"], project_root),
        _path(paths["baseline_run_root"], project_root),
        BaselineConfig(str(baseline["run_id"]), str(baseline["checkpoint_sha256"])),
        BenchmarkConfig(
            int(benchmark["macro_steps"]),
            tuple(int(value) for value in benchmark["permutation_seeds"]),
            int(benchmark["stream_seed"]),
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
    "AdapterConfig",
    "BenchmarkConfig",
    "EvaluationConfig",
    "PRIMARY_CONDITIONS",
    "PhaseConfig",
    "REPLAY_CONDITIONS",
    "RouterConfig",
    "VampLogTRouterConfig",
    "load_config",
]
