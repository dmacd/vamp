"""Strict protocol configuration for the dense Permuted-MNIST LogT study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256
from apm.continual.dense_mlp_adapter import (
    DenseConvergenceConfig,
    DenseOptimizerConfig,
)


ROUTER_CONDITIONS = (
    "router_current_hard",
    "router_uniform_hard",
    "router_range_hard",
    "router_uniform_soft",
    "router_range_soft",
)

INTEGRATOR_CONDITIONS = (
    "integrator_current_only",
    "integrator_uniform_replay",
    "integrator_range_replay",
    "integrator_base_uniform_replay",
)


@dataclass(frozen=True, slots=True)
class DenseBenchmarkConfig:
    """Fixed eight-domain stream and disjoint per-step allocations."""

    macro_steps: int
    permutation_seeds: tuple[int, ...]
    stream_seed: int
    model_batch_size: int
    observer_batch_size: int
    evaluation_batch_size: int

    def __post_init__(self) -> None:
        if (
            self.macro_steps < 1
            or len(self.permutation_seeds) != 7
            or len(set(self.permutation_seeds)) != 7
            or min(self.model_batch_size, self.observer_batch_size, self.evaluation_batch_size) < 1
        ):
            raise ValueError("invalid dense Permuted-MNIST benchmark")

    @property
    def domain_count(self) -> int:
        """Return identity plus seven fixed permutations."""
        return len(self.permutation_seeds) + 1

    @property
    def router_batch_size(self) -> int:
        """Expose observer allocation under the shared benchmark API."""
        return self.observer_batch_size

    @property
    def examples_per_step(self) -> int:
        """Return the complete disjoint allocation per macro-step."""
        return self.model_batch_size + self.observer_batch_size + self.evaluation_batch_size


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Architecture sweep, source split, and eligibility rule."""

    candidate_widths: tuple[tuple[int, int, int], ...]
    seeds: tuple[int, ...]
    split_seed: int
    training_source_examples: int
    validation_source_examples: int
    dropout: float
    optimizer: DenseOptimizerConfig
    convergence: DenseConvergenceConfig
    selection_policy: str
    identity_mean_accuracy_minimum: float
    identity_seed_zero_accuracy_minimum: float
    pooled_gap_from_widest_maximum: float

    def __post_init__(self) -> None:
        if (
            not self.candidate_widths
            or any(len(widths) != 3 or any(width < 1 for width in widths) for widths in self.candidate_widths)
            or tuple(sorted(self.candidate_widths, key=_parameter_count)) != self.candidate_widths
            or not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or 0 not in self.seeds
            or self.training_source_examples + self.validation_source_examples != 60_000
            or not 0.0 <= self.dropout < 1.0
            or self.selection_policy not in {"eligibility_gated", "smallest_candidate"}
            or not 0.0 < self.identity_mean_accuracy_minimum <= 1.0
            or not 0.0 < self.identity_seed_zero_accuracy_minimum <= 1.0
            or not 0.0 <= self.pooled_gap_from_widest_maximum < 1.0
        ):
            raise ValueError("invalid dense calibration protocol")


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """De-novo full-model node training schedule."""

    epochs: int
    dropout: float
    optimizer: DenseOptimizerConfig

    def __post_init__(self) -> None:
        if self.epochs < 1 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid dense node configuration")


@dataclass(frozen=True, slots=True)
class ObserverConfig:
    """Stable slot geometry and inference batching."""

    maximum_levels: int
    inference_batch_size: int
    hidden_normalization: str

    def __post_init__(self) -> None:
        if (
            self.maximum_levels < 1
            or self.inference_batch_size < 1
            or self.hidden_normalization != "per_example_layer_norm"
        ):
            raise ValueError("invalid dense observer configuration")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Matched behavioral-router architecture and training schedule."""

    maximum_levels: int
    hidden_widths: tuple[int, int, int]
    dropout: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minibatch_size: int
    epochs_per_step: int
    target_temperature: float
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.hidden_widths) != 3
            or any(width < 1 for width in self.hidden_widths)
            or not 0.0 <= self.dropout < 1.0
            or min(self.learning_rate, self.gradient_clip_norm, self.target_temperature) <= 0.0
            or self.weight_decay < 0.0
            or min(self.minibatch_size, self.epochs_per_step) < 1
            or self.conditions != ROUTER_CONDITIONS
        ):
            raise ValueError("invalid matched router configuration")


@dataclass(frozen=True, slots=True)
class IntegratorConfig:
    """Persistent and fresh residual-integrator settings."""

    maximum_levels: int
    hidden_widths: tuple[int, int, int]
    dropout: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    minibatch_size: int
    current_source_weight: float
    epochs_per_step: int
    offline_epochs: int
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.hidden_widths) != 3
            or any(width < 1 for width in self.hidden_widths)
            or not 0.0 <= self.dropout < 1.0
            or min(self.learning_rate, self.gradient_clip_norm) <= 0.0
            or self.weight_decay < 0.0
            or min(self.minibatch_size, self.epochs_per_step, self.offline_epochs) < 1
            or self.current_source_weight != 0.5
            or self.conditions != INTEGRATOR_CONDITIONS
        ):
            raise ValueError("invalid dense integrator configuration")


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    """Seeds and fixed replay budget for paired online conditions."""

    seeds: tuple[int, ...]
    historical_budget: int

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or self.historical_budget < 1:
            raise ValueError("invalid dense online phase")


@dataclass(frozen=True, slots=True)
class CeilingConfig:
    """Fresh full-replay integrator restarts and convergence rule."""

    restarts_per_step: int
    convergence: DenseConvergenceConfig

    def __post_init__(self) -> None:
        if self.restarts_per_step < 1:
            raise ValueError("ceiling requires at least one fresh restart")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Fixed test views and headline checkpoint set."""

    test_subset_per_domain: int
    full_checkpoints: tuple[int, ...]
    headline_checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.test_subset_per_domain < 1
            or not self.full_checkpoints
            or tuple(sorted(set(self.full_checkpoints))) != self.full_checkpoints
            or tuple(sorted(set(self.headline_checkpoints))) != self.headline_checkpoints
            or not set(self.headline_checkpoints).issubset(self.full_checkpoints)
        ):
            raise ValueError("invalid dense evaluation protocol")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution-only device and deterministic-progress controls."""

    device: str
    deterministic_algorithms: bool
    progress: bool

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime device must be auto, cpu, or cuda")


@dataclass(frozen=True, slots=True)
class VampLogTDenseConfig:
    """Complete dense-base router/integrator/ceiling experiment."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    calibration_evidence_run: Path | None
    benchmark: DenseBenchmarkConfig
    calibration: CalibrationConfig
    node: NodeConfig
    observer: ObserverConfig
    router: RouterConfig
    integrator: IntegratorConfig
    online: OnlineConfig
    ceiling: CeilingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            self.name != "vamp-logt-dense-permuted-mnist"
            or self.protocol_revision not in {
                "dense-full-model-v1",
                "dense-full-model-v2-posthoc-ungated",
            }
            or self.benchmark.domain_count != 8
            or self.observer.maximum_levels != self.router.maximum_levels
            or self.observer.maximum_levels != self.integrator.maximum_levels
            or max(self.evaluation.full_checkpoints) > self.benchmark.macro_steps
        ):
            raise ValueError("resolved dense protocol violates the frozen plan")
        if self.protocol_revision == "dense-full-model-v1":
            if self.calibration_evidence_run is not None:
                raise ValueError("the original dense protocol cannot import calibration evidence")
        elif (
            self.calibration_evidence_run is None
            or self.calibration.selection_policy != "smallest_candidate"
        ):
            raise ValueError("the post-hoc dense successor requires ungated evidence import")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of the resolved protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible protocol record."""
        record = asdict(self)
        record["artifact_root"] = str(self.artifact_root)
        record["data_root"] = str(self.data_root)
        if self.protocol_revision == "dense-full-model-v1":
            record.pop("calibration_evidence_run")
            record["calibration"].pop("selection_policy")
        else:
            record["calibration_evidence_run"] = str(self.calibration_evidence_run)
        return record


def _parameter_count(widths: tuple[int, int, int]) -> int:
    first, second, third = widths
    return (784 + 1) * first + (first + 1) * second + (second + 1) * third + (third + 1) * 10


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the dense protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _optimizer(row: Mapping[str, object], label: str) -> DenseOptimizerConfig:
    values = _mapping(
        row,
        label,
        {"optimizer", "learning_rate", "weight_decay", "batch_size", "gradient_clip_norm"},
    )
    if str(values["optimizer"]).lower() != "adamw":
        raise ValueError(f"{label} optimizer must be AdamW")
    return DenseOptimizerConfig(
        float(values["learning_rate"]),
        float(values["weight_decay"]),
        int(values["batch_size"]),
        float(values["gradient_clip_norm"]),
    )


def _convergence(row: Mapping[str, object], label: str) -> DenseConvergenceConfig:
    values = _mapping(
        row,
        label,
        {
            "minimum_epochs", "maximum_epochs", "improvement_delta",
            "learning_rate_patience", "learning_rate_factor",
            "minimum_learning_rate", "convergence_patience",
        },
    )
    return DenseConvergenceConfig(
        int(values["minimum_epochs"]),
        int(values["maximum_epochs"]),
        float(values["improvement_delta"]),
        int(values["learning_rate_patience"]),
        float(values["learning_rate_factor"]),
        float(values["minimum_learning_rate"]),
        int(values["convergence_patience"]),
    )


def load_config(path: str | Path) -> VampLogTDenseConfig:
    """Load the one strict YAML surface and reject undeclared choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {
            "experiment", "paths", "benchmark", "calibration", "node", "observer",
            "router", "integrator", "online", "ceiling", "evaluation", "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    revision = str(experiment["protocol_revision"])
    path_keys = {"artifact_root", "data_root"}
    if revision == "dense-full-model-v2-posthoc-ungated":
        path_keys.add("calibration_evidence_run")
    paths = _mapping(root["paths"], "paths", path_keys)
    benchmark = _mapping(
        root["benchmark"],
        "benchmark",
        {
            "macro_steps", "permutation_seeds", "stream_seed", "model_batch_size",
            "observer_batch_size", "evaluation_batch_size",
        },
    )
    calibration_keys = {
        "candidate_widths", "seeds", "split_seed", "training_source_examples",
        "validation_source_examples", "dropout", "optimizer", "convergence",
        "identity_mean_accuracy_minimum", "identity_seed_zero_accuracy_minimum",
        "pooled_gap_from_widest_maximum",
    }
    if revision == "dense-full-model-v2-posthoc-ungated":
        calibration_keys.add("selection_policy")
    calibration = _mapping(
        root["calibration"],
        "calibration",
        calibration_keys,
    )
    node = _mapping(root["node"], "node", {"epochs", "dropout", "optimizer"})
    observer = _mapping(
        root["observer"],
        "observer",
        {"maximum_levels", "inference_batch_size", "hidden_normalization"},
    )
    router = _mapping(
        root["router"],
        "router",
        {
            "maximum_levels", "hidden_widths", "dropout", "optimizer", "learning_rate",
            "weight_decay", "gradient_clip_norm", "minibatch_size", "epochs_per_step",
            "target_temperature", "conditions",
        },
    )
    integrator = _mapping(
        root["integrator"],
        "integrator",
        {
            "maximum_levels", "hidden_widths", "dropout", "optimizer", "learning_rate",
            "weight_decay", "gradient_clip_norm", "minibatch_size", "current_source_weight",
            "epochs_per_step", "offline_epochs", "conditions",
        },
    )
    online = _mapping(root["online"], "online", {"seeds", "historical_budget"})
    ceiling = _mapping(root["ceiling"], "ceiling", {"restarts_per_step", "convergence"})
    evaluation = _mapping(
        root["evaluation"],
        "evaluation",
        {"test_subset_per_domain", "full_checkpoints", "headline_checkpoints"},
    )
    runtime = _mapping(
        root["runtime"], "runtime", {"device", "deterministic_algorithms", "progress"}
    )
    if str(router["optimizer"]).lower() != "adamw" or str(integrator["optimizer"]).lower() != "adamw":
        raise ValueError("router and integrator optimizers must be AdamW")
    project_root = source.parents[2]
    return VampLogTDenseConfig(
        str(experiment["name"]),
        revision,
        _path(paths["artifact_root"], project_root),
        _path(paths["data_root"], project_root),
        (
            _path(paths["calibration_evidence_run"], project_root)
            if "calibration_evidence_run" in paths
            else None
        ),
        DenseBenchmarkConfig(
            int(benchmark["macro_steps"]),
            tuple(int(value) for value in benchmark["permutation_seeds"]),
            int(benchmark["stream_seed"]),
            int(benchmark["model_batch_size"]),
            int(benchmark["observer_batch_size"]),
            int(benchmark["evaluation_batch_size"]),
        ),
        CalibrationConfig(
            tuple(tuple(int(width) for width in widths) for widths in calibration["candidate_widths"]),
            tuple(int(value) for value in calibration["seeds"]),
            int(calibration["split_seed"]),
            int(calibration["training_source_examples"]),
            int(calibration["validation_source_examples"]),
            float(calibration["dropout"]),
            _optimizer(calibration["optimizer"], "calibration.optimizer"),
            _convergence(calibration["convergence"], "calibration.convergence"),
            str(calibration.get("selection_policy", "eligibility_gated")),
            float(calibration["identity_mean_accuracy_minimum"]),
            float(calibration["identity_seed_zero_accuracy_minimum"]),
            float(calibration["pooled_gap_from_widest_maximum"]),
        ),
        NodeConfig(
            int(node["epochs"]),
            float(node["dropout"]),
            _optimizer(node["optimizer"], "node.optimizer"),
        ),
        ObserverConfig(
            int(observer["maximum_levels"]),
            int(observer["inference_batch_size"]),
            str(observer["hidden_normalization"]),
        ),
        RouterConfig(
            int(router["maximum_levels"]),
            tuple(int(value) for value in router["hidden_widths"]),
            float(router["dropout"]),
            float(router["learning_rate"]),
            float(router["weight_decay"]),
            float(router["gradient_clip_norm"]),
            int(router["minibatch_size"]),
            int(router["epochs_per_step"]),
            float(router["target_temperature"]),
            tuple(str(value) for value in router["conditions"]),
        ),
        IntegratorConfig(
            int(integrator["maximum_levels"]),
            tuple(int(value) for value in integrator["hidden_widths"]),
            float(integrator["dropout"]),
            float(integrator["learning_rate"]),
            float(integrator["weight_decay"]),
            float(integrator["gradient_clip_norm"]),
            int(integrator["minibatch_size"]),
            float(integrator["current_source_weight"]),
            int(integrator["epochs_per_step"]),
            int(integrator["offline_epochs"]),
            tuple(str(value) for value in integrator["conditions"]),
        ),
        OnlineConfig(
            tuple(int(value) for value in online["seeds"]),
            int(online["historical_budget"]),
        ),
        CeilingConfig(
            int(ceiling["restarts_per_step"]),
            _convergence(ceiling["convergence"], "ceiling.convergence"),
        ),
        EvaluationConfig(
            int(evaluation["test_subset_per_domain"]),
            tuple(int(value) for value in evaluation["full_checkpoints"]),
            tuple(int(value) for value in evaluation["headline_checkpoints"]),
        ),
        RuntimeConfig(
            str(runtime["device"]),
            bool(runtime["deterministic_algorithms"]),
            bool(runtime["progress"]),
        ),
    )


__all__ = [
    "CalibrationConfig",
    "CeilingConfig",
    "DenseBenchmarkConfig",
    "EvaluationConfig",
    "INTEGRATOR_CONDITIONS",
    "IntegratorConfig",
    "NodeConfig",
    "ObserverConfig",
    "OnlineConfig",
    "ROUTER_CONDITIONS",
    "RouterConfig",
    "RuntimeConfig",
    "VampLogTDenseConfig",
    "load_config",
]
