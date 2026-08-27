"""Strict resolved protocol for NCE/TRE evidence routing on MNIST."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Authenticated VAMP-AF dependency and the recorded comparison values."""

    run_id: str
    base_checkpoint_sha256: str
    feature_cache_sha256: str
    expected_main_mean_accuracy: tuple[tuple[str, float], ...]
    expected_main_mean_oracle_leaf_accuracy: float

    def __post_init__(self) -> None:
        require_sha256(self.run_id, "baseline run ID")
        require_sha256(self.base_checkpoint_sha256, "baseline checkpoint")
        require_sha256(self.feature_cache_sha256, "baseline feature cache")
        required = {"af", "frozen_base", "global_replay", "joint_iid", "oracle_context"}
        if (
            {name for name, _value in self.expected_main_mean_accuracy} != required
            or any(not 0.0 <= value <= 1.0 for _name, value in self.expected_main_mean_accuracy)
            or not 0.0 <= self.expected_main_mean_oracle_leaf_accuracy <= 1.0
        ):
            raise ValueError("invalid authenticated VAMP-AF comparison values")

    @property
    def main_mean_accuracy(self) -> dict[str, float]:
        """Return the pinned condition means as a detached mapping."""
        return dict(self.expected_main_mean_accuracy)


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Fixed blocked five-context MNIST stream and temporal snapshots."""

    rotations_deg: tuple[float, ...]
    label_shifts: tuple[int, ...]
    interpolation: str
    examples_per_context: int
    block_size: int
    static_snapshot_blocks: int
    consolidation_snapshot_blocks: int

    def __post_init__(self) -> None:
        total_examples = self.examples_per_context * len(self.rotations_deg)
        if (
            len(self.rotations_deg) != 5
            or len(self.label_shifts) != 5
            or self.interpolation != "bilinear"
            or self.examples_per_context != 10_000
            or self.block_size < 2
            or total_examples % self.block_size
            or not 1 <= self.static_snapshot_blocks < self.consolidation_snapshot_blocks
            or self.consolidation_snapshot_blocks != self.static_snapshot_blocks + 1
            or self.consolidation_snapshot_blocks > total_examples // self.block_size
        ):
            raise ValueError("invalid fixed MNIST LogT stream")

    @property
    def total_blocks(self) -> int:
        """Return the number of fixed blocks in the complete stream."""
        return self.examples_per_context * len(self.rotations_deg) // self.block_size


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """De-novo top-two-layer adapter replay schedule for every node."""

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
            raise ValueError("invalid adapter replay schedule")


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    """Shared NCE/TRE architecture, corruption, optimizer, and replica choices."""

    reference: str
    direct_bridges: int
    candidate_tre_bridges: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    initial_replacement_probability: float
    independent_replicas: int
    score_batch_size: int

    def __post_init__(self) -> None:
        if (
            self.reference
            not in {
                "discrete_uniform_uint8",
                "frozen_base_training_images_uint8",
            }
            or self.direct_bridges != 1
            or not self.candidate_tre_bridges
            or tuple(sorted(set(self.candidate_tre_bridges))) != self.candidate_tre_bridges
            or any(bridges <= 1 for bridges in self.candidate_tre_bridges)
            or self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.beta1 < 1.0
            or not 0.0 <= self.beta2 < 1.0
            or self.epsilon <= 0.0
            or self.initial_replacement_probability != 1.0 / 784.0
            or self.independent_replicas < 2
            or self.score_batch_size < 1
        ):
            raise ValueError("invalid fixed evidence-estimation protocol")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Known normalized Bernoulli-mixture calibration problem and gates."""

    dimensions: int
    component_probabilities: tuple[float, float]
    reference_probability: float
    tre_bridges: int
    training_steps: int
    batch_size: int
    evaluation_examples: int
    replicas: int
    learning_rate: float
    weight_decay: float
    signed_bias_max_nats: float
    tre_rmse_max_nats: float
    interseed_rmse_max_nats: float
    direct_to_tre_rmse_ratio_min: float

    def __post_init__(self) -> None:
        low, high = self.component_probabilities
        if (
            self.dimensions < 2
            or not 0.0 < low < self.reference_probability < high < 1.0
            or abs((low + high) / 2.0 - self.reference_probability) > 1.0e-12
            or self.tre_bridges < 2
            or self.training_steps < 1
            or self.batch_size < 2
            or self.evaluation_examples < self.batch_size
            or self.replicas < 2
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.signed_bias_max_nats <= 0.0
            or self.tre_rmse_max_nats <= 0.0
            or self.interseed_rmse_max_nats <= 0.0
            or self.direct_to_tre_rmse_ratio_min <= 1.0
        ):
            raise ValueError("invalid normalized ratio-calibration protocol")


@dataclass(frozen=True, slots=True)
class StaticGateConfig:
    """Static-snapshot data allocation and mandatory TRE selection gates."""

    stream_seeds: tuple[int, ...]
    heldout_examples_per_node: int
    adjacent_balanced_accuracy_max: float
    independent_route_agreement_min: float
    classifier_oracle_gap_max: float

    def __post_init__(self) -> None:
        if (
            self.stream_seeds != (0, 1, 2)
            or self.heldout_examples_per_node < 10
            or not 0.5 < self.adjacent_balanced_accuracy_max < 1.0
            or not 0.0 < self.independent_route_agreement_min <= 1.0
            or not 0.0 < self.classifier_oracle_gap_max < 1.0
        ):
            raise ValueError("invalid static evidence-routing gates")


@dataclass(frozen=True, slots=True)
class ConsolidationGateConfig:
    """De-novo merge-control allocation and score-stability gates."""

    heldout_examples_per_merge: int
    raw_score_difference_max_nats: float
    route_agreement_min: float
    classifier_accuracy_gap_max: float
    nce_loss_relative_difference_max: float
    level_offset_slope_max_nats: float

    def __post_init__(self) -> None:
        if (
            self.heldout_examples_per_merge < 10
            or self.raw_score_difference_max_nats <= 0.0
            or not 0.0 < self.route_agreement_min <= 1.0
            or not 0.0 < self.classifier_accuracy_gap_max < 1.0
            or not 0.0 < self.nce_loss_relative_difference_max < 1.0
            or self.level_offset_slope_max_nats <= 0.0
        ):
            raise ValueError("invalid consolidation-stability gates")


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    """Full-stream seeds and evaluation cadence."""

    stream_seeds: tuple[int, ...]
    evaluation_batch_size: int

    def __post_init__(self) -> None:
        if self.stream_seeds != (0, 1, 2) or self.evaluation_batch_size < 1:
            raise ValueError("invalid full-stream comparison protocol")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution-only choices that remain part of the resolved run identity."""

    device: str
    deterministic_algorithms: bool
    progress: bool

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("invalid evidence-routing runtime device")


@dataclass(frozen=True, slots=True)
class VampLogTEvidenceConfig:
    """Complete frozen NCE/TRE LogT evidence-routing experiment."""

    name: str
    protocol_revision: str
    artifact_root: Path
    data_root: Path
    baseline_run_root: Path
    baseline: BaselineConfig
    stream: StreamConfig
    adapter: AdapterConfig
    evidence: EvidenceConfig
    calibration: CalibrationConfig
    static: StaticGateConfig
    consolidation: ConsolidationGateConfig
    online: OnlineConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            self.name != "vamp-logt-nce-tre-mnist"
            or not self.protocol_revision
            or self.baseline_run_root.name != self.baseline.run_id
            or self.stream.total_blocks != 100
            or self.stream.static_snapshot_blocks != 63
            or self.stream.consolidation_snapshot_blocks != 64
            or self.adapter.epochs != self.evidence.epochs
            or self.static.stream_seeds != self.online.stream_seeds
        ):
            raise ValueError("resolved NCE/TRE LogT protocol violates its fixed design")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of the complete resolved protocol."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the resolved configuration as a JSON-compatible record."""
        record = asdict(self)
        for name in ("artifact_root", "data_root", "baseline_run_root"):
            record[name] = str(getattr(self, name))
        record["baseline"]["expected_main_mean_accuracy"] = dict(
            self.baseline.expected_main_mean_accuracy
        )
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the resolved NCE/TRE protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def load_config(path: str | Path) -> VampLogTEvidenceConfig:
    """Load the one strict YAML surface and reject undeclared scientific choices."""
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
            "stream",
            "adapter",
            "evidence",
            "calibration",
            "static",
            "consolidation",
            "online",
            "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    paths = _mapping(root["paths"], "paths", {"artifact_root", "data_root", "baseline_run_root"})
    baseline = _mapping(
        root["baseline"],
        "baseline",
        {
            "run_id",
            "base_checkpoint_sha256",
            "feature_cache_sha256",
            "expected_main_mean_accuracy",
            "expected_main_mean_oracle_leaf_accuracy",
        },
    )
    expected_means = _mapping(
        baseline["expected_main_mean_accuracy"],
        "baseline expected means",
        {"af", "frozen_base", "global_replay", "joint_iid", "oracle_context"},
    )
    stream = _mapping(
        root["stream"],
        "stream",
        {
            "rotations_deg",
            "label_shifts",
            "interpolation",
            "examples_per_context",
            "block_size",
            "static_snapshot_blocks",
            "consolidation_snapshot_blocks",
        },
    )
    adapter = _mapping(
        root["adapter"],
        "adapter",
        {"epochs", "batch_size", "learning_rate", "weight_decay", "beta1", "beta2", "epsilon"},
    )
    evidence = _mapping(
        root["evidence"],
        "evidence",
        {
            "reference",
            "direct_bridges",
            "candidate_tre_bridges",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "initial_replacement_probability",
            "independent_replicas",
            "score_batch_size",
        },
    )
    calibration = _mapping(
        root["calibration"],
        "calibration",
        {
            "dimensions",
            "component_probabilities",
            "reference_probability",
            "tre_bridges",
            "training_steps",
            "batch_size",
            "evaluation_examples",
            "replicas",
            "learning_rate",
            "weight_decay",
            "signed_bias_max_nats",
            "tre_rmse_max_nats",
            "interseed_rmse_max_nats",
            "direct_to_tre_rmse_ratio_min",
        },
    )
    static = _mapping(
        root["static"],
        "static",
        {
            "stream_seeds",
            "heldout_examples_per_node",
            "adjacent_balanced_accuracy_max",
            "independent_route_agreement_min",
            "classifier_oracle_gap_max",
        },
    )
    consolidation = _mapping(
        root["consolidation"],
        "consolidation",
        {
            "heldout_examples_per_merge",
            "raw_score_difference_max_nats",
            "route_agreement_min",
            "classifier_accuracy_gap_max",
            "nce_loss_relative_difference_max",
            "level_offset_slope_max_nats",
        },
    )
    online = _mapping(root["online"], "online", {"stream_seeds", "evaluation_batch_size"})
    runtime = _mapping(root["runtime"], "runtime", {"device", "deterministic_algorithms", "progress"})
    project_root = source.parents[2]
    return VampLogTEvidenceConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["data_root"], project_root),
        _path(paths["baseline_run_root"], project_root),
        BaselineConfig(
            str(baseline["run_id"]),
            str(baseline["base_checkpoint_sha256"]),
            str(baseline["feature_cache_sha256"]),
            tuple(sorted((str(name), float(value)) for name, value in expected_means.items())),
            float(baseline["expected_main_mean_oracle_leaf_accuracy"]),
        ),
        StreamConfig(
            tuple(float(value) for value in stream["rotations_deg"]),
            tuple(int(value) for value in stream["label_shifts"]),
            str(stream["interpolation"]),
            int(stream["examples_per_context"]),
            int(stream["block_size"]),
            int(stream["static_snapshot_blocks"]),
            int(stream["consolidation_snapshot_blocks"]),
        ),
        AdapterConfig(**{name: int(value) if name in {"epochs", "batch_size"} else float(value) for name, value in adapter.items()}),
        EvidenceConfig(
            str(evidence["reference"]),
            int(evidence["direct_bridges"]),
            tuple(int(value) for value in evidence["candidate_tre_bridges"]),
            int(evidence["epochs"]),
            int(evidence["batch_size"]),
            float(evidence["learning_rate"]),
            float(evidence["weight_decay"]),
            float(evidence["beta1"]),
            float(evidence["beta2"]),
            float(evidence["epsilon"]),
            float(evidence["initial_replacement_probability"]),
            int(evidence["independent_replicas"]),
            int(evidence["score_batch_size"]),
        ),
        CalibrationConfig(
            int(calibration["dimensions"]),
            tuple(float(value) for value in calibration["component_probabilities"]),
            float(calibration["reference_probability"]),
            int(calibration["tre_bridges"]),
            int(calibration["training_steps"]),
            int(calibration["batch_size"]),
            int(calibration["evaluation_examples"]),
            int(calibration["replicas"]),
            float(calibration["learning_rate"]),
            float(calibration["weight_decay"]),
            float(calibration["signed_bias_max_nats"]),
            float(calibration["tre_rmse_max_nats"]),
            float(calibration["interseed_rmse_max_nats"]),
            float(calibration["direct_to_tre_rmse_ratio_min"]),
        ),
        StaticGateConfig(
            tuple(int(value) for value in static["stream_seeds"]),
            int(static["heldout_examples_per_node"]),
            float(static["adjacent_balanced_accuracy_max"]),
            float(static["independent_route_agreement_min"]),
            float(static["classifier_oracle_gap_max"]),
        ),
        ConsolidationGateConfig(
            int(consolidation["heldout_examples_per_merge"]),
            float(consolidation["raw_score_difference_max_nats"]),
            float(consolidation["route_agreement_min"]),
            float(consolidation["classifier_accuracy_gap_max"]),
            float(consolidation["nce_loss_relative_difference_max"]),
            float(consolidation["level_offset_slope_max_nats"]),
        ),
        OnlineConfig(
            tuple(int(value) for value in online["stream_seeds"]),
            int(online["evaluation_batch_size"]),
        ),
        RuntimeConfig(
            str(runtime["device"]),
            bool(runtime["deterministic_algorithms"]),
            bool(runtime["progress"]),
        ),
    )


__all__ = [
    "AdapterConfig",
    "BaselineConfig",
    "CalibrationConfig",
    "ConsolidationGateConfig",
    "EvidenceConfig",
    "OnlineConfig",
    "RuntimeConfig",
    "StaticGateConfig",
    "StreamConfig",
    "VampLogTEvidenceConfig",
    "load_config",
]
