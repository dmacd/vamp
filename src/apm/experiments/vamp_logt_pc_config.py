"""Strict resolved protocol for normalized generative-PC LogT evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class PcSourceConfig:
    """Authenticated completed experiment artifacts used only as raw data."""

    nce_run_id: str
    raw_cache_sha256: str
    raw_semantic_sha256: str
    baseline_run_id: str
    feature_cache_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("NCE run ID", self.nce_run_id),
            ("raw cache", self.raw_cache_sha256),
            ("raw semantic identity", self.raw_semantic_sha256),
            ("baseline run ID", self.baseline_run_id),
            ("feature cache", self.feature_cache_sha256),
        ):
            require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class PcModelConfig:
    """Fixed nonlinear density architecture."""

    latent_dim: int
    hidden_dim: int
    image_dim: int
    weight_init_std: float

    def __post_init__(self) -> None:
        if (
            self.latent_dim != 32
            or self.hidden_dim != 128
            or self.image_dim != 784
            or self.weight_init_std <= 0.0
        ):
            raise ValueError("PC evidence architecture differs from the sealed protocol")


@dataclass(frozen=True, slots=True)
class PcEvidenceConfig:
    """Single task-free score permitted by the active experiment."""

    estimator: str

    def __post_init__(self) -> None:
        if self.estimator != "map":
            raise ValueError("the active PC protocol permits only complete MAP scoring")


@dataclass(frozen=True, slots=True)
class PcGaussNewtonEvidenceConfig:
    """Exact generalized Gauss-Newton scoring and diagnostic choices."""

    estimator: str
    primary_score: str
    hessian_diagnostics: str
    negative_direction_epsilons: tuple[float, ...]
    curvature_audit_examples: int
    float64_audit_examples: int
    float64_tolerance_nats: float

    def __post_init__(self) -> None:
        if (
            self.estimator != "generalized_gauss_newton"
            or self.primary_score != "gn1"
            or self.hessian_diagnostics != "exact_hessian_every_query"
            or self.negative_direction_epsilons != (0.01, 0.05, 0.10)
            or self.curvature_audit_examples != 64
            or self.float64_audit_examples != 8
            or self.float64_tolerance_nats != 1.0e-3
        ):
            raise ValueError("GN evidence choices differ from the sealed protocol")


@dataclass(frozen=True, slots=True)
class PcGaussNewtonContinuationEvidenceConfig(PcGaussNewtonEvidenceConfig):
    """User-authorized continuation with precision agreement retained as diagnostic."""

    float64_agreement_role: str

    def __post_init__(self) -> None:
        PcGaussNewtonEvidenceConfig.__post_init__(self)
        if self.float64_agreement_role != "diagnostic_only":
            raise ValueError("GN continuation must retain precision agreement as diagnostic-only")


@dataclass(frozen=True, slots=True)
class PcModelSourceConfig:
    """Authenticated sealed MAP run whose trained models are rescored."""

    map_run_id: str
    required_tree_sha256: str
    map_score_tolerance_nats: float

    def __post_init__(self) -> None:
        require_sha256(self.map_run_id, "MAP model-source run ID")
        require_sha256(self.required_tree_sha256, "MAP model-source tree")
        if self.map_score_tolerance_nats != 1.0e-4:
            raise ValueError("MAP source parity tolerance differs from the sealed protocol")


@dataclass(frozen=True, slots=True)
class PcTrainingConfig:
    """Fixed model and classifier optimization schedules."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    infer_steps: int
    score_batch_size: int
    classifier_epochs: int
    classifier_batch_size: int
    classifier_learning_rate: float
    classifier_weight_decay: float

    def __post_init__(self) -> None:
        if (
            self.epochs != 20
            or self.batch_size != 128
            or self.learning_rate != 1.0e-3
            or self.weight_decay != 1.0e-4
            or self.infer_steps != 80
            or self.score_batch_size < 1
            or self.classifier_epochs != 50
            or self.classifier_batch_size != 128
            or self.classifier_learning_rate != 1.0e-2
            or self.classifier_weight_decay != 1.0e-4
        ):
            raise ValueError("PC evidence training schedule differs from the sealed protocol")


@dataclass(frozen=True, slots=True)
class PcPreflightConfig:
    """Permitted one-node hyperparameter choices and quality gates."""

    train_examples: int
    heldout_examples: int
    image_precisions: tuple[float, ...]
    hidden_precisions: tuple[float, ...]
    inference_step_sizes: tuple[float, ...]
    classifier_accuracy_min: float
    gradient_reduction_min: float

    def __post_init__(self) -> None:
        if (
            self.train_examples != 2_000
            or self.heldout_examples < 128
            or self.image_precisions != (25.0, 100.0)
            or self.hidden_precisions != (1.0, 4.0)
            or self.inference_step_sizes != (0.01, 0.05)
            or self.classifier_accuracy_min != 0.80
            or self.gradient_reduction_min != 10.0
        ):
            raise ValueError("invalid PC one-node preflight protocol")


@dataclass(frozen=True, slots=True)
class PcStreamConfig:
    """Controlled five-context static LogT schedules."""

    rotations_deg: tuple[float, ...]
    label_shifts: tuple[int, ...]
    block_size: int
    static_blocks: int
    minimal_stream_seeds: tuple[int, ...]
    confirmation_stream_seeds: tuple[int, ...]
    model_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.rotations_deg != (0.0, 18.0, 36.0, 54.0, 72.0)
            or self.label_shifts != (0, 2, 4, 6, 8)
            or self.block_size != 250
            or self.static_blocks != 31
            or self.minimal_stream_seeds != (0,)
            or self.confirmation_stream_seeds != (1, 2)
            or self.model_seeds != (0, 1, 2)
        ):
            raise ValueError("invalid controlled 31-block PC stream")


@dataclass(frozen=True, slots=True)
class PcEvaluationConfig:
    """Held-out allocation, bootstrap budget, and static gates."""

    heldout_per_node: int
    focused_examples: int
    bootstrap_resamples: int
    route_agreement_min: float
    classifier_oracle_gap_max: float
    oracle_accuracy_min: float
    novel_win_lower_min: float
    recurrent_win_lower_min: float
    identical_offset_allowance_nats: float
    parent_accuracy_gap_max: float

    def __post_init__(self) -> None:
        if (
            self.heldout_per_node != 128
            or self.focused_examples != 512
            or self.bootstrap_resamples != 10_000
            or self.route_agreement_min != 0.90
            or self.classifier_oracle_gap_max != 0.10
            or self.oracle_accuracy_min != 0.85
            or self.novel_win_lower_min != 0.80
            or self.recurrent_win_lower_min != 0.50
            or self.identical_offset_allowance_nats != 0.25
            or self.parent_accuracy_gap_max != 0.02
        ):
            raise ValueError("invalid PC evidence evaluation protocol")


@dataclass(frozen=True, slots=True)
class PcRuntimeConfig:
    """Execution choices that remain part of run identity."""

    device: str
    progress: bool

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("invalid PC runtime device")


@dataclass(frozen=True, slots=True)
class VampLogTPcConfig:
    """Complete normalized generative-PC experiment configuration."""

    name: str
    protocol_revision: str
    artifact_root: Path
    source_run_root: Path
    baseline_run_root: Path
    source: PcSourceConfig
    model: PcModelConfig
    evidence: (
        PcEvidenceConfig
        | PcGaussNewtonEvidenceConfig
        | PcGaussNewtonContinuationEvidenceConfig
    )
    training: PcTrainingConfig
    preflight: PcPreflightConfig
    stream: PcStreamConfig
    evaluation: PcEvaluationConfig
    runtime: PcRuntimeConfig
    model_source_run_root: Path | None = None
    model_source: PcModelSourceConfig | None = None

    def __post_init__(self) -> None:
        common_changed = (
            self.name != "vamp-logt-generative-pc-evidence-mnist"
            or self.source_run_root.name != self.source.nce_run_id
            or self.baseline_run_root.name != self.source.baseline_run_id
        )
        map_changed = self.protocol_revision == "generative-pc-map-v1" and (
            not isinstance(self.evidence, PcEvidenceConfig)
            or self.model_source_run_root is not None
            or self.model_source is not None
        )
        gn_changed = self.protocol_revision == "generative-pc-gn-v1" and (
            type(self.evidence) is not PcGaussNewtonEvidenceConfig
            or self.model_source_run_root is None
            or self.model_source is None
            or self.model_source_run_root.name != self.model_source.map_run_id
        )
        gn_continuation_changed = self.protocol_revision == "generative-pc-gn-v2" and (
            type(self.evidence) is not PcGaussNewtonContinuationEvidenceConfig
            or self.model_source_run_root is None
            or self.model_source is None
            or self.model_source_run_root.name != self.model_source.map_run_id
        )
        if common_changed or self.protocol_revision not in {
            "generative-pc-map-v1",
            "generative-pc-gn-v1",
            "generative-pc-gn-v2",
        } or map_changed or gn_changed or gn_continuation_changed:
            raise ValueError("resolved generative-PC protocol identity changed")

    @property
    def config_hash(self) -> str:
        """Return the canonical complete protocol identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible resolved configuration."""
        record = asdict(self)
        for name in ("artifact_root", "source_run_root", "baseline_run_root"):
            record[name] = str(getattr(self, name))
        if self.protocol_revision == "generative-pc-map-v1":
            record.pop("model_source_run_root")
            record.pop("model_source")
        else:
            record["model_source_run_root"] = str(self.model_source_run_root)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the generative-PC protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def load_config(path: str | Path) -> VampLogTPcConfig:
    """Load the one strict YAML surface and reject undeclared choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the PC experiment") from error
    source_path = Path(path).resolve()
    raw_root = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_root, Mapping) or "experiment" not in raw_root:
        raise ValueError("configuration keys differ from the generative-PC protocol")
    raw_experiment = _mapping(
        raw_root["experiment"],
        "experiment",
        {"name", "protocol_revision"},
    )
    revision = str(raw_experiment["protocol_revision"])
    gn_protocol = revision in {"generative-pc-gn-v1", "generative-pc-gn-v2"}
    gn_continuation = revision == "generative-pc-gn-v2"
    root_keys = {
        "experiment",
        "paths",
        "source",
        "model",
        "evidence",
        "training",
        "preflight",
        "stream",
        "evaluation",
        "runtime",
    }
    if gn_protocol:
        root_keys.add("model_source")
    root = _mapping(
        raw_root,
        "configuration",
        root_keys,
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    path_keys = {"artifact_root", "source_run_root", "baseline_run_root"}
    if gn_protocol:
        path_keys.add("model_source_run_root")
    paths = _mapping(root["paths"], "paths", path_keys)
    source = _mapping(
        root["source"],
        "source",
        {"nce_run_id", "raw_cache_sha256", "raw_semantic_sha256", "baseline_run_id", "feature_cache_sha256"},
    )
    model = _mapping(root["model"], "model", {"latent_dim", "hidden_dim", "image_dim", "weight_init_std"})
    if gn_protocol:
        evidence_keys = {
            "estimator",
            "primary_score",
            "hessian_diagnostics",
            "negative_direction_epsilons",
            "curvature_audit_examples",
            "float64_audit_examples",
            "float64_tolerance_nats",
        }
        if gn_continuation:
            evidence_keys.add("float64_agreement_role")
        evidence = _mapping(
            root["evidence"],
            "evidence",
            evidence_keys,
        )
        model_source = _mapping(
            root["model_source"],
            "model_source",
            {"map_run_id", "required_tree_sha256", "map_score_tolerance_nats"},
        )
    else:
        evidence = _mapping(root["evidence"], "evidence", {"estimator"})
        model_source = None
    training = _mapping(
        root["training"],
        "training",
        {
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "infer_steps",
            "score_batch_size",
            "classifier_epochs",
            "classifier_batch_size",
            "classifier_learning_rate",
            "classifier_weight_decay",
        },
    )
    preflight = _mapping(
        root["preflight"],
        "preflight",
        {
            "train_examples",
            "heldout_examples",
            "image_precisions",
            "hidden_precisions",
            "inference_step_sizes",
            "classifier_accuracy_min",
            "gradient_reduction_min",
        },
    )
    stream = _mapping(
        root["stream"],
        "stream",
        {
            "rotations_deg",
            "label_shifts",
            "block_size",
            "static_blocks",
            "minimal_stream_seeds",
            "confirmation_stream_seeds",
            "model_seeds",
        },
    )
    evaluation = _mapping(
        root["evaluation"],
        "evaluation",
        {
            "heldout_per_node",
            "focused_examples",
            "bootstrap_resamples",
            "route_agreement_min",
            "classifier_oracle_gap_max",
            "oracle_accuracy_min",
            "novel_win_lower_min",
            "recurrent_win_lower_min",
            "identical_offset_allowance_nats",
            "parent_accuracy_gap_max",
        },
    )
    runtime = _mapping(root["runtime"], "runtime", {"device", "progress"})
    project_root = source_path.parents[2]
    resolved_evidence: (
        PcEvidenceConfig
        | PcGaussNewtonEvidenceConfig
        | PcGaussNewtonContinuationEvidenceConfig
    )
    if gn_protocol:
        evidence_arguments = (
            str(evidence["estimator"]),
            str(evidence["primary_score"]),
            str(evidence["hessian_diagnostics"]),
            tuple(float(value) for value in evidence["negative_direction_epsilons"]),
            int(evidence["curvature_audit_examples"]),
            int(evidence["float64_audit_examples"]),
            float(evidence["float64_tolerance_nats"]),
        )
        resolved_evidence = (
            PcGaussNewtonContinuationEvidenceConfig(
                *evidence_arguments,
                str(evidence["float64_agreement_role"]),
            )
            if gn_continuation
            else PcGaussNewtonEvidenceConfig(*evidence_arguments)
        )
    else:
        resolved_evidence = PcEvidenceConfig(str(evidence["estimator"]))
    return VampLogTPcConfig(
        name=str(experiment["name"]),
        protocol_revision=str(experiment["protocol_revision"]),
        artifact_root=_path(paths["artifact_root"], project_root),
        source_run_root=_path(paths["source_run_root"], project_root),
        baseline_run_root=_path(paths["baseline_run_root"], project_root),
        source=PcSourceConfig(**{name: str(value) for name, value in source.items()}),
        model=PcModelConfig(
            int(model["latent_dim"]),
            int(model["hidden_dim"]),
            int(model["image_dim"]),
            float(model["weight_init_std"]),
        ),
        evidence=resolved_evidence,
        training=PcTrainingConfig(
            int(training["epochs"]),
            int(training["batch_size"]),
            float(training["learning_rate"]),
            float(training["weight_decay"]),
            int(training["infer_steps"]),
            int(training["score_batch_size"]),
            int(training["classifier_epochs"]),
            int(training["classifier_batch_size"]),
            float(training["classifier_learning_rate"]),
            float(training["classifier_weight_decay"]),
        ),
        preflight=PcPreflightConfig(
            int(preflight["train_examples"]),
            int(preflight["heldout_examples"]),
            tuple(float(value) for value in preflight["image_precisions"]),
            tuple(float(value) for value in preflight["hidden_precisions"]),
            tuple(float(value) for value in preflight["inference_step_sizes"]),
            float(preflight["classifier_accuracy_min"]),
            float(preflight["gradient_reduction_min"]),
        ),
        stream=PcStreamConfig(
            tuple(float(value) for value in stream["rotations_deg"]),
            tuple(int(value) for value in stream["label_shifts"]),
            int(stream["block_size"]),
            int(stream["static_blocks"]),
            tuple(int(value) for value in stream["minimal_stream_seeds"]),
            tuple(int(value) for value in stream["confirmation_stream_seeds"]),
            tuple(int(value) for value in stream["model_seeds"]),
        ),
        evaluation=PcEvaluationConfig(
            int(evaluation["heldout_per_node"]),
            int(evaluation["focused_examples"]),
            int(evaluation["bootstrap_resamples"]),
            float(evaluation["route_agreement_min"]),
            float(evaluation["classifier_oracle_gap_max"]),
            float(evaluation["oracle_accuracy_min"]),
            float(evaluation["novel_win_lower_min"]),
            float(evaluation["recurrent_win_lower_min"]),
            float(evaluation["identical_offset_allowance_nats"]),
            float(evaluation["parent_accuracy_gap_max"]),
        ),
        runtime=PcRuntimeConfig(str(runtime["device"]), bool(runtime["progress"])),
        model_source_run_root=(
            _path(paths["model_source_run_root"], project_root) if gn_protocol else None
        ),
        model_source=(
            PcModelSourceConfig(
                str(model_source["map_run_id"]),
                str(model_source["required_tree_sha256"]),
                float(model_source["map_score_tolerance_nats"]),
            )
            if model_source is not None
            else None
        ),
    )


__all__ = [
    "PcEvidenceConfig",
    "PcGaussNewtonEvidenceConfig",
    "PcGaussNewtonContinuationEvidenceConfig",
    "PcModelSourceConfig",
    "PcEvaluationConfig",
    "PcModelConfig",
    "PcPreflightConfig",
    "PcRuntimeConfig",
    "PcSourceConfig",
    "PcStreamConfig",
    "PcTrainingConfig",
    "VampLogTPcConfig",
    "load_config",
]
