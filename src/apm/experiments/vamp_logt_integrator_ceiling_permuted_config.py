"""Strict configuration for the converged Permuted-MNIST integrator ceiling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import file_sha256, record_sha256, require_sha256
from apm.continual.logt_behavioral_integrator import FullReplayConvergenceConfig
from apm.experiments.vamp_logt_integrator_permuted_config import (
    IntegratorBenchmarkConfig,
    IntegratorConfig,
    IntegratorEvaluationConfig,
    VampLogTIntegratorConfig,
    load_config as load_parent_config,
)
from apm.experiments.vamp_logt_router_config import (
    AdapterConfig,
    BaselineConfig,
    RuntimeConfig,
)


@dataclass(frozen=True, slots=True)
class ParentIntegratorConfig:
    """Authenticated completed integrator result and comparison ledgers."""

    run_id: str
    protocol_sha256: str
    summary_sha256: str
    implementation_commit: str
    primary_metric_ledger_sha256: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        require_sha256(self.run_id, "parent integrator run")
        require_sha256(self.protocol_sha256, "parent integrator protocol")
        require_sha256(self.summary_sha256, "parent integrator summary")
        seeds = tuple(seed for seed, _digest in self.primary_metric_ledger_sha256)
        if (
            len(self.implementation_commit) < 7
            or seeds != tuple(sorted(set(seeds)))
            or any(
                not 0 <= seed or not _is_sha256(digest)
                for seed, digest in self.primary_metric_ledger_sha256
            )
        ):
            raise ValueError("invalid parent integrator coordinates")


@dataclass(frozen=True, slots=True)
class IntegratorCeilingPhaseConfig:
    """One resumable ceiling phase with independent fresh restarts."""

    seeds: tuple[int, ...]
    macro_steps: int
    restarts_per_step: int

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or any(seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
            or self.macro_steps < 1
            or self.restarts_per_step < 1
        ):
            raise ValueError("invalid converged-integrator ceiling phase")


@dataclass(frozen=True, slots=True)
class VampLogTIntegratorCeilingConfig:
    """Complete empirical optimization-ceiling protocol."""

    name: str
    protocol_revision: str
    artifact_root: Path
    parent_config_path: Path
    parent_config_sha256: str
    parent_integrator_run_root: Path
    parent_integrator: ParentIntegratorConfig
    parent: VampLogTIntegratorConfig
    convergence: FullReplayConvergenceConfig
    smoke: IntegratorCeilingPhaseConfig
    primary: IntegratorCeilingPhaseConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        require_sha256(self.parent_config_sha256, "parent integrator config")
        if (
            self.name != "vamp-logt-converged-integrator-ceiling-permuted-mnist"
            or self.protocol_revision != "integrated-prediction-ceiling-permuted-v1"
            or self.parent_integrator_run_root.name != self.parent_integrator.run_id
            or self.parent.config_hash != self.parent_integrator.run_id
            or self.parent.smoke.macro_steps != self.smoke.macro_steps
            or self.parent.primary.macro_steps != self.primary.macro_steps
            or self.benchmark.macro_steps != self.primary.macro_steps
            or set(self.primary.seeds)
            != {seed for seed, _digest in self.parent_integrator.primary_metric_ledger_sha256}
            or max(self.evaluation.full_checkpoints) > self.primary.macro_steps
            or not self.runtime.deterministic_algorithms
        ):
            raise ValueError("resolved converged-integrator ceiling violates its plan")

    @property
    def config_hash(self) -> str:
        """Return the canonical identity of the resolved ceiling protocol."""
        return record_sha256(self.as_record())

    @property
    def data_root(self) -> Path:
        """Expose the exact parent MNIST source boundary."""
        return self.parent.data_root

    @property
    def baseline_run_root(self) -> Path:
        """Expose the exact frozen-classifier artifact boundary."""
        return self.parent.baseline_run_root

    @property
    def baseline(self) -> BaselineConfig:
        """Expose the exact frozen-classifier coordinates."""
        return self.parent.baseline

    @property
    def benchmark(self) -> IntegratorBenchmarkConfig:
        """Expose the parent's disjoint stream allocation."""
        return self.parent.benchmark

    @property
    def adapter(self) -> AdapterConfig:
        """Expose the parent's frozen node-training protocol."""
        return self.parent.adapter

    @property
    def integrator(self) -> IntegratorConfig:
        """Expose the parent's fixed MLP and optimizer."""
        return self.parent.integrator

    @property
    def evaluation(self) -> IntegratorEvaluationConfig:
        """Expose the parent's untouched test boundary."""
        return self.parent.evaluation

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        return {
            "artifact_root": str(self.artifact_root),
            "convergence": asdict(self.convergence),
            "name": self.name,
            "parent": self.parent.as_record(),
            "parent_config_path": str(self.parent_config_path),
            "parent_config_sha256": self.parent_config_sha256,
            "parent_integrator": {
                **asdict(self.parent_integrator),
                "primary_metric_ledger_sha256": {
                    str(seed): digest
                    for seed, digest in self.parent_integrator.primary_metric_ledger_sha256
                },
            },
            "parent_integrator_run_root": str(self.parent_integrator_run_root),
            "primary": asdict(self.primary),
            "protocol_revision": self.protocol_revision,
            "runtime": asdict(self.runtime),
            "smoke": asdict(self.smoke),
        }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the converged-integrator protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> VampLogTIntegratorCeilingConfig:
    """Load the sole strict ceiling YAML and its authenticated parent config."""
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
            "parent_integrator",
            "convergence",
            "phases",
            "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "protocol_revision"})
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "parent_config", "parent_integrator_run_root"},
    )
    parent_row = _mapping(
        root["parent_integrator"],
        "parent_integrator",
        {
            "run_id",
            "protocol_sha256",
            "summary_sha256",
            "implementation_commit",
            "parent_config_sha256",
            "primary_metric_ledger_sha256",
        },
    )
    ledger_hashes = parent_row["primary_metric_ledger_sha256"]
    if not isinstance(ledger_hashes, Mapping):
        raise ValueError("parent primary metric hashes must be a seed mapping")
    convergence_row = _mapping(
        root["convergence"],
        "convergence",
        {
            "minimum_epochs",
            "maximum_epochs",
            "improvement_delta",
            "learning_rate_patience",
            "learning_rate_factor",
            "minimum_learning_rate",
            "convergence_patience",
        },
    )
    phases = _mapping(root["phases"], "phases", {"smoke", "primary"})
    runtime_row = _mapping(
        root["runtime"],
        "runtime",
        {"device", "deterministic_algorithms", "progress"},
    )
    project_root = source.parents[2]
    parent_config_path = _path(paths["parent_config"], project_root)
    expected_parent_config_sha256 = str(parent_row["parent_config_sha256"])
    if file_sha256(parent_config_path) != expected_parent_config_sha256:
        raise ValueError("completed integrator config changed or is missing")

    def phase(name: str) -> IntegratorCeilingPhaseConfig:
        row = _mapping(
            phases[name],
            f"phases.{name}",
            {"seeds", "macro_steps", "restarts_per_step"},
        )
        return IntegratorCeilingPhaseConfig(
            tuple(int(value) for value in row["seeds"]),
            int(row["macro_steps"]),
            int(row["restarts_per_step"]),
        )

    return VampLogTIntegratorCeilingConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        parent_config_path,
        expected_parent_config_sha256,
        _path(paths["parent_integrator_run_root"], project_root),
        ParentIntegratorConfig(
            str(parent_row["run_id"]),
            str(parent_row["protocol_sha256"]),
            str(parent_row["summary_sha256"]),
            str(parent_row["implementation_commit"]),
            tuple(
                sorted(
                    (int(seed), str(digest))
                    for seed, digest in ledger_hashes.items()
                )
            ),
        ),
        load_parent_config(parent_config_path),
        FullReplayConvergenceConfig(
            int(convergence_row["minimum_epochs"]),
            int(convergence_row["maximum_epochs"]),
            float(convergence_row["improvement_delta"]),
            int(convergence_row["learning_rate_patience"]),
            float(convergence_row["learning_rate_factor"]),
            float(convergence_row["minimum_learning_rate"]),
            int(convergence_row["convergence_patience"]),
        ),
        phase("smoke"),
        phase("primary"),
        RuntimeConfig(
            str(runtime_row["device"]),
            bool(runtime_row["deterministic_algorithms"]),
            bool(runtime_row["progress"]),
        ),
    )


__all__ = [
    "IntegratorCeilingPhaseConfig",
    "ParentIntegratorConfig",
    "VampLogTIntegratorCeilingConfig",
    "load_config",
]
