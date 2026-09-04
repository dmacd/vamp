"""Strict configuration for a parent recipe promoted from the clean factorial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import os

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_PROMOTED_CONFIG = Path(
    "configs/vision/imagenetr/logt_integrator_promoted_parent_v4.yaml"
)


@dataclass(frozen=True, slots=True)
class PromotedIntegratorConfig:
    """Content identity for the development-selected full-50 confirmation."""

    name: str
    protocol_revision: str
    artifact_root: Path
    factorial_config: Path
    factorial_run_hash: str
    factorial_summary_sha256: str
    selected_condition: str
    head_initialization: str
    weight_decay: float
    seed_schedule: str

    def __post_init__(self) -> None:
        require_sha256(self.factorial_run_hash, "factorial run")
        require_sha256(self.factorial_summary_sha256, "factorial summary")
        if (
            self.name != "imagenetr50_logt_integrator_promoted_parent_v4"
            or self.protocol_revision != "imagenetr50-logt-integrator-promoted-parent-v4"
            or self.head_initialization not in {"fresh", "inherited_union"}
            or self.weight_decay not in {0.0, 0.0005}
            or self.seed_schedule not in {"joint", "parent"}
        ):
            raise ValueError("invalid promoted-integrator configuration")

    @property
    def config_hash(self) -> str:
        """Return the complete promoted experiment identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["artifact_root"] = str(self.artifact_root)
        record["factorial_config"] = str(self.factorial_config)
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the promoted protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    source = Path(os.path.expandvars(str(value))).expanduser()
    return source.resolve() if source.is_absolute() else (project_root / source).resolve()


def load_promoted_integrator_config(
    path: str | Path = DEFAULT_PROMOTED_CONFIG,
) -> PromotedIntegratorConfig:
    """Load the sole config-driven promoted-parent confirmation."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "paths", "factorial_selection"},
    )
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "protocol_revision"}
    )
    paths = _mapping(root["paths"], "paths", {"artifact_root", "factorial_config"})
    selection = _mapping(
        root["factorial_selection"],
        "factorial_selection",
        {
            "run_hash",
            "summary_sha256",
            "selected_condition",
            "head_initialization",
            "weight_decay",
            "seed_schedule",
        },
    )
    return PromotedIntegratorConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(paths["artifact_root"], project_root),
        _path(paths["factorial_config"], project_root),
        str(selection["run_hash"]),
        str(selection["summary_sha256"]),
        str(selection["selected_condition"]),
        str(selection["head_initialization"]),
        float(selection["weight_decay"]),
        str(selection["seed_schedule"]),
    )


__all__ = [
    "DEFAULT_PROMOTED_CONFIG",
    "PromotedIntegratorConfig",
    "load_promoted_integrator_config",
]
