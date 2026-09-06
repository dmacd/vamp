"""Strict configuration for the stage-31 total-parameter-matched control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256


DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_total_param_matched_control_v12.yaml"
)


@dataclass(frozen=True, slots=True)
class FrontierTotalParamMatchedConfig:
    """Complete immutable configuration for the total-active-parameter control."""

    name: str
    protocol_revision: str
    source_config: Path
    source_result: Path
    source_result_sha256: str
    source_result_content_hash: str
    source_protocol_hash: str
    match_scope: str
    lora_parameters_per_rank: int
    classifier_parameters: int
    frontier_lora_parameters: int
    frontier_integrator_parameters: int
    target_rank: int
    target_alpha: int
    dropout: float
    num_workers: int

    def __post_init__(self) -> None:
        for label, identity in (
            ("source result file", self.source_result_sha256),
            ("source result", self.source_result_content_hash),
            ("source protocol", self.source_protocol_hash),
        ):
            require_sha256(identity, label)
        target = self.frontier_active_parameters
        quotient = max(
            1,
            (target - self.classifier_parameters)
            // self.lora_parameters_per_rank,
        )
        nearest_rank = min(
            (quotient, quotient + 1),
            key=lambda rank: (
                abs(
                    rank * self.lora_parameters_per_rank
                    + self.classifier_parameters
                    - target
                ),
                rank,
            ),
        )
        if (
            self.name
            != "imagenetr50_stage31_joint_iid_total_param_matched_v12"
            or self.protocol_revision
            != "imagenetr50-stage31-joint-iid-total-param-matched-v12"
            or self.match_scope != "total_active_parameters"
            or self.lora_parameters_per_rank != 82_944
            or self.classifier_parameters != 95_356
            or self.frontier_lora_parameters != 6_635_520
            or self.frontier_integrator_parameters != 12_055_496
            or self.target_rank != nearest_rank
            or self.target_alpha != self.target_rank
            or self.dropout != 0.0
            or self.num_workers < 0
        ):
            raise ValueError(
                "configuration differs from the total-parameter-matched control"
            )

    @property
    def frontier_active_parameters(self) -> int:
        """Return the adaptive frontier's five LoRAs plus macro integrator."""
        return self.frontier_lora_parameters + self.frontier_integrator_parameters

    @property
    def joint_active_parameters(self) -> int:
        """Return the joint adapter plus its required affine classifier."""
        return (
            self.target_rank * self.lora_parameters_per_rank
            + self.classifier_parameters
        )

    @property
    def parameter_difference(self) -> int:
        """Return joint minus frontier active parameters."""
        return self.joint_active_parameters - self.frontier_active_parameters

    @property
    def config_hash(self) -> str:
        """Return the canonical scientific and runtime identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSON-compatible configuration record."""
        record = asdict(self)
        record["source_config"] = str(self.source_config)
        record["source_result"] = str(self.source_result)
        record.update(
            {
                "frontier_active_parameters": self.frontier_active_parameters,
                "joint_active_parameters": self.joint_active_parameters,
                "parameter_difference": self.parameter_difference,
            }
        )
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"{label} keys differ from the total-parameter-matched protocol"
        )
    return value


def _path(value: object, project_root: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (project_root / expanded).resolve()
    )


def load_frontier_total_param_matched_config(
    path: str | Path = DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG,
) -> FrontierTotalParamMatchedConfig:
    """Load the single config-driven total-parameter-matched control."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    project_root = source.parents[3]
    root = _mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "configuration",
        {"experiment", "source_control", "capacity", "runtime"},
    )
    experiment = _mapping(
        root["experiment"], "experiment", {"name", "protocol_revision"}
    )
    source_control = _mapping(
        root["source_control"],
        "source_control",
        {
            "config",
            "result",
            "result_sha256",
            "result_content_hash",
            "protocol_hash",
        },
    )
    capacity = _mapping(
        root["capacity"],
        "capacity",
        {
            "match_scope",
            "lora_parameters_per_rank",
            "classifier_parameters",
            "frontier_lora_parameters",
            "frontier_integrator_parameters",
            "target_rank",
            "target_alpha",
            "dropout",
        },
    )
    runtime = _mapping(root["runtime"], "runtime", {"num_workers"})
    return FrontierTotalParamMatchedConfig(
        str(experiment["name"]),
        str(experiment["protocol_revision"]),
        _path(source_control["config"], project_root),
        Path(str(source_control["result"])),
        str(source_control["result_sha256"]),
        str(source_control["result_content_hash"]),
        str(source_control["protocol_hash"]),
        str(capacity["match_scope"]),
        int(capacity["lora_parameters_per_rank"]),
        int(capacity["classifier_parameters"]),
        int(capacity["frontier_lora_parameters"]),
        int(capacity["frontier_integrator_parameters"]),
        int(capacity["target_rank"]),
        int(capacity["target_alpha"]),
        float(capacity["dropout"]),
        int(runtime["num_workers"]),
    )


__all__ = [
    "DEFAULT_FRONTIER_TOTAL_PARAM_MATCHED_CONFIG",
    "FrontierTotalParamMatchedConfig",
    "load_frontier_total_param_matched_config",
]
