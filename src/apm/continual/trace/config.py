"""Strict YAML configuration loading for the preregistered TRACE run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from apm.continual.trace.protocol import (
    DATASET_ARCHIVE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SOURCE_ID,
    MODEL_SOURCE_REVISION,
    SEED,
    MergeMethod,
    MergePolicy,
    default_merge_policies,
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Deployment locations and fixed protocol choices not resolved from manifests."""

    store_root: Path
    model_revision: str
    self_terminate: bool
    gpu_count: int
    soft_pause_hours: float
    hard_limit_hours: float
    network_volume_gb: int
    container_disk_gb: int
    policies: tuple[MergePolicy, ...]

    def __post_init__(self) -> None:
        if (
            not self.store_root.is_absolute()
            or self.gpu_count != 2
            or self.soft_pause_hours != 23.5
            or self.hard_limit_hours != 24.0
            or self.network_volume_gb != 150
            or self.container_disk_gb != 50
            or self.policies != default_merge_policies()
        ):
            raise ValueError("configuration differs from the preregistered primary TRACE run")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the one supported primary YAML shape and reject protocol drift."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("TRACE configuration requires PyYAML") from error
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("TRACE configuration must be one YAML mapping")
    protocol = value.get("protocol")
    storage = value.get("storage")
    runtime = value.get("runtime")
    if not all(type(section) is dict for section in (protocol, storage, runtime)):
        raise ValueError("TRACE configuration is missing protocol, storage, or runtime")
    if (
        protocol.get("model_id") != MODEL_ID
        or protocol.get("model_revision") != MODEL_REVISION
        or protocol.get("model_source_id") != MODEL_SOURCE_ID
        or protocol.get("model_source_revision") != MODEL_SOURCE_REVISION
        or protocol.get("dataset_archive_sha256") != DATASET_ARCHIVE_SHA256
        or protocol.get("seed") != SEED
    ):
        raise ValueError("TRACE protocol identifiers differ from the registered contract")
    return ExperimentConfig(
        store_root=Path(str(storage["root"])),
        model_revision=MODEL_REVISION,
        self_terminate=bool(runtime["self_terminate"]),
        gpu_count=int(runtime["gpu_count"]),
        soft_pause_hours=float(runtime["soft_pause_hours"]),
        hard_limit_hours=float(runtime["hard_limit_hours"]),
        network_volume_gb=int(storage["network_volume_gb"]),
        container_disk_gb=int(storage["container_disk_gb"]),
        policies=default_merge_policies(),
    )


def load_merge_policy(path: str | Path) -> MergePolicy:
    """Load one derived-policy YAML for a leaf-reusing rebuild."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("TRACE configuration requires PyYAML") from error
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("TRACE policy must be one YAML mapping")
    return MergePolicy(
        method=cast(MergeMethod, str(value["method"])),
        output_rank=int(value.get("output_rank", 8)),
        parent_alpha=int(value.get("parent_alpha", 32)),
        core_scale=float(value["core_scale"]) if value.get("core_scale") is not None else None,
        repair_fraction=float(value.get("repair_fraction", 0.0)),
        repair_learning_rate=float(value.get("repair_learning_rate", 5.0e-5)),
    )


__all__ = ["ExperimentConfig", "load_experiment_config", "load_merge_policy"]
