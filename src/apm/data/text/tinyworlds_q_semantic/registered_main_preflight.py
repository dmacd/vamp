"""Exact accepted GPU preflight for the five-world main run."""

from __future__ import annotations

from pathlib import Path

from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    QueryPartitionArtifact,
)
from apm.data.text.tinyworlds_q_semantic.preflight import (
    QueryGpuPreflight,
    load_query_gpu_preflight,
)


MAIN_GPU_PREFLIGHT_SHA256 = (
    "28380737a808e4288c9b8b51cd6a97e9c64c60e23a59b51e10fd2ea565e14641"
)


def load_registered_main_gpu_preflight(
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
    preset: QueryExperimentPreset,
    checkpoint_root: str | Path,
) -> QueryGpuPreflight:
    """Strictly load the passing measured resource evidence for main training."""
    preflight = load_query_gpu_preflight(
        Path(checkpoint_root) / "preflight" / MAIN_GPU_PREFLIGHT_SHA256,
        artifact,
        catalog,
        preset,
    )
    if preflight.preflight_sha256 != MAIN_GPU_PREFLIGHT_SHA256:
        raise RuntimeError("registered main GPU preflight identity changed")
    return preflight


__all__ = ["MAIN_GPU_PREFLIGHT_SHA256", "load_registered_main_gpu_preflight"]
