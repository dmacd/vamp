"""Training-safe registry for the approved five-world main partition."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.catalog import (
    ValidationCatalogView,
    load_validation_catalog,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    CATALOG_ROOT,
    QueryExperimentPreset,
    QueryPartitionArtifact,
)
from apm.data.text.tinyworlds_q_semantic.main_freeze import MainExperimentFreeze
from apm.data.text.tinyworlds_q_semantic.partition import load_query_partition
from apm.data.text.tinyworlds_q_semantic.registered_main import (
    load_registered_main_authority,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_catalog import (
    MAIN_CATALOG_SHA256,
)


MAIN_PARTITION_SHA256 = (
    "d8536d0295af4fa56174369430b2e615008e28fb239d7d66a428b36988fa7d6b"
)
MAIN_PARTITION_TREE_SHA256 = (
    "566700c59c9c05e87525806a2fd54ff48d283b57b4212884153a6808b12a9828"
)
MAIN_VALIDATION_SAMPLE_REPORT_SHA256 = (
    "a677d66b572610229a52d4d46b20b30d206f665afeb1c8fc3a82fd5e6c170143"
)


def load_registered_main_partition() -> tuple[
    MainExperimentFreeze,
    QueryExperimentPreset,
    ValidationCatalogView,
    QueryPartitionArtifact,
]:
    """Strictly load main sources without deserializing sealed query prompts."""
    *_pilot_authority, preset, frozen = load_registered_main_authority()
    catalog = load_validation_catalog(
        CATALOG_ROOT / "catalog" / MAIN_CATALOG_SHA256
    )
    partition = load_query_partition(
        CATALOG_ROOT / "partitions" / MAIN_PARTITION_SHA256,
        catalog,
    )
    if (
        partition.partition_sha256 != MAIN_PARTITION_SHA256
        or partition.concept_ids != preset.concept_ids
        or frozen.main_config_sha256 != preset.config_sha256
    ):
        raise RuntimeError("registered main partition authority changed")
    return frozen, preset, catalog, partition


__all__ = [
    "MAIN_PARTITION_SHA256",
    "MAIN_PARTITION_TREE_SHA256",
    "MAIN_VALIDATION_SAMPLE_REPORT_SHA256",
    "load_registered_main_partition",
]
