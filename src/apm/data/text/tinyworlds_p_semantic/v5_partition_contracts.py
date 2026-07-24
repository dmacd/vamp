"""Strict contracts for the balance-eligible semantic-v5 partition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    ControlSelection,
    NormalizationIdentity,
    ProgressEvent,
    SourceIdentity,
    SplitCount,
    TokenizerIdentity,
    WorldCell,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    ControlPair,
    WORLD_LABELS,
    require_sha256,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4SemanticCatalog
from apm.data.text.tinyworlds_p_semantic.v4_partition_contracts import (
    V4_PARTITION_CONFIG_VERSION,
    V4_SEMANTIC_PARTITION_PRESET,
    V4SemanticPartitionFailure,
    V4SemanticPartitionPreset,
)


V5_BENCHMARK_ID = "tinyworlds-p-semantic-v5"
V5_PARTITION_CONFIG_VERSION = "tinyworlds-p-semantic-partition-v5"
V5_PARTITION_FORMAT = "tinyworlds-p-semantic-v5-partition"
V5_PARTITION_TREE_FORMAT = "tinyworlds-p-semantic-v5-tree"
V5_PARTITION_FAILURE_FORMAT = "tinyworlds-p-semantic-v5-partition-failure"
V5_PARTITION_FAILURE_TREE_FORMAT = (
    "tinyworlds-p-semantic-v5-partition-failure-tree"
)
V5_SAMPLE_REPORT_FORMAT = "tinyworlds-p-semantic-v5-sample-report"
V5_SAMPLE_REPORT_TREE_FORMAT = "tinyworlds-p-semantic-v5-sample-report-tree"
V5_PARTITION_SCHEMA_VERSION = 1
V5_PARENT_CATALOG_SHA256 = (
    "ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee"
)
V5_PARENT_PARTITION_FAILURE_SHA256 = (
    "37fca844f6d172de7896e15630f39794ed17b89afdc4cc28611b8a51ba282e07"
)
V5_TOPOLOGY_SELECTION_METHOD = (
    "component-and-control-visible,median-feasible,semantic-lexicographic-v1"
)
V5_CONTROL_ALLOCATION_FAILURE_STAGE = "control_allocation"


@dataclass(frozen=True, slots=True)
class V5SemanticPartitionPreset:
    """Every scientific and execution choice for semantic-v5 partitioning."""

    version: str = V5_PARTITION_CONFIG_VERSION
    parent_catalog_sha256: str = V5_PARENT_CATALOG_SHA256
    parent_partition_failure_sha256: str = V5_PARENT_PARTITION_FAILURE_SHA256
    topology_selection_method: str = V5_TOPOLOGY_SELECTION_METHOD
    public_seed: int = 0
    worker_count: int = 24
    run_record_count: int = 50_000
    shard_target_bytes: int = 32 * 1024 * 1024
    batch_block_documents: int = 1_024
    context_length: int = 256
    batch_size: int = 32
    minimum_role_coverage: float = 0.95
    selected_cell_median_tolerance: float = 0.10
    minimum_component_outside_groups: int = 64
    world_split_weights: tuple[int, int, int] = (80, 10, 10)
    base_split_weights: tuple[int, int, int] = (96, 2, 2)
    control_token_tolerance: float = 0.0025
    control_source_feature_tolerance: float = 0.02
    control_adjective_length_tolerance: float = 0.03
    control_mean_length_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if (
            self.version != V5_PARTITION_CONFIG_VERSION
            or self.parent_catalog_sha256 != V5_PARENT_CATALOG_SHA256
            or self.parent_partition_failure_sha256
            != V5_PARENT_PARTITION_FAILURE_SHA256
            or self.topology_selection_method != V5_TOPOLOGY_SELECTION_METHOD
        ):
            raise ValueError("semantic-v5 fixed partition choice changed")
        if self.v4_shape.as_record() != V4_SEMANTIC_PARTITION_PRESET.as_record():
            raise ValueError("semantic-v5 changed a frozen v4 partition setting")

    @property
    def v4_shape(self) -> V4SemanticPartitionPreset:
        """Validate unchanged mechanics through the preceding strict contract."""
        return V4SemanticPartitionPreset(
            version=V4_PARTITION_CONFIG_VERSION,
            public_seed=self.public_seed,
            worker_count=self.worker_count,
            run_record_count=self.run_record_count,
            shard_target_bytes=self.shard_target_bytes,
            batch_block_documents=self.batch_block_documents,
            context_length=self.context_length,
            batch_size=self.batch_size,
            minimum_role_coverage=self.minimum_role_coverage,
            selected_cell_median_tolerance=self.selected_cell_median_tolerance,
            minimum_component_outside_groups=self.minimum_component_outside_groups,
            world_split_weights=self.world_split_weights,
            base_split_weights=self.base_split_weights,
            control_token_tolerance=self.control_token_tolerance,
            control_source_feature_tolerance=self.control_source_feature_tolerance,
            control_adjective_length_tolerance=(
                self.control_adjective_length_tolerance
            ),
            control_mean_length_tolerance=self.control_mean_length_tolerance,
        )

    def as_record(self) -> dict[str, object]:
        """Return scientific choices while excluding worker and sort-run counts."""
        common = self.v4_shape.as_record()
        return {
            **common,
            "parent_catalog_sha256": self.parent_catalog_sha256,
            "parent_partition_failure_sha256": (
                self.parent_partition_failure_sha256
            ),
            "topology_selection_method": self.topology_selection_method,
            "version": self.version,
        }


V5_SEMANTIC_PARTITION_PRESET = V5SemanticPartitionPreset()


@dataclass(frozen=True, slots=True)
class V5SemanticPartitionInputs:
    """Authenticated sources and destinations for semantic-v5 partitioning."""

    archive_path: Path
    tokenizer_directory: Path
    semantic_catalog_directory: Path
    parent_partition_failure_directory: Path
    output_root: Path
    temporary_directory: Path
    archive_identity: SourceIdentity = CANONICAL_ARCHIVE_IDENTITY
    tokenizer_identity: TokenizerIdentity = CANONICAL_TOKENIZER_IDENTITY
    progress: Callable[[ProgressEvent], None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "archive_path",
            "tokenizer_directory",
            "semantic_catalog_directory",
            "parent_partition_failure_directory",
            "output_root",
            "temporary_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.progress is not None and not callable(self.progress):
            raise TypeError("semantic-v5 partition progress must be callable")


@dataclass(frozen=True, slots=True)
class V5SemanticPartitionArtifact:
    """A strictly authenticated semantic-v5 archive partition."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    semantic_catalog: V4SemanticCatalog
    parent_partition_failure: V4SemanticPartitionFailure
    normalization: NormalizationIdentity
    preset: V5SemanticPartitionPreset
    cells: tuple[WorldCell, ...]
    controls: tuple[ControlSelection, ...]
    pairings: tuple[ControlPair, ...]
    split_counts: tuple[SplitCount, ...]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.partition_sha256, "semantic-v5 partition identity")
        require_sha256(self.manifest_sha256, "semantic-v5 tree identity")
        if type(self.semantic_catalog) is not V4SemanticCatalog:
            raise TypeError("semantic-v5 partition requires the strict v4 catalog")
        if type(self.parent_partition_failure) is not V4SemanticPartitionFailure:
            raise TypeError("semantic-v5 partition requires the strict v4 failure")
        if tuple(getattr(cell, "label", None) for cell in self.cells) != WORLD_LABELS:
            raise ValueError("semantic-v5 partition requires worlds A through E")


@dataclass(frozen=True, slots=True)
class V5ControlShortfall:
    """The exact control arm whose candidate pool was too small."""

    world: str
    split: str
    arm: str
    available_count: int
    required_count: int

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS:
            raise ValueError("semantic-v5 control shortfall has an unknown world")
        if self.split not in ("validation", "test"):
            raise ValueError("semantic-v5 control shortfall has an unknown split")
        if self.arm not in ("row", "column"):
            raise ValueError("semantic-v5 control shortfall has an unknown arm")
        if (
            type(self.available_count) is not int
            or type(self.required_count) is not int
            or not 0 <= self.available_count < self.required_count
        ):
            raise ValueError("semantic-v5 control shortfall counts are inconsistent")

    @property
    def reason(self) -> str:
        """Return the canonical underlying partition-gate message."""
        return (
            f"control:{self.world}:{self.split}:{self.arm} has "
            f"{self.available_count} candidates for {self.required_count} controls"
        )

    def as_record(self) -> dict[str, object]:
        """Return the canonical persisted shortfall record."""
        return {
            "arm": self.arm,
            "available_count": self.available_count,
            "required_count": self.required_count,
            "shortage_count": self.required_count - self.available_count,
            "split": self.split,
            "world": self.world,
        }


@dataclass(frozen=True, slots=True)
class V5SemanticPartitionFailure:
    """Authenticated evidence that v5 could not construct its controls."""

    root: Path
    failure_sha256: str
    catalog_sha256: str
    parent_partition_failure_sha256: str
    seed_identity_sha256: str
    assignments_sha256: str
    reason: str
    shortfall: V5ControlShortfall
    topology_selection: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.failure_sha256, "semantic-v5 partition failure"),
            (self.catalog_sha256, "semantic-v5 partition failure catalog"),
            (
                self.parent_partition_failure_sha256,
                "semantic-v5 parent partition failure",
            ),
            (self.seed_identity_sha256, "semantic-v5 partition failure seed"),
            (self.assignments_sha256, "semantic-v5 failed assignment ledger"),
        ):
            require_sha256(value, label)
        if (
            self.catalog_sha256 != V5_PARENT_CATALOG_SHA256
            or self.parent_partition_failure_sha256
            != V5_PARENT_PARTITION_FAILURE_SHA256
        ):
            raise ValueError("semantic-v5 partition failure parents changed")
        if self.reason != self.shortfall.reason:
            raise ValueError("semantic-v5 partition failure reason changed")
        if type(self.topology_selection) is not dict:
            raise TypeError("semantic-v5 partition failure topology must be a mapping")


@dataclass(frozen=True, slots=True)
class V5SemanticSampleReport:
    """Authenticated validation samples bound to the semantic-v5 partition."""

    root: Path
    report_sha256: str
    partition_sha256: str
    catalog_sha256: str
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.report_sha256, "semantic-v5 sample report"),
            (self.partition_sha256, "semantic-v5 sample partition"),
            (self.catalog_sha256, "semantic-v5 sample catalog"),
        ):
            require_sha256(value, label)
        if type(self.sample_count) is not int or self.sample_count != 16:
            raise ValueError("semantic-v5 report must cover exactly 16 conditions")


__all__ = [
    "V5_BENCHMARK_ID",
    "V5_CONTROL_ALLOCATION_FAILURE_STAGE",
    "V5_PARENT_CATALOG_SHA256",
    "V5_PARENT_PARTITION_FAILURE_SHA256",
    "V5_PARTITION_CONFIG_VERSION",
    "V5_PARTITION_FORMAT",
    "V5_PARTITION_FAILURE_FORMAT",
    "V5_PARTITION_FAILURE_TREE_FORMAT",
    "V5_PARTITION_SCHEMA_VERSION",
    "V5_PARTITION_TREE_FORMAT",
    "V5_SAMPLE_REPORT_FORMAT",
    "V5_SAMPLE_REPORT_TREE_FORMAT",
    "V5_SEMANTIC_PARTITION_PRESET",
    "V5_TOPOLOGY_SELECTION_METHOD",
    "V5ControlShortfall",
    "V5SemanticPartitionArtifact",
    "V5SemanticPartitionFailure",
    "V5SemanticPartitionInputs",
    "V5SemanticPartitionPreset",
    "V5SemanticSampleReport",
]
