"""Strict semantic-v4 partition and sample-report contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    NormalizationIdentity,
    SourceIdentity,
    TokenizerIdentity,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    ControlPair,
    WORLD_LABELS,
    require_sha256,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4SemanticCatalog


V4_PARTITION_CONFIG_VERSION = "tinyworlds-p-semantic-partition-v4"
V4_PARTITION_FORMAT = "tinyworlds-p-semantic-v4-partition"
V4_PARTITION_TREE_FORMAT = "tinyworlds-p-semantic-v4-tree"
V4_PARTITION_FAILURE_FORMAT = "tinyworlds-p-semantic-v4-partition-failure"
V4_PARTITION_FAILURE_TREE_FORMAT = "tinyworlds-p-semantic-v4-partition-failure-tree"
V4_SAMPLE_REPORT_FORMAT = "tinyworlds-p-semantic-v4-sample-report"
V4_SAMPLE_REPORT_TREE_FORMAT = "tinyworlds-p-semantic-v4-sample-report-tree"
V4_PARTITION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class V4SemanticPartitionPreset:
    """Every archive split, topology, control, shard, and pairing choice."""

    version: str = V4_PARTITION_CONFIG_VERSION
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
        if self.version != V4_PARTITION_CONFIG_VERSION:
            raise ValueError("semantic-v4 partition config version changed")
        integers = (
            self.worker_count,
            self.run_record_count,
            self.shard_target_bytes,
            self.batch_block_documents,
            self.context_length,
            self.batch_size,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("semantic-v4 partition sizes must be positive")
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("semantic-v4 partition seed must be nonnegative")
        if (
            type(self.minimum_component_outside_groups) is not int
            or self.minimum_component_outside_groups < 0
        ):
            raise ValueError("semantic-v4 component visibility must be nonnegative")
        for weights in (self.world_split_weights, self.base_split_weights):
            if (
                type(weights) is not tuple
                or len(weights) != 3
                or any(type(value) is not int or value <= 0 for value in weights)
            ):
                raise ValueError("semantic-v4 split weights must be positive triples")
        if (
            not isfinite(self.minimum_role_coverage)
            or not 0.0 <= self.minimum_role_coverage <= 1.0
        ):
            raise ValueError("semantic-v4 role coverage must lie in [0, 1]")
        tolerances = (
            self.selected_cell_median_tolerance,
            self.control_token_tolerance,
            self.control_source_feature_tolerance,
            self.control_adjective_length_tolerance,
            self.control_mean_length_tolerance,
        )
        if any(not isfinite(value) or not 0.0 < value < 1.0 for value in tolerances):
            raise ValueError("semantic-v4 partition tolerances must lie in (0, 1)")

    def as_record(self) -> dict[str, object]:
        """Return scientific choices, excluding execution-only worker/run sizes."""
        return {
            name: list(value) if type(value) is tuple else value
            for name, value in (
                ("base_split_weights", self.base_split_weights),
                ("batch_block_documents", self.batch_block_documents),
                ("batch_size", self.batch_size),
                ("context_length", self.context_length),
                (
                    "control_adjective_length_tolerance",
                    self.control_adjective_length_tolerance,
                ),
                ("control_mean_length_tolerance", self.control_mean_length_tolerance),
                (
                    "control_source_feature_tolerance",
                    self.control_source_feature_tolerance,
                ),
                ("control_token_tolerance", self.control_token_tolerance),
                (
                    "minimum_component_outside_groups",
                    self.minimum_component_outside_groups,
                ),
                ("minimum_role_coverage", self.minimum_role_coverage),
                ("public_seed", self.public_seed),
                (
                    "selected_cell_median_tolerance",
                    self.selected_cell_median_tolerance,
                ),
                ("shard_target_bytes", self.shard_target_bytes),
                ("version", self.version),
                ("world_split_weights", self.world_split_weights),
            )
        }


V4_SEMANTIC_PARTITION_PRESET = V4SemanticPartitionPreset()


@dataclass(frozen=True, slots=True)
class V4SemanticPartitionInputs:
    """Authenticated sources and destinations for a semantic-v4 partition."""

    archive_path: Path
    tokenizer_directory: Path
    semantic_catalog_directory: Path
    output_root: Path
    temporary_directory: Path
    archive_identity: SourceIdentity = CANONICAL_ARCHIVE_IDENTITY
    tokenizer_identity: TokenizerIdentity = CANONICAL_TOKENIZER_IDENTITY
    progress: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "archive_path",
            "tokenizer_directory",
            "semantic_catalog_directory",
            "output_root",
            "temporary_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.progress is not None and not callable(self.progress):
            raise TypeError("semantic-v4 partition progress must be callable")


@dataclass(frozen=True, slots=True)
class V4SemanticPartitionArtifact:
    """A strictly authenticated semantic-v4 archive partition."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    semantic_catalog: V4SemanticCatalog
    normalization: NormalizationIdentity
    preset: V4SemanticPartitionPreset
    cells: tuple[object, ...]
    controls: tuple[object, ...]
    pairings: tuple[ControlPair, ...]
    split_counts: tuple[object, ...]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.partition_sha256, "semantic-v4 partition identity")
        require_sha256(self.manifest_sha256, "semantic-v4 tree identity")
        if type(self.semantic_catalog) is not V4SemanticCatalog:
            raise TypeError("semantic-v4 partition requires its strict catalog")
        if tuple(getattr(cell, "label", None) for cell in self.cells) != WORLD_LABELS:
            raise ValueError("semantic-v4 partition requires worlds A through E")


@dataclass(frozen=True, slots=True)
class V4SemanticSampleReport:
    """Authenticated validation-only samples bound to v4 partition and catalog."""

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
            (self.report_sha256, "semantic-v4 sample report"),
            (self.partition_sha256, "semantic-v4 sample partition"),
            (self.catalog_sha256, "semantic-v4 sample catalog"),
        ):
            require_sha256(value, label)
        if type(self.sample_count) is not int or self.sample_count != 16:
            raise ValueError("semantic-v4 report must cover exactly 16 conditions")


@dataclass(frozen=True, slots=True)
class V4SemanticPartitionFailure:
    """Authenticated evidence that the fixed v4 partition gate stopped."""

    root: Path
    failure_sha256: str
    catalog_sha256: str
    seed_identity_sha256: str
    reason: str
    audit: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.failure_sha256, "semantic-v4 partition failure"),
            (self.catalog_sha256, "semantic-v4 partition failure catalog"),
            (self.seed_identity_sha256, "semantic-v4 partition failure seed"),
        ):
            require_sha256(value, label)
        if type(self.reason) is not str or not self.reason:
            raise ValueError("semantic-v4 partition failure requires a reason")
        if type(self.audit) is not dict:
            raise TypeError("semantic-v4 partition failure audit must be a mapping")


__all__ = [
    "V4_PARTITION_CONFIG_VERSION",
    "V4_PARTITION_FORMAT",
    "V4_PARTITION_FAILURE_FORMAT",
    "V4_PARTITION_FAILURE_TREE_FORMAT",
    "V4_PARTITION_SCHEMA_VERSION",
    "V4_PARTITION_TREE_FORMAT",
    "V4_SAMPLE_REPORT_FORMAT",
    "V4_SAMPLE_REPORT_TREE_FORMAT",
    "V4_SEMANTIC_PARTITION_PRESET",
    "V4SemanticPartitionArtifact",
    "V4SemanticPartitionFailure",
    "V4SemanticPartitionInputs",
    "V4SemanticPartitionPreset",
    "V4SemanticSampleReport",
]
