"""Strict contracts for exact-control-feasible semantic-v6 partitioning."""

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
    SemanticTrainingPreset,
    WORLD_LABELS,
    require_sha256,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4SemanticCatalog
from apm.data.text.tinyworlds_p_semantic.v4_partition_contracts import (
    V4_PARTITION_CONFIG_VERSION,
    V4SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_SEMANTIC_PARTITION_PRESET,
    V5SemanticPartitionFailure,
)
from apm.lm.config import GptNeoConfig


V6_BENCHMARK_ID = "tinyworlds-p-semantic-v6"
V6_PARTITION_CONFIG_VERSION = "tinyworlds-p-semantic-partition-v6"
V6_PARTITION_FORMAT = "tinyworlds-p-semantic-v6-partition"
V6_PARTITION_TREE_FORMAT = "tinyworlds-p-semantic-v6-tree"
V6_PARTITION_FAILURE_FORMAT = "tinyworlds-p-semantic-v6-partition-failure"
V6_PARTITION_FAILURE_TREE_FORMAT = (
    "tinyworlds-p-semantic-v6-partition-failure-tree"
)
V6_SAMPLE_REPORT_FORMAT = "tinyworlds-p-semantic-v6-sample-report"
V6_SAMPLE_REPORT_TREE_FORMAT = "tinyworlds-p-semantic-v6-sample-report-tree"
V6_PARTITION_SCHEMA_VERSION = 1
V6_TRAINING_CONFIG_VERSION = "tinyworlds-p-semantic-training-v6"
V6_PARENT_CATALOG_SHA256 = (
    "ea2e69509a421d3240b92fc727f01819e59e5d0d739d0e24afdb732517d391ee"
)
V6_PARENT_PARTITION_FAILURE_SHA256 = (
    "090b54dbc58f6b2e8a2f500987fe1171002839270a241c26b27f53aae88daa11"
)
V6_TOPOLOGY_SELECTION_METHOD = (
    "component-and-control-visible,median-feasible,"
    "exact-split-controls-feasible,semantic-lexicographic-v1"
)
V6_FEASIBILITY_FAILURE_STAGE = "topology_control_feasibility"
V6_FEASIBILITY_FAILURE_REASON = (
    "no balanced semantic topology completed exact split-level controls"
)


@dataclass(frozen=True, slots=True)
class V6SemanticPartitionPreset:
    """Every scientific and execution choice for semantic-v6 partitioning."""

    version: str = V6_PARTITION_CONFIG_VERSION
    parent_catalog_sha256: str = V6_PARENT_CATALOG_SHA256
    parent_partition_failure_sha256: str = V6_PARENT_PARTITION_FAILURE_SHA256
    topology_selection_method: str = V6_TOPOLOGY_SELECTION_METHOD
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
            self.version != V6_PARTITION_CONFIG_VERSION
            or self.parent_catalog_sha256 != V6_PARENT_CATALOG_SHA256
            or self.parent_partition_failure_sha256
            != V6_PARENT_PARTITION_FAILURE_SHA256
            or self.topology_selection_method != V6_TOPOLOGY_SELECTION_METHOD
        ):
            raise ValueError("semantic-v6 fixed partition choice changed")
        if (
            self.v4_shape.as_record()
            != V5_SEMANTIC_PARTITION_PRESET.v4_shape.as_record()
        ):
            raise ValueError("semantic-v6 changed a frozen partition setting")

    @property
    def v4_shape(self) -> V4SemanticPartitionPreset:
        """Validate unchanged mechanics through the original strict contract."""
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
        return {
            **self.v4_shape.as_record(),
            "parent_catalog_sha256": self.parent_catalog_sha256,
            "parent_partition_failure_sha256": (
                self.parent_partition_failure_sha256
            ),
            "topology_selection_method": self.topology_selection_method,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class V6SemanticTrainingPreset:
    """The frozen seed-zero semantic-v6 base-training contract."""

    version: str = V6_TRAINING_CONFIG_VERSION
    parameter_seed: int = 0
    epochs: int = 5
    calibration_epochs: int = 2
    context_length: int = 256
    microbatch_size: int = 32
    accumulation_microbatches: int = 8
    maximum_learning_rate: float = 5e-4
    minimum_learning_rate: float = 5e-5
    warmup_fraction: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    state_interval_updates: int = 1_000
    allocator_peak_limit_bytes: int = 12 * 1024**3

    def __post_init__(self) -> None:
        reference = SemanticTrainingPreset()
        if self.version != V6_TRAINING_CONFIG_VERSION:
            raise ValueError("semantic-v6 training version changed")
        if self._shared_record != reference.as_record():
            raise ValueError("semantic-v6 changed the frozen base-training contract")

    @property
    def model_config(self) -> GptNeoConfig:
        """Return the registered eight-layer GPT-Neo architecture."""
        return SemanticTrainingPreset().model_config

    @property
    def _shared_record(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in SemanticTrainingPreset.__dataclass_fields__
            },
            "model": SemanticTrainingPreset().as_record()["model"],
        }

    def as_record(self) -> dict[str, object]:
        """Return every behavior-changing v6 training choice."""
        return {"version": self.version, **self._shared_record}


V6_SEMANTIC_TRAINING_PRESET = V6SemanticTrainingPreset()


V6_SEMANTIC_PARTITION_PRESET = V6SemanticPartitionPreset()


@dataclass(frozen=True, slots=True)
class V6SemanticPartitionInputs:
    """Authenticated sources and destinations for semantic-v6 partitioning."""

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
            raise TypeError("semantic-v6 partition progress must be callable")


@dataclass(frozen=True, slots=True)
class V6CandidateFeasibility:
    """The exact split/control result for one balanced semantic candidate."""

    semantic_rank: int
    cells: tuple[tuple[int, int], ...]
    split_assignments_sha256: str
    control_feasible: bool
    controls_sha256: str | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if type(self.semantic_rank) is not int or self.semantic_rank < 0:
            raise ValueError("semantic-v6 feasibility rank must be nonnegative")
        if (
            type(self.cells) is not tuple
            or len(self.cells) != 5
            or any(
                type(cell) is not tuple
                or len(cell) != 2
                or any(type(value) is not int or value < 0 for value in cell)
                for cell in self.cells
            )
        ):
            raise ValueError("semantic-v6 feasibility requires five valid cells")
        require_sha256(
            self.split_assignments_sha256,
            "semantic-v6 candidate split assignments",
        )
        if self.control_feasible:
            if self.controls_sha256 is None or self.failure_reason is not None:
                raise ValueError("semantic-v6 feasible candidate evidence changed")
            require_sha256(self.controls_sha256, "semantic-v6 candidate controls")
        elif (
            self.controls_sha256 is not None
            or type(self.failure_reason) is not str
            or not self.failure_reason
        ):
            raise ValueError("semantic-v6 infeasible candidate evidence changed")

    def as_record(self) -> dict[str, object]:
        """Return the canonical persisted feasibility record."""
        return {
            "cells": [list(cell) for cell in self.cells],
            "control_feasible": self.control_feasible,
            "controls_sha256": self.controls_sha256,
            "failure_reason": self.failure_reason,
            "semantic_rank": self.semantic_rank,
            "split_assignments_sha256": self.split_assignments_sha256,
        }


@dataclass(frozen=True, slots=True)
class V6SemanticPartitionArtifact:
    """A strictly authenticated semantic-v6 archive partition."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    semantic_catalog: V4SemanticCatalog
    parent_partition_failure: V5SemanticPartitionFailure
    normalization: NormalizationIdentity
    preset: V6SemanticPartitionPreset
    cells: tuple[WorldCell, ...]
    controls: tuple[ControlSelection, ...]
    pairings: tuple[ControlPair, ...]
    split_counts: tuple[SplitCount, ...]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.partition_sha256, "semantic-v6 partition identity")
        require_sha256(self.manifest_sha256, "semantic-v6 tree identity")
        if type(self.preset) is not V6SemanticPartitionPreset:
            raise TypeError("semantic-v6 partition requires its strict preset")
        if type(self.semantic_catalog) is not V4SemanticCatalog:
            raise TypeError("semantic-v6 partition requires the strict v4 catalog")
        if type(self.parent_partition_failure) is not V5SemanticPartitionFailure:
            raise TypeError("semantic-v6 partition requires the strict v5 failure")
        if tuple(getattr(cell, "label", None) for cell in self.cells) != WORLD_LABELS:
            raise ValueError("semantic-v6 partition requires worlds A through E")


@dataclass(frozen=True, slots=True)
class V6SemanticPartitionFailure:
    """Authenticated evidence that no balanced v6 candidate was feasible."""

    root: Path
    failure_sha256: str
    catalog_sha256: str
    parent_partition_failure_sha256: str
    seed_identity_sha256: str
    feasibility_sha256: str
    reason: str
    topology_selection: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.failure_sha256, "semantic-v6 partition failure"),
            (self.catalog_sha256, "semantic-v6 partition failure catalog"),
            (
                self.parent_partition_failure_sha256,
                "semantic-v6 parent partition failure",
            ),
            (self.seed_identity_sha256, "semantic-v6 partition failure seed"),
            (self.feasibility_sha256, "semantic-v6 feasibility evidence"),
        ):
            require_sha256(value, label)
        if (
            self.catalog_sha256 != V6_PARENT_CATALOG_SHA256
            or self.parent_partition_failure_sha256
            != V6_PARENT_PARTITION_FAILURE_SHA256
            or self.reason != V6_FEASIBILITY_FAILURE_REASON
        ):
            raise ValueError("semantic-v6 partition failure identity changed")


@dataclass(frozen=True, slots=True)
class V6SemanticSampleReport:
    """Authenticated validation samples bound to the semantic-v6 partition."""

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
            (self.report_sha256, "semantic-v6 sample report"),
            (self.partition_sha256, "semantic-v6 sample partition"),
            (self.catalog_sha256, "semantic-v6 sample catalog"),
        ):
            require_sha256(value, label)
        if type(self.sample_count) is not int or self.sample_count != 16:
            raise ValueError("semantic-v6 report must cover exactly 16 conditions")


__all__ = [
    "V6_BENCHMARK_ID",
    "V6_FEASIBILITY_FAILURE_REASON",
    "V6_FEASIBILITY_FAILURE_STAGE",
    "V6_PARENT_CATALOG_SHA256",
    "V6_PARENT_PARTITION_FAILURE_SHA256",
    "V6_PARTITION_CONFIG_VERSION",
    "V6_PARTITION_FAILURE_FORMAT",
    "V6_PARTITION_FAILURE_TREE_FORMAT",
    "V6_PARTITION_FORMAT",
    "V6_PARTITION_SCHEMA_VERSION",
    "V6_PARTITION_TREE_FORMAT",
    "V6_SAMPLE_REPORT_FORMAT",
    "V6_SAMPLE_REPORT_TREE_FORMAT",
    "V6_SEMANTIC_TRAINING_PRESET",
    "V6_TRAINING_CONFIG_VERSION",
    "V6_SEMANTIC_PARTITION_PRESET",
    "V6_TOPOLOGY_SELECTION_METHOD",
    "V6CandidateFeasibility",
    "V6SemanticPartitionArtifact",
    "V6SemanticPartitionFailure",
    "V6SemanticPartitionInputs",
    "V6SemanticPartitionPreset",
    "V6SemanticSampleReport",
    "V6SemanticTrainingPreset",
]
