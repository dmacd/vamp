"""Versioned contracts for the TinyWorlds-P semantic benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Literal

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    HashedFile,
    NormalizationIdentity,
    SourceIdentity,
    TokenizerIdentity,
)
from apm.lm.config import GptNeoConfig


BENCHMARK_ID = "tinyworlds-p-semantic-v1"
SEMANTIC_CONFIG_VERSION = "tinyworlds-p-semantic-construction-v1"
EVIDENCE_FORMAT = "tinyworlds-p-semantic-encoder-evidence"
CATALOG_FORMAT = "tinyworlds-p-semantic-catalog"
PARTITION_FORMAT = "tinyworlds-p-semantic-partition"
TRAINING_FORMAT = "tinyworlds-p-semantic-training"
SCHEMA_VERSION = 1
WORLD_LABELS = ("A", "B", "C", "D", "E")

ENCODER_IDENTIFIER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
ENCODER_DIMENSION = 384
ENCODER_SNAPSHOT_IDENTITY_SHA256 = (
    "1101bb824cee453866d6dcd2b489b29ad2c55b20de5bbaceda67f38206a21502"
)

Role = Literal["noun", "verb"]
SplitLabel = Literal["train", "validation", "test"]
ExclusionReason = Literal[
    "insufficient_contexts",
    "nonpositive_role_margin",
    "multiple_realized_senses",
    "cluster_boundary_margin",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value using the benchmark's canonical form."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def record_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON value."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: str, label: str) -> None:
    """Require a canonical lowercase SHA-256 string."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One recursively named file in the pinned encoder snapshot."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("encoder file path must be safe and relative")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("encoder file size must be positive")
        require_sha256(self.sha256, f"encoder file {self.relative_path}")

    def as_record(self) -> dict[str, str | int]:
        """Return the canonical encoder-file record."""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class EncoderIdentity:
    """The complete local identity and inference semantics of MiniLM."""

    identifier: str
    revision: str
    dimension: int
    files: tuple[ModelFile, ...]
    pooling: Literal["attention-mask-mean"] = "attention-mask-mean"
    normalization: Literal["l2"] = "l2"
    dtype: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        if self.identifier != ENCODER_IDENTIFIER:
            raise ValueError("semantic-v1 requires the pinned MiniLM identifier")
        if self.revision != ENCODER_REVISION or _REVISION.fullmatch(self.revision) is None:
            raise ValueError("semantic-v1 requires the pinned MiniLM revision")
        if self.dimension != ENCODER_DIMENSION:
            raise ValueError("semantic-v1 requires 384-dimensional MiniLM vectors")
        if type(self.files) is not tuple or not self.files:
            raise ValueError("encoder identity must hash every snapshot file")
        if any(type(item) is not ModelFile for item in self.files):
            raise TypeError("encoder files must be ModelFile values")
        canonical = tuple(sorted(self.files, key=lambda item: item.relative_path))
        if len({item.relative_path for item in canonical}) != len(canonical):
            raise ValueError("encoder file paths must be unique")
        object.__setattr__(self, "files", canonical)

    @property
    def identity_sha256(self) -> str:
        """Return the digest of the complete encoder snapshot and semantics."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical encoder identity."""
        return {
            "dimension": self.dimension,
            "dtype": self.dtype,
            "files": [item.as_record() for item in self.files],
            "identifier": self.identifier,
            "normalization": self.normalization,
            "pooling": self.pooling,
            "revision": self.revision,
        }


NOUN_ANCHORS = (
    "The word {word} is a noun.",
    "This story is about a {word}.",
    "The {word} is here.",
)
VERB_ANCHORS = (
    "The word {word} is a verb.",
    "Someone can {word}.",
    "They decided to {word}.",
)


@dataclass(frozen=True, slots=True)
class SemanticConstructionConfig:
    """Every behavior-changing semantic construction and quality choice."""

    version: str = SEMANTIC_CONFIG_VERSION
    construction_modulus: int = 20
    construction_residue: int = 0
    maximum_contexts_per_word: int = 128
    minimum_contexts_per_word: int = 32
    context_wordpiece_limit: int = 128
    noun_anchors: tuple[str, str, str] = NOUN_ANCHORS
    verb_anchors: tuple[str, str, str] = VERB_ANCHORS
    role_margin_quantile: float = 0.10
    minimum_role_margin: float = 0.0
    maximum_context_silhouette: float = 0.20
    minimum_cluster_margin: float = 0.03
    cluster_count: int = 8
    minimum_cluster_mass_fraction: float = 0.90
    maximum_cluster_mass_fraction: float = 1.10
    maximum_centroid_iterations: int = 100
    maximum_exclusion_passes: int = 5
    minimum_nouns_per_cluster: int = 32
    minimum_verbs_per_cluster: int = 12
    maximum_centroid_pair_cosine: float = 0.90
    minimum_retained_token_fraction: float = 0.40
    representative_contexts_per_cluster: int = 3

    def __post_init__(self) -> None:
        if self.version != SEMANTIC_CONFIG_VERSION:
            raise ValueError("unsupported semantic construction config version")
        integer_values = (
            self.construction_modulus,
            self.maximum_contexts_per_word,
            self.minimum_contexts_per_word,
            self.context_wordpiece_limit,
            self.cluster_count,
            self.maximum_centroid_iterations,
            self.maximum_exclusion_passes,
            self.minimum_nouns_per_cluster,
            self.minimum_verbs_per_cluster,
            self.representative_contexts_per_cluster,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("semantic integer choices must be positive")
        if not 0 <= self.construction_residue < self.construction_modulus:
            raise ValueError("construction residue must lie inside the modulus")
        if self.minimum_contexts_per_word > self.maximum_contexts_per_word:
            raise ValueError("minimum contexts cannot exceed maximum contexts")
        if self.context_wordpiece_limit < 8:
            raise ValueError("context windows must leave room for meaningful text")
        for templates in (self.noun_anchors, self.verb_anchors):
            if (
                type(templates) is not tuple
                or len(templates) != 3
                or any(template.count("{word}") != 1 for template in templates)
            ):
                raise ValueError("each role requires three one-word anchor templates")
        unit_values = (
            self.role_margin_quantile,
            self.maximum_context_silhouette,
            self.minimum_cluster_margin,
            self.maximum_centroid_pair_cosine,
            self.minimum_retained_token_fraction,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_values):
            raise ValueError("semantic fractions must be finite and lie in [0, 1]")
        if not self.minimum_cluster_mass_fraction < 1.0 < self.maximum_cluster_mass_fraction:
            raise ValueError("cluster mass bounds must straddle one")
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.minimum_cluster_mass_fraction,
                self.maximum_cluster_mass_fraction,
            )
        ):
            raise ValueError("cluster mass bounds must be finite and positive")
        if self.minimum_role_margin != 0.0:
            raise ValueError("semantic-v1 requires a strictly positive role margin")

    def evidence_record(self) -> dict[str, object]:
        """Return only choices that affect reusable encoder evidence."""
        return {
            "construction_modulus": self.construction_modulus,
            "construction_residue": self.construction_residue,
            "context_wordpiece_limit": self.context_wordpiece_limit,
            "maximum_contexts_per_word": self.maximum_contexts_per_word,
            "noun_anchors": list(self.noun_anchors),
            "verb_anchors": list(self.verb_anchors),
            "version": self.version,
        }

    def as_record(self) -> dict[str, object]:
        """Return the full immutable semantic-v1 configuration."""
        return {
            **self.evidence_record(),
            "cluster_count": self.cluster_count,
            "maximum_centroid_iterations": self.maximum_centroid_iterations,
            "maximum_centroid_pair_cosine": self.maximum_centroid_pair_cosine,
            "maximum_context_silhouette": self.maximum_context_silhouette,
            "maximum_exclusion_passes": self.maximum_exclusion_passes,
            "maximum_cluster_mass_fraction": self.maximum_cluster_mass_fraction,
            "minimum_cluster_margin": self.minimum_cluster_margin,
            "minimum_cluster_mass_fraction": self.minimum_cluster_mass_fraction,
            "minimum_contexts_per_word": self.minimum_contexts_per_word,
            "minimum_nouns_per_cluster": self.minimum_nouns_per_cluster,
            "minimum_retained_token_fraction": self.minimum_retained_token_fraction,
            "minimum_role_margin": self.minimum_role_margin,
            "minimum_verbs_per_cluster": self.minimum_verbs_per_cluster,
            "representative_contexts_per_cluster": self.representative_contexts_per_cluster,
            "role_margin_quantile": self.role_margin_quantile,
        }


SEMANTIC_CONFIG = SemanticConstructionConfig()


def semantic_config_from_record(record: dict[str, object]) -> SemanticConstructionConfig:
    """Reconstruct a strict semantic configuration from its canonical record."""
    required = set(SEMANTIC_CONFIG.as_record())
    if set(record) != required:
        raise ValueError("semantic configuration fields changed")

    def integer(name: str) -> int:
        value = record[name]
        if type(value) is not int:
            raise ValueError(f"semantic config {name} must be an integer")
        return value

    def number(name: str) -> float:
        value = record[name]
        if type(value) not in (int, float):
            raise ValueError(f"semantic config {name} must be numeric")
        return float(value)

    def templates(name: str) -> tuple[str, str, str]:
        value = record[name]
        if type(value) is not list or len(value) != 3 or any(type(item) is not str for item in value):
            raise ValueError(f"semantic config {name} must contain three strings")
        return value[0], value[1], value[2]

    return SemanticConstructionConfig(
        version=str(record["version"]),
        construction_modulus=integer("construction_modulus"),
        construction_residue=integer("construction_residue"),
        maximum_contexts_per_word=integer("maximum_contexts_per_word"),
        minimum_contexts_per_word=integer("minimum_contexts_per_word"),
        context_wordpiece_limit=integer("context_wordpiece_limit"),
        noun_anchors=templates("noun_anchors"),
        verb_anchors=templates("verb_anchors"),
        role_margin_quantile=number("role_margin_quantile"),
        minimum_role_margin=number("minimum_role_margin"),
        maximum_context_silhouette=number("maximum_context_silhouette"),
        minimum_cluster_margin=number("minimum_cluster_margin"),
        cluster_count=integer("cluster_count"),
        minimum_cluster_mass_fraction=number("minimum_cluster_mass_fraction"),
        maximum_cluster_mass_fraction=number("maximum_cluster_mass_fraction"),
        maximum_centroid_iterations=integer("maximum_centroid_iterations"),
        maximum_exclusion_passes=integer("maximum_exclusion_passes"),
        minimum_nouns_per_cluster=integer("minimum_nouns_per_cluster"),
        minimum_verbs_per_cluster=integer("minimum_verbs_per_cluster"),
        maximum_centroid_pair_cosine=number("maximum_centroid_pair_cosine"),
        minimum_retained_token_fraction=number("minimum_retained_token_fraction"),
        representative_contexts_per_cluster=integer("representative_contexts_per_cluster"),
    )


@dataclass(frozen=True, slots=True)
class SemanticContext:
    """One exact construction-slice sentence selected for a role word."""

    role: Role
    word: str
    normalized_story_sha256: str
    record_id: str
    story_sha256: str
    sentence: str
    target_start: int
    target_stop: int
    selection_sha256: str

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("semantic context requires a role and word")
        for value, label in (
            (self.normalized_story_sha256, "context group"),
            (self.story_sha256, "context story"),
            (self.selection_sha256, "context selection"),
        ):
            require_sha256(value, label)
        if not self.record_id or not self.sentence:
            raise ValueError("semantic context provenance and sentence must be nonempty")
        if not 0 <= self.target_start < self.target_stop <= len(self.sentence):
            raise ValueError("semantic context target span is outside the sentence")

    def as_record(self) -> dict[str, object]:
        """Return exact context text, span, and archive provenance."""
        return {
            "normalized_story_sha256": self.normalized_story_sha256,
            "record_id": self.record_id,
            "role": self.role,
            "selection_sha256": self.selection_sha256,
            "sentence": self.sentence,
            "story_sha256": self.story_sha256,
            "target_start": self.target_start,
            "target_stop": self.target_stop,
            "word": self.word,
        }


@dataclass(frozen=True, slots=True)
class SemanticCluster:
    """One canonical mass-constrained spherical cluster."""

    role: Role
    index: int
    token_mass: int
    centroid: tuple[float, ...]
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb"):
            raise ValueError("semantic cluster role is invalid")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("semantic cluster index must be nonnegative")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("semantic cluster mass must be positive")
        if type(self.centroid) is not tuple or not self.centroid:
            raise ValueError("semantic cluster requires a centroid")
        if any(not isfinite(value) for value in self.centroid) or not np.isclose(
            np.linalg.norm(np.asarray(self.centroid, dtype=np.float64)),
            1.0,
            atol=1e-5,
        ):
            raise ValueError("semantic cluster centroid must be finite and L2 normalized")
        if type(self.words) is not tuple or tuple(sorted(set(self.words))) != self.words:
            raise ValueError("semantic cluster words must be sorted and unique")

    def as_record(self) -> dict[str, object]:
        """Return the canonical cluster representation."""
        return {
            "centroid": list(self.centroid),
            "index": self.index,
            "role": self.role,
            "token_mass": self.token_mass,
            "words": list(self.words),
        }


@dataclass(frozen=True, slots=True)
class SemanticWord:
    """Audit evidence and final disposition for one role-specific word."""

    role: Role
    word: str
    token_mass: int
    context_count: int
    role_margin_q10: float | None
    context_silhouette: float | None
    cluster_margin: float | None
    cluster: int | None
    exclusion_reason: ExclusionReason | None
    vector: tuple[float, ...] | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("semantic word requires a role and normalized word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("semantic word token mass must be positive")
        if type(self.context_count) is not int or self.context_count < 0:
            raise ValueError("semantic word context count must be nonnegative")
        retained = self.exclusion_reason is None
        if retained != (self.cluster is not None and self.vector is not None):
            raise ValueError("retained words require a cluster and vector")
        if self.cluster is not None and self.cluster < 0:
            raise ValueError("semantic word cluster must be nonnegative")
        for value in (self.role_margin_q10, self.context_silhouette, self.cluster_margin):
            if value is not None and not isfinite(value):
                raise ValueError("semantic word metrics must be finite")
        if self.vector is not None and (
            not self.vector
            or any(not isfinite(value) for value in self.vector)
            or not np.isclose(
                np.linalg.norm(np.asarray(self.vector, dtype=np.float64)),
                1.0,
                atol=1e-5,
            )
        ):
            raise ValueError("semantic word vector must be finite and L2 normalized")

    def as_record(self) -> dict[str, object]:
        """Return the complete audit record for this role word."""
        return {
            "cluster": self.cluster,
            "cluster_margin": self.cluster_margin,
            "context_count": self.context_count,
            "context_silhouette": self.context_silhouette,
            "exclusion_reason": self.exclusion_reason,
            "role": self.role,
            "role_margin_q10": self.role_margin_q10,
            "token_mass": self.token_mass,
            "vector": None if self.vector is None else list(self.vector),
            "word": self.word,
        }


@dataclass(frozen=True, slots=True)
class SemanticEvidenceArtifact:
    """Strictly authenticated reusable MiniLM evidence."""

    root: Path
    evidence_sha256: str
    archive_identity: SourceIdentity
    encoder_identity: EncoderIdentity
    config: SemanticConstructionConfig
    embedding_count: int
    dimension: int
    construction_group_count: int
    construction_token_count: int
    nonconstruction_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.evidence_sha256, "semantic evidence identity")
        if self.dimension != self.encoder_identity.dimension:
            raise ValueError("evidence and encoder dimensions disagree")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.embedding_count,
                self.construction_group_count,
                self.construction_token_count,
                self.nonconstruction_token_count,
            )
        ):
            raise ValueError("semantic evidence counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class SemanticCatalog:
    """One immutable content-addressed semantic word catalog."""

    root: Path
    catalog_sha256: str
    evidence_sha256: str
    encoder_identity: EncoderIdentity
    config: SemanticConstructionConfig
    words: tuple[SemanticWord, ...]
    clusters: tuple[SemanticCluster, ...]
    retained_token_count: int
    nonconstruction_token_count: int
    parent_catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.catalog_sha256, "semantic catalog identity")
        require_sha256(self.evidence_sha256, "semantic evidence identity")
        if self.parent_catalog_sha256 is not None:
            require_sha256(self.parent_catalog_sha256, "parent catalog identity")
        if tuple((item.role, item.word) for item in self.words) != tuple(
            sorted((item.role, item.word) for item in self.words)
        ):
            raise ValueError("semantic catalog words must be canonically ordered")
        expected_clusters = tuple(
            (role, index)
            for role in ("noun", "verb")
            for index in range(self.config.cluster_count)
        )
        if tuple((item.role, item.index) for item in self.clusters) != expected_clusters:
            raise ValueError("semantic catalog clusters are incomplete or unordered")
        if not 0 < self.retained_token_count <= self.nonconstruction_token_count:
            raise ValueError("semantic catalog retained mass is invalid")

    @property
    def retained_token_fraction(self) -> float:
        """Return group-token mass retained after both role exclusions."""
        return self.retained_token_count / self.nonconstruction_token_count

    def word_cluster(self, role: Role) -> dict[str, int]:
        """Return retained word-to-cluster assignments for one role."""
        return {
            item.word: item.cluster
            for item in self.words
            if item.role == role and item.cluster is not None
        }


@dataclass(frozen=True, slots=True)
class SemanticPartitionPreset:
    """Archive-native split, shard, control, and pairing choices."""

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
        integers = (
            self.worker_count,
            self.run_record_count,
            self.shard_target_bytes,
            self.batch_block_documents,
            self.context_length,
            self.batch_size,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("semantic partition sizes must be positive")
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("semantic partition seed must be nonnegative")
        if (
            type(self.minimum_component_outside_groups) is not int
            or self.minimum_component_outside_groups < 0
        ):
            raise ValueError("semantic component visibility minimum must be nonnegative")
        for weights in (self.world_split_weights, self.base_split_weights):
            if (
                type(weights) is not tuple
                or len(weights) != 3
                or any(type(value) is not int or value <= 0 for value in weights)
            ):
                raise ValueError("semantic split weights must be three positive integers")
        if (
            not isfinite(self.minimum_role_coverage)
            or not 0.0 <= self.minimum_role_coverage <= 1.0
        ):
            raise ValueError("semantic role coverage minimum must lie in [0, 1]")
        tolerances = (
            self.selected_cell_median_tolerance,
            self.control_token_tolerance,
            self.control_source_feature_tolerance,
            self.control_adjective_length_tolerance,
            self.control_mean_length_tolerance,
        )
        if any(not isfinite(value) or not 0.0 < value < 1.0 for value in tolerances):
            raise ValueError("semantic partition tolerances must lie in (0, 1)")

    def as_record(self) -> dict[str, object]:
        """Return every partition choice in canonical form."""
        return {
            name: list(value) if type(value) is tuple else value
            for name, value in (
                ("base_split_weights", self.base_split_weights),
                ("batch_block_documents", self.batch_block_documents),
                ("batch_size", self.batch_size),
                ("context_length", self.context_length),
                ("control_adjective_length_tolerance", self.control_adjective_length_tolerance),
                ("control_mean_length_tolerance", self.control_mean_length_tolerance),
                ("control_source_feature_tolerance", self.control_source_feature_tolerance),
                ("control_token_tolerance", self.control_token_tolerance),
                ("minimum_component_outside_groups", self.minimum_component_outside_groups),
                ("minimum_role_coverage", self.minimum_role_coverage),
                ("public_seed", self.public_seed),
                ("selected_cell_median_tolerance", self.selected_cell_median_tolerance),
                ("shard_target_bytes", self.shard_target_bytes),
                ("world_split_weights", self.world_split_weights),
            )
        }


SEMANTIC_PARTITION_PRESET = SemanticPartitionPreset()


@dataclass(frozen=True, slots=True)
class ControlPair:
    """One deterministic world/control duplicate-group pairing."""

    world: str
    split: Literal["validation", "test"]
    arm: Literal["row", "column"]
    world_group_sha256: str
    control_group_sha256: str

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS or self.split not in ("validation", "test"):
            raise ValueError("control pair world or split is invalid")
        if self.arm not in ("row", "column"):
            raise ValueError("control pair arm must be row or column")
        for value, label in (
            (self.world_group_sha256, "paired world group"),
            (self.control_group_sha256, "paired control group"),
        ):
            require_sha256(value, label)
        if self.world_group_sha256 == self.control_group_sha256:
            raise ValueError("world/control pair cannot reuse one group")


@dataclass(frozen=True, slots=True)
class SemanticPartitionInputs:
    """Authenticated sources and destinations for a semantic partition build."""

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
            raise TypeError("semantic partition progress must be callable")


@dataclass(frozen=True, slots=True)
class SemanticPartitionArtifact:
    """A strict semantic-v1 partition identity and runtime surface."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    semantic_catalog: SemanticCatalog
    normalization: NormalizationIdentity
    preset: SemanticPartitionPreset
    cells: tuple[object, ...]
    controls: tuple[object, ...]
    pairings: tuple[ControlPair, ...]
    split_counts: tuple[object, ...]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.partition_sha256, "semantic partition identity")
        require_sha256(self.manifest_sha256, "semantic tree identity")
        if type(self.semantic_catalog) is not SemanticCatalog:
            raise TypeError("semantic partition requires an authenticated catalog")
        if tuple(getattr(cell, "label", None) for cell in self.cells) != WORLD_LABELS:
            raise ValueError("semantic partition requires worlds A through E")


@dataclass(frozen=True, slots=True)
class SemanticTrainingPreset:
    """The frozen seed-zero five-epoch semantic-v1 training contract."""

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
        if (self.parameter_seed, self.epochs, self.calibration_epochs) != (0, 5, 2):
            raise ValueError("semantic-v1 seed and epoch budget are fixed")

    @property
    def model_config(self) -> GptNeoConfig:
        """Return the unchanged eight-layer GPT-Neo architecture."""
        return GptNeoConfig(
            vocab_size=50_257,
            max_position_embeddings=2_048,
            hidden_size=256,
            intermediate_size=1_024,
            num_layers=8,
            num_heads=16,
            attention_types=("global", "local") * 4,
            local_window_size=256,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )

    def as_record(self) -> dict[str, object]:
        """Return the complete semantic training contract."""
        model = self.model_config
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            "model": {
                "activation": model.activation,
                "attention_types": list(model.attention_types),
                "attention_dropout": model.attention_dropout,
                "embedding_dropout": model.embedding_dropout,
                "hidden_size": model.hidden_size,
                "initializer_range": model.initializer_range,
                "intermediate_size": model.intermediate_size,
                "layer_norm_epsilon": model.layer_norm_epsilon,
                "local_window_size": model.local_window_size,
                "max_position_embeddings": model.max_position_embeddings,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "residual_dropout": model.residual_dropout,
                "tied_embeddings": True,
                "vocab_size": model.vocab_size,
            },
        }


SEMANTIC_TRAINING_PRESET = SemanticTrainingPreset()


__all__ = [
    "BENCHMARK_ID",
    "CANONICAL_ARCHIVE_IDENTITY",
    "CANONICAL_TOKENIZER_IDENTITY",
    "CATALOG_FORMAT",
    "ControlPair",
    "ENCODER_DIMENSION",
    "ENCODER_IDENTIFIER",
    "ENCODER_REVISION",
    "ENCODER_SNAPSHOT_IDENTITY_SHA256",
    "EVIDENCE_FORMAT",
    "EncoderIdentity",
    "ExclusionReason",
    "HashedFile",
    "ModelFile",
    "NOUN_ANCHORS",
    "PARTITION_FORMAT",
    "Role",
    "SCHEMA_VERSION",
    "SEMANTIC_CONFIG",
    "SEMANTIC_PARTITION_PRESET",
    "SEMANTIC_TRAINING_PRESET",
    "SemanticCatalog",
    "SemanticCluster",
    "SemanticConstructionConfig",
    "SemanticContext",
    "SemanticEvidenceArtifact",
    "SemanticPartitionArtifact",
    "SemanticPartitionInputs",
    "SemanticPartitionPreset",
    "SemanticTrainingPreset",
    "SemanticWord",
    "SourceIdentity",
    "TokenizerIdentity",
    "VERB_ANCHORS",
    "WORLD_LABELS",
    "canonical_json_bytes",
    "record_sha256",
    "require_sha256",
    "semantic_config_from_record",
]
