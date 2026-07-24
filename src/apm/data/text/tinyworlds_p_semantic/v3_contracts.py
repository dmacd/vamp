"""Frozen semantic-first construction contracts for TinyWorlds-P semantic-v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import (
    EncoderIdentity,
    Role,
    SEMANTIC_CONFIG,
    SemanticCluster,
    require_sha256,
)
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    RoleCalibrationReference,
    V2_BENCHMARK_ID,
)


V3_BENCHMARK_ID = "tinyworlds-p-semantic-v3"
V3_SEMANTIC_CONFIG_VERSION = "tinyworlds-p-semantic-construction-v3"
V3_CATALOG_FORMAT = "tinyworlds-p-semantic-v3-catalog"
V3_CATALOG_FAILURE_FORMAT = "tinyworlds-p-semantic-v3-catalog-failure"
V3_SCHEMA_VERSION = 1
V3_REUSED_EVIDENCE_SHA256 = (
    "efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434"
)
V3_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256 = (
    "23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25"
)

V3ExclusionReason = Literal[
    "insufficient_contexts",
    "calibrated_role_outlier",
    "multiple_realized_senses",
    "cluster_boundary_margin",
]


@dataclass(frozen=True, slots=True)
class V3SemanticConstructionConfig:
    """Every behavior-changing v3 calibration and clustering choice."""

    version: str = V3_SEMANTIC_CONFIG_VERSION
    role_margin_quantile: float = 0.10
    role_calibration_method: str = "word-fold-cross-conformal-lower-tail-v1"
    role_calibration_namespace: str = "role-calibration-fold-v1"
    role_calibration_source_benchmark_id: str = V2_BENCHMARK_ID
    role_calibration_source_failure_sha256: str = (
        V3_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256
    )
    role_calibration_fold_count: int = 5
    role_calibration_alpha: float = 0.05
    minimum_calibration_reference_words: int = 48
    minimum_contexts_per_word: int = 32
    maximum_context_silhouette: float = 0.20
    word_vector_anchor_mode: str = "target-role-centroid"
    cluster_count: int = 8
    clustering_method: str = "farthest-first-unweighted-spherical-kmeans-v1"
    cluster_assignment: str = "nearest-centroid-cosine"
    semantic_word_weighting: str = "uniform"
    balance_stage: str = "partition-story-allocation"
    construction_numpy_version: str = "1.26.4"
    minimum_cluster_margin: float = 0.03
    maximum_centroid_iterations: int = 100
    maximum_exclusion_passes: int = 5
    minimum_nouns_per_cluster: int = 32
    minimum_verbs_per_cluster: int = 12
    maximum_centroid_pair_cosine: float = 0.90
    minimum_retained_token_fraction: float = 0.40
    representative_contexts_per_cluster: int = 3

    def __post_init__(self) -> None:
        fixed_text = {
            "version": V3_SEMANTIC_CONFIG_VERSION,
            "role_calibration_method": "word-fold-cross-conformal-lower-tail-v1",
            "role_calibration_namespace": "role-calibration-fold-v1",
            "role_calibration_source_benchmark_id": V2_BENCHMARK_ID,
            "role_calibration_source_failure_sha256": (
                V3_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256
            ),
            "word_vector_anchor_mode": "target-role-centroid",
            "clustering_method": "farthest-first-unweighted-spherical-kmeans-v1",
            "cluster_assignment": "nearest-centroid-cosine",
            "semantic_word_weighting": "uniform",
            "balance_stage": "partition-story-allocation",
            "construction_numpy_version": "1.26.4",
        }
        if any(getattr(self, name) != value for name, value in fixed_text.items()):
            raise ValueError("semantic-v3 fixed construction choice changed")
        integers = (
            self.role_calibration_fold_count,
            self.minimum_calibration_reference_words,
            self.minimum_contexts_per_word,
            self.cluster_count,
            self.maximum_centroid_iterations,
            self.maximum_exclusion_passes,
            self.minimum_nouns_per_cluster,
            self.minimum_verbs_per_cluster,
            self.representative_contexts_per_cluster,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("semantic-v3 integer choices must be positive")
        if self.role_calibration_fold_count < 2:
            raise ValueError("cross-conformal calibration requires at least two folds")
        fractions = (
            self.role_margin_quantile,
            self.role_calibration_alpha,
            self.maximum_context_silhouette,
            self.minimum_cluster_margin,
            self.maximum_centroid_pair_cosine,
            self.minimum_retained_token_fraction,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("semantic-v3 fractions must be finite and lie in [0, 1]")
        if not 0.0 < self.role_calibration_alpha < 1.0:
            raise ValueError("semantic-v3 conformal alpha must lie in (0, 1)")

    def evidence_record(self) -> dict[str, object]:
        """Return the exact v1 encoder-evidence contract reused by v3."""
        return SEMANTIC_CONFIG.evidence_record()

    def as_record(self) -> dict[str, object]:
        """Return the complete immutable v3 construction configuration."""
        return {
            "balance_stage": self.balance_stage,
            "cluster_assignment": self.cluster_assignment,
            "cluster_count": self.cluster_count,
            "clustering_method": self.clustering_method,
            "construction_numpy_version": self.construction_numpy_version,
            "encoder_evidence": self.evidence_record(),
            "maximum_centroid_iterations": self.maximum_centroid_iterations,
            "maximum_centroid_pair_cosine": self.maximum_centroid_pair_cosine,
            "maximum_context_silhouette": self.maximum_context_silhouette,
            "maximum_exclusion_passes": self.maximum_exclusion_passes,
            "minimum_calibration_reference_words": self.minimum_calibration_reference_words,
            "minimum_cluster_margin": self.minimum_cluster_margin,
            "minimum_contexts_per_word": self.minimum_contexts_per_word,
            "minimum_nouns_per_cluster": self.minimum_nouns_per_cluster,
            "minimum_retained_token_fraction": self.minimum_retained_token_fraction,
            "minimum_verbs_per_cluster": self.minimum_verbs_per_cluster,
            "representative_contexts_per_cluster": self.representative_contexts_per_cluster,
            "role_calibration_alpha": self.role_calibration_alpha,
            "role_calibration_fold_count": self.role_calibration_fold_count,
            "role_calibration_method": self.role_calibration_method,
            "role_calibration_namespace": self.role_calibration_namespace,
            "role_calibration_source_benchmark_id": self.role_calibration_source_benchmark_id,
            "role_calibration_source_failure_sha256": self.role_calibration_source_failure_sha256,
            "role_margin_quantile": self.role_margin_quantile,
            "semantic_word_weighting": self.semantic_word_weighting,
            "version": self.version,
            "word_vector_anchor_mode": self.word_vector_anchor_mode,
        }


V3_SEMANTIC_CONFIG = V3SemanticConstructionConfig()


def v3_semantic_config_from_record(
    record: dict[str, object],
) -> V3SemanticConstructionConfig:
    """Reconstruct a strictly shaped v3 semantic configuration."""
    if set(record) != set(V3_SEMANTIC_CONFIG.as_record()):
        raise ValueError("semantic-v3 configuration fields changed")
    if record.get("encoder_evidence") != SEMANTIC_CONFIG.evidence_record():
        raise ValueError("semantic-v3 encoder-evidence contract changed")

    def integer(name: str) -> int:
        value = record[name]
        if type(value) is not int:
            raise ValueError(f"semantic-v3 config {name} must be an integer")
        return value

    def number(name: str) -> float:
        value = record[name]
        if type(value) not in (int, float):
            raise ValueError(f"semantic-v3 config {name} must be numeric")
        return float(value)

    def text(name: str) -> str:
        value = record[name]
        if type(value) is not str:
            raise ValueError(f"semantic-v3 config {name} must be text")
        return value

    return V3SemanticConstructionConfig(
        version=text("version"),
        role_margin_quantile=number("role_margin_quantile"),
        role_calibration_method=text("role_calibration_method"),
        role_calibration_namespace=text("role_calibration_namespace"),
        role_calibration_source_benchmark_id=text(
            "role_calibration_source_benchmark_id"
        ),
        role_calibration_source_failure_sha256=text(
            "role_calibration_source_failure_sha256"
        ),
        role_calibration_fold_count=integer("role_calibration_fold_count"),
        role_calibration_alpha=number("role_calibration_alpha"),
        minimum_calibration_reference_words=integer(
            "minimum_calibration_reference_words"
        ),
        minimum_contexts_per_word=integer("minimum_contexts_per_word"),
        maximum_context_silhouette=number("maximum_context_silhouette"),
        word_vector_anchor_mode=text("word_vector_anchor_mode"),
        cluster_count=integer("cluster_count"),
        clustering_method=text("clustering_method"),
        cluster_assignment=text("cluster_assignment"),
        semantic_word_weighting=text("semantic_word_weighting"),
        balance_stage=text("balance_stage"),
        construction_numpy_version=text("construction_numpy_version"),
        minimum_cluster_margin=number("minimum_cluster_margin"),
        maximum_centroid_iterations=integer("maximum_centroid_iterations"),
        maximum_exclusion_passes=integer("maximum_exclusion_passes"),
        minimum_nouns_per_cluster=integer("minimum_nouns_per_cluster"),
        minimum_verbs_per_cluster=integer("minimum_verbs_per_cluster"),
        maximum_centroid_pair_cosine=number("maximum_centroid_pair_cosine"),
        minimum_retained_token_fraction=number("minimum_retained_token_fraction"),
        representative_contexts_per_cluster=integer(
            "representative_contexts_per_cluster"
        ),
    )


@dataclass(frozen=True, slots=True)
class V3SemanticWord:
    """Complete v3 semantic metrics and final disposition for one word."""

    role: Role
    word: str
    token_mass: int
    context_count: int
    calibration_fold: int | None
    calibration_reference_count: int | None
    role_margin_q10: float | None
    role_conformal_p: float | None
    role_rejection_cutoff: float | None
    context_silhouette: float | None
    cluster_margin: float | None
    cluster: int | None
    exclusion_reason: V3ExclusionReason | None
    vector: tuple[float, ...] | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("semantic-v3 word requires a role and word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("semantic-v3 word mass must be positive")
        if type(self.context_count) is not int or self.context_count < 0:
            raise ValueError("semantic-v3 word context count must be nonnegative")
        retained = self.exclusion_reason is None
        if retained != (self.cluster is not None and self.vector is not None):
            raise ValueError("retained semantic-v3 words require a cluster and vector")
        if self.cluster is not None and self.cluster < 0:
            raise ValueError("semantic-v3 word cluster must be nonnegative")
        metrics = (
            self.role_margin_q10,
            self.role_conformal_p,
            self.role_rejection_cutoff,
            self.context_silhouette,
            self.cluster_margin,
        )
        if any(value is not None and not isfinite(value) for value in metrics):
            raise ValueError("semantic-v3 word metrics must be finite")
        if self.role_conformal_p is not None and not 0.0 < self.role_conformal_p <= 1.0:
            raise ValueError("semantic-v3 role p-value must lie in (0, 1]")
        if self.vector is not None and (
            not self.vector
            or any(not isfinite(value) for value in self.vector)
            or not np.isclose(
                np.linalg.norm(np.asarray(self.vector, dtype=np.float64)),
                1.0,
                atol=1e-5,
            )
        ):
            raise ValueError("semantic-v3 word vector must be normalized")

    def as_record(self) -> dict[str, object]:
        """Return this word's complete persisted audit record."""
        return {
            "calibration_fold": self.calibration_fold,
            "calibration_reference_count": self.calibration_reference_count,
            "cluster": self.cluster,
            "cluster_margin": self.cluster_margin,
            "context_count": self.context_count,
            "context_silhouette": self.context_silhouette,
            "exclusion_reason": self.exclusion_reason,
            "role": self.role,
            "role_conformal_p": self.role_conformal_p,
            "role_margin_q10": self.role_margin_q10,
            "role_rejection_cutoff": self.role_rejection_cutoff,
            "token_mass": self.token_mass,
            "vector": None if self.vector is None else list(self.vector),
            "word": self.word,
        }


@dataclass(frozen=True, slots=True)
class V3SemanticCatalog:
    """One authenticated semantic-v3 word catalog."""

    root: Path
    catalog_sha256: str
    evidence_sha256: str
    encoder_identity: EncoderIdentity
    config: V3SemanticConstructionConfig
    calibration: tuple[RoleCalibrationReference, ...]
    words: tuple[V3SemanticWord, ...]
    clusters: tuple[SemanticCluster, ...]
    retained_token_count: int
    nonconstruction_token_count: int
    parent_catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.catalog_sha256, "semantic-v3 catalog identity")
        require_sha256(self.evidence_sha256, "semantic-v3 evidence identity")
        if self.parent_catalog_sha256 is not None:
            require_sha256(self.parent_catalog_sha256, "semantic-v3 parent identity")
        identities = tuple((item.role, item.word) for item in self.words)
        if identities != tuple(sorted(identities)):
            raise ValueError("semantic-v3 words must be canonically ordered")
        expected_clusters = tuple(
            (role, index)
            for role in ("noun", "verb")
            for index in range(self.config.cluster_count)
        )
        if tuple((item.role, item.index) for item in self.clusters) != expected_clusters:
            raise ValueError("semantic-v3 clusters are incomplete or unordered")
        if not 0 < self.retained_token_count <= self.nonconstruction_token_count:
            raise ValueError("semantic-v3 retained mass is invalid")

    @property
    def retained_token_fraction(self) -> float:
        """Return archive token mass retained after both-role exclusions."""
        return self.retained_token_count / self.nonconstruction_token_count

    def word_cluster(self, role: Role) -> dict[str, int]:
        """Return retained word-to-cluster assignments for one role."""
        return {
            item.word: item.cluster
            for item in self.words
            if item.role == role and item.cluster is not None
        }


@dataclass(frozen=True, slots=True)
class V3CatalogFailureArtifact:
    """Authenticated evidence for a failed semantic-v3 construction."""

    root: Path
    failure_sha256: str
    evidence_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir() or not self.reason:
            raise ValueError("semantic-v3 failure artifact is malformed")
        require_sha256(self.failure_sha256, "semantic-v3 failure identity")
        require_sha256(self.evidence_sha256, "semantic-v3 failure evidence identity")


__all__ = [
    "BoundaryPassMetric",
    "RoleCalibrationReference",
    "V3_BENCHMARK_ID",
    "V3_CATALOG_FAILURE_FORMAT",
    "V3_CATALOG_FORMAT",
    "V3_REUSED_EVIDENCE_SHA256",
    "V3_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256",
    "V3_SCHEMA_VERSION",
    "V3_SEMANTIC_CONFIG",
    "V3_SEMANTIC_CONFIG_VERSION",
    "V3CatalogFailureArtifact",
    "V3ExclusionReason",
    "V3SemanticCatalog",
    "V3SemanticConstructionConfig",
    "V3SemanticWord",
    "v3_semantic_config_from_record",
]
