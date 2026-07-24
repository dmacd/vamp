"""Frozen one-shot, fixed-centroid contracts for TinyWorlds-P semantic-v4."""

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
from apm.data.text.tinyworlds_p_semantic.v3_contracts import V3_BENCHMARK_ID


V4_BENCHMARK_ID = "tinyworlds-p-semantic-v4"
V4_SEMANTIC_CONFIG_VERSION = "tinyworlds-p-semantic-construction-v4"
V4_CATALOG_FORMAT = "tinyworlds-p-semantic-v4-catalog"
V4_CATALOG_FAILURE_FORMAT = "tinyworlds-p-semantic-v4-catalog-failure"
V4_SCHEMA_VERSION = 1
V4_REUSED_EVIDENCE_SHA256 = (
    "efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434"
)
V4_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256 = (
    "23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25"
)
V4_SOURCE_V3_FAILURE_SHA256 = (
    "ae418bfb73cc0e278f1ba9204c81d101e0b95e9cf050597a491d21489cde6146"
)

V4ExclusionReason = Literal[
    "insufficient_contexts",
    "calibrated_role_outlier",
    "role_calibration_failure",
    "multiple_realized_senses",
    "cluster_boundary_margin",
    "semantic_fit_failure",
]


@dataclass(frozen=True, slots=True)
class V4SemanticConstructionConfig:
    """Every behavior-changing v4 calibration and fixed-fit choice."""

    version: str = V4_SEMANTIC_CONFIG_VERSION
    source_v3_failure_sha256: str = V4_SOURCE_V3_FAILURE_SHA256
    role_margin_quantile: float = 0.10
    role_calibration_method: str = "word-fold-cross-conformal-lower-tail-v1"
    role_calibration_namespace: str = "role-calibration-fold-v1"
    role_calibration_source_benchmark_id: str = V2_BENCHMARK_ID
    role_calibration_source_failure_sha256: str = (
        V4_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256
    )
    role_calibration_fold_count: int = 5
    role_calibration_alpha: float = 0.05
    minimum_calibration_reference_words: int = 48
    minimum_contexts_per_word: int = 32
    maximum_context_silhouette: float = 0.20
    word_vector_anchor_mode: str = "target-role-centroid"
    cluster_count: int = 8
    clustering_method: str = "v3-pass-zero-unweighted-spherical-kmeans-v1"
    fit_hash_benchmark_id: str = V3_BENCHMARK_ID
    cluster_assignment: str = "nearest-centroid-cosine"
    semantic_word_weighting: str = "uniform"
    boundary_method: str = "single-screen-frozen-fit-centroids-v1"
    boundary_screen_count: int = 1
    maximum_exclusion_passes: int = 0
    centroid_update_after_screen: bool = False
    balance_stage: str = "partition-story-allocation"
    construction_numpy_version: str = "1.26.4"
    minimum_cluster_margin: float = 0.03
    maximum_centroid_iterations: int = 100
    minimum_nouns_per_cluster: int = 32
    minimum_verbs_per_cluster: int = 12
    maximum_centroid_pair_cosine: float = 0.90
    minimum_retained_token_fraction: float = 0.40
    representative_contexts_per_cluster: int = 3

    def __post_init__(self) -> None:
        fixed = {
            "version": V4_SEMANTIC_CONFIG_VERSION,
            "source_v3_failure_sha256": V4_SOURCE_V3_FAILURE_SHA256,
            "role_calibration_method": "word-fold-cross-conformal-lower-tail-v1",
            "role_calibration_namespace": "role-calibration-fold-v1",
            "role_calibration_source_benchmark_id": V2_BENCHMARK_ID,
            "role_calibration_source_failure_sha256": (
                V4_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256
            ),
            "word_vector_anchor_mode": "target-role-centroid",
            "clustering_method": "v3-pass-zero-unweighted-spherical-kmeans-v1",
            "fit_hash_benchmark_id": V3_BENCHMARK_ID,
            "cluster_assignment": "nearest-centroid-cosine",
            "semantic_word_weighting": "uniform",
            "boundary_method": "single-screen-frozen-fit-centroids-v1",
            "boundary_screen_count": 1,
            "maximum_exclusion_passes": 0,
            "centroid_update_after_screen": False,
            "balance_stage": "partition-story-allocation",
            "construction_numpy_version": "1.26.4",
        }
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError("semantic-v4 fixed construction choice changed")
        positive_integers = (
            self.role_calibration_fold_count,
            self.minimum_calibration_reference_words,
            self.minimum_contexts_per_word,
            self.cluster_count,
            self.maximum_centroid_iterations,
            self.minimum_nouns_per_cluster,
            self.minimum_verbs_per_cluster,
            self.representative_contexts_per_cluster,
        )
        if any(type(value) is not int or value <= 0 for value in positive_integers):
            raise ValueError("semantic-v4 integer choices must be positive")
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
            raise ValueError("semantic-v4 fractions must be finite and lie in [0, 1]")
        if not 0.0 < self.role_calibration_alpha < 1.0:
            raise ValueError("semantic-v4 conformal alpha must lie in (0, 1)")

    def evidence_record(self) -> dict[str, object]:
        """Return the exact v1 encoder-evidence contract reused by v4."""
        return SEMANTIC_CONFIG.evidence_record()

    @property
    def construction_modulus(self) -> int:
        """Expose the permanent v1 construction-slice modulus."""
        return int(self.evidence_record()["construction_modulus"])

    @property
    def construction_residue(self) -> int:
        """Expose the permanent v1 construction-slice residue."""
        return int(self.evidence_record()["construction_residue"])

    def as_record(self) -> dict[str, object]:
        """Return the complete immutable v4 construction configuration."""
        return {
            "balance_stage": self.balance_stage,
            "boundary_method": self.boundary_method,
            "boundary_screen_count": self.boundary_screen_count,
            "centroid_update_after_screen": self.centroid_update_after_screen,
            "cluster_assignment": self.cluster_assignment,
            "cluster_count": self.cluster_count,
            "clustering_method": self.clustering_method,
            "construction_numpy_version": self.construction_numpy_version,
            "encoder_evidence": self.evidence_record(),
            "fit_hash_benchmark_id": self.fit_hash_benchmark_id,
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
            "source_v3_failure_sha256": self.source_v3_failure_sha256,
            "version": self.version,
            "word_vector_anchor_mode": self.word_vector_anchor_mode,
        }


V4_SEMANTIC_CONFIG = V4SemanticConstructionConfig()


def v4_semantic_config_from_record(
    record: dict[str, object],
) -> V4SemanticConstructionConfig:
    """Reconstruct a strictly shaped v4 semantic configuration."""
    if set(record) != set(V4_SEMANTIC_CONFIG.as_record()):
        raise ValueError("semantic-v4 configuration fields changed")
    if record.get("encoder_evidence") != SEMANTIC_CONFIG.evidence_record():
        raise ValueError("semantic-v4 encoder-evidence contract changed")

    def integer(name: str) -> int:
        value = record[name]
        if type(value) is not int:
            raise ValueError(f"semantic-v4 config {name} must be an integer")
        return value

    def number(name: str) -> float:
        value = record[name]
        if type(value) not in (int, float):
            raise ValueError(f"semantic-v4 config {name} must be numeric")
        return float(value)

    def text(name: str) -> str:
        value = record[name]
        if type(value) is not str:
            raise ValueError(f"semantic-v4 config {name} must be text")
        return value

    centroid_update = record["centroid_update_after_screen"]
    if type(centroid_update) is not bool:
        raise ValueError("semantic-v4 centroid-update choice must be boolean")
    return V4SemanticConstructionConfig(
        version=text("version"),
        source_v3_failure_sha256=text("source_v3_failure_sha256"),
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
        fit_hash_benchmark_id=text("fit_hash_benchmark_id"),
        cluster_assignment=text("cluster_assignment"),
        semantic_word_weighting=text("semantic_word_weighting"),
        boundary_method=text("boundary_method"),
        boundary_screen_count=integer("boundary_screen_count"),
        maximum_exclusion_passes=integer("maximum_exclusion_passes"),
        centroid_update_after_screen=centroid_update,
        balance_stage=text("balance_stage"),
        construction_numpy_version=text("construction_numpy_version"),
        minimum_cluster_margin=number("minimum_cluster_margin"),
        maximum_centroid_iterations=integer("maximum_centroid_iterations"),
        minimum_nouns_per_cluster=integer("minimum_nouns_per_cluster"),
        minimum_verbs_per_cluster=integer("minimum_verbs_per_cluster"),
        maximum_centroid_pair_cosine=number("maximum_centroid_pair_cosine"),
        minimum_retained_token_fraction=number("minimum_retained_token_fraction"),
        representative_contexts_per_cluster=integer(
            "representative_contexts_per_cluster"
        ),
    )


@dataclass(frozen=True, slots=True)
class V4SemanticWord:
    """Complete v4 metrics, frozen-fit assignment, and final disposition."""

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
    fit_cluster: int | None
    cluster_margin: float | None
    cluster: int | None
    exclusion_reason: V4ExclusionReason | None
    vector: tuple[float, ...] | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("semantic-v4 word requires a role and word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("semantic-v4 word mass must be positive")
        if type(self.context_count) is not int or self.context_count < 0:
            raise ValueError("semantic-v4 context count must be nonnegative")
        for value in (self.fit_cluster, self.cluster):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("semantic-v4 cluster must be nonnegative")
        metrics = (
            self.role_margin_q10,
            self.role_conformal_p,
            self.role_rejection_cutoff,
            self.context_silhouette,
            self.cluster_margin,
        )
        if any(value is not None and not isfinite(value) for value in metrics):
            raise ValueError("semantic-v4 word metrics must be finite")
        if self.role_conformal_p is not None and not 0.0 < self.role_conformal_p <= 1.0:
            raise ValueError("semantic-v4 role p-value must lie in (0, 1]")
        if self.vector is not None and (
            not self.vector
            or any(not isfinite(value) for value in self.vector)
            or not np.isclose(
                np.linalg.norm(np.asarray(self.vector, dtype=np.float64)),
                1.0,
                atol=1e-5,
            )
        ):
            raise ValueError("semantic-v4 word vector must be normalized")
        if self.fit_cluster is None:
            if self.cluster_margin is not None or self.cluster is not None:
                raise ValueError("semantic-v4 unfitted words cannot have cluster data")
        elif self.vector is None or self.cluster_margin is None:
            raise ValueError("semantic-v4 fitted words require a vector and margin")
        if self.exclusion_reason is None:
            if self.cluster is None or self.cluster != self.fit_cluster:
                raise ValueError("retained semantic-v4 words keep their fit cluster")
        elif self.cluster is not None:
            raise ValueError("excluded semantic-v4 words cannot have a final cluster")
        if self.exclusion_reason == "cluster_boundary_margin" and self.fit_cluster is None:
            raise ValueError("boundary exclusions require a frozen-fit assignment")
        if self.exclusion_reason == "semantic_fit_failure" and (
            self.vector is None or self.fit_cluster is not None
        ):
            raise ValueError("fit failures retain only their candidate vector")

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
            "fit_cluster": self.fit_cluster,
            "role": self.role,
            "role_conformal_p": self.role_conformal_p,
            "role_margin_q10": self.role_margin_q10,
            "role_rejection_cutoff": self.role_rejection_cutoff,
            "token_mass": self.token_mass,
            "vector": None if self.vector is None else list(self.vector),
            "word": self.word,
        }


@dataclass(frozen=True, slots=True)
class V4SemanticCatalog:
    """One authenticated semantic-v4 frozen-centroid word catalog."""

    root: Path
    catalog_sha256: str
    evidence_sha256: str
    encoder_identity: EncoderIdentity
    config: V4SemanticConstructionConfig
    calibration: tuple[RoleCalibrationReference, ...]
    words: tuple[V4SemanticWord, ...]
    fit_clusters: tuple[SemanticCluster, ...]
    clusters: tuple[SemanticCluster, ...]
    retained_token_count: int
    nonconstruction_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.catalog_sha256, "semantic-v4 catalog identity")
        require_sha256(self.evidence_sha256, "semantic-v4 evidence identity")
        identities = tuple((item.role, item.word) for item in self.words)
        if identities != tuple(sorted(identities)):
            raise ValueError("semantic-v4 words must be canonically ordered")
        expected = tuple(
            (role, index)
            for role in ("noun", "verb")
            for index in range(self.config.cluster_count)
        )
        if tuple((item.role, item.index) for item in self.fit_clusters) != expected:
            raise ValueError("semantic-v4 fit clusters are incomplete or unordered")
        if tuple((item.role, item.index) for item in self.clusters) != expected:
            raise ValueError("semantic-v4 retained clusters are incomplete or unordered")
        if not 0 < self.retained_token_count <= self.nonconstruction_token_count:
            raise ValueError("semantic-v4 retained mass is invalid")

    @property
    def retained_token_fraction(self) -> float:
        """Return archive token mass retained after both-role exclusions."""
        return self.retained_token_count / self.nonconstruction_token_count

    def word_cluster(self, role: Role) -> dict[str, int]:
        """Return retained word-to-fixed-cluster assignments for one role."""
        return {
            item.word: item.cluster
            for item in self.words
            if item.role == role and item.cluster is not None
        }


@dataclass(frozen=True, slots=True)
class V4CatalogFailureArtifact:
    """Authenticated evidence for a failed semantic-v4 construction."""

    root: Path
    failure_sha256: str
    evidence_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir() or not self.reason:
            raise ValueError("semantic-v4 failure artifact is malformed")
        require_sha256(self.failure_sha256, "semantic-v4 failure identity")
        require_sha256(self.evidence_sha256, "semantic-v4 failure evidence identity")


__all__ = [
    "BoundaryPassMetric",
    "RoleCalibrationReference",
    "V4_BENCHMARK_ID",
    "V4_CATALOG_FAILURE_FORMAT",
    "V4_CATALOG_FORMAT",
    "V4_REUSED_EVIDENCE_SHA256",
    "V4_ROLE_CALIBRATION_SOURCE_FAILURE_SHA256",
    "V4_SCHEMA_VERSION",
    "V4_SEMANTIC_CONFIG",
    "V4_SEMANTIC_CONFIG_VERSION",
    "V4_SOURCE_V3_FAILURE_SHA256",
    "V4CatalogFailureArtifact",
    "V4ExclusionReason",
    "V4SemanticCatalog",
    "V4SemanticConstructionConfig",
    "V4SemanticWord",
    "v4_semantic_config_from_record",
]
