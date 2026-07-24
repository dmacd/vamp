"""Frozen construction contracts for TinyWorlds-P semantic-v2."""

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


V2_BENCHMARK_ID = "tinyworlds-p-semantic-v2"
V2_SEMANTIC_CONFIG_VERSION = "tinyworlds-p-semantic-construction-v2"
V2_CATALOG_FORMAT = "tinyworlds-p-semantic-v2-catalog"
V2_CATALOG_FAILURE_FORMAT = "tinyworlds-p-semantic-v2-catalog-failure"
V2_SCHEMA_VERSION = 1
V2_REUSED_EVIDENCE_SHA256 = (
    "efd86b448ad78580380ead5e57e809383846b287cd4671746b1cee250e47f434"
)

V2ExclusionReason = Literal[
    "insufficient_contexts",
    "calibrated_role_outlier",
    "multiple_realized_senses",
    "cluster_boundary_margin",
]


@dataclass(frozen=True, slots=True)
class V2SemanticConstructionConfig:
    """Every behavior-changing v2 calibration and clustering choice."""

    version: str = V2_SEMANTIC_CONFIG_VERSION
    role_margin_quantile: float = 0.10
    role_calibration_method: str = "word-fold-cross-conformal-lower-tail-v1"
    role_calibration_namespace: str = "role-calibration-fold-v1"
    role_calibration_fold_count: int = 5
    role_calibration_alpha: float = 0.05
    minimum_calibration_reference_words: int = 48
    minimum_contexts_per_word: int = 32
    maximum_context_silhouette: float = 0.20
    word_vector_anchor_mode: str = "target-role-centroid"
    minimum_cluster_margin: float = 0.03
    cluster_count: int = 8
    minimum_cluster_mass_fraction: float = 0.90
    maximum_cluster_mass_fraction: float = 1.10
    maximum_centroid_iterations: int = 100
    maximum_exclusion_passes: int = 5
    assignment_dead_end_repair: str = "single-prior-word-reassignment"
    minimum_nouns_per_cluster: int = 32
    minimum_verbs_per_cluster: int = 12
    maximum_centroid_pair_cosine: float = 0.90
    minimum_retained_token_fraction: float = 0.40
    representative_contexts_per_cluster: int = 3

    def __post_init__(self) -> None:
        if self.version != V2_SEMANTIC_CONFIG_VERSION:
            raise ValueError("unsupported semantic-v2 construction config version")
        if self.role_calibration_method != "word-fold-cross-conformal-lower-tail-v1":
            raise ValueError("semantic-v2 requires the frozen cross-conformal method")
        if self.role_calibration_namespace != "role-calibration-fold-v1":
            raise ValueError("semantic-v2 requires the frozen calibration namespace")
        if self.word_vector_anchor_mode != "target-role-centroid":
            raise ValueError("semantic-v2 preserves the target-role semantic anchor")
        if self.assignment_dead_end_repair != "single-prior-word-reassignment":
            raise ValueError("semantic-v2 requires the frozen discrete assignment repair")
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
            raise ValueError("semantic-v2 integer choices must be positive")
        if self.role_calibration_fold_count < 2:
            raise ValueError("cross-conformal calibration requires at least two folds")
        unit_values = (
            self.role_margin_quantile,
            self.role_calibration_alpha,
            self.maximum_context_silhouette,
            self.minimum_cluster_margin,
            self.maximum_centroid_pair_cosine,
            self.minimum_retained_token_fraction,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_values):
            raise ValueError("semantic-v2 fractions must be finite and lie in [0, 1]")
        if not 0.0 < self.role_calibration_alpha < 1.0:
            raise ValueError("semantic-v2 conformal alpha must lie in (0, 1)")
        if not self.minimum_cluster_mass_fraction < 1.0 < self.maximum_cluster_mass_fraction:
            raise ValueError("semantic-v2 cluster mass bounds must straddle one")
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.minimum_cluster_mass_fraction,
                self.maximum_cluster_mass_fraction,
            )
        ):
            raise ValueError("semantic-v2 cluster mass bounds must be finite and positive")

    def evidence_record(self) -> dict[str, object]:
        """Return the exact v1 encoder-evidence contract reused by v2."""
        return SEMANTIC_CONFIG.evidence_record()

    def as_record(self) -> dict[str, object]:
        """Return the complete immutable v2 construction configuration."""
        return {
            "assignment_dead_end_repair": self.assignment_dead_end_repair,
            "cluster_count": self.cluster_count,
            "encoder_evidence": self.evidence_record(),
            "maximum_centroid_iterations": self.maximum_centroid_iterations,
            "maximum_centroid_pair_cosine": self.maximum_centroid_pair_cosine,
            "maximum_context_silhouette": self.maximum_context_silhouette,
            "maximum_exclusion_passes": self.maximum_exclusion_passes,
            "maximum_cluster_mass_fraction": self.maximum_cluster_mass_fraction,
            "minimum_calibration_reference_words": self.minimum_calibration_reference_words,
            "minimum_cluster_margin": self.minimum_cluster_margin,
            "minimum_cluster_mass_fraction": self.minimum_cluster_mass_fraction,
            "minimum_contexts_per_word": self.minimum_contexts_per_word,
            "minimum_nouns_per_cluster": self.minimum_nouns_per_cluster,
            "minimum_retained_token_fraction": self.minimum_retained_token_fraction,
            "minimum_verbs_per_cluster": self.minimum_verbs_per_cluster,
            "representative_contexts_per_cluster": self.representative_contexts_per_cluster,
            "role_calibration_alpha": self.role_calibration_alpha,
            "role_calibration_fold_count": self.role_calibration_fold_count,
            "role_calibration_method": self.role_calibration_method,
            "role_calibration_namespace": self.role_calibration_namespace,
            "role_margin_quantile": self.role_margin_quantile,
            "version": self.version,
            "word_vector_anchor_mode": self.word_vector_anchor_mode,
        }


V2_SEMANTIC_CONFIG = V2SemanticConstructionConfig()


def v2_semantic_config_from_record(
    record: dict[str, object],
) -> V2SemanticConstructionConfig:
    """Reconstruct a strictly shaped v2 semantic configuration."""
    if set(record) != set(V2_SEMANTIC_CONFIG.as_record()):
        raise ValueError("semantic-v2 configuration fields changed")
    if record.get("encoder_evidence") != SEMANTIC_CONFIG.evidence_record():
        raise ValueError("semantic-v2 encoder-evidence contract changed")

    def integer(name: str) -> int:
        value = record[name]
        if type(value) is not int:
            raise ValueError(f"semantic-v2 config {name} must be an integer")
        return value

    def number(name: str) -> float:
        value = record[name]
        if type(value) not in (int, float):
            raise ValueError(f"semantic-v2 config {name} must be numeric")
        return float(value)

    def text(name: str) -> str:
        value = record[name]
        if type(value) is not str:
            raise ValueError(f"semantic-v2 config {name} must be text")
        return value

    return V2SemanticConstructionConfig(
        version=text("version"),
        role_margin_quantile=number("role_margin_quantile"),
        role_calibration_method=text("role_calibration_method"),
        role_calibration_namespace=text("role_calibration_namespace"),
        role_calibration_fold_count=integer("role_calibration_fold_count"),
        role_calibration_alpha=number("role_calibration_alpha"),
        minimum_calibration_reference_words=integer(
            "minimum_calibration_reference_words"
        ),
        minimum_contexts_per_word=integer("minimum_contexts_per_word"),
        maximum_context_silhouette=number("maximum_context_silhouette"),
        word_vector_anchor_mode=text("word_vector_anchor_mode"),
        minimum_cluster_margin=number("minimum_cluster_margin"),
        cluster_count=integer("cluster_count"),
        minimum_cluster_mass_fraction=number("minimum_cluster_mass_fraction"),
        maximum_cluster_mass_fraction=number("maximum_cluster_mass_fraction"),
        maximum_centroid_iterations=integer("maximum_centroid_iterations"),
        maximum_exclusion_passes=integer("maximum_exclusion_passes"),
        assignment_dead_end_repair=text("assignment_dead_end_repair"),
        minimum_nouns_per_cluster=integer("minimum_nouns_per_cluster"),
        minimum_verbs_per_cluster=integer("minimum_verbs_per_cluster"),
        maximum_centroid_pair_cosine=number("maximum_centroid_pair_cosine"),
        minimum_retained_token_fraction=number("minimum_retained_token_fraction"),
        representative_contexts_per_cluster=integer(
            "representative_contexts_per_cluster"
        ),
    )


@dataclass(frozen=True, slots=True)
class RoleCalibrationReference:
    """One held-out fold's same-role conformal reference distribution."""

    role: Role
    fold: int
    reference_count: int
    rejection_cutoff: float | None

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or self.fold < 0:
            raise ValueError("role calibration reference is malformed")
        if type(self.reference_count) is not int or self.reference_count <= 0:
            raise ValueError("role calibration reference count must be positive")
        if self.rejection_cutoff is not None and not isfinite(self.rejection_cutoff):
            raise ValueError("role calibration rejection cutoff must be finite")

    def as_record(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "reference_count": self.reference_count,
            "rejection_cutoff": self.rejection_cutoff,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class CalibratedRoleScore:
    """Cross-fitted role evidence for one declared role word."""

    role: Role
    word: str
    fold: int
    raw_margin: float
    reference_count: int
    conformal_p: float
    rejection_cutoff: float | None

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word or self.fold < 0:
            raise ValueError("calibrated role score identity is malformed")
        if type(self.reference_count) is not int or self.reference_count <= 0:
            raise ValueError("calibrated role score requires reference words")
        if not isfinite(self.raw_margin) or not 0.0 < self.conformal_p <= 1.0:
            raise ValueError("calibrated role score values are invalid")
        if self.rejection_cutoff is not None and not isfinite(self.rejection_cutoff):
            raise ValueError("calibrated role cutoff must be finite")


@dataclass(frozen=True, slots=True)
class V2SemanticWord:
    """Complete v2 semantic metrics and final disposition for one word."""

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
    exclusion_reason: V2ExclusionReason | None
    vector: tuple[float, ...] | None = field(repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("semantic-v2 word requires a role and word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("semantic-v2 word mass must be positive")
        if type(self.context_count) is not int or self.context_count < 0:
            raise ValueError("semantic-v2 word context count must be nonnegative")
        retained = self.exclusion_reason is None
        if retained != (self.cluster is not None and self.vector is not None):
            raise ValueError("retained semantic-v2 words require a cluster and vector")
        if self.cluster is not None and self.cluster < 0:
            raise ValueError("semantic-v2 word cluster must be nonnegative")
        for value in (
            self.role_margin_q10,
            self.role_conformal_p,
            self.role_rejection_cutoff,
            self.context_silhouette,
            self.cluster_margin,
        ):
            if value is not None and not isfinite(value):
                raise ValueError("semantic-v2 word metrics must be finite")
        if self.role_conformal_p is not None and not 0.0 < self.role_conformal_p <= 1.0:
            raise ValueError("semantic-v2 role p-value must lie in (0, 1]")
        if self.vector is not None and (
            not self.vector
            or any(not isfinite(value) for value in self.vector)
            or not np.isclose(
                np.linalg.norm(np.asarray(self.vector, dtype=np.float64)),
                1.0,
                atol=1e-5,
            )
        ):
            raise ValueError("semantic-v2 word vector must be normalized")

    def as_record(self) -> dict[str, object]:
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
class BoundaryPassMetric:
    """Auditable state from one boundary-exclusion clustering pass."""

    role: Role
    pass_index: int
    input_word_count: int
    failing_word_count: int
    cluster_masses: tuple[int, ...]
    minimum_margin: float
    margin_q10: float
    median_margin: float

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or self.pass_index < 0:
            raise ValueError("boundary-pass identity is invalid")
        if self.input_word_count <= 0 or not 0 <= self.failing_word_count <= self.input_word_count:
            raise ValueError("boundary-pass counts are invalid")
        if not self.cluster_masses or any(value <= 0 for value in self.cluster_masses):
            raise ValueError("boundary-pass cluster masses must be positive")
        if any(
            not isfinite(value)
            for value in (self.minimum_margin, self.margin_q10, self.median_margin)
        ):
            raise ValueError("boundary-pass margins must be finite")

    def as_record(self) -> dict[str, object]:
        return {
            "cluster_masses": list(self.cluster_masses),
            "failing_word_count": self.failing_word_count,
            "input_word_count": self.input_word_count,
            "margin_q10": self.margin_q10,
            "median_margin": self.median_margin,
            "minimum_margin": self.minimum_margin,
            "pass_index": self.pass_index,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class V2SemanticCatalog:
    """One authenticated semantic-v2 word catalog."""

    root: Path
    catalog_sha256: str
    evidence_sha256: str
    encoder_identity: EncoderIdentity
    config: V2SemanticConstructionConfig
    calibration: tuple[RoleCalibrationReference, ...]
    words: tuple[V2SemanticWord, ...]
    clusters: tuple[SemanticCluster, ...]
    retained_token_count: int
    nonconstruction_token_count: int
    parent_catalog_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.catalog_sha256, "semantic-v2 catalog identity")
        require_sha256(self.evidence_sha256, "semantic-v2 evidence identity")
        if self.parent_catalog_sha256 is not None:
            require_sha256(self.parent_catalog_sha256, "semantic-v2 parent identity")
        if tuple((item.role, item.word) for item in self.words) != tuple(
            sorted((item.role, item.word) for item in self.words)
        ):
            raise ValueError("semantic-v2 words must be canonically ordered")
        expected_clusters = tuple(
            (role, index)
            for role in ("noun", "verb")
            for index in range(self.config.cluster_count)
        )
        if tuple((item.role, item.index) for item in self.clusters) != expected_clusters:
            raise ValueError("semantic-v2 clusters are incomplete or unordered")
        if not 0 < self.retained_token_count <= self.nonconstruction_token_count:
            raise ValueError("semantic-v2 retained mass is invalid")

    @property
    def retained_token_fraction(self) -> float:
        return self.retained_token_count / self.nonconstruction_token_count

    def word_cluster(self, role: Role) -> dict[str, int]:
        return {
            item.word: item.cluster
            for item in self.words
            if item.role == role and item.cluster is not None
        }


@dataclass(frozen=True, slots=True)
class V2CatalogFailureArtifact:
    """Authenticated evidence for a failed semantic-v2 construction."""

    root: Path
    failure_sha256: str
    evidence_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir() or not self.reason:
            raise ValueError("semantic-v2 failure artifact is malformed")
        require_sha256(self.failure_sha256, "semantic-v2 failure identity")
        require_sha256(self.evidence_sha256, "semantic-v2 failure evidence identity")


__all__ = [
    "BoundaryPassMetric",
    "CalibratedRoleScore",
    "RoleCalibrationReference",
    "V2_BENCHMARK_ID",
    "V2_CATALOG_FAILURE_FORMAT",
    "V2_CATALOG_FORMAT",
    "V2_SCHEMA_VERSION",
    "V2_REUSED_EVIDENCE_SHA256",
    "V2_SEMANTIC_CONFIG",
    "V2_SEMANTIC_CONFIG_VERSION",
    "V2CatalogFailureArtifact",
    "V2ExclusionReason",
    "V2SemanticCatalog",
    "V2SemanticConstructionConfig",
    "V2SemanticWord",
    "v2_semantic_config_from_record",
]
