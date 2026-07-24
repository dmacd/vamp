"""Calibrated semantic-v2 screening, clustering, publication, and loading."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import cast

import numpy as np

from apm.data.text.tinyworlds_p_semantic.catalog import (
    WordEvidence,
    load_word_evidence,
)
from apm.data.text.tinyworlds_p_semantic.clustering import (
    BoundaryClustering,
    SemanticGridError,
    SphericalClustering,
    WordVector,
    capacity_constrained_spherical_kmeans,
    compose_word_vector,
    deterministic_two_means_silhouette,
    role_margin_quantile,
    validate_cluster_quality,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    EncoderIdentity,
    ModelFile,
    Role,
    SemanticCluster,
    SemanticEvidenceArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.role_calibration import (
    RoleCalibrationError,
    calibrate_role_margins,
)
from apm.data.text.tinyworlds_p_semantic.v2_audit import (
    CalibratedFailureWord,
    render_calibrated_catalog_audits,
    render_calibrated_failure_audits,
)
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    CalibratedRoleScore,
    RoleCalibrationReference,
    V2_BENCHMARK_ID,
    V2_CATALOG_FAILURE_FORMAT,
    V2_CATALOG_FORMAT,
    V2_SCHEMA_VERSION,
    V2CatalogFailureArtifact,
    V2ExclusionReason,
    V2SemanticCatalog,
    V2SemanticConstructionConfig,
    V2SemanticWord,
    v2_semantic_config_from_record,
)


class V2SemanticCatalogError(ValueError):
    """A semantic-v2 catalog or failure bundle is malformed."""


class V2SemanticGridError(ValueError):
    """The frozen semantic-v2 construction failed and published evidence."""


V2CatalogProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class _ScreenedWord:
    evidence: WordEvidence
    calibration: CalibratedRoleScore | None
    silhouette: float | None
    vector: tuple[float, ...] | None
    reason: V2ExclusionReason | None


@dataclass(frozen=True, slots=True)
class _RoleGridFailure(Exception):
    reason: str
    trace: tuple[BoundaryPassMetric, ...]
    boundary_words: frozenset[str]

    def __str__(self) -> str:
        return self.reason


def build_v2_catalog_from_evidence(
    evidence: SemanticEvidenceArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V2SemanticConstructionConfig,
    *,
    parent_catalog: V2SemanticCatalog | None = None,
    progress: V2CatalogProgress | None = None,
) -> V2SemanticCatalog:
    """Reuse authenticated v1 embeddings under the new v2 catalog contract."""
    if evidence.config.evidence_record() != config.evidence_record():
        raise ValueError("semantic-v2 evidence configuration changed")
    _emit(progress, "evidence-load", 0, 1, "loading authenticated cached vectors")
    word_evidence, pair_masses = load_word_evidence(evidence)
    _emit(progress, "evidence-load", 1, 1, "cached word evidence loaded")
    return build_v2_semantic_catalog(
        word_evidence,
        pair_masses,
        evidence.encoder_identity,
        evidence.evidence_sha256,
        evidence.nonconstruction_token_count,
        output_root,
        temporary_directory,
        config,
        parent_catalog=parent_catalog,
        progress=progress,
    )


def build_v2_semantic_catalog(
    word_evidence: Sequence[WordEvidence],
    pair_masses: Mapping[tuple[str, str], int],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V2SemanticConstructionConfig,
    *,
    parent_catalog: V2SemanticCatalog | None = None,
    progress: V2CatalogProgress | None = None,
) -> V2SemanticCatalog:
    """Apply cross-conformal screening and the unchanged fixed semantic grid."""
    canonical = tuple(sorted(word_evidence, key=lambda item: (item.role, item.word)))
    if len({(item.role, item.word) for item in canonical}) != len(canonical):
        raise ValueError("semantic-v2 word evidence contains duplicate role words")
    _validate_input_masses(canonical, pair_masses, nonconstruction_token_count)
    raw_margins = {}
    _emit(progress, "role-scores", 0, len(canonical), "computing raw role margins")
    for completed, item in enumerate(canonical, start=1):
        if item.context_embeddings.shape[0] >= config.minimum_contexts_per_word:
            raw_margins[(item.role, item.word)] = role_margin_quantile(
                item.target_anchor_embeddings,
                item.opposite_anchor_embeddings,
                item.context_embeddings,
                config.role_margin_quantile,
            )
        if completed % 50 == 0 or completed == len(canonical):
            _emit(
                progress,
                "role-scores",
                completed,
                len(canonical),
                "computing raw role margins",
            )
    _emit(progress, "calibration", 0, 1, "fitting held-out-fold role references")
    try:
        calibrated, calibration = calibrate_role_margins(raw_margins, config)
    except RoleCalibrationError as error:
        failure = _publish_failure(
            tuple(
                _ScreenedWord(
                    evidence=item,
                    calibration=None,
                    silhouette=None,
                    vector=None,
                    reason=(
                        "insufficient_contexts"
                        if item.context_embeddings.shape[0]
                        < config.minimum_contexts_per_word
                        else None
                    ),
                )
                for item in canonical
            ),
            (),
            (),
            {},
            pair_masses,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            f"role calibration failed: {error}",
            output_root,
            temporary_directory,
            config,
            progress=progress,
        )
        raise V2SemanticGridError(
            f"role calibration failed: {error}; failure audit: {failure.root}"
        ) from error

    _emit(progress, "calibration", 1, 1, "cross-conformal role calibration complete")

    screened_items = []
    _emit(progress, "screening", 0, len(canonical), "applying calibrated role and sense gates")
    for completed, item in enumerate(canonical, start=1):
        screened_items.append(_screen_word(item, calibrated, config))
        if completed % 50 == 0 or completed == len(canonical):
            _emit(
                progress,
                "screening",
                completed,
                len(canonical),
                "applying calibrated role and sense gates",
            )
    screened = tuple(screened_items)
    candidates = {
        role: tuple(
            WordVector(
                role=role,
                word=item.evidence.word,
                token_mass=item.evidence.token_mass,
                vector=cast(tuple[float, ...], item.vector),
            )
            for item in screened
            if item.evidence.role == role and item.reason is None
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    representative_contexts = {
        (item.evidence.role, item.evidence.word): item.evidence.contexts[
            : config.representative_contexts_per_cluster
        ]
        for item in screened
        if item.evidence.contexts
    }

    results: dict[Role, BoundaryClustering] = {}
    traces: list[BoundaryPassMetric] = []
    boundary_words: dict[Role, set[str]] = {"noun": set(), "verb": set()}
    failures = []
    roles = cast(tuple[Role, Role], ("noun", "verb"))
    cluster_progress_total = len(roles) * (config.maximum_exclusion_passes + 1)
    _emit(progress, "clustering", 0, cluster_progress_total, "starting fixed v2 clusters")
    for role_index, role in enumerate(roles):
        try:
            result, trace = _cluster_role(
                role,
                candidates[role],
                config,
                progress,
                role_index * (config.maximum_exclusion_passes + 1),
                cluster_progress_total,
            )
            results[role] = result
            traces.extend(trace)
            boundary_words[role].update(word for word, _ in result.excluded_margins)
        except _RoleGridFailure as error:
            traces.extend(error.trace)
            boundary_words[role].update(error.boundary_words)
            failures.append(error.reason)
        _emit(
            progress,
            "clustering",
            (role_index + 1) * (config.maximum_exclusion_passes + 1),
            cluster_progress_total,
            f"{role} clustering complete",
        )
    if failures:
        reason = "; ".join(failures)
        failure = _publish_failure(
            screened,
            calibration,
            tuple(traces),
            boundary_words,
            pair_masses,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            reason,
            output_root,
            temporary_directory,
            config,
            representative_contexts,
            progress=progress,
        )
        raise V2SemanticGridError(f"{reason}; failure audit: {failure.root}")

    assignments = {
        (role, word): cluster
        for role, result in results.items()
        for word, cluster in result.clustering.assignments
    }
    excluded_margins = {
        (role, word): margin
        for role, result in results.items()
        for word, margin in result.excluded_margins
    }
    retained_margins = {
        (role, word): margin
        for role, result in results.items()
        for word, margin in result.clustering.margin_by_word(
            {
                item.word: item.vector
                for item in candidates[role]
                if (role, item.word) in assignments
            }
        ).items()
    }
    retained_nouns = {
        word for (role, word), _ in assignments.items() if role == "noun"
    }
    retained_verbs = {
        word for (role, word), _ in assignments.items() if role == "verb"
    }
    retained_tokens = sum(
        mass
        for (noun, verb), mass in pair_masses.items()
        if noun in retained_nouns and verb in retained_verbs
    )
    if retained_tokens / nonconstruction_token_count < config.minimum_retained_token_fraction:
        reason = (
            "both-role semantic exclusions retain less than "
            f"{config.minimum_retained_token_fraction:.0%} of archive token mass"
        )
        failure = _publish_failure(
            screened,
            calibration,
            tuple(traces),
            boundary_words,
            pair_masses,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            reason,
            output_root,
            temporary_directory,
            config,
            representative_contexts,
            progress=progress,
        )
        raise V2SemanticGridError(f"{reason}; failure audit: {failure.root}")

    words = tuple(
        _final_word(item, assignments, retained_margins, excluded_margins)
        for item in screened
    )
    clusters = tuple(
        SemanticCluster(
            role=role,
            index=index,
            token_mass=results[role].clustering.cluster_masses[index],
            centroid=results[role].clustering.centroids[index],
            words=tuple(
                sorted(
                    word
                    for word, cluster in results[role].clustering.assignments
                    if cluster == index
                )
            ),
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
        for index in range(config.cluster_count)
    )
    return _publish_catalog(
        words,
        clusters,
        calibration,
        tuple(traces),
        pair_masses,
        representative_contexts,
        encoder_identity,
        evidence_sha256,
        nonconstruction_token_count,
        retained_tokens,
        output_root,
        temporary_directory,
        config,
        parent_catalog,
        progress,
    )


def _screen_word(
    evidence: WordEvidence,
    calibrated: Mapping[tuple[Role, str], CalibratedRoleScore],
    config: V2SemanticConstructionConfig,
) -> _ScreenedWord:
    if evidence.context_embeddings.shape[0] < config.minimum_contexts_per_word:
        return _ScreenedWord(evidence, None, None, None, "insufficient_contexts")
    score = calibrated[(evidence.role, evidence.word)]
    if score.conformal_p <= config.role_calibration_alpha:
        return _ScreenedWord(
            evidence,
            score,
            None,
            None,
            "calibrated_role_outlier",
        )
    silhouette = deterministic_two_means_silhouette(evidence.context_embeddings)
    if silhouette > config.maximum_context_silhouette:
        return _ScreenedWord(
            evidence,
            score,
            silhouette,
            None,
            "multiple_realized_senses",
        )
    vector = compose_word_vector(
        evidence.target_anchor_embeddings,
        evidence.context_embeddings,
    )
    return _ScreenedWord(
        evidence,
        score,
        silhouette,
        tuple(float(value) for value in vector),
        None,
    )


def _cluster_role(
    role: Role,
    candidates: Sequence[WordVector],
    config: V2SemanticConstructionConfig,
    progress: V2CatalogProgress | None,
    progress_offset: int,
    progress_total: int,
) -> tuple[BoundaryClustering, tuple[BoundaryPassMetric, ...]]:
    retained = {item.word: item for item in candidates}
    excluded: dict[str, float] = {}
    trace = []
    for pass_index in range(config.maximum_exclusion_passes + 1):
        _emit(
            progress,
            "clustering",
            progress_offset + pass_index,
            progress_total,
            f"{role} boundary pass {pass_index}",
        )
        try:
            clustering = capacity_constrained_spherical_kmeans(
                tuple(retained.values()),
                config.cluster_count,
                minimum_mass_fraction=config.minimum_cluster_mass_fraction,
                maximum_mass_fraction=config.maximum_cluster_mass_fraction,
                maximum_iterations=config.maximum_centroid_iterations,
                benchmark_id=V2_BENCHMARK_ID,
                repair_assignment_dead_ends=True,
            )
        except SemanticGridError as error:
            raise _RoleGridFailure(
                f"{role} clustering failed: {error}",
                tuple(trace),
                frozenset(excluded),
            ) from error
        margins = clustering.margin_by_word(
            {word: item.vector for word, item in retained.items()}
        )
        failing = tuple(
            sorted(
                word
                for word, margin in margins.items()
                if margin < config.minimum_cluster_margin
            )
        )
        values = np.asarray(tuple(margins.values()), dtype=np.float64)
        trace.append(
            BoundaryPassMetric(
                role=role,
                pass_index=pass_index,
                input_word_count=len(retained),
                failing_word_count=len(failing),
                cluster_masses=clustering.cluster_masses,
                minimum_margin=float(np.min(values)),
                margin_q10=float(np.quantile(values, 0.10, method="linear")),
                median_margin=float(np.quantile(values, 0.50, method="linear")),
            )
        )
        _emit(
            progress,
            "clustering",
            progress_offset + pass_index + 1,
            progress_total,
            f"{role} pass {pass_index}: {len(failing)} boundary failures",
        )
        if not failing:
            try:
                validate_cluster_quality(clustering, config)
            except SemanticGridError as error:
                raise _RoleGridFailure(
                    f"{role} cluster quality failed: {error}",
                    tuple(trace),
                    frozenset(excluded),
                ) from error
            return (
                BoundaryClustering(
                    clustering=clustering,
                    excluded_margins=tuple(sorted(excluded.items())),
                    passes=pass_index,
                ),
                tuple(trace),
            )
        if pass_index == config.maximum_exclusion_passes:
            raise _RoleGridFailure(
                f"{role} cluster boundary exclusions did not converge within "
                f"{config.maximum_exclusion_passes} passes "
                f"({len(failing)} words still below {config.minimum_cluster_margin:.2f})",
                tuple(trace),
                frozenset((*excluded, *failing)),
            )
        excluded.update((word, margins[word]) for word in failing)
        retained = {
            word: item for word, item in retained.items() if word not in failing
        }
        if len(retained) < config.cluster_count:
            raise _RoleGridFailure(
                f"{role} boundary exclusions emptied the semantic grid",
                tuple(trace),
                frozenset(excluded),
            )
    raise AssertionError("semantic-v2 boundary loop must return or raise")


def _final_word(
    screened: _ScreenedWord,
    assignments: Mapping[tuple[Role, str], int],
    retained_margins: Mapping[tuple[Role, str], float],
    excluded_margins: Mapping[tuple[Role, str], float],
) -> V2SemanticWord:
    key = (screened.evidence.role, screened.evidence.word)
    boundary = key in excluded_margins
    reason = cast(
        V2ExclusionReason | None,
        "cluster_boundary_margin" if boundary else screened.reason,
    )
    score = screened.calibration
    return V2SemanticWord(
        role=screened.evidence.role,
        word=screened.evidence.word,
        token_mass=screened.evidence.token_mass,
        context_count=screened.evidence.context_embeddings.shape[0],
        calibration_fold=None if score is None else score.fold,
        calibration_reference_count=None if score is None else score.reference_count,
        role_margin_q10=None if score is None else score.raw_margin,
        role_conformal_p=None if score is None else score.conformal_p,
        role_rejection_cutoff=None if score is None else score.rejection_cutoff,
        context_silhouette=screened.silhouette,
        cluster_margin=excluded_margins.get(key, retained_margins.get(key)),
        cluster=None if reason is not None else assignments[key],
        exclusion_reason=reason,
        vector=None if reason is not None else screened.vector,
    )


def _publish_catalog(
    words: Sequence[V2SemanticWord],
    clusters: Sequence[SemanticCluster],
    calibration: Sequence[RoleCalibrationReference],
    trace: Sequence[BoundaryPassMetric],
    pair_masses: Mapping[tuple[str, str], int],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    retained_token_count: int,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V2SemanticConstructionConfig,
    parent_catalog: V2SemanticCatalog | None,
    progress: V2CatalogProgress | None,
) -> V2SemanticCatalog:
    _emit(progress, "publication", 0, 1, "writing semantic-v2 catalog and audits")
    working = _prepare_working_directory(temporary_directory)
    _write_jsonl(working / "words.jsonl", (item.as_record() for item in words))
    _write_json(
        working / "clusters.json",
        {"clusters": [item.as_record() for item in clusters]},
    )
    _write_calibration(working / "calibration.json", config, calibration)
    _write_trace(working / "boundary-trace.json", trace)
    _write_pair_masses(working / "role-pair-masses.jsonl", pair_masses)
    _write_contexts(working / "representative-contexts.jsonl", representative_contexts)
    content = {
        "benchmark_id": V2_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "clusters_sha256": _file_sha256(working / "clusters.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": V2_CATALOG_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "parent_catalog_sha256": (
            None if parent_catalog is None else parent_catalog.catalog_sha256
        ),
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "retained_token_count": retained_token_count,
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": V2_SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    catalog_sha = record_sha256(content)
    _write_json(
        working / "catalog.json",
        {**content, "catalog_sha256": catalog_sha},
    )
    markdown, html = render_calibrated_catalog_audits(
        catalog_sha,
        evidence_sha256,
        config,
        calibration,
        trace,
        words,
        clusters,
        pair_masses,
        representative_contexts,
        None if parent_catalog is None else parent_catalog.words,
        "v2",
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(working, "tinyworlds-p-semantic-v2-catalog-tree", "catalog_sha256", catalog_sha)
    target = _publish_directory(working, output_root, catalog_sha)
    catalog = load_v2_semantic_catalog(target)
    _emit(progress, "publication", 1, 1, "semantic-v2 catalog strictly reloaded")
    return catalog


def _publish_failure(
    screened: Sequence[_ScreenedWord],
    calibration: Sequence[RoleCalibrationReference],
    trace: Sequence[BoundaryPassMetric],
    boundary_words: Mapping[Role, set[str]],
    pair_masses: Mapping[tuple[str, str], int],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    reason: str,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V2SemanticConstructionConfig,
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ] | None = None,
    *,
    progress: V2CatalogProgress | None = None,
) -> V2CatalogFailureArtifact:
    _emit(progress, "publication", 0, 1, "writing semantic-v2 failure audit")
    boundary_words = boundary_words or {"noun": set(), "verb": set()}
    contexts = representative_contexts or {
        (item.evidence.role, item.evidence.word): item.evidence.contexts[
            : config.representative_contexts_per_cluster
        ]
        for item in screened
        if item.evidence.contexts
    }
    failure_words = tuple(
        CalibratedFailureWord(
            role=item.evidence.role,
            word=item.evidence.word,
            token_mass=item.evidence.token_mass,
            context_count=item.evidence.context_embeddings.shape[0],
            calibration_fold=(
                None if item.calibration is None else item.calibration.fold
            ),
            calibration_reference_count=(
                None if item.calibration is None else item.calibration.reference_count
            ),
            role_margin_q10=(
                None if item.calibration is None else item.calibration.raw_margin
            ),
            role_conformal_p=(
                None if item.calibration is None else item.calibration.conformal_p
            ),
            role_rejection_cutoff=(
                None if item.calibration is None else item.calibration.rejection_cutoff
            ),
            context_silhouette=item.silhouette,
            disposition=(
                cast(str, item.reason)
                if item.reason is not None
                else (
                    "cluster_boundary_margin"
                    if item.evidence.word in boundary_words[item.evidence.role]
                    else "semantic_grid_failure"
                )
            ),
            vector=item.vector,
        )
        for item in screened
    )
    working = _prepare_working_directory(temporary_directory)
    _write_jsonl(working / "words.jsonl", (item.as_record() for item in failure_words))
    _write_calibration(working / "calibration.json", config, calibration)
    _write_trace(working / "boundary-trace.json", trace)
    _write_pair_masses(working / "role-pair-masses.jsonl", pair_masses)
    _write_contexts(working / "representative-contexts.jsonl", contexts)
    content = {
        "benchmark_id": V2_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": V2_CATALOG_FAILURE_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "reason": reason,
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": V2_SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    failure_sha = record_sha256(content)
    _write_json(
        working / "failure.json",
        {**content, "failure_sha256": failure_sha},
    )
    markdown, html = render_calibrated_failure_audits(
        failure_sha,
        evidence_sha256,
        reason,
        config,
        calibration,
        trace,
        failure_words,
        contexts,
        "v2",
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(
        working,
        "tinyworlds-p-semantic-v2-catalog-failure-tree",
        "failure_sha256",
        failure_sha,
    )
    failure_root = Path(output_root) / "failures"
    target = failure_root / failure_sha
    if target.exists():
        _discard_empty_or_staged(working)
        failure = load_v2_catalog_failure(target)
        _emit(progress, "publication", 1, 1, "existing failure audit strictly reloaded")
        return failure
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(working, target)
    _fsync_directory(failure_root)
    failure = load_v2_catalog_failure(target)
    _emit(progress, "publication", 1, 1, "semantic-v2 failure audit strictly reloaded")
    return failure


def load_v2_semantic_catalog(path: str | Path) -> V2SemanticCatalog:
    """Strictly authenticate a complete semantic-v2 success catalog."""
    root = _validate_tree_root(
        path,
        "tinyworlds-p-semantic-v2-catalog-tree",
        "catalog_sha256",
        (
            "audit.html",
            "audit.md",
            "boundary-trace.json",
            "calibration.json",
            "catalog.json",
            "clusters.json",
            "representative-contexts.jsonl",
            "role-pair-masses.jsonl",
            "words.jsonl",
        ),
    )
    catalog_sha = root.name
    record = _load_json(root / "catalog.json")
    required = {
        "benchmark_id",
        "boundary_trace_sha256",
        "calibration_sha256",
        "catalog_sha256",
        "clusters_sha256",
        "config",
        "encoder",
        "evidence_sha256",
        "format",
        "nonconstruction_token_count",
        "parent_catalog_sha256",
        "representative_contexts_sha256",
        "retained_token_count",
        "role_pair_masses_sha256",
        "schema_version",
        "words_sha256",
    }
    if set(record) != required:
        raise V2SemanticCatalogError("semantic-v2 catalog fields changed")
    content = {key: value for key, value in record.items() if key != "catalog_sha256"}
    if (
        record.get("benchmark_id") != V2_BENCHMARK_ID
        or record.get("format") != V2_CATALOG_FORMAT
        or record.get("schema_version") != V2_SCHEMA_VERSION
        or record.get("catalog_sha256") != catalog_sha
        or record_sha256(content) != catalog_sha
    ):
        raise V2SemanticCatalogError("semantic-v2 catalog identity changed")
    _validate_payload_hashes(root, record, success=True)
    try:
        config = v2_semantic_config_from_record(_mapping(record, "config"))
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(
            _semantic_word(item, encoder.dimension)
            for item in _iter_jsonl(root / "words.jsonl")
        )
        clusters = _load_clusters(root / "clusters.json", encoder.dimension)
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
    except (TypeError, ValueError) as error:
        raise V2SemanticCatalogError("semantic-v2 catalog payload is invalid") from error
    nonconstruction = _integer(record, "nonconstruction_token_count")
    retained = _integer(record, "retained_token_count")
    if sum(pairs.values()) != nonconstruction:
        raise V2SemanticCatalogError("semantic-v2 pair masses changed")
    _validate_calibration(words, calibration, config)
    _validate_success_quality(words, clusters, pairs, config, retained, nonconstruction)
    parent = record.get("parent_catalog_sha256")
    if parent is not None and (type(parent) is not str or len(parent) != 64):
        raise V2SemanticCatalogError("semantic-v2 parent identity is malformed")
    _validate_trace(trace, config)
    return V2SemanticCatalog(
        root=root.resolve(),
        catalog_sha256=catalog_sha,
        evidence_sha256=_text(record, "evidence_sha256"),
        encoder_identity=encoder,
        config=config,
        calibration=calibration,
        words=words,
        clusters=clusters,
        retained_token_count=retained,
        nonconstruction_token_count=nonconstruction,
        parent_catalog_sha256=cast(str | None, parent),
    )


def load_v2_catalog_failure(path: str | Path) -> V2CatalogFailureArtifact:
    """Strictly authenticate a semantic-v2 failure audit and calibration."""
    root = _validate_tree_root(
        path,
        "tinyworlds-p-semantic-v2-catalog-failure-tree",
        "failure_sha256",
        (
            "audit.html",
            "audit.md",
            "boundary-trace.json",
            "calibration.json",
            "failure.json",
            "representative-contexts.jsonl",
            "role-pair-masses.jsonl",
            "words.jsonl",
        ),
    )
    failure_sha = root.name
    record = _load_json(root / "failure.json")
    required = {
        "benchmark_id",
        "boundary_trace_sha256",
        "calibration_sha256",
        "config",
        "encoder",
        "evidence_sha256",
        "failure_sha256",
        "format",
        "nonconstruction_token_count",
        "reason",
        "representative_contexts_sha256",
        "role_pair_masses_sha256",
        "schema_version",
        "words_sha256",
    }
    if set(record) != required:
        raise V2SemanticCatalogError("semantic-v2 failure fields changed")
    content = {key: value for key, value in record.items() if key != "failure_sha256"}
    if (
        record.get("benchmark_id") != V2_BENCHMARK_ID
        or record.get("format") != V2_CATALOG_FAILURE_FORMAT
        or record.get("schema_version") != V2_SCHEMA_VERSION
        or record.get("failure_sha256") != failure_sha
        or record_sha256(content) != failure_sha
    ):
        raise V2SemanticCatalogError("semantic-v2 failure identity changed")
    _validate_payload_hashes(root, record, success=False)
    try:
        config = v2_semantic_config_from_record(_mapping(record, "config"))
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(_iter_jsonl(root / "words.jsonl"))
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
    except (TypeError, ValueError) as error:
        raise V2SemanticCatalogError("semantic-v2 failure payload is invalid") from error
    if sum(pairs.values()) != _integer(record, "nonconstruction_token_count"):
        raise V2SemanticCatalogError("semantic-v2 failure pair masses changed")
    _validate_failure_calibration(words, calibration, config, encoder.dimension)
    _validate_failure_masses(words, pairs)
    _validate_trace(trace, config)
    tuple(_iter_jsonl(root / "representative-contexts.jsonl"))
    return V2CatalogFailureArtifact(
        root=root.resolve(),
        failure_sha256=failure_sha,
        evidence_sha256=_text(record, "evidence_sha256"),
        reason=_text(record, "reason"),
    )


def _validate_input_masses(
    words: Sequence[WordEvidence],
    pair_masses: Mapping[tuple[str, str], int],
    nonconstruction_token_count: int,
) -> None:
    if sum(pair_masses.values()) != nonconstruction_token_count:
        raise ValueError("role-pair mass does not cover non-construction data")
    expected = {"noun": Counter(), "verb": Counter()}
    for (noun, verb), mass in pair_masses.items():
        if not noun or not verb or type(mass) is not int or mass <= 0:
            raise ValueError("role-pair masses require words and positive mass")
        expected["noun"][noun] += mass
        expected["verb"][verb] += mass
    measured = {
        role: Counter(
            {item.word: item.token_mass for item in words if item.role == role}
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    if measured != expected:
        raise ValueError("semantic-v2 word masses differ from role-pair masses")


def _validate_calibration(
    words: Sequence[V2SemanticWord],
    calibration: Sequence[RoleCalibrationReference],
    config: V2SemanticConstructionConfig,
) -> None:
    raw = {
        (item.role, item.word): cast(float, item.role_margin_q10)
        for item in words
        if item.role_margin_q10 is not None
    }
    scores, expected = calibrate_role_margins(raw, config)
    if tuple(calibration) != expected:
        raise V2SemanticCatalogError("semantic-v2 calibration references changed")
    for word in words:
        score = scores.get((word.role, word.word))
        measured = (
            word.calibration_fold,
            word.calibration_reference_count,
            word.role_conformal_p,
            word.role_rejection_cutoff,
        )
        expected_values = (
            None,
            None,
            None,
            None,
        ) if score is None else (
            score.fold,
            score.reference_count,
            score.conformal_p,
            score.rejection_cutoff,
        )
        if measured != expected_values:
            raise V2SemanticCatalogError("semantic-v2 calibrated word score changed")


def _validate_failure_calibration(
    words: Sequence[Mapping[str, object]],
    calibration: Sequence[RoleCalibrationReference],
    config: V2SemanticConstructionConfig,
    dimension: int,
) -> None:
    identities = tuple((_role(item, "role"), _text(item, "word")) for item in words)
    if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
        raise V2SemanticCatalogError("semantic-v2 failure words are not canonical")
    raw = {
        (_role(item, "role"), _text(item, "word")): cast(
            float, _optional_number(item, "role_margin_q10")
        )
        for item in words
        if item.get("role_margin_q10") is not None
    }
    if not raw:
        if calibration:
            raise V2SemanticCatalogError("failure calibration unexpectedly has references")
        return
    scores, expected = calibrate_role_margins(raw, config)
    if tuple(calibration) != expected:
        raise V2SemanticCatalogError("failure calibration references changed")
    required = set(CalibratedFailureWord.__dataclass_fields__)
    for item in words:
        serialized = set(item)
        expected_fields = (required - {"disposition"}) | {"disposition"}
        if serialized != expected_fields:
            raise V2SemanticCatalogError("semantic-v2 failure word fields changed")
        key = (_role(item, "role"), _text(item, "word"))
        context_count = _integer(item, "context_count")
        _positive_integer(item, "token_mass")
        silhouette = _optional_number(item, "context_silhouette")
        disposition = _text(item, "disposition")
        vector_value = item.get("vector")
        if vector_value is not None:
            if (
                type(vector_value) is not list
                or len(vector_value) != dimension
                or any(type(value) not in (int, float) for value in vector_value)
            ):
                raise V2SemanticCatalogError("semantic-v2 failure vector changed")
            vector = np.asarray(vector_value, dtype=np.float64)
            if not np.all(np.isfinite(vector)) or not np.isclose(
                np.linalg.norm(vector), 1.0, atol=1e-5
            ):
                raise V2SemanticCatalogError("semantic-v2 failure vector is invalid")
        score = scores.get(key)
        if score is None:
            values = (
                item.get("calibration_fold"),
                item.get("calibration_reference_count"),
                item.get("role_conformal_p"),
                item.get("role_rejection_cutoff"),
            )
            if values != (None, None, None, None):
                raise V2SemanticCatalogError("failure word calibration is inconsistent")
        elif (
            item.get("calibration_fold") != score.fold
            or item.get("calibration_reference_count") != score.reference_count
            or item.get("role_conformal_p") != score.conformal_p
            or item.get("role_rejection_cutoff") != score.rejection_cutoff
        ):
            raise V2SemanticCatalogError("failure calibrated word score changed")
        score_pass = score is not None and score.conformal_p > config.role_calibration_alpha
        if disposition == "insufficient_contexts":
            valid = (
                context_count < config.minimum_contexts_per_word
                and score is None
                and silhouette is None
                and vector_value is None
            )
        elif disposition == "calibrated_role_outlier":
            valid = score is not None and not score_pass and silhouette is None and vector_value is None
        elif disposition == "multiple_realized_senses":
            valid = (
                score_pass
                and silhouette is not None
                and silhouette > config.maximum_context_silhouette
                and vector_value is None
            )
        elif disposition in ("cluster_boundary_margin", "semantic_grid_failure"):
            valid = (
                score_pass
                and silhouette is not None
                and silhouette <= config.maximum_context_silhouette
                and vector_value is not None
            ) or (not calibration and score is None and vector_value is None)
        else:
            valid = False
        if not valid:
            raise V2SemanticCatalogError("semantic-v2 failure disposition changed")


def _validate_failure_masses(
    words: Sequence[Mapping[str, object]],
    pairs: Mapping[tuple[str, str], int],
) -> None:
    expected = {"noun": Counter(), "verb": Counter()}
    for (noun, verb), mass in pairs.items():
        expected["noun"][noun] += mass
        expected["verb"][verb] += mass
    measured = {
        role: Counter(
            {
                _text(item, "word"): _positive_integer(item, "token_mass")
                for item in words
                if _role(item, "role") == role
            }
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    if measured != expected:
        raise V2SemanticCatalogError("semantic-v2 failure word masses changed")


def _validate_success_quality(
    words: Sequence[V2SemanticWord],
    clusters: Sequence[SemanticCluster],
    pairs: Mapping[tuple[str, str], int],
    config: V2SemanticConstructionConfig,
    retained: int,
    nonconstruction: int,
) -> None:
    expected_clusters = tuple(
        (role, index)
        for role in ("noun", "verb")
        for index in range(config.cluster_count)
    )
    if tuple((item.role, item.index) for item in clusters) != expected_clusters:
        raise V2SemanticCatalogError("semantic-v2 clusters are incomplete")
    retained_words = {
        role: {item.word for item in words if item.role == role and item.cluster is not None}
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    measured_retained = sum(
        mass
        for (noun, verb), mass in pairs.items()
        if noun in retained_words["noun"] and verb in retained_words["verb"]
    )
    if measured_retained != retained or retained / nonconstruction < config.minimum_retained_token_fraction:
        raise V2SemanticCatalogError("semantic-v2 retained mass gate changed")
    by_cluster: dict[tuple[Role, int], list[V2SemanticWord]] = defaultdict(list)
    for word in words:
        score_pass = (
            word.role_conformal_p is not None
            and word.role_conformal_p > config.role_calibration_alpha
        )
        if word.exclusion_reason == "insufficient_contexts":
            valid = word.context_count < config.minimum_contexts_per_word and word.role_margin_q10 is None
        elif word.exclusion_reason == "calibrated_role_outlier":
            valid = word.role_conformal_p is not None and not score_pass and word.context_silhouette is None
        elif word.exclusion_reason == "multiple_realized_senses":
            valid = score_pass and word.context_silhouette is not None and word.context_silhouette > config.maximum_context_silhouette
        elif word.exclusion_reason == "cluster_boundary_margin":
            valid = score_pass and word.context_silhouette is not None and word.context_silhouette <= config.maximum_context_silhouette and word.cluster_margin is not None and word.cluster_margin < config.minimum_cluster_margin
        elif word.exclusion_reason is None:
            valid = score_pass and word.context_silhouette is not None and word.context_silhouette <= config.maximum_context_silhouette and word.cluster_margin is not None and word.cluster_margin >= config.minimum_cluster_margin
            if word.cluster is not None:
                by_cluster[(word.role, word.cluster)].append(word)
        else:
            valid = False
        if not valid:
            raise V2SemanticCatalogError("semantic-v2 word disposition violates its gates")
    for cluster in clusters:
        members = by_cluster[(cluster.role, cluster.index)]
        if tuple(sorted(item.word for item in members)) != cluster.words:
            raise V2SemanticCatalogError("semantic-v2 cluster membership changed")
        if sum(item.token_mass for item in members) != cluster.token_mass:
            raise V2SemanticCatalogError("semantic-v2 cluster mass changed")
        centroid = _normalized_centroid(
            np.asarray([cast(tuple[float, ...], item.vector) for item in members])
        )
        if not np.allclose(centroid, np.asarray(cluster.centroid), atol=1e-6):
            raise V2SemanticCatalogError("semantic-v2 cluster centroid changed")
    for role in cast(tuple[Role, Role], ("noun", "verb")):
        role_clusters = [item for item in clusters if item.role == role]
        target = sum(item.token_mass for item in role_clusters) / config.cluster_count
        if any(
            not config.minimum_cluster_mass_fraction * target <= item.token_mass <= config.maximum_cluster_mass_fraction * target
            for item in role_clusters
        ):
            raise V2SemanticCatalogError("semantic-v2 cluster mass bounds changed")
        minimum = config.minimum_nouns_per_cluster if role == "noun" else config.minimum_verbs_per_cluster
        if any(len(item.words) < minimum for item in role_clusters):
            raise V2SemanticCatalogError("semantic-v2 cluster word-count gate changed")
        centroids = np.asarray([item.centroid for item in role_clusters])
        if max(
            float(centroids[left] @ centroids[right])
            for left in range(len(centroids))
            for right in range(left + 1, len(centroids))
        ) >= config.maximum_centroid_pair_cosine:
            raise V2SemanticCatalogError("semantic-v2 centroid-pair gate changed")
        centroid_matrix = np.asarray([item.centroid for item in role_clusters], dtype=np.float64)
        for word in (
            item
            for item in words
            if item.role == role and item.exclusion_reason is None
        ):
            assert word.cluster is not None and word.vector is not None
            similarities = centroid_matrix @ np.asarray(word.vector, dtype=np.float64)
            measured_margin = float(similarities[word.cluster]) - max(
                float(value)
                for index, value in enumerate(similarities)
                if index != word.cluster
            )
            if word.cluster_margin is None or not np.isclose(
                word.cluster_margin,
                measured_margin,
                atol=1e-12,
            ):
                raise V2SemanticCatalogError("semantic-v2 retained cluster margin changed")


def _validate_trace(
    trace: Sequence[BoundaryPassMetric],
    config: V2SemanticConstructionConfig,
) -> None:
    by_role: dict[Role, list[BoundaryPassMetric]] = defaultdict(list)
    for item in trace:
        if len(item.cluster_masses) != config.cluster_count:
            raise V2SemanticCatalogError("semantic-v2 trace cluster count changed")
        by_role[item.role].append(item)
    for items in by_role.values():
        if tuple(item.pass_index for item in items) != tuple(range(len(items))):
            raise V2SemanticCatalogError("semantic-v2 trace pass order changed")
        if len(items) > config.maximum_exclusion_passes + 1:
            raise V2SemanticCatalogError("semantic-v2 trace exceeds pass budget")
        for previous, following in zip(items, items[1:]):
            if following.input_word_count != previous.input_word_count - previous.failing_word_count:
                raise V2SemanticCatalogError("semantic-v2 trace exclusion counts changed")


def _write_calibration(
    path: Path,
    config: V2SemanticConstructionConfig,
    items: Sequence[RoleCalibrationReference],
) -> None:
    _write_json(
        path,
        {
            "alpha": config.role_calibration_alpha,
            "fold_count": config.role_calibration_fold_count,
            "method": config.role_calibration_method,
            "namespace": config.role_calibration_namespace,
            "references": [item.as_record() for item in items],
        },
    )


def _load_calibration(
    path: Path,
    config: V2SemanticConstructionConfig,
) -> tuple[RoleCalibrationReference, ...]:
    record = _load_json(path)
    if set(record) != {"alpha", "fold_count", "method", "namespace", "references"}:
        raise V2SemanticCatalogError("semantic-v2 calibration fields changed")
    if (
        record["alpha"] != config.role_calibration_alpha
        or record["fold_count"] != config.role_calibration_fold_count
        or record["method"] != config.role_calibration_method
        or record["namespace"] != config.role_calibration_namespace
        or type(record["references"]) is not list
    ):
        raise V2SemanticCatalogError("semantic-v2 calibration identity changed")
    result = tuple(
        RoleCalibrationReference(
            role=_role(item, "role"),
            fold=_integer(item, "fold"),
            reference_count=_integer(item, "reference_count"),
            rejection_cutoff=_optional_number(item, "rejection_cutoff"),
        )
        for item in record["references"]
        if type(item) is dict
    )
    if len(result) != len(record["references"]):
        raise V2SemanticCatalogError("semantic-v2 calibration records changed")
    expected_order = tuple(
        (role, fold)
        for role in ("noun", "verb")
        for fold in range(config.role_calibration_fold_count)
    )
    if result and tuple((item.role, item.fold) for item in result) != expected_order:
        raise V2SemanticCatalogError("semantic-v2 calibration folds changed")
    return result


def _write_trace(path: Path, items: Sequence[BoundaryPassMetric]) -> None:
    _write_json(path, {"passes": [item.as_record() for item in items]})


def _load_trace(
    path: Path,
    config: V2SemanticConstructionConfig,
) -> tuple[BoundaryPassMetric, ...]:
    record = _load_json(path)
    if set(record) != {"passes"} or type(record["passes"]) is not list:
        raise V2SemanticCatalogError("semantic-v2 trace fields changed")
    result = tuple(
        BoundaryPassMetric(
            role=_role(item, "role"),
            pass_index=_integer(item, "pass_index"),
            input_word_count=_integer(item, "input_word_count"),
            failing_word_count=_integer(item, "failing_word_count"),
            cluster_masses=tuple(_integer_value(value) for value in _list(item, "cluster_masses")),
            minimum_margin=_number(item, "minimum_margin"),
            margin_q10=_number(item, "margin_q10"),
            median_margin=_number(item, "median_margin"),
        )
        for item in record["passes"]
        if type(item) is dict
    )
    if len(result) != len(record["passes"]):
        raise V2SemanticCatalogError("semantic-v2 trace records changed")
    _validate_trace(result, config)
    return result


def _semantic_word(record: Mapping[str, object], dimension: int) -> V2SemanticWord:
    required = set(V2SemanticWord.__dataclass_fields__)
    if set(record) != required:
        raise V2SemanticCatalogError("semantic-v2 word fields changed")
    vector_value = record.get("vector")
    vector = None
    if vector_value is not None:
        if type(vector_value) is not list or len(vector_value) != dimension:
            raise V2SemanticCatalogError("semantic-v2 word vector changed")
        vector = tuple(float(value) for value in vector_value)
    reason = record.get("exclusion_reason")
    if reason not in (
        None,
        "insufficient_contexts",
        "calibrated_role_outlier",
        "multiple_realized_senses",
        "cluster_boundary_margin",
    ):
        raise V2SemanticCatalogError("semantic-v2 exclusion reason changed")
    return V2SemanticWord(
        role=_role(record, "role"),
        word=_text(record, "word"),
        token_mass=_positive_integer(record, "token_mass"),
        context_count=_integer(record, "context_count"),
        calibration_fold=_optional_integer(record, "calibration_fold"),
        calibration_reference_count=_optional_integer(
            record, "calibration_reference_count"
        ),
        role_margin_q10=_optional_number(record, "role_margin_q10"),
        role_conformal_p=_optional_number(record, "role_conformal_p"),
        role_rejection_cutoff=_optional_number(record, "role_rejection_cutoff"),
        context_silhouette=_optional_number(record, "context_silhouette"),
        cluster_margin=_optional_number(record, "cluster_margin"),
        cluster=_optional_integer(record, "cluster"),
        exclusion_reason=cast(V2ExclusionReason | None, reason),
        vector=vector,
    )


def _load_clusters(path: Path, dimension: int) -> tuple[SemanticCluster, ...]:
    payload = _load_json(path)
    if set(payload) != {"clusters"} or type(payload["clusters"]) is not list:
        raise V2SemanticCatalogError("semantic-v2 cluster payload changed")
    result = []
    for record in payload["clusters"]:
        if type(record) is not dict or set(record) != {"centroid", "index", "role", "token_mass", "words"}:
            raise V2SemanticCatalogError("semantic-v2 cluster fields changed")
        centroid = _list(record, "centroid")
        words = _list(record, "words")
        if len(centroid) != dimension or any(type(item) not in (int, float) for item in centroid):
            raise V2SemanticCatalogError("semantic-v2 cluster centroid changed")
        if any(type(item) is not str or not item for item in words):
            raise V2SemanticCatalogError("semantic-v2 cluster words changed")
        result.append(
            SemanticCluster(
                role=_role(record, "role"),
                index=_integer(record, "index"),
                token_mass=_positive_integer(record, "token_mass"),
                centroid=tuple(float(value) for value in centroid),
                words=tuple(words),
            )
        )
    return tuple(result)


def _encoder_identity(record: Mapping[str, object]) -> EncoderIdentity:
    required = {
        "dimension",
        "dtype",
        "files",
        "identifier",
        "normalization",
        "pooling",
        "revision",
    }
    if set(record) != required or type(record.get("files")) is not list:
        raise V2SemanticCatalogError("semantic-v2 encoder fields changed")
    files = tuple(
        ModelFile(
            relative_path=_text(item, "relative_path"),
            size_bytes=_integer(item, "size_bytes"),
            sha256=_text(item, "sha256"),
        )
        for item in record["files"]
        if type(item) is dict
    )
    if len(files) != len(record["files"]):
        raise V2SemanticCatalogError("semantic-v2 encoder files changed")
    identity = EncoderIdentity(
        identifier=_text(record, "identifier"),
        revision=_text(record, "revision"),
        dimension=_positive_integer(record, "dimension"),
        files=files,
        pooling=cast(str, record.get("pooling")),
        normalization=cast(str, record.get("normalization")),
        dtype=cast(str, record.get("dtype")),
    )
    if identity.as_record() != dict(record):
        raise V2SemanticCatalogError("semantic-v2 encoder identity changed")
    return identity


def _load_pairs(path: Path) -> dict[tuple[str, str], int]:
    result = {}
    for record in _iter_jsonl(path):
        if set(record) != {"noun", "token_mass", "verb"}:
            raise V2SemanticCatalogError("semantic-v2 role-pair fields changed")
        key = (_text(record, "noun"), _text(record, "verb"))
        if key in result:
            raise V2SemanticCatalogError("semantic-v2 role-pair duplicated")
        result[key] = _positive_integer(record, "token_mass")
    return result


def _write_pair_masses(path: Path, pairs: Mapping[tuple[str, str], int]) -> None:
    _write_jsonl(
        path,
        (
            {"noun": noun, "token_mass": mass, "verb": verb}
            for (noun, verb), mass in sorted(pairs.items())
        ),
    )


def _write_contexts(
    path: Path,
    contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
) -> None:
    _write_jsonl(
        path,
        (dict(record) for key in sorted(contexts) for record in contexts[key]),
    )


def _validate_payload_hashes(
    root: Path,
    record: Mapping[str, object],
    *,
    success: bool,
) -> None:
    pairs = [
        ("boundary-trace.json", "boundary_trace_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("representative-contexts.jsonl", "representative_contexts_sha256"),
        ("role-pair-masses.jsonl", "role_pair_masses_sha256"),
        ("words.jsonl", "words_sha256"),
    ]
    if success:
        pairs.append(("clusters.json", "clusters_sha256"))
    for filename, field in pairs:
        if _file_sha256(root / filename) != record.get(field):
            raise V2SemanticCatalogError(f"semantic-v2 payload changed: {filename}")


def _prepare_working_directory(path: str | Path) -> Path:
    working = Path(path)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"semantic-v2 temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    return working


def _publish_directory(working: Path, output_root: str | Path, identity: str) -> Path:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    target = output / identity
    if target.exists():
        raise FileExistsError(f"semantic-v2 artifact already exists: {target}")
    os.rename(working, target)
    _fsync_directory(output)
    return target


def _discard_empty_or_staged(path: Path) -> None:
    for candidate in sorted(path.rglob("*"), reverse=True):
        if candidate.is_file():
            candidate.unlink()
        elif candidate.is_dir():
            candidate.rmdir()
    path.rmdir()


def _write_tree(path: Path, format_name: str, identity_field: str, identity: str) -> None:
    files = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": candidate.relative_to(path).as_posix(),
                "sha256": _file_sha256(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    _write_json(
        path / "tree.json",
        {
            identity_field: identity,
            "files": files,
            "format": format_name,
            "schema_version": V2_SCHEMA_VERSION,
        },
    )


def _validate_tree_root(
    path: str | Path,
    format_name: str,
    identity_field: str,
    expected_paths: Sequence[str],
) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise V2SemanticCatalogError("semantic-v2 artifact must be a regular directory")
    tree = _load_json(root / "tree.json")
    if set(tree) != {identity_field, "files", "format", "schema_version"}:
        raise V2SemanticCatalogError("semantic-v2 tree fields changed")
    if tree["format"] != format_name or tree["schema_version"] != V2_SCHEMA_VERSION:
        raise V2SemanticCatalogError("semantic-v2 tree identity changed")
    if tree[identity_field] != root.name or type(tree.get("files")) is not list:
        raise V2SemanticCatalogError("semantic-v2 tree directory identity changed")
    descriptors = tree["files"]
    if any(type(item) is not dict for item in descriptors):
        raise V2SemanticCatalogError("semantic-v2 tree descriptors changed")
    paths = tuple(_text(item, "relative_path") for item in descriptors)
    if paths != tuple(expected_paths):
        raise V2SemanticCatalogError("semantic-v2 tree membership changed")
    actual = tuple(
        sorted(
            candidate.relative_to(root).as_posix()
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.name != "tree.json"
        )
    )
    if actual != tuple(expected_paths) or any(candidate.is_symlink() for candidate in root.rglob("*")):
        raise V2SemanticCatalogError("semantic-v2 artifact membership changed")
    for descriptor in descriptors:
        candidate = root / _text(descriptor, "relative_path")
        if candidate.stat().st_size != _integer(descriptor, "size_bytes") or _file_sha256(candidate) != _text(descriptor, "sha256"):
            raise V2SemanticCatalogError("semantic-v2 artifact file changed")
    return root


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, canonical_json_bytes(value))


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    with path.open("wb") as output:
        for record in records:
            output.write(canonical_json_bytes(record))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    _write_bytes(path, value.encode("utf-8"))


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V2SemanticCatalogError(f"invalid semantic-v2 JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise V2SemanticCatalogError(f"noncanonical semantic-v2 JSON: {path.name}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("rb") as source:
        for line in source:
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise V2SemanticCatalogError(f"invalid semantic-v2 JSONL: {path.name}") from error
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise V2SemanticCatalogError(f"noncanonical semantic-v2 JSONL: {path.name}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalized_centroid(values: np.ndarray) -> np.ndarray:
    centroid = np.mean(np.asarray(values, dtype=np.float32), axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm <= 0.0:
        raise V2SemanticCatalogError("semantic-v2 centroid is invalid")
    return np.asarray(centroid / np.float32(norm), dtype=np.float32)


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V2SemanticCatalogError(f"field {field!r} must be an object")
    return value


def _list(record: Mapping[str, object], field: str) -> list[object]:
    value = record.get(field)
    if type(value) is not list:
        raise V2SemanticCatalogError(f"field {field!r} must be a list")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V2SemanticCatalogError(f"field {field!r} must be nonempty text")
    return value


def _role(record: Mapping[str, object], field: str) -> Role:
    value = _text(record, field)
    if value not in ("noun", "verb"):
        raise V2SemanticCatalogError(f"field {field!r} must be noun or verb")
    return cast(Role, value)


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise V2SemanticCatalogError(f"field {field!r} must be nonnegative integer")
    return value


def _positive_integer(record: Mapping[str, object], field: str) -> int:
    value = _integer(record, field)
    if value <= 0:
        raise V2SemanticCatalogError(f"field {field!r} must be positive")
    return value


def _integer_value(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise V2SemanticCatalogError("list value must be a positive integer")
    return value


def _optional_integer(record: Mapping[str, object], field: str) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise V2SemanticCatalogError(f"field {field!r} must be integer or null")
    return value


def _number(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise V2SemanticCatalogError(f"field {field!r} must be finite numeric")
    return float(value)


def _optional_number(record: Mapping[str, object], field: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise V2SemanticCatalogError(f"field {field!r} must be numeric or null")
    return float(value)


def _emit(
    progress: V2CatalogProgress | None,
    phase: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if progress is not None:
        progress(phase, completed, total, detail)


__all__ = [
    "V2SemanticCatalogError",
    "V2CatalogProgress",
    "V2SemanticGridError",
    "build_v2_catalog_from_evidence",
    "build_v2_semantic_catalog",
    "load_v2_catalog_failure",
    "load_v2_semantic_catalog",
]
