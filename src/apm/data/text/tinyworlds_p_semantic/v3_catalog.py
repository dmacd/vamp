"""Semantic-first v3 screening, clustering, publication, and strict loading."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    WordVector,
    compose_word_vector,
    deterministic_two_means_silhouette,
    role_margin_quantile,
    semantic_first_spherical_kmeans,
    validate_cluster_quality,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    EncoderIdentity,
    Role,
    SemanticCluster,
    SemanticEvidenceArtifact,
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
from apm.data.text.tinyworlds_p_semantic.v2_catalog import (
    V2SemanticCatalogError,
    _discard_empty_or_staged,
    _emit,
    _encoder_identity,
    _file_sha256,
    _integer,
    _iter_jsonl,
    _load_calibration,
    _load_clusters,
    _load_json,
    _load_pairs,
    _load_trace,
    _mapping,
    _number,
    _normalized_centroid,
    _optional_integer,
    _optional_number,
    _positive_integer,
    _prepare_working_directory,
    _publish_directory,
    _role,
    _text,
    _validate_calibration,
    _validate_failure_calibration,
    _validate_failure_masses,
    _validate_input_masses,
    _validate_payload_hashes,
    _validate_trace,
    _validate_tree_root,
    _write_calibration,
    _write_contexts,
    _write_json,
    _write_jsonl,
    _write_pair_masses,
    _write_text,
    _write_trace,
    _write_tree,
    load_v2_catalog_failure,
)
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    CalibratedRoleScore,
    RoleCalibrationReference,
)
from apm.data.text.tinyworlds_p_semantic.v3_contracts import (
    V3_BENCHMARK_ID,
    V3_CATALOG_FAILURE_FORMAT,
    V3_CATALOG_FORMAT,
    V3_SCHEMA_VERSION,
    V3CatalogFailureArtifact,
    V3ExclusionReason,
    V3SemanticCatalog,
    V3SemanticConstructionConfig,
    V3SemanticWord,
    v3_semantic_config_from_record,
)


class V3SemanticCatalogError(ValueError):
    """A semantic-v3 catalog or failure bundle is malformed."""


class V3SemanticGridError(ValueError):
    """The frozen semantic-v3 construction failed and published evidence."""


V3CatalogProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class _ScreenedWord:
    evidence: WordEvidence
    calibration: CalibratedRoleScore | None
    silhouette: float | None
    vector: tuple[float, ...] | None
    reason: V3ExclusionReason | None


@dataclass(frozen=True, slots=True)
class _RoleGridFailure(Exception):
    reason: str
    trace: tuple[BoundaryPassMetric, ...]
    boundary_words: frozenset[str]

    def __str__(self) -> str:
        return self.reason


def build_v3_catalog_from_evidence(
    evidence: SemanticEvidenceArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V3SemanticConstructionConfig,
    *,
    parent_catalog: V3SemanticCatalog | None = None,
    progress: V3CatalogProgress | None = None,
) -> V3SemanticCatalog:
    """Reuse authenticated v1 embeddings under the semantic-first v3 contract."""
    if np.__version__ != config.construction_numpy_version:
        raise RuntimeError(
            "semantic-v3 construction requires NumPy "
            f"{config.construction_numpy_version}, found {np.__version__}"
        )
    if evidence.config.evidence_record() != config.evidence_record():
        raise ValueError("semantic-v3 evidence configuration changed")
    expected_role_scores = _load_v2_role_scores(evidence, config)
    _emit(progress, "evidence-load", 0, 1, "loading authenticated cached vectors")
    word_evidence, pair_masses = load_word_evidence(evidence)
    _emit(progress, "evidence-load", 1, 1, "cached word evidence loaded")
    return build_v3_semantic_catalog(
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
        expected_role_scores=expected_role_scores,
    )


def build_v3_semantic_catalog(
    word_evidence: Sequence[WordEvidence],
    pair_masses: Mapping[tuple[str, str], int],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V3SemanticConstructionConfig,
    *,
    parent_catalog: V3SemanticCatalog | None = None,
    progress: V3CatalogProgress | None = None,
    expected_role_scores: Mapping[tuple[Role, str], CalibratedRoleScore]
    | None = None,
) -> V3SemanticCatalog:
    """Apply v2 calibration and build the frozen semantic-first v3 grid."""
    canonical = tuple(sorted(word_evidence, key=lambda item: (item.role, item.word)))
    if len({(item.role, item.word) for item in canonical}) != len(canonical):
        raise ValueError("semantic-v3 word evidence contains duplicate role words")
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
    _emit(progress, "calibration", 0, 1, "replaying frozen v2 role calibration")
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
        raise V3SemanticGridError(
            f"role calibration failed: {error}; failure audit: {failure.root}"
        ) from error
    if expected_role_scores is not None and calibrated != dict(expected_role_scores):
        raise ValueError("semantic-v3 did not exactly replay the bound v2 role scores")
    _emit(progress, "calibration", 1, 1, "frozen v2 role calibration replayed")

    screened_items = []
    _emit(progress, "screening", 0, len(canonical), "applying role and sense gates")
    for completed, item in enumerate(canonical, start=1):
        screened_items.append(_screen_word(item, calibrated, config))
        if completed % 50 == 0 or completed == len(canonical):
            _emit(
                progress,
                "screening",
                completed,
                len(canonical),
                "applying role and sense gates",
            )
    screened = tuple(screened_items)
    roles = cast(tuple[Role, Role], ("noun", "verb"))
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
        for role in roles
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
    cluster_progress_total = len(roles) * (config.maximum_exclusion_passes + 1)
    _emit(
        progress,
        "clustering",
        0,
        cluster_progress_total,
        "starting semantic-first v3 clusters",
    )
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
        raise V3SemanticGridError(f"{reason}; failure audit: {failure.root}")

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
        raise V3SemanticGridError(f"{reason}; failure audit: {failure.root}")
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
        for role in roles
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


def _load_v2_role_scores(
    evidence: SemanticEvidenceArtifact,
    config: V3SemanticConstructionConfig,
) -> dict[tuple[Role, str], CalibratedRoleScore]:
    """Load the exact authenticated v2 word scores bound by the v3 config."""
    semantic_data_root = evidence.root.parents[2]
    source_root = (
        semantic_data_root
        / "catalog"
        / "v2"
        / "failures"
        / config.role_calibration_source_failure_sha256
    )
    source = load_v2_catalog_failure(source_root)
    if source.evidence_sha256 != evidence.evidence_sha256:
        raise ValueError("semantic-v3 role calibration source evidence changed")
    scores = {
        (_role(record, "role"), _text(record, "word")): CalibratedRoleScore(
            role=_role(record, "role"),
            word=_text(record, "word"),
            fold=_integer(record, "calibration_fold"),
            raw_margin=_number(record, "role_margin_q10"),
            reference_count=_positive_integer(
                record,
                "calibration_reference_count",
            ),
            conformal_p=_number(record, "role_conformal_p"),
            rejection_cutoff=_optional_number(record, "role_rejection_cutoff"),
        )
        for record in _iter_jsonl(source.root / "words.jsonl")
        if record.get("role_margin_q10") is not None
    }
    return scores


def _screen_word(
    evidence: WordEvidence,
    calibrated: Mapping[tuple[Role, str], CalibratedRoleScore],
    config: V3SemanticConstructionConfig,
) -> _ScreenedWord:
    if evidence.context_embeddings.shape[0] < config.minimum_contexts_per_word:
        return _ScreenedWord(evidence, None, None, None, "insufficient_contexts")
    score = calibrated[(evidence.role, evidence.word)]
    if score.conformal_p <= config.role_calibration_alpha:
        return _ScreenedWord(evidence, score, None, None, "calibrated_role_outlier")
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
    config: V3SemanticConstructionConfig,
    progress: V3CatalogProgress | None,
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
            f"{role} nearest-centroid pass {pass_index}",
        )
        try:
            clustering = semantic_first_spherical_kmeans(
                tuple(retained.values()),
                config.cluster_count,
                maximum_iterations=config.maximum_centroid_iterations,
                benchmark_id=V3_BENCHMARK_ID,
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
    raise AssertionError("semantic-v3 boundary loop must return or raise")


def _final_word(
    screened: _ScreenedWord,
    assignments: Mapping[tuple[Role, str], int],
    retained_margins: Mapping[tuple[Role, str], float],
    excluded_margins: Mapping[tuple[Role, str], float],
) -> V3SemanticWord:
    key = (screened.evidence.role, screened.evidence.word)
    boundary = key in excluded_margins
    reason = cast(
        V3ExclusionReason | None,
        "cluster_boundary_margin" if boundary else screened.reason,
    )
    score = screened.calibration
    return V3SemanticWord(
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
    words: Sequence[V3SemanticWord],
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
    config: V3SemanticConstructionConfig,
    parent_catalog: V3SemanticCatalog | None,
    progress: V3CatalogProgress | None,
) -> V3SemanticCatalog:
    _emit(progress, "publication", 0, 1, "writing semantic-v3 catalog and audits")
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
        "benchmark_id": V3_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "clusters_sha256": _file_sha256(working / "clusters.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": V3_CATALOG_FORMAT,
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
        "schema_version": V3_SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    catalog_sha = record_sha256(content)
    _write_json(working / "catalog.json", {**content, "catalog_sha256": catalog_sha})
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
        "v3",
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(
        working,
        "tinyworlds-p-semantic-v3-catalog-tree",
        "catalog_sha256",
        catalog_sha,
    )
    target = _publish_directory(working, output_root, catalog_sha)
    catalog = load_v3_semantic_catalog(target)
    _emit(progress, "publication", 1, 1, "semantic-v3 catalog strictly reloaded")
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
    config: V3SemanticConstructionConfig,
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ] | None = None,
    *,
    progress: V3CatalogProgress | None = None,
) -> V3CatalogFailureArtifact:
    _emit(progress, "publication", 0, 1, "writing semantic-v3 failure audit")
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
        "benchmark_id": V3_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": V3_CATALOG_FAILURE_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "reason": reason,
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": V3_SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    failure_sha = record_sha256(content)
    _write_json(working / "failure.json", {**content, "failure_sha256": failure_sha})
    markdown, html = render_calibrated_failure_audits(
        failure_sha,
        evidence_sha256,
        reason,
        config,
        calibration,
        trace,
        failure_words,
        contexts,
        "v3",
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(
        working,
        "tinyworlds-p-semantic-v3-catalog-failure-tree",
        "failure_sha256",
        failure_sha,
    )
    failure_root = Path(output_root) / "failures"
    target = failure_root / failure_sha
    if target.exists():
        _discard_empty_or_staged(working)
        failure = load_v3_catalog_failure(target)
        _emit(progress, "publication", 1, 1, "existing failure audit strictly reloaded")
        return failure
    failure_root.mkdir(parents=True, exist_ok=True)
    target = _publish_directory(working, failure_root, failure_sha)
    failure = load_v3_catalog_failure(target)
    _emit(progress, "publication", 1, 1, "semantic-v3 failure audit strictly reloaded")
    return failure


def load_v3_semantic_catalog(path: str | Path) -> V3SemanticCatalog:
    """Strictly authenticate a complete semantic-v3 success catalog."""
    try:
        root = _validate_tree_root(
            path,
            "tinyworlds-p-semantic-v3-catalog-tree",
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
            raise V3SemanticCatalogError("semantic-v3 catalog fields changed")
        content = {
            key: value for key, value in record.items() if key != "catalog_sha256"
        }
        if (
            record.get("benchmark_id") != V3_BENCHMARK_ID
            or record.get("format") != V3_CATALOG_FORMAT
            or record.get("schema_version") != V3_SCHEMA_VERSION
            or record.get("catalog_sha256") != catalog_sha
            or record_sha256(content) != catalog_sha
        ):
            raise V3SemanticCatalogError("semantic-v3 catalog identity changed")
        _validate_payload_hashes(root, record, success=True)
        config = v3_semantic_config_from_record(_mapping(record, "config"))
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(
            _semantic_word(item, encoder.dimension)
            for item in _iter_jsonl(root / "words.jsonl")
        )
        clusters = _load_clusters(root / "clusters.json", encoder.dimension)
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
        nonconstruction = _integer(record, "nonconstruction_token_count")
        retained = _integer(record, "retained_token_count")
        if sum(pairs.values()) != nonconstruction:
            raise V3SemanticCatalogError("semantic-v3 pair masses changed")
        _validate_calibration(words, calibration, config)
        _validate_success_quality(
            words,
            clusters,
            pairs,
            config,
            retained,
            nonconstruction,
        )
        parent = record.get("parent_catalog_sha256")
        if parent is not None and (type(parent) is not str or len(parent) != 64):
            raise V3SemanticCatalogError("semantic-v3 parent identity is malformed")
        _validate_trace(trace, config)
        if not trace or any(
            not items or items[-1].failing_word_count != 0
            for items in (
                [item for item in trace if item.role == "noun"],
                [item for item in trace if item.role == "verb"],
            )
        ):
            raise V3SemanticCatalogError("semantic-v3 success trace is incomplete")
        return V3SemanticCatalog(
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
    except V3SemanticCatalogError:
        raise
    except (OSError, TypeError, ValueError, V2SemanticCatalogError) as error:
        raise V3SemanticCatalogError("semantic-v3 catalog payload changed") from error


def load_v3_catalog_failure(path: str | Path) -> V3CatalogFailureArtifact:
    """Strictly authenticate a semantic-v3 failure audit and calibration."""
    try:
        root = _validate_tree_root(
            path,
            "tinyworlds-p-semantic-v3-catalog-failure-tree",
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
            raise V3SemanticCatalogError("semantic-v3 failure fields changed")
        content = {
            key: value for key, value in record.items() if key != "failure_sha256"
        }
        if (
            record.get("benchmark_id") != V3_BENCHMARK_ID
            or record.get("format") != V3_CATALOG_FAILURE_FORMAT
            or record.get("schema_version") != V3_SCHEMA_VERSION
            or record.get("failure_sha256") != failure_sha
            or record_sha256(content) != failure_sha
        ):
            raise V3SemanticCatalogError("semantic-v3 failure identity changed")
        _validate_payload_hashes(root, record, success=False)
        config = v3_semantic_config_from_record(_mapping(record, "config"))
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(_iter_jsonl(root / "words.jsonl"))
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
        if sum(pairs.values()) != _integer(record, "nonconstruction_token_count"):
            raise V3SemanticCatalogError("semantic-v3 failure pair masses changed")
        _validate_failure_calibration(words, calibration, config, encoder.dimension)
        _validate_failure_masses(words, pairs)
        _validate_trace(trace, config)
        tuple(_iter_jsonl(root / "representative-contexts.jsonl"))
        return V3CatalogFailureArtifact(
            root=root.resolve(),
            failure_sha256=failure_sha,
            evidence_sha256=_text(record, "evidence_sha256"),
            reason=_text(record, "reason"),
        )
    except V3SemanticCatalogError:
        raise
    except (OSError, TypeError, ValueError, V2SemanticCatalogError) as error:
        raise V3SemanticCatalogError("semantic-v3 failure payload changed") from error


def _semantic_word(
    record: Mapping[str, object],
    dimension: int,
) -> V3SemanticWord:
    if set(record) != set(V3SemanticWord.__dataclass_fields__):
        raise V3SemanticCatalogError("semantic-v3 word fields changed")
    vector_value = record.get("vector")
    vector = None
    if vector_value is not None:
        if type(vector_value) is not list or len(vector_value) != dimension:
            raise V3SemanticCatalogError("semantic-v3 word vector changed")
        vector = tuple(float(value) for value in vector_value)
    reason = record.get("exclusion_reason")
    if reason not in (
        None,
        "insufficient_contexts",
        "calibrated_role_outlier",
        "multiple_realized_senses",
        "cluster_boundary_margin",
    ):
        raise V3SemanticCatalogError("semantic-v3 exclusion reason changed")
    return V3SemanticWord(
        role=_role(record, "role"),
        word=_text(record, "word"),
        token_mass=_positive_integer(record, "token_mass"),
        context_count=_integer(record, "context_count"),
        calibration_fold=_optional_integer(record, "calibration_fold"),
        calibration_reference_count=_optional_integer(
            record,
            "calibration_reference_count",
        ),
        role_margin_q10=_optional_number(record, "role_margin_q10"),
        role_conformal_p=_optional_number(record, "role_conformal_p"),
        role_rejection_cutoff=_optional_number(record, "role_rejection_cutoff"),
        context_silhouette=_optional_number(record, "context_silhouette"),
        cluster_margin=_optional_number(record, "cluster_margin"),
        cluster=_optional_integer(record, "cluster"),
        exclusion_reason=cast(V3ExclusionReason | None, reason),
        vector=vector,
    )


def _validate_success_quality(
    words: Sequence[V3SemanticWord],
    clusters: Sequence[SemanticCluster],
    pairs: Mapping[tuple[str, str], int],
    config: V3SemanticConstructionConfig,
    retained: int,
    nonconstruction: int,
) -> None:
    expected_clusters = tuple(
        (role, index)
        for role in ("noun", "verb")
        for index in range(config.cluster_count)
    )
    if tuple((item.role, item.index) for item in clusters) != expected_clusters:
        raise V3SemanticCatalogError("semantic-v3 clusters are incomplete")
    roles = cast(tuple[Role, Role], ("noun", "verb"))
    retained_words = {
        role: {
            item.word
            for item in words
            if item.role == role and item.cluster is not None
        }
        for role in roles
    }
    measured_retained = sum(
        mass
        for (noun, verb), mass in pairs.items()
        if noun in retained_words["noun"] and verb in retained_words["verb"]
    )
    if (
        measured_retained != retained
        or retained / nonconstruction < config.minimum_retained_token_fraction
    ):
        raise V3SemanticCatalogError("semantic-v3 retained mass gate changed")
    by_cluster: dict[tuple[Role, int], list[V3SemanticWord]] = defaultdict(list)
    for word in words:
        score_pass = (
            word.role_conformal_p is not None
            and word.role_conformal_p > config.role_calibration_alpha
        )
        if word.exclusion_reason == "insufficient_contexts":
            valid = (
                word.context_count < config.minimum_contexts_per_word
                and word.role_margin_q10 is None
            )
        elif word.exclusion_reason == "calibrated_role_outlier":
            valid = (
                word.role_conformal_p is not None
                and not score_pass
                and word.context_silhouette is None
            )
        elif word.exclusion_reason == "multiple_realized_senses":
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette > config.maximum_context_silhouette
            )
        elif word.exclusion_reason == "cluster_boundary_margin":
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette <= config.maximum_context_silhouette
                and word.cluster_margin is not None
                and word.cluster_margin < config.minimum_cluster_margin
            )
        elif word.exclusion_reason is None:
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette <= config.maximum_context_silhouette
                and word.cluster_margin is not None
                and word.cluster_margin >= config.minimum_cluster_margin
            )
            if word.cluster is not None:
                by_cluster[(word.role, word.cluster)].append(word)
        else:
            valid = False
        if not valid:
            raise V3SemanticCatalogError("semantic-v3 word disposition violates its gates")
    for cluster in clusters:
        members = by_cluster[(cluster.role, cluster.index)]
        if tuple(sorted(item.word for item in members)) != cluster.words:
            raise V3SemanticCatalogError("semantic-v3 cluster membership changed")
        if sum(item.token_mass for item in members) != cluster.token_mass:
            raise V3SemanticCatalogError("semantic-v3 cluster audit mass changed")
        centroid = _normalized_centroid(
            np.asarray([cast(tuple[float, ...], item.vector) for item in members])
        )
        if not np.allclose(centroid, np.asarray(cluster.centroid), atol=1e-6):
            raise V3SemanticCatalogError("semantic-v3 cluster centroid changed")
    for role in roles:
        role_clusters = [item for item in clusters if item.role == role]
        minimum = (
            config.minimum_nouns_per_cluster
            if role == "noun"
            else config.minimum_verbs_per_cluster
        )
        if any(len(item.words) < minimum for item in role_clusters):
            raise V3SemanticCatalogError("semantic-v3 cluster word-count gate changed")
        centroids = np.asarray(
            [item.centroid for item in role_clusters],
            dtype=np.float64,
        )
        if max(
            float(centroids[left] @ centroids[right])
            for left in range(len(centroids))
            for right in range(left + 1, len(centroids))
        ) >= config.maximum_centroid_pair_cosine:
            raise V3SemanticCatalogError("semantic-v3 centroid-pair gate changed")
        for word in (
            item
            for item in words
            if item.role == role and item.exclusion_reason is None
        ):
            assert word.cluster is not None and word.vector is not None
            similarities = centroids @ np.asarray(word.vector, dtype=np.float64)
            nearest = int(np.argmax(similarities))
            if word.cluster != nearest:
                raise V3SemanticCatalogError(
                    "semantic-v3 retained word is not in its nearest cluster"
                )
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
                raise V3SemanticCatalogError("semantic-v3 nearest-cluster margin changed")


__all__ = [
    "V3CatalogProgress",
    "V3SemanticCatalogError",
    "V3SemanticGridError",
    "build_v3_catalog_from_evidence",
    "build_v3_semantic_catalog",
    "load_v3_catalog_failure",
    "load_v3_semantic_catalog",
]
