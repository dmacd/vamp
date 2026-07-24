"""Frozen-centroid v4 screening, publication, and strict replay loading."""

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
    SemanticGridError,
    SphericalClustering,
    WordVector,
    compose_word_vector,
    deterministic_two_means_silhouette,
    role_margin_quantile,
    semantic_first_spherical_kmeans,
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
    _optional_integer,
    _optional_number,
    _positive_integer,
    _prepare_working_directory,
    _publish_directory,
    _role,
    _text,
    _validate_calibration,
    _validate_failure_masses,
    _validate_input_masses,
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
)
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    CalibratedRoleScore,
    RoleCalibrationReference,
)
from apm.data.text.tinyworlds_p_semantic.v3_catalog import (
    V3SemanticCatalogError,
    load_v3_catalog_failure,
)
from apm.data.text.tinyworlds_p_semantic.v3_contracts import V3_SEMANTIC_CONFIG
from apm.data.text.tinyworlds_p_semantic.v4_audit import (
    render_v4_catalog_audits,
    render_v4_failure_audits,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import (
    V4_BENCHMARK_ID,
    V4_CATALOG_FAILURE_FORMAT,
    V4_CATALOG_FORMAT,
    V4_SCHEMA_VERSION,
    V4CatalogFailureArtifact,
    V4ExclusionReason,
    V4SemanticCatalog,
    V4SemanticConstructionConfig,
    V4SemanticWord,
    v4_semantic_config_from_record,
)


class V4SemanticCatalogError(ValueError):
    """A semantic-v4 catalog or failure bundle is malformed."""


class V4SemanticGridError(ValueError):
    """The frozen semantic-v4 construction failed and published evidence."""


V4CatalogProgress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class _ScreenedWord:
    evidence: WordEvidence
    calibration: CalibratedRoleScore | None
    silhouette: float | None
    vector: tuple[float, ...] | None
    reason: V4ExclusionReason | None


def build_v4_catalog_from_evidence(
    evidence: SemanticEvidenceArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V4SemanticConstructionConfig,
    *,
    progress: V4CatalogProgress | None = None,
) -> V4SemanticCatalog:
    """Reuse v1 vectors and authenticate the complete v3 pass-zero source."""
    if np.__version__ != config.construction_numpy_version:
        raise RuntimeError(
            "semantic-v4 construction requires NumPy "
            f"{config.construction_numpy_version}, found {np.__version__}"
        )
    if evidence.config.evidence_record() != config.evidence_record():
        raise ValueError("semantic-v4 evidence configuration changed")
    semantic_data_root = evidence.root.parents[2]
    source_root = (
        semantic_data_root
        / "catalog"
        / "v3"
        / "failures"
        / config.source_v3_failure_sha256
    )
    source = load_v3_catalog_failure(source_root)
    if (
        source.failure_sha256 != config.source_v3_failure_sha256
        or source.evidence_sha256 != evidence.evidence_sha256
    ):
        raise ValueError("semantic-v4 bound v3 source identity changed")
    source_words = {
        (_role(item, "role"), _text(item, "word")): item
        for item in _iter_jsonl(source.root / "words.jsonl")
    }
    source_trace = _load_trace(
        source.root / "boundary-trace.json",
        V3_SEMANTIC_CONFIG,
    )
    source_pass_zero = {
        item.role: item for item in source_trace if item.pass_index == 0
    }
    if set(source_pass_zero) != {"noun", "verb"}:
        raise ValueError("semantic-v4 v3 source lacks both pass-zero fits")
    _emit(progress, "evidence-load", 0, 1, "loading authenticated cached vectors")
    word_evidence, pair_masses = load_word_evidence(evidence)
    _emit(progress, "evidence-load", 1, 1, "cached word evidence loaded")
    return build_v4_semantic_catalog(
        word_evidence,
        pair_masses,
        evidence.encoder_identity,
        evidence.evidence_sha256,
        evidence.nonconstruction_token_count,
        output_root,
        temporary_directory,
        config,
        progress=progress,
        expected_v3_words=source_words,
        expected_v3_pass_zero=source_pass_zero,
    )


def build_v4_semantic_catalog(
    word_evidence: Sequence[WordEvidence],
    pair_masses: Mapping[tuple[str, str], int],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V4SemanticConstructionConfig,
    *,
    progress: V4CatalogProgress | None = None,
    expected_v3_words: Mapping[tuple[Role, str], Mapping[str, object]]
    | None = None,
    expected_v3_pass_zero: Mapping[Role, BoundaryPassMetric] | None = None,
) -> V4SemanticCatalog:
    """Fit once, freeze centroids, and apply the boundary screen exactly once."""
    canonical = tuple(sorted(word_evidence, key=lambda item: (item.role, item.word)))
    if len({(item.role, item.word) for item in canonical}) != len(canonical):
        raise ValueError("semantic-v4 word evidence contains duplicate role words")
    _validate_input_masses(canonical, pair_masses, nonconstruction_token_count)
    representative_contexts = {
        (item.role, item.word): item.contexts[
            : config.representative_contexts_per_cluster
        ]
        for item in canonical
        if item.contexts
    }
    raw_margins: dict[tuple[Role, str], float] = {}
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
    _emit(progress, "calibration", 0, 1, "replaying frozen role calibration")
    try:
        calibrated, calibration = calibrate_role_margins(raw_margins, config)
    except RoleCalibrationError as error:
        reason = f"role calibration failed: {error}"
        words = tuple(
            _calibration_failure_word(
                item,
                raw_margins.get((item.role, item.word)),
                config,
            )
            for item in canonical
        )
        failure = _publish_failure(
            words,
            (),
            (),
            (),
            pair_masses,
            representative_contexts,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            reason,
            output_root,
            temporary_directory,
            config,
            progress,
        )
        raise V4SemanticGridError(f"{reason}; failure audit: {failure.root}") from error
    _emit(progress, "calibration", 1, 1, "frozen role calibration replayed")

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
    if expected_v3_words is not None:
        _validate_v3_word_replay(screened, expected_v3_words)

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
    fits: dict[Role, SphericalClustering] = {}
    margins: dict[tuple[Role, str], float] = {}
    trace = []
    fit_failures = []
    _emit(progress, "clustering", 0, len(roles), "starting frozen pass-zero fits")
    for completed, role in enumerate(roles, start=1):
        try:
            fit = semantic_first_spherical_kmeans(
                candidates[role],
                config.cluster_count,
                maximum_iterations=config.maximum_centroid_iterations,
                benchmark_id=config.fit_hash_benchmark_id,
            )
        except SemanticGridError as error:
            fit_failures.append(f"{role} frozen fit failed: {error}")
            _emit(
                progress,
                "clustering",
                completed,
                len(roles),
                f"{role} frozen fit failed",
            )
            continue
        fit_margins = fit.margin_by_word(
            {item.word: item.vector for item in candidates[role]}
        )
        values = np.asarray(tuple(fit_margins.values()), dtype=np.float64)
        metric = BoundaryPassMetric(
            role=role,
            pass_index=0,
            input_word_count=len(candidates[role]),
            failing_word_count=sum(
                value < config.minimum_cluster_margin
                for value in fit_margins.values()
            ),
            cluster_masses=fit.cluster_masses,
            minimum_margin=float(np.min(values)),
            margin_q10=float(np.quantile(values, 0.10, method="linear")),
            median_margin=float(np.quantile(values, 0.50, method="linear")),
        )
        if (
            expected_v3_pass_zero is not None
            and metric.as_record() != expected_v3_pass_zero[role].as_record()
        ):
            raise ValueError(f"semantic-v4 did not exactly replay v3 {role} pass zero")
        fits[role] = fit
        trace.append(metric)
        margins.update(
            {(role, word): value for word, value in fit_margins.items()}
        )
        _emit(
            progress,
            "clustering",
            completed,
            len(roles),
            f"{role} frozen fit and one-shot screen complete",
        )

    assignments = {
        (role, word): cluster
        for role, fit in fits.items()
        for word, cluster in fit.assignments
    }
    words = tuple(
        _final_word(
            item,
            assignments,
            margins,
            fit_failed=item.reason is None and item.evidence.role not in fits,
            config=config,
        )
        for item in screened
    )
    fit_clusters = _fit_clusters(fits, candidates, config.cluster_count)
    if fit_failures:
        reason = "; ".join(fit_failures)
        failure = _publish_failure(
            words,
            fit_clusters,
            calibration,
            tuple(trace),
            pair_masses,
            representative_contexts,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            reason,
            output_root,
            temporary_directory,
            config,
            progress,
        )
        raise V4SemanticGridError(f"{reason}; failure audit: {failure.root}")

    reasons, retained_tokens = _gate_failures(
        words,
        fit_clusters,
        pair_masses,
        nonconstruction_token_count,
        config,
    )
    if reasons:
        reason = "; ".join(reasons)
        failure = _publish_failure(
            words,
            fit_clusters,
            calibration,
            tuple(trace),
            pair_masses,
            representative_contexts,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            reason,
            output_root,
            temporary_directory,
            config,
            progress,
        )
        raise V4SemanticGridError(f"{reason}; failure audit: {failure.root}")
    clusters = _retained_clusters(words, fit_clusters)
    return _publish_catalog(
        words,
        fit_clusters,
        clusters,
        calibration,
        tuple(trace),
        pair_masses,
        representative_contexts,
        encoder_identity,
        evidence_sha256,
        nonconstruction_token_count,
        retained_tokens,
        output_root,
        temporary_directory,
        config,
        progress,
    )


def _screen_word(
    evidence: WordEvidence,
    calibrated: Mapping[tuple[Role, str], CalibratedRoleScore],
    config: V4SemanticConstructionConfig,
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


def _calibration_failure_word(
    evidence: WordEvidence,
    raw_margin: float | None,
    config: V4SemanticConstructionConfig,
) -> V4SemanticWord:
    insufficient = evidence.context_embeddings.shape[0] < config.minimum_contexts_per_word
    return V4SemanticWord(
        role=evidence.role,
        word=evidence.word,
        token_mass=evidence.token_mass,
        context_count=evidence.context_embeddings.shape[0],
        calibration_fold=None,
        calibration_reference_count=None,
        role_margin_q10=raw_margin,
        role_conformal_p=None,
        role_rejection_cutoff=None,
        context_silhouette=None,
        fit_cluster=None,
        cluster_margin=None,
        cluster=None,
        exclusion_reason=(
            "insufficient_contexts" if insufficient else "role_calibration_failure"
        ),
        vector=None,
    )


def _final_word(
    screened: _ScreenedWord,
    assignments: Mapping[tuple[Role, str], int],
    margins: Mapping[tuple[Role, str], float],
    *,
    fit_failed: bool,
    config: V4SemanticConstructionConfig,
) -> V4SemanticWord:
    key = (screened.evidence.role, screened.evidence.word)
    fit_cluster = assignments.get(key)
    margin = margins.get(key)
    reason = screened.reason
    if fit_failed:
        reason = "semantic_fit_failure"
    elif reason is None and margin is not None and margin < config.minimum_cluster_margin:
        reason = "cluster_boundary_margin"
    score = screened.calibration
    return V4SemanticWord(
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
        fit_cluster=fit_cluster,
        cluster_margin=margin,
        cluster=fit_cluster if reason is None else None,
        exclusion_reason=cast(V4ExclusionReason | None, reason),
        vector=screened.vector,
    )


def _fit_clusters(
    fits: Mapping[Role, SphericalClustering],
    candidates: Mapping[Role, Sequence[WordVector]],
    cluster_count: int,
) -> tuple[SemanticCluster, ...]:
    masses = {
        (item.role, item.word): item.token_mass
        for role_items in candidates.values()
        for item in role_items
    }
    return tuple(
        SemanticCluster(
            role=role,
            index=index,
            token_mass=sum(
                masses[(role, word)]
                for word, cluster in fits[role].assignments
                if cluster == index
            ),
            centroid=fits[role].centroids[index],
            words=tuple(
                word
                for word, cluster in fits[role].assignments
                if cluster == index
            ),
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
        if role in fits
        for index in range(cluster_count)
    )


def _retained_clusters(
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
) -> tuple[SemanticCluster, ...]:
    by_cluster: dict[tuple[Role, int], list[V4SemanticWord]] = defaultdict(list)
    for item in words:
        if item.cluster is not None:
            by_cluster[(item.role, item.cluster)].append(item)
    return tuple(
        SemanticCluster(
            role=fit.role,
            index=fit.index,
            token_mass=sum(
                item.token_mass for item in by_cluster[(fit.role, fit.index)]
            ),
            centroid=fit.centroid,
            words=tuple(
                sorted(item.word for item in by_cluster[(fit.role, fit.index)])
            ),
        )
        for fit in fit_clusters
    )


def _gate_failures(
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
    pair_masses: Mapping[tuple[str, str], int],
    nonconstruction_token_count: int,
    config: V4SemanticConstructionConfig,
) -> tuple[tuple[str, ...], int]:
    reasons = []
    counts: dict[tuple[Role, int], int] = defaultdict(int)
    for item in words:
        if item.cluster is not None:
            counts[(item.role, item.cluster)] += 1
    for role in cast(tuple[Role, Role], ("noun", "verb")):
        minimum = (
            config.minimum_nouns_per_cluster
            if role == "noun"
            else config.minimum_verbs_per_cluster
        )
        failures = tuple(
            (index, counts[(role, index)])
            for index in range(config.cluster_count)
            if counts[(role, index)] < minimum
        )
        if failures:
            detail = ", ".join(f"{index}={count}" for index, count in failures)
            reasons.append(
                f"{role} fixed clusters retain fewer than {minimum} words: {detail}"
            )
        role_clusters = [item for item in fit_clusters if item.role == role]
        if len(role_clusters) == config.cluster_count:
            centroids = np.asarray(
                [item.centroid for item in role_clusters], dtype=np.float64
            )
            maximum_pair = max(
                float(centroids[left] @ centroids[right])
                for left in range(len(centroids))
                for right in range(left + 1, len(centroids))
            )
            if maximum_pair >= config.maximum_centroid_pair_cosine:
                reasons.append(
                    f"{role} fit-centroid pair cosine {maximum_pair:.9f} is not below "
                    f"{config.maximum_centroid_pair_cosine:.2f}"
                )
    retained_nouns = {
        item.word for item in words if item.role == "noun" and item.cluster is not None
    }
    retained_verbs = {
        item.word for item in words if item.role == "verb" and item.cluster is not None
    }
    retained_tokens = sum(
        mass
        for (noun, verb), mass in pair_masses.items()
        if noun in retained_nouns and verb in retained_verbs
    )
    retained_fraction = retained_tokens / nonconstruction_token_count
    if retained_fraction < config.minimum_retained_token_fraction:
        reasons.append(
            "both-role semantic exclusions retain "
            f"{retained_fraction:.9%}, below {config.minimum_retained_token_fraction:.0%}"
        )
    return tuple(reasons), retained_tokens


def _validate_v3_word_replay(
    screened: Sequence[_ScreenedWord],
    source_words: Mapping[tuple[Role, str], Mapping[str, object]],
) -> None:
    if set(source_words) != {
        (item.evidence.role, item.evidence.word) for item in screened
    }:
        raise ValueError("semantic-v4 v3 source word inventory changed")
    for item in screened:
        key = (item.evidence.role, item.evidence.word)
        source = source_words[key]
        score = item.calibration
        expected = {
            "calibration_fold": None if score is None else score.fold,
            "calibration_reference_count": (
                None if score is None else score.reference_count
            ),
            "context_count": item.evidence.context_embeddings.shape[0],
            "context_silhouette": item.silhouette,
            "role_conformal_p": None if score is None else score.conformal_p,
            "role_margin_q10": None if score is None else score.raw_margin,
            "role_rejection_cutoff": None if score is None else score.rejection_cutoff,
            "token_mass": item.evidence.token_mass,
            "vector": None if item.vector is None else list(item.vector),
        }
        if any(source.get(name) != value for name, value in expected.items()):
            raise ValueError(f"semantic-v4 did not exactly replay v3 word {key}")
        disposition = source.get("disposition")
        if item.reason is None:
            valid = disposition in ("cluster_boundary_margin", "semantic_grid_failure")
        else:
            valid = disposition == item.reason
        if not valid:
            raise ValueError(f"semantic-v4 v3 disposition changed for {key}")


def _publish_catalog(
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
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
    config: V4SemanticConstructionConfig,
    progress: V4CatalogProgress | None,
) -> V4SemanticCatalog:
    _emit(progress, "publication", 0, 1, "writing semantic-v4 catalog and audits")
    working = _prepare_working_directory(temporary_directory)
    _write_jsonl(working / "words.jsonl", (item.as_record() for item in words))
    _write_clusters(working / "fit-clusters.json", fit_clusters)
    _write_clusters(working / "clusters.json", clusters)
    _write_calibration(working / "calibration.json", config, calibration)
    _write_trace(working / "boundary-trace.json", trace)
    _write_pair_masses(working / "role-pair-masses.jsonl", pair_masses)
    _write_contexts(working / "representative-contexts.jsonl", representative_contexts)
    content = {
        "benchmark_id": V4_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "clusters_sha256": _file_sha256(working / "clusters.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "fit_clusters_sha256": _file_sha256(working / "fit-clusters.json"),
        "format": V4_CATALOG_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "retained_token_count": retained_token_count,
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": V4_SCHEMA_VERSION,
        "source_v3_failure_sha256": config.source_v3_failure_sha256,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    catalog_sha = record_sha256(content)
    _write_json(working / "catalog.json", {**content, "catalog_sha256": catalog_sha})
    markdown, html = render_v4_catalog_audits(
        catalog_sha,
        evidence_sha256,
        config,
        calibration,
        trace,
        words,
        fit_clusters,
        clusters,
        pair_masses,
        representative_contexts,
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(
        working,
        "tinyworlds-p-semantic-v4-catalog-tree",
        "catalog_sha256",
        catalog_sha,
    )
    target = _publish_directory(working, output_root, catalog_sha)
    catalog = load_v4_semantic_catalog(target)
    _emit(progress, "publication", 1, 1, "semantic-v4 catalog strictly reloaded")
    return catalog


def _publish_failure(
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
    calibration: Sequence[RoleCalibrationReference],
    trace: Sequence[BoundaryPassMetric],
    pair_masses: Mapping[tuple[str, str], int],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    reason: str,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: V4SemanticConstructionConfig,
    progress: V4CatalogProgress | None,
) -> V4CatalogFailureArtifact:
    _emit(progress, "publication", 0, 1, "writing semantic-v4 failure audit")
    working = _prepare_working_directory(temporary_directory)
    _write_jsonl(working / "words.jsonl", (item.as_record() for item in words))
    _write_clusters(working / "fit-clusters.json", fit_clusters)
    _write_calibration(working / "calibration.json", config, calibration)
    _write_trace(working / "boundary-trace.json", trace)
    _write_pair_masses(working / "role-pair-masses.jsonl", pair_masses)
    _write_contexts(working / "representative-contexts.jsonl", representative_contexts)
    content = {
        "benchmark_id": V4_BENCHMARK_ID,
        "boundary_trace_sha256": _file_sha256(working / "boundary-trace.json"),
        "calibration_sha256": _file_sha256(working / "calibration.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "failure_sha256": "",
        "fit_clusters_sha256": _file_sha256(working / "fit-clusters.json"),
        "format": V4_CATALOG_FAILURE_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "reason": reason,
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": V4_SCHEMA_VERSION,
        "source_v3_failure_sha256": config.source_v3_failure_sha256,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    identity_content = {key: value for key, value in content.items() if key != "failure_sha256"}
    failure_sha = record_sha256(identity_content)
    content["failure_sha256"] = failure_sha
    _write_json(working / "failure.json", content)
    markdown, html = render_v4_failure_audits(
        failure_sha,
        evidence_sha256,
        reason,
        config,
        calibration,
        trace,
        words,
        fit_clusters,
        representative_contexts,
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(
        working,
        "tinyworlds-p-semantic-v4-catalog-failure-tree",
        "failure_sha256",
        failure_sha,
    )
    failure_root = Path(output_root) / "failures"
    target = failure_root / failure_sha
    if target.exists():
        _discard_empty_or_staged(working)
        failure = load_v4_catalog_failure(target)
        _emit(progress, "publication", 1, 1, "existing failure audit strictly reloaded")
        return failure
    failure_root.mkdir(parents=True, exist_ok=True)
    target = _publish_directory(working, failure_root, failure_sha)
    failure = load_v4_catalog_failure(target)
    _emit(progress, "publication", 1, 1, "semantic-v4 failure audit strictly reloaded")
    return failure


def load_v4_semantic_catalog(path: str | Path) -> V4SemanticCatalog:
    """Strictly authenticate and reconstruct a semantic-v4 success catalog."""
    try:
        root = _validate_tree_root(
            path,
            "tinyworlds-p-semantic-v4-catalog-tree",
            "catalog_sha256",
            (
                "audit.html",
                "audit.md",
                "boundary-trace.json",
                "calibration.json",
                "catalog.json",
                "clusters.json",
                "fit-clusters.json",
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
            "fit_clusters_sha256",
            "format",
            "nonconstruction_token_count",
            "representative_contexts_sha256",
            "retained_token_count",
            "role_pair_masses_sha256",
            "schema_version",
            "source_v3_failure_sha256",
            "words_sha256",
        }
        if set(record) != required:
            raise V4SemanticCatalogError("semantic-v4 catalog fields changed")
        content = {
            key: value for key, value in record.items() if key != "catalog_sha256"
        }
        if (
            record.get("benchmark_id") != V4_BENCHMARK_ID
            or record.get("format") != V4_CATALOG_FORMAT
            or record.get("schema_version") != V4_SCHEMA_VERSION
            or record.get("catalog_sha256") != catalog_sha
            or record_sha256(content) != catalog_sha
        ):
            raise V4SemanticCatalogError("semantic-v4 catalog identity changed")
        _validate_payload_hashes(root, record, success=True)
        config = v4_semantic_config_from_record(_mapping(record, "config"))
        _validate_source_binding(record, config)
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(
            _semantic_word(item, encoder.dimension)
            for item in _iter_jsonl(root / "words.jsonl")
        )
        fit_clusters = _load_clusters(root / "fit-clusters.json", encoder.dimension)
        clusters = _load_clusters(root / "clusters.json", encoder.dimension)
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
        nonconstruction = _positive_integer(record, "nonconstruction_token_count")
        retained = _positive_integer(record, "retained_token_count")
        if sum(pairs.values()) != nonconstruction:
            raise V4SemanticCatalogError("semantic-v4 pair masses changed")
        _validate_failure_masses(
            tuple(item.as_record() for item in words),
            pairs,
        )
        _validate_calibration(words, calibration, config)
        _validate_word_dispositions(
            words,
            config,
            allow_fit_failure=False,
            allow_calibration_failure=False,
        )
        _validate_success_quality(
            words,
            fit_clusters,
            clusters,
            trace,
            pairs,
            retained,
            nonconstruction,
            config,
        )
        tuple(_iter_jsonl(root / "representative-contexts.jsonl"))
        return V4SemanticCatalog(
            root=root.resolve(),
            catalog_sha256=catalog_sha,
            evidence_sha256=_text(record, "evidence_sha256"),
            encoder_identity=encoder,
            config=config,
            calibration=calibration,
            words=words,
            fit_clusters=fit_clusters,
            clusters=clusters,
            retained_token_count=retained,
            nonconstruction_token_count=nonconstruction,
        )
    except V4SemanticCatalogError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        V2SemanticCatalogError,
        V3SemanticCatalogError,
    ) as error:
        raise V4SemanticCatalogError("semantic-v4 catalog payload changed") from error


def load_v4_catalog_failure(path: str | Path) -> V4CatalogFailureArtifact:
    """Strictly authenticate and replay a semantic-v4 failure bundle."""
    try:
        root = _validate_tree_root(
            path,
            "tinyworlds-p-semantic-v4-catalog-failure-tree",
            "failure_sha256",
            (
                "audit.html",
                "audit.md",
                "boundary-trace.json",
                "calibration.json",
                "failure.json",
                "fit-clusters.json",
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
            "fit_clusters_sha256",
            "format",
            "nonconstruction_token_count",
            "reason",
            "representative_contexts_sha256",
            "role_pair_masses_sha256",
            "schema_version",
            "source_v3_failure_sha256",
            "words_sha256",
        }
        if set(record) != required:
            raise V4SemanticCatalogError("semantic-v4 failure fields changed")
        content = {
            key: value for key, value in record.items() if key != "failure_sha256"
        }
        if (
            record.get("benchmark_id") != V4_BENCHMARK_ID
            or record.get("format") != V4_CATALOG_FAILURE_FORMAT
            or record.get("schema_version") != V4_SCHEMA_VERSION
            or record.get("failure_sha256") != failure_sha
            or record_sha256(content) != failure_sha
        ):
            raise V4SemanticCatalogError("semantic-v4 failure identity changed")
        _validate_payload_hashes(root, record, success=False)
        config = v4_semantic_config_from_record(_mapping(record, "config"))
        _validate_source_binding(record, config)
        encoder = _encoder_identity(_mapping(record, "encoder"))
        calibration = _load_calibration(root / "calibration.json", config)
        trace = _load_trace(root / "boundary-trace.json", config)
        words = tuple(
            _semantic_word(item, encoder.dimension)
            for item in _iter_jsonl(root / "words.jsonl")
        )
        fit_clusters = _load_clusters(root / "fit-clusters.json", encoder.dimension)
        pairs = _load_pairs(root / "role-pair-masses.jsonl")
        nonconstruction = _positive_integer(record, "nonconstruction_token_count")
        if sum(pairs.values()) != nonconstruction:
            raise V4SemanticCatalogError("semantic-v4 failure pair masses changed")
        _validate_failure_masses(
            tuple(item.as_record() for item in words),
            pairs,
        )
        reason = _text(record, "reason")
        calibration_failed = any(
            item.exclusion_reason == "role_calibration_failure" for item in words
        )
        if calibration_failed:
            _validate_failed_calibration(words, calibration, config)
        else:
            _validate_calibration(words, calibration, config)
        _validate_word_dispositions(
            words,
            config,
            allow_fit_failure=True,
            allow_calibration_failure=calibration_failed,
        )
        if calibration_failed:
            if (
                fit_clusters
                or trace
                or not reason.startswith("role calibration failed: ")
            ):
                raise V4SemanticCatalogError(
                    "semantic-v4 calibration failure evidence changed"
                )
        elif not any(
            item.exclusion_reason == "semantic_fit_failure" for item in words
        ):
            expected_fit, expected_trace = _replay_fit(words, config)
            if fit_clusters != expected_fit or tuple(trace) != expected_trace:
                raise V4SemanticCatalogError("semantic-v4 failure frozen fit changed")
            expected_reasons, _ = _gate_failures(
                words,
                fit_clusters,
                pairs,
                nonconstruction,
                config,
            )
            if reason != "; ".join(expected_reasons):
                raise V4SemanticCatalogError("semantic-v4 failure reason changed")
        tuple(_iter_jsonl(root / "representative-contexts.jsonl"))
        return V4CatalogFailureArtifact(
            root=root.resolve(),
            failure_sha256=failure_sha,
            evidence_sha256=_text(record, "evidence_sha256"),
            reason=reason,
        )
    except V4SemanticCatalogError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        V2SemanticCatalogError,
        V3SemanticCatalogError,
    ) as error:
        raise V4SemanticCatalogError("semantic-v4 failure payload changed") from error


def _validate_success_quality(
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
    clusters: Sequence[SemanticCluster],
    trace: Sequence[BoundaryPassMetric],
    pairs: Mapping[tuple[str, str], int],
    retained: int,
    nonconstruction: int,
    config: V4SemanticConstructionConfig,
) -> None:
    expected_keys = tuple(
        (role, index)
        for role in ("noun", "verb")
        for index in range(config.cluster_count)
    )
    if tuple((item.role, item.index) for item in fit_clusters) != expected_keys:
        raise V4SemanticCatalogError("semantic-v4 fit clusters are incomplete")
    if tuple((item.role, item.index) for item in clusters) != expected_keys:
        raise V4SemanticCatalogError("semantic-v4 retained clusters are incomplete")
    expected_fit, expected_trace = _replay_fit(words, config)
    if fit_clusters != expected_fit:
        raise V4SemanticCatalogError("semantic-v4 frozen fit changed")
    if tuple(trace) != expected_trace:
        raise V4SemanticCatalogError("semantic-v4 one-shot trace changed")
    expected_clusters = _retained_clusters(words, fit_clusters)
    if tuple(clusters) != expected_clusters:
        raise V4SemanticCatalogError("semantic-v4 retained cluster inventory changed")
    reasons, measured_retained = _gate_failures(
        words,
        fit_clusters,
        pairs,
        nonconstruction,
        config,
    )
    if reasons or measured_retained != retained:
        raise V4SemanticCatalogError("semantic-v4 catalog no longer passes its gates")


def _replay_fit(
    words: Sequence[V4SemanticWord],
    config: V4SemanticConstructionConfig,
) -> tuple[tuple[SemanticCluster, ...], tuple[BoundaryPassMetric, ...]]:
    roles = cast(tuple[Role, Role], ("noun", "verb"))
    candidates = {
        role: tuple(
            WordVector(
                role=role,
                word=item.word,
                token_mass=item.token_mass,
                vector=cast(tuple[float, ...], item.vector),
            )
            for item in words
            if item.role == role and item.vector is not None
        )
        for role in roles
    }
    fits = {}
    trace = []
    for role in roles:
        fit = semantic_first_spherical_kmeans(
            candidates[role],
            config.cluster_count,
            maximum_iterations=config.maximum_centroid_iterations,
            benchmark_id=config.fit_hash_benchmark_id,
        )
        margins = fit.margin_by_word(
            {item.word: item.vector for item in candidates[role]}
        )
        values = np.asarray(tuple(margins.values()), dtype=np.float64)
        trace.append(
            BoundaryPassMetric(
                role=role,
                pass_index=0,
                input_word_count=len(candidates[role]),
                failing_word_count=sum(
                    margin < config.minimum_cluster_margin
                    for margin in margins.values()
                ),
                cluster_masses=fit.cluster_masses,
                minimum_margin=float(np.min(values)),
                margin_q10=float(np.quantile(values, 0.10, method="linear")),
                median_margin=float(np.quantile(values, 0.50, method="linear")),
            )
        )
        assignments = dict(fit.assignments)
        for item in candidates[role]:
            word = next(word for word in words if word.role == role and word.word == item.word)
            measured_margin = margins[item.word]
            if (
                word.fit_cluster != assignments[item.word]
                or word.cluster_margin is None
                or not np.isclose(word.cluster_margin, measured_margin, atol=1e-12)
            ):
                raise V4SemanticCatalogError(
                    "semantic-v4 word frozen assignment or margin changed"
                )
        fits[role] = fit
    return _fit_clusters(fits, candidates, config.cluster_count), tuple(trace)


def _validate_failed_calibration(
    words: Sequence[V4SemanticWord],
    calibration: Sequence[RoleCalibrationReference],
    config: V4SemanticConstructionConfig,
) -> None:
    if calibration:
        raise V4SemanticCatalogError(
            "semantic-v4 failed calibration unexpectedly has references"
        )
    raw = {
        (item.role, item.word): cast(float, item.role_margin_q10)
        for item in words
        if item.role_margin_q10 is not None
    }
    try:
        calibrate_role_margins(raw, config)
    except RoleCalibrationError:
        return
    raise V4SemanticCatalogError("semantic-v4 recorded a calibration failure that passes")


def _validate_word_dispositions(
    words: Sequence[V4SemanticWord],
    config: V4SemanticConstructionConfig,
    *,
    allow_fit_failure: bool,
    allow_calibration_failure: bool,
) -> None:
    identities = tuple((item.role, item.word) for item in words)
    if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
        raise V4SemanticCatalogError("semantic-v4 words are not canonical")
    for word in words:
        score_pass = (
            word.role_conformal_p is not None
            and word.role_conformal_p > config.role_calibration_alpha
        )
        no_fit = (
            word.fit_cluster is None
            and word.cluster_margin is None
            and word.cluster is None
        )
        if word.exclusion_reason == "insufficient_contexts":
            valid = (
                word.context_count < config.minimum_contexts_per_word
                and word.role_margin_q10 is None
                and word.context_silhouette is None
                and word.vector is None
                and no_fit
            )
        elif word.exclusion_reason == "calibrated_role_outlier":
            valid = (
                word.role_conformal_p is not None
                and not score_pass
                and word.context_silhouette is None
                and word.vector is None
                and no_fit
            )
        elif word.exclusion_reason == "role_calibration_failure":
            valid = (
                allow_calibration_failure
                and word.context_count >= config.minimum_contexts_per_word
                and word.role_margin_q10 is not None
                and word.calibration_fold is None
                and word.calibration_reference_count is None
                and word.role_conformal_p is None
                and word.role_rejection_cutoff is None
                and word.context_silhouette is None
                and word.vector is None
                and no_fit
            )
        elif word.exclusion_reason == "multiple_realized_senses":
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette > config.maximum_context_silhouette
                and word.vector is None
                and no_fit
            )
        elif word.exclusion_reason == "cluster_boundary_margin":
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette <= config.maximum_context_silhouette
                and word.vector is not None
                and word.fit_cluster is not None
                and word.fit_cluster < config.cluster_count
                and word.cluster_margin is not None
                and word.cluster_margin < config.minimum_cluster_margin
                and word.cluster is None
            )
        elif word.exclusion_reason == "semantic_fit_failure":
            valid = (
                allow_fit_failure
                and score_pass
                and word.context_silhouette is not None
                and word.context_silhouette <= config.maximum_context_silhouette
                and word.vector is not None
                and no_fit
            )
        elif word.exclusion_reason is None:
            valid = (
                score_pass
                and word.context_silhouette is not None
                and word.context_silhouette <= config.maximum_context_silhouette
                and word.vector is not None
                and word.fit_cluster is not None
                and word.fit_cluster < config.cluster_count
                and word.cluster_margin is not None
                and word.cluster_margin >= config.minimum_cluster_margin
                and word.cluster == word.fit_cluster
            )
        else:
            valid = False
        if not valid:
            raise V4SemanticCatalogError("semantic-v4 word disposition violates its gates")


def _semantic_word(
    record: Mapping[str, object],
    dimension: int,
) -> V4SemanticWord:
    if set(record) != set(V4SemanticWord.__dataclass_fields__):
        raise V4SemanticCatalogError("semantic-v4 word fields changed")
    vector_value = record.get("vector")
    vector = None
    if vector_value is not None:
        if type(vector_value) is not list or len(vector_value) != dimension:
            raise V4SemanticCatalogError("semantic-v4 word vector changed")
        if any(type(value) not in (int, float) for value in vector_value):
            raise V4SemanticCatalogError("semantic-v4 word vector type changed")
        vector = tuple(float(value) for value in vector_value)
    reason = record.get("exclusion_reason")
    if reason not in (
        None,
        "insufficient_contexts",
        "calibrated_role_outlier",
        "role_calibration_failure",
        "multiple_realized_senses",
        "cluster_boundary_margin",
        "semantic_fit_failure",
    ):
        raise V4SemanticCatalogError("semantic-v4 exclusion reason changed")
    return V4SemanticWord(
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
        fit_cluster=_optional_integer(record, "fit_cluster"),
        cluster_margin=_optional_number(record, "cluster_margin"),
        cluster=_optional_integer(record, "cluster"),
        exclusion_reason=cast(V4ExclusionReason | None, reason),
        vector=vector,
    )


def _validate_payload_hashes(
    root: Path,
    record: Mapping[str, object],
    *,
    success: bool,
) -> None:
    pairs = (
        ("boundary-trace.json", "boundary_trace_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("fit-clusters.json", "fit_clusters_sha256"),
        ("representative-contexts.jsonl", "representative_contexts_sha256"),
        ("role-pair-masses.jsonl", "role_pair_masses_sha256"),
        ("words.jsonl", "words_sha256"),
    ) + (("clusters.json", "clusters_sha256"),) * success
    for filename, field in pairs:
        if _file_sha256(root / filename) != record.get(field):
            raise V4SemanticCatalogError(f"semantic-v4 payload changed: {filename}")


def _validate_source_binding(
    record: Mapping[str, object],
    config: V4SemanticConstructionConfig,
) -> None:
    if record.get("source_v3_failure_sha256") != config.source_v3_failure_sha256:
        raise V4SemanticCatalogError("semantic-v4 v3 source binding changed")


def _write_clusters(path: Path, clusters: Sequence[SemanticCluster]) -> None:
    _write_json(path, {"clusters": [item.as_record() for item in clusters]})


__all__ = [
    "V4CatalogProgress",
    "V4SemanticCatalogError",
    "V4SemanticGridError",
    "build_v4_catalog_from_evidence",
    "build_v4_semantic_catalog",
    "load_v4_catalog_failure",
    "load_v4_semantic_catalog",
]
