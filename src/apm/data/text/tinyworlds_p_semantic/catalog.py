"""Semantic word filtering, clustering, catalog publication, and strict loading."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import cast

import numpy as np

from apm.data.text.tinyworlds_p_semantic.audit import (
    CatalogFailureWord,
    render_catalog_audits,
    render_catalog_failure_audits,
)
from apm.data.text.tinyworlds_p_semantic.clustering import (
    SemanticGridError,
    WordVector,
    cluster_with_boundary_exclusions,
    compose_word_vector,
    deterministic_two_means_silhouette,
    role_margin_quantile,
    validate_cluster_quality,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    CATALOG_FORMAT,
    SCHEMA_VERSION,
    EncoderIdentity,
    ExclusionReason,
    ModelFile,
    Role,
    SemanticCatalog,
    SemanticCluster,
    SemanticConstructionConfig,
    SemanticEvidenceArtifact,
    SemanticWord,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
    semantic_config_from_record,
)
from apm.data.text.tinyworlds_p_semantic.evidence import (
    SemanticEvidenceError,
    load_evidence_arrays,
    load_role_pair_masses,
)


class SemanticCatalogError(ValueError):
    """A semantic catalog is malformed or its content identity changed."""


@dataclass(frozen=True, slots=True)
class SemanticCatalogFailureArtifact:
    """Immutable audit evidence for an automated semantic-grid construction stop."""

    root: Path
    failure_sha256: str
    evidence_sha256: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir() or not self.reason:
            raise ValueError("semantic catalog failure artifact is malformed")
        require_sha256(self.failure_sha256, "semantic catalog failure identity")
        require_sha256(self.evidence_sha256, "semantic failure evidence identity")


@dataclass(frozen=True, slots=True)
class WordEvidence:
    """Reusable anchor/context vectors for one role-specific archive word."""

    role: Role
    word: str
    token_mass: int
    target_anchor_embeddings: np.ndarray
    opposite_anchor_embeddings: np.ndarray
    context_embeddings: np.ndarray
    contexts: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("word evidence requires a role and word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("word evidence token mass must be positive")
        matrices = (
            self.target_anchor_embeddings,
            self.opposite_anchor_embeddings,
            self.context_embeddings,
        )
        if any(type(matrix) is not np.ndarray or matrix.ndim != 2 for matrix in matrices):
            raise TypeError("word evidence embeddings must be matrices")
        dimensions = {matrix.shape[1] for matrix in matrices}
        if len(dimensions) != 1 or not dimensions or 0 in dimensions:
            raise ValueError("word evidence dimensions must agree")
        if self.target_anchor_embeddings.shape[0] != 3 or self.opposite_anchor_embeddings.shape[0] != 3:
            raise ValueError("word evidence requires three anchors for each role")
        if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
            raise ValueError("word evidence embeddings must be finite")


@dataclass(frozen=True, slots=True)
class _ScreenedWord:
    evidence: WordEvidence
    role_margin: float | None
    silhouette: float | None
    vector: tuple[float, ...] | None
    reason: ExclusionReason | None


def build_catalog_from_evidence(
    evidence: SemanticEvidenceArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: SemanticConstructionConfig,
    *,
    parent_catalog: SemanticCatalog | None = None,
) -> SemanticCatalog:
    """Derive a threshold-specific catalog while reusing cached MiniLM vectors."""
    word_evidence, pair_masses = load_word_evidence(evidence)
    return build_semantic_catalog(
        word_evidence,
        pair_masses,
        evidence.encoder_identity,
        evidence.evidence_sha256,
        evidence.nonconstruction_token_count,
        output_root,
        temporary_directory,
        config,
        parent_catalog=parent_catalog,
    )


def load_word_evidence(
    evidence: SemanticEvidenceArtifact,
) -> tuple[tuple[WordEvidence, ...], dict[tuple[str, str], int]]:
    """Materialize authenticated per-word vectors and role-pair masses once."""
    embeddings, index_records, context_records = load_evidence_arrays(evidence)
    pair_masses = load_role_pair_masses(evidence)
    anchor_rows: dict[tuple[Role, str, Role], list[int]] = defaultdict(list)
    context_rows: dict[tuple[Role, str], list[int]] = defaultdict(list)
    for expected_row, record in enumerate(index_records):
        row = _integer(record, "row")
        if row != expected_row:
            raise SemanticEvidenceError("semantic embedding rows are not contiguous")
        role = _role(record, "role")
        word = _text(record, "word")
        if record.get("kind") == "anchor":
            anchor_rows[(role, word, _role(record, "anchor_role"))].append(row)
        elif record.get("kind") == "context":
            context_rows[(role, word)].append(row)
        else:
            raise SemanticEvidenceError("semantic embedding kind is invalid")
    contexts_by_key: dict[tuple[Role, str], list[dict[str, object]]] = defaultdict(list)
    for record in context_records:
        contexts_by_key[(_role(record, "role"), _text(record, "word"))].append(record)
    noun_masses: Counter[str] = Counter()
    verb_masses: Counter[str] = Counter()
    for (noun, verb), mass in pair_masses.items():
        noun_masses[noun] += mass
        verb_masses[verb] += mass
    word_evidence = tuple(
        WordEvidence(
            role=role,
            word=word,
            token_mass=mass,
            target_anchor_embeddings=np.asarray(
                embeddings[anchor_rows[(role, word, role)]], dtype=np.float32
            ),
            opposite_anchor_embeddings=np.asarray(
                embeddings[
                    anchor_rows[(role, word, cast(Role, "verb" if role == "noun" else "noun"))]
                ],
                dtype=np.float32,
            ),
            context_embeddings=np.asarray(
                embeddings[context_rows.get((role, word), ())], dtype=np.float32
            ).reshape((-1, evidence.dimension)),
            contexts=tuple(
                sorted(
                    contexts_by_key.get((role, word), ()),
                    key=lambda item: _text(item, "selection_sha256"),
                )
            ),
        )
        for role, masses in (("noun", noun_masses), ("verb", verb_masses))
        for word, mass in sorted(masses.items())
    )
    return word_evidence, pair_masses


def build_semantic_catalog(
    word_evidence: Sequence[WordEvidence],
    pair_masses: Mapping[tuple[str, str], int],
    encoder_identity: EncoderIdentity,
    evidence_sha256: str,
    nonconstruction_token_count: int,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: SemanticConstructionConfig,
    *,
    parent_catalog: SemanticCatalog | None = None,
) -> SemanticCatalog:
    """Apply all semantic gates and atomically publish a content-addressed catalog."""
    canonical = tuple(sorted(word_evidence, key=lambda item: (item.role, item.word)))
    if len({(item.role, item.word) for item in canonical}) != len(canonical):
        raise ValueError("semantic word evidence contains duplicate role words")
    if sum(pair_masses.values()) != nonconstruction_token_count:
        raise ValueError("role-pair mass does not cover the non-construction archive")
    expected_masses: dict[Role, Counter[str]] = {
        "noun": Counter(),
        "verb": Counter(),
    }
    for (noun, verb), mass in pair_masses.items():
        if (
            type(noun) is not str
            or not noun
            or type(verb) is not str
            or not verb
            or type(mass) is not int
            or mass <= 0
        ):
            raise ValueError("role-pair masses require words and positive integer mass")
        expected_masses["noun"][noun] += mass
        expected_masses["verb"][verb] += mass
    measured_masses = {
        role: Counter(
            {item.word: item.token_mass for item in canonical if item.role == role}
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    if measured_masses != expected_masses:
        raise ValueError("word evidence masses differ from role-pair archive mass")
    screened = tuple(_screen_word(item, config) for item in canonical)
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
    try:
        words, clusters, retained_tokens = _cluster_screened_words(
            screened,
            candidates,
            pair_masses,
            nonconstruction_token_count,
            config,
        )
    except SemanticGridError as error:
        failure = _publish_catalog_failure(
            screened,
            pair_masses,
            representative_contexts,
            encoder_identity,
            evidence_sha256,
            nonconstruction_token_count,
            str(error),
            output_root,
            temporary_directory,
            config,
        )
        raise SemanticGridError(f"{error}; failure audit: {failure.root}") from error
    by_word = {(item.role, item.word): item for item in words}
    working = Path(temporary_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"semantic catalog temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    _write_jsonl(working / "words.jsonl", (item.as_record() for item in words))
    _write_json(working / "clusters.json", {"clusters": [item.as_record() for item in clusters]})
    _write_jsonl(
        working / "role-pair-masses.jsonl",
        (
            {"noun": noun, "token_mass": mass, "verb": verb}
            for (noun, verb), mass in sorted(pair_masses.items())
        ),
    )
    _write_jsonl(
        working / "representative-contexts.jsonl",
        (
            record
            for key in sorted(representative_contexts)
            for record in representative_contexts[key]
        ),
    )
    content = {
        "clusters_sha256": _file_sha256(working / "clusters.json"),
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": CATALOG_FORMAT,
        "nonconstruction_token_count": nonconstruction_token_count,
        "parent_catalog_sha256": None if parent_catalog is None else parent_catalog.catalog_sha256,
        "representative_contexts_sha256": _file_sha256(working / "representative-contexts.jsonl"),
        "retained_token_count": retained_tokens,
        "role_pair_masses_sha256": _file_sha256(working / "role-pair-masses.jsonl"),
        "schema_version": SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    catalog_sha256 = record_sha256(content)
    _write_json(working / "catalog.json", {**content, "catalog_sha256": catalog_sha256})
    markdown, html = render_catalog_audits(
        catalog_sha256,
        evidence_sha256,
        config,
        words,
        clusters,
        pair_masses,
        representative_contexts,
        None if parent_catalog is None else parent_catalog.words,
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_tree(working, catalog_sha256)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    target = output / catalog_sha256
    if target.exists():
        raise FileExistsError(f"semantic catalog already exists: {target}")
    os.rename(working, target)
    _fsync_directory(output)
    catalog = load_semantic_catalog(target)
    if {(item.role, item.word): item for item in catalog.words} != by_word:
        raise RuntimeError("semantic catalog changed during strict publication reload")
    return catalog


def load_semantic_catalog(path: str | Path) -> SemanticCatalog:
    """Strictly authenticate all catalog files, identities, and cluster memberships."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SemanticCatalogError("semantic catalog must be a regular directory")
    tree = _load_json(root / "tree.json")
    if set(tree) != {"catalog_sha256", "files", "format", "schema_version"}:
        raise SemanticCatalogError("semantic catalog tree fields changed")
    if tree["format"] != "tinyworlds-p-semantic-catalog-tree" or tree["schema_version"] != 1:
        raise SemanticCatalogError("unsupported semantic catalog tree")
    catalog_sha = _text(tree, "catalog_sha256")
    if root.name != catalog_sha:
        raise SemanticCatalogError("semantic catalog directory identity changed")
    _validate_tree(root, tree)
    record = _load_json(root / "catalog.json")
    required = {
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
    if set(record) != required or record["format"] != CATALOG_FORMAT or record["schema_version"] != 1:
        raise SemanticCatalogError("semantic catalog identity contract changed")
    content = {key: value for key, value in record.items() if key != "catalog_sha256"}
    if record.get("catalog_sha256") != catalog_sha or record_sha256(content) != catalog_sha:
        raise SemanticCatalogError("semantic catalog content identity is inconsistent")
    for filename, hash_field in (
        ("words.jsonl", "words_sha256"),
        ("clusters.json", "clusters_sha256"),
        ("role-pair-masses.jsonl", "role_pair_masses_sha256"),
        ("representative-contexts.jsonl", "representative_contexts_sha256"),
    ):
        if _file_sha256(root / filename) != record[hash_field]:
            raise SemanticCatalogError(f"semantic catalog file changed: {filename}")
    try:
        config = semantic_config_from_record(_mapping(record, "config"))
        encoder = _encoder_identity(_mapping(record, "encoder"))
    except (TypeError, ValueError) as error:
        raise SemanticCatalogError("semantic catalog contract is invalid") from error
    words = tuple(_semantic_word(item, encoder.dimension) for item in _iter_jsonl(root / "words.jsonl"))
    cluster_payload = _load_json(root / "clusters.json")
    if set(cluster_payload) != {"clusters"} or type(cluster_payload["clusters"]) is not list:
        raise SemanticCatalogError("semantic cluster payload changed")
    clusters = tuple(_semantic_cluster(item, encoder.dimension) for item in cluster_payload["clusters"])
    _validate_memberships(words, clusters, config)
    pairs = {
        (_text(item, "noun"), _text(item, "verb")): _integer(item, "token_mass")
        for item in _iter_jsonl(root / "role-pair-masses.jsonl")
    }
    nonconstruction = _integer(record, "nonconstruction_token_count")
    retained = _integer(record, "retained_token_count")
    if sum(pairs.values()) != nonconstruction:
        raise SemanticCatalogError("catalog role-pair masses changed")
    retained_words = {
        role: {item.word for item in words if item.role == role and item.cluster is not None}
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    measured_retained = sum(
        mass
        for (noun, verb), mass in pairs.items()
        if noun in retained_words["noun"] and verb in retained_words["verb"]
    )
    if measured_retained != retained:
        raise SemanticCatalogError("catalog retained token mass changed")
    _validate_persisted_quality(
        words,
        clusters,
        pairs,
        config,
        retained,
        nonconstruction,
    )
    parent = record.get("parent_catalog_sha256")
    if parent is not None and (type(parent) is not str or len(parent) != 64):
        raise SemanticCatalogError("catalog parent identity is malformed")
    return SemanticCatalog(
        root=root.resolve(),
        catalog_sha256=catalog_sha,
        evidence_sha256=_text(record, "evidence_sha256"),
        encoder_identity=encoder,
        config=config,
        words=words,
        clusters=clusters,
        retained_token_count=retained,
        nonconstruction_token_count=nonconstruction,
        parent_catalog_sha256=parent,
    )


def load_catalog_pair_masses(catalog: SemanticCatalog) -> dict[tuple[str, str], int]:
    """Load the exact noun-by-verb masses authenticated by a catalog."""
    return {
        (_text(record, "noun"), _text(record, "verb")): _integer(record, "token_mass")
        for record in _iter_jsonl(catalog.root / "role-pair-masses.jsonl")
    }


def _cluster_screened_words(
    screened: Sequence[_ScreenedWord],
    candidates: Mapping[Role, Sequence[WordVector]],
    pair_masses: Mapping[tuple[str, str], int],
    nonconstruction_token_count: int,
    config: SemanticConstructionConfig,
) -> tuple[tuple[SemanticWord, ...], tuple[SemanticCluster, ...], int]:
    boundary_results = {
        role: cluster_with_boundary_exclusions(candidates[role], config)
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    for result in boundary_results.values():
        validate_cluster_quality(result.clustering, config)
    assignments = {
        (role, word): cluster
        for role, result in boundary_results.items()
        for word, cluster in result.clustering.assignments
    }
    boundary_margins = {
        (role, word): margin
        for role, result in boundary_results.items()
        for word, margin in result.excluded_margins
    }
    retained_margins = {
        (role, word): margin
        for role, result in boundary_results.items()
        for word, margin in result.clustering.margin_by_word(
            {
                item.word: item.vector
                for item in candidates[role]
                if (role, item.word) in assignments
            }
        ).items()
    }
    words = tuple(
        _final_word(item, assignments, retained_margins, boundary_margins)
        for item in screened
    )
    retained_nouns = {
        item.word
        for item in words
        if item.role == "noun" and item.cluster is not None
    }
    retained_verbs = {
        item.word
        for item in words
        if item.role == "verb" and item.cluster is not None
    }
    retained_tokens = sum(
        mass
        for (noun, verb), mass in pair_masses.items()
        if noun in retained_nouns and verb in retained_verbs
    )
    if (
        retained_tokens / nonconstruction_token_count
        < config.minimum_retained_token_fraction
    ):
        raise SemanticGridError(
            "both-role semantic exclusions retain less than 40% of archive token mass"
        )
    clusters = tuple(
        SemanticCluster(
            role=role,
            index=index,
            token_mass=result.clustering.cluster_masses[index],
            centroid=result.clustering.centroids[index],
            words=tuple(
                sorted(
                    word
                    for word, cluster in result.clustering.assignments
                    if cluster == index
                )
            ),
        )
        for role, result in boundary_results.items()
        for index in range(config.cluster_count)
    )
    return words, clusters, retained_tokens


def _publish_catalog_failure(
    screened: Sequence[_ScreenedWord],
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
    config: SemanticConstructionConfig,
) -> SemanticCatalogFailureArtifact:
    failure_words = tuple(
        CatalogFailureWord(
            role=item.evidence.role,
            word=item.evidence.word,
            token_mass=item.evidence.token_mass,
            context_count=item.evidence.context_embeddings.shape[0],
            role_margin_q10=item.role_margin,
            context_silhouette=item.silhouette,
            exclusion_reason=cast(str, item.reason or "semantic_grid_failure"),
            vector=item.vector,
        )
        for item in screened
    )
    working = Path(temporary_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(
            f"semantic failure-audit temporary directory is not empty: {working}"
        )
    working.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        working / "words.jsonl",
        (
            {
                "context_count": item.context_count,
                "context_silhouette": item.context_silhouette,
                "disposition": item.exclusion_reason,
                "role": item.role,
                "role_margin_q10": item.role_margin_q10,
                "token_mass": item.token_mass,
                "vector": None if item.vector is None else list(item.vector),
                "word": item.word,
            }
            for item in failure_words
        ),
    )
    _write_jsonl(
        working / "representative-contexts.jsonl",
        (
            record
            for key in sorted(representative_contexts)
            for record in representative_contexts[key]
        ),
    )
    _write_jsonl(
        working / "role-pair-masses.jsonl",
        (
            {"noun": noun, "token_mass": mass, "verb": verb}
            for (noun, verb), mass in sorted(pair_masses.items())
        ),
    )
    content = {
        "config": config.as_record(),
        "encoder": encoder_identity.as_record(),
        "evidence_sha256": evidence_sha256,
        "format": "tinyworlds-p-semantic-catalog-failure",
        "nonconstruction_token_count": nonconstruction_token_count,
        "reason": reason,
        "representative_contexts_sha256": _file_sha256(
            working / "representative-contexts.jsonl"
        ),
        "role_pair_masses_sha256": _file_sha256(
            working / "role-pair-masses.jsonl"
        ),
        "schema_version": SCHEMA_VERSION,
        "words_sha256": _file_sha256(working / "words.jsonl"),
    }
    failure_sha = record_sha256(content)
    _write_json(
        working / "failure.json",
        {**content, "failure_sha256": failure_sha},
    )
    markdown, html = render_catalog_failure_audits(
        failure_sha,
        evidence_sha256,
        reason,
        config,
        failure_words,
        representative_contexts,
    )
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", html)
    _write_failure_tree(working, failure_sha)
    failure_root = Path(output_root) / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    target = failure_root / failure_sha
    if target.exists():
        return load_semantic_catalog_failure(target)
    os.rename(working, target)
    _fsync_directory(failure_root)
    return load_semantic_catalog_failure(target)


def load_semantic_catalog_failure(
    path: str | Path,
) -> SemanticCatalogFailureArtifact:
    """Strictly authenticate one content-addressed semantic-grid failure audit."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SemanticCatalogError("semantic catalog failure must be a regular directory")
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"failure_sha256", "files", "format", "schema_version"}
        or tree.get("format") != "tinyworlds-p-semantic-catalog-failure-tree"
        or tree.get("schema_version") != 1
    ):
        raise SemanticCatalogError("semantic catalog failure tree changed")
    failure_sha = _text(tree, "failure_sha256")
    if root.name != failure_sha:
        raise SemanticCatalogError("semantic catalog failure directory identity changed")
    raw_files = tree.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise SemanticCatalogError("semantic catalog failure files changed")
    expected_paths = (
        "audit.html",
        "audit.md",
        "failure.json",
        "representative-contexts.jsonl",
        "role-pair-masses.jsonl",
        "words.jsonl",
    )
    paths = tuple(_text(item, "relative_path") for item in raw_files)
    actual = tuple(
        sorted(
            candidate.relative_to(root).as_posix()
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.name != "tree.json"
        )
    )
    if paths != expected_paths or actual != expected_paths:
        raise SemanticCatalogError("semantic catalog failure file membership changed")
    if any(candidate.is_symlink() for candidate in root.rglob("*")):
        raise SemanticCatalogError("semantic catalog failure cannot contain symlinks")
    for descriptor in raw_files:
        candidate = root / _text(descriptor, "relative_path")
        if (
            candidate.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(candidate) != _text(descriptor, "sha256")
        ):
            raise SemanticCatalogError("semantic catalog failure file changed")
    record = _load_json(root / "failure.json")
    required = {
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
    if set(record) != required or record.get("format") != "tinyworlds-p-semantic-catalog-failure":
        raise SemanticCatalogError("semantic catalog failure identity fields changed")
    content = {key: value for key, value in record.items() if key != "failure_sha256"}
    if record.get("failure_sha256") != failure_sha or record_sha256(content) != failure_sha:
        raise SemanticCatalogError("semantic catalog failure identity changed")
    for filename, field in (
        ("words.jsonl", "words_sha256"),
        ("representative-contexts.jsonl", "representative_contexts_sha256"),
        ("role-pair-masses.jsonl", "role_pair_masses_sha256"),
    ):
        if _file_sha256(root / filename) != record[field]:
            raise SemanticCatalogError("semantic catalog failure payload changed")
    try:
        semantic_config_from_record(_mapping(record, "config"))
        _encoder_identity(_mapping(record, "encoder"))
    except (TypeError, ValueError) as error:
        raise SemanticCatalogError("semantic catalog failure contract is invalid") from error
    tuple(_iter_jsonl(root / "words.jsonl"))
    tuple(_iter_jsonl(root / "representative-contexts.jsonl"))
    pairs = tuple(_iter_jsonl(root / "role-pair-masses.jsonl"))
    if sum(_integer(item, "token_mass") for item in pairs) != _integer(
        record, "nonconstruction_token_count"
    ):
        raise SemanticCatalogError("semantic catalog failure role-pair mass changed")
    return SemanticCatalogFailureArtifact(
        root=root.resolve(),
        failure_sha256=failure_sha,
        evidence_sha256=_text(record, "evidence_sha256"),
        reason=_text(record, "reason"),
    )


def _screen_word(evidence: WordEvidence, config: SemanticConstructionConfig) -> _ScreenedWord:
    context_count = evidence.context_embeddings.shape[0]
    if context_count < config.minimum_contexts_per_word:
        return _ScreenedWord(evidence, None, None, None, "insufficient_contexts")
    margin = role_margin_quantile(
        evidence.target_anchor_embeddings,
        evidence.opposite_anchor_embeddings,
        evidence.context_embeddings,
        config.role_margin_quantile,
    )
    if margin <= config.minimum_role_margin:
        return _ScreenedWord(evidence, margin, None, None, "nonpositive_role_margin")
    silhouette = deterministic_two_means_silhouette(evidence.context_embeddings)
    if silhouette > config.maximum_context_silhouette:
        return _ScreenedWord(evidence, margin, silhouette, None, "multiple_realized_senses")
    vector = compose_word_vector(
        evidence.target_anchor_embeddings,
        evidence.context_embeddings,
    )
    return _ScreenedWord(
        evidence,
        margin,
        silhouette,
        tuple(float(value) for value in vector),
        None,
    )


def _final_word(
    screened: _ScreenedWord,
    assignments: Mapping[tuple[Role, str], int],
    retained_margins: Mapping[tuple[Role, str], float],
    boundary_margins: Mapping[tuple[Role, str], float],
) -> SemanticWord:
    key = (screened.evidence.role, screened.evidence.word)
    boundary = key in boundary_margins
    reason = cast(ExclusionReason | None, "cluster_boundary_margin" if boundary else screened.reason)
    return SemanticWord(
        role=screened.evidence.role,
        word=screened.evidence.word,
        token_mass=screened.evidence.token_mass,
        context_count=screened.evidence.context_embeddings.shape[0],
        role_margin_q10=screened.role_margin,
        context_silhouette=screened.silhouette,
        cluster_margin=boundary_margins.get(key, retained_margins.get(key)),
        cluster=None if reason is not None else assignments[key],
        exclusion_reason=reason,
        vector=None if reason is not None else screened.vector,
    )


def _validate_memberships(
    words: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
    config: SemanticConstructionConfig,
) -> None:
    expected = tuple(
        (role, index)
        for role in ("noun", "verb")
        for index in range(config.cluster_count)
    )
    if tuple((item.role, item.index) for item in clusters) != expected:
        raise SemanticCatalogError("semantic clusters are incomplete or unordered")
    by_cluster: dict[tuple[Role, int], set[str]] = defaultdict(set)
    mass: Counter[tuple[Role, int]] = Counter()
    for word in words:
        if word.cluster is not None:
            by_cluster[(word.role, word.cluster)].add(word.word)
            mass[(word.role, word.cluster)] += word.token_mass
    for cluster in clusters:
        key = (cluster.role, cluster.index)
        if set(cluster.words) != by_cluster[key] or cluster.token_mass != mass[key]:
            raise SemanticCatalogError("semantic cluster membership or mass changed")


def _validate_persisted_quality(
    words: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
    pair_masses: Mapping[tuple[str, str], int],
    config: SemanticConstructionConfig,
    retained_token_count: int,
    nonconstruction_token_count: int,
) -> None:
    expected_masses = {"noun": Counter(), "verb": Counter()}
    for (noun, verb), mass in pair_masses.items():
        expected_masses["noun"][noun] += mass
        expected_masses["verb"][verb] += mass
    measured_masses = {
        role: Counter(
            {item.word: item.token_mass for item in words if item.role == role}
        )
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    if measured_masses != expected_masses:
        raise SemanticCatalogError("semantic word masses differ from role-pair masses")
    for word in words:
        margin = word.role_margin_q10
        silhouette = word.context_silhouette
        cluster_margin = word.cluster_margin
        if word.exclusion_reason == "insufficient_contexts":
            valid = (
                word.context_count < config.minimum_contexts_per_word
                and margin is None
                and silhouette is None
                and cluster_margin is None
            )
        elif word.exclusion_reason == "nonpositive_role_margin":
            valid = (
                word.context_count >= config.minimum_contexts_per_word
                and margin is not None
                and margin <= config.minimum_role_margin
                and silhouette is None
                and cluster_margin is None
            )
        elif word.exclusion_reason == "multiple_realized_senses":
            valid = (
                margin is not None
                and margin > config.minimum_role_margin
                and silhouette is not None
                and silhouette > config.maximum_context_silhouette
                and cluster_margin is None
            )
        elif word.exclusion_reason == "cluster_boundary_margin":
            valid = (
                margin is not None
                and margin > config.minimum_role_margin
                and silhouette is not None
                and silhouette <= config.maximum_context_silhouette
                and cluster_margin is not None
                and cluster_margin < config.minimum_cluster_margin
            )
        else:
            valid = (
                word.exclusion_reason is None
                and margin is not None
                and margin > config.minimum_role_margin
                and silhouette is not None
                and silhouette <= config.maximum_context_silhouette
                and cluster_margin is not None
                and cluster_margin >= config.minimum_cluster_margin
            )
        if not valid or (
            word.exclusion_reason is not None
            and (word.cluster is not None or word.vector is not None)
        ):
            raise SemanticCatalogError(
                f"semantic word disposition violates frozen gates: {word.role}/{word.word}"
            )
    by_key = {(item.role, item.word): item for item in words}
    by_role = {
        role: tuple(item for item in clusters if item.role == role)
        for role in cast(tuple[Role, Role], ("noun", "verb"))
    }
    for role, role_clusters in by_role.items():
        total_mass = sum(item.token_mass for item in role_clusters)
        lower = config.minimum_cluster_mass_fraction * total_mass / config.cluster_count
        upper = config.maximum_cluster_mass_fraction * total_mass / config.cluster_count
        minimum_words = (
            config.minimum_nouns_per_cluster
            if role == "noun"
            else config.minimum_verbs_per_cluster
        )
        if any(
            len(cluster.words) < minimum_words
            or not lower - 1e-9 <= cluster.token_mass <= upper + 1e-9
            for cluster in role_clusters
        ):
            raise SemanticCatalogError("semantic cluster size or mass gate changed")
        centroid_matrix = np.asarray(
            [cluster.centroid for cluster in role_clusters],
            dtype=np.float64,
        )
        if max(
            float(centroid_matrix[left] @ centroid_matrix[right])
            for left in range(config.cluster_count)
            for right in range(left + 1, config.cluster_count)
        ) >= config.maximum_centroid_pair_cosine:
            raise SemanticCatalogError("semantic centroid-pair gate changed")
        for cluster in role_clusters:
            vectors = np.asarray(
                [
                    by_key[(role, word)].vector
                    for word in cluster.words
                ],
                dtype=np.float32,
            )
            centroid = np.mean(vectors, axis=0, dtype=np.float32)
            centroid /= np.linalg.norm(centroid)
            if not np.allclose(
                centroid,
                np.asarray(cluster.centroid, dtype=np.float32),
                atol=1e-6,
            ):
                raise SemanticCatalogError("semantic cluster centroid changed")
            for word in cluster.words:
                persisted = by_key[(role, word)]
                if persisted.vector is None or persisted.cluster_margin is None:
                    raise SemanticCatalogError("semantic retained word evidence changed")
                similarities = centroid_matrix @ np.asarray(
                    persisted.vector,
                    dtype=np.float64,
                )
                margin = float(similarities[cluster.index]) - max(
                    float(value)
                    for index, value in enumerate(similarities)
                    if index != cluster.index
                )
                if not np.isclose(margin, persisted.cluster_margin, atol=1e-6):
                    raise SemanticCatalogError("semantic cluster margin changed")
    if (
        retained_token_count / nonconstruction_token_count
        < config.minimum_retained_token_fraction
    ):
        raise SemanticCatalogError("semantic retained-mass gate changed")


def _semantic_word(record: Mapping[str, object], dimension: int) -> SemanticWord:
    vector_value = record.get("vector")
    if vector_value is not None and (
        type(vector_value) is not list
        or len(vector_value) != dimension
        or any(type(value) not in (int, float) for value in vector_value)
    ):
        raise SemanticCatalogError("semantic word vector is malformed")
    reason = record.get("exclusion_reason")
    allowed_reasons = {
        None,
        "insufficient_contexts",
        "nonpositive_role_margin",
        "multiple_realized_senses",
        "cluster_boundary_margin",
    }
    if reason not in allowed_reasons:
        raise SemanticCatalogError("semantic word exclusion reason is invalid")
    cluster = record.get("cluster")
    if cluster is not None and (type(cluster) is not int or cluster < 0):
        raise SemanticCatalogError("semantic word cluster is invalid")
    return SemanticWord(
        role=_role(record, "role"),
        word=_text(record, "word"),
        token_mass=_integer(record, "token_mass"),
        context_count=_integer(record, "context_count"),
        role_margin_q10=_optional_number(record, "role_margin_q10"),
        context_silhouette=_optional_number(record, "context_silhouette"),
        cluster_margin=_optional_number(record, "cluster_margin"),
        cluster=cluster,
        exclusion_reason=cast(ExclusionReason | None, reason),
        vector=None if vector_value is None else tuple(float(value) for value in vector_value),
    )


def _semantic_cluster(record: object, dimension: int) -> SemanticCluster:
    if type(record) is not dict:
        raise SemanticCatalogError("semantic cluster must be an object")
    centroid = record.get("centroid")
    words = record.get("words")
    if (
        type(centroid) is not list
        or len(centroid) != dimension
        or any(type(value) not in (int, float) for value in centroid)
        or type(words) is not list
        or any(type(value) is not str for value in words)
    ):
        raise SemanticCatalogError("semantic cluster vector or words are malformed")
    return SemanticCluster(
        role=_role(record, "role"),
        index=_integer(record, "index"),
        token_mass=_integer(record, "token_mass"),
        centroid=tuple(float(value) for value in centroid),
        words=tuple(words),
    )


def _encoder_identity(record: Mapping[str, object]) -> EncoderIdentity:
    files = record.get("files")
    if type(files) is not list or any(type(item) is not dict for item in files):
        raise SemanticCatalogError("catalog encoder files are malformed")
    return EncoderIdentity(
        identifier=_text(record, "identifier"),
        revision=_text(record, "revision"),
        dimension=_integer(record, "dimension"),
        files=tuple(
            ModelFile(
                relative_path=_text(item, "relative_path"),
                size_bytes=_integer(item, "size_bytes"),
                sha256=_text(item, "sha256"),
            )
            for item in files
        ),
        pooling=cast(str, record.get("pooling")),
        normalization=cast(str, record.get("normalization")),
        dtype=cast(str, record.get("dtype")),
    )


def _write_tree(root: Path, catalog_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "catalog_sha256": catalog_sha256,
            "files": list(files),
            "format": "tinyworlds-p-semantic-catalog-tree",
            "schema_version": 1,
        },
    )


def _write_failure_tree(root: Path, failure_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "failure_sha256": failure_sha256,
            "files": list(files),
            "format": "tinyworlds-p-semantic-catalog-failure-tree",
            "schema_version": 1,
        },
    )


def _validate_tree(root: Path, tree: Mapping[str, object]) -> None:
    raw = tree.get("files")
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise SemanticCatalogError("semantic catalog tree files are malformed")
    paths = tuple(_text(item, "relative_path") for item in raw)
    if paths != tuple(sorted(set(paths))):
        raise SemanticCatalogError("semantic catalog tree paths are not canonical")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(paths) | {"tree.json"} or any(path.is_symlink() for path in root.rglob("*")):
        raise SemanticCatalogError("semantic catalog tree membership changed")
    for descriptor in raw:
        relative = _text(descriptor, "relative_path")
        path = root / relative
        if path.stat().st_size != _integer(descriptor, "size_bytes") or _file_sha256(path) != _text(
            descriptor, "sha256"
        ):
            raise SemanticCatalogError(f"semantic catalog file changed: {relative}")


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    with path.open("wb") as output:
        for value in values:
            output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemanticCatalogError(f"invalid semantic catalog JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise SemanticCatalogError(f"noncanonical semantic catalog JSON: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise SemanticCatalogError(f"invalid JSONL at {path}:{line_number}") from error
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise SemanticCatalogError(f"noncanonical JSONL at {path}:{line_number}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise SemanticCatalogError(f"field {field!r} must be an object")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise SemanticCatalogError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise SemanticCatalogError(f"field {field!r} must be a nonnegative integer")
    return value


def _role(record: Mapping[str, object], field: str) -> Role:
    value = _text(record, field)
    if value not in ("noun", "verb"):
        raise SemanticCatalogError(f"field {field!r} must be noun or verb")
    return cast(Role, value)


def _optional_number(record: Mapping[str, object], field: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise SemanticCatalogError(f"field {field!r} must be numeric or null")
    return float(value)


__all__ = [
    "SemanticCatalogFailureArtifact",
    "SemanticCatalogError",
    "WordEvidence",
    "build_catalog_from_evidence",
    "build_semantic_catalog",
    "load_catalog_pair_masses",
    "load_word_evidence",
    "load_semantic_catalog_failure",
    "load_semantic_catalog",
]
