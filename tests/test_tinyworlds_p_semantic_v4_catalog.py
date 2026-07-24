from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p_semantic import (
    ENCODER_DIMENSION,
    ENCODER_IDENTIFIER,
    ENCODER_REVISION,
    EncoderIdentity,
    ModelFile,
    V3_BENCHMARK_ID,
    V4_SEMANTIC_CONFIG,
    V4_SOURCE_V3_FAILURE_SHA256,
    V4SemanticCatalogError,
    V4SemanticGridError,
    WordEvidence,
    build_v4_semantic_catalog,
    load_v4_catalog_failure,
    load_v4_semantic_catalog,
)


def _unit(index: int) -> np.ndarray:
    result = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
    result[index] = 1.0
    return result


def _encoder_identity() -> EncoderIdentity:
    return EncoderIdentity(
        identifier=ENCODER_IDENTIFIER,
        revision=ENCODER_REVISION,
        dimension=ENCODER_DIMENSION,
        files=(ModelFile("model.safetensors", 1, sha256(b"v4").hexdigest()),),
    )


def _config():
    return replace(
        V4_SEMANTIC_CONFIG,
        role_calibration_fold_count=3,
        minimum_calibration_reference_words=2,
        minimum_contexts_per_word=4,
        maximum_context_silhouette=1.0,
        cluster_count=3,
        minimum_nouns_per_cluster=2,
        minimum_verbs_per_cluster=2,
        maximum_centroid_pair_cosine=0.95,
    )


def _evidence(role: str, word: str, axis: int, mass: int) -> WordEvidence:
    target = _unit(axis)
    opposite = -target
    return WordEvidence(
        role=role,
        word=word,
        token_mass=mass,
        target_anchor_embeddings=np.stack((target, target, target)),
        opposite_anchor_embeddings=np.stack((opposite, opposite, opposite)),
        context_embeddings=np.stack((target, target, target, target)),
        contexts=(),
    )


def _vector_evidence(
    role: str,
    word: str,
    vector: np.ndarray,
    mass: int,
) -> WordEvidence:
    target = np.asarray(vector, dtype=np.float32)
    target /= np.linalg.norm(target)
    return WordEvidence(
        role=role,
        word=word,
        token_mass=mass,
        target_anchor_embeddings=np.stack((target, target, target)),
        opposite_anchor_embeddings=np.stack((-target, -target, -target)),
        context_embeddings=np.stack((target, target, target, target)),
        contexts=(),
    )


def _fixture():
    nouns = tuple(
        f"noun-{cluster}-{index}" for cluster in range(3) for index in range(2)
    )
    verbs = tuple(
        f"verb-{cluster}-{index}" for cluster in range(3) for index in range(2)
    )
    pairs = {
        (noun, verb): 1 + int(noun.endswith("-0") and verb.endswith("-0"))
        for noun in nouns
        for verb in verbs
    }
    noun_masses = {
        noun: sum(mass for (item, _), mass in pairs.items() if item == noun)
        for noun in nouns
    }
    verb_masses = {
        verb: sum(mass for (_, item), mass in pairs.items() if item == verb)
        for verb in verbs
    }
    evidence = tuple(
        _evidence("noun", noun, int(noun.split("-")[1]), noun_masses[noun])
        for noun in nouns
    ) + tuple(
        _evidence("verb", verb, 4 + int(verb.split("-")[1]), verb_masses[verb])
        for verb in verbs
    )
    return evidence, pairs


def test_v4_contract_is_one_v3_namespaced_fit_with_no_centroid_update() -> None:
    record = V4_SEMANTIC_CONFIG.as_record()

    assert record["source_v3_failure_sha256"] == V4_SOURCE_V3_FAILURE_SHA256
    assert record["fit_hash_benchmark_id"] == V3_BENCHMARK_ID
    assert record["boundary_method"] == "single-screen-frozen-fit-centroids-v1"
    assert record["boundary_screen_count"] == 1
    assert record["maximum_exclusion_passes"] == 0
    assert record["centroid_update_after_screen"] is False


def test_v4_catalog_is_content_addressed_reproducible_and_strict(
    tmp_path: Path,
) -> None:
    evidence, pairs = _fixture()
    progress_events = []
    first = build_v4_semantic_catalog(
        evidence,
        pairs,
        _encoder_identity(),
        "e" * 64,
        sum(pairs.values()),
        tmp_path / "catalog-a",
        tmp_path / "work-a",
        _config(),
        progress=lambda phase, completed, total, detail: progress_events.append(
            (phase, completed, total, detail)
        ),
    )
    replay = build_v4_semantic_catalog(
        tuple(reversed(evidence)),
        dict(reversed(tuple(pairs.items()))),
        _encoder_identity(),
        "e" * 64,
        sum(pairs.values()),
        tmp_path / "catalog-b",
        tmp_path / "work-b",
        _config(),
    )

    assert first.catalog_sha256 == replay.catalog_sha256
    assert first.retained_token_fraction == 1.0
    assert len(first.fit_clusters) == len(first.clusters) == 6
    assert all(
        fit.centroid == retained.centroid
        for fit, retained in zip(first.fit_clusters, first.clusters)
    )
    assert all(word.fit_cluster == word.cluster for word in first.words)
    assert len(tuple((first.root / "boundary-trace.json").read_text())) > 0
    assert {event[0] for event in progress_events} == {
        "role-scores",
        "calibration",
        "screening",
        "clustering",
        "publication",
    }
    assert {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(replay.root): path.read_bytes()
        for path in replay.root.rglob("*")
        if path.is_file()
    }
    audit = (first.root / "audit.md").read_text()
    assert "Semantic v4 Catalog Audit" in audit
    assert "Frozen-fit versus retained inventory" in audit

    words_path = first.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(V4SemanticCatalogError, match="changed"):
        load_v4_semantic_catalog(first.root)


def test_v4_boundary_words_keep_fit_evidence_without_centroid_refit(
    tmp_path: Path,
) -> None:
    role_words = {}
    evidence_items = []
    for role, offset in (("noun", 0), ("verb", 4)):
        items = []
        for cluster in range(3):
            vector = _unit(offset + cluster)
            for index in range(2):
                word = f"{role}-{cluster}-{index}"
                items.append(word)
                evidence_items.append(_vector_evidence(role, word, vector, 7))
        boundary = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
        boundary[offset] = np.sqrt(0.5)
        boundary[offset + 1] = np.sqrt(0.5)
        word = f"{role}-boundary"
        items.append(word)
        evidence_items.append(_vector_evidence(role, word, boundary, 7))
        role_words[role] = tuple(items)
    pairs = {
        (noun, verb): 1
        for noun in role_words["noun"]
        for verb in role_words["verb"]
    }
    catalog = build_v4_semantic_catalog(
        tuple(evidence_items),
        pairs,
        _encoder_identity(),
        "b" * 64,
        sum(pairs.values()),
        tmp_path / "catalog",
        tmp_path / "work",
        replace(_config(), minimum_cluster_margin=0.5),
    )

    boundary_words = {
        item.role: item for item in catalog.words if item.word.endswith("-boundary")
    }
    assert set(boundary_words) == {"noun", "verb"}
    for role, word in boundary_words.items():
        assert word.exclusion_reason == "cluster_boundary_margin"
        assert word.vector is not None
        assert word.fit_cluster is not None
        assert word.cluster is None
        assert word.cluster_margin == pytest.approx(0.15574944)
        fit = next(
            item
            for item in catalog.fit_clusters
            if item.role == role and item.index == word.fit_cluster
        )
        retained = next(
            item
            for item in catalog.clusters
            if item.role == role and item.index == word.fit_cluster
        )
        assert word.word in fit.words
        assert word.word not in retained.words
        assert fit.centroid == retained.centroid
        retained_vectors = np.asarray(
            [
                item.vector
                for item in catalog.words
                if item.role == role and item.cluster == word.fit_cluster
            ],
            dtype=np.float32,
        )
        retained_centroid = np.mean(retained_vectors, axis=0, dtype=np.float32)
        retained_centroid /= np.linalg.norm(retained_centroid)
        assert not np.allclose(retained.centroid, retained_centroid, atol=1e-6)
    trace = json.loads((catalog.root / "boundary-trace.json").read_text())["passes"]
    assert [(item["role"], item["pass_index"], item["failing_word_count"]) for item in trace] == [
        ("noun", 0, 1),
        ("verb", 0, 1),
    ]


def test_v4_failed_count_gate_preserves_and_replays_the_single_fit(
    tmp_path: Path,
) -> None:
    evidence, pairs = _fixture()
    config = replace(
        _config(),
        minimum_nouns_per_cluster=3,
        minimum_verbs_per_cluster=3,
    )

    with pytest.raises(V4SemanticGridError, match="failure audit"):
        build_v4_semantic_catalog(
            evidence,
            pairs,
            _encoder_identity(),
            "f" * 64,
            sum(pairs.values()),
            tmp_path / "catalog",
            tmp_path / "work",
            config,
        )
    failures = tuple((tmp_path / "catalog" / "failures").iterdir())
    assert len(failures) == 1
    failure = load_v4_catalog_failure(failures[0])
    assert "fixed clusters retain fewer than 3 words" in failure.reason
    assert len(tuple((failure.root / "fit-clusters.json").read_text())) > 0
    audit = (failure.root / "audit.md").read_text()
    assert "Semantic v4 Construction Failure Audit" in audit
    assert "Frozen pass-zero fit evidence" in audit
    assert "No semantic-v4 catalog" in audit

    fit_path = failure.root / "fit-clusters.json"
    payload = bytearray(fit_path.read_bytes())
    payload[-2] ^= 1
    fit_path.write_bytes(payload)
    with pytest.raises(V4SemanticCatalogError, match="changed"):
        load_v4_catalog_failure(failure.root)


def test_v4_failed_role_calibration_publishes_strict_evidence(
    tmp_path: Path,
) -> None:
    evidence, pairs = _fixture()
    config = replace(_config(), minimum_calibration_reference_words=100)

    with pytest.raises(V4SemanticGridError, match="failure audit"):
        build_v4_semantic_catalog(
            evidence,
            pairs,
            _encoder_identity(),
            "c" * 64,
            sum(pairs.values()),
            tmp_path / "catalog",
            tmp_path / "work",
            config,
        )
    failure_path = next((tmp_path / "catalog" / "failures").iterdir())
    failure = load_v4_catalog_failure(failure_path)
    assert failure.reason.startswith("role calibration failed: ")
    records = [
        json.loads(line)
        for line in (failure.root / "words.jsonl").read_text().splitlines()
    ]
    assert {item["exclusion_reason"] for item in records} == {
        "role_calibration_failure"
    }
    assert json.loads((failure.root / "boundary-trace.json").read_text()) == {
        "passes": []
    }
