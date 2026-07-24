from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p_semantic import (
    ENCODER_DIMENSION,
    ENCODER_IDENTIFIER,
    ENCODER_REVISION,
    EncoderIdentity,
    ModelFile,
    SEMANTIC_CONFIG,
    SemanticCatalogError,
    SemanticGridError,
    SphericalClustering,
    WordEvidence,
    WordVector,
    build_semantic_catalog,
    capacity_constrained_spherical_kmeans,
    deterministic_two_means_silhouette,
    exact_whole_word_spans,
    is_construction_group,
    load_semantic_catalog,
    load_semantic_catalog_failure,
    role_margin_quantile,
    story_contexts,
)


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _encoder_identity() -> EncoderIdentity:
    return EncoderIdentity(
        identifier=ENCODER_IDENTIFIER,
        revision=ENCODER_REVISION,
        dimension=ENCODER_DIMENSION,
        files=(ModelFile("model.safetensors", 1, sha256(b"x").hexdigest()),),
    )


def _config():
    return replace(
        SEMANTIC_CONFIG,
        cluster_count=3,
        minimum_contexts_per_word=4,
        maximum_context_silhouette=1.0,
        minimum_cluster_mass_fraction=0.80,
        maximum_cluster_mass_fraction=1.20,
        minimum_nouns_per_cluster=2,
        minimum_verbs_per_cluster=2,
        maximum_centroid_pair_cosine=0.95,
    )


def _word_evidence(role: str, word: str, axis: int, mass: int, contexts: int = 4):
    vector = _unit(axis)
    opposite = -vector
    return WordEvidence(
        role=role,
        word=word,
        token_mass=mass,
        target_anchor_embeddings=np.stack((vector, vector, vector)),
        opposite_anchor_embeddings=np.stack((opposite, opposite, opposite)),
        context_embeddings=np.stack((vector,) * contexts),
        contexts=(),
    )


def test_construction_slice_and_exact_contexts_are_namespaced_and_deterministic() -> None:
    hashes = tuple(sha256(str(index).encode()).hexdigest() for index in range(2_000))
    selected = tuple(value for value in hashes if is_construction_group(value, SEMANTIC_CONFIG))

    assert selected == tuple(
        value for value in reversed(tuple(reversed(hashes)))
        if is_construction_group(value, SEMANTIC_CONFIG)
    )
    assert 70 <= len(selected) <= 130
    assert exact_whole_word_spans("A cat met a CAT. A bobcat left.", "cat") == (
        (2, 5),
        (12, 15),
    )
    contexts = story_contexts(
        "noun",
        "cat",
        hashes[0],
        "archive:fixture:0:record",
        hashes[1],
        "A cat slept. A bobcat ran! The CAT woke?",
    )
    assert tuple(item.sentence for item in contexts) == (
        "A cat slept.",
        "The CAT woke?",
    )


def test_role_margin_and_two_means_detect_role_drift_and_split_senses() -> None:
    target, opposite = _unit(0), _unit(1)
    target_anchors = np.stack((target,) * 3)
    opposite_anchors = np.stack((opposite,) * 3)
    contexts = np.stack((target, target, target, opposite))

    assert role_margin_quantile(
        target_anchors,
        opposite_anchors,
        contexts,
        0.10,
    ) < 0.0
    split_contexts = np.stack((_unit(2),) * 8 + (_unit(3),) * 8)
    assert deterministic_two_means_silhouette(split_contexts) > 0.99


def test_capacity_constrained_spherical_kmeans_is_order_independent() -> None:
    words = tuple(
        WordVector(
            role="noun",
            word=f"word-{cluster}-{index}",
            token_mass=10,
            vector=tuple(float(value) for value in _unit(cluster)),
        )
        for cluster in range(3)
        for index in range(4)
    )
    first = capacity_constrained_spherical_kmeans(
        words,
        3,
        minimum_mass_fraction=0.9,
        maximum_mass_fraction=1.1,
    )
    replay = capacity_constrained_spherical_kmeans(
        tuple(reversed(words)),
        3,
        minimum_mass_fraction=0.9,
        maximum_mass_fraction=1.1,
    )

    assert replay == first
    assert first.cluster_masses == (40, 40, 40)
    assert {cluster for _, cluster in first.assignments} == {0, 1, 2}


def test_cluster_margin_uses_capacity_assignment_not_unconstrained_nearest() -> None:
    clustering = SphericalClustering(
        role="noun",
        assignments=(("forced", 1),),
        centroids=(
            tuple(float(value) for value in _unit(0)),
            tuple(float(value) for value in _unit(1)),
        ),
        cluster_masses=(1, 1),
        iterations=1,
    )

    assert clustering.margin_by_word({"forced": _unit(0)})["forced"] == -1.0


def test_catalog_excludes_context_failure_publishes_audits_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    nouns = tuple(f"noun-{cluster}-{index}" for cluster in range(3) for index in range(2))
    verbs = tuple(f"verb-{cluster}-{index}" for cluster in range(3) for index in range(2))
    pair_masses = {(noun, verb): 1 for noun in nouns for verb in verbs}
    pair_masses.update({("bad-noun", verb): 1 for verb in verbs})
    evidence = tuple(
        _word_evidence("noun", noun, int(noun.split("-")[1]), 6)
        for noun in nouns
    ) + (
        _word_evidence("noun", "bad-noun", 8, 6, contexts=2),
    ) + tuple(
        _word_evidence("verb", verb, 4 + int(verb.split("-")[1]), 7)
        for verb in verbs
    )
    first = build_semantic_catalog(
        evidence,
        pair_masses,
        _encoder_identity(),
        "e" * 64,
        sum(pair_masses.values()),
        tmp_path / "catalogs-a",
        tmp_path / "work-a",
        _config(),
    )
    second = build_semantic_catalog(
        tuple(reversed(evidence)),
        dict(reversed(tuple(pair_masses.items()))),
        _encoder_identity(),
        "e" * 64,
        sum(pair_masses.values()),
        tmp_path / "catalogs-b",
        tmp_path / "work-b",
        _config(),
    )

    assert second.catalog_sha256 == first.catalog_sha256
    assert (first.root / "audit.md").is_file()
    assert "All excluded words" in (first.root / "audit.md").read_text()
    assert "<svg" in (first.root / "audit.html").read_text()
    bad = next(item for item in first.words if item.word == "bad-noun")
    assert bad.exclusion_reason == "insufficient_contexts"
    assert first.retained_token_fraction == pytest.approx(36 / 42)
    assert {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second.root): path.read_bytes()
        for path in second.root.rglob("*")
        if path.is_file()
    }

    words_path = first.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(SemanticCatalogError, match="changed"):
        load_semantic_catalog(first.root)


def test_failed_grid_publishes_content_addressed_exhaustive_audit(
    tmp_path: Path,
) -> None:
    nouns = ("noun-0", "noun-1")
    verbs = tuple(f"verb-{index}" for index in range(6))
    pair_masses = {(noun, verb): 1 for noun in nouns for verb in verbs}
    evidence = tuple(
        _word_evidence("noun", noun, index, len(verbs))
        for index, noun in enumerate(nouns)
    ) + tuple(
        _word_evidence("verb", verb, 4 + index % 3, len(nouns))
        for index, verb in enumerate(verbs)
    )
    evidence = (
        replace(
            evidence[0],
            contexts=(
                {
                    "record_id": "archive:fixture:0:record",
                    "sentence": "The noun-0 is here.",
                    "story_sha256": "d" * 64,
                },
            ),
        ),
        *evidence[1:],
    )

    with pytest.raises(SemanticGridError, match="failure audit"):
        build_semantic_catalog(
            evidence,
            pair_masses,
            _encoder_identity(),
            "f" * 64,
            sum(pair_masses.values()),
            tmp_path / "catalogs",
            tmp_path / "failure-work",
            _config(),
        )

    failure_root = next((tmp_path / "catalogs" / "failures").iterdir())
    failure = load_semantic_catalog_failure(failure_root)
    assert failure.evidence_sha256 == "f" * 64
    assert "fewer role words" in failure.reason
    assert "All role words and exclusion reasons" in (
        failure.root / "audit.md"
    ).read_text()
    assert "Candidate-vector geometry" in (failure.root / "audit.html").read_text()
    assert "The noun-0 is here." in (failure.root / "audit.md").read_text()

    failure_words = failure.root / "words.jsonl"
    payload = bytearray(failure_words.read_bytes())
    payload[-2] ^= 1
    failure_words.write_bytes(payload)
    with pytest.raises(SemanticCatalogError, match="changed"):
        load_semantic_catalog_failure(failure.root)
