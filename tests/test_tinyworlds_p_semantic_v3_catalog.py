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
    V2_SEMANTIC_CONFIG,
    V3_SEMANTIC_CONFIG,
    V3SemanticCatalogError,
    V3SemanticGridError,
    WordEvidence,
    WordVector,
    build_v3_semantic_catalog,
    load_v3_catalog_failure,
    load_v3_semantic_catalog,
    role_calibration_fold,
    semantic_first_spherical_kmeans,
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
        files=(ModelFile("model.safetensors", 1, sha256(b"v3").hexdigest()),),
    )


def _config():
    return replace(
        V3_SEMANTIC_CONFIG,
        role_calibration_fold_count=3,
        role_calibration_alpha=0.05,
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


def test_v3_clustering_is_semantic_only_and_reuses_v2_role_folds() -> None:
    words = tuple(
        WordVector(
            role="noun",
            word=f"noun-{cluster}-{index}",
            token_mass=1,
            vector=tuple(float(value) for value in _unit(cluster)),
        )
        for cluster in range(3)
        for index in range(3)
    )
    reweighted = tuple(
        replace(item, token_mass=(index + 1) * 10_000)
        for index, item in enumerate(words)
    )
    first = semantic_first_spherical_kmeans(
        words,
        3,
        benchmark_id="tinyworlds-p-semantic-v3",
    )
    second = semantic_first_spherical_kmeans(
        tuple(reversed(reweighted)),
        3,
        benchmark_id="tinyworlds-p-semantic-v3",
    )

    assert first.assignments == second.assignments
    assert first.cluster_masses != second.cluster_masses
    assert all(
        role_calibration_fold(role, word, V3_SEMANTIC_CONFIG)
        == role_calibration_fold(role, word, V2_SEMANTIC_CONFIG)
        for role in ("noun", "verb")
        for word in (f"{role}-alpha", f"{role}-omega")
    )
    config_record = V3_SEMANTIC_CONFIG.as_record()
    assert config_record["balance_stage"] == "partition-story-allocation"
    assert config_record["construction_numpy_version"] == "1.26.4"
    assert (
        config_record["role_calibration_source_failure_sha256"]
        == "23cedf831ef1ad6331d05b58290705a51fd6da1d0fff65a164d1ec544491be25"
    )
    assert "minimum_cluster_mass_fraction" not in config_record
    assert "maximum_cluster_mass_fraction" not in config_record


def test_v3_catalog_is_content_addressed_reproducible_and_strict(
    tmp_path: Path,
) -> None:
    nouns = tuple(
        f"noun-{cluster}-{index}" for cluster in range(3) for index in range(2)
    )
    verbs = tuple(
        f"verb-{cluster}-{index}" for cluster in range(3) for index in range(2)
    )
    pairs = {
        (noun, verb): 1 + (noun.endswith("-0") and verb.endswith("-0"))
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
    progress_events = []

    first = build_v3_semantic_catalog(
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
    replay = build_v3_semantic_catalog(
        tuple(reversed(evidence)),
        dict(reversed(tuple(pairs.items()))),
        _encoder_identity(),
        "e" * 64,
        sum(pairs.values()),
        tmp_path / "catalog-b",
        tmp_path / "work-b",
        _config(),
    )

    assert replay.catalog_sha256 == first.catalog_sha256
    assert {event[0] for event in progress_events} == {
        "role-scores",
        "calibration",
        "screening",
        "clustering",
        "publication",
    }
    assert first.retained_token_fraction == 1.0
    assert {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(replay.root): path.read_bytes()
        for path in replay.root.rglob("*")
        if path.is_file()
    }
    assert "Semantic v3 Catalog Audit" in (first.root / "audit.md").read_text()
    assert "partition-story-allocation" in (first.root / "audit.md").read_text()
    assert all(word.cluster_margin is not None for word in first.words)

    words_path = first.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(V3SemanticCatalogError, match="changed"):
        load_v3_semantic_catalog(first.root)


def test_v3_failed_grid_publishes_strict_exhaustive_audit(tmp_path: Path) -> None:
    config = replace(
        _config(),
        role_calibration_fold_count=2,
        minimum_calibration_reference_words=1,
        cluster_count=4,
        minimum_nouns_per_cluster=1,
        minimum_verbs_per_cluster=1,
    )

    def across_folds(role: str, count: int) -> tuple[str, ...]:
        selected = []
        observed = set()
        for index in range(100):
            word = f"{role}-{index}"
            selected.append(word)
            observed.add(role_calibration_fold(role, word, config))
            if len(selected) >= count and observed == {0, 1}:
                return tuple(selected)
        raise AssertionError("fixture could not cover calibration folds")

    nouns = across_folds("noun", 3)
    verbs = across_folds("verb", 8)
    pairs = {(noun, verb): 1 for noun in nouns for verb in verbs}
    evidence = tuple(
        _evidence("noun", noun, index % 3, len(verbs))
        for index, noun in enumerate(nouns)
    ) + tuple(
        _evidence("verb", verb, 4 + index % 4, len(nouns))
        for index, verb in enumerate(verbs)
    )

    with pytest.raises(V3SemanticGridError, match="failure audit"):
        build_v3_semantic_catalog(
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
    failure = load_v3_catalog_failure(failures[0])
    assert "clustering failed" in failure.reason
    audit = (failure.root / "audit.md").read_text()
    assert "Semantic v3 Construction Failure Audit" in audit
    assert "All role words and exclusion reasons" in audit
    assert "No semantic-v3 catalog" in audit

    words_path = failure.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(V3SemanticCatalogError, match="changed"):
        load_v3_catalog_failure(failure.root)
