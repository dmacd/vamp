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
    V2SemanticCatalogError,
    V2SemanticGridError,
    WordEvidence,
    build_v2_semantic_catalog,
    calibrate_role_margins,
    load_v2_catalog_failure,
    load_v2_semantic_catalog,
    role_calibration_fold,
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
        files=(ModelFile("model.safetensors", 1, sha256(b"v2").hexdigest()),),
    )


def _config():
    return replace(
        V2_SEMANTIC_CONFIG,
        role_calibration_fold_count=3,
        role_calibration_alpha=0.05,
        minimum_calibration_reference_words=2,
        minimum_contexts_per_word=4,
        maximum_context_silhouette=1.0,
        cluster_count=3,
        minimum_cluster_mass_fraction=0.80,
        maximum_cluster_mass_fraction=1.20,
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


def test_cross_conformal_role_calibration_is_cross_fitted_and_order_independent() -> None:
    config = replace(
        V2_SEMANTIC_CONFIG,
        role_calibration_fold_count=3,
        role_calibration_alpha=0.20,
        minimum_calibration_reference_words=4,
    )
    raw = {
        (role, f"{role}-{index}"): (-10.0 if index == 0 else float(index))
        for role in ("noun", "verb")
        for index in range(12)
    }
    first, references = calibrate_role_margins(raw, config)
    replay, replay_references = calibrate_role_margins(
        dict(reversed(tuple(raw.items()))),
        config,
    )

    assert replay == first
    assert replay_references == references
    assert first[("noun", "noun-0")].conformal_p <= config.role_calibration_alpha
    assert first[("verb", "verb-0")].conformal_p <= config.role_calibration_alpha
    assert first[("noun", "noun-11")].conformal_p > config.role_calibration_alpha

    fold = role_calibration_fold("noun", "noun-5", config)
    same_fold = next(
        word
        for role, word in raw
        if role == "noun"
        and word != "noun-5"
        and role_calibration_fold(role, word, config) == fold
    )
    changed = dict(raw)
    changed[("noun", same_fold)] = -1_000.0
    rescored, _ = calibrate_role_margins(changed, config)
    assert rescored[("noun", "noun-5")] == first[("noun", "noun-5")]


def test_v2_catalog_is_content_addressed_reproducible_and_strict(tmp_path: Path) -> None:
    nouns = tuple(f"noun-{cluster}-{index}" for cluster in range(3) for index in range(2))
    verbs = tuple(f"verb-{cluster}-{index}" for cluster in range(3) for index in range(2))
    pairs = {(noun, verb): 1 for noun in nouns for verb in verbs}
    evidence = tuple(
        _evidence("noun", noun, int(noun.split("-")[1]), len(verbs))
        for noun in nouns
    ) + tuple(
        _evidence("verb", verb, 4 + int(verb.split("-")[1]), len(nouns))
        for verb in verbs
    )
    progress_events = []

    first = build_v2_semantic_catalog(
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
    replay = build_v2_semantic_catalog(
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
    assert progress_events[-1][:3] == ("publication", 1, 1)
    assert first.retained_token_fraction == 1.0
    assert len(first.calibration) == 6
    assert {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(replay.root): path.read_bytes()
        for path in replay.root.rglob("*")
        if path.is_file()
    }
    assert "Cross-conformal role calibration" in (first.root / "audit.md").read_text()
    assert "<svg" in (first.root / "audit.html").read_text()

    words_path = first.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(V2SemanticCatalogError, match="changed"):
        load_v2_semantic_catalog(first.root)


def test_v2_failed_grid_publishes_strict_exhaustive_audit(tmp_path: Path) -> None:
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

    with pytest.raises(V2SemanticGridError, match="failure audit"):
        build_v2_semantic_catalog(
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
    failure = load_v2_catalog_failure(failures[0])
    assert "clustering failed" in failure.reason
    audit = (failure.root / "audit.md").read_text()
    assert "All role words and exclusion reasons" in audit
    assert "Cross-conformal role calibration" in audit
    assert "No semantic-v2 catalog" in audit

    words_path = failure.root / "words.jsonl"
    payload = bytearray(words_path.read_bytes())
    payload[-2] ^= 1
    words_path.write_bytes(payload)
    with pytest.raises(V2SemanticCatalogError, match="changed"):
        load_v2_catalog_failure(failure.root)
