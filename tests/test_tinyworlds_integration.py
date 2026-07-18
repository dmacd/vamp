from __future__ import annotations

from pathlib import Path

import pytest

from apm.data.text.tinyworlds.novelty import (
    ORIGINAL_TINYSTORIES_TRAIN,
    audit_nonce_terms,
    novelty_terms_for_bundles,
)
from apm.data.text.tinyworlds.query_generation import (
    generate_calibration_bundle,
)
from apm.data.text.tinyworlds.persistence import load_tinyworlds_bundle
from apm.data.text.tinyworlds.rendering import (
    TinyWorldsRenderPreset,
    render_tinyworlds_bundle,
)
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_real_tinystories_tokenizer_accepts_exact_tinyworlds_boundaries() -> None:
    tokenizer_path = (
        REPOSITORY_ROOT
        / "checkpoints"
        / "tinystories-8m"
        / "tokenizer"
        / "tokenizer.json"
    )
    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_path)
    rendered = render_tinyworlds_bundle(
        generate_calibration_bundle("a" * 64),
        tokenizer,
        TinyWorldsRenderPreset(1, 1, 1, 4, 4, 1, 256, 256),
    )

    assert all(
        tuple(len(variant.prefix_token_ids) for variant in group.variants)
        == (64, 128, 192)
        for group in rendered.query_groups
    )
    assert all(
        max(
            candidate.competence_batch.input_ids.shape[1] + 1
            for candidate in variant.knowledge_query.candidates
        )
        <= 256
        for group in rendered.query_groups
        for variant in group.variants
    )
    assert {
        group.variants[0].knowledge_query.query_kind
        for group in rendered.query_groups
    } == {
        "ancestor_plus_child",
        "cross_branch",
        "direct",
        "new_instance",
        "one_hop",
        "open_book",
        "revision_sensitive",
        "two_hop",
    }


@pytest.mark.integration
def test_original_pretraining_corpus_has_zero_generated_nonce_hits() -> None:
    corpus_path = (
        REPOSITORY_ROOT
        / "data"
        / "tinystories-original"
        / ORIGINAL_TINYSTORIES_TRAIN.filename
    )
    bundle_root = REPOSITORY_ROOT / "data" / "tinyworlds" / "v1"
    bundles = (
        load_tinyworlds_bundle(bundle_root / "calibration"),
        load_tinyworlds_bundle(bundle_root / "pilot"),
    )

    report = audit_nonce_terms(corpus_path, novelty_terms_for_bundles(bundles))

    assert report.passed
    assert report.hits == ()
