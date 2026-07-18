from __future__ import annotations

from hashlib import sha256

from apm.data.text.curricula import PinnedDatasetFile
from apm.data.text.tinyworlds.novelty import (
    NoveltyTerm,
    ORIGINAL_TINYSTORIES_REVISION,
    audit_nonce_terms,
    novelty_terms_for_bundles,
)
from apm.data.text.tinyworlds.query_generation import (
    generate_calibration_bundle,
    generate_pilot_bundle,
)


def test_streamed_novelty_audit_is_casefolded_and_word_exact(tmp_path) -> None:
    contents = b"A known Glimbit appears. Glimbitish does not count.\n"
    path = tmp_path / "fixture.txt"
    path.write_bytes(contents)
    contract = PinnedDatasetFile(
        filename=path.name,
        size_bytes=len(contents),
        sha256=sha256(contents).hexdigest(),
    )
    known = NoveltyTerm("glimbit", "name", "entity:known")
    novel = NoveltyTerm("zorvane", "role", "entity:novel")

    report = audit_nonce_terms(
        path,
        (known, novel),
        corpus_file=contract,
        revision=ORIGINAL_TINYSTORIES_REVISION,
    )

    assert not report.passed
    assert len(report.hits) == 1
    assert report.hits[0].term == known
    assert report.hits[0].occurrence_count == 1


def test_streamed_novelty_audit_reports_zero_hits_for_nonce_fixture(tmp_path) -> None:
    contents = b"ordinary words only\n"
    path = tmp_path / "fixture.txt"
    path.write_bytes(contents)
    contract = PinnedDatasetFile(
        filename=path.name,
        size_bytes=len(contents),
        sha256=sha256(contents).hexdigest(),
    )

    report = audit_nonce_terms(
        path,
        (NoveltyTerm("zorvane", "class", "class:0"),),
        corpus_file=contract,
        revision=ORIGINAL_TINYSTORIES_REVISION,
    )

    assert report.passed
    assert report.hits == ()


def test_generated_novelty_terms_cover_names_classes_roles_and_inflections() -> None:
    bundles = (
        generate_calibration_bundle("0" * 64),
        generate_pilot_bundle("0" * 64),
    )

    terms = novelty_terms_for_bundles(bundles)

    assert len(terms) == sum(
        1 + len(entity.inflections)
        for bundle in bundles
        for entity in bundle.entities
    )
    assert {term.kind for term in terms} == {
        "name",
        "class",
        "role",
        "inflection",
    }
    assert {term.text for term in terms} == {
        lexical_form
        for bundle in bundles
        for entity in bundle.entities
        for lexical_form in (entity.name, *entity.inflections)
    }
