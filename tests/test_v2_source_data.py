from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import pytest

from apm.data.text.curricula import PinnedDatasetFile, TINYSTORIES_DOCUMENT_SEPARATOR
from apm.data.text.tinyworlds_v2.source_data import (
    ArchiveSourceSelections,
    TINYSTORIES_ALL_DATA_FILENAME,
    TINYSTORIES_ALL_DATA_SHA256,
    TINYSTORIES_ALL_DATA_SIZE_BYTES,
    TINYSTORIES_ALL_DATA_SOURCE,
    TinyStoriesArchiveSource,
    TinyStoriesSourceError,
    canonical_archive_record,
    canonical_jsonl,
    canonical_prompt_metadata_record,
    canonical_reference_story_record,
    canonical_validation_record,
    iter_archive_source_records,
    select_gpt4_archive_records,
    select_validation_story_records,
    verify_tinystories_archive,
)


def _released_record(index: int, *, source: str = "GPT-4") -> dict[str, object]:
    return {
        "story": f"Once there was a little fox number {index}. It was kind.",
        "instruction": {
            "prompt:": (
                f'Write a simple story using the noun "fox{index}", the '
                'adjective "kind", and the verb "garden".'
            ),
            "words": [f"fox{index}", "kind", "garden"],
            "features": ["Dialogue"] if index % 2 else [],
        },
        "summary": f"A kind fox has adventure {index}.",
        "source": source,
    }


def _write_archive(
    path: Path,
    members: tuple[tuple[str, bytes], ...],
) -> TinyStoriesArchiveSource:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    archive_bytes = path.read_bytes()
    return TinyStoriesArchiveSource(
        dataset_id="fixture/TinyStories",
        revision="a" * 40,
        archive_file=PinnedDatasetFile(
            filename=path.name,
            size_bytes=len(archive_bytes),
            sha256=sha256(archive_bytes).hexdigest(),
        ),
    )


def _json_array(records: list[dict[str, object]]) -> bytes:
    return json.dumps(records, ensure_ascii=False).encode("utf-8")


def _write_pinned_text(path: Path, text: str) -> PinnedDatasetFile:
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return PinnedDatasetFile(
        filename=path.name,
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
    )


def test_released_archive_contract_is_exact_and_revision_pinned() -> None:
    assert TINYSTORIES_ALL_DATA_FILENAME == "TinyStories_all_data.tar.gz"
    assert TINYSTORIES_ALL_DATA_SIZE_BYTES == 1_608_001_638
    assert TINYSTORIES_ALL_DATA_SHA256 == (
        "26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699"
    )
    assert TINYSTORIES_ALL_DATA_SOURCE.revision == (
        "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    )
    assert "/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/" in (
        TINYSTORIES_ALL_DATA_SOURCE.download_url
    )


def test_archive_verification_and_streaming_strict_records(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tar.gz"
    long_story = "A" * 70_000 + " end."
    records = [_released_record(0), _released_record(1, source="GPT-3.5")]
    records[0]["story"] = long_story
    source = _write_archive(
        path,
        (
            ("./data0.json", _json_array(records[:1])),
            ("./data1.json", _json_array(records[1:])),
        ),
    )

    assert verify_tinystories_archive(path, source) == path
    loaded = tuple(iter_archive_source_records(path, source))
    assert tuple(record.source for record in loaded) == ("GPT-4", "GPT-3.5")
    assert loaded[0].story == long_story
    assert loaded[0].instruction.words == ("fox0", "kind", "garden")
    assert loaded[0].source_member == "./data0.json"
    assert loaded[0].source_index == 0
    assert len(loaded[0].normalized_story_sha256) == 64
    assert loaded[0].record_id == (
        f"archive:./data0.json:0:{loaded[0].content_sha256}"
    )
    assert canonical_archive_record(loaded[0])["instruction"] == {
        "features": [],
        "prompt:": (
            'Write a simple story using the noun "fox0", the adjective "kind", '
            'and the verb "garden".'
        ),
        "words": ["fox0", "kind", "garden"],
    }


def test_archive_pin_rejects_size_digest_and_filename_drift(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(path, (("data.json", _json_array([_released_record(0)])),))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_tinystories_archive(path, source)

    original = path.with_name("other-name.tar.gz")
    original.write_bytes(path.read_bytes()[:-7])
    with pytest.raises(ValueError, match="expected pinned filename"):
        verify_tinystories_archive(original, source)


@pytest.mark.parametrize(
    "payload, message",
    (
        (
            b'[{"story":"x","instruction":{"prompt:":"p","words":["w"],'
            b'"features":[]},"summary":"s","source":"GPT-4","extra":1}]',
            "fields differ",
        ),
        (
            b'[{"story":"x","story":"y","instruction":{"prompt:":"p",'
            b'"words":["w"],"features":[]},"summary":"s","source":"GPT-4"}]',
            "duplicate field",
        ),
        (
            b'[{"story":"x","instruction":{"prompt":"p","words":["w"],'
            b'"features":[]},"summary":"s","source":"GPT-4"}]',
            "fields differ",
        ),
        (b"{}", "must contain a JSON array"),
        (b"[] trailing", "trailing JSON data"),
        (
            b'[{"story":"x","instruction":{"prompt:":"p","words":["w"],'
            b'"features":[]},"summary":"s","source":"GPT-4"},]',
            "trailing comma",
        ),
    ),
)
def test_archive_parser_rejects_noncanonical_source_shapes(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(path, (("data.json", payload),))
    with pytest.raises(TinyStoriesSourceError, match=message):
        tuple(iter_archive_source_records(path, source))


def test_archive_rejects_unexpected_or_unsafe_regular_members(tmp_path: Path) -> None:
    unexpected = tmp_path / "unexpected.tar.gz"
    source = _write_archive(unexpected, (("README.txt", b"not source data"),))
    with pytest.raises(TinyStoriesSourceError, match="non-JSON"):
        tuple(iter_archive_source_records(unexpected, source))

    unsafe = tmp_path / "unsafe.tar.gz"
    unsafe_source = _write_archive(
        unsafe,
        (("../data.json", _json_array([_released_record(0)])),),
    )
    with pytest.raises(TinyStoriesSourceError, match="unsafe archive member"):
        tuple(iter_archive_source_records(unsafe, unsafe_source))


def test_hash_ranked_archive_cohorts_are_stable_disjoint_and_gpt4(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(
        path,
        (
            (
                "data.json",
                _json_array(
                    [
                        _released_record(index, source="GPT-3.5" if index in (3, 9) else "GPT-4")
                        for index in range(14)
                    ]
                ),
            ),
        ),
    )
    records = tuple(iter_archive_source_records(path, source))
    keyword_arguments = {
        "seed": "phase-1-fixture",
        "prompt_metadata_count": 4,
        "reference_story_count": 4,
        "paired_count": 2,
    }
    first = select_gpt4_archive_records(records, **keyword_arguments)
    reordered = select_gpt4_archive_records(reversed(records), **keyword_arguments)

    first_ids = tuple(
        tuple(record.record_id for record in cohort)
        for cohort in (
            first.prompt_metadata_records,
            first.reference_story_records,
            first.paired_records,
        )
    )
    reordered_ids = tuple(
        tuple(record.record_id for record in cohort)
        for cohort in (
            reordered.prompt_metadata_records,
            reordered.reference_story_records,
            reordered.paired_records,
        )
    )
    assert first_ids == reordered_ids
    assert tuple(map(len, first_ids)) == (4, 4, 2)
    assert len(set().union(*(set(ids) for ids in first_ids))) == 10
    assert all(
        record.source == "GPT-4"
        for cohort in (
            first.prompt_metadata_records,
            first.reference_story_records,
            first.paired_records,
        )
        for record in cohort
    )


def test_archive_selection_fails_instead_of_reusing_records(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(
        path,
        (("data.json", _json_array([_released_record(index) for index in range(4)])),),
    )
    records = tuple(iter_archive_source_records(path, source))
    with pytest.raises(TinyStoriesSourceError, match="not enough eligible"):
        select_gpt4_archive_records(
            records,
            seed="fixture",
            prompt_metadata_count=2,
            reference_story_count=2,
            paired_count=1,
        )


def test_archive_cohorts_are_disjoint_by_normalized_story_content(
    tmp_path: Path,
) -> None:
    records = [_released_record(index) for index in range(8)]
    records[1]["story"] = str(records[0]["story"]).upper().replace(" ", "  ")
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(path, (("data.json", _json_array(records)),))

    selected = select_gpt4_archive_records(
        tuple(iter_archive_source_records(path, source)),
        seed="content-disjoint",
        prompt_metadata_count=2,
        reference_story_count=2,
        paired_count=1,
    )

    cohorts = (
        selected.prompt_metadata_records,
        selected.reference_story_records,
        selected.paired_records,
    )
    hashes = tuple(
        {record.normalized_story_sha256 for record in cohort} for cohort in cohorts
    )
    assert all(len(values) == len(cohort) for values, cohort in zip(hashes, cohorts))
    assert not hashes[0] & hashes[1]
    assert not hashes[0] & hashes[2]
    assert not hashes[1] & hashes[2]


def test_duplicate_archive_occurrences_do_not_change_content_hash_ranking(
    tmp_path: Path,
) -> None:
    base_records = [_released_record(index) for index in range(12)]
    duplicated_records = list(base_records)
    duplicate = _released_record(99)
    duplicate["story"] = str(base_records[4]["story"]).upper().replace(" ", "  ")
    duplicated_records.extend((duplicate, dict(duplicate)))

    base_path = tmp_path / "base.tar.gz"
    base_source = _write_archive(
        base_path,
        (("data.json", _json_array(base_records)),),
    )
    duplicate_path = tmp_path / "duplicates.tar.gz"
    duplicate_source = _write_archive(
        duplicate_path,
        (("data.json", _json_array(duplicated_records)),),
    )
    arguments = {
        "seed": "unique-content-ranking",
        "prompt_metadata_count": 3,
        "reference_story_count": 3,
        "paired_count": 2,
    }

    base = select_gpt4_archive_records(
        tuple(iter_archive_source_records(base_path, base_source)),
        **arguments,
    )
    with_duplicates = select_gpt4_archive_records(
        tuple(iter_archive_source_records(duplicate_path, duplicate_source)),
        **arguments,
    )

    def content_cohorts(
        selection: ArchiveSourceSelections,
    ) -> tuple[frozenset[str], ...]:
        return tuple(
            frozenset(record.normalized_story_sha256 for record in cohort)
            for cohort in (
                selection.prompt_metadata_records,
                selection.reference_story_records,
                selection.paired_records,
            )
        )

    assert content_cohorts(base) == content_cohorts(with_duplicates)


def test_empty_released_strings_are_preserved_but_not_used_as_stories(
    tmp_path: Path,
) -> None:
    records = [_released_record(index) for index in range(4)]
    records[0]["story"] = ""
    instruction = records[1]["instruction"]
    assert type(instruction) is dict
    instruction["prompt:"] = ""
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(path, (("data.json", _json_array(records)),))
    loaded = tuple(iter_archive_source_records(path, source))

    assert loaded[0].story == ""
    selected = select_gpt4_archive_records(
        loaded,
        seed="fixture",
        prompt_metadata_count=1,
        reference_story_count=1,
        paired_count=1,
    )
    assert all(record.story.strip() for record in selected.reference_story_records)
    assert all(record.story.strip() for record in selected.paired_records)
    assert all(
        record.instruction.prompt.strip()
        for record in selected.prompt_metadata_records + selected.paired_records
    )


def test_validation_selection_is_unique_stable_and_preserves_story_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture-valid.txt"
    stories = (
        "Story zero.\n\nA second paragraph.",
        "Story one.",
        "Story zero.\n\nA second paragraph.",
        "Story two.",
        "Story three.",
    )
    aggregate = TINYSTORIES_DOCUMENT_SEPARATOR.join(stories)
    expected_file = _write_pinned_text(path, aggregate)

    first = select_validation_story_records(
        path,
        seed="phase-1-fixture",
        count=4,
        expected_file=expected_file,
    )
    second = select_validation_story_records(
        path,
        seed="phase-1-fixture",
        count=4,
        expected_file=expected_file,
    )
    assert first == second
    assert len({record.content_sha256 for record in first}) == 4
    paragraph_story = next(record for record in first if "second paragraph" in record.story)
    assert "\n\n" in paragraph_story.story
    assert paragraph_story.source_index == 0
    assert canonical_validation_record(paragraph_story)["record_id"] == (
        paragraph_story.record_id
    )


def test_validation_selection_rejects_insufficient_unique_documents(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture-valid.txt"
    expected_file = _write_pinned_text(
        path,
        f"same{TINYSTORIES_DOCUMENT_SEPARATOR}same",
    )
    with pytest.raises(TinyStoriesSourceError, match="only 1 unique"):
        select_validation_story_records(
            path,
            seed="fixture",
            count=2,
            expected_file=expected_file,
        )


def test_validation_selection_excludes_normalized_archive_story_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture-valid.txt"
    stories = ("A Little Cat.", "A little dog.", "A little bird.")
    expected_file = _write_pinned_text(
        path,
        TINYSTORIES_DOCUMENT_SEPARATOR.join(stories),
    )
    excluded = select_validation_story_records(
        path,
        seed="fixture",
        count=1,
        expected_file=expected_file,
    )[0].normalized_story_sha256

    selected = select_validation_story_records(
        path,
        seed="fixture",
        count=2,
        expected_file=expected_file,
        exclude_normalized_story_sha256=frozenset((excluded,)),
    )

    assert excluded not in {item.normalized_story_sha256 for item in selected}


def test_source_jsonl_is_canonical_and_order_preserving(tmp_path: Path) -> None:
    path = tmp_path / "fixture.tar.gz"
    source = _write_archive(path, (("data.json", _json_array([_released_record(0)])),))
    record = next(iter_archive_source_records(path, source))
    payload = canonical_jsonl((canonical_archive_record(record),))
    assert payload.endswith(b"\n")
    assert payload == canonical_jsonl((canonical_archive_record(record),))
    assert b'"content_sha256"' in payload
    assert payload.index(b'"content_sha256"') < payload.index(b'"record_id"')
    prompt_evidence = canonical_prompt_metadata_record(record)
    assert prompt_evidence["story"] == record.story
    assert prompt_evidence["summary"] == record.summary
    assert "prompt" not in canonical_reference_story_record(record)
