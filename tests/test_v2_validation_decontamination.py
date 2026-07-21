from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from apm.data.text.curricula import PinnedDatasetFile, TINYSTORIES_DOCUMENT_SEPARATOR
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import ValidationStoryRecord
from apm.data.text.tinyworlds_v2.validation_decontamination import (
    DecontaminationExpectations,
    build_decontaminated_validation_profile,
)


def _validation_record(index: int, story: str) -> ValidationStoryRecord:
    digest = sha256(story.encode("utf-8")).hexdigest()
    return ValidationStoryRecord(
        record_id=f"v2-validation:{index}:{digest}",
        source_index=index,
        content_sha256=digest,
        story=story,
    )


def test_validation_decontamination_filters_normalized_train_membership(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "fixture-train.txt"
    train_payload = (
        "  ONCE   there was a CAT.  "
        + TINYSTORIES_DOCUMENT_SEPARATOR
        + "A separate train story."
    ).encode("utf-8")
    train_path.write_bytes(train_payload)
    train_file = PinnedDatasetFile(
        filename=train_path.name,
        size_bytes=len(train_payload),
        sha256=sha256(train_payload).hexdigest(),
    )
    records = (
        _validation_record(5, "Once there was a cat."),
        _validation_record(2, "A new blue bird flew home."),
        _validation_record(9, "Mia shared a warm red apple."),
    )
    observations = tuple(
        observe_reference(
            ReferenceRecord(record.record_id, record.story, "GPT-4"),
            model_token_ids=(index + 1, index + 2),
            normalized_nll=1.0 + index / 10,
        )
        for index, record in enumerate(records)
    )
    retained_ids = tuple(record.record_id for record in records[1:])
    overlap_ids = (records[0].record_id,)
    overlap_digest = sha256(
        "".join(f"{record_id}\n" for record_id in overlap_ids).encode("utf-8")
    ).hexdigest()
    retained_digest = sha256(
        "".join(f"{record_id}\n" for record_id in retained_ids).encode("utf-8")
    ).hexdigest()

    fixture_observations = tuple(reversed(observations))
    expected_profile = build_reference_profile(tuple(observations[1:]))
    result = build_decontaminated_validation_profile(
        train_path,
        records,
        fixture_observations,
        expected_train_file=train_file,
        expectations=DecontaminationExpectations(
            3,
            1,
            2,
            overlap_digest,
            retained_digest,
            expected_profile.profile_sha256,
        ),
    )

    assert result.retained_record_ids == retained_ids
    assert tuple(item.record_id for item in result.retained_observations) == retained_ids
    assert result.profile.record_count == 2
    assert result.audit.overlap_count == 1
    assert result.audit.retained_ids_sha256 == retained_digest


def test_validation_decontamination_rejects_a_changed_result_contract(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "fixture-train.txt"
    train_payload = b"A train story."
    train_path.write_bytes(train_payload)
    train_file = PinnedDatasetFile(
        train_path.name,
        len(train_payload),
        sha256(train_payload).hexdigest(),
    )
    record = _validation_record(0, "A different validation story.")
    observation = observe_reference(
        ReferenceRecord(record.record_id, record.story, "GPT-4"),
        model_token_ids=(1, 2),
        normalized_nll=1.0,
    )

    with pytest.raises(ValueError, match="fixed audit"):
        build_decontaminated_validation_profile(
            train_path,
            (record,),
            (observation,),
            expected_train_file=train_file,
            expectations=DecontaminationExpectations(
                1,
                0,
                1,
                sha256(b"").hexdigest(),
                "0" * 64,
                "0" * 64,
            ),
        )
