"""Remove original-training stories from the scored GPT-4 validation cohort.

The published TinyStories validation aggregates contain literal training
stories.  This module keeps that source defect separate from prompt scoring:
it authenticates the original training file, compares only normalized story
identity, and reuses the already-persisted validation observations for every
story that remains.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from apm.data.text.curricula import (
    PinnedDatasetFile,
    TINYSTORIES_DATASET_REVISION,
    _iter_tinystories_document_texts,
    verify_pinned_dataset_file,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
    build_reference_profile,
)
from apm.data.text.tinyworlds_v2.source_data import ValidationStoryRecord
from apm.data.text.tinyworlds_v2.surface import normalized_story_text


TINYSTORIES_ORIGINAL_TRAIN_FILE = PinnedDatasetFile(
    filename="TinyStories-train.txt",
    size_bytes=1_924_281_556,
    sha256="c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
)
VALIDATION_DECONTAMINATION_VERSION = (
    "tinyworlds-v2-validation-decontamination-v1"
)
VALIDATION_IDENTITY_POLICY = (
    "unicode-nfkc-casefold-whitespace-collapse-sha256-with-full-text-confirmation-v1"
)
VALIDATION_INPUT_COUNT = 10_000
VALIDATION_OVERLAP_COUNT = 3_393
VALIDATION_RETAINED_COUNT = 6_607
VALIDATION_OVERLAP_IDS_SHA256 = (
    "baf21f76c087d7ef1c1d22b7f63b5d654862ada893315d6ba1aa524c35a4b4c3"
)
VALIDATION_RETAINED_IDS_SHA256 = (
    "8d9ad964e36ba695b288b0c431921caeb47ff9a42b6d14a43f70093416005e28"
)
VALIDATION_RETAINED_PROFILE_SHA256 = (
    "0bdac5ca35c7f67fcc0560184fda8156991a32a8ce500fe65f358e8e6ddf0c61"
)


def _require_sha256(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DecontaminationExpectations:
    """Fixed coverage values for one validation cohort."""

    input_count: int
    overlap_count: int
    retained_count: int
    overlap_ids_sha256: str
    retained_ids_sha256: str
    retained_profile_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.input_count) is not int
            or type(self.overlap_count) is not int
            or type(self.retained_count) is not int
            or self.input_count <= 0
            or self.overlap_count < 0
            or self.retained_count <= 0
            or self.overlap_count + self.retained_count != self.input_count
        ):
            raise ValueError("decontamination coverage expectations are invalid")
        _require_sha256(self.overlap_ids_sha256, "overlap-ID digest")
        _require_sha256(self.retained_ids_sha256, "retained-ID digest")
        _require_sha256(self.retained_profile_sha256, "retained-profile digest")


PRODUCTION_DECONTAMINATION_EXPECTATIONS = DecontaminationExpectations(
    input_count=VALIDATION_INPUT_COUNT,
    overlap_count=VALIDATION_OVERLAP_COUNT,
    retained_count=VALIDATION_RETAINED_COUNT,
    overlap_ids_sha256=VALIDATION_OVERLAP_IDS_SHA256,
    retained_ids_sha256=VALIDATION_RETAINED_IDS_SHA256,
    retained_profile_sha256=VALIDATION_RETAINED_PROFILE_SHA256,
)


@dataclass(frozen=True, slots=True)
class ValidationDecontaminationAudit:
    """Canonical evidence for one training-membership exclusion pass."""

    input_count: int
    overlap_count: int
    retained_count: int
    overlap_ids_sha256: str
    retained_ids_sha256: str
    retained_profile_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.input_count) is not int
            or type(self.overlap_count) is not int
            or type(self.retained_count) is not int
            or self.input_count <= 0
            or self.overlap_count < 0
            or self.retained_count <= 0
            or self.overlap_count + self.retained_count != self.input_count
        ):
            raise ValueError("decontamination audit coverage is invalid")
        for value, label in (
            (self.overlap_ids_sha256, "overlap-ID digest"),
            (self.retained_ids_sha256, "retained-ID digest"),
            (self.retained_profile_sha256, "retained-profile digest"),
        ):
            _require_sha256(value, label)

    def as_record(self) -> dict[str, object]:
        """Return the strict, source-pinned audit record."""
        return {
            "identity_policy": VALIDATION_IDENTITY_POLICY,
            "input_count": self.input_count,
            "overlap_count": self.overlap_count,
            "overlap_ids_sha256": self.overlap_ids_sha256,
            "retained_count": self.retained_count,
            "retained_ids_sha256": self.retained_ids_sha256,
            "retained_profile_sha256": self.retained_profile_sha256,
            "train_source": {
                "dataset_id": "roneneldan/TinyStories",
                "filename": TINYSTORIES_ORIGINAL_TRAIN_FILE.filename,
                "revision": TINYSTORIES_DATASET_REVISION,
                "sha256": TINYSTORIES_ORIGINAL_TRAIN_FILE.sha256,
                "size_bytes": TINYSTORIES_ORIGINAL_TRAIN_FILE.size_bytes,
            },
            "version": VALIDATION_DECONTAMINATION_VERSION,
        }


@dataclass(frozen=True, slots=True)
class DecontaminatedValidationProfile:
    """Retained observation identities, rebuilt profile, and exclusion proof."""

    retained_record_ids: tuple[str, ...]
    retained_observations: tuple[ReferenceObservation, ...]
    profile: ReferenceProfile
    audit: ValidationDecontaminationAudit

    def __post_init__(self) -> None:
        if len(self.retained_record_ids) != len(self.retained_observations):
            raise ValueError("retained validation identities and observations differ")
        if self.profile.record_count != len(self.retained_record_ids):
            raise ValueError("retained validation profile coverage differs")
        if self.profile.profile_sha256 != self.audit.retained_profile_sha256:
            raise ValueError("retained validation profile digest differs")


def build_decontaminated_validation_profile(
    train_path: str | Path,
    validation_records: Sequence[ValidationStoryRecord],
    observations: Sequence[ReferenceObservation],
    *,
    expected_train_file: PinnedDatasetFile = TINYSTORIES_ORIGINAL_TRAIN_FILE,
    expectations: DecontaminationExpectations = (
        PRODUCTION_DECONTAMINATION_EXPECTATIONS
    ),
) -> DecontaminatedValidationProfile:
    """Authenticate train, exclude normalized matches, and rebuild the profile.

    ``observations`` may contain other reference cohorts.  Every supplied
    validation record must have exactly one matching observation; unrelated
    observations are ignored.
    """
    if any(type(item) is not ValidationStoryRecord for item in validation_records):
        raise TypeError("validation_records must contain ValidationStoryRecord values")
    if any(type(item) is not ReferenceObservation for item in observations):
        raise TypeError("observations must contain ReferenceObservation values")
    records = tuple(validation_records)
    if len(records) != expectations.input_count:
        raise ValueError("validation decontamination input coverage differs")
    record_ids = tuple(record.record_id for record in records)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("validation decontamination record IDs repeat")

    observation_by_id: dict[str, ReferenceObservation] = {}
    expected_ids = set(record_ids)
    for observation in observations:
        if observation.record_id not in expected_ids:
            continue
        if observation.record_id in observation_by_id:
            raise ValueError("validation observation identities repeat")
        observation_by_id[observation.record_id] = observation
    if set(observation_by_id) != expected_ids:
        raise ValueError("validation observations do not cover the source records")

    normalized_by_id = {
        record.record_id: normalized_story_text(record.story) for record in records
    }
    if any(not text for text in normalized_by_id.values()):
        raise ValueError("validation normalization produced an empty story")
    identities: dict[str, str] = {}
    for record_id, text in normalized_by_id.items():
        digest = sha256(text.encode("utf-8")).hexdigest()
        existing = identities.get(digest)
        if existing is not None:
            if normalized_by_id[existing] != text:
                raise RuntimeError("SHA-256 collision in validation identities")
            raise ValueError("validation contains duplicate normalized stories")
        identities[digest] = record_id

    verified_train = verify_pinned_dataset_file(train_path, expected_train_file)
    overlapping: set[str] = set()
    for raw_story in _iter_tinystories_document_texts(verified_train):
        normalized = normalized_story_text(raw_story)
        if not normalized:
            continue
        digest = sha256(normalized.encode("utf-8")).hexdigest()
        candidate_id = identities.get(digest)
        if candidate_id is not None:
            if normalized_by_id[candidate_id] != normalized:
                raise RuntimeError("SHA-256 collision across train and validation")
            overlapping.add(candidate_id)

    overlap_ids = tuple(record_id for record_id in record_ids if record_id in overlapping)
    retained_ids = tuple(record_id for record_id in record_ids if record_id not in overlapping)
    overlap_ids_sha256 = _ordered_id_digest(overlap_ids)
    retained_ids_sha256 = _ordered_id_digest(retained_ids)
    if (
        len(overlap_ids) != expectations.overlap_count
        or len(retained_ids) != expectations.retained_count
        or overlap_ids_sha256 != expectations.overlap_ids_sha256
        or retained_ids_sha256 != expectations.retained_ids_sha256
    ):
        raise ValueError("validation decontamination result differs from its fixed audit")
    retained_observations = tuple(observation_by_id[item] for item in retained_ids)
    profile = build_reference_profile(retained_observations)
    if profile.profile_sha256 != expectations.retained_profile_sha256:
        raise ValueError("validation decontamination profile differs from its fixed audit")
    audit = ValidationDecontaminationAudit(
        input_count=len(record_ids),
        overlap_count=len(overlap_ids),
        retained_count=len(retained_ids),
        overlap_ids_sha256=overlap_ids_sha256,
        retained_ids_sha256=retained_ids_sha256,
        retained_profile_sha256=profile.profile_sha256,
    )
    return DecontaminatedValidationProfile(
        retained_record_ids=retained_ids,
        retained_observations=retained_observations,
        profile=profile,
        audit=audit,
    )


def _ordered_id_digest(record_ids: Sequence[str]) -> str:
    return sha256(
        "".join(f"{record_id}\n" for record_id in record_ids).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DecontaminatedValidationProfile",
    "DecontaminationExpectations",
    "PRODUCTION_DECONTAMINATION_EXPECTATIONS",
    "TINYSTORIES_ORIGINAL_TRAIN_FILE",
    "VALIDATION_DECONTAMINATION_VERSION",
    "VALIDATION_IDENTITY_POLICY",
    "ValidationDecontaminationAudit",
    "build_decontaminated_validation_profile",
]
