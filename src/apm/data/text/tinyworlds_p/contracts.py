"""Source-independent contracts for the TinyWorlds-P archive benchmark."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Literal

from apm.lm.config import GptNeoConfig


BENCHMARK_ID = "tinyworlds-p-archive-v1"
PARTITION_FORMAT = "tinyworlds-p-archive-partition"
PARTITION_SCHEMA_VERSION = 1
PUBLIC_SEED = 0
WORLD_LABELS = ("A", "B", "C", "D", "E")

SplitLabel = Literal["train", "validation", "test"]
BucketNamespace = Literal["noun", "verb", "adjective"]
PartitionRole = Literal["base", "world"]
ArchiveGroupStatus = Literal[
    "eligible",
    "empty_story",
    "unclassifiable_metadata",
    "conflicting_metadata",
]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def require_sha256(value: str, label: str) -> None:
    """Require one canonical lowercase SHA-256 string."""
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """One immutable local source file and its upstream identity."""

    dataset_id: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.dataset_id, self.revision, self.filename)
        ):
            raise ValueError("source identity strings must be nonempty")
        if _REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("source revision must be a lowercase Git SHA")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("source size_bytes must be positive")
        require_sha256(self.sha256, "source SHA-256")

    def as_record(self) -> dict[str, str | int]:
        """Return the canonical JSON representation of this source."""
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "revision": self.revision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class HashedFile:
    """One canonical filename, byte count, and SHA-256 digest."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or Path(self.name).name != self.name:
            raise ValueError("hashed filename must be one nonempty basename")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("hashed file size must be positive")
        require_sha256(self.sha256, f"hash for {self.name}")

    def as_record(self) -> dict[str, str | int]:
        """Return the canonical JSON representation of this file."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    """The pinned tokenizer identity used for mass accounting."""

    kind: str
    identifier: str
    revision: str
    vocab_size: int
    files: tuple[HashedFile, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.kind, self.identifier, self.revision)
        ):
            raise ValueError("tokenizer identity strings must be nonempty")
        if type(self.vocab_size) is not int or not 1 < self.vocab_size <= 65_536:
            raise ValueError("tokenizer vocabulary must fit little-endian uint16")
        if type(self.files) is not tuple or not self.files:
            raise ValueError("tokenizer identity requires hashed files")
        if any(type(item) is not HashedFile for item in self.files):
            raise TypeError("tokenizer files must contain HashedFile values")
        names = tuple(item.name for item in self.files)
        if len(set(names)) != len(names):
            raise ValueError("tokenizer filenames must be unique")
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda item: item.name)))

    @property
    def identity_sha256(self) -> str:
        """Return a digest over the complete canonical tokenizer contract."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical JSON representation of this tokenizer."""
        return {
            "files": [item.as_record() for item in self.files],
            "identifier": self.identifier,
            "kind": self.kind,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True, slots=True)
class NormalizationIdentity:
    """The versioned transform used only for duplicate-story identity."""

    version: str = "tinyworlds-p-normalization-v1"
    unicode_form: Literal["NFKC"] = "NFKC"
    case_folding: bool = True
    whitespace_collapse: bool = True
    canonical_straight_quotes: bool = True

    def __post_init__(self) -> None:
        if self.version != "tinyworlds-p-normalization-v1":
            raise ValueError("unsupported TinyWorlds-P normalization version")
        if self.unicode_form != "NFKC" or not all(
            (self.case_folding, self.whitespace_collapse, self.canonical_straight_quotes)
        ):
            raise ValueError("TinyWorlds-P normalization choices are immutable")

    def as_record(self) -> dict[str, str | bool]:
        """Return the canonical JSON representation of normalization."""
        return {
            "canonical_straight_quotes": self.canonical_straight_quotes,
            "case_folding": self.case_folding,
            "unicode_form": self.unicode_form,
            "version": self.version,
            "whitespace_collapse": self.whitespace_collapse,
        }


@dataclass(frozen=True, slots=True)
class Recipe:
    """One mechanically recovered and normalized TinyStories recipe."""

    noun: str
    verb: str
    adjective: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value for value in self.roles):
            raise ValueError("recipe roles must be normalized nonempty strings")
        if type(self.features) is not tuple or any(
            type(value) is not str or not value for value in self.features
        ):
            raise ValueError("recipe features must be normalized nonempty strings")
        if tuple(sorted(set(self.features))) != self.features:
            raise ValueError("recipe features must be unique and sorted")

    @property
    def roles(self) -> tuple[str, str, str]:
        """Return noun, verb, and adjective in canonical role order."""
        return self.noun, self.verb, self.adjective

    @property
    def feature_signature(self) -> str:
        """Return a stable compact signature of the narrative-feature set."""
        return "+".join(self.features) if self.features else "none"

    def as_record(self) -> dict[str, object]:
        """Return the canonical JSON representation of this recipe."""
        return {
            "adjective": self.adjective,
            "features": list(self.features),
            "noun": self.noun,
            "verb": self.verb,
        }


@dataclass(frozen=True, slots=True)
class ArchiveOccurrence:
    """One exact released archive entity and its temporary story location."""

    record_id: str
    source_member: str
    source_index: int
    content_sha256: str
    story_sha256: str
    source: Literal["GPT-3.5", "GPT-4"]
    spool_offset: int
    byte_length: int
    token_count: int

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("archive occurrence record_id must be nonempty")
        if type(self.source_member) is not str or not self.source_member:
            raise ValueError("archive occurrence member must be nonempty")
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("archive occurrence index must be nonnegative")
        if self.source not in ("GPT-3.5", "GPT-4"):
            raise ValueError("archive occurrence source label is unsupported")
        for value, label in (
            (self.content_sha256, "archive record content hash"),
            (self.story_sha256, "exact archive story hash"),
        ):
            require_sha256(value, label)
        if any(
            type(value) is not int or value < 0
            for value in (self.spool_offset, self.byte_length, self.token_count)
        ):
            raise ValueError("archive occurrence offsets and counts must be nonnegative")
        expected_id = (
            f"archive:{self.source_member}:{self.source_index}:{self.content_sha256}"
        )
        if self.record_id != expected_id:
            raise ValueError("archive occurrence ID does not bind location and content")

    def as_record(self) -> dict[str, object]:
        """Return the canonical sortable occurrence representation."""
        return {
            "byte_length": self.byte_length,
            "content_sha256": self.content_sha256,
            "record_id": self.record_id,
            "source": self.source,
            "source_index": self.source_index,
            "source_member": self.source_member,
            "spool_offset": self.spool_offset,
            "story_sha256": self.story_sha256,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class ArchiveDuplicateGroup:
    """One indivisible normalized archive-story group with full multiplicity."""

    normalized_story_sha256: str
    occurrences: tuple[ArchiveOccurrence, ...]
    status: ArchiveGroupStatus
    recipe: Recipe | None

    def __post_init__(self) -> None:
        require_sha256(self.normalized_story_sha256, "normalized archive story hash")
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise ValueError("archive duplicate group requires occurrences")
        if any(type(item) is not ArchiveOccurrence for item in self.occurrences):
            raise TypeError("archive duplicate group occurrences have the wrong type")
        if tuple(item.record_id for item in self.occurrences) != tuple(
            sorted(item.record_id for item in self.occurrences)
        ):
            raise ValueError("archive duplicate occurrences must be canonically ordered")
        if self.status == "eligible" and self.recipe is None:
            raise ValueError("eligible archive groups require one recipe")
        if self.status != "eligible" and self.recipe is not None:
            raise ValueError("excluded archive groups cannot retain a recipe")

    @property
    def active_token_count(self) -> int:
        """Return token mass across every released occurrence."""
        return sum(item.token_count for item in self.occurrences)


@dataclass(frozen=True, slots=True)
class ArchiveIngestAudit:
    """Complete archive-native record, duplicate, exclusion, and token counts."""

    archive_member_count: int
    archive_record_count: int
    archive_group_count: int
    nonempty_record_count: int
    nonempty_token_count: int
    classified_record_count: int
    classified_token_count: int
    empty_group_count: int
    empty_record_count: int
    unclassifiable_group_count: int
    unclassifiable_record_count: int
    unclassifiable_token_count: int
    conflicting_group_count: int
    conflicting_record_count: int
    conflicting_token_count: int
    eligible_group_count: int
    eligible_record_count: int
    eligible_token_count: int
    duplicate_group_count: int
    maximum_group_multiplicity: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("archive ingest counts must be nonnegative integers")
        if self.archive_member_count == 0 or self.archive_record_count == 0:
            raise ValueError("archive ingest must contain members and records")
        if self.archive_group_count == 0 or self.nonempty_token_count == 0:
            raise ValueError("archive ingest must contain nonempty token mass")
        if self.classified_token_count > self.nonempty_token_count:
            raise ValueError("classified token mass exceeds nonempty archive mass")

    @property
    def role_classification_coverage(self) -> float:
        """Return classified record-token mass over all nonempty record-token mass."""
        return self.classified_token_count / self.nonempty_token_count

    def as_record(self) -> dict[str, int | float]:
        """Return canonical persisted archive-ingest evidence."""
        record = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
        return {
            **record,
            "role_classification_coverage": self.role_classification_coverage,
        }


@dataclass(frozen=True, slots=True)
class WordBucket:
    """One frequency-balanced ingredient bucket."""

    namespace: BucketNamespace
    index: int
    token_mass: int
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.namespace not in ("noun", "verb", "adjective"):
            raise ValueError("unknown bucket namespace")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("bucket index must be nonnegative")
        if type(self.token_mass) is not int or self.token_mass < 0:
            raise ValueError("bucket token mass must be nonnegative")
        if type(self.words) is not tuple or tuple(sorted(set(self.words))) != self.words:
            raise ValueError("bucket words must be unique and sorted")


@dataclass(frozen=True, slots=True)
class WorldCell:
    """One canonically labelled held-out noun-bucket by verb-bucket cell."""

    label: str
    noun_bucket: int
    verb_bucket: int
    active_token_count: int
    group_count: int

    def __post_init__(self) -> None:
        if self.label not in WORLD_LABELS:
            raise ValueError("world cell label must be A through E")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.noun_bucket,
                self.verb_bucket,
                self.active_token_count,
                self.group_count,
            )
        ):
            raise ValueError("world cell indexes and counts must be nonnegative")
        if self.active_token_count == 0 or self.group_count == 0:
            raise ValueError("selected world cells must be nonempty")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """The canonical partition and split assignment for a duplicate group."""

    normalized_story_sha256: str
    status: ArchiveGroupStatus
    role: PartitionRole | None
    world: str | None
    split: SplitLabel | None

    def __post_init__(self) -> None:
        require_sha256(self.normalized_story_sha256, "assignment group hash")
        assigned = self.status == "eligible"
        if assigned != (self.role is not None and self.split is not None):
            raise ValueError("only eligible groups may have partition assignments")
        if self.role == "world" and self.world not in WORLD_LABELS:
            raise ValueError("world assignments require a canonical world label")
        if self.role != "world" and self.world is not None:
            raise ValueError("only world assignments may name a world")


@dataclass(frozen=True, slots=True)
class TokenShard:
    """One immutable exact-text or little-endian uint16 shard."""

    shard_id: int
    kind: Literal["text", "tokens"]
    relative_path: str
    size_bytes: int
    story_count: int

    def __post_init__(self) -> None:
        if type(self.shard_id) is not int or self.shard_id < 0:
            raise ValueError("shard_id must be nonnegative")
        if self.kind not in ("text", "tokens"):
            raise ValueError("unknown shard kind")
        path = Path(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("shard path must be safe and relative")
        if any(
            type(value) is not int or value < 0
            for value in (self.size_bytes, self.story_count)
        ):
            raise ValueError("shard size and story count must be nonnegative")


@dataclass(frozen=True, slots=True)
class DocumentIndex:
    """One archive entity bound to exact source and text/token shard coordinates."""

    record_id: str
    source_member: str
    source_index: int
    content_sha256: str
    story_sha256: str
    normalized_story_sha256: str
    text_shard: int
    text_offset: int
    text_bytes: int
    token_shard: int
    token_offset: int
    token_count: int
    role: PartitionRole
    world: str | None
    split: SplitLabel

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.record_id, self.source_member)
        ):
            raise ValueError("document archive identity strings must be nonempty")
        for value, label in (
            (self.content_sha256, "document content hash"),
            (self.story_sha256, "document story hash"),
            (self.normalized_story_sha256, "document group hash"),
        ):
            require_sha256(value, label)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.source_index,
                self.text_shard,
                self.text_offset,
                self.text_bytes,
                self.token_shard,
                self.token_offset,
                self.token_count,
            )
        ):
            raise ValueError("document coordinates and counts must be nonnegative")
        if self.text_bytes == 0 or self.token_count == 0:
            raise ValueError("indexed documents must be nonempty")
        if self.role not in ("base", "world"):
            raise ValueError("document role must be base or world")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("document split is invalid")
        if self.role == "world" and self.world not in WORLD_LABELS:
            raise ValueError("world document index requires a valid world")
        if self.role == "base" and self.world is not None:
            raise ValueError("base document index cannot name a world")


@dataclass(frozen=True, slots=True)
class ControlSelection:
    """One no-replacement held-in control matched to a world evaluation split."""

    world: str
    split: Literal["validation", "test"]
    group_sha256: tuple[str, ...]
    row_group_count: int
    column_group_count: int
    active_token_count: int

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS or self.split not in ("validation", "test"):
            raise ValueError("control selection requires a world validation/test split")
        if type(self.group_sha256) is not tuple or not self.group_sha256:
            raise ValueError("control selection must contain groups")
        if tuple(sorted(set(self.group_sha256))) != self.group_sha256:
            raise ValueError("control group hashes must be unique and sorted")
        for value in self.group_sha256:
            require_sha256(value, "control group hash")
        if self.row_group_count + self.column_group_count != len(self.group_sha256):
            raise ValueError("control arm counts must cover selected groups")
        if self.active_token_count <= 0:
            raise ValueError("control token count must be positive")


@dataclass(frozen=True, slots=True)
class SplitCount:
    """Persisted group, archive-record, and active-token totals for one split."""

    role: PartitionRole
    world: str | None
    split: SplitLabel
    group_count: int
    occurrence_count: int
    active_token_count: int

    def __post_init__(self) -> None:
        if self.role == "world" and self.world not in WORLD_LABELS:
            raise ValueError("world split count requires a valid world")
        if self.role == "base" and self.world is not None:
            raise ValueError("base split count cannot name a world")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.group_count,
                self.occurrence_count,
                self.active_token_count,
            )
        ):
            raise ValueError("split counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One bounded-work progress update."""

    phase: str
    completed: int
    total: int | None
    detail: str

    def __post_init__(self) -> None:
        if type(self.phase) is not str or not self.phase:
            raise ValueError("progress phase must be nonempty")
        if type(self.completed) is not int or self.completed < 0:
            raise ValueError("progress completed count must be nonnegative")
        if self.total is not None and (
            type(self.total) is not int or self.total < self.completed
        ):
            raise ValueError("progress total must cover completed work")
        if type(self.detail) is not str:
            raise TypeError("progress detail must be text")


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class PartitionInputs:
    """Authenticated archive/tokenizer inputs and bounded build locations."""

    archive_path: Path
    tokenizer_directory: Path
    output_root: Path
    temporary_directory: Path
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    progress: ProgressCallback | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "archive_path",
            "tokenizer_directory",
            "output_root",
            "temporary_directory",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if type(self.archive_identity) is not SourceIdentity:
            raise TypeError("archive_identity must be SourceIdentity")
        if type(self.tokenizer_identity) is not TokenizerIdentity:
            raise TypeError("tokenizer_identity must be TokenizerIdentity")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable")


@dataclass(frozen=True, slots=True)
class PartitionPreset:
    """Immutable archive ingestion and deterministic partition choices."""

    bucket_count: int = 8
    public_seed: int = PUBLIC_SEED
    worker_count: int = 16
    run_record_count: int = 50_000
    shard_target_bytes: int = 32 * 1024 * 1024
    batch_block_documents: int = 1_024
    context_length: int = 256
    batch_size: int = 32
    minimum_role_coverage: float = 0.95
    selected_cell_median_tolerance: float = 0.10
    minimum_component_outside_groups: int = 64
    world_split_weights: tuple[int, int, int] = (80, 10, 10)
    base_split_weights: tuple[int, int, int] = (96, 2, 2)
    control_token_tolerance: float = 0.0025
    control_source_feature_tolerance: float = 0.02
    control_adjective_length_tolerance: float = 0.03
    control_mean_length_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if type(self.bucket_count) is not int or self.bucket_count < 3:
            raise ValueError("five-cell topology requires at least three buckets")
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.worker_count,
                self.run_record_count,
                self.shard_target_bytes,
                self.batch_block_documents,
                self.context_length,
                self.batch_size,
            )
        ):
            raise ValueError("archive worker and sort-run sizes must be positive")
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("public seed must be nonnegative")
        if (
            type(self.minimum_component_outside_groups) is not int
            or self.minimum_component_outside_groups < 0
        ):
            raise ValueError("component visibility minimum must be nonnegative")
        for weights in (self.world_split_weights, self.base_split_weights):
            if (
                type(weights) is not tuple
                or len(weights) != 3
                or any(type(value) is not int or value <= 0 for value in weights)
            ):
                raise ValueError("split weights must be three positive integers")
        if (
            not isfinite(self.minimum_role_coverage)
            or not 0.0 <= self.minimum_role_coverage <= 1.0
        ):
            raise ValueError("role coverage minimum must lie in [0, 1]")
        tolerances = (
            self.selected_cell_median_tolerance,
            self.control_token_tolerance,
            self.control_source_feature_tolerance,
            self.control_adjective_length_tolerance,
            self.control_mean_length_tolerance,
        )
        if any(not isfinite(value) or not 0.0 < value < 1.0 for value in tolerances):
            raise ValueError("partition tolerances must lie in (0, 1)")

    def as_record(self) -> dict[str, object]:
        """Return every behavior-changing algorithm choice canonically."""
        return {
            "base_split_weights": list(self.base_split_weights),
            "batch_block_documents": self.batch_block_documents,
            "batch_size": self.batch_size,
            "bucket_count": self.bucket_count,
            "context_length": self.context_length,
            "control_adjective_length_tolerance": self.control_adjective_length_tolerance,
            "control_mean_length_tolerance": self.control_mean_length_tolerance,
            "control_source_feature_tolerance": self.control_source_feature_tolerance,
            "control_token_tolerance": self.control_token_tolerance,
            "minimum_component_outside_groups": self.minimum_component_outside_groups,
            "minimum_role_coverage": self.minimum_role_coverage,
            "public_seed": self.public_seed,
            "selected_cell_median_tolerance": self.selected_cell_median_tolerance,
            "shard_target_bytes": self.shard_target_bytes,
            "world_split_weights": list(self.world_split_weights),
        }


CANONICAL_ARCHIVE_IDENTITY = SourceIdentity(
    dataset_id="roneneldan/TinyStories",
    revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    filename="TinyStories_all_data.tar.gz",
    size_bytes=1_608_001_638,
    sha256="26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699",
)


CANONICAL_TOKENIZER_IDENTITY = TokenizerIdentity(
    kind="gpt2-bpe",
    identifier="roneneldan/TinyStories-8M",
    revision="8612e3b15c66ffa94eaa6ee0de5c96edd2d630af",
    vocab_size=50_257,
    files=(
        HashedFile(
            "merges.txt",
            456_318,
            "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
        HashedFile(
            "special_tokens_map.json",
            438,
            "98412137ae43c77f8af52eb51b19c3536d3242cb55339167d841005fa94a23b7",
        ),
        HashedFile(
            "tokenizer.json",
            2_107_652,
            "f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5",
        ),
        HashedFile(
            "tokenizer_config.json",
            722,
            "3d76da0fd37493fbfcd3f0fa9757753d31f92e1779ebd9130809b45546a60261",
        ),
        HashedFile(
            "vocab.json",
            798_156,
            "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
        ),
    ),
)

NORMALIZATION_IDENTITY = NormalizationIdentity()
PARTITION_PRESET = PartitionPreset()


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One strict relative path, size, and digest in a published tree."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("artifact path must be safe and relative")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("artifact file size must be nonnegative")
        require_sha256(self.sha256, f"artifact hash for {self.relative_path}")


@dataclass(frozen=True, slots=True)
class PartitionArtifact:
    """A strictly loaded content-addressed archive-only partition."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    normalization: NormalizationIdentity
    preset: PartitionPreset
    buckets: tuple[WordBucket, ...]
    cells: tuple[WorldCell, ...]
    controls: tuple[ControlSelection, ...]
    split_counts: tuple[SplitCount, ...]
    files: tuple[ArtifactFile, ...]
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.partition_sha256, "partition identity")
        require_sha256(self.manifest_sha256, "tree manifest hash")
        if type(self.archive_identity) is not SourceIdentity:
            raise TypeError("partition archive identity has the wrong type")
        if type(self.tokenizer_identity) is not TokenizerIdentity:
            raise TypeError("partition tokenizer identity has the wrong type")
        if type(self.buckets) is not tuple or any(
            type(item) is not WordBucket for item in self.buckets
        ):
            raise TypeError("artifact buckets must be WordBucket values")
        if tuple(cell.label for cell in self.cells) != WORLD_LABELS:
            raise ValueError("artifact cells must be canonically ordered A through E")
        if any(type(item) is not ControlSelection for item in self.controls):
            raise TypeError("artifact controls must be ControlSelection values")
        if any(type(item) is not SplitCount for item in self.split_counts):
            raise TypeError("artifact split counts must be SplitCount values")
        if any(type(item) is not ArtifactFile for item in self.files):
            raise TypeError("artifact files must be ArtifactFile values")
        for label, value in (("PAD", self.pad_token_id), ("EOS", self.eos_token_id)):
            if type(value) is not int or not 0 <= value < self.tokenizer_identity.vocab_size:
                raise ValueError(f"artifact {label} token ID is outside the vocabulary")


@dataclass(frozen=True, slots=True)
class BaseTrainingPreset:
    """The fixed scratch-training policy for the archive-only base."""

    parameter_seed: int = 0
    epochs: int = 5
    calibration_epochs: int = 2
    context_length: int = 256
    microbatch_size: int = 32
    accumulation_microbatches: int = 8
    maximum_learning_rate: float = 5e-4
    minimum_learning_rate: float = 5e-5
    warmup_fraction: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    state_interval_updates: int = 1_000
    allocator_peak_limit_bytes: int = 12 * 1024**3

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.calibration_epochs,
            self.context_length,
            self.microbatch_size,
            self.accumulation_microbatches,
            self.state_interval_updates,
            self.allocator_peak_limit_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("training dimensions and intervals must be positive")
        if self.parameter_seed != 0 or self.epochs != 5 or self.calibration_epochs != 2:
            raise ValueError("archive-v1 seed and epoch budget are fixed")
        if self.calibration_epochs >= self.epochs:
            raise ValueError("calibration must precede final training")
        floats = (
            self.maximum_learning_rate,
            self.minimum_learning_rate,
            self.warmup_fraction,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            self.weight_decay,
            self.gradient_clip_norm,
        )
        if any(not isfinite(value) or value <= 0.0 for value in floats):
            raise ValueError("optimizer values must be finite and positive")
        if self.minimum_learning_rate >= self.maximum_learning_rate:
            raise ValueError("minimum learning rate must be below maximum")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("warmup fraction must lie in (0, 1)")
        if not 0.0 < self.adam_beta1 < 1.0 or not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError("Adam betas must lie in (0, 1)")

    @property
    def model_config(self) -> GptNeoConfig:
        """Return the immutable eight-layer GPT-Neo architecture."""
        return GptNeoConfig(
            vocab_size=50_257,
            max_position_embeddings=2_048,
            hidden_size=256,
            intermediate_size=1_024,
            num_layers=8,
            num_heads=16,
            attention_types=("global", "local") * 4,
            local_window_size=256,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )

    def as_record(self) -> dict[str, object]:
        """Return the complete behavior-changing training contract."""
        config = self.model_config
        return {
            "accumulation_microbatches": self.accumulation_microbatches,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "allocator_peak_limit_bytes": self.allocator_peak_limit_bytes,
            "calibration_epochs": self.calibration_epochs,
            "context_length": self.context_length,
            "epochs": self.epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
            "maximum_learning_rate": self.maximum_learning_rate,
            "microbatch_size": self.microbatch_size,
            "minimum_learning_rate": self.minimum_learning_rate,
            "model": {
                "activation": config.activation,
                "attention_types": list(config.attention_types),
                "attention_dropout": config.attention_dropout,
                "embedding_dropout": config.embedding_dropout,
                "hidden_size": config.hidden_size,
                "initializer_range": config.initializer_range,
                "intermediate_size": config.intermediate_size,
                "layer_norm_epsilon": config.layer_norm_epsilon,
                "local_window_size": config.local_window_size,
                "max_position_embeddings": config.max_position_embeddings,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
                "residual_dropout": config.residual_dropout,
                "tied_embeddings": True,
                "vocab_size": config.vocab_size,
            },
            "parameter_seed": self.parameter_seed,
            "state_interval_updates": self.state_interval_updates,
            "warmup_fraction": self.warmup_fraction,
            "weight_decay": self.weight_decay,
        }


BASE_TRAINING_PRESET = BaseTrainingPreset()


def canonical_record_bytes(record: object) -> bytes:
    """Encode one JSON-compatible value with the artifact convention."""
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def record_sha256(record: object) -> str:
    """Hash one canonical JSON record."""
    return sha256(canonical_record_bytes(record)).hexdigest()


__all__ = [
    "ArchiveDuplicateGroup",
    "ArchiveGroupStatus",
    "ArchiveIngestAudit",
    "ArchiveOccurrence",
    "ArtifactFile",
    "BASE_TRAINING_PRESET",
    "BENCHMARK_ID",
    "BaseTrainingPreset",
    "CANONICAL_ARCHIVE_IDENTITY",
    "CANONICAL_TOKENIZER_IDENTITY",
    "ControlSelection",
    "DocumentIndex",
    "HashedFile",
    "NORMALIZATION_IDENTITY",
    "NormalizationIdentity",
    "PARTITION_PRESET",
    "PARTITION_FORMAT",
    "PARTITION_SCHEMA_VERSION",
    "PartitionArtifact",
    "PartitionInputs",
    "PartitionPreset",
    "ProgressEvent",
    "Recipe",
    "SourceIdentity",
    "SplitAssignment",
    "SplitCount",
    "TokenShard",
    "TokenizerIdentity",
    "WORLD_LABELS",
    "WordBucket",
    "WorldCell",
    "canonical_record_bytes",
    "record_sha256",
    "require_sha256",
]
