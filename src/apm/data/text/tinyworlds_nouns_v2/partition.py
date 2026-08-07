"""Authenticated parent binding and disjoint nouns-v2 partition publication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import heapq
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile

from apm.data.text.curricula import normalize_text
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    NounDecision as V1NounDecision,
    record_sha256 as v1_record_sha256,
)
from apm.data.text.tinyworlds_nouns_v1.partition import (
    decisions_record as v1_decisions_record,
    load_noun_breakdown as load_v1_breakdown,
    load_noun_decisions as load_v1_decisions,
    load_noun_partition as load_v1_partition,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    AUDIT_FORMAT,
    BASE_UNIVERSE_STORY_COUNT,
    BASE_VALIDATION_BUCKET_COUNT,
    BENCHMARK_ID,
    DATA_ROOT,
    EXCLUDED_TRAIN_STORY_COUNT,
    EXCLUDED_VALIDATION_STORY_COUNT,
    EXCLUSION_FORMAT,
    EXPECTED_PURE_COUNTS,
    INDEX_FORMAT,
    MANIFEST_FORMAT,
    MINIMUM_TASK_TRAIN_STORIES,
    MINIMUM_TASK_VALIDATION_STORIES,
    MODEL_POSITION_LIMIT,
    PARENT_BREAKDOWN_SHA256,
    PARENT_DECISIONS_CORE_SHA256,
    PARENT_DECISIONS_SHA256,
    PARENT_PARTITION_SHA256,
    PARENT_STORY_COUNT,
    PARTITION_FORMAT,
    PROBE_STORY_COUNT,
    PURE_TASK_TRAIN_STORY_COUNT,
    PURE_TASK_VALIDATION_STORY_COUNT,
    SCHEMA_VERSION,
    TASK_IDS,
    TRAIN_UNIQUE_STORY_COUNT,
    VALIDATION_UNIQUE_STORY_COUNT,
    NounConceptFamily,
    NounsV2Manifest,
    NounsV2PartitionArtifact,
    NounsV2TaskSummary,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.lm.text import TextTokenizer


PARENT_STORY_LEDGER_SHA256 = (
    "1b0395363669d9a7cd73e7f53a370ec59e1e40ce62227f52095c434f5e79af37"
)
PARENT_STORY_STORE_SHA256 = (
    "3f69b6750b20e8401a42ca16c76851681fdac20c82ee1cd7bb995a3d187c094c"
)
PARENT_TOKEN_STORE_SHA256 = (
    "3a470eb2060b779b21e76c55db964655faf835506495564d4715a4c98778e558"
)
TRAIN_HOLDOUT_NAMESPACE = f"{BENCHMARK_ID}:base-validation"
PROBE_NAMESPACE = f"{BENCHMARK_ID}:probe"
EXAMPLE_NAMESPACE = f"{BENCHMARK_ID}:audit-example"
ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(frozen=True, slots=True)
class StoryAssignment:
    """Selected-concept classification of one story."""

    role: str
    task_id: str | None
    selected_concepts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in ("base", "task", "excluded"):
            raise ValueError("story assignment role is invalid")
        if (self.role == "task") != (self.task_id is not None):
            raise ValueError("only pure task stories may have a task ID")
        if len(self.selected_concepts) != len(set(self.selected_concepts)):
            raise ValueError("selected concepts must be unique")


@dataclass(frozen=True, slots=True)
class FixtureTaskPartition:
    """Small CPU-fixture task split with deterministic probes and updates."""

    task_id: str
    train_story_ids: tuple[str, ...]
    update_story_ids: tuple[str, ...]
    validation_story_ids: tuple[str, ...]
    probe_story_ids: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return a canonical fixture task record."""
        return {
            "probe_story_ids": list(self.probe_story_ids),
            "task_id": self.task_id,
            "train_story_ids": list(self.train_story_ids),
            "update_story_ids": list(self.update_story_ids),
            "validation_story_ids": list(self.validation_story_ids),
        }


@dataclass(frozen=True, slots=True)
class FixtureDisjointPartition:
    """Content-addressed in-memory partition used by fast CPU contracts."""

    task_ids: tuple[str, ...]
    base_universe_story_ids: tuple[str, ...]
    base_train_story_ids: tuple[str, ...]
    base_validation_story_ids: tuple[str, ...]
    excluded_train_story_ids: tuple[str, ...]
    excluded_validation_story_ids: tuple[str, ...]
    tasks: tuple[FixtureTaskPartition, ...]
    probe_count: int

    def __post_init__(self) -> None:
        base = set(self.base_universe_story_ids)
        task_sets = tuple(set(task.train_story_ids) for task in self.tasks)
        if (
            tuple(task.task_id for task in self.tasks) != self.task_ids
            or base != set(self.base_train_story_ids) | set(self.base_validation_story_ids)
            or set(self.base_train_story_ids) & set(self.base_validation_story_ids)
            or any(base & task_set for task_set in task_sets)
            or any(
                left & right
                for index, left in enumerate(task_sets)
                for right in task_sets[index + 1 :]
            )
            or any(
                len(task.probe_story_ids) != self.probe_count
                or set(task.probe_story_ids) & set(task.update_story_ids)
                or set(task.probe_story_ids) | set(task.update_story_ids)
                != set(task.train_story_ids)
                for task in self.tasks
            )
        ):
            raise ValueError("fixture partition is not exactly disjoint")

    @property
    def partition_sha256(self) -> str:
        """Return the fixture partition content identity."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical self-hashing fixture record."""
        core = {
            "base_train_story_ids": list(self.base_train_story_ids),
            "base_universe_story_ids": list(self.base_universe_story_ids),
            "base_validation_story_ids": list(self.base_validation_story_ids),
            "excluded_train_story_ids": list(self.excluded_train_story_ids),
            "excluded_validation_story_ids": list(
                self.excluded_validation_story_ids
            ),
            "format": "tinyworlds-nouns-disjoint-fixture-v2",
            "probe_count": self.probe_count,
            "task_ids": list(self.task_ids),
            "tasks": [task.as_record() for task in self.tasks],
        }
        return {**core, "partition_sha256": record_sha256(core)} if include_hash else core


@dataclass(frozen=True, slots=True)
class ParentStoryRow:
    """One validated pointer and concept record from the parent story ledger."""

    story_id: str
    source_split: str
    story_index: int
    story_offset: int
    byte_length: int
    token_offset: int
    token_count: int
    concept_ids: tuple[str, ...]
    matched_forms: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def index_record(self) -> dict[str, object]:
        """Return the six-field pointer consumed by the shared noun engine."""
        return {
            "byte_length": self.byte_length,
            "story_id": self.story_id,
            "story_index": self.story_index,
            "story_offset": self.story_offset,
            "token_count": self.token_count,
            "token_offset": self.token_offset,
        }


@dataclass(slots=True)
class _ScanState:
    split_roles: Counter[tuple[str, str]]
    raw_story_counts: Counter[tuple[str, str]]
    retained_story_counts: Counter[tuple[str, str]]
    raw_form_counts: Counter[tuple[str, str, str]]
    retained_form_counts: Counter[tuple[str, str, str]]
    base_train_count: int
    base_validation_count: int
    root_probe_heap: list[tuple[int, str, ParentStoryRow]]
    task_probe_heaps: dict[str, list[tuple[int, str, ParentStoryRow]]]
    example_heaps: dict[tuple[str, str, str], list[tuple[int, str, ParentStoryRow]]]


@dataclass(frozen=True, slots=True)
class _Selections:
    base_train_count: int
    base_validation_count: int
    root_probe_ids: tuple[str, ...]
    tasks: tuple[NounsV2TaskSummary, ...]
    example_ids: frozenset[str]


def selected_families_from_review(
    decisions: tuple[V1NounDecision, ...],
) -> tuple[NounConceptFamily, ...]:
    """Select the exact reviewed forms for the frozen ordered task list."""
    by_id = {decision.concept_id: decision for decision in decisions}
    if set(TASK_IDS) - set(by_id):
        raise ValueError("reviewed v1 decisions are missing a selected v2 concept")
    return tuple(
        NounConceptFamily(task_id, by_id[task_id].category, by_id[task_id].forms)
        for task_id in TASK_IDS
    )


def match_selected_forms(
    text: str,
    families: Sequence[NounConceptFamily],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Match selected concept families by case-insensitive whole words."""
    words = frozenset(
        word for word in _words(normalize_text(text).casefold())
    )
    return tuple(
        (family.task_id, matched)
        for family in families
        for matched in (tuple(form for form in family.forms if form in words),)
        if matched
    )


def classify_selected_concepts(
    concept_ids: Iterable[str],
    task_ids: Sequence[str] = TASK_IDS,
) -> StoryAssignment:
    """Classify zero matches as base, one as its task, and multiple as excluded."""
    present = frozenset(concept_ids)
    selected = tuple(task_id for task_id in task_ids if task_id in present)
    return (
        StoryAssignment("base", None, selected)
        if not selected
        else StoryAssignment("task", selected[0], selected)
        if len(selected) == 1
        else StoryAssignment("excluded", None, selected)
    )


def build_fixture_disjoint_partition(
    train_documents: Sequence[str],
    validation_documents: Sequence[str],
    tokenizer: TextTokenizer,
    families: Sequence[NounConceptFamily],
    *,
    minimum_train_stories: int = 1,
    minimum_validation_stories: int = 1,
    probe_count: int = 1,
    base_validation_bucket_count: int = BASE_VALIDATION_BUCKET_COUNT,
) -> FixtureDisjointPartition:
    """Build the production assignment semantics over a small in-memory corpus."""
    if (
        not families
        or minimum_train_stories <= 0
        or minimum_validation_stories <= 0
        or probe_count <= 0
        or base_validation_bucket_count <= 0
    ):
        raise ValueError("fixture thresholds and dimensions must be positive")
    family_ids = tuple(family.task_id for family in families)
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("fixture task families must be unique")
    validation = _fixture_documents(validation_documents, "validation", tokenizer, families)
    train = {
        story_id: row
        for story_id, row in _fixture_documents(
            train_documents, "train", tokenizer, families
        ).items()
        if story_id not in validation
    }
    rows = tuple(train.values()) + tuple(validation.values())
    assignments = {
        row.story_id: classify_selected_concepts(row.concept_ids, family_ids)
        for row in rows
    }
    pure_train = {
        task_id: tuple(
            sorted(
                row.story_id
                for row in train.values()
                if assignments[row.story_id].task_id == task_id
            )
        )
        for task_id in family_ids
    }
    pure_validation = {
        task_id: tuple(
            sorted(
                row.story_id
                for row in validation.values()
                if assignments[row.story_id].task_id == task_id
            )
        )
        for task_id in family_ids
    }
    if any(
        len(pure_train[task_id]) < minimum_train_stories
        or len(pure_validation[task_id]) < minimum_validation_stories
        for task_id in family_ids
    ):
        raise ValueError("fixture purified task falls below its frozen threshold")
    task_ids = tuple(
        sorted(family_ids, key=lambda task_id: (-len(pure_train[task_id]), task_id))
    )
    train_by_id = {row.story_id: row for row in train.values()}
    tasks = tuple(
        _fixture_task(
            task_id,
            pure_train[task_id],
            pure_validation[task_id],
            train_by_id,
            probe_count,
        )
        for task_id in task_ids
    )
    base_universe = tuple(
        sorted(
            row.story_id
            for row in train.values()
            if assignments[row.story_id].role == "base"
        )
    )
    base_validation = tuple(
        story_id
        for story_id in base_universe
        if _hash_bucket(
            TRAIN_HOLDOUT_NAMESPACE,
            story_id,
            base_validation_bucket_count,
        )
        == 0
    )
    base_validation_set = frozenset(base_validation)
    return FixtureDisjointPartition(
        task_ids=task_ids,
        base_universe_story_ids=base_universe,
        base_train_story_ids=tuple(
            story_id
            for story_id in base_universe
            if story_id not in base_validation_set
        ),
        base_validation_story_ids=base_validation,
        excluded_train_story_ids=tuple(
            sorted(
                row.story_id
                for row in train.values()
                if assignments[row.story_id].role == "excluded"
            )
        ),
        excluded_validation_story_ids=tuple(
            sorted(
                row.story_id
                for row in validation.values()
                if assignments[row.story_id].role == "excluded"
            )
        ),
        tasks=tasks,
        probe_count=probe_count,
    )


def load_fixture_disjoint_partition(payload: bytes) -> FixtureDisjointPartition:
    """Strict-load a canonical fixture partition and reject any byte tampering."""
    record = _canonical_object(payload, "nouns-v2 fixture partition")
    supplied = record.pop("partition_sha256", None)
    if (
        supplied != record_sha256(record)
        or record.get("format") != "tinyworlds-nouns-disjoint-fixture-v2"
    ):
        raise ValueError("nouns-v2 fixture partition identity changed")
    task_ids = tuple(
        _text(value, "fixture task ID")
        for value in _list(record.get("task_ids"), "fixture task IDs")
    )
    stories = lambda key: tuple(
        _text(value, key) for value in _list(record.get(key), key)
    )
    tasks = tuple(
        FixtureTaskPartition(
            task_id=_text(item.get("task_id"), "fixture task"),
            train_story_ids=tuple(
                _text(value, "fixture train story")
                for value in _list(item.get("train_story_ids"), "fixture train")
            ),
            update_story_ids=tuple(
                _text(value, "fixture update story")
                for value in _list(item.get("update_story_ids"), "fixture updates")
            ),
            validation_story_ids=tuple(
                _text(value, "fixture validation story")
                for value in _list(
                    item.get("validation_story_ids"), "fixture validation"
                )
            ),
            probe_story_ids=tuple(
                _text(value, "fixture probe story")
                for value in _list(item.get("probe_story_ids"), "fixture probes")
            ),
        )
        for raw in _list(record.get("tasks"), "fixture tasks")
        for item in (_object(raw, "fixture task"),)
    )
    partition = FixtureDisjointPartition(
        task_ids=task_ids,
        base_universe_story_ids=stories("base_universe_story_ids"),
        base_train_story_ids=stories("base_train_story_ids"),
        base_validation_story_ids=stories("base_validation_story_ids"),
        excluded_train_story_ids=stories("excluded_train_story_ids"),
        excluded_validation_story_ids=stories("excluded_validation_story_ids"),
        tasks=tasks,
        probe_count=_integer(record.get("probe_count"), "fixture probe count"),
    )
    if partition.partition_sha256 != supplied:
        raise ValueError("nouns-v2 fixture partition reconstruction changed")
    return partition


def authenticate_parent_manifest(
    parent_root: str | Path,
    breakdown_root: str | Path | None = None,
) -> NounsV2Manifest:
    """Strict-load the parent partition/review and derive the frozen v2 manifest."""
    root = Path(parent_root)
    if root.name != PARENT_PARTITION_SHA256:
        raise ValueError("nouns-v2 requires the frozen nouns-v1 parent partition")
    parent = load_v1_partition(root / "partition.json")
    parent_record = _canonical_object(
        (root / "partition.json").read_bytes(), "parent partition"
    )
    review = (
        Path(breakdown_root)
        if breakdown_root is not None
        else root.parents[1] / "noun-breakdowns" / PARENT_BREAKDOWN_SHA256
    )
    breakdown = load_v1_breakdown(review / "noun-breakdown.json")
    decisions = load_v1_decisions(review / "decisions.json")
    decision_record = _canonical_object(
        (review / "decisions.json").read_bytes(), "parent decisions"
    )
    file_records = _object(parent_record.get("files"), "parent files")
    expected_store_hashes = {
        "stories.bin": PARENT_STORY_STORE_SHA256,
        "stories.jsonl": PARENT_STORY_LEDGER_SHA256,
        "tokens.uint16": PARENT_TOKEN_STORE_SHA256,
    }
    if (
        parent.partition_sha256 != PARENT_PARTITION_SHA256
        or breakdown.breakdown_sha256 != PARENT_BREAKDOWN_SHA256
        or v1_record_sha256(v1_decisions_record(decisions))
        != PARENT_DECISIONS_SHA256
        or decision_record.get("decisions_sha256")
        != PARENT_DECISIONS_CORE_SHA256
        or parent.breakdown_sha256 != PARENT_BREAKDOWN_SHA256
        or parent.source_identity != breakdown.source_identity
        or parent.tokenizer_identity != breakdown.tokenizer_identity
        or parent_record.get("story_count") != PARENT_STORY_COUNT
        or any(
            _object(file_records.get(name), f"parent {name}").get("sha256") != digest
            for name, digest in expected_store_hashes.items()
        )
    ):
        raise ValueError("nouns-v1 parent source, tokenizer, review, or stores changed")
    return NounsV2Manifest(
        parent_partition_sha256=parent.partition_sha256,
        parent_breakdown_sha256=breakdown.breakdown_sha256,
        parent_decisions_sha256=PARENT_DECISIONS_SHA256,
        source_identity=parent.source_identity,
        tokenizer_identity=parent.tokenizer_identity,
        parent_story_count=int(parent_record["story_count"]),
        task_families=selected_families_from_review(decisions),
    )


def publish_manifest(
    manifest: NounsV2Manifest,
    output_root: str | Path = DATA_ROOT,
) -> Path:
    """Publish the canonical v2 manifest once and reject replacement content."""
    path = Path(output_root) / "manifest.json"
    payload = canonical_json_bytes(manifest.as_record())
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError("published nouns-v2 manifest changed")
        return path
    if path.exists():
        raise ValueError("nouns-v2 manifest path is not a regular file")
    _atomic_write(path, payload)
    return path


def load_manifest(path: str | Path) -> NounsV2Manifest:
    """Strict-load a canonical, self-hashing v2 manifest."""
    record = _canonical_object(Path(path).read_bytes(), "nouns-v2 manifest")
    supplied = record.pop("manifest_sha256", None)
    if (
        set(record)
        != {
            "assignment_policy",
            "base_validation_bucket_count",
            "format",
            "minimum_task_train_stories",
            "minimum_task_validation_stories",
            "parent_breakdown_sha256",
            "parent_decisions_sha256",
            "parent_partition_sha256",
            "parent_story_count",
            "probe_story_count",
            "schema_version",
            "source_identity",
            "task_families",
            "tokenizer_identity",
        }
        or supplied != record_sha256(record)
        or record.get("format") != MANIFEST_FORMAT
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("assignment_policy")
        != "zero-selected=base; one-selected=task; two-or-more=excluded"
        or record.get("base_validation_bucket_count")
        != BASE_VALIDATION_BUCKET_COUNT
        or record.get("minimum_task_train_stories")
        != MINIMUM_TASK_TRAIN_STORIES
        or record.get("minimum_task_validation_stories")
        != MINIMUM_TASK_VALIDATION_STORIES
        or record.get("probe_story_count") != PROBE_STORY_COUNT
    ):
        raise ValueError("nouns-v2 manifest identity or policy changed")
    families = tuple(
        NounConceptFamily(
            _text(item.get("task_id"), "manifest task"),
            _text(item.get("category"), "manifest category"),
            tuple(
                _text(form, "manifest form")
                for form in _list(item.get("forms"), "manifest forms")
            ),
        )
        for raw in _list(record.get("task_families"), "manifest task families")
        for item in (_object(raw, "manifest task family"),)
    )
    manifest = NounsV2Manifest(
        parent_partition_sha256=_text(
            record.get("parent_partition_sha256"), "parent partition"
        ),
        parent_breakdown_sha256=_text(
            record.get("parent_breakdown_sha256"), "parent breakdown"
        ),
        parent_decisions_sha256=_text(
            record.get("parent_decisions_sha256"), "parent decisions"
        ),
        source_identity=_object(record.get("source_identity"), "manifest source"),
        tokenizer_identity=_object(
            record.get("tokenizer_identity"), "manifest tokenizer"
        ),
        parent_story_count=_integer(record.get("parent_story_count"), "story count"),
        task_families=families,
    )
    if manifest.manifest_sha256 != supplied:
        raise ValueError("nouns-v2 manifest reconstruction changed")
    return manifest


def find_partition(
    manifest: NounsV2Manifest,
    output_root: str | Path = DATA_ROOT,
) -> NounsV2PartitionArtifact | None:
    """Find the sole strict partition bound to one v2 manifest, if published."""
    parent = Path(output_root) / "partitions"
    candidates = tuple(parent.glob("*/partition.json")) if parent.is_dir() else ()
    matches = tuple(
        path
        for path in candidates
        for record in (_canonical_object(path.read_bytes(), "partition candidate"),)
        if record.get("manifest_sha256") == manifest.manifest_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple nouns-v2 partitions bind the same manifest")
    return load_nouns_v2_partition(matches[0]) if matches else None


def build_nouns_v2_partition(
    manifest: NounsV2Manifest,
    parent_root: str | Path,
    output_root: str | Path = DATA_ROOT,
    *,
    progress: ProgressCallback | None = None,
) -> NounsV2PartitionArtifact:
    """Rebuild and publish every disjoint assignment index from the parent ledger."""
    parent = Path(parent_root)
    authenticated = authenticate_parent_manifest(parent)
    if authenticated != manifest:
        raise ValueError("partition manifest differs from authenticated parent review")
    publish_manifest(manifest, output_root)
    target_parent = Path(output_root) / "partitions"
    target_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".partition-v2.tmp-", dir=target_parent))
    try:
        core = _write_partition_tree(manifest, parent, temporary, progress)
        partition_sha256 = record_sha256(core)
        _write_fsync(
            temporary / "partition.json",
            canonical_json_bytes(
                {**core, "partition_sha256": partition_sha256}
            ),
        )
        target = target_parent / partition_sha256
        if target.is_dir():
            existing = load_nouns_v2_partition(target / "partition.json")
            _require_tree_equal(temporary, target)
            shutil.rmtree(temporary)
            return existing
        if target.exists():
            raise ValueError("nouns-v2 partition target is not a directory")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_nouns_v2_partition(target / "partition.json")


def verify_byte_identical_rebuild(
    partition: NounsV2PartitionArtifact,
    manifest: NounsV2Manifest,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Independently rebuild the complete tree and require the same content hash."""
    work_parent = partition.root.parents[1] / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="partition-rebuild-", dir=work_parent))
    try:
        core = _write_partition_tree(
            manifest,
            partition.parent_root,
            temporary,
            progress,
        )
        if record_sha256(core) != partition.partition_sha256:
            raise ValueError("independent nouns-v2 partition rebuild changed bytes")
        published = _canonical_object(
            (partition.root / "partition.json").read_bytes(), "published partition"
        )
        if core != {key: value for key, value in published.items() if key != "partition_sha256"}:
            raise ValueError("rebuilt nouns-v2 partition record differs")
    finally:
        shutil.rmtree(temporary)


def load_nouns_v2_partition(path: str | Path) -> NounsV2PartitionArtifact:
    """Strict-load all v2 files and reauthenticate the immutable parent stores."""
    manifest_path = Path(path)
    root = manifest_path.parent
    record = _canonical_object(manifest_path.read_bytes(), "nouns-v2 partition")
    supplied = record.pop("partition_sha256", None)
    expected_fields = {
        "base_train_story_count",
        "base_universe_story_count",
        "base_validation_story_count",
        "eos_token_id",
        "excluded_train_story_count",
        "excluded_validation_story_count",
        "files",
        "format",
        "manifest_sha256",
        "pad_token_id",
        "parent_files",
        "parent_partition_sha256",
        "root_probe_story_ids",
        "schema_version",
        "source_identity",
        "story_count",
        "task_ids",
        "tasks",
        "tokenizer_identity",
        "train_unique_story_count",
        "validation_unique_story_count",
    }
    if (
        set(record) != expected_fields
        or supplied != record_sha256(record)
        or root.name != supplied
        or record.get("format") != PARTITION_FORMAT
        or record.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("nouns-v2 partition content identity changed")
    files = _object(record.get("files"), "partition files")
    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if set(files) != actual_files:
        raise ValueError("nouns-v2 partition directory entries changed")
    for relative, raw in files.items():
        item = Path(relative)
        metadata = _object(raw, f"partition file {relative}")
        target = root / item
        if (
            item.is_absolute()
            or ".." in item.parts
            or not target.is_file()
            or target.stat().st_size != metadata.get("size_bytes")
            or _file_sha256(target) != metadata.get("sha256")
        ):
            raise ValueError(f"nouns-v2 partition file changed: {relative}")
    data_root = root.parents[1]
    manifest = load_manifest(data_root / "manifest.json")
    if manifest.manifest_sha256 != record.get("manifest_sha256"):
        raise ValueError("nouns-v2 partition manifest binding changed")
    parent_root = data_root.parent / "tinyworlds-nouns-v1" / "partitions" / str(
        record.get("parent_partition_sha256")
    )
    authenticated = authenticate_parent_manifest(parent_root)
    if authenticated != manifest:
        raise ValueError("nouns-v2 parent authentication changed")
    parent_record = _canonical_object(
        (parent_root / "partition.json").read_bytes(), "parent partition"
    )
    parent_files = _object(record.get("parent_files"), "partition parent files")
    if parent_files != {
        name: _object(parent_record["files"], "parent files")[name]
        for name in ("stories.bin", "stories.jsonl", "tokens.uint16")
    }:
        raise ValueError("nouns-v2 parent store binding changed")
    tasks = tuple(
        _task_summary_from_record(raw)
        for raw in _list(record.get("tasks"), "partition tasks")
    )
    root_probe_ids = tuple(
        _text(value, "root probe")
        for value in _list(record.get("root_probe_story_ids"), "root probes")
    )
    artifact = NounsV2PartitionArtifact(
        root=root,
        parent_root=parent_root,
        partition_sha256=str(supplied),
        manifest_sha256=_text(record.get("manifest_sha256"), "manifest hash"),
        parent_partition_sha256=_text(
            record.get("parent_partition_sha256"), "parent hash"
        ),
        source_identity=_object(record.get("source_identity"), "partition source"),
        tokenizer_identity=_object(
            record.get("tokenizer_identity"), "partition tokenizer"
        ),
        pad_token_id=_integer(record.get("pad_token_id"), "pad token ID"),
        eos_token_id=_integer(record.get("eos_token_id"), "EOS token ID"),
        story_count=_integer(record.get("story_count"), "story count"),
        train_unique_story_count=_integer(
            record.get("train_unique_story_count"), "train story count"
        ),
        validation_unique_story_count=_integer(
            record.get("validation_unique_story_count"), "validation story count"
        ),
        base_universe_story_count=_integer(
            record.get("base_universe_story_count"), "base universe count"
        ),
        base_train_story_count=_integer(
            record.get("base_train_story_count"), "base train count"
        ),
        base_validation_story_count=_integer(
            record.get("base_validation_story_count"), "base validation count"
        ),
        excluded_train_story_count=_integer(
            record.get("excluded_train_story_count"), "excluded train count"
        ),
        excluded_validation_story_count=_integer(
            record.get("excluded_validation_story_count"),
            "excluded validation count",
        ),
        root_probe_story_ids=root_probe_ids,
        task_ids=tuple(
            _text(value, "task ID")
            for value in _list(record.get("task_ids"), "task IDs")
        ),
        tasks=tasks,
    )
    _require_index_counts(artifact)
    return artifact


def _write_partition_tree(
    manifest: NounsV2Manifest,
    parent_root: Path,
    root: Path,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    state = _scan_parent(manifest, parent_root, progress)
    selections = _freeze_selections(manifest, state)
    _write_indexes_and_audits(manifest, parent_root, root, selections, progress)
    _write_summary_csvs(root, selections, manifest.parent_story_count)
    _write_audit_surfaces(root, manifest, selections)
    file_records = {
        path.relative_to(root).as_posix(): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    parent_record = _canonical_object(
        (parent_root / "partition.json").read_bytes(), "parent partition"
    )
    parent_files = _object(parent_record.get("files"), "parent files")
    return {
        "base_train_story_count": selections.base_train_count,
        "base_universe_story_count": BASE_UNIVERSE_STORY_COUNT,
        "base_validation_story_count": selections.base_validation_count,
        "eos_token_id": 50_256,
        "excluded_train_story_count": EXCLUDED_TRAIN_STORY_COUNT,
        "excluded_validation_story_count": EXCLUDED_VALIDATION_STORY_COUNT,
        "files": file_records,
        "format": PARTITION_FORMAT,
        "manifest_sha256": manifest.manifest_sha256,
        "pad_token_id": 50_256,
        "parent_files": {
            name: parent_files[name]
            for name in ("stories.bin", "stories.jsonl", "tokens.uint16")
        },
        "parent_partition_sha256": manifest.parent_partition_sha256,
        "root_probe_story_ids": list(selections.root_probe_ids),
        "schema_version": SCHEMA_VERSION,
        "source_identity": manifest.source_identity,
        "story_count": PARENT_STORY_COUNT,
        "task_ids": list(TASK_IDS),
        "tasks": [task.as_record() for task in selections.tasks],
        "tokenizer_identity": manifest.tokenizer_identity,
        "train_unique_story_count": TRAIN_UNIQUE_STORY_COUNT,
        "validation_unique_story_count": VALIDATION_UNIQUE_STORY_COUNT,
    }


def _scan_parent(
    manifest: NounsV2Manifest,
    parent_root: Path,
    progress: ProgressCallback | None,
) -> _ScanState:
    state = _ScanState(
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        0,
        0,
        [],
        {task_id: [] for task_id in TASK_IDS},
        {},
    )
    family_by_id = {family.task_id: family for family in manifest.task_families}
    for completed, row in enumerate(_parent_rows(parent_root), start=1):
        assignment = classify_selected_concepts(row.concept_ids)
        state.split_roles[(row.source_split, assignment.role)] += 1
        matched_forms = dict(row.matched_forms)
        for task_id in assignment.selected_concepts:
            state.raw_story_counts[(row.source_split, task_id)] += 1
            for form in matched_forms[task_id]:
                if form not in family_by_id[task_id].forms:
                    raise ValueError("parent matched a form outside the reviewed family")
                state.raw_form_counts[(row.source_split, task_id, form)] += 1
        if assignment.role == "task":
            task_id = str(assignment.task_id)
            state.retained_story_counts[(row.source_split, task_id)] += 1
            for form in matched_forms[task_id]:
                state.retained_form_counts[(row.source_split, task_id, form)] += 1
            if row.source_split == "train" and _probe_eligible(row.token_count):
                _push_lowest(
                    state.task_probe_heaps[task_id],
                    PROBE_STORY_COUNT,
                    f"task:{task_id}",
                    row,
                    PROBE_NAMESPACE,
                )
        if assignment.role == "base" and row.source_split == "train":
            held_out = _hash_bucket(
                TRAIN_HOLDOUT_NAMESPACE,
                row.story_id,
                BASE_VALIDATION_BUCKET_COUNT,
            ) == 0
            state.base_validation_count += int(held_out)
            state.base_train_count += int(not held_out)
            if not held_out and _probe_eligible(row.token_count):
                _push_lowest(
                    state.root_probe_heap,
                    PROBE_STORY_COUNT,
                    "root",
                    row,
                    PROBE_NAMESPACE,
                )
        example_task_ids = (
            assignment.selected_concepts
            if assignment.role == "excluded"
            else (assignment.task_id,)
            if assignment.role == "task"
            else ()
        )
        for task_id in example_task_ids:
            key = (row.source_split, assignment.role, str(task_id))
            heap = state.example_heaps.setdefault(key, [])
            _push_lowest(heap, 2, ":".join(key), row, EXAMPLE_NAMESPACE)
        if completed % 25_000 == 0:
            _emit(progress, "partition-scan", completed, manifest.parent_story_count)
    _emit(
        progress,
        "partition-scan",
        sum(state.split_roles.values()),
        manifest.parent_story_count,
    )
    return state


def _freeze_selections(
    manifest: NounsV2Manifest,
    state: _ScanState,
) -> _Selections:
    expected_roles = {
        ("train", "base"): BASE_UNIVERSE_STORY_COUNT,
        ("train", "task"): PURE_TASK_TRAIN_STORY_COUNT,
        ("train", "excluded"): EXCLUDED_TRAIN_STORY_COUNT,
        ("validation", "base"): 22_414,
        ("validation", "task"): PURE_TASK_VALIDATION_STORY_COUNT,
        ("validation", "excluded"): EXCLUDED_VALIDATION_STORY_COUNT,
    }
    if dict(state.split_roles) != expected_roles:
        raise ValueError(
            f"purified partition counts changed: {dict(state.split_roles)!r}"
        )
    expected_by_id = {
        task_id: (train_count, validation_count)
        for task_id, train_count, validation_count in EXPECTED_PURE_COUNTS
    }
    family_by_id = {family.task_id: family for family in manifest.task_families}
    tasks = tuple(
        NounsV2TaskSummary(
            task_id=task_id,
            forms=family_by_id[task_id].forms,
            raw_train_story_count=state.raw_story_counts[("train", task_id)],
            train_story_count=state.retained_story_counts[("train", task_id)],
            update_story_count=(
                state.retained_story_counts[("train", task_id)] - PROBE_STORY_COUNT
            ),
            raw_validation_story_count=state.raw_story_counts[
                ("validation", task_id)
            ],
            validation_story_count=state.retained_story_counts[
                ("validation", task_id)
            ],
            generation_story_count=state.retained_story_counts[
                ("validation", task_id)
            ],
            excluded_train_story_count=(
                state.raw_story_counts[("train", task_id)]
                - state.retained_story_counts[("train", task_id)]
            ),
            excluded_validation_story_count=(
                state.raw_story_counts[("validation", task_id)]
                - state.retained_story_counts[("validation", task_id)]
            ),
            probe_story_ids=_ordered_ids(state.task_probe_heaps[task_id]),
            raw_train_form_counts=tuple(
                (form, state.raw_form_counts[("train", task_id, form)])
                for form in family_by_id[task_id].forms
            ),
            retained_train_form_counts=tuple(
                (form, state.retained_form_counts[("train", task_id, form)])
                for form in family_by_id[task_id].forms
            ),
            raw_validation_form_counts=tuple(
                (form, state.raw_form_counts[("validation", task_id, form)])
                for form in family_by_id[task_id].forms
            ),
            retained_validation_form_counts=tuple(
                (
                    form,
                    state.retained_form_counts[("validation", task_id, form)],
                )
                for form in family_by_id[task_id].forms
            ),
        )
        for task_id in TASK_IDS
    )
    measured = tuple(
        (task.task_id, task.train_story_count, task.validation_story_count)
        for task in tasks
    )
    if measured != EXPECTED_PURE_COUNTS or any(
        task.train_story_count < MINIMUM_TASK_TRAIN_STORIES
        or task.validation_story_count < MINIMUM_TASK_VALIDATION_STORIES
        for task in tasks
    ):
        raise ValueError("purified task order, exact counts, or thresholds changed")
    if tuple(task.task_id for task in tasks) != tuple(
        sorted(TASK_IDS, key=lambda name: (-expected_by_id[name][0], name))
    ):
        raise ValueError("frozen tasks are not in descending purified train order")
    root_ids = _ordered_ids(state.root_probe_heap)
    if len(root_ids) != PROBE_STORY_COUNT or any(
        len(task.probe_story_ids) != PROBE_STORY_COUNT for task in tasks
    ):
        raise ValueError("a purified task or base lacks 36 context-fitting probes")
    example_ids = frozenset(
        row.story_id
        for heap in state.example_heaps.values()
        for _, _, row in heap
    )
    return _Selections(
        state.base_train_count,
        state.base_validation_count,
        root_ids,
        tasks,
        example_ids,
    )


def _write_indexes_and_audits(
    manifest: NounsV2Manifest,
    parent_root: Path,
    root: Path,
    selections: _Selections,
    progress: ProgressCallback | None,
) -> None:
    indexes = root / "indexes"
    audit = root / "audit"
    indexes.mkdir(parents=True)
    audit.mkdir(parents=True)
    names = (
        "base-train",
        "base-validation",
        "root-probes",
        *(
            f"task-{task_id}-{suffix}"
            for task_id in TASK_IDS
            for suffix in ("train", "validation", "probes", "generation")
        ),
    )
    streams = {name: (indexes / f"{name}.jsonl").open("wb") for name in names}
    excluded_streams = {
        split: (audit / f"excluded-{split}.jsonl").open("wb")
        for split in ("train", "validation")
    }
    example_stream = (audit / "examples.jsonl").open("wb")
    task_probes = {
        task.task_id: frozenset(task.probe_story_ids) for task in selections.tasks
    }
    root_probes = frozenset(selections.root_probe_ids)
    base_train_count = 0
    base_validation_count = 0
    story_source = (parent_root / "stories.bin").open("rb")
    try:
        for completed, row in enumerate(_parent_rows(parent_root), start=1):
            assignment = classify_selected_concepts(row.concept_ids)
            pointer = canonical_json_bytes(row.index_record)
            if assignment.role == "base" and row.source_split == "train":
                held_out = _hash_bucket(
                    TRAIN_HOLDOUT_NAMESPACE,
                    row.story_id,
                    BASE_VALIDATION_BUCKET_COUNT,
                ) == 0
                name = "base-validation" if held_out else "base-train"
                streams[name].write(pointer)
                base_validation_count += int(held_out)
                base_train_count += int(not held_out)
            if row.story_id in root_probes:
                streams["root-probes"].write(pointer)
            if assignment.role == "task":
                task_id = str(assignment.task_id)
                if row.source_split == "train":
                    suffix = "probes" if row.story_id in task_probes[task_id] else "train"
                    streams[f"task-{task_id}-{suffix}"].write(pointer)
                else:
                    streams[f"task-{task_id}-validation"].write(pointer)
                    if not _generation_eligible(row.token_count):
                        raise ValueError("a pure validation story cannot be midpoint routed")
                    streams[f"task-{task_id}-generation"].write(pointer)
            elif assignment.role == "excluded":
                forms = dict(row.matched_forms)
                core = {
                    "format": EXCLUSION_FORMAT,
                    "matched_forms": {
                        task_id: list(forms[task_id])
                        for task_id in assignment.selected_concepts
                    },
                    "parent_story_index": row.story_index,
                    "selected_concepts": list(assignment.selected_concepts),
                    "source_split": row.source_split,
                    "story_id": row.story_id,
                    "token_count": row.token_count,
                }
                excluded_streams[row.source_split].write(
                    canonical_json_bytes(
                        {**core, "exclusion_sha256": record_sha256(core)}
                    )
                )
            if row.story_id in selections.example_ids:
                story_source.seek(row.story_offset)
                text = story_source.read(row.byte_length).decode("utf-8", errors="strict")
                if sha256(text.encode("utf-8")).hexdigest() != row.story_id:
                    raise ValueError("audit example parent story bytes changed")
                example_stream.write(
                    canonical_json_bytes(
                        {
                            "assignment": assignment.role,
                            "parent_story_index": row.story_index,
                            "selected_concepts": list(assignment.selected_concepts),
                            "source_split": row.source_split,
                            "story": text,
                            "story_id": row.story_id,
                        }
                    )
                )
            if completed % 25_000 == 0:
                _emit(progress, "partition-write", completed, manifest.parent_story_count)
        if (
            base_train_count != selections.base_train_count
            or base_validation_count != selections.base_validation_count
        ):
            raise ValueError("base holdout changed between partition passes")
    finally:
        story_source.close()
        for stream in (*streams.values(), *excluded_streams.values(), example_stream):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    _emit(progress, "partition-write", manifest.parent_story_count, manifest.parent_story_count)


def _write_summary_csvs(
    root: Path,
    selections: _Selections,
    parent_story_count: int,
) -> None:
    base_fields = (
        "base_universe_story_count",
        "optimizer_train_story_count",
        "internal_validation_story_count",
        "original_training_story_count",
        "universe_share_of_original_training",
        "optimizer_share_of_original_training",
    )
    base_row = {
        "base_universe_story_count": BASE_UNIVERSE_STORY_COUNT,
        "optimizer_train_story_count": selections.base_train_count,
        "internal_validation_story_count": selections.base_validation_count,
        "original_training_story_count": TRAIN_UNIQUE_STORY_COUNT,
        "universe_share_of_original_training": (
            BASE_UNIVERSE_STORY_COUNT / TRAIN_UNIQUE_STORY_COUNT
        ),
        "optimizer_share_of_original_training": (
            selections.base_train_count / TRAIN_UNIQUE_STORY_COUNT
        ),
    }
    with (root / "base-selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=base_fields)
        writer.writeheader()
        writer.writerow(base_row)
        output.flush()
        os.fsync(output.fileno())
    fields = (
        "task_id",
        "raw_train_story_count",
        "train_story_count",
        "update_story_count",
        "excluded_train_story_count",
        "raw_validation_story_count",
        "validation_story_count",
        "excluded_validation_story_count",
    )
    with (root / "task-counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: getattr(task, field) for field in fields}
            for task in selections.tasks
        )
        output.flush()
        os.fsync(output.fileno())
    if parent_story_count != PARENT_STORY_COUNT:
        raise ValueError("parent story population changed before summary publication")


def _write_audit_surfaces(
    root: Path,
    manifest: NounsV2Manifest,
    selections: _Selections,
) -> None:
    core = {
        "base": {
            "internal_validation_story_count": selections.base_validation_count,
            "optimizer_train_story_count": selections.base_train_count,
            "optimizer_visible_share": selections.base_train_count
            / TRAIN_UNIQUE_STORY_COUNT,
            "original_training_story_count": TRAIN_UNIQUE_STORY_COUNT,
            "universe_share": BASE_UNIVERSE_STORY_COUNT / TRAIN_UNIQUE_STORY_COUNT,
            "universe_story_count": BASE_UNIVERSE_STORY_COUNT,
        },
        "excluded": {
            "train_story_count": EXCLUDED_TRAIN_STORY_COUNT,
            "validation_story_count": EXCLUDED_VALIDATION_STORY_COUNT,
        },
        "format": AUDIT_FORMAT,
        "manifest_sha256": manifest.manifest_sha256,
        "parent_story_count": manifest.parent_story_count,
        "schema_version": SCHEMA_VERSION,
        "tasks": [task.as_record() for task in selections.tasks],
    }
    audit = {**core, "audit_sha256": record_sha256(core)}
    _write_fsync(root / "audit" / "audit.json", canonical_json_bytes(audit))
    markdown = render_audit_markdown(audit)
    _write_fsync(root / "audit" / "audit.md", markdown.encode("utf-8"))
    _write_fsync(
        root / "audit" / "audit.html",
        render_audit_html(audit).encode("utf-8"),
    )


def render_audit_markdown(audit: dict[str, object]) -> str:
    """Render the standalone disjointness audit in Markdown."""
    base = _object(audit.get("base"), "audit base")
    excluded = _object(audit.get("excluded"), "audit excluded")
    tasks = tuple(
        _object(raw, "audit task") for raw in _list(audit.get("tasks"), "tasks")
    )
    lines = [
        "# TinyWorlds nouns-v2 disjoint partition audit",
        "",
        "A training story containing none of the 24 selected concept families belongs "
        "to the base. A story containing exactly one belongs only to that task. A "
        "story containing two or more is excluded from every model update.",
        "",
        f"Parent partition: `{PARENT_PARTITION_SHA256}` ({PARENT_STORY_COUNT:,} unique stories).",
        "",
        f"The clean base universe contains {int(base['universe_story_count']):,} stories "
        f"({float(base['universe_share']):.2%} of original training). The deterministic "
        f"2% holdout contains {int(base['internal_validation_story_count']):,}; "
        f"{int(base['optimizer_train_story_count']):,} remain optimizer-visible "
        f"({float(base['optimizer_visible_share']):.2%} of original training).",
        "",
        f"Excluded multi-task ledgers contain {int(excluded['train_story_count']):,} "
        f"training and {int(excluded['validation_story_count']):,} validation stories.",
        "",
        "| task | forms | raw train | pure train | excluded | updates | pure validation | probes |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {task['task_id']} | {', '.join(str(value) for value in task['forms'])} "
            f"| {int(task['raw_train_story_count']):,} "
            f"| {int(task['train_story_count']):,} "
            f"| {int(task['excluded_train_story_count']):,} "
            f"| {int(task['update_story_count']):,} "
            f"| {int(task['validation_story_count']):,} "
            f"| {len(_list(task['probe_story_ids'], 'probe IDs'))} |"
            for task in tasks
        ),
        "",
        "## Probe identities and form retention",
        "",
        *(
            f"### {task['task_id']}\n\n"
            f"Probes: `{ '`, `'.join(str(value) for value in task['probe_story_ids']) }`\n\n"
            f"Raw/pure train form counts: `{json.dumps(task['raw_train_form_counts'], sort_keys=True)}` / "
            f"`{json.dumps(task['retained_train_form_counts'], sort_keys=True)}`."
            for task in tasks
        ),
        "",
        "Exact example text and parent story indices are in `examples.jsonl`; every "
        "excluded record, its selected concepts, matched forms, and provenance are in "
        "the two `excluded-*.jsonl` ledgers.",
        "",
        f"Audit identity: `{audit['audit_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def render_audit_html(audit: dict[str, object]) -> str:
    """Render a standalone folding HTML projection of the audit."""
    tasks = tuple(
        _object(raw, "audit task") for raw in _list(audit.get("tasks"), "tasks")
    )
    cards = "".join(
        "<details><summary>"
        f"{escape(str(task['task_id']))}: {int(task['train_story_count']):,} pure train, "
        f"{int(task['validation_story_count']):,} validation</summary>"
        f"<p><b>Forms:</b> {escape(', '.join(str(value) for value in task['forms']))}</p>"
        f"<p><b>Raw → pure train:</b> {int(task['raw_train_story_count']):,} → "
        f"{int(task['train_story_count']):,}; <b>excluded:</b> "
        f"{int(task['excluded_train_story_count']):,}</p>"
        f"<p><b>Probe IDs:</b> <code>{escape(' '.join(str(value) for value in task['probe_story_ids']))}</code></p>"
        f"<pre>{escape(json.dumps({'raw_train_forms': task['raw_train_form_counts'], 'pure_train_forms': task['retained_train_form_counts'], 'raw_validation_forms': task['raw_validation_form_counts'], 'pure_validation_forms': task['retained_validation_form_counts']}, indent=2, sort_keys=True))}</pre>"
        "</details>"
        for task in tasks
    )
    base = _object(audit.get("base"), "audit base")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds nouns-v2 audit</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem;color:#182331}}details{{border:1px solid #ccd5df;border-radius:9px;padding:.8rem;margin:.7rem 0}}summary{{cursor:pointer;font-weight:650}}code,pre{{font-size:.78rem;word-break:break-all;white-space:pre-wrap}}.metric{{display:inline-block;padding:.4rem .7rem;background:#eef3f8;border-radius:999px;margin:.2rem}}</style></head><body>
<h1>TinyWorlds nouns-v2 disjoint partition audit</h1><p>Zero selected concepts means base; exactly one means that task only; two or more means permanent exclusion.</p>
<p><span class="metric">{int(base['universe_story_count']):,} clean base stories</span><span class="metric">{PURE_TASK_TRAIN_STORY_COUNT:,} pure task stories</span><span class="metric">{EXCLUDED_TRAIN_STORY_COUNT:,} excluded training stories</span><span class="metric">{PURE_TASK_VALIDATION_STORY_COUNT:,} validation pairs</span></p>
{cards}<details><summary>Exact authenticated audit JSON</summary><pre>{escape(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))}</pre></details></body></html>"""


def _parent_rows(parent_root: Path) -> Iterator[ParentStoryRow]:
    ledger = parent_root / "stories.jsonl"
    with ledger.open("rb") as source:
        for expected_index, line in enumerate(source):
            record = _canonical_object(line, "parent story ledger row")
            row = _parent_row(record)
            if row.story_index != expected_index:
                raise ValueError("parent story ledger order changed")
            yield row


def _fixture_documents(
    documents: Sequence[str],
    split: str,
    tokenizer: TextTokenizer,
    families: Sequence[NounConceptFamily],
) -> dict[str, ParentStoryRow]:
    rows: dict[str, ParentStoryRow] = {}
    for story_index, document in enumerate(documents):
        text = normalize_text(document)
        if not text:
            continue
        payload = text.encode("utf-8")
        story_id = sha256(payload).hexdigest()
        matches = match_selected_forms(text, families)
        token_ids = tokenizer.encode(text, add_eos=True)
        rows.setdefault(
            story_id,
            ParentStoryRow(
                story_id=story_id,
                source_split=split,
                story_index=story_index,
                story_offset=0,
                byte_length=len(payload),
                token_offset=0,
                token_count=len(token_ids),
                concept_ids=tuple(task_id for task_id, _ in matches),
                matched_forms=matches,
            ),
        )
    return rows


def _fixture_task(
    task_id: str,
    train_story_ids: tuple[str, ...],
    validation_story_ids: tuple[str, ...],
    train_by_id: dict[str, ParentStoryRow],
    probe_count: int,
) -> FixtureTaskPartition:
    heap: list[tuple[int, str, ParentStoryRow]] = []
    for story_id in train_story_ids:
        row = train_by_id[story_id]
        if _probe_eligible(row.token_count):
            _push_lowest(
                heap,
                probe_count,
                f"task:{task_id}",
                row,
                PROBE_NAMESPACE,
            )
    probe_ids = _ordered_ids(heap)
    if len(probe_ids) != probe_count:
        raise ValueError("fixture task lacks enough context-fitting probes")
    probe_set = frozenset(probe_ids)
    return FixtureTaskPartition(
        task_id=task_id,
        train_story_ids=train_story_ids,
        update_story_ids=tuple(
            story_id for story_id in train_story_ids if story_id not in probe_set
        ),
        validation_story_ids=validation_story_ids,
        probe_story_ids=probe_ids,
    )


def _parent_row(record: dict[str, object]) -> ParentStoryRow:
    matched = _object(record.get("matched_forms"), "parent matched forms")
    concept_ids = tuple(
        _text(value, "parent concept")
        for value in _list(record.get("concept_ids"), "parent concepts")
    )
    if set(matched) != set(concept_ids):
        raise ValueError("parent concepts and matched-form membership differ")
    story_id = _text(record.get("story_id"), "parent story ID")
    require_sha256(story_id, "parent story ID")
    return ParentStoryRow(
        story_id=story_id,
        source_split=_split(record.get("source_split")),
        story_index=_integer(record.get("story_index"), "parent story index"),
        story_offset=_integer(record.get("story_offset"), "parent story offset"),
        byte_length=_integer(record.get("byte_length"), "parent byte length"),
        token_offset=_integer(record.get("token_offset"), "parent token offset"),
        token_count=_integer(record.get("token_count"), "parent token count"),
        concept_ids=concept_ids,
        matched_forms=tuple(
            (
                concept_id,
                tuple(
                    _text(value, "parent matched form")
                    for value in _list(matched[concept_id], "parent matched forms")
                ),
            )
            for concept_id in concept_ids
        ),
    )


def _task_summary_from_record(value: object) -> NounsV2TaskSummary:
    record = _object(value, "partition task")
    forms = tuple(
        _text(form, "task form")
        for form in _list(record.get("forms"), "task forms")
    )
    form_counts = lambda key: tuple(
        (form, _integer(_object(record.get(key), key).get(form), f"{key} {form}"))
        for form in forms
    )
    return NounsV2TaskSummary(
        task_id=_text(record.get("task_id"), "task ID"),
        forms=forms,
        raw_train_story_count=_integer(
            record.get("raw_train_story_count"), "raw train count"
        ),
        train_story_count=_integer(record.get("train_story_count"), "train count"),
        update_story_count=_integer(record.get("update_story_count"), "update count"),
        raw_validation_story_count=_integer(
            record.get("raw_validation_story_count"), "raw validation count"
        ),
        validation_story_count=_integer(
            record.get("validation_story_count"), "validation count"
        ),
        generation_story_count=_integer(
            record.get("generation_story_count"), "generation count"
        ),
        excluded_train_story_count=_integer(
            record.get("excluded_train_story_count"), "excluded train count"
        ),
        excluded_validation_story_count=_integer(
            record.get("excluded_validation_story_count"),
            "excluded validation count",
        ),
        probe_story_ids=tuple(
            _text(story_id, "probe story")
            for story_id in _list(record.get("probe_story_ids"), "probe stories")
        ),
        raw_train_form_counts=form_counts("raw_train_form_counts"),
        retained_train_form_counts=form_counts("retained_train_form_counts"),
        raw_validation_form_counts=form_counts("raw_validation_form_counts"),
        retained_validation_form_counts=form_counts(
            "retained_validation_form_counts"
        ),
    )


def _require_index_counts(artifact: NounsV2PartitionArtifact) -> None:
    counts = {
        "base-train": artifact.base_train_story_count,
        "base-validation": artifact.base_validation_story_count,
        "root-probes": PROBE_STORY_COUNT,
        **{
            f"task-{task.task_id}-{suffix}": expected
            for task in artifact.tasks
            for suffix, expected in (
                ("train", task.update_story_count),
                ("validation", task.validation_story_count),
                ("probes", PROBE_STORY_COUNT),
                ("generation", task.generation_story_count),
            )
        },
    }
    if any(
        _line_count(artifact.root / "indexes" / f"{name}.jsonl") != expected
        for name, expected in counts.items()
    ):
        raise ValueError("nouns-v2 index and logical counts differ")
    if (
        _line_count(artifact.root / "audit" / "excluded-train.jsonl")
        != artifact.excluded_train_story_count
        or _line_count(artifact.root / "audit" / "excluded-validation.jsonl")
        != artifact.excluded_validation_story_count
    ):
        raise ValueError("nouns-v2 exclusion ledger counts differ")


def _push_lowest(
    heap: list[tuple[int, str, ParentStoryRow]],
    limit: int,
    label: str,
    row: ParentStoryRow,
    namespace: str,
) -> None:
    priority = int(
        sha256(
            f"{namespace}\0{label}\0{row.story_id}".encode("utf-8")
        ).hexdigest(),
        16,
    )
    candidate = (-priority, row.story_id, row)
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def _ordered_ids(heap: list[tuple[int, str, ParentStoryRow]]) -> tuple[str, ...]:
    return tuple(
        story_id
        for _, story_id, _ in sorted(heap, key=lambda value: (-value[0], value[1]))
    )


def _hash_bucket(namespace: str, story_id: str, bucket_count: int) -> int:
    return int(
        sha256(f"{namespace}\0{story_id}".encode("utf-8")).hexdigest(), 16
    ) % bucket_count


def _probe_eligible(token_count: int) -> bool:
    return 2 <= token_count <= 257


def _generation_eligible(token_count: int) -> bool:
    midpoint = token_count // 2
    return token_count >= 4 and 2 <= midpoint < MODEL_POSITION_LIMIT


def _words(text: str) -> tuple[str, ...]:
    import re

    return tuple(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _require_tree_equal(left: Path, right: Path) -> None:
    left_files = {
        path.relative_to(left).as_posix(): _file_sha256(path)
        for path in left.rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right).as_posix(): _file_sha256(path)
        for path in right.rglob("*")
        if path.is_file()
    }
    if left_files != right_files:
        raise ValueError("existing nouns-v2 partition differs from reconstruction")


def _line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line)


def _canonical_object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must be a canonical JSON object")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


def _split(value: object) -> str:
    split = _text(value, "source split")
    if split not in ("train", "validation"):
        raise ValueError("source split must be train or validation")
    return split


def _emit(
    progress: ProgressCallback | None,
    phase: str,
    completed: int,
    total: int | None,
) -> None:
    if progress is not None:
        progress(phase, completed, total)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FixtureDisjointPartition",
    "FixtureTaskPartition",
    "ParentStoryRow",
    "StoryAssignment",
    "authenticate_parent_manifest",
    "build_nouns_v2_partition",
    "build_fixture_disjoint_partition",
    "classify_selected_concepts",
    "find_partition",
    "load_manifest",
    "load_fixture_disjoint_partition",
    "load_nouns_v2_partition",
    "match_selected_forms",
    "publish_manifest",
    "render_audit_html",
    "render_audit_markdown",
    "selected_families_from_review",
    "verify_byte_identical_rebuild",
]
