"""Deterministic fact-withholding partition construction and strict loading."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    HashedFile,
    ProgressCallback,
    ProgressEvent,
    SourceIdentity,
    TokenizerIdentity,
)
from apm.data.text.tinyworlds_p.normalization import normalize_story_identity
from apm.data.text.tinyworlds_q_semantic.catalog import (
    ValidationCatalogView,
    render_catalog_audit,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    PARTITION_FORMAT,
    SCHEMA_VERSION,
    ArtifactFile,
    ConceptDefinition,
    PartitionCount,
    QueryPartitionArtifact,
    SemanticFact,
    SemanticQueryCatalog,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.review import is_construction_group
from apm.data.text.tinyworlds_q_semantic.source import QueryStoryGroup


PARTITION_TREE_FORMAT = "tinyworlds-q-semantic-partition-tree-v1"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class QueryPartitionPreset:
    """Immutable fact-withholding, split, and construction gate settings."""

    public_seed: int = 0
    base_split_weights: tuple[int, int, int] = (96, 2, 2)
    node_split_weights: tuple[int, int] = (90, 10)
    minimum_training_groups_per_fact: int = 32
    minimum_base_lexical_groups_per_concept: int = 256

    def __post_init__(self) -> None:
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("partition seed must be nonnegative")
        if self.base_split_weights != (96, 2, 2):
            raise ValueError("base split must remain 96/2/2")
        if self.node_split_weights != (90, 10):
            raise ValueError("node split must remain 90/10")
        if self.minimum_training_groups_per_fact != 32:
            raise ValueError("every fact requires 32 non-construction training groups")
        if self.minimum_base_lexical_groups_per_concept != 256:
            raise ValueError("every concept requires 256 ordinary lexical base groups")

    def as_record(self) -> dict[str, object]:
        """Return the complete canonical partition choices."""
        return {
            "base_split_weights": list(self.base_split_weights),
            "minimum_base_lexical_groups_per_concept": (
                self.minimum_base_lexical_groups_per_concept
            ),
            "minimum_training_groups_per_fact": self.minimum_training_groups_per_fact,
            "node_split_weights": list(self.node_split_weights),
            "public_seed": self.public_seed,
        }


QUERY_PARTITION_PRESET = QueryPartitionPreset()


@dataclass(frozen=True, slots=True)
class StorySemanticMatch:
    """Story-level leakage labels plus authoritative same-sentence evidence."""

    mentioned_concept_ids: tuple[str, ...]
    triggered_fact_ids: tuple[str, ...]
    authoritative_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(
            type(values) is not tuple or len(set(values)) != len(values)
            for values in (
                self.mentioned_concept_ids,
                self.triggered_fact_ids,
                self.authoritative_fact_ids,
            )
        ):
            raise ValueError("semantic story match IDs must be unique tuples")
        if not set(self.authoritative_fact_ids).issubset(self.triggered_fact_ids):
            raise ValueError("authoritative fact matches must also be story triggers")


@dataclass(frozen=True, slots=True)
class _StorySemanticMatcher:
    """One catalog-scoped set of compiled conservative leakage matchers."""

    concepts: tuple[ConceptDefinition, ...]
    facts: tuple[SemanticFact, ...]
    concept_patterns: dict[str, tuple[re.Pattern[str], ...]]
    trigger_patterns: dict[str, tuple[re.Pattern[str], ...]]

    @classmethod
    def compile(
        cls,
        concepts: tuple[ConceptDefinition, ...],
        facts: tuple[SemanticFact, ...],
    ) -> _StorySemanticMatcher:
        return cls(
            concepts=concepts,
            facts=facts,
            concept_patterns={
                concept.concept_id: tuple(
                    _surface_pattern(form) for form in concept.surface_forms
                )
                for concept in concepts
            },
            trigger_patterns={
                fact.fact_id: tuple(
                    _surface_pattern(form) for form in fact.trigger_forms
                )
                for fact in facts
            },
        )

    def match(self, story_text: str) -> StorySemanticMatch:
        normalized_story = normalize_story_identity(story_text)
        mentioned = tuple(
            concept.concept_id
            for concept in self.concepts
            if any(
                pattern.search(normalized_story)
                for pattern in self.concept_patterns[concept.concept_id]
            )
        )
        mentioned_set = set(mentioned)
        triggered = tuple(
            fact.fact_id
            for fact in self.facts
            if fact.concept_id in mentioned_set
            and any(
                pattern.search(normalized_story)
                for pattern in self.trigger_patterns[fact.fact_id]
            )
        )
        if not triggered:
            return StorySemanticMatch(mentioned, (), ())
        triggered_set = set(triggered)
        normalized_sentences = tuple(
            normalize_story_identity(sentence)
            for sentence in _SENTENCE_BOUNDARY.split(story_text)
            if sentence.strip()
        )
        authoritative = tuple(
            fact.fact_id
            for fact in self.facts
            if fact.fact_id in triggered_set
            and any(
                any(
                    pattern.search(sentence)
                    for pattern in self.concept_patterns[fact.concept_id]
                )
                and any(
                    pattern.search(sentence)
                    for pattern in self.trigger_patterns[fact.fact_id]
                )
                for sentence in normalized_sentences
            )
        )
        return StorySemanticMatch(mentioned, triggered, authoritative)


@dataclass(frozen=True, slots=True)
class QueryPartitionAssignment:
    """One no-replacement duplicate-group assignment with leakage evidence."""

    group_sha256: str
    role: str
    concept_id: str | None
    split: str | None
    exclusion_reason: str | None
    mentioned_concept_ids: tuple[str, ...]
    triggered_fact_ids: tuple[str, ...]
    authoritative_fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.group_sha256, "query partition assignment")
        if self.role not in ("base", "node", "construction", "excluded"):
            raise ValueError("query assignment role is invalid")
        if self.role == "node":
            if self.concept_id is None or self.split not in ("train", "validation"):
                raise ValueError("node assignment requires concept and two-way split")
            if self.exclusion_reason is not None:
                raise ValueError("node assignments cannot have an exclusion reason")
        elif self.role == "base":
            if self.concept_id is not None or self.split not in (
                "train",
                "validation",
                "test",
            ):
                raise ValueError("base assignment requires only a three-way split")
            if self.exclusion_reason is not None:
                raise ValueError("base assignments cannot have an exclusion reason")
        elif self.role == "construction":
            if any(value is not None for value in (self.concept_id, self.split, self.exclusion_reason)):
                raise ValueError("construction assignments are permanently unassigned")
        elif (
            self.concept_id is not None
            or self.split is not None
            or type(self.exclusion_reason) is not str
            or not self.exclusion_reason
        ):
            raise ValueError("excluded assignments require only one reason")

    def as_record(self) -> dict[str, object]:
        """Return one canonical streamable assignment."""
        return {
            "authoritative_fact_ids": list(self.authoritative_fact_ids),
            "concept_id": self.concept_id,
            "exclusion_reason": self.exclusion_reason,
            "group_sha256": self.group_sha256,
            "mentioned_concept_ids": list(self.mentioned_concept_ids),
            "role": self.role,
            "split": self.split,
            "triggered_fact_ids": list(self.triggered_fact_ids),
        }


def match_story_semantics(
    story_text: str,
    concepts: tuple[ConceptDefinition, ...],
    facts: tuple[SemanticFact, ...],
) -> StorySemanticMatch:
    """Match leakage at story level and authority only within one sentence."""
    return _StorySemanticMatcher.compile(concepts, facts).match(story_text)


def assign_story_group(
    group: QueryStoryGroup,
    concepts: tuple[ConceptDefinition, ...],
    facts: tuple[SemanticFact, ...],
    preset: QueryPartitionPreset = QUERY_PARTITION_PRESET,
) -> QueryPartitionAssignment:
    """Assign one complete duplicate group without exposing construction evidence."""
    return _assign_story_group(
        group,
        _StorySemanticMatcher.compile(concepts, facts),
        preset,
    )


def _assign_story_group(
    group: QueryStoryGroup,
    matcher: _StorySemanticMatcher,
    preset: QueryPartitionPreset,
) -> QueryPartitionAssignment:
    match = matcher.match(group.normalized_text)
    if is_construction_group(group.normalized_story_sha256):
        return QueryPartitionAssignment(
            group.normalized_story_sha256,
            "construction",
            None,
            None,
            None,
            match.mentioned_concept_ids,
            match.triggered_fact_ids,
            match.authoritative_fact_ids,
        )
    if match.triggered_fact_ids and len(match.mentioned_concept_ids) > 1:
        return QueryPartitionAssignment(
            group.normalized_story_sha256,
            "excluded",
            None,
            None,
            "multi-concept-fact-bearing",
            match.mentioned_concept_ids,
            match.triggered_fact_ids,
            match.authoritative_fact_ids,
        )
    if match.triggered_fact_ids:
        if len(match.mentioned_concept_ids) != 1:
            raise AssertionError("fact-bearing groups must mention at least one concept")
        concept_id = match.mentioned_concept_ids[0]
        split = _weighted_split(
            group.normalized_story_sha256,
            f"node:{concept_id}",
            ("train", "validation"),
            preset.node_split_weights,
            preset.public_seed,
        )
        return QueryPartitionAssignment(
            group.normalized_story_sha256,
            "node",
            concept_id,
            split,
            None,
            match.mentioned_concept_ids,
            match.triggered_fact_ids,
            match.authoritative_fact_ids,
        )
    split = _weighted_split(
        group.normalized_story_sha256,
        "base",
        ("train", "validation", "test"),
        preset.base_split_weights,
        preset.public_seed,
    )
    return QueryPartitionAssignment(
        group.normalized_story_sha256,
        "base",
        None,
        split,
        None,
        match.mentioned_concept_ids,
        match.triggered_fact_ids,
        match.authoritative_fact_ids,
    )


def build_query_partition(
    groups: Iterable[QueryStoryGroup],
    catalog: SemanticQueryCatalog,
    output_root: str | Path,
    *,
    pad_token_id: int,
    eos_token_id: int,
    preset: QueryPartitionPreset = QUERY_PARTITION_PRESET,
    progress: ProgressCallback | None = None,
    total_group_count: int | None = None,
) -> QueryPartitionArtifact:
    """Stream exact groups into an authenticated query-native partition tree."""
    if any(
        type(value) is not int
        or not 0 <= value < catalog.tokenizer_identity.vocab_size
        for value in (pad_token_id, eos_token_id)
    ):
        raise ValueError("query partition special token IDs lie outside the vocabulary")
    if total_group_count is not None and (
        type(total_group_count) is not int or total_group_count <= 0
    ):
        raise ValueError("query partition total_group_count must be positive")
    output = Path(output_root)
    work_root = output / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="query-partition-", dir=work_root))
    print(f"TinyWorlds-Q partition temporary artifacts: {staging}", flush=True)
    fact_order = {fact.fact_id: index for index, fact in enumerate(catalog.facts)}
    semantic_matcher = _StorySemanticMatcher.compile(
        catalog.concepts,
        catalog.facts,
    )
    count_values: dict[tuple[str, str | None, str | None], list[int]] = {}
    training_support = {fact.fact_id: 0 for fact in catalog.facts}
    lexical_exposure = {concept.concept_id: 0 for concept in catalog.concepts}
    construction_seen: set[str] = set()
    leakage_exclusions: list[QueryPartitionAssignment] = []
    previous_group_sha256: str | None = None
    processed_group_count = 0
    progress_interval = max(
        1,
        min(100_000, (total_group_count or 1_000_000) // 100),
    )
    story_offset = 0
    token_offset = 0
    try:
        _emit_progress(
            progress,
            "splits",
            0,
            total_group_count,
            "assigning query-native duplicate groups without replacement",
        )
        indexes_root = staging / "indexes"
        indexes_root.mkdir()
        index_paths = {
            ("base", None, split): indexes_root / f"base-{split}.jsonl"
            for split in ("train", "validation", "test")
        } | {
            ("node", concept.concept_id, split): (
                indexes_root / f"node-{concept.concept_id}-{split}.jsonl"
            )
            for concept in catalog.concepts
            for split in ("train", "validation")
        }
        with ExitStack() as stack:
            assignments = stack.enter_context(
                (staging / "assignments.jsonl").open("wb")
            )
            exclusions = stack.enter_context(
                (staging / "leakage-exclusions.jsonl").open("wb")
            )
            documents = stack.enter_context((staging / "documents.jsonl").open("wb"))
            story_stream = stack.enter_context((staging / "stories.bin").open("wb"))
            token_stream = stack.enter_context((staging / "tokens.uint16").open("wb"))
            index_streams = {
                key: stack.enter_context(path.open("wb"))
                for key, path in index_paths.items()
            }
            for group in groups:
                if type(group) is not QueryStoryGroup:
                    raise TypeError("partition groups must contain QueryStoryGroup values")
                if previous_group_sha256 is not None and group.normalized_story_sha256 <= previous_group_sha256:
                    raise ValueError("query source groups must be strictly hash sorted")
                previous_group_sha256 = group.normalized_story_sha256
                processed_group_count += 1
                assignment = _assign_story_group(
                    group,
                    semantic_matcher,
                    preset,
                )
                assignments.write(canonical_json_bytes(assignment.as_record()))
                if assignment.role == "excluded":
                    exclusions.write(canonical_json_bytes(assignment.as_record()))
                    leakage_exclusions.append(assignment)
                key = (assignment.role, assignment.concept_id, assignment.split)
                totals = count_values.setdefault(key, [0, 0, 0])
                totals[0] += 1
                totals[1] += len(group.occurrences)
                totals[2] += group.token_count
                if assignment.role == "construction":
                    construction_seen.add(group.normalized_story_sha256)
                if assignment.role == "node" and assignment.split == "train":
                    for fact_id in assignment.authoritative_fact_ids:
                        training_support[fact_id] += 1
                if assignment.role == "base" and assignment.split == "train":
                    for concept_id in assignment.mentioned_concept_ids:
                        lexical_exposure[concept_id] += 1
                for occurrence in group.occurrences:
                    story_stream.write(occurrence.story_bytes)
                    token_bytes = np.asarray(occurrence.token_ids, dtype="<u2").tobytes()
                    token_stream.write(token_bytes)
                    document = {
                        "content_sha256": occurrence.content_sha256,
                        "group_sha256": group.normalized_story_sha256,
                        "record_id": occurrence.record_id,
                        "role": assignment.role,
                        "source": occurrence.source,
                        "source_index": occurrence.source_index,
                        "source_member": occurrence.source_member,
                        "split": assignment.split,
                        "story_bytes": len(occurrence.story_bytes),
                        "story_offset": story_offset,
                        "story_sha256": occurrence.story_sha256,
                        "token_count": len(occurrence.token_ids),
                        "token_offset": token_offset,
                        "world": assignment.concept_id,
                    }
                    document_bytes = canonical_json_bytes(document)
                    documents.write(document_bytes)
                    if assignment.role in ("base", "node"):
                        index_streams[
                            (assignment.role, assignment.concept_id, assignment.split)
                        ].write(document_bytes)
                    story_offset += len(occurrence.story_bytes)
                    token_offset += len(occurrence.token_ids)
                if processed_group_count % progress_interval == 0:
                    _emit_progress(
                        progress,
                        "splits",
                        processed_group_count,
                        total_group_count,
                        "withholding registered facts and streaming exact payloads",
                    )
            for stream in (
                assignments,
                exclusions,
                documents,
                story_stream,
                token_stream,
                *index_streams.values(),
            ):
                stream.flush()
                os.fsync(stream.fileno())
        if (
            total_group_count is not None
            and processed_group_count != total_group_count
        ):
            raise ValueError(
                "query source group count differs from the authenticated archive audit"
            )
        _emit_progress(
            progress,
            "splits",
            processed_group_count,
            total_group_count,
            "duplicate-group assignment and exact payload streams completed",
        )
        _require_partition_gates(
            catalog,
            training_support,
            lexical_exposure,
            construction_seen,
            preset,
        )
        counts = _partition_counts(count_values, catalog.concepts)
        audit_markdown = (
            render_catalog_audit(catalog, include_sealed=False)
            + "\n"
            + render_partition_audit(
                catalog,
                counts,
                training_support,
                lexical_exposure,
                tuple(leakage_exclusions),
            )
        )
        (staging / "audit.md").write_text(audit_markdown, encoding="utf-8", newline="\n")
        (staging / "audit.html").write_text(
            _standalone_html(catalog.catalog_sha256, audit_markdown),
            encoding="utf-8",
            newline="\n",
        )
        (staging / "preset.json").write_bytes(canonical_json_bytes(preset.as_record()))
        (staging / "sources.json").write_bytes(
            canonical_json_bytes(
                {
                    "archive": catalog.archive_identity.as_record(),
                    "catalog_sha256": catalog.catalog_sha256,
                    "normalization": catalog.normalization.as_record(),
                    "tokenizer": catalog.tokenizer_identity.as_record(),
                }
            )
        )
        files = tuple(
            _artifact_file(staging, path.relative_to(staging).as_posix())
            for path in sorted(item for item in staging.rglob("*") if item.is_file())
        )
        core = {
            "benchmark_id": BENCHMARK_ID,
            "catalog_sha256": catalog.catalog_sha256,
            "concept_ids": list(catalog.concept_ids),
            "counts": [count.as_record() for count in counts],
            "files": [item.as_record() for item in files],
            "format": PARTITION_FORMAT,
            "lexical_exposure": lexical_exposure,
            "preset": preset.as_record(),
            "schema_version": SCHEMA_VERSION,
            "special_tokens": {
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
            },
            "sources": {
                "archive": catalog.archive_identity.as_record(),
                "tokenizer": catalog.tokenizer_identity.as_record(),
            },
            "training_fact_support": training_support,
        }
        partition_sha256 = record_sha256(core)
        manifest = {
            **core,
            "partition_sha256": partition_sha256,
            "tree_format": PARTITION_TREE_FORMAT,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        _emit_progress(
            progress,
            "publish",
            1,
            2,
            "content identity fixed; performing strict full-tree reload",
        )
        destination = output / "partitions" / partition_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            loaded = load_query_partition(destination, catalog)
            _remove_staging(staging)
            _emit_progress(
                progress,
                "publish",
                2,
                2,
                "existing byte-identical query partition authenticated",
            )
            return loaded
        os.replace(staging, destination)
        loaded = load_query_partition(destination, catalog)
        _emit_progress(
            progress,
            "publish",
            2,
            2,
            "new query partition passed strict full-tree reload",
        )
        return loaded
    except BaseException:
        _remove_staging(staging)
        raise


def load_query_partition(
    directory: str | Path,
    catalog: SemanticQueryCatalog | ValidationCatalogView,
) -> QueryPartitionArtifact:
    """Rehash every byte and reconstruct core leakage/count invariants."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _canonical_json_record(manifest_bytes, "partition manifest")
    if (
        manifest.get("tree_format") != PARTITION_TREE_FORMAT
        or manifest.get("format") != PARTITION_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("benchmark_id") != BENCHMARK_ID
        or manifest.get("catalog_sha256") != catalog.catalog_sha256
        or tuple(_string_list(manifest, "concept_ids")) != catalog.concept_ids
    ):
        raise ValueError("query partition manifest or catalog binding changed")
    file_records = tuple(
        _decode_artifact_file(item) for item in _record_list(manifest, "files")
    )
    expected_entries = {"manifest.json", *(item.relative_path for item in file_records)}
    actual_entries = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        parent.as_posix()
        for entry in expected_entries
        for parent in Path(entry).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if (
        actual_entries != expected_entries
        or actual_directories != expected_directories
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError("query partition tree entries changed")
    for item in file_records:
        path = root / item.relative_path
        if path.stat().st_size != item.size_bytes or _file_sha256(path) != item.sha256:
            raise ValueError(f"query partition file changed: {item.relative_path}")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in ("partition_sha256", "tree_format")
    }
    partition_sha256 = record_sha256(core)
    if (
        manifest.get("partition_sha256") != partition_sha256
        or root.name != partition_sha256
    ):
        raise ValueError("query partition content identity changed")
    counts = tuple(
        _decode_partition_count(item) for item in _record_list(manifest, "counts")
    )
    if _mapping(manifest, "preset") != QUERY_PARTITION_PRESET.as_record():
        raise ValueError("query partition preset changed")
    training_support, lexical_exposure, construction_seen = _validate_assignment_ledger(
        root / "assignments.jsonl",
        root / "leakage-exclusions.jsonl",
        counts,
        catalog.concepts,
        catalog.facts,
    )
    if training_support != _integer_mapping(
        manifest,
        "training_fact_support",
        tuple(fact.fact_id for fact in catalog.facts),
    ) or lexical_exposure != _integer_mapping(
        manifest,
        "lexical_exposure",
        catalog.concept_ids,
    ):
        raise ValueError("query partition semantic gate counts changed")
    _require_partition_gates(
        catalog,
        training_support,
        lexical_exposure,
        construction_seen,
        QUERY_PARTITION_PRESET,
    )
    _validate_document_ledger(root, counts, catalog.concept_ids)
    source_record = _mapping(manifest, "sources")
    special_tokens = _mapping(manifest, "special_tokens")
    pad_token_id = _integer(special_tokens, "pad_token_id")
    eos_token_id = _integer(special_tokens, "eos_token_id")
    archive_identity = _decode_source(_mapping(source_record, "archive"))
    tokenizer_identity = _decode_tokenizer(_mapping(source_record, "tokenizer"))
    if (
        archive_identity != catalog.archive_identity
        or tokenizer_identity != catalog.tokenizer_identity
    ):
        raise ValueError("query partition source identities changed")
    return QueryPartitionArtifact(
        root=root.resolve(),
        partition_sha256=partition_sha256,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        catalog_sha256=catalog.catalog_sha256,
        archive_identity=archive_identity,
        tokenizer_identity=tokenizer_identity,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        concept_ids=catalog.concept_ids,
        counts=counts,
        files=file_records,
    )


def iter_partition_documents(
    artifact: QueryPartitionArtifact,
    *,
    role: str,
    split: str,
    concept_id: str | None = None,
) -> Iterator[tuple[dict[str, object], bytes, tuple[int, ...]]]:
    """Stream authenticated model inputs without materializing a partition split."""
    if role not in ("base", "node"):
        raise ValueError("model input role must be base or node")
    if role == "node" and concept_id not in artifact.concept_ids:
        raise ValueError("node document stream requires an active concept")
    index_path = artifact.root / "indexes" / _index_filename(
        role,
        split,
        concept_id,
    )
    with (
        index_path.open("rb") as ledger,
        (artifact.root / "stories.bin").open("rb") as stories,
        (artifact.root / "tokens.uint16").open("rb") as tokens,
    ):
        for line in ledger:
            record = _canonical_json_record(line, "partition document")
            if (
                record.get("role") != role
                or record.get("split") != split
                or (role == "node" and record.get("world") != concept_id)
            ):
                continue
            story_offset = _integer(record, "story_offset")
            story_size = _integer(record, "story_bytes")
            token_offset = _integer(record, "token_offset")
            token_count = _integer(record, "token_count")
            stories.seek(story_offset)
            story = stories.read(story_size)
            tokens.seek(token_offset * 2)
            token_payload = tokens.read(token_count * 2)
            if sha256(story).hexdigest() != _string(record, "story_sha256"):
                raise ValueError("partition story payload changed")
            token_ids = tuple(
                int(value) for value in np.frombuffer(token_payload, dtype="<u2")
            )
            if len(token_ids) != token_count:
                raise ValueError("partition token payload is truncated")
            yield record, story, token_ids


def tree_sha256(directory: str | Path) -> str:
    """Hash a complete artifact tree by relative path and exact bytes."""
    root = Path(directory)
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def render_partition_audit(
    catalog: SemanticQueryCatalog,
    counts: tuple[PartitionCount, ...],
    training_support: dict[str, int],
    lexical_exposure: dict[str, int],
    leakage_exclusions: tuple[QueryPartitionAssignment, ...] = (),
) -> str:
    """Render world mass, base retention, leakage, and fact-support evidence."""
    base_counts = tuple(count for count in counts if count.role == "base")
    base_groups = sum(count.group_count for count in base_counts)
    base_tokens = sum(count.token_count for count in base_counts)
    lines = [
        "# TinyWorlds-Q partition audit",
        "",
        f"Catalog: `{catalog.catalog_sha256}`",
        "",
        f"Retained base groups: {base_groups}",
        f"Retained base token mass: {base_tokens}",
        "",
        "## Split and world masses",
        "",
        "| Role | Concept | Split | Groups | Occurrences | Tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {count.role} | {count.concept_id or '—'} | {count.split or '—'} | "
        f"{count.group_count} | {count.occurrence_count} | {count.token_count} |"
        for count in counts
    )
    lines.extend(("", "## Fact-bearing training support", ""))
    lines.extend(
        f"- `{fact.fact_id}` ({fact.concept_id}): {training_support[fact.fact_id]} authoritative groups"
        for fact in catalog.facts
    )
    lines.extend(("", "## Ordinary lexical exposure", ""))
    lines.extend(
        f"- `{concept.concept_id}`: {lexical_exposure[concept.concept_id]} non-fact base groups"
        for concept in catalog.concepts
    )
    lines.extend(("", "## Leakage exclusions", ""))
    if leakage_exclusions:
        lines.extend(
            f"- `{assignment.group_sha256}` — {assignment.exclusion_reason}; "
            f"concepts `{', '.join(assignment.mentioned_concept_ids)}`; "
            f"facts `{', '.join(assignment.triggered_fact_ids)}`"
            for assignment in leakage_exclusions
        )
    else:
        lines.append("- No multi-concept fact-bearing groups were found.")
    lines.extend(
        (
            "",
            "Construction groups, every fact-bearing multi-concept group, and their exact "
            "assignment evidence are retained in `assignments.jsonl`; exclusions are also "
            "streamed in `leakage-exclusions.jsonl`. Neither enters model inputs.",
            "",
        )
    )
    return "\n".join(lines)


def _require_partition_gates(
    catalog: SemanticQueryCatalog,
    training_support: dict[str, int],
    lexical_exposure: dict[str, int],
    construction_seen: set[str],
    preset: QueryPartitionPreset,
) -> None:
    missing_construction = tuple(
        (fact.fact_id, group_sha256)
        for fact in catalog.facts
        for group_sha256 in fact.supporting_story_groups
        if group_sha256 not in construction_seen
    )
    if missing_construction:
        raise ValueError(
            f"catalog construction evidence is absent from the archive: {missing_construction[:3]}"
        )
    insufficient_facts = {
        fact_id: count
        for fact_id, count in training_support.items()
        if count < preset.minimum_training_groups_per_fact
    }
    if insufficient_facts:
        raise ValueError(
            f"facts lack 32 authoritative non-construction training groups: {insufficient_facts}"
        )
    insufficient_exposure = {
        concept_id: count
        for concept_id, count in lexical_exposure.items()
        if count < preset.minimum_base_lexical_groups_per_concept
    }
    if insufficient_exposure:
        raise ValueError(
            f"concepts lack 256 ordinary lexical base groups: {insufficient_exposure}"
        )


def _partition_counts(
    values: dict[tuple[str, str | None, str | None], list[int]],
    concepts: tuple[ConceptDefinition, ...],
) -> tuple[PartitionCount, ...]:
    keys = (
        *(("base", None, split) for split in ("train", "validation", "test")),
        *(
            ("node", concept.concept_id, split)
            for concept in concepts
            for split in ("train", "validation")
        ),
        ("construction", None, None),
        ("excluded", None, None),
    )
    return tuple(
        PartitionCount(role, concept_id, split, *values.get(key, (0, 0, 0)))
        for key in keys
        for role, concept_id, split in (key,)
    )


def _weighted_split(
    group_sha256: str,
    namespace: str,
    labels: tuple[str, ...],
    weights: tuple[int, ...],
    seed: int,
) -> str:
    digest = sha256(
        f"{BENCHMARK_ID}\0{seed}\0{namespace}\0{group_sha256}".encode("utf-8")
    ).digest()
    position = int.from_bytes(digest, "big") % sum(weights)
    cumulative = 0
    for label, weight in zip(labels, weights):
        cumulative += weight
        if position < cumulative:
            return label
    raise AssertionError("weighted split did not cover its hash bucket")


def _surface_pattern(surface: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")


def _artifact_file(root: Path, name: str) -> ArtifactFile:
    path = root / name
    return ArtifactFile(name, path.stat().st_size, _file_sha256(path))


def _index_filename(role: str, split: str, concept_id: str | None) -> str:
    if role == "base" and concept_id is None and split in (
        "train",
        "validation",
        "test",
    ):
        return f"base-{split}.jsonl"
    if role == "node" and concept_id is not None and split in (
        "train",
        "validation",
    ):
        return f"node-{concept_id}-{split}.jsonl"
    raise ValueError("partition document selector is invalid")


def _decode_artifact_file(record: dict[str, object]) -> ArtifactFile:
    return ArtifactFile(
        _string(record, "relative_path"),
        _integer(record, "size_bytes"),
        _string(record, "sha256"),
    )


def _decode_partition_count(record: dict[str, object]) -> PartitionCount:
    concept = record.get("concept_id")
    split = record.get("split")
    if concept is not None and type(concept) is not str:
        raise ValueError("partition count concept must be text or null")
    if split is not None and type(split) is not str:
        raise ValueError("partition count split must be text or null")
    return PartitionCount(
        role=_string(record, "role"),  # type: ignore[arg-type]
        concept_id=concept,
        split=split,  # type: ignore[arg-type]
        group_count=_integer(record, "group_count"),
        occurrence_count=_integer(record, "occurrence_count"),
        token_count=_integer(record, "token_count"),
    )


def _validate_assignment_ledger(
    path: Path,
    exclusions_path: Path,
    counts: tuple[PartitionCount, ...],
    concepts: tuple[ConceptDefinition, ...],
    facts: tuple[SemanticFact, ...],
) -> tuple[dict[str, int], dict[str, int], set[str]]:
    observed: dict[tuple[str, str | None, str | None], int] = {}
    training_support = {fact.fact_id: 0 for fact in facts}
    lexical_exposure = {concept.concept_id: 0 for concept in concepts}
    construction_seen: set[str] = set()
    exclusion_digest = sha256()
    exclusion_size = 0
    concept_ids = tuple(concept.concept_id for concept in concepts)
    fact_by_id = {fact.fact_id: fact for fact in facts}
    previous: str | None = None
    with path.open("rb") as stream:
        for line in stream:
            record = _canonical_json_record(line, "partition assignment")
            group_sha256 = _string(record, "group_sha256")
            role = _string(record, "role")
            concept = record.get("concept_id")
            split = record.get("split")
            exclusion_reason = record.get("exclusion_reason")
            mentioned = _string_list(record, "mentioned_concept_ids")
            triggered = _string_list(record, "triggered_fact_ids")
            authoritative = _string_list(record, "authoritative_fact_ids")
            if previous is not None and group_sha256 <= previous:
                raise ValueError("partition assignment ledger is not strictly sorted")
            previous = group_sha256
            if (role == "construction") != is_construction_group(group_sha256):
                raise ValueError("construction group escaped or entered model assignment")
            if len(set(mentioned)) != len(mentioned) or any(
                item not in concept_ids for item in mentioned
            ):
                raise ValueError("partition assignment mentions unknown concepts")
            if len(set(triggered)) != len(triggered) or any(
                item not in fact_by_id for item in triggered
            ):
                raise ValueError("partition assignment triggers unknown facts")
            if len(set(authoritative)) != len(authoritative) or not set(
                authoritative
            ).issubset(triggered):
                raise ValueError("partition authoritative facts changed")
            if any(fact_by_id[item].concept_id not in mentioned for item in triggered):
                raise ValueError("partition fact trigger lacks its target concept")
            if role == "base":
                if (
                    concept is not None
                    or split not in ("train", "validation", "test")
                    or exclusion_reason is not None
                    or triggered
                    or authoritative
                    or split
                    != _weighted_split(
                        group_sha256,
                        "base",
                        ("train", "validation", "test"),
                        QUERY_PARTITION_PRESET.base_split_weights,
                        QUERY_PARTITION_PRESET.public_seed,
                    )
                ):
                    raise ValueError("partition base assignment violates fact withholding")
                if split == "train":
                    for concept_id in mentioned:
                        lexical_exposure[concept_id] += 1
            elif role == "node":
                if (
                    concept not in concept_ids
                    or mentioned != (concept,)
                    or not triggered
                    or any(fact_by_id[item].concept_id != concept for item in triggered)
                    or split not in ("train", "validation")
                    or exclusion_reason is not None
                    or split
                    != _weighted_split(
                        group_sha256,
                        f"node:{concept}",
                        ("train", "validation"),
                        QUERY_PARTITION_PRESET.node_split_weights,
                        QUERY_PARTITION_PRESET.public_seed,
                    )
                ):
                    raise ValueError("partition node assignment violates fact withholding")
                if split == "train":
                    for fact_id in authoritative:
                        training_support[fact_id] += 1
            elif role == "construction":
                if any(
                    value is not None for value in (concept, split, exclusion_reason)
                ):
                    raise ValueError("construction assignment entered a model role")
                construction_seen.add(group_sha256)
            elif role == "excluded":
                if (
                    concept is not None
                    or split is not None
                    or exclusion_reason != "multi-concept-fact-bearing"
                    or not triggered
                    or len(mentioned) <= 1
                ):
                    raise ValueError("partition exclusion semantics changed")
                exclusion_digest.update(line)
                exclusion_size += len(line)
            else:
                raise ValueError("partition assignment role changed")
            key = (role, concept if type(concept) is str else None, split if type(split) is str else None)
            observed[key] = observed.get(key, 0) + 1
    expected = {
        (count.role, count.concept_id, count.split): count.group_count for count in counts
    }
    if observed != {key: value for key, value in expected.items() if value > 0}:
        raise ValueError("partition assignment counts changed")
    if (
        exclusions_path.stat().st_size != exclusion_size
        or _file_sha256(exclusions_path) != exclusion_digest.hexdigest()
    ):
        raise ValueError("partition leakage exclusion ledger changed")
    return training_support, lexical_exposure, construction_seen


def _validate_document_ledger(
    root: Path,
    counts: tuple[PartitionCount, ...],
    concept_ids: tuple[str, ...],
) -> None:
    observed: dict[tuple[str, str | None, str | None], list[int]] = {}
    expected_story_offset = 0
    expected_token_offset = 0
    with (root / "documents.jsonl").open("rb") as stream:
        for line in stream:
            record = _canonical_json_record(line, "partition document")
            role = _string(record, "role")
            concept = record.get("world")
            split = record.get("split")
            if role not in ("base", "node", "construction", "excluded"):
                raise ValueError("partition document role changed")
            if concept is not None and (
                type(concept) is not str or concept not in concept_ids
            ):
                raise ValueError("partition document world changed")
            if split is not None and type(split) is not str:
                raise ValueError("partition document split changed")
            story_offset = _integer(record, "story_offset")
            story_bytes = _integer(record, "story_bytes")
            token_offset = _integer(record, "token_offset")
            token_count = _integer(record, "token_count")
            if (
                story_bytes <= 0
                or token_count <= 0
                or story_offset != expected_story_offset
                or token_offset != expected_token_offset
            ):
                raise ValueError("partition document payload offsets changed")
            require_sha256(_string(record, "group_sha256"), "document group")
            require_sha256(_string(record, "story_sha256"), "document story")
            expected_story_offset += story_bytes
            expected_token_offset += token_count
            key = (
                role,
                concept if type(concept) is str else None,
                split if type(split) is str else None,
            )
            values = observed.setdefault(key, [0, 0])
            values[0] += 1
            values[1] += token_count
    if (root / "stories.bin").stat().st_size != expected_story_offset or (
        root / "tokens.uint16"
    ).stat().st_size != expected_token_offset * 2:
        raise ValueError("partition document payload length changed")
    expected = {
        (count.role, count.concept_id, count.split): (
            count.occurrence_count,
            count.token_count,
        )
        for count in counts
        if count.occurrence_count > 0
    }
    if {key: tuple(value) for key, value in observed.items()} != expected:
        raise ValueError("partition document occurrence or token counts changed")


def _integer_mapping(
    record: dict[str, object],
    field: str,
    expected_keys: tuple[str, ...],
) -> dict[str, int]:
    value = _mapping(record, field)
    if set(value) != set(expected_keys) or any(
        type(item) is not int or item < 0 for item in value.values()
    ):
        raise ValueError(f"partition {field} mapping changed")
    return {key: value[key] for key in expected_keys}  # type: ignore[return-value]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_source(record: dict[str, object]) -> SourceIdentity:
    return SourceIdentity(
        dataset_id=_string(record, "dataset_id"),
        revision=_string(record, "revision"),
        filename=_string(record, "filename"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_string(record, "sha256"),
    )


def _decode_tokenizer(record: dict[str, object]) -> TokenizerIdentity:
    files = tuple(
        HashedFile(
            _string(item, "name"),
            _integer(item, "size_bytes"),
            _string(item, "sha256"),
        )
        for item in _record_list(record, "files")
    )
    return TokenizerIdentity(
        kind=_string(record, "kind"),
        identifier=_string(record, "identifier"),
        revision=_string(record, "revision"),
        vocab_size=_integer(record, "vocab_size"),
        files=files,
    )


def _canonical_json_record(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _mapping(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"partition {field} must be an object")
    return value


def _record_list(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"partition {field} must contain records")
    return tuple(value)  # type: ignore[arg-type]


def _string_list(record: dict[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"partition {field} must contain text")
    return tuple(value)  # type: ignore[arg-type]


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"partition {field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"partition {field} must be a nonnegative integer")
    return value


def _standalone_html(catalog_sha256: str, markdown: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q partition audit</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f7f9;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-catalog-sha256=\"{catalog_sha256}\"><pre>"
        f"{html.escape(markdown)}</pre></body></html>\n"
    )


def _emit_progress(
    progress: ProgressCallback | None,
    phase: str,
    completed: int,
    total: int | None,
    detail: str,
) -> None:
    if progress is not None:
        progress(ProgressEvent(phase, completed, total, detail))


def _remove_staging(staging: Path) -> None:
    if not staging.exists():
        return
    for path in sorted(staging.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    staging.rmdir()


__all__ = [
    "PARTITION_TREE_FORMAT",
    "QUERY_PARTITION_PRESET",
    "QueryPartitionAssignment",
    "QueryPartitionPreset",
    "StorySemanticMatch",
    "assign_story_group",
    "build_query_partition",
    "iter_partition_documents",
    "load_query_partition",
    "match_story_semantics",
    "render_partition_audit",
    "tree_sha256",
]
