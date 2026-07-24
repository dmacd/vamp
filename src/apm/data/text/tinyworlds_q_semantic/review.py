"""Construction-slice fact discovery and human review packet publication."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import re
import tempfile

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    ProgressCallback,
    ProgressEvent,
    SourceIdentity,
)
from apm.data.text.tinyworlds_p.normalization import normalize_story_identity
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    CONSTRUCTION_BUCKET,
    CONSTRUCTION_BUCKET_COUNT,
    CONSTRUCTION_NAMESPACE,
    REVIEW_FORMAT,
    SCHEMA_VERSION,
    ConceptDefinition,
    StoryProvenance,
    canonical_json_bytes,
    record_sha256,
    require_identifier,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.source import QueryStoryGroup


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_DISCOVERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
        "you",
    }
)


@dataclass(frozen=True, slots=True)
class PredicateDefinition:
    """One normalized predicate proposed for exact construction co-occurrence."""

    predicate: str
    relation_category: str

    def __post_init__(self) -> None:
        if (
            type(self.predicate) is not str
            or not self.predicate
            or self.predicate != normalize_story_identity(self.predicate)
        ):
            raise ValueError("review predicates must be exact normalized text")
        require_identifier(self.relation_category, "predicate relation category")

    def as_record(self) -> dict[str, str]:
        """Return the canonical predicate proposal."""
        return {
            "predicate": self.predicate,
            "relation_category": self.relation_category,
        }


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    """One ranked concept/predicate co-occurrence with complete provenance."""

    candidate_id: str
    concept_id: str
    predicate: str
    relation_category: str
    supporting_story_groups: tuple[str, ...]
    evidence: tuple[StoryProvenance, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.candidate_id, "review candidate"),
            (self.concept_id, "review candidate concept"),
            (self.relation_category, "review candidate category"),
        ):
            require_identifier(value, label)
        if self.predicate != normalize_story_identity(self.predicate):
            raise ValueError("review candidate predicate must be normalized")
        if (
            type(self.supporting_story_groups) is not tuple
            or not self.supporting_story_groups
            or self.supporting_story_groups
            != tuple(sorted(set(self.supporting_story_groups)))
        ):
            raise ValueError("review support groups must be nonempty, sorted, and unique")
        for group_sha256 in self.supporting_story_groups:
            require_sha256(group_sha256, "review support group")
        if type(self.evidence) is not tuple or any(
            type(item) is not StoryProvenance for item in self.evidence
        ):
            raise TypeError("review evidence must contain StoryProvenance")
        if tuple(sorted({item.group_sha256 for item in self.evidence})) != self.supporting_story_groups:
            raise ValueError("review evidence must cover every support group")

    def as_record(self) -> dict[str, object]:
        """Return one canonical ranked review candidate."""
        return {
            "candidate_id": self.candidate_id,
            "concept_id": self.concept_id,
            "evidence": [item.as_record() for item in self.evidence],
            "predicate": self.predicate,
            "relation_category": self.relation_category,
            "supporting_story_groups": list(self.supporting_story_groups),
        }


@dataclass(frozen=True, slots=True)
class SemanticReviewPacket:
    """Deterministic construction-slice proposals awaiting human authority."""

    concepts: tuple[ConceptDefinition, ...]
    predicates: tuple[PredicateDefinition, ...]
    candidates: tuple[ReviewCandidate, ...]
    construction_group_count: int
    archive_identity: SourceIdentity = CANONICAL_ARCHIVE_IDENTITY
    packet_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.concepts) is not tuple or not self.concepts or any(
            type(item) is not ConceptDefinition for item in self.concepts
        ):
            raise TypeError("review packet concepts must be immutable definitions")
        if type(self.predicates) is not tuple or not self.predicates or any(
            type(item) is not PredicateDefinition for item in self.predicates
        ):
            raise TypeError("review packet predicates must be immutable definitions")
        if type(self.candidates) is not tuple or any(
            type(item) is not ReviewCandidate for item in self.candidates
        ):
            raise TypeError("review packet candidates must be immutable records")
        if type(self.construction_group_count) is not int or self.construction_group_count < 0:
            raise ValueError("construction group count must be nonnegative")
        concept_order = {concept.concept_id: index for index, concept in enumerate(self.concepts)}
        expected_order = tuple(
            sorted(
                self.candidates,
                key=lambda item: (
                    concept_order[item.concept_id],
                    -len(item.supporting_story_groups),
                    item.relation_category,
                    item.predicate,
                    item.candidate_id,
                ),
            )
        )
        if self.candidates != expected_order:
            raise ValueError("review candidates must use deterministic ranked order")
        object.__setattr__(
            self,
            "packet_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the complete review packet including source and evidence."""
        record: dict[str, object] = {
            "archive_identity": self.archive_identity.as_record(),
            "benchmark_id": BENCHMARK_ID,
            "candidates": [item.as_record() for item in self.candidates],
            "concepts": [item.as_record() for item in self.concepts],
            "construction": {
                "bucket": CONSTRUCTION_BUCKET,
                "bucket_count": CONSTRUCTION_BUCKET_COUNT,
                "group_count": self.construction_group_count,
                "namespace": CONSTRUCTION_NAMESPACE,
            },
            "format": REVIEW_FORMAT,
            "predicates": [item.as_record() for item in self.predicates],
            "schema_version": SCHEMA_VERSION,
        }
        if include_hash:
            record["packet_sha256"] = self.packet_sha256
        return record


def construction_bucket(group_sha256: str) -> int:
    """Map one duplicate group into the namespaced twenty-way construction split."""
    require_sha256(group_sha256, "construction group")
    digest = sha256(
        f"{CONSTRUCTION_NAMESPACE}\0{group_sha256}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % CONSTRUCTION_BUCKET_COUNT


def is_construction_group(group_sha256: str) -> bool:
    """Return whether a duplicate group belongs to the permanent 5% slice."""
    return construction_bucket(group_sha256) == CONSTRUCTION_BUCKET


def build_review_packet(
    groups: Iterable[QueryStoryGroup],
    concepts: tuple[ConceptDefinition, ...],
    predicates: tuple[PredicateDefinition, ...],
    *,
    progress: ProgressCallback | None = None,
    total_group_count: int | None = None,
) -> SemanticReviewPacket:
    """Rank exact same-sentence concept/predicate co-occurrences on construction data."""
    evidence_by_key: dict[tuple[str, str, str], list[StoryProvenance]] = {}
    construction_count = 0
    concept_patterns = {
        concept.concept_id: tuple(_surface_pattern(form) for form in concept.surface_forms)
        for concept in concepts
    }
    predicate_patterns = {
        predicate.predicate: _surface_pattern(predicate.predicate)
        for predicate in predicates
    }
    processed_group_count = 0
    progress_interval = _progress_interval(total_group_count)
    _review_progress(progress, 0, total_group_count, "matching reviewed predicates")
    for group in groups:
        if type(group) is not QueryStoryGroup:
            raise TypeError("review packet groups must contain QueryStoryGroup values")
        processed_group_count += 1
        if processed_group_count % progress_interval == 0:
            _review_progress(
                progress,
                processed_group_count,
                total_group_count,
                "matching exact concept/predicate sentence co-occurrences",
            )
        if not is_construction_group(group.normalized_story_sha256):
            continue
        construction_count += 1
        for occurrence in group.occurrences:
            raw_story = occurrence.story_bytes.decode("utf-8")
            sentences = tuple(
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY.split(raw_story)
                if sentence.strip()
            )
            for sentence_index, sentence in enumerate(sentences):
                normalized_sentence = normalize_story_identity(sentence)
                present_concepts = tuple(
                    concept.concept_id
                    for concept in concepts
                    if any(pattern.search(normalized_sentence) for pattern in concept_patterns[concept.concept_id])
                )
                present_predicates = tuple(
                    predicate
                    for predicate in predicates
                    if predicate_patterns[predicate.predicate].search(normalized_sentence)
                )
                provenance = StoryProvenance(
                    group_sha256=group.normalized_story_sha256,
                    story_sha256=occurrence.story_sha256,
                    record_id=occurrence.record_id,
                    source_member=occurrence.source_member,
                    source_index=occurrence.source_index,
                    sentence_index=sentence_index,
                    sentence_text=sentence,
                )
                for concept_id in present_concepts:
                    for predicate in present_predicates:
                        evidence_by_key.setdefault(
                            (concept_id, predicate.predicate, predicate.relation_category),
                            [],
                        ).append(provenance)
    _finish_review_progress(progress, processed_group_count, total_group_count)
    concept_order = {concept.concept_id: index for index, concept in enumerate(concepts)}
    candidates = tuple(
        sorted(
            (
                _review_candidate(key, tuple(evidence))
                for key, evidence in evidence_by_key.items()
            ),
            key=lambda item: (
                concept_order[item.concept_id],
                -len(item.supporting_story_groups),
                item.relation_category,
                item.predicate,
                item.candidate_id,
            ),
        )
    )
    return SemanticReviewPacket(
        concepts=concepts,
        predicates=predicates,
        candidates=candidates,
        construction_group_count=construction_count,
    )


def discover_review_packet(
    groups: Iterable[QueryStoryGroup],
    concepts: tuple[ConceptDefinition, ...],
    *,
    minimum_group_support: int = 2,
    maximum_candidates_per_concept: int = 200,
    progress: ProgressCallback | None = None,
    total_group_count: int | None = None,
) -> SemanticReviewPacket:
    """Discover and rank exact nearby predicate n-grams on construction stories.

    This intentionally supplies proposals rather than semantic authority. The
    human review stage still chooses relation categories, trigger closure,
    accepted answers, distractors, and final evidence.
    """
    if (
        type(minimum_group_support) is not int
        or minimum_group_support <= 0
        or type(maximum_candidates_per_concept) is not int
        or maximum_candidates_per_concept <= 0
    ):
        raise ValueError("discovery support and candidate limits must be positive")
    evidence_by_key: dict[tuple[str, str], list[StoryProvenance]] = {}
    construction_count = 0
    concept_surfaces = {
        concept.concept_id: frozenset(concept.surface_forms) for concept in concepts
    }
    concept_patterns = {
        concept.concept_id: tuple(_surface_pattern(form) for form in concept.surface_forms)
        for concept in concepts
    }
    processed_group_count = 0
    progress_interval = _progress_interval(total_group_count)
    _review_progress(progress, 0, total_group_count, "discovering predicate proposals")
    for group in groups:
        if type(group) is not QueryStoryGroup:
            raise TypeError("review discovery groups must contain QueryStoryGroup values")
        processed_group_count += 1
        if processed_group_count % progress_interval == 0:
            _review_progress(
                progress,
                processed_group_count,
                total_group_count,
                "ranking nearby sentence-level predicate n-grams",
            )
        if not is_construction_group(group.normalized_story_sha256):
            continue
        construction_count += 1
        for occurrence in group.occurrences:
            story = occurrence.story_bytes.decode("utf-8")
            sentences = tuple(
                sentence.strip()
                for sentence in _SENTENCE_BOUNDARY.split(story)
                if sentence.strip()
            )
            for sentence_index, sentence in enumerate(sentences):
                normalized = normalize_story_identity(sentence)
                words = tuple(_WORD.findall(normalized))
                if not words:
                    continue
                provenance = StoryProvenance(
                    group_sha256=group.normalized_story_sha256,
                    story_sha256=occurrence.story_sha256,
                    record_id=occurrence.record_id,
                    source_member=occurrence.source_member,
                    source_index=occurrence.source_index,
                    sentence_index=sentence_index,
                    sentence_text=sentence,
                )
                for concept in concepts:
                    if not any(
                        pattern.search(normalized)
                        for pattern in concept_patterns[concept.concept_id]
                    ):
                        continue
                    surfaces = concept_surfaces[concept.concept_id]
                    concept_positions = tuple(
                        index for index, word in enumerate(words) if word in surfaces
                    )
                    proposed_phrases = {
                        " ".join(words[start : start + length])
                        for position in concept_positions
                        for start in range(max(0, position - 6), min(len(words), position + 7))
                        for length in (1, 2, 3)
                        if start + length <= len(words)
                        and not start <= position < start + length
                        and any(
                            word not in _DISCOVERY_STOPWORDS
                            for word in words[start : start + length]
                        )
                        and not any(
                            word in surfaces for word in words[start : start + length]
                        )
                    }
                    for phrase in proposed_phrases:
                        evidence_by_key.setdefault(
                            (concept.concept_id, phrase),
                            [],
                        ).append(provenance)
    _finish_review_progress(progress, processed_group_count, total_group_count)
    candidates_by_concept = {
        concept.concept_id: tuple(
            sorted(
                (
                    _review_candidate(
                        (concept.concept_id, phrase, "unclassified"),
                        tuple(evidence),
                    )
                    for (concept_id, phrase), evidence in evidence_by_key.items()
                    if concept_id == concept.concept_id
                    and len({item.group_sha256 for item in evidence})
                    >= minimum_group_support
                ),
                key=lambda item: (
                    -len(item.supporting_story_groups),
                    len(item.predicate.split()),
                    item.predicate,
                    item.candidate_id,
                ),
            )[:maximum_candidates_per_concept]
        )
        for concept in concepts
    }
    candidates = tuple(
        candidate
        for concept in concepts
        for candidate in sorted(
            candidates_by_concept[concept.concept_id],
            key=lambda item: (
                -len(item.supporting_story_groups),
                item.relation_category,
                item.predicate,
                item.candidate_id,
            ),
        )
    )
    predicates = tuple(
        PredicateDefinition(phrase, "unclassified")
        for phrase in sorted({candidate.predicate for candidate in candidates})
    )
    return SemanticReviewPacket(
        concepts=concepts,
        predicates=predicates,
        candidates=candidates,
        construction_group_count=construction_count,
    )


def _review_candidate(
    key: tuple[str, str, str],
    evidence: tuple[StoryProvenance, ...],
) -> ReviewCandidate:
    concept_id, predicate, relation_category = key
    candidate_digest = record_sha256(
        {
            "concept_id": concept_id,
            "predicate": predicate,
            "relation_category": relation_category,
        }
    )[:16]
    ordered_evidence = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.group_sha256,
                item.source_member,
                item.source_index,
                item.sentence_index,
                item.record_id,
            ),
        )
    )
    return ReviewCandidate(
        candidate_id=f"{concept_id}-candidate-{candidate_digest}",
        concept_id=concept_id,
        predicate=predicate,
        relation_category=relation_category,
        supporting_story_groups=tuple(
            sorted({item.group_sha256 for item in ordered_evidence})
        ),
        evidence=ordered_evidence,
    )


def publish_review_packet(
    packet: SemanticReviewPacket,
    output_root: str | Path,
) -> Path:
    """Atomically publish canonical JSON plus Markdown and standalone HTML."""
    root = Path(output_root) / "review" / packet.packet_sha256
    if root.exists():
        _verify_existing_packet(root, packet)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".review-", dir=root.parent))
    try:
        packet_bytes = canonical_json_bytes(packet.as_record())
        markdown = render_review_markdown(packet)
        html_text = render_review_html(packet, markdown)
        (staging / "review.json").write_bytes(packet_bytes)
        (staging / "review.md").write_text(markdown, encoding="utf-8", newline="\n")
        (staging / "review.html").write_text(html_text, encoding="utf-8", newline="\n")
        payloads = {
            name: (staging / name).read_bytes()
            for name in ("review.html", "review.json", "review.md")
        }
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes(
                {
                    "files": [
                        {
                            "name": name,
                            "sha256": sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        }
                        for name, payload in sorted(payloads.items())
                    ],
                    "format": REVIEW_FORMAT,
                    "packet_sha256": packet.packet_sha256,
                    "schema_version": SCHEMA_VERSION,
                }
            )
        )
        os.replace(staging, root)
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if staging.exists():
            staging.rmdir()
        raise
    return root


def load_review_packet(directory: str | Path) -> SemanticReviewPacket:
    """Strictly reconstruct a review packet and its rendered human surfaces."""
    root = Path(directory)
    manifest = _load_canonical_json(root / "manifest.json")
    if (
        manifest.get("format") != REVIEW_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or root.name != manifest.get("packet_sha256")
    ):
        raise ValueError("review packet tree identity changed")
    file_records = _record_tuple(manifest, "files")
    expected_names = {"manifest.json", *(_text(item, "name") for item in file_records)}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("review packet tree entries changed")
    for item in file_records:
        payload = (root / _text(item, "name")).read_bytes()
        if len(payload) != _integer(item, "size_bytes") or sha256(payload).hexdigest() != _text(item, "sha256"):
            raise ValueError("review packet file changed")
    record = _load_canonical_json(root / "review.json")
    concepts = tuple(
        ConceptDefinition(
            _text(item, "concept_id"),
            _text_tuple(item, "surface_forms"),
        )
        for item in _record_tuple(record, "concepts")
    )
    predicates = tuple(
        PredicateDefinition(
            _text(item, "predicate"),
            _text(item, "relation_category"),
        )
        for item in _record_tuple(record, "predicates")
    )
    candidates = tuple(
        ReviewCandidate(
            candidate_id=_text(item, "candidate_id"),
            concept_id=_text(item, "concept_id"),
            predicate=_text(item, "predicate"),
            relation_category=_text(item, "relation_category"),
            supporting_story_groups=_text_tuple(item, "supporting_story_groups"),
            evidence=tuple(
                StoryProvenance(
                    group_sha256=_text(evidence, "group_sha256"),
                    story_sha256=_text(evidence, "story_sha256"),
                    record_id=_text(evidence, "record_id"),
                    source_member=_text(evidence, "source_member"),
                    source_index=_integer(evidence, "source_index"),
                    sentence_index=_integer(evidence, "sentence_index"),
                    sentence_text=_text(evidence, "sentence_text"),
                )
                for evidence in _record_tuple(item, "evidence")
            ),
        )
        for item in _record_tuple(record, "candidates")
    )
    archive_record = _mapping(record, "archive_identity")
    construction = _mapping(record, "construction")
    packet = SemanticReviewPacket(
        concepts=concepts,
        predicates=predicates,
        candidates=candidates,
        construction_group_count=_integer(construction, "group_count"),
        archive_identity=SourceIdentity(
            dataset_id=_text(archive_record, "dataset_id"),
            revision=_text(archive_record, "revision"),
            filename=_text(archive_record, "filename"),
            size_bytes=_integer(archive_record, "size_bytes"),
            sha256=_text(archive_record, "sha256"),
        ),
    )
    if packet.packet_sha256 != manifest.get("packet_sha256") or packet.as_record() != record:
        raise ValueError("review packet semantic content changed")
    if (root / "review.md").read_text(encoding="utf-8") != render_review_markdown(packet) or (root / "review.html").read_text(encoding="utf-8") != render_review_html(packet, render_review_markdown(packet)):
        raise ValueError("review packet rendered audit changed")
    return packet


def render_review_markdown(packet: SemanticReviewPacket) -> str:
    """Render the complete ranked review evidence as deterministic Markdown."""
    lines = [
        "# TinyWorlds-Q semantic review packet",
        "",
        f"Packet: `{packet.packet_sha256}`",
        "",
        f"Construction groups examined: {packet.construction_group_count}",
        "",
        "This packet proposes evidence only. It does not approve semantic facts.",
        "",
    ]
    for candidate in packet.candidates:
        lines.extend(
            (
                f"## {candidate.concept_id}: {candidate.predicate}",
                "",
                f"Candidate: `{candidate.candidate_id}`  ",
                f"Relation: `{candidate.relation_category}`  ",
                f"Distinct construction groups: {len(candidate.supporting_story_groups)}",
                "",
                "Review gates: truth [ ] answer forms [ ] trigger closure [ ] "
                "distractors [ ] evidence [ ]",
                "",
            )
        )
        for evidence in candidate.evidence:
            lines.extend(
                (
                    f"- `{evidence.group_sha256}` — `{evidence.source_member}` "
                    f"record {evidence.source_index}, sentence {evidence.sentence_index}: "
                    f"{evidence.sentence_text}",
                    f"  - record `{evidence.record_id}`; story `{evidence.story_sha256}`",
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_review_html(packet: SemanticReviewPacket, markdown: str) -> str:
    """Render a standalone escaped HTML review surface without dependencies."""
    body = html.escape(markdown)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q semantic review</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f7f9;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-packet-sha256=\"{packet.packet_sha256}\"><pre>{body}</pre></body></html>\n"
    )


def _surface_pattern(surface: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)")


def _progress_interval(total_group_count: int | None) -> int:
    if total_group_count is not None and (
        type(total_group_count) is not int or total_group_count <= 0
    ):
        raise ValueError("review total_group_count must be positive")
    return max(1, min(100_000, (total_group_count or 1_000_000) // 100))


def _review_progress(
    progress: ProgressCallback | None,
    completed: int,
    total: int | None,
    detail: str,
) -> None:
    if progress is not None:
        progress(ProgressEvent("buckets", completed, total, detail))


def _finish_review_progress(
    progress: ProgressCallback | None,
    processed_group_count: int,
    total_group_count: int | None,
) -> None:
    if total_group_count is not None and processed_group_count != total_group_count:
        raise ValueError("review group count differs from the authenticated archive audit")
    _review_progress(
        progress,
        processed_group_count,
        total_group_count,
        "construction-slice proposal discovery completed",
    )


def _verify_existing_packet(root: Path, packet: SemanticReviewPacket) -> None:
    try:
        loaded = load_review_packet(root)
    except (OSError, ValueError) as error:
        raise FileExistsError("existing review packet is incomplete or changed") from error
    if loaded != packet:
        raise FileExistsError("existing review packet does not match requested bytes")


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review JSON {path.name}: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"review JSON is not canonical: {path.name}")
    return value


def _mapping(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"review {field} must be an object")
    return value


def _record_tuple(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"review {field} must contain records")
    return tuple(value)  # type: ignore[arg-type]


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"review {field} must be nonempty text")
    return value


def _text_tuple(record: dict[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"review {field} must contain text")
    return tuple(value)  # type: ignore[arg-type]


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"review {field} must be a nonnegative integer")
    return value


__all__ = [
    "PredicateDefinition",
    "ReviewCandidate",
    "SemanticReviewPacket",
    "build_review_packet",
    "construction_bucket",
    "discover_review_packet",
    "is_construction_group",
    "load_review_packet",
    "publish_review_packet",
    "render_review_html",
    "render_review_markdown",
]
