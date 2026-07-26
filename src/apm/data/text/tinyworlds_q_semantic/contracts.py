"""Immutable public contracts for the query-native TinyWorlds benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Literal, TypeAlias

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    NormalizationIdentity,
    SourceIdentity,
    TokenizerIdentity,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.training import LmTrainConfig


BENCHMARK_ID = "tinyworlds-q-semantic-v1"
CATALOG_FORMAT = "tinyworlds-q-semantic-catalog-v1"
PARTITION_FORMAT = "tinyworlds-q-semantic-partition-v1"
REVIEW_FORMAT = "tinyworlds-q-semantic-review-v1"
RESULT_FORMAT = "tinyworlds-q-semantic-result-v1"
SCHEMA_VERSION = 1
CONSTRUCTION_NAMESPACE = f"{BENCHMARK_ID}:construction"
CONSTRUCTION_BUCKET_COUNT = 20
CONSTRUCTION_BUCKET = 0
CATALOG_ROOT = Path("data/tinyworlds-q-semantic")
CHECKPOINT_ROOT = Path("checkpoints/tinyworlds-q-semantic-v1")
RESULT_ROOT = Path("results/language_cl/tinyworlds-q-semantic-v1")

QueryDirection: TypeAlias = Literal["forward", "reverse"]
QuerySplit: TypeAlias = Literal["validation", "test"]
EvaluationSchedule: TypeAlias = Literal["full", "milestone"]
PartitionRole: TypeAlias = Literal["base", "node", "construction", "excluded"]
PartitionSplit: TypeAlias = Literal["train", "validation", "test"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON-compatible value with the benchmark's canonical form."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def record_sha256(value: object) -> str:
    """Hash one canonical JSON record."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def registered_distractor_order(
    distractors: tuple[str, str, str],
    split: QuerySplit,
    paraphrase_index: int,
) -> tuple[str, str, str]:
    """Rotate one reviewed distractor set by the registered global template index."""
    limits = {"validation": 3, "test": 5}
    if split not in limits:
        raise ValueError("query split must be validation or test")
    if type(paraphrase_index) is not int or not 0 <= paraphrase_index < limits[split]:
        raise ValueError("paraphrase index lies outside its registered split")
    if type(distractors) is not tuple or len(distractors) != 3:
        raise ValueError("registered distractor reuse requires exactly three forms")
    canonical = tuple(sorted(distractors))
    if len(set(canonical)) != 3:
        raise ValueError("registered distractors must be unique")
    global_index = paraphrase_index + (0 if split == "validation" else 3)
    rotation = global_index % len(canonical)
    return canonical[rotation:] + canonical[:rotation]  # type: ignore[return-value]


def require_sha256(value: str, label: str) -> None:
    """Require one lowercase SHA-256 digest."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")


def require_identifier(value: str, label: str) -> None:
    """Require one stable lowercase benchmark identifier."""
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be one canonical lowercase identifier")


def _normalized_surface(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value != value.casefold()
        or "  " in value
        or any(character in value for character in "\r\n\t")
    ):
        raise ValueError(f"{label} must be an exact normalized surface form")


def _unique_tuple(
    values: tuple[str, ...],
    label: str,
    *,
    minimum: int = 1,
    sorted_values: bool = False,
) -> None:
    if type(values) is not tuple or len(values) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} values")
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"{label} values must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique")
    if sorted_values and values != tuple(sorted(values)):
        raise ValueError(f"{label} values must be sorted")


@dataclass(frozen=True, slots=True)
class ConceptDefinition:
    """One concept family and its exact normalized word forms."""

    concept_id: str
    surface_forms: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.concept_id, "concept_id")
        _unique_tuple(self.surface_forms, "concept surface forms")
        tuple(
            _normalized_surface(surface, "concept surface")
            for surface in self.surface_forms
        )
        if self.surface_forms[0] != self.concept_id:
            raise ValueError("the canonical concept surface must equal concept_id")

    def as_record(self) -> dict[str, object]:
        """Return the canonical concept record."""
        return {
            "concept_id": self.concept_id,
            "surface_forms": list(self.surface_forms),
        }


@dataclass(frozen=True, slots=True)
class StoryProvenance:
    """Complete archive provenance for one reviewed evidence sentence."""

    group_sha256: str
    story_sha256: str
    record_id: str
    source_member: str
    source_index: int
    sentence_index: int
    sentence_text: str

    def __post_init__(self) -> None:
        require_sha256(self.group_sha256, "evidence group")
        require_sha256(self.story_sha256, "evidence story")
        if any(
            type(value) is not str or not value
            for value in (self.record_id, self.source_member, self.sentence_text)
        ):
            raise ValueError("evidence provenance strings must be nonempty")
        if any(
            type(value) is not int or value < 0
            for value in (self.source_index, self.sentence_index)
        ):
            raise ValueError("evidence provenance indexes must be nonnegative")

    def as_record(self) -> dict[str, object]:
        """Return the complete canonical provenance record."""
        return {
            "group_sha256": self.group_sha256,
            "record_id": self.record_id,
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "source_index": self.source_index,
            "source_member": self.source_member,
            "story_sha256": self.story_sha256,
        }


@dataclass(frozen=True, slots=True)
class FactReviewDecision:
    """Human approval of every semantic authority required for one fact."""

    fact_id: str
    reviewer: str
    reviewed_at: str
    truth_approved: bool
    answer_forms_approved: bool
    trigger_closure_approved: bool
    distractors_approved: bool
    evidence_approved: bool

    def __post_init__(self) -> None:
        require_identifier(self.fact_id, "review fact_id")
        if any(
            type(value) is not str or not value.strip()
            for value in (self.reviewer, self.reviewed_at)
        ):
            raise ValueError("reviewer and reviewed_at must be nonempty")
        decisions = self.approvals
        if any(type(value) is not bool for value in decisions):
            raise TypeError("fact review approvals must be booleans")

    @property
    def approvals(self) -> tuple[bool, ...]:
        """Return approvals in the registered review order."""
        return (
            self.truth_approved,
            self.answer_forms_approved,
            self.trigger_closure_approved,
            self.distractors_approved,
            self.evidence_approved,
        )

    @property
    def approved(self) -> bool:
        """Return whether every required human decision is affirmative."""
        return all(self.approvals)

    def as_record(self) -> dict[str, object]:
        """Return the canonical human-review record."""
        return {
            "answer_forms_approved": self.answer_forms_approved,
            "distractors_approved": self.distractors_approved,
            "evidence_approved": self.evidence_approved,
            "fact_id": self.fact_id,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "trigger_closure_approved": self.trigger_closure_approved,
            "truth_approved": self.truth_approved,
        }


@dataclass(frozen=True, slots=True)
class RejectedFactCandidate:
    """One reviewed candidate that was deliberately excluded from authority."""

    candidate_id: str
    concept_id: str
    predicate: str
    reason: str
    reviewer: str
    evidence_group_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.candidate_id, "rejected candidate_id")
        require_identifier(self.concept_id, "rejected concept_id")
        _normalized_surface(self.predicate, "rejected predicate")
        if any(
            type(value) is not str or not value.strip()
            for value in (self.reason, self.reviewer)
        ):
            raise ValueError("rejected candidate reason and reviewer are required")
        _unique_tuple(
            self.evidence_group_sha256,
            "rejected evidence groups",
            sorted_values=True,
        )
        for group_sha256 in self.evidence_group_sha256:
            require_sha256(group_sha256, "rejected evidence group")

    def as_record(self) -> dict[str, object]:
        """Return the canonical rejected-candidate record."""
        return {
            "candidate_id": self.candidate_id,
            "concept_id": self.concept_id,
            "evidence_group_sha256": list(self.evidence_group_sha256),
            "predicate": self.predicate,
            "reason": self.reason,
            "reviewer": self.reviewer,
        }


@dataclass(frozen=True, slots=True)
class SemanticFact:
    """One reviewed concept fact and its construction-slice evidence."""

    fact_id: str
    source_candidate_id: str
    concept_id: str
    relation_category: str
    answer_type: str
    canonical_answer: str
    accepted_forms: tuple[str, ...]
    trigger_forms: tuple[str, ...]
    supporting_story_groups: tuple[str, ...]
    evidence: tuple[StoryProvenance, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.fact_id, "fact_id"),
            (self.source_candidate_id, "fact source_candidate_id"),
            (self.concept_id, "fact concept_id"),
            (self.relation_category, "relation_category"),
            (self.answer_type, "answer_type"),
        ):
            require_identifier(value, label)
        _normalized_surface(self.canonical_answer, "canonical answer")
        for values, label in (
            (self.accepted_forms, "accepted forms"),
            (self.trigger_forms, "trigger forms"),
        ):
            _unique_tuple(values, label)
            for value in values:
                _normalized_surface(value, label)
        if self.accepted_forms[0] != self.canonical_answer:
            raise ValueError("canonical_answer must lead accepted_forms")
        _unique_tuple(
            self.supporting_story_groups,
            "supporting construction groups",
            minimum=16,
            sorted_values=True,
        )
        for group_sha256 in self.supporting_story_groups:
            require_sha256(group_sha256, "supporting construction group")
        if type(self.evidence) is not tuple or not self.evidence:
            raise ValueError("semantic facts require complete evidence provenance")
        if any(type(item) is not StoryProvenance for item in self.evidence):
            raise TypeError("semantic fact evidence must contain StoryProvenance")
        evidence_groups = tuple(sorted({item.group_sha256 for item in self.evidence}))
        if evidence_groups != self.supporting_story_groups:
            raise ValueError("fact evidence must cover exactly its supporting groups")

    def as_record(self) -> dict[str, object]:
        """Return the canonical semantic-fact record."""
        return {
            "accepted_forms": list(self.accepted_forms),
            "answer_type": self.answer_type,
            "canonical_answer": self.canonical_answer,
            "concept_id": self.concept_id,
            "evidence": [item.as_record() for item in self.evidence],
            "fact_id": self.fact_id,
            "relation_category": self.relation_category,
            "source_candidate_id": self.source_candidate_id,
            "supporting_story_groups": list(self.supporting_story_groups),
            "trigger_forms": list(self.trigger_forms),
        }


@dataclass(frozen=True, slots=True)
class SemanticQueryTemplate:
    """One reviewed four-choice paraphrase with exact tokenizer boundaries."""

    template_id: str
    fact_id: str
    direction: QueryDirection
    prompt_text: str
    canonical_answer_form: str
    distractors: tuple[str, str, str]
    candidate_grammatical_types: tuple[str, str, str, str]
    split: QuerySplit
    correct_candidate_index: int
    prompt_token_ids: tuple[int, ...]
    combined_candidate_token_ids: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]

    def __post_init__(self) -> None:
        require_identifier(self.template_id, "template_id")
        require_identifier(self.fact_id, "template fact_id")
        if self.direction not in ("forward", "reverse"):
            raise ValueError("query direction must be forward or reverse")
        if self.split not in ("validation", "test"):
            raise ValueError("query split must be validation or test")
        if type(self.prompt_text) is not str or not self.prompt_text.strip():
            raise ValueError("query prompt must contain visible text")
        _normalized_surface(self.canonical_answer_form, "query canonical answer")
        if type(self.distractors) is not tuple or len(self.distractors) != 3:
            raise ValueError("query templates require exactly three distractors")
        for distractor in self.distractors:
            _normalized_surface(distractor, "query distractor")
        if len(set((self.canonical_answer_form, *self.distractors))) != 4:
            raise ValueError("query answer choices must be unique")
        if (
            type(self.correct_candidate_index) is not int
            or not 0 <= self.correct_candidate_index < 4
        ):
            raise ValueError("correct_candidate_index must lie in [0, 3]")
        if (
            type(self.candidate_grammatical_types) is not tuple
            or len(self.candidate_grammatical_types) != 4
            or any(
                type(value) is not str or not value
                for value in self.candidate_grammatical_types
            )
            or len(set(self.candidate_grammatical_types)) != 1
        ):
            raise ValueError("all four candidates must share one grammatical type")
        if (
            type(self.prompt_token_ids) is not tuple
            or len(self.prompt_token_ids) < 2
            or any(type(token) is not int or token < 0 for token in self.prompt_token_ids)
        ):
            raise ValueError("prompt_token_ids must contain at least two token IDs")
        combined = self.combined_candidate_token_ids
        if type(combined) is not tuple or len(combined) != 4:
            raise ValueError("query templates require four combined tokenizations")
        if any(
            type(tokens) is not tuple
            or len(tokens) <= len(self.prompt_token_ids)
            or any(type(token) is not int or token < 0 for token in tokens)
            or tokens[: len(self.prompt_token_ids)] != self.prompt_token_ids
            for tokens in combined
        ):
            raise ValueError("combined tokenizations must extend one exact prompt")
        suffix_lengths = tuple(
            len(tokens) - len(self.prompt_token_ids) for tokens in combined
        )
        if len(set(suffix_lengths)) != 1:
            raise ValueError("all candidate answers must have equal tokenizer length")

    @property
    def candidate_answer_forms(self) -> tuple[str, str, str, str]:
        """Return answer forms in their deterministically balanced positions."""
        distractor_iterator = iter(self.distractors)
        return tuple(
            self.canonical_answer_form
            if index == self.correct_candidate_index
            else next(distractor_iterator)
            for index in range(4)
        )  # type: ignore[return-value]

    @property
    def answer_token_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return only the scored answer-token suffix for each candidate."""
        boundary = len(self.prompt_token_ids)
        return tuple(tokens[boundary:] for tokens in self.combined_candidate_token_ids)

    def as_record(self) -> dict[str, object]:
        """Return the canonical prompt, answer, and tokenization record."""
        return {
            "candidate_grammatical_types": list(self.candidate_grammatical_types),
            "canonical_answer_form": self.canonical_answer_form,
            "combined_candidate_token_ids": [
                list(tokens) for tokens in self.combined_candidate_token_ids
            ],
            "correct_candidate_index": self.correct_candidate_index,
            "direction": self.direction,
            "distractors": list(self.distractors),
            "fact_id": self.fact_id,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "split": self.split,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class SemanticQueryCatalog:
    """Ordered reviewed semantic authority with an automatic content hash."""

    concepts: tuple[ConceptDefinition, ...]
    facts: tuple[SemanticFact, ...]
    templates: tuple[SemanticQueryTemplate, ...]
    reviews: tuple[FactReviewDecision, ...]
    rejected_candidates: tuple[RejectedFactCandidate, ...]
    review_packet_sha256: str
    parent_catalog_sha256: str | None = None
    archive_identity: SourceIdentity = CANONICAL_ARCHIVE_IDENTITY
    tokenizer_identity: TokenizerIdentity = CANONICAL_TOKENIZER_IDENTITY
    normalization: NormalizationIdentity = field(default_factory=NormalizationIdentity)
    catalog_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_catalog(self)
        object.__setattr__(
            self,
            "catalog_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    @property
    def concept_ids(self) -> tuple[str, ...]:
        """Return the ordered concept manifest."""
        return tuple(concept.concept_id for concept in self.concepts)

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the complete canonical catalog payload."""
        record: dict[str, object] = {
            "archive_identity": self.archive_identity.as_record(),
            "benchmark_id": BENCHMARK_ID,
            "concepts": [concept.as_record() for concept in self.concepts],
            "construction": {
                "bucket": CONSTRUCTION_BUCKET,
                "bucket_count": CONSTRUCTION_BUCKET_COUNT,
                "namespace": CONSTRUCTION_NAMESPACE,
            },
            "facts": [fact.as_record() for fact in self.facts],
            "format": CATALOG_FORMAT,
            "normalization": self.normalization.as_record(),
            "parent_catalog_sha256": self.parent_catalog_sha256,
            "rejected_candidates": [
                candidate.as_record() for candidate in self.rejected_candidates
            ],
            "review_packet_sha256": self.review_packet_sha256,
            "reviews": [review.as_record() for review in self.reviews],
            "schema_version": SCHEMA_VERSION,
            "templates": [template.as_record() for template in self.templates],
            "tokenizer_identity": self.tokenizer_identity.as_record(),
        }
        if include_hash:
            record["catalog_sha256"] = self.catalog_sha256
        return record


def _validate_catalog(catalog: SemanticQueryCatalog) -> None:
    tuple_types = (
        (catalog.concepts, ConceptDefinition, "concepts"),
        (catalog.facts, SemanticFact, "facts"),
        (catalog.templates, SemanticQueryTemplate, "templates"),
        (catalog.reviews, FactReviewDecision, "reviews"),
        (catalog.rejected_candidates, RejectedFactCandidate, "rejected candidates"),
    )
    for values, expected_type, label in tuple_types:
        if type(values) is not tuple or any(
            type(value) is not expected_type for value in values
        ):
            raise TypeError(f"catalog {label} have an invalid immutable contract")
    if not catalog.concepts:
        raise ValueError("semantic catalogs require at least one concept")
    concept_ids = catalog.concept_ids
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("catalog concept IDs must be unique")
    all_surfaces = tuple(
        surface for concept in catalog.concepts for surface in concept.surface_forms
    )
    if len(set(all_surfaces)) != len(all_surfaces):
        raise ValueError("concept families must not share normalized surface forms")
    facts_by_concept = {
        concept_id: tuple(fact for fact in catalog.facts if fact.concept_id == concept_id)
        for concept_id in concept_ids
    }
    if any(len(facts) != 12 for facts in facts_by_concept.values()):
        raise ValueError("every concept must contain exactly twelve accepted facts")
    if any(
        len({fact.relation_category for fact in facts}) < 4
        for facts in facts_by_concept.values()
    ):
        raise ValueError("every concept must span at least four relation categories")
    fact_ids = tuple(fact.fact_id for fact in catalog.facts)
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError("catalog fact IDs must be unique")
    if any(fact.concept_id not in concept_ids for fact in catalog.facts):
        raise ValueError("every fact must belong to a catalog concept")
    source_candidate_ids = tuple(fact.source_candidate_id for fact in catalog.facts)
    if len(set(source_candidate_ids)) != len(source_candidate_ids):
        raise ValueError("accepted facts must bind distinct review candidates")
    reviews = {review.fact_id: review for review in catalog.reviews}
    if set(reviews) != set(fact_ids) or any(not review.approved for review in reviews.values()):
        raise ValueError("every accepted fact requires complete affirmative human review")
    templates_by_fact = {
        fact_id: tuple(
            template for template in catalog.templates if template.fact_id == fact_id
        )
        for fact_id in fact_ids
    }
    if any(len(templates) != 8 for templates in templates_by_fact.values()):
        raise ValueError("every fact requires three validation and five test templates")
    fact_by_id = {fact.fact_id: fact for fact in catalog.facts}
    concept_by_id = {concept.concept_id: concept for concept in catalog.concepts}
    for fact_id, templates in templates_by_fact.items():
        validation = tuple(item for item in templates if item.split == "validation")
        test = tuple(item for item in templates if item.split == "test")
        if (
            len(validation) != 3
            or tuple(item.direction for item in validation).count("forward") != 2
            or tuple(item.direction for item in validation).count("reverse") != 1
            or len(test) != 5
            or tuple(item.direction for item in test).count("forward") != 3
            or tuple(item.direction for item in test).count("reverse") != 2
        ):
            raise ValueError("template directions must follow the registered 2/1 and 3/2 design")
        if tuple(sorted(item.correct_candidate_index for item in templates)) != (
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
        ):
            raise ValueError("answer positions must be exactly balanced within each fact")
        for direction in ("forward", "reverse"):
            directional = tuple(
                item for item in templates if item.direction == direction
            )
            distractor_sets = {
                frozenset(item.distractors) for item in directional
            }
            if len(distractor_sets) != 1:
                raise ValueError(
                    "each fact direction must reuse one reviewed distractor set"
                )
        for split in ("validation", "test"):
            split_templates = tuple(item for item in templates if item.split == split)
            for paraphrase_index, template in enumerate(split_templates):
                if template.distractors != registered_distractor_order(
                    template.distractors,
                    split,
                    paraphrase_index,
                ):
                    raise ValueError("distractor positions must use the registered rotation")
        fact = fact_by_id[fact_id]
        concept = concept_by_id[fact.concept_id]
        for template in templates:
            accepted = (
                fact.accepted_forms
                if template.direction == "forward"
                else concept.surface_forms
            )
            if template.canonical_answer_form not in accepted:
                raise ValueError("template canonical answer is outside reviewed forms")
            if set(template.candidate_grammatical_types) != {fact.answer_type if template.direction == "forward" else "concept"}:
                raise ValueError("template candidate grammatical type changed")
    template_ids = tuple(template.template_id for template in catalog.templates)
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("catalog template IDs must be unique")
    if any(candidate.concept_id not in concept_ids for candidate in catalog.rejected_candidates):
        raise ValueError("rejected candidates must name catalog concepts")
    require_sha256(catalog.review_packet_sha256, "review packet")
    if catalog.parent_catalog_sha256 is not None:
        require_sha256(catalog.parent_catalog_sha256, "parent catalog")
    if type(catalog.archive_identity) is not SourceIdentity:
        raise TypeError("catalog archive identity must be SourceIdentity")
    if type(catalog.tokenizer_identity) is not TokenizerIdentity:
        raise TypeError("catalog tokenizer identity must be TokenizerIdentity")
    if type(catalog.normalization) is not NormalizationIdentity:
        raise TypeError("catalog normalization must be NormalizationIdentity")


def validate_parent_catalog_prefix(
    catalog: SemanticQueryCatalog,
    parent: SemanticQueryCatalog,
) -> None:
    """Require a larger catalog to preserve its authenticated parent as a prefix."""
    if catalog.parent_catalog_sha256 != parent.catalog_sha256:
        raise ValueError("catalog does not bind the supplied parent")
    prefix_concepts = catalog.concepts[: len(parent.concepts)]
    prefix_concept_ids = {concept.concept_id for concept in prefix_concepts}
    prefix_facts = tuple(
        fact for fact in catalog.facts if fact.concept_id in prefix_concept_ids
    )
    prefix_fact_ids = {fact.fact_id for fact in prefix_facts}
    prefix_templates = tuple(
        template for template in catalog.templates if template.fact_id in prefix_fact_ids
    )
    prefix_reviews = tuple(
        review for review in catalog.reviews if review.fact_id in prefix_fact_ids
    )
    prefix_rejected = tuple(
        item for item in catalog.rejected_candidates if item.concept_id in prefix_concept_ids
    )
    comparisons = (
        (prefix_concepts, parent.concepts, "concept"),
        (prefix_facts, parent.facts, "fact"),
        (prefix_templates, parent.templates, "template"),
        (prefix_reviews, parent.reviews, "review"),
        (prefix_rejected, parent.rejected_candidates, "rejected-candidate"),
    )
    if any(actual != expected for actual, expected, _ in comparisons):
        changed = next(label for actual, expected, label in comparisons if actual != expected)
        raise ValueError(f"child catalog changed parent {changed} bytes or ordering")
    if (
        catalog.archive_identity != parent.archive_identity
        or catalog.tokenizer_identity != parent.tokenizer_identity
        or catalog.normalization != parent.normalization
    ):
        raise ValueError("child catalog changed a parent source identity")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One authenticated file in a catalog or partition tree."""

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
            raise ValueError("artifact file path must be safe and relative")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("artifact file size must be nonnegative")
        require_sha256(self.sha256, "artifact file")

    def as_record(self) -> dict[str, object]:
        """Return the canonical file identity."""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PartitionCount:
    """Group, occurrence, and token mass for one partition cell."""

    role: PartitionRole
    concept_id: str | None
    split: PartitionSplit | None
    group_count: int
    occurrence_count: int
    token_count: int

    def __post_init__(self) -> None:
        if self.role not in ("base", "node", "construction", "excluded"):
            raise ValueError("partition count role is invalid")
        if self.role == "node":
            if self.concept_id is None or self.split not in ("train", "validation"):
                raise ValueError("node counts require a concept and train/validation split")
        elif self.role == "base":
            if self.concept_id is not None or self.split not in (
                "train",
                "validation",
                "test",
            ):
                raise ValueError("base counts require only a three-way split")
        elif self.concept_id is not None or self.split is not None:
            raise ValueError("construction/excluded counts cannot name concept or split")
        if any(
            type(value) is not int or value < 0
            for value in (self.group_count, self.occurrence_count, self.token_count)
        ):
            raise ValueError("partition counts must be nonnegative")

    def as_record(self) -> dict[str, object]:
        """Return one canonical partition count."""
        return {
            "concept_id": self.concept_id,
            "group_count": self.group_count,
            "occurrence_count": self.occurrence_count,
            "role": self.role,
            "split": self.split,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class QueryPartitionArtifact:
    """Strict loaded query-native partition and its source bindings."""

    root: Path
    partition_sha256: str
    manifest_sha256: str
    catalog_sha256: str
    archive_identity: SourceIdentity
    tokenizer_identity: TokenizerIdentity
    pad_token_id: int
    eos_token_id: int
    concept_ids: tuple[str, ...]
    counts: tuple[PartitionCount, ...]
    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.partition_sha256, "partition"),
            (self.manifest_sha256, "partition manifest"),
            (self.catalog_sha256, "partition catalog"),
        ):
            require_sha256(value, label)
        _unique_tuple(self.concept_ids, "partition concepts")
        if type(self.counts) is not tuple or any(
            type(count) is not PartitionCount for count in self.counts
        ):
            raise TypeError("partition counts must be immutable PartitionCount values")
        if type(self.files) is not tuple or any(
            type(item) is not ArtifactFile for item in self.files
        ):
            raise TypeError("partition files must be immutable ArtifactFile values")
        if any(
            type(value) is not int
            or not 0 <= value < self.tokenizer_identity.vocab_size
            for value in (self.pad_token_id, self.eos_token_id)
        ):
            raise ValueError("partition special token IDs lie outside the vocabulary")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)


@dataclass(frozen=True, slots=True)
class QueryExperimentPreset:
    """Manifest-driven training, evaluation, scaling, and resource choices."""

    concept_ids: tuple[str, ...]
    evaluation_schedule: EvaluationSchedule = "full"
    evaluation_milestones: tuple[int, ...] = ()
    seed: int = 0
    base_epochs: int = 2
    base_maximum_nll: float = 2.2
    base_minimum_epoch_improvement: float = 0.02
    context_length: int = 256
    microbatch_size: int = 32
    accumulation_microbatches: int = 8
    maximum_learning_rate: float = 5e-4
    minimum_learning_rate: float = 5e-5
    warmup_fraction: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    base_weight_decay: float = 0.1
    base_gradient_clip_norm: float = 1.0
    base_state_interval_updates: int = 1_000
    lora_rank: int = 8
    lora_alpha: float = 8.0
    adapter_updates: int = 2_000
    adapter_learning_rate: float = 1e-3
    adapter_weight_decay: float = 0.01
    adapter_gradient_clip_norm: float = 1.0
    freeze_token_embeddings_during_adaptation: bool = True
    pilot_update_budgets: tuple[int, ...] = (500, 1_000, 2_000)
    query_chunk_size: int = 32
    root_probe_count: int = 36
    parent_probe_count: int = 36
    content_key_probe_count: int = 36
    allocator_peak_limit_bytes: int = 12 * 1024**3
    result_size_limit_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        _unique_tuple(self.concept_ids, "experiment concepts")
        for concept_id in self.concept_ids:
            require_identifier(concept_id, "experiment concept")
        if self.evaluation_schedule not in ("full", "milestone"):
            raise ValueError("evaluation schedule must be full or milestone")
        if self.evaluation_schedule == "full":
            if self.evaluation_milestones:
                raise ValueError("full evaluation cannot declare milestones")
            if len(self.concept_ids) > 20:
                raise ValueError("full evaluation is bounded to at most twenty worlds")
        else:
            if (
                type(self.evaluation_milestones) is not tuple
                or tuple(sorted(set(self.evaluation_milestones)))
                != self.evaluation_milestones
                or any(
                    type(stage) is not int
                    or not 1 <= stage <= len(self.concept_ids)
                    for stage in self.evaluation_milestones
                )
            ):
                raise ValueError("milestones must be sorted unique active stages")
        integer_fields = (
            self.seed,
            self.base_epochs,
            self.context_length,
            self.microbatch_size,
            self.accumulation_microbatches,
            self.base_state_interval_updates,
            self.lora_rank,
            self.adapter_updates,
            self.query_chunk_size,
            self.root_probe_count,
            self.parent_probe_count,
            self.content_key_probe_count,
            self.allocator_peak_limit_bytes,
            self.result_size_limit_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields[1:]) or (
            type(self.seed) is not int or self.seed < 0
        ):
            raise ValueError("experiment integer settings must be positive, with nonnegative seed")
        if (
            type(self.pilot_update_budgets) is not tuple
            or tuple(sorted(set(self.pilot_update_budgets)))
            != self.pilot_update_budgets
            or any(type(value) is not int or value <= 0 for value in self.pilot_update_budgets)
        ):
            raise ValueError("pilot budgets must be sorted unique positive updates")
        if any(
            not isfinite(value) or value <= 0.0
            for value in (
                self.base_maximum_nll,
                self.base_minimum_epoch_improvement,
                self.maximum_learning_rate,
                self.minimum_learning_rate,
                self.warmup_fraction,
                self.adam_beta1,
                self.adam_beta2,
                self.adam_epsilon,
                self.base_weight_decay,
                self.base_gradient_clip_norm,
                self.lora_alpha,
                self.adapter_learning_rate,
                self.adapter_weight_decay,
                self.adapter_gradient_clip_norm,
            )
        ):
            raise ValueError("experiment floating settings must be finite and positive")
        if (
            self.seed != 0
            or self.base_epochs != 2
            or self.base_maximum_nll != 2.2
            or self.base_minimum_epoch_improvement != 0.02
            or self.context_length != 256
            or self.microbatch_size != 32
            or self.accumulation_microbatches != 8
            or self.maximum_learning_rate != 5e-4
            or self.minimum_learning_rate != 5e-5
            or self.warmup_fraction != 0.01
            or (self.adam_beta1, self.adam_beta2, self.adam_epsilon)
            != (0.9, 0.95, 1e-8)
            or (self.base_weight_decay, self.base_gradient_clip_norm) != (0.1, 1.0)
            or self.base_state_interval_updates != 1_000
            or (self.lora_rank, self.lora_alpha) != (8, 8.0)
            or self.adapter_updates not in self.pilot_update_budgets
            or (self.adapter_learning_rate, self.adapter_weight_decay, self.adapter_gradient_clip_norm)
            != (1e-3, 0.01, 1.0)
            or self.freeze_token_embeddings_during_adaptation is not True
            or self.pilot_update_budgets != (500, 1_000, 2_000)
            or (self.root_probe_count, self.parent_probe_count, self.content_key_probe_count)
            != (36, 36, 36)
        ):
            raise ValueError("TinyWorlds-Q retained training and pilot settings changed")

    @property
    def active_world_count(self) -> int:
        """Return the manifest-derived active world count."""
        return len(self.concept_ids)

    @property
    def max_nodes(self) -> int:
        """Derive VAMP node capacity from the active manifest."""
        return self.active_world_count + 1

    @property
    def max_edges(self) -> int:
        """Derive VAMP edge capacity from the active manifest."""
        return self.active_world_count

    @property
    def model_config(self) -> GptNeoConfig:
        """Return the retained eight-layer GPT-Neo architecture."""
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

    @property
    def lora_config(self) -> LoraConfig:
        """Return the retained rank-eight all-projection LoRA contract."""
        return LoraConfig(rank=self.lora_rank, alpha=self.lora_alpha)

    @property
    def adapter_train_config(self) -> LmTrainConfig:
        """Return the frozen selected-budget adapter optimizer contract."""
        return LmTrainConfig(
            learning_rate=self.adapter_learning_rate,
            steps=self.adapter_updates,
            batch_size=self.microbatch_size,
            weight_decay=self.adapter_weight_decay,
            gradient_clip_norm=self.adapter_gradient_clip_norm,
        )

    @property
    def config_sha256(self) -> str:
        """Hash every behavior-changing experiment choice."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical dynamic experiment preset."""
        model = self.model_config
        lora = self.lora_config
        adapter = self.adapter_train_config
        scalar_settings = {
            name: list(value) if type(value) is tuple else value
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }
        return {
            **scalar_settings,
            "adapter_train_config": {
                "batch_size": adapter.batch_size,
                "gradient_clip_norm": adapter.gradient_clip_norm,
                "learning_rate": adapter.learning_rate,
                "steps": adapter.steps,
                "weight_decay": adapter.weight_decay,
            },
            "lora_config": {
                "alpha": lora.alpha,
                "rank": lora.rank,
                "target_mask": {
                    name: getattr(lora.target_mask, name)
                    for name in lora.target_mask.__dataclass_fields__
                },
            },
            "model_config": {
                name: list(value) if type(value) is tuple else value
                for name in model.__dataclass_fields__
                for value in (getattr(model, name),)
            },
        }


@dataclass(frozen=True, slots=True)
class SemanticQueryResult:
    """One query-benchmark result projected from shared knowledge scoring."""

    stage: int
    method: str
    concept_id: str
    fact_id: str
    template_id: str
    direction: QueryDirection
    split: QuerySplit
    adapter_concept_id: str | None
    candidate_nll: tuple[float, float, float, float]
    correct_candidate_index: int
    predicted_candidate_index: int
    answer_correct: bool
    correct_answer_margin: float
    selected_node_index: int | None
    oracle_node_index: int | None
    routed_regret: float | None

    def __post_init__(self) -> None:
        if type(self.stage) is not int or self.stage < 0:
            raise ValueError("semantic query result stage must be nonnegative")
        if type(self.method) is not str or not self.method:
            raise ValueError("semantic query result method must be nonempty")
        for value, label in (
            (self.concept_id, "result concept"),
            (self.fact_id, "result fact"),
            (self.template_id, "result template"),
        ):
            require_identifier(value, label)
        if self.direction not in ("forward", "reverse") or self.split not in (
            "validation",
            "test",
        ):
            raise ValueError("semantic result query metadata is invalid")
        if self.adapter_concept_id is not None:
            require_identifier(self.adapter_concept_id, "result adapter concept")
        if (
            type(self.candidate_nll) is not tuple
            or len(self.candidate_nll) != 4
            or any(not isfinite(value) or value < 0.0 for value in self.candidate_nll)
        ):
            raise ValueError("candidate_nll must contain four finite nonnegative values")
        if any(
            type(value) is not int or not 0 <= value < 4
            for value in (self.correct_candidate_index, self.predicted_candidate_index)
        ):
            raise ValueError("semantic result candidate indexes must lie in [0, 3]")
        predicted = min(range(4), key=self.candidate_nll.__getitem__)
        correct = predicted == self.correct_candidate_index
        margin = min(
            value
            for index, value in enumerate(self.candidate_nll)
            if index != self.correct_candidate_index
        ) - self.candidate_nll[self.correct_candidate_index]
        if self.predicted_candidate_index != predicted or self.answer_correct != correct:
            raise ValueError("semantic result prediction fields disagree with candidate NLL")
        if not isfinite(self.correct_answer_margin) or not _close(
            self.correct_answer_margin,
            margin,
        ):
            raise ValueError("semantic result margin disagrees with candidate NLL")
        for value, label in (
            (self.selected_node_index, "selected node"),
            (self.oracle_node_index, "oracle node"),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{label} must be nonnegative when present")
        if self.routed_regret is not None and not isfinite(self.routed_regret):
            raise ValueError("routed regret must be finite when present")

    def as_record(self) -> dict[str, object]:
        """Return one streamable canonical result row."""
        return {
            "answer_correct": self.answer_correct,
            "adapter_concept_id": self.adapter_concept_id,
            "candidate_nll": list(self.candidate_nll),
            "concept_id": self.concept_id,
            "correct_answer_margin": self.correct_answer_margin,
            "correct_candidate_index": self.correct_candidate_index,
            "direction": self.direction,
            "fact_id": self.fact_id,
            "format": RESULT_FORMAT,
            "method": self.method,
            "oracle_node_index": self.oracle_node_index,
            "predicted_candidate_index": self.predicted_candidate_index,
            "routed_regret": self.routed_regret,
            "selected_node_index": self.selected_node_index,
            "split": self.split,
            "stage": self.stage,
            "template_id": self.template_id,
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> SemanticQueryResult:
        """Reconstruct and validate one canonical result-ledger row."""
        if type(record) is not dict:
            raise ValueError("semantic result record must be a JSON object")
        required = {
            "adapter_concept_id",
            "answer_correct",
            "candidate_nll",
            "concept_id",
            "correct_answer_margin",
            "correct_candidate_index",
            "direction",
            "fact_id",
            "format",
            "method",
            "oracle_node_index",
            "predicted_candidate_index",
            "routed_regret",
            "selected_node_index",
            "split",
            "stage",
            "template_id",
        }
        candidate_nll = record.get("candidate_nll")
        numeric_fields = (
            record.get("correct_answer_margin"),
            record.get("routed_regret"),
        )
        optional_indexes = (
            record.get("selected_node_index"),
            record.get("oracle_node_index"),
        )
        if (
            set(record) != required
            or record.get("format") != RESULT_FORMAT
            or type(candidate_nll) is not list
            or len(candidate_nll) != 4
            or any(type(value) not in (int, float) for value in candidate_nll)
            or type(record.get("stage")) is not int
            or type(record.get("correct_candidate_index")) is not int
            or type(record.get("predicted_candidate_index")) is not int
            or type(record.get("answer_correct")) is not bool
            or type(numeric_fields[0]) not in (int, float)
            or (
                numeric_fields[1] is not None
                and type(numeric_fields[1]) not in (int, float)
            )
            or any(
                value is not None and type(value) is not int
                for value in optional_indexes
            )
            or any(
                type(record.get(field)) is not str
                for field in (
                    "concept_id",
                    "direction",
                    "fact_id",
                    "method",
                    "split",
                    "template_id",
                )
            )
            or (
                record.get("adapter_concept_id") is not None
                and type(record.get("adapter_concept_id")) is not str
            )
        ):
            raise ValueError("semantic result record fields changed")
        return cls(
            stage=record["stage"],  # type: ignore[arg-type]
            method=record["method"],  # type: ignore[arg-type]
            concept_id=record["concept_id"],  # type: ignore[arg-type]
            fact_id=record["fact_id"],  # type: ignore[arg-type]
            template_id=record["template_id"],  # type: ignore[arg-type]
            direction=record["direction"],  # type: ignore[arg-type]
            split=record["split"],  # type: ignore[arg-type]
            adapter_concept_id=record["adapter_concept_id"],  # type: ignore[arg-type]
            candidate_nll=tuple(candidate_nll),  # type: ignore[arg-type]
            correct_candidate_index=record["correct_candidate_index"],  # type: ignore[arg-type]
            predicted_candidate_index=record["predicted_candidate_index"],  # type: ignore[arg-type]
            answer_correct=record["answer_correct"],  # type: ignore[arg-type]
            correct_answer_margin=float(numeric_fields[0]),
            selected_node_index=optional_indexes[0],  # type: ignore[arg-type]
            oracle_node_index=optional_indexes[1],  # type: ignore[arg-type]
            routed_regret=(
                None if numeric_fields[1] is None else float(numeric_fields[1])
            ),
        )


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right))


__all__ = [
    "ArtifactFile",
    "BENCHMARK_ID",
    "CATALOG_FORMAT",
    "CATALOG_ROOT",
    "CHECKPOINT_ROOT",
    "CONSTRUCTION_BUCKET",
    "CONSTRUCTION_BUCKET_COUNT",
    "CONSTRUCTION_NAMESPACE",
    "ConceptDefinition",
    "EvaluationSchedule",
    "FactReviewDecision",
    "PARTITION_FORMAT",
    "PartitionCount",
    "QueryDirection",
    "QueryExperimentPreset",
    "QueryPartitionArtifact",
    "QuerySplit",
    "RESULT_ROOT",
    "RejectedFactCandidate",
    "SemanticFact",
    "SemanticQueryCatalog",
    "SemanticQueryResult",
    "SemanticQueryTemplate",
    "StoryProvenance",
    "canonical_json_bytes",
    "record_sha256",
    "registered_distractor_order",
    "require_identifier",
    "require_sha256",
    "validate_parent_catalog_prefix",
]
