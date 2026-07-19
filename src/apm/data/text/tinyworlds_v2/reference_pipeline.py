"""Strict Phase 1 preparation of TinyStories references and neutral briefs.

The source archive remains authoritative.  This module only joins its three
disjoint cohorts to the validation cohort, summarizes released ingredients,
and combines injected tokenizer/checkpoint measurements with deterministic
surface observations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import chain, groupby
import math
import multiprocessing
import unicodedata

from apm.data.text.tinyworlds_v2.bakeoff import NeutralStoryBrief
from apm.data.text.tinyworlds_v2.json_contracts import JsonObject, json_sha256
from apm.data.text.tinyworlds_v2.ingredients import (
    IngredientRoles,
    mechanically_classify_ingredient_roles,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceRecord,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import (
    ArchiveSourceRecord,
    ArchiveSourceSelections,
    ValidationStoryRecord,
)
from apm.data.text.tinyworlds_v2.surface import canonical_feature_labels


PHASE1_PROMPT_METADATA_COUNT = 10_000
PHASE1_ARCHIVE_REFERENCE_COUNT = 10_000
PHASE1_VALIDATION_REFERENCE_COUNT = 10_000
PHASE1_BRIEF_COUNT = 200
REFERENCE_SURFACE_WORKERS = 16

_INGREDIENT_ROLES = ("noun", "verb", "adjective")


class ReferencePipelineError(ValueError):
    """A Phase 1 source cohort or injected observation is incomplete."""


@dataclass(frozen=True, slots=True)
class IngredientPositionFrequencies:
    """Empirical released-word frequencies at one zero-based list position."""

    position: int
    frequencies: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ValueError("ingredient position must be nonnegative")
        _validate_frequency_table(self.frequencies, "ingredient position")


@dataclass(frozen=True, slots=True)
class PromptIngredientProfile:
    """Empirical word-role, raw-position, and narrative-feature distributions."""

    record_count: int
    parsed_role_record_count: int
    unparsed_role_record_count: int
    noun_frequencies: tuple[tuple[str, int], ...]
    verb_frequencies: tuple[tuple[str, int], ...]
    adjective_frequencies: tuple[tuple[str, int], ...]
    word_position_frequencies: tuple[IngredientPositionFrequencies, ...]
    unparsed_word_position_frequencies: tuple[IngredientPositionFrequencies, ...]
    narrative_feature_frequencies: tuple[tuple[str, int], ...]
    narrative_feature_rates: tuple[tuple[str, float], ...]
    narrative_feature_count_frequencies: tuple[tuple[int, int], ...]
    any_narrative_feature_rate: float
    mean_narrative_feature_count: float
    source_record_ids_sha256: str
    profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.record_count) is not int or self.record_count <= 0:
            raise ValueError("ingredient profile record_count must be positive")
        counts = (self.parsed_role_record_count, self.unparsed_role_record_count)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("ingredient role record counts must be nonnegative")
        if sum(counts) != self.record_count:
            raise ValueError("parsed and unparsed role counts must cover the profile")
        for label, frequencies in (
            ("noun", self.noun_frequencies),
            ("verb", self.verb_frequencies),
            ("adjective", self.adjective_frequencies),
            ("narrative feature", self.narrative_feature_frequencies),
        ):
            _validate_frequency_table(
                frequencies,
                label,
                allow_empty=label == "narrative feature" or not counts[0],
            )
        if any(
            sum(count for _, count in frequencies) != self.parsed_role_record_count
            for frequencies in (
                self.noun_frequencies,
                self.verb_frequencies,
                self.adjective_frequencies,
            )
        ):
            raise ValueError("each parsed role must contribute exactly one word")
        _validate_position_tables(
            self.word_position_frequencies,
            "word position",
        )
        _validate_position_tables(
            self.unparsed_word_position_frequencies,
            "unparsed word position",
        )
        _validate_feature_counts(self.narrative_feature_count_frequencies)
        if (
            sum(count for _, count in self.narrative_feature_count_frequencies)
            != self.record_count
        ):
            raise ValueError("feature-count frequencies must cover every prompt record")
        expected_feature_rates = tuple(
            (label, count / self.record_count)
            for label, count in self.narrative_feature_frequencies
        )
        if self.narrative_feature_rates != expected_feature_rates:
            raise ValueError("narrative feature rates do not match their frequencies")
        prompts_with_features = sum(
            frequency
            for feature_count, frequency in self.narrative_feature_count_frequencies
            if feature_count > 0
        )
        total_features = sum(
            feature_count * frequency
            for feature_count, frequency in self.narrative_feature_count_frequencies
        )
        expected_rates = (
            prompts_with_features / self.record_count,
            total_features / self.record_count,
        )
        if not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
            for actual, expected in zip(
                (
                    self.any_narrative_feature_rate,
                    self.mean_narrative_feature_count,
                ),
                expected_rates,
                strict=True,
            )
        ):
            raise ValueError("aggregate narrative feature rates are inconsistent")
        _require_sha256(self.source_record_ids_sha256, "source-record IDs")
        object.__setattr__(
            self,
            "profile_sha256",
            json_sha256(_ingredient_profile_payload(self)),
        )

    @property
    def parsed_role_rate(self) -> float:
        """Return the fraction of prompts with unambiguous explicit role labels."""
        return self.parsed_role_record_count / self.record_count


@dataclass(frozen=True, slots=True)
class ReferenceAnnotation:
    """Raw source-side ingredients accompanying one reference observation.

    Released TinyStories metadata occasionally repeats a word or narrative
    feature.  This record preserves that evidence exactly; presence-based
    measurements canonicalize feature labels only at the observation boundary.
    """

    record_id: str
    source_partition: str
    required_words: tuple[str, ...]
    feature_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.record_id, "reference annotation record_id")
        if self.source_partition not in ("archive", "validation"):
            raise ValueError("reference source_partition must be archive or validation")
        for label, values in (
            ("required_words", self.required_words),
            ("feature_labels", self.feature_labels),
        ):
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise ValueError(f"reference annotation {label} must be nonempty strings")
        if self.source_partition == "validation" and (
            self.required_words or self.feature_labels
        ):
            raise ValueError("validation references cannot invent released ingredients")


@dataclass(frozen=True, slots=True)
class Phase1ReferenceInputs:
    """The fixed-size, fully paired inputs to the Phase 1 model bakeoff."""

    briefs: tuple[NeutralStoryBrief, ...]
    reference_records: tuple[ReferenceRecord, ...]
    reference_annotations: tuple[ReferenceAnnotation, ...]
    ingredient_profile: PromptIngredientProfile

    def __post_init__(self) -> None:
        if type(self.briefs) is not tuple:
            raise TypeError("briefs must be a tuple")
        if type(self.reference_records) is not tuple:
            raise TypeError("reference_records must be a tuple")
        if type(self.reference_annotations) is not tuple:
            raise TypeError("reference_annotations must be a tuple")
        if type(self.ingredient_profile) is not PromptIngredientProfile:
            raise TypeError("ingredient_profile must be a PromptIngredientProfile")
        expected_counts = (
            ("briefs", len(self.briefs), PHASE1_BRIEF_COUNT),
            (
                "reference records",
                len(self.reference_records),
                PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT,
            ),
            (
                "reference annotations",
                len(self.reference_annotations),
                PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT,
            ),
            (
                "prompt ingredient records",
                self.ingredient_profile.record_count,
                PHASE1_PROMPT_METADATA_COUNT,
            ),
        )
        for label, actual, expected in expected_counts:
            if actual != expected:
                raise ReferencePipelineError(
                    f"Phase 1 requires exactly {expected} {label}, got {actual}"
                )
        if any(type(brief) is not NeutralStoryBrief for brief in self.briefs):
            raise TypeError("briefs must contain NeutralStoryBrief values")
        if any(
            type(record) is not ReferenceRecord for record in self.reference_records
        ):
            raise TypeError("reference_records must contain ReferenceRecord values")
        if any(
            type(annotation) is not ReferenceAnnotation
            for annotation in self.reference_annotations
        ):
            raise TypeError("reference_annotations must contain ReferenceAnnotation values")
        brief_ids = tuple(brief.brief_id for brief in self.briefs)
        record_ids = tuple(record.record_id for record in self.reference_records)
        annotation_ids = tuple(
            annotation.record_id for annotation in self.reference_annotations
        )
        if brief_ids != tuple(sorted(brief_ids)) or len(set(brief_ids)) != len(brief_ids):
            raise ValueError("briefs must have unique IDs in canonical order")
        if record_ids != tuple(sorted(record_ids)) or len(set(record_ids)) != len(record_ids):
            raise ValueError("reference records must have unique IDs in canonical order")
        if annotation_ids != record_ids:
            raise ValueError("reference annotations must align with canonical record IDs")
        partition_counts = Counter(
            annotation.source_partition for annotation in self.reference_annotations
        )
        if partition_counts != {
            "archive": PHASE1_ARCHIVE_REFERENCE_COUNT,
            "validation": PHASE1_VALIDATION_REFERENCE_COUNT,
        }:
            raise ValueError("reference partitions must contain 10k records apiece")


@dataclass(frozen=True, slots=True)
class _SurfaceInput:
    record: ReferenceRecord
    annotation: ReferenceAnnotation
    model_token_ids: tuple[int, ...]
    normalized_nll: float


def build_neutral_story_brief(record: ArchiveSourceRecord) -> NeutralStoryBrief:
    """Convert one paired released record into a stable, world-free brief."""
    if type(record) is not ArchiveSourceRecord:
        raise TypeError("record must be an ArchiveSourceRecord")
    roles = mechanically_classify_ingredient_roles(
        record.instruction.prompt,
        record.instruction.words,
    )
    if roles is None:
        raise ReferencePipelineError(
            f"paired record {record.record_id!r} must have exactly three unique "
            "released words with explicit noun, verb, and adjective labels"
        )
    opaque_digest = sha256(
        f"tinyworlds-v2-phase1-brief\0{record.record_id}".encode("utf-8")
    ).hexdigest()
    return NeutralStoryBrief(
        brief_id=f"brief-{opaque_digest[:24]}",
        source_record_id=record.record_id,
        prompt_text=record.instruction.prompt,
        required_words=record.instruction.words,
        requested_features=canonical_feature_labels(record.instruction.features),
        matched_reference_text=record.story,
    )


def build_prompt_ingredient_profile(
    records: Sequence[ArchiveSourceRecord],
) -> PromptIngredientProfile:
    """Aggregate only empirical ingredients present in released prompt metadata."""
    if not records:
        raise ValueError("ingredient profile requires at least one prompt record")
    if any(type(record) is not ArchiveSourceRecord for record in records):
        raise TypeError("ingredient records must be ArchiveSourceRecord values")
    ordered = tuple(sorted(records, key=lambda record: record.record_id))
    if len({record.record_id for record in ordered}) != len(ordered):
        raise ValueError("ingredient prompt record IDs must be unique")
    classifications = tuple(
        mechanically_classify_ingredient_roles(
            record.instruction.prompt,
            record.instruction.words,
        )
        for record in ordered
    )
    role_frequencies = tuple(
        tuple(
            sorted(
                Counter(
                    _normalize_ingredient(getattr(classification, role))
                    for classification in classifications
                    if classification is not None
                ).items()
            )
        )
        for role in _INGREDIENT_ROLES
    )
    position_frequencies = _position_frequencies(
        tuple(
            tuple(_normalize_ingredient(word) for word in record.instruction.words)
            for record in ordered
        )
    )
    unparsed_position_frequencies = _position_frequencies(
        tuple(
            tuple(_normalize_ingredient(word) for word in record.instruction.words)
            for record, classification in zip(ordered, classifications, strict=True)
            if classification is None
        )
    )
    canonical_features = tuple(
        canonical_feature_labels(record.instruction.features) for record in ordered
    )
    feature_frequencies = tuple(
        sorted(
            Counter(
                feature
                for features in canonical_features
                for feature in features
            ).items()
        )
    )
    feature_count_frequencies = tuple(
        sorted(Counter(len(features) for features in canonical_features).items())
    )
    parsed_count = sum(classification is not None for classification in classifications)
    profile_values = {
        "record_count": len(ordered),
        "parsed_role_record_count": parsed_count,
        "unparsed_role_record_count": len(ordered) - parsed_count,
        "noun_frequencies": role_frequencies[0],
        "verb_frequencies": role_frequencies[1],
        "adjective_frequencies": role_frequencies[2],
        "word_position_frequencies": position_frequencies,
        "unparsed_word_position_frequencies": unparsed_position_frequencies,
        "narrative_feature_frequencies": feature_frequencies,
        "narrative_feature_rates": tuple(
            (label, count / len(ordered)) for label, count in feature_frequencies
        ),
        "narrative_feature_count_frequencies": feature_count_frequencies,
        "any_narrative_feature_rate": sum(
            bool(features) for features in canonical_features
        )
        / len(ordered),
        "mean_narrative_feature_count": sum(
            len(features) for features in canonical_features
        )
        / len(ordered),
        "source_record_ids_sha256": json_sha256(
            [record.record_id for record in ordered]
        ),
    }
    return PromptIngredientProfile(**profile_values)


def build_phase1_reference_inputs(
    selections: ArchiveSourceSelections,
    validation_records: Sequence[ValidationStoryRecord],
) -> Phase1ReferenceInputs:
    """Build the fixed 200-brief, 20k-reference Phase 1 input contract."""
    if type(selections) is not ArchiveSourceSelections:
        raise TypeError("selections must be ArchiveSourceSelections")
    if any(type(record) is not ValidationStoryRecord for record in validation_records):
        raise TypeError("validation_records must contain ValidationStoryRecord values")
    actual_counts = (
        (
            "prompt metadata",
            len(selections.prompt_metadata_records),
            PHASE1_PROMPT_METADATA_COUNT,
        ),
        (
            "archive reference stories",
            len(selections.reference_story_records),
            PHASE1_ARCHIVE_REFERENCE_COUNT,
        ),
        ("paired stories", len(selections.paired_records), PHASE1_BRIEF_COUNT),
        (
            "validation reference stories",
            len(validation_records),
            PHASE1_VALIDATION_REFERENCE_COUNT,
        ),
    )
    for label, actual, expected in actual_counts:
        if actual != expected:
            raise ReferencePipelineError(
                f"Phase 1 requires exactly {expected} {label}, got {actual}"
            )

    archive_story_hashes = {
        record.normalized_story_sha256
        for cohort in (
            selections.prompt_metadata_records,
            selections.reference_story_records,
            selections.paired_records,
        )
        for record in cohort
    }
    validation_story_hashes = tuple(
        record.normalized_story_sha256 for record in validation_records
    )
    if len(set(validation_story_hashes)) != len(validation_story_hashes):
        raise ReferencePipelineError(
            "validation references must be unique by normalized story content"
        )
    if archive_story_hashes.intersection(validation_story_hashes):
        raise ReferencePipelineError(
            "archive and validation cohorts must be disjoint by normalized story content"
        )

    briefs = tuple(
        sorted(
            (build_neutral_story_brief(record) for record in selections.paired_records),
            key=lambda brief: brief.brief_id,
        )
    )
    archive_pairs = tuple(
        _archive_reference_pair(record)
        for record in selections.reference_story_records
    )
    validation_pairs = tuple(
        _validation_reference_pair(record) for record in validation_records
    )
    ordered_pairs = tuple(
        sorted(archive_pairs + validation_pairs, key=lambda pair: pair[0].record_id)
    )
    return Phase1ReferenceInputs(
        briefs=briefs,
        reference_records=tuple(pair[0] for pair in ordered_pairs),
        reference_annotations=tuple(pair[1] for pair in ordered_pairs),
        ingredient_profile=build_prompt_ingredient_profile(
            selections.prompt_metadata_records
        ),
    )


def prepare_reference_observations(
    records: Sequence[ReferenceRecord],
    annotations: Sequence[ReferenceAnnotation],
    *,
    model_token_ids_by_record_id: Mapping[str, tuple[int, ...]],
    normalized_nll_by_record_id: Mapping[str, float],
    worker_count: int = REFERENCE_SURFACE_WORKERS,
) -> tuple[ReferenceObservation, ...]:
    """Build observations in deterministic shards and merge by stable record ID."""
    if not records:
        raise ValueError("surface observation preparation requires reference records")
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("surface worker_count must be positive")
    if any(type(record) is not ReferenceRecord for record in records):
        raise TypeError("records must contain ReferenceRecord values")
    if any(type(annotation) is not ReferenceAnnotation for annotation in annotations):
        raise TypeError("annotations must contain ReferenceAnnotation values")
    ordered_records = tuple(sorted(records, key=lambda record: record.record_id))
    record_ids = tuple(record.record_id for record in ordered_records)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("reference record IDs must be unique")
    annotation_by_id = _index_annotations(annotations)
    expected_ids = frozenset(record_ids)
    _require_exact_mapping_ids("annotations", annotation_by_id, expected_ids)
    _require_exact_mapping_ids(
        "model token IDs",
        model_token_ids_by_record_id,
        expected_ids,
    )
    _require_exact_mapping_ids(
        "normalized NLL",
        normalized_nll_by_record_id,
        expected_ids,
    )
    surface_inputs = tuple(
        _surface_input(
            record,
            annotation_by_id[record.record_id],
            model_token_ids_by_record_id[record.record_id],
            normalized_nll_by_record_id[record.record_id],
        )
        for record in ordered_records
    )
    shard_key = lambda item: (
        _surface_shard_index(item.record.record_id, worker_count),
        item.record.record_id,
    )
    sorted_inputs = tuple(sorted(surface_inputs, key=shard_key))
    shards = tuple(
        tuple(group)
        for _, group in groupby(
            sorted_inputs,
            key=lambda item: _surface_shard_index(item.record.record_id, worker_count),
        )
    )
    if len(shards) == 1 or worker_count == 1:
        completed_shards = tuple(_observe_surface_shard(shard) for shard in shards)
    else:
        # Production computes checkpoint NLL on JAX before this surface pass.
        # Never fork a live accelerator runtime: spawned workers receive only
        # the immutable text/statistic inputs and cannot inherit GPU threads.
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(shards)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = tuple(
                executor.submit(_observe_surface_shard, shard) for shard in shards
            )
            completed_shards = tuple(
                future.result() for future in as_completed(futures)
            )
    return tuple(
        sorted(chain.from_iterable(completed_shards), key=lambda item: item.record_id)
    )


def canonical_neutral_story_brief(brief: NeutralStoryBrief) -> JsonObject:
    """Return one neutral brief as a strict JSON-serializable record."""
    if type(brief) is not NeutralStoryBrief:
        raise TypeError("brief must be a NeutralStoryBrief")
    return {
        "brief_id": brief.brief_id,
        "matched_reference_text": brief.matched_reference_text,
        "prompt_text": brief.prompt_text,
        "requested_features": list(brief.requested_features),
        "required_words": list(brief.required_words),
        "source_record_id": brief.source_record_id,
    }


def canonical_reference_record(record: ReferenceRecord) -> JsonObject:
    """Return one genuine reference as a strict JSON-serializable record."""
    if type(record) is not ReferenceRecord:
        raise TypeError("record must be a ReferenceRecord")
    return {
        "prompt_text": record.prompt_text,
        "record_id": record.record_id,
        "source_model": record.source_model,
        "story_text": record.story_text,
    }


def canonical_reference_annotation(annotation: ReferenceAnnotation) -> JsonObject:
    """Return source ingredients for one reference as a serializable record."""
    if type(annotation) is not ReferenceAnnotation:
        raise TypeError("annotation must be a ReferenceAnnotation")
    return {
        "feature_labels": list(annotation.feature_labels),
        "record_id": annotation.record_id,
        "required_words": list(annotation.required_words),
        "source_partition": annotation.source_partition,
    }


def canonical_prompt_ingredient_profile(profile: PromptIngredientProfile) -> JsonObject:
    """Return the complete ingredient profile, including its content digest."""
    if type(profile) is not PromptIngredientProfile:
        raise TypeError("profile must be a PromptIngredientProfile")
    return {
        **_ingredient_profile_payload(profile),
        "profile_sha256": profile.profile_sha256,
    }


def canonical_reference_observation(observation: ReferenceObservation) -> JsonObject:
    """Return one merged surface/tokenizer/NLL observation as a JSON record."""
    if type(observation) is not ReferenceObservation:
        raise TypeError("observation must be a ReferenceObservation")
    return {
        "dialogue_present": observation.dialogue_present,
        "ending_key": observation.ending_key,
        "realized_feature_labels": list(observation.realized_feature_labels),
        "requested_feature_labels": list(observation.feature_labels),
        "model_token_ids": list(observation.model_token_ids),
        "normalized_nll": observation.normalized_nll,
        "opening_key": observation.opening_key,
        "paragraph_count": observation.paragraph_count,
        "record_id": observation.record_id,
        "repeated_ngram_fraction": observation.repeated_ngram_fraction,
        "required_words": list(observation.required_words),
        "sentence_word_counts": list(observation.sentence_word_counts),
        "word_tokens": list(observation.word_tokens),
    }


def _archive_reference_pair(
    record: ArchiveSourceRecord,
) -> tuple[ReferenceRecord, ReferenceAnnotation]:
    prompt = record.instruction.prompt if record.instruction.prompt.strip() else None
    return (
        ReferenceRecord(
            record_id=record.record_id,
            story_text=record.story,
            prompt_text=prompt,
            source_model=record.source,
        ),
        ReferenceAnnotation(
            record_id=record.record_id,
            source_partition="archive",
            required_words=record.instruction.words,
            feature_labels=record.instruction.features,
        ),
    )


def _validation_reference_pair(
    record: ValidationStoryRecord,
) -> tuple[ReferenceRecord, ReferenceAnnotation]:
    return (
        ReferenceRecord(
            record_id=record.record_id,
            story_text=record.story,
            prompt_text=None,
            source_model="GPT-4",
        ),
        ReferenceAnnotation(
            record_id=record.record_id,
            source_partition="validation",
            required_words=(),
            feature_labels=(),
        ),
    )


def _position_frequencies(
    word_rows: tuple[tuple[str, ...], ...],
) -> tuple[IngredientPositionFrequencies, ...]:
    width = max((len(row) for row in word_rows), default=0)
    return tuple(
        IngredientPositionFrequencies(
            position=position,
            frequencies=tuple(
                sorted(
                    Counter(
                        row[position] for row in word_rows if position < len(row)
                    ).items()
                )
            ),
        )
        for position in range(width)
    )


def _surface_input(
    record: ReferenceRecord,
    annotation: ReferenceAnnotation,
    model_token_ids: tuple[int, ...],
    normalized_nll: float,
) -> _SurfaceInput:
    if type(model_token_ids) is not tuple or not model_token_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in model_token_ids
    ):
        raise ReferencePipelineError(
            f"model token IDs for {record.record_id!r} must be a nonempty integer tuple"
        )
    if (
        type(normalized_nll) is not float
        or not math.isfinite(normalized_nll)
        or normalized_nll < 0.0
    ):
        raise ReferencePipelineError(
            f"normalized NLL for {record.record_id!r} must be a finite nonnegative float"
        )
    return _SurfaceInput(record, annotation, model_token_ids, normalized_nll)


def _observe_surface_shard(
    shard: tuple[_SurfaceInput, ...],
) -> tuple[ReferenceObservation, ...]:
    return tuple(
        observe_reference(
            item.record,
            model_token_ids=item.model_token_ids,
            normalized_nll=item.normalized_nll,
            feature_labels=item.annotation.feature_labels,
            required_words=item.annotation.required_words,
        )
        for item in shard
    )


def _index_annotations(
    annotations: Sequence[ReferenceAnnotation],
) -> dict[str, ReferenceAnnotation]:
    indexed = {annotation.record_id: annotation for annotation in annotations}
    if len(indexed) != len(annotations):
        raise ValueError("reference annotation IDs must be unique")
    return indexed


def _require_exact_mapping_ids(
    label: str,
    values: Mapping[str, object],
    expected_ids: frozenset[str],
) -> None:
    if any(type(record_id) is not str for record_id in values):
        raise TypeError(f"{label} mapping keys must be strings")
    actual_ids = frozenset(values)
    if actual_ids != expected_ids:
        missing = tuple(sorted(expected_ids - actual_ids))
        unexpected = tuple(sorted(actual_ids - expected_ids))
        raise ReferencePipelineError(
            f"{label} record IDs differ; missing={missing[:5]!r} "
            f"unexpected={unexpected[:5]!r}"
        )


def _surface_shard_index(record_id: str, worker_count: int) -> int:
    digest = sha256(f"tinyworlds-v2-surface\0{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % worker_count


def _ingredient_profile_payload(profile: PromptIngredientProfile) -> JsonObject:
    positions = lambda values: [
        {
            "frequencies": [[word, count] for word, count in item.frequencies],
            "position": item.position,
        }
        for item in values
    ]
    return {
        "adjective_frequencies": [list(item) for item in profile.adjective_frequencies],
        "any_narrative_feature_rate": profile.any_narrative_feature_rate,
        "mean_narrative_feature_count": profile.mean_narrative_feature_count,
        "narrative_feature_count_frequencies": [
            list(item) for item in profile.narrative_feature_count_frequencies
        ],
        "narrative_feature_frequencies": [
            list(item) for item in profile.narrative_feature_frequencies
        ],
        "narrative_feature_rates": [
            list(item) for item in profile.narrative_feature_rates
        ],
        "noun_frequencies": [list(item) for item in profile.noun_frequencies],
        "parsed_role_record_count": profile.parsed_role_record_count,
        "record_count": profile.record_count,
        "source_record_ids_sha256": profile.source_record_ids_sha256,
        "unparsed_role_record_count": profile.unparsed_role_record_count,
        "unparsed_word_position_frequencies": positions(
            profile.unparsed_word_position_frequencies
        ),
        "verb_frequencies": [list(item) for item in profile.verb_frequencies],
        "word_position_frequencies": positions(profile.word_position_frequencies),
    }


def _validate_frequency_table(
    frequencies: tuple[tuple[str, int], ...],
    label: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(frequencies) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or not item[0].strip()
        or type(item[1]) is not int
        or item[1] <= 0
        for item in frequencies
    ):
        raise ValueError(f"{label} frequencies must be positive (label, count) pairs")
    if not frequencies and not allow_empty:
        raise ValueError(f"{label} frequencies must not be empty")
    if frequencies != tuple(sorted(frequencies)) or len(
        {item[0] for item in frequencies}
    ) != len(frequencies):
        raise ValueError(f"{label} frequencies must have unique labels in order")


def _validate_position_tables(
    positions: tuple[IngredientPositionFrequencies, ...],
    label: str,
) -> None:
    if type(positions) is not tuple or any(
        type(item) is not IngredientPositionFrequencies for item in positions
    ):
        raise TypeError(f"{label} tables must be IngredientPositionFrequencies")
    if tuple(item.position for item in positions) != tuple(range(len(positions))):
        raise ValueError(f"{label} tables must be contiguous and ordered")


def _validate_feature_counts(frequencies: tuple[tuple[int, int], ...]) -> None:
    if type(frequencies) is not tuple or not frequencies or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or item[0] < 0
        or type(item[1]) is not int
        or item[1] <= 0
        for item in frequencies
    ):
        raise ValueError("feature-count frequencies must be nonnegative integer pairs")
    if frequencies != tuple(sorted(frequencies)) or len(
        {item[0] for item in frequencies}
    ) != len(frequencies):
        raise ValueError("feature-count frequencies must be unique and ordered")


def _normalize_ingredient(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


def _require_text(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _require_sha256(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


__all__ = [
    "IngredientPositionFrequencies",
    "IngredientRoles",
    "PHASE1_ARCHIVE_REFERENCE_COUNT",
    "PHASE1_BRIEF_COUNT",
    "PHASE1_PROMPT_METADATA_COUNT",
    "PHASE1_VALIDATION_REFERENCE_COUNT",
    "Phase1ReferenceInputs",
    "PromptIngredientProfile",
    "REFERENCE_SURFACE_WORKERS",
    "ReferenceAnnotation",
    "ReferencePipelineError",
    "build_neutral_story_brief",
    "build_phase1_reference_inputs",
    "build_prompt_ingredient_profile",
    "canonical_neutral_story_brief",
    "canonical_prompt_ingredient_profile",
    "canonical_reference_annotation",
    "canonical_reference_observation",
    "canonical_reference_record",
    "mechanically_classify_ingredient_roles",
    "prepare_reference_observations",
]
