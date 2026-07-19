"""Deterministic TinyStories reference sampling and distribution profiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import accumulate
import json
import math
import re
from statistics import median

from apm.data.text.tinyworlds_v2.surface import (
    canonical_feature_labels,
    lexical_tokens,
    realized_feature_labels,
    repeated_ngram_fraction,
    token_form_counts,
)

_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_SENTENCE_PATTERN = re.compile(r"[^.!?]+(?:[.!?]+|$)")


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """One genuine TinyStories story and its optional generation prompt."""

    record_id: str
    story_text: str
    prompt_text: str | None = None
    source_model: str = "gpt-4"

    def __post_init__(self) -> None:
        _require_text(self.record_id, "record_id")
        _require_text(self.story_text, "story_text")
        if self.prompt_text is not None:
            _require_text(self.prompt_text, "prompt_text")
        _require_text(self.source_model, "source_model")


@dataclass(frozen=True, slots=True)
class ReferenceObservation:
    """Offline surface, tokenizer, and NLL measurements for one story."""

    record_id: str
    word_tokens: tuple[str, ...]
    model_token_ids: tuple[int, ...]
    sentence_word_counts: tuple[int, ...]
    paragraph_count: int
    dialogue_present: bool
    opening_key: str
    ending_key: str
    feature_labels: tuple[str, ...]
    normalized_nll: float
    required_words: tuple[str, ...] = ()
    realized_feature_labels: tuple[str, ...] = ()
    repeated_ngram_fraction: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.record_id, "record_id")
        if not self.word_tokens or any(type(token) is not str for token in self.word_tokens):
            raise ValueError("word_tokens must contain at least one string")
        if not self.model_token_ids or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.model_token_ids
        ):
            raise ValueError("model_token_ids must contain nonnegative integers")
        if not self.sentence_word_counts or any(
            type(count) is not int or count <= 0
            for count in self.sentence_word_counts
        ):
            raise ValueError("sentence_word_counts must contain positive integers")
        if type(self.paragraph_count) is not int or self.paragraph_count <= 0:
            raise ValueError("paragraph_count must be positive")
        if type(self.dialogue_present) is not bool:
            raise TypeError("dialogue_present must be a bool")
        _require_text(self.opening_key, "opening_key")
        _require_text(self.ending_key, "ending_key")
        if type(self.feature_labels) is not tuple or any(
            type(label) is not str or not label.strip()
            for label in self.feature_labels
        ):
            raise ValueError("feature_labels must contain nonempty strings")
        if len(set(self.feature_labels)) != len(self.feature_labels):
            raise ValueError("feature_labels must be unique")
        if type(self.required_words) is not tuple or any(
            type(word) is not str or not word.strip() for word in self.required_words
        ):
            raise ValueError("required_words must contain nonempty strings")
        if type(self.realized_feature_labels) is not tuple or any(
            type(label) is not str or not label.strip()
            for label in self.realized_feature_labels
        ):
            raise ValueError(
                "realized_feature_labels must contain nonempty strings"
            )
        if len(set(self.realized_feature_labels)) != len(
            self.realized_feature_labels
        ):
            raise ValueError("realized_feature_labels must be unique")
        if not set(self.realized_feature_labels).issubset(self.feature_labels):
            raise ValueError("realized features must be a subset of requested features")
        if not math.isfinite(self.repeated_ngram_fraction) or not (
            0.0 <= self.repeated_ngram_fraction <= 1.0
        ):
            raise ValueError("repeated_ngram_fraction must be between zero and one")
        if (
            type(self.normalized_nll) is not float
            or not math.isfinite(self.normalized_nll)
            or self.normalized_nll < 0.0
        ):
            raise ValueError("normalized_nll must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Aggregate reference distributions used by all quality gates."""

    record_count: int
    vocabulary: frozenset[str]
    word_frequencies: tuple[tuple[str, int], ...]
    required_word_frequencies: tuple[tuple[str, int], ...]
    token_probabilities: tuple[tuple[int, float], ...]
    story_word_counts: tuple[int, ...]
    model_token_counts: tuple[int, ...]
    sentence_word_counts: tuple[int, ...]
    paragraph_break_rate: float
    dialogue_rate: float
    feature_rates: tuple[tuple[str, float], ...]
    realized_feature_rates: tuple[tuple[str, float], ...]
    repeated_ngram_fractions: tuple[float, ...]
    digit_bearing_token_rate: float
    numeric_token_rate: float
    alphanumeric_identifier_token_rate: float
    opening_frequencies: tuple[tuple[str, int], ...]
    ending_frequencies: tuple[tuple[str, int], ...]
    normalized_nll_values: tuple[float, ...]
    reference_split_token_jsd: float
    profile_sha256: str

    @property
    def median_story_words(self) -> float:
        """Return the reference median story length in words."""
        return float(median(self.story_word_counts))

    @property
    def median_sentence_words(self) -> float:
        """Return the pooled reference median sentence length in words."""
        return float(median(self.sentence_word_counts))

    @property
    def median_normalized_nll(self) -> float:
        """Return the reference median active-token NLL."""
        return float(median(self.normalized_nll_values))

    @property
    def normalized_nll_iqr(self) -> float:
        """Return the reference NLL interquartile range."""
        return empirical_quantile(self.normalized_nll_values, 0.75) - empirical_quantile(
            self.normalized_nll_values, 0.25
        )

    @property
    def median_repeated_ngram_fraction(self) -> float:
        """Return median deterministic repeated 3--5-gram incidence."""
        return float(median(self.repeated_ngram_fractions))


def observe_reference(
    record: ReferenceRecord,
    *,
    model_token_ids: tuple[int, ...],
    normalized_nll: float,
    feature_labels: tuple[str, ...] = (),
    required_words: tuple[str, ...] = (),
) -> ReferenceObservation:
    """Derive deterministic surface features around injected tokenizer/NLL data."""
    word_tokens = lexical_tokens(record.story_text)
    sentence_word_counts = tuple(
        len(_WORD_PATTERN.findall(match.group(0)))
        for match in _SENTENCE_PATTERN.finditer(record.story_text)
        if _WORD_PATTERN.search(match.group(0)) is not None
    )
    paragraphs = tuple(
        paragraph
        for paragraph in re.split(r"\n\s*\n", record.story_text.strip())
        if paragraph.strip()
    )
    opening_words = word_tokens[:3]
    ending_words = word_tokens[-3:]
    return ReferenceObservation(
        record_id=record.record_id,
        word_tokens=word_tokens,
        model_token_ids=model_token_ids,
        sentence_word_counts=sentence_word_counts,
        paragraph_count=len(paragraphs),
        dialogue_present=any(mark in record.story_text for mark in ('"', "“", "”")),
        opening_key=" ".join(opening_words),
        ending_key=" ".join(ending_words),
        # Released metadata can repeat a feature label.  Presence-based
        # observations use one canonical instance while raw annotations retain
        # the exact released tuple.
        feature_labels=canonical_feature_labels(feature_labels),
        normalized_nll=normalized_nll,
        required_words=tuple(word.casefold() for word in required_words),
        realized_feature_labels=realized_feature_labels(
            record.story_text,
            canonical_feature_labels(feature_labels),
        ),
        repeated_ngram_fraction=repeated_ngram_fraction(word_tokens),
    )


def build_reference_profile(
    observations: tuple[ReferenceObservation, ...],
) -> ReferenceProfile:
    """Aggregate immutable reference measurements into calibrated distributions."""
    if not observations:
        raise ValueError("at least one reference observation is required")
    if len({observation.record_id for observation in observations}) != len(
        observations
    ):
        raise ValueError("reference observation IDs must be unique")
    ordered = tuple(sorted(observations, key=lambda item: item.record_id))
    word_counts = Counter(token for item in ordered for token in item.word_tokens)
    required_word_counts = Counter(
        word.casefold() for item in ordered for word in item.required_words
    )
    token_counts = Counter(token for item in ordered for token in item.model_token_ids)
    total_model_tokens = sum(token_counts.values())
    feature_counts = Counter(
        label for item in ordered for label in item.feature_labels
    )
    realized_feature_counts = Counter(
        label for item in ordered for label in item.realized_feature_labels
    )
    digit_bearing_count, numeric_count, alphanumeric_identifier_count = (
        tuple(
            sum(values)
            for values in zip(
                *(token_form_counts(item.word_tokens) for item in ordered),
                strict=True,
            )
        )
    )
    total_words = sum(len(item.word_tokens) for item in ordered)
    left, right = _stable_profile_halves(ordered)
    reference_split_token_jsd = jensen_shannon_divergence(
        _token_probabilities(left), _token_probabilities(right)
    )
    profile_values = {
        "record_ids": [item.record_id for item in ordered],
        "word_frequencies": sorted(word_counts.items()),
        "required_word_frequencies": sorted(required_word_counts.items()),
        "token_counts": sorted(token_counts.items()),
        "story_word_counts": [len(item.word_tokens) for item in ordered],
        "model_token_counts": [len(item.model_token_ids) for item in ordered],
        "sentence_word_counts": [
            count for item in ordered for count in item.sentence_word_counts
        ],
        "paragraph_counts": [item.paragraph_count for item in ordered],
        "dialogue": [item.dialogue_present for item in ordered],
        "features": sorted(feature_counts.items()),
        "realized_features": sorted(realized_feature_counts.items()),
        "repeated_ngram_fractions": [
            item.repeated_ngram_fraction for item in ordered
        ],
        "token_form_counts": {
            "alphanumeric_identifier": alphanumeric_identifier_count,
            "digit_bearing": digit_bearing_count,
            "numeric": numeric_count,
            "total": total_words,
        },
        "openings": sorted(Counter(item.opening_key for item in ordered).items()),
        "endings": sorted(Counter(item.ending_key for item in ordered).items()),
        "normalized_nll": [item.normalized_nll for item in ordered],
        "reference_split_token_jsd": reference_split_token_jsd,
    }
    profile_sha256 = hashlib.sha256(
        json.dumps(
            profile_values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return ReferenceProfile(
        record_count=len(ordered),
        vocabulary=frozenset(word_counts),
        word_frequencies=tuple(sorted(word_counts.items())),
        required_word_frequencies=tuple(sorted(required_word_counts.items())),
        token_probabilities=tuple(
            (token, count / total_model_tokens)
            for token, count in sorted(token_counts.items())
        ),
        story_word_counts=tuple(len(item.word_tokens) for item in ordered),
        model_token_counts=tuple(len(item.model_token_ids) for item in ordered),
        sentence_word_counts=tuple(
            count for item in ordered for count in item.sentence_word_counts
        ),
        paragraph_break_rate=sum(item.paragraph_count > 1 for item in ordered)
        / len(ordered),
        dialogue_rate=sum(item.dialogue_present for item in ordered) / len(ordered),
        feature_rates=tuple(
            (label, count / len(ordered))
            for label, count in sorted(feature_counts.items())
        ),
        realized_feature_rates=tuple(
            (label, count / len(ordered))
            for label, count in sorted(realized_feature_counts.items())
        ),
        repeated_ngram_fractions=tuple(
            item.repeated_ngram_fraction for item in ordered
        ),
        digit_bearing_token_rate=digit_bearing_count / total_words,
        numeric_token_rate=numeric_count / total_words,
        alphanumeric_identifier_token_rate=(
            alphanumeric_identifier_count / total_words
        ),
        opening_frequencies=tuple(
            sorted(Counter(item.opening_key for item in ordered).items())
        ),
        ending_frequencies=tuple(
            sorted(Counter(item.ending_key for item in ordered).items())
        ),
        normalized_nll_values=tuple(item.normalized_nll for item in ordered),
        reference_split_token_jsd=reference_split_token_jsd,
        profile_sha256=profile_sha256,
    )


def jensen_shannon_divergence(
    left: dict[int, float], right: dict[int, float]
) -> float:
    """Return base-two Jensen-Shannon divergence in the bounded range [0, 1]."""
    keys = frozenset(left) | frozenset(right)
    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2.0 for key in keys}
    relative_entropy = lambda values: sum(
        probability * math.log2(probability / midpoint[key])
        for key, probability in values.items()
        if probability > 0.0
    )
    return (relative_entropy(left) + relative_entropy(right)) / 2.0


def empirical_quantile(values: tuple[float, ...], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""
    if not values:
        raise ValueError("quantiles require at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def empirical_wasserstein_distance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    """Return exact one-dimensional Wasserstein distance for empirical samples."""
    if not left or not right:
        raise ValueError("Wasserstein distance requires two nonempty samples")
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("Wasserstein samples must be finite")
    support = tuple(sorted(set(left) | set(right)))
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_cumulative = tuple(
        accumulate(left_counts[value] / len(left) for value in support)
    )[:-1]
    right_cumulative = tuple(
        accumulate(right_counts[value] / len(right) for value in support)
    )[:-1]
    return sum(
        abs(left_probability - right_probability) * (upper - lower)
        for lower, upper, left_probability, right_probability in zip(
            support[:-1],
            support[1:],
            left_cumulative,
            right_cumulative,
            strict=True,
        )
    )


def _stable_profile_halves(
    observations: tuple[ReferenceObservation, ...],
) -> tuple[tuple[ReferenceObservation, ...], tuple[ReferenceObservation, ...]]:
    ranked = tuple(
        sorted(
            observations,
            key=lambda item: (
                hashlib.sha256(f"reference-split\0{item.record_id}".encode()).digest(),
                item.record_id,
            ),
        )
    )
    midpoint = len(ranked) // 2
    if midpoint == 0:
        return ranked, ranked
    return ranked[:midpoint], ranked[midpoint:]


def _token_probabilities(
    observations: tuple[ReferenceObservation, ...],
) -> dict[int, float]:
    counts = Counter(token for item in observations for token in item.model_token_ids)
    total = sum(counts.values())
    return {token: count / total for token, count in counts.items()}


def _require_text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
