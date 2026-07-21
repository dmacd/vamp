"""Reference-calibrated quality gates for the TinyWorlds-v2 bakeoff."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import math
from statistics import median

from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceProfile,
    empirical_wasserstein_distance,
    jensen_shannon_divergence,
)
from apm.data.text.tinyworlds_v2.surface import (
    repeated_ngram_fraction,
    token_form_counts,
)


SCREEN_ROUTE_ORDER = (
    "ling-2.6-flash",
    "gemma-4-26b-a4b-it",
    "deepseek-v4-flash",
    "mistral-small-2603",
    "qwen3.5-35b-a3b",
    "gemini-3.1-flash-lite",
    "gpt-5.4-mini",
)

TWO_ROUTE_AUTHOR_ORDER = (
    "qwen3.5-35b-a3b",
    "gpt-5.4-mini",
)

BLIND_VERIFIER_DIMENSIONS = (
    "preschool_vocabulary",
    "sentence_simplicity",
    "grammar",
    "plot_coherence",
    "non_repetition",
)


class QualityPhase(str, Enum):
    """A historical screen/full pass or the direct two-author evaluation."""

    SCREEN = "screen"
    FULL = "full"
    DIRECT = "direct"


class QualityOutcome(str, Enum):
    """Machine-readable bakeoff selection outcome."""

    READY_FOR_FULL_BAKEOFF = "ready_for_full_bakeoff"
    QUALITY_QUALIFIED_ROUTES = "quality_qualified_routes"
    NO_QUALITY_QUALIFIED_ROUTE = "no_quality_qualified_route"


@dataclass(frozen=True, slots=True)
class GeneratedObservation:
    """Cached deterministic and model-assisted checks for one generated story."""

    sample_id: str
    route_id: str
    schema_valid: bool
    deterministic_accepted: bool
    required_noun_ok: bool
    required_verb_ok: bool
    required_adjective_ok: bool
    required_feature_ok: bool
    forbidden_identifier_found: bool
    word_tokens: tuple[str, ...]
    model_token_ids: tuple[int, ...]
    sentence_word_counts: tuple[int, ...]
    paragraph_count: int
    dialogue_present: bool
    feature_labels: tuple[str, ...]
    normalized_nll: float | None
    blind_verifier_scores: tuple[tuple[str, float], ...] | None
    blind_verifier_hard_failure: bool
    billed_cost_usd: float
    requested_feature_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.route_id, "route_id")
        bool_fields = (
            self.schema_valid,
            self.deterministic_accepted,
            self.required_noun_ok,
            self.required_verb_ok,
            self.required_adjective_ok,
            self.required_feature_ok,
            self.forbidden_identifier_found,
            self.dialogue_present,
            self.blind_verifier_hard_failure,
        )
        if any(type(value) is not bool for value in bool_fields):
            raise TypeError("generated-observation flags must be bools")
        if self.deterministic_accepted and not self.schema_valid:
            raise ValueError("an accepted observation must have a valid schema")
        if self.deterministic_accepted:
            if not self.word_tokens or not self.model_token_ids:
                raise ValueError("accepted observations require word and model tokens")
            if not self.sentence_word_counts or self.paragraph_count <= 0:
                raise ValueError("accepted observations require valid surface measurements")
            if self.normalized_nll is None:
                raise ValueError("accepted observations require normalized NLL")
        if any(type(token) is not str for token in self.word_tokens):
            raise TypeError("word tokens must be strings")
        if any(type(token) is not int or token < 0 for token in self.model_token_ids):
            raise ValueError("model token IDs must be nonnegative integers")
        if any(
            type(count) is not int or count <= 0
            for count in self.sentence_word_counts
        ):
            raise ValueError("sentence word counts must be positive integers")
        if type(self.paragraph_count) is not int or self.paragraph_count < 0:
            raise ValueError("paragraph_count must be nonnegative")
        if type(self.feature_labels) is not tuple or len(set(self.feature_labels)) != len(
            self.feature_labels
        ):
            raise ValueError("feature_labels must be a tuple of unique labels")
        if any(type(label) is not str or not label for label in self.feature_labels):
            raise ValueError("feature labels must be nonempty strings")
        if (
            type(self.requested_feature_labels) is not tuple
            or len(set(self.requested_feature_labels))
            != len(self.requested_feature_labels)
            or any(
                type(label) is not str or not label
                for label in self.requested_feature_labels
            )
        ):
            raise ValueError(
                "requested_feature_labels must be a tuple of unique labels"
            )
        if not set(self.feature_labels).issubset(self.requested_feature_labels):
            raise ValueError("realized features must be a subset of requested features")
        if self.normalized_nll is not None and (
            type(self.normalized_nll) is not float
            or not math.isfinite(self.normalized_nll)
            or self.normalized_nll < 0.0
        ):
            raise ValueError(
                "normalized_nll must be finite and nonnegative when present"
            )
        if self.blind_verifier_scores is not None:
            _validate_blind_verifier_scores(
                self.blind_verifier_scores,
                name="blind_verifier_scores",
            )
        if not math.isfinite(self.billed_cost_usd) or self.billed_cost_usd < 0.0:
            raise ValueError("billed_cost_usd must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RouteQualityReport:
    """All fixed quality metrics and gate failures for one route."""

    route_id: str
    phase: QualityPhase
    sample_ids: tuple[str, ...]
    schema_valid_rate: float
    deterministic_acceptance_rate: float
    noun_adherence_rate: float
    verb_adherence_rate: float
    adjective_adherence_rate: float
    feature_adherence_rate: float
    forbidden_form_count: int
    vocabulary_coverage: float
    token_unigram_jsd: float
    token_jsd_limit: float
    median_nll_difference: float
    normalized_nll_wasserstein_ratio: float
    median_story_relative_difference: float
    median_sentence_word_difference: float
    paragraph_break_rate_difference: float
    dialogue_rate_difference: float
    max_requested_feature_rate_difference: float
    max_realized_feature_rate_difference: float
    repeated_ngram_fraction_generated_median: float
    repeated_ngram_fraction_reference_median: float
    repeated_ngram_fraction_median_difference: float
    digit_bearing_token_rate_generated: float
    digit_bearing_token_rate_reference: float
    digit_bearing_token_rate_difference: float
    numeric_token_rate_generated: float
    numeric_token_rate_reference: float
    numeric_token_rate_difference: float
    alphanumeric_identifier_token_rate_generated: float
    alphanumeric_identifier_token_rate_reference: float
    alphanumeric_identifier_token_rate_difference: float
    blind_verifier_preschool_vocabulary_generated_mean: float
    blind_verifier_preschool_vocabulary_genuine_mean: float
    blind_verifier_preschool_vocabulary_mean_difference: float
    blind_verifier_sentence_simplicity_generated_mean: float
    blind_verifier_sentence_simplicity_genuine_mean: float
    blind_verifier_sentence_simplicity_mean_difference: float
    blind_verifier_grammar_generated_mean: float
    blind_verifier_grammar_genuine_mean: float
    blind_verifier_grammar_mean_difference: float
    blind_verifier_plot_coherence_generated_mean: float
    blind_verifier_plot_coherence_genuine_mean: float
    blind_verifier_plot_coherence_mean_difference: float
    blind_verifier_non_repetition_generated_mean: float
    blind_verifier_non_repetition_genuine_mean: float
    blind_verifier_non_repetition_mean_difference: float
    blind_verifier_clean_rate: float
    billed_cost_usd: float
    alignment_distance: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether every gate passed."""
        return not self.failures


@dataclass(frozen=True, slots=True)
class QualitySelection:
    """Ordered finalists or an explicit no-qualified-route result."""

    outcome: QualityOutcome
    route_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if type(self.outcome) is not QualityOutcome:
            raise TypeError("outcome must be a QualityOutcome")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("selected route IDs must be unique")
        _require_text(self.reason, "reason")
        if (
            self.outcome is QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE
            and self.route_ids
        ):
            raise ValueError("a no-quality outcome cannot contain routes")


def evaluate_route_quality(
    observations: tuple[GeneratedObservation, ...],
    reference_profile: ReferenceProfile,
    *,
    phase: QualityPhase,
    reference_blind_verifier_means: tuple[tuple[str, float], ...] | None = None,
    matched_reference_profile: ReferenceProfile | None = None,
    expected_feature_rates: tuple[tuple[str, float], ...] | None = None,
) -> RouteQualityReport:
    """Evaluate one route against the immutable screen or full-quality gates."""
    if not observations:
        raise ValueError("at least one generated observation is required")
    route_ids = {observation.route_id for observation in observations}
    if len(route_ids) != 1:
        raise ValueError("all observations must belong to one route")
    if len({item.sample_id for item in observations}) != len(observations):
        raise ValueError("sample IDs must be unique within a route")
    if type(phase) is not QualityPhase:
        raise TypeError("phase must be a QualityPhase")
    if phase is QualityPhase.FULL and reference_blind_verifier_means is None:
        raise ValueError("full quality evaluation requires reference verifier means")
    if reference_blind_verifier_means is not None:
        _validate_blind_verifier_scores(
            reference_blind_verifier_means,
            name="reference_blind_verifier_means",
        )
    if matched_reference_profile is not None and type(
        matched_reference_profile
    ) is not ReferenceProfile:
        raise TypeError("matched_reference_profile must be a ReferenceProfile")
    if expected_feature_rates is not None:
        _validate_expected_feature_rates(expected_feature_rates)
    comparison_profile = (
        reference_profile
        if matched_reference_profile is None
        else matched_reference_profile
    )
    accepted = tuple(item for item in observations if item.deterministic_accepted)
    if phase is QualityPhase.FULL and any(
        item.blind_verifier_scores is None for item in accepted
    ):
        raise ValueError("full quality evaluation requires every accepted verifier score")
    sample_count = len(observations)
    rate = lambda predicate: sum(predicate(item) for item in observations) / sample_count
    accepted_word_count = sum(len(item.word_tokens) for item in accepted)
    vocabulary_coverage = (
        sum(
            token.casefold() in reference_profile.vocabulary
            for item in accepted
            for token in item.word_tokens
        )
        / accepted_word_count
        if accepted_word_count
        else 0.0
    )
    generated_token_probabilities = _token_probabilities(accepted)
    token_unigram_jsd = jensen_shannon_divergence(
        dict(comparison_profile.token_probabilities), generated_token_probabilities
    )
    accepted_nlls = tuple(
        item.normalized_nll
        for item in accepted
        if item.normalized_nll is not None
    )
    nll_difference = (
        abs(float(median(accepted_nlls)) - comparison_profile.median_normalized_nll)
        if accepted_nlls
        else math.inf
    )
    nll_wasserstein = (
        empirical_wasserstein_distance(
            tuple(float(value) for value in accepted_nlls),
            comparison_profile.normalized_nll_values,
        )
        if accepted_nlls
        else math.inf
    )
    reference_iqr = comparison_profile.normalized_nll_iqr
    nll_wasserstein_ratio = (
        nll_wasserstein / reference_iqr
        if reference_iqr > 0.0
        else (0.0 if nll_wasserstein == 0.0 else math.inf)
    )
    story_relative_difference = (
        abs(
            float(median(tuple(len(item.word_tokens) for item in accepted)))
            - comparison_profile.median_story_words
        )
        / comparison_profile.median_story_words
        if accepted
        else math.inf
    )
    generated_sentence_counts = tuple(
        count for item in accepted for count in item.sentence_word_counts
    )
    sentence_difference = (
        abs(
            float(median(generated_sentence_counts))
            - comparison_profile.median_sentence_words
        )
        if generated_sentence_counts
        else math.inf
    )
    paragraph_rate = (
        sum(item.paragraph_count > 1 for item in accepted) / len(accepted)
        if accepted
        else 0.0
    )
    dialogue_rate = (
        sum(item.dialogue_present for item in accepted) / len(accepted)
        if accepted
        else 0.0
    )
    generated_realized_feature_counts = Counter(
        label for item in accepted for label in item.feature_labels
    )
    generated_requested_feature_counts = Counter(
        label for item in observations for label in item.requested_feature_labels
    )
    reference_requested_feature_rates = dict(
        comparison_profile.feature_rates
        if expected_feature_rates is None
        else expected_feature_rates
    )
    requested_feature_names = frozenset(reference_requested_feature_rates) | frozenset(
        generated_requested_feature_counts
    )
    max_requested_feature_difference = max(
        (
            abs(
                generated_requested_feature_counts.get(feature, 0)
                / len(observations)
                - reference_requested_feature_rates.get(feature, 0.0)
            )
            for feature in requested_feature_names
        ),
        default=0.0,
    )
    reference_realized_feature_rates = dict(
        comparison_profile.realized_feature_rates
    )
    realized_feature_names = frozenset(reference_realized_feature_rates) | frozenset(
        generated_realized_feature_counts
    )
    max_realized_feature_difference = max(
        (
            abs(
                generated_realized_feature_counts.get(feature, 0)
                / max(len(accepted), 1)
                - reference_realized_feature_rates.get(feature, 0.0)
            )
            for feature in realized_feature_names
        ),
        default=0.0,
    )
    generated_repetition = tuple(
        repeated_ngram_fraction(item.word_tokens) for item in accepted
    )
    generated_repetition_median = (
        float(median(generated_repetition)) if generated_repetition else 0.0
    )
    reference_repetition_median = (
        comparison_profile.median_repeated_ngram_fraction
    )
    repetition_difference = abs(
        generated_repetition_median - reference_repetition_median
    )
    generated_form_counts = tuple(
        sum(values)
        for values in zip(
            *(token_form_counts(item.word_tokens) for item in accepted),
            strict=True,
        )
    ) if accepted else (0, 0, 0)
    generated_form_rates = tuple(
        count / accepted_word_count if accepted_word_count else 0.0
        for count in generated_form_counts
    )
    reference_form_rates = (
        comparison_profile.digit_bearing_token_rate,
        comparison_profile.numeric_token_rate,
        comparison_profile.alphanumeric_identifier_token_rate,
    )
    form_rate_differences = tuple(
        abs(generated - reference)
        for generated, reference in zip(
            generated_form_rates,
            reference_form_rates,
            strict=True,
        )
    )
    reference_verifier_by_dimension = dict(reference_blind_verifier_means or ())
    generated_verifier_by_dimension = {
        dimension: tuple(
            dict(item.blind_verifier_scores)[dimension]
            for item in accepted
            if item.blind_verifier_scores is not None
        )
        for dimension in BLIND_VERIFIER_DIMENSIONS
    }
    verifier_values: dict[str, float] = {}
    for dimension in BLIND_VERIFIER_DIMENSIONS:
        scores = generated_verifier_by_dimension[dimension]
        generated_mean = sum(scores) / len(scores) if scores else 0.0
        genuine_mean = reference_verifier_by_dimension.get(dimension, 0.0)
        verifier_values[
            f"blind_verifier_{dimension}_generated_mean"
        ] = generated_mean
        verifier_values[
            f"blind_verifier_{dimension}_genuine_mean"
        ] = genuine_mean
        verifier_values[
            f"blind_verifier_{dimension}_mean_difference"
        ] = genuine_mean - generated_mean
    verifier_clean_rate = (
        sum(not item.blind_verifier_hard_failure for item in accepted)
        / len(accepted)
        if accepted
        else 0.0
    )
    values = {
        "schema_valid_rate": rate(lambda item: item.schema_valid),
        "deterministic_acceptance_rate": len(accepted) / sample_count,
        "noun_adherence_rate": rate(
            lambda item: item.schema_valid and item.required_noun_ok
        ),
        "verb_adherence_rate": rate(
            lambda item: item.schema_valid and item.required_verb_ok
        ),
        "adjective_adherence_rate": rate(
            lambda item: item.schema_valid and item.required_adjective_ok
        ),
        "feature_adherence_rate": rate(
            lambda item: item.schema_valid and item.required_feature_ok
        ),
        "forbidden_form_count": sum(
            item.forbidden_identifier_found for item in observations
        ),
        "vocabulary_coverage": vocabulary_coverage,
        "token_unigram_jsd": token_unigram_jsd,
        "token_jsd_limit": max(
            0.10, 5.0 * comparison_profile.reference_split_token_jsd
        ),
        "median_nll_difference": nll_difference,
        "normalized_nll_wasserstein_ratio": nll_wasserstein_ratio,
        "median_story_relative_difference": story_relative_difference,
        "median_sentence_word_difference": sentence_difference,
        "paragraph_break_rate_difference": abs(
            paragraph_rate - comparison_profile.paragraph_break_rate
        ),
        "dialogue_rate_difference": abs(dialogue_rate - comparison_profile.dialogue_rate),
        "max_requested_feature_rate_difference": max_requested_feature_difference,
        "max_realized_feature_rate_difference": max_realized_feature_difference,
        "repeated_ngram_fraction_generated_median": generated_repetition_median,
        "repeated_ngram_fraction_reference_median": reference_repetition_median,
        "repeated_ngram_fraction_median_difference": repetition_difference,
        "digit_bearing_token_rate_generated": generated_form_rates[0],
        "digit_bearing_token_rate_reference": reference_form_rates[0],
        "digit_bearing_token_rate_difference": form_rate_differences[0],
        "numeric_token_rate_generated": generated_form_rates[1],
        "numeric_token_rate_reference": reference_form_rates[1],
        "numeric_token_rate_difference": form_rate_differences[1],
        "alphanumeric_identifier_token_rate_generated": generated_form_rates[2],
        "alphanumeric_identifier_token_rate_reference": reference_form_rates[2],
        "alphanumeric_identifier_token_rate_difference": form_rate_differences[2],
        "blind_verifier_clean_rate": verifier_clean_rate,
        **verifier_values,
    }
    failures = _quality_failures(values, phase)
    alignment_components = (
        max(0.0, 0.98 - vocabulary_coverage) / 0.02,
        token_unigram_jsd / max(values["token_jsd_limit"], 1e-12),
        nll_difference / 0.30,
        nll_wasserstein_ratio / 0.35,
        story_relative_difference / 0.15,
        sentence_difference / 2.0,
        values["paragraph_break_rate_difference"] / 0.10,
        values["dialogue_rate_difference"] / 0.10,
        max_requested_feature_difference / 0.10,
        max_realized_feature_difference / 0.10,
        repetition_difference / 0.05,
        form_rate_differences[0] / 0.01,
        form_rate_differences[1] / 0.01,
        generated_form_rates[2] / 1e-12,
        *(
            max(0.0, 0.95 - values[name]) / 0.05
            for name in (
                "noun_adherence_rate",
                "verb_adherence_rate",
                "adjective_adherence_rate",
                *(("feature_adherence_rate",) if phase is QualityPhase.FULL else ()),
            )
        ),
    )
    if phase is QualityPhase.FULL:
        alignment_components += (
            max(
                (
                    max(
                        verifier_values[
                            f"blind_verifier_{dimension}_mean_difference"
                        ],
                        0.0,
                    )
                    for dimension in BLIND_VERIFIER_DIMENSIONS
                ),
                default=0.0,
            )
            / 0.50,
            max(0.0, 0.90 - verifier_clean_rate) / 0.10,
        )
    finite_alignment = tuple(
        component for component in alignment_components if math.isfinite(component)
    )
    alignment_distance = (
        sum(finite_alignment) / len(finite_alignment)
        if len(finite_alignment) == len(alignment_components)
        else math.inf
    )
    return RouteQualityReport(
        route_id=next(iter(route_ids)),
        phase=phase,
        sample_ids=tuple(sorted(item.sample_id for item in observations)),
        billed_cost_usd=sum(item.billed_cost_usd for item in observations),
        alignment_distance=alignment_distance,
        failures=failures,
        **values,
    )


def select_screen_finalists(
    reports: tuple[RouteQualityReport, ...],
    *,
    route_order: tuple[str, ...] = SCREEN_ROUTE_ORDER,
) -> QualitySelection:
    """Choose cheapest, closest, then Pareto cost/alignment screen finalists."""
    _validate_report_set(reports, QualityPhase.SCREEN, route_order, expected_size=50)
    passing = tuple(report for report in reports if report.passed)
    if not passing:
        return QualitySelection(
            QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE,
            (),
            "no route passed the 50-brief screen gates",
        )
    order_index = {route_id: index for index, route_id in enumerate(route_order)}
    cheapest = min(
        passing, key=lambda item: (item.billed_cost_usd, order_index[item.route_id])
    )
    selected = [cheapest]
    remaining = tuple(item for item in passing if item.route_id != cheapest.route_id)
    if remaining:
        closest = min(
            remaining,
            key=lambda item: (item.alignment_distance, order_index[item.route_id]),
        )
        selected.append(closest)
        remaining = tuple(item for item in remaining if item.route_id != closest.route_id)
    if remaining:
        frontier = tuple(
            candidate
            for candidate in remaining
            if not any(
                other.billed_cost_usd <= candidate.billed_cost_usd
                and other.alignment_distance <= candidate.alignment_distance
                and (
                    other.billed_cost_usd < candidate.billed_cost_usd
                    or other.alignment_distance < candidate.alignment_distance
                )
                for other in remaining
                if other.route_id != candidate.route_id
            )
        )
        cost_values = tuple(item.billed_cost_usd for item in passing)
        alignment_values = tuple(item.alignment_distance for item in passing)
        tradeoff = min(
            frontier,
            key=lambda item: (
                _minmax(item.billed_cost_usd, cost_values)
                + _minmax(item.alignment_distance, alignment_values),
                order_index[item.route_id],
            ),
        )
        selected.append(tradeoff)
    return QualitySelection(
        QualityOutcome.READY_FOR_FULL_BAKEOFF,
        tuple(item.route_id for item in selected),
        "screen gates passed; finalists are ordered by selection role",
    )


def select_full_quality_routes(
    reports: tuple[RouteQualityReport, ...],
    *,
    finalist_order: tuple[str, ...],
) -> QualitySelection:
    """Return every 200-brief finalist that passes the fixed full gates."""
    _validate_report_set(reports, QualityPhase.FULL, finalist_order, expected_size=200)
    by_route = {report.route_id: report for report in reports}
    qualified = tuple(
        route_id for route_id in finalist_order if by_route[route_id].passed
    )
    if not qualified:
        return QualitySelection(
            QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE,
            (),
            "no finalist passed every 200-brief reference-calibrated gate",
        )
    return QualitySelection(
        QualityOutcome.QUALITY_QUALIFIED_ROUTES,
        qualified,
        "one or more finalists passed every full-quality gate",
    )


def select_direct_quality_routes(
    reports: tuple[RouteQualityReport, ...],
    *,
    route_order: tuple[str, ...] = TWO_ROUTE_AUTHOR_ORDER,
) -> QualitySelection:
    """Return qualifying routes from a direct paired 200-story comparison.

    Unlike the historical seven-route funnel, both named author routes are
    evaluated on every brief. There is no screen, expansion, or finalist
    selection stage.
    """
    _validate_report_set(reports, QualityPhase.DIRECT, route_order, expected_size=200)
    by_route = {report.route_id: report for report in reports}
    qualified = tuple(route_id for route_id in route_order if by_route[route_id].passed)
    if not qualified:
        return QualitySelection(
            QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE,
            (),
            "neither author passed every direct 200-brief quality gate",
        )
    return QualitySelection(
        QualityOutcome.QUALITY_QUALIFIED_ROUTES,
        qualified,
        "one or both authors passed every direct 200-brief quality gate",
    )


def _quality_failures(values: dict[str, float | int], phase: QualityPhase) -> tuple[str, ...]:
    gates = [
        (values["schema_valid_rate"] >= (0.98 if phase is QualityPhase.SCREEN else 0.99), "schema_valid_rate"),
        (values["deterministic_acceptance_rate"] >= (0.90 if phase is QualityPhase.SCREEN else 0.95), "deterministic_acceptance_rate"),
        (values["forbidden_form_count"] == 0, "forbidden_forms"),
    ]
    if phase in (QualityPhase.FULL, QualityPhase.DIRECT):
        gates.extend(
            (
                (values[name] >= 0.95, name)
                for name in (
                    "noun_adherence_rate",
                    "verb_adherence_rate",
                    "adjective_adherence_rate",
                    *(("feature_adherence_rate",) if phase is QualityPhase.FULL else ()),
                )
            )
        )
        gates.extend(
            (
                (values["vocabulary_coverage"] >= 0.98, "vocabulary_coverage"),
                (values["token_unigram_jsd"] <= values["token_jsd_limit"], "token_unigram_jsd"),
                (values["median_nll_difference"] <= 0.30, "median_nll_difference"),
                (values["normalized_nll_wasserstein_ratio"] <= 0.35, "normalized_nll_wasserstein_ratio"),
                (values["median_story_relative_difference"] <= 0.15, "median_story_relative_difference"),
                (values["median_sentence_word_difference"] <= 2.0, "median_sentence_word_difference"),
                (values["paragraph_break_rate_difference"] <= 0.10, "paragraph_break_rate_difference"),
                (values["dialogue_rate_difference"] <= 0.10, "dialogue_rate_difference"),
                (
                    values["max_requested_feature_rate_difference"] <= 0.10,
                    "max_requested_feature_rate_difference",
                ),
                (
                    values["repeated_ngram_fraction_median_difference"] <= 0.05,
                    "repeated_ngram_fraction_median_difference",
                ),
                (
                    values["digit_bearing_token_rate_difference"] <= 0.01,
                    "digit_bearing_token_rate_difference",
                ),
                (
                    values["numeric_token_rate_difference"] <= 0.01,
                    "numeric_token_rate_difference",
                ),
                (
                    values["alphanumeric_identifier_token_rate_generated"] == 0.0,
                    "alphanumeric_identifier_token_rate_generated",
                ),
            )
        )
        if phase is QualityPhase.FULL:
            gates.append(
                (
                    values["max_realized_feature_rate_difference"] <= 0.10,
                    "max_realized_feature_rate_difference",
                )
            )
    if phase is QualityPhase.FULL:
        gates.append(
            (
                values["blind_verifier_clean_rate"] >= 0.90,
                "blind_verifier_clean_rate",
            )
        )
        gates.extend(
            (
                (
                    values[
                        f"blind_verifier_{dimension}_mean_difference"
                    ]
                    <= 0.50,
                    f"blind_verifier_{dimension}_mean_difference",
                )
                for dimension in BLIND_VERIFIER_DIMENSIONS
            )
        )
    return tuple(name for passed, name in gates if not passed)


def validate_route_quality_report(report: RouteQualityReport) -> None:
    """Reject malformed or threshold-inconsistent persisted quality reports.

    ``RouteQualityReport.passed`` is intentionally a convenience property, not
    an integrity boundary.  Persisted reports must prove that their recorded
    failure list follows mechanically from their metric values.  The one
    operational failure added outside the metric evaluator is permitted only
    for a full report and remains ordered after all threshold failures.
    """
    if type(report) is not RouteQualityReport:
        raise TypeError("report must be a RouteQualityReport")
    _require_text(report.route_id, "route_id")
    if type(report.phase) is not QualityPhase:
        raise TypeError("report phase must be a QualityPhase")
    if (
        type(report.sample_ids) is not tuple
        or not report.sample_ids
        or any(
            type(sample_id) is not str or not sample_id
            for sample_id in report.sample_ids
        )
        or report.sample_ids != tuple(sorted(report.sample_ids))
        or len(report.sample_ids) != len(set(report.sample_ids))
    ):
        raise ValueError("quality report sample IDs must be nonempty, unique, and ordered")
    if (
        type(report.forbidden_form_count) is not int
        or not 0 <= report.forbidden_form_count <= len(report.sample_ids)
    ):
        raise ValueError(
            "forbidden_form_count must be between zero and the sample count"
        )
    if (
        type(report.failures) is not tuple
        or any(type(name) is not str or not name for name in report.failures)
        or len(report.failures) != len(set(report.failures))
    ):
        raise ValueError("quality report failures must be unique nonempty strings")

    rate_fields = (
        "schema_valid_rate",
        "deterministic_acceptance_rate",
        "noun_adherence_rate",
        "verb_adherence_rate",
        "adjective_adherence_rate",
        "feature_adherence_rate",
        "vocabulary_coverage",
        "paragraph_break_rate_difference",
        "dialogue_rate_difference",
        "max_requested_feature_rate_difference",
        "max_realized_feature_rate_difference",
        "repeated_ngram_fraction_generated_median",
        "repeated_ngram_fraction_reference_median",
        "repeated_ngram_fraction_median_difference",
        "digit_bearing_token_rate_generated",
        "digit_bearing_token_rate_reference",
        "digit_bearing_token_rate_difference",
        "numeric_token_rate_generated",
        "numeric_token_rate_reference",
        "numeric_token_rate_difference",
        "alphanumeric_identifier_token_rate_generated",
        "alphanumeric_identifier_token_rate_reference",
        "alphanumeric_identifier_token_rate_difference",
        "blind_verifier_clean_rate",
    )
    for name in rate_fields:
        value = getattr(report, name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be a finite rate between zero and one")

    verifier_mean_fields = tuple(
        f"blind_verifier_{dimension}_{source}_mean"
        for dimension in BLIND_VERIFIER_DIMENSIONS
        for source in ("generated", "genuine")
    )
    for name in verifier_mean_fields:
        value = getattr(report, name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 5.0
        ):
            raise ValueError(f"{name} must be a finite score between zero and five")
    for dimension in BLIND_VERIFIER_DIMENSIONS:
        name = f"blind_verifier_{dimension}_mean_difference"
        value = getattr(report, name)
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not -5.0 <= value <= 5.0
        ):
            raise ValueError(f"{name} must be a finite score difference")

    finite_nonnegative_fields = (
        "token_unigram_jsd",
        "token_jsd_limit",
        "billed_cost_usd",
    )
    for name in finite_nonnegative_fields:
        value = getattr(report, name)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if report.token_unigram_jsd > 1.0:
        raise ValueError("token_unigram_jsd must not exceed one")
    if not 0.10 <= report.token_jsd_limit <= 5.0:
        raise ValueError("token_jsd_limit must be between 0.10 and 5.0")

    extended_nonnegative_fields = (
        "median_nll_difference",
        "normalized_nll_wasserstein_ratio",
        "median_story_relative_difference",
        "median_sentence_word_difference",
        "alignment_distance",
    )
    for name in extended_nonnegative_fields:
        value = getattr(report, name)
        if type(value) is not float or math.isnan(value) or value < 0.0:
            raise ValueError(f"{name} must be nonnegative and not NaN")

    metric_values = {
        name: getattr(report, name)
        for name in (
            "schema_valid_rate",
            "deterministic_acceptance_rate",
            "noun_adherence_rate",
            "verb_adherence_rate",
            "adjective_adherence_rate",
            "feature_adherence_rate",
            "forbidden_form_count",
            "vocabulary_coverage",
            "token_unigram_jsd",
            "token_jsd_limit",
            "median_nll_difference",
            "normalized_nll_wasserstein_ratio",
            "median_story_relative_difference",
            "median_sentence_word_difference",
            "paragraph_break_rate_difference",
            "dialogue_rate_difference",
            "max_requested_feature_rate_difference",
            "max_realized_feature_rate_difference",
            "repeated_ngram_fraction_median_difference",
            "digit_bearing_token_rate_difference",
            "numeric_token_rate_difference",
            "alphanumeric_identifier_token_rate_generated",
            "blind_verifier_clean_rate",
            *(
                f"blind_verifier_{dimension}_mean_difference"
                for dimension in BLIND_VERIFIER_DIMENSIONS
            ),
        )
    }
    expected = _quality_failures(metric_values, report.phase)
    operational_failure = "reference_verifier_failure"
    if operational_failure in report.failures:
        if report.phase is not QualityPhase.FULL:
            raise ValueError("reference_verifier_failure is valid only for full reports")
        expected += (operational_failure,)
    if report.failures != expected:
        raise ValueError("quality report failures do not follow from its metric values")


def _validate_report_set(
    reports: tuple[RouteQualityReport, ...],
    phase: QualityPhase,
    route_order: tuple[str, ...],
    *,
    expected_size: int,
) -> None:
    if not reports:
        raise ValueError("at least one route report is required")
    if len(set(route_order)) != len(route_order):
        raise ValueError("route order must contain unique route IDs")
    by_route = {report.route_id: report for report in reports}
    if len(by_route) != len(reports):
        raise ValueError("route reports must have unique route IDs")
    if set(by_route) != set(route_order):
        raise ValueError("reports must exactly match the declared route order")
    if any(report.phase is not phase for report in reports):
        raise ValueError(f"all route reports must be for the {phase.value} phase")
    if any(len(report.sample_ids) != expected_size for report in reports):
        raise ValueError(f"every {phase.value} route requires {expected_size} samples")
    sample_ids = {report.sample_ids for report in reports}
    if len(sample_ids) != 1:
        raise ValueError("every route must evaluate the same ordered sample IDs")


def _token_probabilities(
    observations: tuple[GeneratedObservation, ...],
) -> dict[int, float]:
    counts = Counter(token for item in observations for token in item.model_token_ids)
    total = sum(counts.values())
    return {token: count / total for token, count in counts.items()} if total else {}


def _minmax(value: float, values: tuple[float, ...]) -> float:
    lower, upper = min(values), max(values)
    return (value - lower) / (upper - lower) if upper > lower else 0.0


def _require_text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_expected_feature_rates(
    feature_rates: tuple[tuple[str, float], ...],
) -> None:
    if type(feature_rates) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or not item[0]
        or type(item[1]) not in (int, float)
        or not math.isfinite(item[1])
        or not 0.0 <= item[1] <= 1.0
        for item in feature_rates
    ):
        raise ValueError("expected feature rates must be finite named probabilities")
    labels = tuple(item[0] for item in feature_rates)
    if labels != tuple(sorted(labels)) or len(labels) != len(set(labels)):
        raise ValueError("expected feature rate labels must be unique and sorted")


def _validate_blind_verifier_scores(
    scores: tuple[tuple[str, float], ...],
    *,
    name: str,
) -> None:
    if type(scores) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) not in (int, float)
        or not math.isfinite(item[1])
        or not 0.0 <= item[1] <= 5.0
        for item in scores
    ):
        raise ValueError(f"{name} must contain finite scores from zero to five")
    if tuple(item[0] for item in scores) != BLIND_VERIFIER_DIMENSIONS:
        raise ValueError(
            f"{name} must contain every verifier dimension in canonical order"
        )
