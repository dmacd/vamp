from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from apm.data.text.tinyworlds_v2.audit import (
    AuditAllocationError,
    AuditApproval,
    AuditDecision,
    AuditSourceGuess,
    AuditSourceKind,
    AuditSourceRecord,
    build_blinded_audit,
    build_decision_set,
    evaluate_blinded_audit,
    render_audit_html,
    select_human_approved_route,
    validate_audit_approval,
    validate_audit_pair,
)
from apm.data.text.tinyworlds_v2.quality import (
    BLIND_VERIFIER_DIMENSIONS,
    SCREEN_ROUTE_ORDER,
    TWO_ROUTE_AUTHOR_ORDER,
    GeneratedObservation,
    QualityOutcome,
    QualityPhase,
    evaluate_route_quality,
    select_direct_quality_routes,
    select_full_quality_routes,
    select_screen_finalists,
    validate_route_quality_report,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceRecord,
    build_reference_profile,
    jensen_shannon_divergence,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.surface import (
    realized_feature_labels,
    repeated_ngram_fraction,
    token_form_counts,
)


def _verifier_scores(score: float) -> tuple[tuple[str, float], ...]:
    return tuple((dimension, score) for dimension in BLIND_VERIFIER_DIMENSIONS)


def _reference_observations(count: int = 200) -> tuple[ReferenceObservation, ...]:
    return tuple(
        ReferenceObservation(
            record_id=f"reference-{index:03d}",
            word_tokens=("once", "there", "was", "a", "little", "cat"),
            model_token_ids=(index % 2, 1, 2, 3),
            sentence_word_counts=(3, 3),
            paragraph_count=2 if index % 4 == 0 else 1,
            dialogue_present=index % 2 == 0,
            opening_key="once there was",
            ending_key="a little cat",
            feature_labels=("dialogue",) if index % 2 == 0 else (),
            realized_feature_labels=("dialogue",) if index % 2 == 0 else (),
            normalized_nll=1.0 + (index % 5) * 0.1,
            required_words=("cat", "helped"),
        )
        for index in range(count)
    )


def _generated_observations(
    route_id: str,
    count: int,
    *,
    cost_per_sample: float = 0.01,
    nll_offset: float = 0.0,
) -> tuple[GeneratedObservation, ...]:
    return tuple(
        GeneratedObservation(
            sample_id=f"brief-{index:03d}",
            route_id=route_id,
            schema_valid=True,
            deterministic_accepted=True,
            required_noun_ok=True,
            required_verb_ok=True,
            required_adjective_ok=True,
            required_feature_ok=True,
            forbidden_identifier_found=False,
            word_tokens=("once", "there", "was", "a", "little", "cat"),
            model_token_ids=(index % 2, 1, 2, 3),
            sentence_word_counts=(3, 3),
            paragraph_count=2 if index % 4 == 0 else 1,
            dialogue_present=index % 2 == 0,
            feature_labels=("dialogue",) if index % 2 == 0 else (),
            normalized_nll=1.0 + (index % 5) * 0.1 + nll_offset,
            blind_verifier_scores=_verifier_scores(4.0),
            blind_verifier_hard_failure=False,
            billed_cost_usd=cost_per_sample,
            requested_feature_labels=("dialogue",) if index % 2 == 0 else (),
        )
        for index in range(count)
    )


def test_reference_surface_profile_is_order_independent_and_content_addressed() -> None:
    record = ReferenceRecord(
        "story-1",
        'Once there was a cat.\n\n"Hello there!" said Mia.',
        "Write a simple story.",
    )
    observed = observe_reference(
        record,
        model_token_ids=(1, 2, 3, 4),
        normalized_nll=1.25,
        feature_labels=("dialogue",),
        required_words=("cat", "said"),
    )

    assert observed.word_tokens[:3] == ("once", "there", "was")
    assert observed.sentence_word_counts == (5, 2, 2)
    assert observed.paragraph_count == 2
    assert observed.dialogue_present

    observations = _reference_observations()
    forward = build_reference_profile(observations)
    reverse = build_reference_profile(tuple(reversed(observations)))

    assert forward == reverse
    assert len(forward.vocabulary) == 6
    assert dict(forward.required_word_frequencies) == {"cat": 200, "helped": 200}
    assert forward.median_story_words == 6
    assert forward.dialogue_rate == 0.5
    assert forward.paragraph_break_rate == 0.25
    assert forward.normalized_nll_iqr > 0
    assert len(forward.profile_sha256) == 64
    assert jensen_shannon_divergence({1: 1.0}, {2: 1.0}) == 1.0


def test_reference_profile_digest_binds_split_jsd() -> None:
    record_ids = tuple(f"split-{index}" for index in range(4))
    ranked_ids = tuple(
        sorted(
            record_ids,
            key=lambda record_id: (
                sha256(f"reference-split\0{record_id}".encode()).digest(),
                record_id,
            ),
        )
    )
    template = _reference_observations(1)[0]

    def observations(assignments: dict[str, int]) -> tuple[ReferenceObservation, ...]:
        return tuple(
            replace(
                template,
                record_id=record_id,
                model_token_ids=(assignments[record_id],),
            )
            for record_id in record_ids
        )

    separated = {
        record_id: 1 if index < 2 else 2
        for index, record_id in enumerate(ranked_ids)
    }
    mixed = {
        record_id: 1 if index % 2 == 0 else 2
        for index, record_id in enumerate(ranked_ids)
    }
    separated_profile = build_reference_profile(observations(separated))
    mixed_profile = build_reference_profile(observations(mixed))

    assert separated_profile.token_probabilities == mixed_profile.token_probabilities
    assert separated_profile.reference_split_token_jsd == pytest.approx(1.0)
    assert mixed_profile.reference_split_token_jsd == pytest.approx(0.0)
    assert separated_profile.profile_sha256 != mixed_profile.profile_sha256


def test_surface_metrics_include_numbers_and_do_not_trust_feature_annotations() -> None:
    story = (
        'Fox7 saw 3 birds. "Wait for me," said Fox. Suddenly, the birds came back. '
        "The birds came back. The birds came back."
    )
    observed = observe_reference(
        ReferenceRecord("surface", story),
        model_token_ids=(1, 2, 3),
        normalized_nll=1.0,
        feature_labels=("Dialogue", "Twist", "MoralValue"),
    )

    assert "fox7" in observed.word_tokens
    assert "3" in observed.word_tokens
    assert observed.realized_feature_labels == ("Dialogue", "Twist")
    assert token_form_counts(observed.word_tokens) == (2, 1, 1)
    assert observed.repeated_ngram_fraction > 0.0
    assert repeated_ngram_fraction(observed.word_tokens) == (
        observed.repeated_ngram_fraction
    )
    assert realized_feature_labels(story, ("MoralValue",)) == ()
    assert realized_feature_labels("Mia said she was happy.", ("Dialogue",)) == ()


def test_token_forms_distinguish_hyphenated_prose_from_identifier_segments() -> None:
    ordinary = observe_reference(
        ReferenceRecord("ordinary-age", "The 3-year-old dog ran home."),
        model_token_ids=(1, 2),
        normalized_nll=1.0,
    )
    machine_like = observe_reference(
        ReferenceRecord("machine-id", "The R2-D2 toy ran home."),
        model_token_ids=(1, 2),
        normalized_nll=1.0,
    )

    assert token_form_counts(ordinary.word_tokens) == (1, 0, 0)
    assert token_form_counts(machine_like.word_tokens) == (1, 0, 1)


def test_observations_reject_fabricated_realization_and_negative_nll() -> None:
    reference = _reference_observations(1)[0]
    with pytest.raises(ValueError, match="subset of requested"):
        replace(
            reference,
            feature_labels=(),
            realized_feature_labels=("Dialogue",),
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(reference, normalized_nll=-0.1)

    generated = _generated_observations("route", 1)[0]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(generated, normalized_nll=-0.1)


def test_full_quality_gate_passes_matched_data_and_names_failed_metrics() -> None:
    profile = build_reference_profile(_reference_observations())
    observations = _generated_observations("route", 200)
    report = evaluate_route_quality(
        observations,
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )

    assert report.passed
    assert report.schema_valid_rate == 1.0
    assert report.vocabulary_coverage == 1.0
    assert report.median_nll_difference == pytest.approx(0.0)
    assert report.normalized_nll_wasserstein_ratio == pytest.approx(0.0)
    assert report.blind_verifier_grammar_generated_mean == pytest.approx(4.0)
    assert report.blind_verifier_grammar_genuine_mean == pytest.approx(4.2)
    assert report.blind_verifier_grammar_mean_difference == pytest.approx(0.2)
    assert report.repeated_ngram_fraction_median_difference == pytest.approx(0.0)
    assert report.digit_bearing_token_rate_generated == pytest.approx(0.0)

    degraded = tuple(
        replace(
            item,
            word_tokens=("unseen",) * 6,
            normalized_nll=float(item.normalized_nll) + 2.0,
            blind_verifier_hard_failure=True,
        )
        for item in observations
    )
    failed = evaluate_route_quality(
        degraded,
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )

    assert not failed.passed
    assert "vocabulary_coverage" in failed.failures
    assert "median_nll_difference" in failed.failures
    assert "normalized_nll_wasserstein_ratio" in failed.failures
    assert "blind_verifier_clean_rate" in failed.failures

    identifier_heavy = tuple(
        replace(
            item,
            word_tokens=("once", "fox7", "was", "a", "little", "cat"),
        )
        for item in observations
    )
    identifier_report = evaluate_route_quality(
        identifier_heavy,
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )
    assert "alphanumeric_identifier_token_rate_generated" in (
        identifier_report.failures
    )

    one_forbidden_rejection = (
        replace(
            observations[0],
            deterministic_accepted=False,
            forbidden_identifier_found=True,
        ),
        *observations[1:],
    )
    forbidden_report = evaluate_route_quality(
        one_forbidden_rejection,
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )
    assert forbidden_report.forbidden_form_count == 1
    assert "forbidden_forms" in forbidden_report.failures


def test_persisted_quality_report_validation_rejects_forged_failures_and_domains() -> None:
    profile = build_reference_profile(_reference_observations())
    failed = evaluate_route_quality(
        tuple(
            replace(item, word_tokens=("unseen",) * 6)
            for item in _generated_observations("route", 200)
        ),
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )
    validate_route_quality_report(failed)

    with pytest.raises(ValueError, match="failures do not follow"):
        validate_route_quality_report(replace(failed, failures=()))
    with pytest.raises(ValueError, match="between zero and one"):
        validate_route_quality_report(replace(failed, vocabulary_coverage=1.01))
    with pytest.raises(ValueError, match="forbidden_form_count"):
        validate_route_quality_report(
            replace(failed, forbidden_form_count=len(failed.sample_ids) + 1)
        )
    with pytest.raises(ValueError, match="token_jsd_limit"):
        validate_route_quality_report(replace(failed, token_jsd_limit=5.01))


def test_each_blind_verifier_dimension_must_pass_without_compensation() -> None:
    profile = build_reference_profile(_reference_observations())
    compensating_scores = tuple(
        (dimension, 3.0 if dimension == "preschool_vocabulary" else 5.0)
        for dimension in BLIND_VERIFIER_DIMENSIONS
    )
    observations = tuple(
        replace(item, blind_verifier_scores=compensating_scores)
        for item in _generated_observations("route", 200)
    )

    report = evaluate_route_quality(
        observations,
        profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
    )

    assert sum(dict(compensating_scores).values()) / 5 > 4.2
    assert report.blind_verifier_preschool_vocabulary_generated_mean == 3.0
    assert report.blind_verifier_preschool_vocabulary_genuine_mean == 4.2
    assert report.blind_verifier_preschool_vocabulary_mean_difference == pytest.approx(
        1.2
    )
    assert "blind_verifier_preschool_vocabulary_mean_difference" in report.failures
    assert "blind_verifier_grammar_mean_difference" not in report.failures
    assert not report.passed


def test_quality_uses_matched_profile_and_released_feature_rate_override() -> None:
    broad_profile = build_reference_profile(_reference_observations())
    matched_observations = tuple(
        replace(
            item,
            model_token_ids=(9, 8, 7, 6),
            feature_labels=(),
            realized_feature_labels=(),
            normalized_nll=item.normalized_nll + 3.0,
        )
        for item in _reference_observations()
    )
    matched_profile = build_reference_profile(matched_observations)
    generated = tuple(
        replace(
            item,
            model_token_ids=(9, 8, 7, 6),
            feature_labels=(),
            normalized_nll=float(item.normalized_nll) + 3.0,
        )
        for item in _generated_observations("route", 200)
    )

    report = evaluate_route_quality(
        generated,
        broad_profile,
        phase=QualityPhase.FULL,
        reference_blind_verifier_means=_verifier_scores(4.2),
        matched_reference_profile=matched_profile,
        expected_feature_rates=(("dialogue", 0.5),),
    )

    assert report.passed
    assert report.token_unigram_jsd == pytest.approx(0.0)
    assert report.median_nll_difference == pytest.approx(0.0)
    assert report.max_requested_feature_rate_difference == pytest.approx(0.0)
    assert report.max_realized_feature_rate_difference == pytest.approx(0.0)

    with pytest.raises(ValueError, match="unique and sorted"):
        evaluate_route_quality(
            generated,
            broad_profile,
            phase=QualityPhase.FULL,
            reference_blind_verifier_means=_verifier_scores(4.2),
            expected_feature_rates=(("z", 0.2), ("a", 0.2)),
        )


def test_screen_funnel_uses_same_50_briefs_and_returns_explicit_no_quality() -> None:
    profile = build_reference_profile(_reference_observations())
    base = evaluate_route_quality(
        _generated_observations(SCREEN_ROUTE_ORDER[0], 50),
        profile,
        phase=QualityPhase.SCREEN,
    )
    costs = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    distances = (5.0, 1.0, 3.0, 2.0, 1.5, 1.4, 1.3)
    reports = tuple(
        replace(
            base,
            route_id=route_id,
            billed_cost_usd=cost,
            alignment_distance=distance,
        )
        for route_id, cost, distance in zip(
            SCREEN_ROUTE_ORDER, costs, distances, strict=True
        )
    )

    selection = select_screen_finalists(reports)

    assert selection.outcome is QualityOutcome.READY_FOR_FULL_BAKEOFF
    assert selection.route_ids == (
        SCREEN_ROUTE_ORDER[0],
        SCREEN_ROUTE_ORDER[1],
        SCREEN_ROUTE_ORDER[3],
    )

    failed_reports = tuple(
        replace(report, failures=("schema_valid_rate",)) for report in reports
    )
    no_quality = select_screen_finalists(failed_reports)
    assert no_quality.outcome is QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE
    assert no_quality.route_ids == ()


def test_full_funnel_requires_paired_200_brief_reports() -> None:
    profile = build_reference_profile(_reference_observations())
    finalist_order = ("cheap", "close", "tradeoff")
    reports = tuple(
        evaluate_route_quality(
            _generated_observations(route_id, 200),
            profile,
            phase=QualityPhase.FULL,
            reference_blind_verifier_means=_verifier_scores(4.2),
        )
        for route_id in finalist_order
    )

    selection = select_full_quality_routes(reports, finalist_order=finalist_order)
    assert selection.outcome is QualityOutcome.QUALITY_QUALIFIED_ROUTES
    assert selection.route_ids == finalist_order

    with pytest.raises(ValueError, match="same ordered sample IDs"):
        select_full_quality_routes(
            (
                reports[0],
                replace(
                    reports[1],
                    sample_ids=("different",) + reports[1].sample_ids[1:],
                ),
                reports[2],
            ),
            finalist_order=finalist_order,
        )


def test_direct_two_route_quality_uses_all_200_without_a_verifier() -> None:
    profile = build_reference_profile(_reference_observations())
    reports = tuple(
        evaluate_route_quality(
            _generated_observations(route_id, 200),
            profile,
            phase=QualityPhase.DIRECT,
        )
        for route_id in TWO_ROUTE_AUTHOR_ORDER
    )

    selection = select_direct_quality_routes(reports)

    assert selection.outcome is QualityOutcome.QUALITY_QUALIFIED_ROUTES
    assert selection.route_ids == TWO_ROUTE_AUTHOR_ORDER
    assert all(report.blind_verifier_clean_rate == 1.0 for report in reports)


def _audit_sources() -> tuple[
    tuple[AuditSourceRecord, ...], tuple[AuditSourceRecord, ...]
]:
    references = tuple(
        AuditSourceRecord(
            source_id=f"reference-{index:03d}",
            pair_id=f"brief-{index:03d}",
            story_text=f"Mia found a little red ball number {index}.",
            source_prompt=f"Write a simple story from brief {index}.",
            token_count=20,
            base_normalized_nll=1.2,
            automated_style_scores=(("simplicity", 4.0), ("coherence", 4.5)),
            source_kind=AuditSourceKind.REFERENCE,
        )
        for index in range(120)
    )
    generated = tuple(
        AuditSourceRecord(
            source_id=f"generated-{route}-{index:03d}",
            pair_id=f"brief-{index:03d}",
            story_text=f"Ben helped a small cat in story {index}.",
            source_prompt=f"Write a simple story from brief {index}.",
            token_count=18,
            base_normalized_nll=1.3,
            automated_style_scores=(("simplicity", 4.2), ("coherence", 4.3)),
            source_kind=AuditSourceKind.GENERATED,
            route_id=route,
        )
        for route in ("cheap", "close", "tradeoff")
        for index in range(120)
    )
    return references, generated


def _allocation_audit_record(
    pair_id: str,
    *,
    route_id: str | None = None,
) -> AuditSourceRecord:
    source_kind = (
        AuditSourceKind.REFERENCE
        if route_id is None
        else AuditSourceKind.GENERATED
    )
    return AuditSourceRecord(
        source_id=(
            f"reference-{pair_id}"
            if route_id is None
            else f"generated-{route_id}-{pair_id}"
        ),
        pair_id=pair_id,
        story_text=f"Mia helped a little cat in {pair_id}.",
        source_prompt=f"Write a simple story from {pair_id}.",
        token_count=12,
        base_normalized_nll=1.2,
        automated_style_scores=(("simplicity", 4.0), ("coherence", 4.0)),
        source_kind=source_kind,
        route_id=route_id,
    )


def test_blinded_audit_is_balanced_deterministic_and_hides_the_key() -> None:
    references, generated = _audit_sources()
    finalists = ("cheap", "close", "tradeoff")
    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=finalists,
        seed="audit-seed",
    )
    reordered_packet, reordered_key = build_blinded_audit(
        tuple(reversed(references)),
        tuple(reversed(generated)),
        finalist_order=finalists,
        seed="audit-seed",
    )

    assert packet == reordered_packet
    assert key == reordered_key
    assert len(packet.items) == 200
    assert sum(entry.source_kind is AuditSourceKind.REFERENCE for entry in key.entries) == 100
    route_counts = {
        route: sum(entry.route_id == route for entry in key.entries)
        for route in finalists
    }
    assert route_counts == {"cheap": 34, "close": 33, "tradeoff": 33}
    pair_counts = {
        entry.pair_id: sum(other.pair_id == entry.pair_id for other in key.entries)
        for entry in key.entries
    }
    assert set(pair_counts.values()) == {2}
    validate_audit_pair(packet, key)

    rendered = render_audit_html(packet)
    assert "Source identities are not embedded here" in rendered
    assert "Source prompt" in rendered
    assert "Base NLL" in rendered
    assert "automated_style_scores" in rendered
    assert "JSON.stringify({audit_sha256:auditSha256,decisions:decisions})" in rendered
    assert "coherence_rating:Number" in rendered
    assert all(route not in rendered for route in finalists)
    assert all(entry.source_id not in rendered for entry in key.entries)

    tampered = replace(
        packet,
        items=(replace(packet.items[0], story_text="Changed."),) + packet.items[1:],
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_audit_pair(tampered, key)


def test_blinded_audit_reassigns_greedy_choices_to_find_distinct_pairs() -> None:
    seed = "allocation-seed"
    routes = ("broad", "constrained", "separate")
    broad = tuple(
        _allocation_audit_record(f"pair-{index}", route_id=routes[0])
        for index in range(4)
    )
    old_greedy_pairs = tuple(
        record.pair_id
        for record in sorted(
            broad,
            key=lambda item: (
                sha256(
                    f"{seed}\0audit-generated:{routes[0]}\0{item.source_id}".encode()
                ).digest(),
                item.source_id,
            ),
        )[:2]
    )
    generated = broad + tuple(
        _allocation_audit_record(pair_id, route_id=routes[1])
        for pair_id in old_greedy_pairs
    ) + tuple(
        _allocation_audit_record(f"pair-{index}", route_id=routes[2])
        for index in (4, 5)
    )
    references = tuple(
        _allocation_audit_record(f"pair-{index}") for index in range(6)
    )

    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=routes,
        seed=seed,
        reference_count=6,
        generated_count=6,
    )
    repeated_packet, repeated_key = build_blinded_audit(
        tuple(reversed(references)),
        tuple(reversed(generated)),
        finalist_order=routes,
        seed=seed,
        reference_count=6,
        generated_count=6,
    )

    assert (packet, key) == (repeated_packet, repeated_key)
    selected_by_route = {
        route_id: {
            entry.pair_id
            for entry in key.entries
            if entry.route_id == route_id
        }
        for route_id in routes
    }
    assert {len(values) for values in selected_by_route.values()} == {2}
    assert selected_by_route["constrained"] == set(old_greedy_pairs)
    assert selected_by_route["broad"].isdisjoint(old_greedy_pairs)
    assert len(set().union(*selected_by_route.values())) == 6


def test_blinded_audit_names_mathematically_infeasible_pair_allocation() -> None:
    routes = ("first", "second", "third")
    references = tuple(
        _allocation_audit_record(f"pair-{index}") for index in range(6)
    )
    generated = tuple(
        _allocation_audit_record(pair_id, route_id=route_id)
        for route_id, pair_ids in (
            (routes[0], ("pair-0", "pair-1")),
            (routes[1], ("pair-0", "pair-1")),
            (routes[2], ("pair-2", "pair-3")),
        )
        for pair_id in pair_ids
    )

    with pytest.raises(AuditAllocationError, match="globally distinct"):
        build_blinded_audit(
            references,
            generated,
            finalist_order=routes,
            seed="infeasible-allocation",
            reference_count=6,
            generated_count=6,
        )


def test_audit_decisions_gate_route_selection_and_exact_digest_approval() -> None:
    references, generated = _audit_sources()
    finalists = ("cheap", "close", "tradeoff")
    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=finalists,
        seed="audit-seed",
    )
    decisions = tuple(
        AuditDecision(
            item.item_id,
            ts_like_accepted=True,
            simplicity_rating=4,
            coherence_rating=5,
            source_guess=AuditSourceGuess.REFERENCE,
        )
        for item in packet.items
    )
    decision_set = build_decision_set(packet, tuple(reversed(decisions)))
    evaluation = evaluate_blinded_audit(
        packet,
        key,
        decision_set,
        selectable_route_ids=finalists,
    )

    assert evaluation.passed
    assert evaluation.generated_acceptance_rate == 1.0
    assert evaluation.source_discrimination_accuracy == 0.5
    assert select_human_approved_route(
        evaluation,
        qualified_route_ids=finalists,
        projected_costs=(("cheap", 1.0), ("close", 3.0), ("tradeoff", 2.0)),
    ) == "cheap"

    approval = AuditApproval(
        packet.audit_sha256,
        decision_set.decision_sha256,
        approved_by="human-rater",
        approved_at_utc="2026-07-18T12:00:00Z",
        approved=True,
    )
    validate_audit_approval(packet, key, decision_set, evaluation, approval)

    wrong_approval = replace(approval, decision_sha256="0" * 64)
    with pytest.raises(ValueError, match="approval digests"):
        validate_audit_approval(
            packet, key, decision_set, evaluation, wrong_approval
        )
    with pytest.raises(ValueError, match="every audit item"):
        build_decision_set(packet, decisions[:-1])


def test_each_selectable_route_must_clear_human_acceptance_gate() -> None:
    references, generated = _audit_sources()
    finalists = ("cheap", "close", "tradeoff")
    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=finalists,
        seed="audit-seed",
    )
    keyed = {entry.item_id: entry for entry in key.entries}
    rejected_cheap_ids = {
        entry.item_id
        for entry in key.entries
        if entry.route_id == "cheap"
    }
    rejected_cheap_ids = set(sorted(rejected_cheap_ids)[:8])
    decisions = tuple(
        AuditDecision(
            item.item_id,
            ts_like_accepted=item.item_id not in rejected_cheap_ids,
            simplicity_rating=4,
            coherence_rating=4,
            source_guess=(
                AuditSourceGuess.GENERATED
                if keyed[item.item_id].source_kind is AuditSourceKind.REFERENCE
                else AuditSourceGuess.REFERENCE
            ),
        )
        for item in packet.items
    )
    decision_set = build_decision_set(packet, decisions)
    evaluation = evaluate_blinded_audit(
        packet,
        key,
        decision_set,
        selectable_route_ids=finalists,
    )

    assert evaluation.generated_acceptance_rate == pytest.approx(0.92)
    assert not evaluation.passed
    assert evaluation.failures == ("per_route_acceptance_rate",)


def test_audit_compares_all_finalists_but_only_qualified_routes_are_selectable() -> None:
    references, generated = _audit_sources()
    finalists = ("cheap", "close", "tradeoff")
    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=finalists,
        seed="audit-seed",
    )
    decisions = build_decision_set(
        packet,
        tuple(
            AuditDecision(
                item.item_id,
                ts_like_accepted=True,
                simplicity_rating=4,
                coherence_rating=4,
                source_guess=AuditSourceGuess.REFERENCE,
            )
            for item in packet.items
        ),
    )

    evaluation = evaluate_blinded_audit(
        packet,
        key,
        decisions,
        selectable_route_ids=("cheap", "tradeoff"),
    )

    assert {entry.route_id for entry in key.entries if entry.route_id} == set(finalists)
    assert tuple(route for route, _ in evaluation.route_acceptance_rates) == (
        "cheap",
        "tradeoff",
    )
