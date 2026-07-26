"""Manifest-driven compilation of fully approved semantic catalogs."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.approval import PrimaryReviewApproval
from apm.data.text.tinyworlds_q_semantic.catalog import build_reviewed_catalog
from apm.data.text.tinyworlds_q_semantic.contracts import (
    ConceptDefinition,
    FactReviewDecision,
    RejectedFactCandidate,
    SemanticFact,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.query_protocol import (
    make_registered_fact_templates,
)
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    ReverseReviewApproval,
    SemanticReverseReview,
)
from apm.data.text.tinyworlds_q_semantic.review import SemanticReviewPacket
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    ReviewShortlistSpec,
    SemanticReviewShortlist,
)
from apm.lm.text import TextTokenizer


def build_approved_catalog(
    *,
    concepts: tuple[ConceptDefinition, ...],
    shortlist_specs: tuple[ReviewShortlistSpec, ...],
    review_packet: SemanticReviewPacket,
    shortlist: SemanticReviewShortlist,
    primary_approval: PrimaryReviewApproval,
    reverse_review: SemanticReverseReview,
    reverse_approval: ReverseReviewApproval,
    tokenizer: TextTokenizer,
    parent: SemanticQueryCatalog | None = None,
) -> SemanticQueryCatalog:
    """Compile one manifest after authenticating both human approval layers."""
    primary_specs = tuple(
        spec for spec in shortlist_specs if spec.priority == "primary"
    )
    _validate_authority_chain(
        concepts=concepts,
        shortlist_specs=shortlist_specs,
        primary_specs=primary_specs,
        review_packet=review_packet,
        shortlist=shortlist,
        primary_approval=primary_approval,
        reverse_review=reverse_review,
        reverse_approval=reverse_approval,
    )
    candidates = {
        (candidate.concept_id, candidate.predicate): candidate
        for candidate in review_packet.candidates
    }
    selected_keys = tuple(
        (spec.concept_id, spec.source_predicate) for spec in shortlist_specs
    )
    missing_keys = tuple(key for key in selected_keys if key not in candidates)
    if missing_keys:
        raise ValueError(f"catalog shortlist lost review evidence for {missing_keys}")
    facts = tuple(
        SemanticFact(
            fact_id=spec.proposal_id.replace("proposal", "fact"),
            source_candidate_id=candidates[
                (spec.concept_id, spec.source_predicate)
            ].candidate_id,
            concept_id=spec.concept_id,
            relation_category=spec.relation_category,
            answer_type=spec.answer_type,
            canonical_answer=spec.canonical_answer,
            accepted_forms=spec.accepted_forms,
            trigger_forms=spec.trigger_forms,
            supporting_story_groups=candidates[
                (spec.concept_id, spec.source_predicate)
            ].supporting_story_groups,
            evidence=candidates[(spec.concept_id, spec.source_predicate)].evidence,
        )
        for spec in primary_specs
    )
    reviews = tuple(
        FactReviewDecision(
            fact_id=fact.fact_id,
            reviewer=reverse_approval.reviewer,
            reviewed_at=reverse_approval.reviewed_at,
            truth_approved=True,
            answer_forms_approved=True,
            trigger_closure_approved=True,
            distractors_approved=True,
            evidence_approved=True,
        )
        for fact in facts
    )
    fact_by_proposal = {
        spec.proposal_id: fact for spec, fact in zip(primary_specs, facts)
    }
    concept_by_id = {concept.concept_id: concept for concept in concepts}
    reverse_by_proposal = {
        proposal.spec.proposal_id: proposal.spec
        for proposal in reverse_review.proposals
    }
    templates = tuple(
        template
        for spec in primary_specs
        for template in make_registered_fact_templates(
            fact_by_proposal[spec.proposal_id],
            concept_by_id[spec.concept_id],
            forward_prompt=spec.forward_prompt,
            forward_distractors=spec.distractors,
            reverse_prompt=reverse_by_proposal[spec.proposal_id].clue_prompt,
            reverse_distractors=reverse_by_proposal[spec.proposal_id].distractors,
            tokenizer=tokenizer,
        )
    )
    rejected_candidates = tuple(
        RejectedFactCandidate(
            candidate_id=candidates[
                (spec.concept_id, spec.source_predicate)
            ].candidate_id,
            concept_id=spec.concept_id,
            predicate=spec.source_predicate,
            reason=(
                "Not selected because all twelve primary fact slots were approved."
            ),
            reviewer=primary_approval.reviewer,
            evidence_group_sha256=candidates[
                (spec.concept_id, spec.source_predicate)
            ].supporting_story_groups,
        )
        for spec in shortlist_specs
        if spec.priority == "backup"
    )
    return build_reviewed_catalog(
        review_packet=review_packet,
        facts=facts,
        templates=templates,
        reviews=reviews,
        rejected_candidates=rejected_candidates,
        parent=parent,
    )


def _validate_authority_chain(
    *,
    concepts: tuple[ConceptDefinition, ...],
    shortlist_specs: tuple[ReviewShortlistSpec, ...],
    primary_specs: tuple[ReviewShortlistSpec, ...],
    review_packet: SemanticReviewPacket,
    shortlist: SemanticReviewShortlist,
    primary_approval: PrimaryReviewApproval,
    reverse_review: SemanticReverseReview,
    reverse_approval: ReverseReviewApproval,
) -> None:
    if review_packet.concepts != concepts:
        raise ValueError("catalog requires the exact ordered concept manifest")
    if (
        shortlist.review_packet_sha256 != review_packet.packet_sha256
        or tuple(proposal.spec for proposal in shortlist.proposals)
        != shortlist_specs
    ):
        raise ValueError("catalog shortlist changed its reviewed proposal authority")
    expected_primary_ids = tuple(spec.proposal_id for spec in primary_specs)
    if (
        primary_approval.shortlist_sha256 != shortlist.shortlist_sha256
        or primary_approval.approved_proposal_ids != expected_primary_ids
    ):
        raise ValueError("catalog primary approval is incomplete or reordered")
    expected_reverse_pairs = tuple(
        (spec.proposal_id, spec.concept_id) for spec in primary_specs
    )
    actual_reverse_pairs = tuple(
        (proposal.spec.proposal_id, proposal.spec.concept_id)
        for proposal in reverse_review.proposals
    )
    if (
        reverse_review.shortlist_sha256 != shortlist.shortlist_sha256
        or reverse_review.primary_approval_sha256
        != primary_approval.approval_sha256
        or actual_reverse_pairs != expected_reverse_pairs
        or reverse_approval.reverse_review_sha256
        != reverse_review.reverse_review_sha256
        or reverse_approval.approved_proposal_ids != expected_primary_ids
    ):
        raise ValueError("catalog reverse authority chain is incomplete or reordered")
    if primary_approval.reviewer != reverse_approval.reviewer:
        raise ValueError("catalog fact and reverse approvals require one reviewer")


__all__ = ["build_approved_catalog"]
