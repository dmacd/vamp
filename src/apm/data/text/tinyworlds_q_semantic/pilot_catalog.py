"""Approved rabbit/horse fact and query catalog compilation."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.approval import PrimaryReviewApproval
from apm.data.text.tinyworlds_q_semantic.catalog import (
    build_reviewed_catalog,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    FactReviewDecision,
    RejectedFactCandidate,
    SemanticFact,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.manifests import PILOT_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    ReverseReviewApproval,
    SemanticReverseReview,
)
from apm.data.text.tinyworlds_q_semantic.review import SemanticReviewPacket
from apm.data.text.tinyworlds_q_semantic.query_protocol import (
    make_registered_fact_templates,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    PILOT_SHORTLIST_SPECS,
    SemanticReviewShortlist,
)
from apm.lm.text import TextTokenizer


def build_approved_pilot_catalog(
    *,
    review_packet: SemanticReviewPacket,
    shortlist: SemanticReviewShortlist,
    primary_approval: PrimaryReviewApproval,
    reverse_review: SemanticReverseReview,
    reverse_approval: ReverseReviewApproval,
    tokenizer: TextTokenizer,
) -> SemanticQueryCatalog:
    """Compile an official pilot catalog after both explicit review decisions."""
    _validate_authority_chain(
        review_packet,
        shortlist,
        primary_approval,
        reverse_review,
        reverse_approval,
    )
    candidates = {
        (candidate.concept_id, candidate.predicate): candidate
        for candidate in review_packet.candidates
    }
    primary_specs = tuple(
        spec for spec in PILOT_SHORTLIST_SPECS if spec.priority == "primary"
    )
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
    reverse_by_proposal = {
        proposal.spec.proposal_id: proposal.spec
        for proposal in reverse_review.proposals
    }
    templates = tuple(
        template
        for spec in primary_specs
        for template in make_registered_fact_templates(
            fact_by_proposal[spec.proposal_id],
            next(
                concept
                for concept in PILOT_CONCEPTS
                if concept.concept_id == spec.concept_id
            ),
            forward_prompt=spec.forward_prompt,
            forward_distractors=spec.distractors,
            reverse_prompt=reverse_by_proposal[spec.proposal_id].clue_prompt,
            reverse_distractors=reverse_by_proposal[spec.proposal_id].distractors,
            tokenizer=tokenizer,
        )
    )
    rejected_candidates = tuple(
        RejectedFactCandidate(
            candidate_id=candidate.candidate_id,
            concept_id=spec.concept_id,
            predicate=spec.source_predicate,
            reason=(
                "Not selected because all twelve primary fact slots were approved."
            ),
            reviewer=primary_approval.reviewer,
            evidence_group_sha256=candidate.supporting_story_groups,
        )
        for spec in PILOT_SHORTLIST_SPECS
        if spec.priority == "backup"
        for candidate in (candidates[(spec.concept_id, spec.source_predicate)],)
    )
    return build_reviewed_catalog(
        review_packet=review_packet,
        facts=facts,
        templates=templates,
        reviews=reviews,
        rejected_candidates=rejected_candidates,
    )


def _validate_authority_chain(
    review_packet: SemanticReviewPacket,
    shortlist: SemanticReviewShortlist,
    primary_approval: PrimaryReviewApproval,
    reverse_review: SemanticReverseReview,
    reverse_approval: ReverseReviewApproval,
) -> None:
    if review_packet.concepts != PILOT_CONCEPTS:
        raise ValueError("pilot catalog requires the exact pilot concepts")
    if shortlist.review_packet_sha256 != review_packet.packet_sha256:
        raise ValueError("pilot shortlist changed its review packet")
    if primary_approval.shortlist_sha256 != shortlist.shortlist_sha256:
        raise ValueError("pilot primary approval changed its shortlist")
    if (
        reverse_review.shortlist_sha256 != shortlist.shortlist_sha256
        or reverse_review.primary_approval_sha256
        != primary_approval.approval_sha256
        or reverse_approval.reverse_review_sha256
        != reverse_review.reverse_review_sha256
    ):
        raise ValueError("pilot reverse authority chain changed")
    if primary_approval.reviewer != reverse_approval.reviewer:
        raise ValueError("pilot fact and reverse approvals require one reviewer identity")
__all__ = ["build_approved_pilot_catalog"]
