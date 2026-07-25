"""Frozen prompt layouts and answer-scoring rules shared by every catalog."""

from __future__ import annotations

from dataclasses import dataclass

from apm.data.text.tinyworlds_q_semantic.catalog import make_query_template
from apm.data.text.tinyworlds_q_semantic.contracts import (
    ConceptDefinition,
    SemanticFact,
    SemanticQueryTemplate,
    record_sha256,
)
from apm.lm.text import TextTokenizer


@dataclass(frozen=True, slots=True)
class SemanticQueryProtocol:
    """Immutable prompt, candidate, and statistical-unit rules."""

    protocol_id: str
    forward_prompt_patterns: tuple[str, ...]
    reverse_prompt_patterns: tuple[str, ...]
    validation_directions: tuple[str, ...]
    test_directions: tuple[str, ...]
    candidate_count: int
    candidate_text_prefix: str
    score_scope: str
    primary_observation_unit: str
    world_weighting: str

    def __post_init__(self) -> None:
        if (
            self.protocol_id != "tinyworlds-q-semantic-query-protocol-v1"
            or len(self.forward_prompt_patterns) != 5
            or len(self.reverse_prompt_patterns) != 3
            or self.validation_directions != ("forward", "forward", "reverse")
            or self.test_directions
            != ("forward", "forward", "forward", "reverse", "reverse")
            or self.candidate_count != 4
            or self.candidate_text_prefix != " "
            or self.score_scope != "answer-suffix-tokens-only"
            or self.primary_observation_unit != "fact-mean-over-paraphrases"
            or self.world_weighting != "equal"
        ):
            raise ValueError("TinyWorlds-Q query protocol changed")
        if any(
            type(pattern) is not str or "{prompt}" not in pattern
            for pattern in self.forward_prompt_patterns
            + self.reverse_prompt_patterns
        ):
            raise ValueError("query prompt patterns must interpolate the reviewed prompt")

    @property
    def protocol_sha256(self) -> str:
        """Hash every template and answer-scoring rule."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical query protocol record."""
        return {
            "candidate_count": self.candidate_count,
            "candidate_text_prefix": self.candidate_text_prefix,
            "forward_prompt_patterns": list(self.forward_prompt_patterns),
            "primary_observation_unit": self.primary_observation_unit,
            "protocol_id": self.protocol_id,
            "reverse_prompt_patterns": list(self.reverse_prompt_patterns),
            "score_scope": self.score_scope,
            "test_directions": list(self.test_directions),
            "validation_directions": list(self.validation_directions),
            "world_weighting": self.world_weighting,
        }


REGISTERED_QUERY_PROTOCOL = SemanticQueryProtocol(
    protocol_id="tinyworlds-q-semantic-query-protocol-v1",
    forward_prompt_patterns=(
        "{prompt}",
        "Answer this question about {plural}. {prompt}",
        "Choose the correct response. {prompt}",
        "Use general knowledge about {plural}. {prompt}",
        "Select the best completion for this {concept_id} fact. {prompt}",
    ),
    reverse_prompt_patterns=(
        "{prompt}",
        "Choose the concept that fits this fact. {prompt}",
        "Identify the matching concept. {prompt}",
    ),
    validation_directions=("forward", "forward", "reverse"),
    test_directions=("forward", "forward", "forward", "reverse", "reverse"),
    candidate_count=4,
    candidate_text_prefix=" ",
    score_scope="answer-suffix-tokens-only",
    primary_observation_unit="fact-mean-over-paraphrases",
    world_weighting="equal",
)


def make_registered_fact_templates(
    fact: SemanticFact,
    concept: ConceptDefinition,
    *,
    forward_prompt: str,
    forward_distractors: tuple[str, str, str],
    reverse_prompt: str,
    reverse_distractors: tuple[str, str, str],
    tokenizer: TextTokenizer,
    protocol: SemanticQueryProtocol = REGISTERED_QUERY_PROTOCOL,
) -> tuple[SemanticQueryTemplate, ...]:
    """Render the frozen three-validation/five-test layout for one reviewed fact."""
    if fact.concept_id != concept.concept_id:
        raise ValueError("query fact and concept do not match")
    plural = concept.surface_forms[1]
    substitutions = {
        "concept_id": concept.concept_id,
        "plural": plural,
        "prompt": forward_prompt,
    }
    forward_prompts = tuple(
        pattern.format(**substitutions)
        for pattern in protocol.forward_prompt_patterns
    )
    reverse_prompts = tuple(
        pattern.format(**{**substitutions, "prompt": reverse_prompt})
        for pattern in protocol.reverse_prompt_patterns
    )
    layouts = (
        (
            "validation",
            (
                ("forward", forward_prompts[0], forward_distractors),
                ("forward", forward_prompts[1], forward_distractors),
                ("reverse", reverse_prompts[0], reverse_distractors),
            ),
        ),
        (
            "test",
            (
                ("forward", forward_prompts[2], forward_distractors),
                ("forward", forward_prompts[3], forward_distractors),
                ("forward", forward_prompts[4], forward_distractors),
                ("reverse", reverse_prompts[1], reverse_distractors),
                ("reverse", reverse_prompts[2], reverse_distractors),
            ),
        ),
    )
    return tuple(
        make_query_template(
            fact,
            concept,
            template_id=f"{fact.fact_id}-{split}-{paraphrase_index:02d}",
            direction=direction,
            prompt_text=prompt,
            distractors=distractors,
            split=split,
            paraphrase_index=paraphrase_index,
            tokenizer=tokenizer,
        )
        for split, split_layout in layouts
        for paraphrase_index, (direction, prompt, distractors) in enumerate(
            split_layout
        )
    )


__all__ = [
    "REGISTERED_QUERY_PROTOCOL",
    "SemanticQueryProtocol",
    "make_registered_fact_templates",
]
