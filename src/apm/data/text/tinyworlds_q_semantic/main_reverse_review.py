"""Fact-specific reverse choices for the approved five-world main shortlist."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.approval import PrimaryReviewApproval
from apm.data.text.tinyworlds_q_semantic.main_shortlist import MAIN_SHORTLIST_SPECS
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    ReverseChoiceSpec,
    SemanticReverseReview,
    build_reverse_review,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import SemanticReviewShortlist
from apm.lm.text import TextTokenizer


def _reverse_spec(
    proposal_id: str,
    concept_id: str,
    clue_prompt: str,
    distractors: tuple[str, str, str],
) -> ReverseChoiceSpec:
    return ReverseChoiceSpec(proposal_id, concept_id, clue_prompt, distractors)


MAIN_REVERSE_CHOICE_SPECS = tuple(
    _reverse_spec(*values)
    for values in (
        ("cat-proposal-01", "cat", "Which concept is an animal? Answer:", ("robot", "spoon", "rock")),
        ("cat-proposal-02", "cat", "Which concept normally has fur? Answer:", ("fish", "bird", "turtle")),
        ("cat-proposal-03", "cat", "Which concept normally has a tail? Answer:", ("spider", "worm", "ant")),
        ("cat-proposal-04", "cat", "Which concept normally has paws? Answer:", ("fish", "snake", "worm")),
        ("cat-proposal-05", "cat", "Which concept normally has claws? Answer:", ("fish", "snake", "worm")),
        ("cat-proposal-06", "cat", "Which concept commonly meows? Answer:", ("dog", "cow", "pig")),
        ("cat-proposal-07", "cat", "Which concept commonly purrs? Answer:", ("dog", "cow", "duck")),
        ("cat-proposal-08", "cat", "Which concept commonly climbs using paws and claws? Answer:", ("whale", "shark", "dolphin")),
        ("cat-proposal-09", "cat", "Which concept is the traditional adversary of mice? Answer:", ("spoon", "rock", "chair")),
        ("cat-proposal-10", "cat", "Which concept is commonly given fish as food? Answer:", ("cow", "horse", "rabbit")),
        ("cat-proposal-11", "cat", "Which concept can be kept as a domesticated pet? Answer:", ("whale", "shark", "dolphin")),
        ("cat-proposal-12", "cat", "Which concept is generally a small animal? Answer:", ("elephant", "whale", "camel")),
        ("dog-proposal-01", "dog", "Which concept is an animal? Answer:", ("robot", "spoon", "rock")),
        ("dog-proposal-02", "dog", "Which concept normally has fur? Answer:", ("fish", "bird", "turtle")),
        ("dog-proposal-03", "dog", "Which concept normally has a tail? Answer:", ("spider", "worm", "ant")),
        ("dog-proposal-04", "dog", "Which concept normally has paws? Answer:", ("fish", "snake", "worm")),
        ("dog-proposal-05", "dog", "Which concept commonly barks? Answer:", ("cat", "cow", "pig")),
        ("dog-proposal-06", "dog", "Which concept commonly runs on four legs? Answer:", ("fish", "snake", "whale")),
        ("dog-proposal-07", "dog", "Which concept commonly greets people by wagging its tail? Answer:", ("fish", "snake", "worm")),
        ("dog-proposal-08", "dog", "Which concept is traditionally associated with bones? Answer:", ("cow", "horse", "rabbit")),
        ("dog-proposal-09", "dog", "Which concept can be kept as a domesticated pet? Answer:", ("whale", "shark", "dolphin")),
        ("dog-proposal-10", "dog", "Which concept is known as a friendly household companion? Answer:", ("shark", "spider", "worm")),
        ("dog-proposal-11", "dog", "Which concept is known as a loyal companion? Answer:", ("worm", "snail", "feather")),
        ("dog-proposal-12", "dog", "Which concept can be trained to guard people or places? Answer:", ("worm", "snail", "spider")),
        ("bird-proposal-01", "bird", "Which concept is an animal? Answer:", ("robot", "spoon", "rock")),
        ("bird-proposal-02", "bird", "Which concept normally has feathers? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-03", "bird", "Which concept normally has wings? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-04", "bird", "Which concept normally has a beak? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-05", "bird", "Which concept can normally fly through the air? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-06", "bird", "Which concept commonly sings? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-07", "bird", "Which concept uses a nest as a home for its young? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-08", "bird", "Which concept lays eggs? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-09", "bird", "Which concept commonly eats seeds? Answer:", ("lion", "shark", "whale")),
        ("bird-proposal-10", "bird", "Which concept is traditionally shown pulling worms from soil? Answer:", ("cat", "dog", "horse")),
        ("bird-proposal-11", "bird", "Which concept is often a small animal? Answer:", ("elephant", "whale", "camel")),
        ("bird-proposal-12", "bird", "Which concept hatches from eggs? Answer:", ("cat", "dog", "horse")),
        ("robot-proposal-01", "robot", "Which concept can be a programmable toy machine? Answer:", ("spoon", "rock", "book")),
        ("robot-proposal-02", "robot", "Which concept is commonly depicted as a metal machine? Answer:", ("cat", "dog", "bird")),
        ("robot-proposal-03", "robot", "Which nonliving machine can move when powered? Answer:", ("spoon", "rock", "book")),
        ("robot-proposal-04", "robot", "Which machine can be programmed to talk? Answer:", ("spoon", "rock", "chair")),
        ("robot-proposal-05", "robot", "Which machine can be programmed to help people? Answer:", ("spoon", "rock", "chair")),
        ("robot-proposal-06", "robot", "Which machine can be programmed to play games? Answer:", ("spoon", "rock", "chair")),
        ("robot-proposal-07", "robot", "Which machine can be programmed to perform work? Answer:", ("spoon", "rock", "chair")),
        ("robot-proposal-08", "robot", "Which machine can be built to be strong? Answer:", ("cat", "dog", "bird")),
        ("robot-proposal-09", "robot", "Which machine is often depicted with a shiny body? Answer:", ("cat", "dog", "bird")),
        ("robot-proposal-10", "robot", "Which concept commonly uses a battery for power? Answer:", ("cat", "dog", "bird")),
        ("robot-proposal-11", "robot", "Which programmable machine can be broken? Answer:", ("cat", "dog", "bird")),
        ("robot-proposal-12", "robot", "Which broken machine can be fixed to work again? Answer:", ("cat", "dog", "bird")),
        ("dragon-proposal-01", "dragon", "Which fantasy creature is often described as big? Answer:", ("fairy", "elf", "sprite")),
        ("dragon-proposal-02", "dragon", "Which fantasy creature is strongly associated with fire? Answer:", ("elf", "sprite", "ogre")),
        ("dragon-proposal-03", "dragon", "Which fantasy creature is commonly associated with caves? Answer:", ("fairy", "elf", "sprite")),
        ("dragon-proposal-04", "dragon", "Which fantasy creature can normally fly? Answer:", ("troll", "goblin", "ogre")),
        ("dragon-proposal-05", "dragon", "Which fantasy creature normally has wings? Answer:", ("troll", "goblin", "ogre")),
        ("dragon-proposal-06", "dragon", "Which fantasy creature can have scales? Answer:", ("fairy", "elf", "goblin")),
        ("dragon-proposal-07", "dragon", "Which fantasy creature commonly roars? Answer:", ("fairy", "elf", "sprite")),
        ("dragon-proposal-08", "dragon", "Which fantasy creature is often described as physically strong? Answer:", ("fairy", "sprite", "ghost")),
        ("dragon-proposal-09", "dragon", "Which fantasy creature is often described as scary? Answer:", ("fairy", "elf", "sprite")),
        ("dragon-proposal-10", "dragon", "Which fantasy creature is often described as fierce? Answer:", ("fairy", "elf", "sprite")),
        ("dragon-proposal-11", "dragon", "Which concept is a magical fantasy creature? Answer:", ("cat", "dog", "bird")),
        ("dragon-proposal-12", "dragon", "Which fantasy creature is associated with guarding treasure? Answer:", ("fairy", "elf", "sprite")),
    )
)


def build_main_reverse_review(
    shortlist: SemanticReviewShortlist,
    approval: PrimaryReviewApproval,
    tokenizer: TextTokenizer,
) -> SemanticReverseReview:
    """Compile the five-world fact-specific reverse-choice decision surface."""
    return build_reverse_review(
        shortlist,
        approval,
        MAIN_SHORTLIST_SPECS,
        MAIN_REVERSE_CHOICE_SPECS,
        tokenizer,
    )


__all__ = [
    "MAIN_REVERSE_CHOICE_SPECS",
    "build_main_reverse_review",
]
