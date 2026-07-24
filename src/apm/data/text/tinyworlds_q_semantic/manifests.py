"""Registered pilot and main concept manifests for TinyWorlds-Q."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.contracts import ConceptDefinition


PILOT_CONCEPTS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition("rabbit", ("rabbit", "rabbits", "bunny", "bunnies")),
    ConceptDefinition("horse", ("horse", "horses", "pony", "ponies")),
)

MAIN_CONCEPTS: tuple[ConceptDefinition, ...] = (
    ConceptDefinition("cat", ("cat", "cats", "kitten", "kittens")),
    ConceptDefinition("dog", ("dog", "dogs", "puppy", "puppies")),
    ConceptDefinition("bird", ("bird", "birds")),
    ConceptDefinition("robot", ("robot", "robots")),
    ConceptDefinition("dragon", ("dragon", "dragons")),
)

PILOT_CONCEPT_IDS = tuple(concept.concept_id for concept in PILOT_CONCEPTS)
MAIN_CONCEPT_IDS = tuple(concept.concept_id for concept in MAIN_CONCEPTS)


def concept_prefix(
    concepts: tuple[ConceptDefinition, ...],
    world_count: int,
) -> tuple[ConceptDefinition, ...]:
    """Return one nonempty ordered active prefix from a larger catalog."""
    if type(world_count) is not int or not 1 <= world_count <= len(concepts):
        raise ValueError("world_count must select a nonempty concept prefix")
    return concepts[:world_count]


__all__ = [
    "MAIN_CONCEPTS",
    "MAIN_CONCEPT_IDS",
    "PILOT_CONCEPTS",
    "PILOT_CONCEPT_IDS",
    "concept_prefix",
]
