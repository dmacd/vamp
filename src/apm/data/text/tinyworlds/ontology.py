"""Typed predicate registry and ontology validation for TinyWorlds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apm.data.text.tinyworlds.schema import (
    AtomPattern,
    Entity,
    EntityId,
    EntityTypeId,
    GroundAtom,
    HornRule,
    PredicateId,
    QueryAst,
    Variable,
)


class PredicateKind(str, Enum):
    """Whether a predicate is ordinary or carries explicit context."""

    ORDINARY = "ordinary"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True, slots=True)
class PredicateSignature:
    """One predicate's fixed typed argument signature."""

    predicate_id: PredicateId
    argument_types: tuple[EntityTypeId, ...]
    kind: PredicateKind = PredicateKind.ORDINARY

    def __post_init__(self) -> None:
        if type(self.predicate_id) is not PredicateId:
            raise TypeError("predicate_id must be a PredicateId")
        if type(self.argument_types) is not tuple:
            raise TypeError("predicate argument_types must be a tuple")
        if any(type(type_id) is not EntityTypeId for type_id in self.argument_types):
            raise TypeError(
                "predicate argument_types must contain EntityTypeId values"
            )
        if type(self.kind) is not PredicateKind:
            raise TypeError("predicate kind must be a PredicateKind")
        if self.kind is PredicateKind.ORDINARY and len(self.argument_types) not in (
            1,
            2,
        ):
            raise ValueError("ordinary predicates must be unary or binary")
        if self.kind is PredicateKind.CONTEXTUAL and len(self.argument_types) != 3:
            raise ValueError("contextual predicates must be explicitly ternary")

    @property
    def arity(self) -> int:
        """Return the signature's fixed number of arguments."""
        return len(self.argument_types)


@dataclass(frozen=True, slots=True)
class PredicateRegistry:
    """Canonical closed registry of entity types and predicate signatures."""

    entity_types: tuple[EntityTypeId, ...]
    context_type_id: EntityTypeId
    predicates: tuple[PredicateSignature, ...]

    def __post_init__(self) -> None:
        if type(self.entity_types) is not tuple or not self.entity_types:
            raise ValueError("registry entity_types must be a nonempty tuple")
        if any(type(type_id) is not EntityTypeId for type_id in self.entity_types):
            raise TypeError("registry entity_types must contain EntityTypeId values")
        if len(set(self.entity_types)) != len(self.entity_types):
            raise ValueError("registry entity types must be unique")
        if type(self.context_type_id) is not EntityTypeId:
            raise TypeError("context_type_id must be an EntityTypeId")
        if self.context_type_id not in self.entity_types:
            raise ValueError("context_type_id must belong to registry entity_types")
        if type(self.predicates) is not tuple or not self.predicates:
            raise ValueError("registry predicates must be a nonempty tuple")
        if any(type(predicate) is not PredicateSignature for predicate in self.predicates):
            raise TypeError(
                "registry predicates must contain PredicateSignature values"
            )
        predicate_ids = tuple(predicate.predicate_id for predicate in self.predicates)
        if len(set(predicate_ids)) != len(predicate_ids):
            raise ValueError("registry predicate IDs must be unique")
        known_types = frozenset(self.entity_types)
        for predicate in self.predicates:
            unknown_types = tuple(
                type_id
                for type_id in predicate.argument_types
                if type_id not in known_types
            )
            if unknown_types:
                raise ValueError(
                    f"predicate {predicate.predicate_id} uses unknown types: "
                    f"{unknown_types}"
                )
            if (
                predicate.kind is PredicateKind.CONTEXTUAL
                and predicate.argument_types[-1] != self.context_type_id
            ):
                raise ValueError(
                    "a contextual predicate's third argument must use context_type_id"
                )
        object.__setattr__(self, "entity_types", tuple(sorted(self.entity_types)))
        object.__setattr__(
            self,
            "predicates",
            tuple(sorted(self.predicates, key=lambda item: item.predicate_id)),
        )

    def signature(self, predicate_id: PredicateId) -> PredicateSignature:
        """Return one registered signature or reject an unknown predicate."""
        if type(predicate_id) is not PredicateId:
            raise TypeError("predicate_id must be a PredicateId")
        for predicate in self.predicates:
            if predicate.predicate_id == predicate_id:
                return predicate
        raise KeyError(f"unknown predicate ID: {predicate_id}")

    def validate_entities(self, entities: tuple[Entity, ...]) -> None:
        """Validate unique entity IDs and declared registry types."""
        _entity_index(entities, frozenset(self.entity_types))

    def validate_ground_atom(
        self,
        atom: GroundAtom,
        entities: tuple[Entity, ...],
    ) -> None:
        """Validate a ground atom's arity, references, and exact types."""
        if type(atom) is not GroundAtom:
            raise TypeError("atom must be a GroundAtom")
        entity_by_id = _entity_index(entities, frozenset(self.entity_types))
        signature = self.signature(atom.predicate_id)
        if len(atom.arguments) != signature.arity:
            raise ValueError(
                f"predicate {atom.predicate_id} expects arity {signature.arity}"
            )
        for argument, expected_type in zip(
            atom.arguments,
            signature.argument_types,
        ):
            entity = entity_by_id.get(argument)
            if entity is None:
                raise ValueError(f"ground atom references unknown entity: {argument}")
            if entity.entity_type != expected_type:
                raise ValueError(
                    f"predicate {atom.predicate_id} argument {argument} expects "
                    f"type {expected_type}, got {entity.entity_type}"
                )

    def validate_pattern(
        self,
        pattern: AtomPattern,
        entities: tuple[Entity, ...],
        variable_types: dict[Variable, EntityTypeId] | None = None,
    ) -> dict[Variable, EntityTypeId]:
        """Validate one pattern and return consistently inferred variable types."""
        if type(pattern) is not AtomPattern:
            raise TypeError("pattern must be an AtomPattern")
        if variable_types is not None and (
            type(variable_types) is not dict
            or any(
                type(variable) is not Variable
                or type(type_id) is not EntityTypeId
                or type_id not in self.entity_types
                for variable, type_id in variable_types.items()
            )
        ):
            raise TypeError(
                "variable_types must map Variable values to registered EntityTypeId values"
            )
        inferred = {} if variable_types is None else dict(variable_types)
        entity_by_id = _entity_index(entities, frozenset(self.entity_types))
        signature = self.signature(pattern.predicate_id)
        if len(pattern.arguments) != signature.arity:
            raise ValueError(
                f"predicate {pattern.predicate_id} expects arity {signature.arity}"
            )
        for argument, expected_type in zip(
            pattern.arguments,
            signature.argument_types,
        ):
            if type(argument) is EntityId:
                entity = entity_by_id.get(argument)
                if entity is None:
                    raise ValueError(
                        f"atom pattern references unknown entity: {argument}"
                    )
                if entity.entity_type != expected_type:
                    raise ValueError(
                        f"pattern constant {argument} expects type {expected_type}, "
                        f"got {entity.entity_type}"
                    )
            else:
                prior_type = inferred.get(argument)
                if prior_type is not None and prior_type != expected_type:
                    raise ValueError(
                        f"variable {argument.name} has incompatible types "
                        f"{prior_type} and {expected_type}"
                    )
                inferred[argument] = expected_type
        return inferred

    def validate_rule(
        self,
        rule: HornRule,
        entities: tuple[Entity, ...],
    ) -> None:
        """Validate all atoms in one positive-safe rule under shared types."""
        if type(rule) is not HornRule:
            raise TypeError("rule must be a HornRule")
        variable_types: dict[Variable, EntityTypeId] = {}
        for body_atom in rule.body:
            variable_types = self.validate_pattern(
                body_atom,
                entities,
                variable_types,
            )
        self.validate_pattern(rule.head, entities, variable_types)

    def answer_type(
        self,
        query: QueryAst,
        entities: tuple[Entity, ...],
    ) -> EntityTypeId:
        """Validate a query and return the projected variable's entity type."""
        if type(query) is not QueryAst:
            raise TypeError("query must be a QueryAst")
        variable_types: dict[Variable, EntityTypeId] = {}
        for clause in query.clauses:
            variable_types = self.validate_pattern(
                clause,
                entities,
                variable_types,
            )
        return variable_types[query.answer_variable]


def _entity_index(
    entities: tuple[Entity, ...],
    known_types: frozenset[EntityTypeId],
) -> dict[EntityId, Entity]:
    if type(entities) is not tuple:
        raise TypeError("entities must be a tuple")
    if any(type(entity) is not Entity for entity in entities):
        raise TypeError("entities must contain Entity values")
    entity_by_id = {entity.entity_id: entity for entity in entities}
    if len(entity_by_id) != len(entities):
        raise ValueError("entity IDs must be unique")
    unknown_types = tuple(
        entity.entity_type
        for entity in entities
        if entity.entity_type not in known_types
    )
    if unknown_types:
        raise ValueError(f"entities use unknown registry types: {unknown_types}")
    return entity_by_id


__all__ = [
    "PredicateKind",
    "PredicateRegistry",
    "PredicateSignature",
]
