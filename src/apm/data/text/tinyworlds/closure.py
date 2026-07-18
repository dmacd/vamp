"""Deterministic typed closure, query execution, and canonical proofs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import heapq
import json

from apm.data.text.tinyworlds.ontology import PredicateRegistry
from apm.data.text.tinyworlds.schema import (
    AtomId,
    AtomPattern,
    Entity,
    EntityId,
    GroundAtom,
    HornRule,
    PredicateId,
    Proof,
    ProofId,
    ProofStep,
    QueryAst,
    RuleId,
    Variable,
)


SemanticAtomKey = tuple[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ClosureResult:
    """Canonical ground closure and one selected proof for every atom."""

    atoms: tuple[GroundAtom, ...]
    proofs: tuple[Proof, ...]

    def __post_init__(self) -> None:
        if type(self.atoms) is not tuple or type(self.proofs) is not tuple:
            raise TypeError("closure atoms and proofs must be tuples")
        if not self.atoms or len(self.atoms) != len(self.proofs):
            raise ValueError("closure must align one proof with every atom")
        if any(type(atom) is not GroundAtom for atom in self.atoms):
            raise TypeError("closure atoms must contain GroundAtom values")
        if any(type(proof) is not Proof for proof in self.proofs):
            raise TypeError("closure proofs must contain Proof values")
        if len({atom.atom_id for atom in self.atoms}) != len(self.atoms):
            raise ValueError("closure atom IDs must be unique")
        if len({atom.semantic_key for atom in self.atoms}) != len(self.atoms):
            raise ValueError("closure atoms must be semantically unique")
        if len({proof.proof_id for proof in self.proofs}) != len(self.proofs):
            raise ValueError("closure proof IDs must be unique")
        canonical_atoms = tuple(
            sorted(self.atoms, key=lambda atom: (atom.semantic_key, atom.atom_id))
        )
        if canonical_atoms != self.atoms:
            raise ValueError("closure atoms must use canonical semantic order")
        if any(
            atom != proof.conclusion
            for atom, proof in zip(self.atoms, self.proofs)
        ):
            raise ValueError("closure proofs must align with atom order")

    def atom(
        self,
        predicate_id: PredicateId,
        arguments: tuple[EntityId, ...],
    ) -> GroundAtom | None:
        """Return a semantically matching atom when it belongs to the closure."""
        if type(predicate_id) is not PredicateId:
            raise TypeError("predicate_id must be a PredicateId")
        if type(arguments) is not tuple or any(
            type(argument) is not EntityId for argument in arguments
        ):
            raise TypeError("arguments must be a tuple of EntityId values")
        key = (str(predicate_id), tuple(str(argument) for argument in arguments))
        return next((atom for atom in self.atoms if atom.semantic_key == key), None)

    def proof_for(self, atom_id: AtomId) -> Proof:
        """Return the canonical proof for one closure atom ID."""
        if type(atom_id) is not AtomId:
            raise TypeError("atom_id must be an AtomId")
        for atom, proof in zip(self.atoms, self.proofs):
            if atom.atom_id == atom_id:
                return proof
        raise KeyError(f"unknown closure atom ID: {atom_id}")


@dataclass(frozen=True, slots=True)
class _Derivation:
    atom: GroundAtom
    rule_id: RuleId | None
    premises: tuple[_Derivation, ...]
    depth: int


def predicate_topological_order(
    rules: tuple[HornRule, ...],
    predicate_ids: tuple[PredicateId, ...] = (),
) -> tuple[PredicateId, ...]:
    """Return deterministic dependency order or reject any predicate cycle."""
    if type(rules) is not tuple:
        raise TypeError("rules must be a tuple")
    if any(type(rule) is not HornRule for rule in rules):
        raise TypeError("rules must contain HornRule values")
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("rule IDs must be unique")
    if type(predicate_ids) is not tuple or any(
        type(predicate_id) is not PredicateId for predicate_id in predicate_ids
    ):
        raise TypeError("predicate_ids must be a tuple of PredicateId values")
    if len(set(predicate_ids)) != len(predicate_ids):
        raise ValueError("predicate_ids must be unique")

    nodes = set(predicate_ids)
    nodes.update(rule.head.predicate_id for rule in rules)
    nodes.update(atom.predicate_id for rule in rules for atom in rule.body)
    outgoing: dict[PredicateId, set[PredicateId]] = {
        predicate_id: set() for predicate_id in nodes
    }
    indegree = {predicate_id: 0 for predicate_id in nodes}
    for rule in rules:
        for body_predicate in {atom.predicate_id for atom in rule.body}:
            head_predicate = rule.head.predicate_id
            if head_predicate not in outgoing[body_predicate]:
                outgoing[body_predicate].add(head_predicate)
                indegree[head_predicate] += 1

    ready = [predicate_id for predicate_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[PredicateId] = []
    while ready:
        predicate_id = heapq.heappop(ready)
        ordered.append(predicate_id)
        for dependent in sorted(outgoing[predicate_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(nodes):
        cyclic = tuple(sorted(node for node, degree in indegree.items() if degree > 0))
        raise ValueError(f"predicate dependencies must be acyclic: {cyclic}")
    return tuple(ordered)


def compute_closure(
    facts: tuple[GroundAtom, ...],
    rules: tuple[HornRule, ...],
    registry: PredicateRegistry,
    entities: tuple[Entity, ...],
    *,
    max_depth: int = 2,
) -> ClosureResult:
    """Compute finite typed Horn closure and select one canonical proof per atom."""
    if type(facts) is not tuple or not facts:
        raise ValueError("facts must be a nonempty tuple")
    if any(type(fact) is not GroundAtom for fact in facts):
        raise TypeError("facts must contain GroundAtom values")
    if type(rules) is not tuple or any(type(rule) is not HornRule for rule in rules):
        raise TypeError("rules must be a tuple of HornRule values")
    if type(registry) is not PredicateRegistry:
        raise TypeError("registry must be a PredicateRegistry")
    if type(max_depth) is not int or not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be an integer from zero to two")
    registry.validate_entities(entities)
    for fact in facts:
        registry.validate_ground_atom(fact, entities)
    for rule in rules:
        registry.validate_rule(rule, entities)
    if len({fact.atom_id for fact in facts}) != len(facts):
        raise ValueError("fact atom IDs must be unique")
    if len({fact.semantic_key for fact in facts}) != len(facts):
        raise ValueError("direct facts must be semantically unique")
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("rule IDs must be unique")

    predicate_order = predicate_topological_order(
        rules,
        tuple(predicate.predicate_id for predicate in registry.predicates),
    )
    predicate_rank = {
        predicate_id: rank for rank, predicate_id in enumerate(predicate_order)
    }
    ordered_rules = tuple(
        sorted(
            rules,
            key=lambda rule: (predicate_rank[rule.head.predicate_id], rule.rule_id),
        )
    )
    derivations: dict[SemanticAtomKey, _Derivation] = {
        fact.semantic_key: _Derivation(fact, None, (), 0)
        for fact in sorted(facts, key=lambda item: (item.semantic_key, item.atom_id))
    }
    semantic_key_by_id = {fact.atom_id: fact.semantic_key for fact in facts}

    changed = True
    while changed:
        changed = False
        for rule in ordered_rules:
            for binding, premises in _match_patterns(rule.body, derivations):
                depth = 1 + max(premise.depth for premise in premises)
                if depth > max_depth:
                    continue
                arguments = tuple(
                    argument if type(argument) is EntityId else binding[argument]
                    for argument in rule.head.arguments
                )
                semantic_key = (
                    str(rule.head.predicate_id),
                    tuple(str(argument) for argument in arguments),
                )
                existing = derivations.get(semantic_key)
                atom = (
                    existing.atom
                    if existing is not None
                    else GroundAtom(
                        atom_id=_derived_atom_id(semantic_key),
                        predicate_id=rule.head.predicate_id,
                        arguments=arguments,
                    )
                )
                conflicting_key = semantic_key_by_id.get(atom.atom_id)
                if conflicting_key is not None and conflicting_key != semantic_key:
                    raise RuntimeError("derived atom ID collision")
                registry.validate_ground_atom(atom, entities)
                candidate = _Derivation(atom, rule.rule_id, premises, depth)
                if existing is None or _derivation_rank(candidate) < _derivation_rank(
                    existing
                ):
                    derivations[semantic_key] = candidate
                    semantic_key_by_id[atom.atom_id] = semantic_key
                    changed = True

    ordered_derivations = tuple(
        derivations[key] for key in sorted(derivations)
    )
    return ClosureResult(
        atoms=tuple(derivation.atom for derivation in ordered_derivations),
        proofs=tuple(_proof_from_derivation(item) for item in ordered_derivations),
    )


def answer_query(
    closure: ClosureResult,
    query: QueryAst,
    registry: PredicateRegistry,
    entities: tuple[Entity, ...],
) -> tuple[EntityId, ...]:
    """Execute a typed conjunctive query and return sorted unique answers."""
    if type(closure) is not ClosureResult:
        raise TypeError("closure must be a ClosureResult")
    if type(registry) is not PredicateRegistry:
        raise TypeError("registry must be a PredicateRegistry")
    registry.answer_type(query, entities)
    derivations = {
        atom.semantic_key: _Derivation(atom, None, (), 0) for atom in closure.atoms
    }
    return tuple(
        sorted(
            {
                binding[query.answer_variable]
                for binding, _ in _match_patterns(query.clauses, derivations)
            }
        )
    )


def _match_patterns(
    patterns: tuple[AtomPattern, ...],
    derivations: dict[SemanticAtomKey, _Derivation],
) -> tuple[tuple[dict[Variable, EntityId], tuple[_Derivation, ...]], ...]:
    by_predicate: dict[PredicateId, tuple[_Derivation, ...]] = {}
    for predicate_id in {pattern.predicate_id for pattern in patterns}:
        by_predicate[predicate_id] = tuple(
            sorted(
                (
                    derivation
                    for derivation in derivations.values()
                    if derivation.atom.predicate_id == predicate_id
                ),
                key=lambda item: (item.atom.semantic_key, item.atom.atom_id),
            )
        )
    states: tuple[
        tuple[dict[Variable, EntityId], tuple[_Derivation, ...]], ...
    ] = (({}, ()),)
    for pattern in patterns:
        expanded: list[
            tuple[dict[Variable, EntityId], tuple[_Derivation, ...]]
        ] = []
        for binding, premises in states:
            for derivation in by_predicate.get(pattern.predicate_id, ()):
                unified = _unify(pattern, derivation.atom, binding)
                if unified is not None:
                    expanded.append((unified, premises + (derivation,)))
        states = tuple(
            sorted(
                expanded,
                key=lambda state: (
                    tuple(
                        sorted(
                            (variable.name, str(entity_id))
                            for variable, entity_id in state[0].items()
                        )
                    ),
                    tuple(premise.atom.semantic_key for premise in state[1]),
                ),
            )
        )
        if not states:
            break
    return states


def _unify(
    pattern: AtomPattern,
    atom: GroundAtom,
    binding: dict[Variable, EntityId],
) -> dict[Variable, EntityId] | None:
    if pattern.predicate_id != atom.predicate_id or len(pattern.arguments) != len(
        atom.arguments
    ):
        return None
    unified = dict(binding)
    for term, entity_id in zip(pattern.arguments, atom.arguments):
        if type(term) is EntityId:
            if term != entity_id:
                return None
        else:
            prior = unified.get(term)
            if prior is not None and prior != entity_id:
                return None
            unified[term] = entity_id
    return unified


def _derived_atom_id(semantic_key: SemanticAtomKey) -> AtomId:
    payload = json.dumps(
        semantic_key,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return AtomId(f"derived:{sha256(payload).hexdigest()}")


def _derivation_signature(derivation: _Derivation) -> str:
    value = (
        ["fact", str(derivation.atom.atom_id)]
        if derivation.rule_id is None
        else [
            "rule",
            str(derivation.rule_id),
            [_derivation_signature(premise) for premise in derivation.premises],
        ]
    )
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _derivation_rank(derivation: _Derivation) -> tuple[int, int, str]:
    return (
        derivation.depth,
        1 + sum(_derivation_rank(premise)[1] for premise in derivation.premises),
        _derivation_signature(derivation),
    )


def _proof_from_derivation(derivation: _Derivation) -> Proof:
    steps: list[ProofStep] = []
    seen: set[AtomId] = set()

    def visit(item: _Derivation) -> None:
        for premise in item.premises:
            visit(premise)
        if item.atom.atom_id in seen:
            return
        steps.append(
            ProofStep(
                atom=item.atom,
                rule_id=item.rule_id,
                premise_atom_ids=tuple(
                    premise.atom.atom_id for premise in item.premises
                ),
                depth=item.depth,
            )
        )
        seen.add(item.atom.atom_id)

    visit(derivation)
    signature = _derivation_signature(derivation).encode("utf-8")
    return Proof(
        proof_id=ProofId(f"proof:{sha256(signature).hexdigest()}"),
        conclusion_atom_id=derivation.atom.atom_id,
        steps=tuple(steps),
    )


__all__ = [
    "ClosureResult",
    "answer_query",
    "compute_closure",
    "predicate_topological_order",
]
