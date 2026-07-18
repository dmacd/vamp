"""Deterministic calibration and pilot knowledge-world generation."""

from __future__ import annotations

from dataclasses import dataclass

from apm.data.text.tinyworlds.closure import ClosureResult, compute_closure
from apm.data.text.tinyworlds.ontology import (
    PredicateKind,
    PredicateRegistry,
    PredicateSignature,
)
from apm.data.text.tinyworlds.schema import (
    AtomId,
    AtomPattern,
    Entity,
    EntityId,
    EntityTypeId,
    FamilyId,
    GroundAtom,
    HornRule,
    PredicateId,
    RuleId,
    TaskEdgeId,
    TaskId,
    TaskKind,
    TaskSpecification,
    Variable,
    validate_task_specifications,
)
from apm.data.text.tinyworlds.seeds import derive_subseed


TINYWORLDS_VERSION = "tinyworlds-v1"
_FILLER_PREDICATE_VARIANTS = 12
ACTOR_TYPE = EntityTypeId("actor")
PLACE_TYPE = EntityTypeId("place")
ATTRIBUTE_TYPE = EntityTypeId("attribute")
CONTEXT_TYPE = EntityTypeId("context")

CALIBRATION_TASK_IDS = (
    TaskId("calibration_seed"),
    TaskId("calibration_extension"),
    TaskId("calibration_revision"),
    TaskId("calibration_bridge"),
)
PILOT_TASK_IDS = (
    TaskId("willow_seed"),
    TaskId("sunny_seed"),
    TaskId("willow_extension"),
    TaskId("sunny_extension"),
    TaskId("willow_winter_revision"),
    TaskId("sunny_festival_revision"),
    TaskId("willow_bridge"),
    TaskId("sunny_bridge"),
)


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """One contextual value that is explicitly incompatible with its base value."""

    family_id: FamilyId
    base_atom_id: AtomId
    contextual_atom_id: AtomId
    subject_entity_id: EntityId
    base_value_entity_id: EntityId
    revised_value_entity_id: EntityId
    context_entity_id: EntityId

    def __post_init__(self) -> None:
        expected_types = (
            (self.family_id, FamilyId, "family_id"),
            (self.base_atom_id, AtomId, "base_atom_id"),
            (self.contextual_atom_id, AtomId, "contextual_atom_id"),
            (self.subject_entity_id, EntityId, "subject_entity_id"),
            (self.base_value_entity_id, EntityId, "base_value_entity_id"),
            (self.revised_value_entity_id, EntityId, "revised_value_entity_id"),
            (self.context_entity_id, EntityId, "context_entity_id"),
        )
        if any(type(value) is not expected for value, expected, _ in expected_types):
            invalid = next(
                label
                for value, expected, label in expected_types
                if type(value) is not expected
            )
            raise TypeError(f"{invalid} has the wrong typed ID")
        if self.base_value_entity_id == self.revised_value_entity_id:
            raise ValueError("a contextual revision must change the base value")


@dataclass(frozen=True, slots=True)
class SymbolicWorld:
    """One complete pre-rendering world with verified depth-two closure."""

    world_id: str
    master_seed_sha256: str
    registry: PredicateRegistry
    entities: tuple[Entity, ...]
    facts: tuple[GroundAtom, ...]
    rules: tuple[HornRule, ...]
    tasks: tuple[TaskSpecification, ...]
    closure: ClosureResult
    revisions: tuple[RevisionRecord, ...]

    def __post_init__(self) -> None:
        if self.world_id not in ("calibration", "pilot"):
            raise ValueError("world_id must identify the calibration or pilot preset")
        derive_subseed(self.master_seed_sha256, "world-validation", self.world_id)
        if type(self.registry) is not PredicateRegistry:
            raise TypeError("registry must be a PredicateRegistry")
        typed_tuples = (
            (self.entities, Entity, "entities"),
            (self.facts, GroundAtom, "facts"),
            (self.rules, HornRule, "rules"),
            (self.tasks, TaskSpecification, "tasks"),
            (self.revisions, RevisionRecord, "revisions"),
        )
        for values, expected_type, label in typed_tuples:
            if type(values) is not tuple or any(
                type(value) is not expected_type for value in values
            ):
                raise TypeError(f"{label} must be a tuple of {expected_type.__name__}")
        if type(self.closure) is not ClosureResult:
            raise TypeError("closure must be a ClosureResult")
        validate_task_specifications(self.tasks)
        _validate_fixed_world_topology(self)
        self.registry.validate_entities(self.entities)
        if compute_closure(
            self.facts,
            self.rules,
            self.registry,
            self.entities,
        ) != self.closure:
            raise ValueError("world closure does not match its facts and rules")
        _validate_world_references(self)
        _validate_world_quantities(self)
        _validate_revision_records(self)
        head_predicates = {rule.head.predicate_id for rule in self.rules}
        if any(fact.predicate_id in head_predicates for fact in self.facts):
            raise ValueError(
                "inferred rule conclusions cannot be direct world statements"
            )

    def task(self, task_id: TaskId) -> TaskSpecification:
        """Return one task specification by ID."""
        if type(task_id) is not TaskId:
            raise TypeError("task_id must be a TaskId")
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"unknown task ID: {task_id}")

    def task_path(self, task_id: TaskId) -> tuple[TaskId, ...]:
        """Return the family seed-to-task path in canonical task order."""
        reverse_path: list[TaskId] = []
        task = self.task(task_id)
        while True:
            reverse_path.append(task.task_id)
            if task.parent_task_id is None:
                return tuple(reversed(reverse_path))
            task = self.task(task.parent_task_id)


@dataclass(frozen=True, slots=True)
class _TaskLayout:
    task_id: TaskId
    family_id: FamilyId
    kind: TaskKind
    parent_task_id: TaskId | None


@dataclass(frozen=True, slots=True)
class _FamilyLayout:
    family_id: FamilyId
    seed_task_id: TaskId
    extension_task_id: TaskId
    revision_task_id: TaskId
    bridge_task_id: TaskId


def generate_calibration_world(
    master_seed_sha256: str,
    *,
    direct_facts_per_task: int = 24,
) -> SymbolicWorld:
    """Generate the disjoint calibration family with a fixed fact capacity."""
    if direct_facts_per_task not in (24, 36):
        raise ValueError("calibration direct_facts_per_task must be 24 or 36")
    family = FamilyId("calibration")
    layouts = (
        _TaskLayout(CALIBRATION_TASK_IDS[0], family, TaskKind.SEED, None),
        _TaskLayout(
            CALIBRATION_TASK_IDS[1],
            family,
            TaskKind.EXTENSION,
            CALIBRATION_TASK_IDS[0],
        ),
        _TaskLayout(
            CALIBRATION_TASK_IDS[2],
            family,
            TaskKind.REVISION,
            CALIBRATION_TASK_IDS[0],
        ),
        _TaskLayout(
            CALIBRATION_TASK_IDS[3],
            family,
            TaskKind.BRIDGE,
            CALIBRATION_TASK_IDS[2],
        ),
    )
    return _generate_world(
        "calibration",
        master_seed_sha256,
        layouts,
        direct_facts_per_task=direct_facts_per_task,
    )


def generate_pilot_world(
    master_seed_sha256: str,
    *,
    direct_facts_per_task: int = 24,
) -> SymbolicWorld:
    """Generate the interleaved pilot at the calibrated fact capacity."""
    if direct_facts_per_task not in (24, 36):
        raise ValueError("pilot direct_facts_per_task must be 24 or 36")
    willow = FamilyId("willow")
    sunny = FamilyId("sunny")
    layouts = (
        _TaskLayout(PILOT_TASK_IDS[0], willow, TaskKind.SEED, None),
        _TaskLayout(PILOT_TASK_IDS[1], sunny, TaskKind.SEED, None),
        _TaskLayout(
            PILOT_TASK_IDS[2], willow, TaskKind.EXTENSION, PILOT_TASK_IDS[0]
        ),
        _TaskLayout(
            PILOT_TASK_IDS[3], sunny, TaskKind.EXTENSION, PILOT_TASK_IDS[1]
        ),
        _TaskLayout(
            PILOT_TASK_IDS[4], willow, TaskKind.REVISION, PILOT_TASK_IDS[0]
        ),
        _TaskLayout(
            PILOT_TASK_IDS[5], sunny, TaskKind.REVISION, PILOT_TASK_IDS[1]
        ),
        _TaskLayout(
            PILOT_TASK_IDS[6], willow, TaskKind.BRIDGE, PILOT_TASK_IDS[4]
        ),
        _TaskLayout(
            PILOT_TASK_IDS[7], sunny, TaskKind.BRIDGE, PILOT_TASK_IDS[5]
        ),
    )
    return _generate_world(
        "pilot",
        master_seed_sha256,
        layouts,
        direct_facts_per_task=direct_facts_per_task,
    )


def _generate_world(
    world_id: str,
    master_seed_sha256: str,
    layouts: tuple[_TaskLayout, ...],
    *,
    direct_facts_per_task: int,
) -> SymbolicWorld:
    derive_subseed(master_seed_sha256, "world", world_id)
    entities_by_task = {
        layout.task_id: _generate_entities(
            master_seed_sha256,
            world_id,
            layout,
        )
        for layout in layouts
    }
    family_layouts = _family_layouts(layouts)
    signatures = _predicate_signatures(layouts, family_layouts)
    rules_by_task = _rules_by_task(family_layouts)
    facts_by_task, revisions = _facts_by_task(
        master_seed_sha256,
        world_id,
        layouts,
        family_layouts,
        entities_by_task,
        signatures,
        direct_facts_per_task,
    )
    tasks = tuple(
        TaskSpecification(
            task_id=layout.task_id,
            family_id=layout.family_id,
            kind=layout.kind,
            parent_task_id=layout.parent_task_id,
            direct_fact_ids=tuple(
                fact.atom_id for fact in facts_by_task[layout.task_id]
            ),
            rule_ids=tuple(
                rule.rule_id for rule in rules_by_task.get(layout.task_id, ())
            ),
            introduced_entity_ids=tuple(
                entity.entity_id for entity in entities_by_task[layout.task_id]
            ),
            incoming_edge_id=TaskEdgeId(f"edge:{layout.task_id}"),
        )
        for layout in layouts
    )
    entities = tuple(
        entity for layout in layouts for entity in entities_by_task[layout.task_id]
    )
    facts = tuple(
        fact for layout in layouts for fact in facts_by_task[layout.task_id]
    )
    rules = tuple(
        rule
        for layout in layouts
        for rule in rules_by_task.get(layout.task_id, ())
    )
    registry = PredicateRegistry(
        entity_types=(ACTOR_TYPE, PLACE_TYPE, ATTRIBUTE_TYPE, CONTEXT_TYPE),
        context_type_id=CONTEXT_TYPE,
        predicates=tuple(signatures.values()),
    )
    return SymbolicWorld(
        world_id=world_id,
        master_seed_sha256=master_seed_sha256,
        registry=registry,
        entities=entities,
        facts=facts,
        rules=rules,
        tasks=tasks,
        closure=compute_closure(facts, rules, registry, entities),
        revisions=revisions,
    )


def _family_layouts(layouts: tuple[_TaskLayout, ...]) -> tuple[_FamilyLayout, ...]:
    families = tuple(dict.fromkeys(layout.family_id for layout in layouts))
    return tuple(
        _FamilyLayout(
            family_id=family_id,
            seed_task_id=next(
                item.task_id
                for item in layouts
                if item.family_id == family_id and item.kind is TaskKind.SEED
            ),
            extension_task_id=next(
                item.task_id
                for item in layouts
                if item.family_id == family_id and item.kind is TaskKind.EXTENSION
            ),
            revision_task_id=next(
                item.task_id
                for item in layouts
                if item.family_id == family_id and item.kind is TaskKind.REVISION
            ),
            bridge_task_id=next(
                item.task_id
                for item in layouts
                if item.family_id == family_id and item.kind is TaskKind.BRIDGE
            ),
        )
        for family_id in families
    )


def _generate_entities(
    master_seed_sha256: str,
    world_id: str,
    layout: _TaskLayout,
) -> tuple[Entity, ...]:
    type_layouts = {
        TaskKind.SEED: (
            ACTOR_TYPE,
            ACTOR_TYPE,
            ACTOR_TYPE,
            PLACE_TYPE,
            PLACE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            CONTEXT_TYPE,
        ),
        TaskKind.EXTENSION: (
            ACTOR_TYPE,
            ACTOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
        ),
        TaskKind.REVISION: (
            ACTOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
            CONTEXT_TYPE,
        ),
        TaskKind.BRIDGE: (
            ACTOR_TYPE,
            PLACE_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_TYPE,
        ),
    }
    return tuple(
        _entity(
            master_seed_sha256,
            world_id,
            layout.task_id,
            index,
            entity_type,
        )
        for index, entity_type in enumerate(type_layouts[layout.kind])
    )


def _entity(
    master_seed_sha256: str,
    world_id: str,
    task_id: TaskId,
    index: int,
    entity_type: EntityTypeId,
) -> Entity:
    digest = derive_subseed(
        master_seed_sha256,
        "entity",
        world_id,
        str(task_id),
        index,
        str(entity_type),
    )
    name = f"N{digest[:12]}"
    return Entity(
        entity_id=EntityId(f"entity:{task_id}:{index:02d}"),
        entity_type=entity_type,
        name=name,
        inflections=(f"{name}s",),
    )


def _predicate_signatures(
    layouts: tuple[_TaskLayout, ...],
    families: tuple[_FamilyLayout, ...],
) -> dict[PredicateId, PredicateSignature]:
    signatures: dict[PredicateId, PredicateSignature] = {}

    def add(
        predicate_id: PredicateId,
        argument_types: tuple[EntityTypeId, ...],
        kind: PredicateKind = PredicateKind.ORDINARY,
    ) -> None:
        signatures[predicate_id] = PredicateSignature(
            predicate_id,
            argument_types,
            kind,
        )

    for family in families:
        ids = family_predicates(family.family_id)
        for name in ("seed_anchor", "extension_key"):
            add(
                ids[name],
                (ACTOR_TYPE,) if name == "seed_anchor" else (ATTRIBUTE_TYPE,),
            )
        for name in (
            "base_selects",
            "seed_color",
            "open_book_selects",
            "extension_assignment",
            "extension_role",
            "bridge_link",
            "bridge_result",
            "ancestor_child_result",
            "extension_known_role",
            "extension_notable_role",
        ):
            add(ids[name], (ACTOR_TYPE, ATTRIBUTE_TYPE))
        for name in ("seed_transform", "extension_transform"):
            add(ids[name], (ATTRIBUTE_TYPE, ATTRIBUTE_TYPE))
        for name in ("seed_vivid", "seed_notable"):
            add(ids[name], (ACTOR_TYPE, ATTRIBUTE_TYPE))
        for name in ("extension_friend", "extension_trusted"):
            add(ids[name], (ACTOR_TYPE, ACTOR_TYPE))
        for name in ("contextual_selects", "revision_result"):
            add(
                ids[name],
                (ACTOR_TYPE, ATTRIBUTE_TYPE, CONTEXT_TYPE),
                PredicateKind.CONTEXTUAL,
            )
    for layout in layouts:
        for suffix in range(_FILLER_PREDICATE_VARIANTS):
            add(
                _filler_predicate(layout.task_id, "actor_attribute", suffix),
                (ACTOR_TYPE, ATTRIBUTE_TYPE),
            )
            add(
                _filler_predicate(layout.task_id, "attribute_pair", suffix),
                (ATTRIBUTE_TYPE, ATTRIBUTE_TYPE),
            )
        add(
            _filler_predicate(layout.task_id, "actor", 0),
            (ACTOR_TYPE,),
        )
        add(
            _filler_predicate(layout.task_id, "attribute", 0),
            (ATTRIBUTE_TYPE,),
        )
    return signatures


def _rules_by_task(
    families: tuple[_FamilyLayout, ...],
) -> dict[TaskId, tuple[HornRule, ...]]:
    result: dict[TaskId, tuple[HornRule, ...]] = {}
    for family in families:
        predicates = family_predicates(family.family_id)
        actor = Variable("actor")
        source = Variable("source")
        target = Variable("target")
        context = Variable("context")
        base = Variable("base")
        friend = Variable("friend")
        seed_rules = (
            HornRule(
                RuleId(f"rule:{family.seed_task_id}:vivid"),
                AtomPattern(predicates["seed_vivid"], (actor, target)),
                (
                    AtomPattern(predicates["seed_color"], (actor, source)),
                    AtomPattern(predicates["seed_transform"], (source, target)),
                ),
            ),
            HornRule(
                RuleId(f"rule:{family.seed_task_id}:notable"),
                AtomPattern(predicates["seed_notable"], (actor, target)),
                (AtomPattern(predicates["seed_vivid"], (actor, target)),),
            ),
            HornRule(
                RuleId(f"rule:{family.seed_task_id}:bridge"),
                AtomPattern(predicates["bridge_result"], (actor, target)),
                (
                    AtomPattern(predicates["seed_anchor"], (actor,)),
                    AtomPattern(predicates["extension_key"], (target,)),
                    AtomPattern(
                        predicates["contextual_selects"],
                        (actor, target, context),
                    ),
                    AtomPattern(predicates["bridge_link"], (actor, target)),
                ),
            ),
            HornRule(
                RuleId(f"rule:{family.seed_task_id}:revision"),
                AtomPattern(
                    predicates["revision_result"],
                    (actor, target, context),
                ),
                (
                    AtomPattern(predicates["base_selects"], (actor, base)),
                    AtomPattern(
                        predicates["contextual_selects"],
                        (actor, target, context),
                    ),
                ),
            ),
        )
        extension_rules = (
            HornRule(
                RuleId(f"rule:{family.extension_task_id}:ancestor-child"),
                AtomPattern(
                    predicates["ancestor_child_result"],
                    (actor, target),
                ),
                (
                    AtomPattern(predicates["seed_anchor"], (actor,)),
                    AtomPattern(
                        predicates["extension_assignment"],
                        (actor, target),
                    ),
                ),
            ),
            HornRule(
                RuleId(f"rule:{family.extension_task_id}:known-role"),
                AtomPattern(
                    predicates["extension_known_role"],
                    (actor, target),
                ),
                (
                    AtomPattern(
                        predicates["extension_role"],
                        (actor, source),
                    ),
                    AtomPattern(
                        predicates["extension_transform"],
                        (source, target),
                    ),
                ),
            ),
            HornRule(
                RuleId(f"rule:{family.extension_task_id}:notable-role"),
                AtomPattern(
                    predicates["extension_notable_role"],
                    (actor, target),
                ),
                (
                    AtomPattern(
                        predicates["extension_known_role"],
                        (actor, target),
                    ),
                ),
            ),
            HornRule(
                RuleId(f"rule:{family.extension_task_id}:trusted"),
                AtomPattern(
                    predicates["extension_trusted"],
                    (actor, friend),
                ),
                (
                    AtomPattern(
                        predicates["extension_friend"],
                        (actor, friend),
                    ),
                ),
            ),
        )
        result[family.seed_task_id] = seed_rules
        result[family.extension_task_id] = extension_rules
    return result


def _facts_by_task(
    master_seed_sha256: str,
    world_id: str,
    layouts: tuple[_TaskLayout, ...],
    families: tuple[_FamilyLayout, ...],
    entities_by_task: dict[TaskId, tuple[Entity, ...]],
    signatures: dict[PredicateId, PredicateSignature],
    direct_facts_per_task: int,
) -> tuple[dict[TaskId, tuple[GroundAtom, ...]], tuple[RevisionRecord, ...]]:
    required: dict[TaskId, list[GroundAtom]] = {
        layout.task_id: [] for layout in layouts
    }
    revisions: list[RevisionRecord] = []
    for family in families:
        predicates = family_predicates(family.family_id)
        seed_actors = _entity_ids(
            entities_by_task[family.seed_task_id], ACTOR_TYPE
        )
        seed_attributes = _entity_ids(
            entities_by_task[family.seed_task_id], ATTRIBUTE_TYPE
        )
        extension_actors = _entity_ids(
            entities_by_task[family.extension_task_id], ACTOR_TYPE
        )
        extension_attributes = _entity_ids(
            entities_by_task[family.extension_task_id], ATTRIBUTE_TYPE
        )
        revision_context = _entity_ids(
            entities_by_task[family.revision_task_id], CONTEXT_TYPE
        )[0]

        seed_specs = (
            *((predicates["seed_anchor"], (actor,)) for actor in seed_actors),
            *(
                (
                    predicates["base_selects"],
                    (seed_actors[index], seed_attributes[index]),
                )
                for index in range(3)
            ),
            *(
                (
                    predicates["seed_color"],
                    (seed_actors[index], seed_attributes[index]),
                )
                for index in range(3)
            ),
            *(
                (
                    predicates["seed_transform"],
                    (seed_attributes[index], seed_attributes[index + 1]),
                )
                for index in range(3)
            ),
            (
                predicates["open_book_selects"],
                (seed_actors[0], seed_attributes[3]),
            ),
            (
                predicates["open_book_selects"],
                (seed_actors[1], seed_attributes[0]),
            ),
        )
        required[family.seed_task_id].extend(
            _make_fact(
                master_seed_sha256,
                world_id,
                family.seed_task_id,
                predicate_id,
                arguments,
            )
            for predicate_id, arguments in seed_specs
        )

        extension_specs = (
            *(
                (predicates["extension_key"], (attribute,))
                for attribute in extension_attributes
            ),
            *(
                (
                    predicates["extension_assignment"],
                    (seed_actors[index], extension_attributes[index]),
                )
                for index in range(3)
            ),
            (
                predicates["extension_role"],
                (extension_actors[0], extension_attributes[0]),
            ),
            (
                predicates["extension_role"],
                (extension_actors[1], extension_attributes[1]),
            ),
            (
                predicates["extension_transform"],
                (extension_attributes[0], extension_attributes[1]),
            ),
            (
                predicates["extension_transform"],
                (extension_attributes[1], extension_attributes[2]),
            ),
            (
                predicates["extension_friend"],
                (extension_actors[0], seed_actors[0]),
            ),
        )
        required[family.extension_task_id].extend(
            _make_fact(
                master_seed_sha256,
                world_id,
                family.extension_task_id,
                predicate_id,
                arguments,
            )
            for predicate_id, arguments in extension_specs
        )

        contextual_facts = tuple(
            _make_fact(
                master_seed_sha256,
                world_id,
                family.revision_task_id,
                predicates["contextual_selects"],
                (seed_actors[index], extension_attributes[index], revision_context),
            )
            for index in range(3)
        )
        required[family.revision_task_id].extend(contextual_facts)
        bridge_facts = tuple(
            _make_fact(
                master_seed_sha256,
                world_id,
                family.bridge_task_id,
                predicates["bridge_link"],
                (seed_actors[index], extension_attributes[index]),
            )
            for index in range(3)
        )
        required[family.bridge_task_id].extend(bridge_facts)
        base_by_subject = {
            fact.arguments[0]: fact
            for fact in required[family.seed_task_id]
            if fact.predicate_id == predicates["base_selects"]
        }
        revisions.extend(
            RevisionRecord(
                family_id=family.family_id,
                base_atom_id=base_by_subject[fact.arguments[0]].atom_id,
                contextual_atom_id=fact.atom_id,
                subject_entity_id=fact.arguments[0],
                base_value_entity_id=base_by_subject[fact.arguments[0]].arguments[1],
                revised_value_entity_id=fact.arguments[1],
                context_entity_id=fact.arguments[2],
            )
            for fact in contextual_facts
        )

    result = {
        layout.task_id: _fill_task_facts(
            master_seed_sha256,
            world_id,
            layout,
            entities_by_task[layout.task_id],
            tuple(required[layout.task_id]),
            signatures,
            direct_facts_per_task,
        )
        for layout in layouts
    }
    return result, tuple(revisions)


def _fill_task_facts(
    master_seed_sha256: str,
    world_id: str,
    layout: _TaskLayout,
    entities: tuple[Entity, ...],
    required: tuple[GroundAtom, ...],
    signatures: dict[PredicateId, PredicateSignature],
    direct_facts_per_task: int,
) -> tuple[GroundAtom, ...]:
    actors = _entity_ids(entities, ACTOR_TYPE)
    attributes = _entity_ids(entities, ATTRIBUTE_TYPE)
    specs: list[tuple[PredicateId, tuple[EntityId, ...]]] = []
    for suffix in range(_FILLER_PREDICATE_VARIANTS):
        specs.extend(
            (
                _filler_predicate(layout.task_id, "actor_attribute", suffix),
                (actor, attribute),
            )
            for actor in actors
            for attribute in attributes
        )
        specs.extend(
            (
                _filler_predicate(layout.task_id, "attribute_pair", suffix),
                (left, right),
            )
            for left in attributes
            for right in attributes
        )
    specs.extend(
        (_filler_predicate(layout.task_id, "actor", 0), (actor,))
        for actor in actors
    )
    specs.extend(
        (_filler_predicate(layout.task_id, "attribute", 0), (attribute,))
        for attribute in attributes
    )
    candidates = tuple(
        sorted(
            (
                (
                    derive_subseed(
                        master_seed_sha256,
                        "filler-fact-order",
                        world_id,
                        str(layout.task_id),
                        str(predicate_id),
                        *(str(argument) for argument in arguments),
                    ),
                    predicate_id,
                    arguments,
                )
                for predicate_id, arguments in specs
                if predicate_id in signatures
            ),
            key=lambda item: item[0],
        )
    )
    needed = direct_facts_per_task - len(required)
    if needed < 0 or len(candidates) < needed:
        raise RuntimeError(f"insufficient fact capacity for task {layout.task_id}")
    fillers = tuple(
        _make_fact(
            master_seed_sha256,
            world_id,
            layout.task_id,
            predicate_id,
            arguments,
        )
        for _, predicate_id, arguments in candidates[:needed]
    )
    facts = required + fillers
    if len({fact.semantic_key for fact in facts}) != direct_facts_per_task:
        raise RuntimeError(f"task {layout.task_id} generated duplicate facts")
    return tuple(sorted(facts, key=lambda fact: fact.atom_id))


def _make_fact(
    master_seed_sha256: str,
    world_id: str,
    task_id: TaskId,
    predicate_id: PredicateId,
    arguments: tuple[EntityId, ...],
) -> GroundAtom:
    digest = derive_subseed(
        master_seed_sha256,
        "fact",
        world_id,
        str(task_id),
        str(predicate_id),
        *(str(argument) for argument in arguments),
    )
    return GroundAtom(
        atom_id=AtomId(f"fact:{task_id}:{digest[:24]}"),
        predicate_id=predicate_id,
        arguments=arguments,
    )


def family_predicates(family_id: FamilyId) -> dict[str, PredicateId]:
    """Return the stable predicate IDs reserved for one world family."""
    if type(family_id) is not FamilyId:
        raise TypeError("family_id must be a FamilyId")
    names = (
        "seed_anchor",
        "base_selects",
        "seed_color",
        "open_book_selects",
        "seed_transform",
        "seed_vivid",
        "seed_notable",
        "extension_key",
        "extension_assignment",
        "extension_role",
        "extension_transform",
        "extension_known_role",
        "extension_notable_role",
        "extension_friend",
        "extension_trusted",
        "contextual_selects",
        "revision_result",
        "bridge_link",
        "bridge_result",
        "ancestor_child_result",
    )
    return {
        name: PredicateId(f"predicate:{family_id}:{name}") for name in names
    }


def _filler_predicate(
    task_id: TaskId,
    category: str,
    suffix: int,
) -> PredicateId:
    return PredicateId(f"predicate:{task_id}:filler_{category}_{suffix}")


def _entity_ids(
    entities: tuple[Entity, ...],
    entity_type: EntityTypeId,
) -> tuple[EntityId, ...]:
    return tuple(
        entity.entity_id for entity in entities if entity.entity_type == entity_type
    )


def _validate_world_references(world: SymbolicWorld) -> None:
    entity_ids = {entity.entity_id for entity in world.entities}
    fact_ids = {fact.atom_id for fact in world.facts}
    rule_ids = {rule.rule_id for rule in world.rules}
    if len(entity_ids) != len(world.entities):
        raise ValueError("world entity IDs must be unique")
    if len(fact_ids) != len(world.facts):
        raise ValueError("world fact IDs must be unique")
    if len(rule_ids) != len(world.rules):
        raise ValueError("world rule IDs must be unique")
    if {item for task in world.tasks for item in task.introduced_entity_ids} != entity_ids:
        raise ValueError("task entity ownership must cover the world exactly")
    if {item for task in world.tasks for item in task.direct_fact_ids} != fact_ids:
        raise ValueError("task fact ownership must cover the world exactly")
    if {item for task in world.tasks for item in task.rule_ids} != rule_ids:
        raise ValueError("task rule ownership must cover the world exactly")
    if any(task.incoming_edge_id is None for task in world.tasks):
        raise ValueError("generated world tasks must each have an incoming edge")
    fact_by_id = {fact.atom_id: fact for fact in world.facts}
    for bridge_task in (
        task for task in world.tasks if task.kind is TaskKind.BRIDGE
    ):
        predicate_id = family_predicates(bridge_task.family_id)["bridge_link"]
        family_bridge_fact_ids = {
            fact.atom_id
            for fact in world.facts
            if fact.predicate_id == predicate_id
        }
        owned_bridge_fact_ids = {
            fact_id
            for fact_id in bridge_task.direct_fact_ids
            if fact_by_id[fact_id].predicate_id == predicate_id
        }
        if (
            len(family_bridge_fact_ids) != 3
            or owned_bridge_fact_ids != family_bridge_fact_ids
        ):
            raise ValueError(
                "each bridge task must own exactly three family bridge-link facts"
            )


def _validate_fixed_world_topology(world: SymbolicWorld) -> None:
    if world.world_id == "calibration":
        expected = (
            (
                CALIBRATION_TASK_IDS[0],
                FamilyId("calibration"),
                TaskKind.SEED,
                None,
            ),
            (
                CALIBRATION_TASK_IDS[1],
                FamilyId("calibration"),
                TaskKind.EXTENSION,
                CALIBRATION_TASK_IDS[0],
            ),
            (
                CALIBRATION_TASK_IDS[2],
                FamilyId("calibration"),
                TaskKind.REVISION,
                CALIBRATION_TASK_IDS[0],
            ),
            (
                CALIBRATION_TASK_IDS[3],
                FamilyId("calibration"),
                TaskKind.BRIDGE,
                CALIBRATION_TASK_IDS[2],
            ),
        )
    else:
        willow = FamilyId("willow")
        sunny = FamilyId("sunny")
        expected = (
            (PILOT_TASK_IDS[0], willow, TaskKind.SEED, None),
            (PILOT_TASK_IDS[1], sunny, TaskKind.SEED, None),
            (
                PILOT_TASK_IDS[2],
                willow,
                TaskKind.EXTENSION,
                PILOT_TASK_IDS[0],
            ),
            (
                PILOT_TASK_IDS[3],
                sunny,
                TaskKind.EXTENSION,
                PILOT_TASK_IDS[1],
            ),
            (
                PILOT_TASK_IDS[4],
                willow,
                TaskKind.REVISION,
                PILOT_TASK_IDS[0],
            ),
            (
                PILOT_TASK_IDS[5],
                sunny,
                TaskKind.REVISION,
                PILOT_TASK_IDS[1],
            ),
            (
                PILOT_TASK_IDS[6],
                willow,
                TaskKind.BRIDGE,
                PILOT_TASK_IDS[4],
            ),
            (
                PILOT_TASK_IDS[7],
                sunny,
                TaskKind.BRIDGE,
                PILOT_TASK_IDS[5],
            ),
        )
    actual = tuple(
        (
            task.task_id,
            task.family_id,
            task.kind,
            task.parent_task_id,
        )
        for task in world.tasks
    )
    if actual != expected or any(
        task.incoming_edge_id != TaskEdgeId(f"edge:{task.task_id}")
        for task in world.tasks
    ):
        raise ValueError(
            f"{world.world_id} tasks must match the exact fixed v1 topology"
        )


def _validate_world_quantities(world: SymbolicWorld) -> None:
    expected_entities = {
        TaskKind.SEED: 10,
        TaskKind.EXTENSION: 5,
        TaskKind.REVISION: 5,
        TaskKind.BRIDGE: 4,
    }
    expected_rules = {
        TaskKind.SEED: 4,
        TaskKind.EXTENSION: 4,
        TaskKind.REVISION: 0,
        TaskKind.BRIDGE: 0,
    }
    fact_counts = {len(task.direct_fact_ids) for task in world.tasks}
    if len(fact_counts) != 1 or not fact_counts.issubset({24, 36}):
        raise ValueError(
            "all tasks must share one direct-fact count from the fixed capacities"
        )
    for task in world.tasks:
        if len(task.introduced_entity_ids) != expected_entities[task.kind]:
            raise ValueError(f"task {task.task_id} has the wrong entity count")
        if len(task.rule_ids) != expected_rules[task.kind]:
            raise ValueError(f"task {task.task_id} has the wrong rule count")


def _validate_revision_records(world: SymbolicWorld) -> None:
    fact_by_id = {fact.atom_id: fact for fact in world.facts}
    task_by_fact = {
        fact_id: task
        for task in world.tasks
        for fact_id in task.direct_fact_ids
    }
    family_ids = {task.family_id for task in world.tasks}
    unknown_families = tuple(
        sorted(
            {
                record.family_id
                for record in world.revisions
                if record.family_id not in family_ids
            }
        )
    )
    if unknown_families:
        raise ValueError(
            f"revision records reference unknown families: {unknown_families}"
        )
    for family_id in family_ids:
        family_records = tuple(
            record for record in world.revisions if record.family_id == family_id
        )
        if len(family_records) != 3:
            raise ValueError(f"family {family_id} must contain three revisions")
        distinct_fields = (
            ("records", family_records),
            ("subjects", tuple(record.subject_entity_id for record in family_records)),
            ("base facts", tuple(record.base_atom_id for record in family_records)),
            (
                "contextual facts",
                tuple(record.contextual_atom_id for record in family_records),
            ),
        )
        for label, values in distinct_fields:
            if len(set(values)) != 3:
                raise ValueError(
                    f"family {family_id} must contain three distinct revision {label}"
                )
    for record in world.revisions:
        base = fact_by_id.get(record.base_atom_id)
        contextual = fact_by_id.get(record.contextual_atom_id)
        if base is None or contextual is None:
            raise ValueError("revision records must reference direct facts")
        if base.arguments != (
            record.subject_entity_id,
            record.base_value_entity_id,
        ) or contextual.arguments != (
            record.subject_entity_id,
            record.revised_value_entity_id,
            record.context_entity_id,
        ):
            raise ValueError("revision record values do not match referenced atoms")
        base_owner = task_by_fact[base.atom_id]
        contextual_owner = task_by_fact[contextual.atom_id]
        if (
            base_owner.family_id != record.family_id
            or contextual_owner.family_id != record.family_id
        ):
            raise ValueError(
                "revision fact-owner task families must match the record family"
            )
        if base_owner.kind is not TaskKind.SEED:
            raise ValueError("revision base facts must belong to the seed task")
        if contextual_owner.kind is not TaskKind.REVISION:
            raise ValueError("contextual facts must belong to the revision task")
        predicates = family_predicates(record.family_id)
        if base.predicate_id != predicates["base_selects"]:
            raise ValueError("revision base facts must use the family base predicate")
        if contextual.predicate_id != predicates["contextual_selects"]:
            raise ValueError(
                "revision contextual facts must use the family contextual predicate"
            )
        signature = world.registry.signature(contextual.predicate_id)
        if signature.kind is not PredicateKind.CONTEXTUAL:
            raise ValueError("revision facts must use contextual predicates")


__all__ = [
    "ACTOR_TYPE",
    "ATTRIBUTE_TYPE",
    "CALIBRATION_TASK_IDS",
    "CONTEXT_TYPE",
    "PILOT_TASK_IDS",
    "PLACE_TYPE",
    "RevisionRecord",
    "SymbolicWorld",
    "TINYWORLDS_VERSION",
    "generate_calibration_world",
    "generate_pilot_world",
    "family_predicates",
]
