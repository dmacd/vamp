"""Deterministic TinyWorlds story/query plans and exact candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace

from apm.data.text.tinyworlds.closure import ClosureResult, answer_query
from apm.data.text.tinyworlds.schema import (
    AtomId,
    AtomPattern,
    CandidateRole,
    DataSplit,
    Entity,
    EntityId,
    EntityTypeId,
    FamilyId,
    GroundAtom,
    HornRule,
    HoldoutMetadata,
    PredicateId,
    Proof,
    ProofMetadata,
    ProofStep,
    QueryAst,
    QueryCandidate,
    QueryId,
    QueryKind,
    QueryPlan,
    StoryId,
    StoryPlan,
    TaskId,
    TaskKind,
    TaskSpecification,
    Variable,
)
from apm.data.text.tinyworlds.seeds import derive_subseed
from apm.data.text.tinyworlds.world_generation import (
    ACTOR_TYPE,
    ATTRIBUTE_TYPE,
    CONTEXT_TYPE,
    SymbolicWorld,
    TINYWORLDS_VERSION,
    family_predicates,
    generate_calibration_world,
    generate_pilot_world,
)


REQUIRED_QUERY_KINDS = (
    QueryKind.DIRECT,
    QueryKind.ANCESTOR_PLUS_CHILD,
    QueryKind.NEW_INSTANCE,
    QueryKind.ONE_HOP,
    QueryKind.TWO_HOP,
    QueryKind.REVISION_SENSITIVE,
    QueryKind.CROSS_BRANCH,
    QueryKind.OPEN_BOOK,
)
_CANDIDATE_PRIORITY = {
    CandidateRole.INCOMPATIBLE_REVISION: 0,
    CandidateRole.COMPETING_TASK: 1,
    CandidateRole.PARTIAL_PROOF: 2,
    CandidateRole.SAME_TYPE_FILLER: 3,
}
STANDARD_DISTRACTOR_ROLE_CYCLE = (
    CandidateRole.INCOMPATIBLE_REVISION,
    CandidateRole.COMPETING_TASK,
    CandidateRole.PARTIAL_PROOF,
    CandidateRole.SAME_TYPE_FILLER,
)


@dataclass(frozen=True, slots=True)
class TinyWorldsBundle:
    """Immutable verified world plus pre-rendering story and query plans."""

    bundle_id: str
    version: str
    world: SymbolicWorld
    story_plans: tuple[StoryPlan, ...]
    query_plans: tuple[QueryPlan, ...]

    def __post_init__(self) -> None:
        if type(self.bundle_id) is not str or not self.bundle_id:
            raise ValueError("bundle_id must be a nonempty string")
        if self.version != TINYWORLDS_VERSION:
            raise ValueError(f"bundle version must equal {TINYWORLDS_VERSION}")
        if type(self.world) is not SymbolicWorld:
            raise TypeError("world must be a SymbolicWorld")
        expected_bundle_id = f"{self.version}:{self.world.world_id}"
        if self.bundle_id != expected_bundle_id:
            raise ValueError(f"bundle_id must equal {expected_bundle_id}")
        for values, expected_type, label in (
            (self.story_plans, StoryPlan, "story_plans"),
            (self.query_plans, QueryPlan, "query_plans"),
        ):
            if type(values) is not tuple or not values or any(
                type(value) is not expected_type for value in values
            ):
                raise TypeError(f"{label} must be a nonempty tuple")
        if len({plan.story_id for plan in self.story_plans}) != len(
            self.story_plans
        ):
            raise ValueError("story plan IDs must be unique")
        if len({plan.query_ast.query_id for plan in self.query_plans}) != len(
            self.query_plans
        ):
            raise ValueError("query plan IDs must be unique")
        _validate_story_plans(self.world, self.story_plans)
        _validate_query_plans(self.world, self.query_plans)
        _validate_holdout_splits(self.story_plans, self.query_plans)

    @property
    def tasks(self) -> tuple[TaskSpecification, ...]:
        """Expose the world's canonical task specifications."""
        return self.world.tasks

    @property
    def entities(self) -> tuple[Entity, ...]:
        """Expose the world's canonical entities."""
        return self.world.entities

    @property
    def facts(self) -> tuple[GroundAtom, ...]:
        """Expose the world's authoritative direct facts."""
        return self.world.facts

    @property
    def rules(self) -> tuple[HornRule, ...]:
        """Expose the world's authoritative Horn rules."""
        return self.world.rules

    @property
    def closure(self) -> ClosureResult:
        """Expose the world's mechanically verified closure."""
        return self.world.closure


@dataclass(frozen=True, slots=True)
class _QueryTarget:
    kind: QueryKind
    task_id: TaskId
    predicate_id: PredicateId
    arguments: tuple[EntityId | Variable, ...]
    answer_entity_id: EntityId


def generate_calibration_bundle(
    master_seed_sha256: str,
    *,
    direct_facts_per_task: int = 24,
) -> TinyWorldsBundle:
    """Generate the symbolic calibration bundle at a fixed fact capacity."""
    return build_tinyworlds_bundle(
        generate_calibration_world(
            master_seed_sha256,
            direct_facts_per_task=direct_facts_per_task,
        )
    )


def generate_pilot_bundle(
    master_seed_sha256: str,
    *,
    direct_facts_per_task: int = 24,
) -> TinyWorldsBundle:
    """Generate the symbolic pilot bundle at the calibrated fact capacity."""
    return build_tinyworlds_bundle(
        generate_pilot_world(
            master_seed_sha256,
            direct_facts_per_task=direct_facts_per_task,
        )
    )


def build_tinyworlds_bundle(world: SymbolicWorld) -> TinyWorldsBundle:
    """Add deterministic story/query plans to an already verified world."""
    if type(world) is not SymbolicWorld:
        raise TypeError("world must be a SymbolicWorld")
    return TinyWorldsBundle(
        bundle_id=f"{TINYWORLDS_VERSION}:{world.world_id}",
        version=TINYWORLDS_VERSION,
        world=world,
        story_plans=_story_plans(world),
        query_plans=_query_plans(world),
    )


def apply_standard_distractor_mix(bundle: TinyWorldsBundle) -> TinyWorldsBundle:
    """Return the predefined easier mix while preserving every query answer."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    mixed_plans = tuple(
        replace(
            plan,
            candidates=_candidates(
                bundle.world,
                bundle.world.task(plan.task_id).family_id,
                _QueryTarget(
                    plan.kind,
                    plan.task_id,
                    plan.query_ast.clauses[0].predicate_id,
                    plan.query_ast.clauses[0].arguments,
                    plan.answer_entity_id,
                ),
                plan.query_ast.query_id,
                bundle.closure.proof_for(plan.proof.conclusion_atom_id).steps,
                plan.correct_index,
                role_order=_standard_role_order(plan_index),
            ),
        )
        for plan_index, plan in enumerate(bundle.query_plans)
    )
    return TinyWorldsBundle(
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        world=bundle.world,
        story_plans=bundle.story_plans,
        query_plans=mixed_plans,
    )


def _story_plans(world: SymbolicWorld) -> tuple[StoryPlan, ...]:
    fact_by_id = {fact.atom_id: fact for fact in world.facts}
    plans: list[StoryPlan] = []
    for task in world.tasks:
        split_facts = {
            DataSplit.TRAIN: task.direct_fact_ids,
            DataSplit.VALIDATION: task.direct_fact_ids[:8],
            DataSplit.TEST: task.direct_fact_ids[8:16],
        }
        for split in DataSplit:
            fact_ids = split_facts[split]
            rule_ids = task.rule_ids if split is DataSplit.TRAIN else ()
            entity_sequence = tuple(
                str(entity_id)
                for fact_id in fact_ids
                for entity_id in fact_by_id[fact_id].arguments
            )
            record_id = f"story:{world.world_id}:{task.task_id}:{split.value}"
            plans.append(
                StoryPlan(
                    story_id=StoryId(record_id),
                    task_id=task.task_id,
                    split=split,
                    direct_fact_ids=fact_ids,
                    rule_ids=rule_ids,
                    holdout=_holdout(
                        world,
                        record_id,
                        split,
                        template=f"story:{task.kind.value}",
                        plot=str(task.task_id),
                        phrasing="story",
                        entity_sequence=entity_sequence,
                        proof_sequence=tuple(
                            str(item) for item in (*fact_ids, *rule_ids)
                        ),
                    ),
                )
            )
    return tuple(plans)


def _query_plans(world: SymbolicWorld) -> tuple[QueryPlan, ...]:
    family_ids = tuple(dict.fromkeys(task.family_id for task in world.tasks))
    return tuple(
        plan
        for family_id in family_ids
        for plan in _family_query_plans(world, family_id)
    )


def _family_query_plans(
    world: SymbolicWorld,
    family_id: FamilyId,
) -> tuple[QueryPlan, ...]:
    tasks = {
        task.kind: task for task in world.tasks if task.family_id == family_id
    }
    entities_by_id = {entity.entity_id: entity for entity in world.entities}
    task_entities = {
        kind: tuple(entities_by_id[item] for item in task.introduced_entity_ids)
        for kind, task in tasks.items()
    }
    seed_actors = _entities_of_type(task_entities[TaskKind.SEED], ACTOR_TYPE)
    seed_attributes = _entities_of_type(
        task_entities[TaskKind.SEED], ATTRIBUTE_TYPE
    )
    extension_actors = _entities_of_type(
        task_entities[TaskKind.EXTENSION], ACTOR_TYPE
    )
    extension_attributes = _entities_of_type(
        task_entities[TaskKind.EXTENSION], ATTRIBUTE_TYPE
    )
    revision_context = _entities_of_type(
        task_entities[TaskKind.REVISION], CONTEXT_TYPE
    )[0]
    predicates = family_predicates(family_id)
    answer = Variable("answer")
    def targets(variant: int) -> tuple[_QueryTarget, ...]:
        two_hop_actor_index = 2 if variant == 0 else 1
        two_hop_attribute_index = 3 if variant == 0 else 2
        open_book_attribute_index = 3 if variant == 0 else 0
        return (
        _QueryTarget(
            QueryKind.DIRECT,
            tasks[TaskKind.SEED].task_id,
            predicates["base_selects"],
            (seed_actors[variant], answer),
            seed_attributes[variant],
        ),
        _QueryTarget(
            QueryKind.ANCESTOR_PLUS_CHILD,
            tasks[TaskKind.EXTENSION].task_id,
            predicates["ancestor_child_result"],
            (seed_actors[variant], answer),
            extension_attributes[variant],
        ),
        _QueryTarget(
            QueryKind.NEW_INSTANCE,
            tasks[TaskKind.EXTENSION].task_id,
            predicates["extension_known_role"],
            (extension_actors[variant], answer),
            extension_attributes[variant + 1],
        ),
        _QueryTarget(
            QueryKind.ONE_HOP,
            tasks[TaskKind.SEED].task_id,
            predicates["seed_vivid"],
            (seed_actors[variant], answer),
            seed_attributes[variant + 1],
        ),
        _QueryTarget(
            QueryKind.TWO_HOP,
            tasks[TaskKind.SEED].task_id,
            predicates["seed_notable"],
            (seed_actors[two_hop_actor_index], answer),
            seed_attributes[two_hop_attribute_index],
        ),
        _QueryTarget(
            QueryKind.REVISION_SENSITIVE,
            tasks[TaskKind.REVISION].task_id,
            predicates["revision_result"],
            (seed_actors[variant], answer, revision_context),
            extension_attributes[variant],
        ),
        _QueryTarget(
            QueryKind.CROSS_BRANCH,
            tasks[TaskKind.BRIDGE].task_id,
            predicates["bridge_result"],
            (seed_actors[variant], answer),
            extension_attributes[variant],
        ),
        _QueryTarget(
            QueryKind.OPEN_BOOK,
            tasks[TaskKind.SEED].task_id,
            predicates["open_book_selects"],
            (seed_actors[variant], answer),
            seed_attributes[open_book_attribute_index],
        ),
        )
    return tuple(
        _query_plan(
            world,
            family_id,
            target,
            split,
            query_index,
        )
        for variant, split in enumerate((DataSplit.VALIDATION, DataSplit.TEST))
        for query_index, target in enumerate(targets(variant))
    )


def _query_plan(
    world: SymbolicWorld,
    family_id: FamilyId,
    target: _QueryTarget,
    split: DataSplit,
    query_index: int,
) -> QueryPlan:
    query_id = QueryId(
        f"query:{world.world_id}:{family_id}:{split.value}:{target.kind.value}"
    )
    query = QueryAst(
        query_id=query_id,
        answer_variable=Variable("answer"),
        clauses=(AtomPattern(target.predicate_id, target.arguments),),
    )
    answers = answer_query(world.closure, query, world.registry, world.entities)
    if answers != (target.answer_entity_id,):
        raise RuntimeError(
            f"query {query_id} does not have exactly its prescribed answer: {answers}"
        )
    grounded_arguments = tuple(
        target.answer_entity_id if type(argument) is Variable else argument
        for argument in target.arguments
    )
    conclusion = world.closure.atom(
        target.predicate_id,
        grounded_arguments,
    )
    if conclusion is None:
        raise RuntimeError(f"query conclusion is absent from closure: {query_id}")
    proof = world.closure.proof_for(conclusion.atom_id)
    proof_metadata = _proof_metadata(world, proof)
    hard_oracles = (
        ()
        if target.kind is QueryKind.CROSS_BRANCH
        else (target.task_id,)
    )
    candidates = _candidates(
        world,
        family_id,
        target,
        query_id,
        proof.steps,
        query_index % 4,
    )
    entity_sequence = tuple(
        str(argument)
        for argument in grounded_arguments
    ) + tuple(
        str(argument)
        for step in proof.steps
        for argument in step.atom.arguments
    )
    record_id = str(query_id)
    return QueryPlan(
        task_id=target.task_id,
        split=split,
        kind=target.kind,
        query_ast=query,
        answer_entity_id=target.answer_entity_id,
        candidates=candidates,
        correct_index=query_index % 4,
        proof=proof_metadata,
        hard_oracle_task_ids=hard_oracles,
        open_book_fact_ids=(
            proof.supporting_fact_ids
            if target.kind is QueryKind.OPEN_BOOK
            else ()
        ),
        holdout=_holdout(
            world,
            record_id,
            split,
            template=f"query:{target.kind.value}",
            plot=f"{family_id}:{target.kind.value}",
            phrasing=target.kind.value,
            entity_sequence=entity_sequence,
            proof_sequence=(str(proof.proof_id),),
        ),
    )


def _proof_metadata(world: SymbolicWorld, proof: Proof) -> ProofMetadata:
    task_order = {task.task_id: index for index, task in enumerate(world.tasks)}
    owner_by_fact = {
        fact_id: task.task_id
        for task in world.tasks
        for fact_id in task.direct_fact_ids
    }
    owner_by_rule = {
        rule_id: task.task_id for task in world.tasks for rule_id in task.rule_ids
    }
    required = {
        *(owner_by_fact[fact_id] for fact_id in proof.supporting_fact_ids),
        *(owner_by_rule[rule_id] for rule_id in proof.supporting_rule_ids),
    }
    required_tasks = tuple(sorted(required, key=lambda item: task_order[item]))
    edge_by_task = {
        task.task_id: task.incoming_edge_id for task in world.tasks
    }
    return ProofMetadata(
        proof_id=proof.proof_id,
        conclusion_atom_id=proof.conclusion_atom_id,
        supporting_fact_ids=proof.supporting_fact_ids,
        supporting_rule_ids=proof.supporting_rule_ids,
        required_task_ids=required_tasks,
        required_edge_ids=tuple(edge_by_task[task_id] for task_id in required_tasks),
        depth=proof.depth,
    )


def _candidates(
    world: SymbolicWorld,
    family_id: FamilyId,
    target: _QueryTarget,
    query_id: QueryId,
    proof_steps: tuple[ProofStep, ...],
    correct_index: int,
    *,
    role_order: tuple[CandidateRole, ...] = STANDARD_DISTRACTOR_ROLE_CYCLE,
) -> tuple[QueryCandidate, ...]:
    entity_by_id = {entity.entity_id: entity for entity in world.entities}
    answer_type = entity_by_id[target.answer_entity_id].entity_type
    owner_by_entity = {
        entity_id: task.task_id
        for task in world.tasks
        for entity_id in task.introduced_entity_ids
    }
    matching_revisions = tuple(
        record
        for record in world.revisions
        if record.family_id == family_id
        and target.answer_entity_id
        in (record.base_value_entity_id, record.revised_value_entity_id)
    )
    incompatible = tuple(
        record.revised_value_entity_id
        if target.answer_entity_id == record.base_value_entity_id
        else record.base_value_entity_id
        for record in matching_revisions
    )
    competing = tuple(
        entity.entity_id
        for entity in world.entities
        if entity.entity_type == answer_type
        and owner_by_entity[entity.entity_id] != target.task_id
    )
    partial = tuple(
        argument
        for step in proof_steps
        for argument in step.atom.arguments
        if entity_by_id[argument].entity_type == answer_type
        and argument != target.answer_entity_id
    )
    fillers = tuple(
        entity.entity_id
        for entity in world.entities
        if entity.entity_type == answer_type
    )
    selected: list[QueryCandidate] = []
    used = {target.answer_entity_id}
    pool_by_role = {
        CandidateRole.INCOMPATIBLE_REVISION: incompatible,
        CandidateRole.COMPETING_TASK: competing,
        CandidateRole.PARTIAL_PROOF: partial,
        CandidateRole.SAME_TYPE_FILLER: fillers,
    }
    for role in role_order:
        pool = pool_by_role[role]
        for entity_id in _seed_ordered_entities(world, query_id, role, pool):
            if entity_id not in used:
                selected.append(QueryCandidate(entity_id, role))
                used.add(entity_id)
                if role is not CandidateRole.SAME_TYPE_FILLER:
                    break
            if len(selected) == 3:
                break
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(f"query {query_id} lacks three same-type distractors")
    candidates = sorted(selected, key=lambda candidate: _CANDIDATE_PRIORITY[candidate.role])
    candidates.insert(
        correct_index,
        QueryCandidate(target.answer_entity_id, CandidateRole.CORRECT),
    )
    return tuple(candidates)


def _standard_role_order(query_index: int) -> tuple[CandidateRole, ...]:
    omitted = STANDARD_DISTRACTOR_ROLE_CYCLE[
        query_index % len(STANDARD_DISTRACTOR_ROLE_CYCLE)
    ]
    return tuple(
        role for role in STANDARD_DISTRACTOR_ROLE_CYCLE if role is not omitted
    ) + (omitted,)


def _seed_ordered_entities(
    world: SymbolicWorld,
    query_id: QueryId,
    role: CandidateRole,
    entity_ids: tuple[EntityId, ...],
) -> tuple[EntityId, ...]:
    return tuple(
        sorted(
            set(entity_ids),
            key=lambda entity_id: derive_subseed(
                world.master_seed_sha256,
                "query-distractor",
                world.world_id,
                str(query_id),
                role.value,
                str(entity_id),
            ),
        )
    )


def _holdout(
    world: SymbolicWorld,
    record_id: str,
    split: DataSplit,
    *,
    template: str,
    plot: str,
    phrasing: str,
    entity_sequence: tuple[str, ...],
    proof_sequence: tuple[str, ...],
) -> HoldoutMetadata:
    def axis(name: str, *values: str) -> str:
        digest = derive_subseed(
            world.master_seed_sha256,
            "split-axis",
            world.world_id,
            name,
            *values,
        )
        return f"{name}:{digest[:24]}"

    symbolic_text_sha256 = derive_subseed(
        world.master_seed_sha256,
        "symbolic-text",
        world.world_id,
        record_id,
        split.value,
    )
    return HoldoutMetadata(
        template_family_id=axis("template", split.value, template),
        plot_id=axis("plot", split.value, plot),
        query_phrasing_id=axis("phrasing", split.value, phrasing),
        entity_combination_id=axis("combination", *entity_sequence),
        proof_chain_id=axis("proof", *proof_sequence),
        symbolic_text_sha256=symbolic_text_sha256,
    )


def _validate_story_plans(
    world: SymbolicWorld,
    plans: tuple[StoryPlan, ...],
) -> None:
    fact_ids = {fact.atom_id for fact in world.facts}
    rule_ids = {rule.rule_id for rule in world.rules}
    task_by_id = {task.task_id: task for task in world.tasks}
    expected_pairs = {
        (task.task_id, split) for task in world.tasks for split in DataSplit
    }
    if len(plans) != len(expected_pairs) or {
        (plan.task_id, plan.split) for plan in plans
    } != expected_pairs:
        raise ValueError("story plans must cover every task and split exactly once")
    for plan in plans:
        task = task_by_id[plan.task_id]
        if not set(plan.direct_fact_ids).issubset(task.direct_fact_ids):
            raise ValueError("story facts must be direct facts of the owning task")
        if not set(plan.rule_ids).issubset(task.rule_ids):
            raise ValueError("story rules must be rules of the owning task")
        if not set(plan.direct_fact_ids).issubset(fact_ids) or not set(
            plan.rule_ids
        ).issubset(rule_ids):
            raise ValueError("story plans contain dangling symbolic IDs")
        expected_facts = {
            DataSplit.TRAIN: task.direct_fact_ids,
            DataSplit.VALIDATION: task.direct_fact_ids[:8],
            DataSplit.TEST: task.direct_fact_ids[8:16],
        }[plan.split]
        expected_rules = task.rule_ids if plan.split is DataSplit.TRAIN else ()
        if (
            plan.direct_fact_ids != expected_facts
            or plan.rule_ids != expected_rules
        ):
            raise ValueError(
                "story plan content must match the canonical fixed split slice"
            )


def _validate_query_plans(
    world: SymbolicWorld,
    plans: tuple[QueryPlan, ...],
) -> None:
    entity_by_id = {entity.entity_id: entity for entity in world.entities}
    fact_ids = {fact.atom_id for fact in world.facts}
    proof_by_id = {proof.proof_id: proof for proof in world.closure.proofs}
    task_by_id = {task.task_id: task for task in world.tasks}
    for plan in plans:
        if plan.task_id not in task_by_id:
            raise ValueError(f"query references unknown owning task: {plan.task_id}")
        if plan.answer_entity_id not in entity_by_id:
            raise ValueError(
                f"query references unknown answer entity: {plan.answer_entity_id}"
            )
        dangling_candidates = tuple(
            candidate.entity_id
            for candidate in plan.candidates
            if candidate.entity_id not in entity_by_id
        )
        if dangling_candidates:
            raise ValueError(
                f"query candidates reference unknown entities: {dangling_candidates}"
            )
        dangling_constants = tuple(
            argument
            for clause in plan.query_ast.clauses
            for argument in clause.arguments
            if type(argument) is EntityId and argument not in entity_by_id
        )
        if dangling_constants:
            raise ValueError(
                f"query AST references unknown entities: {dangling_constants}"
            )
        dangling_oracles = tuple(
            task_id
            for task_id in plan.hard_oracle_task_ids
            if task_id not in task_by_id
        )
        if dangling_oracles:
            raise ValueError(
                f"query hard oracle references unknown tasks: {dangling_oracles}"
            )
    family_by_task = {task.task_id: task.family_id for task in world.tasks}
    family_ids = tuple(dict.fromkeys(family_by_task.values()))
    for family_id in family_ids:
        family_plans = tuple(
            plan for plan in plans if family_by_task[plan.task_id] == family_id
        )
        for split in (DataSplit.VALIDATION, DataSplit.TEST):
            split_plans = tuple(
                plan for plan in family_plans if plan.split is split
            )
            if tuple(plan.kind for plan in split_plans) != REQUIRED_QUERY_KINDS:
                raise ValueError(
                    "every family and evaluation split must contain all query kinds"
                )
            indices = tuple(
                plan.correct_index for plan in split_plans
            )
            if tuple(sorted(indices)) != (0, 0, 1, 1, 2, 2, 3, 3):
                raise ValueError("candidate indices must balance within each split")
    hard_candidate_matches: list[bool] = []
    standard_candidate_matches: list[bool] = []
    for plan_index, plan in enumerate(plans):
        answers = answer_query(
            world.closure,
            plan.query_ast,
            world.registry,
            world.entities,
        )
        if answers != (plan.answer_entity_id,):
            raise ValueError("query plans must have exactly one graph answer")
        answer_type = entity_by_id[plan.answer_entity_id].entity_type
        if any(
            entity_by_id[candidate.entity_id].entity_type != answer_type
            for candidate in plan.candidates
        ):
            raise ValueError("query candidates must all have the answer's type")
        actual_proof = proof_by_id.get(plan.proof.proof_id)
        if actual_proof is None:
            raise ValueError("query proof metadata must match canonical closure proof")
        authoritative_metadata = _proof_metadata(world, actual_proof)
        if plan.proof != authoritative_metadata:
            raise ValueError(
                "query proof task/edge support and metadata must match "
                "authoritative fact/rule ownership"
            )
        grounded_answer_atoms = _grounded_answer_atoms(plan)
        if (
            actual_proof.conclusion.predicate_id,
            actual_proof.conclusion.arguments,
        ) not in grounded_answer_atoms:
            raise ValueError(
                "query canonical proof conclusion must equal its grounded unique answer"
            )
        hard_candidates = _replay_candidates(world, plan, actual_proof, None)
        standard_candidates = _replay_candidates(
            world,
            plan,
            actual_proof,
            _standard_role_order(plan_index),
        )
        hard_candidate_matches.append(plan.candidates == hard_candidates)
        standard_candidate_matches.append(plan.candidates == standard_candidates)
        if not set(plan.open_book_fact_ids).issubset(fact_ids):
            raise ValueError("query open-book facts contain dangling fact IDs")
        if (
            plan.kind is QueryKind.OPEN_BOOK
            and plan.open_book_fact_ids != actual_proof.supporting_fact_ids
        ):
            raise ValueError(
                "open-book facts must equal the canonical proof's supporting facts"
            )
        noncorrect_roles = tuple(
            candidate.role
            for candidate in plan.candidates
            if candidate.role is not CandidateRole.CORRECT
        )
        if tuple(_CANDIDATE_PRIORITY[role] for role in noncorrect_roles) != tuple(
            sorted(_CANDIDATE_PRIORITY[role] for role in noncorrect_roles)
        ):
            raise ValueError("distractors must follow the required priority")
        required_tasks = set(plan.proof.required_task_ids)
        required_edges = set(plan.proof.required_edge_ids)
        if plan.kind is QueryKind.CROSS_BRANCH:
            required_kinds = {task_by_id[item].kind for item in required_tasks}
            if required_kinds != {
                TaskKind.SEED,
                TaskKind.EXTENSION,
                TaskKind.REVISION,
                TaskKind.BRIDGE,
            }:
                raise ValueError(
                    "cross-branch proof must require seed, extension, revision, bridge"
                )
            if any(
                required_edges.issubset(
                    {
                        task_by_id[path_task].incoming_edge_id
                        for path_task in world.task_path(task.task_id)
                    }
                )
                for task in world.tasks
            ):
                raise ValueError("no hard path may completely support a bridge query")
        else:
            if plan.hard_oracle_task_ids != (plan.task_id,):
                raise ValueError(
                    "non-cross query hard oracle must be its owning task node"
                )
            oracle_path = set(world.task_path(plan.task_id))
            oracle_edges = {
                task_by_id[task_id].incoming_edge_id for task_id in oracle_path
            }
            if not required_tasks.issubset(
                oracle_path
            ) or not required_edges.issubset(oracle_edges):
                raise ValueError(
                    "hard oracle path must contain all required query task/edge support"
                )
    if not all(hard_candidate_matches) and not all(standard_candidate_matches):
        raise ValueError(
            "all query candidates must use one complete predefined distractor policy"
        )


def _grounded_answer_atoms(
    plan: QueryPlan,
) -> tuple[tuple[PredicateId, tuple[EntityId, ...]], ...]:
    """Ground answer-bearing query clauses using the verified unique answer."""
    answer_variable = plan.query_ast.answer_variable
    grounded: list[tuple[PredicateId, tuple[EntityId, ...]]] = []
    for clause in plan.query_ast.clauses:
        if answer_variable not in clause.variables:
            continue
        arguments: list[EntityId] = []
        for argument in clause.arguments:
            if argument == answer_variable:
                arguments.append(plan.answer_entity_id)
            elif type(argument) is EntityId:
                arguments.append(argument)
            else:
                break
        else:
            grounded.append((clause.predicate_id, tuple(arguments)))
    if not grounded:
        raise ValueError(
            "query proof requires an answer-bearing clause groundable by its unique answer"
        )
    return tuple(grounded)


def _replay_candidates(
    world: SymbolicWorld,
    plan: QueryPlan,
    proof: Proof,
    role_order: tuple[CandidateRole, ...] | None,
) -> tuple[QueryCandidate, ...]:
    target = _QueryTarget(
        plan.kind,
        plan.task_id,
        plan.query_ast.clauses[0].predicate_id,
        plan.query_ast.clauses[0].arguments,
        plan.answer_entity_id,
    )
    family_id = world.task(plan.task_id).family_id
    if role_order is None:
        return _candidates(
            world,
            family_id,
            target,
            plan.query_ast.query_id,
            proof.steps,
            plan.correct_index,
        )
    return _candidates(
        world,
        family_id,
        target,
        plan.query_ast.query_id,
        proof.steps,
        plan.correct_index,
        role_order=role_order,
    )


def _validate_holdout_splits(
    story_plans: tuple[StoryPlan, ...],
    query_plans: tuple[QueryPlan, ...],
) -> None:
    records = tuple(
        (plan.split, plan.holdout) for plan in (*story_plans, *query_plans)
    )
    for field_name in (
        "template_family_id",
        "plot_id",
        "query_phrasing_id",
        "entity_combination_id",
        "proof_chain_id",
        "symbolic_text_sha256",
    ):
        values_by_split = {
            split: {
                getattr(metadata, field_name)
                for record_split, metadata in records
                if record_split is split
            }
            for split in DataSplit
        }
        if any(
            values_by_split[left] & values_by_split[right]
            for left, right in (
                (DataSplit.TRAIN, DataSplit.VALIDATION),
                (DataSplit.TRAIN, DataSplit.TEST),
                (DataSplit.VALIDATION, DataSplit.TEST),
            )
        ):
            raise ValueError(f"{field_name} must be disjoint across splits")


def _entities_of_type(
    entities: tuple[Entity, ...],
    entity_type: EntityTypeId,
) -> tuple[EntityId, ...]:
    return tuple(
        entity.entity_id for entity in entities if entity.entity_type == entity_type
    )


__all__ = [
    "REQUIRED_QUERY_KINDS",
    "TinyWorldsBundle",
    "build_tinyworlds_bundle",
    "apply_standard_distractor_mix",
    "generate_calibration_bundle",
    "generate_pilot_bundle",
    "STANDARD_DISTRACTOR_ROLE_CYCLE",
]
