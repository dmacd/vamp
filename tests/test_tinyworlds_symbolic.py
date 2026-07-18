from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from apm.data.text.tinyworlds import (
    AtomId,
    AtomPattern,
    CALIBRATION_TASK_IDS,
    CandidateRole,
    DataSplit,
    Entity,
    EntityId,
    EntityTypeId,
    FamilyId,
    GroundAtom,
    HornRule,
    PredicateId,
    PredicateKind,
    PredicateRegistry,
    PredicateSignature,
    PILOT_TASK_IDS,
    QueryAst,
    QueryId,
    QueryKind,
    RuleId,
    TaskId,
    TaskKind,
    TaskSpecification,
    Variable,
    apply_standard_distractor_mix,
    answer_query,
    compute_closure,
    derive_master_seed,
    derive_subseed,
    family_predicates,
    generate_calibration_bundle,
    generate_calibration_world,
    generate_pilot_bundle,
    generate_pilot_world,
    predicate_topological_order,
    subseed_uint64,
    validate_task_specifications,
)


PERSON = EntityTypeId("person")
COLOR = EntityTypeId("color")
CONTEXT = EntityTypeId("context")
PARENT_OF = PredicateId("parent_of")
RELATED_TO = PredicateId("related_to")
KNOWS = PredicateId("knows")
REMEMBERS = PredicateId("remembers")
HAS_COLOR = PredicateId("has_color")
CONTEXTUAL_TRUST = PredicateId("contextual_trust")


def _master_seed() -> str:
    return derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )


def _entities() -> tuple[Entity, ...]:
    return (
        Entity(EntityId("ada"), PERSON, "Ada", ("Ada's",)),
        Entity(EntityId("bo"), PERSON, "Bo"),
        Entity(EntityId("cy"), PERSON, "Cy"),
        Entity(EntityId("red"), COLOR, "red"),
        Entity(EntityId("winter"), CONTEXT, "winter"),
    )


def _registry() -> PredicateRegistry:
    return PredicateRegistry(
        entity_types=(PERSON, COLOR, CONTEXT),
        context_type_id=CONTEXT,
        predicates=(
            PredicateSignature(PARENT_OF, (PERSON, PERSON)),
            PredicateSignature(RELATED_TO, (PERSON, PERSON)),
            PredicateSignature(KNOWS, (PERSON, PERSON)),
            PredicateSignature(REMEMBERS, (PERSON, PERSON)),
            PredicateSignature(HAS_COLOR, (PERSON, COLOR)),
            PredicateSignature(
                CONTEXTUAL_TRUST,
                (PERSON, PERSON, CONTEXT),
                PredicateKind.CONTEXTUAL,
            ),
        ),
    )


def _facts() -> tuple[GroundAtom, ...]:
    return (
        GroundAtom(
            AtomId("fact:ada-parent-bo"),
            PARENT_OF,
            (EntityId("ada"), EntityId("bo")),
        ),
        GroundAtom(
            AtomId("fact:bo-parent-cy"),
            PARENT_OF,
            (EntityId("bo"), EntityId("cy")),
        ),
        GroundAtom(
            AtomId("fact:ada-red"),
            HAS_COLOR,
            (EntityId("ada"), EntityId("red")),
        ),
        GroundAtom(
            AtomId("fact:trust-winter"),
            CONTEXTUAL_TRUST,
            (EntityId("ada"), EntityId("bo"), EntityId("winter")),
        ),
    )


def _rules() -> tuple[HornRule, ...]:
    source = Variable("source")
    target = Variable("target")
    parent_pattern = AtomPattern(PARENT_OF, (source, target))
    related_pattern = AtomPattern(RELATED_TO, (source, target))
    knows_pattern = AtomPattern(KNOWS, (source, target))
    return (
        HornRule(
            RuleId("r-related-z"),
            related_pattern,
            (parent_pattern,),
        ),
        HornRule(
            RuleId("r-related-a"),
            related_pattern,
            (parent_pattern,),
        ),
        HornRule(
            RuleId("r-knows"),
            knows_pattern,
            (related_pattern,),
        ),
        HornRule(
            RuleId("r-remembers"),
            AtomPattern(REMEMBERS, (source, target)),
            (knows_pattern,),
        ),
    )


def test_ids_entities_atoms_and_registry_are_strict_and_immutable() -> None:
    ada = _entities()[0]

    assert type(ada.entity_id) is EntityId
    with pytest.raises(FrozenInstanceError):
        ada.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="EntityId"):
        GroundAtom(AtomId("bad"), PARENT_OF, ("ada", EntityId("bo")))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unary or binary"):
        PredicateSignature(
            PredicateId("ordinary_ternary"),
            (PERSON, PERSON, CONTEXT),
        )
    with pytest.raises(ValueError, match="explicitly ternary"):
        PredicateSignature(
            PredicateId("contextual_binary"),
            (PERSON, CONTEXT),
            PredicateKind.CONTEXTUAL,
        )
    invalid_context_registry = PredicateRegistry
    with pytest.raises(ValueError, match="third argument"):
        invalid_context_registry(
            entity_types=(PERSON, CONTEXT),
            context_type_id=CONTEXT,
            predicates=(
                PredicateSignature(
                    PredicateId("bad_context"),
                    (PERSON, PERSON, PERSON),
                    PredicateKind.CONTEXTUAL,
                ),
            ),
        )

    registry = _registry()
    registry.validate_ground_atom(_facts()[-1], _entities())
    assert registry.signature(CONTEXTUAL_TRUST).arity == 3


def test_positive_safety_and_shared_variable_types_are_enforced() -> None:
    x = Variable("x")
    y = Variable("y")
    unbound = Variable("unbound")

    with pytest.raises(ValueError, match="head variable"):
        HornRule(
            RuleId("unsafe"),
            AtomPattern(RELATED_TO, (x, unbound)),
            (AtomPattern(PARENT_OF, (x, y)),),
        )

    ill_typed = HornRule(
        RuleId("ill-typed"),
        AtomPattern(RELATED_TO, (x, y)),
        (
            AtomPattern(PARENT_OF, (x, y)),
            AtomPattern(HAS_COLOR, (y, x)),
        ),
    )
    with pytest.raises(ValueError, match="incompatible types"):
        _registry().validate_rule(ill_typed, _entities())


def test_dependency_cycles_are_rejected_before_closure() -> None:
    x = Variable("x")
    y = Variable("y")
    forward = HornRule(
        RuleId("forward"),
        AtomPattern(RELATED_TO, (x, y)),
        (AtomPattern(PARENT_OF, (x, y)),),
    )
    reverse = HornRule(
        RuleId("reverse"),
        AtomPattern(PARENT_OF, (x, y)),
        (AtomPattern(RELATED_TO, (x, y)),),
    )

    with pytest.raises(ValueError, match="acyclic"):
        predicate_topological_order((forward, reverse))
    with pytest.raises(ValueError, match="acyclic"):
        compute_closure(
            _facts(),
            (forward, reverse),
            _registry(),
            _entities(),
        )


def test_depth_two_closure_and_canonical_proofs_are_order_independent() -> None:
    first = compute_closure(_facts(), _rules(), _registry(), _entities())
    reordered = compute_closure(
        tuple(reversed(_facts())),
        tuple(reversed(_rules())),
        _registry(),
        tuple(reversed(_entities())),
    )

    assert first == reordered
    related = first.atom(RELATED_TO, (EntityId("ada"), EntityId("bo")))
    knows = first.atom(KNOWS, (EntityId("ada"), EntityId("bo")))
    remembers = first.atom(REMEMBERS, (EntityId("ada"), EntityId("bo")))
    assert related is not None and knows is not None
    assert remembers is None
    related_proof = first.proof_for(related.atom_id)
    knows_proof = first.proof_for(knows.atom_id)
    assert related_proof.depth == 1
    assert related_proof.supporting_rule_ids == (RuleId("r-related-a"),)
    assert knows_proof.depth == 2
    assert knows_proof.supporting_fact_ids == (AtomId("fact:ada-parent-bo"),)
    assert knows_proof.supporting_rule_ids == (
        RuleId("r-knows"),
        RuleId("r-related-a"),
    )
    assert tuple(step.depth for step in knows_proof.steps) == (0, 1, 2)

    one_hop_only = compute_closure(
        _facts(),
        _rules(),
        _registry(),
        _entities(),
        max_depth=1,
    )
    assert one_hop_only.atom(KNOWS, (EntityId("ada"), EntityId("bo"))) is None
    with pytest.raises(ValueError, match="zero to two"):
        compute_closure(
            _facts(),
            _rules(),
            _registry(),
            _entities(),
            max_depth=3,
        )


def test_query_ast_is_canonical_typed_and_executes_against_closure() -> None:
    answer = Variable("answer")
    query = QueryAst(
        QueryId("q-ada-knows"),
        answer,
        (
            AtomPattern(HAS_COLOR, (EntityId("ada"), EntityId("red"))),
            AtomPattern(KNOWS, (EntityId("ada"), answer)),
        ),
    )

    assert query.clauses == tuple(
        sorted(query.clauses, key=lambda clause: clause.canonical_key)
    )
    assert _registry().answer_type(query, _entities()) == PERSON
    assert answer_query(
        compute_closure(_facts(), _rules(), _registry(), _entities()),
        query,
        _registry(),
        _entities(),
    ) == (EntityId("bo"),)

    with pytest.raises(ValueError, match="answer variable"):
        QueryAst(
            QueryId("q-unbound"),
            Variable("missing"),
            (AtomPattern(PARENT_OF, (EntityId("ada"), answer)),),
        )


def test_task_specs_validate_seed_extension_revision_bridge_topology() -> None:
    seed = TaskSpecification(
        TaskId("willow-seed"),
        FamilyId("willow"),
        TaskKind.SEED,
        None,
        (AtomId("f-seed"),),
        (RuleId("rule-seed"),),
    )
    extension = TaskSpecification(
        TaskId("willow-extension"),
        FamilyId("willow"),
        TaskKind.EXTENSION,
        seed.task_id,
        (AtomId("f-extension"),),
        (RuleId("rule-extension"),),
    )
    revision = TaskSpecification(
        TaskId("willow-revision"),
        FamilyId("willow"),
        TaskKind.REVISION,
        seed.task_id,
        (AtomId("f-revision"),),
        (),
    )
    bridge = TaskSpecification(
        TaskId("willow-bridge"),
        FamilyId("willow"),
        TaskKind.BRIDGE,
        revision.task_id,
        (AtomId("f-bridge"),),
        (),
    )

    validate_task_specifications((seed, extension, revision, bridge))
    with pytest.raises(ValueError, match="revision parent"):
        validate_task_specifications(
            (
                seed,
                TaskSpecification(
                    TaskId("bad-bridge"),
                    FamilyId("willow"),
                    TaskKind.BRIDGE,
                    seed.task_id,
                    (AtomId("f-bad"),),
                    (),
                ),
            )
        )
    with pytest.raises(ValueError, match="appear earlier"):
        validate_task_specifications((extension, seed))


def test_master_and_namespaced_seed_derivation_is_pinned_and_independent() -> None:
    master = derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )

    assert master == "38c6f7ed8303f018600b77de85eac4f77c326f44e83c33d394be4110c7bd9950"
    willow_entity_seed = derive_subseed(master, "entities", "willow", 3)
    assert willow_entity_seed == (
        "8b80a0f841b88fe486852db2744c0685d04ba023b6576bb4d0f037181b809cdc"
    )
    assert subseed_uint64(master, "entities", "willow", 3) == 10_052_211_356_405_895_140
    assert derive_subseed(master, "entities", "willow", 3) == willow_entity_seed
    assert derive_subseed(master, "entities", "willow", 4) != willow_entity_seed
    assert derive_subseed(master, "rules", "willow", 3) != willow_entity_seed
    with pytest.raises(ValueError, match="SHA-256"):
        derive_subseed("not-a-digest", "entities")


def test_calibration_and_pilot_topologies_and_initial_quantities() -> None:
    calibration = generate_calibration_world(_master_seed())
    pilot = generate_pilot_world(_master_seed())

    assert tuple(task.task_id for task in calibration.tasks) == CALIBRATION_TASK_IDS
    assert tuple(task.task_id for task in pilot.tasks) == PILOT_TASK_IDS
    assert tuple(task.parent_task_id for task in calibration.tasks) == (
        None,
        CALIBRATION_TASK_IDS[0],
        CALIBRATION_TASK_IDS[0],
        CALIBRATION_TASK_IDS[2],
    )
    assert tuple(task.parent_task_id for task in pilot.tasks) == (
        None,
        None,
        PILOT_TASK_IDS[0],
        PILOT_TASK_IDS[1],
        PILOT_TASK_IDS[0],
        PILOT_TASK_IDS[1],
        PILOT_TASK_IDS[4],
        PILOT_TASK_IDS[5],
    )
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
    for world in (calibration, pilot):
        assert all(
            len(task.introduced_entity_ids) == expected_entities[task.kind]
            and len(task.direct_fact_ids) == 24
            and len(task.rule_ids) == expected_rules[task.kind]
            for task in world.tasks
        )
        for bridge in (task for task in world.tasks if task.kind is TaskKind.BRIDGE):
            bridge_predicate = family_predicates(bridge.family_id)["bridge_link"]
            bridge_fact_ids = {
                fact.atom_id
                for fact in world.facts
                if fact.predicate_id == bridge_predicate
            }
            assert len(bridge_fact_ids) == 3
            assert bridge_fact_ids.issubset(bridge.direct_fact_ids)
    assert not {entity.name for entity in calibration.entities} & {
        entity.name for entity in pilot.entities
    }


def test_world_and_query_generation_is_byte_choice_deterministic_by_namespace() -> None:
    master = _master_seed()
    first = generate_pilot_bundle(master)
    derive_subseed(master, "unrelated-new-namespace", "extra-record")
    repeated = generate_pilot_bundle(master)
    changed_master = derive_master_seed(
        "tinyworlds-v1",
        1,
        "a" * 64,
        "b" * 64,
    )
    changed = generate_pilot_bundle(changed_master)

    assert first == repeated
    assert tuple(entity.entity_id for entity in first.entities) == tuple(
        entity.entity_id for entity in changed.entities
    )
    assert tuple(entity.name for entity in first.entities) != tuple(
        entity.name for entity in changed.entities
    )
    assert tuple(fact.atom_id for fact in first.facts) != tuple(
        fact.atom_id for fact in changed.facts
    )


def test_every_family_has_unique_exact_queries_and_balanced_candidates() -> None:
    bundle = generate_pilot_bundle(_master_seed())
    entity_by_id = {entity.entity_id: entity for entity in bundle.entities}
    family_by_task = {task.task_id: task.family_id for task in bundle.tasks}

    for family_id in (FamilyId("willow"), FamilyId("sunny")):
        plans = tuple(
            plan
            for plan in bundle.query_plans
            if family_by_task[plan.task_id] == family_id
        )
        for split in (DataSplit.VALIDATION, DataSplit.TEST):
            split_plans = tuple(plan for plan in plans if plan.split is split)
            assert tuple(plan.kind for plan in split_plans) == tuple(QueryKind)
            assert sorted(
                plan.correct_index for plan in split_plans
            ) == [0, 0, 1, 1, 2, 2, 3, 3]
        for plan in plans:
            assert answer_query(
                bundle.closure,
                plan.query_ast,
                bundle.world.registry,
                bundle.entities,
            ) == (plan.answer_entity_id,)
            assert len(plan.candidates) == 4
            assert len({candidate.entity_id for candidate in plan.candidates}) == 4
            assert all(
                entity_by_id[candidate.entity_id].entity_type
                == entity_by_id[plan.answer_entity_id].entity_type
                for candidate in plan.candidates
            )
            wrong_roles = tuple(
                candidate.role
                for candidate in plan.candidates
                if candidate.role is not CandidateRole.CORRECT
            )
            priority = {
                CandidateRole.INCOMPATIBLE_REVISION: 0,
                CandidateRole.COMPETING_TASK: 1,
                CandidateRole.PARTIAL_PROOF: 2,
                CandidateRole.SAME_TYPE_FILLER: 3,
            }
            assert tuple(priority[role] for role in wrong_roles) == tuple(
                sorted(priority[role] for role in wrong_roles)
            )


def test_query_proof_depth_and_task_support_match_reasoning_kind() -> None:
    bundle = generate_calibration_bundle(_master_seed())

    for split in (DataSplit.VALIDATION, DataSplit.TEST):
        by_kind = {
            plan.kind: plan
            for plan in bundle.query_plans
            if plan.split is split
        }
        assert set(by_kind) == set(QueryKind)
        assert by_kind[QueryKind.DIRECT].proof.depth == 0
        assert by_kind[QueryKind.ONE_HOP].proof.depth == 1
        assert by_kind[QueryKind.TWO_HOP].proof.depth == 2
        assert by_kind[QueryKind.NEW_INSTANCE].proof.depth == 1
        assert by_kind[QueryKind.ANCESTOR_PLUS_CHILD].proof.required_task_ids == (
            CALIBRATION_TASK_IDS[0],
            CALIBRATION_TASK_IDS[1],
        )
        assert by_kind[QueryKind.REVISION_SENSITIVE].proof.required_task_ids == (
            CALIBRATION_TASK_IDS[0],
            CALIBRATION_TASK_IDS[2],
        )
        assert len(by_kind[QueryKind.TWO_HOP].proof.supporting_rule_ids) == 2
        assert by_kind[QueryKind.OPEN_BOOK].open_book_fact_ids == (
            by_kind[QueryKind.OPEN_BOOK].proof.supporting_fact_ids
        )


def test_revisions_are_contextual_incompatible_and_retain_both_answers() -> None:
    bundle = generate_pilot_bundle(_master_seed())
    fact_by_id = {fact.atom_id: fact for fact in bundle.facts}
    family_by_task = {task.task_id: task.family_id for task in bundle.tasks}

    for family_id in (FamilyId("willow"), FamilyId("sunny")):
        records = tuple(
            record for record in bundle.world.revisions if record.family_id == family_id
        )
        assert len(records) == 3
        assert all(
            record.base_value_entity_id != record.revised_value_entity_id
            for record in records
        )
        for record in records:
            base = fact_by_id[record.base_atom_id]
            contextual = fact_by_id[record.contextual_atom_id]
            assert base.arguments[1] == record.base_value_entity_id
            assert contextual.arguments[1] == record.revised_value_entity_id
            assert contextual.arguments[2] == record.context_entity_id
            assert (
                bundle.world.registry.signature(contextual.predicate_id).kind
                is PredicateKind.CONTEXTUAL
            )
        revision_plan = next(
            plan
            for plan in bundle.query_plans
            if family_by_task[plan.task_id] == family_id
            and plan.kind is QueryKind.REVISION_SENSITIVE
        )
        direct_plan = next(
            plan
            for plan in bundle.query_plans
            if family_by_task[plan.task_id] == family_id
            and plan.kind is QueryKind.DIRECT
        )
        matching = next(
            record
            for record in records
            if record.subject_entity_id
            in revision_plan.query_ast.clauses[0].arguments
        )
        assert direct_plan.answer_entity_id == matching.base_value_entity_id
        assert revision_plan.answer_entity_id == matching.revised_value_entity_id


def test_bridge_queries_require_both_siblings_and_have_no_hard_oracle() -> None:
    bundle = generate_pilot_bundle(_master_seed())
    task_by_id = {task.task_id: task for task in bundle.tasks}
    family_by_task = {task.task_id: task.family_id for task in bundle.tasks}

    bridge_plans = tuple(
        plan for plan in bundle.query_plans if plan.kind is QueryKind.CROSS_BRANCH
    )
    assert len(bridge_plans) == 4
    for plan in bridge_plans:
        assert plan.hard_oracle_task_ids == ()
        required_kinds = {
            task_by_id[task_id].kind for task_id in plan.proof.required_task_ids
        }
        assert required_kinds == set(TaskKind)
        extension_task = next(
            task
            for task in bundle.tasks
            if task.family_id == family_by_task[plan.task_id]
            and task.kind is TaskKind.EXTENSION
        )
        assert extension_task.task_id not in bundle.world.task_path(plan.task_id)
        required_edges = set(plan.proof.required_edge_ids)
        assert all(
            not required_edges.issubset(
                {
                    task_by_id[path_task].incoming_edge_id
                    for path_task in bundle.world.task_path(task.task_id)
                }
            )
            for task in bundle.tasks
        )


def test_symbolic_splits_are_disjoint_and_training_contains_no_conclusions() -> None:
    bundle = generate_calibration_bundle(_master_seed())
    records = tuple(
        (plan.split, plan.holdout)
        for plan in (*bundle.story_plans, *bundle.query_plans)
    )
    for field_name in (
        "template_family_id",
        "plot_id",
        "query_phrasing_id",
        "entity_combination_id",
        "proof_chain_id",
        "symbolic_text_sha256",
    ):
        by_split = {
            split: {
                getattr(metadata, field_name)
                for record_split, metadata in records
                if record_split is split
            }
            for split in DataSplit
        }
        assert not by_split[DataSplit.TRAIN] & by_split[DataSplit.VALIDATION]
        assert not by_split[DataSplit.TRAIN] & by_split[DataSplit.TEST]
        assert not by_split[DataSplit.VALIDATION] & by_split[DataSplit.TEST]

    direct_ids = {fact.atom_id for fact in bundle.facts}
    assert all(
        set(plan.direct_fact_ids).issubset(direct_ids)
        for plan in bundle.story_plans
        if plan.split is DataSplit.TRAIN
    )
    direct_semantics = {fact.semantic_key for fact in bundle.facts}
    assert all(
        proof.conclusion.semantic_key not in direct_semantics
        for proof in bundle.closure.proofs
        if proof.depth > 0
    )


def test_calibration_can_add_deterministic_filler_facts_without_reshuffling_core() -> None:
    baseline = generate_calibration_bundle(_master_seed())
    expanded = generate_calibration_bundle(
        _master_seed(),
        direct_facts_per_task=36,
    )

    assert all(len(task.direct_fact_ids) == 24 for task in baseline.tasks)
    assert all(len(task.direct_fact_ids) == 36 for task in expanded.tasks)
    assert {fact.atom_id for fact in baseline.facts}.issubset(
        {fact.atom_id for fact in expanded.facts}
    )
    assert baseline.entities == expanded.entities
    assert baseline.rules == expanded.rules
    assert baseline.world.revisions == expanded.world.revisions


def test_standard_distractor_mix_preserves_answers_and_changes_role_omissions() -> None:
    hard = generate_calibration_bundle(_master_seed())
    standard = apply_standard_distractor_mix(hard)

    assert tuple(plan.answer_entity_id for plan in standard.query_plans) == tuple(
        plan.answer_entity_id for plan in hard.query_plans
    )
    assert tuple(plan.correct_index for plan in standard.query_plans) == tuple(
        plan.correct_index for plan in hard.query_plans
    )
    assert any(
        tuple(candidate.role for candidate in mixed.candidates)
        != tuple(candidate.role for candidate in hard_plan.candidates)
        for hard_plan, mixed in zip(hard.query_plans, standard.query_plans)
    )
