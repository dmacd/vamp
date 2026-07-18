"""Strict immutable symbolic contracts for TinyWorlds knowledge graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata


_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_VARIABLE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class _TypedId(str):
    """Runtime-distinct, immutable, serialization-friendly identifier base."""

    __slots__ = ()

    def __new__(cls, value: str):
        if type(value) is not str:
            raise TypeError(f"{cls.__name__} requires a plain string")
        if _ID_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"{cls.__name__} must match {_ID_PATTERN.pattern!r}"
            )
        return str.__new__(cls, value)


class EntityId(_TypedId):
    """Unique entity identity."""


class EntityTypeId(_TypedId):
    """Unique ontology entity-type identity."""


class PredicateId(_TypedId):
    """Unique predicate identity."""


class AtomId(_TypedId):
    """Unique ground-atom identity."""


class RuleId(_TypedId):
    """Unique Horn-rule identity."""


class TaskId(_TypedId):
    """Unique continual-learning task identity."""


class FamilyId(_TypedId):
    """Unique fictional-world family identity."""


class QueryId(_TypedId):
    """Unique symbolic-query identity."""


class ProofId(_TypedId):
    """Unique canonical-proof identity."""


class TaskEdgeId(_TypedId):
    """Unique root-or-task-to-task graph-edge identity."""


class StoryId(_TypedId):
    """Unique symbolic story-plan identity."""


@dataclass(frozen=True, slots=True)
class Entity:
    """One typed nonce entity and its visible lexical forms."""

    entity_id: EntityId
    entity_type: EntityTypeId
    name: str
    inflections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_id(self.entity_id, EntityId, "entity_id")
        _require_exact_id(self.entity_type, EntityTypeId, "entity_type")
        _validate_surface_form(self.name, "entity name")
        if type(self.inflections) is not tuple:
            raise TypeError("entity inflections must be a tuple")
        if any(type(form) is not str for form in self.inflections):
            raise TypeError("entity inflections must contain strings")
        for form in self.inflections:
            _validate_surface_form(form, "entity inflection")
        if len({form.casefold() for form in self.inflections}) != len(
            self.inflections
        ):
            raise ValueError("entity inflections must be case-insensitively unique")
        if self.name.casefold() in {form.casefold() for form in self.inflections}:
            raise ValueError("entity name cannot be repeated as an inflection")


@dataclass(frozen=True, slots=True, order=True)
class Variable:
    """A named logic variable used by rule and query atom patterns."""

    name: str

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("variable name must be a string")
        if _VARIABLE_PATTERN.fullmatch(self.name) is None:
            raise ValueError(
                f"variable name must match {_VARIABLE_PATTERN.pattern!r}"
            )


Term = EntityId | Variable


@dataclass(frozen=True, slots=True)
class GroundAtom:
    """One fully grounded predicate application."""

    atom_id: AtomId
    predicate_id: PredicateId
    arguments: tuple[EntityId, ...]

    def __post_init__(self) -> None:
        _require_exact_id(self.atom_id, AtomId, "atom_id")
        _require_exact_id(self.predicate_id, PredicateId, "predicate_id")
        if type(self.arguments) is not tuple:
            raise TypeError("ground-atom arguments must be a tuple")
        if not 1 <= len(self.arguments) <= 3:
            raise ValueError("ground atoms must have between one and three arguments")
        if any(type(argument) is not EntityId for argument in self.arguments):
            raise TypeError("ground-atom arguments must be EntityId values")

    @property
    def semantic_key(self) -> tuple[str, tuple[str, ...]]:
        """Return the identity-independent predicate-and-arguments key."""
        return (
            str(self.predicate_id),
            tuple(str(argument) for argument in self.arguments),
        )


@dataclass(frozen=True, slots=True)
class AtomPattern:
    """One predicate application containing constants and/or variables."""

    predicate_id: PredicateId
    arguments: tuple[Term, ...]

    def __post_init__(self) -> None:
        _require_exact_id(self.predicate_id, PredicateId, "predicate_id")
        if type(self.arguments) is not tuple:
            raise TypeError("atom-pattern arguments must be a tuple")
        if not 1 <= len(self.arguments) <= 3:
            raise ValueError(
                "atom patterns must have between one and three arguments"
            )
        if any(
            type(argument) not in (EntityId, Variable)
            for argument in self.arguments
        ):
            raise TypeError(
                "atom-pattern arguments must be EntityId or Variable values"
            )

    @property
    def variables(self) -> frozenset[Variable]:
        """Return every variable mentioned by the pattern."""
        return frozenset(
            argument
            for argument in self.arguments
            if type(argument) is Variable
        )

    @property
    def canonical_key(self) -> tuple[str, tuple[str, ...]]:
        """Return a stable structural key that distinguishes terms by kind."""
        return (
            str(self.predicate_id),
            tuple(
                f"entity:{argument}"
                if type(argument) is EntityId
                else f"variable:{argument.name}"
                for argument in self.arguments
            ),
        )


@dataclass(frozen=True, slots=True)
class HornRule:
    """One positive, range-restricted Horn rule with a canonical body."""

    rule_id: RuleId
    head: AtomPattern
    body: tuple[AtomPattern, ...]

    def __post_init__(self) -> None:
        _require_exact_id(self.rule_id, RuleId, "rule_id")
        if type(self.head) is not AtomPattern:
            raise TypeError("Horn-rule head must be an AtomPattern")
        if type(self.body) is not tuple or not self.body:
            raise ValueError("Horn-rule body must be a nonempty tuple")
        if any(type(atom) is not AtomPattern for atom in self.body):
            raise TypeError("Horn-rule body must contain AtomPattern values")
        if len(set(self.body)) != len(self.body):
            raise ValueError("Horn-rule body cannot repeat an atom pattern")
        canonical_body = tuple(sorted(self.body, key=lambda atom: atom.canonical_key))
        object.__setattr__(self, "body", canonical_body)
        body_variables = frozenset(
            variable for atom in canonical_body for variable in atom.variables
        )
        if not self.head.variables.issubset(body_variables):
            raise ValueError(
                "positive-safe Horn rules require every head variable in the body"
            )


class TaskKind(str, Enum):
    """Supported TinyWorlds task roles within one family topology."""

    SEED = "seed"
    EXTENSION = "extension"
    REVISION = "revision"
    BRIDGE = "bridge"


@dataclass(frozen=True, slots=True)
class TaskSpecification:
    """One task's family/topology role and directly introduced knowledge IDs."""

    task_id: TaskId
    family_id: FamilyId
    kind: TaskKind
    parent_task_id: TaskId | None
    direct_fact_ids: tuple[AtomId, ...]
    rule_ids: tuple[RuleId, ...]
    introduced_entity_ids: tuple[EntityId, ...] = ()
    incoming_edge_id: TaskEdgeId | None = None

    def __post_init__(self) -> None:
        _require_exact_id(self.task_id, TaskId, "task_id")
        _require_exact_id(self.family_id, FamilyId, "family_id")
        if type(self.kind) is not TaskKind:
            raise TypeError("task kind must be a TaskKind")
        if self.parent_task_id is not None:
            _require_exact_id(self.parent_task_id, TaskId, "parent_task_id")
        if self.kind is TaskKind.SEED and self.parent_task_id is not None:
            raise ValueError("seed tasks attach to the root and have no task parent")
        if self.kind is not TaskKind.SEED and self.parent_task_id is None:
            raise ValueError("non-seed tasks require a task parent")
        if self.parent_task_id == self.task_id:
            raise ValueError("tasks cannot parent themselves")
        fact_ids = _validated_id_tuple(
            self.direct_fact_ids,
            AtomId,
            "direct_fact_ids",
            require_nonempty=True,
        )
        rule_ids = _validated_id_tuple(
            self.rule_ids,
            RuleId,
            "rule_ids",
            require_nonempty=False,
        )
        entity_ids = _validated_id_tuple(
            self.introduced_entity_ids,
            EntityId,
            "introduced_entity_ids",
            require_nonempty=False,
        )
        if self.incoming_edge_id is not None:
            _require_exact_id(
                self.incoming_edge_id,
                TaskEdgeId,
                "incoming_edge_id",
            )
        object.__setattr__(self, "direct_fact_ids", tuple(sorted(fact_ids)))
        object.__setattr__(self, "rule_ids", tuple(sorted(rule_ids)))
        object.__setattr__(
            self,
            "introduced_entity_ids",
            tuple(sorted(entity_ids)),
        )


def validate_task_specifications(
    tasks: tuple[TaskSpecification, ...],
) -> None:
    """Validate a parent-before-child collection of family task trees."""
    if type(tasks) is not tuple or not tasks:
        raise ValueError("task specifications must be a nonempty tuple")
    if any(type(task) is not TaskSpecification for task in tasks):
        raise TypeError("tasks must contain TaskSpecification values")
    task_by_id: dict[TaskId, TaskSpecification] = {}
    seed_by_family: dict[FamilyId, TaskId] = {}
    fact_owners: dict[AtomId, TaskId] = {}
    rule_owners: dict[RuleId, TaskId] = {}
    entity_owners: dict[EntityId, TaskId] = {}
    edge_owners: dict[TaskEdgeId, TaskId] = {}
    allowed_parent_kind = {
        TaskKind.EXTENSION: TaskKind.SEED,
        TaskKind.REVISION: TaskKind.SEED,
        TaskKind.BRIDGE: TaskKind.REVISION,
    }
    for task in tasks:
        if task.task_id in task_by_id:
            raise ValueError(f"duplicate task ID: {task.task_id}")
        if task.kind is TaskKind.SEED:
            if task.family_id in seed_by_family:
                raise ValueError(f"family has multiple seed tasks: {task.family_id}")
            seed_by_family[task.family_id] = task.task_id
        else:
            parent = task_by_id.get(task.parent_task_id)
            if parent is None:
                raise ValueError(
                    f"task parent must appear earlier: {task.parent_task_id}"
                )
            if parent.family_id != task.family_id:
                raise ValueError("task and parent must belong to the same family")
            if parent.kind is not allowed_parent_kind[task.kind]:
                raise ValueError(
                    f"{task.kind.value} task requires a "
                    f"{allowed_parent_kind[task.kind].value} parent"
                )
        for fact_id in task.direct_fact_ids:
            if fact_id in fact_owners:
                raise ValueError(
                    f"fact {fact_id} is introduced by multiple tasks"
                )
            fact_owners[fact_id] = task.task_id
        for rule_id in task.rule_ids:
            if rule_id in rule_owners:
                raise ValueError(
                    f"rule {rule_id} is introduced by multiple tasks"
                )
            rule_owners[rule_id] = task.task_id
        for entity_id in task.introduced_entity_ids:
            if entity_id in entity_owners:
                raise ValueError(
                    f"entity {entity_id} is introduced by multiple tasks"
                )
            entity_owners[entity_id] = task.task_id
        if task.incoming_edge_id is not None:
            if task.incoming_edge_id in edge_owners:
                raise ValueError(
                    f"edge {task.incoming_edge_id} is used by multiple tasks"
                )
            edge_owners[task.incoming_edge_id] = task.task_id
        task_by_id[task.task_id] = task


@dataclass(frozen=True, slots=True)
class QueryAst:
    """Canonical conjunctive query with one projected answer variable."""

    query_id: QueryId
    answer_variable: Variable
    clauses: tuple[AtomPattern, ...]

    def __post_init__(self) -> None:
        _require_exact_id(self.query_id, QueryId, "query_id")
        if type(self.answer_variable) is not Variable:
            raise TypeError("query answer_variable must be a Variable")
        if type(self.clauses) is not tuple or not self.clauses:
            raise ValueError("query clauses must be a nonempty tuple")
        if any(type(clause) is not AtomPattern for clause in self.clauses):
            raise TypeError("query clauses must contain AtomPattern values")
        if len(set(self.clauses)) != len(self.clauses):
            raise ValueError("query clauses cannot repeat an atom pattern")
        canonical_clauses = tuple(
            sorted(self.clauses, key=lambda clause: clause.canonical_key)
        )
        object.__setattr__(self, "clauses", canonical_clauses)
        variables = frozenset(
            variable for clause in canonical_clauses for variable in clause.variables
        )
        if self.answer_variable not in variables:
            raise ValueError("query answer variable must occur in a query clause")


@dataclass(frozen=True, slots=True)
class ProofStep:
    """One fact leaf or rule application in a topologically ordered proof."""

    atom: GroundAtom
    rule_id: RuleId | None
    premise_atom_ids: tuple[AtomId, ...]
    depth: int

    def __post_init__(self) -> None:
        if type(self.atom) is not GroundAtom:
            raise TypeError("proof-step atom must be a GroundAtom")
        if self.rule_id is not None:
            _require_exact_id(self.rule_id, RuleId, "rule_id")
        premise_ids = _validated_id_tuple(
            self.premise_atom_ids,
            AtomId,
            "premise_atom_ids",
            require_nonempty=self.rule_id is not None,
        )
        object.__setattr__(self, "premise_atom_ids", tuple(sorted(premise_ids)))
        if type(self.depth) is not int or not 0 <= self.depth <= 2:
            raise ValueError("proof-step depth must be an integer from zero to two")
        if self.rule_id is None and (self.depth != 0 or premise_ids):
            raise ValueError("fact proof steps have depth zero and no premises")
        if self.rule_id is not None and self.depth == 0:
            raise ValueError("derived proof steps must have positive depth")


@dataclass(frozen=True, slots=True)
class Proof:
    """A complete depth-limited proof ending in its declared conclusion."""

    proof_id: ProofId
    conclusion_atom_id: AtomId
    steps: tuple[ProofStep, ...]

    def __post_init__(self) -> None:
        _require_exact_id(self.proof_id, ProofId, "proof_id")
        _require_exact_id(self.conclusion_atom_id, AtomId, "conclusion_atom_id")
        if type(self.steps) is not tuple or not self.steps:
            raise ValueError("proof steps must be a nonempty tuple")
        if any(type(step) is not ProofStep for step in self.steps):
            raise TypeError("proof steps must contain ProofStep values")
        if self.steps[-1].atom.atom_id != self.conclusion_atom_id:
            raise ValueError("the final proof step must be the conclusion")
        depth_by_atom: dict[AtomId, int] = {}
        semantic_keys: set[tuple[str, tuple[str, ...]]] = set()
        for step in self.steps:
            atom_id = step.atom.atom_id
            if atom_id in depth_by_atom or step.atom.semantic_key in semantic_keys:
                raise ValueError("proof steps must have unique atoms")
            missing = tuple(
                premise
                for premise in step.premise_atom_ids
                if premise not in depth_by_atom
            )
            if missing:
                raise ValueError(
                    f"proof premises must precede their conclusion: {missing}"
                )
            expected_depth = (
                0
                if step.rule_id is None
                else 1
                + max(depth_by_atom[premise] for premise in step.premise_atom_ids)
            )
            if step.depth != expected_depth:
                raise ValueError("proof-step depth must equal one plus premise depth")
            depth_by_atom[atom_id] = step.depth
            semantic_keys.add(step.atom.semantic_key)

    @property
    def conclusion(self) -> GroundAtom:
        """Return the proof's final ground atom."""
        return self.steps[-1].atom

    @property
    def depth(self) -> int:
        """Return the derivation depth of the conclusion."""
        return self.steps[-1].depth

    @property
    def supporting_fact_ids(self) -> tuple[AtomId, ...]:
        """Return canonical IDs of all direct facts used by the proof."""
        return tuple(
            sorted(step.atom.atom_id for step in self.steps if step.rule_id is None)
        )

    @property
    def supporting_rule_ids(self) -> tuple[RuleId, ...]:
        """Return canonical distinct IDs of all applied rules."""
        return tuple(
            sorted(
                {
                    step.rule_id
                    for step in self.steps
                    if step.rule_id is not None
                }
            )
        )


class DataSplit(str, Enum):
    """Symbolic split assigned before any natural-language rendering."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class QueryKind(str, Enum):
    """Required TinyWorlds semantic query families."""

    DIRECT = "direct"
    ANCESTOR_PLUS_CHILD = "ancestor_plus_child"
    NEW_INSTANCE = "new_instance"
    ONE_HOP = "one_hop"
    TWO_HOP = "two_hop"
    REVISION_SENSITIVE = "revision_sensitive"
    CROSS_BRANCH = "cross_branch"
    OPEN_BOOK = "open_book"


class CandidateRole(str, Enum):
    """Correct-answer or prioritized distractor provenance."""

    CORRECT = "correct"
    INCOMPATIBLE_REVISION = "incompatible_revision"
    COMPETING_TASK = "competing_task"
    PARTIAL_PROOF = "partial_proof"
    SAME_TYPE_FILLER = "same_type_filler"


@dataclass(frozen=True, slots=True)
class HoldoutMetadata:
    """All symbolic axes that must remain disjoint across data splits."""

    template_family_id: str
    plot_id: str
    query_phrasing_id: str
    entity_combination_id: str
    proof_chain_id: str
    symbolic_text_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("template_family_id", self.template_family_id),
            ("plot_id", self.plot_id),
            ("query_phrasing_id", self.query_phrasing_id),
            ("entity_combination_id", self.entity_combination_id),
            ("proof_chain_id", self.proof_chain_id),
        ):
            if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{label} must be a canonical symbolic ID")
        if (
            type(self.symbolic_text_sha256) is not str
            or _SHA256_PATTERN.fullmatch(self.symbolic_text_sha256) is None
        ):
            raise ValueError(
                "symbolic_text_sha256 must be a lowercase hexadecimal SHA-256"
            )


@dataclass(frozen=True, slots=True)
class StoryPlan:
    """One pre-rendering story plan containing only authoritative direct IDs."""

    story_id: StoryId
    task_id: TaskId
    split: DataSplit
    direct_fact_ids: tuple[AtomId, ...]
    rule_ids: tuple[RuleId, ...]
    holdout: HoldoutMetadata

    def __post_init__(self) -> None:
        _require_exact_id(self.story_id, StoryId, "story_id")
        _require_exact_id(self.task_id, TaskId, "task_id")
        if type(self.split) is not DataSplit:
            raise TypeError("story split must be a DataSplit")
        facts = _validated_id_tuple(
            self.direct_fact_ids,
            AtomId,
            "direct_fact_ids",
            require_nonempty=True,
        )
        rules = _validated_id_tuple(
            self.rule_ids,
            RuleId,
            "rule_ids",
            require_nonempty=False,
        )
        if type(self.holdout) is not HoldoutMetadata:
            raise TypeError("story holdout must be HoldoutMetadata")
        object.__setattr__(self, "direct_fact_ids", tuple(sorted(facts)))
        object.__setattr__(self, "rule_ids", tuple(sorted(rules)))


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    """One same-type answer candidate with explicit selection provenance."""

    entity_id: EntityId
    role: CandidateRole

    def __post_init__(self) -> None:
        _require_exact_id(self.entity_id, EntityId, "entity_id")
        if type(self.role) is not CandidateRole:
            raise TypeError("candidate role must be a CandidateRole")


@dataclass(frozen=True, slots=True)
class ProofMetadata:
    """Canonical proof identity plus exact task and edge support."""

    proof_id: ProofId
    conclusion_atom_id: AtomId
    supporting_fact_ids: tuple[AtomId, ...]
    supporting_rule_ids: tuple[RuleId, ...]
    required_task_ids: tuple[TaskId, ...]
    required_edge_ids: tuple[TaskEdgeId, ...]
    depth: int

    def __post_init__(self) -> None:
        _require_exact_id(self.proof_id, ProofId, "proof_id")
        _require_exact_id(
            self.conclusion_atom_id,
            AtomId,
            "conclusion_atom_id",
        )
        fields = (
            (
                "supporting_fact_ids",
                self.supporting_fact_ids,
                AtomId,
                True,
            ),
            (
                "supporting_rule_ids",
                self.supporting_rule_ids,
                RuleId,
                False,
            ),
            ("required_task_ids", self.required_task_ids, TaskId, True),
            ("required_edge_ids", self.required_edge_ids, TaskEdgeId, True),
        )
        for label, values, expected_type, require_nonempty in fields:
            normalized = _validated_id_tuple(
                values,
                expected_type,
                label,
                require_nonempty=require_nonempty,
            )
            object.__setattr__(self, label, tuple(normalized))
        if type(self.depth) is not int or not 0 <= self.depth <= 2:
            raise ValueError("proof metadata depth must be from zero to two")


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """One exact four-candidate semantic query fixed before rendering."""

    task_id: TaskId
    split: DataSplit
    kind: QueryKind
    query_ast: QueryAst
    answer_entity_id: EntityId
    candidates: tuple[QueryCandidate, ...]
    correct_index: int
    proof: ProofMetadata
    hard_oracle_task_ids: tuple[TaskId, ...]
    open_book_fact_ids: tuple[AtomId, ...]
    holdout: HoldoutMetadata

    def __post_init__(self) -> None:
        _require_exact_id(self.task_id, TaskId, "task_id")
        if type(self.split) is not DataSplit or self.split is DataSplit.TRAIN:
            raise ValueError("query split must be validation or test")
        if type(self.kind) is not QueryKind:
            raise TypeError("query kind must be a QueryKind")
        if type(self.query_ast) is not QueryAst:
            raise TypeError("query_ast must be a QueryAst")
        _require_exact_id(
            self.answer_entity_id,
            EntityId,
            "answer_entity_id",
        )
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) != 4
            or any(type(candidate) is not QueryCandidate for candidate in self.candidates)
        ):
            raise ValueError("queries must contain exactly four QueryCandidate values")
        candidate_ids = tuple(candidate.entity_id for candidate in self.candidates)
        if len(set(candidate_ids)) != 4:
            raise ValueError("query candidate entities must be unique")
        if type(self.correct_index) is not int or not 0 <= self.correct_index < 4:
            raise ValueError("correct_index must be from zero to three")
        if candidate_ids[self.correct_index] != self.answer_entity_id:
            raise ValueError("correct_index must identify answer_entity_id")
        correct_roles = tuple(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.role is CandidateRole.CORRECT
        )
        if correct_roles != (self.correct_index,):
            raise ValueError("exactly the indexed candidate must have correct role")
        if type(self.proof) is not ProofMetadata:
            raise TypeError("query proof must be ProofMetadata")
        oracle_ids = _validated_id_tuple(
            self.hard_oracle_task_ids,
            TaskId,
            "hard_oracle_task_ids",
            require_nonempty=False,
        )
        if self.kind is QueryKind.CROSS_BRANCH and oracle_ids:
            raise ValueError("cross-branch queries have no complete hard oracle")
        if self.kind is not QueryKind.CROSS_BRANCH and len(oracle_ids) != 1:
            raise ValueError("non-cross-branch queries require one hard oracle task")
        open_book_ids = _validated_id_tuple(
            self.open_book_fact_ids,
            AtomId,
            "open_book_fact_ids",
            require_nonempty=self.kind is QueryKind.OPEN_BOOK,
        )
        if self.kind is not QueryKind.OPEN_BOOK and open_book_ids:
            raise ValueError("only open-book queries may expose supporting facts")
        if type(self.holdout) is not HoldoutMetadata:
            raise TypeError("query holdout must be HoldoutMetadata")
        object.__setattr__(self, "hard_oracle_task_ids", oracle_ids)
        object.__setattr__(self, "open_book_fact_ids", open_book_ids)


def _require_exact_id(value: object, expected_type: type[_TypedId], label: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(f"{label} must be a {expected_type.__name__}")


def _validated_id_tuple(
    values: object,
    expected_type: type[_TypedId],
    label: str,
    *,
    require_nonempty: bool,
) -> tuple[_TypedId, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    if require_nonempty and not values:
        raise ValueError(f"{label} must not be empty")
    if any(type(value) is not expected_type for value in values):
        raise TypeError(f"{label} must contain {expected_type.__name__} values")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must contain unique IDs")
    return values


def _validate_surface_form(value: object, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty without outer whitespace")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{label} must use Unicode NFC normalization")


__all__ = [
    "AtomId",
    "AtomPattern",
    "CandidateRole",
    "DataSplit",
    "Entity",
    "EntityId",
    "EntityTypeId",
    "FamilyId",
    "GroundAtom",
    "HornRule",
    "HoldoutMetadata",
    "PredicateId",
    "Proof",
    "ProofId",
    "ProofStep",
    "QueryAst",
    "QueryCandidate",
    "QueryId",
    "QueryKind",
    "QueryPlan",
    "RuleId",
    "TaskId",
    "TaskEdgeId",
    "TaskKind",
    "TaskSpecification",
    "Term",
    "Variable",
    "ProofMetadata",
    "StoryId",
    "StoryPlan",
    "validate_task_specifications",
]
