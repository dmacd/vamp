"""Canonical, checksummed persistence for symbolic TinyWorlds bundles.

The JSON records in this module are deliberately dependency-free and strict.
They are the durable boundary between symbolic generation and later rendering:
unknown fields, non-canonical JSON, unlisted files, digest mismatches, and
dangling symbolic references are all rejected at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, TypeVar

from apm.data.text.tinyworlds.closure import ClosureResult
from apm.data.text.tinyworlds.ontology import (
    PredicateKind,
    PredicateRegistry,
    PredicateSignature,
)
from apm.data.text.tinyworlds.query_generation import TinyWorldsBundle
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
    HoldoutMetadata,
    HornRule,
    PredicateId,
    Proof,
    ProofId,
    ProofMetadata,
    ProofStep,
    QueryAst,
    QueryCandidate,
    QueryId,
    QueryKind,
    QueryPlan,
    RuleId,
    StoryId,
    StoryPlan,
    TaskEdgeId,
    TaskId,
    TaskKind,
    TaskSpecification,
    Variable,
)
from apm.data.text.tinyworlds.world_generation import (
    RevisionRecord,
    SymbolicWorld,
)


_FORMAT = "apm.tinyworlds.symbolic-bundle"
_SCHEMA_VERSION = 1
_ARTIFACT_PATHS = (
    "ontology.json",
    "entities.jsonl",
    "facts.jsonl",
    "rules.jsonl",
    "tasks.jsonl",
    "revisions.jsonl",
    "stories.jsonl",
    "queries.jsonl",
    "proofs.jsonl",
    "knowledge.metta",
)
_SHA256_LENGTH = 64


class TinyWorldsBundleError(ValueError):
    """A persisted TinyWorlds bundle is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class BundleFileDigest:
    """Integrity metadata for one canonical bundle artifact."""

    path: str
    sha256: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        if type(self.path) is not str or self.path not in _ARTIFACT_PATHS:
            raise TinyWorldsBundleError(f"unknown bundle artifact path: {self.path!r}")
        _require_digest(self.sha256, f"artifact {self.path} SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise TinyWorldsBundleError("artifact size_bytes must be nonnegative")
        if type(self.record_count) is not int or self.record_count < 0:
            raise TinyWorldsBundleError("artifact record_count must be nonnegative")


@dataclass(frozen=True, slots=True)
class TinyWorldsBundleManifest:
    """Self-authenticating manifest for one immutable symbolic bundle tree."""

    format: str
    schema_version: int
    bundle_id: str
    version: str
    world_id: str
    master_seed_sha256: str
    artifacts: tuple[BundleFileDigest, ...]
    bundle_sha256: str

    def __post_init__(self) -> None:
        if self.format != _FORMAT:
            raise TinyWorldsBundleError(f"unsupported bundle format: {self.format!r}")
        if type(self.schema_version) is not int or self.schema_version != _SCHEMA_VERSION:
            raise TinyWorldsBundleError(
                f"unsupported TinyWorlds schema version: {self.schema_version!r}"
            )
        for label, value in (
            ("bundle_id", self.bundle_id),
            ("version", self.version),
            ("world_id", self.world_id),
        ):
            if type(value) is not str or not value:
                raise TinyWorldsBundleError(f"manifest {label} must be nonempty")
        _require_digest(self.master_seed_sha256, "manifest master seed")
        if type(self.artifacts) is not tuple or any(
            type(item) is not BundleFileDigest for item in self.artifacts
        ):
            raise TinyWorldsBundleError("manifest artifacts must be file digests")
        paths = tuple(item.path for item in self.artifacts)
        if paths != _ARTIFACT_PATHS:
            raise TinyWorldsBundleError(
                "manifest must list the exact canonical artifact set and order"
            )
        _require_digest(self.bundle_sha256, "manifest bundle SHA-256")
        expected = _digest_bytes(_canonical_json_bytes(_manifest_core(self)))
        if self.bundle_sha256 != expected:
            raise TinyWorldsBundleError("manifest bundle SHA-256 mismatch")


def write_tinyworlds_bundle(
    bundle: TinyWorldsBundle,
    directory: str | Path,
) -> TinyWorldsBundleManifest:
    """Atomically write a new immutable canonical bundle directory.

    Existing targets are never overwritten.  All artifact bytes are finalized
    and hashed in a sibling temporary directory before the directory rename.
    """
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"TinyWorlds bundle target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        payloads, record_counts = _bundle_payloads(bundle)
        for relative_path in _ARTIFACT_PATHS:
            payload = payloads[relative_path]
            (temporary / relative_path).write_bytes(payload)
        manifest = _manifest_for_payloads(bundle, payloads, record_counts)
        (temporary / "manifest.json").write_bytes(
            _canonical_json_bytes(_manifest_record(manifest))
        )
        os.rename(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def tinyworlds_bundle_sha256(bundle: TinyWorldsBundle) -> str:
    """Return the exact manifest digest without writing a symbolic bundle tree."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    payloads, record_counts = _bundle_payloads(bundle)
    return _manifest_for_payloads(bundle, payloads, record_counts).bundle_sha256


def load_tinyworlds_manifest(
    directory: str | Path,
) -> TinyWorldsBundleManifest:
    """Load and validate only a bundle's strict canonical manifest."""
    root = Path(directory)
    manifest_record = _read_canonical_json(root / "manifest.json")
    manifest = _decode_manifest(manifest_record)
    actual_paths = tuple(
        sorted(path.name for path in root.iterdir() if path.name != "manifest.json")
    )
    if actual_paths != tuple(sorted(_ARTIFACT_PATHS)):
        raise TinyWorldsBundleError(
            "bundle directory contains missing or unlisted artifact files"
        )
    for artifact in manifest.artifacts:
        path = root / artifact.path
        if path.is_symlink() or not path.is_file():
            raise TinyWorldsBundleError(
                f"bundle artifact must be a regular file: {artifact.path}"
            )
        payload = path.read_bytes()
        if len(payload) != artifact.size_bytes:
            raise TinyWorldsBundleError(
                f"artifact size mismatch: {artifact.path}"
            )
        if _digest_bytes(payload) != artifact.sha256:
            raise TinyWorldsBundleError(
                f"artifact digest mismatch: {artifact.path}"
            )
    return manifest


def load_tinyworlds_bundle(directory: str | Path) -> TinyWorldsBundle:
    """Strictly load, revalidate, and mechanically reconstruct a bundle."""
    root = Path(directory)
    manifest = load_tinyworlds_manifest(root)
    artifact_by_path = {item.path: item for item in manifest.artifacts}

    ontology_record = _read_canonical_json(root / "ontology.json")
    _require_record_count(artifact_by_path["ontology.json"], 1)
    registry = _decode_registry(ontology_record)
    entities = _load_jsonl(
        root / "entities.jsonl",
        artifact_by_path["entities.jsonl"],
        _decode_entity,
    )
    facts = _load_jsonl(
        root / "facts.jsonl",
        artifact_by_path["facts.jsonl"],
        _decode_atom,
    )
    rules = _load_jsonl(
        root / "rules.jsonl",
        artifact_by_path["rules.jsonl"],
        _decode_rule,
    )
    tasks = _load_jsonl(
        root / "tasks.jsonl",
        artifact_by_path["tasks.jsonl"],
        _decode_task,
    )
    revisions = _load_jsonl(
        root / "revisions.jsonl",
        artifact_by_path["revisions.jsonl"],
        _decode_revision,
    )
    stories = _load_jsonl(
        root / "stories.jsonl",
        artifact_by_path["stories.jsonl"],
        _decode_story,
    )
    queries = _load_jsonl(
        root / "queries.jsonl",
        artifact_by_path["queries.jsonl"],
        _decode_query,
    )
    proofs = _load_jsonl(
        root / "proofs.jsonl",
        artifact_by_path["proofs.jsonl"],
        _decode_proof,
    )
    metta_payload = (root / "knowledge.metta").read_bytes()
    _require_record_count(
        artifact_by_path["knowledge.metta"],
        len(metta_payload.splitlines()),
    )
    if not proofs:
        raise TinyWorldsBundleError("proofs.jsonl must not be empty")
    closure = ClosureResult(
        atoms=tuple(proof.conclusion for proof in proofs),
        proofs=proofs,
    )
    try:
        world = SymbolicWorld(
            world_id=manifest.world_id,
            master_seed_sha256=manifest.master_seed_sha256,
            registry=registry,
            entities=entities,
            facts=facts,
            rules=rules,
            tasks=tasks,
            closure=closure,
            revisions=revisions,
        )
        bundle = TinyWorldsBundle(
            bundle_id=manifest.bundle_id,
            version=manifest.version,
            world=world,
            story_plans=stories,
            query_plans=queries,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TinyWorldsBundleError(
            f"bundle symbolic validation failed: {error}"
        ) from error
    expected_metta = export_metta_atomese(bundle)
    if metta_payload != expected_metta:
        raise TinyWorldsBundleError(
            "knowledge.metta does not match the authoritative symbolic records"
        )
    return bundle


def export_metta_atomese(bundle: TinyWorldsBundle) -> bytes:
    """Return a deterministic dependency-free MeTTa/Atomese S-expression export."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    world = bundle.world
    lines = [
        ";; Canonical TinyWorlds MeTTa/Atomese text export",
        f"(TinyWorldsBundle {_quoted(bundle.bundle_id)} {_quoted(bundle.version)})",
    ]
    lines.extend(
        f"(EntityType {_quoted(str(type_id))})"
        for type_id in world.registry.entity_types
    )
    lines.extend(
        "(PredicateSignature "
        f"{_quoted(str(item.predicate_id))} {_quoted(item.kind.value)} "
        f"({_space_quoted(item.argument_types)}))"
        for item in world.registry.predicates
    )
    lines.extend(
        "(Entity "
        f"{_quoted(str(entity.entity_id))} "
        f"{_quoted(str(entity.entity_type))} {_quoted(entity.name)} "
        f"({_space_quoted(entity.inflections)}))"
        for entity in world.entities
    )
    lines.extend(
        f"(Fact {_quoted(str(atom.atom_id))} {_metta_atom(atom)})"
        for atom in world.facts
    )
    lines.extend(
        "(Rule "
        f"{_quoted(str(rule.rule_id))} "
        f"(Head {_metta_pattern(rule.head)}) "
        f"(Body {' '.join(_metta_pattern(item) for item in rule.body)}))"
        for rule in world.rules
    )
    lines.extend(
        "(Task "
        f"{_quoted(str(task.task_id))} {_quoted(str(task.family_id))} "
        f"{_quoted(task.kind.value)} "
        f"{_quoted(str(task.parent_task_id)) if task.parent_task_id else 'Root'} "
        f"{_quoted(str(task.incoming_edge_id)) if task.incoming_edge_id else 'RootEdge'})"
        for task in world.tasks
    )
    lines.extend(
        "(Query "
        f"{_quoted(str(plan.query_ast.query_id))} "
        f"{_quoted(str(plan.task_id))} {_quoted(plan.split.value)} "
        f"{_quoted(plan.kind.value)} "
        f"(Clauses {' '.join(_metta_pattern(item) for item in plan.query_ast.clauses)}) "
        f"(Answer {_quoted(str(plan.answer_entity_id))}))"
        for plan in bundle.query_plans
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bundle_payloads(
    bundle: TinyWorldsBundle,
) -> tuple[dict[str, bytes], dict[str, int]]:
    world = bundle.world
    records: dict[str, tuple[dict[str, object], ...]] = {
        "entities.jsonl": tuple(_encode_entity(item) for item in world.entities),
        "facts.jsonl": tuple(_encode_atom(item) for item in world.facts),
        "rules.jsonl": tuple(_encode_rule(item) for item in world.rules),
        "tasks.jsonl": tuple(_encode_task(item) for item in world.tasks),
        "revisions.jsonl": tuple(
            _encode_revision(item) for item in world.revisions
        ),
        "stories.jsonl": tuple(
            _encode_story(item) for item in bundle.story_plans
        ),
        "queries.jsonl": tuple(
            _encode_query(item) for item in bundle.query_plans
        ),
        "proofs.jsonl": tuple(
            _encode_proof(item) for item in world.closure.proofs
        ),
    }
    payloads = {
        path: _canonical_jsonl_bytes(values) for path, values in records.items()
    }
    payloads["ontology.json"] = _canonical_json_bytes(
        _encode_registry(world.registry)
    )
    payloads["knowledge.metta"] = export_metta_atomese(bundle)
    counts = {path: len(values) for path, values in records.items()}
    counts["ontology.json"] = 1
    counts["knowledge.metta"] = len(payloads["knowledge.metta"].splitlines())
    return payloads, counts


def _manifest_for_payloads(
    bundle: TinyWorldsBundle,
    payloads: dict[str, bytes],
    record_counts: dict[str, int],
) -> TinyWorldsBundleManifest:
    artifacts = tuple(
        BundleFileDigest(
            path=path,
            sha256=_digest_bytes(payloads[path]),
            size_bytes=len(payloads[path]),
            record_count=record_counts[path],
        )
        for path in _ARTIFACT_PATHS
    )
    core = _manifest_core_values(
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        world_id=bundle.world.world_id,
        master_seed_sha256=bundle.world.master_seed_sha256,
        artifacts=artifacts,
    )
    return TinyWorldsBundleManifest(
        format=_FORMAT,
        schema_version=_SCHEMA_VERSION,
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        world_id=bundle.world.world_id,
        master_seed_sha256=bundle.world.master_seed_sha256,
        artifacts=artifacts,
        bundle_sha256=_digest_bytes(_canonical_json_bytes(core)),
    )


def _encode_registry(registry: PredicateRegistry) -> dict[str, object]:
    return {
        "context_type_id": str(registry.context_type_id),
        "entity_types": [str(item) for item in registry.entity_types],
        "predicates": [
            {
                "argument_types": [str(item) for item in predicate.argument_types],
                "kind": predicate.kind.value,
                "predicate_id": str(predicate.predicate_id),
            }
            for predicate in registry.predicates
        ],
    }


def _decode_registry(record: object) -> PredicateRegistry:
    data = _fields(
        record,
        ("context_type_id", "entity_types", "predicates"),
        "ontology",
    )
    predicate_records = _list(data["predicates"], "ontology predicates")
    predicates = tuple(
        _decode_predicate_signature(item) for item in predicate_records
    )
    return PredicateRegistry(
        entity_types=tuple(
            EntityTypeId(_string(item, "entity type"))
            for item in _list(data["entity_types"], "entity_types")
        ),
        context_type_id=EntityTypeId(
            _string(data["context_type_id"], "context_type_id")
        ),
        predicates=predicates,
    )


def _decode_predicate_signature(record: object) -> PredicateSignature:
    data = _fields(
        record,
        ("argument_types", "kind", "predicate_id"),
        "predicate signature",
    )
    return PredicateSignature(
        predicate_id=PredicateId(_string(data["predicate_id"], "predicate_id")),
        argument_types=tuple(
            EntityTypeId(_string(item, "predicate argument type"))
            for item in _list(data["argument_types"], "argument_types")
        ),
        kind=PredicateKind(_string(data["kind"], "predicate kind")),
    )


def _encode_entity(entity: Entity) -> dict[str, object]:
    return {
        "entity_id": str(entity.entity_id),
        "entity_type": str(entity.entity_type),
        "inflections": list(entity.inflections),
        "name": entity.name,
    }


def _decode_entity(record: object) -> Entity:
    data = _fields(
        record,
        ("entity_id", "entity_type", "inflections", "name"),
        "entity",
    )
    return Entity(
        entity_id=EntityId(_string(data["entity_id"], "entity_id")),
        entity_type=EntityTypeId(_string(data["entity_type"], "entity_type")),
        name=_string(data["name"], "entity name"),
        inflections=tuple(
            _string(item, "entity inflection")
            for item in _list(data["inflections"], "inflections")
        ),
    )


def _encode_atom(atom: GroundAtom) -> dict[str, object]:
    return {
        "arguments": [str(item) for item in atom.arguments],
        "atom_id": str(atom.atom_id),
        "predicate_id": str(atom.predicate_id),
    }


def _decode_atom(record: object) -> GroundAtom:
    data = _fields(record, ("arguments", "atom_id", "predicate_id"), "atom")
    return GroundAtom(
        atom_id=AtomId(_string(data["atom_id"], "atom_id")),
        predicate_id=PredicateId(_string(data["predicate_id"], "predicate_id")),
        arguments=tuple(
            EntityId(_string(item, "atom argument"))
            for item in _list(data["arguments"], "atom arguments")
        ),
    )


def _encode_pattern(pattern: AtomPattern) -> dict[str, object]:
    return {
        "arguments": [
            {
                "kind": "entity" if type(item) is EntityId else "variable",
                "value": str(item) if type(item) is EntityId else item.name,
            }
            for item in pattern.arguments
        ],
        "predicate_id": str(pattern.predicate_id),
    }


def _decode_pattern(record: object) -> AtomPattern:
    data = _fields(record, ("arguments", "predicate_id"), "atom pattern")
    terms = tuple(
        _decode_term(item)
        for item in _list(data["arguments"], "pattern arguments")
    )
    return AtomPattern(
        predicate_id=PredicateId(_string(data["predicate_id"], "predicate_id")),
        arguments=terms,
    )


def _decode_term(record: object) -> EntityId | Variable:
    data = _fields(record, ("kind", "value"), "pattern term")
    kind = _string(data["kind"], "term kind")
    value = _string(data["value"], "term value")
    if kind == "entity":
        return EntityId(value)
    if kind == "variable":
        return Variable(value)
    raise TinyWorldsBundleError(f"unknown pattern term kind: {kind!r}")


def _encode_rule(rule: HornRule) -> dict[str, object]:
    return {
        "body": [_encode_pattern(item) for item in rule.body],
        "head": _encode_pattern(rule.head),
        "rule_id": str(rule.rule_id),
    }


def _decode_rule(record: object) -> HornRule:
    data = _fields(record, ("body", "head", "rule_id"), "rule")
    return HornRule(
        rule_id=RuleId(_string(data["rule_id"], "rule_id")),
        head=_decode_pattern(data["head"]),
        body=tuple(
            _decode_pattern(item)
            for item in _list(data["body"], "rule body")
        ),
    )


def _encode_task(task: TaskSpecification) -> dict[str, object]:
    return {
        "direct_fact_ids": [str(item) for item in task.direct_fact_ids],
        "family_id": str(task.family_id),
        "incoming_edge_id": (
            None if task.incoming_edge_id is None else str(task.incoming_edge_id)
        ),
        "introduced_entity_ids": [
            str(item) for item in task.introduced_entity_ids
        ],
        "kind": task.kind.value,
        "parent_task_id": (
            None if task.parent_task_id is None else str(task.parent_task_id)
        ),
        "rule_ids": [str(item) for item in task.rule_ids],
        "task_id": str(task.task_id),
    }


def _decode_task(record: object) -> TaskSpecification:
    data = _fields(
        record,
        (
            "direct_fact_ids",
            "family_id",
            "incoming_edge_id",
            "introduced_entity_ids",
            "kind",
            "parent_task_id",
            "rule_ids",
            "task_id",
        ),
        "task",
    )
    parent = _optional_string(data["parent_task_id"], "parent_task_id")
    edge = _optional_string(data["incoming_edge_id"], "incoming_edge_id")
    return TaskSpecification(
        task_id=TaskId(_string(data["task_id"], "task_id")),
        family_id=FamilyId(_string(data["family_id"], "family_id")),
        kind=TaskKind(_string(data["kind"], "task kind")),
        parent_task_id=None if parent is None else TaskId(parent),
        direct_fact_ids=tuple(
            AtomId(_string(item, "task fact ID"))
            for item in _list(data["direct_fact_ids"], "direct_fact_ids")
        ),
        rule_ids=tuple(
            RuleId(_string(item, "task rule ID"))
            for item in _list(data["rule_ids"], "rule_ids")
        ),
        introduced_entity_ids=tuple(
            EntityId(_string(item, "introduced entity ID"))
            for item in _list(
                data["introduced_entity_ids"], "introduced_entity_ids"
            )
        ),
        incoming_edge_id=None if edge is None else TaskEdgeId(edge),
    )


def _encode_revision(record: RevisionRecord) -> dict[str, object]:
    return {
        "base_atom_id": str(record.base_atom_id),
        "base_value_entity_id": str(record.base_value_entity_id),
        "context_entity_id": str(record.context_entity_id),
        "contextual_atom_id": str(record.contextual_atom_id),
        "family_id": str(record.family_id),
        "revised_value_entity_id": str(record.revised_value_entity_id),
        "subject_entity_id": str(record.subject_entity_id),
    }


def _decode_revision(record: object) -> RevisionRecord:
    keys = (
        "base_atom_id",
        "base_value_entity_id",
        "context_entity_id",
        "contextual_atom_id",
        "family_id",
        "revised_value_entity_id",
        "subject_entity_id",
    )
    data = _fields(record, keys, "revision")
    return RevisionRecord(
        family_id=FamilyId(_string(data["family_id"], "family_id")),
        base_atom_id=AtomId(_string(data["base_atom_id"], "base_atom_id")),
        contextual_atom_id=AtomId(
            _string(data["contextual_atom_id"], "contextual_atom_id")
        ),
        subject_entity_id=EntityId(
            _string(data["subject_entity_id"], "subject_entity_id")
        ),
        base_value_entity_id=EntityId(
            _string(data["base_value_entity_id"], "base_value_entity_id")
        ),
        revised_value_entity_id=EntityId(
            _string(data["revised_value_entity_id"], "revised_value_entity_id")
        ),
        context_entity_id=EntityId(
            _string(data["context_entity_id"], "context_entity_id")
        ),
    )


def _encode_holdout(value: HoldoutMetadata) -> dict[str, object]:
    return {
        "entity_combination_id": value.entity_combination_id,
        "plot_id": value.plot_id,
        "proof_chain_id": value.proof_chain_id,
        "query_phrasing_id": value.query_phrasing_id,
        "symbolic_text_sha256": value.symbolic_text_sha256,
        "template_family_id": value.template_family_id,
    }


def _decode_holdout(record: object) -> HoldoutMetadata:
    keys = (
        "entity_combination_id",
        "plot_id",
        "proof_chain_id",
        "query_phrasing_id",
        "symbolic_text_sha256",
        "template_family_id",
    )
    data = _fields(record, keys, "holdout metadata")
    return HoldoutMetadata(
        template_family_id=_string(
            data["template_family_id"], "template_family_id"
        ),
        plot_id=_string(data["plot_id"], "plot_id"),
        query_phrasing_id=_string(
            data["query_phrasing_id"], "query_phrasing_id"
        ),
        entity_combination_id=_string(
            data["entity_combination_id"], "entity_combination_id"
        ),
        proof_chain_id=_string(data["proof_chain_id"], "proof_chain_id"),
        symbolic_text_sha256=_string(
            data["symbolic_text_sha256"], "symbolic_text_sha256"
        ),
    )


def _encode_story(story: StoryPlan) -> dict[str, object]:
    return {
        "direct_fact_ids": [str(item) for item in story.direct_fact_ids],
        "holdout": _encode_holdout(story.holdout),
        "rule_ids": [str(item) for item in story.rule_ids],
        "split": story.split.value,
        "story_id": str(story.story_id),
        "task_id": str(story.task_id),
    }


def _decode_story(record: object) -> StoryPlan:
    data = _fields(
        record,
        ("direct_fact_ids", "holdout", "rule_ids", "split", "story_id", "task_id"),
        "story",
    )
    return StoryPlan(
        story_id=StoryId(_string(data["story_id"], "story_id")),
        task_id=TaskId(_string(data["task_id"], "task_id")),
        split=DataSplit(_string(data["split"], "story split")),
        direct_fact_ids=tuple(
            AtomId(_string(item, "story fact ID"))
            for item in _list(data["direct_fact_ids"], "story direct_fact_ids")
        ),
        rule_ids=tuple(
            RuleId(_string(item, "story rule ID"))
            for item in _list(data["rule_ids"], "story rule_ids")
        ),
        holdout=_decode_holdout(data["holdout"]),
    )


def _encode_query_ast(query: QueryAst) -> dict[str, object]:
    return {
        "answer_variable": query.answer_variable.name,
        "clauses": [_encode_pattern(item) for item in query.clauses],
        "query_id": str(query.query_id),
    }


def _decode_query_ast(record: object) -> QueryAst:
    data = _fields(
        record, ("answer_variable", "clauses", "query_id"), "query AST"
    )
    return QueryAst(
        query_id=QueryId(_string(data["query_id"], "query_id")),
        answer_variable=Variable(
            _string(data["answer_variable"], "answer_variable")
        ),
        clauses=tuple(
            _decode_pattern(item)
            for item in _list(data["clauses"], "query clauses")
        ),
    )


def _encode_proof_metadata(proof: ProofMetadata) -> dict[str, object]:
    return {
        "conclusion_atom_id": str(proof.conclusion_atom_id),
        "depth": proof.depth,
        "proof_id": str(proof.proof_id),
        "required_edge_ids": [str(item) for item in proof.required_edge_ids],
        "required_task_ids": [str(item) for item in proof.required_task_ids],
        "supporting_fact_ids": [str(item) for item in proof.supporting_fact_ids],
        "supporting_rule_ids": [str(item) for item in proof.supporting_rule_ids],
    }


def _decode_proof_metadata(record: object) -> ProofMetadata:
    keys = (
        "conclusion_atom_id",
        "depth",
        "proof_id",
        "required_edge_ids",
        "required_task_ids",
        "supporting_fact_ids",
        "supporting_rule_ids",
    )
    data = _fields(record, keys, "proof metadata")
    return ProofMetadata(
        proof_id=ProofId(_string(data["proof_id"], "proof_id")),
        conclusion_atom_id=AtomId(
            _string(data["conclusion_atom_id"], "conclusion_atom_id")
        ),
        supporting_fact_ids=tuple(
            AtomId(_string(item, "supporting fact ID"))
            for item in _list(data["supporting_fact_ids"], "supporting_fact_ids")
        ),
        supporting_rule_ids=tuple(
            RuleId(_string(item, "supporting rule ID"))
            for item in _list(data["supporting_rule_ids"], "supporting_rule_ids")
        ),
        required_task_ids=tuple(
            TaskId(_string(item, "required task ID"))
            for item in _list(data["required_task_ids"], "required_task_ids")
        ),
        required_edge_ids=tuple(
            TaskEdgeId(_string(item, "required edge ID"))
            for item in _list(data["required_edge_ids"], "required_edge_ids")
        ),
        depth=_integer(data["depth"], "proof depth"),
    )


def _encode_query(query: QueryPlan) -> dict[str, object]:
    return {
        "answer_entity_id": str(query.answer_entity_id),
        "candidates": [
            {"entity_id": str(item.entity_id), "role": item.role.value}
            for item in query.candidates
        ],
        "correct_index": query.correct_index,
        "hard_oracle_task_ids": [str(item) for item in query.hard_oracle_task_ids],
        "holdout": _encode_holdout(query.holdout),
        "kind": query.kind.value,
        "open_book_fact_ids": [str(item) for item in query.open_book_fact_ids],
        "proof": _encode_proof_metadata(query.proof),
        "query_ast": _encode_query_ast(query.query_ast),
        "split": query.split.value,
        "task_id": str(query.task_id),
    }


def _decode_query(record: object) -> QueryPlan:
    keys = (
        "answer_entity_id",
        "candidates",
        "correct_index",
        "hard_oracle_task_ids",
        "holdout",
        "kind",
        "open_book_fact_ids",
        "proof",
        "query_ast",
        "split",
        "task_id",
    )
    data = _fields(record, keys, "query")
    candidates = tuple(
        _decode_candidate(item)
        for item in _list(data["candidates"], "query candidates")
    )
    return QueryPlan(
        task_id=TaskId(_string(data["task_id"], "task_id")),
        split=DataSplit(_string(data["split"], "query split")),
        kind=QueryKind(_string(data["kind"], "query kind")),
        query_ast=_decode_query_ast(data["query_ast"]),
        answer_entity_id=EntityId(
            _string(data["answer_entity_id"], "answer_entity_id")
        ),
        candidates=candidates,
        correct_index=_integer(data["correct_index"], "correct_index"),
        proof=_decode_proof_metadata(data["proof"]),
        hard_oracle_task_ids=tuple(
            TaskId(_string(item, "hard oracle task ID"))
            for item in _list(
                data["hard_oracle_task_ids"], "hard_oracle_task_ids"
            )
        ),
        open_book_fact_ids=tuple(
            AtomId(_string(item, "open-book fact ID"))
            for item in _list(data["open_book_fact_ids"], "open_book_fact_ids")
        ),
        holdout=_decode_holdout(data["holdout"]),
    )


def _decode_candidate(record: object) -> QueryCandidate:
    data = _fields(record, ("entity_id", "role"), "query candidate")
    return QueryCandidate(
        entity_id=EntityId(_string(data["entity_id"], "candidate entity_id")),
        role=CandidateRole(_string(data["role"], "candidate role")),
    )


def _encode_proof(proof: Proof) -> dict[str, object]:
    return {
        "conclusion_atom_id": str(proof.conclusion_atom_id),
        "proof_id": str(proof.proof_id),
        "steps": [
            {
                "atom": _encode_atom(step.atom),
                "depth": step.depth,
                "premise_atom_ids": [str(item) for item in step.premise_atom_ids],
                "rule_id": None if step.rule_id is None else str(step.rule_id),
            }
            for step in proof.steps
        ],
    }


def _decode_proof(record: object) -> Proof:
    data = _fields(
        record, ("conclusion_atom_id", "proof_id", "steps"), "proof"
    )
    return Proof(
        proof_id=ProofId(_string(data["proof_id"], "proof_id")),
        conclusion_atom_id=AtomId(
            _string(data["conclusion_atom_id"], "conclusion_atom_id")
        ),
        steps=tuple(
            _decode_proof_step(item)
            for item in _list(data["steps"], "proof steps")
        ),
    )


def _decode_proof_step(record: object) -> ProofStep:
    data = _fields(
        record,
        ("atom", "depth", "premise_atom_ids", "rule_id"),
        "proof step",
    )
    rule = _optional_string(data["rule_id"], "proof-step rule_id")
    return ProofStep(
        atom=_decode_atom(data["atom"]),
        rule_id=None if rule is None else RuleId(rule),
        premise_atom_ids=tuple(
            AtomId(_string(item, "premise atom ID"))
            for item in _list(data["premise_atom_ids"], "premise_atom_ids")
        ),
        depth=_integer(data["depth"], "proof-step depth"),
    )


def _manifest_core(manifest: TinyWorldsBundleManifest) -> dict[str, object]:
    return _manifest_core_values(
        bundle_id=manifest.bundle_id,
        version=manifest.version,
        world_id=manifest.world_id,
        master_seed_sha256=manifest.master_seed_sha256,
        artifacts=manifest.artifacts,
    )


def _manifest_core_values(
    *,
    bundle_id: str,
    version: str,
    world_id: str,
    master_seed_sha256: str,
    artifacts: tuple[BundleFileDigest, ...],
) -> dict[str, object]:
    return {
        "artifacts": [
            {
                "path": item.path,
                "record_count": item.record_count,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in artifacts
        ],
        "bundle_id": bundle_id,
        "format": _FORMAT,
        "master_seed_sha256": master_seed_sha256,
        "schema_version": _SCHEMA_VERSION,
        "version": version,
        "world_id": world_id,
    }


def _manifest_record(manifest: TinyWorldsBundleManifest) -> dict[str, object]:
    return {**_manifest_core(manifest), "bundle_sha256": manifest.bundle_sha256}


def _decode_manifest(record: object) -> TinyWorldsBundleManifest:
    keys = (
        "artifacts",
        "bundle_id",
        "bundle_sha256",
        "format",
        "master_seed_sha256",
        "schema_version",
        "version",
        "world_id",
    )
    data = _fields(record, keys, "manifest")
    artifacts = tuple(
        _decode_file_digest(item)
        for item in _list(data["artifacts"], "manifest artifacts")
    )
    return TinyWorldsBundleManifest(
        format=_string(data["format"], "manifest format"),
        schema_version=_integer(data["schema_version"], "schema_version"),
        bundle_id=_string(data["bundle_id"], "bundle_id"),
        version=_string(data["version"], "version"),
        world_id=_string(data["world_id"], "world_id"),
        master_seed_sha256=_string(
            data["master_seed_sha256"], "master_seed_sha256"
        ),
        artifacts=artifacts,
        bundle_sha256=_string(data["bundle_sha256"], "bundle_sha256"),
    )


def _decode_file_digest(record: object) -> BundleFileDigest:
    data = _fields(
        record,
        ("path", "record_count", "sha256", "size_bytes"),
        "artifact digest",
    )
    return BundleFileDigest(
        path=_string(data["path"], "artifact path"),
        sha256=_string(data["sha256"], "artifact sha256"),
        size_bytes=_integer(data["size_bytes"], "artifact size_bytes"),
        record_count=_integer(data["record_count"], "artifact record_count"),
    )


T = TypeVar("T")


def _load_jsonl(
    path: Path,
    digest: BundleFileDigest,
    decode: Callable[[object], T],
) -> tuple[T, ...]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise TinyWorldsBundleError(f"JSONL must end in a newline: {path.name}")
    lines = payload.splitlines(keepends=True)
    _require_record_count(digest, len(lines))
    values: list[T] = []
    for index, line in enumerate(lines, start=1):
        record = _loads_strict(line, f"{path.name}:{index}")
        if line != _canonical_json_bytes(record):
            raise TinyWorldsBundleError(
                f"non-canonical JSON record at {path.name}:{index}"
            )
        try:
            values.append(decode(record))
        except (KeyError, TypeError, ValueError) as error:
            raise TinyWorldsBundleError(
                f"invalid {path.name} record {index}: {error}"
            ) from error
    return tuple(values)


def _read_canonical_json(path: Path) -> object:
    try:
        payload = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError) as error:
        raise TinyWorldsBundleError(f"missing bundle file: {path.name}") from error
    record = _loads_strict(payload, path.name)
    if payload != _canonical_json_bytes(record):
        raise TinyWorldsBundleError(f"non-canonical JSON file: {path.name}")
    return record


def _loads_strict(payload: bytes, label: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TinyWorldsBundleError(
                    f"duplicate JSON field {key!r} in {label}"
                )
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TinyWorldsBundleError(f"invalid JSON in {label}: {error}") from error


def _canonical_json_bytes(record: object) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(records: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _fields(
    record: object,
    expected: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(record) is not dict:
        raise TinyWorldsBundleError(f"{label} must be a JSON object")
    actual = set(record)
    wanted = set(expected)
    if actual != wanted:
        unknown = tuple(sorted(actual - wanted))
        missing = tuple(sorted(wanted - actual))
        raise TinyWorldsBundleError(
            f"{label} fields differ; unknown={unknown}, missing={missing}"
        )
    return record


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TinyWorldsBundleError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TinyWorldsBundleError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TinyWorldsBundleError(f"{label} must be an integer")
    return value


def _require_digest(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TinyWorldsBundleError(
            f"{label} must be a lowercase hexadecimal SHA-256"
        )


def _require_record_count(digest: BundleFileDigest, actual: int) -> None:
    if digest.record_count != actual:
        raise TinyWorldsBundleError(
            f"artifact record-count mismatch: {digest.path}"
        )


def _digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _quoted(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _space_quoted(values: tuple[object, ...]) -> str:
    return " ".join(_quoted(value) for value in values)


def _metta_atom(atom: GroundAtom) -> str:
    arguments = " ".join(_quoted(str(item)) for item in atom.arguments)
    return f"({_quoted(str(atom.predicate_id))} {arguments})"


def _metta_pattern(pattern: AtomPattern) -> str:
    arguments = " ".join(
        _quoted(str(item)) if type(item) is EntityId else f"${item.name}"
        for item in pattern.arguments
    )
    return f"({_quoted(str(pattern.predicate_id))} {arguments})"


__all__ = [
    "BundleFileDigest",
    "TinyWorldsBundleError",
    "TinyWorldsBundleManifest",
    "export_metta_atomese",
    "load_tinyworlds_bundle",
    "load_tinyworlds_manifest",
    "tinyworlds_bundle_sha256",
    "write_tinyworlds_bundle",
]
