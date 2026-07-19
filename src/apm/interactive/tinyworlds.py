"""Dependency-light data and widget helpers for the TinyWorlds playground."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from apm.continual.tinyworlds_calibration import (
    CalibrationValidationTrial,
    TinyWorldsCalibrationResult,
)
from apm.continual.tinyworlds_calibration_profile import (
    calibration_artifact_tree_sha256,
    load_calibration_result,
)
from apm.data.text.tinyworlds import (
    DataSplit,
    EntityId,
    GroundAtom,
    HornRule,
    QueryKind,
    QueryPlan,
    RenderedQueryGroup,
    RenderedQueryVariant,
    RenderedStory,
    RenderedTinyWorlds,
    TINYWORLDS_VERSION,
    TinyWorldsBundle,
    TinyWorldsRenderPreset,
    answer_query,
    apply_standard_distractor_mix,
    compute_closure,
    derive_master_seed,
    generate_calibration_bundle,
    generate_pilot_bundle,
    load_tinyworlds_bundle,
    render_tinyworlds_bundle,
    tinyworlds_bundle_sha256,
)


DEFAULT_RESULT_RELATIVE_PATH = Path(
    "results/language_cl/tinyworlds-v1/knowledge-graph/"
    "calibration-stopped-seed0-e314a9704528"
)
DEMO_RENDER_PRESET = TinyWorldsRenderPreset(
    training_stories_per_task=12,
    validation_stories_per_task=3,
    test_stories_per_task=1,
    validation_query_groups_per_task=4,
    test_query_groups_per_task=1,
    root_validation_stories=2,
    story_token_count=256,
    context_length=256,
)
_TRIAL_FILENAMES = frozenset(
    {
        "candidate_scores.jsonl",
        "checkpointed_transfer.jsonl",
        "manifest.json",
        "model.npz",
        "parent_search.jsonl",
    }
)
_NEURAL_SCORE_FIELDS = frozenset(
    {
        "candidate_nll",
        "correct",
        "correct_candidate_index",
        "method",
        "metric",
        "predicted_candidate_index",
        "prefix_length",
        "query_id",
        "task_id",
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "adapter_checksum",
        "parent_node_id",
        "roles",
        "stream",
        "task_id",
        "training_loss",
        "update",
        "validation_candidate_accuracy",
        "validation_correct_nll",
    }
)
_PARENT_FIELDS = frozenset(
    {
        "correct_candidate_nll_by_query_and_node",
        "mean_correct_candidate_nll",
        "node_ids",
        "selected_node_id",
        "selected_node_index",
        "task_id",
        "validation_query_ids",
        "validation_suite_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One persisted four-candidate neural score comparison."""

    query_id: str
    task_id: str
    metric: str
    method: str
    prefix_length: int
    candidate_nll: tuple[float, float, float, float]
    correct_candidate_index: int
    predicted_candidate_index: int

    @property
    def correct(self) -> bool:
        """Whether the minimum-NLL candidate is the declared answer."""
        return self.predicted_candidate_index == self.correct_candidate_index

    @property
    def margin(self) -> float:
        """Return minimum wrong NLL minus correct-answer NLL."""
        wrong = tuple(
            value
            for index, value in enumerate(self.candidate_nll)
            if index != self.correct_candidate_index
        )
        return min(wrong) - self.candidate_nll[self.correct_candidate_index]

    @property
    def group_id(self) -> str:
        """Return the rendered semantic group shared by prefix variants."""
        marker = f":prefix-{self.prefix_length}"
        if not self.query_id.endswith(marker):
            raise ValueError("candidate score query ID has no prefix suffix")
        return self.query_id[: -len(marker)]


@dataclass(frozen=True, slots=True)
class TransferCheckpoint:
    """One immutable update checkpoint from a parent-transfer trial."""

    task_id: str
    stream: str
    parent_node_id: str
    roles: tuple[str, ...]
    update: int
    training_loss: float | None
    validation_candidate_accuracy: float
    validation_correct_nll: float
    adapter_checksum: str


@dataclass(frozen=True, slots=True)
class ParentSearchRecord:
    """Validation-only candidate-parent NLLs and the selected parent."""

    task_id: str
    node_ids: tuple[str, ...]
    mean_correct_candidate_nll: tuple[float, ...]
    selected_node_id: str
    selected_node_index: int
    validation_query_count: int


@dataclass(frozen=True, slots=True)
class GraphNodeRecord:
    """One learned VAMP node and its committed parent."""

    node_id: str
    parent_id: str | None
    trained_task: str | None
    train_stage: int


@dataclass(frozen=True, slots=True)
class CalibrationTrialArtifact:
    """Strict, analysis-sized projection of one immutable Phase 4 trial."""

    trial: CalibrationValidationTrial
    directory: Path
    candidate_scores: tuple[CandidateScore, ...]
    exact_kg_rows: int
    checkpoints: tuple[TransferCheckpoint, ...]
    parent_search: tuple[ParentSearchRecord, ...]
    learned_graph: tuple[GraphNodeRecord, ...]
    allocator_peak_bytes: int
    allocator_peak_target_bytes: int

    def scores_for_metric(self, metric: str) -> tuple[CandidateScore, ...]:
        """Return candidate rows for one named evidence metric."""
        return tuple(row for row in self.candidate_scores if row.metric == metric)


@dataclass(frozen=True, slots=True)
class TinyWorldsLab:
    """Validated Phase 4 evidence and the authoritative symbolic pool."""

    repository_root: Path
    result_directory: Path
    result: TinyWorldsCalibrationResult
    calibration_bundle: TinyWorldsBundle
    trials: tuple[CalibrationTrialArtifact, ...]
    tokenizer_path: Path

    def trial_for_facts(self, facts_per_task: int) -> CalibrationTrialArtifact:
        """Return the unique validation trial with the requested fact budget."""
        matches = tuple(
            artifact
            for artifact in self.trials
            if artifact.trial.request.config.facts_per_task == facts_per_task
        )
        if len(matches) != 1:
            raise KeyError(f"expected one trial for {facts_per_task} facts")
        return matches[0]


@dataclass(frozen=True, slots=True)
class TinyWorldsDemo:
    """Small in-memory rendering produced by the real deterministic pipeline."""

    public_seed: int
    world_name: Literal["calibration", "pilot"]
    fact_capacity: int
    distractor_policy: Literal["hard", "standard_mix"]
    bundle: TinyWorldsBundle
    rendered: RenderedTinyWorlds


@dataclass(frozen=True, slots=True)
class SupportFactInspection:
    """One proof leaf, its task exposure position, and ablation outcome."""

    fact_id: str
    task_id: str
    exposure_position: int
    atom_text: str
    answer_survives_removal: bool


@dataclass(frozen=True, slots=True)
class ProofStepInspection:
    """Readable projection of one canonical proof step."""

    atom_id: str
    atom_text: str
    depth: int
    rule_id: str | None
    premise_atom_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateInspection:
    """One symbolic answer candidate with its deterministic role."""

    index: int
    entity_id: str
    name: str
    role: str
    correct: bool


@dataclass(frozen=True, slots=True)
class HardSupportInspection:
    """Required-edge support available on one hard root-to-node path."""

    node_id: str
    path_edge_ids: tuple[str, ...]
    required_edge_recall: float


@dataclass(frozen=True, slots=True)
class QueryInspection:
    """Mechanically checked query, proof, candidates, and counterfactuals."""

    query_id: str
    task_id: str
    kind: QueryKind
    query_text: str
    answer_entity_id: str
    answer_name: str
    proof_depth: int
    answers_by_max_depth: tuple[tuple[int, tuple[str, ...]], ...]
    candidates: tuple[CandidateInspection, ...]
    proof_steps: tuple[ProofStepInspection, ...]
    support_facts: tuple[SupportFactInspection, ...]
    required_task_ids: tuple[str, ...]
    required_edge_ids: tuple[str, ...]
    hard_oracle_task_ids: tuple[str, ...]
    hard_support: tuple[HardSupportInspection, ...]


@dataclass(frozen=True, slots=True)
class ExactKgKindSummary:
    """Exact executor results for one semantic query family."""

    kind: QueryKind
    successes: int
    trials: int


@dataclass(frozen=True, slots=True)
class ExactKgSummary:
    """Exact symbolic executor accuracy over unique semantic plans."""

    successes: int
    trials: int
    by_kind: tuple[ExactKgKindSummary, ...]

    @property
    def accuracy(self) -> float:
        """Return exact semantic-plan accuracy."""
        return self.successes / self.trials


@dataclass(frozen=True, slots=True)
class DiagnosticCandidate:
    """Rendered candidate identity paired with a persisted NLL."""

    index: int
    entity_id: str
    name: str
    role: str
    answer_text: str
    suffix_token_count: int
    nll: float
    correct: bool
    predicted: bool


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    """Human-readable join of one rendered query and its saved scores."""

    score: CandidateScore
    prefix_text: str
    cue_regime: str
    eligible_task_ids: tuple[str, ...]
    candidates: tuple[DiagnosticCandidate, ...]


def load_tinyworlds_lab(
    result_directory: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> TinyWorldsLab:
    """Strictly load the stopped calibration, raw trials, and symbolic pool."""
    root = _resolve_repository_root(repository_root)
    result_root = (
        root / DEFAULT_RESULT_RELATIVE_PATH
        if result_directory is None
        else Path(result_directory).expanduser().resolve()
    )
    result = load_calibration_result(result_root)
    bundle = load_tinyworlds_bundle(result_root / "symbolic-calibration-pool")
    if tinyworlds_bundle_sha256(bundle) != result.identity.calibration_bundle_sha256:
        raise ValueError("calibration result references a different symbolic pool")
    expected_master_seed = derive_master_seed(
        TINYWORLDS_VERSION,
        result.identity.public_seed,
        result.identity.base_manifest_sha256,
        result.identity.base_parameter_checksum,
    )
    if bundle.world.master_seed_sha256 != expected_master_seed:
        raise ValueError("calibration symbolic pool uses a different master seed")
    trials = tuple(
        _load_trial_artifact(result_root, trial)
        for trial in result.validation_trials
    )
    return TinyWorldsLab(
        repository_root=root,
        result_directory=result_root,
        result=result,
        calibration_bundle=bundle,
        trials=trials,
        tokenizer_path=(
            root / "checkpoints/tinystories-8m/tokenizer/tokenizer.json"
        ),
    )


def generate_tinyworlds_demo(
    lab: TinyWorldsLab,
    *,
    public_seed: int = 0,
    world_name: Literal["calibration", "pilot"] = "calibration",
    fact_capacity: int = 36,
    distractor_policy: Literal["hard", "standard_mix"] = "hard",
) -> TinyWorldsDemo:
    """Generate and render a small world through the production APIs."""
    if type(lab) is not TinyWorldsLab:
        raise TypeError("lab must be a TinyWorldsLab")
    if type(public_seed) is not int or public_seed < 0:
        raise ValueError("public_seed must be a nonnegative integer")
    if world_name not in ("calibration", "pilot"):
        raise ValueError("world_name must be calibration or pilot")
    if fact_capacity not in (24, 36):
        raise ValueError("fact_capacity must be 24 or 36")
    if distractor_policy not in ("hard", "standard_mix"):
        raise ValueError("unknown distractor policy")
    from apm.lm.text import TokenizersTextTokenizer

    master_seed = derive_master_seed(
        TINYWORLDS_VERSION,
        public_seed,
        lab.result.identity.base_manifest_sha256,
        lab.result.identity.base_parameter_checksum,
    )
    if world_name == "calibration" and public_seed == 0 and fact_capacity == 36:
        bundle = lab.calibration_bundle
    else:
        generator = (
            generate_calibration_bundle
            if world_name == "calibration"
            else generate_pilot_bundle
        )
        bundle = generator(master_seed, direct_facts_per_task=fact_capacity)
    if distractor_policy == "standard_mix":
        bundle = apply_standard_distractor_mix(bundle)
    tokenizer = TokenizersTextTokenizer.from_file(lab.tokenizer_path)
    return TinyWorldsDemo(
        public_seed=public_seed,
        world_name=world_name,
        fact_capacity=fact_capacity,
        distractor_policy=distractor_policy,
        bundle=bundle,
        rendered=render_tinyworlds_bundle(bundle, tokenizer, DEMO_RENDER_PRESET),
    )


def exact_kg_summary(
    source: TinyWorldsDemo | TinyWorldsBundle,
    *,
    split: DataSplit = DataSplit.VALIDATION,
) -> ExactKgSummary:
    """Evaluate unique semantic plans with the authoritative graph executor."""
    bundle = source.bundle if type(source) is TinyWorldsDemo else source
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("source must be a TinyWorldsDemo or TinyWorldsBundle")
    if type(split) is not DataSplit or split is DataSplit.TRAIN:
        raise ValueError("exact knowledge evaluation requires validation or test")
    plans = tuple(plan for plan in bundle.query_plans if plan.split is split)
    outcomes = tuple(
        answer_query(
            bundle.closure,
            plan.query_ast,
            bundle.world.registry,
            bundle.entities,
        )
        == (plan.answer_entity_id,)
        for plan in plans
    )
    by_kind = tuple(
        ExactKgKindSummary(
            kind=kind,
            successes=sum(
                outcome
                for plan, outcome in zip(plans, outcomes)
                if plan.kind is kind
            ),
            trials=sum(plan.kind is kind for plan in plans),
        )
        for kind in QueryKind
        if any(plan.kind is kind for plan in plans)
    )
    return ExactKgSummary(sum(outcomes), len(outcomes), by_kind)


def inspect_query(
    source: TinyWorldsDemo | TinyWorldsBundle,
    kind: QueryKind,
    *,
    split: DataSplit = DataSplit.VALIDATION,
) -> QueryInspection:
    """Inspect one semantic plan, including depth limits and support ablations."""
    bundle = source.bundle if type(source) is TinyWorldsDemo else source
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("source must be a TinyWorldsDemo or TinyWorldsBundle")
    if type(kind) is not QueryKind:
        raise TypeError("kind must be a QueryKind")
    if type(split) is not DataSplit or split is DataSplit.TRAIN:
        raise ValueError("query inspection requires validation or test")
    plan = _query_plan(bundle, kind, split)
    names = {entity.entity_id: entity.name for entity in bundle.entities}
    proof = bundle.closure.proof_for(plan.proof.conclusion_atom_id)
    depth_answers = tuple(
        (
            depth,
            tuple(
                str(answer)
                for answer in answer_query(
                    compute_closure(
                        bundle.facts,
                        bundle.rules,
                        bundle.world.registry,
                        bundle.entities,
                        max_depth=depth,
                    ),
                    plan.query_ast,
                    bundle.world.registry,
                    bundle.entities,
                )
            ),
        )
        for depth in range(3)
    )
    support_facts = tuple(
        _support_fact_inspection(bundle, plan, fact_id, names)
        for fact_id in plan.proof.supporting_fact_ids
    )
    return QueryInspection(
        query_id=str(plan.query_ast.query_id),
        task_id=str(plan.task_id),
        kind=plan.kind,
        query_text=_format_query(plan, names),
        answer_entity_id=str(plan.answer_entity_id),
        answer_name=names[plan.answer_entity_id],
        proof_depth=proof.depth,
        answers_by_max_depth=depth_answers,
        candidates=tuple(
            CandidateInspection(
                index=index,
                entity_id=str(candidate.entity_id),
                name=names[candidate.entity_id],
                role=candidate.role.value,
                correct=index == plan.correct_index,
            )
            for index, candidate in enumerate(plan.candidates)
        ),
        proof_steps=tuple(
            ProofStepInspection(
                atom_id=str(step.atom.atom_id),
                atom_text=_format_atom(step.atom, names),
                depth=step.depth,
                rule_id=None if step.rule_id is None else str(step.rule_id),
                premise_atom_ids=tuple(str(value) for value in step.premise_atom_ids),
            )
            for step in proof.steps
        ),
        support_facts=support_facts,
        required_task_ids=tuple(str(value) for value in plan.proof.required_task_ids),
        required_edge_ids=tuple(str(value) for value in plan.proof.required_edge_ids),
        hard_oracle_task_ids=tuple(str(value) for value in plan.hard_oracle_task_ids),
        hard_support=_hard_support_inspections(bundle, plan),
    )


def candidate_diagnostic(
    lab: TinyWorldsLab,
    demo: TinyWorldsDemo,
    *,
    facts_per_task: int,
    metric: str,
) -> CandidateDiagnostic:
    """Join the first visible saved score row to its exact rendered candidates."""
    if (
        demo.bundle.world.master_seed_sha256
        != lab.calibration_bundle.world.master_seed_sha256
        or demo.world_name != "calibration"
        or demo.fact_capacity != 36
        or demo.distractor_policy != "hard"
    ):
        raise ValueError(
            "persisted candidate scores can only be joined to the seed-0 hard "
            "36-fact calibration demo"
        )
    artifact = lab.trial_for_facts(facts_per_task)
    groups = {group.group_id: group for group in demo.rendered.query_groups}
    matches = tuple(
        row
        for row in artifact.scores_for_metric(metric)
        if row.group_id in groups
    )
    if not matches:
        raise KeyError(
            f"demo has no rendered group for metric {metric!r}; regenerate seed 0"
        )
    score = matches[0]
    group = groups[score.group_id]
    variant = next(
        item for item in group.variants if item.knowledge_query.query_id == score.query_id
    )
    names = {entity.entity_id: entity.name for entity in demo.bundle.entities}
    role_by_entity = {
        str(candidate.entity_id): candidate.role.value
        for candidate in group.group_plan.candidates
    }
    return CandidateDiagnostic(
        score=score,
        prefix_text=variant.prefix_text,
        cue_regime=variant.knowledge_query.cue_regime,
        eligible_task_ids=variant.knowledge_query.eligible_task_ids,
        candidates=tuple(
            DiagnosticCandidate(
                index=index,
                entity_id=entity_id,
                name=names[EntityId(entity_id)],
                role=role_by_entity[entity_id],
                answer_text=candidate.answer_text,
                suffix_token_count=int(np.sum(candidate.competence_batch.loss_mask)),
                nll=score.candidate_nll[index],
                correct=index == score.correct_candidate_index,
                predicted=index == score.predicted_candidate_index,
            )
            for index, (entity_id, candidate) in enumerate(
                zip(variant.candidate_entity_ids, variant.knowledge_query.candidates)
            )
        ),
    )


def status_html(lab: TinyWorldsLab) -> str:
    """Render the terminal Phase 4 status as a compact HTML card."""
    stopped = lab.result.stop_reason is not None
    outcome = "Stopped as designed" if stopped else "Calibration passed"
    reason = (
        lab.result.stop_reason.value if lab.result.stop_reason is not None else "none"
    )
    return _card(
        "TinyWorlds v1 status",
        (
            ("Outcome", outcome),
            ("Reason", reason),
            ("Pilot authorized", str(lab.result.pilot_authorized)),
            ("Validation trials", str(len(lab.trials))),
            ("Phase 5", "not launched" if stopped else "authorized"),
        ),
        accent="#b45309" if stopped else "#047857",
    )


def tinyworlds_playground(
    lab: TinyWorldsLab,
    demo: TinyWorldsDemo,
) -> object:
    """Build the optional ipywidgets tabbed playground without eager UI imports."""
    from apm.interactive.tinyworlds_widgets import build_tinyworlds_playground

    return build_tinyworlds_playground(lab, demo)


def _load_trial_artifact(
    result_root: Path,
    trial: CalibrationValidationTrial,
) -> CalibrationTrialArtifact:
    directory = result_root / "validation" / trial.artifact_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"trial artifact is not a regular directory: {directory}")
    entries = tuple(directory.iterdir())
    if any(path.is_symlink() for path in entries):
        raise ValueError("trial artifact contains a symlink")
    if {path.name for path in entries} != _TRIAL_FILENAMES:
        raise ValueError("trial artifact file set changed")
    if calibration_artifact_tree_sha256(directory) != trial.artifact_sha256:
        raise ValueError("trial artifact tree checksum mismatch")
    manifest = _load_json_record(directory / "manifest.json")
    _require_fields(
        manifest,
        {
            "artifact_id",
            "artifacts",
            "evidence",
            "execution_sha256",
            "format",
            "model",
            "model_file_sha256",
            "request",
            "resource_evidence",
            "schema_version",
        },
        "trial manifest",
    )
    if (
        _string(manifest["artifact_id"], "artifact_id") != trial.artifact_id
        or _string(manifest["execution_sha256"], "execution_sha256")
        != trial.execution_sha256
        or _string(manifest["format"], "format")
        != "apm.tinyworlds.calibration-trial"
        or _integer(manifest["schema_version"], "schema_version") != 2
    ):
        raise ValueError("trial manifest identity changed")
    _validate_trial_file_hashes(directory, manifest)
    _validate_trial_request(manifest["request"], trial)
    _validate_manifest_evidence(manifest["evidence"], trial)
    candidate_scores, exact_rows = _load_candidate_scores(
        directory / "candidate_scores.jsonl"
    )
    checkpoints = _load_checkpoints(directory / "checkpointed_transfer.jsonl")
    parents = _load_parent_search(directory / "parent_search.jsonl")
    model = _record(manifest["model"], "model")
    graph = _load_graph(model)
    resources = _record(manifest["resource_evidence"], "resource_evidence")
    peak = _integer(resources.get("allocator_peak_bytes"), "allocator_peak_bytes")
    target = _integer(
        resources.get("allocator_peak_target_bytes"),
        "allocator_peak_target_bytes",
    )
    if peak > target:
        raise ValueError("trial allocator peak exceeds its recorded target")
    if exact_rows != trial.evidence.exact_kg.trials:
        raise ValueError("exact-KG score count differs from trial evidence")
    _validate_score_evidence(candidate_scores, trial)
    _validate_checkpoint_schedules(checkpoints, trial)
    return CalibrationTrialArtifact(
        trial=trial,
        directory=directory,
        candidate_scores=candidate_scores,
        exact_kg_rows=exact_rows,
        checkpoints=checkpoints,
        parent_search=parents,
        learned_graph=graph,
        allocator_peak_bytes=peak,
        allocator_peak_target_bytes=target,
    )


def _validate_trial_file_hashes(
    directory: Path,
    manifest: dict[str, object],
) -> None:
    artifact_hashes = _record(manifest["artifacts"], "artifacts")
    expected_record_files = (
        "candidate_scores.jsonl",
        "checkpointed_transfer.jsonl",
        "parent_search.jsonl",
    )
    _require_fields(artifact_hashes, set(expected_record_files), "artifact hashes")
    if any(
        _string(artifact_hashes[name], f"hash for {name}")
        != _file_sha256(directory / name)
        for name in expected_record_files
    ):
        raise ValueError("trial record checksum mismatch")
    if _string(manifest["model_file_sha256"], "model_file_sha256") != (
        _file_sha256(directory / "model.npz")
    ):
        raise ValueError("trial model checksum mismatch")


def _validate_trial_request(
    value: object,
    trial: CalibrationValidationTrial,
) -> None:
    request = _record(value, "trial request")
    _require_fields(
        request,
        {"config", "locked_scratch_rerun", "purpose", "trial_index"},
        "trial request",
    )
    config = _record(request["config"], "trial config")
    expected_config = trial.request.config
    expected = {
        "distractor_policy": expected_config.distractor_policy.value,
        "exposures_per_fact": expected_config.exposures_per_fact,
        "facts_per_task": expected_config.facts_per_task,
        "lora_rank": expected_config.lora_rank,
        "update_budget": expected_config.update_budget,
    }
    if config != expected or request != {
        "config": expected,
        "locked_scratch_rerun": trial.request.locked_scratch_rerun,
        "purpose": trial.request.purpose.value,
        "trial_index": trial.request.trial_index,
    }:
        raise ValueError("trial request differs from calibration result")


def _validate_manifest_evidence(
    value: object,
    trial: CalibrationValidationTrial,
) -> None:
    evidence = trial.evidence
    binomial = lambda item: {
        "successes": item.successes,
        "trials": item.trials,
    }
    snapshots = lambda values: [
        {
            "adapter_sha256": item.adapter_sha256,
            "answers_sha256": item.answers_sha256,
            "logits_sha256": item.logits_sha256,
            "node_id": item.node_id,
        }
        for item in values
    ]
    expected = {
        "committed_node_stability": {
            "after": snapshots(evidence.committed_node_stability.after),
            "before": snapshots(evidence.committed_node_stability.before),
        },
        "exact_kg": binomial(evidence.exact_kg),
        "frozen_novel_binding": binomial(evidence.frozen_novel_binding),
        "frozen_one_hop": binomial(evidence.frozen_one_hop),
        "independent_direct_recall": binomial(evidence.independent_direct_recall),
        "independent_one_hop": binomial(evidence.independent_one_hop),
        "old_contextual_answer": binomial(evidence.old_contextual_answer),
        "paired_revision_consistency": binomial(evidence.paired_revision_consistency),
        "revision_contextual_answer": binomial(evidence.revision_contextual_answer),
    }
    if _record(value, "trial evidence") != expected:
        raise ValueError("trial manifest evidence differs from calibration result")


def _validate_score_evidence(
    scores: tuple[CandidateScore, ...],
    trial: CalibrationValidationTrial,
) -> None:
    expected = (
        ("frozen_novel_binding", trial.evidence.frozen_novel_binding),
        ("independent_direct_recall", trial.evidence.independent_direct_recall),
        ("frozen_one_hop", trial.evidence.frozen_one_hop),
        ("independent_one_hop", trial.evidence.independent_one_hop),
        ("old_contextual_answer", trial.evidence.old_contextual_answer),
        ("revision_contextual_answer", trial.evidence.revision_contextual_answer),
    )
    if {row.metric for row in scores} != {name for name, _ in expected}:
        raise ValueError("candidate score metric coverage changed")
    for name, evidence in expected:
        rows = tuple(row for row in scores if row.metric == name)
        if len(rows) != evidence.trials or sum(row.correct for row in rows) != (
            evidence.successes
        ):
            raise ValueError(f"candidate rows differ from {name} evidence")


def _validate_checkpoint_schedules(
    checkpoints: tuple[TransferCheckpoint, ...],
    trial: CalibrationValidationTrial,
) -> None:
    budget = trial.request.config.update_budget
    expected_updates = (0,) + tuple(
        2**power for power in range(budget.bit_length()) if 2**power < budget
    ) + (budget,)
    identities = tuple(
        dict.fromkeys(
            (row.task_id, row.stream, row.parent_node_id, row.roles)
            for row in checkpoints
        )
    )
    if any(
        tuple(
            row.update
            for row in checkpoints
            if (row.task_id, row.stream, row.parent_node_id, row.roles) == identity
        )
        != expected_updates
        for identity in identities
    ):
        raise ValueError("transfer checkpoint schedule differs from the fixed budget")


def _load_candidate_scores(path: Path) -> tuple[tuple[CandidateScore, ...], int]:
    records = _load_jsonl(path)
    exact_rows = tuple(record for record in records if record.get("metric") == "exact_kg")
    neural = tuple(record for record in records if record.get("metric") != "exact_kg")
    for record in exact_rows:
        _require_fields(
            record,
            {
                "answer_entity_ids",
                "correct",
                "group_id",
                "method",
                "metric",
                "query_id",
                "task_id",
            },
            "exact-KG score",
        )
        answer_ids = _string_tuple(record["answer_entity_ids"], "answer_entity_ids")
        if (
            len(answer_ids) != 1
            or record["correct"] is not True
            or record["method"] != "exact_kg"
            or record["metric"] != "exact_kg"
        ):
            raise ValueError("exact-KG row is inconsistent")
    scores = tuple(_candidate_score(record) for record in neural)
    if len({row.query_id + "\0" + row.metric for row in scores}) != len(scores):
        raise ValueError("candidate score identities are not unique")
    return scores, len(exact_rows)


def _candidate_score(record: dict[str, object]) -> CandidateScore:
    _require_fields(record, set(_NEURAL_SCORE_FIELDS), "candidate score")
    values = _float_tuple(record["candidate_nll"], "candidate_nll", length=4)
    correct_index = _index(record["correct_candidate_index"], "correct index", 4)
    predicted_index = _index(
        record["predicted_candidate_index"], "predicted index", 4
    )
    expected_prediction = min(range(4), key=values.__getitem__)
    if predicted_index != expected_prediction:
        raise ValueError("candidate prediction is not the minimum NLL")
    expected_correct = predicted_index == correct_index
    if record["correct"] is not expected_correct:
        raise ValueError("candidate correctness flag is inconsistent")
    prefix_length = _integer(record["prefix_length"], "prefix_length")
    if prefix_length not in (64, 128, 192):
        raise ValueError("candidate prefix length is unsupported")
    return CandidateScore(
        query_id=_string(record["query_id"], "query_id"),
        task_id=_string(record["task_id"], "task_id"),
        metric=_string(record["metric"], "metric"),
        method=_string(record["method"], "method"),
        prefix_length=prefix_length,
        candidate_nll=values,
        correct_candidate_index=correct_index,
        predicted_candidate_index=predicted_index,
    )


def _load_checkpoints(path: Path) -> tuple[TransferCheckpoint, ...]:
    checkpoints = tuple(_checkpoint(record) for record in _load_jsonl(path))
    identities = tuple(
        (row.task_id, row.stream, row.parent_node_id, row.roles, row.update)
        for row in checkpoints
    )
    if len(set(identities)) != len(identities):
        raise ValueError("transfer checkpoint identities are not unique")
    grouped = tuple(
        tuple(
            row
            for row in checkpoints
            if (row.task_id, row.stream, row.parent_node_id, row.roles) == identity
        )
        for identity in dict.fromkeys(value[:-1] for value in identities)
    )
    if any(
        tuple(row.update for row in rows) != tuple(sorted(row.update for row in rows))
        for rows in grouped
    ):
        raise ValueError("transfer checkpoint updates are not ordered")
    return checkpoints


def _checkpoint(record: dict[str, object]) -> TransferCheckpoint:
    _require_fields(record, set(_CHECKPOINT_FIELDS), "transfer checkpoint")
    training_loss = record["training_loss"]
    return TransferCheckpoint(
        task_id=_string(record["task_id"], "task_id"),
        stream=_string(record["stream"], "stream"),
        parent_node_id=_string(record["parent_node_id"], "parent_node_id"),
        roles=_string_tuple(record["roles"], "roles"),
        update=_integer(record["update"], "update"),
        training_loss=(
            None if training_loss is None else _finite_float(training_loss, "training_loss")
        ),
        validation_candidate_accuracy=_probability(
            record["validation_candidate_accuracy"],
            "validation_candidate_accuracy",
        ),
        validation_correct_nll=_finite_float(
            record["validation_correct_nll"], "validation_correct_nll"
        ),
        adapter_checksum=_string(record["adapter_checksum"], "adapter_checksum"),
    )


def _load_parent_search(path: Path) -> tuple[ParentSearchRecord, ...]:
    records = tuple(_parent_search(record) for record in _load_jsonl(path))
    if len({record.task_id for record in records}) != len(records):
        raise ValueError("parent-search task IDs are not unique")
    return records


def _parent_search(record: dict[str, object]) -> ParentSearchRecord:
    _require_fields(record, set(_PARENT_FIELDS), "parent search")
    node_ids = _string_tuple(record["node_ids"], "node_ids")
    means = _float_tuple(
        record["mean_correct_candidate_nll"],
        "mean_correct_candidate_nll",
        length=len(node_ids),
    )
    selected_index = _index(
        record["selected_node_index"], "selected_node_index", len(node_ids)
    )
    selected_id = _string(record["selected_node_id"], "selected_node_id")
    if selected_id != node_ids[selected_index] or selected_index != min(
        range(len(means)), key=means.__getitem__
    ):
        raise ValueError("parent search selection is inconsistent")
    query_ids = _string_tuple(record["validation_query_ids"], "validation_query_ids")
    matrix = _list(record["correct_candidate_nll_by_query_and_node"], "parent matrix")
    if len(matrix) != len(query_ids):
        raise ValueError("parent score matrix does not align with query IDs")
    matrix_rows = tuple(
        _float_tuple(row, "parent score row", length=len(node_ids))
        for row in matrix
    )
    if any(len(row) != len(node_ids) for row in matrix_rows):
        raise ValueError("parent score row has the wrong node count")
    return ParentSearchRecord(
        task_id=_string(record["task_id"], "task_id"),
        node_ids=node_ids,
        mean_correct_candidate_nll=means,
        selected_node_id=selected_id,
        selected_node_index=selected_index,
        validation_query_count=len(query_ids),
    )


def _load_graph(model: dict[str, object]) -> tuple[GraphNodeRecord, ...]:
    graph_values = _list(model.get("graph"), "learned graph")
    graph = tuple(
        GraphNodeRecord(
            node_id=_string(item.get("node_id"), "node_id"),
            parent_id=(
                None
                if item.get("parent_id") is None
                else _string(item.get("parent_id"), "parent_id")
            ),
            trained_task=(
                None
                if item.get("trained_task") is None
                else _string(item.get("trained_task"), "trained_task")
            ),
            train_stage=_integer(item.get("train_stage"), "train_stage"),
        )
        for value in graph_values
        for item in (_record(value, "graph node"),)
    )
    if not graph or graph[0] != GraphNodeRecord("root", None, None, 0):
        raise ValueError("learned graph has an invalid root")
    preceding_ids: set[str] = set()
    for node in graph:
        if node.node_id in preceding_ids or (
            node.parent_id is not None and node.parent_id not in preceding_ids
        ):
            raise ValueError("learned graph is not parent-before-child")
        preceding_ids.add(node.node_id)
    return graph


def _support_fact_inspection(
    bundle: TinyWorldsBundle,
    plan: QueryPlan,
    fact_id,
    names: dict[EntityId, str],
) -> SupportFactInspection:
    owner = next(task for task in bundle.tasks if fact_id in task.direct_fact_ids)
    fact = next(value for value in bundle.facts if value.atom_id == fact_id)
    reduced = tuple(value for value in bundle.facts if value.atom_id != fact_id)
    answers = answer_query(
        compute_closure(
            reduced,
            bundle.rules,
            bundle.world.registry,
            bundle.entities,
        ),
        plan.query_ast,
        bundle.world.registry,
        bundle.entities,
    )
    return SupportFactInspection(
        fact_id=str(fact_id),
        task_id=str(owner.task_id),
        exposure_position=owner.direct_fact_ids.index(fact_id) + 1,
        atom_text=_format_atom(fact, names),
        answer_survives_removal=plan.answer_entity_id in answers,
    )


def _hard_support_inspections(
    bundle: TinyWorldsBundle,
    plan: QueryPlan,
) -> tuple[HardSupportInspection, ...]:
    required = {str(edge_id) for edge_id in plan.proof.required_edge_ids}
    return tuple(
        HardSupportInspection(
            node_id=str(task.task_id),
            path_edge_ids=tuple(
                f"edge:{task_id}" for task_id in bundle.world.task_path(task.task_id)
            ),
            required_edge_recall=(
                len(
                    required.intersection(
                        f"edge:{task_id}"
                        for task_id in bundle.world.task_path(task.task_id)
                    )
                )
                / len(required)
            ),
        )
        for task in bundle.tasks
    )


def _query_plan(
    bundle: TinyWorldsBundle,
    kind: QueryKind,
    split: DataSplit,
) -> QueryPlan:
    matches = tuple(
        plan
        for plan in bundle.query_plans
        if plan.kind is kind and plan.split is split
    )
    if not matches:
        raise KeyError(f"no {split.value} {kind.value} query is available")
    return matches[0]


def _format_atom(atom: GroundAtom, names: dict[EntityId, str]) -> str:
    return f"{atom.predicate_id}({', '.join(names[value] for value in atom.arguments)})"


def _format_pattern(pattern, names: dict[EntityId, str]) -> str:
    arguments = ", ".join(
        names[value] if type(value) is EntityId else f"?{value.name}"
        for value in pattern.arguments
    )
    return f"{pattern.predicate_id}({arguments})"


def _format_query(plan: QueryPlan, names: dict[EntityId, str]) -> str:
    clauses = " AND ".join(_format_pattern(value, names) for value in plan.query_ast.clauses)
    return f"SELECT ?{plan.query_ast.answer_variable.name} WHERE {clauses}"


def _format_rule(rule: HornRule, names: dict[EntityId, str]) -> str:
    body = " AND ".join(_format_pattern(value, names) for value in rule.body)
    return f"{body} -> {_format_pattern(rule.head, names)}"


def _resolve_repository_root(value: str | Path | None) -> Path:
    if value is not None:
        root = Path(value).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"not an APM repository root: {root}")
        return root
    starts = (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve())
    candidates = tuple(
        parent
        for start in starts
        for parent in (start, *start.parents)
        if (parent / "pyproject.toml").is_file()
    )
    if not candidates:
        raise FileNotFoundError("could not locate the repository root")
    return candidates[0]


def _load_json_record(path: Path) -> dict[str, object]:
    value = _loads_strict(path.read_text(encoding="utf-8"))
    return _record(value, path.name)


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        _record(_loads_strict(line), f"{path.name} line {line_number}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line
    )


def _loads_strict(payload: str) -> object:
    return json.loads(payload, object_pairs_hook=_unique_object)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("JSON object contains a duplicate key")
    return dict(pairs)


def _record(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON list")
    return list(value)


def _require_fields(
    record: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(record) != expected:
        raise ValueError(f"{label} fields changed")


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _probability(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _index(value: object, label: str, length: int) -> int:
    result = _integer(value, label)
    if result >= length:
        raise ValueError(f"{label} is outside its sequence")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(item, label) for item in _list(value, label))


def _float_tuple(
    value: object,
    label: str,
    *,
    length: int,
) -> tuple[float, ...]:
    result = tuple(_finite_float(item, label) for item in _list(value, label))
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _card(
    title: str,
    rows: tuple[tuple[str, str], ...],
    *,
    accent: str = "#2563eb",
) -> str:
    body = "".join(
        "<div style='display:grid;grid-template-columns:11rem 1fr;gap:.35rem'>"
        f"<strong>{escape(label)}</strong><span>{escape(value)}</span></div>"
        for label, value in rows
    )
    return (
        f"<section style='border-left:5px solid {accent};padding:.8rem 1rem;"
        "background:#f8fafc;border-radius:.25rem;margin:.4rem 0'>"
        f"<h3 style='margin:.05rem 0 .65rem'>{escape(title)}</h3>{body}</section>"
    )
