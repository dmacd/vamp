"""Replay-rotation and old-task adaptation diagnosis on ImageNet-R-50."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import time

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.behavior_replay_workflow import (
    _stored_integrator_protocol,
    seed_row_cache_shards,
)
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.integrator_artifacts import (
    IntegratorBootstrap,
    IntegratorStore,
)
from apm.continual.vision.imagenetr.integrator_bank import class_stratified_reservoir
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_model import (
    IntegratorFitResult,
    IntegratorState,
    create_integrator_state,
    fit_fresh_integrator,
    parameter_count,
    predict_integrator,
)
from apm.continual.vision.imagenetr.integrator_observations import (
    FrontierTensors,
    accuracy,
)
from apm.continual.vision.imagenetr.integrator_persistence import (
    load_integrator_fit,
    publish_integrator_fit,
    restore_integrator_checkpoint,
    save_integrator_checkpoint,
)
from apm.continual.vision.imagenetr.integrator_workflow import (
    HISTORICAL_NAMESPACE,
    _control_metrics,
    _frontier_tensors,
    _hierarchy_frontier,
    _partition_rows,
    _supervision,
    _test_rows,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    PromotedBootstrap,
    _build,
    bootstrap_promoted_integrator,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.replay_adaptation_config import (
    DEFAULT_REPLAY_ADAPTATION_CONFIG,
    ReplayAdaptationConfig,
    load_replay_adaptation_config,
)
from apm.continual.vision.imagenetr.replay_adaptation_training import (
    fit_replay_epochs,
    reset_adamw,
    task_uniform_weights,
)


FEATURE_VARIANT = "behavior"
COMMON_STATE_NAME = "persistent-node-adapted-behavior-common-seed-v1"
FULL_HISTORY_CONDITION = "fresh_full_history"


@dataclass(frozen=True, slots=True)
class ReplayAdaptationProtocol:
    """Content identity binding the diagnosis to frozen source artifacts."""

    promoted_run_hash: str
    hierarchy_policy_hash: str
    hierarchy_complete_sha256: str
    behavior_run_hash: str
    behavior_protocol_sha256: str
    behavior_result_sha256: str
    behavior_training_seal_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-replay-adaptation-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("promoted run", self.promoted_run_hash),
            ("hierarchy policy", self.hierarchy_policy_hash),
            ("hierarchy completion", self.hierarchy_complete_sha256),
            ("behavior run", self.behavior_run_hash),
            ("behavior protocol", self.behavior_protocol_sha256),
            ("behavior result", self.behavior_result_sha256),
            ("behavior training seal", self.behavior_training_seal_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("config", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-replay-adaptation-protocol-v1":
            raise ValueError("invalid replay-adaptation protocol")

    @property
    def content_hash(self) -> str:
        """Return the stable diagnostic run namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical JSON-compatible protocol record."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class ReplayAdaptationBootstrap:
    """Authenticated source hierarchy and isolated diagnostic output context."""

    project_root: Path
    config: ReplayAdaptationConfig
    source: PromotedBootstrap
    behavior_run: Path
    protocol: ReplayAdaptationProtocol
    store: IntegratorStore
    integrator: IntegratorBootstrap


@dataclass(frozen=True, slots=True)
class OnlineCondition:
    """One cell of the replay sampler, objective, and optimizer factorial."""

    sampler: str
    weighting: str
    optimizer: str

    @property
    def condition_id(self) -> str:
        """Return the stable artifact and report identifier."""
        return f"{self.sampler}__{self.weighting}__{self.optimizer}"


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """Current rows plus one deterministic bounded historical draw."""

    current: tuple[ImageRecord, ...]
    historical: tuple[ImageRecord, ...]
    namespace: str

    @property
    def rows(self) -> tuple[ImageRecord, ...]:
        """Return the stable current-plus-history training order."""
        return self.current + self.historical


@dataclass(frozen=True, slots=True)
class OnlineInvocation:
    """New optimizer work performed while advancing one online condition."""

    condition_id: str
    new_optimizer_steps: int
    new_stages: int


def online_conditions(config: ReplayAdaptationConfig) -> tuple[OnlineCondition, ...]:
    """Expand the fixed eight-cell paired online matrix."""
    return tuple(OnlineCondition(*values) for values in config.conditions)


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/imagenetr50_logt_replay_adaptation_protocol.md",
        project_root / "scripts/vision/imagenetr/run_replay_adaptation_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        project_root / "src/apm/continual/logt_behavioral_integrator.py",
        project_root / "src/apm/continual/logt_behavioral_router.py",
        project_root / "src/apm/experiments/vamp_logt_mlp_permuted_online.py",
        project_root / "configs/vamp_logt_mlp_permuted_mnist/primary.yaml",
        package,
    )


def _behavior_paths(
    config: ReplayAdaptationConfig,
) -> tuple[Path, Path, Path, Path]:
    run = config.behavior_artifact_root / "runs" / config.behavior_run_hash
    return (
        run,
        run / "protocol" / "protocol.json",
        run / "evaluations" / "result.json",
        run / "protocol" / "training_seal.json",
    )


def bootstrap_replay_adaptation(
    config_path: str | Path = DEFAULT_REPLAY_ADAPTATION_CONFIG,
) -> ReplayAdaptationBootstrap:
    """Authenticate sources and prepare the content-addressed diagnostic run."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_replay_adaptation_config(resolved)
    current_source = bootstrap_promoted_integrator(config.source_config)
    source_store = IntegratorStore(
        current_source.promotion.artifact_root, config.promoted_run_hash
    )
    stored_protocol = _stored_integrator_protocol(
        load_canonical_json(source_store.run / "protocol" / "protocol.json")
    )
    source = replace(
        current_source,
        integrator=replace(
            current_source.integrator,
            protocol=stored_protocol,
            store=source_store,
            code_manifest=load_canonical_json(
                source_store.run / "protocol" / "code_manifest.json"
            ),
            environment_manifest=load_canonical_json(
                source_store.run / "protocol" / "environment_manifest.json"
            ),
        ),
    )
    atomic_write(
        current_source.promotion.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": config.promoted_run_hash,
                "schema_version": "imagenetr50-promoted-integrator-latest-v1",
            }
        ),
    )
    hierarchy_completion = (
        source.integrator.store.run
        / "hierarchies"
        / config.hierarchy_policy_hash
        / "complete_050.json"
    )
    behavior_run, behavior_protocol, behavior_result, behavior_seal = _behavior_paths(
        config
    )
    observed = (
        file_sha256(hierarchy_completion),
        file_sha256(behavior_protocol),
        file_sha256(behavior_result),
        file_sha256(behavior_seal),
    )
    expected = (
        config.hierarchy_complete_sha256,
        config.behavior_protocol_sha256,
        config.behavior_result_sha256,
        config.behavior_training_seal_sha256,
    )
    if observed != expected:
        raise ValueError("configured replay source artifacts changed")
    source_seal = load_canonical_json(behavior_seal)
    if (
        int(source_seal.get("test_requests_before_seal", -1)) != 0
        or load_canonical_json(behavior_protocol).get("content_hash")
        != config.behavior_run_hash
    ):
        raise ValueError("source behavior run was not cleanly sealed")
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = ReplayAdaptationProtocol(
        config.promoted_run_hash,
        config.hierarchy_policy_hash,
        config.hierarchy_complete_sha256,
        config.behavior_run_hash,
        config.behavior_protocol_sha256,
        config.behavior_result_sha256,
        config.behavior_training_seal_sha256,
        source.integrator.manifest.content_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    store = IntegratorStore(config.artifact_root, protocol.content_hash)
    store.prepare(protocol)  # type: ignore[arg-type]
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("source_behavior_protocol.json", load_canonical_json(behavior_protocol)),
        ("source_behavior_seal.json", source_seal),
    ):
        publish_immutable_json(store.run / "protocol" / filename, record)
    publish_immutable_bytes(
        store.run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-replay-adaptation-latest-v1",
            }
        ),
    )
    return ReplayAdaptationBootstrap(
        project_root,
        config,
        source,
        behavior_run,
        protocol,
        store,
        replace(source.integrator, store=store),
    )


def _write_state(
    bootstrap: ReplayAdaptationBootstrap, phase: str, **values: object
) -> None:
    atomic_write(
        bootstrap.store.run / "state" / "workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-replay-adaptation-state-v1",
                **values,
            }
        ),
    )


def _stat_fingerprint(root: Path, pattern: str = "*") -> dict[str, object]:
    files = tuple(sorted(path for path in root.rglob(pattern) if path.is_file()))
    rows = tuple(
        {
            "mtime_ns": path.stat().st_mtime_ns,
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    )
    return {
        "entries_hash": record_sha256(rows),
        "file_count": len(rows),
        "mtime_ns_sum": sum(int(row["mtime_ns"]) for row in rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def _stable_cache_record(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"fallback_file_copies", "linked_shards", "reused_shards"}
    }


def _selection(
    bootstrap: ReplayAdaptationBootstrap, stage: int, sampler: str
) -> ReplaySelection:
    current = tuple(
        sorted(
            _partition_rows(bootstrap.integrator, "fit", (stage - 1,)),
            key=lambda row: row.image_id,
        )
    )
    if stage == 1:
        return ReplaySelection(current, (), "none")
    history = _partition_rows(
        bootstrap.integrator, "fit", tuple(range(stage - 1))
    )
    namespace = (
        HISTORICAL_NAMESPACE
        if sampler == "static"
        else f"imagenetr50-integrator-history-stage-draw-v1:{bootstrap.config.seed}:{stage}"
    )
    if sampler not in bootstrap.config.sampler_modes:
        raise ValueError("unknown replay sampler")
    reservoir = class_stratified_reservoir(
        history, bootstrap.config.historical_capacity, namespace
    )
    by_id = {row.image_id: row for row in history}
    historical = tuple(
        sorted(
            (by_id[image_id] for image_id in reservoir.image_ids),
            key=lambda row: row.image_id,
        )
    )
    if {row.image_id for row in current} & {row.image_id for row in historical}:
        raise ValueError("current and historical replay rows overlap")
    return ReplaySelection(current, historical, namespace)


def _selection_diagnostics(
    bootstrap: ReplayAdaptationBootstrap,
    stage: int,
    sampler: str,
    selected: ReplaySelection,
    previously_selected: frozenset[str],
    cumulative_selected: frozenset[str],
) -> dict[str, object]:
    historical_ids = frozenset(row.image_id for row in selected.historical)
    overlap = len(historical_ids & previously_selected)
    cumulative = cumulative_selected | historical_ids
    return {
        "cumulative_historical_unique": len(cumulative),
        "current_examples": len(selected.current),
        "current_ids_hash": record_sha256([row.image_id for row in selected.current]),
        "historical_capacity": bootstrap.config.historical_capacity,
        "historical_examples": len(selected.historical),
        "historical_ids_hash": record_sha256(
            [row.image_id for row in selected.historical]
        ),
        "historical_novel_since_previous": len(historical_ids - previously_selected),
        "historical_overlap_fraction": (
            None if not historical_ids else overlap / len(historical_ids)
        ),
        "namespace": selected.namespace,
        "sampler": sampler,
        "schema_version": "imagenetr50-replay-selection-v1",
        "stage": stage,
        "training_examples": len(selected.rows),
        "training_ids_hash": record_sha256(
            [row.image_id for row in selected.rows]
        ),
    }


def _selection_history(
    bootstrap: ReplayAdaptationBootstrap, sampler: str, through_stage: int
) -> tuple[frozenset[str], frozenset[str]]:
    previous: frozenset[str] = frozenset()
    cumulative: frozenset[str] = frozenset()
    for stage in range(1, through_stage + 1):
        historical = frozenset(
            row.image_id for row in _selection(bootstrap, stage, sampler).historical
        )
        previous = historical
        cumulative = cumulative | historical
    return previous, cumulative


def _task_accuracies(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    task_ids = labels // 4
    return {
        str(task + 1): accuracy(predictions[task_ids == task], labels[task_ids == task])
        for task in torch.unique(task_ids, sorted=True).tolist()
    }


def _evaluation_metrics(
    state: IntegratorState, tensors: FrontierTensors, device: torch.device
) -> dict[str, object]:
    predictions = predict_integrator(
        state.model,
        tensors.observations(FEATURE_VARIANT),
        512,
        device,
    )
    task_metrics = _task_accuracies(predictions, tensors.labels)
    current_task = max(int(task) for task in task_metrics)
    old_values = tuple(
        value for task, value in task_metrics.items() if int(task) < current_task
    )
    return {
        "accuracy": accuracy(predictions, tensors.labels),
        "current_task_accuracy": task_metrics[str(current_task)],
        "old_task_macro_accuracy": (
            None if not old_values else math.fsum(old_values) / len(old_values)
        ),
        "task_accuracies": task_metrics,
    }


def _condition_root(
    bootstrap: ReplayAdaptationBootstrap, condition: OnlineCondition
) -> Path:
    return bootstrap.store.run / "integrators" / "online" / condition.condition_id


def _checkpoint_path(
    bootstrap: ReplayAdaptationBootstrap,
    condition: OnlineCondition,
    stage: int,
) -> Path:
    return _condition_root(bootstrap, condition) / "checkpoints" / f"stage_{stage:03d}.pt"


def _new_online_state(
    bootstrap: ReplayAdaptationBootstrap, device: torch.device
) -> IntegratorState:
    return create_integrator_state(
        COMMON_STATE_NAME,
        bootstrap.integrator.config.maximum_levels,
        FEATURE_VARIANT,
        bootstrap.integrator.config.optimization,
        bootstrap.config.seed,
        device,
    )


def _restore_online_state(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    condition: OnlineCondition,
    stage: int,
    device: torch.device,
) -> IntegratorState:
    state = _new_online_state(bootstrap, device)
    selected = _selection(bootstrap, stage, condition.sampler)
    _nodes, _slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
    restore_integrator_checkpoint(
        _checkpoint_path(bootstrap, condition, stage),
        state,
        stage,
        FEATURE_VARIANT,
        frontier_hash,
        record_sha256([row.image_id for row in selected.rows]),
    )
    return state


def _train_online_condition(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    condition: OnlineCondition,
    device: torch.device,
) -> OnlineInvocation:
    root = _condition_root(bootstrap, condition)
    ledger = ChainedJsonlLedger(
        root / "training_metrics.jsonl",
        "imagenetr50-replay-adaptation-online-stage-v1",
    )
    rows_by_stage = {int(row["stage"]): row for row in ledger.rows}
    if set(rows_by_stage) != set(range(1, len(rows_by_stage) + 1)):
        raise ValueError("online replay stages are not a contiguous prefix")
    completed = len(rows_by_stage)
    state = (
        _new_online_state(bootstrap, device)
        if completed == 0
        else _restore_online_state(
            bootstrap, hierarchy, condition, completed, device
        )
    )
    previous_ids, cumulative_ids = _selection_history(
        bootstrap, condition.sampler, completed
    )
    new_steps = new_stages = 0
    from tqdm.auto import tqdm

    progress = tqdm(
        range(completed + 1, 51),
        desc=condition.condition_id,
        unit="task",
    )
    for stage in progress:
        selected = _selection(bootstrap, stage, condition.sampler)
        selection_record = _selection_diagnostics(
            bootstrap,
            stage,
            condition.sampler,
            selected,
            previous_ids,
            cumulative_ids,
        )
        historical_ids = frozenset(row.image_id for row in selected.historical)
        previous_ids = historical_ids
        cumulative_ids = cumulative_ids | historical_ids
        nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
        if condition.optimizer == "reset_each_stage":
            reset_adamw(state, bootstrap.integrator.config.optimization)
        tensors = _frontier_tensors(
            bootstrap.integrator,
            nodes,
            slots,
            bootstrap.integrator.config.maximum_levels,
            selected.rows,
            device,
            f"replay_adaptation_{condition.sampler}_fit",
        )
        supervision = _supervision(tensors, FEATURE_VARIANT)
        fit = fit_replay_epochs(
            state,
            supervision,
            bootstrap.config.online_epochs,
            bootstrap.integrator.config.optimization,
            bootstrap.config.seed,
            stage,
            device,
            condition.weighting,
        )
        selected_metrics = _evaluation_metrics(state, tensors, device)
        weights = (
            torch.ones(len(tensors.labels))
            if condition.weighting == "example_uniform"
            else task_uniform_weights(tensors.labels)
        )
        task_weight_totals = tuple(
            float(weights[tensors.labels // 4 == task].sum())
            for task in torch.unique(tensors.labels // 4, sorted=True).tolist()
        )
        checkpoint = _checkpoint_path(bootstrap, condition, stage)
        save_integrator_checkpoint(
            checkpoint,
            state,
            stage,
            FEATURE_VARIANT,
            frontier_hash,
            str(selection_record["training_ids_hash"]),
        )
        metric = {
            "cache_hits": tensors.cache_hits,
            "cache_misses": tensors.cache_misses,
            "condition": condition.condition_id,
            "fit": asdict(fit),
            "frontier_hash": frontier_hash,
            "live_nodes": len(nodes),
            "node_example_forwards": tensors.node_example_forwards,
            "node_example_forwards_bound": len(selected.rows) * len(nodes),
            "optimizer_policy": condition.optimizer,
            "optimizer_steps_total": state.optimizer_steps,
            "parameter_count": parameter_count(state.model),
            "selected_training": selected_metrics,
            "selection": selection_record,
            "stage": stage,
            "task_weight_total_max": max(task_weight_totals),
            "task_weight_total_min": min(task_weight_totals),
            "weighting": condition.weighting,
        }
        ledger.append(metric)
        if stage - 1 not in bootstrap.config.diagnostic_stages:
            previous_checkpoint = _checkpoint_path(
                bootstrap, condition, stage - 1
            )
            if previous_checkpoint.is_file():
                previous_checkpoint.unlink()
        new_steps += fit.optimizer_steps
        new_stages += 1
        progress.set_postfix(
            live=len(nodes),
            train=f"{fit.train_accuracy:.1f}%",
            unique=len(cumulative_ids),
        )
    ledger.require_unique_keys(("stage",))
    return OnlineInvocation(condition.condition_id, new_steps, new_stages)


def _full_rows(
    bootstrap: ReplayAdaptationBootstrap, partition: str, stage: int
) -> tuple[ImageRecord, ...]:
    return tuple(
        sorted(
            _partition_rows(
                bootstrap.integrator, partition, tuple(range(stage))
            ),
            key=lambda row: row.image_id,
        )
    )


def _full_tensors(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    partition: str,
    stage: int,
    device: torch.device,
) -> FrontierTensors:
    nodes, slots, _frontier_hash = _hierarchy_frontier(hierarchy, stage)
    rows = (
        _test_rows(bootstrap.integrator, tuple(range(stage)))
        if partition == "test"
        else _full_rows(bootstrap, partition, stage)
    )
    return _frontier_tensors(
        bootstrap.integrator,
        nodes,
        slots,
        bootstrap.integrator.config.maximum_levels,
        rows,
        device,
        f"replay_adaptation_{partition}_population",
    )


def _fresh_job_hash(
    bootstrap: ReplayAdaptationBootstrap, stage: int, restart: int
) -> str:
    return record_sha256(
        {
            "feature_variant": FEATURE_VARIANT,
            "protocol_hash": bootstrap.protocol.content_hash,
            "restart": restart,
            "schema_version": "imagenetr50-replay-adaptation-full-history-job-v1",
            "stage": stage,
        }
    )


def _fit_or_load_full_history(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    stage: int,
    restart: int,
    fitting: FrontierTensors,
    validation: FrontierTensors,
    device: torch.device,
) -> tuple[IntegratorState, IntegratorFitResult, Path, bool]:
    job_hash = _fresh_job_hash(bootstrap, stage, restart)
    name = f"fresh-full-history-stage{stage}-restart{restart}"
    state = create_integrator_state(
        name,
        bootstrap.integrator.config.maximum_levels,
        FEATURE_VARIANT,
        bootstrap.integrator.config.optimization,
        bootstrap.config.seed,
        device,
    )
    target = bootstrap.store.run / "integrators" / "fresh_full_history" / job_hash
    if target.is_dir():
        return state, load_integrator_fit(target, state), target, True
    _nodes, _slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
    fit_ids = record_sha256(list(fitting.image_ids))
    validation_ids = record_sha256(list(validation.image_ids))
    checkpoint = bootstrap.store.run / "work" / f"fresh_{job_hash}.pt"
    result = fit_fresh_integrator(
        state,
        _supervision(fitting, FEATURE_VARIANT),
        _supervision(validation, FEATURE_VARIANT),
        bootstrap.integrator.config.optimization,
        bootstrap.config.seed + restart,
        stage,
        device,
        progress=True,
        checkpoint_path=checkpoint,
        checkpoint_key=job_hash,
    )
    artifact = publish_integrator_fit(
        bootstrap.store,
        "fresh_full_history",
        job_hash,
        state,
        result,
        {
            "fit_ids_hash": fit_ids,
            "frontier_hash": frontier_hash,
            "restart": restart,
            "stage": stage,
            "validation_ids_hash": validation_ids,
            "variant": FEATURE_VARIANT,
        },
    )
    if checkpoint.is_file():
        checkpoint.unlink()
    return state, result, artifact, False


def _full_history_selection_path(
    bootstrap: ReplayAdaptationBootstrap, stage: int
) -> Path:
    return (
        bootstrap.store.run
        / "integrators"
        / "fresh_full_history"
        / f"stage_{stage:03d}_selection.json"
    )


def _full_history_candidate_rows(
    run_root: Path,
    candidates: tuple[
        tuple[IntegratorState, IntegratorFitResult, Path, bool], ...
    ],
) -> tuple[dict[str, object], ...]:
    """Describe scientific candidates without transient cache-reuse state."""
    return tuple(
        {
            "artifact": str(artifact.relative_to(run_root)),
            "fit": asdict(result),
            "restart": restart,
        }
        for restart, (_state, result, artifact, _reused) in enumerate(candidates)
    )


def _fit_full_history_restarts(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    stage: int,
    fitting: FrontierTensors,
    validation: FrontierTensors,
    device: torch.device,
) -> tuple[IntegratorState, IntegratorFitResult, dict[str, object], int]:
    selection_path = _full_history_selection_path(bootstrap, stage)
    candidates = tuple(
        _fit_or_load_full_history(
            bootstrap,
            hierarchy,
            stage,
            restart,
            fitting,
            validation,
            device,
        )
        for restart in range(bootstrap.config.full_history_restarts)
    )
    best_index = min(
        range(len(candidates)),
        key=lambda index: float(candidates[index][1].validation_loss),
    )
    rows = _full_history_candidate_rows(bootstrap.store.run, candidates)
    core: dict[str, object] = {
        "candidates": list(rows),
        "schema_version": "imagenetr50-replay-adaptation-full-history-selection-v1",
        "selected_artifact": rows[best_index]["artifact"],
        "selected_restart": best_index,
        "stage": stage,
    }
    record = {**core, "content_hash": record_sha256(core)}
    if selection_path.is_file():
        if load_canonical_json(selection_path) != record:
            raise ValueError("full-history selection changed on exact replay")
    else:
        publish_immutable_json(selection_path, record)
    selected = candidates[best_index]
    new_fits = sum(not reused for _state, _result, _artifact, reused in candidates)
    return selected[0], selected[1], record, new_fits


def _load_selected_full_history(
    bootstrap: ReplayAdaptationBootstrap,
    stage: int,
    device: torch.device,
) -> tuple[IntegratorState, IntegratorFitResult, dict[str, object]]:
    selection = load_canonical_json(_full_history_selection_path(bootstrap, stage))
    restart = int(selection["selected_restart"])
    state = create_integrator_state(
        f"fresh-full-history-stage{stage}-restart{restart}",
        bootstrap.integrator.config.maximum_levels,
        FEATURE_VARIANT,
        bootstrap.integrator.config.optimization,
        bootstrap.config.seed,
        device,
    )
    artifact = bootstrap.store.run / str(selection["selected_artifact"])
    return state, load_integrator_fit(artifact, state), selection


def _development_evaluation(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> tuple[int, int]:
    ledger = ChainedJsonlLedger(
        bootstrap.store.run / "evaluations" / "development_metrics.jsonl",
        "imagenetr50-replay-adaptation-development-v1",
    )
    completed = {
        (str(row["condition"]), int(row["stage"])) for row in ledger.rows
    }
    new_rows = new_full_fits = 0
    conditions = online_conditions(bootstrap.config)
    for stage in bootstrap.config.diagnostic_stages:
        fitting = _full_tensors(bootstrap, hierarchy, "fit", stage, device)
        validation = _full_tensors(
            bootstrap, hierarchy, "validation", stage, device
        )
        for condition in conditions:
            key = (condition.condition_id, stage)
            if key in completed:
                continue
            state = _restore_online_state(
                bootstrap, hierarchy, condition, stage, device
            )
            selected_training = next(
                dict(row["selected_training"])
                for row in ChainedJsonlLedger(
                    _condition_root(bootstrap, condition) / "training_metrics.jsonl",
                    "imagenetr50-replay-adaptation-online-stage-v1",
                ).rows
                if int(row["stage"]) == stage
            )
            ledger.append(
                {
                    "condition": condition.condition_id,
                    "fit_population": _evaluation_metrics(state, fitting, device),
                    "kind": "online",
                    "selected_training": selected_training,
                    "stage": stage,
                    "validation": _evaluation_metrics(state, validation, device),
                    "validation_controls": _control_metrics(validation),
                }
            )
            new_rows += 1
        full_state, full_fit, selection, fitted = _fit_full_history_restarts(
            bootstrap, hierarchy, stage, fitting, validation, device
        )
        new_full_fits += fitted
        key = (FULL_HISTORY_CONDITION, stage)
        if key not in completed:
            fit_metrics = _evaluation_metrics(full_state, fitting, device)
            ledger.append(
                {
                    "condition": FULL_HISTORY_CONDITION,
                    "fit_population": fit_metrics,
                    "full_history_fit": asdict(full_fit),
                    "full_history_selection": selection,
                    "kind": "offline_diagnostic",
                    "selected_training": fit_metrics,
                    "stage": stage,
                    "validation": _evaluation_metrics(
                        full_state, validation, device
                    ),
                    "validation_controls": _control_metrics(validation),
                }
            )
            new_rows += 1
        del fitting, validation
        torch.cuda.empty_cache()
    ledger.require_unique_keys(("condition", "stage"))
    return new_rows, new_full_fits


def _training_seal(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "training_seal.json"
    if target.is_file():
        return load_canonical_json(target)
    behavior_ledger = ChainedJsonlLedger(
        bootstrap.store.run / "ledgers" / "behavior_requests.jsonl",
        "imagenetr50-integrator-behavior-request-v1",
    )
    test_requests = sum("test" in row["splits"] for row in behavior_ledger.rows)
    if test_requests:
        raise RuntimeError("test behavior was opened before model selection sealed")
    checkpoints = {
        f"{condition.condition_id}:stage{stage}": file_sha256(
            _checkpoint_path(bootstrap, condition, stage)
        )
        for condition in online_conditions(bootstrap.config)
        for stage in bootstrap.config.diagnostic_stages
    }
    full_history = {
        str(stage): file_sha256(_full_history_selection_path(bootstrap, stage))
        for stage in bootstrap.config.diagnostic_stages
    }
    core: dict[str, object] = {
        "behavior_ledger_rows": len(behavior_ledger.rows),
        "behavior_ledger_tail_hash": behavior_ledger.tail_hash,
        "diagnostic_stages": list(bootstrap.config.diagnostic_stages),
        "full_history_selections": full_history,
        "hierarchy_policy_hash": hierarchy.policy.content_hash,
        "online_checkpoints": checkpoints,
        "schema_version": "imagenetr50-replay-adaptation-training-seal-v1",
        "test_requests_before_seal": test_requests,
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _test_evaluation(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> int:
    ledger = ChainedJsonlLedger(
        bootstrap.store.run / "evaluations" / "test_metrics.jsonl",
        "imagenetr50-replay-adaptation-test-v1",
    )
    completed = {
        (str(row["condition"]), int(row["stage"])) for row in ledger.rows
    }
    new_rows = 0
    for stage in bootstrap.config.diagnostic_stages:
        tensors = _full_tensors(bootstrap, hierarchy, "test", stage, device)
        states = tuple(
            (
                condition.condition_id,
                _restore_online_state(
                    bootstrap, hierarchy, condition, stage, device
                ),
            )
            for condition in online_conditions(bootstrap.config)
        )
        full_state, _fit, _selection_record = _load_selected_full_history(
            bootstrap, stage, device
        )
        for condition, state in (*states, (FULL_HISTORY_CONDITION, full_state)):
            if (condition, stage) in completed:
                continue
            ledger.append(
                {
                    "condition": condition,
                    "controls": _control_metrics(tensors),
                    "metrics": _evaluation_metrics(state, tensors, device),
                    "stage": stage,
                }
            )
            new_rows += 1
        del tensors
        torch.cuda.empty_cache()
    ledger.require_unique_keys(("condition", "stage"))
    return new_rows


def _preflight(
    bootstrap: ReplayAdaptationBootstrap, device: torch.device
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    train_ids = {
        row.image_id for row in bootstrap.integrator.manifest.images if row.split == "train"
    }
    test_ids = {
        row.image_id for row in bootstrap.integrator.manifest.images if row.split == "test"
    }
    static = _selection(bootstrap, 31, "static")
    rotating = _selection(bootstrap, 31, "rotating")
    replay_rotation = len(
        {row.image_id for row in static.historical}
        ^ {row.image_id for row in rotating.historical}
    )
    state = _new_online_state(bootstrap, device)
    toy_labels = torch.tensor((0, 1, 4, 5, 5), dtype=torch.int64)
    weights = task_uniform_weights(toy_labels)
    task_totals = tuple(
        float(weights[toy_labels // 4 == task].sum()) for task in (0, 1)
    )
    core: dict[str, object] = {
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "condition_count": len(online_conditions(bootstrap.config)),
        "fit_images": len(bootstrap.integrator.split.fit_image_ids),
        "input_dimensions": state.model.input_dim,
        "parameter_count": parameter_count(state.model),
        "replay_rotation_symmetric_difference_stage31": replay_rotation,
        "schema_version": "imagenetr50-replay-adaptation-preflight-v1",
        "task_uniform_weight_totals": list(task_totals),
        "test_images": len(test_ids),
        "test_train_overlap": len(test_ids & train_ids),
        "train_images": len(train_ids),
        "validation_images": len(bootstrap.integrator.split.validation_image_ids),
    }
    if (
        not core["bf16_supported"]
        or core["condition_count"] != 8
        or core["fit_images"] != 19_200
        or core["validation_images"] != 4_800
        or core["input_dimensions"] != 8_214
        or core["test_train_overlap"] != 0
        or replay_rotation <= 0
        or not math.isclose(task_totals[0], task_totals[1], abs_tol=1e-6)
    ):
        raise RuntimeError("replay-adaptation preflight failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _copy_source_references(bootstrap: ReplayAdaptationBootstrap) -> None:
    for filename in (
        "result.json",
        "source_locked_test.json",
        "stage_matched_joint_iid.json",
    ):
        source = bootstrap.behavior_run / "evaluations" / filename
        if source.is_file():
            publish_immutable_bytes(
                bootstrap.store.run / "evaluations" / f"source_{filename}",
                source.read_bytes(),
            )


def _publish_result(bootstrap: ReplayAdaptationBootstrap) -> dict[str, object]:
    target = bootstrap.store.run / "evaluations" / "result.json"
    if target.is_file():
        return load_canonical_json(target)
    development = ChainedJsonlLedger(
        bootstrap.store.run / "evaluations" / "development_metrics.jsonl",
        "imagenetr50-replay-adaptation-development-v1",
    )
    test = ChainedJsonlLedger(
        bootstrap.store.run / "evaluations" / "test_metrics.jsonl",
        "imagenetr50-replay-adaptation-test-v1",
    )
    source_result = load_canonical_json(
        bootstrap.behavior_run / "evaluations" / "result.json"
    )
    prior = next(
        dict(condition)
        for condition in source_result["conditions"]
        if int(condition["historical_capacity"]) == 8192
    )
    prior_rows = {
        int(row["stage"]): dict(row)
        for row in prior["stage_metrics"]
        if int(row["stage"]) in bootstrap.config.diagnostic_stages
    }
    joint_source = load_canonical_json(
        bootstrap.behavior_run / "evaluations" / "stage_matched_joint_iid.json"
    )
    joint = {
        int(row["stage"]): float(row["accuracy"])
        for row in joint_source["rows"]
        if int(row["stage"]) in bootstrap.config.diagnostic_stages
    }
    core: dict[str, object] = {
        "development_metrics": list(development.rows),
        "online_conditions": [
            asdict(condition) | {"condition_id": condition.condition_id}
            for condition in online_conditions(bootstrap.config)
        ],
        "permuted_mnist_reference_protocol": {
            "current_examples": 256,
            "current_source_weight": 0.5,
            "historical_examples": 256,
            "sampler": "uniform_without_replacement_when_possible",
            "seed_includes_macro_step": True,
        },
        "prior_all_train_h8192": [prior_rows[stage] for stage in sorted(prior_rows)],
        "protocol_hash": bootstrap.protocol.content_hash,
        "role": "post_hoc_replay_adaptation_diagnosis",
        "schema_version": "imagenetr50-replay-adaptation-result-v1",
        "stage_matched_joint_iid": joint,
        "test_metrics": list(test.rows),
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    _copy_source_references(bootstrap)
    return result


def _reuse_proof(
    bootstrap: ReplayAdaptationBootstrap,
    hierarchy: HierarchyBuildResult,
    source_before: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "reuse_proof.json"
    if target.is_file():
        return load_canonical_json(target)
    checkpoint_root = bootstrap.store.run / "integrators"
    before = _stat_fingerprint(checkpoint_root)
    invocations = tuple(
        _train_online_condition(bootstrap, hierarchy, condition, device)
        for condition in online_conditions(bootstrap.config)
    )
    development_rows, full_fits = _development_evaluation(
        bootstrap, hierarchy, device
    )
    test_rows = _test_evaluation(bootstrap, hierarchy, device)
    rebuilt = _build(bootstrap.source, "all_train", 50, device, progress=False)
    after = _stat_fingerprint(checkpoint_root)
    source_after = _stat_fingerprint(
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.hierarchy_policy_hash
    )
    checks = {
        "all_diagnostic_models_unchanged": before == after,
        "all_online_conditions_zero_new_optimizer_steps": all(
            invocation.new_optimizer_steps == 0 for invocation in invocations
        ),
        "all_online_conditions_zero_new_stages": all(
            invocation.new_stages == 0 for invocation in invocations
        ),
        "development_rows_unchanged": development_rows == 0,
        "full_history_fits_reused": full_fits == 0,
        "source_hierarchy_unchanged": dict(source_before) == source_after,
        "source_leaf_optimizer_steps_zero": rebuilt.work.leaf_optimizer_steps == 0,
        "source_parent_optimizer_steps_zero": rebuilt.work.parent_optimizer_steps == 0,
        "test_rows_unchanged": test_rows == 0,
    }
    core: dict[str, object] = {
        "acceptance": checks,
        "checkpoint_fingerprint_after": after,
        "checkpoint_fingerprint_before": before,
        "integrity_passed": all(checks.values()),
        "online_invocations": [asdict(invocation) for invocation in invocations],
        "schema_version": "imagenetr50-replay-adaptation-reuse-proof-v1",
        "source_hierarchy_after": source_after,
        "source_hierarchy_before": dict(source_before),
        "source_hierarchy_rebuild_work": asdict(rebuilt.work),
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def run_replay_adaptation(
    config_path: str | Path = DEFAULT_REPLAY_ADAPTATION_CONFIG,
) -> Path:
    """Run or exactly resume the complete replay-adaptation diagnosis."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the replay-adaptation diagnosis requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_replay_adaptation(config_path)
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=8, desc="replay adaptation diagnosis", unit="phase")
    print(f"Temporary/resumable artifact directory: {bootstrap.store.run}", flush=True)
    print("[phase 1/8] Authenticate protocol and paired replay semantics", flush=True)
    _write_state(bootstrap, "PREFLIGHT")
    _preflight(bootstrap, device)
    overall.update(1)

    print("[phase 2/8] Reuse hierarchy and seed training-only behavior cache", flush=True)
    source_hierarchy_root = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.hierarchy_policy_hash
    )
    source_before = _stat_fingerprint(source_hierarchy_root)
    hierarchy = _build(bootstrap.source, "all_train", 50, device, progress=False)
    if (
        hierarchy.policy.content_hash != bootstrap.config.hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("the fresh-parent hierarchy was not reused exactly")
    image_splits = {
        row.image_id: row.split for row in bootstrap.integrator.manifest.images
    }
    cache_seed = seed_row_cache_shards(
        bootstrap.behavior_run / "cache" / "behaviors",
        bootstrap.store.run / "cache" / "behaviors",
        image_splits,
        frozenset({"train"}),
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "train_cache_seed.json",
        _stable_cache_record(cache_seed),
    )
    overall.update(1)

    print("[phase 3/8] Train eight paired online replay conditions", flush=True)
    _write_state(bootstrap, "TRAIN_ONLINE")
    invocations = tuple(
        _train_online_condition(bootstrap, hierarchy, condition, device)
        for condition in online_conditions(bootstrap.config)
    )
    overall.update(1)

    print("[phase 4/8] Evaluate fit/validation and fit full-history diagnostics", flush=True)
    _write_state(bootstrap, "DEVELOPMENT_EVALUATION")
    development_rows, full_fits = _development_evaluation(
        bootstrap, hierarchy, device
    )
    overall.update(1)

    print("[phase 5/8] Seal every trained and validation-selected model", flush=True)
    seal = _training_seal(bootstrap, hierarchy)
    overall.update(1)

    print("[phase 6/8] Open locked test cache and evaluate stages 31 and 50", flush=True)
    _write_state(bootstrap, "LOCKED_TEST_EVALUATION", training_seal=seal)
    all_cache_seed = seed_row_cache_shards(
        bootstrap.behavior_run / "cache" / "behaviors",
        bootstrap.store.run / "cache" / "behaviors",
        image_splits,
        frozenset({"train", "test"}),
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "all_cache_seed.json",
        _stable_cache_record(all_cache_seed),
    )
    test_rows = _test_evaluation(bootstrap, hierarchy, device)
    result = _publish_result(bootstrap)
    overall.update(1)

    print("[phase 7/8] Prove exact zero-work replay", flush=True)
    proof = _reuse_proof(bootstrap, hierarchy, source_before, device)
    if not bool(proof["integrity_passed"]):
        raise RuntimeError("replay-adaptation reuse proof failed")
    overall.update(1)

    print("[phase 8/8] Generate Markdown, HTML, PDF, plots, and tables", flush=True)
    elapsed = time.monotonic() - started
    invocation_core: dict[str, object] = {
        "development_rows": development_rows,
        "elapsed_seconds": elapsed,
        "full_history_fits": full_fits,
        "online_invocations": [asdict(invocation) for invocation in invocations],
        "schema_version": "imagenetr50-replay-adaptation-invocation-v1",
        "test_rows": test_rows,
    }
    atomic_write(
        bootstrap.store.run / "state" / "last_invocation.json",
        canonical_json_bytes(
            {**invocation_core, "content_hash": record_sha256(invocation_core)}
        ),
    )
    _write_state(bootstrap, "COMPLETE", result=result, reuse_proof=proof)
    from apm.continual.vision.imagenetr.replay_adaptation_reporting import (
        write_replay_adaptation_report,
    )

    write_replay_adaptation_report(bootstrap.store.run)
    overall.update(1)
    overall.close()
    print(f"Replay-adaptation report complete in {elapsed:.1f} seconds.", flush=True)
    return bootstrap.store.run


if __name__ == "__main__":
    print(run_replay_adaptation())


__all__ = [
    "FULL_HISTORY_CONDITION",
    "OnlineCondition",
    "ReplayAdaptationBootstrap",
    "ReplayAdaptationProtocol",
    "ReplaySelection",
    "bootstrap_replay_adaptation",
    "online_conditions",
    "run_replay_adaptation",
]
