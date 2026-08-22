"""Config-driven phase-gated workflow for recursive ImageNet-R learned routing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import math
import time

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.protocol import JobSpec
from apm.continual.vision.imagenetr.router_artifacts import inference_inventory
from apm.continual.vision.imagenetr.router_evaluation import (
    RouterEvaluation,
    RouterNodeScoreProvider,
    centroid_scoring_nodes,
    evaluate_router_frontier,
)
from apm.continual.vision.imagenetr.router_experiment import (
    FlatRunResult,
    NodeFeatureRegistry,
    RecursiveRunResult,
    RouterBootstrap,
    bootstrap_router_protocol,
    condition_id,
    make_router_policy,
    run_flat_frontier,
    run_recursive_policy,
)
from apm.continual.vision.imagenetr.router_features import (
    RouterFeatureUniverse,
    load_router_feature_universe,
)
from apm.continual.vision.imagenetr.router_protocol import RouterPolicy
from apm.continual.vision.imagenetr.router_reporting import (
    load_router_evaluation,
    publish_router_evaluation,
    write_router_report,
)
from apm.continual.vision.imagenetr.router_scores import (
    RouterQuery,
    ScoringNode,
    move_scorer,
    score_nodes,
)
from apm.continual.vision.imagenetr.router_teacher import ImageNetRouterTeacher
from apm.continual.vision.imagenetr.router_training import RouterTrainingData
from apm.continual.vision.imagenetr.scheduler import LocalScheduler


DEFAULT_ROUTER_CONFIG = Path(
    "configs/vision/imagenetr/recursive_router_oracle_recovery_v1.yaml"
)


@dataclass(frozen=True, slots=True)
class MatrixCondition:
    """One predeclared matrix row and its canonical router policy."""

    matrix_id: str
    policy: RouterPolicy | None
    role: str
    conditional: bool = False

    def as_record(self) -> dict[str, object]:
        return {
            "conditional": self.conditional,
            "matrix_id": self.matrix_id,
            "policy": self.policy.as_record() if self.policy else None,
            "role": self.role,
        }


def _write_state(run: Path, phase: str, **values: object) -> None:
    atomic_write(
        run / "state" / "router_workflow_state.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "router_run_hash": run.name,
                "schema_version": "imagenetr50-router-workflow-state-v1",
                **values,
            }
        ),
    )


def _tree(bootstrap: RouterBootstrap, condition: str):
    try:
        return bootstrap.base.tree_map[condition]
    except KeyError as error:  # pragma: no cover - Phase-0 invariant
        raise ValueError(f"sealed inference tree is missing: {condition}") from error


def _matrix(bootstrap: RouterBootstrap, seed: int) -> tuple[MatrixCondition, ...]:
    config = bootstrap.config
    u100, svd0, svd5 = (
        _tree(bootstrap, value) for value in ("I-U100", "I-SVD0", "I-SVD5")
    )
    rows = [
        MatrixCondition("A0", None, "existing frozen-feature centroid baseline"),
        MatrixCondition("A1", make_router_policy(config, u100, "r0", "flat_full", seed), "capacity floor"),
        MatrixCondition("A2", make_router_policy(config, u100, "r1", "flat_full", seed), "descriptor capacity"),
        MatrixCondition("A3", make_router_policy(config, u100, "r3", "flat_full", seed), "adapter-response capacity"),
        MatrixCondition("A4", make_router_policy(config, u100, "r2", "flat_full", seed), "conditional nonlinear capacity", True),
        MatrixCondition("A5", make_router_policy(config, svd5, "r1", "flat_full", seed), "cheap-node descriptor brittleness"),
        MatrixCondition("A6", make_router_policy(config, svd5, "r3", "flat_full", seed), "cheap-node response brittleness"),
    ]
    for matrix_id, architecture, maintenance, role in (
        ("B0", "r1", "flat_seen_data", "descriptor causal-data ceiling"),
        ("B1", "r1", "exact", "descriptor functional oracle"),
        ("B2", "r1", "u100", "descriptor parent replay ceiling"),
        ("B3", "r1", "svd0", "descriptor zero-example merge"),
        ("B4", "r1", "svd5", "descriptor scalable condition"),
        ("B5", "r3", "flat_seen_data", "response causal-data ceiling"),
        ("B6", "r3", "exact", "response functional oracle"),
        ("B7", "r3", "u100", "response parent replay ceiling"),
        ("B8", "r3", "svd0", "response zero-example merge"),
        ("B9", "r3", "svd5", "response scalable headline"),
    ):
        rows.append(
            MatrixCondition(
                matrix_id,
                make_router_policy(config, u100, architecture, maintenance, seed),
                role,
            )
        )
    for matrix_id, tree, architecture, role in (
        ("C1", svd0, "r1", "descriptor cheap-inference transfer"),
        ("C2", svd5, "r1", "descriptor repaired-inference transfer"),
        ("C3", svd0, "r3", "response cheap-inference transfer"),
        ("C4", svd5, "r3", "full cheap inference/router condition"),
    ):
        rows.append(
            MatrixCondition(
                matrix_id,
                make_router_policy(config, tree, architecture, "svd5", seed),
                role,
            )
        )
    return tuple(rows)


def _publish_matrix(bootstrap: RouterBootstrap) -> tuple[MatrixCondition, ...]:
    rows = _matrix(bootstrap, bootstrap.config.seed)
    core: dict[str, object] = {
        "capacity_gate": {
            "fallback": "run A4 and stop before B if both A2 and A3 exceed a 1.0-point validation oracle gap",
            "main_rows": ["A0", "A1", "A2", "A3"],
        },
        "conditions": [row.as_record() for row in rows],
        "replication_rule": {
            "additional_seeds": list(bootstrap.config.router_seeds[1:]),
            "rows": ["B4", "B9", "C1", "C2", "C3", "C4"],
            "trigger": "B4 or B9 reaches 78.5% routed test accuracy with <=1.0-point oracle gap",
        },
        "router_run_hash": bootstrap.protocol.content_hash,
        "schema_version": "imagenetr50-recursive-router-matrix-v1",
    }
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "matrix.json",
        {**core, "content_hash": record_sha256(core)},
    )
    return rows


def _job(
    bootstrap: RouterBootstrap,
    kind: str,
    policy: RouterPolicy,
    task_count: int,
) -> JobSpec:
    return JobSpec.create(
        bootstrap.protocol.content_hash,
        kind,
        payload={
            "policy_hash": policy.content_hash,
            "task_count": task_count,
        },
    )


def _ensure_flat(
    scheduler: LocalScheduler,
    bootstrap: RouterBootstrap,
    policy: RouterPolicy,
    stage: int,
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
    device: torch.device,
) -> FlatRunResult:
    tree = _tree(bootstrap, policy.inference_condition)
    spec = _job(bootstrap, "router_flat", policy, stage)
    holder: dict[str, FlatRunResult] = {}

    def perform() -> Mapping[str, object]:
        result = run_flat_frontier(
            bootstrap, policy, tree, stage, data, features, device
        )
        holder["result"] = result
        return {
            "artifact": str(result.artifact_path),
            "epochs": result.epochs,
            "optimizer_steps": result.optimizer_steps,
            "reused": result.reused,
        }

    scheduler.execute(spec, perform)
    return holder.get("result") or run_flat_frontier(
        bootstrap, policy, tree, stage, data, features, device
    )


def _ensure_recursive(
    scheduler: LocalScheduler,
    bootstrap: RouterBootstrap,
    policy: RouterPolicy,
    task_count: int,
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
    device: torch.device,
) -> RecursiveRunResult:
    tree = _tree(bootstrap, policy.inference_condition)
    spec = _job(bootstrap, "router_recursive", policy, task_count)
    holder: dict[str, RecursiveRunResult] = {}

    def perform() -> Mapping[str, object]:
        result = run_recursive_policy(
            bootstrap,
            policy,
            tree,
            data,
            features,
            device,
            task_count,
            True,
        )
        holder["result"] = result
        return {
            "created_nodes": result.created_nodes,
            "optimizer_steps": result.optimizer_steps,
            "reused_nodes": result.reused_nodes,
            "stages": len(result.snapshots),
        }

    scheduler.execute(spec, perform)
    return holder.get("result") or run_recursive_policy(
        bootstrap,
        policy,
        tree,
        data,
        features,
        device,
        task_count,
    )


def _evaluation_target(run: Path, matrix_id: str, split: str, stage: int) -> Path:
    return run / "evaluations" / matrix_id / split / f"stage_{stage:03d}"


def _ensure_evaluation(
    bootstrap: RouterBootstrap,
    matrix_id: str,
    policy: RouterPolicy | None,
    split: str,
    stage: int,
    universe: RouterFeatureUniverse,
    image_ids: Sequence[str],
    nodes: Sequence[ScoringNode],
    score_provider: RouterNodeScoreProvider,
    device: torch.device,
) -> RouterEvaluation:
    target = _evaluation_target(bootstrap.store.run, matrix_id, split, stage)
    if target.is_dir():
        return load_router_evaluation(target)
    inference_condition = policy.inference_condition if policy else "I-U100"
    tree = _tree(bootstrap, inference_condition)
    inference_nodes = tree.snapshots[stage - 1].nodes
    result = evaluate_router_frontier(
        condition_id=matrix_id,
        inference_condition=inference_condition,
        architecture=policy.architecture if policy else "centroid",
        maintenance=policy.maintenance if policy else "existing",
        router_seed=policy.router_seed if policy else bootstrap.config.seed,
        split=split,
        stage=stage,
        universe=universe,
        image_ids=image_ids,
        scoring_nodes=nodes,
        inference_nodes=inference_nodes,
        score_provider=score_provider,
        device=device,
    )
    publish_router_evaluation(bootstrap.store.run, result)
    return result


def _selection_accuracy(
    data: RouterTrainingData,
    nodes: Sequence[ScoringNode],
    stage: int,
    device: torch.device,
) -> float:
    ids = data.ids("validation", stage)
    query, labels, tasks = data.batch(ids, device)
    for node in nodes:
        move_scorer(node.scorer, device)
    values = score_nodes(query, nodes).detach().cpu()
    classes = tuple((node.node_id, node.represented_class_ids) for node in nodes)
    targets = ImageNetRouterTeacher().target_indices(
        stage,
        query.image_ids,
        labels.detach().cpu(),
        tasks.detach().cpu(),
        classes,
    )
    result = 100.0 * float((torch.argmax(values, dim=-1) == targets).float().mean())
    for node in nodes:
        move_scorer(node.scorer, torch.device("cpu"))
    return result


def _phase0_preflight(
    bootstrap: RouterBootstrap,
    train: RouterFeatureUniverse,
    elapsed: float,
) -> dict[str, object]:
    cache_files = tuple(
        (bootstrap.store.run / "features" / "cls_activations").glob(
            "*/tensors.safetensors"
        )
    )
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or len(train.image_ids) != 24_000
        or len(train.cls_activations) != 8
        or set(bootstrap.split.fit_image_ids) & set(bootstrap.split.validation_image_ids)
    ):
        raise RuntimeError("router real-model/data/GPU preflight failed")
    core: dict[str, object] = {
        "activation_cache_bytes": sum(path.stat().st_size for path in cache_files),
        "activation_cache_seconds": elapsed,
        "activation_rows_per_second": len(train.image_ids) / max(elapsed, 1.0e-9),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cuda_device": torch.cuda.get_device_name(0),
        "inference_inventory_hash": bootstrap.base.inventory_hash,
        "response_modules": sorted(train.cls_activations),
        "router_fit_images": len(bootstrap.split.fit_image_ids),
        "router_validation_images": len(bootstrap.split.validation_image_ids),
        "schema_version": "imagenetr50-router-real-preflight-v1",
        "test_images_used": 0,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "router_preflight.json", record
    )
    return record


def _run_smoke(
    scheduler: LocalScheduler,
    bootstrap: RouterBootstrap,
    matrix: Mapping[str, MatrixCondition],
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
    device: torch.device,
) -> dict[str, object]:
    path = bootstrap.store.run / "diagnostics" / "smoke.json"
    if path.is_file():
        return load_canonical_json(path)
    stage = bootstrap.config.smoke_tasks
    flat = _ensure_flat(
        scheduler,
        bootstrap,
        matrix["A1"].policy,  # type: ignore[arg-type]
        stage,
        data,
        features,
        device,
    )
    selection = {"A1": _selection_accuracy(data, flat.nodes, stage, device)}
    recursive = {}
    for matrix_id in ("B1", "B2", "B3", "B4", "B6", "B7", "B8", "B9"):
        policy = matrix[matrix_id].policy
        assert policy is not None
        result = _ensure_recursive(
            scheduler,
            bootstrap,
            policy,
            stage,
            data,
            features,
            device,
        )
        value = _selection_accuracy(data, result.stage_frontiers[-1], stage, device)
        if not math.isfinite(value):
            raise RuntimeError(f"smoke router selection is non-finite: {matrix_id}")
        selection[matrix_id] = value
        recursive[matrix_id] = {
            "created_nodes": result.created_nodes,
            "optimizer_steps": result.optimizer_steps,
            "reused_nodes": result.reused_nodes,
        }
    exact_errors = [
        row["functional"]["mean_mass_error"]
        for matrix_id in ("B1", "B6")
        for row in _ensure_recursive(
            scheduler,
            bootstrap,
            matrix[matrix_id].policy,  # type: ignore[arg-type]
            stage,
            data,
            features,
            device,
        ).merge_diagnostics
    ]
    if exact_errors and max(exact_errors) > 1.0e-6:
        raise RuntimeError("exact-LSE smoke did not preserve collapsed frontier mass")
    if recursive["B4"]["optimizer_steps"] <= 0 or recursive["B9"]["optimizer_steps"] <= 0:
        raise RuntimeError("five-percent repair smoke performed no router optimizer work")
    core: dict[str, object] = {
        "exact_max_mass_error": max(exact_errors, default=0.0),
        "inference_inventory_hash": bootstrap.base.inventory_hash,
        "inference_optimizer_steps": 0,
        "leaf_optimizer_steps": 0,
        "recursive": recursive,
        "schema_version": "imagenetr50-router-eight-task-smoke-v1",
        "selection_accuracy": selection,
        "stage": stage,
        "status": "PASS",
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(path, record)
    return record


def _recursive_validation(
    bootstrap: RouterBootstrap,
    matrix_id: str,
    condition: MatrixCondition,
    result: RecursiveRunResult,
    train: RouterFeatureUniverse,
    data: RouterTrainingData,
    provider: RouterNodeScoreProvider,
    device: torch.device,
) -> dict[int, RouterEvaluation]:
    evaluations = {}
    for stage in bootstrap.config.evaluation_checkpoints:
        evaluations[stage] = _ensure_evaluation(
            bootstrap,
            matrix_id,
            condition.policy,
            "validation",
            stage,
            train,
            data.ids("validation", stage),
            result.stage_frontiers[stage - 1],
            provider,
            device,
        )
        write_router_report(
            bootstrap.store.run,
            status={"completed_validation_checkpoint": stage, "condition": matrix_id},
        )
    return evaluations


def _coherent(
    metrics: Mapping[str, RouterEvaluation],
) -> tuple[bool, dict[str, object]]:
    rows: dict[str, object] = {}
    passed = True
    for architecture, flat_id, replay_id, scalable_id in (
        ("r1", "B0", "B2", "B4"),
        ("r3", "B5", "B7", "B9"),
    ):
        flat = metrics[flat_id].metric.routed_accuracy
        replay = metrics[replay_id].metric.routed_accuracy
        scalable = metrics[scalable_id].metric.routed_accuracy
        replay_gap = flat - replay
        scalable_gap = replay - scalable
        baseline = metrics["A0"].metric.routed_accuracy
        replay_gain = replay - baseline
        scalable_gain = scalable - baseline
        recovery = 1.0 if replay_gain <= 0 else scalable_gain / replay_gain
        architecture_pass = replay_gap <= 1.0 and (
            scalable_gap <= 1.0 or recovery >= 0.95
        )
        rows[architecture] = {
            "flat_accuracy": flat,
            "passed": architecture_pass,
            "replay_accuracy": replay,
            "replay_to_scalable_gap": scalable_gap,
            "scalable_accuracy": scalable,
            "scalable_gain_recovery": recovery,
        }
        passed = passed and architecture_pass
    return passed, rows


def _all_stage_test(
    bootstrap: RouterBootstrap,
    matrix_id: str,
    condition: MatrixCondition,
    stage_nodes: Sequence[Sequence[ScoringNode]],
    test: RouterFeatureUniverse,
    provider: RouterNodeScoreProvider,
    device: torch.device,
) -> tuple[RouterEvaluation, ...]:
    return tuple(
        _ensure_evaluation(
            bootstrap,
            matrix_id,
            condition.policy,
            "test",
            stage,
            test,
            tuple(
                image_id
                for image_id, task in zip(test.image_ids, test.task_ids.tolist())
                if task < stage
            ),
            stage_nodes[stage - 1],
            provider,
            device,
        )
        for stage in range(1, len(stage_nodes) + 1)
    )


def run_router_workflow(config_path: str | Path = DEFAULT_ROUTER_CONFIG) -> Path:
    """Run Phase 0, smoke, gated matrix, sealed test evaluation, reuse proof, and report."""
    bootstrap = bootstrap_router_protocol(config_path)
    run = bootstrap.store.run
    matrix_rows = _publish_matrix(bootstrap)
    matrix = {row.matrix_id: row for row in matrix_rows}
    scheduler = LocalScheduler(run / "state" / "scheduler_state.json", run.name)
    if not torch.cuda.is_available():
        raise RuntimeError("the local recursive-router workflow requires the RTX-class CUDA GPU")
    device = torch.device("cuda:0")
    base_template = create_pinned_backbone(bootstrap.checkpoint)
    backbone_factory: Callable[[], torch.nn.Module] = lambda: deepcopy(base_template)

    _write_state(run, "FEATURE_PREFLIGHT")
    started = time.monotonic()
    train = load_router_feature_universe(
        bootstrap.store,
        bootstrap.base.run_root,
        bootstrap.base.manifest,
        str(bootstrap.base.protocol_record["model_manifest_hash"]),
        "train",
        True,
        bootstrap.config,
        backbone_factory,
        bootstrap.base.prepared_root,
        bootstrap.test_transform,
        device,
    )
    _phase0_preflight(bootstrap, train, time.monotonic() - started)
    data = RouterTrainingData(train, bootstrap.split)
    features = NodeFeatureRegistry(bootstrap.store, bootstrap.config)

    _write_state(run, "SMOKE")
    smoke = _run_smoke(scheduler, bootstrap, matrix, data, features, device)
    if smoke["status"] != "PASS":
        raise RuntimeError("eight-task learned-router smoke did not pass")

    provider = RouterNodeScoreProvider(
        bootstrap.base,
        bootstrap.store,
        bootstrap.config,
        bootstrap.split.validation_image_ids,
        backbone_factory,
        bootstrap.test_transform,
        device,
    )
    _write_state(run, "CAPACITY_GATE")
    capacity: dict[str, tuple[FlatRunResult | None, RouterEvaluation]] = {}
    final_u100 = _tree(bootstrap, "I-U100").final.nodes
    centroid_features = {
        node.logical_node.node_id: features.get(node, "r1") for node in final_u100
    }
    centroid = centroid_scoring_nodes(
        train, bootstrap.split.fit_image_ids, final_u100, centroid_features
    )
    capacity["A0"] = (
        None,
        _ensure_evaluation(
            bootstrap,
            "A0",
            None,
            "validation",
            50,
            train,
            bootstrap.split.validation_image_ids,
            centroid,
            provider,
            device,
        ),
    )
    for matrix_id in ("A1", "A2", "A3"):
        policy = matrix[matrix_id].policy
        assert policy is not None
        flat = _ensure_flat(
            scheduler, bootstrap, policy, 50, data, features, device
        )
        evaluation = _ensure_evaluation(
            bootstrap,
            matrix_id,
            policy,
            "validation",
            50,
            train,
            bootstrap.split.validation_image_ids,
            flat.nodes,
            provider,
            device,
        )
        capacity[matrix_id] = (flat, evaluation)
    gate_open = any(
        capacity[matrix_id][1].metric.oracle_gap <= 1.0
        for matrix_id in ("A2", "A3")
    )
    gate_core: dict[str, object] = {
        "gate_open": gate_open,
        "rows": {
            key: value[1].metric.as_record() for key, value in capacity.items()
        },
        "schema_version": "imagenetr50-router-capacity-gate-v1",
    }
    publish_immutable_json(
        run / "diagnostics" / "capacity_gate.json",
        {**gate_core, "content_hash": record_sha256(gate_core)},
    )
    if not gate_open:
        policy = matrix["A4"].policy
        assert policy is not None
        flat = _ensure_flat(
            scheduler, bootstrap, policy, 50, data, features, device
        )
        capacity["A4"] = (
            flat,
            _ensure_evaluation(
                bootstrap,
                "A4",
                policy,
                "validation",
                50,
                train,
                bootstrap.split.validation_image_ids,
                flat.nodes,
                provider,
                device,
            ),
        )
        publish_immutable_json(
            run / "state" / "matrix_sealed.json",
            {
                "branch": "capacity_failure",
                "content_hash": record_sha256(
                    {"branch": "capacity_failure", "schema_version": "imagenetr50-router-matrix-seal-v1"}
                ),
                "schema_version": "imagenetr50-router-matrix-seal-v1",
            },
        )
        write_router_report(run, status=scheduler.summary())
        _write_state(run, "COMPLETE_CAPACITY_FAILURE", gate_open=False)
        return run

    for matrix_id in ("A5", "A6"):
        policy = matrix[matrix_id].policy
        assert policy is not None
        flat = _ensure_flat(
            scheduler, bootstrap, policy, 50, data, features, device
        )
        capacity[matrix_id] = (
            flat,
            _ensure_evaluation(
                bootstrap,
                matrix_id,
                policy,
                "validation",
                50,
                train,
                bootstrap.split.validation_image_ids,
                flat.nodes,
                provider,
                device,
            ),
        )

    _write_state(run, "PRIMARY_RECURSIVE")
    flat_seen: dict[str, tuple[FlatRunResult, ...]] = {}
    recursive: dict[str, RecursiveRunResult] = {}
    final_validation: dict[str, RouterEvaluation] = {}
    for matrix_id in ("B0", "B5"):
        policy = matrix[matrix_id].policy
        assert policy is not None
        from tqdm.auto import tqdm

        runs = tuple(
            _ensure_flat(
                scheduler, bootstrap, policy, stage, data, features, device
            )
            for stage in tqdm(
                range(1, 51),
                desc=f"{policy.inference_condition} {policy.architecture}/flat_seen_data",
                unit="stage",
            )
        )
        flat_seen[matrix_id] = runs
        for stage in bootstrap.config.evaluation_checkpoints:
            evaluation = _ensure_evaluation(
                bootstrap,
                matrix_id,
                policy,
                "validation",
                stage,
                train,
                data.ids("validation", stage),
                runs[stage - 1].nodes,
                provider,
                device,
            )
            if stage == 50:
                final_validation[matrix_id] = evaluation
    for matrix_id in ("B1", "B2", "B3", "B4", "B6", "B7", "B8", "B9"):
        condition = matrix[matrix_id]
        assert condition.policy is not None
        result = _ensure_recursive(
            scheduler,
            bootstrap,
            condition.policy,
            50,
            data,
            features,
            device,
        )
        recursive[matrix_id] = result
        evaluated = _recursive_validation(
            bootstrap,
            matrix_id,
            condition,
            result,
            train,
            data,
            provider,
            device,
        )
        final_validation[matrix_id] = evaluated[50]
    final_validation["A0"] = capacity["A0"][1]
    coherent, coherence_rows = _coherent(final_validation)
    coherence_core: dict[str, object] = {
        "passed": coherent,
        "rows": coherence_rows,
        "schema_version": "imagenetr50-router-coherence-gate-v1",
    }
    publish_immutable_json(
        run / "diagnostics" / "coherence_gate.json",
        {**coherence_core, "content_hash": record_sha256(coherence_core)},
    )
    if not coherent:
        write_router_report(run, status=scheduler.summary())
        _write_state(run, "STOPPED_COHERENCE_GATE", coherence=coherence_rows)
        return run

    _write_state(run, "TRANSFER")
    for matrix_id in ("C1", "C2", "C3", "C4"):
        condition = matrix[matrix_id]
        assert condition.policy is not None
        result = _ensure_recursive(
            scheduler,
            bootstrap,
            condition.policy,
            50,
            data,
            features,
            device,
        )
        recursive[matrix_id] = result
        _recursive_validation(
            bootstrap,
            matrix_id,
            condition,
            result,
            train,
            data,
            provider,
            device,
        )

    seal_core: dict[str, object] = {
        "branch": "primary_matrix",
        "completed_conditions": sorted((*capacity, *flat_seen, *recursive)),
        "inference_inventory_hash": bootstrap.base.inventory_hash,
        "schema_version": "imagenetr50-router-matrix-seal-v1",
    }
    publish_immutable_json(
        run / "state" / "matrix_sealed.json",
        {**seal_core, "content_hash": record_sha256(seal_core)},
    )

    _write_state(run, "SEALED_TEST")
    test = load_router_feature_universe(
        bootstrap.store,
        bootstrap.base.run_root,
        bootstrap.base.manifest,
        str(bootstrap.base.protocol_record["model_manifest_hash"]),
        "test",
        True,
        bootstrap.config,
        backbone_factory,
        bootstrap.base.prepared_root,
        bootstrap.test_transform,
        device,
    )
    test_provider = RouterNodeScoreProvider(
        bootstrap.base,
        bootstrap.store,
        bootstrap.config,
        bootstrap.split.validation_image_ids,
    )
    test_final: dict[str, RouterEvaluation] = {}
    for matrix_id, (flat, _validation) in capacity.items():
        policy = matrix[matrix_id].policy
        nodes = centroid if matrix_id == "A0" else flat.nodes  # type: ignore[union-attr]
        result = _ensure_evaluation(
            bootstrap,
            matrix_id,
            policy,
            "test",
            50,
            test,
            test.image_ids,
            nodes,
            test_provider,
            device,
        )
        test_final[matrix_id] = result
    for matrix_id, runs in flat_seen.items():
        evaluations = _all_stage_test(
            bootstrap,
            matrix_id,
            matrix[matrix_id],
            tuple(run.nodes for run in runs),
            test,
            test_provider,
            device,
        )
        test_final[matrix_id] = evaluations[-1]
    for matrix_id, result in recursive.items():
        evaluations = _all_stage_test(
            bootstrap,
            matrix_id,
            matrix[matrix_id],
            result.stage_frontiers,
            test,
            test_provider,
            device,
        )
        test_final[matrix_id] = evaluations[-1]

    triggered = any(
        test_final[matrix_id].metric.routed_accuracy >= 78.5
        and test_final[matrix_id].metric.oracle_gap <= 1.0
        for matrix_id in ("B4", "B9")
    )
    replication_rows: dict[str, RecursiveRunResult] = {}
    if triggered:
        _write_state(run, "PREDECLARED_REPLICATION")
        for seed in bootstrap.config.router_seeds[1:]:
            seed_matrix = {row.matrix_id: row for row in _matrix(bootstrap, seed)}
            for base_id in ("B4", "B9", "C1", "C2", "C3", "C4"):
                condition = seed_matrix[base_id]
                assert condition.policy is not None
                result = _ensure_recursive(
                    scheduler,
                    bootstrap,
                    condition.policy,
                    50,
                    data,
                    features,
                    device,
                )
                replicated_id = f"{base_id}-S{seed}"
                replication_rows[replicated_id] = result
                _all_stage_test(
                    bootstrap,
                    replicated_id,
                    condition,
                    result.stage_frontiers,
                    test,
                    test_provider,
                    device,
                )

    _write_state(run, "REUSE_PROOF")
    reuse_rows = {}
    for matrix_id in ("B4", "B9"):
        policy = matrix[matrix_id].policy
        assert policy is not None
        result = run_recursive_policy(
            bootstrap,
            policy,
            _tree(bootstrap, policy.inference_condition),
            data,
            features,
            device,
            50,
        )
        if result.created_nodes != 0 or result.reused_nodes != 92:
            raise RuntimeError("router reuse demonstration repeated completed node work")
        reuse_rows[matrix_id] = {
            "created_nodes": result.created_nodes,
            "reused_nodes": result.reused_nodes,
            "router_optimizer_steps_reexecuted": 0,
        }
    after = inference_inventory(bootstrap.base)
    before = load_canonical_json(run / "protocol" / "inference_inventory_before.json")
    if after != before:
        raise RuntimeError("sealed inference inventory changed during router-only execution")
    publish_immutable_json(run / "protocol" / "inference_inventory_after.json", after)
    reuse_core: dict[str, object] = {
        "inference_inventory_unchanged": True,
        "inference_parent_optimizer_steps": 0,
        "leaf_optimizer_steps": 0,
        "policies": reuse_rows,
        "replication_triggered": triggered,
        "schema_version": "imagenetr50-router-reuse-proof-v1",
    }
    publish_immutable_json(
        run / "diagnostics" / "reuse_proof.json",
        {**reuse_core, "content_hash": record_sha256(reuse_core)},
    )
    report = write_router_report(run, status=scheduler.summary())
    _write_state(
        run,
        "COMPLETE",
        report=str(report),
        replication_triggered=triggered,
    )
    return run


__all__ = [
    "DEFAULT_ROUTER_CONFIG",
    "MatrixCondition",
    "run_router_workflow",
]
