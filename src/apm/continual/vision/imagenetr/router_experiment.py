"""Authenticated bootstrap and resumable flat/recursive learned-router execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping, Sequence
import shutil
import tempfile

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.bank import MergeEvent, simulate_topology
from apm.continual.vision.imagenetr.data import image_transforms
from apm.continual.vision.imagenetr.manifests import (
    installed_environment_manifest,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import (
    InferenceNodeRef,
    RouterStore,
    SealedInferenceTree,
    SealedRouterBase,
    build_router_split,
    inference_inventory,
    load_router_node,
    load_router_snapshot,
    load_sealed_router_base,
    publish_router_node,
    publish_router_snapshot,
    router_split_from_record,
)
from apm.continual.vision.imagenetr.router_config import RouterConfig, load_router_config
from apm.continual.vision.imagenetr.router_descriptor import (
    NodeRouterFeatures,
    descriptor_config_record,
    load_or_build_node_features,
    response_config_record,
)
from apm.continual.vision.imagenetr.router_features import RouterFeatureUniverse
from apm.continual.vision.imagenetr.router_merge import (
    functional_merge_diagnostics,
    svd_merge_scorers,
)
from apm.continual.vision.imagenetr.router_protocol import (
    RouterNodeArtifact,
    RouterPolicy,
    RouterProtocol,
    RouterSplit,
    RouterStageSnapshot,
    router_node_from_record,
)
from apm.continual.vision.imagenetr.router_scores import (
    ExactLSEScorer,
    RouterQuery,
    ScoringNode,
    make_scorer,
)
from apm.continual.vision.imagenetr.router_training import (
    RouterTrainingData,
    RouterTrainingResult,
    fit_flat_frontier,
    fit_new_leaf,
    fit_parent,
    negative_reservoirs,
    repair_reservoir,
)


ROUTER_PACKAGES = (
    "apm",
    "numpy",
    "pandas",
    "Pillow",
    "pyarrow",
    "PyYAML",
    "safetensors",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
)


@dataclass(frozen=True, slots=True)
class RouterBootstrap:
    """All frozen inputs and paths for one learned-router protocol."""

    project_root: Path
    config_path: Path
    config: RouterConfig
    base: SealedRouterBase
    split: RouterSplit
    protocol: RouterProtocol
    store: RouterStore
    checkpoint: Path
    test_transform: object
    code_manifest: dict[str, object]
    environment_manifest: dict[str, object]
    source_drift_audit: dict[str, object]


@dataclass(frozen=True, slots=True)
class FlatRunResult:
    """One immutable independently fitted frontier."""

    policy: RouterPolicy
    stage: int
    nodes: tuple[ScoringNode, ...]
    optimizer_steps: int
    epochs: int
    best_validation_loss: float
    artifact_path: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class RecursiveRunResult:
    """All durable frontiers and work evidence for one recursive policy."""

    policy: RouterPolicy
    stage_frontiers: tuple[tuple[ScoringNode, ...], ...]
    snapshots: tuple[RouterStageSnapshot, ...]
    merge_diagnostics: tuple[dict[str, object], ...]
    optimizer_steps: int
    reused_nodes: int
    created_nodes: int


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parents[3]


def _material_router_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    shared = project_root / "src/apm/continual/artifacts.py"
    names = (
        "artifacts.py",
        "bank.py",
        "checkpoints.py",
        "config.py",
        "constants.py",
        "data.py",
        "evaluation.py",
        "heads.py",
        "lora.py",
        "manifests.py",
        "merging/common.py",
        "model.py",
        "protocol.py",
        "proxy_memory.py",
        "router_artifacts.py",
        "router_config.py",
        "router_descriptor.py",
        "router_evaluation.py",
        "router_experiment.py",
        "router_features.py",
        "router_merge.py",
        "router_protocol.py",
        "router_reporting.py",
        "router_scores.py",
        "router_teacher.py",
        "router_training.py",
        "router_workflow.py",
        "routing.py",
        "scheduler.py",
    )
    existing = tuple(package / name for name in names if (package / name).is_file())
    script = project_root / "scripts/vision/imagenetr/run_router_local.sh"
    optional = (script,) if script.is_file() else ()
    return (shared, config_path.resolve(), *existing, *optional)


def _source_drift(project_root: Path, base: SealedRouterBase, handoff: str) -> dict[str, object]:
    sealed = load_canonical_json(base.run_root / "protocol" / "code_manifest.json")
    rows = []
    for row in sealed["files"]:
        path = project_root / str(row["path"])
        current = file_sha256(path) if path.is_file() else None
        status = "unchanged" if current == row["sha256"] else ("missing" if current is None else "changed")
        rows.append(
            {
                "current_sha256": current,
                "path": row["path"],
                "sealed_sha256": row["sha256"],
                "status": status,
            }
        )
    core: dict[str, object] = {
        "handoff_commit": handoff,
        "rows": rows,
        "schema_version": "imagenetr50-router-source-drift-audit-v1",
        "sealed_code_manifest_hash": sealed["content_hash"],
    }
    return {**core, "content_hash": record_sha256(core)}


def _checkpoint(base: SealedRouterBase, config: RouterConfig) -> Path:
    model = load_canonical_json(base.run_root / "protocol" / "model_manifest.json")
    candidates = tuple((config.data_root / "model_cache").rglob(str(model["sha256"])))
    candidates = tuple(path for path in candidates if path.is_file())
    if len(candidates) != 1 or file_sha256(candidates[0]) != model["sha256"]:
        raise FileNotFoundError("the sealed pinned backbone checkpoint is not locally authenticated")
    return candidates[0]


def bootstrap_router_protocol(config_path: str | Path) -> RouterBootstrap:
    """Authenticate sealed dependencies and freeze the isolated router protocol."""
    source = Path(config_path).resolve()
    project_root = _project_root(source)
    config = load_router_config(source)
    base = load_sealed_router_base(config)
    split = build_router_split(base.manifest, config.fit_fraction, config.seed)
    environment = installed_environment_manifest(ROUTER_PACKAGES)
    missing = tuple(
        row["name"] for row in environment["packages"] if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    code = material_tree_manifest(_material_router_paths(project_root, source))
    protocol = RouterProtocol(
        config.sealed_run_hash,
        base.manifest.content_hash,
        str(base.protocol_record["model_manifest_hash"]),
        base.inventory_hash,
        config.inference_policies,
        split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    store = RouterStore(config.artifact_root, protocol.content_hash)
    drift = _source_drift(project_root, base, config.handoff_commit)
    link_core: dict[str, object] = {
        "authenticated_inference_inventory_hash": base.inventory_hash,
        "handoff_commit": config.handoff_commit,
        "inference_policies": [list(value) for value in config.inference_policies],
        "router_protocol_hash": protocol.content_hash,
        "schema_version": "imagenetr50-router-protocol-link-v1",
        "sealed_dataset_hash": base.manifest.content_hash,
        "sealed_model_hash": base.protocol_record["model_manifest_hash"],
        "sealed_run_hash": config.sealed_run_hash,
        "source_drift_audit_hash": drift["content_hash"],
    }
    store.prepare(protocol, split, {**link_core, "content_hash": record_sha256(link_core)})
    protocol_root = store.run / "protocol"
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("inference_inventory_before.json", inference_inventory(base)),
        ("source_drift_audit.json", drift),
    ):
        publish_immutable_json(protocol_root / filename, record)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required by the vision environment") from error
    publish_immutable_bytes(
        store.run / "config_resolved.yaml",
        yaml.safe_dump(config.as_record(), sort_keys=True).encode("utf-8"),
    )
    audit_core: dict[str, object] = {
        "inference_inventory_hash": base.inventory_hash,
        "missing_dependencies": [],
        "router_fit_images": len(split.fit_image_ids),
        "router_validation_images": len(split.validation_image_ids),
        "schema_version": "imagenetr50-router-phase0-audit-v1",
        "sealed_inference_optimizer_steps_requested": 0,
        "sealed_leaf_optimizer_steps_requested": 0,
        "source_changed_files": sum(row["status"] == "changed" for row in drift["rows"]),
        "source_missing_files": sum(row["status"] == "missing" for row in drift["rows"]),
        "trees": [
            {
                "condition": tree.condition,
                "nodes": len(tree.nodes),
                "policy_hash": tree.policy_hash,
                "snapshots": len(tree.snapshots),
            }
            for tree in base.trees
        ],
    }
    audit = {**audit_core, "content_hash": record_sha256(audit_core)}
    publish_immutable_json(protocol_root / "phase0_audit.json", audit)
    phase0 = (
        "# ImageNet-R-50 learned-router Phase 0\n\n"
        f"- Router protocol: `{protocol.content_hash}`\n"
        f"- Sealed inference run: `{config.sealed_run_hash}`\n"
        f"- Inference inventory: `{base.inventory_hash}`\n"
        f"- Router fit/validation: {len(split.fit_image_ids):,}/{len(split.validation_image_ids):,}\n"
        f"- Authenticated trees: {', '.join(tree.condition for tree in base.trees)}\n"
        "- Requested inference and leaf optimizer steps: 0\n"
        f"- Sealed-source drift: {audit_core['source_changed_files']} changed, "
        f"{audit_core['source_missing_files']} missing of {len(drift['rows'])} material files.\n"
    )
    publish_immutable_bytes(protocol_root / "PHASE0.md", phase0.encode("utf-8"))
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-router-latest-run-v1",
            }
        ),
    )
    _train_transform, test_transform = image_transforms(base.primary_config.input_size)
    return RouterBootstrap(
        project_root,
        source,
        config,
        base,
        split,
        protocol,
        store,
        _checkpoint(base, config),
        test_transform,
        code,
        environment,
        drift,
    )


def latest_router_run(config_path: str | Path) -> tuple[RouterConfig, Path]:
    """Resolve the latest prepared router run without mutating it."""
    config = load_router_config(config_path)
    latest = load_canonical_json(config.artifact_root / "LATEST_RUN.json")
    if latest.get("schema_version") != "imagenetr50-router-latest-run-v1":
        raise ValueError("unknown router latest-run record")
    run_hash = str(latest["run_hash"])
    return config, config.artifact_root / "runs" / run_hash


def make_router_policy(
    config: RouterConfig,
    tree: SealedInferenceTree,
    architecture: str,
    maintenance: str,
    router_seed: int,
) -> RouterPolicy:
    """Create one canonical matrix-policy identity."""
    rank = 1 if architecture == "r0" else config.primary_rank
    repair = config.primary_repair_fraction if maintenance == "svd5" else 0.0
    training_record = {
        **asdict(config.training),
        "schema_version": "imagenetr50-router-training-config-v1",
    }
    return RouterPolicy(
        tree.condition,
        tree.policy_hash,
        architecture,
        rank,
        maintenance,
        repair,
        router_seed,
        record_sha256(descriptor_config_record(config)),
        record_sha256(response_config_record(config)),
        record_sha256(training_record),
    )


def condition_id(policy: RouterPolicy) -> str:
    """Return a stable readable condition name without replacing content identity."""
    return (
        f"{policy.inference_condition}_{policy.architecture}_"
        f"{policy.maintenance}_seed{policy.router_seed}"
    )


def _node_seed(policy: RouterPolicy, logical_node_id: str, purpose: str) -> int:
    return int(
        sha256(
            f"imagenetr50-router-node-seed-v1\0{policy.router_seed}\0{logical_node_id}\0{purpose}".encode()
        ).hexdigest()[:15],
        16,
    )


class NodeFeatureRegistry:
    """Memoize authenticated descriptor/kernel features across matrix conditions."""

    def __init__(self, store: RouterStore, config: RouterConfig) -> None:
        self.store = store
        self.config = config
        self._values: dict[tuple[str, bool], NodeRouterFeatures] = {}

    def get(self, node: InferenceNodeRef, architecture: str) -> NodeRouterFeatures:
        include_response = architecture == "r3"
        key = (node.node_hash, include_response)
        if key not in self._values:
            self._values[key] = load_or_build_node_features(
                self.store, node, self.config, include_response
            )
        return self._values[key]


def _source_fit_count(
    data: RouterTrainingData, node: InferenceNodeRef, stage: int
) -> int:
    return len(data.ids("fit", stage, node.artifact.represented_task_ids))


def _fresh_node(
    policy: RouterPolicy,
    inference: InferenceNodeRef,
    features: NodeRouterFeatures,
    source_fit_count: int,
    purpose: str,
) -> ScoringNode:
    return ScoringNode(
        inference.logical_node.node_id,
        make_scorer(
            policy.architecture,
            policy.rank,
            _node_seed(policy, inference.logical_node.node_id, purpose),
        ),
        features,
        inference.artifact.represented_task_ids,
        inference.artifact.represented_class_ids,
        source_fit_count,
    )


def _flat_root(store: RouterStore, policy: RouterPolicy, stage: int) -> Path:
    return store.run / "flat_runs" / policy.content_hash / f"stage_{stage:03d}"


def _load_flat(
    root: Path,
    policy: RouterPolicy,
    stage: int,
    inference_nodes: Sequence[InferenceNodeRef],
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
) -> FlatRunResult:
    from safetensors.torch import load_file

    validate_artifact_directory(root)
    record = load_canonical_json(root / "frontier.json")
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if record.get("content_hash") != record_sha256(core):
        raise ValueError("flat router frontier record changed")
    expected_ids = [node.logical_node.node_id for node in inference_nodes]
    if (
        record["policy_hash"] != policy.content_hash
        or record["stage"] != stage
        or record["logical_node_ids"] != expected_ids
        or record["inference_node_hashes"] != [node.node_hash for node in inference_nodes]
        or file_sha256(root / "scorers.safetensors") != record["scorer_sha256"]
    ):
        raise ValueError("flat router frontier differs from requested semantics")
    state = load_file(root / "scorers.safetensors", device="cpu")
    nodes = []
    for index, inference in enumerate(inference_nodes):
        scorer = make_scorer(
            policy.architecture,
            policy.rank,
            _node_seed(policy, inference.logical_node.node_id, f"flat:{stage}"),
        )
        prefix = f"node_{index:02d}."
        scorer.load_state_dict(
            {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
        nodes.append(
            ScoringNode(
                inference.logical_node.node_id,
                scorer,
                features.get(inference, policy.architecture),
                inference.artifact.represented_task_ids,
                inference.artifact.represented_class_ids,
                _source_fit_count(data, inference, stage),
            )
        )
    training = record["training"]
    return FlatRunResult(
        policy,
        stage,
        tuple(nodes),
        int(training["optimizer_steps"]),
        int(training["epochs"]),
        float(training["best_validation_loss"]),
        root,
        True,
    )


def run_flat_frontier(
    bootstrap: RouterBootstrap,
    policy: RouterPolicy,
    tree: SealedInferenceTree,
    stage: int,
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
    device: torch.device,
) -> FlatRunResult:
    """Fit or reuse a complete independently initialized stage frontier."""
    from safetensors.torch import save_file

    inference_nodes = tree.snapshots[stage - 1].nodes
    root = _flat_root(bootstrap.store, policy, stage)
    if root.is_dir():
        return _load_flat(root, policy, stage, inference_nodes, data, features)
    nodes = tuple(
        _fresh_node(
            policy,
            inference,
            features.get(inference, policy.architecture),
            _source_fit_count(data, inference, stage),
            f"flat:{stage}",
        )
        for inference in inference_nodes
    )
    if stage == 1:
        fit_ids = data.ids("fit", stage)
        validation_ids = data.ids("validation", stage)
        result = RouterTrainingResult(0, 0, 0.0, fit_ids, validation_ids)
    else:
        result = fit_flat_frontier(
            data,
            nodes,
            stage,
            bootstrap.config.training,
            policy.router_seed,
            bootstrap.config.validation_batch_size,
            device,
            bootstrap.store.run
            / "work"
            / "checkpoints"
            / policy.content_hash
            / f"flat_stage_{stage:03d}.pt",
            f"{policy.content_hash}:flat:{stage}",
        )
    work = Path(tempfile.mkdtemp(prefix="flat-router.", dir=bootstrap.store.run / "work"))
    try:
        tensors = {
            f"node_{index:02d}.{key}": value.detach().to(device="cpu").contiguous()
            for index, node in enumerate(nodes)
            for key, value in sorted(node.scorer.state_dict().items())  # type: ignore[union-attr]
        }
        save_file(
            tensors,
            work / "scorers.safetensors",
            metadata={"schema_version": "imagenetr50-flat-router-scorers-v1"},
        )
        core: dict[str, object] = {
            "architecture": policy.architecture,
            "inference_node_hashes": [node.node_hash for node in inference_nodes],
            "logical_node_ids": [node.logical_node.node_id for node in inference_nodes],
            "policy_hash": policy.content_hash,
            "router_run_hash": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-flat-router-frontier-v1",
            "scorer_sha256": file_sha256(work / "scorers.safetensors"),
            "stage": stage,
            "training": result.as_record(),
        }
        publish_immutable_json(work / "frontier.json", {**core, "content_hash": record_sha256(core)})
        publish_artifact_directory(work, root)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return FlatRunResult(
        policy,
        stage,
        nodes,
        result.optimizer_steps,
        result.epochs,
        result.best_validation_loss,
        root,
        False,
    )


def _inference_map(tree: SealedInferenceTree) -> dict[str, InferenceNodeRef]:
    return {node.logical_node.node_id: node for node in tree.nodes}


def _load_recursive_node(
    bootstrap: RouterBootstrap,
    policy: RouterPolicy,
    inference: InferenceNodeRef,
    features: NodeFeatureRegistry,
    children: tuple[ScoringNode, ...],
) -> tuple[ScoringNode, RouterNodeArtifact, dict[str, object]]:
    path = bootstrap.store.node(policy.content_hash, inference.logical_node.node_id)
    artifact = router_node_from_record(load_canonical_json(path / "node.json"))
    if (
        artifact.router_run_hash != bootstrap.protocol.content_hash
        or artifact.policy_hash != policy.content_hash
        or artifact.inference_node_hash != inference.node_hash
    ):
        raise ValueError("persisted recursive router node belongs to another dependency")
    node = load_router_node(
        path,
        features.get(inference, policy.architecture),
        children,
        _node_seed(policy, inference.logical_node.node_id, artifact.maintenance),
        bootstrap.config.mlp_hidden,
    )
    assert isinstance(node, ScoringNode)
    return node, artifact, load_canonical_json(path / "work.json")


def _merge_functional_record(
    data: RouterTrainingData,
    before: Sequence[ScoringNode],
    left: ScoringNode,
    right: ScoringNode,
    parent: ScoringNode,
    stage: int,
) -> dict[str, object]:
    validation_ids = data.ids("validation", stage)
    query, _labels, _tasks = data.batch(validation_ids, torch.device("cpu"))
    left_index, right_index = before.index(left), before.index(right)
    if left_index > right_index:
        left_index, right_index = right_index, left_index
    return functional_merge_diagnostics(
        query, before, left_index, right_index, parent
    ).as_record()


def _events_by_stage(task_count: int) -> dict[int, tuple[MergeEvent, ...]]:
    events, _snapshots = simulate_topology(task_count)
    grouped: dict[int, list[MergeEvent]] = defaultdict(list)
    for event in events:
        grouped[event.stage].append(event)
    return {stage: tuple(values) for stage, values in grouped.items()}


def run_recursive_policy(
    bootstrap: RouterBootstrap,
    policy: RouterPolicy,
    tree: SealedInferenceTree,
    data: RouterTrainingData,
    features: NodeFeatureRegistry,
    device: torch.device,
    task_count: int = 50,
    show_progress: bool = False,
) -> RecursiveRunResult:
    """Build or resume one causal recursive learned-router hierarchy."""
    if policy.maintenance not in {"exact", "u100", "svd0", "svd5"}:
        raise ValueError("recursive runner requires one P-* maintenance policy")
    if not 1 <= task_count <= 50:
        raise ValueError("recursive router task count is outside 1..50")
    policy_root = bootstrap.store.policy_root(policy.content_hash)
    policy_root.mkdir(parents=True, exist_ok=True)
    publish_immutable_json(policy_root / "policy.json", policy.as_record())
    inference_by_id = _inference_map(tree)
    events_by_stage = _events_by_stage(task_count)
    by_id: dict[str, ScoringNode] = {}
    active_ids: list[str] = []
    stage_frontiers: list[tuple[ScoringNode, ...]] = []
    snapshots: list[RouterStageSnapshot] = []
    merge_rows: list[dict[str, object]] = []
    cumulative_steps = 0
    reused_nodes = 0
    created_nodes = 0

    stages: Sequence[int] = tuple(range(1, task_count + 1))
    if show_progress:
        from tqdm.auto import tqdm

        stages = tqdm(
            stages,
            desc=f"{tree.condition} {policy.architecture}/{policy.maintenance}",
            unit="leaf",
            miniters=1,
        )
    for stage in stages:
        sealed_snapshot = tree.snapshots[stage - 1]
        leaf_inference = next(
            node
            for node in sealed_snapshot.nodes
            if node.logical_node.level == 0 and node.logical_node.first_task == stage - 1
        )
        leaf_path = bootstrap.store.node(policy.content_hash, leaf_inference.logical_node.node_id)
        if leaf_path.is_dir():
            leaf, artifact, work = _load_recursive_node(
                bootstrap, policy, leaf_inference, features, ()
            )
            cumulative_steps += int(work["optimizer_steps"])
            reused_nodes += 1
        else:
            leaf = _fresh_node(
                policy,
                leaf_inference,
                features.get(leaf_inference, policy.architecture),
                _source_fit_count(data, leaf_inference, stage),
                "leaf",
            )
            if stage == 1:
                result = RouterTrainingResult(
                    0,
                    0,
                    0.0,
                    (),
                    data.ids("validation", stage),
                )
            else:
                result = fit_new_leaf(
                    data,
                    leaf,
                    tuple(by_id[node_id] for node_id in active_ids),
                    stage,
                    bootstrap.config.training,
                    policy.router_seed,
                    bootstrap.config.validation_batch_size,
                    device,
                    f"{policy.content_hash}:stage:{stage}:leaf",
                    bootstrap.store.run
                    / "work"
                    / "checkpoints"
                    / policy.content_hash
                    / f"leaf_{leaf.node_id}.pt",
                )
            artifact = publish_router_node(
                bootstrap.store,
                policy,
                leaf_inference,
                leaf,
                (),
                "leaf",
                stage,
                result.training_image_ids,
                (),
                result.optimizer_steps,
            )
            cumulative_steps += result.optimizer_steps
            created_nodes += 1
        by_id[leaf.node_id] = leaf
        active_ids.append(leaf.node_id)

        for event in events_by_stage.get(stage, ()):
            left = by_id[event.left.node_id]
            right = by_id[event.right.node_id]
            before = tuple(by_id[node_id] for node_id in active_ids)
            other = tuple(
                node for node in before if node.node_id not in {left.node_id, right.node_id}
            )
            parent_inference = inference_by_id[event.parent.node_id]
            parent_path = bootstrap.store.node(policy.content_hash, event.parent.node_id)
            if parent_path.is_dir():
                parent, parent_artifact, work = _load_recursive_node(
                    bootstrap, policy, parent_inference, features, (left, right)
                )
                cumulative_steps += int(work["optimizer_steps"])
                reused_nodes += 1
                parameter_record: dict[str, object] | None = None
            else:
                parent_features = features.get(parent_inference, policy.architecture)
                source_count = left.source_fit_count + right.source_fit_count
                training_ids: tuple[str, ...] = ()
                repair_ids: tuple[str, ...] = ()
                optimizer_steps = 0
                parameter_record = None
                if policy.maintenance == "exact":
                    scorer = ExactLSEScorer(left, right)
                    maintenance = "exact"
                elif policy.maintenance == "u100":
                    scorer = make_scorer(
                        policy.architecture,
                        policy.rank,
                        _node_seed(policy, event.parent.node_id, "u100"),
                    )
                    maintenance = "u100"
                else:
                    if not isinstance(left.scorer, torch.nn.Module) or not isinstance(
                        right.scorer, torch.nn.Module
                    ):
                        raise ValueError("compact parent merge requires fixed-size child scorers")
                    scorer, parameter = svd_merge_scorers(
                        left.scorer,
                        right.scorer,
                        (left.source_fit_count, right.source_fit_count),
                        policy.rank,
                        _node_seed(policy, event.parent.node_id, policy.maintenance),
                        bootstrap.config.mlp_hidden,
                    )
                    parameter_record = parameter.as_record()
                    maintenance = policy.maintenance
                parent = ScoringNode(
                    event.parent.node_id,
                    scorer,
                    parent_features,
                    parent_inference.artifact.represented_task_ids,
                    parent_inference.artifact.represented_class_ids,
                    source_count,
                )
                if policy.maintenance == "u100":
                    result = fit_parent(
                        data,
                        parent,
                        left,
                        right,
                        other,
                        stage,
                        data.ids("fit", stage),
                        bootstrap.config.training,
                        policy.router_seed,
                        bootstrap.config.validation_batch_size,
                        device,
                        f"{policy.content_hash}:merge:{event.sequence}:u100",
                        bootstrap.store.run
                        / "work"
                        / "checkpoints"
                        / policy.content_hash
                        / f"parent_{event.parent.node_id}.pt",
                    )
                    training_ids = result.training_image_ids
                    optimizer_steps = result.optimizer_steps
                elif policy.maintenance == "svd5":
                    repair_ids = repair_reservoir(
                        data,
                        parent.represented_task_ids,
                        stage,
                        policy.repair_fraction,
                        f"{policy.content_hash}:merge:{event.sequence}:repair",
                    )
                    negatives = negative_reservoirs(
                        data,
                        other,
                        stage,
                        bootstrap.config.training.negatives_per_live_node,
                        f"{policy.content_hash}:merge:{event.sequence}:negatives",
                    )
                    result = fit_parent(
                        data,
                        parent,
                        left,
                        right,
                        other,
                        stage,
                        repair_ids + negatives,
                        bootstrap.config.training,
                        policy.router_seed,
                        bootstrap.config.validation_batch_size,
                        device,
                        f"{policy.content_hash}:merge:{event.sequence}:svd5",
                        bootstrap.store.run
                        / "work"
                        / "checkpoints"
                        / policy.content_hash
                        / f"parent_{event.parent.node_id}.pt",
                    )
                    training_ids = result.training_image_ids
                    optimizer_steps = result.optimizer_steps
                parent_artifact = publish_router_node(
                    bootstrap.store,
                    policy,
                    parent_inference,
                    parent,
                    (str(left.router_hash), str(right.router_hash)),
                    maintenance,
                    stage,
                    training_ids,
                    repair_ids,
                    optimizer_steps,
                )
                cumulative_steps += optimizer_steps
                created_nodes += 1
            parent.router_hash = parent_artifact.content_hash
            diagnostic_path = (
                policy_root / "diagnostics" / f"merge_{event.sequence:03d}.json"
            )
            if diagnostic_path.is_file():
                diagnostic = load_canonical_json(diagnostic_path)
                diagnostic_core = {
                    key: value for key, value in diagnostic.items() if key != "content_hash"
                }
                if (
                    diagnostic.get("content_hash") != record_sha256(diagnostic_core)
                    or diagnostic.get("policy_hash") != policy.content_hash
                    or diagnostic.get("event", {}).get("merge_id") != event.merge_id
                ):
                    raise ValueError("persisted router merge diagnostic changed")
            else:
                functional = _merge_functional_record(
                    data, before, left, right, parent, stage
                )
                diagnostic_core = {
                    "event": {
                        "merge_id": event.merge_id,
                        "parent_logical_node_id": event.parent.node_id,
                        "sequence": event.sequence,
                        "stage": event.stage,
                    },
                    "functional": functional,
                    "parameter": parameter_record,
                    "policy_hash": policy.content_hash,
                    "schema_version": "imagenetr50-router-merge-diagnostic-v1",
                }
                diagnostic = {
                    **diagnostic_core,
                    "content_hash": record_sha256(diagnostic_core),
                }
                publish_immutable_json(diagnostic_path, diagnostic)
            merge_rows.append(diagnostic)
            by_id[parent.node_id] = parent
            active_ids = [
                node_id
                for node_id in active_ids
                if node_id not in {left.node_id, right.node_id}
            ] + [parent.node_id]

        frontier = tuple(by_id[node.logical_node.node_id] for node in sealed_snapshot.nodes)
        active_ids = [node.node_id for node in frontier]
        if any(node.router_hash is None for node in frontier):
            raise ValueError("recursive frontier contains an unpublished router node")
        snapshot = RouterStageSnapshot(
            bootstrap.protocol.content_hash,
            policy.content_hash,
            tree.condition,
            stage,
            tuple(node.node_id for node in frontier),
            tuple(node.node_hash for node in sealed_snapshot.nodes),
            tuple(str(node.router_hash) for node in frontier),
            cumulative_steps,
        )
        snapshot_path = bootstrap.store.snapshot(policy.content_hash, stage)
        if snapshot_path.is_file():
            persisted = load_router_snapshot(bootstrap.store, policy.content_hash, stage)
            if persisted != snapshot:
                raise ValueError("resumed recursive frontier differs from its checkpoint")
        else:
            publish_router_snapshot(bootstrap.store, snapshot)
        snapshots.append(snapshot)
        stage_frontiers.append(frontier)

    return RecursiveRunResult(
        policy,
        tuple(stage_frontiers),
        tuple(snapshots),
        tuple(merge_rows),
        cumulative_steps,
        reused_nodes,
        created_nodes,
    )


def reload_router_bootstrap(config_path: str | Path, run_path: Path) -> RouterBootstrap:
    """Read an existing prepared protocol for report/status tests without changing it."""
    config = load_router_config(config_path)
    protocol_record = load_canonical_json(run_path / "protocol" / "protocol.json")
    split = router_split_from_record(
        load_canonical_json(run_path / "protocol" / "router_split.json")
    )
    base = load_sealed_router_base(config)
    protocol_values = dict(protocol_record)
    supplied = protocol_values.pop("content_hash")
    protocol_values["inference_policies"] = tuple(
        tuple(value) for value in protocol_values["inference_policies"]
    )
    protocol = RouterProtocol(**protocol_values)
    if protocol.content_hash != supplied or run_path.name != supplied:
        raise ValueError("existing router protocol identity changed")
    environment = load_canonical_json(run_path / "protocol" / "environment_manifest.json")
    code = load_canonical_json(run_path / "protocol" / "code_manifest.json")
    drift = load_canonical_json(run_path / "protocol" / "source_drift_audit.json")
    _train, test = image_transforms(base.primary_config.input_size)
    return RouterBootstrap(
        _project_root(Path(config_path)),
        Path(config_path).resolve(),
        config,
        base,
        split,
        protocol,
        RouterStore(config.artifact_root, protocol.content_hash),
        _checkpoint(base, config),
        test,
        code,
        environment,
        drift,
    )


__all__ = [
    "FlatRunResult",
    "NodeFeatureRegistry",
    "RecursiveRunResult",
    "RouterBootstrap",
    "bootstrap_router_protocol",
    "condition_id",
    "latest_router_run",
    "make_router_policy",
    "reload_router_bootstrap",
    "run_flat_frontier",
    "run_recursive_policy",
]
