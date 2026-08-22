"""Sealed-inference loading and content-addressed learned-router storage."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
import shutil
import tempfile

import torch

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    NodeBundle,
    load_node_bundle,
    node_artifact_from_record,
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.bank import LogicalNode, simulate_topology
from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    load_dataset_manifest,
    largest_remainder_counts,
)
from apm.continual.vision.imagenetr.protocol import NodeArtifact
from apm.continual.vision.imagenetr.router_config import RouterConfig
from apm.continual.vision.imagenetr.router_protocol import RouterProtocol, RouterSplit
from apm.continual.vision.imagenetr.router_protocol import (
    RouterNodeArtifact,
    RouterPolicy,
    RouterStageSnapshot,
    router_node_from_record,
    router_snapshot_from_record,
)


@dataclass(frozen=True, slots=True)
class InferenceNodeRef:
    """Authenticated location and semantics of one sealed inference node."""

    logical_node: LogicalNode
    node_hash: str
    directory: Path
    artifact: NodeArtifact
    artifact_directory_hash: str

    def load(self) -> NodeBundle:
        """Load and revalidate the tensor-bearing bundle on demand."""
        bundle = load_node_bundle(self.directory)
        if bundle.artifact.content_hash != self.node_hash:
            raise ValueError("sealed inference node hash changed while loading")
        return bundle


@dataclass(frozen=True, slots=True)
class SealedSnapshot:
    """One policy snapshot mapped to authenticated node references."""

    stage: int
    nodes: tuple[InferenceNodeRef, ...]


@dataclass(frozen=True, slots=True)
class SealedInferenceTree:
    """Read-only materialization of a completed inference hierarchy."""

    condition: str
    policy_hash: str
    snapshots: tuple[SealedSnapshot, ...]
    nodes: tuple[InferenceNodeRef, ...]

    @property
    def final(self) -> SealedSnapshot:
        return self.snapshots[-1]


@dataclass(frozen=True, slots=True)
class SealedRouterBase:
    """All authenticated read-only inputs needed by the router experiment."""

    run_root: Path
    prepared_root: Path
    manifest: DatasetManifest
    primary_config: ImageNetRConfig
    protocol_record: dict[str, object]
    trees: tuple[SealedInferenceTree, ...]
    inventory_hash: str
    inventory_rows: tuple[dict[str, object], ...]

    @property
    def tree_map(self) -> dict[str, SealedInferenceTree]:
        return {tree.condition: tree for tree in self.trees}


def build_router_split(
    manifest: DatasetManifest,
    fit_fraction: float,
    seed: int,
) -> RouterSplit:
    """Freeze an exact class-stratified fit/validation split of training rows."""
    if not 0.0 < fit_fraction < 1.0 or seed < 0:
        raise ValueError("invalid router split fraction or seed")
    train = tuple(row for row in manifest.images if row.split == "train")
    desired_fit = int(round(fit_fraction * len(train)))
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for row in train:
        grouped[f"{row.remapped_class_index:03d}"].append(row)
    allocation = largest_remainder_counts(
        {name: len(rows) for name, rows in grouped.items()}, desired_fit
    )
    fit_ids: set[str] = set()
    namespace = f"imagenetr50-router-split-v1:{seed}:{fit_fraction:.12g}"
    for name, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                sha256(f"{namespace}\0{name}\0{row.image_id}\0{row.priority}".encode()).hexdigest(),
                row.image_id,
            ),
        )
        fit_ids.update(row.image_id for row in ranked[: allocation[name]])
    fit = tuple(row.image_id for row in train if row.image_id in fit_ids)
    validation = tuple(row.image_id for row in train if row.image_id not in fit_ids)
    if len(fit) != desired_fit or len(validation) != len(train) - desired_fit:
        raise ValueError("router split does not have the exact requested sizes")
    return RouterSplit(manifest.content_hash, fit, validation, namespace)


def router_split_from_record(record: Mapping[str, object]) -> RouterSplit:
    """Parse and authenticate a persisted router split."""
    values = dict(record)
    supplied = str(values.pop("content_hash", ""))
    values["fit_image_ids"] = tuple(values["fit_image_ids"])
    values["validation_image_ids"] = tuple(values["validation_image_ids"])
    split = RouterSplit(**values)
    if split.content_hash != supplied:
        raise ValueError("router split content hash changed")
    return split


def _leaf_directories(run_root: Path) -> dict[int, Path]:
    leaves: dict[int, Path] = {}
    for task_root in sorted((run_root / "leaves").glob("task_*")):
        candidates = tuple(path for path in task_root.iterdir() if path.is_dir())
        if len(candidates) != 1:
            raise ValueError(f"sealed leaf directory is ambiguous: {task_root}")
        task = int(task_root.name.split("_")[-1])
        leaves[task] = candidates[0]
    if tuple(sorted(leaves)) != tuple(range(50)):
        raise ValueError("sealed run does not contain exactly 50 leaves")
    return leaves


def _node_ref(logical: LogicalNode, directory: Path, expected_hash: str) -> InferenceNodeRef:
    artifact_directory_hash = validate_artifact_directory(directory)
    artifact = node_artifact_from_record(load_canonical_json(directory / "node.json"))
    if artifact.content_hash != expected_hash or artifact.level != logical.level:
        raise ValueError("snapshot node identity differs from its sealed directory")
    if artifact.represented_task_ids != logical.task_ids:
        raise ValueError("snapshot logical interval differs from node semantics")
    return InferenceNodeRef(
        logical,
        expected_hash,
        directory,
        artifact,
        artifact_directory_hash,
    )


def load_sealed_tree(
    run_root: Path,
    condition: str,
    policy_hash: str,
    leaf_directories: Mapping[int, Path] | None = None,
) -> SealedInferenceTree:
    """Authenticate one required policy, all snapshots, and referenced nodes."""
    require_sha256(policy_hash, "sealed inference policy")
    tree_root = run_root / "trees" / policy_hash
    policy = load_canonical_json(tree_root / "policy.json")
    if policy.get("content_hash") != policy_hash:
        raise ValueError("sealed inference policy record changed")
    raw = load_canonical_json(tree_root / "snapshots.json")
    if (
        raw.get("schema_version") != "imagenetr50-materialized-snapshots-v1"
        or raw.get("policy_hash") != policy_hash
        or len(raw.get("snapshots", ())) != 50
    ):
        raise ValueError("sealed inference snapshots are incomplete")
    _events, topology = simulate_topology(50)
    leaves = dict(leaf_directories or _leaf_directories(run_root))
    by_logical: dict[str, InferenceNodeRef] = {}
    snapshots: list[SealedSnapshot] = []
    for expected, row in zip(topology, raw["snapshots"]):
        if int(row["stage"]) != expected.stage:
            raise ValueError("sealed snapshot stage differs from the fixed topology")
        logical_ids = tuple(row["logical_node_ids"])
        node_hashes = tuple(row["node_hashes"])
        if logical_ids != tuple(node.node_id for node in expected.live_nodes):
            raise ValueError("sealed logical snapshot differs from fixed topology")
        refs = []
        for logical, node_hash in zip(expected.live_nodes, node_hashes):
            if logical.node_id not in by_logical:
                directory = (
                    leaves[logical.first_task]
                    if logical.level == 0
                    else tree_root / "nodes" / logical.node_id
                )
                by_logical[logical.node_id] = _node_ref(logical, directory, str(node_hash))
            if by_logical[logical.node_id].node_hash != node_hash:
                raise ValueError("one logical node maps to multiple sealed tensor identities")
            refs.append(by_logical[logical.node_id])
        snapshots.append(SealedSnapshot(expected.stage, tuple(refs)))
    return SealedInferenceTree(
        condition,
        policy_hash,
        tuple(snapshots),
        tuple(by_logical[key] for key in sorted(by_logical)),
    )


def load_sealed_router_base(config: RouterConfig) -> SealedRouterBase:
    """Authenticate the complete read-only inference authority."""
    run_root = config.inference_artifact_root / "runs" / config.sealed_run_hash
    protocol = load_canonical_json(run_root / "protocol" / "protocol_manifest.json")
    if protocol.get("content_hash") != config.sealed_run_hash:
        raise ValueError("configured sealed run does not match its protocol")
    manifest = load_dataset_manifest(config.data_root / "imagenet-r" / "dataset_manifest.json")
    if manifest.content_hash != protocol.get("dataset_manifest_hash"):
        raise ValueError("local prepared dataset differs from the sealed run")
    project_root = next(
        (
            candidate
            for candidate in (
                config.inference_artifact_root,
                *config.inference_artifact_root.parents,
            )
            if (candidate / "pyproject.toml").is_file()
        ),
        None,
    )
    if project_root is None:
        raise ValueError("cannot resolve the repository root for the primary config")
    primary_config = load_config(project_root / "configs/vision/imagenetr/primary.yaml")
    if primary_config.config_hash != protocol.get("config_hash"):
        raise ValueError("current primary config differs from the sealed run")
    leaves = _leaf_directories(run_root)
    trees = tuple(
        load_sealed_tree(run_root, condition, policy_hash, leaves)
        for condition, policy_hash in config.inference_policies
    )
    unique: dict[tuple[str, str], InferenceNodeRef] = {}
    for tree in trees:
        for node in tree.nodes:
            unique[(tree.condition, node.logical_node.node_id)] = node
    rows = tuple(
        {
            "artifact_directory_hash": node.artifact_directory_hash,
            "condition": condition,
            "inference_node_hash": node.node_hash,
            "level": node.logical_node.level,
            "logical_node_id": logical_id,
            "path": str(node.directory),
            "represented_tasks": list(node.logical_node.task_ids),
        }
        for (condition, logical_id), node in sorted(unique.items())
    )
    inventory_hash = record_sha256(
        {
            "rows": list(rows),
            "schema_version": "imagenetr50-router-inference-inventory-v1",
            "sealed_run_hash": config.sealed_run_hash,
        }
    )
    return SealedRouterBase(
        run_root,
        config.data_root / "imagenet-r",
        manifest,
        primary_config,
        dict(protocol),
        trees,
        inventory_hash,
        rows,
    )


@dataclass(frozen=True, slots=True)
class RouterStore:
    """Stable paths belonging to one recursive-router protocol."""

    root: Path
    run_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.run_hash, "router run")

    @property
    def run(self) -> Path:
        return self.root / "runs" / self.run_hash

    def prepare(
        self,
        protocol: RouterProtocol,
        split: RouterSplit,
        protocol_link: Mapping[str, object],
    ) -> None:
        """Create the isolated layout and immutably publish protocol records."""
        if protocol.content_hash != self.run_hash or split.content_hash != protocol.split_hash:
            raise ValueError("router store inputs differ from its namespace")
        for relative in (
            "protocol",
            "features",
            "descriptors",
            "response_kernels",
            "flat_runs",
            "recursive_runs",
            "diagnostics",
            "reports",
            "state",
            "work",
        ):
            (self.run / relative).mkdir(parents=True, exist_ok=True)
        publish_immutable_json(self.run / "protocol" / "protocol.json", protocol.as_record())
        publish_immutable_json(self.run / "protocol" / "router_split.json", split.as_record())
        publish_immutable_json(self.run / "protocol" / "protocol_link.json", dict(protocol_link))

    def policy_root(self, policy_hash: str) -> Path:
        require_sha256(policy_hash, "router policy")
        return self.run / "recursive_runs" / policy_hash

    def node(self, policy_hash: str, logical_node_id: str) -> Path:
        require_sha256(logical_node_id, "logical node")
        return self.policy_root(policy_hash) / "nodes" / logical_node_id

    def snapshot(self, policy_hash: str, stage: int) -> Path:
        if not 1 <= stage <= 50:
            raise ValueError("router stage is outside 1..50")
        return self.policy_root(policy_hash) / "snapshots" / f"stage_{stage:03d}.json"


def publish_router_snapshot(
    store: RouterStore,
    snapshot: RouterStageSnapshot,
) -> None:
    """Publish or byte-validate one durable recursive frontier boundary."""
    if snapshot.router_run_hash != store.run_hash:
        raise ValueError("router snapshot belongs to another run")
    publish_immutable_json(
        store.snapshot(snapshot.policy_hash, snapshot.stage), snapshot.as_record()
    )


def load_router_snapshot(
    store: RouterStore, policy_hash: str, stage: int
) -> RouterStageSnapshot:
    """Load and authenticate one durable recursive frontier boundary."""
    return router_snapshot_from_record(
        load_canonical_json(store.snapshot(policy_hash, stage))
    )


def publish_router_node(
    store: RouterStore,
    policy: RouterPolicy,
    inference_node: InferenceNodeRef,
    scoring_node: object,
    parent_router_hashes: tuple[str, ...],
    maintenance: str,
    stage_created: int,
    training_image_ids: tuple[str, ...],
    repair_image_ids: tuple[str, ...],
    optimizer_steps: int,
) -> RouterNodeArtifact:
    """Atomically publish one learned or exact router node and its work record."""
    from safetensors.torch import save_file

    from apm.continual.vision.imagenetr.router_scores import ExactLSEScorer, ScoringNode

    if not isinstance(scoring_node, ScoringNode):
        raise TypeError("router publication requires a ScoringNode")
    scorer = scoring_node.scorer
    target = store.node(policy.content_hash, inference_node.logical_node.node_id)
    if target.is_dir():
        validate_artifact_directory(target)
        artifact = router_node_from_record(load_canonical_json(target / "node.json"))
        expected_training_hash = (
            record_sha256(list(training_image_ids)) if training_image_ids else None
        )
        expected_repair_hash = (
            record_sha256(list(repair_image_ids)) if repair_image_ids else None
        )
        if (
            artifact.router_run_hash != store.run_hash
            or artifact.policy_hash != policy.content_hash
            or artifact.inference_node_hash != inference_node.node_hash
            or artifact.logical_node_id != inference_node.logical_node.node_id
            or artifact.stage_created != stage_created
            or artifact.maintenance != maintenance
            or artifact.architecture != scoring_node.scorer.architecture
            or artifact.rank != scoring_node.scorer.rank
            or artifact.parent_router_hashes != parent_router_hashes
            or artifact.training_ids_hash != expected_training_hash
            or artifact.repair_ids_hash != expected_repair_hash
            or artifact.optimizer_steps != optimizer_steps
        ):
            raise ValueError("persisted router node differs from the requested work")
        scoring_node.router_hash = artifact.content_hash
        return artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="router-node.", dir=store.run / "work"))
    try:
        if isinstance(scorer, torch.nn.Module):
            state = {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in sorted(scorer.state_dict().items())
            }
        elif isinstance(scorer, ExactLSEScorer):
            state = {"exact_lse_marker": torch.empty(0)}
        else:
            raise TypeError("unsupported router scorer for persistence")
        scorer_path = work / "scorer.safetensors"
        save_file(
            state,
            scorer_path,
            metadata={
                "architecture": scorer.architecture,
                "schema_version": "imagenetr50-router-scorer-v1",
            },
        )
        scorer_sha = file_sha256(scorer_path)
        training_hash = (
            record_sha256(list(training_image_ids)) if training_image_ids else None
        )
        repair_hash = record_sha256(list(repair_image_ids)) if repair_image_ids else None
        response_sha = (
            scoring_node.features.response_kernel_sha256
            if scorer.architecture == "r3"
            else None
        )
        artifact = RouterNodeArtifact(
            router_run_hash=store.run_hash,
            policy_hash=policy.content_hash,
            inference_node_hash=inference_node.node_hash,
            logical_node_id=inference_node.logical_node.node_id,
            stage_created=stage_created,
            level=inference_node.logical_node.level,
            represented_task_ids=inference_node.artifact.represented_task_ids,
            represented_class_ids=inference_node.artifact.represented_class_ids,
            represented_fit_count=scoring_node.source_fit_count,
            architecture=scorer.architecture,
            rank=scorer.rank,
            maintenance=maintenance,
            scorer_sha256=scorer_sha,
            descriptor_sha256=scoring_node.features.descriptor_sha256,
            response_kernel_sha256=response_sha,
            parent_router_hashes=parent_router_hashes,
            training_ids_hash=training_hash,
            repair_ids_hash=repair_hash,
            optimizer_steps=optimizer_steps,
        )
        publish_immutable_json(work / "node.json", artifact.as_record())
        publish_immutable_json(
            work / "work.json",
            {
                "optimizer_steps": optimizer_steps,
                "repair_image_ids": list(repair_image_ids),
                "schema_version": "imagenetr50-router-node-work-v1",
                "training_image_ids": list(training_image_ids),
            },
        )
        publish_artifact_directory(work, target)
        scoring_node.router_hash = artifact.content_hash
        return artifact
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load_router_node(
    path: str | Path,
    features: object,
    children: tuple[object, ...] = (),
    scorer_seed: int = 0,
    mlp_hidden: int = 64,
) -> object:
    """Load a persisted scorer and pair it with already-authenticated node features."""
    from safetensors import safe_open

    from apm.continual.vision.imagenetr.router_descriptor import NodeRouterFeatures
    from apm.continual.vision.imagenetr.router_scores import (
        ExactLSEScorer,
        ScoringNode,
        load_scorer,
    )

    root = Path(path)
    validate_artifact_directory(root)
    artifact = router_node_from_record(load_canonical_json(root / "node.json"))
    if not isinstance(features, NodeRouterFeatures):
        raise TypeError("router-node loading requires authenticated node features")
    scorer_path = root / "scorer.safetensors"
    if file_sha256(scorer_path) != artifact.scorer_sha256:
        raise ValueError("router scorer tensor bytes changed")
    if artifact.architecture == "exact":
        if len(children) != 2 or not all(isinstance(child, ScoringNode) for child in children):
            raise ValueError("exact router loading requires its two child scoring nodes")
        scorer = ExactLSEScorer(children[0], children[1])
    else:
        with safe_open(scorer_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("schema_version") != "imagenetr50-router-scorer-v1":
                raise ValueError("unknown persisted router scorer schema")
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        scorer = load_scorer(
            artifact.architecture,
            artifact.rank,
            scorer_seed,
            state,
            mlp_hidden,
        )
    node = ScoringNode(
        artifact.logical_node_id,
        scorer,
        features,
        artifact.represented_task_ids,
        artifact.represented_class_ids,
        artifact.represented_fit_count,
        artifact.content_hash,
    )
    return node


def inference_inventory(base: SealedRouterBase) -> dict[str, object]:
    """Return the authenticated inventory record persisted by Phase 0."""
    core: dict[str, object] = {
        "rows": list(base.inventory_rows),
        "schema_version": "imagenetr50-router-inference-inventory-v1",
        "sealed_run_hash": base.protocol_record["content_hash"],
    }
    if record_sha256(core) != base.inventory_hash:
        raise ValueError("in-memory sealed inventory changed")
    return {**core, "content_hash": base.inventory_hash}


__all__ = [
    "InferenceNodeRef",
    "RouterStore",
    "SealedInferenceTree",
    "SealedRouterBase",
    "SealedSnapshot",
    "build_router_split",
    "inference_inventory",
    "load_sealed_router_base",
    "load_sealed_tree",
    "load_router_node",
    "load_router_snapshot",
    "publish_router_snapshot",
    "publish_router_node",
    "router_split_from_record",
]
