from dataclasses import replace
from pathlib import Path

import torch

from apm.continual.artifacts import load_canonical_json, record_sha256
from apm.continual.vision.imagenetr.bank import LogicalNode, simulate_topology
from apm.continual.vision.imagenetr.protocol import NodeArtifact
from apm.continual.vision.imagenetr.router_artifacts import (
    InferenceNodeRef,
    RouterStore,
    SealedInferenceTree,
    SealedSnapshot,
)
from apm.continual.vision.imagenetr.router_config import (
    RouterTrainingConfig,
    load_router_config,
)
from apm.continual.vision.imagenetr.router_descriptor import (
    NodeRouterFeatures,
    selected_response_modules,
)
from apm.continual.vision.imagenetr.router_experiment import (
    RouterBootstrap,
    run_recursive_policy,
)
from apm.continual.vision.imagenetr.router_features import RouterFeatureUniverse
from apm.continual.vision.imagenetr.router_protocol import (
    RouterPolicy,
    RouterProtocol,
    RouterSplit,
)
from apm.continual.vision.imagenetr.router_training import RouterTrainingData


def _inference(logical: LogicalNode) -> InferenceNodeRef:
    classes = tuple(
        class_id
        for task in logical.task_ids
        for class_id in range(4 * task, 4 * task + 4)
    )
    artifact = NodeArtifact(
        run_hash="1" * 64,
        software_manifest_hash="2" * 64,
        git_commit="synthetic",
        creation_timestamp_utc="2026-08-21T00:00:00Z",
        level=logical.level,
        first_task=logical.first_task,
        last_task=logical.last_task,
        represented_task_ids=logical.task_ids,
        represented_class_ids=classes,
        represented_train_image_count=16 * len(logical.task_ids),
        parent_hashes=() if logical.level == 0 else ("3" * 64, "4" * 64),
        unrepaired_parent_hash=None,
        consolidation_method="leaf" if logical.level == 0 else "svd",
        consolidation_config_hash="5" * 64,
        repair_config_hash="6" * 64,
        lora_sha256=record_sha256({"logical": logical.node_id}),
        classifier_sha256=record_sha256({"classes": classes}),
        proxy_image_ids=(),
        repair_image_ids=(),
        source_priority_hash="7" * 64,
        training_optimizer_steps=0,
    )
    return InferenceNodeRef(
        logical,
        artifact.content_hash,
        Path("unused"),
        artifact,
        "8" * 64,
    )


def _tree() -> SealedInferenceTree:
    _events, topology = simulate_topology(8)
    by_id = {
        node.node_id: _inference(node)
        for snapshot in topology
        for node in snapshot.live_nodes
    }
    snapshots = tuple(
        SealedSnapshot(
            snapshot.stage,
            tuple(by_id[node.node_id] for node in snapshot.live_nodes),
        )
        for snapshot in topology
    )
    return SealedInferenceTree(
        "I-U100",
        "9" * 64,
        snapshots,
        tuple(by_id[key] for key in sorted(by_id)),
    )


class _Features:
    def __init__(self, config) -> None:
        self.modules = selected_response_modules(config)

    def get(self, node: InferenceNodeRef, architecture: str) -> NodeRouterFeatures:
        descriptor = torch.zeros(128)
        descriptor[list(node.logical_node.task_ids)] = 1.0
        kernels = {}
        if architecture == "r3":
            kernels = {
                name: torch.nn.functional.one_hot(
                    torch.tensor([(index + node.logical_node.first_task) % 8]),
                    num_classes=768,
                ).to(torch.float32)
                for index, name in enumerate(self.modules)
            }
        return NodeRouterFeatures(
            descriptor,
            kernels,
            "a" * 64,
            "b" * 64 if architecture == "r3" else None,
        )


def _training(config) -> tuple[RouterTrainingData, RouterSplit]:
    image_ids = tuple(f"{index:064x}" for index in range(8 * 16))
    tasks = torch.arange(8).repeat_interleave(16)
    labels = 4 * tasks + torch.arange(16).repeat(8) % 4
    prelogits = torch.zeros(len(image_ids), 768)
    prelogits[torch.arange(len(image_ids)), tasks] = 6.0
    prelogits[:, 16] = torch.linspace(-1.0, 1.0, len(image_ids))
    activations = {
        name: prelogits.to(torch.bfloat16).clone()
        for name in selected_response_modules(config)
    }
    universe = RouterFeatureUniverse(
        image_ids, labels, tasks, prelogits, activations
    )
    fit = tuple(
        image_id for index, image_id in enumerate(image_ids) if index % 4 != 0
    )
    validation = tuple(
        image_id for index, image_id in enumerate(image_ids) if index % 4 == 0
    )
    split = RouterSplit("c" * 64, fit, validation, "synthetic-eight-task")
    return RouterTrainingData(universe, split), split


def test_synthetic_eight_task_all_parent_families_and_r3_resume(tmp_path: Path) -> None:
    source = Path("configs/vision/imagenetr/recursive_router_oracle_recovery_v1.yaml")
    config = replace(
        load_router_config(source),
        artifact_root=tmp_path,
        training=RouterTrainingConfig(0.01, 0.0, 64, 2, 1, 1.0, 2, 1.0),
        validation_batch_size=128,
        num_workers=0,
    )
    data, split = _training(config)
    tree = _tree()
    protocol = RouterProtocol(
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "1" * 64,
        (("I-U100", tree.policy_hash), ("I-SVD0", "2" * 64), ("I-SVD5", "3" * 64)),
        split.content_hash,
        "4" * 64,
        "5" * 64,
        "6" * 64,
    )
    store = RouterStore(tmp_path, protocol.content_hash)
    (store.run / "work").mkdir(parents=True)
    bootstrap = RouterBootstrap(
        Path.cwd(),
        source,
        config,
        None,  # type: ignore[arg-type]
        split,
        protocol,
        store,
        Path("unused"),
        object(),
        {},
        {},
        {},
    )
    features = _Features(config)
    results = {}
    for architecture, maintenance in (
        ("r1", "exact"),
        ("r1", "u100"),
        ("r1", "svd0"),
        ("r1", "svd5"),
        ("r3", "svd5"),
    ):
        policy = RouterPolicy(
            "I-U100",
            tree.policy_hash,
            architecture,
            8,
            maintenance,
            0.05 if maintenance == "svd5" else 0.0,
            1993,
            "7" * 64,
            "8" * 64,
            "9" * 64,
        )
        results[(architecture, maintenance)] = run_recursive_policy(
            bootstrap,
            policy,
            tree,
            data,
            features,  # type: ignore[arg-type]
            torch.device("cpu"),
            8,
        )
    assert all(len(result.snapshots) == 8 for result in results.values())
    assert all(len(result.stage_frontiers[-1]) == 4 for result in results.values())
    assert max(
        row["functional"]["mean_mass_error"]
        for row in results[("r1", "exact")].merge_diagnostics
    ) < 1.0e-6
    assert results[("r1", "svd5")].optimizer_steps > results[("r1", "svd0")].optimizer_steps
    assert results[("r3", "svd5")].optimizer_steps > 0

    svd0 = results[("r1", "svd0")]
    for snapshot in svd0.snapshots:
        for logical_id in snapshot.logical_node_ids:
            node = load_canonical_json(
                store.node(svd0.policy.content_hash, logical_id) / "node.json"
            )
            if node["maintenance"] == "svd0":
                work = load_canonical_json(
                    store.node(svd0.policy.content_hash, logical_id) / "work.json"
                )
                assert work["optimizer_steps"] == 0
                assert work["training_image_ids"] == []

    first = results[("r3", "svd5")]
    resumed = run_recursive_policy(
        bootstrap,
        first.policy,
        tree,
        data,
        features,  # type: ignore[arg-type]
        torch.device("cpu"),
        8,
    )
    assert resumed.created_nodes == 0
    assert resumed.reused_nodes == 12
    assert tuple(row.content_hash for row in resumed.snapshots) == tuple(
        row.content_hash for row in first.snapshots
    )
