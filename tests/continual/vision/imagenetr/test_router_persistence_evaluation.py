from pathlib import Path

import torch

from apm.continual.vision.imagenetr.bank import LogicalNode
from apm.continual.vision.imagenetr.protocol import NodeArtifact
from apm.continual.vision.imagenetr.router_artifacts import (
    InferenceNodeRef,
    RouterStore,
    load_router_node,
    load_router_snapshot,
    publish_router_node,
    publish_router_snapshot,
)
from apm.continual.vision.imagenetr.router_descriptor import NodeRouterFeatures
from apm.continual.vision.imagenetr.router_evaluation import evaluate_router_frontier
from apm.continual.vision.imagenetr.router_features import RouterFeatureUniverse
from apm.continual.vision.imagenetr.router_protocol import (
    RouterPolicy,
    RouterStageSnapshot,
)
from apm.continual.vision.imagenetr.router_reporting import (
    load_router_evaluation,
    publish_router_evaluation,
    write_router_report,
)
from apm.continual.vision.imagenetr.router_scores import R0Scorer, R1Scorer, ScoringNode
from apm.continual.vision.imagenetr.routing import NodeScores


SHA = "a" * 64


def _artifact(logical: LogicalNode, node_hash_seed: str) -> NodeArtifact:
    classes = tuple(
        class_id
        for task in logical.task_ids
        for class_id in range(4 * task, 4 * task + 4)
    )
    return NodeArtifact(
        run_hash="1" * 64,
        software_manifest_hash="2" * 64,
        git_commit="abc",
        creation_timestamp_utc="2026-08-21T00:00:00Z",
        level=logical.level,
        first_task=logical.first_task,
        last_task=logical.last_task,
        represented_task_ids=logical.task_ids,
        represented_class_ids=classes,
        represented_train_image_count=10 * len(logical.task_ids),
        parent_hashes=() if logical.level == 0 else ("3" * 64, "4" * 64),
        unrepaired_parent_hash=None,
        consolidation_method="leaf" if logical.level == 0 else "svd",
        consolidation_config_hash="5" * 64,
        repair_config_hash="6" * 64,
        lora_sha256=node_hash_seed * 64,
        classifier_sha256="8" * 64,
        proxy_image_ids=(),
        repair_image_ids=(),
        source_priority_hash="9" * 64,
        training_optimizer_steps=0,
    )


def _reference(logical: LogicalNode, seed: str = "7") -> InferenceNodeRef:
    artifact = _artifact(logical, seed)
    return InferenceNodeRef(logical, artifact.content_hash, Path("unused"), artifact, "b" * 64)


def _policy() -> RouterPolicy:
    return RouterPolicy(
        "I-U100",
        "c" * 64,
        "r1",
        8,
        "svd0",
        0.0,
        1993,
        "d" * 64,
        "e" * 64,
        "f" * 64,
    )


def test_router_node_and_stage_snapshot_round_trip(tmp_path: Path) -> None:
    store = RouterStore(tmp_path, SHA)
    (store.run / "work").mkdir(parents=True)
    inference = _reference(LogicalNode(0, 0, 0))
    features = NodeRouterFeatures(torch.arange(128, dtype=torch.float32), {}, "1" * 64, None)
    node = ScoringNode(
        inference.logical_node.node_id,
        R1Scorer(seed=4),
        features,
        (0,),
        (0, 1, 2, 3),
        8,
    )
    artifact = publish_router_node(
        store,
        _policy(),
        inference,
        node,
        (),
        "leaf",
        1,
        (),
        (),
        0,
    )
    loaded = load_router_node(store.node(_policy().content_hash, node.node_id), features)
    assert isinstance(loaded, ScoringNode)
    assert loaded.router_hash == artifact.content_hash
    for key, value in node.scorer.state_dict().items():
        torch.testing.assert_close(value, loaded.scorer.state_dict()[key])

    snapshot = RouterStageSnapshot(
        SHA,
        _policy().content_hash,
        "I-U100",
        1,
        (node.node_id,),
        (inference.node_hash,),
        (artifact.content_hash,),
        0,
    )
    publish_router_snapshot(store, snapshot)
    assert load_router_snapshot(store, _policy().content_hash, 1) == snapshot


class _Scores:
    def __init__(self, values: dict[str, NodeScores]) -> None:
        self.values = values

    def project(self, node: InferenceNodeRef, split: str, image_ids: tuple[str, ...]) -> NodeScores:
        assert split == "validation"
        assert len(image_ids) == 4
        return self.values[node.node_hash]


def test_task_free_evaluation_and_report_artifact_round_trip(tmp_path: Path) -> None:
    image_ids = tuple(f"{index:064x}" for index in range(4))
    labels = torch.tensor((0, 1, 4, 5))
    tasks = labels // 4
    prelogits = torch.zeros(4, 768)
    prelogits[:2, 0] = 4.0
    prelogits[2:, 0] = -4.0
    universe = RouterFeatureUniverse(image_ids, labels, tasks, prelogits, {})
    first, second = _reference(LogicalNode(0, 0, 0), "7"), _reference(
        LogicalNode(0, 1, 1), "a"
    )
    fixed = NodeRouterFeatures(torch.zeros(128), {}, "1" * 64, None)
    left, right = R0Scorer(seed=1), R0Scorer(seed=2)
    with torch.no_grad():
        left.query_weight.zero_()
        left.query_weight[0] = 2.0
        right.query_weight.zero_()
        right.query_weight[0] = -2.0
    nodes = (
        ScoringNode(first.logical_node.node_id, left, fixed, (0,), (0, 1, 2, 3), 8),
        ScoringNode(second.logical_node.node_id, right, fixed, (1,), (4, 5, 6, 7), 8),
    )
    cached = {
        first.node_hash: NodeScores(
            first.node_hash,
            (0, 1, 2, 3),
            torch.tensor(((5.0, 0, 0, 0), (0, 5.0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0))),
            torch.zeros(4, 4),
        ),
        second.node_hash: NodeScores(
            second.node_hash,
            (4, 5, 6, 7),
            torch.tensor(((1.0, 0, 0, 0), (1, 0, 0, 0), (5.0, 0, 0, 0), (0, 5.0, 0, 0))),
            torch.zeros(4, 4),
        ),
    }
    result = evaluate_router_frontier(
        condition_id="SYNTH",
        inference_condition="I-U100",
        architecture="r0",
        maintenance="flat_full",
        router_seed=1993,
        split="validation",
        stage=2,
        universe=universe,
        image_ids=image_ids,
        scoring_nodes=nodes,
        inference_nodes=(first, second),
        score_provider=_Scores(cached),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    assert result.metric.routed_accuracy == 100.0
    assert result.metric.oracle_gap == 0.0
    published = publish_router_evaluation(tmp_path, result)
    assert not published.reused
    loaded = load_router_evaluation(published.path)
    assert loaded.metric == result.metric
    assert len(loaded.per_image) == 4
    report = write_router_report(tmp_path, status={"phase": "synthetic"})
    assert report.is_file()
    assert "## Validation capacity gate" in report.read_text(encoding="utf-8")
    assert (tmp_path / "reports" / "REPORT.html").is_file()
