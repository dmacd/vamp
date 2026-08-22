from pathlib import Path

import torch

from apm.continual.vision.imagenetr.router_config import RouterTrainingConfig
from apm.continual.vision.imagenetr.router_descriptor import NodeRouterFeatures
from apm.continual.vision.imagenetr.router_features import RouterFeatureUniverse
from apm.continual.vision.imagenetr.router_protocol import RouterSplit
from apm.continual.vision.imagenetr.router_scores import R0Scorer, ScoringNode
from apm.continual.vision.imagenetr.router_training import (
    RouterTrainingData,
    fit_new_leaf,
)


def _synthetic() -> tuple[RouterTrainingData, tuple[str, ...]]:
    image_ids = tuple(f"{index:064x}" for index in range(32))
    labels = torch.tensor([index % 8 for index in range(32)])
    tasks = labels // 4
    features = torch.zeros(32, 768)
    features[tasks == 0, 0] = 4.0
    features[tasks == 1, 0] = -4.0
    features[:, 1] = torch.linspace(-1.0, 1.0, 32)
    universe = RouterFeatureUniverse(image_ids, labels, tasks, features, {})
    fit = tuple(value for index, value in enumerate(image_ids) if index % 4 != 0)
    validation = tuple(value for index, value in enumerate(image_ids) if index % 4 == 0)
    split = RouterSplit("a" * 64, fit, validation, "synthetic")
    return RouterTrainingData(universe, split), fit


def test_causal_leaf_fit_uses_only_fit_rows_and_keeps_old_state_exact() -> None:
    data, fit_ids = _synthetic()
    features = NodeRouterFeatures(torch.zeros(128), {}, "b" * 64, None)
    old_scorer = R0Scorer(seed=1)
    with torch.no_grad():
        old_scorer.query_weight.zero_()
        old_scorer.query_weight[0] = 1.0
    old = ScoringNode("old", old_scorer, features, (0,), (0, 1, 2, 3), 12)
    new = ScoringNode("new", R0Scorer(seed=2), features, (1,), (4, 5, 6, 7), 12)
    before = {key: value.clone() for key, value in old_scorer.state_dict().items()}
    config = RouterTrainingConfig(0.02, 0.0, 16, 30, 5, 1.0, 4, 1.0)
    result = fit_new_leaf(
        data,
        new,
        (old,),
        2,
        config,
        1993,
        32,
        torch.device("cpu"),
        "synthetic",
    )
    assert result.optimizer_steps > 0
    assert set(result.training_image_ids) <= set(fit_ids)
    assert len(result.training_image_ids) == len(data.ids("fit", 2, (1,))) + 4
    for key, value in old_scorer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0.0, atol=0.0)


def test_completed_leaf_checkpoint_reloads_selected_state_without_steps(
    tmp_path: Path,
) -> None:
    data, _fit_ids = _synthetic()
    features = NodeRouterFeatures(torch.zeros(128), {}, "b" * 64, None)
    old_scorer = R0Scorer(seed=1)
    with torch.no_grad():
        old_scorer.query_weight.zero_()
        old_scorer.query_weight[0] = 1.0
    old = ScoringNode("old", old_scorer, features, (0,), (0, 1, 2, 3), 12)
    config = RouterTrainingConfig(0.02, 0.0, 16, 4, 2, 1.0, 4, 1.0)
    checkpoint = tmp_path / "leaf.pt"
    first = ScoringNode("new", R0Scorer(seed=2), features, (1,), (4, 5, 6, 7), 12)
    first_result = fit_new_leaf(
        data,
        first,
        (old,),
        2,
        config,
        1993,
        32,
        torch.device("cpu"),
        "checkpoint-resume",
        checkpoint,
    )
    expected = {key: value.clone() for key, value in first.scorer.state_dict().items()}
    resumed = ScoringNode("new", R0Scorer(seed=2), features, (1,), (4, 5, 6, 7), 12)
    resumed_result = fit_new_leaf(
        data,
        resumed,
        (old,),
        2,
        config,
        1993,
        32,
        torch.device("cpu"),
        "checkpoint-resume",
        checkpoint,
    )
    assert resumed_result == first_result
    for key, value in resumed.scorer.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0.0, atol=0.0)
