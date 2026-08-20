import torch
from torch import nn

from apm.continual.vision.imagenetr.calibration import (
    CalibrationExamples,
    fit_affine_calibration,
)
from apm.continual.vision.imagenetr.heads import ClassifierRows, union_classifier_rows
from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.lora import LoRALinear
from apm.continual.vision.imagenetr.routing import (
    GroundTruth,
    NodeScores,
    TaskFreeQuery,
    exhaustive_predictions,
    routed_node_predictions,
    true_node_oracle_predictions,
)


def test_zero_lora_has_exact_frozen_affine_parity_and_only_factors_train() -> None:
    base = nn.Linear(5, 7)
    layer = LoRALinear(base, rank=3, alpha=3, initialization_seed=4)
    inputs = torch.randn(11, 5)
    torch.testing.assert_close(layer(inputs), base(inputs), rtol=0.0, atol=0.0)
    layer(inputs).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.lora_a.grad is not None
    assert layer.lora_b.grad is not None


def test_classifier_union_preserves_disjoint_affine_rows_exactly() -> None:
    left = ClassifierRows((0, 2), torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0.1, 0.2]))
    right = ClassifierRows((1, 3), torch.tensor([[5.0, 6.0], [7.0, 8.0]]), torch.tensor([0.3, 0.4]))
    union = union_classifier_rows((left, right))
    assert union.class_ids == (0, 1, 2, 3)
    torch.testing.assert_close(
        union.weight,
        torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]),
    )
    torch.testing.assert_close(union.bias, torch.tensor([0.1, 0.3, 0.2, 0.4]))


def test_restoring_inactive_rows_blocks_sgd_weight_decay_drift() -> None:
    classifier = AffineClassifier((0, 1, 2, 3), 3, initialization_seed=8)
    frozen = classifier.selected_rows((0, 1))
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, weight_decay=0.2)
    optimizer.zero_grad(set_to_none=True)
    classifier(torch.ones(4, 3)).sum().backward()
    classifier.mask_inactive_gradients((2, 3))
    optimizer.step()
    classifier.restore_rows(frozen)
    restored = classifier.selected_rows((0, 1))
    torch.testing.assert_close(restored.weight, frozen.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored.bias, frozen.bias, rtol=0.0, atol=0.0)


def test_task_free_modes_and_true_node_oracle_are_separate() -> None:
    query = TaskFreeQuery(("x", "y"))
    truth = GroundTruth(query.image_ids, torch.tensor([0, 2]))
    nodes = (
        NodeScores("a" * 64, (0, 1), torch.tensor([[3.0, 1.0], [1.0, 0.0]]), torch.tensor([[2.0, 0.0], [1.0, 0.0]])),
        NodeScores("b" * 64, (2, 3), torch.tensor([[2.0, 0.0], [0.0, -1.0]]), torch.tensor([[1.0, 0.0], [0.0, -1.0]])),
    )
    assert exhaustive_predictions(query, nodes, "raw").tolist() == [0, 0]
    assert routed_node_predictions(query, ("b" * 64, "a" * 64), nodes).tolist() == [2, 0]
    assert true_node_oracle_predictions(query, truth, nodes).tolist() == [0, 2]


def test_affine_calibration_is_deterministic_positive_and_zero_mean() -> None:
    examples = CalibrationExamples(("a", "b", "c", "d"), torch.tensor([0, 1, 2, 3]))
    logits = {
        "a" * 64: torch.tensor([[4.0, 1.0], [1.0, 4.0], [3.0, 2.0], [3.0, 2.0]]),
        "b" * 64: torch.tensor([[0.2, 0.1], [0.1, 0.2], [0.8, 0.1], [0.1, 0.8]]),
    }
    classes = {"a" * 64: (0, 1), "b" * 64: (2, 3)}
    first = fit_affine_calibration(examples, logits, classes, steps=20)
    second = fit_affine_calibration(examples, logits, classes, steps=20)
    assert first == second
    assert all(value > 0.0 for value in first.temperatures)
    assert abs(sum(first.offsets)) < 1e-8
