from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.macro_token_cache import (
    MacroTokenPopulation,
    MacroTokenShard,
)
from apm.continual.vision.imagenetr.macro_token_model import (
    CLASS_COUNT,
    MAXIMUM_SLOTS,
    META_INPUT_DIMENSION,
    META_SLOT_DIMENSION,
    TOKEN_COUNT,
    TOKEN_DIMENSION,
    MacroTokenClassifier,
    MacroTokenInputs,
    behavior_control_features,
    behavior_meta_features,
    class_owner_targets,
    parameter_count,
    predicted_owner_class_predictions,
)


def _inputs(batch_size: int = 3) -> MacroTokenInputs:
    slots = torch.tensor((0, 2, 5), dtype=torch.int64)
    tokens = torch.randn(batch_size, len(slots), TOKEN_COUNT, TOKEN_DIMENSION)
    raw = torch.zeros(batch_size, MAXIMUM_SLOTS, CLASS_COUNT)
    ownership = torch.zeros(MAXIMUM_SLOTS, CLASS_COUNT, dtype=torch.bool)
    for slot, classes in ((0, range(0, 4)), (2, range(4, 8)), (5, range(8, 12))):
        class_ids = torch.tensor(tuple(classes))
        raw[:, slot, class_ids] = torch.randn(batch_size, len(class_ids))
        ownership[slot, class_ids] = True
    active = ownership.any(dim=1)
    return MacroTokenInputs(
        tokens,
        slots,
        behavior_meta_features(raw, ownership, active),
        raw,
        ownership,
        active,
        ownership.any(dim=0),
    )


def test_macro_token_parameter_counts_and_nested_initialization() -> None:
    one = MacroTokenClassifier(1, 0.1, 1993)
    two = MacroTokenClassifier(2, 0.1, 1993)
    assert parameter_count(one) == 12_055_496
    assert parameter_count(two) == 19_143_368
    shared = set(one.state_dict()) & set(two.state_dict())
    assert shared
    assert all(torch.equal(one.state_dict()[name], two.state_dict()[name]) for name in shared)


def test_positionwise_slot_projection_equals_dense_zero_padded_linear() -> None:
    inputs = _inputs(2)
    model = MacroTokenClassifier(1, 0.0, 11)
    captured: list[torch.Tensor] = []
    handle = model.encoder.blocks[0].register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach())
    )
    try:
        model(inputs)
    finally:
        handle.remove()
    padded = torch.zeros(2, TOKEN_COUNT, MAXIMUM_SLOTS, TOKEN_DIMENSION)
    padded[:, :, inputs.slot_indices] = inputs.node_tokens.transpose(1, 2)
    expected = F.linear(
        padded.flatten(2),
        model.encoder.slot_projection_weight.reshape(TOKEN_DIMENSION, -1),
        model.encoder.slot_projection_bias,
    ) + model.encoder.position_embedding
    assert torch.allclose(captured[0][:, :TOKEN_COUNT], expected, atol=1e-5, rtol=1e-5)
    assert captured[0].shape == (2, TOKEN_COUNT + 1, TOKEN_DIMENSION)


def test_meta_and_v6_control_layout_preserve_stable_slots() -> None:
    inputs = _inputs(2)
    assert inputs.meta_features.shape == (2, META_INPUT_DIMENSION)
    slot_meta = inputs.meta_features.reshape(2, MAXIMUM_SLOTS, META_SLOT_DIMENSION)
    assert torch.equal(slot_meta[:, :, -1].bool(), inputs.active_slot_mask[None].expand(2, -1))
    features, baseline = behavior_control_features(inputs)
    assert features.shape == (2, 8_214)
    assert baseline.shape == (2, CLASS_COUNT)
    assert torch.isneginf(baseline[:, 12:]).all()
    assert torch.equal(features.reshape(2, MAXIMUM_SLOTS, 1369)[:, 1], torch.zeros(2, 1369))


def test_owner_targets_and_predicted_owner_routing_are_separate_from_inputs() -> None:
    inputs = _inputs(3)
    labels = torch.tensor((1, 6, 10), dtype=torch.int64)
    owners = class_owner_targets(labels, inputs.ownership)
    assert owners.tolist() == [0, 2, 5]
    predictions = predicted_owner_class_predictions(inputs, owners)
    expected = torch.tensor(
        [
            inputs.raw_scores[0, 0, :4].argmax(),
            4 + inputs.raw_scores[1, 2, 4:8].argmax(),
            8 + inputs.raw_scores[2, 5, 8:12].argmax(),
        ]
    )
    assert torch.equal(predictions, expected)
    assert not hasattr(inputs, "labels")
    assert not hasattr(inputs, "owner_targets")


def _row(index: int, label: int) -> ImageRecord:
    identity = f"{index + 1:064x}"
    return ImageRecord(
        identity,
        f"class/image-{index}.jpg",
        f"train/class/image-{index}.jpg",
        identity,
        "class",
        label,
        label,
        label // 4,
        "train",
        f"{index + 101:064x}",
        1,
    )


def test_streaming_population_loads_aligned_node_shards(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    rows = tuple(_row(index, index) for index in range(4))
    nodes = tuple(
        BehaviorNode(
            f"{slot + 11:064x}",
            slot,
            (slot,),
            {},
            ClassifierRows(
                tuple(range(4 * slot, 4 * slot + 4)),
                torch.randn(4, TOKEN_DIMENSION),
                torch.randn(4),
            ),
        )
        for slot in (0, 1)
    )
    paths = tuple(tmp_path / f"node-{index}.safetensors" for index in range(2))
    for index, path in enumerate(paths):
        save_file(
            {
                "tokens": torch.full(
                    (4, TOKEN_COUNT, TOKEN_DIMENSION), index + 1, dtype=torch.bfloat16
                ),
                "raw_scores": torch.randn(4, 4),
            },
            path,
        )
    population = MacroTokenPopulation(
        "1" * 64,
        "2" * 64,
        "fit",
        nodes,
        (0, 1),
        (MacroTokenShard(rows, paths),),
        0,
        8,
        8,
        sum(path.stat().st_size for path in paths),
    )
    supervision = population.load(population.shards[0])
    assert supervision.inputs.node_tokens.shape == (4, 2, TOKEN_COUNT, TOKEN_DIMENSION)
    assert supervision.labels.tolist() == [0, 1, 2, 3]
    assert supervision.owner_targets.tolist() == [0, 0, 0, 0]
    assert supervision.inputs.active_slot_mask.tolist() == [True, True, False, False, False, False]
