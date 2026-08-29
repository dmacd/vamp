from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from pyrsistent import pmap

from apm.continual.logt_behavioral_router import (
    LevelSlotRouter,
    build_router_supervision,
    create_condition_state,
    router_selections,
    sample_example_balanced,
    sample_range_balanced,
    train_condition,
)
from apm.continual.logt_evidence_bank import empty_logt_state, insert_block
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    top_two_base_state,
    top_two_hidden_logits,
    top_two_logits,
    zero_top_two_adapter,
)
from apm.experiments.vamp_logt_router_config import load_config
from apm.experiments.vamp_logt_router_data import ExampleBatch, build_stream_allocations
from apm.experiments.vamp_logt_router_metrics import fixed_policy_selections, routing_metric_rows


def _topology(blocks: int, block_size: int = 1):
    state = empty_logt_state(block_size)
    for block in range(blocks):
        state, _leaf, _merges = insert_block(
            state,
            tuple(range(block * block_size, (block + 1) * block_size)),
        )
    return state


def _batch(steps: tuple[int, ...]) -> ExampleBatch:
    rows = len(steps)
    return ExampleBatch(
        torch.arange(rows * 784, dtype=torch.float32).reshape(rows, 1, 28, 28) / max(rows * 784, 1),
        torch.arange(rows, dtype=torch.int64) % 10,
        torch.zeros(rows, dtype=torch.int64),
        torch.arange(rows, dtype=torch.int64),
        torch.tensor(steps, dtype=torch.int64),
    )


def _supervision(rows: int = 8):
    topology = _topology(3)
    base = top_two_base_state(
        torch.randn(2, 3) * 0.01,
        torch.zeros(2),
        torch.randn(10, 2) * 0.01,
        torch.zeros(10),
    )
    adapters = pmap(
        {node.node_id: zero_top_two_adapter(base) for node in topology.active_nodes}
    )
    labels = torch.arange(rows, dtype=torch.int64) % 10
    result = build_router_supervision(
        topology.active_nodes,
        adapters,
        torch.randn(rows, 3),
        labels,
        base,
        7,
        0.10,
        torch.device("cpu"),
        4,
    )
    return topology, base, adapters, labels, result


def test_top_two_hidden_forward_is_exactly_the_existing_logit_forward() -> None:
    base = top_two_base_state(
        torch.randn(4, 3), torch.randn(4), torch.randn(10, 4), torch.randn(10)
    )
    adapter = zero_top_two_adapter(base)
    features = torch.randn(5, 3)
    hidden, logits = top_two_hidden_logits(features, base, adapter)
    assert hidden.shape == (5, 4)
    assert torch.equal(logits, top_two_logits(features, base, adapter))


def test_slot_features_targets_and_mask_are_label_isolated_and_deterministic() -> None:
    topology, _base, _adapters, _labels, supervision = _supervision()
    assert supervision.features.shape == (8, 7 * (2 + 10 + 1))
    slots = supervision.features.reshape(8, 7, 13)
    active = {node.level for node in topology.active_nodes}
    assert supervision.active_mask.tolist() == [level in active for level in range(7)]
    assert torch.equal(slots[:, 2:], torch.zeros_like(slots[:, 2:]))
    assert torch.equal(slots[:, 0, -1], torch.ones(8))
    assert torch.equal(slots[:, 1, -1], torch.ones(8))
    assert supervision.hard_targets.tolist() == [0] * 8
    assert torch.allclose(supervision.soft_targets.sum(dim=1), torch.ones(8))
    assert torch.equal(
        supervision.soft_targets[:, ~supervision.active_mask],
        torch.zeros_like(supervision.soft_targets[:, ~supervision.active_mask]),
    )
    assert not any(tensor.requires_grad for tensor in supervision.tensors)


def test_archived_targets_are_recomputed_after_a_frontier_carry() -> None:
    base = top_two_base_state(
        torch.zeros((2, 3)),
        torch.ones(2),
        torch.zeros((10, 2)),
        torch.zeros(10),
    )
    trunk = torch.zeros((5, 3))
    labels = torch.zeros(5, dtype=torch.int64)
    before = _topology(3)
    before_adapters = {
        node.node_id: TopTwoAdapterState(
            *zero_top_two_adapter(base).tensors[:3],
            torch.nn.functional.one_hot(
                torch.tensor(1 if node.level == 0 else 0), num_classes=10
            ).to(torch.float32),
        )
        for node in before.active_nodes
    }
    old_targets = build_router_supervision(
        before.active_nodes,
        before_adapters,
        trunk,
        labels,
        base,
        7,
        0.10,
        torch.device("cpu"),
        5,
    ).hard_targets
    after = _topology(4)
    after_adapters = {
        node.node_id: zero_top_two_adapter(base) for node in after.active_nodes
    }
    new_supervision = build_router_supervision(
        after.active_nodes,
        after_adapters,
        trunk,
        labels,
        base,
        7,
        0.10,
        torch.device("cpu"),
        5,
    )
    assert old_targets.tolist() == [1] * 5
    assert new_supervision.hard_targets.tolist() == [2] * 5
    assert new_supervision.active_mask.tolist() == [False, False, True, False, False, False, False]


def test_masked_router_can_never_select_an_inactive_level() -> None:
    router = LevelSlotRouter(6, 4, (8, 6, 4), 0.0)
    with torch.no_grad():
        router.network[-1].bias.copy_(torch.tensor([0.0, 100.0, 50.0, 25.0]))
    features = torch.randn(7, 6)
    mask = torch.tensor([True, False, True, False])
    selections = router(features, mask).argmax(dim=1)
    assert set(selections.tolist()) <= {0, 2}


def test_example_and_range_balancing_have_the_required_sampling_mass() -> None:
    archive = _batch(tuple(step for step in range(1, 7) for _row in range(5)))
    topology = _topology(7, block_size=5)
    example = sample_example_balanced(archive, 6_000, 11, 7)
    ranged = sample_range_balanced(archive, topology.active_nodes, 6_000, 11, 7)
    example_early = float((example.batch.macro_steps <= 4).float().mean().item())
    range_early = float((ranged.batch.macro_steps <= 4).float().mean().item())
    assert example_early == pytest.approx(2.0 / 3.0, abs=0.03)
    assert range_early == pytest.approx(0.5, abs=0.03)
    assert len(example.archive_indices) == len(ranged.archive_indices) == 6_000
    assert sum(row[2] for row in ranged.range_draw_counts) == 6_000
    assert max(ranged.batch.macro_steps.tolist()) < 7


def test_router_training_states_are_independent_and_keep_node_tensors_frozen() -> None:
    _topology_state, _base, adapters, _labels, supervision = _supervision(16)
    protocol = load_config("configs/vamp_logt_router_mnist/primary.yaml")
    router_config = replace(
        protocol.router,
        hidden_widths=(16, 8, 4),
        dropout=0.0,
        minibatch_size=8,
    )
    left = create_condition_state("example_hard", supervision.features.shape[1], router_config, 3, torch.device("cpu"))
    right = create_condition_state("range_hard", supervision.features.shape[1], router_config, 3, torch.device("cpu"))
    before = tuple(tensor.clone() for adapter in adapters.values() for tensor in adapter.tensors)
    result = train_condition(
        left,
        supervision,
        supervision,
        "hard",
        2,
        router_config,
        3,
        4,
        torch.device("cpu"),
    )
    assert result.optimizer_steps == 8
    assert left.optimizer_steps == 8
    assert right.optimizer_steps == 0
    assert all(
        torch.equal(previous, current)
        for previous, current in zip(
            before,
            (tensor for adapter in adapters.values() for tensor in adapter.tensors),
        )
    )
    selections, _probabilities, _inactive = router_selections(
        left.router, supervision, torch.device("cpu"), 8
    )
    assert set(selections.tolist()) <= {0, 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not visible")
def test_soft_router_training_keeps_cpu_supervision_device_safe_on_cuda() -> None:
    _topology_state, _base, _adapters, _labels, supervision = _supervision(16)
    protocol = load_config("configs/vamp_logt_router_mnist/primary.yaml")
    router_config = replace(
        protocol.router,
        hidden_widths=(16, 8, 4),
        dropout=0.0,
        minibatch_size=8,
    )
    device = torch.device("cuda")
    state = create_condition_state(
        "example_soft",
        supervision.features.shape[1],
        router_config,
        3,
        device,
    )
    result = train_condition(
        state,
        supervision,
        supervision,
        "soft",
        1,
        router_config,
        3,
        4,
        device,
    )
    assert result.mean_last_epoch_loss >= 0.0


def test_metric_rows_keep_oracle_dominance_and_zero_oracle_regret() -> None:
    topology, _base, _adapters, labels, supervision = _supervision(10)
    examples = ExampleBatch(
        torch.zeros((10, 1, 28, 28)),
        labels,
        torch.zeros(10, dtype=torch.int64),
        torch.arange(10, dtype=torch.int64),
        torch.arange(1, 11, dtype=torch.int64),
    )
    oracle = fixed_policy_selections("oracle", topology.active_nodes, supervision, 0)
    rows = routing_metric_rows(
        condition="oracle",
        selections=oracle,
        probabilities=None,
        inactive_attempts=None,
        examples=examples,
        supervision=supervision,
        nodes=topology.active_nodes,
        run_seed=0,
        macro_step=10,
        evaluation_scope="evaluation_archive",
        near_oracle_thresholds=(0.01, 0.05, 0.10),
    )
    assert rows[0]["mean_regret"] == pytest.approx(0.0)
    assert rows[0]["oracle_match_rate"] == 1.0
    assert rows[0]["selected_mean_cross_entropy"] == rows[0]["oracle_mean_cross_entropy"]


def test_stream_allocations_are_blockwise_complete_disjoint_and_seed_varying() -> None:
    config = load_config("configs/vamp_logt_router_mnist/primary.yaml")
    left = build_stream_allocations(config.benchmark, 0)
    repeated = build_stream_allocations(config.benchmark, 0)
    right = build_stream_allocations(config.benchmark, 1)
    assert left == repeated
    assert left != right
    assert len(left) == 64
    for offset in range(0, 64, 8):
        assert {row.domain_id for row in left[offset : offset + 8]} == set(range(8))
    for domain in range(8):
        rows = tuple(
            value
            for allocation in left
            if allocation.domain_id == domain
            for value in (
                allocation.model_indices
                + allocation.router_indices
                + allocation.evaluation_indices
            )
        )
        assert len(rows) == len(set(rows)) == 8 * 640


def test_primary_config_rejects_unknown_scientific_keys(tmp_path: Path) -> None:
    source = Path("configs/vamp_logt_router_mnist/primary.yaml")
    record = yaml.safe_load(source.read_text(encoding="utf-8"))
    record["router"]["hidden_choice"] = 7
    invalid = tmp_path / "configs" / "vamp_logt_router_mnist" / "primary.yaml"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(yaml.safe_dump(record), encoding="utf-8")
    with pytest.raises(ValueError, match="router keys"):
        load_config(invalid)
