from dataclasses import replace

from pyrsistent import pmap
import torch

from apm.continual.logt_behavioral_integrator import (
    FullReplayConvergenceConfig,
    IntegratorConditionState,
    IntegratorSupervision,
    LevelSlotIntegrator,
    build_base_observations,
    build_node_observations,
    inactive_slots_are_zero,
    prediction_logits,
    train_converged_full_replay,
    train_condition,
)
from apm.continual.logt_evidence_bank import empty_logt_state, insert_block
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    top_two_base_state,
    zero_top_two_adapter,
)
from apm.experiments.vamp_logt_integrator_metrics import (
    fixed_control_logits,
    prediction_metric_rows,
)
from apm.experiments.vamp_logt_integrator_rotated_config import load_config
from apm.experiments.vamp_logt_integrator_rotated_reporting import (
    _retention_means,
)
from apm.experiments.vamp_logt_router_data import ExampleBatch


def _topology(blocks: int):
    state = empty_logt_state(1)
    for block in range(blocks):
        state, _leaf, _merges = insert_block(state, (block,))
    return state


def _observations(blocks: int = 3, rows: int = 8):
    torch.manual_seed(19)
    topology = _topology(blocks)
    base = top_two_base_state(
        torch.randn(2, 3) * 0.1,
        torch.randn(2) * 0.1,
        torch.randn(10, 2) * 0.1,
        torch.randn(10) * 0.1,
    )
    adapters = pmap(
        {
            node.node_id: TopTwoAdapterState(
                *zero_top_two_adapter(base).tensors[:3],
                torch.nn.functional.one_hot(
                    torch.tensor(node.level), num_classes=10
                ).to(torch.float32),
            )
            for node in topology.active_nodes
        }
    )
    trunk = torch.randn(rows, 3)
    observations = build_node_observations(
        topology.active_nodes,
        adapters,
        trunk,
        base,
        7,
        torch.device("cpu"),
        4,
    )
    return topology, base, adapters, trunk, observations


def _small_state(input_dim: int, slot_dim: int) -> IntegratorConditionState:
    integrator = LevelSlotIntegrator(input_dim, 7, slot_dim, (16, 8, 4), 0.0)
    return IntegratorConditionState(
        "integrator_example_replay",
        integrator,
        torch.optim.AdamW(integrator.parameters(), lr=1.0e-3),
    )


def test_fixed_slots_are_label_free_and_inactive_slots_are_exactly_zero() -> None:
    topology, _base, _adapters, _trunk, observations = _observations()
    assert observations.features.shape == (8, 7 * 13)
    assert observations.active_mask.tolist() == [True, True, False, False, False, False, False]
    assert inactive_slots_are_zero(observations, 13)
    slots = observations.features.reshape(8, 7, 13)
    assert torch.equal(slots[:, :2, -1], torch.ones((8, 2)))
    assert {node.level for node in topology.active_nodes} == {0, 1}

    labels = torch.arange(8, dtype=torch.int64) % 10
    reversed_labels = labels.flip(0)
    assert torch.equal(
        IntegratorSupervision(observations, labels).observations.features,
        IntegratorSupervision(observations, reversed_labels).observations.features,
    )


def test_zero_residual_has_exact_mean_ensemble_and_one_node_parity() -> None:
    topology, _base, _adapters, _trunk, observations = _observations()
    state = _small_state(observations.features.shape[1], 13)
    assert torch.equal(
        state.integrator.input_layer.weight[:, 13:],
        torch.zeros_like(state.integrator.input_layer.weight[:, 13:]),
    )
    logits = prediction_logits(state.integrator, observations, torch.device("cpu"), 4)
    assert torch.equal(logits, observations.baseline_log_probabilities)
    expected_mean = torch.logsumexp(
        observations.node_log_probabilities[:, observations.active_mask], dim=1
    ) - torch.log(torch.tensor(float(len(topology.active_nodes))))
    assert torch.equal(logits, expected_mean)

    _single_topology, _base, _adapters, _trunk, single = _observations(blocks=4)
    single_logits = prediction_logits(
        state.integrator, single, torch.device("cpu"), 4
    )
    active_level = int(torch.where(single.active_mask)[0].item())
    assert torch.equal(
        single_logits, single.node_log_probabilities[:, active_level]
    )


def test_base_control_uses_only_slot_zero_and_frozen_base_prediction() -> None:
    _topology_state, base, _adapters, trunk, _observations_result = _observations()
    observations = build_base_observations(
        trunk, base, 7, torch.device("cpu"), 4
    )
    assert observations.active_mask.tolist() == [True, False, False, False, False, False, False]
    assert inactive_slots_are_zero(observations, 13)
    assert torch.equal(
        observations.baseline_log_probabilities,
        observations.node_log_probabilities[:, 0],
    )


def test_direct_labels_survive_a_carry_while_current_node_features_change() -> None:
    _before_topology, base, _before_adapters, trunk, before = _observations(blocks=3)
    after_topology = _topology(4)
    after_adapters = pmap(
        {
            node.node_id: zero_top_two_adapter(base)
            for node in after_topology.active_nodes
        }
    )
    after = build_node_observations(
        after_topology.active_nodes,
        after_adapters,
        trunk,
        base,
        7,
        torch.device("cpu"),
        4,
    )
    labels = torch.arange(len(trunk), dtype=torch.int64) % 10
    before_supervision = IntegratorSupervision(before, labels)
    after_supervision = IntegratorSupervision(after, labels)
    assert torch.equal(before_supervision.labels, after_supervision.labels)
    assert not torch.equal(
        before_supervision.observations.features,
        after_supervision.observations.features,
    )


def test_training_states_are_independent_and_node_tensors_remain_frozen() -> None:
    _topology_state, _base, adapters, _trunk, observations = _observations(rows=16)
    labels = torch.arange(16, dtype=torch.int64) % 10
    supervision = IntegratorSupervision(observations, labels)
    left = _small_state(observations.features.shape[1], 13)
    right = _small_state(observations.features.shape[1], 13)
    before_left = tuple(parameter.clone() for parameter in left.integrator.parameters())
    before_right = tuple(
        parameter.clone() for parameter in right.integrator.parameters()
    )
    before_nodes = {
        node_id: tuple(tensor.clone() for tensor in adapters[node_id].tensors)
        for node_id in sorted(adapters)
    }
    protocol = load_config(
        "configs/vamp_logt_integrator_rotated_mnist/primary.yaml"
    )
    training_config = replace(protocol.integrator, dropout=0.0, minibatch_size=8)
    result = train_condition(
        left,
        supervision,
        supervision,
        2,
        training_config,
        3,
        4,
        torch.device("cpu"),
    )
    assert result.optimizer_steps == 8
    assert result.objective_after < result.objective_before
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_left, left.integrator.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_right, right.integrator.parameters(), strict=True)
    )
    assert all(
        torch.equal(before, after)
        for node_id in sorted(adapters)
        for before, after in zip(
            before_nodes[node_id], adapters[node_id].tensors, strict=True
        )
    )


def test_full_replay_convergence_uses_every_example_and_restores_the_best() -> None:
    _topology_state, _base, _adapters, _trunk, observations = _observations(rows=16)
    training = IntegratorSupervision(
        observations, torch.arange(16, dtype=torch.int64) % 10
    )
    validation = IntegratorSupervision(
        observations, torch.arange(16, dtype=torch.int64).flip(0) % 10
    )
    protocol = load_config("configs/vamp_logt_integrator_rotated_mnist/primary.yaml")
    training_config = replace(protocol.integrator, dropout=0.0, minibatch_size=8)
    convergence = FullReplayConvergenceConfig(
        minimum_epochs=1,
        maximum_epochs=12,
        improvement_delta=100.0,
        learning_rate_patience=1,
        learning_rate_factor=0.5,
        minimum_learning_rate=2.5e-4,
        convergence_patience=2,
    )
    torch.manual_seed(41)
    state = _small_state(observations.features.shape[1], 13)
    result = train_converged_full_replay(
        state,
        training,
        validation,
        training_config,
        convergence,
        5,
        3,
        torch.device("cpu"),
    )
    restored_logits = prediction_logits(
        state.integrator, validation.observations, torch.device("cpu"), 8
    )
    restored_loss = torch.nn.functional.cross_entropy(
        restored_logits, validation.labels
    ).item()
    assert result.converged
    assert result.stop_reason == "minimum_learning_rate_plateau"
    assert result.epochs_ran == 4
    assert result.training_example_presentations == 4 * len(training.labels)
    assert result.validation_example_presentations == 6 * len(validation.labels)
    assert result.best_validation_loss == min(
        row.validation_loss for row in result.history
    )
    assert abs(restored_loss - result.best_validation_loss) < 1.0e-7


def test_full_replay_safety_cap_is_not_reported_as_convergence() -> None:
    _topology_state, _base, _adapters, _trunk, observations = _observations(rows=8)
    supervision = IntegratorSupervision(
        observations, torch.arange(8, dtype=torch.int64) % 10
    )
    protocol = load_config("configs/vamp_logt_integrator_rotated_mnist/primary.yaml")
    training_config = replace(protocol.integrator, dropout=0.0, minibatch_size=4)
    torch.manual_seed(43)
    state = _small_state(observations.features.shape[1], 13)
    result = train_converged_full_replay(
        state,
        supervision,
        supervision,
        training_config,
        FullReplayConvergenceConfig(1, 2, 100.0, 1, 0.5, 1.0e-8, 2),
        7,
        2,
        torch.device("cpu"),
    )
    assert not result.converged
    assert result.stop_reason == "maximum_epochs"
    assert result.epochs_ran == 2


def test_fixed_controls_use_stable_level_slots() -> None:
    topology, _base, _adapters, _trunk, observations = _observations()
    labels = torch.arange(len(observations.features), dtype=torch.int64) % 10
    mean = fixed_control_logits(
        "mean_ensemble", topology.active_nodes, observations, labels, 7
    )
    recent = fixed_control_logits(
        "most_recent_range", topology.active_nodes, observations, labels, 7
    )
    assert torch.equal(mean, observations.baseline_log_probabilities)
    recent_level = max(topology.active_nodes, key=lambda node: node.last_block).level
    assert torch.equal(recent, observations.node_log_probabilities[:, recent_level])


def test_temporal_range_and_retention_aggregation_use_the_live_frontier() -> None:
    topology, _base, _adapters, _trunk, observations = _observations(rows=4)
    labels = torch.arange(4, dtype=torch.int64)
    examples = ExampleBatch(
        torch.zeros((4, 1, 28, 28)),
        labels,
        torch.zeros(4, dtype=torch.int64),
        torch.arange(4, dtype=torch.int64),
        torch.tensor((1, 1, 2, 3), dtype=torch.int64),
    )
    flat_logits = torch.zeros((4, 10))
    correct_logits = torch.nn.functional.one_hot(labels, num_classes=10).to(
        torch.float32
    ) * 5.0
    conditions = (
        ("integrator_no_replay", flat_logits),
        ("integrator_example_replay", correct_logits),
    )
    rows = tuple(
        {
            **row,
            "row_type": "evaluation",
        }
        for condition, logits in conditions
        for row in prediction_metric_rows(
            condition=condition,
            logits=logits,
            examples=examples,
            node_observations=observations,
            nodes=topology.active_nodes,
            run_seed=0,
            macro_step=15,
            evaluation_scope="evaluation_archive",
        )
    )
    no_replay = tuple(
        row for row in rows if row["condition"] == "integrator_no_replay"
    )
    counts = {str(row["group"]): int(row["example_count"]) for row in no_replay}
    assert counts["range:1-2"] == 3
    assert counts["range:3-3"] == 1
    assert counts["current_range"] == 1
    assert counts["older_ranges"] == 3
    range_losses = [
        float(row["mean_cross_entropy"])
        for row in no_replay
        if str(row["group"]).startswith("range:")
    ]
    micro = next(row for row in no_replay if row["group"] == "micro")
    assert float(micro["range_macro_mean_cross_entropy"]) == sum(range_losses) / 2
    assert float(micro["worst_range_mean_cross_entropy"]) == max(range_losses)
    retention = _retention_means(rows)
    assert retention["integrator_example_replay"]["older_mean_cross_entropy"] < retention[
        "integrator_no_replay"
    ]["older_mean_cross_entropy"]
    assert retention["integrator_example_replay"]["current_accuracy_loss_pp"] < 0.0
