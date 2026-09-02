from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pyrsistent import pmap

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.dense_mlp_adapter import (
    DenseConvergenceConfig,
    DenseExamples,
    DenseMlpState,
    DenseMnistMLP,
    DenseOptimizerConfig,
    dense_hidden_logits,
    dense_state,
    fit_dense_model,
    zero_dense_delta,
)
from apm.continual.logt_behavioral_integrator import (
    IntegratorObservations,
    IntegratorSupervision,
    create_condition_state,
    train_condition,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_mlp_permuted_amended_reporting import (
    load_analysis_seeds,
    write_amended_results,
)
from apm.experiments.vamp_logt_mlp_permuted_calibration import (
    initialize_dense_state,
    select_calibrated_width,
)
from apm.experiments.vamp_logt_mlp_permuted_ceiling import (
    run_baseline_extension,
    run_ceiling,
)
from apm.experiments.vamp_logt_mlp_permuted_config import load_config
from apm.experiments.vamp_logt_mlp_permuted_data import (
    PermutedMnistBenchmark,
    StepAllocation,
    build_stream_allocations,
    stratified_source_split,
)
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import (
    DenseFrontier,
    _empty_dense_logt_state,
    _insert_dense_block,
    build_dense_observations,
    build_hierarchy_tape,
    load_frontier,
)
from apm.experiments.vamp_logt_mlp_permuted_online import run_online_seed
from apm.experiments.vamp_logt_mlp_permuted_reporting import (
    CONDITION_PROTOCOL,
    PLOT_STYLES,
    write_results,
)
from apm.experiments.vamp_logt_mlp_permuted_scaling import (
    _capacity_cumulative_series,
    _growth_fits,
    _run_scaling,
    _run_scaling_seed,
    _shared_hierarchy_work,
    _timed_state_creation,
)
from apm.experiments.vamp_logt_router_reporting import _html


def _benchmark() -> PermutedMnistBenchmark:
    generator = torch.Generator().manual_seed(17)
    return PermutedMnistBenchmark(
        torch.rand((12, 1, 28, 28), generator=generator),
        torch.arange(12, dtype=torch.int64) % 10,
        torch.rand((5, 1, 28, 28), generator=generator),
        torch.arange(5, dtype=torch.int64) % 10,
        tuple(torch.roll(torch.arange(784), shifts=domain) for domain in range(8)),
        (
            StepAllocation(1, 0, (0, 1), (2, 3), (4,)),
            StepAllocation(2, 1, (0, 1), (2, 3), (4,)),
        ),
        ((0, 1),) * 8,
    )


def test_shared_html_renderer_emits_semantic_tables(tmp_path: Path) -> None:
    rendered = _html(
        "\n".join(
            (
                "# Report",
                "",
                "| Condition | Accuracy |",
                "|---|---:|",
                "| A < B | 98.2% |",
            )
        ),
        tmp_path,
        "Report",
    )

    assert '<div class="table-scroll"><table>' in rendered
    assert '<th scope="col" class="align-right">Accuracy</th>' in rendered
    assert '<td class="align-left">A &lt; B</td>' in rendered
    assert '<td class="align-right">98.2%</td>' in rendered
    assert '<p class="table-row">| Condition | Accuracy |</p>' not in rendered


def _tiny_config(tmp_path: Path):
    source = load_config("configs/vamp_logt_mlp_permuted_mnist/primary.yaml")
    return replace(
        source,
        artifact_root=tmp_path / "artifacts",
        benchmark=replace(
            source.benchmark,
            macro_steps=2,
            model_batch_size=2,
            observer_batch_size=2,
            evaluation_batch_size=1,
        ),
        calibration=replace(
            source.calibration,
            candidate_widths=((8, 6, 4),),
            seeds=(0,),
            dropout=0.0,
        ),
        node=replace(
            source.node,
            epochs=1,
            dropout=0.0,
            optimizer=DenseOptimizerConfig(0.001, 0.0001, 4, 1.0),
        ),
        observer=replace(source.observer, maximum_levels=2, inference_batch_size=8),
        router=replace(
            source.router,
            maximum_levels=2,
            hidden_widths=(8, 6, 4),
            dropout=0.0,
            minibatch_size=2,
            epochs_per_step=1,
        ),
        integrator=replace(
            source.integrator,
            maximum_levels=2,
            hidden_widths=(8, 6, 4),
            dropout=0.0,
            minibatch_size=2,
            epochs_per_step=1,
            offline_epochs=1,
        ),
        online=replace(source.online, seeds=(0,), historical_budget=2),
        ceiling=replace(
            source.ceiling,
            restarts_per_step=2,
            convergence=DenseConvergenceConfig(1, 8, 100.0, 1, 0.5, 0.00025, 1),
        ),
        evaluation=replace(
            source.evaluation,
            test_subset_per_domain=2,
            full_checkpoints=(2,),
            headline_checkpoints=(2,),
        ),
        runtime=replace(source.runtime, device="cpu", progress=False),
    )


def _publish_base(config, run_root: Path) -> DenseMlpState:
    base = initialize_dense_state((8, 6, 4), 0.0, 71)
    path = run_root / "base" / "model.pt"
    atomic_torch_save(
        path,
        {
            "config_hash": config.config_hash,
            "hidden_widths": (8, 6, 4),
            "parameters": base.tensors,
            "schema_version": "vamp-logt-dense-base-v1",
        },
    )
    publish_immutable_json(
        run_root / "calibration" / "summary.json",
        {
            "base_checkpoint_sha256": file_sha256(path),
            "config_hash": config.config_hash,
            "selected_hidden_widths": [8, 6, 4],
            "status": "complete",
        },
    )
    return base


def test_production_protocol_matches_the_frozen_dense_plan() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_mnist/primary.yaml")
    assert config.calibration.candidate_widths == (
        (1024, 1024, 512),
        (1536, 1536, 768),
        (2048, 2048, 1024),
    )
    assert config.calibration.seeds == (0, 1, 2)
    assert config.node.epochs == 20
    assert config.online.seeds == (0, 1, 2, 3, 4)
    assert config.online.historical_budget == 256
    assert config.ceiling.restarts_per_step == 3
    assert config.evaluation.headline_checkpoints == (15, 31, 63)


def test_scaling_protocol_is_exactly_five_seeds_two_policies_and_two_conditions() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_100_scaling.yaml")
    assert config.artifact_root == Path(
        "artifacts/vamp-logt-mlp-permuted-mnist-100-scaling-five-seed"
    ).resolve()
    assert config.calibration_evidence_run == Path(
        "artifacts/vamp-logt-mlp-permuted-mnist-ungated/runs/"
        "d38c612562699eb55c578a48a8ea94639596c4009a539c1092b63b76eb4f26c0"
    ).resolve()
    assert config.benchmark.domain_count == config.benchmark.macro_steps == 100
    assert config.benchmark.permutation_seeds == tuple(range(1001, 1100))
    assert config.online.seeds == (0, 1, 2, 3, 4)
    assert config.scaling is not None
    assert config.scaling.hierarchy_node_capacities == (1, 2)
    assert config.scaling.predecessor_run.name == (
        "6d4f13fdf7d3aad964b1d8becae3fa7130e05422548576912e6e830506a3710c"
    )
    assert config.integrator.conditions == ("integrator_uniform_replay",)
    assert config.integrator.offline_epochs == 20
    assert config.evaluation.full_checkpoints == (1, 2, 4, 8, 10, 16, 26, 41, 66, 100)


def test_capacity_protocol_is_one_seed_one_policy_and_two_paired_sample_arms() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_100_capacity.yaml")
    assert config.config_hash == (
        "26adf88c61114a8cd32e6aa25dcc2c4aa8bc62b7c020db9878a9d42281492dd2"
    )
    assert config.online.seeds == (0,)
    assert config.scaling is not None
    assert config.scaling.hierarchy_node_capacities == (1,)
    assert config.scaling.training_sample_multipliers == (1, 2)
    assert config.scaling.base_hidden_widths == (2272, 2272, 1136)
    assert config.integrator.hidden_widths == (1912, 956, 478)
    assert config.integrator.conditions == ("integrator_uniform_replay",)
    base = DenseMnistMLP(config.scaling.base_hidden_widths, config.calibration.dropout)
    assert sum(parameter.numel() for parameter in base.parameters()) == 9_541_274
    integrator = create_condition_state(
        "capacity-parameter-count",
        7 * (1136 + 11),
        1136 + 11,
        config.integrator,
        0,
        torch.device("cpu"),
    )
    assert sum(
        parameter.numel() for parameter in integrator.integrator.parameters()
    ) == 17_650_160


def test_doubled_training_allocations_retain_standard_roles_and_evaluation() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_100_capacity.yaml")
    standard = build_stream_allocations(config, 0)
    doubled = build_stream_allocations(config, 0, training_sample_multiplier=2)
    assert len(standard) == len(doubled) == 100
    for base, expanded in zip(standard, doubled, strict=True):
        assert expanded.domain_id == base.domain_id
        assert expanded.model_indices[:256] == base.model_indices
        assert expanded.observer_indices[:256] == base.observer_indices
        assert expanded.evaluation_indices == base.evaluation_indices
        assert len(expanded.model_indices) == len(expanded.observer_indices) == 512
        combined = (
            expanded.model_indices
            + expanded.observer_indices
            + expanded.evaluation_indices
        )
        assert len(combined) == len(set(combined))


def test_capacity_work_multiplier_and_t_log_fit_are_exact() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_100_capacity.yaml")
    standard = _shared_hierarchy_work(config, 8, training_sample_multiplier=1)
    doubled = _shared_hierarchy_work(config, 8, training_sample_multiplier=2)
    assert doubled["shared_forward_example_passes"] == (
        2 * standard["shared_forward_example_passes"]
    )
    steps = (1, 2, 4, 8, 10, 16, 26, 41, 66, 100)
    values = tuple(3.0 * step * __import__("math").log2(step + 1) for step in steps)
    fits = _growth_fits(steps, values)
    assert fits["t_log2_t_plus_1"]["coefficient"] == pytest.approx(3.0)
    assert fits["t_log2_t_plus_1"]["r_squared"] == pytest.approx(1.0)

    rows = tuple(
        {
            "arm": "persistent",
            "condition": "uniform",
            "macro_step": step,
            "training_seconds": value,
        }
        for step, value in enumerate((1.0, 2.0, 3.0), start=1)
    )
    cumulative_steps, cumulative_values = _capacity_cumulative_series(
        rows,
        "persistent",
        "uniform",
        "training_seconds",
    )
    assert cumulative_steps == (1, 2, 3)
    assert cumulative_values == (1.0, 3.0, 6.0)


def test_dense_benchmark_accepts_dynamic_permutation_counts() -> None:
    generator = torch.Generator().manual_seed(29)
    benchmark = PermutedMnistBenchmark(
        torch.rand((6, 1, 28, 28), generator=generator),
        torch.arange(6, dtype=torch.int64) % 10,
        torch.rand((3, 1, 28, 28), generator=generator),
        torch.arange(3, dtype=torch.int64) % 10,
        tuple(torch.roll(torch.arange(784), shifts=domain) for domain in range(100)),
        (StepAllocation(1, 99, (0, 1), (2, 3), (4,)),),
        ((0,),) * 100,
    )
    assert benchmark.step(1).observer.domain_ids.tolist() == [99, 99]
    assert benchmark.test_domain(99, full=False).domain_ids.tolist() == [99]


def test_integrator_training_reports_only_optimizer_model_passes(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    active_mask = torch.tensor((True, False))
    observations = lambda rows: IntegratorObservations(
        torch.zeros((rows, 30)),
        torch.zeros((rows, 2, 10)),
        active_mask,
        torch.zeros((rows, 10)),
    )
    current = IntegratorSupervision(observations(4), torch.tensor((0, 1, 2, 3)))
    historical = IntegratorSupervision(observations(4), torch.tensor((4, 5, 6, 7)))
    state = create_condition_state(
        "work-accounting-test",
        30,
        15,
        config.integrator,
        31,
        torch.device("cpu"),
    )
    result = train_condition(
        state,
        current,
        historical,
        2,
        config.integrator,
        31,
        1,
        torch.device("cpu"),
    )
    assert result.training_forward_example_passes == 16
    assert result.training_backward_example_passes == 16
    assert result.training_forward_calls == 16
    assert result.training_backward_calls == 8
    assert result.training_wall_seconds >= 0.0


def test_shared_hierarchy_work_counts_leaf_and_binary_carries() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_100_scaling.yaml")
    leaf_only = _shared_hierarchy_work(config, 7)
    three_carries = _shared_hierarchy_work(config, 8)
    assert leaf_only["created_node_count"] == 1
    assert leaf_only["shared_forward_example_passes"] == 20 * 256
    assert three_carries["created_node_count"] == 4
    assert three_carries["shared_forward_example_passes"] == 20 * 256 * (1 + 2 + 4 + 8)
    assert three_carries["shared_backward_example_passes"] == three_carries[
        "shared_forward_example_passes"
    ]
    capacity_two_merge = _shared_hierarchy_work(config, 3, 2)
    assert capacity_two_merge["created_node_count"] == 2
    assert capacity_two_merge["shared_forward_example_passes"] == 20 * 256 * (1 + 2)


def test_two_node_policy_partitions_stream_and_preserves_surviving_slots() -> None:
    topology = _empty_dense_logt_state(2, 2)
    created_nodes = 0
    previous_slots = {}
    for block in range(100):
        topology, leaf, merges = _insert_dense_block(
            topology,
            (2 * block, 2 * block + 1),
        )
        created_nodes += 1 + len(merges)
        slots = {
            node.node_id: (node.level, rank)
            for level, nodes in topology.active_by_level.items()
            for rank, node in enumerate(nodes)
        }
        assert all(
            slots[node_id] == slot
            for node_id, slot in previous_slots.items()
            if node_id in slots
        )
        previous_slots = slots
    assert len(topology.active_nodes) == 9
    assert created_nodes == 191
    assert all(len(nodes) <= 2 for nodes in topology.active_by_level.values())


def test_two_node_features_append_secondary_slots_and_copy_initial_weights(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    base = initialize_dense_state((8, 6, 4), 0.0, 71)
    topology = _empty_dense_logt_state(2, 2)
    for block in range(2):
        topology, _leaf, _merges = _insert_dense_block(
            topology,
            (2 * block, 2 * block + 1),
        )
    deltas = pmap({node.node_id: zero_dense_delta(base) for node in topology.active_nodes})
    frontier = DenseFrontier(
        2,
        topology.active_nodes,
        deltas,
        pmap({node.node_id: node.node_id for node in topology.active_nodes}),
        2,
    )
    observations = build_dense_observations(
        frontier,
        _benchmark().step(1).observer,
        base,
        2,
        0.1,
        torch.device("cpu"),
        8,
    ).integrator
    slots = observations.features.reshape(2, 4, -1)
    assert observations.active_mask.tolist() == [True, False, True, False]
    assert torch.equal(slots[:, 1], torch.zeros_like(slots[:, 1]))
    assert torch.equal(slots[:, 3], torch.zeros_like(slots[:, 3]))

    slot_dim = base.embedding_dim + 11
    one, _ = _timed_state_creation(
        "integrator_uniform_replay",
        2 * slot_dim,
        slot_dim,
        config,
        0,
        1,
        torch.device("cpu"),
    )
    two, _ = _timed_state_creation(
        "integrator_uniform_replay",
        4 * slot_dim,
        slot_dim,
        config,
        0,
        2,
        torch.device("cpu"),
    )
    assert torch.equal(
        two.integrator.input_layer.weight[:, : one.integrator.input_dim],
        one.integrator.input_layer.weight,
    )
    assert torch.equal(
        two.integrator.input_layer.weight[:, one.integrator.input_dim :],
        torch.zeros_like(two.integrator.input_layer.weight[:, one.integrator.input_dim :]),
    )


def test_zero_delta_has_exact_parity_and_every_affine_tensor_matters() -> None:
    base = initialize_dense_state((8, 6, 4), 0.0, 3)
    images = torch.rand((5, 784), generator=torch.Generator().manual_seed(5))
    hidden, logits = dense_hidden_logits(images, base, zero_dense_delta(base))
    model = DenseMnistMLP((8, 6, 4), 0.0)
    from apm.continual.dense_mlp_adapter import load_dense_state

    load_dense_state(model, base)
    model.eval()
    expected_hidden, expected_logits = model.hidden_logits(images)
    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(logits, expected_logits)
    for tensor_index in range(len(base.tensors)):
        tensors = [torch.zeros_like(tensor) for tensor in base.tensors]
        tensors[tensor_index].fill_(0.25)
        changed_hidden, changed_logits = dense_hidden_logits(images, base, DenseMlpState(tuple(tensors)))
        assert not (
            torch.equal(changed_hidden, hidden) and torch.equal(changed_logits, logits)
        )


def test_dense_fit_is_deterministic_with_dropout_and_serializable(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(11)
    examples = DenseExamples(
        torch.rand((12, 1, 28, 28), generator=generator),
        torch.arange(12, dtype=torch.int64) % 10,
        (torch.arange(784),),
    )
    initial = initialize_dense_state((8, 6, 4), 0.2, 13)
    arguments = (examples, initial, DenseOptimizerConfig(0.001, 0.0001, 4, 1.0), 19, torch.device("cpu"))
    first = fit_dense_model(*arguments, fixed_epochs=2, dropout=0.2)
    second = fit_dense_model(*arguments, fixed_epochs=2, dropout=0.2)
    assert all(
        not torch.equal(before, after)
        for before, after in zip(initial.tensors, first.state.tensors, strict=True)
    )
    assert all(torch.equal(left, right) for left, right in zip(first.state.tensors, second.state.tensors, strict=True))
    path = tmp_path / "state.pt"
    atomic_torch_save(path, {"parameters": first.state.tensors})
    restored = DenseMlpState(tuple(torch.load(path, weights_only=True)["parameters"]))
    assert all(torch.equal(left, right) for left, right in zip(first.state.tensors, restored.tensors, strict=True))


def test_stratified_split_is_exact_disjoint_and_deterministic() -> None:
    labels = torch.arange(100, dtype=torch.int64) % 10
    first = stratified_source_split(labels, 20, 23)
    second = stratified_source_split(labels, 20, 23)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    training, validation = first
    assert len(training) == 80 and len(validation) == 20
    assert not set(training.tolist()) & set(validation.tolist())
    assert torch.equal(torch.bincount(labels[validation], minlength=10), torch.full((10,), 2))


def test_width_selection_uses_identity_gate_and_pooled_gap(tmp_path: Path) -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_mnist/primary.yaml")
    rows = []
    pooled = {config.calibration.candidate_widths[0]: 0.70, config.calibration.candidate_widths[1]: 0.799, config.calibration.candidate_widths[2]: 0.80}
    for widths in config.calibration.candidate_widths:
        for seed in config.calibration.seeds:
            rows.extend((
                {"hidden_widths": list(widths), "seed": seed, "fit": "identity", "validation_accuracy": 0.991},
                {"hidden_widths": list(widths), "seed": seed, "fit": "pooled", "validation_accuracy": pooled[widths]},
            ))
    assert select_calibrated_width(tuple(rows), config) == config.calibration.candidate_widths[1]
    for row in rows:
        if tuple(row["hidden_widths"]) == config.calibration.candidate_widths[1] and row["fit"] == "identity" and row["seed"] == 0:
            row["validation_accuracy"] = 0.98
    assert select_calibrated_width(tuple(rows), config) == config.calibration.candidate_widths[2]


def test_ungated_successor_always_selects_the_smallest_calibrated_width() -> None:
    config = load_config("configs/vamp_logt_mlp_permuted_mnist_ungated/primary.yaml")
    rows = []
    for widths in config.calibration.candidate_widths:
        for seed in config.calibration.seeds:
            rows.extend((
                {
                    "hidden_widths": list(widths),
                    "seed": seed,
                    "fit": "identity",
                    "validation_accuracy": 0.01,
                },
                {
                    "hidden_widths": list(widths),
                    "seed": seed,
                    "fit": "pooled",
                    "validation_accuracy": 0.01,
                },
            ))
    assert config.calibration.selection_policy == "smallest_candidate"
    assert select_calibrated_width(tuple(rows), config) == (1024, 1024, 512)


def test_condition_names_are_literal_and_plot_identities_are_distinct() -> None:
    assert CONDITION_PROTOCOL["integrator_uniform_replay"][0] == "Integrator — uniform-history replay"
    assert CONDITION_PROTOCOL["frozen_base_mlp"][0] == "Frozen calibrated base MLP"
    assert CONDITION_PROTOCOL["converged_cumulative_mlp"][0] == "Converged cumulative MLP"
    assert (
        CONDITION_PROTOCOL["converged_base_only_integrator"][0]
        == "Converged integrator over the frozen base MLP"
    )
    integrator_names = (
        "integrator_current_only",
        "integrator_uniform_replay",
        "integrator_range_replay",
        "integrator_base_uniform_replay",
        "mean_ensemble",
        "fresh_cumulative_four_epoch_integrator",
        "pooled_single_mlp_reference",
        "converged_full_replay_integrator",
    )
    identities = tuple(PLOT_STYLES[name] for name in integrator_names)
    assert len({identity[0] for identity in identities}) == len(identities)
    assert len(set(identities)) == len(identities)


def test_analysis_amendment_requires_an_ordered_configured_seed_subset(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    run_root = config.artifact_root / "runs" / config.config_hash
    publish_immutable_json(
        run_root / "analysis_amendment.json",
        {
            "config_hash": config.config_hash,
            "included_seeds": [0],
            "originally_declared_seeds": [0],
            "schema_version": "vamp-logt-dense-analysis-amendment-v1",
            "status": "active",
        },
    )
    assert load_analysis_seeds(run_root, config) == (0,)


def test_two_step_hierarchy_online_ceiling_and_resume_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(tmp_path)
    run_root = config.artifact_root / "runs" / config.config_hash
    base = _publish_base(config, run_root)
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "config": {"benchmark": {"macro_steps": 2}},
            "config_hash": config.config_hash,
            "schema_version": "test-protocol-v1",
        },
    )
    benchmark = _benchmark()
    for module in (
        "apm.experiments.vamp_logt_mlp_permuted_hierarchy",
        "apm.experiments.vamp_logt_mlp_permuted_online",
        "apm.experiments.vamp_logt_mlp_permuted_ceiling",
        "apm.experiments.vamp_logt_mlp_permuted_scaling",
    ):
        monkeypatch.setattr(f"{module}.build_benchmark", lambda _config, _seed: benchmark)
    hierarchy = build_hierarchy_tape(config, run_root, torch.device("cpu"))
    assert hierarchy[0]["created_node_count"] == 3
    assert len(tuple((run_root / "hierarchy" / "seed-0" / "nodes").glob("*/delta.pt"))) == 3
    frontier = load_frontier(config, run_root, 0, 1)
    observation = build_dense_observations(
        frontier,
        benchmark.step(1).observer,
        base,
        2,
        0.1,
        torch.device("cpu"),
        8,
    )
    slots = observation.integrator.features.reshape(2, 2, -1)
    assert slots.shape == (2, 2, 15)
    assert torch.equal(slots[:, 1], torch.zeros_like(slots[:, 1]))
    assert torch.equal(
        observation.integrator.baseline_log_probabilities,
        observation.integrator.node_log_probabilities[:, 0],
    )
    scaling = _run_scaling(config, run_root, torch.device("cpu"))
    scaling_ledger = run_root / "scaling" / "seed-0" / "metrics.jsonl"
    scaling_before = scaling_ledger.read_bytes()
    assert scaling["status"] == "complete"
    assert scaling["full_replay_checkpoints"] == [2]
    assert _run_scaling(config, run_root, torch.device("cpu")) == scaling
    assert scaling_ledger.read_bytes() == scaling_before
    capacity_two_root = run_root / "policies" / "two_nodes_per_level"
    build_hierarchy_tape(
        config,
        run_root,
        torch.device("cpu"),
        max_nodes_per_level=2,
        hierarchy_root=capacity_two_root / "hierarchy",
    )
    capacity_two = _run_scaling_seed(
        config,
        run_root,
        2,
        0,
        torch.device("cpu"),
    )
    capacity_two_ledger = capacity_two_root / "scaling" / "seed-0" / "metrics.jsonl"
    capacity_two_before = capacity_two_ledger.read_bytes()
    assert capacity_two["status"] == "complete"
    assert capacity_two["max_nodes_per_level"] == 2
    assert _run_scaling_seed(
        config,
        run_root,
        2,
        0,
        torch.device("cpu"),
    ) == capacity_two
    assert capacity_two_ledger.read_bytes() == capacity_two_before
    online = run_online_seed(config, run_root, 0, base, torch.device("cpu"))
    ledger_before = (online.directory / "metrics.jsonl").read_bytes()
    restored = run_online_seed(config, run_root, 0, base, torch.device("cpu"))
    assert restored.summary == online.summary
    assert (online.directory / "metrics.jsonl").read_bytes() == ledger_before
    accounting = [
        row
        for row in __import__("apm.experiments.vamp_logt_router_reporting", fromlist=["_load_jsonl"])._load_jsonl(online.directory / "metrics.jsonl")
        if row.get("row_type") == "accounting"
    ]
    assert accounting[-1]["replay_optimizer_updates_per_step"] > accounting[-1]["current_only_optimizer_updates_per_step"]
    ceiling = run_ceiling(config, run_root, torch.device("cpu"))
    ceiling_ledger = run_root / "ceiling" / "seed-0" / "metrics.jsonl"
    ceiling_before = ceiling_ledger.read_bytes()
    assert ceiling[0]["final_macro_step"] == 2
    assert all(ceiling[0]["acceptance"].values())
    assert run_ceiling(config, run_root, torch.device("cpu")) == ceiling
    assert ceiling_ledger.read_bytes() == ceiling_before
    baselines = run_baseline_extension(config, run_root, torch.device("cpu"))
    baseline_seed = baselines["seed_summaries"][0]
    baseline_ledger = run_root / "baselines" / "seed-0" / "metrics.jsonl"
    baseline_before = baseline_ledger.read_bytes()
    assert baseline_seed["evaluation_checkpoints"] == [2]
    assert all(baseline_seed["acceptance"].values())
    assert run_baseline_extension(config, run_root, torch.device("cpu")) == baselines
    assert baseline_ledger.read_bytes() == baseline_before
    aggregate = write_results(run_root, config)
    assert aggregate["status"] == "complete"
    assert aggregate["single_seed_baseline_extension"]["status"] == "complete"
    assert {
        row["domain_cells"]
        for row in aggregate["single_seed_baseline_extension"][
            "checkpoint_metrics"
        ]["2"].values()
    } == {2}
    assert (run_root / "RESULTS.md").is_file()
    assert "Converged full-replay integrator ceiling" in (run_root / "RESULTS.md").read_text()
    assert "Seed-0 cumulative baseline extension" in (run_root / "RESULTS.md").read_text()
    assert (run_root / "plots" / "01_integrator_accuracy.png").is_file()
    assert (run_root / "plots" / "04_single_seed_cumulative_baselines.png").is_file()
    publish_immutable_json(
        run_root / "analysis_amendment.json",
        {
            "config_hash": config.config_hash,
            "decision_stage": {
                "ceiling_completed_seed_count": 0,
                "ceiling_in_progress_macro_step": 1,
                "ceiling_in_progress_seed": 0,
                "online_completed_seed_count": 1,
            },
            "excluded_completed_online_seeds": [],
            "included_seeds": [0],
            "originally_declared_seeds": [0],
            "reason": "test amendment",
            "schema_version": "vamp-logt-dense-analysis-amendment-v1",
            "status": "active",
        },
    )
    analysis_root = write_amended_results(run_root, config)
    assert (analysis_root / "summary.json").is_file()
    amended_summary = load_canonical_json(analysis_root / "summary.json")
    assert amended_summary["analysis_seeds"] == [0]
    assert amended_summary["criteria"]["ceiling_every_step_cells"] == 2
    assert amended_summary["single_seed_baseline_extension"]["run_seed"] == 0
    assert "Analysis-set amendment" in (analysis_root / "RESULTS.md").read_text()
    assert "Seed-0 cumulative baseline extension" in (analysis_root / "RESULTS.md").read_text()


def test_new_dense_modules_do_not_define_or_load_a_convolutional_base() -> None:
    root = Path("src/apm")
    sources = tuple(root.glob("continual/dense_mlp_adapter.py")) + tuple(
        root.glob("experiments/vamp_logt_mlp_permuted_*.py")
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "AddressCNN" not in text
    assert "nn.Conv2d" not in text


@pytest.mark.benchmark
def test_cuda_dense_structural_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible in this process")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    config = _tiny_config(tmp_path)
    run_root = config.artifact_root / "runs" / config.config_hash
    base = _publish_base(config, run_root)
    benchmark = _benchmark()
    for module in (
        "apm.experiments.vamp_logt_mlp_permuted_hierarchy",
        "apm.experiments.vamp_logt_mlp_permuted_online",
        "apm.experiments.vamp_logt_mlp_permuted_scaling",
    ):
        monkeypatch.setattr(f"{module}.build_benchmark", lambda _config, _seed: benchmark)
    build_hierarchy_tape(config, run_root, device)
    result = run_online_seed(config, run_root, 0, base, device)
    assert result.summary["final_macro_step"] == 2
    assert all(result.summary["acceptance"].values())
    capacity_two_root = run_root / "policies" / "two_nodes_per_level"
    build_hierarchy_tape(
        config,
        run_root,
        device,
        max_nodes_per_level=2,
        hierarchy_root=capacity_two_root / "hierarchy",
    )
    scaling = _run_scaling_seed(config, run_root, 2, 0, device)
    assert scaling["status"] == "complete"
    assert scaling["max_nodes_per_level"] == 2
    production_state = initialize_dense_state((2048, 2048, 1024), 0.2, 101)
    examples = DenseExamples(
        torch.rand((4, 1, 28, 28), generator=torch.Generator().manual_seed(103)),
        torch.arange(4, dtype=torch.int64),
        (torch.arange(784),),
    )
    fit = fit_dense_model(
        examples,
        production_state,
        DenseOptimizerConfig(0.001, 0.0001, 256, 1.0),
        107,
        device,
        fixed_epochs=1,
        dropout=0.2,
    )
    assert fit.state.parameter_count == 7_912_458
    capacity_config = load_config("configs/vamp_logt_mlp_permuted_100_capacity.yaml")
    large_base = initialize_dense_state((2272, 2272, 1136), 0.2, 109)
    large_fit = fit_dense_model(
        examples,
        large_base,
        capacity_config.calibration.optimizer,
        113,
        device,
        fixed_epochs=1,
        dropout=0.2,
    )
    assert large_fit.state.parameter_count == 9_541_274
    slot_dim = 1136 + 11
    input_dim = 7 * slot_dim
    large_integrator = create_condition_state(
        "capacity-cuda-smoke",
        input_dim,
        slot_dim,
        capacity_config.integrator,
        127,
        device,
    )
    assert sum(
        parameter.numel() for parameter in large_integrator.integrator.parameters()
    ) == 17_650_160
    active_mask = torch.tensor((True, False, False, False, False, False, False))
    observations = IntegratorObservations(
        torch.zeros((2, input_dim)),
        torch.zeros((2, 7, 10)),
        active_mask,
        torch.zeros((2, 10)),
    )
    work = train_condition(
        large_integrator,
        IntegratorSupervision(observations, torch.tensor((0, 1))),
        None,
        1,
        capacity_config.integrator,
        127,
        1,
        device,
    )
    assert work.training_forward_example_passes == 2
    assert work.training_backward_example_passes == 2
