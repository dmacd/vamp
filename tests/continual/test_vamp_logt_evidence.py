from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch
import yaml
from pyrsistent import pmap

from apm.continual.logt_evidence_bank import empty_logt_state, insert_block
from apm.continual.nce_tre_evidence import ConditionalEvidenceCNN
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    top_two_base_state,
    zero_top_two_adapter,
)
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_evidence_config import load_config
from apm.experiments.vamp_logt_evidence_data import (
    AuthenticatedBaseline,
    NodeHoldout,
    RawFeatureTable,
    stream_training_table,
)
from apm.experiments.vamp_logt_evidence_reporting import CONDITION_DEFINITIONS
from apm.experiments.vamp_logt_evidence_training import (
    evaluate_routing,
    score_evidence_bank,
    train_node_evidence,
)
from apm.experiments.vamp_logt_evidence_workflow import _build_bank_to_blocks


def _two_node_topology():
    state = empty_logt_state(2)
    for block in range(3):
        state, _leaf, _merges = insert_block(
            state, tuple(range(2 * block, 2 * block + 2))
        )
    return state


def _constant_model(value: float) -> ConditionalEvidenceCNN:
    model = ConditionalEvidenceCNN(1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.scalar.bias.fill_(value)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_primary_config_is_strict_and_keeps_the_fixed_de_risking_protocol(tmp_path: Path) -> None:
    source = Path("configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml")
    config = load_config(source)
    assert config.stream.block_size == 500
    assert config.stream.total_blocks == 100
    assert config.stream.static_snapshot_blocks == 63
    assert config.evidence.candidate_tre_bridges == (2, 4, 8, 16)
    assert config.evidence.reference == "frozen_base_training_images_uint8"
    assert config.static.stream_seeds == (0, 1, 2)
    assert config.calibration.component_probabilities == (0.3, 0.7)
    assert config.calibration.training_steps == 2_500
    record = yaml.safe_load(source.read_text(encoding="utf-8"))
    record["evidence"]["hidden_scientific_choice"] = 7
    invalid = tmp_path / "configs" / "vamp_logt_evidence_mnist" / "nce_tre.yaml"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(yaml.safe_dump(record), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence keys"):
        load_config(invalid)


def test_evidence_scores_route_without_labels_and_match_the_oracle_adapter() -> None:
    topology = _two_node_topology()
    earlier, later = topology.active_nodes
    raw = torch.zeros((4, 1, 28, 28), dtype=torch.uint8)
    table = RawFeatureTable(
        raw,
        torch.ones((4, 1), dtype=torch.float32),
        torch.ones(4, dtype=torch.int64),
        torch.zeros(4, dtype=torch.int64),
        torch.arange(4, dtype=torch.int64),
    )
    base = top_two_base_state(
        torch.zeros((1, 1)),
        torch.ones(1),
        torch.zeros((2, 1)),
        torch.zeros(2),
    )
    zero = zero_top_two_adapter(base)
    later_adapter = TopTwoAdapterState(
        zero.embedding_weight,
        zero.embedding_bias,
        zero.classifier_weight,
        torch.tensor([-1.0, 1.0]),
    )
    adapters = pmap({earlier.node_id: zero, later.node_id: later_adapter})
    scores = score_evidence_bank(
        topology.active_nodes,
        {earlier.node_id: _constant_model(-1.0), later.node_id: _constant_model(1.0)},
        table.raw_images,
        torch.device("cpu"),
        4,
    )
    holdout = NodeHoldout(
        table,
        (later.node_id,) * 4,
        ((0, 1, 0, 0, 0),) * 4,
    )
    result = evaluate_routing(
        topology.active_nodes,
        adapters,
        scores,
        table,
        base,
        torch.device("cpu"),
        4,
        holdout,
        {
            earlier.node_id: (1, 0, 0, 0, 0),
            later.node_id: (0, 1, 0, 0, 0),
        },
    )
    assert result.routed_accuracy == 1.0
    assert result.oracle_accuracy == 1.0
    assert result.route_oracle_agreement == 1.0
    assert result.exact_source_accuracy == 1.0
    assert result.equivalent_source_accuracy == 1.0
    assert result.routing_regret_nats == pytest.approx(0.0)


def test_complete_condition_definitions_are_full_sentences() -> None:
    assert set(CONDITION_DEFINITIONS) == {
        "direct_nce",
        "tre",
        "oracle_node",
        "vamp_af",
        "global_replay",
        "joint_iid",
        "oracle_context",
        "frozen_base",
        "oracle_leaf",
    }
    assert all(text[0].isupper() and text.endswith(".") for text in CONDITION_DEFINITIONS.values())


def test_evidence_training_api_cannot_accept_labels_contexts_or_frozen_features() -> None:
    assert tuple(inspect.signature(train_node_evidence).parameters) == (
        "raw_images",
        "reference_raw_images",
        "config",
        "bridges",
        "seed",
        "device",
        "show_progress",
    )
    assert tuple(inspect.signature(score_evidence_bank).parameters) == (
        "nodes",
        "models",
        "raw_images",
        "device",
        "batch_size",
    )


def test_stream_order_is_blocked_and_deterministic_without_resampling() -> None:
    rows_per_context = 3
    rows = 5 * rows_per_context
    table = RawFeatureTable(
        torch.arange(rows, dtype=torch.uint8)[:, None, None, None].expand(-1, 1, 28, 28).clone(),
        torch.arange(rows, dtype=torch.float32)[:, None],
        torch.arange(rows, dtype=torch.int64) % 2,
        torch.arange(5, dtype=torch.int64).repeat_interleave(rows_per_context),
        torch.arange(rows, dtype=torch.int64),
    )
    model = AddressCNN()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = top_two_base_state(
        model.embedding.weight,
        model.embedding.bias,
        model.classifier.weight,
        model.classifier.bias,
    )
    baseline = AuthenticatedBaseline(
        model, base, table, table, None, None, (), "0" * 64, {}
    )
    left = stream_training_table(baseline, 7)
    right = stream_training_table(baseline, 7)
    assert torch.equal(left.raw_images, right.raw_images)
    assert left.context_ids.tolist() == [
        context for context in range(5) for _row in range(rows_per_context)
    ]
    assert sorted(left.raw_images[:, 0, 0, 0].tolist()) == list(range(rows))


def test_two_block_model_bank_is_resumable_and_deletes_merged_children(tmp_path: Path) -> None:
    config = load_config("configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml")
    config = replace(
        config,
        adapter=replace(config.adapter, epochs=1, batch_size=128),
        evidence=replace(config.evidence, epochs=1, batch_size=128),
        runtime=replace(config.runtime, device="cpu", progress=False),
    )
    rows = 1_000
    stream = RawFeatureTable(
        torch.randint(0, 256, (rows, 1, 28, 28), dtype=torch.uint8),
        torch.randn(rows, 4),
        torch.arange(rows, dtype=torch.int64) % 2,
        torch.zeros(rows, dtype=torch.int64),
        torch.arange(rows, dtype=torch.int64),
    )
    model = AddressCNN()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = top_two_base_state(
        torch.randn(3, 4) * 0.01,
        torch.zeros(3),
        torch.randn(2, 3) * 0.01,
        torch.zeros(2),
    )
    reference = torch.randint(0, 256, (32, 1, 28, 28), dtype=torch.uint8)
    baseline = AuthenticatedBaseline(
        model,
        base,
        stream,
        stream,
        reference,
        "1" * 64,
        (),
        "0" * 64,
        {},
    )
    directory = tmp_path / "bank"
    bank = _build_bank_to_blocks(
        config,
        stream,
        baseline,
        0,
        directory,
        torch.device("cpu"),
        2,
        {"direct": 1},
        "test",
    )
    resumed = _build_bank_to_blocks(
        config,
        stream,
        baseline,
        0,
        directory,
        torch.device("cpu"),
        2,
        {"direct": 1},
        "test",
    )
    assert [node.level for node in bank.topology.active_nodes] == [1]
    assert bank.counters.evidence_train_example_updates == 1_000
    assert bank.counters.evidence_merge_example_updates == 1_000
    assert resumed.counters == bank.counters
    assert tuple(path.name for path in (directory / "nodes").iterdir()) == (
        bank.topology.active_nodes[0].node_id,
    )
