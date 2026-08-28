from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import jax
import numpy as np
import pytest
from pyrsistent import pmap

from apm.continual.logt_evidence_bank import TemporalNode, empty_logt_state, insert_block
from apm.continual.artifacts import canonical_json_bytes
from apm.experiments.vamp_logt_pc_config import load_config
from apm.experiments.vamp_logt_pc_data import (
    CONDITIONS,
    PcNodeHoldout,
    PcRawTable,
    build_condition_stream,
    build_node_holdout,
    condition_block_contexts,
)
from apm.experiments.vamp_logt_pc_state import (
    ActivePcBank,
    PcWorkCounters,
    load_bank_checkpoint,
    require_pc_work_bound,
    retire_inactive_models,
    save_bank_checkpoint,
)
from apm.experiments.vamp_logt_pc_training import SCORE_NAMES
from apm.experiments.vamp_logt_pc_training import GnNodeReplicaEvaluation
from apm.experiments.vamp_logt_pc_workflow import run_analytic_phase
from apm.experiments.vamp_logt_pc_gn_workflow import (
    _gn_static_metrics,
    map_source_tree_sha256,
    run_gn_analytic_phase,
)
from apm.models.fabricpc_density_backend import FabricPcDensityBackend, PcGaussNewtonScores


CONFIG = Path("configs/vamp_logt_pc_mnist/minimal.yaml")
GN_CONFIG = Path("configs/vamp_logt_pc_mnist/gauss_newton.yaml")
GN_V2_CONFIG = Path("configs/vamp_logt_pc_mnist/gauss_newton_v2.yaml")


def test_minimal_config_is_strict_and_frozen(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    assert config.model.latent_dim == 32
    assert config.model.hidden_dim == 128
    assert config.protocol_revision == "generative-pc-map-v1"
    assert config.evidence.estimator == "map"
    assert SCORE_NAMES == ("map",)
    assert config.training.infer_steps == 80
    assert config.training.score_batch_size == 4
    assert config.stream.block_size == 250
    assert config.stream.static_blocks == 31
    assert config.stream.model_seeds == (0, 1, 2)

    text = CONFIG.read_text(encoding="utf-8")
    changed = tmp_path / "changed.yaml"
    changed.write_text(text.replace("  progress: true", "  progress: true\n  undeclared: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime keys"):
        load_config(changed)

    changed.write_text(text.replace("  infer_steps: 80", "  infer_steps: 40"), encoding="utf-8")
    with pytest.raises(ValueError, match="training schedule"):
        load_config(changed)

    changed.write_text(text.replace("  estimator: map", "  estimator: laplace"), encoding="utf-8")
    with pytest.raises(ValueError, match="only complete MAP"):
        load_config(changed)


def test_gauss_newton_config_and_map_source_are_strict() -> None:
    map_config = load_config(CONFIG)
    config = load_config(GN_CONFIG)
    assert map_config.config_hash == "c4643cd904ae9802c6a427868b954e6ff54b960a6c589231ccd9b3ddfb4e06a7"
    assert config.protocol_revision == "generative-pc-gn-v1"
    assert config.evidence.estimator == "generalized_gauss_newton"
    assert config.evidence.primary_score == "gn1"
    assert config.evidence.negative_direction_epsilons == (0.01, 0.05, 0.10)
    assert config.model_source is not None
    assert config.model_source_run_root is not None
    digest, records = map_source_tree_sha256(config.model_source_run_root)
    assert digest == config.model_source.required_tree_sha256
    assert len(records) == 106

    continuation = load_config(GN_V2_CONFIG)
    assert continuation.protocol_revision == "generative-pc-gn-v2"
    assert continuation.evidence.float64_agreement_role == "diagnostic_only"
    assert continuation.config_hash != config.config_hash


def test_gauss_newton_analytic_gate(tmp_path: Path) -> None:
    summary = run_gn_analytic_phase(load_config(GN_CONFIG), tmp_path)
    assert summary["passed"] is True
    assert summary["linear_gaussian_gn0_at_mode_maximum_error_nats"] < 1.0e-10
    assert summary["linear_gaussian_gn1_away_from_mode_maximum_error_nats"] < 1.0e-10
    assert summary["nonlinear_minimum_hessian_eigenvalue"] < 0.0
    assert summary["nonlinear_minimum_gauss_newton_eigenvalue"] > 0.0


def test_gauss_newton_static_metrics_condition_raw_h_routes_on_coverage() -> None:
    config = load_config(GN_CONFIG)
    nodes = tuple(
        TemporalNode(
            f"{index + 1:x}" * 64,
            level,
            0,
            2**level - 1,
            (index,),
            () if level == 0 else ("a" * 64, "b" * 64),
        )
        for index, level in enumerate((4, 3, 2, 1, 0))
    )
    stream = _table(1).select(np.arange(5, dtype=np.int64))
    table = PcRawTable(
        np.zeros((10, 784), dtype=np.uint8),
        np.zeros(10, dtype=np.int64),
        np.zeros(10, dtype=np.int64),
        np.arange(100, 110, dtype=np.int64),
    )
    holdout = PcNodeHoldout(table, tuple(node.node_id for node in nodes for _ in range(2)))
    source_indices = np.repeat(np.arange(5), 2)

    def evaluation(node_index: int, focused: bool = False) -> GnNodeReplicaEvaluation:
        rows = 10
        map_score = np.full(rows, float(node_index), dtype=np.float32)
        if not focused:
            map_score = -np.abs(node_index - source_indices).astype(np.float32)
        elif node_index == 4:
            map_score.fill(1.0)
        elif node_index == 0:
            map_score.fill(0.0)
        hessian_score = map_score.copy()
        hessian_ok = np.ones(rows, dtype=bool)
        if node_index == 2:
            hessian_score[0] = np.nan
            hessian_ok[0] = False
        evidence = PcGaussNewtonScores(
            map_score,
            map_score,
            hessian_score,
            map_score,
            map_score + 0.5,
            np.ones(rows, dtype=np.float32),
            np.ones(rows, dtype=np.float32),
            np.where(hessian_ok, 0.5, -0.5).astype(np.float32),
            np.ones(rows, dtype=np.float32),
            hessian_ok,
            np.ones(rows, dtype=bool),
            np.full((rows, 3), np.nan, dtype=np.float32),
            np.full((rows, 3), np.nan, dtype=np.float32),
        )
        logits = np.zeros((rows, 10), dtype=np.float32)
        return GnNodeReplicaEvaluation(
            evidence,
            logits,
            np.zeros(rows, dtype=np.float32),
            np.zeros(rows, dtype=np.int64),
        )

    evaluations = {
        seed: {node.node_id: evaluation(index) for index, node in enumerate(nodes)}
        for seed in config.stream.model_seeds
    }
    focused_evaluations = {
        seed: {
            nodes[0].node_id: evaluation(0, focused=True),
            nodes[-1].node_id: evaluation(4, focused=True),
        }
        for seed in config.stream.model_seeds
    }
    summary, raw = _gn_static_metrics(
        config,
        "novel_leaf",
        0,
        stream,
        nodes,
        holdout,
        table,
        evaluations,
        focused_evaluations,
    )

    assert summary["gauss_newton_cholesky_successes"] == summary["total_scored_states"]
    assert summary["hessian_route_diagnostics"][0]["coverage_fraction"] == 0.9
    assert summary["passed_by_score"]["gn1"] is True
    assert raw["general_0_h_laplace_routes"][0] == -1
    canonical_json_bytes(summary)


def test_controlled_schedules_have_exact_context_counts_and_disjoint_rows() -> None:
    train = _table(4_500)
    expected = {
        "novel_leaf": (4_000, 2_000, 1_000, 500, 250),
        "recurrent_leaf_1_8": (3_500, 2_000, 1_000, 500, 750),
        "identical_regime": (0, 2_000, 1_000, 500, 4_250),
    }
    for condition in CONDITIONS:
        stream = build_condition_stream(train, condition, 3)
        assert tuple(np.bincount(stream.context_ids, minlength=5)) == expected[condition]
        assert len(np.unique(stream.source_rows)) == 31 * 250
        assert tuple(stream.context_ids.reshape(31, 250)[:, 0]) == condition_block_contexts(condition)


def test_map_analytic_gate_does_not_construct_hessians(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG)
    monkeypatch.setattr(
        jax,
        "hessian",
        lambda *_args, **_kwargs: pytest.fail("MAP analytic validation constructed a Hessian"),
    )

    summary = run_analytic_phase(config, tmp_path)

    assert summary["passed"] is True
    assert summary["estimator"] == "map"
    assert "floor_maximum_absolute_errors_nats" not in summary


def test_static_topology_and_holdout_follow_node_mixtures() -> None:
    train = _table(4_500)
    test = _table(1_000)
    stream = build_condition_stream(train, "recurrent_leaf_1_8", 0)
    topology = empty_logt_state(250)
    for block in range(31):
        topology, _leaf, _merges = insert_block(
            topology,
            tuple(range(block * 250, (block + 1) * 250)),
        )
    assert tuple(node.level for node in topology.active_nodes) == (4, 3, 2, 1, 0)
    holdout = build_node_holdout(topology.active_nodes, stream, test, 128, 9)
    assert len(holdout.table.labels) == 640
    assert len(set(holdout.source_node_ids)) == 5
    history_contexts = np.bincount(holdout.table.context_ids[:128], minlength=5)
    np.testing.assert_array_equal(history_contexts, np.asarray([112, 0, 0, 0, 16]))


def test_bank_checkpoint_precedes_scoped_child_retirement(tmp_path: Path) -> None:
    topology = empty_logt_state(2)
    topology, first, _ = insert_block(topology, (0, 1))
    topology, second, merges = insert_block(topology, (2, 3))
    parent = merges[0].parent
    counters = PcWorkCounters(active_pc_models=1)
    bank = ActivePcBank(topology, pmap({parent.node_id: (0,)}), counters)
    models = tmp_path / "models"
    for node in (first, second, parent):
        directory = models / node.node_id / "replica-0"
        directory.mkdir(parents=True)
        (directory / "complete").write_text("ok", encoding="utf-8")

    checkpoint = tmp_path / "bank.json"
    save_bank_checkpoint(checkpoint, bank)
    retired = retire_inactive_models(models, {parent.node_id})
    loaded = load_bank_checkpoint(checkpoint)

    assert set(retired) == {first.node_id, second.node_id}
    assert loaded == bank
    assert (models / parent.node_id).is_dir()
    assert not (models / first.node_id).exists()
    assert not (models / second.node_id).exists()


def test_pc_work_bound_counts_density_classifier_and_fixed_inference() -> None:
    counters = PcWorkCounters()
    counters = counters.with_fit(1_000, 2_500, 50, merge=False, infer_steps=4)
    counters = counters.with_scoring(20, 3, 4, hessians=False, settle_passes=1)
    assert counters.pc_leaf_example_presentations == 1_000
    assert counters.classifier_example_presentations == 2_500
    assert counters.pc_inference_state_updates == (1_000 + 50) * 4 + 20 * 3 * 4
    assert counters.pc_laplace_hessian_evals == 0
    assert counters.pc_importance_audit_samples == 0
    counters = counters.with_gauss_newton_scoring(
        5,
        2,
        4,
        negative_hessian_states=3,
        direction_epsilon_count=3,
    )
    assert counters.pc_gauss_newton_matrix_evals == 10
    assert counters.pc_gauss_newton_cholesky_solves == 10
    assert counters.pc_exact_hessian_evals == 10
    assert counters.pc_negative_direction_probes == 18
    require_pc_work_bound(counters, 4, 250, 1, 3, 1)
    with pytest.raises(RuntimeError, match="density work"):
        require_pc_work_bound(counters, 1, 250, 1, 3, 1)


def test_raw_table_and_evidence_api_have_no_feature_context_or_node_size_channel() -> None:
    assert {field.name for field in fields(PcRawTable)} == {
        "raw_images",
        "labels",
        "context_ids",
        "source_rows",
    }
    assert tuple(FabricPcDensityBackend.score_images.__annotations__) == (
        "params",
        "images",
        "return",
    )


def _table(per_context: int) -> PcRawTable:
    rows = 5 * per_context
    contexts = np.repeat(np.arange(5, dtype=np.int64), per_context)
    raw = np.zeros((rows, 784), dtype=np.uint8)
    raw[:, 0] = contexts
    return PcRawTable(
        raw,
        np.arange(rows, dtype=np.int64) % 10,
        contexts,
        np.arange(rows, dtype=np.int64),
    )
