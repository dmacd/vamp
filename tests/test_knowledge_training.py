from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.continual.knowledge_training as training_module
import apm.continual.tinyworlds_pilot_run as pilot_module
from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.knowledge_training import (
    KnowledgeParentContext,
    KnowledgeValidationSuite,
    commit_selected_counterfactual_edge,
    plan_parent_counterfactuals,
    run_parent_counterfactuals,
    score_knowledge_parent_nodes,
    select_knowledge_parent_from_scores,
    validate_parent_counterfactual_resume,
)
from apm.continual.language_tasks import build_prefix_suffix_batches
from apm.continual.tinyworlds_pilot_run import (
    save_tinyworlds_transfer_chunk,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    init_candidate_lora_train_state,
)
from apm.lm.workflow import (
    CandidateTrainingCheckpoint,
    LmLossTrace,
)
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=24,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _query(query_index: int, correct_index: int) -> KnowledgeQuery:
    prefix = (1 + query_index, 2 + query_index, 3 + query_index)
    batches = tuple(
        build_prefix_suffix_batches(
            prefix + (10 + candidate_index, 14 + candidate_index),
            prefix_length=len(prefix),
            suffix_length=2,
        )
        for candidate_index in range(4)
    )
    return KnowledgeQuery(
        query_id=f"validation-query-{query_index}",
        task_id=TaskId("willow-extension"),
        family_id="willow",
        query_kind="direct",
        candidates=tuple(
            KnowledgeCandidate(answer, competence)
            for answer, (_, competence) in zip(
                ("amber", "blue", "coral", "dune"),
                batches,
            )
        ),
        router_batch=batches[0][0],
        correct_candidate_index=correct_index,
        proof_id=f"proof-{query_index}",
        support_ids=(f"fact-{query_index}",),
        required_edge_ids=(NodeId("willow-seed"),),
        cue_regime="cue_sufficient",
        visible_cue_ids=("cue-willow",),
        eligible_task_ids=(TaskId("willow-extension"),),
        novelty_regime="direct",
        reasoning_type="direct",
        reasoning_depth=0,
        prefix_length=len(prefix),
        mode="closed_book",
        oracle_node_ids=(NodeId("willow-seed"),),
    )


def _validation_suite() -> KnowledgeValidationSuite:
    return KnowledgeValidationSuite(
        suite_id="willow-extension-validation",
        split="validation",
        task_id=TaskId("willow-extension"),
        family_id="willow",
        queries=(_query(0, 1), _query(1, 2)),
    )


def _graph_fixture():
    model_config = _model_config()
    lora_config = LoraConfig(rank=1, alpha=1.0)
    edges = tuple(
        init_lora_edge(jax.random.PRNGKey(seed), model_config, lora_config)
        for seed in (1, 2)
    )
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("willow-seed"),
        NodeId("root"),
        TaskId("willow-seed"),
        1,
        edges[0],
    )
    graph = add_memory_node(
        graph,
        NodeId("sunny-seed"),
        NodeId("root"),
        TaskId("sunny-seed"),
        2,
        edges[1],
    )
    packed = pack_lora_memory(
        graph,
        model_config,
        lora_config,
        max_nodes=5,
        max_edges=4,
    )
    return model_config, lora_config, graph, packed


def _hard_scores() -> np.ndarray:
    scores = np.full((2, 4, 5), np.inf, dtype=np.float32)
    scores[:, :, :3] = 8.0
    scores[0, 1, :3] = (4.0, 1.0, 1.0)
    scores[1, 2, :3] = (4.0, 3.0, 3.0)
    scores[:, 0, 2] = 0.01
    return scores


def _context() -> KnowledgeParentContext:
    return KnowledgeParentContext(
        task_id=TaskId("willow-extension"),
        family_id="willow",
        true_parent_node_id=NodeId("willow-seed"),
        node_family_ids=(
            (NodeId("root"), None),
            (NodeId("willow-seed"), "willow"),
            (NodeId("sunny-seed"), "sunny"),
        ),
    )


def test_parent_search_uses_validation_correct_candidates_and_insertion_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _validation_suite()
    model_config, lora_config, graph, packed = _graph_fixture()
    hard_scores = _hard_scores()

    result = select_knowledge_parent_from_scores(
        suite,
        graph,
        packed,
        hard_scores,
    )

    np.testing.assert_array_equal(
        result.correct_candidate_nll_by_query_and_node,
        ((4.0, 1.0, 1.0), (4.0, 3.0, 3.0)),
    )
    assert result.mean_correct_candidate_nll == (4.0, 2.0, 2.0)
    assert result.selected_node_index == 1
    assert result.selected_node_id == NodeId("willow-seed")
    assert result.scoring_basis == "mean_validation_correct_candidate_nll"
    assert result.validation_query_ids == suite.query_ids
    assert not result.correct_candidate_nll_by_query_and_node.flags.writeable
    with pytest.raises(FrozenInstanceError):
        result.selected_node_index = 2  # type: ignore[misc]

    observed_queries: list[tuple[KnowledgeQuery, ...]] = []

    def fake_score(*args, **kwargs):
        del kwargs
        observed_queries.append(args[4])
        return hard_scores

    monkeypatch.setattr(training_module, "score_hard_node_candidates", fake_score)
    scored = score_knowledge_parent_nodes(
        suite,
        graph,
        packed,
        init_gpt_neo_params(jax.random.PRNGKey(3), model_config),
        model_config,
        lora_config,
    )
    assert observed_queries == [suite.queries]
    assert scored.selected_node_id == result.selected_node_id


def test_validation_boundary_rejects_test_split_and_mixed_task_queries() -> None:
    suite = _validation_suite()

    with pytest.raises(ValueError, match="validation data only"):
        replace(suite, split="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="suite task and family"):
        replace(
            suite,
            queries=(
                replace(suite.queries[0], task_id=TaskId("other-task")),
            ),
        )


def test_counterfactual_plan_covers_roles_and_marks_missing_other_family() -> None:
    suite = _validation_suite()
    _, _, graph, packed = _graph_fixture()
    search = select_knowledge_parent_from_scores(
        suite,
        graph,
        packed,
        _hard_scores(),
    )

    plan = plan_parent_counterfactuals(search, _context())

    assert tuple(target.role for target in plan.targets) == (
        "root",
        "true_parent",
        "selected_parent",
        "strongest_other_family",
    )
    assert tuple(target.parent_node_id for target in plan.targets) == (
        NodeId("root"),
        NodeId("willow-seed"),
        NodeId("willow-seed"),
        NodeId("sunny-seed"),
    )
    assert plan.available_parent_ids == (
        NodeId("root"),
        NodeId("willow-seed"),
        NodeId("sunny-seed"),
    )

    one_family_search = replace(
        search,
        node_ids=search.node_ids[:2],
        correct_candidate_nll_by_query_and_node=(
            search.correct_candidate_nll_by_query_and_node[:, :2]
        ),
        mean_correct_candidate_nll=search.mean_correct_candidate_nll[:2],
        selected_node_index=1,
        selected_node_id=NodeId("willow-seed"),
    )
    one_family_context = replace(
        _context(),
        node_family_ids=_context().node_family_ids[:2],
    )
    one_family_plan = plan_parent_counterfactuals(
        one_family_search,
        one_family_context,
    )
    assert not one_family_plan.targets[-1].available
    assert one_family_plan.targets[-1].parent_node_id is None


def test_counterfactual_training_resumes_one_shared_initialization_and_commits_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    suite = _validation_suite()
    model_config, lora_config, graph, packed = _graph_fixture()
    search = select_knowledge_parent_from_scores(
        suite,
        graph,
        packed,
        _hard_scores(),
    )
    plan = plan_parent_counterfactuals(search, _context())
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=4,
        batch_size=1,
        weight_decay=0.0,
    )
    initial_state = init_candidate_lora_train_state(
        init_lora_edge(jax.random.PRNGKey(10), model_config, lora_config),
        jax.random.PRNGKey(11),
        train_config,
    )
    train_batches = (
        TokenBatch(
            input_ids=np.asarray(((1, 2, 3, 4),), dtype=np.int32),
            attention_mask=np.ones((1, 4), dtype=np.bool_),
            target_ids=np.asarray(((2, 3, 4, 5),), dtype=np.int32),
            loss_mask=np.ones((1, 4), dtype=np.bool_),
        ),
    )
    initial_state_ids: list[int] = []
    parent_coefficients_seen: list[np.ndarray] = []
    validation_queries_seen: list[tuple[KnowledgeQuery, ...]] = []

    def fake_candidate_scores(*args, **kwargs):
        del kwargs
        validation_queries_seen.append(args[4])
        scores = np.full((len(args[4]), 4), 2.0, dtype=np.float32)
        for row, query in enumerate(args[4]):
            scores[row, query.correct_candidate_index] = 0.5
        return scores

    def fake_resumable(
        state,
        batches,
        base_params,
        config,
        memory,
        edge_config,
        parent_coefficients,
        candidate_index,
        training_config,
        *,
        stop_update=None,
        validation_function=None,
    ):
        del batches, base_params, config, memory, edge_config, candidate_index
        assert validation_function is not None
        start = int(state.step)
        target = training_config.steps if stop_update is None else stop_update
        if start == 0:
            initial_state_ids.append(id(state))
        parent_values = np.asarray(parent_coefficients, dtype=np.float32)
        parent_coefficients_seen.append(parent_values)
        marker = float(1 + np.dot(parent_values, np.arange(1, 5)))

        def state_at(update: int) -> LmTrainState:
            delta = np.float32(marker * 1e-4)
            trainable = state.trainable
            for _ in range(start, update):
                trainable = jax.tree_util.tree_map(
                    lambda value: value + delta,
                    trainable,
                )
            return LmTrainState(
                trainable=trainable,
                opt_state=state.opt_state,
                rng_key=state.rng_key,
                step=jnp.asarray(update, dtype=jnp.int32),
            )

        checkpoint_updates = sorted(
            {start, target}
            | {
                2**power
                for power in range(training_config.steps.bit_length())
                if start < 2**power <= target
            }
        )
        checkpoints = tuple(
            CandidateTrainingCheckpoint(
                update=update,
                state=state_at(update),
                training_loss=(None if update == start else 1.0 / update),
                validation_candidate_accuracy=validation_function(
                    state_at(update).trainable,
                    update,
                )[0],
                validation_correct_nll=validation_function(
                    state_at(update).trainable,
                    update,
                )[1],
            )
            for update in checkpoint_updates
        )
        final_state = state_at(target)
        trace = LmLossTrace(
            tuple(1.0 / update for update in range(start + 1, target + 1))
        )
        return final_state, trace, checkpoints

    monkeypatch.setattr(
        training_module,
        "score_edge_coefficient_candidates",
        fake_candidate_scores,
    )
    monkeypatch.setattr(
        training_module,
        "run_resumable_candidate_edge_updates",
        fake_resumable,
    )
    mutated_query = replace(
        suite.queries[0],
        candidates=(
            replace(suite.queries[0].candidates[0], answer_text="amber-mutated"),
            *suite.queries[0].candidates[1:],
        ),
    )
    mutated_suite = replace(
        suite,
        queries=(mutated_query, suite.queries[1]),
    )
    with pytest.raises(ValueError, match="must match parent selection"):
        run_parent_counterfactuals(
            plan,
            mutated_suite,
            initial_state,
            train_batches,
            init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
            model_config,
            packed,
            lora_config,
            train_config,
            stop_update=2,
        )

    partial_one = run_parent_counterfactuals(
        plan,
        suite,
        initial_state,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
        stop_update=1,
    )
    partial = run_parent_counterfactuals(
        plan,
        suite,
        initial_state,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
        stop_update=2,
    )

    assert len(partial.diagnostics.trials) == 3
    assert len(set(initial_state_ids)) == 1
    initial_checksums = {
        trial.initial_state_checksum for trial in partial.diagnostics.trials
    }
    assert len(initial_checksums) == 1
    assert partial.diagnostics.trials[1].roles == (
        "true_parent",
        "selected_parent",
    )
    assert all(
        tuple(checkpoint.update for checkpoint in trial.checkpoints) == (0, 1, 2)
        for trial in partial.diagnostics.trials
    )

    checkpoint_root = tmp_path / "checkpoint-root"
    save_tinyworlds_transfer_chunk(
        checkpoint_root / "stage-03-chunk-000001",
        3,
        partial_one,
    )
    partial_directory = checkpoint_root / "stage-03-chunk-000002"
    save_tinyworlds_transfer_chunk(partial_directory, 3, partial)
    stale_atomic_temporary = (
        checkpoint_root / ".stage-03-chunk-000004.tmp-999999"
    )
    stale_atomic_temporary.mkdir()
    partial_checksums = tuple(
        trial.final_state_checksum for trial in partial.diagnostics.trials
    )
    del partial
    restored = pilot_module._load_latest_transfer_chunk(
        checkpoint_root,
        3,
        plan,
        initial_state,
        train_config.steps,
    )
    assert restored is not None
    assert tuple(
        trial.final_state_checksum for trial in restored.diagnostics.trials
    ) == partial_checksums
    assert all(int(state.step) == 2 for state in restored.final_states)

    gapped_root = tmp_path / "gapped-checkpoint-root"
    save_tinyworlds_transfer_chunk(
        gapped_root / "stage-03-chunk-000002",
        3,
        restored,
    )
    with pytest.raises(ValueError, match="checkpoint prefix"):
        pilot_module._load_latest_transfer_chunk(
            gapped_root,
            3,
            plan,
            initial_state,
            train_config.steps,
        )

    unexpected_root = tmp_path / "unexpected-checkpoint-root"
    save_tinyworlds_transfer_chunk(
        unexpected_root / "stage-03-chunk-000002",
        3,
        restored,
    )
    (unexpected_root / ".stage-03-chunk-000004.tmp-not-a-pid").mkdir()
    with pytest.raises(ValueError, match="unexpected transfer temporary"):
        pilot_module._load_latest_transfer_chunk(
            unexpected_root,
            3,
            plan,
            initial_state,
            train_config.steps,
        )

    tampered_root = tmp_path / "tampered-checkpoint-root"
    save_tinyworlds_transfer_chunk(
        tampered_root / "stage-03-chunk-000001",
        3,
        partial_one,
    )
    save_tinyworlds_transfer_chunk(
        tampered_root / "stage-03-chunk-000002",
        3,
        restored,
    )
    tampered_chunk = tampered_root / "stage-03-chunk-000004"
    tampered_chunk.mkdir()
    (tampered_chunk / "metadata.json").write_text("not canonical\n")
    with pytest.raises(ValueError, match="missing or unlisted files"):
        pilot_module._load_latest_transfer_chunk(
            tampered_root,
            3,
            plan,
            initial_state,
            train_config.steps,
        )

    completed = run_parent_counterfactuals(
        plan,
        suite,
        restored,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
    )

    uninterrupted = run_parent_counterfactuals(
        plan,
        suite,
        initial_state,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
    )
    partial_three = run_parent_counterfactuals(
        plan,
        suite,
        initial_state,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
        stop_update=3,
    )
    completed_from_three = run_parent_counterfactuals(
        plan,
        suite,
        partial_three,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
    )
    assert all(
        tuple(checkpoint.update for checkpoint in trial.checkpoints)
        == (0, 1, 2, 4)
        for trial in completed_from_three.diagnostics.trials
    )
    assert uninterrupted.execution_sha256 == completed.execution_sha256
    assert uninterrupted.diagnostics == completed.diagnostics
    assert tuple(
        trial.final_state_checksum for trial in uninterrupted.diagnostics.trials
    ) == tuple(
        trial.final_state_checksum for trial in completed.diagnostics.trials
    )
    resumed_directory = tmp_path / "resumed-final"
    uninterrupted_directory = tmp_path / "uninterrupted-final"
    save_tinyworlds_transfer_chunk(resumed_directory, 3, completed)
    save_tinyworlds_transfer_chunk(
        uninterrupted_directory,
        3,
        uninterrupted,
    )
    loaded_final = pilot_module.load_tinyworlds_transfer_chunk(
        resumed_directory,
        3,
        plan,
        initial_state,
    )
    with pytest.raises(ValueError, match="initial state"):
        pilot_module.load_tinyworlds_transfer_chunk(
            resumed_directory,
            3,
            plan,
            replace(initial_state, rng_key=jax.random.PRNGKey(99)),
        )
    validate_parent_counterfactual_resume(
        plan,
        suite,
        loaded_final,
        train_batches,
        init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
        model_config,
        packed,
        lora_config,
        train_config,
    )
    with pytest.raises(ValueError, match="inputs have changed"):
        validate_parent_counterfactual_resume(
            plan,
            suite,
            loaded_final,
            train_batches,
            init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
            model_config,
            packed,
            lora_config,
            replace(train_config, learning_rate=2e-2),
        )

    dangling_target = tmp_path / "dangling-transfer-chunk"
    dangling_target.symlink_to(
        tmp_path / "missing-transfer-chunk",
        target_is_directory=True,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        save_tinyworlds_transfer_chunk(dangling_target, 3, completed)
    assert dangling_target.is_symlink()
    resumed_files = tuple(
        (
            str(path.relative_to(resumed_directory)),
            path.read_bytes(),
        )
        for path in sorted(resumed_directory.rglob("*"))
        if path.is_file()
    )
    uninterrupted_files = tuple(
        (
            str(path.relative_to(uninterrupted_directory)),
            path.read_bytes(),
        )
        for path in sorted(uninterrupted_directory.rglob("*"))
        if path.is_file()
    )
    assert resumed_files == uninterrupted_files

    assert all(trial.final_update == 4 for trial in completed.diagnostics.trials)
    assert all(
        tuple(checkpoint.update for checkpoint in trial.checkpoints) == (0, 1, 2, 4)
        for trial in completed.diagnostics.trials
    )
    malformed_trial = completed.diagnostics.trials[0]
    with pytest.raises(ValueError, match="power-of-two"):
        replace(
            malformed_trial,
            checkpoints=(
                malformed_trial.checkpoints[0],
                malformed_trial.checkpoints[2],
                malformed_trial.checkpoints[-1],
            ),
        )
    assert all(len(trial.step_losses) == 4 for trial in completed.diagnostics.trials)
    assert completed.diagnostics.selected_parent_recovered
    transfer_records = pilot_module._counterfactual_transfer_records(
        completed.diagnostics,
        3,
    )
    for role in (
        "root",
        "true_parent",
        "selected_parent",
        "strongest_other_family",
    ):
        assert tuple(
            record.require("update")
            for record in transfer_records
            if record.require("parent_kind") == role
        ) == (0, 1, 2, 4)
    other_family_trial = completed.diagnostics.trial_for_role(
        "strongest_other_family"
    )
    assert other_family_trial is not None
    assert other_family_trial.parent_node_id == NodeId("sunny-seed")
    assert validation_queries_seen
    assert set(validation_queries_seen) == {suite.queries}
    assert all(values.shape == (4,) for values in parent_coefficients_seen)
    with pytest.raises(ValueError, match="inputs have changed"):
        run_parent_counterfactuals(
            plan,
            suite,
            restored,
            train_batches,
            init_gpt_neo_params(jax.random.PRNGKey(12), model_config),
            model_config,
            packed,
            lora_config,
            replace(train_config, learning_rate=2e-2),
        )

    committed = commit_selected_counterfactual_edge(graph, completed)

    assert len(graph.nodes) == 3
    assert len(committed.nodes) == 4
    assert committed.nodes[-1].node_id == NodeId("willow-extension")
    assert committed.nodes[-1].parent_id == NodeId("willow-seed")
    assert committed.nodes[-1].incoming_edge is completed.selected_state.trainable
    selected_trial_index = tuple(
        trial.parent_node_id for trial in completed.diagnostics.trials
    ).index(NodeId("willow-seed"))
    assert committed.nodes[-1].incoming_edge is completed.final_states[
        selected_trial_index
    ].trainable
    with pytest.raises(ValueError, match="next graph stage"):
        commit_selected_counterfactual_edge(
            graph,
            completed,
            train_stage=99,
        )
