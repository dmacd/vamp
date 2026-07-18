from __future__ import annotations

from dataclasses import dataclass, fields, replace
from hashlib import sha256
from pathlib import Path
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.continual.tinyworlds_pilot_run as pilot_module
import apm.continual.knowledge_training as knowledge_training_module
from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.knowledge_evaluation import KNOWLEDGE_AGGREGATION_AXES
from apm.continual.knowledge_training import (
    KnowledgeCounterfactualTraining,
    KnowledgeTransferDiagnostics,
    ParentTransferTrialDiagnostic,
    TransferCheckpointDiagnostic,
)
from apm.continual.language_tasks import (
    AddressBook,
    CompetenceBatch,
    LanguageCurriculum,
    LanguageTask,
    RouterBatch,
    build_prefix_suffix_batches,
)
from apm.continual.tinyworlds_report import (
    TINYWORLDS_NATURAL_CONTINUATION_METHODS,
    TINYWORLDS_REPORT_CUE_REGIMES,
    TINYWORLDS_REPORT_METHODS,
    TINYWORLDS_REPORT_PREFIX_LENGTHS,
    TINYWORLDS_REPORT_TASK_IDS,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig, LmTrainState
from apm.lm.workflow import LmLossTrace
from apm.data.text.tinyworlds.query_generation import generate_pilot_bundle
from apm.data.text.tinyworlds.rendering import (
    TinyWorldsRenderPreset,
    render_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.schema import (
    CandidateRole,
    DataSplit,
    QueryKind,
    TaskId as SymbolicTaskId,
    TaskKind,
)
from apm.memory.graph import (
    NodeId,
    TaskId,
    add_memory_node,
    init_memory_graph,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=16,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _knowledge_model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=64,
        max_position_embeddings=256,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=1,
        attention_types=("global",),
        local_window_size=4,
    )


@dataclass(frozen=True)
class _WhitespaceTokenizer:
    @property
    def vocab_size(self) -> int:
        return 65_536

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        tokens = tuple(
            2 + int.from_bytes(sha256(word.encode("utf-8")).digest()[:2], "big")
            for word in re.findall(r"\S+", text)
        )
        return tokens + ((self.eos_token_id,) if add_eos else ())

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _training_task(task_id: str, marker: int) -> LanguageTask:
    batch = TokenBatch(
        input_ids=np.asarray(((marker, 2, 3, 4),), dtype=np.int32),
        attention_mask=np.ones((1, 4), dtype=np.bool_),
        target_ids=np.asarray(((2, 3, 4, 5),), dtype=np.int32),
        loss_mask=np.ones((1, 4), dtype=np.bool_),
    )
    return LanguageTask(
        task_id=TaskId(task_id),
        train_batches=(batch,),
        validation_examples=(),
        test_examples=(),
    )


def _curriculum() -> LanguageCurriculum:
    return LanguageCurriculum(
        tasks=(
            _training_task("task-a", 1),
            _training_task("task-b", 5),
        ),
        max_nodes=3,
        max_edges=2,
    )


def _artifact_tree(directory: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (str(path.relative_to(directory)), path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    )


def _compact_token_array(values: np.ndarray, vocab_size: int) -> np.ndarray:
    """Map rendered fixture IDs into a tiny vocabulary without changing masks."""
    source = np.asarray(values)
    return np.where(
        source < 2,
        source,
        2 + (source - 2) % (vocab_size - 2),
    ).astype(np.int32)


def _compact_knowledge_query(
    query: KnowledgeQuery,
    vocab_size: int,
) -> KnowledgeQuery:
    router = replace(
        query.router_batch,
        input_ids=_compact_token_array(query.router_batch.input_ids, vocab_size),
        target_ids=_compact_token_array(query.router_batch.target_ids, vocab_size),
    )
    candidates = tuple(
        replace(
            candidate,
            competence_batch=replace(
                candidate.competence_batch,
                input_ids=_compact_token_array(
                    candidate.competence_batch.input_ids,
                    vocab_size,
                ),
                target_ids=_compact_token_array(
                    candidate.competence_batch.target_ids,
                    vocab_size,
                ),
            ),
        )
        for candidate in query.candidates
    )
    return replace(query, router_batch=router, candidates=candidates)


def _bounded_all_slice_queries(vocab_size: int) -> tuple[KnowledgeQuery, ...]:
    """Select a small rendered matrix covering every task and query family."""
    bundle = generate_pilot_bundle("f" * 64)
    rendered = render_tinyworlds_bundle(
        bundle,
        _WhitespaceTokenizer(),
        TinyWorldsRenderPreset(1, 1, 1, 8, 8, 1, 256, 256),
    )
    remaining = [
        group for group in rendered.query_groups if group.split is DataSplit.TEST
    ]
    expected_kinds = {kind.value for kind in QueryKind}
    assert {
        group.variants[0].knowledge_query.query_kind for group in remaining
    } == expected_kinds
    requirements = {
        ("task_id", task_id) for task_id in TINYWORLDS_REPORT_TASK_IDS
    } | {("query_kind", query_kind) for query_kind in expected_kinds}
    selected = []
    while requirements:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(
                    requirements
                    & {
                        ("task_id", remaining[index].task_id),
                        (
                            "query_kind",
                            remaining[index]
                            .variants[0]
                            .knowledge_query.query_kind,
                        ),
                    }
                ),
                -index,
            ),
        )
        group = remaining.pop(best_index)
        covered = requirements & {
            ("task_id", group.task_id),
            ("query_kind", group.variants[0].knowledge_query.query_kind),
        }
        assert covered
        requirements.difference_update(covered)
        selected.append(group)
    assert len(selected) <= 12
    return tuple(
        _compact_knowledge_query(variant.knowledge_query, vocab_size)
        for group in selected
        for variant in group.variants
    )


def _assert_adapter_runs_equal(first, second, stream: str) -> None:
    first_records = first.stages if stream == "sequential" else first.adapters
    second_records = second.stages if stream == "sequential" else second.adapters
    assert tuple(record.task_id for record in first_records) == tuple(
        record.task_id for record in second_records
    )
    assert tuple(record.step_losses for record in first_records) == tuple(
        record.step_losses for record in second_records
    )
    for first_record, second_record in zip(first_records, second_records):
        first_leaves = jax.tree_util.tree_leaves(first_record.adapter)
        second_leaves = jax.tree_util.tree_leaves(second_record.adapter)
        for first_leaf, second_leaf in zip(first_leaves, second_leaves):
            np.testing.assert_array_equal(first_leaf, second_leaf)
    np.testing.assert_array_equal(first.rng_key, second.rng_key)


@pytest.mark.parametrize("stream", ("sequential", "independent"))
def test_baseline_stream_resumes_from_shared_full_state_in_a_fresh_outer_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: str,
) -> None:
    interrupted = False

    def fake_resumable(state, *args, stop_update=None, **kwargs):
        nonlocal interrupted
        del kwargs
        train_config = args[7]
        target = train_config.steps if stop_update is None else stop_update
        start = int(state.step)
        if target == 4 and start == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated process interruption")
        current = state
        losses = []
        for update in range(start + 1, target + 1):
            current = LmTrainState(
                trainable=jax.tree_util.tree_map(
                    lambda value: value + jnp.asarray(1e-4, value.dtype),
                    current.trainable,
                ),
                opt_state=jax.tree_util.tree_map(
                    lambda value: value + jnp.ones_like(value),
                    current.opt_state,
                ),
                rng_key=jax.random.split(current.rng_key)[0],
                step=jnp.asarray(update, dtype=jnp.int32),
            )
            losses.append(float(update) / 10.0)
        return current, LmLossTrace(tuple(losses)), ()

    monkeypatch.setattr(
        pilot_module,
        "run_resumable_candidate_edge_updates",
        fake_resumable,
    )
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=4,
        batch_size=1,
        weight_decay=0.0,
    )
    curriculum = _curriculum()
    resume_root = tmp_path / "stable-cache" / stream
    clean_root = tmp_path / "clean-cache" / stream
    training_function = (
        pilot_module._train_resumable_sequential_lora
        if stream == "sequential"
        else pilot_module._train_resumable_independent_lora
    )
    arguments = (
        curriculum,
        base_params,
        model_config,
        lora_config,
        train_config,
        jax.random.PRNGKey(7),
    )

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        training_function(
            *arguments,
            resume_root,
            "a" * 64,
        )
    assert (resume_root / "01-task-a" / "update-0000002").is_dir()

    resumed = training_function(
        *arguments,
        resume_root,
        "a" * 64,
    )
    clean = training_function(
        *arguments,
        clean_root,
        "a" * 64,
    )

    _assert_adapter_runs_equal(resumed, clean, stream)
    assert _artifact_tree(resume_root) == _artifact_tree(clean_root)


def _evaluation_batches(prefix_length: int) -> tuple[RouterBatch, CompetenceBatch]:
    router_width = prefix_length - 1
    competence_width = prefix_length + 3
    input_ids = np.arange(competence_width, dtype=np.int32)[None, :] % 13
    target_ids = np.roll(input_ids, -1, axis=1)
    attention = np.ones((1, competence_width), dtype=np.bool_)
    competence_loss = np.zeros((1, competence_width), dtype=np.bool_)
    competence_loss[:, router_width:] = True
    return (
        RouterBatch(
            input_ids=input_ids[:, :router_width],
            attention_mask=attention[:, :router_width],
            target_ids=target_ids[:, :router_width],
            loss_mask=attention[:, :router_width],
        ),
        CompetenceBatch(
            input_ids=input_ids,
            attention_mask=attention,
            target_ids=target_ids,
            loss_mask=competence_loss,
        ),
    )


def test_natural_continuation_rows_cover_every_task_method_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    microbatches: list[int | None] = []

    def fake_pack(graph, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(node_count=len(graph.nodes))

    def fake_pack_root(*args, **kwargs):
        del args, kwargs
        return None, SimpleNamespace(node_count=2)

    def fake_competence(*args, evaluation_microbatch_size=None, **kwargs):
        del kwargs
        memory = args[2]
        batch = args[4]
        microbatches.append(evaluation_microbatch_size)
        return np.repeat(
            np.arange(1, memory.node_count + 1, dtype=np.float32)[None, :],
            batch.input_ids.shape[0],
            axis=0,
        )

    def fake_route(*args, evaluation_microbatch_size=None, **kwargs):
        del kwargs
        batch = args[6]
        microbatches.append(evaluation_microbatch_size)
        return SimpleNamespace(
            selected_indices=np.zeros((batch.input_ids.shape[0],), dtype=np.int32)
        )

    monkeypatch.setattr(pilot_module, "pack_lora_memory", fake_pack)
    monkeypatch.setattr(pilot_module, "pack_root_adapter", fake_pack_root)
    monkeypatch.setattr(pilot_module, "competence_nll_by_node", fake_competence)
    monkeypatch.setattr(pilot_module, "route_language_prefix", fake_route)
    tasks = tuple(
        SimpleNamespace(
            task_id=TaskId(task_id),
            test_examples=tuple(
                SimpleNamespace(
                    router_batch=router,
                    competence_batch=competence,
                )
                for prefix_length in TINYWORLDS_REPORT_PREFIX_LENGTHS
                for router, competence in (_evaluation_batches(prefix_length),)
            ),
        )
        for task_id in TINYWORLDS_REPORT_TASK_IDS
    )
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId(TINYWORLDS_REPORT_TASK_IDS[0]),
        NodeId("root"),
        TaskId(TINYWORLDS_REPORT_TASK_IDS[0]),
        1,
        object(),
    )
    inputs = SimpleNamespace(
        base_artifact=SimpleNamespace(
            checkpoint=SimpleNamespace(params=object(), config=object())
        ),
        lora_config=object(),
        prepared=SimpleNamespace(
            language=SimpleNamespace(
                curriculum=SimpleNamespace(tasks=tasks)
            )
        ),
        execution_preset=SimpleNamespace(
            evaluation_microbatch_size=8,
            random_router_seed=0,
        ),
    )
    adaptations = SimpleNamespace(
        sequential=SimpleNamespace(
            stages=(SimpleNamespace(adapter=object()),)
        ),
        independent=SimpleNamespace(
            adapters=(
                SimpleNamespace(
                    task_id=TaskId(TINYWORLDS_REPORT_TASK_IDS[0]),
                    adapter=object(),
                ),
            )
        ),
        vamp_stages=(
            SimpleNamespace(graph=graph, address_book=object()),
        ),
    )

    records = pilot_module._natural_continuation_records(
        inputs,
        adaptations,
        1,
    )

    represented = {
        (
            record.require("method"),
            record.require("task_id"),
            record.require("prefix_length"),
        )
        for record in records
    }
    expected = {
        (method, task_id, prefix_length)
        for method in TINYWORLDS_NATURAL_CONTINUATION_METHODS
        for task_id in TINYWORLDS_REPORT_TASK_IDS
        for prefix_length in TINYWORLDS_REPORT_PREFIX_LENGTHS
    }
    assert represented == expected
    assert len(records) == len(expected)
    assert microbatches and set(microbatches) == {8}


def test_exact_kg_gate_is_computed_from_every_test_variant() -> None:
    bundle = generate_pilot_bundle("b" * 64)
    rendered = render_tinyworlds_bundle(
        bundle,
        _WhitespaceTokenizer(),
        TinyWorldsRenderPreset(1, 1, 1, 1, 1, 1, 256, 256),
    )

    evidence = pilot_module._exact_kg_test_evidence(
        SimpleNamespace(symbolic_bundle=bundle, rendered=rendered)
    )

    test_variant_count = sum(
        len(group.variants)
        for group in rendered.query_groups
        if group.split is DataSplit.TEST
    )
    assert evidence.trials == test_variant_count
    assert evidence.successes == evidence.trials
    assert evidence.accuracy == 1.0

    test_groups = tuple(
        group for group in rendered.query_groups if group.split is DataSplit.TEST
    )
    first_group = test_groups[0]
    first_variant = first_group.variants[0]
    tampered_variant = SimpleNamespace(
        candidate_entity_ids=(first_variant.candidate_entity_ids[0],) * 4,
        knowledge_query=first_variant.knowledge_query,
    )
    tampered_group = SimpleNamespace(
        split=first_group.split,
        symbolic_query_id=first_group.symbolic_query_id,
        variants=(tampered_variant, *first_group.variants[1:]),
    )
    tampered = pilot_module._exact_kg_test_evidence(
        SimpleNamespace(
            symbolic_bundle=bundle,
            rendered=SimpleNamespace(
                query_groups=(tampered_group, *test_groups[1:])
            ),
        )
    )
    assert tampered.trials == evidence.trials
    assert tampered.successes == evidence.successes - 1


def test_bounded_two_task_knowledge_stage_covers_all_methods_and_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production stage evaluator with real scores and routers."""
    model_config = _knowledge_model_config()
    queries = _bounded_all_slice_queries(model_config.vocab_size)
    assert {str(query.task_id) for query in queries} == set(
        TINYWORLDS_REPORT_TASK_IDS
    )
    assert {query.query_kind for query in queries} == {
        kind.value for kind in QueryKind
    }
    assert {query.prefix_length for query in queries} == set(
        TINYWORLDS_REPORT_PREFIX_LENGTHS
    )
    assert {query.cue_regime for query in queries} == set(
        TINYWORLDS_REPORT_CUE_REGIMES
    )
    assert {query.reasoning_depth for query in queries} == {0, 1, 2}
    assert {query.mode for query in queries} == {"closed_book", "open_book"}

    lora_config = LoraConfig(rank=1, alpha=1.0)
    base_params = init_gpt_neo_params(jax.random.PRNGKey(20), model_config)
    edges = tuple(
        jax.tree_util.tree_map(
            lambda value, scale=scale: jnp.full_like(value, scale),
            init_lora_edge(
                jax.random.PRNGKey(21 + index),
                model_config,
                lora_config,
            ),
        )
        for index, scale in enumerate((0.005, 0.01))
    )
    root_id = NodeId("root")
    committed_ids = tuple(
        NodeId(task_id) for task_id in TINYWORLDS_REPORT_TASK_IDS[:2]
    )
    graph = init_memory_graph(root_id)
    for stage, (node_id, edge) in enumerate(
        zip(committed_ids, edges),
        start=1,
    ):
        graph = add_memory_node(
            graph,
            node_id,
            root_id,
            TaskId(str(node_id)),
            stage,
            edge,
        )
    address_keys = np.zeros((9, model_config.hidden_size), dtype=np.float32)
    address_keys[:3, :3] = np.eye(3, dtype=np.float32)
    address_book = AddressBook(
        node_ids=(root_id, *committed_ids, *(None for _ in range(6))),
        keys=address_keys,
        valid_node_mask=np.asarray((True, True, True) + (False,) * 6),
    )
    inputs = SimpleNamespace(
        base_artifact=SimpleNamespace(
            checkpoint=SimpleNamespace(params=base_params, config=model_config)
        ),
        lora_config=lora_config,
        prepared=SimpleNamespace(test_queries=queries),
        execution_preset=SimpleNamespace(
            evaluation_microbatch_size=4,
            random_router_seed=0,
        ),
    )
    adaptations = SimpleNamespace(
        sequential=SimpleNamespace(
            stages=tuple(SimpleNamespace(adapter=edge) for edge in edges)
        ),
        independent=SimpleNamespace(
            adapters=tuple(
                SimpleNamespace(task_id=TaskId(str(node_id)), adapter=edge)
                for node_id, edge in zip(committed_ids, edges)
            )
        ),
        vamp_stages=(
            SimpleNamespace(graph=graph, address_book=address_book),
            SimpleNamespace(graph=graph, address_book=address_book),
        ),
    )
    monkeypatch.setattr(
        pilot_module,
        "_time_hard_router_cold_queries",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        pilot_module,
        "_time_ebt_cold_queries",
        lambda *args, **kwargs: 0.0,
    )

    result = pilot_module._evaluate_stage_knowledge(inputs, adaptations, stage=2)

    assert tuple(
        evaluation.method for evaluation in result.evaluations
    ) == TINYWORLDS_REPORT_METHODS
    assert result.hard_candidate_nll.shape == (len(queries), 4, 9)
    assert np.all(np.isfinite(result.hard_candidate_nll[:, :, :3]))
    assert np.all(np.isposinf(result.hard_candidate_nll[:, :, 3:]))
    expected_query_ids = tuple(query.query_id for query in queries)
    for evaluation in result.evaluations:
        assert tuple(row.query_id for row in evaluation.queries) == expected_query_ids
        expected_slices = {("all", "all")}
        expected_slices.update(
            (axis, getattr(row, axis))
            for axis in KNOWLEDGE_AGGREGATION_AXES
            if axis != "all"
            for row in evaluation.queries
        )
        assert {
            (aggregate.grouping_axis, aggregate.grouping_value)
            for aggregate in evaluation.aggregates
        } == expected_slices
        for row in evaluation.queries:
            assert np.all(np.isfinite(row.candidate_nll))
            for field_name in (
                "candidate_margin",
                "correct_answer_nll",
                "routed_correct_answer_nll",
                "task_oracle_correct_answer_nll",
                "best_hard_node_correct_answer_nll",
                "routed_regret",
                "task_oracle_regret",
                "best_hard_node_regret",
                "address_entropy",
                "address_margin",
                "hard_required_edge_recall",
                "soft_required_edge_mean_coefficient",
            ):
                value = getattr(row, field_name)
                assert value is None or np.isfinite(value)
        for aggregate in evaluation.aggregates:
            for field in fields(aggregate):
                value = getattr(aggregate, field.name)
                if isinstance(value, (float, np.floating)):
                    assert np.isfinite(value)

    routed = next(
        evaluation
        for evaluation in result.evaluations
        if evaluation.method == "vamp_exhaustive"
    )
    cross_branch = tuple(
        row for row in routed.queries if row.query_kind == "cross_branch"
    )
    assert cross_branch
    assert all(row.task_oracle_node_index is None for row in cross_branch)
    assert all(row.hard_required_edge_recall == 0.0 for row in cross_branch)
    soft_methods = tuple(
        evaluation
        for evaluation in result.evaluations
        if evaluation.method.endswith("_soft")
    )
    assert len(soft_methods) == 2
    assert all(
        evaluation.edge_coefficients is not None
        and np.all(np.isfinite(evaluation.edge_coefficients))
        for evaluation in soft_methods
    )


def test_vamp_stage_uses_runtime_distinct_symbolic_task_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = generate_pilot_bundle("c" * 64)
    symbolic_task = bundle.tasks[0]
    task_id = TaskId(str(symbolic_task.task_id))
    family_id = str(
        bundle.world.task(SymbolicTaskId(str(task_id))).family_id
    )
    prefix = (1, 2, 3)
    batches = tuple(
        build_prefix_suffix_batches(
            prefix + (8 + index, 12 + index),
            prefix_length=len(prefix),
            suffix_length=2,
        )
        for index in range(4)
    )
    query = KnowledgeQuery(
        query_id="typed-symbolic-task-query",
        task_id=task_id,
        family_id=family_id,
        query_kind="direct",
        candidates=tuple(
            KnowledgeCandidate(answer, batch[1])
            for answer, batch in zip(("a", "b", "c", "d"), batches)
        ),
        router_batch=batches[0][0],
        correct_candidate_index=0,
        proof_id="typed-proof",
        support_ids=("typed-fact",),
        required_edge_ids=(),
        cue_regime="cue_sufficient",
        visible_cue_ids=("typed-cue",),
        eligible_task_ids=(task_id,),
        novelty_regime="direct",
        reasoning_type="direct",
        reasoning_depth=0,
        prefix_length=len(prefix),
        mode="closed_book",
        oracle_node_ids=(NodeId("root"),),
    )
    train_batch = TokenBatch(
        input_ids=np.asarray(((1, 2, 3, 4),), dtype=np.int32),
        attention_mask=np.ones((1, 4), dtype=np.bool_),
        target_ids=np.asarray(((2, 3, 4, 5),), dtype=np.int32),
        loss_mask=np.ones((1, 4), dtype=np.bool_),
    )
    task = LanguageTask(
        task_id=task_id,
        train_batches=(train_batch,),
        validation_examples=(),
        test_examples=(),
        parent_probes=(batches[0][0],),
        content_key_probes=(batches[0][0],),
    )
    model_config = _model_config()
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=1,
        batch_size=1,
        weight_decay=0.0,
    )
    address_book = AddressBook(
        node_ids=(NodeId("root"),) + (None,) * 8,
        keys=np.zeros((9, model_config.hidden_size), dtype=np.float32),
        valid_node_mask=np.asarray((True,) + (False,) * 8),
    )
    current = pilot_module._VampStageState(
        graph=init_memory_graph(NodeId("root")),
        address_book=address_book,
        rng_key=jax.random.PRNGKey(4),
        stage_metrics=(),
        parent_diagnostics=(),
    )
    inputs = SimpleNamespace(
        prepared=SimpleNamespace(validation_queries=(query,)),
        symbolic_bundle=bundle,
        lora_config=lora_config,
        train_config=train_config,
        execution_preset=SimpleNamespace(evaluation_microbatch_size=8),
    )

    def fake_hard_scores(*args, **kwargs):
        del args, kwargs
        scores = np.full((1, 4, 9), np.inf, dtype=np.float32)
        scores[0, :, 0] = (0.5, 1.0, 1.5, 2.0)
        return scores

    def fake_resume(_root, _stage, plan, state_template, _budget):
        final_state = replace(
            state_template,
            step=jnp.asarray(1, dtype=jnp.int32),
        )
        adapter_checksum = knowledge_training_module._tree_checksum(
            state_template.trainable
        )
        roles = tuple(
            target.role
            for target in plan.targets
            if target.parent_node_id == NodeId("root")
        )
        trial = ParentTransferTrialDiagnostic(
            parent_node_index=0,
            parent_node_id=NodeId("root"),
            roles=roles,
            parent_validation_mean_correct_nll=(
                plan.parent_search.mean_correct_candidate_nll[0]
            ),
            initial_state_checksum=knowledge_training_module._tree_checksum(
                state_template
            ),
            final_adapter_checksum=adapter_checksum,
            final_state_checksum=knowledge_training_module._tree_checksum(
                final_state
            ),
            final_update=1,
            step_losses=(0.1,),
            checkpoints=(
                TransferCheckpointDiagnostic(0, None, 0.25, 0.5, adapter_checksum),
                TransferCheckpointDiagnostic(1, 0.1, 0.5, 0.4, adapter_checksum),
            ),
        )
        diagnostics = KnowledgeTransferDiagnostics(plan, (trial,))
        return KnowledgeCounterfactualTraining(
            diagnostics,
            (final_state,),
            "d" * 64,
        )

    validated_resumes: list[KnowledgeCounterfactualTraining] = []

    def fake_validate_resume(_plan, _suite, training, *args):
        del args
        validated_resumes.append(training)

    monkeypatch.setattr(
        pilot_module,
        "_score_hard_candidates_grouped",
        fake_hard_scores,
    )
    monkeypatch.setattr(
        pilot_module,
        "_load_latest_transfer_chunk",
        fake_resume,
    )
    monkeypatch.setattr(
        pilot_module,
        "validate_parent_counterfactual_resume",
        fake_validate_resume,
    )
    monkeypatch.setattr(
        pilot_module,
        "derive_node_content_key",
        lambda *args, **kwargs: np.asarray(
            (1.0,) + (0.0,) * (model_config.hidden_size - 1),
            dtype=np.float32,
        ),
    )

    result = pilot_module._train_vamp_stage(
        inputs,
        current,
        task,
        1,
        object(),
        model_config,
        tmp_path,
    )

    assert result.graph.nodes[-1].node_id == NodeId(str(symbolic_task.task_id))
    assert result.graph.nodes[-1].parent_id == NodeId("root")
    assert len(validated_resumes) == 1
    assert validated_resumes[0].diagnostics.trials[0].final_update == 1


def test_revision_retention_uses_variant_local_rotated_candidate_indices() -> None:
    task_specs = tuple(
        SimpleNamespace(
            task_id=task_id,
            family_id=family_id,
            kind=kind,
        )
        for family_id, seed_id, revision_id in (
            ("willow", "willow-seed", "willow-revision"),
            ("sunny", "sunny-seed", "sunny-revision"),
        )
        for task_id, kind in (
            (seed_id, TaskKind.SEED),
            (revision_id, TaskKind.REVISION),
        )
    )
    plans = []
    groups = []
    test_queries = []
    graph = init_memory_graph(NodeId("root"))
    for family_index, family_id in enumerate(("willow", "sunny")):
        seed_id = f"{family_id}-seed"
        revision_id = f"{family_id}-revision"
        graph = add_memory_node(
            graph,
            NodeId(seed_id),
            NodeId("root"),
            TaskId(seed_id),
            family_index * 2 + 1,
            object(),
        )
        graph = add_memory_node(
            graph,
            NodeId(revision_id),
            NodeId(seed_id),
            TaskId(revision_id),
            family_index * 2 + 2,
            object(),
        )
        symbolic_query_id = f"{family_id}-revision-query"
        plans.append(
            SimpleNamespace(
                query_ast=SimpleNamespace(query_id=symbolic_query_id),
                kind=QueryKind.REVISION_SENSITIVE,
                candidates=(
                    SimpleNamespace(
                        entity_id=f"{family_id}-old",
                        role=CandidateRole.INCOMPATIBLE_REVISION,
                    ),
                ),
            )
        )
        variants = []
        for variant_index, (candidate_ids, correct_index) in enumerate(
            (
                (
                    (
                        f"{family_id}-new",
                        f"{family_id}-filler-a",
                        f"{family_id}-old",
                        f"{family_id}-filler-b",
                    ),
                    0,
                ),
                (
                    (
                        f"{family_id}-old",
                        f"{family_id}-filler-a",
                        f"{family_id}-filler-b",
                        f"{family_id}-new",
                    ),
                    3,
                ),
            )
        ):
            query_id = f"{symbolic_query_id}:variant-{variant_index}"
            knowledge_query = SimpleNamespace(
                query_id=query_id,
                correct_candidate_index=correct_index,
            )
            test_queries.append(knowledge_query)
            variants.append(
                SimpleNamespace(
                    candidate_entity_ids=candidate_ids,
                    knowledge_query=knowledge_query,
                )
            )
        groups.append(
            SimpleNamespace(
                split=DataSplit.TEST,
                task_id=revision_id,
                symbolic_query_id=symbolic_query_id,
                variants=tuple(variants),
            )
        )
    node_index = {
        str(node.node_id): index for index, node in enumerate(graph.nodes)
    }
    hard_scores = np.full(
        (len(test_queries), 4, len(graph.nodes)),
        10.0,
        dtype=np.float32,
    )
    for row, query in enumerate(test_queries):
        family_id = "willow" if query.query_id.startswith("willow") else "sunny"
        variant = next(
            variant
            for group in groups
            for variant in group.variants
            if variant.knowledge_query.query_id == query.query_id
        )
        old_index = variant.candidate_entity_ids.index(f"{family_id}-old")
        hard_scores[row, old_index, node_index[f"{family_id}-seed"]] = 0.0
        hard_scores[
            row,
            query.correct_candidate_index,
            node_index[f"{family_id}-revision"],
        ] = 0.0
    inputs = SimpleNamespace(
        prepared=SimpleNamespace(test_queries=tuple(test_queries)),
        symbolic_bundle=SimpleNamespace(
            tasks=task_specs,
            query_plans=tuple(plans),
        ),
        rendered=SimpleNamespace(query_groups=tuple(groups)),
    )

    records = pilot_module._revision_retention_records(
        inputs,
        graph,
        hard_scores,
    )

    assert len(records) == 2
    assert all(record.require("old_context_accuracy") == 1.0 for record in records)
    assert all(
        record.require("revision_context_accuracy") == 1.0 for record in records
    )
    assert all(
        record.require("paired_revision_consistency") == 1.0
        for record in records
    )
