from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.data.text.tinyworlds_nouns_v2.addressing_study_keys as key_module
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
    EBT_CONTRACT_FORMAT,
    EBT_ROW_FORMAT,
    RETRIEVAL_CONTRACT_FORMAT,
    TIMING_ROW_FORMAT,
    EbtStudyRow,
    RetrievalStudyRow,
    canonical_json_bytes,
    contract_record,
    record_sha256,
    validate_jsonl_rows,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
    AddressingKeyArtifact,
    analytic_final_hidden_residual,
    midpoint_probe_batch,
    score_key_scheme,
    stable_hopfield_result,
)
from apm.lm.compact_lora_memory import (
    compact_node_weights_to_edge_coefficients,
    expand_compact_edge_coefficients,
    gather_compact_lora_memory,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import node_weights_to_edge_coefficients, pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.address_refinement import (
    EbtConfig,
    refine_compact_ebt_address,
    refine_ebt_address,
)
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=17,
        max_position_embeddings=32,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=2, alpha=2.0)


def _branching_memory():
    parents = ("root", "a", "b", "root", "d", "a", "root", "g")
    node_ids = ("a", "b", "c", "d", "e", "f", "g", "h")
    graph = init_memory_graph(NodeId("root"))
    for index, (node_id, parent_id) in enumerate(zip(node_ids, parents), start=1):
        initialized = init_lora_edge(
            jax.random.PRNGKey(index),
            _model_config(),
            _lora_config(),
        )
        edge = jax.tree_util.tree_map(
            lambda value: jnp.full_like(value, 0.01 * index),
            initialized,
        )
        graph = add_memory_node(
            graph,
            NodeId(node_id),
            NodeId(parent_id),
            TaskId(f"task-{node_id}"),
            index,
            edge,
        )
    return pack_lora_memory(graph, _model_config(), _lora_config(), 9, 8)


def _prefix_batch() -> RouterBatch:
    return RouterBatch(
        input_ids=np.asarray(((1, 2, 3, 4), (4, 3, 2, 1)), dtype=np.int32),
        attention_mask=np.ones((2, 4), dtype=np.bool_),
        target_ids=np.asarray(((2, 3, 4, 5), (3, 2, 1, 0)), dtype=np.int32),
        loss_mask=np.ones((2, 4), dtype=np.bool_),
    )


def test_analytic_residual_matches_autodiff_and_ignores_masked_padding() -> None:
    generator = np.random.default_rng(4)
    hidden = jnp.asarray(generator.normal(size=(2, 4, 5)), dtype=jnp.float32)
    embeddings = jnp.asarray(generator.normal(size=(7, 5)), dtype=jnp.float32)
    targets = jnp.asarray(((0, 1, 2, 3), (3, 4, 5, 6)), dtype=jnp.int32)
    mask = jnp.asarray(((1, 1, 0, 0), (1, 1, 1, 0)), dtype=jnp.float32)

    def masked_cross_entropy(current_hidden: jax.Array) -> jax.Array:
        logits = jnp.einsum("bth,vh->btv", current_hidden, embeddings)
        token_losses = -jax.nn.log_softmax(logits, axis=-1)[
            jnp.arange(2)[:, None],
            jnp.arange(4)[None, :],
            targets,
        ]
        return jnp.sum(token_losses * mask, axis=1) / jnp.sum(mask, axis=1)

    autodiff = jax.jacrev(masked_cross_entropy)(hidden)
    pooled = jnp.stack(
        tuple(jnp.sum(autodiff[row, row], axis=0) for row in range(2))
    )
    expected = pooled / jnp.linalg.norm(pooled, axis=-1, keepdims=True)
    logits = jnp.einsum("bth,vh->btv", hidden, embeddings)
    actual = analytic_final_hidden_residual(logits, embeddings, targets, mask)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1.0, atol=1e-6)

    padded_logits = jnp.concatenate(
        (logits, jnp.asarray(generator.normal(size=(2, 3, 7)), dtype=jnp.float32)),
        axis=1,
    )
    padded_targets = jnp.concatenate(
        (targets, jnp.asarray(((6, 6, 6), (0, 0, 0)), dtype=jnp.int32)),
        axis=1,
    )
    padded_mask = jnp.concatenate((mask, jnp.zeros((2, 3))), axis=1)
    padded = analytic_final_hidden_residual(
        padded_logits,
        embeddings,
        padded_targets,
        padded_mask,
    )
    np.testing.assert_allclose(padded, actual, rtol=1e-6, atol=1e-6)


def test_midpoint_probe_batch_includes_root_and_excludes_suffix_and_task_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = {
        name: tuple(
            SimpleNamespace(
                story_id=sha256(f"{name}-{index}".encode("utf-8")).hexdigest()
            )
            for index in range(36)
        )
        for name in ("root-probes", "task-pear-probes")
    }
    tokens = {
        entry.story_id: (1, 2, 3, 4, 13, 14, 15, 16)
        for group in entries.values()
        for entry in group
    }

    class FakeStore:
        def __init__(self, partition: object) -> None:
            del partition

        def tokens(self, entry: object) -> tuple[int, ...]:
            return tokens[entry.story_id]

    monkeypatch.setattr(key_module, "IndexedStoryStore", FakeStore)
    monkeypatch.setattr(
        key_module,
        "load_story_index",
        lambda partition, name: entries[name],
    )
    partition = SimpleNamespace(task_ids=("pear",), pad_token_id=0)

    batch, node_ids, probe_ids = midpoint_probe_batch(partition, _model_config())

    assert node_ids == ("root", "pear")
    assert tuple(map(len, probe_ids)) == (36, 36)
    assert batch.input_ids.shape == (72, 32)
    np.testing.assert_array_equal(
        batch.input_ids[:, :3],
        np.tile(np.asarray((1, 2, 3), dtype=np.int32), (72, 1)),
    )
    np.testing.assert_array_equal(
        batch.target_ids[:, :3],
        np.tile(np.asarray((2, 3, 4), dtype=np.int32), (72, 1)),
    )
    assert not np.any(np.isin(batch.input_ids, (13, 14, 15, 16)))
    assert int(np.sum(batch.loss_mask)) == 72 * 3


def test_prototype_scoring_and_stable_ties_are_deterministic(tmp_path) -> None:
    content_centroids = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    content_prototypes = np.repeat(content_centroids[:, None, :], 36, axis=1)
    content_prototypes[1, 0] = np.asarray((1.0, 0.0), dtype=np.float32)
    fused_prototypes = np.concatenate(
        (content_prototypes / np.sqrt(2.0), content_prototypes / np.sqrt(2.0)),
        axis=-1,
    ).astype(np.float32)
    fused_centroids = np.asarray(
        ((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 1.0)),
        dtype=np.float32,
    ) / np.sqrt(np.float32(2.0))
    fused_centroids = fused_centroids.astype(np.float32)
    keys = AddressingKeyArtifact(
        base_parameter_checksum="c" * 64,
        partition_sha256="d" * 64,
        vamp_tensor_checksum="e" * 64,
        node_ids=("root", "pear"),
        probe_story_ids=tuple(tuple(f"{node}-{index}" for index in range(36)) for node in range(2)),
        canonical_full_centroids=content_centroids.copy(),
        midpoint_content_centroids=content_centroids.copy(),
        midpoint_content_prototypes=content_prototypes,
        midpoint_content_residual_centroids=fused_centroids,
        midpoint_content_residual_prototypes=fused_prototypes,
        tensor_sha256="a" * 64,
        artifact_sha256="b" * 64,
        root=tmp_path,
    )
    query = np.asarray(((1.0, 0.0),), dtype=np.float32)
    residual = query.copy()

    centroid_scores = score_key_scheme(
        query,
        residual,
        keys,
        "midpoint_content_centroid",
    )
    prototype_scores = score_key_scheme(
        query,
        residual,
        keys,
        "midpoint_content_prototype",
    )
    fused_scores = score_key_scheme(
        query,
        residual,
        keys,
        "midpoint_content_residual_prototype",
    )
    np.testing.assert_allclose(centroid_scores, ((1.0, 0.0),), atol=1e-6)
    np.testing.assert_allclose(prototype_scores, ((1.0, 1.0),), atol=1e-6)
    np.testing.assert_allclose(fused_scores, ((1.0, 1.0),), atol=1e-6)

    tied = stable_hopfield_result(np.zeros((2, 25), dtype=np.float32))
    np.testing.assert_array_equal(tied.selected_indices, (0, 0))
    np.testing.assert_array_equal(tied.top_k_indices, np.tile(np.arange(8), (2, 1)))
    np.testing.assert_array_equal(
        tied.top_k_indices,
        stable_hopfield_result(np.zeros((2, 25), dtype=np.float32)).top_k_indices,
    )


@pytest.mark.parametrize(
    "candidates",
    (
        np.asarray(((0, 2, 5, 8), (0, 3, 4, 7)), dtype=np.int32),
        np.asarray(
            (
                (0, 1, 2, 3, 4, 5, 7, 8),
                (0, 1, 2, 3, 4, 6, 7, 8),
            ),
            dtype=np.int32,
        ),
    ),
)
def test_compact_top_k_is_physical_and_matches_dense_eager_jit_and_gradients(
    candidates: np.ndarray,
) -> None:
    packed = _branching_memory()
    compact = gather_compact_lora_memory(packed, candidates)
    candidate_weights = jax.nn.softmax(
        jnp.arange(candidates.size, dtype=jnp.float32).reshape(candidates.shape),
        axis=-1,
    )
    compact_coefficients = compact_node_weights_to_edge_coefficients(
        candidate_weights,
        compact,
    )
    dense_weights = jnp.zeros((2, 9), dtype=jnp.float32).at[
        jnp.arange(2)[:, None],
        jnp.asarray(candidates),
    ].set(candidate_weights)
    dense_coefficients = node_weights_to_edge_coefficients(dense_weights, packed)
    expanded = expand_compact_edge_coefficients(compact, compact_coefficients, 8)
    np.testing.assert_allclose(expanded, dense_coefficients, atol=1e-7)
    assert compact.valid_edge_mask.shape[1] in (4, 8, 12, 16, 20, 24)
    assert 0 in candidates
    for row, mask in zip(
        np.asarray(compact.source_edge_indices),
        np.asarray(compact.valid_edge_mask),
    ):
        assert np.all(np.diff(row[mask]) > 0)
    for compact_leaf, dense_leaf in zip(
        jax.tree_util.tree_leaves(compact.edge_bank),
        jax.tree_util.tree_leaves(packed.edge_bank),
    ):
        for row_index in range(2):
            valid = np.asarray(compact.valid_edge_mask[row_index])
            sources = np.asarray(compact.source_edge_indices[row_index])[valid]
            np.testing.assert_array_equal(
                np.asarray(compact_leaf[row_index])[valid],
                np.asarray(dense_leaf)[sources],
            )
            assert np.count_nonzero(np.asarray(compact_leaf[row_index])[~valid]) == 0

    params = init_gpt_neo_params(jax.random.PRNGKey(11), _model_config())
    batch = _prefix_batch()

    def dense_logits(weights: jax.Array) -> jax.Array:
        full_weights = jnp.zeros((2, 9), dtype=jnp.float32).at[
            jnp.arange(2)[:, None],
            jnp.asarray(candidates),
        ].set(weights)
        coefficients = node_weights_to_edge_coefficients(full_weights, packed)
        return apply_gpt_neo(
            params,
            _model_config(),
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.attention_mask),
            lora_memory=packed,
            edge_coefficients=coefficients,
            lora_config=_lora_config(),
        ).logits

    def compact_logits(weights: jax.Array) -> jax.Array:
        coefficients = compact_node_weights_to_edge_coefficients(weights, compact)
        return apply_gpt_neo(
            params,
            _model_config(),
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.attention_mask),
            lora_memory=compact,
            edge_coefficients=coefficients,
            lora_config=_lora_config(),
        ).logits

    np.testing.assert_allclose(
        compact_logits(candidate_weights),
        dense_logits(candidate_weights),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        jax.jit(compact_logits)(candidate_weights),
        jax.jit(dense_logits)(candidate_weights),
        rtol=3e-6,
        atol=3e-6,
    )
    compact_gradient = jax.grad(lambda value: jnp.sum(compact_logits(value)))(
        candidate_weights
    )
    dense_gradient = jax.grad(lambda value: jnp.sum(dense_logits(value)))(
        candidate_weights
    )
    np.testing.assert_allclose(compact_gradient, dense_gradient, rtol=2e-5, atol=2e-5)

    scores = np.full((2, 9), -10.0, dtype=np.float32)
    for row_index, row_candidates in enumerate(candidates):
        scores[row_index, row_candidates] = np.linspace(2.0, 1.0, len(row_candidates))
    hopfield = stable_hopfield_result(scores, top_k=candidates.shape[1])
    np.testing.assert_array_equal(hopfield.top_k_indices, candidates)
    config = EbtConfig(steps=1, initialization="hopfield_top_k")
    compact_result = refine_compact_ebt_address(
        params,
        _model_config(),
        compact,
        _lora_config(),
        batch,
        hopfield,
        config,
    )
    dense_result = refine_ebt_address(
        params,
        _model_config(),
        packed,
        _lora_config(),
        batch,
        config,
        hopfield_result=hopfield,
    )
    np.testing.assert_allclose(
        compact_result.candidate_probabilities,
        np.take_along_axis(
            np.asarray(dense_result.node_probabilities),
            candidates,
            axis=1,
        ),
        rtol=3e-5,
        atol=3e-5,
    )
    np.testing.assert_array_equal(
        compact_result.selected_node_indices,
        dense_result.selected_indices,
    )


def test_jsonl_contracts_reject_tampering_and_resume_only_an_interrupted_tail(
    tmp_path,
) -> None:
    retrieval_contract = contract_record(RETRIEVAL_CONTRACT_FORMAT, {"study": "x"})
    ebt_contract = contract_record(EBT_CONTRACT_FORMAT, {"study": "x"})
    assert retrieval_contract["contract_sha256"] != ebt_contract["contract_sha256"]
    contract_sha = str(retrieval_contract["contract_sha256"])
    story_id = "c" * 64
    record = RetrievalStudyRow(
        retrieval_contract_sha256=contract_sha,
        scheme="canonical_full_centroid",
        task_noun="pear",
        story_id=story_id,
        oracle_node_index=1,
        top_8_indices=tuple(range(8)),
        entropy=1.0,
        score_margin=0.25,
        prefix_token_count=4,
    ).as_record()
    expected = (("pear", story_id, "canonical_full_centroid"),)
    ledger = tmp_path / "retrieval.jsonl"
    complete_payload = canonical_json_bytes(record)
    ledger.write_bytes(complete_payload + b'{"interrupted"')

    from apm.data.text.tinyworlds_nouns_v2.addressing_study import (
        load_timing_ledger,
        repair_interrupted_tail,
        validate_ebt_ledger,
        validate_retrieval_ledger,
    )

    repair_interrupted_tail(ledger)
    assert ledger.read_bytes() == complete_payload
    assert validate_jsonl_rows(
        ledger,
        expected_format=record["format"],
        contract_field="retrieval_contract_sha256",
        contract_sha256=contract_sha,
        key_fields=("task_noun", "story_id", "scheme"),
        expected_keys=set(expected),
        require_complete=True,
    ) == set(expected)
    assert validate_retrieval_ledger(
        ledger,
        contract_sha,
        expected,
        require_complete=True,
    ) == set(expected)
    resumed = ledger.read_bytes()
    repair_interrupted_tail(ledger)
    assert ledger.read_bytes() == resumed

    second_story_id = "d" * 64
    second_record = RetrievalStudyRow(
        retrieval_contract_sha256=contract_sha,
        scheme="canonical_full_centroid",
        task_noun="pear",
        story_id=second_story_id,
        oracle_node_index=1,
        top_8_indices=tuple(range(8)),
        entropy=1.0,
        score_margin=0.25,
        prefix_token_count=4,
    ).as_record()
    second_payload = canonical_json_bytes(second_record)
    ledger.write_bytes(complete_payload + second_payload[: len(second_payload) // 2])
    repair_interrupted_tail(ledger)
    with ledger.open("ab") as stream:
        stream.write(second_payload)
    uninterrupted = complete_payload + second_payload
    assert ledger.read_bytes() == uninterrupted
    expected_pair = (
        ("pear", story_id, "canonical_full_centroid"),
        ("pear", second_story_id, "canonical_full_centroid"),
    )
    assert validate_retrieval_ledger(
        ledger,
        contract_sha,
        expected_pair,
        require_complete=True,
    ) == set(expected_pair)

    ledger.write_bytes(complete_payload + complete_payload)
    with pytest.raises(ValueError, match="duplicate"):
        validate_jsonl_rows(
            ledger,
            expected_format=record["format"],
            contract_field="retrieval_contract_sha256",
            contract_sha256=contract_sha,
            key_fields=("task_noun", "story_id", "scheme"),
            expected_keys=set(expected),
            require_complete=True,
        )

    tampered = dict(record)
    tampered["entropy"] = 9.0
    ledger.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="identity"):
        validate_jsonl_rows(
            ledger,
            expected_format=record["format"],
            contract_field="retrieval_contract_sha256",
            contract_sha256=contract_sha,
            key_fields=("task_noun", "story_id", "scheme"),
            expected_keys=set(expected),
            require_complete=True,
        )

    ebt_row = EbtStudyRow(
        ebt_contract_sha256=str(ebt_contract["contract_sha256"]),
        scheme="canonical_full_centroid",
        mode="compact",
        candidate_width=4,
        task_noun="pear",
        story_id=story_id,
        oracle_node_index=1,
        candidate_node_indices=(0, 1, 2, 3),
        selected_node_index=1,
        selected_path=("root", "pear"),
        gathered_edge_count=3,
        selected_path_edge_count=1,
        physical_edge_capacity=4,
        prefix_token_count=4,
        prefix_width_bucket=32,
        suffix_total_nll=8.0,
        suffix_token_count=4,
        suffix_mean_nll=2.0,
        oracle_suffix_mean_nll=2.0,
        retrieval_entropy=1.0,
        retrieval_margin=0.1,
        final_entropy=0.8,
        final_margin=0.2,
    ).as_record()
    assert ebt_row["format"] == EBT_ROW_FORMAT
    assert ebt_row["model_forward_equivalent_prefix_tokens"] == 92
    assert ebt_row["active_lora_edge_evaluations"] == 276
    ebt_ledger = tmp_path / "ebt.jsonl"
    ebt_ledger.write_bytes(canonical_json_bytes(ebt_row))
    ebt_expected = (
        ("pear", story_id, "canonical_full_centroid", "compact", 4),
    )
    assert validate_ebt_ledger(
        ebt_ledger,
        str(ebt_contract["contract_sha256"]),
        ebt_expected,
        require_complete=True,
    ) == set(ebt_expected)

    ledger.write_bytes(canonical_json_bytes(second_record) + complete_payload)
    with pytest.raises(ValueError, match="canonical resumable prefix"):
        validate_retrieval_ledger(
            ledger,
            contract_sha,
            (
                ("pear", story_id, "canonical_full_centroid"),
                ("pear", second_story_id, "canonical_full_centroid"),
            ),
            require_complete=True,
        )

    timing_core = {
        "batch_size": 8,
        "candidate_width": 4,
        "cold_compile_seconds": 1.0,
        "ebt_contract_sha256": str(ebt_contract["contract_sha256"]),
        "format": TIMING_ROW_FORMAT,
        "mode": "compact",
        "physical_edge_capacity": 4,
        "prefix_width_bucket": 32,
        "warm_kernel_mean_seconds": 0.1,
        "warm_kernel_seconds": [0.1] * 5,
        "warm_throughput_examples_per_second": 80.0,
    }
    timing = {**timing_core, "result_sha256": record_sha256(timing_core)}
    timing_ledger = tmp_path / "timing.jsonl"
    timing_payload = canonical_json_bytes(timing)
    timing_ledger.write_bytes(timing_payload + b'{"interrupted"')
    repair_interrupted_tail(timing_ledger)
    assert timing_ledger.read_bytes() == timing_payload
    assert load_timing_ledger(
        timing_ledger,
        str(ebt_contract["contract_sha256"]),
    ) == (timing,)
    changed_core = {**timing_core, "unexpected": True}
    changed = {**changed_core, "result_sha256": record_sha256(changed_core)}
    timing_ledger.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="timing row identity"):
        load_timing_ledger(
            timing_ledger,
            str(ebt_contract["contract_sha256"]),
        )
