from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import apm.continual.tinyworlds_pilot_run as pilot_module
from apm.continual.language_adaptation_artifact import (
    LANGUAGE_ADAPTATION_FORMAT,
    LANGUAGE_ADAPTATION_MANIFEST,
    LANGUAGE_ADAPTATION_TENSORS,
    LanguageAdaptationArtifact,
    extract_language_adaptation_artifact,
    flatten_lora_edge,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
    unflatten_lora_edge,
)
from apm.continual.language_baseline_training import (
    train_language_adaptation_baselines,
)
from apm.continual.language_tasks import (
    AddressBook,
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig
from apm.memory.graph import MemoryGraph, NodeId


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=10,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _training_batch(tokens: tuple[int, ...]) -> TokenBatch:
    return TokenBatch(
        input_ids=np.asarray((tokens[:-1],), dtype=np.int32),
        attention_mask=np.ones((1, len(tokens) - 1), dtype=np.bool_),
        target_ids=np.asarray((tokens[1:],), dtype=np.int32),
        loss_mask=np.ones((1, len(tokens) - 1), dtype=np.bool_),
    )


def _router_rows(batch: RouterBatch, count: int) -> RouterBatch:
    return RouterBatch(
        input_ids=np.repeat(batch.input_ids, count, axis=0),
        attention_mask=np.repeat(batch.attention_mask, count, axis=0),
        target_ids=np.repeat(batch.target_ids, count, axis=0),
        loss_mask=np.repeat(batch.loss_mask, count, axis=0),
    )


def _language_task(task_id: str, tokens: tuple[int, ...]) -> LanguageTask:
    router_batch, competence_batch = build_prefix_suffix_batches(
        tokens + (tokens[-1],),
        prefix_length=3,
        suffix_length=2,
    )
    typed_task_id = TaskId(task_id)
    example = LanguageEvaluationExample(
        router_batch=router_batch,
        competence_batch=competence_batch,
        task_id=typed_task_id,
        oracle_node_id=NodeId(str(typed_task_id)),
    )
    return LanguageTask(
        task_id=typed_task_id,
        train_batches=(_training_batch(tokens),),
        validation_examples=(example,),
        test_examples=(example,),
    )


@pytest.fixture(scope="module")
def adaptation_artifact() -> LanguageAdaptationArtifact:
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    tasks = (
        _language_task("task-a", (1, 2, 3, 4, 5)),
        _language_task("task-b", (5, 4, 3, 2, 1)),
    )
    curriculum = LanguageCurriculum(
        tasks=tasks,
        max_nodes=3,
        max_edges=2,
    )
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=1,
        batch_size=1,
        weight_decay=0.0,
    )
    base_checkpoint = BaseCheckpointRef(
        directory=Path("frozen/base-checkpoint"),
        manifest_sha256="1" * 64,
        parameter_checksum=parameter_checksum(base_params, model_config),
    )
    adaptations = train_language_adaptation_baselines(
        curriculum,
        (_router_rows(tasks[0].validation_examples[0].router_batch, 2),),
        base_checkpoint,
        base_params,
        model_config,
        lora_config,
        train_config,
        jax.random.PRNGKey(7),
    )
    return extract_language_adaptation_artifact(
        adaptations,
        model_config,
        lora_config,
        config_hashes={"data": "2" * 64, "benchmark": "3" * 64},
    )


def _assert_edges_equal(left, right, model_config, lora_config) -> None:
    left_tensors = flatten_lora_edge(left, model_config, lora_config)
    right_tensors = flatten_lora_edge(right, model_config, lora_config)
    assert tuple(left_tensors) == tuple(right_tensors)
    for name in left_tensors:
        np.testing.assert_array_equal(left_tensors[name], right_tensors[name])


def test_lora_flattening_is_canonical_and_strict() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=2.0)
    edge = init_lora_edge(jax.random.PRNGKey(4), model_config, lora_config)
    tensors = flatten_lora_edge(edge, model_config, lora_config)

    assert tuple(tensors) == tuple(sorted(tensors))
    assert tuple(tensors) == (
        "blocks.0.attention_output.left",
        "blocks.0.attention_output.right",
        "blocks.0.key.left",
        "blocks.0.key.right",
        "blocks.0.mlp_input.left",
        "blocks.0.mlp_input.right",
        "blocks.0.mlp_output.left",
        "blocks.0.mlp_output.right",
        "blocks.0.query.left",
        "blocks.0.query.right",
        "blocks.0.value.left",
        "blocks.0.value.right",
    )
    _assert_edges_equal(
        edge,
        unflatten_lora_edge(tensors, model_config, lora_config),
        model_config,
        lora_config,
    )

    missing = dict(tensors)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="tensor names"):
        unflatten_lora_edge(missing, model_config, lora_config)
    wrong_dtype = dict(tensors)
    first_name = next(iter(wrong_dtype))
    wrong_dtype[first_name] = np.asarray(wrong_dtype[first_name], dtype=np.float16)
    with pytest.raises(TypeError, match="dtype"):
        unflatten_lora_edge(wrong_dtype, model_config, lora_config)


def test_extraction_preserves_all_training_state_and_hashes(
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    artifact = adaptation_artifact

    assert artifact.task_order == (TaskId("task-a"), TaskId("task-b"))
    assert tuple(name for name, _ in artifact.config_hashes) == (
        "benchmark",
        "data",
        "lora",
        "model",
        "training",
    )
    assert artifact.base_checkpoint.directory == Path("frozen/base-checkpoint")
    assert len(artifact.sequential_stages) == 2
    assert len(artifact.independent_adapters) == 2
    assert len(artifact.vamp_graph.nodes) == 3
    assert artifact.vamp_graph.nodes[1].incoming_edge is not None
    assert artifact.vamp_stages[0].parent_node_id == NodeId("root")
    assert artifact.vamp_stages[0].parent_mean_node_nll[1] == float("inf")
    assert len(artifact.tensor_checksums) > 70
    assert len(artifact.tensor_checksum) == 64
    assert not artifact.address_book.keys.flags.writeable
    assert not artifact.rng_state.vamp.flags.writeable


def test_pilot_config_hashes_match_the_shared_artifact_contract(
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    inputs = SimpleNamespace(
        base_artifact=SimpleNamespace(
            checkpoint=SimpleNamespace(config=adaptation_artifact.model_config)
        ),
        lora_config=adaptation_artifact.lora_config,
        train_config=adaptation_artifact.train_config,
        profile_sha256="4" * 64,
        rendered=SimpleNamespace(bundle_id="rendered-test-bundle"),
    )

    rebuilt = replace(
        adaptation_artifact,
        config_hashes=pilot_module._adaptation_config_hashes(inputs),
    )

    assert dict(rebuilt.config_hashes)["model"] == dict(
        adaptation_artifact.config_hashes
    )["model"]
    assert dict(rebuilt.config_hashes)["lora"] == dict(
        adaptation_artifact.config_hashes
    )["lora"]
    assert dict(rebuilt.config_hashes)["training"] == dict(
        adaptation_artifact.config_hashes
    )["training"]


def test_adaptation_artifact_round_trip_is_exact_and_content_checked(
    tmp_path: Path,
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    output_directory = save_language_adaptation_artifact(
        tmp_path / "adaptation",
        adaptation_artifact,
    )
    loaded = load_language_adaptation_artifact(output_directory)

    assert {path.name for path in output_directory.iterdir()} == {
        LANGUAGE_ADAPTATION_MANIFEST,
        LANGUAGE_ADAPTATION_TENSORS,
    }
    assert loaded.tensor_checksum == adaptation_artifact.tensor_checksum
    assert loaded.tensor_checksums == adaptation_artifact.tensor_checksums
    assert loaded.config_hashes == adaptation_artifact.config_hashes
    assert loaded.task_order == adaptation_artifact.task_order
    assert loaded.base_checkpoint == adaptation_artifact.base_checkpoint
    assert loaded.vamp_graph.nodes[1].parent_id == NodeId("root")
    np.testing.assert_array_equal(loaded.address_book.keys, adaptation_artifact.address_book.keys)
    np.testing.assert_array_equal(loaded.rng_state.vamp, adaptation_artifact.rng_state.vamp)
    _assert_edges_equal(
        loaded.sequential_stages[0].adapter,
        adaptation_artifact.sequential_stages[0].adapter,
        loaded.model_config,
        loaded.lora_config,
    )
    _assert_edges_equal(
        loaded.vamp_graph.nodes[1].incoming_edge,
        adaptation_artifact.vamp_graph.nodes[1].incoming_edge,
        loaded.model_config,
        loaded.lora_config,
    )

    manifest = json.loads(
        (output_directory / LANGUAGE_ADAPTATION_MANIFEST).read_text(encoding="utf-8")
    )
    assert manifest["format"] == LANGUAGE_ADAPTATION_FORMAT
    assert "base_checkpoint" in manifest
    assert "base_params" not in manifest
    tensor_contents = (output_directory / LANGUAGE_ADAPTATION_TENSORS).read_bytes()
    header_length = struct.unpack("<Q", tensor_contents[:8])[0]
    header = json.loads(tensor_contents[8 : 8 + header_length].decode("utf-8"))
    assert header["__metadata__"]["format"] == LANGUAGE_ADAPTATION_FORMAT
    assert header["__metadata__"]["tensor_checksum"] == loaded.tensor_checksum
    assert not any("token_embedding" in name for name in header)

    second_directory = save_language_adaptation_artifact(
        tmp_path / "adaptation-copy",
        adaptation_artifact,
    )
    assert {
        path.name: path.read_bytes() for path in second_directory.iterdir()
    } == {
        path.name: path.read_bytes() for path in output_directory.iterdir()
    }


def test_partial_stage_round_trip_preserves_fixed_pilot_capacity(
    tmp_path: Path,
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    artifact = adaptation_artifact
    address_book = AddressBook(
        node_ids=artifact.address_book.node_ids[:2] + (None,) * 7,
        keys=np.concatenate(
            (
                artifact.address_book.keys[:2],
                np.zeros((7, artifact.address_book.key_dim), dtype=np.float32),
            ),
            axis=0,
        ),
        valid_node_mask=np.concatenate(
            (
                artifact.address_book.valid_node_mask[:2],
                np.zeros((7,), dtype=np.bool_),
            )
        ),
    )
    first_vamp_stage = replace(
        artifact.vamp_stages[0],
        parent_mean_node_nll=(
            artifact.vamp_stages[0].parent_mean_node_nll + (float("inf"),) * 6
        ),
    )
    partial = replace(
        artifact,
        task_order=artifact.task_order[:1],
        sequential_stages=artifact.sequential_stages[:1],
        independent_adapters=artifact.independent_adapters[:1],
        vamp_graph=MemoryGraph(artifact.vamp_graph.nodes[:2]),
        address_book=address_book,
        vamp_stages=(first_vamp_stage,),
        max_nodes=9,
        max_edges=8,
    )

    directory = save_language_adaptation_artifact(
        tmp_path / "partial-stage-fixed-capacity",
        partial,
    )
    loaded = load_language_adaptation_artifact(directory)

    assert loaded.task_order == artifact.task_order[:1]
    assert len(loaded.vamp_graph.nodes) == 2
    assert loaded.max_nodes == 9
    assert loaded.max_edges == 8
    assert loaded.address_book.keys.shape == (9, artifact.address_book.key_dim)
    assert loaded.vamp_stages[0].parent_mean_node_nll[2:] == (
        float("inf"),
    ) * 7


def test_artifact_rejects_file_manifest_and_directory_tampering(
    tmp_path: Path,
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    tensor_directory = save_language_adaptation_artifact(
        tmp_path / "tensor-tamper",
        adaptation_artifact,
    )
    tensor_path = tensor_directory / LANGUAGE_ADAPTATION_TENSORS
    tensor_contents = bytearray(tensor_path.read_bytes())
    tensor_contents[-1] ^= 1
    tensor_path.write_bytes(tensor_contents)
    with pytest.raises(ValueError, match="tensor file hash"):
        load_language_adaptation_artifact(tensor_directory)

    manifest_directory = save_language_adaptation_artifact(
        tmp_path / "manifest-tamper",
        adaptation_artifact,
    )
    manifest_path = manifest_directory / LANGUAGE_ADAPTATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_checkpoint"]["directory"] = "different/base"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload hash"):
        load_language_adaptation_artifact(manifest_directory)

    unknown_field_directory = save_language_adaptation_artifact(
        tmp_path / "unknown-field",
        adaptation_artifact,
    )
    unknown_manifest_path = unknown_field_directory / LANGUAGE_ADAPTATION_MANIFEST
    unknown_manifest = json.loads(
        unknown_manifest_path.read_text(encoding="utf-8")
    )
    unknown_manifest["unexpected"] = True
    unknown_manifest_path.write_text(
        json.dumps(unknown_manifest, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields do not match"):
        load_language_adaptation_artifact(unknown_field_directory)

    extra_directory = save_language_adaptation_artifact(
        tmp_path / "extra-entry",
        adaptation_artifact,
    )
    (extra_directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="entries are not canonical"):
        load_language_adaptation_artifact(extra_directory)


def test_atomic_save_refuses_to_replace_existing_directory(
    tmp_path: Path,
    adaptation_artifact: LanguageAdaptationArtifact,
) -> None:
    target = tmp_path / "owned"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_language_adaptation_artifact(target, adaptation_artifact)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".owned.tmp-*"))
