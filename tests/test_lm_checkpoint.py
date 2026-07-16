from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.checkpoint import (
    BaseCheckpointRef,
    CheckpointFileHash,
    CheckpointProvenance,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    flatten_gpt_neo_params,
    load_gpt_neo_checkpoint,
    parameter_checksum,
    save_gpt_neo_checkpoint,
    unflatten_gpt_neo_params,
)
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params


def _config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=17,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        num_heads=2,
        attention_types=("global", "local"),
        local_window_size=3,
    )


def _params() -> GptNeoParams:
    return init_gpt_neo_params(jax.random.PRNGKey(42), _config())


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _tokenizer_metadata() -> TokenizerCheckpointMetadata:
    return TokenizerCheckpointMetadata(
        kind="character",
        identifier="tinyshakespeare-char-v1",
        revision="6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e",
        files=(
            CheckpointFileHash("vocabulary.json", _sha256(b"vocabulary")),
            CheckpointFileHash("tokenizer.json", _sha256(b"tokenizer")),
        ),
    )


def _source_metadata() -> SourceCheckpointMetadata:
    return SourceCheckpointMetadata(
        identifier="karpathy/char-rnn",
        revision="6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e",
        sha256=_sha256(b"tinyshakespeare-source"),
    )


def _provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        producer="test-checkpoint-writer",
        producer_version="1.2.3",
        library_versions=(("numpy", np.__version__), ("jax", jax.__version__)),
        environment=(("python", "3.11.9"), ("platform", "test-platform")),
    )


def _save(directory: Path) -> BaseCheckpointRef:
    return save_gpt_neo_checkpoint(
        directory,
        _params(),
        _config(),
        tokenizer=_tokenizer_metadata(),
        source=_source_metadata(),
    )


def _assert_params_equal(left: GptNeoParams, right: GptNeoParams) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_schema_v1_checkpoint_round_trip_is_exact_and_content_addressed(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    params = _params()

    reference = save_gpt_neo_checkpoint(
        checkpoint_directory,
        params,
        _config(),
        tokenizer=_tokenizer_metadata(),
        source=_source_metadata(),
    )
    loaded = load_gpt_neo_checkpoint(reference)

    assert reference.directory == checkpoint_directory.resolve()
    assert loaded.reference == reference
    assert loaded.config == _config()
    assert loaded.tokenizer == _tokenizer_metadata()
    assert loaded.source == _source_metadata()
    assert loaded.provenance.producer == "apm.checkpoint"
    assert dict(loaded.provenance.library_versions)["jax"] == jax.__version__
    assert reference.parameter_checksum == parameter_checksum(params, _config())
    assert (checkpoint_directory / "model.safetensors").is_file()
    assert (checkpoint_directory / "manifest.json").is_file()
    _assert_params_equal(loaded.params, params)


def test_safetensors_file_has_a_standard_header_and_contiguous_offsets(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    _save(checkpoint_directory)
    contents = (checkpoint_directory / "model.safetensors").read_bytes()
    header_length = struct.unpack("<Q", contents[:8])[0]
    header = json.loads(contents[8 : 8 + header_length].decode("utf-8"))
    tensor_names = sorted(name for name in header if name != "__metadata__")

    assert contents[8:9] == b"{"
    assert header_length % 8 == 0
    assert header["__metadata__"]["format"] == "apm-gpt-neo"
    assert header["__metadata__"]["schema_version"] == "1"
    assert tensor_names == sorted(flatten_gpt_neo_params(_params()))
    expected_offset = 0
    for name in tensor_names:
        tensor = header[name]
        assert tensor["dtype"] == "F32"
        assert tensor["data_offsets"][0] == expected_offset
        expected_offset = tensor["data_offsets"][1]
    assert 8 + header_length + expected_offset == len(contents)


def test_manifest_records_config_tokenizer_source_provenance_and_tensors(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    reference = save_gpt_neo_checkpoint(
        checkpoint_directory,
        _params(),
        _config(),
        tokenizer=_tokenizer_metadata(),
        source=_source_metadata(),
        provenance=_provenance(),
    )
    manifest_bytes = (checkpoint_directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)

    assert manifest["schema_version"] == 1
    assert manifest["format"] == "apm-gpt-neo"
    assert manifest["parameter_checksum"] == reference.parameter_checksum
    assert manifest["config"]["attention_types"] == ["global", "local"]
    assert manifest["source"]["sha256"] == _source_metadata().sha256
    assert manifest["provenance"]["producer"] == _provenance().producer
    assert manifest["provenance"]["library_versions"] == dict(
        _provenance().library_versions
    )
    tokenizer_hashes = {
        value["name"]: value["sha256"]
        for value in manifest["tokenizer"]["files"]
    }
    assert tokenizer_hashes == {
        value.name: value.sha256
        for value in _tokenizer_metadata().files
    }
    assert set(manifest["tensors"]) == set(flatten_gpt_neo_params(_params()))
    assert hashlib.sha256(manifest_bytes).hexdigest() == reference.manifest_sha256


def test_canonical_flatten_and_strict_unflatten_round_trip() -> None:
    params = _params()
    flattened = flatten_gpt_neo_params(params)

    assert tuple(flattened) == tuple(sorted(flattened))
    assert "blocks.0.attention.query.kernel" in flattened
    assert "blocks.0.attention.query.bias" not in flattened
    assert "blocks.0.attention.output.bias" in flattened
    assert "blocks.1.mlp.output_projection.kernel" in flattened
    assert len(flattened) == 4 + 13 * _config().num_layers
    _assert_params_equal(
        unflatten_gpt_neo_params(flattened, _config()),
        params,
    )


@pytest.mark.parametrize("defect", ("missing", "unexpected", "shape", "dtype"))
def test_unflatten_rejects_every_schema_mismatch(defect: str) -> None:
    tensors: dict[str, jax.Array | np.ndarray] = flatten_gpt_neo_params(_params())
    first_name = next(iter(tensors))
    if defect == "missing":
        tensors.pop(first_name)
    elif defect == "unexpected":
        tensors["unexpected.tensor"] = jnp.zeros((1,), dtype=jnp.float32)
    elif defect == "shape":
        tensors[first_name] = jnp.ravel(tensors[first_name])
    else:
        tensors[first_name] = np.asarray(tensors[first_name], dtype=np.float16)

    expected_message = defect if defect in ("shape", "dtype") else "tensor names"
    with pytest.raises((TypeError, ValueError), match=expected_message):
        unflatten_gpt_neo_params(tensors, _config())


def test_model_file_tampering_is_rejected_before_deserialization(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    _save(checkpoint_directory)
    model_path = checkpoint_directory / "model.safetensors"
    contents = bytearray(model_path.read_bytes())
    contents[-1] ^= 0x01
    model_path.write_bytes(contents)

    with pytest.raises(ValueError, match="model file hash"):
        load_gpt_neo_checkpoint(checkpoint_directory)


def test_reference_detects_manifest_tampering(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    reference = _save(checkpoint_directory)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["identifier"] = "tampered/source"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest hash"):
        load_gpt_neo_checkpoint(reference)


def test_path_loading_rejects_internally_inconsistent_manifest_metadata(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    _save(checkpoint_directory)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["identifier"] = "tampered-tokenizer"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer hash"):
        load_gpt_neo_checkpoint(checkpoint_directory)


def test_path_loading_rejects_inconsistent_provenance_hash(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    _save(checkpoint_directory)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["producer"] = "tampered-producer"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance hash"):
        load_gpt_neo_checkpoint(checkpoint_directory)


def test_save_refuses_to_replace_an_existing_directory_atomically(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "base"
    checkpoint_directory.mkdir()
    marker = checkpoint_directory / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _save(checkpoint_directory)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".base.tmp-*"))


@pytest.mark.parametrize(
    "constructor",
    (
        lambda: CheckpointFileHash("file", "not-a-hash"),
        lambda: SourceCheckpointMetadata("source", "revision", "0" * 63),
        lambda: TokenizerCheckpointMetadata("char", "id", "rev", ()),
        lambda: CheckpointProvenance("producer", "1", (), (("python", "3"),)),
        lambda: CheckpointProvenance(
            "producer",
            "1",
            (("numpy", "1"), ("numpy", "2")),
            (("python", "3"),),
        ),
    ),
)
def test_checkpoint_metadata_rejects_missing_or_invalid_hashes(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()
