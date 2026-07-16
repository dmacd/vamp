"""Strict schema-v1 checkpoints for immutable GPT-Neo base models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.parameters import (
    AttentionParams,
    GptNeoParams,
    LayerNormParams,
    LinearParams,
    MlpParams,
    TransformerBlockParams,
)


SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "apm-gpt-neo"
MODEL_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000

_NUMPY_TO_SAFETENSORS = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("int16"): "I16",
    np.dtype("int8"): "I8",
    np.dtype("uint64"): "U64",
    np.dtype("uint32"): "U32",
    np.dtype("uint16"): "U16",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}
_SAFETENSORS_TO_NUMPY = {
    code: dtype.newbyteorder("<")
    for dtype, code in _NUMPY_TO_SAFETENSORS.items()
}


@dataclass(frozen=True)
class CheckpointFileHash:
    """One immutable filename and SHA-256 digest pair."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("checkpoint file hash name must not be empty")
        _validate_sha256(self.sha256, f"hash for {self.name}")


@dataclass(frozen=True)
class TokenizerCheckpointMetadata:
    """Immutable tokenizer identity and hashes of its canonical artifacts."""

    kind: str
    identifier: str
    revision: str
    files: tuple[CheckpointFileHash, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.kind, self.identifier, self.revision)
        ):
            raise ValueError("tokenizer kind, identifier, and revision must not be empty")
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("tokenizer metadata must include at least one hashed artifact")
        if any(not isinstance(value, CheckpointFileHash) for value in self.files):
            raise TypeError("tokenizer files must contain CheckpointFileHash values")
        names = tuple(file_hash.name for file_hash in self.files)
        if len(set(names)) != len(names):
            raise ValueError("tokenizer artifact names must be unique")
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda value: value.name)))


@dataclass(frozen=True)
class SourceCheckpointMetadata:
    """Immutable source identity, revision, and source-content digest."""

    identifier: str
    revision: str
    sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.identifier, self.revision)
        ):
            raise ValueError("source identifier and revision must not be empty")
        _validate_sha256(self.sha256, "source checkpoint hash")


@dataclass(frozen=True)
class CheckpointProvenance:
    """Immutable producer, library, and environment identity for an artifact."""

    producer: str
    producer_version: str
    library_versions: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.producer, self.producer_version)
        ):
            raise ValueError("provenance producer and version must not be empty")
        object.__setattr__(
            self,
            "library_versions",
            _canonical_provenance_pairs(self.library_versions, "library versions"),
        )
        object.__setattr__(
            self,
            "environment",
            _canonical_provenance_pairs(self.environment, "environment"),
        )


@dataclass(frozen=True)
class BaseCheckpointRef:
    """Content-addressed reference to one published immutable base checkpoint."""

    directory: Path
    manifest_sha256: str
    parameter_checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        _validate_sha256(self.manifest_sha256, "manifest hash")
        _validate_sha256(self.parameter_checksum, "parameter checksum")


@dataclass(frozen=True)
class LoadedGptNeoCheckpoint:
    """Typed values recovered from a validated schema-v1 checkpoint."""

    reference: BaseCheckpointRef
    config: GptNeoConfig
    params: GptNeoParams
    tokenizer: TokenizerCheckpointMetadata
    source: SourceCheckpointMetadata
    provenance: CheckpointProvenance


@dataclass(frozen=True)
class _TensorSpec:
    name: str
    shape: tuple[int, ...]


def flatten_gpt_neo_params(params: GptNeoParams) -> dict[str, jax.Array]:
    """Flatten typed GPT-Neo parameters under stable canonical tensor names."""
    tensors: dict[str, jax.Array] = {
        "token_embedding": params.token_embedding,
        "position_embedding": params.position_embedding,
    }
    for layer_index, block in enumerate(params.blocks):
        prefix = f"blocks.{layer_index}"
        tensors.update(
            {
                f"{prefix}.attention_norm.scale": block.attention_norm.scale,
                f"{prefix}.attention_norm.bias": block.attention_norm.bias,
                f"{prefix}.attention.query.kernel": block.attention.query.kernel,
                f"{prefix}.attention.key.kernel": block.attention.key.kernel,
                f"{prefix}.attention.value.kernel": block.attention.value.kernel,
                f"{prefix}.attention.output.kernel": block.attention.output.kernel,
                f"{prefix}.mlp_norm.scale": block.mlp_norm.scale,
                f"{prefix}.mlp_norm.bias": block.mlp_norm.bias,
                f"{prefix}.mlp.input_projection.kernel": block.mlp.input_projection.kernel,
                f"{prefix}.mlp.output_projection.kernel": block.mlp.output_projection.kernel,
            }
        )
        optional_tensors = (
            (f"{prefix}.attention.query.bias", block.attention.query.bias),
            (f"{prefix}.attention.key.bias", block.attention.key.bias),
            (f"{prefix}.attention.value.bias", block.attention.value.bias),
            (f"{prefix}.attention.output.bias", block.attention.output.bias),
            (f"{prefix}.mlp.input_projection.bias", block.mlp.input_projection.bias),
            (f"{prefix}.mlp.output_projection.bias", block.mlp.output_projection.bias),
        )
        tensors.update(
            {
                name: tensor
                for name, tensor in optional_tensors
                if tensor is not None
            }
        )
    tensors.update(
        {
            "final_norm.scale": params.final_norm.scale,
            "final_norm.bias": params.final_norm.bias,
        }
    )
    return dict(sorted(tensors.items()))


def unflatten_gpt_neo_params(
    tensors: Mapping[str, jax.Array | np.ndarray],
    config: GptNeoConfig,
) -> GptNeoParams:
    """Strictly reconstruct a float32 GPT-Neo parameter tree from canonical tensors."""
    arrays = _validated_parameter_arrays(tensors, config)

    def linear(prefix: str, biased: bool) -> LinearParams:
        return LinearParams(
            kernel=jnp.asarray(arrays[f"{prefix}.kernel"]),
            bias=jnp.asarray(arrays[f"{prefix}.bias"]) if biased else None,
        )

    blocks = tuple(
        TransformerBlockParams(
            attention_norm=LayerNormParams(
                scale=jnp.asarray(arrays[f"blocks.{layer_index}.attention_norm.scale"]),
                bias=jnp.asarray(arrays[f"blocks.{layer_index}.attention_norm.bias"]),
            ),
            attention=AttentionParams(
                query=linear(f"blocks.{layer_index}.attention.query", False),
                key=linear(f"blocks.{layer_index}.attention.key", False),
                value=linear(f"blocks.{layer_index}.attention.value", False),
                output=linear(f"blocks.{layer_index}.attention.output", True),
            ),
            mlp_norm=LayerNormParams(
                scale=jnp.asarray(arrays[f"blocks.{layer_index}.mlp_norm.scale"]),
                bias=jnp.asarray(arrays[f"blocks.{layer_index}.mlp_norm.bias"]),
            ),
            mlp=MlpParams(
                input_projection=linear(
                    f"blocks.{layer_index}.mlp.input_projection",
                    True,
                ),
                output_projection=linear(
                    f"blocks.{layer_index}.mlp.output_projection",
                    True,
                ),
            ),
        )
        for layer_index in range(config.num_layers)
    )
    return GptNeoParams(
        token_embedding=jnp.asarray(arrays["token_embedding"]),
        position_embedding=jnp.asarray(arrays["position_embedding"]),
        blocks=blocks,
        final_norm=LayerNormParams(
            scale=jnp.asarray(arrays["final_norm.scale"]),
            bias=jnp.asarray(arrays["final_norm.bias"]),
        ),
    )


def parameter_checksum(params: GptNeoParams, config: GptNeoConfig) -> str:
    """Return a deterministic SHA-256 digest over canonical float32 parameters."""
    arrays = _validated_parameter_arrays(flatten_gpt_neo_params(params), config)
    return _parameter_checksum_from_arrays(arrays)


def save_gpt_neo_checkpoint(
    directory: Path,
    params: GptNeoParams,
    config: GptNeoConfig,
    *,
    tokenizer: TokenizerCheckpointMetadata,
    source: SourceCheckpointMetadata,
    provenance: CheckpointProvenance | None = None,
) -> BaseCheckpointRef:
    """Atomically publish a strict schema-v1 GPT-Neo checkpoint directory."""
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        checkpoint_provenance = provenance or _native_checkpoint_provenance()
        arrays = _validated_parameter_arrays(flatten_gpt_neo_params(params), config)
        params_checksum = _parameter_checksum_from_arrays(arrays)
        safetensors_metadata = {
            "format": CHECKPOINT_FORMAT,
            "schema_version": str(SCHEMA_VERSION),
            "parameter_checksum": params_checksum,
        }
        model_path = temporary / MODEL_FILENAME
        _write_safetensors(model_path, arrays, safetensors_metadata)
        config_payload = _config_to_payload(config)
        tokenizer_payload = _tokenizer_to_payload(tokenizer)
        provenance_payload = _provenance_to_payload(checkpoint_provenance)
        tensor_payload = {
            name: {
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "nbytes": int(array.nbytes),
            }
            for name, array in arrays.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": CHECKPOINT_FORMAT,
            "model_file": MODEL_FILENAME,
            "model_file_sha256": _file_sha256(model_path),
            "parameter_checksum": params_checksum,
            "config": config_payload,
            "config_sha256": _payload_sha256(config_payload),
            "tokenizer": tokenizer_payload,
            "tokenizer_sha256": _payload_sha256(tokenizer_payload),
            "source": _source_to_payload(source),
            "provenance": provenance_payload,
            "provenance_sha256": _payload_sha256(provenance_payload),
            "tensors": tensor_payload,
        }
        manifest_path = temporary / MANIFEST_FILENAME
        _write_file(manifest_path, _stable_json_bytes(manifest, newline=True))
        manifest_sha256 = _file_sha256(manifest_path)
        _load_checkpoint_directory(temporary, None)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return BaseCheckpointRef(
            directory=target.resolve(),
            manifest_sha256=manifest_sha256,
            parameter_checksum=params_checksum,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_gpt_neo_checkpoint(
    checkpoint: Path | BaseCheckpointRef,
) -> LoadedGptNeoCheckpoint:
    """Load a checkpoint only after its manifest, container, and tensors validate."""
    reference = checkpoint if isinstance(checkpoint, BaseCheckpointRef) else None
    directory = reference.directory if reference is not None else Path(checkpoint)
    return _load_checkpoint_directory(directory, reference)


def _load_checkpoint_directory(
    directory: Path,
    expected_reference: BaseCheckpointRef | None,
) -> LoadedGptNeoCheckpoint:
    checkpoint_directory = Path(directory)
    manifest_path = checkpoint_directory / MANIFEST_FILENAME
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        expected_reference is not None
        and manifest_sha256 != expected_reference.manifest_sha256
    ):
        raise ValueError("checkpoint manifest hash does not match its reference")
    manifest = _parse_manifest(manifest_bytes)
    config_payload = _expect_dict(manifest["config"], "config")
    tokenizer_payload = _expect_dict(manifest["tokenizer"], "tokenizer")
    provenance_payload = _expect_dict(manifest["provenance"], "provenance")
    if _payload_sha256(config_payload) != manifest["config_sha256"]:
        raise ValueError("checkpoint config hash mismatch")
    if _payload_sha256(tokenizer_payload) != manifest["tokenizer_sha256"]:
        raise ValueError("checkpoint tokenizer hash mismatch")
    if _payload_sha256(provenance_payload) != manifest["provenance_sha256"]:
        raise ValueError("checkpoint provenance hash mismatch")
    config = _config_from_payload(config_payload)
    tokenizer = _tokenizer_from_payload(tokenizer_payload)
    source = _source_from_payload(_expect_dict(manifest["source"], "source"))
    provenance = _provenance_from_payload(provenance_payload)
    model_path = checkpoint_directory / str(manifest["model_file"])
    if _file_sha256(model_path) != manifest["model_file_sha256"]:
        raise ValueError("checkpoint model file hash mismatch")
    tensors, safetensors_metadata = _read_safetensors(model_path)
    expected_safetensors_metadata = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": str(SCHEMA_VERSION),
        "parameter_checksum": str(manifest["parameter_checksum"]),
    }
    if safetensors_metadata != expected_safetensors_metadata:
        raise ValueError("safetensors metadata does not match the manifest")
    arrays = _validated_parameter_arrays(tensors, config)
    _validate_manifest_tensors(
        _expect_dict(manifest["tensors"], "tensors"),
        arrays,
    )
    params_checksum = _parameter_checksum_from_arrays(arrays)
    if params_checksum != manifest["parameter_checksum"]:
        raise ValueError("checkpoint parameter checksum mismatch")
    if (
        expected_reference is not None
        and params_checksum != expected_reference.parameter_checksum
    ):
        raise ValueError("checkpoint parameters do not match their reference")
    reference = BaseCheckpointRef(
        directory=checkpoint_directory.resolve(),
        manifest_sha256=manifest_sha256,
        parameter_checksum=params_checksum,
    )
    return LoadedGptNeoCheckpoint(
        reference=reference,
        config=config,
        params=unflatten_gpt_neo_params(arrays, config),
        tokenizer=tokenizer,
        source=source,
        provenance=provenance,
    )


def _parameter_specs(config: GptNeoConfig) -> tuple[_TensorSpec, ...]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    specifications = [
        _TensorSpec("token_embedding", (config.vocab_size, hidden)),
        _TensorSpec(
            "position_embedding",
            (config.max_position_embeddings, hidden),
        ),
    ]
    for layer_index in range(config.num_layers):
        prefix = f"blocks.{layer_index}"
        specifications.extend(
            (
                _TensorSpec(f"{prefix}.attention_norm.scale", (hidden,)),
                _TensorSpec(f"{prefix}.attention_norm.bias", (hidden,)),
                _TensorSpec(f"{prefix}.attention.query.kernel", (hidden, hidden)),
                _TensorSpec(f"{prefix}.attention.key.kernel", (hidden, hidden)),
                _TensorSpec(f"{prefix}.attention.value.kernel", (hidden, hidden)),
                _TensorSpec(f"{prefix}.attention.output.kernel", (hidden, hidden)),
                _TensorSpec(f"{prefix}.attention.output.bias", (hidden,)),
                _TensorSpec(f"{prefix}.mlp_norm.scale", (hidden,)),
                _TensorSpec(f"{prefix}.mlp_norm.bias", (hidden,)),
                _TensorSpec(
                    f"{prefix}.mlp.input_projection.kernel",
                    (hidden, intermediate),
                ),
                _TensorSpec(
                    f"{prefix}.mlp.input_projection.bias",
                    (intermediate,),
                ),
                _TensorSpec(
                    f"{prefix}.mlp.output_projection.kernel",
                    (intermediate, hidden),
                ),
                _TensorSpec(f"{prefix}.mlp.output_projection.bias", (hidden,)),
            )
        )
    specifications.extend(
        (
            _TensorSpec("final_norm.scale", (hidden,)),
            _TensorSpec("final_norm.bias", (hidden,)),
        )
    )
    return tuple(specifications)


def _validated_parameter_arrays(
    tensors: Mapping[str, jax.Array | np.ndarray],
    config: GptNeoConfig,
) -> dict[str, np.ndarray]:
    specifications = _parameter_specs(config)
    expected_names = {specification.name for specification in specifications}
    actual_names = set(tensors)
    missing = tuple(sorted(expected_names - actual_names))
    unexpected = tuple(sorted(actual_names - expected_names))
    if missing or unexpected:
        raise ValueError(
            f"parameter tensor names do not match; missing={missing}, unexpected={unexpected}"
        )
    arrays: dict[str, np.ndarray] = {}
    for specification in specifications:
        array = np.asarray(tensors[specification.name])
        if array.shape != specification.shape:
            raise ValueError(
                f"parameter tensor {specification.name} has shape {array.shape}; "
                f"expected {specification.shape}"
            )
        if array.dtype != np.dtype("float32"):
            raise TypeError(
                f"parameter tensor {specification.name} has dtype {array.dtype}; expected float32"
            )
        arrays[specification.name] = np.ascontiguousarray(array)
    return dict(sorted(arrays.items()))


def _parameter_checksum_from_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = _canonical_safetensors_array(arrays[name])
        descriptor = _stable_json_bytes(
            {
                "name": name,
                "dtype": array.dtype.name,
                "shape": list(array.shape),
            }
        )
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_safetensors(
    path: Path,
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, str],
) -> None:
    arrays = {
        name: _canonical_safetensors_array(array)
        for name, array in sorted(tensors.items())
    }
    if "__metadata__" in arrays:
        raise ValueError("__metadata__ is reserved by the safetensors format")
    data_offset = 0
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    for name, array in arrays.items():
        data_end = data_offset + array.nbytes
        header[name] = {
            "dtype": _NUMPY_TO_SAFETENSORS[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [data_offset, data_end],
        }
        data_offset = data_end
    raw_header = _stable_json_bytes(header)
    padded_header = raw_header + b" " * (-len(raw_header) % 8)
    contents = (
        struct.pack("<Q", len(padded_header))
        + padded_header
        + b"".join(array.tobytes(order="C") for array in arrays.values())
    )
    _write_file(path, contents)


def _read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    contents = path.read_bytes()
    if len(contents) < 10:
        raise ValueError("safetensors file is too short")
    header_length = struct.unpack("<Q", contents[:8])[0]
    if (
        header_length < 2
        or header_length > _MAX_SAFETENSORS_HEADER_BYTES
        or 8 + header_length > len(contents)
    ):
        raise ValueError("safetensors header length is invalid")
    raw_header = contents[8 : 8 + header_length]
    try:
        header_value = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("safetensors header is not valid JSON") from error
    header = _expect_dict(header_value, "safetensors header")
    metadata_value = header.pop("__metadata__", {})
    metadata_payload = _expect_dict(metadata_value, "safetensors metadata")
    if any(not isinstance(value, str) for value in metadata_payload.values()):
        raise ValueError("safetensors metadata values must be strings")
    metadata = {name: str(value) for name, value in metadata_payload.items()}
    data = memoryview(contents)[8 + header_length :]
    parsed: list[tuple[int, int, str, np.dtype, tuple[int, ...]]] = []
    for name, tensor_value in header.items():
        tensor = _expect_dict(tensor_value, f"safetensors tensor {name}")
        _require_exact_keys(tensor, {"dtype", "shape", "data_offsets"}, name)
        dtype_code = tensor["dtype"]
        if not isinstance(dtype_code, str) or dtype_code not in _SAFETENSORS_TO_NUMPY:
            raise ValueError(f"unsupported safetensors dtype for {name}: {dtype_code}")
        shape = _parse_nonnegative_int_list(tensor["shape"], f"shape for {name}")
        offsets = _parse_nonnegative_int_list(
            tensor["data_offsets"],
            f"data offsets for {name}",
        )
        if len(offsets) != 2 or offsets[1] < offsets[0]:
            raise ValueError(f"invalid safetensors data offsets for {name}")
        dtype = _SAFETENSORS_TO_NUMPY[dtype_code]
        element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if offsets[1] - offsets[0] != element_count * dtype.itemsize:
            raise ValueError(f"safetensors byte count does not match shape for {name}")
        parsed.append((offsets[0], offsets[1], name, dtype, shape))
    parsed.sort()
    expected_offset = 0
    arrays: dict[str, np.ndarray] = {}
    for start, end, name, dtype, shape in parsed:
        if start != expected_offset or end > len(data):
            raise ValueError("safetensors tensor data must be contiguous and in bounds")
        arrays[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(shape).copy()
        expected_offset = end
    if expected_offset != len(data):
        raise ValueError("safetensors file contains unindexed tensor data")
    return dict(sorted(arrays.items())), dict(sorted(metadata.items()))


def _canonical_safetensors_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    native_dtype = value.dtype.newbyteorder("=")
    if native_dtype not in _NUMPY_TO_SAFETENSORS:
        raise TypeError(f"unsupported safetensors dtype: {value.dtype}")
    little_dtype = native_dtype.newbyteorder("<")
    return np.ascontiguousarray(value.astype(little_dtype, copy=False))


def _config_to_payload(config: GptNeoConfig) -> dict[str, object]:
    return {
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "attention_types": list(config.attention_types),
        "local_window_size": config.local_window_size,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "initializer_range": config.initializer_range,
        "activation": config.activation,
        "embedding_dropout": config.embedding_dropout,
        "attention_dropout": config.attention_dropout,
        "residual_dropout": config.residual_dropout,
    }


def _config_from_payload(payload: Mapping[str, object]) -> GptNeoConfig:
    expected_keys = {
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_heads",
        "attention_types",
        "local_window_size",
        "layer_norm_epsilon",
        "initializer_range",
        "activation",
        "embedding_dropout",
        "attention_dropout",
        "residual_dropout",
    }
    _require_exact_keys(payload, expected_keys, "config")
    attention_values = payload["attention_types"]
    if not isinstance(attention_values, list) or any(
        not isinstance(value, str) for value in attention_values
    ):
        raise ValueError("config attention_types must be a list of strings")
    activation = payload["activation"]
    if not isinstance(activation, str):
        raise ValueError("config activation must be a string")
    return GptNeoConfig(
        vocab_size=_require_int(payload["vocab_size"], "vocab_size"),
        max_position_embeddings=_require_int(
            payload["max_position_embeddings"],
            "max_position_embeddings",
        ),
        hidden_size=_require_int(payload["hidden_size"], "hidden_size"),
        intermediate_size=_require_int(
            payload["intermediate_size"],
            "intermediate_size",
        ),
        num_layers=_require_int(payload["num_layers"], "num_layers"),
        num_heads=_require_int(payload["num_heads"], "num_heads"),
        attention_types=tuple(attention_values),  # type: ignore[arg-type]
        local_window_size=_require_int(
            payload["local_window_size"],
            "local_window_size",
        ),
        layer_norm_epsilon=_require_float(
            payload["layer_norm_epsilon"],
            "layer_norm_epsilon",
        ),
        initializer_range=_require_float(
            payload["initializer_range"],
            "initializer_range",
        ),
        activation=activation,  # type: ignore[arg-type]
        embedding_dropout=_require_float(
            payload["embedding_dropout"],
            "embedding_dropout",
        ),
        attention_dropout=_require_float(
            payload["attention_dropout"],
            "attention_dropout",
        ),
        residual_dropout=_require_float(
            payload["residual_dropout"],
            "residual_dropout",
        ),
    )


def _tokenizer_to_payload(metadata: TokenizerCheckpointMetadata) -> dict[str, object]:
    return {
        "kind": metadata.kind,
        "identifier": metadata.identifier,
        "revision": metadata.revision,
        "files": [
            {"name": file_hash.name, "sha256": file_hash.sha256}
            for file_hash in sorted(metadata.files, key=lambda value: value.name)
        ],
    }


def _tokenizer_from_payload(payload: Mapping[str, object]) -> TokenizerCheckpointMetadata:
    _require_exact_keys(payload, {"kind", "identifier", "revision", "files"}, "tokenizer")
    kind = _require_string(payload["kind"], "tokenizer kind")
    identifier = _require_string(payload["identifier"], "tokenizer identifier")
    revision = _require_string(payload["revision"], "tokenizer revision")
    file_values = payload["files"]
    if not isinstance(file_values, list):
        raise ValueError("tokenizer files must be a list")
    parsed_files = []
    for value in file_values:
        file_payload = _expect_dict(value, "tokenizer file")
        _require_exact_keys(
            file_payload,
            {"name", "sha256"},
            "tokenizer file",
        )
        parsed_files.append(
            CheckpointFileHash(
                name=_require_string(file_payload["name"], "tokenizer file name"),
                sha256=_require_string(file_payload["sha256"], "tokenizer file hash"),
            )
        )
    return TokenizerCheckpointMetadata(kind, identifier, revision, tuple(parsed_files))


def _source_to_payload(metadata: SourceCheckpointMetadata) -> dict[str, object]:
    return {
        "identifier": metadata.identifier,
        "revision": metadata.revision,
        "sha256": metadata.sha256,
    }


def _source_from_payload(payload: Mapping[str, object]) -> SourceCheckpointMetadata:
    _require_exact_keys(payload, {"identifier", "revision", "sha256"}, "source")
    return SourceCheckpointMetadata(
        identifier=_require_string(payload["identifier"], "source identifier"),
        revision=_require_string(payload["revision"], "source revision"),
        sha256=_require_string(payload["sha256"], "source hash"),
    )


def _native_checkpoint_provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        producer="apm.checkpoint",
        producer_version=str(SCHEMA_VERSION),
        library_versions=(
            ("jax", jax.__version__),
            ("numpy", np.__version__),
        ),
        environment=(
            ("platform", platform.platform()),
            ("python", platform.python_version()),
        ),
    )


def _provenance_to_payload(metadata: CheckpointProvenance) -> dict[str, object]:
    return {
        "producer": metadata.producer,
        "producer_version": metadata.producer_version,
        "library_versions": dict(metadata.library_versions),
        "environment": dict(metadata.environment),
    }


def _provenance_from_payload(payload: Mapping[str, object]) -> CheckpointProvenance:
    _require_exact_keys(
        payload,
        {"producer", "producer_version", "library_versions", "environment"},
        "provenance",
    )
    return CheckpointProvenance(
        producer=_require_string(payload["producer"], "provenance producer"),
        producer_version=_require_string(
            payload["producer_version"],
            "provenance producer version",
        ),
        library_versions=_string_mapping_pairs(
            payload["library_versions"],
            "provenance library versions",
        ),
        environment=_string_mapping_pairs(
            payload["environment"],
            "provenance environment",
        ),
    )


def _parse_manifest(contents: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest is not valid JSON") from error
    manifest = _expect_dict(value, "manifest")
    expected_keys = {
        "schema_version",
        "format",
        "model_file",
        "model_file_sha256",
        "parameter_checksum",
        "config",
        "config_sha256",
        "tokenizer",
        "tokenizer_sha256",
        "source",
        "provenance",
        "provenance_sha256",
        "tensors",
    }
    _require_exact_keys(manifest, expected_keys, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if manifest["format"] != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if manifest["model_file"] != MODEL_FILENAME:
        raise ValueError("checkpoint model filename is not canonical")
    for field_name in (
        "model_file_sha256",
        "parameter_checksum",
        "config_sha256",
        "tokenizer_sha256",
        "provenance_sha256",
    ):
        value = _require_string(manifest[field_name], field_name)
        _validate_sha256(value, field_name)
    return manifest


def _validate_manifest_tensors(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(payload) != set(arrays):
        raise ValueError("manifest tensor names do not match the model file")
    for name, array in arrays.items():
        tensor = _expect_dict(payload[name], f"manifest tensor {name}")
        _require_exact_keys(tensor, {"shape", "dtype", "nbytes"}, name)
        shape = _parse_nonnegative_int_list(tensor["shape"], f"shape for {name}")
        if shape != array.shape:
            raise ValueError(f"manifest shape mismatch for {name}")
        if tensor["dtype"] != array.dtype.name:
            raise ValueError(f"manifest dtype mismatch for {name}")
        if _require_int(tensor["nbytes"], f"nbytes for {name}") != array.nbytes:
            raise ValueError(f"manifest byte-count mismatch for {name}")


def _write_file(path: Path, contents: bytes) -> None:
    with path.open("xb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _stable_json_bytes(payload: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + suffix
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _expect_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    if set(payload) != expected:
        missing = tuple(sorted(expected - set(payload)))
        unexpected = tuple(sorted(set(payload) - expected))
        raise ValueError(
            f"{context} fields do not match; missing={missing}, unexpected={unexpected}"
        )


def _parse_nonnegative_int_list(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    parsed = tuple(_require_int(item, context) for item in value)
    if any(item < 0 for item in parsed):
        raise ValueError(f"{context} must contain nonnegative integers")
    return parsed


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _require_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _require_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _canonical_provenance_pairs(
    pairs: tuple[tuple[str, str], ...],
    context: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(pairs, tuple) or not pairs:
        raise ValueError(f"provenance {context} must be a nonempty tuple")
    if any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(value, str) or not value for value in pair)
        for pair in pairs
    ):
        raise ValueError(f"provenance {context} must contain nonempty string pairs")
    names = tuple(name for name, _ in pairs)
    if len(set(names)) != len(names):
        raise ValueError(f"provenance {context} names must be unique")
    return tuple(sorted(pairs))


def _string_mapping_pairs(value: object, context: str) -> tuple[tuple[str, str], ...]:
    payload = _expect_dict(value, context)
    if any(not isinstance(item, str) or not item for item in payload.values()):
        raise ValueError(f"{context} values must be nonempty strings")
    return tuple((name, str(item)) for name, item in sorted(payload.items()))


def _validate_sha256(value: str, context: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
