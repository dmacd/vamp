"""Strict, dependency-free conversion of pinned TinyStories GPT-Neo weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import jax
import numpy as np

from apm.lm.checkpoint import (
    BaseCheckpointRef,
    CheckpointFileHash,
    CheckpointProvenance,
    LoadedGptNeoCheckpoint,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
    unflatten_gpt_neo_params,
)
from apm.lm.config import AttentionType, GptNeoConfig
from apm.lm.parameters import GptNeoParams


TINYSTORIES_CONVERTER_VERSION = "1"
TINYSTORIES_ARTIFACT_SCHEMA_VERSION = 1
TINYSTORIES_ARTIFACT_FORMAT = "apm-tinystories-8m"
TINYSTORIES_ARTIFACT_MANIFEST = "artifact.json"
TINYSTORIES_CHECKPOINT_DIRECTORY = "checkpoint"
TINYSTORIES_TOKENIZER_DIRECTORY = "tokenizer"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONVERTER_PRODUCER = "apm.tinystories_conversion"


@dataclass(frozen=True)
class PinnedArtifactFile:
    """Expected filename, byte count, and SHA-256 digest for one source file."""

    name: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or Path(self.name).name != self.name
        ):
            raise ValueError("pinned artifact name must be a nonempty basename")
        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ValueError("pinned artifact hash must be a lowercase SHA-256 digest")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("pinned artifact size must be a positive integer")


@dataclass(frozen=True)
class TinyStoriesSourceContract:
    """Complete immutable identity of the supported TinyStories source snapshot."""

    model_id: str
    revision: str
    transformers_version: str
    config_file: PinnedArtifactFile
    model_file: PinnedArtifactFile
    tokenizer_files: tuple[PinnedArtifactFile, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.model_id,
                self.revision,
                self.transformers_version,
            )
        ):
            raise ValueError("TinyStories source identity fields must not be empty")
        if not isinstance(self.config_file, PinnedArtifactFile) or not isinstance(
            self.model_file,
            PinnedArtifactFile,
        ):
            raise TypeError("TinyStories config and model files must be pinned artifacts")
        if not isinstance(self.tokenizer_files, tuple) or not self.tokenizer_files:
            raise ValueError("TinyStories source must contain pinned tokenizer files")
        if any(
            not isinstance(value, PinnedArtifactFile)
            for value in self.tokenizer_files
        ):
            raise TypeError("TinyStories tokenizer files must be pinned artifacts")
        names = tuple(value.name for value in self.tokenizer_files)
        if len(names) != len(set(names)):
            raise ValueError("TinyStories tokenizer filenames must be unique")
        object.__setattr__(
            self,
            "tokenizer_files",
            tuple(sorted(self.tokenizer_files, key=lambda value: value.name)),
        )


TINYSTORIES_SOURCE = TinyStoriesSourceContract(
    model_id="roneneldan/TinyStories-8M",
    revision="8612e3b15c66ffa94eaa6ee0de5c96edd2d630af",
    transformers_version="4.28.1",
    config_file=PinnedArtifactFile(
        "config.json",
        "5ff16b03beb4466bde520469a815a2d439e16896655d1151c3b44686b387a42d",
        1_161,
    ),
    model_file=PinnedArtifactFile(
        "pytorch_model.bin",
        "22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716",
        112_405_309,
    ),
    tokenizer_files=(
        PinnedArtifactFile(
            "merges.txt",
            "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            456_318,
        ),
        PinnedArtifactFile(
            "special_tokens_map.json",
            "98412137ae43c77f8af52eb51b19c3536d3242cb55339167d841005fa94a23b7",
            438,
        ),
        PinnedArtifactFile(
            "tokenizer.json",
            "f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5",
            2_107_652,
        ),
        PinnedArtifactFile(
            "tokenizer_config.json",
            "3d76da0fd37493fbfcd3f0fa9757753d31f92e1779ebd9130809b45546a60261",
            722,
        ),
        PinnedArtifactFile(
            "vocab.json",
            "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
            798_156,
        ),
    ),
)


@dataclass(frozen=True)
class ConvertedGptNeoModel:
    """A validated GPT-Neo configuration and converted immutable parameters."""

    config: GptNeoConfig
    params: GptNeoParams


@dataclass(frozen=True)
class ConvertedTinyStoriesModel:
    """A converted model bound to the exact supported TinyStories source."""

    model: ConvertedGptNeoModel
    source: TinyStoriesSourceContract


@dataclass(frozen=True)
class LoadedTinyStoriesArtifact:
    """A validated local TinyStories artifact and its copied tokenizer files."""

    directory: Path
    checkpoint: LoadedGptNeoCheckpoint
    tokenizer_files: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(
            self,
            "tokenizer_files",
            tuple(Path(path) for path in self.tokenizer_files),
        )


@dataclass(frozen=True)
class _ParameterMapping:
    source: str
    target: str
    shape: tuple[int, ...]
    transpose: bool = False


ArrayValue = np.ndarray | jax.Array


def gpt_neo_config_from_transformers(
    payload: Mapping[str, object],
) -> GptNeoConfig:
    """Translate and validate the GPT-Neo fields used by Transformers 4.28.1."""
    model_type = _required_string(payload, "model_type")
    if model_type != "gpt_neo":
        raise ValueError("Transformers config model_type must be 'gpt_neo'")
    transformers_version = _required_string(payload, "transformers_version")
    if transformers_version != TINYSTORIES_SOURCE.transformers_version:
        raise ValueError(
            "Transformers config version must be exactly "
            f"{TINYSTORIES_SOURCE.transformers_version}"
        )
    activation = _required_string(payload, "activation_function")
    if activation != "gelu_new":
        raise ValueError("Transformers config activation_function must be 'gelu_new'")
    hidden_size = _required_int(payload, "hidden_size")
    intermediate_value = _required_field(payload, "intermediate_size")
    intermediate_size = (
        4 * hidden_size
        if intermediate_value is None
        else _strict_int(intermediate_value, "intermediate_size")
    )
    num_layers = _required_int(payload, "num_layers")
    attention_types = _expand_attention_types(
        _required_field(payload, "attention_types"),
        num_layers,
    )
    attention_layers = _string_sequence(
        _required_field(payload, "attention_layers"),
        "attention_layers",
    )
    if attention_layers != attention_types:
        raise ValueError(
            "Transformers attention_layers does not match expanded attention_types"
        )
    return GptNeoConfig(
        vocab_size=_required_int(payload, "vocab_size"),
        max_position_embeddings=_required_int(
            payload,
            "max_position_embeddings",
        ),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=_required_int(payload, "num_heads"),
        attention_types=attention_types,
        local_window_size=_required_int(payload, "window_size"),
        layer_norm_epsilon=_required_float(payload, "layer_norm_epsilon"),
        initializer_range=_required_float(payload, "initializer_range"),
        activation="gelu_new",
        embedding_dropout=_required_float(payload, "embed_dropout"),
        attention_dropout=_required_float(payload, "attention_dropout"),
        residual_dropout=_required_float(payload, "resid_dropout"),
    )


def convert_gpt_neo_state_dict(
    state_dict: Mapping[str, ArrayValue],
    transformers_config: Mapping[str, object],
) -> ConvertedGptNeoModel:
    """Strictly convert a complete Transformers 4.28.1 GPT-Neo state dict."""
    config = gpt_neo_config_from_transformers(transformers_config)
    mappings = _parameter_mappings(config)
    expected_names = {
        *(mapping.source for mapping in mappings),
        "lm_head.weight",
        *(
            name
            for layer_index in range(config.num_layers)
            for name in _attention_buffer_names(layer_index)
        ),
    }
    actual_names = set(state_dict)
    missing = tuple(sorted(expected_names - actual_names))
    unexpected = tuple(sorted(actual_names - expected_names))
    if missing or unexpected:
        raise ValueError(
            "source state-dict keys do not match; "
            f"missing={missing}, unexpected={unexpected}"
        )
    arrays = {
        mapping.source: _validated_float32_array(
            state_dict[mapping.source],
            mapping.shape,
            mapping.source,
        )
        for mapping in mappings
    }
    lm_head = _validated_float32_array(
        state_dict["lm_head.weight"],
        (config.vocab_size, config.hidden_size),
        "lm_head.weight",
    )
    token_embedding = arrays["transformer.wte.weight"]
    if lm_head.tobytes(order="C") != token_embedding.tobytes(order="C"):
        raise ValueError("lm_head.weight is not exactly tied to transformer.wte.weight")
    _validate_attention_buffers(state_dict, config)
    canonical_tensors = {
        mapping.target: np.ascontiguousarray(
            arrays[mapping.source].T if mapping.transpose else arrays[mapping.source]
        )
        for mapping in mappings
    }
    return ConvertedGptNeoModel(
        config=config,
        params=unflatten_gpt_neo_params(canonical_tensors, config),
    )


def convert_tinystories_state_dict(
    state_dict: Mapping[str, ArrayValue],
    config_contents: bytes,
) -> ConvertedTinyStoriesModel:
    """Convert only when config bytes match the exact pinned TinyStories snapshot."""
    _validate_pinned_contents(config_contents, TINYSTORIES_SOURCE.config_file)
    config_payload = _parse_json_object(config_contents, "TinyStories config")
    converted = convert_gpt_neo_state_dict(state_dict, config_payload)
    if converted.config != _pinned_gpt_neo_config():
        raise ValueError("TinyStories config does not match the pinned architecture")
    return ConvertedTinyStoriesModel(converted, TINYSTORIES_SOURCE)


def validate_pinned_artifact_file(
    path: Path,
    expected: PinnedArtifactFile,
) -> None:
    """Reject a downloaded source file unless size and SHA-256 are exact."""
    source_path = Path(path)
    if source_path.name != expected.name:
        raise ValueError(
            f"source filename {source_path.name!r} does not match {expected.name!r}"
        )
    stat = source_path.stat()
    if stat.st_size != expected.size:
        raise ValueError(
            f"source file {expected.name} has size {stat.st_size}; expected {expected.size}"
        )
    if _file_sha256(source_path) != expected.sha256:
        raise ValueError(f"source file {expected.name} has an unexpected SHA-256 digest")


def tinystories_conversion_provenance(
    *,
    library_versions: tuple[tuple[str, str], ...],
    environment: tuple[tuple[str, str], ...],
) -> CheckpointProvenance:
    """Build canonical checkpoint provenance for the pinned converter."""
    return CheckpointProvenance(
        producer=_CONVERTER_PRODUCER,
        producer_version=TINYSTORIES_CONVERTER_VERSION,
        library_versions=library_versions,
        environment=environment,
    )


def save_tinystories_artifact(
    directory: Path,
    converted: ConvertedTinyStoriesModel,
    *,
    tokenizer_files: Mapping[str, bytes],
    provenance: CheckpointProvenance,
) -> LoadedTinyStoriesArtifact:
    """Atomically publish a checkpoint plus exact pinned tokenizer artifacts."""
    if converted.source != TINYSTORIES_SOURCE:
        raise ValueError("converted model is not bound to the pinned TinyStories source")
    _validate_conversion_provenance(provenance)
    _validate_tokenizer_contents(tokenizer_files)
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"TinyStories artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        tokenizer_directory = temporary / TINYSTORIES_TOKENIZER_DIRECTORY
        tokenizer_directory.mkdir()
        for artifact in TINYSTORIES_SOURCE.tokenizer_files:
            _write_file(tokenizer_directory / artifact.name, tokenizer_files[artifact.name])
        checkpoint_ref = save_gpt_neo_checkpoint(
            temporary / TINYSTORIES_CHECKPOINT_DIRECTORY,
            converted.model.params,
            converted.model.config,
            tokenizer=_tokenizer_metadata(),
            source=SourceCheckpointMetadata(
                TINYSTORIES_SOURCE.model_id,
                TINYSTORIES_SOURCE.revision,
                TINYSTORIES_SOURCE.model_file.sha256,
            ),
            provenance=provenance,
        )
        manifest = {
            "schema_version": TINYSTORIES_ARTIFACT_SCHEMA_VERSION,
            "format": TINYSTORIES_ARTIFACT_FORMAT,
            "source": _source_contract_payload(TINYSTORIES_SOURCE),
            "checkpoint": {
                "directory": TINYSTORIES_CHECKPOINT_DIRECTORY,
                "manifest_sha256": checkpoint_ref.manifest_sha256,
                "parameter_checksum": checkpoint_ref.parameter_checksum,
            },
            "tokenizer_directory": TINYSTORIES_TOKENIZER_DIRECTORY,
        }
        _write_file(
            temporary / TINYSTORIES_ARTIFACT_MANIFEST,
            _stable_json_bytes(manifest, newline=True),
        )
        _load_tinystories_artifact(temporary)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return _load_tinystories_artifact(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_tinystories_artifact(directory: Path) -> LoadedTinyStoriesArtifact:
    """Load only a complete schema-v1 artifact matching the pinned source."""
    return _load_tinystories_artifact(Path(directory))


def _parameter_mappings(config: GptNeoConfig) -> tuple[_ParameterMapping, ...]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    global_mappings = (
        _ParameterMapping(
            "transformer.wte.weight",
            "token_embedding",
            (config.vocab_size, hidden),
        ),
        _ParameterMapping(
            "transformer.wpe.weight",
            "position_embedding",
            (config.max_position_embeddings, hidden),
        ),
        _ParameterMapping("transformer.ln_f.weight", "final_norm.scale", (hidden,)),
        _ParameterMapping("transformer.ln_f.bias", "final_norm.bias", (hidden,)),
    )
    blocks = tuple(
        mapping
        for layer_index in range(config.num_layers)
        for mapping in _block_parameter_mappings(layer_index, hidden, intermediate)
    )
    return global_mappings[:2] + blocks + global_mappings[2:]


def _block_parameter_mappings(
    layer_index: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[_ParameterMapping, ...]:
    source = f"transformer.h.{layer_index}"
    target = f"blocks.{layer_index}"
    return (
        _ParameterMapping(
            f"{source}.ln_1.weight",
            f"{target}.attention_norm.scale",
            (hidden_size,),
        ),
        _ParameterMapping(
            f"{source}.ln_1.bias",
            f"{target}.attention_norm.bias",
            (hidden_size,),
        ),
        _ParameterMapping(
            f"{source}.attn.attention.q_proj.weight",
            f"{target}.attention.query.kernel",
            (hidden_size, hidden_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.attn.attention.k_proj.weight",
            f"{target}.attention.key.kernel",
            (hidden_size, hidden_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.attn.attention.v_proj.weight",
            f"{target}.attention.value.kernel",
            (hidden_size, hidden_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.attn.attention.out_proj.weight",
            f"{target}.attention.output.kernel",
            (hidden_size, hidden_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.attn.attention.out_proj.bias",
            f"{target}.attention.output.bias",
            (hidden_size,),
        ),
        _ParameterMapping(
            f"{source}.ln_2.weight",
            f"{target}.mlp_norm.scale",
            (hidden_size,),
        ),
        _ParameterMapping(
            f"{source}.ln_2.bias",
            f"{target}.mlp_norm.bias",
            (hidden_size,),
        ),
        _ParameterMapping(
            f"{source}.mlp.c_fc.weight",
            f"{target}.mlp.input_projection.kernel",
            (intermediate_size, hidden_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.mlp.c_fc.bias",
            f"{target}.mlp.input_projection.bias",
            (intermediate_size,),
        ),
        _ParameterMapping(
            f"{source}.mlp.c_proj.weight",
            f"{target}.mlp.output_projection.kernel",
            (hidden_size, intermediate_size),
            True,
        ),
        _ParameterMapping(
            f"{source}.mlp.c_proj.bias",
            f"{target}.mlp.output_projection.bias",
            (hidden_size,),
        ),
    )


def _validate_attention_buffers(
    state_dict: Mapping[str, ArrayValue],
    config: GptNeoConfig,
) -> None:
    causal = np.tril(
        np.ones(
            (config.max_position_embeddings, config.max_position_embeddings),
            dtype=np.bool_,
        )
    )
    local = np.bitwise_xor(
        causal,
        np.tril(causal, -config.local_window_size),
    )
    masks = {"global": causal, "local": local}
    for layer_index, attention_type in enumerate(config.attention_types):
        bias_name, masked_bias_name = _attention_buffer_names(layer_index)
        bias = np.asarray(state_dict[bias_name])
        expected_bias = masks[attention_type][None, None, :, :]
        if bias.dtype != np.dtype("bool"):
            raise TypeError(f"source tensor {bias_name} has dtype {bias.dtype}; expected bool")
        if bias.shape != expected_bias.shape:
            raise ValueError(
                f"source tensor {bias_name} has shape {bias.shape}; expected {expected_bias.shape}"
            )
        if not np.array_equal(bias, expected_bias):
            raise ValueError(f"source tensor {bias_name} does not match its {attention_type} mask")
        masked_bias = _validated_float32_array(
            state_dict[masked_bias_name],
            (),
            masked_bias_name,
        )
        if masked_bias.tobytes() != np.asarray(-1e9, dtype=np.float32).tobytes():
            raise ValueError(f"source tensor {masked_bias_name} must equal float32 -1e9")


def _attention_buffer_names(layer_index: int) -> tuple[str, str]:
    prefix = f"transformer.h.{layer_index}.attn.attention"
    return f"{prefix}.bias", f"{prefix}.masked_bias"


def _validated_float32_array(
    value: ArrayValue,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(
            f"source tensor {name} has shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype("float32"):
        raise TypeError(
            f"source tensor {name} has dtype {array.dtype}; expected float32"
        )
    return np.ascontiguousarray(array)


def _expand_attention_types(value: object, num_layers: int) -> tuple[AttentionType, ...]:
    groups = _sequence(value, "attention_types")
    parsed_groups = tuple(_parse_attention_group(group) for group in groups)
    expanded = tuple(
        attention_type
        for pattern, repetitions in parsed_groups
        for _ in range(repetitions)
        for attention_type in pattern
    )
    if len(expanded) != num_layers:
        raise ValueError(
            "expanded attention_types must contain exactly one entry per layer"
        )
    return expanded


def _parse_attention_group(
    value: object,
) -> tuple[tuple[AttentionType, ...], int]:
    group = _sequence(value, "attention_types group")
    if len(group) != 2:
        raise ValueError("each attention_types group must contain pattern and count")
    pattern = _string_sequence(group[0], "attention_types pattern")
    if not pattern or any(value not in ("global", "local") for value in pattern):
        raise ValueError("attention_types patterns may contain only global and local")
    repetitions = _strict_int(group[1], "attention_types repetition count")
    if repetitions <= 0:
        raise ValueError("attention_types repetition count must be positive")
    return pattern, repetitions


def _pinned_gpt_neo_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=50_257,
        max_position_embeddings=2_048,
        hidden_size=256,
        intermediate_size=1_024,
        num_layers=8,
        num_heads=16,
        attention_types=("global", "local") * 4,
        local_window_size=256,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        activation="gelu_new",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def _tokenizer_metadata() -> TokenizerCheckpointMetadata:
    return TokenizerCheckpointMetadata(
        kind="gpt2-bpe",
        identifier=TINYSTORIES_SOURCE.model_id,
        revision=TINYSTORIES_SOURCE.revision,
        files=tuple(
            CheckpointFileHash(artifact.name, artifact.sha256)
            for artifact in TINYSTORIES_SOURCE.tokenizer_files
        ),
    )


def _validate_tokenizer_contents(tokenizer_files: Mapping[str, bytes]) -> None:
    expected_names = {artifact.name for artifact in TINYSTORIES_SOURCE.tokenizer_files}
    actual_names = set(tokenizer_files)
    if actual_names != expected_names:
        raise ValueError(
            "tokenizer filenames do not match; "
            f"missing={tuple(sorted(expected_names - actual_names))}, "
            f"unexpected={tuple(sorted(actual_names - expected_names))}"
        )
    for artifact in TINYSTORIES_SOURCE.tokenizer_files:
        contents = tokenizer_files[artifact.name]
        if not isinstance(contents, bytes):
            raise TypeError(f"tokenizer file {artifact.name} contents must be bytes")
        _validate_pinned_contents(contents, artifact)


def _validate_pinned_contents(contents: bytes, expected: PinnedArtifactFile) -> None:
    if not isinstance(contents, bytes):
        raise TypeError(f"source file {expected.name} contents must be bytes")
    if len(contents) != expected.size:
        raise ValueError(
            f"source file {expected.name} has size {len(contents)}; expected {expected.size}"
        )
    if hashlib.sha256(contents).hexdigest() != expected.sha256:
        raise ValueError(f"source file {expected.name} has an unexpected SHA-256 digest")


def _load_tinystories_artifact(directory: Path) -> LoadedTinyStoriesArtifact:
    expected_entries = {
        TINYSTORIES_ARTIFACT_MANIFEST,
        TINYSTORIES_CHECKPOINT_DIRECTORY,
        TINYSTORIES_TOKENIZER_DIRECTORY,
    }
    actual_entries = {path.name for path in directory.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError("TinyStories artifact directory entries are not canonical")
    manifest = _parse_json_object(
        (directory / TINYSTORIES_ARTIFACT_MANIFEST).read_bytes(),
        "TinyStories artifact manifest",
    )
    _require_exact_keys(
        manifest,
        {"schema_version", "format", "source", "checkpoint", "tokenizer_directory"},
        "TinyStories artifact manifest",
    )
    if manifest["schema_version"] != TINYSTORIES_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported TinyStories artifact schema version")
    if manifest["format"] != TINYSTORIES_ARTIFACT_FORMAT:
        raise ValueError("unsupported TinyStories artifact format")
    if manifest["source"] != _source_contract_payload(TINYSTORIES_SOURCE):
        raise ValueError("TinyStories artifact source contract does not match")
    if manifest["tokenizer_directory"] != TINYSTORIES_TOKENIZER_DIRECTORY:
        raise ValueError("TinyStories tokenizer directory is not canonical")
    checkpoint_payload = _expect_dict(manifest["checkpoint"], "checkpoint reference")
    _require_exact_keys(
        checkpoint_payload,
        {"directory", "manifest_sha256", "parameter_checksum"},
        "checkpoint reference",
    )
    if checkpoint_payload["directory"] != TINYSTORIES_CHECKPOINT_DIRECTORY:
        raise ValueError("TinyStories checkpoint directory is not canonical")
    checkpoint_directory = directory / TINYSTORIES_CHECKPOINT_DIRECTORY
    if {path.name for path in checkpoint_directory.iterdir()} != {
        "manifest.json",
        "model.safetensors",
    }:
        raise ValueError("TinyStories checkpoint directory entries are not canonical")
    checkpoint = load_gpt_neo_checkpoint(
        BaseCheckpointRef(
            checkpoint_directory,
            _required_string(checkpoint_payload, "manifest_sha256"),
            _required_string(checkpoint_payload, "parameter_checksum"),
        )
    )
    expected_source = SourceCheckpointMetadata(
        TINYSTORIES_SOURCE.model_id,
        TINYSTORIES_SOURCE.revision,
        TINYSTORIES_SOURCE.model_file.sha256,
    )
    if checkpoint.source != expected_source:
        raise ValueError("checkpoint source metadata does not match TinyStories")
    if checkpoint.tokenizer != _tokenizer_metadata():
        raise ValueError("checkpoint tokenizer metadata does not match TinyStories")
    if checkpoint.config != _pinned_gpt_neo_config():
        raise ValueError("checkpoint config does not match TinyStories")
    _validate_conversion_provenance(checkpoint.provenance)
    tokenizer_directory = directory / TINYSTORIES_TOKENIZER_DIRECTORY
    expected_tokenizer_names = {
        artifact.name for artifact in TINYSTORIES_SOURCE.tokenizer_files
    }
    if {path.name for path in tokenizer_directory.iterdir()} != expected_tokenizer_names:
        raise ValueError("TinyStories tokenizer directory entries are not canonical")
    tokenizer_paths = tuple(
        tokenizer_directory / artifact.name
        for artifact in TINYSTORIES_SOURCE.tokenizer_files
    )
    for path, artifact in zip(tokenizer_paths, TINYSTORIES_SOURCE.tokenizer_files):
        validate_pinned_artifact_file(path, artifact)
    return LoadedTinyStoriesArtifact(
        directory.resolve(),
        checkpoint,
        tuple(path.resolve() for path in tokenizer_paths),
    )


def _validate_conversion_provenance(provenance: CheckpointProvenance) -> None:
    if provenance.producer != _CONVERTER_PRODUCER:
        raise ValueError("TinyStories artifact provenance has the wrong producer")
    if provenance.producer_version != TINYSTORIES_CONVERTER_VERSION:
        raise ValueError("TinyStories artifact provenance has the wrong converter version")
    library_versions = dict(provenance.library_versions)
    if not {"torch", "transformers"}.issubset(library_versions):
        raise ValueError("TinyStories provenance must identify torch and transformers")
    if library_versions["transformers"] != TINYSTORIES_SOURCE.transformers_version:
        raise ValueError("TinyStories provenance has the wrong transformers version")
    if not {"python", "platform"}.issubset(dict(provenance.environment)):
        raise ValueError("TinyStories provenance must identify Python and the platform")


def _source_contract_payload(source: TinyStoriesSourceContract) -> dict[str, object]:
    artifact_payload = lambda artifact: {
        "name": artifact.name,
        "sha256": artifact.sha256,
        "size": artifact.size,
    }
    return {
        "model_id": source.model_id,
        "revision": source.revision,
        "transformers_version": source.transformers_version,
        "config_file": artifact_payload(source.config_file),
        "model_file": artifact_payload(source.model_file),
        "tokenizer_files": [
            artifact_payload(artifact) for artifact in source.tokenizer_files
        ],
    }


def _parse_json_object(contents: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    return _expect_dict(value, context)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _expect_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(name, str) for name in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{context} fields do not match; "
            f"missing={tuple(sorted(expected - set(payload)))}, "
            f"unexpected={tuple(sorted(set(payload) - expected))}"
        )


def _required_field(payload: Mapping[str, object], name: str) -> object:
    if name not in payload:
        raise ValueError(f"Transformers config is missing {name}")
    return payload[name]


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = _required_field(payload, name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strict_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    return _strict_int(_required_field(payload, name), name)


def _required_float(payload: Mapping[str, object], name: str) -> float:
    value = _required_field(payload, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _sequence(value: object, context: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a sequence")
    return tuple(value)


def _string_sequence(value: object, context: str) -> tuple[AttentionType, ...]:
    sequence = _sequence(value, context)
    if any(not isinstance(item, str) for item in sequence):
        raise ValueError(f"{context} must contain strings")
    return tuple(sequence)  # type: ignore[return-value]


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


def _stable_json_bytes(payload: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + suffix
    ).encode("utf-8")
