"""Strict safetensors checkpoints for complete resumable language-model states."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.training import LmTrainState


LM_TRAIN_STATE_ARTIFACT_FORMAT = "apm.lm.train-state"
LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION = 1
LM_TRAIN_STATE_MANIFEST = "manifest.json"
LM_TRAIN_STATE_TENSORS = "state.safetensors"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
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
_MAX_HEADER_BYTES = 100_000_000

TrainableT = TypeVar("TrainableT")


@dataclass(frozen=True, slots=True)
class LmTrainStateArtifactManifest:
    """Validated identity and content hashes for one immutable state tuple."""

    identity_sha256: str
    state_count: int
    tensor_file_sha256: str
    payload_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.identity_sha256, "training-state identity"),
            (self.tensor_file_sha256, "training-state tensor file"),
            (self.payload_sha256, "training-state payload"),
        ):
            _require_sha256(value, label)
        if type(self.state_count) is not int or self.state_count <= 0:
            raise ValueError("training-state state_count must be positive")


def lm_train_state_checksum(state: LmTrainState[object]) -> str:
    """Return a structure-, dtype-, shape-, and byte-sensitive state digest."""
    if not isinstance(state, LmTrainState):
        raise TypeError("state must be an LmTrainState")
    digest = sha256()
    leaves, structure = jax.tree_util.tree_flatten(state)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        value = np.asarray(leaf)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def write_lm_train_state_artifact(
    directory: str | Path,
    identity_sha256: str,
    states: tuple[LmTrainState[TrainableT], ...],
) -> LmTrainStateArtifactManifest:
    """Atomically write complete trainable, optimizer, RNG, and step state."""
    _require_sha256(identity_sha256, "training-state identity")
    if not isinstance(states, tuple) or not states or any(
        not isinstance(state, LmTrainState) for state in states
    ):
        raise ValueError("training-state artifacts require a nonempty state tuple")
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"training-state artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        arrays, state_records = _flatten_states(states)
        tensor_path = temporary / LM_TRAIN_STATE_TENSORS
        _write_safetensors(
            tensor_path,
            arrays,
            {
                "format": LM_TRAIN_STATE_ARTIFACT_FORMAT,
                "identity_sha256": identity_sha256,
                "schema_version": str(LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION),
            },
        )
        tensor_sha256 = _file_sha256(tensor_path)
        core = {
            "format": LM_TRAIN_STATE_ARTIFACT_FORMAT,
            "identity_sha256": identity_sha256,
            "schema_version": LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION,
            "state_count": len(states),
            "states": state_records,
            "tensor_file": LM_TRAIN_STATE_TENSORS,
            "tensor_file_sha256": tensor_sha256,
        }
        payload_sha256 = sha256(_canonical_json_bytes(core)).hexdigest()
        _write_file(
            temporary / LM_TRAIN_STATE_MANIFEST,
            _canonical_json_bytes({**core, "payload_sha256": payload_sha256}),
        )
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
        return LmTrainStateArtifactManifest(
            identity_sha256,
            len(states),
            tensor_sha256,
            payload_sha256,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_lm_train_state_artifact(
    directory: str | Path,
    identity_sha256: str,
    templates: tuple[LmTrainState[TrainableT], ...],
) -> tuple[LmTrainState[TrainableT], ...]:
    """Strictly load states against deterministic pytree templates."""
    _require_sha256(identity_sha256, "training-state identity")
    if not isinstance(templates, tuple) or not templates or any(
        not isinstance(state, LmTrainState) for state in templates
    ):
        raise ValueError("training-state loading requires nonempty templates")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"training-state artifact must be a directory: {root}")
    if {path.name for path in root.iterdir()} != {
        LM_TRAIN_STATE_MANIFEST,
        LM_TRAIN_STATE_TENSORS,
    }:
        raise ValueError("training-state artifact entries are not canonical")
    manifest_payload = (root / LM_TRAIN_STATE_MANIFEST).read_bytes()
    manifest = _parse_canonical_json(manifest_payload, "training-state manifest")
    _require_fields(
        manifest,
        (
            "format",
            "identity_sha256",
            "payload_sha256",
            "schema_version",
            "state_count",
            "states",
            "tensor_file",
            "tensor_file_sha256",
        ),
        "training-state manifest",
    )
    state_count = _require_integer(manifest["state_count"], "state count")
    if (
        manifest["format"] != LM_TRAIN_STATE_ARTIFACT_FORMAT
        or manifest["schema_version"] != LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION
        or manifest["identity_sha256"] != identity_sha256
        or manifest["tensor_file"] != LM_TRAIN_STATE_TENSORS
        or state_count != len(templates)
    ):
        raise ValueError("training-state artifact identity changed")
    supplied_payload_sha = _require_string(
        manifest["payload_sha256"],
        "payload SHA-256",
    )
    core = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if supplied_payload_sha != sha256(_canonical_json_bytes(core)).hexdigest():
        raise ValueError("training-state manifest payload checksum mismatch")
    tensor_path = root / LM_TRAIN_STATE_TENSORS
    if manifest["tensor_file_sha256"] != _file_sha256(tensor_path):
        raise ValueError("training-state tensor file checksum mismatch")
    arrays, metadata = _read_safetensors(tensor_path)
    if metadata != {
        "format": LM_TRAIN_STATE_ARTIFACT_FORMAT,
        "identity_sha256": identity_sha256,
        "schema_version": str(LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION),
    }:
        raise ValueError("training-state safetensors metadata changed")
    state_records = _require_list(manifest["states"], "training states")
    if len(state_records) != len(templates):
        raise ValueError("training-state record count changed")
    expected_tensor_names: set[str] = set()
    for raw_record in state_records:
        record = _require_record(raw_record, "training-state record")
        _require_fields(
            record,
            ("leaves", "state_sha256", "structure_sha256", "update"),
            "training-state record",
        )
        _require_sha256(record["state_sha256"], "state SHA-256")
        _require_sha256(record["structure_sha256"], "structure SHA-256")
        _require_integer(record["update"], "state update")
        for raw_spec in _require_list(record["leaves"], "state leaves"):
            spec = _require_record(raw_spec, "state leaf")
            _require_fields(spec, ("dtype", "name", "shape"), "state leaf")
            expected_tensor_names.add(
                _require_string(spec["name"], "leaf tensor name")
            )
    if set(arrays) != expected_tensor_names:
        raise ValueError("training-state safetensor names changed")
    return tuple(
        _reconstruct_state(
            template,
            _require_record(raw_record, "training-state record"),
            arrays,
            state_index,
        )
        for state_index, (template, raw_record) in enumerate(
            zip(templates, state_records)
        )
    )


def _flatten_states(
    states: tuple[LmTrainState[TrainableT], ...],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for state_index, state in enumerate(states):
        leaves, structure = jax.tree_util.tree_flatten(state)
        leaf_records: list[dict[str, object]] = []
        for leaf_index, leaf in enumerate(leaves):
            name = f"states.{state_index:04d}.leaves.{leaf_index:04d}"
            value = _canonical_array(np.asarray(leaf))
            arrays[name] = value
            leaf_records.append(
                {
                    "dtype": value.dtype.str,
                    "name": name,
                    "shape": list(value.shape),
                }
            )
        records.append(
            {
                "leaves": leaf_records,
                "state_sha256": lm_train_state_checksum(state),
                "structure_sha256": sha256(
                    str(structure).encode("utf-8")
                ).hexdigest(),
                "update": int(state.step),
            }
        )
    return arrays, records


def _reconstruct_state(
    template: LmTrainState[TrainableT],
    record: dict[str, object],
    arrays: dict[str, np.ndarray],
    state_index: int,
) -> LmTrainState[TrainableT]:
    _require_fields(
        record,
        ("leaves", "state_sha256", "structure_sha256", "update"),
        "training-state record",
    )
    template_leaves, structure = jax.tree_util.tree_flatten(template)
    if record["structure_sha256"] != sha256(
        str(structure).encode("utf-8")
    ).hexdigest():
        raise ValueError("training-state pytree structure changed")
    leaf_records = _require_list(record["leaves"], "training-state leaves")
    if len(leaf_records) != len(template_leaves):
        raise ValueError("training-state leaf count changed")
    leaves: list[jax.Array] = []
    for leaf_index, (template_leaf, raw_spec) in enumerate(
        zip(template_leaves, leaf_records)
    ):
        spec = _require_record(raw_spec, "training-state leaf")
        _require_fields(spec, ("dtype", "name", "shape"), "state leaf")
        name = f"states.{state_index:04d}.leaves.{leaf_index:04d}"
        if spec["name"] != name or name not in arrays:
            raise ValueError("training-state leaf name changed")
        value = arrays[name]
        shape = tuple(
            _require_integer(item, "leaf dimension")
            for item in _require_list(spec["shape"], "leaf shape")
        )
        template_value = np.asarray(template_leaf)
        if (
            value.shape != shape
            or value.shape != template_value.shape
            or value.dtype.str != spec["dtype"]
            or value.dtype != template_value.dtype
        ):
            raise ValueError("training-state leaf dtype or shape changed")
        leaves.append(jnp.asarray(value))
    state = jax.tree_util.tree_unflatten(structure, leaves)
    if not isinstance(state, LmTrainState):
        raise TypeError("training-state tensors did not reconstruct LmTrainState")
    if (
        int(state.step) != record["update"]
        or lm_train_state_checksum(state) != record["state_sha256"]
    ):
        raise ValueError("training-state checksum mismatch")
    return state


def _write_safetensors(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, str],
) -> None:
    canonical = {name: _canonical_array(value) for name, value in sorted(arrays.items())}
    offset = 0
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    payloads: list[bytes] = []
    for name, value in canonical.items():
        payload = value.tobytes(order="C")
        header[name] = {
            "data_offsets": [offset, offset + len(payload)],
            "dtype": _NUMPY_TO_SAFETENSORS[value.dtype],
            "shape": list(value.shape),
        }
        payloads.append(payload)
        offset += len(payload)
    header_payload = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    _write_file(
        path,
        struct.pack("<Q", len(header_payload))
        + header_payload
        + b"".join(payloads),
    )


def _read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    contents = path.read_bytes()
    if len(contents) < 8:
        raise ValueError("training-state safetensors file is too short")
    header_length = struct.unpack("<Q", contents[:8])[0]
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES or (
        8 + header_length > len(contents)
    ):
        raise ValueError("training-state safetensors header length is invalid")
    header = _parse_json_object(
        contents[8 : 8 + header_length],
        "training-state safetensors header",
    )
    metadata = _require_record(
        header.pop("__metadata__", None),
        "safetensors metadata",
    )
    if any(type(key) is not str or type(value) is not str for key, value in metadata.items()):
        raise ValueError("safetensors metadata must contain string pairs")
    data = memoryview(contents)[8 + header_length :]
    arrays: dict[str, np.ndarray] = {}
    expected_offset = 0
    for name, raw_spec in sorted(header.items()):
        spec = _require_record(raw_spec, f"safetensor {name}")
        _require_fields(spec, ("data_offsets", "dtype", "shape"), f"tensor {name}")
        dtype_code = _require_string(spec["dtype"], f"tensor {name} dtype")
        if dtype_code not in _SAFETENSORS_TO_NUMPY:
            raise ValueError(f"unsupported safetensors dtype: {dtype_code}")
        shape = tuple(
            _require_integer(item, f"tensor {name} dimension")
            for item in _require_list(spec["shape"], f"tensor {name} shape")
        )
        offsets = _require_list(spec["data_offsets"], f"tensor {name} offsets")
        if len(offsets) != 2:
            raise ValueError(f"tensor {name} offsets are invalid")
        start, end = tuple(
            _require_integer(item, f"tensor {name} offset") for item in offsets
        )
        dtype = _SAFETENSORS_TO_NUMPY[dtype_code]
        byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if start != expected_offset or end - start != byte_count or end > len(data):
            raise ValueError("safetensors tensor data is not contiguous")
        arrays[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(shape).copy()
        expected_offset = end
    if expected_offset != len(data):
        raise ValueError("safetensors file contains unindexed bytes")
    return arrays, {str(key): str(value) for key, value in metadata.items()}


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    if dtype not in _NUMPY_TO_SAFETENSORS:
        raise TypeError(f"unsupported training-state dtype: {array.dtype}")
    converted = array.astype(dtype, copy=False)
    if converted.ndim == 0:
        return np.array(converted, dtype=dtype).reshape(())
    return np.ascontiguousarray(converted)


def _canonical_json_bytes(record: dict[str, object]) -> bytes:
    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _parse_canonical_json(payload: bytes, label: str) -> dict[str, object]:
    record = _parse_json_object(payload, label)
    if payload != _canonical_json_bytes(record):
        raise ValueError(f"{label} is not canonical JSON")
    return record


def _parse_json_object(payload: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    return _require_record(value, label)


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_record(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_fields(
    record: dict[str, object],
    fields: tuple[str, ...],
    label: str,
) -> None:
    expected = set(fields)
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"{label} fields changed; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


__all__ = [
    "LM_TRAIN_STATE_ARTIFACT_FORMAT",
    "LM_TRAIN_STATE_ARTIFACT_SCHEMA_VERSION",
    "LM_TRAIN_STATE_MANIFEST",
    "LM_TRAIN_STATE_TENSORS",
    "LmTrainStateArtifactManifest",
    "lm_train_state_checksum",
    "load_lm_train_state_artifact",
    "write_lm_train_state_artifact",
]
