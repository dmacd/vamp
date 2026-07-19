"""Strict canonical JSON primitives for TinyWorlds-v2 artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import TypeAlias, Union


JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[
    JsonScalar,
    list["JsonValue"],
    dict[str, "JsonValue"],
]
JsonObject: TypeAlias = dict[str, JsonValue]


class CanonicalJsonError(ValueError):
    """JSON input is malformed, ambiguous, or outside the supported contract."""


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode one validated JSON value without a trailing newline."""
    _validate_json_value(value, "$")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_line_bytes(value: JsonValue) -> bytes:
    """Encode one validated JSON value as a canonical JSONL record."""
    return canonical_json_bytes(value) + b"\n"


def strict_json_loads(payload: bytes, *, label: str = "JSON") -> JsonValue:
    """Decode JSON while rejecting duplicate fields and non-finite numbers."""

    def pairs_hook(pairs: list[tuple[str, JsonValue]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalJsonError(
                    f"duplicate JSON field {key!r} in {label}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> float:
        raise CanonicalJsonError(f"non-finite number {value!r} in {label}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalJsonError(f"invalid {label}: {error}") from error
    _validate_json_value(value, "$")
    return value


def canonical_json_loads(payload: bytes, *, label: str = "JSON") -> JsonValue:
    """Decode a value and require its bytes to use canonical JSON encoding."""
    value = strict_json_loads(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise CanonicalJsonError(f"non-canonical {label}")
    return value


def json_sha256(value: JsonValue) -> str:
    """Return the SHA-256 digest of a value's canonical JSON encoding."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    return sha256(payload).hexdigest()


def require_json_object(value: JsonValue, *, label: str) -> JsonObject:
    """Narrow one decoded JSON value to a plain object."""
    if type(value) is not dict:
        raise CanonicalJsonError(f"{label} must be a JSON object")
    return value


def require_exact_fields(
    record: JsonObject,
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    """Reject missing and unknown fields at a strict artifact boundary."""
    actual = set(record)
    wanted = set(expected)
    if actual != wanted:
        raise CanonicalJsonError(
            f"{label} fields differ; "
            f"unknown={tuple(sorted(actual - wanted))}, "
            f"missing={tuple(sorted(wanted - actual))}"
        )


def _validate_json_value(value: object, path: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalJsonError(f"non-finite float at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJsonError(f"non-string object key at {path}")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise CanonicalJsonError(
        f"unsupported JSON value {type(value).__name__} at {path}"
    )
