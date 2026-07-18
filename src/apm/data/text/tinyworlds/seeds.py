"""Domain-separated SHA-256 seed derivation for reproducible TinyWorlds."""

from __future__ import annotations

from hashlib import sha256
import json
import re


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MASTER_DOMAIN = b"tinyworlds-master-seed-v1\0"
_SUBSEED_DOMAIN = b"tinyworlds-subseed-v1\0"


def derive_master_seed(
    benchmark_version: str,
    public_seed: int,
    base_manifest_sha256: str,
    base_parameter_checksum: str,
) -> str:
    """Derive the benchmark master seed from all immutable base identities."""
    _validate_component(benchmark_version, "benchmark_version")
    if type(public_seed) is not int or public_seed < 0:
        raise ValueError("public_seed must be a nonnegative integer")
    _validate_sha256(base_manifest_sha256, "base_manifest_sha256")
    _validate_sha256(base_parameter_checksum, "base_parameter_checksum")
    payload = _canonical_json(
        (
            benchmark_version,
            public_seed,
            base_manifest_sha256,
            base_parameter_checksum,
        )
    )
    return sha256(_MASTER_DOMAIN + payload).hexdigest()


def derive_subseed(
    master_seed_sha256: str,
    namespace: str,
    *stable_components: str | int,
) -> str:
    """Derive one namespace/record seed without consuming shared RNG state."""
    _validate_sha256(master_seed_sha256, "master_seed_sha256")
    _validate_component(namespace, "namespace")
    for component in stable_components:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise TypeError("stable seed components must be strings or integers")
        if isinstance(component, str):
            _validate_component(component, "stable seed component")
    payload = bytes.fromhex(master_seed_sha256) + _canonical_json(
        (namespace, *stable_components)
    )
    return sha256(_SUBSEED_DOMAIN + payload).hexdigest()


def subseed_uint64(
    master_seed_sha256: str,
    namespace: str,
    *stable_components: str | int,
) -> int:
    """Return the leading unsigned 64 bits of a namespaced SHA-256 seed."""
    digest = derive_subseed(
        master_seed_sha256,
        namespace,
        *stable_components,
    )
    return int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=False)


def _canonical_json(values: tuple[str | int, ...]) -> bytes:
    return json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256")


def _validate_component(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be nonempty without outer whitespace")


__all__ = [
    "derive_master_seed",
    "derive_subseed",
    "subseed_uint64",
]
