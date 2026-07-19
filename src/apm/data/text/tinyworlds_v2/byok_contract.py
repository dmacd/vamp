"""Canonical sanitized authorization evidence for OpenRouter paid boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from apm.data.text.tinyworlds_v2.generation_schema import GenerationContractError
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    require_exact_fields,
    require_json_object,
    strict_json_loads,
)


_FIELDS = (
    "attestation_sha256",
    "attested_at_utc",
    "checked_at_utc",
    "decision",
    "endpoint",
    "expires_at_utc",
    "method",
    "response_body_sha256",
    "source",
    "status_code",
    "total_count",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_ATTESTATION_SECONDS = 24 * 60 * 60


def canonical_byok_authorization(value: JsonValue) -> JsonObject:
    """Validate and detach one allowed, secret-free zero-BYOK proof."""
    record = require_json_object(value, label="BYOK authorization")
    require_exact_fields(record, _FIELDS, label="BYOK authorization")
    checked = _utc(record["checked_at_utc"], "BYOK checked_at_utc")
    if record["decision"] != "allowed" or record["total_count"] != 0:
        raise GenerationContractError(
            "paid reservation requires an allowed zero-BYOK authorization"
        )
    source = record["source"]
    if source == "management_api":
        if (
            record["endpoint"] != "/api/v1/byok"
            or record["method"] != "GET"
            or record["status_code"] != 200
            or not _is_sha256(record["response_body_sha256"])
            or any(
                record[field] is not None
                for field in (
                    "attestation_sha256",
                    "attested_at_utc",
                    "expires_at_utc",
                )
            )
        ):
            raise GenerationContractError(
                "management BYOK authorization is incomplete or unsanitized"
            )
    elif source == "manual_attestation":
        if (
            not _is_sha256(record["attestation_sha256"])
            or any(
                record[field] is not None
                for field in (
                    "endpoint",
                    "method",
                    "response_body_sha256",
                    "status_code",
                )
            )
        ):
            raise GenerationContractError(
                "manual BYOK authorization is incomplete or unsanitized"
            )
        attested = _utc(record["attested_at_utc"], "BYOK attested_at_utc")
        expires = _utc(record["expires_at_utc"], "BYOK expires_at_utc")
        lifetime = (expires - attested).total_seconds()
        if not 0 < lifetime <= _MAX_ATTESTATION_SECONDS or not (
            attested <= checked < expires
        ):
            raise GenerationContractError(
                "manual BYOK authorization was not active at its check time"
            )
    else:
        raise GenerationContractError("BYOK authorization source is unsupported")
    # Canonical round-trip detaches caller-owned dictionaries and nested data.
    return require_json_object(
        strict_json_loads(canonical_json_bytes(record), label="BYOK authorization"),
        label="BYOK authorization",
    )


def _utc(value: JsonValue, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise GenerationContractError(f"{label} must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise GenerationContractError(f"{label} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise GenerationContractError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: JsonValue) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


__all__ = ["canonical_byok_authorization"]
