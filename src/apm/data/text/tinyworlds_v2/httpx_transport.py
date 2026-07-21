"""Optional production HTTP transport and secret loading for OpenRouter."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
from typing import Protocol

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    VERIFIER_MODEL,
    CandidateModelSpec,
)
from apm.data.text.tinyworlds_v2.catalog import CatalogPayloads
from apm.data.text.tinyworlds_v2.openrouter import (
    HttpTransport,
    TransportError,
    TransportResponse,
)


OPENROUTER_ORIGIN = "https://openrouter.ai"
OPENROUTER_KEY_FILENAME = "openrouter-tinyworlds-key.txt"
OPENROUTER_MANAGEMENT_KEY_ENVIRONMENT = "OPENROUTER_MANAGEMENT_API_KEY"


class CatalogTransport(Protocol):
    """Minimal public GET boundary used to snapshot the model catalog."""

    def get(self, *, url: str, timeout_seconds: float) -> bytes:
        """Return exact successful response bytes."""


@dataclass(frozen=True, slots=True)
class HttpxTransport(HttpTransport, CatalogTransport):
    """Thin synchronous ``httpx`` adapter imported only in production runs."""

    user_agent: str = "apm-tinyworlds-v2/1"

    def post(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        body: bytes,
        timeout_seconds: float,
    ) -> TransportResponse:
        """POST exact request bytes and retain response body plus headers."""
        if type(body) is not bytes:
            raise TypeError("HTTP body must be bytes")
        _validate_openrouter_url(url)
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                "OpenRouter generation requires the optional 'generation' dependency"
            ) from error
        try:
            response = httpx.post(
                url,
                content=body,
                headers={**dict(headers), "User-Agent": self.user_agent},
                timeout=timeout_seconds,
            )
        except httpx.TransportError as error:
            raise TransportError(type(error).__name__) from None
        return TransportResponse(
            status_code=response.status_code,
            headers=tuple(response.headers.multi_items()),
            body=response.content,
        )

    def get_authenticated(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        timeout_seconds: float,
    ) -> TransportResponse:
        """GET exact authenticated generation stats and retain raw response data."""
        _validate_openrouter_url(url)
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                "OpenRouter generation requires the optional 'generation' dependency"
            ) from error
        try:
            response = httpx.get(
                url,
                headers={**dict(headers), "User-Agent": self.user_agent},
                timeout=timeout_seconds,
            )
        except httpx.TransportError as error:
            raise TransportError(type(error).__name__) from None
        return TransportResponse(
            status_code=response.status_code,
            headers=tuple(response.headers.multi_items()),
            body=response.content,
        )

    def get(self, *, url: str, timeout_seconds: float) -> bytes:
        """GET one public catalog response and fail without exposing its body."""
        _validate_openrouter_url(url)
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                "OpenRouter catalog access requires the optional 'generation' dependency"
            ) from error
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=timeout_seconds,
            )
        except httpx.TransportError as error:
            raise TransportError(type(error).__name__) from None
        if not 200 <= response.status_code < 300:
            digest = sha256(response.content).hexdigest()
            raise RuntimeError(
                f"OpenRouter catalog returned HTTP {response.status_code}; "
                f"body SHA-256={digest}"
            )
        return response.content


def load_openrouter_api_key(
    repository_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Read the key from the environment or a mode-0600 local fallback.

    The returned value is never logged or serialized by this module.  Callers
    must pass it only to ``OpenRouterClient``, whose dataclass field suppresses
    representation and comparison.
    """
    resolved_environment = os.environ if environment is None else environment
    environment_key = resolved_environment.get("OPENROUTER_API_KEY")
    if environment_key is not None:
        return _validate_key_text(environment_key)
    key_path = Path(repository_root) / OPENROUTER_KEY_FILENAME
    if key_path.is_symlink() or not key_path.is_file():
        raise FileNotFoundError(
            "OPENROUTER_API_KEY is unset and the local fallback key is missing"
        )
    mode = stat.S_IMODE(key_path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(
            f"local OpenRouter key must have mode 0600, found {mode:04o}"
        )
    # A credential file is conventionally a single line terminated by one
    # platform newline.  Remove that record terminator only; leading spaces,
    # extra blank lines, embedded newlines, and other surrounding whitespace
    # remain invalid and are never silently normalized.
    value = key_path.read_text(encoding="utf-8")
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return _validate_key_text(value)


def load_openrouter_management_api_key(
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Load an optional management-only key without falling back to inference."""
    resolved_environment = os.environ if environment is None else environment
    value = resolved_environment.get(OPENROUTER_MANAGEMENT_KEY_ENVIRONMENT)
    return None if value is None else _validate_key_text(value)


def fetch_catalog_payloads(
    transport: CatalogTransport,
    *,
    model_specs: tuple[CandidateModelSpec, ...] = (*CANDIDATE_MODELS, VERIFIER_MODEL),
    timeout_seconds: float = 60.0,
) -> CatalogPayloads:
    """Fetch public catalog evidence for one explicit ordered model plan."""
    if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
        raise ValueError("catalog timeout_seconds must be positive")
    if (
        type(model_specs) is not tuple
        or not model_specs
        or any(type(spec) is not CandidateModelSpec for spec in model_specs)
    ):
        raise TypeError("catalog model_specs must be a nonempty model tuple")
    # Author and verifier route identities may share an underlying model. Its
    # endpoint response is fetched once and reused when the route locks are
    # resolved.
    specs_by_model_id: dict[str, CandidateModelSpec] = {}
    for spec in model_specs:
        specs_by_model_id.setdefault(spec.request_model_id, spec)
    specs = tuple(specs_by_model_id.values())
    model_url = f"{OPENROUTER_ORIGIN}/api/v1/models"

    def endpoint_payload(model_id: str) -> tuple[str, bytes]:
        url = f"{OPENROUTER_ORIGIN}/api/v1/models/{model_id}/endpoints"
        return model_id, transport.get(url=url, timeout_seconds=float(timeout_seconds))

    with ThreadPoolExecutor(max_workers=len(specs) + 1) as executor:
        models_future = executor.submit(
            transport.get,
            url=model_url,
            timeout_seconds=float(timeout_seconds),
        )
        endpoint_futures = tuple(
            executor.submit(endpoint_payload, spec.request_model_id) for spec in specs
        )
        models_payload = models_future.result()
        endpoints = tuple(future.result() for future in endpoint_futures)
    return CatalogPayloads(
        models=models_payload,
        endpoints=endpoints,
        model_plan_ids=tuple(spec.request_model_id for spec in specs),
    )


def _validate_key_text(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("OpenRouter API key must be one nonempty trimmed line")
    return value


def _validate_openrouter_url(url: str) -> None:
    if (
        type(url) is not str
        or not url.startswith(f"{OPENROUTER_ORIGIN}/api/v1/")
        or "@" in url
        or "#" in url
    ):
        raise ValueError("HTTP transport accepts only OpenRouter API v1 URLs")


__all__ = [
    "CatalogTransport",
    "HttpxTransport",
    "OPENROUTER_KEY_FILENAME",
    "OPENROUTER_MANAGEMENT_KEY_ENVIRONMENT",
    "OPENROUTER_ORIGIN",
    "fetch_catalog_payloads",
    "load_openrouter_api_key",
    "load_openrouter_management_api_key",
]
