from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import CANDIDATE_MODELS, VERIFIER_MODEL
from apm.data.text.tinyworlds_v2.httpx_transport import (
    HttpxTransport,
    fetch_catalog_payloads,
    load_openrouter_api_key,
)


class _FakeCatalogTransport:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, *, url: str, timeout_seconds: float) -> bytes:
        assert timeout_seconds == 7.0
        self.urls.append(url)
        return json.dumps({"url": url}, sort_keys=True).encode()


def test_local_key_requires_exact_private_mode(tmp_path: Path) -> None:
    key_path = tmp_path / "openrouter-tinyworlds-key.txt"
    key_path.write_text("secret-test-key", encoding="utf-8")
    key_path.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        load_openrouter_api_key(tmp_path, environment={})

    key_path.chmod(0o600)
    assert load_openrouter_api_key(tmp_path, environment={}) == "secret-test-key"


def test_environment_key_takes_precedence_without_touching_fallback(tmp_path: Path) -> None:
    assert (
        load_openrouter_api_key(
            tmp_path,
            environment={"OPENROUTER_API_KEY": "environment-secret"},
        )
        == "environment-secret"
    )


def test_catalog_fetch_is_complete_and_returns_fixed_order() -> None:
    transport = _FakeCatalogTransport()
    payloads = fetch_catalog_payloads(transport, timeout_seconds=7.0)
    expected_ids = tuple(
        model.request_model_id for model in (*CANDIDATE_MODELS, VERIFIER_MODEL)
    )

    assert tuple(model_id for model_id, _ in payloads.endpoints) == expected_ids
    assert len(transport.urls) == len(expected_ids) + 1
    assert any(url.endswith("/api/v1/models") for url in transport.urls)


def test_authenticated_stats_get_retains_exact_http_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Headers:
        def multi_items(self):
            return [("x-stats", "one"), ("x-stats", "two")]

    def get(url, *, headers, timeout):
        observed.update(url=url, headers=headers, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            headers=_Headers(),
            content=b'{"data":{"total_cost":0.01}}',
        )

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(get=get, TransportError=OSError),
    )
    response = HttpxTransport(user_agent="fixture-agent").get_authenticated(
        url="https://openrouter.ai/api/v1/generation?id=gen-1",
        headers=(("Authorization", "Bearer fixture-secret"),),
        timeout_seconds=9.0,
    )

    assert response.status_code == 200
    assert response.headers == (("x-stats", "one"), ("x-stats", "two"))
    assert response.body == b'{"data":{"total_cost":0.01}}'
    assert observed["url"] == "https://openrouter.ai/api/v1/generation?id=gen-1"
    assert observed["headers"] == {
        "Authorization": "Bearer fixture-secret",
        "User-Agent": "fixture-agent",
    }
    assert observed["timeout"] == 9.0
