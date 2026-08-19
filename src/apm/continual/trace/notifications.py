"""Non-fatal webhook notifications for long-running TRACE jobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from urllib.request import Request, urlopen


def notify(
    event: str,
    run_hash: str,
    details: Mapping[str, object],
    request_sender: Callable[[Request, float], int] | None = None,
) -> bool:
    """Post a redacted JSON event, returning false when no webhook or delivery fails."""
    webhook = os.environ.get("VAMP_NOTIFY_WEBHOOK_URL")
    if not webhook:
        return False
    payload = json.dumps(
        {
            "details": dict(details),
            "event": event,
            "format": "trace-notification-v1",
            "run_hash": run_hash,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    request = Request(
        webhook,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        status = (request_sender or _send_request)(request, 15.0)
        return 200 <= status < 300
    except Exception as error:
        print(f"TRACE notification failed ({type(error).__name__}); experiment continues")
        return False


def _send_request(request: Request, timeout: float) -> int:
    with urlopen(request, timeout=timeout) as response:
        return int(response.status)


__all__ = ["notify"]
