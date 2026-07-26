#!/usr/bin/env python3
"""Publish and validation-only reload the approved five-world main catalog."""

from apm.data.text.tinyworlds_q_semantic.registered_main_catalog import (
    publish_registered_main_catalog,
)


def main() -> int:
    """Authenticate the review chain and publish its exact catalog tree."""
    _frozen, catalog, validation = publish_registered_main_catalog()
    print(f"Main catalog: {catalog.catalog_sha256}", flush=True)
    print(f"Approved facts: {len(catalog.facts)}", flush=True)
    print(f"Validation queries loaded: {len(validation.templates)}", flush=True)
    print(f"Sealed test queries: {len(catalog.facts) * 5}", flush=True)
    print("The sealed test file was authenticated but not opened.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
