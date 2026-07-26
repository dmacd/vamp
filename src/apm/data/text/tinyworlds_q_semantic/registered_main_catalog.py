"""Exact published identity of the approved five-world main catalog."""

from __future__ import annotations

from pathlib import Path

from apm.data.text.tinyworlds_q_semantic.catalog import (
    ValidationCatalogView,
    load_validation_catalog,
    publish_catalog,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    CATALOG_ROOT,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.main_catalog import (
    build_approved_main_catalog,
)
from apm.data.text.tinyworlds_q_semantic.main_freeze import MainExperimentFreeze
from apm.data.text.tinyworlds_q_semantic.registered_main_review import (
    TOKENIZER_PATH,
    load_registered_main_complete_review_authority,
)
from apm.lm.text import TokenizersTextTokenizer


MAIN_CATALOG_SHA256 = (
    "0ffd78e81d1da4a4fbd20b49bc02f3dec94560085f4490a357c7f73239f9e8ba"
)


def publish_registered_main_catalog(
    output_root: str | Path = CATALOG_ROOT,
) -> tuple[MainExperimentFreeze, SemanticQueryCatalog, ValidationCatalogView]:
    """Rebuild, publish, and validation-only reload the exact main catalog."""
    frozen, packet, shortlist, primary, reverse_review, reverse = (
        load_registered_main_complete_review_authority()
    )
    tokenizer = TokenizersTextTokenizer.from_file(TOKENIZER_PATH)
    catalog = build_approved_main_catalog(
        review_packet=packet,
        shortlist=shortlist,
        primary_approval=primary,
        reverse_review=reverse_review,
        reverse_approval=reverse,
        tokenizer=tokenizer,
    )
    if catalog.catalog_sha256 != MAIN_CATALOG_SHA256:
        raise RuntimeError("registered main catalog identity changed")
    validation = load_validation_catalog(publish_catalog(catalog, output_root))
    if validation.catalog_sha256 != catalog.catalog_sha256:
        raise RuntimeError("registered main validation-only reload changed")
    return frozen, catalog, validation


__all__ = ["MAIN_CATALOG_SHA256", "publish_registered_main_catalog"]
