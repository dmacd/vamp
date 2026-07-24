#!/usr/bin/env python3
"""Reuse pinned MiniLM evidence and construct the frozen semantic-v2 grid."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"


def _load_evidence():
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        ENCODER_SNAPSHOT_IDENTITY_SHA256,
        V2_REUSED_EVIDENCE_SHA256,
        V2_SEMANTIC_CONFIG,
        load_semantic_evidence,
    )

    root = (
        SEMANTIC_DATA_ROOT
        / "evidence"
        / "v1"
        / V2_REUSED_EVIDENCE_SHA256
    )
    evidence = load_semantic_evidence(root)
    if (
        evidence.evidence_sha256 != V2_REUSED_EVIDENCE_SHA256
        or evidence.archive_identity != CANONICAL_ARCHIVE_IDENTITY
        or evidence.encoder_identity.identity_sha256
        != ENCODER_SNAPSHOT_IDENTITY_SHA256
        or evidence.config.evidence_record() != V2_SEMANTIC_CONFIG.evidence_record()
    ):
        raise RuntimeError("cached encoder evidence does not match semantic-v2")
    return evidence


def _existing_catalog(evidence):
    from apm.data.text.tinyworlds_p_semantic import (
        V2_SEMANTIC_CONFIG,
        load_v2_semantic_catalog,
    )

    root = SEMANTIC_DATA_ROOT / "catalog" / "v2"
    candidates = (
        tuple(
            load_v2_semantic_catalog(path)
            for path in sorted(root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.evidence_sha256 == evidence.evidence_sha256
        and item.encoder_identity == evidence.encoder_identity
        and item.config == V2_SEMANTIC_CONFIG
    )
    if len(matches) > 1:
        raise RuntimeError("multiple catalogs have the frozen semantic-v2 identity")
    return matches[0] if matches else None


def _existing_failure(evidence):
    from apm.data.text.tinyworlds_p_semantic import (
        V2_SEMANTIC_CONFIG,
        load_v2_catalog_failure,
    )

    root = SEMANTIC_DATA_ROOT / "catalog" / "v2" / "failures"
    matches = []
    if root.is_dir():
        for path in sorted(root.glob("[0-9a-f]" * 64)):
            if not (path / "tree.json").is_file():
                continue
            artifact = load_v2_catalog_failure(path)
            record = json.loads((path / "failure.json").read_bytes())
            if (
                artifact.evidence_sha256 == evidence.evidence_sha256
                and record.get("config") == V2_SEMANTIC_CONFIG.as_record()
            ):
                matches.append(artifact)
    if len(matches) > 1:
        raise RuntimeError("multiple failures have the frozen semantic-v2 identity")
    return matches[0] if matches else None


def _report_failure(failure) -> int:
    print(f"failure audit: {failure.root}")
    print(f"failure SHA-256: {failure.failure_sha256}")
    print(f"automated stop: {failure.reason}")
    print("catalog/partition/training/sealed test: not authorized")
    return 2


def main() -> int:
    from apm.data.text.tinyworlds_p_semantic import (
        V2_SEMANTIC_CONFIG,
        V2SemanticGridError,
        build_v2_catalog_from_evidence,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import SemanticProgressReporter

    evidence = _load_evidence()
    print(f"strict reused encoder evidence: {evidence.root}", flush=True)
    catalog = _existing_catalog(evidence)
    if catalog is not None:
        print(f"strict existing semantic-v2 catalog: {catalog.root}")
        print(f"catalog SHA-256: {catalog.catalog_sha256}")
        print(f"retained archive token mass: {catalog.retained_token_fraction:.3%}")
        return 0
    failure = _existing_failure(evidence)
    if failure is not None:
        return _report_failure(failure)

    work_root = SEMANTIC_DATA_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-construction-v2-", dir=work_root))
    print(f"temporary artifact directory: {working}", flush=True)
    reporter = SemanticProgressReporter(
        "TinyWorlds-P semantic-v2 catalog",
        {
            "evidence-load": 20.0,
            "role-scores": 15.0,
            "calibration": 5.0,
            "screening": 20.0,
            "clustering": 30.0,
            "publication": 10.0,
        },
    )
    try:
        try:
            catalog = build_v2_catalog_from_evidence(
                evidence,
                SEMANTIC_DATA_ROOT / "catalog" / "v2",
                working / "catalog-publication",
                V2_SEMANTIC_CONFIG,
                progress=reporter,
            )
        except V2SemanticGridError:
            failure = _existing_failure(evidence)
            if failure is None:
                raise RuntimeError("semantic-v2 stopped without an authenticated audit")
            return _report_failure(failure)
    finally:
        reporter.close()
    print(f"catalog: {catalog.root}")
    print(f"catalog SHA-256: {catalog.catalog_sha256}")
    print(f"retained archive token mass: {catalog.retained_token_fraction:.3%}")
    print(f"Markdown audit: {catalog.root / 'audit.md'}")
    print(f"HTML audit: {catalog.root / 'audit.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
