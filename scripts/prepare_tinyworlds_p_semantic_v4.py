#!/usr/bin/env python3
"""Reuse pinned evidence and run the frozen semantic-v4 centroid screen."""

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
        V4_REUSED_EVIDENCE_SHA256,
        V4_SEMANTIC_CONFIG,
        load_semantic_evidence,
    )

    root = (
        SEMANTIC_DATA_ROOT
        / "evidence"
        / "v1"
        / V4_REUSED_EVIDENCE_SHA256
    )
    evidence = load_semantic_evidence(root)
    if (
        evidence.evidence_sha256 != V4_REUSED_EVIDENCE_SHA256
        or evidence.archive_identity != CANONICAL_ARCHIVE_IDENTITY
        or evidence.encoder_identity.identity_sha256
        != ENCODER_SNAPSHOT_IDENTITY_SHA256
        or evidence.config.evidence_record() != V4_SEMANTIC_CONFIG.evidence_record()
    ):
        raise RuntimeError("cached encoder evidence does not match semantic-v4")
    return evidence


def _existing_catalog(evidence):
    from apm.data.text.tinyworlds_p_semantic import (
        V4_SEMANTIC_CONFIG,
        load_v4_semantic_catalog,
    )

    root = SEMANTIC_DATA_ROOT / "catalog" / "v4"
    candidate_paths = (
        tuple(
            path
            for path in sorted(root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
            and json.loads((path / "catalog.json").read_bytes()).get("config")
            == V4_SEMANTIC_CONFIG.as_record()
        )
        if root.is_dir()
        else ()
    )
    candidates = tuple(load_v4_semantic_catalog(path) for path in candidate_paths)
    matches = tuple(
        item
        for item in candidates
        if item.evidence_sha256 == evidence.evidence_sha256
        and item.encoder_identity == evidence.encoder_identity
        and item.config == V4_SEMANTIC_CONFIG
    )
    if len(matches) > 1:
        raise RuntimeError("multiple catalogs have the frozen semantic-v4 identity")
    return matches[0] if matches else None


def _existing_failure(evidence):
    from apm.data.text.tinyworlds_p_semantic import (
        V4_SEMANTIC_CONFIG,
        load_v4_catalog_failure,
    )

    root = SEMANTIC_DATA_ROOT / "catalog" / "v4" / "failures"
    matches = []
    if root.is_dir():
        for path in sorted(root.glob("[0-9a-f]" * 64)):
            if not (path / "tree.json").is_file():
                continue
            record = json.loads((path / "failure.json").read_bytes())
            if record.get("config") != V4_SEMANTIC_CONFIG.as_record():
                continue
            artifact = load_v4_catalog_failure(path)
            if artifact.evidence_sha256 == evidence.evidence_sha256:
                matches.append(artifact)
    if len(matches) > 1:
        raise RuntimeError("multiple failures have the frozen semantic-v4 identity")
    return matches[0] if matches else None


def _report_failure(failure) -> int:
    print(f"failure audit: {failure.root}")
    print(f"failure SHA-256: {failure.failure_sha256}")
    print(f"automated stop: {failure.reason}")
    print("partition/training/checkpoint/sealed test: not authorized")
    return 2


def main() -> int:
    from apm.data.text.tinyworlds_p_semantic import (
        V4_SEMANTIC_CONFIG,
        V4SemanticGridError,
        build_v4_catalog_from_evidence,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import SemanticProgressReporter

    evidence = _load_evidence()
    print(f"strict reused encoder evidence: {evidence.root}", flush=True)
    catalog = _existing_catalog(evidence)
    if catalog is not None:
        print(f"strict existing semantic-v4 catalog: {catalog.root}")
        print(f"catalog SHA-256: {catalog.catalog_sha256}")
        print(f"retained archive token mass: {catalog.retained_token_fraction:.3%}")
        return 0
    failure = _existing_failure(evidence)
    if failure is not None:
        return _report_failure(failure)

    work_root = SEMANTIC_DATA_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-construction-v4-", dir=work_root))
    print(f"temporary artifact directory: {working}", flush=True)
    reporter = SemanticProgressReporter(
        "TinyWorlds-P semantic-v4 catalog",
        {
            "evidence-load": 20.0,
            "role-scores": 15.0,
            "calibration": 5.0,
            "screening": 20.0,
            "clustering": 20.0,
            "publication": 20.0,
        },
    )
    try:
        try:
            catalog = build_v4_catalog_from_evidence(
                evidence,
                SEMANTIC_DATA_ROOT / "catalog" / "v4",
                working / "catalog-publication",
                V4_SEMANTIC_CONFIG,
                progress=reporter,
            )
        except V4SemanticGridError:
            failure = _existing_failure(evidence)
            if failure is None:
                raise RuntimeError("semantic-v4 stopped without an authenticated audit")
            return _report_failure(failure)
    finally:
        reporter.close()
    print(f"catalog: {catalog.root}")
    print(f"catalog SHA-256: {catalog.catalog_sha256}")
    print(f"retained archive token mass: {catalog.retained_token_fraction:.3%}")
    print(f"Markdown audit: {catalog.root / 'audit.md'}")
    print(f"HTML audit: {catalog.root / 'audit.html'}")
    print("partition remains a separately frozen and verified next stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
