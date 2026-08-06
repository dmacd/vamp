#!/usr/bin/env python3
"""Build the forward-only presentation from the completed sealed main result."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    canonical_json_bytes,
)
from apm.data.text.tinyworlds_q_semantic.execution import (
    completed_sealed_result_sha256,
    load_sealed_transaction,
)
from apm.data.text.tinyworlds_q_semantic.interactive_report import (
    build_interactive_report_data,
    publish_interactive_report,
)
from apm.data.text.tinyworlds_q_semantic.manifests import MAIN_CONCEPT_IDS
from apm.data.text.tinyworlds_q_semantic.report import load_semantic_report
from apm.data.text.tinyworlds_q_semantic.result_stream import (
    load_semantic_result_ledger,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPOSITORY_ROOT / "results" / "language_cl" / "tinyworlds-q-semantic-v1"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT / "docs" / "TINYWORLDS_Q_SEMANTIC_V1_INTERACTIVE_REPORT.html"
)


def main() -> int:
    """Authenticate frozen inputs and deterministically publish one standalone page."""
    preset = QueryExperimentPreset(MAIN_CONCEPT_IDS)
    transaction = load_sealed_transaction(RESULT_ROOT / "main-sealed-transaction")
    completed_report_sha256 = completed_sealed_result_sha256(transaction)
    if completed_report_sha256 is None:
        raise RuntimeError("the main sealed transaction has not completed")
    if transaction.config_sha256 != preset.config_sha256:
        raise ValueError("completed transaction does not use the registered main preset")

    report_root = (
        RESULT_ROOT
        / "main-report"
        / preset.config_sha256[:16]
        / completed_report_sha256
    )
    result_record = _canonical_record(report_root / "result.json")
    source_bindings = {
        "catalog_sha256": _required_sha256(result_record, "catalog_sha256"),
        "partition_sha256": _required_sha256(result_record, "partition_sha256"),
        "selected_base_sha256": _required_sha256(
            result_record,
            "selected_base_sha256",
        ),
        "adapters_sha256": _required_sha256(result_record, "adapters_sha256"),
        "preflight_sha256": _required_sha256(result_record, "preflight_sha256"),
        "transaction_sha256": _required_sha256(
            result_record,
            "transaction_sha256",
        ),
    }
    transaction_bindings = (
        transaction.catalog_sha256,
        transaction.partition_sha256,
        transaction.selected_base_sha256,
        transaction.adapters_sha256,
        transaction.transaction_sha256,
    )
    report_transaction_bindings = (
        source_bindings["catalog_sha256"],
        source_bindings["partition_sha256"],
        source_bindings["selected_base_sha256"],
        source_bindings["adapters_sha256"],
        source_bindings["transaction_sha256"],
    )
    if report_transaction_bindings != transaction_bindings:
        raise ValueError("completed report does not match its sealed transaction")

    report = load_semantic_report(
        report_root,
        **source_bindings,
        preset=preset,
    )
    results = load_semantic_result_ledger(report.root / "results.jsonl")
    audit_path = transaction.root / "sealed-catalog-audit.md"
    audit_markdown = audit_path.read_text(encoding="utf-8")
    data = build_interactive_report_data(
        audit_markdown,
        results,
        result_record,
        MAIN_CONCEPT_IDS,
    )
    if data.report_sha256 != completed_report_sha256:
        raise ValueError("interactive report source identity changed")
    published = publish_interactive_report(data, OUTPUT_PATH)
    output_sha256 = sha256(published.read_bytes()).hexdigest()
    print(f"Interactive report: {published}")
    print(f"HTML SHA-256: {output_sha256}")
    print(
        f"Included: {data.question_count} forward questions, "
        f"{data.sample_count} fixed examples, {data.fact_count} reviewed facts, "
        f"{len(data.methods)} methods; excluded "
        f"{data.excluded_reverse_question_count} reverse questions"
    )
    return 0


def _canonical_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid report record: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"report record is not canonical: {path}")
    return value


def _required_sha256(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"report {field} is not a SHA-256 identity")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
