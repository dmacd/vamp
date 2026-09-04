from __future__ import annotations

from pathlib import Path

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf


def test_pdf_renderer_reuses_an_authenticated_matching_render(tmp_path: Path) -> None:
    html = atomic_write(tmp_path / "REPORT.html", b"<html><body>result</body></html>")
    pdf = atomic_write(tmp_path / "REPORT.pdf", b"%PDF-1.4\n%%EOF\n")
    identity_core: dict[str, object] = {
        "generator": "headless Chrome print-to-PDF",
        "html_sha256": file_sha256(html),
        "pdf_sha256": file_sha256(pdf),
        "schema_version": "imagenetr50-integrator-pdf-v1",
        "size_bytes": pdf.stat().st_size,
    }
    atomic_write(
        tmp_path / "REPORT.pdf.json",
        canonical_json_bytes(
            {**identity_core, "content_hash": record_sha256(identity_core)}
        ),
    )

    rendered = render_integrator_pdf(html)

    assert rendered == pdf
    manifest = load_canonical_json(tmp_path / "report_manifest.json")
    assert {row["path"] for row in manifest["files"]} == {
        "REPORT.html",
        "REPORT.pdf",
        "REPORT.pdf.json",
    }
