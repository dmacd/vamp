"""Print the standalone integrator HTML report to an atomic PDF artifact."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_reporting import write_report_manifest


def render_integrator_pdf(html_path: str | Path) -> Path:
    """Render the self-contained HTML with headless Chrome and validate PDF structure."""
    html = Path(html_path).resolve()
    if not html.is_file():
        raise FileNotFoundError(f"integrator HTML report is missing: {html}")
    output = html.with_suffix(".pdf")
    identity_path = output.with_suffix(".pdf.json")
    html_sha256 = file_sha256(html)
    if output.is_file() and identity_path.is_file():
        identity = load_canonical_json(identity_path)
        identity_core = {
            key: value for key, value in identity.items() if key != "content_hash"
        }
        if (
            identity.get("content_hash") == record_sha256(identity_core)
            and identity.get("html_sha256") == html_sha256
            and identity.get("pdf_sha256") == file_sha256(output)
        ):
            write_report_manifest(html.parent)
            return output
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Google Chrome or Chromium is required to render the PDF report")
    temporary_root = Path.cwd() / "tmp" / "pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root / f"{html.parent.parent.name}-{html_sha256[:16]}.pdf"
    print(f"PDF intermediate directory: {temporary_root}", flush=True)
    completed = subprocess.run(
        (
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={temporary}",
            html.as_uri(),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not temporary.is_file():
        raise RuntimeError(
            "headless Chrome failed to render the integrator PDF: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = temporary.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-1024:]:
        raise ValueError("rendered integrator report is not a complete PDF")
    output = atomic_write(output, payload)
    identity_core: dict[str, object] = {
        "generator": "headless Chrome print-to-PDF",
        "html_sha256": html_sha256,
        "pdf_sha256": file_sha256(output),
        "schema_version": "imagenetr50-integrator-pdf-v1",
        "size_bytes": output.stat().st_size,
    }
    atomic_write(
        identity_path,
        canonical_json_bytes(
            {**identity_core, "content_hash": record_sha256(identity_core)}
        ),
    )
    write_report_manifest(html.parent)
    return output


__all__ = ["render_integrator_pdf"]
