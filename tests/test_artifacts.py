from __future__ import annotations

import numpy as np

from apm.training.artifacts import (
    ReconstructionSnapshot,
    ReportImage,
    write_html_report,
    write_png_grid,
    write_svg_heatmap,
    write_svg_line_chart,
)


def test_write_png_grid_creates_browser_viewable_png(tmp_path) -> None:
    output_path = tmp_path / "grid.png"
    images = np.stack((np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)), axis=0)

    write_png_grid(output_path, images, columns=2, pad=1)

    png_bytes = output_path.read_bytes()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png_bytes
    assert b"IDAT" in png_bytes


def test_write_svg_line_chart_renders_metric_series(tmp_path) -> None:
    output_path = tmp_path / "loss.svg"
    rows = [
        {"epoch": 1, "train_loss": 4.0, "test_loss": 5.0},
        {"epoch": 2, "train_loss": 3.0, "test_loss": 4.0},
    ]

    write_svg_line_chart(
        output_path,
        rows,
        (("train_loss", "train"), ("test_loss", "test")),
        "Loss",
        "loss",
    )

    svg_text = output_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "train" in svg_text
    assert "test" in svg_text


def test_write_svg_heatmap_renders_labels_and_values(tmp_path) -> None:
    output_path = tmp_path / "heatmap.svg"

    write_svg_heatmap(output_path, np.asarray([[0.25, 0.75]], dtype=np.float32), ("row_a",), ("col_a", "col_b"), "Heat")

    svg_text = output_path.read_text(encoding="utf-8")
    assert "row_a" in svg_text
    assert "col_b" in svg_text
    assert "0.750" in svg_text


def test_write_html_report_links_charts_snapshots_and_samples(tmp_path) -> None:
    output_path = tmp_path / "report.html"

    write_html_report(
        output_path,
        "Run Report",
        {"train": {"epochs": 2}},
        ({"epoch": 2, "train_loss": 3.0, "test_label_patch_accuracy": 0.75},),
        chart_images=(ReportImage("Loss", "loss.svg"),),
        reconstruction_snapshots=(ReconstructionSnapshot(1, "recon_epoch_001.png", "masked_epoch_001.png"),),
        sample_images=(ReportImage("Samples", "sample_grid.png"),),
    )

    report_text = output_path.read_text(encoding="utf-8")
    assert "Final Metrics" in report_text
    assert "loss.svg" in report_text
    assert "recon_epoch_001.png" in report_text
    assert "sample_grid.png" in report_text
    assert "report-lightbox" in report_text
    assert 'data-lightbox-src="loss.svg"' in report_text
