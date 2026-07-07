"""Small artifact writers for dependency-light experiment outputs."""

from __future__ import annotations

import binascii
import html
import json
import struct
import zlib
from typing import NamedTuple, Sequence, TypeAlias
from pathlib import Path

import numpy as np

MetricsRow: TypeAlias = dict[str, int | float]
LineSeries: TypeAlias = tuple[str, str]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHART_COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ca8a04", "#0891b2")


class ReportImage(NamedTuple):
    """Image reference rendered in the HTML report."""

    title: str
    filename: str


class ReconstructionSnapshot(NamedTuple):
    """Paired reconstruction images captured after an epoch."""

    epoch: int
    reconstruction_filename: str
    masked_label_filename: str


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Append metrics rows to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_pgm_grid(path: Path, images: np.ndarray, columns: int = 8, pad: int = 2) -> None:
    """Write a grayscale PGM grid from [n, h, w] or [n, pixels] images in [0, 1]."""
    _write_pgm(path, _image_grid(images, columns, pad))


def write_png_grid(path: Path, images: np.ndarray, columns: int = 8, pad: int = 2) -> None:
    """Write a browser-viewable grayscale PNG grid from [n, h, w] or [n, pixels] images in [0, 1]."""
    _write_png(path, _image_grid(images, columns, pad))


def write_svg_line_chart(
    path: Path,
    rows: Sequence[MetricsRow],
    series: Sequence[LineSeries],
    title: str,
    y_label: str,
    width: int = 820,
    height: int = 320,
) -> None:
    """Write an SVG line chart for one or more metrics sharing an epoch axis."""
    if not rows:
        raise ValueError("cannot write a chart without metric rows")
    available_series = tuple((key, label) for key, label in series if all(key in row for row in rows))
    if not available_series:
        raise ValueError("cannot write a chart without matching metric series")
    x_values = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float32)
    y_values_by_key = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key, _ in available_series
    }
    y_values = np.concatenate(tuple(y_values_by_key.values()))
    margins = (64, 42, 24, 52)
    left, top, right, bottom = margins
    chart_width, chart_height = width - left - right, height - top - bottom
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = _expanded_range(float(np.min(y_values)), float(np.max(y_values)))
    point_x = lambda value: left + _range_fraction(float(value), x_min, x_max) * chart_width
    point_y = lambda value: top + (1.0 - _range_fraction(float(value), y_min, y_max)) * chart_height
    axis_markup = _chart_axes(title, y_label, width, height, margins, x_min, x_max, y_min, y_max)
    line_markup = "\n".join(
        _line_series_markup(
            color=CHART_COLORS[index % len(CHART_COLORS)],
            label=label,
            points=tuple((point_x(x_value), point_y(y_value)) for x_value, y_value in zip(x_values, y_values_by_key[key])),
            legend_y=top + 18 + index * 18,
            legend_x=left + 16,
        )
        for index, (key, label) in enumerate(available_series)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                "<style>text{font-family:Inter,Arial,sans-serif;fill:#111827} .muted{fill:#6b7280;font-size:12px}</style>",
                axis_markup,
                line_markup,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_svg_heatmap(
    path: Path,
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    value_format: str = ".3f",
) -> None:
    """Write a compact SVG heatmap with labeled rows and columns."""
    value_array = np.asarray(values, dtype=np.float32)
    if value_array.shape != (len(row_labels), len(column_labels)):
        raise ValueError(
            f"heatmap shape {value_array.shape} does not match {len(row_labels)} rows and {len(column_labels)} columns"
        )
    cell_width, cell_height = 112, 44
    left, top = 128, 58
    width = left + cell_width * len(column_labels) + 24
    height = top + cell_height * len(row_labels) + 36
    minimum, maximum = float(np.min(value_array)), float(np.max(value_array))
    cells = "\n".join(
        _heatmap_cell(
            x=left + col_index * cell_width,
            y=top + row_index * cell_height,
            width=cell_width,
            height=cell_height,
            value=float(value_array[row_index, col_index]),
            minimum=minimum,
            maximum=maximum,
            value_format=value_format,
        )
        for row_index in range(value_array.shape[0])
        for col_index in range(value_array.shape[1])
    )
    row_markup = "\n".join(
        f'<text class="muted" x="18" y="{top + row_index * cell_height + 27}">{html.escape(label)}</text>'
        for row_index, label in enumerate(row_labels)
    )
    column_markup = "\n".join(
        f'<text class="muted" x="{left + col_index * cell_width + 8}" y="{top - 14}">{html.escape(label)}</text>'
        for col_index, label in enumerate(column_labels)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                "<style>text{font-family:Inter,Arial,sans-serif;fill:#111827}.muted{fill:#4b5563;font-size:12px}</style>",
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="18" y="28" font-size="18" font-weight="700">{html.escape(title)}</text>',
                column_markup,
                row_markup,
                cells,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_html_report(
    path: Path,
    title: str,
    config_payload: dict[str, object],
    metrics_rows: Sequence[MetricsRow],
    chart_images: Sequence[ReportImage],
    reconstruction_snapshots: Sequence[ReconstructionSnapshot],
    sample_images: Sequence[ReportImage],
) -> None:
    """Write a self-contained HTML index that references generated charts and image grids."""
    if not metrics_rows:
        raise ValueError("cannot write a report without metric rows")
    final_metrics = metrics_rows[-1]
    html_body = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            _report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{html.escape(title)}</h1>",
            "<section>",
            "<h2>Final Metrics</h2>",
            _metrics_table(final_metrics),
            "</section>",
            "<section>",
            "<h2>Training Curves</h2>",
            _image_grid_markup(chart_images),
            "</section>",
            "<section>",
            "<h2>Reconstruction Snapshots</h2>",
            _snapshot_markup(reconstruction_snapshots),
            "</section>",
            "<section>",
            "<h2>Generated Samples</h2>",
            _image_grid_markup(sample_images),
            "</section>",
            "<section>",
            "<h2>Configuration</h2>",
            f"<pre>{html.escape(json.dumps(config_payload, indent=2, sort_keys=True))}</pre>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_body + "\n", encoding="utf-8")


def _image_grid(images: np.ndarray, columns: int, pad: int) -> np.ndarray:
    image_array = _image_batch(images)
    rows = int(np.ceil(image_array.shape[0] / columns))
    grid = np.zeros(
        (rows * image_array.shape[1] + (rows - 1) * pad, columns * image_array.shape[2] + (columns - 1) * pad),
        dtype=np.float32,
    )
    for index, image in enumerate(image_array):
        row, col = divmod(index, columns)
        row_start = row * (image_array.shape[1] + pad)
        col_start = col * (image_array.shape[2] + pad)
        grid[row_start : row_start + image_array.shape[1], col_start : col_start + image_array.shape[2]] = image
    return grid


def _image_batch(images: np.ndarray) -> np.ndarray:
    image_array = np.asarray(images, dtype=np.float32)
    if image_array.ndim == 2:
        side = int(np.sqrt(image_array.shape[1]))
        if side * side != image_array.shape[1]:
            raise ValueError(f"flat images must have square pixel count, got {image_array.shape[1]}")
        return image_array.reshape(image_array.shape[0], side, side)
    if image_array.ndim == 3:
        return image_array
    raise ValueError(f"expected [n, h, w] or [n, pixels] images, got shape {image_array.shape}")


def _write_pgm(path: Path, image: np.ndarray) -> None:
    scaled = np.clip(image, 0.0, 1.0)
    image_bytes = (scaled * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        output_file.write(f"P5\n{image_bytes.shape[1]} {image_bytes.shape[0]}\n255\n".encode("ascii"))
        output_file.write(image_bytes.tobytes())


def _write_png(path: Path, image: np.ndarray) -> None:
    scaled = np.clip(image, 0.0, 1.0)
    image_bytes = (scaled * 255.0).round().astype(np.uint8)
    scanlines = b"".join(b"\x00" + image_bytes[row_index].tobytes() for row_index in range(image_bytes.shape[0]))
    chunks = (
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", image_bytes.shape[1], image_bytes.shape[0], 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(scanlines)),
        _png_chunk(b"IEND", b""),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_SIGNATURE + b"".join(chunks))


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", binascii.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _expanded_range(minimum: float, maximum: float) -> tuple[float, float]:
    spread = maximum - minimum
    padding = spread * 0.06 if spread > 0.0 else max(abs(maximum) * 0.05, 1.0)
    return minimum - padding, maximum + padding


def _range_fraction(value: float, minimum: float, maximum: float) -> float:
    return 0.5 if minimum == maximum else (value - minimum) / (maximum - minimum)


def _chart_axes(
    title: str,
    y_label: str,
    width: int,
    height: int,
    margins: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    left, top, right, bottom = margins
    chart_width, chart_height = width - left - right, height - top - bottom
    return "\n".join(
        [
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="{left}" y="24" font-size="18" font-weight="700">{html.escape(title)}</text>',
            f'<line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="#9ca3af"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#9ca3af"/>',
            f'<text class="muted" x="{left}" y="{height - 16}">epoch {x_min:g}</text>',
            f'<text class="muted" x="{width - right - 56}" y="{height - 16}">epoch {x_max:g}</text>',
            f'<text class="muted" x="12" y="{top + 4}">{y_max:.4g}</text>',
            f'<text class="muted" x="12" y="{top + chart_height}">{y_min:.4g}</text>',
            f'<text class="muted" x="{left}" y="{height - 34}">{html.escape(y_label)}</text>',
        ]
    )


def _line_series_markup(color: str, label: str, points: tuple[tuple[float, float], ...], legend_x: int, legend_y: int) -> str:
    point_markup = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    markers = "\n".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in points)
    return "\n".join(
        [
            f'<polyline points="{point_markup}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>',
            markers,
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>',
            f'<text class="muted" x="{legend_x + 16}" y="{legend_y}">{html.escape(label)}</text>',
        ]
    )


def _heatmap_cell(
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    minimum: float,
    maximum: float,
    value_format: str,
) -> str:
    intensity = _range_fraction(value, minimum, maximum)
    fill = _heatmap_color(intensity)
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="#ffffff"/>',
            f'<text x="{x + 8}" y="{y + 27}" font-size="13">{html.escape(format(value, value_format))}</text>',
        ]
    )


def _heatmap_color(intensity: float) -> str:
    clipped = max(0.0, min(1.0, intensity))
    red = int(239 - clipped * 202)
    green = int(246 - clipped * 147)
    blue = int(255 - clipped * 20)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _metrics_table(metrics: MetricsRow) -> str:
    preferred_keys = (
        "epoch",
        "train_loss",
        "train_eval_loss",
        "test_loss",
        "train_eval_label_patch_accuracy",
        "test_label_patch_accuracy",
        "train_eval_energy_classifier_accuracy",
        "test_energy_classifier_accuracy",
        "test_reconstruction_bce",
        "test_digit_bce",
        "test_label_patch_bce",
        "test_kl",
    )
    ordered_keys = tuple(key for key in preferred_keys if key in metrics) + tuple(
        key for key in sorted(metrics) if key not in preferred_keys
    )
    rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{_format_metric(metrics[key])}</td></tr>"
        for key in ordered_keys
    )
    return f"<table>{rows}</table>"


def _format_metric(value: int | float) -> str:
    return str(value) if isinstance(value, int) else f"{value:.6g}"


def _image_grid_markup(images: Sequence[ReportImage]) -> str:
    return '<div class="image-grid">' + "\n".join(_image_card(image) for image in images) + "</div>"


def _image_card(image: ReportImage) -> str:
    return (
        '<figure class="image-card">'
        f'<img src="{html.escape(image.filename)}" alt="{html.escape(image.title)}">'
        f"<figcaption>{html.escape(image.title)}</figcaption>"
        "</figure>"
    )


def _snapshot_markup(snapshots: Sequence[ReconstructionSnapshot]) -> str:
    snapshot_cards = "\n".join(
        '<article class="snapshot">'
        f"<h3>Epoch {snapshot.epoch}</h3>"
        + _image_grid_markup(
            (
                ReportImage("Full reconstruction", snapshot.reconstruction_filename),
                ReportImage("Masked-label reconstruction", snapshot.masked_label_filename),
            )
        )
        + "</article>"
        for snapshot in snapshots
    )
    return f'<div class="snapshots">{snapshot_cards}</div>'


def _report_css() -> str:
    return """
body { margin: 0; background: #f8fafc; color: #111827; font: 15px/1.45 Inter, Arial, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }
h1 { margin: 0 0 24px; font-size: 28px; }
h2 { margin: 0 0 14px; font-size: 20px; }
h3 { margin: 0 0 10px; font-size: 16px; }
section { margin: 0 0 28px; }
table { border-collapse: collapse; width: 100%; background: #ffffff; }
th, td { border: 1px solid #d1d5db; padding: 7px 9px; text-align: left; }
th { width: 260px; background: #f3f4f6; font-weight: 650; }
pre { overflow-x: auto; background: #111827; color: #f9fafb; padding: 16px; border-radius: 6px; }
.image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
.image-card { margin: 0; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; }
.image-card img { display: block; width: 100%; height: auto; image-rendering: pixelated; }
.image-card figcaption { margin-top: 8px; color: #374151; font-size: 13px; }
.snapshots { display: grid; gap: 18px; }
.snapshot { border-top: 1px solid #d1d5db; padding-top: 16px; }
""".strip()
