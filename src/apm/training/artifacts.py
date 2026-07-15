"""Small artifact writers for dependency-light experiment outputs."""

from __future__ import annotations

import binascii
import html
import json
import os
import struct
import tempfile
import zlib
from typing import NamedTuple, Sequence, TypeAlias
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"))

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
    x_label: str = "stage",
) -> None:
    """Write an SVG line chart for one or more metrics sharing an epoch axis."""
    if not rows:
        raise ValueError("cannot write a chart without metric rows")
    available_series = tuple((key, label) for key, label in series if all(key in row for row in rows))
    if not available_series:
        raise ValueError("cannot write a chart without matching metric series")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise ImportError("matplotlib is required to write report line charts") from exc

    x_values = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float64)
    fig_width = max(width / 96.0, 8.5)
    fig_height = max(height / 96.0, 3.8)
    figure, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    for key, label in available_series:
        y_values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        axis.plot(x_values, y_values, marker="o", linewidth=2.0, markersize=4.5, label=label)

    axis.set_title(title, fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="major", color="#d1d5db", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=9)
    if "accuracy" in y_label.lower():
        all_y = np.concatenate([np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key, _ in available_series])
        if np.all((0.0 <= all_y) & (all_y <= 1.0)):
            axis.set_ylim(-0.02, 1.02)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
            report_lightbox_markup(),
            "<script>",
            report_lightbox_script(),
            "</script>",
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
        f'<figure class="image-card" role="button" tabindex="0" data-lightbox-src="{html.escape(image.filename)}" '
        f'data-lightbox-caption="{html.escape(image.title)}">'
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


def report_lightbox_markup() -> str:
    """Return the shared zoomable report lightbox markup."""
    return "\n".join(
        [
            '<div id="report-lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="Image viewer" hidden>',
            '<figure class="lightbox-figure">',
            '<div class="lightbox-toolbar">',
            '<div class="lightbox-zoom-controls">',
            '<button type="button" class="lightbox-tool" data-lightbox-action="zoom-out" aria-label="Zoom out" title="Zoom out">-</button>',
            '<output id="report-lightbox-zoom" class="lightbox-zoom" aria-live="polite">Fit</output>',
            '<button type="button" class="lightbox-tool" data-lightbox-action="zoom-in" aria-label="Zoom in" title="Zoom in">+</button>',
            '<button type="button" class="lightbox-tool lightbox-fit" data-lightbox-action="fit" aria-label="Fit to window" title="Fit to window">Fit</button>',
            '<a id="report-lightbox-open" class="lightbox-tool lightbox-open" target="_blank" rel="noopener" title="Open original image">Open</a>',
            '</div>',
            '<button type="button" class="lightbox-tool lightbox-close" aria-label="Close" title="Close">x</button>',
            '</div>',
            '<div id="report-lightbox-viewport" class="lightbox-viewport" tabindex="0">',
            '<img id="report-lightbox-image" alt="" draggable="false">',
            '</div>',
            '<figcaption id="report-lightbox-caption"></figcaption>',
            "</figure>",
            "</div>",
        ]
    )


def report_lightbox_script() -> str:
    """Return shared zoom, pan, and keyboard behavior for report images."""
    return r"""
const lightbox = document.getElementById("report-lightbox");
const lightboxImage = document.getElementById("report-lightbox-image");
const lightboxCaption = document.getElementById("report-lightbox-caption");
const lightboxViewport = document.getElementById("report-lightbox-viewport");
const lightboxZoom = document.getElementById("report-lightbox-zoom");
const openButton = document.getElementById("report-lightbox-open");
const closeButton = lightbox.querySelector(".lightbox-close");
const zoomInButton = lightbox.querySelector('[data-lightbox-action="zoom-in"]');
const zoomOutButton = lightbox.querySelector('[data-lightbox-action="zoom-out"]');
const fitButton = lightbox.querySelector('[data-lightbox-action="fit"]');
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.25;
let zoom = 1;
let fitScale = 1;
let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragScrollLeft = 0;
let dragScrollTop = 0;

function imageSize() {
  return {
    width: lightboxImage.naturalWidth || 1,
    height: lightboxImage.naturalHeight || 1,
  };
}

function calculateFitScale() {
  const size = imageSize();
  const availableWidth = Math.max(1, lightboxViewport.clientWidth - 32);
  const availableHeight = Math.max(1, lightboxViewport.clientHeight - 32);
  fitScale = Math.min(availableWidth / size.width, availableHeight / size.height);
}

function renderZoom(preserveCenter = true) {
  const oldWidth = Math.max(1, lightboxViewport.scrollWidth);
  const oldHeight = Math.max(1, lightboxViewport.scrollHeight);
  const centerX = (lightboxViewport.scrollLeft + lightboxViewport.clientWidth / 2) / oldWidth;
  const centerY = (lightboxViewport.scrollTop + lightboxViewport.clientHeight / 2) / oldHeight;
  const size = imageSize();
  lightboxImage.style.width = `${Math.max(1, size.width * fitScale * zoom)}px`;
  lightboxZoom.value = zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`;
  lightboxZoom.textContent = lightboxZoom.value;
  zoomOutButton.disabled = zoom <= MIN_ZOOM;
  zoomInButton.disabled = zoom >= MAX_ZOOM;
  lightboxViewport.classList.toggle("is-zoomed", zoom > 1);
  if (preserveCenter) {
    requestAnimationFrame(() => {
      lightboxViewport.scrollLeft = centerX * lightboxViewport.scrollWidth - lightboxViewport.clientWidth / 2;
      lightboxViewport.scrollTop = centerY * lightboxViewport.scrollHeight - lightboxViewport.clientHeight / 2;
    });
  }
}

function setZoom(nextZoom, preserveCenter = true) {
  zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
  renderZoom(preserveCenter);
}

function fitImage() {
  zoom = 1;
  calculateFitScale();
  renderZoom(false);
  lightboxViewport.scrollLeft = 0;
  lightboxViewport.scrollTop = 0;
}

function openLightbox(card) {
  const src = card.dataset.lightboxSrc;
  const caption = card.dataset.lightboxCaption || src;
  lightboxImage.alt = caption;
  lightboxCaption.textContent = caption;
  openButton.href = src;
  lightbox.hidden = false;
  document.body.classList.add("modal-open");
  lightboxImage.src = src;
  if (lightboxImage.complete) {
    fitImage();
  }
  lightboxViewport.focus();
}

function closeLightbox() {
  lightbox.hidden = true;
  document.body.classList.remove("modal-open");
  lightboxImage.removeAttribute("src");
  lightboxImage.removeAttribute("style");
  openButton.removeAttribute("href");
  dragging = false;
}

lightboxImage.addEventListener("load", fitImage);
document.querySelectorAll("[data-lightbox-src]").forEach((card) => {
  card.addEventListener("click", () => openLightbox(card));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLightbox(card);
    }
  });
});

zoomInButton.addEventListener("click", () => setZoom(zoom * ZOOM_STEP));
zoomOutButton.addEventListener("click", () => setZoom(zoom / ZOOM_STEP));
fitButton.addEventListener("click", fitImage);
closeButton.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});
lightboxViewport.addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.metaKey) {
    return;
  }
  event.preventDefault();
  setZoom(event.deltaY < 0 ? zoom * ZOOM_STEP : zoom / ZOOM_STEP);
}, { passive: false });
lightboxViewport.addEventListener("dblclick", () => setZoom(zoom === 1 ? 2 : 1));
lightboxViewport.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || zoom <= 1) {
    return;
  }
  dragging = true;
  dragStartX = event.clientX;
  dragStartY = event.clientY;
  dragScrollLeft = lightboxViewport.scrollLeft;
  dragScrollTop = lightboxViewport.scrollTop;
  lightboxViewport.classList.add("is-dragging");
  lightboxViewport.setPointerCapture(event.pointerId);
});
lightboxViewport.addEventListener("pointermove", (event) => {
  if (!dragging) {
    return;
  }
  lightboxViewport.scrollLeft = dragScrollLeft - (event.clientX - dragStartX);
  lightboxViewport.scrollTop = dragScrollTop - (event.clientY - dragStartY);
});
function finishDrag(event) {
  dragging = false;
  lightboxViewport.classList.remove("is-dragging");
  if (lightboxViewport.hasPointerCapture(event.pointerId)) {
    lightboxViewport.releasePointerCapture(event.pointerId);
  }
}
lightboxViewport.addEventListener("pointerup", finishDrag);
lightboxViewport.addEventListener("pointercancel", finishDrag);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) {
    closeLightbox();
  } else if (!lightbox.hidden && (event.key === "+" || event.key === "=")) {
    event.preventDefault();
    setZoom(zoom * ZOOM_STEP);
  } else if (!lightbox.hidden && event.key === "-") {
    event.preventDefault();
    setZoom(zoom / ZOOM_STEP);
  } else if (!lightbox.hidden && event.key === "0") {
    event.preventDefault();
    fitImage();
  }
});
""".strip()


def report_lightbox_css() -> str:
    """Return shared styling for the zoomable report lightbox."""
    return """
body.modal-open { overflow: hidden; }
.lightbox[hidden] { display: none; }
.lightbox { box-sizing: border-box; position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; padding: 18px; background: rgba(15, 23, 42, 0.9); }
.lightbox-figure { box-sizing: border-box; width: min(96vw, 1800px); height: min(94vh, 1100px); min-height: 320px; margin: 0; padding: 0; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; overflow: hidden; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34); }
.lightbox-toolbar { min-height: 44px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 10px; border-bottom: 1px solid #d1d5db; background: #f8fafc; }
.lightbox-zoom-controls { display: flex; align-items: center; gap: 6px; }
.lightbox-tool { box-sizing: border-box; min-width: 34px; height: 32px; display: inline-grid; place-items: center; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0 9px; background: #ffffff; color: #111827; font: 600 14px/1 Inter, Arial, sans-serif; text-decoration: none; cursor: pointer; }
.lightbox-tool:hover { border-color: #64748b; background: #f1f5f9; }
.lightbox-tool:focus-visible { outline: 2px solid #2563eb; outline-offset: 2px; }
.lightbox-tool:disabled { color: #94a3b8; cursor: default; }
.lightbox-fit, .lightbox-open { min-width: 48px; }
.lightbox-close { min-width: 32px; flex: 0 0 32px; padding: 0; font-size: 16px; }
.lightbox-zoom { width: 54px; color: #334155; font: 600 12px/1 Inter, Arial, sans-serif; text-align: center; }
.lightbox-viewport { min-width: 0; min-height: 0; overflow: auto; padding: 16px; overscroll-behavior: contain; background: #e2e8f0; outline: none; }
.lightbox-viewport:focus-visible { box-shadow: inset 0 0 0 2px #2563eb; }
.lightbox-viewport.is-zoomed { cursor: grab; }
.lightbox-viewport.is-dragging { cursor: grabbing; user-select: none; }
.lightbox-viewport img { width: auto; max-width: none; max-height: none; height: auto; margin: 0 auto; object-fit: initial; }
.lightbox-viewport img[src$=".png"] { image-rendering: pixelated; }
.lightbox-figure figcaption { min-height: 20px; margin: 0; padding: 8px 12px; border-top: 1px solid #d1d5db; color: #334155; font-size: 13px; overflow-wrap: anywhere; }
@media (max-width: 640px) {
  .lightbox { padding: 8px; }
  .lightbox-figure { width: 100%; height: 100%; }
  .lightbox-toolbar { gap: 6px; }
  .lightbox-tool { padding: 0 7px; }
}
@media (max-width: 420px) {
  .lightbox-open { display: none; }
  .lightbox-zoom { width: 44px; }
}
""".strip()


def _report_css() -> str:
    base_css = """
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
.image-card { margin: 0; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; cursor: zoom-in; transition: border-color 120ms ease, box-shadow 120ms ease; }
.image-card:focus { outline: 2px solid #2563eb; outline-offset: 2px; }
.image-card:hover { border-color: #94a3b8; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08); }
.image-card img { display: block; width: 100%; height: auto; image-rendering: auto; }
.image-card img[src$=".png"] { image-rendering: pixelated; }
.image-card figcaption { margin-top: 8px; color: #374151; font-size: 13px; }
.snapshots { display: grid; gap: 18px; }
.snapshot { border-top: 1px solid #d1d5db; padding-top: 16px; }
""".strip()
    return f"{base_css}\n{report_lightbox_css()}"
