"""Train and report a FabricPC generative MNIST label-canvas spike."""

from __future__ import annotations

import argparse
import html
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax

from apm.data import load_mnist
from apm.data.mnist import balanced_task_subset, identity_permutation, make_permuted_task
from apm.models.fabricpc_backend import FabricPcBackend, FabricPcConfig, FabricPcTrainConfig
from apm.training.artifacts import append_jsonl, write_json, write_png_grid, write_svg_line_chart

DEFAULT_RUN_DIR = Path("results") / "fabricpc_mnist_spike"


def main() -> None:
    """Run the FabricPC MNIST spike and write report artifacts."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    task = balanced_task_subset(
        make_permuted_task(load_mnist(allow_download=True), identity_permutation(), "P0"),
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    train_config = FabricPcTrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
        eval_batch_size=args.eval_batch_size,
        eval_train_count=args.eval_train_count,
        eval_test_count=args.eval_test_count,
        show_progress=not args.no_progress,
    )
    model_config = FabricPcConfig(latent_dim=args.latent_dim, hidden_widths=tuple(args.hidden_widths))
    backend = FabricPcBackend(model_config=model_config, train_config=train_config)
    state = backend.init_state(jax.random.PRNGKey(args.seed))
    state, metrics_rows = backend.continue_train(
        state,
        task.train_canvases(),
        task.test_canvases(),
        task.train_labels,
        task.test_labels,
        collect_epoch_metrics=True,
    )
    _write_outputs(args.output_dir, backend, state.params, task, metrics_rows, args.sample_count)
    print(args.output_dir)


def _write_outputs(
    run_dir: Path,
    backend: FabricPcBackend,
    params: object,
    task: object,
    metrics_rows: list[dict[str, int | float]],
    sample_count: int,
) -> None:
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    append_jsonl(metrics_path, metrics_rows)
    write_json(
        run_dir / "config.json",
        {
            "model": {"kind": backend.kind},
            "fabricpc": asdict(backend.model_config),
            "train": asdict(backend.train_config),
            "stream": {
                "kind": "mnist_label_canvas_identity",
                "train_examples": int(task.train_labels.shape[0]),
                "test_examples": int(task.test_labels.shape[0]),
            },
        },
    )
    _write_charts(run_dir, metrics_rows)
    canvases = task.test_canvases()[:sample_count]
    write_png_grid(run_dir / "test_originals.png", canvases)
    write_png_grid(run_dir / "test_full_reconstructions.png", backend.reconstruct(params, canvases, jax.random.PRNGKey(30_000)))
    write_png_grid(
        run_dir / "test_masked_label_reconstructions.png",
        backend.reconstruct(params, canvases, jax.random.PRNGKey(30_001), mask_label=True),
    )
    _write_report(run_dir, metrics_rows)


def _write_charts(run_dir: Path, metrics_rows: list[dict[str, int | float]]) -> None:
    write_svg_line_chart(
        run_dir / "energy_curves.svg",
        metrics_rows,
        (
            ("train_loss", "train step energy"),
            ("train_eval_loss", "train eval energy"),
            ("test_loss", "test eval energy"),
        ),
        "FabricPC Energy",
        "energy",
    )
    write_svg_line_chart(
        run_dir / "mse_curves.svg",
        metrics_rows,
        (
            ("train_eval_digit_mse", "train digit MSE"),
            ("test_digit_mse", "test digit MSE"),
            ("train_eval_label_patch_mse", "train label MSE"),
            ("test_label_patch_mse", "test label MSE"),
        ),
        "FabricPC Reconstruction MSE",
        "MSE",
    )
    write_svg_line_chart(
        run_dir / "accuracy_curves.svg",
        metrics_rows,
        (
            ("train_eval_energy_classifier_accuracy", "train energy accuracy"),
            ("test_energy_classifier_accuracy", "test energy accuracy"),
            ("train_eval_label_patch_accuracy", "train label accuracy"),
            ("test_label_patch_accuracy", "test label accuracy"),
        ),
        "FabricPC Label Accuracy",
        "accuracy",
    )


def _write_report(run_dir: Path, metrics_rows: list[dict[str, int | float]]) -> None:
    final = metrics_rows[-1] if metrics_rows else {}
    run_dir.joinpath("report.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                "<title>FabricPC MNIST Spike</title>",
                "<style>",
                _report_css(),
                "</style>",
                "</head>",
                "<body><main>",
                "<h1>FabricPC MNIST Spike</h1>",
                "<section><h2>Final Metrics</h2>",
                _metrics_table(final),
                "</section>",
                "<section><h2>Curves</h2>",
                _figure_grid(("energy_curves.svg", "mse_curves.svg", "accuracy_curves.svg")),
                "</section>",
                "<section><h2>Reconstructions</h2>",
                _figure_grid(
                    (
                        "test_originals.png",
                        "test_full_reconstructions.png",
                        "test_masked_label_reconstructions.png",
                    )
                ),
                "</section>",
                "</main>",
                _report_lightbox(),
                "<script>",
                _report_script(),
                "</script>",
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _metrics_table(metrics: dict[str, int | float]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{float(value):.6g}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    return f"<table>{rows}</table>"


def _figure_grid(filenames: tuple[str, ...]) -> str:
    return '<div class="grid">' + "\n".join(
        (
            f'<figure class="figure-card" role="button" tabindex="0" '
            f'data-lightbox-src="{html.escape(filename)}" data-lightbox-caption="{html.escape(filename)}">'
            f'<img src="{html.escape(filename)}" alt="{html.escape(filename)}">'
            f"<figcaption>{html.escape(filename)}</figcaption>"
            "</figure>"
        )
        for filename in filenames
    ) + "</div>"


def _report_lightbox() -> str:
    return "\n".join(
        [
            '<div id="report-lightbox" class="lightbox" hidden>',
            '<button type="button" class="lightbox-close" aria-label="Close">Close</button>',
            '<figure class="lightbox-figure">',
            '<img id="report-lightbox-image" alt="">',
            '<figcaption id="report-lightbox-caption"></figcaption>',
            "</figure>",
            "</div>",
        ]
    )


def _report_script() -> str:
    return r"""
const lightbox = document.getElementById("report-lightbox");
const lightboxImage = document.getElementById("report-lightbox-image");
const lightboxCaption = document.getElementById("report-lightbox-caption");
const closeButton = lightbox.querySelector(".lightbox-close");

function openLightbox(card) {
  const src = card.dataset.lightboxSrc;
  const caption = card.dataset.lightboxCaption || src;
  lightboxImage.src = src;
  lightboxImage.alt = caption;
  lightboxCaption.textContent = caption;
  lightbox.hidden = false;
  document.body.classList.add("modal-open");
  closeButton.focus();
}

function closeLightbox() {
  lightbox.hidden = true;
  document.body.classList.remove("modal-open");
  lightboxImage.removeAttribute("src");
}

document.querySelectorAll(".figure-card").forEach((card) => {
  card.addEventListener("click", () => openLightbox(card));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLightbox(card);
    }
  });
});

closeButton.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) {
    closeLightbox();
  }
});
""".strip()


def _report_css() -> str:
    return """
body { margin: 0; background: #f8fafc; color: #111827; font: 15px/1.45 Inter, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }
h1 { margin: 0 0 24px; font-size: 28px; }
h2 { margin: 0 0 14px; font-size: 20px; }
section { margin: 0 0 28px; }
table { border-collapse: collapse; width: 100%; background: #ffffff; }
th, td { border: 1px solid #d1d5db; padding: 7px 9px; text-align: left; vertical-align: top; }
th { width: 280px; background: #f3f4f6; font-weight: 650; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
figure { margin: 0; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; }
.figure-card { cursor: zoom-in; transition: border-color 120ms ease, box-shadow 120ms ease; }
.figure-card:focus { outline: 2px solid #2563eb; outline-offset: 2px; }
.figure-card:hover { border-color: #94a3b8; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08); }
img { display: block; width: 100%; height: auto; image-rendering: auto; }
img[src$=".png"] { image-rendering: pixelated; }
figcaption { margin-top: 8px; color: #374151; font-size: 13px; }
body.modal-open { overflow: hidden; }
.lightbox[hidden] { display: none; }
.lightbox { position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; padding: 28px; background: rgba(15, 23, 42, 0.88); }
.lightbox-close { position: fixed; top: 18px; right: 20px; border: 1px solid #cbd5e1; background: #ffffff; color: #111827; border-radius: 6px; padding: 8px 12px; font: inherit; cursor: pointer; }
.lightbox-figure { width: min(96vw, 1600px); max-height: 92vh; margin: 0; padding: 14px; display: flex; flex-direction: column; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34); }
.lightbox-figure img { width: 100%; height: auto; max-height: calc(92vh - 72px); object-fit: contain; }
.lightbox-figure img[src$=".png"] { image-rendering: pixelated; }
.lightbox-figure figcaption { margin-top: 10px; color: #111827; font-size: 14px; overflow-wrap: anywhere; }
""".strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--train-count", type=int, default=2_000)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-widths", type=int, nargs="+", default=(256, 128))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--infer-steps", type=int, default=40)
    parser.add_argument("--eta-infer", type=float, default=0.05)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-train-count", type=int, default=2_000)
    parser.add_argument("--eval-test-count", type=int, default=2_000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
