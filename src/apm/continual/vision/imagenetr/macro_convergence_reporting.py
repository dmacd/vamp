"""Publication-style reporting for the stage-31 macro-token convergence audit."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import json
import math

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf


def _validated_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-macro-convergence-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or result.get("test_evaluations") != 0
    ):
        raise ValueError("macro convergence result does not authenticate")
    return result


def _history_rows(
    run: Path, entries: Sequence[Mapping[str, object]], family: str
) -> tuple[dict[str, object], ...]:
    rows = []
    for entry in entries:
        cell = dict(entry.get("cell", {}))
        history = (run / str(entry["history"])).read_text(encoding="utf-8")
        for line in history.splitlines():
            raw = json.loads(line)
            rows.append(
                {
                    "effective_batch_size": cell.get("effective_batch_size"),
                    "epoch": int(raw["epoch"]),
                    "family": family,
                    "gradient_norm_mean": raw.get("gradient_norm_mean"),
                    "image_presentations": int(raw["image_presentations"]),
                    "learning_rate": raw.get(
                        "learning_rate", raw.get("lora_learning_rate")
                    ),
                    "optimizer_steps": int(raw["optimizer_steps"]),
                    "peak_learning_rate": cell.get("peak_learning_rate"),
                    "role": entry.get("role", family),
                    "schedule": cell.get("schedule", "fixed_joint_recipe"),
                    "seed": cell.get("seed", 1993),
                    "train_objective_accuracy": float(
                        raw["train_objective_accuracy"]
                    ),
                    "train_objective_nll": float(raw["train_objective_nll"]),
                    "validation_accuracy": float(raw["validation_accuracy"]),
                    "validation_nll": float(raw["validation_nll"]),
                    "wall_seconds": float(raw["wall_seconds"]),
                }
            )
    return tuple(rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_table_family(
    report_root: Path, name: str, rows: Sequence[Mapping[str, object]]
) -> None:
    projected = tuple(dict(row) for row in rows)
    _write_csv(report_root / f"{name}.csv", projected)
    atomic_write(
        report_root / f"{name}.json", canonical_json_bytes({"rows": projected})
    )
    try:
        import pandas as pd

        pd.DataFrame(projected).to_parquet(
            report_root / f"{name}.parquet", index=False
        )
    except ImportError:  # pragma: no cover - vision environment gate
        pass


def _candidate_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "best_epoch": int(dict(entry["fit"])["best_epoch"]),
            "best_optimizer_steps": int(
                dict(entry["fit"])["best_optimizer_steps"]
            ),
            "effective_batch_size": int(
                dict(entry["cell"])["effective_batch_size"]
            ),
            "fit_accuracy": float(dict(entry["fit"])["train_accuracy"]),
            "fit_minus_validation_accuracy": float(
                dict(entry["fit"])["train_accuracy"]
            )
            - float(dict(entry["fit"])["validation_accuracy"]),
            "learning_rate": float(dict(entry["cell"])["peak_learning_rate"]),
            "optimizer_steps": int(dict(entry["fit"])["optimizer_steps"]),
            "validation_accuracy": float(
                dict(entry["fit"])["validation_accuracy"]
            ),
            "validation_nll": float(dict(entry["fit"])["validation_nll"]),
            "wall_seconds": float(dict(entry["fit"])["wall_seconds"]),
        }
        for entry in result["screening_candidates"]
    )


def _replication_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "best_epoch": int(dict(entry["fit"])["best_epoch"]),
            "fit_accuracy": float(dict(entry["fit"])["train_accuracy"]),
            "fit_minus_validation_accuracy": float(
                dict(entry["fit"])["train_accuracy"]
            )
            - float(dict(entry["fit"])["validation_accuracy"]),
            "seed": int(dict(entry["cell"])["seed"]),
            "validation_accuracy": float(
                dict(entry["fit"])["validation_accuracy"]
            ),
            "validation_nll": float(dict(entry["fit"])["validation_nll"]),
        }
        for entry in result["replications"]
    )


def _plot_learning_curves(
    path: Path, histories: Sequence[Mapping[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    screening = tuple(
        row
        for row in histories
        if row["family"] == "macro" and row["role"] == "screening"
    )
    legacy = tuple(row for row in histories if row["role"] == "legacy_control")
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex="col")
    colors = {0.00003: "#4c78a8", 0.0001: "#f58518", 0.0003: "#54a24b"}
    for column, batch_size in enumerate((64, 128, 512)):
        selected = tuple(
            row for row in screening if row["effective_batch_size"] == batch_size
        )
        for learning_rate in (0.00003, 0.0001, 0.0003):
            curve = tuple(
                row for row in selected if row["peak_learning_rate"] == learning_rate
            )
            label = f"peak LR {learning_rate:g}"
            for axis, metric in zip(
                axes[:, column],
                ("validation_accuracy", "validation_nll"),
                strict=True,
            ):
                axis.plot(
                    [row["epoch"] for row in curve],
                    [row[metric] for row in curve],
                    color=colors[learning_rate],
                    label=label,
                    linewidth=1.8,
                )
        if batch_size == 512:
            for axis, metric in zip(
                axes[:, column],
                ("validation_accuracy", "validation_nll"),
                strict=True,
            ):
                axis.plot(
                    [row["epoch"] for row in legacy],
                    [row[metric] for row in legacy],
                    color="#777777",
                    linestyle="--",
                    label="legacy constant LR",
                )
        axes[0, column].set_title(f"Effective batch {batch_size}")
        axes[1, column].set_xlabel("Epoch")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Validation accuracy (%)")
    axes[1, 0].set_ylabel("Validation NLL")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Stage-31 macro-token clean learning curves", fontsize=14)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_update_curves(
    path: Path, histories: Sequence[Mapping[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    screening = tuple(
        row
        for row in histories
        if row["family"] == "macro" and row["role"] == "screening"
    )
    figure, axis = plt.subplots(figsize=(9.8, 5.2))
    colors = {64: "#4c78a8", 128: "#f58518", 512: "#54a24b"}
    styles = {0.00003: ":", 0.0001: "--", 0.0003: "-"}
    for batch_size in (64, 128, 512):
        for learning_rate in (0.00003, 0.0001, 0.0003):
            curve = tuple(
                row
                for row in screening
                if row["effective_batch_size"] == batch_size
                and row["peak_learning_rate"] == learning_rate
            )
            axis.plot(
                [row["optimizer_steps"] for row in curve],
                [row["validation_accuracy"] for row in curve],
                color=colors[batch_size],
                linestyle=styles[learning_rate],
                linewidth=1.7,
                label=f"batch {batch_size}, LR {learning_rate:g}",
            )
    axis.set(
        xlabel="Optimizer updates",
        ylabel="Validation accuracy (%)",
        title="Validation accuracy versus optimizer updates",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(
    path: Path,
    candidates: Sequence[Mapping[str, object]],
    replications: Sequence[Mapping[str, object]],
    result: Mapping[str, object],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(
        [
            [
                next(
                    float(row["validation_accuracy"])
                    for row in candidates
                    if row["effective_batch_size"] == batch
                    and row["learning_rate"] == rate
                )
                for rate in (0.00003, 0.0001, 0.0003)
            ]
            for batch in (64, 128, 512)
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    image = axes[0].imshow(matrix, cmap="viridis", vmin=matrix.min() - 0.2)
    for row_index in range(3):
        for column_index in range(3):
            axes[0].text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row_index, column_index] < matrix.mean()
                    else "black"
                ),
                fontsize=10,
            )
    axes[0].set_xticks(range(3), ("3e-5", "1e-4", "3e-4"))
    axes[0].set_yticks(range(3), ("64", "128", "512"))
    axes[0].set(
        xlabel="Peak learning rate",
        ylabel="Effective batch size",
        title="Best seed-1993 validation accuracy (%)",
    )
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    joint_fit = dict(dict(result["clean_joint"])["fit"])
    legacy_fit = dict(dict(result["legacy_control"])["fit"])
    labels = ("Legacy\nmacro", "Selected\nmacro mean", "Joint IID\nepoch 5")
    macro_values = tuple(float(row["validation_accuracy"]) for row in replications)
    values = (
        float(legacy_fit["validation_accuracy"]),
        math.fsum(macro_values) / len(macro_values),
        float(joint_fit["fixed_validation_accuracy"]),
    )
    axes[1].bar(labels, values, color=("#999999", "#4c78a8", "#e45756"))
    axes[1].errorbar(
        1,
        values[1],
        yerr=[[values[1] - min(macro_values)], [max(macro_values) - values[1]]],
        color="black",
        capsize=5,
        fmt="none",
    )
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.25, f"{value:.2f}", ha="center")
    axes[1].set(
        ylabel="Clean validation accuracy (%)",
        title="Same-split validation comparison",
    )
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _candidate_table(rows: Sequence[Mapping[str, object]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{int(row['effective_batch_size'])}</td>"
        f"<td>{float(row['learning_rate']):g}</td>"
        f"<td>{int(row['best_epoch'])}</td>"
        f"<td>{int(row['best_optimizer_steps']):,}</td>"
        f"<td>{float(row['fit_accuracy']):.3f}</td>"
        f"<td>{float(row['validation_accuracy']):.3f}</td>"
        f"<td>{float(row['validation_nll']):.4f}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Batch</th><th>Peak LR</th><th>Best epoch</th>"
        "<th>Best updates</th><th>Fit acc.</th><th>Validation acc.</th>"
        f"<th>Validation NLL</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _report_text(
    result: Mapping[str, object],
    replications: Sequence[Mapping[str, object]],
) -> str:
    winner = dict(dict(result["selection"])["winner"])
    joint_fit = dict(dict(result["clean_joint"])["fit"])
    legacy_fit = dict(dict(result["legacy_control"])["fit"])
    mean_accuracy = math.fsum(
        float(row["validation_accuracy"]) for row in replications
    ) / len(replications)
    mean_nll = math.fsum(float(row["validation_nll"]) for row in replications) / len(
        replications
    )
    gap = mean_accuracy - float(joint_fit["fixed_validation_accuracy"])
    seed_values = ", ".join(
        f"{int(row['seed'])}: {float(row['validation_accuracy']):.3f}%"
        for row in replications
    )
    best_epochs = tuple(int(row["best_epoch"]) for row in replications)
    convergence_statement = (
        "At least one selected run reached its best validation NLL at epoch 50, so the audit does not establish convergence."
        if max(best_epochs) == 50
        else "Every selected run peaked before epoch 50; more epochs under the same schedule are not supported by validation NLL."
    )
    return f"""# ImageNet-R Stage-31 Macro-Token Convergence Audit

## Main result

The clean-selected warmup-cosine schedule used effective batch {int(winner['effective_batch_size'])} and peak learning rate {float(winner['peak_learning_rate']):g}. It reached {mean_accuracy:.3f}% mean validation accuracy and {mean_nll:.4f} mean validation NLL over three seeds ({seed_values}). The same-split joint-IID control reached {float(joint_fit['fixed_validation_accuracy']):.3f}% after its fixed fifth epoch. The macro mean therefore differed from joint IID by {gap:+.3f} percentage points.

The exact legacy rerun reached {float(legacy_fit['validation_accuracy']):.3f}% at epoch {int(legacy_fit['best_epoch'])}. The selected screening cell changed seed-1993 validation accuracy by {float(winner['validation_accuracy']) - float(legacy_fit['validation_accuracy']):+.3f} points. {convergence_statement}

## What was tested

All macro models used the same one-block 12,055,496-parameter architecture and the same frozen stage-31 hierarchy representations. Seed 1993 crossed effective batches 64, 128, and 512 with peak AdamW learning rates 3e-5, 1e-4, and 3e-4. Each cell used a five-percent linear warmup followed by cosine decay through epoch 50. The legacy control used constant 3e-4 for 20 epochs. The selected schedule was repeated with seeds 1994 and 1995.

The joint-IID control used the identical 12,194 fit and 3,049 validation identities, but it trained a fresh rank-16 QKV-plus-fc1 LoRA and affine 124-class head for five epochs. It therefore measures what joint feature adaptation can achieve on this clean split; it is not a gate.

## Interpretation

The training and validation curves distinguish insufficient fitting from poor generalization. A macro model that continues reducing fit NLL while validation NLL rises has passed its useful fitting point. A schedule that keeps improving validation NLL near epoch 50 remains a convergence candidate. The comparison cannot isolate architecture from representation quality because joint IID updates its LoRA features while the macro classifier receives fixed node-specific features.

The experiment never requested a test image. These are development measurements only and do not revise the locked-test result from v8.

## Reproducibility

The source v8 run, fit-only hierarchy, split, environment, code, and resolved configuration are content-addressed. Epoch rows were hash-chained and fsynced before the next epoch. An immediate cache replay used {int(dict(result['reuse_proof'])['cache_hits']):,} cached node-example rows, performed zero adapted-token forwards, and preserved both population identities.
"""


def _html_report(
    markdown: str,
    candidates: Sequence[Mapping[str, object]],
    curves: Path,
    updates: Path,
    summary: Path,
) -> str:
    sections = tuple(section.split("\n", 1) for section in markdown.split("\n## "))
    title = sections[0][0].removeprefix("# ")
    rendered_sections = []
    for heading, body in sections[1:]:
        paragraphs = "".join(
            f"<p>{escape(paragraph.strip())}</p>"
            for paragraph in body.strip().split("\n\n")
        )
        rendered_sections.append(f"<h2>{escape(heading)}</h2>{paragraphs}")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 15mm 16mm 16mm; }}
body {{ font-family: Arial, sans-serif; color: #17212b; font-size: 10.5pt; line-height: 1.38; }}
h1 {{ color: #173b57; font-size: 23pt; margin: 0 0 8mm; }}
h2 {{ color: #24577a; font-size: 15pt; margin: 6mm 0 2mm; border-bottom: 1px solid #b9cad6; }}
p {{ margin: 0 0 3mm; }}
.subtitle {{ color: #526777; margin-bottom: 7mm; }}
.figure {{ break-inside: avoid; margin: 5mm 0; text-align: center; }}
.figure img {{ width: 100%; max-height: 170mm; object-fit: contain; }}
.caption {{ font-size: 8.5pt; color: #526777; text-align: left; margin-top: 1mm; }}
.page {{ break-before: page; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.2pt; margin-top: 3mm; }}
th {{ background: #e8f0f5; color: #173b57; }}
th, td {{ border: 1px solid #bac8d1; padding: 1.5mm; text-align: right; }}
th:first-child, td:first-child {{ text-align: center; }}
</style></head><body>
<h1>{escape(title)}</h1><div class="subtitle">Clean-only optimization audit - stage 31 - locked test unopened</div>
{''.join(rendered_sections[:2])}
<div class="figure"><img src="{_image_uri(summary)}"><div class="caption">Figure 1. Best clean validation accuracy across the optimizer matrix and the same-split comparison. The macro error bar is the observed three-seed range, not a confidence interval.</div></div>
<div class="page"><h2>Full optimizer matrix</h2>{_candidate_table(candidates)}
<div class="figure"><img src="{_image_uri(curves)}"><div class="caption">Figure 2. Validation accuracy and negative log likelihood (NLL) through every scheduled epoch. The dashed legacy curve appears only in the effective-batch-512 panel.</div></div></div>
<div class="page"><h2>Update-count view</h2><div class="figure"><img src="{_image_uri(updates)}"><div class="caption">Figure 3. The same validation measurements indexed by optimizer updates rather than epochs.</div></div>
{''.join(rendered_sections[2:])}</div>
</body></html>"""


def write_macro_convergence_report(run: str | Path) -> Path:
    """Write plots, compact ledgers, Markdown, HTML, and the final PDF report."""
    run_path = Path(run).resolve()
    report_root = run_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    result = _validated_result(run_path)
    candidates = _candidate_rows(result)
    replications = _replication_rows(result)
    all_macro_entries = (
        dict(result["legacy_control"]),
        *(dict(row) for row in result["screening_candidates"]),
        *(
            dict(row)
            for row in result["replications"]
            if int(dict(row["cell"])["seed"]) != 1993
        ),
    )
    histories = (
        *_history_rows(run_path, all_macro_entries, "macro"),
        *_history_rows(run_path, (dict(result["clean_joint"]),), "joint_iid"),
    )
    _write_table_family(report_root, "epoch_history", histories)
    _write_table_family(report_root, "screening_candidates", candidates)
    _write_table_family(report_root, "selected_replications", replications)
    curves = report_root / "validation_learning_curves.png"
    updates = report_root / "validation_by_optimizer_updates.png"
    summary = report_root / "same_split_summary.png"
    _plot_learning_curves(curves, histories)
    _plot_update_curves(updates, histories)
    _plot_summary(summary, candidates, replications, result)
    markdown = _report_text(result, replications)
    atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    atomic_write(
        report_root / "REPORT.html",
        _html_report(markdown, candidates, curves, updates, summary).encode("utf-8"),
    )
    return render_integrator_pdf(report_root / "REPORT.html")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ...macro_convergence_reporting RUN_DIRECTORY")
    print(write_macro_convergence_report(sys.argv[1]))


__all__ = ["write_macro_convergence_report"]
