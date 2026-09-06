"""Publication-style reporting for the stage-31 frontier-LoRA adaptation audit."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
import csv
from html import escape
import json
from pathlib import Path

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf


CONDITION_LABELS = {
    1024: "Frontier LoRA adaptation (H=1,024)",
    2048: "Frontier LoRA adaptation (H=2,048)",
    4096: "Frontier LoRA adaptation (H=4,096)",
    8192: "Frontier LoRA adaptation (H=8,192)",
    11827: "Frontier LoRA adaptation (H=11,827; full fit)",
}
FROZEN_ONLINE_LABEL = "Frozen frontier, online full fit"
FROZEN_CACHED_LABEL = "Frozen macro, cached full fit (seed 1993)"
JOINT_LABEL = "Joint IID, one shared LoRA"


def _validated_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-frontier-adaptation-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or result.get("test_evaluations") != 0
    ):
        raise ValueError("frontier adaptation result does not authenticate")
    return result


def _history(run: Path, cell: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line)
        for line in (run / str(cell["history"])).read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(rows) != 50:
        raise ValueError("frontier adaptation history is incomplete")
    return rows


def _summary_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    references = dict(result["references"])
    joint = dict(references["joint_iid"])
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        fit = dict(cell["fit"])
        history = _history(run, cell)
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        simultaneous = tuple(
            row
            for row in history
            if float(row["validation_accuracy"]) >= float(joint["accuracy"])
            and float(row["validation_nll"]) <= float(joint["nll"])
        )
        rows.append(
            {
                "accuracy_gap_to_joint_at_min_nll_pp": float(
                    fit["validation_accuracy_at_best_nll"]
                )
                - float(joint["accuracy"]),
                "adapt_lora": adapted,
                "best_nll_epoch": int(fit["best_nll_epoch"]),
                "condition": (
                    CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
                ),
                "historical_capacity": capacity,
                "image_presentations": int(fit["image_presentations"]),
                "max_accuracy_epoch": int(fit["max_accuracy_epoch"]),
                "max_validation_accuracy": float(fit["max_validation_accuracy"]),
                "nll_gap_to_joint": float(fit["best_validation_nll"])
                - float(joint["nll"]),
                "peak_vram_bytes": int(fit["peak_vram_bytes"]),
                "simultaneous_joint_match_epoch": (
                    None if not simultaneous else int(simultaneous[0]["epoch"])
                ),
                "train_accuracy_at_best": float(fit["train_accuracy_at_best"]),
                "train_nll_at_best": float(fit["train_nll_at_best"]),
                "trainable_parameters": int(fit["trainable_parameters"]),
                "validation_accuracy_at_best_nll": float(
                    fit["validation_accuracy_at_best_nll"]
                ),
                "validation_nll_at_max_accuracy": float(
                    fit["validation_nll_at_max_accuracy"]
                ),
                "validation_nll_minimum": float(fit["best_validation_nll"]),
                "wall_seconds": float(fit["wall_seconds"]),
            }
        )
    return tuple(rows)


def _history_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        label = CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
        rows.extend(
            {
                "adapt_lora": adapted,
                "condition": label,
                "epoch": int(row["epoch"]),
                "gradient_norm_mean": float(row["gradient_norm_mean"]),
                "historical_capacity": capacity,
                "image_presentations": int(row["image_presentations"]),
                "lora_learning_rate": row["lora_learning_rate"],
                "macro_learning_rate": float(row["macro_learning_rate"]),
                "optimizer_steps": int(row["optimizer_steps"]),
                "train_objective_accuracy": float(row["train_objective_accuracy"]),
                "train_objective_nll": float(row["train_objective_nll"]),
                "validation_accuracy": float(row["validation_accuracy"]),
                "validation_nll": float(row["validation_nll"]),
                "wall_seconds": float(row["wall_seconds"]),
            }
            for row in _history(run, cell)
        )
    return tuple(rows)


def _displacement_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for raw_cell in result["cells"]:
        cell = dict(raw_cell)
        specification = dict(cell["cell"])
        capacity = int(specification["historical_capacity"])
        adapted = bool(specification["adapt_lora"])
        label = CONDITION_LABELS[capacity] if adapted else FROZEN_ONLINE_LABEL
        rows.extend(
            {
                **dict(displacement),
                "adapt_lora": adapted,
                "condition": label,
                "historical_capacity": capacity,
            }
            for displacement in cell["displacements"]
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


def _plot_primary(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    references: Mapping[str, object],
) -> None:
    import matplotlib.pyplot as plt

    adapted = tuple(row for row in summaries if row["adapt_lora"])
    frozen = next(row for row in summaries if not row["adapt_lora"])
    joint = dict(references["joint_iid"])
    cached = dict(references["frozen_macro_seed1993"])
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    x_values = [int(row["historical_capacity"]) for row in adapted]
    specifications = (
        ("validation_accuracy_at_best_nll", "Validation accuracy (%)", "accuracy"),
        ("validation_nll_minimum", "Validation NLL", "nll"),
    )
    for axis, (metric, ylabel, reference_key) in zip(axes, specifications, strict=True):
        axis.plot(
            x_values,
            [float(row[metric]) for row in adapted],
            color="#2166ac",
            marker="o",
            linewidth=2.2,
            label="Frontier LoRA adaptation",
        )
        axis.scatter(
            [int(frozen["historical_capacity"])],
            [float(frozen[metric])],
            color="#7b3294",
            marker="D",
            s=62,
            label=FROZEN_ONLINE_LABEL,
            zorder=4,
        )
        axis.axhline(
            float(joint[reference_key]),
            color="#b2182b",
            linestyle="--",
            linewidth=1.8,
            label=JOINT_LABEL,
        )
        axis.axhline(
            float(cached[reference_key]),
            color="#666666",
            linestyle=":",
            linewidth=1.8,
            label=FROZEN_CACHED_LABEL,
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(x_values, ["1k", "2k", "4k", "8k", "full\n11,827"])
        axis.set_xlabel("Historical replay identities (H)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
    axes[0].set_title("Minimum-NLL checkpoint accuracy")
    axes[1].set_title("Minimum validation NLL")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Stage-31 representation adaptation versus replay population")
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(
    path: Path,
    histories: Sequence[Mapping[str, object]],
    references: Mapping[str, object],
) -> None:
    import matplotlib.pyplot as plt

    joint = dict(references["joint_iid"])
    colors = {
        1024: "#4393c3",
        2048: "#2166ac",
        4096: "#1b7837",
        8192: "#e08214",
        11827: "#b2182b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for capacity in colors:
        curve = tuple(
            row
            for row in histories
            if row["adapt_lora"] and row["historical_capacity"] == capacity
        )
        for axis, metric in zip(
            axes, ("validation_accuracy", "validation_nll"), strict=True
        ):
            axis.plot(
                [row["epoch"] for row in curve],
                [row[metric] for row in curve],
                color=colors[capacity],
                linewidth=1.7,
                label=CONDITION_LABELS[capacity],
            )
    frozen = tuple(row for row in histories if not row["adapt_lora"])
    for axis, metric in zip(
        axes, ("validation_accuracy", "validation_nll"), strict=True
    ):
        axis.plot(
            [row["epoch"] for row in frozen],
            [row[metric] for row in frozen],
            color="#7b3294",
            linestyle=":",
            linewidth=1.8,
            label=FROZEN_ONLINE_LABEL,
        )
    axes[0].axhline(
        float(joint["accuracy"]),
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label=JOINT_LABEL,
    )
    axes[1].axhline(
        float(joint["nll"]),
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label=JOINT_LABEL,
    )
    axes[0].set(title="Accuracy over training", ylabel="Validation accuracy (%)")
    axes[1].set(title="NLL over training", ylabel="Validation NLL")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    figure.suptitle("All checkpoints; dashed black is the five-epoch joint-IID endpoint")
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_displacements(
    path: Path, displacements: Sequence[Mapping[str, object]]
) -> None:
    import matplotlib.pyplot as plt

    adapted = tuple(row for row in displacements if row["adapt_lora"])
    figure, axis = plt.subplots(figsize=(9.8, 4.8))
    for level in range(5):
        rows = tuple(row for row in adapted if int(row["level"]) == level)
        tasks = rows[0]["represented_task_ids"]
        label = (
            f"level {level}, task {int(tasks[0]) + 1}"
            if len(tasks) == 1
            else f"level {level}, tasks {int(tasks[0]) + 1}-{int(tasks[-1]) + 1}"
        )
        axis.plot(
            [int(row["historical_capacity"]) for row in rows],
            [float(row["dense_update_relative_change"]) for row in rows],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(
        [1024, 2048, 4096, 8192, 11827],
        ["1k", "2k", "4k", "8k", "full\n11,827"],
    )
    axis.set(
        xlabel="Historical replay identities (H)",
        ylabel="Relative dense LoRA-update change",
        title="How far each sealed frontier adapter moved",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_frontier(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    intervals = ((31, 31), (29, 30), (25, 28), (17, 24), (1, 16))
    figure, axis = plt.subplots(figsize=(10.2, 4.6))
    for level, (first, last) in enumerate(intervals):
        y = 4.3 - 0.8 * level
        label = f"level {level}: task {first}" if first == last else f"level {level}: tasks {first}-{last}"
        box = FancyBboxPatch(
            (0.4, y - 0.25),
            3.6,
            0.5,
            boxstyle="round,pad=0.04",
            facecolor="#d9eaf4",
            edgecolor="#2166ac",
            linewidth=1.3,
        )
        axis.add_patch(box)
        axis.text(2.2, y, label, ha="center", va="center", fontsize=10)
        axis.add_patch(
            FancyArrowPatch(
                (4.05, y),
                (6.0, 2.7),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#666666",
                linewidth=1.1,
            )
        )
    macro = FancyBboxPatch(
        (6.05, 2.2),
        3.3,
        1.0,
        boxstyle="round,pad=0.06",
        facecolor="#fddbc7",
        edgecolor="#b2182b",
        linewidth=1.5,
    )
    axis.add_patch(macro)
    axis.text(7.7, 2.7, "shared macro-token\nintegrator", ha="center", va="center", fontsize=11)
    axis.text(2.2, 0.45, "Each arrow carries 197 adapted 768-value tokens plus local affine scores.", ha="center", fontsize=9)
    axis.text(7.7, 1.6, "124-way task-free prediction", ha="center", fontsize=9)
    axis.set(xlim=(0, 9.8), ylim=(0.1, 4.9), title="Task-31 fragmented frontier under joint adaptation")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _table_html(rows: Sequence[Mapping[str, object]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{int(row['best_nll_epoch'])}</td>"
        f"<td>{float(row['validation_accuracy_at_best_nll']):.3f}%</td>"
        f"<td>{float(row['validation_nll_minimum']):.4f}</td>"
        f"<td>{float(row['max_validation_accuracy']):.3f}%</td>"
        f"<td>{float(row['validation_nll_at_max_accuracy']):.4f}</td>"
        f"<td>{float(row['wall_seconds']) / 60:.1f} min</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Condition</th><th>Min-NLL epoch</th>"
        "<th>Accuracy at min NLL</th><th>Minimum NLL</th>"
        "<th>Maximum accuracy</th><th>NLL at max accuracy</th><th>Runtime</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def write_frontier_adaptation_report(run: str | Path) -> Path:
    """Generate compact ledgers, plots, Markdown, HTML, and a validated PDF."""
    run_path = Path(run).resolve()
    result = _validated_result(run_path)
    report_root = run_path / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    summaries = _summary_rows(run_path, result)
    histories = _history_rows(run_path, result)
    displacements = _displacement_rows(result)
    for name, rows in (
        ("condition_summary", summaries),
        ("epoch_history", histories),
        ("adapter_displacements", displacements),
    ):
        _write_table_family(report_root, name, rows)
    references = dict(result["references"])
    _plot_primary(report_root / "accuracy_nll_vs_h.png", summaries, references)
    _plot_learning_curves(
        report_root / "validation_learning_curves.png", histories, references
    )
    _plot_displacements(
        report_root / "adapter_displacements.png", displacements
    )
    _plot_frontier(report_root / "stage31_frontier.png")
    adaptive = tuple(row for row in summaries if row["adapt_lora"])
    best_nll = min(adaptive, key=lambda row: float(row["validation_nll_minimum"]))
    best_accuracy = max(
        adaptive, key=lambda row: float(row["max_validation_accuracy"])
    )
    first_match = next(
        (
            row
            for row in adaptive
            if row["simultaneous_joint_match_epoch"] is not None
        ),
        None,
    )
    joint = dict(references["joint_iid"])
    cached = dict(references["frozen_macro_seed1993"])
    frozen_online = next(row for row in summaries if not row["adapt_lora"])
    if first_match is None:
        match_sentence = (
            "No adaptive checkpoint simultaneously reaches both joint-IID references."
        )
    elif int(first_match["historical_capacity"]) < 11827:
        match_sentence = (
            f"{first_match['condition']} is the smallest H to reach both joint-IID "
            f"references, first doing so at epoch {first_match['simultaneous_joint_match_epoch']}."
        )
    else:
        match_sentence = (
            "Only the full-fit adaptive condition reaches both joint-IID references, "
            f"first doing so at epoch {first_match['simultaneous_joint_match_epoch']}."
        )
    markdown = f"""# ImageNet-R stage-31 frontier-LoRA adaptation

## Result

The minimum-NLL adaptive condition is **{best_nll['condition']}**, with
**{float(best_nll['validation_accuracy_at_best_nll']):.3f}% validation accuracy**
and **{float(best_nll['validation_nll_minimum']):.4f} NLL** at epoch
{int(best_nll['best_nll_epoch'])}. The largest observed adaptive accuracy is
**{float(best_accuracy['max_validation_accuracy']):.3f}%** from
**{best_accuracy['condition']}** at epoch {int(best_accuracy['max_accuracy_epoch'])},
where NLL is {float(best_accuracy['validation_nll_at_max_accuracy']):.4f}.
{match_sentence}

The matched joint-IID reference is {float(joint['accuracy']):.3f}% / {float(joint['nll']):.4f}
NLL. The previous frozen cached macro result at the same seed is
{float(cached['accuracy']):.3f}% / {float(cached['nll']):.4f}; the new
augmentation-matched frozen control is
{float(frozen_online['validation_accuracy_at_best_nll']):.3f}% /
{float(frozen_online['validation_nll_minimum']):.4f}.

![Accuracy and NLL versus H](accuracy_nll_vs_h.png)

## What changed

The task-31 frontier contains five sealed rank-16 LoRAs over disjoint task
intervals. Every adaptive condition starts from those exact tensors and the
same seed-1993 macro head. The base ViT and all five node classifiers stay
frozen; the five node LoRAs and macro head train jointly from task-free inputs.
Every population includes all 367 current-task images. H is a nested uniform
hash-order prefix of the 11,827-image historical partition, so maximum H is
exactly the 12,194-image full fit. The 3,049 validation identities remain
excluded from optimization.

![Stage-31 frontier](stage31_frontier.png)

## Optimization behavior

![Validation learning curves](validation_learning_curves.png)

![Adapter displacement](adapter_displacements.png)

The head uses the previous minimum-NLL winner: effective batch 64, peak AdamW
learning rate 3e-5, and 50 warmup-cosine epochs. Newly adaptive LoRAs use peak
5e-4 from the joint-IID recipe under the same AdamW schedule. That LoRA choice
is a starting point, not a tuned optimum. Checkpoint selection is minimum
validation NLL; maximum accuracy is a separately labeled diagnostic.

## Interpretation boundaries

This is one seed on a validation split, not a final test estimate. Repeated
epoch evaluation makes the maximum-accuracy statistic exploratory. The
full-fit frozen control isolates online augmentation and image forwarding from
LoRA adaptation. The five H cells differ in both unique identities and total
optimizer updates because each receives 50 full passes. No test identity was
requested. Exact replay authenticated all six cells with zero new optimizer
steps and left the source hierarchy unchanged.
"""
    atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    table = _table_html(summaries)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R frontier-LoRA adaptation</title><style>
@page {{ size: Letter; margin: 0.55in; }}
* {{ box-sizing: border-box; }} body {{ font-family: Arial, sans-serif; color:#17202a; margin:0; line-height:1.35; font-size:10.5pt; }}
h1 {{ font-size:23pt; margin:0 0 5px; color:#16324f; }} h2 {{ font-size:15pt; color:#16324f; margin:16px 0 7px; }}
.lede {{ font-size:12pt; color:#425466; margin:0 0 12px; }} .callout {{ background:#eef5f9; border-left:4px solid #2166ac; padding:10px 13px; margin:9px 0 12px; }}
figure {{ margin:10px 0 12px; break-inside:avoid; }} figure img {{ width:100%; max-height:5.6in; object-fit:contain; }} figcaption {{ color:#59636e; font-size:8.5pt; margin-top:3px; }}
table {{ width:100%; border-collapse:collapse; font-size:7.7pt; margin:8px 0 12px; }} th {{ background:#16324f; color:white; text-align:left; padding:5px; }} td {{ border-bottom:1px solid #ccd5dd; padding:5px; vertical-align:top; }}
.page {{ break-before:page; }} .small {{ font-size:9pt; color:#44515e; }} footer {{ margin-top:12px; border-top:1px solid #ccd5dd; padding-top:5px; color:#687580; font-size:8pt; }}
</style></head><body>
<h1>ImageNet-R stage-31 frontier-LoRA adaptation</h1><p class="lede">Can jointly adapting five fragmented node representations close the same-split joint-IID accuracy and NLL gap?</p>
<div class="callout"><strong>Primary result.</strong> The minimum-NLL adaptive condition is {escape(str(best_nll['condition']))}: <strong>{float(best_nll['validation_accuracy_at_best_nll']):.3f}% accuracy / {float(best_nll['validation_nll_minimum']):.4f} NLL</strong>. The joint-IID reference is {float(joint['accuracy']):.3f}% / {float(joint['nll']):.4f}. {escape(match_sentence)}</div>
<figure><img src="{_image_uri(report_root / 'accuracy_nll_vs_h.png')}" alt="Accuracy and NLL versus replay population"><figcaption>All blue points jointly update the five frontier LoRAs and macro head. Checkpoints minimize validation NLL. Horizontal references use the identical fit/validation identities but different architectures.</figcaption></figure>
<h2>Complete condition summary</h2>{table}<p class="small">Maximum accuracy and its NLL are reported separately to expose calibration tradeoffs. Every cell also includes all 367 current-task identities; maximum historical H=11,827 is exactly the 12,194-image full fit.</p>
<div class="page"><h2>Architecture and experimental boundary</h2><p>The task-31 frontier has five nodes at levels 0-4, covering task intervals 31, 29-30, 25-28, 17-24, and 1-16. Each node supplies its own final 197 x 768 LoRA-adapted token sequence and immutable local affine scores. The macro transformer combines them without task IDs or labels.</p>
<figure><img src="{_image_uri(report_root / 'stage31_frontier.png')}" alt="Five nodes feeding the macro-token integrator"><figcaption>The base ViT and node classifiers are frozen. Only five rank-16 LoRAs and the shared 12.06M-parameter macro head can move.</figcaption></figure>
<p>Every cell includes all 367 current-task images. H=1,024, 2,048, 4,096, 8,192, and 11,827 are nested prefixes of one deterministic uniform draw from the historical tasks without replacement. Maximum H therefore gives exactly all 12,194 fit identities. Every cell starts independently from identical sealed node tensors and macro initialization. The validation partition has 3,049 clean identities and never contributes gradients. No test image is opened.</p>
<h2>Why the frozen online control exists</h2><p>The previous frozen macro was trained from cached center-crop tokens. This run loads images online with deterministic random training augmentation. The purple full-fit control follows that new path while freezing the LoRAs, so its difference from the cached gray reference measures the pipeline/augmentation change rather than representation adaptation.</p></div>
<div class="page"><h2>Optimization behavior</h2><figure><img src="{_image_uri(report_root / 'validation_learning_curves.png')}" alt="Validation learning curves"><figcaption>Every epoch is retained in a hash-chained history. The dashed black line is the five-epoch joint-IID endpoint, not a stopping gate.</figcaption></figure>
<figure><img src="{_image_uri(report_root / 'adapter_displacements.png')}" alt="Relative LoRA update displacement"><figcaption>Scale-aware Frobenius movement of each dense LoRA update, measured from its sealed source node at the selected minimum-NLL checkpoint.</figcaption></figure></div>
<div class="page"><h2>Interpretation</h2><p>The head schedule is the previous minimum-NLL winner: effective batch 64, peak AdamW LR 3e-5, 50 epochs, 5% warmup, and cosine decay. The LoRA peak LR 5e-4 is imported from the same-split joint recipe but run here in AdamW under the shared schedule. It has not been tuned for this coupled model.</p>
<p><strong>Limits.</strong> This is a one-seed validation screen. The H cells change both unique data and total optimizer updates. Maximum accuracy is exploratory because all 50 validation checkpoints are visible. A promising condition needs replication and a focused LoRA/head learning-rate audit before any locked-test use.</p>
<h2>Integrity and reuse</h2><p>The training seal records zero fit-validation overlap, zero test overlap, zero test evaluations, and unchanged source hierarchy files. A second pass authenticated all six completed cells without constructing their five ViTs or taking an optimizer step. Large checkpoints and model weights remain local; compact histories, tables, plots, and protocol records are the report surface.</p>
<footer>Protocol {escape(str(result['content_hash']))[:16]}... | stage 31 | seed 1993 | generated from authenticated result and history ledgers</footer></div>
</body></html>"""
    atomic_write(report_root / "REPORT.html", html.encode("utf-8"))
    return render_integrator_pdf(report_root / "REPORT.html")


__all__ = [
    "CONDITION_LABELS",
    "FROZEN_CACHED_LABEL",
    "FROZEN_ONLINE_LABEL",
    "JOINT_LABEL",
    "write_frontier_adaptation_report",
]
