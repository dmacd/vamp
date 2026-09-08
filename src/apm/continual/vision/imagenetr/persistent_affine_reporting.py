"""Publication-style reporting for the full-series persistent affine experiment."""

from __future__ import annotations

import base64
import csv
from html import escape
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Final

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)


LABEL_H4096: Final[str] = "Persistent single-affine + adaptive node LoRAs, H=4,096"
LABEL_H8192: Final[str] = "Persistent single-affine + adaptive node LoRAs, H=8,192"
LABEL_STAGE_JOINT: Final[str] = "Stage-matched joint IID, rank 16"
LABEL_RANK_JOINT: Final[str] = "Aggregate-rank-matched joint IID"
LABEL_ORACLE_H4096: Final[str] = "True-node oracle for persistent H=4,096 (diagnostic)"
LABEL_ORACLE_H8192: Final[str] = "True-node oracle for persistent H=8,192 (diagnostic)"


def _validate_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-persistent-affine-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or set(result.get("arms", {})) != {"4096", "8192"}
        or any(len(rows) != 50 for rows in result["arms"].values())
        or len(result.get("stage_matched_joint", ())) != 50
        or len(result.get("rank_matched_joint", ())) != 50
    ):
        raise ValueError("persistent affine result is incomplete or unauthenticated")
    return result


def _series(result: dict[str, object]) -> dict[str, list[float]]:
    arms = result["arms"]
    return {
        LABEL_H4096: [float(row["evaluation"]["accuracy"]) for row in arms["4096"]],
        LABEL_H8192: [float(row["evaluation"]["accuracy"]) for row in arms["8192"]],
        LABEL_STAGE_JOINT: [float(row["accuracy"]) for row in result["stage_matched_joint"]],
        LABEL_RANK_JOINT: [float(row["accuracy"]) for row in result["rank_matched_joint"]],
        LABEL_ORACLE_H4096: [
            float(row["evaluation"]["true_node_oracle_accuracy"])
            for row in arms["4096"]
        ],
        LABEL_ORACLE_H8192: [
            float(row["evaluation"]["true_node_oracle_accuracy"])
            for row in arms["8192"]
        ],
    }


def _summary_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    series = _series(result)
    rows = []
    for label, values in series.items():
        final = values[-1]
        rows.append(
            {
                "condition": label,
                "final_accuracy": final,
                "incremental_accuracy": math.fsum(values) / len(values),
                "fragmented_stage_accuracy": math.fsum(
                    value for stage, value in enumerate(values, 1) if stage.bit_count() > 1
                )
                / sum(stage.bit_count() > 1 for stage in range(1, 51)),
                "one_node_stage_accuracy": math.fsum(
                    value for stage, value in enumerate(values, 1) if stage.bit_count() == 1
                )
                / sum(stage.bit_count() == 1 for stage in range(1, 51)),
            }
        )
    return tuple(rows)


def _stage_rows(result: dict[str, object]) -> tuple[dict[str, object], ...]:
    arms = result["arms"]
    stage_joint = result["stage_matched_joint"]
    rank_joint = result["rank_matched_joint"]
    rows = []
    for index in range(50):
        stage = index + 1
        h4, h8 = arms["4096"][index], arms["8192"][index]
        rows.append(
            {
                "stage": stage,
                "live_nodes": stage.bit_count(),
                "rank_matched_joint_rank": int(rank_joint[index]["rank"]),
                "persistent_affine_h4096_accuracy": float(h4["evaluation"]["accuracy"]),
                "persistent_affine_h4096_nll": float(h4["evaluation"]["nll"]),
                "persistent_affine_h8192_accuracy": float(h8["evaluation"]["accuracy"]),
                "persistent_affine_h8192_nll": float(h8["evaluation"]["nll"]),
                "stage_matched_joint_rank16_accuracy": float(stage_joint[index]["accuracy"]),
                "rank_matched_joint_accuracy": float(rank_joint[index]["accuracy"]),
                "rank_matched_joint_nll": rank_joint[index]["nll"],
                "h4096_gap_to_stage_joint": float(h4["evaluation"]["accuracy"]) - float(stage_joint[index]["accuracy"]),
                "h8192_gap_to_stage_joint": float(h8["evaluation"]["accuracy"]) - float(stage_joint[index]["accuracy"]),
                "h4096_gap_to_rank_joint": float(h4["evaluation"]["accuracy"]) - float(rank_joint[index]["accuracy"]),
                "h8192_gap_to_rank_joint": float(h8["evaluation"]["accuracy"]) - float(rank_joint[index]["accuracy"]),
                "h4096_true_node_oracle_accuracy": float(h4["evaluation"]["true_node_oracle_accuracy"]),
                "h8192_true_node_oracle_accuracy": float(h8["evaluation"]["true_node_oracle_accuracy"]),
            }
        )
    return tuple(rows)


def _write_tables(reports: Path, stage_rows: tuple[dict[str, object], ...], summaries: tuple[dict[str, object], ...]) -> None:
    for name, rows in (("stage_metrics", stage_rows), ("condition_summary", summaries)):
        json_path = reports / f"{name}.json"
        atomic_write(json_path, canonical_json_bytes(list(rows)))
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(reports / f"{name}.csv", buffer.getvalue().encode("utf-8"))
        try:
            import pandas as pd

            temporary = reports / f".{name}.parquet.tmp"
            pd.DataFrame(rows).to_parquet(temporary, index=False)
            os.replace(temporary, reports / f"{name}.parquet")
        except ImportError:  # pragma: no cover - environment gate catches this in runs
            pass


def _plot_accuracy(reports: Path, result: dict[str, object]) -> Path:
    import matplotlib.pyplot as plt

    stages = list(range(1, 51))
    series = _series(result)
    colors = {
        LABEL_H4096: "#1f77b4",
        LABEL_H8192: "#d95f02",
        LABEL_STAGE_JOINT: "#222222",
        LABEL_RANK_JOINT: "#2ca02c",
        LABEL_ORACLE_H4096: "#6baed6",
        LABEL_ORACLE_H8192: "#fdae6b",
    }
    styles = {
        LABEL_H4096: "-",
        LABEL_H8192: "-",
        LABEL_STAGE_JOINT: "--",
        LABEL_RANK_JOINT: "-.",
        LABEL_ORACLE_H4096: ":",
        LABEL_ORACLE_H8192: ":",
    }
    figure, axis = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    for label, values in series.items():
        axis.plot(
            stages,
            values,
            label=label,
            color=colors[label],
            linestyle=styles[label],
            linewidth=1.7 if "oracle" in label.lower() else 2.2,
        )
    for stage in (2, 4, 8, 16, 32):
        axis.axvline(stage, color="#aaaaaa", linewidth=0.7, alpha=0.45)
    axis.set(xlabel="Tasks observed", ylabel="Test top-1 accuracy (%)", xlim=(1, 50))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right", fontsize=7.5, frameon=True, ncol=2)
    axis.set_title("Full-stream accuracy with stage-matched data prefixes")
    path = reports / "stage_accuracy.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_gaps(reports: Path, stage_rows: tuple[dict[str, object], ...]) -> Path:
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in stage_rows]
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), sharex=True, constrained_layout=True)
    for label, prefix, color in (
        ("H=4,096 minus joint IID", "h4096", "#1f77b4"),
        ("H=8,192 minus joint IID", "h8192", "#d95f02"),
    ):
        axes[0].plot(stages, [float(row[f"{prefix}_gap_to_stage_joint"]) for row in stage_rows], label=label, color=color, linewidth=2)
        axes[1].plot(stages, [float(row[f"{prefix}_gap_to_rank_joint"]) for row in stage_rows], label=label, color=color, linewidth=2)
    for axis, title in zip(
        axes,
        ("Gap to stage-matched joint IID, rank 16", "Gap to aggregate-rank-matched joint IID"),
        strict=True,
    ):
        axis.axhline(0, color="#222222", linewidth=1)
        axis.set_ylabel("Accuracy gap (points)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8.5)
    axes[-1].set_xlabel("Tasks observed")
    path = reports / "joint_iid_gaps.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_nll(reports: Path, result: dict[str, object]) -> Path:
    import matplotlib.pyplot as plt

    stages = list(range(1, 51))
    arms = result["arms"]
    rank = result["rank_matched_joint"]
    figure, axis = plt.subplots(figsize=(11.2, 5.3), constrained_layout=True)
    axis.plot(stages, [float(row["evaluation"]["nll"]) for row in arms["4096"]], label=LABEL_H4096, color="#1f77b4", linewidth=2)
    axis.plot(stages, [float(row["evaluation"]["nll"]) for row in arms["8192"]], label=LABEL_H8192, color="#d95f02", linewidth=2)
    rank_stages = [stage for stage, row in zip(stages, rank, strict=True) if row["nll"] is not None]
    rank_values = [float(row["nll"]) for row in rank if row["nll"] is not None]
    axis.plot(rank_stages, rank_values, label=LABEL_RANK_JOINT, color="#2ca02c", linestyle="-.", linewidth=2)
    axis.set(xlabel="Tasks observed", ylabel="Test negative log likelihood", xlim=(1, 50))
    axis.set_title("Test NLL (rank-16 source artifacts did not retain logits)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8.2)
    path = reports / "stage_nll.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _png_data(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _selected_table(stage_rows: tuple[dict[str, object], ...]) -> str:
    selected = {1, 2, 4, 8, 16, 31, 32, 50}
    lines = [
        "| Stage | Nodes | H=4,096 | H=8,192 | H4 oracle | H8 oracle | Joint r16 | Rank-matched | Matched rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        if int(row["stage"]) not in selected:
            continue
        lines.append(
            "| {stage} | {live_nodes} | {persistent_affine_h4096_accuracy:.3f}% | "
            "{persistent_affine_h8192_accuracy:.3f}% | {h4096_true_node_oracle_accuracy:.3f}% | "
            "{h8192_true_node_oracle_accuracy:.3f}% | {stage_matched_joint_rank16_accuracy:.3f}% | "
            "{rank_matched_joint_accuracy:.3f}% | {rank_matched_joint_rank} |".format(**row)
        )
    return "\n".join(lines)


def _interpretation(summaries: tuple[dict[str, object], ...]) -> str:
    by_name = {str(row["condition"]): row for row in summaries}
    h4, h8 = by_name[LABEL_H4096], by_name[LABEL_H8192]
    stage, rank = by_name[LABEL_STAGE_JOINT], by_name[LABEL_RANK_JOINT]
    return (
        f"At task 50, H=4,096 reaches {h4['final_accuracy']:.3f}% and H=8,192 reaches "
        f"{h8['final_accuracy']:.3f}%. The corresponding rank-16 and aggregate-rank joint-IID "
        f"references are {stage['final_accuracy']:.3f}% and {rank['final_accuracy']:.3f}%. "
        f"Across all 50 stages, incremental accuracy is {h4['incremental_accuracy']:.3f}% for H=4,096 "
        f"and {h8['incremental_accuracy']:.3f}% for H=8,192, versus {stage['incremental_accuracy']:.3f}% "
        f"and {rank['incremental_accuracy']:.3f}% for the two joint-IID curves."
    )


def _write_markdown(
    reports: Path,
    result: dict[str, object],
    stage_rows: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
) -> Path:
    text = f"""# ImageNet-R-50 persistent single-affine frontier

## Result

{_interpretation(summaries)}

![Full-stream test accuracy](stage_accuracy.png)

The solid blue/orange curves are the only adaptive frontier conditions. Both use
one affine layer over the live nodes' 768-value pre-classifier vectors; neither
uses macro tokens or metadata. The dashed black curve is a fresh rank-16 joint
fit at every data prefix. The green curve is also fresh joint IID, with rank
`16 x popcount(stage)` so its LoRA rank equals the sum across live frontier
nodes. Dotted blue/orange curves are label-aware true-node diagnostics for the
corresponding adapted frontiers. Vertical guides mark power-of-two consolidation
stages.

## Selected stages

{_selected_table(stage_rows)}

![Accuracy gaps to joint-IID controls](joint_iid_gaps.png)

Positive values mean the persistent affine frontier is ahead. The top panel
holds joint rank fixed at 16; the bottom panel adjusts joint rank to the number
of live nodes. Neither reference is an execution gate.

![Test NLL](stage_nll.png)

NLL is shown for both persistent arms and newly trained aggregate-rank models.
The imported one-node rank-16 artifacts retained predictions but not logits,
so those six green NLL points are intentionally absent; accuracy is complete.

## Protocol

- H=4,096 uses four epochs per arrival; H=8,192 uses five.
- Replay is class-stratified and deterministically redrawn at every stage.
- The affine head and AdamW state persist. A node LoRA and its moments persist
  only while the exact hierarchy-node hash remains live.
- A leaf or consolidation parent enters from its authenticated source model.
- Local classifiers and the ViT base stay frozen. Evaluation is task-free.
- The test split is used only after each stage is sealed and never selects a
  checkpoint, replay population, or condition.

## Resource and provenance

This report authenticates result `{result['content_hash']}` under protocol
`{result['protocol_hash']}`. Source hierarchy unchanged:
`{result['hierarchy_source_unchanged']}`. New work in the completing invocation:
`{json.dumps(result['invocation_work'], sort_keys=True)}`.
"""
    path = reports / "REPORT.md"
    atomic_write(path, text.encode("utf-8"))
    return path


def _write_html(
    reports: Path,
    result: dict[str, object],
    stage_rows: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    images: tuple[Path, ...],
) -> Path:
    selected = tuple(
        row
        for row in stage_rows
        if int(row["stage"]) in {1, 2, 4, 8, 16, 31, 32, 50}
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{row['stage']}</td><td>{row['live_nodes']}</td>"
        f"<td>{row['persistent_affine_h4096_accuracy']:.3f}%</td>"
        f"<td>{row['persistent_affine_h8192_accuracy']:.3f}%</td>"
        f"<td>{row['h4096_true_node_oracle_accuracy']:.3f}%</td>"
        f"<td>{row['h8192_true_node_oracle_accuracy']:.3f}%</td>"
        f"<td>{row['stage_matched_joint_rank16_accuracy']:.3f}%</td>"
        f"<td>{row['rank_matched_joint_accuracy']:.3f}%</td>"
        f"<td>{row['rank_matched_joint_rank']}</td></tr>"
        for row in selected
    )
    embedded = tuple(_png_data(image) for image in images)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ImageNet-R-50 persistent single-affine frontier</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1120px;margin:36px auto;color:#20252a;padding:0 22px}}
h1,h2{{color:#16324f}} img{{max-width:100%;height:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #b8c2cc;padding:7px;text-align:right}} th{{background:#16324f;color:white}}
code{{background:#eef2f5;padding:2px 4px}} .note{{color:#4b5563}}
</style></head><body>
<h1>ImageNet-R-50 persistent single-affine frontier</h1>
<p>{escape(_interpretation(summaries))}</p>
<img src="data:image/png;base64,{embedded[0]}" alt="Full-stream test accuracy">
<p class="note">The solid blue and orange curves are the only adaptive-frontier conditions. They use a single affine layer over node-specific pre-classifier vectors, with no macro tokens or metadata. The dotted curves are the corresponding label-aware true-node diagnostics.</p>
<h2>Selected stages</h2>
<table><thead><tr><th>Stage</th><th>Nodes</th><th>H=4,096</th><th>H=8,192</th><th>H4 oracle</th><th>H8 oracle</th><th>Joint r16</th><th>Rank-matched</th><th>Rank</th></tr></thead><tbody>{table_rows}</tbody></table>
<img src="data:image/png;base64,{embedded[1]}" alt="Accuracy gaps to joint-IID controls">
<p>Positive values mean the persistent affine frontier is ahead of the named joint-IID reference. Neither reference is an execution gate.</p>
<img src="data:image/png;base64,{embedded[2]}" alt="Test negative log likelihood">
<h2>Protocol</h2>
<ul><li>H=4,096 uses four epochs per arrival; H=8,192 uses five.</li>
<li>Replay is class-stratified and deterministically redrawn at every stage.</li>
<li>Affine parameters and AdamW state persist. Node LoRAs persist exactly while their hierarchy-node hashes remain live.</li>
<li>New leaves and consolidation parents enter from authenticated source models. The base ViT and local classifiers stay frozen.</li>
<li>The test split is used only after each stage is sealed.</li></ul>
<h2>Provenance</h2>
<p>Protocol <code>{escape(str(result['protocol_hash']))}</code><br>
Result <code>{escape(str(result['content_hash']))}</code><br>
Source hierarchy unchanged: <code>{result['hierarchy_source_unchanged']}</code><br>
Completing-invocation work: <code>{escape(json.dumps(result['invocation_work'], sort_keys=True))}</code></p>
</body></html>"""
    path = reports / "REPORT.html"
    atomic_write(path, html.encode("utf-8"))
    return path


def _write_pdf(
    output: Path,
    result: dict[str, object],
    summaries: tuple[dict[str, object], ...],
    stage_rows: tuple[dict[str, object], ...],
    images: tuple[Path, ...],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def wrapped(text: str, width: int = 108) -> str:
        return "\n".join(textwrap.wrap(text, width=width))

    def footer(figure, page: int) -> None:
        figure.text(
            0.06,
            0.025,
            "ImageNet-R-50 persistent single-affine frontier",
            fontsize=8,
            color="#666666",
        )
        figure.text(
            0.94,
            0.025,
            f"Page {page}",
            fontsize=8,
            color="#666666",
            horizontalalignment="right",
        )

    def image_axis(figure, image: Path, bounds: tuple[float, float, float, float]) -> None:
        axis = figure.add_axes(bounds)
        axis.imshow(plt.imread(image))
        axis.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".persistent-affine-", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    selected = tuple(
        row
        for row in stage_rows
        if int(row["stage"]) in {1, 2, 4, 8, 16, 31, 32, 50}
    )
    with PdfPages(
        temporary,
        metadata={
            "Title": "ImageNet-R-50 persistent single-affine frontier",
            "Author": "APM continual-learning experiment",
            "Subject": "Full 50-task adaptive-LoRA single-affine experiment",
        },
    ) as document:
        first = plt.figure(figsize=(8.5, 11), facecolor="white")
        first.text(
            0.5,
            0.95,
            "ImageNet-R-50 persistent single-affine frontier",
            fontsize=17,
            weight="bold",
            color="#16324f",
            horizontalalignment="center",
        )
        first.text(
            0.5,
            0.915,
            "Full 50-task extension of the H=4,096 and H=8,192 adaptive-LoRA conditions",
            fontsize=11.5,
            horizontalalignment="center",
        )
        first.text(
            0.06,
            0.865,
            wrapped(_interpretation(summaries), 90),
            fontsize=10.2,
            va="top",
        )
        image_axis(first, images[0], (0.055, 0.30, 0.89, 0.46))
        first.text(
            0.06,
            0.265,
            wrapped(
            "The adaptive curves use only one affine layer over node-specific pre-classifier vectors. Their dotted true-node curves are label-aware diagnostics, not deployable conditions. The black control is fresh rank-16 joint IID at each prefix. The green control uses rank 16 times the number of live nodes."
            ),
            fontsize=9.2,
            va="top",
            color="#40464d",
        )
        footer(first, 1)
        document.savefig(first)
        plt.close(first)

        second = plt.figure(figsize=(8.5, 11), facecolor="white")
        second.text(0.06, 0.95, "Selected stages and control gaps", fontsize=17, weight="bold", color="#16324f")
        table_axis = second.add_axes((0.055, 0.67, 0.89, 0.22))
        table_axis.axis("off")
        table = table_axis.table(
            cellText=[
                [
                    row["stage"], row["live_nodes"],
                    f"{row['persistent_affine_h4096_accuracy']:.2f}%",
                    f"{row['persistent_affine_h8192_accuracy']:.2f}%",
                    f"{row['h4096_true_node_oracle_accuracy']:.2f}%",
                    f"{row['h8192_true_node_oracle_accuracy']:.2f}%",
                    f"{row['stage_matched_joint_rank16_accuracy']:.2f}%",
                    f"{row['rank_matched_joint_accuracy']:.2f}%",
                    row["rank_matched_joint_rank"],
                ]
                for row in selected
            ],
            colLabels=(
                "Stage", "Nodes", "H=4,096", "H=8,192", "H4 oracle",
                "H8 oracle", "Joint r16", "Rank-matched", "Rank",
            ),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.45)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#b8c2cc")
            if row == 0:
                cell.set_facecolor("#16324f")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f2f5f7")
        image_axis(second, images[1], (0.055, 0.12, 0.89, 0.51))
        second.text(
            0.06,
            0.09,
            wrapped(
                "Positive gap means the persistent frontier is ahead of the named joint-IID reference. Power-of-two stages have one node; fragmented stages have two or more.",
                105,
            ),
            fontsize=8.8,
            color="#40464d",
            va="top",
        )
        footer(second, 2)
        document.savefig(second)
        plt.close(second)

        third = plt.figure(figsize=(8.5, 11), facecolor="white")
        third.text(0.06, 0.95, "Likelihood and protocol", fontsize=17, weight="bold", color="#16324f")
        image_axis(third, images[2], (0.055, 0.51, 0.89, 0.38))
        third.text(
            0.06,
            0.49,
            wrapped(
                "Rank-16 source artifacts retained hard predictions rather than logits, so rank-matched NLL is absent at stages 1, 2, 4, 8, 16, and 32. Accuracy remains available at all 50 stages."
            ),
            fontsize=8.8,
            va="top",
            color="#40464d",
        )
        third.text(
            0.06,
            0.415,
            "Exact online update rule",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.39,
            wrapped(
                "Every arrival uses all current-task images plus a deterministic stage-keyed class-stratified historical draw. H=4,096 receives four passes and H=8,192 receives five. AdamW state and affine parameters carry forward. A node adapter and its moments carry only if its hierarchy-node hash survives; new leaves and consolidation parents start from authenticated source adapters. The base ViT and local classifiers never train.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        third.text(
            0.06,
            0.275,
            "Comparison boundary",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.25,
            wrapped(
                "Stage-matched joint IID trains one fresh rank-16 adapter for five epochs on every prefix. Aggregate-rank-matched joint IID uses rank and alpha equal to 16 times popcount(stage). It matches total live LoRA rank, but not pretrained node state, multiple ViT paths, affine-head parameters, or deployment compute. Neither curve gates execution.",
                105,
            ),
            fontsize=9.1,
            va="top",
        )
        third.text(
            0.06,
            0.145,
            "Provenance",
            fontsize=12,
            weight="bold",
            color="#16324f",
        )
        third.text(
            0.06,
            0.12,
            f"Protocol {result['protocol_hash']}\nResult {result['content_hash']}\nSource hierarchy unchanged: {result['hierarchy_source_unchanged']}",
            fontsize=7.7,
            va="top",
            family="monospace",
        )
        footer(third, 3)
        document.savefig(third)
        plt.close(third)
    os.replace(temporary, output)


def write_persistent_affine_report(run: str | Path) -> Path:
    """Build tables, figures, Markdown/HTML projections, and the final PDF."""
    run_path = Path(run).resolve()
    result = _validate_result(run_path)
    reports = run_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summaries = _summary_rows(result)
    stage_rows = _stage_rows(result)
    _write_tables(reports, stage_rows, summaries)
    images = (
        _plot_accuracy(reports, result),
        _plot_gaps(reports, stage_rows),
        _plot_nll(reports, result),
    )
    markdown = _write_markdown(reports, result, stage_rows, summaries)
    html = _write_html(reports, result, stage_rows, summaries, images)
    project_root = run_path.parents[4]
    pdf = project_root / "output/pdf/imagenetr50_persistent_affine_v15_report.pdf"
    _write_pdf(pdf, result, summaries, stage_rows, images)
    manifest_core = {
        "condition_summary_sha256": file_sha256(reports / "condition_summary.json"),
        "html_sha256": file_sha256(html),
        "markdown_sha256": file_sha256(markdown),
        "pdf": str(pdf),
        "pdf_sha256": file_sha256(pdf),
        "result_hash": result["content_hash"],
        "schema_version": "imagenetr50-persistent-affine-report-manifest-v1",
        "stage_metrics_sha256": file_sha256(reports / "stage_metrics.json"),
    }
    atomic_write(
        reports / "report_manifest.json",
        canonical_json_bytes({**manifest_core, "content_hash": record_sha256(manifest_core)}),
    )
    return pdf


__all__ = [
    "LABEL_H4096",
    "LABEL_H8192",
    "LABEL_ORACLE_H4096",
    "LABEL_ORACLE_H8192",
    "LABEL_RANK_JOINT",
    "LABEL_STAGE_JOINT",
    "write_persistent_affine_report",
]
