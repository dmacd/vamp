"""Standalone publication for joint-IID LoRA plus a tied embedding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from html import escape
import io
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
    atomic_write,
    file_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_lora_embedding import (
    LoraEmbeddingInputs,
    RANKS,
    REPORT_FORMAT,
)


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


def publish_lora_embedding_report(
    inputs: LoraEmbeddingInputs,
    analysis: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Publish deterministic CSVs, an accessible plot, and two reports."""
    output = inputs.result_directory
    output.mkdir(parents=True, exist_ok=True)
    analysis_core = {
        "analysis": dict(analysis),
        "contract_sha256": inputs.contract_sha256,
        "format": REPORT_FORMAT,
    }
    atomic_write(
        output / "analysis.json",
        canonical_json_bytes(
            {**analysis_core, "analysis_sha256": record_sha256(analysis_core)}
        ),
    )
    exports = {
        "aggregate.csv": _records(analysis["aggregate"], "aggregate"),
        "bootstrap.csv": _records(analysis["bootstrap"], "bootstrap"),
        "embedding-only.csv": _records(analysis["embedding_only"], "embedding-only"),
        "ledger-provenance.csv": _records(
            analysis["ledger_provenance"], "ledger provenance"
        ),
        "per-task.csv": _records(analysis["per_task"], "per-task"),
        "training.csv": _records(analysis["training"], "training"),
    }
    for name, rows in exports.items():
        atomic_write(output / name, _csv_bytes(rows))
    plot_path = _publish_plot(output, analysis)
    markdown_path = atomic_write(
        output / "report.md",
        _markdown(inputs, analysis).encode("utf-8"),
    )
    html_path = atomic_write(
        output / "report.html",
        _html(inputs, analysis, plot_path.read_text(encoding="utf-8")).encode(
            "utf-8"
        ),
    )
    artifact_names = (
        "aggregate.csv",
        "allocator.json",
        "analysis.json",
        "bootstrap.csv",
        "contract.json",
        "embedding-lora-nll.svg",
        "embedding-only.csv",
        "execution.json",
        "ledger-provenance.csv",
        "per-task.csv",
        "report.html",
        "report.md",
        "training.csv",
    )
    if any(not (output / name).is_file() for name in artifact_names):
        raise FileNotFoundError("embedding-LoRA publication bundle is incomplete")
    manifest_core = {
        "artifacts": {
            name: file_sha256(output / name) for name in sorted(artifact_names)
        },
        "contract_sha256": inputs.contract_sha256,
        "format": f"{REPORT_FORMAT}-manifest",
        "schema_version": 1,
    }
    manifest_path = atomic_write(
        output / "manifest.json",
        canonical_json_bytes(
            {**manifest_core, "manifest_sha256": record_sha256(manifest_core)}
        ),
    )
    return markdown_path, html_path, manifest_path


def _publish_plot(output: Path, analysis: Mapping[str, object]) -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": "tinyworlds-nouns-v2-joint-iid-lora-embedding",
        }
    )
    aggregate = {
        str(row["condition"]): row
        for row in _records(analysis["aggregate"], "aggregate")
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    for axis, (metric, title) in zip(
        axes,
        (
            ("story_mean_nll", "Story-weighted suffix NLL"),
            ("token_mean_nll", "Token-weighted suffix NLL"),
        ),
        strict=True,
    ):
        for prefix, label, color, marker in (
            ("lora_rank", "Projection LoRA", "#0072B2", "o"),
            (
                "lora_embedding_rank",
                "Projection LoRA + tied embedding",
                "#009E73",
                "s",
            ),
        ):
            values = [float(aggregate[f"{prefix}_{rank}"][metric]) for rank in RANKS]
            axis.plot(
                RANKS,
                values,
                color=color,
                marker=marker,
                markersize=7,
                linewidth=2.2,
                label=label,
            )
            for rank, value in zip(RANKS, values, strict=True):
                axis.annotate(
                    f"{value:.4f}",
                    (rank, value),
                    xytext=(0, 8 if prefix == "lora_rank" else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        full = float(aggregate["full_model"][metric])
        axis.axhline(
            full,
            color="#D55E00",
            linestyle="--",
            linewidth=2,
            label="Joint-IID full model",
        )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("LoRA rank")
        axis.set_ylabel("Negative log-likelihood (nats/token)")
        axis.set_xticks(RANKS, [str(rank) for rank in RANKS])
        axis.grid(axis="y", alpha=0.28)
        axis.legend(fontsize=8.5)
    title = "Effect of jointly training the tied embedding with projection LoRA"
    description = (
        "Two panels compare projection-only LoRA and projection LoRA plus a "
        "jointly trained tied input-output embedding at ranks 8 and 32. A "
        "dashed line marks the authenticated joint-IID full-model control."
    )
    stream = io.StringIO()
    figure.savefig(
        stream,
        format="svg",
        metadata={"Date": None, "Description": description, "Title": title},
    )
    plt.close(figure)
    return atomic_write(
        output / "embedding-lora-nll.svg",
        _accessible_svg(stream.getvalue(), title, description).encode("utf-8"),
    )


def _markdown(
    inputs: LoraEmbeddingInputs,
    analysis: Mapping[str, object],
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    embedding_only = _records(analysis["embedding_only"], "embedding-only")
    training = _records(analysis["training"], "training")
    headline = _finding(aggregate)
    aggregate_rows = "\n".join(
        f"| {row['label']} | {_rank(row['rank'])} | "
        f"{float(row['story_mean_nll']):.6f} | {float(row['token_mean_nll']):.6f} | "
        f"{100 * float(row['suffix_token_accuracy']):.3f}% |"
        for row in aggregate
    )
    interval_rows = "\n".join(
        f"| {str(row['condition']).replace('_', ' ')} | "
        f"{str(row['reference']).replace('_', ' ')} | "
        f"{str(row['metric']).replace('_', ' ')} | {float(row['estimate']):+.6f} | "
        f"[{float(row['lower_95']):+.6f}, {float(row['upper_95']):+.6f}] |"
        for row in bootstrap
    )
    training_rows = "\n".join(
        f"| {int(row['rank'])} | {int(row['adapter_parameter_count']):,} | "
        f"{int(row['embedding_parameter_count']):,} | "
        f"{int(row['total_trainable_parameter_count']):,} | "
        f"{100 * float(row['trainable_to_base_parameter_fraction']):.2f}% | "
        f"{float(row['final_training_loss']):.5f} | "
        f"{float(row['embedding_delta_relative_frobenius']):.4f} | "
        f"{float(row['runtime_seconds']) / 60:.1f} min |"
        for row in training
    )
    embedding_rows = "\n".join(
        f"| {int(row['rank'])} | {float(row['story_mean_nll']):.6f} | "
        f"{float(row['token_mean_nll']):.6f} |"
        for row in embedding_only
    )
    task_rows = "\n".join(
        f"| {row['task_id']} | {row['label']} | "
        f"{float(row['story_mean_nll']):.5f} | {float(row['token_mean_nll']):.5f} | "
        f"{100 * float(row['suffix_token_accuracy']):.2f}% |"
        for row in _records(analysis["per_task"], "per-task")
    )
    provenance = _object(analysis["provenance"], "provenance")
    comparability = _object(analysis["comparability"], "comparability")
    allocator = _object(analysis["allocator"], "allocator")
    execution = _object(analysis["execution"], "execution")
    return f"""# TinyWorlds nouns-v2 joint-IID LoRA plus tied embedding

{headline}

This addendum tests whether the frozen token embedding/output classifier caused
the projection-only LoRA gap. It uses the exact 98,304-story joint-IID training
population and the exact 4,440-story final suffix evaluation from the
[temporal-consolidation report](../report.md).

![Joint-IID LoRA with and without a trained tied embedding](embedding-lora-nll.svg)

| Condition | Rank | Story NLL | Token NLL | Suffix token accuracy |
|---|---:|---:|---:|---:|
{aggregate_rows}

Story NLL weights every story equally. Token NLL weights all 476,035 evaluator-only
suffix targets equally. Accuracy is teacher-forced next-token accuracy, not
routing accuracy.

<details>
<summary>Method and trainable parameters</summary>

The new conditions train all six LoRA projections in every transformer block
and one tied token matrix used both for input lookup and output logits. The
original transformer kernels, position embedding, layer norms, and biases stay
frozen. Both ranks use alpha equal to rank, hence LoRA scale one.

One joint loss and one combined global-norm clip feed two AdamW groups: LoRA at
`1e-3` and the tied embedding at `5e-5`; both use weight decay `0.01`. Training
uses four epochs, batch 32, context 256, and 15,024 updates.

| Rank | LoRA params | Embedding params | Total trainable | Base fraction | Final train loss | Embedding relative displacement | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
{training_rows}

The trained-embedding-only diagnostic disables the jointly learned LoRA at
evaluation time. It is not a separately optimized baseline and therefore
should not be interpreted as an additive decomposition.

| Training rank | Embedding-only story NLL | Embedding-only token NLL |
|---:|---:|---:|
{embedding_rows}

</details>

<details>
<summary>Paired uncertainty</summary>

Differences are condition minus reference. Intervals use the deterministic
seed-zero 10,000-sample paired bootstrap stratified by noun; negative NLL
favors the condition.

| Condition | Reference | Metric | Difference | 95% interval |
|---|---|---|---:|---:|
{interval_rows}

</details>

<details>
<summary>Per-noun results</summary>

| Noun | Condition | Story NLL | Token NLL | Token accuracy |
|---|---|---:|---:|---:|
{task_rows}

</details>

<details>
<summary>Provenance and execution</summary>

- Contract: `{inputs.contract_sha256}`
- Parent rank-sweep contract: `{provenance['parent_rank_sweep_contract_sha256']}`
- Parent rank-sweep manifest: `{provenance['parent_rank_sweep_manifest_sha256']}`
- Exact batch/random namespace: `{comparability['batch_namespace_sha256']}`
- Exact suffix targets: {int(comparability['exact_suffix_target_count']):,}
- Allocator peak: {int(allocator['peak_bytes_in_use']) / 1024**3:.2f} GiB of 12 GiB
- End-to-end runtime: {float(execution['end_to_end_seconds']) / 60:.1f} minutes

CSV exports beside this report preserve aggregate, per-task, uncertainty,
training, embedding-only, and ledger-provenance records.

</details>
"""


def _html(
    inputs: LoraEmbeddingInputs,
    analysis: Mapping[str, object],
    svg: str,
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    training = _records(analysis["training"], "training")
    embedding_only = _records(analysis["embedding_only"], "embedding-only")
    per_task = _records(analysis["per_task"], "per-task")
    allocator = _object(analysis["allocator"], "allocator")
    execution = _object(analysis["execution"], "execution")
    provenance = _object(analysis["provenance"], "provenance")
    aggregate_table = _table(
        ("Condition", "Rank", "Story NLL", "Token NLL", "Token accuracy"),
        tuple(
            (
                str(row["label"]),
                _rank(row["rank"]),
                f"{float(row['story_mean_nll']):.6f}",
                f"{float(row['token_mean_nll']):.6f}",
                f"{100 * float(row['suffix_token_accuracy']):.3f}%",
            )
            for row in aggregate
        ),
    )
    interval_table = _table(
        ("Condition", "Reference", "Metric", "Difference", "95% interval"),
        tuple(
            (
                str(row["condition"]).replace("_", " "),
                str(row["reference"]).replace("_", " "),
                str(row["metric"]).replace("_", " "),
                f"{float(row['estimate']):+.6f}",
                f"[{float(row['lower_95']):+.6f}, {float(row['upper_95']):+.6f}]",
            )
            for row in bootstrap
        ),
    )
    training_table = _table(
        ("Rank", "LoRA params", "Embedding params", "Total", "Base fraction", "Final loss", "Runtime"),
        tuple(
            (
                str(row["rank"]),
                f"{int(row['adapter_parameter_count']):,}",
                f"{int(row['embedding_parameter_count']):,}",
                f"{int(row['total_trainable_parameter_count']):,}",
                f"{100 * float(row['trainable_to_base_parameter_fraction']):.2f}%",
                f"{float(row['final_training_loss']):.5f}",
                f"{float(row['runtime_seconds']) / 60:.1f} min",
            )
            for row in training
        ),
    )
    embedding_table = _table(
        ("Training rank", "Embedding-only story NLL", "Embedding-only token NLL"),
        tuple(
            (
                str(row["rank"]),
                f"{float(row['story_mean_nll']):.6f}",
                f"{float(row['token_mean_nll']):.6f}",
            )
            for row in embedding_only
        ),
    )
    task_table = _table(
        ("Noun", "Condition", "Story NLL", "Token NLL", "Token accuracy"),
        tuple(
            (
                str(row["task_id"]),
                str(row["label"]),
                f"{float(row['story_mean_nll']):.5f}",
                f"{float(row['token_mean_nll']):.5f}",
                f"{100 * float(row['suffix_token_accuracy']):.2f}%",
            )
            for row in per_task
        ),
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds nouns-v2 LoRA plus tied embedding</title>
<style>
:root{{--ink:#17212b;--muted:#52606d;--paper:#fff;--panel:#f5f8fb;--line:#cbd5df;--green:#009e73}}
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--paper);line-height:1.55;margin:0}}
main{{max-width:1160px;margin:auto;padding:28px 24px 60px}} h1{{line-height:1.2}}
.lede{{font-size:1.08rem;background:#e9f7f2;border-left:5px solid var(--green);padding:14px 18px}}
.plot{{overflow-x:auto;border:1px solid var(--line);padding:8px;background:white}} .plot svg{{width:100%;height:auto;min-width:760px}}
details{{margin:18px 0;border:1px solid var(--line);border-radius:7px;background:var(--panel);padding:10px 14px}}
summary{{cursor:pointer;font-weight:700}} table{{border-collapse:collapse;width:100%;font-size:.91rem;margin:12px 0;background:white}}
th,td{{border:1px solid var(--line);padding:7px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#e8eef4}}
code{{overflow-wrap:anywhere}} .muted{{color:var(--muted)}}
</style></head><body><main>
<h1>TinyWorlds nouns-v2 joint-IID LoRA plus tied embedding</h1>
<p class="lede">{escape(_finding(aggregate))}</p>
<p>The experiment jointly trains the tied token lookup/output-classifier matrix with all six projection LoRA targets, using the exact parent joint-IID training and final suffix populations.</p>
<div class="plot">{svg}</div>{aggregate_table}
<p class="muted">Token accuracy is teacher-forced next-token accuracy, not routing accuracy.</p>
<details><summary>Method and trainable parameters</summary><p>LoRA uses LR 1e-3; the tied embedding uses LR 5e-5. Both AdamW groups use weight decay 0.01 after one combined global clip at 1.0. The original transformer, positions, norms, and biases remain frozen.</p>{training_table}<p>The embedding-only rows disable the jointly trained LoRA and are diagnostic rather than separately optimized controls.</p>{embedding_table}</details>
<details><summary>Paired uncertainty</summary><p>Seed-zero 10,000-sample paired bootstrap stratified by noun. Negative condition-minus-reference NLL favors the condition.</p>{interval_table}</details>
<details><summary>Per-noun results</summary>{task_table}</details>
<details><summary>Provenance and execution</summary><ul>
<li>Contract: <code>{inputs.contract_sha256}</code></li>
<li>Parent sweep: <code>{provenance['parent_rank_sweep_contract_sha256']}</code></li>
<li>Allocator peak: {int(allocator['peak_bytes_in_use']) / 1024**3:.2f} GiB / 12 GiB</li>
<li>End-to-end runtime: {float(execution['end_to_end_seconds']) / 60:.1f} minutes</li>
</ul><p>CSV exports preserve all aggregate, per-noun, uncertainty, training, diagnostic, and ledger records.</p></details>
</main></body></html>"""


def _finding(aggregate: Sequence[Mapping[str, object]]) -> str:
    values = {str(row["condition"]): row for row in aggregate}
    full = float(values["full_model"]["story_mean_nll"])
    clauses = []
    for rank in RANKS:
        standard = float(values[f"lora_rank_{rank}"]["story_mean_nll"])
        joint = float(values[f"lora_embedding_rank_{rank}"]["story_mean_nll"])
        gap = standard - full
        recovery = (standard - joint) / gap if gap > 0.0 else 0.0
        clauses.append(
            f"rank {rank}: {joint:.6f} story NLL, "
            f"{100 * recovery:.1f}% of its projection-only gap recovered"
        )
    return (
        "Jointly training the tied embedding gives "
        + "; ".join(clauses)
        + f". The full-model reference is {full:.6f}."
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("embedding-LoRA CSV export cannot be empty")
    fields = tuple(sorted({str(key) for row in rows for key in row}))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _accessible_svg(svg: str, title: str, description: str) -> str:
    start = svg.find("<svg ")
    if start < 0:
        raise ValueError("matplotlib output contains no SVG root")
    root_stop = svg.find(">", start)
    root = svg[start : root_stop + 1]
    accessible = root[:-1] + ' role="img" aria-labelledby="plot-title plot-desc">'
    return (
        svg[:start]
        + accessible
        + f'<title id="plot-title">{escape(title)}</title>'
        + f'<desc id="plot-desc">{escape(description)}</desc>'
        + svg[root_stop + 1 :]
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f'<th scope="col">{escape(value)}</th>' for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div style="overflow-x:auto"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _rank(value: object) -> str:
    return "—" if value is None else str(int(value))


def _records(value: object, label: str) -> tuple[dict[str, object], ...]:
    if type(value) not in (list, tuple) or any(type(row) is not dict for row in value):
        raise ValueError(f"embedding-LoRA {label} rows are malformed")
    return tuple(dict(row) for row in value)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"embedding-LoRA {label} record is malformed")
    return dict(value)


__all__ = ["publish_lora_embedding_report"]
