"""Standalone Markdown/HTML publication for the joint-IID LoRA rank sweep."""

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
from apm.continual.artifacts import (
    atomic_write,
    file_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
    JointIidRankSweepInputs,
    RANKS,
    REPORT_FORMAT,
)


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


def publish_joint_iid_rank_sweep_report(
    inputs: JointIidRankSweepInputs,
    analysis: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Publish deterministic data exports, plot, reports, and manifest."""
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
        "ledger-provenance.csv": _records(
            analysis["ledger_provenance"],
            "ledger provenance",
        ),
        "per-task.csv": _records(analysis["per_task"], "per-task"),
        "training.csv": _records(analysis["training"], "training"),
    }
    for name, rows in exports.items():
        atomic_write(output / name, _csv_bytes(rows))
    plot_path = _publish_plot(output, analysis)
    markdown_path = atomic_write(
        output / "report.md",
        _markdown_report(inputs, analysis).encode("utf-8"),
    )
    html_path = atomic_write(
        output / "report.html",
        _html_report(inputs, analysis, plot_path.read_text(encoding="utf-8")).encode(
            "utf-8"
        ),
    )
    artifact_names = (
        "aggregate.csv",
        "allocator.json",
        "analysis.json",
        "bootstrap.csv",
        "contract.json",
        "execution.json",
        "ledger-provenance.csv",
        "per-task.csv",
        "rank-sweep-nll.svg",
        "report.html",
        "report.md",
        "training.csv",
    )
    if any(not (output / name).is_file() for name in artifact_names):
        raise FileNotFoundError("rank-sweep publication bundle is incomplete")
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
            "svg.hashsalt": "tinyworlds-nouns-v2-joint-iid-lora-rank-sweep",
        }
    )
    aggregate = {
        str(row["condition"]): row
        for row in _records(analysis["aggregate"], "aggregate")
    }
    ranks = list(RANKS)
    positions = list(range(len(ranks)))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    specifications = (
        ("story_mean_nll", "Story-weighted suffix NLL"),
        ("token_mean_nll", "Token-weighted suffix NLL"),
    )
    for axis, (metric, title) in zip(axes, specifications, strict=True):
        values = [float(aggregate[f"rank_{rank}"][metric]) for rank in ranks]
        baseline = float(aggregate["full_model"][metric])
        axis.plot(
            positions,
            values,
            color="#0072B2",
            marker="o",
            markersize=7,
            linewidth=2.2,
            label="LoRA sweep",
        )
        axis.axhline(
            baseline,
            color="#D55E00",
            linestyle="--",
            linewidth=2.0,
            label="Joint-IID full model",
        )
        for position, value in zip(positions, values, strict=True):
            axis.annotate(
                f"{value:.4f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("LoRA rank (alpha = rank; scale = 1)")
        axis.set_ylabel("Negative log-likelihood (nats/token)")
        axis.set_xticks(positions, [str(rank) for rank in ranks])
        axis.grid(axis="y", alpha=0.28)
        axis.legend(fontsize=9)
    title = "Joint-IID LoRA rank sweep on the exact nouns-v2 final suffix evaluation"
    description = (
        "Two panels show story-weighted and token-weighted suffix negative "
        "log-likelihood for LoRA ranks 4, 8, 16, and 32. A dashed line marks "
        "the already published joint-IID full-model control."
    )
    stream = io.StringIO()
    figure.savefig(
        stream,
        format="svg",
        metadata={"Date": None, "Description": description, "Title": title},
    )
    plt.close(figure)
    return atomic_write(
        output / "rank-sweep-nll.svg",
        _accessible_svg(stream.getvalue(), title, description).encode("utf-8"),
    )


def _markdown_report(
    inputs: JointIidRankSweepInputs,
    analysis: Mapping[str, object],
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    training = _records(analysis["training"], "training")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    finding = _finding(aggregate)
    aggregate_rows = "\n".join(
        f"| {row['label']} | {_rank_cell(row['rank'])} | "
        f"{float(row['story_mean_nll']):.6f} | {float(row['token_mean_nll']):.6f} | "
        f"{100 * float(row['suffix_token_accuracy']):.3f}% | {int(row['story_count']):,} | "
        f"{int(row['token_count']):,} |"
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
        f"| {int(row['rank'])} | {float(row['alpha']):.0f} | "
        f"{int(row['adapter_parameter_count']):,} | "
        f"{100 * float(row['adapter_to_base_parameter_fraction']):.3f}% | "
        f"{int(row['optimizer_updates']):,} | {float(row['final_training_loss']):.5f} | "
        f"{float(row['runtime_seconds']) / 60:.1f} min | "
        f"{'reused' if bool(row['reused_canonical_artifact']) else 'new'} |"
        for row in training
    )
    per_task_rows = "\n".join(
        f"| {row['task_id']} | {row['label']} | {float(row['story_mean_nll']):.5f} | "
        f"{float(row['token_mean_nll']):.5f} | "
        f"{100 * float(row['suffix_token_accuracy']):.2f}% |"
        for row in _records(analysis["per_task"], "per-task")
    )
    provenance = _object(analysis["provenance"], "provenance")
    comparability = _object(analysis["comparability"], "comparability")
    allocator = _object(analysis["allocator"], "allocator")
    execution = _object(analysis["execution"], "execution")
    return f"""# TinyWorlds nouns-v2 joint-IID LoRA rank sweep

{finding}

This addendum evaluates LoRA ranks 4, 8, 16, and 32 on the exact 4,440-story
final suffix condition used by the [temporal-consolidation report](../report.md).
The rank-8 rows and joint-IID full-model rows are the original authenticated
results, not recomputed approximations.

![Joint-IID suffix NLL by LoRA rank](rank-sweep-nll.svg)

| Condition | Rank | Story NLL | Token NLL | Suffix token accuracy | Stories | Suffix tokens |
|---|---:|---:|---:|---:|---:|---:|
{aggregate_rows}

The story-weighted NLL gives every story equal weight, matching the report's
primary final-quality figure. Token-weighted NLL sums all suffix losses before
dividing by all suffix tokens. Both are teacher-forced next-token cross-entropy
on the evaluator-only story suffix; “token accuracy” is included only as the
fraction of those suffix targets whose most likely token was correct.

<details>
<summary>Method and direct-comparability controls</summary>

Every new adapter sees the same 98,304 selected training stories for four
epochs and the same 15,024 minibatches as canonical rank 8. The epoch-order and
random namespaces are both the canonical rank-8 job identity. AdamW settings
remain batch 32, LR `1e-3`, weight decay `0.01`, gradient clipping `1.0`, and
context length 256. Alpha equals rank, so all four conditions have LoRA scale
`alpha / rank = 1`; only low-rank capacity changes.

Rank 8 is strict-loaded from the original adapter and its original 4,440-row
ledger. The full-model line is also the original joint-IID control (trained
under its already published full-model schedule), so it is a quality reference,
not a parameter-matched LoRA condition. All ledgers share exact story order,
suffix token masks, and {int(aggregate[0]['token_count']):,} suffix targets.
The maximum base-path NLL drift induced by compiling different rank shapes is
`{float(comparability['base_path_max_abs_story_nll_drift']):.3g}`.

| Rank | Alpha | Trainable parameters | Fraction of base | Updates | Final train loss | Runtime | Source |
|---:|---:|---:|---:|---:|---:|---:|---|
{training_rows}

</details>

<details>
<summary>Paired uncertainty intervals</summary>

Intervals use the preregistered deterministic seed-zero, 10,000-sample paired
bootstrap, stratified by noun. Differences are condition minus reference, so a
negative NLL difference favors the swept rank.

| Condition | Reference | Metric | Difference | 95% interval |
|---|---|---|---:|---:|
{interval_rows}

</details>

<details>
<summary>Per-noun results</summary>

| Noun | Condition | Story NLL | Token NLL | Token accuracy |
|---|---|---:|---:|---:|
{per_task_rows}

</details>

<details>
<summary>Provenance and execution</summary>

- Sweep contract: `{inputs.contract_sha256}`
- Parent temporal contract: `{provenance['parent_contract_sha256']}`
- Parent publication manifest: `{provenance['parent_manifest_sha256']}`
- Canonical rank-8 job: `{provenance['canonical_rank8_job_sha256']}`
- Canonical full-model job: `{provenance['canonical_full_model_job_sha256']}`
- Exact batch/random namespace: `{comparability['batch_namespace_sha256']}`
- Allocator peak: {int(allocator['peak_bytes_in_use']) / 1024**3:.2f} GiB of a 12 GiB gate
- End-to-end runtime: {float(execution['end_to_end_seconds']) / 60:.1f} minutes

Raw aggregates, per-task rows, paired intervals, training metadata, and ledger
hashes are exported as CSV beside this report. The HTML report is standalone
and embeds the same accessible SVG.

</details>
"""


def _html_report(
    inputs: JointIidRankSweepInputs,
    analysis: Mapping[str, object],
    svg: str,
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    training = _records(analysis["training"], "training")
    per_task = _records(analysis["per_task"], "per-task")
    comparability = _object(analysis["comparability"], "comparability")
    provenance = _object(analysis["provenance"], "provenance")
    allocator = _object(analysis["allocator"], "allocator")
    execution = _object(analysis["execution"], "execution")
    aggregate_table = _html_table(
        ("Condition", "Rank", "Story NLL", "Token NLL", "Token accuracy"),
        tuple(
            (
                str(row["label"]),
                _rank_cell(row["rank"]),
                f"{float(row['story_mean_nll']):.6f}",
                f"{float(row['token_mean_nll']):.6f}",
                f"{100 * float(row['suffix_token_accuracy']):.3f}%",
            )
            for row in aggregate
        ),
    )
    interval_table = _html_table(
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
    training_table = _html_table(
        ("Rank", "Alpha", "Parameters", "Base fraction", "Updates", "Runtime", "Source"),
        tuple(
            (
                str(row["rank"]),
                f"{float(row['alpha']):.0f}",
                f"{int(row['adapter_parameter_count']):,}",
                f"{100 * float(row['adapter_to_base_parameter_fraction']):.3f}%",
                f"{int(row['optimizer_updates']):,}",
                f"{float(row['runtime_seconds']) / 60:.1f} min",
                "reused" if bool(row["reused_canonical_artifact"]) else "new",
            )
            for row in training
        ),
    )
    per_task_table = _html_table(
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
<title>TinyWorlds nouns-v2 joint-IID LoRA rank sweep</title>
<style>
:root{{--ink:#17212b;--muted:#52606d;--paper:#fff;--panel:#f5f8fb;--line:#cbd5df;--blue:#0072b2}}
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--paper);line-height:1.55;margin:0}}
main{{max-width:1120px;margin:auto;padding:28px 24px 60px}} h1,h2{{line-height:1.2}}
.lede{{font-size:1.08rem;background:#eaf4fb;border-left:5px solid var(--blue);padding:14px 18px}}
.plot{{overflow-x:auto;border:1px solid var(--line);padding:8px;background:white}} .plot svg{{width:100%;height:auto;min-width:760px}}
details{{margin:18px 0;border:1px solid var(--line);border-radius:7px;background:var(--panel);padding:10px 14px}}
summary{{cursor:pointer;font-weight:700}} table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:12px 0;background:white}}
th,td{{border:1px solid var(--line);padding:7px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#e8eef4}}
code{{overflow-wrap:anywhere}} .muted{{color:var(--muted)}}
</style></head><body><main>
<h1>TinyWorlds nouns-v2 joint-IID LoRA rank sweep</h1>
<p class="lede">{escape(_finding(aggregate))}</p>
<p>This addendum uses the exact 4,440-story final suffix condition from the temporal study. Rank 8 and the joint-IID full-model control are authenticated original rows.</p>
<div class="plot">{svg}</div>
{aggregate_table}
<p class="muted">Story NLL weights stories equally; token NLL weights suffix tokens equally. Token accuracy is teacher-forced top-one next-token accuracy, not routing accuracy.</p>
<details><summary>Method and direct-comparability controls</summary>
<p>Every new adapter uses the same 98,304 stories, four epochs, 15,024 minibatches, batch order, and random namespace as canonical rank 8. Alpha equals rank, so scale is one. The full-model control retains its original full-model optimizer schedule and is a quality reference rather than a parameter-matched condition.</p>
<p>All ledgers have exact story order and suffix masks. Maximum base-path NLL drift across rank-shaped compilations is <code>{float(comparability['base_path_max_abs_story_nll_drift']):.3g}</code>.</p>
{training_table}</details>
<details><summary>Paired uncertainty intervals</summary><p>Seed-zero 10,000-sample paired bootstrap, stratified by noun. Negative condition-minus-reference NLL favors the swept rank.</p>{interval_table}</details>
<details><summary>Per-noun results</summary>{per_task_table}</details>
<details><summary>Provenance and execution</summary><ul>
<li>Sweep contract: <code>{inputs.contract_sha256}</code></li>
<li>Parent contract: <code>{provenance['parent_contract_sha256']}</code></li>
<li>Canonical rank-8 job: <code>{provenance['canonical_rank8_job_sha256']}</code></li>
<li>Canonical full-model job: <code>{provenance['canonical_full_model_job_sha256']}</code></li>
<li>Allocator peak: {int(allocator['peak_bytes_in_use']) / 1024**3:.2f} GiB / 12 GiB</li>
<li>End-to-end runtime: {float(execution['end_to_end_seconds']) / 60:.1f} minutes</li>
</ul><p>CSV exports beside the reports preserve aggregate, per-task, bootstrap, training, and ledger provenance records.</p></details>
</main></body></html>"""


def _finding(aggregate: Sequence[Mapping[str, object]]) -> str:
    by_condition = {str(row["condition"]): row for row in aggregate}
    rank8 = float(by_condition["rank_8"]["story_mean_nll"])
    full_model = float(by_condition["full_model"]["story_mean_nll"])
    best_rank = min(RANKS, key=lambda rank: float(by_condition[f"rank_{rank}"]["story_mean_nll"]))
    best = float(by_condition[f"rank_{best_rank}"]["story_mean_nll"])
    gap = rank8 - full_model
    recovered = (rank8 - best) / gap if gap > 0.0 else 0.0
    return (
        f"Rank {best_rank} is the best LoRA condition at {best:.6f} story-weighted "
        f"suffix NLL. Relative to canonical rank 8 ({rank8:.6f}), it "
        f"{'recovers' if recovered >= 0 else 'widens'} {abs(100 * recovered):.1f}% of the "
        f"gap to the joint-IID full model ({full_model:.6f})."
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("rank-sweep CSV export cannot be empty")
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
    accessible_root = root[:-1] + ' role="img" aria-labelledby="plot-title plot-desc">'
    return (
        svg[:start]
        + accessible_root
        + f'<title id="plot-title">{escape(title)}</title>'
        + f'<desc id="plot-desc">{escape(description)}</desc>'
        + svg[root_stop + 1 :]
    )


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    head = "".join(f"<th scope=\"col\">{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div style=\"overflow-x:auto\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _rank_cell(value: object) -> str:
    return "—" if value is None else str(int(value))


def _records(value: object, label: str) -> tuple[dict[str, object], ...]:
    if type(value) not in (list, tuple) or any(type(row) is not dict for row in value):
        raise ValueError(f"rank-sweep {label} rows are malformed")
    return tuple(dict(row) for row in value)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"rank-sweep {label} record is malformed")
    return dict(value)


__all__ = ["publish_joint_iid_rank_sweep_report"]
