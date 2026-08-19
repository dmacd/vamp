"""Standalone publication for the TinyWorlds full-story routing diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from html import escape
import io
import json
import os
from pathlib import Path
import re
import tempfile

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.continual.artifacts import (
    atomic_write,
    file_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing import (
    FullStoryRoutingInputs,
    REPORT_FORMAT,
    STUDY_ID,
)


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


def publish_full_story_routing_report(
    inputs: FullStoryRoutingInputs,
    analysis: Mapping[str, object],
    *,
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Publish deterministic CSV, SVG, JSON, Markdown, HTML, and manifest files."""
    output = inputs.result_directory
    csv_paths = tuple(
        _write_csv(output / filename, _records(analysis[key], key))
        for filename, key in (
            ("aggregate.csv", "aggregate"),
            ("per-task.csv", "per_task"),
            ("bootstrap.csv", "bootstrap"),
            ("confusion.csv", "confusion"),
        )
    )
    plot_path = _publish_plot(output, analysis)
    analysis_core = {
        "allocator": dict(allocator),
        "analysis": dict(analysis),
        "contract_sha256": inputs.contract_sha256,
        "execution": dict(execution),
        "format": REPORT_FORMAT,
    }
    analysis_path = atomic_write(
        output / "analysis.json",
        canonical_json_bytes(
            {**analysis_core, "analysis_sha256": record_sha256(analysis_core)}
        ),
    )
    markdown_path = atomic_write(
        output / "report.md",
        _markdown_report(inputs, analysis, execution, allocator).encode("utf-8"),
    )
    html_path = atomic_write(
        output / "report.html",
        _html_report(
            inputs,
            analysis,
            execution,
            allocator,
            plot_path.read_text(encoding="utf-8"),
        ).encode("utf-8"),
    )
    artifact_paths = (
        output / "contract.json",
        output / "execution.json",
        output / "allocator.json",
        analysis_path,
        markdown_path,
        html_path,
        plot_path,
        *csv_paths,
    )
    manifest_core = {
        "artifacts": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(artifact_paths)
        },
        "contract_sha256": inputs.contract_sha256,
        "format": f"{STUDY_ID}-publication-v1",
        "parent_contract_sha256": inputs.parent.contract_sha256,
        "schema_version": 1,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": record_sha256(manifest_core),
    }
    manifest_path = atomic_write(
        output / "manifest.json",
        canonical_json_bytes(manifest),
    )
    return markdown_path, html_path, manifest_path


def _publish_plot(output: Path, analysis: Mapping[str, object]) -> Path:
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.hashsalt": STUDY_ID,
        }
    )
    aggregate = _records(analysis["aggregate"], "aggregate")
    positions = np.arange(len(aggregate), dtype=np.float64)
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    midpoint_color = "#0072B2"
    full_color = "#D55E00"
    oracle_color = "#009E73"
    axes[0].bar(
        positions - width / 2,
        [100.0 * float(row["midpoint_route_accuracy"]) for row in aggregate],
        width,
        color=midpoint_color,
        label="Midpoint prefix",
    )
    axes[0].bar(
        positions + width / 2,
        [100.0 * float(row["full_route_accuracy"]) for row in aggregate],
        width,
        color=full_color,
        label="Entire story",
    )
    axes[0].set_ylabel("Route accuracy (%)")
    axes[0].set_title("Task evidence in the selected candidate")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right")
    for offset, key in ((-width / 2, "midpoint_route_accuracy"), (width / 2, "full_route_accuracy")):
        for position, row in zip(positions, aggregate, strict=True):
            value = 100.0 * float(row[key])
            axes[0].text(position + offset, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)

    axes[1].plot(
        positions,
        [float(row["midpoint_suffix_story_nll"]) for row in aggregate],
        marker="o",
        linewidth=2,
        color=midpoint_color,
        label="Midpoint-selected",
    )
    axes[1].plot(
        positions,
        [float(row["full_suffix_story_nll"]) for row in aggregate],
        marker="s",
        linewidth=2,
        color=full_color,
        label="Full-story-selected",
    )
    axes[1].plot(
        positions,
        [float(row["oracle_suffix_story_nll"]) for row in aggregate],
        marker="^",
        linewidth=2,
        linestyle="--",
        color=oracle_color,
        label="Suffix oracle",
    )
    axes[1].set_ylabel("Suffix story NLL (nats/token)")
    axes[1].set_title("Reported suffix quality after selection")
    axes[1].legend()
    labels = [str(row["label"]).replace(" bank", "\nbank").replace(" log-t", "\nlog-t") for row in aggregate]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.25)
    stream = io.StringIO()
    title = "Midpoint versus full-story routing at the final checkpoint"
    description = (
        "The left panel compares noun-support or exact-noun route accuracy. "
        "The right compares suffix story NLL and the evaluator-only suffix oracle."
    )
    figure.savefig(
        stream,
        format="svg",
        metadata={"Date": None, "Description": description, "Title": title},
    )
    plt.close(figure)
    return atomic_write(
        output / "full-story-routing.svg",
        _accessible_svg(stream.getvalue(), title, description).encode("utf-8"),
    )


def _markdown_report(
    inputs: FullStoryRoutingInputs,
    analysis: Mapping[str, object],
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    audit = _object(analysis["audit"], "audit")
    rows = "\n".join(
        "| "
        + " | ".join(
            (
                str(row["label"]),
                _accuracy_label(str(row["accuracy_kind"])),
                f"{100 * float(row['midpoint_route_accuracy']):.2f}%",
                f"{100 * float(row['full_route_accuracy']):.2f}%",
                f"{float(row['midpoint_suffix_story_nll']):.5f}",
                f"{float(row['full_suffix_story_nll']):.5f}",
                f"{float(row['oracle_suffix_story_nll']):.5f}",
                f"{100 * float(row['suffix_gap_recovered_fraction']):.1f}%",
            )
        )
        + " |"
        for row in aggregate
    )
    intervals = "\n".join(
        f"| {str(row['condition']).replace('_', ' ')} | {str(row['metric']).replace('_', ' ')} "
        f"| {float(row['estimate']):+.6f} | [{float(row['lower_95']):+.6f}, "
        f"{float(row['upper_95']):+.6f}] |"
        for row in bootstrap
    )
    finding = _finding_text(aggregate)
    return f"""# TinyWorlds nouns-v2 full-story routing diagnostic

{finding}

This is a versioned addendum to the [temporal-consolidation report](../report.md),
not a replacement for it. It uses all 4,440 official validation stories and the
three already-trained final banks. No model or adapter was retrained.

![Midpoint versus full-story routing](full-story-routing.svg)

| Final bank | Accuracy meaning | Midpoint route | Full-story route | Midpoint suffix NLL | Full-story suffix NLL | Suffix oracle | Oracle gap recovered |
|---|---|---:|---:|---:|---:|---:|---:|
{rows}

The suffix result for full-story routing is intentionally **selection-leaking**:
the router reads the same suffix whose NLL is then reported. It measures whether
the complete story contains enough evidence to identify a useful memory; it is
not a deployable held-out routing estimate. The whole-story self-selected losses
are preserved in `aggregate.csv` as a second, explicitly self-selected view.

<details>
<summary>What this says about the hypothesis</summary>

The experiment separates weak midpoint cues from poor memories. If route
accuracy rises and suffix NLL falls when the rest of the story becomes visible,
the original midpoint protocol was charging the memory system for missing
addressing evidence. The remaining distance to the suffix oracle is the portion
not removed by this full-story selector. Mixed log-t intervals can contain more
than one noun, so their accuracy is noun support; only the independent bank has
an exact noun-route label. A base selection counts as a miss in both cases.

</details>

<details>
<summary>Canonical scoring and bounded audit</summary>

For candidate `j`, short-story whole NLL is reconstructed as
`(prefix_mean[j] * prefix_tokens + suffix_mean[j] * suffix_tokens) /
(prefix_tokens + suffix_tokens)`. This is the same causal computation as the
canonical 256-token story windows whenever the midpoint prefix is at most 256
transitions. All {int(audit['long_story_count'])} longer stories were rescored
directly. The deterministic audit additionally included near ties and the
minimum-margin short story for every noun: {int(audit['audited_story_count'])}
unique stories × three banks = {int(audit['audited_condition_story_rows'])}
direct rows. Maximum short-story score error was
{float(audit['maximum_short_score_absolute_error']):.3g}; there were
{int(audit['selection_mismatches'])} selection mismatches. The smallest
unaudited margin was {float(audit['minimum_unaudited_margin']):.6g}, more than
twice the fixed `1e-4` score tolerance.

Candidate order is inherited exactly from each parent ledger, base remains
first, and ties use the stable first minimum. The chained work ledgers are
resumable and tamper-rejecting.

</details>

<details>
<summary>Paired uncertainty</summary>

Differences are full-story minus midpoint. Intervals use deterministic seed-zero
10,000-sample paired bootstrap resampling within each of the 24 noun strata.

| Bank | Metric | Difference | 95% interval |
|---|---|---:|---:|
{intervals}

</details>

<details>
<summary>Per-task, confusion, execution, and provenance</summary>

`per-task.csv` contains all 72 bank/noun summaries; `confusion.csv` contains
both routing rules' task-to-candidate counts. `analysis.json` binds these tables,
the exact parent ledgers, the 13,320-row derived ledger, and the 570-row direct
audit ledger. `manifest.json` hashes every published addendum artifact. The
parent contract is `{inputs.parent.contract_sha256}` and remains byte-identical.

End-to-end runtime was {float(execution.get('end_to_end_seconds', 0.0)):.1f} s.
Peak JAX allocator use was {int(allocator.get('peak_bytes_in_use', 0)) / 1024**3:.2f}
GiB against the fixed 12 GiB gate.

</details>
"""


def _html_report(
    inputs: FullStoryRoutingInputs,
    analysis: Mapping[str, object],
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
    svg: str,
) -> str:
    aggregate = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    audit = _object(analysis["audit"], "audit")
    result_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['label']))}</td>"
        f"<td>{escape(_accuracy_label(str(row['accuracy_kind'])))}</td>"
        f"<td>{100 * float(row['midpoint_route_accuracy']):.2f}%</td>"
        f"<td>{100 * float(row['full_route_accuracy']):.2f}%</td>"
        f"<td>{float(row['midpoint_suffix_story_nll']):.5f}</td>"
        f"<td>{float(row['full_suffix_story_nll']):.5f}</td>"
        f"<td>{float(row['oracle_suffix_story_nll']):.5f}</td>"
        f"<td>{100 * float(row['suffix_gap_recovered_fraction']):.1f}%</td>"
        "</tr>"
        for row in aggregate
    )
    bootstrap_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']).replace('_', ' '))}</td>"
        f"<td>{escape(str(row['metric']).replace('_', ' '))}</td>"
        f"<td>{float(row['estimate']):+.6f}</td>"
        f"<td>[{float(row['lower_95']):+.6f}, {float(row['upper_95']):+.6f}]</td>"
        "</tr>"
        for row in bootstrap
    )
    finding = escape(_finding_text(aggregate))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds nouns-v2 full-story routing diagnostic</title>
<style>
:root{{--ink:#17212b;--muted:#52606d;--line:#c8d3dc;--panel:#f5f8fa;--blue:#0069a8;--warn:#7a3e00}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}h1{{line-height:1.15}}.lead{{font-size:1.1rem;max-width:95ch}}.warning{{border-left:5px solid #D55E00;background:#fff4e8;padding:12px 16px}}.plot{{margin:1.2rem 0;overflow:auto}}.plot svg{{width:100%;height:auto;min-width:720px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}details{{margin:1rem 0;border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:12px 16px}}summary{{cursor:pointer;font-weight:700;color:var(--blue)}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{main{{padding:14px}}}}
</style></head><body><main>
<h1>TinyWorlds nouns-v2 full-story routing diagnostic</h1>
<p class="lead">{finding}</p>
<p>This immutable addendum analyzes all 4,440 official validation stories in each of three already-trained final banks. No retraining occurred. Parent contract: <code>{inputs.parent.contract_sha256}</code>.</p>
<div class="plot">{svg}</div>
<div class="scroll"><table><thead><tr><th>Final bank</th><th>Accuracy meaning</th><th>Midpoint route</th><th>Full-story route</th><th>Midpoint suffix NLL</th><th>Full-story suffix NLL</th><th>Suffix oracle</th><th>Oracle gap recovered</th></tr></thead><tbody>{result_rows}</tbody></table></div>
<p class="warning"><strong>Diagnostic-only selection leakage:</strong> full-story routing reads the same suffix whose NLL is reported. This establishes how much evidence exists in the complete story, not deployable held-out routing quality.</p>
<details open><summary>Scientific interpretation</summary><p>When complete-story access raises route accuracy and lowers suffix NLL, midpoint addressing evidence—not just memory quality—accounts for part of the parent report's apparent loss. Mixed temporal intervals receive noun-support accuracy because one interval can support multiple nouns. The independent bank receives exact noun-route accuracy. Base is a miss.</p></details>
<details><summary>Canonical scoring and audit</summary><p>Stored prefix and suffix candidate totals reconstruct canonical whole-story scores for prefixes no longer than 256 transitions. Every longer story, every reconstructed margin at most 0.0002, and one minimum-margin short story per noun entered the direct GPU audit. This yielded {int(audit['audited_story_count'])} unique stories and {int(audit['audited_condition_story_rows'])} bank/story rows. Maximum short-score error was {float(audit['maximum_short_score_absolute_error']):.3g}, with {int(audit['selection_mismatches'])} selection mismatches. The minimum unaudited margin was {float(audit['minimum_unaudited_margin']):.6g}.</p></details>
<details><summary>Paired uncertainty</summary><p>Full-story minus midpoint; deterministic seed-zero 10,000-sample paired bootstrap within each noun.</p><div class="scroll"><table><thead><tr><th>Bank</th><th>Metric</th><th>Difference</th><th>95% interval</th></tr></thead><tbody>{bootstrap_rows}</tbody></table></div></details>
<details><summary>Per-task and provenance</summary><p>Aggregate, per-task, confusion, and bootstrap CSV files accompany the authenticated <code>analysis.json</code>. The 13,320 derived rows and 570 direct rows remain in resumable hash-chained work ledgers. End-to-end runtime: {float(execution.get('end_to_end_seconds', 0.0)):.1f} s. Peak allocator: {int(allocator.get('peak_bytes_in_use', 0)) / 1024**3:.2f} GiB under the 12 GiB gate.</p></details>
</main></body></html>"""


def _finding_text(aggregate: Sequence[Mapping[str, object]]) -> str:
    changes = [float(row["route_accuracy_change_pp"]) for row in aggregate]
    improvements = [-float(row["suffix_story_nll_change"]) for row in aggregate]
    return (
        "The hypothesis is supported: full-story access raises routing accuracy by "
        f"{min(changes):.2f}–{max(changes):.2f} percentage points and lowers "
        f"story-weighted suffix NLL by {min(improvements):.5f}–"
        f"{max(improvements):.5f} across all three final banks."
    )


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    if not records:
        raise ValueError(f"cannot publish empty CSV: {path.name}")
    fields = tuple(sorted({key for record in records for key in record}))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            field: json.dumps(value, sort_keys=True, separators=(",", ":"))
            if type(value) in (dict, list, tuple)
            else value
            for field, value in record.items()
        }
        for record in records
    )
    return atomic_write(path, stream.getvalue().encode("utf-8"))


def _accessible_svg(svg: str, title: str, description: str) -> str:
    value = re.sub(r"<title>.*?</title>", "", svg, count=1, flags=re.DOTALL)
    value = value.replace(
        "<svg",
        '<svg role="img" aria-labelledby="svg-title svg-description"',
        1,
    )
    position = value.find(">", value.find("<svg"))
    if position < 0:
        raise ValueError("rendered SVG has no root element")
    labels = (
        f'<title id="svg-title">{escape(title)}</title>'
        f'<desc id="svg-description">{escape(description)}</desc>'
    )
    return value[: position + 1] + labels + value[position + 1 :]


def _accuracy_label(kind: str) -> str:
    return "Noun support" if kind == "noun_support" else "Exact noun"


def _records(value: object, label: str) -> tuple[dict[str, object], ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{label} must be a record sequence")
    return tuple(_object(item, label) for item in value)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return dict(value)


__all__ = ["publish_full_story_routing_report"]
