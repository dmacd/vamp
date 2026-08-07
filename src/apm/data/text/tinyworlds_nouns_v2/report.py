"""Artifact-only Markdown and folding HTML reports for disjoint nouns-v2."""

from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.report import (
    _judge_summary,
    _publish_graph_artifacts,
    _representative_examples,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASE_UNIVERSE_STORY_COUNT,
    CONDITIONS,
    EXCLUDED_TRAIN_STORY_COUNT,
    EXCLUDED_VALIDATION_STORY_COUNT,
    HALF_STORY_FORMAT,
    JUDGE_FORMAT,
    PURE_TASK_TRAIN_STORY_COUNT,
    PURE_TASK_VALIDATION_STORY_COUNT,
    REPORT_FORMAT,
    TASK_IDS,
    TRAIN_UNIQUE_STORY_COUNT,
    WHOLE_STORY_FORMAT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise import summarize_stagewise_ledger


def publish_nouns_v2_report(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    whole_story_path: str | Path,
    generation_path: str | Path,
    stagewise_path: str | Path,
    result_root: str | Path,
    *,
    judge_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Strict-load completed ledgers and publish both standalone reports."""
    whole_rows = _jsonl(Path(whole_story_path))
    generation_rows = _jsonl(Path(generation_path))
    judge_rows = (
        _jsonl(Path(judge_path))
        if judge_path is not None and Path(judge_path).is_file()
        else ()
    )
    if not adaptations:
        raise ValueError("nouns-v2 reporting requires every VAMP stage")
    adaptation = adaptations[-1]
    _validate_coverage(partition, whole_rows, generation_rows, judge_rows)
    continual_learning = summarize_stagewise_ledger(
        stagewise_path,
        partition,
        adaptations,
    )
    _validate_final_stage_suffix_parity(stagewise_path, generation_rows)
    data = build_report_data(
        partition,
        preset,
        adaptation,
        whole_rows,
        generation_rows,
        judge_rows,
        continual_learning,
    )
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    _publish_graph_artifacts(root, adaptation)
    _atomic_write(
        root / "vamp-graph.svg",
        render_vamp_graph_svg(data["graph"]).encode("utf-8"),
    )
    _publish_confusion_csv(root, data)
    _publish_stagewise_csvs(root, data)
    for name in ("base-selection.csv", "task-counts.csv"):
        destination = root / name
        source = partition.root / name
        if destination.is_file():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"published nouns-v2 {name} changed")
        else:
            shutil.copy2(source, destination)
    markdown_path = root / "report.md"
    html_path = root / "report.html"
    _atomic_write(markdown_path, render_report_markdown(data).encode("utf-8"))
    _atomic_write(html_path, render_report_html(data).encode("utf-8"))
    return markdown_path, html_path


def build_report_data(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    adaptation: LanguageAdaptationArtifact,
    whole_rows: tuple[dict[str, object], ...],
    generation_rows: tuple[dict[str, object], ...],
    judge_rows: tuple[dict[str, object], ...],
    continual_learning: dict[str, object],
) -> dict[str, object]:
    """Build the complete JSON-compatible v2 report view."""
    by_task_condition: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in whole_rows:
        by_task_condition[(str(row["task_noun"]), str(row["condition"]))].append(row)
    task_metrics = tuple(
        _task_metrics(
            task.task_id,
            task.train_story_count,
            task.validation_story_count,
            {
                condition: by_task_condition[(task.task_id, condition)]
                for condition in CONDITIONS
            },
        )
        for task in partition.tasks
    )
    overall = {
        condition: _nll_summary(
            tuple(row for row in whole_rows if row["condition"] == condition)
        )
        for condition in CONDITIONS
    }
    suffix = {
        condition: _suffix_summary(generation_rows, condition)
        for condition in CONDITIONS
    }
    confusion_counts: CounterKey = defaultdict(int)
    for row in whole_rows:
        condition = str(row["condition"])
        if condition in CONDITIONS[2:]:
            confusion_counts[
                (condition, str(row["task_noun"]), str(row["selected_node"]))
            ] += 1
    confusion = {
        condition: {
            task_id: {
                node_id: confusion_counts[(condition, task_id, node_id)]
                for node_id in ("root", *TASK_IDS)
            }
            for task_id in TASK_IDS
        }
        for condition in CONDITIONS[2:]
    }
    graph = tuple(
        {
            "depth": node.depth,
            "node": str(node.node_id),
            "parent": None if node.parent_id is None else str(node.parent_id),
            "stage": node.train_stage,
        }
        for node in adaptation.vamp_graph.nodes
    )
    core = {
        "base": {
            "internal_validation_story_count": partition.base_validation_story_count,
            "optimizer_share": partition.base_train_story_count
            / TRAIN_UNIQUE_STORY_COUNT,
            "optimizer_train_story_count": partition.base_train_story_count,
            "universe_share": BASE_UNIVERSE_STORY_COUNT
            / TRAIN_UNIQUE_STORY_COUNT,
            "universe_story_count": BASE_UNIVERSE_STORY_COUNT,
        },
        "config_sha256": preset.config_sha256,
        "construction": {
            "excluded_train_story_count": EXCLUDED_TRAIN_STORY_COUNT,
            "excluded_validation_story_count": EXCLUDED_VALIDATION_STORY_COUNT,
            "pure_task_train_story_count": PURE_TASK_TRAIN_STORY_COUNT,
            "pure_validation_pair_count": PURE_TASK_VALIDATION_STORY_COUNT,
        },
        "continual_learning": continual_learning,
        "examples": list(_representative_examples(generation_rows)),
        "format": REPORT_FORMAT,
        "graph": list(graph),
        "judge": _judge_summary(judge_rows),
        "overall_conditions": overall,
        "partition_sha256": partition.partition_sha256,
        "routing_confusion": confusion,
        "suffix_conditions": suffix,
        "task_metrics": list(task_metrics),
    }
    return {**core, "report_sha256": record_sha256(core)}


def render_report_markdown(data: dict[str, object]) -> str:
    """Render the v2-only report in plain-language Markdown."""
    base = _object(data["base"], "report base")
    construction = _object(data["construction"], "report construction")
    continual = _object(data["continual_learning"], "continual learning")
    overall = _object(data["overall_conditions"], "overall conditions")
    suffix = _object(data["suffix_conditions"], "suffix conditions")
    task_metrics = tuple(
        _object(raw, "task metric") for raw in _list(data["task_metrics"], "tasks")
    )
    graph = tuple(
        _object(raw, "graph row") for raw in _list(data["graph"], "graph")
    )
    lines = [
        "# TinyWorlds nouns-v2 disjoint benchmark",
        "",
        "This report covers nouns-v2 only. Base and task training stories cannot "
        "overlap: zero selected noun families means base, exactly one means that "
        "single task, and two or more means permanent exclusion from every update.",
        "",
        "## Dataset construction",
        "",
        f"The clean base universe contains {int(base['universe_story_count']):,} "
        f"stories ({float(base['universe_share']):.2%} of original training). "
        f"After the deterministic 2% internal holdout, "
        f"{int(base['optimizer_train_story_count']):,} stories are optimizer-visible "
        f"({float(base['optimizer_share']):.2%} of the original archive).",
        "",
        f"The 24 tasks contain {int(construction['pure_task_train_story_count']):,} "
        f"pure training stories and {int(construction['pure_validation_pair_count']):,} "
        f"official-validation pairs. The audit permanently excludes "
        f"{int(construction['excluded_train_story_count']):,} training and "
        f"{int(construction['excluded_validation_story_count']):,} validation stories "
        "that mention multiple selected task families.",
        "",
        "## Learned VAMP graph",
        "",
        "![VAMP node dependency graph](vamp-graph.svg)",
        "",
        *(
            f"- `{row['node']}` attached to "
            f"`{row['parent'] if row['parent'] is not None else 'none (root)'}` "
            f"at depth {row['depth']}."
            for row in graph
        ),
        "",
        "## Stagewise continual-learning audit",
        "",
        f"The audit contains {int(continual['row_count']):,} midpoint-prefix "
        "task/story/stage cases. Every task is measured when introduced and after "
        "every later stage. The stored oracle follows that task's immutable VAMP "
        "node; exhaustive, Hopfield, and both EBT variants are task-free routers "
        "over the graph available at that stage.",
        "",
        f"The largest absolute stored-oracle NLL drift is "
        f"{float(continual['oracle_max_absolute_drift']):.6g}. Forgetting below is "
        "final task NLL minus its best earlier NLL (higher is worse); backward "
        "transfer is introduction NLL minus final NLL (higher is better).",
        "",
        "| condition | final story NLL | final token NLL | final route accuracy | mean forgetting | max forgetting | backward transfer | route accuracy change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            _continual_markdown_row(condition, continual)
            for condition in CONDITIONS
        ),
        "",
        "| stage | new task | retained stories | exhaustive | Hopfield | EBT uniform | EBT Hopfield | oracle NLL |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *(
            _stagewise_markdown_row(_object(raw, "stage summary"))
            for raw in _list(continual["stages"], "stage summaries")
        ),
        "",
        "Detailed task-level introduction, best, and final measurements are in "
        "`stagewise-task-metrics.csv`; the complete stage curves are in "
        "`stagewise-summary.csv`.",
        "",
        "## Whole-story NLL and routing",
        "",
        "| condition | story-weighted NLL | token-weighted NLL | perplexity | route accuracy | oracle regret |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            _overall_markdown_row(condition, _object(overall[condition], condition))
            for condition in CONDITIONS
        ),
        "",
        "| task | training stories | validation | base NLL | oracle NLL | acquisition | exhaustive | Hopfield | EBT uniform | EBT Hopfield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(_task_markdown_row(row) for row in task_metrics),
        "",
        "## Midpoint-only routing and true-suffix NLL",
        "",
        "The router saw only the exact first token half. The saved second half was "
        "used afterward for NLL and as the reference continuation.",
        "",
        "| condition | suffix story NLL | suffix token NLL | route accuracy |",
        "|---|---:|---:|---:|",
        *(
            f"| {condition} | {float(_object(suffix[condition], condition)['story_mean_nll']):.3f} "
            f"| {float(_object(suffix[condition], condition)['token_mean_nll']):.3f} "
            f"| {float(_object(suffix[condition], condition)['routing_accuracy']):.1%} |"
            for condition in CONDITIONS
        ),
        "",
        "## Representative completions",
        "",
        "The standalone HTML report contains folding successful, weak, correctly "
        "routed, and misrouted examples with the true suffix and all six greedy "
        "continuations.",
        "",
    ]
    judge = _object(data["judge"], "judge")
    lines.extend(
        _judge_markdown(judge)
        if judge.get("available")
        else (
            "## Optional OpenRouter judge",
            "",
            "No external judgment is attached. All local model work and reporting "
            "are complete; `--judge` can add judgments without repeating them.",
            "",
        )
    )
    lines.extend((f"Report identity: `{data['report_sha256']}`", ""))
    return "\n".join(lines)


def render_report_html(data: dict[str, object]) -> str:
    """Render a standalone interactive folding HTML report."""
    base = _object(data["base"], "report base")
    construction = _object(data["construction"], "construction")
    continual = _object(data["continual_learning"], "continual learning")
    overall = _object(data["overall_conditions"], "overall")
    task_metrics = tuple(
        _object(raw, "task metric") for raw in _list(data["task_metrics"], "tasks")
    )
    examples = tuple(
        _object(raw, "example") for raw in _list(data["examples"], "examples")
    )
    overall_rows = "".join(
        "<tr>"
        f"<td>{escape(condition)}</td>"
        f"<td>{float(_object(overall[condition], condition)['story_mean_nll']):.3f}</td>"
        f"<td>{float(_object(overall[condition], condition)['token_mean_nll']):.3f}</td>"
        f"<td>{float(_object(overall[condition], condition)['routing_accuracy']):.1%}</td>"
        f"<td>{float(_object(overall[condition], condition)['mean_regret']):+.3f}</td></tr>"
        for condition in CONDITIONS
    )
    task_rows = "".join(_task_html_row(row) for row in task_metrics)
    cards = "".join(_example_html(row) for row in examples)
    task_options = "".join(
        f"<option value='{escape(task_id)}'>{escape(task_id)}</option>"
        for task_id in TASK_IDS
    )
    graph_list = "".join(
        f"<li>{escape(str(row['parent'] or 'root'))} → "
        f"<b>{escape(str(row['node']))}</b> (depth {row['depth']})</li>"
        for raw in _list(data["graph"], "graph")
        for row in (_object(raw, "graph row"),)
        if row["node"] != "root"
    )
    graph_svg = render_vamp_graph_svg(data["graph"])
    route_chart = render_stagewise_route_chart_svg(continual)
    continual_conditions = _object(
        continual["condition_summaries"], "continual conditions"
    )
    continual_rows = "".join(
        _continual_html_row(
            condition,
            _object(continual_conditions[condition], condition),
        )
        for condition in CONDITIONS
    )
    stage_rows = "".join(
        _stagewise_html_row(_object(raw, "stage summary"))
        for raw in _list(continual["stages"], "stage summaries")
    )
    embedded = escape(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds nouns-v2 report</title>
<style>:root{{--ink:#172331;--line:#d5dde6;--wash:#f3f6f9;--paper:#fff;--accent:#315d9b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:2rem 1rem 5rem}}section,details.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:1.1rem;margin:1rem 0}}summary{{cursor:pointer;font-weight:650}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}.visual{{overflow:auto;background:#fbfcfe;border:1px solid var(--line);border-radius:10px;padding:.5rem}}.visual svg{{display:block;min-width:760px;width:100%;height:auto}}.chips{{display:flex;gap:.5rem;flex-wrap:wrap}}.chip{{background:#e7eef8;border-radius:999px;padding:.3rem .7rem}}pre{{white-space:pre-wrap;background:var(--wash);padding:.7rem;border-radius:8px}}.hidden{{display:none}}select{{font:inherit;padding:.35rem}}</style></head><body><main>
<h1>TinyWorlds nouns-v2 disjoint benchmark</h1><p>Every base/task boundary is story-disjoint. Multi-task stories are audited but never used for updates.</p><div class="chips"><span class="chip">{int(base['universe_story_count']):,} clean base stories</span><span class="chip">{int(construction['pure_task_train_story_count']):,} pure task stories</span><span class="chip">{int(construction['pure_validation_pair_count']):,} validation pairs</span></div>
<details class="card" open><summary>Disjoint construction</summary><p>Zero selected concepts → base; exactly one → its sole task; two or more → permanent exclusion. The base universe covers {float(base['universe_share']):.2%} of original training; its optimizer-visible share after the internal holdout is {float(base['optimizer_share']):.2%}.</p></details>
<details class="card" open><summary>Learned 24-stage VAMP dependency graph</summary><div class="visual">{graph_svg}</div><details><summary>Edge list</summary><ul>{graph_list}</ul></details></details>
<section><h2>Stagewise continual learning</h2><p>{int(continual['row_count']):,} task/story/stage cases measure every learned task after every later stage. Stored-oracle drift isolates parameter retention; router curves show interference from adding candidate nodes. Largest absolute oracle NLL drift: {float(continual['oracle_max_absolute_drift']):.6g}.</p><div class="visual">{route_chart}</div><div class="scroll"><table><thead><tr><th>Condition</th><th>Final story NLL</th><th>Final token NLL</th><th>Final accuracy</th><th>Mean forgetting</th><th>Max forgetting</th><th>Backward transfer</th><th>Accuracy change</th></tr></thead><tbody>{continual_rows}</tbody></table></div><details><summary>All 24 stage aggregates</summary><div class="scroll"><table><thead><tr><th>Stage</th><th>New task</th><th>Stories</th><th>Exhaustive</th><th>Hopfield</th><th>EBT-U</th><th>EBT-H</th><th>Oracle NLL</th></tr></thead><tbody>{stage_rows}</tbody></table></div></details></section>
<section><h2>Whole-story loss and routing</h2><div class="scroll"><table><thead><tr><th>Condition</th><th>Story NLL</th><th>Token NLL</th><th>Accuracy</th><th>Regret</th></tr></thead><tbody>{overall_rows}</tbody></table></div><h3>Per task</h3><div class="scroll"><table><thead><tr><th>Task</th><th>Train</th><th>Validation</th><th>Base</th><th>Oracle</th><th>Gain</th><th>Exhaustive</th><th>Hopfield</th><th>EBT-U</th><th>EBT-H</th></tr></thead><tbody>{task_rows}</tbody></table></div></section>
<section><h2>Explore midpoint completions</h2><p>Each router received only the first half. All conditions had the same deterministic token budget.</p><label>Task: <select id="filter"><option value="all">all</option>{task_options}</select></label><div id="examples">{cards}</div></section>
<details class="card"><summary>Exact report data and identities</summary><pre>{embedded}</pre></details><script>const f=document.getElementById('filter');f.addEventListener('change',()=>document.querySelectorAll('[data-task]').forEach(x=>x.classList.toggle('hidden',f.value!=='all'&&x.dataset.task!==f.value)));</script></main></body></html>"""


def render_vamp_graph_svg(raw_graph: object) -> str:
    """Render the learned dependency tree as deterministic standalone SVG."""
    if type(raw_graph) is not list or not raw_graph:
        raise TypeError("VAMP graph rendering requires a nonempty row list")
    graph = tuple(_object(raw, "graph row") for raw in raw_graph)
    by_node = {str(row["node"]): row for row in graph}
    if len(by_node) != len(graph) or "root" not in by_node:
        raise ValueError("VAMP graph rows must contain unique nodes and one root")
    children: dict[str, tuple[str, ...]] = {
        node_id: tuple(
            str(candidate["node"])
            for candidate in graph
            if candidate["parent"] == node_id
        )
        for node_id in by_node
    }

    def place_subtree(
        node_id: str,
        next_leaf: int,
        active: frozenset[str],
    ) -> tuple[dict[str, float], int]:
        if node_id in active:
            raise ValueError("VAMP graph rendering found a dependency cycle")
        descendants = children[node_id]
        if not descendants:
            return {node_id: 42.0 + next_leaf * 58.0}, next_leaf + 1
        positions: dict[str, float] = {}
        cursor = next_leaf
        for child in descendants:
            child_positions, cursor = place_subtree(
                child,
                cursor,
                active | {node_id},
            )
            positions.update(child_positions)
        positions[node_id] = sum(positions[child] for child in descendants) / len(
            descendants
        )
        return positions, cursor

    y_by_node, leaf_count = place_subtree("root", 0, frozenset())
    if set(y_by_node) != set(by_node):
        raise ValueError("VAMP graph contains nodes disconnected from root")
    node_width = 146.0
    node_height = 34.0
    x_by_node = {
        node_id: 30.0 + int(row["depth"]) * 184.0
        for node_id, row in by_node.items()
    }
    width = max(x_by_node.values()) + node_width + 30.0
    height = max(110.0, 84.0 + max(1, leaf_count - 1) * 58.0)
    palette = ("#334155", "#315d9b", "#287271", "#8c5e24", "#744e91", "#a13d4f")
    edges = "".join(
        (
            f"<path d='M{x_by_node[str(row['parent'])] + node_width:.1f},"
            f"{y_by_node[str(row['parent'])]:.1f} C"
            f"{x_by_node[str(row['parent'])] + node_width + 32:.1f},"
            f"{y_by_node[str(row['parent'])]:.1f} "
            f"{x_by_node[str(row['node'])] - 32:.1f},"
            f"{y_by_node[str(row['node'])]:.1f} "
            f"{x_by_node[str(row['node'])]:.1f},"
            f"{y_by_node[str(row['node'])]:.1f}'/>"
        )
        for row in graph
        if row["parent"] is not None
    )
    nodes = "".join(
        (
            f"<g><title>stage {int(row['stage'])}: {escape(str(row['node']))}; "
            f"parent {escape(str(row['parent'] or 'none'))}</title>"
            f"<rect x='{x_by_node[str(row['node'])]:.1f}' "
            f"y='{y_by_node[str(row['node'])] - node_height / 2:.1f}' "
            f"width='{node_width:.1f}' height='{node_height:.1f}' rx='8' "
            f"fill='{palette[min(int(row['depth']), len(palette) - 1)]}'/>"
            f"<text x='{x_by_node[str(row['node'])] + 10:.1f}' "
            f"y='{y_by_node[str(row['node'])] + 5:.1f}'>"
            f"{int(row['stage']):02d} · {escape(str(row['node']))}</text></g>"
        )
        for row in graph
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-labelledby='graph-title graph-desc' viewBox='0 0 {width:.0f} {height:.0f}'>"
        "<title id='graph-title'>Learned VAMP node dependencies</title>"
        "<desc id='graph-desc'>Directed edges run from each parent node to the "
        "task node trained from it. Labels begin with the training stage.</desc>"
        "<style>path{fill:none;stroke:#9aa8b7;stroke-width:1.8}"
        "text{fill:#fff;font:12px system-ui,sans-serif;font-weight:650}</style>"
        f"{edges}{nodes}</svg>"
    )


def render_stagewise_route_chart_svg(continual: dict[str, object]) -> str:
    """Render stagewise task-free route accuracy as inline deterministic SVG."""
    stages = tuple(
        _object(raw, "stage summary")
        for raw in _list(continual["stages"], "stage summaries")
    )
    if len(stages) < 2:
        raise ValueError("stagewise route chart requires at least two stages")
    conditions = CONDITIONS[2:]
    colors = ("#315d9b", "#287271", "#c06c2b", "#a13d4f")
    labels = ("exhaustive", "Hopfield", "EBT uniform", "EBT Hopfield")
    width, height = 920.0, 350.0
    left, right, top, bottom = 62.0, 22.0, 28.0, 62.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(stage_index: int, accuracy: float) -> tuple[float, float]:
        return (
            left + (stage_index - 1) * plot_width / (len(stages) - 1),
            top + (1.0 - accuracy) * plot_height,
        )

    grid = "".join(
        (
            f"<line x1='{left:.1f}' y1='{top + (1-value) * plot_height:.1f}' "
            f"x2='{width-right:.1f}' y2='{top + (1-value) * plot_height:.1f}'/>"
            f"<text x='{left-10:.1f}' y='{top + (1-value) * plot_height + 4:.1f}' "
            f"text-anchor='end'>{value:.0%}</text>"
        )
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    series = []
    for condition, color, label in zip(conditions, colors, labels):
        accuracies = tuple(
            float(
                _object(
                    _object(stage["conditions"], "stage conditions")[condition],
                    condition,
                )["routing_accuracy"]
            )
            for stage in stages
        )
        points = tuple(
            point(index, accuracy)
            for index, accuracy in enumerate(accuracies, start=1)
        )
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        markers = "".join(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.7'><title>stage "
            f"{index}: {label} {accuracy:.2%}</title></circle>"
            for index, ((x, y), accuracy) in enumerate(
                zip(points, accuracies), start=1
            )
        )
        series.append(
            f"<g style='color:{color}'><polyline points='{polyline}'/>"
            f"{markers}</g>"
        )
    tick_stages = tuple(
        stage for stage in (1, 4, 8, 12, 16, 20, len(stages))
        if stage <= len(stages)
    )
    x_ticks = "".join(
        f"<text x='{point(stage, 0.0)[0]:.1f}' y='{height-bottom+22:.1f}' "
        f"text-anchor='middle'>{stage}</text>"
        for stage in dict.fromkeys(tick_stages)
    )
    legend = "".join(
        f"<g transform='translate({left + index * 190:.0f},{height-13:.0f})' "
        f"style='color:{color}'><line x1='0' y1='0' x2='25' y2='0'/>"
        f"<text x='32' y='4'>{escape(label)}</text></g>"
        for index, (color, label) in enumerate(zip(colors, labels))
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-labelledby='route-title route-desc' viewBox='0 0 {width:.0f} {height:.0f}'>"
        "<title id='route-title'>Stagewise task-free routing accuracy</title>"
        "<desc id='route-desc'>Four routing methods evaluated on every learned task "
        "after each VAMP training stage.</desc>"
        "<style>line{stroke:#d5dde6;stroke-width:1}polyline{fill:none;stroke:currentColor;"
        "stroke-width:2.4}circle{fill:currentColor}text{fill:#425466;font:11px "
        "system-ui,sans-serif}.axis{fill:#172331;font-weight:650}</style>"
        f"{grid}{''.join(series)}{x_ticks}{legend}"
        f"<text class='axis' x='{left + plot_width/2:.1f}' y='{height-bottom+42:.1f}' "
        "text-anchor='middle'>VAMP training stage</text>"
        f"<text class='axis' transform='translate(16 {top+plot_height/2:.1f}) rotate(-90)' "
        "text-anchor='middle'>routing accuracy</text></svg>"
    )


CounterKey = dict[tuple[str, str, str], int]


def _task_metrics(
    task_id: str,
    train_count: int,
    validation_count: int,
    rows: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    summaries = {
        condition: _nll_summary(tuple(condition_rows))
        for condition, condition_rows in rows.items()
    }
    return {
        "acquisition": float(summaries["base"]["story_mean_nll"])
        - float(summaries["oracle"]["story_mean_nll"]),
        "conditions": summaries,
        "task": task_id,
        "train_story_count": train_count,
        "validation_story_count": validation_count,
    }


def _nll_summary(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    total_nll = sum(float(row["total_nll"]) for row in rows)
    tokens = sum(int(row["token_count"]) for row in rows)
    story_mean = _mean(float(row["mean_nll"]) for row in rows)
    return {
        "mean_regret": _mean(float(row["regret_vs_oracle"]) for row in rows),
        "routing_accuracy": _mean(float(bool(row["oracle_match"])) for row in rows),
        "story_count": len(rows),
        "story_mean_nll": story_mean,
        "story_perplexity": math.exp(min(story_mean, 700.0)),
        "token_count": tokens,
        "token_mean_nll": total_nll / tokens,
    }


def _suffix_summary(
    rows: tuple[dict[str, object], ...],
    condition: str,
) -> dict[str, object]:
    values = tuple(
        _object(_object(row["results"], "results")[condition], condition)
        for row in rows
    )
    total_nll = sum(float(value["total_nll"]) for value in values)
    tokens = sum(int(value["token_count"]) for value in values)
    return {
        "routing_accuracy": _mean(
            float(value["selected_node"] == row["task_noun"])
            for value, row in zip(values, rows)
        ),
        "story_mean_nll": _mean(float(value["mean_nll"]) for value in values),
        "token_mean_nll": total_nll / tokens,
    }


def _validate_coverage(
    partition: NounsV2PartitionArtifact,
    whole_rows: tuple[dict[str, object], ...],
    generation_rows: tuple[dict[str, object], ...],
    judge_rows: tuple[dict[str, object], ...],
) -> None:
    for row in whole_rows:
        _require_hashed_row(row, WHOLE_STORY_FORMAT)
    for row in generation_rows:
        _require_hashed_row(row, HALF_STORY_FORMAT)
    for row in judge_rows:
        _require_hashed_row(row, JUDGE_FORMAT, hash_field="result_sha256")
    whole_keys = {
        (str(row["task_noun"]), str(row["story_id"]), str(row["condition"]))
        for row in whole_rows
    }
    generation_keys = {
        (str(row["task_noun"]), str(row["story_id"])) for row in generation_rows
    }
    judge_keys = {
        (str(row["task_noun"]), str(row["story_id"])) for row in judge_rows
    }
    if (
        len(whole_rows) != PURE_TASK_VALIDATION_STORY_COUNT * len(CONDITIONS)
        or len(whole_keys) != len(whole_rows)
        or len(generation_rows) != PURE_TASK_VALIDATION_STORY_COUNT
        or len(generation_keys) != len(generation_rows)
        or (judge_rows and judge_keys != generation_keys)
        or any(
            sum(key[0] == task.task_id and key[2] == condition for key in whole_keys)
            != task.validation_story_count
            for task in partition.tasks
            for condition in CONDITIONS
        )
        or any(
            sum(key[0] == task.task_id for key in generation_keys)
            != task.validation_story_count
            for task in partition.tasks
        )
    ):
        raise ValueError("nouns-v2 result ledgers do not exactly cover the partition")


def _validate_final_stage_suffix_parity(
    stagewise_path: str | Path,
    generation_rows: tuple[dict[str, object], ...],
) -> None:
    generation = {
        (str(row["task_noun"]), str(row["story_id"])): _object(
            row["results"], "generation results"
        )
        for row in generation_rows
    }
    matched: set[tuple[str, str]] = set()
    with Path(stagewise_path).open("rb") as source:
        for line in source:
            row = _object(json.loads(line), "stagewise parity row")
            if row["stage_index"] != len(TASK_IDS):
                continue
            key = (str(row["task_noun"]), str(row["story_id"]))
            if key in matched or key not in generation:
                raise ValueError("final stagewise suffix keys differ from generation")
            stage_results = _object(row["results"], "stagewise results")
            for condition in CONDITIONS:
                measured = _object(stage_results[condition], condition)
                reference = _object(generation[key][condition], condition)
                if any(
                    measured[field] != reference[field]
                    for field in (
                        "mean_nll",
                        "selected_node",
                        "selected_path",
                        "token_count",
                        "total_nll",
                    )
                ):
                    raise ValueError(
                        "final stagewise suffix measurement differs from generation"
                    )
            matched.add(key)
    if matched != set(generation):
        raise ValueError("final stagewise suffix coverage differs from generation")


def _require_hashed_row(
    row: dict[str, object],
    expected_format: str,
    *,
    hash_field: str = "result_sha256",
) -> None:
    supplied = row.get(hash_field)
    core = {key: value for key, value in row.items() if key != hash_field}
    if row.get("format") != expected_format or supplied != record_sha256(core):
        raise ValueError(f"{expected_format} row identity changed")


def _overall_markdown_row(condition: str, summary: dict[str, object]) -> str:
    return (
        f"| {condition} | {float(summary['story_mean_nll']):.3f} | "
        f"{float(summary['token_mean_nll']):.3f} | "
        f"{float(summary['story_perplexity']):.2f} | "
        f"{float(summary['routing_accuracy']):.1%} | "
        f"{float(summary['mean_regret']):+.3f} |"
    )


def _continual_markdown_row(
    condition: str,
    continual: dict[str, object],
) -> str:
    summaries = _object(continual["condition_summaries"], "continual conditions")
    summary = _object(summaries[condition], condition)
    return (
        f"| {condition} | {float(summary['final_story_mean_nll']):.3f} | "
        f"{float(summary['final_token_mean_nll']):.3f} | "
        f"{float(summary['final_routing_accuracy']):.1%} | "
        f"{float(summary['mean_task_forgetting']):+.4f} | "
        f"{float(summary['max_task_forgetting']):+.4f} | "
        f"{float(summary['mean_backward_transfer']):+.4f} | "
        f"{float(summary['mean_route_accuracy_change']):+.1%} |"
    )


def _stagewise_markdown_row(stage: dict[str, object]) -> str:
    conditions = _object(stage["conditions"], "stage conditions")
    accuracies = tuple(
        float(_object(conditions[condition], condition)["routing_accuracy"])
        for condition in CONDITIONS[2:]
    )
    oracle = _object(conditions["oracle"], "oracle")
    return (
        f"| {int(stage['stage_index'])} | {stage['introduced_task']} | "
        f"{int(stage['story_count']):,} | "
        + " | ".join(f"{accuracy:.1%}" for accuracy in accuracies)
        + f" | {float(oracle['story_mean_nll']):.3f} |"
    )


def _continual_html_row(condition: str, summary: dict[str, object]) -> str:
    values = (
        condition,
        f"{float(summary['final_story_mean_nll']):.3f}",
        f"{float(summary['final_token_mean_nll']):.3f}",
        f"{float(summary['final_routing_accuracy']):.1%}",
        f"{float(summary['mean_task_forgetting']):+.4f}",
        f"{float(summary['max_task_forgetting']):+.4f}",
        f"{float(summary['mean_backward_transfer']):+.4f}",
        f"{float(summary['mean_route_accuracy_change']):+.1%}",
    )
    return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"


def _stagewise_html_row(stage: dict[str, object]) -> str:
    return "<tr>" + "".join(
        f"<td>{escape(value.strip())}</td>"
        for value in _stagewise_markdown_row(stage).strip("|").split("|")
    ) + "</tr>"


def _task_markdown_row(row: dict[str, object]) -> str:
    conditions = _object(row["conditions"], "task conditions")
    values = tuple(
        _object(conditions[name], name) for name in CONDITIONS
    )
    return (
        f"| {row['task']} | {int(row['train_story_count']):,} | "
        f"{int(row['validation_story_count']):,} | "
        f"{float(values[0]['story_mean_nll']):.3f} | "
        f"{float(values[1]['story_mean_nll']):.3f} | "
        f"{float(row['acquisition']):+.3f} | "
        + " | ".join(f"{float(value['routing_accuracy']):.1%}" for value in values[2:])
        + " |"
    )


def _task_html_row(row: dict[str, object]) -> str:
    return "<tr>" + "".join(
        f"<td>{escape(value)}</td>"
        for value in _task_markdown_row(row).strip("|").split("|")
    ) + "</tr>"


def _example_html(row: dict[str, object]) -> str:
    results = _object(row["results"], "example results")
    completions = "".join(
        "<details><summary>"
        f"{escape(condition)} — route {escape(str(_object(results[condition], condition)['selected_node']))}"
        "</summary><pre>"
        f"{escape(str(_object(results[condition], condition)['generated_continuation']))}"
        "</pre></details>"
        for condition in CONDITIONS
    )
    return (
        f"<details class='card' data-task='{escape(str(row['task']))}'><summary>"
        f"{escape(str(row['task']))}: {escape(str(row['sample_reason']))} "
        f"(oracle gain {float(row['base_to_oracle_gain']):+.3f})</summary>"
        f"<p><b>Prefix</b></p><pre>{escape(str(row['prefix']))}</pre>"
        f"<p><b>Reference</b></p><pre>{escape(str(row['reference_continuation']))}</pre>"
        f"{completions}</details>"
    )


def _judge_markdown(judge: dict[str, object]) -> tuple[str, ...]:
    conditions = _object(judge["conditions"], "judge conditions")
    return (
        "## Optional OpenRouter judge",
        "",
        "| source | mean score | mean rank | win vs base | win vs oracle | win vs reference |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {name} | {float(values['mean_overall']):.2f} | "
            f"{float(values['mean_rank']):.2f} | "
            f"{float(values['win_rate_vs_base']):.1%} | "
            f"{float(values['win_rate_vs_oracle']):.1%} | "
            f"{float(values['win_rate_vs_reference']):.1%} |"
            for name, raw in conditions.items()
            for values in (_object(raw, "judge condition"),)
        ),
        "",
    )


def _publish_confusion_csv(root: Path, data: dict[str, object]) -> None:
    confusion = _object(data["routing_confusion"], "routing confusion")
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=root, newline="", encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        writer = csv.writer(output)
        writer.writerow(("condition", "oracle_task", "selected_node", "story_count"))
        writer.writerows(
            (condition, task_id, node_id, int(count))
            for condition, raw_matrix in confusion.items()
            for task_id, raw_counts in _object(raw_matrix, "matrix").items()
            for node_id, count in _object(raw_counts, "counts").items()
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, root / "routing-confusion.csv")


def _publish_stagewise_csvs(root: Path, data: dict[str, object]) -> None:
    continual = _object(data["continual_learning"], "continual learning")
    stage_rows = tuple(
        (
            int(stage["stage_index"]),
            str(stage["introduced_task"]),
            int(stage["story_count"]),
            condition,
            float(summary["story_mean_nll"]),
            float(summary["token_mean_nll"]),
            float(summary["routing_accuracy"]),
            float(summary["mean_regret"]),
        )
        for raw_stage in _list(continual["stages"], "stage summaries")
        for stage in (_object(raw_stage, "stage summary"),)
        for condition, raw_summary in _object(
            stage["conditions"], "stage conditions"
        ).items()
        for summary in (_object(raw_summary, condition),)
    )
    task_rows = tuple(
        (
            str(task["task"]),
            int(task["introduction_stage"]),
            condition,
            int(summary["best_stage"]),
            float(summary["introduction_story_mean_nll"]),
            float(summary["best_story_mean_nll"]),
            float(summary["final_story_mean_nll"]),
            float(summary["forgetting"]),
            float(summary["backward_transfer"]),
            float(summary["introduction_routing_accuracy"]),
            float(summary["final_routing_accuracy"]),
            float(summary["accuracy_change"]),
        )
        for raw_task in _list(continual["task_metrics"], "stagewise task metrics")
        for task in (_object(raw_task, "stagewise task metric"),)
        for condition, raw_summary in _object(
            task["conditions"], "task conditions"
        ).items()
        for summary in (_object(raw_summary, condition),)
    )
    _write_csv(
        root / "stagewise-summary.csv",
        (
            "stage",
            "introduced_task",
            "story_count",
            "condition",
            "story_mean_nll",
            "token_mean_nll",
            "routing_accuracy",
            "mean_oracle_regret",
        ),
        stage_rows,
    )
    _write_csv(
        root / "stagewise-task-metrics.csv",
        (
            "task",
            "introduction_stage",
            "condition",
            "best_stage",
            "introduction_story_mean_nll",
            "best_story_mean_nll",
            "final_story_mean_nll",
            "forgetting",
            "backward_transfer",
            "introduction_routing_accuracy",
            "final_routing_accuracy",
            "accuracy_change",
        ),
        task_rows,
    )


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, newline="", encoding="utf-8"
    ) as output:
        temporary = Path(output.name)
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_bytes().splitlines()
        if line
    )
    if any(canonical_json_bytes(row).rstrip(b"\n") != line for row, line in zip(rows, path.read_bytes().splitlines())):
        raise ValueError(f"report ledger is not canonical JSONL: {path}")
    return tuple(_object(row, "report ledger row") for row in rows)


def _mean(values) -> float:
    measured = tuple(values)
    if not measured:
        raise ValueError("report mean requires values")
    return sum(measured) / len(measured)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


__all__ = [
    "build_report_data",
    "publish_nouns_v2_report",
    "render_report_html",
    "render_report_markdown",
    "render_stagewise_route_chart_svg",
    "render_vamp_graph_svg",
]
