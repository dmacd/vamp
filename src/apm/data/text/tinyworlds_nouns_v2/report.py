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
    BASELINE_CONDITIONS,
    CONDITIONS,
    EXCLUDED_TRAIN_STORY_COUNT,
    EXCLUDED_VALIDATION_STORY_COUNT,
    HALF_STORY_FORMAT,
    JUDGE_FORMAT,
    PURE_TASK_TRAIN_STORY_COUNT,
    PURE_TASK_VALIDATION_STORY_COUNT,
    REPORT_FORMAT,
    FULL_FINETUNE_CONDITIONS,
    TASK_IDS,
    TRAIN_UNIQUE_STORY_COUNT,
    WHOLE_STORY_FORMAT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.baseline_stagewise import (
    summarize_baseline_stagewise_ledger,
)
from apm.data.text.tinyworlds_nouns_v2.full_finetune import FullFinetuneStage
from apm.data.text.tinyworlds_nouns_v2.full_finetune_stagewise import (
    summarize_full_finetune_stagewise_ledger,
)
from apm.data.text.tinyworlds_nouns_v2.report_plots import (
    render_dependency_graph_svg,
    render_line_chart_svg,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise import summarize_stagewise_ledger


def publish_nouns_v2_report(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    baseline_stages: tuple[LanguageAdaptationArtifact, ...],
    full_finetune_stages: tuple[FullFinetuneStage, ...],
    whole_story_path: str | Path,
    generation_path: str | Path,
    stagewise_path: str | Path,
    baseline_stagewise_path: str | Path,
    full_finetune_stagewise_path: str | Path,
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
    baseline_continual_learning = summarize_baseline_stagewise_ledger(
        baseline_stagewise_path,
        partition,
        baseline_stages,
        adaptations,
        stagewise_path,
    )
    full_finetune_continual_learning = summarize_full_finetune_stagewise_ledger(
        full_finetune_stagewise_path,
        partition,
        preset,
        full_finetune_stages,
        adaptations,
        stagewise_path,
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
        baseline_continual_learning,
        full_finetune_continual_learning,
    )
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    _publish_graph_artifacts(root, adaptation)
    _atomic_write(
        root / "vamp-graph.svg",
        render_vamp_graph_svg(data["graph"]).encode("utf-8"),
    )
    _atomic_write(
        root / "stagewise-routing.svg",
        render_stagewise_route_chart_svg(continual_learning).encode("utf-8"),
    )
    _atomic_write(
        root / "continual-nll-comparison.svg",
        render_comparative_nll_chart_svg(
            continual_learning,
            baseline_continual_learning,
            full_finetune_continual_learning,
        ).encode("utf-8"),
    )
    _publish_confusion_csv(root, data)
    _publish_stagewise_csvs(root, data)
    _publish_baseline_stagewise_csvs(root, data)
    _publish_full_finetune_stagewise_csvs(root, data)
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
    baseline_continual_learning: dict[str, object],
    full_finetune_continual_learning: dict[str, object],
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
        "baseline_continual_learning": baseline_continual_learning,
        "config_sha256": preset.config_sha256,
        "construction": {
            "excluded_train_story_count": EXCLUDED_TRAIN_STORY_COUNT,
            "excluded_validation_story_count": EXCLUDED_VALIDATION_STORY_COUNT,
            "pure_task_train_story_count": PURE_TASK_TRAIN_STORY_COUNT,
            "pure_validation_pair_count": PURE_TASK_VALIDATION_STORY_COUNT,
        },
        "continual_learning": continual_learning,
        "full_finetune_continual_learning": full_finetune_continual_learning,
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
    baseline_continual = _object(
        data["baseline_continual_learning"],
        "baseline continual learning",
    )
    full_finetune_continual = _object(
        data["full_finetune_continual_learning"],
        "full-finetune continual learning",
    )
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
        "<details>",
        "<summary>Complete parent-to-child edge list</summary>",
        "",
        *(
            f"- `{row['node']}` attached to "
            f"`{row['parent'] if row['parent'] is not None else 'none (root)'}` "
            f"at depth {row['depth']}."
            for row in graph
        ),
        "",
        "</details>",
        "",
        "## Sequential controls, independent adapters, and VAMP",
        "",
        "The sequential control is one rank-eight LoRA that is updated in place "
        "for all 24 tasks; it receives no task identity at evaluation. The "
        "independent control trains one fresh root LoRA per task and evaluates "
        "with the correct task adapter, so it is a task-aware isolation ceiling "
        "rather than a deployable task-free router. The full-finetune control "
        "updates every GPT-Neo parameter sequentially and also receives no task "
        "identity at evaluation.",
        "",
        f"The largest absolute independent-adapter NLL drift is "
        f"{float(baseline_continual['independent_max_absolute_drift']):.6g}. "
        "All systems use the same base, task order, 2,000-update budget, "
        "validation stories, midpoint split, and true-suffix loss. Adapter "
        "methods use rank/alpha eight and learning rate 1e-3; full fine-tuning "
        "uses every model parameter at learning rate 5e-5.",
        "",
        "![Stagewise continual-learning NLL comparison](continual-nll-comparison.svg)",
        "",
        "| system | task identity | final story NLL | final token NLL | final route accuracy | mean forgetting | max forgetting | backward transfer |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *_comparison_markdown_rows(
            continual,
            baseline_continual,
            full_finetune_continual,
        ),
        "",
        "<details>",
        "<summary>All 24 sequential and independent stage aggregates</summary>",
        "",
        "| stage | new task | retained stories | sequential LoRA | full fine-tune | independent | LoRA deficit | full deficit |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *(
            _baseline_stage_markdown_row(
                _object(raw_baseline, "baseline stage summary"),
                _object(raw_full, "full-finetune stage summary"),
            )
            for raw_baseline, raw_full in zip(
                _list(baseline_continual["stages"], "baseline stages"),
                _list(full_finetune_continual["stages"], "full-finetune stages"),
                strict=True,
            )
        ),
        "",
        "</details>",
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
        "![Stagewise task-free routing accuracy](stagewise-routing.svg)",
        "",
        "| condition | final story NLL | final token NLL | final route accuracy | mean forgetting | max forgetting | backward transfer | route accuracy change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            _continual_markdown_row(condition, continual)
            for condition in CONDITIONS
        ),
        "",
        "<details>",
        "<summary>All 24 VAMP stage aggregates</summary>",
        "",
        "| stage | new task | retained stories | exhaustive | Hopfield | EBT uniform | EBT Hopfield | oracle NLL |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *(
            _stagewise_markdown_row(_object(raw, "stage summary"))
            for raw in _list(continual["stages"], "stage summaries")
        ),
        "",
        "</details>",
        "",
        "Detailed task-level introduction, best, and final measurements are in "
        "`stagewise-task-metrics.csv`, `baseline-stagewise-task-metrics.csv`, and "
        "`full-finetune-stagewise-task-metrics.csv`; "
        "the complete stage curves are in `stagewise-summary.csv` and "
        "`baseline-stagewise-summary.csv`, and "
        "`full-finetune-stagewise-summary.csv`.",
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
        "<details>",
        "<summary>Whole-story results for every task</summary>",
        "",
        "| task | training stories | validation | base NLL | oracle NLL | acquisition | exhaustive | Hopfield | EBT uniform | EBT Hopfield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(_task_markdown_row(row) for row in task_metrics),
        "",
        "</details>",
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
    baseline_continual = _object(
        data["baseline_continual_learning"],
        "baseline continual learning",
    )
    full_finetune_continual = _object(
        data["full_finetune_continual_learning"],
        "full-finetune continual learning",
    )
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
    comparison_chart = render_comparative_nll_chart_svg(
        continual,
        baseline_continual,
        full_finetune_continual,
    )
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
    comparison_rows = "".join(
        _comparison_html_rows(
            continual,
            baseline_continual,
            full_finetune_continual,
        )
    )
    baseline_stage_rows = "".join(
        _baseline_stage_html_row(
            _object(raw_baseline, "baseline stage summary"),
            _object(raw_full, "full-finetune stage summary"),
        )
        for raw_baseline, raw_full in zip(
            _list(baseline_continual["stages"], "baseline stages"),
            _list(full_finetune_continual["stages"], "full-finetune stages"),
            strict=True,
        )
    )
    embedded = escape(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds nouns-v2 report</title>
<style>:root{{--ink:#172331;--line:#cbd5e1;--wash:#f1f5f9;--paper:#fff;--accent:#315d9b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:17px/1.6 system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:2rem 1rem 5rem}}section,details.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:1.15rem;margin:1rem 0}}summary{{cursor:pointer;font-weight:700;font-size:1.08rem}}table{{border-collapse:collapse;width:100%;font-size:.96rem}}th,td{{padding:.55rem;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:#334155;background:#f8fafc}}.scroll{{overflow:auto}}.visual{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;padding:.7rem;margin:.8rem 0}}.visual svg{{display:block;min-width:780px;width:100%;height:auto}}.chips{{display:flex;gap:.5rem;flex-wrap:wrap}}.chip{{background:#dbeafe;color:#1e3a5f;border-radius:999px;padding:.3rem .7rem}}pre{{white-space:pre-wrap;background:var(--wash);padding:.7rem;border-radius:8px}}.hidden{{display:none}}select{{font:inherit;padding:.35rem}}</style></head><body><main>
<h1>TinyWorlds nouns-v2 disjoint benchmark</h1><p>Every base/task boundary is story-disjoint. Multi-task stories are audited but never used for updates.</p><div class="chips"><span class="chip">{int(base['universe_story_count']):,} clean base stories</span><span class="chip">{int(construction['pure_task_train_story_count']):,} pure task stories</span><span class="chip">{int(construction['pure_validation_pair_count']):,} validation pairs</span></div>
<details class="card" open><summary>Disjoint construction</summary><p>Zero selected concepts → base; exactly one → its sole task; two or more → permanent exclusion. The base universe covers {float(base['universe_share']):.2%} of original training; its optimizer-visible share after the internal holdout is {float(base['optimizer_share']):.2%}.</p></details>
<details class="card" open><summary>Learned 24-stage VAMP dependency graph</summary><div class="visual">{graph_svg}</div><details><summary>Edge list</summary><ul>{graph_list}</ul></details></details>
<details class="card" open><summary>Sequential controls, independent adapters, and VAMP</summary><p>One sequential control continually overwrites a rank-eight LoRA; the other fine-tunes every GPT-Neo parameter. Neither receives task identity. The independent control uses a fresh task-matched root LoRA and is a task-aware ceiling. Every row uses the same base, task order, 2,000-update budget, validation stories, midpoint split, and true suffix. Adapter methods use learning rate 1e-3; full fine-tuning uses 5e-5.</p><p>Largest absolute independent-adapter NLL drift: {float(baseline_continual['independent_max_absolute_drift']):.6g}.</p><div class="visual">{comparison_chart}</div><div class="scroll"><table><thead><tr><th>System</th><th>Task identity</th><th>Final story NLL</th><th>Final token NLL</th><th>Final route accuracy</th><th>Mean forgetting</th><th>Max forgetting</th><th>Backward transfer</th></tr></thead><tbody>{comparison_rows}</tbody></table></div><details><summary>All 24 stored-baseline stages</summary><div class="scroll"><table><thead><tr><th>Stage</th><th>New task</th><th>Stories</th><th>Sequential LoRA</th><th>Full fine-tune</th><th>Independent</th><th>LoRA deficit</th><th>Full deficit</th></tr></thead><tbody>{baseline_stage_rows}</tbody></table></div></details></details>
<details class="card" open><summary>VAMP routing across all 24 stages</summary><p>{int(continual['row_count']):,} task/story/stage cases measure every learned task after every later stage. Stored-oracle drift isolates parameter retention; router curves show interference from adding candidate nodes. Largest absolute oracle NLL drift: {float(continual['oracle_max_absolute_drift']):.6g}.</p><div class="visual">{route_chart}</div><div class="scroll"><table><thead><tr><th>Condition</th><th>Final story NLL</th><th>Final token NLL</th><th>Final accuracy</th><th>Mean forgetting</th><th>Max forgetting</th><th>Backward transfer</th><th>Accuracy change</th></tr></thead><tbody>{continual_rows}</tbody></table></div><details><summary>All 24 VAMP stage aggregates</summary><div class="scroll"><table><thead><tr><th>Stage</th><th>New task</th><th>Stories</th><th>Exhaustive</th><th>Hopfield</th><th>EBT-U</th><th>EBT-H</th><th>Oracle NLL</th></tr></thead><tbody>{stage_rows}</tbody></table></div></details></details>
<details class="card"><summary>Whole-story loss and routing</summary><div class="scroll"><table><thead><tr><th>Condition</th><th>Story NLL</th><th>Token NLL</th><th>Accuracy</th><th>Regret</th></tr></thead><tbody>{overall_rows}</tbody></table></div><details><summary>Per-task whole-story results</summary><div class="scroll"><table><thead><tr><th>Task</th><th>Train</th><th>Validation</th><th>Base</th><th>Oracle</th><th>Gain</th><th>Exhaustive</th><th>Hopfield</th><th>EBT-U</th><th>EBT-H</th></tr></thead><tbody>{task_rows}</tbody></table></div></details></details>
<details class="card"><summary>Explore midpoint completions</summary><p>Each router received only the first half. All conditions had the same deterministic token budget.</p><label>Task: <select id="filter"><option value="all">all</option>{task_options}</select></label><div id="examples">{cards}</div></details>
<details class="card"><summary>Exact report data and identities</summary><pre>{embedded}</pre></details><script>const f=document.getElementById('filter');f.addEventListener('change',()=>document.querySelectorAll('[data-task]').forEach(x=>x.classList.toggle('hidden',f.value!=='all'&&x.dataset.task!==f.value)));</script></main></body></html>"""


def render_vamp_graph_svg(raw_graph: object) -> str:
    """Render the learned dependency tree through Graphviz."""
    return render_dependency_graph_svg(raw_graph)


def render_stagewise_route_chart_svg(continual: dict[str, object]) -> str:
    """Render stagewise task-free routing accuracy through Matplotlib."""
    stages = tuple(
        _object(raw, "stage summary")
        for raw in _list(continual["stages"], "stage summaries")
    )
    rows = tuple(
        {
            "stage": float(stage["stage_index"]),
            **{
                condition: float(
                    _object(
                        _object(stage["conditions"], "stage conditions")[condition],
                        condition,
                    )["routing_accuracy"]
                )
                for condition in CONDITIONS[2:]
            },
        }
        for stage in stages
    )
    return render_line_chart_svg(
        rows,
        (
            ("vamp_exhaustive", "VAMP exhaustive"),
            ("vamp_hopfield", "VAMP Hopfield"),
            ("vamp_ebt_uniform", "VAMP EBT uniform"),
            ("vamp_ebt_hopfield", "VAMP EBT Hopfield"),
        ),
        title="Task-free routing as the VAMP graph grows",
        y_label="routing accuracy",
        y_bounds=(0.0, 1.0),
    )


def render_comparative_nll_chart_svg(
    continual: dict[str, object],
    baseline_continual: dict[str, object],
    full_finetune_continual: dict[str, object],
) -> str:
    """Render matched stagewise suffix NLL for stored and routed systems."""
    vamp_stages = tuple(
        _object(raw, "VAMP stage")
        for raw in _list(continual["stages"], "VAMP stages")
    )
    baseline_stages = tuple(
        _object(raw, "baseline stage")
        for raw in _list(baseline_continual["stages"], "baseline stages")
    )
    full_stages = tuple(
        _object(raw, "full-finetune stage")
        for raw in _list(
            full_finetune_continual["stages"],
            "full-finetune stages",
        )
    )
    if len(vamp_stages) != len(baseline_stages) or len(vamp_stages) != len(
        full_stages
    ):
        raise ValueError("comparison plot stage counts differ")
    rows = tuple(
        {
            "stage": float(vamp_stage["stage_index"]),
            "sequential": _stage_condition_nll(baseline_stage, "sequential_single_lora"),
            "full_finetune": _stage_condition_nll(
                full_stage,
                FULL_FINETUNE_CONDITIONS[0],
            ),
            "independent": _stage_condition_nll(baseline_stage, "independent_root_lora"),
            "vamp_oracle": _stage_condition_nll(vamp_stage, "oracle"),
            "vamp_exhaustive": _stage_condition_nll(vamp_stage, "vamp_exhaustive"),
            "vamp_ebt_uniform": _stage_condition_nll(vamp_stage, "vamp_ebt_uniform"),
        }
        for vamp_stage, baseline_stage, full_stage in zip(
            vamp_stages,
            baseline_stages,
            full_stages,
            strict=True,
        )
    )
    return render_line_chart_svg(
        rows,
        (
            ("sequential", "Sequential single LoRA"),
            ("full_finetune", "Sequential full fine-tune"),
            ("independent", "Independent task adapter"),
            ("vamp_oracle", "VAMP stored oracle"),
            ("vamp_exhaustive", "VAMP exhaustive"),
            ("vamp_ebt_uniform", "VAMP EBT uniform"),
        ),
        title="Continual-learning suffix loss under matched budgets",
        y_label="story-weighted true-suffix NLL",
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


def _comparison_values(
    continual: dict[str, object],
    baseline_continual: dict[str, object],
    full_finetune_continual: dict[str, object],
) -> tuple[dict[str, object], ...]:
    baseline_summaries = _object(
        baseline_continual["condition_summaries"],
        "baseline condition summaries",
    )
    vamp_summaries = _object(
        continual["condition_summaries"],
        "VAMP condition summaries",
    )
    full_finetune_summaries = _object(
        full_finetune_continual["condition_summaries"],
        "full-finetune condition summaries",
    )
    baseline_rows = tuple(
        {
            "backward_transfer": float(summary["mean_backward_transfer"]),
            "final_route_accuracy": None,
            "final_story_mean_nll": float(summary["final_story_mean_nll"]),
            "final_token_mean_nll": float(summary["final_token_mean_nll"]),
            "max_forgetting": float(summary["max_task_forgetting"]),
            "mean_forgetting": float(summary["mean_task_forgetting"]),
            "system": label,
            "task_identity": task_identity,
        }
        for condition, label, task_identity in (
            ("sequential_single_lora", "sequential single LoRA", "no"),
            ("independent_root_lora", "independent root LoRA", "required"),
        )
        for summary in (_object(baseline_summaries[condition], condition),)
    )
    full_finetune_rows = tuple(
        {
            "backward_transfer": float(summary["mean_backward_transfer"]),
            "final_route_accuracy": None,
            "final_story_mean_nll": float(summary["final_story_mean_nll"]),
            "final_token_mean_nll": float(summary["final_token_mean_nll"]),
            "max_forgetting": float(summary["max_task_forgetting"]),
            "mean_forgetting": float(summary["mean_task_forgetting"]),
            "system": "sequential full fine-tune",
            "task_identity": "no",
        }
        for summary in (
            _object(
                full_finetune_summaries[FULL_FINETUNE_CONDITIONS[0]],
                FULL_FINETUNE_CONDITIONS[0],
            ),
        )
    )
    vamp_rows = tuple(
        {
            "backward_transfer": float(summary["mean_backward_transfer"]),
            "final_route_accuracy": float(summary["final_routing_accuracy"]),
            "final_story_mean_nll": float(summary["final_story_mean_nll"]),
            "final_token_mean_nll": float(summary["final_token_mean_nll"]),
            "max_forgetting": float(summary["max_task_forgetting"]),
            "mean_forgetting": float(summary["mean_task_forgetting"]),
            "system": label,
            "task_identity": task_identity,
        }
        for condition, label, task_identity in (
            ("oracle", "VAMP stored oracle", "required"),
            ("vamp_exhaustive", "VAMP exhaustive", "no"),
            ("vamp_hopfield", "VAMP Hopfield", "no"),
            ("vamp_ebt_uniform", "VAMP EBT uniform", "no"),
            ("vamp_ebt_hopfield", "VAMP EBT Hopfield", "no"),
        )
        for summary in (_object(vamp_summaries[condition], condition),)
    )
    return baseline_rows[:1] + full_finetune_rows + baseline_rows[1:] + vamp_rows


def _comparison_markdown_rows(
    continual: dict[str, object],
    baseline_continual: dict[str, object],
    full_finetune_continual: dict[str, object],
) -> tuple[str, ...]:
    return tuple(
        f"| {row['system']} | {row['task_identity']} | "
        f"{float(row['final_story_mean_nll']):.3f} | "
        f"{float(row['final_token_mean_nll']):.3f} | "
        f"{_optional_accuracy(row['final_route_accuracy'])} | "
        f"{float(row['mean_forgetting']):+.4f} | "
        f"{float(row['max_forgetting']):+.4f} | "
        f"{float(row['backward_transfer']):+.4f} |"
        for row in _comparison_values(
            continual,
            baseline_continual,
            full_finetune_continual,
        )
    )


def _comparison_html_rows(
    continual: dict[str, object],
    baseline_continual: dict[str, object],
    full_finetune_continual: dict[str, object],
) -> tuple[str, ...]:
    return tuple(
        "<tr>"
        + "".join(
            f"<td>{escape(value)}</td>"
            for value in (
                str(row["system"]),
                str(row["task_identity"]),
                f"{float(row['final_story_mean_nll']):.3f}",
                f"{float(row['final_token_mean_nll']):.3f}",
                _optional_accuracy(row["final_route_accuracy"]),
                f"{float(row['mean_forgetting']):+.4f}",
                f"{float(row['max_forgetting']):+.4f}",
                f"{float(row['backward_transfer']):+.4f}",
            )
        )
        + "</tr>"
        for row in _comparison_values(
            continual,
            baseline_continual,
            full_finetune_continual,
        )
    )


def _optional_accuracy(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _stage_condition_nll(stage: dict[str, object], condition: str) -> float:
    return float(
        _object(_object(stage["conditions"], "stage conditions")[condition], condition)[
            "story_mean_nll"
        ]
    )


def _baseline_stage_markdown_row(
    stage: dict[str, object],
    full_finetune_stage: dict[str, object],
) -> str:
    conditions = _object(stage["conditions"], "baseline stage conditions")
    sequential = _object(conditions["sequential_single_lora"], "sequential")
    independent = _object(conditions["independent_root_lora"], "independent")
    full_finetune = _object(
        _object(
            full_finetune_stage["conditions"],
            "full-finetune stage conditions",
        )[FULL_FINETUNE_CONDITIONS[0]],
        "full-finetune stage",
    )
    return (
        f"| {int(stage['stage_index'])} | {stage['introduced_task']} | "
        f"{int(stage['story_count']):,} | "
        f"{float(sequential['story_mean_nll']):.3f} | "
        f"{float(full_finetune['story_mean_nll']):.3f} | "
        f"{float(independent['story_mean_nll']):.3f} | "
        f"{float(sequential['mean_deficit_vs_independent']):+.3f} | "
        f"{float(full_finetune['story_mean_nll']) - float(independent['story_mean_nll']):+.3f} |"
    )


def _baseline_stage_html_row(
    stage: dict[str, object],
    full_finetune_stage: dict[str, object],
) -> str:
    return "<tr>" + "".join(
        f"<td>{escape(value.strip())}</td>"
        for value in _baseline_stage_markdown_row(stage, full_finetune_stage)
        .strip("|")
        .split("|")
    ) + "</tr>"


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


def _publish_baseline_stagewise_csvs(
    root: Path,
    data: dict[str, object],
) -> None:
    baseline = _object(
        data["baseline_continual_learning"],
        "baseline continual learning",
    )
    stage_rows = tuple(
        (
            int(stage["stage_index"]),
            str(stage["introduced_task"]),
            int(stage["story_count"]),
            condition,
            float(summary["story_mean_nll"]),
            float(summary["token_mean_nll"]),
            float(summary["mean_deficit_vs_independent"]),
        )
        for raw_stage in _list(baseline["stages"], "baseline stage summaries")
        for stage in (_object(raw_stage, "baseline stage summary"),)
        for condition in BASELINE_CONDITIONS
        for summary in (
            _object(
                _object(stage["conditions"], "baseline stage conditions")[condition],
                condition,
            ),
        )
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
        )
        for raw_task in _list(baseline["task_metrics"], "baseline task metrics")
        for task in (_object(raw_task, "baseline task metric"),)
        for condition in BASELINE_CONDITIONS
        for summary in (
            _object(
                _object(task["conditions"], "baseline task conditions")[condition],
                condition,
            ),
        )
    )
    _write_csv(
        root / "baseline-stagewise-summary.csv",
        (
            "stage",
            "introduced_task",
            "story_count",
            "condition",
            "story_mean_nll",
            "token_mean_nll",
            "mean_deficit_vs_independent",
        ),
        stage_rows,
    )
    _write_csv(
        root / "baseline-stagewise-task-metrics.csv",
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
        ),
        task_rows,
    )


def _publish_full_finetune_stagewise_csvs(
    root: Path,
    data: dict[str, object],
) -> None:
    full_finetune = _object(
        data["full_finetune_continual_learning"],
        "full-finetune continual learning",
    )
    baseline = _object(
        data["baseline_continual_learning"],
        "baseline continual learning",
    )
    full_stages = tuple(
        _object(raw, "full-finetune stage summary")
        for raw in _list(full_finetune["stages"], "full-finetune stages")
    )
    baseline_stages = tuple(
        _object(raw, "baseline stage summary")
        for raw in _list(baseline["stages"], "baseline stages")
    )
    if len(full_stages) != len(baseline_stages):
        raise ValueError("full-finetune and baseline stage counts differ")
    stage_rows: list[tuple[object, ...]] = []
    for full_stage, baseline_stage in zip(
        full_stages,
        baseline_stages,
        strict=True,
    ):
        if any(
            full_stage[field] != baseline_stage[field]
            for field in ("introduced_task", "stage_index", "story_count")
        ):
            raise ValueError("full-finetune and baseline stage coverage differs")
        full_summary = _object(
            _object(full_stage["conditions"], "full-finetune stage conditions")[
                FULL_FINETUNE_CONDITIONS[0]
            ],
            "full-finetune stage condition",
        )
        independent = _object(
            _object(baseline_stage["conditions"], "baseline stage conditions")[
                "independent_root_lora"
            ],
            "independent stage condition",
        )
        stage_rows.append(
            (
                int(full_stage["stage_index"]),
                str(full_stage["introduced_task"]),
                int(full_stage["story_count"]),
                FULL_FINETUNE_CONDITIONS[0],
                float(full_summary["story_mean_nll"]),
                float(full_summary["token_mean_nll"]),
                float(independent["story_mean_nll"]),
                float(full_summary["story_mean_nll"])
                - float(independent["story_mean_nll"]),
            )
        )

    full_tasks = tuple(
        _object(raw, "full-finetune task metric")
        for raw in _list(full_finetune["task_metrics"], "full-finetune task metrics")
    )
    baseline_tasks = tuple(
        _object(raw, "baseline task metric")
        for raw in _list(baseline["task_metrics"], "baseline task metrics")
    )
    if len(full_tasks) != len(baseline_tasks):
        raise ValueError("full-finetune and baseline task counts differ")
    task_rows: list[tuple[object, ...]] = []
    for full_task, baseline_task in zip(full_tasks, baseline_tasks, strict=True):
        if any(
            full_task[field] != baseline_task[field]
            for field in ("introduction_stage", "task")
        ):
            raise ValueError("full-finetune and baseline task coverage differs")
        full_summary = _object(
            _object(full_task["conditions"], "full-finetune task conditions")[
                FULL_FINETUNE_CONDITIONS[0]
            ],
            "full-finetune task condition",
        )
        independent = _object(
            _object(baseline_task["conditions"], "baseline task conditions")[
                "independent_root_lora"
            ],
            "independent task condition",
        )
        task_rows.append(
            (
                str(full_task["task"]),
                int(full_task["introduction_stage"]),
                FULL_FINETUNE_CONDITIONS[0],
                int(full_summary["best_stage"]),
                float(full_summary["introduction_story_mean_nll"]),
                float(full_summary["best_story_mean_nll"]),
                float(full_summary["final_story_mean_nll"]),
                float(full_summary["forgetting"]),
                float(full_summary["backward_transfer"]),
                float(independent["final_story_mean_nll"]),
                float(full_summary["final_story_mean_nll"])
                - float(independent["final_story_mean_nll"]),
            )
        )
    _write_csv(
        root / "full-finetune-stagewise-summary.csv",
        (
            "stage",
            "introduced_task",
            "story_count",
            "condition",
            "story_mean_nll",
            "token_mean_nll",
            "independent_story_mean_nll",
            "mean_deficit_vs_independent",
        ),
        tuple(stage_rows),
    )
    _write_csv(
        root / "full-finetune-stagewise-task-metrics.csv",
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
            "independent_final_story_mean_nll",
            "final_deficit_vs_independent",
        ),
        tuple(task_rows),
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
    "render_comparative_nll_chart_svg",
    "render_report_html",
    "render_report_markdown",
    "render_stagewise_route_chart_svg",
    "render_vamp_graph_svg",
]
