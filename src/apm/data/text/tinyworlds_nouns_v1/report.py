"""Artifact-only Markdown and interactive HTML reports for noun overlap."""

from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    CONDITIONS,
    NounBreakdown,
    NounPartitionArtifact,
    NounsExperimentPreset,
    canonical_json_bytes,
    record_sha256,
)


REPORT_FORMAT = "tinyworlds-nouns-report-v1"


def publish_noun_report(
    partition: NounPartitionArtifact,
    breakdown: NounBreakdown,
    preset: NounsExperimentPreset,
    adaptation: LanguageAdaptationArtifact,
    whole_story_path: str | Path,
    generation_path: str | Path,
    result_root: str | Path,
    *,
    judge_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Reconstruct and publish both reports solely from persisted artifacts."""
    whole_rows = _jsonl(Path(whole_story_path))
    generation_rows = _jsonl(Path(generation_path))
    judge_rows = (
        _jsonl(Path(judge_path))
        if judge_path is not None and Path(judge_path).is_file()
        else ()
    )
    _validate_result_coverage(partition, whole_rows, generation_rows)
    if judge_rows:
        generation_keys = {
            (str(row["task_noun"]), str(row["story_id"])) for row in generation_rows
        }
        judge_keys = {
            (str(row["task_noun"]), str(row["story_id"])) for row in judge_rows
        }
        if len(judge_rows) != len(judge_keys) or judge_keys != generation_keys:
            raise ValueError("noun judge rows do not cover the generation ledger")
    report_data = build_report_data(
        partition,
        breakdown,
        preset,
        adaptation,
        whole_rows,
        generation_rows,
        judge_rows,
    )
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    _publish_graph_artifacts(root, adaptation)
    _publish_confusion_csv(root, report_data)
    for source_name in ("base-selection.csv", "task-counts.csv"):
        destination = root / source_name
        if not destination.is_file():
            shutil.copy2(partition.root / source_name, destination)
    markdown = render_report_markdown(report_data)
    html = render_report_html(report_data)
    markdown_path = root / "report.md"
    html_path = root / "report.html"
    _atomic_write(markdown_path, markdown.encode("utf-8"))
    _atomic_write(html_path, html.encode("utf-8"))
    return markdown_path, html_path


def build_report_data(
    partition: NounPartitionArtifact,
    breakdown: NounBreakdown,
    preset: NounsExperimentPreset,
    adaptation: LanguageAdaptationArtifact,
    whole_rows: tuple[dict[str, object], ...],
    generation_rows: tuple[dict[str, object], ...],
    judge_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Build the complete JSON-compatible report view from immutable rows."""
    nll_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in whole_rows:
        nll_groups[(str(row["task_noun"]), str(row["condition"]))].append(row)
    task_metrics = []
    for task in partition.tasks:
        condition_rows = {
            condition: nll_groups[(task.task_id, condition)]
            for condition in CONDITIONS
        }
        base_mean = _mean(float(row["mean_nll"]) for row in condition_rows["base"])
        oracle_mean = _mean(float(row["mean_nll"]) for row in condition_rows["oracle"])
        task_metrics.append(
            {
                "base_overlap_fraction": task.base_overlap_story_count
                / task.train_story_count,
                "base_to_oracle_gain": base_mean - oracle_mean,
                "conditions": {
                    condition: _nll_summary(rows)
                    for condition, rows in condition_rows.items()
                },
                "task": task.task_id,
                "train_story_count": task.train_story_count,
                "validation_story_count": task.validation_story_count,
            }
        )
    overall_conditions = {
        condition: _nll_summary(
            [row for row in whole_rows if row["condition"] == condition]
        )
        for condition in CONDITIONS
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
    overlap = tuple(
        {
            "task": task.task_id,
            "base_count": task.base_overlap_story_count,
            "base_fraction": task.base_overlap_story_count / task.train_story_count,
            "other_tasks": dict(task.overlap_counts),
        }
        for task in partition.tasks
    )
    examples = _representative_examples(generation_rows)
    judge = _judge_summary(judge_rows)
    overlap_analysis = _overlap_analysis(task_metrics, partition)
    confusion_counts: dict[tuple[str, str, str], int] = defaultdict(int)
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
                for node_id in ("root", *partition.task_ids)
            }
            for task_id in partition.task_ids
        }
        for condition in CONDITIONS[2:]
    }
    core = {
        "base": {
            "concept_ids": list(partition.base_concept_ids),
            "story_coverage": breakdown.base_selection[-1].cumulative_story_coverage,
            "token_coverage": breakdown.base_selection[-1].cumulative_token_coverage,
            "train_story_count": partition.base_train_story_count,
            "validation_story_count": partition.base_validation_story_count,
        },
        "breakdown_sha256": breakdown.breakdown_sha256,
        "config_sha256": preset.config_sha256,
        "examples": list(examples),
        "format": REPORT_FORMAT,
        "graph": list(graph),
        "judge": judge,
        "overall_conditions": overall_conditions,
        "overlap": list(overlap),
        "overlap_analysis": overlap_analysis,
        "partition_sha256": partition.partition_sha256,
        "routing_confusion": confusion,
        "task_metrics": task_metrics,
    }
    return {**core, "report_sha256": record_sha256(core)}


def render_report_markdown(data: dict[str, object]) -> str:
    """Render the presentation report in concise, plain-language Markdown."""
    base = _record(data["base"])
    graph = tuple(_record(row) for row in _rows(data["graph"]))
    task_metrics = tuple(_record(row) for row in _rows(data["task_metrics"]))
    overlap = tuple(_record(row) for row in _rows(data["overlap"]))
    overall = _record(data["overall_conditions"])
    lines = [
        "# TinyWorlds noun-overlap experiment",
        "",
        "This run asks whether small adapters can learn overlapping kinds of stories, "
        "and whether the model can choose the right adapter from the story itself.",
        "",
        "## What went into the original model?",
        "",
        f"The base nouns were **{', '.join(str(value) for value in base['concept_ids'])}**. "
        f"Together they covered {float(base['story_coverage']):.1%} of unique training "
        f"stories and {float(base['token_coverage']):.1%} of their tokens.",
        "",
        "## What memory tree did training build?",
        "",
        *(
            f"- `{row['node']}` attached to "
            f"`{row['parent'] if row['parent'] is not None else 'none (root)'}` "
            f"at depth {row['depth']}."
            for row in graph
        ),
        "",
        "## Did each adapter learn its kind of story?",
        "",
        "A positive gain means the named adapter gave the held-out stories lower loss "
        "than the untouched base. Lower loss is better.",
        "",
        "Across all task/story pairs:",
        "",
        "| condition | story-weighted loss | token-weighted loss | perplexity | route accuracy | regret vs own adapter |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            _overall_markdown_row(condition, _record(overall[condition]))
            for condition in CONDITIONS
        ),
        "",
        "| noun | base loss | own-adapter loss | gain | base overlap |",
        "|---|---:|---:|---:|---:|",
        *(
            _task_markdown_row(row)
            for row in task_metrics
        ),
        "",
        "## Could the model find the adapter without being told the noun label?",
        "",
        "| noun | exhaustive | Hopfield | EBT uniform | EBT Hopfield |",
        "|---|---:|---:|---:|---:|",
        *(_routing_markdown_row(row) for row in task_metrics),
        "",
        "## How much did the noun datasets overlap?",
        "",
        "The base column counts task stories that the original model had already seen. "
        "The final column lists the largest shared-story counts with other adapters; "
        "one story may appear in several counts.",
        "",
        "| noun | also in base | base share | largest task overlaps |",
        "|---|---:|---:|---|",
        *(_overlap_markdown_row(row) for row in overlap),
        "",
        "## Story continuations",
        "",
        "The HTML report contains a deterministic mix of strong, weak, correctly routed, "
        "and misrouted prefixes, with the real ending and all six model continuations in "
        "folding sections. It highlights cases where routing "
        "found the noun's adapter and where it chose another branch.",
        "",
    ]
    judge = _record(data["judge"])
    if judge.get("available"):
        lines.extend(
            (
                "## External story-quality judge",
                "",
                "| continuation source | mean overall | mean rank | win vs base | win vs oracle | win vs reference |",
                "|---|---:|---:|---:|---:|---:|",
                *(
                    f"| {name} | {float(values['mean_overall']):.2f} | "
                    f"{float(values['mean_rank']):.2f} | "
                    f"{float(values['win_rate_vs_base']):.1%} | "
                    f"{float(values['win_rate_vs_oracle']):.1%} | "
                    f"{float(values['win_rate_vs_reference']):.1%} |"
                    for name, raw_values in _record(judge["conditions"]).items()
                    for values in (_record(raw_values),)
                ),
                "",
            )
        )
    else:
        lines.extend(
            (
                "## External story-quality judge",
                "",
                "No external judging result is present yet. The local losses and saved "
                "continuations are complete and can be judged on resume.",
                "",
            )
        )
    lines.extend(
        (
            "## How to read overlap",
            "",
            "Stories are deliberately allowed to belong to several noun tasks and to "
            "the base. The overlap table is descriptive: it can suggest transfer or "
            "blurred specialization, but it does not by itself prove that overlap caused "
            "either outcome.",
            "",
            _overlap_plain_language(_record(data["overlap_analysis"])),
            "",
            f"Artifact identity: `{data['report_sha256']}`",
            "",
        )
    )
    return "\n".join(lines)


def render_report_html(data: dict[str, object]) -> str:
    """Render a standalone interactive report with filters and folding examples."""
    task_metrics = tuple(_record(row) for row in _rows(data["task_metrics"]))
    examples = tuple(_record(row) for row in _rows(data["examples"]))
    graph = tuple(_record(row) for row in _rows(data["graph"]))
    base = _record(data["base"])
    overall = _record(data["overall_conditions"])
    task_options = "".join(
        f"<option value='{escape(str(row['task']))}'>{escape(str(row['task']))}</option>"
        for row in task_metrics
    )
    metric_rows = "".join(_task_html_row(row) for row in task_metrics)
    overall_rows = "".join(
        _overall_html_row(condition, _record(overall[condition]))
        for condition in CONDITIONS
    )
    graph_rows = "".join(
        "<li><b>"
        f"{escape(str(row['node']))}</b> → {escape(str(row['parent'] or 'root'))} "
        f"(depth {row['depth']})</li>"
        for row in graph[1:]
    )
    example_cards = "".join(_example_html(row) for row in examples)
    judge_html = _judge_html(_record(data["judge"]))
    confusion_html = _confusion_html(_record(data["routing_confusion"]))
    overlap_html = _overlap_html(tuple(_record(row) for row in _rows(data["overlap"])))
    embedded = escape(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds noun overlap</title><style>
:root{{--ink:#16202a;--muted:#5b6875;--line:#d8e0e8;--paper:#fff;--wash:#f4f7fa;--good:#176b4d;--bad:#a43b35;--accent:#315d9b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:2rem 1rem 5rem}}section,details.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:1.2rem;margin:1rem 0}}
h1{{font-size:2.2rem;margin-bottom:.3rem}}h2{{margin-top:.2rem}}.lede{{font-size:1.15rem;max-width:800px}}.chips{{display:flex;gap:.5rem;flex-wrap:wrap}}.chip{{background:#e8eef8;border-radius:999px;padding:.25rem .7rem}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.55rem;border-bottom:1px solid var(--line)}}th{{position:sticky;top:0;background:white}}.scroll{{overflow:auto}}
summary{{cursor:pointer;font-weight:650}}details details{{margin:.6rem 0;padding:.5rem;border-left:3px solid var(--line)}}pre{{white-space:pre-wrap;background:var(--wash);padding:.8rem;border-radius:8px}}
.good{{color:var(--good)}}.bad{{color:var(--bad)}}label{{font-weight:650}}select{{font:inherit;padding:.35rem}}.hidden{{display:none}}small{{color:var(--muted)}}
</style></head><body><main>
<h1>TinyWorlds noun-overlap experiment</h1><p class="lede">Can small memory adapters learn overlapping kinds of children's stories—and can the model pick the useful memory from the story alone?</p>
<div class="chips"><span class="chip">Base nouns: {escape(', '.join(str(value) for value in base['concept_ids']))}</span><span class="chip">Story coverage: {float(base['story_coverage']):.1%}</span><span class="chip">{len(task_metrics)} adapter tasks</span></div>
<section><h2>Experiment in one minute</h2><ol><li>Train a fresh language model on stories matching the base nouns.</li><li>Give each remaining noun its own LoRA memory. Stories can belong to several nouns.</li><li>Ask each memory to score unseen stories, then try four ways of choosing a memory without revealing the answer.</li><li>Give every method the first half of a story and inspect how it finishes.</li></ol></section>
<details class="card" open><summary>Which memories were learned, and where were they attached?</summary><ul>{graph_rows}</ul><p><small>After the first task, the root was scored but could not be selected. This prevents a trivial star while preserving the raw root comparison.</small></p></details>
<details class="card" open><summary>Loss and routing results</summary><p><b>Gain</b> is base loss minus own-adapter loss. Positive is better. Routing percentages show how often a method chose that story's named noun adapter.</p><h3>All task/story pairs</h3><div class="scroll"><table><thead><tr><th>Condition</th><th>Story-weighted loss</th><th>Token-weighted loss</th><th>Perplexity</th><th>Route accuracy</th><th>Regret</th></tr></thead><tbody>{overall_rows}</tbody></table></div><h3>By noun</h3><div class="scroll"><table><thead><tr><th>Noun</th><th>Base loss</th><th>Own adapter</th><th>Gain</th><th>Base overlap</th><th>Exhaustive route</th><th>Hopfield route</th><th>EBT uniform</th><th>EBT Hopfield</th></tr></thead><tbody>{metric_rows}</tbody></table></div></details>
	{confusion_html}
	{overlap_html}
	{judge_html}
<section><h2>Explore story examples</h2><p>These reproducibly selected cases mix strong and weak own-adapter gains with correct and incorrect routing. Fold open a story to compare the true ending with all six completions.</p><label for="task-filter">Show noun: </label><select id="task-filter"><option value="all">all nouns</option>{task_options}</select><div id="examples">{example_cards}</div></section>
<details class="card"><summary>What overlap can and cannot tell us</summary><p>Overlap is intentional here: the same story may train several noun memories, and a task story may already have appeared in base training. {escape(_overlap_plain_language(_record(data['overlap_analysis'])))} It is not a controlled causal test, because noun frequency, story difficulty, and overlap all vary together.</p></details>
<details class="card"><summary>Exact artifact identities</summary><pre>{embedded}</pre></details>
<script>const filter=document.getElementById('task-filter');filter.addEventListener('change',()=>{{document.querySelectorAll('[data-task]').forEach(card=>card.classList.toggle('hidden',filter.value!=='all'&&card.dataset.task!==filter.value));}});</script>
</main></body></html>"""


def _nll_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    total_nll = sum(float(row["total_nll"]) for row in rows)
    token_count = sum(int(row["token_count"]) for row in rows)
    story_mean = _mean(float(row["mean_nll"]) for row in rows)
    return {
        "mean_perplexity": _mean(float(row["perplexity"]) for row in rows),
        "mean_regret": _mean(float(row["regret_vs_oracle"]) for row in rows),
        "routing_accuracy": _mean(float(bool(row["oracle_match"])) for row in rows),
        "story_mean_nll": story_mean,
        "story_perplexity": math.exp(min(story_mean, 700.0)),
        "story_count": len(rows),
        "token_mean_nll": total_nll / token_count,
        "token_count": token_count,
    }


def _example_view(row: dict[str, object]) -> dict[str, object]:
    results = _record(row["results"])
    oracle = _record(results["oracle"])
    base = _record(results["base"])
    return {
        "base_to_oracle_gain": float(base["mean_nll"]) - float(oracle["mean_nll"]),
        "full_original_story": row["full_original_story"],
        "prefix": row["prefix"],
        "reference_continuation": row["reference_continuation"],
        "results": results,
        "story_id": row["story_id"],
        "task": row["task_noun"],
        "task_free_route_matches": sum(
            _record(results[condition])["selected_node"] == row["task_noun"]
            for condition in CONDITIONS[2:]
        ),
    }


def _representative_examples(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    views = tuple(_example_view(row) for row in rows)
    task_ids = tuple(sorted({str(view["task"]) for view in views}))
    selected: list[dict[str, object]] = []
    for task_id in task_ids:
        task_views = tuple(view for view in views if view["task"] == task_id)
        ranked_groups = (
            (
                "strong own-adapter gain",
                sorted(
                    task_views,
                    key=lambda view: (
                        -float(view["base_to_oracle_gain"]),
                        str(view["story_id"]),
                    ),
                )[:3],
            ),
            (
                "weak or negative own-adapter gain",
                sorted(
                    task_views,
                    key=lambda view: (
                        float(view["base_to_oracle_gain"]),
                        str(view["story_id"]),
                    ),
                )[:3],
            ),
            (
                "routing success",
                sorted(
                    task_views,
                    key=lambda view: (
                        -int(view["task_free_route_matches"]),
                        str(view["story_id"]),
                    ),
                )[:3],
            ),
            (
                "routing failure",
                sorted(
                    task_views,
                    key=lambda view: (
                        int(view["task_free_route_matches"]),
                        str(view["story_id"]),
                    ),
                )[:3],
            ),
        )
        candidates = tuple(
            (reason, view) for reason, group in ranked_groups for view in group
        )
        story_ids = tuple(
            dict.fromkeys(str(view["story_id"]) for _, view in candidates)
        )[:12]
        selected.extend(
            {
                **next(
                    view
                    for _, view in candidates
                    if str(view["story_id"]) == story_id
                ),
                "sample_reason": next(
                    reason
                    for reason, view in candidates
                    if str(view["story_id"]) == story_id
                ),
            }
            for story_id in story_ids
        )
    return tuple(selected)


def _judge_summary(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    if not rows:
        return {"available": False}
    cases = tuple(_judge_case(row) for row in rows)
    task_ids = tuple(sorted({task for task, _, _ in cases}))
    return {
        "available": True,
        "case_count": len(rows),
        "conditions": _summarize_judge_cases(cases),
        "tasks": {
            task_id: _summarize_judge_cases(
                tuple(case for case in cases if case[0] == task_id)
            )
            for task_id in task_ids
        },
    }


def _judge_case(
    row: dict[str, object],
) -> tuple[str, dict[str, float], tuple[str, ...]]:
    mapping = {
        label: source for label, source in _record(row["label_sources"]).items()
    }
    parsed = _record(row["parsed"])
    scores = {
        mapping[str(score["candidate"])]: float(score["overall"])
        for score in (_record(item) for item in _rows(parsed["scores"]))
    }
    ranking = tuple(mapping[str(label)] for label in _rows(parsed["ranking"]))
    return str(row["task_noun"]), scores, ranking


def _summarize_judge_cases(
    cases: tuple[tuple[str, dict[str, float], tuple[str, ...]], ...],
) -> dict[str, object]:
    return {
        source: {
            "mean_overall": _mean(scores[source] for _, scores, _ in cases),
            "mean_rank": _mean(
                float(ranking.index(source) + 1) for _, _, ranking in cases
            ),
            **{
                f"win_rate_vs_{comparator}": _mean(
                    1.0
                    if scores[source] > scores[comparator]
                    else 0.5
                    if scores[source] == scores[comparator]
                    else 0.0
                    for _, scores, _ in cases
                )
                for comparator in ("base", "oracle", "reference")
            },
        }
        for source in (*CONDITIONS, "reference")
    }


def _overlap_analysis(
    task_metrics: list[dict[str, object]],
    partition: NounPartitionArtifact,
) -> dict[str, object]:
    base_overlap = tuple(float(row["base_overlap_fraction"]) for row in task_metrics)
    base_loss = tuple(
        float(_record(_record(row["conditions"])["base"])["story_mean_nll"])
        for row in task_metrics
    )
    gains = tuple(float(row["base_to_oracle_gain"]) for row in task_metrics)
    exhaustive_accuracy = tuple(
        float(
            _record(_record(row["conditions"])["vamp_exhaustive"])[
                "routing_accuracy"
            ]
        )
        for row in task_metrics
    )
    cross_task_overlap = tuple(
        sum(count for noun, count in task.overlap_counts if noun != task.task_id)
        / max(1, (len(partition.tasks) - 1) * task.train_story_count)
        for task in partition.tasks
    )
    return {
        "base_overlap_vs_base_loss_correlation": _pearson(base_overlap, base_loss),
        "base_overlap_vs_adapter_gain_correlation": _pearson(base_overlap, gains),
        "cross_task_overlap_vs_exhaustive_accuracy_correlation": _pearson(
            cross_task_overlap,
            exhaustive_accuracy,
        ),
        "task_count": len(partition.tasks),
    }


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return None if denominator == 0.0 else numerator / denominator


def _overlap_plain_language(analysis: dict[str, object]) -> str:
    base_loss = analysis["base_overlap_vs_base_loss_correlation"]
    gain = analysis["base_overlap_vs_adapter_gain_correlation"]
    routing = analysis["cross_task_overlap_vs_exhaustive_accuracy_correlation"]
    if any(value is None for value in (base_loss, gain, routing)):
        return "There were too few varying noun tasks for an overlap trend summary."
    return (
        "Across nouns, base overlap versus untouched-base loss had correlation "
        f"{float(base_loss):+.2f}; base overlap versus adapter gain was "
        f"{float(gain):+.2f}; and cross-task overlap versus exhaustive routing "
        f"accuracy was {float(routing):+.2f}. These are descriptive associations, "
        "not evidence that overlap caused the differences."
    )


def _validate_result_coverage(
    partition: NounPartitionArtifact,
    whole_rows: tuple[dict[str, object], ...],
    generation_rows: tuple[dict[str, object], ...],
) -> None:
    expected_whole = sum(task.validation_story_count for task in partition.tasks) * len(CONDITIONS)
    expected_generation = sum(task.generation_story_count for task in partition.tasks)
    whole_keys = {
        (str(row.get("task_noun")), str(row.get("story_id")), str(row.get("condition")))
        for row in whole_rows
    }
    generation_keys = {
        (str(row.get("task_noun")), str(row.get("story_id")))
        for row in generation_rows
    }
    if (
        len(whole_rows) != expected_whole
        or len(whole_keys) != expected_whole
        or len(generation_rows) != expected_generation
        or len(generation_keys) != expected_generation
    ):
        raise ValueError("noun report inputs do not contain every required result row")
    task_by_id = {task.task_id: task for task in partition.tasks}
    if (
        any(task not in task_by_id or condition not in CONDITIONS for task, _, condition in whole_keys)
        or any(task not in task_by_id for task, _ in generation_keys)
        or any(
            sum(key[0] == task.task_id for key in generation_keys)
            != task.generation_story_count
            for task in partition.tasks
        )
        or any(
            sum(key[0] == task.task_id and key[2] == condition for key in whole_keys)
            != task.validation_story_count
            for task in partition.tasks
            for condition in CONDITIONS
        )
    ):
        raise ValueError("noun report result keys do not match partition tasks")


def _publish_graph_artifacts(root: Path, adaptation: LanguageAdaptationArtifact) -> None:
    graph = [
        {
            "depth": node.depth,
            "node_id": str(node.node_id),
            "parent_id": None if node.parent_id is None else str(node.parent_id),
            "stage": node.train_stage,
        }
        for node in adaptation.vamp_graph.nodes
    ]
    _atomic_write(root / "vamp-graph.json", canonical_json_bytes(graph))
    dot = ["digraph tinyworlds_nouns {", '  rankdir="LR";']
    dot.extend(
        f'  "{node.parent_id}" -> "{node.node_id}";'
        for node in adaptation.vamp_graph.nodes[1:]
    )
    dot.append("}")
    _atomic_write(root / "vamp-graph.dot", ("\n".join(dot) + "\n").encode("utf-8"))
    with tempfile.NamedTemporaryFile("w", delete=False, dir=root, newline="", encoding="utf-8") as output:
        temporary = Path(output.name)
        writer = csv.writer(output)
        writer.writerow(("stage", "task", "node", "eligible", "mean_prefix_nll", "selected"))
        for stage in adaptation.vamp_stages:
            for node_index, score in enumerate(stage.parent_mean_node_nll[: stage.stage_index]):
                writer.writerow(
                    (
                        stage.stage_index,
                        stage.task_id,
                        adaptation.vamp_graph.nodes[node_index].node_id,
                        node_index == 0 if stage.stage_index == 1 else node_index > 0,
                        score,
                        node_index == stage.parent_node_index,
                    )
                )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, root / "parent-scores.csv")


def _publish_confusion_csv(root: Path, data: dict[str, object]) -> None:
    confusion = _record(data["routing_confusion"])
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=root,
        newline="",
        encoding="utf-8",
    ) as output:
        temporary = Path(output.name)
        writer = csv.writer(output)
        writer.writerow(("condition", "oracle_task", "selected_node", "story_count"))
        writer.writerows(
            (condition, task_id, node_id, int(count))
            for condition, raw_matrix in confusion.items()
            for task_id, raw_counts in _record(raw_matrix).items()
            for node_id, count in _record(raw_counts).items()
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, root / "routing-confusion.csv")


def _overall_markdown_row(condition: str, summary: dict[str, object]) -> str:
    return (
        f"| {condition} | {float(summary['story_mean_nll']):.3f} | "
        f"{float(summary['token_mean_nll']):.3f} | "
        f"{float(summary['story_perplexity']):.2f} | "
        f"{float(summary['routing_accuracy']):.1%} | "
        f"{float(summary['mean_regret']):+.3f} |"
    )


def _task_markdown_row(row: dict[str, object]) -> str:
    conditions = _record(row["conditions"])
    base = _record(conditions["base"])
    oracle = _record(conditions["oracle"])
    gain = float(base["story_mean_nll"]) - float(oracle["story_mean_nll"])
    return (
        f"| {row['task']} | {float(base['story_mean_nll']):.3f} | "
        f"{float(oracle['story_mean_nll']):.3f} | {gain:+.3f} | "
        f"{float(row['base_overlap_fraction']):.1%} |"
    )


def _routing_markdown_row(row: dict[str, object]) -> str:
    conditions = _record(row["conditions"])
    values = tuple(
        float(_record(conditions[name])["routing_accuracy"])
        for name in CONDITIONS[2:]
    )
    return f"| {row['task']} | " + " | ".join(f"{value:.1%}" for value in values) + " |"


def _overlap_markdown_row(row: dict[str, object]) -> str:
    overlaps = _largest_other_overlaps(row)
    summary = ", ".join(f"{name}: {count:,}" for name, count in overlaps) or "none"
    return (
        f"| {row['task']} | {int(row['base_count']):,} | "
        f"{float(row['base_fraction']):.1%} | {summary} |"
    )


def _task_html_row(row: dict[str, object]) -> str:
    conditions = _record(row["conditions"])
    base = _record(conditions["base"])
    oracle = _record(conditions["oracle"])
    gain = float(base["story_mean_nll"]) - float(oracle["story_mean_nll"])
    routes = tuple(
        float(_record(conditions[name])["routing_accuracy"])
        for name in CONDITIONS[2:]
    )
    return (
        f"<tr><td>{escape(str(row['task']))}</td>"
        f"<td>{float(base['story_mean_nll']):.3f}</td>"
        f"<td>{float(oracle['story_mean_nll']):.3f}</td>"
        f"<td class={'good' if gain >= 0 else 'bad'}>{gain:+.3f}</td>"
        f"<td>{float(row['base_overlap_fraction']):.1%}</td>"
        + "".join(f"<td>{value:.1%}</td>" for value in routes)
        + "</tr>"
    )


def _overall_html_row(condition: str, summary: dict[str, object]) -> str:
    return (
        f"<tr><td>{escape(condition)}</td>"
        f"<td>{float(summary['story_mean_nll']):.3f}</td>"
        f"<td>{float(summary['token_mean_nll']):.3f}</td>"
        f"<td>{float(summary['story_perplexity']):.2f}</td>"
        f"<td>{float(summary['routing_accuracy']):.1%}</td>"
        f"<td>{float(summary['mean_regret']):+.3f}</td></tr>"
    )


def _confusion_html(confusion: dict[str, object]) -> str:
    sections = []
    for condition, raw_matrix in confusion.items():
        matrix = _record(raw_matrix)
        task_ids = tuple(matrix)
        node_ids = tuple(_record(matrix[task_ids[0]])) if task_ids else ()
        heading = "".join(f"<th>{escape(node_id)}</th>" for node_id in node_ids)
        rows = "".join(
            "<tr>"
            f"<th>{escape(task_id)}</th>"
            + "".join(
                f"<td>{int(_record(matrix[task_id])[node_id])}</td>"
                for node_id in node_ids
            )
            + "</tr>"
            for task_id in task_ids
        )
        sections.append(
            "<details><summary>"
            f"{escape(condition)}</summary><div class='scroll'><table><thead><tr>"
            f"<th>Named noun ↓ / selected node →</th>{heading}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div></details>"
        )
    return (
        "<details class='card'><summary>Routing confusion matrices</summary>"
        "<p>Rows are the noun whose validation set supplied the story; columns are "
        "the node selected without that label.</p>"
        + "".join(sections)
        + "</details>"
    )


def _overlap_html(rows: tuple[dict[str, object], ...]) -> str:
    cards = "".join(
        "<details><summary>"
        f"{escape(str(row['task']))}: {int(row['base_count']):,} stories also in base "
        f"({float(row['base_fraction']):.1%})</summary><ul>"
        + (
            "".join(
                f"<li>{escape(name)}: {count:,} shared training stories</li>"
                for name, count in _largest_other_overlaps(row, limit=None)
            )
            or "<li>No shared stories with another retained task.</li>"
        )
        + "</ul></details>"
        for row in rows
    )
    return (
        "<details class='card'><summary>Dataset overlap by noun</summary>"
        "<p>Each fold shows how much of that adapter's training set was already in "
        "the base and how many stories it shares with every other retained adapter. "
        "Membership is intentionally nonexclusive.</p>"
        f"{cards}</details>"
    )


def _largest_other_overlaps(
    row: dict[str, object],
    *,
    limit: int | None = 5,
) -> tuple[tuple[str, int], ...]:
    values = tuple(
        sorted(
            (
                (str(name), int(count))
                for name, count in _record(row["other_tasks"]).items()
                if name != row["task"] and int(count) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return values if limit is None else values[:limit]


def _example_html(row: dict[str, object]) -> str:
    results = _record(row["results"])
    condition_sections = "".join(
        "<details><summary>"
        f"{escape(condition)} — node {escape(str(result['selected_node']))}; "
        f"ending loss {float(result['mean_nll']):.3f}</summary>"
        f"<pre>{escape(str(result['generated_continuation']))}</pre></details>"
        for condition in CONDITIONS
        for result in (_record(results[condition]),)
    )
    return (
        f"<details class='card example' data-task='{escape(str(row['task']))}'>"
        f"<summary>{escape(str(row['task']))}: {escape(str(row['prefix']))[:180]}…</summary>"
        f"<p><b>Why shown:</b> {escape(str(row['sample_reason']))}; "
        f"{int(row['task_free_route_matches'])}/4 task-free routers chose the named node.</p>"
        f"<h3>Given first half</h3><pre>{escape(str(row['prefix']))}</pre>"
        f"<h3>Real ending</h3><pre>{escape(str(row['reference_continuation']))}</pre>"
        f"<h3>Model endings</h3>{condition_sections}</details>"
    )


def _judge_html(judge: dict[str, object]) -> str:
    if not judge.get("available"):
        return "<details class='card'><summary>External story judge</summary><p>Not run yet; saved local continuations are ready for judging on resume.</p></details>"
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{float(values['mean_overall']):.2f}</td>"
        f"<td>{float(values['mean_rank']):.2f}</td>"
        f"<td>{float(values['win_rate_vs_base']):.1%}</td>"
        f"<td>{float(values['win_rate_vs_oracle']):.1%}</td>"
        f"<td>{float(values['win_rate_vs_reference']):.1%}</td></tr>"
        for name, raw_values in _record(judge["conditions"]).items()
        for values in (_record(raw_values),)
    )
    return f"<details class='card'><summary>External story-quality judge</summary><div class='scroll'><table><tr><th>Source</th><th>Overall / 5</th><th>Mean rank</th><th>Win vs base</th><th>Win vs oracle</th><th>Win vs reference</th></tr>{rows}</table></div><details><summary>Per-noun judge summaries</summary><pre>{escape(json.dumps(judge.get('tasks', {}), ensure_ascii=False, indent=2))}</pre></details></details>"


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(
        _record(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _mean(values) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("report mean requires observations")
    return sum(materialized) / len(materialized)


def _record(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("report value must be an object")
    return value


def _rows(value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError("report value must be an array")
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
    "REPORT_FORMAT",
    "build_report_data",
    "publish_noun_report",
    "render_report_html",
    "render_report_markdown",
]
