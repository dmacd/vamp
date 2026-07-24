"""Markdown and self-contained HTML audits for semantic word catalogs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
import json
import math

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import (
    Role,
    SemanticCluster,
    SemanticConstructionConfig,
    SemanticWord,
)


@dataclass(frozen=True, slots=True)
class CatalogFailureWord:
    """One word's complete pre-clustering disposition in a failed semantic grid."""

    role: Role
    word: str
    token_mass: int
    context_count: int
    role_margin_q10: float | None
    context_silhouette: float | None
    exclusion_reason: str
    vector: tuple[float, ...] | None


def render_catalog_failure_audits(
    failure_sha256: str,
    evidence_sha256: str,
    reason: str,
    config: SemanticConstructionConfig,
    words: Sequence[CatalogFailureWord],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
) -> tuple[str, str]:
    """Render exhaustive Markdown and HTML evidence for a stopped semantic grid."""
    counts = {
        role: {
            disposition: sum(
                item.role == role and item.exclusion_reason == disposition
                for item in words
            )
            for disposition in sorted({item.exclusion_reason for item in words})
        }
        for role in ("noun", "verb")
    }
    rows = [
        "| Role | Word | Mass | Contexts | Role q10 | Silhouette | Disposition |",
        "|---|---|---:|---:|---:|---:|---|",
        *(
            f"| {item.role} | `{_markdown_escape(item.word)}` | {item.token_mass:,} | "
            f"{item.context_count} | {_metric(item.role_margin_q10)} | "
            f"{_metric(item.context_silhouette)} | {item.exclusion_reason} |"
            for item in words
        ),
    ]
    representatives = tuple(
        sorted(
            (item for item in words if item.vector is not None),
            key=lambda item: (item.role, -item.token_mass, item.word),
        )
    )
    context_lines = []
    for item in representatives[:40]:
        records = representative_contexts.get((item.role, item.word), ())
        if records:
            record = records[0]
            context_lines.append(
                f"- {item.role} `{item.word}` — “{str(record.get('sentence', '')).replace(chr(10), ' ')}” "
                f"(`{record.get('record_id', '')}`, story `{record.get('story_sha256', '')}`)"
            )
    markdown = "\n".join(
        (
            "# TinyWorlds-P Semantic v1 Construction Failure Audit",
            "",
            f"Failure SHA-256: `{failure_sha256}`  ",
            f"Encoder-evidence SHA-256: `{evidence_sha256}`",
            "",
            f"Automated stop: **{reason}**",
            "",
            "No clusters, cell-mass heatmap, partition, training run, or sealed-test opening exists for this failed grid.",
            "",
            "## Frozen configuration",
            "",
            "```json",
            json.dumps(config.as_record(), indent=2, sort_keys=True),
            "```",
            "",
            "## Pre-clustering disposition counts",
            "",
            "```json",
            json.dumps(counts, indent=2, sort_keys=True),
            "```",
            "",
            "## Candidate-vector PCA and cell-mass status",
            "",
            "Candidate-vector PCA plots are embedded in the self-contained HTML audit. A cluster cell-mass heatmap cannot be formed because the fixed eight-cluster gate failed.",
            "",
            "## Representative exact archive contexts",
            "",
            *(context_lines or ["No word passed the pre-clustering gates."]),
            "",
            "## All role words and exclusion reasons",
            "",
            *rows,
        )
    ).rstrip() + "\n"
    html_rows = "".join(
        f"<tr><td>{item.role}</td><td><code>{escape(item.word)}</code></td>"
        f"<td>{item.token_mass:,}</td><td>{item.context_count}</td>"
        f"<td>{_metric(item.role_margin_q10)}</td>"
        f"<td>{_metric(item.context_silhouette)}</td>"
        f"<td>{escape(item.exclusion_reason)}</td></tr>"
        for item in words
    )
    html_contexts = "".join(
        f"<li>{escape(line.removeprefix('- '))}</li>" for line in context_lines
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds-P Semantic v1 Construction Failure</title><style>body{{font:14px/1.45 system-ui,sans-serif;margin:2rem auto;max-width:1500px;padding:0 1rem;color:#172033}}code,pre{{font-family:ui-monospace,monospace}}pre{{background:#f4f6fa;padding:1rem;overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border:1px solid #ccd3df;padding:.35rem;text-align:left}}th{{background:#edf1f7;position:sticky;top:0}}.scroll{{overflow:auto;max-height:55rem}}.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:1rem}}svg{{border:1px solid #ccd3df;background:white;width:100%;height:auto}}.stop{{background:#fff1f0;border-left:5px solid #dc2626;padding:1rem}}</style></head><body><h1>TinyWorlds-P Semantic v1 Construction Failure Audit</h1><p>Failure <code>{failure_sha256}</code><br>Encoder evidence <code>{evidence_sha256}</code></p><p class="stop"><strong>Automated stop:</strong> {escape(reason)}<br>No partition, model training, or sealed-test opening is authorized.</p><h2>Frozen configuration</h2><pre>{escape(json.dumps(config.as_record(),indent=2,sort_keys=True))}</pre><h2>Pre-clustering disposition counts</h2><pre>{escape(json.dumps(counts,indent=2,sort_keys=True))}</pre><h2>Candidate-vector geometry</h2><div class="plots">{_failure_pca_svg(words, 'noun')}{_failure_pca_svg(words, 'verb')}<svg viewBox="0 0 600 400"><text x="20" y="35">Cell-mass heatmap unavailable: eight-cluster gate failed</text></svg></div><h2>Representative exact archive contexts</h2><ul>{html_contexts or '<li>No word passed the pre-clustering gates.</li>'}</ul><h2>All role words and exclusion reasons</h2><div class="scroll"><table><thead><tr><th>Role</th><th>Word</th><th>Mass</th><th>Contexts</th><th>Role q10</th><th>Silhouette</th><th>Disposition</th></tr></thead><tbody>{html_rows}</tbody></table></div></body></html>"""
    return markdown, html


def _failure_pca_svg(words: Sequence[CatalogFailureWord], role: Role) -> str:
    selected = tuple(item for item in words if item.role == role and item.vector is not None)
    if len(selected) < 2:
        return f'<svg viewBox="0 0 600 400"><text x="20" y="30">{role}: fewer than two candidate vectors</text></svg>'
    values = np.asarray([item.vector for item in selected], dtype=np.float64)
    centered = values - np.mean(values, axis=0)
    coordinates = np.linalg.svd(centered, full_matrices=False)[0][:, :2]
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    x, y = _scale(coordinates[:, 0], 35, 565), _scale(coordinates[:, 1], 365, 35)
    points = "".join(
        f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="3" fill="#2563eb"><title>{escape(item.word)}</title></circle>'
        for index, item in enumerate(selected)
    )
    return f'<svg viewBox="0 0 600 400"><text x="18" y="24">{role} pre-clustering candidates — PCA</text>{points}</svg>'


def render_catalog_audits(
    catalog_sha256: str,
    evidence_sha256: str,
    config: SemanticConstructionConfig,
    words: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
    pair_masses: Mapping[tuple[str, str], int],
    representative_contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
    parent_words: Sequence[SemanticWord] | None = None,
) -> tuple[str, str]:
    """Render exhaustive Markdown and standalone HTML from one frozen catalog."""
    retained = tuple(item for item in words if item.exclusion_reason is None)
    excluded = tuple(item for item in words if item.exclusion_reason is not None)
    summaries = _cluster_summaries(retained, clusters)
    heatmap = _cell_masses(retained, pair_masses, config.cluster_count)
    differences = _parent_differences(words, parent_words)
    markdown = _markdown(
        catalog_sha256,
        evidence_sha256,
        config,
        retained,
        excluded,
        clusters,
        summaries,
        heatmap,
        representative_contexts,
        differences,
    )
    html = _html(
        catalog_sha256,
        evidence_sha256,
        config,
        retained,
        excluded,
        clusters,
        summaries,
        heatmap,
        representative_contexts,
        differences,
    )
    return markdown, html


def _cluster_summaries(
    retained: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
) -> dict[tuple[Role, int], dict[str, tuple[str, ...] | str]]:
    by_key = {(cluster.role, cluster.index): cluster for cluster in clusters}
    grouped: dict[tuple[Role, int], list[tuple[float, str]]] = defaultdict(list)
    for word in retained:
        assert word.cluster is not None and word.vector is not None
        centroid = np.asarray(by_key[(word.role, word.cluster)].centroid)
        grouped[(word.role, word.cluster)].append(
            (float(centroid @ np.asarray(word.vector)), word.word)
        )
    result: dict[tuple[Role, int], dict[str, tuple[str, ...] | str]] = {}
    for key, scored in grouped.items():
        ordered = tuple(word for _, word in sorted(scored, key=lambda item: (-item[0], item[1])))
        boundary = tuple(word for _, word in sorted(scored, key=lambda item: (item[0], item[1]))[:5])
        result[key] = {
            "nearest": ordered[:5],
            "median": ordered[len(ordered) // 2],
            "boundary": boundary,
        }
    return result


def _cell_masses(
    retained: Sequence[SemanticWord],
    pair_masses: Mapping[tuple[str, str], int],
    count: int,
) -> np.ndarray:
    noun_clusters = {
        word.word: word.cluster
        for word in retained
        if word.role == "noun"
    }
    verb_clusters = {
        word.word: word.cluster
        for word in retained
        if word.role == "verb"
    }
    matrix = np.zeros((count, count), dtype=np.int64)
    for (noun, verb), mass in pair_masses.items():
        noun_cluster, verb_cluster = noun_clusters.get(noun), verb_clusters.get(verb)
        if noun_cluster is not None and verb_cluster is not None:
            matrix[noun_cluster, verb_cluster] += mass
    return matrix


def _parent_differences(
    words: Sequence[SemanticWord],
    parent: Sequence[SemanticWord] | None,
) -> tuple[str, ...]:
    if parent is None:
        return ("No parent catalog was supplied.",)
    before = {(item.role, item.word): item for item in parent}
    after = {(item.role, item.word): item for item in words}
    messages = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old is None:
            messages.append(f"added {key[0]} `{key[1]}`")
        elif new is None:
            messages.append(f"removed {key[0]} `{key[1]}`")
        elif (old.exclusion_reason, old.cluster) != (new.exclusion_reason, new.cluster):
            messages.append(
                f"changed {key[0]} `{key[1]}`: "
                f"{old.exclusion_reason or f'cluster {old.cluster}'} → "
                f"{new.exclusion_reason or f'cluster {new.cluster}'}"
            )
    return tuple(messages) or ("No word dispositions differ from the parent.",)


def _markdown(
    catalog_sha: str,
    evidence_sha: str,
    config: SemanticConstructionConfig,
    retained: Sequence[SemanticWord],
    excluded: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
    summaries: Mapping[tuple[Role, int], Mapping[str, tuple[str, ...] | str]],
    heatmap: np.ndarray,
    contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
    differences: Sequence[str],
) -> str:
    lines = [
        "# TinyWorlds-P Semantic v1 Catalog Audit",
        "",
        f"Catalog SHA-256: `{catalog_sha}`  ",
        f"Encoder-evidence SHA-256: `{evidence_sha}`",
        "",
        "The labels below are canonical numeric identifiers, not human-authored cluster names.",
        "",
        "## Frozen configuration",
        "",
        "```json",
        json.dumps(config.as_record(), indent=2, sort_keys=True),
        "```",
        "",
        "## Cluster mass and boundary audit",
        "",
        "| Role | Cluster | Words | Token mass | Nearest | Median | Boundary |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for cluster in clusters:
        summary = summaries[(cluster.role, cluster.index)]
        lines.append(
            "| "
            + " | ".join(
                (
                    cluster.role,
                    str(cluster.index),
                    str(len(cluster.words)),
                    f"{cluster.token_mass:,}",
                    ", ".join(f"`{word}`" for word in summary["nearest"]),
                    f"`{summary['median']}`",
                    ", ".join(f"`{word}`" for word in summary["boundary"]),
                )
            )
            + " |"
        )
    lines.extend(("", "## Retained cell token mass", "", _markdown_heatmap(heatmap)))
    lines.extend(("", "## Representative exact archive contexts", ""))
    for cluster in clusters:
        summary = summaries[(cluster.role, cluster.index)]
        representative_words = tuple(summary["nearest"][:1]) + (str(summary["median"]),) + tuple(
            summary["boundary"][:1]
        )
        lines.extend((f"### {cluster.role} cluster {cluster.index}", ""))
        for word in dict.fromkeys(representative_words):
            records = contexts.get((cluster.role, word), ())
            if not records:
                lines.append(f"- `{word}` — no selected exact context")
                continue
            record = records[0]
            sentence = str(record.get("sentence", "")).replace("\n", " ")
            lines.append(
                f"- `{word}` — “{sentence}” "
                f"(`{record.get('record_id', '')}`, story `{record.get('story_sha256', '')}`)"
            )
        lines.append("")
    lines.extend(("## All retained words", "", _markdown_word_table(retained)))
    lines.extend(("", "## All excluded words", "", _markdown_word_table(excluded)))
    lines.extend(("", "## Differences from optional parent", ""))
    lines.extend(f"- {message}" for message in differences)
    return "\n".join(lines).rstrip() + "\n"


def _markdown_word_table(words: Sequence[SemanticWord]) -> str:
    lines = [
        "| Role | Word | Mass | Contexts | Role q10 | Silhouette | Cluster margin | Disposition |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for word in words:
        disposition = (
            f"cluster {word.cluster}" if word.exclusion_reason is None else word.exclusion_reason
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    word.role,
                    f"`{_markdown_escape(word.word)}`",
                    f"{word.token_mass:,}",
                    str(word.context_count),
                    _metric(word.role_margin_q10),
                    _metric(word.context_silhouette),
                    _metric(word.cluster_margin),
                    str(disposition),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _markdown_heatmap(matrix: np.ndarray) -> str:
    header = "| N\\V | " + " | ".join(str(index) for index in range(matrix.shape[1])) + " |"
    divider = "|---|" + "---:|" * matrix.shape[1]
    rows = [
        f"| {row} | " + " | ".join(f"{int(value):,}" for value in matrix[row]) + " |"
        for row in range(matrix.shape[0])
    ]
    return "\n".join((header, divider, *rows))


def _html(
    catalog_sha: str,
    evidence_sha: str,
    config: SemanticConstructionConfig,
    retained: Sequence[SemanticWord],
    excluded: Sequence[SemanticWord],
    clusters: Sequence[SemanticCluster],
    summaries: Mapping[tuple[Role, int], Mapping[str, tuple[str, ...] | str]],
    heatmap: np.ndarray,
    contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
    differences: Sequence[str],
) -> str:
    cluster_rows = "".join(
        "<tr>"
        f"<td>{cluster.role}</td><td>{cluster.index}</td><td>{len(cluster.words)}</td>"
        f"<td>{cluster.token_mass:,}</td>"
        f"<td>{escape(', '.join(summaries[(cluster.role, cluster.index)]['nearest']))}</td>"
        f"<td>{escape(str(summaries[(cluster.role, cluster.index)]['median']))}</td>"
        f"<td>{escape(', '.join(summaries[(cluster.role, cluster.index)]['boundary']))}</td>"
        "</tr>"
        for cluster in clusters
    )
    context_blocks = []
    for cluster in clusters:
        summary = summaries[(cluster.role, cluster.index)]
        selected = tuple(summary["nearest"][:1]) + (str(summary["median"]),) + tuple(
            summary["boundary"][:1]
        )
        items = []
        for word in dict.fromkeys(selected):
            records = contexts.get((cluster.role, word), ())
            if records:
                record = records[0]
                items.append(
                    f"<li><code>{escape(word)}</code> — “{escape(str(record.get('sentence', '')))}” "
                    f"<small>{escape(str(record.get('record_id', '')))}; "
                    f"story {escape(str(record.get('story_sha256', '')))}</small></li>"
                )
            else:
                items.append(f"<li><code>{escape(word)}</code> — no selected context</li>")
        context_blocks.append(
            f"<h3>{cluster.role} cluster {cluster.index}</h3><ul>{''.join(items)}</ul>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds-P Semantic v1 Catalog Audit</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:2rem auto;max-width:1500px;padding:0 1rem;color:#172033}}
code,pre{{font-family:ui-monospace,monospace}} pre{{background:#f4f6fa;padding:1rem;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border:1px solid #ccd3df;padding:.35rem;text-align:left;vertical-align:top}}
th{{background:#edf1f7;position:sticky;top:0}} .scroll{{overflow:auto;max-height:48rem;margin-bottom:2rem}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:1rem}} svg{{border:1px solid #ccd3df;background:white;width:100%;height:auto}}
.retained{{background:#effaf1}} .excluded{{background:#fff1f0}} small{{color:#586174}}
</style></head><body>
<h1>TinyWorlds-P Semantic v1 Catalog Audit</h1>
<p>Catalog <code>{escape(catalog_sha)}</code><br>Encoder evidence <code>{escape(evidence_sha)}</code></p>
<p>Clusters use canonical numeric identifiers only; no human or model-authored names affect construction.</p>
<h2>Frozen configuration</h2><pre>{escape(json.dumps(config.as_record(), indent=2, sort_keys=True))}</pre>
<h2>Semantic geometry</h2><div class="plots">{_pca_svg(retained, 'noun')}{_pca_svg(retained, 'verb')}{_heatmap_svg(heatmap)}</div>
<h2>Cluster mass and boundary audit</h2><div class="scroll"><table><thead><tr><th>Role</th><th>Cluster</th><th>Words</th><th>Token mass</th><th>Nearest</th><th>Median</th><th>Boundary</th></tr></thead><tbody>{cluster_rows}</tbody></table></div>
<h2>Retained cell token mass</h2>{_heatmap_table(heatmap)}
<h2>Representative exact archive contexts</h2>{''.join(context_blocks)}
<h2>All retained words</h2><div class="scroll">{_word_table(retained, 'retained')}</div>
<h2>All excluded words</h2><div class="scroll">{_word_table(excluded, 'excluded')}</div>
<h2>Differences from optional parent</h2><ul>{''.join(f'<li>{escape(message)}</li>' for message in differences)}</ul>
</body></html>"""


def _word_table(words: Sequence[SemanticWord], css_class: str) -> str:
    rows = "".join(
        f"<tr class={css_class}><td>{word.role}</td><td><code>{escape(word.word)}</code></td>"
        f"<td>{word.token_mass:,}</td><td>{word.context_count}</td>"
        f"<td>{_metric(word.role_margin_q10)}</td><td>{_metric(word.context_silhouette)}</td>"
        f"<td>{_metric(word.cluster_margin)}</td>"
        f"<td>{escape(str(word.exclusion_reason or f'cluster {word.cluster}'))}</td></tr>"
        for word in words
    )
    return (
        "<table><thead><tr><th>Role</th><th>Word</th><th>Mass</th><th>Contexts</th>"
        "<th>Role q10</th><th>Silhouette</th><th>Cluster margin</th><th>Disposition</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _pca_svg(words: Sequence[SemanticWord], role: Role) -> str:
    selected = tuple(word for word in words if word.role == role and word.vector is not None)
    if len(selected) < 2:
        return f'<svg viewBox="0 0 600 400"><text x="20" y="30">{role}: insufficient PCA points</text></svg>'
    values = np.asarray([word.vector for word in selected], dtype=np.float64)
    centered = values - np.mean(values, axis=0)
    coordinates = np.linalg.svd(centered, full_matrices=False)[0][:, :2]
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    x, y = _scale(coordinates[:, 0], 35, 565), _scale(coordinates[:, 1], 365, 35)
    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#65a30d")
    points = "".join(
        f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="2.4" fill="{palette[int(word.cluster) % len(palette)]}"><title>{escape(word.word)}; cluster {word.cluster}</title></circle>'
        for index, word in enumerate(selected)
    )
    return f'<svg viewBox="0 0 600 400" role="img"><text x="18" y="24">{role} word vectors — PCA</text>{points}</svg>'


def _heatmap_svg(matrix: np.ndarray) -> str:
    count = matrix.shape[0]
    maximum = max(1, int(np.max(matrix)))
    size = 320 / count
    cells = "".join(
        f'<rect x="{70 + column * size:.2f}" y="{45 + row * size:.2f}" width="{size:.2f}" height="{size:.2f}" fill="rgb({245 - int(190 * matrix[row, column] / maximum)}, {248 - int(130 * matrix[row, column] / maximum)}, 255)"><title>N{row}×V{column}: {int(matrix[row, column]):,} tokens</title></rect>'
        for row in range(count)
        for column in range(count)
    )
    return f'<svg viewBox="0 0 430 400" role="img"><text x="18" y="24">Retained noun × verb token mass</text>{cells}</svg>'


def _heatmap_table(matrix: np.ndarray) -> str:
    header = "".join(f"<th>V{column}</th>" for column in range(matrix.shape[1]))
    rows = "".join(
        f"<tr><th>N{row}</th>{''.join(f'<td>{int(value):,}</td>' for value in matrix[row])}</tr>"
        for row in range(matrix.shape[0])
    )
    return f"<div class=scroll><table><thead><tr><th>N\\V</th>{header}</tr></thead><tbody>{rows}</tbody></table></div>"


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if math.isclose(minimum, maximum):
        return np.full_like(values, (low + high) / 2)
    return low + (values - minimum) * (high - low) / (maximum - minimum)


def _metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`")


__all__ = [
    "CatalogFailureWord",
    "render_catalog_audits",
    "render_catalog_failure_audits",
]
