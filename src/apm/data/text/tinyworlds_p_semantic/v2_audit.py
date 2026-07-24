"""Markdown and standalone HTML audits for calibrated semantic construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
import json
from typing import Protocol

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import Role, SemanticCluster
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    RoleCalibrationReference,
)


class AuditConfig(Protocol):
    """Configuration surface required by the calibrated audit renderer."""

    cluster_count: int
    minimum_cluster_margin: float

    def as_record(self) -> dict[str, object]: ...


class AuditWord(Protocol):
    """Word surface required by the calibrated audit renderer."""

    role: Role
    word: str
    token_mass: int
    calibration_fold: int | None
    role_margin_q10: float | None
    role_conformal_p: float | None
    context_silhouette: float | None
    cluster_margin: float | None
    cluster: int | None
    exclusion_reason: str | None
    vector: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class CalibratedFailureWord:
    """One word's complete disposition when a calibrated grid stops."""

    role: Role
    word: str
    token_mass: int
    context_count: int
    calibration_fold: int | None
    calibration_reference_count: int | None
    role_margin_q10: float | None
    role_conformal_p: float | None
    role_rejection_cutoff: float | None
    context_silhouette: float | None
    disposition: str
    vector: tuple[float, ...] | None

    def as_record(self) -> dict[str, object]:
        return {
            "calibration_fold": self.calibration_fold,
            "calibration_reference_count": self.calibration_reference_count,
            "context_count": self.context_count,
            "context_silhouette": self.context_silhouette,
            "disposition": self.disposition,
            "role": self.role,
            "role_conformal_p": self.role_conformal_p,
            "role_margin_q10": self.role_margin_q10,
            "role_rejection_cutoff": self.role_rejection_cutoff,
            "token_mass": self.token_mass,
            "vector": None if self.vector is None else list(self.vector),
            "word": self.word,
        }


def render_calibrated_failure_audits(
    failure_sha256: str,
    evidence_sha256: str,
    reason: str,
    config: AuditConfig,
    calibration: Sequence[RoleCalibrationReference],
    boundary_trace: Sequence[BoundaryPassMetric],
    words: Sequence[CalibratedFailureWord],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
    benchmark_version: str,
) -> tuple[str, str]:
    """Render exhaustive failure evidence without implying a valid catalog."""
    counts = {
        role: dict(
            sorted(
                Counter(
                    item.disposition for item in words if item.role == role
                ).items()
            )
        )
        for role in ("noun", "verb")
    }
    calibration_rows = _calibration_markdown(calibration)
    boundary_rows = _boundary_markdown(
        boundary_trace,
        config.minimum_cluster_margin,
    )
    word_rows = _failure_word_markdown(words)
    context_lines = _representative_context_lines(words, representative_contexts)
    markdown = "\n".join(
        (
            f"# TinyWorlds-P Semantic {benchmark_version} Construction Failure Audit",
            "",
            f"Failure SHA-256: `{failure_sha256}`  ",
            f"Reused encoder-evidence SHA-256: `{evidence_sha256}`",
            "",
            f"Automated stop: **{reason}**",
            "",
            f"No semantic-{benchmark_version} catalog, partition, training run, or sealed-test opening exists for this failed grid.",
            "",
            "## Frozen configuration",
            "",
            "```json",
            json.dumps(config.as_record(), indent=2, sort_keys=True),
            "```",
            "",
            "## Cross-conformal role calibration",
            "",
            calibration_rows,
            "",
            "## Pre-clustering disposition counts",
            "",
            "```json",
            json.dumps(counts, indent=2, sort_keys=True),
            "```",
            "",
            "## Boundary-exclusion trace",
            "",
            boundary_rows or "No clustering pass completed.",
            "",
            "## Candidate geometry and cell-mass status",
            "",
            "Candidate PCA plots are embedded in the self-contained HTML audit. A cell-mass heatmap is unavailable because the fixed grid failed.",
            "",
            "## Representative exact archive contexts",
            "",
            *(context_lines or ("No word passed the calibrated pre-clustering screens.",)),
            "",
            "## All role words and exclusion reasons",
            "",
            word_rows,
        )
    ).rstrip() + "\n"

    html_rows = "".join(_failure_word_html(item) for item in words)
    html_contexts = "".join(
        f"<li>{escape(line.removeprefix('- '))}</li>" for line in context_lines
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds-P Semantic {benchmark_version} Construction Failure</title>{_style()}</head><body><h1>TinyWorlds-P Semantic {benchmark_version} Construction Failure Audit</h1><p>Failure <code>{failure_sha256}</code><br>Reused encoder evidence <code>{evidence_sha256}</code></p><p class="stop"><strong>Automated stop:</strong> {escape(reason)}<br>No catalog, partition, model training, or sealed-test opening is authorized.</p><h2>Frozen configuration</h2><pre>{escape(json.dumps(config.as_record(), indent=2, sort_keys=True))}</pre><h2>Cross-conformal role calibration</h2>{_calibration_html(calibration)}<h2>Pre-clustering disposition counts</h2><pre>{escape(json.dumps(counts, indent=2, sort_keys=True))}</pre><h2>Boundary-exclusion trace</h2>{_boundary_html(boundary_trace, config.minimum_cluster_margin)}<h2>Candidate geometry</h2><div class="plots">{_pca_svg(words, 'noun')}{_pca_svg(words, 'verb')}<svg viewBox="0 0 600 400"><text x="20" y="35">Cell-mass heatmap unavailable: fixed grid failed</text></svg></div><h2>Representative exact archive contexts</h2><ul>{html_contexts or '<li>No calibrated candidates.</li>'}</ul><h2>All role words and exclusion reasons</h2><div class="scroll"><table><thead><tr><th>Role</th><th>Word</th><th>Mass</th><th>Contexts</th><th>Fold</th><th>Reference n</th><th>Raw q10</th><th>Conformal p</th><th>Cutoff</th><th>Silhouette</th><th>Disposition</th></tr></thead><tbody>{html_rows}</tbody></table></div></body></html>"""
    return markdown, html


def render_calibrated_catalog_audits(
    catalog_sha256: str,
    evidence_sha256: str,
    config: AuditConfig,
    calibration: Sequence[RoleCalibrationReference],
    boundary_trace: Sequence[BoundaryPassMetric],
    words: Sequence[AuditWord],
    clusters: Sequence[SemanticCluster],
    pair_masses: Mapping[tuple[str, str], int],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
    parent_words: Sequence[AuditWord] | None,
    benchmark_version: str,
) -> tuple[str, str]:
    """Render all success metrics, inventories, examples, and parent differences."""
    retained = tuple(item for item in words if item.exclusion_reason is None)
    excluded = tuple(item for item in words if item.exclusion_reason is not None)
    summaries = _cluster_summaries(retained, clusters)
    heatmap = _cell_masses(retained, pair_masses, config.cluster_count)
    differences = _parent_differences(words, parent_words)
    cluster_rows = [
        "| Role | Cluster | Words | Token mass | Nearest | Median | Boundary |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for cluster in clusters:
        summary = summaries[(cluster.role, cluster.index)]
        cluster_rows.append(
            f"| {cluster.role} | {cluster.index} | {len(cluster.words)} | "
            f"{cluster.token_mass:,} | "
            f"{', '.join(f'`{word}`' for word in summary['nearest'])} | "
            f"`{summary['median']}` | "
            f"{', '.join(f'`{word}`' for word in summary['boundary'])} |"
        )
    context_lines = _cluster_context_lines(
        clusters,
        summaries,
        representative_contexts,
    )
    markdown = "\n".join(
        (
            f"# TinyWorlds-P Semantic {benchmark_version} Catalog Audit",
            "",
            f"Catalog SHA-256: `{catalog_sha256}`  ",
            f"Reused encoder-evidence SHA-256: `{evidence_sha256}`",
            "",
            "Clusters have canonical numeric identifiers; no generated names or human labels affect construction.",
            "",
            "## Frozen configuration",
            "",
            "```json",
            json.dumps(config.as_record(), indent=2, sort_keys=True),
            "```",
            "",
            "## Cross-conformal role calibration",
            "",
            _calibration_markdown(calibration),
            "",
            "## Boundary-exclusion trace",
            "",
            _boundary_markdown(boundary_trace, config.minimum_cluster_margin),
            "",
            "## Cluster mass and semantic inventory",
            "",
            *cluster_rows,
            "",
            "## Retained cell token mass",
            "",
            _heatmap_markdown(heatmap),
            "",
            "## Representative exact archive contexts",
            "",
            *context_lines,
            "",
            "## All retained words",
            "",
            _semantic_word_markdown(retained),
            "",
            "## All excluded words",
            "",
            _semantic_word_markdown(excluded),
            "",
            "## Differences from optional parent",
            "",
            *(f"- {item}" for item in differences),
        )
    ).rstrip() + "\n"
    cluster_html = "".join(
        f"<tr><td>{item.role}</td><td>{item.index}</td><td>{len(item.words)}</td>"
        f"<td>{item.token_mass:,}</td><td>{escape(', '.join(summaries[(item.role, item.index)]['nearest']))}</td>"
        f"<td>{escape(str(summaries[(item.role, item.index)]['median']))}</td>"
        f"<td>{escape(', '.join(summaries[(item.role, item.index)]['boundary']))}</td></tr>"
        for item in clusters
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds-P Semantic {benchmark_version} Catalog</title>{_style()}</head><body><h1>TinyWorlds-P Semantic {benchmark_version} Catalog Audit</h1><p>Catalog <code>{catalog_sha256}</code><br>Reused encoder evidence <code>{evidence_sha256}</code></p><h2>Frozen configuration</h2><pre>{escape(json.dumps(config.as_record(), indent=2, sort_keys=True))}</pre><h2>Cross-conformal role calibration</h2>{_calibration_html(calibration)}<h2>Boundary-exclusion trace</h2>{_boundary_html(boundary_trace, config.minimum_cluster_margin)}<h2>Semantic geometry</h2><div class="plots">{_pca_svg(retained, 'noun')}{_pca_svg(retained, 'verb')}{_heatmap_svg(heatmap)}</div><h2>Clusters</h2><table><thead><tr><th>Role</th><th>Cluster</th><th>Words</th><th>Mass</th><th>Nearest</th><th>Median</th><th>Boundary</th></tr></thead><tbody>{cluster_html}</tbody></table><h2>Representative exact archive contexts</h2><pre>{escape(chr(10).join(context_lines))}</pre><h2>All retained words</h2>{_semantic_word_html(retained)}<h2>All excluded words</h2>{_semantic_word_html(excluded)}<h2>Parent differences</h2><ul>{''.join(f'<li>{escape(item)}</li>' for item in differences)}</ul></body></html>"""
    return markdown, html


def _calibration_markdown(items: Sequence[RoleCalibrationReference]) -> str:
    lines = [
        "| Role | Held-out fold | Reference words | Rejection cutoff |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item.role} | {item.fold} | {item.reference_count} | "
        f"{_metric(item.rejection_cutoff)} |"
        for item in items
    )
    return "\n".join(lines)


def _calibration_html(items: Sequence[RoleCalibrationReference]) -> str:
    rows = "".join(
        f"<tr><td>{item.role}</td><td>{item.fold}</td><td>{item.reference_count}</td>"
        f"<td>{_metric(item.rejection_cutoff)}</td></tr>"
        for item in items
    )
    return f"<table><thead><tr><th>Role</th><th>Held-out fold</th><th>Reference words</th><th>Rejection cutoff</th></tr></thead><tbody>{rows}</tbody></table>"


def _boundary_markdown(
    items: Sequence[BoundaryPassMetric],
    minimum_margin: float,
) -> str:
    if not items:
        return ""
    lines = [
        f"| Role | Pass | Input words | Below {minimum_margin:g} | Cluster masses | Min margin | q10 | Median |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item.role} | {item.pass_index} | {item.input_word_count} | "
        f"{item.failing_word_count} | {', '.join(f'{mass:,}' for mass in item.cluster_masses)} | "
        f"{item.minimum_margin:.6f} | {item.margin_q10:.6f} | {item.median_margin:.6f} |"
        for item in items
    )
    return "\n".join(lines)


def _boundary_html(
    items: Sequence[BoundaryPassMetric],
    minimum_margin: float,
) -> str:
    if not items:
        return "<p>No clustering pass completed.</p>"
    rows = "".join(
        f"<tr><td>{item.role}</td><td>{item.pass_index}</td><td>{item.input_word_count}</td>"
        f"<td>{item.failing_word_count}</td><td>{escape(', '.join(f'{mass:,}' for mass in item.cluster_masses))}</td>"
        f"<td>{item.minimum_margin:.6f}</td><td>{item.margin_q10:.6f}</td><td>{item.median_margin:.6f}</td></tr>"
        for item in items
    )
    return f"<table><thead><tr><th>Role</th><th>Pass</th><th>Input words</th><th>Below {minimum_margin:g}</th><th>Cluster masses</th><th>Min</th><th>q10</th><th>Median</th></tr></thead><tbody>{rows}</tbody></table>"


def _failure_word_markdown(words: Sequence[CalibratedFailureWord]) -> str:
    lines = [
        "| Role | Word | Mass | Contexts | Fold | Reference n | Raw q10 | Conformal p | Cutoff | Silhouette | Disposition |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {item.role} | `{_markdown_escape(item.word)}` | {item.token_mass:,} | "
        f"{item.context_count} | {_optional_integer(item.calibration_fold)} | "
        f"{_optional_integer(item.calibration_reference_count)} | {_metric(item.role_margin_q10)} | "
        f"{_metric(item.role_conformal_p)} | {_metric(item.role_rejection_cutoff)} | "
        f"{_metric(item.context_silhouette)} | {item.disposition} |"
        for item in words
    )
    return "\n".join(lines)


def _failure_word_html(item: CalibratedFailureWord) -> str:
    return (
        f"<tr><td>{item.role}</td><td><code>{escape(item.word)}</code></td>"
        f"<td>{item.token_mass:,}</td><td>{item.context_count}</td>"
        f"<td>{_optional_integer(item.calibration_fold)}</td>"
        f"<td>{_optional_integer(item.calibration_reference_count)}</td>"
        f"<td>{_metric(item.role_margin_q10)}</td><td>{_metric(item.role_conformal_p)}</td>"
        f"<td>{_metric(item.role_rejection_cutoff)}</td>"
        f"<td>{_metric(item.context_silhouette)}</td><td>{escape(item.disposition)}</td></tr>"
    )


def _semantic_word_markdown(words: Sequence[AuditWord]) -> str:
    lines = [
        "| Role | Word | Mass | Fold | Raw q10 | Conformal p | Silhouette | Cluster margin | Disposition |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {item.role} | `{_markdown_escape(item.word)}` | {item.token_mass:,} | "
        f"{_optional_integer(item.calibration_fold)} | {_metric(item.role_margin_q10)} | "
        f"{_metric(item.role_conformal_p)} | {_metric(item.context_silhouette)} | "
        f"{_metric(item.cluster_margin)} | "
        f"{item.exclusion_reason or f'cluster {item.cluster}'} |"
        for item in words
    )
    return "\n".join(lines)


def _semantic_word_html(words: Sequence[AuditWord]) -> str:
    rows = "".join(
        f"<tr><td>{item.role}</td><td><code>{escape(item.word)}</code></td>"
        f"<td>{item.token_mass:,}</td><td>{_optional_integer(item.calibration_fold)}</td>"
        f"<td>{_metric(item.role_margin_q10)}</td><td>{_metric(item.role_conformal_p)}</td>"
        f"<td>{_metric(item.context_silhouette)}</td><td>{_metric(item.cluster_margin)}</td>"
        f"<td>{escape(str(item.exclusion_reason or f'cluster {item.cluster}'))}</td></tr>"
        for item in words
    )
    return f"<div class=\"scroll\"><table><thead><tr><th>Role</th><th>Word</th><th>Mass</th><th>Fold</th><th>Raw q10</th><th>Conformal p</th><th>Silhouette</th><th>Cluster margin</th><th>Disposition</th></tr></thead><tbody>{rows}</tbody></table></div>"


def _representative_context_lines(
    words: Sequence[CalibratedFailureWord],
    contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
) -> tuple[str, ...]:
    selected = sorted(
        (item for item in words if item.vector is not None),
        key=lambda item: (item.role, -item.token_mass, item.word),
    )[:40]
    lines = []
    for item in selected:
        records = contexts.get((item.role, item.word), ())
        if records:
            record = records[0]
            sentence = str(record.get("sentence", "")).replace("\n", " ")
            lines.append(
                f"- {item.role} `{item.word}` — “{sentence}” "
                f"(`{record.get('record_id', '')}`, story `{record.get('story_sha256', '')}`)"
            )
    return tuple(lines)


def _cluster_summaries(
    words: Sequence[AuditWord],
    clusters: Sequence[SemanticCluster],
) -> dict[tuple[Role, int], dict[str, tuple[str, ...] | str]]:
    by_key = {(item.role, item.index): item for item in clusters}
    grouped: dict[tuple[Role, int], list[tuple[float, str]]] = defaultdict(list)
    for word in words:
        assert word.cluster is not None and word.vector is not None
        score = float(
            np.asarray(by_key[(word.role, word.cluster)].centroid)
            @ np.asarray(word.vector)
        )
        grouped[(word.role, word.cluster)].append((score, word.word))
    result = {}
    for key, values in grouped.items():
        descending = tuple(word for _, word in sorted(values, key=lambda item: (-item[0], item[1])))
        ascending = tuple(word for _, word in sorted(values, key=lambda item: (item[0], item[1])))
        result[key] = {
            "nearest": descending[:5],
            "median": descending[len(descending) // 2],
            "boundary": ascending[:5],
        }
    return result


def _cell_masses(
    words: Sequence[AuditWord],
    pair_masses: Mapping[tuple[str, str], int],
    count: int,
) -> np.ndarray:
    nouns = {item.word: item.cluster for item in words if item.role == "noun"}
    verbs = {item.word: item.cluster for item in words if item.role == "verb"}
    result = np.zeros((count, count), dtype=np.int64)
    for (noun, verb), mass in pair_masses.items():
        noun_cluster, verb_cluster = nouns.get(noun), verbs.get(verb)
        if noun_cluster is not None and verb_cluster is not None:
            result[noun_cluster, verb_cluster] += mass
    return result


def _cluster_context_lines(
    clusters: Sequence[SemanticCluster],
    summaries: Mapping[tuple[Role, int], Mapping[str, tuple[str, ...] | str]],
    contexts: Mapping[tuple[Role, str], Sequence[Mapping[str, object]]],
) -> tuple[str, ...]:
    lines = []
    for cluster in clusters:
        summary = summaries[(cluster.role, cluster.index)]
        selected = tuple(summary["nearest"][:1]) + (str(summary["median"]),) + tuple(
            summary["boundary"][:1]
        )
        lines.extend((f"### {cluster.role} cluster {cluster.index}", ""))
        for word in dict.fromkeys(selected):
            records = contexts.get((cluster.role, word), ())
            if not records:
                lines.append(f"- `{word}` — no selected exact context")
                continue
            record = records[0]
            sentence = str(record.get("sentence", "")).replace("\n", " ")
            lines.append(
                f"- `{word}` — “{sentence}” (`{record.get('record_id', '')}`, "
                f"story `{record.get('story_sha256', '')}`)"
            )
        lines.append("")
    return tuple(lines)


def _parent_differences(
    words: Sequence[AuditWord],
    parent: Sequence[AuditWord] | None,
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


def _pca_svg(words: Sequence[object], role: Role) -> str:
    selected = tuple(
        item
        for item in words
        if getattr(item, "role", None) == role and getattr(item, "vector", None) is not None
    )
    if len(selected) < 2:
        return f'<svg viewBox="0 0 600 400"><text x="20" y="30">{role}: fewer than two vectors</text></svg>'
    values = np.asarray([getattr(item, "vector") for item in selected], dtype=np.float64)
    centered = values - np.mean(values, axis=0)
    coordinates = np.linalg.svd(centered, full_matrices=False)[0][:, :2]
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    x = _scale(coordinates[:, 0], 35, 565)
    y = _scale(coordinates[:, 1], 365, 35)
    points = "".join(
        f'<circle cx="{x[index]:.2f}" cy="{y[index]:.2f}" r="3" fill="#2563eb"><title>{escape(str(getattr(item, "word")))}</title></circle>'
        for index, item in enumerate(selected)
    )
    return f'<svg viewBox="0 0 600 400"><text x="18" y="24">{role} semantic vectors — PCA</text>{points}</svg>'


def _heatmap_markdown(matrix: np.ndarray) -> str:
    header = "| N\\V | " + " | ".join(str(index) for index in range(matrix.shape[1])) + " |"
    divider = "|---|" + "---:|" * matrix.shape[1]
    rows = [
        f"| {row} | " + " | ".join(f"{int(value):,}" for value in matrix[row]) + " |"
        for row in range(matrix.shape[0])
    ]
    return "\n".join((header, divider, *rows))


def _heatmap_svg(matrix: np.ndarray) -> str:
    maximum = max(1, int(np.max(matrix)))
    cells = []
    size = 42
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            intensity = int(245 - 180 * value / maximum)
            cells.append(
                f'<rect x="{70 + column * size}" y="{40 + row * size}" width="{size}" height="{size}" fill="rgb({intensity},{intensity + 5},255)"><title>N{row} × V{column}: {value:,}</title></rect>'
            )
    extent = 90 + matrix.shape[0] * size
    return f'<svg viewBox="0 0 {extent} {extent}"><text x="15" y="24">Retained cell token mass</text>{"".join(cells)}</svg>'


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum == minimum:
        return np.full_like(values, (low + high) / 2, dtype=np.float64)
    return low + (values - minimum) * (high - low) / (maximum - minimum)


def _metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _optional_integer(value: int | None) -> str:
    return "—" if value is None else str(value)


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`")


def _style() -> str:
    return """<style>body{font:14px/1.45 system-ui,sans-serif;margin:2rem auto;max-width:1500px;padding:0 1rem;color:#172033}code,pre{font-family:ui-monospace,monospace}pre{background:#f4f6fa;padding:1rem;overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #ccd3df;padding:.35rem;text-align:left;vertical-align:top}th{background:#edf1f7;position:sticky;top:0}.scroll{overflow:auto;max-height:55rem}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:1rem}svg{border:1px solid #ccd3df;background:white;width:100%;height:auto}.stop{background:#fff1f0;border-left:5px solid #dc2626;padding:1rem}</style>"""


__all__ = [
    "AuditConfig",
    "AuditWord",
    "CalibratedFailureWord",
    "render_calibrated_catalog_audits",
    "render_calibrated_failure_audits",
]
