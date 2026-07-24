"""Audit adapters that make semantic-v4's frozen-fit semantics explicit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from apm.data.text.tinyworlds_p_semantic.contracts import Role, SemanticCluster
from apm.data.text.tinyworlds_p_semantic.v2_audit import (
    CalibratedFailureWord,
    render_calibrated_catalog_audits,
    render_calibrated_failure_audits,
)
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    BoundaryPassMetric,
    RoleCalibrationReference,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import (
    V4SemanticConstructionConfig,
    V4SemanticWord,
)


def render_v4_failure_audits(
    failure_sha256: str,
    evidence_sha256: str,
    reason: str,
    config: V4SemanticConstructionConfig,
    calibration: Sequence[RoleCalibrationReference],
    boundary_trace: Sequence[BoundaryPassMetric],
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
) -> tuple[str, str]:
    """Render a failure audit plus every available frozen-fit assignment."""
    base_words = tuple(
        CalibratedFailureWord(
            role=item.role,
            word=item.word,
            token_mass=item.token_mass,
            context_count=item.context_count,
            calibration_fold=item.calibration_fold,
            calibration_reference_count=item.calibration_reference_count,
            role_margin_q10=item.role_margin_q10,
            role_conformal_p=item.role_conformal_p,
            role_rejection_cutoff=item.role_rejection_cutoff,
            context_silhouette=item.context_silhouette,
            disposition=(
                item.exclusion_reason
                if item.exclusion_reason is not None
                else "fixed_centroid_candidate"
            ),
            vector=item.vector,
        )
        for item in words
    )
    markdown, html = render_calibrated_failure_audits(
        failure_sha256,
        evidence_sha256,
        reason,
        config,
        calibration,
        boundary_trace,
        base_words,
        representative_contexts,
        "v4",
    )
    fitted = tuple(item for item in words if item.fit_cluster is not None)
    fit_lines = (
        "## Frozen pass-zero fit evidence",
        "",
        "Centroids and assignments below are from the all-candidate v3 pass-zero fit. "
        "No centroid was updated and no survivor was reassigned after screening.",
        "",
        "| Role | Word | Fit cluster | Fixed margin | Final disposition |",
        "|---|---|---:|---:|---|",
        *(
            f"| {item.role} | `{_markdown_escape(item.word)}` | {item.fit_cluster} | "
            f"{item.cluster_margin:.9f} | "
            f"{item.exclusion_reason or f'cluster {item.cluster}'} |"
            for item in fitted
            if item.cluster_margin is not None
        ),
        "",
        "### Fit clusters",
        "",
        "| Role | Cluster | Fit words | Fit token mass |",
        "|---|---:|---:|---:|",
        *(
            f"| {item.role} | {item.index} | {len(item.words)} | "
            f"{item.token_mass:,} |"
            for item in fit_clusters
        ),
    )
    markdown = markdown.rstrip() + "\n\n" + "\n".join(fit_lines) + "\n"
    rows = "".join(
        f"<tr><td>{item.role}</td><td><code>{escape(item.word)}</code></td>"
        f"<td>{item.fit_cluster}</td><td>{item.cluster_margin:.9f}</td>"
        f"<td>{escape(str(item.exclusion_reason or f'cluster {item.cluster}'))}</td></tr>"
        for item in fitted
        if item.cluster_margin is not None
    )
    cluster_rows = "".join(
        f"<tr><td>{item.role}</td><td>{item.index}</td>"
        f"<td>{len(item.words)}</td><td>{item.token_mass:,}</td></tr>"
        for item in fit_clusters
    )
    supplement = (
        "<h2>Frozen pass-zero fit evidence</h2>"
        "<p>Centroids and assignments are from the all-candidate v3 pass-zero fit. "
        "No centroid was updated and no survivor was reassigned after screening.</p>"
        "<div class=\"scroll\"><table><thead><tr><th>Role</th><th>Word</th>"
        "<th>Fit cluster</th><th>Fixed margin</th><th>Final disposition</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
        "<h3>Fit clusters</h3><table><thead><tr><th>Role</th><th>Cluster</th>"
        f"<th>Fit words</th><th>Fit token mass</th></tr></thead><tbody>{cluster_rows}"
        "</tbody></table>"
    )
    return markdown, html.replace("</body>", supplement + "</body>")


def render_v4_catalog_audits(
    catalog_sha256: str,
    evidence_sha256: str,
    config: V4SemanticConstructionConfig,
    calibration: Sequence[RoleCalibrationReference],
    boundary_trace: Sequence[BoundaryPassMetric],
    words: Sequence[V4SemanticWord],
    fit_clusters: Sequence[SemanticCluster],
    clusters: Sequence[SemanticCluster],
    pair_masses: Mapping[tuple[str, str], int],
    representative_contexts: Mapping[
        tuple[Role, str], Sequence[Mapping[str, object]]
    ],
) -> tuple[str, str]:
    """Render the ordinary catalog audit plus fit-versus-retained counts."""
    markdown, html = render_calibrated_catalog_audits(
        catalog_sha256,
        evidence_sha256,
        config,
        calibration,
        boundary_trace,
        words,
        clusters,
        pair_masses,
        representative_contexts,
        None,
        "v4",
    )
    retained_by_key = {(item.role, item.index): item for item in clusters}
    fit_lines = (
        "## Frozen-fit versus retained inventory",
        "",
        "Published centroids are the all-candidate fit centroids. Only the word "
        "inventory and audit mass change after the one-shot boundary screen.",
        "",
        "| Role | Cluster | Fit words | Retained words | Fit mass | Retained mass |",
        "|---|---:|---:|---:|---:|---:|",
        *(
            f"| {fit.role} | {fit.index} | {len(fit.words)} | "
            f"{len(retained_by_key[(fit.role, fit.index)].words)} | "
            f"{fit.token_mass:,} | "
            f"{retained_by_key[(fit.role, fit.index)].token_mass:,} |"
            for fit in fit_clusters
        ),
    )
    markdown = markdown.rstrip() + "\n\n" + "\n".join(fit_lines) + "\n"
    rows = "".join(
        f"<tr><td>{fit.role}</td><td>{fit.index}</td><td>{len(fit.words)}</td>"
        f"<td>{len(retained_by_key[(fit.role, fit.index)].words)}</td>"
        f"<td>{fit.token_mass:,}</td>"
        f"<td>{retained_by_key[(fit.role, fit.index)].token_mass:,}</td></tr>"
        for fit in fit_clusters
    )
    supplement = (
        "<h2>Frozen-fit versus retained inventory</h2>"
        "<p>Published centroids are the all-candidate fit centroids. Only word "
        "inventory and audit mass change after the one-shot screen.</p>"
        "<table><thead><tr><th>Role</th><th>Cluster</th><th>Fit words</th>"
        "<th>Retained words</th><th>Fit mass</th><th>Retained mass</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>"
    )
    return markdown, html.replace("</body>", supplement + "</body>")


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("`", "\\`")


__all__ = ["render_v4_catalog_audits", "render_v4_failure_audits"]
