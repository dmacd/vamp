"""Build an auditable report for a post-launch seed-count amendment."""

from __future__ import annotations

import argparse
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
)
from apm.experiments.vamp_logt_mlp_permuted_config import (
    VampLogTDenseConfig,
    load_config,
)
from apm.experiments.vamp_logt_mlp_permuted_reporting import write_results
from apm.experiments.vamp_logt_router_reporting import _html, _load_jsonl


AMENDMENT_FILENAME = "analysis_amendment.json"
AMENDMENT_SCHEMA = "vamp-logt-dense-analysis-amendment-v1"
DEFAULT_CONFIG = Path("configs/vamp_logt_mlp_permuted_mnist_ungated/primary.yaml")

FINDING_LABELS = {
    "integrator_current_only": "Integrator — current only",
    "integrator_uniform_replay": "Integrator — uniform-history replay",
    "integrator_range_replay": "Integrator — range-balanced replay",
    "integrator_base_uniform_replay": "Base-only integrator — uniform replay",
    "mean_ensemble": "Equal-probability mean ensemble",
    "best_single_node": "Best active node (label-aware oracle)",
    "fresh_cumulative_four_epoch_integrator": "Fresh cumulative integrator — four epochs",
    "pooled_single_mlp_reference": "Pooled single MLP reference",
    "converged_full_replay_integrator": "Converged full-replay integrator ceiling",
    "router_current_hard": "Router — current only, hard target",
    "router_uniform_hard": "Router — uniform history, hard target",
    "router_range_hard": "Router — range-balanced history, hard target",
    "router_uniform_soft": "Router — uniform history, soft target",
    "router_range_soft": "Router — range-balanced history, soft target",
}

INTEGRATOR_FINDING_CONDITIONS = (
    "integrator_current_only",
    "integrator_uniform_replay",
    "integrator_range_replay",
    "mean_ensemble",
    "integrator_base_uniform_replay",
    "fresh_cumulative_four_epoch_integrator",
    "converged_full_replay_integrator",
    "pooled_single_mlp_reference",
    "best_single_node",
)

ROUTER_FINDING_CONDITIONS = (
    "router_current_hard",
    "router_uniform_hard",
    "router_range_hard",
    "router_uniform_soft",
    "router_range_soft",
)


def load_analysis_seeds(run_root: Path, config: VampLogTDenseConfig) -> tuple[int, ...]:
    """Load and validate the seed subset declared by the analysis amendment."""
    amendment_path = run_root / AMENDMENT_FILENAME
    if not amendment_path.is_file():
        raise FileNotFoundError(f"analysis amendment is missing: {amendment_path}")
    amendment = load_canonical_json(amendment_path)
    if amendment.get("schema_version") != AMENDMENT_SCHEMA:
        raise ValueError("analysis amendment has an unsupported schema")
    if amendment.get("status") != "active":
        raise ValueError("analysis amendment is not active")
    if amendment.get("config_hash") != config.config_hash:
        raise ValueError("analysis amendment config hash does not match the run")
    declared = tuple(int(seed) for seed in amendment.get("originally_declared_seeds", ()))
    if declared != config.online.seeds:
        raise ValueError("analysis amendment does not name the configured seed sequence")
    included = tuple(int(seed) for seed in amendment.get("included_seeds", ()))
    if not included or included != tuple(sorted(set(included))):
        raise ValueError("analysis seeds must be non-empty, unique, and sorted")
    if not set(included).issubset(declared):
        raise ValueError("analysis amendment includes an undeclared seed")
    return included


def write_amended_results(
    run_root: Path,
    config: VampLogTDenseConfig,
) -> Path:
    """Render a complete report using only the amended seed subset."""
    included = load_analysis_seeds(run_root, config)
    _require_complete_seed_artifacts(run_root, included)
    analysis_root = run_root / "analysis" / f"{len(included)}-seed-primary"
    _prepare_analysis_view(run_root, analysis_root, included)
    analysis_config = SimpleNamespace(
        calibration=config.calibration,
        config_hash=config.config_hash,
        evaluation=config.evaluation,
        online=replace(config.online, seeds=included),
    )
    summary = write_results(analysis_root, analysis_config)
    amendment_path = run_root / AMENDMENT_FILENAME
    amendment = load_canonical_json(amendment_path)
    criteria = dict(summary["criteria"])
    criteria["ceiling_every_step_cells"] = _count_learned_ceiling_cells(
        analysis_root,
        included,
    )
    findings = _build_headline_findings(
        analysis_root,
        config,
        included,
        all_decisions_pass=bool(criteria["all_pass"]),
    )
    enriched_summary = {
        **summary,
        "criteria": criteria,
        "analysis_amendment_sha256": file_sha256(amendment_path),
        "analysis_seeds": list(included),
        "excluded_completed_online_seeds": amendment.get(
            "excluded_completed_online_seeds", []
        ),
        "excluded_partial_ceiling_seeds": amendment.get(
            "excluded_partial_ceiling_seeds", {}
        ),
        "generated_online_seed_count": len(
            tuple((run_root / "online").glob("seed-*/summary.json"))
        ),
        "headline_findings": findings,
        "reporting_source_sha256": file_sha256(Path(__file__)),
        "source_run_root": str(run_root),
    }
    atomic_write(analysis_root / "summary.json", canonical_json_bytes(enriched_summary))
    markdown_path = analysis_root / "RESULTS.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown = _amend_markdown(markdown, amendment, included, findings)
    atomic_write(markdown_path, markdown.encode("utf-8"))
    atomic_write(
        analysis_root / "RESULTS.html",
        _html(markdown, analysis_root, "Three-seed dense Permuted-MNIST analysis").encode(
            "utf-8"
        ),
    )
    atomic_write(
        run_root / "LATEST_ANALYSIS.json",
        canonical_json_bytes(
            {
                "analysis_root": str(analysis_root),
                "analysis_seeds": list(included),
                "config_hash": config.config_hash,
                "schema_version": "vamp-logt-dense-latest-analysis-v1",
                "status": enriched_summary["status"],
            }
        ),
    )
    return analysis_root


def _count_learned_ceiling_cells(
    analysis_root: Path,
    seeds: tuple[int, ...],
) -> int:
    """Count only learned-ceiling cells, excluding its mean-ensemble control."""
    return sum(
        1
        for seed in seeds
        for row in _load_jsonl(
            analysis_root / "ceiling" / f"seed-{seed}" / "metrics.jsonl"
        )
        if row.get("row_type") == "ceiling_evaluation"
        and row.get("condition") == "converged_full_replay_integrator"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
    )


def _build_headline_findings(
    analysis_root: Path,
    config: VampLogTDenseConfig,
    seeds: tuple[int, ...],
    *,
    all_decisions_pass: bool,
) -> dict[str, object]:
    checkpoints = config.evaluation.headline_checkpoints
    online_rows = tuple(
        row
        for seed in seeds
        for row in _load_jsonl(
            analysis_root / "online" / f"seed-{seed}" / "metrics.jsonl"
        )
        if int(row.get("macro_step", -1)) in checkpoints
    )
    ceiling_rows = tuple(
        row
        for seed in seeds
        for row in _load_jsonl(
            analysis_root / "ceiling" / f"seed-{seed}" / "metrics.jsonl"
        )
        if int(row.get("macro_step", -1)) in checkpoints
    )
    online_integrator_conditions = tuple(
        condition
        for condition in INTEGRATOR_FINDING_CONDITIONS
        if condition != "converged_full_replay_integrator"
    )
    integrator = {
        condition: _condition_metrics(
            online_rows,
            seeds,
            checkpoints,
            condition,
            row_type="integrator_evaluation",
            accuracy_field="accuracy",
            cross_entropy_field="mean_cross_entropy",
        )
        for condition in online_integrator_conditions
    }
    integrator["converged_full_replay_integrator"] = _condition_metrics(
        ceiling_rows,
        seeds,
        checkpoints,
        "converged_full_replay_integrator",
        row_type="ceiling_evaluation",
        accuracy_field="accuracy",
        cross_entropy_field="mean_cross_entropy",
    )
    routers = {
        condition: _condition_metrics(
            online_rows,
            seeds,
            checkpoints,
            condition,
            row_type="router_evaluation",
            accuracy_field="selected_accuracy",
            cross_entropy_field="selected_mean_cross_entropy",
        )
        for condition in ROUTER_FINDING_CONDITIONS
    }
    checkpoint_metrics = {
        str(checkpoint): {
            condition: _condition_metrics(
                ceiling_rows
                if condition == "converged_full_replay_integrator"
                else online_rows,
                seeds,
                (checkpoint,),
                condition,
                row_type=(
                    "ceiling_evaluation"
                    if condition == "converged_full_replay_integrator"
                    else "integrator_evaluation"
                ),
                accuracy_field="accuracy",
                cross_entropy_field="mean_cross_entropy",
            )
            for condition in (
                "integrator_current_only",
                "integrator_uniform_replay",
                "integrator_range_replay",
                "fresh_cumulative_four_epoch_integrator",
                "converged_full_replay_integrator",
            )
        }
        for checkpoint in checkpoints
    }
    retention = {
        condition: {
            group: _optional_condition_metrics(
                online_rows,
                seeds,
                checkpoints,
                condition,
                row_type="integrator_evaluation",
                accuracy_field="accuracy",
                cross_entropy_field="mean_cross_entropy",
                evaluation_scope="evaluation_archive",
                group=group,
            )
            for group in ("current_range", "older_ranges")
        }
        for condition in (
            "integrator_current_only",
            "integrator_uniform_replay",
            "integrator_range_replay",
        )
    }
    best_replay = min(
        ("integrator_uniform_replay", "integrator_range_replay"),
        key=lambda condition: float(
            integrator[condition]["cross_entropy"]["mean"]
        ),
    )
    best_router = min(
        ROUTER_FINDING_CONDITIONS[1:],
        key=lambda condition: float(routers[condition]["cross_entropy"]["mean"]),
    )
    highest_accuracy_router = max(
        ROUTER_FINDING_CONDITIONS[1:],
        key=lambda condition: float(routers[condition]["accuracy"]["mean"]),
    )
    current = integrator["integrator_current_only"]
    replay = integrator[best_replay]
    mean_ensemble = integrator["mean_ensemble"]
    four_epoch = integrator["fresh_cumulative_four_epoch_integrator"]
    ceiling = integrator["converged_full_replay_integrator"]
    base_only = integrator["integrator_base_uniform_replay"]
    positive_gap = float(current["cross_entropy"]["mean"]) - float(
        four_epoch["cross_entropy"]["mean"]
    )
    closure = (
        None
        if positive_gap <= 0.0
        else (
            float(current["cross_entropy"]["mean"])
            - float(replay["cross_entropy"]["mean"])
        )
        / positive_gap
    )
    return {
        "aggregation": (
            "Equal-weight mean over full-test permutation cells and headline "
            "checkpoints within each seed; variability is sample standard deviation "
            "across seed-level means."
        ),
        "all_decisions_pass": all_decisions_pass,
        "best_online_replay_condition": best_replay,
        "best_router_by_cross_entropy": best_router,
        "checkpoints": checkpoint_metrics,
        "deltas": {
            "base_only_accuracy_gain_percentage_points": 100.0
            * (
                float(replay["accuracy"]["mean"])
                - float(base_only["accuracy"]["mean"])
            ),
            "ceiling_accuracy_gap_percentage_points": 100.0
            * (
                float(ceiling["accuracy"]["mean"])
                - float(replay["accuracy"]["mean"])
            ),
            "ceiling_cross_entropy_gap": float(replay["cross_entropy"]["mean"])
            - float(ceiling["cross_entropy"]["mean"]),
            "four_epoch_gap_closure": closure,
            "mean_ensemble_accuracy_gain_percentage_points": 100.0
            * (
                float(replay["accuracy"]["mean"])
                - float(mean_ensemble["accuracy"]["mean"])
            ),
            "mean_ensemble_cross_entropy_reduction": float(
                mean_ensemble["cross_entropy"]["mean"]
            )
            - float(replay["cross_entropy"]["mean"]),
            "replay_accuracy_gain_percentage_points": 100.0
            * (
                float(replay["accuracy"]["mean"])
                - float(current["accuracy"]["mean"])
            ),
            "replay_cross_entropy_reduction": float(
                current["cross_entropy"]["mean"]
            )
            - float(replay["cross_entropy"]["mean"]),
            "router_accuracy_gap_percentage_points": 100.0
            * (
                float(replay["accuracy"]["mean"])
                - float(routers[best_router]["accuracy"]["mean"])
            ),
            "router_cross_entropy_reduction": float(
                routers[best_router]["cross_entropy"]["mean"]
            )
            - float(replay["cross_entropy"]["mean"]),
        },
        "headline_checkpoints": list(checkpoints),
        "highest_accuracy_router": highest_accuracy_router,
        "integrator": integrator,
        "retention": retention,
        "router": routers,
        "seed_count": len(seeds),
    }


def _condition_metrics(
    rows: Sequence[Mapping[str, object]],
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    condition: str,
    *,
    row_type: str,
    accuracy_field: str,
    cross_entropy_field: str,
    evaluation_scope: str = "full_test",
    group: str = "micro",
) -> dict[str, object]:
    selectors = {
        "condition": condition,
        "evaluation_scope": evaluation_scope,
        "group": group,
        "row_type": row_type,
    }
    return {
        "accuracy": _seed_metric(
            rows,
            seeds,
            checkpoints,
            selectors,
            accuracy_field,
        ),
        "cross_entropy": _seed_metric(
            rows,
            seeds,
            checkpoints,
            selectors,
            cross_entropy_field,
        ),
    }


def _optional_condition_metrics(
    rows: Sequence[Mapping[str, object]],
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    condition: str,
    *,
    row_type: str,
    accuracy_field: str,
    cross_entropy_field: str,
    evaluation_scope: str,
    group: str,
) -> dict[str, object] | None:
    if not any(
        int(row.get("macro_step", -1)) in checkpoints
        and row.get("condition") == condition
        and row.get("row_type") == row_type
        and row.get("evaluation_scope") == evaluation_scope
        and row.get("group") == group
        for row in rows
    ):
        return None
    return _condition_metrics(
        rows,
        seeds,
        checkpoints,
        condition,
        row_type=row_type,
        accuracy_field=accuracy_field,
        cross_entropy_field=cross_entropy_field,
        evaluation_scope=evaluation_scope,
        group=group,
    )


def _seed_metric(
    rows: Sequence[Mapping[str, object]],
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    selectors: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    seed_means = []
    for seed in seeds:
        values = [
            float(row[field])
            for row in rows
            if int(row.get("run_seed", -1)) == seed
            and int(row.get("macro_step", -1)) in checkpoints
            and all(row.get(key) == value for key, value in selectors.items())
        ]
        if not values:
            raise ValueError(
                f"no {field} rows for seed {seed}, checkpoints {checkpoints}, "
                f"and selectors {dict(selectors)}"
            )
        seed_means.append(statistics.fmean(values))
    return {
        "mean": statistics.fmean(seed_means),
        "per_seed": seed_means,
        "sample_standard_deviation": (
            statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
        ),
    }


def main() -> None:
    """Render the amended analysis without reopening generation phases."""
    parser = argparse.ArgumentParser(
        description="Render a seed-subset report for the dense Permuted-MNIST run"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    run_root = config.artifact_root / "runs" / config.config_hash
    analysis_root = write_amended_results(run_root, config)
    print(f"Amended analysis: {analysis_root}", flush=True)


def _require_complete_seed_artifacts(run_root: Path, seeds: tuple[int, ...]) -> None:
    for phase in ("online", "ceiling"):
        for seed in seeds:
            directory = run_root / phase / f"seed-{seed}"
            summary_path = directory / "summary.json"
            metrics_path = directory / "metrics.jsonl"
            if not summary_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f"{phase} seed {seed} is incomplete")
            summary = load_canonical_json(summary_path)
            if summary.get("status") != "complete" or int(
                summary.get("final_macro_step", -1)
            ) != config_macro_steps(run_root):
                raise ValueError(f"{phase} seed {seed} does not cover the full stream")


def config_macro_steps(run_root: Path) -> int:
    """Return the macro-step count frozen in the canonical run protocol."""
    protocol = load_canonical_json(run_root / "protocol.json")
    return int(protocol["config"]["benchmark"]["macro_steps"])


def _prepare_analysis_view(
    run_root: Path,
    analysis_root: Path,
    seeds: tuple[int, ...],
) -> None:
    analysis_root.mkdir(parents=True, exist_ok=True)
    _ensure_relative_symlink(run_root / "calibration", analysis_root / "calibration")
    baseline_root = run_root / "baselines"
    if (baseline_root / "summary.json").is_file():
        _ensure_relative_symlink(baseline_root, analysis_root / "baselines")
    for phase in ("online", "ceiling"):
        phase_root = analysis_root / phase
        phase_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            _ensure_relative_symlink(
                run_root / phase / f"seed-{seed}",
                phase_root / f"seed-{seed}",
            )


def _ensure_relative_symlink(source: Path, link: Path) -> None:
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f"analysis link points to the wrong source: {link}")
        return
    if link.exists():
        raise FileExistsError(f"analysis view path already exists: {link}")
    relative_source = os.path.relpath(source, start=link.parent)
    link.symlink_to(relative_source, target_is_directory=source.is_dir())


def _amend_markdown(
    markdown: str,
    amendment: dict[str, object],
    seeds: tuple[int, ...],
    findings: Mapping[str, object],
) -> str:
    status_line = "Status: **complete**."
    excluded_online = amendment.get("excluded_completed_online_seeds", [])
    excluded_partial = amendment.get("excluded_partial_ceiling_seeds", {})
    decision_stage = amendment.get("decision_stage", {})
    if isinstance(decision_stage, Mapping) and decision_stage:
        completed_ceiling_seeds = int(
            decision_stage.get("ceiling_completed_seed_count", 0)
        )
        completion_text = (
            "before any ceiling seed had completed"
            if completed_ceiling_seeds == 0
            else f"after {completed_ceiling_seeds} ceiling seeds had completed"
        )
        timing = (
            f"while ceiling seed {decision_stage.get('ceiling_in_progress_seed')} was "
            f"running at macro-step {decision_stage.get('ceiling_in_progress_macro_step')} "
            f"and {completion_text}"
        )
    else:
        timing = "before the amended analysis was generated"
    exclusion_text = (
        f"Completed online seeds {excluded_online} remain in the source artifact but are "
        "excluded from every number and plot in this report. "
        if excluded_online
        else "No completed online seed is excluded from this report. "
    )
    partial_text = _partial_evidence_text(excluded_partial)
    amendment_text = (
        f"{status_line}\n\n"
        "## Analysis-set amendment\n\n"
        f"Primary results use seeds `{list(seeds)}` (n={len(seeds)}). "
        f"The run originally declared seeds `{amendment['originally_declared_seeds']}`. "
        f"The seed count was reduced for exploratory compute control {timing}. "
        f"{exclusion_text}{partial_text}\n\n"
        f"Recorded reason: {amendment['reason']}"
    )
    if status_line not in markdown:
        raise ValueError("generated report does not contain the expected status line")
    markdown = markdown.replace(status_line, amendment_text, 1)
    inherited_claim = (
        "Architecture selection used validation only; test metrics were opened after selection."
    )
    amended_claim = (
        "This successor selected the smallest candidate by an explicit post-hoc amendment; "
        "accuracy thresholds were non-operative. It imported and authenticated the original "
        "calibration evidence rather than rerunning it."
    )
    if inherited_claim not in markdown:
        raise ValueError("generated report does not contain the inherited calibration wording")
    markdown = markdown.replace(inherited_claim, amended_claim, 1)
    condition_heading = "## Conditions in plain language"
    if condition_heading not in markdown:
        raise ValueError("generated report does not contain the condition heading")
    return markdown.replace(
        condition_heading,
        f"{_findings_markdown(findings)}\n\n{condition_heading}",
        1,
    )


def _findings_markdown(findings: Mapping[str, object]) -> str:
    seed_count = int(findings["seed_count"])
    integrator = findings["integrator"]
    routers = findings["router"]
    retention = findings["retention"]
    checkpoints = findings["checkpoints"]
    deltas = findings["deltas"]
    best_replay = str(findings["best_online_replay_condition"])
    best_router = str(findings["best_router_by_cross_entropy"])
    highest_accuracy_router = str(findings["highest_accuracy_router"])
    closure = deltas["four_epoch_gap_closure"]
    closure_text = (
        "The current-only to fresh-four-epoch cross-entropy gap is non-positive, "
        "so a closure fraction is undefined."
        if closure is None
        else (
            "The bounded online replay integrator closes "
            f"{100.0 * float(closure):.1f}% of the current-only to "
            "fresh-four-epoch cross-entropy gap."
        )
    )
    verdict = (
        "**Promising under all seven frozen decision rules.**"
        if findings["all_decisions_pass"]
        else "**Mixed under the frozen decision rules.**"
    )
    lines = [
        "## Three-seed result",
        "",
        f"Verdict: {verdict}",
        "",
        (
            "Each headline value first averages the full-test permutation cells at macro-steps "
            f"{findings['headline_checkpoints']} within a seed. Values are mean ± sample "
            f"standard deviation across the {seed_count} seed means; the standard deviation "
            "is descriptive, not a confidence interval."
        ),
        "",
        "| Condition | Accuracy | Cross-entropy |",
        "|---|---:|---:|",
    ]
    for condition in INTEGRATOR_FINDING_CONDITIONS:
        metrics = integrator[condition]
        lines.append(
            f"| {FINDING_LABELS[condition]} | "
            f"{_format_metric(metrics['accuracy'], percentage=True)} | "
            f"{_format_metric(metrics['cross_entropy'])} |"
        )
    lines.extend(
        (
            "",
            "### Evolution across the headline checkpoints",
            "",
            "Each cell is accuracy / cross-entropy, averaged across the three seeds and all "
            "eight test permutations.",
            "",
            "| Macro-step | Current only | Uniform replay | Range replay | "
            "Fresh four-epoch | Converged ceiling |",
            "|---:|---:|---:|---:|---:|---:|",
        )
    )
    for checkpoint in findings["headline_checkpoints"]:
        checkpoint_row = checkpoints[str(checkpoint)]
        cells = [
            _format_accuracy_and_loss(checkpoint_row[condition])
            for condition in (
                "integrator_current_only",
                "integrator_uniform_replay",
                "integrator_range_replay",
                "fresh_cumulative_four_epoch_integrator",
                "converged_full_replay_integrator",
            )
        ]
        lines.append(f"| {checkpoint} | " + " | ".join(cells) + " |")
    lines.extend(
        (
            "",
            "### Retention on the evaluation archive",
            "",
            "| Condition | Current-range accuracy / CE | Older-range accuracy / CE |",
            "|---|---:|---:|",
        )
    )
    for condition in (
        "integrator_current_only",
        "integrator_uniform_replay",
        "integrator_range_replay",
    ):
        condition_retention = retention[condition]
        lines.append(
            f"| {FINDING_LABELS[condition]} | "
            f"{_format_accuracy_and_loss(condition_retention['current_range'])} | "
            f"{_format_accuracy_and_loss(condition_retention['older_ranges'])} |"
        )
    lines.extend(
        (
            "",
            "### Interpretation",
            "",
            (
                f"- {FINDING_LABELS[best_replay]} is the replay condition with the lower "
                f"headline cross-entropy. Relative to current-only training it gains "
                f"{float(deltas['replay_accuracy_gain_percentage_points']):.2f} accuracy "
                f"points and reduces cross-entropy by "
                f"{float(deltas['replay_cross_entropy_reduction']):.4f}."
            ),
            "",
            (
                "- Uniform and range-balanced replay are practically tied: "
                f"range-balanced replay is {_accuracy_difference(integrator):+.2f} "
                "accuracy points relative to uniform replay, while its cross-entropy is "
                f"{_cross_entropy_difference(integrator):+.4f}. "
                f"With n={seed_count}, this does not identify a sampler winner."
            ),
            "",
            (
                f"- {closure_text} It remains "
                f"{float(deltas['ceiling_accuracy_gap_percentage_points']):.2f} accuracy "
                "points and "
                f"{float(deltas['ceiling_cross_entropy_gap']):.4f} cross-entropy behind the "
                "converged full-replay integrator."
            ),
            "",
            (
                "- Frozen temporal-node features matter. The full-node replay integrator "
                f"beats the base-only replay ablation by "
                f"{float(deltas['base_only_accuracy_gain_percentage_points']):.2f} accuracy "
                "points."
            ),
            "",
            (
                f"- Against the router selected by cross-entropy "
                f"({FINDING_LABELS[best_router]}), the replay integrator gains "
                f"{float(deltas['router_accuracy_gap_percentage_points']):.2f} accuracy "
                "points and reduces cross-entropy by "
                f"{float(deltas['router_cross_entropy_reduction']):.4f}. The highest-accuracy "
                f"router is instead {FINDING_LABELS[highest_accuracy_router]} at "
                f"{100.0 * float(routers[highest_accuracy_router]['accuracy']['mean']):.2f}% "
                "accuracy, but with cross-entropy "
                f"{float(routers[highest_accuracy_router]['cross_entropy']['mean']):.4f}."
            ),
            "",
            (
                "- The converged trace is an empirical optimization ceiling for this fixed "
                "feature representation, integrator architecture, convergence rule, and "
                "three-restart search. It is not a mathematical upper bound on every possible "
                "integrator. The label-aware best-node oracle is also not deployable; it uses "
                "test labels directly."
            ),
            "",
            (
                f"- The evidence is encouraging but still exploratory: n={seed_count} is too "
                "small for a precise uncertainty estimate. Seeds 3 and 4 should be added only "
                "as a predeclared extension, without changing the conditions or headline cells."
            ),
        )
    )
    return "\n".join(lines)


def _format_metric(metric: Mapping[str, object], *, percentage: bool = False) -> str:
    scale = 100.0 if percentage else 1.0
    digits = 2 if percentage else 4
    mean = scale * float(metric["mean"])
    deviation = scale * float(metric["sample_standard_deviation"])
    suffix = "%" if percentage else ""
    return f"{mean:.{digits}f} ± {deviation:.{digits}f}{suffix}"


def _format_accuracy_and_loss(metrics: Mapping[str, object] | None) -> str:
    if metrics is None:
        return "not available"
    accuracy = 100.0 * float(metrics["accuracy"]["mean"])
    cross_entropy = float(metrics["cross_entropy"]["mean"])
    return f"{accuracy:.2f}% / {cross_entropy:.4f}"


def _partial_evidence_text(value: object) -> str:
    if not isinstance(value, Mapping) or not value:
        return "There is no excluded partial ceiling evidence."
    descriptions = []
    for seed, evidence in value.items():
        if not isinstance(evidence, Mapping):
            descriptions.append(f"seed {seed}")
            continue
        descriptions.append(
            f"seed {seed}: {int(evidence.get('completed_macro_steps', 0))} "
            "macro-steps and no completed summary"
        )
    return "Excluded partial ceiling evidence is retained for " + "; ".join(
        descriptions
    ) + "."


def _accuracy_difference(integrator: Mapping[str, object]) -> float:
    range_accuracy = integrator["integrator_range_replay"]["accuracy"]["mean"]
    uniform_accuracy = integrator["integrator_uniform_replay"]["accuracy"]["mean"]
    return 100.0 * (float(range_accuracy) - float(uniform_accuracy))


def _cross_entropy_difference(integrator: Mapping[str, object]) -> float:
    range_loss = integrator["integrator_range_replay"]["cross_entropy"]["mean"]
    uniform_loss = integrator["integrator_uniform_replay"]["cross_entropy"]["mean"]
    return float(range_loss) - float(uniform_loss)


__all__ = [
    "AMENDMENT_FILENAME",
    "AMENDMENT_SCHEMA",
    "DEFAULT_CONFIG",
    "config_macro_steps",
    "load_analysis_seeds",
    "main",
    "write_amended_results",
]


if __name__ == "__main__":
    main()
