"""Complete-sentence Markdown and HTML reports for generative-PC evidence."""

from __future__ import annotations

from html import escape
import io
from pathlib import Path
from typing import Mapping

import numpy as np

from apm.continual.artifacts import atomic_write, publish_immutable_json
from apm.experiments.vamp_logt_pc_config import VampLogTPcConfig


CONDITION_DEFINITIONS = {
    "map": (
        "The MAP condition settles each candidate model's 160 inferred values for 80 fixed "
        "gradient steps, evaluates the complete normalized log joint at that resulting state, "
        "and routes to the node with the largest score. The score includes the standard-normal "
        "latent prior and every Gaussian normalization constant. It is not a marginal likelihood."
    ),
    "oracle": (
        "The label-aware oracle evaluates every node-local classifier with the true digit "
        "target and chooses the smallest cross-entropy loss; it is diagnostic and cannot be "
        "used for task-free routing."
    ),
    "novel_leaf": (
        "The novel-leaf schedule assigns sixteen C0 blocks to the level-four history, then "
        "eight C1, four C2, two C3, and one new C4 leaf block."
    ),
    "recurrent_leaf_1_8": (
        "The recurrent-leaf schedule assigns fourteen C0 and two disjoint C4 blocks to the "
        "level-four history, followed by C1, C2, C3, and a separate C4 leaf."
    ),
    "identical_regime": (
        "The identical-regime schedule trains both the sixteen-block level-four history and "
        "the one-block leaf on disjoint examples from C4, making node level and sample count "
        "the controlled difference."
    ),
}


def write_result_report(
    run_root: Path,
    config: VampLogTPcConfig,
    phases: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Write one self-contained account of reached phases and the gated verdict."""
    verdict, status = _verdict(phases)
    summary: dict[str, object] = {
        "condition_definitions": CONDITION_DEFINITIONS,
        "phases": {name: dict(value) for name, value in phases.items()},
        "schema_version": "vamp-logt-generative-pc-map-result-v1",
        "status": status,
        "verdict": verdict,
    }
    publish_immutable_json(run_root / "summary.json", summary)
    figures = _write_figures(run_root)
    markdown = _markdown(config, phases, status, verdict, figures)
    atomic_write(run_root / "HANDOFF.md", markdown.encode("utf-8"))
    atomic_write(run_root / "report.md", markdown.encode("utf-8"))
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Generative-PC LogT "
        "evidence</title><style>body{max-width:1100px;margin:2rem auto;font-family:system-ui;"
        "line-height:1.45}pre{white-space:pre-wrap}</style></head><body><pre>"
        + escape(markdown)
        + "</pre></body></html>"
    )
    atomic_write(run_root / "report.html", html.encode("utf-8"))
    return summary


def _markdown(
    config: VampLogTPcConfig,
    phases: Mapping[str, Mapping[str, object]],
    status: str,
    verdict: str,
    figures: tuple[Path, ...],
) -> str:
    definitions = "\n".join(f"- `{name}`: {text}" for name, text in CONDITION_DEFINITIONS.items())
    phase_lines = []
    for name, phase in phases.items():
        phase_lines.append(
            f"- `{name}` completed with `passed={phase.get('passed')}` under schema "
            f"`{phase.get('schema_version', 'unknown')}`."
        )
    selected = phases.get("preflight", {}).get("selected_protocol")
    static_text = _static_text(phases.get("static_minimal"), phases.get("static_confirmation"))
    consolidation_text = _consolidation_text(phases.get("consolidation"))
    figure_lines = "\n\n".join(
        f"![{path.stem.replace('_', ' ')}](plots/{path.name})" for path in figures
    )
    return f"""# MAP generative-PC routing for LogT-VAMP on MNIST

## Outcome

The workflow status is `{status}`, and the preregistered verdict is `{verdict}`. A gated stop means that the runner deliberately did not execute later phases; it does not mean that an exception was reclassified as an experimental result.

The experiment uses Python 3.11, JAX and JAXlib 0.7.0, and FabricPC 0.4.0 at commit `138941ef5763ab202c7df07879d3f21678e6cc0a`. Every active temporal node owns an independently trained normalized generative predictive-coding model and a stopped-gradient linear classifier. The sole task-free routing score is the complete MAP joint score defined below. The runner does not construct a Hessian, apply a curvature correction, or estimate a marginal likelihood. Evidence receives only a normalized raw image. Labels enter classifier training and oracle diagnostics only, while contexts enter schedule construction and provenance only. No frozen CNN values, adapters, node-size priors, or fitted score offsets enter routing.

The selected global preflight settings were `{selected}`.

## Conditions

{definitions}

## Completed phases

{chr(10).join(phase_lines)}

## Static routing

{static_text}

## Partial carry

{consolidation_text}

## Work and interpretation boundary

The density and classifier schedules have fixed epochs, batches, and inference iterations. The persisted counters separately record leaf and merge presentations, latent-state updates, exhaustive model evaluations, and the live-model gauge. The Hessian and importance-sampling counters must remain zero in this MAP-only protocol. The runner asserts the declared fixed-multiple LogT training bounds after every committed block.

The earlier corrected NCE/TRE experiment is historical motivation only. It used a different estimator and stopped after every static schedule failed, so its numerical scores are not treated as a matched control here.

## Figures

{figure_lines or 'No static score figure was produced because the workflow stopped before static evaluation.'}
"""


def _static_text(
    minimal: Mapping[str, object] | None,
    confirmation: Mapping[str, object] | None,
) -> str:
    if minimal is None:
        return "The static experiment was not reached because an earlier implementation or model-quality gate failed."
    lines = [
        f"The minimal stream-seed phase passed: `{minimal['passed']}`. Its scores passing all three controlled conditions were `{minimal['passing_scores']}`."
    ]
    for condition in minimal["conditions"]:
        summaries = condition["score_summaries"]
        details = ", ".join(
            f"{score} passed={record['passed']} with minimum replica agreement "
            f"{float(record['minimum_route_agreement']):.3f} and worst oracle gap "
            f"{float(record['worst_oracle_gap']):.3f}"
            for score, record in summaries.items()
        )
        lines.append(
            f"For stream seed {condition['stream_seed']} under `{condition['condition']}`, {details}."
        )
    if confirmation is None:
        lines.append("Confirmation was not run because no score passed every minimal static gate.")
    else:
        lines.append(
            f"Confirmation passed: `{confirmation['passed']}`. The scores remaining after stream seeds 1 and 2 were `{confirmation['passing_scores']}`."
        )
    return " ".join(lines)


def _consolidation_text(consolidation: Mapping[str, object] | None) -> str:
    if consolidation is None:
        return "The block-27 to block-28 partial carry was not reached because static evidence did not pass confirmation."
    return (
        f"The partial carry passed: `{consolidation['passed']}`. The committed level-two parent "
        f"was `{consolidation['parent_node_id']}`, its retired children were absent after the "
        "bank checkpoint, and its independent twin was evaluated without averaging, "
        "distillation, inherited offsets, or child-weight initialization."
    )


def _verdict(phases: Mapping[str, Mapping[str, object]]) -> tuple[str, str]:
    if not bool(phases.get("analytic", {}).get("passed", False)):
        return "inconclusive", "blocked_after_analytic"
    if not bool(phases.get("preflight", {}).get("passed", False)):
        return "inconclusive", "blocked_after_preflight"
    minimal = phases.get("static_minimal")
    if minimal is None:
        return "inconclusive", "blocked_before_static"
    if not bool(minimal.get("passed", False)):
        oracle_values = [
            float(row["oracle_accuracy"])
            for condition in minimal["conditions"]
            for row in condition["replica_metrics"]
        ]
        verdict = "inconclusive" if min(oracle_values) < 0.85 else "not_supported_by_this_implementation"
        return verdict, "blocked_after_static_minimal"
    confirmation = phases.get("static_confirmation")
    if confirmation is None or not bool(confirmation.get("passed", False)):
        return "partially_supported", "blocked_after_static_confirmation"
    consolidation = phases.get("consolidation")
    if consolidation is None or not bool(consolidation.get("passed", False)):
        return "partially_supported", "blocked_after_partial_carry"
    return "supported", "complete"


def _write_figures(run_root: Path) -> tuple[Path, ...]:
    sources = [
        run_root / "static" / "minimal" / "seed-0" / condition / "raw_scores.npz"
        for condition in ("novel_leaf", "recurrent_leaf_1_8", "identical_regime")
    ]
    if any(not path.is_file() for path in sources):
        return ()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (condition, source) in enumerate(
        zip(("novel_leaf", "recurrent_leaf_1_8", "identical_regime"), sources, strict=True)
    ):
        with np.load(source, allow_pickle=False) as payload:
            margins = payload["focused_0_map_leaf"] - payload["focused_0_map_history"]
            matrix = payload["general_0_map_scores"]
            source_indices = payload["general_source_node_indices"]
            mean_matrix = np.stack(
                [np.mean(matrix[:, source_indices == index], axis=1) for index in range(5)],
                axis=1,
            )
        axes[0, column].hist(margins, bins=40)
        axes[0, column].axvline(0.0, color="black", linewidth=1)
        axes[0, column].set_title(f"{condition}: leaf minus history")
        axes[0, column].set_xlabel("MAP joint-score margin (nats)")
        image = axes[1, column].imshow(mean_matrix, aspect="auto")
        axes[1, column].set_title(f"{condition}: mean raw score matrix")
        axes[1, column].set_xlabel("held-out source node")
        axes[1, column].set_ylabel("candidate node")
        figure.colorbar(image, ax=axes[1, column], shrink=0.8)
    plots = run_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    output = plots / "static_map_scores.png"
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    plt.close(figure)
    atomic_write(output, buffer.getvalue())
    return (output,)


__all__ = ["CONDITION_DEFINITIONS", "write_result_report"]
