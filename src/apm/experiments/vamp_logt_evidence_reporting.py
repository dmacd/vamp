"""Plain-language result documents and plots for LogT NCE/TRE evidence routing."""

from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path
import json
import os
from collections.abc import Mapping

import numpy as np

from apm.continual.artifacts import atomic_write, publish_immutable_json
from apm.experiments.vamp_logt_evidence_config import VampLogTEvidenceConfig


CONDITION_DEFINITIONS = {
    "direct_nce": (
        "Direct NCE trains one full-capacity raw-image discriminator per active temporal "
        "node to separate that node's lightly corrupted replay distribution from the one "
        "shared configured raw-image reference, and it routes to the largest single "
        "discriminator logit."
    ),
    "tre": (
        "TRE trains the same full-capacity raw-image architecture with a fixed bridge index, "
        "learns balanced density ratios between consecutive coordinate-replacement waymarks, "
        "sums those logits, and routes to the active node with the largest sum."
    ),
    "oracle_node": (
        "The oracle-node condition evaluates every active adapter with the true digit target "
        "and chooses the adapter with the smallest classification loss, so it is a diagnostic "
        "upper bound and not an available task-free routing rule."
    ),
    "vamp_af": (
        "The recorded VAMP-AF condition uses the sealed PCA-median spatial address tree and "
        "its learned top-two-layer adapters from the earlier three-seed experiment."
    ),
    "global_replay": (
        "The recorded global-replay condition maintains one top-two-layer adapter trained "
        "online with replay drawn from every example seen so far."
    ),
    "joint_iid": (
        "The recorded joint-IID condition trains one top-two-layer adapter on shuffled data "
        "with the same total number of presentations as the online comparison."
    ),
    "oracle_context": (
        "The recorded oracle-context condition uses the known MNIST context to select one "
        "independently trained context adapter, so it is not a task-free method."
    ),
    "frozen_base": (
        "The recorded frozen-base condition applies the shared CNN without any adapter update."
    ),
    "oracle_leaf": (
        "The recorded oracle-leaf diagnostic evaluates every final VAMP-AF leaf with the true "
        "target and chooses the leaf with the smallest classification loss."
    ),
}


def write_result_report(
    run_root: Path,
    config: VampLogTEvidenceConfig,
    baseline_summary: Mapping[str, object],
    phases: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Write the same complete-sentence experiment account as Markdown and standalone HTML."""
    expected = ("calibration", "static", "consolidation", "online")
    failed = next(
        (
            phase
            for phase in expected
            if phase in phases and phases[phase].get("passed") is False
        ),
        None,
    )
    missing = next((phase for phase in expected if phase not in phases), None)
    status = "complete" if not failed and not missing else f"blocked_after_{failed or missing}"
    summary: dict[str, object] = {
        "baseline": dict(baseline_summary),
        "completed_phases": list(phases),
        "condition_definitions": CONDITION_DEFINITIONS,
        "evidence_reference": config.evidence.reference,
        "phases": {name: dict(value) for name, value in phases.items()},
        "schema_version": "vamp-logt-nce-tre-result-v1",
        "status": status,
    }
    publish_immutable_json(run_root / "summary.json", summary)
    plot_paths = _write_plots(run_root / "plots", config, phases)
    markdown = _markdown_report(config, baseline_summary, phases, status, plot_paths)
    atomic_write(run_root / "HANDOFF.md", markdown.encode("utf-8"))
    atomic_write(run_root / "report.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "report.html",
        _html_report(config, baseline_summary, phases, status, plot_paths, summary).encode(
            "utf-8"
        ),
    )
    return summary


def _markdown_report(
    config: VampLogTEvidenceConfig,
    baseline: Mapping[str, object],
    phases: Mapping[str, Mapping[str, object]],
    status: str,
    plot_paths: tuple[Path, ...],
) -> str:
    definition_lines = "\n".join(
        f"- `{name}`: {definition}" for name, definition in CONDITION_DEFINITIONS.items()
    )
    calibration_text = _calibration_markdown(phases.get("calibration"))
    static_text = _static_markdown(phases.get("static"))
    consolidation_text = _consolidation_markdown(phases.get("consolidation"))
    online_text = _online_markdown(phases.get("online"), baseline)
    plot_lines = "\n\n".join(
        f"![{path.stem.replace('_', ' ')}](plots/{path.name})" for path in plot_paths
    )
    return f"""# NCE/TRE evidence routing for LogT-VAMP on MNIST

## Outcome

The experiment status is `{status}`. A blocked status means a preregistered gate failed and later phases were deliberately not run; it does not mean the runner crashed. The selected TRE schedule, if static routing passed, was chosen once before consolidation or online evaluation.

This experiment uses 500 examples per level-zero block, 100 blocks across five blocked contexts, a true binary counter with at most one live node per level, de-novo top-two-layer adapters, and de-novo raw-image evidence models. {_reference_description(config)} The near-data endpoint replaces each pixel with probability 1/784, and every TRE schedule linearly increases that probability to one.

## Conditions

{definition_lines}

<details>
<summary>Normalized estimator calibration</summary>

{calibration_text}
</details>

<details>
<summary>Static 63-block routing and K selection</summary>

{static_text}
</details>

<details>
<summary>Block-64 consolidation control</summary>

{consolidation_text}
</details>

<details>
<summary>Full 100-block online comparison</summary>

{online_text}
</details>

## Figures

{plot_lines or 'No downstream figures were produced because the experiment stopped at an earlier gate.'}

## Interpretation boundary

Digit labels are used only to train and evaluate adapters and to construct the diagnostic oracle. Context identifiers are used only to build the frozen benchmark stream and to stratify diagnostics. Neither labels, contexts, PCA features, frozen CNN features, nor adapter responses enter an evidence-model input. The active reference specification is `{_specification_path(config)}`.
"""


def _calibration_markdown(summary: Mapping[str, object] | None) -> str:
    if summary is None:
        return "This phase was not reached."
    gates = summary["gates"]
    return (
        f"The normalized calibration passed: `{summary['passed']}`. TRE's mean log-ratio "
        f"RMSE was {float(summary['tre_mean_rmse_nats']):.4f} nats, direct NCE's mean RMSE "
        f"was {float(summary['direct_mean_rmse_nats']):.4f} nats, the maximum absolute TRE "
        f"bias was {float(summary['tre_maximum_absolute_signed_bias_nats']):.4f} nats, and "
        f"the largest independent-training disagreement was "
        f"{float(summary['interseed_tre_rmse_nats']):.4f} nats. "
        + _gate_sentences(gates)
    )


def _static_markdown(summary: Mapping[str, object] | None) -> str:
    if summary is None:
        return "This phase was not reached."
    table = [
        "| K | maximum adjacent accuracy | minimum replica route agreement | maximum oracle gap | passed |",
        "|---:|---:|---:|---:|:---:|",
        *(
            f"| {int(row['bridges'])} | {float(row['maximum_adjacent_balanced_accuracy']):.4f} | "
            f"{float(row['minimum_independent_route_agreement']):.4f} | "
            f"{float(row['maximum_classifier_oracle_gap']):.4f} | {bool(row['passed'])} |"
            for row in summary["candidate_results"]
        ),
    ]
    selected = summary.get("selected_tre_bridges")
    return (
        f"Static routing passed: `{summary['passed']}`. The smallest schedule satisfying "
        f"every gate was K={selected if selected is not None else 'none'}. Balanced adjacent "
        "accuracy measures how easily a held-out adjacent pair can still be separated; values "
        "near one indicate saturation, while the 0.90 gate requires meaningful overlap. "
        "Replica route agreement is the fraction of the same held-out images sent to the same "
        "node by two independently initialized evidence banks. The oracle gap is oracle-node "
        "classification accuracy minus evidence-routed classification accuracy.\n\n"
        + "\n".join(table)
    )


def _consolidation_markdown(summary: Mapping[str, object] | None) -> str:
    if summary is None:
        return "This phase was not reached."
    return (
        f"Consolidation passed: `{summary['passed']}`. Normal consolidation and its independent "
        "de-novo control both use the complete child-union replay, identical K and optimizer "
        "budgets, and different initializations. A raw-score difference is measured in nats "
        "before routing. Route agreement measures whether replacing only the normal parent "
        "with its independent twin changes the chosen node. The classifier gap measures the "
        "absolute routed-accuracy change. The loss difference compares held-out balanced NCE "
        "losses, and the level slope tests whether the normal-minus-control score offset grows "
        "systematically with consolidation depth. "
        + _gate_sentences(summary["gates"])
    )


def _online_markdown(
    summary: Mapping[str, object] | None,
    baseline: Mapping[str, object],
) -> str:
    baseline_means = baseline["main_mean_accuracy"]
    if summary is None:
        return (
            "This phase was not reached. The unchanged recorded baselines remain VAMP-AF "
            f"{float(baseline_means['af']):.4f}, global replay "
            f"{float(baseline_means['global_replay']):.4f}, joint IID "
            f"{float(baseline_means['joint_iid']):.4f}, oracle context "
            f"{float(baseline_means['oracle_context']):.4f}, and frozen base "
            f"{float(baseline_means['frozen_base']):.4f}."
        )
    means = summary["final_mean_accuracy"]
    return (
        f"Across the three full streams, direct NCE finished at {float(means['direct_nce']):.4f}, "
        f"TRE finished at {float(means['tre']):.4f}, and the label-aware node oracle finished "
        f"at {float(means['oracle_node']):.4f}. The unchanged recorded comparison means are "
        f"VAMP-AF {float(baseline_means['af']):.4f}, global replay "
        f"{float(baseline_means['global_replay']):.4f}, joint IID "
        f"{float(baseline_means['joint_iid']):.4f}, oracle context "
        f"{float(baseline_means['oracle_context']):.4f}, frozen base "
        f"{float(baseline_means['frozen_base']):.4f}, and VAMP-AF oracle leaf "
        f"{float(baseline['main_mean_oracle_leaf_accuracy']):.4f}. Routing regret is the "
        "evidence-routed classification loss minus the minimum loss available from any active "
        "adapter on the same example, measured in nats."
    )


def _gate_sentences(gates: Mapping[str, object]) -> str:
    return " ".join(
        f"The `{name}` gate {'passed' if bool(value) else 'failed'}."
        for name, value in gates.items()
    )


def _write_plots(
    directory: Path,
    config: VampLogTEvidenceConfig,
    phases: Mapping[str, Mapping[str, object]],
) -> tuple[Path, ...]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vamp-logt-matplotlib")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    static = phases.get("static")
    if static is not None:
        candidates = static["candidate_results"]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        bridges = [int(row["bridges"]) for row in candidates]
        series = (
            ("maximum_adjacent_balanced_accuracy", "maximum adjacent accuracy", 0.9),
            ("minimum_independent_route_agreement", "minimum replica route agreement", 0.9),
            ("maximum_classifier_oracle_gap", "maximum classifier oracle gap", 0.1),
        )
        for axis, (field, title, threshold) in zip(axes, series):
            axis.plot(bridges, [float(row[field]) for row in candidates], marker="o")
            axis.axhline(threshold, color="tab:red", linestyle="--", label="gate")
            axis.set(xlabel="fixed bridge count K", ylabel=title, xticks=bridges)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle("Static TRE schedule selection")
        figure.tight_layout()
        path = directory / "static_schedule_selection.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)
    consolidation = phases.get("consolidation")
    if consolidation is not None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for result in consolidation["seed_results"]:
            rows = result["merge_results"]
            axis.plot(
                [int(row["level"]) for row in rows],
                [float(row["mean_signed_raw_score_difference_nats"]) for row in rows],
                marker="o",
                label=f"seed {result['stream_seed']}",
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(
            xlabel="merged parent level",
            ylabel="normal minus control score (nats)",
            title="Consolidation score-offset check",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = directory / "consolidation_score_offsets.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)
    online = phases.get("online")
    if online is not None:
        figure, axis = plt.subplots(figsize=(9, 5))
        rows = [row for result in online["seed_results"] for row in result["evaluations"]]
        for condition in ("direct_nce", "tre", "oracle_node"):
            for seed in config.online.stream_seeds:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["condition"] == condition and int(row["stream_seed"]) == seed
                    ),
                    key=lambda row: int(row["processed_blocks"]),
                )
                axis.plot(
                    [int(row["processed_blocks"]) for row in selected],
                    [float(row["accuracy"]) for row in selected],
                    marker="o",
                    alpha=0.65,
                    label=f"{condition}, seed {seed}",
                )
        axis.set(
            xlabel="processed 500-example blocks",
            ylabel="accuracy over contexts seen so far",
            title="Online LogT evidence routing",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=3)
        figure.tight_layout()
        path = directory / "online_accuracy.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)

        figure, axes = plt.subplots(1, len(config.online.stream_seeds), figsize=(14, 4.5), sharey=True)
        for axis, result in zip(np.atleast_1d(axes), online["seed_results"]):
            intervals = result["active_intervals"]
            for index, interval in enumerate(intervals):
                axis.barh(
                    index,
                    int(interval["last_block"]) - int(interval["first_block"]) + 1,
                    left=int(interval["first_block"]),
                )
            axis.set(
                xlabel="stream block",
                title=f"seed {result['stream_seed']}",
                yticks=range(len(intervals)),
                yticklabels=[f"level {row['level']}" for row in intervals],
            )
            axis.grid(axis="x", alpha=0.25)
        figure.suptitle("Final disjoint LogT frontier")
        figure.tight_layout()
        path = directory / "final_logt_frontier.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def _html_report(
    config: VampLogTEvidenceConfig,
    baseline: Mapping[str, object],
    phases: Mapping[str, Mapping[str, object]],
    status: str,
    plot_paths: tuple[Path, ...],
    summary: Mapping[str, object],
) -> str:
    definitions = "".join(
        f"<li><code>{escape(name)}</code>: {escape(text)}</li>"
        for name, text in CONDITION_DEFINITIONS.items()
    )
    sections = "".join(
        f"<details open><summary>{escape(title)}</summary><p>{escape(text)}</p></details>"
        for title, text in (
            ("Normalized estimator calibration", _calibration_markdown(phases.get("calibration"))),
            ("Static routing and K selection", _static_markdown(phases.get("static"))),
            ("Consolidation control", _consolidation_markdown(phases.get("consolidation"))),
            ("Full online comparison", _online_markdown(phases.get("online"), baseline)),
        )
    )
    images = "".join(
        f'<figure><img alt="{escape(path.stem)}" src="data:image/png;base64,'
        f'{b64encode(path.read_bytes()).decode("ascii")}"><figcaption>'
        f'{escape(path.stem.replace("_", " "))}</figcaption></figure>'
        for path in plot_paths
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NCE/TRE LogT MNIST result</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#17202a}}
h1,h2{{color:#154360}} details{{margin:1rem 0;border:1px solid #ccd6dd;border-radius:8px;padding:.8rem}}
summary{{font-weight:700;cursor:pointer}} code,pre{{background:#f4f6f7}} pre{{padding:1rem;overflow:auto}}
img{{max-width:100%;height:auto}} figure{{margin:2rem 0}} figcaption{{text-align:center;color:#566573}}
</style></head><body>
<h1>NCE/TRE evidence routing for LogT-VAMP on MNIST</h1>
<p><strong>Status:</strong> {escape(status)}. A blocked status records a failed preregistered gate and intentionally omits later phases.</p>
<p>{escape(_reference_description(config))}</p>
<h2>Conditions</h2><ul>{definitions}</ul>{sections}
<h2>Figures</h2>{images or '<p>No downstream figures were produced.</p>'}
<details><summary>Machine-readable summary</summary><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></details>
</body></html>"""


def _reference_description(config: VampLogTEvidenceConfig) -> str:
    if config.evidence.reference == "discrete_uniform_uint8":
        return (
            "The shared reference is independent discrete uniform noise over pixel values "
            "from 0 through 255."
        )
    if config.evidence.reference == "frozen_base_training_images_uint8":
        return (
            "The shared reference is the uniform empirical distribution over all 60,000 "
            "original, unrotated MNIST training images used to train the authenticated "
            "frozen CNN; each waymark draws one complete donor image, and complete "
            "replacement is exactly that distribution."
        )
    raise ValueError("reporting received an unsupported evidence reference")


def _specification_path(config: VampLogTEvidenceConfig) -> str:
    if config.evidence.reference == "frozen_base_training_images_uint8":
        return "docs/NCE_TRE_BASE_REFERENCE.md"
    return "docs/Codex Handoff_ NCE-TRE Evidence Routing for LogT-VAMP on MNIST.md"


__all__ = ["CONDITION_DEFINITIONS", "write_result_report"]
