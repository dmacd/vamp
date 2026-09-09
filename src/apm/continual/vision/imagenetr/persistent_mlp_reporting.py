"""Authenticate and explain the MLP extension without importing training runtimes."""

from __future__ import annotations

from itertools import accumulate
import math
from pathlib import Path

from apm.continual.artifacts import file_sha256, load_canonical_json, record_sha256
from apm.continual.vision.imagenetr.persistent_affine_resources import ResourceWork


LABEL_MLP = "Persistent two-layer MLP + adaptive node LoRAs, H=4,096"
LABEL_ORACLE_MLP = "True-node oracle for persistent MLP H=4,096 (diagnostic)"
MLP_CONDITION_ID = "persistent_mlp_h4096"


def mlp_detail_cells(result: dict[str, object]) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Compare the paired H=4,096 heads at the selected diagnostic stages."""
    columns = ("Stage", "Nodes", "Affine acc.", "MLP acc.", "Affine NLL", "MLP NLL", "Affine oracle", "MLP oracle")
    cells = tuple(
        (str(row["stage"]), str(row["live_nodes"]))
        + tuple(
            f"{source['evaluation'][field]:.3f}" + ("%" if field != "nll" else "")
            for field in ("accuracy", "nll", "true_node_oracle_accuracy")
            for source in (prior, row)
        )
        for row, prior in zip(result["mlp"]["rows"], result["arms"]["4096"], strict=True)
        if row["stage"] in {1, 2, 4, 8, 16, 31, 32, 50}
    )
    return columns, cells


def load_mlp_extension(run: Path, reference: dict[str, object]) -> dict[str, object] | None:
    """Load a sealed complete extension and reject mismatched data or frontier history."""
    pointer_path = run / "reports/mlp_extension.json"
    if not pointer_path.is_file():
        return None
    pointer = load_canonical_json(pointer_path)
    if (
        pointer.get("schema_version") != "imagenetr50-persistent-mlp-report-extension-v1"
        or record_sha256({key: value for key, value in pointer.items() if key != "content_hash"}) != pointer.get("content_hash")
        or pointer.get("reference_result_hash") != reference["content_hash"]
    ):
        raise ValueError("MLP report pointer does not authenticate against the affine result")
    source_run = run.parents[2] / "persistent_mlp_v16/runs" / Path(pointer["run"]).name
    path = source_run / "evaluations/result.json"
    if file_sha256(path) != pointer["result_sha256"]:
        raise ValueError("MLP report source file changed")
    result = load_canonical_json(path)
    if (
        result.get("schema_version") != "imagenetr50-persistent-mlp-result-v1"
        or record_sha256({key: value for key, value in result.items() if key != "content_hash"}) != result.get("content_hash")
        or result.get("content_hash") != pointer["result_hash"]
        or result.get("reference_result_hash") != reference["content_hash"]
        or result.get("hidden_dimension") != 1024 or result.get("historical_capacity") != 4096
        or not result.get("source_hierarchy_unchanged")
        or not result.get("matched_replay_and_exposure")
        or len(result.get("rows", ())) != 50
    ):
        raise ValueError("MLP report source is incomplete or does not authenticate")
    for stage, (row, prior) in enumerate(zip(result["rows"], reference["arms"]["4096"], strict=True), 1):
        if (
            row["stage"] != stage
            or any(row[field] != prior[field] for field in ("node_hashes", "slots", "carry", "population"))
            or row["fit"]["image_presentations"] != prior["fit"]["image_presentations"]
        ):
            raise ValueError("MLP report source differs from the matched affine task stream")
    return result


def mlp_resource_rows(
    result: dict[str, object], base_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Charge the common source hierarchy and measured MLP adaptation separately."""
    if "mlp" not in result:
        return ()
    affine = tuple(row for row in base_rows if row["condition_id"] == "persistent_h4096")
    increments = tuple(
        ResourceWork(
            training_forward_images=prior["hierarchy_forward_images"] + row["fit"]["image_presentations"] * row["live_nodes"],
            recompute_forward_images=row["fit"]["image_presentations"] * row["live_nodes"],
            training_backward_images=prior["hierarchy_forward_images"] + row["fit"]["image_presentations"] * row["live_nodes"],
            training_wall_seconds=prior["hierarchy_wall_seconds"] + row["fit"]["wall_seconds"],
            hierarchy_forward_images=prior["hierarchy_forward_images"],
            hierarchy_wall_seconds=prior["hierarchy_wall_seconds"],
            adaptation_forward_images=row["fit"]["image_presentations"] * row["live_nodes"],
            adaptation_wall_seconds=row["fit"]["wall_seconds"],
            evaluation_forward_images=row["evaluation"]["examples"] * row["live_nodes"],
            evaluation_wall_seconds=row["evaluation"]["wall_seconds"],
        )
        for row, prior in zip(result["mlp"]["rows"], affine, strict=True)
    )
    return tuple(
        {
            "condition_id": MLP_CONDITION_ID, "stage": stage, **increment.as_record(),
            **{f"cumulative_{name}": value for name, value in cumulative.as_record().items()},
        }
        for stage, (increment, cumulative) in enumerate(zip(increments, accumulate(increments), strict=True), 1)
    )


def mlp_explanation(result: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """Describe the actual MLP outcome, controlled change, and remaining uncertainty."""
    if "mlp" not in result:
        return ()
    mlp = result["mlp"]
    rows, affine = mlp["rows"], result["arms"]["4096"]
    last, prior = rows[-1]["evaluation"], affine[-1]["evaluation"]
    accuracy_gap, nll_gap = last["accuracy"] - prior["accuracy"], last["nll"] - prior["nll"]
    fragmented = tuple(index for index in range(50) if (index + 1).bit_count() > 1)
    fragmented_gap = math.fsum(rows[index]["evaluation"]["accuracy"] - affine[index]["evaluation"]["accuracy"] for index in fragmented) / len(fragmented)
    return (
        ("Measured effect of the second layer",
         f"At task 50 the MLP reaches {last['accuracy']:.3f}% accuracy and {last['nll']:.4f} NLL, "
         f"a change of {accuracy_gap:+.3f} accuracy points and {nll_gap:+.4f} NLL relative to affine H=4,096. "
         f"Its mean accuracy change over the {len(fragmented)} fragmented stages is {fragmented_gap:+.3f} points. "
         f"Its final true-node diagnostic is {last['true_node_oracle_accuracy']:.3f}%."),
        ("Architecture and size",
         "Every input vector is the final 768-value latent from a live node's own base ViT plus adaptive rank-16 LoRA. "
         "The head is Linear(768 x live nodes, 1024), ReLU, then Linear(1024, 200). At five nodes it has "
         "4,138,184 head parameters, versus 768,200 for affine. Both have 6,635,520 trainable LoRA parameters. "
         "There is no skip connection, normalization, dropout, metadata, score input, or macro token."),
        ("Initialization and continuing state",
         "The first 400 hidden units split the 200 source-union logits into positive and negative parts; their signed "
         "output reproduces the source union. Another 624 units begin as random latent projections with zero output "
         "weights. Both dense matrices train freely. Surviving input blocks, hidden biases, old-class output rows, "
         "and AdamW moments persist. New node blocks and adapters initialize from the sealed source. The shared "
         "MLP output layer is not reset at consolidation."),
        ("What this comparison tests",
         "The MLP uses exactly the affine H=4,096 replay identities, augmentation/order seeds, four epochs, batch 64, "
         "learning rates, and LoRA carry/reset boundaries. Both joint curves and both affine arms are reused unchanged. "
         "This tests the larger nonlinear head under the existing optimization budget, not whether a separately tuned "
         "MLP has converged. It is one seed, with no test-selected checkpoint. Rank-matched joint IID does not match "
         "the MLP's extra parameters or multiple ViT paths."),
    )
