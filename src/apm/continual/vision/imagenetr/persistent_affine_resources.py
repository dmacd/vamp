"""Recover training costs, including reused source models, for the v15 report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import accumulate
import math
from pathlib import Path

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)


SOURCE_COST_SCHEMA = "imagenetr50-persistent-affine-source-costs-v1"
CONDITION_IDS = ("persistent_h4096", "joint_rank16", "joint_rank_matched")


def _source_node_cost(directory: Path, project_root: Path) -> dict[str, object]:
    """Authenticate source metadata and training counters without loading weights."""
    from apm.continual.vision.imagenetr.artifacts import node_artifact_from_record

    manifest = load_canonical_json(directory / "artifact.json")
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if record_sha256(manifest_core) != manifest["artifact_sha256"]:
        raise ValueError("source training artifact manifest changed")
    files = {row["path"]: row for row in manifest["files"]}
    for name in ("node.json", "training_metrics.json"):
        if file_sha256(directory / name) != files[name]["sha256"]:
            raise ValueError(f"source training evidence changed: {directory / name}")
    node = node_artifact_from_record(load_canonical_json(directory / "node.json"))
    metrics = load_canonical_json(directory / "training_metrics.json")
    if (
        metrics["optimizer_steps"] != node.training_optimizer_steps
        or metrics["image_presentations"] != 5 * node.represented_train_image_count
        or not math.isfinite(float(metrics["wall_seconds"]))
        or float(metrics["wall_seconds"]) <= 0
    ):
        raise ValueError("source node does not have the recorded five-pass training budget")
    return {
        "node_hash": node.content_hash,
        "level": node.level,
        "first_task": node.first_task,
        "last_task": node.last_task,
        "parent_hashes": list(node.parent_hashes),
        "train_examples": node.represented_train_image_count,
        "metrics": metrics,
        "source_directory": directory.relative_to(project_root).as_posix(),
        "artifact_manifest": manifest,
    }


def load_source_training_costs(run: Path, result: dict[str, object]) -> dict[str, object]:
    """Seal compact source costs once so the report can rebuild without model files."""
    target = run / "reports/source_training_costs.json"
    protocol = load_canonical_json(run / "protocol/protocol.json")
    if target.is_file():
        record = load_canonical_json(target)
        core = {key: value for key, value in record.items() if key != "content_hash"}
        if (
            record.get("schema_version") != SOURCE_COST_SCHEMA
            or record.get("content_hash") != record_sha256(core)
            or record.get("protocol_hash") != result["protocol_hash"]
            or record.get("hierarchy_complete_sha256") != protocol["source_hierarchy_complete_sha256"]
            or record["joint_final"]["node_hash"] != result["stage_matched_joint"][-1]["source_joint_node_hash"]
        ):
            raise ValueError("cached source cost identity changed")
        return record

    import yaml

    project_root = run.parents[4]
    config = load_canonical_json(run / "config_resolved.json")
    source_config = project_root / "configs/vision/imagenetr" / Path(config["source_config"]).name
    if file_sha256(source_config) != config["source_config_sha256"]:
        raise ValueError("source configuration changed while recovering training costs")
    source_root = yaml.safe_load(source_config.read_text())["paths"]["artifact_root"]
    source_run = project_root / source_root / "runs" / protocol["source_run_hash"]
    source_protocol_path = source_run / "protocol/protocol.json"
    if file_sha256(source_protocol_path) != protocol["source_protocol_sha256"]:
        raise ValueError("source protocol changed while recovering training costs")
    source_protocol = load_canonical_json(source_protocol_path)
    hierarchy = source_run / "hierarchies" / protocol["source_hierarchy_policy_hash"]
    complete_path = hierarchy / "complete_050.json"
    if file_sha256(complete_path) != protocol["source_hierarchy_complete_sha256"]:
        raise ValueError("source hierarchy changed while recovering training costs")
    complete = load_canonical_json(complete_path)
    nodes = tuple(
        _source_node_cost(path.parent, project_root)
        for path in sorted((hierarchy / "nodes").glob("*/node.json"))
    )
    if sorted(row["node_hash"] for row in nodes) != complete["node_hashes"]:
        raise ValueError("source costs omit a hierarchy node, including an intermediate parent")
    if len(nodes) != 97 or sum(row["level"] == 0 for row in nodes) != 50:
        raise ValueError("expected all 50 leaves and 47 full-union parents")
    primary_run = run.parents[2] / "runs" / source_protocol["sealed_run_hash"]
    joint_candidates = tuple((primary_run / "baselines/joint_iid_lora_r16").glob("*/node.json"))
    if len(joint_candidates) != 1:
        raise ValueError("expected one original offline joint-IID training artifact")
    joint = _source_node_cost(joint_candidates[0].parent, project_root)
    if joint["node_hash"] != result["stage_matched_joint"][-1]["source_joint_node_hash"]:
        raise ValueError("original joint-IID model differs from the reused stage-50 model")
    core = {
        "schema_version": SOURCE_COST_SCHEMA,
        "protocol_hash": result["protocol_hash"],
        "hierarchy_complete_sha256": protocol["source_hierarchy_complete_sha256"],
        "hierarchy_complete": complete,
        "hierarchy_nodes": sorted(nodes, key=lambda row: (row["last_task"], row["level"])),
        "joint_final": joint,
        "scope": "Original recorded training jobs charged once per condition, including reused models; excludes tuning and prior experiments.",
    }
    record = {**core, "content_hash": record_sha256(core)}
    atomic_write(target, canonical_json_bytes(record))
    return record


@dataclass(frozen=True)
class ResourceWork:
    """Image-path counters and wall time; no forward/backward FLOP equivalence implied."""

    training_forward_images: int = 0
    recompute_forward_images: int = 0
    training_backward_images: int = 0
    training_wall_seconds: float = 0.0
    hierarchy_forward_images: int = 0
    hierarchy_wall_seconds: float = 0.0
    adaptation_forward_images: int = 0
    adaptation_wall_seconds: float = 0.0
    evaluation_forward_images: int = 0
    evaluation_wall_seconds: float | None = 0.0

    def __add__(self, other: ResourceWork) -> ResourceWork:
        """Sum work while preserving missing evaluation timings as unknown."""
        left, right = asdict(self), asdict(other)
        return ResourceWork(**{
            field: None if left[field] is None or right[field] is None else left[field] + right[field]
            for field in left
        })

    def as_record(self) -> dict[str, int | float | None]:
        """Return separate pass counts plus the checkpoint-inclusive forward total."""
        return {
            **asdict(self),
            "training_forward_images_including_recompute": self.training_forward_images + self.recompute_forward_images,
        }


def resource_stage_rows(
    result: dict[str, object], sources: dict[str, object], activation_recomputation: bool
) -> tuple[dict[str, object], ...]:
    """Charge source jobs at subtree completion and multiply replay by every live node.

    Reused stage-50 rank-16 training has zero invocation counters in its source
    stage ledger. The original offline job supplies its real training cost.
    Rank-16 points reused by the aggregate-rank curve are charged once within
    that condition. Conditions are alternative algorithms, never summed together.
    """
    nodes = sources["hierarchy_nodes"]
    by_hash = {node["node_hash"]: node for node in nodes}
    projected: list[dict[str, object]] = []
    for condition in CONDITION_IDS:
        work: list[ResourceWork] = []
        for index, arm in enumerate(result["arms"]["4096"]):
            stage = index + 1
            joint = result["stage_matched_joint"][index]
            if arm["stage"] != stage or joint["stage"] != stage:
                raise ValueError("resource stages must be complete and ordered")
            if condition == "persistent_h4096":
                if len(arm["node_hashes"]) != arm["live_nodes"] or arm["live_nodes"] != stage.bit_count():
                    raise ValueError("resource multiplicity differs from the live frontier")
                if any(node_hash not in by_hash for node_hash in arm["node_hashes"]):
                    raise ValueError("resource sources omit a live frontier node")
                arriving = tuple(node for node in nodes if node["last_task"] + 1 == stage)
                hierarchy_images = sum(node["metrics"]["image_presentations"] for node in arriving)
                hierarchy_seconds = math.fsum(node["metrics"]["wall_seconds"] for node in arriving)
                adaptation_images = arm["live_nodes"] * arm["fit"]["image_presentations"]
                if arm["fit"]["image_presentations"] != arm["fit"]["epochs"] * arm["population"]["training_examples"]:
                    raise ValueError("adaptation image counters disagree with completed epochs")
                work.append(ResourceWork(
                    training_forward_images=hierarchy_images + adaptation_images,
                    recompute_forward_images=adaptation_images if activation_recomputation else 0,
                    training_backward_images=hierarchy_images + adaptation_images,
                    training_wall_seconds=hierarchy_seconds + arm["fit"]["wall_seconds"],
                    hierarchy_forward_images=hierarchy_images,
                    hierarchy_wall_seconds=hierarchy_seconds,
                    adaptation_forward_images=adaptation_images,
                    adaptation_wall_seconds=arm["fit"]["wall_seconds"],
                    evaluation_forward_images=arm["live_nodes"] * arm["evaluation"]["examples"],
                    evaluation_wall_seconds=arm["evaluation"]["wall_seconds"],
                ))
            else:
                row = joint if condition == "joint_rank16" else result["rank_matched_joint"][index]
                if row.get("reused_source_model", False):
                    original = sources["joint_final"]
                    if row["source_joint_node_hash"] != original["node_hash"]:
                        raise ValueError("reused joint model has no matching original training costs")
                    images = original["metrics"]["image_presentations"]
                    seconds = original["metrics"]["wall_seconds"]
                else:
                    images, seconds = row["image_presentations"], row["training_seconds"]
                if images != 5 * row["train_examples"]:
                    raise ValueError("joint training costs omit part of the five-pass budget")
                evaluation_seconds = (
                    joint["evaluation_seconds"]
                    if condition == "joint_rank16" or row.get("reused_stage_matched_rank16", False)
                    else None
                )
                work.append(ResourceWork(
                    training_forward_images=images,
                    training_backward_images=images,
                    training_wall_seconds=seconds,
                    evaluation_forward_images=joint["test_examples"],
                    evaluation_wall_seconds=evaluation_seconds,
                ))
        projected.extend(
            {
                "condition_id": condition,
                "stage": stage,
                **increment.as_record(),
                **{f"cumulative_{field}": value for field, value in cumulative.as_record().items()},
            }
            for stage, (increment, cumulative) in enumerate(zip(work, accumulate(work), strict=True), 1)
        )
    return tuple(projected)
