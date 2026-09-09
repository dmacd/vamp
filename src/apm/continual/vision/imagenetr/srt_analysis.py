"""Streaming audit and analysis of committed per-image rehearsal evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.vision.imagenetr.srt_config import presentation_budget
from apm.continual.vision.imagenetr.srt_evidence import read_event_chunk, read_sealed, trace_batches


GAP_EDGES = np.array([0, 1, *(2 ** power for power in range(1, 41)), np.inf])
GAP_FIELDS = ("step_gap", "presentation_gap", "stage_gap", "clock_gap", "interval_before", "lateness")


@dataclass(frozen=True, slots=True)
class ReplayAnalysis:
    """Compact projections, never a retained copy of the full event stream."""

    stages: tuple[dict[str, object], ...]
    samples: tuple[dict[str, object], ...]
    histograms: tuple[dict[str, object], ...]
    timelines: tuple[dict[str, object], ...]
    totals: dict[str, object]


def analyze_replay_job(root: Path, image_index: Path, condition: str) -> ReplayAnalysis:
    """Audit one complete stream while reducing small checkpoint-sized chunks."""
    result = read_sealed(root / "result.json", "imagenetr50-srt-job-result-v1")
    job = read_sealed(root / "job.json", "imagenetr50-srt-job-v1")
    index = pq.read_table(image_index).to_pylist()
    if (len(result["rows"]) != 50 or [row["image"] for row in index] != list(range(len(index)))
            or result["job_hash"] != job["content_hash"]
            or record_sha256([row["image_id"] for row in index]) != job["training_ids_hash"]):
        raise ValueError("SRT report requires all fifty stages and an ordered image index")
    stage_rows, timeline = [], []
    sample_counts = np.zeros(len(index), dtype=np.int64)
    historical_counts = np.zeros_like(sample_counts)
    current_counts = np.zeros_like(sample_counts)
    first_steps = np.zeros_like(sample_counts)
    last_steps = np.zeros_like(sample_counts)
    last_stages = np.zeros_like(sample_counts)
    last_due = np.zeros(len(index), dtype=object)
    last_quality = np.full(len(index), -1, dtype=np.int64)
    last_confidence = np.zeros(len(index), dtype=np.float64)
    last_clock = np.zeros(len(index), dtype=object)
    lateness_sums = np.zeros(len(index), dtype=np.float64)
    gap_lists = [[] for _ in index]
    histograms = defaultdict(lambda: np.zeros(len(GAP_EDGES) - 1, dtype=np.int64))
    selected_images = frozenset(
        item["image"] for stage in (1, 16, 31, 50)
        for item in sorted((item for item in index if item["arrival_stage"] == stage),
                           key=lambda item: sha256(f"srt-timeline-v1:{item['image_id']}".encode()).hexdigest())[:2]
    )
    presentations, steps, final_clock = 0, 0, 0
    cumulative_training, cumulative_evaluation = 0.0, 0.0
    for stage, row in enumerate(result["rows"], 1):
        sealed_stage = read_sealed(root / f"stages/{stage:03d}/result.json", "imagenetr50-srt-stage-v1")
        if sealed_stage != row or row["stage"] != stage:
            raise ValueError("SRT result and sealed stage history disagree")
        batches = trace_batches(root, row["trace_chunks"])
        if [batch["step"] for batch in batches] != list(range(steps + 1, steps + 1 + len(batches))):
            raise ValueError("SRT batch history has a gap or duplicated optimizer step")
        previous_steps = steps
        batch_map = {batch["step"]: batch for batch in batches}
        stage_counts = Counter()
        quality_counts = Counter()
        for chunk in row["trace_chunks"]:
            path = root / chunk["path"] / "events.parquet"
            if file_sha256(path) != chunk["events_sha256"]:
                raise ValueError("SRT report encountered an altered committed event chunk")
            events = read_event_chunk(path)
            if len(events) != chunk["event_count"]:
                raise ValueError("SRT chunk event count differs from its commit")
            per_step = Counter()
            per_step_old = Counter()
            unique = set()
            for event in events:
                image, step = event["image"], event["step"]
                if (not 0 <= image < len(index) or event["stage"] != stage
                        or index[image]["arrival_stage"] != event["arrival_stage"]
                        or event["arrival_stage"] > stage or (step, image) in unique):
                    raise ValueError("invalid, future, or duplicated image in SRT evidence")
                unique.add((step, image))
                if event["exposure"] != sample_counts[image] + 1:
                    raise ValueError("SRT per-image exposure ordinals are not consecutive")
                if sample_counts[image] and event["step_gap"] != step - last_steps[image]:
                    raise ValueError("recorded replay interval differs from actual optimizer history")
                if bool(event["kind"] == "introduction") != bool(sample_counts[image] == 0):
                    raise ValueError("initial image coverage differs from the review history")
                if event["kind"] == "historical" and event["arrival_stage"] >= stage:
                    raise ValueError("current-task work was mislabeled as historical replay")
                if result["method"] == "srt":
                    quality = sum(event["confidence"] >= threshold for threshold in result["policy"]["thresholds"])
                    if event["quality"] != quality or (event["lateness"] is not None and event["lateness"] < 0):
                        raise ValueError("SRT quality or due-only scheduling failed its event audit")
                    last_due[image], last_quality[image] = event["due_after"], quality
                    quality_counts[quality] += 1
                    if event["lateness"] is not None:
                        lateness_sums[image] += event["lateness"]
                sample_counts[image] += 1
                historical_counts[image] += int(event["kind"] == "historical")
                current_counts[image] += int(event["kind"] == "current_review")
                if not first_steps[image]:
                    first_steps[image] = step
                last_steps[image], last_stages[image] = step, stage
                last_clock[image], last_confidence[image] = event["clock"], event["confidence"]
                if event["step_gap"] is not None:
                    gap_lists[image].append(event["step_gap"])
                per_step[step] += 1
                per_step_old[step] += int(event["kind"] == "historical")
                stage_counts[event["kind"]] += 1
                if image in selected_images:
                    timeline.append({"condition": condition, "image_id": index[image]["image_id"], **event,
                                     "stage_position": stage - 1 + (step - previous_steps) / len(batches)})
            for step, count in per_step.items():
                if count != batch_map[step]["presentations"] or per_step_old[step] != batch_map[step]["historical_count"]:
                    raise ValueError("SRT event counts differ from the matched batch ledger")
            for field in GAP_FIELDS:
                for kind in ("historical", "current_review"):
                    relevant = tuple(event for event in events if event["kind"] == kind and event[field] is not None)
                    if not relevant:
                        continue
                    for previous_quality in ("all", *sorted({event["previous_quality"] for event in relevant if event["previous_quality"] is not None})):
                        values = [event[field] for event in relevant if previous_quality == "all" or event["previous_quality"] == previous_quality]
                        histograms[(field, kind, str(previous_quality))] += np.histogram(values, GAP_EDGES)[0]
        stage_presentations = sum(stage_counts.values())
        expected = presentation_budget(row["current_examples"], row["historical_examples_available"], result["capacity"])
        if (stage_presentations != expected or stage_presentations != row["fit"]["image_presentations"]
                or stage_counts["introduction"] != row["current_examples"] or len(batches) != row["fit"]["optimizer_steps"]):
            raise ValueError("SRT stage budget, first exposure coverage, or optimizer count failed audit")
        predictions_path = root / f"stages/{stage:03d}/predictions.parquet"
        if file_sha256(predictions_path) != row["predictions_sha256"]:
            raise ValueError("SRT held-out predictions changed after stage sealing")
        predictions = pq.read_table(predictions_path).to_pydict()
        accuracy = 100 * sum(first == second for first, second in zip(predictions["prediction"], predictions["label"], strict=True)) / len(predictions["label"])
        nll = math.fsum(predictions["nll"]) / len(predictions["nll"])
        if abs(accuracy - row["evaluation"]["accuracy"]) > 1e-10 or abs(nll - row["evaluation"]["nll"]) > 1e-6:
            raise ValueError("SRT test metrics cannot be reconstructed from saved predictions")
        presentations += stage_presentations
        steps += len(batches)
        cumulative_training += row["fit"]["wall_seconds"]
        cumulative_evaluation += row["evaluation"]["wall_seconds"]
        final_clock = row["scheduler"]["clock"]
        stage_rows.append({
            "condition": condition, "method": result["method"], "capacity": result["capacity"], "stage": stage,
            "accuracy": row["evaluation"]["accuracy"], "nll": row["evaluation"]["nll"],
            "current_examples": row["current_examples"], "historical_available": row["historical_examples_available"],
            "introductions": stage_counts["introduction"], "current_reviews": stage_counts["current_review"],
            "historical_reviews": stage_counts["historical"], "actual_historical_fraction": stage_counts["historical"] / stage_presentations,
            "actual_historical_fraction_after_introduction": stage_counts["historical"] / (stage_presentations - stage_counts["introduction"]),
            "target_historical_fraction_after_introduction": result["policy"]["historical_fraction"],
            "training_forwards": stage_presentations, "training_backwards": stage_presentations,
            "quality_forwards": 0, "recompute_forwards": 0, "optimizer_steps": len(batches),
            "mean_batch_size": stage_presentations / len(batches), "minimum_batch_size": min(batch["presentations"] for batch in batches),
            "full_batch_fraction": sum(batch["presentations"] == 64 for batch in batches) / len(batches),
            "cumulative_presentations": presentations, "cumulative_optimizer_steps": steps,
            "training_wall_seconds": row["fit"]["wall_seconds"], "cumulative_training_wall_seconds": cumulative_training,
            "evaluation_forwards": row["evaluation"]["examples"], "evaluation_wall_seconds": row["evaluation"]["wall_seconds"],
            "cumulative_evaluation_wall_seconds": cumulative_evaluation,
            "loader_wall_seconds": row["fit"]["loader_wall_seconds"], "scheduler_wall_seconds": row["fit"]["scheduler_wall_seconds"],
            "checkpoint_wall_seconds": row["fit"]["checkpoint_wall_seconds"], "peak_vram_bytes": row["fit"]["peak_vram_bytes"],
            "clock_advance": sum(batch["clock_advance"] for batch in batches), "final_clock": final_clock,
            "historical_due": row["scheduler"]["historical_due"] if result["method"] == "srt" else None,
            "current_due": row["scheduler"]["current_due"] if result["method"] == "srt" else None,
            "never_historically_replayed": row["scheduler"]["never_historically_replayed"],
            **{f"quality_{quality}_count": quality_counts[quality] if result["method"] == "srt" else None for quality in range(6)},
        })
    if (presentations != result["image_presentations"] or steps != result["optimizer_steps"]
            or not np.all(sample_counts > 0)):
        raise ValueError("SRT stream totals or complete first exposure coverage failed audit")
    samples = tuple({
        "condition": condition, **item, "presentations": int(sample_counts[image]),
        "historical_reviews": int(historical_counts[image]), "current_reviews": int(current_counts[image]),
        "first_step": int(first_steps[image]), "last_step": int(last_steps[image]), "last_stage": int(last_stages[image]),
        "mean_step_gap": float(np.mean(gap_lists[image])) if gap_lists[image] else None,
        "median_step_gap": float(np.median(gap_lists[image])) if gap_lists[image] else None,
        "p90_step_gap": float(np.quantile(gap_lists[image], .9)) if gap_lists[image] else None,
        "minimum_step_gap": min(gap_lists[image]) if gap_lists[image] else None,
        "maximum_step_gap": max(gap_lists[image]) if gap_lists[image] else None,
        "last_confidence": float(last_confidence[image]),
        "last_quality": int(last_quality[image]) if result["method"] == "srt" else None,
        "next_due_clock": int(last_due[image]) if result["method"] == "srt" else None,
        "mean_lateness": float(lateness_sums[image] / max(1, sample_counts[image] - 1)) if result["method"] == "srt" else None,
        "censored_wait_steps": int(steps - last_steps[image]), "next_review_censored": True,
        "terminal_overdue_ticks": int(max(0, final_clock - last_due[image])) if result["method"] == "srt" else None,
        "no_historical_review": bool(historical_counts[image] == 0),
    } for image, item in enumerate(index))
    histogram_rows = tuple({
        "condition": condition, "metric": field, "kind": kind, "previous_quality": quality,
        "lower": int(GAP_EDGES[position]), "upper": int(GAP_EDGES[position + 1]) if np.isfinite(GAP_EDGES[position + 1]) else None,
        "count": int(count),
    } for (field, kind, quality), counts in sorted(histograms.items()) for position, count in enumerate(counts))
    totals = {
        "condition": condition, "final_accuracy": result["rows"][-1]["evaluation"]["accuracy"],
        "final_nll": result["rows"][-1]["evaluation"]["nll"],
        "mean_stage_accuracy": math.fsum(row["evaluation"]["accuracy"] for row in result["rows"]) / 50,
        "training_presentations": presentations, "optimizer_steps": steps,
        "training_wall_seconds": cumulative_training, "evaluation_wall_seconds": cumulative_evaluation,
        "historical_reviews": int(historical_counts.sum()), "current_reviews": int(current_counts.sum()),
        "unique_historically_replayed": int(np.count_nonzero(historical_counts)),
        "never_historically_replayed": int(np.count_nonzero(historical_counts == 0)),
        "older_images_never_historically_replayed": int(sum(historical_counts[image] == 0 for image, item in enumerate(index) if item["arrival_stage"] < 50)),
        "peak_vram_bytes": max(row["fit"]["peak_vram_bytes"] for row in result["rows"]),
        "clock_advance": sum(row["clock_advance"] for row in stage_rows),
        "terminal_overdue_images": int(np.count_nonzero(last_due < final_clock)) if result["method"] == "srt" else None,
    }
    return ReplayAnalysis(tuple(stage_rows), samples, histogram_rows, tuple(timeline), totals)
