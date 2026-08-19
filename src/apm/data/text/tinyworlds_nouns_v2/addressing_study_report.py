"""Deterministic CSV, SVG, Graphviz, Markdown, and HTML study reports."""

from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
from html import escape
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedStoryStore
from apm.data.text.tinyworlds_nouns_v2.addressing_study import (
    AddressingStudyInputs,
    load_timing_ledger,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    EBT_ENTROPY_PENALTY,
    EBT_LEARNING_RATE,
    EBT_STEPS,
    EBT_TEMPERATURE,
    HOPFIELD_BETA,
    KEY_SCHEMES,
    REPORT_FORMAT,
    STUDY_MANIFEST_FORMAT,
    TOP8_ACCURACY_NONINFERIORITY_MARGIN,
    TOP8_NLL_NONINFERIORITY_MARGIN,
    canonical_json_bytes,
    mean,
    record_sha256,
)
from apm.memory.graph import path_incidence_matrix


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


def publish_addressing_study_report(
    inputs: AddressingStudyInputs,
    retrieval_contract: dict[str, object],
    ebt_contract: dict[str, object],
    retrieval_path: str | Path,
    ebt_path: str | Path,
    timing_path: str | Path,
    output_directory: str | Path,
    *,
    parity: dict[str, object],
    runtimes: dict[str, float],
    allocator: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Publish deterministic analysis exports and separate standalone reports."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    retrieval_rows = tuple(_jsonl_rows(Path(retrieval_path)))
    ebt_rows = tuple(_jsonl_rows(Path(ebt_path)))
    timing_rows = load_timing_ledger(
        timing_path,
        str(ebt_contract["contract_sha256"]),
    )
    measured_analysis = analyze_study_rows(
        inputs,
        retrieval_rows,
        ebt_rows,
        timing_rows,
        runtimes,
    )
    bindings = dict(retrieval_contract["bindings"])
    analysis = {
        **measured_analysis,
        "provenance": {
            **measured_analysis["provenance"],
            "addressing_key_artifact_sha256": bindings[
                "addressing_key_artifact_sha256"
            ],
            "ebt_contract_sha256": ebt_contract["contract_sha256"],
            "retrieval_contract_sha256": retrieval_contract["contract_sha256"],
        },
    }
    _write_analysis_exports(output, analysis)
    cost_svg = output / "cumulative-addressing-cost.svg"
    quality_svg = output / "final-checkpoint-quality-latency.svg"
    _atomic_write(
        cost_svg,
        render_cumulative_cost_svg(analysis["stage_costs"]).encode("utf-8"),
    )
    _atomic_write(
        quality_svg,
        render_quality_latency_svg(analysis["experiment_1"]).encode("utf-8"),
    )
    graph_paths = tuple(
        publish_inclusion_graph(
            inputs,
            retrieval_rows,
            width,
            output,
        )
        for width in (4, 8)
    )
    analysis_record = {
        "analysis": analysis,
        "allocator": allocator,
        "ebt_contract_sha256": ebt_contract["contract_sha256"],
        "format": REPORT_FORMAT,
        "parity": parity,
        "retrieval_contract_sha256": retrieval_contract["contract_sha256"],
        "runtimes": runtimes,
    }
    analysis_path = output / "analysis.json"
    _atomic_write(
        analysis_path,
        canonical_json_bytes(
            {**analysis_record, "analysis_sha256": record_sha256(analysis_record)}
        ),
    )
    markdown_path = output / "report.md"
    html_path = output / "report.html"
    _atomic_write(
        markdown_path,
        render_markdown_report(analysis, parity, allocator, runtimes).encode("utf-8"),
    )
    embedded_svgs = {
        path.name: path.read_text(encoding="utf-8")
        for path in (cost_svg, quality_svg, *(item[1] for item in graph_paths))
    }
    _atomic_write(
        html_path,
        render_html_report(
            analysis,
            parity,
            allocator,
            runtimes,
            embedded_svgs,
        ).encode("utf-8"),
    )
    source_artifact_paths = tuple(
        output / relative
        for relative in (
            "allocator.json",
            "ebt-contract.json",
            "ebt.jsonl",
            "execution-times.json",
            "keys/keys.safetensors",
            "keys/manifest.json",
            "parity.json",
            "retrieval-contract.json",
            "retrieval.jsonl",
            "timing.jsonl",
        )
    )
    if any(not path.is_file() for path in source_artifact_paths):
        missing = tuple(path.name for path in source_artifact_paths if not path.is_file())
        raise FileNotFoundError(f"standalone study artifacts are missing: {missing}")
    artifact_paths = (
        *source_artifact_paths,
        analysis_path,
        markdown_path,
        html_path,
        output / "aggregate.csv",
        output / "per-task.csv",
        output / "timing.csv",
        output / "cost.csv",
        cost_svg,
        quality_svg,
        *(path for pair in graph_paths for path in pair),
    )
    manifest_core = {
        "artifacts": {
            path.relative_to(output).as_posix(): _file_sha256(path)
            for path in sorted(artifact_paths)
        },
        "ebt_contract_sha256": ebt_contract["contract_sha256"],
        "format": STUDY_MANIFEST_FORMAT,
        "retrieval_contract_sha256": retrieval_contract["contract_sha256"],
        "schema_version": 1,
    }
    manifest_path = output / "manifest.json"
    _atomic_write(
        manifest_path,
        canonical_json_bytes(
            {**manifest_core, "manifest_sha256": record_sha256(manifest_core)}
        ),
    )
    return markdown_path, html_path, manifest_path


def analyze_study_rows(
    inputs: AddressingStudyInputs,
    retrieval_rows: tuple[dict[str, object], ...],
    ebt_rows: tuple[dict[str, object], ...],
    timing_rows: tuple[dict[str, object], ...],
    runtimes: dict[str, float],
) -> dict[str, object]:
    """Compute preregistered aggregate, task, timing, cost, and bootstrap results."""
    retrieval_aggregate = tuple(
        _retrieval_summary(
            scheme,
            tuple(row for row in retrieval_rows if row["scheme"] == scheme),
        )
        for scheme in KEY_SCHEMES
    )
    method_specs = tuple(
        (str(row["scheme"]), str(row["mode"]), int(row["candidate_width"]))
        for row in ebt_rows
    )
    unique_methods = tuple(dict.fromkeys(method_specs))
    retrieval_by_scheme = {
        str(row["scheme"]): row for row in retrieval_aggregate
    }
    ebt_aggregate = tuple(
        {
            **_ebt_summary(
                scheme,
                mode,
                width,
                tuple(
                    row
                    for row in ebt_rows
                    if (
                        row["scheme"],
                        row["mode"],
                        row["candidate_width"],
                    )
                    == (scheme, mode, width)
                ),
                timing_rows,
            ),
            "true_node_recall_at_4": retrieval_by_scheme[scheme]["top_4_recall"],
            "true_node_recall_at_8": retrieval_by_scheme[scheme]["top_8_recall"],
        }
        for scheme, mode, width in unique_methods
    )
    per_task = tuple(
        _per_task_summary(task_id, scheme, mode, width, retrieval_rows, ebt_rows)
        for task_id in inputs.partition.task_ids
        for scheme, mode, width in unique_methods
    )
    retrieval_bootstrap = _retrieval_bootstrap(retrieval_rows)
    ebt_bootstrap = _ebt_bootstrap(ebt_rows)
    experiment_1 = tuple(
        row
        for row in ebt_aggregate
        if row["scheme"] == "canonical_full_centroid"
        and (
            row["mode"] == "dense_all"
            or int(row["candidate_width"]) in (4, 8)
        )
    )
    dense = next(row for row in experiment_1 if row["mode"] == "dense_all")
    top_8 = next(
        row
        for row in experiment_1
        if row["mode"] == "compact" and row["candidate_width"] == 8
    )
    noninferiority = {
        "accuracy_loss": float(dense["route_accuracy"])
        - float(top_8["route_accuracy"]),
        "accuracy_margin": TOP8_ACCURACY_NONINFERIORITY_MARGIN,
        "accuracy_pass": float(dense["route_accuracy"])
        - float(top_8["route_accuracy"])
        <= TOP8_ACCURACY_NONINFERIORITY_MARGIN,
        "story_nll_increase": float(top_8["story_mean_nll"])
        - float(dense["story_mean_nll"]),
        "story_nll_margin": TOP8_NLL_NONINFERIORITY_MARGIN,
        "story_nll_pass": float(top_8["story_mean_nll"])
        - float(dense["story_mean_nll"])
        <= TOP8_NLL_NONINFERIORITY_MARGIN,
    }
    return {
        "ebt_aggregate": ebt_aggregate,
        "ebt_bootstrap": ebt_bootstrap,
        "experiment_1": experiment_1,
        "noninferiority": noninferiority,
        "per_task": per_task,
        "provenance": {
            "base_parameter_checksum": inputs.selected_base.reference.parameter_checksum,
            "base_training_sha256": inputs.selected_base.training_sha256,
            "canonical_artifact_hashes": dict(inputs.canonical_hashes),
            "canonical_ledger_hashes": dict(inputs.canonical_ledger_hashes),
            "canonical_run_sha256": inputs.canonical_run["run_sha256"],
            "partition_sha256": inputs.partition.partition_sha256,
            "vamp_tensor_checksum": inputs.adaptation.tensor_checksum,
        },
        "retrieval_aggregate": retrieval_aggregate,
        "retrieval_bootstrap": retrieval_bootstrap,
        "runtimes": runtimes,
        "stage_costs": _stage_costs(inputs),
        "timing": timing_rows,
    }


def _retrieval_summary(
    scheme: str,
    rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if len(rows) != 4_440:
        raise ValueError(f"retrieval scheme {scheme} does not contain 4,440 rows")
    return {
        "mean_entropy": mean(float(row["entropy"]) for row in rows),
        "mean_hopfield_dot_products": 25.0,
        "mean_margin": mean(float(row["score_margin"]) for row in rows),
        "mean_model_forward_equivalent_prefix_tokens": mean(
            float(row["prefix_token_count"]) for row in rows
        ),
        "row_count": len(rows),
        "scheme": scheme,
        "top_1_recall": mean(float(bool(row["top_1_hit"])) for row in rows),
        "top_4_recall": mean(float(bool(row["top_4_hit"])) for row in rows),
        "top_8_recall": mean(float(bool(row["top_8_hit"])) for row in rows),
    }


def _ebt_summary(
    scheme: str,
    mode: str,
    width: int,
    rows: tuple[dict[str, object], ...],
    timing_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if len(rows) != 4_440:
        raise ValueError(f"EBT method {(scheme, mode, width)} lacks 4,440 rows")
    shape_counts = Counter(
        (
            row["mode"],
            row["candidate_width"],
            row["prefix_width_bucket"],
            row["physical_edge_capacity"],
        )
        for row in rows
    )
    timing_by_shape = {
        (
            row["mode"],
            row["candidate_width"],
            row["prefix_width_bucket"],
            row["physical_edge_capacity"],
        ): row
        for row in timing_rows
    }
    if not set(shape_counts) <= set(timing_by_shape):
        raise ValueError("EBT aggregate is missing synchronized shape timings")
    weighted_latency = math.fsum(
        count * float(timing_by_shape[shape]["warm_kernel_mean_seconds"])
        for shape, count in shape_counts.items()
    ) / len(rows)
    return {
        "candidate_width": width,
        "cold_compilation_seconds": math.fsum(
            float(timing_by_shape[shape]["cold_compile_seconds"])
            for shape in shape_counts
        ),
        "mean_active_lora_edge_evaluations": mean(
            float(row["active_lora_edge_evaluations"]) for row in rows
        ),
        "mean_final_entropy": mean(float(row["final_entropy"]) for row in rows),
        "mean_final_margin": mean(float(row["final_margin"]) for row in rows),
        "mean_gathered_edge_count": mean(
            float(row["gathered_edge_count"]) for row in rows
        ),
        "mean_hopfield_dot_products": mean(
            float(row["hopfield_dot_products"]) for row in rows
        ),
        "mean_model_forward_equivalent_prefix_tokens": mean(
            float(row["model_forward_equivalent_prefix_tokens"]) for row in rows
        ),
        "mean_oracle_regret": mean(float(row["oracle_regret"]) for row in rows),
        "mean_physical_edge_capacity": mean(
            float(row["physical_edge_capacity"]) for row in rows
        ),
        "mean_selected_path_edge_count": mean(
            float(row["selected_path_edge_count"]) for row in rows
        ),
        "mode": mode,
        "route_accuracy": mean(float(bool(row["oracle_match"])) for row in rows),
        "row_count": len(rows),
        "scheme": scheme,
        "story_mean_nll": mean(float(row["suffix_mean_nll"]) for row in rows),
        "token_mean_nll": math.fsum(float(row["suffix_total_nll"]) for row in rows)
        / math.fsum(float(row["suffix_token_count"]) for row in rows),
        "warm_gpu_latency_seconds": weighted_latency,
        "warm_throughput_examples_per_second": 8.0 / weighted_latency,
    }


def _per_task_summary(
    task_id: str,
    scheme: str,
    mode: str,
    width: int,
    retrieval_rows: tuple[dict[str, object], ...],
    ebt_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    method_rows = tuple(
        row
        for row in ebt_rows
        if (
            row["task_noun"],
            row["scheme"],
            row["mode"],
            row["candidate_width"],
        )
        == (task_id, scheme, mode, width)
    )
    retrieval = tuple(
        row
        for row in retrieval_rows
        if row["task_noun"] == task_id and row["scheme"] == scheme
    )
    if not method_rows or not retrieval:
        raise ValueError("per-task report cell has no rows")
    confusion = Counter(int(row["selected_node_index"]) for row in method_rows)
    return {
        "candidate_width": width,
        "confusion_counts": {
            str(index): confusion.get(index, 0) for index in range(25)
        },
        "mean_final_entropy": mean(
            float(row["final_entropy"]) for row in method_rows
        ),
        "mean_final_margin": mean(
            float(row["final_margin"]) for row in method_rows
        ),
        "mean_retrieval_entropy": mean(
            float(row["entropy"]) for row in retrieval
        ),
        "mean_retrieval_margin": mean(
            float(row["score_margin"]) for row in retrieval
        ),
        "mode": mode,
        "route_accuracy": mean(float(bool(row["oracle_match"])) for row in method_rows),
        "scheme": scheme,
        "story_count": len(method_rows),
        "story_mean_nll": mean(float(row["suffix_mean_nll"]) for row in method_rows),
        "task": task_id,
        "top_1_recall": mean(float(bool(row["top_1_hit"])) for row in retrieval),
        "top_4_recall": mean(float(bool(row["top_4_hit"])) for row in retrieval),
        "top_8_recall": mean(float(bool(row["top_8_hit"])) for row in retrieval),
    }


def _retrieval_bootstrap(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    case_keys = tuple(
        (str(row["task_noun"]), str(row["story_id"]))
        for row in rows
        if row["scheme"] == "canonical_full_centroid"
    )
    if len(case_keys) != len(set(case_keys)) or len(case_keys) != 4_440:
        raise ValueError("canonical retrieval rows do not define unique paired cases")
    rows_by_key = {
        (str(row["task_noun"]), str(row["story_id"]), str(row["scheme"])): row
        for row in rows
    }
    canonical = np.asarray(
        [
            float(bool(rows_by_key[(*key, "canonical_full_centroid")]["top_8_hit"]))
            for key in case_keys
        ],
        dtype=np.float64,
    )
    return tuple(
        {
            "difference": float(np.mean(differences)),
            "lower_95": interval[0],
            "metric": "top_8_recall",
            "reference": "canonical_full_centroid",
            "scheme": scheme,
            "upper_95": interval[1],
        }
        for scheme in KEY_SCHEMES
        for differences in (
            np.asarray(
                [
                    float(bool(rows_by_key[(*key, scheme)]["top_8_hit"]))
                    for key in case_keys
                ],
                dtype=np.float64,
            )
            - canonical,
        )
        for interval in (_paired_bootstrap_interval(differences),)
    )


def _ebt_bootstrap(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    canonical_rows = tuple(
        row
        for row in rows
        if row["scheme"] == "canonical_full_centroid"
        and row["mode"] == "compact"
        and row["candidate_width"] == 8
    )
    case_keys = tuple(
        (str(row["task_noun"]), str(row["story_id"])) for row in canonical_rows
    )
    if len(case_keys) != len(set(case_keys)) or len(case_keys) != 4_440:
        raise ValueError("canonical top-8 EBT rows do not define paired cases")
    rows_by_key = {
        (str(row["task_noun"]), str(row["story_id"]), str(row["scheme"])): row
        for row in rows
        if row["mode"] == "compact" and row["candidate_width"] == 8
    }
    canonical = np.asarray(
        [
            float(rows_by_key[(*key, "canonical_full_centroid")]["suffix_mean_nll"])
            for key in case_keys
        ],
        dtype=np.float64,
    )
    return tuple(
        {
            "difference": float(np.mean(differences)),
            "lower_95": interval[0],
            "metric": "compact_top_8_story_nll",
            "reference": "canonical_full_centroid",
            "scheme": scheme,
            "upper_95": interval[1],
        }
        for scheme in KEY_SCHEMES
        for differences in (
            np.asarray(
                [
                    float(rows_by_key[(*key, scheme)]["suffix_mean_nll"])
                    for key in case_keys
                ],
                dtype=np.float64,
            )
            - canonical,
        )
        for interval in (_paired_bootstrap_interval(differences),)
    )


def _paired_bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("paired bootstrap requires finite one-dimensional differences")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    chunk_size = 250
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk_size):
        stop = min(BOOTSTRAP_REPETITIONS, start + chunk_size)
        indices = generator.integers(
            0,
            values.size,
            size=(stop - start, values.size),
            dtype=np.int32,
        )
        samples[start:stop] = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _stage_costs(inputs: AddressingStudyInputs) -> tuple[dict[str, object], ...]:
    store = IndexedStoryStore(inputs.partition)
    task_prefix_tokens = {
        task_id: sum(
            max(1, len(store.tokens(entry)) // 2 - 1)
            for entry in entries
        )
        for task_id, entries in inputs.validation_entries
    }
    task_case_counts = {
        task_id: len(entries) for task_id, entries in inputs.validation_entries
    }
    methods = (
        "canonical_exhaustive",
        "canonical_hopfield",
        "canonical_ebt_uniform",
        "canonical_ebt_hopfield",
    )
    cumulative = {
        method: {
            "active_lora_edge_evaluations": 0,
            "hopfield_dot_products": 0,
            "model_forward_equivalent_prefix_tokens": 0,
        }
        for method in methods
    }
    rows: list[dict[str, object]] = []
    for stage in range(1, len(inputs.partition.task_ids) + 1):
        learned_tasks = inputs.partition.task_ids[:stage]
        prefix_tokens = sum(task_prefix_tokens[task] for task in learned_tasks)
        case_count = sum(task_case_counts[task] for task in learned_tasks)
        node_count = stage + 1
        edge_count = stage
        increments = {
            "canonical_exhaustive": (
                prefix_tokens * node_count,
                0,
                prefix_tokens * node_count * edge_count,
            ),
            "canonical_hopfield": (
                prefix_tokens,
                case_count * node_count,
                0,
            ),
            "canonical_ebt_uniform": (
                prefix_tokens * 23,
                0,
                prefix_tokens * 23 * edge_count,
            ),
            "canonical_ebt_hopfield": (
                prefix_tokens * 24,
                case_count * node_count,
                prefix_tokens * 23 * edge_count,
            ),
        }
        for method in methods:
            forward_tokens, dot_products, edge_evaluations = increments[method]
            cumulative[method]["model_forward_equivalent_prefix_tokens"] += forward_tokens
            cumulative[method]["hopfield_dot_products"] += dot_products
            cumulative[method]["active_lora_edge_evaluations"] += edge_evaluations
            rows.append(
                {
                    "active_lora_edge_evaluations": cumulative[method][
                        "active_lora_edge_evaluations"
                    ],
                    "hopfield_dot_products": cumulative[method][
                        "hopfield_dot_products"
                    ],
                    "method": method,
                    "model_forward_equivalent_prefix_tokens": cumulative[method][
                        "model_forward_equivalent_prefix_tokens"
                    ],
                    "stage": stage,
                }
            )
    return tuple(rows)


def _write_analysis_exports(output: Path, analysis: dict[str, object]) -> None:
    retrieval = tuple(analysis["retrieval_aggregate"])
    ebt = tuple(analysis["ebt_aggregate"])
    aggregate_rows = tuple(
        {
            "candidate_width": "",
            "mean_active_lora_edge_evaluations": 0.0,
            "mean_entropy": row["mean_entropy"],
            "mean_gathered_edge_count": "",
            "mean_hopfield_dot_products": row["mean_hopfield_dot_products"],
            "mean_margin": row["mean_margin"],
            "mean_model_forward_equivalent_prefix_tokens": row[
                "mean_model_forward_equivalent_prefix_tokens"
            ],
            "mean_oracle_regret": "",
            "mean_physical_edge_capacity": "",
            "mean_selected_path_edge_count": "",
            "mode": "retrieval",
            "route_accuracy": row["top_1_recall"],
            "scheme": row["scheme"],
            "story_mean_nll": "",
            "token_mean_nll": "",
            "top_1_recall": row["top_1_recall"],
            "top_4_recall": row["top_4_recall"],
            "top_8_recall": row["top_8_recall"],
            "warm_gpu_latency_seconds": "",
            "warm_throughput_examples_per_second": "",
        }
        for row in retrieval
    ) + tuple(
        {
            "candidate_width": row["candidate_width"],
            "mean_active_lora_edge_evaluations": row[
                "mean_active_lora_edge_evaluations"
            ],
            "mean_entropy": row["mean_final_entropy"],
            "mean_gathered_edge_count": row["mean_gathered_edge_count"],
            "mean_hopfield_dot_products": row["mean_hopfield_dot_products"],
            "mean_margin": row["mean_final_margin"],
            "mean_model_forward_equivalent_prefix_tokens": row[
                "mean_model_forward_equivalent_prefix_tokens"
            ],
            "mean_oracle_regret": row["mean_oracle_regret"],
            "mean_physical_edge_capacity": row["mean_physical_edge_capacity"],
            "mean_selected_path_edge_count": row["mean_selected_path_edge_count"],
            "mode": row["mode"],
            "route_accuracy": row["route_accuracy"],
            "scheme": row["scheme"],
            "story_mean_nll": row["story_mean_nll"],
            "token_mean_nll": row["token_mean_nll"],
            "top_1_recall": "",
            "top_4_recall": row["true_node_recall_at_4"],
            "top_8_recall": row["true_node_recall_at_8"],
            "warm_gpu_latency_seconds": row["warm_gpu_latency_seconds"],
            "warm_throughput_examples_per_second": row[
                "warm_throughput_examples_per_second"
            ],
        }
        for row in ebt
    )
    _write_csv(output / "aggregate.csv", aggregate_rows)
    _write_csv(
        output / "per-task.csv",
        tuple(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "confusion_counts"
                },
                "confusion_counts": json.dumps(
                    row["confusion_counts"],
                    separators=(",", ":"),
                ),
            }
            for row in analysis["per_task"]
        ),
    )
    _write_csv(output / "timing.csv", tuple(analysis["timing"]))
    _write_csv(output / "cost.csv", tuple(analysis["stage_costs"]))


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    if not rows:
        raise ValueError(f"CSV export requires rows: {path.name}")
    fields = tuple(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path.name}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, buffer.getvalue().encode("utf-8"))


def render_cumulative_cost_svg(rows: object) -> str:
    """Render cumulative stage-sequence costs in three incomparable-unit panels."""
    values = tuple(rows)
    if len(values) != 24 * 4:
        raise ValueError("cumulative cost plot requires 24 stages and four methods")
    plt = _matplotlib_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    methods = (
        ("canonical_exhaustive", "Exhaustive", "#0072B2"),
        ("canonical_hopfield", "Hopfield", "#009E73"),
        ("canonical_ebt_uniform", "EBT-uniform", "#D55E00"),
        ("canonical_ebt_hopfield", "EBT-H", "#CC79A7"),
    )
    panels = (
        (
            "model_forward_equivalent_prefix_tokens",
            "Model-forward-equivalent prefix tokens",
        ),
        ("hopfield_dot_products", "Hopfield dot products"),
        ("active_lora_edge_evaluations", "Active LoRA-edge evaluations"),
    )
    for axis, (field, label) in zip(axes, panels):
        for method, display, color in methods:
            selected = tuple(row for row in values if row["method"] == method)
            axis.plot(
                [row["stage"] for row in selected],
                [row[field] for row in selected],
                color=color,
                linewidth=2.4,
                label=display,
            )
        axis.set_xlabel("Finalized curriculum stage", fontsize=10)
        axis.set_ylabel(label, fontsize=10)
        axis.tick_params(labelsize=9, colors="#263442")
        axis.grid(True, color="#cbd5e1", alpha=0.75, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    figure.suptitle(
        "Cumulative addressing cost across the 24-stage sequence",
        fontsize=15,
        fontweight="bold",
    )
    return _figure_svg(
        figure,
        plt,
        "Cumulative stage-sequence addressing cost",
        "Three panels keep model-forward tokens, Hopfield dot products, and active LoRA-edge evaluations in separate units.",
    )


def render_quality_latency_svg(rows: object) -> str:
    """Render final canonical dense/top-4/top-8 quality and compact latency."""
    values = tuple(rows)
    if len(values) != 3:
        raise ValueError("final-checkpoint plot requires dense, top-4, and top-8 rows")
    ordered = tuple(
        next(
            row
            for row in values
            if (
                (label == "Dense all" and row["mode"] == "dense_all")
                or (
                    label != "Dense all"
                    and row["mode"] == "compact"
                    and row["candidate_width"] == int(label.split("-")[1])
                )
            )
        )
        for label in ("Dense all", "Top-4", "Top-8")
    )
    labels = ("Dense all", "Compact top-4", "Compact top-8")
    colors = ("#4C78A8", "#F58518", "#54A24B")
    plt = _matplotlib_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), constrained_layout=True)
    measurements = (
        ("story_mean_nll", "Suffix story NLL", None),
        ("route_accuracy", "Route accuracy", (0.0, 1.0)),
        ("warm_gpu_latency_seconds", "Warm GPU latency (s / 8 rows)", None),
    )
    for axis, (field, label, bounds) in zip(axes, measurements):
        bars = axis.bar(
            labels,
            [float(row[field]) for row in ordered],
            color=colors,
            edgecolor="#263442",
            linewidth=0.7,
        )
        axis.set_ylabel(label, fontsize=10)
        axis.tick_params(axis="x", labelrotation=18, labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(True, axis="y", color="#cbd5e1", alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        if bounds is not None:
            axis.set_ylim(*bounds)
        axis.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    figure.suptitle(
        "Final-checkpoint dense versus physically compact EBT-H",
        fontsize=15,
        fontweight="bold",
    )
    return _figure_svg(
        figure,
        plt,
        "Final-checkpoint compact EBT-H quality and latency",
        "Suffix NLL, route accuracy, and synchronized warm GPU latency for dense-all, compact top-4, and compact top-8 canonical-key EBT-H.",
    )


def publish_inclusion_graph(
    inputs: AddressingStudyInputs,
    retrieval_rows: tuple[dict[str, object], ...],
    width: int,
    output: Path,
) -> tuple[Path, Path]:
    """Render a Graphviz VAMP graph colored by candidate and path activation."""
    if width not in (4, 8):
        raise ValueError("inclusion graph width must be four or eight")
    graph = inputs.adaptation.vamp_graph
    paths = path_incidence_matrix(graph)
    node_counts = np.zeros(len(graph.nodes), dtype=np.int64)
    edge_counts = np.zeros(len(graph.nodes) - 1, dtype=np.int64)
    for row in retrieval_rows:
        candidates = np.asarray(row["top_8_indices"][:width], dtype=np.int32)
        node_counts[candidates] += 1
        edge_counts += np.any(paths[candidates] != 0.0, axis=0)
    denominator = len(retrieval_rows)
    node_frequencies = node_counts / denominator
    edge_frequencies = edge_counts / denominator
    node_lines = tuple(
        "  "
        + _dot_quote(str(node.node_id))
        + " [label="
        + _dot_label(
            f"{node.train_stage:02d} · {node.node_id}\\n"
            f"candidate {node_frequencies[index]:.1%}"
        )
        + ", fillcolor="
        + _dot_quote(_frequency_color(float(node_frequencies[index])))
        + ", fontcolor="
        + _dot_quote(_contrast_color(_frequency_color(float(node_frequencies[index]))))
        + "];"
        for index, node in enumerate(graph.nodes)
    )
    edge_lines = tuple(
        "  "
        + _dot_quote(str(node.parent_id))
        + " -> "
        + _dot_quote(str(node.node_id))
        + " [color="
        + _dot_quote(_frequency_color(float(edge_frequencies[index - 1])))
        + f", penwidth={1.2 + 5.0 * edge_frequencies[index - 1]:.3f}, label="
        + _dot_quote(f"{edge_frequencies[index - 1]:.1%}")
        + "];"
        for index, node in enumerate(graph.nodes)
        if node.parent_id is not None
    )
    dot = "\n".join(
        (
            f"digraph addressing_top_{width} {{",
            '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", nodesep="0.28", ranksep="0.7"];',
            '  node [shape="box", style="rounded,filled", fontname="DejaVu Sans", fontsize="12", margin="0.14,0.08"];',
            '  edge [fontname="DejaVu Sans", fontsize="9", arrowsize="0.75"];',
            *node_lines,
            *edge_lines,
            "}",
            "",
        )
    )
    dot_path = output / f"vamp-graph-top{width}.dot"
    svg_path = output / f"vamp-graph-top{width}.svg"
    _atomic_write(dot_path, dot.encode("utf-8"))
    try:
        rendered = subprocess.run(
            ("dot", "-Tsvg", str(dot_path)),
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("Graphviz 'dot' is required for the addressing report") from error
    except subprocess.SubprocessError as error:
        raise RuntimeError("Graphviz could not render an addressing-study graph") from error
    accessible = _accessible_svg(
        rendered,
        f"VAMP dependency graph under compact top-{width} retrieval",
        "Nodes are colored by candidate-inclusion frequency and edges by union-path activation frequency.",
        f"graph-top-{width}",
    )
    _atomic_write(svg_path, accessible.encode("utf-8"))
    return dot_path, svg_path


def render_markdown_report(
    analysis: dict[str, object],
    parity: dict[str, object],
    allocator: dict[str, object],
    runtimes: dict[str, float],
) -> str:
    """Render the readable Markdown report with collapsible technical detail."""
    experiment_1 = tuple(analysis["experiment_1"])
    retrieval = tuple(analysis["retrieval_aggregate"])
    ebt = tuple(analysis["ebt_aggregate"])
    noninferiority = dict(analysis["noninferiority"])
    experiment_2 = tuple(
        (
            retrieval_row,
            next(
                row
                for row in ebt
                if row["scheme"] == retrieval_row["scheme"]
                and row["mode"] == "compact"
                and row["candidate_width"] == 8
            ),
        )
        for retrieval_row in retrieval
    )
    result_text = (
        "passed both preregistered non-inferiority margins"
        if noninferiority["story_nll_pass"] and noninferiority["accuracy_pass"]
        else "did not pass both preregistered non-inferiority margins"
    )
    lines = [
        "# TinyWorlds Nouns-v2 bounded addressing study",
        "",
        "This is a frozen final-checkpoint study. It did not retrain the base, alter a "
        "VAMP edge, or replace any canonical nouns-v1/v2 artifact.",
        "",
        "## Result",
        "",
        f"Compact top-8 {result_text}. Its story-NLL change versus dense-all was "
        f"{float(noninferiority['story_nll_increase']):+.4f} (allowed ≤ "
        f"{float(noninferiority['story_nll_margin']):.2f}); its route-accuracy loss "
        f"was {float(noninferiority['accuracy_loss']):+.2%} (allowed ≤ "
        f"{float(noninferiority['accuracy_margin']):.0%}). The result is reported "
        "regardless of that verdict.",
        "",
        "![Final-checkpoint compact quality and latency](final-checkpoint-quality-latency.svg)",
        "",
        "### Experiment 1 — dense versus physically compact EBT-H",
        "",
        "| method | route accuracy | true-node recall@4 | true-node recall@8 | story NLL | token NLL | oracle regret | active path edges | gathered bank edges | warm GPU latency / 8 rows | throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {_method_label(row)} | {float(row['route_accuracy']):.2%} | "
            f"{float(row['true_node_recall_at_4']):.2%} | "
            f"{float(row['true_node_recall_at_8']):.2%} | "
            f"{float(row['story_mean_nll']):.4f} | {float(row['token_mean_nll']):.4f} | "
            f"{float(row['mean_oracle_regret']):+.4f} | "
            f"{float(row['mean_selected_path_edge_count']):.2f} | "
            f"{float(row['mean_gathered_edge_count']):.2f} | "
            f"{float(row['warm_gpu_latency_seconds']):.4f} s | "
            f"{float(row['warm_throughput_examples_per_second']):.1f}/s |"
            for row in _ordered_experiment_1(experiment_1)
        ),
        "",
        "### Experiment 2 — frozen key schemes",
        "",
        "Recall@8 is primary; compact top-8 suffix NLL is secondary.",
        "",
        "| key scheme | recall@1 | recall@4 | recall@8 | retrieval entropy | margin | compact top-8 accuracy | compact top-8 story NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {retrieval_row['scheme']} | {float(retrieval_row['top_1_recall']):.2%} | "
            f"{float(retrieval_row['top_4_recall']):.2%} | "
            f"{float(retrieval_row['top_8_recall']):.2%} | "
            f"{float(retrieval_row['mean_entropy']):.3f} | "
            f"{float(retrieval_row['mean_margin']):.4f} | "
            f"{float(ebt_row['route_accuracy']):.2%} | "
            f"{float(ebt_row['story_mean_nll']):.4f} |"
            for retrieval_row, ebt_row in experiment_2
        ),
        "",
        "<details><summary>Method and interpretation</summary>",
        "",
        "Frozen-base keys can work because the base's final hidden states already carry "
        "lexical and topical information; a centroid or nearest prototype can therefore "
        "retrieve a task without a learned router. They are suspect here because the "
        "canonical keys summarize complete probe stories while every real query stops at "
        "the midpoint, and the frozen base was never optimized to separate these 24 "
        "disjoint noun memories.",
        "",
        "For active prefix transition `t`, the residual signature is computed exactly as "
        "`g_t = softmax(logits_t) @ token_embedding - token_embedding[target_t]`. "
        "The study masked-means those gradients, L2-normalizes the result, and fuses it "
        "with unit content as `[content / sqrt(2), residual / sqrt(2)]`. No validation "
        "example contributes to a key. Router inputs contain only prefix transitions; "
        "task identity and suffix tokens remain evaluator metadata.",
        "",
        "Every scheme uses all 36 registered probes for every node, including the root. "
        f"All EBT runs use {EBT_STEPS} Adam steps, learning rate {EBT_LEARNING_RATE:g}, "
        f"temperature {EBT_TEMPERATURE:g}, entropy penalty "
        f"{EBT_ENTROPY_PENALTY:g}, and Hopfield beta {HOPFIELD_BETA:g}.",
        "",
        "Logical masking keeps a dense 24-edge bank resident and assigns zero "
        "coefficients outside a shortlist. Physical compaction instead gathers each "
        "row's insertion-ordered union of shortlisted root-to-node edges and executes "
        "only those factors in a 4/8/12/16/20/24 capacity bucket. Both optimize the same "
        "four or eight candidate logits.",
        "",
        "</details>",
        "",
        "<details><summary>Paired seed-zero bootstrap intervals</summary>",
        "",
        "| metric | scheme | paired difference vs canonical keys | 95% interval |",
        "|---|---|---:|---:|",
        *(
            f"| {row['metric']} | {row['scheme']} | {float(row['difference']):+.5f} | "
            f"[{float(row['lower_95']):+.5f}, {float(row['upper_95']):+.5f}] |"
            for row in (
                *analysis["retrieval_bootstrap"],
                *analysis["ebt_bootstrap"],
            )
        ),
        "",
        f"Each interval uses {BOOTSTRAP_REPETITIONS:,} paired resamples with seed "
        f"{BOOTSTRAP_SEED}.",
        "",
        "</details>",
        "",
        "<details><summary>Per-task results and confusion</summary>",
        "",
        "The machine-readable per-task table includes a 25-node confusion-count object "
        "plus retrieval/final entropy and margin for every task/method cell: "
        "[per-task.csv](per-task.csv).",
        "",
        "| task | scheme | mode | width | recall@8 | route accuracy | story NLL |",
        "|---|---|---|---:|---:|---:|---:|",
        *(
            f"| {row['task']} | {row['scheme']} | {row['mode']} | "
            f"{row['candidate_width']} | {float(row['top_8_recall']):.2%} | "
            f"{float(row['route_accuracy']):.2%} | {float(row['story_mean_nll']):.4f} |"
            for row in analysis["per_task"]
            if row["candidate_width"] == 8 and row["mode"] == "compact"
        ),
        "",
        "</details>",
        "",
        "<details><summary>Timing and operation accounting</summary>",
        "",
        "Five synchronized warm repetitions were measured for every observed "
        "prefix-width/physical-edge shape; cold compilation is separate in "
        "[timing.csv](timing.csv). GPU kernel latency, end-to-end wall time, "
        "model-forward-equivalent prefix tokens, Hopfield dot products, and active "
        "LoRA-edge evaluations are not conflated.",
        "",
        f"End-to-end evaluation wall time: {float(runtimes['end_to_end']):.1f} s. "
        f"Observed allocator peak: {int(allocator['peak_bytes_in_use']) / 2**30:.2f} GiB "
        f"against the {int(allocator['allocator_limit_bytes']) / 2**30:.0f} GiB gate.",
        "",
        "![Cumulative stage-sequence addressing cost](cumulative-addressing-cost.svg)",
        "",
        "The cumulative plot keeps incomparable operation units in separate panels. "
        "Detailed values are in [cost.csv](cost.csv) and aggregates in "
        "[aggregate.csv](aggregate.csv).",
        "",
        "</details>",
        "",
        "<details><summary>Top-4 and top-8 VAMP dependency graphs</summary>",
        "",
        "Nodes are colored by candidate-inclusion frequency across all five schemes; "
        "edges are colored and weighted by compact union-path activation.",
        "",
        "![Top-4 VAMP inclusion graph](vamp-graph-top4.svg)",
        "",
        "![Top-8 VAMP inclusion graph](vamp-graph-top8.svg)",
        "",
        "</details>",
        "",
        "<details><summary>Provenance and numerical gates</summary>",
        "",
        f"Real-checkpoint compact/dense parity used tolerance `{parity['tolerance']}`; "
        f"maximum differences: `{json.dumps(parity['maximum_absolute_differences'], sort_keys=True)}`.",
        "",
        f"Canonical run: `{analysis['provenance']['canonical_run_sha256']}`  ",
        f"Key artifact: `{analysis['provenance']['addressing_key_artifact_sha256']}`  ",
        f"Retrieval contract: `{analysis['provenance']['retrieval_contract_sha256']}`  ",
        f"EBT contract: `{analysis['provenance']['ebt_contract_sha256']}`  ",
        f"Partition: `{analysis['provenance']['partition_sha256']}`  ",
        f"Selected base: `{analysis['provenance']['base_parameter_checksum']}`  ",
        f"Final VAMP tensors: `{analysis['provenance']['vamp_tensor_checksum']}`",
        "",
        "Every source ledger, contract, result row, key tensor, and report projection "
        "is independently content-addressed. Canonical hashes were checked again after "
        "publication.",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def render_html_report(
    analysis: dict[str, object],
    parity: dict[str, object],
    allocator: dict[str, object],
    runtimes: dict[str, float],
    svgs: dict[str, str],
) -> str:
    """Render a self-contained folding HTML report with embedded accessible SVGs."""
    experiment_1 = _ordered_experiment_1(tuple(analysis["experiment_1"]))
    retrieval = tuple(analysis["retrieval_aggregate"])
    ebt = tuple(analysis["ebt_aggregate"])
    noninferiority = dict(analysis["noninferiority"])
    verdict = (
        "Passed both preregistered margins"
        if noninferiority["story_nll_pass"] and noninferiority["accuracy_pass"]
        else "Did not pass both preregistered margins"
    )
    experiment_1_rows = "".join(
        "<tr>"
        f"<td>{escape(_method_label(row))}</td>"
        f"<td>{float(row['route_accuracy']):.2%}</td>"
        f"<td>{float(row['true_node_recall_at_4']):.2%}</td>"
        f"<td>{float(row['true_node_recall_at_8']):.2%}</td>"
        f"<td>{float(row['story_mean_nll']):.4f}</td>"
        f"<td>{float(row['token_mean_nll']):.4f}</td>"
        f"<td>{float(row['mean_oracle_regret']):+.4f}</td>"
        f"<td>{float(row['mean_selected_path_edge_count']):.2f}</td>"
        f"<td>{float(row['mean_gathered_edge_count']):.2f}</td>"
        f"<td>{float(row['warm_gpu_latency_seconds']):.4f} s</td>"
        f"<td>{float(row['warm_throughput_examples_per_second']):.1f}/s</td>"
        "</tr>"
        for row in experiment_1
    )
    experiment_2_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['scheme']))}</code></td>"
        f"<td>{float(row['top_1_recall']):.2%}</td>"
        f"<td>{float(row['top_4_recall']):.2%}</td>"
        f"<td>{float(row['top_8_recall']):.2%}</td>"
        f"<td>{float(row['mean_entropy']):.3f}</td>"
        f"<td>{float(row['mean_margin']):.4f}</td>"
        f"<td>{float(compact['route_accuracy']):.2%}</td>"
        f"<td>{float(compact['story_mean_nll']):.4f}</td>"
        "</tr>"
        for row in retrieval
        for compact in (
            next(
                value
                for value in ebt
                if value["scheme"] == row["scheme"]
                and value["mode"] == "compact"
                and value["candidate_width"] == 8
            ),
        )
    )
    bootstrap_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['metric']))}</td>"
        f"<td><code>{escape(str(row['scheme']))}</code></td>"
        f"<td>{float(row['difference']):+.5f}</td>"
        f"<td>[{float(row['lower_95']):+.5f}, {float(row['upper_95']):+.5f}]</td>"
        "</tr>"
        for row in (*analysis["retrieval_bootstrap"], *analysis["ebt_bootstrap"])
    )
    task_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['task']))}</td>"
        f"<td><code>{escape(str(row['scheme']))}</code></td>"
        f"<td>{float(row['top_8_recall']):.2%}</td>"
        f"<td>{float(row['route_accuracy']):.2%}</td>"
        f"<td>{float(row['story_mean_nll']):.4f}</td>"
        f"<td><code>{escape(json.dumps(row['confusion_counts'], separators=(',', ':')))}</code></td>"
        "</tr>"
        for row in analysis["per_task"]
        if row["candidate_width"] == 8 and row["mode"] == "compact"
    )
    timing_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['mode']))}</td>"
        f"<td>{int(row['candidate_width'])}</td>"
        f"<td>{int(row['prefix_width_bucket'])}</td>"
        f"<td>{int(row['physical_edge_capacity'])}</td>"
        f"<td>{float(row['cold_compile_seconds']):.4f} s</td>"
        f"<td>{float(row['warm_kernel_mean_seconds']):.4f} s</td>"
        f"<td>{float(row['warm_throughput_examples_per_second']):.1f}/s</td>"
        "</tr>"
        for row in analysis["timing"]
    )
    provenance_rows = "".join(
        f"<tr><td><code>{escape(str(name))}</code></td><td><code>{escape(str(value))}</code></td></tr>"
        for name, value in analysis["provenance"].items()
        if name != "canonical_artifact_hashes"
        and name != "canonical_ledger_hashes"
    )
    style = """
body{font:16px/1.55 system-ui,-apple-system,sans-serif;max-width:1260px;margin:auto;padding:2rem;color:#172331;background:#f7fafc}
h1,h2{line-height:1.2;color:#102a43}.lead{font-size:1.08rem;max-width:85ch}.cards{display:flex;gap:.8rem;flex-wrap:wrap}.metric{background:#fff;border:1px solid #cad6e2;border-radius:10px;padding:.7rem 1rem;min-width:180px}.metric b{display:block;font-size:1.18rem;color:#0b5f80}
details{background:#fff;border:1px solid #c9d5e0;border-radius:10px;margin:1rem 0;padding:.8rem 1rem}summary{cursor:pointer;font-weight:700;color:#193b57;font-size:1.05rem}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem;margin:.8rem 0}th,td{border-bottom:1px solid #d6e0e8;padding:.5rem .6rem;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{background:#eaf1f6;color:#17344d;position:sticky;top:0}code{font-size:.82em;word-break:break-all}.plot{background:white;border:1px solid #d2dde6;border-radius:10px;padding:.5rem;margin:1rem 0}.plot svg{width:100%;height:auto}.pass{color:#176b45}.fail{color:#9b2c2c}a{color:#075985}@media(max-width:700px){body{padding:1rem}table{font-size:.78rem}}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyWorlds Nouns-v2 bounded addressing study</title><style>{style}</style></head><body>
<h1>TinyWorlds Nouns-v2 bounded addressing study</h1>
<p class="lead">A frozen final-checkpoint comparison of dense EBT-H, physically compact top-4/top-8 EBT-H, and five non-learned content/error retrieval schemes. No model was retrained and no canonical artifact was rewritten.</p>
<div class="cards"><div class="metric"><b class="{'pass' if noninferiority['story_nll_pass'] and noninferiority['accuracy_pass'] else 'fail'}">{escape(verdict)}</b>top-8 non-inferiority</div><div class="metric"><b>{float(noninferiority['story_nll_increase']):+.4f}</b>story-NLL change (margin {float(noninferiority['story_nll_margin']):.2f})</div><div class="metric"><b>{float(noninferiority['accuracy_loss']):+.2%}</b>accuracy loss (margin {float(noninferiority['accuracy_margin']):.0%})</div></div>
<div class="plot">{svgs['final-checkpoint-quality-latency.svg']}</div>
<h2>Experiment 1</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Route accuracy</th><th>True-node recall@4</th><th>True-node recall@8</th><th>Story NLL</th><th>Token NLL</th><th>Oracle regret</th><th>Active path edges</th><th>Gathered bank edges</th><th>Warm GPU / 8 rows</th><th>Throughput</th></tr></thead><tbody>{experiment_1_rows}</tbody></table></div>
<h2>Experiment 2</h2><p>Recall@8 is primary; compact top-8 suffix NLL is secondary.</p><div class="scroll"><table><thead><tr><th>Key scheme</th><th>Recall@1</th><th>Recall@4</th><th>Recall@8</th><th>Entropy</th><th>Margin</th><th>Top-8 accuracy</th><th>Top-8 story NLL</th></tr></thead><tbody>{experiment_2_rows}</tbody></table></div>
<details><summary>Method and interpretation</summary><p>Frozen-base keys can work because the base’s hidden states carry lexical and topical cues. They are suspect here because canonical keys summarize complete probes while deployment queries stop at the midpoint, and the base was not trained to separate these 24 memories.</p><p>For transition <i>t</i>, the exact residual is <code>g_t = softmax(logits_t) @ token_embedding - token_embedding[target_t]</code>. The masked mean is L2-normalized and fused as <code>[content / sqrt(2), residual / sqrt(2)]</code>. All 36 registered probes—including root probes—are used. Validation stories, suffix tokens, and task identity never enter a key or query.</p><p>Every EBT run uses {EBT_STEPS} Adam steps, learning rate {EBT_LEARNING_RATE:g}, temperature {EBT_TEMPERATURE:g}, entropy penalty {EBT_ENTROPY_PENALTY:g}, and Hopfield beta {HOPFIELD_BETA:g}.</p><p>Logical masking retains all 24 edge factors and zeros coefficients. Physical compaction gathers each row’s insertion-ordered union of candidate paths, buckets it to 4/8/12/16/20/24 edges, optimizes only four or eight logits, and executes only those gathered factors.</p></details>
<details><summary>Paired seed-zero bootstrap intervals</summary><div class="scroll"><table><thead><tr><th>Metric</th><th>Scheme</th><th>Difference</th><th>95% interval</th></tr></thead><tbody>{bootstrap_rows}</tbody></table></div><p>{BOOTSTRAP_REPETITIONS:,} paired resamples, deterministic seed {BOOTSTRAP_SEED}.</p></details>
<details><summary>Per-task confusion and quality</summary><p>The confusion column is ordered by the 25 insertion-ordered nodes; the CSV also carries per-task retrieval/final entropy and margin. <a href="per-task.csv">CSV export</a>.</p><div class="scroll"><table><thead><tr><th>Task</th><th>Scheme</th><th>Recall@8</th><th>Route accuracy</th><th>Story NLL</th><th>Selected-node counts</th></tr></thead><tbody>{task_rows}</tbody></table></div></details>
<details><summary>Timing and cost</summary><p>Five synchronized warm repetitions per observed shape are reported separately from cold compilation. Kernel latency, end-to-end evaluation wall time ({float(runtimes['end_to_end']):.1f} s), model-forward-equivalent prefix tokens, Hopfield dot products, and active LoRA-edge evaluations retain separate units. Peak allocator use was {int(allocator['peak_bytes_in_use']) / 2**30:.2f} GiB under the {int(allocator['allocator_limit_bytes']) / 2**30:.0f} GiB gate.</p><div class="scroll"><table><thead><tr><th>Mode</th><th>Width</th><th>Prefix bucket</th><th>Edge bucket</th><th>Cold compile</th><th>Warm kernel</th><th>Throughput</th></tr></thead><tbody>{timing_rows}</tbody></table></div><div class="plot">{svgs['cumulative-addressing-cost.svg']}</div><p><a href="timing.csv">Timing CSV</a> · <a href="cost.csv">Cost CSV</a> · <a href="aggregate.csv">Aggregate CSV</a></p></details>
<details><summary>Top-4 and top-8 VAMP dependency graphs</summary><p>Nodes show candidate inclusion; edges show union-path activation.</p><div class="plot">{svgs['vamp-graph-top4.svg']}</div><div class="plot">{svgs['vamp-graph-top8.svg']}</div></details>
<details><summary>Provenance and numerical gates</summary><p>Real-checkpoint compact/dense tolerance: <code>{parity['tolerance']}</code>. Maximum absolute differences: <code>{escape(json.dumps(parity['maximum_absolute_differences'], sort_keys=True))}</code>.</p><div class="scroll"><table><tbody>{provenance_rows}</tbody></table></div><p>Every source ledger, contract, result row, key tensor, and report projection is content-addressed. Protected nouns-v1/v2 hashes were checked again after publication.</p></details>
</body></html>"""


def _ordered_experiment_1(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        next(
            row
            for row in rows
            if (
                row["mode"] == mode
                and int(row["candidate_width"]) == width
            )
        )
        for mode, width in (("dense_all", 25), ("compact", 4), ("compact", 8))
    )


def _method_label(row: dict[str, object]) -> str:
    return (
        "Dense all-node EBT-H"
        if row["mode"] == "dense_all"
        else f"Compact top-{int(row['candidate_width'])} EBT-H"
    )


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["svg.hashsalt"] = "tinyworlds-nouns-v2-addressing-v1"
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("matplotlib is required for the addressing report") from error
    return plt


def _figure_svg(figure, plt, title: str, description: str) -> str:
    output = io.StringIO()
    figure.savefig(
        output,
        format="svg",
        facecolor="white",
        metadata={"Date": None, "Creator": "VAMP addressing study"},
    )
    plt.close(figure)
    identifier = f"plot-{sha256(title.encode('utf-8')).hexdigest()[:12]}"
    return _accessible_svg(output.getvalue(), title, description, identifier)


def _accessible_svg(
    svg: str,
    title: str,
    description: str,
    identifier: str,
) -> str:
    start = svg.find("<svg")
    if start < 0:
        raise ValueError("renderer did not produce SVG")
    fragment = svg[start:]
    close = fragment.find(">")
    if close < 0:
        raise ValueError("renderer produced malformed SVG")
    title_id = f"{identifier}-title"
    description_id = f"{identifier}-description"
    opening = fragment[:close]
    opening += (
        f' role="img" aria-labelledby="{title_id} {description_id}"'
        if "role=" not in opening
        else ""
    )
    accessible_text = (
        f'<title id="{title_id}">{escape(title)}</title>'
        f'<desc id="{description_id}">{escape(description)}</desc>'
    )
    return opening + ">" + accessible_text + fragment[close + 1 :]


def _frequency_color(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("graph frequency must be in [0, 1]")
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex
    except ImportError as error:
        raise ImportError("matplotlib is required for graph colors") from error
    return str(to_hex(colormaps["cividis"](0.12 + 0.83 * value), keep_alpha=False))


def _contrast_color(hex_color: str) -> str:
    red, green, blue = (
        int(hex_color[index : index + 2], 16) / 255.0 for index in (1, 3, 5)
    )
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.58 else "#ffffff"


def _dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dot_label(value: str) -> str:
    """Quote a DOT label while retaining Graphviz newline escapes."""
    return _dot_quote(value).replace("\\\\n", "\\n")


def _jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


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
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "analyze_study_rows",
    "publish_addressing_study_report",
    "publish_inclusion_graph",
    "render_cumulative_cost_svg",
    "render_html_report",
    "render_markdown_report",
    "render_quality_latency_svg",
]
