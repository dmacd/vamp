"""Bounded stagewise evaluation of nouns-v2 stored adapter baselines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_baseline_training import pack_root_adapter
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    EvaluationProgress,
    _HalfStoryCase,
    _half_story_chunks,
    _nll_by_node_per_window,
    _stack_token_batches,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASELINE_CONDITIONS,
    BASELINE_STAGEWISE_FORMAT,
    STAGEWISE_CASE_COUNT,
    TASK_IDS,
    BaselineStagewiseClRow,
    BaselineStagewiseConditionResult,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora import LoraEdge
from apm.lm.parameters import GptNeoParams


_TOP_LEVEL_FIELDS = {
    "baseline_tensor_checksum",
    "format",
    "introduced_task",
    "result_sha256",
    "results",
    "stage_index",
    "story_id",
    "task_noun",
    "vamp_tensor_checksum",
}
_RESULT_FIELDS = {
    "adapter_task",
    "condition",
    "deficit_vs_independent",
    "mean_nll",
    "token_count",
    "total_nll",
}


def evaluate_stagewise_baselines(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    baseline_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stagewise_path: str | Path,
    output_path: str | Path,
    *,
    progress: EvaluationProgress | None = None,
) -> Path:
    """Evaluate sequential and task-matched independent adapters at every stage."""
    metadata = _stage_metadata(partition, baseline_stages, vamp_stages)
    entries_by_task = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries_by_task)
    output = Path(output_path)
    if output.is_file():
        completed = validate_baseline_stagewise_ledger(
            output,
            partition,
            baseline_stages,
            vamp_stages,
            require_complete=True,
            entries_by_task=entries_by_task,
        )
        if completed != expected:
            raise ValueError("published baseline stagewise coverage changed")
        _validate_vamp_reference_parity(output, vamp_stagewise_path)
        return output

    work = output.with_name(f".{output.name}.work")
    work.parent.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns-v2 baseline CL ledger: {work.resolve()}", flush=True)
    _repair_interrupted_tail(work)
    completed = validate_baseline_stagewise_ledger(
        work,
        partition,
        baseline_stages,
        vamp_stages,
        require_complete=False,
        entries_by_task=entries_by_task,
    )
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    store = IndexedStoryStore(partition)
    total = len(expected)
    finished = len(completed)
    with work.open("ab") as ledger:
        for stage_index, baseline in enumerate(baseline_stages, start=1):
            learned_entries = {
                task_id: entries_by_task[task_id]
                for task_id in partition.task_ids[:stage_index]
            }
            completed_at_stage = {
                (task_id, story_id)
                for existing_stage, task_id, story_id in completed
                if existing_stage == str(stage_index)
            }
            chunks = _half_story_chunks(
                partition,
                learned_entries,
                completed_at_stage,
                store,
                preset,
                baseline.model_config.max_position_embeddings,
            )
            independent_by_task = {
                str(record.task_id): record.adapter
                for record in baseline.independent_adapters
            }
            for cases in chunks:
                sequential_totals = _score_cases(
                    loaded.params,
                    baseline,
                    baseline.sequential_stages[-1].adapter,
                    cases,
                    preset.evaluation_chunk_size,
                )
                independent_totals = _score_independent_cases(
                    loaded.params,
                    baseline,
                    independent_by_task,
                    cases,
                    preset.evaluation_chunk_size,
                )
                rows = tuple(
                    _result_row(
                        partition,
                        metadata,
                        stage_index,
                        case,
                        sequential_total,
                        independent_total,
                    )
                    for case, sequential_total, independent_total in zip(
                        cases,
                        sequential_totals,
                        independent_totals,
                    )
                )
                ledger.write(b"".join(canonical_json_bytes(row) for row in rows))
                ledger.flush()
                os.fsync(ledger.fileno())
                for row in rows:
                    key = (
                        str(row["stage_index"]),
                        str(row["task_noun"]),
                        str(row["story_id"]),
                    )
                    completed.add(key)
                    finished += 1
                    if progress is not None:
                        progress("baseline-stagewise-cl", finished, total)
    if completed != expected:
        raise RuntimeError(
            f"baseline stagewise ledger has {len(completed):,} of {total:,} rows"
        )
    os.replace(work, output)
    _validate_vamp_reference_parity(output, vamp_stagewise_path)
    return output


def validate_baseline_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    baseline_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    *,
    require_complete: bool,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]] | None = None,
) -> set[tuple[str, str, str]]:
    """Strictly validate baseline row identities, bindings, and coverage."""
    metadata = _stage_metadata(partition, baseline_stages, vamp_stages)
    entries = entries_by_task or _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries)
    keys: set[tuple[str, str, str]] = set()
    source = Path(path)
    if not source.is_file():
        if require_complete:
            raise FileNotFoundError(source)
        return keys
    with source.open("rb") as stream:
        for line in stream:
            row = _validated_row(line, metadata, partition.task_ids)
            key = (
                str(row["stage_index"]),
                str(row["task_noun"]),
                str(row["story_id"]),
            )
            if key in keys or key not in expected:
                raise ValueError("baseline stagewise ledger key changed")
            keys.add(key)
    if require_complete and keys != expected:
        raise ValueError(
            f"baseline stagewise ledger has {len(keys):,} of {len(expected):,} rows"
        )
    return keys


def summarize_baseline_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    baseline_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stagewise_path: str | Path,
) -> dict[str, object]:
    """Stream the strict baseline ledger into comparison-ready CL summaries."""
    source = Path(path)
    metadata = _stage_metadata(partition, baseline_stages, vamp_stages)
    entries = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries)
    keys: set[tuple[str, str, str]] = set()

    def validated_rows():
        with source.open("rb") as stream:
            for line in stream:
                row = _validated_row(line, metadata, partition.task_ids)
                key = (
                    str(row["stage_index"]),
                    str(row["task_noun"]),
                    str(row["story_id"]),
                )
                if key in keys or key not in expected:
                    raise ValueError("baseline report ledger keys changed")
                keys.add(key)
                yield row

    measured = summarize_baseline_stagewise_rows(validated_rows(), partition.task_ids)
    if keys != expected:
        raise ValueError("baseline report ledger coverage changed")
    _validate_vamp_reference_parity(source, vamp_stagewise_path)
    core = {
        **measured,
        "baseline_tensor_checksums": [
            metadata[index]["baseline_checksum"]
            for index in range(1, len(baseline_stages) + 1)
        ],
        "format": BASELINE_STAGEWISE_FORMAT,
        "ledger_sha256": _file_sha256(source),
        "vamp_stagewise_sha256": _file_sha256(Path(vamp_stagewise_path)),
    }
    return {**core, "summary_sha256": record_sha256(core)}


def summarize_baseline_stagewise_rows(
    rows: Iterable[dict[str, object]],
    task_ids: tuple[str, ...] = TASK_IDS,
) -> dict[str, object]:
    """Compute stored-baseline stage curves, forgetting, and backward transfer."""
    task_position = {task_id: index for index, task_id in enumerate(task_ids, start=1)}
    values: dict[tuple[int, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    row_count = 0
    for row in rows:
        stage = int(row["stage_index"])
        task = str(row["task_noun"])
        if task not in task_position or not task_position[task] <= stage <= len(task_ids):
            raise ValueError("baseline summary row is outside the learned-task triangle")
        results = _object(row["results"], "baseline results")
        if set(results) != set(BASELINE_CONDITIONS):
            raise ValueError("baseline summary conditions changed")
        row_count += 1
        for condition in BASELINE_CONDITIONS:
            result = _object(results[condition], condition)
            bucket = values[(stage, task, condition)]
            bucket[0] += 1.0
            bucket[1] += float(result["mean_nll"])
            bucket[2] += float(result["total_nll"])
            bucket[3] += float(result["token_count"])
            bucket[4] += float(result["deficit_vs_independent"])

    def task_stage(stage: int, task: str, condition: str) -> dict[str, object]:
        count, mean_sum, total, tokens, deficit = values[(stage, task, condition)]
        if count <= 0.0 or tokens <= 0.0:
            raise ValueError("baseline summary is missing a task/stage cell")
        return {
            "mean_deficit_vs_independent": deficit / count,
            "story_count": int(count),
            "story_mean_nll": mean_sum / count,
            "token_count": int(tokens),
            "token_mean_nll": total / tokens,
        }

    stages = tuple(
        {
            "conditions": {
                condition: _aggregate_cells(
                    tuple(
                        task_stage(stage, task, condition)
                        for task in task_ids[:stage]
                    )
                )
                for condition in BASELINE_CONDITIONS
            },
            "introduced_task": task_ids[stage - 1],
            "learned_task_count": stage,
            "stage_index": stage,
            "story_count": sum(
                int(task_stage(stage, task_ids[index], BASELINE_CONDITIONS[0])["story_count"])
                for index in range(stage)
            ),
        }
        for stage in range(1, len(task_ids) + 1)
    )
    task_metrics = tuple(
        {
            "conditions": {
                condition: _longitudinal_metrics(
                    tuple(
                        (stage, task_stage(stage, task, condition))
                        for stage in range(task_position[task], len(task_ids) + 1)
                    )
                )
                for condition in BASELINE_CONDITIONS
            },
            "introduction_stage": task_position[task],
            "task": task,
        }
        for task in task_ids
    )
    condition_summaries = {
        condition: _condition_summary(condition, stages, task_metrics)
        for condition in BASELINE_CONDITIONS
    }
    independent_drifts = tuple(
        abs(
            float(task_stage(stage, task, "independent_root_lora")["story_mean_nll"])
            - float(
                task_stage(task_position[task], task, "independent_root_lora")[
                    "story_mean_nll"
                ]
            )
        )
        for task in task_ids
        for stage in range(task_position[task], len(task_ids) + 1)
    )
    return {
        "condition_summaries": condition_summaries,
        "independent_max_absolute_drift": max(independent_drifts),
        "row_count": row_count,
        "stage_count": len(task_ids),
        "stages": list(stages),
        "task_metrics": list(task_metrics),
    }


def _result_row(
    partition: NounsV2PartitionArtifact,
    metadata: dict[int, dict[str, object]],
    stage_index: int,
    case: _HalfStoryCase,
    sequential_total: float,
    independent_total: float,
) -> dict[str, object]:
    token_count = int(np.sum(case.suffix_windows.loss_mask))
    sequential_mean = sequential_total / token_count
    independent_mean = independent_total / token_count
    return BaselineStagewiseClRow(
        stage_index=stage_index,
        introduced_task=partition.task_ids[stage_index - 1],
        baseline_tensor_checksum=str(metadata[stage_index]["baseline_checksum"]),
        vamp_tensor_checksum=str(metadata[stage_index]["vamp_checksum"]),
        task_noun=case.task_id,
        story_id=case.entry.story_id,
        results=(
            BaselineStagewiseConditionResult(
                "sequential_single_lora",
                partition.task_ids[stage_index - 1],
                sequential_total,
                token_count,
                sequential_mean,
                sequential_mean - independent_mean,
            ),
            BaselineStagewiseConditionResult(
                "independent_root_lora",
                case.task_id,
                independent_total,
                token_count,
                independent_mean,
                0.0,
            ),
        ),
    ).as_record()


def _score_independent_cases(
    base_params: GptNeoParams,
    baseline: LanguageAdaptationArtifact,
    independent_by_task: dict[str, LoraEdge],
    cases: tuple[_HalfStoryCase, ...],
    microbatch_size: int,
) -> tuple[float, ...]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        positions[case.task_id].append(index)
    totals = [0.0] * len(cases)
    for task_id, indices in positions.items():
        selected_cases = tuple(cases[index] for index in indices)
        measured = _score_cases(
            base_params,
            baseline,
            independent_by_task[task_id],
            selected_cases,
            microbatch_size,
        )
        for index, total in zip(indices, measured):
            totals[index] = total
    return tuple(totals)


def _score_cases(
    base_params: GptNeoParams,
    baseline: LanguageAdaptationArtifact,
    adapter: LoraEdge,
    cases: tuple[_HalfStoryCase, ...],
    microbatch_size: int,
) -> tuple[float, ...]:
    _, packed = pack_root_adapter(
        adapter,
        baseline.model_config,
        baseline.lora_config,
    )
    windows = _stack_token_batches(tuple(case.suffix_windows for case in cases))
    per_window = _nll_by_node_per_window(
        base_params,
        baseline,
        packed,
        windows,
        microbatch_size,
        node_indices=(1,),
    )[0]
    boundaries = np.cumsum(
        (0,) + tuple(case.suffix_windows.input_ids.shape[0] for case in cases)
    )
    return tuple(
        float(np.sum(per_window[start:stop], dtype=np.float64))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    )


def _stage_metadata(
    partition: NounsV2PartitionArtifact,
    baseline_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
) -> dict[int, dict[str, object]]:
    if (
        len(baseline_stages) != len(partition.task_ids)
        or len(vamp_stages) != len(partition.task_ids)
    ):
        raise ValueError("baseline stagewise audit requires all baseline and VAMP stages")
    metadata = {}
    for stage_index, (baseline, vamp) in enumerate(
        zip(baseline_stages, vamp_stages),
        start=1,
    ):
        expected = partition.task_ids[:stage_index]
        if (
            tuple(str(task) for task in baseline.task_order) != expected
            or tuple(str(task) for task in vamp.task_order) != expected
            or len(baseline.sequential_stages) != stage_index
            or len(baseline.independent_adapters) != stage_index
        ):
            raise ValueError("baseline stage is not the canonical learned-task prefix")
        metadata[stage_index] = {
            "baseline_checksum": baseline.tensor_checksum,
            "introduced_task": expected[-1],
            "vamp_checksum": vamp.tensor_checksum,
        }
    return metadata


def _validated_row(
    line: bytes,
    metadata: dict[int, dict[str, object]],
    task_ids: tuple[str, ...],
) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise ValueError("baseline stagewise ledger has an interrupted tail")
    row = json.loads(line)
    if type(row) is not dict or canonical_json_bytes(row) != line:
        raise ValueError("baseline stagewise ledger is not canonical JSONL")
    core = {key: value for key, value in row.items() if key != "result_sha256"}
    stage = row.get("stage_index")
    if (
        set(row) != _TOP_LEVEL_FIELDS
        or row.get("format") != BASELINE_STAGEWISE_FORMAT
        or row.get("result_sha256") != record_sha256(core)
        or type(stage) is not int
        or stage not in metadata
    ):
        raise ValueError("baseline stagewise row identity changed")
    stage_info = metadata[stage]
    task = row.get("task_noun")
    if (
        row.get("introduced_task") != stage_info["introduced_task"]
        or row.get("baseline_tensor_checksum") != stage_info["baseline_checksum"]
        or row.get("vamp_tensor_checksum") != stage_info["vamp_checksum"]
        or type(task) is not str
        or task not in task_ids[:stage]
    ):
        raise ValueError("baseline stagewise row binding changed")
    results = _object(row.get("results"), "baseline results")
    if set(results) != set(BASELINE_CONDITIONS):
        raise ValueError("baseline stagewise conditions changed")
    validated = {
        condition: _validated_result(results[condition], condition)
        for condition in BASELINE_CONDITIONS
    }
    sequential = validated["sequential_single_lora"]
    independent = validated["independent_root_lora"]
    if (
        sequential["adapter_task"] != stage_info["introduced_task"]
        or independent["adapter_task"] != task
        or sequential["token_count"] != independent["token_count"]
        or float(independent["deficit_vs_independent"]) != 0.0
        or not math.isclose(
            float(sequential["deficit_vs_independent"]),
            float(sequential["mean_nll"]) - float(independent["mean_nll"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("baseline stagewise comparison metadata changed")
    return row


def _validated_result(value: object, condition: str) -> dict[str, object]:
    result = _object(value, condition)
    numeric = tuple(
        result.get(field)
        for field in ("deficit_vs_independent", "mean_nll", "total_nll")
    )
    if (
        set(result) != _RESULT_FIELDS
        or result.get("condition") != condition
        or type(result.get("adapter_task")) is not str
        or type(result.get("token_count")) is not int
        or int(result["token_count"]) <= 0
        or any(
            type(number) not in (int, float) or not math.isfinite(float(number))
            for number in numeric
        )
        or float(result["mean_nll"]) < 0.0
        or float(result["total_nll"]) < 0.0
        or not math.isclose(
            float(result["mean_nll"]),
            float(result["total_nll"]) / int(result["token_count"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("baseline stagewise result changed")
    return result


def _validate_vamp_reference_parity(
    baseline_path: str | Path,
    vamp_path: str | Path,
) -> None:
    with Path(baseline_path).open("rb") as baseline_stream, Path(vamp_path).open(
        "rb"
    ) as vamp_stream:
        for baseline_line, vamp_line in zip(
            baseline_stream,
            vamp_stream,
            strict=True,
        ):
            baseline = _object(json.loads(baseline_line), "baseline parity row")
            vamp = _object(json.loads(vamp_line), "VAMP parity row")
            if any(
                baseline[field] != vamp[field]
                for field in ("stage_index", "story_id", "task_noun")
            ) or baseline["vamp_tensor_checksum"] != vamp["stage_tensor_checksum"]:
                raise ValueError("baseline and VAMP stagewise keys differ")
            baseline_results = _object(baseline["results"], "baseline parity results")
            vamp_oracle = _object(
                _object(vamp["results"], "VAMP parity results")["oracle"],
                "VAMP oracle",
            )
            if any(
                int(_object(baseline_results[condition], condition)["token_count"])
                != int(vamp_oracle["token_count"])
                for condition in BASELINE_CONDITIONS
            ):
                raise ValueError("baseline and VAMP suffix budgets differ")


def _aggregate_cells(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    stories = sum(int(cell["story_count"]) for cell in cells)
    tokens = sum(int(cell["token_count"]) for cell in cells)
    return {
        "mean_deficit_vs_independent": sum(
            float(cell["mean_deficit_vs_independent"]) * int(cell["story_count"])
            for cell in cells
        )
        / stories,
        "story_count": stories,
        "story_mean_nll": sum(
            float(cell["story_mean_nll"]) * int(cell["story_count"])
            for cell in cells
        )
        / stories,
        "token_count": tokens,
        "token_mean_nll": sum(
            float(cell["token_mean_nll"]) * int(cell["token_count"])
            for cell in cells
        )
        / tokens,
    }


def _longitudinal_metrics(
    series: tuple[tuple[int, dict[str, object]], ...],
) -> dict[str, object]:
    introduction = series[0][1]
    final = series[-1][1]
    best_stage, best = min(series, key=lambda item: float(item[1]["story_mean_nll"]))
    return {
        "backward_transfer": float(introduction["story_mean_nll"])
        - float(final["story_mean_nll"]),
        "best_stage": best_stage,
        "best_story_mean_nll": float(best["story_mean_nll"]),
        "final_story_mean_nll": float(final["story_mean_nll"]),
        "forgetting": float(final["story_mean_nll"])
        - float(best["story_mean_nll"]),
        "introduction_story_mean_nll": float(introduction["story_mean_nll"]),
    }


def _condition_summary(
    condition: str,
    stages: tuple[dict[str, object], ...],
    task_metrics: tuple[dict[str, object], ...],
) -> dict[str, object]:
    final = _object(_object(stages[-1]["conditions"], "final conditions")[condition], condition)
    tasks = tuple(
        _object(_object(task["conditions"], "task conditions")[condition], condition)
        for task in task_metrics
    )
    return {
        "final_mean_deficit_vs_independent": float(
            final["mean_deficit_vs_independent"]
        ),
        "final_story_mean_nll": float(final["story_mean_nll"]),
        "final_token_mean_nll": float(final["token_mean_nll"]),
        "max_task_forgetting": max(float(task["forgetting"]) for task in tasks),
        "mean_backward_transfer": _mean(
            float(task["backward_transfer"]) for task in tasks
        ),
        "mean_task_forgetting": _mean(float(task["forgetting"]) for task in tasks),
    }


def _generation_entries(
    partition: NounsV2PartitionArtifact,
) -> dict[str, tuple[StoryIndexEntry, ...]]:
    return {
        task_id: load_story_index(partition, f"task-{task_id}-generation")
        for task_id in partition.task_ids
    }


def _expected_keys(
    task_ids: tuple[str, ...],
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]],
) -> set[tuple[str, str, str]]:
    expected = {
        (str(stage), task_id, entry.story_id)
        for stage in range(1, len(task_ids) + 1)
        for task_id in task_ids[:stage]
        for entry in entries_by_task[task_id]
    }
    if task_ids == TASK_IDS and len(expected) != STAGEWISE_CASE_COUNT:
        raise ValueError("canonical baseline stagewise case count changed")
    return expected


def _repair_interrupted_tail(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = stream.tell() - 1
        while position > 0:
            chunk_start = max(0, position - 64 * 1024)
            stream.seek(chunk_start)
            chunk = stream.read(position - chunk_start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                stream.truncate(chunk_start + newline + 1)
                stream.flush()
                os.fsync(stream.fileno())
                return
            position = chunk_start
        stream.truncate(0)
        stream.flush()
        os.fsync(stream.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    measured = tuple(values)
    if not measured:
        raise ValueError("baseline mean requires values")
    return sum(measured) / len(measured)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


__all__ = [
    "evaluate_stagewise_baselines",
    "summarize_baseline_stagewise_ledger",
    "summarize_baseline_stagewise_rows",
    "validate_baseline_stagewise_ledger",
]
