"""Resumable longitudinal evaluation of every immutable nouns-v2 VAMP stage."""

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
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    _half_story_chunks,
    _nll_by_node_per_window,
    _node_path,
    _prefix_chunk_selections,
    _stack_token_batches,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    CONDITIONS,
    STAGEWISE_CASE_COUNT,
    STAGEWISE_FORMAT,
    TASK_IDS,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    StagewiseClRow,
    StagewiseConditionResult,
    canonical_json_bytes,
    record_sha256,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora_memory import pack_lora_memory


_TOP_LEVEL_FIELDS = {
    "format",
    "introduced_task",
    "result_sha256",
    "results",
    "stage_index",
    "stage_tensor_checksum",
    "story_id",
    "task_noun",
}
_CONDITION_FIELDS = {
    "condition",
    "mean_nll",
    "oracle_match",
    "regret_vs_oracle",
    "selected_node",
    "selected_path",
    "token_count",
    "total_nll",
}


def evaluate_stagewise_continual_learning(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    output_path: str | Path,
    *,
    progress=None,
) -> Path:
    """Evaluate every learned task at every subsequent VAMP graph stage."""
    stage_metadata = _stage_metadata(partition, adaptations)
    entries_by_task = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries_by_task)
    output = Path(output_path)
    if output.is_file():
        completed = validate_stagewise_ledger(
            output,
            partition,
            adaptations,
            require_complete=True,
            entries_by_task=entries_by_task,
        )
        if completed != expected:
            raise ValueError("published stagewise ledger coverage changed")
        return output

    work = output.with_name(f".{output.name}.work")
    work.parent.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns-v2 stagewise CL ledger: {work.resolve()}", flush=True)
    _repair_interrupted_tail(work)
    completed = validate_stagewise_ledger(
        work,
        partition,
        adaptations,
        require_complete=False,
        entries_by_task=entries_by_task,
    )
    if not completed <= expected:
        raise ValueError("stagewise work ledger contains unexpected rows")

    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    store = IndexedStoryStore(partition)
    total = len(expected)
    finished = len(completed)
    with work.open("ab") as ledger:
        for stage_index, adaptation in enumerate(adaptations, start=1):
            stage_key = str(stage_index)
            learned_entries = {
                task_id: entries_by_task[task_id]
                for task_id in partition.task_ids[:stage_index]
            }
            completed_at_stage = {
                (task_id, story_id)
                for existing_stage, task_id, story_id in completed
                if existing_stage == stage_key
            }
            packed = pack_lora_memory(
                adaptation.vamp_graph,
                adaptation.model_config,
                adaptation.lora_config,
                adaptation.max_nodes,
                adaptation.max_edges,
            )
            chunks = _half_story_chunks(
                partition,
                learned_entries,
                completed_at_stage,
                store,
                preset,
                adaptation.model_config.max_position_embeddings,
            )
            for cases in chunks:
                selections = _prefix_chunk_selections(
                    cases,
                    loaded.params,
                    adaptation,
                    packed,
                    preset.evaluation_chunk_size,
                )
                suffix_windows = _stack_token_batches(
                    tuple(case.suffix_windows for case in cases)
                )
                selected_indices = tuple(
                    sorted(
                        {
                            node_index
                            for selection in selections
                            for node_index in selection.values()
                        }
                    )
                )
                per_window = _nll_by_node_per_window(
                    loaded.params,
                    adaptation,
                    packed,
                    suffix_windows,
                    preset.evaluation_chunk_size,
                    node_indices=selected_indices,
                )
                node_row = {
                    node_index: row_index
                    for row_index, node_index in enumerate(selected_indices)
                }
                boundaries = np.cumsum(
                    (0,)
                    + tuple(
                        case.suffix_windows.input_ids.shape[0] for case in cases
                    )
                )
                payloads: list[bytes] = []
                keys: list[tuple[str, str, str]] = []
                for case, selection, start, stop in zip(
                    cases,
                    selections,
                    boundaries[:-1],
                    boundaries[1:],
                ):
                    token_count = int(np.sum(case.suffix_windows.loss_mask))
                    totals = {
                        node_index: float(
                            np.sum(
                                per_window[node_row[node_index], start:stop],
                                dtype=np.float64,
                            )
                        )
                        for node_index in set(selection.values())
                    }
                    oracle_total = totals[case.oracle_index]
                    oracle_mean = oracle_total / token_count
                    results = tuple(
                        StagewiseConditionResult(
                            condition=condition,
                            selected_node=str(
                                adaptation.vamp_graph.nodes[
                                    selection[condition]
                                ].node_id
                            ),
                            selected_path=_node_path(
                                adaptation,
                                selection[condition],
                            ),
                            oracle_match=selection[condition] == case.oracle_index,
                            total_nll=totals[selection[condition]],
                            token_count=token_count,
                            mean_nll=totals[selection[condition]] / token_count,
                            regret_vs_oracle=(
                                totals[selection[condition]] / token_count
                                - oracle_mean
                            ),
                        )
                        for condition in CONDITIONS
                    )
                    row = StagewiseClRow(
                        stage_index=stage_index,
                        introduced_task=partition.task_ids[stage_index - 1],
                        stage_tensor_checksum=stage_metadata[stage_index]["checksum"],
                        task_noun=case.task_id,
                        story_id=case.entry.story_id,
                        results=results,
                    ).as_record()
                    payloads.append(canonical_json_bytes(row))
                    keys.append((stage_key, case.task_id, case.entry.story_id))
                ledger.write(b"".join(payloads))
                ledger.flush()
                os.fsync(ledger.fileno())
                for key in keys:
                    completed.add(key)
                    finished += 1
                    if progress is not None:
                        progress("stagewise-cl", finished, total)
    if completed != expected:
        raise RuntimeError(
            f"stagewise ledger has {len(completed):,} of {total:,} rows"
        )
    os.replace(work, output)
    return output


def expected_stagewise_row_count(partition: NounsV2PartitionArtifact) -> int:
    """Return the exact task/story/stage count implied by the partition."""
    count = sum(
        task.validation_story_count * (len(partition.task_ids) - index)
        for index, task in enumerate(partition.tasks)
    )
    if partition.task_ids == TASK_IDS and count != STAGEWISE_CASE_COUNT:
        raise ValueError("canonical nouns-v2 stagewise case count changed")
    return count


def validate_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    *,
    require_complete: bool,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]] | None = None,
) -> set[tuple[str, str, str]]:
    """Strictly validate identities, stage bindings, paths, and ledger coverage."""
    metadata = _stage_metadata(partition, adaptations)
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
            row = _validated_row(line, metadata)
            key = (
                str(row["stage_index"]),
                str(row["task_noun"]),
                str(row["story_id"]),
            )
            if key in keys:
                raise ValueError("stagewise ledger contains a duplicate row")
            if key not in expected:
                raise ValueError("stagewise ledger contains an unexpected row")
            keys.add(key)
    if require_complete and keys != expected:
        raise ValueError(
            f"stagewise ledger has {len(keys):,} of {len(expected):,} expected rows"
        )
    return keys


def summarize_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
) -> dict[str, object]:
    """Stream a strict ledger into report-ready continual-learning summaries."""
    source = Path(path)
    metadata = _stage_metadata(partition, adaptations)
    entries = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries)
    keys: set[tuple[str, str, str]] = set()

    def validated_rows():
        with source.open("rb") as stream:
            for line in stream:
                row = _validated_row(line, metadata)
                key = (
                    str(row["stage_index"]),
                    str(row["task_noun"]),
                    str(row["story_id"]),
                )
                if key in keys or key not in expected:
                    raise ValueError("stagewise report ledger keys changed")
                keys.add(key)
                yield row

    measured = summarize_stagewise_rows(validated_rows(), partition.task_ids)
    if keys != expected:
        raise ValueError("stagewise report ledger coverage changed")
    summary = {
        **measured,
        "format": STAGEWISE_FORMAT,
        "ledger_sha256": _file_sha256(source),
        "stage_tensor_checksums": [
            metadata[index]["checksum"]
            for index in range(1, len(adaptations) + 1)
        ],
    }
    return {**summary, "summary_sha256": record_sha256(summary)}


def summarize_stagewise_rows(
    rows: Iterable[dict[str, object]],
    task_ids: tuple[str, ...] = TASK_IDS,
) -> dict[str, object]:
    """Compute stage curves, forgetting, and backward transfer from rows."""
    task_position = {task_id: index for index, task_id in enumerate(task_ids, start=1)}
    values: dict[tuple[int, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    row_count = 0
    stage_story_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        stage = int(row["stage_index"])
        task = str(row["task_noun"])
        if task not in task_position or not task_position[task] <= stage <= len(task_ids):
            raise ValueError("stagewise summary row is outside the learned-task triangle")
        results = _object(row["results"], "stagewise results")
        if set(results) != set(CONDITIONS):
            raise ValueError("stagewise summary conditions changed")
        row_count += 1
        stage_story_counts[stage] += 1
        for condition in CONDITIONS:
            result = _object(results[condition], condition)
            bucket = values[(stage, task, condition)]
            bucket[0] += 1.0
            bucket[1] += float(result["mean_nll"])
            bucket[2] += float(result["total_nll"])
            bucket[3] += float(result["token_count"])
            bucket[4] += float(bool(result["oracle_match"]))
            bucket[5] += float(result["regret_vs_oracle"])

    def task_stage(stage: int, task: str, condition: str) -> dict[str, object]:
        count, mean_sum, total, tokens, matches, regret = values[
            (stage, task, condition)
        ]
        if count <= 0.0 or tokens <= 0.0:
            raise ValueError("stagewise summary is missing a task/stage cell")
        return {
            "mean_regret": regret / count,
            "routing_accuracy": matches / count,
            "story_count": int(count),
            "story_mean_nll": mean_sum / count,
            "token_count": int(tokens),
            "token_mean_nll": total / tokens,
        }

    stage_summaries = []
    for stage in range(1, len(task_ids) + 1):
        conditions = {}
        for condition in CONDITIONS:
            cells = tuple(
                task_stage(stage, task, condition) for task in task_ids[:stage]
            )
            stories = sum(int(cell["story_count"]) for cell in cells)
            tokens = sum(int(cell["token_count"]) for cell in cells)
            conditions[condition] = {
                "mean_regret": sum(
                    float(cell["mean_regret"]) * int(cell["story_count"])
                    for cell in cells
                )
                / stories,
                "routing_accuracy": sum(
                    float(cell["routing_accuracy"]) * int(cell["story_count"])
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
        if stage_story_counts[stage] != int(conditions["oracle"]["story_count"]):
            raise ValueError("stagewise summary stage coverage is inconsistent")
        stage_summaries.append(
            {
                "conditions": conditions,
                "introduced_task": task_ids[stage - 1],
                "learned_task_count": stage,
                "stage_index": stage,
                "story_count": stage_story_counts[stage],
            }
        )

    per_task = []
    for task in task_ids:
        introduction = task_position[task]
        condition_metrics = {}
        for condition in CONDITIONS:
            series = tuple(
                (stage, task_stage(stage, task, condition))
                for stage in range(introduction, len(task_ids) + 1)
            )
            intro = series[0][1]
            final = series[-1][1]
            best_stage, best = min(
                series,
                key=lambda item: float(item[1]["story_mean_nll"]),
            )
            condition_metrics[condition] = {
                "accuracy_change": float(final["routing_accuracy"])
                - float(intro["routing_accuracy"]),
                "backward_transfer": float(intro["story_mean_nll"])
                - float(final["story_mean_nll"]),
                "best_stage": best_stage,
                "best_story_mean_nll": float(best["story_mean_nll"]),
                "final_routing_accuracy": float(final["routing_accuracy"]),
                "final_story_mean_nll": float(final["story_mean_nll"]),
                "forgetting": float(final["story_mean_nll"])
                - float(best["story_mean_nll"]),
                "introduction_routing_accuracy": float(
                    intro["routing_accuracy"]
                ),
                "introduction_story_mean_nll": float(
                    intro["story_mean_nll"]
                ),
            }
        per_task.append(
            {
                "conditions": condition_metrics,
                "introduction_stage": introduction,
                "task": task,
            }
        )

    condition_summaries = {}
    for condition in CONDITIONS:
        task_values = tuple(
            _object(_object(row["conditions"], "task conditions")[condition], condition)
            for row in per_task
        )
        final_stage = _object(
            _object(stage_summaries[-1]["conditions"], "final conditions")[condition],
            condition,
        )
        condition_summaries[condition] = {
            "final_routing_accuracy": float(final_stage["routing_accuracy"]),
            "final_story_mean_nll": float(final_stage["story_mean_nll"]),
            "final_token_mean_nll": float(final_stage["token_mean_nll"]),
            "introduction_macro_routing_accuracy": _mean(
                float(value["introduction_routing_accuracy"])
                for value in task_values
            ),
            "max_task_forgetting": max(
                float(value["forgetting"]) for value in task_values
            ),
            "mean_backward_transfer": _mean(
                float(value["backward_transfer"]) for value in task_values
            ),
            "mean_route_accuracy_change": _mean(
                float(value["accuracy_change"]) for value in task_values
            ),
            "mean_task_forgetting": _mean(
                float(value["forgetting"]) for value in task_values
            ),
        }
    oracle_drifts = tuple(
        abs(
            float(task_stage(stage, task, "oracle")["story_mean_nll"])
            - float(
                task_stage(task_position[task], task, "oracle")[
                    "story_mean_nll"
                ]
            )
        )
        for task in task_ids
        for stage in range(task_position[task], len(task_ids) + 1)
    )
    return {
        "condition_summaries": condition_summaries,
        "oracle_max_absolute_drift": max(oracle_drifts),
        "row_count": row_count,
        "stage_count": len(task_ids),
        "stages": stage_summaries,
        "task_metrics": per_task,
    }


def _stage_metadata(
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
) -> dict[int, dict[str, object]]:
    if len(adaptations) != len(partition.task_ids) or not adaptations:
        raise ValueError("stagewise audit requires every frozen VAMP stage")
    metadata = {}
    for stage_index, adaptation in enumerate(adaptations, start=1):
        expected_tasks = partition.task_ids[:stage_index]
        node_ids = tuple(str(node.node_id) for node in adaptation.vamp_graph.nodes)
        if (
            tuple(str(task) for task in adaptation.task_order) != expected_tasks
            or node_ids != ("root", *expected_tasks)
            or len(adaptation.vamp_stages) != stage_index
        ):
            raise ValueError("VAMP stage is not the canonical learned-task prefix")
        metadata[stage_index] = {
            "adaptation": adaptation,
            "checksum": adaptation.tensor_checksum,
            "introduced_task": expected_tasks[-1],
            "paths": {
                str(node.node_id): _node_path(adaptation, node_index)
                for node_index, node in enumerate(adaptation.vamp_graph.nodes)
            },
        }
    return metadata


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
    return {
        (str(stage), task_id, entry.story_id)
        for stage in range(1, len(task_ids) + 1)
        for task_id in task_ids[:stage]
        for entry in entries_by_task[task_id]
    }


def _validated_row(
    line: bytes,
    metadata: dict[int, dict[str, object]],
) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise ValueError("stagewise ledger has an interrupted tail")
    row = json.loads(line)
    if type(row) is not dict or canonical_json_bytes(row) != line:
        raise ValueError("stagewise ledger is not canonical JSONL")
    if set(row) != _TOP_LEVEL_FIELDS:
        raise ValueError("stagewise row fields changed")
    supplied = row["result_sha256"]
    core = {key: value for key, value in row.items() if key != "result_sha256"}
    if row["format"] != STAGEWISE_FORMAT or supplied != record_sha256(core):
        raise ValueError("stagewise row identity changed")
    stage = row["stage_index"]
    if type(stage) is not int or stage not in metadata:
        raise ValueError("stagewise row stage changed")
    stage_info = metadata[stage]
    task = row["task_noun"]
    if (
        row["introduced_task"] != stage_info["introduced_task"]
        or row["stage_tensor_checksum"] != stage_info["checksum"]
        or type(task) is not str
        or task not in TASK_IDS[:stage]
    ):
        raise ValueError("stagewise row binding changed")
    paths = _object(stage_info["paths"], "stage paths")
    results = _object(row["results"], "stagewise results")
    if set(results) != set(CONDITIONS):
        raise ValueError("stagewise result conditions changed")
    validated: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        result = _object(results[condition], condition)
        if set(result) != _CONDITION_FIELDS or result["condition"] != condition:
            raise ValueError("stagewise condition fields changed")
        selected = result["selected_node"]
        path = result["selected_path"]
        if (
            type(selected) is not str
            or selected not in paths
            or type(path) is not list
            or tuple(path) != tuple(paths[selected])
            or type(result["oracle_match"]) is not bool
            or result["oracle_match"] != (selected == task)
            or type(result["token_count"]) is not int
            or int(result["token_count"]) <= 0
        ):
            raise ValueError("stagewise route metadata changed")
        numeric = tuple(result[field] for field in ("total_nll", "mean_nll", "regret_vs_oracle"))
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("stagewise NLL metrics changed")
        if float(result["total_nll"]) < 0.0 or float(result["mean_nll"]) < 0.0:
            raise ValueError("stagewise NLL metrics must be nonnegative")
        if not math.isclose(
            float(result["mean_nll"]),
            float(result["total_nll"]) / int(result["token_count"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("stagewise mean and total NLL differ")
        validated[condition] = result
    oracle = validated["oracle"]
    if validated["base"]["selected_node"] != "root" or oracle["selected_node"] != task:
        raise ValueError("stagewise stored baseline selections changed")
    token_counts = {int(result["token_count"]) for result in validated.values()}
    if len(token_counts) != 1:
        raise ValueError("stagewise conditions score different suffix budgets")
    oracle_mean = float(oracle["mean_nll"])
    for result in validated.values():
        if not math.isclose(
            float(result["regret_vs_oracle"]),
            float(result["mean_nll"]) - oracle_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("stagewise oracle regret changed")
    return row


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
        raise ValueError("stagewise mean requires values")
    return sum(measured) / len(measured)


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


__all__ = [
    "evaluate_stagewise_continual_learning",
    "expected_stagewise_row_count",
    "summarize_stagewise_ledger",
    "summarize_stagewise_rows",
    "validate_stagewise_ledger",
]
