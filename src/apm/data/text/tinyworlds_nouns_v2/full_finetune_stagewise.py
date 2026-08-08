"""Bounded stagewise evaluation of sequential full-model fine-tuning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import json
import math
import os
from pathlib import Path

import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    EvaluationProgress,
    _HalfStoryCase,
    _half_story_chunks,
    _nll_by_node_per_window,
    _stack_token_batches,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    StoryIndexEntry,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    FULL_FINETUNE_CONDITIONS,
    FULL_FINETUNE_STAGEWISE_FORMAT,
    TASK_IDS,
    FullFinetuneStagewiseClRow,
    FullFinetuneStagewiseConditionResult,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.full_finetune import (
    FullFinetuneStage,
    nouns_v2_full_finetune_train_config,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise_common import (
    expected_stagewise_keys as _expected_keys,
    file_sha256 as _file_sha256,
    generation_entries as _generation_entries,
    longitudinal_metrics as _longitudinal_metrics,
    mean as _mean,
    object_record as _object,
    repair_interrupted_tail as _repair_interrupted_tail,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams


_TOP_LEVEL_FIELDS = {
    "format",
    "introduced_task",
    "result_sha256",
    "results",
    "stage_index",
    "stage_parameter_checksum",
    "story_id",
    "task_noun",
    "vamp_tensor_checksum",
}
_RESULT_FIELDS = {
    "condition",
    "mean_nll",
    "model_task",
    "token_count",
    "total_nll",
}


def evaluate_stagewise_full_finetune(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    full_stages: tuple[FullFinetuneStage, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stagewise_path: str | Path,
    output_path: str | Path,
    *,
    progress: EvaluationProgress | None = None,
) -> Path:
    """Evaluate the sequential full model over every learned-task triangle cell."""
    metadata = _stage_metadata(partition, full_stages, vamp_stages)
    entries_by_task = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries_by_task)
    output = Path(output_path)
    if output.is_file():
        completed = validate_full_finetune_stagewise_ledger(
            output,
            partition,
            full_stages,
            vamp_stages,
            require_complete=True,
            entries_by_task=entries_by_task,
        )
        if completed != expected:
            raise ValueError("published full-finetune stagewise coverage changed")
        _validate_vamp_reference_parity(output, vamp_stagewise_path)
        return output
    work = output.with_name(f".{output.name}.work")
    work.parent.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns-v2 full-finetune CL ledger: {work.resolve()}", flush=True)
    _repair_interrupted_tail(work)
    completed = validate_full_finetune_stagewise_ledger(
        work,
        partition,
        full_stages,
        vamp_stages,
        require_complete=False,
        entries_by_task=entries_by_task,
    )
    store = IndexedStoryStore(partition)
    total = len(expected)
    finished = len(completed)
    with work.open("ab") as ledger:
        for stage_index, (stage, vamp) in enumerate(
            zip(full_stages, vamp_stages),
            start=1,
        ):
            completed_at_stage = {
                (task_id, story_id)
                for existing_stage, task_id, story_id in completed
                if existing_stage == str(stage_index)
            }
            chunks = _half_story_chunks(
                partition,
                {
                    task_id: entries_by_task[task_id]
                    for task_id in partition.task_ids[:stage_index]
                },
                completed_at_stage,
                store,
                preset,
                vamp.model_config.max_position_embeddings,
            )
            loaded = load_gpt_neo_checkpoint(stage.checkpoint)
            packed = pack_lora_memory(
                vamp.vamp_graph,
                vamp.model_config,
                vamp.lora_config,
                vamp.max_nodes,
                vamp.max_edges,
            )
            for cases in chunks:
                totals = _score_cases(
                    loaded.params,
                    vamp,
                    packed,
                    cases,
                    preset.evaluation_chunk_size,
                )
                rows = tuple(
                    _result_row(partition, metadata, stage_index, case, total_nll)
                    for case, total_nll in zip(cases, totals)
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
                        progress("full-finetune-stagewise-cl", finished, total)
    if completed != expected:
        raise RuntimeError(
            f"full-finetune stagewise ledger has {len(completed):,} of {total:,} rows"
        )
    _validate_vamp_reference_parity(work, vamp_stagewise_path)
    os.replace(work, output)
    return output


def validate_full_finetune_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    full_stages: tuple[FullFinetuneStage, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    *,
    require_complete: bool,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]] | None = None,
) -> set[tuple[str, str, str]]:
    """Strictly validate full-model row identities, bindings, and coverage."""
    metadata = _stage_metadata(partition, full_stages, vamp_stages)
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
                raise ValueError("full-finetune stagewise ledger key changed")
            keys.add(key)
    if require_complete and keys != expected:
        raise ValueError(
            f"full-finetune stagewise ledger has {len(keys):,} of {len(expected):,} rows"
        )
    return keys


def summarize_full_finetune_stagewise_ledger(
    path: str | Path,
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    full_stages: tuple[FullFinetuneStage, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    vamp_stagewise_path: str | Path,
) -> dict[str, object]:
    """Stream the strict full-model ledger into continual-learning summaries."""
    source = Path(path)
    metadata = _stage_metadata(partition, full_stages, vamp_stages)
    entries = _generation_entries(partition)
    expected = _expected_keys(partition.task_ids, entries)
    keys: set[tuple[str, str, str]] = set()

    def validated_rows() -> Iterable[dict[str, object]]:
        with source.open("rb") as stream:
            for line in stream:
                row = _validated_row(line, metadata, partition.task_ids)
                key = (
                    str(row["stage_index"]),
                    str(row["task_noun"]),
                    str(row["story_id"]),
                )
                if key in keys or key not in expected:
                    raise ValueError("full-finetune report ledger keys changed")
                keys.add(key)
                yield row

    measured = summarize_full_finetune_stagewise_rows(
        validated_rows(),
        partition.task_ids,
    )
    if keys != expected:
        raise ValueError("full-finetune report ledger coverage changed")
    _validate_vamp_reference_parity(source, vamp_stagewise_path)
    train_config = nouns_v2_full_finetune_train_config(preset)
    core = {
        **measured,
        "format": FULL_FINETUNE_STAGEWISE_FORMAT,
        "ledger_sha256": _file_sha256(source),
        "parameter_checksums": [stage.parameter_checksum for stage in full_stages],
        "run_sha256": full_stages[-1].run_sha256,
        "train_config": {
            "batch_size": train_config.batch_size,
            "gradient_clip_norm": train_config.gradient_clip_norm,
            "learning_rate": train_config.learning_rate,
            "steps": train_config.steps,
            "weight_decay": train_config.weight_decay,
        },
        "vamp_stagewise_sha256": _file_sha256(Path(vamp_stagewise_path)),
    }
    return {**core, "summary_sha256": record_sha256(core)}


def summarize_full_finetune_stagewise_rows(
    rows: Iterable[dict[str, object]],
    task_ids: tuple[str, ...] = TASK_IDS,
) -> dict[str, object]:
    """Compute full-model stage curves, forgetting, and backward transfer."""
    positions = {task_id: index for index, task_id in enumerate(task_ids, start=1)}
    values: dict[tuple[int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )
    row_count = 0
    for row in rows:
        stage = int(row["stage_index"])
        task = str(row["task_noun"])
        if task not in positions or not positions[task] <= stage <= len(task_ids):
            raise ValueError("full-finetune summary row is outside the learned triangle")
        result = _object(row["results"], "full-finetune results")[
            FULL_FINETUNE_CONDITIONS[0]
        ]
        measured = _object(result, "full-finetune result")
        bucket = values[(stage, task)]
        bucket[0] += 1.0
        bucket[1] += float(measured["mean_nll"])
        bucket[2] += float(measured["total_nll"])
        bucket[3] += float(measured["token_count"])
        row_count += 1

    def task_stage(stage: int, task: str) -> dict[str, object]:
        count, mean_sum, total, tokens = values[(stage, task)]
        if count <= 0.0 or tokens <= 0.0:
            raise ValueError("full-finetune summary is missing a task/stage cell")
        return {
            "story_count": int(count),
            "story_mean_nll": mean_sum / count,
            "token_count": int(tokens),
            "token_mean_nll": total / tokens,
        }

    stages = tuple(
        {
            "conditions": {
                FULL_FINETUNE_CONDITIONS[0]: _aggregate_cells(
                    tuple(task_stage(stage, task) for task in task_ids[:stage])
                )
            },
            "introduced_task": task_ids[stage - 1],
            "learned_task_count": stage,
            "stage_index": stage,
            "story_count": sum(
                int(task_stage(stage, task)["story_count"])
                for task in task_ids[:stage]
            ),
        }
        for stage in range(1, len(task_ids) + 1)
    )
    task_metrics = tuple(
        {
            "conditions": {
                FULL_FINETUNE_CONDITIONS[0]: _longitudinal_metrics(
                    tuple(
                        (stage, task_stage(stage, task))
                        for stage in range(positions[task], len(task_ids) + 1)
                    )
                )
            },
            "introduction_stage": positions[task],
            "task": task,
        }
        for task in task_ids
    )
    final = _object(
        _object(stages[-1]["conditions"], "final conditions")[
            FULL_FINETUNE_CONDITIONS[0]
        ],
        "full-finetune final",
    )
    task_values = tuple(
        _object(
            _object(task["conditions"], "task conditions")[
                FULL_FINETUNE_CONDITIONS[0]
            ],
            "full-finetune task",
        )
        for task in task_metrics
    )
    condition_summary = {
        "final_story_mean_nll": float(final["story_mean_nll"]),
        "final_token_mean_nll": float(final["token_mean_nll"]),
        "max_task_forgetting": max(float(task["forgetting"]) for task in task_values),
        "mean_backward_transfer": _mean(
            float(task["backward_transfer"]) for task in task_values
        ),
        "mean_task_forgetting": _mean(
            float(task["forgetting"]) for task in task_values
        ),
    }
    return {
        "condition_summaries": {
            FULL_FINETUNE_CONDITIONS[0]: condition_summary,
        },
        "row_count": row_count,
        "stage_count": len(task_ids),
        "stages": list(stages),
        "task_metrics": list(task_metrics),
    }


def _result_row(
    partition: NounsV2PartitionArtifact,
    metadata: dict[int, dict[str, str]],
    stage_index: int,
    case: _HalfStoryCase,
    total_nll: float,
) -> dict[str, object]:
    token_count = int(np.sum(case.suffix_windows.loss_mask))
    return FullFinetuneStagewiseClRow(
        stage_index=stage_index,
        introduced_task=partition.task_ids[stage_index - 1],
        stage_parameter_checksum=metadata[stage_index]["parameter_checksum"],
        vamp_tensor_checksum=metadata[stage_index]["vamp_checksum"],
        task_noun=case.task_id,
        story_id=case.entry.story_id,
        results=(
            FullFinetuneStagewiseConditionResult(
                FULL_FINETUNE_CONDITIONS[0],
                partition.task_ids[stage_index - 1],
                total_nll,
                token_count,
                total_nll / token_count,
            ),
        ),
    ).as_record()


def _score_cases(
    params: GptNeoParams,
    vamp: LanguageAdaptationArtifact,
    packed: PackedLoraMemory,
    cases: tuple[_HalfStoryCase, ...],
    microbatch_size: int,
) -> tuple[float, ...]:
    windows = _stack_token_batches(tuple(case.suffix_windows for case in cases))
    per_window = _nll_by_node_per_window(
        params,
        vamp,
        packed,
        windows,
        microbatch_size,
        node_indices=(0,),
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
    full_stages: tuple[FullFinetuneStage, ...],
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
) -> dict[int, dict[str, str]]:
    if len(full_stages) != len(partition.task_ids) or len(vamp_stages) != len(
        partition.task_ids
    ):
        raise ValueError("full-finetune audit requires every full and VAMP stage")
    metadata: dict[int, dict[str, str]] = {}
    for stage_index, (full, vamp) in enumerate(
        zip(full_stages, vamp_stages),
        start=1,
    ):
        expected = partition.task_ids[:stage_index]
        if full.task_order != expected or tuple(str(task) for task in vamp.task_order) != expected:
            raise ValueError("full-finetune stage is not the canonical learned prefix")
        metadata[stage_index] = {
            "introduced_task": expected[-1],
            "parameter_checksum": full.parameter_checksum,
            "vamp_checksum": vamp.tensor_checksum,
        }
    return metadata


def _validated_row(
    line: bytes,
    metadata: dict[int, dict[str, str]],
    task_ids: tuple[str, ...],
) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise ValueError("full-finetune stagewise ledger has an interrupted tail")
    row = json.loads(line)
    if type(row) is not dict or canonical_json_bytes(row) != line:
        raise ValueError("full-finetune stagewise ledger is not canonical JSONL")
    core = {key: value for key, value in row.items() if key != "result_sha256"}
    stage = row.get("stage_index")
    if (
        set(row) != _TOP_LEVEL_FIELDS
        or row.get("format") != FULL_FINETUNE_STAGEWISE_FORMAT
        or row.get("result_sha256") != record_sha256(core)
        or type(stage) is not int
        or stage not in metadata
    ):
        raise ValueError("full-finetune stagewise row identity changed")
    info = metadata[stage]
    task = row.get("task_noun")
    results = _object(row.get("results"), "full-finetune results")
    if (
        row.get("introduced_task") != info["introduced_task"]
        or row.get("stage_parameter_checksum") != info["parameter_checksum"]
        or row.get("vamp_tensor_checksum") != info["vamp_checksum"]
        or type(task) is not str
        or task not in task_ids[:stage]
        or set(results) != set(FULL_FINETUNE_CONDITIONS)
    ):
        raise ValueError("full-finetune stagewise row binding changed")
    result = _object(results[FULL_FINETUNE_CONDITIONS[0]], "full-finetune result")
    numeric = tuple(result.get(field) for field in ("mean_nll", "total_nll"))
    if (
        set(result) != _RESULT_FIELDS
        or result.get("condition") != FULL_FINETUNE_CONDITIONS[0]
        or result.get("model_task") != info["introduced_task"]
        or type(result.get("token_count")) is not int
        or int(result["token_count"]) <= 0
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in numeric
        )
        or not math.isclose(
            float(result["mean_nll"]),
            float(result["total_nll"]) / int(result["token_count"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("full-finetune stagewise result changed")
    return row


def _validate_vamp_reference_parity(
    full_path: str | Path,
    vamp_path: str | Path,
) -> None:
    with Path(full_path).open("rb") as full_stream, Path(vamp_path).open(
        "rb"
    ) as vamp_stream:
        for full_line, vamp_line in zip(full_stream, vamp_stream, strict=True):
            full = _object(json.loads(full_line), "full-finetune parity row")
            vamp = _object(json.loads(vamp_line), "VAMP parity row")
            if any(
                full[field] != vamp[field]
                for field in ("stage_index", "story_id", "task_noun")
            ) or full["vamp_tensor_checksum"] != vamp["stage_tensor_checksum"]:
                raise ValueError("full-finetune and VAMP stagewise keys differ")
            result = _object(full["results"], "full-finetune parity results")[
                FULL_FINETUNE_CONDITIONS[0]
            ]
            oracle = _object(vamp["results"], "VAMP parity results")["oracle"]
            if int(_object(result, "full-finetune parity result")["token_count"]) != int(
                _object(oracle, "VAMP oracle")["token_count"]
            ):
                raise ValueError("full-finetune and VAMP suffix budgets differ")


def _aggregate_cells(cells: tuple[dict[str, object], ...]) -> dict[str, object]:
    stories = sum(int(cell["story_count"]) for cell in cells)
    tokens = sum(int(cell["token_count"]) for cell in cells)
    return {
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


__all__ = [
    "evaluate_stagewise_full_finetune",
    "summarize_full_finetune_stagewise_ledger",
    "summarize_full_finetune_stagewise_rows",
    "validate_full_finetune_stagewise_ledger",
]
