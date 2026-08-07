"""Streamed whole-story routing, suffix NLL, and greedy completion evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_routing import route_language_prefix, trace_ebt_language_prefix
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    CONDITIONS,
    Condition,
    NounPartitionArtifact,
    NounsExperimentPreset,
    WholeStoryNllRow,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
    load_story_index,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.generation import greedy_generate
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora_memory import edge_coefficients_for_node, pack_lora_memory
from apm.lm.losses import per_token_nll
from apm.lm.text import TextTokenizer
from apm.lm.text_data import TokenBatch
from apm.memory.content_addressing import HopfieldConfig, hopfield_address
from apm.memory.content_keys import encode_frozen_base_content
from apm.memory.graph import memory_node_path
from apm.memory.address_refinement import EbtConfig


WHOLE_STORY_FORMAT = "tinyworlds-nouns-whole-story-nll-v1"
HALF_STORY_FORMAT = "tinyworlds-nouns-half-story-generation-v1"
_MAX_EBT_ROWS = 8
_MAX_GENERATION_ROWS = 72
_GENERATION_WINDOW_CHUNK_MULTIPLIER = 4
EvaluationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class PrefixOnlyQuery:
    """Router-visible midpoint prefix with no continuation-bearing field."""

    story_id: str
    prompt_token_ids: tuple[int, ...]
    router_batch: RouterBatch

    def __post_init__(self) -> None:
        require_sha256(self.story_id, "prefix-only story")
        if len(self.prompt_token_ids) < 2:
            raise ValueError("prefix-only routing requires at least two tokens")
        if self.router_batch.input_ids.shape[0] != 1:
            raise ValueError("prefix-only routing requires one router row")


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """One addressed condition's true suffix loss and greedy continuation."""

    condition: Condition
    selected_node: str
    selected_path: tuple[str, ...]
    total_nll: float
    token_count: int
    mean_nll: float
    generated_continuation: str
    generated_token_count: int
    eos_reached: bool

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS or not self.selected_node:
            raise ValueError("completion condition or node is invalid")
        if not self.selected_path or self.token_count <= 0:
            raise ValueError("completion path and suffix token count are required")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.total_nll, self.mean_nll)
        ):
            raise ValueError("completion NLL must be finite and nonnegative")
        if self.generated_token_count < 0 or type(self.eos_reached) is not bool:
            raise ValueError("completion generation metadata is invalid")

    def as_record(self) -> dict[str, object]:
        """Return one canonical nested generation result."""
        return {
            "condition": self.condition,
            "eos_reached": self.eos_reached,
            "generated_continuation": self.generated_continuation,
            "generated_token_count": self.generated_token_count,
            "mean_nll": self.mean_nll,
            "selected_node": self.selected_node,
            "selected_path": list(self.selected_path),
            "token_count": self.token_count,
            "total_nll": self.total_nll,
        }


@dataclass(frozen=True, slots=True)
class HalfStoryGenerationRow:
    """One task/story prefix, reference, and all six addressed continuations."""

    task_noun: str
    story_id: str
    prefix: str
    reference_continuation: str
    full_original_story: str
    results: tuple[CompletionResult, ...]

    def __post_init__(self) -> None:
        require_sha256(self.story_id, "half-story story")
        if not self.task_noun or not self.prefix or not self.full_original_story:
            raise ValueError("half-story text and task metadata must be nonempty")
        if tuple(result.condition for result in self.results) != CONDITIONS:
            raise ValueError("half-story rows require all six conditions in order")

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSONL generation row."""
        return {
            "format": HALF_STORY_FORMAT,
            "full_original_story": self.full_original_story,
            "prefix": self.prefix,
            "reference_continuation": self.reference_continuation,
            "results": {
                result.condition: result.as_record() for result in self.results
            },
            "story_id": self.story_id,
            "task_noun": self.task_noun,
        }


@dataclass(frozen=True, slots=True)
class _WholeStoryCase:
    task_id: str
    entry: StoryIndexEntry
    oracle_index: int
    windows: TokenBatch


@dataclass(frozen=True, slots=True)
class _HalfStoryCase:
    task_id: str
    entry: StoryIndexEntry
    oracle_index: int
    tokens: tuple[int, ...]
    midpoint: int
    query: PrefixOnlyQuery
    suffix_windows: TokenBatch
    budget: int


@dataclass(frozen=True, slots=True)
class _GenerationBin:
    """One deterministic group bounded by distinct rows and context width."""

    indices: tuple[int, ...]
    row_count: int
    prompt_width: int
    budget: int


def build_prefix_only_query(
    story_id: str,
    token_ids: tuple[int, ...],
    pad_token_id: int,
    maximum_position_embeddings: int,
) -> PrefixOnlyQuery:
    """Split at the exact token midpoint and expose only its first half."""
    midpoint = len(token_ids) // 2
    prefix = token_ids[:midpoint]
    if len(prefix) < 2 or len(prefix) >= maximum_position_embeddings:
        raise ValueError("generation prefix leaves no model-position output capacity")
    width = len(prefix) - 1
    input_ids = np.full((1, width), pad_token_id, dtype=np.int32)
    target_ids = np.full((1, width), pad_token_id, dtype=np.int32)
    attention = np.zeros((1, width), dtype=np.bool_)
    transitions = len(prefix) - 1
    input_ids[0, :transitions] = prefix[:-1]
    target_ids[0, :transitions] = prefix[1:]
    attention[0, :transitions] = True
    return PrefixOnlyQuery(
        story_id,
        prefix,
        RouterBatch(input_ids, attention, target_ids, attention),
    )


def evaluate_whole_story_nll(
    partition: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
    adaptation: LanguageAdaptationArtifact,
    output_path: str | Path,
    *,
    progress: EvaluationProgress | None = None,
) -> Path:
    """Stream every official-validation task/story under all six conditions."""
    output = Path(output_path)
    entries_by_task = {
        task_id: load_story_index(partition, f"task-{task_id}-validation")
        for task_id in partition.task_ids
    }
    expected_keys = {
        (task_id, entry.story_id, condition)
        for task_id, entries in entries_by_task.items()
        for entry in entries
        for condition in CONDITIONS
    }
    if output.is_file():
        if _completed_keys(output, ("task_noun", "story_id", "condition")) != expected_keys:
            raise ValueError("published whole-story ledger coverage changed")
        return output
    work = output.with_name(f".{output.name}.work")
    work.parent.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns streamed NLL ledger: {work.resolve()}", flush=True)
    completed = _completed_keys(work, ("task_noun", "story_id", "condition"))
    if not completed <= expected_keys:
        raise ValueError("whole-story work ledger contains unexpected rows")
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    packed = pack_lora_memory(
        adaptation.vamp_graph,
        adaptation.model_config,
        adaptation.lora_config,
        adaptation.max_nodes,
        adaptation.max_edges,
    )
    store = IndexedStoryStore(partition)
    total = sum(len(entries) for entries in entries_by_task.values())
    finished_stories = sum(
        all((task, entry.story_id, condition) in completed for condition in CONDITIONS)
        for task, entries in entries_by_task.items()
        for entry in entries
    )
    chunks = _whole_story_chunks(
        partition,
        entries_by_task,
        completed,
        store,
        preset,
    )
    with work.open("ab") as ledger:
        for cases in chunks:
            windows = _stack_token_batches(tuple(case.windows for case in cases))
            per_window_totals = _nll_by_node_per_window(
                loaded.params,
                adaptation,
                packed,
                windows,
                preset.evaluation_chunk_size,
            )
            boundaries = np.cumsum(
                (0,) + tuple(case.windows.input_ids.shape[0] for case in cases)
            )
            node_totals_by_case = tuple(
                tuple(
                    float(value)
                    for value in np.sum(
                        per_window_totals[:, start:stop],
                        axis=1,
                        dtype=np.float64,
                    )
                )
                for start, stop in zip(boundaries[:-1], boundaries[1:])
            )
            token_counts = tuple(
                int(np.sum(case.windows.loss_mask)) for case in cases
            )
            selections = _whole_story_chunk_selections(
                loaded.params,
                adaptation,
                packed,
                windows,
                boundaries,
                node_totals_by_case,
                preset.evaluation_chunk_size,
            )
            for case, node_totals, token_count, selected in zip(
                cases,
                node_totals_by_case,
                token_counts,
                selections,
            ):
                keys = tuple(
                    (case.task_id, case.entry.story_id, condition)
                    for condition in CONDITIONS
                )
                oracle_mean = node_totals[case.oracle_index] / token_count
                for condition in CONDITIONS:
                    key = (case.task_id, case.entry.story_id, condition)
                    if key in completed:
                        continue
                    node_index = (
                        0
                        if condition == "base"
                        else case.oracle_index
                        if condition == "oracle"
                        else selected[condition]
                    )
                    mean_nll = node_totals[node_index] / token_count
                    row = WholeStoryNllRow(
                        task_noun=case.task_id,
                        story_id=case.entry.story_id,
                        condition=condition,
                        selected_node=str(adaptation.vamp_graph.nodes[node_index].node_id),
                        selected_path=_node_path(adaptation, node_index),
                        oracle_node=str(
                            adaptation.vamp_graph.nodes[case.oracle_index].node_id
                        ),
                        oracle_match=node_index == case.oracle_index,
                        total_nll=node_totals[node_index],
                        token_count=token_count,
                        mean_nll=mean_nll,
                        perplexity=math.exp(min(mean_nll, 700.0)),
                        regret_vs_oracle=mean_nll - oracle_mean,
                    )
                    ledger.write(
                        canonical_json_bytes(
                            _versioned_result_record(
                                partition,
                                "whole_story_format",
                                WHOLE_STORY_FORMAT,
                                row.as_record(),
                            )
                        )
                    )
                    completed.add(key)
                ledger.flush()
                os.fsync(ledger.fileno())
                finished_stories += 1
                if progress is not None:
                    progress("whole-story-nll", finished_stories, total)
    expected_rows = len(expected_keys)
    if completed != expected_keys:
        raise RuntimeError(
            f"whole-story ledger has {len(completed)} of {expected_rows} rows"
        )
    os.replace(work, output)
    return output


def evaluate_half_story_generations(
    partition: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
    adaptation: LanguageAdaptationArtifact,
    tokenizer: TextTokenizer,
    output_path: str | Path,
    *,
    progress: EvaluationProgress | None = None,
) -> Path:
    """Route midpoint prefixes, score true suffixes, and greedily complete them."""
    output = Path(output_path)
    entries_by_task = {
        task_id: load_story_index(partition, f"task-{task_id}-generation")
        for task_id in partition.task_ids
    }
    expected_keys = {
        (task_id, entry.story_id)
        for task_id, entries in entries_by_task.items()
        for entry in entries
    }
    if output.is_file():
        if _completed_keys(output, ("task_noun", "story_id")) != expected_keys:
            raise ValueError("published generation ledger coverage changed")
        return output
    work = output.with_name(f".{output.name}.work")
    work.parent.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns streamed generation ledger: {work.resolve()}", flush=True)
    completed = _completed_keys(work, ("task_noun", "story_id"))
    if not completed <= expected_keys:
        raise ValueError("generation work ledger contains unexpected rows")
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    packed = pack_lora_memory(
        adaptation.vamp_graph,
        adaptation.model_config,
        adaptation.lora_config,
        adaptation.max_nodes,
        adaptation.max_edges,
    )
    store = IndexedStoryStore(partition)
    total = sum(len(entries) for entries in entries_by_task.values())
    finished = len(completed)
    chunks = _half_story_chunks(
        partition,
        entries_by_task,
        completed,
        store,
        preset,
        adaptation.model_config.max_position_embeddings,
    )
    with work.open("ab") as ledger:
        for cases in chunks:
            suffix_windows = _stack_token_batches(
                tuple(case.suffix_windows for case in cases)
            )
            per_window_totals = _nll_by_node_per_window(
                loaded.params,
                adaptation,
                packed,
                suffix_windows,
                preset.evaluation_chunk_size,
            )
            boundaries = np.cumsum(
                (0,) + tuple(
                    case.suffix_windows.input_ids.shape[0] for case in cases
                )
            )
            node_totals_by_case = tuple(
                tuple(
                    float(value)
                    for value in np.sum(
                        per_window_totals[:, start:stop],
                        axis=1,
                        dtype=np.float64,
                    )
                )
                for start, stop in zip(boundaries[:-1], boundaries[1:])
            )
            suffix_token_counts = tuple(
                int(np.sum(case.suffix_windows.loss_mask)) for case in cases
            )
            selections = _prefix_chunk_selections(
                cases,
                loaded.params,
                adaptation,
                packed,
                preset.evaluation_chunk_size,
            )
            results_by_case = _completion_results_for_cases(
                cases,
                selections,
                node_totals_by_case,
                suffix_token_counts,
                loaded.params,
                adaptation,
                packed,
                tokenizer,
                partition,
                preset.evaluation_chunk_size,
            )
            for case, results in zip(cases, results_by_case):
                row = HalfStoryGenerationRow(
                    task_noun=case.task_id,
                    story_id=case.entry.story_id,
                    prefix=tokenizer.decode(case.query.prompt_token_ids),
                    reference_continuation=tokenizer.decode(
                        case.tokens[case.midpoint :]
                    ),
                    full_original_story=store.text(case.entry),
                    results=results,
                )
                ledger.write(
                    canonical_json_bytes(
                        _versioned_result_record(
                            partition,
                            "half_story_format",
                            HALF_STORY_FORMAT,
                            row.as_record(),
                        )
                    )
                )
                ledger.flush()
                os.fsync(ledger.fileno())
                completed.add((case.task_id, case.entry.story_id))
                finished += 1
                if progress is not None:
                    progress("half-story-generation", finished, total)
    if completed != expected_keys:
        raise RuntimeError(f"generation ledger has {len(completed)} of {total} rows")
    os.replace(work, output)
    return output


def _half_story_chunks(
    partition: NounPartitionArtifact,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]],
    completed: set[tuple[str, ...]],
    store: IndexedStoryStore,
    preset: NounsExperimentPreset,
    maximum_position_embeddings: int,
):
    """Yield pending generation cases with bounded suffix-window storage."""
    chunk: list[_HalfStoryCase] = []
    window_count = 0
    window_limit = (
        preset.evaluation_chunk_size * _GENERATION_WINDOW_CHUNK_MULTIPLIER
    )
    for task_id, entries in entries_by_task.items():
        oracle_index = partition.task_ids.index(task_id) + 1
        for entry in entries:
            if (task_id, entry.story_id) in completed:
                continue
            tokens = store.tokens(entry)
            midpoint = len(tokens) // 2
            query = build_prefix_only_query(
                entry.story_id,
                tokens,
                partition.pad_token_id,
                maximum_position_embeddings,
            )
            suffix_windows = _story_windows(
                tokens,
                preset.context_length,
                partition.pad_token_id,
                first_target_index=midpoint,
            )
            budget = min(
                len(tokens) - midpoint,
                maximum_position_embeddings - midpoint,
            )
            case = _HalfStoryCase(
                task_id,
                entry,
                oracle_index,
                tokens,
                midpoint,
                query,
                suffix_windows,
                budget,
            )
            case_windows = suffix_windows.input_ids.shape[0]
            if chunk and window_count + case_windows > window_limit:
                yield tuple(chunk)
                chunk = []
                window_count = 0
            chunk.append(case)
            window_count += case_windows
            if window_count >= window_limit:
                yield tuple(chunk)
                chunk = []
                window_count = 0
    if chunk:
        yield tuple(chunk)


def _prefix_chunk_selections(
    cases: tuple[_HalfStoryCase, ...],
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    evaluation_chunk_size: int,
) -> tuple[dict[Condition, int], ...]:
    """Route one bounded set of midpoint-only prefixes in shared GPU calls."""
    batch = _stack_prefix_queries(cases, adaptation.model_config.max_position_embeddings)
    original_count = len(cases)
    router_chunk_size = min(evaluation_chunk_size, _MAX_EBT_ROWS)
    padded_count = (
        (original_count + router_chunk_size - 1) // router_chunk_size
    ) * router_chunk_size
    padded = _pad_router_batch(batch, padded_count)
    selected_by_condition: dict[Condition, np.ndarray] = {}
    for condition in CONDITIONS[2:]:
        decision = route_language_prefix(
            condition,
            base_params,
            adaptation.model_config,
            packed,
            adaptation.lora_config,
            adaptation.address_book,
            padded,
            hopfield_config=HopfieldConfig(),
            ebt_config=EbtConfig(),
            evaluation_microbatch_size=router_chunk_size,
        )
        selected_by_condition[condition] = np.asarray(
            decision.selected_indices,
            dtype=np.int32,
        )[:original_count]
    return tuple(
        {
            "base": 0,
            "oracle": case.oracle_index,
            **{
                condition: int(selected_by_condition[condition][index])
                for condition in CONDITIONS[2:]
            },
        }
        for index, case in enumerate(cases)
    )


def _stack_prefix_queries(
    cases: tuple[_HalfStoryCase, ...],
    maximum_position_embeddings: int,
) -> RouterBatch:
    if not cases:
        raise ValueError("prefix stacking requires at least one generation case")
    maximum_width = max(case.query.router_batch.input_ids.shape[1] for case in cases)
    bucket_width = min(
        maximum_position_embeddings,
        ((maximum_width + 31) // 32) * 32,
    )
    shape = (len(cases), bucket_width)
    inputs = np.zeros(shape, dtype=np.int32)
    targets = np.zeros(shape, dtype=np.int32)
    attention = np.zeros(shape, dtype=np.bool_)
    losses = np.zeros(shape, dtype=np.bool_)
    for row, case in enumerate(cases):
        source = case.query.router_batch
        width = source.input_ids.shape[1]
        inputs[row, :width] = source.input_ids[0]
        targets[row, :width] = source.target_ids[0]
        attention[row, :width] = source.attention_mask[0]
        losses[row, :width] = source.loss_mask[0]
    return RouterBatch(inputs, attention, targets, losses)


def _completion_results_for_cases(
    cases: tuple[_HalfStoryCase, ...],
    selections: tuple[dict[Condition, int], ...],
    node_totals: tuple[tuple[float, ...], ...],
    token_counts: tuple[int, ...],
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    tokenizer: TextTokenizer,
    partition: NounPartitionArtifact,
    evaluation_chunk_size: int,
) -> tuple[tuple[CompletionResult, ...], ...]:
    """Generate several stories while keeping every story's six rows together."""
    if not (
        len(cases) == len(selections) == len(node_totals) == len(token_counts)
    ):
        raise ValueError("generation case metadata must have identical lengths")
    if evaluation_chunk_size <= 0:
        raise ValueError("evaluation chunk size must be positive")
    packed_indices = _generation_case_bins(
        cases,
        selections,
        adaptation.model_config.max_position_embeddings,
    )
    results: list[tuple[CompletionResult, ...] | None] = [None] * len(cases)
    for indices in packed_indices:
        generated = _generate_completion_chunk(
            tuple(cases[index] for index in indices),
            tuple(selections[index] for index in indices),
            tuple(node_totals[index] for index in indices),
            tuple(token_counts[index] for index in indices),
            base_params,
            adaptation,
            packed,
            tokenizer,
            partition,
            row_capacity=_MAX_GENERATION_ROWS,
        )
        for index, value in zip(indices, generated):
            results[index] = value
    if any(value is None for value in results):
        raise RuntimeError("generation packing did not cover every case")
    return tuple(value for value in results if value is not None)


def _generation_case_bins(
    cases: tuple[_HalfStoryCase, ...],
    selections: tuple[dict[Condition, int], ...],
    maximum_position_embeddings: int,
    *,
    row_capacity: int = _MAX_GENERATION_ROWS,
) -> tuple[tuple[int, ...], ...]:
    """First-fit decreasing bins minimize padded generation without reordering output."""
    if len(cases) != len(selections) or not cases:
        raise ValueError("generation binning requires matching nonempty inputs")
    if row_capacity <= 0 or maximum_position_embeddings <= 0:
        raise ValueError("generation bin capacities must be positive")
    bins: tuple[_GenerationBin, ...] = ()
    ordered_indices = sorted(
        range(len(cases)),
        key=lambda index: (
            -cases[index].budget,
            -len(cases[index].query.prompt_token_ids),
            index,
        ),
    )
    for index in ordered_indices:
        row_count = len(set(selections[index].values()))
        prompt_width = len(cases[index].query.prompt_token_ids)
        budget = cases[index].budget
        for bin_index, candidate in enumerate(bins):
            candidate_rows = candidate.row_count + row_count
            candidate_prompt = max(candidate.prompt_width, prompt_width)
            candidate_budget = max(candidate.budget, budget)
            if (
                candidate_rows <= row_capacity
                and candidate_prompt + candidate_budget
                <= maximum_position_embeddings
            ):
                bins = (
                    bins[:bin_index]
                    + (
                        _GenerationBin(
                            indices=candidate.indices + (index,),
                            row_count=candidate_rows,
                            prompt_width=candidate_prompt,
                            budget=candidate_budget,
                        ),
                    )
                    + bins[bin_index + 1 :]
                )
                break
        else:
            if row_count > row_capacity:
                raise ValueError("one generation case exceeds the row capacity")
            if prompt_width + budget > maximum_position_embeddings:
                raise ValueError("one generation case exceeds the model context window")
            bins += (
                _GenerationBin(
                    indices=(index,),
                    row_count=row_count,
                    prompt_width=prompt_width,
                    budget=budget,
                ),
            )
    return tuple(tuple(candidate.indices) for candidate in bins)


def _generate_completion_chunk(
    cases: tuple[_HalfStoryCase, ...],
    selections: tuple[dict[Condition, int], ...],
    node_totals: tuple[tuple[float, ...], ...],
    token_counts: tuple[int, ...],
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    tokenizer: TextTokenizer,
    partition: NounPartitionArtifact,
    *,
    row_capacity: int | None = None,
) -> tuple[tuple[CompletionResult, ...], ...]:
    if not cases:
        raise ValueError("generation requires at least one case")
    generation_pairs: list[tuple[int, int]] = []
    generated_row_by_case_node: dict[tuple[int, int], int] = {}
    for case_index, selection in enumerate(selections):
        for condition in CONDITIONS:
            key = (case_index, selection[condition])
            if key not in generated_row_by_case_node:
                generated_row_by_case_node[key] = len(generation_pairs)
                generation_pairs.append(key)
    capacity = len(generation_pairs) if row_capacity is None else row_capacity
    if capacity < len(generation_pairs):
        raise ValueError("generation row capacity cannot hold distinct story/node pairs")
    prompt_width = max(len(case.query.prompt_token_ids) for case in cases)
    maximum_budget = max(case.budget for case in cases)
    prompt = np.full(
        (capacity, prompt_width),
        partition.pad_token_id,
        dtype=np.int32,
    )
    attention = np.zeros_like(prompt, dtype=np.bool_)
    padded_pairs = tuple(generation_pairs) + (generation_pairs[0],) * (
        capacity - len(generation_pairs)
    )
    for row, (case_index, _) in enumerate(padded_pairs):
        case = cases[case_index]
        width = len(case.query.prompt_token_ids)
        prompt[row, :width] = case.query.prompt_token_ids
        attention[row, :width] = True
    node_indices = np.asarray(
        tuple(node_index for _, node_index in padded_pairs),
        dtype=np.int32,
    )
    generated = np.asarray(
        greedy_generate(
            base_params,
            adaptation.model_config,
            prompt,
            attention,
            maximum_budget,
            eos_token_id=partition.eos_token_id,
            pad_token_id=partition.pad_token_id,
            lora_memory=packed,
            lora_config=adaptation.lora_config,
            node_index=node_indices,
        )
    )
    results = []
    for case_index, case in enumerate(cases):
        prompt_length = len(case.query.prompt_token_ids)
        generated_rows = tuple(
            generated[
                generated_row_by_case_node[
                    (case_index, selections[case_index][condition])
                ],
                prompt_length : prompt_length + case.budget,
            ]
            for condition in CONDITIONS
        )
        case_results = []
        for condition, node_index, generated_row in zip(
            CONDITIONS,
            tuple(selections[case_index][condition] for condition in CONDITIONS),
            generated_rows,
        ):
            eos_positions = np.flatnonzero(generated_row == partition.eos_token_id)
            retained_count = (
                int(eos_positions[0] + 1)
                if len(eos_positions)
                else len(generated_row)
            )
            retained = tuple(
                int(value) for value in generated_row[:retained_count]
            )
            total_nll = node_totals[case_index][node_index]
            case_results.append(
                CompletionResult(
                    condition=condition,
                    selected_node=str(
                        adaptation.vamp_graph.nodes[node_index].node_id
                    ),
                    selected_path=_node_path(adaptation, node_index),
                    total_nll=total_nll,
                    token_count=token_counts[case_index],
                    mean_nll=total_nll / token_counts[case_index],
                    generated_continuation=tokenizer.decode(retained),
                    generated_token_count=retained_count,
                    eos_reached=bool(len(eos_positions)),
                )
            )
        results.append(tuple(case_results))
    return tuple(results)


def _completion_results(
    selections: dict[Condition, int],
    query: PrefixOnlyQuery,
    node_totals: tuple[float, ...],
    token_count: int,
    budget: int,
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    tokenizer: TextTokenizer,
    partition: NounPartitionArtifact,
) -> tuple[CompletionResult, ...]:
    node_indices = tuple(selections[condition] for condition in CONDITIONS)
    prompt = np.repeat(
        np.asarray((query.prompt_token_ids,), dtype=np.int32),
        len(CONDITIONS),
        axis=0,
    )
    generated_rows = np.asarray(
        greedy_generate(
            base_params,
            adaptation.model_config,
            prompt,
            np.ones_like(prompt, dtype=np.bool_),
            budget,
            eos_token_id=partition.eos_token_id,
            pad_token_id=partition.pad_token_id,
            lora_memory=packed,
            lora_config=adaptation.lora_config,
            node_index=np.asarray(node_indices, dtype=np.int32),
        )
    )[:, len(query.prompt_token_ids) :]

    def result_for_row(
        condition: Condition,
        node_index: int,
        generated: np.ndarray,
    ) -> CompletionResult:
        eos_positions = np.flatnonzero(generated == partition.eos_token_id)
        retained_count = (
            int(eos_positions[0] + 1) if len(eos_positions) else len(generated)
        )
        retained = tuple(int(value) for value in generated[:retained_count])
        return CompletionResult(
            condition=condition,
            selected_node=str(adaptation.vamp_graph.nodes[node_index].node_id),
            selected_path=_node_path(adaptation, node_index),
            total_nll=node_totals[node_index],
            token_count=token_count,
            mean_nll=node_totals[node_index] / token_count,
            generated_continuation=tokenizer.decode(retained),
            generated_token_count=retained_count,
            eos_reached=bool(len(eos_positions)),
        )

    return tuple(
        result_for_row(condition, node_index, generated)
        for condition, node_index, generated in zip(
            CONDITIONS,
            node_indices,
            generated_rows,
        )
    )


def _whole_story_chunks(
    partition: NounPartitionArtifact,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]],
    completed: set[tuple[str, ...]],
    store: IndexedStoryStore,
    preset: NounsExperimentPreset,
):
    """Yield consecutive pending cases with a bounded total window count."""
    chunk: list[_WholeStoryCase] = []
    window_count = 0
    for task_id, entries in entries_by_task.items():
        oracle_index = partition.task_ids.index(task_id) + 1
        for entry in entries:
            if all(
                (task_id, entry.story_id, condition) in completed
                for condition in CONDITIONS
            ):
                continue
            windows = _story_windows(
                store.tokens(entry),
                preset.context_length,
                partition.pad_token_id,
            )
            case = _WholeStoryCase(task_id, entry, oracle_index, windows)
            case_windows = windows.input_ids.shape[0]
            if chunk and window_count + case_windows > preset.evaluation_chunk_size:
                yield tuple(chunk)
                chunk = []
                window_count = 0
            chunk.append(case)
            window_count += case_windows
            if window_count >= preset.evaluation_chunk_size:
                yield tuple(chunk)
                chunk = []
                window_count = 0
    if chunk:
        yield tuple(chunk)


def _whole_story_chunk_selections(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    windows: TokenBatch,
    boundaries: np.ndarray,
    node_totals_by_case: tuple[tuple[float, ...], ...],
    microbatch_size: int,
) -> tuple[dict[Condition, int], ...]:
    """Select all four task-free routers for one bounded story chunk."""
    valid_count = len(adaptation.vamp_graph.nodes)
    router_batch = RouterBatch(
        windows.input_ids,
        windows.attention_mask,
        windows.target_ids,
        windows.loss_mask,
    )
    weights = np.sum(windows.loss_mask, axis=1, dtype=np.float64)
    exhaustive = tuple(
        int(np.argmin(np.asarray(node_totals[:valid_count])))
        for node_totals in node_totals_by_case
    )
    embeddings = _frozen_embeddings_in_chunks(
        base_params,
        adaptation,
        router_batch,
        microbatch_size,
    )
    centroids = np.stack(
        tuple(
            np.average(
                embeddings[start:stop],
                axis=0,
                weights=weights[start:stop],
            )
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        )
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("whole-story content centroid is not finite and nonzero")
    centroids /= norms
    hopfield = tuple(
        int(value)
        for value in np.asarray(
            hopfield_address(
                centroids,
                adaptation.address_book,
                HopfieldConfig(),
            ).selected_indices
        )
    )
    ebt_by_condition: dict[str, tuple[int, ...]] = {}
    ebt_chunk_size = min(microbatch_size, _MAX_EBT_ROWS)
    for condition in ("vamp_ebt_uniform", "vamp_ebt_hopfield"):
        logits = _ebt_logits_in_chunks(
            condition,
            base_params,
            adaptation,
            packed,
            router_batch,
            ebt_chunk_size,
        )
        ebt_by_condition[condition] = tuple(
            int(
                np.argmax(
                    np.average(
                        logits[start:stop, :valid_count],
                        axis=0,
                        weights=weights[start:stop],
                    )
                )
            )
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        )
    return tuple(
        {
            "base": 0,
            "oracle": 0,
            "vamp_exhaustive": exhaustive[index],
            "vamp_hopfield": hopfield[index],
            "vamp_ebt_uniform": ebt_by_condition["vamp_ebt_uniform"][index],
            "vamp_ebt_hopfield": ebt_by_condition["vamp_ebt_hopfield"][index],
        }
        for index in range(len(node_totals_by_case))
    )


def _frozen_embeddings_in_chunks(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    batch: RouterBatch,
    chunk_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, batch.input_ids.shape[0], chunk_size):
        stop = min(start + chunk_size, batch.input_ids.shape[0])
        chunk = _pad_router_batch(_slice_router_batch(batch, start, stop), chunk_size)
        encoded = encode_frozen_base_content(
            base_params,
            adaptation.model_config,
            jnp.asarray(chunk.input_ids),
            jnp.asarray(chunk.attention_mask),
            evaluation_microbatch_size=chunk_size,
        )
        outputs.append(np.asarray(encoded)[: stop - start])
    return np.concatenate(outputs, axis=0)


def _ebt_logits_in_chunks(
    condition: str,
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    batch: RouterBatch,
    chunk_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, batch.input_ids.shape[0], chunk_size):
        stop = min(start + chunk_size, batch.input_ids.shape[0])
        chunk = _pad_router_batch(_slice_router_batch(batch, start, stop), chunk_size)
        result = trace_ebt_language_prefix(
            condition,
            base_params,
            adaptation.model_config,
            packed,
            adaptation.lora_config,
            adaptation.address_book,
            chunk,
            hopfield_config=HopfieldConfig(),
            ebt_config=EbtConfig(),
        )
        outputs.append(np.asarray(result.final_node_logits)[: stop - start])
    return np.concatenate(outputs, axis=0)


def _story_windows(
    token_ids: tuple[int, ...],
    context_length: int,
    pad_token_id: int,
    *,
    first_target_index: int = 1,
) -> TokenBatch:
    starts = tuple(range(0, len(token_ids) - 1, context_length))
    shape = (len(starts), context_length)
    inputs = np.full(shape, pad_token_id, dtype=np.int32)
    targets = np.full(shape, pad_token_id, dtype=np.int32)
    attention = np.zeros(shape, dtype=np.bool_)
    losses = np.zeros(shape, dtype=np.bool_)
    for row, start in enumerate(starts):
        chunk = token_ids[start : start + context_length + 1]
        transitions = len(chunk) - 1
        inputs[row, :transitions] = chunk[:-1]
        targets[row, :transitions] = chunk[1:]
        attention[row, :transitions] = True
        global_target_indices = np.arange(start + 1, start + transitions + 1)
        losses[row, :transitions] = global_target_indices >= first_target_index
    if not np.any(losses):
        raise ValueError("story window selector contains no target tokens")
    return TokenBatch(inputs, attention, targets, losses)


def _stack_token_batches(batches: tuple[TokenBatch, ...]) -> TokenBatch:
    if not batches:
        raise ValueError("token-batch stacking requires at least one batch")
    return TokenBatch(
        np.concatenate(tuple(batch.input_ids for batch in batches), axis=0),
        np.concatenate(tuple(batch.attention_mask for batch in batches), axis=0),
        np.concatenate(tuple(batch.target_ids for batch in batches), axis=0),
        np.concatenate(tuple(batch.loss_mask for batch in batches), axis=0),
    )


def _slice_router_batch(batch: RouterBatch, start: int, stop: int) -> RouterBatch:
    return RouterBatch(
        batch.input_ids[start:stop],
        batch.attention_mask[start:stop],
        batch.target_ids[start:stop],
        batch.loss_mask[start:stop],
    )


def _pad_router_batch(batch: RouterBatch, row_count: int) -> RouterBatch:
    current = batch.input_ids.shape[0]
    if current <= 0 or current > row_count:
        raise ValueError("router padding requires one to row_count source rows")
    if current == row_count:
        return batch
    repeats = row_count - current

    def padded(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values, np.repeat(values[:1], repeats, axis=0)), axis=0)

    return RouterBatch(
        padded(batch.input_ids),
        padded(batch.attention_mask),
        padded(batch.target_ids),
        padded(batch.loss_mask),
    )


def _pad_token_batch(batch: TokenBatch, row_count: int) -> TokenBatch:
    current = batch.input_ids.shape[0]
    if current <= 0 or current > row_count:
        raise ValueError("token padding requires one to row_count source rows")
    if current == row_count:
        return batch
    repeats = row_count - current

    def padded(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values, np.repeat(values[:1], repeats, axis=0)), axis=0)

    return TokenBatch(
        padded(batch.input_ids),
        padded(batch.attention_mask),
        padded(batch.target_ids),
        padded(batch.loss_mask),
    )


def _nll_by_node_per_window(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    windows: TokenBatch,
    microbatch_size: int,
    *,
    node_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return bounded, shape-stable node totals for every input window."""
    window_count = windows.input_ids.shape[0]
    node_count = len(adaptation.vamp_graph.nodes)
    selected = tuple(range(node_count)) if node_indices is None else node_indices
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(index) is not int or not 0 <= index < node_count for index in selected)
    ):
        raise ValueError("NLL node indices must be unique valid graph indices")
    totals = np.empty((len(selected), window_count), dtype=np.float64)
    for output_index, node_index in enumerate(selected):
        coefficients = edge_coefficients_for_node(packed, node_index)
        for start in range(0, window_count, microbatch_size):
            stop = min(start + microbatch_size, window_count)
            chunk = _pad_token_batch(
                TokenBatch(
                    windows.input_ids[start:stop],
                    windows.attention_mask[start:stop],
                    windows.target_ids[start:stop],
                    windows.loss_mask[start:stop],
                ),
                microbatch_size,
            )
            result = apply_gpt_neo(
                base_params,
                adaptation.model_config,
                jnp.asarray(chunk.input_ids),
                jnp.asarray(chunk.attention_mask),
                lora_memory=packed,
                edge_coefficients=coefficients,
                lora_config=adaptation.lora_config,
                training=False,
            )
            losses = per_token_nll(
                result.logits,
                jnp.asarray(chunk.target_ids),
            )
            row_totals = np.asarray(
                jnp.sum(
                    losses * jnp.asarray(chunk.loss_mask, dtype=jnp.float32),
                    axis=-1,
                ),
                dtype=np.float64,
            )
            totals[output_index, start:stop] = row_totals[: stop - start]
    return totals


def _nll_by_node(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    windows: TokenBatch,
    microbatch_size: int,
) -> tuple[tuple[float, ...], int]:
    token_count = int(np.sum(windows.loss_mask))
    per_window = _nll_by_node_per_window(
        base_params,
        adaptation,
        packed,
        windows,
        microbatch_size,
    )
    totals = np.sum(per_window, axis=1, dtype=np.float64)
    return tuple(float(value) for value in totals), token_count


def _node_path(
    adaptation: LanguageAdaptationArtifact,
    node_index: int,
) -> tuple[str, ...]:
    node = adaptation.vamp_graph.nodes[node_index]
    return tuple(
        str(path_node.node_id)
        for path_node in memory_node_path(adaptation.vamp_graph, node.node_id)
    )


def _completed_keys(path: Path, fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        complete = payload[: payload.rfind(b"\n") + 1]
        _atomic_write(path, complete)
        payload = complete
    keys: set[tuple[str, ...]] = set()
    for line in payload.splitlines():
        record = json.loads(line)
        if line + b"\n" != canonical_json_bytes(record):
            raise ValueError("resumable evaluation ledger is not canonical JSONL")
        key = tuple(str(record[field]) for field in fields)
        if key in keys:
            raise ValueError("resumable evaluation ledger contains duplicate rows")
        keys.add(key)
    return keys


def _versioned_result_record(
    partition: object,
    format_attribute: str,
    default_format: str,
    record: dict[str, object],
) -> dict[str, object]:
    format_name = getattr(partition, format_attribute, default_format)
    if type(format_name) is not str or not format_name:
        raise ValueError("noun result format must be nonempty")
    core = {**record, "format": format_name}
    return (
        {**core, "result_sha256": record_sha256(core)}
        if format_name != default_format
        else core
    )


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
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


__all__ = [
    "CompletionResult",
    "HALF_STORY_FORMAT",
    "HalfStoryGenerationRow",
    "PrefixOnlyQuery",
    "WHOLE_STORY_FORMAT",
    "build_prefix_only_query",
    "evaluate_half_story_generations",
    "evaluate_whole_story_nll",
]
