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
    with work.open("ab") as ledger:
        for task_id, entries in entries_by_task.items():
            oracle_index = partition.task_ids.index(task_id) + 1
            for entry in entries:
                keys = tuple((task_id, entry.story_id, condition) for condition in CONDITIONS)
                if all(key in completed for key in keys):
                    continue
                tokens = store.tokens(entry)
                windows = _story_windows(
                    tokens,
                    preset.context_length,
                    partition.pad_token_id,
                )
                node_totals, token_count = _nll_by_node(
                    loaded.params,
                    adaptation,
                    packed,
                    windows,
                    preset.evaluation_chunk_size,
                )
                selected = _whole_story_selections(
                    loaded.params,
                    adaptation,
                    packed,
                    windows,
                    node_totals,
                    token_count,
                    preset.evaluation_chunk_size,
                )
                oracle_mean = node_totals[oracle_index] / token_count
                for condition in CONDITIONS:
                    key = (task_id, entry.story_id, condition)
                    if key in completed:
                        continue
                    node_index = (
                        0
                        if condition == "base"
                        else oracle_index
                        if condition == "oracle"
                        else selected[condition]
                    )
                    mean_nll = node_totals[node_index] / token_count
                    row = WholeStoryNllRow(
                        task_noun=task_id,
                        story_id=entry.story_id,
                        condition=condition,
                        selected_node=str(adaptation.vamp_graph.nodes[node_index].node_id),
                        selected_path=_node_path(adaptation, node_index),
                        oracle_node=str(adaptation.vamp_graph.nodes[oracle_index].node_id),
                        oracle_match=node_index == oracle_index,
                        total_nll=node_totals[node_index],
                        token_count=token_count,
                        mean_nll=mean_nll,
                        perplexity=math.exp(min(mean_nll, 700.0)),
                        regret_vs_oracle=mean_nll - oracle_mean,
                    )
                    ledger.write(
                        canonical_json_bytes(
                            {"format": WHOLE_STORY_FORMAT, **row.as_record()}
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
    with work.open("ab") as ledger:
        for task_id, entries in entries_by_task.items():
            oracle_index = partition.task_ids.index(task_id) + 1
            for entry in entries:
                key = (task_id, entry.story_id)
                if key in completed:
                    continue
                tokens = store.tokens(entry)
                midpoint = len(tokens) // 2
                query = build_prefix_only_query(
                    entry.story_id,
                    tokens,
                    partition.pad_token_id,
                    adaptation.model_config.max_position_embeddings,
                )
                selections = _prefix_selections(
                    query,
                    oracle_index,
                    loaded.params,
                    adaptation,
                    packed,
                    preset.evaluation_chunk_size,
                )
                suffix_windows = _story_windows(
                    tokens,
                    preset.context_length,
                    partition.pad_token_id,
                    first_target_index=midpoint,
                )
                node_totals, suffix_tokens = _nll_by_node(
                    loaded.params,
                    adaptation,
                    packed,
                    suffix_windows,
                    preset.evaluation_chunk_size,
                )
                budget = min(
                    len(tokens) - midpoint,
                    adaptation.model_config.max_position_embeddings - midpoint,
                )
                results = _completion_results(
                    selections,
                    query,
                    node_totals,
                    suffix_tokens,
                    budget,
                    loaded.params,
                    adaptation,
                    packed,
                    tokenizer,
                    partition,
                )
                row = HalfStoryGenerationRow(
                    task_noun=task_id,
                    story_id=entry.story_id,
                    prefix=tokenizer.decode(query.prompt_token_ids),
                    reference_continuation=tokenizer.decode(tokens[midpoint:]),
                    full_original_story=store.text(entry),
                    results=results,
                )
                ledger.write(canonical_json_bytes(row.as_record()))
                ledger.flush()
                os.fsync(ledger.fileno())
                completed.add(key)
                finished += 1
                if progress is not None:
                    progress("half-story-generation", finished, total)
    if completed != expected_keys:
        raise RuntimeError(f"generation ledger has {len(completed)} of {total} rows")
    os.replace(work, output)
    return output


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


def _whole_story_selections(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    windows: TokenBatch,
    node_totals: tuple[float, ...],
    token_count: int,
    microbatch_size: int,
) -> dict[Condition, int]:
    valid_count = len(adaptation.vamp_graph.nodes)
    router_batch = RouterBatch(
        windows.input_ids,
        windows.attention_mask,
        windows.target_ids,
        windows.loss_mask,
    )
    weights = np.sum(windows.loss_mask, axis=1, dtype=np.float64)
    exhaustive = int(np.argmin(np.asarray(node_totals[:valid_count]) / token_count))
    embeddings = np.asarray(
        encode_frozen_base_content(
            base_params,
            adaptation.model_config,
            jnp.asarray(windows.input_ids),
            jnp.asarray(windows.attention_mask),
            evaluation_microbatch_size=microbatch_size,
        )
    )
    centroid = np.average(embeddings, axis=0, weights=weights)
    centroid /= np.linalg.norm(centroid)
    hopfield = int(
        np.asarray(
            hopfield_address(
                centroid[None, :], adaptation.address_book, HopfieldConfig()
            ).selected_indices
        )[0]
    )
    ebt = {}
    for condition in ("vamp_ebt_uniform", "vamp_ebt_hopfield"):
        result = trace_ebt_language_prefix(
            condition,
            base_params,
            adaptation.model_config,
            packed,
            adaptation.lora_config,
            adaptation.address_book,
            router_batch,
            hopfield_config=HopfieldConfig(),
            ebt_config=EbtConfig(),
        )
        logits = np.asarray(result.final_node_logits)[:, :valid_count]
        ebt[condition] = int(np.argmax(np.average(logits, axis=0, weights=weights)))
    return {
        "base": 0,
        "oracle": 0,
        "vamp_exhaustive": exhaustive,
        "vamp_hopfield": hopfield,
        "vamp_ebt_uniform": ebt["vamp_ebt_uniform"],
        "vamp_ebt_hopfield": ebt["vamp_ebt_hopfield"],
    }


def _prefix_selections(
    query: PrefixOnlyQuery,
    oracle_index: int,
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    microbatch_size: int,
) -> dict[Condition, int]:
    selected: dict[Condition, int] = {"base": 0, "oracle": oracle_index}
    for condition in CONDITIONS[2:]:
        decision = route_language_prefix(
            condition,
            base_params,
            adaptation.model_config,
            packed,
            adaptation.lora_config,
            adaptation.address_book,
            query.router_batch,
            hopfield_config=HopfieldConfig(),
            ebt_config=EbtConfig(),
            evaluation_microbatch_size=microbatch_size,
        )
        selected[condition] = int(np.asarray(decision.selected_indices)[0])
    return selected


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


def _nll_by_node(
    base_params,
    adaptation: LanguageAdaptationArtifact,
    packed,
    windows: TokenBatch,
    microbatch_size: int,
) -> tuple[tuple[float, ...], int]:
    token_count = int(np.sum(windows.loss_mask))
    totals = []
    for node_index in range(len(adaptation.vamp_graph.nodes)):
        node_total = 0.0
        coefficients = edge_coefficients_for_node(packed, node_index)
        for start in range(0, windows.input_ids.shape[0], microbatch_size):
            stop = min(start + microbatch_size, windows.input_ids.shape[0])
            result = apply_gpt_neo(
                base_params,
                adaptation.model_config,
                jnp.asarray(windows.input_ids[start:stop]),
                jnp.asarray(windows.attention_mask[start:stop]),
                lora_memory=packed,
                edge_coefficients=coefficients,
                lora_config=adaptation.lora_config,
                training=False,
            )
            losses = per_token_nll(
                result.logits,
                jnp.asarray(windows.target_ids[start:stop]),
            )
            node_total += float(
                jnp.sum(
                    losses
                    * jnp.asarray(
                        windows.loss_mask[start:stop], dtype=jnp.float32
                    )
                )
            )
        totals.append(node_total)
    return tuple(totals), token_count


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
