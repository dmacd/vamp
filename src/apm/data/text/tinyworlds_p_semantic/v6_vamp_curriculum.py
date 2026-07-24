"""Test-isolated five-world curriculum for the semantic-v6 VAMP study."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from heapq import nsmallest
import json
from pathlib import Path

import numpy as np

from apm.continual.language_evaluation import (
    LanguageEvaluationCondition,
    LanguageEvaluationSuite,
    LanguageExampleProvenance,
    LanguageSuiteExample,
)
from apm.continual.language_tasks import (
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evidence import exact_whole_word_spans
from apm.data.text.tinyworlds_p_semantic.v6_batching import (
    count_v6_partition_microbatches,
    iter_v6_partition_batches,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6SemanticPartitionArtifact,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.lm.text import TextTokenizer
from apm.lm.text_data import TokenBatch


V6VampPreparationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class V6VampSpanAnchor:
    """One deterministic full-length story span with exact partition provenance."""

    task_id: str
    split: str
    normalized_story_sha256: str
    record_id: str
    token_offset: int
    token_ids: tuple[int, ...]
    sequence_sha256: str

    def __post_init__(self) -> None:
        if self.task_id not in ("root", *V6_VAMP_EXPERIMENT_PRESET.task_order):
            raise ValueError("semantic-v6 anchor has an unknown task")
        if self.split not in ("validation", "test"):
            raise ValueError("semantic-v6 anchor split must be validation or test")
        _require_sha256(self.normalized_story_sha256, "semantic-v6 anchor group")
        _require_sha256(self.sequence_sha256, "semantic-v6 anchor sequence")
        if not self.record_id:
            raise ValueError("semantic-v6 anchor record ID must not be empty")
        if type(self.token_offset) is not int or self.token_offset < 0:
            raise ValueError("semantic-v6 anchor token offset must be nonnegative")
        if len(self.token_ids) != V6_VAMP_EXPERIMENT_PRESET.context_length:
            raise ValueError("semantic-v6 anchors must contain exactly 256 tokens")
        if any(type(value) is not int or value < 0 for value in self.token_ids):
            raise ValueError("semantic-v6 anchor token IDs must be nonnegative integers")
        token_bytes = np.asarray(self.token_ids, dtype="<u2").tobytes()
        if sha256(token_bytes).hexdigest() != self.sequence_sha256:
            raise ValueError("semantic-v6 anchor sequence hash changed")

    def as_record(self) -> dict[str, object]:
        """Return provenance without duplicating the authenticated token array."""
        return {
            "normalized_story_sha256": self.normalized_story_sha256,
            "record_id": self.record_id,
            "sequence_sha256": self.sequence_sha256,
            "split": self.split,
            "task_id": self.task_id,
            "token_offset": self.token_offset,
        }


@dataclass(frozen=True, slots=True)
class V6PreparedVampCurriculum:
    """Training batches and validation-only probes for all five worlds."""

    curriculum: LanguageCurriculum
    root_validation_probes: tuple[RouterBatch, ...]
    validation_anchors: tuple[V6VampSpanAnchor, ...]
    curriculum_sha256: str
    training_batch_counts: tuple[tuple[str, int], ...]
    training_active_tokens: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        expected_tasks = tuple(TaskId(world) for world in V6_VAMP_EXPERIMENT_PRESET.task_order)
        if tuple(task.task_id for task in self.curriculum.tasks) != expected_tasks:
            raise ValueError("semantic-v6 VAMP task order changed")
        if (
            self.curriculum.max_nodes != V6_VAMP_EXPERIMENT_PRESET.max_nodes
            or self.curriculum.max_edges != V6_VAMP_EXPERIMENT_PRESET.max_edges
        ):
            raise ValueError("semantic-v6 VAMP graph capacity changed")
        if any(task.test_examples for task in self.curriculum.tasks):
            raise ValueError("training curriculum must not contain sealed-test examples")
        if len(self.root_validation_probes) != V6_VAMP_EXPERIMENT_PRESET.root_probe_count:
            raise ValueError("semantic-v6 root probe count changed")
        expected_anchor_count = (
            len(expected_tasks) * V6_VAMP_EXPERIMENT_PRESET.parent_probe_count
        )
        if len(self.validation_anchors) != expected_anchor_count:
            raise ValueError("semantic-v6 validation anchor count changed")
        _require_sha256(self.curriculum_sha256, "semantic-v6 VAMP curriculum")


def prepare_v6_vamp_training_curriculum(
    artifact: V6SemanticPartitionArtifact,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
    *,
    progress: V6VampPreparationProgress | None = None,
) -> V6PreparedVampCurriculum:
    """Build all training batches and validation probes without reading a test index."""
    _require_canonical_sources(artifact, preset)
    root_anchors = select_v6_vamp_anchors(
        artifact,
        "root",
        "base/validation",
        preset.root_probe_count,
        preset,
    )
    anchors_by_world = {
        world: select_v6_vamp_anchors(
            artifact,
            world,
            f"world/{world}/validation",
            preset.parent_probe_count,
            preset,
        )
        for world in preset.task_order
    }
    tasks: list[LanguageTask] = []
    batch_counts: list[tuple[str, int]] = []
    active_tokens: list[tuple[str, int]] = []
    for world in preset.task_order:
        selector = f"world/{world}/train"
        total = count_v6_partition_microbatches(artifact, selector)
        batches: list[TokenBatch] = []
        for completed, batch in enumerate(
            iter_v6_partition_batches(artifact, selector, epoch=0),
            start=1,
        ):
            batches.append(batch)
            if progress is not None:
                progress(f"prepare world {world}", completed, total)
        world_anchors = anchors_by_world[world]
        examples = tuple(
            _language_example(
                anchor,
                preset.primary_prefix_length,
                preset,
                artifact.pad_token_id,
            )
            for anchor in world_anchors
        )
        tasks.append(
            LanguageTask(
                task_id=TaskId(world),
                train_batches=tuple(batches),
                validation_examples=examples,
                test_examples=(),
                parent_probes=tuple(example.router_batch for example in examples),
                content_key_probes=tuple(
                    example.router_batch for example in examples
                ),
            )
        )
        batch_counts.append((world, len(batches)))
        active_tokens.append(
            (
                world,
                sum(int(np.sum(batch.loss_mask)) for batch in batches),
            )
        )
    curriculum = LanguageCurriculum(
        tasks=tuple(tasks),
        max_nodes=preset.max_nodes,
        max_edges=preset.max_edges,
    )
    validation_anchors = tuple(
        anchor
        for world in preset.task_order
        for anchor in anchors_by_world[world]
    )
    curriculum_identity = record_sha256(
        {
            "anchors": [anchor.as_record() for anchor in validation_anchors],
            "config_sha256": preset.config_sha256,
            "partition_sha256": artifact.partition_sha256,
            "root_anchors": [anchor.as_record() for anchor in root_anchors],
            "training_active_tokens": active_tokens,
            "training_batch_counts": batch_counts,
        }
    )
    return V6PreparedVampCurriculum(
        curriculum=curriculum,
        root_validation_probes=tuple(
            _language_example(
                anchor,
                preset.primary_prefix_length,
                preset,
                artifact.pad_token_id,
            ).router_batch
            for anchor in root_anchors
        ),
        validation_anchors=validation_anchors,
        curriculum_sha256=curriculum_identity,
        training_batch_counts=tuple(batch_counts),
        training_active_tokens=tuple(active_tokens),
    )


def prepare_v6_vamp_test_suite(
    artifact: V6SemanticPartitionArtifact,
    tokenizer: TextTokenizer,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
) -> LanguageEvaluationSuite:
    """Construct the fixed nested-prefix suite after sealed-test authorization."""
    _require_canonical_sources(artifact, preset)
    anchors = tuple(
        anchor
        for world in preset.task_order
        for anchor in select_v6_vamp_anchors(
            artifact,
            world,
            f"world/{world}/test",
            preset.evaluation_examples_per_world,
            preset,
        )
    )
    conditions = tuple(
        LanguageEvaluationCondition(
            condition_id=f"prefix-{prefix_length}",
            prefix_tokens=prefix_length,
            suffix_tokens=preset.suffix_length,
        )
        for prefix_length in preset.prefix_lengths
    )
    examples = tuple(
        _suite_example(
            anchor,
            condition,
            artifact,
            tokenizer,
            preset,
        )
        for anchor in anchors
        for condition in conditions
    )
    suite_sha256 = record_sha256(
        {
            "anchors": [anchor.as_record() for anchor in anchors],
            "conditions": [condition.condition_id for condition in conditions],
            "config_sha256": preset.config_sha256,
            "partition_sha256": artifact.partition_sha256,
        }
    )
    return LanguageEvaluationSuite(
        suite_id=f"{preset.experiment_id}:{suite_sha256}",
        benchmark_label="semantic conjunction continual adaptation",
        primary_condition_id=f"prefix-{preset.primary_prefix_length}",
        conditions=conditions,
        examples=examples,
    )


def select_v6_vamp_anchors(
    artifact: V6SemanticPartitionArtifact,
    task_id: str,
    selector: str,
    count: int,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
) -> tuple[V6VampSpanAnchor, ...]:
    """Choose at most one lowest-hash 256-token span from each story group."""
    split = selector.rsplit("/", 1)[-1]
    if split not in ("validation", "test"):
        raise ValueError("semantic-v6 VAMP anchors may use only validation or test")
    records = tuple(_iter_index(artifact.root / "indexes" / _index_filename(selector)))
    candidates = tuple(
        _anchor_candidate(record, task_id, split, preset)
        for record in records
        if _integer(record, "token_count") >= preset.context_length
    )
    if len({candidate[1] for candidate in candidates}) != len(candidates):
        raise ValueError("semantic-v6 anchor source repeats a duplicate group")
    selected = tuple(nsmallest(count, candidates, key=lambda item: (item[0], item[1])))
    if len(selected) != count:
        raise ValueError(
            f"{selector} supplies {len(selected)} full spans; requires {count}"
        )
    return tuple(
        _materialize_anchor(artifact, task_id, split, record, start)
        for _, _, start, record in selected
    )


def _anchor_candidate(
    record: Mapping[str, object],
    task_id: str,
    split: str,
    preset: V6VampExperimentPreset,
) -> tuple[str, str, int, Mapping[str, object]]:
    group = _text(record, "normalized_story_sha256")
    last_start = _integer(record, "token_count") - preset.context_length
    starts = tuple(dict.fromkeys((*range(0, last_start + 1, 32), last_start)))
    ranked = tuple(
        (
            sha256(
                (
                    f"{preset.experiment_id}\0anchor\0{task_id}\0{split}\0"
                    f"{group}\0{start}"
                ).encode("utf-8")
            ).hexdigest(),
            start,
        )
        for start in starts
    )
    span_hash, start = min(ranked)
    return span_hash, group, start, record


def _materialize_anchor(
    artifact: V6SemanticPartitionArtifact,
    task_id: str,
    split: str,
    record: Mapping[str, object],
    start: int,
) -> V6VampSpanAnchor:
    shard_id = _integer(record, "token_shard")
    story_offset = _integer(record, "token_offset")
    shard = np.memmap(
        artifact.root / "shards" / f"tokens-{shard_id:06d}.uint16",
        dtype="<u2",
        mode="r",
    )
    tokens = np.asarray(
        shard[
            story_offset
            + start : story_offset
            + start
            + V6_VAMP_EXPERIMENT_PRESET.context_length
        ],
        dtype=np.uint16,
    )
    if len(tokens) != V6_VAMP_EXPERIMENT_PRESET.context_length:
        raise ValueError("semantic-v6 anchor exceeds its authenticated token shard")
    token_ids = tuple(int(value) for value in tokens)
    return V6VampSpanAnchor(
        task_id=task_id,
        split=split,
        normalized_story_sha256=_text(record, "normalized_story_sha256"),
        record_id=_text(record, "record_id"),
        token_offset=start,
        token_ids=token_ids,
        sequence_sha256=sha256(tokens.astype("<u2", copy=False).tobytes()).hexdigest(),
    )


def _language_example(
    anchor: V6VampSpanAnchor,
    prefix_length: int,
    preset: V6VampExperimentPreset,
    pad_token_id: int,
) -> LanguageEvaluationExample:
    router, competence = build_prefix_suffix_batches(
        anchor.token_ids,
        prefix_length,
        preset.suffix_length,
        pad_token_id=pad_token_id,
    )
    task_id = TaskId(anchor.task_id)
    return LanguageEvaluationExample(
        router_batch=router,
        competence_batch=competence,
        task_id=task_id,
        oracle_node_id=NodeId(str(task_id)),
    )


def _suite_example(
    anchor: V6VampSpanAnchor,
    condition: LanguageEvaluationCondition,
    artifact: V6SemanticPartitionArtifact,
    tokenizer: TextTokenizer,
    preset: V6VampExperimentPreset,
) -> LanguageSuiteExample:
    example = _language_example(
        anchor,
        condition.prefix_tokens,
        preset,
        artifact.pad_token_id,
    )
    visible = _visible_target_concepts(
        tokenizer.decode(anchor.token_ids[: condition.prefix_tokens]),
        anchor.task_id,
        artifact,
        preset,
    )
    roles = {concept.split(":", 1)[0] for concept in visible}
    cue_regime = (
        "cue_sufficient"
        if roles == {"noun", "verb"}
        else "cue_present"
        if roles
        else "cue_hidden_or_ambiguous"
    )
    pair_hash = record_sha256(
        {
            "group": anchor.normalized_story_sha256,
            "sequence": anchor.sequence_sha256,
            "start": anchor.token_offset,
            "task": anchor.task_id,
        }
    )
    return LanguageSuiteExample(
        pair_id=pair_hash,
        condition_id=condition.condition_id,
        split="test",
        example=example,
        provenance=LanguageExampleProvenance(
            source_document_id=anchor.record_id,
            token_offset=anchor.token_offset,
            pair_hash=pair_hash,
        ),
        cue_regime=cue_regime,
        visible_concept_ids=visible,
    )


def _visible_target_concepts(
    prefix_text: str,
    world: str,
    artifact: V6SemanticPartitionArtifact,
    preset: V6VampExperimentPreset,
) -> tuple[str, ...]:
    coordinates = {
        label: (noun_cluster, verb_cluster)
        for label, noun_cluster, verb_cluster in preset.world_coordinates
    }
    noun_cluster, verb_cluster = coordinates[world]
    target_clusters = {
        (cluster.role, cluster.index): cluster
        for cluster in artifact.semantic_catalog.clusters
        if (cluster.role, cluster.index)
        in (("noun", noun_cluster), ("verb", verb_cluster))
    }
    if set(target_clusters) != {("noun", noun_cluster), ("verb", verb_cluster)}:
        raise ValueError("semantic-v6 target cluster inventory changed")
    return tuple(
        sorted(
            f"{role}:{cluster_index}:{word}"
            for (role, cluster_index), cluster in target_clusters.items()
            for word in cluster.words
            if exact_whole_word_spans(prefix_text, word)
        )
    )


def _require_canonical_sources(
    artifact: V6SemanticPartitionArtifact,
    preset: V6VampExperimentPreset,
) -> None:
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 VAMP requires its strict partition")
    if type(preset) is not V6VampExperimentPreset:
        raise TypeError("semantic-v6 VAMP requires its strict preset")
    actual_coordinates = tuple(
        (cell.label, cell.noun_bucket, cell.verb_bucket) for cell in artifact.cells
    )
    if (
        artifact.partition_sha256 != preset.partition_sha256
        or artifact.semantic_catalog.catalog_sha256 != preset.catalog_sha256
        or actual_coordinates != preset.world_coordinates
    ):
        raise ValueError("semantic-v6 VAMP source identity changed")


def _index_filename(selector: str) -> str:
    parts = selector.split("/")
    if len(parts) == 2 and parts == ["base", "validation"]:
        return "base-validation.jsonl"
    if (
        len(parts) == 3
        and parts[0] == "world"
        and parts[1] in V6_VAMP_EXPERIMENT_PRESET.task_order
        and parts[2] in ("validation", "test")
    ):
        return f"world-{parts[1]}-{parts[2]}.jsonl"
    raise ValueError("semantic-v6 VAMP anchor selector is not validation or test")


def _iter_index(path: Path):
    with path.open("rb") as source:
        for line in source:
            record = json.loads(line)
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError(f"semantic-v6 VAMP index is not canonical: {path}")
            yield record


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"semantic-v6 VAMP field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 VAMP field {field!r} must be nonnegative")
    return value


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = [
    "V6PreparedVampCurriculum",
    "V6VampPreparationProgress",
    "V6VampSpanAnchor",
    "prepare_v6_vamp_test_suite",
    "prepare_v6_vamp_training_curriculum",
    "select_v6_vamp_anchors",
]
