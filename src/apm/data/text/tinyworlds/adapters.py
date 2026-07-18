"""Adapters from rendered TinyWorlds records to language and knowledge tasks."""

from __future__ import annotations

from dataclasses import dataclass

from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.continual.language_tasks import LanguageCurriculum, LanguageTask
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    PreparedLanguageCurriculum,
    RawTextTask,
    prepare_language_curriculum,
)
from apm.data.text.tinyworlds.rendering import RenderedStory, RenderedTinyWorlds
from apm.data.text.tinyworlds.schema import DataSplit
from apm.lm.text import TextTokenizer


@dataclass(frozen=True, slots=True)
class TinyWorldsTrainingDataConfig:
    """Fixed fact exposure and document-packing policy for adapter training."""

    facts_per_task: int = 24
    exposures_per_fact: int = 32
    batch_size: int = 32
    context_length: int = 256
    evaluation_examples_per_task: int = 128

    def __post_init__(self) -> None:
        values = (
            self.facts_per_task,
            self.exposures_per_fact,
            self.batch_size,
            self.context_length,
            self.evaluation_examples_per_task,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("training data dimensions must be positive integers")
        if self.context_length != 256:
            raise ValueError("TinyWorlds uses the fixed 256-token context")


TINYWORLDS_TRAINING_DATA_CONFIG = TinyWorldsTrainingDataConfig()


@dataclass(frozen=True, slots=True)
class PreparedTinyWorldsCurriculum:
    """Natural-continuation language data plus explicit semantic query suites."""

    rendered_bundle_id: str
    language: PreparedLanguageCurriculum
    validation_queries: tuple[KnowledgeQuery, ...]
    test_queries: tuple[KnowledgeQuery, ...]
    training_story_ids: tuple[str, ...]
    training_config: TinyWorldsTrainingDataConfig

    def __post_init__(self) -> None:
        if type(self.rendered_bundle_id) is not str or not self.rendered_bundle_id:
            raise ValueError("rendered_bundle_id must be nonempty")
        if type(self.language) is not PreparedLanguageCurriculum:
            raise TypeError("language must be a PreparedLanguageCurriculum")
        for label, queries in (
            ("validation_queries", self.validation_queries),
            ("test_queries", self.test_queries),
        ):
            if type(queries) is not tuple or not queries or any(
                type(query) is not KnowledgeQuery for query in queries
            ):
                raise TypeError(f"{label} must contain KnowledgeQuery values")
        if type(self.training_story_ids) is not tuple or not self.training_story_ids:
            raise ValueError("training_story_ids must be a nonempty tuple")
        if type(self.training_config) is not TinyWorldsTrainingDataConfig:
            raise TypeError("training_config must be a TinyWorldsTrainingDataConfig")
        validation_ids = {query.query_id for query in self.validation_queries}
        test_ids = {query.query_id for query in self.test_queries}
        if validation_ids & test_ids:
            raise ValueError("knowledge validation and test query IDs must be disjoint")


def prepare_tinyworlds_curriculum(
    rendered: RenderedTinyWorlds,
    tokenizer: TextTokenizer,
    training_config: TinyWorldsTrainingDataConfig = TINYWORLDS_TRAINING_DATA_CONFIG,
) -> PreparedTinyWorldsCurriculum:
    """Prepare document-safe training and semantic-boundary knowledge evaluation."""
    if type(rendered) is not RenderedTinyWorlds:
        raise TypeError("rendered must be a RenderedTinyWorlds")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    if type(training_config) is not TinyWorldsTrainingDataConfig:
        raise TypeError("training_config must be a TinyWorldsTrainingDataConfig")
    task_ids = tuple(
        dict.fromkeys(
            story.task_id
            for story in rendered.stories
            if story.task_id is not None
        )
    )
    validation_queries = _queries_for_split(rendered, DataSplit.VALIDATION)
    test_queries = _queries_for_split(rendered, DataSplit.TEST)
    selected_by_task = {
        task_id: _select_fact_exposure_stories(
            tuple(
                story
                for story in rendered.stories
                if story.task_id == task_id and story.split is DataSplit.TRAIN
            ),
            training_config,
        )
        for task_id in task_ids
    }
    raw_tasks = tuple(
        RawTextTask(
            task_id=task_id,
            train_texts=tuple(story.text for story in selected_by_task[task_id]),
            validation_texts=tuple(
                story.text
                for story in rendered.stories
                if story.task_id == task_id
                and story.split is DataSplit.VALIDATION
                and story.purpose == "natural_continuation"
            ),
            test_texts=tuple(
                story.text
                for story in rendered.stories
                if story.task_id == task_id
                and story.split is DataSplit.TEST
                and story.purpose == "natural_continuation"
            ),
        )
        for task_id in task_ids
    )
    root_texts = tuple(
        story.text for story in rendered.stories if story.purpose == "root_validation"
    )
    base_prepared = prepare_language_curriculum(
        rendered.bundle_id,
        raw_tasks,
        root_texts,
        tokenizer,
        LanguageDataBuildConfig(
            context_length=training_config.context_length,
            batch_size=training_config.batch_size,
            stride=training_config.context_length,
            prefix_lengths=(64, 128, 192),
            suffix_length=64,
            examples_per_task_and_prefix=(
                training_config.evaluation_examples_per_task
            ),
            primary_prefix_length=64,
        ),
    )
    tasks = tuple(
        _task_with_semantic_probes(
            task,
            validation_queries,
            training_config.evaluation_examples_per_task,
        )
        for task in base_prepared.curriculum.tasks
    )
    prepared = PreparedLanguageCurriculum(
        curriculum_id=base_prepared.curriculum_id,
        curriculum=LanguageCurriculum(
            tasks=tasks,
            max_nodes=len(tasks) + 1,
            max_edges=len(tasks),
        ),
        root_validation_probes=base_prepared.root_validation_probes,
        evaluation_sweeps=base_prepared.evaluation_sweeps,
        build_config=base_prepared.build_config,
    )
    return PreparedTinyWorldsCurriculum(
        rendered_bundle_id=rendered.bundle_id,
        language=prepared,
        validation_queries=validation_queries,
        test_queries=test_queries,
        training_story_ids=tuple(
            story.story_id
            for task_id in task_ids
            for story in selected_by_task[task_id]
        ),
        training_config=training_config,
    )


def _select_fact_exposure_stories(
    stories: tuple[RenderedStory, ...],
    config: TinyWorldsTrainingDataConfig,
) -> tuple[RenderedStory, ...]:
    if not stories:
        raise ValueError("every task requires rendered training stories")
    observed_fact_order = tuple(
        dict.fromkeys(
            fact_id
            for story in stories
            for alignment in story.alignments
            for fact_id in alignment.fact_ids
        )
    )
    if len(observed_fact_order) < config.facts_per_task:
        raise ValueError(
            f"training corpus exposes {len(observed_fact_order)} facts; "
            f"requires {config.facts_per_task}"
        )
    selected_facts = observed_fact_order[: config.facts_per_task]
    exposure_counts = {fact_id: 0 for fact_id in selected_facts}
    selected: list[RenderedStory] = []
    for _ in range(config.exposures_per_fact):
        for story in stories:
            story_facts = tuple(
                fact_id
                for alignment in story.alignments
                for fact_id in alignment.fact_ids
                if fact_id in exposure_counts
            )
            if (
                len(story_facts) == 1
                and exposure_counts[story_facts[0]] < config.exposures_per_fact
            ):
                selected.append(story)
                exposure_counts[story_facts[0]] += 1
        if all(count == config.exposures_per_fact for count in exposure_counts.values()):
            return tuple(selected)
    raise ValueError(
        "rendered training corpus cannot satisfy the exact per-fact exposure budget"
    )


def _queries_for_split(
    rendered: RenderedTinyWorlds,
    split: DataSplit,
) -> tuple[KnowledgeQuery, ...]:
    return tuple(
        variant.knowledge_query
        for group in rendered.query_groups
        if group.split is split
        for variant in group.variants
    )


def _task_with_semantic_probes(
    task: LanguageTask,
    validation_queries: tuple[KnowledgeQuery, ...],
    probe_count: int,
) -> LanguageTask:
    task_queries = tuple(
        query
        for query in validation_queries
        if query.task_id == task.task_id and query.prefix_length == 64
    )
    if len(task_queries) < probe_count:
        raise ValueError(
            f"task {task.task_id} has {len(task_queries)} semantic probes; "
            f"requires {probe_count}"
        )
    return LanguageTask(
        task_id=task.task_id,
        train_batches=task.train_batches,
        validation_examples=task.validation_examples,
        test_examples=task.test_examples,
        parent_probes=tuple(query.router_batch for query in task_queries[:probe_count]),
        content_key_probes=tuple(
            example.router_batch for example in task.validation_examples[:probe_count]
        ),
    )


__all__ = [
    "TINYWORLDS_TRAINING_DATA_CONFIG",
    "PreparedTinyWorldsCurriculum",
    "TinyWorldsTrainingDataConfig",
    "prepare_tinyworlds_curriculum",
]
