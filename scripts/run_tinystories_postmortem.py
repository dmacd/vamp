"""Retrain once and publish the fixed TinyStories topic post-mortem."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from apm.continual.language_adaptation_artifact import (
        LanguageAdaptationArtifact,
    )
    from apm.continual.language_benchmark_run import (
        LanguageBenchmarkResult,
        LanguageBenchmarkSettings,
    )
    from apm.continual.language_evaluation import LanguageEvaluationSuite
    from apm.continual.language_evaluation_run import (
        LanguageEvaluationBenchmark,
    )
    from apm.data.text.curricula import (
        TinyStoriesSingleGpuPreset,
        TinyStoriesTopicDataset,
    )
    from apm.data.text.language_tasks import PreparedLanguageCurriculum
    from apm.data.text.tinystories_evaluation import (
        TinyStoriesEvaluationTaskDocuments,
    )
    from apm.lm.lora import LoraConfig
    from apm.lm.text import TokenizersTextTokenizer
    from apm.lm.tinystories_conversion import LoadedTinyStoriesArtifact
    from apm.lm.training import LmTrainConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "tinystories-v2"
BASE_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m"
ADAPTATION_ARTIFACT_ROOT = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "language-adaptations"
    / "tinystories-v2-gpt4-topic"
)
RESULTS_ROOT = REPOSITORY_ROOT / "results"
HISTORICAL_REPORT_DIRECTORY = (
    RESULTS_ROOT
    / "language_cl"
    / "tinystories-v2-gpt4"
    / "topic"
    / "single-gpu-seed0-9f715620e7c2"
)


@dataclass(frozen=True)
class _Phase:
    number: int
    name: str
    estimated_seconds: int


PHASES = (
    _Phase(1, "preserve the completed historical report", 5),
    _Phase(2, "prepare the exact training inputs", 360),
    _Phase(3, "run the exact benchmark retrain once", 7_200),
    _Phase(4, "publish and reload the adaptation artifact", 120),
    _Phase(5, "build the complete paired evaluation suite", 300),
    _Phase(6, "evaluate the reloaded artifact without training", 3_600),
    _Phase(7, "write and reproduce the enhanced report", 180),
)


@dataclass(frozen=True)
class _TrainingInputs:
    preset: TinyStoriesSingleGpuPreset
    topic_dataset: TinyStoriesTopicDataset
    base_artifact: LoadedTinyStoriesArtifact
    tokenizer: TokenizersTextTokenizer
    prepared: PreparedLanguageCurriculum
    lora_config: LoraConfig
    train_config: LmTrainConfig
    settings: LanguageBenchmarkSettings


@dataclass(frozen=True)
class _PersistedAdaptation:
    directory: Path
    identity: str
    artifact: LanguageAdaptationArtifact


@dataclass(frozen=True)
class _EvaluationInputs:
    task_documents: tuple[TinyStoriesEvaluationTaskDocuments, ...]
    suite: LanguageEvaluationSuite


@dataclass(frozen=True)
class _EvaluationOutcome:
    benchmark: LanguageEvaluationBenchmark
    tensor_checksums_before: tuple[tuple[str, str], ...]
    tensor_checksums_after: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ReportOutcome:
    directory: Path
    tree_checksum: str
    file_count: int


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the displayed progress."""

    def close(self) -> None:
        """Close the displayed progress."""

    def write(self, message: str) -> object:
        """Print without corrupting the displayed progress."""


ResultT = TypeVar("ResultT")


class _PostmortemProgress:
    """Emit phase lines, persistent events, and phase/overall ETA bars."""

    def __init__(self, temporary_directory: Path) -> None:
        self._temporary_directory = temporary_directory
        self._overall_bar: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None

    def __enter__(self) -> _PostmortemProgress:
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall_bar = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="TinyStories post-mortem overall",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if self._overall_bar is not None:
            self._overall_bar.close()

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one operation while streaming its progress and outcome."""
        if self._overall_bar is None or self._tqdm_factory is None:
            raise RuntimeError("post-mortem progress must be entered before use")
        line = f"Phase {phase.number}/{len(PHASES)}: {phase.name}"
        self._overall_bar.write(line)
        started_at = monotonic()
        _append_jsonl(
            self._temporary_directory / "progress.jsonl",
            {"event": "phase_started", "name": phase.name, "phase": phase.number},
        )
        phase_bar = self._tqdm_factory(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)}",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )
        stop = Event()
        timer = Thread(
            target=_advance_eta_bars,
            args=(stop, phase_bar, self._overall_bar, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            result = operation()
        except BaseException as error:
            stop.set()
            timer.join()
            _append_jsonl(
                self._temporary_directory / "progress.jsonl",
                {
                    "elapsed_seconds": monotonic() - started_at,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "event": "phase_failed",
                    "name": phase.name,
                    "phase": phase.number,
                },
            )
            raise
        else:
            stop.set()
            timer.join()
            remaining = max(0.0, phase.estimated_seconds - phase_bar.n)
            phase_bar.update(remaining)
            self._overall_bar.update(remaining)
            _append_jsonl(
                self._temporary_directory / "progress.jsonl",
                {
                    "elapsed_seconds": monotonic() - started_at,
                    "event": "phase_completed",
                    "name": phase.name,
                    "phase": phase.number,
                },
            )
            return result
        finally:
            stop.set()
            if timer.is_alive():
                timer.join()
            phase_bar.close()


def _advance_eta_bars(
    stop: Event,
    phase_bar: _TqdmBar,
    overall_bar: _TqdmBar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds - 1:
            phase_bar.update(1)
            overall_bar.update(1)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise FileNotFoundError(f"report directory does not exist: {directory}")
    files = tuple(sorted(path for path in directory.rglob("*") if path.is_file()))
    if not files:
        raise ValueError(f"report directory contains no files: {directory}")
    return {
        path.relative_to(directory).as_posix(): path.read_bytes() for path in files
    }


def _tree_checksum(files: dict[str, bytes]) -> str:
    digest = sha256()
    for relative_path, contents in files.items():
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def _snapshot_historical_report() -> dict[str, bytes]:
    return _directory_bytes(HISTORICAL_REPORT_DIRECTORY)


def _prepare_training_inputs() -> _TrainingInputs:
    from apm.continual.language_benchmark_run import LanguageBenchmarkSettings
    from apm.data.text.curricula import (
        TINYSTORIES_SINGLE_GPU_PRESET,
        TINYSTORIES_V2_SOURCE,
        load_tinystories_topic_dataset,
    )
    from apm.data.text.language_tasks import (
        LanguageDataBuildConfig,
        prepare_language_curriculum,
        raw_tasks_from_document_curriculum,
    )
    from apm.lm.lora import LoraConfig
    from apm.lm.text import TokenizersTextTokenizer
    from apm.lm.tinystories_conversion import load_tinystories_artifact
    from apm.lm.training import LmTrainConfig

    preset = TINYSTORIES_SINGLE_GPU_PRESET
    topic_dataset = load_tinystories_topic_dataset(
        DATA_DIRECTORY / TINYSTORIES_V2_SOURCE.train_file.filename,
        DATA_DIRECTORY / TINYSTORIES_V2_SOURCE.validation_file.filename,
        TINYSTORIES_V2_SOURCE,
        preset.stories_per_task,
    )
    base_artifact = load_tinystories_artifact(BASE_ARTIFACT_DIRECTORY)
    tokenizer = TokenizersTextTokenizer.from_file(
        BASE_ARTIFACT_DIRECTORY / "tokenizer" / "tokenizer.json"
    )
    build_config = LanguageDataBuildConfig(
        context_length=preset.context_length,
        batch_size=preset.batch_size,
        stride=preset.context_length,
        prefix_lengths=preset.evaluation.prefix_lengths,
        suffix_length=preset.evaluation.suffix_length,
        examples_per_task_and_prefix=(
            preset.evaluation_examples_per_task_and_prefix
        ),
        primary_prefix_length=64,
    )
    prepared = prepare_language_curriculum(
        topic_dataset.curriculum.curriculum_id,
        raw_tasks_from_document_curriculum(topic_dataset.curriculum),
        tuple(document.text for document in topic_dataset.root_validation),
        tokenizer,
        build_config,
    )
    lora_config = LoraConfig(rank=preset.lora_rank, alpha=preset.lora_alpha)
    train_config = LmTrainConfig(
        learning_rate=1e-3,
        steps=preset.adapter_steps_per_task,
        batch_size=preset.batch_size,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
    )
    settings = LanguageBenchmarkSettings(
        seed=0,
        random_router_seed=0,
        evaluation_microbatch_size=8,
        peak_device_memory_target_bytes=preset.peak_device_memory_gib * 1024**3,
    )
    return _TrainingInputs(
        preset,
        topic_dataset,
        base_artifact,
        tokenizer,
        prepared,
        lora_config,
        train_config,
        settings,
    )


def _run_exact_benchmark(inputs: _TrainingInputs) -> LanguageBenchmarkResult:
    from apm.continual.language_benchmark_run import run_language_benchmark

    checkpoint = inputs.base_artifact.checkpoint
    return run_language_benchmark(
        inputs.prepared,
        checkpoint.reference,
        checkpoint.params,
        checkpoint.config,
        inputs.lora_config,
        inputs.train_config,
        inputs.tokenizer,
        inputs.settings,
    )


def _training_config_payload(inputs: _TrainingInputs) -> dict[str, object]:
    from apm.data.text.curricula import (
        TINYSTORIES_TOPICS,
        TINYSTORIES_V2_SOURCE,
    )

    checkpoint = inputs.base_artifact.checkpoint
    return {
        "adapter": asdict(inputs.lora_config),
        "base_checkpoint": {
            "manifest_sha256": checkpoint.reference.manifest_sha256,
            "parameter_checksum": checkpoint.reference.parameter_checksum,
        },
        "benchmark": asdict(inputs.settings),
        "checkpoint_source": asdict(checkpoint.source),
        "curriculum": {
            "max_edges": inputs.prepared.curriculum.max_edges,
            "max_nodes": inputs.prepared.curriculum.max_nodes,
            "task_ids": [
                str(task.task_id) for task in inputs.prepared.curriculum.tasks
            ],
        },
        "data": asdict(inputs.prepared.build_config),
        "dataset_source": asdict(TINYSTORIES_V2_SOURCE),
        "model": asdict(checkpoint.config),
        "optimizer": asdict(inputs.train_config),
        "preset": asdict(inputs.preset),
        "tokenizer": asdict(checkpoint.tokenizer),
        "topic_lexicons": {
            topic.name: {
                concept.name: list(concept.forms) for concept in topic.concepts
            }
            for topic in TINYSTORIES_TOPICS
        },
        "topic_rule": {
            "matching": "case-folded whole words",
            "minimum_distinct_concepts": 2,
            "minimum_winner_margin": 1,
            "selection": "lowest normalized-content SHA-256",
        },
    }


def _artifact_identity_payload(
    artifact: LanguageAdaptationArtifact,
) -> dict[str, object]:
    return {
        "base_checkpoint": {
            "manifest_sha256": artifact.base_checkpoint.manifest_sha256,
            "parameter_checksum": artifact.base_checkpoint.parameter_checksum,
        },
        "capacities": {
            "max_edges": artifact.max_edges,
            "max_nodes": artifact.max_nodes,
        },
        "config_hashes": dict(artifact.config_hashes),
        "graph": [
            {
                "depth": node.depth,
                "node_id": str(node.node_id),
                "parent_id": (
                    None if node.parent_id is None else str(node.parent_id)
                ),
                "train_stage": node.train_stage,
                "trained_task": (
                    None if node.trained_task is None else str(node.trained_task)
                ),
            }
            for node in artifact.vamp_graph.nodes
        ],
        "task_order": [str(task_id) for task_id in artifact.task_order],
        "tensor_checksum": artifact.tensor_checksum,
        "tensor_checksums": dict(artifact.tensor_checksums),
    }


def _persist_adaptation(
    inputs: _TrainingInputs,
    benchmark: LanguageBenchmarkResult,
) -> _PersistedAdaptation:
    from apm.continual.language_adaptation_artifact import (
        extract_language_adaptation_artifact,
        load_language_adaptation_artifact,
        save_language_adaptation_artifact,
    )
    from apm.continual.language_report import canonical_config_json

    training_config_hash = sha256(
        canonical_config_json(_training_config_payload(inputs)).encode("utf-8")
    ).hexdigest()
    extracted = extract_language_adaptation_artifact(
        benchmark.adaptations,
        inputs.base_artifact.checkpoint.config,
        inputs.lora_config,
        config_hashes={"training_run": training_config_hash},
    )
    identity = sha256(
        canonical_config_json(_artifact_identity_payload(extracted)).encode("utf-8")
    ).hexdigest()
    directory = ADAPTATION_ARTIFACT_ROOT / identity
    if directory.exists():
        loaded = load_language_adaptation_artifact(directory)
    else:
        save_language_adaptation_artifact(directory, extracted)
        loaded = load_language_adaptation_artifact(directory)
    loaded_identity = sha256(
        canonical_config_json(_artifact_identity_payload(loaded)).encode("utf-8")
    ).hexdigest()
    if loaded_identity != identity:
        raise RuntimeError("reloaded adaptation artifact identity changed")
    if loaded.tensor_checksums != extracted.tensor_checksums:
        raise RuntimeError("reloaded adaptation artifact tensor checksums changed")
    return _PersistedAdaptation(directory.resolve(), identity, loaded)


def _build_evaluation_inputs(
    inputs: _TrainingInputs,
) -> _EvaluationInputs:
    from apm.data.text.curricula import TINYSTORIES_V2_SOURCE
    from apm.data.text.tinystories_evaluation import (
        build_tinystories_postmortem_suite,
        load_complete_classified_tinystories_test_half,
    )

    task_documents = load_complete_classified_tinystories_test_half(
        DATA_DIRECTORY / TINYSTORIES_V2_SOURCE.validation_file.filename,
        TINYSTORIES_V2_SOURCE,
    )
    return _EvaluationInputs(
        task_documents,
        build_tinystories_postmortem_suite(task_documents, inputs.tokenizer),
    )


def _evaluate_loaded_artifact(
    inputs: _TrainingInputs,
    persisted: _PersistedAdaptation,
    evaluation_inputs: _EvaluationInputs,
) -> _EvaluationOutcome:
    from apm.continual.language_evaluation_run import evaluate_language_benchmark

    tensor_checksums_before = persisted.artifact.tensor_checksums
    checkpoint = inputs.base_artifact.checkpoint
    benchmark = evaluate_language_benchmark(
        persisted.artifact,
        evaluation_inputs.suite,
        checkpoint.params,
        checkpoint.config,
        inputs.lora_config,
        inputs.settings,
    )
    tensor_checksums_after = persisted.artifact.tensor_checksums
    if tensor_checksums_before != tensor_checksums_after:
        raise RuntimeError("evaluation changed a persisted adaptation tensor")
    if (
        benchmark.adaptation_checksum_before != persisted.artifact.tensor_checksum
        or benchmark.adaptation_checksum_after != persisted.artifact.tensor_checksum
    ):
        raise RuntimeError("evaluation aggregate adaptation checksum is inconsistent")
    return _EvaluationOutcome(
        benchmark,
        tensor_checksums_before,
        tensor_checksums_after,
    )


def _evaluation_suite_payload(
    evaluation_inputs: _EvaluationInputs,
) -> dict[str, object]:
    suite = evaluation_inputs.suite
    return {
        "benchmark_label": suite.benchmark_label,
        "conditions": [asdict(condition) for condition in suite.conditions],
        "example_count": len(suite.examples),
        "pair_count": len(
            {
                (str(example.task_id), example.pair_id)
                for example in suite.examples
            }
        ),
        "primary_condition_id": suite.primary_condition_id,
        "provenance_policy": {
            "anchor_selection": (
                "128 exact 256-token spans per task at stride 32 with "
                "source-story round-robin selection"
            ),
            "classification_scope": "complete classified official test half",
            "cue_derivation": "visible prefix text only; no example filtering",
            "pairing": (
                "span-level nested prefixes sharing one ordered pair ID and "
                "answer-suffix anchor"
            ),
            "provenance_fields": [
                "source_document_id",
                "token_offset",
                "pair_hash",
            ],
        },
        "suite_id": suite.suite_id,
        "tasks": [
            {
                "classified_test_documents": len(task.documents),
                "pair_count": len(
                    {
                        example.pair_id
                        for example in suite.examples
                        if example.task_id == task.task_id
                    }
                ),
                "source_story_count": len(
                    {
                        example.provenance.source_document_id
                        for example in suite.examples
                        if example.task_id == task.task_id
                    }
                ),
                "task_id": str(task.task_id),
                "topic": task.topic,
            }
            for task in evaluation_inputs.task_documents
        ],
    }


def _write_verified_report(
    historical_bytes: dict[str, bytes],
    inputs: _TrainingInputs,
    benchmark: LanguageBenchmarkResult,
    persisted: _PersistedAdaptation,
    evaluation_inputs: _EvaluationInputs,
    evaluation: _EvaluationOutcome,
) -> _ReportOutcome:
    from apm.continual.language_evaluation import (
        IN_DOMAIN_TOPIC_SPECIALIZATION,
    )
    from apm.continual.language_report import (
        LanguageReportManifest,
        canonical_config_json,
        language_report_directory,
        write_language_report,
    )
    from apm.continual.language_report_build import build_language_report_bundle

    if evaluation_inputs.suite.benchmark_label != IN_DOMAIN_TOPIC_SPECIALIZATION:
        raise RuntimeError("post-mortem suite benchmark label is not canonical")
    manifest = LanguageReportManifest(
        dataset="tinystories-v2-gpt4",
        curriculum="topic",
        preset="single-gpu-postmortem",
        seed=inputs.settings.seed,
        interpretation=IN_DOMAIN_TOPIC_SPECIALIZATION,
        config_json=canonical_config_json(
            {
                "adaptation_artifact": {
                    "config_hashes": dict(persisted.artifact.config_hashes),
                    "identity": persisted.identity,
                    "tensor_checksum": persisted.artifact.tensor_checksum,
                    "tensor_checksums": dict(persisted.artifact.tensor_checksums),
                },
                "evaluation": {
                    "adaptation_checksum_after": (
                        evaluation.benchmark.adaptation_checksum_after
                    ),
                    "adaptation_checksum_before": (
                        evaluation.benchmark.adaptation_checksum_before
                    ),
                    "tensor_checksums_after": dict(
                        evaluation.tensor_checksums_after
                    ),
                    "tensor_checksums_before": dict(
                        evaluation.tensor_checksums_before
                    ),
                },
                "evaluation_suite": _evaluation_suite_payload(evaluation_inputs),
                "training": _training_config_payload(inputs),
            }
        ),
    )
    expected_directory = language_report_directory(RESULTS_ROOT, manifest)
    if expected_directory.resolve() == HISTORICAL_REPORT_DIRECTORY.resolve():
        raise RuntimeError("post-mortem report would overwrite the historical report")
    bundle = build_language_report_bundle(
        manifest,
        inputs.prepared,
        benchmark,
        inputs.base_artifact.checkpoint.params,
        evaluation_inputs.suite,
        evaluation.benchmark,
    )
    output_directory = write_language_report(RESULTS_ROOT, bundle)
    first_render = _directory_bytes(output_directory)
    required_artifacts = {
        "cue_coverage.jsonl",
        "cue_metrics.jsonl",
        "evaluation_examples.jsonl",
    }
    missing = required_artifacts.difference(first_render)
    if missing:
        raise RuntimeError(
            f"enhanced report omitted required artifacts: {sorted(missing)}"
        )
    repeated_directory = write_language_report(RESULTS_ROOT, bundle)
    second_render = _directory_bytes(repeated_directory)
    if output_directory.resolve() != repeated_directory.resolve():
        raise RuntimeError("repeated report render changed its output directory")
    if first_render != second_render:
        raise RuntimeError("completed report did not reproduce byte-for-byte")
    if historical_bytes != _directory_bytes(HISTORICAL_REPORT_DIRECTORY):
        raise RuntimeError("post-mortem run changed the completed historical report")
    return _ReportOutcome(
        output_directory.resolve(),
        _tree_checksum(second_render),
        len(second_render),
    )


def main() -> None:
    """Run the fixed retrain, persisted evaluation, and reproducible report."""
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="tinystories-postmortem-")
    ).resolve()
    print(f"Temporary artifact directory: {temporary_directory}", flush=True)
    with _PostmortemProgress(temporary_directory) as progress:
        historical_bytes = progress.run(PHASES[0], _snapshot_historical_report)
        inputs = progress.run(PHASES[1], _prepare_training_inputs)
        benchmark = progress.run(
            PHASES[2], lambda: _run_exact_benchmark(inputs)
        )
        persisted = progress.run(
            PHASES[3], lambda: _persist_adaptation(inputs, benchmark)
        )
        evaluation_inputs = progress.run(
            PHASES[4], lambda: _build_evaluation_inputs(inputs)
        )
        evaluation = progress.run(
            PHASES[5],
            lambda: _evaluate_loaded_artifact(
                inputs,
                persisted,
                evaluation_inputs,
            ),
        )
        report = progress.run(
            PHASES[6],
            lambda: _write_verified_report(
                historical_bytes,
                inputs,
                benchmark,
                persisted,
                evaluation_inputs,
                evaluation,
            ),
        )
    _write_json(
        temporary_directory / "result.json",
        {
            "adaptation_artifact_directory": str(persisted.directory),
            "adaptation_artifact_identity": persisted.identity,
            "adaptation_tensor_checksum": persisted.artifact.tensor_checksum,
            "adaptation_tensor_checksums_after": dict(
                evaluation.tensor_checksums_after
            ),
            "adaptation_tensor_checksums_before": dict(
                evaluation.tensor_checksums_before
            ),
            "evaluation_suite_id": evaluation_inputs.suite.suite_id,
            "historical_report_directory": str(
                HISTORICAL_REPORT_DIRECTORY.resolve()
            ),
            "report_directory": str(report.directory),
            "report_file_count": report.file_count,
            "report_tree_checksum": report.tree_checksum,
        },
    )
    print(f"Adaptation artifact: {persisted.directory}", flush=True)
    print(f"Post-mortem report: {report.directory}", flush=True)


if __name__ == "__main__":
    main()
