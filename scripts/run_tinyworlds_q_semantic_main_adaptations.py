#!/usr/bin/env python3
"""Train all frozen main adapters, evaluate validation, and freeze test inputs."""

from __future__ import annotations

import os
from pathlib import Path
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.continual.language_adaptation_artifact import (  # noqa: E402
    load_language_adaptation_artifact,
)
from apm.data.text.tinyworlds_q_semantic.adaptation import (  # noqa: E402
    prepare_query_adaptation,
    train_or_resume_query_adaptations,
)
from apm.data.text.tinyworlds_q_semantic.base_execution import (  # noqa: E402
    run_or_resume_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import (  # noqa: E402
    evaluate_staged_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.execution import (  # noqa: E402
    publish_sealed_transaction,
)
from apm.data.text.tinyworlds_q_semantic.main_validation import (  # noqa: E402
    find_main_validation_artifact,
    publish_main_validation_artifact,
)
from apm.data.text.tinyworlds_q_semantic.queries import (  # noqa: E402
    compile_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_partition import (  # noqa: E402
    load_registered_main_partition,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_preflight import (  # noqa: E402
    load_registered_main_gpu_preflight,
)
from apm.data.text.tinyworlds_q_semantic.result_stream import (  # noqa: E402
    stream_semantic_results,
)
from apm.data.text.tinyworlds_q_semantic.training import (  # noqa: E402
    allocator_peak_bytes,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint  # noqa: E402
from apm.lm.text import TokenizersTextTokenizer  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
WORK_ROOT = CHECKPOINT_ROOT / "work" / "main"
RESULT_ROOT = (
    REPOSITORY_ROOT / "results" / "language_cl" / "tinyworlds-q-semantic-v1"
)
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)


def main() -> int:
    """Resume all model systems and stop at a closed sealed-test transaction."""
    print("Phase 1/6: authenticating partition, preflight, and selected base.", flush=True)
    _frozen, preset, catalog, partition = load_registered_main_partition()
    preflight = load_registered_main_gpu_preflight(
        partition,
        catalog,
        preset,
        CHECKPOINT_ROOT,
    )
    selected_base = run_or_resume_query_selected_base(
        partition,
        preset,
        preflight,
        WORK_ROOT,
        CHECKPOINT_ROOT,
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    print("Phase 2/6: compiling validation-only parent and router prefixes.", flush=True)
    prepared = prepare_query_adaptation(catalog, partition, tokenizer, preset)

    adaptation_root = CHECKPOINT_ROOT / "main-adaptations" / preset.config_sha256
    completed_before = _completed_stage_count(
        adaptation_root / "stages",
        preset.active_world_count,
    )
    total_updates = 3 * preset.active_world_count * preset.adapter_updates
    completed_offset = 3 * completed_before * preset.adapter_updates
    adaptation_started = time.monotonic()
    method_index = {"sequential": 0, "independent": 1, "vamp": 2}
    concept_index = {
        concept_id: index for index, concept_id in enumerate(preset.concept_ids)
    }

    def adaptation_progress(
        method: str,
        concept_id: str,
        step: int,
        loss: float,
        phase_total: int,
    ) -> None:
        if step != 1 and step % 100 != 0 and step != phase_total:
            return
        completed = (
            concept_index[concept_id] * 3 * preset.adapter_updates
            + method_index[method] * preset.adapter_updates
            + step
        )
        elapsed = time.monotonic() - adaptation_started
        new_updates = max(1, completed - completed_offset)
        remaining = elapsed * (total_updates - completed) / new_updates
        print(
            f"Main {method}/{concept_id} update {step:,}/{phase_total:,}; "
            f"loss {loss:.6f}; overall ETA {_duration(remaining)}",
            flush=True,
        )

    print(
        f"Phase 3/6: resuming independent, sequential, and VAMP training "
        f"({total_updates:,} total updates).",
        flush=True,
    )
    trained = train_or_resume_query_adaptations(
        prepared,
        partition,
        selected_base,
        adaptation_root,
        preset,
        progress=adaptation_progress,
    )
    adaptation_seconds = time.monotonic() - adaptation_started

    print("Phase 4/6: verifying exact completed-stage resume parity.", flush=True)
    before_resume = load_language_adaptation_artifact(trained.stage_directory)
    resumed = train_or_resume_query_adaptations(
        prepared,
        partition,
        selected_base,
        adaptation_root,
        preset,
    )
    resume_verified = (
        trained.adaptation.tensor_checksum == before_resume.tensor_checksum
        and resumed.adaptation.tensor_checksum == before_resume.tensor_checksum
        and resumed.stage_directory == trained.stage_directory
    )
    if not resume_verified:
        raise RuntimeError("main adaptation resume changed immutable state")
    stage_directories = tuple(
        adaptation_root / "stages" / f"stage-{stage:03d}"
        for stage in range(1, preset.active_world_count + 1)
    )
    adaptation_started_ns = adaptation_root.stat().st_mtime_ns
    adaptation_completed_ns = (
        stage_directories[-1] / "manifest.json"
    ).stat().st_mtime_ns
    if adaptation_completed_ns < adaptation_started_ns:
        raise ValueError("main adaptation stage timestamps are inconsistent")
    adaptation_stage_wall_interval = (
        adaptation_completed_ns - adaptation_started_ns
    ) / 1_000_000_000

    print("Phase 5/6: evaluating and freezing validation-only query results.", flush=True)
    validation_root = RESULT_ROOT / "main-validation"
    existing = find_main_validation_artifact(
        validation_root,
        artifact=partition,
        preset=preset,
        selected_base=selected_base,
        preflight=preflight,
        prepared=prepared,
        stage_directories=stage_directories,
    )
    if existing is None:
        validation_started = time.monotonic()
        validation_phase_started: dict[str, float] = {}
        queries = compile_semantic_queries(
            catalog,
            tokenizer,
            split="validation",
            maximum_context_tokens=preset.context_length,
        )
        loaded_base = load_gpt_neo_checkpoint(selected_base.checkpoint)

        def evaluation_progress(phase: str, completed: int, total: int) -> None:
            phase_started = validation_phase_started.setdefault(phase, time.monotonic())
            if completed != 1 and completed != total and completed % max(1, total // 10) != 0:
                return
            elapsed = time.monotonic() - phase_started
            remaining = elapsed * (total - completed) / max(1, completed)
            print(
                f"Main validation {phase}: {completed:,}/{total:,} chunks; "
                f"phase ETA {_duration(remaining)}",
                flush=True,
            )

        results, _streamed = stream_semantic_results(
            RESULT_ROOT / "work",
            "main-validation",
            lambda result_sink: evaluate_staged_semantic_queries(
                queries,
                loaded_base,
                tuple(
                    load_language_adaptation_artifact(path)
                    for path in stage_directories
                ),
                preset,
                tokenizer.pad_token_id,
                progress=evaluation_progress,
                result_sink=result_sink,
            ),
            size_limit_bytes=preset.result_size_limit_bytes,
        )
        validation_seconds = time.monotonic() - validation_started
        existing = publish_main_validation_artifact(
            validation_root,
            artifact=partition,
            preset=preset,
            selected_base=selected_base,
            preflight=preflight,
            prepared=prepared,
            stage_directories=stage_directories,
            results=results,
            resume_verified=True,
            runtime_seconds={
                "adaptation_or_resume": adaptation_seconds,
                "adaptation_stage_wall_interval": adaptation_stage_wall_interval,
                "validation_evaluation": validation_seconds,
            },
            allocator_peak_bytes=max(
                selected_base.allocator_peak_bytes,
                preflight.measurement.allocator_peak_bytes,
                trained.allocator_peak_bytes,
                resumed.allocator_peak_bytes,
                allocator_peak_bytes(),
            ),
        )
    print(f"Main validation freeze: {existing.validation_sha256}", flush=True)
    print(f"Validation report: {existing.root / 'validation.md'}", flush=True)

    print("Phase 6/6: freezing the still-closed sealed-test transaction.", flush=True)
    transaction = publish_sealed_transaction(
        RESULT_ROOT / "main-sealed-transaction",
        catalog_sha256=partition.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        adapters_sha256=existing.validation_sha256,
        config_sha256=preset.config_sha256,
    )
    print(f"Sealed transaction: {transaction.transaction_sha256}", flush=True)
    print("The transaction is frozen but unopened; sealed test remains closed.", flush=True)
    return 0


def _completed_stage_count(stages_root: Path, maximum: int) -> int:
    present = tuple(
        (stages_root / f"stage-{stage:03d}" / "manifest.json").is_file()
        for stage in range(1, maximum + 1)
    )
    completed = next((index for index, value in enumerate(present) if not value), maximum)
    if any(present[completed:]):
        raise ValueError("main adaptation stages are not one contiguous prefix")
    return completed


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
