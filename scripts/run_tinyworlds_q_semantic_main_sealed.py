#!/usr/bin/env python3
"""Open the frozen main test once, publish its report, and close transaction."""

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
)
from apm.data.text.tinyworlds_q_semantic.catalog import (  # noqa: E402
    load_sealed_catalog,
    publish_opened_sealed_audit,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import (  # noqa: E402
    evaluate_staged_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.execution import (  # noqa: E402
    begin_sealed_test,
    completed_sealed_result_sha256,
    complete_sealed_test,
    load_sealed_transaction,
)
from apm.data.text.tinyworlds_q_semantic.final_analysis import (  # noqa: E402
    compute_registered_final_effects,
)
from apm.data.text.tinyworlds_q_semantic.generation import (  # noqa: E402
    generate_semantic_fact_inspections,
)
from apm.data.text.tinyworlds_q_semantic.main_validation import (  # noqa: E402
    find_main_validation_artifact,
)
from apm.data.text.tinyworlds_q_semantic.queries import (  # noqa: E402
    compile_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_catalog import (  # noqa: E402
    MAIN_CATALOG_SHA256,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_partition import (  # noqa: E402
    load_registered_main_partition,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_preflight import (  # noqa: E402
    load_registered_main_gpu_preflight,
)
from apm.data.text.tinyworlds_q_semantic.report import (  # noqa: E402
    find_semantic_report,
    publish_semantic_report,
)
from apm.data.text.tinyworlds_q_semantic.result_stream import (  # noqa: E402
    stream_semantic_results,
)
from apm.data.text.tinyworlds_q_semantic.selected_base import (  # noqa: E402
    load_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.training import (  # noqa: E402
    allocator_peak_bytes,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint  # noqa: E402
from apm.lm.text import TokenizersTextTokenizer  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
RESULT_ROOT = (
    REPOSITORY_ROOT / "results" / "language_cl" / "tinyworlds-q-semantic-v1"
)
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)


def main() -> int:
    """Execute or safely finish the one authenticated sealed transaction."""
    print("Phase 1/6: authenticating all frozen pre-test artifacts.", flush=True)
    _frozen, preset, validation_catalog, partition = load_registered_main_partition()
    preflight = load_registered_main_gpu_preflight(
        partition,
        validation_catalog,
        preset,
        CHECKPOINT_ROOT,
    )
    transaction = load_sealed_transaction(
        RESULT_ROOT / "main-sealed-transaction"
    )
    selected_base = load_query_selected_base(
        CHECKPOINT_ROOT / "base" / transaction.selected_base_sha256,
        partition,
        preset,
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    prepared = prepare_query_adaptation(
        validation_catalog,
        partition,
        tokenizer,
        preset,
    )
    stage_directories = tuple(
        CHECKPOINT_ROOT
        / "main-adaptations"
        / preset.config_sha256
        / "stages"
        / f"stage-{stage:03d}"
        for stage in range(1, preset.active_world_count + 1)
    )
    validation = find_main_validation_artifact(
        RESULT_ROOT / "main-validation",
        artifact=partition,
        preset=preset,
        selected_base=selected_base,
        preflight=preflight,
        prepared=prepared,
        stage_directories=stage_directories,
    )
    if validation is None:
        raise RuntimeError("sealed test requires the immutable main validation freeze")
    expected_transaction = (
        partition.catalog_sha256,
        partition.partition_sha256,
        selected_base.selection_sha256,
        validation.validation_sha256,
        preset.config_sha256,
    )
    actual_transaction = (
        transaction.catalog_sha256,
        transaction.partition_sha256,
        transaction.selected_base_sha256,
        transaction.adapters_sha256,
        transaction.config_sha256,
    )
    if actual_transaction != expected_transaction:
        raise ValueError("sealed transaction does not bind the frozen main artifacts")

    existing = find_semantic_report(
        RESULT_ROOT / "main-report",
        catalog_sha256=partition.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        adapters_sha256=validation.validation_sha256,
        preflight_sha256=preflight.preflight_sha256,
        transaction_sha256=transaction.transaction_sha256,
        preset=preset,
    )
    completed = completed_sealed_result_sha256(transaction)
    if completed is not None:
        if existing is None or existing.report_sha256 != completed:
            raise RuntimeError("sealed completion does not name its strict report")
        print(f"Sealed result already complete: {existing.report_sha256}", flush=True)
        print(f"Report: {existing.root / 'report.md'}", flush=True)
        return 0
    if existing is not None:
        if not transaction.test_access_authorized:
            raise RuntimeError("published sealed report lacks its authenticated opening")
        complete_sealed_test(transaction, existing.report_sha256)
        print(f"Recovered and closed sealed result: {existing.report_sha256}", flush=True)
        return 0

    print("Phase 2/6: durably opening the sole sealed-test transaction.", flush=True)
    begin_sealed_test(transaction)
    catalog = load_sealed_catalog(
        CATALOG_ROOT / "catalog" / MAIN_CATALOG_SHA256,
        transaction,
    )
    publish_opened_sealed_audit(catalog, transaction, transaction.root)
    queries = compile_semantic_queries(
        catalog,
        tokenizer,
        split="test",
        maximum_context_tokens=preset.context_length,
    )

    print("Phase 3/6: evaluating every frozen stage and method.", flush=True)
    loaded_base = load_gpt_neo_checkpoint(selected_base.checkpoint)
    stages = tuple(
        load_language_adaptation_artifact(path) for path in stage_directories
    )
    evaluation_started = time.monotonic()
    evaluation_phase_started: dict[str, float] = {}

    def progress(phase: str, completed_chunks: int, total_chunks: int) -> None:
        phase_started = evaluation_phase_started.setdefault(phase, time.monotonic())
        if (
            completed_chunks != 1
            and completed_chunks != total_chunks
            and completed_chunks % max(1, total_chunks // 10) != 0
        ):
            return
        elapsed = time.monotonic() - phase_started
        remaining = elapsed * (total_chunks - completed_chunks) / max(
            1,
            completed_chunks,
        )
        print(
            f"Main sealed {phase}: {completed_chunks:,}/{total_chunks:,} chunks; "
            f"phase ETA {_duration(remaining)}",
            flush=True,
        )

    results, _streamed = stream_semantic_results(
        RESULT_ROOT / "work",
        "main-sealed",
        lambda result_sink: evaluate_staged_semantic_queries(
            queries,
            loaded_base,
            stages,
            preset,
            tokenizer.pad_token_id,
            progress=progress,
            result_sink=result_sink,
        ),
        size_limit_bytes=preset.result_size_limit_bytes,
    )
    evaluation_seconds = time.monotonic() - evaluation_started

    print("Phase 4/6: running frozen secondary generation inspection.", flush=True)
    generation_started = time.monotonic()
    generation = generate_semantic_fact_inspections(
        catalog,
        loaded_base,
        stages[-1],
        preset,
        tokenizer,
    )
    generation_seconds = time.monotonic() - generation_started

    print("Phase 5/6: computing preregistered fact-level effects.", flush=True)
    analysis_started = time.monotonic()
    effects = compute_registered_final_effects(results, preset)
    analysis_seconds = time.monotonic() - analysis_started
    validation_runtime = dict(validation.runtime_seconds)
    runtime_seconds = {
        "base_training_projected": dict(preflight.runtime_seconds)["base_training"],
        **validation_runtime,
        "sealed_evaluation": evaluation_seconds,
        "sealed_generation": generation_seconds,
        "sealed_fact_analysis": analysis_seconds,
    }
    peak = max(
        preflight.measurement.allocator_peak_bytes,
        selected_base.allocator_peak_bytes,
        validation.allocator_peak_bytes,
        allocator_peak_bytes(),
    )
    if peak > preset.allocator_peak_limit_bytes:
        raise MemoryError("sealed evaluation exceeded the frozen allocator limit")

    print("Phase 6/6: atomically publishing and closing the sealed result.", flush=True)
    report = publish_semantic_report(
        RESULT_ROOT / "main-report",
        catalog_sha256=partition.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        adapters_sha256=validation.validation_sha256,
        preflight_sha256=preflight.preflight_sha256,
        transaction_sha256=transaction.transaction_sha256,
        preset=preset,
        results=results,
        effects=effects,
        generation=generation,
        runtime_seconds=runtime_seconds,
        memory_bytes={
            "allocator_limit": preset.allocator_peak_limit_bytes,
            "allocator_peak": peak,
            "preflight_allocator_peak": preflight.measurement.allocator_peak_bytes,
        },
    )
    complete_sealed_test(transaction, report.report_sha256)
    print(f"Sealed result: {report.report_sha256}", flush=True)
    print(f"Report: {report.root / 'report.md'}", flush=True)
    return 0


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
