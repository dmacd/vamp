"""Immutable validation-only freeze for the five-world model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import html
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

from apm.continual.language_adaptation_artifact import (
    LanguageAdaptationArtifact,
    load_language_adaptation_artifact,
)
from apm.data.text.tinyworlds_q_semantic.adaptation import PreparedQueryAdaptation
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    SemanticQueryResult,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.preflight import QueryGpuPreflight
from apm.data.text.tinyworlds_q_semantic.evaluation import (
    SEMANTIC_QUERY_METHODS,
    SEMANTIC_ROUTED_METHODS,
)
from apm.data.text.tinyworlds_q_semantic.final_protocol import (
    REGISTERED_FINAL_EVALUATION_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.scaling import (
    estimate_resources,
    evaluation_schedule,
)
from apm.data.text.tinyworlds_q_semantic.selected_base import QuerySelectedBase


MAIN_VALIDATION_FORMAT = "tinyworlds-q-semantic-main-validation-v1"
MAIN_VALIDATION_TREE_FORMAT = "tinyworlds-q-semantic-main-validation-tree-v1"


@dataclass(frozen=True, slots=True)
class MainValidationArtifact:
    """Frozen adapters, validation rows, resources, and resume evidence."""

    root: Path
    validation_sha256: str
    catalog_sha256: str
    partition_sha256: str
    selected_base_sha256: str
    preflight_sha256: str
    config_sha256: str
    preparation_sha256: str
    final_adaptation_manifest_sha256: str
    results_sha256: str
    result_count: int
    allocator_peak_bytes: int
    runtime_seconds: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.validation_sha256, "main validation"),
            (self.catalog_sha256, "main validation catalog"),
            (self.partition_sha256, "main validation partition"),
            (self.selected_base_sha256, "main validation base"),
            (self.preflight_sha256, "main validation preflight"),
            (self.config_sha256, "main validation config"),
            (self.preparation_sha256, "main validation preparation"),
            (
                self.final_adaptation_manifest_sha256,
                "main validation final adaptation",
            ),
            (self.results_sha256, "main validation results"),
        ):
            require_sha256(value, label)
        if type(self.result_count) is not int or self.result_count <= 0:
            raise ValueError("main validation result count must be positive")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("main validation allocator peak must be nonnegative")
        if (
            type(self.runtime_seconds) is not tuple
            or not self.runtime_seconds
            or len({name for name, _ in self.runtime_seconds})
            != len(self.runtime_seconds)
            or any(
                type(name) is not str
                or not name
                or not isfinite(seconds)
                or seconds < 0.0
                for name, seconds in self.runtime_seconds
            )
        ):
            raise ValueError("main validation runtime evidence is invalid")


def find_main_validation_artifact(
    output_root: str | Path,
    *,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    selected_base: QuerySelectedBase,
    preflight: QueryGpuPreflight,
    prepared: PreparedQueryAdaptation,
    stage_directories: tuple[Path, ...],
) -> MainValidationArtifact | None:
    """Find the sole strict validation freeze matching all current model artifacts."""
    _validate_sources(
        artifact,
        preset,
        selected_base,
        preflight,
        prepared,
        stage_directories,
    )
    stage_records = list(_stage_records(stage_directories, preset))
    root = Path(output_root)
    candidates = (
        tuple(
            path
            for path in sorted(root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _validation_candidate_matches(
                path,
                artifact,
                preset,
                selected_base,
                preflight,
                prepared,
                stage_records,
            )
        )
        if root.is_dir()
        else ()
    )
    if len(candidates) > 1:
        raise RuntimeError("multiple main validation freezes bind the model artifacts")
    return (
        load_main_validation_artifact(
            candidates[0],
            artifact=artifact,
            preset=preset,
            selected_base=selected_base,
            preflight=preflight,
            prepared=prepared,
            stage_directories=stage_directories,
        )
        if candidates
        else None
    )


def publish_main_validation_artifact(
    output_root: str | Path,
    *,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    selected_base: QuerySelectedBase,
    preflight: QueryGpuPreflight,
    prepared: PreparedQueryAdaptation,
    stage_directories: tuple[Path, ...],
    results: tuple[SemanticQueryResult, ...],
    resume_verified: bool,
    runtime_seconds: Mapping[str, float],
    allocator_peak_bytes: int,
) -> MainValidationArtifact:
    """Freeze validation-visible evidence before authorizing the sealed test."""
    _validate_sources(
        artifact,
        preset,
        selected_base,
        preflight,
        prepared,
        stage_directories,
    )
    _validate_validation_results(results, preset)
    if resume_verified is not True:
        raise ValueError("main validation requires exact completed-stage resume parity")
    if (
        not runtime_seconds
        or any(not isfinite(value) or value < 0.0 for value in runtime_seconds.values())
        or type(allocator_peak_bytes) is not int
        or not 0 <= allocator_peak_bytes <= preset.allocator_peak_limit_bytes
    ):
        raise ValueError("main validation runtime or memory evidence is invalid")
    validation_estimate = estimate_resources(preset, queries_per_world=36)
    sealed_estimate = estimate_resources(preset, queries_per_world=60)
    if len(results) != validation_estimate.result_rows:
        raise ValueError("main validation exact result-row projection changed")
    if sealed_estimate.estimated_result_bytes > preset.result_size_limit_bytes:
        raise OSError("main sealed result projection exceeds the frozen limit")
    stage_records = _stage_records(stage_directories, preset)
    results_sha256 = _results_sha256(results)
    summaries = _validation_summaries(results)
    core = {
        "allocator_peak_bytes": allocator_peak_bytes,
        "benchmark_id": BENCHMARK_ID,
        "catalog_sha256": artifact.catalog_sha256,
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "format": MAIN_VALIDATION_FORMAT,
        "final_evaluation_protocol": (
            REGISTERED_FINAL_EVALUATION_PROTOCOL.as_record()
        ),
        "final_evaluation_protocol_sha256": (
            REGISTERED_FINAL_EVALUATION_PROTOCOL.protocol_sha256
        ),
        "partition_sha256": artifact.partition_sha256,
        "preflight_sha256": preflight.preflight_sha256,
        "preparation_sha256": prepared.preparation_sha256,
        "preflight_result_rows_recorded": preflight.estimate.result_rows,
        "result_count": len(results),
        "result_size_limit_bytes": preset.result_size_limit_bytes,
        "results_sha256": results_sha256,
        "resume_verified": True,
        "runtime_seconds": dict(sorted(runtime_seconds.items())),
        "sealed_test_opened": False,
        "selected_base_sha256": selected_base.selection_sha256,
        "sealed_result_bytes_projected": sealed_estimate.estimated_result_bytes,
        "sealed_result_rows_projected": sealed_estimate.result_rows,
        "stage_artifacts": list(stage_records),
        "validation_summaries": list(summaries),
        "validation_result_rows_projected": validation_estimate.result_rows,
    }
    validation_sha256 = record_sha256(core)
    root = Path(output_root) / validation_sha256
    if root.exists():
        return load_main_validation_artifact(
            root,
            artifact=artifact,
            preset=preset,
            selected_base=selected_base,
            preflight=preflight,
            prepared=prepared,
            stage_directories=stage_directories,
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".main-validation-", dir=root.parent))
    print(f"TinyWorlds-Q main validation temporary artifacts: {staging}", flush=True)
    try:
        record = {**core, "validation_sha256": validation_sha256}
        ledger_path = staging / "validation-results.jsonl"
        with ledger_path.open("wb") as ledger:
            for result in results:
                ledger.write(canonical_json_bytes(result.as_record()))
            ledger.flush()
            os.fsync(ledger.fileno())
        if _file_sha256(ledger_path) != results_sha256:
            raise AssertionError("main validation result ledger changed while writing")
        if ledger_path.stat().st_size > preset.result_size_limit_bytes:
            raise OSError("main validation result ledger exceeds the frozen limit")
        markdown = _render_validation_report(record)
        payloads = {
            "validation.json": canonical_json_bytes(record),
            "validation.md": markdown.encode("utf-8"),
            "validation.html": _standalone_html(markdown).encode("utf-8"),
        }
        for name, payload in payloads.items():
            _write_file(staging / name, payload)
        files = tuple(sorted(path for path in staging.iterdir() if path.is_file()))
        manifest = {
            "files": [
                {
                    "name": path.name,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
            "format": MAIN_VALIDATION_TREE_FORMAT,
            "schema_version": 1,
            "validation_sha256": validation_sha256,
        }
        _write_file(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, root)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_main_validation_artifact(
        root,
        artifact=artifact,
        preset=preset,
        selected_base=selected_base,
        preflight=preflight,
        prepared=prepared,
        stage_directories=stage_directories,
    )


def load_main_validation_artifact(
    directory: str | Path,
    *,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    selected_base: QuerySelectedBase,
    preflight: QueryGpuPreflight,
    prepared: PreparedQueryAdaptation,
    stage_directories: tuple[Path, ...],
) -> MainValidationArtifact:
    """Authenticate the complete validation freeze without opening sealed prompts."""
    _validate_sources(
        artifact,
        preset,
        selected_base,
        preflight,
        prepared,
        stage_directories,
    )
    root = Path(directory)
    manifest = _canonical_json(root / "manifest.json", "main validation manifest")
    if (
        set(manifest)
        != {"files", "format", "schema_version", "validation_sha256"}
        or manifest.get("format") != MAIN_VALIDATION_TREE_FORMAT
        or manifest.get("schema_version") != 1
        or manifest.get("validation_sha256") != root.name
    ):
        raise ValueError("main validation tree identity changed")
    _verify_files(root, manifest)
    record = _canonical_json(root / "validation.json", "main validation record")
    required = {
        "allocator_peak_bytes",
        "benchmark_id",
        "catalog_sha256",
        "config",
        "config_sha256",
        "format",
        "final_evaluation_protocol",
        "final_evaluation_protocol_sha256",
        "partition_sha256",
        "preflight_sha256",
        "preparation_sha256",
        "preflight_result_rows_recorded",
        "result_count",
        "result_size_limit_bytes",
        "results_sha256",
        "resume_verified",
        "runtime_seconds",
        "sealed_test_opened",
        "selected_base_sha256",
        "sealed_result_bytes_projected",
        "sealed_result_rows_projected",
        "stage_artifacts",
        "validation_sha256",
        "validation_summaries",
        "validation_result_rows_projected",
    }
    core = {key: value for key, value in record.items() if key != "validation_sha256"}
    if (
        set(record) != required
        or record.get("format") != MAIN_VALIDATION_FORMAT
        or record.get("benchmark_id") != BENCHMARK_ID
        or record.get("validation_sha256") != root.name
        or record_sha256(core) != root.name
        or record.get("sealed_test_opened") is not False
        or record.get("resume_verified") is not True
        or record.get("catalog_sha256") != artifact.catalog_sha256
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("selected_base_sha256") != selected_base.selection_sha256
        or record.get("preflight_sha256") != preflight.preflight_sha256
        or record.get("preparation_sha256") != prepared.preparation_sha256
        or record.get("config") != preset.as_record()
        or record.get("config_sha256") != preset.config_sha256
        or record.get("final_evaluation_protocol")
        != REGISTERED_FINAL_EVALUATION_PROTOCOL.as_record()
        or record.get("final_evaluation_protocol_sha256")
        != REGISTERED_FINAL_EVALUATION_PROTOCOL.protocol_sha256
        or record.get("stage_artifacts")
        != list(_stage_records(stage_directories, preset))
    ):
        raise ValueError("main validation source binding changed")
    validation_estimate = estimate_resources(preset, queries_per_world=36)
    sealed_estimate = estimate_resources(preset, queries_per_world=60)
    if (
        record.get("validation_result_rows_projected")
        != validation_estimate.result_rows
        or record.get("sealed_result_rows_projected")
        != sealed_estimate.result_rows
        or record.get("sealed_result_bytes_projected")
        != sealed_estimate.estimated_result_bytes
        or record.get("preflight_result_rows_recorded")
        != preflight.estimate.result_rows
        or record.get("result_size_limit_bytes")
        != preset.result_size_limit_bytes
        or sealed_estimate.estimated_result_bytes > preset.result_size_limit_bytes
    ):
        raise ValueError("main validation result-size accounting changed")
    results_sha256 = _text(record, "results_sha256")
    result_count = _integer(record, "result_count")
    ledger = root / "validation-results.jsonl"
    if (
        result_count != validation_estimate.result_rows
        or _file_sha256(ledger) != results_sha256
        or _canonical_jsonl_count(ledger) != result_count
        or ledger.stat().st_size > preset.result_size_limit_bytes
    ):
        raise ValueError("main validation result ledger changed")
    markdown = _render_validation_report(record)
    if (
        (root / "validation.md").read_text(encoding="utf-8") != markdown
        or (root / "validation.html").read_text(encoding="utf-8")
        != _standalone_html(markdown)
    ):
        raise ValueError("main validation rendered report changed")
    stages = record.get("stage_artifacts")
    runtime = record.get("runtime_seconds")
    assert type(stages) is list and stages and type(stages[-1]) is dict
    if type(runtime) is not dict or any(type(name) is not str for name in runtime):
        raise ValueError("main validation runtime mapping changed")
    return MainValidationArtifact(
        root=root.resolve(),
        validation_sha256=root.name,
        catalog_sha256=artifact.catalog_sha256,
        partition_sha256=artifact.partition_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        preflight_sha256=preflight.preflight_sha256,
        config_sha256=preset.config_sha256,
        preparation_sha256=prepared.preparation_sha256,
        final_adaptation_manifest_sha256=_text(
            stages[-1],
            "manifest_sha256",
        ),
        results_sha256=results_sha256,
        result_count=result_count,
        allocator_peak_bytes=_bounded_allocator_peak(record, preset),
        runtime_seconds=tuple(
            sorted((name, _number(runtime, name)) for name in runtime)
        ),
    )


def _validate_sources(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    selected_base: QuerySelectedBase,
    preflight: QueryGpuPreflight,
    prepared: PreparedQueryAdaptation,
    stage_directories: tuple[Path, ...],
) -> None:
    if (
        type(artifact) is not QueryPartitionArtifact
        or type(preset) is not QueryExperimentPreset
        or type(selected_base) is not QuerySelectedBase
        or type(preflight) is not QueryGpuPreflight
        or type(prepared) is not PreparedQueryAdaptation
        or artifact.catalog_sha256 != selected_base.catalog_sha256
        or artifact.partition_sha256 != selected_base.partition_sha256
        or artifact.catalog_sha256 != preflight.catalog_sha256
        or artifact.partition_sha256 != preflight.partition_sha256
        or preflight.config_sha256 != preset.config_sha256
        or prepared.catalog_sha256 != artifact.catalog_sha256
        or prepared.partition_sha256 != artifact.partition_sha256
        or prepared.config_sha256 != preset.config_sha256
        or prepared.concept_ids != preset.concept_ids
        or type(stage_directories) is not tuple
        or len(stage_directories) != preset.active_world_count
    ):
        raise ValueError("main validation sources do not share one frozen identity")


def _stage_records(
    stage_directories: tuple[Path, ...],
    preset: QueryExperimentPreset,
) -> tuple[dict[str, object], ...]:
    return tuple(
        _stage_record(Path(directory), stage, preset)
        for stage, directory in enumerate(stage_directories, start=1)
    )


def _stage_record(
    directory: Path,
    stage: int,
    preset: QueryExperimentPreset,
) -> dict[str, object]:
    loaded: LanguageAdaptationArtifact = load_language_adaptation_artifact(directory)
    expected_order = preset.concept_ids[:stage]
    if (
        tuple(str(task_id) for task_id in loaded.task_order) != expected_order
        or loaded.model_config != preset.model_config
        or loaded.lora_config != preset.lora_config
        or loaded.train_config != preset.adapter_train_config
        or loaded.max_nodes != preset.max_nodes
        or loaded.max_edges != preset.max_edges
    ):
        raise ValueError(f"main adaptation stage {stage} changed frozen settings")
    return {
        "manifest_sha256": _file_sha256(directory / "manifest.json"),
        "stage": stage,
        "task_order": list(expected_order),
        "tensor_checksum": loaded.tensor_checksum,
    }


def _validate_validation_results(
    results: tuple[SemanticQueryResult, ...],
    preset: QueryExperimentPreset,
) -> None:
    if (
        type(results) is not tuple
        or not results
        or any(result.split != "validation" for result in results)
        or {result.method for result in results} != set(SEMANTIC_QUERY_METHODS)
    ):
        raise ValueError("main validation rows have incomplete methods or wrong split")
    expected_cells = {
        (cell.stage, cell.concept_id) for cell in evaluation_schedule(preset)
    }
    for method in SEMANTIC_QUERY_METHODS:
        method_rows = tuple(row for row in results if row.method == method)
        primary = tuple(
            row
            for row in method_rows
            if row.adapter_concept_id in (None, row.concept_id)
        )
        expected = (
            {(0, concept_id) for concept_id in preset.concept_ids}
            if method == "base"
            else expected_cells
        )
        if {(row.stage, row.concept_id) for row in primary} != expected:
            raise ValueError(f"main validation schedule changed for {method}")
        if any(
            len(cell_rows) != 36
            or len({row.template_id for row in cell_rows}) != 36
            for cell in expected
            for cell_rows in (
                tuple(
                    row
                    for row in primary
                    if (row.stage, row.concept_id) == cell
                ),
            )
        ):
            raise ValueError(f"main validation cell coverage changed for {method}")
        if method == "independent":
            expected_independent = {
                (stage, query_concept, adapter_concept)
                for stage in range(1, preset.active_world_count + 1)
                for query_concept in preset.concept_ids[:stage]
                for adapter_concept in preset.concept_ids[:stage]
            }
            observed_independent = {
                (row.stage, row.concept_id, row.adapter_concept_id)
                for row in method_rows
            }
            if observed_independent != expected_independent or any(
                len(cell_rows) != 36
                or len({row.template_id for row in cell_rows}) != 36
                for cell in expected_independent
                for cell_rows in (
                    tuple(
                        row
                        for row in method_rows
                        if (row.stage, row.concept_id, row.adapter_concept_id)
                        == cell
                    ),
                )
            ):
                raise ValueError("main independent specificity coverage changed")
        elif len(method_rows) != len(primary):
            raise ValueError(f"main validation has unexpected adapter rows for {method}")
        if method in SEMANTIC_ROUTED_METHODS:
            if any(
                row.selected_node_index is None
                or row.oracle_node_index is None
                or row.routed_regret is None
                for row in method_rows
            ):
                raise ValueError(f"main validation routing evidence is missing for {method}")
        elif any(
            row.selected_node_index is not None
            or row.oracle_node_index is not None
            or row.routed_regret is not None
            for row in method_rows
        ):
            raise ValueError(f"main validation has unexpected routing evidence for {method}")


def _validation_summaries(
    results: tuple[SemanticQueryResult, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "accuracy": sum(row.answer_correct for row in rows) / len(rows),
            "mean_margin": sum(row.correct_answer_margin for row in rows) / len(rows),
            "method": method,
            "query_count": len(rows),
        }
        for method in SEMANTIC_QUERY_METHODS
        for rows in (
            tuple(
                row
                for row in results
                if row.method == method
                and row.adapter_concept_id in (None, row.concept_id)
            ),
        )
    )


def _results_sha256(results: tuple[SemanticQueryResult, ...]) -> str:
    digest = sha256()
    for result in results:
        digest.update(canonical_json_bytes(result.as_record()))
    return digest.hexdigest()


def _render_validation_report(record: dict[str, object]) -> str:
    summaries = record.get("validation_summaries")
    if type(summaries) is not list or any(type(item) is not dict for item in summaries):
        raise ValueError("main validation summaries changed")
    lines = [
        "# TinyWorlds-Q main validation freeze",
        "",
        f"Validation artifact: `{record['validation_sha256']}`",
        "",
        "This is an operational validation-only freeze, not the sealed result. "
        "It carries no scientific pass/fail verdict.",
        "",
        f"Result rows: {record['result_count']}",
        f"Projected sealed rows: {record['sealed_result_rows_projected']}",
        f"Projected sealed bytes: {record['sealed_result_bytes_projected']}",
        f"Frozen result-size limit: {record['result_size_limit_bytes']} bytes",
        f"Allocator peak: {record['allocator_peak_bytes']} bytes",
        "",
        "| Method | Query rows | Accuracy | Mean margin |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['method']} | {item['query_count']} | "
        f"{float(item['accuracy']):.6f} | {float(item['mean_margin']):.6f} |"
        for item in summaries
    )
    lines.extend(("", "The sealed test remained closed.", ""))
    return "\n".join(lines)


def _standalone_html(markdown: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q main validation</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f7f9;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body><pre>{html.escape(markdown)}</pre></body></html>\n"
    )


def _verify_files(root: Path, manifest: dict[str, object]) -> None:
    descriptors = manifest.get("files")
    if type(descriptors) is not list or any(type(item) is not dict for item in descriptors):
        raise ValueError("main validation file descriptors changed")
    expected = {"manifest.json"}
    for descriptor in descriptors:
        assert type(descriptor) is dict
        if set(descriptor) != {"name", "sha256", "size_bytes"}:
            raise ValueError("main validation file descriptor fields changed")
        name = descriptor.get("name")
        digest = descriptor.get("sha256")
        size = descriptor.get("size_bytes")
        if (
            type(name) is not str
            or Path(name).name != name
            or type(digest) is not str
            or type(size) is not int
        ):
            raise ValueError("main validation file descriptor is invalid")
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _file_sha256(path) != digest
        ):
            raise ValueError(f"main validation file changed: {name}")
        expected.add(name)
    if (
        not root.is_dir()
        or root.is_symlink()
        or {path.name for path in root.iterdir()} != expected
        or any(path.is_symlink() or not path.is_file() for path in root.iterdir())
    ):
        raise ValueError("main validation tree entries changed")


def _canonical_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("rb") as source:
        for line in source:
            record = json.loads(line)
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError("main validation ledger is noncanonical")
            count += 1
    return count


def _validation_candidate_matches(
    root: Path,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    selected_base: QuerySelectedBase,
    preflight: QueryGpuPreflight,
    prepared: PreparedQueryAdaptation,
    stage_records: list[dict[str, object]],
) -> bool:
    path = root / "validation.json"
    if not path.is_file():
        return False
    record = _canonical_json(path, "main validation candidate")
    return (
        record.get("catalog_sha256") == artifact.catalog_sha256
        and record.get("partition_sha256") == artifact.partition_sha256
        and record.get("selected_base_sha256") == selected_base.selection_sha256
        and record.get("preflight_sha256") == preflight.preflight_sha256
        and record.get("preparation_sha256") == prepared.preparation_sha256
        and record.get("config_sha256") == preset.config_sha256
        and record.get("stage_artifacts") == stage_records
    )


def _canonical_json(path: Path, label: str) -> dict[str, object]:
    payload = path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError(f"noncanonical {label}: {path}")
    return record


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    if root.exists():
        root.rmdir()


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str:
        raise ValueError(f"main validation {field} must be text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"main validation {field} must be an integer")
    return value


def _bounded_allocator_peak(
    record: dict[str, object],
    preset: QueryExperimentPreset,
) -> int:
    value = _integer(record, "allocator_peak_bytes")
    if value > preset.allocator_peak_limit_bytes:
        raise MemoryError("main validation allocator peak exceeds the frozen limit")
    return value


def _number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float) or not isfinite(float(value)):
        raise ValueError(f"main validation {field} must be finite numeric data")
    return float(value)


__all__ = [
    "MAIN_VALIDATION_FORMAT",
    "MainValidationArtifact",
    "find_main_validation_artifact",
    "load_main_validation_artifact",
    "publish_main_validation_artifact",
]
