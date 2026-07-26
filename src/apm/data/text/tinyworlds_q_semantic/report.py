"""Descriptive Markdown/HTML publication for TinyWorlds-Q experiments."""

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

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryResult,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import (
    SEMANTIC_QUERY_METHODS,
    SEMANTIC_ROUTED_METHODS,
)
from apm.data.text.tinyworlds_q_semantic.final_protocol import (
    REGISTERED_FINAL_EVALUATION_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.final_analysis import (
    compute_registered_final_effects,
)
from apm.data.text.tinyworlds_q_semantic.scaling import (
    estimate_resources,
    evaluation_schedule,
    render_schedule_report,
)
from apm.data.text.tinyworlds_q_semantic.statistics import (
    BootstrapEstimate,
    CANONICAL_BOOTSTRAP_REPLICATES,
    FactObservation,
    GenerationInspection,
    average_paraphrases,
    bootstrap_fact_metric,
)


REPORT_FORMAT = "tinyworlds-q-semantic-report-v1"


@dataclass(frozen=True, slots=True)
class SemanticReportArtifact:
    """One immutable descriptive result bundle with no scientific verdict."""

    root: Path
    report_sha256: str
    catalog_sha256: str
    partition_sha256: str
    config_sha256: str
    selected_base_sha256: str
    adapters_sha256: str
    preflight_sha256: str
    transaction_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.report_sha256, "semantic report"),
            (self.catalog_sha256, "report catalog"),
            (self.partition_sha256, "report partition"),
            (self.config_sha256, "report config"),
            (self.selected_base_sha256, "report selected base"),
            (self.adapters_sha256, "report adapters"),
            (self.preflight_sha256, "report preflight"),
            (self.transaction_sha256, "report sealed transaction"),
        ):
            require_sha256(value, label)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)


def publish_semantic_report(
    output_root: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    adapters_sha256: str,
    preflight_sha256: str,
    transaction_sha256: str,
    preset: QueryExperimentPreset,
    results: tuple[SemanticQueryResult, ...],
    effects: tuple[BootstrapEstimate, ...],
    generation: tuple[GenerationInspection, ...],
    runtime_seconds: Mapping[str, float],
    memory_bytes: Mapping[str, int],
    bootstrap_replicates: int = 10_000,
) -> SemanticReportArtifact:
    """Publish effect sizes, fact intervals, routing, resources, and generations."""
    for value, label in (
        (catalog_sha256, "report catalog"),
        (partition_sha256, "report partition"),
        (selected_base_sha256, "report selected base"),
        (adapters_sha256, "report adapters"),
        (preflight_sha256, "report preflight"),
        (transaction_sha256, "report sealed transaction"),
    ):
        require_sha256(value, label)
    if bootstrap_replicates != CANONICAL_BOOTSTRAP_REPLICATES:
        raise ValueError("final semantic reports require exactly 10,000 bootstraps")
    if any(
        effect.replicate_count != CANONICAL_BOOTSTRAP_REPLICATES
        for effect in effects
    ):
        raise ValueError("every final semantic effect requires 10,000 bootstraps")
    _validate_report_coverage(preset, results)
    if len(results) != estimate_resources(preset, queries_per_world=60).result_rows:
        raise ValueError("semantic report exact result-row count changed")
    registered_effects = compute_registered_final_effects(results, preset)
    if effects != registered_effects:
        raise ValueError("semantic report effects changed the frozen analysis")
    methods = {result.method for result in results}
    if methods != set(SEMANTIC_QUERY_METHODS):
        raise ValueError("semantic report methods changed the frozen method set")
    if tuple(item.concept_id for item in generation) != preset.concept_ids:
        raise ValueError("semantic generation inspection changed ordered worlds")
    if (
        not runtime_seconds
        or not memory_bytes
        or any(not isfinite(value) or value < 0.0 for value in runtime_seconds.values())
        or any(type(value) is not int or value < 0 for value in memory_bytes.values())
    ):
        raise ValueError("semantic report runtime and memory evidence is invalid")
    allocator_peak = memory_bytes.get("allocator_peak")
    if (
        type(allocator_peak) is not int
        or allocator_peak > preset.allocator_peak_limit_bytes
    ):
        raise MemoryError("semantic report allocator peak exceeds the frozen limit")
    observations = average_paraphrases(results)
    summaries = _condition_summaries(observations, bootstrap_replicates)
    results_digest = sha256()
    result_bytes = 0
    for result in results:
        payload = canonical_json_bytes(result.as_record())
        results_digest.update(payload)
        result_bytes += len(payload)
    if result_bytes > preset.result_size_limit_bytes:
        raise OSError("semantic result ledger exceeds the frozen size limit")
    results_sha256 = results_digest.hexdigest()
    report_record = {
        "bootstrap_replicates": bootstrap_replicates,
        "adapters_sha256": adapters_sha256,
        "catalog_sha256": catalog_sha256,
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "effects": [item.as_record() for item in effects],
        "final_evaluation_protocol": (
            REGISTERED_FINAL_EVALUATION_PROTOCOL.as_record()
        ),
        "final_evaluation_protocol_sha256": (
            REGISTERED_FINAL_EVALUATION_PROTOCOL.protocol_sha256
        ),
        "format": REPORT_FORMAT,
        "generation": [item.as_record() for item in generation],
        "memory_bytes": dict(sorted(memory_bytes.items())),
        "partition_sha256": partition_sha256,
        "preflight_sha256": preflight_sha256,
        "result_count": len(results),
        "result_bytes": result_bytes,
        "result_size_limit_bytes": preset.result_size_limit_bytes,
        "results_sha256": results_sha256,
        "runtime_seconds": dict(sorted(runtime_seconds.items())),
        "selected_base_sha256": selected_base_sha256,
        "summaries": [item.as_record() for item in summaries],
        "transaction_sha256": transaction_sha256,
    }
    report_sha256 = record_sha256(report_record)
    root = Path(output_root) / f"{preset.config_sha256[:16]}" / report_sha256
    if root.exists():
        _verify_existing(root, report_sha256)
        return SemanticReportArtifact(
            root.resolve(),
            report_sha256,
            catalog_sha256,
            partition_sha256,
            preset.config_sha256,
            selected_base_sha256,
            adapters_sha256,
            preflight_sha256,
            transaction_sha256,
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".semantic-report-", dir=root.parent))
    try:
        markdown = render_semantic_report(
            report_sha256,
            preset,
            summaries,
            effects,
            generation,
            runtime_seconds,
            memory_bytes,
        )
        payloads = {
            "result.json": canonical_json_bytes(
                {**report_record, "report_sha256": report_sha256}
            ),
            "report.md": markdown.encode("utf-8"),
            "report.html": _standalone_html(report_sha256, markdown).encode("utf-8"),
            "schedule.md": render_schedule_report(preset).encode("utf-8"),
        }
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        with (staging / "results.jsonl").open("wb") as ledger:
            written_digest = sha256()
            for result in results:
                payload = canonical_json_bytes(result.as_record())
                ledger.write(payload)
                written_digest.update(payload)
            ledger.flush()
            os.fsync(ledger.fileno())
        if written_digest.hexdigest() != results_sha256:
            raise AssertionError("semantic result ledger changed during publication")
        if (staging / "results.jsonl").stat().st_size != result_bytes:
            raise AssertionError("semantic result ledger size changed during publication")
        published_files = tuple(
            sorted(path for path in staging.iterdir() if path.is_file())
        )
        manifest = {
            "files": [
                {
                    "name": path.name,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in published_files
            ],
            "format": REPORT_FORMAT,
            "report_sha256": report_sha256,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, root)
    except BaseException:
        _remove_tree(staging)
        raise
    return SemanticReportArtifact(
        root.resolve(),
        report_sha256,
        catalog_sha256,
        partition_sha256,
        preset.config_sha256,
        selected_base_sha256,
        adapters_sha256,
        preflight_sha256,
        transaction_sha256,
    )


def find_semantic_report(
    output_root: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    adapters_sha256: str,
    preflight_sha256: str,
    transaction_sha256: str,
    preset: QueryExperimentPreset,
) -> SemanticReportArtifact | None:
    """Find the sole strict final report matching one sealed transaction."""
    root = Path(output_root) / preset.config_sha256[:16]
    candidates = (
        tuple(
            path
            for path in sorted(root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _report_candidate_matches(
                path,
                catalog_sha256,
                partition_sha256,
                selected_base_sha256,
                adapters_sha256,
                preflight_sha256,
                transaction_sha256,
                preset.config_sha256,
            )
        )
        if root.is_dir()
        else ()
    )
    if len(candidates) > 1:
        raise RuntimeError("multiple semantic reports bind one sealed transaction")
    return (
        load_semantic_report(
            candidates[0],
            catalog_sha256=catalog_sha256,
            partition_sha256=partition_sha256,
            selected_base_sha256=selected_base_sha256,
            adapters_sha256=adapters_sha256,
            preflight_sha256=preflight_sha256,
            transaction_sha256=transaction_sha256,
            preset=preset,
        )
        if candidates
        else None
    )


def load_semantic_report(
    directory: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    adapters_sha256: str,
    preflight_sha256: str,
    transaction_sha256: str,
    preset: QueryExperimentPreset,
) -> SemanticReportArtifact:
    """Strictly authenticate a final report and every frozen model binding."""
    root = Path(directory)
    try:
        _verify_existing(root, root.name)
    except FileExistsError as error:
        raise ValueError("semantic report tree authentication failed") from error
    result = _canonical_json(root / "result.json")
    expected = {
        "catalog_sha256": catalog_sha256,
        "partition_sha256": partition_sha256,
        "selected_base_sha256": selected_base_sha256,
        "adapters_sha256": adapters_sha256,
        "preflight_sha256": preflight_sha256,
        "transaction_sha256": transaction_sha256,
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
    }
    if any(result.get(field) != value for field, value in expected.items()):
        raise ValueError("semantic report frozen source binding changed")
    if _canonical_jsonl_count(root / "results.jsonl") != result.get("result_count"):
        raise ValueError("semantic report result count changed")
    if (
        (root / "results.jsonl").stat().st_size != result.get("result_bytes")
        or result.get("result_size_limit_bytes") != preset.result_size_limit_bytes
        or (root / "results.jsonl").stat().st_size > preset.result_size_limit_bytes
    ):
        raise ValueError("semantic report result-size accounting changed")
    return SemanticReportArtifact(
        root=root.resolve(),
        report_sha256=root.name,
        catalog_sha256=catalog_sha256,
        partition_sha256=partition_sha256,
        config_sha256=preset.config_sha256,
        selected_base_sha256=selected_base_sha256,
        adapters_sha256=adapters_sha256,
        preflight_sha256=preflight_sha256,
        transaction_sha256=transaction_sha256,
    )


def _report_candidate_matches(
    root: Path,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    adapters_sha256: str,
    preflight_sha256: str,
    transaction_sha256: str,
    config_sha256: str,
) -> bool:
    path = root / "result.json"
    if not path.is_file():
        return False
    result = _canonical_json(path)
    return (
        result.get("catalog_sha256") == catalog_sha256
        and result.get("partition_sha256") == partition_sha256
        and result.get("selected_base_sha256") == selected_base_sha256
        and result.get("adapters_sha256") == adapters_sha256
        and result.get("preflight_sha256") == preflight_sha256
        and result.get("transaction_sha256") == transaction_sha256
        and result.get("config_sha256") == config_sha256
    )


def _condition_summaries(
    observations: tuple[FactObservation, ...],
    replicates: int,
) -> tuple[BootstrapEstimate, ...]:
    primary = tuple(
        item
        for item in observations
        if (
            item.adapter_concept_id == item.concept_id
            if item.method == "independent"
            else item.adapter_concept_id is None
        )
    )
    keys = tuple(
        sorted(
            {
                (item.stage, item.method, item.split)
                for item in primary
            },
            key=lambda item: (item[0], item[1], item[2]),
        )
    )
    return tuple(
        estimate
        for key in keys
        for rows in (
            tuple(
                item
                for item in primary
                if (item.stage, item.method, item.split) == key
            ),
        )
        for metric in (
            "accuracy",
            "margin",
            *(
                ("router_accuracy",)
                if all(item.router_accuracy is not None for item in rows)
                else ()
            ),
            *(
                ("routed_regret",)
                if all(item.routed_regret is not None for item in rows)
                else ()
            ),
        )
        for raw_estimate in (
            bootstrap_fact_metric(
                rows,
                metric,  # type: ignore[arg-type]
                replicates=replicates,
                identity=f"stage-{key[0]}:{key[1]}:{key[2]}:primary",
            ),
        )
        for estimate in (
            BootstrapEstimate(
                metric=(
                    f"stage-{key[0]}:{key[1]}:{key[2]}:primary:{metric}"
                ),
                point=raw_estimate.point,
                lower=raw_estimate.lower,
                upper=raw_estimate.upper,
                replicate_count=raw_estimate.replicate_count,
                fact_count=raw_estimate.fact_count,
                world_count=raw_estimate.world_count,
            ),
        )
    )


def _validate_report_coverage(
    preset: QueryExperimentPreset,
    results: tuple[SemanticQueryResult, ...],
) -> None:
    if not results or any(result.split != "test" for result in results):
        raise ValueError("final semantic reports require sealed-test query rows")
    expected_cells = {
        (cell.stage, cell.concept_id) for cell in evaluation_schedule(preset)
    }
    for method in SEMANTIC_QUERY_METHODS:
        method_rows = tuple(row for row in results if row.method == method)
        primary_rows = tuple(
            row
            for row in method_rows
            if row.adapter_concept_id in (None, row.concept_id)
        )
        expected = (
            {(0, concept_id) for concept_id in preset.concept_ids}
            if method == "base"
            else expected_cells
        )
        observed = {(row.stage, row.concept_id) for row in primary_rows}
        if observed != expected:
            raise ValueError(f"semantic report schedule coverage changed for {method}")
        for cell in expected:
            cell_rows = tuple(
                row
                for row in primary_rows
                if (row.stage, row.concept_id) == cell
            )
            if len(cell_rows) != 60 or len(
                {row.template_id for row in cell_rows}
            ) != 60:
                raise ValueError(
                    f"semantic report cell {method}/{cell} must contain 60 queries"
                )
        if method == "independent":
            expected_specificity = {
                (stage, query_concept, adapter_concept)
                for stage in range(1, preset.active_world_count + 1)
                for query_concept in preset.concept_ids[:stage]
                for adapter_concept in preset.concept_ids[:stage]
            }
            if {
                (row.stage, row.concept_id, row.adapter_concept_id)
                for row in method_rows
            } != expected_specificity or any(
                len(cell_rows) != 60
                or len({row.template_id for row in cell_rows}) != 60
                for cell in expected_specificity
                for cell_rows in (
                    tuple(
                        row
                        for row in method_rows
                        if (row.stage, row.concept_id, row.adapter_concept_id)
                        == cell
                    ),
                )
            ):
                raise ValueError("semantic report independent specificity changed")
        elif len(method_rows) != len(primary_rows):
            raise ValueError(f"semantic report has unexpected adapter rows for {method}")
        if method in SEMANTIC_ROUTED_METHODS:
            if any(
                row.selected_node_index is None
                or row.oracle_node_index is None
                or row.routed_regret is None
                for row in method_rows
            ):
                raise ValueError(f"semantic report routing evidence is missing for {method}")
        elif any(
            row.selected_node_index is not None
            or row.oracle_node_index is not None
            or row.routed_regret is not None
            for row in method_rows
        ):
            raise ValueError(f"semantic report has unexpected routing evidence for {method}")


def render_semantic_report(
    report_sha256: str,
    preset: QueryExperimentPreset,
    summaries: tuple[BootstrapEstimate, ...],
    effects: tuple[BootstrapEstimate, ...],
    generation: tuple[GenerationInspection, ...],
    runtime_seconds: Mapping[str, float],
    memory_bytes: Mapping[str, int],
) -> str:
    """Render descriptive evidence without adding a scientific pass/fail label."""
    lines = [
        "# TinyWorlds-Q semantic result",
        "",
        f"Report: `{report_sha256}`",
        "",
        "This result is descriptive evidence. It has no VAMP scientific pass/fail verdict.",
        "",
        f"Ordered worlds: {', '.join(preset.concept_ids)}",
        f"Dynamic graph capacity: {preset.max_nodes} nodes / {preset.max_edges} edges",
        "",
        "## Fact-level estimates",
        "",
        "Templates were averaged within facts before aggregation; worlds receive equal weight.",
        "",
        "| Metric | Estimate | 95% interval | Facts | Worlds |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item.metric} | {item.point:.6f} | [{item.lower:.6f}, {item.upper:.6f}] | "
        f"{item.fact_count} | {item.world_count} |"
        for item in summaries
    )
    lines.extend(("", "## Acquisition, specificity, and retention effects", ""))
    lines.extend(
        f"- {item.metric}: {item.point:.6f} "
        f"(95% fact-resampled interval [{item.lower:.6f}, {item.upper:.6f}])"
        for item in effects
    )
    lines.extend(("", "## Runtime and memory", ""))
    lines.extend(f"- {name}: {value:.6f} seconds" for name, value in sorted(runtime_seconds.items()))
    lines.extend(f"- {name}: {value} bytes" for name, value in sorted(memory_bytes.items()))
    lines.extend(
        (
            "",
            "## Secondary exact-trigger generation inspection",
            "",
            "Each world uses its matching final independent adapter and the "
            "frozen greedy-generation budget.",
            "",
        )
    )
    for item in generation:
        lines.extend(
            (
                f"### {item.concept_id}",
                "",
                f"Prompt: {item.prompt}",
                "",
                f"Output: {item.output}",
                "",
                f"Registered-trigger recall: {item.recall:.6f}; facts: "
                f"{', '.join(item.recalled_fact_ids) if item.recalled_fact_ids else 'none'}",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _verify_existing(root: Path, report_sha256: str) -> None:
    try:
        manifest = _canonical_json(root / "manifest.json")
        if (
            set(manifest) != {"files", "format", "report_sha256"}
            or manifest.get("format") != REPORT_FORMAT
            or manifest.get("report_sha256") != report_sha256
            or root.name != report_sha256
        ):
            raise ValueError("semantic report manifest identity changed")
        raw_files = manifest.get("files")
        if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
            raise ValueError("semantic report file descriptors changed")
        expected_names = {"manifest.json"}
        for item in raw_files:
            if set(item) != {"name", "sha256", "size_bytes"}:
                raise ValueError("semantic report file descriptor fields changed")
            name = item.get("name")
            size = item.get("size_bytes")
            digest = item.get("sha256")
            if (
                type(name) is not str
                or Path(name).name != name
                or type(size) is not int
                or size < 0
                or type(digest) is not str
            ):
                raise ValueError("semantic report file descriptor is invalid")
            path = root / name
            if path.stat().st_size != size or _file_sha256(path) != digest:
                raise ValueError(f"semantic report file changed: {name}")
            expected_names.add(name)
        if (
            {path.name for path in root.iterdir() if path.is_file()} != expected_names
            or any(path.is_dir() or path.is_symlink() for path in root.iterdir())
        ):
            raise ValueError("semantic report tree entries changed")
        result_record = _canonical_json(root / "result.json")
        if result_record.get("report_sha256") != report_sha256:
            raise ValueError("semantic report result identity changed")
        if (
            result_record.get("final_evaluation_protocol")
            != REGISTERED_FINAL_EVALUATION_PROTOCOL.as_record()
            or result_record.get("final_evaluation_protocol_sha256")
            != REGISTERED_FINAL_EVALUATION_PROTOCOL.protocol_sha256
        ):
            raise ValueError("semantic report final protocol changed")
        core = {
            key: value
            for key, value in result_record.items()
            if key != "report_sha256"
        }
        if record_sha256(core) != report_sha256:
            raise ValueError("semantic report content identity changed")
        if _file_sha256(root / "results.jsonl") != result_record.get(
            "results_sha256"
        ):
            raise ValueError("semantic report result ledger identity changed")
    except (OSError, ValueError) as error:
        raise FileExistsError(
            "existing semantic report does not match requested identity"
        ) from error


def _canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic report JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical semantic report JSON: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("rb") as source:
        for line in source:
            record = json.loads(line)
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError("semantic report result ledger is noncanonical")
            count += 1
    return count


def _standalone_html(report_sha256: str, markdown: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q semantic result</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;background:#f5f7f9;"
        "padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-report-sha256=\"{report_sha256}\"><pre>"
        f"{html.escape(markdown)}</pre></body></html>\n"
    )


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


__all__ = [
    "REPORT_FORMAT",
    "SemanticReportArtifact",
    "find_semantic_report",
    "load_semantic_report",
    "publish_semantic_report",
    "render_semantic_report",
]
