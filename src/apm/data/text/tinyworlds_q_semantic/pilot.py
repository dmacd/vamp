"""Immutable pilot learnability decision and validation-only publication."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from apm.data.text.tinyworlds_q_semantic.evaluation import PilotBudgetEvaluation
from apm.data.text.tinyworlds_q_semantic.execution import select_pilot_budget
from apm.data.text.tinyworlds_q_semantic.report import REQUIRED_QUERY_METHODS
from apm.data.text.tinyworlds_q_semantic.scaling import evaluation_schedule


PILOT_RESULT_FORMAT = "tinyworlds-q-semantic-pilot-result-v1"


@dataclass(frozen=True, slots=True)
class SemanticPilotResult:
    """One strict validation-only pilot decision and its published directory."""

    root: Path
    pilot_sha256: str
    selected_updates: int
    selected_config_sha256: str
    sealed_test_opened: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.pilot_sha256, "pilot result"),
            (self.selected_config_sha256, "selected pilot config"),
        ):
            require_sha256(value, label)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if type(self.selected_updates) is not int or self.selected_updates <= 0:
            raise ValueError("selected pilot updates must be positive")
        if self.sealed_test_opened is not False:
            raise ValueError("pilot publication cannot open the sealed test")


def publish_semantic_pilot_result(
    output_root: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    preflight_sha256: str,
    preset: QueryExperimentPreset,
    budgets: tuple[PilotBudgetEvaluation, ...],
    independent_sweep_sha256: str,
    independent_sweep_manifest_sha256: str,
    selected_adaptation_manifest_sha256: str,
    selected_validation_results: tuple[SemanticQueryResult, ...],
    resume_verified: bool,
    runtime_seconds: float,
    allocator_peak_bytes: int,
) -> SemanticPilotResult:
    """Publish all three budget outcomes and the selected two-world exercise."""
    for value, label in (
        (catalog_sha256, "pilot catalog"),
        (partition_sha256, "pilot partition"),
        (selected_base_sha256, "pilot selected base"),
        (preflight_sha256, "pilot preflight"),
    ):
        require_sha256(value, label)
    expected_budgets = preset.pilot_update_budgets
    if tuple(item.budget.updates for item in budgets) != expected_budgets:
        raise ValueError("pilot evaluations must follow all registered budgets")
    for value, label in (
        (independent_sweep_sha256, "pilot independent sweep"),
        (independent_sweep_manifest_sha256, "pilot independent sweep manifest"),
        (selected_adaptation_manifest_sha256, "selected pilot adaptation manifest"),
    ):
        require_sha256(value, label)
    selected_updates = select_pilot_budget(
        tuple(item.budget for item in budgets),
        preset,
    )
    selected_config = replace(preset, adapter_updates=selected_updates)
    if (
        type(resume_verified) is not bool
        or not resume_verified
        or type(runtime_seconds) not in (int, float)
        or not isfinite(runtime_seconds)
        or runtime_seconds < 0.0
        or type(allocator_peak_bytes) is not int
        or not 0 <= allocator_peak_bytes <= preset.allocator_peak_limit_bytes
    ):
        raise ValueError("pilot operational evidence is incomplete")
    if (
        not selected_validation_results
        or any(row.split != "validation" for row in selected_validation_results)
        or set(row.method for row in selected_validation_results)
        != set(REQUIRED_QUERY_METHODS)
    ):
        raise ValueError("selected pilot exercise lacks required validation methods")
    _validate_selected_validation(selected_validation_results, preset)
    budget_ledgers = tuple(
        (
            item.budget.updates,
            _result_ledger(item.results),
        )
        for item in budgets
    )
    selected_ledger = _result_ledger(selected_validation_results)
    budget_records = tuple(
        {
            "adaptation_tensor_checksum": item.adaptation_tensor_checksum,
            "base_accuracy": [list(value) for value in item.budget.base_accuracy],
            "concept_accuracy": [
                list(value) for value in item.budget.concept_accuracy
            ],
            "config_sha256": replace(
                preset,
                adapter_updates=item.budget.updates,
            ).config_sha256,
            "passed": item.budget.passes,
            "results_sha256": sha256(dict(budget_ledgers)[item.budget.updates]).hexdigest(),
            "updates": item.budget.updates,
        }
        for item in budgets
    )
    core = {
        "allocator_peak_bytes": allocator_peak_bytes,
        "budgets": list(budget_records),
        "catalog_sha256": catalog_sha256,
        "format": PILOT_RESULT_FORMAT,
        "independent_sweep_manifest_sha256": independent_sweep_manifest_sha256,
        "independent_sweep_sha256": independent_sweep_sha256,
        "partition_sha256": partition_sha256,
        "preflight_sha256": preflight_sha256,
        "resume_verified": resume_verified,
        "runtime_seconds": float(runtime_seconds),
        "sealed_test_opened": False,
        "selected_base_sha256": selected_base_sha256,
        "selected_adaptation_manifest_sha256": selected_adaptation_manifest_sha256,
        "selected_config": selected_config.as_record(),
        "selected_config_sha256": selected_config.config_sha256,
        "selected_results_sha256": sha256(selected_ledger).hexdigest(),
        "selected_updates": selected_updates,
    }
    pilot_sha256 = record_sha256(core)
    destination = Path(output_root) / pilot_sha256
    if destination.exists():
        return load_semantic_pilot_result(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pilot-result-", dir=destination.parent))
    try:
        markdown = render_semantic_pilot_report(
            pilot_sha256,
            budget_records,
            selected_updates,
            selected_config.config_sha256,
            resume_verified,
            float(runtime_seconds),
            allocator_peak_bytes,
        )
        payloads = {
            "pilot.json": canonical_json_bytes(
                {**core, "pilot_sha256": pilot_sha256}
            ),
            "report.md": markdown.encode("utf-8"),
            "report.html": (
                "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>TinyWorlds-Q pilot result</title>"
                "<style>body{font:15px/1.5 system-ui;max-width:1000px;margin:2rem auto;"
                "padding:0 1rem}pre{white-space:pre-wrap;background:#f5f7f9;"
                "padding:1.25rem;border-radius:8px}</style></head>"
                f"<body data-pilot-sha256=\"{pilot_sha256}\"><pre>"
                f"{html.escape(markdown)}</pre></body></html>\n"
            ).encode("utf-8"),
            "selected-validation.jsonl": selected_ledger,
            **{
                f"budget-{updates:04d}-validation.jsonl": payload
                for updates, payload in budget_ledgers
            },
        }
        for name, payload in payloads.items():
            _write_file(staging / name, payload)
        manifest = {
            "files": [
                {
                    "name": name,
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(payloads.items())
            ],
            "format": PILOT_RESULT_FORMAT,
            "pilot_sha256": pilot_sha256,
        }
        _write_file(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, destination)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_semantic_pilot_result(destination)


def load_semantic_pilot_result(root: str | Path) -> SemanticPilotResult:
    """Authenticate one complete pilot publication without reading sealed data."""
    directory = Path(root)
    manifest = _canonical_json(directory / "manifest.json")
    pilot = _canonical_json(directory / "pilot.json")
    if (
        set(manifest) != {"files", "format", "pilot_sha256"}
        or manifest.get("format") != PILOT_RESULT_FORMAT
        or pilot.get("format") != PILOT_RESULT_FORMAT
        or pilot.get("pilot_sha256") != manifest.get("pilot_sha256")
        or directory.name != manifest.get("pilot_sha256")
    ):
        raise ValueError("pilot result identity changed")
    pilot_sha256 = str(manifest["pilot_sha256"])
    core = {key: value for key, value in pilot.items() if key != "pilot_sha256"}
    if record_sha256(core) != pilot_sha256:
        raise ValueError("pilot result content hash changed")
    raw_files = manifest.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise ValueError("pilot result file descriptors changed")
    expected_names = {"manifest.json"}
    payload_by_name = {}
    for descriptor in raw_files:
        if set(descriptor) != {"name", "sha256", "size_bytes"}:
            raise ValueError("pilot result file descriptor changed")
        name = descriptor["name"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("pilot result filename changed")
        payload = (directory / name).read_bytes()
        if (
            len(payload) != descriptor["size_bytes"]
            or sha256(payload).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"pilot result file changed: {name}")
        payload_by_name[name] = payload
        expected_names.add(name)
    if (
        {path.name for path in directory.iterdir()} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in directory.iterdir())
    ):
        raise ValueError("pilot result tree entries changed")
    if pilot.get("sealed_test_opened") is not False:
        raise PermissionError("pilot result claims sealed-test access")
    if sha256(payload_by_name["selected-validation.jsonl"]).hexdigest() != pilot.get(
        "selected_results_sha256"
    ):
        raise ValueError("selected pilot result ledger identity changed")
    raw_budgets = pilot.get("budgets")
    if type(raw_budgets) is not list or any(
        type(record) is not dict for record in raw_budgets
    ):
        raise ValueError("pilot budget records changed")
    for budget in raw_budgets:
        updates = budget.get("updates")
        if type(updates) is not int:
            raise ValueError("pilot budget update count changed")
        ledger_name = f"budget-{updates:04d}-validation.jsonl"
        if (
            ledger_name not in payload_by_name
            or sha256(payload_by_name[ledger_name]).hexdigest()
            != budget.get("results_sha256")
        ):
            raise ValueError("pilot budget result ledger identity changed")
    return SemanticPilotResult(
        root=directory.resolve(),
        pilot_sha256=pilot_sha256,
        selected_updates=int(pilot["selected_updates"]),
        selected_config_sha256=str(pilot["selected_config_sha256"]),
        sealed_test_opened=False,
    )


def render_semantic_pilot_report(
    pilot_sha256: str,
    budgets: tuple[dict[str, object], ...],
    selected_updates: int,
    selected_config_sha256: str,
    resume_verified: bool,
    runtime_seconds: float,
    allocator_peak_bytes: int,
) -> str:
    """Render the mandatory learnability gate without a VAMP verdict."""
    first_base = budgets[0]["base_accuracy"]
    if type(first_base) is not list:
        raise ValueError("pilot report base accuracies changed")
    concept_ids = tuple(str(value[0]) for value in first_base)
    accuracy_headers = tuple(
        heading
        for concept_id in concept_ids
        for heading in (f"{concept_id.title()} base", f"{concept_id.title()} adapter")
    )
    lines = [
        "# TinyWorlds-Q semantic pilot result",
        "",
        f"Pilot: `{pilot_sha256}`",
        "",
        "The sealed test was not opened. This pilot applies only the mandatory "
        "learnability and operational gates; it does not assign a VAMP verdict.",
        "",
        "| " + " | ".join(("Updates", *accuracy_headers, "Passed")) + " |",
        "| " + " | ".join(("---:", *("---:" for _ in accuracy_headers), ":---:")) + " |",
    ]
    for budget in budgets:
        base = dict(budget["base_accuracy"])  # type: ignore[arg-type]
        adapted = dict(budget["concept_accuracy"])  # type: ignore[arg-type]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(budget["updates"]),
                    *(
                        value
                        for concept_id in concept_ids
                        for value in (
                            f"{base[concept_id]:.6f}",
                            f"{adapted[concept_id]:.6f}",
                        )
                    ),
                    str(budget["passed"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            f"Selected update budget: **{selected_updates:,}**",
            f"Selected config: `{selected_config_sha256}`",
            f"Exact completed-stage resume verified: `{resume_verified}`",
            f"Runtime: `{runtime_seconds:.6f}` seconds",
            f"Allocator peak: `{allocator_peak_bytes}` bytes",
            "",
        )
    )
    return "\n".join(lines)


def _result_ledger(results: tuple[SemanticQueryResult, ...]) -> bytes:
    if not results:
        raise ValueError("pilot result ledger cannot be empty")
    return b"".join(canonical_json_bytes(row.as_record()) for row in results)


def _validate_selected_validation(
    results: tuple[SemanticQueryResult, ...],
    preset: QueryExperimentPreset,
) -> None:
    expected_stage_cells = {
        (cell.stage, cell.concept_id) for cell in evaluation_schedule(preset)
    }
    for method in REQUIRED_QUERY_METHODS:
        method_rows = tuple(row for row in results if row.method == method)
        primary = tuple(
            row
            for row in method_rows
            if row.adapter_concept_id in (None, row.concept_id)
        )
        expected = (
            {(0, concept_id) for concept_id in preset.concept_ids}
            if method == "base"
            else expected_stage_cells
        )
        if {(row.stage, row.concept_id) for row in primary} != expected:
            raise ValueError(f"pilot validation schedule changed for {method}")
        for cell in expected:
            cell_rows = tuple(
                row
                for row in primary
                if (row.stage, row.concept_id) == cell
            )
            if len(cell_rows) != 36 or len(
                {row.template_id for row in cell_rows}
            ) != 36:
                raise ValueError(
                    f"pilot validation cell {method}/{cell} must contain 36 queries"
                )


def _canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid pilot JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical pilot JSON: {path.name}")
    return value


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


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
    "PILOT_RESULT_FORMAT",
    "SemanticPilotResult",
    "load_semantic_pilot_result",
    "publish_semantic_pilot_result",
    "render_semantic_pilot_report",
]
