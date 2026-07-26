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
from apm.data.text.tinyworlds_q_semantic.execution import (
    AMENDED_PILOT_LEARNABILITY_POLICY,
    ORIGINAL_PILOT_LEARNABILITY_POLICY,
    PilotBudgetResult,
    PilotLearnabilityPolicy,
    pilot_budget_passes,
    select_pilot_budget,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import SEMANTIC_QUERY_METHODS
from apm.data.text.tinyworlds_q_semantic.scaling import evaluation_schedule


PILOT_RESULT_FORMAT = "tinyworlds-q-semantic-pilot-result-v1"
PILOT_FAILURE_FORMAT = "tinyworlds-q-semantic-pilot-failure-v1"


@dataclass(frozen=True, slots=True)
class SemanticPilotResult:
    """One strict validation-only pilot decision and its published directory."""

    root: Path
    pilot_sha256: str
    selected_updates: int
    selected_config_sha256: str
    learnability_policy_sha256: str
    protocol_amendment_sha256: str | None
    sealed_test_opened: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.pilot_sha256, "pilot result"),
            (self.selected_config_sha256, "selected pilot config"),
            (self.learnability_policy_sha256, "pilot learnability policy"),
        ):
            require_sha256(value, label)
        if self.protocol_amendment_sha256 is not None:
            require_sha256(
                self.protocol_amendment_sha256,
                "pilot protocol amendment",
            )
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if type(self.selected_updates) is not int or self.selected_updates <= 0:
            raise ValueError("selected pilot updates must be positive")
        if self.sealed_test_opened is not False:
            raise ValueError("pilot publication cannot open the sealed test")


@dataclass(frozen=True, slots=True)
class SemanticPilotFailure:
    """One strict validation-only pilot stop and its published directory."""

    root: Path
    failure_sha256: str
    reason: str
    sealed_test_opened: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.failure_sha256, "pilot failure")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if self.reason != "pilot_learnability_gate_failed":
            raise ValueError("pilot failure reason changed")
        if self.sealed_test_opened is not False:
            raise ValueError("pilot failure publication cannot open the sealed test")


def publish_semantic_pilot_result(
    output_root: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    preflight_sha256: str,
    preset: QueryExperimentPreset,
    learnability_policy: PilotLearnabilityPolicy,
    protocol_amendment_sha256: str | None,
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
    _validate_pilot_authorization(
        learnability_policy,
        protocol_amendment_sha256,
    )
    selected_updates = select_pilot_budget(
        tuple(item.budget for item in budgets),
        preset,
        learnability_policy,
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
        != set(SEMANTIC_QUERY_METHODS)
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
        _budget_record(
            item,
            dict(budget_ledgers)[item.budget.updates],
            preset,
            learnability_policy,
        )
        for item in budgets
    )
    core = {
        "allocator_peak_bytes": allocator_peak_bytes,
        "budgets": list(budget_records),
        "catalog_sha256": catalog_sha256,
        "format": PILOT_RESULT_FORMAT,
        "independent_sweep_manifest_sha256": independent_sweep_manifest_sha256,
        "independent_sweep_sha256": independent_sweep_sha256,
        "learnability_policy": learnability_policy.as_record(),
        "learnability_policy_sha256": learnability_policy.policy_sha256,
        "partition_sha256": partition_sha256,
        "preflight_sha256": preflight_sha256,
        "protocol_amendment_sha256": protocol_amendment_sha256,
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
            learnability_policy,
            protocol_amendment_sha256,
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


def publish_semantic_pilot_failure(
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
    allocator_peak_bytes: int,
) -> SemanticPilotFailure:
    """Publish exact all-budget evidence when no pilot budget passes."""
    for value, label in (
        (catalog_sha256, "pilot failure catalog"),
        (partition_sha256, "pilot failure partition"),
        (selected_base_sha256, "pilot failure selected base"),
        (preflight_sha256, "pilot failure preflight"),
        (independent_sweep_sha256, "pilot failure independent sweep"),
        (
            independent_sweep_manifest_sha256,
            "pilot failure independent sweep manifest",
        ),
    ):
        require_sha256(value, label)
    if (
        tuple(item.budget.updates for item in budgets)
        != preset.pilot_update_budgets
        or any(
            pilot_budget_passes(
                item.budget,
                ORIGINAL_PILOT_LEARNABILITY_POLICY,
            )
            for item in budgets
        )
    ):
        raise ValueError("pilot failure requires every registered budget to fail")
    if (
        type(allocator_peak_bytes) is not int
        or not 0 <= allocator_peak_bytes <= preset.allocator_peak_limit_bytes
    ):
        raise ValueError("pilot failure allocator evidence is invalid")
    budget_ledgers = tuple(
        (item.budget.updates, _result_ledger(item.results)) for item in budgets
    )
    budget_records = tuple(
        _budget_record(
            item,
            dict(budget_ledgers)[item.budget.updates],
            preset,
            ORIGINAL_PILOT_LEARNABILITY_POLICY,
        )
        for item in budgets
    )
    core = {
        "allocator_peak_bytes": allocator_peak_bytes,
        "budgets": list(budget_records),
        "catalog_sha256": catalog_sha256,
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "format": PILOT_FAILURE_FORMAT,
        "independent_sweep_manifest_sha256": independent_sweep_manifest_sha256,
        "independent_sweep_sha256": independent_sweep_sha256,
        "partition_sha256": partition_sha256,
        "preflight_sha256": preflight_sha256,
        "reason": "pilot_learnability_gate_failed",
        "sealed_test_opened": False,
        "selected_base_sha256": selected_base_sha256,
    }
    failure_sha256 = record_sha256(core)
    destination = Path(output_root) / failure_sha256
    if destination.exists():
        return load_semantic_pilot_failure(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pilot-failure-", dir=destination.parent))
    try:
        markdown = render_semantic_pilot_failure_report(
            failure_sha256,
            budget_records,
            allocator_peak_bytes,
        )
        payloads = {
            "failure.json": canonical_json_bytes(
                {**core, "failure_sha256": failure_sha256}
            ),
            "report.md": markdown.encode("utf-8"),
            "report.html": (
                "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<title>TinyWorlds-Q pilot failure</title>"
                "<style>body{font:15px/1.5 system-ui;max-width:1000px;margin:2rem auto;"
                "padding:0 1rem}pre{white-space:pre-wrap;background:#f5f7f9;"
                "padding:1.25rem;border-radius:8px}</style></head>"
                f"<body data-failure-sha256=\"{failure_sha256}\"><pre>"
                f"{html.escape(markdown)}</pre></body></html>\n"
            ).encode("utf-8"),
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
            "format": PILOT_FAILURE_FORMAT,
            "failure_sha256": failure_sha256,
        }
        _write_file(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, destination)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_semantic_pilot_failure(destination)


def load_semantic_pilot_failure(root: str | Path) -> SemanticPilotFailure:
    """Authenticate one complete pilot stop without reading sealed data."""
    directory = Path(root)
    manifest = _canonical_json(directory / "manifest.json")
    failure = _canonical_json(directory / "failure.json")
    required = {
        "allocator_peak_bytes",
        "budgets",
        "catalog_sha256",
        "config",
        "config_sha256",
        "failure_sha256",
        "format",
        "independent_sweep_manifest_sha256",
        "independent_sweep_sha256",
        "partition_sha256",
        "preflight_sha256",
        "reason",
        "sealed_test_opened",
        "selected_base_sha256",
    }
    if (
        set(manifest) != {"files", "format", "failure_sha256"}
        or set(failure) != required
        or manifest.get("format") != PILOT_FAILURE_FORMAT
        or failure.get("format") != PILOT_FAILURE_FORMAT
        or failure.get("failure_sha256") != manifest.get("failure_sha256")
        or directory.name != manifest.get("failure_sha256")
    ):
        raise ValueError("pilot failure identity changed")
    failure_sha256 = str(manifest["failure_sha256"])
    core = {key: value for key, value in failure.items() if key != "failure_sha256"}
    if record_sha256(core) != failure_sha256:
        raise ValueError("pilot failure content hash changed")
    payload_by_name = _verify_publication_files(directory, manifest)
    if failure.get("sealed_test_opened") is not False:
        raise PermissionError("pilot failure claims sealed-test access")
    budgets = failure.get("budgets")
    if type(budgets) is not list or not budgets or any(
        type(record) is not dict for record in budgets
    ):
        raise ValueError("pilot failure budget records changed")
    for budget in budgets:
        updates = budget.get("updates")
        name = f"budget-{updates:04d}-validation.jsonl" if type(updates) is int else ""
        if (
            set(budget)
            != {
                "acquisition",
                "adaptation_tensor_checksum",
                "base_accuracy",
                "concept_accuracy",
                "config_sha256",
                "passed",
                "results_sha256",
                "updates",
            }
            or not name
            or name not in payload_by_name
            or sha256(payload_by_name[name]).hexdigest()
            != budget.get("results_sha256")
            or budget.get("passed") is not False
        ):
            raise ValueError("pilot failure budget ledger changed")
    return SemanticPilotFailure(
        root=directory.resolve(),
        failure_sha256=failure_sha256,
        reason=str(failure["reason"]),
        sealed_test_opened=False,
    )


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
    payload_by_name = _verify_publication_files(directory, manifest)
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
    policy = _policy_from_result_record(pilot)
    amendment = pilot.get("protocol_amendment_sha256")
    if amendment is not None and type(amendment) is not str:
        raise ValueError("pilot protocol amendment identity changed")
    _validate_pilot_authorization(policy, amendment)
    budget_results = tuple(_pilot_budget_from_record(record) for record in raw_budgets)
    selected_updates = int(pilot["selected_updates"])
    if selected_updates != select_pilot_budget(
        budget_results,
        _preset_from_selected_result(pilot, budget_results),
        policy,
    ):
        raise ValueError("pilot selected budget changed")
    return SemanticPilotResult(
        root=directory.resolve(),
        pilot_sha256=pilot_sha256,
        selected_updates=selected_updates,
        selected_config_sha256=str(pilot["selected_config_sha256"]),
        learnability_policy_sha256=policy.policy_sha256,
        protocol_amendment_sha256=amendment,
        sealed_test_opened=False,
    )


def render_semantic_pilot_report(
    pilot_sha256: str,
    budgets: tuple[dict[str, object], ...],
    selected_updates: int,
    selected_config_sha256: str,
    learnability_policy: PilotLearnabilityPolicy,
    protocol_amendment_sha256: str | None,
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
        for heading in (
            f"{concept_id.title()} base",
            f"{concept_id.title()} adapter",
            f"{concept_id.title()} gain",
        )
    )
    lines = [
        "# TinyWorlds-Q semantic pilot result",
        "",
        f"Pilot: `{pilot_sha256}`",
        "",
        "The sealed test was not opened. This pilot applies only the mandatory "
        "learnability and operational gates; it does not assign a VAMP verdict.",
        "",
        f"Learnability policy: `{learnability_policy.policy_sha256}` "
        f"(`{learnability_policy.policy_id}`)",
        *(
            ()
            if protocol_amendment_sha256 is None
            else (f"Protocol amendment: `{protocol_amendment_sha256}`",)
        ),
        *(
            ("Acquisition remains descriptive and is not an authorization gate.",)
            if learnability_policy.acquisition_role == "descriptive"
            else ()
        ),
        "",
        "| " + " | ".join(("Updates", *accuracy_headers, "Passed")) + " |",
        "| " + " | ".join(("---:", *("---:" for _ in accuracy_headers), ":---:")) + " |",
    ]
    for budget in budgets:
        base = dict(budget["base_accuracy"])  # type: ignore[arg-type]
        adapted = dict(budget["concept_accuracy"])  # type: ignore[arg-type]
        acquisition = dict(budget["acquisition"])  # type: ignore[arg-type]
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
                            f"{acquisition[concept_id]:+.6f}",
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


def render_semantic_pilot_failure_report(
    failure_sha256: str,
    budgets: tuple[dict[str, object], ...],
    allocator_peak_bytes: int,
) -> str:
    """Render the mandatory all-budget pilot stop with acquisition deltas."""
    first_base = budgets[0]["base_accuracy"]
    if type(first_base) is not list:
        raise ValueError("pilot failure base accuracies changed")
    concept_ids = tuple(str(value[0]) for value in first_base)
    headers = tuple(
        heading
        for concept_id in concept_ids
        for heading in (
            f"{concept_id.title()} base",
            f"{concept_id.title()} adapter",
            f"{concept_id.title()} gain",
        )
    )
    lines = [
        "# TinyWorlds-Q semantic pilot learnability stop",
        "",
        f"Failure: `{failure_sha256}`",
        "",
        "The sealed test was not opened. No registered update budget passed both "
        "the 60% accuracy and 15-percentage-point acquisition gates for both worlds. "
        "Sequential/VAMP and main execution therefore remained unauthorized.",
        "",
        "| " + " | ".join(("Updates", *headers, "Passed")) + " |",
        "| " + " | ".join(("---:", *("---:" for _ in headers), ":---:")) + " |",
    ]
    for budget in budgets:
        base = dict(budget["base_accuracy"])  # type: ignore[arg-type]
        adapted = dict(budget["concept_accuracy"])  # type: ignore[arg-type]
        acquisition = dict(budget["acquisition"])  # type: ignore[arg-type]
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
                            f"{acquisition[concept_id]:+.6f}",
                        )
                    ),
                    str(budget["passed"]),
                )
            )
            + " |"
        )
    lines.extend(("", f"Allocator peak: `{allocator_peak_bytes}` bytes", ""))
    return "\n".join(lines)


def _budget_record(
    evaluation: PilotBudgetEvaluation,
    ledger: bytes,
    preset: QueryExperimentPreset,
    policy: PilotLearnabilityPolicy,
) -> dict[str, object]:
    base = dict(evaluation.budget.base_accuracy)
    return {
        "acquisition": [
            [concept_id, accuracy - base[concept_id]]
            for concept_id, accuracy in evaluation.budget.concept_accuracy
        ],
        "adaptation_tensor_checksum": evaluation.adaptation_tensor_checksum,
        "base_accuracy": [list(value) for value in evaluation.budget.base_accuracy],
        "concept_accuracy": [
            list(value) for value in evaluation.budget.concept_accuracy
        ],
        "config_sha256": replace(
            preset,
            adapter_updates=evaluation.budget.updates,
        ).config_sha256,
        "passed": pilot_budget_passes(evaluation.budget, policy),
        "results_sha256": sha256(ledger).hexdigest(),
        "updates": evaluation.budget.updates,
    }


def _validate_pilot_authorization(
    policy: PilotLearnabilityPolicy,
    protocol_amendment_sha256: str | None,
) -> None:
    if policy == ORIGINAL_PILOT_LEARNABILITY_POLICY:
        if protocol_amendment_sha256 is not None:
            raise ValueError("original pilot policy cannot cite an amendment")
        return
    if policy == AMENDED_PILOT_LEARNABILITY_POLICY:
        if protocol_amendment_sha256 is None:
            raise ValueError("amended pilot policy requires its authorization artifact")
        require_sha256(protocol_amendment_sha256, "pilot protocol amendment")
        return
    raise ValueError("unregistered pilot learnability policy")


def _policy_from_result_record(
    record: dict[str, object],
) -> PilotLearnabilityPolicy:
    raw = record.get("learnability_policy")
    if type(raw) is not dict:
        raise ValueError("pilot learnability policy changed")
    try:
        policy = PilotLearnabilityPolicy(
            policy_id=str(raw["policy_id"]),
            minimum_accuracy=float(raw["minimum_accuracy"]),
            acquisition_role=str(raw["acquisition_role"]),  # type: ignore[arg-type]
            minimum_acquisition=(
                None
                if raw["minimum_acquisition"] is None
                else float(raw["minimum_acquisition"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pilot learnability policy changed") from error
    if (
        policy.as_record() != raw
        or policy.policy_sha256 != record.get("learnability_policy_sha256")
    ):
        raise ValueError("pilot learnability policy identity changed")
    return policy


def _pilot_budget_from_record(record: object) -> PilotBudgetResult:
    if type(record) is not dict:
        raise ValueError("pilot budget record changed")

    def pairs(field: str) -> tuple[tuple[str, float], ...]:
        raw = record.get(field)
        if type(raw) is not list:
            raise ValueError(f"pilot {field} changed")
        result = []
        for pair in raw:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) not in (int, float)
            ):
                raise ValueError(f"pilot {field} changed")
            result.append((pair[0], float(pair[1])))
        return tuple(result)

    updates = record.get("updates")
    if type(updates) is not int:
        raise ValueError("pilot budget updates changed")
    return PilotBudgetResult(
        updates,
        pairs("concept_accuracy"),
        pairs("base_accuracy"),
    )


def _preset_from_selected_result(
    record: dict[str, object],
    budgets: tuple[PilotBudgetResult, ...],
) -> QueryExperimentPreset:
    raw = record.get("selected_config")
    if type(raw) is not dict or not budgets:
        raise ValueError("pilot selected config changed")
    concept_ids = tuple(concept_id for concept_id, _ in budgets[0].concept_accuracy)
    adapter_updates = record.get("selected_updates")
    if type(adapter_updates) is not int:
        raise ValueError("pilot selected updates changed")
    preset = QueryExperimentPreset(concept_ids, adapter_updates=adapter_updates)
    if raw != preset.as_record() or record.get("selected_config_sha256") != preset.config_sha256:
        raise ValueError("pilot selected config identity changed")
    return preset


def _verify_publication_files(
    directory: Path,
    manifest: dict[str, object],
) -> dict[str, bytes]:
    raw_files = manifest.get("files")
    if (
        type(raw_files) is not list
        or not raw_files
        or any(type(item) is not dict for item in raw_files)
    ):
        raise ValueError("pilot publication file descriptors changed")
    expected_names = {"manifest.json"}
    payload_by_name = {}
    for descriptor in raw_files:
        assert type(descriptor) is dict
        if set(descriptor) != {"name", "sha256", "size_bytes"}:
            raise ValueError("pilot publication file descriptor changed")
        name = descriptor["name"]
        if (
            type(name) is not str
            or Path(name).name != name
            or name in payload_by_name
        ):
            raise ValueError("pilot publication filename changed")
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"pilot publication file entry changed: {name}")
        payload = path.read_bytes()
        if (
            len(payload) != descriptor["size_bytes"]
            or sha256(payload).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"pilot publication file changed: {name}")
        payload_by_name[name] = payload
        expected_names.add(name)
    if (
        {path.name for path in directory.iterdir()} != expected_names
        or any(not path.is_file() or path.is_symlink() for path in directory.iterdir())
    ):
        raise ValueError("pilot publication tree entries changed")
    return payload_by_name


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
    "PILOT_FAILURE_FORMAT",
    "PILOT_RESULT_FORMAT",
    "SemanticPilotFailure",
    "SemanticPilotResult",
    "load_semantic_pilot_failure",
    "load_semantic_pilot_result",
    "publish_semantic_pilot_failure",
    "publish_semantic_pilot_result",
    "render_semantic_pilot_failure_report",
    "render_semantic_pilot_report",
]
