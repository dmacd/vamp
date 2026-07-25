"""Immutable post-pilot protocol amendments that preserve original outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    SCHEMA_VERSION,
    QueryExperimentPreset,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.execution import (
    AMENDED_PILOT_LEARNABILITY_POLICY,
    ORIGINAL_PILOT_LEARNABILITY_POLICY,
    PilotBudgetResult,
    PilotLearnabilityPolicy,
    pilot_budget_passes,
    select_pilot_budget,
)
from apm.data.text.tinyworlds_q_semantic.pilot import (
    SemanticPilotFailure,
    load_semantic_pilot_failure,
)


PILOT_PROTOCOL_AMENDMENT_FORMAT = "tinyworlds-q-semantic-pilot-amendment-v1"


@dataclass(frozen=True, slots=True)
class SemanticPilotProtocolAmendment:
    """One human-authorized policy change bound to an immutable failed pilot."""

    root: Path
    amendment_sha256: str
    failure_sha256: str
    policy: PilotLearnabilityPolicy
    selected_updates: int
    reviewer: str
    decided_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        require_sha256(self.amendment_sha256, "pilot protocol amendment")
        require_sha256(self.failure_sha256, "amended pilot failure")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if self.policy != AMENDED_PILOT_LEARNABILITY_POLICY:
            raise ValueError("pilot amendment policy changed")
        if type(self.selected_updates) is not int or self.selected_updates <= 0:
            raise ValueError("pilot amendment selected updates must be positive")
        if any(
            type(value) is not str or not value.strip()
            for value in (self.reviewer, self.decided_at)
        ):
            raise ValueError("pilot amendment reviewer and time are required")


def publish_semantic_pilot_protocol_amendment(
    output_root: str | Path,
    failure: SemanticPilotFailure,
    preset: QueryExperimentPreset,
    *,
    reviewer: str,
    decided_at: str,
    rationale: str,
) -> SemanticPilotProtocolAmendment:
    """Authorize the selected pilot exercise without rewriting the failed gate."""
    if not isinstance(failure, SemanticPilotFailure):
        raise TypeError("pilot amendment requires a strict SemanticPilotFailure")
    if not isinstance(preset, QueryExperimentPreset):
        raise TypeError("pilot amendment requires the frozen experiment preset")
    if any(
        type(value) is not str or not value.strip()
        for value in (reviewer, decided_at, rationale)
    ):
        raise ValueError("pilot amendment requires reviewer, time, and rationale")
    strict_failure = load_semantic_pilot_failure(failure.root)
    if strict_failure != failure:
        raise ValueError("pilot amendment failure reload changed")
    failure_record = _load_canonical_json(failure.root / "failure.json")
    if (
        failure_record.get("config") != preset.as_record()
        or failure_record.get("config_sha256") != preset.config_sha256
        or failure_record.get("failure_sha256") != failure.failure_sha256
    ):
        raise ValueError("pilot amendment preset does not match the failed run")
    budgets = _failure_budgets(failure_record, preset)
    if any(
        pilot_budget_passes(result, ORIGINAL_PILOT_LEARNABILITY_POLICY)
        for result in budgets
    ):
        raise ValueError("pilot amendment requires a genuine original-policy failure")
    selected_updates = select_pilot_budget(
        budgets,
        preset,
        AMENDED_PILOT_LEARNABILITY_POLICY,
    )
    core = {
        "amended_policy": AMENDED_PILOT_LEARNABILITY_POLICY.as_record(),
        "amended_policy_sha256": (
            AMENDED_PILOT_LEARNABILITY_POLICY.policy_sha256
        ),
        "authorization_scope": "selected-pilot-sequential-vamp-exercise",
        "benchmark_id": BENCHMARK_ID,
        "catalog_sha256": failure_record["catalog_sha256"],
        "decided_at": decided_at,
        "failure_sha256": failure.failure_sha256,
        "format": PILOT_PROTOCOL_AMENDMENT_FORMAT,
        "independent_sweep_sha256": failure_record["independent_sweep_sha256"],
        "original_policy": ORIGINAL_PILOT_LEARNABILITY_POLICY.as_record(),
        "original_policy_sha256": (
            ORIGINAL_PILOT_LEARNABILITY_POLICY.policy_sha256
        ),
        "partition_sha256": failure_record["partition_sha256"],
        "preflight_sha256": failure_record["preflight_sha256"],
        "rationale": rationale,
        "reviewer": reviewer,
        "schema_version": SCHEMA_VERSION,
        "sealed_test_opened": False,
        "selected_base_sha256": failure_record["selected_base_sha256"],
        "selected_updates": selected_updates,
    }
    amendment_sha256 = record_sha256(core)
    destination = Path(output_root) / amendment_sha256
    markdown = _render_amendment_markdown(amendment_sha256, core)
    content_payloads = {
        "amendment.json": canonical_json_bytes(
            {**core, "amendment_sha256": amendment_sha256}
        ),
        "report.md": markdown.encode("utf-8"),
        "report.html": (
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>TinyWorlds-Q pilot protocol amendment</title>"
            "<style>body{font:15px/1.5 system-ui;max-width:1000px;margin:2rem auto;"
            "padding:0 1rem}pre{white-space:pre-wrap;background:#f5f7f9;"
            "padding:1.25rem;border-radius:8px}</style></head>"
            f"<body data-amendment-sha256=\"{amendment_sha256}\"><pre>"
            f"{html.escape(markdown)}</pre></body></html>\n"
        ).encode("utf-8"),
    }
    manifest = {
        "amendment_sha256": amendment_sha256,
        "files": [
            {
                "name": name,
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(content_payloads.items())
        ],
        "format": PILOT_PROTOCOL_AMENDMENT_FORMAT,
        "schema_version": SCHEMA_VERSION,
    }
    payloads = {
        **content_payloads,
        "manifest.json": canonical_json_bytes(manifest),
    }
    if destination.exists():
        loaded = load_semantic_pilot_protocol_amendment(
            destination,
            failure,
            preset,
        )
        if loaded.reviewer != reviewer or loaded.decided_at != decided_at:
            raise FileExistsError("existing pilot amendment changed")
        return loaded
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".pilot-amendment-", dir=destination.parent)
    )
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        os.replace(staging, destination)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_semantic_pilot_protocol_amendment(destination, failure, preset)


def load_semantic_pilot_protocol_amendment(
    directory: str | Path,
    failure: SemanticPilotFailure,
    preset: QueryExperimentPreset,
) -> SemanticPilotProtocolAmendment:
    """Strictly authenticate one amendment against its exact failed evidence."""
    root = Path(directory)
    strict_failure = load_semantic_pilot_failure(failure.root)
    if strict_failure != failure:
        raise ValueError("pilot amendment failure reload changed")
    manifest = _load_canonical_json(root / "manifest.json")
    record = _load_canonical_json(root / "amendment.json")
    if (
        set(manifest)
        != {"amendment_sha256", "files", "format", "schema_version"}
        or manifest.get("format") != PILOT_PROTOCOL_AMENDMENT_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or record.get("format") != PILOT_PROTOCOL_AMENDMENT_FORMAT
        or record.get("amendment_sha256") != manifest.get("amendment_sha256")
        or root.name != manifest.get("amendment_sha256")
    ):
        raise ValueError("pilot amendment identity changed")
    _verify_files(root, manifest)
    amendment_sha256 = str(manifest["amendment_sha256"])
    core = {key: value for key, value in record.items() if key != "amendment_sha256"}
    if record_sha256(core) != amendment_sha256:
        raise ValueError("pilot amendment content hash changed")
    failure_record = _load_canonical_json(failure.root / "failure.json")
    budgets = _failure_budgets(failure_record, preset)
    selected_updates = select_pilot_budget(
        budgets,
        preset,
        AMENDED_PILOT_LEARNABILITY_POLICY,
    )
    expected_bindings = {
        "amended_policy": AMENDED_PILOT_LEARNABILITY_POLICY.as_record(),
        "amended_policy_sha256": AMENDED_PILOT_LEARNABILITY_POLICY.policy_sha256,
        "authorization_scope": "selected-pilot-sequential-vamp-exercise",
        "benchmark_id": BENCHMARK_ID,
        "catalog_sha256": failure_record["catalog_sha256"],
        "failure_sha256": failure.failure_sha256,
        "format": PILOT_PROTOCOL_AMENDMENT_FORMAT,
        "independent_sweep_sha256": failure_record["independent_sweep_sha256"],
        "original_policy": ORIGINAL_PILOT_LEARNABILITY_POLICY.as_record(),
        "original_policy_sha256": ORIGINAL_PILOT_LEARNABILITY_POLICY.policy_sha256,
        "partition_sha256": failure_record["partition_sha256"],
        "preflight_sha256": failure_record["preflight_sha256"],
        "schema_version": SCHEMA_VERSION,
        "sealed_test_opened": False,
        "selected_base_sha256": failure_record["selected_base_sha256"],
        "selected_updates": selected_updates,
    }
    if any(record.get(key) != value for key, value in expected_bindings.items()):
        raise ValueError("pilot amendment binding changed")
    for field in ("decided_at", "rationale", "reviewer"):
        if type(record.get(field)) is not str or not str(record[field]).strip():
            raise ValueError(f"pilot amendment {field} changed")
    markdown = _render_amendment_markdown(amendment_sha256, core)
    if (
        (root / "report.md").read_text(encoding="utf-8") != markdown
        or f'data-amendment-sha256="{amendment_sha256}"'
        not in (root / "report.html").read_text(encoding="utf-8")
    ):
        raise ValueError("pilot amendment report changed")
    return SemanticPilotProtocolAmendment(
        root=root.resolve(),
        amendment_sha256=amendment_sha256,
        failure_sha256=failure.failure_sha256,
        policy=AMENDED_PILOT_LEARNABILITY_POLICY,
        selected_updates=selected_updates,
        reviewer=str(record["reviewer"]),
        decided_at=str(record["decided_at"]),
    )


def _failure_budgets(
    failure_record: dict[str, object],
    preset: QueryExperimentPreset,
) -> tuple[PilotBudgetResult, ...]:
    raw_budgets = failure_record.get("budgets")
    if type(raw_budgets) is not list or any(
        type(value) is not dict for value in raw_budgets
    ):
        raise ValueError("pilot amendment failure budgets changed")
    results = []
    for raw in raw_budgets:
        assert type(raw) is dict
        if raw.get("passed") is not False:
            raise ValueError("pilot amendment requires preserved failed budget rows")
        results.append(
            PilotBudgetResult(
                updates=_integer(raw, "updates"),
                concept_accuracy=_accuracy_pairs(raw, "concept_accuracy"),
                base_accuracy=_accuracy_pairs(raw, "base_accuracy"),
            )
        )
    result = tuple(results)
    if (
        tuple(item.updates for item in result) != preset.pilot_update_budgets
        or any(
            tuple(concept_id for concept_id, _ in item.concept_accuracy)
            != preset.concept_ids
            for item in result
        )
    ):
        raise ValueError("pilot amendment failure budget order changed")
    return result


def _accuracy_pairs(
    record: dict[str, object],
    field: str,
) -> tuple[tuple[str, float], ...]:
    raw = record.get(field)
    if type(raw) is not list:
        raise ValueError(f"pilot amendment {field} changed")
    pairs = []
    for pair in raw:
        if (
            type(pair) is not list
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) not in (int, float)
        ):
            raise ValueError(f"pilot amendment {field} pair changed")
        pairs.append((pair[0], float(pair[1])))
    return tuple(pairs)


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"pilot amendment {field} changed")
    return value


def _render_amendment_markdown(
    amendment_sha256: str,
    record: dict[str, object],
) -> str:
    return (
        "# TinyWorlds-Q pilot protocol amendment\n\n"
        f"Amendment: `{amendment_sha256}`  \n"
        f"Original failure: `{record['failure_sha256']}`  \n"
        f"Reviewer: `{record['reviewer']}`  \n"
        f"Decided at: `{record['decided_at']}`\n\n"
        "The original 60%-accuracy plus 15-percentage-point acquisition gate "
        "remains failed and immutable. This post-pilot amendment makes acquisition "
        "descriptive and retains 60% validation accuracy as the authorization gate.\n\n"
        f"Rationale: {record['rationale']}\n\n"
        f"Selected update budget: **{int(record['selected_updates']):,}**\n\n"
        "Scope: selected-budget sequential/VAMP pilot exercise. Main work is "
        "authorized only after that exercise publishes successfully. The sealed "
        "test was not opened.\n"
    )


def _verify_files(root: Path, manifest: dict[str, object]) -> None:
    raw_files = manifest.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise ValueError("pilot amendment file descriptors changed")
    expected_names = {"manifest.json"}
    for descriptor in raw_files:
        assert type(descriptor) is dict
        if set(descriptor) != {"name", "sha256", "size_bytes"}:
            raise ValueError("pilot amendment file descriptor changed")
        name = descriptor.get("name")
        if type(name) is not str or Path(name).name != name or name in expected_names:
            raise ValueError("pilot amendment filename changed")
        expected_names.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("pilot amendment file entry changed")
        payload = path.read_bytes()
        if (
            len(payload) != descriptor.get("size_bytes")
            or sha256(payload).hexdigest() != descriptor.get("sha256")
        ):
            raise ValueError("pilot amendment file changed")
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("pilot amendment tree entries changed")


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid pilot amendment JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"pilot amendment JSON is not canonical: {path.name}")
    return value


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


__all__ = [
    "PILOT_PROTOCOL_AMENDMENT_FORMAT",
    "SemanticPilotProtocolAmendment",
    "load_semantic_pilot_protocol_amendment",
    "publish_semantic_pilot_protocol_amendment",
]
