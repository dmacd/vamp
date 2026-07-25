"""Content-addressed authorization boundary between pilot and main construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
)
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
)
from apm.data.text.tinyworlds_q_semantic.manifests import (
    MAIN_CONCEPTS,
    MAIN_CONCEPT_IDS,
)
from apm.data.text.tinyworlds_q_semantic.pilot import (
    SemanticPilotFailure,
    SemanticPilotResult,
    load_semantic_pilot_failure,
    load_semantic_pilot_result,
)
from apm.data.text.tinyworlds_q_semantic.pilot_authorization import (
    SemanticPilotProtocolAmendment,
    load_semantic_pilot_protocol_amendment,
)
from apm.data.text.tinyworlds_q_semantic.query_protocol import (
    REGISTERED_QUERY_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.report import REQUIRED_QUERY_METHODS
from apm.data.text.tinyworlds_q_semantic.statistics import (
    CANONICAL_BOOTSTRAP_REPLICATES,
)


MAIN_EXPERIMENT_FREEZE_FORMAT = "tinyworlds-q-semantic-main-freeze-v1"


@dataclass(frozen=True, slots=True)
class MainExperimentFreeze:
    """Strict loaded main configuration and its complete pilot authority chain."""

    root: Path
    freeze_sha256: str
    pilot_sha256: str
    protocol_amendment_sha256: str
    main_config_sha256: str
    query_protocol_sha256: str
    concept_manifest_sha256: str
    selected_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.freeze_sha256, "main experiment freeze"),
            (self.pilot_sha256, "main-authorizing pilot"),
            (self.protocol_amendment_sha256, "main-authorizing amendment"),
            (self.main_config_sha256, "main experiment config"),
            (self.query_protocol_sha256, "main query protocol"),
            (self.concept_manifest_sha256, "main concept manifest"),
        ):
            require_sha256(value, label)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if type(self.selected_updates) is not int or self.selected_updates <= 0:
            raise ValueError("main selected update budget must be positive")


def publish_main_experiment_freeze(
    output_root: str | Path,
    pilot: SemanticPilotResult,
    amendment: SemanticPilotProtocolAmendment,
    failure: SemanticPilotFailure,
    pilot_preset: QueryExperimentPreset,
    main_preset: QueryExperimentPreset,
    *,
    authorized_by: str,
    authorized_at: str,
    authorization: str,
) -> MainExperimentFreeze:
    """Freeze the main run before any five-world semantic artifact is built."""
    _validate_authority_chain(pilot, amendment, failure, pilot_preset, main_preset)
    if any(
        type(value) is not str or not value.strip()
        for value in (authorized_by, authorized_at, authorization)
    ):
        raise ValueError("main freeze requires authorization identity, time, and text")
    concepts = [concept.as_record() for concept in MAIN_CONCEPTS]
    concept_manifest_sha256 = record_sha256(concepts)
    evaluation_protocol = {
        "bootstrap_replicates": CANONICAL_BOOTSTRAP_REPLICATES,
        "generation_role": "secondary-exact-trigger-inspection",
        "methods": list(REQUIRED_QUERY_METHODS),
        "parent_and_router_inputs": "validation-question-prefixes-only",
        "sealed_test_openings": 1,
    }
    core = {
        "archive_identity": CANONICAL_ARCHIVE_IDENTITY.as_record(),
        "authorization": authorization,
        "authorized_at": authorized_at,
        "authorized_by": authorized_by,
        "benchmark_id": BENCHMARK_ID,
        "concept_manifest": concepts,
        "concept_manifest_sha256": concept_manifest_sha256,
        "evaluation_protocol": evaluation_protocol,
        "format": MAIN_EXPERIMENT_FREEZE_FORMAT,
        "fresh_main_base_required": True,
        "main_catalog_sha256": None,
        "main_config": main_preset.as_record(),
        "main_config_sha256": main_preset.config_sha256,
        "max_edges": main_preset.max_edges,
        "max_nodes": main_preset.max_nodes,
        "parent_catalog_sha256": None,
        "pilot_failure_sha256": failure.failure_sha256,
        "pilot_learnability_policy_sha256": pilot.learnability_policy_sha256,
        "pilot_sha256": pilot.pilot_sha256,
        "protocol_amendment_sha256": amendment.amendment_sha256,
        "query_protocol": REGISTERED_QUERY_PROTOCOL.as_record(),
        "query_protocol_sha256": REGISTERED_QUERY_PROTOCOL.protocol_sha256,
        "schema_version": SCHEMA_VERSION,
        "sealed_test_opened": False,
        "selected_updates": pilot.selected_updates,
        "tokenizer_identity": CANONICAL_TOKENIZER_IDENTITY.as_record(),
    }
    freeze_sha256 = record_sha256(core)
    destination = Path(output_root) / freeze_sha256
    markdown = _render_markdown(freeze_sha256, core)
    content_payloads = {
        "freeze.json": canonical_json_bytes(
            {**core, "freeze_sha256": freeze_sha256}
        ),
        "report.md": markdown.encode("utf-8"),
        "report.html": (
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>TinyWorlds-Q main experiment freeze</title>"
            "<style>body{font:15px/1.5 system-ui;max-width:1000px;margin:2rem auto;"
            "padding:0 1rem}pre{white-space:pre-wrap;background:#f5f7f9;"
            "padding:1.25rem;border-radius:8px}</style></head>"
            f"<body data-freeze-sha256=\"{freeze_sha256}\"><pre>"
            f"{html.escape(markdown)}</pre></body></html>\n"
        ).encode("utf-8"),
    }
    manifest = {
        "files": [
            {
                "name": name,
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(content_payloads.items())
        ],
        "format": MAIN_EXPERIMENT_FREEZE_FORMAT,
        "freeze_sha256": freeze_sha256,
        "schema_version": SCHEMA_VERSION,
    }
    payloads = {
        **content_payloads,
        "manifest.json": canonical_json_bytes(manifest),
    }
    if destination.exists():
        return load_main_experiment_freeze(
            destination,
            pilot,
            amendment,
            failure,
            pilot_preset,
            main_preset,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".main-freeze-", dir=destination.parent))
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        os.replace(staging, destination)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_main_experiment_freeze(
        destination,
        pilot,
        amendment,
        failure,
        pilot_preset,
        main_preset,
    )


def load_main_experiment_freeze(
    directory: str | Path,
    pilot: SemanticPilotResult,
    amendment: SemanticPilotProtocolAmendment,
    failure: SemanticPilotFailure,
    pilot_preset: QueryExperimentPreset,
    main_preset: QueryExperimentPreset,
) -> MainExperimentFreeze:
    """Authenticate one main freeze against the strict pilot authority chain."""
    _validate_authority_chain(pilot, amendment, failure, pilot_preset, main_preset)
    root = Path(directory)
    manifest = _load_canonical_json(root / "manifest.json")
    record = _load_canonical_json(root / "freeze.json")
    if (
        set(manifest) != {"files", "format", "freeze_sha256", "schema_version"}
        or manifest.get("format") != MAIN_EXPERIMENT_FREEZE_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
        or record.get("format") != MAIN_EXPERIMENT_FREEZE_FORMAT
        or record.get("freeze_sha256") != manifest.get("freeze_sha256")
        or root.name != manifest.get("freeze_sha256")
    ):
        raise ValueError("main experiment freeze identity changed")
    _verify_files(root, manifest)
    freeze_sha256 = str(manifest["freeze_sha256"])
    core = {key: value for key, value in record.items() if key != "freeze_sha256"}
    if record_sha256(core) != freeze_sha256:
        raise ValueError("main experiment freeze content hash changed")
    concepts = [concept.as_record() for concept in MAIN_CONCEPTS]
    expected = {
        "archive_identity": CANONICAL_ARCHIVE_IDENTITY.as_record(),
        "benchmark_id": BENCHMARK_ID,
        "concept_manifest": concepts,
        "concept_manifest_sha256": record_sha256(concepts),
        "evaluation_protocol": {
            "bootstrap_replicates": CANONICAL_BOOTSTRAP_REPLICATES,
            "generation_role": "secondary-exact-trigger-inspection",
            "methods": list(REQUIRED_QUERY_METHODS),
            "parent_and_router_inputs": "validation-question-prefixes-only",
            "sealed_test_openings": 1,
        },
        "format": MAIN_EXPERIMENT_FREEZE_FORMAT,
        "fresh_main_base_required": True,
        "main_catalog_sha256": None,
        "main_config": main_preset.as_record(),
        "main_config_sha256": main_preset.config_sha256,
        "max_edges": main_preset.max_edges,
        "max_nodes": main_preset.max_nodes,
        "parent_catalog_sha256": None,
        "pilot_failure_sha256": failure.failure_sha256,
        "pilot_learnability_policy_sha256": pilot.learnability_policy_sha256,
        "pilot_sha256": pilot.pilot_sha256,
        "protocol_amendment_sha256": amendment.amendment_sha256,
        "query_protocol": REGISTERED_QUERY_PROTOCOL.as_record(),
        "query_protocol_sha256": REGISTERED_QUERY_PROTOCOL.protocol_sha256,
        "schema_version": SCHEMA_VERSION,
        "sealed_test_opened": False,
        "selected_updates": pilot.selected_updates,
        "tokenizer_identity": CANONICAL_TOKENIZER_IDENTITY.as_record(),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("main experiment freeze binding changed")
    for field in ("authorization", "authorized_at", "authorized_by"):
        if type(record.get(field)) is not str or not str(record[field]).strip():
            raise ValueError(f"main experiment freeze {field} changed")
    markdown = _render_markdown(freeze_sha256, core)
    if (
        (root / "report.md").read_text(encoding="utf-8") != markdown
        or f'data-freeze-sha256="{freeze_sha256}"'
        not in (root / "report.html").read_text(encoding="utf-8")
    ):
        raise ValueError("main experiment freeze report changed")
    return MainExperimentFreeze(
        root=root.resolve(),
        freeze_sha256=freeze_sha256,
        pilot_sha256=pilot.pilot_sha256,
        protocol_amendment_sha256=amendment.amendment_sha256,
        main_config_sha256=main_preset.config_sha256,
        query_protocol_sha256=REGISTERED_QUERY_PROTOCOL.protocol_sha256,
        concept_manifest_sha256=record_sha256(concepts),
        selected_updates=pilot.selected_updates,
    )


def _validate_authority_chain(
    pilot: SemanticPilotResult,
    amendment: SemanticPilotProtocolAmendment,
    failure: SemanticPilotFailure,
    pilot_preset: QueryExperimentPreset,
    main_preset: QueryExperimentPreset,
) -> None:
    strict_failure = load_semantic_pilot_failure(failure.root)
    strict_amendment = load_semantic_pilot_protocol_amendment(
        amendment.root,
        strict_failure,
        pilot_preset,
    )
    strict_pilot = load_semantic_pilot_result(pilot.root)
    expected_main = QueryExperimentPreset(
        MAIN_CONCEPT_IDS,
        adapter_updates=strict_pilot.selected_updates,
    )
    if (
        strict_failure != failure
        or strict_amendment != amendment
        or strict_pilot != pilot
        or strict_pilot.protocol_amendment_sha256 != amendment.amendment_sha256
        or strict_pilot.learnability_policy_sha256
        != AMENDED_PILOT_LEARNABILITY_POLICY.policy_sha256
        or strict_pilot.selected_updates != amendment.selected_updates
        or main_preset != expected_main
    ):
        raise ValueError("main experiment pilot authority chain changed")


def _render_markdown(freeze_sha256: str, record: dict[str, object]) -> str:
    concepts = ", ".join(
        concept["concept_id"]  # type: ignore[index]
        for concept in record["concept_manifest"]  # type: ignore[union-attr]
    )
    return (
        "# TinyWorlds-Q main experiment freeze\n\n"
        f"Freeze: `{freeze_sha256}`  \n"
        f"Authorizing pilot: `{record['pilot_sha256']}`  \n"
        f"Protocol amendment: `{record['protocol_amendment_sha256']}`  \n"
        f"Authorized by: `{record['authorized_by']}`  \n"
        f"Authorized at: `{record['authorized_at']}`\n\n"
        f"Authorization: {record['authorization']}\n\n"
        f"Main order: **{concepts}**  \n"
        f"Adapter updates per world/system: **{int(record['selected_updates']):,}**  \n"
        f"Main config: `{record['main_config_sha256']}`  \n"
        f"Query protocol: `{record['query_protocol_sha256']}`\n\n"
        "The five-world catalog, partition, and base do not exist at this boundary. "
        "They must be built under this frozen validation-only protocol. Main uses "
        "a fresh seed-zero base; no pilot or semantic-v6 parameters may be reused. "
        "The sealed test remains closed.\n"
    )


def _verify_files(root: Path, manifest: dict[str, object]) -> None:
    raw_files = manifest.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise ValueError("main experiment freeze file descriptors changed")
    expected_names = {"manifest.json"}
    for descriptor in raw_files:
        assert type(descriptor) is dict
        if set(descriptor) != {"name", "sha256", "size_bytes"}:
            raise ValueError("main experiment freeze file descriptor changed")
        name = descriptor.get("name")
        if type(name) is not str or Path(name).name != name or name in expected_names:
            raise ValueError("main experiment freeze filename changed")
        expected_names.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("main experiment freeze file entry changed")
        payload = path.read_bytes()
        if (
            len(payload) != descriptor.get("size_bytes")
            or sha256(payload).hexdigest() != descriptor.get("sha256")
        ):
            raise ValueError("main experiment freeze file changed")
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("main experiment freeze tree entries changed")


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid main experiment freeze JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"main experiment freeze JSON is not canonical: {path.name}")
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
    "MAIN_EXPERIMENT_FREEZE_FORMAT",
    "MainExperimentFreeze",
    "load_main_experiment_freeze",
    "publish_main_experiment_freeze",
]
