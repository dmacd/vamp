"""Operational gates, pilot selection, stage artifacts, and sealed transactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Literal

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)


STAGE_ARTIFACT_FORMAT = "tinyworlds-q-semantic-stage-v1"
SEALED_TRANSACTION_FORMAT = "tinyworlds-q-semantic-sealed-transaction-v1"

PilotAcquisitionRole = Literal["gate", "descriptive"]


@dataclass(frozen=True, slots=True)
class PilotLearnabilityPolicy:
    """One hashable rule for turning validation evidence into pilot authorization."""

    policy_id: str
    minimum_accuracy: float
    acquisition_role: PilotAcquisitionRole
    minimum_acquisition: float | None

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("pilot policy_id must be nonempty")
        if (
            type(self.minimum_accuracy) is not float
            or not isfinite(self.minimum_accuracy)
            or not 0.0 <= self.minimum_accuracy <= 1.0
        ):
            raise ValueError("pilot minimum accuracy must be a finite probability")
        if self.acquisition_role not in ("gate", "descriptive"):
            raise ValueError("pilot acquisition role must be gate or descriptive")
        if self.acquisition_role == "gate":
            if (
                type(self.minimum_acquisition) is not float
                or not isfinite(self.minimum_acquisition)
                or not 0.0 <= self.minimum_acquisition <= 1.0
            ):
                raise ValueError("gated pilot acquisition requires a finite probability")
        elif self.minimum_acquisition is not None:
            raise ValueError("descriptive pilot acquisition cannot declare a threshold")

    @property
    def policy_sha256(self) -> str:
        """Hash the complete authorization rule independently of training config."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical policy record."""
        return {
            "acquisition_role": self.acquisition_role,
            "minimum_accuracy": self.minimum_accuracy,
            "minimum_acquisition": self.minimum_acquisition,
            "policy_id": self.policy_id,
        }


ORIGINAL_PILOT_LEARNABILITY_POLICY = PilotLearnabilityPolicy(
    policy_id="original-accuracy-and-acquisition-gates",
    minimum_accuracy=0.60,
    acquisition_role="gate",
    minimum_acquisition=0.15,
)

AMENDED_PILOT_LEARNABILITY_POLICY = PilotLearnabilityPolicy(
    policy_id="post-pilot-absolute-accuracy-gate",
    minimum_accuracy=0.60,
    acquisition_role="descriptive",
    minimum_acquisition=None,
)


@dataclass(frozen=True, slots=True)
class BaseQualityDecision:
    """Mandatory two-epoch held-in quality and memory gate."""

    epoch_nll: tuple[float, float]
    allocator_peak_bytes: int
    allocator_limit_bytes: int
    passed: bool = field(init=False)
    reason: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.epoch_nll) is not tuple
            or len(self.epoch_nll) != 2
            or type(self.allocator_peak_bytes) is not int
            or type(self.allocator_limit_bytes) is not int
            or self.allocator_peak_bytes < 0
            or self.allocator_limit_bytes <= 0
        ):
            raise ValueError("base gate requires two losses and measured byte counts")
        first, second = self.epoch_nll
        if not all(isfinite(value) and value >= 0.0 for value in self.epoch_nll):
            passed, reason = False, "non_finite_nll"
        elif self.allocator_peak_bytes > self.allocator_limit_bytes:
            passed, reason = False, "allocator_peak_exceeded"
        elif second > 2.2:
            passed, reason = False, "held_in_nll_above_2.2"
        elif first - second < 0.02:
            passed, reason = False, "epoch_improvement_below_0.02"
        else:
            passed, reason = True, "passed"
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class PilotBudgetResult:
    """Validation accuracy evidence for one fixed pilot adapter budget."""

    updates: int
    concept_accuracy: tuple[tuple[str, float], ...]
    base_accuracy: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if type(self.updates) is not int or self.updates <= 0:
            raise ValueError("pilot update budget must be positive")
        for values, label in (
            (self.concept_accuracy, "pilot concept accuracy"),
            (self.base_accuracy, "pilot base accuracy"),
        ):
            if (
                type(values) is not tuple
                or not values
                or len({concept_id for concept_id, _ in values}) != len(values)
                or any(
                    type(concept_id) is not str
                    or not concept_id
                    or not isfinite(value)
                    or not 0.0 <= value <= 1.0
                    for concept_id, value in values
                )
            ):
                raise ValueError(f"{label} is invalid")
        if tuple(item[0] for item in self.concept_accuracy) != tuple(
            item[0] for item in self.base_accuracy
        ):
            raise ValueError("pilot and base accuracies must align by concept")

    @property
    def acquisition(self) -> tuple[tuple[str, float], ...]:
        """Return the paired absolute accuracy gain for every pilot world."""
        base_by_concept = dict(self.base_accuracy)
        return tuple(
            (concept_id, accuracy - base_by_concept[concept_id])
            for concept_id, accuracy in self.concept_accuracy
        )


def pilot_budget_passes(
    result: PilotBudgetResult,
    policy: PilotLearnabilityPolicy,
) -> bool:
    """Apply one explicit policy to immutable per-world validation evidence."""
    if not isinstance(result, PilotBudgetResult):
        raise TypeError("pilot policy requires a PilotBudgetResult")
    if not isinstance(policy, PilotLearnabilityPolicy):
        raise TypeError("pilot policy must be a PilotLearnabilityPolicy")
    acquisition_by_concept = dict(result.acquisition)
    return all(
        accuracy >= policy.minimum_accuracy
        and (
            policy.acquisition_role == "descriptive"
            or acquisition_by_concept[concept_id] >= policy.minimum_acquisition
        )
        for concept_id, accuracy in result.concept_accuracy
    )


def select_pilot_budget(
    results: tuple[PilotBudgetResult, ...],
    preset: QueryExperimentPreset,
    policy: PilotLearnabilityPolicy,
) -> int:
    """Select the smallest registered budget passing one explicit policy."""
    if not isinstance(policy, PilotLearnabilityPolicy):
        raise TypeError("pilot selection requires an explicit learnability policy")
    by_budget = {result.updates: result for result in results}
    if len(results) != len(by_budget) or set(by_budget) != set(preset.pilot_update_budgets):
        raise ValueError("pilot results must cover every registered update budget")
    if any(
        tuple(concept_id for concept_id, _ in result.concept_accuracy)
        != preset.concept_ids
        for result in results
    ):
        raise ValueError("pilot results must follow the complete active concept manifest")
    passing = tuple(
        budget
        for budget in preset.pilot_update_budgets
        if pilot_budget_passes(by_budget[budget], policy)
    )
    if not passing:
        raise RuntimeError(
            "pilot learnability policy failed at 500, 1000, and 2000 updates"
        )
    return passing[0]


@dataclass(frozen=True, slots=True)
class StageArtifact:
    """One immutable resumable system stage bound to an ordered concept prefix."""

    root: Path
    system: str
    stage: int
    completed_concept_ids: tuple[str, ...]
    config_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if type(self.system) is not str or not self.system:
            raise ValueError("stage system must be nonempty")
        if type(self.stage) is not int or self.stage < 0:
            raise ValueError("stage index must be nonnegative")
        if len(self.completed_concept_ids) != self.stage:
            raise ValueError("stage prefix length must equal its index")
        require_sha256(self.config_sha256, "stage config")
        require_sha256(self.state_sha256, "stage state")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)


def publish_stage_artifact(
    root: str | Path,
    preset: QueryExperimentPreset,
    *,
    system: str,
    stage: int,
    payloads: Mapping[str, bytes],
) -> StageArtifact:
    """Atomically persist one stage so interruption resumes only complete prefixes."""
    if type(stage) is not int or not 0 <= stage <= preset.active_world_count:
        raise ValueError("stage lies outside the active manifest")
    if not payloads or any(
        type(name) is not str
        or not name
        or Path(name).name != name
        or type(payload) is not bytes
        for name, payload in payloads.items()
    ):
        raise ValueError("stage payloads require safe basenames and exact bytes")
    destination = Path(root) / system / f"stage-{stage:03d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".stage-{stage:03d}-", dir=destination.parent))
    try:
        files = []
        for name, payload in sorted(payloads.items()):
            (staging / name).write_bytes(payload)
            files.append(
                {
                    "name": name,
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        core = {
            "completed_concept_ids": list(preset.concept_ids[:stage]),
            "config_sha256": preset.config_sha256,
            "files": files,
            "format": STAGE_ARTIFACT_FORMAT,
            "stage": stage,
            "system": system,
        }
        state_sha256 = record_sha256(core)
        (staging / "manifest.json").write_bytes(
            canonical_json_bytes({**core, "state_sha256": state_sha256})
        )
        if destination.exists():
            existing = load_stage_artifact(destination, preset, system=system)
            if existing.state_sha256 != state_sha256:
                raise FileExistsError("immutable stage artifact already differs")
            _remove_tree(staging)
            return existing
        os.replace(staging, destination)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_stage_artifact(destination, preset, system=system)


def load_stage_artifact(
    directory: str | Path,
    preset: QueryExperimentPreset,
    *,
    system: str,
) -> StageArtifact:
    """Strictly authenticate one complete stage and every tensor/state payload."""
    root = Path(directory)
    manifest = _load_canonical_json(root / "manifest.json")
    if (
        manifest.get("format") != STAGE_ARTIFACT_FORMAT
        or manifest.get("system") != system
        or manifest.get("config_sha256") != preset.config_sha256
    ):
        raise ValueError("stage artifact identity changed")
    stage = _integer(manifest, "stage")
    completed = _string_tuple(manifest, "completed_concept_ids")
    if completed != preset.concept_ids[:stage]:
        raise ValueError("stage artifact does not preserve the active prefix")
    file_records = _record_tuple(manifest, "files")
    expected = {"manifest.json", *(_string(item, "name") for item in file_records)}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("stage artifact entries changed")
    for record in file_records:
        payload = (root / _string(record, "name")).read_bytes()
        if len(payload) != _integer(record, "size_bytes") or sha256(payload).hexdigest() != _string(record, "sha256"):
            raise ValueError("stage artifact payload changed")
    core = {key: value for key, value in manifest.items() if key != "state_sha256"}
    state_sha256 = record_sha256(core)
    if manifest.get("state_sha256") != state_sha256:
        raise ValueError("stage artifact state identity changed")
    return StageArtifact(
        root=root.resolve(),
        system=system,
        stage=stage,
        completed_concept_ids=completed,
        config_sha256=preset.config_sha256,
        state_sha256=state_sha256,
    )


def latest_stage_artifact(
    root: str | Path,
    preset: QueryExperimentPreset,
    *,
    system: str,
) -> StageArtifact | None:
    """Return the newest strict complete stage, ignoring incomplete temp directories."""
    system_root = Path(root) / system
    if not system_root.is_dir():
        return None
    candidates = tuple(
        sorted(
            (
                path
                for path in system_root.iterdir()
                if path.is_dir() and path.name.startswith("stage-")
            ),
            reverse=True,
        )
    )
    return next(
        (
            load_stage_artifact(path, preset, system=system)
            for path in candidates
            if (path / "manifest.json").is_file()
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class SealedTestTransaction:
    """Durable binding proving test access follows all frozen artifacts."""

    root: Path
    catalog_sha256: str
    partition_sha256: str
    selected_base_sha256: str
    adapters_sha256: str
    config_sha256: str
    transaction_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.catalog_sha256, "sealed catalog"),
            (self.partition_sha256, "sealed partition"),
            (self.selected_base_sha256, "sealed base"),
            (self.adapters_sha256, "sealed adapters"),
            (self.config_sha256, "sealed config"),
            (self.transaction_sha256, "sealed transaction"),
        ):
            require_sha256(value, label)

    @property
    def transaction_authenticated(self) -> bool:
        """Authenticate the frozen binding without opening sealed content."""
        try:
            loaded = load_sealed_transaction(self.root)
        except (OSError, ValueError):
            return False
        return loaded == self

    @property
    def test_access_authorized(self) -> bool:
        """Authorize only one opened, incomplete evaluation transaction."""
        if not self.transaction_authenticated or (self.root / "sealed-complete.json").exists():
            return False
        path = self.root / "sealed-open.json"
        expected = canonical_json_bytes(
            {
                "status": "opened",
                "transaction_sha256": self.transaction_sha256,
            }
        )
        return path.is_file() and path.read_bytes() == expected


def publish_sealed_transaction(
    root: str | Path,
    *,
    catalog_sha256: str,
    partition_sha256: str,
    selected_base_sha256: str,
    adapters_sha256: str,
    config_sha256: str,
) -> SealedTestTransaction:
    """Durably freeze every artifact and setting before any test index is read."""
    transaction_root = Path(root)
    transaction_root.mkdir(parents=True, exist_ok=True)
    core = {
        "adapters_sha256": adapters_sha256,
        "catalog_sha256": catalog_sha256,
        "config_sha256": config_sha256,
        "format": SEALED_TRANSACTION_FORMAT,
        "partition_sha256": partition_sha256,
        "selected_base_sha256": selected_base_sha256,
    }
    for value in (
        catalog_sha256,
        partition_sha256,
        selected_base_sha256,
        adapters_sha256,
        config_sha256,
    ):
        require_sha256(value, "sealed transaction input")
    transaction_sha256 = record_sha256(core)
    payload = canonical_json_bytes({**core, "transaction_sha256": transaction_sha256})
    path = transaction_root / "sealed-transaction.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("a different sealed transaction already exists")
    else:
        temporary = transaction_root / ".sealed-transaction.json.tmp"
        temporary.write_bytes(payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return load_sealed_transaction(transaction_root)


def load_sealed_transaction(root: str | Path) -> SealedTestTransaction:
    """Authenticate one frozen transaction without accessing a test query."""
    transaction_root = Path(root)
    record = _load_canonical_json(transaction_root / "sealed-transaction.json")
    if record.get("format") != SEALED_TRANSACTION_FORMAT:
        raise ValueError("sealed transaction format changed")
    core = {key: value for key, value in record.items() if key != "transaction_sha256"}
    transaction_sha256 = record_sha256(core)
    if record.get("transaction_sha256") != transaction_sha256:
        raise ValueError("sealed transaction identity changed")
    return SealedTestTransaction(
        root=transaction_root.resolve(),
        catalog_sha256=_string(record, "catalog_sha256"),
        partition_sha256=_string(record, "partition_sha256"),
        selected_base_sha256=_string(record, "selected_base_sha256"),
        adapters_sha256=_string(record, "adapters_sha256"),
        config_sha256=_string(record, "config_sha256"),
        transaction_sha256=transaction_sha256,
    )


def begin_sealed_test(transaction: SealedTestTransaction) -> Path:
    """Open or resume the one test transaction while rejecting any other binding."""
    if not transaction.transaction_authenticated:
        raise PermissionError("sealed transaction is not durably authenticated")
    if (transaction.root / "sealed-complete.json").exists():
        raise RuntimeError("sealed test transaction is already complete")
    path = transaction.root / "sealed-open.json"
    payload = canonical_json_bytes(
        {
            "status": "opened",
            "transaction_sha256": transaction.transaction_sha256,
        }
    )
    if path.exists() and path.read_bytes() != payload:
        raise PermissionError("sealed test was opened by a different transaction")
    if not path.exists():
        temporary = transaction.root / ".sealed-open.json.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return path


def complete_sealed_test(
    transaction: SealedTestTransaction,
    result_sha256: str,
) -> Path:
    """Close the one test transaction after a durable result publication."""
    require_sha256(result_sha256, "sealed result")
    if not transaction.test_access_authorized:
        raise PermissionError("sealed test is not in one authenticated open transaction")
    path = transaction.root / "sealed-complete.json"
    payload = canonical_json_bytes(
        {
            "result_sha256": result_sha256,
            "status": "complete",
            "transaction_sha256": transaction.transaction_sha256,
        }
    )
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError("sealed transaction already names a different result")
    if not path.exists():
        temporary = transaction.root / ".sealed-complete.json.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return path


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact JSON {path.name}: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"artifact JSON is not canonical: {path.name}")
    return value


def _record_tuple(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"artifact {field} must contain records")
    return tuple(value)  # type: ignore[arg-type]


def _string_tuple(record: dict[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"artifact {field} must contain text")
    return tuple(value)  # type: ignore[arg-type]


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"artifact {field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"artifact {field} must be nonnegative")
    return value


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
    "BaseQualityDecision",
    "PilotBudgetResult",
    "SEALED_TRANSACTION_FORMAT",
    "STAGE_ARTIFACT_FORMAT",
    "SealedTestTransaction",
    "StageArtifact",
    "begin_sealed_test",
    "complete_sealed_test",
    "latest_stage_artifact",
    "load_sealed_transaction",
    "load_stage_artifact",
    "publish_sealed_transaction",
    "publish_stage_artifact",
    "select_pilot_budget",
]
