"""Run the fixed TinyWorlds Phase 4 calibration without research switches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar


if TYPE_CHECKING:
    from apm.continual.tinyworlds_calibration import (
        CalibrationGateDecision,
        CalibrationIdentity,
        CalibrationValidationObservation,
        CalibrationValidationRequest,
        LockedCalibrationTestObservation,
        LockedCalibrationTestRequest,
        TinyWorldsCalibrationEvidence,
        TinyWorldsCalibrationResult,
    )
    from apm.continual.tinyworlds_calibration_run import (
        TinyWorldsAcceleratorCalibrationEvaluator,
        TinyWorldsCalibrationPool,
    )
    from apm.data.text.tinyworlds.query_generation import TinyWorldsBundle
    from apm.lm.text import TokenizersTextTokenizer
    from apm.lm.tinystories_conversion import LoadedTinyStoriesArtifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m"
RESULTS_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "language_cl"
    / "tinyworlds-v1"
    / "knowledge-graph"
)
PUBLIC_SEED = 0
_PROMOTION_MANIFEST = "bundle_manifest.json"
_PROMOTION_FORMAT = "apm.tinyworlds.calibration-result-bundle"
_PROMOTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _Phase:
    number: int
    name: str
    estimated_seconds: int


PHASES = (
    _Phase(1, "load the frozen TinyStories artifact and 36-fact world", 30),
    _Phase(2, "render the shared calibration story/query pool", 900),
    _Phase(3, "run or resume the fixed validation ladder and locked test", 36_000),
    _Phase(4, "validate and atomically promote the hashed profile", 120),
)


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the bar."""

    def close(self) -> None:
        """Close the bar."""

    def write(self, message: str) -> object:
        """Print without corrupting the bar."""


ResultT = TypeVar("ResultT")


class _JsonlJournal:
    """Batch append canonical progress and sequential result records."""

    def __init__(self, directory: Path, batch_size: int = 8) -> None:
        self.directory = directory
        self.batch_size = batch_size
        self.progress_path = directory / "progress.jsonl"
        self.results_path = directory / "sequential_results.jsonl"
        self._progress: list[dict[str, object]] = []
        self._results: list[dict[str, object]] = []
        self._sequence_index = 0

    def progress(self, record: dict[str, object]) -> None:
        self._progress.append(record)
        if len(self._progress) >= self.batch_size:
            self._flush(self.progress_path, self._progress)

    def result(self, record: dict[str, object]) -> None:
        self._results.append(
            {"sequence_index": self._sequence_index, **record}
        )
        self._sequence_index += 1
        if len(self._results) >= self.batch_size:
            self._flush(self.results_path, self._results)

    def flush(self) -> None:
        self._flush(self.progress_path, self._progress)
        self._flush(self.results_path, self._results)

    @staticmethod
    def _flush(path: Path, buffer: list[dict[str, object]]) -> None:
        if not buffer:
            return
        payload = "".join(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in buffer
        )
        with path.open("a", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        buffer.clear()


class _Progress:
    """Print phase lines and maintain phase/overall ETA bars."""

    def __init__(self, journal: _JsonlJournal) -> None:
        self.journal = journal
        self._overall: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None

    def __enter__(self) -> _Progress:
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="TinyWorlds calibration overall",
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
        if self._overall is not None:
            self._overall.close()

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one phase while emitting durable lifecycle events."""
        if self._overall is None or self._tqdm_factory is None:
            raise RuntimeError("calibration progress must be entered")
        line = f"Phase {phase.number}/{len(PHASES)}: {phase.name}"
        self._overall.write(line)
        started = monotonic()
        self.journal.progress(
            {
                "event": "phase_started",
                "name": phase.name,
                "phase": phase.number,
                "phase_count": len(PHASES),
            }
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
            args=(stop, phase_bar, self._overall, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            result = operation()
        except BaseException as error:
            stop.set()
            timer.join()
            phase_bar.close()
            self.journal.progress(
                {
                    "elapsed_seconds": monotonic() - started,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "event": "phase_failed",
                    "name": phase.name,
                    "phase": phase.number,
                    "phase_count": len(PHASES),
                }
            )
            self.journal.flush()
            raise
        stop.set()
        timer.join()
        remaining = max(0.0, phase.estimated_seconds - phase_bar.n)
        phase_bar.update(remaining)
        self._overall.update(remaining)
        phase_bar.close()
        self.journal.progress(
            {
                "elapsed_seconds": monotonic() - started,
                "event": "phase_completed",
                "name": phase.name,
                "phase": phase.number,
                "phase_count": len(PHASES),
            }
        )
        self.journal.flush()
        return result


@dataclass(frozen=True, slots=True)
class _PreparedWorld:
    base_artifact: LoadedTinyStoriesArtifact
    tokenizer: TokenizersTextTokenizer
    bundle: TinyWorldsBundle
    symbolic_bundle_sha256: str


def _prepare_world(temporary_directory: Path) -> _PreparedWorld:
    from apm.data.text.tinyworlds import (
        TINYWORLDS_VERSION,
        derive_master_seed,
        generate_calibration_bundle,
        write_tinyworlds_bundle,
    )
    from apm.lm.text import TokenizersTextTokenizer
    from apm.lm.tinystories_conversion import load_tinystories_artifact

    base_artifact = load_tinystories_artifact(BASE_ARTIFACT_DIRECTORY)
    tokenizer = TokenizersTextTokenizer.from_file(
        BASE_ARTIFACT_DIRECTORY / "tokenizer" / "tokenizer.json"
    )
    checkpoint = base_artifact.checkpoint
    master_seed = derive_master_seed(
        TINYWORLDS_VERSION,
        PUBLIC_SEED,
        checkpoint.reference.manifest_sha256,
        checkpoint.reference.parameter_checksum,
    )
    bundle = generate_calibration_bundle(master_seed, direct_facts_per_task=36)
    manifest = write_tinyworlds_bundle(
        bundle,
        temporary_directory / "symbolic-calibration-pool",
    )
    return _PreparedWorld(
        base_artifact,
        tokenizer,
        bundle,
        manifest.bundle_sha256,
    )


def _render_pool(prepared: _PreparedWorld) -> TinyWorldsCalibrationPool:
    from apm.continual.tinyworlds_calibration_run import (
        render_tinyworlds_calibration_pool,
    )

    return render_tinyworlds_calibration_pool(
        prepared.bundle,
        prepared.tokenizer,
        symbolic_bundle_sha256=prepared.symbolic_bundle_sha256,
    )


def _identity(prepared: _PreparedWorld) -> CalibrationIdentity:
    from apm.continual.tinyworlds_calibration import CalibrationIdentity
    from apm.data.text.tinyworlds import TINYWORLDS_VERSION

    checkpoint = prepared.base_artifact.checkpoint
    tokenizer_path = BASE_ARTIFACT_DIRECTORY / "tokenizer" / "tokenizer.json"
    return CalibrationIdentity(
        benchmark_version=TINYWORLDS_VERSION,
        public_seed=PUBLIC_SEED,
        calibration_bundle_sha256=prepared.symbolic_bundle_sha256,
        base_manifest_sha256=checkpoint.reference.manifest_sha256,
        base_parameter_checksum=checkpoint.reference.parameter_checksum,
        tokenizer_sha256=sha256(tokenizer_path.read_bytes()).hexdigest(),
    )


def _run_calibration(
    prepared: _PreparedWorld,
    pool: TinyWorldsCalibrationPool,
    journal: _JsonlJournal,
) -> tuple[TinyWorldsCalibrationResult, Path]:
    from apm.continual.tinyworlds_calibration import run_tinyworlds_calibration
    from apm.continual.tinyworlds_calibration_run import (
        CALIBRATION_EXECUTION_PRESET,
        TinyWorldsAcceleratorCalibrationEvaluator,
    )

    identity = _identity(prepared)
    cache_identity = sha256(
        json.dumps(
            {
                "base_manifest_sha256": identity.base_manifest_sha256,
                "bundle_sha256": identity.calibration_bundle_sha256,
                "execution": repr(CALIBRATION_EXECUTION_PRESET),
                "tokenizer_sha256": identity.tokenizer_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact_root = RESULTS_ROOT / f".calibration-cache-seed0-{cache_identity[:16]}"
    evaluator = TinyWorldsAcceleratorCalibrationEvaluator(
        identity,
        pool,
        prepared.tokenizer,
        prepared.base_artifact.checkpoint.reference,
        prepared.base_artifact.checkpoint.params,
        prepared.base_artifact.checkpoint.config,
        artifact_root,
        event_sink=journal.progress,
    )
    wrapped = _JournaledEvaluator(evaluator, journal)
    return run_tinyworlds_calibration(identity, wrapped), artifact_root


class _JournaledEvaluator:
    """Persist each protocol observation immediately after it is produced."""

    def __init__(
        self,
        evaluator: TinyWorldsAcceleratorCalibrationEvaluator,
        journal: _JsonlJournal,
    ) -> None:
        self.evaluator = evaluator
        self.journal = journal

    def evaluate_validation(
        self,
        request: CalibrationValidationRequest,
    ) -> CalibrationValidationObservation:
        observation = self.evaluator.evaluate_validation(request)
        evidence = observation.evidence
        self.journal.result(
            {
                "artifact_id": observation.artifact_id,
                "kind": "validation",
                "passed": all(
                    decision.passed
                    for decision in _gate_decisions(evidence)
                ),
                "purpose": request.purpose.value,
                "trial_index": request.trial_index,
            }
        )
        return observation

    def evaluate_locked_test(
        self,
        request: LockedCalibrationTestRequest,
    ) -> LockedCalibrationTestObservation:
        observation = self.evaluator.evaluate_locked_test(request)
        self.journal.result(
            {
                "artifact_id": observation.artifact_id,
                "kind": "locked_test",
                "validation_artifact_id": request.validation_artifact_id,
            }
        )
        return observation


def _gate_decisions(
    evidence: TinyWorldsCalibrationEvidence,
) -> tuple[CalibrationGateDecision, ...]:
    from apm.continual.tinyworlds_calibration import calibration_gate_decisions

    return calibration_gate_decisions(evidence)


def _promote(
    calibration_result: TinyWorldsCalibrationResult,
    artifact_root: Path,
    temporary_directory: Path,
) -> Path:
    from apm.continual.tinyworlds_calibration_profile import (
        calibration_profile_sha256,
        calibration_result_sha256,
        write_calibration_result,
        write_calibration_profile,
    )

    profile = calibration_result.profile
    profile_sha = None if profile is None else calibration_profile_sha256(profile)
    result_sha = calibration_result_sha256(calibration_result)
    target_prefix = (
        "calibration-seed0"
        if profile is not None
        else "calibration-stopped-seed0"
    )
    target = RESULTS_ROOT / f"{target_prefix}-{result_sha[:12]}"
    if target.exists():
        _validate_promoted_bundle(target, result_sha)
        return target.resolve()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=RESULTS_ROOT)
    )
    try:
        if profile is not None:
            write_calibration_profile(profile, temporary)
        write_calibration_result(calibration_result, temporary)
        validation_target = temporary / "validation"
        validation_target.mkdir()
        for trial in calibration_result.validation_trials:
            shutil.copytree(
                artifact_root / "validation" / trial.artifact_id,
                validation_target / trial.artifact_id,
            )
        if profile is not None:
            test_target = temporary / "test"
            test_target.mkdir()
            shutil.copytree(
                artifact_root / "test" / profile.locked_test.artifact_id,
                test_target / profile.locked_test.artifact_id,
            )
        shutil.copytree(
            temporary_directory / "symbolic-calibration-pool",
            temporary / "symbolic-calibration-pool",
        )
        for name in ("progress.jsonl", "sequential_results.jsonl"):
            source = temporary_directory / name
            if not source.is_file():
                raise RuntimeError(f"calibration journal is missing: {name}")
            shutil.copy2(source, temporary / name)
        _write_promotion_manifest(temporary, result_sha, profile_sha)
        _validate_promoted_bundle(temporary, result_sha)
        _fsync_directory_tree(temporary)
        os.rename(temporary, target)
        _fsync_directory(RESULTS_ROOT)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target.resolve()


def _write_promotion_manifest(
    directory: Path,
    result_sha256: str,
    profile_sha256: str | None,
) -> None:
    target = directory / _PROMOTION_MANIFEST
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"promotion manifest already exists: {target}")
    files = tuple(
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    )
    core = {
        "files": files,
        "format": _PROMOTION_FORMAT,
        "profile_sha256": profile_sha256,
        "result_sha256": result_sha256,
        "schema_version": _PROMOTION_SCHEMA_VERSION,
    }
    payload_sha256 = sha256(_canonical_json_bytes(core)).hexdigest()
    with target.open("wb") as output:
        output.write(
            _canonical_json_bytes(
                {**core, "payload_sha256": payload_sha256}
            )
        )
        output.flush()
        os.fsync(output.fileno())


def _validate_promoted_bundle(directory: Path, result_sha256: str) -> None:
    from apm.continual.tinyworlds_calibration_profile import (
        CALIBRATION_PROFILE_FILENAME,
        CALIBRATION_RESULT_FILENAME,
        calibration_artifact_tree_sha256,
        calibration_profile_sha256,
        load_calibration_result,
        load_calibration_result_sha256,
    )
    from apm.continual.tinyworlds_calibration_run import (
        CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
        _load_trial_artifact,
        _locked_test_request_record,
        _validation_request_record,
        load_calibration_trial_resource_evidence,
    )
    from apm.data.text.tinyworlds import load_tinyworlds_bundle
    from apm.data.text.tinyworlds.persistence import load_tinyworlds_manifest

    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("calibration result bundle must be a regular directory")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise RuntimeError("calibration result bundle must not contain symlinks")
    manifest_path = directory / _PROMOTION_MANIFEST
    payload = manifest_path.read_bytes()
    manifest = _load_strict_json_object(payload, "promotion manifest")
    if payload != _canonical_json_bytes(manifest):
        raise RuntimeError("promotion manifest is not canonical JSON")
    expected_fields = {
        "files",
        "format",
        "payload_sha256",
        "profile_sha256",
        "result_sha256",
        "schema_version",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("promotion manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if (
        manifest["format"] != _PROMOTION_FORMAT
        or manifest["schema_version"] != _PROMOTION_SCHEMA_VERSION
        or manifest["result_sha256"] != result_sha256
        or manifest["payload_sha256"]
        != sha256(_canonical_json_bytes(core)).hexdigest()
    ):
        raise RuntimeError("promotion manifest identity or checksum changed")
    raw_files = manifest["files"]
    if type(raw_files) is not list:
        raise RuntimeError("promotion manifest files must be a JSON array")
    records: list[dict[str, object]] = []
    for value in raw_files:
        if type(value) is not dict or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("promotion file record fields changed")
        records.append(value)
    listed_paths = tuple(record["path"] for record in records)
    if (
        any(type(value) is not str or not value for value in listed_paths)
        or tuple(sorted(listed_paths)) != listed_paths
        or len(set(listed_paths)) != len(listed_paths)
    ):
        raise RuntimeError("promotion file paths are not canonical")
    actual_paths = tuple(
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != manifest_path
    )
    if listed_paths != actual_paths:
        raise RuntimeError("promotion bundle file set changed")
    for record in records:
        relative = str(record["path"])
        path = directory / relative
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or path.stat().st_size != record["size_bytes"]
            or type(record["sha256"]) is not str
            or _file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"promotion file checksum changed: {relative}")
    if load_calibration_result_sha256(directory) != result_sha256:
        raise RuntimeError("promoted calibration result hash differs")
    result = load_calibration_result(directory)
    profile = result.profile
    expected_profile_sha = (
        None if profile is None else calibration_profile_sha256(profile)
    )
    if manifest["profile_sha256"] != expected_profile_sha:
        raise RuntimeError("promoted calibration profile hash differs")
    expected_top_level = {
        CALIBRATION_RESULT_FILENAME,
        _PROMOTION_MANIFEST,
        "progress.jsonl",
        "sequential_results.jsonl",
        "symbolic-calibration-pool",
        "validation",
    }
    if profile is not None:
        expected_top_level.update((CALIBRATION_PROFILE_FILENAME, "test"))
    if {path.name for path in directory.iterdir()} != expected_top_level:
        raise RuntimeError("promoted calibration top-level entries changed")
    expected_validation_ids = {
        trial.artifact_id
        for trial in result.validation_trials
    }
    validation_root = directory / "validation"
    if {path.name for path in validation_root.iterdir()} != expected_validation_ids:
        raise RuntimeError("promoted validation trial set changed")
    for trial in result.validation_trials:
        trial_root = validation_root / trial.artifact_id
        if calibration_artifact_tree_sha256(trial_root) != trial.artifact_sha256:
            raise RuntimeError(
                f"promoted validation artifact tree changed: {trial.artifact_id}"
            )
        resource_evidence = load_calibration_trial_resource_evidence(
            trial_root,
            expected_target_bytes=CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
        )
        evidence, _ = _load_trial_artifact(
            trial_root,
            trial.artifact_id,
            trial.execution_sha256,
            request_record=_validation_request_record(trial.request),
            runtime_resource_evidence=resource_evidence,
        )
        if evidence != trial.evidence:
            raise RuntimeError(
                f"promoted validation evidence changed: {trial.artifact_id}"
            )
    test_root = directory / "test"
    if profile is None:
        if test_root.exists() or test_root.is_symlink():
            raise RuntimeError("stopped calibration must not contain test artifacts")
    else:
        locked = profile.locked_test
        if {path.name for path in test_root.iterdir()} != {locked.artifact_id}:
            raise RuntimeError("promoted locked-test trial set changed")
        locked_root = test_root / locked.artifact_id
        if calibration_artifact_tree_sha256(locked_root) != locked.artifact_sha256:
            raise RuntimeError("promoted locked-test artifact tree changed")
        resource_evidence = load_calibration_trial_resource_evidence(
            locked_root,
            expected_target_bytes=CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
        )
        evidence, _ = _load_trial_artifact(
            locked_root,
            locked.artifact_id,
            locked.execution_sha256,
            request_record=_locked_test_request_record(locked.request),
            runtime_resource_evidence=resource_evidence,
        )
        if evidence != locked.evidence:
            raise RuntimeError("promoted locked-test evidence changed")
    symbolic_root = directory / "symbolic-calibration-pool"
    symbolic_manifest = load_tinyworlds_manifest(symbolic_root)
    load_tinyworlds_bundle(symbolic_root)
    if symbolic_manifest.bundle_sha256 != (
        result.identity.calibration_bundle_sha256
    ):
        raise RuntimeError("promoted calibration world identity changed")


def _advance_eta_bars(
    stop: Event,
    phase_bar: _TqdmBar,
    overall_bar: _TqdmBar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds:
            phase_bar.update(1.0)
            overall_bar.update(1.0)


def _fsync_directory_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as source:
                os.fsync(source.fileno())
    for path in sorted(
        (value for value in directory.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ) + [directory]:
        _fsync_directory(path)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(record: dict[str, object]) -> bytes:
    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_strict_json_object(payload: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise RuntimeError(f"duplicate field in {label}: {key}")
            record[key] = value
        return record

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def main() -> None:
    """Execute the one canonical calibration workflow."""
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="tinyworlds-calibration-")
    ).resolve()
    print(f"Temporary artifact directory: {temporary_directory}", flush=True)
    journal = _JsonlJournal(temporary_directory)
    with _Progress(journal) as progress:
        prepared = progress.run(
            PHASES[0],
            lambda: _prepare_world(temporary_directory),
        )
        pool = progress.run(PHASES[1], lambda: _render_pool(prepared))
        result, artifact_root = progress.run(
            PHASES[2],
            lambda: _run_calibration(prepared, pool, journal),
        )
        final_directory = progress.run(
            PHASES[3],
            lambda: _promote(
                result,
                artifact_root,
                temporary_directory,
            ),
        )
    journal.flush()
    print(f"Calibration result: {final_directory}", flush=True)
    if result.profile is not None:
        print(f"Calibration profile: {final_directory}", flush=True)
    else:
        if result.stop_reason is None:
            raise RuntimeError("stopped calibration did not record a reason")
        print(f"Calibration stopped: {result.stop_reason.value}", flush=True)


if __name__ == "__main__":
    main()
