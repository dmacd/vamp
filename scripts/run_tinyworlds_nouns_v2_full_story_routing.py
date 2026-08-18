#!/usr/bin/env python3
"""Run the fixed TinyWorlds nouns-v2 full-story routing diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import monotonic
from typing import Protocol, TypeVar


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
_existing_xla_flags = tuple(
    flag
    for flag in os.environ.get("XLA_FLAGS", "").split()
    if not flag.startswith("--xla_gpu_enable_command_buffer")
)
os.environ["XLA_FLAGS"] = " ".join(
    (*_existing_xla_flags, "--xla_gpu_enable_command_buffer=")
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _Phase:
    number: int
    description: str
    estimated_seconds: int


PHASES = (
    _Phase(1, "authenticate the parent publication, ledgers, and final banks", 75),
    _Phase(2, "resume the bounded canonical GPU audit", 180),
    _Phase(3, "verify, derive 13,320 paired rows, and bootstrap", 35),
    _Phase(4, "publish, exactly regenerate, and enforce immutability", 30),
)


class _Bar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the bar."""

    def close(self) -> None:
        """Close the bar."""

    def write(self, message: str) -> object:
        """Print without corrupting bars."""


ResultT = TypeVar("ResultT")


class _Progress:
    """Display phase/overall ETAs and the exact direct-audit coverage."""

    def __init__(self) -> None:
        self._overall: _Bar | None = None
        self._phase: _Bar | None = None
        self._audit: _Bar | None = None

    def __enter__(self) -> _Progress:
        from tqdm.auto import tqdm

        self._overall = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="Full-story routing overall ETA",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
        )
        self._tqdm = tqdm
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if self._audit is not None:
            self._audit.close()
        if self._phase is not None:
            self._phase.close()
        if self._overall is not None:
            self._overall.close()

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one operation while advancing estimated phase and overall ETAs."""
        if self._overall is None:
            raise RuntimeError("full-story progress is not active")
        self._overall.write(f"Phase {phase.number}/{len(PHASES)}: {phase.description}.")
        self._phase = self._tqdm(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)} ETA",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )
        stop = Event()
        timer = Thread(
            target=_advance_estimates,
            args=(stop, self._phase, self._overall, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            return operation()
        finally:
            stop.set()
            timer.join()
            remaining = max(0.0, phase.estimated_seconds - self._phase.n)
            self._phase.update(remaining)
            self._overall.update(remaining)
            self._phase.close()
            self._phase = None

    def audit_update(
        self,
        completed: int,
        total: int,
        metrics: Mapping[str, float],
    ) -> None:
        """Display exact flushed direct-score rows and measured ETA."""
        del metrics
        if self._audit is None:
            self._audit = self._tqdm(
                total=total,
                initial=completed,
                desc="Direct canonical audit",
                unit="bank-stories",
                position=2,
                dynamic_ncols=True,
                leave=False,
            )
        elif completed > self._audit.n:
            self._audit.update(completed - self._audit.n)
        if completed == total:
            self._audit.close()
            self._audit = None


def _advance_estimates(
    stop: Event,
    phase_bar: _Bar,
    overall_bar: _Bar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds - 1:
            phase_bar.update(1)
            overall_bar.update(1)


def main() -> int:
    """Execute or exactly replay the sole GPU-zero diagnostic configuration."""
    import jax

    from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
    from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing import (
        ALLOCATOR_LIMIT_BYTES,
        analyze_full_story_routing,
        assert_parent_unchanged,
        authenticate_full_story_routing_inputs,
        run_or_resume_case_derivation,
        run_or_resume_direct_audit,
        verify_direct_audit,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing_report import (
        publish_full_story_routing_report,
    )

    started = monotonic()
    with _Progress() as progress:
        inputs = progress.run(
            PHASES[0],
            lambda: authenticate_full_story_routing_inputs(REPOSITORY_ROOT),
        )
        print(f"Persistent temporary directory: {inputs.work_directory}", flush=True)
        completion_path = inputs.result_directory / "completion.json"
        published = completion_path.is_file()
        if published:
            _load_measurement(completion_path, "completion", inputs.contract_sha256)
        prior_snapshot = _publication_snapshot(inputs.result_directory) if published else ()
        direct_path = progress.run(
            PHASES[1],
            lambda: run_or_resume_direct_audit(inputs, progress=progress.audit_update),
        )

        def analyze() -> tuple[Path, dict[str, object], dict[str, object]]:
            audit = verify_direct_audit(inputs, direct_path)
            cases = run_or_resume_case_derivation(inputs, direct_path)
            return cases, audit, analyze_full_story_routing(inputs, cases, audit)

        case_path, audit, analysis = progress.run(PHASES[2], analyze)
        del case_path, audit

        def publish() -> Path:
            measurement_paths = (
                inputs.result_directory / "execution.json",
                inputs.result_directory / "allocator.json",
            )
            if all(path.is_file() for path in measurement_paths):
                execution = _load_measurement(
                    measurement_paths[0],
                    "execution",
                    inputs.contract_sha256,
                )["durations_seconds"]
                allocator = _load_measurement(
                    measurement_paths[1],
                    "allocator",
                    inputs.contract_sha256,
                )
            else:
                allocator = _allocator_measurement(jax, ALLOCATOR_LIMIT_BYTES)
                execution = {
                    "end_to_end_seconds": monotonic() - started,
                    "source_rows": 13_320.0,
                    "direct_rows": 570.0,
                }
                _publish_measurement(
                    inputs.result_directory / "execution.json",
                    inputs.contract_sha256,
                    "execution",
                    {"durations_seconds": execution},
                )
                _publish_measurement(
                    inputs.result_directory / "allocator.json",
                    inputs.contract_sha256,
                    "allocator",
                    {
                        key: value
                        for key, value in allocator.items()
                        if key not in ("contract_sha256", "format", "result_sha256")
                    },
                )
            report, _, _ = publish_full_story_routing_report(
                inputs,
                analysis,
                execution={str(key): float(value) for key, value in dict(execution).items()},
                allocator=allocator,
            )
            first = _publication_snapshot(inputs.result_directory)
            publish_full_story_routing_report(
                inputs,
                analysis,
                execution={str(key): float(value) for key, value in dict(execution).items()},
                allocator=allocator,
            )
            if _publication_snapshot(inputs.result_directory) != first:
                raise RuntimeError("full-story report regeneration changed bytes")
            if published and first != prior_snapshot:
                raise RuntimeError("published full-story replay changed bytes")
            assert_parent_unchanged(inputs)
            manifest = _load_manifest_identity(
                inputs.result_directory / "manifest.json",
                inputs.contract_sha256,
            )
            _publish_measurement(
                completion_path,
                inputs.contract_sha256,
                "completion",
                {"manifest_sha256": manifest["manifest_sha256"], "status": "complete"},
            )
            return report

        report = progress.run(PHASES[3], publish)
    _notify_completion(report)
    print(f"Complete report: {report}", flush=True)
    return 0


def _allocator_measurement(jax_module: object, limit: int) -> dict[str, object]:
    devices = tuple(jax_module.local_devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("full-story routing diagnostic requires JAX GPU 0")
    statistics = tuple(device.memory_stats() or {} for device in devices)
    peak = max(
        int(values.get("peak_bytes_in_use", values.get("bytes_in_use", 0)))
        for values in statistics
    )
    if peak <= 0 or peak > limit:
        raise RuntimeError(f"allocator peak {peak} violates the fixed {limit}-byte gate")
    return {
        "allocator_limit_bytes": limit,
        "device_kind": [str(device.device_kind) for device in devices],
        "device_platform": "gpu",
        "peak_bytes_in_use": peak,
    }


def _publish_measurement(
    path: Path,
    contract_sha256: str,
    kind: str,
    values: Mapping[str, object],
) -> dict[str, object]:
    from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        publish_immutable_json,
    )

    core = {
        "contract_sha256": contract_sha256,
        "format": f"tinyworlds-nouns-v2-temporal-full-story-routing-{kind}-v1",
        **dict(values),
    }
    record = {**core, "result_sha256": record_sha256(core)}
    publish_immutable_json(path, record)
    return record


def _load_measurement(
    path: Path,
    kind: str,
    contract_sha256: str,
) -> dict[str, object]:
    from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        load_canonical_json,
    )

    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "result_sha256"}
    if (
        record.get("format")
        != f"tinyworlds-nouns-v2-temporal-full-story-routing-{kind}-v1"
        or record.get("contract_sha256") != contract_sha256
        or record.get("result_sha256") != record_sha256(core)
    ):
        raise ValueError(f"published full-story {kind} measurement changed")
    return record


def _publication_snapshot(directory: Path) -> tuple[tuple[str, str], ...]:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        file_sha256,
        load_canonical_json,
    )

    manifest = load_canonical_json(directory / "manifest.json")
    return tuple(
        (relative, file_sha256(directory / relative))
        for relative in sorted((*dict(manifest["artifacts"]), "manifest.json"))
    )


def _load_manifest_identity(path: Path, contract_sha256: str) -> dict[str, object]:
    from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        load_canonical_json,
    )

    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "manifest_sha256"}
    if (
        record.get("contract_sha256") != contract_sha256
        or record.get("manifest_sha256") != record_sha256(core)
    ):
        raise ValueError("published full-story manifest identity changed")
    return record


def _notify_completion(report: Path) -> None:
    try:
        subprocess.run(
            (
                "notify-send",
                "TinyWorlds nouns-v2 full-story routing complete",
                str(report),
            ),
            check=False,
        )
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
