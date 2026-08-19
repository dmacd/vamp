#!/usr/bin/env python3
"""Run the fixed TinyWorlds nouns-v2 joint-IID LoRA rank sweep."""

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
    _Phase(1, "authenticate parent controls, checkpoints, and ledgers", 75),
    _Phase(2, "materialize the exact 4,440 parent validation cases", 35),
    _Phase(3, "train or strict-load ranks 4, 8, 16, and 32", 7_600),
    _Phase(4, "evaluate every new rank on the canonical suffixes", 720),
    _Phase(5, "bootstrap, publish, replay, and enforce immutability", 75),
)
TRAINING_ESTIMATES = {4: 3_900, 8: 0, 16: 1_700, 32: 2_000}
EVALUATION_ESTIMATES = {4: 240, 8: 0, 16: 240, 32: 240}


class _Bar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the progress bar."""

    def close(self) -> None:
        """Close the progress bar."""

    def set_postfix(self, ordered_dict: Mapping[str, object] | None = None, **values: object) -> object:
        """Update progress metrics."""

    def write(self, message: str) -> object:
        """Print without corrupting active bars."""


ResultT = TypeVar("ResultT")


class _Progress:
    """Render exact rank bars plus phase and overall live ETAs."""

    def __init__(self) -> None:
        self._overall: _Bar | None = None
        self._rank_bar: _Bar | None = None
        self._active_rank: int | None = None
        self._rank_completed: dict[tuple[str, int], int] = {}

    def __enter__(self) -> _Progress:
        from tqdm.auto import tqdm

        self._tqdm = tqdm
        self._overall = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="Joint-IID rank sweep overall ETA",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
            mininterval=5.0,
        )
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if self._rank_bar is not None:
            self._rank_bar.close()
        if self._overall is not None:
            self._overall.close()

    def run_estimated(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run a short phase while advancing phase and overall estimated time."""
        if self._overall is None:
            raise RuntimeError("rank-sweep progress is not active")
        self._overall.write(f"Phase {phase.number}/{len(PHASES)}: {phase.description}.")
        phase_bar = self._tqdm(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)} ETA",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
            mininterval=5.0,
        )
        stop = Event()
        timer = Thread(
            target=_advance_estimates,
            args=(stop, phase_bar, self._overall, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            return operation()
        finally:
            stop.set()
            timer.join()
            remainder = max(0.0, phase.estimated_seconds - phase_bar.n)
            phase_bar.update(remainder)
            self._overall.update(remainder)
            phase_bar.close()

    def begin_dynamic(self, phase: _Phase) -> None:
        """Announce a long phase whose callbacks provide exact coverage."""
        if self._overall is None:
            raise RuntimeError("rank-sweep progress is not active")
        self._overall.write(f"Phase {phase.number}/{len(PHASES)}: {phase.description}.")

    def finish_dynamic(self, kind: str, phase: _Phase) -> None:
        """Close the current rank bar and fill any unused phase estimate."""
        if self._rank_bar is not None:
            self._rank_bar.close()
            self._rank_bar = None
        self._active_rank = None
        estimates = TRAINING_ESTIMATES if kind == "train" else EVALUATION_ESTIMATES
        accounted = sum(
            estimates[rank]
            * min(1.0, self._rank_completed.get((kind, rank), 0) / max(1, total))
            for rank, total in (
                (rank, 15_024 if kind == "train" else 4_440)
                for rank in estimates
            )
        )
        remainder = max(0.0, phase.estimated_seconds - accounted)
        if self._overall is not None:
            self._overall.update(remainder)

    def training_update(
        self,
        rank: int,
        completed: int,
        total: int,
        loss: float,
        elapsed: float,
    ) -> None:
        """Advance one rank's exact optimizer-update bar and overall estimate."""
        self._rank_update(
            "train",
            rank,
            completed,
            total,
            TRAINING_ESTIMATES[rank],
            {"loss": f"{loss:.4f}", "elapsed-min": f"{elapsed / 60:.1f}"},
            "updates",
        )

    def evaluation_update(
        self,
        rank: int,
        completed: int,
        total: int,
        metrics: Mapping[str, float],
    ) -> None:
        """Advance one rank's exact suffix-story bar and overall estimate."""
        postfix = {
            key.replace("story_nll", "NLL"): f"{value:.4f}"
            for key, value in metrics.items()
            if key in ("story_nll", "token_accuracy")
        }
        self._rank_update(
            "eval",
            rank,
            completed,
            total,
            EVALUATION_ESTIMATES[rank],
            postfix,
            "stories",
        )

    def _rank_update(
        self,
        kind: str,
        rank: int,
        completed: int,
        total: int,
        estimated_seconds: int,
        postfix: Mapping[str, object],
        unit: str,
    ) -> None:
        if self._overall is None:
            raise RuntimeError("rank-sweep progress is not active")
        key = (kind, rank)
        previous = self._rank_completed.get(key, 0)
        if self._active_rank != rank:
            if self._rank_bar is not None:
                self._rank_bar.close()
            self._rank_bar = self._tqdm(
                total=total,
                initial=min(previous, completed),
                desc=f"Rank {rank} {'training' if kind == 'train' else 'evaluation'} ETA",
                unit=unit,
                position=1,
                dynamic_ncols=True,
                leave=False,
                mininterval=5.0,
            )
            self._active_rank = rank
        delta = max(0, completed - previous)
        if self._rank_bar is not None and completed > self._rank_bar.n:
            self._rank_bar.update(completed - self._rank_bar.n)
            if postfix and (completed == total or completed % 32 == 0):
                self._rank_bar.set_postfix(postfix)
        self._overall.update(estimated_seconds * delta / max(1, total))
        self._rank_completed[key] = max(previous, completed)


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
    """Execute or exactly replay the sole GPU-zero rank-sweep configuration."""
    import jax

    from apm.data.text.tinyworlds_nouns_v2.contracts import TASK_IDS
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        build_validation_cases,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
        ALLOCATOR_LIMIT_BYTES,
        analyze_joint_iid_rank_sweep,
        assert_rank_sweep_inputs_unchanged,
        authenticate_joint_iid_rank_sweep_inputs,
        run_or_resume_rank_evaluation,
        run_or_resume_rank_training,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep_report import (
        publish_joint_iid_rank_sweep_report,
    )

    started = monotonic()
    timings: dict[str, float] = {}
    with _Progress() as progress:
        phase_started = monotonic()
        inputs = progress.run_estimated(
            PHASES[0],
            lambda: authenticate_joint_iid_rank_sweep_inputs(REPOSITORY_ROOT),
        )
        timings["authentication_seconds"] = monotonic() - phase_started
        print(f"Persistent temporary directory: {inputs.work_directory}", flush=True)
        completion_path = inputs.result_directory / "completion.json"
        published = completion_path.is_file()
        if published:
            _load_measurement(completion_path, "completion", inputs.contract_sha256)
        prior_snapshot = _publication_snapshot(inputs.result_directory) if published else ()

        phase_started = monotonic()
        cases_by_task = dict(
            progress.run_estimated(
                PHASES[1],
                lambda: build_validation_cases(inputs.parent),
            )
        )
        cases = tuple(
            case for task_id in TASK_IDS for case in cases_by_task[task_id]
        )
        timings["materialization_seconds"] = monotonic() - phase_started

        progress.begin_dynamic(PHASES[2])
        phase_started = monotonic()
        artifacts = run_or_resume_rank_training(
            inputs,
            progress=progress.training_update,
        )
        timings["training_wall_seconds"] = monotonic() - phase_started
        progress.finish_dynamic("train", PHASES[2])

        progress.begin_dynamic(PHASES[3])
        phase_started = monotonic()
        ledgers = run_or_resume_rank_evaluation(
            inputs,
            artifacts,
            cases,
            progress=progress.evaluation_update,
        )
        timings["evaluation_wall_seconds"] = monotonic() - phase_started
        progress.finish_dynamic("eval", PHASES[3])

        def analyze_and_publish() -> Path:
            measurement_paths = (
                inputs.result_directory / "execution.json",
                inputs.result_directory / "allocator.json",
            )
            if all(path.is_file() for path in measurement_paths):
                execution = dict(
                    _load_measurement(
                        measurement_paths[0],
                        "execution",
                        inputs.contract_sha256,
                    )["durations_seconds"]
                )
                allocator = _load_measurement(
                    measurement_paths[1],
                    "allocator",
                    inputs.contract_sha256,
                )
            else:
                execution = {
                    **timings,
                    "end_to_end_seconds": monotonic() - started,
                    "evaluated_rows": 4.0 * len(cases),
                    "new_training_jobs": 3.0,
                }
                allocator = _allocator_measurement(jax, ALLOCATOR_LIMIT_BYTES)
                _publish_measurement(
                    measurement_paths[0],
                    inputs.contract_sha256,
                    "execution",
                    {"durations_seconds": execution},
                )
                _publish_measurement(
                    measurement_paths[1],
                    inputs.contract_sha256,
                    "allocator",
                    {
                        key: value
                        for key, value in allocator.items()
                        if key not in ("contract_sha256", "format", "result_sha256")
                    },
                )
            analysis = analyze_joint_iid_rank_sweep(
                inputs,
                artifacts,
                ledgers,
                execution={str(key): float(value) for key, value in execution.items()},
                allocator=allocator,
            )
            report, _, _ = publish_joint_iid_rank_sweep_report(inputs, analysis)
            first = _publication_snapshot(inputs.result_directory)
            publish_joint_iid_rank_sweep_report(inputs, analysis)
            if _publication_snapshot(inputs.result_directory) != first:
                raise RuntimeError("rank-sweep report regeneration changed bytes")
            if published and first != prior_snapshot:
                raise RuntimeError("published rank-sweep replay changed bytes")
            assert_rank_sweep_inputs_unchanged(inputs)
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

        report = progress.run_estimated(PHASES[4], analyze_and_publish)
    _notify_completion(report)
    print(f"Complete report: {report}", flush=True)
    return 0


def _allocator_measurement(jax_module: object, limit: int) -> dict[str, object]:
    devices = tuple(jax_module.local_devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("joint-IID rank sweep requires JAX GPU 0")
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
        "format": f"tinyworlds-nouns-v2-joint-iid-lora-rank-sweep-{kind}-v1",
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
        != f"tinyworlds-nouns-v2-joint-iid-lora-rank-sweep-{kind}-v1"
        or record.get("contract_sha256") != contract_sha256
        or record.get("result_sha256") != record_sha256(core)
    ):
        raise ValueError(f"published rank-sweep {kind} measurement changed")
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
        raise ValueError("published rank-sweep manifest identity changed")
    return record


def _notify_completion(report: Path) -> None:
    try:
        subprocess.run(
            (
                "notify-send",
                "TinyWorlds nouns-v2 joint-IID LoRA rank sweep complete",
                str(report),
            ),
            check=False,
        )
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
