#!/usr/bin/env python3
"""Run the sole TinyWorlds nouns-v2 log-t temporal consolidation study."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from time import monotonic


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


def main() -> int:
    """Execute or exactly replay the fixed GPU-zero temporal study."""
    import jax

    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        assert_canonical_artifacts_unchanged,
        authenticate_temporal_study_inputs,
        build_validation_cases,
        run_or_resume_final_controls,
        run_or_resume_order_evaluation,
        run_or_resume_order_training,
        run_or_resume_shared_training,
        study_dashboard_jobs,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
        ALLOCATOR_LIMIT_BYTES,
        TEMPORAL_ORDERS,
        record_sha256,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
        start_dashboard_server,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_distortion import (
        run_or_resume_merge_distortion,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        load_canonical_json,
        publish_immutable_json,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_report import (
        publish_temporal_consolidation_report,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing import (
        expected_timing_shapes,
        maximum_timing_allocator_peak_bytes,
        run_or_resume_isolated_timing_audit,
    )

    started = monotonic()
    print("Phase 1: authenticate canonical nouns-v2 inputs and ledgers.", flush=True)
    inputs = authenticate_temporal_study_inputs(REPOSITORY_ROOT)
    jobs = study_dashboard_jobs(inputs)
    recorder = _ConsoleProgressRecorder(
        inputs.work_directory,
        inputs.contract_sha256,
        jobs,
    )
    published_manifest = inputs.result_directory / "manifest.json"
    replay = published_manifest.is_file()
    if not replay:
        recorder.update(
            "authenticate",
            1,
            status="complete",
            elapsed_seconds=monotonic() - started,
        )
    server = start_dashboard_server(inputs.work_directory, inputs.result_directory)
    print(f"Live dashboard: {server.url}", flush=True)
    print(f"Persistent temporary directory: {inputs.work_directory}", flush=True)
    active_recorder = _ReadOnlyProgress(recorder) if replay else recorder
    try:
        print("Phase 2: materialize isolated midpoint cases.", flush=True)
        phase_started = monotonic()
        cases = build_validation_cases(inputs)
        active_recorder.update(
            "materialize",
            1,
            status="complete",
            elapsed_seconds=monotonic() - phase_started,
        )

        print("Phase 3: train or strict-load shared adapters and IID controls.", flush=True)
        shared = run_or_resume_shared_training(inputs, active_recorder)

        print("Phase 4: train or strict-load both temporal orderings.", flush=True)
        orderings = tuple(
            run_or_resume_order_training(
                inputs,
                shared,
                order,
                active_recorder,
            )
            for order in TEMPORAL_ORDERS
        )

        print("Phase 5: time every observed inference shape.", flush=True)
        timing_path = run_or_resume_isolated_timing_audit(
            inputs,
            orderings,
            dict(cases),
            progress=_bounded_progress(active_recorder, "timing"),
        )
        active_recorder.update(
            "timing",
            len(expected_timing_shapes(inputs)),
            status="complete",
            elapsed_seconds=_job_elapsed(active_recorder, "timing"),
        )

        print("Phase 6: audit every merge and lineage increment.", flush=True)
        guarded_progress = _AllocatorGuardProgress(
            active_recorder,
            jax,
            ALLOCATOR_LIMIT_BYTES,
        )
        distortion_paths = {
            ordering.order: run_or_resume_merge_distortion(
                inputs,
                ordering,
                dict(cases),
                progress=_bounded_progress(
                    guarded_progress,
                    f"distortion-{ordering.order}",
                ),
            )
            for ordering in orderings
        }
        for ordering in orderings:
            job_id = f"distortion-{ordering.order}"
            active_recorder.update(
                job_id,
                _job_total(active_recorder, job_id),
                status="complete",
                elapsed_seconds=_job_elapsed(active_recorder, job_id),
            )

        print("Phase 7: stream sentinel, macro, and final-control evaluation.", flush=True)
        evaluation_directories = {
            ordering.order: run_or_resume_order_evaluation(
                inputs,
                shared,
                ordering,
                dict(cases),
                guarded_progress,
            )
            for ordering in orderings
        }
        final_control_directory = run_or_resume_final_controls(
            inputs,
            shared,
            dict(cases),
            guarded_progress,
        )

        measured_allocator = _allocator_measurement(jax, ALLOCATOR_LIMIT_BYTES)
        measured_allocator["peak_bytes_in_use"] = max(
            int(measured_allocator["peak_bytes_in_use"]),
            maximum_timing_allocator_peak_bytes(
                timing_path,
                inputs.contract_sha256,
            ),
        )
        allocator_core = {
            "contract_sha256": inputs.contract_sha256,
            "format": "tinyworlds-nouns-v2-temporal-consolidation-allocator-v1",
            **measured_allocator,
        }
        allocator_path = inputs.result_directory / "allocator.json"
        if allocator_path.is_file():
            allocator_record = load_canonical_json(allocator_path)
            allocator_record_core = {
                key: value
                for key, value in allocator_record.items()
                if key != "result_sha256"
            }
            if (
                allocator_record.get("contract_sha256") != inputs.contract_sha256
                or allocator_record.get("result_sha256")
                != record_sha256(allocator_record_core)
            ):
                raise ValueError("published temporal allocator measurement changed")
            allocator = {
                key: allocator_record[key]
                for key in (
                    "allocator_limit_bytes",
                    "device_kind",
                    "device_platform",
                    "peak_bytes_in_use",
                )
            }
        else:
            publish_immutable_json(
                allocator_path,
                {**allocator_core, "result_sha256": record_sha256(allocator_core)},
            )
            allocator = measured_allocator

        print("Phase 8: publish reports, verify immutability, and replay reports.", flush=True)
        execution_path = inputs.result_directory / "execution.json"
        if replay:
            original_snapshot = _publication_snapshot(inputs.result_directory)
            execution = _load_execution(execution_path, inputs.contract_sha256)
            report, _, _ = publish_temporal_consolidation_report(
                inputs,
                shared,
                orderings,
                evaluation_directories,
                final_control_directory,
                distortion_paths,
                timing_path,
                recorder,
                execution=execution,
                allocator=allocator,
            )
            if _publication_snapshot(inputs.result_directory) != original_snapshot:
                raise RuntimeError("published temporal report replay changed bytes")
        else:
            report_started = monotonic()
            publish_temporal_consolidation_report(
                inputs,
                shared,
                orderings,
                evaluation_directories,
                final_control_directory,
                distortion_paths,
                timing_path,
                recorder,
                execution=_execution_summary(recorder, monotonic() - started),
                allocator=allocator,
            )
            immutability_started = monotonic()
            assert_canonical_artifacts_unchanged(inputs)
            recorder.update(
                "immutability",
                1,
                status="complete",
                elapsed_seconds=monotonic() - immutability_started,
            )
            recorder.update(
                "report",
                1,
                status="complete",
                elapsed_seconds=monotonic() - report_started,
            )
            execution = _execution_summary(recorder, monotonic() - started)
            execution_core = {
                "contract_sha256": inputs.contract_sha256,
                "durations_seconds": execution,
                "format": "tinyworlds-nouns-v2-temporal-consolidation-execution-v1",
            }
            publish_immutable_json(
                execution_path,
                {
                    **execution_core,
                    "result_sha256": record_sha256(execution_core),
                },
            )
            report, _, _ = publish_temporal_consolidation_report(
                inputs,
                shared,
                orderings,
                evaluation_directories,
                final_control_directory,
                distortion_paths,
                timing_path,
                recorder,
                execution=execution,
                allocator=allocator,
            )
            first_snapshot = _publication_snapshot(inputs.result_directory)
            publish_temporal_consolidation_report(
                inputs,
                shared,
                orderings,
                evaluation_directories,
                final_control_directory,
                distortion_paths,
                timing_path,
                recorder,
                execution=execution,
                allocator=allocator,
            )
            if _publication_snapshot(inputs.result_directory) != first_snapshot:
                raise RuntimeError("temporal report regeneration is not byte-identical")
        assert_canonical_artifacts_unchanged(inputs)
        _notify_completion(report)
        print(f"Complete report: {report}", flush=True)
        return 0
    except BaseException as error:
        if not replay:
            recorder.fail_current(str(error))
        raise
    finally:
        recorder.close()
        server.stop()


class _ConsoleProgressRecorder:
    """ProgressRecorder with one exact job bar and one weighted overall bar."""

    def __init__(self, work_directory: Path, contract_sha256: str, jobs) -> None:
        from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
            ProgressRecorder,
        )
        from tqdm.auto import tqdm

        self._recorder = ProgressRecorder(work_directory, contract_sha256, jobs)
        self.jobs = self._recorder.jobs
        self.ledger = self._recorder.ledger
        self._tqdm = tqdm
        snapshot = self._recorder.snapshot()
        self._overall = tqdm(
            total=1000,
            initial=int(1000 * float(snapshot["overall_fraction"])),
            desc="Temporal study overall",
            unit="‰",
            dynamic_ncols=True,
            position=0,
        )
        self._job_bar = None
        self._job_id = None

    def update(self, job_id: str, completed: int, **keywords):
        """Persist an update and refresh exact job/weighted overall bars."""
        row = self._recorder.update(
            job_id,
            completed,
            ignore_stale_replay=True,
            **keywords,
        )
        if self._job_id != job_id:
            if self._job_bar is not None:
                self._job_bar.close()
            job = next(value for value in self.jobs if value.job_id == job_id)
            self._job_bar = self._tqdm(
                total=job.total,
                initial=min(completed, job.total),
                desc=f"{job.phase}: {job_id}",
                unit=job.unit,
                dynamic_ncols=True,
                position=1,
                leave=False,
            )
            self._job_id = job_id
        elif self._job_bar is not None and completed > self._job_bar.n:
            self._job_bar.update(completed - self._job_bar.n)
        fraction = int(1000 * float(self._recorder.snapshot()["overall_fraction"]))
        if fraction > self._overall.n:
            self._overall.update(fraction - self._overall.n)
        return row

    def snapshot(self):
        """Return the authenticated dashboard projection."""
        return self._recorder.snapshot()

    @property
    def work_directory(self):
        return self._recorder.work_directory

    @property
    def status_path(self):
        return self._recorder.status_path

    def close(self) -> None:
        """Close terminal progress bars."""
        if self._job_bar is not None:
            self._job_bar.close()
        self._overall.close()

    def fail_current(self, message: str) -> None:
        """Persist a failure for the currently visible job, when known."""
        latest = {
            str(job["job_id"]): str(job["status"])
            for job in self._recorder.snapshot()["jobs"]
        }
        if self._job_id is not None and latest.get(self._job_id) == "running":
            self._recorder.fail(
                self._job_id,
                message,
                _job_elapsed(self, self._job_id),
            )


class _ReadOnlyProgress:
    """Suppress progress mutations during an exact published-run replay."""

    def __init__(self, recorder: _ConsoleProgressRecorder) -> None:
        self._recorder = recorder
        self.jobs = recorder.jobs
        self.ledger = recorder.ledger

    def update(self, job_id: str, completed: int, **keywords):
        del job_id, completed, keywords
        return {}


class _AllocatorGuardProgress:
    """Reject a live evaluation update once its allocator peak exceeds the gate."""

    def __init__(self, recorder, jax_module, allocator_limit: int) -> None:
        self._recorder = recorder
        self._jax = jax_module
        self._allocator_limit = allocator_limit
        self.jobs = recorder.jobs
        self.ledger = recorder.ledger

    def update(self, job_id: str, completed: int, **keywords):
        row = self._recorder.update(job_id, completed, **keywords)
        _allocator_measurement(self._jax, self._allocator_limit)
        return row


def _bounded_progress(recorder, job_id: str):
    started = monotonic()
    elapsed_offset = _job_elapsed(recorder, job_id)

    def update(completed: int, total: int, metrics) -> None:
        del total
        recorder.update(
            job_id,
            completed,
            status="running",
            elapsed_seconds=elapsed_offset + monotonic() - started,
            metrics=metrics,
        )

    return update


def _job_total(recorder, job_id: str) -> int:
    return next(job.total for job in recorder.jobs if job.job_id == job_id)


def _job_elapsed(recorder, job_id: str) -> float:
    rows = tuple(row for row in recorder.ledger.rows if row.get("job_id") == job_id)
    return float(rows[-1]["elapsed_seconds"]) if rows else 0.0


def _allocator_measurement(jax_module, limit: int) -> dict[str, object]:
    devices = tuple(jax_module.local_devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("temporal consolidation requires GPU-backed JAX on GPU 0")
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


def _execution_summary(recorder, end_to_end_seconds: float) -> dict[str, float]:
    snapshot = recorder.snapshot()
    durations = {
        str(job["job_id"]): float(job["elapsed_seconds"])
        for job in snapshot["jobs"]
    }
    return {**durations, "end_to_end_seconds": end_to_end_seconds}


def _publication_snapshot(directory: Path) -> tuple[tuple[str, str], ...]:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        file_sha256,
        load_canonical_json,
    )

    manifest = load_canonical_json(directory / "manifest.json")
    artifacts = dict(manifest["artifacts"])
    return tuple(
        (relative, file_sha256(directory / relative))
        for relative in sorted((*artifacts, "manifest.json"))
    )


def _load_execution(path: Path, contract_sha256: str) -> dict[str, float]:
    from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
        load_canonical_json,
    )

    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "result_sha256"}
    durations = record.get("durations_seconds")
    if (
        record.get("format")
        != "tinyworlds-nouns-v2-temporal-consolidation-execution-v1"
        or record.get("contract_sha256") != contract_sha256
        or record.get("result_sha256") != record_sha256(core)
        or type(durations) is not dict
    ):
        raise ValueError("published temporal execution measurement changed")
    values = {str(key): float(value) for key, value in durations.items()}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("published temporal execution duration is negative")
    return values


def _notify_completion(report: Path) -> None:
    try:
        subprocess.run(
            (
                "notify-send",
                "TinyWorlds nouns-v2 temporal consolidation complete",
                str(report),
            ),
            check=False,
        )
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
