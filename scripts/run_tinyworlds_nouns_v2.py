#!/usr/bin/env python3
"""Build, run, resume, optionally judge, and report TinyWorlds nouns-v2."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_enable_command_buffer=".strip()
    )

from apm.data.text.tinyworlds_nouns_v2.contracts import (  # noqa: E402
    BENCHMARK_ID,
    CHECKPOINT_ROOT,
    DATA_ROOT,
    DEFAULT_TOKENIZER_PATH,
    RESULT_ROOT,
    RUN_MANIFEST_FORMAT,
    NounsV2ExperimentPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.partition import (  # noqa: E402
    authenticate_parent_manifest,
    build_nouns_v2_partition,
    find_partition,
    publish_manifest,
    verify_byte_identical_rebuild,
)
from apm.lm.text import TokenizersTextTokenizer  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / DATA_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / CHECKPOINT_ROOT
RESULT_DIRECTORY = REPOSITORY_ROOT / RESULT_ROOT
TOKENIZER_PATH = REPOSITORY_ROOT / DEFAULT_TOKENIZER_PATH
PARENT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data/tinyworlds-nouns-v1/partitions"
    / "04ca2acf85f9505f0b7568b1696fbf290a8d2cbf78387dcfd6e815258fcc28b8"
)


def main(argv: list[str] | None = None) -> int:
    """Advance the canonical local run, adding external judgment only on request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="judge persisted generations through OpenRouter, then rebuild the report",
    )
    arguments = parser.parse_args(argv)
    started = time.monotonic()
    print("Phase 1/12: authenticate the immutable nouns-v1 parent.", flush=True)
    manifest = authenticate_parent_manifest(PARENT_DIRECTORY)
    manifest_path = publish_manifest(manifest, DATA_DIRECTORY)
    print(
        f"Nouns-v2 manifest {manifest.manifest_sha256}: {manifest_path}",
        flush=True,
    )

    print("Phase 2/12: build and independently reconstruct the disjoint partition.", flush=True)
    partition = find_partition(manifest, DATA_DIRECTORY)
    if partition is None:
        partition = build_nouns_v2_partition(
            manifest,
            PARENT_DIRECTORY,
            DATA_DIRECTORY,
            progress=_partition_progress(started),
        )
    verify_byte_identical_rebuild(
        partition,
        manifest,
        progress=_partition_progress(started),
    )
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _copy_exact(partition.root / "partition.json", RESULT_DIRECTORY / "partition.json")
    _publish_run_manifest(
        partition.partition_sha256,
        manifest.manifest_sha256,
        "partition_complete",
    )
    _publish_execution_status(
        "partition_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        None,
        None,
        None,
    )
    print(
        f"Partition {partition.partition_sha256}; {len(partition.task_ids)} disjoint tasks.",
        flush=True,
    )

    # JAX-bearing modules remain behind complete parent authentication/reconstruction.
    from apm.data.text.tinyworlds_nouns_v2.evaluation import (
        evaluate_half_story_generations,
        evaluate_whole_story_nll,
    )
    from apm.data.text.tinyworlds_nouns_v2.baselines import (
        load_nouns_v2_baseline_stages,
        run_or_resume_nouns_v2_baselines,
    )
    from apm.data.text.tinyworlds_nouns_v2.baseline_stagewise import (
        evaluate_stagewise_baselines,
    )
    from apm.data.text.tinyworlds_nouns_v2.full_finetune import (
        load_nouns_v2_full_finetune_stages,
        run_or_resume_nouns_v2_full_finetune,
    )
    from apm.data.text.tinyworlds_nouns_v2.full_finetune_stagewise import (
        evaluate_stagewise_full_finetune,
    )
    from apm.data.text.tinyworlds_nouns_v2.experiment import (
        load_nouns_v2_vamp_stages,
        run_or_load_nouns_v2_gpu_preflight,
        run_or_resume_nouns_v2_base,
        run_or_resume_nouns_v2_vamp,
    )
    from apm.data.text.tinyworlds_nouns_v2.stagewise import (
        evaluate_stagewise_continual_learning,
        expected_stagewise_row_count,
    )
    from apm.data.text.tinyworlds_nouns_v2.judging import (
        DEFAULT_JUDGE_MODEL,
        JudgeCredentialsMissing,
        judge_nouns_v2_generation_ledger,
    )
    from apm.data.text.tinyworlds_nouns_v2.report import publish_nouns_v2_report

    preset = NounsV2ExperimentPreset()
    tokenizer = TokenizersTextTokenizer.from_file(TOKENIZER_PATH)
    print("Phase 3/12: GPU preflight and fresh two-epoch seed-zero base.", flush=True)
    preflight = run_or_load_nouns_v2_gpu_preflight(
        partition,
        preset,
        CHECKPOINT_DIRECTORY,
    )
    print(
        f"GPU preflight {preflight.preflight_sha256}; peak "
        f"{preflight.measured_peak_bytes / 2**30:.2f} GiB.",
        flush=True,
    )
    selected_base = run_or_resume_nouns_v2_base(
        partition,
        preset,
        preflight,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    _publish_run_manifest(partition.partition_sha256, manifest.manifest_sha256, "base_complete")
    _publish_execution_status(
        "base_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        None,
        None,
    )

    print("Phase 4/12: ordered 24-stage VAMP adapter graph.", flush=True)
    adaptation = run_or_resume_nouns_v2_vamp(
        partition,
        preset,
        selected_base,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    adaptations = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        CHECKPOINT_DIRECTORY,
    )
    if adaptations[-1].tensor_checksum != adaptation.tensor_checksum:
        raise RuntimeError("strict stage audit and completed VAMP graph differ")
    _publish_run_manifest(partition.partition_sha256, manifest.manifest_sha256, "vamp_complete")
    _publish_execution_status(
        "vamp_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        adaptation.tensor_checksum,
        None,
    )

    print(
        "Phase 5/12: sequential and independent adapters, 48,000 updates each.",
        flush=True,
    )
    baselines = run_or_resume_nouns_v2_baselines(
        partition,
        preset,
        selected_base,
        adaptations,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    baseline_stages = load_nouns_v2_baseline_stages(
        partition,
        preset,
        selected_base,
        adaptations,
        CHECKPOINT_DIRECTORY,
    )
    if baseline_stages[-1].tensor_checksum != baselines.tensor_checksum:
        raise RuntimeError("strict baseline audit and completed controls differ")
    _publish_run_manifest(
        partition.partition_sha256,
        manifest.manifest_sha256,
        "baselines_complete",
        vamp_tensor_checksum=adaptation.tensor_checksum,
        baseline_tensor_checksum=baselines.tensor_checksum,
    )
    _publish_execution_status(
        "baselines_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        adaptation.tensor_checksum,
        baselines.tensor_checksum,
    )

    print(
        "Phase 6/12: sequential full-model control, 48,000 parameter updates.",
        flush=True,
    )
    full_finetune = run_or_resume_nouns_v2_full_finetune(
        partition,
        preset,
        selected_base,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    full_finetune_stages = load_nouns_v2_full_finetune_stages(
        partition,
        preset,
        selected_base,
        CHECKPOINT_DIRECTORY,
    )
    if full_finetune_stages[-1].parameter_checksum != full_finetune.parameter_checksum:
        raise RuntimeError("strict full-finetune audit and completed control differ")
    _publish_run_manifest(
        partition.partition_sha256,
        manifest.manifest_sha256,
        "full_finetune_complete",
        vamp_tensor_checksum=adaptation.tensor_checksum,
        baseline_tensor_checksum=baselines.tensor_checksum,
        full_finetune_parameter_checksum=full_finetune.parameter_checksum,
        full_finetune_run_sha256=full_finetune.run_sha256,
    )
    _publish_execution_status(
        "full_finetune_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        adaptation.tensor_checksum,
        baselines.tensor_checksum,
        full_finetune.parameter_checksum,
    )

    print("Phase 7/12: all 26,640 whole-story NLL and routing rows.", flush=True)
    whole_path = evaluate_whole_story_nll(
        partition,
        preset,
        selected_base,
        adaptation,
        RESULT_DIRECTORY / "whole-story-nll.jsonl",
        progress=_evaluation_progress(started),
    )
    print("Phase 8/12: midpoint-only routing, suffix NLL, and greedy completions.", flush=True)
    generation_path = evaluate_half_story_generations(
        partition,
        preset,
        selected_base,
        adaptation,
        tokenizer,
        RESULT_DIRECTORY / "half-story-generations.jsonl",
        progress=_evaluation_progress(started),
    )
    stagewise_count = expected_stagewise_row_count(partition)
    print(
        f"Phase 9/12: {stagewise_count:,} VAMP continual-learning cases.",
        flush=True,
    )
    stagewise_path = evaluate_stagewise_continual_learning(
        partition,
        preset,
        selected_base,
        adaptations,
        RESULT_DIRECTORY / "stagewise-cl.jsonl",
        progress=_evaluation_progress(started),
    )
    print(
        f"Phase 10/12: {stagewise_count:,} sequential/independent adapter cases.",
        flush=True,
    )
    baseline_stagewise_path = evaluate_stagewise_baselines(
        partition,
        preset,
        selected_base,
        baseline_stages,
        adaptations,
        stagewise_path,
        RESULT_DIRECTORY / "baseline-stagewise-cl.jsonl",
        progress=_evaluation_progress(started),
    )
    print(
        f"Phase 11/12: {stagewise_count:,} sequential full-model cases.",
        flush=True,
    )
    full_finetune_stagewise_path = evaluate_stagewise_full_finetune(
        partition,
        preset,
        full_finetune_stages,
        adaptations,
        stagewise_path,
        RESULT_DIRECTORY / "full-finetune-stagewise-cl.jsonl",
        progress=_evaluation_progress(started),
    )
    _publish_run_manifest(
        partition.partition_sha256,
        manifest.manifest_sha256,
        "local_evaluation_complete",
        vamp_tensor_checksum=adaptation.tensor_checksum,
        baseline_tensor_checksum=baselines.tensor_checksum,
        full_finetune_parameter_checksum=full_finetune.parameter_checksum,
        full_finetune_run_sha256=full_finetune.run_sha256,
        vamp_stagewise_sha256=_file_sha256(stagewise_path),
        baseline_stagewise_sha256=_file_sha256(baseline_stagewise_path),
        full_finetune_stagewise_sha256=_file_sha256(
            full_finetune_stagewise_path
        ),
    )
    _publish_execution_status(
        "stagewise_audit_complete",
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        adaptation.tensor_checksum,
        baselines.tensor_checksum,
        full_finetune.parameter_checksum,
    )

    print("Phase 12/12: comparative reports, plots, and dependency graph.", flush=True)
    markdown, html = publish_nouns_v2_report(
        partition,
        preset,
        adaptations,
        baseline_stages,
        full_finetune_stages,
        whole_path,
        generation_path,
        stagewise_path,
        baseline_stagewise_path,
        full_finetune_stagewise_path,
        RESULT_DIRECTORY,
    )
    phase = "local_complete"
    judge_path = None
    if arguments.judge:
        print("Optional judgment: resuming from local generations.", flush=True)
        try:
            judge_path = judge_nouns_v2_generation_ledger(
                generation_path,
                RESULT_DIRECTORY,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                model=os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
                progress=_evaluation_progress(started),
            )
        except JudgeCredentialsMissing as error:
            print(str(error), flush=True)
            _notify("TinyWorlds nouns-v2", "Local run complete; judge credentials missing.")
            return 4
        markdown, html = publish_nouns_v2_report(
            partition,
            preset,
            adaptations,
            baseline_stages,
            full_finetune_stages,
            whole_path,
            generation_path,
            stagewise_path,
            baseline_stagewise_path,
            full_finetune_stagewise_path,
            RESULT_DIRECTORY,
            judge_path=judge_path,
        )
        phase = "complete_with_judge"
    _publish_run_manifest(
        partition.partition_sha256,
        manifest.manifest_sha256,
        phase,
        vamp_tensor_checksum=adaptation.tensor_checksum,
        baseline_tensor_checksum=baselines.tensor_checksum,
        full_finetune_parameter_checksum=full_finetune.parameter_checksum,
        full_finetune_run_sha256=full_finetune.run_sha256,
        vamp_stagewise_sha256=_file_sha256(stagewise_path),
        baseline_stagewise_sha256=_file_sha256(baseline_stagewise_path),
        full_finetune_stagewise_sha256=_file_sha256(
            full_finetune_stagewise_path
        ),
        report_markdown_sha256=_file_sha256(markdown),
        report_html_sha256=_file_sha256(html),
    )
    _publish_execution_status(
        phase,
        partition.partition_sha256,
        manifest.manifest_sha256,
        selected_base.training_sha256,
        adaptation.tensor_checksum,
        baselines.tensor_checksum,
        full_finetune.parameter_checksum,
    )
    print(f"Markdown report: {markdown}", flush=True)
    print(f"Interactive report: {html}", flush=True)
    if judge_path is not None:
        print(f"Judge ledger: {judge_path}", flush=True)
    _notify("TinyWorlds nouns-v2", f"Run reached {phase.replace('_', ' ')}.")
    return 0


def _publish_run_manifest(
    partition_sha256: str,
    manifest_sha256: str,
    phase: str,
    *,
    vamp_tensor_checksum: str | None = None,
    baseline_tensor_checksum: str | None = None,
    full_finetune_parameter_checksum: str | None = None,
    full_finetune_run_sha256: str | None = None,
    vamp_stagewise_sha256: str | None = None,
    baseline_stagewise_sha256: str | None = None,
    full_finetune_stagewise_sha256: str | None = None,
    report_markdown_sha256: str | None = None,
    report_html_sha256: str | None = None,
) -> None:
    preset = NounsV2ExperimentPreset()
    core = {
        "benchmark_id": BENCHMARK_ID,
        "config_sha256": preset.config_sha256,
        "format": RUN_MANIFEST_FORMAT,
        "manifest_sha256": manifest_sha256,
        "partition_sha256": partition_sha256,
        "phase": phase,
        **{
            name: value
            for name, value in (
                ("baseline_stagewise_sha256", baseline_stagewise_sha256),
                ("baseline_tensor_checksum", baseline_tensor_checksum),
                (
                    "full_finetune_parameter_checksum",
                    full_finetune_parameter_checksum,
                ),
                ("full_finetune_run_sha256", full_finetune_run_sha256),
                (
                    "full_finetune_stagewise_sha256",
                    full_finetune_stagewise_sha256,
                ),
                ("report_html_sha256", report_html_sha256),
                ("report_markdown_sha256", report_markdown_sha256),
                ("vamp_stagewise_sha256", vamp_stagewise_sha256),
                ("vamp_tensor_checksum", vamp_tensor_checksum),
            )
            if value is not None
        },
    }
    _atomic_write(
        RESULT_DIRECTORY / "run-manifest.json",
        canonical_json_bytes({**core, "run_sha256": record_sha256(core)}),
    )


def _publish_execution_status(
    phase: str,
    partition_sha256: str,
    manifest_sha256: str,
    base_training_sha256: str | None,
    adaptation_checksum: str | None,
    baseline_checksum: str | None,
    full_finetune_checksum: str | None = None,
) -> None:
    lines = [
        "# TinyWorlds nouns-v2 execution status",
        "",
        f"Current phase: `{phase}`",
        "",
        f"Manifest: `{manifest_sha256}`",
        "",
        f"Partition: `{partition_sha256}`",
        "",
        "The partition is story-disjoint and was independently rebuilt byte-for-byte "
        "before GPU execution.",
        "",
    ]
    if base_training_sha256 is not None:
        lines.extend((f"Fresh base training identity: `{base_training_sha256}`", ""))
    if adaptation_checksum is not None:
        lines.extend((f"VAMP tensor checksum: `{adaptation_checksum}`", ""))
    if baseline_checksum is not None:
        lines.extend((f"Baseline tensor checksum: `{baseline_checksum}`", ""))
    if full_finetune_checksum is not None:
        lines.extend(
            (
                "Sequential full-model parameter checksum: "
                f"`{full_finetune_checksum}`",
                "",
            )
        )
    _atomic_write(
        RESULT_DIRECTORY / "execution-report.md",
        "\n".join(lines).encode("utf-8"),
    )


def _partition_progress(started: float):
    phase_started: dict[str, float] = {}
    last_completed: dict[str, int] = {}

    def progress(phase: str, completed: int, total: int | None) -> None:
        phase_started.setdefault(phase, time.monotonic())
        if (
            total is not None
            and completed < total
            and completed - last_completed.get(phase, 0) < 100_000
        ):
            return
        last_completed[phase] = completed
        elapsed = max(time.monotonic() - phase_started[phase], 1e-6)
        remaining = (
            elapsed * (total - completed) / max(1, completed)
            if total is not None
            else None
        )
        eta = f"; phase ETA {_duration(remaining)}" if remaining is not None else ""
        total_text = f"/{total:,}" if total is not None else ""
        print(
            f"  [{phase}] {_progress_bar(completed, total)} "
            f"{completed:,}{total_text} stories; "
            f"{completed / elapsed:,.0f}/s{eta}; overall "
            f"{_duration(time.monotonic() - started)}",
            flush=True,
        )

    return progress


def _training_progress(started: float):
    phase_started: dict[str, float] = {}
    initial_completed: dict[str, int] = {}

    def progress(phase: str, completed: int, total: int, value: float) -> None:
        phase_started.setdefault(phase, time.monotonic())
        initial_completed.setdefault(phase, completed - 1)
        elapsed = time.monotonic() - phase_started[phase]
        observed = completed - initial_completed[phase]
        remaining = elapsed * (total - completed) / max(1, observed)
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(
                f"  [{phase}] {_progress_bar(completed, total)} "
                f"{completed:,}/{total:,}; NLL {value:.5f}; "
                f"phase ETA {_duration(remaining)}; overall "
                f"{_duration(time.monotonic() - started)}",
                flush=True,
            )

    return progress


def _evaluation_progress(started: float):
    phase_started: dict[str, float] = {}
    initial_completed: dict[str, int] = {}

    def progress(phase: str, completed: int, total: int) -> None:
        phase_started.setdefault(phase, time.monotonic())
        initial_completed.setdefault(phase, completed - 1)
        elapsed = time.monotonic() - phase_started[phase]
        observed = completed - initial_completed[phase]
        remaining = elapsed * (total - completed) / max(1, observed)
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(
                f"  [{phase}] {_progress_bar(completed, total)} "
                f"{completed:,}/{total:,}; phase ETA "
                f"{_duration(remaining)}; overall "
                f"{_duration(time.monotonic() - started)}",
                flush=True,
            )

    return progress


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _progress_bar(completed: int, total: int | None, width: int = 20) -> str:
    if total is None or total <= 0:
        return "[working]"
    ratio = min(1.0, max(0.0, completed / total))
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio:6.2%}"


def _copy_exact(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.is_file() and destination.read_bytes() == payload:
        return
    _atomic_write(destination, payload)


def _file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _notify(title: str, message: str) -> None:
    try:
        subprocess.run(
            ("notify-send", title, message),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
