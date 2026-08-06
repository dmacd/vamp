#!/usr/bin/env python3
"""Build, approve, run, resume, judge, and report TinyWorlds nouns v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_nouns_v1.contracts import (  # noqa: E402
    CHECKPOINT_ROOT,
    DATA_ROOT,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TOKENIZER_PATH,
    RESULT_ROOT,
    NounsExperimentPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v1.partition import (  # noqa: E402
    NounApprovalRequired,
    approve_noun_breakdown,
    build_noun_partition,
    load_noun_decisions,
    load_noun_partition,
    load_scanned_noun_breakdown,
    publish_initial_noun_decisions,
    publish_noun_breakdown,
    require_noun_approval,
    scan_pinned_noun_breakdown,
)
from apm.lm.text import TokenizersTextTokenizer  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / DATA_ROOT
CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / CHECKPOINT_ROOT
RESULT_DIRECTORY = REPOSITORY_ROOT / RESULT_ROOT
SOURCE_DIRECTORY = REPOSITORY_ROOT / DEFAULT_SOURCE_ROOT
TOKENIZER_PATH = REPOSITORY_ROOT / DEFAULT_TOKENIZER_PATH
DECISIONS_PATH = DATA_DIRECTORY / "noun-decisions.json"
SCAN_DATABASE = DATA_DIRECTORY / "work" / "noun-scan.sqlite3"


def main(argv: list[str] | None = None) -> int:
    """Advance the sole canonical experiment until complete or human-gated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-noun-breakdown",
        metavar="SHA256",
        help="record manual approval for the exact currently rebuilt review packet",
    )
    arguments = parser.parse_args(argv)
    started = time.monotonic()
    print("Phase 1/7: noun scan and manual-review packet.", flush=True)
    publish_initial_noun_decisions(DECISIONS_PATH)
    decisions = load_noun_decisions(DECISIONS_PATH)
    tokenizer = TokenizersTextTokenizer.from_file(TOKENIZER_PATH)
    try:
        breakdown = load_scanned_noun_breakdown(
            SCAN_DATABASE,
            decisions,
            DATA_DIRECTORY,
        )
        print(f"Reusing noun scan {breakdown.breakdown_sha256}.", flush=True)
    except (FileNotFoundError, ValueError):
        breakdown, _database = scan_pinned_noun_breakdown(
            SOURCE_DIRECTORY / "TinyStoriesV2-GPT4-train.txt",
            SOURCE_DIRECTORY / "TinyStoriesV2-GPT4-valid.txt",
            tokenizer,
            TOKENIZER_PATH,
            decisions,
            SCAN_DATABASE.parent,
            progress=_scan_progress(started),
        )
    review_directory = publish_noun_breakdown(
        breakdown,
        decisions,
        DATA_DIRECTORY,
    )
    print(f"Noun review packet: {review_directory / 'noun-breakdown.html'}", flush=True)
    print(f"Editable decisions: {DECISIONS_PATH}", flush=True)
    if arguments.approve_noun_breakdown is not None:
        approval_path = approve_noun_breakdown(
            breakdown,
            decisions,
            arguments.approve_noun_breakdown,
            DATA_DIRECTORY,
        )
        print(f"Recorded exact manual approval: {approval_path}", flush=True)
        print("Rerun the no-argument command to begin execution.", flush=True)
        _notify("TinyWorlds nouns", "Noun breakdown approved; execution is ready.")
        return 0
    try:
        approval = require_noun_approval(breakdown, decisions, DATA_DIRECTORY)
    except NounApprovalRequired:
        print("", flush=True)
        print("STOPPED AT THE MANUAL NOUN-APPROVAL GATE.", flush=True)
        print(
            "Review every noun and example, edit noun-decisions.json if needed, rerun "
            "to rebuild, then approve the unchanged hash with:",
            flush=True,
        )
        print(
            "python scripts/run_tinyworlds_nouns_v1.py "
            f"--approve-noun-breakdown {breakdown.breakdown_sha256}",
            flush=True,
        )
        _notify("TinyWorlds nouns", "Noun breakdown is ready for manual review.")
        return 3

    print("Phase 2/7: approved overlapping partition.", flush=True)
    partition = _matching_partition(breakdown.breakdown_sha256, approval.approval_sha256)
    if partition is None:
        partition = build_noun_partition(
            breakdown,
            approval,
            decisions,
            SCAN_DATABASE,
            DATA_DIRECTORY,
        )
    print(
        f"Partition {partition.partition_sha256}; {len(partition.task_ids)} noun tasks.",
        flush=True,
    )

    # Training code is deliberately imported only after the exact manual gate passes.
    from apm.data.text.tinyworlds_nouns_v1.evaluation import (
        evaluate_half_story_generations,
        evaluate_whole_story_nll,
    )
    from apm.data.text.tinyworlds_nouns_v1.experiment import (
        run_or_load_noun_gpu_preflight,
        run_or_resume_noun_base,
        run_or_resume_noun_vamp,
    )
    from apm.data.text.tinyworlds_nouns_v1.judging import (
        DEFAULT_JUDGE_MODEL,
        JudgeCredentialsMissing,
        judge_generation_ledger,
    )
    from apm.data.text.tinyworlds_nouns_v1.report import publish_noun_report

    preset = NounsExperimentPreset()
    result_root = RESULT_DIRECTORY
    result_root.mkdir(parents=True, exist_ok=True)
    _publish_run_manifest(
        result_root,
        partition.partition_sha256,
        approval.approval_sha256,
        preset,
        "partition_complete",
    )
    print("Phase 3/7: GPU preflight and fresh two-epoch seed-zero base.", flush=True)
    preflight = run_or_load_noun_gpu_preflight(
        partition,
        preset,
        CHECKPOINT_DIRECTORY,
    )
    print(
        f"GPU preflight {preflight.preflight_sha256}; peak "
        f"{preflight.measured_peak_bytes / 2**30:.2f} GiB.",
        flush=True,
    )
    selected_base = run_or_resume_noun_base(
        partition,
        preset,
        preflight,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    _publish_run_manifest(
        result_root,
        partition.partition_sha256,
        approval.approval_sha256,
        preset,
        "base_complete",
    )
    print("Phase 4/7: VAMP noun-task adapters.", flush=True)
    adaptation = run_or_resume_noun_vamp(
        partition,
        preset,
        selected_base,
        CHECKPOINT_DIRECTORY,
        progress=_training_progress(started),
    )
    _publish_run_manifest(
        result_root,
        partition.partition_sha256,
        approval.approval_sha256,
        preset,
        "vamp_complete",
    )
    print("Phase 5/7: whole-story loss and routing.", flush=True)
    whole_path = evaluate_whole_story_nll(
        partition,
        preset,
        selected_base,
        adaptation,
        result_root / "whole-story-nll.jsonl",
        progress=_evaluation_progress(started),
    )
    print("Phase 6/7: midpoint routing and greedy story endings.", flush=True)
    generation_path = evaluate_half_story_generations(
        partition,
        preset,
        selected_base,
        adaptation,
        tokenizer,
        result_root / "half-story-generations.jsonl",
        progress=_evaluation_progress(started),
    )
    _publish_run_manifest(
        result_root,
        partition.partition_sha256,
        approval.approval_sha256,
        preset,
        "local_evaluation_complete",
    )
    print("Phase 7/7: external judging and final report.", flush=True)
    judge_path = None
    try:
        judge_path = judge_generation_ledger(
            generation_path,
            result_root,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            model=os.environ.get("OPENROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            progress=_evaluation_progress(started),
        )
    except JudgeCredentialsMissing as error:
        print(str(error), flush=True)
    markdown, html = publish_noun_report(
        partition,
        breakdown,
        preset,
        adaptation,
        whole_path,
        generation_path,
        result_root,
        judge_path=judge_path,
    )
    _copy_partition_manifest(partition.root / "partition.json", result_root)
    final_phase = "complete" if judge_path is not None else "awaiting_judge_credentials"
    _publish_run_manifest(
        result_root,
        partition.partition_sha256,
        approval.approval_sha256,
        preset,
        final_phase,
    )
    print(f"Markdown report: {markdown}", flush=True)
    print(f"Interactive report: {html}", flush=True)
    _notify("TinyWorlds nouns", f"Run reached {final_phase.replace('_', ' ')}.")
    return 0 if judge_path is not None else 4


def _matching_partition(breakdown_sha256: str, approval_sha256: str):
    candidates = tuple(
        path
        for path in (DATA_DIRECTORY / "partitions").glob("*/partition.json")
        if path.is_file()
    ) if (DATA_DIRECTORY / "partitions").is_dir() else ()
    matches = tuple(
        path
        for path in candidates
        for record in (json.loads(path.read_text(encoding="utf-8")),)
        if record.get("breakdown_sha256") == breakdown_sha256
        and record.get("approval_sha256") == approval_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple noun partitions bind the same manual approval")
    return load_noun_partition(matches[0]) if matches else None


def _publish_run_manifest(
    root: Path,
    partition_sha256: str,
    approval_sha256: str,
    preset: NounsExperimentPreset,
    phase: str,
) -> None:
    core = {
        "approval_sha256": approval_sha256,
        "config_sha256": preset.config_sha256,
        "format": "tinyworlds-nouns-run-v1",
        "partition_sha256": partition_sha256,
        "phase": phase,
    }
    _atomic_write(
        root / "run-manifest.json",
        canonical_json_bytes({**core, "manifest_sha256": record_sha256(core)}),
    )


def _copy_partition_manifest(source: Path, root: Path) -> None:
    _atomic_write(root / "partition.json", source.read_bytes())


def _scan_progress(started: float):
    phase_started: dict[str, float] = {}
    last_completed: dict[str, int] = {}

    def progress(phase: str, completed: int, total: int | None) -> None:
        phase_started.setdefault(phase, time.monotonic())
        if (
            phase.endswith("-bytes")
            and total is not None
            and completed < total
            and completed - last_completed.get(phase, 0) < 128 * 1024**2
        ):
            return
        last_completed[phase] = completed
        elapsed = max(time.monotonic() - phase_started[phase], 1e-6)
        rate = completed / elapsed
        if phase.endswith("-bytes"):
            amount = f"{completed / 2**30:.2f}"
            suffix = f"/{total / 2**30:.2f} GiB" if total is not None else " GiB"
            rate_text = f"{rate / 2**20:.1f} MiB/s"
        else:
            amount = f"{completed:,}"
            suffix = f"/{total:,} stories" if total is not None else " stories"
            rate_text = f"{rate:,.0f}/s"
        remaining = (
            elapsed * (total - completed) / max(1, completed)
            if total is not None
            else None
        )
        eta = f"; phase ETA {_duration(remaining)}" if remaining is not None else ""
        print(
            f"  [{phase}] {amount}{suffix}; {rate_text}{eta}; "
            f"overall elapsed {_duration(time.monotonic() - started)}",
            flush=True,
        )

    return progress


def _training_progress(started: float):
    phase_started: dict[str, float] = {}

    def progress(phase: str, completed: int, total: int, value: float) -> None:
        phase_started.setdefault(phase, time.monotonic())
        elapsed = time.monotonic() - phase_started[phase]
        remaining = elapsed * (total - completed) / max(1, completed)
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(
                f"  [{phase}] {completed:,}/{total:,}; NLL {value:.5f}; "
                f"phase ETA {_duration(remaining)}; overall {_duration(time.monotonic() - started)}",
                flush=True,
            )

    return progress


def _evaluation_progress(started: float):
    phase_started: dict[str, float] = {}

    def progress(phase: str, completed: int, total: int) -> None:
        phase_started.setdefault(phase, time.monotonic())
        elapsed = time.monotonic() - phase_started[phase]
        remaining = elapsed * (total - completed) / max(1, completed)
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(
                f"  [{phase}] {completed:,}/{total:,}; phase ETA {_duration(remaining)}; "
                f"overall {_duration(time.monotonic() - started)}",
                flush=True,
            )

    return progress


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _atomic_write(path: Path, payload: bytes) -> None:
    import tempfile

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
