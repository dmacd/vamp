"""Public run, resume, status, policy-rebuild, and worker commands for TRACE."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.trace.artifacts import TraceStore, validate_artifact_directory
from apm.continual.trace.config import load_experiment_config, load_merge_policy
from apm.continual.trace.dag import build_policy_dag, build_primary_dag
from apm.continual.trace.data import (
    arrival_identities,
    download_pinned_archive,
    extract_pinned_archive,
    load_examples,
    prepare_dataset,
)
from apm.continual.trace.jobs import JobLedger
from apm.continual.trace.modeling import (
    model_source_manifest,
    prepare_model_source,
)
from apm.continual.trace.notifications import notify
from apm.continual.trace.protocol import RunContract
from apm.continual.trace.reporting import build_report
from apm.continual.trace.runpod import require_primary_runpod
from apm.continual.trace.runtime_budget import (
    RuntimeBudget,
    publish_safe_to_terminate,
    terminate_current_runpod,
)
from apm.continual.trace.scheduler import (
    CHECKPOINTED_EXIT_CODE,
    eta_snapshot,
    run_coordinator,
)
from apm.continual.trace.worker import execute_job, install_pause_handlers


EVENTS_FORMAT = "trace-run-events-v1"


def main(arguments: list[str] | None = None) -> int:
    """Parse and execute one TRACE command, returning a process exit status."""
    parser = _parser()
    namespace = parser.parse_args(arguments)
    try:
        if namespace.command == "self-test":
            return _self_test()
        if namespace.command == "run":
            return _run(Path(namespace.config))
        if namespace.command == "resume":
            return _resume(Path(namespace.run))
        if namespace.command == "status":
            return _status(Path(namespace.run))
        if namespace.command == "rebuild-policy":
            return _rebuild_policy(Path(namespace.run), Path(namespace.policy))
        if namespace.command == "report":
            run = Path(namespace.run)
            _validate_run(run)
            result = build_report(run, interim=namespace.interim)
            print(result.markdown_path)
            return 0
        if namespace.command == "_worker":
            install_pause_handlers()
            try:
                execute_job(Path(namespace.run), str(namespace.job_id))
            except InterruptedError:
                return CHECKPOINTED_EXIT_CODE
            return 0
    except Exception as error:
        print(f"TRACE {namespace.command} failed: {error}", file=sys.stderr)
        return 1
    parser.error("a TRACE command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m apm.continual.trace.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="verify the packaged TRACE runtime")
    run = subparsers.add_parser("run", help="prepare and execute the primary experiment")
    run.add_argument("--config", default="configs/trace/primary.yaml")
    resume = subparsers.add_parser("resume", help="resume an existing content-addressed run")
    resume.add_argument("--run", required=True)
    status = subparsers.add_parser("status", help="show durable scheduler and ETA state")
    status.add_argument("--run", required=True)
    rebuild = subparsers.add_parser(
        "rebuild-policy", help="reuse immutable leaves for a new merge/repair policy"
    )
    rebuild.add_argument("--run", required=True)
    rebuild.add_argument("--policy", required=True)
    report = subparsers.add_parser("report", help="regenerate reports from durable results")
    report.add_argument("--run", required=True)
    report.add_argument("--interim", action="store_true")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run", required=True)
    worker.add_argument("--job-id", required=True)
    return parser


def _run(config_path: Path) -> int:
    session_started = datetime.now(timezone.utc)
    config = load_experiment_config(config_path)
    preflight = require_primary_runpod(config.store_root)
    cache = config.store_root / "cache" / "trace"
    archive = download_pinned_archive(cache / "LLM-CL-Benchmark_500.tar.xz")
    extracted = extract_pinned_archive(archive, cache / "LLM-CL-Benchmark_500")
    dataset_cache = cache / "prepared-v1"
    manifest = prepare_dataset(
        extracted,
        dataset_cache / "dataset.json",
        dataset_cache / "examples.jsonl",
    )
    model_manifest = prepare_model_source()
    revision = config.model_revision
    contract = RunContract(
        dataset_manifest_sha256=manifest.manifest_sha256,
        model_revision=revision,
        tokenizer_revision=revision,
        code_revision=_code_identity(),
        dependency_environment_sha256=_dependency_identity(),
        model_manifest_sha256=record_sha256(model_manifest),
    )
    store = TraceStore.from_contract(contract, config.store_root)
    store.prepare(contract)
    publish_immutable_json(store.run / "manifests" / "dataset.json", manifest.as_record())
    publish_immutable_bytes(
        store.run / "manifests" / "examples.jsonl",
        (dataset_cache / "examples.jsonl").read_bytes(),
    )
    arrivals = arrival_identities(manifest.examples)
    publish_immutable_json(
        store.run / "manifests" / "arrivals.json",
        {"arrival_ids": list(arrivals), "format": "trace-arrivals-v1"},
    )
    publish_immutable_json(store.run / "manifests" / "model.json", model_manifest)
    publish_immutable_json(
        store.run / "manifests" / "policies.json",
        {
            "format": "trace-primary-policies-v1",
            "policies": [policy.as_record() for policy in config.policies],
        },
    )
    publish_immutable_json(
        store.run / "manifests" / "dependencies.json",
        _dependency_record(),
    )
    publish_immutable_json(
        store.run / "manifests" / "runpod.json",
        {
            **preflight.as_record(),
            "container_disk_gb": config.container_disk_gb,
            "network_volume_gb": config.network_volume_gb,
            "self_terminate": config.self_terminate,
        },
    )
    jobs = JobLedger(store.run / "manifests" / "jobs.jsonl")
    jobs.register(build_primary_dag(arrivals, config.policies, store.run_hash))
    print(f"TRACE run: {store.run_hash}\nTRACE state: {store.run}")
    return _coordinate_session(
        store.run,
        config.self_terminate,
        started_at=session_started,
    )


def _resume(run: Path) -> int:
    session_started = datetime.now(timezone.utc)
    _validate_run(run)
    deployment = _require_runpod_deployment(run)
    return _coordinate_session(
        run,
        self_terminate=bool(deployment["self_terminate"]),
        started_at=session_started,
    )


def _coordinate_session(
    run: Path,
    self_terminate: bool,
    *,
    started_at: datetime | None = None,
) -> int:
    started = started_at or datetime.now(timezone.utc)
    budget = RuntimeBudget(started)
    session_id = record_sha256({"run": run.name, "started_at": started.isoformat()})
    publish_immutable_json(
        run / "state" / "sessions" / f"{session_id}.json",
        {**budget.as_record(), "session_id": session_id},
    )
    configured = _notify_event(run, "notification_self_test", {"session_id": session_id})
    _notify_event(
        run,
        "experiment_started",
        {"notification_self_test_succeeded": configured, "session_id": session_id},
    )
    try:
        run_coordinator(run, budget)
    except Exception as error:
        _notify_event(run, "experiment_failed_irrecoverably", {"error_type": type(error).__name__})
        raise
    jobs = JobLedger(run / "manifests" / "jobs.jsonl")
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    publish_immutable_json(
        run / "state" / "sessions" / f"{session_id}-summary.json",
        {
            "elapsed_seconds": elapsed,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "format": "trace-runtime-session-summary-v1",
            "job_state_counts": jobs.state_counts(),
            "session_id": session_id,
        },
    )
    if jobs.all_complete():
        report = build_report(run)
        _notify_event(
            run,
            "experiment_completed",
            {
                "elapsed_seconds": elapsed,
                "report_path": str(report.markdown_path),
            },
        )
        reports = (report.markdown_path, report.html_path)
    else:
        stamp = started.strftime("%Y%m%dT%H%M%SZ")
        report = build_report(run, interim=True, name=f"INTERIM_{stamp}")
        details = {
            "completed_conditions": jobs.state_counts()["COMPLETE"],
            "elapsed_seconds": elapsed,
            "estimated_remaining_runtime": _eta_record(run),
            "interim_report_path": str(report.markdown_path),
            "current_best_provisional_scores": _provisional_scores(run),
            "network_volume_identifier": load_canonical_json(
                run / "manifests" / "runpod.json"
            ).get("network_volume_id", "not-provided"),
            "resume_command": f"python -m apm.continual.trace.cli resume --run {run}",
            "unfinished_conditions": len(jobs.statuses) - jobs.state_counts()["COMPLETE"],
        }
        _notify_event(run, "experiment_paused_at_24h", details)
        reports = (report.markdown_path, report.html_path)
    marker = publish_safe_to_terminate(run / "state", jobs, reports)
    if self_terminate:
        if os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_POD_ID"):
            terminate_current_runpod(
                marker,
                run / "state" / "sessions" / f"{session_id}-termination.json",
            )
        else:
            print("TRACE state is safe; RunPod termination credentials were not supplied")
    return 0


def _rebuild_policy(run: Path, policy_path: Path) -> int:
    session_started = datetime.now(timezone.utc)
    _validate_run(run)
    deployment = _require_runpod_deployment(run)
    policy = load_merge_policy(policy_path)
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    if not ledger.all_complete():
        raise RuntimeError("the primary DAG must complete before a leaf-reusing policy rebuild")
    leaf_statuses = tuple(
        status for status in ledger.statuses if status.spec.kind == "train_leaf"
    )
    embedding = tuple(
        status for status in ledger.statuses if status.spec.kind == "prepare_prompt_embeddings"
    )
    if (
        len(leaf_statuses) != 40
        or any(status.state != "COMPLETE" for status in leaf_statuses)
        or len(embedding) != 1
        or embedding[0].state != "COMPLETE"
    ):
        raise RuntimeError("policy rebuild requires all immutable leaves and embeddings")
    arrivals_record = load_canonical_json(run / "manifests" / "arrivals.json")
    arrivals = tuple(str(value) for value in _list(arrivals_record["arrival_ids"]))
    ordered_leaves = tuple(
        next(
            status.spec.job_id
            for status in leaf_statuses
            if int(status.spec.payload["arrival"]) == arrival
        )
        for arrival in range(1, 41)
    )
    hashes_before = _leaf_hashes(run, arrivals)
    ledger.register(
        build_policy_dag(
            arrivals,
            policy,
            ordered_leaves,
            hashes_before,
            embedding[0].spec.job_id,
            run.name,
        )
    )
    return _coordinate_session(
        run,
        self_terminate=bool(deployment["self_terminate"]),
        started_at=session_started,
    )


def _status(run: Path) -> int:
    _validate_run(run)
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    print(
        json.dumps(
            {
                "eta": _eta_record(run),
                "run": str(run),
                "states": ledger.state_counts(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_runpod_deployment(run: Path) -> dict[str, object]:
    """Authenticate current Pod/volume facts against the run's deployment manifest."""
    preflight = require_primary_runpod(run.parents[1])
    deployment = load_canonical_json(run / "manifests" / "runpod.json")
    if deployment.get("network_volume_id") != preflight.network_volume_id:
        raise RuntimeError("TRACE session is attached to a different Network Volume")
    return deployment


def _eta_record(run: Path) -> dict[str, object]:
    result = eta_snapshot(run)
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    complete = ledger.state_counts()["COMPLETE"]
    return {
        **result,
        "live_fraction_complete": complete / len(ledger.statuses) if ledger.statuses else 0.0,
    }


def _notify_event(run: Path, event: str, details: dict[str, object]) -> bool:
    succeeded = notify(event, run.name, details)
    ChainedJsonlLedger(run / "logs" / "events.jsonl", EVENTS_FORMAT).append(
        {
            "delivery_succeeded": succeeded,
            "event": event,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return succeeded


def _provisional_scores(run: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    for path in (run / "evaluations").rglob("result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        scores = record.get("router_scores")
        if type(scores) is not dict:
            continue
        condition = str(record.get("condition") or record.get("policy_hash") or "unknown")
        for router, score in scores.items():
            key = f"{condition}/{router}"
            best[key] = max(best.get(key, float("-inf")), float(score))
    return dict(sorted(best.items()))


def _validate_run(run: Path) -> None:
    if not run.is_dir() or run.parent.name != "runs":
        raise ValueError("--run must name one content-addressed TRACE run directory")
    contract = load_canonical_json(run / "manifests" / "run.json")
    if record_sha256(contract) != run.name:
        raise ValueError("TRACE run directory differs from its immutable contract")
    if contract.get("code_revision") != _code_identity():
        raise ValueError("installed TRACE code differs from the run contract")
    if contract.get("dependency_environment_sha256") != _dependency_identity():
        raise ValueError("installed TRACE dependency environment differs from the run contract")
    if load_canonical_json(run / "manifests" / "dependencies.json") != _dependency_record():
        raise ValueError("installed TRACE dependency environment differs from its manifest")
    dataset = load_canonical_json(run / "manifests" / "dataset.json")
    if record_sha256(dataset) != contract.get("dataset_manifest_sha256"):
        raise ValueError("TRACE dataset manifest differs from the run contract")
    model = load_canonical_json(run / "manifests" / "model.json")
    if (
        record_sha256(model) != contract.get("model_manifest_sha256")
        or model != model_source_manifest()
    ):
        raise ValueError("TRACE model manifest differs from the run contract")
    examples = load_examples(run / "manifests" / "examples.jsonl")
    arrivals = load_canonical_json(run / "manifests" / "arrivals.json")
    if tuple(str(value) for value in _list(arrivals["arrival_ids"])) != arrival_identities(
        examples
    ):
        raise ValueError("TRACE arrival identities differ from the prepared examples")
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    for status in ledger.statuses:
        if status.spec.payload.get("run_contract_hash") != run.name:
            raise ValueError("registered job belongs to another run contract")
        if status.state == "COMPLETE":
            receipt = load_canonical_json(
                run / "state" / "job_outputs" / f"{status.spec.job_id}.json"
            )
            outputs = receipt.get("outputs")
            if type(outputs) is not dict:
                raise ValueError("completed job receipt has malformed outputs")
            expected_output = record_sha256(
                {
                    "job_id": status.spec.job_id,
                    "kind": status.spec.kind,
                    "outputs": outputs,
                }
            )
            if (
                receipt.get("job_id") != status.spec.job_id
                or receipt.get("kind") != status.spec.kind
                or receipt.get("output_sha256") != expected_output
                or status.output_sha256 != expected_output
            ):
                raise ValueError("completed job receipt differs from the scheduler ledger")
            _validate_receipt_outputs(run, outputs)


def _validate_receipt_outputs(run: Path, outputs: dict[str, object]) -> None:
    artifact_directory = outputs.get("artifact_directory")
    if artifact_directory is not None and validate_artifact_directory(
        str(artifact_directory)
    ) != outputs.get("artifact_sha256"):
        raise ValueError("completed job artifact differs from its receipt")
    artifacts = outputs.get("artifacts")
    if type(artifacts) is list:
        for artifact in artifacts:
            if type(artifact) is not dict:
                raise ValueError("baseline artifact receipt is malformed")
            _validate_receipt_outputs(run, artifact)
    for path_field, hash_field in (
        ("result_path", "result_sha256"),
        ("diagnostic_path", "diagnostic_sha256"),
        ("calibration_result", "calibration_result_sha256"),
        ("acceptance_path", "acceptance_sha256"),
    ):
        path = outputs.get(path_field)
        digest = outputs.get(hash_field)
        if path is not None and file_sha256(str(path)) != digest:
            raise ValueError(f"completed output {path_field} differs from its receipt")
    tensor_digest = outputs.get("tensor_sha256")
    if tensor_digest is not None and file_sha256(
        run / "manifests" / "prompt_embeddings.safetensors"
    ) != tensor_digest:
        raise ValueError("prompt embedding cache differs from its receipt")


def _leaf_hashes(run: Path, arrivals: tuple[str, ...]) -> tuple[str, ...]:
    hashes = []
    for identity in arrivals:
        directory = run / "leaves" / identity
        validate_artifact_directory(directory)
        hashes.append(file_sha256(directory / "adapter.safetensors"))
    return tuple(hashes)


def _code_identity() -> str:
    root = Path(__file__).resolve().parents[4]
    source_files = tuple(sorted((root / "src").rglob("*.py")))
    support_files = tuple(
        path
        for base in (
            root / "configs" / "trace",
            root / "docker" / "trace",
            root / "scripts" / "trace",
        )
        if base.exists()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    )
    files = tuple(
        (str(path.relative_to(root)), file_sha256(path))
        for path in (*source_files, *support_files, root / "pyproject.toml")
    )
    if not files:
        raise RuntimeError("could not resolve the installed TRACE source tree")
    return f"tree-sha256:{record_sha256(files)}"


def _dependency_identity() -> str:
    return record_sha256(_dependency_record())


def _dependency_record() -> dict[str, object]:
    """Return exact material package, Python, CUDA, and lock identities."""
    root = Path(__file__).resolve().parents[4]
    lock = root / "docker" / "trace" / "requirements.lock"
    packages = (
        "accelerate",
        "datasets",
        "fuzzywuzzy",
        "huggingface-hub",
        "matplotlib",
        "nltk",
        "numpy",
        "packaging",
        "pandas",
        "peft",
        "pyarrow",
        "pytest",
        "pytest-xdist",
        "python-Levenshtein",
        "pyrsistent",
        "PyYAML",
        "rouge",
        "sacrebleu",
        "sacremoses",
        "safetensors",
        "torch",
        "tokenizers",
        "tqdm",
        "transformers",
    )
    import torch

    return {
        "cuda_runtime": torch.version.cuda,
        "format": "trace-dependency-environment-v1",
        "lock_sha256": file_sha256(lock) if lock.is_file() else None,
        "packages": {name: metadata.version(name) for name in packages},
        "python": platform.python_version(),
    }


def _self_test() -> int:
    modules = (
        "accelerate",
        "datasets",
        "fuzzywuzzy",
        "matplotlib",
        "nltk",
        "numpy",
        "packaging",
        "pandas",
        "peft",
        "pyarrow",
        "rouge",
        "sacrebleu",
        "sacremoses",
        "safetensors",
        "torch",
        "tokenizers",
        "transformers",
        "yaml",
    )
    for module in modules:
        importlib.import_module(module)
    from apm.continual.trace.lineage import build_hierarchy
    from apm.continual.trace.modeling import peft_round_trip_self_test

    arrivals = tuple(record_sha256({"self_test_arrival": value}) for value in range(40))
    hierarchy, merges = build_hierarchy(arrivals)
    if len(hierarchy.active_nodes) != 7 or len(merges) != 33:
        raise RuntimeError("TRACE hierarchy self-test failed")
    peft_round_trip_self_test()
    print("TRACE package self-test passed")
    return 0


def _list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError("TRACE manifest list field is malformed")
    return value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
