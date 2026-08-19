"""Execution of one independently resumable TRACE DAG job."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import threading
from typing import cast

import torch

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.trace.adapter_io import ADAPTER_FILENAME
from apm.continual.trace.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.trace.bank import (
    node_artifact_directory,
    node_from_record,
    node_reservoir,
    reservoir_record,
)
from apm.continual.trace.baselines import baseline_plans
from apm.continual.trace.consolidation import (
    consolidate_adapters,
    materialize_precompress_adapter,
)
from apm.continual.trace.data import TraceExample, load_examples
from apm.continual.trace.embedding_cache import (
    build_prompt_embedding_cache,
    hierarchy_centroids,
    load_prompt_embeddings,
)
from apm.continual.trace.evaluation import (
    Candidate,
    CandidateOutput,
    PeftCandidateRuntime,
    RoutedPrediction,
    evaluate_candidate_cache,
    route_outputs,
    task_aware_candidates,
)
from apm.continual.trace.jobs import JobLedger, JobSpec
from apm.continual.trace.leaf_training import run_leaf_training
from apm.continual.trace.lineage import HierarchyNode
from apm.continual.trace.metrics import score_task
from apm.continual.trace.modeling import load_fresh_lora_bundle
from apm.continual.trace.protocol import (
    MODEL_ID,
    TASKS,
    MergeMethod,
    MergePolicy,
)
from apm.continual.trace.reporting import build_report
from apm.continual.trace.reservoirs import merge_reservoirs
from apm.continual.trace.training_jobs import (
    repair_training_config,
    run_training_artifact,
)
from apm.continual.trace.training_plans import repair_plan, retrained_parent_plan


_PAUSE = threading.Event()


def install_pause_handlers() -> None:
    """Translate coordinator termination signals into safe-boundary requests."""

    def request_pause(_signum: int, _frame: object) -> None:
        _PAUSE.set()

    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)


def execute_job(run_directory: str | Path, job_id: str) -> str:
    """Execute one registered job and publish its idempotent completion receipt."""
    run = Path(run_directory)
    spec = _registered_job(run, job_id)
    if spec.payload.get("run_contract_hash") != run.name:
        raise ValueError("TRACE job belongs to a different run contract")
    receipt = run / "state" / "job_outputs" / f"{job_id}.json"
    if receipt.is_file():
        persisted = load_canonical_json(receipt)
        if persisted.get("job_id") != job_id or persisted.get("kind") != spec.kind:
            raise ValueError("job receipt belongs to a different semantic job")
        output_sha256 = str(persisted["output_sha256"])
        if output_sha256 != record_sha256(
            {"job_id": job_id, "kind": spec.kind, "outputs": persisted["outputs"]}
        ):
            raise ValueError("job receipt output identity changed")
        return output_sha256

    examples = load_examples(run / "manifests" / "examples.jsonl")
    handlers: dict[str, Callable[[Path, JobSpec, Sequence[TraceExample]], Mapping[str, object]]] = {
        "train_leaf": _train_leaf,
        "train_baseline": _train_baseline,
        "merge_node": _merge_node,
        "prepare_prompt_embeddings": _prepare_embeddings,
        "evaluate_vamp": _evaluate_vamp,
        "evaluate_baseline": _evaluate_baseline,
        "evaluate_final_baseline": _evaluate_final_baseline,
        "retrained_parent_oracle": _retrained_parent_oracle,
        "build_report": _build_report,
        "build_policy_report": _build_report,
    }
    if spec.kind not in handlers:
        raise ValueError(f"unsupported TRACE job kind: {spec.kind}")
    outputs = dict(handlers[spec.kind](run, spec, examples))
    core: dict[str, object] = {
        "job_id": job_id,
        "kind": spec.kind,
        "outputs": outputs,
    }
    output_sha256 = record_sha256(core)
    publish_immutable_json(
        receipt,
        {
            **core,
            "format": "trace-job-output-v1",
            "output_sha256": output_sha256,
        },
    )
    return output_sha256


def _registered_job(run: Path, job_id: str) -> JobSpec:
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    matches = tuple(status.spec for status in ledger.statuses if status.spec.job_id == job_id)
    if len(matches) != 1:
        raise KeyError(f"TRACE job is not registered exactly once: {job_id}")
    return matches[0]


def _model_revision(run: Path) -> str:
    manifest = load_canonical_json(run / "manifests" / "run.json")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("run manifest model identity changed")
    return str(manifest["model_revision"])


def _train_leaf(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    result = run_leaf_training(
        examples,
        int(spec.payload["arrival"]),
        _model_revision(run),
        "cuda",
        run,
        should_pause=_PAUSE.is_set,
    )
    return {
        "artifact_directory": str(result.artifact_directory),
        "artifact_sha256": result.artifact_sha256,
        "optimizer_steps": result.training.optimizer_steps,
        "presentations": result.training.presentations,
    }


def _train_baseline(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    condition = str(spec.payload["condition"])
    plans = baseline_plans(examples)
    if condition not in plans:
        raise ValueError(f"unregistered TRACE baseline: {condition}")
    results = []
    for plan in plans[condition]:
        suffix = plan.name.removeprefix("taskwise_lora-")
        target = (
            run / "baselines" / condition / suffix
            if condition == "taskwise_lora"
            else run / "baselines" / condition / "final"
        )
        result = run_training_artifact(
            plan=plan,
            examples_by_id={example.example_id: example for example in examples},
            model_revision=_model_revision(run),
            device="cuda",
            target_directory=target,
            checkpoint_path=run / "checkpoints" / f"baseline-{plan.plan_hash}.pt",
            ledger_path=run / "logs" / f"baseline-{plan.plan_hash}.jsonl",
            work_root=run / "work",
            snapshot_root=(
                run / "baselines" / condition / "snapshots"
                if condition in ("seq_lora_reference", "seq_lora_40")
                else None
            ),
            should_pause=_PAUSE.is_set,
        )
        results.append(
            {
                "artifact_directory": str(result.artifact_directory),
                "artifact_sha256": result.artifact_sha256,
                "plan_hash": plan.plan_hash,
            }
        )
    return {"condition": condition, "artifacts": results}


def _merge_node(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    payload = spec.payload
    policy = _policy(_mapping(payload["policy"]))
    if str(payload["policy_hash"]) != policy.policy_hash:
        raise ValueError("merge policy hash differs from its payload")
    left = node_from_record(_mapping(payload["left"]))
    right = node_from_record(_mapping(payload["right"]))
    parent = node_from_record(_mapping(payload["parent"]))
    target = run / "derived" / policy.policy_hash / "nodes" / parent.node_id
    expected_diagnostic = (
        run
        / "evaluations"
        / "merge_diagnostics"
        / policy.policy_hash
        / f"{parent.node_id}.json"
    )
    if target.is_dir() and expected_diagnostic.is_file():
        return {
            "artifact_directory": str(target),
            "artifact_sha256": validate_artifact_directory(target),
            "diagnostic_path": (
                str(expected_diagnostic) if expected_diagnostic.is_file() else None
            ),
            "diagnostic_sha256": (
                file_sha256(expected_diagnostic)
                if expected_diagnostic.is_file()
                else None
            ),
            "merge_cache_reused": True,
        }
    left_directory = node_artifact_directory(run, left, policy.policy_hash)
    right_directory = node_artifact_directory(run, right, policy.policy_hash)
    child_hashes = (
        validate_artifact_directory(left_directory),
        validate_artifact_directory(right_directory),
    )
    cache = (
        run
        / "merge_cache"
        / f"{child_hashes[0]}__{child_hashes[1]}"
        / policy.merge_config_hash
    )
    cache_reused = cache.is_dir()
    if cache_reused:
        cache_sha256 = validate_artifact_directory(cache)
    else:
        with TemporaryDirectory(prefix="merge-cache-", dir=run / "work") as temporary:
            output = Path(temporary)
            consolidate_adapters(
                left_directory / ADAPTER_FILENAME,
                right_directory / ADAPTER_FILENAME,
                left.represented_examples,
                right.represented_examples,
                policy,
                output,
                retain_precompress=(
                    bool(payload["diagnostic_precompress"])
                    and policy.method == "core_tsv_r8"
                ),
                device="cuda",
            )
            cache_sha256 = publish_artifact_directory(output, cache)
    left_reservoir = node_reservoir(run, left, policy)
    right_reservoir = node_reservoir(run, right, policy)
    repair_entries, parent_reservoir = merge_reservoirs(
        left_reservoir,
        right_reservoir,
        parent.represented_examples,
        policy.repair_fraction,
    )
    merge_metrics = load_canonical_json(cache / "merge_metrics.json")
    lineage = {
        "cache_artifact_sha256": cache_sha256,
        "child_artifact_sha256": list(child_hashes),
        "format": "trace-derived-node-lineage-v1",
        "merge_config_hash": policy.merge_config_hash,
        "node": parent.as_record(),
        "policy": policy.as_record(),
        "policy_hash": policy.policy_hash,
        "repair_config_hash": policy.repair_config_hash,
        "task_composition": dict(
            sorted(
                Counter(
                    example.task
                    for example in examples
                    if example.arrival is not None
                    and parent.start_arrival <= example.arrival <= parent.end_arrival
                ).items()
            )
        ),
    }
    records: tuple[tuple[str, Mapping[str, object]], ...] = (
        ("lineage.json", lineage),
        ("merge_metrics.json", merge_metrics),
        ("reservoir_priorities.json", reservoir_record(parent_reservoir)),
        (
            "repair_config.json",
            {
                "format": "trace-repair-config-v1",
                "repair_config_hash": policy.repair_config_hash,
                "repair_examples": len(repair_entries),
                "repair_fraction": policy.repair_fraction,
            },
        ),
    )
    if repair_entries:
        plan = repair_plan(
            {example.example_id: example for example in examples},
            tuple(entry.example_id for entry in repair_entries),
            parent.node_id,
        )
        result = run_training_artifact(
            plan=plan,
            examples_by_id={example.example_id: example for example in examples},
            model_revision=_model_revision(run),
            device="cuda",
            target_directory=target,
            checkpoint_path=run / "checkpoints" / f"repair-{policy.policy_hash}-{parent.node_id}.pt",
            ledger_path=run / "logs" / f"repair-{policy.policy_hash}-{parent.node_id}.jsonl",
            work_root=run / "work",
            config=repair_training_config(
                rank=policy.output_rank,
                alpha=policy.parent_alpha,
                learning_rate=policy.repair_learning_rate,
            ),
            initial_adapter=cache / ADAPTER_FILENAME,
            extra_records=records,
            should_pause=_PAUSE.is_set,
        )
        artifact_sha256 = result.artifact_sha256
    else:
        with TemporaryDirectory(prefix="derived-node-", dir=run / "work") as temporary:
            output = Path(temporary)
            shutil.copyfile(cache / ADAPTER_FILENAME, output / ADAPTER_FILENAME)
            shutil.copyfile(cache / "adapter_config.json", output / "adapter_config.json")
            for filename, record in records:
                publish_immutable_json(output / filename, record)
            artifact_sha256 = publish_artifact_directory(output, target)
    diagnostic_path = _merge_validation_diagnostics(
        run,
        spec,
        examples,
        policy,
        left,
        right,
        parent,
        left_directory,
        right_directory,
        cache,
        target,
        include_precompression=bool(payload["diagnostic_precompress"]),
    )
    return {
        "artifact_directory": str(target),
        "artifact_sha256": artifact_sha256,
        "diagnostic_path": str(diagnostic_path) if diagnostic_path is not None else None,
        "diagnostic_sha256": (
            file_sha256(diagnostic_path) if diagnostic_path is not None else None
        ),
        "merge_cache_reused": cache_reused,
        "repair_examples": len(repair_entries),
    }


def _prepare_embeddings(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    del spec
    tensor_path = run / "manifests" / "prompt_embeddings.safetensors"
    if tensor_path.is_file():
        load_prompt_embeddings(tensor_path)
        digest = file_sha256(tensor_path)
    else:
        bundle = load_fresh_lora_bundle(
            _model_revision(run), "cuda", "trace-frozen-prompt-embeddings"
        )
        runtime = PeftCandidateRuntime(bundle.model, bundle.tokenizer)
        digest = build_prompt_embedding_cache(
            runtime,
            examples,
            run / "logs" / "prompt_embeddings.jsonl",
            tensor_path,
            _PAUSE.is_set,
        )
    return {"embedding_count": len(examples), "tensor_sha256": digest}


def _evaluate_vamp(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    payload = spec.payload
    policy_hash = str(payload["policy_hash"])
    stage = int(payload["stage"])
    task_index = int(payload["task_index"])
    nodes = tuple(
        node_from_record(_mapping(value)) for value in _sequence(payload["active_nodes"])
    )
    candidates = (
        Candidate("base", None),
        *tuple(
            _adapter_candidate(
                node.node_id,
                node_artifact_directory(run, node, policy_hash) / ADAPTER_FILENAME,
            )
            for node in nodes
        ),
    )
    task = TASKS[task_index - 1]
    validation = _task_examples(examples, task.name, "validation")
    test = _task_examples(examples, task.name, "test")
    output_root = run / "evaluations" / policy_hash / f"stage-{stage:02d}" / task.name
    result_path = output_root / "result.json"
    if result_path.is_file():
        return {"result_path": str(result_path), "result_sha256": file_sha256(result_path)}
    bundle = load_fresh_lora_bundle(_model_revision(run), "cuda", spec.job_id)
    runtime = PeftCandidateRuntime(bundle.model, bundle.tokenizer)
    validation_outputs = evaluate_candidate_cache(
        runtime,
        candidates,
        validation,
        stage,
        output_root / "validation-candidates.jsonl",
        _PAUSE.is_set,
    )
    test_outputs = evaluate_candidate_cache(
        runtime,
        candidates,
        test,
        stage,
        output_root / "test-candidates.jsonl",
        _PAUSE.is_set,
    )
    embeddings = load_prompt_embeddings(run / "manifests" / "prompt_embeddings.safetensors")
    task_selection = task_aware_candidates(candidates, validation, validation_outputs)
    centroids = hierarchy_centroids(nodes, examples, embeddings)
    validation_routed = route_outputs(
        candidates,
        validation,
        validation_outputs,
        centroids,
        {example.example_id: embeddings[example.example_id] for example in validation},
        task_selection,
    )
    routed = route_outputs(
        candidates,
        test,
        test_outputs,
        centroids,
        {example.example_id: embeddings[example.example_id] for example in test},
        task_selection,
    )
    router_scores = _routed_scores(task.name, test, routed)
    validation_router_scores = _routed_scores(
        task.name,
        validation,
        validation_routed,
    )
    record: dict[str, object] = {
        "candidate_evaluation": {
            "test": _candidate_accounting(test_outputs),
            "validation": _candidate_accounting(validation_outputs),
        },
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "condition": _policy_condition(_policy(_mapping(payload["policy"]))),
        "format": "trace-evaluation-result-v1",
        "policy_hash": policy_hash,
        "router_scores": router_scores,
        "stage": stage,
        "task": task.name,
        "task_aware_candidate": task_selection[task.name],
        "task_index": task_index,
        "test_examples": len(test),
        "validation_examples": len(validation),
        "validation_router_scores": validation_router_scores,
    }
    publish_immutable_json(result_path, record)
    return {"result_path": str(result_path), "result_sha256": file_sha256(result_path)}


def _evaluate_baseline(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    condition = str(spec.payload["condition"])
    stage = int(spec.payload["stage"])
    snapshot = (
        TASKS[stage - 1].name
        if condition == "seq_lora_reference"
        else f"arrival-{stage * 5:02d}"
    )
    adapter = run / "baselines" / condition / "snapshots" / snapshot / ADAPTER_FILENAME
    return _evaluate_direct_adapter(run, spec, examples, condition, stage, adapter)


def _evaluate_final_baseline(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    condition = str(spec.payload["condition"])
    task_index = int(spec.payload["task_index"])
    adapter = (
        None
        if condition == "frozen_base"
        else (
            run / "baselines" / condition / TASKS[task_index - 1].name / ADAPTER_FILENAME
            if condition == "taskwise_lora"
            else run / "baselines" / condition / "final" / ADAPTER_FILENAME
        )
    )
    return _evaluate_direct_adapter(run, spec, examples, condition, 8, adapter)


def _evaluate_direct_adapter(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
    condition: str,
    stage: int,
    adapter: Path | None,
) -> Mapping[str, object]:
    task_index = int(spec.payload["task_index"])
    task = TASKS[task_index - 1]
    test = _task_examples(examples, task.name, "test")
    output_root = run / "evaluations" / condition / f"stage-{stage:02d}" / task.name
    result_path = output_root / "result.json"
    if result_path.is_file():
        return {"result_path": str(result_path), "result_sha256": file_sha256(result_path)}
    candidate = (
        Candidate("base", None)
        if adapter is None
        else _adapter_candidate(condition, adapter)
    )
    bundle = load_fresh_lora_bundle(_model_revision(run), "cuda", spec.job_id)
    outputs = evaluate_candidate_cache(
        PeftCandidateRuntime(bundle.model, bundle.tokenizer),
        (candidate,),
        test,
        stage,
        output_root / "test-candidates.jsonl",
        _PAUSE.is_set,
    )
    score = score_task(
        task.name,
        tuple(example.prompt for example in test),
        tuple(output.prediction for output in outputs),
        tuple(example.answer for example in test),
    )
    publish_immutable_json(
        result_path,
        {
            "candidate_evaluation": {"test": _candidate_accounting(outputs)},
            "condition": condition,
            "format": "trace-evaluation-result-v1",
            "router_scores": {"direct": score},
            "stage": stage,
            "task": task.name,
            "task_index": task_index,
            "test_examples": len(test),
        },
    )
    return {"result_path": str(result_path), "result_sha256": file_sha256(result_path)}


def _retrained_parent_oracle(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    start, end = int(spec.payload["start_arrival"]), int(spec.payload["end_arrival"])
    plan = retrained_parent_plan(examples, start, end)
    target = run / "baselines" / "retrained_parents" / f"{start:02d}-{end:02d}"
    result = run_training_artifact(
        plan=plan,
        examples_by_id={example.example_id: example for example in examples},
        model_revision=_model_revision(run),
        device="cuda",
        target_directory=target,
        checkpoint_path=run / "checkpoints" / f"oracle-{plan.plan_hash}.pt",
        ledger_path=run / "logs" / f"oracle-{plan.plan_hash}.jsonl",
        work_root=run / "work",
        should_pause=_PAUSE.is_set,
        extra_records=((
            "calibration.json",
            {
                "end_arrival": end,
                "format": "trace-retrained-parent-calibration-v1",
                "node": _mapping(spec.payload["node"]),
                "policy_hashes": list(_sequence(spec.payload["policy_hashes"])),
                "start_arrival": start,
            },
        ),),
    )
    node = node_from_record(_mapping(spec.payload["node"]))
    policy_hashes = tuple(str(value) for value in _sequence(spec.payload["policy_hashes"]))
    calibration_path = (
        run / "evaluations" / "retrained_parents" / f"{start:02d}-{end:02d}" / "result.json"
    )
    if not calibration_path.is_file():
        task_names = tuple(
            task.name
            for task in TASKS
            if any(
                example.task == task.name
                and example.arrival is not None
                and start <= example.arrival <= end
                for example in examples
            )
        )
        validation = tuple(
            example
            for example in examples
            if example.split == "validation" and example.task in task_names
        )
        candidates = (
            _adapter_candidate("retrained_parent", target / ADAPTER_FILENAME),
            *tuple(
                _adapter_candidate(
                    policy_hash,
                    node_artifact_directory(run, node, policy_hash) / ADAPTER_FILENAME,
                )
                for policy_hash in policy_hashes
            ),
        )
        bundle = load_fresh_lora_bundle(_model_revision(run), "cuda", spec.job_id)
        outputs = evaluate_candidate_cache(
            PeftCandidateRuntime(bundle.model, bundle.tokenizer),
            candidates,
            validation,
            min(8, (end + 4) // 5),
            calibration_path.parent / "candidates.jsonl",
            _PAUSE.is_set,
        )
        by_key = {
            (output.example_id, output.candidate_id): output.prediction for output in outputs
        }
        scores = {
            task_name: {
                candidate.candidate_id: score_task(
                    task_name,
                    tuple(example.prompt for example in validation if example.task == task_name),
                    tuple(
                        by_key[(example.example_id, candidate.candidate_id)]
                        for example in validation
                        if example.task == task_name
                    ),
                    tuple(example.answer for example in validation if example.task == task_name),
                )
                for candidate in candidates
            }
            for task_name in task_names
        }
        publish_immutable_json(
            calibration_path,
            {
                "candidate_evaluation": {
                    "validation": _candidate_accounting(outputs)
                },
                "candidate_scores": scores,
                "format": "trace-retrained-parent-result-v1",
                "node": node.as_record(),
                "split": "validation",
            },
        )
    return {
        "artifact_directory": str(target),
        "artifact_sha256": result.artifact_sha256,
        "calibration_result": str(calibration_path),
        "calibration_result_sha256": file_sha256(calibration_path),
        "presentations": result.training.presentations,
    }


def _build_report(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
) -> Mapping[str, object]:
    del examples
    acceptance_path: Path | None = None
    if spec.kind == "build_policy_report":
        expected = tuple(
            str(value) for value in _sequence(spec.payload["leaf_adapter_hashes"])
        )
        arrivals = load_canonical_json(run / "manifests" / "arrivals.json")
        identities = tuple(str(value) for value in _sequence(arrivals["arrival_ids"]))
        if len(expected) != 40 or len(identities) != 40:
            raise ValueError("policy reuse acceptance requires all 40 leaves")
        observed_values = []
        for identity in identities:
            directory = run / "leaves" / identity
            validate_artifact_directory(directory)
            observed_values.append(file_sha256(directory / ADAPTER_FILENAME))
        observed = tuple(observed_values)
        if observed != expected:
            raise RuntimeError("immutable leaf adapter hashes changed during policy rebuild")
        policy = _policy(_mapping(spec.payload["policy"]))
        if policy.policy_hash != str(spec.payload["policy_hash"]):
            raise ValueError("policy report hash differs from its policy payload")
        acceptance_path = publish_immutable_json(
            run / "reports" / f"artifact-reuse-{policy.policy_hash}.json",
            {
                "format": "trace-artifact-reuse-acceptance-v1",
                "leaf_adapter_hashes_unchanged": True,
                "leaf_hashes": list(observed),
                "leaf_training_steps_reused_percent": 100,
                "new_gradient_work": "repair_only"
                if policy.repair_fraction
                else "none",
                "policy": policy.as_record(),
                "policy_hash": policy.policy_hash,
            },
        )
    report = build_report(run)
    return {
        "acceptance_path": str(acceptance_path) if acceptance_path is not None else None,
        "acceptance_sha256": (
            file_sha256(acceptance_path) if acceptance_path is not None else None
        ),
        "calibration_csv_path": str(report.calibration_csv_path),
        "csv_path": str(report.csv_path),
        "evaluation_rows": report.evaluation_rows,
        "html_path": str(report.html_path),
        "lineage_svg_path": str(report.lineage_svg_path),
        "markdown_path": str(report.markdown_path),
        "merge_diagnostics_csv_path": str(report.merge_diagnostics_csv_path),
        "parquet_path": str(report.parquet_path),
    }


def _task_examples(
    examples: Sequence[TraceExample],
    task: str,
    split: str,
) -> tuple[TraceExample, ...]:
    selected = tuple(example for example in examples if example.task == task and example.split == split)
    if not selected:
        raise ValueError(f"TRACE {task}/{split} split is empty")
    return selected


def _merge_validation_diagnostics(
    run: Path,
    spec: JobSpec,
    examples: Sequence[TraceExample],
    policy: MergePolicy,
    left: HierarchyNode,
    right: HierarchyNode,
    parent: HierarchyNode,
    left_directory: Path,
    right_directory: Path,
    cache: Path,
    target: Path,
    *,
    include_precompression: bool,
) -> Path:
    result_path = (
        run
        / "evaluations"
        / "merge_diagnostics"
        / policy.policy_hash
        / f"{parent.node_id}.json"
    )
    if result_path.is_file():
        return result_path
    task_names = tuple(
        task.name
        for task in TASKS
        if any(
            example.task == task.name
            and example.arrival is not None
            and parent.start_arrival <= example.arrival <= parent.end_arrival
            for example in examples
        )
    )
    validation = tuple(
        example
        for task_name in task_names
        for example in sorted(
            (
                item
                for item in examples
                if item.split == "validation" and item.task == task_name
            ),
            key=lambda item: item.example_id,
        )[:16]
    )
    torch.cuda.empty_cache()
    bundle = load_fresh_lora_bundle(_model_revision(run), "cuda", spec.job_id)
    runtime = PeftCandidateRuntime(bundle.model, bundle.tokenizer)
    base_candidates = (
        _adapter_candidate("left_child", left_directory / ADAPTER_FILENAME),
        _adapter_candidate("right_child", right_directory / ADAPTER_FILENAME),
        _adapter_candidate("post_compression", cache / ADAPTER_FILENAME),
        *(
            (_adapter_candidate("post_repair", target / ADAPTER_FILENAME),)
            if policy.repair_fraction
            else ()
        ),
    )
    with TemporaryDirectory(prefix="precompress-diagnostic-", dir=run / "work") as temporary:
        candidates = base_candidates
        if policy.method == "core_tsv_r8" and include_precompression:
            precompress_path, rank, alpha = materialize_precompress_adapter(
                cache / "core_cache.safetensors",
                temporary,
                policy,
            )
            candidates = (
                *base_candidates,
                Candidate("pre_compression", precompress_path, rank=rank, alpha=alpha),
            )
        nll = {
            candidate.candidate_id: runtime.answer_nll(candidate, validation)
            for candidate in candidates
        }
        if not policy.repair_fraction:
            nll["post_repair"] = nll["post_compression"]
    best_child = min(nll["left_child"], nll["right_child"])
    record: dict[str, object] = {
        "answer_nll": nll,
        "best_child_answer_nll": best_child,
        "format": "trace-merge-validation-diagnostic-v1",
        "merge_damage": nll["post_compression"] - best_child,
        "node": parent.as_record(),
        "policy": policy.as_record(),
        "policy_hash": policy.policy_hash,
        "postcompression_damage": (
            nll["post_compression"] - nll["pre_compression"]
            if "pre_compression" in nll
            else None
        ),
        "repair_recovery": nll["post_compression"] - nll["post_repair"],
        "task_names": list(task_names),
        "validation_examples": len(validation),
    }
    publish_immutable_json(result_path, record)
    if _PAUSE.is_set():
        raise InterruptedError("TRACE merge diagnostics paused after durable publication")
    return result_path


def _adapter_candidate(candidate_id: str, adapter_path: Path) -> Candidate:
    config = json.loads(
        (adapter_path.parent / "adapter_config.json").read_text(encoding="utf-8")
    )
    if type(config) is not dict:
        raise ValueError("adapter configuration is malformed")
    return Candidate(
        candidate_id,
        adapter_path,
        rank=int(config["r"]),
        alpha=int(config["lora_alpha"]),
    )


def _policy(value: Mapping[str, object]) -> MergePolicy:
    return MergePolicy(
        method=cast(MergeMethod, str(value["method"])),
        output_rank=int(value["output_rank"]),
        parent_alpha=int(value["parent_alpha"]),
        core_scale=float(value["core_scale"]) if value["core_scale"] is not None else None,
        repair_fraction=float(value["repair_fraction"]),
        repair_learning_rate=float(value["repair_learning_rate"]),
        algorithm_version=str(value["algorithm_version"]),
    )


def _policy_condition(policy: MergePolicy) -> str:
    stem = (
        f"vamp_svd_r{policy.output_rank}"
        if policy.method == "svd_mean_r8"
        else (
            f"vamp_core_tsv_r{policy.output_rank}_"
            f"scale{int(round((policy.core_scale or 0.0) * 10)):02d}"
        )
    )
    if policy.parent_alpha != 32:
        stem += f"_alpha{policy.parent_alpha}"
    if policy.repair_learning_rate != 5.0e-5:
        stem += f"_repairlr{policy.repair_learning_rate:g}"
    return f"{stem}_repair{int(round(policy.repair_fraction * 100)):03d}"


def _candidate_accounting(
    outputs: Sequence[CandidateOutput],
) -> dict[str, float | int]:
    """Aggregate exact resumable candidate-case token and wall-clock accounting."""
    return {
        "candidate_cases": len(outputs),
        "case_wall_seconds": sum(output.case_wall_seconds for output in outputs),
        "generated_tokens": sum(output.generated_tokens or 0 for output in outputs),
        "prompt_tokens": sum(output.prompt_tokens or 0 for output in outputs),
    }


def _routed_scores(
    task: str,
    examples: Sequence[TraceExample],
    routed: Sequence[RoutedPrediction],
) -> dict[str, float]:
    """Score each registered router from one immutable routed-output projection."""
    return {
        router: score_task(
            task,
            tuple(example.prompt for example in examples),
            tuple(
                next(
                    row.prediction
                    for row in routed
                    if row.example_id == example.example_id and row.router == router
                )
                for example in examples
            ),
            tuple(example.answer for example in examples),
        )
        for router in (
            "prompt_nll",
            "frozen_prompt_centroid",
            "task_aware",
            "answer_oracle",
        )
    }


def _mapping(value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError("TRACE job mapping payload is malformed")
    return value


def _sequence(value: object) -> Sequence[object]:
    if type(value) is not list:
        raise ValueError("TRACE job sequence payload is malformed")
    return value


__all__ = ["execute_job", "install_pause_handlers"]
