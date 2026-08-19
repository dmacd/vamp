"""Deterministic primary TRACE job graph construction."""

from __future__ import annotations

from collections.abc import Sequence

from apm.continual.artifacts import require_sha256
from apm.continual.trace.jobs import JobSpec
from apm.continual.trace.lineage import HierarchyNode, empty_hierarchy, insert_arrival
from apm.continual.trace.protocol import EVALUATION_ARRIVALS, MergePolicy


_DIAGNOSTIC_INTERVALS = {(1, 2), (5, 6), (1, 4), (1, 16)}


def build_primary_dag(
    arrival_ids: Sequence[str],
    policies: Sequence[MergePolicy],
    run_contract_hash: str,
) -> tuple[JobSpec, ...]:
    """Build leaf, baseline, hierarchy, diagnostic, evaluation, and report jobs."""
    if len(arrival_ids) != 40 or len(policies) != 4:
        raise ValueError("primary TRACE DAG requires 40 arrivals and four VAMP policies")
    require_sha256(run_contract_hash, "TRACE run contract")
    jobs: list[JobSpec] = []
    state = empty_hierarchy()
    policy_producers: dict[tuple[str, str], str] = {}
    nodes_by_interval: dict[tuple[int, int], HierarchyNode] = {}
    snapshots: dict[int, tuple[HierarchyNode, ...]] = {}
    for arrival, arrival_id in enumerate(arrival_ids, start=1):
        next_state, merges = insert_arrival(state, arrival_id)
        leaf = next(node for node in next_state.levels[0] if node.start_arrival == arrival)
        leaf_job = JobSpec.create(
            "train_leaf",
            "gpu",
            10,
            (),
            {
                "arrival": arrival,
                "arrival_id": arrival_id,
                "node": leaf.as_record(),
                "run_contract_hash": run_contract_hash,
            },
        )
        jobs.append(leaf_job)
        nodes_by_interval[(arrival, arrival)] = leaf
        for policy in policies:
            policy_producers[(policy.policy_hash, leaf.node_id)] = leaf_job.job_id
            for merge in merges:
                dependencies = (
                    policy_producers[(policy.policy_hash, merge.left.node_id)],
                    policy_producers[(policy.policy_hash, merge.right.node_id)],
                )
                merge_job = JobSpec.create(
                    "merge_node",
                    "gpu",
                    20 if policy.method == "svd_mean_r8" else 21,
                    dependencies,
                    {
                        "diagnostic_precompress": (
                            merge.parent.start_arrival,
                            merge.parent.end_arrival,
                        )
                        in _DIAGNOSTIC_INTERVALS,
                        "left": merge.left.as_record(),
                        "parent": merge.parent.as_record(),
                        "policy": policy.as_record(),
                        "policy_hash": policy.policy_hash,
                        "right": merge.right.as_record(),
                        "run_contract_hash": run_contract_hash,
                    },
                )
                jobs.append(merge_job)
                policy_producers[(policy.policy_hash, merge.parent.node_id)] = merge_job.job_id
                nodes_by_interval[
                    (merge.parent.start_arrival, merge.parent.end_arrival)
                ] = merge.parent
        state = next_state
        if arrival in EVALUATION_ARRIVALS:
            snapshots[arrival] = state.active_nodes

    baseline_jobs = tuple(
        JobSpec.create(
            "train_baseline",
            "gpu",
            priority,
            (),
            {"condition": condition, "run_contract_hash": run_contract_hash},
        )
        for condition, priority in (
            ("seq_lora_reference", 5),
            ("seq_lora_40", 30),
            ("joint_iid_lora", 40),
            ("taskwise_lora", 41),
        )
    )
    jobs.extend(baseline_jobs)
    embedding_job = JobSpec.create(
        "prepare_prompt_embeddings",
        "gpu",
        45,
        (),
        {
            "encoder": "frozen_base_final_hidden_mean_v1",
            "run_contract_hash": run_contract_hash,
        },
    )
    jobs.append(embedding_job)
    evaluation_jobs: list[JobSpec] = []
    for stage, arrival in enumerate(EVALUATION_ARRIVALS, start=1):
        for policy in policies:
            dependencies = (
                embedding_job.job_id,
                *tuple(
                policy_producers[(policy.policy_hash, node.node_id)]
                for node in snapshots[arrival]
                ),
            )
            for task_index in range(1, stage + 1):
                evaluation_jobs.append(
                    JobSpec.create(
                        "evaluate_vamp",
                        "gpu",
                        50,
                        dependencies,
                        {
                            "active_nodes": [node.as_record() for node in snapshots[arrival]],
                            "arrival": arrival,
                            "method": policy.method,
                            "policy": policy.as_record(),
                            "policy_hash": policy.policy_hash,
                            "stage": stage,
                            "task_index": task_index,
                            "run_contract_hash": run_contract_hash,
                        },
                    )
                )
        for condition in ("seq_lora_reference", "seq_lora_40"):
            baseline = next(
                job for job in baseline_jobs if job.payload["condition"] == condition
            )
            for task_index in range(1, stage + 1):
                evaluation_jobs.append(
                    JobSpec.create(
                        "evaluate_baseline",
                        "gpu",
                        51,
                        (baseline.job_id,),
                        {
                            "condition": condition,
                            "stage": stage,
                            "task_index": task_index,
                            "run_contract_hash": run_contract_hash,
                        },
                    )
                )
    final_baselines = tuple(
        JobSpec.create(
            "evaluate_final_baseline",
            "gpu",
            52,
            (baseline.job_id,) if condition != "frozen_base" else (),
            {
                "condition": condition,
                "run_contract_hash": run_contract_hash,
                "task_index": task_index,
            },
        )
        for condition in ("frozen_base", "joint_iid_lora", "taskwise_lora")
        for baseline in (
            next(
                (job for job in baseline_jobs if job.payload["condition"] == condition),
                baseline_jobs[0],
            ),
        )
        for task_index in range(1, 9)
    )
    oracle_jobs = tuple(
        JobSpec.create(
            "retrained_parent_oracle",
            "gpu",
            60,
            tuple(
                policy_producers[(policy.policy_hash, nodes_by_interval[(start, end)].node_id)]
                for policy in policies
            ),
            {
                "end_arrival": end,
                "node": nodes_by_interval[(start, end)].as_record(),
                "policy_hashes": [policy.policy_hash for policy in policies],
                "run_contract_hash": run_contract_hash,
                "start_arrival": start,
            },
        )
        for start, end in sorted(_DIAGNOSTIC_INTERVALS)
    )
    jobs.extend(evaluation_jobs)
    jobs.extend(final_baselines)
    jobs.extend(oracle_jobs)
    report = JobSpec.create(
        "build_report",
        "cpu",
        100,
        tuple(job.job_id for job in (*evaluation_jobs, *final_baselines, *oracle_jobs)),
        {"report": "primary", "run_contract_hash": run_contract_hash},
    )
    jobs.append(report)
    return tuple(jobs)


def build_policy_dag(
    arrival_ids: Sequence[str],
    policy: MergePolicy,
    leaf_job_ids_by_arrival: Sequence[str],
    leaf_adapter_hashes: Sequence[str],
    embedding_job_id: str,
    run_contract_hash: str,
) -> tuple[JobSpec, ...]:
    """Build a leaf-reusing derived-policy tree, evaluations, and report jobs."""
    if (
        len(arrival_ids) != 40
        or len(leaf_job_ids_by_arrival) != 40
        or len(leaf_adapter_hashes) != 40
    ):
        raise ValueError("policy rebuild requires all 40 immutable leaves")
    require_sha256(run_contract_hash, "TRACE run contract")
    for digest in leaf_adapter_hashes:
        require_sha256(digest, "TRACE leaf adapter")
    jobs: list[JobSpec] = []
    state = empty_hierarchy()
    producers: dict[str, str] = {}
    snapshots: dict[int, tuple[HierarchyNode, ...]] = {}
    for arrival, (arrival_id, leaf_job_id) in enumerate(
        zip(arrival_ids, leaf_job_ids_by_arrival),
        start=1,
    ):
        state, merges = insert_arrival(state, arrival_id)
        leaf = next(node for node in state.levels[0] if node.start_arrival == arrival)
        producers[leaf.node_id] = leaf_job_id
        for merge in merges:
            merge_job = JobSpec.create(
                "merge_node",
                "gpu",
                20 if policy.method == "svd_mean_r8" else 21,
                (producers[merge.left.node_id], producers[merge.right.node_id]),
                {
                    "diagnostic_precompress": (
                        merge.parent.start_arrival,
                        merge.parent.end_arrival,
                    )
                    in _DIAGNOSTIC_INTERVALS,
                    "left": merge.left.as_record(),
                    "parent": merge.parent.as_record(),
                    "policy": policy.as_record(),
                    "policy_hash": policy.policy_hash,
                    "right": merge.right.as_record(),
                    "run_contract_hash": run_contract_hash,
                },
            )
            jobs.append(merge_job)
            producers[merge.parent.node_id] = merge_job.job_id
        if arrival in EVALUATION_ARRIVALS:
            snapshots[arrival] = state.active_nodes
    evaluations = tuple(
        JobSpec.create(
            "evaluate_vamp",
            "gpu",
            50,
            (
                embedding_job_id,
                *tuple(producers[node.node_id] for node in snapshots[arrival]),
            ),
            {
                "active_nodes": [node.as_record() for node in snapshots[arrival]],
                "arrival": arrival,
                "method": policy.method,
                "policy": policy.as_record(),
                "policy_hash": policy.policy_hash,
                "stage": stage,
                "task_index": task_index,
                "run_contract_hash": run_contract_hash,
            },
        )
        for stage, arrival in enumerate(EVALUATION_ARRIVALS, start=1)
        for task_index in range(1, stage + 1)
    )
    jobs.extend(evaluations)
    jobs.append(
        JobSpec.create(
            "build_policy_report",
            "cpu",
            100,
            tuple(job.job_id for job in evaluations),
            {
                "leaf_adapter_hashes": list(leaf_adapter_hashes),
                "policy": policy.as_record(),
                "policy_hash": policy.policy_hash,
                "report": "policy-rebuild",
                "run_contract_hash": run_contract_hash,
            },
        )
    )
    return tuple(jobs)


__all__ = ["build_policy_dag", "build_primary_dag"]
