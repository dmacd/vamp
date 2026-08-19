from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    canonical_json_bytes,
    record_sha256,
)
from apm.continual.trace.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.trace.config import load_merge_policy
from apm.continual.trace.lineage import build_hierarchy
from apm.continual.trace.modeling import (
    model_source_manifest,
    model_source_manifest_sha256,
    verify_model_source_metadata,
)
from apm.continual.trace.protocol import (
    MODEL_FILE_IDENTITIES,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SOURCE_ID,
    MODEL_SOURCE_REVISION,
    MergePolicy,
    default_merge_policies,
)
from apm.continual.trace.reservoirs import (
    merge_reservoirs,
    prioritized_entries,
    select_reservoir,
)
from apm.continual.trace.training_jobs import repair_training_config
from apm.continual.trace.worker import _policy_condition


def test_public_model_source_is_pinned_to_canonical_meta_bytes() -> None:
    manifest = model_source_manifest()
    siblings = tuple(
        SimpleNamespace(
            blob_id=git_blob,
            lfs={"sha256": sha256} if path == "model.safetensors" else None,
            rfilename=path,
            size=size,
        )
        for path, size, sha256, git_blob in MODEL_FILE_IDENTITIES
    )

    verify_model_source_metadata(
        SimpleNamespace(sha=MODEL_SOURCE_REVISION, siblings=siblings)
    )

    assert manifest["canonical_model_id"] == MODEL_ID
    assert manifest["canonical_revision"] == MODEL_REVISION
    assert manifest["source_model_id"] == MODEL_SOURCE_ID
    assert manifest["source_revision"] == MODEL_SOURCE_REVISION
    assert model_source_manifest_sha256() == record_sha256(manifest)


def test_public_model_source_rejects_metadata_drift() -> None:
    with pytest.raises(RuntimeError, match="unexpected revision"):
        verify_model_source_metadata(SimpleNamespace(sha="changed", siblings=()))


def test_trace_hierarchy_has_registered_arrival_40_topology() -> None:
    arrival_ids = tuple(record_sha256({"arrival": arrival}) for arrival in range(1, 41))

    hierarchy, merges = build_hierarchy(arrival_ids)

    assert hierarchy.topology() == (
        ((39, 39), (40, 40)),
        ((37, 38),),
        ((33, 36),),
        ((17, 24), (25, 32)),
        ((1, 16),),
    )
    assert len(hierarchy.active_nodes) == 7
    assert len(merges) == 33
    assert sum(merge.parent.represented_examples for merge in merges) == 12_200
    assert sum(round(merge.parent.represented_examples * 0.05) for merge in merges) == 610
    assert sum(node.represented_arrivals for node in hierarchy.active_nodes) == 40


def test_trace_policy_hash_covers_every_derivation_choice() -> None:
    policies = default_merge_policies()

    assert len({policy.policy_hash for policy in policies}) == 4
    assert MergePolicy("core_tsv_r8", core_scale=0.3).policy_hash != MergePolicy(
        "core_tsv_r8", core_scale=0.5
    ).policy_hash
    assert MergePolicy("svd_mean_r8", repair_fraction=0.0).merge_config_hash == MergePolicy(
        "svd_mean_r8", repair_fraction=0.05
    ).merge_config_hash
    with pytest.raises(ValueError, match="Core scale"):
        MergePolicy("svd_mean_r8", core_scale=0.3)


def test_repair_reservoir_is_deterministic_and_composable() -> None:
    identities = tuple(record_sha256({"example": index}) for index in range(200))
    left = select_reservoir(prioritized_entries(identities[:100]), 100, 0.05)
    right = select_reservoir(prioritized_entries(identities[100:]), 100, 0.05)

    repair, parent = merge_reservoirs(left, right, 200, 0.05)

    assert len(repair) == 10
    assert len(parent) == 10
    assert parent == tuple(sorted((*left, *right)))
    reverse_repair, reverse_parent = merge_reservoirs(right, left, 200, 0.05)
    assert (reverse_repair, reverse_parent) == (repair, parent)


def test_repair_training_matches_a_nondefault_parent_rank_and_alpha() -> None:
    config = repair_training_config(rank=12, alpha=36, learning_rate=2.0e-5)

    assert config.rank == 12
    assert config.alpha == 36
    assert config.scale == 3.0
    assert config.learning_rate == 2.0e-5
    assert _policy_condition(
        MergePolicy("core_tsv_r8", output_rank=12, core_scale=0.5)
    ) == "vamp_core_tsv_r12_scale05_repair000"


def test_policy_loader_keeps_nondefault_repair_optimizer_identity(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "method: svd_mean_r8\n"
        "output_rank: 12\n"
        "parent_alpha: 36\n"
        "repair_fraction: 0.10\n"
        "repair_learning_rate: 2.0e-5\n",
        encoding="utf-8",
    )

    policy = load_merge_policy(policy_path)

    assert policy.repair_learning_rate == 2.0e-5
    assert _policy_condition(policy) == "vamp_svd_r12_alpha36_repairlr2e-05_repair010"


def test_shared_ledger_repairs_only_a_torn_final_row(tmp_path: Path) -> None:
    ledger_path = tmp_path / "events.jsonl"
    ledger = ChainedJsonlLedger(ledger_path, "test-events-v1")
    first = ledger.append({"event": "one"})
    with ledger_path.open("ab") as output:
        output.write(b'{"event":"torn"')

    repaired = ChainedJsonlLedger(ledger_path, "test-events-v1")

    assert repaired.rows == (first,)
    assert ledger_path.read_bytes() == canonical_json_bytes(first)


def test_artifact_directory_publication_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.safetensors").write_bytes(b"adapter-one")
    target = tmp_path / "published"

    identity = publish_artifact_directory(source, target)

    assert validate_artifact_directory(target) == identity
    assert json.loads((target / "artifact.json").read_text())["artifact_sha256"] == identity
    (source / "adapter.safetensors").write_bytes(b"adapter-two")
    with pytest.raises(ValueError, match="immutable artifact changed"):
        publish_artifact_directory(source, target)
