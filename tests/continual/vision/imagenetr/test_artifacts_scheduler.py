from pathlib import Path

import pytest
import torch

from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.protocol import (
    JobSpec,
    MergePolicy,
    NodeArtifact,
    material_tree_manifest,
)
from apm.continual.vision.imagenetr.proxy_memory import TensorCache
from apm.continual.vision.imagenetr.scheduler import LocalScheduler


SHA = "1" * 64


def _node(**changes: object) -> NodeArtifact:
    values: dict[str, object] = {
        "run_hash": SHA,
        "software_manifest_hash": "2" * 64,
        "git_commit": "abc",
        "creation_timestamp_utc": "2026-08-19T00:00:00Z",
        "level": 0,
        "first_task": 0,
        "last_task": 0,
        "represented_task_ids": (0,),
        "represented_class_ids": (0, 1, 2, 3),
        "represented_train_image_count": 100,
        "parent_hashes": (),
        "unrepaired_parent_hash": None,
        "consolidation_method": "leaf",
        "consolidation_config_hash": "3" * 64,
        "repair_config_hash": "4" * 64,
        "lora_sha256": "5" * 64,
        "classifier_sha256": "6" * 64,
        "proxy_image_ids": ("7" * 64,),
        "repair_image_ids": (),
        "source_priority_hash": "8" * 64,
        "training_optimizer_steps": 10,
    }
    values.update(changes)
    return NodeArtifact(**values)


def test_node_identity_excludes_informational_time_and_git_but_binds_tensors() -> None:
    original = _node()
    informational = _node(
        git_commit="def", creation_timestamp_utc="2026-08-20T00:00:00Z"
    )
    changed_tensor = _node(lora_sha256="9" * 64)
    assert original.content_hash == informational.content_hash
    assert original.content_hash != changed_tensor.content_hash


def test_merge_cache_is_repair_independent_but_policy_identity_is_not() -> None:
    base = dict(
        method="svd",
        output_rank=16,
        scale=1.0,
        weighting="source_image_count",
        repair_config_hash="a" * 64,
        proxy_size=16,
    )
    no_repair = MergePolicy(**base, repair_fraction=0.0)
    repaired = MergePolicy(**base, repair_fraction=0.05)
    assert no_repair.merge_cache_hash == repaired.merge_cache_hash
    assert no_repair.content_hash != repaired.content_hash


def test_scheduler_resume_never_reexecutes_complete_job(tmp_path: Path) -> None:
    spec = JobSpec.create(SHA, "unit", payload={"value": 1})
    calls = []
    scheduler = LocalScheduler(tmp_path / "state.json", SHA)
    assert scheduler.execute(spec, lambda: calls.append(1) or {"ok": True}) == {"ok": True}
    resumed = LocalScheduler(tmp_path / "state.json", SHA)
    assert resumed.execute(spec, lambda: calls.append(2) or {"ok": False}) == {"ok": True}
    assert calls == [1]


def test_immutable_directory_publication_detects_tampering(tmp_path: Path) -> None:
    work, target = tmp_path / "work", tmp_path / "target"
    work.mkdir()
    (work / "value.txt").write_text("first", encoding="utf-8")
    publish_artifact_directory(work, target)
    validate_artifact_directory(target)
    (target / "value.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_artifact_directory(target)


def test_tensor_cache_publishes_once_and_reuses_exact_semantic_key(tmp_path: Path) -> None:
    cache = TensorCache(tmp_path / "cache", "unit-cache-v1")
    calls = []
    first, first_hit = cache.get_or_compute(
        {"input": "fixed"},
        lambda: calls.append(1) or {"value": torch.arange(5)},
    )
    second, second_hit = cache.get_or_compute(
        {"input": "fixed"},
        lambda: calls.append(2) or {"value": torch.zeros(5)},
    )
    assert not first_hit and second_hit and calls == [1]
    torch.testing.assert_close(first["value"], second["value"])
    assert not tuple((tmp_path / "cache").glob(".*"))


def test_material_code_identity_does_not_depend_on_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source, config = project / "src", project / "config.yaml"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    config.write_text("seed: 1993\n", encoding="utf-8")
    first = material_tree_manifest((source, config))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    second = material_tree_manifest((source, config))
    assert first == second
