from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

import pytest

from apm.data.text.tinyworlds_v2 import phase1_artifacts as artifact_module
from apm.data.text.tinyworlds_v2.json_contracts import canonical_json_bytes
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    PHASE1_ARTIFACT_FORMAT,
    PHASE1_ARTIFACT_SCHEMA_VERSION,
    Phase1ArtifactBuilder,
    Phase1ArtifactError,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
    promote_phase1_artifact_tree,
)


def _build_tree(root: Path) -> None:
    root.mkdir()
    builder = Phase1ArtifactBuilder(root)
    builder.write_json("profile/reference.json", {"documents": 20_000})
    builder.append_jsonl("routes/cheap/accepted.jsonl", {"story_id": "s2"})
    builder.append_jsonl("routes/cheap/accepted.jsonl", {"story_id": "s1"})
    builder.write_bytes(
        "cache/requests/a/attempts/000001/response.body",
        b"{ provider bytes are deliberately not canonical JSON }\n",
    )
    builder.finalize()


def _rewrite_manifest(
    root: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    manifest_path = root / "manifest.json"
    record = json.loads(manifest_path.read_bytes())
    assert isinstance(record, dict)
    mutate(record)
    core = {key: value for key, value in record.items() if key != "manifest_sha256"}
    record["manifest_sha256"] = sha256(canonical_json_bytes(core)).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(record))


def _artifact_record(record: dict[str, object], path: str) -> dict[str, object]:
    artifacts = record["artifacts"]
    assert isinstance(artifacts, list)
    return next(
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("path") == path
    )


def test_builder_lists_nested_files_and_validates_canonical_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reference-staging"
    _build_tree(root)

    manifest = load_phase1_artifact_tree(root)
    assert manifest.format == PHASE1_ARTIFACT_FORMAT
    assert manifest.schema_version == PHASE1_ARTIFACT_SCHEMA_VERSION
    assert tuple(artifact.path for artifact in manifest.artifacts) == (
        "cache/requests/a/attempts/000001/response.body",
        "profile/reference.json",
        "routes/cheap/accepted.jsonl",
    )
    by_path = {artifact.path: artifact for artifact in manifest.artifacts}
    assert by_path["profile/reference.json"].record_count == 1
    assert by_path["routes/cheap/accepted.jsonl"].record_count == 2
    assert by_path[
        "cache/requests/a/attempts/000001/response.body"
    ].record_count == 0
    assert "manifest.json" not in by_path
    assert canonical_jsonl_bytes(({"a": 1}, {"b": 2})) == (
        b'{"a":1}\n{"b":2}\n'
    )


def test_rebuilt_tree_has_byte_identical_artifacts_and_manifest(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root in roots:
        _build_tree(root)

    relative_paths = tuple(
        path.relative_to(roots[0])
        for path in sorted(roots[0].rglob("*"))
        if path.is_file()
    )
    assert relative_paths
    assert all(
        (roots[0] / path).read_bytes() == (roots[1] / path).read_bytes()
        for path in relative_paths
    )


def test_builder_is_write_once_and_closes_after_finalization(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    builder = Phase1ArtifactBuilder(root)
    builder.write_json("result.json", {"ok": True})
    with pytest.raises(FileExistsError, match="already exists"):
        builder.write_json("result.json", {"ok": False})
    builder.finalize()
    with pytest.raises(Phase1ArtifactError, match="already finalized"):
        builder.write_bytes("late.body", b"late")
    with pytest.raises(Phase1ArtifactError, match="already finalized"):
        builder.finalize()


def test_promotion_is_atomic_and_refuses_existing_destination(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _build_tree(staging)
    destination = tmp_path / "data" / "tinyworlds-v2" / "reference"

    assert promote_phase1_artifact_tree(staging, destination) == destination
    assert not staging.exists()
    load_phase1_artifact_tree(destination)

    second_staging = tmp_path / "second-staging"
    _build_tree(second_staging)
    with pytest.raises(FileExistsError, match="already exists"):
        promote_phase1_artifact_tree(second_staging, destination)
    assert second_staging.is_dir()


def test_promotion_never_replaces_destination_created_during_rename_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "race-staging"
    _build_tree(staging)
    destination = tmp_path / "published" / "reference"
    original_rename = artifact_module._rename_directory_noreplace
    raced_inode: list[int] = []

    def create_empty_destination_then_rename(source: Path, target: Path) -> None:
        target.mkdir()
        raced_inode.append(target.stat().st_ino)
        original_rename(source, target)

    monkeypatch.setattr(
        artifact_module,
        "_rename_directory_noreplace",
        create_empty_destination_then_rename,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        promote_phase1_artifact_tree(staging, destination)

    assert destination.is_dir()
    assert destination.stat().st_ino == raced_inode[0]
    assert tuple(destination.iterdir()) == ()
    assert staging.is_dir()
    load_phase1_artifact_tree(staging)


def test_builder_can_promote_before_or_after_explicit_finalization(
    tmp_path: Path,
) -> None:
    for finalize_first in (False, True):
        root = tmp_path / f"staging-{finalize_first}"
        root.mkdir()
        builder = Phase1ArtifactBuilder(root)
        builder.write_json("result.json", {"finalize_first": finalize_first})
        if finalize_first:
            builder.finalize()
        destination = tmp_path / f"reference-{finalize_first}"
        assert builder.promote(destination) == destination
        load_phase1_artifact_tree(destination)

    guarded_root = tmp_path / "staging-guarded"
    guarded_root.mkdir()
    guarded = Phase1ArtifactBuilder(guarded_root)
    guarded.write_json("result.json", {"guarded": True})
    manifest = guarded.finalize()
    with pytest.raises(Phase1ArtifactError, match="semantic validation"):
        guarded.promote(
            tmp_path / "reference-guarded",
            expected_manifest_sha256="0" * 64,
        )
    assert guarded_root.is_dir()
    assert guarded.promote(
        tmp_path / "reference-guarded",
        expected_manifest_sha256=manifest.manifest_sha256,
    ) == tmp_path / "reference-guarded"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("sha256", "0" * 64, "digest mismatch"),
        ("size_bytes", 999_999, "size mismatch"),
        ("record_count", 999_999, "record count mismatch"),
    ),
)
def test_loader_rejects_manifested_integrity_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    root = tmp_path / field
    _build_tree(root)

    def mutate(record: dict[str, object]) -> None:
        _artifact_record(record, "routes/cheap/accepted.jsonl")[field] = replacement

    _rewrite_manifest(root, mutate)
    with pytest.raises(Phase1ArtifactError, match=message):
        load_phase1_artifact_tree(root)


@pytest.mark.parametrize(
    ("path", "payload", "message"),
    (
        ("profile/reference.json", b'{ "documents": 20000 }', "non-canonical"),
        (
            "routes/cheap/accepted.jsonl",
            b'{"story_id":"s2"}\n{"story_id":"s1"}',
            "JSONL framing",
        ),
    ),
)
def test_loader_rejects_noncanonical_derived_artifacts_even_when_rehashed(
    tmp_path: Path,
    path: str,
    payload: bytes,
    message: str,
) -> None:
    root = tmp_path / "reference"
    _build_tree(root)
    (root / path).write_bytes(payload)

    def mutate(record: dict[str, object]) -> None:
        artifact = _artifact_record(record, path)
        artifact["sha256"] = sha256(payload).hexdigest()
        artifact["size_bytes"] = len(payload)
        artifact["record_count"] = 1 if path.endswith(".json") else 2

    _rewrite_manifest(root, mutate)
    with pytest.raises(Phase1ArtifactError, match=message):
        load_phase1_artifact_tree(root)


def test_loader_rejects_unknown_missing_and_symlink_entries(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown"
    _build_tree(unknown)
    (unknown / "surprise.body").write_bytes(b"not manifested")
    with pytest.raises(Phase1ArtifactError, match="unlisted files"):
        load_phase1_artifact_tree(unknown)

    missing = tmp_path / "missing"
    _build_tree(missing)
    (missing / "profile" / "reference.json").unlink()
    with pytest.raises(Phase1ArtifactError, match="missing or unlisted files"):
        load_phase1_artifact_tree(missing)

    linked = tmp_path / "linked"
    _build_tree(linked)
    target = linked / "cache" / "requests" / "a" / "attempts" / "000001"
    body = target / "response.body"
    body.unlink()
    os.symlink(linked / "profile" / "reference.json", body)
    with pytest.raises(Phase1ArtifactError, match="symlink"):
        load_phase1_artifact_tree(linked)


def test_loader_authenticates_opaque_raw_response_bytes(tmp_path: Path) -> None:
    root = tmp_path / "raw-tamper"
    _build_tree(root)
    response = root / "cache/requests/a/attempts/000001/response.body"
    response.write_bytes(response.read_bytes() + b"tampered")
    with pytest.raises(Phase1ArtifactError, match="size mismatch"):
        load_phase1_artifact_tree(root)


@pytest.mark.parametrize("malicious_path", ("../escape.body", "/escape.body", "a\\b"))
def test_loader_rejects_manifest_path_traversal(
    tmp_path: Path, malicious_path: str
) -> None:
    root = tmp_path / "reference"
    _build_tree(root)

    def mutate(record: dict[str, object]) -> None:
        _artifact_record(record, "profile/reference.json")["path"] = malicious_path
        artifacts = record["artifacts"]
        assert isinstance(artifacts, list)
        artifacts.sort(
            key=lambda artifact: artifact["path"] if isinstance(artifact, dict) else ""
        )

    _rewrite_manifest(root, mutate)
    with pytest.raises(Phase1ArtifactError, match="unsafe|canonical relative"):
        load_phase1_artifact_tree(root)


def test_loader_rejects_noncanonical_and_self_digest_manifest_changes(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical"
    _build_tree(noncanonical)
    record = json.loads((noncanonical / "manifest.json").read_bytes())
    (noncanonical / "manifest.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    with pytest.raises(Phase1ArtifactError, match="non-canonical"):
        load_phase1_artifact_tree(noncanonical)

    wrong_digest = tmp_path / "wrong-digest"
    _build_tree(wrong_digest)
    record = json.loads((wrong_digest / "manifest.json").read_bytes())
    record["version"] = "tampered"
    (wrong_digest / "manifest.json").write_bytes(canonical_json_bytes(record))
    with pytest.raises(Phase1ArtifactError, match="self-digest"):
        load_phase1_artifact_tree(wrong_digest)


def test_manifest_rejects_unknown_fields_and_empty_directories(tmp_path: Path) -> None:
    unknown_field = tmp_path / "unknown-field"
    _build_tree(unknown_field)
    record = json.loads((unknown_field / "manifest.json").read_bytes())
    record["extra"] = True
    (unknown_field / "manifest.json").write_bytes(canonical_json_bytes(record))
    with pytest.raises(Phase1ArtifactError, match="unknown=.*extra"):
        load_phase1_artifact_tree(unknown_field)

    empty_directory = tmp_path / "empty-directory"
    _build_tree(empty_directory)
    (empty_directory / "unlisted-empty").mkdir()
    with pytest.raises(Phase1ArtifactError, match="unlisted directories"):
        load_phase1_artifact_tree(empty_directory)
