"""Strict publication boundary for a fresh accepted TinyWorlds-Q base."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.execution import BaseQualityDecision
from apm.data.text.tinyworlds_q_semantic.training import (
    QueryBaseTrainingConfig,
    QueryBaseTrainingResult,
    QuerySplitNll,
    query_base_training_identity,
)
from apm.lm.checkpoint import (
    BaseCheckpointRef,
    CheckpointFileHash,
    LoadedGptNeoCheckpoint,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)


QUERY_SELECTED_BASE_FORMAT = "tinyworlds-q-semantic-selected-base-v1"
QUERY_SELECTED_BASE_TREE_FORMAT = "tinyworlds-q-semantic-selected-base-tree-v1"


@dataclass(frozen=True, slots=True)
class QueryBaseEpochEvidence:
    """Held-in validation evidence for one completed scratch-base epoch."""

    epoch: int
    validation: QuerySplitNll

    def __post_init__(self) -> None:
        if self.epoch not in (1, 2) or self.validation.split != "validation":
            raise ValueError("query base evidence requires validation for epoch one or two")

    def as_record(self) -> dict[str, object]:
        """Return the canonical epoch-evidence row."""
        return {
            "active_tokens": self.validation.active_tokens,
            "epoch": self.epoch,
            "nll": self.validation.nll,
            "split": self.validation.split,
        }


@dataclass(frozen=True, slots=True)
class QuerySelectedBase:
    """One accepted fresh base bound to the query catalog and partition."""

    directory: Path
    selection_sha256: str
    catalog_sha256: str
    partition_sha256: str
    base_config_sha256: str
    training_sha256: str
    epoch_evidence: tuple[QueryBaseEpochEvidence, QueryBaseEpochEvidence]
    allocator_peak_bytes: int
    checkpoint: BaseCheckpointRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        for value, label in (
            (self.selection_sha256, "selected base"),
            (self.catalog_sha256, "selected base catalog"),
            (self.partition_sha256, "selected base partition"),
            (self.base_config_sha256, "selected base config"),
            (self.training_sha256, "selected base training"),
        ):
            require_sha256(value, label)
        if tuple(item.epoch for item in self.epoch_evidence) != (1, 2):
            raise ValueError("selected base evidence must contain epochs one and two")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("selected base allocator peak must be nonnegative")


def publish_query_selected_base(
    result: QueryBaseTrainingResult,
    epoch_evidence: tuple[QueryBaseEpochEvidence, QueryBaseEpochEvidence],
    allocator_peak_bytes: int,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    output_root: str | Path,
) -> QuerySelectedBase:
    """Accept and atomically publish only a complete passing fresh base run."""
    if type(result) is not QueryBaseTrainingResult:
        raise TypeError("selected query base requires a query-native training result")
    if (
        type(epoch_evidence) is not tuple
        or len(epoch_evidence) != 2
        or any(type(item) is not QueryBaseEpochEvidence for item in epoch_evidence)
        or tuple(item.epoch for item in epoch_evidence) != (1, 2)
    ):
        raise ValueError("selected query base requires ordered two-epoch evidence")
    config = QueryBaseTrainingConfig.from_preset(preset)
    base_config_sha256 = record_sha256(config.as_record())
    expected_training_sha256, planned_updates = query_base_training_identity(
        artifact,
        preset,
        config,
    )
    if (
        result.training_sha256 != expected_training_sha256
        or result.planned_optimizer_updates != planned_updates
        or result.cursor.epoch != preset.base_epochs
        or result.cursor.optimizer_update != planned_updates
        or int(result.state.step) != planned_updates
    ):
        raise ValueError("selected query base is not the complete registered run")
    decision = BaseQualityDecision(
        tuple(item.validation.nll for item in epoch_evidence),  # type: ignore[arg-type]
        allocator_peak_bytes,
        preset.allocator_peak_limit_bytes,
    )
    if not decision.passed:
        raise RuntimeError(f"query base quality gate failed: {decision.reason}")

    publication_root = Path(output_root) / "base"
    publication_root.mkdir(parents=True, exist_ok=True)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-query-base-", dir=work_root))
    try:
        checkpoint = save_gpt_neo_checkpoint(
            staging / "checkpoint",
            result.state.trainable,
            config.model_config,
            tokenizer=_checkpoint_tokenizer(artifact),
            source=SourceCheckpointMetadata(
                identifier=(
                    f"{artifact.archive_identity.dataset_id}/"
                    f"{artifact.archive_identity.filename}"
                ),
                revision=artifact.archive_identity.revision,
                sha256=artifact.archive_identity.sha256,
            ),
        )
        shutil.copyfile(result.trace_path, staging / "training-progress.jsonl")
        core = {
            "allocator_peak_bytes": allocator_peak_bytes,
            "archive_identity": artifact.archive_identity.as_record(),
            "catalog_sha256": artifact.catalog_sha256,
            "checkpoint": {
                "manifest_sha256": checkpoint.manifest_sha256,
                "parameter_checksum": checkpoint.parameter_checksum,
            },
            "base_training_config": config.as_record(),
            "base_training_config_sha256": base_config_sha256,
            "epoch_evidence": [item.as_record() for item in epoch_evidence],
            "format": QUERY_SELECTED_BASE_FORMAT,
            "partition_sha256": artifact.partition_sha256,
            "quality_gate": {"passed": decision.passed, "reason": decision.reason},
            "sealed_test_opened": False,
            "tokenizer_identity": artifact.tokenizer_identity.as_record(),
            "training_progress_sha256": _file_sha256(
                staging / "training-progress.jsonl"
            ),
            "training_sha256": result.training_sha256,
        }
        selection_sha256 = record_sha256(core)
        _write_file(
            staging / "manifest.json",
            canonical_json_bytes({**core, "selection_sha256": selection_sha256}),
        )
        _write_file(
            staging / "training-report.md",
            (
                "# TinyWorlds-Q selected fresh base\n\n"
                f"Selection: `{selection_sha256}`\n\n"
                f"Epoch-one held-in NLL: {epoch_evidence[0].validation.nll:.9f}\n\n"
                f"Epoch-two held-in NLL: {epoch_evidence[1].validation.nll:.9f}\n\n"
                f"Allocator peak: {allocator_peak_bytes} bytes\n\n"
                "The operational base gate passed. No sealed query was opened.\n"
            ).encode("utf-8"),
        )
        _write_tree(staging, selection_sha256)
        target = publication_root / selection_sha256
        if target.exists():
            existing = load_query_selected_base(target, artifact, preset)
            _remove_tree(staging)
            return existing
        os.replace(staging, target)
        return load_query_selected_base(target, artifact, preset)
    except BaseException:
        _remove_tree(staging)
        raise


def load_query_selected_base(
    directory: str | Path,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
) -> QuerySelectedBase:
    """Strictly authenticate the q-native base, checkpoint, and source bindings."""
    root = Path(directory)
    tree = _canonical_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "schema_version", "selection_sha256"}
        or tree.get("format") != QUERY_SELECTED_BASE_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("selection_sha256") != root.name
    ):
        raise ValueError("query selected-base tree identity changed")
    _verify_tree_files(root, tree)
    manifest = _canonical_json(root / "manifest.json")
    required = {
        "allocator_peak_bytes",
        "archive_identity",
        "base_training_config",
        "base_training_config_sha256",
        "catalog_sha256",
        "checkpoint",
        "epoch_evidence",
        "format",
        "partition_sha256",
        "quality_gate",
        "sealed_test_opened",
        "selection_sha256",
        "tokenizer_identity",
        "training_progress_sha256",
        "training_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != QUERY_SELECTED_BASE_FORMAT
        or manifest.get("selection_sha256") != root.name
        or manifest.get("sealed_test_opened") is not False
    ):
        raise ValueError("query selected-base manifest changed")
    core = {key: value for key, value in manifest.items() if key != "selection_sha256"}
    if record_sha256(core) != root.name:
        raise ValueError("query selected-base content identity changed")
    base_config = QueryBaseTrainingConfig.from_preset(preset)
    base_config_sha256 = record_sha256(base_config.as_record())
    if (
        manifest.get("catalog_sha256") != artifact.catalog_sha256
        or manifest.get("partition_sha256") != artifact.partition_sha256
        or manifest.get("archive_identity") != artifact.archive_identity.as_record()
        or manifest.get("tokenizer_identity") != artifact.tokenizer_identity.as_record()
        or manifest.get("base_training_config") != base_config.as_record()
        or manifest.get("base_training_config_sha256") != base_config_sha256
        or artifact.concept_ids[: preset.active_world_count] != preset.concept_ids
    ):
        raise ValueError("query selected-base partition or experiment binding changed")
    if _file_sha256(root / "training-progress.jsonl") != manifest.get(
        "training_progress_sha256"
    ):
        raise ValueError("query selected-base training trace changed")
    evidence_raw = manifest.get("epoch_evidence")
    if type(evidence_raw) is not list or len(evidence_raw) != 2 or any(
        type(item) is not dict for item in evidence_raw
    ):
        raise ValueError("query selected-base epoch evidence changed")
    evidence = tuple(_decode_epoch(item) for item in evidence_raw)
    allocator_peak = _integer(manifest, "allocator_peak_bytes")
    decision = BaseQualityDecision(
        tuple(item.validation.nll for item in evidence),  # type: ignore[arg-type]
        allocator_peak,
        preset.allocator_peak_limit_bytes,
    )
    if not decision.passed or manifest.get("quality_gate") != {
        "passed": True,
        "reason": "passed",
    }:
        raise ValueError("query selected-base quality gate changed")
    loaded = load_gpt_neo_checkpoint(root / "checkpoint")
    checkpoint_record = manifest.get("checkpoint")
    if (
        type(checkpoint_record) is not dict
        or checkpoint_record
        != {
            "manifest_sha256": loaded.reference.manifest_sha256,
            "parameter_checksum": loaded.reference.parameter_checksum,
        }
        or loaded.config != preset.model_config
        or not _checkpoint_sources_match(loaded, artifact)
    ):
        raise ValueError("query selected-base checkpoint changed")
    return QuerySelectedBase(
        directory=root.resolve(),
        selection_sha256=root.name,
        catalog_sha256=artifact.catalog_sha256,
        partition_sha256=artifact.partition_sha256,
        base_config_sha256=base_config_sha256,
        training_sha256=_text(manifest, "training_sha256"),
        epoch_evidence=evidence,  # type: ignore[arg-type]
        allocator_peak_bytes=allocator_peak,
        checkpoint=loaded.reference,
    )


def _checkpoint_tokenizer(
    artifact: QueryPartitionArtifact,
) -> TokenizerCheckpointMetadata:
    identity = artifact.tokenizer_identity
    return TokenizerCheckpointMetadata(
        kind=identity.kind,
        identifier=identity.identifier,
        revision=identity.revision,
        files=tuple(
            CheckpointFileHash(item.name, item.sha256) for item in identity.files
        ),
    )


def _checkpoint_sources_match(
    loaded: LoadedGptNeoCheckpoint,
    artifact: QueryPartitionArtifact,
) -> bool:
    tokenizer = artifact.tokenizer_identity
    archive = artifact.archive_identity
    return (
        loaded.tokenizer.kind == tokenizer.kind
        and loaded.tokenizer.identifier == tokenizer.identifier
        and loaded.tokenizer.revision == tokenizer.revision
        and tuple((item.name, item.sha256) for item in loaded.tokenizer.files)
        == tuple((item.name, item.sha256) for item in tokenizer.files)
        and loaded.source.identifier == f"{archive.dataset_id}/{archive.filename}"
        and loaded.source.revision == archive.revision
        and loaded.source.sha256 == archive.sha256
    )


def _decode_epoch(record: dict[str, object]) -> QueryBaseEpochEvidence:
    if set(record) != {"active_tokens", "epoch", "nll", "split"}:
        raise ValueError("query selected-base epoch fields changed")
    nll = record.get("nll")
    if type(nll) not in (int, float):
        raise ValueError("query selected-base epoch NLL changed")
    return QueryBaseEpochEvidence(
        epoch=_integer(record, "epoch"),
        validation=QuerySplitNll(
            split=_text(record, "split"),
            active_tokens=_integer(record, "active_tokens"),
            nll=float(nll),
        ),
    )


def _write_tree(root: Path, selection_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    _write_file(
        root / "tree.json",
        canonical_json_bytes(
            {
                "files": list(files),
                "format": QUERY_SELECTED_BASE_TREE_FORMAT,
                "schema_version": 1,
                "selection_sha256": selection_sha256,
            }
        ),
    )


def _verify_tree_files(root: Path, tree: dict[str, object]) -> None:
    files = tree.get("files")
    if type(files) is not list or any(type(item) is not dict for item in files):
        raise ValueError("query selected-base tree descriptors changed")
    described = tuple(_text(item, "relative_path") for item in files)
    actual = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    expected_directories = {
        parent.as_posix()
        for relative_path in (*described, "tree.json")
        for parent in Path(relative_path).parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if (
        described != actual
        or expected_directories != actual_directories
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError("query selected-base tree entries changed")
    for item in files:
        path = root / _text(item, "relative_path")
        if (
            path.stat().st_size != _integer(item, "size_bytes")
            or _file_sha256(path) != _text(item, "sha256")
        ):
            raise ValueError(f"query selected-base file changed: {path}")


def _canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid query selected-base JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical query selected-base JSON: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"query selected-base {field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"query selected-base {field} must be nonnegative")
    return value


__all__ = [
    "QUERY_SELECTED_BASE_FORMAT",
    "QUERY_SELECTED_BASE_TREE_FORMAT",
    "QueryBaseEpochEvidence",
    "QuerySelectedBase",
    "load_query_selected_base",
    "publish_query_selected_base",
]
