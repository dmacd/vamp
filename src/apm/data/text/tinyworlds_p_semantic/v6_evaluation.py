"""Per-group semantic-v6 validation and one-shot sealed evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import jax

from apm.data.text.tinyworlds_p_semantic.evaluation import (
    SplitGroupEvaluation,
    _paired_losses,
    _evaluate_epoch_validation_core,
    _evaluate_partition_split_core,
    _evaluate_sealed_test_once_core,
    _count_evaluation_batches,
    load_group_losses,
    semantic_validation_record,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    WORLD_LABELS,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.statistics import (
    CANONICAL_REPLICATES,
    EmpiricalGap,
    SemanticEpochValidation,
    WorldEmpiricalGap,
    summarize_empirical_gaps,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_SEMANTIC_TRAINING_PRESET,
    V6SemanticPartitionArtifact,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.parameters import GptNeoParams


V6_VALIDATION_TREE_FORMAT = "tinyworlds-p-semantic-v6-validation-tree"
V6_SEALED_TEST_TREE_FORMAT = "tinyworlds-p-semantic-v6-sealed-test-tree"
V6EvaluationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class V6SemanticSealedTest:
    """The single semantic-v6 test evaluation for one frozen checkpoint."""

    selected_epoch: int
    partition_sha256: str
    evaluation_identity_sha256: str
    held_in: SplitGroupEvaluation
    validation: SemanticEpochValidation
    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not 2 <= self.selected_epoch <= 5:
            raise ValueError("sealed semantic-v6 epoch must lie in 2-5")
        _require_sha256(self.partition_sha256, "semantic-v6 sealed partition")
        _require_sha256(
            self.evaluation_identity_sha256,
            "semantic-v6 sealed evaluation",
        )
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)


def evaluate_v6_partition_split(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    split: str,
    ledger_path: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    progress: V6EvaluationProgress | None = None,
) -> SplitGroupEvaluation:
    """Persist exact per-group losses for one authenticated v6 split."""
    _require_v6_partition(artifact)
    return _evaluate_partition_split_core(
        params,
        artifact,
        split,
        ledger_path,
        model_config or V6_SEMANTIC_TRAINING_PRESET.model_config,
        progress=progress,
    )


def evaluate_v6_forced_lora_split(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    split: str,
    ledger_path: str | Path,
    lora_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    edge_coefficients: jax.Array,
    model_config: GptNeoConfig | None = None,
    *,
    progress: V6EvaluationProgress | None = None,
) -> SplitGroupEvaluation:
    """Persist per-group losses while forcing one previously trained adapter path."""
    _require_v6_partition(artifact)
    return _evaluate_partition_split_core(
        params,
        artifact,
        split,
        ledger_path,
        model_config or V6_SEMANTIC_TRAINING_PRESET.model_config,
        progress=progress,
        lora_memory=lora_memory,
        lora_config=lora_config,
        edge_coefficients=edge_coefficients,
    )


def count_v6_evaluation_batches(
    artifact: V6SemanticPartitionArtifact,
    split: str,
) -> int:
    """Count the exact grouped batches used by validation or sealed evaluation."""
    _require_v6_partition(artifact)
    return _count_evaluation_batches(artifact, split)


def evaluate_v6_epoch_validation(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    epoch: int,
    output_directory: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    replicates: int = CANONICAL_REPLICATES,
    progress: V6EvaluationProgress | None = None,
) -> SemanticEpochValidation:
    """Evaluate all v6 validation worlds and their paired comparisons."""
    _require_v6_partition(artifact)
    return _evaluate_epoch_validation_core(
        params,
        artifact,
        epoch,
        output_directory,
        model_config or V6_SEMANTIC_TRAINING_PRESET.model_config,
        tree_format=V6_VALIDATION_TREE_FORMAT,
        replicates=replicates,
        progress=progress,
    )


def evaluate_v6_sealed_test_once(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    selected_epoch: int,
    output_directory: str | Path,
    model_config: GptNeoConfig | None = None,
    *,
    replicates: int = CANONICAL_REPLICATES,
    progress: V6EvaluationProgress | None = None,
) -> V6SemanticSealedTest:
    """Evaluate the v6 test split once after checkpoint selection is frozen."""
    _require_v6_partition(artifact)
    result = _evaluate_sealed_test_once_core(
        params,
        artifact,
        selected_epoch,
        output_directory,
        model_config or V6_SEMANTIC_TRAINING_PRESET.model_config,
        tree_format=V6_SEALED_TEST_TREE_FORMAT,
        replicates=replicates,
        progress=progress,
    )
    return V6SemanticSealedTest(
        selected_epoch=result.selected_epoch,
        partition_sha256=artifact.partition_sha256,
        evaluation_identity_sha256=_text(
            _load_json(Path(result.directory) / "sealed-test.json"),
            "evaluation_identity_sha256",
        ),
        held_in=result.held_in,
        validation=result.validation,
        directory=result.directory,
    )


def load_v6_epoch_validation(
    directory: str | Path,
    artifact: V6SemanticPartitionArtifact,
    epoch: int,
) -> SemanticEpochValidation:
    """Recompute and authenticate one completed semantic-v6 validation epoch."""
    _require_v6_partition(artifact)
    root = Path(directory)
    tree = _authenticated_evaluation_tree(
        root,
        V6_VALIDATION_TREE_FORMAT,
        "validation",
    )
    record = _load_json(root / "validation.json")
    if set(record) != {
        "evaluation_identity_sha256",
        "held_in",
        "partition_sha256",
        "validation",
    } or record.get("partition_sha256") != artifact.partition_sha256:
        raise ValueError("semantic-v6 validation source binding changed")
    identity = _evaluation_identity(root, artifact, epoch, "validation")
    if (
        record.get("evaluation_identity_sha256") != identity
        or tree.get("evaluation_identity_sha256") != identity
    ):
        raise ValueError("semantic-v6 validation identity changed")
    held = _held_evaluation(root, _mapping(record, "held_in"), "base/validation")
    return _recomputed_validation(
        root,
        _mapping(record, "validation"),
        artifact,
        epoch,
        "validation",
        identity,
        held,
    )


def load_v6_sealed_test(
    directory: str | Path,
    artifact: V6SemanticPartitionArtifact,
    selected_epoch: int,
) -> V6SemanticSealedTest:
    """Recompute and authenticate a completed semantic-v6 sealed evaluation."""
    _require_v6_partition(artifact)
    root = Path(directory)
    tree = _authenticated_evaluation_tree(
        root,
        V6_SEALED_TEST_TREE_FORMAT,
        "sealed-test",
    )
    record = _load_json(root / "sealed-test.json")
    if set(record) != {
        "evaluation_identity_sha256",
        "held_in",
        "partition_sha256",
        "selected_epoch",
        "test",
    }:
        raise ValueError("semantic-v6 sealed-test record fields changed")
    if _load_json(root / "sealed-open.json") != {
        "partition_sha256": artifact.partition_sha256,
        "selected_epoch": selected_epoch,
    } or (
        record.get("partition_sha256") != artifact.partition_sha256
        or record.get("selected_epoch") != selected_epoch
    ):
        raise ValueError("semantic-v6 sealed-test source binding changed")
    identity = _evaluation_identity(root, artifact, selected_epoch, "test")
    if (
        record.get("evaluation_identity_sha256") != identity
        or tree.get("evaluation_identity_sha256") != identity
    ):
        raise ValueError("semantic-v6 sealed-test identity changed")
    held = _held_evaluation(root, _mapping(record, "held_in"), "base/test")
    validation = _recomputed_validation(
        root,
        _mapping(record, "test"),
        artifact,
        selected_epoch,
        "test",
        identity,
        held,
    )
    return V6SemanticSealedTest(
        selected_epoch,
        artifact.partition_sha256,
        identity,
        held,
        validation,
        root.resolve(),
    )


def _authenticated_evaluation_tree(
    root: Path,
    format_name: str,
    label: str,
) -> dict[str, object]:
    tree = _load_json(root / "tree.json")
    if (
        set(tree)
        != {"evaluation_identity_sha256", "files", "format", "schema_version"}
        or tree.get("format") != format_name
        or tree.get("schema_version") != 1
    ):
        raise ValueError(f"semantic-v6 {label} tree changed")
    descriptors = tree.get("files")
    if type(descriptors) is not list or any(
        type(item) is not dict or set(item) != {"name", "sha256", "size_bytes"}
        for item in descriptors
    ):
        raise ValueError(f"semantic-v6 {label} descriptors changed")
    expected_names = tuple(
        path.name
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "tree.json"
    )
    if tuple(_text(item, "name") for item in descriptors) != expected_names:
        raise ValueError(f"semantic-v6 {label} file set changed")
    for descriptor in descriptors:
        path = root / _text(descriptor, "name")
        if (
            path.is_symlink()
            or path.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(path) != _text(descriptor, "sha256")
        ):
            raise ValueError(f"semantic-v6 {label} file changed: {path}")
    return tree


def _evaluation_identity(
    root: Path,
    artifact: V6SemanticPartitionArtifact,
    epoch: int,
    split: str,
) -> str:
    record: dict[str, object] = {
        "ledgers": {
            f"{role}/{world}": _file_sha256(
                root / f"{role}-{world}-{split}.groups.jsonl"
            )
            for world in WORLD_LABELS
            for role in ("world", "control")
        },
        "partition_sha256": artifact.partition_sha256,
        "split": split,
    }
    record["selected_epoch" if split == "test" else "epoch"] = epoch
    return record_sha256(record)


def _held_evaluation(
    root: Path,
    record: Mapping[str, object],
    split: str,
) -> SplitGroupEvaluation:
    if set(record) != {
        "active_tokens",
        "group_count",
        "ledger",
        "ledger_sha256",
        "loss_sum",
        "nll",
        "split",
    } or record.get("split") != split:
        raise ValueError("semantic-v6 held-in summary changed")
    held = SplitGroupEvaluation(
        split=_text(record, "split"),
        active_tokens=_integer(record, "active_tokens"),
        loss_sum=_number(record, "loss_sum"),
        ledger_path=root / _text(record, "ledger"),
        ledger_sha256=_text(record, "ledger_sha256"),
        group_count=_integer(record, "group_count"),
    )
    held_losses = load_group_losses(held.ledger_path)
    if (
        _file_sha256(held.ledger_path) != held.ledger_sha256
        or held.group_count != len(held_losses)
        or held.active_tokens != sum(item.active_tokens for item in held_losses)
        or not math.isclose(
            held.loss_sum,
            math.fsum(item.loss_sum for item in held_losses),
            rel_tol=1e-12,
            abs_tol=1e-6,
        )
        or not math.isclose(
            _number(record, "nll"),
            held.nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("semantic-v6 held-in aggregates changed")
    return held


def _recomputed_validation(
    root: Path,
    record: Mapping[str, object],
    artifact: V6SemanticPartitionArtifact,
    epoch: int,
    split: str,
    identity: str,
    held: SplitGroupEvaluation,
) -> SemanticEpochValidation:
    persisted = v6_validation_from_record(record)
    pairs_by_world = tuple(
        _paired_losses(
            artifact,
            world,
            split,
            root / f"world-{world}-{split}.groups.jsonl",
            root / f"control-{world}-{split}.groups.jsonl",
        )
        for world in WORLD_LABELS
    )
    worlds, mean = summarize_empirical_gaps(
        pairs_by_world,
        identity,
        replicates=CANONICAL_REPLICATES,
    )
    validation = SemanticEpochValidation(
        epoch=epoch,
        held_in_nll=held.nll,
        worlds=worlds,
        mean_empirical=mean,
        allocator_peak_bytes=persisted.allocator_peak_bytes,
    )
    if semantic_validation_record(validation) != dict(record):
        raise ValueError(f"semantic-v6 {split} statistics changed")
    return validation


def _require_v6_partition(artifact: V6SemanticPartitionArtifact) -> None:
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 evaluation requires its strict partition")


def v6_validation_from_record(
    record: Mapping[str, object],
) -> SemanticEpochValidation:
    """Strictly reconstruct one persisted semantic validation record."""
    worlds = record.get("worlds")
    if type(worlds) is not list or any(type(item) is not dict for item in worlds):
        raise ValueError("semantic-v6 sealed-test worlds changed")
    return SemanticEpochValidation(
        epoch=_integer(record, "epoch"),
        held_in_nll=_number(record, "held_in_nll"),
        worlds=tuple(
            WorldEmpiricalGap(
                world=_text(item, "world"),
                world_nll=_number(item, "world_nll"),
                control_nll=_number(item, "control_nll"),
                empirical=_empirical_from_record(_mapping(item, "empirical")),
            )
            for item in worlds
        ),
        mean_empirical=_empirical_from_record(_mapping(record, "mean")),
        allocator_peak_bytes=_integer(record, "allocator_peak_bytes"),
    )


def _empirical_from_record(record: Mapping[str, object]) -> EmpiricalGap:
    return EmpiricalGap(
        observed_gap=_number(record, "observed_gap"),
        bootstrap_lower=_number(record, "bootstrap_lower"),
        bootstrap_upper=_number(record, "bootstrap_upper"),
        placebo_probability=_number(record, "placebo_probability"),
        replicate_count=_integer(record, "replicate_count"),
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic-v6 evaluation JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical semantic-v6 evaluation JSON: {path}")
    return value


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"semantic-v6 evaluation field {field!r} must be an object")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"semantic-v6 evaluation field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 evaluation field {field!r} must be nonnegative")
    return value


def _number(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float):
        raise ValueError(f"semantic-v6 evaluation field {field!r} must be numeric")
    return float(value)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SplitGroupEvaluation",
    "V6EvaluationProgress",
    "V6_SEALED_TEST_TREE_FORMAT",
    "V6_VALIDATION_TREE_FORMAT",
    "V6SemanticSealedTest",
    "evaluate_v6_epoch_validation",
    "evaluate_v6_forced_lora_split",
    "evaluate_v6_partition_split",
    "evaluate_v6_sealed_test_once",
    "count_v6_evaluation_batches",
    "load_v6_epoch_validation",
    "load_v6_sealed_test",
    "v6_validation_from_record",
    "load_group_losses",
    "semantic_validation_record",
]
