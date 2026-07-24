"""Disposable GPU timing and memory preflight for semantic-v6 training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from apm.data.text.tinyworlds_p import training as archive_training
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.partition_runtime import semantic_runtime_view
from apm.data.text.tinyworlds_p_semantic.v6_batching import (
    count_v6_partition_microbatches,
    iter_v6_partition_batches,
)
from apm.data.text.tinyworlds_p_semantic.v6_evaluation import (
    count_v6_evaluation_batches,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6SemanticPartitionArtifact,
)
from apm.data.text.tinyworlds_p_semantic.v6_training import (
    TrainingCursor,
    V6StreamingTrainingConfig,
)
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch


V6_PREFLIGHT_FORMAT = "tinyworlds-p-semantic-v6-gpu-preflight"
V6_PREFLIGHT_TREE_FORMAT = "tinyworlds-p-semantic-v6-gpu-preflight-tree"
V6_PREFLIGHT_IDENTITY_NAMESPACE = f"{V6_BENCHMARK_ID}-gpu-preflight-v1"
V6_PREFLIGHT_RESUME_FORMAT = "tinyworlds-p-semantic-v6-preflight-resume"
V6PreflightProgress = Callable[[TrainingCursor, float, int], None]


@dataclass(frozen=True, slots=True)
class V6GpuPreflight:
    """Authenticated two-update timing, finite-loss, and allocator evidence."""

    directory: Path
    preflight_sha256: str
    partition_sha256: str
    training_config_sha256: str
    seconds_per_update: float
    seconds_per_evaluation_batch: float
    allocator_peak_bytes: int
    updates_per_epoch: int
    validation_batches_per_epoch: int
    estimated_calibration_seconds: float
    estimated_base_pass_path_seconds: float
    estimated_adapter_training_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        _require_sha256(self.preflight_sha256)
        _require_sha256(self.partition_sha256)
        _require_sha256(self.training_config_sha256)
        timings = (
            self.seconds_per_update,
            self.seconds_per_evaluation_batch,
            self.estimated_calibration_seconds,
            self.estimated_base_pass_path_seconds,
            self.estimated_adapter_training_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in timings):
            raise ValueError("semantic-v6 preflight timings must be finite and positive")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("semantic-v6 preflight allocator peak is invalid")
        if any(
            type(value) is not int or value <= 0
            for value in (self.updates_per_epoch, self.validation_batches_per_epoch)
        ):
            raise ValueError("semantic-v6 preflight work counts must be positive")


def run_and_publish_v6_gpu_preflight(
    artifact: V6SemanticPartitionArtifact,
    config: V6StreamingTrainingConfig,
    working_directory: str | Path,
    publication_root: str | Path,
    *,
    progress: V6PreflightProgress | None = None,
) -> V6GpuPreflight:
    """Run two isolated updates, time one warm evaluation batch, and publish evidence."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 preflight requires its strict partition")
    if type(config) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 preflight requires its strict config")
    working = Path(working_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError("semantic-v6 preflight working directory is not empty")
    callback_times: list[float] = []
    callback_losses: list[float] = []

    def record_update(cursor: TrainingCursor, nll: float, planned: int) -> None:
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError("semantic-v6 preflight update loss is not finite")
        callback_times.append(time.monotonic())
        callback_losses.append(nll)
        if progress is not None:
            progress(cursor, nll, planned)

    result = archive_training.run_streaming_base_training(
        semantic_runtime_view(artifact, V6SemanticPartitionArtifact),
        working / "disposable-training",
        config.archive_config,
        stop_after_update=2,
        progress=record_update,
        identity_namespace=V6_PREFLIGHT_IDENTITY_NAMESPACE,
        resume_format=V6_PREFLIGHT_RESUME_FORMAT,
    )
    if len(callback_times) != 2 or result.cursor.optimizer_update != 2:
        raise RuntimeError("semantic-v6 preflight did not complete exactly two updates")
    seconds_per_update = callback_times[1] - callback_times[0]
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0.0:
        raise RuntimeError("semantic-v6 preflight update timing is invalid")
    batch = next(iter_v6_partition_batches(artifact, "base/validation", epoch=0))
    seconds_per_evaluation_batch = _warm_evaluation_seconds(
        result.state.trainable,
        config,
        batch,
    )
    peak = allocator_peak_bytes()
    if peak > config.allocator_peak_limit_bytes:
        raise MemoryError(
            f"semantic-v6 preflight peak {peak:,} exceeds "
            f"{config.allocator_peak_limit_bytes:,} bytes"
        )
    return _publish_preflight(
        artifact,
        config,
        callback_losses,
        seconds_per_update,
        seconds_per_evaluation_batch,
        peak,
        Path(publication_root),
    )


def load_v6_gpu_preflight(directory: str | Path) -> V6GpuPreflight:
    """Authenticate a published semantic-v6 GPU preflight."""
    root = Path(directory)
    tree = _load_json(root / "tree.json")
    if (
        tree.get("format") != V6_PREFLIGHT_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("preflight_sha256") != root.name
    ):
        raise ValueError("semantic-v6 preflight tree changed")
    descriptors = tree.get("files")
    if type(descriptors) is not list or any(type(item) is not dict for item in descriptors):
        raise ValueError("semantic-v6 preflight descriptors changed")
    actual = tuple(
        path.name
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "tree.json"
    )
    if tuple(_text(item, "name") for item in descriptors) != actual:
        raise ValueError("semantic-v6 preflight file set changed")
    for descriptor in descriptors:
        path = root / _text(descriptor, "name")
        if (
            path.is_symlink()
            or path.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(path) != _text(descriptor, "sha256")
        ):
            raise ValueError(f"semantic-v6 preflight file changed: {path}")
    record = _load_json(root / "preflight.json")
    required = {
        "allocator_peak_bytes",
        "device_kind",
        "disposable_update_count",
        "format",
        "jax_version",
        "losses",
        "numpy_version",
        "partition_sha256",
        "platform",
        "preflight_sha256",
        "runtime_estimates_seconds",
        "sealed_test_opened",
        "seconds_per_evaluation_batch",
        "seconds_per_update",
        "training_config",
        "training_config_sha256",
        "updates_per_epoch",
        "validation_batches_per_epoch",
    }
    if (
        set(record) != required
        or record.get("format") != V6_PREFLIGHT_FORMAT
        or record.get("preflight_sha256") != root.name
    ):
        raise ValueError("semantic-v6 preflight record changed")
    content = {key: value for key, value in record.items() if key != "preflight_sha256"}
    if record_sha256(content) != root.name:
        raise ValueError("semantic-v6 preflight identity changed")
    expected_config = V6StreamingTrainingConfig.from_preset().as_record()
    expected_config_sha256 = record_sha256(expected_config)
    losses = record.get("losses")
    if (
        record.get("training_config") != expected_config
        or record.get("training_config_sha256") != expected_config_sha256
        or record.get("sealed_test_opened") is not False
        or record.get("disposable_update_count") != 2
        or record.get("platform") != "gpu"
        or type(losses) is not list
        or len(losses) != 2
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in losses
        )
        or _integer(record, "allocator_peak_bytes")
        > V6StreamingTrainingConfig.from_preset().allocator_peak_limit_bytes
    ):
        raise ValueError("semantic-v6 preflight frozen evidence changed")
    estimates = _mapping(record, "runtime_estimates_seconds")
    return V6GpuPreflight(
        directory=root.resolve(),
        preflight_sha256=root.name,
        partition_sha256=_text(record, "partition_sha256"),
        training_config_sha256=_text(record, "training_config_sha256"),
        seconds_per_update=_number(record, "seconds_per_update"),
        seconds_per_evaluation_batch=_number(
            record,
            "seconds_per_evaluation_batch",
        ),
        allocator_peak_bytes=_integer(record, "allocator_peak_bytes"),
        updates_per_epoch=_integer(record, "updates_per_epoch"),
        validation_batches_per_epoch=_integer(
            record,
            "validation_batches_per_epoch",
        ),
        estimated_calibration_seconds=_number(estimates, "calibration"),
        estimated_base_pass_path_seconds=_number(estimates, "base_pass_path"),
        estimated_adapter_training_seconds=_number(estimates, "adapter_training_proxy"),
    )


def _warm_evaluation_seconds(
    params: GptNeoParams,
    config: V6StreamingTrainingConfig,
    batch: TokenBatch,
) -> float:
    def evaluate(value: TokenBatch):
        output = apply_gpt_neo(
            params,
            config.model_config,
            jnp.asarray(value.input_ids, dtype=jnp.int32),
            jnp.asarray(value.attention_mask, dtype=jnp.bool_),
        )
        mask = jnp.asarray(value.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            output.logits,
            jnp.asarray(value.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask) / jnp.sum(mask)

    compiled = jax.jit(evaluate)
    first = compiled(batch)
    first.block_until_ready()
    started = time.monotonic()
    warm = compiled(batch)
    warm.block_until_ready()
    elapsed = time.monotonic() - started
    if not math.isfinite(float(warm)) or elapsed <= 0.0:
        raise RuntimeError("semantic-v6 preflight evaluation was not finite")
    return elapsed


def _publish_preflight(
    artifact: V6SemanticPartitionArtifact,
    config: V6StreamingTrainingConfig,
    losses: list[float],
    seconds_per_update: float,
    seconds_per_evaluation_batch: float,
    peak: int,
    publication_root: Path,
) -> V6GpuPreflight:
    microbatches = count_v6_partition_microbatches(artifact, "base/train")
    updates_per_epoch = math.ceil(
        microbatches / config.accumulation_microbatches
    )
    validation_batches = sum(
        count_v6_evaluation_batches(artifact, selector)
        for selector in _evaluation_selectors("validation")
    )
    calibration_seconds = (
        2 * updates_per_epoch * seconds_per_update
        + 2 * validation_batches * seconds_per_evaluation_batch
    )
    base_pass_seconds = (
        config.epochs * updates_per_epoch * seconds_per_update
        + config.epochs * validation_batches * seconds_per_evaluation_batch
    )
    adapter_training_proxy = 3 * 5 * 2_000 * seconds_per_update
    training_config_sha256 = record_sha256(config.as_record())
    device = jax.devices()[0]
    content = {
        "allocator_peak_bytes": peak,
        "device_kind": device.device_kind,
        "disposable_update_count": 2,
        "format": V6_PREFLIGHT_FORMAT,
        "jax_version": jax.__version__,
        "losses": losses,
        "numpy_version": np.__version__,
        "partition_sha256": artifact.partition_sha256,
        "platform": device.platform,
        "runtime_estimates_seconds": {
            "adapter_training_proxy": adapter_training_proxy,
            "base_pass_path": base_pass_seconds,
            "calibration": calibration_seconds,
        },
        "sealed_test_opened": False,
        "seconds_per_evaluation_batch": seconds_per_evaluation_batch,
        "seconds_per_update": seconds_per_update,
        "training_config": config.as_record(),
        "training_config_sha256": training_config_sha256,
        "updates_per_epoch": updates_per_epoch,
        "validation_batches_per_epoch": validation_batches,
    }
    identity = record_sha256(content)
    publication_root.mkdir(parents=True, exist_ok=True)
    target = publication_root / identity
    if target.exists():
        return load_v6_gpu_preflight(target)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-v6-preflight-", dir=work_root))
    _write_json(staging / "preflight.json", {**content, "preflight_sha256": identity})
    _write_text(staging / "preflight.md", _preflight_report(identity, content))
    _write_tree(staging, identity)
    os.rename(staging, target)
    _fsync_directory(publication_root)
    return load_v6_gpu_preflight(target)


def _evaluation_selectors(split: str) -> tuple[str, ...]:
    return (
        f"base/{split}",
        *(
            f"{role}/{world}/{split}"
            for world in ("A", "B", "C", "D", "E")
            for role in ("world", "control")
        ),
    )


def _preflight_report(identity: str, content: Mapping[str, object]) -> str:
    estimates = content["runtime_estimates_seconds"]
    assert isinstance(estimates, dict)
    return (
        "# TinyWorlds-P Semantic-v6 GPU Preflight\n\n"
        f"Preflight SHA-256: `{identity}`\n\n"
        "Exactly two disposable optimizer updates completed with finite losses. "
        "No validation gap was computed, no parameters are reusable by the real "
        "run, and the sealed test remained closed.\n\n"
        f"Warm update time: {float(content['seconds_per_update']):.4f} seconds.\n\n"
        f"Warm evaluation-batch time: "
        f"{float(content['seconds_per_evaluation_batch']):.4f} seconds.\n\n"
        f"Allocator peak: {int(content['allocator_peak_bytes']) / 2**30:.3f} GiB.\n\n"
        f"Estimated two-epoch calibration: "
        f"{_duration(float(estimates['calibration']))}.\n\n"
        f"Estimated five-epoch base pass path: "
        f"{_duration(float(estimates['base_pass_path']))}.\n\n"
        "The adapter estimate uses the base-update time as an explicit proxy and "
        "will be refined after a base checkpoint exists.\n"
    )


def _duration(seconds: float) -> str:
    rounded = round(max(0.0, seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"


def _write_tree(root: Path, identity: str) -> None:
    files = tuple(
        {
            "name": path.name,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": V6_PREFLIGHT_TREE_FORMAT,
            "preflight_sha256": identity,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic-v6 preflight JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical semantic-v6 preflight JSON: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"semantic-v6 preflight field {field!r} must be an object")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"semantic-v6 preflight field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 preflight field {field!r} must be nonnegative")
    return value


def _number(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float):
        raise ValueError(f"semantic-v6 preflight field {field!r} must be numeric")
    return float(value)


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("semantic-v6 preflight identity must be SHA-256")


__all__ = [
    "V6GpuPreflight",
    "V6PreflightProgress",
    "V6_PREFLIGHT_FORMAT",
    "V6_PREFLIGHT_IDENTITY_NAMESPACE",
    "V6_PREFLIGHT_TREE_FORMAT",
    "load_v6_gpu_preflight",
    "run_and_publish_v6_gpu_preflight",
]
