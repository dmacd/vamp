"""Disposable GPU timing, capacity, and memory evidence for query-v1."""

from __future__ import annotations

from dataclasses import dataclass, replace
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

from apm.continual.language_routing import route_language_prefix
from apm.continual.language_tasks import AddressBook, RouterBatch
from apm.data.text.tinyworlds_q_semantic.adaptation import (
    prepare_query_adaptation,
)
from apm.data.text.tinyworlds_q_semantic.batching import (
    count_query_partition_microbatches,
    iter_query_partition_batches,
)
from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.scaling import (
    PreflightMeasurement,
    ResourceEstimate,
    estimate_resources,
    evaluation_schedule,
    projected_runtime_seconds,
    require_preflight_capacity,
)
from apm.data.text.tinyworlds_q_semantic.training import (
    QueryBaseTrainingConfig,
    allocator_peak_bytes,
    query_base_training_identity,
    run_query_base_training,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.losses import per_token_nll
from apm.lm.text import TextTokenizer
from apm.lm.text_data import TokenBatch
from apm.memory.graph import NodeId, init_memory_graph
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


QUERY_PREFLIGHT_FORMAT = "tinyworlds-q-semantic-gpu-preflight-v1"
QUERY_PREFLIGHT_TREE_FORMAT = "tinyworlds-q-semantic-gpu-preflight-tree-v1"


@dataclass(frozen=True, slots=True)
class QueryGpuPreflight:
    """Authenticated disposable timings and frozen resource projections."""

    directory: Path
    preflight_sha256: str
    catalog_sha256: str
    partition_sha256: str
    config_sha256: str
    training_sha256: str
    losses: tuple[float, float]
    measurement: PreflightMeasurement
    estimate: ResourceEstimate
    seconds_per_evaluation_batch: float
    base_updates_per_epoch: int
    base_validation_batches: int
    runtime_seconds: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        for value, label in (
            (self.preflight_sha256, "query preflight"),
            (self.catalog_sha256, "query preflight catalog"),
            (self.partition_sha256, "query preflight partition"),
            (self.config_sha256, "query preflight config"),
            (self.training_sha256, "query preflight training"),
        ):
            require_sha256(value, label)
        if (
            type(self.losses) is not tuple
            or len(self.losses) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in self.losses)
        ):
            raise ValueError("query preflight requires two finite losses")
        if (
            not math.isfinite(self.seconds_per_evaluation_batch)
            or self.seconds_per_evaluation_batch <= 0.0
            or type(self.base_updates_per_epoch) is not int
            or self.base_updates_per_epoch <= 0
            or type(self.base_validation_batches) is not int
            or self.base_validation_batches <= 0
        ):
            raise ValueError("query preflight base-work measurements are invalid")
        if (
            type(self.runtime_seconds) is not tuple
            or not self.runtime_seconds
            or len({name for name, _ in self.runtime_seconds})
            != len(self.runtime_seconds)
            or any(
                type(name) is not str
                or not name
                or not math.isfinite(value)
                or value <= 0.0
                for name, value in self.runtime_seconds
            )
        ):
            raise ValueError("query preflight runtime projections are invalid")


def run_and_publish_query_gpu_preflight(
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
    tokenizer: TextTokenizer,
    preset: QueryExperimentPreset,
    working_directory: str | Path,
    publication_root: str | Path,
) -> QueryGpuPreflight:
    """Run two disposable updates and measure every registered resource family."""
    if type(artifact) is not QueryPartitionArtifact:
        raise TypeError("query GPU preflight requires a strict partition")
    if type(catalog) is not ValidationCatalogView:
        raise TypeError("query GPU preflight requires a validation-only catalog")
    if (
        artifact.catalog_sha256 != catalog.catalog_sha256
        or artifact.concept_ids[: preset.active_world_count] != preset.concept_ids
    ):
        raise ValueError("query GPU preflight sources do not match the preset")
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("query GPU preflight requires a CUDA JAX device")
    working = Path(working_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError("query GPU preflight working directory is not empty")

    update_times: list[float] = []
    losses: list[float] = []

    def record_update(_cursor, nll: float, _planned: int) -> None:
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError("query GPU preflight produced a non-finite loss")
        update_times.append(time.monotonic())
        losses.append(nll)

    result = run_query_base_training(
        artifact,
        preset,
        working / "disposable-training",
        stop_after_update=2,
        progress=record_update,
    )
    if len(update_times) != 2 or result.cursor.optimizer_update != 2:
        raise RuntimeError("query GPU preflight did not complete exactly two updates")
    seconds_per_update = update_times[1] - update_times[0]
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0.0:
        raise RuntimeError("query GPU preflight update timing is invalid")

    config = QueryBaseTrainingConfig.from_preset(preset)
    validation_batch = next(
        iter_query_partition_batches(
            artifact,
            preset,
            role="base",
            split="validation",
            epoch=0,
        )
    )
    evaluation_seconds = _warm_evaluation_seconds(
        result.state.trainable,
        config,
        validation_batch,
    )
    prepared = prepare_query_adaptation(catalog, artifact, tokenizer, preset)
    probes = _stack_router_batches(prepared.task_probes[0].parent_probes)
    graph = init_memory_graph(NodeId("root"))
    packed = pack_lora_memory(
        graph,
        preset.model_config,
        preset.lora_config,
        preset.max_nodes,
        preset.max_edges,
    )
    parent_seconds = _warm_parent_seconds(
        result.state.trainable,
        preset,
        packed,
        probes,
    )
    address_book = AddressBook(
        node_ids=(NodeId("root"),) + (None,) * preset.max_edges,
        keys=np.zeros(
            (preset.max_nodes, preset.model_config.hidden_size),
            dtype=np.float32,
        ),
        valid_node_mask=np.asarray(
            (True,) + (False,) * preset.max_edges,
            dtype=np.bool_,
        ),
    )
    routing_seconds = _warm_routing_seconds(
        result.state.trainable,
        preset,
        packed,
        address_book,
        probes,
    )
    row_count = probes.input_ids.shape[0]
    estimate = estimate_resources(preset)
    peak = allocator_peak_bytes()
    measurement = PreflightMeasurement(
        seconds_per_training_update=seconds_per_update,
        seconds_per_parent_probe=parent_seconds / row_count,
        seconds_per_routing_score=routing_seconds / row_count,
        allocator_peak_bytes=peak,
        projected_result_bytes=estimate.estimated_result_bytes,
    )
    require_preflight_capacity(preset, estimate, measurement)
    base_updates_per_epoch = math.ceil(
        count_query_partition_microbatches(
            artifact,
            preset,
            role="base",
            split="train",
        )
        / preset.accumulation_microbatches
    )
    base_validation_batches = count_query_partition_microbatches(
        artifact,
        preset,
        role="base",
        split="validation",
    )
    runtime = {
        **projected_runtime_seconds(estimate, measurement),
        "base_training": (
            preset.base_epochs * base_updates_per_epoch * seconds_per_update
        ),
        "base_validation": (
            preset.base_epochs * base_validation_batches * evaluation_seconds
        ),
    }
    return _publish_preflight(
        artifact,
        catalog,
        preset,
        result.training_sha256,
        tuple(losses),  # type: ignore[arg-type]
        measurement,
        estimate,
        evaluation_seconds,
        base_updates_per_epoch,
        base_validation_batches,
        tuple(sorted(runtime.items())),
        device.device_kind,
        Path(publication_root),
    )


def load_query_gpu_preflight(
    directory: str | Path,
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
    preset: QueryExperimentPreset,
) -> QueryGpuPreflight:
    """Strictly authenticate one published query-v1 GPU preflight."""
    root = Path(directory)
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "preflight_sha256", "schema_version"}
        or tree.get("format") != QUERY_PREFLIGHT_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("preflight_sha256") != root.name
    ):
        raise ValueError("query GPU preflight tree changed")
    descriptors = _record_list(tree, "files")
    expected = {"tree.json", *(_text(item, "name") for item in descriptors)}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("query GPU preflight tree entries changed")
    for item in descriptors:
        path = root / _text(item, "name")
        if (
            path.is_symlink()
            or path.stat().st_size != _integer(item, "size_bytes")
            or _file_sha256(path) != _text(item, "sha256")
        ):
            raise ValueError("query GPU preflight file changed")
    record = _load_json(root / "preflight.json")
    required_fields = {
        "base_updates_per_epoch",
        "base_validation_batches",
        "benchmark_id",
        "catalog_sha256",
        "config",
        "config_sha256",
        "device_kind",
        "disposable_update_count",
        "estimate",
        "format",
        "jax_version",
        "losses",
        "measurement",
        "numpy_version",
        "partition_sha256",
        "platform",
        "preflight_sha256",
        "runtime_seconds",
        "sealed_test_opened",
        "seconds_per_evaluation_batch",
        "training_sha256",
    }
    if (
        set(record) != required_fields
        or record.get("format") != QUERY_PREFLIGHT_FORMAT
        or record.get("benchmark_id") != BENCHMARK_ID
        or record.get("preflight_sha256") != root.name
        or record.get("catalog_sha256") != catalog.catalog_sha256
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("config") != preset.as_record()
        or record.get("config_sha256") != preset.config_sha256
        or record.get("platform") != "gpu"
        or record.get("sealed_test_opened") is not False
        or record.get("disposable_update_count") != 2
    ):
        raise ValueError("query GPU preflight binding changed")
    core = {key: value for key, value in record.items() if key != "preflight_sha256"}
    if record_sha256(core) != root.name:
        raise ValueError("query GPU preflight identity changed")
    measurement_record = _mapping(record, "measurement")
    if set(measurement_record) != {
        "allocator_peak_bytes",
        "projected_result_bytes",
        "seconds_per_parent_probe",
        "seconds_per_routing_score",
        "seconds_per_training_update",
    }:
        raise ValueError("query GPU preflight measurement fields changed")
    measurement = PreflightMeasurement(
        seconds_per_training_update=_number(
            measurement_record,
            "seconds_per_training_update",
        ),
        seconds_per_parent_probe=_number(
            measurement_record,
            "seconds_per_parent_probe",
        ),
        seconds_per_routing_score=_number(
            measurement_record,
            "seconds_per_routing_score",
        ),
        allocator_peak_bytes=_integer(measurement_record, "allocator_peak_bytes"),
        projected_result_bytes=_integer(
            measurement_record,
            "projected_result_bytes",
        ),
    )
    estimate_record = _mapping(record, "estimate")
    if set(estimate_record) != {
        "estimated_peak_bytes",
        "estimated_result_bytes",
        "parent_probe_scores",
        "result_rows",
        "routing_candidate_scores",
        "training_updates",
        "world_count",
    }:
        raise ValueError("query GPU preflight estimate fields changed")
    estimate = ResourceEstimate(
        world_count=_integer(estimate_record, "world_count"),
        training_updates=_integer(estimate_record, "training_updates"),
        parent_probe_scores=_integer(estimate_record, "parent_probe_scores"),
        routing_candidate_scores=_integer(
            estimate_record,
            "routing_candidate_scores",
        ),
        result_rows=_integer(estimate_record, "result_rows"),
        estimated_result_bytes=_integer(
            estimate_record,
            "estimated_result_bytes",
        ),
        estimated_peak_bytes=_integer(estimate_record, "estimated_peak_bytes"),
    )
    corrected_estimate = estimate_resources(preset)
    # The first pilot/main v1 preflights counted the matching independent
    # adapter as one ordinary method and omitted the forced cross-adapter rows.
    # Those immutable measurements remain valid evidence: reconstruct that
    # exact historical count, then apply the corrected storage gate as an
    # additional requirement.  Newly published preflights use the corrected
    # estimate directly.
    legacy_rows = len(evaluation_schedule(preset)) * 60 * 9
    legacy_estimate = replace(
        corrected_estimate,
        result_rows=legacy_rows,
        estimated_result_bytes=legacy_rows * 1_024,
    )
    if estimate not in (corrected_estimate, legacy_estimate):
        raise ValueError("query GPU preflight resource estimate changed")
    require_preflight_capacity(preset, corrected_estimate, measurement)
    losses = _number_list(record, "losses")
    runtime = _mapping(record, "runtime_seconds")
    loaded = QueryGpuPreflight(
        directory=root.resolve(),
        preflight_sha256=root.name,
        catalog_sha256=_text(record, "catalog_sha256"),
        partition_sha256=_text(record, "partition_sha256"),
        config_sha256=_text(record, "config_sha256"),
        training_sha256=_text(record, "training_sha256"),
        losses=tuple(losses),  # type: ignore[arg-type]
        measurement=measurement,
        estimate=estimate,
        seconds_per_evaluation_batch=_number(
            record,
            "seconds_per_evaluation_batch",
        ),
        base_updates_per_epoch=_integer(record, "base_updates_per_epoch"),
        base_validation_batches=_integer(record, "base_validation_batches"),
        runtime_seconds=tuple(
            sorted((name, _number(runtime, name)) for name in runtime)
        ),
    )
    expected_training_sha256, _ = query_base_training_identity(
        artifact,
        preset,
    )
    expected_base_updates = math.ceil(
        count_query_partition_microbatches(
            artifact,
            preset,
            role="base",
            split="train",
        )
        / preset.accumulation_microbatches
    )
    expected_validation_batches = count_query_partition_microbatches(
        artifact,
        preset,
        role="base",
        split="validation",
    )
    expected_runtime = {
        **projected_runtime_seconds(estimate, measurement),
        "base_training": (
            preset.base_epochs
            * expected_base_updates
            * measurement.seconds_per_training_update
        ),
        "base_validation": (
            preset.base_epochs
            * expected_validation_batches
            * loaded.seconds_per_evaluation_batch
        ),
    }
    if (
        loaded.training_sha256 != expected_training_sha256
        or loaded.base_updates_per_epoch != expected_base_updates
        or loaded.base_validation_batches != expected_validation_batches
        or dict(loaded.runtime_seconds) != expected_runtime
    ):
        raise ValueError("query GPU preflight derived evidence changed")
    expected_markdown = _render_preflight(loaded)
    if (root / "preflight.md").read_text(encoding="utf-8") != expected_markdown:
        raise ValueError("query GPU preflight report changed")
    return loaded


def _warm_evaluation_seconds(
    params,
    config: QueryBaseTrainingConfig,
    batch: TokenBatch,
) -> float:
    def evaluate(value: TokenBatch):
        result = apply_gpt_neo(
            params,
            config.model_config,
            jnp.asarray(value.input_ids, dtype=jnp.int32),
            jnp.asarray(value.attention_mask, dtype=jnp.bool_),
        )
        mask = jnp.asarray(value.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            result.logits,
            jnp.asarray(value.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask) / jnp.sum(mask)

    compiled = jax.jit(evaluate)
    compiled(batch).block_until_ready()
    started = time.monotonic()
    value = compiled(batch)
    value.block_until_ready()
    elapsed = time.monotonic() - started
    if not math.isfinite(float(value)) or elapsed <= 0.0:
        raise RuntimeError("query GPU preflight evaluation timing is invalid")
    return elapsed


def _warm_parent_seconds(params, preset, packed, probes: RouterBatch) -> float:
    def score():
        return exhaustive_prefix_nll_address(
            params,
            preset.model_config,
            packed,
            preset.lora_config,
            probes,
            evaluation_microbatch_size=preset.query_chunk_size,
        )

    score().node_scores.block_until_ready()
    started = time.monotonic()
    result = score()
    result.node_scores.block_until_ready()
    elapsed = time.monotonic() - started
    if elapsed <= 0.0:
        raise RuntimeError("query GPU preflight parent timing is invalid")
    return elapsed


def _warm_routing_seconds(
    params,
    preset,
    packed,
    address_book: AddressBook,
    probes: RouterBatch,
) -> float:
    def route():
        return route_language_prefix(
            "vamp_exhaustive",
            params,
            preset.model_config,
            packed,
            preset.lora_config,
            address_book,
            probes,
            evaluation_microbatch_size=preset.query_chunk_size,
        )

    route().node_scores.block_until_ready()
    started = time.monotonic()
    result = route()
    result.node_scores.block_until_ready()
    elapsed = time.monotonic() - started
    if elapsed <= 0.0:
        raise RuntimeError("query GPU preflight routing timing is invalid")
    return elapsed


def _stack_router_batches(probes: tuple[RouterBatch, ...]) -> RouterBatch:
    if not probes or len({probe.input_ids.shape[1] for probe in probes}) != 1:
        raise ValueError("query GPU preflight probes must share one width")
    return RouterBatch(
        input_ids=np.concatenate(tuple(probe.input_ids for probe in probes), axis=0),
        attention_mask=np.concatenate(
            tuple(probe.attention_mask for probe in probes),
            axis=0,
        ),
        target_ids=np.concatenate(tuple(probe.target_ids for probe in probes), axis=0),
        loss_mask=np.concatenate(tuple(probe.loss_mask for probe in probes), axis=0),
    )


def _publish_preflight(
    artifact,
    catalog,
    preset,
    training_sha256,
    losses,
    measurement,
    estimate,
    evaluation_seconds,
    base_updates_per_epoch,
    base_validation_batches,
    runtime_seconds,
    device_kind,
    publication_root: Path,
) -> QueryGpuPreflight:
    content = {
        "base_updates_per_epoch": base_updates_per_epoch,
        "base_validation_batches": base_validation_batches,
        "benchmark_id": BENCHMARK_ID,
        "catalog_sha256": catalog.catalog_sha256,
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "device_kind": device_kind,
        "disposable_update_count": 2,
        "estimate": estimate.as_record(),
        "format": QUERY_PREFLIGHT_FORMAT,
        "jax_version": jax.__version__,
        "losses": list(losses),
        "measurement": {
            "allocator_peak_bytes": measurement.allocator_peak_bytes,
            "projected_result_bytes": measurement.projected_result_bytes,
            "seconds_per_parent_probe": measurement.seconds_per_parent_probe,
            "seconds_per_routing_score": measurement.seconds_per_routing_score,
            "seconds_per_training_update": measurement.seconds_per_training_update,
        },
        "numpy_version": np.__version__,
        "partition_sha256": artifact.partition_sha256,
        "platform": "gpu",
        "runtime_seconds": dict(runtime_seconds),
        "sealed_test_opened": False,
        "seconds_per_evaluation_batch": evaluation_seconds,
        "training_sha256": training_sha256,
    }
    identity = record_sha256(content)
    publication_root.mkdir(parents=True, exist_ok=True)
    target = publication_root / identity
    if target.exists():
        return load_query_gpu_preflight(target, artifact, catalog, preset)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-query-preflight-", dir=work_root))
    try:
        (staging / "preflight.json").write_bytes(
            canonical_json_bytes({**content, "preflight_sha256": identity})
        )
        preview = QueryGpuPreflight(
            directory=staging,
            preflight_sha256=identity,
            catalog_sha256=catalog.catalog_sha256,
            partition_sha256=artifact.partition_sha256,
            config_sha256=preset.config_sha256,
            training_sha256=training_sha256,
            losses=losses,
            measurement=measurement,
            estimate=estimate,
            seconds_per_evaluation_batch=evaluation_seconds,
            base_updates_per_epoch=base_updates_per_epoch,
            base_validation_batches=base_validation_batches,
            runtime_seconds=runtime_seconds,
        )
        (staging / "preflight.md").write_text(
            _render_preflight(preview),
            encoding="utf-8",
            newline="\n",
        )
        _write_tree(staging, identity)
        os.replace(staging, target)
    except BaseException:
        _remove_tree(staging)
        raise
    return load_query_gpu_preflight(target, artifact, catalog, preset)


def _render_preflight(preflight: QueryGpuPreflight) -> str:
    runtime = dict(preflight.runtime_seconds)
    return (
        "# TinyWorlds-Q GPU preflight\n\n"
        f"Preflight: `{preflight.preflight_sha256}`\n\n"
        f"Partition: `{preflight.partition_sha256}`\n\n"
        f"Disposable losses: {preflight.losses[0]:.9f}, "
        f"{preflight.losses[1]:.9f}\n\n"
        f"Warm update: {preflight.measurement.seconds_per_training_update:.6f} s\n\n"
        f"Warm evaluation batch: {preflight.seconds_per_evaluation_batch:.6f} s\n\n"
        f"Allocator peak: {preflight.measurement.allocator_peak_bytes} bytes\n\n"
        f"Projected base training: {_duration(runtime['base_training'])}\n\n"
        f"Projected adapter training proxy: {_duration(runtime['training'])}\n\n"
        f"Projected result bytes: {preflight.measurement.projected_result_bytes}\n\n"
        "All frozen resource limits passed. The parameters are disposable and the "
        "sealed test remained closed.\n"
    )


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
    (root / "tree.json").write_bytes(
        canonical_json_bytes(
            {
                "files": list(files),
                "format": QUERY_PREFLIGHT_TREE_FORMAT,
                "preflight_sha256": identity,
                "schema_version": 1,
            }
        )
    )


def _duration(seconds: float) -> str:
    rounded = round(seconds)
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _load_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid query GPU preflight JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"noncanonical query GPU preflight JSON: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


def _record_list(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"query GPU preflight {field} must contain records")
    return tuple(value)  # type: ignore[arg-type]


def _mapping(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"query GPU preflight {field} must be an object")
    return value


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"query GPU preflight {field} must be text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"query GPU preflight {field} must be nonnegative")
    return value


def _number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"query GPU preflight {field} must be finite")
    return float(value)


def _number_list(record: dict[str, object], field: str) -> tuple[float, ...]:
    value = record.get(field)
    if type(value) is not list or any(
        type(item) not in (int, float) or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"query GPU preflight {field} must contain finite numbers")
    return tuple(float(item) for item in value)


__all__ = [
    "QUERY_PREFLIGHT_FORMAT",
    "QUERY_PREFLIGHT_TREE_FORMAT",
    "QueryGpuPreflight",
    "load_query_gpu_preflight",
    "run_and_publish_query_gpu_preflight",
]
