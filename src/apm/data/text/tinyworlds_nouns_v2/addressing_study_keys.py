"""Frozen midpoint content and analytic-error keys for the addressing study."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v1.evaluation import build_prefix_only_query
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
    HOPFIELD_BETA,
    KEY_ARTIFACT_FORMAT,
    KeyScheme,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    PROBE_STORY_COUNT,
    NounsV2PartitionArtifact,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.parameters import GptNeoParams
from apm.memory.content_addressing import HopfieldAddressResult


KEY_TENSOR_FILENAME = "keys.safetensors"
KEY_MANIFEST_FILENAME = "manifest.json"
_SQRT_TWO = np.sqrt(np.float32(2.0))


@dataclass(frozen=True, slots=True, eq=False)
class AddressingKeyArtifact:
    """Authenticated centroids and all 36 prototypes for five frozen schemes."""

    base_parameter_checksum: str
    partition_sha256: str
    vamp_tensor_checksum: str
    node_ids: tuple[str, ...]
    probe_story_ids: tuple[tuple[str, ...], ...]
    canonical_full_centroids: np.ndarray
    midpoint_content_centroids: np.ndarray
    midpoint_content_prototypes: np.ndarray
    midpoint_content_residual_centroids: np.ndarray
    midpoint_content_residual_prototypes: np.ndarray
    tensor_sha256: str
    artifact_sha256: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.base_parameter_checksum, "addressing-key base"),
            (self.partition_sha256, "addressing-key partition"),
            (self.vamp_tensor_checksum, "addressing-key VAMP tensors"),
            (self.tensor_sha256, "addressing-key tensors"),
            (self.artifact_sha256, "addressing-key artifact"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        node_count = len(self.node_ids)
        if node_count == 0 or len(self.probe_story_ids) != node_count:
            raise ValueError("addressing keys require matching nonempty nodes and probes")
        if any(len(row) != PROBE_STORY_COUNT for row in self.probe_story_ids):
            raise ValueError("every addressing node requires all 36 registered probes")
        arrays = (
            self.canonical_full_centroids,
            self.midpoint_content_centroids,
            self.midpoint_content_prototypes,
            self.midpoint_content_residual_centroids,
            self.midpoint_content_residual_prototypes,
        )
        expected_prefixes = (
            (node_count,),
            (node_count,),
            (node_count, PROBE_STORY_COUNT),
            (node_count,),
            (node_count, PROBE_STORY_COUNT),
        )
        if any(
            array.shape[: len(prefix)] != prefix
            or array.dtype != np.dtype(np.float32)
            or np.any(~np.isfinite(array))
            for array, prefix in zip(arrays, expected_prefixes)
        ):
            raise ValueError("addressing key arrays have invalid shapes or values")
        if self.canonical_full_centroids.shape[1] != self.midpoint_content_centroids.shape[1]:
            raise ValueError("canonical and midpoint content dimensions differ")
        if (
            self.midpoint_content_residual_centroids.shape[1]
            != 2 * self.midpoint_content_centroids.shape[1]
        ):
            raise ValueError("fused addressing dimension must be twice content width")
        for array in arrays:
            norms = np.linalg.norm(array, axis=-1)
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise ValueError("every addressing centroid and prototype must be unit norm")
            array.flags.writeable = False


def analytic_final_hidden_residual(
    logits: jax.Array,
    token_embedding: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
) -> jax.Array:
    """Compute the normalized masked-mean cross-entropy gradient analytically."""
    scores = jnp.asarray(logits, dtype=jnp.float32)
    embeddings = jnp.asarray(token_embedding, dtype=jnp.float32)
    targets = jnp.asarray(target_ids, dtype=jnp.int32)
    mask = jnp.asarray(loss_mask, dtype=jnp.float32)
    if scores.ndim != 3 or embeddings.ndim != 2:
        raise ValueError("residual logits and token embedding must have ranks three and two")
    if targets.shape != scores.shape[:2] or mask.shape != targets.shape:
        raise ValueError("residual targets and mask must match [batch, transitions]")
    if scores.shape[-1] != embeddings.shape[0]:
        raise ValueError("residual vocabulary dimensions differ")
    probabilities = jax.nn.softmax(scores, axis=-1)
    expected_embeddings = jnp.einsum(
        "btv,vh->bth",
        probabilities,
        embeddings,
    )
    gradients = expected_embeddings - embeddings[targets]
    pooled = jnp.sum(gradients * mask[..., None], axis=1) / jnp.sum(
        mask,
        axis=1,
        keepdims=True,
    )
    return _l2_normalize(pooled)


@partial(jax.jit, static_argnames=("model_config",))
def _encode_midpoint_batch(
    base_params: GptNeoParams,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
    *,
    model_config: GptNeoConfig,
) -> tuple[jax.Array, jax.Array]:
    result = apply_gpt_neo(
        base_params,
        model_config,
        input_ids,
        attention_mask,
        training=False,
    )
    content_weights = attention_mask.astype(jnp.float32)[..., None]
    content = _l2_normalize(
        jnp.sum(result.final_hidden * content_weights, axis=1)
        / jnp.sum(content_weights, axis=1)
    )
    residual = analytic_final_hidden_residual(
        result.logits,
        base_params.token_embedding,
        target_ids,
        loss_mask,
    )
    return content.astype(jnp.float32), residual.astype(jnp.float32)


def encode_midpoint_content_and_residual(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    prefix_batch: RouterBatch,
    *,
    microbatch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode prefix-only content and error signatures in bounded microbatches."""
    if not isinstance(prefix_batch, RouterBatch):
        raise TypeError("midpoint encoding requires a RouterBatch")
    if type(microbatch_size) is not int or microbatch_size <= 0:
        raise ValueError("midpoint encoding microbatch size must be positive")
    row_count = prefix_batch.input_ids.shape[0]
    if row_count == 0 or np.any(np.sum(prefix_batch.loss_mask, axis=-1) == 0):
        raise ValueError("midpoint encoding requires active prefix transitions")
    encoded = tuple(
        _encode_midpoint_batch(
            base_params,
            jnp.asarray(prefix_batch.input_ids[start:stop], dtype=jnp.int32),
            jnp.asarray(prefix_batch.attention_mask[start:stop], dtype=jnp.bool_),
            jnp.asarray(prefix_batch.target_ids[start:stop], dtype=jnp.int32),
            jnp.asarray(prefix_batch.loss_mask[start:stop], dtype=jnp.bool_),
            model_config=model_config,
        )
        for start in range(0, row_count, microbatch_size)
        for stop in (min(row_count, start + microbatch_size),)
    )
    content = np.concatenate(
        tuple(np.asarray(values[0], dtype=np.float32) for values in encoded),
        axis=0,
    )
    residual = np.concatenate(
        tuple(np.asarray(values[1], dtype=np.float32) for values in encoded),
        axis=0,
    )
    return content, residual


def midpoint_probe_batch(
    partition: NounsV2PartitionArtifact,
    model_config: GptNeoConfig,
) -> tuple[RouterBatch, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Build all root/task probes with the exact validation-query midpoint rule."""
    store = IndexedStoryStore(partition)
    node_ids = ("root", *partition.task_ids)
    index_names = ("root-probes",) + tuple(
        f"task-{task_id}-probes" for task_id in partition.task_ids
    )
    entries_by_node = tuple(load_story_index(partition, name) for name in index_names)
    if any(len(entries) != PROBE_STORY_COUNT for entries in entries_by_node):
        raise ValueError("addressing study requires all 36 probes for every node")
    queries = tuple(
        build_prefix_only_query(
            entry.story_id,
            store.tokens(entry),
            partition.pad_token_id,
            model_config.max_position_embeddings,
        )
        for entries in entries_by_node
        for entry in entries
    )
    batch = stack_prefix_only_queries(
        tuple(query.router_batch for query in queries),
        model_config.max_position_embeddings,
    )
    return (
        batch,
        node_ids,
        tuple(tuple(entry.story_id for entry in entries) for entries in entries_by_node),
    )


def stack_prefix_only_queries(
    batches: tuple[RouterBatch, ...],
    maximum_position_embeddings: int,
) -> RouterBatch:
    """Right-pad prefix-only rows into the existing 32-token width buckets."""
    if not batches or any(batch.input_ids.shape[0] != 1 for batch in batches):
        raise ValueError("prefix stacking requires nonempty one-row RouterBatch values")
    maximum_width = max(batch.input_ids.shape[1] for batch in batches)
    bucket_width = min(
        maximum_position_embeddings,
        ((maximum_width + 31) // 32) * 32,
    )
    shape = (len(batches), bucket_width)
    arrays = tuple(np.zeros(shape, dtype=dtype) for dtype in (np.int32, np.int32, np.bool_, np.bool_))
    inputs, targets, attention, losses = arrays
    for row, batch in enumerate(batches):
        width = batch.input_ids.shape[1]
        inputs[row, :width] = batch.input_ids[0]
        targets[row, :width] = batch.target_ids[0]
        attention[row, :width] = batch.attention_mask[0]
        losses[row, :width] = batch.loss_mask[0]
    return RouterBatch(inputs, attention, targets, losses)


def build_or_load_addressing_keys(
    partition: NounsV2PartitionArtifact,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    adaptation: LanguageAdaptationArtifact,
    output_directory: str | Path,
) -> AddressingKeyArtifact:
    """Derive once or strict-load the frozen probe-key tensor artifact."""
    root = Path(output_directory)
    manifest_path = root / KEY_MANIFEST_FILENAME
    if manifest_path.is_file():
        artifact = load_addressing_keys(root)
        _validate_addressing_key_bindings(artifact, partition, adaptation)
        return artifact
    if root.exists() and any(root.iterdir()):
        raise ValueError("addressing-key directory is nonempty but incomplete")
    root.mkdir(parents=True, exist_ok=True)
    probe_batch, node_ids, probe_story_ids = midpoint_probe_batch(
        partition,
        model_config,
    )
    validation_ids = {
        entry.story_id
        for task_id in partition.task_ids
        for entry in load_story_index(partition, f"task-{task_id}-generation")
    }
    if validation_ids & {story_id for row in probe_story_ids for story_id in row}:
        raise ValueError("validation examples must not contribute to addressing keys")
    content, residual = encode_midpoint_content_and_residual(
        base_params,
        model_config,
        probe_batch,
    )
    node_count = len(node_ids)
    hidden_size = model_config.hidden_size
    content_prototypes = content.reshape(node_count, PROBE_STORY_COUNT, hidden_size)
    residual_prototypes = residual.reshape(node_count, PROBE_STORY_COUNT, hidden_size)
    fused_prototypes = np.concatenate(
        (content_prototypes / _SQRT_TWO, residual_prototypes / _SQRT_TWO),
        axis=-1,
    ).astype(np.float32)
    arrays = {
        "canonical_full_centroids": np.asarray(
            adaptation.address_book.keys[:node_count],
            dtype=np.float32,
        ),
        "midpoint_content_centroids": _normalize(
            np.mean(content_prototypes, axis=1, dtype=np.float32)
        ),
        "midpoint_content_prototypes": content_prototypes,
        "midpoint_content_residual_centroids": _normalize(
            np.mean(fused_prototypes, axis=1, dtype=np.float32)
        ),
        "midpoint_content_residual_prototypes": fused_prototypes,
    }
    tensor_path = root / KEY_TENSOR_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(prefix=".keys.", dir=root)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            arrays,
            temporary,
            metadata={"format": KEY_ARTIFACT_FORMAT},
        )
        os.replace(temporary, tensor_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    tensor_sha256 = _file_sha256(tensor_path)
    core = {
        "base_parameter_checksum": adaptation.base_checkpoint.parameter_checksum,
        "format": KEY_ARTIFACT_FORMAT,
        "node_ids": list(node_ids),
        "partition_sha256": partition.partition_sha256,
        "probe_story_ids": [list(row) for row in probe_story_ids],
        "tensor_file": KEY_TENSOR_FILENAME,
        "tensor_sha256": tensor_sha256,
        "vamp_tensor_checksum": adaptation.tensor_checksum,
    }
    _atomic_write(
        manifest_path,
        canonical_json_bytes({**core, "artifact_sha256": record_sha256(core)}),
    )
    artifact = load_addressing_keys(root)
    _validate_addressing_key_bindings(artifact, partition, adaptation)
    return artifact


def load_addressing_keys(path: str | Path) -> AddressingKeyArtifact:
    """Strict-load the self-hashing manifest and deterministic safetensors keys."""
    root = Path(path)
    payload = (root / KEY_MANIFEST_FILENAME).read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError("addressing-key manifest is not canonical JSON")
    supplied = record.get("artifact_sha256")
    core = {key: value for key, value in record.items() if key != "artifact_sha256"}
    tensor_path = root / KEY_TENSOR_FILENAME
    if (
        set(record)
        != {
            "artifact_sha256",
            "base_parameter_checksum",
            "format",
            "node_ids",
            "partition_sha256",
            "probe_story_ids",
            "tensor_file",
            "tensor_sha256",
            "vamp_tensor_checksum",
        }
        or record.get("format") != KEY_ARTIFACT_FORMAT
        or record.get("tensor_file") != KEY_TENSOR_FILENAME
        or supplied != record_sha256(core)
        or not tensor_path.is_file()
        or record.get("tensor_sha256") != _file_sha256(tensor_path)
        or {item.name for item in root.iterdir()}
        != {KEY_MANIFEST_FILENAME, KEY_TENSOR_FILENAME}
    ):
        raise ValueError("addressing-key artifact identity changed")
    arrays = load_file(tensor_path)
    expected_names = {
        "canonical_full_centroids",
        "midpoint_content_centroids",
        "midpoint_content_prototypes",
        "midpoint_content_residual_centroids",
        "midpoint_content_residual_prototypes",
    }
    if set(arrays) != expected_names:
        raise ValueError("addressing-key tensor names changed")
    return AddressingKeyArtifact(
        base_parameter_checksum=str(record["base_parameter_checksum"]),
        partition_sha256=str(record["partition_sha256"]),
        vamp_tensor_checksum=str(record["vamp_tensor_checksum"]),
        node_ids=tuple(str(value) for value in record["node_ids"]),
        probe_story_ids=tuple(
            tuple(str(value) for value in row) for row in record["probe_story_ids"]
        ),
        canonical_full_centroids=np.asarray(arrays["canonical_full_centroids"]),
        midpoint_content_centroids=np.asarray(arrays["midpoint_content_centroids"]),
        midpoint_content_prototypes=np.asarray(arrays["midpoint_content_prototypes"]),
        midpoint_content_residual_centroids=np.asarray(
            arrays["midpoint_content_residual_centroids"]
        ),
        midpoint_content_residual_prototypes=np.asarray(
            arrays["midpoint_content_residual_prototypes"]
        ),
        tensor_sha256=str(record["tensor_sha256"]),
        artifact_sha256=str(supplied),
        root=root,
    )


def _validate_addressing_key_bindings(
    keys: AddressingKeyArtifact,
    partition: NounsV2PartitionArtifact,
    adaptation: LanguageAdaptationArtifact,
) -> None:
    expected_node_ids = tuple(str(node.node_id) for node in adaptation.vamp_graph.nodes)
    expected_probe_ids = tuple(
        tuple(entry.story_id for entry in load_story_index(partition, index_name))
        for index_name in (
            "root-probes",
            *(f"task-{task_id}-probes" for task_id in partition.task_ids),
        )
    )
    validation_ids = {
        entry.story_id
        for task_id in partition.task_ids
        for entry in load_story_index(partition, f"task-{task_id}-generation")
    }
    if (
        keys.base_parameter_checksum
        != adaptation.base_checkpoint.parameter_checksum
        or keys.partition_sha256 != partition.partition_sha256
        or keys.vamp_tensor_checksum != adaptation.tensor_checksum
        or keys.node_ids != expected_node_ids
        or keys.probe_story_ids != expected_probe_ids
        or validation_ids & {story_id for row in keys.probe_story_ids for story_id in row}
    ):
        raise ValueError("addressing-key artifact bindings changed")


def score_key_scheme(
    content_queries: np.ndarray,
    residual_queries: np.ndarray,
    keys: AddressingKeyArtifact,
    scheme: KeyScheme,
) -> np.ndarray:
    """Score every node by centroid cosine or maximum prototype cosine."""
    content = np.asarray(content_queries, dtype=np.float32)
    residual = np.asarray(residual_queries, dtype=np.float32)
    if content.ndim != 2 or residual.shape != content.shape:
        raise ValueError("content and residual queries must share [batch, hidden]")
    if scheme == "canonical_full_centroid":
        return content @ keys.canonical_full_centroids.T
    if scheme == "midpoint_content_centroid":
        return content @ keys.midpoint_content_centroids.T
    if scheme == "midpoint_content_prototype":
        return np.max(
            np.einsum(
                "bh,nph->bnp",
                content,
                keys.midpoint_content_prototypes,
            ),
            axis=-1,
        )
    fused = np.concatenate(
        (content / _SQRT_TWO, residual / _SQRT_TWO),
        axis=-1,
    ).astype(np.float32)
    if scheme == "midpoint_content_residual_centroid":
        return fused @ keys.midpoint_content_residual_centroids.T
    if scheme == "midpoint_content_residual_prototype":
        return np.max(
            np.einsum(
                "bh,nph->bnp",
                fused,
                keys.midpoint_content_residual_prototypes,
            ),
            axis=-1,
        )
    raise ValueError(f"unknown addressing key scheme: {scheme}")


def stable_hopfield_result(
    node_scores: np.ndarray,
    *,
    top_k: int = 8,
    beta: float = HOPFIELD_BETA,
) -> HopfieldAddressResult:
    """Convert scores to stable index-tiebroken Hopfield probabilities and ranks."""
    scores = np.asarray(node_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < top_k:
        raise ValueError("Hopfield scores must be nonempty and hold the requested top-k")
    if np.any(~np.isfinite(scores)) or top_k <= 0 or beta <= 0.0:
        raise ValueError("Hopfield scores, width, and beta must be finite and positive")
    node_indices = np.arange(scores.shape[1], dtype=np.int32)
    ranking = np.stack(
        tuple(np.lexsort((node_indices, -row)) for row in scores),
        axis=0,
    ).astype(np.int32)
    shifted = beta * scores - np.max(beta * scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted).astype(np.float32)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    entropy = -np.sum(
        np.where(
            probabilities > 0.0,
            probabilities * np.log(np.maximum(probabilities, np.finfo(np.float32).tiny)),
            0.0,
        ),
        axis=-1,
    )
    margins = scores[np.arange(scores.shape[0]), ranking[:, 0]] - scores[
        np.arange(scores.shape[0]), ranking[:, 1]
    ]
    return HopfieldAddressResult(
        selected_indices=jnp.asarray(ranking[:, 0], dtype=jnp.int32),
        node_probabilities=jnp.asarray(probabilities, dtype=jnp.float32),
        node_scores=jnp.asarray(scores, dtype=jnp.float32),
        score_margin=jnp.asarray(margins, dtype=jnp.float32),
        entropy=jnp.asarray(entropy, dtype=jnp.float32),
        top_k_indices=jnp.asarray(ranking[:, :top_k], dtype=jnp.int32),
    )


def _l2_normalize(vectors: jax.Array) -> jax.Array:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, jnp.finfo(jnp.float32).tiny)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("addressing-key centroid cannot be zero or nonfinite")
    return np.asarray(values / norms, dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "AddressingKeyArtifact",
    "analytic_final_hidden_residual",
    "build_or_load_addressing_keys",
    "encode_midpoint_content_and_residual",
    "load_addressing_keys",
    "midpoint_probe_batch",
    "score_key_scheme",
    "stable_hopfield_result",
    "stack_prefix_only_queries",
]
