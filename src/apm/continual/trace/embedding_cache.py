"""Resumable frozen-base prompt embedding cache and hierarchy centroids."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm

from apm.continual.artifacts import ChainedJsonlLedger, file_sha256, publish_immutable_bytes
from apm.continual.trace.data import TraceExample
from apm.continual.trace.evaluation import CandidateRuntime
from apm.continual.trace.lineage import HierarchyNode
from apm.continual.trace.routing import NodeCentroid, PromptQuery, training_centroid


EMBEDDING_LEDGER_FORMAT = "trace-prompt-embedding-v1"


def build_prompt_embedding_cache(
    runtime: CandidateRuntime,
    examples: Sequence[TraceExample],
    ledger_path: str | Path,
    tensor_path: str | Path,
    should_pause: Callable[[], bool] = lambda: False,
) -> str:
    """Embed every prompt once, checkpoint each row, and publish one tensor bundle."""
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    if not ordered or len({example.example_id for example in ordered}) != len(ordered):
        raise ValueError("embedding cache requires unique examples")
    ledger = ChainedJsonlLedger(ledger_path, EMBEDDING_LEDGER_FORMAT)
    persisted = tuple(str(row["example_id"]) for row in ledger.rows)
    expected = tuple(example.example_id for example in ordered)
    if persisted != expected[: len(persisted)]:
        raise ValueError("prompt embedding ledger is not the expected example prefix")
    print(f"TRACE phase: frozen-base prompt embeddings ({len(ordered):,} prompts)")
    bar = tqdm(
        total=len(ordered),
        initial=len(persisted),
        desc="TRACE prompt embeddings",
        unit="prompt",
        dynamic_ncols=True,
    )
    try:
        for example in ordered[len(persisted) :]:
            embedding = runtime.prompt_embedding(PromptQuery(example.example_id, example.prompt))
            ledger.append(
                {
                    "embedding": embedding.to(torch.float32).tolist(),
                    "example_id": example.example_id,
                }
            )
            bar.update(1)
            if should_pause():
                raise InterruptedError("TRACE embedding cache paused after a durable prompt")
    finally:
        bar.close()
    dimensions = {len(row["embedding"]) for row in ledger.rows}
    if len(dimensions) != 1:
        raise ValueError("prompt embedding dimensions differ")
    tensors = {
        str(row["example_id"]): torch.tensor(row["embedding"], dtype=torch.float32)
        for row in ledger.rows
    }
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("TRACE prompt cache requires safetensors") from error
    target = Path(tensor_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        existing = load_prompt_embeddings(target)
        if set(existing) != set(tensors) or any(
            not torch.equal(existing[key], value) for key, value in tensors.items()
        ):
            raise ValueError("immutable prompt embedding cache changed")
    else:
        temporary = target.with_name(f".{target.name}.building")
        save_file(tensors, temporary, metadata={"format": EMBEDDING_LEDGER_FORMAT})
        try:
            publish_immutable_bytes(target, temporary.read_bytes())
        finally:
            temporary.unlink(missing_ok=True)
    return file_sha256(target)


def load_prompt_embeddings(path: str | Path) -> dict[str, Tensor]:
    """Load the frozen-base prompt embedding tensor bundle on CPU."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("TRACE prompt cache requires safetensors") from error
    return dict(load_file(Path(path), device="cpu"))


def hierarchy_centroids(
    active_nodes: Sequence[HierarchyNode],
    examples: Sequence[TraceExample],
    embeddings: Mapping[str, Tensor],
) -> tuple[NodeCentroid, ...]:
    """Build base and active-node centroids from cached training prompt embeddings."""
    last_arrival = max(node.end_arrival for node in active_nodes)
    training = tuple(
        example
        for example in examples
        if example.split == "train"
        and example.arrival is not None
        and example.arrival <= last_arrival
    )
    base = training_centroid(
        "base",
        torch.stack(tuple(embeddings[example.example_id] for example in training)),
    )
    nodes = tuple(
        training_centroid(
            node.node_id,
            torch.stack(
                tuple(
                    embeddings[example.example_id]
                    for example in training
                    if example.arrival is not None
                    and node.start_arrival <= example.arrival <= node.end_arrival
                )
            ),
        )
        for node in active_nodes
    )
    return (base, *nodes)


__all__ = [
    "EMBEDDING_LEDGER_FORMAT",
    "build_prompt_embedding_cache",
    "hierarchy_centroids",
    "load_prompt_embeddings",
]
