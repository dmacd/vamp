"""Authenticated frozen base and deterministic Permuted-MNIST allocations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, load_canonical_json
from apm.continual.top_two_adapter import TopTwoBaseState, top_two_base_state
from apm.data.mnist.loader import load_mnist
from apm.data.mnist.permutations import identity_permutation, random_digit_permutation
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_router_config import BenchmarkConfig, VampLogTRouterConfig


@dataclass(frozen=True, slots=True)
class FrozenClassifierDependency:
    """Authenticated frozen CNN and its detached adaptable suffix."""

    model: AddressCNN
    base: TopTwoBaseState
    protocol_sha256: str
    data_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ExampleBatch:
    """Images, labels, and diagnostic provenance for one immutable batch."""

    images: Tensor
    labels: Tensor
    domain_ids: Tensor
    source_indices: Tensor
    macro_steps: Tensor

    def __post_init__(self) -> None:
        rows = len(self.labels)
        if (
            self.images.shape != (rows, 1, 28, 28)
            or self.images.dtype != torch.float32
            or self.labels.shape != (rows,)
            or self.labels.dtype != torch.int64
            or self.domain_ids.shape != (rows,)
            or self.domain_ids.dtype != torch.int64
            or self.source_indices.shape != (rows,)
            or self.source_indices.dtype != torch.int64
            or self.macro_steps.shape != (rows,)
            or self.macro_steps.dtype != torch.int64
            or not torch.isfinite(self.images).all()
        ):
            raise ValueError("Permuted-MNIST batch arrays are misaligned")

    def select(self, indices: Tensor) -> "ExampleBatch":
        """Return the requested immutable row selection."""
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError("example selections require a vector of int64 indices")
        return ExampleBatch(
            self.images[indices],
            self.labels[indices],
            self.domain_ids[indices],
            self.source_indices[indices],
            self.macro_steps[indices],
        )


@dataclass(frozen=True, slots=True)
class StepAllocation:
    """Disjoint source rows assigned to one macro-step."""

    macro_step: int
    domain_id: int
    model_indices: tuple[int, ...]
    router_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        combined = self.model_indices + self.router_indices + self.evaluation_indices
        if (
            self.macro_step < 1
            or self.domain_id < 0
            or not combined
            or len(set(combined)) != len(combined)
        ):
            raise ValueError("invalid or overlapping per-step source allocation")


@dataclass(frozen=True, slots=True)
class StepBatches:
    """Materialized model, router, and untouched evaluation batches."""

    model: ExampleBatch
    router: ExampleBatch
    evaluation: ExampleBatch


@dataclass(frozen=True, slots=True)
class PermutedMnistBenchmark:
    """Lazy eight-domain benchmark with deterministic source allocations."""

    train_images: Tensor
    train_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    permutations: tuple[Tensor, ...]
    allocations: tuple[StepAllocation, ...]
    test_subset_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        train_rows = len(self.train_labels)
        test_rows = len(self.test_labels)
        if (
            self.train_images.shape != (train_rows, 1, 28, 28)
            or self.test_images.shape != (test_rows, 1, 28, 28)
            or train_rows < 1
            or test_rows < 1
            or self.train_images.dtype != torch.float32
            or self.test_images.dtype != torch.float32
            or self.train_labels.shape != (train_rows,)
            or self.test_labels.shape != (test_rows,)
            or len(self.permutations) != 8
            or len(self.test_subset_indices) != 8
        ):
            raise ValueError("Permuted-MNIST benchmark has unexpected dimensions")
        used = {domain: [] for domain in range(8)}
        for allocation in self.allocations:
            used[allocation.domain_id].extend(
                allocation.model_indices
                + allocation.router_indices
                + allocation.evaluation_indices
            )
        if any(len(rows) != len(set(rows)) for rows in used.values()):
            raise ValueError("a domain reuses a training example before exhaustion")

    def step(self, macro_step: int) -> StepBatches:
        """Materialize all three disjoint batches for one one-based macro-step."""
        if not 1 <= macro_step <= len(self.allocations):
            raise ValueError("macro-step is outside the prepared stream")
        allocation = self.allocations[macro_step - 1]
        return StepBatches(
            *(
                self._batch(allocation.domain_id, rows, macro_step, train=True)
                for rows in (
                    allocation.model_indices,
                    allocation.router_indices,
                    allocation.evaluation_indices,
                )
            )
        )

    def test_domain(self, domain_id: int, *, full: bool) -> ExampleBatch:
        """Materialize either the fixed test subset or complete transformed domain."""
        if not 0 <= domain_id < len(self.permutations):
            raise ValueError("unknown Permuted-MNIST domain")
        rows = (
            tuple(range(len(self.test_labels)))
            if full
            else self.test_subset_indices[domain_id]
        )
        return self._batch(domain_id, rows, 0, train=False)

    def _batch(
        self,
        domain_id: int,
        rows: tuple[int, ...],
        macro_step: int,
        *,
        train: bool,
    ) -> ExampleBatch:
        source = self.train_images if train else self.test_images
        labels = self.train_labels if train else self.test_labels
        indices = torch.tensor(rows, dtype=torch.int64)
        selected = source[indices]
        permutation = self.permutations[domain_id]
        images = selected.flatten(1)[:, permutation].reshape(-1, 1, 28, 28)
        return ExampleBatch(
            images.contiguous(),
            labels[indices],
            torch.full((len(indices),), domain_id, dtype=torch.int64),
            indices,
            torch.full((len(indices),), macro_step, dtype=torch.int64),
        )


def named_seed(seed: int, *parts: object) -> int:
    """Derive a stable independent 63-bit seed from semantic coordinates."""
    payload = "\0".join(("vamp-logt-router-v1", str(seed), *(str(part) for part in parts)))
    return int(sha256(payload.encode("utf-8")).hexdigest()[:15], 16)


def resolved_device(name: str) -> torch.device:
    """Resolve the configured accelerator and reject unavailable explicit CUDA."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_frozen_classifier(config: VampLogTRouterConfig) -> FrozenClassifierDependency:
    """Authenticate and load the sealed VAMP-AF classifier dependency."""
    root = config.baseline_run_root
    protocol_path = root / "protocol.json"
    checkpoint_path = root / "base" / "model.pt"
    manifest_path = root / "base" / "manifest.json"
    missing = tuple(
        str(path)
        for path in (protocol_path, checkpoint_path, manifest_path)
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"sealed classifier dependency is incomplete: {missing}")
    protocol = load_canonical_json(protocol_path)
    manifest = load_canonical_json(manifest_path)
    data_hashes = protocol.get("data_sha256")
    if (
        protocol.get("config_hash") != config.baseline.run_id
        or protocol.get("base_checkpoint_sha256")
        != config.baseline.checkpoint_sha256
        or manifest.get("checkpoint_sha256") != config.baseline.checkpoint_sha256
        or file_sha256(checkpoint_path) != config.baseline.checkpoint_sha256
        or not isinstance(data_hashes, dict)
        or any(
            file_sha256(config.data_root / name) != expected
            for name, expected in data_hashes.items()
        )
    ):
        raise ValueError("sealed classifier or MNIST source differs from the protocol")
    model = AddressCNN()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = top_two_base_state(
        model.embedding.weight,
        model.embedding.bias,
        model.classifier.weight,
        model.classifier.bias,
    )
    return FrozenClassifierDependency(
        model,
        base,
        file_sha256(protocol_path),
        tuple(sorted((str(name), str(value)) for name, value in data_hashes.items())),
    )


def build_benchmark(
    config: VampLogTRouterConfig,
    run_seed: int,
) -> PermutedMnistBenchmark:
    """Build one seed's deterministic allocations without materializing all domains."""
    arrays = load_mnist(
        root=config.data_root,
        allow_download=False,
        npz_cache_path=None,
    )
    benchmark = config.benchmark
    permutations = (
        identity_permutation(),
        *(random_digit_permutation(seed) for seed in benchmark.permutation_seeds),
    )
    allocations = build_stream_allocations(benchmark, run_seed)
    test_subsets = tuple(
        tuple(
            int(value)
            for value in np.random.default_rng(
                named_seed(run_seed, "test-subset", domain)
            ).permutation(10_000)[: config.evaluation.test_subset_per_domain]
        )
        for domain in range(8)
    )
    return PermutedMnistBenchmark(
        torch.from_numpy(arrays.train_images).unsqueeze(1),
        torch.from_numpy(arrays.train_labels),
        torch.from_numpy(arrays.test_images).unsqueeze(1),
        torch.from_numpy(arrays.test_labels),
        tuple(torch.from_numpy(value.copy()) for value in permutations),
        allocations,
        test_subsets,
    )


def build_stream_allocations(
    benchmark: BenchmarkConfig,
    run_seed: int,
) -> tuple[StepAllocation, ...]:
    """Allocate every seed-varying training row without loading MNIST pixels."""
    schedule_generator = np.random.default_rng(benchmark.stream_seed)
    schedule = tuple(
        int(domain)
        for _block in range((benchmark.macro_steps + 7) // 8)
        for domain in schedule_generator.permutation(8)
    )[: benchmark.macro_steps]
    orders = tuple(
        np.random.default_rng(named_seed(run_seed, "domain-order", domain)).permutation(60_000)
        for domain in range(8)
    )
    cursors = [0] * 8
    allocations = []
    for macro_step, domain_id in enumerate(schedule, start=1):
        start = cursors[domain_id]
        stop = start + benchmark.examples_per_step
        if stop > 60_000:
            raise RuntimeError("a domain exhausted before the configured horizon")
        rows = orders[domain_id][start:stop]
        model_stop = benchmark.model_batch_size
        router_stop = model_stop + benchmark.router_batch_size
        allocations.append(
            StepAllocation(
                macro_step,
                domain_id,
                tuple(int(value) for value in rows[:model_stop]),
                tuple(int(value) for value in rows[model_stop:router_stop]),
                tuple(int(value) for value in rows[router_stop:]),
            )
        )
        cursors[domain_id] = stop
    return tuple(allocations)


def concatenate_batches(batches: tuple[ExampleBatch, ...]) -> ExampleBatch:
    """Concatenate a nonempty chronological batch sequence."""
    if not batches:
        raise ValueError("cannot concatenate an empty example archive")
    return ExampleBatch(
        *(torch.cat(tuple(getattr(batch, field) for batch in batches)) for field in (
            "images",
            "labels",
            "domain_ids",
            "source_indices",
            "macro_steps",
        ))
    )


__all__ = [
    "ExampleBatch",
    "FrozenClassifierDependency",
    "PermutedMnistBenchmark",
    "StepAllocation",
    "StepBatches",
    "build_benchmark",
    "build_stream_allocations",
    "concatenate_batches",
    "load_frozen_classifier",
    "named_seed",
    "resolved_device",
]
