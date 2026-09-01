"""Deterministic raw-pixel data for the dense Permuted-MNIST experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, record_sha256
from apm.data.mnist.loader import MNIST_RAW_FILES, load_mnist
from apm.data.mnist.permutations import identity_permutation, random_digit_permutation
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig


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
            or self.labels.shape != self.domain_ids.shape != self.source_indices.shape != self.macro_steps.shape
            or self.labels.dtype != torch.int64
            or self.domain_ids.dtype != torch.int64
            or self.source_indices.dtype != torch.int64
            or self.macro_steps.dtype != torch.int64
            or not torch.isfinite(self.images).all()
        ):
            raise ValueError("dense Permuted-MNIST batch arrays are misaligned")

    def select(self, indices: Tensor) -> "ExampleBatch":
        """Return the requested immutable row selection."""
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError("example selections require an int64 vector")
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
    observer_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        combined = self.model_indices + self.observer_indices + self.evaluation_indices
        if self.macro_step < 1 or not 0 <= self.domain_id < 8 or len(set(combined)) != len(combined):
            raise ValueError("invalid or overlapping dense stream allocation")


@dataclass(frozen=True, slots=True)
class StepBatches:
    """Materialized model, observer, and held-out evaluation batches."""

    model: ExampleBatch
    observer: ExampleBatch
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
        if (
            self.train_images.shape != (len(self.train_labels), 1, 28, 28)
            or self.test_images.shape != (len(self.test_labels), 1, 28, 28)
            or self.train_images.dtype != torch.float32
            or self.test_images.dtype != torch.float32
            or self.train_labels.dtype != self.test_labels.dtype != torch.int64
            or len(self.permutations) != 8
            or len(self.test_subset_indices) != 8
        ):
            raise ValueError("dense Permuted-MNIST benchmark has unexpected dimensions")
        used = {domain: [] for domain in range(8)}
        for allocation in self.allocations:
            used[allocation.domain_id].extend(
                allocation.model_indices + allocation.observer_indices + allocation.evaluation_indices
            )
        if any(len(rows) != len(set(rows)) for rows in used.values()):
            raise ValueError("a domain reuses a training example before exhaustion")

    def step(self, macro_step: int) -> StepBatches:
        """Materialize all three disjoint batches for one one-based step."""
        if not 1 <= macro_step <= len(self.allocations):
            raise ValueError("macro-step is outside the prepared stream")
        allocation = self.allocations[macro_step - 1]
        return StepBatches(
            *(self._batch(allocation.domain_id, rows, macro_step, train=True) for rows in (
                allocation.model_indices,
                allocation.observer_indices,
                allocation.evaluation_indices,
            ))
        )

    def test_domain(self, domain_id: int, *, full: bool) -> ExampleBatch:
        """Materialize a fixed subset or the complete transformed test domain."""
        if not 0 <= domain_id < 8:
            raise ValueError("unknown Permuted-MNIST domain")
        rows = tuple(range(len(self.test_labels))) if full else self.test_subset_indices[domain_id]
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
        images = source[indices].flatten(1)[:, self.permutations[domain_id]].reshape(-1, 1, 28, 28)
        return ExampleBatch(
            images.contiguous(),
            labels[indices],
            torch.full((len(indices),), domain_id, dtype=torch.int64),
            indices,
            torch.full((len(indices),), macro_step, dtype=torch.int64),
        )


def named_seed(seed: int, *parts: object) -> int:
    """Derive a stable independent 63-bit seed from semantic coordinates."""
    payload = "\0".join(("vamp-logt-dense-v1", str(seed), *(str(part) for part in parts)))
    return int(sha256(payload.encode("utf-8")).hexdigest()[:15], 16)


def resolved_device(name: str) -> torch.device:
    """Resolve the configured accelerator and reject unavailable explicit CUDA."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_benchmark(config: VampLogTDenseConfig, run_seed: int) -> PermutedMnistBenchmark:
    """Build one seed's stream without materializing all transformed domains."""
    arrays = load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)
    permutations = (
        torch.from_numpy(identity_permutation().copy()),
        *(
            torch.from_numpy(random_digit_permutation(seed).copy())
            for seed in config.benchmark.permutation_seeds
        ),
    )
    allocations = build_stream_allocations(config, run_seed)
    test_subsets = tuple(
        tuple(
            int(value)
            for value in np.random.default_rng(named_seed(run_seed, "test-subset", domain)).permutation(10_000)[
                : config.evaluation.test_subset_per_domain
            ]
        )
        for domain in range(8)
    )
    return PermutedMnistBenchmark(
        torch.from_numpy(arrays.train_images).unsqueeze(1),
        torch.from_numpy(arrays.train_labels),
        torch.from_numpy(arrays.test_images).unsqueeze(1),
        torch.from_numpy(arrays.test_labels),
        permutations,
        allocations,
        test_subsets,
    )


def build_stream_allocations(
    config: VampLogTDenseConfig,
    run_seed: int,
) -> tuple[StepAllocation, ...]:
    """Allocate every seed-varying source row without loading pixels."""
    benchmark = config.benchmark
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
        observer_stop = model_stop + benchmark.observer_batch_size
        allocations.append(
            StepAllocation(
                macro_step,
                domain_id,
                tuple(int(value) for value in rows[:model_stop]),
                tuple(int(value) for value in rows[model_stop:observer_stop]),
                tuple(int(value) for value in rows[observer_stop:]),
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
            "images", "labels", "domain_ids", "source_indices", "macro_steps",
        ))
    )


def stratified_source_split(labels: Tensor, validation_count: int, seed: int) -> tuple[Tensor, Tensor]:
    """Return deterministic disjoint train/validation IDs with proportional classes."""
    if labels.ndim != 1 or labels.dtype != torch.int64 or not 0 < validation_count < len(labels):
        raise ValueError("invalid stratified split request")
    class_counts = torch.bincount(labels, minlength=10)
    exact = class_counts.to(torch.float64) * validation_count / len(labels)
    quotas = torch.floor(exact).to(torch.int64)
    remainder = validation_count - int(quotas.sum().item())
    fractions = exact - quotas
    priority = sorted(range(10), key=lambda digit: (-float(fractions[digit]), digit))
    quotas[torch.tensor(priority[:remainder], dtype=torch.int64)] += 1
    validation_rows = []
    training_rows = []
    for digit in range(10):
        rows = torch.nonzero(labels == digit, as_tuple=True)[0]
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(named_seed(seed, "split", digit)))
        selected = rows[order]
        cut = int(quotas[digit].item())
        validation_rows.append(selected[:cut])
        training_rows.append(selected[cut:])
    training = torch.cat(training_rows)
    validation = torch.cat(validation_rows)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(named_seed(seed, "training-order")))]
    validation = validation[torch.randperm(len(validation), generator=torch.Generator().manual_seed(named_seed(seed, "validation-order")))]
    return training, validation


def source_manifest(config: VampLogTDenseConfig) -> dict[str, object]:
    """Hash the raw IDX population and fixed permutation definitions."""
    raw_files = []
    for name in MNIST_RAW_FILES:
        candidates = (config.data_root / name, config.data_root / f"{name}.gz")
        try:
            raw_files.append(next(path for path in candidates if path.is_file()))
        except StopIteration as error:
            raise FileNotFoundError(f"MNIST source file is missing: {name}") from error
    permutations = (
        identity_permutation(),
        *(random_digit_permutation(seed) for seed in config.benchmark.permutation_seeds),
    )
    return {
        "idx_sha256": {path.name: file_sha256(path) for path in raw_files},
        "permutations_sha256": record_sha256([value.tolist() for value in permutations]),
    }


__all__ = [
    "ExampleBatch",
    "PermutedMnistBenchmark",
    "StepAllocation",
    "StepBatches",
    "build_benchmark",
    "build_stream_allocations",
    "concatenate_batches",
    "named_seed",
    "resolved_device",
    "source_manifest",
    "stratified_source_split",
]
