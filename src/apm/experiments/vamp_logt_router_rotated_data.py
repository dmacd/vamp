"""Authenticated VAMP-AF Rotated-MNIST allocations for behavioral routing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, load_canonical_json
from apm.data.mnist.loader import load_mnist
from apm.experiments.vamp_logt_router_data import (
    ExampleBatch,
    StepAllocation,
    StepBatches,
    named_seed,
)
from apm.experiments.vamp_logt_router_rotated_config import (
    RotatedBenchmarkConfig,
    VampLogTRotatedRouterConfig,
)


@dataclass(frozen=True, slots=True)
class RotatedMnistBenchmark:
    """Lazy five-context benchmark over exact VAMP-AF source identities."""

    train_images: Tensor
    train_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    rotations_deg: tuple[float, ...]
    label_shifts: tuple[int, ...]
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
            or self.train_labels.dtype != torch.int64
            or self.test_labels.dtype != torch.int64
            or len(self.rotations_deg) != 5
            or len(self.label_shifts) != 5
            or len(self.test_subset_indices) != 5
        ):
            raise ValueError("Rotated-MNIST benchmark has unexpected dimensions")
        used = {domain: [] for domain in range(5)}
        for allocation in self.allocations:
            used[allocation.domain_id].extend(
                allocation.model_indices
                + allocation.router_indices
                + allocation.evaluation_indices
            )
        if any(len(rows) != len(set(rows)) for rows in used.values()):
            raise ValueError("a rotated context reuses a training identity")

    def step(self, macro_step: int) -> StepBatches:
        """Materialize one blocked context's three disjoint batches."""
        if not 1 <= macro_step <= len(self.allocations):
            raise ValueError("macro-step is outside the prepared rotated stream")
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
        """Materialize a fixed subset or the full transformed test context."""
        if not 0 <= domain_id < len(self.rotations_deg):
            raise ValueError("unknown Rotated-MNIST context")
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
        images = rotate_images(source[indices], self.rotations_deg[domain_id])
        shifted = (labels[indices] + self.label_shifts[domain_id]) % 10
        return ExampleBatch(
            images,
            shifted,
            torch.full((len(indices),), domain_id, dtype=torch.int64),
            indices,
            torch.full((len(indices),), macro_step, dtype=torch.int64),
        )


def rotate_images(images: Tensor, angle: float) -> Tensor:
    """Apply the exact VAMP-AF bilinear, zero-filled rotation."""
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("torchvision is required by the vision environment") from error
    return rotate(
        images,
        float(angle),
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    ).contiguous()


def build_benchmark(
    config: VampLogTRotatedRouterConfig,
    phase_name: str,
    run_seed: int,
) -> RotatedMnistBenchmark:
    """Build one phase/seed allocation and authenticate the parent task."""
    if phase_name not in {"smoke", "primary"}:
        raise ValueError("rotated benchmark phase must be smoke or primary")
    arrays = load_mnist(
        root=config.data_root,
        allow_download=False,
        npz_cache_path=None,
    )
    _validate_parent_task(config)
    main_indices = np.asarray(
        balanced_indices(
            arrays.train_labels,
            config.task.primary_source_examples_per_context,
            config.task.source_selection_seed,
        ),
        dtype=np.int64,
    )
    actual_hash = source_indices_sha256(main_indices)
    if actual_hash != config.task.source_indices_sha256:
        raise ValueError("recomputed VAMP-AF source identities changed")
    context_pools = _context_pools(
        arrays.train_labels,
        main_indices,
        config,
        phase_name,
    )
    context_steps = (
        config.task.smoke_context_steps
        if phase_name == "smoke"
        else config.task.primary_context_steps
    )
    allocations = build_stream_allocations(
        config.benchmark,
        context_steps,
        context_pools,
        run_seed,
    )
    test_subsets = tuple(
        tuple(
            int(value)
            for value in np.random.default_rng(
                named_seed(run_seed, "rotated-test-subset", domain)
            ).permutation(len(arrays.test_labels))[
                : config.evaluation.test_subset_per_domain
            ]
        )
        for domain in range(config.task.domain_count)
    )
    return RotatedMnistBenchmark(
        torch.from_numpy(arrays.train_images).unsqueeze(1),
        torch.from_numpy(arrays.train_labels),
        torch.from_numpy(arrays.test_images).unsqueeze(1),
        torch.from_numpy(arrays.test_labels),
        config.task.rotations_deg,
        config.task.label_shifts,
        allocations,
        test_subsets,
    )


def build_stream_allocations(
    benchmark: RotatedBenchmarkConfig,
    context_steps: tuple[int, ...],
    context_pools: tuple[np.ndarray, ...],
    run_seed: int,
) -> tuple[StepAllocation, ...]:
    """Allocate one deterministic blocked stream from per-context identities."""
    if len(context_steps) != len(context_pools):
        raise ValueError("context steps and source pools are misaligned")
    schedule = tuple(
        context_id
        for context_id, step_count in enumerate(context_steps)
        for _ in range(step_count)
    )
    orders = tuple(
        np.random.default_rng(run_seed * 10_000 + context_id).permutation(pool)
        for context_id, pool in enumerate(context_pools)
    )
    cursors = [0] * len(context_pools)
    allocations = []
    for macro_step, context_id in enumerate(schedule, start=1):
        start = cursors[context_id]
        stop = start + benchmark.examples_per_step
        if stop > len(orders[context_id]):
            raise RuntimeError("a VAMP-AF context exhausted before its blocked schedule")
        rows = orders[context_id][start:stop]
        model_stop = benchmark.model_batch_size
        router_stop = model_stop + benchmark.router_batch_size
        allocations.append(
            StepAllocation(
                macro_step,
                context_id,
                tuple(int(value) for value in rows[:model_stop]),
                tuple(int(value) for value in rows[model_stop:router_stop]),
                tuple(int(value) for value in rows[router_stop:]),
            )
        )
        cursors[context_id] = stop
    return tuple(allocations)


def balanced_indices(labels: np.ndarray, count: int, seed: int) -> tuple[int, ...]:
    """Reproduce VAMP-AF's deterministic per-label source selection."""
    values = np.asarray(labels, dtype=np.int64)
    unique = tuple(int(value) for value in np.unique(values))
    target = min(int(count), values.shape[0])
    base, remainder = divmod(target, len(unique))
    rng = np.random.default_rng(seed)
    selected = tuple(
        int(index)
        for label_index, label in enumerate(unique)
        for index in rng.choice(
            np.flatnonzero(values == label),
            size=base + int(label_index < remainder),
            replace=False,
        )
    )
    return tuple(sorted(selected))


def source_indices_sha256(indices: np.ndarray) -> str:
    """Hash ordered source identities with an explicit portable byte order."""
    values = np.asarray(indices, dtype="<i8")
    return sha256(values.tobytes()).hexdigest()


def _context_pools(
    labels: np.ndarray,
    main_indices: np.ndarray,
    config: VampLogTRotatedRouterConfig,
    phase_name: str,
) -> tuple[np.ndarray, ...]:
    if phase_name == "primary":
        return tuple(main_indices.copy() for _ in config.task.label_shifts)
    pools = []
    for shift in config.task.label_shifts:
        shifted = (labels[main_indices] + shift) % 10
        positions = np.asarray(
            balanced_indices(
                shifted,
                config.task.smoke_source_examples_per_context,
                config.task.source_selection_seed,
            ),
            dtype=np.int64,
        )
        pools.append(main_indices[positions])
    return tuple(pools)


def _validate_parent_task(config: VampLogTRotatedRouterConfig) -> None:
    protocol = load_canonical_json(config.baseline_run_root / "protocol.json")
    parent = protocol.get("config", {})
    data = parent.get("data", {}) if isinstance(parent, dict) else {}
    material = protocol.get("material_source_sha256", {})
    project_root = Path(__file__).resolve().parents[3]
    parent_config_path = project_root / "configs/vamp_af_mnist/poc.yaml"
    if (
        protocol.get("config_hash") != config.baseline.run_id
        or protocol.get("base_checkpoint_sha256") != config.baseline.checkpoint_sha256
        or data.get("rotations_deg") != list(config.task.rotations_deg)
        or data.get("label_shifts") != list(config.task.label_shifts)
        or data.get("interpolation") != config.task.interpolation
        or data.get("main_examples_per_context")
        != config.task.primary_source_examples_per_context
        or parent.get("base", {}).get("seed") != config.task.source_selection_seed
        or material.get("configs/vamp_af_mnist/poc.yaml")
        != config.task.vamp_af_config_sha256
        or file_sha256(parent_config_path) != config.task.vamp_af_config_sha256
    ):
        raise ValueError("VAMP-AF parent task differs from the rotated-router protocol")


__all__ = [
    "RotatedMnistBenchmark",
    "balanced_indices",
    "build_benchmark",
    "build_stream_allocations",
    "rotate_images",
    "source_indices_sha256",
]
