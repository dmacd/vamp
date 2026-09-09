"""Bounded image loading after adaptive selection, with stateless augmentations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import multiprocessing
from pathlib import Path

import torch
from torch import Tensor

from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset, image_transforms


@dataclass(frozen=True, slots=True)
class Presentation:
    """One selected image and its one-based global exposure ordinal."""

    row: ImageRecord
    ordinal: int


def _worker_start() -> None:
    torch.set_num_threads(1)


def _load_chunk(
    presentations: tuple[Presentation, ...], root: Path, seed: int, training: bool,
) -> tuple[Tensor, Tensor]:
    transform = image_transforms()[0 if training else 1]
    decoded = tuple(
        ManifestDataset(root, (item.row,), transform, seed, item.ordinal)[0]
        for item in presentations
    )
    return torch.stack(tuple(item[0] for item in decoded)), torch.tensor(tuple(item[1] for item in decoded))


class SelectedImageLoader:
    """Load only selected batches; workers hold no model or scheduling authority."""

    def __init__(self, root: Path, seed: int, workers: int) -> None:
        self.root, self.seed, self.workers = root, seed, workers
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_start,
        ) if workers else None

    def load(self, items: tuple[Presentation, ...], training: bool) -> tuple[Tensor, Tensor]:
        """Decode one chosen batch with no lookahead across optimizer feedback."""
        if not items or (training and any(item.row.split != "train" for item in items)):
            raise ValueError("training loaders accept only explicit training-image presentations")
        function = partial(_load_chunk, root=self.root, seed=self.seed, training=training)
        if self.executor is None:
            return function(items)
        chunks = tuple(items[offset::self.workers] for offset in range(min(self.workers, len(items))))
        results = tuple(self.executor.map(function, chunks))
        # Restore original selection order after the strided worker partition.
        positions = tuple(index for offset in range(len(chunks)) for index in range(offset, len(items), self.workers))
        inverse = torch.tensor(sorted(range(len(positions)), key=positions.__getitem__))
        return tuple(torch.cat(tuple(result[column] for result in results))[inverse] for column in (0, 1))

    def close(self) -> None:
        """Release the bounded worker pool before another accelerator job starts."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    def __enter__(self) -> SelectedImageLoader:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()
