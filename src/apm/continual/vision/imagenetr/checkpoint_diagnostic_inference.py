"""Resumable, inference-only collection of authenticated checkpoint logits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from apm.continual.artifacts import file_sha256, publish_immutable_bytes, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.checkpoint_diagnostic_math import score_rows
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.srt_data import Presentation, SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """Identity of one checkpoint/population forward pass with fixed chunk boundaries."""

    protocol_hash: str
    endpoint: str
    split: str
    model_sha256: str
    image_ids_hash: str
    batch_size: int = 64
    chunk_images: int = 1024
    classes: int = 200

    def __post_init__(self) -> None:
        if self.split not in {"test", "train"} or self.batch_size < 1 or self.chunk_images % self.batch_size or self.chunk_images < 1:
            raise ValueError("invalid fixed inference population/batch boundaries")


def publish_logit_table(path: Path, rows: tuple[dict[str, object], ...], logits: np.ndarray) -> str:
    """Serialize exact float32 logits as fixed-size vectors without Python float expansion."""
    if len(rows) != len(logits) or logits.ndim != 2:
        raise ValueError("logit table identities and matrix differ")
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(logits, dtype=np.float32).ravel()), logits.shape[1])
    table = pa.Table.from_pylist(list(rows)).append_column("logits", vectors)
    stream = BytesIO()
    pq.write_table(table, stream, compression="zstd")
    payload = stream.getvalue()
    publish_immutable_bytes(path, payload)
    return sha256(payload).hexdigest()


def load_collection(root: Path) -> tuple[tuple[dict[str, object], ...], np.ndarray, dict[str, object]]:
    """Authenticate all committed chunks before returning ordered identities and logits."""
    job = read_sealed(root / "job.json", "imagenetr50-checkpoint-collection-job-v1")
    result = read_sealed(root / "result.json", "imagenetr50-checkpoint-collection-v1")
    if result["job_hash"] != job["content_hash"] or result["optimizer_steps"] != 0 or not result["model_unchanged"]:
        raise ValueError("collection result changed or claims model updates")
    tables = ()
    for chunk in result["chunks"]:
        folder = root / "chunks" / f"{chunk['offset']:06d}"
        if read_sealed(folder / "result.json") != chunk or file_sha256(folder / "logits.parquet") != chunk["logits_sha256"]:
            raise ValueError("committed logit chunk changed")
        if chunk["job_hash"] != job["content_hash"] or chunk["offset"] != sum(table.num_rows for table in tables):
            raise ValueError("logit chunk ordering/identity changed")
        table = pq.read_table(folder / "logits.parquet")
        if table.num_rows != chunk["examples"]:
            raise ValueError("logit chunk population changed")
        tables += (table,)
    table = pa.concat_tables(tables).combine_chunks()
    rows = tuple(table.drop(["logits"]).to_pylist())
    if (len(rows) != result["model_forward_images"] or len({row["image_id"] for row in rows}) != len(rows)
            or record_sha256(tuple(row["image_id"] for row in rows)) != job["spec"]["image_ids_hash"]):
        raise ValueError("collection population no longer matches its job")
    logits = table["logits"].chunk(0).values.to_numpy().reshape(len(rows), job["spec"]["classes"])
    return rows, logits, result


def model_tensor_hashes(model: torch.nn.Module) -> dict[str, str]:
    """Fingerprint all parameters and buffers without retaining another model copy."""
    return {name: sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def collect_logits(
    root: Path, spec: CollectionSpec, rows: tuple[ImageRecord, ...], loader: SelectedImageLoader,
    device: torch.device, model_factory: Callable[[], torch.nn.Module],
    progress: Callable[[str, int, int, float], None] | None = None, stop_after_chunks: int | None = None,
) -> tuple[dict[str, object], int]:
    """Collect clean-view logits with zero optimizer steps and immutable chunk resume."""
    if not rows or any(row.split != spec.split for row in rows) or record_sha256(tuple(row.image_id for row in rows)) != spec.image_ids_hash:
        raise ValueError("inference population differs from the fixed job")
    job = sealed_record({"schema_version": "imagenetr50-checkpoint-collection-job-v1", "spec": asdict(spec)})
    publish_immutable_json(root / "job.json", job)
    if (root / "result.json").is_file():
        _, _, result = load_collection(root)
        return result, 0
    chunks, model, original, forwarded = (), None, {}, 0
    started = time.monotonic()
    with tqdm(total=len(rows), desc=f"{spec.endpoint} {spec.split}", unit="image", mininterval=5) as bar:
        try:
            for offset in range(0, len(rows), spec.chunk_images):
                selected = rows[offset:offset + spec.chunk_images]
                folder = root / "chunks" / f"{offset:06d}"
                if (folder / "result.json").is_file():
                    chunk = read_sealed(folder / "result.json")
                    if (chunk["job_hash"] != job["content_hash"] or chunk["offset"] != offset or chunk["examples"] != len(selected)
                            or file_sha256(folder / "logits.parquet") != chunk["logits_sha256"]):
                        raise ValueError("resumed inference chunk changed")
                    if pq.read_table(folder / "logits.parquet", columns=["image_id"])["image_id"].to_pylist() != [row.image_id for row in selected]:
                        raise ValueError("resumed inference identities differ")
                else:
                    if model is None:
                        model = model_factory().to(device)
                        original = model_tensor_hashes(model)
                        model.eval().requires_grad_(False)
                    tick, outputs = time.monotonic(), ()
                    with torch.inference_mode():
                        for start in range(0, len(selected), spec.batch_size):
                            batch = selected[start:start + spec.batch_size]
                            images, labels = loader.load(tuple(Presentation(row, 0) for row in batch), False)
                            if labels.tolist() != [row.remapped_class_index for row in batch]:
                                raise ValueError("loader labels differ from the image manifest")
                            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                                logits = model(images.to(device)).float().cpu().numpy()
                            if logits.shape != (len(batch), spec.classes):
                                raise ValueError("model returned a different class population")
                            outputs += (logits,)
                    logits = np.concatenate(outputs)
                    scores = score_rows(logits, np.array([row.remapped_class_index for row in selected]))
                    metadata = tuple({"image_id": row.image_id, "label": row.remapped_class_index, "task": row.task_index + 1, **score}
                                     for row, score in zip(selected, scores, strict=True))
                    digest = publish_logit_table(folder / "logits.parquet", metadata, logits)
                    chunk = sealed_record({"schema_version": "imagenetr50-checkpoint-logit-chunk-v1", "job_hash": job["content_hash"],
                                           "offset": offset, "examples": len(selected), "logits_sha256": digest,
                                           "wall_seconds": time.monotonic() - tick})
                    publish_immutable_json(folder / "result.json", chunk)
                    forwarded += len(selected)
                chunks += (chunk,)
                bar.update(len(selected))
                if progress is not None:
                    progress(f"{spec.endpoint}/{spec.split}", offset + len(selected), len(rows), time.monotonic() - started)
                if stop_after_chunks is not None and len(chunks) >= stop_after_chunks:
                    raise InterruptedError("intentional diagnostic chunk-boundary interruption")
            if model is not None:
                if original != model_tensor_hashes(model):
                    raise ValueError("inference changed checkpoint parameters")
            result = sealed_record({"schema_version": "imagenetr50-checkpoint-collection-v1", "job_hash": job["content_hash"],
                                    "chunks": chunks, "model_forward_images": len(rows), "optimizer_steps": 0, "model_unchanged": True,
                                    "wall_seconds": sum(chunk["wall_seconds"] for chunk in chunks)})
            publish_immutable_json(root / "result.json", result)
            return load_collection(root)[2], forwarded
        finally:
            del model
