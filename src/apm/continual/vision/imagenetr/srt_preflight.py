"""Opt-in real-data correctness checks before the finite SRT calibration matrix."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from apm.continual.artifacts import publish_immutable_json
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone, require_trainable_boundary
from apm.continual.vision.imagenetr.srt_config import SRTConfig
from apm.continual.vision.imagenetr.srt_data import Presentation, SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_event_chunk, read_sealed, sealed_record
from apm.continual.vision.imagenetr.srt_training import optimization_step, optimizer_for, run_training_job


def real_srt_preflight(
    root: Path, protocol_hash: str, checkpoint: Path, config: SRTConfig,
    fitting: tuple[ImageRecord, ...], validation: tuple[ImageRecord, ...],
    loader: SelectedImageLoader, device: torch.device,
) -> dict[str, object]:
    """Check zero-adapter parity, BF16 batch 64, eight tasks, and exact interruption recovery."""
    if (root / "result.json").is_file():
        return read_sealed(root / "result.json", "imagenetr50-srt-preflight-v1")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SRT preflight requires the local BF16 CUDA accelerator")
    images, labels = loader.load(tuple(Presentation(row, 1) for row in fitting if row.task_index == 0)[:64], True)
    backbone = create_pinned_backbone(checkpoint).to(device).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = backbone.forward_head(backbone.forward_features(images.to(device)), pre_logits=True)
    model = AdapterVisionModel(backbone, (0, 1, 2, 3), initialization_seed=config.seed).to(device).eval()
    require_trainable_boundary(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model.features(images.to(device))
    parity = float((expected.float() - actual.float()).abs().max().item())
    if parity > 0.002:
        raise ValueError("zero-LoRA parity failed in the real SRT model")
    optimizer = optimizer_for(model, config)
    torch.cuda.reset_peak_memory_stats(device)
    _, loss_sum, _, norm = optimization_step(model, optimizer, images, labels, device)
    peak = torch.cuda.max_memory_allocated(device)
    del model, backbone, optimizer, images, labels, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    small_fit = tuple(row for class_id in range(32)
                      for row in tuple(row for row in fitting if row.remapped_class_index == class_id)[:2])
    small_validation = tuple(next(row for row in validation if row.remapped_class_index == class_id) for class_id in range(32))
    policy = next(policy for policy in config.policies if policy.name == "standard_rho50_unit8")
    common = dict(protocol_hash=protocol_hash, checkpoint_path=checkpoint, config=config, policy=policy,
                  capacity=1024, training_rows=small_fit, evaluation_rows=small_validation,
                  loader=loader, device=device, target_stage=8)
    continuous = run_training_job(root / "continuous", method="srt", **common)
    interrupted_root = root / "interrupted"
    if not (interrupted_root / "checkpoint.pt").is_file():
        stopped = run_training_job(interrupted_root, method="srt", stop_after_steps=5, **common)
        if not stopped.get("stopped_for_resume_test"):
            raise ValueError("the real interruption fixture did not stop at its requested boundary")
    recovered = run_training_job(interrupted_root, method="srt", **common)
    control = run_training_job(root / "uniform", method="uniform", paired_root=root / "continuous", **common)
    for expected_row, restored in zip(continuous["rows"], recovered["rows"], strict=True):
        if (expected_row["model_sha256"] != restored["model_sha256"]
                or expected_row["predictions_sha256"] != restored["predictions_sha256"]):
            raise ValueError("real interrupted SRT differs from uninterrupted weights or predictions")
    event_streams = tuple(
        tuple(event for row in result["rows"] for chunk in row["trace_chunks"]
              for event in read_event_chunk(job_root / chunk["path"] / "events.parquet"))
        for job_root, result in ((root / "continuous", continuous), (interrupted_root, recovered))
    )
    if event_streams[0] != event_streams[1]:
        raise ValueError("real interrupted SRT changed a sample review event")
    reuse = run_training_job(root / "continuous", method="srt", **common)
    if reuse["invocation_optimizer_steps"] != 0:
        raise ValueError("completed SRT preflight did not reuse its models")
    result = sealed_record({
        "schema_version": "imagenetr50-srt-preflight-v1", "protocol_hash": protocol_hash,
        "zero_lora_max_absolute_error": parity, "batch64_loss": loss_sum / 64,
        "gradient_norm": norm, "peak_vram_bytes": peak, "smoke_tasks": 8,
        "identical_resume_weights_predictions_events": True, "zero_step_reuse": True,
        "srt_steps": continuous["optimizer_steps"], "uniform_steps": control["optimizer_steps"],
        "srt_presentations": continuous["image_presentations"], "uniform_presentations": control["image_presentations"],
    })
    publish_immutable_json(root / "result.json", result)
    return result
