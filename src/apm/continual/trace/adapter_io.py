"""PEFT-compatible immutable adapter tensor input and output."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from apm.continual.artifacts import publish_immutable_json
from apm.continual.trace.protocol import TrainingConfig


ADAPTER_FILENAME = "adapter.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def save_adapter(
    model: torch.nn.Module,
    directory: str | Path,
    config: TrainingConfig = TrainingConfig(),
) -> Path:
    """Save adapter-name-free PEFT tensor keys and a canonical LoRA config."""
    try:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("TRACE adapter persistence requires peft and safetensors") from error
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(model)
    tensors = {
        str(name): value.detach().to(device="cpu").contiguous()
        for name, value in state.items()
        if isinstance(value, Tensor)
    }
    if not tensors:
        raise ValueError("PEFT model exposed no adapter tensors")
    save_file(tensors, target / ADAPTER_FILENAME, metadata={"format": "pt"})
    publish_immutable_json(
        target / ADAPTER_CONFIG_FILENAME,
        {
            "alpha_pattern": {},
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": config.alpha,
            "lora_dropout": config.dropout,
            "peft_type": "LORA",
            "r": config.rank,
            "rank_pattern": {},
            "target_modules": list(config.target_modules),
            "task_type": "CAUSAL_LM",
        },
    )
    return target / ADAPTER_FILENAME


def load_adapter_state(
    model: torch.nn.Module,
    adapter_path: str | Path,
) -> None:
    """Load canonical adapter tensors through PEFT's supported state API."""
    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("TRACE adapter loading requires peft and safetensors") from error
    state = load_file(Path(adapter_path), device="cpu")
    result = set_peft_model_state_dict(model, state)
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if unexpected:
        raise ValueError(f"adapter contains unexpected PEFT keys: {unexpected}")


def load_adapter_tensors(path: str | Path) -> Mapping[str, Tensor]:
    """Load adapter tensors without constructing a model, for merge jobs."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("TRACE adapter loading requires safetensors") from error
    return load_file(Path(path), device="cpu")


__all__ = [
    "ADAPTER_CONFIG_FILENAME",
    "ADAPTER_FILENAME",
    "load_adapter_state",
    "load_adapter_tensors",
    "save_adapter",
]
