"""RunPod-specific preflight checks without infrastructure side effects."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class RunPodPreflight:
    """Authenticated facts required before the primary two-worker run starts."""

    gpu_names: tuple[str, ...]
    store_root: Path
    workspace_is_mount: bool
    bf16_supported: bool
    network_volume_id: str

    def as_record(self) -> dict[str, object]:
        """Return a manifest-ready preflight record without any secret values."""
        return {
            "bf16_supported": self.bf16_supported,
            "format": "trace-runpod-preflight-v1",
            "gpu_names": list(self.gpu_names),
            "network_volume_id": self.network_volume_id,
            "runpod_pod_id_present": bool(os.environ.get("RUNPOD_POD_ID")),
            "store_root": str(self.store_root),
            "workspace_is_mount": self.workspace_is_mount,
        }


def require_primary_runpod(store_root: str | Path) -> RunPodPreflight:
    """Require 2x4090, BF16 CUDA, and an independently mounted `/workspace`."""
    root = Path(store_root)
    workspace = Path("/workspace")
    required_environment = ("RUNPOD_API_KEY", "RUNPOD_POD_ID", "RUNPOD_VOLUME_ID")
    if any(not os.environ.get(name) for name in required_environment):
        raise RuntimeError("primary TRACE requires RunPod Pod/API/volume environment identity")
    names = tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
    result = RunPodPreflight(
        gpu_names=names,
        store_root=root,
        workspace_is_mount=os.path.ismount(workspace),
        bf16_supported=(
            len(names) == 2
            and all(_bf16_supported(index) for index in range(len(names)))
        ),
        network_volume_id=str(os.environ["RUNPOD_VOLUME_ID"]),
    )
    try:
        root.relative_to(workspace)
    except ValueError as error:
        raise RuntimeError("primary TRACE state must live under /workspace") from error
    if not result.workspace_is_mount:
        raise RuntimeError("/workspace is not an independently mounted persistent volume")
    if len(names) != 2 or any("4090" not in name for name in names):
        raise RuntimeError(f"primary TRACE requires exactly two RTX 4090 GPUs, found {names}")
    if not result.bf16_supported:
        raise RuntimeError("both TRACE GPUs must support BF16")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".trace-volume-write-probe"
    probe.write_bytes(b"persistent-volume-preflight\n")
    probe.unlink()
    return result


def _bf16_supported(index: int) -> bool:
    with torch.cuda.device(index):
        return bool(torch.cuda.is_bf16_supported())


__all__ = ["RunPodPreflight", "require_primary_runpod"]
