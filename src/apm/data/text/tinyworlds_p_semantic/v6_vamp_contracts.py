"""Frozen contracts for the first semantic-v6 VAMP experiment."""

from __future__ import annotations

from dataclasses import dataclass

from apm.data.text.tinyworlds_p_semantic.contracts import record_sha256
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_PARENT_CATALOG_SHA256,
)
from apm.lm.lora import LoraConfig
from apm.lm.training import LmTrainConfig
from apm.memory.address_refinement import EbtConfig
from apm.memory.content_addressing import HopfieldConfig


V6_VAMP_EXPERIMENT_ID = "tinyworlds-p-semantic-v6-vamp-chain-v1"
V6_VAMP_CONFIG_VERSION = "tinyworlds-p-semantic-v6-vamp-config-v1"
V6_CANONICAL_PARTITION_SHA256 = (
    "3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa"
)
V6_CANONICAL_SAMPLE_REPORT_SHA256 = (
    "b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579"
)
V6_VAMP_TASK_ORDER = ("A", "B", "C", "D", "E")
V6_VAMP_WORLD_COORDINATES = (
    ("A", 2, 4),
    ("B", 7, 4),
    ("C", 7, 6),
    ("D", 2, 6),
    ("E", 3, 2),
)


@dataclass(frozen=True, slots=True)
class V6VampExperimentPreset:
    """Every behavior-changing choice for the one-order exploratory study."""

    version: str = V6_VAMP_CONFIG_VERSION
    experiment_id: str = V6_VAMP_EXPERIMENT_ID
    partition_sha256: str = V6_CANONICAL_PARTITION_SHA256
    catalog_sha256: str = V6_PARENT_CATALOG_SHA256
    sample_report_sha256: str = V6_CANONICAL_SAMPLE_REPORT_SHA256
    task_order: tuple[str, ...] = V6_VAMP_TASK_ORDER
    world_coordinates: tuple[tuple[str, int, int], ...] = (
        V6_VAMP_WORLD_COORDINATES
    )
    seed: int = 0
    random_router_seed: int = 0
    context_length: int = 256
    batch_size: int = 32
    lora_rank: int = 8
    lora_alpha: float = 8.0
    adapter_steps_per_task: int = 2_000
    adapter_learning_rate: float = 1e-3
    adapter_weight_decay: float = 0.01
    adapter_gradient_clip_norm: float = 1.0
    root_probe_count: int = 128
    parent_probe_count: int = 128
    content_key_probe_count: int = 128
    evaluation_examples_per_world: int = 128
    prefix_lengths: tuple[int, ...] = (16, 32, 64, 128)
    suffix_length: int = 128
    primary_prefix_length: int = 64
    max_nodes: int = 6
    max_edges: int = 5
    evaluation_microbatch_size: int = 8
    hopfield_beta: float = 10.0
    hopfield_top_k: int = 4
    ebt_steps: int = 20
    ebt_learning_rate: float = 0.1
    ebt_tau: float = 1.0
    ebt_entropy_penalty: float = 0.01
    timing_warm_repetitions: int = 5
    sample_new_tokens: int = 32
    specificity_replicates: int = 10_000
    allocator_peak_limit_bytes: int = 12 * 1024**3

    def __post_init__(self) -> None:
        if self.as_record() != _canonical_record():
            raise ValueError("semantic-v6 VAMP experiment choices are frozen")

    @property
    def lora_config(self) -> LoraConfig:
        """Return the all-projection rank-eight adapter contract."""
        return LoraConfig(rank=self.lora_rank, alpha=self.lora_alpha)

    @property
    def train_config(self) -> LmTrainConfig:
        """Return the fixed per-world update budget and optimizer settings."""
        return LmTrainConfig(
            learning_rate=self.adapter_learning_rate,
            steps=self.adapter_steps_per_task,
            batch_size=self.batch_size,
            weight_decay=self.adapter_weight_decay,
            gradient_clip_norm=self.adapter_gradient_clip_norm,
        )

    @property
    def hopfield_config(self) -> HopfieldConfig:
        """Return the frozen content-addressing settings."""
        return HopfieldConfig(beta=self.hopfield_beta, top_k=self.hopfield_top_k)

    @property
    def ebt_config(self) -> EbtConfig:
        """Return the frozen energy-based test-time refinement settings."""
        return EbtConfig(
            steps=self.ebt_steps,
            learning_rate=self.ebt_learning_rate,
            tau=self.ebt_tau,
            entropy_penalty=self.ebt_entropy_penalty,
        )

    @property
    def config_sha256(self) -> str:
        """Return the content identity of the complete experiment preset."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical experiment manifest payload."""
        return {
            name: (
                [list(item) for item in value]
                if name == "world_coordinates"
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }


def _canonical_record() -> dict[str, object]:
    return {
        "adapter_gradient_clip_norm": 1.0,
        "adapter_learning_rate": 1e-3,
        "adapter_steps_per_task": 2_000,
        "adapter_weight_decay": 0.01,
        "allocator_peak_limit_bytes": 12 * 1024**3,
        "batch_size": 32,
        "catalog_sha256": V6_PARENT_CATALOG_SHA256,
        "content_key_probe_count": 128,
        "context_length": 256,
        "evaluation_examples_per_world": 128,
        "evaluation_microbatch_size": 8,
        "hopfield_beta": 10.0,
        "hopfield_top_k": 4,
        "ebt_steps": 20,
        "ebt_learning_rate": 0.1,
        "ebt_tau": 1.0,
        "ebt_entropy_penalty": 0.01,
        "experiment_id": V6_VAMP_EXPERIMENT_ID,
        "lora_alpha": 8.0,
        "lora_rank": 8,
        "max_edges": 5,
        "max_nodes": 6,
        "parent_probe_count": 128,
        "partition_sha256": V6_CANONICAL_PARTITION_SHA256,
        "prefix_lengths": [16, 32, 64, 128],
        "primary_prefix_length": 64,
        "random_router_seed": 0,
        "root_probe_count": 128,
        "sample_new_tokens": 32,
        "sample_report_sha256": V6_CANONICAL_SAMPLE_REPORT_SHA256,
        "seed": 0,
        "specificity_replicates": 10_000,
        "suffix_length": 128,
        "task_order": list(V6_VAMP_TASK_ORDER),
        "timing_warm_repetitions": 5,
        "version": V6_VAMP_CONFIG_VERSION,
        "world_coordinates": [list(item) for item in V6_VAMP_WORLD_COORDINATES],
    }


V6_VAMP_EXPERIMENT_PRESET = V6VampExperimentPreset()


__all__ = [
    "V6_CANONICAL_PARTITION_SHA256",
    "V6_CANONICAL_SAMPLE_REPORT_SHA256",
    "V6_VAMP_CONFIG_VERSION",
    "V6_VAMP_EXPERIMENT_ID",
    "V6_VAMP_EXPERIMENT_PRESET",
    "V6_VAMP_TASK_ORDER",
    "V6_VAMP_WORLD_COORDINATES",
    "V6VampExperimentPreset",
]
