"""Immutable protocol contracts for the TRACE log-t VAMP experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from apm.continual.artifacts import record_sha256, require_sha256


PROTOCOL_FORMAT = "trace-logt-vamp-run-v2"
DATASET_FORMAT = "trace-llm-cl-benchmark-500-v1"
DATASET_NAME = "LLM-CL-Benchmark_500"
DATASET_ARCHIVE_SHA256 = (
    "956caf12b59add0c7d961cf8ecbad0307e1abca8db7de8873c37d92dd709e9c2"
)
TREE_LORA_REVISION = "1c7260c42b34e1961283797c742f08b9c3842501"
CORE_SPACE_REVISION = "c8c0f69dd4587eaefce61414dc6ac26ee5ad31f0"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
MODEL_SOURCE_ID = "alpindale/Llama-3.2-1B-Instruct"
MODEL_SOURCE_REVISION = "f92201d8185818a9d079b3b52efdab4b68bdd17f"
# Each source blob matches the corresponding blob in MODEL_ID at MODEL_REVISION.
# The SHA-256 values authenticate the downloaded bytes independently of Git/Xet.
MODEL_FILE_IDENTITIES: tuple[tuple[str, int, str, str], ...] = (
    (
        "config.json",
        877,
        "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb",
        "3e3aaf51a035cb5092d9f6827a0dc074657ba88c",
    ),
    (
        "generation_config.json",
        189,
        "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36",
        "75ae08310d6d23df373ee2644b497192b3cce6d8",
    ),
    (
        "model.safetensors",
        2_471_645_608,
        "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
        "5daf23edd80cedb879a4a18e6049a19b2417a3f7",
    ),
    (
        "special_tokens_map.json",
        296,
        "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
        "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
    ),
    (
        "tokenizer.json",
        9_085_657,
        "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
    ),
    (
        "tokenizer_config.json",
        54_528,
        "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
        "4ff488a165e900e5129cda7c20ab32d568d2a475",
    ),
)
SEED = 1234
ARRIVALS_PER_TASK = 5
EXAMPLES_PER_ARRIVAL = 100
ARRIVAL_COUNT = 40
TRAIN_EXAMPLE_COUNT = 4_000
TRAIN_PRESENTATION_COUNT = 20_000
LEVEL_CAPACITY = 2
LORA_SCALE = 4.0
EVALUATION_ARRIVALS = (5, 10, 15, 20, 25, 30, 35, 40)
CORE_SCALE_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
REPAIR_FRACTION_GRID = (0.0, 0.02, 0.05, 0.10, 0.25)

MetricName: TypeAlias = Literal["accuracy", "rouge_l", "similarity", "sari"]
MergeMethod: TypeAlias = Literal["svd_mean_r8", "core_tsv_r8"]
RouterName: TypeAlias = Literal[
    "prompt_nll", "frozen_prompt_centroid", "task_aware", "answer_oracle"
]
JobState: TypeAlias = Literal[
    "PENDING", "RUNNING", "CHECKPOINTED", "COMPLETE", "FAILED", "PAUSED"
]


@dataclass(frozen=True, slots=True)
class TraceTask:
    """One task in the fixed TRACE curriculum."""

    name: str
    epochs: int
    metric: MetricName


TASKS: tuple[TraceTask, ...] = (
    TraceTask("C-STANCE", 5, "accuracy"),
    TraceTask("FOMC", 3, "accuracy"),
    TraceTask("MeetingBank", 7, "rouge_l"),
    TraceTask("Py150", 5, "similarity"),
    TraceTask("ScienceQA", 3, "accuracy"),
    TraceTask("NumGLUE-cm", 5, "accuracy"),
    TraceTask("NumGLUE-ds", 5, "accuracy"),
    TraceTask("20Minuten", 7, "sari"),
)
TASK_NAMES = tuple(task.name for task in TASKS)
TASK_BY_NAME = {task.name: task for task in TASKS}

PRIMARY_CONDITIONS = (
    "seq_lora_reference",
    "seq_lora_40",
    "joint_iid_lora",
    "taskwise_lora",
    "vamp_svd_r8_repair000",
    "vamp_svd_r8_repair005",
    "vamp_core_tsv_r8_scale03_repair000",
    "vamp_core_tsv_r8_scale03_repair005",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Exact leaf or repair optimizer and LoRA configuration."""

    rank: int = 8
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024
    max_answer_length: int = 512

    def __post_init__(self) -> None:
        if (
            self.rank <= 0
            or self.alpha <= 0
            or not 0.0 <= self.dropout < 1.0
            or self.learning_rate <= 0.0
            or self.micro_batch_size != 1
            or self.gradient_accumulation_steps != 8
            or self.target_modules != ("q_proj", "v_proj")
        ):
            raise ValueError("training configuration violates the TRACE contract")

    @property
    def scale(self) -> float:
        """Return the PEFT LoRA multiplier alpha divided by rank."""
        return self.alpha / self.rank

    def as_record(self) -> dict[str, object]:
        """Return the canonical training configuration."""
        return {
            "alpha": self.alpha,
            "betas": [self.beta1, self.beta2],
            "bias": "none",
            "dropout": self.dropout,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_answer_length": self.max_answer_length,
            "max_prompt_length": self.max_prompt_length,
            "micro_batch_size": self.micro_batch_size,
            "optimizer": "Adam",
            "padding_side": "left",
            "rank": self.rank,
            "scheduler": "constant",
            "target_modules": list(self.target_modules),
            "truncation_side": "left",
            "warmup_steps": 0,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True, slots=True)
class MergePolicy:
    """A complete immutable identity for one derived hierarchy policy."""

    method: MergeMethod
    output_rank: int = 8
    parent_alpha: int = 32
    core_scale: float | None = None
    repair_fraction: float = 0.0
    repair_learning_rate: float = 5.0e-5
    algorithm_version: str = "trace-merge-v1"

    def __post_init__(self) -> None:
        if (
            self.output_rank <= 0
            or self.parent_alpha <= 0
            or self.repair_learning_rate <= 0.0
            or not self.algorithm_version
        ):
            raise ValueError("merge rank, alpha, optimizer, and version must be valid")
        if self.method == "core_tsv_r8":
            if self.core_scale not in CORE_SCALE_GRID:
                raise ValueError("Core TSV scale is outside the registered grid")
        elif self.core_scale is not None:
            raise ValueError("SVD mean does not accept a Core scale")
        if self.repair_fraction not in REPAIR_FRACTION_GRID:
            raise ValueError("repair fraction is outside the registered grid")

    @property
    def parent_scale(self) -> float:
        """Return the scale encoded by the saved parent adapter."""
        return self.parent_alpha / self.output_rank

    @property
    def policy_hash(self) -> str:
        """Return the content identity of the complete derivation policy."""
        return record_sha256(self.as_record())

    @property
    def merge_config_hash(self) -> str:
        """Return the reusable parameter-merge identity, excluding replay repair."""
        return record_sha256(self.merge_record())

    @property
    def repair_config_hash(self) -> str:
        """Return the repair-only identity layered on a cached parameter merge."""
        return record_sha256(
            {
                "learning_rate": self.repair_learning_rate,
                "repair_fraction": self.repair_fraction,
                "reservoir_rule": "lowest-sha256-priority-v1",
            }
        )

    def merge_record(self) -> dict[str, object]:
        """Return the replay-independent parameter-space merge configuration."""
        return {
            "algorithm_version": self.algorithm_version,
            "core_scale": self.core_scale,
            "isotropize": False,
            "method": self.method,
            "output_rank": self.output_rank,
            "parent_alpha": self.parent_alpha,
        }

    def as_record(self) -> dict[str, object]:
        """Return the canonical policy record."""
        return {
            "algorithm_version": self.algorithm_version,
            "core_scale": self.core_scale,
            "isotropize": False,
            "method": self.method,
            "output_rank": self.output_rank,
            "parent_alpha": self.parent_alpha,
            "repair_fraction": self.repair_fraction,
            "repair_learning_rate": self.repair_learning_rate,
            "reservoir_rule": "lowest-sha256-priority-v1",
        }


@dataclass(frozen=True, slots=True)
class RunContract:
    """Content-addressed identity of one prepared TRACE experiment."""

    dataset_manifest_sha256: str
    model_revision: str
    tokenizer_revision: str
    code_revision: str
    dependency_environment_sha256: str
    model_manifest_sha256: str
    source_archive_sha256: str = DATASET_ARCHIVE_SHA256
    seed: int = SEED
    model_id: str = MODEL_ID
    format: str = PROTOCOL_FORMAT

    def __post_init__(self) -> None:
        for value, label in (
            (self.dataset_manifest_sha256, "dataset manifest"),
            (self.dependency_environment_sha256, "dependency environment"),
            (self.model_manifest_sha256, "model manifest"),
            (self.source_archive_sha256, "dataset archive"),
        ):
            require_sha256(value, label)
        if not all(
            value.strip()
            for value in (
                self.model_revision,
                self.tokenizer_revision,
                self.code_revision,
            )
        ):
            raise ValueError("resolved model, tokenizer, and code revisions are required")
        if self.seed != SEED or self.model_id != MODEL_ID or self.format != PROTOCOL_FORMAT:
            raise ValueError("run identity differs from the registered TRACE protocol")

    @property
    def run_contract_hash(self) -> str:
        """Return the directory identity for this complete run contract."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical run manifest."""
        return {
            "code_revision": self.code_revision,
            "core_space_revision": CORE_SPACE_REVISION,
            "dataset": DATASET_NAME,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dependency_environment_sha256": self.dependency_environment_sha256,
            "format": self.format,
            "model_id": self.model_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_revision": self.model_revision,
            "seed": self.seed,
            "source_archive_sha256": self.source_archive_sha256,
            "tokenizer_revision": self.tokenizer_revision,
            "training": TrainingConfig().as_record(),
            "tree_lora_revision": TREE_LORA_REVISION,
        }


def default_merge_policies() -> tuple[MergePolicy, ...]:
    """Return the four preregistered VAMP policies in report order."""
    return (
        MergePolicy("svd_mean_r8", repair_fraction=0.0),
        MergePolicy("svd_mean_r8", repair_fraction=0.05),
        MergePolicy("core_tsv_r8", core_scale=0.3, repair_fraction=0.0),
        MergePolicy("core_tsv_r8", core_scale=0.3, repair_fraction=0.05),
    )


def task_for_arrival(arrival: int) -> TraceTask:
    """Return the fixed task associated with a one-based arrival."""
    if type(arrival) is not int or not 1 <= arrival <= ARRIVAL_COUNT:
        raise ValueError("arrival must be in the inclusive range 1..40")
    return TASKS[(arrival - 1) // ARRIVALS_PER_TASK]


def stable_seed(*parts: object, seed: int = SEED) -> int:
    """Derive a scheduling-independent nonnegative 63-bit seed."""
    return int(record_sha256({"parts": list(parts), "seed": seed})[:16], 16) & (2**63 - 1)


def default_store_root() -> Path:
    """Return the required persistent RunPod store root."""
    return Path("/workspace/vamp-trace")


__all__ = [
    "ARRIVAL_COUNT",
    "ARRIVALS_PER_TASK",
    "CORE_SCALE_GRID",
    "CORE_SPACE_REVISION",
    "DATASET_ARCHIVE_SHA256",
    "DATASET_FORMAT",
    "DATASET_NAME",
    "EVALUATION_ARRIVALS",
    "EXAMPLES_PER_ARRIVAL",
    "JobState",
    "LEVEL_CAPACITY",
    "LORA_SCALE",
    "MODEL_FILE_IDENTITIES",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_SOURCE_ID",
    "MODEL_SOURCE_REVISION",
    "MergeMethod",
    "MergePolicy",
    "PRIMARY_CONDITIONS",
    "PROTOCOL_FORMAT",
    "REPAIR_FRACTION_GRID",
    "RouterName",
    "RunContract",
    "SEED",
    "TASKS",
    "TASK_BY_NAME",
    "TASK_NAMES",
    "TRAIN_EXAMPLE_COUNT",
    "TRAIN_PRESENTATION_COUNT",
    "TREE_LORA_REVISION",
    "TraceTask",
    "TrainingConfig",
    "default_merge_policies",
    "default_store_root",
    "stable_seed",
    "task_for_arrival",
]
