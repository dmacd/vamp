"""Immutable contracts for the disjoint TinyWorlds nouns-v2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Literal, TypeAlias


BENCHMARK_ID = "tinyworlds-nouns-v2"
SCHEMA_VERSION = 2
MANIFEST_FORMAT = "tinyworlds-nouns-manifest-v2"
PARTITION_FORMAT = "tinyworlds-nouns-partition-v2"
INDEX_FORMAT = "tinyworlds-nouns-story-index-v2"
EXCLUSION_FORMAT = "tinyworlds-nouns-multitask-exclusion-v2"
AUDIT_FORMAT = "tinyworlds-nouns-disjoint-audit-v2"
PRESET_FORMAT = "tinyworlds-nouns-experiment-preset-v2"
RUN_MANIFEST_FORMAT = "tinyworlds-nouns-run-v2"
WHOLE_STORY_FORMAT = "tinyworlds-nouns-whole-story-nll-v2"
HALF_STORY_FORMAT = "tinyworlds-nouns-half-story-generation-v2"
STAGEWISE_FORMAT = "tinyworlds-nouns-stagewise-cl-v2"
JUDGE_REQUEST_FORMAT = "tinyworlds-nouns-judge-request-v2"
JUDGE_FORMAT = "tinyworlds-nouns-judge-result-v2"
REPORT_FORMAT = "tinyworlds-nouns-stagewise-report-v2"
BASE_TRAINING_FORMAT = "tinyworlds-nouns-base-training-v2"
BASE_SELECTION_FORMAT = "tinyworlds-nouns-selected-base-v2"
VAMP_STAGE_FORMAT = "tinyworlds-nouns-vamp-stage-v2"
GPU_PREFLIGHT_FORMAT = "tinyworlds-nouns-gpu-preflight-v2"

DATA_ROOT = Path("data/tinyworlds-nouns-v2")
CHECKPOINT_ROOT = Path("checkpoints/tinyworlds-nouns-v2")
RESULT_ROOT = Path("results/language_cl/tinyworlds-nouns-v2")
PARENT_DATA_ROOT = Path("data/tinyworlds-nouns-v1")
DEFAULT_TOKENIZER_PATH = Path(
    "checkpoints/tinystories-8m/tokenizer/tokenizer.json"
)

PARENT_PARTITION_SHA256 = (
    "04ca2acf85f9505f0b7568b1696fbf290a8d2cbf78387dcfd6e815258fcc28b8"
)
PARENT_BREAKDOWN_SHA256 = (
    "df60e7d00e5887f97c3e867c68a214333190595c15d1e0d39999b653d0eeed35"
)
PARENT_DECISIONS_SHA256 = (
    "96f5c41cf6acf7ba4e5acd8bdedcc0b7bf5cbb254786cc9b1481ac3554efb325"
)
PARENT_DECISIONS_CORE_SHA256 = (
    "269e34f92db4b3bd1a1cab36929ecb517491ae026006a344d83afdcb12f2a906"
)
PARENT_STORY_COUNT = 2_745_124
TRAIN_UNIQUE_STORY_COUNT = 2_717_494
VALIDATION_UNIQUE_STORY_COUNT = 27_630
BASE_UNIVERSE_STORY_COUNT = 2_210_934
PURE_TASK_TRAIN_STORY_COUNT = 429_199
PURE_TASK_VALIDATION_STORY_COUNT = 4_440
STAGEWISE_CASE_COUNT = 72_256
EXCLUDED_TRAIN_STORY_COUNT = 77_361
EXCLUDED_VALIDATION_STORY_COUNT = 776
MINIMUM_TASK_TRAIN_STORIES = 256
MINIMUM_TASK_VALIDATION_STORIES = 64
PROBE_STORY_COUNT = 36
BASE_VALIDATION_BUCKET_COUNT = 50
MODEL_POSITION_LIMIT = 2_048

TASK_IDS = (
    "mouse",
    "rabbit",
    "boat",
    "brother",
    "parent",
    "duck",
    "sister",
    "pet",
    "bicycle",
    "grandma",
    "lion",
    "fairy",
    "train",
    "cow",
    "wheel",
    "monkey",
    "princess",
    "plane",
    "elephant",
    "neighbor",
    "dragon",
    "queen",
    "horse",
    "bus",
)

EXPECTED_PURE_COUNTS = (
    ("mouse", 42_511, 404),
    ("rabbit", 40_241, 413),
    ("boat", 34_428, 343),
    ("brother", 33_318, 329),
    ("parent", 28_570, 287),
    ("duck", 25_386, 272),
    ("sister", 24_675, 277),
    ("pet", 24_230, 262),
    ("bicycle", 17_860, 172),
    ("grandma", 17_767, 169),
    ("lion", 14_889, 160),
    ("fairy", 14_624, 169),
    ("train", 13_091, 141),
    ("cow", 12_580, 117),
    ("wheel", 12_121, 138),
    ("monkey", 9_677, 115),
    ("princess", 9_615, 82),
    ("plane", 8_778, 94),
    ("elephant", 8_271, 97),
    ("neighbor", 8_016, 88),
    ("dragon", 7_918, 84),
    ("queen", 7_281, 81),
    ("horse", 7_175, 79),
    ("bus", 6_177, 67),
)

Condition: TypeAlias = Literal[
    "base",
    "oracle",
    "vamp_exhaustive",
    "vamp_hopfield",
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
]
CONDITIONS: tuple[Condition, ...] = (
    "base",
    "oracle",
    "vamp_exhaustive",
    "vamp_hopfield",
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*\Z")
_WORD = re.compile(r"[^\W_]+\Z", flags=re.UNICODE)


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value canonically with one trailing newline."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def record_sha256(value: object) -> str:
    """Return the SHA-256 of one canonical JSON value."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: str, label: str) -> None:
    """Require a lowercase SHA-256 value."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def require_identifier(value: str, label: str) -> None:
    """Require a portable lowercase identifier."""
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable lowercase identifier")


def _require_forms(forms: tuple[str, ...], label: str) -> None:
    if (
        type(forms) is not tuple
        or not forms
        or len(set(forms)) != len(forms)
        or any(
            type(form) is not str
            or form != form.casefold()
            or _WORD.fullmatch(form) is None
            for form in forms
        )
    ):
        raise ValueError(f"{label} must contain unique case-folded words")


@dataclass(frozen=True, slots=True)
class NounConceptFamily:
    """One selected task family copied exactly from the reviewed v1 surface."""

    task_id: str
    category: str
    forms: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task family")
        require_identifier(self.category, "task category")
        _require_forms(self.forms, "task forms")

    def as_record(self) -> dict[str, object]:
        """Return the canonical family record."""
        return {
            "category": self.category,
            "forms": list(self.forms),
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class NounsV2Manifest:
    """Authenticated immutable definition of the disjoint benchmark."""

    parent_partition_sha256: str
    parent_breakdown_sha256: str
    parent_decisions_sha256: str
    source_identity: dict[str, object]
    tokenizer_identity: dict[str, object]
    parent_story_count: int
    task_families: tuple[NounConceptFamily, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.parent_partition_sha256, "parent partition"),
            (self.parent_breakdown_sha256, "parent breakdown"),
            (self.parent_decisions_sha256, "parent decisions"),
        ):
            require_sha256(value, label)
        if self.parent_story_count <= 0:
            raise ValueError("parent story count must be positive")
        if tuple(family.task_id for family in self.task_families) != TASK_IDS:
            raise ValueError("v2 task families must follow the frozen 24-task order")

    @property
    def manifest_sha256(self) -> str:
        """Return the independent v2 manifest identity."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical self-hashing manifest record."""
        core = {
            "assignment_policy": "zero-selected=base; one-selected=task; two-or-more=excluded",
            "base_validation_bucket_count": BASE_VALIDATION_BUCKET_COUNT,
            "format": MANIFEST_FORMAT,
            "minimum_task_train_stories": MINIMUM_TASK_TRAIN_STORIES,
            "minimum_task_validation_stories": MINIMUM_TASK_VALIDATION_STORIES,
            "parent_breakdown_sha256": self.parent_breakdown_sha256,
            "parent_decisions_sha256": self.parent_decisions_sha256,
            "parent_partition_sha256": self.parent_partition_sha256,
            "parent_story_count": self.parent_story_count,
            "probe_story_count": PROBE_STORY_COUNT,
            "schema_version": SCHEMA_VERSION,
            "source_identity": self.source_identity,
            "task_families": [family.as_record() for family in self.task_families],
            "tokenizer_identity": self.tokenizer_identity,
        }
        return {**core, "manifest_sha256": record_sha256(core)} if include_hash else core


@dataclass(frozen=True, slots=True)
class NounsV2TaskSummary:
    """Raw, pure, excluded, probe, update, and validation counts for one task."""

    task_id: str
    forms: tuple[str, ...]
    raw_train_story_count: int
    train_story_count: int
    update_story_count: int
    raw_validation_story_count: int
    validation_story_count: int
    generation_story_count: int
    excluded_train_story_count: int
    excluded_validation_story_count: int
    probe_story_ids: tuple[str, ...]
    raw_train_form_counts: tuple[tuple[str, int], ...]
    retained_train_form_counts: tuple[tuple[str, int], ...]
    raw_validation_form_counts: tuple[tuple[str, int], ...]
    retained_validation_form_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "v2 task")
        _require_forms(self.forms, "v2 task forms")
        counts = (
            self.raw_train_story_count,
            self.train_story_count,
            self.update_story_count,
            self.raw_validation_story_count,
            self.validation_story_count,
            self.generation_story_count,
            self.excluded_train_story_count,
            self.excluded_validation_story_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("v2 task counts must be nonnegative integers")
        if (
            self.update_story_count != self.train_story_count - PROBE_STORY_COUNT
            or len(self.probe_story_ids) != PROBE_STORY_COUNT
            or len(set(self.probe_story_ids)) != PROBE_STORY_COUNT
            or self.generation_story_count != self.validation_story_count
            or self.excluded_train_story_count
            != self.raw_train_story_count - self.train_story_count
            or self.excluded_validation_story_count
            != self.raw_validation_story_count - self.validation_story_count
        ):
            raise ValueError("v2 task retained, excluded, and probe counts differ")
        for story_id in self.probe_story_ids:
            require_sha256(story_id, "task probe")
        for label, values in (
            ("raw train", self.raw_train_form_counts),
            ("retained train", self.retained_train_form_counts),
            ("raw validation", self.raw_validation_form_counts),
            ("retained validation", self.retained_validation_form_counts),
        ):
            if tuple(form for form, _ in values) != self.forms or any(
                type(count) is not int or count < 0 for _, count in values
            ):
                raise ValueError(f"{label} form counts must follow the frozen forms")

    def as_record(self) -> dict[str, object]:
        """Return the canonical task summary."""
        return {
            "excluded_train_story_count": self.excluded_train_story_count,
            "excluded_validation_story_count": self.excluded_validation_story_count,
            "forms": list(self.forms),
            "generation_story_count": self.generation_story_count,
            "probe_story_ids": list(self.probe_story_ids),
            "raw_train_form_counts": dict(self.raw_train_form_counts),
            "raw_train_story_count": self.raw_train_story_count,
            "raw_validation_form_counts": dict(self.raw_validation_form_counts),
            "raw_validation_story_count": self.raw_validation_story_count,
            "retained_train_form_counts": dict(self.retained_train_form_counts),
            "retained_validation_form_counts": dict(
                self.retained_validation_form_counts
            ),
            "task_id": self.task_id,
            "train_story_count": self.train_story_count,
            "update_story_count": self.update_story_count,
            "validation_story_count": self.validation_story_count,
        }


@dataclass(frozen=True, slots=True)
class NounsV2PartitionArtifact:
    """Strict-loaded v2 indexes plus the authenticated immutable parent store."""

    root: Path
    parent_root: Path
    partition_sha256: str
    manifest_sha256: str
    parent_partition_sha256: str
    source_identity: dict[str, object]
    tokenizer_identity: dict[str, object]
    pad_token_id: int
    eos_token_id: int
    story_count: int
    train_unique_story_count: int
    validation_unique_story_count: int
    base_universe_story_count: int
    base_train_story_count: int
    base_validation_story_count: int
    excluded_train_story_count: int
    excluded_validation_story_count: int
    root_probe_story_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    tasks: tuple[NounsV2TaskSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "parent_root", Path(self.parent_root))
        for value, label in (
            (self.partition_sha256, "v2 partition"),
            (self.manifest_sha256, "v2 manifest"),
            (self.parent_partition_sha256, "v2 parent partition"),
        ):
            require_sha256(value, label)
        if self.task_ids != TASK_IDS or tuple(
            task.task_id for task in self.tasks
        ) != self.task_ids:
            raise ValueError("v2 task summaries must follow the frozen order")
        if len(self.root_probe_story_ids) != PROBE_STORY_COUNT:
            raise ValueError("v2 partition requires exactly 36 root probes")
        if self.base_train_story_count + self.base_validation_story_count != (
            self.base_universe_story_count
        ):
            raise ValueError("v2 base holdout must cover the complete clean universe")

    @property
    def benchmark_id(self) -> str:
        """Return the namespace used for deterministic engine ordering."""
        return BENCHMARK_ID

    @property
    def story_store_path(self) -> Path:
        """Return the authenticated parent story byte store."""
        return self.parent_root / "stories.bin"

    @property
    def token_store_path(self) -> Path:
        """Return the authenticated parent token store."""
        return self.parent_root / "tokens.uint16"

    @property
    def max_nodes(self) -> int:
        """Return root plus the 24 frozen task nodes."""
        return len(self.task_ids) + 1

    @property
    def max_edges(self) -> int:
        """Return the 24 task-derived edge capacity."""
        return len(self.task_ids)

    @property
    def base_training_format(self) -> str:
        """Return the versioned shared-engine base training format."""
        return BASE_TRAINING_FORMAT

    @property
    def base_selection_format(self) -> str:
        """Return the versioned selected-base format."""
        return BASE_SELECTION_FORMAT

    @property
    def vamp_stage_format(self) -> str:
        """Return the versioned VAMP stage format."""
        return VAMP_STAGE_FORMAT

    @property
    def gpu_preflight_format(self) -> str:
        """Return the versioned GPU preflight format."""
        return GPU_PREFLIGHT_FORMAT

    @property
    def whole_story_format(self) -> str:
        """Return the versioned whole-story row format."""
        return WHOLE_STORY_FORMAT

    @property
    def half_story_format(self) -> str:
        """Return the versioned midpoint-generation row format."""
        return HALF_STORY_FORMAT


@dataclass(frozen=True, slots=True)
class NounsV2ExperimentPreset:
    """Frozen executable settings for nouns-v2, hashed with its own format."""

    seed: int = 0
    context_length: int = 256
    microbatch_size: int = 32
    accumulation_microbatches: int = 8
    base_epochs: int = 2
    maximum_learning_rate: float = 5e-4
    minimum_learning_rate: float = 5e-5
    warmup_fraction: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    base_weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    base_checkpoint_interval: int = 1_000
    lora_rank: int = 8
    lora_alpha: float = 8.0
    adapter_updates: int = 2_000
    adapter_learning_rate: float = 1e-3
    adapter_weight_decay: float = 0.01
    allocator_peak_limit_bytes: int = 12 * 1024**3
    evaluation_chunk_size: int = 32

    def __post_init__(self) -> None:
        positive_integers = (
            self.context_length,
            self.microbatch_size,
            self.accumulation_microbatches,
            self.base_epochs,
            self.base_checkpoint_interval,
            self.lora_rank,
            self.adapter_updates,
            self.allocator_peak_limit_bytes,
            self.evaluation_chunk_size,
        )
        positive_floats = (
            self.maximum_learning_rate,
            self.minimum_learning_rate,
            self.warmup_fraction,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            self.base_weight_decay,
            self.gradient_clip_norm,
            self.lora_alpha,
            self.adapter_learning_rate,
            self.adapter_weight_decay,
        )
        if self.seed != 0 or any(
            type(value) is not int or value <= 0 for value in positive_integers
        ):
            raise ValueError("nouns-v2 dimensions and seed are frozen")
        if any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or value <= 0.0
            for value in positive_floats
        ):
            raise ValueError("nouns-v2 floating settings must be positive")

    @property
    def config_sha256(self) -> str:
        """Return the independent preset identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return every frozen setting with the v2 preset format."""
        return {
            "format": PRESET_FORMAT,
            "schema_version": SCHEMA_VERSION,
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
            },
        }


@dataclass(frozen=True, slots=True)
class WholeStoryNllRow:
    """One self-identifying nouns-v2 task/story/condition NLL result."""

    task_noun: str
    story_id: str
    condition: Condition
    selected_node: str
    selected_path: tuple[str, ...]
    oracle_node: str
    oracle_match: bool
    total_nll: float
    token_count: int
    mean_nll: float
    perplexity: float
    regret_vs_oracle: float

    def __post_init__(self) -> None:
        require_identifier(self.task_noun, "NLL task")
        require_sha256(self.story_id, "NLL story")
        if self.condition not in CONDITIONS:
            raise ValueError("unknown nouns-v2 NLL condition")
        if not self.selected_node or not self.oracle_node or not self.selected_path:
            raise ValueError("NLL routing metadata must be nonempty")
        if type(self.oracle_match) is not bool or self.token_count <= 0:
            raise ValueError("NLL token count and oracle flag are invalid")
        if any(
            not math.isfinite(value)
            for value in (
                self.total_nll,
                self.mean_nll,
                self.perplexity,
                self.regret_vs_oracle,
            )
        ):
            raise ValueError("NLL metrics must be finite")

    def as_record(self) -> dict[str, object]:
        """Return the canonical, content-addressed v2 JSONL row."""
        core = {
            "condition": self.condition,
            "format": WHOLE_STORY_FORMAT,
            "mean_nll": self.mean_nll,
            "oracle_match": self.oracle_match,
            "oracle_node": self.oracle_node,
            "perplexity": self.perplexity,
            "regret_vs_oracle": self.regret_vs_oracle,
            "selected_node": self.selected_node,
            "selected_path": list(self.selected_path),
            "story_id": self.story_id,
            "task_noun": self.task_noun,
            "token_count": self.token_count,
            "total_nll": self.total_nll,
        }
        return {**core, "result_sha256": record_sha256(core)}


@dataclass(frozen=True, slots=True)
class StagewiseConditionResult:
    """One condition's midpoint route and held-out suffix loss at one stage."""

    condition: Condition
    selected_node: str
    selected_path: tuple[str, ...]
    oracle_match: bool
    total_nll: float
    token_count: int
    mean_nll: float
    regret_vs_oracle: float

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError("unknown nouns-v2 stagewise condition")
        if not self.selected_node or not self.selected_path:
            raise ValueError("stagewise routing metadata must be nonempty")
        if type(self.oracle_match) is not bool or self.token_count <= 0:
            raise ValueError("stagewise token count and oracle flag are invalid")
        if any(
            not math.isfinite(value)
            for value in (self.total_nll, self.mean_nll, self.regret_vs_oracle)
        ):
            raise ValueError("stagewise NLL metrics must be finite")
        if self.total_nll < 0.0 or self.mean_nll < 0.0:
            raise ValueError("stagewise NLL metrics must be nonnegative")

    def as_record(self) -> dict[str, object]:
        """Return one nested stagewise condition record."""
        return {
            "condition": self.condition,
            "mean_nll": self.mean_nll,
            "oracle_match": self.oracle_match,
            "regret_vs_oracle": self.regret_vs_oracle,
            "selected_node": self.selected_node,
            "selected_path": list(self.selected_path),
            "token_count": self.token_count,
            "total_nll": self.total_nll,
        }


@dataclass(frozen=True, slots=True)
class StagewiseClRow:
    """One task/story measurement under every condition at one VAMP stage."""

    stage_index: int
    introduced_task: str
    stage_tensor_checksum: str
    task_noun: str
    story_id: str
    results: tuple[StagewiseConditionResult, ...]

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or not 1 <= self.stage_index <= len(TASK_IDS):
            raise ValueError("stagewise stage index is outside the frozen task order")
        if self.introduced_task != TASK_IDS[self.stage_index - 1]:
            raise ValueError("stagewise introduced task does not match its stage")
        require_sha256(self.stage_tensor_checksum, "stagewise adaptation")
        require_identifier(self.task_noun, "stagewise task")
        require_sha256(self.story_id, "stagewise story")
        if self.task_noun not in TASK_IDS[: self.stage_index]:
            raise ValueError("stagewise rows may only evaluate learned tasks")
        if tuple(result.condition for result in self.results) != CONDITIONS:
            raise ValueError("stagewise rows require all six conditions in order")

    def as_record(self) -> dict[str, object]:
        """Return the canonical content-addressed stagewise JSONL row."""
        core = {
            "format": STAGEWISE_FORMAT,
            "introduced_task": self.introduced_task,
            "results": {
                result.condition: result.as_record() for result in self.results
            },
            "stage_index": self.stage_index,
            "stage_tensor_checksum": self.stage_tensor_checksum,
            "story_id": self.story_id,
            "task_noun": self.task_noun,
        }
        return {**core, "result_sha256": record_sha256(core)}


__all__ = [
    "AUDIT_FORMAT",
    "BASE_SELECTION_FORMAT",
    "BASE_TRAINING_FORMAT",
    "BASE_UNIVERSE_STORY_COUNT",
    "BENCHMARK_ID",
    "CHECKPOINT_ROOT",
    "CONDITIONS",
    "DATA_ROOT",
    "EXPECTED_PURE_COUNTS",
    "EXCLUDED_TRAIN_STORY_COUNT",
    "EXCLUDED_VALIDATION_STORY_COUNT",
    "GPU_PREFLIGHT_FORMAT",
    "HALF_STORY_FORMAT",
    "JUDGE_FORMAT",
    "JUDGE_REQUEST_FORMAT",
    "MANIFEST_FORMAT",
    "NounConceptFamily",
    "NounsV2ExperimentPreset",
    "NounsV2Manifest",
    "NounsV2PartitionArtifact",
    "NounsV2TaskSummary",
    "PARENT_BREAKDOWN_SHA256",
    "PARENT_DECISIONS_SHA256",
    "PARENT_PARTITION_SHA256",
    "PARTITION_FORMAT",
    "PURE_TASK_TRAIN_STORY_COUNT",
    "PURE_TASK_VALIDATION_STORY_COUNT",
    "REPORT_FORMAT",
    "RESULT_ROOT",
    "RUN_MANIFEST_FORMAT",
    "SCHEMA_VERSION",
    "STAGEWISE_FORMAT",
    "STAGEWISE_CASE_COUNT",
    "StagewiseClRow",
    "StagewiseConditionResult",
    "TASK_IDS",
    "TRAIN_UNIQUE_STORY_COUNT",
    "VALIDATION_UNIQUE_STORY_COUNT",
    "VAMP_STAGE_FORMAT",
    "WHOLE_STORY_FORMAT",
    "WholeStoryNllRow",
    "canonical_json_bytes",
    "record_sha256",
    "require_identifier",
    "require_sha256",
]
