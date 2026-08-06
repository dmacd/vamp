"""Immutable public contracts for the TinyWorlds noun-overlap experiment."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Literal, TypeAlias


BENCHMARK_ID = "tinyworlds-nouns-v1"
SCHEMA_VERSION = 1
BREAKDOWN_FORMAT = "tinyworlds-nouns-breakdown-v1"
APPROVAL_FORMAT = "tinyworlds-nouns-approval-v1"
PARTITION_FORMAT = "tinyworlds-nouns-partition-v1"
RUN_MANIFEST_FORMAT = "tinyworlds-nouns-run-v1"

DATA_ROOT = Path("data/tinyworlds-nouns-v1")
CHECKPOINT_ROOT = Path("checkpoints/tinyworlds-nouns-v1")
RESULT_ROOT = Path("results/language_cl/tinyworlds-nouns-v1")
DEFAULT_SOURCE_ROOT = Path("data/tinystories-v2")
DEFAULT_TOKENIZER_PATH = Path(
    "checkpoints/tinystories-8m/tokenizer/tokenizer.json"
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
    """Encode one artifact value as deterministic UTF-8 JSON plus newline."""
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
    """Hash one canonical JSON-compatible value."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: str, label: str) -> None:
    """Require one lowercase SHA-256 string."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def require_identifier(value: str, label: str) -> None:
    """Require one portable lowercase identifier."""
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
class NounDecision:
    """One manually reviewable noun family and its exact accepted forms."""

    concept_id: str
    category: str
    forms: tuple[str, ...]
    included: bool
    reason: str

    def __post_init__(self) -> None:
        require_identifier(self.concept_id, "noun concept")
        require_identifier(self.category, "noun category")
        _require_forms(self.forms, "noun forms")
        if type(self.included) is not bool or not self.reason.strip():
            raise ValueError("noun decisions require a boolean and review reason")

    def as_record(self) -> dict[str, object]:
        """Return the canonical editable decision record."""
        return {
            "category": self.category,
            "concept_id": self.concept_id,
            "forms": list(self.forms),
            "included": self.included,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class NounEvidence:
    """One deterministic complete-story example with exact provenance."""

    story_id: str
    source_split: Literal["train", "validation"]
    matched_forms: tuple[str, ...]
    story: str

    def __post_init__(self) -> None:
        require_sha256(self.story_id, "evidence story")
        if self.source_split not in ("train", "validation"):
            raise ValueError("evidence source split is invalid")
        _require_forms(self.matched_forms, "evidence forms")
        if not self.story or self.story != self.story.strip():
            raise ValueError("evidence story must be nonempty normalized text")

    def as_record(self) -> dict[str, object]:
        """Return one canonical evidence row."""
        return {
            "matched_forms": list(self.matched_forms),
            "source_split": self.source_split,
            "story": self.story,
            "story_id": self.story_id,
        }


@dataclass(frozen=True, slots=True)
class NounBreakdownRow:
    """Counts, proposal, projected role, and examples for one noun family."""

    decision: NounDecision
    train_story_count: int
    validation_story_count: int
    train_form_counts: tuple[tuple[str, int], ...]
    validation_form_counts: tuple[tuple[str, int], ...]
    train_prevalence: float
    validation_prevalence: float
    threshold_eligible: bool
    projected_role: Literal["base", "task", "excluded", "below_threshold"]
    evidence: tuple[NounEvidence, ...]

    def __post_init__(self) -> None:
        counts = (self.train_story_count, self.validation_story_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("noun story counts must be nonnegative")
        for label, values in (
            ("train", self.train_form_counts),
            ("validation", self.validation_form_counts),
        ):
            if (
                type(values) is not tuple
                or tuple(form for form, _ in values) != self.decision.forms
                or any(type(count) is not int or count < 0 for _, count in values)
            ):
                raise ValueError(f"{label} form counts must follow accepted forms")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.train_prevalence, self.validation_prevalence)
        ):
            raise ValueError("noun prevalence must lie in [0, 1]")
        if type(self.threshold_eligible) is not bool:
            raise TypeError("threshold_eligible must be boolean")
        if self.projected_role not in (
            "base",
            "task",
            "excluded",
            "below_threshold",
        ):
            raise ValueError("noun projected role is invalid")
        if type(self.evidence) is not tuple or any(
            type(item) is not NounEvidence for item in self.evidence
        ):
            raise TypeError("noun evidence must be an immutable evidence tuple")

    def as_record(self) -> dict[str, object]:
        """Return one canonical breakdown row."""
        return {
            "decision": self.decision.as_record(),
            "evidence": [item.as_record() for item in self.evidence],
            "projected_role": self.projected_role,
            "threshold_eligible": self.threshold_eligible,
            "train_form_counts": dict(self.train_form_counts),
            "train_prevalence": self.train_prevalence,
            "train_story_count": self.train_story_count,
            "validation_form_counts": dict(self.validation_form_counts),
            "validation_prevalence": self.validation_prevalence,
            "validation_story_count": self.validation_story_count,
        }


@dataclass(frozen=True, slots=True)
class BaseSelectionStep:
    """One greedy addition to the approved noun-union base."""

    concept_id: str
    noun_story_count: int
    new_story_count: int
    cumulative_story_count: int
    cumulative_story_coverage: float
    new_token_count: int
    cumulative_token_count: int
    cumulative_token_coverage: float

    def __post_init__(self) -> None:
        require_identifier(self.concept_id, "base-selection noun")
        integers = (
            self.noun_story_count,
            self.new_story_count,
            self.cumulative_story_count,
            self.new_token_count,
            self.cumulative_token_count,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("base-selection counts must be nonnegative")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (
                self.cumulative_story_coverage,
                self.cumulative_token_coverage,
            )
        ):
            raise ValueError("base-selection coverage must lie in [0, 1]")

    def as_record(self) -> dict[str, object]:
        """Return one canonical base-selection row."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class NounBreakdown:
    """Complete human-review boundary derived from exact pinned sources."""

    source_identity: dict[str, object]
    tokenizer_identity: dict[str, object]
    train_unique_story_count: int
    validation_unique_story_count: int
    train_token_count: int
    validation_token_count: int
    rows: tuple[NounBreakdownRow, ...]
    base_selection: tuple[BaseSelectionStep, ...]

    def __post_init__(self) -> None:
        counts = (
            self.train_unique_story_count,
            self.validation_unique_story_count,
            self.train_token_count,
            self.validation_token_count,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("breakdown corpus counts must be positive")
        if not self.rows or len({row.decision.concept_id for row in self.rows}) != len(
            self.rows
        ):
            raise ValueError("breakdown noun rows must be nonempty and unique")
        selected = tuple(step.concept_id for step in self.base_selection)
        if len(set(selected)) != len(selected):
            raise ValueError("base-selection nouns must be unique")

    @property
    def breakdown_sha256(self) -> str:
        """Return the exact review identity."""
        return record_sha256(self.as_record(include_hash=False))

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return projected task nouns in deterministic count order."""
        return tuple(
            row.decision.concept_id
            for row in sorted(
                (row for row in self.rows if row.projected_role == "task"),
                key=lambda row: (-row.train_story_count, row.decision.concept_id),
            )
        )

    @property
    def base_concept_ids(self) -> tuple[str, ...]:
        """Return greedy base nouns in selection order."""
        return tuple(step.concept_id for step in self.base_selection)

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical review artifact record."""
        core = {
            "base_selection": [step.as_record() for step in self.base_selection],
            "format": BREAKDOWN_FORMAT,
            "rows": [row.as_record() for row in self.rows],
            "schema_version": SCHEMA_VERSION,
            "source_identity": self.source_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "train_token_count": self.train_token_count,
            "train_unique_story_count": self.train_unique_story_count,
            "validation_token_count": self.validation_token_count,
            "validation_unique_story_count": self.validation_unique_story_count,
        }
        return (
            {**core, "breakdown_sha256": record_sha256(core)}
            if include_hash
            else core
        )


@dataclass(frozen=True, slots=True)
class NounApproval:
    """Manual authorization bound to one exact noun breakdown."""

    breakdown_sha256: str
    decision_sha256: str
    source_sha256: str
    approval_statement: str = "I manually approve this exact noun breakdown."

    def __post_init__(self) -> None:
        for value, label in (
            (self.breakdown_sha256, "approved breakdown"),
            (self.decision_sha256, "approved decision file"),
            (self.source_sha256, "approved source identity"),
        ):
            require_sha256(value, label)
        if not self.approval_statement:
            raise ValueError("noun approval statement must not be empty")

    @property
    def approval_sha256(self) -> str:
        """Return the content identity of this approval."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical approval record."""
        core = {
            "approval_statement": self.approval_statement,
            "breakdown_sha256": self.breakdown_sha256,
            "decision_sha256": self.decision_sha256,
            "format": APPROVAL_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
        }
        return (
            {**core, "approval_sha256": record_sha256(core)}
            if include_hash
            else core
        )


@dataclass(frozen=True, slots=True)
class NounTaskSummary:
    """Partition counts and overlap evidence for one retained noun task."""

    task_id: str
    train_story_count: int
    update_story_count: int
    validation_story_count: int
    generation_story_count: int
    probe_story_ids: tuple[str, ...]
    base_overlap_story_count: int
    overlap_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task noun")
        counts = (
            self.train_story_count,
            self.update_story_count,
            self.validation_story_count,
            self.generation_story_count,
            self.base_overlap_story_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("task summary counts must be nonnegative")
        if len(self.probe_story_ids) != 36 or len(set(self.probe_story_ids)) != 36:
            raise ValueError("tasks require 36 unique addressing probes")
        if self.generation_story_count > self.validation_story_count:
            raise ValueError("generation stories must be a subset of validation stories")
        for story_id in self.probe_story_ids:
            require_sha256(story_id, "task-selected story")
        if tuple(name for name, _ in self.overlap_counts) != tuple(
            sorted(name for name, _ in self.overlap_counts)
        ):
            raise ValueError("task overlap counts must be noun-sorted")

    def as_record(self) -> dict[str, object]:
        """Return one canonical task summary."""
        return {
            "base_overlap_story_count": self.base_overlap_story_count,
            "generation_story_count": self.generation_story_count,
            "overlap_counts": dict(self.overlap_counts),
            "probe_story_ids": list(self.probe_story_ids),
            "task_id": self.task_id,
            "train_story_count": self.train_story_count,
            "update_story_count": self.update_story_count,
            "validation_story_count": self.validation_story_count,
        }


@dataclass(frozen=True, slots=True)
class NounPartitionArtifact:
    """Loaded content-addressed noun partition and its runtime paths."""

    root: Path
    partition_sha256: str
    breakdown_sha256: str
    approval_sha256: str
    source_identity: dict[str, object]
    tokenizer_identity: dict[str, object]
    pad_token_id: int
    eos_token_id: int
    base_concept_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    base_train_story_count: int
    base_validation_story_count: int
    root_probe_story_ids: tuple[str, ...]
    tasks: tuple[NounTaskSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        for value, label in (
            (self.partition_sha256, "noun partition"),
            (self.breakdown_sha256, "partition breakdown"),
            (self.approval_sha256, "partition approval"),
        ):
            require_sha256(value, label)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.pad_token_id,
                self.eos_token_id,
                self.base_train_story_count,
                self.base_validation_story_count,
            )
        ):
            raise ValueError("partition token IDs and counts must be nonnegative")
        if len(self.root_probe_story_ids) != 36:
            raise ValueError("partition requires exactly 36 root probes")
        if tuple(task.task_id for task in self.tasks) != self.task_ids:
            raise ValueError("partition task summaries must follow task order")

    @property
    def max_nodes(self) -> int:
        """Return root-plus-task VAMP capacity."""
        return len(self.task_ids) + 1

    @property
    def max_edges(self) -> int:
        """Return task-derived VAMP edge capacity."""
        return len(self.task_ids)


@dataclass(frozen=True, slots=True)
class NounsExperimentPreset:
    """Frozen executable settings for the noun-overlap experiment."""

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
        if self.seed != 0 or any(
            type(value) is not int or value <= 0 for value in positive_integers
        ):
            raise ValueError("noun experiment dimensions and seed are frozen")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
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
        ):
            raise ValueError("noun experiment floating settings must be positive")

    @property
    def config_sha256(self) -> str:
        """Hash every frozen scalar setting."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the complete experiment configuration."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class WholeStoryNllRow:
    """One streamed task/story/condition whole-story result."""

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
            raise ValueError("unknown noun NLL condition")
        if not self.selected_node or not self.oracle_node or not self.selected_path:
            raise ValueError("NLL routing metadata must be nonempty")
        if type(self.oracle_match) is not bool or self.token_count <= 0:
            raise ValueError("NLL token count and oracle flag are invalid")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.total_nll,
                self.mean_nll,
                self.perplexity,
            )
        ):
            raise ValueError("NLL metrics must be finite and nonnegative")
        if not math.isfinite(self.regret_vs_oracle):
            raise ValueError("NLL regret must be finite")

    def as_record(self) -> dict[str, object]:
        """Return one canonical JSONL result row."""
        return {
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "selected_path"
            },
            "selected_path": list(self.selected_path),
        }


__all__ = [
    "APPROVAL_FORMAT",
    "BENCHMARK_ID",
    "BREAKDOWN_FORMAT",
    "CHECKPOINT_ROOT",
    "CONDITIONS",
    "Condition",
    "DATA_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "DEFAULT_TOKENIZER_PATH",
    "NounApproval",
    "NounBreakdown",
    "NounBreakdownRow",
    "NounDecision",
    "NounEvidence",
    "NounPartitionArtifact",
    "NounTaskSummary",
    "NounsExperimentPreset",
    "PARTITION_FORMAT",
    "RESULT_ROOT",
    "RUN_MANIFEST_FORMAT",
    "SCHEMA_VERSION",
    "BaseSelectionStep",
    "WholeStoryNllRow",
    "canonical_json_bytes",
    "record_sha256",
    "require_identifier",
    "require_sha256",
]
