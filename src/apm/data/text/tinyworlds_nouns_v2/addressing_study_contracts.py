"""Independent v1 contracts for the nouns-v2 bounded-addressing study."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, TypeAlias

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    PURE_TASK_VALIDATION_STORY_COUNT,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)


ADDRESSING_STUDY_ID = "tinyworlds-nouns-v2-addressing-study"
RETRIEVAL_CONTRACT_FORMAT = "tinyworlds-nouns-v2-addressing-retrieval-contract-v1"
EBT_CONTRACT_FORMAT = "tinyworlds-nouns-v2-addressing-ebt-contract-v1"
RETRIEVAL_ROW_FORMAT = "tinyworlds-nouns-v2-addressing-retrieval-row-v1"
EBT_ROW_FORMAT = "tinyworlds-nouns-v2-addressing-ebt-row-v1"
TIMING_ROW_FORMAT = "tinyworlds-nouns-v2-addressing-timing-row-v1"
KEY_ARTIFACT_FORMAT = "tinyworlds-nouns-v2-addressing-keys-v1"
STUDY_MANIFEST_FORMAT = "tinyworlds-nouns-v2-addressing-study-manifest-v1"
REPORT_FORMAT = "tinyworlds-nouns-v2-addressing-report-v1"

RETRIEVAL_CASE_COUNT = PURE_TASK_VALIDATION_STORY_COUNT
RETRIEVAL_ROW_COUNT = 5 * RETRIEVAL_CASE_COUNT
EBT_ROW_COUNT = 11 * RETRIEVAL_CASE_COUNT
COMPACT_WIDTHS = (4, 8)
EBT_STEPS = 20
EBT_LEARNING_RATE = 0.1
EBT_TEMPERATURE = 1.0
EBT_ENTROPY_PENALTY = 0.01
HOPFIELD_BETA = 10.0
EBT_MODEL_FORWARD_COUNT = EBT_STEPS + 3
MICROBATCH_SIZE = 8
WARM_TIMING_REPETITIONS = 5
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 0
TOP8_NLL_NONINFERIORITY_MARGIN = 0.02
TOP8_ACCURACY_NONINFERIORITY_MARGIN = 0.02
COMPACT_PARITY_TOLERANCE = 1e-3

KeyScheme: TypeAlias = Literal[
    "canonical_full_centroid",
    "midpoint_content_centroid",
    "midpoint_content_prototype",
    "midpoint_content_residual_centroid",
    "midpoint_content_residual_prototype",
]
KEY_SCHEMES: tuple[KeyScheme, ...] = (
    "canonical_full_centroid",
    "midpoint_content_centroid",
    "midpoint_content_prototype",
    "midpoint_content_residual_centroid",
    "midpoint_content_residual_prototype",
)


@dataclass(frozen=True, slots=True)
class RetrievalStudyRow:
    """One scheme's frozen-key retrieval result for one midpoint prefix."""

    retrieval_contract_sha256: str
    scheme: KeyScheme
    task_noun: str
    story_id: str
    oracle_node_index: int
    top_8_indices: tuple[int, ...]
    entropy: float
    score_margin: float
    prefix_token_count: int

    def __post_init__(self) -> None:
        require_sha256(self.retrieval_contract_sha256, "retrieval contract")
        require_sha256(self.story_id, "retrieval story")
        if self.scheme not in KEY_SCHEMES:
            raise ValueError("unknown addressing-study key scheme")
        if (
            type(self.oracle_node_index) is not int
            or not 1 <= self.oracle_node_index < 25
            or len(self.top_8_indices) != 8
            or len(set(self.top_8_indices)) != 8
            or any(
                type(value) is not int or not 0 <= value < 25
                for value in self.top_8_indices
            )
            or type(self.prefix_token_count) is not int
            or self.prefix_token_count <= 0
        ):
            raise ValueError("retrieval indices and token count are invalid")
        if not math.isfinite(self.entropy) or self.entropy < 0.0:
            raise ValueError("retrieval entropy must be finite and nonnegative")
        if not math.isfinite(self.score_margin) or self.score_margin < 0.0:
            raise ValueError("retrieval margin must be finite and nonnegative")

    def as_record(self) -> dict[str, object]:
        """Return a canonical self-hashing retrieval JSONL record."""
        core = {
            "entropy": self.entropy,
            "format": RETRIEVAL_ROW_FORMAT,
            "oracle_node_index": self.oracle_node_index,
            "prefix_token_count": self.prefix_token_count,
            "retrieval_contract_sha256": self.retrieval_contract_sha256,
            "scheme": self.scheme,
            "score_margin": self.score_margin,
            "story_id": self.story_id,
            "task_noun": self.task_noun,
            "top_1_hit": self.top_8_indices[0] == self.oracle_node_index,
            "top_4_hit": self.oracle_node_index in self.top_8_indices[:4],
            "top_8_hit": self.oracle_node_index in self.top_8_indices,
            "top_8_indices": list(self.top_8_indices),
        }
        return {**core, "result_sha256": record_sha256(core)}


@dataclass(frozen=True, slots=True)
class EbtStudyRow:
    """One dense or compact EBT-H route and true-suffix competence result."""

    ebt_contract_sha256: str
    scheme: KeyScheme
    mode: Literal["dense_all", "compact"]
    candidate_width: int
    task_noun: str
    story_id: str
    oracle_node_index: int
    candidate_node_indices: tuple[int, ...]
    selected_node_index: int
    selected_path: tuple[str, ...]
    gathered_edge_count: int
    selected_path_edge_count: int
    physical_edge_capacity: int
    prefix_token_count: int
    prefix_width_bucket: int
    suffix_total_nll: float
    suffix_token_count: int
    suffix_mean_nll: float
    oracle_suffix_mean_nll: float
    retrieval_entropy: float
    retrieval_margin: float
    final_entropy: float
    final_margin: float

    def __post_init__(self) -> None:
        require_sha256(self.ebt_contract_sha256, "EBT contract")
        require_sha256(self.story_id, "EBT story")
        if self.scheme not in KEY_SCHEMES or self.mode not in ("dense_all", "compact"):
            raise ValueError("unknown addressing-study EBT method")
        if self.mode == "dense_all" and (
            self.scheme != "canonical_full_centroid" or self.candidate_width != 25
        ):
            raise ValueError("dense control must use all 25 canonical candidates")
        if self.mode == "compact" and self.candidate_width not in COMPACT_WIDTHS:
            raise ValueError("compact EBT width must be four or eight")
        if len(self.candidate_node_indices) != self.candidate_width:
            raise ValueError("EBT candidate list must match its logical width")
        if (
            len(set(self.candidate_node_indices)) != self.candidate_width
            or any(
                type(value) is not int or not 0 <= value < 25
                for value in self.candidate_node_indices
            )
            or type(self.selected_node_index) is not int
            or type(self.oracle_node_index) is not int
            or self.selected_node_index not in self.candidate_node_indices
            or not 1 <= self.oracle_node_index < 25
        ):
            raise ValueError("EBT candidate and selected node indices are invalid")
        integer_values = (
            self.oracle_node_index,
            self.selected_node_index,
            self.gathered_edge_count,
            self.selected_path_edge_count,
            self.physical_edge_capacity,
            self.prefix_token_count,
            self.prefix_width_bucket,
            self.suffix_token_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_values):
            raise ValueError("EBT integer measurements must be nonnegative")
        if (
            self.prefix_token_count <= 0
            or self.suffix_token_count <= 0
            or self.prefix_width_bucket <= 0
            or self.prefix_width_bucket % 32 != 0
        ):
            raise ValueError("EBT prefix and suffix token counts must be positive")
        if not self.selected_path:
            raise ValueError("EBT selected path must include at least the root")
        if self.selected_path_edge_count != len(self.selected_path) - 1:
            raise ValueError("selected path edge count differs from path metadata")
        if (
            any(type(value) is not str or not value for value in self.selected_path)
            or self.selected_path_edge_count > self.gathered_edge_count
            or self.gathered_edge_count > self.physical_edge_capacity
        ):
            raise ValueError("EBT path and physical edge measurements are invalid")
        if self.mode == "dense_all" and (
            self.candidate_node_indices != tuple(range(25))
            or self.gathered_edge_count != 24
            or self.physical_edge_capacity != 24
        ):
            raise ValueError("dense control must execute the complete 24-edge bank")
        if self.mode == "compact" and self.physical_edge_capacity not in (
            4,
            8,
            12,
            16,
            20,
            24,
        ):
            raise ValueError("compact physical edge capacity is not bucketed")
        numeric = (
            self.suffix_total_nll,
            self.suffix_mean_nll,
            self.oracle_suffix_mean_nll,
            self.retrieval_entropy,
            self.retrieval_margin,
            self.final_entropy,
            self.final_margin,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("EBT floating measurements must be finite and nonnegative")
        if not math.isclose(
            self.suffix_mean_nll,
            self.suffix_total_nll / self.suffix_token_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("EBT suffix total and mean NLL differ")

    def as_record(self) -> dict[str, object]:
        """Return a canonical self-hashing EBT JSONL record."""
        forward_tokens = self.prefix_token_count * EBT_MODEL_FORWARD_COUNT
        edge_evaluations = forward_tokens * self.gathered_edge_count
        core = {
            "active_lora_edge_evaluations": edge_evaluations,
            "candidate_node_indices": list(self.candidate_node_indices),
            "candidate_width": self.candidate_width,
            "ebt_contract_sha256": self.ebt_contract_sha256,
            "final_entropy": self.final_entropy,
            "final_margin": self.final_margin,
            "format": EBT_ROW_FORMAT,
            "gathered_edge_count": self.gathered_edge_count,
            "hopfield_dot_products": 25,
            "mode": self.mode,
            "model_forward_equivalent_prefix_tokens": forward_tokens,
            "oracle_match": self.selected_node_index == self.oracle_node_index,
            "oracle_node_index": self.oracle_node_index,
            "oracle_regret": self.suffix_mean_nll - self.oracle_suffix_mean_nll,
            "oracle_suffix_mean_nll": self.oracle_suffix_mean_nll,
            "physical_edge_capacity": self.physical_edge_capacity,
            "prefix_token_count": self.prefix_token_count,
            "prefix_width_bucket": self.prefix_width_bucket,
            "retrieval_entropy": self.retrieval_entropy,
            "retrieval_margin": self.retrieval_margin,
            "scheme": self.scheme,
            "selected_node_index": self.selected_node_index,
            "selected_path": list(self.selected_path),
            "selected_path_edge_count": self.selected_path_edge_count,
            "story_id": self.story_id,
            "suffix_mean_nll": self.suffix_mean_nll,
            "suffix_token_count": self.suffix_token_count,
            "suffix_total_nll": self.suffix_total_nll,
            "task_noun": self.task_noun,
        }
        return {**core, "result_sha256": record_sha256(core)}


def contract_record(
    format_name: str,
    core: dict[str, object],
) -> dict[str, object]:
    """Add the independent contract format and self-hash to one core record."""
    if format_name not in (RETRIEVAL_CONTRACT_FORMAT, EBT_CONTRACT_FORMAT):
        raise ValueError("unknown addressing-study contract format")
    complete = {"format": format_name, **core}
    return {**complete, "contract_sha256": record_sha256(complete)}


def load_contract(path: str | Path, expected_format: str) -> dict[str, object]:
    """Strict-load one canonical self-hashing study contract."""
    payload = Path(path).read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError("addressing-study contract is not canonical JSON")
    supplied = record.get("contract_sha256")
    core = {key: value for key, value in record.items() if key != "contract_sha256"}
    if (
        expected_format not in (RETRIEVAL_CONTRACT_FORMAT, EBT_CONTRACT_FORMAT)
        or record.get("format") != expected_format
        or supplied != record_sha256(core)
    ):
        raise ValueError("addressing-study contract identity changed")
    return record


def validate_jsonl_rows(
    path: str | Path,
    *,
    expected_format: str,
    contract_field: str,
    contract_sha256: str,
    key_fields: tuple[str, ...],
    expected_keys: set[tuple[object, ...]],
    require_complete: bool,
) -> set[tuple[object, ...]]:
    """Reject malformed, tampered, duplicate, unexpected, or incomplete rows."""
    require_sha256(contract_sha256, "ledger contract")
    source = Path(path)
    if not source.is_file():
        if require_complete:
            raise FileNotFoundError(source)
        return set()
    completed: set[tuple[object, ...]] = set()
    with source.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("addressing-study ledger has an interrupted tail")
            row = json.loads(line)
            if type(row) is not dict or canonical_json_bytes(row) != line:
                raise ValueError("addressing-study ledger is not canonical JSONL")
            supplied = row.get("result_sha256")
            core = {key: value for key, value in row.items() if key != "result_sha256"}
            if (
                row.get("format") != expected_format
                or row.get(contract_field) != contract_sha256
                or supplied != record_sha256(core)
            ):
                raise ValueError("addressing-study ledger row identity changed")
            key = tuple(row.get(field) for field in key_fields)
            if key in completed:
                raise ValueError("addressing-study ledger contains a duplicate row")
            if key not in expected_keys:
                raise ValueError("addressing-study ledger contains an unexpected row")
            completed.add(key)
    if require_complete and completed != expected_keys:
        raise ValueError(
            f"addressing-study ledger has {len(completed):,} of "
            f"{len(expected_keys):,} expected rows"
        )
    return completed


def mean(values: Iterable[float]) -> float:
    """Return a strict finite arithmetic mean."""
    measured = tuple(float(value) for value in values)
    if not measured or any(not math.isfinite(value) for value in measured):
        raise ValueError("addressing-study mean requires finite values")
    return math.fsum(measured) / len(measured)


__all__ = [
    "ADDRESSING_STUDY_ID",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "COMPACT_WIDTHS",
    "COMPACT_PARITY_TOLERANCE",
    "EBT_CONTRACT_FORMAT",
    "EBT_ENTROPY_PENALTY",
    "EBT_LEARNING_RATE",
    "EBT_MODEL_FORWARD_COUNT",
    "EBT_ROW_COUNT",
    "EBT_ROW_FORMAT",
    "EBT_STEPS",
    "EBT_TEMPERATURE",
    "EbtStudyRow",
    "HOPFIELD_BETA",
    "KEY_ARTIFACT_FORMAT",
    "KEY_SCHEMES",
    "KeyScheme",
    "MICROBATCH_SIZE",
    "REPORT_FORMAT",
    "RETRIEVAL_CASE_COUNT",
    "RETRIEVAL_CONTRACT_FORMAT",
    "RETRIEVAL_ROW_COUNT",
    "RETRIEVAL_ROW_FORMAT",
    "RetrievalStudyRow",
    "STUDY_MANIFEST_FORMAT",
    "TIMING_ROW_FORMAT",
    "TOP8_ACCURACY_NONINFERIORITY_MARGIN",
    "TOP8_NLL_NONINFERIORITY_MARGIN",
    "WARM_TIMING_REPETITIONS",
    "canonical_json_bytes",
    "contract_record",
    "load_contract",
    "mean",
    "record_sha256",
    "validate_jsonl_rows",
]
