"""Re-score the sealed TRACE evidence with a task-known provenance router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import base64
import csv
from dataclasses import dataclass
from hashlib import sha256
import html
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Literal, Protocol, TypeAlias, cast

from tqdm.auto import tqdm

from apm.continual.artifacts import canonical_json_bytes, file_sha256
from apm.continual.trace.data import TraceExample, load_examples
from apm.continual.trace.lineage import build_hierarchy
from apm.continual.trace.metrics import (
    HeadlineMetrics,
    headline_metrics,
    per_example_task_scores,
    score_task,
)
from apm.continual.trace.protocol import ARRIVALS_PER_TASK, TASK_NAMES
from apm.continual.trace.task_known import TaskKnownRoute, build_task_known_routes


RUN_ID = "c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5"
FOLLOWUP_FORMAT = "trace-task-known-provenance-followup-v1"
ROUTER_RULE = (
    "maximize task-arrival coverage count; break ties by node purity, then "
    "end_arrival recency"
)
RECONSTRUCTION_TOLERANCE = 1.0e-8
CRAFT_URL = "https://arxiv.org/html/2605.05732v2"
CRAFT_OP = 44.17
CRAFT_OP_SPREAD = 0.35
CRAFT_BWT = 0.87
CRAFT_BWT_SPREAD = 0.19
FOCUS_CONDITION = "vamp_svd_r8_repair005"
VAMP_CONDITIONS = (
    "vamp_svd_r8_repair000",
    "vamp_svd_r8_repair005",
    "vamp_core_tsv_r8_scale03_repair000",
    "vamp_core_tsv_r8_scale03_repair005",
    "vamp_core_tsv_r8_scale05_repair000",
    "vamp_core_tsv_r8_scale05_repair010",
)
SCORING_PACKAGES = {
    "datasets": "2.21.0",
    "fuzzywuzzy": "0.18.0",
    "huggingface-hub": "0.34.4",
    "nltk": "3.9.1",
    "numpy": "1.26.4",
    "pandas": "2.3.2",
    "pyarrow": "21.0.0",
    "python-Levenshtein": "0.27.1",
    "rouge": "1.0.1",
    "sacrebleu": "2.5.1",
    "sacremoses": "0.1.1",
}

Split: TypeAlias = Literal["test", "validation"]
FollowupRouter: TypeAlias = Literal[
    "task_known_provenance", "task_known_validation"
]
JsonObject: TypeAlias = dict[str, object]


class SariMetric(Protocol):
    """Minimal cached metric interface needed by the offline follow-up."""

    def compute(
        self,
        *,
        sources: Sequence[str],
        predictions: Sequence[str],
        references: Sequence[Sequence[str]],
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class CandidateIndexEntry:
    """One authenticated raw-candidate file from the reviewer bundle index."""

    relative_path: str
    condition: str
    policy_hash: str
    stage: int
    task: str
    split: Split
    rows: int
    bytes: int
    sha256: str

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> CandidateIndexEntry:
        """Parse one VAMP row from the committed candidate index."""
        split = row["split"]
        if split not in ("test", "validation"):
            raise ValueError(f"unexpected candidate split: {split}")
        return cls(
            relative_path=row["relative_path"],
            condition=row["condition"],
            policy_hash=row["policy_hash"],
            stage=int(row["stage"]),
            task=row["task"],
            split=cast(Split, split),
            rows=int(row["rows"]),
            bytes=int(row["bytes"]),
            sha256=row["sha256"],
        )


@dataclass(frozen=True, slots=True)
class RawCandidate:
    """The score-relevant fields from one immutable generation record."""

    example_id: str
    candidate_id: str
    prompt_nll: float
    prediction: str


@dataclass(frozen=True, slots=True)
class FollowupScore:
    """One task-known provenance score for a condition, stage, task, and split."""

    condition: str
    policy_hash: str
    split: Split
    router: Literal["task_known_provenance"]
    stage: int
    task_index: int
    task: str
    candidate_id: str
    score: float

    def as_record(self) -> dict[str, object]:
        """Return a deterministic CSV/JSON-compatible score record."""
        return {
            "candidate_id": self.candidate_id,
            "condition": self.condition,
            "policy_hash": self.policy_hash,
            "router": self.router,
            "score": self.score,
            "split": self.split,
            "stage": self.stage,
            "task": self.task,
            "task_index": self.task_index,
        }


@dataclass(frozen=True, slots=True)
class SummaryRow:
    """One test-set continual-learning summary under a named task-known router."""

    condition: str
    router: FollowupRouter
    metrics: HeadlineMetrics

    def as_record(self) -> dict[str, object]:
        """Return explicitly labeled headline metrics for CSV output."""
        return {
            "bwt_clipped_negative_only": self.metrics.clipped_negative_backward_transfer,
            "bwt_signed": self.metrics.signed_backward_transfer,
            "condition": self.condition,
            "forgetting_craft_bwt": self.metrics.forgetting,
            "op": self.metrics.overall_performance,
            "router": self.router,
            "source_router": "task_aware" if self.router == "task_known_validation" else "",
            "split": "test",
        }


@dataclass(frozen=True, slots=True)
class TaskShardResult:
    """Compact validation metadata returned by one deterministic task process."""

    task: str
    score_path: str
    score_count: int
    reconstruction_checks: int
    maximum_error: float


@dataclass(frozen=True, slots=True)
class FollowupResult:
    """Published follow-up paths and the main scientific summaries."""

    output_root: Path
    report_markdown: Path
    report_html: Path
    manifest: Path
    summaries: tuple[SummaryRow, ...]


def load_candidate_index(bundle_root: str | Path) -> tuple[CandidateIndexEntry, ...]:
    """Load and validate the 432 VAMP candidate-file index entries."""
    path = Path(bundle_root) / "candidate-index.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(
            CandidateIndexEntry.from_csv_row(cast(dict[str, str], row))
            for row in csv.DictReader(handle)
            if row["condition"] in VAMP_CONDITIONS
        )
    expected_keys = {
        (condition, stage, TASK_NAMES[task_index - 1], split)
        for condition in VAMP_CONDITIONS
        for stage in range(1, len(TASK_NAMES) + 1)
        for task_index in range(1, stage + 1)
        for split in ("test", "validation")
    }
    actual_keys = {
        (row.condition, row.stage, row.task, row.split)
        for row in rows
    }
    if len(rows) != 432 or actual_keys != expected_keys:
        raise ValueError("candidate index does not contain the complete six-policy matrix")
    if len(actual_keys) != len(rows):
        raise ValueError("candidate index contains duplicate VAMP cell/split rows")
    condition_hashes = {
        condition: {row.policy_hash for row in rows if row.condition == condition}
        for condition in VAMP_CONDITIONS
    }
    if any(len(hashes) != 1 for hashes in condition_hashes.values()):
        raise ValueError("a VAMP condition maps to multiple policy hashes")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                VAMP_CONDITIONS.index(row.condition),
                row.stage,
                TASK_NAMES.index(row.task),
                row.split,
            ),
        )
    )


def verify_evidence_bundle(
    bundle_root: str | Path,
    entries: Sequence[CandidateIndexEntry],
) -> int:
    """Hash every source file and cross-check the relevant candidate index rows."""
    root = Path(bundle_root)
    evidence = root / "evidence-volume"
    manifest_rows = tuple(
        line.split(maxsplit=1)
        for line in (evidence / "SOURCE_SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(manifest_rows) != 1_798:
        raise ValueError("source evidence manifest must contain exactly 1,798 files")
    for expected, relative in tqdm(
        manifest_rows,
        desc="Evidence SHA-256",
        unit="file",
        dynamic_ncols=True,
        leave=False,
    ):
        path = evidence / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source evidence: {relative}")
        if file_sha256(path) != expected:
            raise ValueError(f"source evidence hash mismatch: {relative}")
    evaluations = evidence / "runs" / RUN_ID / "evaluations"
    for entry in entries:
        path = evaluations / entry.relative_path
        if path.stat().st_size != entry.bytes or file_sha256(path) != entry.sha256:
            raise ValueError(f"candidate index mismatch: {entry.relative_path}")
    if sum(entry.rows for entry in entries) != 301_968:
        raise ValueError("VAMP candidate index row total is not 301,968")
    return len(manifest_rows)


def verify_scoring_environment() -> dict[str, str]:
    """Require the package versions used by the sealed TRACE scoring run."""
    installed = {
        package: importlib.metadata.version(package)
        for package in SCORING_PACKAGES
    }
    mismatches = {
        package: (expected, installed[package])
        for package, expected in SCORING_PACKAGES.items()
        if installed[package] != expected
    }
    if mismatches:
        rendered = ", ".join(
            f"{package} expected {expected}, found {actual}"
            for package, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(f"TRACE scoring dependency mismatch: {rendered}")
    return installed


def prepare_sari_metric() -> tuple[str, str]:
    """Materialize the pinned SARI implementation and return its source identity."""
    try:
        from datasets import load_metric
    except ImportError as error:
        raise RuntimeError(
            "Install the CPU-only analysis environment with .[trace-analysis]"
        ) from error
    metric = load_metric("sari", trust_remote_code=True)
    source = Path(inspect.getfile(type(metric))).resolve()
    return source.name, file_sha256(source)


def _cached_sari_metric(cache_dir: Path) -> SariMetric:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_metric

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cast(
        SariMetric,
        load_metric(
            "sari",
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        ),
    )


def _run_root(bundle_root: Path) -> Path:
    run = bundle_root / "evidence-volume" / "runs" / RUN_ID
    if not run.is_dir():
        raise FileNotFoundError(f"missing sealed TRACE run: {run}")
    return run


def _load_json_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected a JSON object: {path}")
    return cast(JsonObject, value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping")
    return cast(dict[str, object], value)


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or not all(type(item) is str for item in value):
        raise ValueError(f"{label} must be a string list")
    return tuple(cast(list[str], value))


def _task_examples(
    examples: Sequence[TraceExample],
    task: str,
    split: Split,
) -> tuple[TraceExample, ...]:
    selected = tuple(
        example
        for example in examples
        if example.task == task and example.split == split
    )
    if not selected:
        raise ValueError(f"no {split} examples found for {task}")
    if len({example.example_id for example in selected}) != len(selected):
        raise ValueError(f"duplicate {task} {split} example identity")
    return selected


def _load_candidate_outputs(
    path: Path,
    entry: CandidateIndexEntry,
    examples: Sequence[TraceExample],
    candidate_order: Sequence[str],
) -> tuple[RawCandidate, ...]:
    expected_order = tuple(
        (example.example_id, candidate_id)
        for candidate_id in candidate_order
        for example in examples
    )
    records: list[RawCandidate] = []
    observed_order: list[tuple[str, str]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        value = json.loads(line)
        if type(value) is not dict or line != canonical_json_bytes(value):
            raise ValueError(f"candidate JSONL is not canonical: {path}")
        row = cast(JsonObject, value)
        if (
            row.get("format") != "trace-candidate-evaluation-v1"
            or row.get("task") != entry.task
            or row.get("split") != entry.split
            or int(cast(int, row.get("stage"))) != entry.stage
        ):
            raise ValueError(f"candidate row metadata differs from its index: {path}")
        prompt_nll = float(cast(float, row.get("prompt_nll")))
        if not math.isfinite(prompt_nll):
            raise ValueError(f"non-finite prompt NLL in {path}")
        record = RawCandidate(
            example_id=str(row.get("example_id")),
            candidate_id=str(row.get("candidate_id")),
            prompt_nll=prompt_nll,
            prediction=str(row.get("prediction")),
        )
        records.append(record)
        observed_order.append((record.example_id, record.candidate_id))
    if len(records) != entry.rows or tuple(observed_order) != expected_order:
        raise ValueError(
            "candidate rows are not the exact deterministic candidate/example product: "
            f"{path}"
        )
    return tuple(records)


def _candidate_score(
    task: str,
    examples: Sequence[TraceExample],
    outputs: Mapping[tuple[str, str], RawCandidate],
    candidate_id: str,
    sari_metric: SariMetric | None,
) -> float:
    return _task_score(
        task,
        tuple(example.prompt for example in examples),
        tuple(outputs[(example.example_id, candidate_id)].prediction for example in examples),
        tuple(example.answer for example in examples),
        sari_metric,
    )


def _selected_score(
    task: str,
    examples: Sequence[TraceExample],
    outputs: Mapping[tuple[str, str], RawCandidate],
    selected: Sequence[str],
    sari_metric: SariMetric | None,
) -> float:
    if len(examples) != len(selected):
        raise ValueError("router selection count differs from example count")
    return _task_score(
        task,
        tuple(example.prompt for example in examples),
        tuple(
            outputs[(example.example_id, candidate_id)].prediction
            for example, candidate_id in zip(examples, selected)
        ),
        tuple(example.answer for example in examples),
        sari_metric,
    )


def _task_score(
    task: str,
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
    sari_metric: SariMetric | None,
) -> float:
    if task != "20Minuten":
        return score_task(task, prompts, predictions, targets)
    if sari_metric is None:
        raise ValueError("20Minuten scoring requires the cached SARI metric")
    result = sari_metric.compute(
        sources=prompts,
        predictions=predictions,
        references=tuple((target,) for target in targets),
    )
    if "sari" not in result:
        raise ValueError("cached SARI metric returned an unexpected payload")
    return float(cast(float, result["sari"]))


def _per_example_scores(
    task: str,
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
    sari_metric: SariMetric | None,
) -> tuple[float, ...]:
    if task != "20Minuten":
        return per_example_task_scores(task, prompts, predictions, targets)
    if sari_metric is None:
        raise ValueError("20Minuten scoring requires the cached SARI metric")
    return tuple(
        _task_score(task, (prompt,), (prediction,), (target,), sari_metric)
        for prompt, prediction, target in zip(prompts, predictions, targets)
    )


def _reconstructed_scores(
    task: str,
    examples: Sequence[TraceExample],
    outputs: Mapping[tuple[str, str], RawCandidate],
    candidate_order: Sequence[str],
    task_known_validation_candidate: str,
    sari_metric: SariMetric | None,
) -> dict[str, float]:
    prompt_nll_selections = tuple(
        min(
            candidate_order,
            key=lambda candidate_id: outputs[
                (example.example_id, candidate_id)
            ].prompt_nll,
        )
        for example in examples
    )
    per_candidate = {
        candidate_id: _per_example_scores(
            task,
            tuple(example.prompt for example in examples),
            tuple(
                outputs[(example.example_id, candidate_id)].prediction
                for example in examples
            ),
            tuple(example.answer for example in examples),
            sari_metric,
        )
        for candidate_id in candidate_order
    }
    oracle_selections = tuple(
        max(
            candidate_order,
            key=lambda candidate_id: per_candidate[candidate_id][example_index],
        )
        for example_index in range(len(examples))
    )
    return {
        "answer_oracle": _selected_score(
            task, examples, outputs, oracle_selections, sari_metric
        ),
        "prompt_nll": _selected_score(
            task, examples, outputs, prompt_nll_selections, sari_metric
        ),
        "task_aware": _candidate_score(
            task,
            examples,
            outputs,
            task_known_validation_candidate,
            sari_metric,
        ),
    }


def _task_validation_candidate(
    task: str,
    examples: Sequence[TraceExample],
    outputs: Mapping[tuple[str, str], RawCandidate],
    candidate_order: Sequence[str],
    sari_metric: SariMetric | None,
) -> str:
    scores = {
        candidate_id: _candidate_score(
            task, examples, outputs, candidate_id, sari_metric
        )
        for candidate_id in candidate_order
    }
    return max(candidate_order, key=lambda candidate_id: scores[candidate_id])


def _entry_path(bundle_root: Path, entry: CandidateIndexEntry) -> Path:
    return _run_root(bundle_root) / "evaluations" / entry.relative_path


def _score_condition_task_shard(
    bundle_root_text: str,
    condition: str,
    task: str,
    score_path_text: str,
) -> TaskShardResult:
    bundle_root = Path(bundle_root_text)
    score_path = Path(score_path_text)
    examples = load_examples(_run_root(bundle_root) / "manifests" / "examples.jsonl")
    entries = tuple(
        entry
        for entry in load_candidate_index(bundle_root)
        if entry.task == task and entry.condition == condition
    )
    sari_metric = (
        _cached_sari_metric(score_path.parent / f"{score_path.stem}-sari-metric-cache")
        if task == "20Minuten"
        else None
    )
    arrival_record = _load_json_object(
        _run_root(bundle_root) / "manifests" / "arrivals.json"
    )
    routes = {
        (route.stage, route.task): route
        for route in build_task_known_routes(
            _string_sequence(arrival_record.get("arrival_ids"), "arrival_ids")
        )
    }
    score_count = 0
    checks = 0
    maximum_error = 0.0
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("wb") as output:
        stages = tuple(sorted({entry.stage for entry in entries}))
        for stage in stages:
            split_entries = {
                entry.split: entry
                for entry in entries
                if entry.condition == condition and entry.stage == stage
            }
            if set(split_entries) != {"test", "validation"}:
                raise ValueError(f"incomplete candidate splits for {condition}/{stage}/{task}")
            result_path = (
                _run_root(bundle_root)
                / "evaluations"
                / split_entries["test"].policy_hash
                / f"stage-{stage:02d}"
                / task
                / "result.json"
            )
            result = _load_json_object(result_path)
            candidate_order = _string_sequence(
                result.get("candidate_ids"), "candidate_ids"
            )
            if not candidate_order or candidate_order[0] != "base":
                raise ValueError("evaluation candidate order must start with the base")
            route = routes[(stage, task)]
            active_ids = tuple(
                node.node_id
                for node in build_hierarchy(
                    _string_sequence(arrival_record.get("arrival_ids"), "arrival_ids")[
                        : stage * ARRIVALS_PER_TASK
                    ]
                )[0].active_nodes
            )
            if set(candidate_order) != {"base", *active_ids}:
                raise ValueError(
                    f"result candidates differ from logical lineage for {condition}/{stage}/{task}"
                )
            if route.candidate_id not in candidate_order:
                raise ValueError("task-known provenance route is absent from candidate outputs")
            split_examples = {
                split: _task_examples(examples, task, cast(Split, split))
                for split in ("test", "validation")
            }
            split_outputs = {
                split: _load_candidate_outputs(
                    _entry_path(bundle_root, split_entries[cast(Split, split)]),
                    split_entries[cast(Split, split)],
                    split_examples[cast(Split, split)],
                    candidate_order,
                )
                for split in ("test", "validation")
            }
            by_split = {
                split: {
                    (row.example_id, row.candidate_id): row
                    for row in split_outputs[cast(Split, split)]
                }
                for split in ("test", "validation")
            }
            task_known_validation_candidate = _task_validation_candidate(
                task,
                split_examples["validation"],
                by_split["validation"],
                candidate_order,
                sari_metric,
            )
            if task_known_validation_candidate != str(result.get("task_aware_candidate")):
                raise ValueError(
                    f"validation-selected candidate mismatch for {condition}/{stage}/{task}"
                )
            for split in ("test", "validation"):
                canonical_split = cast(Split, split)
                reconstructed = _reconstructed_scores(
                    task,
                    split_examples[canonical_split],
                    by_split[canonical_split],
                    candidate_order,
                    task_known_validation_candidate,
                    sari_metric,
                )
                score_field = (
                    "router_scores" if split == "test" else "validation_router_scores"
                )
                expected_scores = _mapping(result.get(score_field), score_field)
                errors = tuple(
                    abs(score - float(cast(float, expected_scores[router])))
                    for router, score in reconstructed.items()
                )
                maximum_error = max(maximum_error, *errors)
                checks += len(errors)
                if any(error > RECONSTRUCTION_TOLERANCE for error in errors):
                    raise ValueError(
                        f"reconstructed router score mismatch for {condition}/{stage}/{task}/{split}"
                    )
                provenance_score = _candidate_score(
                    task,
                    split_examples[canonical_split],
                    by_split[canonical_split],
                    route.candidate_id,
                    sari_metric,
                )
                task_index = TASK_NAMES.index(task) + 1
                score = FollowupScore(
                    condition=condition,
                    policy_hash=split_entries[canonical_split].policy_hash,
                    split=canonical_split,
                    router="task_known_provenance",
                    stage=stage,
                    task_index=task_index,
                    task=task,
                    candidate_id=route.candidate_id,
                    score=provenance_score,
                )
                output.write(canonical_json_bytes(score.as_record()))
                output.flush()
                score_count += 1
    return TaskShardResult(
        task=task,
        score_path=str(score_path),
        score_count=score_count,
        reconstruction_checks=checks,
        maximum_error=maximum_error,
    )


def _load_shard_scores(results: Sequence[TaskShardResult]) -> tuple[FollowupScore, ...]:
    rows: list[FollowupScore] = []
    for result in results:
        for line in Path(result.score_path).read_bytes().splitlines(keepends=True):
            value = json.loads(line)
            if type(value) is not dict or line != canonical_json_bytes(value):
                raise ValueError(f"noncanonical task shard: {result.score_path}")
            row = cast(JsonObject, value)
            split = str(row["split"])
            if split not in ("test", "validation"):
                raise ValueError(f"unexpected shard split: {split}")
            rows.append(
                FollowupScore(
                    condition=str(row["condition"]),
                    policy_hash=str(row["policy_hash"]),
                    split=cast(Split, split),
                    router="task_known_provenance",
                    stage=int(cast(int, row["stage"])),
                    task_index=int(cast(int, row["task_index"])),
                    task=str(row["task"]),
                    candidate_id=str(row["candidate_id"]),
                    score=float(cast(float, row["score"])),
                )
            )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                VAMP_CONDITIONS.index(row.condition),
                row.stage,
                row.task_index,
                row.split,
            ),
        )
    )
    if len(ordered) != 432 or len(
        {
            (row.condition, row.stage, row.task, row.split)
            for row in ordered
        }
    ) != 432:
        raise ValueError("task shards did not produce the complete 432-score matrix")
    return ordered


def _primary_score_rows(bundle_root: Path) -> tuple[dict[str, str], ...]:
    path = bundle_root / "final" / "reports" / "primary-scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(cast(dict[str, str], row) for row in csv.DictReader(handle))


def _score_matrix(
    scores: Sequence[FollowupScore],
    condition: str,
    split: Split = "test",
) -> dict[tuple[int, int], float]:
    matrix = {
        (row.task_index, row.stage): row.score
        for row in scores
        if row.condition == condition and row.split == split
    }
    headline_metrics(matrix)
    return matrix


def _primary_matrix(
    primary_rows: Sequence[Mapping[str, str]],
    condition: str,
    split: Split,
    router: str,
) -> dict[tuple[int, int], float]:
    matrix = {
        (int(row["task_index"]), int(row["stage"])): float(row["score"])
        for row in primary_rows
        if row["condition"] == condition
        and row["split"] == split
        and row["router"] == router
    }
    headline_metrics(matrix)
    return matrix


def _summaries(
    scores: Sequence[FollowupScore],
    primary_rows: Sequence[Mapping[str, str]],
) -> tuple[SummaryRow, ...]:
    return tuple(
        SummaryRow(
            condition=condition,
            router=router,
            metrics=headline_metrics(
                _score_matrix(scores, condition)
                if router == "task_known_provenance"
                else _primary_matrix(primary_rows, condition, "test", "task_aware")
            ),
        )
        for condition in VAMP_CONDITIONS
        for router in ("task_known_provenance", "task_known_validation")
    )


def _taskwise_op(primary_rows: Sequence[Mapping[str, str]]) -> float:
    values = tuple(
        float(row["score"])
        for row in primary_rows
        if row["condition"] == "taskwise_lora"
        and row["split"] == "test"
        and row["router"] == "direct"
        and int(row["stage"]) == 8
    )
    if len(values) != len(TASK_NAMES):
        raise ValueError("primary score table lacks the taskwise-LoRA final row")
    return sum(values) / len(values)


def _verify_predeclared_focus(primary_rows: Sequence[Mapping[str, str]]) -> None:
    final_validation_op = {
        condition: sum(
            float(row["score"])
            for row in primary_rows
            if row["condition"] == condition
            and row["split"] == "validation"
            and row["router"] == "task_aware"
            and int(row["stage"]) == 8
        )
        / len(TASK_NAMES)
        for condition in VAMP_CONDITIONS
    }
    if max(VAMP_CONDITIONS, key=lambda condition: final_validation_op[condition]) != FOCUS_CONDITION:
        raise ValueError("predeclared focus is not the pre-existing validation winner")


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    records: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(records)


def _display_condition(condition: str) -> str:
    return {
        "vamp_svd_r8_repair000": "SVD, no repair",
        "vamp_svd_r8_repair005": "SVD, 5% repair",
        "vamp_core_tsv_r8_scale03_repair000": "Core 0.3, no repair",
        "vamp_core_tsv_r8_scale03_repair005": "Core 0.3, 5% repair",
        "vamp_core_tsv_r8_scale05_repair000": "Core 0.5, no repair",
        "vamp_core_tsv_r8_scale05_repair010": "Core 0.5, 10% repair",
    }[condition]


def _summary_lookup(
    summaries: Sequence[SummaryRow],
) -> dict[tuple[str, FollowupRouter], HeadlineMetrics]:
    return {
        (row.condition, row.router): row.metrics
        for row in summaries
    }


def _plot_op_bwt(path: Path, summaries: Sequence[SummaryRow], taskwise_op: float) -> None:
    import matplotlib.pyplot as plt

    lookup = _summary_lookup(summaries)
    figure, axes = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index, condition in enumerate(VAMP_CONDITIONS):
        provenance = lookup[(condition, "task_known_provenance")]
        validation = lookup[(condition, "task_known_validation")]
        color = colors[index]
        axes.plot(
            (provenance.forgetting, validation.forgetting),
            (provenance.overall_performance, validation.overall_performance),
            color=color,
            linewidth=1.8,
            alpha=0.8,
        )
        axes.scatter(
            provenance.forgetting,
            provenance.overall_performance,
            color=color,
            marker="o",
            s=75,
            label=f"{_display_condition(condition)}: provenance",
        )
        axes.scatter(
            validation.forgetting,
            validation.overall_performance,
            facecolors="none",
            edgecolors=color,
            marker="s",
            linewidths=1.8,
            s=75,
            label=f"{_display_condition(condition)}: validation-selected",
        )
    axes.errorbar(
        CRAFT_BWT,
        CRAFT_OP,
        xerr=CRAFT_BWT_SPREAD,
        yerr=CRAFT_OP_SPREAD,
        color="#111827",
        marker="*",
        markersize=14,
        capsize=4,
        linestyle="none",
        label="CRAFT 1B (external, 3-seed mean ± spread)",
    )
    axes.axhline(
        taskwise_op,
        color="#4b5563",
        linestyle="--",
        linewidth=1.3,
        label=f"Taskwise LoRA OP {taskwise_op:.3f} (BWT unavailable)",
    )
    axes.set_xlabel("Forgetting = diagonal − final (CRAFT BWT convention; lower is better)")
    axes.set_ylabel("Overall performance (OP; higher is better)")
    axes.set_title("Task-known provenance versus validation-selected node lookup")
    axes.grid(True, alpha=0.22)
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    figure.savefig(
        path,
        dpi=180,
        metadata={"Software": "APM TRACE task-known follow-up"},
    )
    plt.close(figure)


def _plot_route_coverage(path: Path, routes: Sequence[TaskKnownRoute]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    coverage = np.full((len(TASK_NAMES), len(TASK_NAMES)), np.nan)
    route_by_key = {(route.task_index, route.stage): route for route in routes}
    for route in routes:
        coverage[route.task_index - 1, route.stage - 1] = route.coverage
    figure, axes = plt.subplots(figsize=(12.5, 8.2), constrained_layout=True)
    image = axes.imshow(coverage, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    for (task_index, stage), route in route_by_key.items():
        axes.text(
            stage - 1,
            task_index - 1,
            f"{route.coverage_count}/5\n{route.interval}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="white" if route.coverage >= 0.7 else "#111827",
        )
    axes.set_xticks(range(len(TASK_NAMES)), range(1, len(TASK_NAMES) + 1))
    axes.set_yticks(range(len(TASK_NAMES)), TASK_NAMES)
    axes.set_xlabel("Evaluation stage")
    axes.set_ylabel("Known task")
    axes.set_title("Selected node: task-arrival coverage and represented interval")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82)
    colorbar.set_label("Fraction of the task's five arrivals in the selected node")
    figure.savefig(
        path,
        dpi=180,
        metadata={"Software": "APM TRACE task-known follow-up"},
    )
    plt.close(figure)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        (
            "| " + " | ".join(cell(header) for header in headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *(
                "| " + " | ".join(cell(value) for value in row) + " |"
                for row in rows
            ),
        )
    )


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    heading = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>"


def _summary_table_rows(
    summaries: Sequence[SummaryRow],
    final_route_alignment: Mapping[str, Mapping[str, bool]],
) -> tuple[tuple[str, ...], ...]:
    lookup = _summary_lookup(summaries)
    return tuple(
        (
            _display_condition(condition),
            f"{lookup[(condition, 'task_known_provenance')].overall_performance:.3f}",
            f"{lookup[(condition, 'task_known_provenance')].forgetting:.3f}",
            f"{lookup[(condition, 'task_known_validation')].overall_performance:.3f}",
            f"{lookup[(condition, 'task_known_validation')].forgetting:.3f}",
            f"{lookup[(condition, 'task_known_provenance')].overall_performance - lookup[(condition, 'task_known_validation')].overall_performance:+.3f}",
            f"{sum(final_route_alignment[condition].values())}/8",
        )
        for condition in VAMP_CONDITIONS
    )


def _focus_table_rows(
    scores: Sequence[FollowupScore],
    routes: Sequence[TaskKnownRoute],
    primary_rows: Sequence[Mapping[str, str]],
    final_route_alignment: Mapping[str, Mapping[str, bool]],
) -> tuple[tuple[str, ...], ...]:
    matrix = _score_matrix(scores, FOCUS_CONDITION)
    validation_matrix = _primary_matrix(
        primary_rows, FOCUS_CONDITION, "test", "task_aware"
    )
    final_routes = {route.task_index: route for route in routes if route.stage == 8}
    return tuple(
        (
            TASK_NAMES[task_index - 1],
            f"{matrix[(task_index, task_index)]:.3f}",
            f"{validation_matrix[(task_index, task_index)]:.3f}",
            f"{matrix[(task_index, 8)]:.3f}",
            f"{validation_matrix[(task_index, 8)]:.3f}",
            f"{matrix[(task_index, 8)] - validation_matrix[(task_index, 8)]:+.3f}",
            f"{matrix[(task_index, task_index)] - matrix[(task_index, 8)]:.3f}",
            "yes"
            if final_route_alignment[FOCUS_CONDITION][TASK_NAMES[task_index - 1]]
            else "no",
            final_routes[task_index].interval,
            f"{final_routes[task_index].coverage_count}/5",
            f"{final_routes[task_index].purity:.3f}",
        )
        for task_index in range(1, len(TASK_NAMES) + 1)
    )


def _route_table_rows(routes: Sequence[TaskKnownRoute]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            str(route.stage),
            route.task,
            route.interval,
            route.candidate_id,
            f"{route.coverage_count}/{route.task_arrivals}",
            f"{route.coverage:.3f}",
            f"{route.purity:.3f}",
        )
        for route in routes
    )


def _scientific_finding(
    summaries: Sequence[SummaryRow],
    final_route_alignment: Mapping[str, Mapping[str, bool]],
) -> tuple[str, str]:
    lookup = _summary_lookup(summaries)
    best_condition = max(
        VAMP_CONDITIONS,
        key=lambda condition: lookup[
            (condition, "task_known_provenance")
        ].overall_performance,
    )
    best = lookup[(best_condition, "task_known_provenance")]
    focus_provenance = lookup[(FOCUS_CONDITION, "task_known_provenance")]
    focus_validation = lookup[(FOCUS_CONDITION, "task_known_validation")]
    delta = (
        focus_provenance.overall_performance
        - focus_validation.overall_performance
    )
    final_matches = sum(final_route_alignment[FOCUS_CONDITION].values())
    diagonal_delta = (
        focus_provenance.overall_performance
        + focus_provenance.forgetting
        - focus_validation.overall_performance
        - focus_validation.forgetting
    )
    forgetting_delta = focus_provenance.forgetting - focus_validation.forgetting
    finding = (
        f"The strongest provenance-routed condition is {_display_condition(best_condition)} "
        f"at {best.overall_performance:.3f} OP with {best.forgetting:.3f} points of "
        "forgetting. "
        f"For the predeclared focus condition, {_display_condition(FOCUS_CONDITION)}, "
        f"provenance routing reaches {focus_provenance.overall_performance:.3f} OP versus "
        f"{focus_validation.overall_performance:.3f} for validation-selected lookup "
        f"({delta:+.3f} points); both routers choose the same final node for "
        f"{final_matches}/8 tasks."
    )
    interpretation = (
        "At the final stage, lineage provenance therefore approximates the validation lookup "
        "for this SVD condition. Its lower forgetting value must not be read as cleaner "
        f"retention: forgetting is {abs(forgetting_delta):.3f} points lower while its mean "
        f"diagonal starting score is {abs(diagonal_delta):.3f} points lower. Because each router can "
        "choose a different node at each stage, diagonal−final mixes routing quality with "
        "retention."
    )
    return finding, interpretation


def _final_route_alignment(
    bundle_root: Path,
    entries: Sequence[CandidateIndexEntry],
    routes: Sequence[TaskKnownRoute],
) -> dict[str, dict[str, bool]]:
    final_routes = {route.task: route.candidate_id for route in routes if route.stage == 8}
    policy_hashes = {
        condition: next(
            entry.policy_hash for entry in entries if entry.condition == condition
        )
        for condition in VAMP_CONDITIONS
    }
    return {
        condition: {
            task: str(
                _load_json_object(
                    _run_root(bundle_root)
                    / "evaluations"
                    / policy_hashes[condition]
                    / "stage-08"
                    / task
                    / "result.json"
                ).get("task_aware_candidate")
            )
            == final_routes[task]
            for task in TASK_NAMES
        }
        for condition in VAMP_CONDITIONS
    }


def _report_text(
    scores: Sequence[FollowupScore],
    routes: Sequence[TaskKnownRoute],
    summaries: Sequence[SummaryRow],
    primary_rows: Sequence[Mapping[str, str]],
    final_route_alignment: Mapping[str, Mapping[str, bool]],
    taskwise_op: float,
    reconstruction_checks: int,
    maximum_error: float,
) -> tuple[str, str]:
    summary_headers = (
        "Condition",
        "Provenance OP",
        "Provenance forgetting",
        "Validation-selected OP",
        "Validation-selected forgetting",
        "OP delta",
        "Same final node",
    )
    focus_headers = (
        "Task",
        "Prov. diagonal",
        "Valid. diagonal",
        "Prov. final",
        "Valid. final",
        "Final delta",
        "Prov. forgetting",
        "Same final node",
        "Final node",
        "Coverage",
        "Purity",
    )
    route_headers = (
        "Stage",
        "Task",
        "Node interval",
        "Candidate ID",
        "Coverage",
        "Coverage fraction",
        "Purity",
    )
    summary_rows = _summary_table_rows(summaries, final_route_alignment)
    focus_rows = _focus_table_rows(
        scores, routes, primary_rows, final_route_alignment
    )
    route_rows = _route_table_rows(routes)
    finding, interpretation = _scientific_finding(
        summaries, final_route_alignment
    )
    focus_matrix = _score_matrix(scores, FOCUS_CONDITION)
    focus_validation_matrix = _primary_matrix(
        primary_rows, FOCUS_CONDITION, "test", "task_aware"
    )
    focus_mismatches = tuple(
        task
        for task in TASK_NAMES
        if not final_route_alignment[FOCUS_CONDITION][task]
    )
    focus_difference_text = (
        "At stage 8 the routers differ only for "
        + " and ".join(focus_mismatches)
        + ": the provenance-minus-validation final deltas are "
        + ", ".join(
            f"{task} {focus_matrix[(TASK_NAMES.index(task) + 1, 8)] - focus_validation_matrix[(TASK_NAMES.index(task) + 1, 8)]:+.3f}"
            for task in focus_mismatches
        )
        + " points. The other six final task scores are identical."
    )
    markdown = f"""# TRACE task-known provenance follow-up

This CPU-only follow-up re-scores the sealed TRACE Log-t VAMP generations with a task-known provenance router. Given task identity, the router selects the live node containing the most arrivals from that task; ties prefer greater node purity and then the most recent node. It never uses prompts, answers, validation scores, or test scores to choose a node.

## Finding

{finding}

{interpretation}

{_markdown_table(summary_headers, summary_rows)}

![OP and forgetting comparison](op-bwt.png)

## What the comparison means

`task_known_provenance` is the fixed lineage rule tested here. `task_known_validation` is the existing result key `task_aware`, renamed in this report because it chooses one validation-best node per known task. Both are task-known controls and require an O(number of tasks) lookup table; neither supports the task-free or O(log T) addressing claim.

Forgetting is meaningful within a fixed router, but its absolute value is not a clean router comparison here. Changing the router changes both the diagonal starting scores and the final scores. The final OP and per-task final deltas are the direct addressing comparison.

The independent taskwise-LoRA reference reaches **{taskwise_op:.3f} OP**. Published CRAFT Llama-3.2-1B reports **{CRAFT_OP:.2f} ± {CRAFT_OP_SPREAD:.2f} OP** and **{CRAFT_BWT:.2f} ± {CRAFT_BWT_SPREAD:.2f} BWT** across three seeds ([CRAFT arXiv v2]({CRAFT_URL})). CRAFT's positive BWT quantity corresponds to this report's `forgetting = diagonal − final`, not the native signed BWT. These are contextual values, not a controlled head-to-head: CRAFT uses LoReFT, different task epochs, a 2e-4 learning rate, zero dropout, effective batch four, and three seeds; this TRACE run uses LoRA/VAMP, different epochs, 1e-4, 0.1 dropout, effective batch eight, and one seed/order.

## Predeclared focus: SVD with 5% repair

This condition was fixed before the new scores were computed because it had the highest pre-existing final validation OP under `task_aware`. It was not selected from the provenance test results.

{focus_difference_text}

{_markdown_table(focus_headers, focus_rows)}

## Route structure

All six VAMP policies share the same logical lineage, so the 36 stage/task decisions below are policy-independent. Coverage is the fraction of a task's five training arrivals represented by the selected node. Purity is the fraction of the selected node's arrivals belonging to that task.

![Task coverage and selected node intervals](route-coverage.png)

<details>
<summary>Complete 36-route audit</summary>

{_markdown_table(route_headers, route_rows)}

</details>

## Verification and limits

- The analysis hash-verified all 1,798 evidence files and cross-checked 432 VAMP candidate JSONLs containing 301,968 candidate/example rows.
- It reconstructed `prompt_nll`, `task_aware`, and `answer_oracle` for every condition, triangular cell, and split: **{reconstruction_checks:,} aggregate checks**, maximum absolute error **{maximum_error:.3g}**, acceptance tolerance **{RECONSTRUCTION_TOLERANCE:.0e}**.
- The frozen-centroid router cannot be reconstructed because prompt embeddings were deliberately excluded from the reviewer bundle.
- This follow-up reuses one completed run, one task order, and the same test generations. It estimates the effect of changing only the lookup rule; it does not estimate seed variation or establish statistical significance.
- No model weights were loaded, no generations were added, and no GPU work was performed.

Machine-readable details are in `scores.csv`, `routes.csv`, `summary.csv`, and `manifest.json`.
"""
    html_text = _html_report(
        finding=finding,
        interpretation=interpretation,
        focus_difference_text=focus_difference_text,
        summary_headers=summary_headers,
        summary_rows=summary_rows,
        focus_headers=focus_headers,
        focus_rows=focus_rows,
        route_headers=route_headers,
        route_rows=route_rows,
        taskwise_op=taskwise_op,
        reconstruction_checks=reconstruction_checks,
        maximum_error=maximum_error,
    )
    return markdown, html_text


def _html_report(
    *,
    finding: str,
    interpretation: str,
    focus_difference_text: str,
    summary_headers: Sequence[str],
    summary_rows: Sequence[Sequence[str]],
    focus_headers: Sequence[str],
    focus_rows: Sequence[Sequence[str]],
    route_headers: Sequence[str],
    route_rows: Sequence[Sequence[str]],
    taskwise_op: float,
    reconstruction_checks: int,
    maximum_error: float,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACE task-known provenance follow-up</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#526174; --line:#d8dee9; --panel:#f5f7fb; --accent:#2357a5; }}
body {{ margin:0; background:#eef2f7; color:var(--ink); font:16px/1.58 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 28px 72px; background:white; box-shadow:0 0 28px #18223518; }}
h1 {{ font-size:2.1rem; line-height:1.15; margin:0 0 18px; }} h2 {{ margin-top:42px; border-bottom:2px solid var(--line); padding-bottom:7px; }}
p,li {{ max-width:90ch; }} .finding {{ border-left:5px solid var(--accent); background:#edf4ff; padding:16px 20px; margin:22px 0; }}
.table-wrap {{ overflow-x:auto; margin:18px 0 24px; }} table {{ border-collapse:collapse; width:100%; font-size:.91rem; }}
th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:right; vertical-align:top; }} th:first-child,td:first-child,td:nth-child(2) {{ text-align:left; }} th {{ background:var(--panel); }}
figure {{ margin:28px 0; }} img {{ width:100%; height:auto; border:1px solid var(--line); }} figcaption {{ color:var(--muted); font-size:.9rem; }}
details {{ border:1px solid var(--line); border-radius:6px; padding:12px 16px; margin:20px 0; }} summary {{ cursor:pointer; font-weight:650; }}
code {{ background:#eef1f5; padding:.1em .3em; border-radius:3px; }} a {{ color:var(--accent); }} .muted {{ color:var(--muted); }}
</style>
</head>
<body><main>
<h1>TRACE task-known provenance follow-up</h1>
<p>This CPU-only follow-up re-scores the sealed TRACE generations with a fixed task-known lineage rule. Node selection uses task-arrival coverage, node purity, and recency—never prompts, answers, validation scores, or test scores.</p>
<section><h2>Finding</h2><div class="finding"><p>{html.escape(finding)}</p><p>{html.escape(interpretation)}</p></div>
{_html_table(summary_headers, summary_rows)}
<figure><img src="data:image/png;base64,{{OP_IMAGE}}" alt="OP versus forgetting"><figcaption>Filled circles use fixed provenance; hollow squares use validation-selected node lookup. Lines connect the same VAMP condition.</figcaption></figure></section>
<section><h2>What the comparison means</h2>
<p><code>task_known_provenance</code> is the fixed lineage rule. <code>task_known_validation</code> is the existing <code>task_aware</code> result renamed to state its information regime. Both are task-known O(number of tasks) controls; neither supports the task-free or O(log T) addressing claim.</p>
<p>Forgetting is meaningful within a fixed router, but changing the router changes both diagonal and final scores. Final OP and per-task final deltas are the direct addressing comparison.</p>
<p>The independent taskwise-LoRA reference reaches <strong>{taskwise_op:.3f} OP</strong>. Published CRAFT Llama-3.2-1B reports <strong>{CRAFT_OP:.2f} ± {CRAFT_OP_SPREAD:.2f} OP</strong> and <strong>{CRAFT_BWT:.2f} ± {CRAFT_BWT_SPREAD:.2f} BWT</strong> across three seeds (<a href="{CRAFT_URL}">CRAFT arXiv v2</a>). CRAFT BWT maps to <code>diagonal − final</code> here. The external row is context only because the adaptation method, task epochs, learning rate, dropout, effective batch, and seed count differ.</p></section>
<section><h2>Predeclared focus: SVD with 5% repair</h2><p>This condition was fixed from the pre-existing validation-selected results before computing the new scores.</p><p>{html.escape(focus_difference_text)}</p>{_html_table(focus_headers, focus_rows)}</section>
<section><h2>Route structure</h2><p>All six policies share these lineage decisions. Each cell gives task coverage and the selected arrival interval.</p>
<figure><img src="data:image/png;base64,{{ROUTE_IMAGE}}" alt="Task coverage heatmap"><figcaption>Coverage counts are out of five task arrivals; future tasks are blank.</figcaption></figure>
<details><summary>Complete 36-route audit</summary>{_html_table(route_headers, route_rows)}</details></section>
<section><h2>Verification and limits</h2><ul>
<li>Hash-verified 1,798 source files and 432 VAMP candidate files containing 301,968 rows.</li>
<li>Passed {reconstruction_checks:,} aggregate router reconstructions; maximum absolute error {maximum_error:.3g}, tolerance {RECONSTRUCTION_TOLERANCE:.0e}.</li>
<li>The frozen-centroid router is not reconstructable because prompt embeddings were deliberately excluded.</li>
<li>This is one completed run, seed, task order, and reused test set. It does not estimate seed variation or establish statistical significance.</li>
<li>No model weights, new inference, or GPU work were used.</li>
</ul><p class="muted">Machine-readable records: scores.csv, routes.csv, summary.csv, and manifest.json.</p></section>
</main></body></html>
"""


def _analysis_source_sha256(bundle_root: Path) -> str:
    docs_root = bundle_root.parents[1]
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("task_known.py").resolve(),
        Path(__file__).with_name("metrics.py").resolve(),
        docs_root / "TRACE_task_known_followup_idea.md",
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("follow-up source or predeclared idea document is missing")
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_artifacts(
    staging_root: Path,
    bundle_root: Path,
    entries: Sequence[CandidateIndexEntry],
    scores: Sequence[FollowupScore],
    routes: Sequence[TaskKnownRoute],
    summaries: Sequence[SummaryRow],
    taskwise_op: float,
    dependency_versions: Mapping[str, str],
    sari_module: str,
    sari_sha256: str,
    source_file_count: int,
    reconstruction_checks: int,
    maximum_error: float,
) -> None:
    staging_root.mkdir(parents=True, exist_ok=False)
    primary_rows = _primary_score_rows(bundle_root)
    final_route_alignment = _final_route_alignment(bundle_root, entries, routes)
    _write_csv(
        staging_root / "scores.csv",
        (
            "condition",
            "policy_hash",
            "split",
            "router",
            "stage",
            "task_index",
            "task",
            "candidate_id",
            "score",
        ),
        tuple(score.as_record() for score in scores),
    )
    _write_csv(
        staging_root / "routes.csv",
        (
            "stage",
            "task_index",
            "task",
            "candidate_id",
            "start_arrival",
            "end_arrival",
            "interval",
            "represented_arrivals",
            "task_arrivals",
            "coverage_count",
            "coverage",
            "purity",
        ),
        tuple(route.as_record() for route in routes),
    )
    _write_csv(
        staging_root / "summary.csv",
        (
            "condition",
            "split",
            "router",
            "source_router",
            "op",
            "forgetting_craft_bwt",
            "bwt_signed",
            "bwt_clipped_negative_only",
        ),
        tuple(row.as_record() for row in summaries),
    )
    _plot_op_bwt(staging_root / "op-bwt.png", summaries, taskwise_op)
    _plot_route_coverage(staging_root / "route-coverage.png", routes)
    markdown, html_text = _report_text(
        scores,
        routes,
        summaries,
        primary_rows,
        final_route_alignment,
        taskwise_op,
        reconstruction_checks,
        maximum_error,
    )
    op_image = base64.b64encode((staging_root / "op-bwt.png").read_bytes()).decode(
        "ascii"
    )
    route_image = base64.b64encode(
        (staging_root / "route-coverage.png").read_bytes()
    ).decode("ascii")
    html_text = html_text.replace("{OP_IMAGE}", op_image).replace(
        "{ROUTE_IMAGE}", route_image
    )
    if "{OP_IMAGE}" in html_text or "{ROUTE_IMAGE}" in html_text:
        raise AssertionError("self-contained HTML image substitution failed")
    (staging_root / "report.md").write_text(markdown, encoding="utf-8")
    (staging_root / "report.html").write_text(html_text, encoding="utf-8")

    run = _run_root(bundle_root)
    policy_hashes = {
        condition: next(
            entry.policy_hash for entry in entries if entry.condition == condition
        )
        for condition in VAMP_CONDITIONS
    }
    component_names = (
        "op-bwt.png",
        "report.html",
        "report.md",
        "route-coverage.png",
        "routes.csv",
        "scores.csv",
        "summary.csv",
    )
    manifest: dict[str, object] = {
        "analysis": {
            "candidate_files": len(entries),
            "candidate_rows": sum(entry.rows for entry in entries),
            "focus_condition": FOCUS_CONDITION,
            "focus_selection_basis": (
                "highest pre-existing stage-8 validation OP under source router task_aware"
            ),
            "final_candidate_matches_task_known_validation": {
                condition: sum(final_route_alignment[condition].values())
                for condition in VAMP_CONDITIONS
            },
            "format": FOLLOWUP_FORMAT,
            "new_score_rows": len(scores),
            "route_rows": len(routes),
            "router": "task_known_provenance",
            "router_rule": ROUTER_RULE,
            "task_arrivals_denominator": ARRIVALS_PER_TASK,
        },
        "environment": {
            "packages": dict(sorted(dependency_versions.items())),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sari_metric_module": sari_module,
            "sari_metric_module_sha256": sari_sha256,
        },
        "external_context": {
            "craft_bwt": CRAFT_BWT,
            "craft_bwt_spread": CRAFT_BWT_SPREAD,
            "craft_op": CRAFT_OP,
            "craft_op_spread": CRAFT_OP_SPREAD,
            "craft_runs": 3,
            "craft_url": CRAFT_URL,
            "comparison_status": "contextual_only_not_controlled",
            "metric_mapping": "CRAFT BWT corresponds to TRACE forgetting=diagonal-final",
        },
        "format": FOLLOWUP_FORMAT,
        "inputs": {
            "analysis_source_sha256": _analysis_source_sha256(bundle_root),
            "arrivals_manifest_sha256": file_sha256(run / "manifests" / "arrivals.json"),
            "candidate_index_sha256": file_sha256(bundle_root / "candidate-index.csv"),
            "dependency_manifest_sha256": file_sha256(
                run / "manifests" / "dependencies.json"
            ),
            "evidence_git_commit": "d727b0c944bdaf733f056b6d4a286c78907b1405",
            "examples_sha256": file_sha256(run / "manifests" / "examples.jsonl"),
            "idea_sha256": file_sha256(
                bundle_root.parents[1] / "TRACE_task_known_followup_idea.md"
            ),
            "policy_hashes": policy_hashes,
            "primary_scores_sha256": file_sha256(
                bundle_root / "final" / "reports" / "primary-scores.csv"
            ),
            "run_id": RUN_ID,
            "run_manifest_sha256": file_sha256(run / "manifests" / "run.json"),
            "source_file_count": source_file_count,
            "source_sha256sums_sha256": file_sha256(
                bundle_root / "evidence-volume" / "SOURCE_SHA256SUMS"
            ),
        },
        "outputs": {
            name: {"bytes": (staging_root / name).stat().st_size, "sha256": file_sha256(staging_root / name)}
            for name in component_names
        },
        "validation": {
            "maximum_absolute_error": maximum_error,
            "reconstructed_routers": ["answer_oracle", "prompt_nll", "task_aware"],
            "reconstruction_checks": reconstruction_checks,
            "tolerance": RECONSTRUCTION_TOLERANCE,
        },
    }
    (staging_root / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _publish_staged_directory(staging_root: Path, output_root: Path) -> None:
    if not output_root.exists():
        os.replace(staging_root, output_root)
        return
    if not output_root.is_dir():
        raise ValueError(f"follow-up output is not a directory: {output_root}")
    if _directory_hashes(staging_root) == _directory_hashes(output_root):
        shutil.rmtree(staging_root)
        return
    backup = output_root.with_name(f".{output_root.name}.previous")
    if backup.exists():
        raise FileExistsError(f"stale follow-up publication backup: {backup}")
    os.replace(output_root, backup)
    try:
        os.replace(staging_root, output_root)
    except BaseException:
        os.replace(backup, output_root)
        raise
    shutil.rmtree(backup)


def build_task_known_followup(
    bundle_root: str | Path,
    output_root: str | Path | None = None,
) -> FollowupResult:
    """Validate, re-score, report, and atomically publish the TRACE follow-up."""
    bundle = Path(bundle_root).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"TRACE reviewer bundle not found: {bundle}")
    destination = (
        Path(output_root).resolve()
        if output_root is not None
        else bundle / "followups" / "task-known-provenance"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="trace-task-known-"))
    matplotlib_config = work_root / "matplotlib"
    matplotlib_config.mkdir()
    previous_matplotlib_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
    print(f"TRACE task-known temporary artifacts: {work_root}", flush=True)
    overall = tqdm(total=4, desc="TRACE follow-up overall", unit="phase", dynamic_ncols=True)
    try:
        print("TRACE follow-up phase 1/4: verify evidence integrity", flush=True)
        entries = load_candidate_index(bundle)
        source_file_count = verify_evidence_bundle(bundle, entries)
        overall.update(1)

        print("TRACE follow-up phase 2/4: verify scorer and cache SARI", flush=True)
        dependency_versions = verify_scoring_environment()
        sari_module, sari_sha256 = prepare_sari_metric()
        overall.update(1)

        print("TRACE follow-up phase 3/4: reconstruct routers and score provenance", flush=True)
        available_cpus = os.cpu_count() or 1
        worker_count = min(len(TASK_NAMES), max(1, math.floor(available_cpus * 0.75)))
        specs = tuple(
            (
                str(bundle),
                condition,
                task,
                str(
                    work_root
                    / f"{condition_index:02d}-{task_index:02d}-{condition}-{task}.jsonl"
                ),
            )
            for condition_index, condition in enumerate(VAMP_CONDITIONS, start=1)
            for task_index, task in enumerate(TASK_NAMES, start=1)
        )
        shard_results: list[TaskShardResult] = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_score_condition_task_shard, *spec): (spec[1], spec[2])
                for spec in specs
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Condition/task shards",
                unit="shard",
                dynamic_ncols=True,
                leave=False,
            ):
                shard_results.append(future.result())
        ordered_shards = tuple(
            sorted(
                shard_results,
                key=lambda result: (
                    TASK_NAMES.index(result.task),
                    result.score_path,
                ),
            )
        )
        scores = _load_shard_scores(ordered_shards)
        reconstruction_checks = sum(
            result.reconstruction_checks for result in ordered_shards
        )
        maximum_error = max(result.maximum_error for result in ordered_shards)
        if reconstruction_checks != 1_296 or maximum_error > RECONSTRUCTION_TOLERANCE:
            raise ValueError("aggregate reconstruction acceptance gate failed")
        overall.update(1)

        print("TRACE follow-up phase 4/4: render and publish reports", flush=True)
        arrivals = _load_json_object(_run_root(bundle) / "manifests" / "arrivals.json")
        routes = build_task_known_routes(
            _string_sequence(arrivals.get("arrival_ids"), "arrival_ids")
        )
        primary_rows = _primary_score_rows(bundle)
        _verify_predeclared_focus(primary_rows)
        summaries = _summaries(scores, primary_rows)
        taskwise_op = _taskwise_op(primary_rows)
        staging_root = Path(
            tempfile.mkdtemp(prefix=".task-known-provenance-stage-", dir=destination.parent)
        )
        staging_root.rmdir()
        _build_artifacts(
            staging_root,
            bundle,
            entries,
            scores,
            routes,
            summaries,
            taskwise_op,
            dependency_versions,
            sari_module,
            sari_sha256,
            source_file_count,
            reconstruction_checks,
            maximum_error,
        )
        _publish_staged_directory(staging_root, destination)
        overall.update(1)
    finally:
        overall.close()
        if previous_matplotlib_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_matplotlib_config
    print(f"TRACE task-known follow-up published: {destination}", flush=True)
    return FollowupResult(
        output_root=destination,
        report_markdown=destination / "report.md",
        report_html=destination / "report.html",
        manifest=destination / "manifest.json",
        summaries=summaries,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the one-path TRACE task-known follow-up command."""
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    if len(values) != 1:
        raise SystemExit(
            "usage: python -m apm.continual.trace.task_known_followup "
            "docs/experiments/trace-logt-vamp"
        )
    result = build_task_known_followup(values[0])
    print(f"Markdown report: {result.report_markdown}")
    print(f"Self-contained HTML report: {result.report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateIndexEntry",
    "FollowupResult",
    "FollowupScore",
    "SummaryRow",
    "build_task_known_followup",
    "load_candidate_index",
    "main",
    "prepare_sari_metric",
    "verify_evidence_bundle",
    "verify_scoring_environment",
]
