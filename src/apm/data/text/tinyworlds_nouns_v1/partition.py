"""Corpus scan, manual review gate, and overlapping noun partition artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import csv
from dataclasses import dataclass
from hashlib import sha256
import heapq
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import numpy as np

from apm.data.text.curricula import (
    TINYSTORIES_DOCUMENT_SEPARATOR,
    TINYSTORIES_TOPICS,
    TINYSTORIES_V2_SOURCE,
    normalize_text,
)
from apm.data.text.tinyworlds_p.contracts import CANONICAL_TOKENIZER_IDENTITY
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    APPROVAL_FORMAT,
    BENCHMARK_ID,
    BREAKDOWN_FORMAT,
    DATA_ROOT,
    PARTITION_FORMAT,
    BaseSelectionStep,
    NounApproval,
    NounBreakdown,
    NounBreakdownRow,
    NounDecision,
    NounEvidence,
    NounPartitionArtifact,
    NounTaskSummary,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.lm.text import TextTokenizer


DECISIONS_FORMAT = "tinyworlds-nouns-decisions-v1"
SCAN_FORMAT = "tinyworlds-nouns-scan-v1"
STORY_LEDGER_FORMAT = "tinyworlds-nouns-story-ledger-v1"
TRAIN_HOLDOUT_NAMESPACE = f"{BENCHMARK_ID}:base-validation"
PROBE_NAMESPACE = f"{BENCHMARK_ID}:probe"
EVIDENCE_NAMESPACE = f"{BENCHMARK_ID}:evidence"
MINIMUM_TASK_TRAIN_STORIES = 256
MINIMUM_TASK_VALIDATION_STORIES = 64
PROBE_STORY_COUNT = 36
BASE_TARGET_COVERAGE = 0.5
BASE_VALIDATION_BUCKET_COUNT = 50
MODEL_POSITION_LIMIT = 2_048
SOURCE_READ_BYTES = 1024 * 1024
PROGRESS_STORY_INTERVAL = 25_000

ProgressCallback = Callable[[str, int, int | None], None]
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class NounApprovalRequired(RuntimeError):
    """Execution reached the mandatory human noun-review boundary."""

    def __init__(self, breakdown: NounBreakdown, review_directory: Path) -> None:
        super().__init__(
            "manual approval is required for noun breakdown "
            f"{breakdown.breakdown_sha256}"
        )
        self.breakdown = breakdown
        self.review_directory = review_directory


@dataclass(frozen=True, slots=True)
class ScannedStory:
    """One normalized unique source story used by CPU fixtures."""

    story_id: str
    source_split: str
    text: str
    token_ids: tuple[int, ...]
    concept_mask: int


def initial_noun_decisions() -> tuple[NounDecision, ...]:
    """Return the complete editable proposal, excluding known bad classes."""
    exclusions = {
        "saw": "Excluded: corpus use is overwhelmingly the past tense of see.",
        "friend": "Excluded: generic relationship noun without a bounded topic class.",
    }
    return tuple(
        NounDecision(
            concept_id=concept.name,
            category=topic.name,
            forms=concept.forms,
            included=concept.name not in exclusions,
            reason=exclusions.get(
                concept.name,
                "Proposed for inclusion pending manual homonym and scope review.",
            ),
        )
        for topic in TINYSTORIES_TOPICS
        for concept in topic.concepts
    )


def decisions_record(decisions: tuple[NounDecision, ...]) -> dict[str, object]:
    """Return the self-checking canonical noun-decision document."""
    _validate_decision_manifest(decisions)
    core = {
        "decisions": [decision.as_record() for decision in decisions],
        "format": DECISIONS_FORMAT,
        "schema_version": 1,
    }
    return {**core, "decisions_sha256": record_sha256(core)}


def publish_initial_noun_decisions(path: str | Path) -> Path:
    """Create the editable decision proposal once, preserving later human edits."""
    target = Path(path)
    if target.is_file():
        decisions = load_noun_decisions(target)
        normalized = canonical_json_bytes(decisions_record(decisions))
        if target.read_bytes() != normalized:
            _atomic_write(target, normalized)
        return target
    if target.exists():
        raise ValueError(f"noun decision path is not a regular file: {target}")
    _atomic_write(target, canonical_json_bytes(decisions_record(initial_noun_decisions())))
    return target


def load_noun_decisions(path: str | Path) -> tuple[NounDecision, ...]:
    """Load the editable decision table; its publisher reseals manual edits."""
    source = Path(path)
    payload = source.read_bytes()
    record = _json_object(payload, "noun decisions")
    if set(record) != {
        "decisions",
        "decisions_sha256",
        "format",
        "schema_version",
    }:
        raise ValueError("noun decision fields changed")
    if (
        record["format"] != DECISIONS_FORMAT
        or record["schema_version"] != 1
    ):
        raise ValueError("noun decision format is invalid")
    raw_decisions = _list(record["decisions"], "noun decisions")
    decisions = tuple(_decision_from_record(item) for item in raw_decisions)
    _validate_decision_manifest(decisions)
    return decisions


def build_breakdown_from_documents(
    train_documents: Sequence[str],
    validation_documents: Sequence[str],
    tokenizer: TextTokenizer,
    decisions: tuple[NounDecision, ...],
    *,
    source_identity: dict[str, object] | None = None,
    tokenizer_identity: dict[str, object] | None = None,
    evidence_count: int = 8,
) -> tuple[NounBreakdown, tuple[ScannedStory, ...]]:
    """Build a complete deterministic breakdown from a small in-memory fixture."""
    if type(evidence_count) is not int or evidence_count <= 0:
        raise ValueError("evidence_count must be positive")
    _validate_decision_manifest(decisions)
    validation_by_id = _unique_fixture_documents(
        validation_documents,
        "validation",
        tokenizer,
        decisions,
    )
    train_by_id = {
        story.story_id: story
        for story in _unique_fixture_documents(
            train_documents,
            "train",
            tokenizer,
            decisions,
        ).values()
        if story.story_id not in validation_by_id
    }
    stories = tuple(validation_by_id.values()) + tuple(train_by_id.values())
    breakdown = _breakdown_from_scanned_stories(
        stories,
        decisions,
        source_identity=source_identity or {"fixture": "in-memory"},
        tokenizer_identity=tokenizer_identity
        or {
            "kind": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
        },
        evidence_count=evidence_count,
    )
    return breakdown, stories


def scan_pinned_noun_breakdown(
    train_path: str | Path,
    validation_path: str | Path,
    tokenizer: TextTokenizer,
    tokenizer_path: str | Path,
    decisions: tuple[NounDecision, ...],
    work_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[NounBreakdown, Path]:
    """Scan both pinned aggregates once into a resumable disk-backed story store."""
    _validate_decision_manifest(decisions)
    train_source = Path(train_path)
    validation_source = Path(validation_path)
    tokenizer_source = Path(tokenizer_path)
    _require_expected_source_name(train_source, TINYSTORIES_V2_SOURCE.train_file.filename)
    _require_expected_source_name(
        validation_source,
        TINYSTORIES_V2_SOURCE.validation_file.filename,
    )
    expected_tokenizer = next(
        item
        for item in CANONICAL_TOKENIZER_IDENTITY.files
        if item.name == "tokenizer.json"
    )
    if (
        tokenizer_source.name != expected_tokenizer.name
        or tokenizer_source.stat().st_size != expected_tokenizer.size_bytes
        or _file_sha256(tokenizer_source) != expected_tokenizer.sha256
        or tokenizer.vocab_size != CANONICAL_TOKENIZER_IDENTITY.vocab_size
    ):
        raise ValueError("noun scan tokenizer differs from the pinned GPT-2 identity")

    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds nouns temporary scan artifacts: {work.resolve()}", flush=True)
    final_database = work / "noun-scan.sqlite3"
    temporary_database = work / ".noun-scan.sqlite3.tmp"
    if temporary_database.exists():
        temporary_database.unlink()
    connection = sqlite3.connect(temporary_database)
    try:
        _initialize_scan_database(connection)
        counters = _empty_scan_counters(decisions)
        for split, path, expected in (
            ("validation", validation_source, TINYSTORIES_V2_SOURCE.validation_file),
            ("train", train_source, TINYSTORIES_V2_SOURCE.train_file),
        ):
            _emit(progress, f"scan-{split}", 0, None)
            source_index = 0
            for source_index, normalized in enumerate(
                _iter_verified_documents(
                    path,
                    expected.size_bytes,
                    expected.sha256,
                    progress=progress,
                    progress_phase=f"scan-{split}-bytes",
                ),
                start=1,
            ):
                _insert_scanned_story(
                    connection,
                    normalized,
                    split,
                    tokenizer,
                    decisions,
                    counters,
                )
                if source_index % PROGRESS_STORY_INTERVAL == 0:
                    connection.commit()
                    _emit(progress, f"scan-{split}", source_index, None)
            connection.commit()
            _emit(progress, f"scan-{split}", source_index, source_index)
        breakdown = _breakdown_from_database(
            connection,
            decisions,
            _source_identity_record(),
            CANONICAL_TOKENIZER_IDENTITY.as_record(),
            counters,
            evidence_count=8,
        )
        scan_record = {
            "breakdown_sha256": breakdown.breakdown_sha256,
            "decisions_sha256": record_sha256(decisions_record(decisions)),
            "format": SCAN_FORMAT,
            "source_sha256": record_sha256(_source_identity_record()),
        }
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("scan", canonical_json_bytes(scan_record).decode("utf-8")),
        )
        connection.commit()
    except BaseException:
        connection.close()
        if temporary_database.exists():
            temporary_database.unlink()
        raise
    connection.close()
    os.replace(temporary_database, final_database)
    return breakdown, final_database


def load_scanned_noun_breakdown(
    scan_database: str | Path,
    decisions: tuple[NounDecision, ...],
    output_root: str | Path = DATA_ROOT,
) -> NounBreakdown:
    """Reuse a complete scan only when its decisions, sources, and packet still bind."""
    _validate_decision_manifest(decisions)
    database = Path(scan_database)
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        scan = _scan_metadata(connection)
    finally:
        connection.close()
    if (
        scan.get("decisions_sha256") != record_sha256(decisions_record(decisions))
        or scan.get("source_sha256") != record_sha256(_source_identity_record())
    ):
        raise ValueError("stored noun scan belongs to different decisions or sources")
    breakdown_sha256 = _text(scan.get("breakdown_sha256"), "scan breakdown")
    breakdown = load_noun_breakdown(
        Path(output_root)
        / "noun-breakdowns"
        / breakdown_sha256
        / "noun-breakdown.json"
    )
    if breakdown.source_identity != _source_identity_record():
        raise ValueError("stored noun scan source identity changed")
    return breakdown


def publish_noun_breakdown(
    breakdown: NounBreakdown,
    decisions: tuple[NounDecision, ...],
    output_root: str | Path = DATA_ROOT,
) -> Path:
    """Atomically publish the JSON, Markdown, and standalone HTML review packet."""
    _validate_decision_manifest(decisions)
    target = Path(output_root) / "noun-breakdowns" / breakdown.breakdown_sha256
    if target.is_dir():
        _require_breakdown_packet(target, breakdown, decisions)
        return target
    if target.exists():
        raise ValueError(f"noun breakdown target is not a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        markdown = render_noun_breakdown_markdown(breakdown)
        payloads = {
            "noun-breakdown.json": canonical_json_bytes(breakdown.as_record()),
            "noun-breakdown.md": markdown.encode("utf-8"),
            "noun-breakdown.html": render_noun_breakdown_html(breakdown).encode(
                "utf-8"
            ),
            "decisions.json": canonical_json_bytes(decisions_record(decisions)),
        }
        for name, payload in payloads.items():
            _write_fsync(temporary / name, payload)
        manifest_core = {
            "breakdown_sha256": breakdown.breakdown_sha256,
            "files": {
                name: sha256(payload).hexdigest()
                for name, payload in sorted(payloads.items())
            },
            "format": BREAKDOWN_FORMAT,
        }
        _write_fsync(
            temporary / "manifest.json",
            canonical_json_bytes(
                {**manifest_core, "manifest_sha256": record_sha256(manifest_core)}
            ),
        )
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _require_breakdown_packet(target, breakdown, decisions)
    return target


def _require_breakdown_packet(
    root: Path,
    breakdown: NounBreakdown,
    decisions: tuple[NounDecision, ...],
) -> None:
    payloads = {
        "decisions.json": canonical_json_bytes(decisions_record(decisions)),
        "noun-breakdown.html": render_noun_breakdown_html(breakdown).encode("utf-8"),
        "noun-breakdown.json": canonical_json_bytes(breakdown.as_record()),
        "noun-breakdown.md": render_noun_breakdown_markdown(breakdown).encode("utf-8"),
    }
    if {path.name for path in root.iterdir()} != {*payloads, "manifest.json"}:
        raise ValueError("noun breakdown packet entries changed")
    manifest = _canonical_object(
        (root / "manifest.json").read_bytes(),
        "noun breakdown manifest",
    )
    supplied = manifest.pop("manifest_sha256", None)
    expected_files = {
        name: sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    if (
        set(manifest) != {"breakdown_sha256", "files", "format"}
        or supplied != record_sha256(manifest)
        or manifest.get("breakdown_sha256") != breakdown.breakdown_sha256
        or manifest.get("format") != BREAKDOWN_FORMAT
        or manifest.get("files") != expected_files
        or any((root / name).read_bytes() != payload for name, payload in payloads.items())
    ):
        raise ValueError("noun breakdown packet changed")


def load_noun_breakdown(path: str | Path) -> NounBreakdown:
    """Strictly load a published noun breakdown JSON document."""
    record = _canonical_object(Path(path).read_bytes(), "noun breakdown")
    expected = {
        "base_selection",
        "breakdown_sha256",
        "format",
        "rows",
        "schema_version",
        "source_identity",
        "tokenizer_identity",
        "train_token_count",
        "train_unique_story_count",
        "validation_token_count",
        "validation_unique_story_count",
    }
    if set(record) != expected:
        raise ValueError("noun breakdown fields changed")
    rows = tuple(_breakdown_row_from_record(item) for item in _list(record["rows"], "rows"))
    steps = tuple(
        _base_step_from_record(item)
        for item in _list(record["base_selection"], "base selection")
    )
    breakdown = NounBreakdown(
        source_identity=_object(record["source_identity"], "source identity"),
        tokenizer_identity=_object(record["tokenizer_identity"], "tokenizer identity"),
        train_unique_story_count=_integer(
            record["train_unique_story_count"], "train unique stories"
        ),
        validation_unique_story_count=_integer(
            record["validation_unique_story_count"], "validation unique stories"
        ),
        train_token_count=_integer(record["train_token_count"], "train tokens"),
        validation_token_count=_integer(
            record["validation_token_count"], "validation tokens"
        ),
        rows=rows,
        base_selection=steps,
    )
    if (
        record["format"] != BREAKDOWN_FORMAT
        or record["schema_version"] != 1
        or record["breakdown_sha256"] != breakdown.breakdown_sha256
    ):
        raise ValueError("noun breakdown identity changed")
    return breakdown


def approve_noun_breakdown(
    breakdown: NounBreakdown,
    decisions: tuple[NounDecision, ...],
    requested_sha256: str,
    output_root: str | Path = DATA_ROOT,
) -> Path:
    """Record manual approval only when the requested and rebuilt hashes agree."""
    require_sha256(requested_sha256, "requested noun approval")
    if requested_sha256 != breakdown.breakdown_sha256:
        raise ValueError(
            "requested noun breakdown does not match the current review packet"
        )
    approval = NounApproval(
        breakdown_sha256=breakdown.breakdown_sha256,
        decision_sha256=record_sha256(decisions_record(decisions)),
        source_sha256=record_sha256(breakdown.source_identity),
    )
    target = Path(output_root) / "noun-approvals" / f"{approval.approval_sha256}.json"
    payload = canonical_json_bytes(approval.as_record())
    if target.is_file():
        if target.read_bytes() != payload:
            raise ValueError("existing noun approval bytes changed")
        return target
    _atomic_write(target, payload)
    return target


def require_noun_approval(
    breakdown: NounBreakdown,
    decisions: tuple[NounDecision, ...],
    output_root: str | Path = DATA_ROOT,
) -> NounApproval:
    """Return the sole matching manual approval or stop before GPU execution."""
    root = Path(output_root) / "noun-approvals"
    candidates = tuple(sorted(root.glob("*.json"))) if root.is_dir() else ()
    matching = tuple(
        approval
        for path in candidates
        for approval in (_load_noun_approval(path),)
        if approval.breakdown_sha256 == breakdown.breakdown_sha256
    )
    expected_decisions = record_sha256(decisions_record(decisions))
    expected_source = record_sha256(breakdown.source_identity)
    valid = tuple(
        approval
        for approval in matching
        if approval.decision_sha256 == expected_decisions
        and approval.source_sha256 == expected_source
    )
    if len(valid) != 1:
        review_directory = (
            Path(output_root) / "noun-breakdowns" / breakdown.breakdown_sha256
        )
        raise NounApprovalRequired(breakdown, review_directory)
    return valid[0]


def render_noun_breakdown_markdown(breakdown: NounBreakdown) -> str:
    """Render the human decision packet with all noun families and provenance."""
    lines = [
        "# TinyWorlds noun breakdown — manual approval required",
        "",
        f"Breakdown SHA-256: `{breakdown.breakdown_sha256}`",
        "",
        "No training or GPU initialization is authorized by this file alone. Review "
        "every noun, its exact forms, and the complete-story examples before recording "
        "approval.",
        "",
        f"Unique training stories: {breakdown.train_unique_story_count:,}",
        f"Unique official-validation stories: {breakdown.validation_unique_story_count:,}",
        "",
        "## Greedy base projection",
        "",
        "| noun | noun stories | new stories | cumulative story coverage | cumulative token coverage |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {step.concept_id} | {step.noun_story_count:,} | "
            f"{step.new_story_count:,} | {step.cumulative_story_coverage:.2%} | "
            f"{step.cumulative_token_coverage:.2%} |"
            for step in breakdown.base_selection
        ),
        "",
        "## Noun families",
        "",
    ]
    for row in breakdown.rows:
        decision = row.decision
        lines.extend(
            (
                f"<details><summary><strong>{decision.concept_id}</strong> — "
                f"{row.projected_role}; train {row.train_story_count:,}; "
                f"validation {row.validation_story_count:,}</summary>",
                "",
                f"- Category: `{decision.category}`",
                f"- Exact forms: {', '.join(f'`{form}`' for form in decision.forms)}",
                f"- Proposal: {'include' if decision.included else 'exclude'} — {decision.reason}",
                f"- Task threshold met: {'yes' if row.threshold_eligible else 'no'}",
                f"- Train prevalence: {row.train_prevalence:.3%}",
                f"- Validation prevalence: {row.validation_prevalence:.3%}",
                "",
                "Per-form story counts: "
                + "; ".join(
                    f"{form}={train:,} train/{valid:,} validation"
                    for (form, train), (_, valid) in zip(
                        row.train_form_counts,
                        row.validation_form_counts,
                    )
                ),
                "",
                *(
                    f"- `{item.source_split}` `{item.story_id}`; matched "
                    f"{', '.join(item.matched_forms)}\n\n  {item.story}"
                    for item in row.evidence
                ),
                "",
                "</details>",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def render_noun_breakdown_html(breakdown: NounBreakdown) -> str:
    """Render a standalone folding review page without external dependencies."""
    base_rows = "".join(
        "<tr>"
        f"<td>{escape(step.concept_id)}</td><td>{step.noun_story_count:,}</td>"
        f"<td>{step.new_story_count:,}</td>"
        f"<td>{step.cumulative_story_coverage:.2%}</td>"
        f"<td>{step.cumulative_token_coverage:.2%}</td></tr>"
        for step in breakdown.base_selection
    )
    rows = "".join(
        "<details><summary>"
        f"<b>{escape(row.decision.concept_id)}</b> — {row.projected_role}; "
        f"{row.train_story_count:,} train / {row.validation_story_count:,} validation"
        "</summary>"
        f"<p><b>Forms:</b> {escape(', '.join(row.decision.forms))}</p>"
        f"<p><b>Proposal:</b> {'include' if row.decision.included else 'exclude'} — "
        f"{escape(row.decision.reason)}</p>"
        f"<p><b>Prevalence:</b> {row.train_prevalence:.3%} train; "
        f"{row.validation_prevalence:.3%} validation. "
        f"<b>Task threshold:</b> {'met' if row.threshold_eligible else 'not met'}.</p>"
        "<p><b>Per-form counts:</b> "
        + escape(
            "; ".join(
                f"{form}={train:,} train/{valid:,} validation"
                for (form, train), (_, valid) in zip(
                    row.train_form_counts,
                    row.validation_form_counts,
                )
            )
        )
        + "</p>"
        + "".join(
            "<article><code>"
            f"{item.source_split} {item.story_id}</code><p>{escape(item.story)}</p></article>"
            for item in row.evidence
        )
        + "</details>"
        for row in breakdown.rows
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TinyWorlds noun review</title><style>"
        "body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "details{border:1px solid #ccd5df;border-radius:8px;margin:.7rem 0;padding:.7rem}"
        "summary{cursor:pointer}article{background:#f5f7fa;padding:.5rem;margin:.5rem 0}"
        "code{font-size:.78rem;word-break:break-all}</style></head><body>"
        "<h1>TinyWorlds noun breakdown</h1>"
        f"<p><b>Manual approval required.</b> Exact hash: <code>{breakdown.breakdown_sha256}</code></p>"
        f"<p>{breakdown.train_unique_story_count:,} unique training stories; "
        f"{breakdown.validation_unique_story_count:,} official-validation stories.</p>"
        "<h2>Projected greedy base</h2><table><tr><th>Noun</th><th>Noun stories</th>"
        "<th>New union stories</th><th>Story coverage</th><th>Token coverage</th></tr>"
        f"{base_rows}</table><h2>All noun families</h2>"
        f"{rows}</body></html>"
    )


def match_noun_forms(
    text: str,
    decisions: tuple[NounDecision, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return every noun family matched by exact case-insensitive whole words."""
    _validate_decision_manifest(decisions)
    words = frozenset(_WORD_PATTERN.findall(normalize_text(text).casefold()))
    return tuple(
        (decision.concept_id, matched)
        for decision in decisions
        for matched in (
            tuple(form for form in decision.forms if form in words),
        )
        if matched
    )


def build_noun_partition(
    breakdown: NounBreakdown,
    approval: NounApproval,
    decisions: tuple[NounDecision, ...],
    scan_database: str | Path,
    output_root: str | Path = DATA_ROOT,
) -> NounPartitionArtifact:
    """Publish one overlapping partition from its approved disk-backed scan."""
    _validate_decision_manifest(decisions)
    if (
        approval.breakdown_sha256 != breakdown.breakdown_sha256
        or approval.decision_sha256 != record_sha256(decisions_record(decisions))
        or approval.source_sha256 != record_sha256(breakdown.source_identity)
    ):
        raise ValueError("partition approval does not bind the current noun breakdown")
    database = Path(scan_database)
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        scan = _scan_metadata(connection)
        if (
            scan.get("breakdown_sha256") != breakdown.breakdown_sha256
            or scan.get("decisions_sha256")
            != record_sha256(decisions_record(decisions))
            or scan.get("source_sha256") != record_sha256(breakdown.source_identity)
        ):
            raise ValueError("noun scan does not bind the approved breakdown")
        selections = _select_partition_stories(
            connection,
            breakdown,
            decisions,
        )
        target_parent = Path(output_root) / "partitions"
        target_parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".partition.tmp-", dir=target_parent))
        try:
            logical = _write_partition_payloads(
                connection,
                temporary,
                breakdown,
                approval,
                decisions,
                selections,
            )
            file_records = {
                path.relative_to(temporary).as_posix(): {
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            core = {
                **logical,
                "files": file_records,
                "format": PARTITION_FORMAT,
                "schema_version": 1,
            }
            partition_sha256 = record_sha256(core)
            _write_fsync(
                temporary / "partition.json",
                canonical_json_bytes(
                    {**core, "partition_sha256": partition_sha256}
                ),
            )
            target = target_parent / partition_sha256
            if target.is_dir():
                shutil.rmtree(temporary)
            elif target.exists():
                raise ValueError(f"partition target is not a directory: {target}")
            else:
                os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    finally:
        connection.close()
    return load_noun_partition(target / "partition.json")


def load_noun_partition(path: str | Path) -> NounPartitionArtifact:
    """Load and byte-verify one content-addressed noun partition."""
    manifest_path = Path(path)
    root = manifest_path.parent
    record = _canonical_object(manifest_path.read_bytes(), "noun partition")
    if "partition_sha256" not in record:
        raise ValueError("noun partition has no content identity")
    supplied_sha256 = record.pop("partition_sha256")
    if (
        type(supplied_sha256) is not str
        or supplied_sha256 != record_sha256(record)
        or root.name != supplied_sha256
        or record.get("format") != PARTITION_FORMAT
        or record.get("schema_version") != 1
    ):
        raise ValueError("noun partition content identity changed")
    expected_fields = {
        "approval_sha256",
        "base_concept_ids",
        "base_train_story_count",
        "base_validation_story_count",
        "breakdown_sha256",
        "eos_token_id",
        "files",
        "format",
        "pad_token_id",
        "root_probe_story_ids",
        "schema_version",
        "source_identity",
        "story_count",
        "task_ids",
        "tasks",
        "token_count",
        "tokenizer_identity",
    }
    if set(record) != expected_fields:
        raise ValueError("noun partition fields changed")
    files = _object(record.get("files"), "partition files")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(files) != actual_files:
        raise ValueError("noun partition directory entries changed")
    for relative, raw_file in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("noun partition file path escapes its root")
        file_record = _object(raw_file, f"partition file {relative}")
        file_path = root / relative_path
        if (
            not file_path.is_file()
            or file_path.stat().st_size
            != _integer(file_record.get("size_bytes"), f"{relative} size")
            or _file_sha256(file_path) != file_record.get("sha256")
        ):
            raise ValueError(f"noun partition file changed: {relative}")
    raw_tasks = _list(record.get("tasks"), "partition tasks")
    tasks = tuple(_task_summary_from_record(item) for item in raw_tasks)
    root_probe_ids = tuple(
        _text(value, "root probe")
        for value in _list(record.get("root_probe_story_ids"), "root probes")
    )
    if (
        _index_story_ids(root / "indexes" / "base-train.jsonl", materialize=False)
        != _integer(record.get("base_train_story_count"), "base train count")
        or _index_story_ids(
            root / "indexes" / "base-validation.jsonl", materialize=False
        )
        != _integer(record.get("base_validation_story_count"), "base validation count")
        or _index_story_ids(root / "indexes" / "root-probes.jsonl")
        != frozenset(root_probe_ids)
        or any(
            _index_story_ids(
                root / "indexes" / f"task-{task.task_id}-train.jsonl",
                materialize=False,
            )
            != task.update_story_count
            or _index_story_ids(
                root / "indexes" / f"task-{task.task_id}-validation.jsonl",
                materialize=False,
            )
            != task.validation_story_count
            or _index_story_ids(
                root / "indexes" / f"task-{task.task_id}-generation.jsonl",
                materialize=False,
            )
            != task.generation_story_count
            or _index_story_ids(
                root / "indexes" / f"task-{task.task_id}-probes.jsonl"
            )
            != frozenset(task.probe_story_ids)
            for task in tasks
        )
    ):
        raise ValueError("noun partition indexes and logical counts differ")
    return NounPartitionArtifact(
        root=root,
        partition_sha256=supplied_sha256,
        breakdown_sha256=_text(record.get("breakdown_sha256"), "breakdown hash"),
        approval_sha256=_text(record.get("approval_sha256"), "approval hash"),
        source_identity=_object(record.get("source_identity"), "partition source"),
        tokenizer_identity=_object(
            record.get("tokenizer_identity"), "partition tokenizer"
        ),
        pad_token_id=_integer(record.get("pad_token_id"), "pad token ID"),
        eos_token_id=_integer(record.get("eos_token_id"), "EOS token ID"),
        base_concept_ids=tuple(
            _text(value, "base concept")
            for value in _list(record.get("base_concept_ids"), "base concepts")
        ),
        task_ids=tuple(
            _text(value, "task noun")
            for value in _list(record.get("task_ids"), "task IDs")
        ),
        base_train_story_count=_integer(
            record.get("base_train_story_count"), "base train count"
        ),
        base_validation_story_count=_integer(
            record.get("base_validation_story_count"), "base validation count"
        ),
        root_probe_story_ids=root_probe_ids,
        tasks=tasks,
    )


@dataclass(slots=True)
class _ScanCounters:
    story_counts: dict[str, int]
    token_counts: dict[str, int]
    concept_counts: dict[str, dict[str, int]]
    form_counts: dict[str, dict[str, dict[str, int]]]
    evidence_heaps: dict[str, list[tuple[int, str, NounEvidence]]]
    evidence_limit: int


@dataclass(frozen=True, slots=True)
class _PartitionSelections:
    base_train_count: int
    base_validation_count: int
    root_probe_ids: tuple[str, ...]
    task_summaries: tuple[NounTaskSummary, ...]


def _empty_scan_counters(
    decisions: tuple[NounDecision, ...],
    evidence_limit: int = 8,
) -> _ScanCounters:
    return _ScanCounters(
        story_counts={"train": 0, "validation": 0},
        token_counts={"train": 0, "validation": 0},
        concept_counts={
            split: {decision.concept_id: 0 for decision in decisions}
            for split in ("train", "validation")
        },
        form_counts={
            split: {
                decision.concept_id: {form: 0 for form in decision.forms}
                for decision in decisions
            }
            for split in ("train", "validation")
        },
        evidence_heaps={decision.concept_id: [] for decision in decisions},
        evidence_limit=evidence_limit,
    )


def _validate_decision_manifest(decisions: tuple[NounDecision, ...]) -> None:
    expected = tuple(
        (topic.name, concept.name)
        for topic in TINYSTORIES_TOPICS
        for concept in topic.concepts
    )
    actual = tuple((decision.category, decision.concept_id) for decision in decisions)
    if type(decisions) is not tuple or actual != expected:
        raise ValueError("noun decisions must preserve the complete topic catalog order")
    forms = tuple(
        (form, decision.concept_id)
        for decision in decisions
        for form in decision.forms
    )
    if len({form for form, _ in forms}) != len(forms):
        raise ValueError("one exact surface form cannot belong to multiple noun classes")


def _unique_fixture_documents(
    documents: Sequence[str],
    split: str,
    tokenizer: TextTokenizer,
    decisions: tuple[NounDecision, ...],
) -> dict[str, ScannedStory]:
    result: dict[str, ScannedStory] = {}
    for raw_text in documents:
        text = normalize_text(raw_text)
        if not text:
            continue
        story_id = sha256(text.encode("utf-8")).hexdigest()
        matches = match_noun_forms(text, decisions)
        mask = sum(
            1 << index
            for index, decision in enumerate(decisions)
            if any(name == decision.concept_id for name, _ in matches)
        )
        story = ScannedStory(
            story_id,
            split,
            text,
            tokenizer.encode(text, add_eos=True),
            mask,
        )
        existing = result.get(story_id)
        if existing is not None and existing.text != text:
            raise RuntimeError("SHA-256 collision in fixture stories")
        result.setdefault(story_id, story)
    return result


def _breakdown_from_scanned_stories(
    stories: Sequence[ScannedStory],
    decisions: tuple[NounDecision, ...],
    *,
    source_identity: dict[str, object],
    tokenizer_identity: dict[str, object],
    evidence_count: int,
) -> NounBreakdown:
    counters = _empty_scan_counters(decisions, evidence_count)
    for story in stories:
        matches = match_noun_forms(story.text, decisions)
        _record_story_counts(
            counters,
            story.source_split,
            story.story_id,
            story.text,
            story.token_ids,
            matches,
        )
    return _assemble_breakdown(
        decisions,
        source_identity,
        tokenizer_identity,
        counters,
        (
            (story.source_split, story.concept_mask, len(story.token_ids))
            for story in stories
        ),
    )


def _breakdown_from_database(
    connection: sqlite3.Connection,
    decisions: tuple[NounDecision, ...],
    source_identity: dict[str, object],
    tokenizer_identity: dict[str, object],
    counters: _ScanCounters,
    *,
    evidence_count: int,
) -> NounBreakdown:
    if counters.evidence_limit != evidence_count:
        raise ValueError("scan evidence count changed")
    return _assemble_breakdown(
        decisions,
        source_identity,
        tokenizer_identity,
        counters,
        (
            (str(split), int(mask), int(token_count))
            for split, mask, token_count in connection.execute(
                "SELECT source_split, concept_mask, token_count FROM stories"
            )
        ),
    )


def _assemble_breakdown(
    decisions: tuple[NounDecision, ...],
    source_identity: dict[str, object],
    tokenizer_identity: dict[str, object],
    counters: _ScanCounters,
    memberships: Iterable[tuple[str, int, int]],
) -> NounBreakdown:
    train_count = counters.story_counts["train"]
    validation_count = counters.story_counts["validation"]
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("noun breakdown requires nonempty train and validation sources")
    included_indices = tuple(
        index for index, decision in enumerate(decisions) if decision.included
    )
    ordered_indices = tuple(
        sorted(
            included_indices,
            key=lambda index: (
                -counters.concept_counts["train"][decisions[index].concept_id],
                decisions[index].concept_id,
            ),
        )
    )
    rank_by_index = {concept_index: rank for rank, concept_index in enumerate(ordered_indices)}
    new_story_counts = [0] * len(ordered_indices)
    new_token_counts = [0] * len(ordered_indices)
    for split, concept_mask, token_count in memberships:
        if split != "train":
            continue
        matching_ranks = tuple(
            rank_by_index[index]
            for index in included_indices
            if concept_mask & (1 << index)
        )
        if matching_ranks:
            first_rank = min(matching_ranks)
            new_story_counts[first_rank] += 1
            new_token_counts[first_rank] += token_count
    cumulative_stories = 0
    cumulative_tokens = 0
    steps: list[BaseSelectionStep] = []
    for rank, concept_index in enumerate(ordered_indices):
        cumulative_stories += new_story_counts[rank]
        cumulative_tokens += new_token_counts[rank]
        concept_id = decisions[concept_index].concept_id
        steps.append(
            BaseSelectionStep(
                concept_id=concept_id,
                noun_story_count=counters.concept_counts["train"][concept_id],
                new_story_count=new_story_counts[rank],
                cumulative_story_count=cumulative_stories,
                cumulative_story_coverage=cumulative_stories / train_count,
                new_token_count=new_token_counts[rank],
                cumulative_token_count=cumulative_tokens,
                cumulative_token_coverage=(
                    cumulative_tokens / counters.token_counts["train"]
                ),
            )
        )
        if cumulative_stories / train_count >= BASE_TARGET_COVERAGE:
            break
    if not steps or steps[-1].cumulative_story_coverage < BASE_TARGET_COVERAGE:
        raise ValueError("approved nouns cannot reach the required 50% base coverage")
    base_ids = {step.concept_id for step in steps}
    rows = tuple(
        _breakdown_row(decision, counters, train_count, validation_count, base_ids)
        for decision in decisions
    )
    return NounBreakdown(
        source_identity=source_identity,
        tokenizer_identity=tokenizer_identity,
        train_unique_story_count=train_count,
        validation_unique_story_count=validation_count,
        train_token_count=counters.token_counts["train"],
        validation_token_count=counters.token_counts["validation"],
        rows=rows,
        base_selection=tuple(steps),
    )


def _breakdown_row(
    decision: NounDecision,
    counters: _ScanCounters,
    train_total: int,
    validation_total: int,
    base_ids: set[str],
) -> NounBreakdownRow:
    train_count = counters.concept_counts["train"][decision.concept_id]
    validation_count = counters.concept_counts["validation"][decision.concept_id]
    threshold = (
        train_count >= MINIMUM_TASK_TRAIN_STORIES
        and validation_count >= MINIMUM_TASK_VALIDATION_STORIES
    )
    role = (
        "excluded"
        if not decision.included
        else "base"
        if decision.concept_id in base_ids
        else "task"
        if threshold
        else "below_threshold"
    )
    evidence = tuple(
        item[2]
        for item in sorted(
            counters.evidence_heaps[decision.concept_id],
            key=lambda value: (-value[0], value[1]),
        )
    )
    return NounBreakdownRow(
        decision=decision,
        train_story_count=train_count,
        validation_story_count=validation_count,
        train_form_counts=tuple(
            (form, counters.form_counts["train"][decision.concept_id][form])
            for form in decision.forms
        ),
        validation_form_counts=tuple(
            (
                form,
                counters.form_counts["validation"][decision.concept_id][form],
            )
            for form in decision.forms
        ),
        train_prevalence=train_count / train_total,
        validation_prevalence=validation_count / validation_total,
        threshold_eligible=threshold,
        projected_role=role,
        evidence=evidence,
    )


def _initialize_scan_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE stories("
        "story_id TEXT PRIMARY KEY, source_split TEXT NOT NULL, text BLOB NOT NULL, "
        "token_ids BLOB NOT NULL, token_count INTEGER NOT NULL, "
        "concept_mask INTEGER NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )


def _insert_scanned_story(
    connection: sqlite3.Connection,
    text: str,
    split: str,
    tokenizer: TextTokenizer,
    decisions: tuple[NounDecision, ...],
    counters: _ScanCounters,
) -> None:
    normalized = normalize_text(text)
    if not normalized:
        return
    story_bytes = normalized.encode("utf-8")
    story_id = sha256(story_bytes).hexdigest()
    matches = match_noun_forms(normalized, decisions)
    matched_ids = {concept_id for concept_id, _ in matches}
    concept_mask = sum(
        1 << index
        for index, decision in enumerate(decisions)
        if decision.concept_id in matched_ids
    )
    token_ids = tokenizer.encode(normalized, add_eos=True)
    token_payload = np.asarray(token_ids, dtype="<u2").tobytes()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO stories VALUES (?, ?, ?, ?, ?, ?)",
        (story_id, split, story_bytes, token_payload, len(token_ids), concept_mask),
    )
    if cursor.rowcount:
        _record_story_counts(
            counters,
            split,
            story_id,
            normalized,
            token_ids,
            matches,
        )


def _record_story_counts(
    counters: _ScanCounters,
    split: str,
    story_id: str,
    text: str,
    token_ids: Sequence[int],
    matches: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    counters.story_counts[split] += 1
    counters.token_counts[split] += len(token_ids)
    for concept_id, forms in matches:
        counters.concept_counts[split][concept_id] += 1
        for form in forms:
            counters.form_counts[split][concept_id][form] += 1
        if split not in ("train", "validation"):
            raise ValueError("scan split must be train or validation")
        evidence = NounEvidence(
            story_id,
            "train" if split == "train" else "validation",
            forms,
            text,
        )
        priority = int(
            sha256(
                f"{EVIDENCE_NAMESPACE}\0{concept_id}\0{story_id}".encode("utf-8")
            ).hexdigest(),
            16,
        )
        heap = counters.evidence_heaps[concept_id]
        candidate = (-priority, story_id, evidence)
        if len(heap) < counters.evidence_limit:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)


def _iter_verified_documents(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    progress: ProgressCallback | None = None,
    progress_phase: str = "scan-source-bytes",
) -> Iterator[str]:
    separator = TINYSTORIES_DOCUMENT_SEPARATOR.encode("utf-8")
    pending = b""
    digest = sha256()
    measured_size = 0
    with path.open("rb") as source:
        while block := source.read(SOURCE_READ_BYTES):
            digest.update(block)
            measured_size += len(block)
            pieces = (pending + block).split(separator)
            for payload in pieces[:-1]:
                normalized = normalize_text(payload.decode("utf-8", errors="strict"))
                if normalized:
                    yield normalized
            pending = pieces[-1]
            _emit(progress, progress_phase, measured_size, expected_size)
    normalized = normalize_text(pending.decode("utf-8", errors="strict"))
    if normalized:
        yield normalized
    if measured_size != expected_size or digest.hexdigest() != expected_sha256:
        raise ValueError(f"pinned TinyStories source changed: {path}")


def _source_identity_record() -> dict[str, object]:
    source = TINYSTORIES_V2_SOURCE
    return {
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "train": {
            "filename": source.train_file.filename,
            "sha256": source.train_file.sha256,
            "size_bytes": source.train_file.size_bytes,
        },
        "validation": {
            "filename": source.validation_file.filename,
            "sha256": source.validation_file.sha256,
            "size_bytes": source.validation_file.size_bytes,
        },
    }


def _select_partition_stories(
    connection: sqlite3.Connection,
    breakdown: NounBreakdown,
    decisions: tuple[NounDecision, ...],
) -> _PartitionSelections:
    index_by_id = {
        decision.concept_id: index for index, decision in enumerate(decisions)
    }
    base_mask = sum(1 << index_by_id[name] for name in breakdown.base_concept_ids)
    task_ids = breakdown.task_ids
    task_masks = {name: 1 << index_by_id[name] for name in task_ids}
    root_heap: list[tuple[int, str]] = []
    task_probe_heaps = {name: [] for name in task_ids}
    train_counts = {name: 0 for name in task_ids}
    validation_counts = {name: 0 for name in task_ids}
    generation_counts = {name: 0 for name in task_ids}
    base_overlap = {name: 0 for name in task_ids}
    overlap = {name: {other: 0 for other in task_ids} for name in task_ids}
    base_train_count = 0
    base_validation_count = 0
    for story_id, split, token_count, concept_mask in connection.execute(
        "SELECT story_id, source_split, token_count, concept_mask FROM stories"
    ):
        story_id = str(story_id)
        split = str(split)
        token_count = int(token_count)
        concept_mask = int(concept_mask)
        in_base = bool(concept_mask & base_mask)
        held_out = _hash_bucket(TRAIN_HOLDOUT_NAMESPACE, story_id, 50) == 0
        if in_base and split == "train":
            if held_out:
                base_validation_count += 1
            else:
                base_train_count += 1
                if 2 <= token_count <= 257:
                    _push_lowest_id(root_heap, PROBE_STORY_COUNT, "root", story_id)
        matched_tasks = tuple(
            name for name in task_ids if concept_mask & task_masks[name]
        )
        for name in matched_tasks:
            if split == "train":
                train_counts[name] += 1
                if in_base:
                    base_overlap[name] += 1
                if 2 <= token_count <= 257:
                    _push_lowest_id(
                        task_probe_heaps[name],
                        PROBE_STORY_COUNT,
                        f"task:{name}",
                        story_id,
                    )
            else:
                validation_counts[name] += 1
                generation_counts[name] += int(_generation_eligible(token_count))
            if split == "train":
                for other in matched_tasks:
                    overlap[name][other] += 1
    root_ids = _ordered_heap_ids(root_heap)
    if len(root_ids) != PROBE_STORY_COUNT:
        raise ValueError("base has fewer than 36 context-fitting root probes")
    summaries = tuple(
        NounTaskSummary(
            task_id=name,
            train_story_count=train_counts[name],
            update_story_count=train_counts[name] - PROBE_STORY_COUNT,
            validation_story_count=validation_counts[name],
            generation_story_count=generation_counts[name],
            probe_story_ids=_ordered_heap_ids(task_probe_heaps[name]),
            base_overlap_story_count=base_overlap[name],
            overlap_counts=tuple(sorted(overlap[name].items())),
        )
        for name in task_ids
    )
    return _PartitionSelections(
        base_train_count,
        base_validation_count,
        root_ids,
        summaries,
    )


def _write_partition_payloads(
    connection: sqlite3.Connection,
    root: Path,
    breakdown: NounBreakdown,
    approval: NounApproval,
    decisions: tuple[NounDecision, ...],
    selections: _PartitionSelections,
) -> dict[str, object]:
    indexes = root / "indexes"
    indexes.mkdir()
    story_stream = (root / "stories.bin").open("wb")
    token_stream = (root / "tokens.uint16").open("wb")
    ledger_stream = (root / "stories.jsonl").open("wb")
    index_names = (
        "base-train",
        "base-validation",
        "root-probes",
        *(
            f"task-{task_id}-{suffix}"
            for task_id in breakdown.task_ids
            for suffix in ("train", "validation", "probes", "generation")
        ),
    )
    index_streams = {
        name: (indexes / f"{name}.jsonl").open("wb") for name in index_names
    }
    index_by_id = {
        decision.concept_id: index for index, decision in enumerate(decisions)
    }
    base_mask = sum(1 << index_by_id[name] for name in breakdown.base_concept_ids)
    task_masks = {
        name: 1 << index_by_id[name] for name in breakdown.task_ids
    }
    root_probes = set(selections.root_probe_ids)
    task_probes = {
        task.task_id: set(task.probe_story_ids) for task in selections.task_summaries
    }
    story_offset = 0
    token_offset = 0
    written_story_count = 0
    try:
        for story_index, row in enumerate(
            connection.execute(
                "SELECT story_id, source_split, text, token_ids, token_count, "
                "concept_mask FROM stories ORDER BY story_id"
            )
        ):
            story_id, split, raw_text, raw_tokens, token_count, concept_mask = row
            story_id = str(story_id)
            split = str(split)
            text_payload = bytes(raw_text)
            token_payload = bytes(raw_tokens)
            token_count = int(token_count)
            concept_mask = int(concept_mask)
            normalized = text_payload.decode("utf-8", errors="strict")
            if (
                sha256(text_payload).hexdigest() != story_id
                or normalize_text(normalized) != normalized
                or len(token_payload) != token_count * 2
            ):
                raise ValueError("noun scan story payload changed before partitioning")
            matched_ids = tuple(
                decision.concept_id
                for index, decision in enumerate(decisions)
                if concept_mask & (1 << index)
            )
            matched_forms = dict(match_noun_forms(normalized, decisions))
            if tuple(matched_forms) != matched_ids:
                raise ValueError("noun scan masks and exact matched forms differ")
            entry = {
                "byte_length": len(text_payload),
                "concept_ids": list(matched_ids),
                "format": STORY_LEDGER_FORMAT,
                "matched_forms": {
                    concept_id: list(forms)
                    for concept_id, forms in matched_forms.items()
                },
                "source_split": split,
                "story_id": story_id,
                "story_index": story_index,
                "story_offset": story_offset,
                "token_count": token_count,
                "token_offset": token_offset,
            }
            index_entry = {
                key: entry[key]
                for key in (
                    "byte_length",
                    "story_id",
                    "story_index",
                    "story_offset",
                    "token_count",
                    "token_offset",
                )
            }
            story_stream.write(text_payload)
            token_stream.write(token_payload)
            ledger_stream.write(canonical_json_bytes(entry))
            in_base = bool(concept_mask & base_mask)
            held_out = _hash_bucket(TRAIN_HOLDOUT_NAMESPACE, story_id, 50) == 0
            if split == "train" and in_base:
                index_streams[
                    "base-validation" if held_out else "base-train"
                ].write(canonical_json_bytes(index_entry))
            if story_id in root_probes:
                index_streams["root-probes"].write(canonical_json_bytes(index_entry))
            for task_id, task_mask in task_masks.items():
                if not concept_mask & task_mask:
                    continue
                if split == "train":
                    if story_id in task_probes[task_id]:
                        index_streams[f"task-{task_id}-probes"].write(
                            canonical_json_bytes(index_entry)
                        )
                    else:
                        index_streams[f"task-{task_id}-train"].write(
                            canonical_json_bytes(index_entry)
                        )
                else:
                    index_streams[f"task-{task_id}-validation"].write(
                        canonical_json_bytes(index_entry)
                    )
                    if _generation_eligible(token_count):
                        index_streams[f"task-{task_id}-generation"].write(
                            canonical_json_bytes(index_entry)
                        )
            story_offset += len(text_payload)
            token_offset += token_count
            written_story_count += 1
    finally:
        for stream in (story_stream, token_stream, ledger_stream, *index_streams.values()):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    _write_base_selection_csv(root / "base-selection.csv", breakdown.base_selection)
    _write_task_counts_csv(root / "task-counts.csv", selections.task_summaries)
    return {
        "approval_sha256": approval.approval_sha256,
        "base_concept_ids": list(breakdown.base_concept_ids),
        "base_train_story_count": selections.base_train_count,
        "base_validation_story_count": selections.base_validation_count,
        "breakdown_sha256": breakdown.breakdown_sha256,
        "eos_token_id": 50_256,
        "pad_token_id": 50_256,
        "root_probe_story_ids": list(selections.root_probe_ids),
        "source_identity": breakdown.source_identity,
        "story_count": written_story_count,
        "task_ids": list(breakdown.task_ids),
        "tasks": [task.as_record() for task in selections.task_summaries],
        "token_count": token_offset,
        "tokenizer_identity": breakdown.tokenizer_identity,
    }


def _write_base_selection_csv(
    path: Path,
    steps: tuple[BaseSelectionStep, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(steps[0].as_record()))
        writer.writeheader()
        writer.writerows(step.as_record() for step in steps)
        output.flush()
        os.fsync(output.fileno())


def _write_task_counts_csv(
    path: Path,
    tasks: tuple[NounTaskSummary, ...],
) -> None:
    fields = (
        "task_id",
        "train_story_count",
        "update_story_count",
        "validation_story_count",
        "generation_story_count",
        "base_overlap_story_count",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: getattr(task, field) for field in fields} for task in tasks
        )
        output.flush()
        os.fsync(output.fileno())


def _push_lowest_id(
    heap: list[tuple[int, str]],
    limit: int,
    label: str,
    story_id: str,
    *,
    namespace: str = PROBE_NAMESPACE,
) -> None:
    priority = int(
        sha256(f"{namespace}\0{label}\0{story_id}".encode("utf-8")).hexdigest(),
        16,
    )
    candidate = (-priority, story_id)
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def _ordered_heap_ids(heap: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(story_id for _, story_id in sorted(heap, key=lambda value: (-value[0], value[1])))


def _hash_bucket(namespace: str, story_id: str, bucket_count: int) -> int:
    return int(
        sha256(f"{namespace}\0{story_id}".encode("utf-8")).hexdigest(), 16
    ) % bucket_count


def _generation_eligible(token_count: int) -> bool:
    """Return whether the midpoint leaves a routable prefix and one output token."""
    midpoint = token_count // 2
    return token_count >= 4 and 2 <= midpoint < MODEL_POSITION_LIMIT


def _index_story_ids(
    path: Path,
    *,
    materialize: bool = True,
) -> frozenset[str] | int:
    story_ids: set[str] = set()
    row_count = 0
    with path.open("rb") as source:
        for line in source:
            record = _canonical_object(line, f"noun index {path.name}")
            story_id = _text(record.get("story_id"), "indexed story")
            require_sha256(story_id, "indexed story")
            if story_id in story_ids:
                raise ValueError(f"noun index contains duplicate stories: {path.name}")
            story_ids.add(story_id)
            row_count += 1
    return frozenset(story_ids) if materialize else row_count


def _scan_metadata(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute("SELECT value FROM metadata WHERE key='scan'").fetchone()
    if row is None:
        raise ValueError("noun scan metadata is missing")
    return _canonical_object(str(row[0]).encode("utf-8"), "noun scan metadata")


def _load_noun_approval(path: Path) -> NounApproval:
    record = _canonical_object(path.read_bytes(), "noun approval")
    expected = {
        "approval_sha256",
        "approval_statement",
        "breakdown_sha256",
        "decision_sha256",
        "format",
        "schema_version",
        "source_sha256",
    }
    if set(record) != expected:
        raise ValueError("noun approval fields changed")
    approval = NounApproval(
        breakdown_sha256=_text(record["breakdown_sha256"], "approved breakdown"),
        decision_sha256=_text(record["decision_sha256"], "approved decisions"),
        source_sha256=_text(record["source_sha256"], "approved source"),
        approval_statement=_text(record["approval_statement"], "approval statement"),
    )
    if (
        record["format"] != APPROVAL_FORMAT
        or record["schema_version"] != 1
        or record["approval_sha256"] != approval.approval_sha256
    ):
        raise ValueError("noun approval identity changed")
    return approval


def _decision_from_record(value: object) -> NounDecision:
    record = _object(value, "noun decision")
    if set(record) != {"category", "concept_id", "forms", "included", "reason"}:
        raise ValueError("noun decision fields changed")
    included = record["included"]
    if type(included) is not bool:
        raise TypeError("noun decision included field must be boolean")
    return NounDecision(
        concept_id=_text(record["concept_id"], "noun concept"),
        category=_text(record["category"], "noun category"),
        forms=tuple(_text(item, "noun form") for item in _list(record["forms"], "forms")),
        included=included,
        reason=_text(record["reason"], "noun reason"),
    )


def _breakdown_row_from_record(value: object) -> NounBreakdownRow:
    record = _object(value, "noun breakdown row")
    decision = _decision_from_record(record.get("decision"))
    evidence = tuple(
        _evidence_from_record(item)
        for item in _list(record.get("evidence"), "noun evidence")
    )
    train_forms = _object(record.get("train_form_counts"), "train form counts")
    validation_forms = _object(
        record.get("validation_form_counts"), "validation form counts"
    )
    threshold = record.get("threshold_eligible")
    if type(threshold) is not bool:
        raise TypeError("threshold_eligible must be boolean")
    projected_role = _text(record.get("projected_role"), "projected role")
    if projected_role not in ("base", "task", "excluded", "below_threshold"):
        raise ValueError("projected noun role is invalid")
    return NounBreakdownRow(
        decision=decision,
        train_story_count=_integer(record.get("train_story_count"), "train stories"),
        validation_story_count=_integer(
            record.get("validation_story_count"), "validation stories"
        ),
        train_form_counts=tuple(
            (form, _integer(train_forms.get(form), f"train form {form}"))
            for form in decision.forms
        ),
        validation_form_counts=tuple(
            (form, _integer(validation_forms.get(form), f"validation form {form}"))
            for form in decision.forms
        ),
        train_prevalence=_number(record.get("train_prevalence"), "train prevalence"),
        validation_prevalence=_number(
            record.get("validation_prevalence"), "validation prevalence"
        ),
        threshold_eligible=threshold,
        projected_role=projected_role,
        evidence=evidence,
    )


def _evidence_from_record(value: object) -> NounEvidence:
    record = _object(value, "noun evidence")
    source_split = _text(record.get("source_split"), "evidence split")
    if source_split not in ("train", "validation"):
        raise ValueError("evidence split is invalid")
    return NounEvidence(
        story_id=_text(record.get("story_id"), "evidence story"),
        source_split="train" if source_split == "train" else "validation",
        matched_forms=tuple(
            _text(item, "evidence form")
            for item in _list(record.get("matched_forms"), "matched forms")
        ),
        story=_text(record.get("story"), "evidence text"),
    )


def _base_step_from_record(value: object) -> BaseSelectionStep:
    record = _object(value, "base-selection row")
    return BaseSelectionStep(
        concept_id=_text(record.get("concept_id"), "base noun"),
        noun_story_count=_integer(record.get("noun_story_count"), "noun stories"),
        new_story_count=_integer(record.get("new_story_count"), "new stories"),
        cumulative_story_count=_integer(
            record.get("cumulative_story_count"), "cumulative stories"
        ),
        cumulative_story_coverage=_number(
            record.get("cumulative_story_coverage"), "story coverage"
        ),
        new_token_count=_integer(record.get("new_token_count"), "new tokens"),
        cumulative_token_count=_integer(
            record.get("cumulative_token_count"), "cumulative tokens"
        ),
        cumulative_token_coverage=_number(
            record.get("cumulative_token_coverage"), "token coverage"
        ),
    )


def _task_summary_from_record(value: object) -> NounTaskSummary:
    record = _object(value, "task summary")
    overlap = _object(record.get("overlap_counts"), "overlap counts")
    return NounTaskSummary(
        task_id=_text(record.get("task_id"), "task ID"),
        train_story_count=_integer(record.get("train_story_count"), "task train"),
        update_story_count=_integer(record.get("update_story_count"), "task update"),
        validation_story_count=_integer(
            record.get("validation_story_count"), "task validation"
        ),
        generation_story_count=_integer(
            record.get("generation_story_count"), "task generation"
        ),
        probe_story_ids=tuple(
            _text(item, "probe story")
            for item in _list(record.get("probe_story_ids"), "probe stories")
        ),
        base_overlap_story_count=_integer(
            record.get("base_overlap_story_count"), "base overlap"
        ),
        overlap_counts=tuple(
            sorted(
                (
                    _text(name, "overlap noun"),
                    _integer(count, f"overlap {name}"),
                )
                for name, count in overlap.items()
            )
        ),
    )


def _canonical_object(payload: bytes, label: str) -> dict[str, object]:
    value = _json_object(payload, label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not a canonical JSON object")
    return value


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field {key!r} in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} is not a JSON object")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be nonempty text")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be a nonnegative integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _require_expected_source_name(path: Path, expected_name: str) -> None:
    if not path.is_file() or path.name != expected_name:
        raise ValueError(f"expected pinned source {expected_name}: {path}")


def _emit(
    progress: ProgressCallback | None,
    phase: str,
    completed: int,
    total: int | None,
) -> None:
    if progress is not None:
        progress(phase, completed, total)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(SOURCE_READ_BYTES):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BASE_TARGET_COVERAGE",
    "BASE_VALIDATION_BUCKET_COUNT",
    "DECISIONS_FORMAT",
    "GENERATION_STORY_COUNT",
    "MINIMUM_TASK_TRAIN_STORIES",
    "MINIMUM_TASK_VALIDATION_STORIES",
    "NounApprovalRequired",
    "PROBE_STORY_COUNT",
    "ScannedStory",
    "approve_noun_breakdown",
    "build_breakdown_from_documents",
    "build_noun_partition",
    "decisions_record",
    "initial_noun_decisions",
    "load_noun_breakdown",
    "load_noun_decisions",
    "load_noun_partition",
    "load_scanned_noun_breakdown",
    "match_noun_forms",
    "publish_initial_noun_decisions",
    "publish_noun_breakdown",
    "render_noun_breakdown_html",
    "render_noun_breakdown_markdown",
    "require_noun_approval",
    "scan_pinned_noun_breakdown",
]
