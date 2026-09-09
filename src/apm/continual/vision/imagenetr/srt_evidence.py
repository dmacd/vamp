"""Content-addressed, checkpoint-committed SRT trace chunks and result records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apm.continual.artifacts import (
    canonical_json_bytes, file_sha256, load_canonical_json,
    publish_immutable_bytes, publish_immutable_json, record_sha256,
)


CLOCK_FIELDS = frozenset({"clock", "clock_gap", "due_before", "due_after", "interval_before", "interval_after", "lateness"})

EVENT_SCHEMA = pa.schema([
    (name, pa.string() if name == "kind" or name in CLOCK_FIELDS else pa.float64() if name in {
        "confidence", "ease_before", "ease_after",
    } else pa.int64())
    for name in (
        "image", "arrival_stage", "stage", "exposure", "kind", "step", "presentations", "clock",
        "confidence", "quality", "previous_quality", "step_gap", "presentation_gap", "stage_gap", "clock_gap",
        "due_before", "lateness", "ease_before", "ease_after", "successes_before", "successes_after",
        "interval_before", "interval_after", "due_after",
    )
])


def sealed_record(values: dict[str, object]) -> dict[str, object]:
    """Bind a JSON record to its exact contents."""
    return {**values, "content_hash": record_sha256(values)}


def read_sealed(path: Path, schema: str | None = None) -> dict[str, object]:
    """Reject an altered or schema-mismatched scientific record."""
    record = load_canonical_json(path)
    if (record.get("content_hash") != record_sha256({key: value for key, value in record.items() if key != "content_hash"})
            or (schema is not None and record.get("schema_version") != schema)):
        raise ValueError(f"unauthenticated SRT evidence: {path}")
    return record


def write_parquet(path: Path, records: Sequence[dict[str, object]], schema: pa.Schema | None = None) -> str:
    """Publish a compact immutable analysis table without retaining a whole run."""
    stream = BytesIO()
    if schema == EVENT_SCHEMA:
        records = tuple({**row, **{name: str(row[name]) if row[name] is not None else None for name in CLOCK_FIELDS}}
                        for row in records)
    pq.write_table(pa.Table.from_pylist(list(records), schema=schema), stream, compression="zstd")
    payload = stream.getvalue()
    publish_immutable_bytes(path, payload)
    return sha256(payload).hexdigest()


def read_event_chunk(path: Path) -> tuple[dict[str, object], ...]:
    """Decode lossless decimal clock integers, which can exceed machine widths."""
    return tuple({**row, **{name: int(row[name]) if row[name] is not None else None for name in CLOCK_FIELDS}}
                 for row in pq.read_table(path).to_pylist())


@dataclass(frozen=True, slots=True)
class TraceBuffer:
    """At most one checkpoint interval of uncommitted event and step rows."""

    events: tuple[dict[str, object], ...] = ()
    batches: tuple[dict[str, object], ...] = ()

    def append(self, events: tuple[dict[str, object], ...], batch: dict[str, object]) -> TraceBuffer:
        """Return a new bounded buffer for the next successful optimizer update."""
        return TraceBuffer(self.events + events, self.batches + (batch,))

    def publish(self, root: Path) -> dict[str, object]:
        """Write trace files first; only a subsequent checkpoint commits them."""
        if not self.batches:
            raise ValueError("cannot publish an empty trace chunk")
        identity = record_sha256({"batches": self.batches, "events": self.events})
        relative = Path("trace") / identity
        event_hash = write_parquet(root / relative / "events.parquet", self.events, EVENT_SCHEMA)
        batches = sealed_record({"schema_version": "imagenetr50-srt-batches-v1", "rows": self.batches})
        publish_immutable_json(root / relative / "batches.json", batches)
        return {
            "path": relative.as_posix(), "events_sha256": event_hash,
            "batches_sha256": file_sha256(root / relative / "batches.json"),
            "first_step": self.batches[0]["step"], "last_step": self.batches[-1]["step"],
            "stage": self.batches[0]["stage"], "event_count": len(self.events),
        }


def trace_batches(root: Path, chunks: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
    """Read only checkpoint-committed batch records, authenticating each file."""
    rows = ()
    for chunk in chunks:
        path = root / chunk["path"] / "batches.json"
        if file_sha256(path) != chunk["batches_sha256"]:
            raise ValueError("committed SRT batch evidence changed")
        rows += tuple(read_sealed(path, "imagenetr50-srt-batches-v1")["rows"])
    return rows
