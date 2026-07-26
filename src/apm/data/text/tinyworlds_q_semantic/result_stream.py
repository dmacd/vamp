"""Incremental temporary ledgers for long-running semantic evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_q_semantic.contracts import (
    SemanticQueryResult,
    canonical_json_bytes,
    require_identifier,
    require_sha256,
)


SemanticResultSink = Callable[[tuple[SemanticQueryResult, ...]], None]
SemanticResultProducer = Callable[
    [SemanticResultSink],
    tuple[SemanticQueryResult, ...],
]


@dataclass(frozen=True, slots=True)
class StreamedSemanticResultLedger:
    """One complete temporary ledger verified against its returned result tuple."""

    directory: Path
    path: Path
    results_sha256: str
    result_count: int
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "path", Path(self.path))
        require_sha256(self.results_sha256, "streamed semantic results")
        if (
            not self.directory.is_dir()
            or not self.path.is_file()
            or self.path.parent != self.directory
            or type(self.result_count) is not int
            or self.result_count <= 0
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
            or self.path.stat().st_size != self.size_bytes
        ):
            raise ValueError("streamed semantic result ledger is incomplete")


def stream_semantic_results(
    output_root: str | Path,
    label: str,
    producer: SemanticResultProducer,
    *,
    size_limit_bytes: int,
) -> tuple[tuple[SemanticQueryResult, ...], StreamedSemanticResultLedger]:
    """Write result batches as produced and retain the directory for recovery."""
    require_identifier(label, "semantic result stream label")
    if type(size_limit_bytes) is not int or size_limit_bytes <= 0:
        raise ValueError("semantic result stream requires a positive size limit")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=root))
    path = directory / "results.jsonl"
    print(f"TinyWorlds-Q result stream temporary artifacts: {directory}", flush=True)
    with path.open("xb") as stream:
        def write_rows(rows: tuple[SemanticQueryResult, ...]) -> None:
            if type(rows) is not tuple or any(
                type(row) is not SemanticQueryResult for row in rows
            ):
                raise TypeError("semantic result sink requires exact result tuples")
            for row in rows:
                stream.write(canonical_json_bytes(row.as_record()))
            if stream.tell() > size_limit_bytes:
                raise OSError("semantic result stream exceeds the frozen size limit")
            stream.flush()

        results = producer(write_rows)
        stream.flush()
        os.fsync(stream.fileno())
    expected_payload_sha256 = _results_sha256(results)
    if (
        _file_sha256(path) != expected_payload_sha256
        or path.stat().st_size > size_limit_bytes
    ):
        raise ValueError("streamed semantic results differ from returned results")
    return results, StreamedSemanticResultLedger(
        directory=directory.resolve(),
        path=path.resolve(),
        results_sha256=expected_payload_sha256,
        result_count=len(results),
        size_bytes=path.stat().st_size,
    )


def _results_sha256(results: tuple[SemanticQueryResult, ...]) -> str:
    if type(results) is not tuple or not results:
        raise ValueError("semantic result producer returned no result tuple")
    digest = sha256()
    for result in results:
        digest.update(canonical_json_bytes(result.as_record()))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SemanticResultProducer",
    "SemanticResultSink",
    "StreamedSemanticResultLedger",
    "stream_semantic_results",
]
