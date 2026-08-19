"""Canonical, crash-safe persistence primitives for continual experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Final


_GENESIS_HASH: Final[str] = "0" * 64
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")


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
    """Return the SHA-256 digest of one canonical JSON-compatible value."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: str, label: str) -> None:
    """Require a lowercase hexadecimal SHA-256 digest."""
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: str | Path) -> None:
    """Flush directory metadata for an already-created directory."""
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: str | Path, payload: bytes) -> Path:
    """Atomically replace one file and fsync its containing directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def publish_immutable_bytes(path: str | Path, payload: bytes) -> Path:
    """Publish bytes once or require identity with an existing artifact."""
    target = Path(path)
    if target.is_file():
        if target.read_bytes() != payload:
            raise ValueError(f"immutable artifact changed: {target}")
        return target
    if target.exists():
        raise ValueError(f"immutable artifact path is not a file: {target}")
    return atomic_write(target, payload)


def publish_immutable_json(path: str | Path, record: Mapping[str, object]) -> Path:
    """Publish canonical JSON once or require byte identity with the prior file."""
    target = Path(path)
    payload = canonical_json_bytes(dict(record))
    if target.is_file() and target.read_bytes() != payload:
        raise ValueError(f"immutable JSON changed: {target}")
    return publish_immutable_bytes(target, payload)


def load_canonical_json(path: str | Path) -> dict[str, object]:
    """Load one canonical JSON object without accepting alternate encodings."""
    source = Path(path)
    payload = source.read_bytes()
    value = json.loads(payload)
    if type(value) is not dict or payload != canonical_json_bytes(value):
        raise ValueError(f"JSON is not one canonical object: {source}")
    return value


class ChainedJsonlLedger:
    """Append and validate monotonically sequenced, hash-chained JSONL rows."""

    def __init__(self, path: str | Path, row_format: str) -> None:
        if not row_format:
            raise ValueError("ledger row format must be nonempty")
        self.path = Path(path)
        self.row_format = row_format
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = self._load_and_repair()

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        """Return a detached immutable-order view of validated rows."""
        return tuple(dict(row) for row in self._rows)

    @property
    def next_sequence(self) -> int:
        """Return the next monotonic sequence number."""
        return len(self._rows)

    @property
    def tail_hash(self) -> str:
        """Return the last row hash or the fixed genesis hash."""
        return str(self._rows[-1]["result_sha256"]) if self._rows else _GENESIS_HASH

    def append(self, values: Mapping[str, object]) -> dict[str, object]:
        """Append, flush, and fsync one canonical chained row."""
        return self.append_many((values,))[0]

    def append_many(
        self,
        values: Iterable[Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        """Append a bounded row batch with one flush and fsync."""
        batch = tuple(values)
        if not batch:
            return ()
        forbidden = {"format", "previous_sha256", "result_sha256", "sequence"}
        if any(forbidden & set(item) for item in batch):
            raise ValueError("ledger values contain reserved fields")
        rows: list[dict[str, object]] = []
        previous = self.tail_hash
        for offset, item in enumerate(batch):
            core = {
                **dict(item),
                "format": self.row_format,
                "previous_sha256": previous,
                "sequence": self.next_sequence + offset,
            }
            row = {**core, "result_sha256": record_sha256(core)}
            rows.append(row)
            previous = str(row["result_sha256"])
        with self.path.open("ab") as output:
            output.write(b"".join(canonical_json_bytes(row) for row in rows))
            output.flush()
            os.fsync(output.fileno())
        self._rows.extend(rows)
        return tuple(dict(row) for row in rows)

    def after(self, sequence: int) -> tuple[dict[str, object], ...]:
        """Return rows whose sequence is strictly greater than the argument."""
        if type(sequence) is not int or sequence < -1:
            raise ValueError("event cursor must be an integer at least -1")
        return tuple(dict(row) for row in self._rows[sequence + 1 :])

    def require_unique_keys(self, fields: Iterable[str]) -> None:
        """Reject duplicate composite identities across validated rows."""
        names = tuple(fields)
        if not names:
            raise ValueError("ledger uniqueness requires at least one field")
        keys = tuple(tuple(row.get(name) for name in names) for row in self._rows)
        if len(set(keys)) != len(keys):
            raise ValueError(f"ledger contains duplicate keys for {names}")

    def truncate(self, row_count: int) -> None:
        """Atomically discard an authenticated suffix after checkpoint rollback."""
        if type(row_count) is not int or not 0 <= row_count <= len(self._rows):
            raise ValueError("ledger truncation count is outside validated coverage")
        if row_count == len(self._rows):
            return
        retained = self._rows[:row_count]
        atomic_write(self.path, b"".join(canonical_json_bytes(row) for row in retained))
        self._rows = list(retained)

    def _load_and_repair(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        payload = self.path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            final_newline = payload.rfind(b"\n")
            complete = payload[: final_newline + 1] if final_newline >= 0 else b""
            atomic_write(self.path, complete)
            payload = complete
        rows: list[dict[str, object]] = []
        previous = _GENESIS_HASH
        for expected_sequence, line in enumerate(payload.splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("ledger contains malformed complete JSON") from error
            if type(row) is not dict or line + b"\n" != canonical_json_bytes(row):
                raise ValueError("ledger row is not canonical JSONL")
            supplied = row.get("result_sha256")
            core = {key: value for key, value in row.items() if key != "result_sha256"}
            if (
                row.get("format") != self.row_format
                or row.get("sequence") != expected_sequence
                or row.get("previous_sha256") != previous
                or supplied != record_sha256(core)
            ):
                raise ValueError("ledger sequence, chain, format, or hash changed")
            previous = str(supplied)
            rows.append(row)
        return rows


__all__ = [
    "ChainedJsonlLedger",
    "atomic_write",
    "canonical_json_bytes",
    "file_sha256",
    "fsync_directory",
    "load_canonical_json",
    "publish_immutable_bytes",
    "publish_immutable_json",
    "record_sha256",
    "require_sha256",
]
