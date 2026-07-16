"""Verified TinyShakespeare preparation and deterministic causal text batches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import urlopen

import jax
import numpy as np

from apm.lm.text import CharTokenizer, TextTokenizer


@dataclass(frozen=True)
class TinyShakespeareSourceRef:
    """Immutable provenance and integrity metadata for TinyShakespeare."""

    repository: str
    revision: str
    relative_path: str
    expected_sha256: str
    expected_size: int

    @property
    def url(self) -> str:
        """Return the immutable raw-file URL for this source revision."""
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{self.revision}/{self.relative_path}"
        )


TINY_SHAKESPEARE_SOURCE = TinyShakespeareSourceRef(
    repository="karpathy/char-rnn",
    revision="6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e",
    relative_path="data/tinyshakespeare/input.txt",
    expected_sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    expected_size=1_115_394,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class TokenBatch:
    """Validated immutable causal-LM arrays with a JAX PyTree representation."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray

    def __post_init__(self) -> None:
        """Normalize dtypes and require four matching rank-two shapes."""
        input_ids = _immutable_token_ids(self.input_ids, "input_ids")
        target_ids = _immutable_token_ids(self.target_ids, "target_ids")
        attention_mask = _immutable_mask(self.attention_mask, "attention_mask")
        loss_mask = _immutable_mask(self.loss_mask, "loss_mask")
        shapes = {
            input_ids.shape,
            attention_mask.shape,
            target_ids.shape,
            loss_mask.shape,
        }
        if input_ids.ndim != 2 or len(shapes) != 1:
            raise ValueError("TokenBatch fields must share one rank-two [batch, sequence] shape")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "loss_mask", loss_mask)

    def tree_flatten(self):
        """Expose all four arrays as dynamic JAX PyTree leaves."""
        return (
            self.input_ids,
            self.attention_mask,
            self.target_ids,
            self.loss_mask,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        """Rebuild a batch without coercing JAX tracers through NumPy."""
        del auxiliary_data
        batch = object.__new__(cls)
        for field_name, child in zip(
            ("input_ids", "attention_mask", "target_ids", "loss_mask"),
            children,
        ):
            object.__setattr__(batch, field_name, child)
        return batch


@dataclass(frozen=True)
class TextSplits:
    """Contiguous raw-character train, validation, and test spans."""

    train: str
    validation: str
    test: str


@dataclass(frozen=True)
class TextDataPreset:
    """Fixed context, batch, and window stride for deterministic text data."""

    context_length: int
    batch_size: int
    stride: int

    def __post_init__(self) -> None:
        """Reject non-positive fixed data dimensions."""
        if self.context_length <= 0 or self.batch_size <= 0 or self.stride <= 0:
            raise ValueError("text data preset dimensions must be positive")


TINY_SHAKESPEARE_STANDARD_PRESET = TextDataPreset(
    context_length=256,
    batch_size=32,
    stride=256,
)


@dataclass(frozen=True)
class TinyShakespeareData:
    """Tokenizer, aligned raw splits, and fixed-shape batches for one corpus."""

    tokenizer: CharTokenizer
    text_splits: TextSplits
    train_batches: tuple[TokenBatch, ...]
    validation_batches: tuple[TokenBatch, ...]
    test_batches: tuple[TokenBatch, ...]


def prepare_tiny_shakespeare(
    destination_path: str | Path,
    source_ref: TinyShakespeareSourceRef = TINY_SHAKESPEARE_SOURCE,
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> Path:
    """Explicitly fetch and integrity-check the pinned corpus when absent or invalid."""
    destination = Path(destination_path)
    if destination.is_file():
        existing_payload = destination.read_bytes()
        if _source_bytes_are_valid(existing_payload, source_ref):
            return destination
    payload = (fetch_bytes or _fetch_bytes)(source_ref.url)
    _verify_source_bytes(payload, source_ref)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.download")
    temporary_path.write_bytes(payload)
    temporary_path.replace(destination)
    return destination


def load_tiny_shakespeare(
    corpus_path: str | Path,
    source_ref: TinyShakespeareSourceRef = TINY_SHAKESPEARE_SOURCE,
) -> str:
    """Load and verify a local corpus without performing network access."""
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"TinyShakespeare corpus not found: {path}")
    payload = path.read_bytes()
    _verify_source_bytes(payload, source_ref)
    return payload.decode("utf-8")


def split_text_contiguously(text: str) -> TextSplits:
    """Split raw characters contiguously into deterministic 90/5/5 spans."""
    train_size = 90 * len(text) // 100
    validation_size = 5 * len(text) // 100
    validation_end = train_size + validation_size
    return TextSplits(
        train=text[:train_size],
        validation=text[train_size:validation_end],
        test=text[validation_end:],
    )


def causal_token_windows(
    token_ids: Sequence[int],
    context_length: int,
    pad_token_id: int,
    *,
    stride: int | None = None,
) -> TokenBatch:
    """Create deterministic fixed-length causal windows with right padding."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    window_stride = context_length if stride is None else stride
    if window_stride <= 0:
        raise ValueError("stride must be positive")
    tokens = np.asarray(tuple(token_ids), dtype=np.int64)
    if np.any(tokens < 0) or pad_token_id < 0:
        raise ValueError("token IDs must be non-negative")
    starts = tuple(range(0, max(len(tokens) - 1, 0), window_stride))
    shape = (len(starts), context_length)
    input_ids = np.full(shape, pad_token_id, dtype=np.int32)
    target_ids = np.full(shape, pad_token_id, dtype=np.int32)
    attention_mask = np.zeros(shape, dtype=np.bool_)
    loss_mask = np.zeros(shape, dtype=np.bool_)
    for row_index, start in enumerate(starts):
        chunk = tokens[start : start + context_length + 1]
        transition_count = len(chunk) - 1
        input_ids[row_index, :transition_count] = chunk[:-1]
        target_ids[row_index, :transition_count] = chunk[1:]
        attention_mask[row_index, :transition_count] = True
        loss_mask[row_index, :transition_count] = True
    return TokenBatch(input_ids, attention_mask, target_ids, loss_mask)


def batch_token_windows(
    windows: TokenBatch,
    batch_size: int,
    pad_token_id: int,
) -> tuple[TokenBatch, ...]:
    """Group windows into fixed-size batches, padding the final batch with empty rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")
    window_count, context_length = windows.input_ids.shape
    return tuple(
        _padded_window_batch(
            windows,
            start,
            min(start + batch_size, window_count),
            batch_size,
            context_length,
            pad_token_id,
        )
        for start in range(0, window_count, batch_size)
    )


def build_tiny_shakespeare_data(
    corpus_text: str,
    preset: TextDataPreset = TINY_SHAKESPEARE_STANDARD_PRESET,
) -> TinyShakespeareData:
    """Split before fitting/tokenizing and build deterministic batches per span."""
    text_splits = split_text_contiguously(corpus_text)
    tokenizer = CharTokenizer.from_training_text(text_splits.train)
    split_batches = tuple(
        _tokenize_and_batch(split_text, tokenizer, preset)
        for split_text in (
            text_splits.train,
            text_splits.validation,
            text_splits.test,
        )
    )
    return TinyShakespeareData(
        tokenizer=tokenizer,
        text_splits=text_splits,
        train_batches=split_batches[0],
        validation_batches=split_batches[1],
        test_batches=split_batches[2],
    )


def _tokenize_and_batch(
    text: str,
    tokenizer: TextTokenizer,
    preset: TextDataPreset,
) -> tuple[TokenBatch, ...]:
    windows = causal_token_windows(
        tokenizer.encode(text, add_eos=True),
        preset.context_length,
        tokenizer.pad_token_id,
        stride=preset.stride,
    )
    return batch_token_windows(windows, preset.batch_size, tokenizer.pad_token_id)


def _padded_window_batch(
    windows: TokenBatch,
    start: int,
    stop: int,
    batch_size: int,
    context_length: int,
    pad_token_id: int,
) -> TokenBatch:
    shape = (batch_size, context_length)
    input_ids = np.full(shape, pad_token_id, dtype=np.int32)
    target_ids = np.full(shape, pad_token_id, dtype=np.int32)
    attention_mask = np.zeros(shape, dtype=np.bool_)
    loss_mask = np.zeros(shape, dtype=np.bool_)
    row_count = stop - start
    input_ids[:row_count] = windows.input_ids[start:stop]
    target_ids[:row_count] = windows.target_ids[start:stop]
    attention_mask[:row_count] = windows.attention_mask[start:stop]
    loss_mask[:row_count] = windows.loss_mask[start:stop]
    return TokenBatch(input_ids, attention_mask, target_ids, loss_mask)


def _immutable_token_ids(values: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{field_name} must contain integer token IDs")
    if np.any(array < 0):
        raise ValueError(f"{field_name} must contain non-negative token IDs")
    normalized = np.array(array, dtype=np.int32, copy=True)
    normalized.flags.writeable = False
    return normalized


def _immutable_mask(values: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.all(np.logical_or(array == 0, array == 1)):
        raise ValueError(f"{field_name} must contain only zero/one values")
    normalized = np.array(array, dtype=np.bool_, copy=True)
    normalized.flags.writeable = False
    return normalized


def _fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def _source_bytes_are_valid(
    payload: bytes,
    source_ref: TinyShakespeareSourceRef,
) -> bool:
    return len(payload) == source_ref.expected_size and (
        sha256(payload).hexdigest() == source_ref.expected_sha256
    )


def _verify_source_bytes(
    payload: bytes,
    source_ref: TinyShakespeareSourceRef,
) -> None:
    if not _source_bytes_are_valid(payload, source_ref):
        raise ValueError(
            "TinyShakespeare source integrity check failed: "
            f"expected {source_ref.expected_size} bytes and SHA-256 "
            f"{source_ref.expected_sha256}, received {len(payload)} bytes and "
            f"{sha256(payload).hexdigest()}"
        )
