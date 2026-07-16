"""Immutable adapters for deterministic character and tokenizer.json tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
GPT2_END_OF_TEXT_TOKEN = "<|endoftext|>"


@runtime_checkable
class TextTokenizer(Protocol):
    """Minimal immutable text-tokenizer interface used by language VAMP."""

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs accepted by the tokenizer."""
        ...

    @property
    def pad_token_id(self) -> int:
        """Return the token ID used for right padding."""
        ...

    @property
    def eos_token_id(self) -> int:
        """Return the token ID used for end-of-sequence transitions."""
        ...

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        """Encode text as an immutable token-ID sequence."""
        ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs, optionally retaining readable special tokens."""
        ...


class _TokenizersEncoding(Protocol):
    """Structural subset of ``tokenizers.Encoding`` used by the adapter."""

    @property
    def ids(self) -> Sequence[int]:
        """Return encoded token IDs."""
        ...


@runtime_checkable
class _TokenizersBackend(Protocol):
    """Structural subset of ``tokenizers.Tokenizer`` used by the adapter."""

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """Return the backend vocabulary size."""
        ...

    def token_to_id(self, token: str) -> int | None:
        """Resolve one token string to its ID when present."""
        ...

    def encode(
        self,
        sequence: str,
        *,
        add_special_tokens: bool = True,
    ) -> _TokenizersEncoding:
        """Encode one string with optional backend post-processing."""
        ...

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode one token-ID sequence."""
        ...


@dataclass(frozen=True)
class CharTokenizer:
    """Deterministic character tokenizer fitted only from training text."""

    vocabulary: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate reserved IDs and the sorted unique character vocabulary."""
        if self.vocabulary[:2] != (PAD_TOKEN, EOS_TOKEN):
            raise ValueError("character vocabulary must reserve PAD=0 and EOS=1")
        characters = self.vocabulary[2:]
        if any(len(character) != 1 for character in characters):
            raise ValueError("non-special character tokens must contain one character")
        if characters != tuple(sorted(set(characters))):
            raise ValueError("character vocabulary must be sorted and unique")

    @classmethod
    def from_training_text(cls, training_text: str) -> CharTokenizer:
        """Build a sorted vocabulary from training characters only."""
        return cls((PAD_TOKEN, EOS_TOKEN, *sorted(set(training_text))))

    @property
    def vocab_size(self) -> int:
        """Return the number of reserved and character tokens."""
        return len(self.vocabulary)

    @property
    def pad_token_id(self) -> int:
        """Return the fixed PAD token ID."""
        return PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        """Return the fixed EOS token ID."""
        return EOS_TOKEN_ID

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        """Encode known characters and optionally append EOS."""
        token_by_character = {
            character: token_id
            for token_id, character in enumerate(self.vocabulary[2:], start=2)
        }
        unknown_characters = tuple(
            sorted(set(text).difference(token_by_character))
        )
        if unknown_characters:
            raise ValueError(
                f"text contains characters absent from the training vocabulary: {unknown_characters}"
            )
        encoded = tuple(token_by_character[character] for character in text)
        return encoded + ((EOS_TOKEN_ID,) if add_eos else ())

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode IDs and reject values outside the fitted vocabulary."""
        invalid_ids = tuple(
            token_id
            for token_id in token_ids
            if token_id < 0 or token_id >= self.vocab_size
        )
        if invalid_ids:
            raise ValueError(f"token IDs are outside the vocabulary: {invalid_ids}")
        tokens = tuple(self.vocabulary[token_id] for token_id in token_ids)
        return "".join(
            token
            for token in tokens
            if not skip_special_tokens or token not in (PAD_TOKEN, EOS_TOKEN)
        )


@dataclass(frozen=True, init=False)
class TokenizersTextTokenizer:
    """Immutable ``tokenizers.Tokenizer`` adapter for GPT-2 tokenizer JSON."""

    _backend: _TokenizersBackend
    _vocab_size: int
    _pad_token_id: int
    _eos_token_id: int

    def __init__(
        self,
        backend: _TokenizersBackend,
        *,
        pad_token_id: int | None = None,
    ) -> None:
        """Bind a structural backend and resolve its immutable special IDs."""
        if not isinstance(backend, _TokenizersBackend):
            raise TypeError("backend must satisfy the tokenizers Tokenizer interface")
        vocab_size = backend.get_vocab_size(with_added_tokens=True)
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("tokenizer vocabulary size must be a positive integer")
        eos_token_id = backend.token_to_id(GPT2_END_OF_TEXT_TOKEN)
        if eos_token_id is None:
            raise ValueError(
                f"tokenizer vocabulary is missing EOS {GPT2_END_OF_TEXT_TOKEN!r}"
            )
        _validate_token_id(eos_token_id, vocab_size, "EOS")
        resolved_pad_token_id = (
            eos_token_id if pad_token_id is None else pad_token_id
        )
        _validate_token_id(resolved_pad_token_id, vocab_size, "PAD")

        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_vocab_size", vocab_size)
        object.__setattr__(self, "_pad_token_id", resolved_pad_token_id)
        object.__setattr__(self, "_eos_token_id", eos_token_id)

    @classmethod
    def from_file(
        cls,
        tokenizer_json: str | Path,
        *,
        pad_token_id: int | None = None,
    ) -> TokenizersTextTokenizer:
        """Load one local tokenizer.json without importing optional code eagerly."""
        tokenizer_path = Path(tokenizer_json)
        if tokenizer_path.name != "tokenizer.json":
            raise ValueError("TinyStories tokenizer path must name tokenizer.json")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise ImportError(
                "loading tokenizer.json requires the optional 'tokenizers' package; "
                "install the apm[lm] extra"
            ) from error
        return cls(
            Tokenizer.from_file(str(tokenizer_path)),
            pad_token_id=pad_token_id,
        )

    @property
    def vocab_size(self) -> int:
        """Return the validated tokenizer vocabulary size."""
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return the explicit padding ID, which defaults to GPT-2 EOS."""
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Return the ID resolved from the GPT-2 end-of-text token."""
        return self._eos_token_id

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        """Encode without backend-added specials and optionally append one EOS."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(add_eos) is not bool:
            raise TypeError("add_eos must be a boolean")
        encoding = self._backend.encode(text, add_special_tokens=False)
        token_ids = _validated_token_ids(
            encoding.ids,
            self.vocab_size,
            "encoded token",
        )
        return token_ids + ((self.eos_token_id,) if add_eos else ())

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Validate and decode IDs through the underlying tokenizer backend."""
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a boolean")
        validated_ids = _validated_token_ids(
            token_ids,
            self.vocab_size,
            "token",
        )
        return self._backend.decode(
            list(validated_ids),
            skip_special_tokens=skip_special_tokens,
        )


def _validated_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
    label: str,
) -> tuple[int, ...]:
    validated_ids = tuple(token_ids)
    invalid_ids = tuple(
        token_id
        for token_id in validated_ids
        if type(token_id) is not int or token_id < 0 or token_id >= vocab_size
    )
    if invalid_ids:
        raise ValueError(f"{label} IDs are outside the vocabulary: {invalid_ids}")
    return validated_ids


def _validate_token_id(token_id: int, vocab_size: int, label: str) -> None:
    if type(token_id) is not int or token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{label} token ID is outside the vocabulary")
