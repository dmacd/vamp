from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
import importlib.util
from pathlib import Path

import pytest

from apm.lm import (
    GPT2_END_OF_TEXT_TOKEN,
    TextTokenizer,
    TokenizersTextTokenizer,
)


@dataclass(frozen=True)
class _FakeEncoding:
    ids: tuple[int, ...]


class _FakeTokenizerBackend:
    _token_ids = {
        "a": 0,
        "b": 1,
        " ": 2,
        "[PAD]": 3,
        GPT2_END_OF_TEXT_TOKEN: 50_256,
    }

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        assert with_added_tokens
        return 50_257

    def token_to_id(self, token: str) -> int | None:
        return self._token_ids.get(token)

    def encode(
        self,
        sequence: str,
        *,
        add_special_tokens: bool = True,
    ) -> _FakeEncoding:
        token_ids = tuple(self._token_ids[character] for character in sequence)
        return _FakeEncoding(
            token_ids + ((50_256,) if add_special_tokens else ())
        )

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        token_by_id = {
            token_id: token for token, token_id in self._token_ids.items()
        }
        return "".join(
            token_by_id[token_id]
            for token_id in ids
            if not skip_special_tokens or token_id != 50_256
        )


class _MissingEosBackend(_FakeTokenizerBackend):
    def token_to_id(self, token: str) -> int | None:
        if token == GPT2_END_OF_TEXT_TOKEN:
            return None
        return super().token_to_id(token)


class _InvalidEncodingBackend(_FakeTokenizerBackend):
    def encode(
        self,
        sequence: str,
        *,
        add_special_tokens: bool = True,
    ) -> _FakeEncoding:
        return _FakeEncoding((self.get_vocab_size(),))


def test_structural_adapter_matches_backend_encoding_and_adds_eos_explicitly() -> None:
    backend = _FakeTokenizerBackend()
    tokenizer = TokenizersTextTokenizer(backend)
    direct_ids = backend.encode("ab a", add_special_tokens=False).ids

    assert isinstance(tokenizer, TextTokenizer)
    assert tokenizer.vocab_size == 50_257
    assert tokenizer.eos_token_id == 50_256
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.encode("ab a") == direct_ids
    assert tokenizer.encode("ab a", add_eos=True) == (*direct_ids, 50_256)
    assert tokenizer.decode((*direct_ids, 50_256)) == "ab a"
    assert tokenizer.decode(
        (*direct_ids, 50_256),
        skip_special_tokens=False,
    ) == f"ab a{GPT2_END_OF_TEXT_TOKEN}"


def test_adapter_ids_are_frozen_configurable_and_range_checked() -> None:
    tokenizer = TokenizersTextTokenizer(
        _FakeTokenizerBackend(),
        pad_token_id=3,
    )

    assert tokenizer.pad_token_id == 3
    with pytest.raises(FrozenInstanceError):
        tokenizer._pad_token_id = 2  # type: ignore[misc]
    for invalid_pad_token_id in (-1, 50_257, True):
        with pytest.raises(ValueError, match="PAD token ID"):
            TokenizersTextTokenizer(
                _FakeTokenizerBackend(),
                pad_token_id=invalid_pad_token_id,
            )
    with pytest.raises(ValueError, match="missing EOS"):
        TokenizersTextTokenizer(_MissingEosBackend())
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.decode((50_257,))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        TokenizersTextTokenizer(_InvalidEncodingBackend()).encode("a")


def test_from_file_has_an_actionable_optional_dependency_boundary(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("tokenizers") is not None:
        pytest.skip("optional tokenizers dependency is installed")
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")

    with pytest.raises(ImportError, match=r"apm\[lm\]"):
        TokenizersTextTokenizer.from_file(tokenizer_json)


def test_real_tokenizer_json_matches_tokenizer_from_file(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                GPT2_END_OF_TEXT_TOKEN: 3,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer_json = tmp_path / "tokenizer.json"
    backend.save(str(tokenizer_json))
    direct = Tokenizer.from_file(str(tokenizer_json))

    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_json)
    expected_ids = tuple(
        direct.encode("hello world", add_special_tokens=False).ids
    )

    assert tokenizer.encode("hello world") == expected_ids
    assert tokenizer.encode("hello world", add_eos=True) == (*expected_ids, 3)
    assert tokenizer.decode(expected_ids) == direct.decode(
        list(expected_ids),
        skip_special_tokens=True,
    )
    assert tokenizer.eos_token_id == tokenizer.pad_token_id == 3
