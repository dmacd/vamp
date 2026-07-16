from __future__ import annotations

from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.text import (
    EOS_TOKEN,
    EOS_TOKEN_ID,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    CharTokenizer,
    TextTokenizer,
)
from apm.lm.text_data import (
    TINY_SHAKESPEARE_SOURCE,
    TINY_SHAKESPEARE_STANDARD_PRESET,
    TextDataPreset,
    TinyShakespeareSourceRef,
    TokenBatch,
    batch_token_windows,
    build_tiny_shakespeare_data,
    causal_token_windows,
    load_tiny_shakespeare,
    prepare_tiny_shakespeare,
    split_text_contiguously,
)


def test_char_tokenizer_reserves_special_ids_and_sorts_training_characters() -> None:
    tokenizer = CharTokenizer.from_training_text("cabca\n")

    assert isinstance(tokenizer, TextTokenizer)
    assert tokenizer.vocabulary == (PAD_TOKEN, EOS_TOKEN, "\n", "a", "b", "c")
    assert tokenizer.pad_token_id == PAD_TOKEN_ID == 0
    assert tokenizer.eos_token_id == EOS_TOKEN_ID == 1
    encoded = tokenizer.encode("cab", add_eos=True)
    assert encoded == (5, 3, 4, EOS_TOKEN_ID)
    assert tokenizer.decode(encoded) == "cab"
    assert tokenizer.decode((PAD_TOKEN_ID, *encoded), skip_special_tokens=False) == (
        "<PAD>cab<EOS>"
    )


def test_char_tokenizer_rejects_characters_absent_from_training() -> None:
    tokenizer = CharTokenizer.from_training_text("abc")

    with pytest.raises(ValueError, match="absent from the training vocabulary"):
        tokenizer.encode("abcd")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.decode((tokenizer.vocab_size,))


def test_pinned_tiny_shakespeare_source_metadata_is_exact() -> None:
    assert TINY_SHAKESPEARE_SOURCE.revision == "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
    assert TINY_SHAKESPEARE_SOURCE.expected_size == 1_115_394
    assert TINY_SHAKESPEARE_SOURCE.expected_sha256 == (
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    )
    assert TINY_SHAKESPEARE_SOURCE.url.endswith(
        "/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
    )


def test_prepare_is_explicit_verified_and_local_load_is_network_free(tmp_path) -> None:
    payload = b"small deterministic corpus\n"
    source_ref = TinyShakespeareSourceRef(
        repository="example/repository",
        revision="immutable-revision",
        relative_path="input.txt",
        expected_sha256=sha256(payload).hexdigest(),
        expected_size=len(payload),
    )
    fetched_urls: list[str] = []
    destination = tmp_path / "tinyshakespeare" / "input.txt"

    prepared_path = prepare_tiny_shakespeare(
        destination,
        source_ref,
        fetch_bytes=lambda url: fetched_urls.append(url) or payload,
    )

    assert prepared_path == destination
    assert fetched_urls == [source_ref.url]
    assert load_tiny_shakespeare(destination, source_ref) == payload.decode("utf-8")
    prepare_tiny_shakespeare(
        destination,
        source_ref,
        fetch_bytes=lambda url: pytest.fail(f"unexpected fetch: {url}"),
    )


def test_prepare_rejects_bad_hash_before_writing(tmp_path) -> None:
    expected = b"expected"
    source_ref = TinyShakespeareSourceRef(
        repository="example/repository",
        revision="revision",
        relative_path="input.txt",
        expected_sha256=sha256(expected).hexdigest(),
        expected_size=len(expected),
    )
    destination = tmp_path / "input.txt"

    with pytest.raises(ValueError, match="integrity check failed"):
        prepare_tiny_shakespeare(
            destination,
            source_ref,
            fetch_bytes=lambda url: b"corrupt!",
        )

    assert not destination.exists()


def test_contiguous_split_is_applied_to_characters_before_windowing() -> None:
    text = "".join(str(index % 10) for index in range(103))
    splits = split_text_contiguously(text)

    assert len(splits.train) == 92
    assert len(splits.validation) == 5
    assert len(splits.test) == 6
    assert splits.train + splits.validation + splits.test == text
    assert splits.validation == text[92:97]


def test_token_batch_validates_shapes_dtypes_immutability_and_jax_pytree() -> None:
    batch = TokenBatch(
        np.asarray([[2, 3]], dtype=np.int64),
        np.asarray([[1, 1]], dtype=np.int8),
        np.asarray([[3, 1]], dtype=np.int64),
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )

    assert batch.input_ids.dtype == np.int32
    assert batch.attention_mask.dtype == np.bool_
    assert not batch.input_ids.flags.writeable
    assert not batch.loss_mask.flags.writeable
    active_count = jax.jit(lambda token_batch: jnp.sum(token_batch.loss_mask))(batch)
    assert int(active_count) == 2

    with pytest.raises(ValueError, match="share one rank-two"):
        TokenBatch(
            np.zeros((1, 2), dtype=np.int32),
            np.zeros((1, 3), dtype=np.bool_),
            np.zeros((1, 2), dtype=np.int32),
            np.zeros((1, 2), dtype=np.bool_),
        )


def test_causal_windows_are_deterministic_fixed_length_and_right_padded() -> None:
    windows = causal_token_windows(
        (2, 3, 4, 5, EOS_TOKEN_ID),
        context_length=3,
        pad_token_id=PAD_TOKEN_ID,
    )

    np.testing.assert_array_equal(windows.input_ids, [[2, 3, 4], [5, 0, 0]])
    np.testing.assert_array_equal(windows.target_ids, [[3, 4, 5], [1, 0, 0]])
    np.testing.assert_array_equal(
        windows.attention_mask,
        [[True, True, True], [True, False, False]],
    )
    np.testing.assert_array_equal(windows.loss_mask, windows.attention_mask)


def test_window_batches_pad_the_final_batch_to_one_fixed_shape() -> None:
    windows = causal_token_windows(
        (2, 3, 4, 5, EOS_TOKEN_ID),
        context_length=3,
        pad_token_id=PAD_TOKEN_ID,
    )

    batches = batch_token_windows(windows, batch_size=3, pad_token_id=PAD_TOKEN_ID)

    assert len(batches) == 1
    assert batches[0].input_ids.shape == (3, 3)
    np.testing.assert_array_equal(batches[0].attention_mask[2], False)
    np.testing.assert_array_equal(batches[0].loss_mask[2], False)


def test_high_level_data_fits_train_vocab_and_windows_each_split_independently() -> None:
    corpus = "ba" * 45 + "a" * 5 + "b" * 5
    preset = TextDataPreset(context_length=4, batch_size=3, stride=4)

    data = build_tiny_shakespeare_data(corpus, preset)

    assert data.tokenizer.vocabulary == (PAD_TOKEN, EOS_TOKEN, "a", "b")
    assert data.text_splits.train == corpus[:90]
    assert data.text_splits.validation == corpus[90:95]
    assert data.text_splits.test == corpus[95:]
    validation_targets = data.validation_batches[0].target_ids[
        data.validation_batches[0].loss_mask
    ]
    test_targets = data.test_batches[0].target_ids[data.test_batches[0].loss_mask]
    assert set(validation_targets.tolist()) <= {
        data.tokenizer.encode("a")[0],
        EOS_TOKEN_ID,
    }
    assert set(test_targets.tolist()) <= {
        data.tokenizer.encode("b")[0],
        EOS_TOKEN_ID,
    }
    assert TINY_SHAKESPEARE_STANDARD_PRESET == TextDataPreset(256, 32, 256)
