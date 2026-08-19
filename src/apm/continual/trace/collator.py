"""Reference-compatible causal-LM collation without chat templating."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
from torch import Tensor

from apm.continual.trace.protocol import TrainingConfig


class TokenizedText(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]


class Tokenizer(Protocol):
    """The narrow Hugging Face tokenizer surface used by TRACE."""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_token_id: int | None
    padding_side: str
    truncation_side: str

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: None,
    ) -> TokenizedText: ...

    def batch_decode(
        self,
        sequences: Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]: ...


class TraceTensorBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor


@dataclass(frozen=True, slots=True)
class TraceDataCollator:
    """Port the TreeLoRA left-padding and answer-only label semantics exactly."""

    tokenizer: Tokenizer
    config: TrainingConfig = TrainingConfig()
    pad_to_multiple_of: int = 8
    label_pad_token_id: int = -100

    def __post_init__(self) -> None:
        if (
            self.tokenizer.padding_side != "left"
            or self.tokenizer.truncation_side != "left"
            or self.tokenizer.pad_token_id is None
            or self.pad_to_multiple_of <= 0
        ):
            raise ValueError("TRACE requires a left-padding, left-truncating tokenizer")

    def training_batch(self, rows: Sequence[Mapping[str, str]]) -> TraceTensorBatch:
        """Collate prompt-plus-answer rows with loss only on final answer tokens."""
        if not rows:
            raise ValueError("cannot collate an empty TRACE batch")
        limit = self.config.max_prompt_length + self.config.max_answer_length
        tokenized = tuple(
            (
                self._tokenize(
                    str(row["prompt"]) + str(row["answer"]),
                    limit,
                    add_bos=True,
                    add_eos=True,
                ),
                len(
                    self._tokenize(
                        str(row["answer"]),
                        limit,
                        add_bos=False,
                        add_eos=True,
                    )["input_ids"]
                ),
            )
            for row in rows
        )
        padded_length = self._padded_length(
            max(len(item[0]["input_ids"]) for item in tokenized)
        )
        input_ids = tuple(
            [self.tokenizer.pad_token_id] * (padded_length - len(item["input_ids"]))
            + item["input_ids"]
            for item, _ in tokenized
        )
        attention = tuple(
            [0] * (padded_length - len(item["attention_mask"]))
            + item["attention_mask"]
            for item, _ in tokenized
        )
        labels = tuple(
            [self.label_pad_token_id] * (padded_length - label_length)
            + item["input_ids"][-label_length:]
            for item, label_length in tokenized
        )
        return {
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def inference_batch(self, prompts: Sequence[str]) -> dict[str, Tensor]:
        """Collate prompt-only inputs without adding EOS or exposing answers."""
        if not prompts:
            raise ValueError("cannot collate an empty TRACE prompt batch")
        tokenized = tuple(
            self._tokenize(
                prompt,
                self.config.max_prompt_length,
                add_bos=True,
                add_eos=False,
            )
            for prompt in prompts
        )
        padded_length = self._padded_length(
            max(len(item["input_ids"]) for item in tokenized)
        )
        return {
            "attention_mask": torch.tensor(
                tuple(
                    [0] * (padded_length - len(item["attention_mask"]))
                    + item["attention_mask"]
                    for item in tokenized
                ),
                dtype=torch.long,
            ),
            "input_ids": torch.tensor(
                tuple(
                    [self.tokenizer.pad_token_id] * (padded_length - len(item["input_ids"]))
                    + item["input_ids"]
                    for item in tokenized
                ),
                dtype=torch.long,
            ),
        }

    def _tokenize(
        self,
        text: str,
        cutoff: int,
        *,
        add_bos: bool,
        add_eos: bool,
    ) -> TokenizedText:
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=cutoff,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )
        input_ids = list(result["input_ids"])
        attention = list(result["attention_mask"])
        if len(input_ids) < cutoff and add_eos:
            input_ids.append(self.tokenizer.eos_token_id)
            attention.append(1)
        if len(input_ids) < cutoff and add_bos:
            input_ids.insert(0, self.tokenizer.bos_token_id)
            attention.insert(0, 1)
        return {"attention_mask": attention, "input_ids": input_ids}

    def _padded_length(self, length: int) -> int:
        return (
            (length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of


__all__ = ["Tokenizer", "TraceDataCollator", "TraceTensorBatch"]
