from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as functional

from apm.continual.artifacts import publish_immutable_json, record_sha256
from apm.continual.trace.artifacts import publish_artifact_directory
from apm.continual.trace.collator import TraceDataCollator
from apm.continual.trace.data import TraceExample
from apm.continual.trace.protocol import TrainingConfig
from apm.continual.trace.training import train_adapter
from apm.continual.trace.training_jobs import run_training_artifact
from apm.continual.trace.training_plans import TrainingPlan


@dataclass
class CharacterTokenizer:
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int | None = 0
    padding_side: str = "left"
    truncation_side: str = "left"

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: None,
    ) -> dict[str, list[int]]:
        del add_special_tokens, padding, return_tensors
        values = [3 + ord(character) % 13 for character in text]
        if truncation:
            values = values[-max_length:]
        return {"attention_mask": [1] * len(values), "input_ids": values}

    def batch_decode(
        self,
        sequences: Tensor,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        del skip_special_tokens, clean_up_tokenization_spaces
        return ["".join(chr(int(value)) for value in row) for row in sequences]


class TinyAdapterLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(99)
        self.register_buffer("base", torch.randn(19, 19, generator=generator) * 0.05)
        self.adapter = torch.nn.Parameter(torch.zeros(19, 19))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        one_hot = functional.one_hot(input_ids, num_classes=19).to(torch.float32)
        logits = self.dropout(one_hot) @ (self.base + self.adapter)
        loss = functional.cross_entropy(
            logits[:, :-1].reshape(-1, 19),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


def _training_fixture() -> tuple[dict[str, TraceExample], TrainingPlan]:
    examples = tuple(
        TraceExample(
            example_id=record_sha256({"example": index}),
            task="C-STANCE",
            split="train",
            source_index=index,
            prompt=f"question {index}: ",
            answer=f"answer {index}",
            arrival=1,
        )
        for index in range(10)
    )
    order = tuple(example.example_id for example in examples) * 2
    return (
        {example.example_id: example for example in examples},
        TrainingPlan("tiny", order, (("tiny", len(order)),)),
    )


def test_collator_masks_every_prompt_and_padding_token() -> None:
    collator = TraceDataCollator(CharacterTokenizer(), TrainingConfig(), 8)

    batch = collator.training_batch(
        (
            {"prompt": "long prompt ", "answer": "yes"},
            {"prompt": "p ", "answer": "no"},
        )
    )

    assert batch["input_ids"].shape[1] % 8 == 0
    for input_ids, mask, labels in zip(
        batch["input_ids"], batch["attention_mask"], batch["labels"]
    ):
        supervised = labels != -100
        assert torch.all(mask[supervised] == 1)
        assert torch.equal(labels[supervised], input_ids[supervised])
        assert torch.all(labels[~supervised] == -100)


def test_prompt_only_collation_never_adds_an_answer_or_eos() -> None:
    tokenizer = CharacterTokenizer()
    collator = TraceDataCollator(tokenizer)

    batch = collator.inference_batch(("abc",))

    visible = batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()
    assert visible[0] == tokenizer.bos_token_id
    assert visible[-1] != tokenizer.eos_token_id
    assert "labels" not in batch


def test_training_resume_matches_uninterrupted_adapter_exactly(tmp_path: Path) -> None:
    examples, plan = _training_fixture()
    config = TrainingConfig(learning_rate=1.0e-3)
    uninterrupted = TinyAdapterLM()
    resumed = TinyAdapterLM()

    train_adapter(
        uninterrupted,
        CharacterTokenizer(),
        examples,
        plan,
        tmp_path / "full.pt",
        tmp_path / "full.jsonl",
        config,
        checkpoint_step_interval=1,
    )
    calls = 0

    def pause_after_first_step() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    with pytest.raises(InterruptedError, match="safe boundary"):
        train_adapter(
            resumed,
            CharacterTokenizer(),
            examples,
            plan,
            tmp_path / "resume.pt",
            tmp_path / "resume.jsonl",
            config,
            checkpoint_step_interval=1,
            should_pause=pause_after_first_step,
        )
    result = train_adapter(
        resumed,
        CharacterTokenizer(),
        examples,
        plan,
        tmp_path / "resume.pt",
        tmp_path / "resume.jsonl",
        config,
        checkpoint_step_interval=1,
    )

    torch.testing.assert_close(resumed.adapter, uninterrupted.adapter, rtol=0.0, atol=0.0)
    assert result.presentations == 20
    assert result.optimizer_steps == 3


def test_published_training_artifact_is_reused_without_loading_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples, plan = _training_fixture()
    source = tmp_path / "published-source"
    source.mkdir()
    (source / "adapter.safetensors").write_bytes(b"already-trained")
    publish_immutable_json(
        source / "train_metrics.json",
        {
            "checkpoint_path": str(tmp_path / "checkpoint.pt"),
            "elapsed_seconds": 1.0,
            "format": "trace-training-metrics-v1",
            "mean_loss": 0.5,
            "optimizer_steps": 3,
            "plan_hash": plan.plan_hash,
            "presentations": len(plan.example_ids),
            "tokens": 123,
        },
    )
    target = tmp_path / "published"
    identity = publish_artifact_directory(source, target)

    def fail_model_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("immutable adapter reuse must not construct a model")

    monkeypatch.setattr(
        "apm.continual.trace.training_jobs.load_fresh_lora_bundle",
        fail_model_load,
    )
    result = run_training_artifact(
        plan,
        examples,
        "model-revision",
        "cpu",
        target,
        tmp_path / "checkpoint.pt",
        tmp_path / "steps.jsonl",
        tmp_path / "work",
    )

    assert result.artifact_sha256 == identity
    assert result.training.tokens == 123
