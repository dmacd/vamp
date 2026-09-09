"""Small CPU checks for growing-head carry, recall timing, traces, and selection."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.lora import LoRALinear
from apm.continual.vision.imagenetr.srt_config import load_srt_config
from apm.continual.vision.imagenetr.srt_evidence import TraceBuffer, read_event_chunk, trace_batches
from apm.continual.vision.imagenetr.srt_scheduler import (
    RecallPolicy, ReviewBatch, ReviewScheduler, observe_batch, uniform_indices,
)
from apm.continual.vision.imagenetr.srt_training import (
    grow_classifier, optimization_step, optimizer_for, restore_trainables, trainable_state,
)
from apm.continual.vision.imagenetr.srt_workflow import select_candidates


class TinyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2, initialization_seed=1993)
        self.classifier = AffineClassifier(tuple(range(4)), 4, 11993)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(images))


def test_quality_uses_preupdate_all_class_probability_and_only_trainables_change() -> None:
    torch.manual_seed(1)
    model = TinyAdapter()
    optimizer = optimizer_for(model, load_srt_config())
    images, labels = torch.randn(8, 4), torch.arange(8) % 4
    expected = model(images).softmax(-1)[torch.arange(8), labels].detach()
    frozen = {name: parameter.clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    confidence, _, _, gradient = optimization_step(model, optimizer, images, labels, torch.device("cpu"))
    torch.testing.assert_close(torch.tensor(confidence), expected)
    assert gradient > 0 and not torch.equal(model.adapter.lora_b, torch.zeros_like(model.adapter.lora_b))
    assert all(torch.equal(dict(model.named_parameters())[name], value) for name, value in frozen.items())
    assert not torch.equal(model(images).softmax(-1)[torch.arange(8), labels].detach(), expected)


def test_growing_classifier_preserves_old_rows_and_momentum_then_all_rows_train() -> None:
    model = TinyAdapter()
    optimizer = optimizer_for(model, load_srt_config())
    images, labels = torch.randn(8, 4), torch.arange(8) % 4
    optimization_step(model, optimizer, images, labels, torch.device("cpu"))
    old_rows = model.classifier.rows()
    old_momentum = optimizer.state[model.classifier.weight]["momentum_buffer"].clone()
    old_lora = model.adapter.lora_b.detach().clone()
    grow_classifier(model, optimizer, 8, 1993)
    torch.testing.assert_close(model.classifier.weight[:4], old_rows.weight, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[model.classifier.weight]["momentum_buffer"][:4], old_momentum, rtol=0, atol=0)
    assert not torch.count_nonzero(optimizer.state[model.classifier.weight]["momentum_buffer"][4:])
    assert torch.equal(model.adapter.lora_b, old_lora)
    optimization_step(model, optimizer, images, torch.arange(8), torch.device("cpu"))
    assert not torch.equal(model.classifier.weight[:4], old_rows.weight)


def test_weight_and_optimizer_resume_is_exact_on_cpu() -> None:
    model = TinyAdapter()
    optimizer = optimizer_for(model, load_srt_config())
    images, labels = torch.randn(8, 4), torch.arange(8) % 4
    optimization_step(model, optimizer, images, labels, torch.device("cpu"))
    resumed = deepcopy(model)
    following = optimizer_for(resumed, load_srt_config())
    restore_trainables(resumed, trainable_state(model))
    following.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(3):
        optimization_step(model, optimizer, images, labels, torch.device("cpu"))
        optimization_step(resumed, following, images, labels, torch.device("cpu"))
    assert all(torch.equal(value, trainable_state(resumed)[name]) for name, value in trainable_state(model).items())
    with pytest.raises(ValueError):
        restore_trainables(resumed, {})


def test_trace_commit_does_not_adopt_orphan_chunks(tmp_path) -> None:
    policy = RecallPolicy("standard", (.1, .25, .5, .75, .9), .5, 1)
    _, events = observe_batch(ReviewScheduler().arrive(1), ReviewBatch((0,), 0, 1, True), (.8,), policy, "srt", 1, 0)
    batch = {"step": 1, "stage": 1, "presentations": 1}
    committed = TraceBuffer(events, (batch,)).publish(tmp_path)
    TraceBuffer(events, ({**batch, "step": 2},)).publish(tmp_path)
    assert trace_batches(tmp_path, (committed,)) == (batch,)
    assert read_event_chunk(tmp_path / committed["path"] / "events.parquet") == events
    assert len(list((tmp_path / "trace").iterdir())) == 2


def test_sparse_uniform_draw_and_selection_ties() -> None:
    for count in (0, 1, 9, 100):
        values = uniform_indices(100, count, np.random.RandomState(13))
        assert len(set(values)) == count and all(0 <= value < 100 for value in values)
    candidates = ({"mean_accuracy": 80, "mean_nll": 1.0, "policy": {"a": 2}},
                  {"mean_accuracy": 80, "mean_nll": .9, "policy": {"a": 2}},
                  {"mean_accuracy": 80, "mean_nll": .9, "policy": {"a": 1}},
                  {"mean_accuracy": 79, "mean_nll": .1, "policy": {"a": 1}})
    assert select_candidates(candidates, 2) == (candidates[2], candidates[1])
