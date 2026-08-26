import torch
from torch.nn import functional as F

from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoOptimizerConfig,
    sum_top_two_adapters,
    top_two_base_state,
    top_two_logits,
    train_top_two_adapter_step,
    zero_top_two_adapter,
    zero_top_two_adamw,
)


def _base():
    embedding = torch.nn.Linear(4, 3)
    classifier = torch.nn.Linear(3, 2)
    return top_two_base_state(
        embedding.weight,
        embedding.bias,
        classifier.weight,
        classifier.bias,
    )


def test_zero_top_two_adapter_preserves_base_suffix_exactly() -> None:
    torch.manual_seed(3)
    base = _base()
    features = torch.randn(5, 4)
    expected = F.linear(
        F.relu(F.linear(features, base.embedding_weight, base.embedding_bias)),
        base.classifier_weight,
        base.classifier_bias,
    )
    actual = top_two_logits(features, base, zero_top_two_adapter(base))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_cumulative_top_two_adapters_sum_each_parameter_delta() -> None:
    base = _base()
    one = TopTwoAdapterState(*(torch.ones_like(tensor) for tensor in base.tensors))
    two = TopTwoAdapterState(*(2.0 * torch.ones_like(tensor) for tensor in base.tensors))
    total = sum_top_two_adapters((one, two), base)
    for tensor in total.tensors:
        torch.testing.assert_close(tensor, 3.0 * torch.ones_like(tensor))


def test_appending_a_zero_child_preserves_realistic_parent_logits_exactly() -> None:
    torch.manual_seed(11)
    embedding = torch.nn.Linear(64 * 7 * 7, 128)
    classifier = torch.nn.Linear(128, 10)
    base = top_two_base_state(
        embedding.weight,
        embedding.bias,
        classifier.weight,
        classifier.bias,
    )
    parent = TopTwoAdapterState(*(torch.randn_like(tensor) for tensor in base.tensors))
    zero = zero_top_two_adapter(base)
    features = torch.randn(128, 64 * 7 * 7)
    before = top_two_logits(features, base, sum_top_two_adapters((parent,), base))
    after = top_two_logits(features, base, sum_top_two_adapters((parent, zero), base))
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_explicit_top_two_adamw_step_matches_torch_reference() -> None:
    torch.manual_seed(7)
    base = _base()
    features = torch.randn(8, 4)
    labels = torch.arange(8) % 2
    initial = zero_top_two_adapter(base)
    config = TopTwoOptimizerConfig(0.001, 0.0001, 0.9, 0.999, 1e-8)
    actual, _optimizer, _loss = train_top_two_adapter_step(
        features,
        labels,
        base,
        zero_top_two_adapter(base),
        initial,
        zero_top_two_adamw(initial),
        config,
    )

    parameters = [torch.nn.Parameter(torch.zeros_like(tensor)) for tensor in base.tensors]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    embedding_weight, embedding_bias, classifier_weight, classifier_bias = parameters
    hidden = F.relu(
        F.linear(
            features,
            base.embedding_weight + embedding_weight,
            base.embedding_bias + embedding_bias,
        )
    )
    logits = F.linear(
        hidden,
        base.classifier_weight + classifier_weight,
        base.classifier_bias + classifier_bias,
    )
    F.cross_entropy(logits, labels).backward()
    optimizer.step()
    for committed, reference in zip(actual.tensors, parameters):
        torch.testing.assert_close(committed, reference, rtol=1e-6, atol=1e-7)
