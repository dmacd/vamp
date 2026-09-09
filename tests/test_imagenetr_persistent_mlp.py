"""Dense MLP initialization, semantic persistence, and matched-protocol checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from apm.continual.vision.imagenetr.persistent_affine_config import load_persistent_affine_config
from apm.continual.vision.imagenetr.persistent_affine_training import (
    create_optimizer, export_named_optimizer_state, restore_named_optimizer_state,
)
from apm.continual.vision.imagenetr.persistent_integrator_head import PersistentIntegratorHead
from apm.continual.vision.imagenetr.persistent_mlp_config import load_persistent_mlp_config


def test_mlp_is_two_dense_layers_with_exact_union_initialization_and_free_latent_units() -> None:
    generator = torch.Generator().manual_seed(21)
    weight, bias = torch.randn(3, 10, generator=generator), torch.randn(3, generator=generator)
    inputs = torch.randn(13, 10, generator=generator)
    head = PersistentIntegratorHead(weight, bias, 16, 1993)
    torch.testing.assert_close(head(inputs), inputs @ weight.T + bias)
    assert sum(isinstance(module, nn.Linear) for module in head.modules()) == 2
    assert torch.count_nonzero(head.input.weight[6:]) > 0
    assert torch.count_nonzero(head.output.weight[:, 6:]) == 0
    torch.nn.functional.cross_entropy(head(inputs), torch.arange(13) % 3).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    assert torch.count_nonzero(head.output.weight.grad[:, 6:]) > 0
    repeated = PersistentIntegratorHead(weight, bias, 16, 1993)
    assert all(torch.equal(value, repeated.state_dict()[name]) for name, value in head.state_dict().items())


def test_mlp_hidden_coordinates_and_output_rows_survive_frontier_replacement() -> None:
    old = PersistentIntegratorHead(torch.zeros(3, 10), torch.zeros(3), 16, 3)
    with torch.no_grad():
        for index, parameter in enumerate(old.parameters(), 1):
            parameter.fill_(index)
        old.input.weight[:, 5:].fill_(7)
    current = PersistentIntegratorHead(torch.randn(3, 10), torch.randn(3), 16, 4)
    initial = {name: value.clone() for name, value in current.state_dict().items()}
    current.carry(old.state_dict(), ("retired", "survivor"), ("survivor", "new"), (0, 1))
    assert torch.all(current.input.weight[:, :5] == 7)
    torch.testing.assert_close(current.input.weight[:, 5:], initial["input.weight"][:, 5:])
    assert torch.all(current.input.bias[[0, 1, 3, 4, 6, 15]] == 2)
    torch.testing.assert_close(current.input.bias[[2, 5]], initial["input.bias"][[2, 5]])
    assert torch.all(current.output.weight[:2] == 3)
    torch.testing.assert_close(current.output.weight[2], initial["output.weight"][2])
    assert torch.all(current.output.bias[:2] == 4)
    torch.testing.assert_close(current.output.bias[2], initial["output.bias"][2])
    clone = PersistentIntegratorHead(torch.zeros(3, 10), torch.zeros(3), 16, 1)
    clone.load_state_dict(current.state_dict(), strict=True)
    values = torch.randn(4, 10)
    torch.testing.assert_close(clone(values), current(values), rtol=0, atol=0)


class _TinyNode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.factors = nn.ModuleList(nn.Module() for _ in range(24))
        for module in self.factors:
            module.register_parameter("lora_a", nn.Parameter(torch.ones(1)))
            module.register_parameter("lora_b", nn.Parameter(torch.ones(1)))


def _optimizer_fixture(hashes: tuple[str, ...]) -> SimpleNamespace:
    nodes = tuple(_TinyNode() for _ in hashes)
    return SimpleNamespace(
        integrator=PersistentIntegratorHead(torch.randn(200, 4 * len(nodes)), torch.zeros(200), 1024, 19),
        node_hashes=hashes, node_models=nodes, seen_class_ids=(0, 1, 2, 3),
        lora_parameters=tuple(parameter for node in nodes for parameter in node.parameters()),
    )


def test_mlp_named_adam_state_carries_shared_layers_and_only_surviving_node_columns() -> None:
    config = load_persistent_affine_config()
    old = _optimizer_fixture(("old", "survivor"))
    optimizer = create_optimizer(old, config)
    sum(parameter.square().sum() for group in optimizer.param_groups for parameter in group["params"]).backward()
    optimizer.step()
    snapshot = export_named_optimizer_state(optimizer, old)
    current = _optimizer_fixture(("survivor", "new"))
    continued = create_optimizer(current, config)
    assert restore_named_optimizer_state(continued, current, snapshot) == 52
    input_moment = continued.state[current.integrator.input.weight]["exp_avg"]
    old_moment = optimizer.state[old.integrator.input.weight]["exp_avg"]
    torch.testing.assert_close(input_moment[:, :4], old_moment[:, 4:], rtol=0, atol=0)
    assert torch.count_nonzero(input_moment[:, 4:]) == 0
    output_moment = continued.state[current.integrator.output.weight]["exp_avg"]
    torch.testing.assert_close(output_moment[:4], optimizer.state[old.integrator.output.weight]["exp_avg"][:4], rtol=0, atol=0)
    assert torch.count_nonzero(output_moment[4:]) == 0
    assert all(parameter not in continued.state for parameter in current.node_models[1].parameters())
    assert all(parameter in continued.state for parameter in current.node_models[0].parameters())


def test_mlp_config_is_one_h4096_condition_and_head_size_is_linear_in_live_nodes() -> None:
    config = load_persistent_mlp_config()
    assert config.historical_capacity == 4096 and config.hidden_dimension == 1024
    with pytest.raises(ValueError, match="single H=4,096"):
        replace(config, historical_capacity=8192)
    with pytest.raises(ValueError, match="single H=4,096"):
        replace(config, hidden_dimension=2048)
    for nodes in (1, 3, 5):
        head = PersistentIntegratorHead(torch.zeros(200, 768 * nodes), torch.zeros(200), 1024, 1993)
        assert sum(parameter.numel() for parameter in head.parameters()) == 786432 * nodes + 206024
