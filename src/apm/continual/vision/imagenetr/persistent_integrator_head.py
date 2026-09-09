"""Affine and two-layer heads with semantic input-block persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


class PersistentIntegratorHead(nn.Module):
    """Map concatenated node latents to classes, optionally through a ReLU MLP.

    The first 2C hidden units represent positive and negative source-union
    logits. Their signed output reproduces the affine union exactly at birth.
    Remaining hidden units start with random latent projections and zero output
    weights. All weights train: this is a dense MLP, not a score-input model or
    an affine skip connection. Its hidden coordinate system stays fixed across
    frontier changes, allowing shared state to survive consolidation.
    """

    def __init__(
        self, union_weight: Tensor, union_bias: Tensor, hidden_dimension: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        classes, inputs = union_weight.shape
        if union_bias.shape != (classes,) or (hidden_dimension and hidden_dimension < 2 * classes):
            raise ValueError("MLP width must retain both signs of every source-union logit")
        self.hidden_dimension = hidden_dimension
        self.classes = classes
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.input = nn.Linear(inputs, hidden_dimension or classes)
            self.output = nn.Linear(hidden_dimension, classes) if hidden_dimension else None
        with torch.no_grad():
            self.input.bias.zero_()
            self.input.weight[:classes].copy_(union_weight)
            self.input.bias[:classes].copy_(union_bias)
            if self.output is not None:
                self.input.weight[classes:2 * classes].copy_(-union_weight)
                self.input.bias[classes:2 * classes].copy_(-union_bias)
                self.output.weight.zero_()
                self.output.bias.zero_()
                identity = torch.eye(classes)
                self.output.weight[:, :classes].copy_(identity)
                self.output.weight[:, classes:2 * classes].copy_(-identity)

    def forward(self, features: Tensor) -> Tensor:
        """Apply either one affine layer or two fully connected layers with ReLU."""
        hidden = self.input(features)
        return hidden if self.output is None else self.output(torch.relu(hidden))

    def project_parameter(
        self, name: str, old: Tensor, old_hashes: Sequence[str],
        new_hashes: Sequence[str], old_seen: Sequence[int], initial: Tensor,
    ) -> Tensor:
        """Copy surviving semantic coordinates into source weights or zero moments."""
        output = initial.clone()
        if name == "input.weight":
            feature_dimension = initial.shape[1] // len(new_hashes)
            if old.shape != (initial.shape[0], feature_dimension * len(old_hashes)):
                raise ValueError("prior integrator input shape changed")
            old_index = {node_hash: index for index, node_hash in enumerate(old_hashes)}
            for index, node_hash in enumerate(new_hashes):
                if node_hash in old_index:
                    prior = old_index[node_hash] * feature_dimension
                    start = index * feature_dimension
                    output[:, start:start + feature_dimension].copy_(old[:, prior:prior + feature_dimension])
        else:
            if old.shape != initial.shape:
                raise ValueError("shared integrator coordinate system changed")
            indices = tuple(old_seen)
            if name == "input.bias" and self.hidden_dimension:
                indices += tuple(value + self.classes for value in old_seen)
                indices += tuple(range(2 * self.classes, self.hidden_dimension))
            selected = torch.tensor(indices, dtype=torch.int64)
            output.index_copy_(0, selected.to(output.device), old[selected.to(old.device)].to(output))
        return output

    def carry(
        self, state: Mapping[str, Tensor], old_hashes: Sequence[str],
        new_hashes: Sequence[str], old_seen: Sequence[int],
    ) -> None:
        """Retain surviving input blocks and shared hidden/output state exactly."""
        if set(state) != set(self.state_dict()):
            raise ValueError("prior integrator architecture differs")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(self.project_parameter(
                    name, state[name], old_hashes, new_hashes, old_seen, parameter,
                ))
