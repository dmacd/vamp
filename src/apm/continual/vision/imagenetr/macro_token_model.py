"""Macro-token transformer integration over fixed ImageNet-R hierarchy slots."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASS_COUNT = 200
MAXIMUM_SLOTS = 6
TOKEN_COUNT = 197
TOKEN_DIMENSION = 768
META_SLOT_DIMENSION = 3 * CLASS_COUNT + 1
META_INPUT_DIMENSION = MAXIMUM_SLOTS * META_SLOT_DIMENSION


@dataclass(frozen=True, slots=True)
class MacroTokenInputs:
    """Detached label-free node tokens and behavior fields for one image batch."""

    node_tokens: Tensor
    slot_indices: Tensor
    meta_features: Tensor
    raw_scores: Tensor
    ownership: Tensor
    active_slot_mask: Tensor
    seen_class_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = len(self.node_tokens)
        active_nodes = len(self.slot_indices)
        expected_active = torch.where(self.active_slot_mask)[0].to(torch.int64)
        if (
            self.node_tokens.shape
            != (batch_size, active_nodes, TOKEN_COUNT, TOKEN_DIMENSION)
            or self.slot_indices.shape != (active_nodes,)
            or self.slot_indices.dtype != torch.int64
            or not torch.equal(self.slot_indices.cpu(), expected_active.cpu())
            or self.meta_features.shape != (batch_size, META_INPUT_DIMENSION)
            or self.raw_scores.shape
            != (batch_size, MAXIMUM_SLOTS, CLASS_COUNT)
            or self.ownership.shape != (MAXIMUM_SLOTS, CLASS_COUNT)
            or self.ownership.dtype != torch.bool
            or self.active_slot_mask.shape != (MAXIMUM_SLOTS,)
            or self.active_slot_mask.dtype != torch.bool
            or self.seen_class_mask.shape != (CLASS_COUNT,)
            or self.seen_class_mask.dtype != torch.bool
            or not bool(self.active_slot_mask.any())
            or not torch.equal(self.ownership.any(dim=0), self.seen_class_mask)
            or any(
                tensor.requires_grad
                for tensor in (self.node_tokens, self.meta_features, self.raw_scores)
            )
            or not bool(torch.isfinite(self.node_tokens).all())
            or not bool(torch.isfinite(self.meta_features).all())
            or not bool(torch.isfinite(self.raw_scores).all())
        ):
            raise ValueError("macro-token inputs are malformed or attached")

    def to(self, device: torch.device) -> "MacroTokenInputs":
        """Move one detached batch to the requested device."""
        return MacroTokenInputs(
            self.node_tokens.to(device=device, dtype=torch.float32, non_blocking=True),
            self.slot_indices.to(device=device, non_blocking=True),
            self.meta_features.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            self.raw_scores.to(device=device, dtype=torch.float32, non_blocking=True),
            self.ownership.to(device=device, non_blocking=True),
            self.active_slot_mask.to(device=device, non_blocking=True),
            self.seen_class_mask.to(device=device, non_blocking=True),
        )


@dataclass(frozen=True, slots=True)
class MacroTokenSupervision:
    """Class and diagnostic owner targets kept outside the inference inputs."""

    inputs: MacroTokenInputs
    labels: Tensor
    owner_targets: Tensor

    def __post_init__(self) -> None:
        batch_size = len(self.inputs.node_tokens)
        if (
            self.labels.shape != (batch_size,)
            or self.owner_targets.shape != (batch_size,)
            or self.labels.dtype != torch.int64
            or self.owner_targets.dtype != torch.int64
            or self.labels.requires_grad
            or self.owner_targets.requires_grad
            or not bool(self.inputs.seen_class_mask[self.labels].all())
            or not bool(self.inputs.active_slot_mask[self.owner_targets].all())
            or not bool(self.inputs.ownership[self.owner_targets, self.labels].all())
        ):
            raise ValueError("macro-token supervision is malformed")


def behavior_meta_features(
    raw_scores: Tensor, ownership: Tensor, active_slot_mask: Tensor
) -> Tensor:
    """Build the exact 3,606-value raw/local/ownership/active META input."""
    batch_size = len(raw_scores)
    if (
        raw_scores.shape != (batch_size, MAXIMUM_SLOTS, CLASS_COUNT)
        or ownership.shape != (MAXIMUM_SLOTS, CLASS_COUNT)
        or ownership.dtype != torch.bool
        or active_slot_mask.shape != (MAXIMUM_SLOTS,)
        or active_slot_mask.dtype != torch.bool
    ):
        raise ValueError("META source tensors have the wrong shape")
    local = torch.stack(
        tuple(
            torch.zeros_like(raw_scores[:, slot]).index_copy(
                1,
                torch.where(ownership[slot])[0],
                F.log_softmax(raw_scores[:, slot, ownership[slot]].float(), dim=1).to(
                    raw_scores.dtype
                ),
            )
            if active_slot_mask[slot]
            else torch.zeros_like(raw_scores[:, slot])
            for slot in range(MAXIMUM_SLOTS)
        ),
        dim=1,
    )
    repeated_ownership = ownership.to(raw_scores.dtype)[None].expand(batch_size, -1, -1)
    repeated_active = active_slot_mask.to(raw_scores.dtype)[None, :, None].expand(
        batch_size, -1, -1
    )
    return torch.cat(
        (raw_scores, local, repeated_ownership, repeated_active), dim=2
    ).flatten(1)


def class_owner_targets(labels: Tensor, ownership: Tensor) -> Tensor:
    """Return the unique active hierarchy slot owning each supervised class."""
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("owner targets require one-dimensional integer labels")
    matches = ownership[:, labels].T
    if matches.shape != (len(labels), MAXIMUM_SLOTS) or not bool(
        (matches.sum(dim=1) == 1).all()
    ):
        raise ValueError("every label must have exactly one owning slot")
    return matches.to(torch.int64).argmax(dim=1)


def behavior_control_features(inputs: MacroTokenInputs) -> tuple[Tensor, Tensor]:
    """Reconstruct the v6 final-CLS behavior vector and raw-union baseline."""
    batch_size = len(inputs.node_tokens)
    final_tokens = torch.zeros(
        (batch_size, MAXIMUM_SLOTS, TOKEN_DIMENSION),
        dtype=inputs.node_tokens.dtype,
        device=inputs.node_tokens.device,
    )
    final_tokens[:, inputs.slot_indices] = inputs.node_tokens[:, :, 0]
    slot_meta = inputs.meta_features.reshape(
        batch_size, MAXIMUM_SLOTS, META_SLOT_DIMENSION
    )
    features = torch.cat((final_tokens, slot_meta), dim=2).flatten(1)
    baseline = torch.full(
        (batch_size, CLASS_COUNT),
        -torch.inf,
        dtype=inputs.raw_scores.dtype,
        device=inputs.raw_scores.device,
    )
    for slot in torch.where(inputs.active_slot_mask)[0].tolist():
        owned = inputs.ownership[slot]
        baseline[:, owned] = inputs.raw_scores[:, slot, owned]
    return features, baseline


class MacroTransformerBlock(nn.Module):
    """One pre-normalized 12-head transformer block with a four-times MLP."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(TOKEN_DIMENSION, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            TOKEN_DIMENSION,
            12,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(TOKEN_DIMENSION, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(TOKEN_DIMENSION, 4 * TOKEN_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * TOKEN_DIMENSION, TOKEN_DIMENSION),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Integrate spatial and META tokens without returning attention weights."""
        normalized = self.attention_norm(tokens)
        attended = self.attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = tokens + self.attention_output_dropout(attended)
        return tokens + self.mlp(self.mlp_norm(tokens))


def _component_seed(seed: int, name: str) -> int:
    return int(
        sha256(f"imagenetr50-macro-token-v1\0{seed}\0{name}".encode()).hexdigest()[:15],
        16,
    )


class MacroTokenEncoder(nn.Module):
    """Fuse corresponding node tokens and integrate them with one META token."""

    def __init__(self, depth: int, dropout: float, seed: int) -> None:
        super().__init__()
        if depth not in (1, 2) or not 0.0 <= dropout < 1.0 or seed < 0:
            raise ValueError("macro-token encoder configuration is invalid")
        self.depth = depth
        self.slot_projection_weight = nn.Parameter(
            torch.empty(TOKEN_DIMENSION, MAXIMUM_SLOTS, TOKEN_DIMENSION)
        )
        self.slot_projection_bias = nn.Parameter(torch.empty(TOKEN_DIMENSION))
        self.position_embedding = nn.Parameter(
            torch.empty(1, TOKEN_COUNT, TOKEN_DIMENSION)
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(META_INPUT_DIMENSION, 256),
            nn.GELU(),
            nn.LayerNorm(256, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(256, TOKEN_DIMENSION),
        )
        self.blocks = nn.ModuleList(
            MacroTransformerBlock(dropout) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(TOKEN_DIMENSION, eps=1e-6)
        self._reset_parameters(seed)

    def _reset_parameters(self, seed: int) -> None:
        devices = tuple(
            sorted(
                {
                    parameter.device.index or 0
                    for parameter in self.parameters()
                    if parameter.device.type == "cuda"
                }
            )
        )
        for name, parameter in self.named_parameters():
            with torch.random.fork_rng(devices=list(devices)):
                torch.manual_seed(_component_seed(seed, name))
                with torch.no_grad():
                    if name == "position_embedding":
                        nn.init.trunc_normal_(parameter, std=0.02)
                    elif name == "slot_projection_weight":
                        nn.init.xavier_uniform_(
                            parameter.reshape(TOKEN_DIMENSION, -1)
                        )
                    elif name.endswith("weight") and parameter.ndim >= 2:
                        nn.init.xavier_uniform_(parameter)
                    elif name.endswith("weight"):
                        parameter.fill_(1.0)
                    else:
                        parameter.zero_()

    def forward(
        self, node_tokens: Tensor, slot_indices: Tensor, meta_features: Tensor
    ) -> Tensor:
        """Return the final macro-CLS representation for classification or probing."""
        batch_size, active_nodes = node_tokens.shape[:2]
        if (
            node_tokens.shape
            != (batch_size, active_nodes, TOKEN_COUNT, TOKEN_DIMENSION)
            or slot_indices.shape != (active_nodes,)
            or slot_indices.dtype != torch.int64
            or meta_features.shape != (batch_size, META_INPUT_DIMENSION)
            or bool(torch.any(slot_indices < 0))
            or bool(torch.any(slot_indices >= MAXIMUM_SLOTS))
            or len(torch.unique(slot_indices)) != active_nodes
        ):
            raise ValueError("macro-token encoder inputs have the wrong shape")
        selected_weights = self.slot_projection_weight[:, slot_indices, :]
        macro_tokens = torch.einsum(
            "bapd,oad->bpo", node_tokens, selected_weights
        ) + self.slot_projection_bias
        macro_tokens = macro_tokens + self.position_embedding
        meta_token = self.meta_encoder(meta_features)[:, None]
        tokens = torch.cat((macro_tokens, meta_token), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)[:, 0]


class MacroTokenClassifier(nn.Module):
    """Direct 200-way classifier over the integrated macro-CLS token."""

    def __init__(self, depth: int, dropout: float, seed: int) -> None:
        super().__init__()
        self.encoder = MacroTokenEncoder(depth, dropout, seed)
        self.classifier = nn.Linear(TOKEN_DIMENSION, CLASS_COUNT)
        self._reset_classifier(seed)

    def _reset_classifier(self, seed: int) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(_component_seed(seed, "classifier.weight"))
            nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, inputs: MacroTokenInputs) -> Tensor:
        """Expose macro-CLS for the explicitly diagnostic frozen linear probe."""
        return self.encode_components(
            inputs.node_tokens, inputs.slot_indices, inputs.meta_features
        )

    def encode_components(
        self, node_tokens: Tensor, slot_indices: Tensor, meta_features: Tensor
    ) -> Tensor:
        """Encode either detached caches or attached node outputs identically."""
        return self.encoder(node_tokens, slot_indices, meta_features)

    def forward_components(
        self,
        node_tokens: Tensor,
        slot_indices: Tensor,
        meta_features: Tensor,
        seen_class_mask: Tensor,
    ) -> Tensor:
        """Classify differentiable node components without weakening cache checks."""
        if (
            seen_class_mask.shape != (CLASS_COUNT,)
            or seen_class_mask.dtype != torch.bool
            or not bool(seen_class_mask.any())
        ):
            raise ValueError("macro-token seen-class mask is malformed")
        logits = self.classifier(
            self.encode_components(node_tokens, slot_indices, meta_features)
        )
        return logits.masked_fill(~seen_class_mask, -torch.inf)

    def forward(self, inputs: MacroTokenInputs) -> Tensor:
        """Return direct class logits with classes outside the frontier masked."""
        return self.forward_components(
            inputs.node_tokens,
            inputs.slot_indices,
            inputs.meta_features,
            inputs.seen_class_mask,
        )


class MacroOwnerClassifier(nn.Module):
    """Diagnostic task-free predictor of the active slot owning the true class."""

    def __init__(self, depth: int, dropout: float, seed: int) -> None:
        super().__init__()
        self.encoder = MacroTokenEncoder(depth, dropout, seed)
        self.classifier = nn.Linear(TOKEN_DIMENSION, MAXIMUM_SLOTS)
        with torch.random.fork_rng():
            torch.manual_seed(_component_seed(seed, "owner_classifier.weight"))
            nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs: MacroTokenInputs) -> Tensor:
        """Return six slot logits while masking levels absent from the frontier."""
        logits = self.classifier(
            self.encoder(inputs.node_tokens, inputs.slot_indices, inputs.meta_features)
        )
        return logits.masked_fill(~inputs.active_slot_mask, -torch.inf)


def parameter_count(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def predicted_owner_class_predictions(
    inputs: MacroTokenInputs, owner_predictions: Tensor
) -> Tensor:
    """Classify within the raw-logit rows of each predicted owner slot."""
    if owner_predictions.shape != (len(inputs.node_tokens),):
        raise ValueError("owner predictions do not align with the input batch")
    predictions = torch.empty_like(owner_predictions)
    for slot in torch.where(inputs.active_slot_mask)[0].tolist():
        rows = owner_predictions == slot
        owned = inputs.ownership[slot]
        class_ids = torch.where(owned)[0]
        predictions[rows] = class_ids[
            inputs.raw_scores[rows, slot][:, owned].argmax(dim=1)
        ]
    return predictions


__all__ = [
    "CLASS_COUNT",
    "MAXIMUM_SLOTS",
    "META_INPUT_DIMENSION",
    "META_SLOT_DIMENSION",
    "MacroOwnerClassifier",
    "MacroTokenClassifier",
    "MacroTokenEncoder",
    "MacroTokenInputs",
    "MacroTokenSupervision",
    "TOKEN_COUNT",
    "TOKEN_DIMENSION",
    "behavior_control_features",
    "behavior_meta_features",
    "class_owner_targets",
    "parameter_count",
    "predicted_owner_class_predictions",
]
