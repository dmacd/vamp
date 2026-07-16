from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket

import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.attention import attention_pattern
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.generation import greedy_generate
from apm.lm.gpt_neo import apply_gpt_neo, embed_tokens
from apm.lm.losses import mean_token_nll
from apm.lm.parity import (
    ParitySnapshot,
    assert_parity,
    compare_parity_snapshots,
    ordered_capture_spec,
)


@pytest.mark.integration
def test_local_tinystories_checkpoint_matches_hugging_face(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the complete parity ladder using only explicitly prepared local artifacts."""
    hf_directory = _prepared_directory("TINYSTORIES_HF_MODEL_DIR")
    jax_directory = _prepared_directory("TINYSTORIES_JAX_CHECKPOINT_DIR")
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        pytest.skip("TinyStories parity requires the hf-convert optional dependencies")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    class NetworkDeniedSocket(socket.socket):
        def connect(self, address):
            raise AssertionError(f"network access is forbidden during local parity: {address}")

        def connect_ex(self, address):
            raise AssertionError(f"network access is forbidden during local parity: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)

    import torch
    import torch.nn.functional as torch_functional
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded = load_gpt_neo_checkpoint(jax_directory)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_directory,
        local_files_only=True,
    ).eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_directory,
        local_files_only=True,
        use_fast=True,
    )
    runtime_tokenizer = Tokenizer.from_file(
        str(jax_directory.parent / "tokenizer" / "tokenizer.json")
    )
    if getattr(hf_model.config, "model_type", None) != "gpt_neo":
        pytest.fail(f"expected a GPT-Neo artifact, received {hf_model.config.model_type!r}")
    if len(hf_model.transformer.h) != loaded.config.num_layers:
        pytest.fail("HF and converted checkpoints have different block counts")

    parity_text = "Once upon a time, a little dog found a red ball."
    hf_token_ids = hf_tokenizer.encode(parity_text, add_special_tokens=False)
    runtime_token_ids = runtime_tokenizer.encode(
        parity_text,
        add_special_tokens=False,
    ).ids
    if runtime_token_ids != hf_token_ids:
        pytest.fail("pinned runtime and Hugging Face tokenization differ")
    active_ids = runtime_token_ids[:8]
    token_ids = np.asarray(
        ((
            *active_ids,
            hf_tokenizer.eos_token_id,
            hf_tokenizer.eos_token_id,
        ),),
        dtype=np.int64,
    )
    attention_mask = np.asarray(
        ((True,) * len(active_ids) + (False, False),)
    )
    position_ids = np.arange(token_ids.shape[1], dtype=np.int64)[None, :]
    target_ids = np.roll(token_ids, shift=-1, axis=1)
    loss_mask = attention_mask.astype(np.float32)
    torch_token_ids = torch.from_numpy(token_ids)
    torch_attention_mask = torch.from_numpy(attention_mask.astype(np.int64))
    torch_position_ids = torch.from_numpy(position_ids)

    residual_inputs: dict[int, object] = {}
    post_attention: dict[int, object] = {}
    post_mlp: dict[int, object] = {}
    hook_handles = []

    def block_pre_hook(layer_index: int):
        def capture(_module, inputs):
            residual_inputs[layer_index] = inputs[0].detach()

        return capture

    def attention_hook(layer_index: int):
        def capture(_module, _inputs, outputs):
            attention_output = outputs[0] if isinstance(outputs, tuple) else outputs
            post_attention[layer_index] = (
                residual_inputs[layer_index] + attention_output
            ).detach()

        return capture

    def block_hook(layer_index: int):
        def capture(_module, _inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            post_mlp[layer_index] = hidden.detach()

        return capture

    for layer_index, block in enumerate(hf_model.transformer.h):
        hook_handles.extend(
            (
                block.register_forward_pre_hook(block_pre_hook(layer_index)),
                block.attn.register_forward_hook(attention_hook(layer_index)),
                block.register_forward_hook(block_hook(layer_index)),
            )
        )

    try:
        with torch.inference_mode():
            hf_outputs = hf_model(
                input_ids=torch_token_ids,
                attention_mask=torch_attention_mask,
                position_ids=torch_position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    if set(post_attention) != set(range(loaded.config.num_layers)) or set(post_mlp) != set(
        range(loaded.config.num_layers)
    ):
        pytest.fail("failed to capture both HF residual sites for every block")

    jax_result = apply_gpt_neo(
        loaded.params,
        loaded.config,
        jnp.asarray(token_ids, dtype=jnp.int32),
        jnp.asarray(attention_mask),
        position_ids=jnp.asarray(position_ids, dtype=jnp.int32),
        capture=ordered_capture_spec(loaded.config),
    )
    with torch.inference_mode():
        hf_embedding_output = (
            hf_model.transformer.wte(torch_token_ids)
            + hf_model.transformer.wpe(torch_position_ids)
        )
        hf_token_losses = torch_functional.cross_entropy(
            hf_outputs.logits.reshape(-1, hf_outputs.logits.shape[-1]).float(),
            torch.from_numpy(target_ids).reshape(-1),
            reduction="none",
        ).reshape(target_ids.shape)
        hf_nll = torch.sum(hf_token_losses * torch.from_numpy(loss_mask)) / torch.sum(
            torch.from_numpy(loss_mask)
        )
    jax_nll = mean_token_nll(
        jax_result.logits,
        jnp.asarray(target_ids, dtype=jnp.int32),
        jnp.asarray(loss_mask),
    )
    sequence_length = token_ids.shape[1]
    hf_masks = tuple(
        np.asarray(
            block.attn.attention.bias[0, 0, :sequence_length, :sequence_length]
            .detach()
            .cpu()
        )
        for block in hf_model.transformer.h
    )
    jax_masks = tuple(
        np.asarray(
            attention_pattern(
                sequence_length,
                attention_type,
                loaded.config.local_window_size,
            )
        )
        for attention_type in loaded.config.attention_types
    )
    hf_captures = tuple(
        np.asarray(
            (
                post_attention[layer_index]
                if location == "post_attention"
                else post_mlp[layer_index]
            )
            .detach()
            .cpu()
        )
        for layer_index in range(loaded.config.num_layers)
        for location in ("post_attention", "post_mlp")
    )
    prompt_ids = np.asarray((tuple(runtime_token_ids[:3]),), dtype=np.int64)
    hf_greedy_ids = torch.from_numpy(prompt_ids)
    with torch.inference_mode():
        for _ in range(3):
            greedy_outputs = hf_model(
                input_ids=hf_greedy_ids,
                attention_mask=torch.ones_like(hf_greedy_ids),
                use_cache=False,
                return_dict=True,
            )
            hf_greedy_ids = torch.cat(
                (hf_greedy_ids, torch.argmax(greedy_outputs.logits[:, -1], dim=-1)[:, None]),
                dim=1,
            )
    jax_greedy_ids = greedy_generate(
        loaded.params,
        loaded.config,
        jnp.asarray(prompt_ids, dtype=jnp.int32),
        jnp.ones(prompt_ids.shape, dtype=jnp.bool_),
        max_new_tokens=3,
    )

    expected = ParitySnapshot(
        token_ids=token_ids,
        embedding_output=np.asarray(hf_embedding_output.detach().cpu()),
        position_ids=position_ids,
        attention_masks=hf_masks,
        captured_hidden=hf_captures,
        final_hidden=np.asarray(hf_outputs.hidden_states[-1].detach().cpu()),
        logits=np.asarray(hf_outputs.logits.detach().cpu()),
        normalized_nll=np.asarray(hf_nll.detach().cpu()),
        greedy_token_ids=np.asarray(hf_greedy_ids.detach().cpu()),
    )
    actual = ParitySnapshot(
        token_ids=np.asarray(token_ids, dtype=np.int32),
        embedding_output=np.asarray(
            embed_tokens(
                loaded.params,
                jnp.asarray(token_ids, dtype=jnp.int32),
                jnp.asarray(position_ids, dtype=jnp.int32),
            )
        ),
        position_ids=np.asarray(position_ids, dtype=np.int32),
        attention_masks=jax_masks,
        captured_hidden=tuple(np.asarray(value) for value in jax_result.captured_hidden),
        final_hidden=np.asarray(jax_result.final_hidden),
        logits=np.asarray(jax_result.logits),
        normalized_nll=np.asarray(jax_nll),
        greedy_token_ids=np.asarray(jax_greedy_ids),
    )

    assert_parity(compare_parity_snapshots(expected, actual, loaded.config))


def _prepared_directory(environment_name: str) -> Path:
    configured = os.environ.get(environment_name)
    if not configured:
        pytest.skip(f"set {environment_name} to a locally prepared artifact directory")
    directory = Path(configured)
    if not directory.is_dir():
        pytest.skip(f"local artifact directory does not exist: {directory}")
    return directory
