"""Lazy Hugging Face and PEFT construction for the frozen TRACE base model."""

from __future__ import annotations

from dataclasses import dataclass
import os
from tempfile import TemporaryDirectory

import torch

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.trace.collator import Tokenizer
from apm.continual.trace.protocol import (
    MODEL_FILE_IDENTITIES,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SOURCE_ID,
    MODEL_SOURCE_REVISION,
    TrainingConfig,
    stable_seed,
)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """One frozen-base PEFT model and its exact tokenizer revision."""

    model: torch.nn.Module
    tokenizer: Tokenizer
    model_revision: str
    tokenizer_revision: str


def model_source_manifest() -> dict[str, object]:
    """Return the authenticated canonical-model and public-source identity."""
    return {
        "canonical_model_id": MODEL_ID,
        "canonical_revision": MODEL_REVISION,
        "files": [
            {
                "git_blob": git_blob,
                "path": path,
                "sha256": sha256,
                "size": size,
            }
            for path, size, sha256, git_blob in MODEL_FILE_IDENTITIES
        ],
        "format": "trace-model-v2",
        "source_model_id": MODEL_SOURCE_ID,
        "source_revision": MODEL_SOURCE_REVISION,
    }


def model_source_manifest_sha256() -> str:
    """Return the content identity of the authenticated model source."""
    return record_sha256(model_source_manifest())


def verify_model_source_metadata(info: object) -> None:
    """Require the public source revision and Git/Xet objects pinned by TRACE."""
    revision = str(getattr(info, "sha", ""))
    if revision != MODEL_SOURCE_REVISION:
        raise RuntimeError("TRACE model source resolved to an unexpected revision")
    siblings = {
        str(getattr(sibling, "rfilename", "")): sibling
        for sibling in getattr(info, "siblings", ())
    }
    for path, size, sha256, git_blob in MODEL_FILE_IDENTITIES:
        sibling = siblings.get(path)
        if sibling is None:
            raise RuntimeError(f"TRACE model source is missing {path}")
        if int(getattr(sibling, "size", -1)) != size:
            raise RuntimeError(f"TRACE model source size changed for {path}")
        if str(getattr(sibling, "blob_id", "")) != git_blob:
            raise RuntimeError(f"TRACE model source Git blob changed for {path}")
        lfs = getattr(sibling, "lfs", None) or {}
        if path == "model.safetensors" and lfs.get("sha256") != sha256:
            raise RuntimeError("TRACE model tensor Xet/LFS digest changed")


def resolve_model_revision() -> str:
    """Validate the public source metadata and return the canonical Meta revision."""
    try:
        from huggingface_hub import model_info
    except ImportError as error:
        raise RuntimeError("TRACE preparation requires huggingface_hub") from error
    info = model_info(
        MODEL_SOURCE_ID,
        revision=MODEL_SOURCE_REVISION,
        files_metadata=True,
        token=False,
    )
    verify_model_source_metadata(info)
    return MODEL_REVISION


def prepare_model_source() -> dict[str, object]:
    """Download and hash every training-required byte from the public mirror."""
    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as error:
        raise RuntimeError("TRACE preparation requires huggingface_hub") from error
    info = model_info(
        MODEL_SOURCE_ID,
        revision=MODEL_SOURCE_REVISION,
        files_metadata=True,
        token=False,
    )
    verify_model_source_metadata(info)
    snapshot = snapshot_download(
        MODEL_SOURCE_ID,
        revision=MODEL_SOURCE_REVISION,
        allow_patterns=[path for path, *_ in MODEL_FILE_IDENTITIES],
        token=False,
    )
    root = os.fspath(snapshot)
    for path, size, sha256, _git_blob in MODEL_FILE_IDENTITIES:
        source = os.path.join(root, path)
        if os.path.getsize(source) != size or file_sha256(source) != sha256:
            raise RuntimeError(f"TRACE downloaded model source changed for {path}")
    return model_source_manifest()


def load_fresh_lora_bundle(
    revision: str,
    device: str | torch.device,
    job_identity: str,
    config: TrainingConfig = TrainingConfig(),
) -> ModelBundle:
    """Load the frozen BF16 base and attach one fresh zero-effect LoRA."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("TRACE training requires transformers and peft") from error
    if revision != MODEL_REVISION or not job_identity.strip():
        raise ValueError("canonical model revision and job identity are required")
    seed = stable_seed("model-initialization", job_identity)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE_ID,
        revision=MODEL_SOURCE_REVISION,
        token=False,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_device = torch.device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_SOURCE_ID,
        revision=MODEL_SOURCE_REVISION,
        token=False,
        torch_dtype=dtype,
    )
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            bias="none",
            target_modules=list(config.target_modules),
        ),
    ).to(target_device)
    trainable_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("TRACE model has trainable parameters outside LoRA")
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        tokenizer_revision=revision,
    )


def peft_round_trip_self_test() -> None:
    """Verify zero-effect LoRA and exact PEFT save/load on a tiny Llama model."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as error:
        raise RuntimeError("TRACE self-test requires transformers and peft") from error
    from apm.continual.trace.adapter_io import load_adapter_state, save_adapter

    configuration = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        vocab_size=64,
    )
    torch.manual_seed(7)
    base = LlamaForCausalLM(configuration).eval()
    base_state = {
        name: value.detach().clone() for name, value in base.state_dict().items()
    }
    input_ids = torch.tensor([[1, 4, 9, 2]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        base_logits = base(input_ids=input_ids, attention_mask=attention_mask).logits
    adapter_configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    adapted = get_peft_model(base, adapter_configuration).eval()
    with torch.no_grad():
        zero_logits = adapted(input_ids=input_ids, attention_mask=attention_mask).logits
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.02)
        expected = adapted(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(zero_logits, base_logits, rtol=0.0, atol=0.0)
    with TemporaryDirectory(prefix="trace-peft-self-test-") as temporary:
        adapter_path = save_adapter(adapted, temporary)
        reloaded_base = LlamaForCausalLM(configuration)
        reloaded_base.load_state_dict(base_state)
        reloaded = get_peft_model(reloaded_base, adapter_configuration).eval()
        load_adapter_state(reloaded, adapter_path)
        with torch.no_grad():
            actual = reloaded(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


__all__ = [
    "ModelBundle",
    "load_fresh_lora_bundle",
    "model_source_manifest",
    "model_source_manifest_sha256",
    "peft_round_trip_self_test",
    "prepare_model_source",
    "resolve_model_revision",
    "verify_model_source_metadata",
]
