"""Secondary exact-trigger generation with matching independent adapters."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_baseline_training import pack_root_adapter
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.final_protocol import (
    REGISTERED_FINAL_EVALUATION_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.statistics import (
    GenerationInspection,
    generation_prompts,
    inspect_generation,
)
from apm.lm.checkpoint import LoadedGptNeoCheckpoint
from apm.lm.generation import greedy_generate
from apm.lm.lora import LoraEdge
from apm.lm.text import TextTokenizer


def generate_semantic_fact_inspections(
    catalog: SemanticQueryCatalog,
    base: LoadedGptNeoCheckpoint,
    adaptation: LanguageAdaptationArtifact,
    preset: QueryExperimentPreset,
    tokenizer: TextTokenizer,
) -> tuple[GenerationInspection, ...]:
    """Generate one reviewed prompt per world and score exact trigger recall."""
    protocol = REGISTERED_FINAL_EVALUATION_PROTOCOL
    if (
        catalog.concept_ids[: preset.active_world_count] != preset.concept_ids
        or tuple(str(task_id) for task_id in adaptation.task_order)
        != preset.concept_ids
        or tuple(str(adapter.task_id) for adapter in adaptation.independent_adapters)
        != preset.concept_ids
        or adaptation.base_checkpoint.manifest_sha256
        != base.reference.manifest_sha256
        or adaptation.base_checkpoint.parameter_checksum
        != base.reference.parameter_checksum
        or adaptation.model_config != base.config
        or adaptation.lora_config != preset.lora_config
        or protocol.generation_method != "independent"
    ):
        raise ValueError("semantic generation sources changed frozen identities")
    prompt_by_concept = dict(
        generation_prompts(catalog.concepts[: preset.active_world_count])
    )
    outputs = tuple(
        (
            concept_id,
            prompt_by_concept[concept_id],
            _generate_continuation(
                prompt_by_concept[concept_id],
                adapter.adapter,
                base,
                preset,
                tokenizer,
                protocol.generation_max_new_tokens,
            ),
        )
        for concept_id, adapter in zip(
            preset.concept_ids,
            adaptation.independent_adapters,
        )
    )
    return inspect_generation(
        catalog,
        outputs,
        concept_ids=preset.concept_ids,
    )


def _generate_continuation(
    prompt: str,
    adapter: LoraEdge,
    base: LoadedGptNeoCheckpoint,
    preset: QueryExperimentPreset,
    tokenizer: TextTokenizer,
    maximum_new_tokens: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    if not prompt_ids or len(prompt_ids) + maximum_new_tokens > preset.context_length:
        raise ValueError("semantic generation prompt exceeds the frozen context")
    _graph, packed = pack_root_adapter(
        adapter,
        base.config,
        preset.lora_config,
    )
    generated = greedy_generate(
        base.params,
        base.config,
        jnp.asarray((prompt_ids,), dtype=jnp.int32),
        jnp.ones((1, len(prompt_ids)), dtype=jnp.bool_),
        maximum_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        lora_memory=packed,
        lora_config=preset.lora_config,
        node_index=1,
    )
    continuation_ids = tuple(
        int(value)
        for value in np.asarray(generated)[0, len(prompt_ids) :]
    )
    return (
        tokenizer.decode(continuation_ids)
        or tokenizer.decode(continuation_ids, skip_special_tokens=False)
        or "<EOS>"
    )


__all__ = ["generate_semantic_fact_inspections"]
