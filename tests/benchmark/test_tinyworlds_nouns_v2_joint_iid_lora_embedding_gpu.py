from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedStoryStore
from apm.data.text.tinyworlds_nouns_v2.contracts import TASK_IDS
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ALLOCATOR_LIMIT_BYTES,
    CONTEXT_LENGTH,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    build_adapter_bank,
    build_midpoint_case,
    evaluate_case_batch,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    StoryEpochBatches,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_lora_embedding import (
    LoraEmbeddingJob,
    assert_lora_embedding_inputs_unchanged,
    authenticate_lora_embedding_inputs,
    train_or_load_lora_embedding,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
    rank_lora_config,
)
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes


pytestmark = pytest.mark.benchmark
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_real_gpu_rank32_joint_training_and_suffix_scoring_stay_bounded(
    tmp_path: Path,
) -> None:
    """Compile the largest joint trainable tree and score one exact suffix."""
    devices = tuple(jax.local_devices())
    assert len(devices) == 1 and devices[0].platform == "gpu"
    inputs = authenticate_lora_embedding_inputs(REPOSITORY_ROOT)
    story_ids = inputs.rank_sweep.all_story_ids[:32]
    entries = tuple(inputs.parent.train_entry_lookup[story_id] for story_id in story_ids)
    namespace = inputs.rank_sweep.canonical_rank8_job.identity_sha256
    job = LoraEmbeddingJob(
        inputs.contract_sha256,
        32,
        story_ids,
        (inputs.rank_sweep.source_shard_ids[0],),
        namespace,
        namespace,
    )
    batches = StoryEpochBatches(
        inputs.parent.partition,
        entries,
        context_length=CONTEXT_LENGTH,
        batch_size=32,
        namespace=namespace,
    )
    artifact = train_or_load_lora_embedding(
        job,
        batches,
        inputs.parent.loaded_base.params,
        inputs.parent.loaded_base.config,
        tmp_path / "output",
        tmp_path / "work",
    )
    assert not np.array_equal(
        np.asarray(artifact.trainable.token_embedding),
        np.asarray(inputs.parent.loaded_base.params.token_embedding),
    )
    task_id, validation_entries = inputs.parent.validation_entries[0]
    case = build_midpoint_case(
        inputs.parent.partition,
        IndexedStoryStore(inputs.parent.partition),
        task_id,
        validation_entries[0],
        context_length=CONTEXT_LENGTH,
        maximum_position_embeddings=inputs.parent.loaded_base.config.max_position_embeddings,
    )
    bank = build_adapter_bank(
        (
            AdapterCandidate(
                "rank-32-embedding-smoke",
                artifact.adapter_sha256,
                artifact.trainable.adapter,
                tuple((noun, 8) for noun in TASK_IDS),
            ),
        ),
        inputs.parent.loaded_base.config,
        rank_lora_config(32),
    )
    result = evaluate_case_batch(
        (case,),
        contract_sha256=inputs.contract_sha256,
        evaluation_id="embedding-lora-real-gpu-smoke",
        dataset="timing",
        method="joint_iid_lora_embedding_rank_32",
        order=None,
        stage=192,
        routing="forced_adapter",
        base_params=inputs.parent.loaded_base.params._replace(
            token_embedding=artifact.trainable.token_embedding
        ),
        model_config=inputs.parent.loaded_base.config,
        bank=bank,
        evaluation_batch_size=32,
    )
    assert len(result) == 1 and np.isfinite(result[0].suffix_mean_nll)
    assert (
        result[0].candidate_ids[result[0].selected_index]
        == "rank-32-embedding-smoke"
    )
    assert allocator_peak_bytes() <= ALLOCATOR_LIMIT_BYTES
    assert_lora_embedding_inputs_unchanged(inputs)
