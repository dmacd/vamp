from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedStoryStore
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
    assert_canonical_artifacts_unchanged,
    authenticate_temporal_study_inputs,
)
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
    TrainingInterrupted,
    TrainingJob,
    train_or_load_lora,
)
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes


pytestmark = pytest.mark.benchmark
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_real_data_gpu_training_resume_evaluation_and_allocator_gate(
    tmp_path: Path,
) -> None:
    """Compile real kernels and prove exact resume on a bounded source subset."""
    devices = tuple(jax.local_devices())
    assert len(devices) == 1 and devices[0].platform == "gpu"
    inputs = authenticate_temporal_study_inputs(REPOSITORY_ROOT)
    first_shard = inputs.shards[0]
    entry_lookup = inputs.train_entry_lookup
    story_ids = first_shard.story_ids[:32]
    entries = tuple(entry_lookup[story_id] for story_id in story_ids)
    job = TrainingJob(
        inputs.contract_sha256,
        "real-gpu-resume-smoke",
        "level_zero",
        story_ids,
        (first_shard.shard_id,),
    )
    batches = StoryEpochBatches(
        inputs.partition,
        entries,
        context_length=CONTEXT_LENGTH,
        batch_size=32,
        namespace=job.identity_sha256,
    )

    direct = train_or_load_lora(
        job,
        batches,
        inputs.loaded_base.params,
        inputs.loaded_base.config,
        tmp_path / "direct-output",
        tmp_path / "direct-work",
    )
    with pytest.raises(TrainingInterrupted):
        train_or_load_lora(
            job,
            batches,
            inputs.loaded_base.params,
            inputs.loaded_base.config,
            tmp_path / "resumed-output",
            tmp_path / "resumed-work",
            stop_after_update=1,
        )
    resumed = train_or_load_lora(
        job,
        batches,
        inputs.loaded_base.params,
        inputs.loaded_base.config,
        tmp_path / "resumed-output",
        tmp_path / "resumed-work",
    )
    assert direct.optimizer_updates == resumed.optimizer_updates == len(batches)
    assert direct.adapter_sha256 == resumed.adapter_sha256

    task_id, validation_entries = inputs.validation_entries[0]
    case = build_midpoint_case(
        inputs.partition,
        IndexedStoryStore(inputs.partition),
        task_id,
        validation_entries[0],
        context_length=CONTEXT_LENGTH,
        maximum_position_embeddings=inputs.loaded_base.config.max_position_embeddings,
    )
    bank = build_adapter_bank(
        (
            AdapterCandidate(
                "smoke-adapter",
                resumed.adapter_sha256,
                resumed.adapter,
                ((task_id, 1),),
                level=0,
                start_arrival=1,
                end_arrival=1,
            ),
        ),
        inputs.loaded_base.config,
    )
    result = evaluate_case_batch(
        (case,),
        contract_sha256=inputs.contract_sha256,
        evaluation_id="real-gpu-smoke",
        dataset="timing",
        method="log_t",
        order="blocked",
        stage=1,
        routing="exhaustive",
        base_params=inputs.loaded_base.params,
        model_config=inputs.loaded_base.config,
        bank=bank,
        physical_router_rows=8,
    )
    assert len(result) == 1
    assert np.isfinite(result[0].suffix_mean_nll)
    assert result[0].candidate_ids == ("base", "smoke-adapter")
    assert allocator_peak_bytes() <= ALLOCATOR_LIMIT_BYTES
    assert_canonical_artifacts_unchanged(inputs)
