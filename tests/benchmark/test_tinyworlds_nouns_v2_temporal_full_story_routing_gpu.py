from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedStoryStore
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    CONTEXT_LENGTH,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    build_story_windows,
    score_token_windows_by_candidate,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing import (
    ALLOCATOR_LIMIT_BYTES,
    AUDIT_ABSOLUTE_TOLERANCE,
    assert_parent_unchanged,
    authenticate_full_story_routing_inputs,
    reconstructed_whole_story_scores,
    stable_minimum,
)
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes


pytestmark = pytest.mark.benchmark
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_real_gpu_canonical_score_matches_short_reconstruction_and_allocator() -> None:
    """Compile the real final bank and directly audit one short official story."""
    devices = tuple(jax.local_devices())
    assert len(devices) == 1 and devices[0].platform == "gpu"
    inputs = authenticate_full_story_routing_inputs(REPOSITORY_ROOT)
    source = next(
        row
        for row in inputs.source_rows["blocked_log_t"]
        if int(row["prefix_token_count"]) <= CONTEXT_LENGTH
        and stable_minimum(reconstructed_whole_story_scores(row))[1]
        > 2 * AUDIT_ABSOLUTE_TOLERANCE
    )
    entry = next(
        entry
        for _, entries in inputs.parent.validation_entries
        for entry in entries
        if entry.story_id == source["story_id"]
    )
    windows = build_story_windows(
        IndexedStoryStore(inputs.parent.partition).tokens(entry),
        CONTEXT_LENGTH,
        inputs.parent.partition.pad_token_id,
        first_target_index=1,
    )
    bank = inputs.bank_by_condition["blocked_log_t"]
    totals, _ = score_token_windows_by_candidate(
        windows,
        base_params=inputs.parent.loaded_base.params,
        model_config=inputs.parent.loaded_base.config,
        bank=bank,
        candidate_indices=tuple(range(len(bank.candidate_ids))),
    )
    token_count = int(np.sum(windows.loss_mask))
    direct = tuple(float(np.sum(values)) / token_count for values in totals)
    reconstructed = reconstructed_whole_story_scores(source)
    assert np.max(np.abs(np.asarray(direct) - reconstructed)) <= AUDIT_ABSOLUTE_TOLERANCE
    assert stable_minimum(direct)[0] == stable_minimum(reconstructed)[0]
    assert allocator_peak_bytes() <= ALLOCATOR_LIMIT_BYTES
    assert_parent_unchanged(inputs)
