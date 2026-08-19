from __future__ import annotations

from pathlib import Path

import jax
import pytest

from apm.data.text.tinyworlds_nouns_v2.addressing_study import (
    assert_canonical_hashes_unchanged,
    authenticate_addressing_study_inputs,
    build_study_contracts,
    enforce_nouns_v2_allocator_gate,
    verify_compact_real_parity,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
    build_or_load_addressing_keys,
)


pytestmark = pytest.mark.benchmark
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = (
    REPOSITORY_ROOT
    / "results/language_cl/tinyworlds-nouns-v2/addressing-study"
)


def test_real_gpu_key_derivation_compact_parity_and_allocator_gate() -> None:
    devices = jax.local_devices()
    assert devices and all(device.platform == "gpu" for device in devices)
    assert len(devices) == 1
    inputs = authenticate_addressing_study_inputs(REPOSITORY_ROOT)
    keys = build_or_load_addressing_keys(
        inputs.partition,
        inputs.base_params,
        inputs.adaptation.model_config,
        inputs.adaptation,
        STUDY_ROOT / "keys",
    )
    retrieval_contract, ebt_contract = build_study_contracts(
        inputs,
        keys,
        STUDY_ROOT,
    )
    parity = verify_compact_real_parity(inputs, keys)
    allocator = enforce_nouns_v2_allocator_gate(inputs.preset)

    assert retrieval_contract["bindings"]["vamp_tensor_checksum"] == (
        inputs.adaptation.tensor_checksum
    )
    assert ebt_contract["retrieval_contract_sha256"] == (
        retrieval_contract["contract_sha256"]
    )
    assert parity["actual_rows"] == 8
    assert max(parity["maximum_absolute_differences"].values()) <= parity["tolerance"]
    assert allocator["peak_bytes_in_use"] <= allocator["allocator_limit_bytes"]
    assert_canonical_hashes_unchanged(REPOSITORY_ROOT, inputs.canonical_hashes)
