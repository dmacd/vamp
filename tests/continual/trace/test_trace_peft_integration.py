from __future__ import annotations

import pytest

from apm.continual.trace.modeling import peft_round_trip_self_test


@pytest.mark.integration
def test_zero_lora_and_peft_round_trip_preserve_logits() -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    peft_round_trip_self_test()
