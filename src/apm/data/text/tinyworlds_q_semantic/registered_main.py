"""Exact real-artifact registry for the authorized five-world main run."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    RESULT_ROOT,
)
from apm.data.text.tinyworlds_q_semantic.main_freeze import (
    MainExperimentFreeze,
    publish_main_experiment_freeze,
)
from apm.data.text.tinyworlds_q_semantic.manifests import (
    MAIN_CONCEPT_IDS,
    PILOT_CONCEPT_IDS,
)
from apm.data.text.tinyworlds_q_semantic.pilot import (
    SemanticPilotFailure,
    SemanticPilotResult,
    load_semantic_pilot_failure,
    load_semantic_pilot_result,
)
from apm.data.text.tinyworlds_q_semantic.pilot_authorization import (
    SemanticPilotProtocolAmendment,
    load_semantic_pilot_protocol_amendment,
)


PILOT_FAILURE_SHA256 = (
    "aad4811425c10b0faf5f6f452067e35a58d6cee397970711951e50bfad2247f5"
)
PILOT_AMENDMENT_SHA256 = (
    "2855b647928700a119ea6e95365379719ad733d45c6ede20cafcd1593a64458c"
)
PILOT_RESULT_SHA256 = (
    "55c97f2a649ea434f79e729b2eaff01753a254ce0a5c26e53a1095d4df0364c7"
)
AUTHORIZED_AT = "2026-07-25T20:05:38Z"


def load_registered_main_authority() -> tuple[
    SemanticPilotFailure,
    SemanticPilotProtocolAmendment,
    SemanticPilotResult,
    QueryExperimentPreset,
    QueryExperimentPreset,
    MainExperimentFreeze,
]:
    """Load the exact failure, amendment, pilot, presets, and main freeze."""
    pilot_preset = QueryExperimentPreset(PILOT_CONCEPT_IDS)
    failure = load_semantic_pilot_failure(
        RESULT_ROOT / "pilot-failure" / PILOT_FAILURE_SHA256
    )
    amendment = load_semantic_pilot_protocol_amendment(
        RESULT_ROOT / "pilot-amendment" / PILOT_AMENDMENT_SHA256,
        failure,
        pilot_preset,
    )
    pilot = load_semantic_pilot_result(
        RESULT_ROOT / "pilot" / PILOT_RESULT_SHA256
    )
    main_preset = QueryExperimentPreset(
        MAIN_CONCEPT_IDS,
        adapter_updates=pilot.selected_updates,
    )
    frozen = publish_main_experiment_freeze(
        RESULT_ROOT / "main-freeze",
        pilot,
        amendment,
        failure,
        pilot_preset,
        main_preset,
        authorized_by="interactive-user",
        authorized_at=AUTHORIZED_AT,
        authorization="Okay sounds reasonable, lets move forward",
    )
    return failure, amendment, pilot, pilot_preset, main_preset, frozen


__all__ = [
    "AUTHORIZED_AT",
    "PILOT_AMENDMENT_SHA256",
    "PILOT_FAILURE_SHA256",
    "PILOT_RESULT_SHA256",
    "load_registered_main_authority",
]
