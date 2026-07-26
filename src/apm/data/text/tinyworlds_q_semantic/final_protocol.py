"""Frozen descriptive analysis choices applied after the one sealed opening."""

from __future__ import annotations

from dataclasses import dataclass

from apm.data.text.tinyworlds_q_semantic.contracts import record_sha256
from apm.data.text.tinyworlds_q_semantic.evaluation import (
    SEMANTIC_QUERY_METHODS,
    SEMANTIC_ROUTED_METHODS,
)
from apm.data.text.tinyworlds_q_semantic.statistics import (
    CANONICAL_BOOTSTRAP_REPLICATES,
)


@dataclass(frozen=True, slots=True)
class SemanticFinalEvaluationProtocol:
    """All report choices that must be fixed before sealed queries are visible."""

    methods: tuple[str, ...]
    acquisition_methods: tuple[str, ...]
    retention_methods: tuple[str, ...]
    specificity_method: str
    effect_metrics: tuple[str, ...]
    bootstrap_replicates: int
    generation_method: str
    generation_max_new_tokens: int
    condition_summary: str
    result_row_accounting: str
    estimated_bytes_per_result_row: int

    def __post_init__(self) -> None:
        if (
            self.methods != SEMANTIC_QUERY_METHODS
            or self.acquisition_methods != self.methods[1:]
            or self.retention_methods != self.methods[1:]
            or self.specificity_method != "independent"
            or self.effect_metrics != ("accuracy", "margin")
            or self.bootstrap_replicates != CANONICAL_BOOTSTRAP_REPLICATES
            or self.generation_method != "independent"
            or type(self.generation_max_new_tokens) is not int
            or self.generation_max_new_tokens <= 0
            or self.condition_summary
            != "stage-method-primary-matching-fact-average"
            or self.result_row_accounting
            != "base-plus-scheduled-methods-plus-independent-specificity-matrix"
            or self.estimated_bytes_per_result_row != 1_024
        ):
            raise ValueError("final semantic evaluation protocol changed")

    def as_record(self) -> dict[str, object]:
        """Return the complete sealed-analysis contract."""
        return {
            "acquisition_methods": list(self.acquisition_methods),
            "bootstrap_replicates": self.bootstrap_replicates,
            "condition_summary": self.condition_summary,
            "effect_metrics": list(self.effect_metrics),
            "generation_max_new_tokens": self.generation_max_new_tokens,
            "generation_method": self.generation_method,
            "estimated_bytes_per_result_row": self.estimated_bytes_per_result_row,
            "methods": list(self.methods),
            "retention_methods": list(self.retention_methods),
            "result_row_accounting": self.result_row_accounting,
            "router_metrics": ["router_accuracy", "routed_regret"],
            "router_methods": list(SEMANTIC_ROUTED_METHODS),
            "scoring": "answer-tokens-only-four-candidate-nll",
            "specificity_method": self.specificity_method,
            "test_openings": 1,
        }

    @property
    def protocol_sha256(self) -> str:
        """Return the content identity of every final-analysis choice."""
        return record_sha256(self.as_record())


REGISTERED_FINAL_EVALUATION_PROTOCOL = SemanticFinalEvaluationProtocol(
    methods=SEMANTIC_QUERY_METHODS,
    acquisition_methods=SEMANTIC_QUERY_METHODS[1:],
    retention_methods=SEMANTIC_QUERY_METHODS[1:],
    specificity_method="independent",
    effect_metrics=("accuracy", "margin"),
    bootstrap_replicates=CANONICAL_BOOTSTRAP_REPLICATES,
    generation_method="independent",
    generation_max_new_tokens=96,
    condition_summary="stage-method-primary-matching-fact-average",
    result_row_accounting=(
        "base-plus-scheduled-methods-plus-independent-specificity-matrix"
    ),
    estimated_bytes_per_result_row=1_024,
)


__all__ = [
    "REGISTERED_FINAL_EVALUATION_PROTOCOL",
    "SemanticFinalEvaluationProtocol",
]
