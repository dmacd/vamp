"""Immutable continual-learning contracts with dependency-lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORT_MODULES: Final[dict[str, str]] = {
    **{
        name: "apm.continual.language_tasks"
        for name in (
            "AddressBook",
            "AddressResult",
            "BaseCheckpointRef",
            "CompetenceBatch",
            "LanguageCurriculum",
            "LanguageEvaluationExample",
            "LanguageTask",
            "NodeId",
            "RouterBatch",
            "TaskId",
            "build_prefix_suffix_batches",
        )
    },
    **{
        name: "apm.continual.knowledge_tasks"
        for name in ("CueRegime", "KnowledgeCandidate", "KnowledgeMode", "KnowledgeQuery")
    },
    **{
        name: "apm.continual.language_evaluation"
        for name in (
            "IN_DOMAIN_TOPIC_SPECIALIZATION",
            "LanguageEvaluationCondition",
            "LanguageEvaluationSuite",
            "LanguageExampleProvenance",
            "LanguageSuiteExample",
        )
    },
    **{
        name: "apm.continual.language_run"
        for name in ("ParentSearchResult", "advance_language_vamp_run", "score_parent_nodes")
    },
    **{
        name: "apm.continual.knowledge_evaluation"
        for name in (
            "KNOWLEDGE_AGGREGATION_AXES",
            "KnowledgeAddressDecision",
            "KnowledgeEvaluationAggregate",
            "KnowledgeMethodEvaluation",
            "KnowledgeQueryEvaluation",
            "aggregate_knowledge_evaluations",
            "evaluate_ebt_knowledge_methods",
            "evaluate_knowledge_method",
        )
    },
    **{
        name: "apm.continual.knowledge_training"
        for name in (
            "KnowledgeCounterfactualTraining",
            "KnowledgeParentContext",
            "KnowledgeParentSearchResult",
            "KnowledgeTransferDiagnostics",
            "KnowledgeValidationSuite",
            "ParentCounterfactualPlan",
            "ParentCounterfactualTarget",
            "ParentTransferTrialDiagnostic",
            "TransferCheckpointDiagnostic",
            "commit_selected_counterfactual_edge",
            "plan_parent_counterfactuals",
            "run_parent_counterfactuals",
            "score_knowledge_parent_nodes",
            "select_knowledge_parent_from_scores",
            "validate_parent_counterfactual_resume",
        )
    },
    **{
        name: "apm.continual.language_adaptation_artifact"
        for name in (
            "LanguageAdaptationArtifact",
            "extract_language_adaptation_artifact",
            "extract_language_vamp_artifact",
            "load_language_adaptation_artifact",
            "save_language_adaptation_artifact",
        )
    },
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> object:
    """Resolve established public exports only when callers request them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
