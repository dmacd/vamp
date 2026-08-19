"""TRACE reference task metrics and continual-learning aggregates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from apm.continual.trace.protocol import TASK_BY_NAME, TASK_NAMES


@dataclass(frozen=True, slots=True)
class HeadlineMetrics:
    """Final OP and three explicitly labeled backward-transfer quantities."""

    overall_performance: float
    forgetting: float
    signed_backward_transfer: float
    clipped_negative_backward_transfer: float

    def as_record(self) -> dict[str, float]:
        """Return report-ready metric labels without conflating BWT definitions."""
        return {
            "bwt_clipped_negative_only": self.clipped_negative_backward_transfer,
            "bwt_signed": self.signed_backward_transfer,
            "forgetting": self.forgetting,
            "op": self.overall_performance,
        }


def prefix_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Reproduce TreeLoRA's stripped prediction-prefix exact-match percentage."""
    return sum(_prefix_scores(predictions, targets)) / len(predictions)


def rouge_l(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Return TreeLoRA's Rouge package ROUGE-L F1 as a percentage."""
    return sum(_rouge_l_scores(predictions, targets)) / len(predictions)


def _rouge_l_scores(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> tuple[float, ...]:
    """Return per-example reference ROUGE-L percentages with one scorer instance."""
    _require_pairs(predictions, targets)
    try:
        from rouge import Rouge
    except ImportError as error:
        raise RuntimeError("TRACE ROUGE-L requires rouge==1.0.1") from error
    scorer = Rouge(metrics=["rouge-l"])
    # The reference calls score_rouge(target, prediction), including its newline quirk.
    return tuple(
        100.0
        * float(
            scorer.get_scores(
                target,
                prediction + "\n" if "\n" not in prediction else prediction,
                avg=True,
            )["rouge-l"]["f"]
        )
        if prediction and target
        else 0.0
        for prediction, target in zip(predictions, targets)
    )


def py150_similarity(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Return fuzzywuzzy ratio after TRACE code-literal restoration."""
    return sum(_py150_scores(predictions, targets)) / len(predictions)


def sari(
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    """Return the reference Hugging Face SARI score as one numeric percentage."""
    _require_pairs(predictions, targets)
    if len(prompts) != len(predictions):
        raise ValueError("SARI prompts and outputs differ in length")
    try:
        from datasets import load_metric
    except ImportError as error:
        raise RuntimeError("TRACE SARI requires the pinned datasets metric stack") from error
    result = load_metric("sari", trust_remote_code=True).compute(
        sources=list(prompts),
        predictions=list(predictions),
        references=[[target] for target in targets],
    )
    if not isinstance(result, Mapping) or "sari" not in result:
        raise RuntimeError("SARI metric returned an unexpected payload")
    return float(result["sari"])


def score_task(
    task: str,
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
) -> float:
    """Score one complete task using the registered TRACE primary metric."""
    if task not in TASK_BY_NAME:
        raise ValueError(f"unknown TRACE task: {task}")
    metric = TASK_BY_NAME[task].metric
    if task == "ScienceQA":
        return prefix_accuracy(
            tuple(_science_qa_parts(value)[0] for value in predictions),
            tuple(_science_qa_parts(value)[0] for value in targets),
        )
    if metric == "accuracy":
        return prefix_accuracy(predictions, targets)
    if metric == "rouge_l":
        return rouge_l(predictions, targets)
    if metric == "similarity":
        return py150_similarity(predictions, targets)
    if metric == "sari":
        return sari(prompts, predictions, targets)
    raise AssertionError("unreachable TRACE metric")


def per_example_task_scores(
    task: str,
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
) -> tuple[float, ...]:
    """Return candidate-comparable per-example scores for the answer oracle."""
    _require_pairs(predictions, targets)
    if task not in TASK_BY_NAME:
        raise ValueError(f"unknown TRACE task: {task}")
    metric = TASK_BY_NAME[task].metric
    if task == "ScienceQA":
        return _prefix_scores(
            tuple(_science_qa_parts(value)[0] for value in predictions),
            tuple(_science_qa_parts(value)[0] for value in targets),
        )
    if metric == "accuracy":
        return _prefix_scores(predictions, targets)
    if metric == "rouge_l":
        return _rouge_l_scores(predictions, targets)
    if metric == "similarity":
        return _py150_scores(predictions, targets)
    if metric == "sari":
        return _per_example_sari(prompts, predictions, targets)
    raise AssertionError("unreachable TRACE metric")


def headline_metrics(score_matrix: Mapping[tuple[int, int], float]) -> HeadlineMetrics:
    """Compute OP, forgetting, signed BWT, and clipped negative-only BWT."""
    required = {
        (task_index, stage_index)
        for stage_index in range(1, 9)
        for task_index in range(1, stage_index + 1)
    }
    if required - set(score_matrix):
        raise ValueError("TRACE triangular matrix is incomplete")
    final_scores = tuple(score_matrix[(task_index, 8)] for task_index in range(1, 9))
    diagonal = tuple(score_matrix[(task_index, task_index)] for task_index in range(1, 9))
    transfers = tuple(final - initial for final, initial in zip(final_scores, diagonal))
    return HeadlineMetrics(
        overall_performance=sum(final_scores) / 8,
        forgetting=sum(initial - final for initial, final in zip(diagonal, final_scores)) / 8,
        signed_backward_transfer=sum(transfers) / 8,
        clipped_negative_backward_transfer=sum(min(transfer, 0.0) for transfer in transfers) / 8,
    )


def _restore_code_literals(code: str) -> str:
    value = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    for kind, literal in re.findall(r"<(STR|NUM|CHAR)_LIT:(.*?)>", value, re.S):
        value = value.replace(f"<{kind}_LIT:{literal}>", literal)
    return value


def _prefix_scores(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> tuple[float, ...]:
    _require_pairs(predictions, targets)
    return tuple(
        100.0
        if prediction_prefix and target and prediction_prefix == target
        else 0.0
        for prediction, target in zip(predictions, targets)
        for stripped in (str(prediction).strip(),)
        for prediction_prefix in (stripped[: min(len(target), len(stripped))],)
    )


def _py150_scores(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> tuple[float, ...]:
    _require_pairs(predictions, targets)
    try:
        from fuzzywuzzy import fuzz
    except ImportError as error:
        raise RuntimeError("TRACE Py150 similarity requires fuzzywuzzy==0.18.0") from error
    return tuple(
        float(fuzz.ratio(_restore_code_literals(prediction), _restore_code_literals(target)))
        if prediction and target
        else 0.0
        for prediction, target in zip(predictions, targets)
    )


def _per_example_sari(
    prompts: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
) -> tuple[float, ...]:
    if len(prompts) != len(predictions):
        raise ValueError("SARI prompts and outputs differ in length")
    try:
        from datasets import load_metric
    except ImportError as error:
        raise RuntimeError("TRACE SARI requires the pinned datasets metric stack") from error
    metric = load_metric("sari", trust_remote_code=True)
    return tuple(
        float(
            metric.compute(
                sources=[prompt],
                predictions=[prediction],
                references=[[target]],
            )["sari"]
        )
        for prompt, prediction, target in zip(prompts, predictions, targets)
    )


def _science_qa_parts(value: str) -> tuple[str, str]:
    stripped = str(value).strip()
    if not stripped:
        return "N/A", "N/A"
    return stripped[0], stripped[2:] if len(stripped) >= 2 else "N/A"


def _require_pairs(predictions: Sequence[str], targets: Sequence[str]) -> None:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("TRACE metrics require nonempty aligned predictions and targets")


__all__ = [
    "HeadlineMetrics",
    "headline_metrics",
    "per_example_task_scores",
    "prefix_accuracy",
    "py150_similarity",
    "rouge_l",
    "sari",
    "score_task",
]
