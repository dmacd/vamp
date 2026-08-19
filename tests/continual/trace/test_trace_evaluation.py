from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from apm.continual.artifacts import record_sha256
from apm.continual.trace.data import TraceExample
from apm.continual.trace.evaluation import (
    Candidate,
    CandidateCaseResult,
    CandidateOutput,
    evaluate_candidate_cache,
    route_outputs,
    task_aware_candidates,
)
from apm.continual.trace.metrics import (
    headline_metrics,
    per_example_task_scores,
    prefix_accuracy,
    score_task,
)
from apm.continual.trace.routing import (
    NodeCentroid,
    PromptQuery,
    length_normalized_prompt_nll,
    select_centroid,
)


class FakeRuntime:
    def prompt_nll(self, candidate: Candidate, query: PromptQuery) -> float:
        del query
        return 2.0 if candidate.candidate_id == "base" else 1.0

    def prompt_embedding(self, query: PromptQuery) -> Tensor:
        del query
        return torch.tensor([1.0, 0.0])

    def generate(self, candidate: Candidate, query: PromptQuery, seed: int) -> str:
        del query, seed
        return "yes" if candidate.candidate_id != "base" else "no"


class BatchedFakeRuntime(FakeRuntime):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate_batch(
        self,
        candidate: Candidate,
        queries: tuple[PromptQuery, ...],
        seeds: tuple[int, ...],
    ) -> tuple[CandidateCaseResult, ...]:
        self.batch_sizes.append(len(queries))
        return tuple(
            CandidateCaseResult(
                self.prompt_nll(candidate, query),
                f"{self.generate(candidate, query, seed)}-{seed}",
                prompt_tokens=len(query.prompt),
                generated_tokens=2,
            )
            for query, seed in zip(queries, seeds)
        )


def _example(index: int, split: str = "validation") -> TraceExample:
    return TraceExample(
        example_id=record_sha256({"example": index, "split": split}),
        task="C-STANCE",
        split=split,
        source_index=index,
        prompt=f"question {index}",
        answer="yes",
        arrival=None if split != "train" else 1,
    )


def test_reference_accuracy_is_reported_on_a_percentage_scale() -> None:
    assert prefix_accuracy(("yes extra", "wrong"), ("yes", "right")) == 50.0


def test_sari_explicitly_trusts_the_pinned_metric_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeMetric:
        def compute(self, **values: object) -> dict[str, float]:
            assert values
            return {"sari": 42.0}

    def load_metric(name: str, **options: object) -> FakeMetric:
        calls.append((name, options))
        return FakeMetric()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_metric=load_metric))

    assert score_task("20Minuten", ("prompt",), ("prediction",), ("target",)) == 42.0
    assert per_example_task_scores(
        "20Minuten",
        ("prompt 1", "prompt 2"),
        ("prediction 1", "prediction 2"),
        ("target 1", "target 2"),
    ) == (42.0, 42.0)
    assert calls == [
        ("sari", {"trust_remote_code": True}),
        ("sari", {"trust_remote_code": True}),
    ]


def test_headline_metrics_keep_signed_and_clipped_bwt_distinct() -> None:
    matrix = {
        (task, stage): float(50 + task + stage)
        for stage in range(1, 9)
        for task in range(1, stage + 1)
    }
    matrix[(1, 1)] = 80.0
    matrix[(1, 8)] = 70.0
    matrix[(2, 2)] = 60.0
    matrix[(2, 8)] = 65.0

    result = headline_metrics(matrix)

    assert result.forgetting == -result.signed_backward_transfer
    assert result.clipped_negative_backward_transfer < result.signed_backward_transfer


def test_prompt_nll_excludes_padding_and_initial_token() -> None:
    logits = torch.zeros(1, 4, 5)
    logits[0, 1, 3] = 8.0
    logits[0, 2, 4] = 8.0
    input_ids = torch.tensor([[0, 2, 3, 4]])
    attention = torch.tensor([[0, 1, 1, 1]])

    score = length_normalized_prompt_nll(logits, input_ids, attention)

    assert score.shape == (1,)
    assert float(score[0]) < 0.01


def test_centroid_router_has_stable_first_tie_breaking() -> None:
    centroids = (
        NodeCentroid("base", torch.tensor([1.0, 0.0]), 100),
        NodeCentroid("node", torch.tensor([1.0, 0.0]), 100),
    )
    assert select_centroid(torch.tensor([1.0, 0.0]), centroids) == "base"


def test_candidate_evaluation_resumes_at_the_exact_case_boundary(tmp_path: Path) -> None:
    candidates = (Candidate("base", None), Candidate("node", Path("node.safetensors")))
    examples = (_example(1), _example(2))
    paused = False

    def pause_once() -> bool:
        nonlocal paused
        if paused:
            return False
        paused = True
        return True

    with pytest.raises(InterruptedError, match="durable case"):
        evaluate_candidate_cache(
            FakeRuntime(),
            candidates,
            examples,
            1,
            tmp_path / "evaluation.jsonl",
            pause_once,
        )
    outputs = evaluate_candidate_cache(
        FakeRuntime(), candidates, examples, 1, tmp_path / "evaluation.jsonl"
    )

    assert len(outputs) == 4
    assert len({(output.example_id, output.candidate_id) for output in outputs}) == 4


def test_candidate_evaluation_uses_durable_fixed_batches(tmp_path: Path) -> None:
    runtime = BatchedFakeRuntime()
    examples = tuple(_example(index) for index in range(5))

    outputs = evaluate_candidate_cache(
        runtime,
        (Candidate("base", None),),
        examples,
        1,
        tmp_path / "batched.jsonl",
        batch_size=2,
    )

    assert runtime.batch_sizes == [2, 2, 1]
    assert len({output.generation_seed for output in outputs}) == 5
    assert all(str(output.generation_seed) in output.prediction for output in outputs)
    assert sum(output.generated_tokens or 0 for output in outputs) == 10


def test_validation_selection_and_all_router_derivations_share_cached_outputs() -> None:
    candidates = (Candidate("base", None), Candidate("node", Path("node.safetensors")))
    examples = (_example(1), _example(2))
    outputs = tuple(
        CandidateOutput(
            example.example_id,
            candidate.candidate_id,
            2.0 if candidate.candidate_id == "base" else 1.0,
            "no" if candidate.candidate_id == "base" else "yes",
            123,
        )
        for example in examples
        for candidate in candidates
    )
    task_routes = task_aware_candidates(candidates, examples, outputs)
    centroids = (
        NodeCentroid("base", torch.tensor([0.0, 1.0]), 100),
        NodeCentroid("node", torch.tensor([1.0, 0.0]), 100),
    )

    routed = route_outputs(
        candidates,
        examples,
        outputs,
        centroids,
        {example.example_id: torch.tensor([1.0, 0.0]) for example in examples},
        task_routes,
    )

    assert len(routed) == 8
    assert {row.router for row in routed} == {
        "prompt_nll",
        "frozen_prompt_centroid",
        "task_aware",
        "answer_oracle",
    }
    assert all(row.candidate_id == "node" for row in routed)
