"""Resumable candidate generation and router-derived TRACE evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from apm.continual.artifacts import ChainedJsonlLedger, record_sha256
from apm.continual.trace.collator import Tokenizer, TraceDataCollator
from apm.continual.trace.data import TraceExample
from apm.continual.trace.metrics import per_example_task_scores, score_task
from apm.continual.trace.protocol import RouterName, TrainingConfig, stable_seed
from apm.continual.trace.routing import (
    BASE_CANDIDATE,
    NodeCentroid,
    PromptQuery,
    length_normalized_prompt_nll,
    mean_pool_hidden,
    select_best_candidate,
    select_centroid,
    select_lowest_nll,
)


EVALUATION_LEDGER_FORMAT = "trace-candidate-evaluation-v1"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One frozen-base or active-adapter inference candidate."""

    candidate_id: str
    adapter_path: Path | None
    rank: int = 8
    alpha: int = 32

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or (self.candidate_id == BASE_CANDIDATE) != (self.adapter_path is None)
            or self.rank <= 0
            or self.alpha <= 0
        ):
            raise ValueError("only the base candidate may omit an adapter path")


@dataclass(frozen=True, slots=True)
class CandidateOutput:
    """One candidate's prompt-only score and generated continuation."""

    example_id: str
    candidate_id: str
    prompt_nll: float
    prediction: str
    generation_seed: int
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    case_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            (self.prompt_tokens is not None and self.prompt_tokens <= 0)
            or (self.generated_tokens is not None and self.generated_tokens < 0)
            or self.case_wall_seconds < 0.0
        ):
            raise ValueError("candidate token and timing accounting is invalid")

    def as_record(self, stage: int, split: str, task: str) -> dict[str, object]:
        """Return a resumable cache row without copying target answers."""
        return {
            "candidate_id": self.candidate_id,
            "case_wall_seconds": self.case_wall_seconds,
            "example_id": self.example_id,
            "generated_tokens": self.generated_tokens,
            "generation_seed": self.generation_seed,
            "prediction": self.prediction,
            "prompt_tokens": self.prompt_tokens,
            "prompt_nll": self.prompt_nll,
            "split": split,
            "stage": stage,
            "task": task,
        }


@dataclass(frozen=True, slots=True)
class RoutedPrediction:
    """The selected candidate and prediction under one router."""

    example_id: str
    router: RouterName
    candidate_id: str
    prediction: str


@dataclass(frozen=True, slots=True)
class CandidateCaseResult:
    """One runtime result plus exact prompt and generation token counts when known."""

    prompt_nll: float
    prediction: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None


class CandidateRuntime(Protocol):
    """Inference operations needed by the resumable evaluation loop."""

    def prompt_nll(self, candidate: Candidate, query: PromptQuery) -> float: ...

    def prompt_embedding(self, query: PromptQuery) -> Tensor: ...

    def generate(self, candidate: Candidate, query: PromptQuery, seed: int) -> str: ...


@runtime_checkable
class BatchedCandidateRuntime(Protocol):
    """Optional bounded-batch interface used by the production GPU runtime."""

    def evaluate_batch(
        self,
        candidate: Candidate,
        queries: Sequence[PromptQuery],
        seeds: Sequence[int],
    ) -> tuple[CandidateCaseResult, ...]: ...


class PeftCandidateRuntime:
    """Single-model PEFT runtime that lazily registers immutable candidates."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        config: TrainingConfig = TrainingConfig(),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.collator = TraceDataCollator(tokenizer, config)
        self.device = next(model.parameters()).device
        self._adapter_names: dict[str, str] = {}

    def prompt_nll(self, candidate: Candidate, query: PromptQuery) -> float:
        """Return answer-isolated length-normalized prompt NLL for one candidate."""
        batch = {
            key: value.to(self.device)
            for key, value in self.collator.inference_batch((query.prompt,)).items()
        }
        self.model.eval()
        with self._activated(candidate), torch.no_grad():
            output = self.model(**batch)
        scores = length_normalized_prompt_nll(
            getattr(output, "logits"),
            batch["input_ids"],
            batch["attention_mask"],
        )
        return float(scores[0].to(torch.float32).cpu().item())

    def prompt_embedding(self, query: PromptQuery) -> Tensor:
        """Return one normalized final-layer frozen-base prompt embedding."""
        batch = {
            key: value.to(self.device)
            for key, value in self.collator.inference_batch((query.prompt,)).items()
        }
        self.model.eval()
        with self.model.disable_adapter(), torch.no_grad():
            output = self.model(**batch, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, tuple) or not hidden_states:
            raise RuntimeError("model did not expose final hidden states")
        return mean_pool_hidden(hidden_states[-1], batch["attention_mask"])[0].cpu()

    def answer_nll(
        self,
        candidate: Candidate,
        examples: Sequence[TraceExample],
        batch_size: int = 4,
    ) -> float:
        """Return diagnostic answer-only validation NLL without generation."""
        if not examples or batch_size <= 0:
            raise ValueError("answer NLL requires validation examples")
        losses = []
        self.model.eval()
        with self._activated(candidate), torch.no_grad():
            for start in range(0, len(examples), batch_size):
                selected = examples[start : start + batch_size]
                batch = {
                    key: value.to(self.device)
                    for key, value in self.collator.training_batch(
                        tuple(
                            {"answer": example.answer, "prompt": example.prompt}
                            for example in selected
                        )
                    ).items()
                }
                output = self.model(**batch)
                logits = getattr(output, "logits", None)
                if not isinstance(logits, Tensor) or logits.ndim != 3:
                    raise TypeError("causal LM did not return validation logits")
                targets = batch["labels"][:, 1:]
                observed = targets != -100
                counts = observed.sum(dim=1)
                if torch.any(counts == 0):
                    raise ValueError("answer NLL found an example without target tokens")
                token_losses = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).to(torch.float32),
                    targets.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape(targets.shape)
                losses.extend(
                    (
                        (token_losses * observed).sum(dim=1) / counts
                    ).to(device="cpu", dtype=torch.float32).tolist()
                )
        return sum(losses) / len(losses)

    def generate(self, candidate: Candidate, query: PromptQuery, seed: int) -> str:
        """Generate the reference stochastic continuation under a stable per-case seed."""
        batch = {
            key: value.to(self.device)
            for key, value in self.collator.inference_batch((query.prompt,)).items()
        }
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt_length = batch["input_ids"].shape[1]
        pad_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        with self._activated(candidate), torch.no_grad():
            generated = self.model.generate(
                **batch,
                max_new_tokens=512,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                temperature=0.1,
                do_sample=True,
                num_return_sequences=1,
                use_cache=True,
            )
        decoded = self.tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0])

    def evaluate_batch(
        self,
        candidate: Candidate,
        queries: Sequence[PromptQuery],
        seeds: Sequence[int],
    ) -> tuple[CandidateCaseResult, ...]:
        """Score and sample a bounded batch with an independent RNG per example."""
        if not queries or len(queries) != len(seeds):
            raise ValueError("candidate batches require aligned queries and seeds")
        batch = {
            key: value.to(self.device)
            for key, value in self.collator.inference_batch(
                tuple(query.prompt for query in queries)
            ).items()
        }
        generators = tuple(
            torch.Generator(device=self.device).manual_seed(seed) for seed in seeds
        )
        self.model.eval()
        with self._activated(candidate), torch.no_grad():
            output = self.model(**batch, use_cache=True, return_dict=True)
            logits = getattr(output, "logits")
            nll = length_normalized_prompt_nll(
                logits,
                batch["input_ids"],
                batch["attention_mask"],
            )
            generated = self._sample_from_prefill(
                logits[:, -1, :],
                getattr(output, "past_key_values"),
                batch["attention_mask"],
                generators,
            )
        decoded = self._decode_generated(generated)
        prompt_tokens = batch["attention_mask"].sum(dim=1).tolist()
        return tuple(
            CandidateCaseResult(
                float(score.to(torch.float32).cpu().item()),
                prediction,
                int(prompt_count),
                len(generated_ids),
            )
            for score, prediction, prompt_count, generated_ids in zip(
                nll,
                decoded,
                prompt_tokens,
                generated,
            )
        )

    def _sample_from_prefill(
        self,
        next_logits: Tensor,
        past_key_values: object,
        attention_mask: Tensor,
        generators: Sequence[torch.Generator],
    ) -> tuple[tuple[int, ...], ...]:
        """Autoregress with a shared KV-cache and stable row-local generators."""
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.unk_token_id
        if pad is None:
            pad = eos
        rows: list[list[int]] = [[] for _ in generators]
        finished = torch.zeros(len(generators), dtype=torch.bool, device=self.device)
        logits = next_logits
        cache = past_key_values
        mask = attention_mask
        for _ in range(512):
            probabilities = torch.softmax(logits.to(torch.float32) / 0.1, dim=-1)
            sampled = torch.stack(
                tuple(
                    torch.multinomial(probabilities[index], 1, generator=generator)[0]
                    for index, generator in enumerate(generators)
                )
            )
            sampled = torch.where(finished, torch.full_like(sampled, pad), sampled)
            active_before = ~finished
            for index, token in enumerate(sampled.tolist()):
                if bool(active_before[index]):
                    rows[index].append(int(token))
            finished = finished | (sampled == eos)
            if bool(torch.all(finished)):
                break
            mask = torch.cat((mask, active_before.to(mask.dtype).unsqueeze(1)), dim=1)
            output = self.model(
                input_ids=sampled.unsqueeze(1),
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            logits = getattr(output, "logits")[:, -1, :]
            cache = getattr(output, "past_key_values")
        return tuple(tuple(row) for row in rows)

    def _decode_generated(
        self,
        generated: Sequence[Sequence[int]],
    ) -> tuple[str, ...]:
        pad = self.tokenizer.unk_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        width = max((len(row) for row in generated), default=0)
        tensor = torch.tensor(
            tuple(tuple(row) + (pad,) * (width - len(row)) for row in generated),
            dtype=torch.long,
        )
        return tuple(
            str(value)
            for value in self.tokenizer.batch_decode(
                tensor,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    def _activated(self, candidate: Candidate) -> AbstractContextManager[object]:
        if candidate.adapter_path is None:
            return self.model.disable_adapter()
        adapter_name = self._ensure_adapter(candidate)
        self.model.set_adapter(adapter_name)
        return nullcontext()

    def _ensure_adapter(self, candidate: Candidate) -> str:
        if candidate.candidate_id in self._adapter_names:
            return self._adapter_names[candidate.candidate_id]
        try:
            from peft import LoraConfig, TaskType, set_peft_model_state_dict
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("TRACE evaluation requires peft and safetensors") from error
        if candidate.adapter_path is None:
            raise ValueError("base candidate has no adapter")
        adapter_name = f"trace_{record_sha256(candidate.candidate_id)[:16]}"
        self.model.add_adapter(
            adapter_name,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=candidate.rank,
                lora_alpha=candidate.alpha,
                lora_dropout=self.config.dropout,
                bias="none",
                target_modules=list(self.config.target_modules),
            ),
        )
        result = set_peft_model_state_dict(
            self.model,
            load_file(candidate.adapter_path, device="cpu"),
            adapter_name=adapter_name,
        )
        unexpected = tuple(getattr(result, "unexpected_keys", ()))
        if unexpected:
            raise ValueError(f"candidate adapter has unexpected keys: {unexpected}")
        self._adapter_names[candidate.candidate_id] = adapter_name
        return adapter_name


def evaluate_candidate_cache(
    runtime: CandidateRuntime,
    candidates: Sequence[Candidate],
    examples: Sequence[TraceExample],
    stage: int,
    ledger_path: str | Path,
    should_pause: Callable[[], bool] = lambda: False,
    batch_size: int = 4,
) -> tuple[CandidateOutput, ...]:
    """Generate each candidate/example pair once and resume from authenticated rows."""
    if not 1 <= stage <= 8 or not candidates or not examples or batch_size <= 0:
        raise ValueError("candidate evaluation requires a stage, candidates, and examples")
    if any(candidate.candidate_id == BASE_CANDIDATE for candidate in candidates[1:]):
        raise ValueError("the frozen base candidate must be first when present")
    ledger = ChainedJsonlLedger(ledger_path, EVALUATION_LEDGER_FORMAT)
    expected = tuple((example, candidate) for candidate in candidates for example in examples)
    expected_keys = tuple((example.example_id, candidate.candidate_id) for example, candidate in expected)
    persisted_keys = tuple(
        (str(row["example_id"]), str(row["candidate_id"])) for row in ledger.rows
    )
    if persisted_keys != expected_keys[: len(persisted_keys)]:
        raise ValueError("evaluation ledger is not the expected deterministic prefix")
    print(
        f"TRACE phase: stage {stage} candidate evaluation "
        f"({len(expected):,} candidate/example cases)"
    )
    bar = tqdm(
        total=len(expected),
        initial=len(persisted_keys),
        desc=f"TRACE evaluation stage {stage}",
        unit="case",
        dynamic_ncols=True,
    )
    try:
        remaining = expected[len(persisted_keys) :]
        cursor = 0
        while cursor < len(remaining):
            first_candidate = remaining[cursor][1]
            group_end = cursor
            while (
                group_end < len(remaining)
                and remaining[group_end][1].candidate_id == first_candidate.candidate_id
                and group_end - cursor < batch_size
            ):
                group_end += 1
            batch_pairs = remaining[cursor:group_end]
            queries = tuple(
                PromptQuery(example.example_id, example.prompt)
                for example, _ in batch_pairs
            )
            seeds = tuple(
                stable_seed("generation", stage, candidate.candidate_id, example.example_id)
                for example, candidate in batch_pairs
            )
            batch_started = time.monotonic()
            if isinstance(runtime, BatchedCandidateRuntime):
                results = runtime.evaluate_batch(first_candidate, queries, seeds)
            else:
                results = tuple(
                    CandidateCaseResult(
                        runtime.prompt_nll(candidate, query),
                        runtime.generate(candidate, query, seed),
                    )
                    for (example, candidate), query, seed in zip(batch_pairs, queries, seeds)
                )
            if len(results) != len(batch_pairs):
                raise RuntimeError("candidate runtime returned the wrong batch size")
            case_wall_seconds = (time.monotonic() - batch_started) / len(batch_pairs)
            outputs = tuple(
                CandidateOutput(
                    example_id=example.example_id,
                    candidate_id=candidate.candidate_id,
                    prompt_nll=result.prompt_nll,
                    prediction=result.prediction,
                    generation_seed=seed,
                    prompt_tokens=result.prompt_tokens,
                    generated_tokens=result.generated_tokens,
                    case_wall_seconds=case_wall_seconds,
                )
                for (example, candidate), seed, result in zip(
                    batch_pairs,
                    seeds,
                    results,
                )
            )
            ledger.append_many(
                output.as_record(stage, example.split, example.task)
                for output, (example, _) in zip(outputs, batch_pairs)
            )
            bar.update(len(outputs))
            cursor = group_end
            if should_pause():
                raise InterruptedError("TRACE evaluation paused after a durable case batch")
    finally:
        bar.close()
    return tuple(
        CandidateOutput(
            example_id=str(row["example_id"]),
            candidate_id=str(row["candidate_id"]),
            prompt_nll=float(row["prompt_nll"]),
            prediction=str(row["prediction"]),
            generation_seed=int(row["generation_seed"]),
            prompt_tokens=(
                int(row["prompt_tokens"]) if row["prompt_tokens"] is not None else None
            ),
            generated_tokens=(
                int(row["generated_tokens"])
                if row["generated_tokens"] is not None
                else None
            ),
            case_wall_seconds=float(row["case_wall_seconds"]),
        )
        for row in ledger.rows
    )


def task_aware_candidates(
    candidates: Sequence[Candidate],
    validation_examples: Sequence[TraceExample],
    outputs: Sequence[CandidateOutput],
) -> dict[str, str]:
    """Select one validation-best candidate per encountered task."""
    candidate_order = tuple(candidate.candidate_id for candidate in candidates)
    by_key = {(output.example_id, output.candidate_id): output for output in outputs}
    tasks = tuple(dict.fromkeys(example.task for example in validation_examples))
    selections: dict[str, str] = {}
    for task in tasks:
        task_examples = tuple(example for example in validation_examples if example.task == task)
        scores = {
            candidate_id: score_task(
                task,
                tuple(example.prompt for example in task_examples),
                tuple(by_key[(example.example_id, candidate_id)].prediction for example in task_examples),
                tuple(example.answer for example in task_examples),
            )
            for candidate_id in candidate_order
        }
        selections[task] = select_best_candidate(candidate_order, scores)
    return selections


def route_outputs(
    candidates: Sequence[Candidate],
    examples: Sequence[TraceExample],
    outputs: Sequence[CandidateOutput],
    centroids: Sequence[NodeCentroid],
    query_embeddings: Mapping[str, Tensor],
    task_aware: Mapping[str, str],
) -> tuple[RoutedPrediction, ...]:
    """Derive all four routers from immutable candidate outputs."""
    candidate_order = tuple(candidate.candidate_id for candidate in candidates)
    if {centroid.candidate_id for centroid in centroids} != set(candidate_order):
        raise ValueError("centroid routing must cover base and every active candidate")
    by_key = {(output.example_id, output.candidate_id): output for output in outputs}
    tasks = {example.task for example in examples}
    if len(tasks) != 1:
        raise ValueError("one routed TRACE evaluation must contain exactly one task")
    task = next(iter(tasks))
    oracle_scores = {
        candidate: per_example_task_scores(
            task,
            tuple(example.prompt for example in examples),
            tuple(by_key[(example.example_id, candidate)].prediction for example in examples),
            tuple(example.answer for example in examples),
        )
        for candidate in candidate_order
    }
    routed: list[RoutedPrediction] = []
    for example_index, example in enumerate(examples):
        case_outputs = {candidate: by_key[(example.example_id, candidate)] for candidate in candidate_order}
        nll_candidate = select_lowest_nll(
            candidate_order,
            {candidate: output.prompt_nll for candidate, output in case_outputs.items()},
        )
        centroid_candidate = select_centroid(query_embeddings[example.example_id], centroids)
        task_candidate = task_aware[example.task]
        example_oracle_scores = {
            candidate: scores[example_index]
            for candidate, scores in oracle_scores.items()
        }
        oracle_candidate = select_best_candidate(candidate_order, example_oracle_scores)
        selections: tuple[tuple[RouterName, str], ...] = (
            ("prompt_nll", nll_candidate),
            ("frozen_prompt_centroid", centroid_candidate),
            ("task_aware", task_candidate),
            ("answer_oracle", oracle_candidate),
        )
        routed.extend(
            RoutedPrediction(
                example_id=example.example_id,
                router=router,
                candidate_id=candidate_id,
                prediction=case_outputs[candidate_id].prediction,
            )
            for router, candidate_id in selections
        )
    return tuple(routed)


__all__ = [
    "Candidate",
    "CandidateCaseResult",
    "CandidateOutput",
    "CandidateRuntime",
    "BatchedCandidateRuntime",
    "EVALUATION_LEDGER_FORMAT",
    "PeftCandidateRuntime",
    "RoutedPrediction",
    "evaluate_candidate_cache",
    "route_outputs",
    "task_aware_candidates",
]
