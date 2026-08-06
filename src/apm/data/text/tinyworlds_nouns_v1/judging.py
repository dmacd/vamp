"""Resumable, anonymously shuffled OpenRouter judging for noun completions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Protocol

from apm.data.text.tinyworlds_nouns_v1.contracts import (
    CONDITIONS,
    BENCHMARK_ID,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)


DEFAULT_JUDGE_MODEL = "z-ai/glm-5.2"
JUDGE_FORMAT = "tinyworlds-nouns-judge-result-v1"
JUDGE_REQUEST_FORMAT = "tinyworlds-nouns-judge-request-v1"
JUDGE_SOURCE_NAMES = (*CONDITIONS, "reference")
JudgeProgress = Callable[[str, int, int], None]


class JudgeCredentialsMissing(RuntimeError):
    """Local experiment phases finished but judging has no API credential."""


class JudgeResponseError(RuntimeError):
    """One judge response is terminally malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class JudgeHttpResponse:
    """Exact HTTP status and response body returned by an injected transport."""

    status_code: int
    body: bytes


class JudgeTransport(Protocol):
    """Minimal OpenRouter boundary used by production and deterministic tests."""

    def model_available(self, model: str) -> bool:
        """Return whether the exact public model slug is currently listed."""
        ...

    def post(self, api_key: str, body: bytes) -> JudgeHttpResponse:
        """Send one chat-completion request and return exact response bytes."""
        ...


@dataclass(frozen=True, slots=True)
class JudgeCandidateScore:
    """One anonymous candidate's four integer scores and short explanation."""

    candidate: str
    coherence: int
    writing_quality: int
    ending_quality: int
    overall: int
    reason: str

    def __post_init__(self) -> None:
        if len(self.candidate) != 1 or not "A" <= self.candidate <= "G":
            raise ValueError("judge candidate labels must be A through G")
        if any(
            type(value) is not int or not 1 <= value <= 5
            for value in (
                self.coherence,
                self.writing_quality,
                self.ending_quality,
                self.overall,
            )
        ):
            raise ValueError("judge scores must be integers from one through five")
        if not self.reason.strip():
            raise ValueError("judge score reason must not be empty")

    def as_record(self) -> dict[str, object]:
        """Return one canonical score object."""
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ParsedJudgeResult:
    """Schema-checked scores and complete seven-way ranking."""

    scores: tuple[JudgeCandidateScore, ...]
    ranking: tuple[str, ...]

    def __post_init__(self) -> None:
        labels = tuple(chr(ord("A") + index) for index in range(7))
        if tuple(sorted(score.candidate for score in self.scores)) != labels:
            raise ValueError("judge scores must contain every candidate exactly once")
        if tuple(sorted(self.ranking)) != labels or len(set(self.ranking)) != 7:
            raise ValueError("judge ranking must contain all seven candidates once")

    def as_record(self) -> dict[str, object]:
        """Return the canonical parsed model result."""
        return {
            "ranking": list(self.ranking),
            "scores": [score.as_record() for score in self.scores],
        }


@dataclass(frozen=True, slots=True)
class AnonymizedJudgeRequest:
    """One persisted request plus the evaluator-only label/source map."""

    task_noun: str
    story_id: str
    model: str
    label_sources: tuple[tuple[str, str], ...]
    body: bytes

    def __post_init__(self) -> None:
        require_sha256(self.story_id, "judge story")
        labels = tuple(label for label, _ in self.label_sources)
        sources = tuple(source for _, source in self.label_sources)
        if labels != tuple(chr(ord("A") + index) for index in range(7)):
            raise ValueError("judge labels must be ordered A through G")
        if set(sources) != set(JUDGE_SOURCE_NAMES):
            raise ValueError("judge request must anonymize all six systems and reference")
        if not self.model or not self.body:
            raise ValueError("judge request model and body must be nonempty")

    @property
    def request_sha256(self) -> str:
        """Return the exact request identity without secrets."""
        return sha256(self.body).hexdigest()


class HttpxOpenRouterJudgeTransport:
    """Small synchronous OpenRouter transport with no stored credential."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("judge timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def model_available(self, model: str) -> bool:
        """Check the public model catalog for the exact requested slug."""
        import httpx

        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return any(item.get("id") == model for item in payload.get("data", ()))

    def post(self, api_key: str, body: bytes) -> JudgeHttpResponse:
        """Post one exact JSON body to OpenRouter's chat-completions endpoint."""
        import httpx

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=body,
            timeout=self._timeout_seconds,
        )
        return JudgeHttpResponse(response.status_code, response.content)


def anonymize_judge_request(
    generation_record: dict[str, object],
    model: str,
) -> AnonymizedJudgeRequest:
    """Deterministically shuffle all six completions and the true reference."""
    task_noun = _text(generation_record.get("task_noun"), "judge task")
    story_id = _text(generation_record.get("story_id"), "judge story")
    prefix = _text(generation_record.get("prefix"), "judge prefix")
    results = _object(generation_record.get("results"), "generation results")
    continuation_by_source = {
        condition: _text(
            _object(results.get(condition), f"result {condition}").get(
                "generated_continuation"
            ),
            f"continuation {condition}",
            allow_empty=True,
        )
        for condition in CONDITIONS
    }
    continuation_by_source["reference"] = _text(
        generation_record.get("reference_continuation"),
        "reference continuation",
        allow_empty=True,
    )
    ordered_sources = tuple(
        sorted(
            JUDGE_SOURCE_NAMES,
            key=lambda source: (
                sha256(
                    f"{BENCHMARK_ID}\0judge-shuffle\0{model}\0"
                    f"{task_noun}\0{story_id}\0{source}".encode("utf-8")
                ).hexdigest(),
                source,
            ),
        )
    )
    label_sources = tuple(
        (chr(ord("A") + index), source)
        for index, source in enumerate(ordered_sources)
    )
    candidates = "\n\n".join(
        f"Candidate {label}:\n{continuation_by_source[source]}"
        for label, source in label_sources
    )
    instruction = (
        "You are comparing seven possible continuations of the same children's "
        "story. Judge only the supplied text. For every candidate, score coherence, "
        "writing_quality, ending_quality, and overall as integers from 1 to 5, then "
        "rank all seven candidates best to worst. Return JSON only with fields scores "
        "and ranking. Each score object must have candidate, coherence, "
        "writing_quality, ending_quality, overall, and a brief reason. Include A "
        "through G exactly once in both scores and ranking.\n\n"
        f"Story prefix:\n{prefix}\n\n{candidates}"
    )
    body = canonical_json_bytes(
        {
            "messages": [{"content": instruction, "role": "user"}],
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
    )
    return AnonymizedJudgeRequest(task_noun, story_id, model, label_sources, body)


def parse_judge_content(content: str) -> ParsedJudgeResult:
    """Strictly parse the model-authored inner JSON result."""
    try:
        record = json.loads(content)
    except json.JSONDecodeError as error:
        raise JudgeResponseError("judge content is not JSON") from error
    if type(record) is not dict or set(record) != {"scores", "ranking"}:
        raise JudgeResponseError("judge content fields must be scores and ranking")
    raw_scores = record["scores"]
    raw_ranking = record["ranking"]
    if type(raw_scores) is not list or type(raw_ranking) is not list:
        raise JudgeResponseError("judge scores and ranking must be arrays")
    try:
        scores = tuple(
            JudgeCandidateScore(
                candidate=_text(item.get("candidate"), "score candidate"),
                coherence=_score(item.get("coherence"), "coherence"),
                writing_quality=_score(
                    item.get("writing_quality"), "writing quality"
                ),
                ending_quality=_score(item.get("ending_quality"), "ending quality"),
                overall=_score(item.get("overall"), "overall"),
                reason=_text(item.get("reason"), "score reason"),
            )
            for raw_item in raw_scores
            for item in (_object(raw_item, "judge score"),)
        )
        return ParsedJudgeResult(
            tuple(sorted(scores, key=lambda score: score.candidate)),
            tuple(_text(item, "ranking candidate") for item in raw_ranking),
        )
    except (TypeError, ValueError) as error:
        raise JudgeResponseError(str(error)) from error


def judge_generation_ledger(
    generation_path: str | Path,
    output_root: str | Path,
    *,
    api_key: str | None,
    model: str = DEFAULT_JUDGE_MODEL,
    transport: JudgeTransport | None = None,
    maximum_attempts: int = 3,
    progress: JudgeProgress | None = None,
) -> Path:
    """Judge every persisted generation case, saving each request and response."""
    root = Path(output_root)
    final = root / "judge-results.jsonl"
    if not model or maximum_attempts < 1:
        raise ValueError("judge model and retry budget must be valid")
    generations = tuple(
        json.loads(line)
        for line in Path(generation_path).read_text(encoding="utf-8").splitlines()
        if line
    )
    generation_keys = tuple(
        (
            _text(record.get("task_noun"), "generation task"),
            _text(record.get("story_id"), "generation story"),
        )
        for record in generations
    )
    if not generation_keys or len(set(generation_keys)) != len(generation_keys):
        raise ValueError("judging requires unique nonempty task/story generation cases")
    expected = {
        (request.task_noun, request.story_id, request.request_sha256)
        for generation in generations
        for request in (anonymize_judge_request(generation, model),)
    }
    if final.is_file():
        if _completed_judge_keys(final, truncate_incomplete=False) != expected:
            raise ValueError("published judge ledger coverage changed")
        return final
    if not api_key:
        raise JudgeCredentialsMissing(
            "OPENROUTER_API_KEY is absent; local experiment phases are complete and "
            "the canonical runner will resume at judging."
        )
    client = transport or HttpxOpenRouterJudgeTransport()
    if not client.model_available(model):
        raise RuntimeError(f"configured OpenRouter judge model is unavailable: {model}")
    request_root = root / "judge-requests"
    raw_root = root / "judge-raw"
    request_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    work = root / ".judge-results.jsonl.work"
    completed = _completed_judge_keys(work, truncate_incomplete=True)
    if not completed <= expected:
        raise ValueError("judge work ledger contains unexpected cases")
    with work.open("ab") as ledger:
        for generation in generations:
            request = anonymize_judge_request(generation, model)
            key = (request.task_noun, request.story_id, request.request_sha256)
            if key in completed:
                continue
            request_record = {
                "body": json.loads(request.body),
                "format": JUDGE_REQUEST_FORMAT,
                "label_sources": dict(request.label_sources),
                "model": model,
                "request_sha256": request.request_sha256,
                "story_id": request.story_id,
                "task_noun": request.task_noun,
            }
            request_path = request_root / f"{request.request_sha256}.json"
            _write_once(request_path, canonical_json_bytes(request_record))
            raw_path = raw_root / f"{request.request_sha256}.json"
            if not raw_path.is_file():
                response = _request_with_retries(
                    client,
                    api_key,
                    request.body,
                    maximum_attempts,
                )
                _write_once(raw_path, response.body)
            raw_body = raw_path.read_bytes()
            content = _openrouter_content(raw_body)
            parsed = parse_judge_content(content)
            result_core = {
                "format": JUDGE_FORMAT,
                "label_sources": dict(request.label_sources),
                "model": model,
                "parsed": parsed.as_record(),
                "raw_response_sha256": sha256(raw_body).hexdigest(),
                "request_sha256": request.request_sha256,
                "story_id": request.story_id,
                "task_noun": request.task_noun,
            }
            ledger.write(
                canonical_json_bytes(
                    {**result_core, "result_sha256": record_sha256(result_core)}
                )
            )
            ledger.flush()
            os.fsync(ledger.fileno())
            completed.add(key)
            if progress is not None:
                progress("openrouter-judging", len(completed), len(expected))
    if completed != expected:
        raise RuntimeError("judge result ledger is incomplete")
    os.replace(work, final)
    return final


def _completed_judge_keys(
    path: Path,
    *,
    truncate_incomplete: bool,
) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        if not truncate_incomplete:
            raise ValueError("published judge ledger ends with an incomplete row")
        payload = payload[: payload.rfind(b"\n") + 1]
        _write_replacement(path, payload)
    keys: set[tuple[str, str, str]] = set()
    for line in payload.splitlines():
        record = json.loads(line)
        if line + b"\n" != canonical_json_bytes(record):
            raise ValueError("judge ledger is not canonical JSONL")
        supplied = record.pop("result_sha256", None)
        if supplied != record_sha256(record):
            raise ValueError("judge result identity changed")
        key = (
            _text(record.get("task_noun"), "judge task"),
            _text(record.get("story_id"), "judge story"),
            _text(record.get("request_sha256"), "judge request"),
        )
        if key in keys:
            raise ValueError("judge ledger contains duplicate cases")
        keys.add(key)
    return keys


def _request_with_retries(
    transport: JudgeTransport,
    api_key: str,
    body: bytes,
    maximum_attempts: int,
) -> JudgeHttpResponse:
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = transport.post(api_key, body)
        except OSError:
            if attempt == maximum_attempts:
                raise
            time.sleep(2 ** (attempt - 1))
            continue
        if 200 <= response.status_code < 300:
            return response
        if response.status_code != 429 and response.status_code < 500:
            raise RuntimeError(
                f"OpenRouter judge returned terminal HTTP {response.status_code}; "
                f"body SHA-256={sha256(response.body).hexdigest()}"
            )
        if attempt == maximum_attempts:
            raise RuntimeError(
                f"OpenRouter judge retries exhausted at HTTP {response.status_code}"
            )
        time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable judge retry state")


def _openrouter_content(payload: bytes) -> str:
    try:
        record = json.loads(payload)
        return _text(
            record["choices"][0]["message"]["content"],
            "OpenRouter judge content",
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise JudgeResponseError(
            f"OpenRouter response schema is invalid; SHA-256={sha256(payload).hexdigest()}"
        ) from error


def _write_once(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable judge artifact changed: {path}")
        return
    _write_replacement(path, payload)


def _write_replacement(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise TypeError(f"{label} must be text")
    return value


def _score(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"judge {label} score must be an integer")
    return value


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "AnonymizedJudgeRequest",
    "HttpxOpenRouterJudgeTransport",
    "JudgeCandidateScore",
    "JudgeCredentialsMissing",
    "JudgeHttpResponse",
    "JudgeResponseError",
    "JudgeTransport",
    "ParsedJudgeResult",
    "anonymize_judge_request",
    "judge_generation_ledger",
    "parse_judge_content",
]
