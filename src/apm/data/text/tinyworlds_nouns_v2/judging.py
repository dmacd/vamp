"""Optional versioned OpenRouter judging for persisted nouns-v2 generations."""

from __future__ import annotations

from pathlib import Path

from apm.data.text.tinyworlds_nouns_v1.judging import (
    DEFAULT_JUDGE_MODEL,
    JudgeCredentialsMissing,
    JudgeTransport,
    anonymize_judge_request,
    judge_generation_ledger,
    parse_judge_content,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BENCHMARK_ID,
    JUDGE_FORMAT,
    JUDGE_REQUEST_FORMAT,
)


def judge_nouns_v2_generation_ledger(
    generation_path: str | Path,
    output_root: str | Path,
    *,
    api_key: str | None,
    model: str = DEFAULT_JUDGE_MODEL,
    transport: JudgeTransport | None = None,
    maximum_attempts: int = 3,
    progress=None,
) -> Path:
    """Judge saved local rows without rerunning any model computation."""
    return judge_generation_ledger(
        generation_path,
        output_root,
        api_key=api_key,
        model=model,
        transport=transport,
        maximum_attempts=maximum_attempts,
        progress=progress,
        benchmark_id=BENCHMARK_ID,
        request_format=JUDGE_REQUEST_FORMAT,
        result_format=JUDGE_FORMAT,
    )


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JudgeCredentialsMissing",
    "anonymize_judge_request",
    "judge_nouns_v2_generation_ledger",
    "parse_judge_content",
]
