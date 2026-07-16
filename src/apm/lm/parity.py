"""Pure layerwise parity records and comparisons for converted GPT-Neo models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import CapturePoint, CaptureSpec

DEFAULT_RTOL = 2e-4
DEFAULT_ATOL = 2e-4


@dataclass(frozen=True)
class ParitySnapshot:
    """Ordered values required by the complete GPT-Neo parity ladder."""

    token_ids: object
    embedding_output: object
    position_ids: object
    attention_masks: tuple[object, ...]
    captured_hidden: tuple[object, ...]
    final_hidden: object
    logits: object
    normalized_nll: object
    greedy_token_ids: object


@dataclass(frozen=True)
class ParityErrorRecord:
    """Immutable absolute-error summary for one parity site or block capture."""

    site: str
    layer_index: int | None
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    max_absolute_error: float
    mean_absolute_error: float
    passed: bool
    exact: bool

    @property
    def label(self) -> str:
        """Return a compact layer-qualified site name."""
        return self.site if self.layer_index is None else f"layer {self.layer_index} {self.site}"


@dataclass(frozen=True)
class ParityReport:
    """One fixed-tolerance ordered parity report."""

    records: tuple[ParityErrorRecord, ...]
    rtol: float
    atol: float

    @property
    def passed(self) -> bool:
        """Return whether every parity site satisfied its fixed criterion."""
        return all(record.passed for record in self.records)

    @property
    def failures(self) -> tuple[ParityErrorRecord, ...]:
        """Return failed records in parity-ladder order."""
        return tuple(record for record in self.records if not record.passed)


def ordered_capture_spec(config: GptNeoConfig) -> CaptureSpec:
    """Request post-attention then post-MLP residuals for every block."""
    return CaptureSpec(
        points=tuple(
            CapturePoint(layer_index, location)
            for layer_index in range(config.num_layers)
            for location in ("post_attention", "post_mlp")
        )
    )


def compare_parity_snapshots(
    expected: ParitySnapshot,
    actual: ParitySnapshot,
    config: GptNeoConfig,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityReport:
    """Compare the complete parity ladder without changing its tolerance."""
    _validate_tolerances(rtol, atol)
    _validate_snapshot_structure(expected, config, "expected")
    _validate_snapshot_structure(actual, config, "actual")
    records: list[ParityErrorRecord] = [
        compare_parity_values(
            "token_ids",
            expected.token_ids,
            actual.token_ids,
            exact=True,
            rtol=rtol,
            atol=atol,
        ),
        compare_parity_values(
            "embedding_output",
            expected.embedding_output,
            actual.embedding_output,
            rtol=rtol,
            atol=atol,
        ),
        compare_parity_values(
            "position_ids",
            expected.position_ids,
            actual.position_ids,
            exact=True,
            rtol=rtol,
            atol=atol,
        ),
    ]
    records.extend(
        compare_parity_values(
            f"attention_mask.{attention_type}",
            expected.attention_masks[layer_index],
            actual.attention_masks[layer_index],
            layer_index=layer_index,
            exact=True,
            rtol=rtol,
            atol=atol,
        )
        for layer_index, attention_type in enumerate(config.attention_types)
    )
    records.extend(
        compare_parity_values(
            capture_point.location,
            expected.captured_hidden[capture_index],
            actual.captured_hidden[capture_index],
            layer_index=capture_point.layer_index,
            rtol=rtol,
            atol=atol,
        )
        for capture_index, capture_point in enumerate(ordered_capture_spec(config).points)
    )
    records.extend(
        (
            compare_parity_values(
                "final_hidden",
                expected.final_hidden,
                actual.final_hidden,
                rtol=rtol,
                atol=atol,
            ),
            compare_parity_values(
                "logits",
                expected.logits,
                actual.logits,
                rtol=rtol,
                atol=atol,
            ),
            compare_parity_values(
                "normalized_nll",
                expected.normalized_nll,
                actual.normalized_nll,
                rtol=rtol,
                atol=atol,
            ),
            compare_parity_values(
                "greedy_token_ids",
                expected.greedy_token_ids,
                actual.greedy_token_ids,
                exact=True,
                rtol=rtol,
                atol=atol,
            ),
        )
    )
    return ParityReport(records=tuple(records), rtol=rtol, atol=atol)


def compare_parity_values(
    site: str,
    expected: object,
    actual: object,
    *,
    layer_index: int | None = None,
    exact: bool = False,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityErrorRecord:
    """Return a pure shape, absolute-error, and tolerance comparison record."""
    _validate_tolerances(rtol, atol)
    if not site:
        raise ValueError("parity site must not be empty")
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    expected_shape = tuple(expected_array.shape)
    actual_shape = tuple(actual_array.shape)
    if expected_shape != actual_shape:
        return ParityErrorRecord(
            site=site,
            layer_index=layer_index,
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            max_absolute_error=float("inf"),
            mean_absolute_error=float("inf"),
            passed=False,
            exact=exact,
        )
    absolute_error = np.abs(
        expected_array.astype(np.float64) - actual_array.astype(np.float64)
    )
    finite_errors = np.isfinite(absolute_error)
    max_absolute_error = (
        float(np.max(absolute_error))
        if absolute_error.size and np.all(finite_errors)
        else (0.0 if absolute_error.size == 0 else float("inf"))
    )
    mean_absolute_error = (
        float(np.mean(absolute_error))
        if absolute_error.size and np.all(finite_errors)
        else (0.0 if absolute_error.size == 0 else float("inf"))
    )
    passed = (
        bool(np.array_equal(expected_array, actual_array))
        if exact
        else bool(
            np.allclose(
                expected_array,
                actual_array,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        )
    )
    return ParityErrorRecord(
        site=site,
        layer_index=layer_index,
        expected_shape=expected_shape,
        actual_shape=actual_shape,
        max_absolute_error=max_absolute_error,
        mean_absolute_error=mean_absolute_error,
        passed=passed,
        exact=exact,
    )


def assert_parity(report: ParityReport) -> None:
    """Raise one detailed assertion containing every failed parity site."""
    if report.passed:
        return
    failure_lines = tuple(
        (
            f"{record.label}: shapes {record.expected_shape}/{record.actual_shape}, "
            f"max_abs={record.max_absolute_error:.8g}, "
            f"mean_abs={record.mean_absolute_error:.8g}, "
            f"criterion={'exact' if record.exact else f'rtol={report.rtol:g}, atol={report.atol:g}'}"
        )
        for record in report.failures
    )
    raise AssertionError("GPT-Neo parity failed:\n" + "\n".join(failure_lines))


def _validate_snapshot_structure(
    snapshot: ParitySnapshot,
    config: GptNeoConfig,
    name: str,
) -> None:
    if len(snapshot.attention_masks) != config.num_layers:
        raise ValueError(f"{name} snapshot must contain one attention mask per layer")
    if len(snapshot.captured_hidden) != 2 * config.num_layers:
        raise ValueError(
            f"{name} snapshot must contain ordered post-attention/post-MLP captures"
        )


def _validate_tolerances(rtol: float, atol: float) -> None:
    if not isfinite(rtol) or not isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("parity tolerances must be finite and nonnegative")
