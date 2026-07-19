"""Deterministic generation preflight estimates and the Phase 1 cost gate."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import os
from pathlib import Path
from threading import Lock
from typing import Iterable

from apm.data.text.tinyworlds_v2.byok_contract import canonical_byok_authorization
from apm.data.text.tinyworlds_v2.generation_cache import (
    CostJournalEntry,
    ImmutableRawCache,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    GenerationContractError,
    RawAttempt,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import JsonObject, json_sha256


PHASE1_HARD_CAP_USD = "15.00"
_MILLION = Decimal(1_000_000)
_DISPLAY_QUANTUM = Decimal("0.000001")


class CostCapExceeded(RuntimeError):
    """The conservative generation preflight exceeds its authorized cap."""


class CostJournalRecoveryRequired(RuntimeError):
    """A prior billable POST has no complete durable response observation."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            "runtime OpenRouter work is halted by cost-journal recovery: "
            f"{reason}"
        )
        self.reason = reason


class PaidRunLockUnavailable(RuntimeError):
    """Another process already owns the raw cache's paid-run lease."""


@contextmanager
def exclusive_paid_run_lock(raw_cache: Path) -> Iterator[Path]:
    """Hold one nonblocking cross-process lease for a raw cache lifecycle."""
    if not isinstance(raw_cache, Path):
        raise TypeError("raw_cache must be a Path")
    if not raw_cache.name:
        raise GenerationContractError("raw cache must have a final path component")
    raw_cache.parent.mkdir(parents=True, exist_ok=True)
    lock_path = raw_cache.with_name(f".{raw_cache.name}.paid-run.lock")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PaidRunLockUnavailable(
                "another process already owns the TinyWorlds-v2 paid-run lock"
            ) from error
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class RequestCostUpperBound:
    """A per-attempt ceiling derived solely from frozen request/route data."""

    request_sha256: str
    route_id: str
    request_body_bytes: int
    maximum_output_tokens: int
    upper_bound_usd: str

    def as_record(self) -> JsonObject:
        return {
            "maximum_output_tokens": self.maximum_output_tokens,
            "request_body_bytes": self.request_body_bytes,
            "request_sha256": self.request_sha256,
            "route_id": self.route_id,
            "upper_bound_usd": self.upper_bound_usd,
        }


@dataclass(frozen=True, slots=True)
class CostReservation:
    """Opaque identity for one in-flight completion POST reservation."""

    request_sha256: str
    attempt_number: int
    upper_bound_usd: str


@dataclass(frozen=True, slots=True)
class RuntimeCostSnapshot:
    """Thread-safe runtime-cap evidence, separating actual and unknown charges."""

    hard_cap_usd: str
    provider_reported_actual_usd: str
    conservative_unknown_charge_usd: str
    charged_total_usd: str
    in_flight_reserved_usd: str
    provider_reported_attempt_count: int
    unknown_cost_attempt_count: int
    in_flight_attempt_count: int
    cancelled_before_post_count: int
    halted_reason: str | None

    def as_record(self) -> JsonObject:
        return {
            "charged_total_usd": self.charged_total_usd,
            "cancelled_before_post_count": self.cancelled_before_post_count,
            "conservative_unknown_charge_usd": self.conservative_unknown_charge_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "halted_reason": self.halted_reason,
            "in_flight_attempt_count": self.in_flight_attempt_count,
            "in_flight_reserved_usd": self.in_flight_reserved_usd,
            "provider_reported_actual_usd": self.provider_reported_actual_usd,
            "provider_reported_attempt_count": self.provider_reported_attempt_count,
            "unknown_cost_attempt_count": self.unknown_cost_attempt_count,
        }


class RuntimeCostLedger:
    """Inclusive hard-cap ledger shared by every concurrent OpenRouter worker.

    Cached HTTP responses are reconciled before new work.  A response without
    a provider-reported cost consumes its complete request upper bound, but the
    bound remains explicitly classified as an unknown conservative charge and
    is never mislabeled as actual billing.
    """

    def __init__(self, hard_cap_usd: str = PHASE1_HARD_CAP_USD) -> None:
        self._hard_cap = _decimal(hard_cap_usd, "hard cap")
        self._lock = Lock()
        self._charged: dict[tuple[str, int], tuple[Decimal, bool]] = {}
        self._reservations: dict[tuple[str, int], Decimal] = {}
        self._cancelled: set[tuple[str, int]] = set()
        self._halted_reason: str | None = None
        self._cache: ImmutableRawCache | None = None
        self._journal_entries: dict[
            tuple[str, int], CostJournalEntry
        ] = {}
        self._missing_response_keys: set[tuple[str, int]] = set()
        self._byok_authorization: JsonObject | None = None

    def authorize_byok(self, evidence: JsonObject) -> None:
        """Install one sanitized allowed authorization for later reservations."""
        authorization = canonical_byok_authorization(evidence)
        with self._lock:
            self._byok_authorization = authorization

    @property
    def hard_cap_usd(self) -> str:
        return _display_usd(self._hard_cap)

    def bootstrap(
        self,
        cache: ImmutableRawCache,
        routes: Iterable[RouteLock],
    ) -> None:
        """Reconcile the complete dedicated raw cache before the first POST."""
        self.attach_cache(cache)
        route_by_lock = {route.lock_sha256: route for route in routes}
        if not route_by_lock:
            raise GenerationContractError(
                "runtime cost ledger requires at least one route lock"
            )
        for request in cache.load_all_requests():
            route = cache.load_route_lock(request.request_sha256)
            current = route_by_lock.get(request.route_lock_sha256)
            if current is not None and current.lock_sha256 != route.lock_sha256:
                raise GenerationContractError(
                    "cached request route differs from its current route lock"
                )
            if route.route_id not in {item.route_id for item in route_by_lock.values()}:
                raise GenerationContractError(
                    "cached request route ID is outside the Phase 1 plan"
                )
            bound = request_cost_upper_bound(request, route)
            self._reconcile_cached_bound(
                request,
                cache.load_attempts(request),
                bound,
            )
        self.ensure_recovery_complete()

    def attach_cache(self, cache: ImmutableRawCache) -> None:
        """Attach and reconcile the durable write-ahead cost journal once."""
        if type(cache) is not ImmutableRawCache:
            raise TypeError("cache must be an ImmutableRawCache")
        with self._lock:
            if self._cache is not None:
                if self._cache.root != cache.root:
                    raise GenerationContractError(
                        "runtime cost ledger cannot span multiple raw caches"
                    )
                return
            self._cache = cache
            for entry in cache.load_cost_journal():
                key = (entry.request_sha256, entry.attempt_number)
                self._journal_entries[key] = entry
                if entry.cancelled_before_post:
                    self._cancelled.add(key)
                    continue
                amount = Decimal(
                    entry.upper_bound_usd
                    if entry.charged_usd is None
                    else entry.charged_usd
                )
                known = entry.provider_reported_actual is True
                self._record_charge_locked(
                    key,
                    amount,
                    known,
                )
                self._missing_response_keys.add(key)

            for request in cache.load_all_requests():
                self._reconcile_journal_attempts_locked(
                    request,
                    cache.load_attempts(request),
                )
            self._refresh_recovery_halt_locked()

    def ensure_recovery_complete(self) -> None:
        """Raise a production-safe stop when a journal entry lacks its response."""
        with self._lock:
            if self._halted_reason in {
                "billed_attempt_response_missing",
                "orphaned_cost_reservation",
            }:
                raise CostJournalRecoveryRequired(self._halted_reason)

    def reconcile_cached(
        self,
        request: CanonicalRequest,
        route: RouteLock,
        attempts: tuple[RawAttempt, ...],
    ) -> None:
        """Idempotently charge all cached HTTP responses for one request."""
        upper_bound = request_cost_upper_bound(request, route)
        self._reconcile_cached_bound(request, attempts, upper_bound)

    def _reconcile_cached_bound(
        self,
        request: CanonicalRequest,
        attempts: tuple[RawAttempt, ...],
        upper_bound: RequestCostUpperBound,
    ) -> None:
        with self._lock:
            self._reconcile_journal_attempts_locked(request, attempts)
            for attempt in attempts:
                if attempt.request_sha256 != request.request_sha256:
                    raise GenerationContractError(
                        "cached billing attempt belongs to another request"
                    )
                key = (request.request_sha256, attempt.attempt_number)
                actual = _cached_attempt_actual_cost(attempt)
                amount = (
                    Decimal(upper_bound.upper_bound_usd)
                    if actual is None
                    else _decimal(actual, "provider-reported billed cost")
                )
                self._record_charge_locked(key, amount, actual is not None)
            self._refresh_recovery_halt_locked()

    def reserve(
        self,
        request: CanonicalRequest,
        route: RouteLock,
        attempt_number: int,
    ) -> CostReservation:
        """Reserve one safe upper bound, rejecting before POST when cap would exceed."""
        if type(attempt_number) is not int or attempt_number < 1:
            raise GenerationContractError("attempt_number must be positive")
        bound = request_cost_upper_bound(request, route)
        amount = Decimal(bound.upper_bound_usd)
        key = (request.request_sha256, attempt_number)
        with self._lock:
            if self._halted_reason in {
                "billed_attempt_response_missing",
                "orphaned_cost_reservation",
            }:
                raise CostJournalRecoveryRequired(self._halted_reason)
            if self._halted_reason is not None:
                raise CostCapExceeded(
                    f"runtime OpenRouter work is halted: {self._halted_reason}"
                )
            if self._byok_authorization is None:
                raise GenerationContractError(
                    "runtime cost reservation lacks zero-BYOK authorization"
                )
            if (
                key in self._charged
                or key in self._reservations
                or key in self._cancelled
            ):
                raise GenerationContractError(
                    "runtime cost attempt was already charged, reserved, or cancelled"
                )
            projected = self._charged_total_locked() + sum(
                self._reservations.values(), Decimal(0)
            ) + amount
            # The cap is inclusive: exactly $15.00 is authorized; any amount
            # above it is rejected before the billable transport is invoked.
            if projected > self._hard_cap:
                self._halted_reason = "runtime_cap_reservation_denied"
                raise CostCapExceeded(
                    "runtime OpenRouter cap would be exceeded before POST: "
                    f"projected ${format(projected, 'f')} > "
                    f"${format(self._hard_cap, 'f')}"
                )
            if self._cache is None:
                raise GenerationContractError(
                    "runtime cost ledger must attach a raw cache before reserve"
                )
            self._cache.prepare_request(request, route)
            self._cache.store_cost_reservation(
                request.request_sha256,
                attempt_number,
                format(amount, "f"),
                self._byok_authorization,
            )
            self._journal_entries[key] = CostJournalEntry(
                request_sha256=request.request_sha256,
                attempt_number=attempt_number,
                upper_bound_usd=format(amount, "f"),
                charged_usd=None,
                provider_reported_actual=None,
                cancelled_before_post=False,
                byok_authorization=self._byok_authorization,
                byok_authorization_sha256=json_sha256(
                    self._byok_authorization
                ),
            )
            self._reservations[key] = amount
        return CostReservation(
            request_sha256=request.request_sha256,
            attempt_number=attempt_number,
            upper_bound_usd=format(amount, "f"),
        )

    def settle_ambiguous_transport_failure(
        self,
        reservation: CostReservation,
    ) -> None:
        """Charge the bound when a POST may have run despite no response."""
        with self._lock:
            amount = self._reservation_amount_locked(reservation)
            key = (reservation.request_sha256, reservation.attempt_number)
            assert self._cache is not None
            self._cache.store_cost_settlement(
                reservation.request_sha256,
                reservation.attempt_number,
                charged_usd=format(amount, "f"),
                provider_reported_actual=False,
            )
            previous = self._journal_entries[key]
            self._journal_entries[key] = CostJournalEntry(
                request_sha256=reservation.request_sha256,
                attempt_number=reservation.attempt_number,
                upper_bound_usd=reservation.upper_bound_usd,
                charged_usd=format(amount, "f"),
                provider_reported_actual=False,
                cancelled_before_post=False,
                byok_authorization=previous.byok_authorization,
                byok_authorization_sha256=previous.byok_authorization_sha256,
            )
            self._reservations.pop(key)
            self._record_charge_locked(key, amount, False)
            if self._halted_reason in (None, "runtime_cap_reservation_denied"):
                self._halted_reason = "provider_billing_unknown"

    def settle_response(
        self,
        reservation: CostReservation,
        billed_cost_usd: str | None,
    ) -> None:
        """Replace an in-flight reserve with actual cost or a fail-closed bound."""
        with self._lock:
            bound = self._reservation_amount_locked(reservation)
            actual_known = billed_cost_usd is not None
            amount = (
                bound
                if billed_cost_usd is None
                else _decimal(billed_cost_usd, "provider-reported billed cost")
            )
            if amount > bound:
                self._halted_reason = "provider_cost_exceeds_reserved_bound"
                raise GenerationContractError(
                    "provider-reported cost exceeds the locked request upper bound"
                )
            assert self._cache is not None
            self._cache.store_cost_settlement(
                reservation.request_sha256,
                reservation.attempt_number,
                charged_usd=format(amount, "f"),
                provider_reported_actual=actual_known,
            )
            key = (reservation.request_sha256, reservation.attempt_number)
            previous = self._journal_entries[key]
            self._journal_entries[key] = CostJournalEntry(
                request_sha256=reservation.request_sha256,
                attempt_number=reservation.attempt_number,
                upper_bound_usd=reservation.upper_bound_usd,
                charged_usd=format(amount, "f"),
                provider_reported_actual=actual_known,
                cancelled_before_post=False,
                byok_authorization=previous.byok_authorization,
                byok_authorization_sha256=previous.byok_authorization_sha256,
            )
            self._reservations.pop(key)
            self._record_charge_locked(key, amount, actual_known)
            if not actual_known and self._halted_reason in (
                None,
                "runtime_cap_reservation_denied",
            ):
                self._halted_reason = "provider_billing_unknown"

    def ensure_reservation_postable(
        self,
        reservation: CostReservation,
    ) -> None:
        """Close the reserve-to-transport race after another worker halts."""
        with self._lock:
            self._reservation_amount_locked(reservation)
            # A denied *new* reservation is not evidence that an earlier,
            # already-authorized reservation became unsafe to POST.  Its full
            # bound is already included in the inclusive cap calculation.
            if self._halted_reason in (None, "runtime_cap_reservation_denied"):
                return
            assert self._cache is not None
            key = (reservation.request_sha256, reservation.attempt_number)
            self._cache.store_cost_cancellation(
                reservation.request_sha256,
                reservation.attempt_number,
            )
            previous = self._journal_entries[key]
            self._journal_entries[key] = CostJournalEntry(
                request_sha256=reservation.request_sha256,
                attempt_number=reservation.attempt_number,
                upper_bound_usd=reservation.upper_bound_usd,
                charged_usd=None,
                provider_reported_actual=None,
                cancelled_before_post=True,
                byok_authorization=previous.byok_authorization,
                byok_authorization_sha256=previous.byok_authorization_sha256,
            )
            self._reservations.pop(key)
            self._cancelled.add(key)
            raise CostCapExceeded(
                "runtime OpenRouter work halted before the reserved POST began: "
                f"{self._halted_reason}"
            )

    def halt(self, reason: str) -> None:
        """Prevent every later worker from beginning another completion POST."""
        if type(reason) is not str or not reason.strip() or reason != reason.strip():
            raise GenerationContractError("runtime halt reason must be nonempty")
        with self._lock:
            if self._halted_reason is None or (
                self._halted_reason == "runtime_cap_reservation_denied"
                and reason != "runtime_cap_reservation_denied"
            ):
                self._halted_reason = reason

    def snapshot(self) -> RuntimeCostSnapshot:
        """Return immutable evidence without exposing mutable ledger internals."""
        with self._lock:
            actual = sum(
                (amount for amount, known in self._charged.values() if known),
                Decimal(0),
            )
            unknown = sum(
                (amount for amount, known in self._charged.values() if not known),
                Decimal(0),
            )
            reserved = sum(self._reservations.values(), Decimal(0))
            return RuntimeCostSnapshot(
                hard_cap_usd=_display_usd(self._hard_cap),
                provider_reported_actual_usd=format(actual, "f"),
                conservative_unknown_charge_usd=format(unknown, "f"),
                charged_total_usd=format(actual + unknown, "f"),
                in_flight_reserved_usd=format(reserved, "f"),
                provider_reported_attempt_count=sum(
                    known for _, known in self._charged.values()
                ),
                unknown_cost_attempt_count=sum(
                    not known for _, known in self._charged.values()
                ),
                in_flight_attempt_count=len(self._reservations),
                cancelled_before_post_count=len(self._cancelled),
                halted_reason=self._halted_reason,
            )

    def _record_charge_locked(
        self,
        key: tuple[str, int],
        amount: Decimal,
        actual_known: bool,
    ) -> None:
        previous = self._charged.get(key)
        observation = (amount, actual_known)
        if previous is not None:
            previous_amount, previous_known = previous
            if not previous_known and actual_known:
                if amount > previous_amount:
                    raise GenerationContractError(
                        "provider-reported cost exceeds the prior unknown-cost bound"
                    )
                self._charged[key] = observation
                return
            if previous != observation:
                raise GenerationContractError(
                    "cached cost observation changed after reconciliation"
                )
            return
        if self._charged_total_locked() + amount > self._hard_cap:
            self._halted_reason = "cached_charges_exceed_runtime_cap"
            raise CostCapExceeded(
                "cached OpenRouter charges exceed the authorized hard cap"
            )
        self._charged[key] = observation

    def _reconcile_journal_attempts_locked(
        self,
        request: CanonicalRequest,
        attempts: tuple[RawAttempt, ...],
    ) -> None:
        """Resolve journal orphans from immutable cached HTTP observations."""
        assert self._cache is not None
        for attempt in attempts:
            key = (request.request_sha256, attempt.attempt_number)
            entry = self._journal_entries.get(key)
            if entry is None:
                continue
            if entry.cancelled_before_post:
                raise GenerationContractError(
                    "cancelled-before-POST reservation has a raw HTTP attempt"
                )
            self._missing_response_keys.discard(key)
            actual = _cached_attempt_actual_cost(attempt)
            if entry.charged_usd is None and actual is not None:
                amount = _decimal(actual, "provider-reported billed cost")
                if amount > Decimal(entry.upper_bound_usd):
                    raise GenerationContractError(
                        "provider-reported cost exceeds its journal reservation"
                    )
                self._cache.store_cost_settlement(
                    entry.request_sha256,
                    entry.attempt_number,
                    charged_usd=format(amount, "f"),
                    provider_reported_actual=True,
                )
                entry = CostJournalEntry(
                    request_sha256=entry.request_sha256,
                    attempt_number=entry.attempt_number,
                    upper_bound_usd=entry.upper_bound_usd,
                    charged_usd=format(amount, "f"),
                    provider_reported_actual=True,
                    cancelled_before_post=False,
                    byok_authorization=entry.byok_authorization,
                    byok_authorization_sha256=entry.byok_authorization_sha256,
                )
                self._journal_entries[key] = entry

    def _refresh_recovery_halt_locked(self) -> None:
        if self._halted_reason not in {
            None,
            "billed_attempt_response_missing",
            "orphaned_cost_reservation",
            "provider_billing_unknown",
        }:
            return
        missing_actual = any(
            self._journal_entries[key].provider_reported_actual is True
            for key in self._missing_response_keys
        )
        if missing_actual:
            self._halted_reason = "billed_attempt_response_missing"
            return
        if self._missing_response_keys:
            self._halted_reason = "orphaned_cost_reservation"
            return
        if any(not known for _, known in self._charged.values()):
            self._halted_reason = "provider_billing_unknown"
            return
        self._halted_reason = None

    def _reservation_amount_locked(self, reservation: CostReservation) -> Decimal:
        key = (reservation.request_sha256, reservation.attempt_number)
        amount = self._reservations.get(key)
        if amount is None or amount != Decimal(reservation.upper_bound_usd):
            raise GenerationContractError("runtime cost reservation is unknown")
        return amount

    def _charged_total_locked(self) -> Decimal:
        return sum((amount for amount, _ in self._charged.values()), Decimal(0))


def request_cost_upper_bound(
    request: CanonicalRequest,
    route: RouteLock,
) -> RequestCostUpperBound:
    """Derive a safe per-POST ceiling from exact bytes, max output, and lock prices."""
    if type(request) is not CanonicalRequest:
        raise TypeError("request must be a CanonicalRequest")
    if type(route) is not RouteLock:
        raise TypeError("route must be a RouteLock")
    if request.route_lock_sha256 != route.lock_sha256:
        raise GenerationContractError("request references a different route lock")
    return _request_cost_upper_bound_from_body(
        request,
        route_id=route.route_id,
        expected_prompt_price=_decimal(
            route.input_usd_per_million, "locked input price"
        ),
        expected_completion_price=_decimal(
            route.output_usd_per_million, "locked output price"
        ),
    )


def _cached_attempt_actual_cost(attempt: RawAttempt) -> str | None:
    """Use the final direct-or-generation-stats provider cost observation."""
    if attempt.response is None:
        return None
    if attempt.response.billed_cost_usd is not None:
        return attempt.response.billed_cost_usd
    return next(
        (
            stats.billed_cost_usd
            for stats in reversed(attempt.response.generation_stats_attempts)
            if stats.billed_cost_usd is not None
        ),
        None,
    )


def _request_cost_upper_bound_from_body(
    request: CanonicalRequest,
    *,
    route_id: str,
    expected_prompt_price: Decimal | None,
    expected_completion_price: Decimal | None,
) -> RequestCostUpperBound:
    body = request.body
    token_fields = tuple(
        field for field in ("max_tokens", "max_completion_tokens") if field in body
    )
    if len(token_fields) != 1:
        raise GenerationContractError(
            "request must contain exactly one supported maximum-output field"
        )
    output_tokens = body[token_fields[0]]
    if type(output_tokens) is not int or output_tokens <= 0:
        raise GenerationContractError(
            "request maximum output tokens must be a positive integer"
        )
    provider = body.get("provider")
    if type(provider) is not dict or type(provider.get("max_price")) is not dict:
        raise GenerationContractError("request must contain provider.max_price")
    max_price = provider["max_price"]
    prompt_price = _decimal_json_number(max_price.get("prompt"), "prompt max price")
    completion_price = _decimal_json_number(
        max_price.get("completion"), "completion max price"
    )
    if (
        expected_prompt_price is not None
        and prompt_price < expected_prompt_price
    ) or (
        expected_completion_price is not None
        and completion_price < expected_completion_price
    ):
        raise GenerationContractError(
            "request provider.max_price is below the locked route prices"
        )
    request_bytes = len(request.body_bytes)
    upper_bound = (
        Decimal(request_bytes) * prompt_price
        + Decimal(output_tokens) * completion_price
    ) / _MILLION
    return RequestCostUpperBound(
        request_sha256=request.request_sha256,
        route_id=route_id,
        request_body_bytes=request_bytes,
        maximum_output_tokens=output_tokens,
        upper_bound_usd=format(upper_bound, "f"),
    )


@dataclass(frozen=True, slots=True)
class TokenWorkload:
    """Expected per-request tokens and conservative retry allowance."""

    label: str
    request_count: int
    input_tokens_per_request: int
    output_tokens_per_request: int
    retry_allowance_basis_points: int = 2_000
    conservative_input_tokens_per_request: int | None = None
    conservative_output_tokens_per_request: int | None = None

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label.strip():
            raise GenerationContractError("workload label must be nonempty")
        for label, value in (
            ("request_count", self.request_count),
            ("input_tokens_per_request", self.input_tokens_per_request),
            ("output_tokens_per_request", self.output_tokens_per_request),
        ):
            if type(value) is not int or value < 0:
                raise GenerationContractError(f"{label} must be nonnegative")
        if type(self.retry_allowance_basis_points) is not int or not (
            0 <= self.retry_allowance_basis_points <= 10_000
        ):
            raise GenerationContractError(
                "retry_allowance_basis_points must be between 0 and 10000"
            )
        for label, conservative, expected in (
            (
                "conservative_input_tokens_per_request",
                self.conservative_input_tokens_per_request,
                self.input_tokens_per_request,
            ),
            (
                "conservative_output_tokens_per_request",
                self.conservative_output_tokens_per_request,
                self.output_tokens_per_request,
            ),
        ):
            if conservative is not None and (
                type(conservative) is not int or conservative < expected
            ):
                raise GenerationContractError(
                    f"{label} must be an integer no smaller than expected tokens"
                )

    @property
    def expected_input_tokens(self) -> int:
        """Return total input tokens before the retry allowance."""
        return self.request_count * self.input_tokens_per_request

    @property
    def expected_output_tokens(self) -> int:
        """Return total output tokens before the retry allowance."""
        return self.request_count * self.output_tokens_per_request

    @property
    def conservative_input_tokens(self) -> int:
        per_request = (
            self.input_tokens_per_request
            if self.conservative_input_tokens_per_request is None
            else self.conservative_input_tokens_per_request
        )
        return self.request_count * per_request

    @property
    def conservative_output_tokens(self) -> int:
        per_request = (
            self.output_tokens_per_request
            if self.conservative_output_tokens_per_request is None
            else self.conservative_output_tokens_per_request
        )
        return self.request_count * per_request

    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible workload record."""
        return {
            "conservative_input_tokens_per_request": (
                self.input_tokens_per_request
                if self.conservative_input_tokens_per_request is None
                else self.conservative_input_tokens_per_request
            ),
            "conservative_output_tokens_per_request": (
                self.output_tokens_per_request
                if self.conservative_output_tokens_per_request is None
                else self.conservative_output_tokens_per_request
            ),
            "input_tokens_per_request": self.input_tokens_per_request,
            "label": self.label,
            "output_tokens_per_request": self.output_tokens_per_request,
            "request_count": self.request_count,
            "retry_allowance_basis_points": self.retry_allowance_basis_points,
        }


@dataclass(frozen=True, slots=True)
class RouteWorkload:
    """One locked route paired with its expected token workload."""

    route: RouteLock
    workload: TokenWorkload

    def __post_init__(self) -> None:
        if type(self.route) is not RouteLock:
            raise TypeError("route must be a RouteLock")
        if type(self.workload) is not TokenWorkload:
            raise TypeError("workload must be a TokenWorkload")


@dataclass(frozen=True, slots=True)
class RouteCostEstimate:
    """Expected and conservative cost for one locked route workload."""

    route_id: str
    workload_label: str
    request_count: int
    expected_input_tokens: int
    expected_output_tokens: int
    conservative_input_tokens: int
    conservative_output_tokens: int
    expected_usd: str
    conservative_usd: str

    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible per-route estimate."""
        return {
            "conservative_input_tokens": self.conservative_input_tokens,
            "conservative_output_tokens": self.conservative_output_tokens,
            "conservative_usd": self.conservative_usd,
            "expected_input_tokens": self.expected_input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "expected_usd": self.expected_usd,
            "request_count": self.request_count,
            "route_id": self.route_id,
            "workload_label": self.workload_label,
        }


@dataclass(frozen=True, slots=True)
class DirectBatchQuote:
    """A dated direct-OpenAI Batch price used only for comparison."""

    model_snapshot: str
    input_usd_per_million: str
    output_usd_per_million: str
    source_url: str
    quoted_on: str

    def __post_init__(self) -> None:
        _decimal(self.input_usd_per_million, "batch input price")
        _decimal(self.output_usd_per_million, "batch output price")
        if type(self.model_snapshot) is not str or not self.model_snapshot:
            raise GenerationContractError("model_snapshot must be nonempty")
        if not self.source_url.startswith("https://"):
            raise GenerationContractError("source_url must use HTTPS")


@dataclass(frozen=True, slots=True)
class OpenAIBatchComparison:
    """Projected direct Batch cost for the complete preflight token volume."""

    model_snapshot: str
    expected_usd: str
    conservative_usd: str
    source_url: str
    quoted_on: str

    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible direct Batch comparison."""
        return {
            "conservative_usd": self.conservative_usd,
            "expected_usd": self.expected_usd,
            "model_snapshot": self.model_snapshot,
            "quoted_on": self.quoted_on,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class CostPreflight:
    """Aggregate cost decision made before any billable request is sent."""

    route_estimates: tuple[RouteCostEstimate, ...]
    expected_usd: str
    conservative_usd: str
    hard_cap_usd: str
    permitted: bool
    openai_batch_comparisons: tuple[OpenAIBatchComparison, ...]

    def as_record(self) -> JsonObject:
        """Return the complete serializable cost gate and comparison evidence."""
        return {
            "conservative_usd": self.conservative_usd,
            "expected_usd": self.expected_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "openai_batch_comparisons": [
                comparison.as_record()
                for comparison in self.openai_batch_comparisons
            ],
            "permitted": self.permitted,
            "route_estimates": [
                estimate.as_record() for estimate in self.route_estimates
            ],
        }


def _decimal(value: str, label: str) -> Decimal:
    if type(value) is not str:
        raise TypeError(f"{label} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise GenerationContractError(f"{label} must be decimal") from error
    if not result.is_finite() or result < 0:
        raise GenerationContractError(f"{label} must be finite and nonnegative")
    return result


def _decimal_json_number(value: object, label: str) -> Decimal:
    if type(value) not in (int, float, str):
        raise GenerationContractError(f"{label} must be a decimal JSON value")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise GenerationContractError(f"{label} must be decimal") from error
    if not result.is_finite() or result < 0:
        raise GenerationContractError(
            f"{label} must be finite and nonnegative"
        )
    return result


OPENAI_BATCH_QUOTES = (
    DirectBatchQuote(
        model_snapshot="gpt-5.4-mini-2026-03-17",
        input_usd_per_million="0.375",
        output_usd_per_million="2.25",
        source_url="https://developers.openai.com/api/docs/guides/batch",
        quoted_on="2026-07-18",
    ),
    DirectBatchQuote(
        model_snapshot="gpt-5.4-2026-03-05",
        input_usd_per_million="1.25",
        output_usd_per_million="7.50",
        source_url="https://developers.openai.com/api/docs/guides/batch",
        quoted_on="2026-07-18",
    ),
)


def estimate_route_cost(route_workload: RouteWorkload) -> RouteCostEstimate:
    """Calculate an exact decimal estimate for one route and token workload."""
    if type(route_workload) is not RouteWorkload:
        raise TypeError("route_workload must be a RouteWorkload")
    route, workload = route_workload.route, route_workload.workload
    expected = _token_cost(
        workload.expected_input_tokens,
        workload.expected_output_tokens,
        route.input_usd_per_million,
        route.output_usd_per_million,
    )
    retry_multiplier = Decimal(1) + (
        Decimal(workload.retry_allowance_basis_points) / Decimal(10_000)
    )
    conservative = _token_cost(
        workload.conservative_input_tokens,
        workload.conservative_output_tokens,
        route.input_usd_per_million,
        route.output_usd_per_million,
    )
    return RouteCostEstimate(
        route_id=route.route_id,
        workload_label=workload.label,
        request_count=workload.request_count,
        expected_input_tokens=workload.expected_input_tokens,
        expected_output_tokens=workload.expected_output_tokens,
        conservative_input_tokens=workload.conservative_input_tokens,
        conservative_output_tokens=workload.conservative_output_tokens,
        expected_usd=_display_usd(expected),
        conservative_usd=_display_usd(conservative * retry_multiplier),
    )


def build_cost_preflight(
    route_workloads: tuple[RouteWorkload, ...],
    *,
    hard_cap_usd: str = PHASE1_HARD_CAP_USD,
    batch_quotes: tuple[DirectBatchQuote, ...] = OPENAI_BATCH_QUOTES,
) -> CostPreflight:
    """Aggregate route estimates, compare Batch prices, and apply the hard cap."""
    if type(route_workloads) is not tuple or any(
        type(item) is not RouteWorkload for item in route_workloads
    ):
        raise TypeError("route_workloads must be a tuple of RouteWorkload values")
    if not route_workloads:
        raise GenerationContractError("cost preflight requires at least one workload")
    cap = _decimal(hard_cap_usd, "hard cap")
    estimates = tuple(estimate_route_cost(item) for item in route_workloads)
    expected = sum((Decimal(item.expected_usd) for item in estimates), Decimal(0))
    conservative = sum(
        (Decimal(item.conservative_usd) for item in estimates),
        Decimal(0),
    )
    total_input = sum(
        item.workload.expected_input_tokens for item in route_workloads
    )
    total_output = sum(
        item.workload.expected_output_tokens for item in route_workloads
    )
    conservative_input = sum(
        item.workload.conservative_input_tokens for item in route_workloads
    )
    conservative_output = sum(
        item.workload.conservative_output_tokens for item in route_workloads
    )
    largest_retry_multiplier = max(
        Decimal(1)
        + Decimal(item.workload.retry_allowance_basis_points) / Decimal(10_000)
        for item in route_workloads
    )
    comparisons = tuple(
        _batch_comparison(
            quote,
            input_tokens=total_input,
            output_tokens=total_output,
            conservative_input_tokens=conservative_input,
            conservative_output_tokens=conservative_output,
            retry_multiplier=largest_retry_multiplier,
        )
        for quote in batch_quotes
    )
    return CostPreflight(
        route_estimates=estimates,
        expected_usd=_display_usd(expected),
        conservative_usd=_display_usd(conservative),
        hard_cap_usd=_display_usd(cap),
        permitted=conservative <= cap,
        openai_batch_comparisons=comparisons,
    )


def enforce_cost_cap(preflight: CostPreflight) -> None:
    """Stop billable work unless the conservative preflight is within its cap."""
    if type(preflight) is not CostPreflight:
        raise TypeError("preflight must be a CostPreflight")
    if not preflight.permitted:
        raise CostCapExceeded(
            f"conservative estimate ${preflight.conservative_usd} exceeds "
            f"the ${preflight.hard_cap_usd} hard cap"
        )


def _batch_comparison(
    quote: DirectBatchQuote,
    *,
    input_tokens: int,
    output_tokens: int,
    conservative_input_tokens: int,
    conservative_output_tokens: int,
    retry_multiplier: Decimal,
) -> OpenAIBatchComparison:
    expected = _token_cost(
        input_tokens,
        output_tokens,
        quote.input_usd_per_million,
        quote.output_usd_per_million,
    )
    conservative = _token_cost(
        conservative_input_tokens,
        conservative_output_tokens,
        quote.input_usd_per_million,
        quote.output_usd_per_million,
    )
    return OpenAIBatchComparison(
        model_snapshot=quote.model_snapshot,
        expected_usd=_display_usd(expected),
        conservative_usd=_display_usd(conservative * retry_multiplier),
        source_url=quote.source_url,
        quoted_on=quote.quoted_on,
    )


def _token_cost(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: str,
    output_usd_per_million: str,
) -> Decimal:
    return (
        Decimal(input_tokens) * _decimal(input_usd_per_million, "input price")
        + Decimal(output_tokens) * _decimal(output_usd_per_million, "output price")
    ) / _MILLION


def _display_usd(value: Decimal) -> str:
    return format(value.quantize(_DISPLAY_QUANTUM, rounding=ROUND_CEILING), "f")
