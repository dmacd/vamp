"""Append-only, content-addressed raw cache for external generation calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import shutil
import tempfile

from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    GenerationContractError,
    RawAttempt,
    RawGenerationStatsAttempt,
    RawGenerationStatsResponse,
    RawHttpResponse,
    ResponseProvenance,
    RouteLock,
    TokenUsage,
)
from apm.data.text.tinyworlds_v2.byok_contract import (
    canonical_byok_authorization,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_loads,
    json_sha256,
    require_exact_fields,
    require_json_object,
)


_ATTEMPT_FORMAT = "apm.tinyworlds-v2.raw-attempt"
_ATTEMPT_SCHEMA_VERSION = 4
_STATS_ATTEMPT_FORMAT = "apm.tinyworlds-v2.raw-generation-stats-attempt"
_STATS_ATTEMPT_SCHEMA_VERSION = 1
_REQUEST_FILES = frozenset(
    {"attempts", "request-body.json", "request.json", "route-lock.json"}
)
_ATTEMPT_FILES = frozenset({"generation-stats", "metadata.json", "response.body"})
_STATS_ATTEMPT_FILES = frozenset({"metadata.json", "response.body"})
_COST_JOURNAL_DIRECTORY = "runtime-cost-journal"
_COST_RESERVATION_FORMAT = "apm.tinyworlds-v2.cost-reservation"
_COST_SETTLEMENT_FORMAT = "apm.tinyworlds-v2.cost-settlement"
_COST_CANCELLATION_FORMAT = "apm.tinyworlds-v2.cost-cancellation"
_ROUTE_LOCK_FORMAT = "apm.tinyworlds-v2.cached-route-lock"


class GenerationCacheError(ValueError):
    """A cached generation record is malformed or fails integrity checks."""


@dataclass(frozen=True, slots=True)
class CostJournalEntry:
    """One durable pre-POST reservation and its optional settlement."""

    request_sha256: str
    attempt_number: int
    upper_bound_usd: str
    charged_usd: str | None
    provider_reported_actual: bool | None
    cancelled_before_post: bool
    byok_authorization: JsonObject
    byok_authorization_sha256: str


@dataclass(frozen=True, slots=True)
class ImmutableRawCache:
    """Content-addressed request directories with immutable numbered attempts."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("cache root must be a Path")

    def prepare_request(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
    ) -> Path:
        """Persist one request together with its complete historical route lock."""
        if type(request) is not CanonicalRequest:
            raise TypeError("request must be a CanonicalRequest")
        if type(route_lock) is not RouteLock:
            raise TypeError("route_lock must be a RouteLock")
        if request.route_lock_sha256 != route_lock.lock_sha256:
            raise GenerationCacheError("request and cached route lock differ")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._request_directory(request.request_sha256)
        if target.exists() or target.is_symlink():
            if (
                self.load_request(request.request_sha256) != request
                or self.load_route_lock(request.request_sha256).lock_sha256
                != route_lock.lock_sha256
            ):
                raise GenerationCacheError(
                    "request/route cache identity collision or modified evidence"
                )
            return target
        temporary = Path(
            tempfile.mkdtemp(prefix=".request-tmp-", dir=self.root)
        )
        try:
            (temporary / "request.json").write_bytes(
                canonical_json_bytes(request.as_record())
            )
            (temporary / "request-body.json").write_bytes(request.body_bytes)
            (temporary / "route-lock.json").write_bytes(
                canonical_json_bytes(_route_lock_record(route_lock))
            )
            (temporary / "attempts").mkdir()
            try:
                os.rename(temporary, target)
            except FileExistsError:
                # Concurrent workers may prepare the same content-addressed
                # request. The winner's immutable evidence must be equivalent;
                # the volatile catalog digest may differ across fresh checks.
                shutil.rmtree(temporary)
                if (
                    self.load_request(request.request_sha256) != request
                    or self.load_route_lock(request.request_sha256).lock_sha256
                    != route_lock.lock_sha256
                ):
                    raise GenerationCacheError(
                        "concurrent request preparation produced different evidence"
                    )
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target

    def store_attempt(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
        attempt: RawAttempt,
    ) -> Path:
        """Atomically append one numbered response or transport-error attempt."""
        if type(attempt) is not RawAttempt:
            raise TypeError("attempt must be a RawAttempt")
        if attempt.request_sha256 != request.request_sha256:
            raise GenerationCacheError("attempt belongs to a different request")
        attempts_directory = self.prepare_request(request, route_lock) / "attempts"
        journal_directory = (
            self.root
            / _COST_JOURNAL_DIRECTORY
            / _cost_journal_name(request.request_sha256, attempt.attempt_number)
        )
        if journal_directory.exists() and _load_cost_journal_entry(
            journal_directory
        ).cancelled_before_post:
            raise GenerationCacheError(
                "cancelled-before-POST reservation cannot acquire a raw attempt"
            )
        target = attempts_directory / f"{attempt.attempt_number:06d}"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"raw attempt already exists: {target}")
        temporary = Path(
            tempfile.mkdtemp(prefix=".attempt-tmp-", dir=attempts_directory)
        )
        try:
            core = _attempt_core_record(attempt)
            metadata = {**core, "attempt_sha256": json_sha256(core)}
            (temporary / "metadata.json").write_bytes(
                canonical_json_bytes(metadata)
            )
            if attempt.response is not None:
                (temporary / "response.body").write_bytes(attempt.response.body)
            stats_directory = temporary / "generation-stats"
            stats_directory.mkdir()
            for stats_attempt in (
                attempt.response.generation_stats_attempts
                if attempt.response is not None
                else ()
            ):
                _store_generation_stats_attempt(stats_directory, stats_attempt)
            os.rename(temporary, target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target

    def store_generation_stats_attempt(
        self,
        request: CanonicalRequest,
        completion_attempt_number: int,
        attempt: RawGenerationStatsAttempt,
    ) -> Path:
        """Append one stats lookup without mutating its cached completion response."""
        if type(completion_attempt_number) is not int or completion_attempt_number < 1:
            raise GenerationCacheError(
                "completion attempt number must be positive"
            )
        if type(attempt) is not RawGenerationStatsAttempt:
            raise TypeError("attempt must be a RawGenerationStatsAttempt")
        if self.load_request(request.request_sha256) != request:
            raise GenerationCacheError("stats request differs from cached request")
        completion_directory = (
            self._request_directory(request.request_sha256)
            / "attempts"
            / f"{completion_attempt_number:06d}"
        )
        _require_regular_directory(completion_directory, "completion attempt")
        stats_directory = completion_directory / "generation-stats"
        _require_regular_directory(stats_directory, "generation-stats directory")
        return _store_generation_stats_attempt(stats_directory, attempt)

    def load_request(self, request_sha256: str) -> CanonicalRequest:
        """Strictly load and re-hash one cached request and its exact body."""
        directory = self._request_directory(request_sha256)
        _require_regular_directory(directory, "request cache entry")
        actual_files = frozenset(path.name for path in directory.iterdir())
        if actual_files != _REQUEST_FILES:
            raise GenerationCacheError(
                "request entry contains missing or unknown files"
            )
        _require_regular_directory(directory / "attempts", "attempts directory")
        metadata_path = directory / "request.json"
        body_path = directory / "request-body.json"
        _require_regular_file(metadata_path, "request metadata")
        _require_regular_file(body_path, "request body")
        metadata = require_json_object(
            canonical_json_loads(
                metadata_path.read_bytes(),
                label="cached request metadata",
            ),
            label="cached request metadata",
        )
        require_exact_fields(
            metadata,
            (
                "body_sha256",
                "endpoint",
                "format",
                "method",
                "request_sha256",
                "route_lock_sha256",
                "schema_version",
                "transport_protocol",
            ),
            label="cached request metadata",
        )
        if metadata["format"] != "apm.tinyworlds-v2.generation-request":
            raise GenerationCacheError("unsupported request cache format")
        if metadata["schema_version"] != 2:
            raise GenerationCacheError("unsupported request cache schema")
        body = body_path.read_bytes()
        if bytes_sha256(body) != _string(metadata["body_sha256"], "body_sha256"):
            raise GenerationCacheError("cached request body digest mismatch")
        try:
            request = CanonicalRequest(
                request_sha256=_string(
                    metadata["request_sha256"], "request_sha256"
                ),
                route_lock_sha256=_string(
                    metadata["route_lock_sha256"], "route_lock_sha256"
                ),
                method=_string(metadata["method"], "method"),
                endpoint=_string(metadata["endpoint"], "endpoint"),
                transport_protocol=_string(
                    metadata["transport_protocol"],
                    "transport_protocol",
                ),
                body_json=body.decode("utf-8"),
            )
        except (GenerationContractError, UnicodeDecodeError) as error:
            raise GenerationCacheError(f"invalid cached request: {error}") from error
        if request.request_sha256 != request_sha256:
            raise GenerationCacheError("request directory name does not match identity")
        if request.as_record() != metadata:
            raise GenerationCacheError("cached request metadata mismatch")
        return request

    def load_route_lock(self, request_sha256: str) -> RouteLock:
        """Load the complete immutable route semantics used by one request."""
        directory = self._request_directory(request_sha256)
        _require_regular_directory(directory, "request cache entry")
        path = directory / "route-lock.json"
        _require_regular_file(path, "cached route lock")
        record = require_json_object(
            canonical_json_loads(path.read_bytes(), label="cached route lock"),
            label="cached route lock",
        )
        require_exact_fields(
            record,
            (
                "canonical_model",
                "catalog_sha256",
                "format",
                "input_usd_per_million",
                "lock_sha256",
                "output_usd_per_million",
                "provider_slug",
                "quantization",
                "requested_model",
                "returned_provider",
                "route_id",
                "schema_version",
            ),
            label="cached route lock",
        )
        if record["format"] != _ROUTE_LOCK_FORMAT or record["schema_version"] != 1:
            raise GenerationCacheError("unsupported cached route-lock contract")
        try:
            route = RouteLock(
                route_id=_string(record["route_id"], "route_id"),
                catalog_sha256=_string(record["catalog_sha256"], "catalog_sha256"),
                requested_model=_string(record["requested_model"], "requested_model"),
                canonical_model=_string(record["canonical_model"], "canonical_model"),
                provider_slug=_string(record["provider_slug"], "provider_slug"),
                returned_provider=_string(
                    record["returned_provider"], "returned_provider"
                ),
                quantization=_string(record["quantization"], "quantization"),
                input_usd_per_million=_string(
                    record["input_usd_per_million"], "input_usd_per_million"
                ),
                output_usd_per_million=_string(
                    record["output_usd_per_million"], "output_usd_per_million"
                ),
            )
        except GenerationContractError as error:
            raise GenerationCacheError(f"invalid cached route lock: {error}") from error
        if record["lock_sha256"] != route.lock_sha256:
            raise GenerationCacheError("cached route-lock digest mismatch")
        request = self.load_request(request_sha256)
        if request.route_lock_sha256 != route.lock_sha256:
            raise GenerationCacheError("cached request and route-lock evidence differ")
        return route

    def load_attempts(self, request: CanonicalRequest) -> tuple[RawAttempt, ...]:
        """Strictly load every immutable attempt in numeric order."""
        if self.load_request(request.request_sha256) != request:
            raise GenerationCacheError("attempt request differs from cached request")
        self.load_route_lock(request.request_sha256)
        attempts_directory = self._request_directory(
            request.request_sha256
        ) / "attempts"
        entries = tuple(sorted(attempts_directory.iterdir(), key=lambda path: path.name))
        expected_names = tuple(
            f"{number:06d}" for number in range(1, len(entries) + 1)
        )
        if tuple(path.name for path in entries) != expected_names:
            raise GenerationCacheError(
                "cached attempts must be contiguous and canonically numbered"
            )
        return tuple(
            _load_attempt(path, request.request_sha256) for path in entries
        )

    def load_all_requests(self) -> tuple[CanonicalRequest, ...]:
        """Strictly load every request in content-identity order."""
        if not self.root.exists():
            return ()
        _require_regular_directory(self.root, "raw cache root")
        entries = tuple(
            path
            for path in sorted(self.root.iterdir(), key=lambda path: path.name)
            if path.name != _COST_JOURNAL_DIRECTORY
        )
        return tuple(self.load_request(path.name) for path in entries)

    def store_cost_reservation(
        self,
        request_sha256: str,
        attempt_number: int,
        upper_bound_usd: str,
        byok_authorization: JsonObject,
    ) -> None:
        """Durably persist a reservation before its completion POST begins."""
        authorization = canonical_byok_authorization(byok_authorization)
        entry = CostJournalEntry(
            request_sha256,
            attempt_number,
            _cost_string(upper_bound_usd, "upper_bound_usd"),
            None,
            None,
            False,
            authorization,
            json_sha256(authorization),
        )
        journal = self.root / _COST_JOURNAL_DIRECTORY
        journal.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.root)
        target = journal / _cost_journal_name(entry.request_sha256, attempt_number)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"cost reservation already exists: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=".reservation-tmp-", dir=journal))
        try:
            reservation: JsonObject = {
                "attempt_number": entry.attempt_number,
                "byok_authorization": entry.byok_authorization,
                "byok_authorization_sha256": entry.byok_authorization_sha256,
                "format": _COST_RESERVATION_FORMAT,
                "request_sha256": entry.request_sha256,
                "schema_version": 2,
                "upper_bound_usd": entry.upper_bound_usd,
            }
            _write_fsynced(
                temporary / "reservation.json",
                canonical_json_bytes(reservation),
            )
            _fsync_directory(temporary)
            os.rename(temporary, target)
            _fsync_directory(journal)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def store_cost_settlement(
        self,
        request_sha256: str,
        attempt_number: int,
        *,
        charged_usd: str,
        provider_reported_actual: bool,
    ) -> None:
        """Durably settle one reservation without modifying its reservation file."""
        target = (
            self.root
            / _COST_JOURNAL_DIRECTORY
            / _cost_journal_name(request_sha256, attempt_number)
        )
        entry = _load_cost_journal_entry(target)
        if entry.charged_usd is not None or entry.cancelled_before_post:
            raise FileExistsError(f"cost reservation is already terminal: {target}")
        charged = _cost_string(charged_usd, "charged_usd")
        if type(provider_reported_actual) is not bool:
            raise TypeError("provider_reported_actual must be bool")
        settlement: JsonObject = {
            "charged_usd": charged,
            "format": _COST_SETTLEMENT_FORMAT,
            "provider_reported_actual": provider_reported_actual,
            "schema_version": 1,
        }
        _atomic_write_once(
            target / "settlement.json",
            canonical_json_bytes(settlement),
        )

    def store_cost_cancellation(
        self,
        request_sha256: str,
        attempt_number: int,
    ) -> None:
        """Record that an authorized reservation was cancelled before POST."""
        target = (
            self.root
            / _COST_JOURNAL_DIRECTORY
            / _cost_journal_name(request_sha256, attempt_number)
        )
        entry = _load_cost_journal_entry(target)
        if entry.charged_usd is not None or entry.cancelled_before_post:
            raise FileExistsError(f"cost reservation is already terminal: {target}")
        attempt_directory = (
            self._request_directory(request_sha256)
            / "attempts"
            / f"{attempt_number:06d}"
        )
        if attempt_directory.exists() or attempt_directory.is_symlink():
            raise GenerationCacheError(
                "raw completion attempt cannot be cancelled before POST"
            )
        cancellation: JsonObject = {
            "format": _COST_CANCELLATION_FORMAT,
            "schema_version": 1,
            "state": "cancelled_before_post",
        }
        _atomic_write_once(
            target / "cancellation.json",
            canonical_json_bytes(cancellation),
        )

    def load_cost_journal(self) -> tuple[CostJournalEntry, ...]:
        """Strictly load every write-ahead reservation in identity order."""
        journal = self.root / _COST_JOURNAL_DIRECTORY
        if not journal.exists():
            return ()
        _require_regular_directory(journal, "runtime cost journal")
        return tuple(
            _load_cost_journal_entry(path)
            for path in sorted(journal.iterdir(), key=lambda path: path.name)
        )

    def find_success(self, request: CanonicalRequest) -> RawHttpResponse | None:
        """Return the first cached 2xx response, if generation already succeeded."""
        return next(
            (
                attempt.response
                for attempt in self.load_attempts(request)
                if attempt.response is not None
                and 200 <= attempt.response.status_code < 300
            ),
            None,
        )

    def _request_directory(self, request_sha256: str) -> Path:
        if (
            type(request_sha256) is not str
            or len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
        ):
            raise GenerationCacheError("request identity must be a SHA-256 digest")
        return self.root / request_sha256


def _attempt_core_record(attempt: RawAttempt) -> JsonObject:
    response = attempt.response
    response_record: JsonValue = None
    if response is not None:
        response_record = {
            "billed_cost_usd": response.billed_cost_usd,
            "body_sha256": bytes_sha256(response.body),
            "body_size_bytes": len(response.body),
            "headers": [[name, value] for name, value in response.headers],
            "provenance": (
                None if response.provenance is None else response.provenance.as_record()
            ),
            "status_code": response.status_code,
            "usage": None if response.usage is None else response.usage.as_record(),
        }
    return {
        "attempt_number": attempt.attempt_number,
        "format": _ATTEMPT_FORMAT,
        "observed_at_utc": attempt.observed_at_utc,
        "request_sha256": attempt.request_sha256,
        "response": response_record,
        "schema_version": _ATTEMPT_SCHEMA_VERSION,
        "submission_catalog_sha256": attempt.submission_catalog_sha256,
        "transport_error_type": attempt.transport_error_type,
    }


def _route_lock_record(route: RouteLock) -> JsonObject:
    return {
        **route.as_record(),
        "format": _ROUTE_LOCK_FORMAT,
        "lock_sha256": route.lock_sha256,
        "schema_version": 1,
    }


def _load_attempt(directory: Path, request_sha256: str) -> RawAttempt:
    _require_regular_directory(directory, "raw attempt")
    actual_files = frozenset(path.name for path in directory.iterdir())
    if "metadata.json" not in actual_files or not actual_files <= _ATTEMPT_FILES:
        raise GenerationCacheError("attempt contains missing or unknown files")
    metadata_path = directory / "metadata.json"
    _require_regular_file(metadata_path, "attempt metadata")
    metadata = require_json_object(
        canonical_json_loads(metadata_path.read_bytes(), label="attempt metadata"),
        label="attempt metadata",
    )
    require_exact_fields(
        metadata,
        (
            "attempt_number",
            "attempt_sha256",
            "format",
            "observed_at_utc",
            "request_sha256",
            "response",
            "schema_version",
            "submission_catalog_sha256",
            "transport_error_type",
        ),
        label="attempt metadata",
    )
    supplied_digest = _string(metadata["attempt_sha256"], "attempt_sha256")
    core = {key: value for key, value in metadata.items() if key != "attempt_sha256"}
    if supplied_digest != json_sha256(core):
        raise GenerationCacheError("attempt metadata digest mismatch")
    if metadata["format"] != _ATTEMPT_FORMAT:
        raise GenerationCacheError("unsupported raw-attempt format")
    if metadata["schema_version"] != _ATTEMPT_SCHEMA_VERSION:
        raise GenerationCacheError("unsupported raw-attempt schema")
    stats_directory = directory / "generation-stats"
    stats_attempts = (
        _load_generation_stats_attempts(stats_directory)
        if stats_directory.exists() or stats_directory.is_symlink()
        else ()
    )
    response_value = metadata["response"]
    if response_value is None and stats_attempts:
        raise GenerationCacheError(
            "transport-error attempt cannot contain generation-stats attempts"
        )
    response = (
        None
        if response_value is None
        else _decode_response(
            require_json_object(response_value, label="cached HTTP response"),
            directory / "response.body",
            stats_attempts,
        )
    )
    expected_files = {"metadata.json"}
    if stats_directory.exists() or stats_directory.is_symlink():
        expected_files.add("generation-stats")
    if response is not None:
        expected_files.add("response.body")
    if actual_files != frozenset(expected_files):
        raise GenerationCacheError(
            "attempt response metadata/body presence differs"
        )
    try:
        attempt = RawAttempt(
            request_sha256=_string(
                metadata["request_sha256"], "request_sha256"
            ),
            attempt_number=_integer(metadata["attempt_number"], "attempt_number"),
            observed_at_utc=_string(
                metadata["observed_at_utc"], "observed_at_utc"
            ),
            submission_catalog_sha256=_string(
                metadata["submission_catalog_sha256"],
                "submission_catalog_sha256",
            ),
            response=response,
            transport_error_type=_optional_string(
                metadata["transport_error_type"], "transport_error_type"
            ),
        )
    except GenerationContractError as error:
        raise GenerationCacheError(f"invalid raw attempt: {error}") from error
    if attempt.request_sha256 != request_sha256:
        raise GenerationCacheError("attempt references a different request")
    if directory.name != f"{attempt.attempt_number:06d}":
        raise GenerationCacheError("attempt number does not match its directory")
    return attempt


def _decode_response(
    record: JsonObject,
    body_path: Path,
    generation_stats_attempts: tuple[RawGenerationStatsAttempt, ...],
) -> RawHttpResponse:
    require_exact_fields(
        record,
        (
            "billed_cost_usd",
            "body_sha256",
            "body_size_bytes",
            "headers",
            "provenance",
            "status_code",
            "usage",
        ),
        label="cached HTTP response",
    )
    _require_regular_file(body_path, "cached response body")
    body = body_path.read_bytes()
    if len(body) != _integer(record["body_size_bytes"], "body_size_bytes"):
        raise GenerationCacheError("cached response body size mismatch")
    if bytes_sha256(body) != _string(record["body_sha256"], "body_sha256"):
        raise GenerationCacheError("cached response body digest mismatch")
    headers_value = record["headers"]
    if type(headers_value) is not list:
        raise GenerationCacheError("cached response headers must be a JSON array")
    headers = tuple(_header_pair(value) for value in headers_value)
    provenance_value, usage_value = record["provenance"], record["usage"]
    direct_billed_cost = _optional_string(
        record["billed_cost_usd"], "billed_cost_usd"
    )
    stats_costs = tuple(
        attempt.billed_cost_usd
        for attempt in generation_stats_attempts
        if attempt.billed_cost_usd is not None
    )
    return RawHttpResponse(
        status_code=_integer(record["status_code"], "status_code"),
        headers=headers,
        body=body,
        provenance=(
            None
            if provenance_value is None
            else _decode_provenance(
                require_json_object(provenance_value, label="response provenance")
            )
        ),
        usage=(
            None
            if usage_value is None
            else _decode_usage(
                require_json_object(usage_value, label="response usage")
            )
        ),
        billed_cost_usd=(
            direct_billed_cost
            if direct_billed_cost is not None
            else stats_costs[-1]
            if stats_costs
            else None
        ),
        generation_stats_attempts=generation_stats_attempts,
    )


def _raw_http_record(response: RawGenerationStatsResponse) -> JsonObject:
    return {
        "body_sha256": bytes_sha256(response.body),
        "body_size_bytes": len(response.body),
        "headers": [[name, value] for name, value in response.headers],
        "status_code": response.status_code,
    }


def _store_generation_stats_attempt(
    directory: Path,
    attempt: RawGenerationStatsAttempt,
) -> Path:
    """Atomically append one numbered stats lookup beneath a completion."""
    _require_regular_directory(directory, "generation-stats directory")
    entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    expected_number = len(entries) + 1
    if tuple(path.name for path in entries) != tuple(
        f"{number:06d}" for number in range(1, expected_number)
    ):
        raise GenerationCacheError(
            "generation-stats attempts must be contiguous and canonically numbered"
        )
    if attempt.attempt_number != expected_number:
        raise GenerationCacheError(
            "generation-stats attempt number is not the next append position"
        )
    target = directory / f"{attempt.attempt_number:06d}"
    temporary = Path(tempfile.mkdtemp(prefix=".stats-tmp-", dir=directory))
    try:
        response_record: JsonValue = None
        if attempt.response is not None:
            response_record = _raw_http_record(attempt.response)
        core: JsonObject = {
            "attempt_number": attempt.attempt_number,
            "billed_cost_usd": attempt.billed_cost_usd,
            "format": _STATS_ATTEMPT_FORMAT,
            "observed_at_utc": attempt.observed_at_utc,
            "response": response_record,
            "schema_version": _STATS_ATTEMPT_SCHEMA_VERSION,
            "transport_error_type": attempt.transport_error_type,
        }
        (temporary / "metadata.json").write_bytes(
            canonical_json_bytes({**core, "attempt_sha256": json_sha256(core)})
        )
        if attempt.response is not None:
            (temporary / "response.body").write_bytes(attempt.response.body)
        os.rename(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _load_generation_stats_attempts(
    directory: Path,
) -> tuple[RawGenerationStatsAttempt, ...]:
    _require_regular_directory(directory, "generation-stats directory")
    entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    if tuple(path.name for path in entries) != tuple(
        f"{number:06d}" for number in range(1, len(entries) + 1)
    ):
        raise GenerationCacheError(
            "generation-stats attempts must be contiguous and canonically numbered"
        )
    return tuple(_load_generation_stats_attempt(path) for path in entries)


def _load_generation_stats_attempt(directory: Path) -> RawGenerationStatsAttempt:
    _require_regular_directory(directory, "generation-stats attempt")
    actual_files = frozenset(path.name for path in directory.iterdir())
    if "metadata.json" not in actual_files or not actual_files <= _STATS_ATTEMPT_FILES:
        raise GenerationCacheError(
            "generation-stats attempt contains missing or unknown files"
        )
    metadata_path = directory / "metadata.json"
    _require_regular_file(metadata_path, "generation-stats attempt metadata")
    metadata = require_json_object(
        canonical_json_loads(
            metadata_path.read_bytes(),
            label="generation-stats attempt metadata",
        ),
        label="generation-stats attempt metadata",
    )
    require_exact_fields(
        metadata,
        (
            "attempt_number",
            "attempt_sha256",
            "billed_cost_usd",
            "format",
            "observed_at_utc",
            "response",
            "schema_version",
            "transport_error_type",
        ),
        label="generation-stats attempt metadata",
    )
    core = {key: value for key, value in metadata.items() if key != "attempt_sha256"}
    if _string(metadata["attempt_sha256"], "attempt_sha256") != json_sha256(core):
        raise GenerationCacheError(
            "generation-stats attempt metadata digest mismatch"
        )
    if metadata["format"] != _STATS_ATTEMPT_FORMAT:
        raise GenerationCacheError("unsupported generation-stats attempt format")
    if metadata["schema_version"] != _STATS_ATTEMPT_SCHEMA_VERSION:
        raise GenerationCacheError("unsupported generation-stats attempt schema")
    response_value = metadata["response"]
    response = (
        None
        if response_value is None
        else _decode_generation_stats(
            require_json_object(
                response_value,
                label="generation-stats HTTP response",
            ),
            directory / "response.body",
        )
    )
    expected_files = {"metadata.json"}
    if response is not None:
        expected_files.add("response.body")
    if actual_files != frozenset(expected_files):
        raise GenerationCacheError(
            "generation-stats response metadata/body presence differs"
        )
    try:
        attempt = RawGenerationStatsAttempt(
            attempt_number=_integer(
                metadata["attempt_number"], "attempt_number"
            ),
            observed_at_utc=_string(
                metadata["observed_at_utc"], "observed_at_utc"
            ),
            response=response,
            transport_error_type=_optional_string(
                metadata["transport_error_type"], "transport_error_type"
            ),
            billed_cost_usd=_optional_string(
                metadata["billed_cost_usd"], "billed_cost_usd"
            ),
        )
    except GenerationContractError as error:
        raise GenerationCacheError(
            f"invalid generation-stats attempt: {error}"
        ) from error
    if directory.name != f"{attempt.attempt_number:06d}":
        raise GenerationCacheError(
            "generation-stats attempt number does not match its directory"
        )
    return attempt


def _decode_generation_stats(
    record: JsonObject,
    body_path: Path,
) -> RawGenerationStatsResponse:
    require_exact_fields(
        record,
        ("body_sha256", "body_size_bytes", "headers", "status_code"),
        label="generation-stats HTTP response",
    )
    _require_regular_file(body_path, "cached generation-stats response body")
    body = body_path.read_bytes()
    if len(body) != _integer(record["body_size_bytes"], "body_size_bytes"):
        raise GenerationCacheError(
            "cached generation-stats response body size mismatch"
        )
    if bytes_sha256(body) != _string(record["body_sha256"], "body_sha256"):
        raise GenerationCacheError(
            "cached generation-stats response body digest mismatch"
        )
    headers_value = record["headers"]
    if type(headers_value) is not list:
        raise GenerationCacheError(
            "cached generation-stats response headers must be a JSON array"
        )
    return RawGenerationStatsResponse(
        status_code=_integer(record["status_code"], "status_code"),
        headers=tuple(_header_pair(value) for value in headers_value),
        body=body,
    )


def _decode_provenance(record: JsonObject) -> ResponseProvenance:
    require_exact_fields(
        record,
        (
            "generation_id",
            "requested_model",
            "returned_model",
            "returned_provider",
        ),
        label="response provenance",
    )
    return ResponseProvenance(
        generation_id=_string(record["generation_id"], "generation_id"),
        requested_model=_string(record["requested_model"], "requested_model"),
        returned_model=_string(record["returned_model"], "returned_model"),
        returned_provider=_string(
            record["returned_provider"], "returned_provider"
        ),
    )


def _decode_usage(record: JsonObject) -> TokenUsage:
    require_exact_fields(
        record,
        (
            "cached_input_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ),
        label="response usage",
    )
    return TokenUsage(
        input_tokens=_integer(record["input_tokens"], "input_tokens"),
        output_tokens=_integer(record["output_tokens"], "output_tokens"),
        total_tokens=_integer(record["total_tokens"], "total_tokens"),
        cached_input_tokens=_integer(
            record["cached_input_tokens"], "cached_input_tokens"
        ),
    )


def _header_pair(value: JsonValue) -> tuple[str, str]:
    if type(value) is not list or len(value) != 2:
        raise GenerationCacheError("cached header must contain two strings")
    return _string(value[0], "header name"), _string(value[1], "header value")


def _string(value: JsonValue, label: str) -> str:
    if type(value) is not str:
        raise GenerationCacheError(f"{label} must be a string")
    return value


def _optional_string(value: JsonValue, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise GenerationCacheError(f"{label} must be an integer")
    return value


def _cost_journal_name(request_sha256: str, attempt_number: int) -> str:
    if (
        type(request_sha256) is not str
        or len(request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in request_sha256)
    ):
        raise GenerationCacheError("cost journal request identity must be SHA-256")
    if type(attempt_number) is not int or attempt_number < 1:
        raise GenerationCacheError("cost journal attempt number must be positive")
    return f"{request_sha256}-{attempt_number:06d}"


def _cost_string(value: JsonValue, label: str) -> str:
    if type(value) is not str:
        raise GenerationCacheError(f"{label} must be a decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise GenerationCacheError(f"{label} must be decimal") from error
    if not amount.is_finite() or amount < 0:
        raise GenerationCacheError(f"{label} must be finite and nonnegative")
    return value


def _load_cost_journal_entry(directory: Path) -> CostJournalEntry:
    _require_regular_directory(directory, "runtime cost journal entry")
    files = frozenset(path.name for path in directory.iterdir())
    if files not in (
        frozenset({"reservation.json"}),
        frozenset({"reservation.json", "settlement.json"}),
        frozenset({"reservation.json", "cancellation.json"}),
    ):
        raise GenerationCacheError(
            "runtime cost journal entry contains missing or unknown files"
        )
    reservation_path = directory / "reservation.json"
    _require_regular_file(reservation_path, "cost reservation")
    reservation = require_json_object(
        canonical_json_loads(
            reservation_path.read_bytes(),
            label="cost reservation",
        ),
        label="cost reservation",
    )
    require_exact_fields(
        reservation,
        (
            "attempt_number",
            "byok_authorization",
            "byok_authorization_sha256",
            "format",
            "request_sha256",
            "schema_version",
            "upper_bound_usd",
        ),
        label="cost reservation",
    )
    if (
        reservation["format"] != _COST_RESERVATION_FORMAT
        or reservation["schema_version"] != 2
    ):
        raise GenerationCacheError("unsupported cost reservation contract")
    request_sha256 = _string(reservation["request_sha256"], "request_sha256")
    attempt_number = _integer(reservation["attempt_number"], "attempt_number")
    if directory.name != _cost_journal_name(request_sha256, attempt_number):
        raise GenerationCacheError("cost reservation directory identity mismatch")
    upper_bound = _cost_string(reservation["upper_bound_usd"], "upper_bound_usd")
    try:
        authorization = canonical_byok_authorization(
            reservation["byok_authorization"]
        )
    except (GenerationContractError, TypeError, ValueError) as error:
        raise GenerationCacheError(f"invalid BYOK authorization: {error}") from error
    authorization_sha256 = _string(
        reservation["byok_authorization_sha256"],
        "byok_authorization_sha256",
    )
    if authorization_sha256 != json_sha256(authorization):
        raise GenerationCacheError("BYOK authorization digest mismatch")
    charged: str | None = None
    actual: bool | None = None
    cancelled = False
    if "settlement.json" in files:
        settlement_path = directory / "settlement.json"
        _require_regular_file(settlement_path, "cost settlement")
        settlement = require_json_object(
            canonical_json_loads(
                settlement_path.read_bytes(),
                label="cost settlement",
            ),
            label="cost settlement",
        )
        require_exact_fields(
            settlement,
            (
                "charged_usd",
                "format",
                "provider_reported_actual",
                "schema_version",
            ),
            label="cost settlement",
        )
        if (
            settlement["format"] != _COST_SETTLEMENT_FORMAT
            or settlement["schema_version"] != 1
        ):
            raise GenerationCacheError("unsupported cost settlement contract")
        charged = _cost_string(settlement["charged_usd"], "charged_usd")
        actual_value = settlement["provider_reported_actual"]
        if type(actual_value) is not bool:
            raise GenerationCacheError(
                "provider_reported_actual must be boolean"
            )
        actual = actual_value
        if not actual and Decimal(charged) != Decimal(upper_bound):
            raise GenerationCacheError(
                "unknown-cost settlement must charge its complete upper bound"
            )
        if Decimal(charged) > Decimal(upper_bound):
            raise GenerationCacheError(
                "cost settlement exceeds its reserved upper bound"
            )
    elif "cancellation.json" in files:
        cancellation_path = directory / "cancellation.json"
        _require_regular_file(cancellation_path, "cost cancellation")
        cancellation = require_json_object(
            canonical_json_loads(
                cancellation_path.read_bytes(),
                label="cost cancellation",
            ),
            label="cost cancellation",
        )
        require_exact_fields(
            cancellation,
            ("format", "schema_version", "state"),
            label="cost cancellation",
        )
        if (
            cancellation["format"] != _COST_CANCELLATION_FORMAT
            or cancellation["schema_version"] != 1
            or cancellation["state"] != "cancelled_before_post"
        ):
            raise GenerationCacheError("unsupported cost cancellation contract")
        cancelled = True
    return CostJournalEntry(
        request_sha256=request_sha256,
        attempt_number=attempt_number,
        upper_bound_usd=upper_bound,
        charged_usd=charged,
        provider_reported_actual=actual,
        cancelled_before_post=cancelled,
        byok_authorization=authorization,
        byok_authorization_sha256=authorization_sha256,
    )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable journal file already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise GenerationCacheError(f"{label} must be a regular directory: {path}")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise GenerationCacheError(f"{label} must be a regular file: {path}")
