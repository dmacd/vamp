"""Isolated, checksummed 3-by-7 human preview of Phase 1 generator routes.

The preview is diagnostic only.  Its source records, request identities, raw
cache, cost authorization, artifact format, and destination are deliberately
separate from the Phase 1 funnel.  No preview observation is eligible for
route selection or any later benchmark gate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import html
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import NoReturn

from apm.data.text.tinyworlds_v2.audit_io import validate_phase1_semantics
from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    GENERATION_REQUEST_V1,
    SYNTHETIC_STORY_REQUEST_V2,
    SYNTHETIC_STORY_REQUEST_V3,
    VERIFIER_MODEL,
    GenerationRequestContract,
    NeutralStoryBrief,
    validate_generated_story,
)
from apm.data.text.tinyworlds_v2.catalog import (
    CatalogPayloads,
    ResolvedRouteCatalog,
    resolve_openrouter_catalog,
)
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    CostCapExceeded,
    CostPreflight,
    RouteWorkload,
    RuntimeCostLedger,
    TokenWorkload,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
    request_cost_upper_bound,
)
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.httpx_transport import (
    HttpxTransport,
    fetch_catalog_payloads,
    load_openrouter_api_key,
)
from apm.data.text.tinyworlds_v2.ingredients import (
    mechanically_classify_ingredient_roles,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    bytes_sha256,
    canonical_json_bytes,
    canonical_json_line_bytes,
    canonical_json_loads,
    json_sha256,
    require_exact_fields,
    require_json_object,
    strict_json_loads,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterBillingUnknown,
    OpenRouterClient,
    RetryPolicy,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    GeneratedSample,
    GenerationJob,
    build_generation_jobs,
    execute_generation_jobs,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    canonical_neutral_story_brief,
)
from apm.data.text.tinyworlds_v2.route_lock import (
    validate_locked_request_body,
    validate_route_semantics,
)
from apm.data.text.tinyworlds_v2.surface import (
    canonical_feature_labels,
    realized_feature_labels,
)


GENERATOR_PREVIEW_VERSION = "tinyworlds-v2-phase1-route-preview-3x7-v3"
GENERATOR_PREVIEW_ARCHIVE_VERSION = "tinyworlds-v2-phase1-route-preview-3x7-v1"
GENERATOR_PREVIEW_SELECTION_SEED = "tinyworlds-v2-phase1-generator-preview-v1"
GENERATOR_PREVIEW_HARD_CAP_USD = "0.05"
GENERATOR_PREVIEW_PRIOR_SPEND_USD = "0.008248631"
GENERATOR_PREVIEW_RUN_CAP_USD = "0.041751369"
GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST = 1
GENERATOR_PREVIEW_WORKERS = 3
GENERATOR_PREVIEW_REQUEST_CONTRACT = SYNTHETIC_STORY_REQUEST_V3
GENERATOR_PREVIEW_FORMAT = "apm.tinyworlds-v2.generator-preview"
GENERATOR_PREVIEW_SCHEMA_VERSION = 1
GENERATOR_PREVIEW_SOURCE_RECORD_IDS = (
    "archive:./data12.json:53291:"
    "a0cf1f6d2a17a474049876e4138ed35e927d5654e304903f48550a59f4c2d7bb",
    "archive:./data07.json:68215:"
    "86a0d029c5cf9da3c806b42c11e7acf91929d4e4df279f759cdd645fc79e967b",
    "archive:./data28.json:27610:"
    "ccb5459d0f913a496cd2145f0b3538f1f8cd0e25c4663bf58446d8696666e02f",
)

_PROMPT_RECORD_FIELDS = (
    "content_sha256",
    "features",
    "normalized_story_sha256",
    "prompt",
    "record_id",
    "source",
    "source_index",
    "source_member",
    "story",
    "summary",
    "words",
)
_MANIFEST_FILE = "manifest.json"


class GeneratorPreviewError(ValueError):
    """The isolated generator preview violates its fixed contract."""


@dataclass(frozen=True, slots=True)
class GeneratorPreviewPaths:
    """Fixed source, cache, authorization, and artifact locations."""

    repository_root: Path
    source_artifact: Path
    tokenizer: Path
    raw_cache: Path
    destination: Path
    byok_attestation: Path

    @classmethod
    def from_repository(cls, repository_root: str | Path) -> "GeneratorPreviewPaths":
        root = Path(repository_root).resolve()
        tinyworlds_root = root / "data" / "tinyworlds-v2"
        return cls(
            repository_root=root,
            source_artifact=tinyworlds_root / "reference",
            tokenizer=(
                root
                / "checkpoints"
                / "tinystories-8m"
                / "tokenizer"
                / "tokenizer.json"
            ),
            raw_cache=(
                tinyworlds_root
                / "cache"
                / "phase1-route-preview-3x7-v3-openrouter"
            ),
            destination=(
                tinyworlds_root
                / "previews"
                / "phase1-route-preview-3x7-v3"
            ),
            byok_attestation=(
                root / "openrouter-tinyworlds-preview-no-byok-attestation.json"
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneratorPreviewValidation:
    """Authenticated preview identity and its complete result records."""

    manifest_sha256: str
    briefs: tuple[NeutralStoryBrief, ...]
    results: tuple[JsonObject, ...]
    actual_cost_usd: str
    version: str


@dataclass(frozen=True, slots=True)
class _PreviewArtifactBuilder:
    """Small write-once builder for the preview's distinct artifact format."""

    root: Path

    def __post_init__(self) -> None:
        if not self.root.is_dir() or self.root.is_symlink():
            raise GeneratorPreviewError("preview staging root must be a directory")

    def write_bytes(self, relative: str, payload: bytes) -> Path:
        path = self._new_path(relative)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def write_json(self, relative: str, value: JsonValue) -> Path:
        if not relative.endswith(".json"):
            raise GeneratorPreviewError("preview JSON paths must end in .json")
        return self.write_bytes(relative, canonical_json_bytes(value))

    def write_jsonl(self, relative: str, values: Sequence[JsonValue]) -> Path:
        if not relative.endswith(".jsonl"):
            raise GeneratorPreviewError("preview JSONL paths must end in .jsonl")
        return self.write_bytes(
            relative,
            b"".join(canonical_json_line_bytes(value) for value in values),
        )

    def append_jsonl(self, relative: str, value: JsonValue) -> Path:
        if not relative.endswith(".jsonl"):
            raise GeneratorPreviewError("preview JSONL paths must end in .jsonl")
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise GeneratorPreviewError("preview JSONL path is not a regular file")
        mode = "ab" if path.exists() else "xb"
        with path.open(mode) as stream:
            stream.write(canonical_json_line_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def finalize(self) -> JsonObject:
        manifest_path = self.root / _MANIFEST_FILE
        if manifest_path.exists() or manifest_path.is_symlink():
            raise GeneratorPreviewError("preview staging tree is already finalized")
        artifacts = []
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise GeneratorPreviewError("preview artifact tree contains a symlink")
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            payload = path.read_bytes()
            record_count = _validate_artifact_payload(relative, payload)
            artifacts.append(
                {
                    "path": relative,
                    "record_count": record_count,
                    "sha256": bytes_sha256(payload),
                    "size_bytes": len(payload),
                }
            )
        if not artifacts:
            raise GeneratorPreviewError("preview artifact tree is empty")
        core: JsonObject = {
            "artifacts": artifacts,
            "format": GENERATOR_PREVIEW_FORMAT,
            "schema_version": GENERATOR_PREVIEW_SCHEMA_VERSION,
            "version": GENERATOR_PREVIEW_VERSION,
        }
        manifest: JsonObject = {
            **core,
            "manifest_sha256": json_sha256(core),
        }
        self.write_json(_MANIFEST_FILE, manifest)
        return manifest

    def _new_path(self, relative: str) -> Path:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"preview artifact already exists: {relative}")
        return path

    def _path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if (
            type(relative) is not str
            or not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or relative != pure.as_posix()
        ):
            raise GeneratorPreviewError("preview artifact path is not canonical")
        return self.root.joinpath(*pure.parts)


class _NetworkForbiddenTransport:
    """Fail if replay attempts either a completion or metadata request."""

    def post(self, **_arguments: object) -> NoReturn:
        raise GeneratorPreviewError("preview replay attempted an HTTP POST")

    def get_authenticated(self, **_arguments: object) -> NoReturn:
        raise GeneratorPreviewError("preview replay attempted an authenticated GET")


class _ReplayCostLedger(RuntimeCostLedger):
    """Interpret cached bytes without scheduling or accounting new work."""

    def attach_cache(self, _cache: ImmutableRawCache) -> None:
        return None

    def reconcile_cached(
        self,
        _request: object,
        _route: object,
        _attempts: object,
    ) -> None:
        return None

    def ensure_recovery_complete(self) -> None:
        return None

    def halt(self, _reason: str) -> None:
        return None

    def reserve(self, *_arguments: object, **_keywords: object) -> NoReturn:
        raise GeneratorPreviewError("preview replay attempted a billable reservation")


class _PreviewRuntimeCostLedger(RuntimeCostLedger):
    """Continue after an exact recognized missing-cost failure charges its bound."""

    def _refresh_recovery_halt_locked(self) -> None:
        super()._refresh_recovery_halt_locked()
        if (
            self._halted_reason == "provider_billing_unknown"
            and self._cache is not None
            and _only_bounded_missing_cost_failures_are_unknown(self._cache)
        ):
            # Unknown cost remains conservatively charged at its complete
            # request bound. Only exact pre-inference rejections and an exact
            # gateway-timeout shape with exhausted missing-generation lookups
            # clear the stop bit. Every other missing-cost shape is terminal.
            self._halted_reason = None

    def acknowledge_bounded_missing_cost_failure(self) -> None:
        with self._lock:
            if (
                self._halted_reason != "provider_billing_unknown"
                or self._cache is None
                or not _only_bounded_missing_cost_failures_are_unknown(
                    self._cache
                )
            ):
                raise GeneratorPreviewError(
                    "cannot continue after an unrecognized missing-cost response"
                )
            self._halted_reason = None


def load_generator_preview_briefs(path: str | Path) -> tuple[NeutralStoryBrief, ...]:
    """Select one deterministic 0-, 1-, and 2+-feature profiling record."""
    records = _jsonl_objects(Path(path))
    best: dict[int, tuple[str, JsonObject, tuple[str, ...]]] = {}
    for record in records:
        require_exact_fields(record, _PROMPT_RECORD_FIELDS, label="prompt metadata")
        prompt = _text(record["prompt"], "prompt")
        record_id = _text(record["record_id"], "record_id")
        story = _text(record["story"], "story")
        words = _text_tuple(record["words"], "words")
        features = canonical_feature_labels(_text_tuple(record["features"], "features"))
        if (
            record.get("source") != "GPT-4"
            or len(words) != 3
            or len({word.casefold() for word in words}) != 3
            or mechanically_classify_ingredient_roles(prompt, words) is None
        ):
            continue
        bucket = min(len(features), 2)
        rank = sha256(
            f"{GENERATOR_PREVIEW_SELECTION_SEED}\0{bucket}\0{record_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        candidate = (rank, record, features)
        if bucket not in best or rank < best[bucket][0]:
            best[bucket] = candidate
    if set(best) != {0, 1, 2}:
        raise GeneratorPreviewError("profiling cohort lacks all preview feature strata")

    briefs = []
    for bucket in (0, 1, 2):
        _, record, features = best[bucket]
        record_id = _text(record["record_id"], "record_id")
        digest = sha256(
            f"{GENERATOR_PREVIEW_SELECTION_SEED}\0{record_id}".encode("utf-8")
        ).hexdigest()
        briefs.append(
            NeutralStoryBrief(
                brief_id=f"preview-brief-{digest[:24]}",
                source_record_id=record_id,
                prompt_text=_text(record["prompt"], "prompt"),
                required_words=_text_tuple(record["words"], "words"),
                requested_features=features,
                matched_reference_text=_text(record["story"], "story"),
            )
        )
    selected = tuple(briefs)
    if tuple(brief.source_record_id for brief in selected) != (
        GENERATOR_PREVIEW_SOURCE_RECORD_IDS
    ):
        raise GeneratorPreviewError("preview source selection differs from its frozen IDs")
    return selected


def build_generator_preview_preflight(
    jobs: tuple[GenerationJob, ...],
    encode_text: Callable[[str], tuple[int, ...]],
) -> CostPreflight:
    """Estimate the exact 21 logical jobs with no retry allowance."""
    _validate_job_matrix(jobs)
    workloads = []
    for model in CANDIDATE_MODELS:
        route_jobs = tuple(job for job in jobs if job.route.route_id == model.route_id)
        request_tokens = tuple(
            len(encode_text(job.request.body_json)) for job in route_jobs
        )
        output_tokens = tuple(
            len(encode_text(job.brief.matched_reference_text)) for job in route_jobs
        )
        workloads.append(
            RouteWorkload(
                route_jobs[0].route,
                TokenWorkload(
                    label="generator-route-preview-3x7-v3",
                    request_count=3,
                    input_tokens_per_request=math.ceil(sum(request_tokens) / 3),
                    output_tokens_per_request=math.ceil(sum(output_tokens) / 3),
                    conservative_input_tokens_per_request=max(
                        2 * value + 512 for value in request_tokens
                    ),
                    conservative_output_tokens_per_request=512,
                    retry_allowance_basis_points=0,
                ),
            )
        )
    return build_cost_preflight(
        tuple(workloads),
        hard_cap_usd=GENERATOR_PREVIEW_RUN_CAP_USD,
    )


def run_generator_preview(
    repository_root: str | Path,
    *,
    authorize_paid_preview: bool = False,
    emit: Callable[[str], None] = print,
) -> Path:
    """Run the isolated 21-request preview and atomically publish its bundle."""
    paths = GeneratorPreviewPaths.from_repository(repository_root)
    if paths.destination.is_dir():
        validate_generator_preview(paths.destination)
        verify_generator_preview_replay(paths.destination)
        emit(f"Existing validated preview: {paths.destination}")
        emit("No source, catalog, key, or paid generation work was repeated.")
        return paths.destination
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"preview destination is not a directory: {paths.destination}")
    if authorize_paid_preview is not True:
        raise PermissionError("generator preview requires explicit paid authorization")

    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".phase1-route-preview-3x7-v3-",
            dir=paths.destination.parent,
        )
    )
    builder = _PreviewArtifactBuilder(temporary)
    emit(f"Temporary preview directory: {temporary}")
    emit("Preview 1/3: validate disjoint source briefs and freeze exact requests")

    source = validate_phase1_semantics(paths.source_artifact)
    briefs = load_generator_preview_briefs(
        paths.source_artifact / "prompt_metadata_sample.jsonl"
    )
    initial_payloads, initial_catalog = _fetch_catalog()
    _write_catalog(
        builder,
        "catalog/initial",
        initial_payloads,
        initial_catalog,
    )
    jobs = build_generation_jobs(
        briefs,
        CANDIDATE_MODELS,
        initial_catalog.generator_routes,
        request_contract=GENERATOR_PREVIEW_REQUEST_CONTRACT,
    )
    _validate_job_matrix(jobs)

    from apm.lm.text import TokenizersTextTokenizer

    tokenizer = TokenizersTextTokenizer.from_file(paths.tokenizer)
    encode_text = lambda text: tokenizer.encode(text, add_eos=False)
    token_preflight = build_generator_preview_preflight(jobs, encode_text)
    exact_bounds = tuple(request_cost_upper_bound(job.request, job.route) for job in jobs)
    exact_total = sum(
        (Decimal(bound.upper_bound_usd) for bound in exact_bounds),
        Decimal(0),
    )
    enforce_cost_cap(token_preflight)
    if (
        exact_total + Decimal(GENERATOR_PREVIEW_PRIOR_SPEND_USD)
        > Decimal(GENERATOR_PREVIEW_HARD_CAP_USD)
    ):
        raise CostCapExceeded(
            "cumulative corrected-preview exposure exceeds its $0.05 hard cap"
        )

    builder.write_jsonl(
        "briefs.jsonl",
        tuple(canonical_neutral_story_brief(brief) for brief in briefs),
    )
    builder.write_jsonl(
        "requests.jsonl",
        tuple(
            {
                **job.request.as_record(),
                "body": job.request.body,
                "brief_id": job.brief.brief_id,
                "route_id": job.route.route_id,
            }
            for job in jobs
        ),
    )
    builder.write_jsonl(
        "request_cost_bounds.jsonl",
        tuple(bound.as_record() for bound in exact_bounds),
    )
    builder.write_json("token_cost_estimates.json", token_preflight.as_record())
    builder.write_json(
        "cost_authorization.json",
        {
            "authorized_attempts_per_request": GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST,
            "exact_request_byte_upper_bound_usd": format(exact_total, "f"),
            "cumulative_hard_cap_usd": GENERATOR_PREVIEW_HARD_CAP_USD,
            "logical_request_count": len(jobs),
            "prior_interrupted_run_spend_usd": GENERATOR_PREVIEW_PRIOR_SPEND_USD,
            "retry_allowance": "none",
            "run_hard_cap_usd": GENERATOR_PREVIEW_RUN_CAP_USD,
        },
    )
    builder.write_json(
        "preview_plan.json",
        {
            "eligible_for_phase1_funnel": False,
            "eligible_for_route_selection": False,
            "route_ids": [model.route_id for model in CANDIDATE_MODELS],
            "request_contract_version": GENERATOR_PREVIEW_REQUEST_CONTRACT.version,
            "scientific_role": "diagnostic_only",
            "selection_seed": GENERATOR_PREVIEW_SELECTION_SEED,
            "source_manifest_sha256": source.manifest.manifest_sha256,
            "source_record_ids": [brief.source_record_id for brief in briefs],
            "version": GENERATOR_PREVIEW_VERSION,
        },
    )
    emit(
        "  Expected token-priced spend "
        f"${token_preflight.expected_usd}; exact one-attempt ceiling "
        f"${format(exact_total, 'f')} / ${GENERATOR_PREVIEW_RUN_CAP_USD} run cap; "
        f"${GENERATOR_PREVIEW_HARD_CAP_USD} cumulative cap"
    )

    emit("Preview 2/3: run three prompts through each of seven pinned routes")
    samples: list[GeneratedSample] = []
    with exclusive_paid_run_lock(paths.raw_cache):
        cache = ImmutableRawCache(paths.raw_cache)
        _validate_cache_membership(cache, jobs, require_complete=False)
        ledger = _PreviewRuntimeCostLedger(GENERATOR_PREVIEW_RUN_CAP_USD)
        ledger.bootstrap(cache, initial_catalog.generator_routes)
        _enforce_resume_exposure(ledger, cache, jobs)

        # The inference key is intentionally read only after both independent
        # preflights, exact cache membership, and recovery accounting pass.
        client = OpenRouterClient(
            api_key=load_openrouter_api_key(paths.repository_root),
            management_api_key=None,
            byok_attestation_path=paths.byok_attestation,
            transport=HttpxTransport(),
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=1),
            cost_ledger=ledger,
            require_byok_preflight=True,
        )
        byok = client.verify_no_byok()
        builder.write_json("byok_preflight.json", byok.as_record())

        for route_index, model in enumerate(CANDIDATE_MODELS):
            route_id = model.route_id
            emit(f"  [{route_index + 1}/7] {route_id}: refresh route, then 3 calls")
            fresh_payloads, fresh_catalog = _fetch_catalog()
            namespace = f"catalog/submission/{route_index:02d}-{route_id}"
            _write_catalog(builder, namespace, fresh_payloads, fresh_catalog)
            initial_route = initial_catalog.generator_routes[route_index]
            fresh_route = fresh_catalog.generator_routes[route_index]
            try:
                validate_route_semantics(initial_route, fresh_route)
            except Exception:
                ledger.halt("catalog_route_drift")
                raise
            route_jobs = tuple(
                replace(job, route=fresh_route)
                for job in jobs
                if job.route.route_id == route_id
            )
            route_samples = _execute_preview_route_jobs(
                route_jobs,
                client,
                ledger,
                cache,
            )
            for sample in route_samples:
                builder.append_jsonl("preview_results.jsonl", sample.as_record())
                samples.append(sample)
            builder.append_jsonl(
                "progress.jsonl",
                {
                    "accepted_count": sum(
                        sample.validation.accepted for sample in route_samples
                    ),
                    "event": "route_batch_complete",
                    "request_count": len(route_samples),
                    "route_id": route_id,
                },
            )

        _validate_cache_membership(cache, jobs, require_complete=True)
        runtime = ledger.snapshot()
        builder.write_json("runtime_cost_ledger.json", runtime.as_record())
        builder.write_json(
            "cost_actuals.json",
            _cost_actuals(cache, jobs, runtime.as_record(), samples),
        )
        shutil.copytree(paths.raw_cache, temporary / "raw_cache")

    completed = tuple(samples)
    builder.write_bytes("preview.md", _preview_markdown(briefs, completed).encode("utf-8"))
    builder.write_bytes("preview.html", _preview_html(briefs, completed).encode("utf-8"))
    builder.write_json(
        "status.json",
        {
            "accepted_count": sum(sample.validation.accepted for sample in completed),
            "eligible_for_phase1_funnel": False,
            "eligible_for_route_selection": False,
            "request_count": len(completed),
            "scientific_role": "diagnostic_only",
            "status": "awaiting_preview_review",
        },
    )

    emit("Preview 3/3: zero-network replay, validate, and atomically promote")
    verify_generator_preview_replay(temporary, require_manifest=False)
    builder.write_json(
        "replay.json",
        {
            "network_requests": 0,
            "replayed_request_count": len(completed),
            "status": "passed",
        },
    )
    manifest = builder.finalize()
    validate_generator_preview(temporary)
    try:
        os.rename(temporary, paths.destination)
    except FileExistsError:
        raise FileExistsError(f"preview destination already exists: {paths.destination}")
    validate_generator_preview(paths.destination)
    emit(
        "Generator preview promoted: "
        f"{paths.destination} (manifest {manifest['manifest_sha256']})"
    )
    emit(f"Human comparison: {paths.destination / 'preview.html'}")
    return paths.destination


def validate_generator_preview(root: str | Path) -> GeneratorPreviewValidation:
    """Strictly authenticate the complete diagnostic preview bundle."""
    resolved = Path(root)
    manifest = _load_preview_manifest(resolved)
    version = _text(manifest["version"], "preview version")
    request_contract = _preview_request_contract(version)
    briefs = _load_briefs(resolved / "briefs.jsonl")
    if len(briefs) != 3:
        raise GeneratorPreviewError("preview must contain exactly three briefs")
    initial = _load_catalog(resolved / "catalog" / "initial")
    jobs = build_generation_jobs(
        briefs,
        CANDIDATE_MODELS,
        initial.generator_routes,
        request_contract=request_contract,
    )
    _validate_job_matrix(jobs)
    request_records = _jsonl_objects(resolved / "requests.jsonl")
    expected_requests = tuple(
        {
            **job.request.as_record(),
            "body": job.request.body,
            "brief_id": job.brief.brief_id,
            "route_id": job.route.route_id,
        }
        for job in jobs
    )
    if request_records != expected_requests:
        raise GeneratorPreviewError("preview request stream differs from its plan")
    preview_plan = require_json_object(
        canonical_json_loads(
            (resolved / "preview_plan.json").read_bytes(),
            label="preview plan",
        ),
        label="preview plan",
    )
    recorded_contract = preview_plan.get("request_contract_version")
    if version == GENERATOR_PREVIEW_VERSION:
        if recorded_contract != request_contract.version:
            raise GeneratorPreviewError("preview request contract version differs")
    elif recorded_contract is not None:
        raise GeneratorPreviewError("archived preview invents a request contract version")
    for route_index, model in enumerate(CANDIDATE_MODELS):
        submitted = _load_catalog(
            resolved
            / "catalog"
            / "submission"
            / f"{route_index:02d}-{model.route_id}"
        )
        validate_route_semantics(
            initial.generator_routes[route_index],
            submitted.generator_routes[route_index],
        )

    results = _jsonl_objects(resolved / "preview_results.jsonl")
    if len(results) != 21:
        raise GeneratorPreviewError("preview must contain exactly 21 results")
    expected_pairs = tuple(
        (job.route.route_id, job.brief.brief_id, job.request.request_sha256)
        for job in jobs
    )
    actual_pairs = tuple(
        (
            _text(record.get("route_id"), "result route_id"),
            _text(record.get("brief_id"), "result brief_id"),
            _text(record.get("request_sha256"), "result request_sha256"),
        )
        for record in results
    )
    if actual_pairs != expected_pairs:
        raise GeneratorPreviewError("preview result matrix differs from its 3-by-7 plan")

    cache = ImmutableRawCache(resolved / "raw_cache")
    _validate_cache_membership(cache, jobs, require_complete=True)
    status = require_json_object(
        canonical_json_loads((resolved / "status.json").read_bytes(), label="preview status"),
        label="preview status",
    )
    expected_status = {
        "accepted_count": sum(
            bool(require_json_object(record["validation"], label="validation").get("accepted"))
            for record in results
        ),
        "eligible_for_phase1_funnel": False,
        "eligible_for_route_selection": False,
        "request_count": 21,
        "scientific_role": "diagnostic_only",
        "status": "awaiting_preview_review",
    }
    if status != expected_status:
        raise GeneratorPreviewError("preview status differs from derived results")
    actuals = require_json_object(
        canonical_json_loads(
            (resolved / "cost_actuals.json").read_bytes(),
            label="preview cost actuals",
        ),
        label="preview cost actuals",
    )
    actual_cost = _text(actuals.get("provider_reported_actual_usd"), "actual cost")
    if Decimal(actual_cost) > Decimal(GENERATOR_PREVIEW_RUN_CAP_USD):
        raise GeneratorPreviewError("preview actual cost exceeds its hard cap")
    return GeneratorPreviewValidation(
        manifest_sha256=_text(manifest["manifest_sha256"], "manifest digest"),
        briefs=briefs,
        results=results,
        actual_cost_usd=actual_cost,
        version=version,
    )


def verify_generator_preview_replay(
    root: str | Path,
    *,
    require_manifest: bool = True,
) -> None:
    """Reparse all 21 cached responses with a transport that forbids network."""
    source = Path(root)
    if require_manifest:
        validation = validate_generator_preview(source)
        briefs = validation.briefs
        request_contract = _preview_request_contract(validation.version)
    else:
        briefs = _load_briefs(source / "briefs.jsonl")
        request_contract = GENERATOR_PREVIEW_REQUEST_CONTRACT
    with tempfile.TemporaryDirectory(prefix="tinyworlds-v2-preview-replay-") as name:
        cache_root = Path(name) / "raw_cache"
        shutil.copytree(source / "raw_cache", cache_root)
        cache = ImmutableRawCache(cache_root)
        route_by_id: dict[str, RouteLock] = {}
        for request in cache.load_all_requests():
            route = cache.load_route_lock(request.request_sha256)
            previous = route_by_id.setdefault(route.route_id, route)
            validate_route_semantics(previous, route)
        routes = tuple(route_by_id[model.route_id] for model in CANDIDATE_MODELS)
        jobs = build_generation_jobs(
            briefs,
            CANDIDATE_MODELS,
            routes,
            request_contract=request_contract,
        )
        _validate_cache_membership(cache, jobs, require_complete=True)
        client = OpenRouterClient(
            api_key="offline-preview-replay-never-sent",
            transport=_NetworkForbiddenTransport(),
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=1),
            cost_ledger=_ReplayCostLedger(),
            sleeper=lambda _seconds: None,
            require_byok_preflight=False,
        )
        samples = tuple(
            _bounded_missing_cost_sample(job, cache)
            if _request_missing_cost_failure_kind(cache, job) is not None
            else execute_generation_jobs((job,), client, max_workers=1)[0]
            for job in jobs
        )
    expected = b"".join(
        canonical_json_line_bytes(sample.as_record()) for sample in samples
    )
    if expected != (source / "preview_results.jsonl").read_bytes():
        raise GeneratorPreviewError("zero-network preview result replay differs")
    if _preview_markdown(briefs, samples).encode("utf-8") != (
        source / "preview.md"
    ).read_bytes():
        raise GeneratorPreviewError("zero-network preview Markdown replay differs")
    if _preview_html(briefs, samples).encode("utf-8") != (
        source / "preview.html"
    ).read_bytes():
        raise GeneratorPreviewError("zero-network preview HTML replay differs")


def _write_catalog(
    builder: _PreviewArtifactBuilder,
    namespace: str,
    payloads: CatalogPayloads,
    resolved: ResolvedRouteCatalog,
) -> None:
    if payloads.snapshot_sha256 != resolved.snapshot_sha256:
        raise GeneratorPreviewError("catalog payload and routes differ")
    builder.write_bytes(f"{namespace}/models.response", payloads.models)
    endpoint_by_model = dict(payloads.endpoints)
    for model in (*CANDIDATE_MODELS, VERIFIER_MODEL):
        builder.write_bytes(
            f"{namespace}/endpoints/{model.route_id}.response",
            endpoint_by_model[model.request_model_id],
        )
    builder.write_json(
        f"{namespace}/routes.json",
        {
            "generator_routes": [route.as_record() for route in resolved.generator_routes],
            "snapshot_sha256": resolved.snapshot_sha256,
            "verifier_route": resolved.verifier_route.as_record(),
        },
    )


def _fetch_catalog() -> tuple[CatalogPayloads, ResolvedRouteCatalog]:
    payloads = fetch_catalog_payloads(HttpxTransport())
    return payloads, resolve_openrouter_catalog(payloads)


def _load_catalog(root: Path) -> ResolvedRouteCatalog:
    payloads = CatalogPayloads(
        models=(root / "models.response").read_bytes(),
        endpoints=tuple(
            (
                model.request_model_id,
                (root / "endpoints" / f"{model.route_id}.response").read_bytes(),
            )
            for model in (*CANDIDATE_MODELS, VERIFIER_MODEL)
        ),
    )
    return resolve_openrouter_catalog(payloads)


def _validate_job_matrix(jobs: tuple[GenerationJob, ...]) -> None:
    if len(jobs) != 21:
        raise GeneratorPreviewError("generator preview requires exactly 21 jobs")
    expected = tuple(
        (model.route_id, brief_index)
        for model in CANDIDATE_MODELS
        for brief_index in range(3)
    )
    brief_order = tuple(dict.fromkeys(job.brief.brief_id for job in jobs))
    actual = tuple(
        (job.route.route_id, brief_order.index(job.brief.brief_id)) for job in jobs
    )
    if actual != expected or len({job.request.request_sha256 for job in jobs}) != 21:
        raise GeneratorPreviewError("preview jobs do not form a unique route-major 3-by-7 matrix")


def _execute_preview_route_jobs(
    jobs: tuple[GenerationJob, ...],
    client: OpenRouterClient,
    ledger: _PreviewRuntimeCostLedger,
    cache: ImmutableRawCache,
) -> tuple[GeneratedSample, ...]:
    """Use one canary, then two workers; preserve bounded routing failures."""
    first = _execute_preview_job(jobs[0], client, ledger, cache)
    if first.error_kind in {
        "OpenRouterDistillationRoutingRejected",
        "OpenRouterProviderJsonInstructionRejected",
        "OpenRouterProviderSeedRejected",
    }:
        rest = tuple(
            _execute_preview_job(job, client, ledger, cache) for job in jobs[1:]
        )
    else:
        with ThreadPoolExecutor(
            max_workers=min(GENERATOR_PREVIEW_WORKERS, len(jobs) - 1)
        ) as executor:
            rest = tuple(
                executor.map(
                    lambda job: _execute_preview_job(job, client, ledger, cache),
                    jobs[1:],
                )
            )
    return (first, *rest)


def _execute_preview_job(
    job: GenerationJob,
    client: OpenRouterClient,
    ledger: _PreviewRuntimeCostLedger,
    cache: ImmutableRawCache,
) -> GeneratedSample:
    try:
        return execute_generation_jobs((job,), client, max_workers=1)[0]
    except OpenRouterBillingUnknown:
        if _request_missing_cost_failure_kind(cache, job) is None:
            raise
        ledger.acknowledge_bounded_missing_cost_failure()
        return _bounded_missing_cost_sample(job, cache)


def _bounded_missing_cost_sample(
    job: GenerationJob,
    cache: ImmutableRawCache,
) -> GeneratedSample:
    error_kind = _request_missing_cost_failure_kind(cache, job)
    if error_kind is None:
        raise GeneratorPreviewError("request is not a bounded missing-cost failure")
    _, validation = validate_generated_story(job.brief, b"")
    return GeneratedSample(
        job=job,
        payload=None,
        validation=validation,
        generation_id=None,
        input_tokens=0,
        output_tokens=0,
        billed_cost_usd="0",
        error_kind=error_kind,
    )


def _request_missing_cost_failure_kind(
    cache: ImmutableRawCache,
    job: GenerationJob,
) -> str | None:
    try:
        attempts = cache.load_attempts(job.request)
    except ValueError:
        return None
    return (
        None
        if len(attempts) != 1
        else _bounded_missing_cost_failure_kind(attempts[0].response)
    )


def _only_bounded_missing_cost_failures_are_unknown(
    cache: ImmutableRawCache,
) -> bool:
    unknown_entries = tuple(
        entry
        for entry in cache.load_cost_journal()
        if not entry.cancelled_before_post
        and entry.provider_reported_actual is not True
    )
    if not unknown_entries:
        return False
    for entry in unknown_entries:
        request = cache.load_request(entry.request_sha256)
        attempts = cache.load_attempts(request)
        if (
            entry.attempt_number > len(attempts)
            or _bounded_missing_cost_failure_kind(
                attempts[entry.attempt_number - 1].response
            )
            is None
        ):
            return False
    return True


def _bounded_missing_cost_failure_kind(response: object) -> str | None:
    if response is None or getattr(response, "status_code", None) not in {
        200,
        400,
        404,
        502,
    }:
        return None
    if getattr(response, "billed_cost_usd", None) is not None:
        return None
    try:
        record = require_json_object(
            strict_json_loads(response.body, label="routing rejection"),
            label="routing rejection",
        )
        error = require_json_object(record.get("error"), label="routing error")
        stats_are_missing = _all_generation_stats_are_missing(response)
        if (
            error.get("code") == 504
            and _text(error.get("message"), "routing error message").strip()
            == "error code: 524"
            and stats_are_missing
        ):
            return "OpenRouterProviderGatewayTimeout"
        if (
            error.get("code") == 502
            and "'messages' must contain the word 'json'" in _text(
                error.get("message"), "routing error message"
            )
            and "'response_format' of type 'json_object'" in _text(
                error.get("message"), "routing error message"
            )
            and stats_are_missing
        ):
            return "OpenRouterProviderJsonInstructionRejected"
        metadata = require_json_object(
            record.get("openrouter_metadata"),
            label="routing metadata",
        )
        endpoints = require_json_object(
            metadata.get("endpoints"),
            label="routing endpoints",
        )
        available = endpoints.get("available")
        if (
            error.get("code") == 404
            and "allow text distillation" in _text(
                error.get("message"), "routing error message"
            )
            and metadata.get("attempt") == 0
            and metadata.get("is_byok") is False
            and type(available) is list
            and bool(available)
            and all(
                type(endpoint) is dict and endpoint.get("selected") is False
                for endpoint in available
            )
        ):
            return "OpenRouterDistillationRoutingRejected"
        error_metadata = require_json_object(
            error.get("metadata"),
            label="provider error metadata",
        )
        raw = require_json_object(
            strict_json_loads(
                _text(error_metadata.get("raw"), "raw provider error").encode(
                    "utf-8"
                ),
                label="raw provider error",
            ),
            label="raw provider error",
        )
        raw_error = require_json_object(
            raw.get("error"),
            label="raw provider error payload",
        )
        if (
            error.get("code") == 400
            and error.get("message") == "Provider returned error"
            and error_metadata.get("is_byok") is False
            and error_metadata.get("provider_name") == "Alibaba"
            and error_metadata.get("provider_error_code") == "invalid_value"
            and raw_error.get("message") == "'seed' must be Integer"
            and raw_error.get("code") == "invalid_value"
            and metadata.get("attempt") == 1
            and metadata.get("is_byok") is False
            and type(available) is list
            and bool(available)
            and all(
                type(endpoint) is dict and endpoint.get("selected") is False
                for endpoint in available
            )
        ):
            return "OpenRouterProviderSeedRejected"
        return None
    except (TypeError, ValueError):
        return None


def _all_generation_stats_are_missing(response: object) -> bool:
    """Recognize only a complete sequence of exact missing-generation lookups."""
    attempts = getattr(response, "generation_stats_attempts", ())
    if not attempts:
        return False
    try:
        for attempt in attempts:
            if attempt.response is None or attempt.response.status_code != 404:
                return False
            record = require_json_object(
                strict_json_loads(
                    attempt.response.body,
                    label="missing generation stats",
                ),
                label="missing generation stats",
            )
            error = require_json_object(
                record.get("error"),
                label="missing generation stats error",
            )
            message = _text(
                error.get("message"),
                "missing generation stats message",
            )
            if (
                error.get("code") != 404
                or not message.startswith("Generation gen-")
                or not message.endswith(" not found")
            ):
                return False
        return True
    except (TypeError, ValueError):
        return False


def _validate_cache_membership(
    cache: ImmutableRawCache,
    jobs: tuple[GenerationJob, ...],
    *,
    require_complete: bool,
) -> None:
    planned = {job.request.request_sha256: job for job in jobs}
    cached = cache.load_all_requests()
    for request in cached:
        job = planned.get(request.request_sha256)
        if job is None or request != job.request:
            raise GeneratorPreviewError("preview cache contains an unplanned request")
        route = cache.load_route_lock(request.request_sha256)
        validate_locked_request_body(route, request)
        validate_route_semantics(job.route, route)
        attempts = cache.load_attempts(request)
        if len(attempts) > GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST:
            raise GeneratorPreviewError("preview cache exceeds its one-attempt contract")
    journal = cache.load_cost_journal()
    if any(entry.cancelled_before_post for entry in journal):
        raise GeneratorPreviewError("preview cache contains a cancelled paid reservation")
    attempt_ids = {
        (request.request_sha256, attempt.attempt_number)
        for request in cached
        for attempt in cache.load_attempts(request)
    }
    journal_ids = {(entry.request_sha256, entry.attempt_number) for entry in journal}
    if attempt_ids != journal_ids:
        raise GeneratorPreviewError("preview attempts and cost journal differ")
    if require_complete:
        if set(planned) != {request.request_sha256 for request in cached}:
            raise GeneratorPreviewError("complete preview cache lacks planned requests")
        if len(attempt_ids) != 21:
            raise GeneratorPreviewError("complete preview cache must contain 21 attempts")


def _enforce_resume_exposure(
    ledger: RuntimeCostLedger,
    cache: ImmutableRawCache,
    jobs: tuple[GenerationJob, ...],
) -> None:
    cached_attempted = {
        request.request_sha256
        for request in cache.load_all_requests()
        if cache.load_attempts(request)
    }
    remaining = sum(
        (
            Decimal(request_cost_upper_bound(job.request, job.route).upper_bound_usd)
            for job in jobs
            if job.request.request_sha256 not in cached_attempted
        ),
        Decimal(0),
    )
    charged = Decimal(ledger.snapshot().charged_total_usd)
    if charged + remaining > Decimal(GENERATOR_PREVIEW_RUN_CAP_USD):
        raise CostCapExceeded("resumed preview exposure exceeds its run hard cap")


def _cost_actuals(
    cache: ImmutableRawCache,
    jobs: tuple[GenerationJob, ...],
    runtime: JsonObject,
    samples: Sequence[GeneratedSample],
) -> JsonObject:
    route_by_request = {
        job.request.request_sha256: job.route.route_id for job in jobs
    }
    per_route: dict[str, Decimal] = {
        model.route_id: Decimal(0) for model in CANDIDATE_MODELS
    }
    unknown_per_route: dict[str, Decimal] = {
        model.route_id: Decimal(0) for model in CANDIDATE_MODELS
    }
    for entry in cache.load_cost_journal():
        amount = Decimal(
            entry.upper_bound_usd if entry.charged_usd is None else entry.charged_usd
        )
        route_id = route_by_request[entry.request_sha256]
        if entry.provider_reported_actual is True:
            per_route[route_id] += amount
        else:
            unknown_per_route[route_id] += amount
    return {
        "cumulative_hard_cap_usd": GENERATOR_PREVIEW_HARD_CAP_USD,
        "prior_interrupted_run_spend_usd": GENERATOR_PREVIEW_PRIOR_SPEND_USD,
        "run_hard_cap_usd": GENERATOR_PREVIEW_RUN_CAP_USD,
        "provider_reported_actual_usd": _text(
            runtime.get("provider_reported_actual_usd"), "runtime actual cost"
        ),
        "conservative_unknown_charge_usd": _text(
            runtime.get("conservative_unknown_charge_usd"), "runtime unknown cost"
        ),
        "route_costs": [
            {
                "conservative_unknown_charge_usd": format(unknown_per_route[model.route_id], "f"),
                "provider_reported_actual_usd": format(per_route[model.route_id], "f"),
                "route_id": model.route_id,
            }
            for model in CANDIDATE_MODELS
        ],
        "sample_reported_cost_sum_usd": format(
            sum((Decimal(sample.billed_cost_usd) for sample in samples), Decimal(0)),
            "f",
        ),
    }


def _preview_markdown(
    briefs: tuple[NeutralStoryBrief, ...],
    samples: Sequence[GeneratedSample],
) -> str:
    lines = [
        "# TinyWorlds-v2 generator route preview",
        "",
        "Diagnostic only: these records are ineligible for the Phase 1 funnel and route selection.",
        "",
        "The full canonical request bodies are preserved in `requests.jsonl`.",
        "",
    ]
    by_pair = {
        (sample.job.brief.brief_id, sample.job.route.route_id): sample
        for sample in samples
    }
    for index, brief in enumerate(briefs, start=1):
        representative = next(
            sample for sample in samples if sample.job.brief.brief_id == brief.brief_id
        )
        messages = representative.job.request.body["messages"]
        assert type(messages) is list and len(messages) == 2
        lines.extend(
            (
                f"## Prompt {index}: {', '.join(brief.required_words)}",
                "",
                f"Features: {', '.join(brief.requested_features) or 'none'}",
                "",
                "### System message",
                "",
                *_indented(_text(messages[0]["content"], "system message")),
                "",
                "### User message",
                "",
                *_indented(_text(messages[1]["content"], "user message")),
                "",
                "### Genuine GPT-4 TinyStories reference",
                "",
                *_indented(brief.matched_reference_text),
                "",
            )
        )
        for model in CANDIDATE_MODELS:
            sample = by_pair[(brief.brief_id, model.route_id)]
            story = (
                "[No schema-valid story returned]"
                if sample.payload is None
                else sample.payload.story
            )
            realized = (
                ()
                if sample.payload is None
                else realized_feature_labels(story, brief.requested_features)
            )
            lines.extend(
                (
                    f"### {model.route_id}",
                    "",
                    f"Accepted: `{sample.validation.accepted}` · "
                    f"words/evidence/schema: `{sample.validation.required_words_present}`/"
                    f"`{sample.validation.evidence_valid}`/`{sample.validation.schema_valid}` · "
                    f"realized features: `{', '.join(realized) or 'none'}` · "
                    f"tokens: `{sample.input_tokens} in / {sample.output_tokens} out` · "
                    f"cost: `${sample.billed_cost_usd}`",
                    "",
                    *_indented(story),
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def _preview_html(
    briefs: tuple[NeutralStoryBrief, ...],
    samples: Sequence[GeneratedSample],
) -> str:
    by_pair = {
        (sample.job.brief.brief_id, sample.job.route.route_id): sample
        for sample in samples
    }
    sections = []
    for index, brief in enumerate(briefs, start=1):
        representative = next(
            sample for sample in samples if sample.job.brief.brief_id == brief.brief_id
        )
        messages = representative.job.request.body["messages"]
        cards = []
        for model in CANDIDATE_MODELS:
            sample = by_pair[(brief.brief_id, model.route_id)]
            story = "[No schema-valid story returned]" if sample.payload is None else sample.payload.story
            cards.append(
                "<article><h3>"
                + html.escape(model.route_id)
                + "</h3><p><strong>Accepted:</strong> "
                + str(sample.validation.accepted)
                + " &middot; <strong>tokens:</strong> "
                + f"{sample.input_tokens} in / {sample.output_tokens} out"
                + " &middot; <strong>cost:</strong> $"
                + html.escape(sample.billed_cost_usd)
                + "</p><pre>"
                + html.escape(story)
                + "</pre></article>"
            )
        sections.append(
            f"<section><h2>Prompt {index}: {html.escape(', '.join(brief.required_words))}</h2>"
            f"<p><strong>Features:</strong> {html.escape(', '.join(brief.requested_features) or 'none')}</p>"
            "<details><summary>Exact messages and genuine reference</summary>"
            "<h3>System</h3><pre>"
            + html.escape(_text(messages[0]["content"], "system message"))
            + "</pre><h3>User</h3><pre>"
            + html.escape(_text(messages[1]["content"], "user message"))
            + "</pre><h3>Genuine GPT-4 TinyStories reference</h3><pre>"
            + html.escape(brief.matched_reference_text)
            + "</pre></details>"
            + "".join(cards)
            + "</section>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>TinyWorlds-v2 generator preview</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}"
        "section{border-top:2px solid #ddd;margin-top:2.5rem}article{background:#f7f7f8;border:1px solid #ddd;"
        "border-radius:8px;margin:1rem 0;padding:0 1rem}pre{white-space:pre-wrap;font:15px/1.45 Georgia,serif}"
        "summary{cursor:pointer;font-weight:700}</style></head><body>"
        "<h1>TinyWorlds-v2 generator route preview</h1>"
        "<p><strong>Diagnostic only.</strong> These 21 samples cannot enter the Phase 1 funnel or route selection.</p>"
        + "".join(sections)
        + "</body></html>\n"
    )


def _load_preview_manifest(root: Path) -> JsonObject:
    if not root.is_dir() or root.is_symlink():
        raise GeneratorPreviewError("preview root must be a regular directory")
    manifest = require_json_object(
        canonical_json_loads((root / _MANIFEST_FILE).read_bytes(), label="preview manifest"),
        label="preview manifest",
    )
    require_exact_fields(
        manifest,
        ("artifacts", "format", "manifest_sha256", "schema_version", "version"),
        label="preview manifest",
    )
    if (
        manifest["format"] != GENERATOR_PREVIEW_FORMAT
        or manifest["schema_version"] != GENERATOR_PREVIEW_SCHEMA_VERSION
        or manifest["version"] not in {
            GENERATOR_PREVIEW_ARCHIVE_VERSION,
            "tinyworlds-v2-phase1-route-preview-3x7-v2",
            GENERATOR_PREVIEW_VERSION,
        }
    ):
        raise GeneratorPreviewError("preview manifest identity differs")
    core: JsonObject = {
        "artifacts": manifest["artifacts"],
        "format": manifest["format"],
        "schema_version": manifest["schema_version"],
        "version": manifest["version"],
    }
    if manifest["manifest_sha256"] != json_sha256(core):
        raise GeneratorPreviewError("preview manifest self-digest differs")
    values = manifest["artifacts"]
    if type(values) is not list:
        raise GeneratorPreviewError("preview manifest artifacts must be an array")
    descriptors: dict[str, JsonObject] = {}
    for index, value in enumerate(values):
        record = require_json_object(value, label=f"preview artifact {index}")
        require_exact_fields(
            record,
            ("path", "record_count", "sha256", "size_bytes"),
            label=f"preview artifact {index}",
        )
        relative = _text(record["path"], "artifact path")
        if relative in descriptors:
            raise GeneratorPreviewError("preview manifest repeats an artifact path")
        descriptors[relative] = record
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != _MANIFEST_FILE
    }
    if actual != set(descriptors):
        raise GeneratorPreviewError("preview tree contains missing or unlisted files")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise GeneratorPreviewError("preview tree contains a symlink")
    for relative, descriptor in descriptors.items():
        payload = root.joinpath(*PurePosixPath(relative).parts).read_bytes()
        if (
            descriptor["sha256"] != bytes_sha256(payload)
            or descriptor["size_bytes"] != len(payload)
            or descriptor["record_count"] != _validate_artifact_payload(relative, payload)
        ):
            raise GeneratorPreviewError(f"preview artifact digest/size differs: {relative}")
    return manifest


def _preview_request_contract(version: str) -> GenerationRequestContract:
    """Resolve the exact request semantics recorded by a preview version."""
    contracts = {
        GENERATOR_PREVIEW_ARCHIVE_VERSION: GENERATION_REQUEST_V1,
        "tinyworlds-v2-phase1-route-preview-3x7-v2": SYNTHETIC_STORY_REQUEST_V2,
        GENERATOR_PREVIEW_VERSION: GENERATOR_PREVIEW_REQUEST_CONTRACT,
    }
    try:
        return contracts[version]
    except KeyError as error:
        raise GeneratorPreviewError("preview request contract is unknown") from error


def _validate_artifact_payload(relative: str, payload: bytes) -> int:
    if relative.endswith(".json"):
        canonical_json_loads(payload, label=relative)
        return 1
    if relative.endswith(".jsonl"):
        if payload and not payload.endswith(b"\n"):
            raise GeneratorPreviewError(f"preview JSONL lacks trailing newline: {relative}")
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            canonical_json_loads(line, label=f"{relative} line {index + 1}")
        return len(lines)
    return 0


def _load_briefs(path: Path) -> tuple[NeutralStoryBrief, ...]:
    briefs = []
    for record in _jsonl_objects(path):
        require_exact_fields(
            record,
            (
                "brief_id",
                "matched_reference_text",
                "prompt_text",
                "requested_features",
                "required_words",
                "source_record_id",
            ),
            label="preview brief",
        )
        briefs.append(
            NeutralStoryBrief(
                brief_id=_text(record["brief_id"], "brief_id"),
                source_record_id=_text(record["source_record_id"], "source_record_id"),
                prompt_text=_text(record["prompt_text"], "prompt_text"),
                required_words=_text_tuple(record["required_words"], "required_words"),
                requested_features=_text_tuple(
                    record["requested_features"], "requested_features"
                ),
                matched_reference_text=_text(
                    record["matched_reference_text"], "matched_reference_text"
                ),
            )
        )
    return tuple(briefs)


def _jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    if not path.is_file() or path.is_symlink():
        raise GeneratorPreviewError(f"preview JSONL is missing: {path}")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise GeneratorPreviewError(f"preview JSONL lacks trailing newline: {path}")
    return tuple(
        require_json_object(
            canonical_json_loads(line, label=f"{path.name} line {index + 1}"),
            label=f"{path.name} line {index + 1}",
        )
        for index, line in enumerate(payload.splitlines())
    )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise GeneratorPreviewError(f"{label} must be nonempty text")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise GeneratorPreviewError(f"{label} must be a text array")
    return tuple(value)


def _indented(text: str) -> tuple[str, ...]:
    return tuple(f"    {line}" if line else "" for line in text.strip().splitlines())


__all__ = [
    "GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST",
    "GENERATOR_PREVIEW_ARCHIVE_VERSION",
    "GENERATOR_PREVIEW_HARD_CAP_USD",
    "GENERATOR_PREVIEW_PRIOR_SPEND_USD",
    "GENERATOR_PREVIEW_REQUEST_CONTRACT",
    "GENERATOR_PREVIEW_RUN_CAP_USD",
    "GENERATOR_PREVIEW_SELECTION_SEED",
    "GENERATOR_PREVIEW_SOURCE_RECORD_IDS",
    "GENERATOR_PREVIEW_VERSION",
    "GeneratorPreviewError",
    "GeneratorPreviewPaths",
    "GeneratorPreviewValidation",
    "build_generator_preview_preflight",
    "load_generator_preview_briefs",
    "run_generator_preview",
    "validate_generator_preview",
    "verify_generator_preview_replay",
]
