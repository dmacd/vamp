"""Bounded paid-generation and GPU runner for the TinyWorlds-v2 LoRA sidebar."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from hashlib import sha256
import math
from pathlib import Path
import tempfile
from time import monotonic

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import flatten_lora_edge
from apm.data.text.tinyworlds_v2.bakeoff import assistant_message_content
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    CostPreflight,
    RouteWorkload,
    RuntimeCostLedger,
    TokenWorkload,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    RawHttpResponse,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    canonical_json_loads,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.openrouter import OpenRouterClient
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    Phase1ArtifactManifest,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.reasoning_sidebar import (
    REASONING_SIDEBAR_BATCH_SIZE,
    REASONING_SIDEBAR_GENERATION_VARIANTS,
    REASONING_SIDEBAR_HARD_CAP_USD,
    REASONING_SIDEBAR_LORA_RANK,
    REASONING_SIDEBAR_MAX_OUTPUT_TOKENS,
    REASONING_SIDEBAR_STORIES_PER_EVIDENCE,
    REASONING_SIDEBAR_UPDATE_BUDGET,
    REASONING_SIDEBAR_VERSION,
    SidebarReferenceRecord,
    SidebarScoreSummary,
    SidebarStoryPlan,
    SidebarStoryValidation,
    build_sidebar_author_requests,
    build_sidebar_evidence_plans,
    build_sidebar_queries,
    build_sidebar_reference_control,
    build_sidebar_training_batches,
    reasoning_sidebar_world,
    select_sidebar_training_stories,
    sidebar_query_score_records,
    summarize_sidebar_scores,
    validate_sidebar_story,
)
from apm.data.text.tinyworlds_v2.route_lock import validate_route_semantics
from apm.data.text.tinyworlds_v2.two_route_bakeoff import (
    TwoRouteDependencies,
    TwoRoutePaths,
    production_two_route_dependencies,
)
from apm.lm.candidate_scoring import (
    score_edge_coefficient_candidates,
    score_frozen_base_candidates,
)
from apm.lm.checkpoint import parameter_checksum
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    pack_lora_memory,
    packed_with_candidate_edge,
)
from apm.lm.text import TokenizersTextTokenizer
from apm.lm.text_data import TokenBatch
from apm.lm.tinystories_conversion import (
    LoadedTinyStoriesArtifact,
    load_tinystories_artifact,
)
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    init_candidate_lora_train_state,
)
from apm.lm.workflow import (
    evaluate_normalized_nll,
    run_resumable_candidate_edge_updates,
)
from apm.memory.graph import NodeId, init_memory_graph


REASONING_SIDEBAR_REFERENCE_MANIFEST_SHA256 = (
    "50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74"
)
REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE = 8
REASONING_SIDEBAR_GENERATION_WORKERS = 8
REASONING_SIDEBAR_REFERENCE_PATH = (
    "data/tinyworlds-v2/prompt-tuning-v3/comparator/records.jsonl"
)


@dataclass(frozen=True, slots=True)
class ReasoningSidebarPaths:
    """All fixed local inputs, caches, and publication paths for the sidebar."""

    repository_root: Path
    checkpoint: Path
    tokenizer: Path
    reference_artifact: Path
    reference_records: Path
    raw_cache: Path
    destination: Path

    @classmethod
    def from_repository(cls, repository_root: str | Path) -> "ReasoningSidebarPaths":
        """Resolve the one supported sidebar layout beneath a repository root."""
        root = Path(repository_root).resolve()
        data_root = root / "data" / "tinyworlds-v2"
        base = root / "checkpoints" / "tinystories-8m"
        reference_artifact = data_root / "prompt-tuning-v3"
        return cls(
            repository_root=root,
            checkpoint=base,
            tokenizer=base / "tokenizer" / "tokenizer.json",
            reference_artifact=reference_artifact,
            reference_records=reference_artifact / "comparator" / "records.jsonl",
            # The earlier pre-POST implementation attempt wrote per-token
            # max-price plans into ``reasoning-sidebar-v1``.  Keep those
            # immutable failed plans separate from this corrected per-million
            # request contract so crash reconciliation never mixes units.
            raw_cache=data_root / "cache" / "reasoning-sidebar-v1-price-units-v2",
            destination=data_root / "reasoning-sidebar-v1",
        )


@dataclass(frozen=True, slots=True)
class AuthorGeneration:
    """One exact paid response and its locally derived training eligibility."""

    route_id: str
    plan: SidebarStoryPlan
    request: CanonicalRequest
    response: RawHttpResponse
    story: str
    validation: SidebarStoryValidation


@dataclass(frozen=True, slots=True)
class SidebarArmCorpus:
    """One equal-sized, evidence-aligned collection of training stories."""

    arm_id: str
    stories: tuple[str, ...]
    records: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class SidebarArmResult:
    """Final LoRA state and all fixed validation/test measurements for one arm."""

    arm_id: str
    state: LmTrainState[LoraEdge]
    adapter_checksum: str
    frozen_corpus_nll: float
    adapted_corpus_nll: float
    validation_scores: np.ndarray
    test_scores: np.ndarray
    validation_summaries: JsonObject
    test_summaries: JsonObject
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ReasoningSidebarRunResult:
    """Published sidebar identity and its paid/runtime headline values."""

    directory: Path
    manifest_sha256: str
    actual_cost_usd: str
    elapsed_seconds: float


def run_reasoning_sidebar(
    staging_directory: str | Path,
    paths: ReasoningSidebarPaths,
    *,
    emit: Callable[[str], None] = print,
) -> ReasoningSidebarRunResult:
    """Generate paired corpora, train three identical LoRAs, and publish evidence."""
    staging = Path(staging_directory)
    if not staging.is_dir():
        raise FileNotFoundError(f"sidebar staging directory is missing: {staging}")
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"sidebar destination exists: {paths.destination}")
    started = monotonic()
    builder = Phase1ArtifactBuilder(staging, version=REASONING_SIDEBAR_VERSION)
    _progress(builder, "run_started")

    emit("Sidebar phase 1/5: loading the frozen model, tokenizer, and clean control prose.")
    base_artifact = load_tinystories_artifact(paths.checkpoint)
    tokenizer = TokenizersTextTokenizer.from_file(paths.tokenizer)
    reference_records = _load_reference_records(paths)
    control = build_sidebar_reference_control(reference_records, tokenizer)
    validation_queries = build_sidebar_queries(tokenizer, "validation")
    test_queries = build_sidebar_queries(tokenizer, "test")
    control_corpus = SidebarArmCorpus(
        arm_id="tinystories-control",
        stories=tuple(item[1] for item in control),
        records=tuple(
            _selected_story_record(
                "tinystories-control",
                item[0],
                item[1],
                item[2],
                source_record_id=item[3],
            )
            for item in control
        ),
    )
    _write_offline_plan(
        builder,
        paths,
        base_artifact,
        validation_queries,
        test_queries,
        control_corpus,
    )
    _progress(builder, "offline_preparation_completed")

    emit("Sidebar phase 2/5: resolving pinned Qwen/GPT routes and pricing 72 calls.")
    dependencies = production_two_route_dependencies(
        replace(
            TwoRoutePaths.from_repository(paths.repository_root),
            raw_cache=paths.raw_cache,
            destination=paths.destination,
        )
    )
    catalog_payloads, routes = dependencies.fetch_routes()
    requests = build_sidebar_author_requests(routes)
    preflight = _build_sidebar_cost_preflight(requests, routes, dependencies.encode_text)
    builder.write_bytes("catalog/models.response", catalog_payloads.models)
    builder.write_json("routes.json", {"routes": [route.as_record() for route in routes]})
    builder.write_bytes(
        "requests.jsonl",
        canonical_jsonl_bytes(
            _request_record(route_id, plan, route, request)
            for route_id, plan, route, request in requests
        ),
    )
    builder.write_json("cost_estimates.json", preflight.as_record())
    emit(
        "Cost preflight: "
        f"expected ${preflight.expected_usd}; conservative "
        f"${preflight.conservative_usd} / ${preflight.hard_cap_usd} cap."
    )
    enforce_cost_cap(preflight)
    _progress(builder, "cost_preflight_completed")

    emit("Sidebar phase 3/5: generating paired evidence-bearing stories.")
    generations, actual_cost = _generate_sidebar_corpora(
        builder,
        paths,
        dependencies,
        requests,
        routes,
        tokenizer,
    )
    authored_corpora = _author_corpora(generations)
    corpora = (control_corpus, *authored_corpora)
    builder.write_bytes(
        "selected_stories.jsonl",
        canonical_jsonl_bytes(
            record for corpus in corpora for record in corpus.records
        ),
    )
    _progress(builder, "paid_generation_completed")

    emit(
        "Sidebar phase 4/5: training three rank-8 LoRAs for 512 updates each "
        "from the same initialization."
    )
    arm_results, frozen_scores, resource = _train_sidebar_arms(
        builder,
        base_artifact,
        tokenizer,
        corpora,
        validation_queries,
        test_queries,
        emit,
    )
    _write_score_artifacts(
        builder,
        validation_queries,
        test_queries,
        frozen_scores,
        arm_results,
    )
    _progress(builder, "training_completed")

    emit("Sidebar phase 5/5: validating and atomically promoting the result.")
    elapsed = monotonic() - started
    result_record = _result_record(
        actual_cost,
        elapsed,
        frozen_scores,
        validation_queries,
        test_queries,
        arm_results,
        resource,
    )
    builder.write_json("results.json", result_record)
    builder.write_json("cost_actuals.json", {"actual_billed_usd": actual_cost})
    builder.write_bytes(
        "report.md",
        _render_report(result_record, corpora).encode("utf-8"),
    )
    _progress(builder, "run_completed")
    manifest = builder.finalize()
    validated = validate_reasoning_sidebar(staging)
    if validated.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("sidebar validation changed the artifact identity")
    destination = builder.promote(
        paths.destination,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    return ReasoningSidebarRunResult(
        directory=destination,
        manifest_sha256=manifest.manifest_sha256,
        actual_cost_usd=actual_cost,
        elapsed_seconds=elapsed,
    )


def validate_reasoning_sidebar(
    directory: str | Path,
) -> Phase1ArtifactManifest:
    """Strictly validate the sidebar tree and its cross-file scientific coverage."""
    root = Path(directory)
    manifest = load_phase1_artifact_tree(root)
    configuration = _json_object(root / "configuration.json")
    if configuration.get("version") != REASONING_SIDEBAR_VERSION:
        raise ValueError("sidebar configuration version changed")
    results = _json_object(root / "results.json")
    if results.get("status") != "completed":
        raise ValueError("sidebar result is not complete")
    arms = results.get("arms")
    if type(arms) is not dict or tuple(arms) != (
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
        "tinystories-control",
    ):
        raise ValueError("sidebar result arm coverage changed")
    _require_finite_numbers(results)
    generation_records = _jsonl_objects(root / "generation_results.jsonl")
    expected_generation_count = (
        2
        * len(build_sidebar_evidence_plans())
        * REASONING_SIDEBAR_GENERATION_VARIANTS
    )
    if len(generation_records) != expected_generation_count:
        raise ValueError("sidebar paid generation coverage changed")
    selected = _jsonl_objects(root / "selected_stories.jsonl")
    expected_selected = (
        3
        * len(build_sidebar_evidence_plans())
        * REASONING_SIDEBAR_STORIES_PER_EVIDENCE
    )
    if len(selected) != expected_selected:
        raise ValueError("sidebar selected-story coverage changed")
    counts = {
        arm_id: sum(record.get("arm_id") == arm_id for record in selected)
        for arm_id in arms
    }
    if set(counts.values()) != {expected_selected // 3}:
        raise ValueError("sidebar arms do not contain equal story counts")
    if any(
        type(record.get("story")) is not str
        or not record["story"].lstrip().startswith(str(record.get("exact_sentence")))
        for record in selected
    ):
        raise ValueError("sidebar selected story lost its leading evidence")
    query_scores = _jsonl_objects(root / "scores" / "test.jsonl")
    if len(query_scores) != 4 * 32:
        raise ValueError("sidebar test-score method/query coverage changed")
    if {
        record.get("method") for record in query_scores
    } != {"frozen", *arms.keys()}:
        raise ValueError("sidebar test score methods changed")
    return manifest


def _load_reference_records(
    paths: ReasoningSidebarPaths,
) -> tuple[SidebarReferenceRecord, ...]:
    manifest = load_phase1_artifact_tree(paths.reference_artifact)
    if manifest.manifest_sha256 != REASONING_SIDEBAR_REFERENCE_MANIFEST_SHA256:
        raise ValueError("sidebar control requires the pinned decontaminated comparator")
    records = _jsonl_objects(paths.reference_records)
    return tuple(
        SidebarReferenceRecord(
            record_id=_text(record.get("record_id"), "reference record ID"),
            story=_text(record.get("story"), "reference story"),
        )
        for record in records
    )


def _write_offline_plan(
    builder: Phase1ArtifactBuilder,
    paths: ReasoningSidebarPaths,
    base_artifact: LoadedTinyStoriesArtifact,
    validation_queries: Sequence[object],
    test_queries: Sequence[object],
    control_corpus: SidebarArmCorpus,
) -> None:
    facts, rules = reasoning_sidebar_world()
    checkpoint = base_artifact.checkpoint
    builder.write_json(
        "configuration.json",
        {
            "base_manifest_sha256": checkpoint.reference.manifest_sha256,
            "base_parameter_checksum": checkpoint.reference.parameter_checksum,
            "batch_size": REASONING_SIDEBAR_BATCH_SIZE,
            "context_length": 256,
            "evaluation_microbatch_size": REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
            "generation_variants_per_evidence": REASONING_SIDEBAR_GENERATION_VARIANTS,
            "lora_alpha": float(REASONING_SIDEBAR_LORA_RANK),
            "lora_rank": REASONING_SIDEBAR_LORA_RANK,
            "optimizer": {
                "gradient_clip_norm": 1.0,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            },
            "reference_manifest_sha256": REASONING_SIDEBAR_REFERENCE_MANIFEST_SHA256,
            "reference_records_path": REASONING_SIDEBAR_REFERENCE_PATH,
            "selected_stories_per_evidence": REASONING_SIDEBAR_STORIES_PER_EVIDENCE,
            "test_query_count": len(test_queries),
            "update_budget": REASONING_SIDEBAR_UPDATE_BUDGET,
            "validation_query_count": len(validation_queries),
            "version": REASONING_SIDEBAR_VERSION,
        },
    )
    builder.write_json(
        "world.json",
        {
            "facts": [asdict(fact) for fact in facts],
            "rules": [asdict(rule) for rule in rules],
            "training_conclusion_policy": (
                "no named child and derived meeting place co-occur in training"
            ),
        },
    )
    builder.write_bytes(
        "control_stories.jsonl",
        canonical_jsonl_bytes(control_corpus.records),
    )
    builder.write_json(
        "base_identity.json",
        {
            "artifact_directory": str(paths.checkpoint.relative_to(paths.repository_root)),
            "manifest_sha256": checkpoint.reference.manifest_sha256,
            "parameter_checksum": parameter_checksum(
                checkpoint.params,
                checkpoint.config,
            ),
            "tokenizer_sha256": sha256(paths.tokenizer.read_bytes()).hexdigest(),
        },
    )


def _build_sidebar_cost_preflight(
    requests: Sequence[tuple[str, SidebarStoryPlan, RouteLock, CanonicalRequest]],
    routes: Sequence[RouteLock],
    encode_text: Callable[[str], tuple[int, ...]],
) -> CostPreflight:
    workloads = tuple(
        RouteWorkload(
            route,
            TokenWorkload(
                label="reasoning-sidebar-authored-corpus",
                request_count=len(route_requests),
                input_tokens_per_request=math.ceil(
                    sum(len(encode_text(item[3].body_json)) for item in route_requests)
                    / len(route_requests)
                ),
                output_tokens_per_request=200,
                conservative_input_tokens_per_request=max(
                    2 * len(encode_text(item[3].body_json)) + 512
                    for item in route_requests
                ),
                conservative_output_tokens_per_request=(
                    REASONING_SIDEBAR_MAX_OUTPUT_TOKENS
                ),
                retry_allowance_basis_points=10_000,
            ),
        )
        for route in routes
        for route_requests in (
            tuple(item for item in requests if item[0] == route.route_id),
        )
    )
    return build_cost_preflight(
        workloads,
        hard_cap_usd=REASONING_SIDEBAR_HARD_CAP_USD,
    )


def _generate_sidebar_corpora(
    builder: Phase1ArtifactBuilder,
    paths: ReasoningSidebarPaths,
    dependencies: TwoRouteDependencies,
    requests: Sequence[tuple[str, SidebarStoryPlan, RouteLock, CanonicalRequest]],
    routes: Sequence[RouteLock],
    tokenizer: TokenizersTextTokenizer,
) -> tuple[tuple[AuthorGeneration, ...], str]:
    from tqdm.auto import tqdm

    with exclusive_paid_run_lock(paths.raw_cache):
        fresh_routes = tuple(dependencies.revalidate_route(route) for route in routes)
        for locked, fresh in zip(routes, fresh_routes, strict=True):
            validate_route_semantics(locked, fresh)
        fresh_by_id = {route.route_id: route for route in fresh_routes}
        cache = ImmutableRawCache(paths.raw_cache)
        client = dependencies.make_client(dependencies.load_api_key(), cache)
        if not isinstance(client, OpenRouterClient):
            raise TypeError("production sidebar requires the OpenRouter client")
        client = replace(
            client,
            cost_ledger=RuntimeCostLedger(REASONING_SIDEBAR_HARD_CAP_USD),
        )
        client.cost_ledger.bootstrap(cache, fresh_routes)
        if client.require_byok_preflight:
            builder.write_json("byok_preflight.json", client.verify_no_byok().as_record())
        bar = tqdm(
            total=len(requests),
            desc="Phase 3/5 paid stories",
            unit="story",
            dynamic_ncols=True,
            leave=True,
        )
        ordered: list[AuthorGeneration | None] = [None] * len(requests)

        def execute(index: int) -> tuple[int, AuthorGeneration]:
            route_id, plan, _route, request = requests[index]
            response = client.generate(request, fresh_by_id[route_id])
            story = assistant_message_content(response.body)
            return index, AuthorGeneration(
                route_id=route_id,
                plan=plan,
                request=request,
                response=response,
                story=story,
                validation=validate_sidebar_story(plan, story, tokenizer),
            )

        try:
            with ThreadPoolExecutor(
                max_workers=REASONING_SIDEBAR_GENERATION_WORKERS
            ) as executor:
                futures = tuple(executor.submit(execute, index) for index in range(len(requests)))
                for future in as_completed(futures):
                    index, generation = future.result()
                    ordered[index] = generation
                    bar.update()
        finally:
            bar.close()
        if any(generation is None for generation in ordered):
            raise RuntimeError("sidebar generation executor lost a result")
        generations = tuple(
            generation for generation in ordered if generation is not None
        )
        builder.write_bytes(
            "generation_results.jsonl",
            canonical_jsonl_bytes(_generation_record(item) for item in generations),
        )
        ledger = client.cost_ledger.snapshot()
        builder.write_json("runtime_cost_ledger.json", ledger.as_record())
        actual_cost = ledger.provider_reported_actual_usd
        if Decimal(actual_cost) != sum(
            (Decimal(item.response.billed_cost_usd or "0") for item in generations),
            Decimal(0),
        ):
            raise ValueError("sidebar billed results differ from the runtime ledger")
        return generations, actual_cost


def _author_corpora(
    generations: Sequence[AuthorGeneration],
) -> tuple[SidebarArmCorpus, ...]:
    return tuple(
        SidebarArmCorpus(
            arm_id=route_id,
            stories=tuple(item[1] for item in selected),
            records=tuple(
                _selected_story_record(
                    route_id,
                    item[0],
                    item[1],
                    item[2],
                    source_record_id=None,
                )
                for item in selected
            ),
        )
        for route_id in ("qwen3.5-35b-a3b", "gpt-5.4-mini")
        for candidates in (
            tuple(
                (item.plan, item.story, item.validation)
                for item in generations
                if item.route_id == route_id
            ),
        )
        for selected in (select_sidebar_training_stories(candidates),)
    )


def _train_sidebar_arms(
    builder: Phase1ArtifactBuilder,
    base_artifact: LoadedTinyStoriesArtifact,
    tokenizer: TokenizersTextTokenizer,
    corpora: Sequence[SidebarArmCorpus],
    validation_queries,
    test_queries,
    emit: Callable[[str], None],
) -> tuple[
    tuple[SidebarArmResult, ...],
    tuple[np.ndarray, np.ndarray],
    JsonObject,
]:
    checkpoint = base_artifact.checkpoint
    devices = jax.local_devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError("the reasoning sidebar requires the local GPU")
    device = devices[0]
    lora_config = LoraConfig(
        rank=REASONING_SIDEBAR_LORA_RANK,
        alpha=float(REASONING_SIDEBAR_LORA_RANK),
    )
    train_config = LmTrainConfig(
        learning_rate=1e-3,
        steps=REASONING_SIDEBAR_UPDATE_BUDGET,
        batch_size=REASONING_SIDEBAR_BATCH_SIZE,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
    )
    empty_graph = init_memory_graph(NodeId("root"))
    packed = pack_lora_memory(
        empty_graph,
        checkpoint.config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )
    frozen_validation = score_frozen_base_candidates(
        checkpoint.params,
        checkpoint.config,
        tuple(validation_queries),
        evaluation_microbatch_size=REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
    )
    frozen_test = score_frozen_base_candidates(
        checkpoint.params,
        checkpoint.config,
        tuple(test_queries),
        evaluation_microbatch_size=REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
    )
    initial_edge = init_lora_edge(
        jax.random.PRNGKey(0x51DEBA),
        checkpoint.config,
        lora_config,
    )
    initial_training_key = jax.random.PRNGKey(0xC0FFEE)
    results = tuple(
        _train_one_arm(
            builder,
            corpus,
            build_sidebar_training_batches(corpus.stories, tokenizer),
            checkpoint.params,
            checkpoint.config,
            lora_config,
            train_config,
            packed,
            initial_edge,
            initial_training_key,
            validation_queries,
            test_queries,
            emit,
        )
        for corpus in corpora
    )
    stats = device.memory_stats() or {}
    resource = {
        "allocator_peak_bytes": (
            int(stats["peak_bytes_in_use"]) if "peak_bytes_in_use" in stats else None
        ),
        "device_kind": device.device_kind,
        "platform": device.platform,
    }
    return results, (frozen_validation, frozen_test), resource


def _train_one_arm(
    builder: Phase1ArtifactBuilder,
    corpus: SidebarArmCorpus,
    batches: tuple[TokenBatch, ...],
    base_params,
    model_config,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    packed: PackedLoraMemory,
    initial_edge: LoraEdge,
    initial_training_key: jax.Array,
    validation_queries,
    test_queries,
    emit: Callable[[str], None],
) -> SidebarArmResult:
    started = monotonic()
    emit(f"  Training {corpus.arm_id} (1 batch, 24 stories, 512 updates).")
    state = init_candidate_lora_train_state(
        initial_edge,
        initial_training_key,
        train_config,
    )
    frozen_corpus_nll = evaluate_normalized_nll(
        base_params,
        model_config,
        batches,
    )

    def score(adapter: LoraEdge, update: int) -> tuple[float, float]:
        scores = _score_adapter(
            adapter,
            base_params,
            model_config,
            packed,
            lora_config,
            validation_queries,
        )
        summaries = _score_summaries(validation_queries, scores)
        overall = summaries["overall"]
        if type(overall) is not dict:
            raise TypeError("sidebar overall summary is malformed")
        builder.append_jsonl(
            f"training/{corpus.arm_id}/learning_curve.jsonl",
            {
                "adapter_checksum": _adapter_checksum(adapter, model_config, lora_config),
                "summaries": summaries,
                "update": update,
            },
        )
        emit(
            f"    update {update:>3}/512: validation accuracy "
            f"{100 * float(overall['accuracy']):5.1f}%"
        )
        return float(overall["accuracy"]), float(overall["correct_nll"])

    state, trace, checkpoints = run_resumable_candidate_edge_updates(
        state,
        batches,
        base_params,
        model_config,
        packed,
        lora_config,
        jnp.zeros((1,), dtype=jnp.float32),
        0,
        train_config,
        validation_function=score,
    )
    builder.write_bytes(
        f"training/{corpus.arm_id}/checkpoint_losses.jsonl",
        canonical_jsonl_bytes(
            {
                "training_loss": checkpoint.training_loss,
                "update": checkpoint.update,
                "validation_candidate_accuracy": (
                    checkpoint.validation_candidate_accuracy
                ),
                "validation_correct_nll": checkpoint.validation_correct_nll,
            }
            for checkpoint in checkpoints
        ),
    )
    builder.write_json(
        f"training/{corpus.arm_id}/trace_summary.json",
        {
            "final_loss": trace.step_losses[-1],
            "first_loss": trace.step_losses[0],
            "mean_last_32_loss": float(np.mean(trace.step_losses[-32:])),
            "step_count": len(trace.step_losses),
        },
    )
    adapter_memory = packed_with_candidate_edge(packed, state.trainable, 0)
    adapted_corpus_nll = evaluate_normalized_nll(
        base_params,
        model_config,
        batches,
        lora_memory=adapter_memory,
        edge_coefficients=jnp.ones((1,), dtype=jnp.float32),
        lora_config=lora_config,
    )
    validation_scores = _score_adapter(
        state.trainable,
        base_params,
        model_config,
        packed,
        lora_config,
        validation_queries,
    )
    test_scores = _score_adapter(
        state.trainable,
        base_params,
        model_config,
        packed,
        lora_config,
        test_queries,
    )
    adapter_checksum = _adapter_checksum(state.trainable, model_config, lora_config)
    _write_adapter(
        builder,
        corpus.arm_id,
        state.trainable,
        adapter_checksum,
        model_config,
        lora_config,
        parameter_checksum(base_params, model_config),
    )
    return SidebarArmResult(
        arm_id=corpus.arm_id,
        state=state,
        adapter_checksum=adapter_checksum,
        frozen_corpus_nll=frozen_corpus_nll,
        adapted_corpus_nll=adapted_corpus_nll,
        validation_scores=validation_scores,
        test_scores=test_scores,
        validation_summaries=_score_summaries(validation_queries, validation_scores),
        test_summaries=_score_summaries(test_queries, test_scores),
        elapsed_seconds=monotonic() - started,
    )


def _score_adapter(
    adapter: LoraEdge,
    base_params,
    model_config,
    packed: PackedLoraMemory,
    lora_config: LoraConfig,
    queries,
) -> np.ndarray:
    memory = packed_with_candidate_edge(packed, adapter, 0)
    return score_edge_coefficient_candidates(
        base_params,
        model_config,
        memory,
        lora_config,
        tuple(queries),
        np.ones((len(queries), 1), dtype=np.float32),
        evaluation_microbatch_size=REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
    )


def _score_summaries(queries, scores: np.ndarray) -> JsonObject:
    query_tuple = tuple(queries)
    groups = {
        "overall": tuple(range(len(query_tuple))),
        "direct": tuple(
            index
            for index, query in enumerate(query_tuple)
            if query.reasoning_type == "direct"
        ),
        "one_hop": tuple(
            index
            for index, query in enumerate(query_tuple)
            if query.reasoning_type == "one_hop"
        ),
    }
    return {
        label: _summary_record(
            summarize_sidebar_scores(
                tuple(query_tuple[index] for index in indices),
                scores[np.asarray(indices)],
            )
        )
        for label, indices in groups.items()
    }


def _write_score_artifacts(
    builder: Phase1ArtifactBuilder,
    validation_queries,
    test_queries,
    frozen_scores: tuple[np.ndarray, np.ndarray],
    arm_results: Sequence[SidebarArmResult],
) -> None:
    for split, queries, frozen, adapted in (
        (
            "validation",
            validation_queries,
            frozen_scores[0],
            tuple(result.validation_scores for result in arm_results),
        ),
        (
            "test",
            test_queries,
            frozen_scores[1],
            tuple(result.test_scores for result in arm_results),
        ),
    ):
        methods = (("frozen", frozen),) + tuple(
            (result.arm_id, scores)
            for result, scores in zip(arm_results, adapted, strict=True)
        )
        builder.write_bytes(
            f"scores/{split}.jsonl",
            canonical_jsonl_bytes(
                {"method": method, **record}
                for method, scores in methods
                for record in sidebar_query_score_records(queries, scores)
            ),
        )


def _result_record(
    actual_cost: str,
    elapsed_seconds: float,
    frozen_scores: tuple[np.ndarray, np.ndarray],
    validation_queries,
    test_queries,
    arm_results: Sequence[SidebarArmResult],
    resource: JsonObject,
) -> JsonObject:
    return {
        "actual_billed_usd": actual_cost,
        "arms": {
            result.arm_id: {
                "adapted_corpus_nll": result.adapted_corpus_nll,
                "adapter_checksum": result.adapter_checksum,
                "elapsed_seconds": result.elapsed_seconds,
                "frozen_corpus_nll": result.frozen_corpus_nll,
                "test": result.test_summaries,
                "validation": result.validation_summaries,
            }
            for result in sorted(arm_results, key=lambda value: value.arm_id)
        },
        "elapsed_seconds": elapsed_seconds,
        "frozen": {
            "test": _score_summaries(test_queries, frozen_scores[1]),
            "validation": _score_summaries(validation_queries, frozen_scores[0]),
        },
        "resource": resource,
        "status": "completed",
        "version": REASONING_SIDEBAR_VERSION,
    }


def _render_report(
    result: JsonObject,
    corpora: Sequence[SidebarArmCorpus],
) -> str:
    arms = result["arms"]
    frozen = result["frozen"]
    if type(arms) is not dict or type(frozen) is not dict:
        raise TypeError("sidebar report result is malformed")
    frozen_test = frozen["test"]
    if type(frozen_test) is not dict:
        raise TypeError("sidebar frozen test result is malformed")
    table_rows = tuple(
        "| "
        + " | ".join(
            (
                arm_id,
                f"{float(record['frozen_corpus_nll']):.3f}",
                f"{float(record['adapted_corpus_nll']):.3f}",
                _percent(_nested(record, "test", "direct", "accuracy")),
                _percent(_nested(record, "test", "direct", "paired_consistency")),
                _percent(_nested(record, "test", "one_hop", "accuracy")),
                _percent(_nested(record, "test", "one_hop", "paired_consistency")),
            )
        )
        + " |"
        for arm_id, record in arms.items()
        if type(record) is dict
    )
    control = arms["tinystories-control"]
    if type(control) is not dict:
        raise TypeError("sidebar control result is malformed")
    control_direct = _nested(control, "test", "direct", "accuracy")
    control_hop = _nested(control, "test", "one_hop", "accuracy")
    interpretation = (
        "The in-distribution control did not reach 75% direct recall, so this "
        "small task does not establish a usable learnability baseline; author "
        "mismatch cannot be separated from a weak LoRA/task setup."
        if control_direct < 0.75
        else (
            "The in-distribution control learned the direct bindings. Compare each "
            "author's direct-recall gap to that control to estimate the practical "
            "penalty from its surrounding prose."
        )
    )
    hop_interpretation = (
        "The control also crossed 50% one-hop accuracy, so author/control gaps on "
        "that column are evidence about compositional use of learned facts and rules."
        if control_hop >= 0.5
        else "The control stayed below 50% one-hop accuracy, so one-hop author gaps are inconclusive."
    )
    samples = tuple(
        f"### {corpus.arm_id}\n\n{corpus.stories[0]}"
        for corpus in corpora
    )
    return "\n".join(
        (
            "# TinyWorlds-v2 LoRA learnability sidebar",
            "",
            "This is an exploratory diagnostic, not a Phase 1 gate. Each arm saw the "
            "same eight arbitrary child→badge facts and four badge→place rules, the "
            "same 24-document count, the same rank-8 LoRA initialization, and the "
            "same 512-update schedule. Only the prose following each exact leading "
            "evidence sentence changed.",
            "",
            "| training context | frozen corpus NLL | adapted corpus NLL | direct accuracy | direct both-paraphrase | one-hop accuracy | one-hop both-paraphrase |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            f"The frozen model scored {_percent(_nested(frozen_test, 'direct', 'accuracy'))} "
            f"on direct probes and {_percent(_nested(frozen_test, 'one_hop', 'accuracy'))} "
            "on one-hop probes; four choices make chance 25%.",
            "",
            interpretation,
            "",
            hop_interpretation,
            "",
            "Here, **direct recall** means retrieving an explicit statement such as "
            "‘Mia's club badge was red.’ **One-hop** means combining that fact with "
            "one learned rule—‘every red badge meant meeting by the pond’—to answer "
            "where Mia meets. No sentence directly pairing a named child with the "
            "derived place appeared in training.",
            "",
            "The test contains eight semantic children with two held-out query "
            "phrasings each. Those 16 rows are not 16 independent facts, so this "
            "sidebar should be read as a mechanism check and effect-size estimate, "
            "not a high-powered statistical result. Because the evidence clause is "
            "canonical and leading in every arm, the test isolates interference from "
            "the author-specific continuation; it does not prove that unconstrained "
            "facts can always be extracted from arbitrary generated prose.",
            "",
            "## Representative selected training stories",
            "",
            *samples,
            "",
        )
    )


def _request_record(
    route_id: str,
    plan: SidebarStoryPlan,
    route: RouteLock,
    request: CanonicalRequest,
) -> JsonObject:
    return {
        **request.as_record(),
        "body": request.body,
        "evidence_id": plan.evidence.evidence_id,
        "route": route.as_record(),
        "route_id": route_id,
        "story_plan_id": plan.story_plan_id,
        "variant_index": plan.variant_index,
    }


def _generation_record(generation: AuthorGeneration) -> JsonObject:
    response = generation.response
    if response.provenance is None or response.usage is None:
        raise ValueError("sidebar generation lacks provenance or usage")
    return {
        "billed_cost_usd": response.billed_cost_usd,
        "evidence_id": generation.plan.evidence.evidence_id,
        "generation_id": response.provenance.generation_id,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "request_sha256": generation.request.request_sha256,
        "route_id": generation.route_id,
        "story": generation.story,
        "story_plan_id": generation.plan.story_plan_id,
        "validation": _validation_record(generation.validation),
        "variant_index": generation.plan.variant_index,
    }


def _selected_story_record(
    arm_id: str,
    plan: SidebarStoryPlan,
    story: str,
    validation: SidebarStoryValidation,
    *,
    source_record_id: str | None,
) -> JsonObject:
    return {
        "arm_id": arm_id,
        "evidence_id": plan.evidence.evidence_id,
        "evidence_kind": plan.evidence.kind,
        "exact_sentence": plan.evidence.exact_sentence,
        "source_record_id": source_record_id,
        "story": story,
        "story_plan_id": plan.story_plan_id,
        "validation": _validation_record(validation),
        "variant_index": plan.variant_index,
    }


def _validation_record(validation: SidebarStoryValidation) -> JsonObject:
    return {
        "accepted": validation.accepted,
        "rejection_reasons": list(validation.rejection_reasons),
        "story_sha256": validation.story_sha256,
        "token_count": validation.token_count,
        "word_count": validation.word_count,
    }


def _summary_record(summary: SidebarScoreSummary) -> JsonObject:
    return {
        "accuracy": summary.accuracy,
        "correct_nll": summary.correct_nll,
        "margin": summary.margin,
        "paired_consistency": summary.paired_consistency,
        "query_count": summary.query_count,
    }


def _adapter_checksum(adapter: LoraEdge, model_config, lora_config: LoraConfig) -> str:
    digest = sha256()
    for name, value in sorted(flatten_lora_edge(adapter, model_config, lora_config).items()):
        array = np.asarray(value, dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_adapter(
    builder: Phase1ArtifactBuilder,
    arm_id: str,
    adapter: LoraEdge,
    adapter_checksum: str,
    model_config,
    lora_config: LoraConfig,
    base_parameter_checksum: str,
) -> None:
    """Persist one final LoRA as a checksummed, base-referenced safetensors file."""
    from safetensors.numpy import save_file

    tensors = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in sorted(
            flatten_lora_edge(adapter, model_config, lora_config).items()
        )
    }
    path = builder.root / "adapters" / arm_id / "adapter.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"sidebar adapter already exists: {path}")
    save_file(
        tensors,
        str(path),
        metadata={
            "adapter_checksum": adapter_checksum,
            "arm_id": arm_id,
            "base_parameter_checksum": base_parameter_checksum,
            "format": "apm.tinyworlds-v2.reasoning-sidebar-lora",
            "version": REASONING_SIDEBAR_VERSION,
        },
    )
    builder.write_json(
        f"adapters/{arm_id}/metadata.json",
        {
            "adapter_checksum": adapter_checksum,
            "arm_id": arm_id,
            "base_parameter_checksum": base_parameter_checksum,
            "lora_alpha": lora_config.alpha,
            "lora_rank": lora_config.rank,
            "tensor_count": len(tensors),
        },
    )


def _progress(builder: Phase1ArtifactBuilder, event: str) -> None:
    builder.append_jsonl("progress.jsonl", {"event": event})


def _json_object(path: Path) -> JsonObject:
    return require_json_object(canonical_json_loads(path.read_bytes(), label=str(path)), label=str(path))


def _jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    return tuple(
        require_json_object(
            canonical_json_loads(line.rstrip(b"\n"), label=f"{path} line {index}"),
            label=f"{path} line {index}",
        )
        for index, line in enumerate(path.read_bytes().splitlines(keepends=True), start=1)
    )


def _require_finite_numbers(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("sidebar result contains a non-finite number")
    if type(value) is dict:
        for child in value.values():
            _require_finite_numbers(child)
    elif type(value) is list:
        for child in value:
            _require_finite_numbers(child)


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _nested(record: JsonObject, *path: str) -> float:
    value: object = record
    for key in path:
        if type(value) is not dict or key not in value:
            raise ValueError(f"sidebar result is missing {'.'.join(path)}")
        value = value[key]
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"sidebar metric {'.'.join(path)} is not finite")
    return float(value)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    """Run the fixed sidebar or validate and summarize its existing artifact."""
    repository_root = Path(__file__).resolve().parents[5]
    paths = ReasoningSidebarPaths.from_repository(repository_root)
    if paths.destination.is_dir():
        manifest = validate_reasoning_sidebar(paths.destination)
        result = _json_object(paths.destination / "results.json")
        print(f"Existing reasoning sidebar: {paths.destination}")
        print(f"Manifest: {manifest.manifest_sha256}")
        print(f"Report: {paths.destination / 'report.md'}")
        print(f"Actual paid generation cost: ${result['actual_billed_usd']}")
        return
    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="tinyworlds-v2-reasoning-sidebar-",
            dir=paths.destination.parent,
        )
    )
    print(f"Temporary artifact directory: {staging}", flush=True)
    result = run_reasoning_sidebar(staging, paths)
    print(f"Reasoning sidebar: {result.directory}")
    print(f"Manifest: {result.manifest_sha256}")
    print(f"Actual paid generation cost: ${result.actual_cost_usd}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")


__all__ = [
    "REASONING_SIDEBAR_REFERENCE_MANIFEST_SHA256",
    "ReasoningSidebarPaths",
    "ReasoningSidebarRunResult",
    "run_reasoning_sidebar",
    "validate_reasoning_sidebar",
]
