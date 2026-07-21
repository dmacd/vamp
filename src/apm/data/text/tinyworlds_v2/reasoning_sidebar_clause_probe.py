"""Zero-training clause-completion diagnostic for a completed reasoning sidebar."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import math
from pathlib import Path
import tempfile
from time import monotonic

import jax
import numpy as np

from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.continual.language_adaptation_artifact import (
    flatten_lora_edge,
    unflatten_lora_edge,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    canonical_json_loads,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    Phase1ArtifactManifest,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.reasoning_sidebar import (
    REASONING_SIDEBAR_LORA_RANK,
    REASONING_SIDEBAR_VERSION,
    build_sidebar_clause_completion_queries,
    sidebar_query_score_records,
)
from apm.data.text.tinyworlds_v2.reasoning_sidebar_run import (
    REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
    ReasoningSidebarPaths,
    validate_reasoning_sidebar,
)
from apm.lm.candidate_scoring import (
    score_edge_coefficient_candidates,
    score_frozen_base_candidates,
)
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import pack_lora_memory, packed_with_candidate_edge
from apm.lm.text import TokenizersTextTokenizer
from apm.lm.tinystories_conversion import load_tinystories_artifact
from apm.memory.graph import NodeId, init_memory_graph


CLAUSE_PROBE_VERSION = "tinyworlds-v2-reasoning-sidebar-clause-probe-v1"
CLAUSE_PROBE_DIRECTORY = "reasoning-sidebar-v1-clause-probe"
_METHODS = (
    "frozen",
    "tinystories-control",
    "qwen3.5-35b-a3b",
    "gpt-5.4-mini",
)


def run_sidebar_clause_probe(
    staging_directory: str | Path,
    paths: ReasoningSidebarPaths,
) -> tuple[Path, str]:
    """Score exact seen-clause prefixes under the frozen model and three adapters."""
    staging = Path(staging_directory)
    destination = paths.destination.parent / CLAUSE_PROBE_DIRECTORY
    if not staging.is_dir():
        raise FileNotFoundError(f"clause-probe staging directory is missing: {staging}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"clause-probe destination exists: {destination}")
    started = monotonic()
    source_manifest = validate_reasoning_sidebar(paths.destination)
    builder = Phase1ArtifactBuilder(staging, version=CLAUSE_PROBE_VERSION)
    builder.write_json(
        "configuration.json",
        {
            "evaluation_microbatch_size": REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
            "source_manifest_sha256": source_manifest.manifest_sha256,
            "source_version": REASONING_SIDEBAR_VERSION,
            "version": CLAUSE_PROBE_VERSION,
        },
    )
    print("Clause probe phase 1/3: loading the frozen base and persisted LoRAs.", flush=True)
    base = load_tinystories_artifact(paths.checkpoint).checkpoint
    tokenizer = TokenizersTextTokenizer.from_file(paths.tokenizer)
    queries = build_sidebar_clause_completion_queries(tokenizer)
    lora_config = LoraConfig(
        rank=REASONING_SIDEBAR_LORA_RANK,
        alpha=float(REASONING_SIDEBAR_LORA_RANK),
    )
    adapters = tuple(
        (
            arm_id,
            _load_adapter(
                paths.destination / "adapters" / arm_id,
                base.config,
                lora_config,
            ),
        )
        for arm_id in _METHODS[1:]
    )

    print("Clause probe phase 2/3: scoring four methods on 12 exact prefixes.", flush=True)
    from tqdm.auto import tqdm

    packed = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        base.config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )
    progress = tqdm(
        total=len(_METHODS),
        desc="Clause completion methods",
        unit="method",
        dynamic_ncols=True,
        leave=True,
    )
    method_scores: list[tuple[str, np.ndarray]] = []
    frozen = score_frozen_base_candidates(
        base.params,
        base.config,
        queries,
        evaluation_microbatch_size=REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE,
    )
    method_scores.append(("frozen", frozen))
    progress.update()
    for arm_id, adapter in adapters:
        scores = score_edge_coefficient_candidates(
            base.params,
            base.config,
            packed_with_candidate_edge(packed, adapter, 0),
            lora_config,
            queries,
            np.ones((len(queries), 1), dtype=np.float32),
            evaluation_microbatch_size=(
                REASONING_SIDEBAR_EVALUATION_MICROBATCH_SIZE
            ),
        )
        method_scores.append((arm_id, scores))
        progress.update()
    progress.close()

    result = {
        "elapsed_seconds": monotonic() - started,
        "methods": {
            method: _method_summary(queries, scores)
            for method, scores in method_scores
        },
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "status": "completed",
        "version": CLAUSE_PROBE_VERSION,
    }
    builder.write_bytes(
        "scores.jsonl",
        canonical_jsonl_bytes(
            {"method": method, **record}
            for method, scores in method_scores
            for record in sidebar_query_score_records(queries, scores)
        ),
    )
    builder.write_json("results.json", result)
    builder.write_bytes("report.md", _report(result).encode("utf-8"))
    print("Clause probe phase 3/3: validating and promoting the diagnostic.", flush=True)
    manifest = builder.finalize()
    validated = validate_sidebar_clause_probe(staging)
    if validated.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("clause-probe validation changed artifact identity")
    promoted = builder.promote(
        destination,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    return promoted, manifest.manifest_sha256


def validate_sidebar_clause_probe(
    directory: str | Path,
) -> Phase1ArtifactManifest:
    """Validate a completed clause probe and its exact method/query coverage."""
    root = Path(directory)
    manifest = load_phase1_artifact_tree(root)
    result = _json_object(root / "results.json")
    if (
        result.get("version") != CLAUSE_PROBE_VERSION
        or result.get("status") != "completed"
    ):
        raise ValueError("clause-probe result identity changed")
    methods = result.get("methods")
    if type(methods) is not dict or tuple(methods) != tuple(sorted(_METHODS)):
        raise ValueError("clause-probe methods changed")
    _require_finite_numbers(result)
    scores = _jsonl_objects(root / "scores.jsonl")
    if len(scores) != len(_METHODS) * 12 or {
        record.get("method") for record in scores
    } != set(_METHODS):
        raise ValueError("clause-probe score coverage changed")
    return manifest


def _load_adapter(
    directory: Path,
    model_config,
    lora_config: LoraConfig,
) -> LoraEdge:
    from safetensors import safe_open
    from safetensors.numpy import load_file

    metadata = _json_object(directory / "metadata.json")
    path = directory / "adapter.safetensors"
    tensors = load_file(str(path))
    with safe_open(str(path), framework="np") as contents:
        stored_metadata = contents.metadata()
    if (
        stored_metadata.get("adapter_checksum") != metadata.get("adapter_checksum")
        or stored_metadata.get("arm_id") != metadata.get("arm_id")
    ):
        raise ValueError("persisted sidebar adapter metadata changed")
    adapter = unflatten_lora_edge(tensors, model_config, lora_config)
    if _adapter_checksum(adapter, model_config, lora_config) != metadata.get(
        "adapter_checksum"
    ):
        raise ValueError("persisted sidebar adapter checksum changed")
    return adapter


def _adapter_checksum(adapter: LoraEdge, model_config, lora_config: LoraConfig) -> str:
    digest = sha256()
    for name, value in sorted(flatten_lora_edge(adapter, model_config, lora_config).items()):
        array = np.asarray(value, dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _method_summary(
    queries: Sequence[KnowledgeQuery],
    scores: np.ndarray,
) -> JsonObject:
    return {
        reasoning_type: _candidate_summary(
            tuple(query for query in queries if query.reasoning_type == reasoning_type),
            scores[
                np.asarray(
                    [
                        index
                        for index, query in enumerate(queries)
                        if query.reasoning_type == reasoning_type
                    ]
                )
            ],
        )
        for reasoning_type in (
            "fact_clause_completion",
            "rule_clause_completion",
        )
    }


def _candidate_summary(
    queries: Sequence[KnowledgeQuery],
    scores: np.ndarray,
) -> JsonObject:
    score_array = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(
        [query.correct_candidate_index for query in queries], dtype=np.int32
    )
    rows = np.arange(len(queries))
    wrong = score_array.copy()
    wrong[rows, correct] = np.inf
    return {
        "accuracy": float(np.mean(np.argmin(score_array, axis=1) == correct)),
        "correct_nll": float(np.mean(score_array[rows, correct])),
        "margin": float(
            np.mean(np.min(wrong, axis=1) - score_array[rows, correct])
        ),
        "query_count": len(queries),
    }


def _report(result: JsonObject) -> str:
    methods = result.get("methods")
    if type(methods) is not dict:
        raise TypeError("clause-probe result methods are malformed")
    rows = tuple(
        f"| {method} | "
        f"{100 * _metric(record, 'fact_clause_completion', 'accuracy'):.1f}% | "
        f"{100 * _metric(record, 'rule_clause_completion', 'accuracy'):.1f}% | "
        f"{_metric(record, 'fact_clause_completion', 'margin'):.3f} | "
        f"{_metric(record, 'rule_clause_completion', 'margin'):.3f} |"
        for method, record in methods.items()
        if type(record) is dict
    )
    return "\n".join(
        (
            "# Exact-clause completion follow-up",
            "",
            "This zero-training diagnostic reuses the immutable adapters from the "
            "reasoning sidebar. It asks for the next token after the exact evidence "
            "prefixes seen in training, before any paraphrase or one-hop composition.",
            "",
            "| method | fact-clause accuracy | rule-clause accuracy | fact margin | rule margin |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "High clause completion together with chance paraphrase/one-hop accuracy "
            "means the LoRA stored the literal continuations but did not expose them "
            "as queryable, compositional knowledge. Low clause completion would mean "
            "even literal binding storage failed.",
            "",
        )
    )


def _metric(record: JsonObject, group: str, name: str) -> float:
    value = record.get(group)
    metric = None if type(value) is not dict else value.get(name)
    if type(metric) not in (int, float) or not math.isfinite(float(metric)):
        raise ValueError("clause-probe report metric is missing or non-finite")
    return float(metric)


def _json_object(path: Path) -> JsonObject:
    return require_json_object(
        canonical_json_loads(path.read_bytes(), label=str(path)),
        label=str(path),
    )


def _jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    return tuple(
        require_json_object(
            canonical_json_loads(line, label=f"{path} line {index}"),
            label=f"{path} line {index}",
        )
        for index, line in enumerate(path.read_bytes().splitlines(), start=1)
    )


def _require_finite_numbers(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("clause-probe result contains a non-finite number")
    if type(value) is dict:
        for child in value.values():
            _require_finite_numbers(child)
    elif type(value) is list:
        for child in value:
            _require_finite_numbers(child)


def main() -> None:
    """Run the fixed clause probe or validate its existing artifact."""
    repository_root = Path(__file__).resolve().parents[5]
    paths = ReasoningSidebarPaths.from_repository(repository_root)
    destination = paths.destination.parent / CLAUSE_PROBE_DIRECTORY
    if destination.is_dir():
        manifest = validate_sidebar_clause_probe(destination)
        print(f"Existing clause probe: {destination}")
        print(f"Manifest: {manifest.manifest_sha256}")
        print(f"Report: {destination / 'report.md'}")
        return
    staging = Path(
        tempfile.mkdtemp(
            prefix="tinyworlds-v2-clause-probe-",
            dir=paths.destination.parent,
        )
    )
    print(f"Temporary artifact directory: {staging}", flush=True)
    destination, manifest_sha256 = run_sidebar_clause_probe(staging, paths)
    print(f"Clause probe: {destination}")
    print(f"Manifest: {manifest_sha256}")


__all__ = [
    "CLAUSE_PROBE_DIRECTORY",
    "CLAUSE_PROBE_VERSION",
    "run_sidebar_clause_probe",
    "validate_sidebar_clause_probe",
]
