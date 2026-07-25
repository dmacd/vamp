#!/usr/bin/env python3
"""Run the registered TinyWorlds-Q rabbit/horse pilot gates."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time

from apm.continual.language_adaptation_artifact import (
    load_language_adaptation_artifact,
)
from apm.data.text.tinyworlds_q_semantic.adaptation import (
    prepare_query_adaptation,
    train_or_resume_query_adaptations,
)
from apm.data.text.tinyworlds_q_semantic.catalog import load_validation_catalog
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_q_semantic.execution import (
    BaseQualityDecision,
    select_pilot_budget,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import (
    evaluate_pilot_budget,
    evaluate_staged_semantic_queries,
)
from apm.data.text.tinyworlds_q_semantic.manifests import PILOT_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.partition import load_query_partition
from apm.data.text.tinyworlds_q_semantic.preflight import (
    load_query_gpu_preflight,
    run_and_publish_query_gpu_preflight,
)
from apm.data.text.tinyworlds_q_semantic.pilot import (
    load_semantic_pilot_result,
    publish_semantic_pilot_result,
)
from apm.data.text.tinyworlds_q_semantic.pilot_sweep import (
    train_or_resume_pilot_independent_sweep,
)
from apm.data.text.tinyworlds_q_semantic.queries import compile_semantic_queries
from apm.data.text.tinyworlds_q_semantic.selected_base import (
    QueryBaseEpochEvidence,
    load_query_selected_base,
    publish_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.training import (
    QueryBaseTrainingConfig,
    QuerySplitNll,
    allocator_peak_bytes,
    evaluate_query_base_nll,
    init_query_base_train_state,
    load_query_training_checkpoint,
    query_base_training_identity,
    run_query_base_training,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
WORK_ROOT = CHECKPOINT_ROOT / "work"
RESULT_ROOT = (
    REPOSITORY_ROOT / "results" / "language_cl" / "tinyworlds-q-semantic-v1"
)
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
CATALOG_SHA256 = (
    "5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f"
)
PARTITION_SHA256 = (
    "419e6c8b6362add9af081885066559cc34b18f5c7044894f343c7caf0091ad0c"
)


def _eta(seconds: float) -> str:
    remaining = max(0, round(seconds))
    hours, remainder = divmod(remaining, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sources():
    catalog = load_validation_catalog(DATA_ROOT / "catalog" / CATALOG_SHA256)
    partition = load_query_partition(
        DATA_ROOT / "partitions" / PARTITION_SHA256,
        catalog,
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    preset = QueryExperimentPreset(
        tuple(concept.concept_id for concept in PILOT_CONCEPTS),
        adapter_updates=2_000,
    )
    return catalog, partition, tokenizer, preset


def _matching_preflight(catalog, partition, preset):
    root = CHECKPOINT_ROOT / "preflight"
    if not root.is_dir():
        return None
    matches = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or len(path.name) != 64:
            continue
        record_path = path / "preflight.json"
        if not record_path.is_file():
            continue
        payload = record_path.read_bytes()
        record = json.loads(payload)
        if type(record) is not dict or canonical_json_bytes(record) != payload:
            raise ValueError(f"noncanonical GPU preflight candidate: {path}")
        if (
            record.get("catalog_sha256") != catalog.catalog_sha256
            or record.get("partition_sha256") != partition.partition_sha256
            or record.get("config_sha256") != preset.config_sha256
        ):
            continue
        preflight = load_query_gpu_preflight(
            path,
            partition,
            catalog,
            preset,
        )
        matches.append(preflight)
    if len(matches) > 1:
        raise RuntimeError("multiple GPU preflights bind the pilot sources")
    return matches[0] if matches else None


def _run_preflight(sources=None):
    catalog, partition, tokenizer, preset = sources or _sources()
    existing = _matching_preflight(catalog, partition, preset)
    if existing is not None:
        print(f"Using strict GPU preflight {existing.directory}.", flush=True)
        return existing
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="pilot-gpu-preflight-", dir=WORK_ROOT))
    preflight = run_and_publish_query_gpu_preflight(
        partition,
        catalog,
        tokenizer,
        preset,
        working,
        CHECKPOINT_ROOT / "preflight",
    )
    print(f"GPU preflight: {preflight.preflight_sha256}", flush=True)
    print(f"Preflight report: {preflight.directory / 'preflight.md'}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return preflight


def _matching_selected_base(partition, preset):
    root = CHECKPOINT_ROOT / "base"
    if not root.is_dir():
        return None
    matches = []
    for path in sorted(root.iterdir()):
        manifest_path = path / "manifest.json"
        if not path.is_dir() or len(path.name) != 64 or not manifest_path.is_file():
            continue
        payload = manifest_path.read_bytes()
        record = json.loads(payload)
        if type(record) is not dict or canonical_json_bytes(record) != payload:
            raise ValueError(f"noncanonical selected-base candidate: {path}")
        if (
            record.get("catalog_sha256") != partition.catalog_sha256
            or record.get("partition_sha256") != partition.partition_sha256
        ):
            continue
        matches.append(load_query_selected_base(path, partition, preset))
    if len(matches) > 1:
        raise RuntimeError("multiple selected bases bind the pilot partition")
    return matches[0] if matches else None


def _latest_base_checkpoint(working: Path, training_sha256: str) -> Path | None:
    states = working / "states"
    if not states.is_dir():
        return None
    candidates = []
    for path in sorted(states.iterdir()):
        resume_path = path / "resume.json"
        if not path.is_dir() or not resume_path.is_file():
            continue
        payload = resume_path.read_bytes()
        record = json.loads(payload)
        if type(record) is not dict or canonical_json_bytes(record) != payload:
            raise ValueError(f"noncanonical base checkpoint candidate: {path}")
        cursor = record.get("cursor")
        if (
            record.get("training_sha256") != training_sha256
            or type(cursor) is not dict
        ):
            raise ValueError(f"base checkpoint binding changed: {path}")
        values = tuple(
            cursor.get(field)
            for field in (
                "optimizer_update",
                "epoch",
                "block",
                "microbatch",
                "schedule_position",
            )
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"base checkpoint cursor changed: {path}")
        candidates.append((values, path))
    return max(candidates, default=((), None))[1]


def _epoch_checkpoint(working: Path, epoch: int) -> Path:
    matches = tuple(
        sorted((working / "states").glob(f"epoch-{epoch:02d}-update-*"))
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one complete checkpoint for epoch {epoch}")
    return matches[0]


def _load_epoch_evidence(
    working: Path,
    epoch: int,
    training_sha256: str,
    partition_sha256: str,
    config_sha256: str,
) -> QueryBaseEpochEvidence | None:
    path = working / f"epoch-{epoch:02d}-validation.json"
    if not path.is_file():
        return None
    payload = path.read_bytes()
    record = json.loads(payload)
    required = {
        "active_tokens",
        "base_config_sha256",
        "epoch",
        "format",
        "nll",
        "partition_sha256",
        "split",
        "training_sha256",
    }
    if (
        type(record) is not dict
        or canonical_json_bytes(record) != payload
        or set(record) != required
        or record.get("format") != "tinyworlds-q-semantic-base-epoch-evidence-v1"
        or record.get("epoch") != epoch
        or record.get("split") != "validation"
        or record.get("training_sha256") != training_sha256
        or record.get("partition_sha256") != partition_sha256
        or record.get("base_config_sha256") != config_sha256
        or type(record.get("active_tokens")) is not int
        or type(record.get("nll")) not in (int, float)
    ):
        raise ValueError(f"base epoch-{epoch} evidence changed")
    return QueryBaseEpochEvidence(
        epoch,
        QuerySplitNll(
            "validation",
            int(record["active_tokens"]),
            float(record["nll"]),
        ),
    )


def _publish_epoch_evidence(
    working: Path,
    evidence: QueryBaseEpochEvidence,
    training_sha256: str,
    partition_sha256: str,
    config_sha256: str,
) -> Path:
    path = working / f"epoch-{evidence.epoch:02d}-validation.json"
    payload = canonical_json_bytes(
        {
            "active_tokens": evidence.validation.active_tokens,
            "base_config_sha256": config_sha256,
            "epoch": evidence.epoch,
            "format": "tinyworlds-q-semantic-base-epoch-evidence-v1",
            "nll": evidence.validation.nll,
            "partition_sha256": partition_sha256,
            "split": evidence.validation.split,
            "training_sha256": training_sha256,
        }
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"different epoch evidence already exists: {path}")
        return path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    os.replace(temporary, path)
    return path


def _evaluate_epoch(
    epoch: int,
    params,
    partition,
    preset,
    config,
    working,
    training_sha256,
    config_sha256,
) -> QueryBaseEpochEvidence:
    def progress(_split: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(
                f"TinyWorlds-Q epoch {epoch} validation "
                f"{completed:,}/{total:,}",
                flush=True,
            )

    validation = evaluate_query_base_nll(
        params,
        partition,
        preset,
        "validation",
        config,
        progress=progress,
    )
    evidence = QueryBaseEpochEvidence(epoch, validation)
    _publish_epoch_evidence(
        working,
        evidence,
        training_sha256,
        partition.partition_sha256,
        config_sha256,
    )
    print(
        f"TinyWorlds-Q epoch {epoch} held-in NLL: {validation.nll:.9f}",
        flush=True,
    )
    return evidence


def _run_base(sources, preflight):
    catalog, partition, _tokenizer, preset = sources
    if preflight is None:
        raise RuntimeError("the real base requires a passing GPU preflight")
    selected = _matching_selected_base(partition, preset)
    if selected is not None:
        print(f"Using strict selected base {selected.directory}.", flush=True)
        return selected
    config = QueryBaseTrainingConfig.from_preset(preset)
    config_sha256 = record_sha256(config.as_record())
    training_sha256, planned_updates = query_base_training_identity(
        partition,
        preset,
        config,
    )
    working = WORK_ROOT / f"pilot-base-{training_sha256}"
    working.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    print(
        f"Phase: seed-zero base training ({planned_updates:,} optimizer updates).",
        flush=True,
    )

    def training_progress(cursor, nll: float, planned: int) -> None:
        update = cursor.optimizer_update
        if update == 1 or update % 100 == 0 or update == planned:
            elapsed = time.monotonic() - phase_started
            remaining = elapsed * (planned - update) / max(1, update)
            print(
                f"TinyWorlds-Q base update {update:,}/{planned:,}; "
                f"NLL {nll:.6f}; phase ETA {_eta(remaining)}",
                flush=True,
            )

    evidence = []
    result = None
    for epoch in (1, 2):
        existing_evidence = _load_epoch_evidence(
            working,
            epoch,
            training_sha256,
            partition.partition_sha256,
            config_sha256,
        )
        latest = _latest_base_checkpoint(working, training_sha256)
        completed_epoch = 0
        if latest is not None:
            latest_record = json.loads((latest / "resume.json").read_bytes())
            latest_cursor = latest_record["cursor"]
            assert isinstance(latest_cursor, dict)
            completed_epoch = int(latest_cursor["epoch"])
        if completed_epoch < epoch:
            result = run_query_base_training(
                partition,
                preset,
                working,
                config,
                resume_from=latest,
                stop_after_epoch=epoch,
                progress=training_progress,
            )
        if existing_evidence is None:
            if result is not None and result.cursor.epoch == epoch:
                params = result.state.trainable
            else:
                template = init_query_base_train_state(config, planned_updates)
                state, cursor = load_query_training_checkpoint(
                    _epoch_checkpoint(working, epoch),
                    training_sha256,
                    template,
                )
                if cursor.epoch != epoch:
                    raise ValueError("epoch checkpoint cursor changed")
                params = state.trainable
            existing_evidence = _evaluate_epoch(
                epoch,
                params,
                partition,
                preset,
                config,
                working,
                training_sha256,
                config_sha256,
            )
        evidence.append(existing_evidence)
        if result is not None and epoch == 1:
            del result
            result = None
            gc.collect()

    latest = _latest_base_checkpoint(working, training_sha256)
    if latest is None:
        raise RuntimeError("complete base training has no resumable state")
    if result is None or result.cursor.epoch != 2:
        result = run_query_base_training(
            partition,
            preset,
            working,
            config,
            resume_from=latest,
            progress=training_progress,
        )
    peak = allocator_peak_bytes()
    decision = BaseQualityDecision(
        tuple(item.validation.nll for item in evidence),  # type: ignore[arg-type]
        peak,
        preset.allocator_peak_limit_bytes,
    )
    decision_path = working / "base-decision.json"
    decision_payload = canonical_json_bytes(
        {
            "allocator_peak_bytes": peak,
            "epoch_nll": [item.validation.nll for item in evidence],
            "format": "tinyworlds-q-semantic-base-decision-v1",
            "partition_sha256": partition.partition_sha256,
            "passed": decision.passed,
            "reason": decision.reason,
            "training_sha256": training_sha256,
        }
    )
    if decision_path.exists() and decision_path.read_bytes() != decision_payload:
        raise FileExistsError("base quality decision changed")
    if not decision_path.exists():
        temporary = decision_path.with_suffix(".json.tmp")
        temporary.write_bytes(decision_payload)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, decision_path)
    if not decision.passed:
        raise RuntimeError(f"pilot base gate failed: {decision.reason}")
    selected = publish_query_selected_base(
        result,
        tuple(evidence),  # type: ignore[arg-type]
        peak,
        partition,
        preset,
        CHECKPOINT_ROOT,
    )
    print(f"Selected base: {selected.selection_sha256}", flush=True)
    print(f"Selected-base report: {selected.directory / 'training-report.md'}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return selected


def _matching_pilot_result(catalog, partition, selected_base, preflight):
    root = RESULT_ROOT / "pilot"
    if not root.is_dir():
        return None
    matches = []
    for path in sorted(root.iterdir()):
        record_path = path / "pilot.json"
        if not path.is_dir() or len(path.name) != 64 or not record_path.is_file():
            continue
        payload = record_path.read_bytes()
        record = json.loads(payload)
        if type(record) is not dict or canonical_json_bytes(record) != payload:
            raise ValueError(f"noncanonical semantic pilot candidate: {path}")
        if (
            record.get("catalog_sha256") != catalog.catalog_sha256
            or record.get("partition_sha256") != partition.partition_sha256
            or record.get("selected_base_sha256") != selected_base.selection_sha256
            or record.get("preflight_sha256") != preflight.preflight_sha256
        ):
            continue
        matches.append(load_semantic_pilot_result(path))
    if len(matches) > 1:
        raise RuntimeError("multiple pilot results bind the same frozen inputs")
    return matches[0] if matches else None


def _run_pilot_budget(
    queries,
    loaded_base,
    budget_snapshot,
    preset,
):
    def evaluation_progress(phase: str, completed: int, total: int) -> None:
        print(f"Pilot validation {phase}: {completed:,}/{total:,} chunks", flush=True)

    evaluation = evaluate_pilot_budget(
        queries,
        loaded_base,
        budget_snapshot.adapters,
        budget_snapshot.tensor_checksum,
        preset,
        progress=evaluation_progress,
    )
    print(
        "Pilot budget "
        f"{preset.adapter_updates:,}: "
        + ", ".join(
            f"{concept_id}={accuracy:.3f}"
            for concept_id, accuracy in evaluation.budget.concept_accuracy
        )
        + f"; passed={evaluation.budget.passes}",
        flush=True,
    )
    return evaluation


def _run_pilot(sources, preflight, selected_base):
    catalog, partition, tokenizer, registered_preset = sources
    existing = _matching_pilot_result(
        catalog,
        partition,
        selected_base,
        preflight,
    )
    if existing is not None:
        print(f"Using strict pilot result {existing.root}.", flush=True)
        return existing
    loaded_base = load_gpt_neo_checkpoint(selected_base.checkpoint)
    overall_started = time.monotonic()
    maximum_budget = max(registered_preset.pilot_update_budgets)
    maximum_preset = replace(
        registered_preset,
        adapter_updates=maximum_budget,
    )
    maximum_prepared = prepare_query_adaptation(
        catalog,
        partition,
        tokenizer,
        maximum_preset,
    )
    maximum_training_updates = (
        3 * registered_preset.active_world_count * maximum_budget
    )
    concept_index = {
        concept_id: index
        for index, concept_id in enumerate(registered_preset.concept_ids)
    }

    def sweep_progress(
        concept_id: str,
        update: int,
        loss: float,
        phase_total: int,
    ) -> None:
        if update != 1 and update % 100 != 0 and update != phase_total:
            return
        completed = concept_index[concept_id] * maximum_budget + update
        elapsed = time.monotonic() - overall_started
        remaining = elapsed * (maximum_training_updates - completed) / max(1, completed)
        print(
            f"Pilot independent/{concept_id} update {update:,}/{phase_total:,}; "
            f"loss {loss:.6f}; conservative ETA {_eta(remaining)}",
            flush=True,
        )

    print(
        "Phase: pilot independent sweep with exact 500/1,000/2,000 snapshots.",
        flush=True,
    )
    sweep = train_or_resume_pilot_independent_sweep(
        maximum_prepared,
        partition,
        selected_base,
        CHECKPOINT_ROOT / "pilot-independent-sweep" / maximum_preset.config_sha256,
        maximum_preset,
        progress=sweep_progress,
    )
    queries = tuple(
        query
        for query in compile_semantic_queries(catalog, tokenizer)
        if query.concept_id in registered_preset.concept_ids
    )
    budget_evaluations = tuple(
        _run_pilot_budget(
            queries,
            loaded_base,
            sweep.budget(budget),
            replace(registered_preset, adapter_updates=budget),
        )
        for budget in registered_preset.pilot_update_budgets
    )
    selected_updates = select_pilot_budget(
        tuple(item.budget for item in budget_evaluations),
        registered_preset,
    )
    selected_preset = replace(
        registered_preset,
        adapter_updates=selected_updates,
    )
    selected_prepared = prepare_query_adaptation(
        catalog,
        partition,
        tokenizer,
        selected_preset,
    )
    selected_working = (
        CHECKPOINT_ROOT / "pilot-adaptations" / selected_preset.config_sha256
    )
    selected_snapshot = sweep.budget(selected_updates)
    selected_total_updates = (
        registered_preset.active_world_count * maximum_budget
        + 2 * registered_preset.active_world_count * selected_updates
    )
    method_index = {"sequential": 0, "vamp": 1}

    def selected_progress(
        method: str,
        concept_id: str,
        step: int,
        loss: float,
        phase_total: int,
    ) -> None:
        if step != 1 and step % 100 != 0 and step != phase_total:
            return
        completed = (
            registered_preset.active_world_count * maximum_budget
            + (
                concept_index[concept_id] * 2 + method_index[method]
            )
            * selected_updates
            + step
        )
        elapsed = time.monotonic() - overall_started
        remaining = elapsed * (selected_total_updates - completed) / max(1, completed)
        print(
            f"Selected pilot {method}/{concept_id} update {step:,}/{phase_total:,}; "
            f"loss {loss:.6f}; overall ETA {_eta(remaining)}",
            flush=True,
        )

    print(
        f"Phase: selected {selected_updates:,}-update sequential and VAMP exercise.",
        flush=True,
    )
    trained = train_or_resume_query_adaptations(
        selected_prepared,
        partition,
        selected_base,
        selected_working,
        selected_preset,
        progress=selected_progress,
        independent_adapters=selected_snapshot.adapters,
        independent_rng_by_stage=sweep.rng_by_stage,
        additional_config_hashes={"pilot-independent-sweep": sweep.sweep_sha256},
    )
    before_resume = load_language_adaptation_artifact(
        selected_working
        / "stages"
        / f"stage-{selected_preset.active_world_count:03d}"
    )
    resumed = train_or_resume_query_adaptations(
        selected_prepared,
        partition,
        selected_base,
        selected_working,
        selected_preset,
        independent_adapters=selected_snapshot.adapters,
        independent_rng_by_stage=sweep.rng_by_stage,
        additional_config_hashes={"pilot-independent-sweep": sweep.sweep_sha256},
    )
    resume_verified = (
        trained.adaptation.tensor_checksum == before_resume.tensor_checksum
        and resumed.adaptation.tensor_checksum == before_resume.tensor_checksum
        and resumed.stage_directory
        == (
            selected_working
            / "stages"
            / f"stage-{selected_preset.active_world_count:03d}"
        ).resolve()
    )
    if not resume_verified:
        raise RuntimeError("selected pilot adaptation resume changed immutable state")
    stage_artifacts = tuple(
        load_language_adaptation_artifact(
            selected_working / "stages" / f"stage-{stage:03d}"
        )
        for stage in range(1, selected_preset.active_world_count + 1)
    )
    selected_queries = tuple(
        query
        for query in compile_semantic_queries(catalog, tokenizer)
        if query.concept_id in selected_preset.concept_ids
    )

    def evaluation_progress(phase: str, completed: int, total: int) -> None:
        print(f"Selected pilot {phase}: {completed:,}/{total:,} chunks", flush=True)

    selected_results = evaluate_staged_semantic_queries(
        selected_queries,
        loaded_base,
        stage_artifacts,
        selected_preset,
        tokenizer.pad_token_id,
        progress=evaluation_progress,
    )
    peak = max(
        allocator_peak_bytes(),
        trained.allocator_peak_bytes,
        resumed.allocator_peak_bytes,
    )
    result = publish_semantic_pilot_result(
        RESULT_ROOT / "pilot",
        catalog_sha256=catalog.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        preflight_sha256=preflight.preflight_sha256,
        preset=registered_preset,
        budgets=budget_evaluations,
        independent_sweep_sha256=sweep.sweep_sha256,
        independent_sweep_manifest_sha256=_file_sha256(
            sweep.stage_directory / "manifest.json"
        ),
        selected_adaptation_manifest_sha256=_file_sha256(
            resumed.stage_directory / "manifest.json"
        ),
        selected_validation_results=selected_results,
        resume_verified=resume_verified,
        runtime_seconds=time.monotonic() - overall_started,
        allocator_peak_bytes=peak,
    )
    print(f"Pilot result: {result.pilot_sha256}", flush=True)
    print(f"Pilot report directory: {result.root}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "base", "pilot"),
        default="preflight",
        help="highest registered pilot stage to execute",
    )
    args = parser.parse_args()
    sources = _sources()
    preflight = _run_preflight(sources)
    if args.stage in ("base", "pilot"):
        selected_base = _run_base(sources, preflight)
        if args.stage == "pilot":
            _run_pilot(sources, preflight, selected_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
