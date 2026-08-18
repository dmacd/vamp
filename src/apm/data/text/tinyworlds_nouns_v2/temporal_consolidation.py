"""Authenticated orchestration for the TinyWorlds nouns-v2 temporal study."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import monotonic

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    NounSelectedBase,
    StoryIndexEntry,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASELINE_STAGEWISE_FORMAT,
    CONDITIONS,
    FULL_FINETUNE_STAGEWISE_FORMAT,
    HALF_STORY_FORMAT,
    PROBE_STORY_COUNT,
    RUN_MANIFEST_FORMAT,
    TASK_IDS,
    WHOLE_STORY_FORMAT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.experiment import (
    load_nouns_v2_gpu_preflight,
    load_nouns_v2_selected_base,
    load_nouns_v2_vamp_stages,
)
from apm.data.text.tinyworlds_nouns_v2.partition import find_partition, load_manifest
from apm.data.text.tinyworlds_nouns_v2.stagewise import validate_stagewise_ledger
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ARRIVAL_COUNT,
    CONTEXT_LENGTH,
    EVALUATION_ROW_FORMAT,
    MACRO_CHECKPOINT_INTERVAL,
    PHYSICAL_BATCH_SIZE,
    SENTINEL_STORIES_PER_TASK,
    SHARDS_PER_TASK,
    STUDY_ID,
    TEMPORAL_ORDERS,
    TemporalChunk,
    TemporalHierarchyState,
    TemporalMerge,
    TemporalOrder,
    TemporalShard,
    build_contract_record,
    contract_bytes,
    empty_hierarchy,
    expected_final_intervals,
    insert_arrival,
    select_temporal_shards,
    select_validation_sentinel,
    simulate_hierarchy,
    temporal_arrivals,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
    ProgressRecorder,
    StudyJob,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    MidpointCase,
    build_adapter_bank,
    build_midpoint_case,
    evaluate_to_ledger,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    AdapterArtifact,
    FullModelArtifact,
    StoryEpochBatches,
    TrainingJob,
    train_or_load_full_model,
    train_or_load_lora,
)
from apm.lm.checkpoint import LoadedGptNeoCheckpoint, load_gpt_neo_checkpoint


@dataclass(frozen=True, slots=True)
class TemporalStudyInputs:
    """Strict-loaded canonical sources and the derived executable contract."""

    repository_root: Path
    partition: NounsV2PartitionArtifact
    preset: NounsV2ExperimentPreset
    selected_base: NounSelectedBase
    loaded_base: LoadedGptNeoCheckpoint
    final_vamp_tensor_checksum: str
    canonical_hashes: tuple[tuple[str, str], ...]
    train_entries: tuple[tuple[str, tuple[StoryIndexEntry, ...]], ...]
    validation_entries: tuple[tuple[str, tuple[StoryIndexEntry, ...]], ...]
    shards: tuple[TemporalShard, ...]
    sentinel: tuple[tuple[str, tuple[str, ...]], ...]
    contract: dict[str, object]
    result_directory: Path
    checkpoint_directory: Path
    work_directory: Path

    def __post_init__(self) -> None:
        for name in ("repository_root", "result_directory", "checkpoint_directory", "work_directory"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if len(self.shards) != ARRIVAL_COUNT:
            raise ValueError("temporal study inputs do not contain 192 shards")

    @property
    def contract_sha256(self) -> str:
        """Return the immutable executable contract identity."""
        return str(self.contract["contract_sha256"])

    @property
    def train_entry_lookup(self) -> dict[str, StoryIndexEntry]:
        """Return selected-source pointers keyed by story identity."""
        return {
            entry.story_id: entry
            for _, entries in self.train_entries
            for entry in entries
        }

    @property
    def validation_by_task(self) -> dict[str, tuple[StoryIndexEntry, ...]]:
        """Return official midpoint validation pointers by noun."""
        return dict(self.validation_entries)


@dataclass(frozen=True, slots=True)
class SharedArtifacts:
    """Order-independent level-zero and offline control artifacts."""

    level_zero: tuple[tuple[str, AdapterArtifact], ...]
    independent_noun: tuple[tuple[str, AdapterArtifact], ...]
    iid_lora: AdapterArtifact
    iid_full_model: FullModelArtifact

    @property
    def level_zero_by_shard(self) -> dict[str, AdapterArtifact]:
        """Return level-zero artifacts keyed by shard identity."""
        return dict(self.level_zero)


@dataclass(frozen=True, slots=True)
class OrderingArtifacts:
    """Completed order-specific sequential stages and merge artifacts."""

    order: TemporalOrder
    sequential: tuple[AdapterArtifact, ...]
    chunks: tuple[tuple[str, AdapterArtifact], ...]
    final_state: TemporalHierarchyState
    merges: tuple[TemporalMerge, ...]

    @property
    def chunks_by_id(self) -> dict[str, AdapterArtifact]:
        """Return every trained live/retired chunk adapter by descriptor ID."""
        return dict(self.chunks)


def authenticate_temporal_study_inputs(
    repository_root: str | Path,
) -> TemporalStudyInputs:
    """Authenticate canonical nouns-v2 sources and publish the derived contract."""
    root = Path(repository_root).resolve()
    data_root = root / "data/tinyworlds-nouns-v2"
    checkpoint_root = root / "checkpoints/tinyworlds-nouns-v2"
    canonical_results = root / "results/language_cl/tinyworlds-nouns-v2"
    manifest = load_manifest(data_root / "manifest.json")
    partition = find_partition(manifest, data_root)
    if partition is None:
        raise FileNotFoundError("canonical nouns-v2 partition is not published")
    preset = NounsV2ExperimentPreset()
    selected_paths = tuple(checkpoint_root.glob("base/*/selected.json"))
    if len(selected_paths) != 1:
        raise ValueError("temporal study requires one selected nouns-v2 base")
    selected_record = load_canonical_json(selected_paths[0])
    selected_core = {
        key: value for key, value in selected_record.items() if key != "selection_sha256"
    }
    if (
        selected_record.get("selection_sha256") != record_sha256(selected_core)
        or selected_paths[0].parent.name != selected_record.get("training_sha256")
    ):
        raise ValueError("selected nouns-v2 base identity changed")
    preflight_paths = tuple(
        path
        for path in (checkpoint_root / "preflight").glob("*.json")
        if load_canonical_json(path).get("preflight_sha256")
        == selected_record.get("preflight_sha256")
    )
    if len(preflight_paths) != 1:
        raise ValueError("selected nouns-v2 base has no unique GPU preflight")
    preflight = load_nouns_v2_gpu_preflight(
        partition,
        preset,
        preflight_paths[0],
    )
    selected_base = load_nouns_v2_selected_base(
        partition,
        preset,
        preflight,
        selected_paths[0].parent,
    )
    loaded_base = load_gpt_neo_checkpoint(selected_base.reference)
    vamp_stages = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        checkpoint_root,
    )
    canonical_run = _canonical_run_manifest(canonical_results / "run-manifest.json")
    if (
        canonical_run.get("partition_sha256") != partition.partition_sha256
        or canonical_run.get("config_sha256") != preset.config_sha256
        or canonical_run.get("vamp_tensor_checksum") != vamp_stages[-1].tensor_checksum
    ):
        raise ValueError("canonical nouns-v2 run bindings changed")
    train_entries = tuple(
        (task, load_story_index(partition, f"task-{task}-train"))
        for task in TASK_IDS
    )
    probe_entries = tuple(
        (task, load_story_index(partition, f"task-{task}-probes"))
        for task in TASK_IDS
    )
    validation_entries = tuple(
        (task, load_story_index(partition, f"task-{task}-generation"))
        for task in TASK_IDS
    )
    validation_ids = tuple(
        entry.story_id for _, entries in validation_entries for entry in entries
    )
    if len(validation_ids) != 4_440 or len(set(validation_ids)) != 4_440:
        raise ValueError("temporal study validation coverage changed")
    root_probe_entries = load_story_index(partition, "root-probes")
    if (
        len(root_probe_entries) != PROBE_STORY_COUNT
        or {entry.story_id for entry in root_probe_entries}
        != set(partition.root_probe_story_ids)
        or any(
        len(entries) != 36
        or {entry.story_id for entry in entries}
        != set(partition.tasks[index].probe_story_ids)
        for index, (_, entries) in enumerate(probe_entries)
        )
    ):
        raise ValueError("temporal study probe bindings changed")
    probe_ids = {
        entry.story_id
        for entries in (root_probe_entries, *(entries for _, entries in probe_entries))
        for entry in entries
    }
    if len(probe_ids) != 25 * PROBE_STORY_COUNT or probe_ids & set(validation_ids):
        raise ValueError("temporal study probes overlap each other or validation")
    _authenticate_canonical_ledgers(
        canonical_results,
        partition,
        vamp_stages,
        validation_entries,
        canonical_run,
    )
    shards = select_temporal_shards(
        dict(train_entries),
        {
            task: tuple(entry.story_id for entry in entries)
            for task, entries in probe_entries
        },
        validation_ids,
    )
    sentinel = select_validation_sentinel(dict(validation_entries))
    canonical_hashes = canonical_artifact_hashes(root)
    bindings = {
        "base_manifest_sha256": selected_base.reference.manifest_sha256,
        "base_parameter_checksum": selected_base.reference.parameter_checksum,
        "base_training_sha256": selected_base.training_sha256,
        "canonical_artifact_hashes": dict(canonical_hashes),
        "canonical_run_sha256": canonical_run["run_sha256"],
        "final_vamp_tensor_checksum": vamp_stages[-1].tensor_checksum,
        "partition_sha256": partition.partition_sha256,
        "preset_sha256": preset.config_sha256,
        "probe_set_sha256": record_sha256(
            [
                ["root", [entry.story_id for entry in root_probe_entries]],
                *[
                    [task, [entry.story_id for entry in entries]]
                    for task, entries in probe_entries
                ],
            ]
        ),
        "validation_set_sha256": record_sha256(
            [[task, [entry.story_id for entry in entries]] for task, entries in validation_entries]
        ),
    }
    contract = build_contract_record(bindings=bindings, shards=shards, sentinel=sentinel)
    contract_sha = str(contract["contract_sha256"])
    result_directory = (
        canonical_results / "temporal-consolidation" / contract_sha
    )
    checkpoint_directory = (
        checkpoint_root / "temporal-consolidation" / contract_sha
    )
    work_directory = (
        canonical_results / "temporal-consolidation" / ".work-v1" / contract_sha
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    contract_path = result_directory / "contract.json"
    if contract_path.is_file():
        if contract_path.read_bytes() != contract_bytes(contract):
            raise ValueError("published temporal study contract changed")
    else:
        publish_immutable_json(contract_path, contract)
    return TemporalStudyInputs(
        root,
        partition,
        preset,
        selected_base,
        loaded_base,
        vamp_stages[-1].tensor_checksum,
        canonical_hashes,
        train_entries,
        validation_entries,
        shards,
        sentinel,
        contract,
        result_directory,
        checkpoint_directory,
        work_directory,
    )


def canonical_artifact_hashes(
    repository_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Hash protected nouns-v1/v2 sources without including study outputs."""
    root = Path(repository_root).resolve()
    paths: set[Path] = set()
    for version, final_stage in (("v1", 42), ("v2", 24)):
        data = root / f"data/tinyworlds-nouns-{version}"
        results = root / f"results/language_cl/tinyworlds-nouns-{version}"
        paths.update(path for path in (data / "manifest.json",) if path.is_file())
        paths.update(data.glob("partitions/*/partition.json"))
        paths.update(path for path in results.iterdir() if path.is_file())
        selected = tuple(
            root.glob(f"checkpoints/tinyworlds-nouns-{version}/base/*/selected.json")
        )
        if len(selected) != 1:
            raise ValueError(f"nouns-{version} selected-base count changed")
        paths.update(path for path in selected[0].parent.rglob("*") if path.is_file())
        final_records = tuple(
            root.glob(
                f"checkpoints/tinyworlds-nouns-{version}/vamp/*/"
                f"stage-{final_stage:03d}-*/stage.json"
            )
        )
        if len(final_records) != 1:
            raise ValueError(f"nouns-{version} final VAMP stage count changed")
        paths.update(
            path for path in final_records[0].parent.rglob("*") if path.is_file()
        )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("protected canonical artifact set is incomplete")
    return tuple(
        (path.relative_to(root).as_posix(), file_sha256(path))
        for path in sorted(paths)
    )


def assert_canonical_artifacts_unchanged(inputs: TemporalStudyInputs) -> None:
    """Reject any mutation of the protected canonical nouns artifacts."""
    after = canonical_artifact_hashes(inputs.repository_root)
    if after != inputs.canonical_hashes:
        before_map, after_map = dict(inputs.canonical_hashes), dict(after)
        changed = tuple(
            path
            for path in sorted(set(before_map) | set(after_map))
            if before_map.get(path) != after_map.get(path)
        )
        raise RuntimeError(f"canonical nouns artifacts changed: {changed}")


def study_dashboard_jobs(inputs: TemporalStudyInputs) -> tuple[StudyJob, ...]:
    """Return the complete exact-unit job matrix used by the live dashboard."""
    entry_lookup = inputs.train_entry_lookup
    shard_updates = {
        shard.shard_id: _optimizer_updates(shard.story_ids, entry_lookup)
        for shard in inputs.shards
    }
    schedules = {
        order: simulate_hierarchy(inputs.shards, order)[1]
        for order in TEMPORAL_ORDERS
    }
    shard_by_id = {shard.shard_id: shard for shard in inputs.shards}
    merge_updates = {
        (order, merge.parent.chunk_id): _optimizer_updates(
            tuple(
                story_id
                for shard_id in merge.parent.shard_ids
                for story_id in shard_by_id[shard_id].story_ids
            ),
            entry_lookup,
        )
        for order, merges in schedules.items()
        for merge in merges
    }
    independent_updates = sum(
        _optimizer_updates(
            tuple(
                story_id
                for shard in inputs.shards
                if shard.task_id == task_id
                for story_id in shard.story_ids
            ),
            entry_lookup,
        )
        for task_id in TASK_IDS
    )
    all_story_ids = tuple(
        story_id for shard in inputs.shards for story_id in shard.story_ids
    )
    iid_updates = _optimizer_updates(all_story_ids, entry_lookup)
    validation_counts = {
        task: len(entries) for task, entries in inputs.validation_entries
    }
    sentinel_story_cases = {
        "blocked": SENTINEL_STORIES_PER_TASK * sum(
            SHARDS_PER_TASK * learned for learned in range(1, len(TASK_IDS) + 1)
        ),
        "round_robin": SENTINEL_STORIES_PER_TASK
        * (sum(range(1, len(TASK_IDS) + 1)) + len(TASK_IDS) * (ARRIVAL_COUNT - len(TASK_IDS))),
    }
    blocked_macro = sum(
        sum(validation_counts[task] for task in TASK_IDS[:learned])
        for learned in range(1, len(TASK_IDS) + 1)
    )
    round_macro = (
        sum(validation_counts[task] for task in TASK_IDS[:8])
        + sum(validation_counts[task] for task in TASK_IDS[:16])
        + 22 * sum(validation_counts.values())
    )
    shared_jobs = (
        StudyJob("authenticate", "preflight", "authenticate canonical sources", 1, "gate", 120.0),
        StudyJob("materialize", "data", "materialize 192 immutable shards and cases", 1, "contract", 120.0),
        StudyJob(
            "level-zero",
            "shared training",
            "train 192 shared level-zero adapters",
            sum(shard_updates.values()),
            "updates",
            sum(shard_updates.values()) / 16.0,
        ),
        StudyJob(
            "independent-noun",
            "shared training",
            "train 24 independent noun adapters",
            independent_updates,
            "updates",
            independent_updates / 16.0,
        ),
        StudyJob(
            "iid-lora",
            "shared training",
            "train the joint-IID rank-eight LoRA",
            iid_updates,
            "updates",
            iid_updates / 16.0,
        ),
        StudyJob(
            "iid-full-model",
            "shared training",
            "train the joint-IID full model",
            iid_updates,
            "updates",
            iid_updates / 4.0,
        ),
    )
    ordering_jobs: list[StudyJob] = []
    for order in TEMPORAL_ORDERS:
        ordering_jobs.append(
            StudyJob(
                f"stack-{order}",
                f"{order} consolidation",
                f"materialize the live {order} hierarchy after each arrival",
                ARRIVAL_COUNT,
                "arrivals",
                120.0,
            )
        )
        ordering_jobs.append(
            StudyJob(
                f"sequential-{order}",
                f"{order} training",
                f"train the {order} sequential forgetting control",
                sum(shard_updates.values()),
                "updates",
                sum(shard_updates.values()) / 16.0,
            )
        )
        short_updates = sum(
            updates
            for (merge_order, _), updates in merge_updates.items()
            if merge_order == order and updates / 16.0 <= 300.0
        )
        ordering_jobs.append(
            StudyJob(
                f"merges-{order}-short",
                f"{order} consolidation",
                f"train {order} merge jobs projected at five minutes or less",
                short_updates,
                "updates",
                max(1.0, short_updates / 16.0),
            )
        )
        ordering_jobs.extend(
            StudyJob(
                _merge_progress_id(order, merge.parent.chunk_id),
                f"{order} consolidation",
                f"merge level {merge.parent.level} interval "
                f"{merge.parent.start_arrival}-{merge.parent.end_arrival}",
                merge_updates[(order, merge.parent.chunk_id)],
                "updates",
                merge_updates[(order, merge.parent.chunk_id)] / 16.0,
            )
            for merge in schedules[order]
            if merge_updates[(order, merge.parent.chunk_id)] / 16.0 > 300.0
        )
        ordering_jobs.extend(
            (
                StudyJob(
                    f"distortion-{order}",
                    f"{order} consolidation",
                    f"audit {order} source and sentinel merge distortion",
                    2 * sum(merge.parent.size for merge in schedules[order]),
                    "leaf-datasets",
                    8.0 * 2 * sum(
                        merge.parent.size for merge in schedules[order]
                    ),
                ),
                StudyJob(
                    f"sentinel-{order}",
                    f"{order} evaluation",
                    f"stream every-arrival {order} sentinel evaluation",
                    sentinel_story_cases[order] * 3,
                    "story-methods",
                    sentinel_story_cases[order] * 3 / 8.0,
                ),
                StudyJob(
                    f"macro-{order}",
                    f"{order} evaluation",
                    f"stream every-eight-arrival {order} full evaluation",
                    (blocked_macro if order == "blocked" else round_macro) * 3,
                    "story-methods",
                    (blocked_macro if order == "blocked" else round_macro) * 3 / 8.0,
                ),
            )
        )
    final_story_methods = 4_440 * 3
    return shared_jobs + tuple(ordering_jobs) + (
        StudyJob(
            "final-controls",
            "final evaluation",
            "evaluate independent noun, IID LoRA, and IID full-model controls",
            final_story_methods,
            "story-methods",
            final_story_methods / 4.0,
        ),
        StudyJob(
            "timing",
            "timing",
            "cold and five-repeat warm timing audit",
            _timing_shape_count(inputs),
            "shapes",
            60.0 * _timing_shape_count(inputs),
        ),
        StudyJob("report", "publication", "publish and regenerate reports", 1, "bundle", 300.0),
        StudyJob("immutability", "publication", "verify canonical hashes unchanged", 1, "gate", 60.0),
    )


def run_or_resume_shared_training(
    inputs: TemporalStudyInputs,
    progress: ProgressRecorder,
) -> SharedArtifacts:
    """Train or strict-load level-zero, independent, and both IID controls."""
    entry_lookup = inputs.train_entry_lookup
    base_params = inputs.loaded_base.params
    model_config = inputs.loaded_base.config
    level_zero: list[tuple[str, AdapterArtifact]] = []
    completed_updates = 0
    completed_runtime = 0.0
    for shard in inputs.shards:
        entries = tuple(entry_lookup[story_id] for story_id in shard.story_ids)
        job = TrainingJob(
            inputs.contract_sha256,
            f"level-zero-{shard.task_id}-{shard.shard_index}",
            "level_zero",
            shard.story_ids,
            (shard.shard_id,),
        )
        batches = _story_batches(inputs, entries, job)
        artifact = train_or_load_lora(
            job,
            batches,
            base_params,
            model_config,
            inputs.checkpoint_directory / "level-zero",
            inputs.work_directory / "training",
            progress=_training_progress(
                progress,
                "level-zero",
                completed_updates,
                completed_runtime,
            ),
        )
        completed_updates += artifact.optimizer_updates
        completed_runtime += artifact.runtime_seconds
        progress.update(
            "level-zero",
            completed_updates,
            status="complete" if completed_updates == _job_total(progress, "level-zero") else "running",
            elapsed_seconds=completed_runtime,
            metrics={"last_loss": _last_loss(artifact.directory / "losses.jsonl")},
        )
        level_zero.append((shard.shard_id, artifact))

    independent: list[tuple[str, AdapterArtifact]] = []
    completed_updates = 0
    completed_runtime = 0.0
    for task_id in TASK_IDS:
        task_shards = tuple(shard for shard in inputs.shards if shard.task_id == task_id)
        story_ids = tuple(story for shard in task_shards for story in shard.story_ids)
        job = TrainingJob(
            inputs.contract_sha256,
            f"independent-{task_id}",
            "independent_noun",
            story_ids,
            tuple(shard.shard_id for shard in task_shards),
        )
        artifact = train_or_load_lora(
            job,
            _story_batches(inputs, tuple(entry_lookup[story] for story in story_ids), job),
            base_params,
            model_config,
            inputs.checkpoint_directory / "independent-noun",
            inputs.work_directory / "training",
            progress=_training_progress(
                progress,
                "independent-noun",
                completed_updates,
                completed_runtime,
            ),
        )
        completed_updates += artifact.optimizer_updates
        completed_runtime += artifact.runtime_seconds
        progress.update(
            "independent-noun",
            completed_updates,
            status="complete" if completed_updates == _job_total(progress, "independent-noun") else "running",
            elapsed_seconds=completed_runtime,
        )
        independent.append((task_id, artifact))

    all_story_ids = tuple(story for shard in inputs.shards for story in shard.story_ids)
    all_entries = tuple(entry_lookup[story] for story in all_story_ids)
    iid_lora_job = TrainingJob(
        inputs.contract_sha256,
        "joint-iid-lora",
        "joint_iid_lora",
        all_story_ids,
        tuple(shard.shard_id for shard in inputs.shards),
    )
    iid_lora = train_or_load_lora(
        iid_lora_job,
        _story_batches(inputs, all_entries, iid_lora_job),
        base_params,
        model_config,
        inputs.checkpoint_directory / "joint-iid-lora",
        inputs.work_directory / "training",
        progress=_training_progress(progress, "iid-lora", 0, 0.0),
    )
    progress.update(
        "iid-lora",
        iid_lora.optimizer_updates,
        status="complete",
        elapsed_seconds=iid_lora.runtime_seconds,
        metrics={"last_loss": _last_loss(iid_lora.directory / "losses.jsonl")},
    )
    iid_full_job = TrainingJob(
        inputs.contract_sha256,
        "joint-iid-full-model",
        "joint_iid_full_model",
        all_story_ids,
        tuple(shard.shard_id for shard in inputs.shards),
    )
    iid_full = train_or_load_full_model(
        iid_full_job,
        _story_batches(inputs, all_entries, iid_full_job),
        inputs.selected_base,
        inputs.checkpoint_directory / "joint-iid-full-model",
        inputs.work_directory / "training",
        progress=_training_progress(progress, "iid-full-model", 0, 0.0),
    )
    progress.update(
        "iid-full-model",
        iid_full.optimizer_updates,
        status="complete",
        elapsed_seconds=iid_full.runtime_seconds,
        metrics={"last_loss": _last_loss(iid_full.directory / "losses.jsonl")},
    )
    return SharedArtifacts(tuple(level_zero), tuple(independent), iid_lora, iid_full)


def run_or_resume_order_training(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    order: TemporalOrder,
    progress: ProgressRecorder,
) -> OrderingArtifacts:
    """Train or strict-load one sequential stream and every synchronous merge."""
    entry_lookup = inputs.train_entry_lookup
    arrivals = temporal_arrivals(inputs.shards, order)
    sequential: list[AdapterArtifact] = []
    previous: AdapterArtifact | None = None
    sequential_completed = 0
    sequential_runtime = 0.0
    state = empty_hierarchy(order)
    chunks: dict[str, AdapterArtifact] = {}
    all_merges: list[TemporalMerge] = []
    short_merge_completed = 0
    short_merge_runtime = 0.0
    all_merge_runtime = 0.0
    for arrival_index, shard in enumerate(arrivals, start=1):
        sequential_job = TrainingJob(
            inputs.contract_sha256,
            f"sequential-{order}-{arrival_index:03d}",
            "sequential",
            shard.story_ids,
            (shard.shard_id,),
            order=order,
            level=0,
            start_arrival=arrival_index,
            end_arrival=arrival_index,
            initial_adapter_sha256=(
                None if previous is None else previous.adapter_sha256
            ),
        )
        previous = train_or_load_lora(
            sequential_job,
            _story_batches(
                inputs,
                tuple(entry_lookup[story] for story in shard.story_ids),
                sequential_job,
            ),
            inputs.loaded_base.params,
            inputs.loaded_base.config,
            inputs.checkpoint_directory / "sequential" / order,
            inputs.work_directory / "training",
            initial_adapter=None if previous is None else previous.adapter,
            progress=_training_progress(
                progress,
                f"sequential-{order}",
                sequential_completed,
                sequential_runtime,
            ),
        )
        sequential_completed += previous.optimizer_updates
        sequential_runtime += previous.runtime_seconds
        sequential.append(previous)
        progress.update(
            f"sequential-{order}",
            sequential_completed,
            status="complete" if arrival_index == ARRIVAL_COUNT else "running",
            elapsed_seconds=sequential_runtime,
        )
        state, merges = insert_arrival(state, shard)
        level_zero = next(
            chunk
            for chunk in (
                *state.active_chunks,
                *(candidate for merge in merges for candidate in (merge.left, merge.right)),
            )
            if chunk.level == 0 and chunk.end_arrival == arrival_index
        )
        chunks[level_zero.chunk_id] = shared.level_zero_by_shard[shard.shard_id]
        for merge in merges:
            story_ids = tuple(
                story
                for shard_id in merge.parent.shard_ids
                for story in next(
                    source.story_ids for source in inputs.shards if source.shard_id == shard_id
                )
            )
            job = TrainingJob(
                inputs.contract_sha256,
                f"merge-{order}-{merge.parent.start_arrival:03d}-{merge.parent.end_arrival:03d}",
                "merge",
                story_ids,
                merge.parent.shard_ids,
                lineage_ids=merge.parent.parent_chunk_ids,
                order=order,
                level=merge.parent.level,
                start_arrival=merge.parent.start_arrival,
                end_arrival=merge.parent.end_arrival,
            )
            estimated_seconds = len(_story_batches(
                inputs,
                tuple(entry_lookup[story] for story in story_ids),
                job,
            )) / 16.0
            progress_id = (
                _merge_progress_id(order, merge.parent.chunk_id)
                if estimated_seconds > 300.0
                else f"merges-{order}-short"
            )
            offset = 0 if estimated_seconds > 300.0 else short_merge_completed
            artifact = train_or_load_lora(
                job,
                _story_batches(
                    inputs,
                    tuple(entry_lookup[story] for story in story_ids),
                    job,
                ),
                inputs.loaded_base.params,
                inputs.loaded_base.config,
                inputs.checkpoint_directory / "merges" / order,
                inputs.work_directory / "training",
                progress=_training_progress(
                    progress,
                    progress_id,
                    offset,
                    0.0 if estimated_seconds > 300.0 else short_merge_runtime,
                ),
            )
            if estimated_seconds <= 300.0:
                short_merge_completed += artifact.optimizer_updates
                short_merge_runtime += artifact.runtime_seconds
                completed = short_merge_completed
                status = "complete" if completed == _job_total(progress, progress_id) else "running"
                elapsed = short_merge_runtime
            else:
                completed = artifact.optimizer_updates
                status = "complete"
                elapsed = artifact.runtime_seconds
            all_merge_runtime += artifact.runtime_seconds
            progress.update(
                progress_id,
                completed,
                status=status,
                elapsed_seconds=elapsed,
                detail={
                    "active_chunk_count": len(state.active_chunks),
                    "active_intervals": [
                        f"{chunk.start_arrival}-{chunk.end_arrival}"
                        for chunk in state.active_chunks
                    ],
                    "arrival": arrival_index,
                    "order": order,
                },
            )
            chunks[merge.parent.chunk_id] = artifact
            all_merges.append(merge)
        active_ids = {chunk.chunk_id for chunk in state.active_chunks}
        progress.update(
            f"stack-{order}",
            arrival_index,
            status="complete" if arrival_index == ARRIVAL_COUNT else "running",
            elapsed_seconds=sequential_runtime + all_merge_runtime,
            detail={
                "active_adapter_bytes": sum(
                    (chunks[chunk_id].directory / "adapter.safetensors").stat().st_size
                    for chunk_id in active_ids
                ),
                "active_chunk_count": len(active_ids),
                "active_intervals": [
                    f"{chunk.start_arrival}-{chunk.end_arrival} (L{chunk.level})"
                    for chunk in state.active_chunks
                ],
                "archive_adapter_bytes": sum(
                    (artifact.directory / "adapter.safetensors").stat().st_size
                    for artifact in chunks.values()
                ),
                "archive_chunk_count": len(chunks),
                "arrival": arrival_index,
                "carry": [
                    f"L{merge.left.level}→L{merge.parent.level}"
                    for merge in merges
                ],
                "order": order,
            },
        )
    if tuple((chunk.start_arrival, chunk.end_arrival) for chunk in state.active_chunks) != expected_final_intervals():
        raise RuntimeError("completed temporal hierarchy has the wrong final intervals")
    return OrderingArtifacts(
        order,
        tuple(sequential),
        tuple(sorted(chunks.items())),
        state,
        tuple(all_merges),
    )


def build_validation_cases(
    inputs: TemporalStudyInputs,
) -> tuple[tuple[str, tuple[MidpointCase, ...]], ...]:
    """Materialize the 4,440 structurally isolated midpoint evaluation cases."""
    from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedStoryStore

    store = IndexedStoryStore(inputs.partition)
    return tuple(
        (
            task,
            tuple(
                build_midpoint_case(
                    inputs.partition,
                    store,
                    task,
                    entry,
                    context_length=CONTEXT_LENGTH,
                    maximum_position_embeddings=inputs.loaded_base.config.max_position_embeddings,
                )
                for entry in entries
            ),
        )
        for task, entries in inputs.validation_entries
    )


def run_or_resume_order_evaluation(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    ordering: OrderingArtifacts,
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    progress: ProgressRecorder,
) -> Path:
    """Replay an ordering into bounded per-stage, per-method JSONL ledgers."""
    order = ordering.order
    ledger_directory = inputs.work_directory / f"evaluation-{order}"
    ledger_directory.mkdir(parents=True, exist_ok=True)
    _reject_unexpected_evaluation_files(
        ledger_directory,
        _expected_order_evaluation_filenames(order),
    )
    _validate_evaluation_file_prefixes(
        ledger_directory,
        _expected_order_evaluation_file_counts(inputs, order, cases_by_task),
        inputs.contract_sha256,
    )
    chunk_artifacts = ordering.chunks_by_id
    sentinel_sets = {
        task_id: frozenset(story_ids) for task_id, story_ids in inputs.sentinel
    }
    state = empty_hierarchy(order)
    encountered: set[str] = set()
    encountered_counts: Counter[str] = Counter()
    sentinel_completed = _evaluation_directory_row_count(
        ledger_directory,
        "sentinel-",
    )
    macro_completed = _evaluation_directory_row_count(
        ledger_directory,
        "macro-",
    )
    for arrival_index, shard in enumerate(temporal_arrivals(inputs.shards, order), start=1):
        encountered.add(shard.task_id)
        encountered_counts[shard.task_id] += 1
        state, _ = insert_arrival(state, shard)
        temporal_candidates = tuple(
            _chunk_candidate(chunk, chunk_artifacts[chunk.chunk_id])
            for chunk in state.active_chunks
        )
        log_bank = build_adapter_bank(temporal_candidates, inputs.loaded_base.config)
        sequential_artifact = ordering.sequential[arrival_index - 1]
        sequential_bank = build_adapter_bank(
            (
                AdapterCandidate(
                    f"sequential-{arrival_index:03d}",
                    sequential_artifact.adapter_sha256,
                    sequential_artifact.adapter,
                    tuple(
                        (task_id, encountered_counts[task_id])
                        for task_id in TASK_IDS
                        if encountered_counts[task_id]
                    ),
                    level=0,
                    start_arrival=arrival_index,
                    end_arrival=arrival_index,
                ),
            ),
            inputs.loaded_base.config,
        )
        base_bank = build_adapter_bank((), inputs.loaded_base.config)
        sentinel_cases = tuple(
            case
            for task in TASK_IDS
            if task in encountered
            for case in cases_by_task[task]
            if case.entry.story_id in sentinel_sets[task]
        )
        sentinel_completed = _evaluate_methods(
            inputs,
            ledger_directory,
            sentinel_cases,
            order,
            arrival_index,
            "sentinel",
            log_bank,
            sequential_bank,
            base_bank,
            progress,
            f"sentinel-{order}",
            sentinel_completed,
        )
        if arrival_index % MACRO_CHECKPOINT_INTERVAL == 0:
            macro_cases = tuple(
                case
                for task in TASK_IDS
                if task in encountered
                for case in cases_by_task[task]
            )
            macro_completed = _evaluate_methods(
                inputs,
                ledger_directory,
                macro_cases,
                order,
                arrival_index,
                "macro",
                log_bank,
                sequential_bank,
                base_bank,
                progress,
                f"macro-{order}",
                macro_completed,
            )
    return ledger_directory


def run_or_resume_final_controls(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    progress: ProgressRecorder,
) -> Path:
    """Stream each requested offline final control to its own bounded ledger."""
    ledger_directory = inputs.work_directory / "evaluation-final-controls"
    ledger_directory.mkdir(parents=True, exist_ok=True)
    _reject_unexpected_evaluation_files(
        ledger_directory,
        tuple(
            f"final-stage-{ARRIVAL_COUNT:03d}-{method}.jsonl"
            for method in (
                "independent_noun_exhaustive",
                "joint_iid_lora",
                "joint_iid_full_model",
            )
        ),
    )
    _validate_evaluation_file_prefixes(
        ledger_directory,
        tuple(
            (
                _evaluation_filename("final", ARRIVAL_COUNT, method),
                sum(len(cases_by_task[task_id]) for task_id in TASK_IDS),
            )
            for method in (
                "independent_noun_exhaustive",
                "joint_iid_lora",
                "joint_iid_full_model",
            )
        ),
        inputs.contract_sha256,
    )
    cases = tuple(case for task in TASK_IDS for case in cases_by_task[task])
    independent = build_adapter_bank(
        tuple(
            AdapterCandidate(
                f"noun-{task}",
                artifact.adapter_sha256,
                artifact.adapter,
                ((task, SHARDS_PER_TASK),),
            )
            for task, artifact in shared.independent_noun
        ),
        inputs.loaded_base.config,
    )
    iid_lora = build_adapter_bank(
        (
            AdapterCandidate(
                "joint-iid-lora",
                shared.iid_lora.adapter_sha256,
                shared.iid_lora.adapter,
                tuple((task, SHARDS_PER_TASK) for task in TASK_IDS),
            ),
        ),
        inputs.loaded_base.config,
    )
    full = load_gpt_neo_checkpoint(shared.iid_full_model.checkpoint)
    base_bank = build_adapter_bank((), inputs.loaded_base.config)
    methods = (
        ("independent_noun_exhaustive", "exhaustive", inputs.loaded_base.params, independent),
        ("joint_iid_lora", "forced_adapter", inputs.loaded_base.params, iid_lora),
        ("joint_iid_full_model", "forced_base", full.params, base_bank),
    )
    completed = _evaluation_directory_row_count(ledger_directory, "final-")
    for method, routing, params, bank in methods:
        ledger = ChainedJsonlLedger(
            ledger_directory / _evaluation_filename("final", ARRIVAL_COUNT, method),
            EVALUATION_ROW_FORMAT,
        )
        before = len(ledger.rows)
        progress_offset = completed - before
        evaluate_to_ledger(
            cases,
            contract_sha256=inputs.contract_sha256,
            evaluation_id="final-controls",
            dataset="final",
            method=method,
            order=None,
            stage=ARRIVAL_COUNT,
            routing=routing,  # type: ignore[arg-type]
            base_params=params,
            model_config=inputs.loaded_base.config,
            bank=bank,
            ledger=ledger,
            progress=_evaluation_progress(
                progress,
                "final-controls",
                progress_offset,
            ),
        )
        completed = progress_offset + len(ledger.rows)
        progress.update(
            "final-controls",
            completed,
            status="complete" if completed == _job_total(progress, "final-controls") else "running",
            elapsed_seconds=_job_elapsed(progress, "final-controls"),
        )
    return ledger_directory


def _evaluate_methods(
    inputs: TemporalStudyInputs,
    ledger_directory: Path,
    cases: Sequence[MidpointCase],
    order: TemporalOrder,
    stage: int,
    dataset: str,
    log_bank,
    sequential_bank,
    base_bank,
    progress: ProgressRecorder,
    progress_id: str,
    completed: int,
) -> int:
    methods = (
        ("base", "forced_base", base_bank),
        ("sequential_lora", "forced_adapter", sequential_bank),
        ("log_t", "exhaustive", log_bank),
    )
    for method, routing, bank in methods:
        ledger = ChainedJsonlLedger(
            ledger_directory / _evaluation_filename(dataset, stage, method),
            EVALUATION_ROW_FORMAT,
        )
        before = len(ledger.rows)
        progress_offset = completed - before
        evaluate_to_ledger(
            cases,
            contract_sha256=inputs.contract_sha256,
            evaluation_id=f"{dataset}-{order}-{stage:03d}",
            dataset=dataset,  # type: ignore[arg-type]
            method=method,
            order=order,
            stage=stage,
            routing=routing,  # type: ignore[arg-type]
            base_params=inputs.loaded_base.params,
            model_config=inputs.loaded_base.config,
            bank=bank,
            ledger=ledger,
            progress=_evaluation_progress(progress, progress_id, progress_offset),
        )
        completed = progress_offset + len(ledger.rows)
        progress.update(
            progress_id,
            completed,
            status="complete" if completed == _job_total(progress, progress_id) else "running",
            elapsed_seconds=_job_elapsed(progress, progress_id),
            detail={
                "active_chunk_count": len(log_bank.candidates),
                "active_intervals": [candidate.candidate_id for candidate in log_bank.candidates],
                "arrival": stage,
                "order": order,
            },
        )
    return completed


def expected_order_evaluation_keys(
    inputs: TemporalStudyInputs,
    order: TemporalOrder,
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
) -> tuple[tuple[object, ...], ...]:
    """Return exact append order for one temporal stream's evaluation ledger."""
    sentinel_sets = {
        task_id: frozenset(story_ids) for task_id, story_ids in inputs.sentinel
    }
    encountered: set[str] = set()
    keys: list[tuple[object, ...]] = []
    for stage, shard in enumerate(temporal_arrivals(inputs.shards, order), start=1):
        encountered.add(shard.task_id)
        sentinel_cases = tuple(
            case
            for task_id in TASK_IDS
            if task_id in encountered
            for case in cases_by_task[task_id]
            if case.entry.story_id in sentinel_sets[task_id]
        )
        keys.extend(
            _evaluation_keys_for_methods(
                sentinel_cases,
                f"sentinel-{order}-{stage:03d}",
                "sentinel",
                order,
                stage,
                ("base", "sequential_lora", "log_t"),
            )
        )
        if stage % MACRO_CHECKPOINT_INTERVAL == 0:
            macro_cases = tuple(
                case
                for task_id in TASK_IDS
                if task_id in encountered
                for case in cases_by_task[task_id]
            )
            keys.extend(
                _evaluation_keys_for_methods(
                    macro_cases,
                    f"macro-{order}-{stage:03d}",
                    "macro",
                    order,
                    stage,
                    ("base", "sequential_lora", "log_t"),
                )
            )
    return tuple(keys)


def expected_final_control_keys(
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
) -> tuple[tuple[object, ...], ...]:
    """Return exact append order for the three offline final controls."""
    cases = tuple(case for task_id in TASK_IDS for case in cases_by_task[task_id])
    return _evaluation_keys_for_methods(
        cases,
        "final-controls",
        "final",
        None,
        ARRIVAL_COUNT,
        (
            "independent_noun_exhaustive",
            "joint_iid_lora",
            "joint_iid_full_model",
        ),
    )


def _evaluation_keys_for_methods(
    cases: Sequence[MidpointCase],
    evaluation_id: str,
    dataset: str,
    order: str | None,
    stage: int,
    methods: Sequence[str],
) -> tuple[tuple[object, ...], ...]:
    case_order = tuple(
        case
        for bucket in sorted({case.prefix_width_bucket for case in cases})
        for case in cases
        if case.prefix_width_bucket == bucket
    )
    return tuple(
        (
            evaluation_id,
            dataset,
            method,
            order,
            stage,
            case.task_id,
            case.entry.story_id,
        )
        for method in methods
        for case in case_order
    )


def _require_evaluation_prefix(
    rows: Sequence[Mapping[str, object]],
    contract_sha256: str,
    expected: Sequence[tuple[object, ...]],
) -> None:
    observed = tuple(
        (
            row.get("evaluation_id"),
            row.get("dataset"),
            row.get("method"),
            row.get("order"),
            row.get("stage"),
            row.get("task_id"),
            row.get("story_id"),
        )
        for row in rows
    )
    if (
        any(row.get("contract_sha256") != contract_sha256 for row in rows)
        or observed != tuple(expected[: len(observed)])
    ):
        raise ValueError("temporal evaluation ledger is not a canonical prefix")


def _evaluation_filename(dataset: str, stage: int, method: str) -> str:
    return f"{dataset}-stage-{stage:03d}-{method}.jsonl"


def _expected_order_evaluation_filenames(order: TemporalOrder) -> tuple[str, ...]:
    del order
    methods = ("base", "sequential_lora", "log_t")
    return tuple(
        _evaluation_filename(dataset, stage, method)
        for stage in range(1, ARRIVAL_COUNT + 1)
        for dataset in (
            ("sentinel", "macro")
            if stage % MACRO_CHECKPOINT_INTERVAL == 0
            else ("sentinel",)
        )
        for method in methods
    )


def _expected_order_evaluation_file_counts(
    inputs: TemporalStudyInputs,
    order: TemporalOrder,
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
) -> tuple[tuple[str, int], ...]:
    encountered: set[str] = set()
    specifications: list[tuple[str, int]] = []
    for stage, shard in enumerate(temporal_arrivals(inputs.shards, order), start=1):
        encountered.add(shard.task_id)
        counts = [("sentinel", SENTINEL_STORIES_PER_TASK * len(encountered))]
        if stage % MACRO_CHECKPOINT_INTERVAL == 0:
            counts.append(
                (
                    "macro",
                    sum(len(cases_by_task[task_id]) for task_id in encountered),
                )
            )
        specifications.extend(
            (_evaluation_filename(dataset, stage, method), count)
            for dataset, count in counts
            for method in ("base", "sequential_lora", "log_t")
        )
    return tuple(specifications)


def _reject_unexpected_evaluation_files(
    directory: Path,
    expected_names: Sequence[str],
) -> None:
    observed = {path.name for path in directory.iterdir()}
    unexpected = observed - set(expected_names)
    if unexpected or any(not path.is_file() for path in directory.iterdir()):
        raise ValueError(
            f"temporal evaluation directory contains unexpected entries: "
            f"{tuple(sorted(unexpected))}"
        )


def _evaluation_directory_row_count(directory: Path, prefix: str) -> int:
    return sum(
        len(ChainedJsonlLedger(path, EVALUATION_ROW_FORMAT).rows)
        for path in sorted(directory.glob(f"{prefix}*.jsonl"))
    )


def _validate_evaluation_file_prefixes(
    directory: Path,
    specifications: Sequence[tuple[str, int]],
    contract_sha256: str,
) -> None:
    encountered_gap = False
    for name, expected_count in specifications:
        path = directory / name
        if not path.is_file():
            encountered_gap = True
            continue
        ledger = ChainedJsonlLedger(path, EVALUATION_ROW_FORMAT)
        count = len(ledger.rows)
        if (
            count > expected_count
            or (encountered_gap and count)
            or any(
                row.get("contract_sha256") != contract_sha256
                for row in ledger.rows
            )
        ):
            raise ValueError("temporal evaluation files are not one canonical prefix")
        if count < expected_count:
            encountered_gap = True


def _story_batches(
    inputs: TemporalStudyInputs,
    entries: Sequence[StoryIndexEntry],
    job: TrainingJob,
) -> StoryEpochBatches:
    return StoryEpochBatches(
        inputs.partition,
        entries,
        context_length=CONTEXT_LENGTH,
        batch_size=PHYSICAL_BATCH_SIZE,
        namespace=job.identity_sha256,
    )


def _optimizer_updates(
    story_ids: Sequence[str],
    entry_lookup: Mapping[str, StoryIndexEntry],
) -> int:
    windows = sum(
        math.ceil((entry_lookup[story_id].token_count - 1) / CONTEXT_LENGTH)
        for story_id in story_ids
    )
    return math.ceil(windows / PHYSICAL_BATCH_SIZE) * 4


def _timing_shape_count(inputs: TemporalStudyInputs) -> int:
    candidate_counts: set[int] = set()
    for order in TEMPORAL_ORDERS:
        state = empty_hierarchy(order)
        for shard in temporal_arrivals(inputs.shards, order):
            state, _ = insert_arrival(state, shard)
            candidate_counts.add(len(state.active_chunks))
    prefix_buckets = {
        32 * math.ceil(((entry.token_count // 2) - 1) / 32)
        for _, entries in inputs.validation_entries
        for entry in entries
    }
    return len(candidate_counts) * (len(prefix_buckets) + 1)


def _chunk_candidate(
    chunk: TemporalChunk,
    artifact: AdapterArtifact,
) -> AdapterCandidate:
    return AdapterCandidate(
        f"interval-{chunk.start_arrival:03d}-{chunk.end_arrival:03d}-l{chunk.level}",
        artifact.adapter_sha256,
        artifact.adapter,
        chunk.task_counts,
        level=chunk.level,
        start_arrival=chunk.start_arrival,
        end_arrival=chunk.end_arrival,
    )


def _training_progress(
    recorder: ProgressRecorder,
    progress_id: str,
    offset: int,
    elapsed_offset: float,
):
    last_reported = [-1]

    def update(job_id: str, completed: int, total: int, loss: float, elapsed: float) -> None:
        del job_id
        aggregate = offset + completed
        if completed == 1 or completed == total or aggregate - last_reported[0] >= 32:
            recorder.update(
                progress_id,
                aggregate,
                status="running",
                elapsed_seconds=elapsed_offset + elapsed,
                metrics={"current_loss": loss},
            )
            last_reported[0] = aggregate

    return update


def _evaluation_progress(
    recorder: ProgressRecorder,
    progress_id: str,
    offset: int,
):
    started = monotonic()
    elapsed_offset = _job_elapsed(recorder, progress_id)

    def update(completed: int, total: int, metrics: dict[str, float]) -> None:
        del total
        recorder.update(
            progress_id,
            offset + completed,
            status="running",
            elapsed_seconds=elapsed_offset + monotonic() - started,
            metrics={key: round(value, 6) for key, value in metrics.items()},
        )

    return update


def _job_total(recorder: ProgressRecorder, job_id: str) -> int:
    return next(job.total for job in recorder.jobs if job.job_id == job_id)


def _job_elapsed(recorder: ProgressRecorder, job_id: str) -> float:
    rows = tuple(row for row in recorder.ledger.rows if row.get("job_id") == job_id)
    return float(rows[-1]["elapsed_seconds"]) if rows else 0.0


def _last_loss(path: Path) -> float:
    rows = ChainedJsonlLedger(path, f"{STUDY_ID}-training-row-v1").rows
    return float(rows[-1]["loss"])


def _merge_progress_id(order: TemporalOrder, chunk_id: str) -> str:
    return f"merge-{order}-{chunk_id[:16]}"


def _canonical_run_manifest(path: Path) -> dict[str, object]:
    record = load_canonical_json(path)
    supplied = record.get("run_sha256")
    core = {key: value for key, value in record.items() if key != "run_sha256"}
    if (
        record.get("format") != RUN_MANIFEST_FORMAT
        or record.get("phase") not in ("local_complete", "complete_with_judge")
        or supplied != record_sha256(core)
    ):
        raise ValueError("canonical nouns-v2 run manifest changed")
    return record


def _authenticate_canonical_ledgers(
    result_directory: Path,
    partition: NounsV2PartitionArtifact,
    vamp_stages: tuple[LanguageAdaptationArtifact, ...],
    validation_entries: tuple[tuple[str, tuple[StoryIndexEntry, ...]], ...],
    run_manifest: Mapping[str, object],
) -> None:
    """Validate canonical result identities, coverage, and manifest hashes."""
    story_keys = {
        (task_id, entry.story_id)
        for task_id, entries in validation_entries
        for entry in entries
    }
    _validate_result_ledger(
        result_directory / "whole-story-nll.jsonl",
        WHOLE_STORY_FORMAT,
        {
            (task_id, story_id, condition)
            for task_id, story_id in story_keys
            for condition in CONDITIONS
        },
        ("task_noun", "story_id", "condition"),
    )
    _validate_result_ledger(
        result_directory / "half-story-generations.jsonl",
        HALF_STORY_FORMAT,
        story_keys,
        ("task_noun", "story_id"),
    )
    validate_stagewise_ledger(
        result_directory / "stagewise-cl.jsonl",
        partition,
        vamp_stages,
        require_complete=True,
    )
    for name, row_format in (
        ("baseline-stagewise-cl.jsonl", BASELINE_STAGEWISE_FORMAT),
        ("full-finetune-stagewise-cl.jsonl", FULL_FINETUNE_STAGEWISE_FORMAT),
    ):
        _validate_self_hashed_ledger(result_directory / name, row_format, 72_256)
    compact_contract = load_canonical_json(
        result_directory / "compact-stagewise-contract.json"
    )
    compact_core = {
        key: value
        for key, value in compact_contract.items()
        if key != "contract_sha256"
    }
    if (
        compact_contract.get("format")
        != "tinyworlds-nouns-v2-compact-stagewise-contract-v1"
        or compact_contract.get("contract_sha256") != record_sha256(compact_core)
    ):
        raise ValueError("canonical compact-stagewise contract changed")
    _validate_self_hashed_ledger(
        result_directory / "compact-stagewise-cl.jsonl",
        "tinyworlds-nouns-v2-compact-stagewise-row-v1",
        72_256,
    )
    expected_hashes = {
        "baseline-stagewise-cl.jsonl": run_manifest.get("baseline_stagewise_sha256"),
        "compact-stagewise-cl.jsonl": run_manifest.get("compact_stagewise_sha256"),
        "full-finetune-stagewise-cl.jsonl": run_manifest.get(
            "full_finetune_stagewise_sha256"
        ),
        "report.html": run_manifest.get("report_html_sha256"),
        "report.md": run_manifest.get("report_markdown_sha256"),
        "stagewise-cl.jsonl": run_manifest.get("vamp_stagewise_sha256"),
    }
    changed = tuple(
        name
        for name, expected in expected_hashes.items()
        if expected != file_sha256(result_directory / name)
    )
    if changed:
        raise ValueError(f"canonical run manifest artifact hashes changed: {changed}")


def _validate_result_ledger(
    path: Path,
    expected_format: str,
    expected_keys: set[tuple[str, ...]],
    key_fields: tuple[str, ...],
) -> None:
    """Validate a canonical self-hashed result ledger and its exact keys."""
    observed: set[tuple[str, ...]] = set()
    for record in _canonical_jsonl_objects(path):
        supplied = record.get("result_sha256")
        core = {key: value for key, value in record.items() if key != "result_sha256"}
        key = tuple(str(record.get(field)) for field in key_fields)
        if (
            record.get("format") != expected_format
            or supplied != record_sha256(core)
            or key in observed
            or key not in expected_keys
        ):
            raise ValueError(f"canonical ledger row changed: {path.name}")
        observed.add(key)
    if observed != expected_keys:
        raise ValueError(f"canonical ledger coverage changed: {path.name}")


def _validate_self_hashed_ledger(
    path: Path,
    expected_format: str,
    expected_count: int,
) -> None:
    """Validate every row hash and the exact size of a canonical ledger."""
    count = 0
    hashes: set[str] = set()
    for record in _canonical_jsonl_objects(path):
        supplied = record.get("result_sha256")
        core = {key: value for key, value in record.items() if key != "result_sha256"}
        if (
            record.get("format") != expected_format
            or supplied != record_sha256(core)
            or supplied in hashes
        ):
            raise ValueError(f"canonical ledger identity changed: {path.name}")
        hashes.add(str(supplied))
        count += 1
    if count != expected_count:
        raise ValueError(f"canonical ledger row count changed: {path.name}")


def _canonical_jsonl_objects(path: Path) -> Iterator[dict[str, object]]:
    """Yield strict canonical JSONL objects without retaining the full ledger."""
    with path.open("rb") as source:
        for line in source:
            value = json.loads(line)
            if (
                not line.endswith(b"\n")
                or type(value) is not dict
                or canonical_json_bytes(value) != line
            ):
                raise ValueError(f"ledger is not canonical JSONL: {path}")
            yield value


__all__ = [
    "OrderingArtifacts",
    "SharedArtifacts",
    "TemporalStudyInputs",
    "assert_canonical_artifacts_unchanged",
    "authenticate_temporal_study_inputs",
    "build_validation_cases",
    "canonical_artifact_hashes",
    "expected_final_control_keys",
    "expected_order_evaluation_keys",
    "run_or_resume_final_controls",
    "run_or_resume_order_evaluation",
    "run_or_resume_order_training",
    "run_or_resume_shared_training",
    "study_dashboard_jobs",
]
