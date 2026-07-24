"""Resumable task-boundary training for the semantic-v6 VAMP comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp

from apm.continual.language_adaptation_artifact import (
    LanguageAdaptationArtifact,
    extract_language_adaptation_artifact,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_baseline_training import (
    IndependentRootAdapter,
    IndependentRootLoraProgress,
    LanguageAdaptationBaselines,
    SequentialLoraProgress,
    SequentialLoraStage,
    advance_independent_root_lora_progress,
    advance_sequential_lora_progress,
    complete_independent_root_lora_progress,
    complete_sequential_lora_progress,
    init_independent_root_lora_progress,
    init_sequential_lora_progress,
)
from apm.continual.language_run import (
    LanguageStageMetrics,
    LanguageVampRun,
    advance_language_vamp_run,
    init_language_vamp_run,
    score_parent_nodes,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes
from apm.data.text.tinyworlds_p_semantic.v6_milestone import (
    V6SelectedBase,
    load_v6_selected_base,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_curriculum import (
    V6PreparedVampCurriculum,
)
from apm.lm.checkpoint import BaseCheckpointRef, load_gpt_neo_checkpoint


V6_VAMP_ADAPTATION_FORMAT = "tinyworlds-p-semantic-v6-vamp-adaptation"
V6_VAMP_ADAPTATION_TREE_FORMAT = (
    "tinyworlds-p-semantic-v6-vamp-adaptation-tree"
)
V6VampTrainingProgress = Callable[[str, str, int, float, int], None]


@dataclass(frozen=True, slots=True)
class V6VampAdaptationPublication:
    """A complete five-world adapter artifact whose test data remains sealed."""

    directory: Path
    run_sha256: str
    partition_sha256: str
    selected_base_sha256: str
    curriculum_sha256: str
    config_sha256: str
    allocator_peak_bytes: int
    adaptation: LanguageAdaptationArtifact

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        for value, label in (
            (self.run_sha256, "semantic-v6 VAMP run"),
            (self.partition_sha256, "semantic-v6 VAMP partition"),
            (self.selected_base_sha256, "semantic-v6 VAMP selected base"),
            (self.curriculum_sha256, "semantic-v6 VAMP curriculum"),
            (self.config_sha256, "semantic-v6 VAMP config"),
        ):
            _require_sha256(value, label)
        if (
            type(self.allocator_peak_bytes) is not int
            or self.allocator_peak_bytes < 0
        ):
            raise ValueError("semantic-v6 VAMP allocator peak must be nonnegative")
        if (
            self.allocator_peak_bytes
            > V6_VAMP_EXPERIMENT_PRESET.allocator_peak_limit_bytes
        ):
            raise ValueError("semantic-v6 VAMP allocator peak exceeds its frozen limit")


def train_or_resume_v6_vamp_adaptations(
    prepared: V6PreparedVampCurriculum,
    selected_base: V6SelectedBase,
    working_directory: str | Path,
    publication_root: str | Path,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
    *,
    progress: V6VampTrainingProgress | None = None,
) -> V6VampAdaptationPublication:
    """Train all baselines, checkpoint every world, and publish the frozen adapters."""
    if type(prepared) is not V6PreparedVampCurriculum:
        raise TypeError("semantic-v6 VAMP requires its prepared curriculum")
    if type(selected_base) is not V6SelectedBase:
        raise TypeError("semantic-v6 VAMP requires its selected base")
    selected_base = load_v6_selected_base(selected_base.directory)
    _require_bindings(prepared, selected_base, preset)
    checkpoint = load_gpt_neo_checkpoint(selected_base.checkpoint.directory)
    base_params = checkpoint.params
    model_config = checkpoint.config
    working = Path(working_directory)
    stages_root = working / "stages"
    stages_root.mkdir(parents=True, exist_ok=True)
    existing_stages = _completed_stage_directories(stages_root, len(preset.task_order))
    if existing_stages:
        artifact = load_language_adaptation_artifact(existing_stages[-1])
        sequential, independent, vamp = _restore_progress(
            artifact,
            prepared,
            selected_base.checkpoint,
            preset,
        )
    else:
        sequential_key, independent_key, vamp_key = jax.random.split(
            jax.random.PRNGKey(preset.seed),
            3,
        )
        sequential = init_sequential_lora_progress(
            base_params,
            model_config,
            preset.lora_config,
            preset.train_config,
            sequential_key,
        )
        independent = init_independent_root_lora_progress(
            base_params,
            model_config,
            preset.train_config,
            independent_key,
        )
        vamp = init_language_vamp_run(
            selected_base.checkpoint,
            base_params,
            model_config,
            prepared.root_validation_probes,
            vamp_key,
            max_nodes=preset.max_nodes,
            max_edges=preset.max_edges,
            key_probe_count=preset.root_probe_count,
            evaluation_microbatch_size=preset.evaluation_microbatch_size,
        )
    completed_count = len(vamp.completed_tasks)
    for stage_index, task in enumerate(
        prepared.curriculum.tasks[completed_count:],
        start=completed_count + 1,
    ):
        world = str(task.task_id)
        sequential = advance_sequential_lora_progress(
            sequential,
            task,
            base_params,
            model_config,
            preset.lora_config,
            training_progress=_method_progress(progress, "sequential", world),
        )
        independent = advance_independent_root_lora_progress(
            independent,
            task,
            base_params,
            model_config,
            preset.lora_config,
            training_progress=_method_progress(progress, "independent", world),
        )
        parent = score_parent_nodes(
            vamp,
            task.parent_probes,
            base_params,
            model_config,
            preset.lora_config,
            evaluation_microbatch_size=preset.evaluation_microbatch_size,
        )
        vamp = advance_language_vamp_run(
            vamp,
            task,
            base_params,
            model_config,
            preset.lora_config,
            preset.train_config,
            parent,
            key_probe_count=preset.content_key_probe_count,
            evaluation_microbatch_size=preset.evaluation_microbatch_size,
            training_progress=_method_progress(progress, "vamp", world),
        )
        stage_artifact = extract_language_adaptation_artifact(
            _completed_baselines(sequential, independent, vamp, preset),
            model_config,
            preset.lora_config,
            config_hashes=_config_hashes(prepared, selected_base, preset),
        )
        save_language_adaptation_artifact(
            stages_root / f"stage-{stage_index:02d}",
            stage_artifact,
        )
    final_stage = stages_root / f"stage-{len(preset.task_order):02d}"
    final_artifact = load_language_adaptation_artifact(final_stage)
    return _publish_adaptations(
        final_artifact,
        prepared,
        selected_base,
        Path(publication_root),
        preset,
    )


def load_v6_vamp_adaptation_publication(
    directory: str | Path,
) -> V6VampAdaptationPublication:
    """Authenticate the complete test-sealed semantic-v6 adapter publication."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("semantic-v6 VAMP publication must be a directory")
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "run_sha256", "schema_version"}
        or tree.get("format") != V6_VAMP_ADAPTATION_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("run_sha256") != root.name
    ):
        raise ValueError("semantic-v6 VAMP publication tree changed")
    descriptors = tree.get("files")
    if type(descriptors) is not list or any(type(item) is not dict for item in descriptors):
        raise ValueError("semantic-v6 VAMP publication descriptors changed")
    actual_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    described_paths = tuple(_text(item, "relative_path") for item in descriptors)
    if described_paths != actual_paths:
        raise ValueError("semantic-v6 VAMP publication file set changed")
    for descriptor in descriptors:
        path = root / _text(descriptor, "relative_path")
        size = descriptor.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or path.is_symlink()
            or path.stat().st_size != size
            or _file_sha256(path) != _text(descriptor, "sha256")
        ):
            raise ValueError(f"semantic-v6 VAMP publication file changed: {path}")
    manifest = _load_json(root / "manifest.json")
    required = {
        "adaptation_tensor_checksum",
        "allocator_peak_bytes",
        "base_checkpoint",
        "config",
        "config_sha256",
        "curriculum_sha256",
        "format",
        "partition_sha256",
        "run_sha256",
        "sealed_test_opened",
        "selected_base_sha256",
        "task_order",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != V6_VAMP_ADAPTATION_FORMAT
        or manifest.get("sealed_test_opened") is not False
        or manifest.get("run_sha256") != root.name
    ):
        raise ValueError("semantic-v6 VAMP publication manifest changed")
    content = {key: value for key, value in manifest.items() if key != "run_sha256"}
    if record_sha256(content) != root.name:
        raise ValueError("semantic-v6 VAMP publication identity changed")
    adaptation = load_language_adaptation_artifact(root / "adaptations")
    expected_hashes = {
        "semantic-v6-curriculum": _text(manifest, "curriculum_sha256"),
        "semantic-v6-experiment": V6_VAMP_EXPERIMENT_PRESET.config_sha256,
        "semantic-v6-partition": V6_VAMP_EXPERIMENT_PRESET.partition_sha256,
        "semantic-v6-sample-report": V6_VAMP_EXPERIMENT_PRESET.sample_report_sha256,
        "semantic-v6-selected-base": _text(manifest, "selected_base_sha256"),
    }
    base_checkpoint = manifest.get("base_checkpoint")
    if (
        adaptation.tensor_checksum != _text(manifest, "adaptation_tensor_checksum")
        or manifest.get("config") != V6_VAMP_EXPERIMENT_PRESET.as_record()
        or manifest.get("config_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.config_sha256
        or manifest.get("partition_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.partition_sha256
        or manifest.get("task_order") != list(V6_VAMP_EXPERIMENT_PRESET.task_order)
        or dict(adaptation.config_hashes) != expected_hashes
        or type(base_checkpoint) is not dict
        or base_checkpoint
        != {
            "manifest_sha256": adaptation.base_checkpoint.manifest_sha256,
            "parameter_checksum": adaptation.base_checkpoint.parameter_checksum,
        }
    ):
        raise ValueError("semantic-v6 VAMP adaptation checksum changed")
    return V6VampAdaptationPublication(
        directory=root.resolve(),
        run_sha256=root.name,
        partition_sha256=_text(manifest, "partition_sha256"),
        selected_base_sha256=_text(manifest, "selected_base_sha256"),
        curriculum_sha256=_text(manifest, "curriculum_sha256"),
        config_sha256=_text(manifest, "config_sha256"),
        allocator_peak_bytes=_integer(manifest, "allocator_peak_bytes"),
        adaptation=adaptation,
    )


def _completed_baselines(
    sequential: SequentialLoraProgress,
    independent: IndependentRootLoraProgress,
    vamp: LanguageVampRun,
    preset: V6VampExperimentPreset,
) -> LanguageAdaptationBaselines:
    return LanguageAdaptationBaselines(
        sequential_single_lora=complete_sequential_lora_progress(sequential),
        independent_root_lora=complete_independent_root_lora_progress(independent),
        vamp=vamp,
        train_config=preset.train_config,
        base_parameter_checksum=vamp.base_checkpoint.parameter_checksum,
    )


def _restore_progress(
    artifact: LanguageAdaptationArtifact,
    prepared: V6PreparedVampCurriculum,
    base_checkpoint: BaseCheckpointRef,
    preset: V6VampExperimentPreset,
) -> tuple[SequentialLoraProgress, IndependentRootLoraProgress, LanguageVampRun]:
    expected_hashes = _config_hashes(
        prepared,
        load_v6_selected_base(base_checkpoint.directory.parent),
        preset,
    )
    if (
        artifact.base_checkpoint.manifest_sha256 != base_checkpoint.manifest_sha256
        or artifact.base_checkpoint.parameter_checksum
        != base_checkpoint.parameter_checksum
        or artifact.model_config
        != load_gpt_neo_checkpoint(base_checkpoint.directory).config
        or artifact.lora_config != preset.lora_config
        or artifact.train_config != preset.train_config
        or any(dict(artifact.config_hashes).get(name) != value for name, value in expected_hashes.items())
    ):
        raise ValueError("semantic-v6 VAMP resume identity changed")
    task_count = len(artifact.task_order)
    expected_order = tuple(task.task_id for task in prepared.curriculum.tasks[:task_count])
    if artifact.task_order != expected_order:
        raise ValueError("semantic-v6 VAMP resume task order changed")
    sequential = SequentialLoraProgress(
        stages=tuple(
            SequentialLoraStage(
                stage_index=record.stage_index,
                task_id=record.task_id,
                adapter=record.adapter,
                step_losses=record.training_trace,
            )
            for record in artifact.sequential_stages
        ),
        current_adapter=artifact.sequential_stages[-1].adapter,
        rng_key=jnp.asarray(artifact.rng_state.sequential_single_lora),
        train_config=artifact.train_config,
        base_parameter_checksum=artifact.base_checkpoint.parameter_checksum,
    )
    independent = IndependentRootLoraProgress(
        adapters=tuple(
            IndependentRootAdapter(
                task_id=record.task_id,
                adapter=record.adapter,
                step_losses=record.training_trace,
            )
            for record in artifact.independent_adapters
        ),
        rng_key=jnp.asarray(artifact.rng_state.independent_root_lora),
        train_config=artifact.train_config,
        base_parameter_checksum=artifact.base_checkpoint.parameter_checksum,
    )
    vamp = LanguageVampRun(
        base_checkpoint=base_checkpoint,
        graph=artifact.vamp_graph,
        address_book=artifact.address_book,
        rng_key=jnp.asarray(artifact.rng_state.vamp),
        completed_tasks=prepared.curriculum.tasks[:task_count],
        stage_metrics=tuple(
            LanguageStageMetrics(
                stage_index=record.stage_index,
                task_id=record.task_id,
                parent_node_index=record.parent_node_index,
                parent_node_id=record.parent_node_id,
                parent_mean_node_nll=record.parent_mean_node_nll,
                candidate_step_losses=record.training_trace,
                task_metrics=(),
            )
            for record in artifact.vamp_stages
        ),
        max_nodes=artifact.max_nodes,
        max_edges=artifact.max_edges,
    )
    return sequential, independent, vamp


def _config_hashes(
    prepared: V6PreparedVampCurriculum,
    selected_base: V6SelectedBase,
    preset: V6VampExperimentPreset,
) -> dict[str, str]:
    return {
        "semantic-v6-curriculum": prepared.curriculum_sha256,
        "semantic-v6-experiment": preset.config_sha256,
        "semantic-v6-partition": preset.partition_sha256,
        "semantic-v6-sample-report": preset.sample_report_sha256,
        "semantic-v6-selected-base": selected_base.selection_sha256,
    }


def _require_bindings(
    prepared: V6PreparedVampCurriculum,
    selected_base: V6SelectedBase,
    preset: V6VampExperimentPreset,
) -> None:
    if type(preset) is not V6VampExperimentPreset:
        raise TypeError("semantic-v6 VAMP requires its strict preset")
    if (
        selected_base.partition_sha256 != preset.partition_sha256
        or selected_base.catalog_sha256 != preset.catalog_sha256
        or selected_base.sample_report_sha256 != preset.sample_report_sha256
        or tuple(str(task.task_id) for task in prepared.curriculum.tasks)
        != preset.task_order
    ):
        raise ValueError("semantic-v6 VAMP base or curriculum binding changed")


def _completed_stage_directories(root: Path, maximum: int) -> tuple[Path, ...]:
    candidates = tuple(sorted(root.glob("stage-[0-9][0-9]")))
    expected = tuple(root / f"stage-{index:02d}" for index in range(1, len(candidates) + 1))
    if candidates != expected or len(candidates) > maximum:
        raise ValueError("semantic-v6 VAMP stage checkpoints are not a contiguous prefix")
    return candidates


def _method_progress(
    progress: V6VampTrainingProgress | None,
    method: str,
    world: str,
) -> Callable[[int, float, int], None] | None:
    if progress is None:
        return None
    return lambda step, loss, total: progress(method, world, step, loss, total)


def _publish_adaptations(
    adaptation: LanguageAdaptationArtifact,
    prepared: V6PreparedVampCurriculum,
    selected_base: V6SelectedBase,
    publication_root: Path,
    preset: V6VampExperimentPreset,
) -> V6VampAdaptationPublication:
    peak = allocator_peak_bytes()
    if peak > preset.allocator_peak_limit_bytes:
        raise MemoryError(
            f"semantic-v6 VAMP peak {peak:,} exceeds "
            f"{preset.allocator_peak_limit_bytes:,} bytes"
        )
    content = {
        "adaptation_tensor_checksum": adaptation.tensor_checksum,
        "allocator_peak_bytes": peak,
        "base_checkpoint": {
            "manifest_sha256": adaptation.base_checkpoint.manifest_sha256,
            "parameter_checksum": adaptation.base_checkpoint.parameter_checksum,
        },
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "curriculum_sha256": prepared.curriculum_sha256,
        "format": V6_VAMP_ADAPTATION_FORMAT,
        "partition_sha256": preset.partition_sha256,
        "sealed_test_opened": False,
        "selected_base_sha256": selected_base.selection_sha256,
        "task_order": list(preset.task_order),
    }
    run_sha256 = record_sha256(content)
    publication_root.mkdir(parents=True, exist_ok=True)
    target = publication_root / run_sha256
    if target.exists():
        return load_v6_vamp_adaptation_publication(target)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-semantic-v6-vamp-", dir=work_root))
    save_language_adaptation_artifact(staging / "adaptations", adaptation)
    _write_json(staging / "manifest.json", {**content, "run_sha256": run_sha256})
    _write_text(
        staging / "training-report.md",
        (
            "# TinyWorlds-P Semantic-v6 VAMP Adapters\n\n"
            f"Run SHA-256: `{run_sha256}`\n\n"
            "All five adaptation worlds are complete. The sealed test remains unopened.\n"
        ),
    )
    _write_tree(staging, run_sha256)
    os.rename(staging, target)
    _fsync_directory(publication_root)
    return load_v6_vamp_adaptation_publication(target)


def _write_tree(root: Path, run_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": V6_VAMP_ADAPTATION_TREE_FORMAT,
            "run_sha256": run_sha256,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic-v6 VAMP JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical semantic-v6 VAMP JSON: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"semantic-v6 VAMP field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 VAMP field {field!r} must be nonnegative")
    return value


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = [
    "V6VampAdaptationPublication",
    "V6VampTrainingProgress",
    "V6_VAMP_ADAPTATION_FORMAT",
    "V6_VAMP_ADAPTATION_TREE_FORMAT",
    "load_v6_vamp_adaptation_publication",
    "train_or_resume_v6_vamp_adaptations",
]
