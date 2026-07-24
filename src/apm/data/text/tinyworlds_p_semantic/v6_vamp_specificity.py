"""Forced-adapter world/control specificity audit for semantic-v6 VAMP."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path

import jax

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_baseline_training import pack_root_adapter
from apm.data.text.tinyworlds_p_semantic.contracts import (
    WORLD_LABELS,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evaluation import (
    SplitGroupEvaluation,
    load_group_losses,
)
from apm.data.text.tinyworlds_p_semantic.statistics import GroupLoss
from apm.data.text.tinyworlds_p_semantic.v6_evaluation import (
    evaluate_v6_forced_lora_split,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6SemanticPartitionArtifact,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_statistics import (
    AdapterSpecificity,
    AdapterSpecificityPair,
    paired_adapter_specificity,
)
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    pack_lora_memory,
)
from apm.lm.parameters import GptNeoParams


V6SpecificityProgress = Callable[[str, str, str, int, int], None]


@dataclass(frozen=True, slots=True)
class ForcedAdapterPath:
    """One named immutable adapter memory and hard execution path."""

    method: str
    memory: PackedLoraMemory
    coefficients: jax.Array


@dataclass(frozen=True, slots=True)
class V6SpecificityAudit:
    """All row/column specificity intervals and their authenticated ledgers."""

    results: tuple[AdapterSpecificity, ...]
    ledger_sha256: tuple[tuple[str, str], ...]
    identity_sha256: str
    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.results or not self.directory.is_dir():
            raise ValueError("semantic-v6 specificity audit is incomplete")


def evaluate_v6_adapter_specificity(
    artifact: V6SemanticPartitionArtifact,
    adaptation: LanguageAdaptationArtifact,
    base_params: GptNeoParams,
    base_sealed_directory: str | Path,
    output_directory: str | Path,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
    *,
    progress: V6SpecificityProgress | None = None,
) -> V6SpecificityAudit:
    """Compare each forced world adapter with both persisted control arms."""
    _require_inputs(artifact, adaptation, preset)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    base_root = Path(base_sealed_directory)
    base_losses = {
        (role, world): _losses_by_group(
            base_root / f"{role}-{world}-test.groups.jsonl"
        )
        for world in WORLD_LABELS
        for role in ("world", "control")
    }
    paths_by_world = _forced_paths(adaptation, preset)
    evaluations = {
        (path.method, world, role): _evaluate_or_load(
            base_params,
            artifact,
            f"{role}/{world}/test",
            output / f"{path.method}-{world}-{role}-test.groups.jsonl",
            path,
            adaptation.lora_config,
            progress,
            world,
            role,
        )
        for world in WORLD_LABELS
        for path in paths_by_world[world]
        for role in ("world", "control")
    }
    identity = record_sha256(
        {
            "adaptation_tensor_checksum": adaptation.tensor_checksum,
            "base_ledgers": {
                f"{role}/{world}": _file_sha256(
                    base_root / f"{role}-{world}-test.groups.jsonl"
                )
                for world in WORLD_LABELS
                for role in ("world", "control")
            },
            "config_sha256": preset.config_sha256,
            "forced_ledgers": {
                f"{method}/{world}/{role}": evaluation.ledger_sha256
                for (method, world, role), evaluation in evaluations.items()
            },
            "partition_sha256": artifact.partition_sha256,
        }
    )
    results = tuple(
        paired_adapter_specificity(
            _specificity_pairs(
                artifact,
                world,
                arm,
                base_losses[("world", world)],
                base_losses[("control", world)],
                _losses_by_group(
                    evaluations[(method, world, "world")].ledger_path
                ),
                _losses_by_group(
                    evaluations[(method, world, "control")].ledger_path
                ),
            ),
            method,
            identity,
            replicates=preset.specificity_replicates,
        )
        for world in WORLD_LABELS
        for method in ("sequential_single_lora", "independent_root_lora", "vamp_oracle")
        for arm in ("row", "column")
    )
    _write_json(
        output / "specificity.json",
        {
            "identity_sha256": identity,
            "results": [_specificity_record(result) for result in results],
        },
    )
    return V6SpecificityAudit(
        results=results,
        ledger_sha256=tuple(
            sorted(
                (
                    f"{method}/{world}/{role}",
                    evaluation.ledger_sha256,
                )
                for (method, world, role), evaluation in evaluations.items()
            )
        ),
        identity_sha256=identity,
        directory=output.resolve(),
    )


def _forced_paths(
    adaptation: LanguageAdaptationArtifact,
    preset: V6VampExperimentPreset,
) -> dict[str, tuple[ForcedAdapterPath, ...]]:
    _, sequential_memory = pack_root_adapter(
        adaptation.sequential_stages[-1].adapter,
        adaptation.model_config,
        adaptation.lora_config,
    )
    independent = {
        str(record.task_id): record.adapter for record in adaptation.independent_adapters
    }
    vamp_memory = pack_lora_memory(
        adaptation.vamp_graph,
        adaptation.model_config,
        adaptation.lora_config,
        adaptation.max_nodes,
        adaptation.max_edges,
    )
    return {
        world: (
            ForcedAdapterPath(
                "sequential_single_lora",
                sequential_memory,
                edge_coefficients_for_node(sequential_memory, 1),
            ),
            _root_path(
                "independent_root_lora",
                independent[world],
                adaptation,
            ),
            ForcedAdapterPath(
                "vamp_oracle",
                vamp_memory,
                edge_coefficients_for_node(
                    vamp_memory,
                    next(
                        index
                        for index, node in enumerate(adaptation.vamp_graph.nodes)
                        if str(node.node_id) == world
                    ),
                ),
            ),
        )
        for world in preset.task_order
    }


def _root_path(
    method: str,
    adapter: LoraEdge,
    adaptation: LanguageAdaptationArtifact,
) -> ForcedAdapterPath:
    _, memory = pack_root_adapter(
        adapter,
        adaptation.model_config,
        adaptation.lora_config,
    )
    return ForcedAdapterPath(method, memory, edge_coefficients_for_node(memory, 1))


def _evaluate_or_load(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    split: str,
    path: Path,
    adapter: ForcedAdapterPath,
    lora_config: LoraConfig,
    progress: V6SpecificityProgress | None,
    world: str,
    role: str,
) -> SplitGroupEvaluation:
    if path.exists():
        return _summarize_ledger(split, path)
    callback = (
        None
        if progress is None
        else lambda _split, completed, total: progress(
            adapter.method,
            world,
            role,
            completed,
            total,
        )
    )
    return evaluate_v6_forced_lora_split(
        params,
        artifact,
        split,
        path,
        adapter.memory,
        lora_config,
        adapter.coefficients,
        progress=callback,
    )


def _specificity_pairs(
    artifact: V6SemanticPartitionArtifact,
    world: str,
    arm: str,
    base_world: Mapping[str, GroupLoss],
    base_control: Mapping[str, GroupLoss],
    adapted_world: Mapping[str, GroupLoss],
    adapted_control: Mapping[str, GroupLoss],
) -> tuple[AdapterSpecificityPair, ...]:
    return tuple(
        AdapterSpecificityPair(
            world=world,
            arm=arm,
            base_world=base_world[pair.world_group_sha256],
            adapted_world=adapted_world[pair.world_group_sha256],
            base_control=base_control[pair.control_group_sha256],
            adapted_control=adapted_control[pair.control_group_sha256],
        )
        for pair in artifact.pairings
        if pair.world == world and pair.split == "test" and pair.arm == arm
    )


def _losses_by_group(path: Path) -> dict[str, GroupLoss]:
    return {loss.normalized_story_sha256: loss for loss in load_group_losses(path)}


def _summarize_ledger(split: str, path: Path) -> SplitGroupEvaluation:
    losses = load_group_losses(path)
    return SplitGroupEvaluation(
        split=split,
        active_tokens=sum(item.active_tokens for item in losses),
        loss_sum=sum(item.loss_sum for item in losses),
        ledger_path=path,
        ledger_sha256=_file_sha256(path),
        group_count=len(losses),
    )


def _specificity_record(result: AdapterSpecificity) -> dict[str, object]:
    return {
        name: getattr(result, name) for name in result.__dataclass_fields__
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inputs(
    artifact: V6SemanticPartitionArtifact,
    adaptation: LanguageAdaptationArtifact,
    preset: V6VampExperimentPreset,
) -> None:
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 specificity requires its strict partition")
    if not isinstance(adaptation, LanguageAdaptationArtifact):
        raise TypeError("semantic-v6 specificity requires trained adapters")
    if type(preset) is not V6VampExperimentPreset:
        raise TypeError("semantic-v6 specificity requires its strict preset")
    if (
        artifact.partition_sha256 != preset.partition_sha256
        or tuple(str(task) for task in adaptation.task_order) != preset.task_order
    ):
        raise ValueError("semantic-v6 specificity source identity changed")


__all__ = [
    "ForcedAdapterPath",
    "V6SpecificityAudit",
    "V6SpecificityProgress",
    "evaluate_v6_adapter_specificity",
]
