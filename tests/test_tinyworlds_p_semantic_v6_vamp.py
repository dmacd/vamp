from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apm.data.text.tinyworlds_p_semantic.statistics import GroupLoss
from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_statistics import (
    AdapterSpecificityPair,
    paired_adapter_specificity,
)
from apm.data.text.tinyworlds_p_semantic.v6_milestone import V6SelectedBase
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_PARENT_CATALOG_SHA256,
    V6SemanticPartitionArtifact,
)
from apm.data.text.tinyworlds_p_semantic.v6_training import (
    V6StreamingTrainingConfig,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_publication import (
    begin_v6_vamp_sealed_transaction,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_training import (
    V6VampAdaptationPublication,
)
from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.lm.checkpoint import BaseCheckpointRef


def _group(label: str, nll: float, tokens: int = 100) -> GroupLoss:
    return GroupLoss(sha256(label.encode()).hexdigest(), nll * tokens, tokens)


def _specificity_pairs() -> tuple[AdapterSpecificityPair, ...]:
    return tuple(
        AdapterSpecificityPair(
            world="A",
            arm="row",
            base_world=_group(f"base-world-{index}", 1.20 + index / 10_000),
            adapted_world=_group(f"base-world-{index}", 0.90 + index / 10_000),
            base_control=_group(f"base-control-{index}", 1.10 + index / 10_000),
            adapted_control=_group(f"base-control-{index}", 1.05 + index / 10_000),
        )
        for index in range(24)
    )


def test_specificity_bootstrap_is_deterministic_and_uses_paired_improvements() -> None:
    pairs = _specificity_pairs()
    first = paired_adapter_specificity(
        pairs,
        "vamp_oracle",
        "a" * 64,
        replicates=2_000,
    )
    replay = paired_adapter_specificity(
        tuple(reversed(pairs)),
        "vamp_oracle",
        "a" * 64,
        replicates=2_000,
    )

    assert first.world_improvement == pytest.approx(0.30)
    assert first.control_improvement == pytest.approx(0.05)
    assert first.specificity == pytest.approx(0.25)
    assert first.bootstrap_lower > 0.24
    assert replay == first


def test_specificity_rejects_misaligned_base_and_adapter_ledgers() -> None:
    pair = _specificity_pairs()[0]

    with pytest.raises(ValueError, match="misaligned"):
        replace(pair, adapted_world=_group("different-world", 0.9))


def test_v6_vamp_preset_freezes_router_and_optimizer_choices() -> None:
    preset = V6_VAMP_EXPERIMENT_PRESET

    assert preset.hopfield_config.beta == 10.0
    assert preset.hopfield_config.top_k == 4
    assert preset.ebt_config.steps == 20
    assert preset.train_config.steps == 2_000
    assert preset.config_sha256 == (
        "ca16318486600745e8a49903f495819741082f120fa7b95b3f9277efa83ada73"
    )
    with pytest.raises(ValueError, match="frozen"):
        V6VampExperimentPreset(hopfield_beta=9.0)


def test_v6_sealed_evaluation_returns_the_strict_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apm.data.text.tinyworlds_p_semantic.v6_evaluation as evaluation

    artifact = object.__new__(V6SemanticPartitionArtifact)
    object.__setattr__(artifact, "partition_sha256", "a" * 64)
    held_in = object()
    validation = object()
    core_result = SimpleNamespace(
        selected_epoch=2,
        held_in=held_in,
        validation=validation,
        directory=tmp_path,
    )
    (tmp_path / "sealed-test.json").write_bytes(
        canonical_json_bytes({"evaluation_identity_sha256": "b" * 64})
    )
    monkeypatch.setattr(
        evaluation,
        "_evaluate_sealed_test_once_core",
        lambda *_args, **_kwargs: core_result,
    )

    result = evaluation.evaluate_v6_sealed_test_once(
        object(),
        artifact,
        2,
        tmp_path / "unused",
    )

    assert type(result) is evaluation.V6SemanticSealedTest
    assert result.selected_epoch == 2
    assert result.partition_sha256 == artifact.partition_sha256
    assert result.evaluation_identity_sha256 == "b" * 64
    assert result.held_in is held_in
    assert result.validation is validation
    assert result.directory == tmp_path


def test_v6_resume_discovers_latest_checkpoint_and_trims_trace_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apm.data.text.tinyworlds_p_semantic.v6_training as training

    artifact = object.__new__(V6SemanticPartitionArtifact)
    object.__setattr__(artifact, "partition_sha256", "a" * 64)
    config = V6StreamingTrainingConfig.from_preset()
    monkeypatch.setattr(training, "count_v6_partition_microbatches", lambda *_: 8)
    training_sha256, _ = training._training_identity(artifact, config)
    states = tmp_path / "states"
    states.mkdir()

    def write_checkpoint(update: int) -> Path:
        directory = states / f"update-{update:09d}"
        directory.mkdir()
        (directory / "resume.json").write_bytes(
            canonical_json_bytes(
                {
                    "cursor": {
                        "block": 0,
                        "epoch": 0,
                        "microbatch": update,
                        "optimizer_update": update,
                        "schedule_position": update,
                    },
                    "format": training.V6_RESUME_FORMAT,
                    "state_sha256": "b" * 64,
                    "training_sha256": training_sha256,
                    "version": 1,
                }
            )
        )
        return directory

    write_checkpoint(2)
    latest = write_checkpoint(4)
    retained = b"".join(
        canonical_json_bytes({"optimizer_update": update})
        for update in range(1, 5)
    )
    (tmp_path / "progress.jsonl").write_bytes(retained + b"interrupted tail")
    monkeypatch.setattr(training, "init_v6_streaming_train_state", lambda *_: object())
    monkeypatch.setattr(
        training,
        "load_v6_streaming_checkpoint",
        lambda directory, identity, _template: (
            object(),
            training._resume_cursor(Path(directory), identity),
        ),
    )
    monkeypatch.setattr(training, "lm_train_state_checksum", lambda _state: "c" * 64)

    result = training.load_latest_v6_streaming_result(
        artifact,
        tmp_path,
        config,
    )

    assert result is not None
    assert result.cursor.optimizer_update == 4
    assert result.checkpoints[0].directory == latest.resolve()
    assert result.trace_path.read_bytes() == retained


def test_sealed_transaction_is_durable_idempotent_and_identity_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apm.data.text.tinyworlds_p_semantic.v6_vamp_publication as publication

    preset = V6_VAMP_EXPERIMENT_PRESET
    partition = object.__new__(V6SemanticPartitionArtifact)
    object.__setattr__(partition, "partition_sha256", preset.partition_sha256)
    selected_directory = tmp_path / "selected"
    selected_directory.mkdir()
    selected = V6SelectedBase(
        directory=selected_directory,
        selection_sha256="1" * 64,
        training_sha256="2" * 64,
        partition_sha256=preset.partition_sha256,
        catalog_sha256=V6_PARENT_CATALOG_SHA256,
        sample_report_sha256=preset.sample_report_sha256,
        selected_epoch=2,
        checkpoint=BaseCheckpointRef(
            directory=selected_directory / "base",
            manifest_sha256="3" * 64,
            parameter_checksum="4" * 64,
        ),
    )
    adaptation_directory = tmp_path / "adaptations"
    adaptation_directory.mkdir()
    adaptations = V6VampAdaptationPublication(
        directory=adaptation_directory,
        run_sha256="5" * 64,
        partition_sha256=preset.partition_sha256,
        selected_base_sha256=selected.selection_sha256,
        curriculum_sha256="6" * 64,
        config_sha256=preset.config_sha256,
        allocator_peak_bytes=0,
        adaptation=object.__new__(LanguageAdaptationArtifact),
    )
    monkeypatch.setattr(publication, "load_v6_selected_base", lambda _path: selected)
    monkeypatch.setattr(
        publication,
        "load_v6_vamp_adaptation_publication",
        lambda _path: adaptations,
    )
    transaction = tmp_path / "transaction"

    first = begin_v6_vamp_sealed_transaction(
        partition,
        selected,
        adaptations,
        transaction,
        preset,
    )
    second = begin_v6_vamp_sealed_transaction(
        partition,
        selected,
        adaptations,
        transaction,
        preset,
    )

    assert first == second == transaction.resolve()
    marker = transaction / "sealed-transaction.json"
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "adaptation_run_sha256": adaptations.run_sha256,
        "config_sha256": preset.config_sha256,
        "format": "tinyworlds-p-semantic-v6-sealed-transaction",
        "partition_sha256": preset.partition_sha256,
        "selected_base_sha256": selected.selection_sha256,
    }
    marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        begin_v6_vamp_sealed_transaction(
            partition,
            selected,
            adaptations,
            transaction,
            preset,
        )
