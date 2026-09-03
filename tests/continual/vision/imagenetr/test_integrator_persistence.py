from __future__ import annotations

from pathlib import Path

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_artifacts import IntegratorStore
from apm.continual.vision.imagenetr.integrator_config import load_integrator_config
from apm.continual.vision.imagenetr.integrator_model import (
    IntegratorFitResult,
    create_integrator_state,
)
from apm.continual.vision.imagenetr.integrator_persistence import (
    load_integrator_fit,
    publish_integrator_fit,
    restore_integrator_checkpoint,
    save_integrator_checkpoint,
)
from apm.continual.vision.imagenetr.integrator_reporting import (
    _joint_iid_stage_rows,
    _write_tables,
    write_integrator_report,
)
from apm.continual.vision.imagenetr.proxy_memory import TensorCache


def test_checkpoint_and_immutable_safetensors_round_trip(tmp_path: Path) -> None:
    config = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_ungated_v3.yaml"
    )
    device = torch.device("cpu")
    state = create_integrator_state(
        "round-trip", 2, "scores", config.optimization, 7, device
    )
    before = {name: value.clone() for name, value in state.model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint.pt"
    identity = "a" * 64
    save_integrator_checkpoint(checkpoint, state, 2, "scores", identity, identity)
    with torch.no_grad():
        next(state.model.parameters()).add_(1.0)
    restore_integrator_checkpoint(checkpoint, state, 2, "scores", identity, identity)
    assert all(
        torch.equal(before[name], value)
        for name, value in state.model.state_dict().items()
    )

    store = IntegratorStore(tmp_path / "artifacts", "b" * 64)
    store.run.mkdir(parents=True)
    result = IntegratorFitResult(
        1, 1, 1, 1.0, 50.0, 1.1, 49.0, 4, 8, 4, 4, 0, 0.1, False
    )
    artifact = publish_integrator_fit(
        store, "unit", "c" * 64, state, result, {"purpose": "round-trip"}
    )
    restored = create_integrator_state(
        "round-trip", 2, "scores", config.optimization, 8, device
    )
    loaded = load_integrator_fit(artifact, restored)
    assert loaded == result
    assert all(
        torch.equal(state.model.state_dict()[name], value)
        for name, value in restored.model.state_dict().items()
    )


def test_checkpoint_rejects_a_different_frontier(tmp_path: Path) -> None:
    config = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_ungated_v3.yaml"
    )
    state = create_integrator_state(
        "boundary", 2, "scores", config.optimization, 7, torch.device("cpu")
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_integrator_checkpoint(checkpoint, state, 1, "scores", "a" * 64, "b" * 64)
    try:
        restore_integrator_checkpoint(
            checkpoint, state, 1, "scores", "c" * 64, "b" * 64
        )
    except ValueError as error:
        assert "boundary" in str(error)
    else:
        raise AssertionError("a checkpoint from another frontier was accepted")


def test_row_cache_reuses_overlapping_image_identities_in_request_order(
    tmp_path: Path,
) -> None:
    cache = TensorCache(tmp_path / "cache", "unit-row-cache-v1")
    computed: list[tuple[str, ...]] = []

    def compute(image_ids: tuple[str, ...]) -> dict[str, torch.Tensor]:
        computed.append(image_ids)
        return {
            "value": torch.tensor(
                [[float(int(image_id))] for image_id in image_ids], dtype=torch.float32
            )
        }

    first, first_hits, first_misses = cache.get_or_compute_rows(
        {"model": "fixed"}, ("1", "2"), compute
    )
    second, second_hits, second_misses = cache.get_or_compute_rows(
        {"model": "fixed"}, ("2", "3", "1"), compute
    )
    assert (first_hits, first_misses) == (0, 2)
    assert (second_hits, second_misses) == (2, 1)
    assert computed == [("1", "2"), ("3",)]
    assert first["value"].squeeze(1).tolist() == [1.0, 2.0]
    assert second["value"].squeeze(1).tolist() == [2.0, 3.0, 1.0]


def test_partial_report_is_markdown_and_self_contained_html(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    atomic_write(
        tmp_path / "state" / "workflow.json",
        canonical_json_bytes({"phase": "PREFLIGHT"}),
    )
    report = write_integrator_report(tmp_path)
    html = report.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "<table" not in html or "<thead>" in html
    assert (tmp_path / "reports" / "REPORT.md").is_file()
    assert (tmp_path / "reports" / "lineage.png").is_file()
    assert (tmp_path / "reports" / "resource_accounting.json").is_file()


def test_clean_history_record_round_trips_and_report_is_explicitly_ungated(
    tmp_path: Path,
) -> None:
    persistent = {"2048": [{"accuracy": 76.0, "stage": 16}]}
    core: dict[str, object] = {
        "fresh": [{"mean_validation_accuracy": 77.25, "stage": 16}],
        "hierarchy_controls": [
            {
                "controls": {"raw_union": 70.0, "true_node_oracle": 77.5},
                "stage": 16,
            }
        ],
        "parent_training": "full_union",
        "persistent": persistent,
        "role": "report_only_development",
        "schema_version": "imagenetr50-integrator-clean-development-v3",
        "selected_historical_capacity": 2048,
        "selected_parent_training": "full_union",
        "selected_variant": "scores",
    }
    record = {**core, "content_hash": record_sha256(core)}
    target = tmp_path / "evaluations" / "clean_development.json"
    publish_immutable_json(target, record)
    atomic_write(
        tmp_path / "evaluations" / "development_task50.json",
        canonical_json_bytes(
            {
                "fresh": [{"mean_validation_accuracy": 68.25, "stage": 50}],
                "hierarchy_controls": {
                    "controls": {"raw_union": 66.75, "true_node_oracle": 74.5},
                    "stage": 50,
                },
                "persistent": [
                    {
                        "accuracy": 63.125,
                        "historical_capacity": 2048,
                        "stage": 50,
                    }
                ],
                "selection": {"selected_historical_capacity": 2048},
            }
        ),
    )
    assert load_canonical_json(target) == record
    (tmp_path / "state").mkdir()
    atomic_write(
        tmp_path / "state" / "workflow.json",
        canonical_json_bytes({"phase": "CLEAN_DEVELOPMENT"}),
    )

    write_integrator_report(tmp_path)

    markdown = (tmp_path / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "H=2048 - fresh (pp)" in markdown
    assert "-1.250" in markdown
    assert "No accuracy or comparator value gates execution" in markdown
    assert "Gate open" not in markdown
    assert "clean-validation checkpoints (tasks 2/4/8/16/50)" in markdown
    projection = load_canonical_json(
        tmp_path / "reports" / "clean_history_selection.json"
    )
    assert [row["stage"] for row in projection["rows"]] == [16, 50]
    assert projection["rows"][-1]["fresh_mean_accuracy"] == 68.25
    assert projection["rows"][-1]["h2048_accuracy"] == 63.125
    assert (tmp_path / "reports" / "clean_history_selection.csv").is_file()
    assert (tmp_path / "reports" / "clean_history_selection.parquet").is_file()


def test_joint_iid_curve_is_authenticated_and_projected_by_stage(
    tmp_path: Path,
) -> None:
    sealed_run_hash = "a" * 64
    source_root = tmp_path / "sealed"
    source = source_root / "runs" / sealed_run_hash / "reports" / "stage_accuracy.csv"
    source.parent.mkdir(parents=True)
    accuracies = [95.0 - stage / 4 for stage in range(1, 51)]
    atomic_write(
        source,
        (
            "accuracy,condition,diagnostic,score_mode,stage\n"
            + "".join(
                f"{accuracy},joint_iid_lora_r16,False,raw,{stage}\n"
                for stage, accuracy in enumerate(accuracies, start=1)
            )
        ).encode("utf-8"),
    )
    atomic_write(
        tmp_path / "config_resolved.json",
        canonical_json_bytes(
            {
                "inference_artifact_root": str(source_root),
                "sealed_run_hash": sealed_run_hash,
            }
        ),
    )
    locked = {
        "local_references": {
            "joint_iid_incremental": sum(accuracies) / len(accuracies),
            "joint_iid_last": accuracies[-1],
        },
        "stage_metrics": [
            {"accuracy": 50.0, "controls": {}, "stage": stage}
            for stage in range(1, 51)
        ],
        "task_accuracy_matrix": [],
    }

    joint_rows = _joint_iid_stage_rows(tmp_path, locked)
    flattened, _task_rows = _write_tables(
        tmp_path / "reports", locked, joint_rows
    )

    assert len(joint_rows) == 50
    assert flattened[0]["offline_joint_iid"] == accuracies[0]
    assert flattened[-1]["offline_joint_iid"] == accuracies[-1]
