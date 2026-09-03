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
from apm.continual.vision.imagenetr.integrator_reporting import write_integrator_report
from apm.continual.vision.imagenetr.proxy_memory import TensorCache


def test_checkpoint_and_immutable_safetensors_round_trip(tmp_path: Path) -> None:
    config = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_v2.yaml"
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
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_v2.yaml"
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


def test_clean_history_record_round_trips_and_report_exposes_failed_gate(
    tmp_path: Path,
) -> None:
    persistent = {
        str(capacity): [{"accuracy": accuracy, "stage": 16}]
        for capacity, accuracy in ((512, 72.0), (1024, 74.0), (2048, 76.0))
    }
    core: dict[str, object] = {
        "fresh": [{"mean_validation_accuracy": 77.25, "stage": 16}],
        "hierarchy_controls": [
            {
                "controls": {"raw_union": 70.0, "true_node_oracle": 77.5},
                "stage": 16,
            }
        ],
        "gate_open": False,
        "parent_training": "full_union",
        "persistent": persistent,
        "reason": "no bounded historical reservoir passed",
        "schema_version": "imagenetr50-integrator-clean-development-v2",
        "selected_historical_capacity": None,
        "selected_parent_training": "full_union",
        "selected_variant": "scores",
    }
    record = {**core, "content_hash": record_sha256(core)}
    target = tmp_path / "evaluations" / "clean_development.json"
    publish_immutable_json(target, record)
    assert load_canonical_json(target) == record
    (tmp_path / "state").mkdir()
    atomic_write(
        tmp_path / "state" / "workflow.json",
        canonical_json_bytes({"phase": "COMPLETE_HISTORY_SELECTION_FAILURE"}),
    )

    write_integrator_report(tmp_path)

    markdown = (tmp_path / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "H=2048 - fresh (pp)" in markdown
    assert "-1.250" in markdown
    assert "Gate open: False" in markdown
    assert (tmp_path / "reports" / "clean_history_selection.csv").is_file()
    assert (tmp_path / "reports" / "clean_history_selection.parquet").is_file()
