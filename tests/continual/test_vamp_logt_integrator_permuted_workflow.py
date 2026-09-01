from dataclasses import replace
from pathlib import Path

import pytest
import torch

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256
from apm.continual.top_two_adapter import top_two_base_state
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_integrator_permuted_config import load_config
from apm.experiments.vamp_logt_integrator_permuted_reporting import (
    ARCHIVE_PLOT_STYLES,
    CEILING_CONDITION,
    CONDITION_PLOT_STYLES,
    _load_ceiling_overlay_rows,
)
from apm.experiments.vamp_logt_integrator_permuted_workflow import run_phase_seed
from apm.experiments.vamp_logt_router_data import (
    FrozenClassifierDependency,
    PermutedMnistBenchmark,
    StepAllocation,
)


def _dependency() -> FrozenClassifierDependency:
    torch.manual_seed(31)
    model = AddressCNN()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenClassifierDependency(
        model,
        top_two_base_state(
            model.embedding.weight,
            model.embedding.bias,
            model.classifier.weight,
            model.classifier.bias,
        ),
        "0" * 64,
        (),
    )


def _benchmark() -> PermutedMnistBenchmark:
    generator = torch.Generator().manual_seed(17)
    train_images = torch.rand((12, 1, 28, 28), generator=generator)
    test_images = torch.rand((5, 1, 28, 28), generator=generator)
    return PermutedMnistBenchmark(
        train_images,
        torch.arange(12, dtype=torch.int64) % 10,
        test_images,
        torch.arange(5, dtype=torch.int64) % 10,
        tuple(torch.roll(torch.arange(784), shifts=domain) for domain in range(8)),
        (
            StepAllocation(1, 0, (0, 1), (2, 3), (4,)),
            StepAllocation(2, 1, (0, 1), (2, 3), (4,)),
        ),
        ((0, 1),) * 8,
    )


def test_integrator_protocol_is_the_exact_permuted_mnist_task() -> None:
    config = load_config(
        "configs/vamp_logt_integrator_permuted_mnist/primary.yaml"
    )
    assert config.benchmark.permutation_seeds == (1001, 1002, 1003, 1004, 1005, 1006, 1007)
    assert config.benchmark.stream_seed == 20260827
    assert config.integrator.maximum_levels == 7
    assert config.integrator.hidden_widths == (1024, 512, 256)
    assert config.primary.historical_budget == 256
    assert config.primary.seeds == (0, 1, 2, 3, 4)


def test_integrator_plot_series_have_redundant_unique_identities() -> None:
    condition_styles = tuple(CONDITION_PLOT_STYLES.values())
    archive_styles = tuple(ARCHIVE_PLOT_STYLES.values())
    assert len({style.color for style in condition_styles}) == len(condition_styles)
    assert len(
        {
            (style.color, style.linestyle, style.marker)
            for style in condition_styles
        }
    ) == len(condition_styles)
    assert len({style.color for style in archive_styles}) == len(archive_styles)
    assert len(
        {(style.color, style.linestyle, style.marker) for style in archive_styles}
    ) == len(archive_styles)


def test_ceiling_overlay_authenticates_parent_and_covers_every_step(
    tmp_path: Path,
) -> None:
    source = load_config(
        "configs/vamp_logt_integrator_permuted_mnist/primary.yaml"
    )
    config = replace(
        source,
        benchmark=replace(source.benchmark, macro_steps=1),
        smoke=replace(source.smoke, seeds=(0,), macro_steps=1),
        primary=replace(source.primary, seeds=(0,), macro_steps=1),
        evaluation=replace(source.evaluation, full_checkpoints=(1,)),
    )
    parent_root = tmp_path / ("a" * 64)
    parent_root.mkdir()
    atomic_write(parent_root / "protocol.json", canonical_json_bytes({"parent": 1}))
    atomic_write(parent_root / "summary.json", canonical_json_bytes({"status": "complete"}))
    ledger_path = parent_root / "primary" / "seed-0" / "metrics.jsonl"
    atomic_write(ledger_path, b"sealed parent ledger\n")
    parent_protocol_sha256 = file_sha256(parent_root / "protocol.json")
    parent_summary_sha256 = file_sha256(parent_root / "summary.json")
    parent_ledger_sha256 = file_sha256(ledger_path)

    ceiling_root = tmp_path / ("b" * 64)
    atomic_write(
        ceiling_root / "protocol.json",
        canonical_json_bytes(
            {
                "config": {
                    "parent_integrator": {
                        "primary_metric_ledger_sha256": {
                            "0": parent_ledger_sha256
                        },
                        "protocol_sha256": parent_protocol_sha256,
                        "run_id": parent_root.name,
                        "summary_sha256": parent_summary_sha256,
                    }
                },
                "config_hash": ceiling_root.name,
                "parent_integrator_metric_ledger_sha256": {
                    "0": parent_ledger_sha256
                },
                "parent_integrator_protocol_sha256": parent_protocol_sha256,
                "parent_integrator_summary_sha256": parent_summary_sha256,
            }
        ),
    )
    atomic_write(
        ceiling_root / "summary.json",
        canonical_json_bytes(
            {
                "ceiling_certified": True,
                "completed_primary_seeds": 1,
                "status": "complete",
            }
        ),
    )
    for phase in ("smoke", "primary"):
        seed_root = ceiling_root / phase / "seed-0"
        atomic_write(
            seed_root / "summary.json",
            canonical_json_bytes(
                {
                    "acceptance": {"converged": True},
                    "final_macro_step": 1,
                    "phase": phase,
                    "run_seed": 0,
                }
            ),
        )
        atomic_write(
            seed_root / "metrics.jsonl",
            canonical_json_bytes(
                {
                    "accuracy": 0.8,
                    "condition": CEILING_CONDITION,
                    "evaluation_scope": "full_test",
                    "group": "micro",
                    "macro_step": 1,
                    "mean_cross_entropy": 0.4,
                    "row_type": "evaluation",
                    "run_seed": 0,
                }
            ),
        )

    overlay = _load_ceiling_overlay_rows(parent_root, ceiling_root, config)
    assert set(overlay) == {("smoke", 0), ("primary", 0)}
    assert overlay[("primary", 0)][0]["accuracy"] == 0.8

    atomic_write(parent_root / "summary.json", canonical_json_bytes({"changed": True}))
    with pytest.raises(ValueError, match="does not authenticate"):
        _load_ceiling_overlay_rows(parent_root, ceiling_root, config)


def test_permuted_two_step_integrator_report_and_resume_are_exact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = load_config(
        "configs/vamp_logt_integrator_permuted_mnist/primary.yaml"
    )
    config = replace(
        source,
        artifact_root=tmp_path / "artifacts",
        benchmark=replace(
            source.benchmark,
            model_batch_size=2,
            integrator_batch_size=2,
            evaluation_batch_size=1,
        ),
        adapter=replace(source.adapter, epochs=1, batch_size=4),
        integrator=replace(
            source.integrator,
            dropout=0.0,
            minibatch_size=4,
        ),
        smoke=replace(
            source.smoke,
            macro_steps=2,
            historical_budget=2,
            integrator_epochs_per_step=1,
        ),
        evaluation=replace(
            source.evaluation,
            test_subset_per_domain=2,
            inference_batch_size=8,
        ),
        runtime=replace(source.runtime, device="cpu", progress=False),
    )
    benchmark = _benchmark()
    monkeypatch.setattr(
        "apm.experiments.vamp_logt_integrator_permuted_workflow.build_benchmark",
        lambda _config, _seed: benchmark,
    )
    run_root = tmp_path / "run"
    first = run_phase_seed(
        config,
        "smoke",
        config.smoke,
        0,
        _dependency(),
        run_root,
        torch.device("cpu"),
    )
    metrics_before = (first.directory / "metrics.jsonl").read_bytes()
    second = run_phase_seed(
        config,
        "smoke",
        config.smoke,
        0,
        _dependency(),
        run_root,
        torch.device("cpu"),
    )
    assert first.summary == second.summary
    assert (first.directory / "metrics.jsonl").read_bytes() == metrics_before
    assert first.summary["final_macro_step"] == 2
    assert first.summary["acceptance"]["exact_historical_budget"]
    assert first.summary["acceptance"]["fixed_budget_training_work"]
    assert first.summary["acceptance"]["one_node_initial_parity"]
    report = (first.directory / "RESULTS.md").read_text(encoding="utf-8")
    assert "prediction integrator" in report
    assert "Labels supervise only the final ten-class integrator output" in report
