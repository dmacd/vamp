from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from apm.continual.top_two_adapter import top_two_base_state
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_router_data import (
    FrozenClassifierDependency,
    StepAllocation,
)
from apm.experiments.vamp_logt_router_rotated_config import load_config
from apm.experiments.vamp_logt_router_rotated_data import (
    RotatedMnistBenchmark,
    balanced_indices,
    build_stream_allocations,
    source_indices_sha256,
)
from apm.experiments.vamp_logt_router_rotated_workflow import run_phase_seed


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


def _benchmark() -> RotatedMnistBenchmark:
    generator = torch.Generator().manual_seed(17)
    train_images = torch.rand((12, 1, 28, 28), generator=generator)
    test_images = torch.rand((5, 1, 28, 28), generator=generator)
    return RotatedMnistBenchmark(
        train_images,
        torch.arange(12, dtype=torch.int64) % 10,
        test_images,
        torch.arange(5, dtype=torch.int64) % 10,
        (0.0, 18.0, 36.0, 54.0, 72.0),
        (0, 2, 4, 6, 8),
        (
            StepAllocation(1, 0, (0, 1), (2, 3), (4,)),
            StepAllocation(2, 1, (0, 1), (2, 3), (4,)),
        ),
        ((0, 1),) * 5,
    )


def test_rotated_protocol_is_the_exact_vamp_af_task() -> None:
    config = load_config("configs/vamp_logt_router_rotated_mnist/primary.yaml")
    assert config.task.rotations_deg == (0.0, 18.0, 36.0, 54.0, 72.0)
    assert config.task.label_shifts == (0, 2, 4, 6, 8)
    assert config.task.primary_context_steps == (13, 13, 13, 13, 12)
    assert config.task.smoke_context_steps == (1, 1, 1, 1, 1)
    assert config.primary.seeds == (0, 1, 2, 3, 4)


def test_balanced_source_selection_and_hash_are_deterministic() -> None:
    labels = np.tile(np.arange(10, dtype=np.int64), 20)
    first = balanced_indices(labels, 100, 0)
    second = balanced_indices(labels, 100, 0)
    assert first == second
    selected_labels = labels[np.asarray(first)]
    assert np.bincount(selected_labels, minlength=10).tolist() == [10] * 10
    assert source_indices_sha256(np.asarray(first, dtype=np.int64)) == source_indices_sha256(
        np.asarray(second, dtype=np.int64)
    )


def test_blocked_allocations_are_disjoint_fixed_order_and_seed_varying() -> None:
    source = load_config("configs/vamp_logt_router_rotated_mnist/primary.yaml")
    benchmark = replace(
        source.benchmark,
        model_batch_size=2,
        router_batch_size=2,
        evaluation_batch_size=1,
    )
    pools = tuple(np.arange(30, dtype=np.int64) + domain * 100 for domain in range(5))
    context_steps = (2, 1, 0, 0, 1)
    left = build_stream_allocations(benchmark, context_steps, pools, 0)
    same = build_stream_allocations(benchmark, context_steps, pools, 0)
    other = build_stream_allocations(benchmark, context_steps, pools, 1)
    assert left == same
    assert left != other
    assert [row.domain_id for row in left] == [0, 0, 1, 4]
    for row in left:
        combined = row.model_indices + row.router_indices + row.evaluation_indices
        assert len(combined) == len(set(combined)) == 5


def test_materialized_context_applies_rotation_and_label_shift() -> None:
    benchmark = _benchmark()
    identity = benchmark.step(1).model
    rotated = benchmark.step(2).model
    assert torch.allclose(identity.images, benchmark.train_images[:2], atol=3.0e-6, rtol=0.0)
    assert identity.labels.tolist() == [0, 1]
    assert rotated.labels.tolist() == [2, 3]
    assert not torch.equal(rotated.images, benchmark.train_images[:2])
    assert identity.domain_ids.tolist() == [0, 0]
    assert rotated.domain_ids.tolist() == [1, 1]


def test_rotated_two_step_carry_report_and_resume_are_exact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = load_config("configs/vamp_logt_router_rotated_mnist/primary.yaml")
    config = replace(
        source,
        artifact_root=tmp_path / "artifacts",
        task=replace(source.task, smoke_context_steps=(1, 1, 0, 0, 0)),
        benchmark=replace(
            source.benchmark,
            model_batch_size=2,
            router_batch_size=2,
            evaluation_batch_size=1,
        ),
        adapter=replace(source.adapter, epochs=1, batch_size=4),
        router=replace(
            source.router,
            hidden_widths=(16, 8, 4),
            dropout=0.0,
            minibatch_size=4,
        ),
        smoke=replace(
            source.smoke,
            macro_steps=2,
            historical_budget=2,
            router_epochs_per_step=1,
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
        "apm.experiments.vamp_logt_router_rotated_workflow.build_benchmark",
        lambda _config, _phase, _seed: benchmark,
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
    report = (first.directory / "RESULTS.md").read_text(encoding="utf-8")
    assert "blocked VAMP-AF task" in report
    assert "Permuted-MNIST" not in report
