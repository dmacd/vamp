from dataclasses import replace
from pathlib import Path

import torch

from apm.continual.top_two_adapter import top_two_base_state
from apm.data.mnist.permutations import identity_permutation
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_router_config import load_config
from apm.experiments.vamp_logt_router_data import (
    FrozenClassifierDependency,
    PermutedMnistBenchmark,
    StepAllocation,
)
from apm.experiments.vamp_logt_router_reporting import (
    _primary_test_rows,
    _range_hypothesis,
    write_results,
)
from apm.experiments.vamp_logt_router_workflow import run_phase_seed


def _dependency() -> FrozenClassifierDependency:
    torch.manual_seed(4)
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
    generator = torch.Generator().manual_seed(7)
    train_images = torch.rand((12, 1, 28, 28), generator=generator)
    test_images = torch.rand((5, 1, 28, 28), generator=generator)
    permutation = torch.from_numpy(identity_permutation())
    return PermutedMnistBenchmark(
        train_images,
        torch.arange(12, dtype=torch.int64) % 10,
        test_images,
        torch.arange(5, dtype=torch.int64) % 10,
        (permutation,) * 8,
        (
            StepAllocation(1, 0, (0, 1), (2, 3), (4,)),
            StepAllocation(2, 1, (0, 1), (2, 3), (4,)),
        ),
        ((0, 1),) * 8,
    )


def test_two_step_carry_metrics_and_resume_are_exact(tmp_path: Path, monkeypatch) -> None:
    source = load_config("configs/vamp_logt_router_mnist/primary.yaml")
    config = replace(
        source,
        artifact_root=tmp_path / "artifacts",
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
        "apm.experiments.vamp_logt_router_workflow.build_benchmark",
        lambda _config, _seed: benchmark,
    )
    run_root = tmp_path / "run"
    dependency = _dependency()
    first = run_phase_seed(
        config,
        "smoke",
        config.smoke,
        0,
        dependency,
        run_root,
        torch.device("cpu"),
    )
    metrics_before = (first.directory / "metrics.jsonl").read_bytes()
    second = run_phase_seed(
        config,
        "smoke",
        config.smoke,
        0,
        dependency,
        run_root,
        torch.device("cpu"),
    )
    assert first.summary == second.summary
    assert (first.directory / "metrics.jsonl").read_bytes() == metrics_before
    assert first.summary["final_macro_step"] == 2
    assert first.summary["active_frontier"][0]["level"] == 1
    assert first.summary["acceptance"]["exact_historical_budget"]
    assert first.summary["acceptance"]["nonnegative_routing_regret"]
    nodes = tuple((first.directory / "nodes").iterdir())
    assert len(nodes) == 1
    assert (first.directory / "state" / "checkpoint.pt").is_file()
    assert (first.directory / "RESULTS.md").is_file()


def test_primary_joint_reference_and_aggregate_report(tmp_path: Path, monkeypatch) -> None:
    source = load_config("configs/vamp_logt_router_mnist/primary.yaml")
    config = replace(
        source,
        artifact_root=tmp_path / "artifacts",
        benchmark=replace(
            source.benchmark,
            macro_steps=2,
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
        primary=replace(
            source.primary,
            seeds=(0,),
            macro_steps=2,
            historical_budget=2,
            router_epochs_per_step=1,
        ),
        evaluation=replace(
            source.evaluation,
            test_subset_per_domain=2,
            inference_batch_size=8,
            full_checkpoints=(1, 2),
        ),
        runtime=replace(source.runtime, device="cpu", progress=False),
    )
    benchmark = _benchmark()
    monkeypatch.setattr(
        "apm.experiments.vamp_logt_router_workflow.build_benchmark",
        lambda _config, _seed: benchmark,
    )
    run_root = tmp_path / "run"
    result = run_phase_seed(
        config,
        "primary",
        config.primary,
        0,
        _dependency(),
        run_root,
        torch.device("cpu"),
    )
    aggregate = write_results(run_root, config, (result,))

    assert aggregate["status"] == "complete"
    assert result.summary["work"]["joint_reference_example_updates"] == 6
    assert (result.directory / "joint_reference" / "step-1.pt").is_file()
    assert (result.directory / "joint_reference" / "step-2.pt").is_file()
    assert (run_root / "summary.json").is_file()
    assert (run_root / "RESULTS.html").is_file()
    assert "<details open>" in (run_root / "RESULTS.html").read_text()


def test_aggregate_views_use_unmixed_test_and_range_archive_rows() -> None:
    common = {
        "condition": "example_hard",
        "group": "micro",
        "macro_step": 15,
        "row_type": "evaluation",
        "run_seed": 0,
    }
    test_rows = (
        {**common, "evaluation_scope": "test_subset", "mean_regret": 1.0},
        {**common, "evaluation_scope": "full_test", "mean_regret": 2.0},
        {**common, "macro_step": 14, "evaluation_scope": "test_subset", "mean_regret": 3.0},
    )
    selected = _primary_test_rows(test_rows)
    assert [(row["macro_step"], row["evaluation_scope"]) for row in selected] == [
        (15, "full_test"),
        (14, "test_subset"),
    ]

    archive_rows = (
        {
            **common,
            "evaluation_scope": "evaluation_archive",
            "worst_range_mean_regret": 0.4,
        },
        {
            **common,
            "condition": "range_hard",
            "evaluation_scope": "evaluation_archive",
            "worst_range_mean_regret": 0.3,
        },
    )
    assert _range_hypothesis(archive_rows) is True
