from dataclasses import replace
from pathlib import Path

import torch

from apm.continual.logt_behavioral_integrator import FullReplayConvergenceConfig
from apm.continual.artifacts import load_canonical_json
from apm.continual.top_two_adapter import top_two_base_state
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_integrator_ceiling_rotated_config import load_config
from apm.experiments.vamp_logt_integrator_ceiling_rotated_reporting import write_results
from apm.experiments.vamp_logt_integrator_ceiling_rotated_workflow import run_phase_seed
from apm.experiments.vamp_logt_router_data import (
    FrozenClassifierDependency,
    StepAllocation,
)
from apm.experiments.vamp_logt_router_rotated_data import RotatedMnistBenchmark


def _dependency() -> FrozenClassifierDependency:
    torch.manual_seed(61)
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
    generator = torch.Generator().manual_seed(67)
    return RotatedMnistBenchmark(
        torch.rand((12, 1, 28, 28), generator=generator),
        torch.arange(12, dtype=torch.int64) % 10,
        torch.rand((5, 1, 28, 28), generator=generator),
        torch.arange(5, dtype=torch.int64) % 10,
        (0.0, 18.0, 36.0, 54.0, 72.0),
        (0, 2, 4, 6, 8),
        (
            StepAllocation(1, 0, (0, 1), (2, 3), (4,)),
            StepAllocation(2, 1, (0, 1), (2, 3), (4,)),
        ),
        ((0, 1),) * 5,
    )


def test_ceiling_protocol_wraps_the_completed_integrator_exactly() -> None:
    config = load_config(
        "configs/vamp_logt_integrator_ceiling_rotated_mnist/primary.yaml"
    )
    assert config.parent.config_hash == config.parent_integrator.run_id
    assert config.primary.restarts_per_step == 3
    assert config.convergence.maximum_epochs == 200
    assert config.convergence.minimum_learning_rate == 1.0e-5
    assert config.integrator.hidden_widths == (1024, 512, 256)


def test_two_step_ceiling_uses_disjoint_validation_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = load_config(
        "configs/vamp_logt_integrator_ceiling_rotated_mnist/primary.yaml"
    )
    parent = replace(
        source.parent,
        task=replace(source.task, smoke_context_steps=(1, 1, 0, 0, 0)),
        benchmark=replace(
            source.benchmark,
            model_batch_size=2,
            integrator_batch_size=2,
            evaluation_batch_size=1,
        ),
        adapter=replace(source.adapter, epochs=1, batch_size=4),
        integrator=replace(source.integrator, dropout=0.0, minibatch_size=4),
        smoke=replace(
            source.parent.smoke,
            macro_steps=2,
            historical_budget=2,
            integrator_epochs_per_step=1,
        ),
        evaluation=replace(
            source.evaluation,
            test_subset_per_domain=2,
            inference_batch_size=8,
        ),
        runtime=replace(source.parent.runtime, device="cpu", progress=False),
    )
    parent_coordinates = replace(
        source.parent_integrator,
        run_id=parent.config_hash,
    )
    config = replace(
        source,
        artifact_root=tmp_path / "artifacts",
        parent_integrator_run_root=tmp_path / parent.config_hash,
        parent_integrator=parent_coordinates,
        parent=parent,
        convergence=FullReplayConvergenceConfig(
            minimum_epochs=1,
            maximum_epochs=10,
            improvement_delta=100.0,
            learning_rate_patience=1,
            learning_rate_factor=0.5,
            minimum_learning_rate=2.5e-4,
            convergence_patience=2,
        ),
        smoke=replace(source.smoke, macro_steps=2),
        runtime=replace(source.runtime, device="cpu", progress=False),
    )
    benchmark = _benchmark()
    monkeypatch.setattr(
        "apm.experiments.vamp_logt_integrator_ceiling_rotated_workflow.build_benchmark",
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
    assert all(first.summary["acceptance"].values())
    assert not tuple((first.directory / "integrators").glob("step-*/*.pt"))
    assert len(tuple((first.directory / "integrators").glob("step-*/restart-*.json"))) == 2
    text = (first.directory / "RESULTS.md").read_text(encoding="utf-8")
    assert "test rows below never selected an epoch or restart" in text
    parent_summary = load_canonical_json(
        source.parent_integrator_run_root / "summary.json"
    )
    aggregate = write_results(run_root, config, parent_summary)
    assert aggregate["status"] == "partial"
    assert aggregate["ceiling_high_checkpoint_means"] is None
