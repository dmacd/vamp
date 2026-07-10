"""Run the Stage 1 dense-delta addressed-parameter-memory benchmark."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import numpy as np

from apm.data import load_mnist
from apm.data.mnist import TaskDataset, make_permuted_mnist_stream
from apm.memory import (
    DenseMemoryGraph,
    EdgeVisualStats,
    NodeVisualStats,
    add_dense_delta_node,
    address_confusion_matrix,
    best_parent_by_observed_energy,
    edge_memory_stats,
    effective_params,
    evaluate_addressed_on_arrays,
    evaluate_addressed_on_task,
    evaluate_node_on_arrays,
    evaluate_node_on_task,
    graph_memory_bytes,
    init_dense_memory_graph,
    node_ids,
    node_memory_bytes,
    select_addresses,
    task_node_ids,
    write_memory_graph_svg,
)
from apm.memory.dense import ParamTree
from apm.models.backends import ModelBackend, make_model_backend
from apm.training.artifacts import append_jsonl, write_json, write_png_grid, write_svg_heatmap, write_svg_line_chart

RUN_DIR = Path("results") / "stage1_apm" / "permuted_mnist_dense_delta"
TASK_SEEDS = (0, 1, 2)
TRAIN_EXAMPLES_PER_TASK = 10_000
TEST_EXAMPLES_PER_TASK = 2_000
REPLAY_EXAMPLES_PER_TASK = 1_000
TASK_EPOCHS = 5
REPORT_CANVAS_COUNT = 32
PARENT_PROBE_COUNT = 1024
ACCURACY_KEY = "energy_classifier_accuracy"
SUMMARY_WORK_UNITS = 1
METADATA_WORK_UNITS = 6
CURVE_WORK_UNITS = 5
HEATMAP_WORK_UNITS = 3
REPORT_WORK_UNITS = 1


@dataclass(frozen=True)
class GraphSnapshot:
    """Report references for one committed memory graph snapshot."""

    stage: int
    filename: str
    node_count: int
    memory_bytes: int


@dataclass(frozen=True)
class BaselineRun:
    """Sequential baseline metrics plus the final parameter state."""

    rows: list[dict[str, object]]
    final_params: ParamTree


class BenchmarkProgress:
    """Top-level progress bar for full benchmark work."""

    def __init__(self, total_units: int, enabled: bool) -> None:
        self.total_units = max(1, total_units)
        self.enabled = enabled
        self._bar: Any | None = None
        self._last_phase: str | None = None

    def __enter__(self) -> "BenchmarkProgress":
        if not self.enabled:
            return self
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return self
        self._bar = tqdm(
            total=self.total_units,
            desc="benchmark",
            unit="work",
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._bar is not None:
            self._bar.close()

    def set_phase(self, phase: str) -> None:
        if self.enabled and phase != self._last_phase:
            message = f"[benchmark] {phase}"
            if self._bar is not None:
                self._bar.write(message)
            else:
                print(message, flush=True)
            self._last_phase = phase
        if self._bar is not None:
            self._bar.set_postfix_str(phase)

    def advance(self, units: int = 1, phase: str | None = None) -> None:
        if phase is not None:
            self.set_phase(phase)
        if self._bar is None:
            return
        self._bar.update(max(0, units))


def main() -> None:
    """Run the default Stage 1 benchmark and write report artifacts."""
    tasks = make_permuted_mnist_stream(
        load_mnist(allow_download=True),
        permutation_seeds=TASK_SEEDS,
        train_count=TRAIN_EXAMPLES_PER_TASK,
        test_count=TEST_EXAMPLES_PER_TASK,
    )
    run_stage1_benchmark(
        RUN_DIR,
        tasks,
        {
            "kind": "permuted_mnist",
            "task_seeds": TASK_SEEDS,
            "train_examples_per_task": TRAIN_EXAMPLES_PER_TASK,
            "test_examples_per_task": TEST_EXAMPLES_PER_TASK,
            "replay_examples_per_task": REPLAY_EXAMPLES_PER_TASK,
            "task_names": tuple(task.spec.name for task in tasks),
        },
        "Stage 1 PermutedMNIST Dense-Delta APM",
        task_epochs=TASK_EPOCHS,
        replay_examples_per_task=REPLAY_EXAMPLES_PER_TASK,
        parent_probe_count=PARENT_PROBE_COUNT,
        report_canvas_count=REPORT_CANVAS_COUNT,
        show_progress=True,
    )


def run_stage1_benchmark(
    run_dir: Path,
    tasks: tuple[TaskDataset, ...],
    stream_payload: dict[str, object],
    report_title: str,
    task_epochs: int,
    replay_examples_per_task: int,
    parent_probe_count: int,
    report_canvas_count: int,
    model_kind: str = "vae",
    show_progress: bool = False,
    include_baselines: bool = False,
) -> None:
    """Run Stage 1 baselines and dense-delta memory on a supplied task stream."""
    global RUN_DIR, REPLAY_EXAMPLES_PER_TASK, PARENT_PROBE_COUNT, REPORT_CANVAS_COUNT
    RUN_DIR = run_dir
    REPLAY_EXAMPLES_PER_TASK = replay_examples_per_task
    PARENT_PROBE_COUNT = parent_probe_count
    REPORT_CANVAS_COUNT = report_canvas_count
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backend = make_model_backend(model_kind, task_epochs=task_epochs, show_progress=False)
    config_payload = {
        **_config_payload(backend, tasks, stream_payload),
        "benchmark": {"include_baselines": include_baselines},
    }
    total_units = _benchmark_work_units(
        tasks,
        backend,
        task_epochs,
        replay_examples_per_task,
        parent_probe_count,
        report_canvas_count,
        include_baselines,
    )
    with BenchmarkProgress(total_units, show_progress) as progress:
        baseline_params: dict[str, ParamTree] = {}
        metrics_rows: list[dict[str, object]] = []
        if include_baselines:
            online_run = _run_sequential_baseline("online_sgd", tasks, backend, replay=False, progress=progress)
            replay_run = _run_sequential_baseline("replay_sgd", tasks, backend, replay=True, progress=progress)
            baseline_params = {
                "online_sgd": online_run.final_params,
                "replay_sgd": replay_run.final_params,
            }
            metrics_rows.extend(online_run.rows + replay_run.rows)
        graph, memory_rows, address_rows, energy_rows, graph_snapshots = _run_dense_memory(tasks, backend, progress)
        metrics_rows.extend(memory_rows)
        final_train_rows = _final_split_rows(
            tasks,
            graph,
            baseline_params,
            backend,
            split="train",
            address_rows=address_rows,
            energy_rows=energy_rows,
            progress=progress,
        )
        final_summary = _summary_payload(config_payload, graph, metrics_rows, address_rows, graph_snapshots, final_train_rows)
        progress.advance(SUMMARY_WORK_UNITS, "summarize final metrics")

        metrics_path = RUN_DIR / "metrics.jsonl"
        metrics_path.unlink(missing_ok=True)
        final_train_metrics_path = RUN_DIR / "final_train_metrics.jsonl"
        final_train_metrics_path.unlink(missing_ok=True)
        address_diagnostics_path = RUN_DIR / "address_diagnostics.jsonl"
        address_diagnostics_path.unlink(missing_ok=True)
        energy_diagnostics_path = RUN_DIR / "observed_energy_diagnostics.jsonl"
        energy_diagnostics_path.unlink(missing_ok=True)
        append_jsonl(metrics_path, metrics_rows)
        append_jsonl(final_train_metrics_path, final_train_rows)
        append_jsonl(address_diagnostics_path, address_rows)
        append_jsonl(energy_diagnostics_path, energy_rows)
        write_json(RUN_DIR / "config.json", config_payload)
        write_json(RUN_DIR / "summary.json", final_summary)
        progress.advance(METADATA_WORK_UNITS, "write metrics/config")

        _write_curves(metrics_rows, address_rows, graph_snapshots)
        progress.advance(CURVE_WORK_UNITS, "write curves")
        _write_heatmaps(tasks, graph, metrics_rows, final_train_rows, address_rows, energy_rows)
        progress.advance(HEATMAP_WORK_UNITS, "write heatmaps")
        _write_reconstruction_grids(tasks, graph, backend, progress)
        _write_report(tasks, graph, final_summary, graph_snapshots, report_title)
        progress.advance(REPORT_WORK_UNITS, "write report")
    print(RUN_DIR)


def _run_sequential_baseline(
    algorithm: str,
    tasks: tuple[TaskDataset, ...],
    backend: ModelBackend,
    replay: bool,
    progress: BenchmarkProgress | None = None,
) -> BaselineRun:
    state = backend.init_state(jax.random.PRNGKey(backend.train_config.seed))
    rows: list[dict[str, object]] = []
    replay_canvases: list[np.ndarray] = []
    replay_labels: list[np.ndarray] = []
    for stage, task in enumerate(tasks, start=1):
        phase_prefix = f"{algorithm} stage {stage}/{len(tasks)} {task.spec.name}"
        train_canvases = task.train_canvases()
        train_labels = task.train_labels
        if replay_canvases:
            train_canvases = np.concatenate((train_canvases, *replay_canvases), axis=0)
            train_labels = np.concatenate((train_labels, *replay_labels), axis=0)
        _progress_phase(progress, f"{phase_prefix}: train")
        state, _ = backend.continue_train(
            state,
            train_canvases,
            task.test_canvases(),
            train_labels,
            task.test_labels,
            collect_epoch_metrics=False,
            progress_callback=_progress_callback(progress, 1, f"{phase_prefix}: train"),
        )
        _progress_phase(progress, f"{phase_prefix}: train complete")
        rows.extend(
            _evaluate_params_across_tasks(
                algorithm,
                stage,
                task.spec.name,
                state.params,
                tasks[:stage],
                backend,
                progress,
                phase_prefix,
            )
        )
        if replay:
            indices = _balanced_indices(task.train_labels, REPLAY_EXAMPLES_PER_TASK, seed=stage * 10_000)
            replay_canvases.append(task.train_canvases()[indices])
            replay_labels.append(task.train_labels[indices])
    return BaselineRun(rows=rows, final_params=state.params)


def _run_dense_memory(
    tasks: tuple[TaskDataset, ...],
    backend: ModelBackend,
    progress: BenchmarkProgress | None = None,
) -> tuple[
    DenseMemoryGraph,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[GraphSnapshot, ...],
]:
    root_state = backend.init_state(jax.random.PRNGKey(backend.train_config.seed))
    graph = init_dense_memory_graph(root_state.params)
    rows: list[dict[str, object]] = []
    address_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    graph_snapshots: list[GraphSnapshot] = []
    for stage, task in enumerate(tasks, start=1):
        phase_prefix = f"dense memory stage {stage}/{len(tasks)} {task.spec.name}"
        probe_count = min(PARENT_PROBE_COUNT, task.train_labels.shape[0])
        parent_id = best_parent_by_observed_energy(
            graph,
            task.train_canvases(),
            jax.random.fold_in(root_state.rng_key, stage),
            backend.train_config,
            probe_count=PARENT_PROBE_COUNT,
            backend=backend,
            progress_callback=_progress_callback(
                progress,
                1,
                f"{phase_prefix}: parent probe",
            ),
        )
        parent_params = effective_params(graph, parent_id)
        child_state = backend.init_state_from_params(parent_params, jax.random.fold_in(root_state.rng_key, 100 + stage))
        _progress_phase(progress, f"{phase_prefix}: train from {parent_id}")
        child_state, _ = backend.continue_train(
            child_state,
            task.train_canvases(),
            task.test_canvases(),
            task.train_labels,
            task.test_labels,
            collect_epoch_metrics=False,
            progress_callback=_progress_callback(progress, 1, f"{phase_prefix}: train from {parent_id}"),
        )
        _progress_phase(progress, f"{phase_prefix}: train complete")
        child_id = f"node_{stage}_{task.spec.name}"
        graph = add_dense_delta_node(graph, child_id, parent_id, child_state.params, task.spec.name, stage)
        rows.extend(
            _evaluate_memory_across_tasks(
                graph,
                tasks[:stage],
                stage,
                task.spec.name,
                backend,
                address_rows,
                energy_rows,
                progress,
            )
        )
        graph_snapshots.append(_write_graph_snapshot(graph, tasks[:stage], stage, backend, progress))
    return graph, rows, address_rows, energy_rows, tuple(graph_snapshots)


def _evaluate_params_across_tasks(
    algorithm: str,
    stage: int,
    train_task: str,
    params: ParamTree,
    tasks: tuple[TaskDataset, ...],
    backend: ModelBackend,
    progress: BenchmarkProgress | None = None,
    phase_prefix: str = "baseline",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        phase = f"{phase_prefix}: eval {task.spec.name}"
        _progress_phase(progress, phase)
        metrics = backend.evaluate(
            params,
            task.test_canvases(),
            task.test_labels,
            jax.random.PRNGKey(stage * 100 + task_index),
            progress_callback=_progress_callback(progress, 1, phase),
        )
        rows.append(_metric_row(algorithm, stage, train_task, task.spec.name, metrics))
    return rows


def _evaluate_memory_across_tasks(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    stage: int,
    train_task: str,
    backend: ModelBackend,
    address_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
    progress: BenchmarkProgress | None = None,
) -> list[dict[str, object]]:
    oracle_nodes = task_node_ids(graph)
    rows: list[dict[str, object]] = []
    candidate_count = len(node_ids(graph))
    for task_index, task in enumerate(tasks):
        phase_prefix = f"dense memory stage {stage}: eval {task.spec.name}"
        oracle_node_id = oracle_nodes[task.spec.name]
        oracle_phase = f"{phase_prefix}: oracle"
        _progress_phase(progress, oracle_phase)
        oracle_metrics = evaluate_node_on_task(
            graph,
            oracle_node_id,
            task,
            jax.random.PRNGKey(stage * 1_000 + task_index),
            backend.train_config,
            backend,
            progress_callback=_progress_callback(progress, 1, oracle_phase),
        )
        address_phase = f"{phase_prefix}: address {candidate_count} nodes"
        eval_phase = f"{phase_prefix}: addressed metrics"
        _progress_phase(progress, address_phase)
        addressed_eval = evaluate_addressed_on_task(
            graph,
            task,
            oracle_node_id,
            jax.random.PRNGKey(stage * 2_000 + task_index),
            backend.train_config,
            backend=backend,
            progress_callback=_progress_callback(
                progress,
                1,
                address_phase,
            ),
            eval_progress_callback=_progress_callback(progress, 1, eval_phase),
        )
        rows.extend(
            (
                _metric_row("memory_oracle", stage, train_task, task.spec.name, oracle_metrics, {"node_id": oracle_node_id}),
                _metric_row(
                    "addressed_memory",
                    stage,
                    train_task,
                    task.spec.name,
                    addressed_eval.metrics,
                    {
                        "oracle_node_id": oracle_node_id,
                        "address_accuracy": addressed_eval.address_accuracy,
                        "mean_selected_energy": addressed_eval.mean_selected_energy,
                        "candidates_scored": len(node_ids(graph)),
                    },
                ),
            )
        )
        _append_address_diagnostics(
            address_rows,
            energy_rows,
            stage,
            "test",
            task.spec.name,
            addressed_eval,
        )
    return rows


def _write_graph_snapshot(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    stage: int,
    backend: ModelBackend,
    progress: BenchmarkProgress | None = None,
) -> GraphSnapshot:
    eval_by_node = _evaluate_graph_nodes(graph, tasks, backend, stage, progress)
    winner_by_task = _winning_nodes_by_task(graph, tasks, eval_by_node, backend.accuracy_key)
    filename = f"memory_graph_stage_{stage}.svg"
    write_memory_graph_svg(
        RUN_DIR / filename,
        graph,
        _node_visual_stats(graph, tasks, eval_by_node, winner_by_task, backend.accuracy_key),
        _edge_visual_stats(graph, eval_by_node),
        f"Memory Graph After Stage {stage}",
    )
    return GraphSnapshot(stage=stage, filename=filename, node_count=len(node_ids(graph)), memory_bytes=graph_memory_bytes(graph))


def _evaluate_graph_nodes(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    backend: ModelBackend,
    stage: int,
    progress: BenchmarkProgress | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    eval_by_node: dict[tuple[str, str], dict[str, float]] = {}
    for node_index, node_id in enumerate(node_ids(graph)):
        for task_index, task in enumerate(tasks):
            phase = f"dense memory stage {stage}: graph eval {node_id} on {task.spec.name}"
            _progress_phase(progress, phase)
            eval_by_node[(node_id, task.spec.name)] = evaluate_node_on_task(
                graph,
                node_id,
                task,
                jax.random.PRNGKey(stage * 10_000 + node_index * 100 + task_index),
                backend.train_config,
                backend,
                progress_callback=_progress_callback(progress, 1, phase),
            )
    return eval_by_node


def _winning_nodes_by_task(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    eval_by_node: dict[tuple[str, str], dict[str, float]],
    accuracy_key: str,
) -> dict[str, str]:
    return {
        task.spec.name: max(
            node_ids(graph),
            key=lambda node_id: (
                eval_by_node[(node_id, task.spec.name)][accuracy_key],
                -eval_by_node[(node_id, task.spec.name)]["loss"],
            ),
        )
        for task in tasks
    }


def _node_visual_stats(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    eval_by_node: dict[tuple[str, str], dict[str, float]],
    winner_by_task: dict[str, str],
    accuracy_key: str,
) -> dict[str, NodeVisualStats]:
    return {
        node.node_id: NodeVisualStats(
            node_id=node.node_id,
            trained_task=node.trained_task,
            depth=node.depth,
            memory_bytes=node_memory_bytes(graph, node.node_id),
            eval_wins=tuple(task_name for task_name, winner_id in winner_by_task.items() if winner_id == node.node_id),
            best_task_accuracy=max(
                (eval_by_node[(node.node_id, task.spec.name)][accuracy_key] for task in tasks),
                default=0.0,
            ),
        )
        for node in graph.nodes
    }


def _edge_visual_stats(
    graph: DenseMemoryGraph,
    eval_by_node: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str], EdgeVisualStats]:
    return {
        (stats.parent_id, stats.child_id): EdgeVisualStats(
            parent_id=stats.parent_id,
            child_id=stats.child_id,
            child_task=stats.child_task,
            delta_l2_norm=stats.delta_l2_norm,
            delta_bytes=stats.delta_bytes,
            eval_gain=_edge_eval_gain(stats.parent_id, stats.child_id, stats.child_task, eval_by_node),
        )
        for stats in edge_memory_stats(graph)
    }


def _edge_eval_gain(
    parent_id: str,
    child_id: str,
    child_task: str,
    eval_by_node: dict[tuple[str, str], dict[str, float]],
) -> float:
    if (parent_id, child_task) not in eval_by_node or (child_id, child_task) not in eval_by_node:
        return 0.0
    return eval_by_node[(child_id, child_task)][ACCURACY_KEY] - eval_by_node[(parent_id, child_task)][ACCURACY_KEY]


def _benchmark_work_units(
    tasks: tuple[TaskDataset, ...],
    backend: ModelBackend,
    task_epochs: int,
    replay_examples_per_task: int,
    parent_probe_count: int,
    report_canvas_count: int,
    include_baselines: bool,
) -> int:
    total = 0
    if include_baselines:
        for stage, task in enumerate(tasks, start=1):
            total += _training_work_units(backend, task.train_labels.shape[0], task_epochs)
            total += sum(_eval_work_units(backend, eval_task.test_labels.shape[0]) for eval_task in tasks[:stage])

        for stage, task in enumerate(tasks, start=1):
            replay_count = sum(min(replay_examples_per_task, previous_task.train_labels.shape[0]) for previous_task in tasks[: stage - 1])
            train_count = task.train_labels.shape[0] + replay_count
            total += _training_work_units(backend, train_count, task_epochs)
            total += sum(_eval_work_units(backend, eval_task.test_labels.shape[0]) for eval_task in tasks[:stage])

    for stage, task in enumerate(tasks, start=1):
        parent_candidates = stage
        graph_candidates = stage + 1
        probe_count = min(parent_probe_count, task.train_labels.shape[0])
        total += parent_candidates * _eval_work_units(backend, probe_count)
        total += _training_work_units(backend, task.train_labels.shape[0], task_epochs)
        total += sum(
            _eval_work_units(backend, eval_task.test_labels.shape[0])
            + graph_candidates * _eval_work_units(backend, eval_task.test_labels.shape[0])
            + _eval_work_units(backend, eval_task.test_labels.shape[0])
            for eval_task in tasks[:stage]
        )
        total += graph_candidates * sum(_eval_work_units(backend, eval_task.test_labels.shape[0]) for eval_task in tasks[:stage])

    final_node_count = len(tasks) + 1
    for task in tasks:
        if include_baselines:
            total += 2 * _eval_work_units(backend, task.train_labels.shape[0])
        total += _eval_work_units(backend, task.train_labels.shape[0])
        total += final_node_count * _eval_work_units(backend, task.train_labels.shape[0])
        total += _eval_work_units(backend, task.train_labels.shape[0])
        recon_count = min(report_canvas_count, task.test_labels.shape[0])
        total += 2 + final_node_count * _eval_work_units(backend, recon_count)

    return total + SUMMARY_WORK_UNITS + METADATA_WORK_UNITS + CURVE_WORK_UNITS + HEATMAP_WORK_UNITS + REPORT_WORK_UNITS


def _training_work_units(backend: ModelBackend, example_count: int, epochs: int) -> int:
    return max(1, epochs) * _batched_work_units(example_count, _batch_size(backend))


def _eval_work_units(backend: ModelBackend, example_count: int) -> int:
    return _batched_work_units(example_count, _eval_batch_size(backend))


def _batched_work_units(example_count: int, batch_size: int) -> int:
    return max(1, ceil(max(1, example_count) / max(1, batch_size)))


def _batch_size(backend: ModelBackend) -> int:
    return int(getattr(backend.train_config, "batch_size", 256))


def _eval_batch_size(backend: ModelBackend) -> int:
    return int(getattr(backend.train_config, "eval_batch_size", _batch_size(backend)))


def _progress_phase(progress: BenchmarkProgress | None, phase: str) -> None:
    if progress is not None:
        progress.set_phase(phase)


def _progress_advance(progress: BenchmarkProgress | None, units: int, phase: str | None = None) -> None:
    if progress is not None:
        progress.advance(units, phase)


def _progress_callback(progress: BenchmarkProgress | None, units: int, phase: str):
    def advance() -> None:
        _progress_advance(progress, units, phase)

    return advance if progress is not None else None


def _append_address_diagnostics(
    address_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
    stage: int,
    split: str,
    eval_task: str,
    addressed_eval: object,
) -> None:
    selected_counts = getattr(addressed_eval, "selected_counts")
    candidate_mean_energies = getattr(addressed_eval, "candidate_mean_energies")
    address_rows.extend(
        {
            "stage": stage,
            "split": split,
            "eval_task": eval_task,
            "selected_node_id": selected_node_id,
            "count": count,
        }
        for selected_node_id, count in selected_counts.items()
    )
    energy_rows.extend(
        {
            "stage": stage,
            "split": split,
            "eval_task": eval_task,
            "node_id": node_id,
            "mean_observed_energy": mean_energy,
        }
        for node_id, mean_energy in candidate_mean_energies.items()
    )


def _address_count_tuples(
    address_rows: list[dict[str, object]],
    final_stage: int,
    split: str,
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (str(row["eval_task"]), str(row["selected_node_id"]), int(row["count"]))
        for row in address_rows
        if int(row["stage"]) == final_stage and str(row.get("split", "test")) == split
    )


def _observed_energy_heatmap(
    energy_rows: list[dict[str, object]],
    final_stage: int,
    split: str,
    task_names: tuple[str, ...],
    memory_node_ids: tuple[str, ...],
) -> np.ndarray:
    matrix = np.full((len(task_names), len(memory_node_ids)), np.nan, dtype=np.float32)
    task_index = {task_name: index for index, task_name in enumerate(task_names)}
    node_index = {node_id: index for index, node_id in enumerate(memory_node_ids)}
    for row in energy_rows:
        if int(row["stage"]) != final_stage or str(row.get("split", "test")) != split:
            continue
        matrix[task_index[str(row["eval_task"])], node_index[str(row["node_id"])]] = float(row["mean_observed_energy"])
    if np.isnan(matrix).any():
        missing = np.argwhere(np.isnan(matrix))
        raise ValueError(f"missing observed-energy diagnostics for {split}: {missing.tolist()}")
    return matrix


def _write_curves(
    metrics_rows: list[dict[str, object]],
    address_rows: list[dict[str, object]],
    graph_snapshots: tuple[GraphSnapshot, ...],
) -> None:
    write_svg_line_chart(
        RUN_DIR / "average_accuracy_curves.svg",
        _stage_algorithm_chart_rows(metrics_rows, ACCURACY_KEY),
        tuple((algorithm, algorithm) for algorithm in _algorithms(metrics_rows)),
        "Average Accuracy by Stage",
        "energy classifier accuracy",
    )
    write_svg_line_chart(
        RUN_DIR / "average_loss_curves.svg",
        _stage_algorithm_chart_rows(metrics_rows, "loss"),
        tuple((algorithm, algorithm) for algorithm in _algorithms(metrics_rows)),
        "Average Loss by Stage",
        "loss",
    )
    write_svg_line_chart(
        RUN_DIR / "forgetting_curves.svg",
        _forgetting_chart_rows(metrics_rows),
        tuple((algorithm, algorithm) for algorithm in _algorithms(metrics_rows)),
        "Mean Forgetting by Stage",
        "best previous accuracy - current",
    )
    write_svg_line_chart(
        RUN_DIR / "addressing_cost_curves.svg",
        _addressing_chart_rows(metrics_rows, address_rows),
        (("candidates_scored", "candidates scored"), ("address_accuracy", "address accuracy")),
        "Addressing Cost and Accuracy",
        "value",
    )
    write_svg_line_chart(
        RUN_DIR / "memory_growth_curves.svg",
        tuple(
            {
                "epoch": snapshot.stage,
                "graph_nodes": snapshot.node_count,
                "memory_mb": snapshot.memory_bytes / 1_000_000.0,
            }
            for snapshot in graph_snapshots
        ),
        (("graph_nodes", "graph nodes"), ("memory_mb", "memory MB")),
        "Memory Growth",
        "count / MB",
    )


def _write_heatmaps(
    tasks: tuple[TaskDataset, ...],
    graph: DenseMemoryGraph,
    metrics_rows: list[dict[str, object]],
    final_train_rows: list[dict[str, object]],
    address_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
) -> None:
    task_names = tuple(task.spec.name for task in tasks)
    memory_node_ids = node_ids(graph)
    final_stage = len(tasks)
    algorithms = _algorithms(metrics_rows)
    test_retention = np.asarray(
        [
            [
                _row_for(metrics_rows, algorithm, final_stage, task_name)[ACCURACY_KEY]
                for task_name in task_names
            ]
            for algorithm in algorithms
        ],
        dtype=np.float32,
    )
    train_retention = np.asarray(
        [
            [
                _row_for(final_train_rows, algorithm, final_stage, task_name)[ACCURACY_KEY]
                for task_name in task_names
            ]
            for algorithm in algorithms
        ],
        dtype=np.float32,
    )
    final_test_address_rows = _address_count_tuples(address_rows, final_stage, "test")
    final_train_address_rows = _address_count_tuples(address_rows, final_stage, "train")
    (RUN_DIR / "final_retention_heatmap.svg").unlink(missing_ok=True)
    write_svg_heatmap(
        RUN_DIR / "final_test_retention_heatmap.svg",
        test_retention,
        algorithms,
        task_names,
        "Final Test Retention (Held-Out Test Split)",
    )
    write_svg_heatmap(
        RUN_DIR / "final_train_retention_heatmap.svg",
        train_retention,
        algorithms,
        task_names,
        "Final Train Retention (Training Split)",
    )
    write_svg_heatmap(
        RUN_DIR / "address_confusion_heatmap.svg",
        address_confusion_matrix(final_test_address_rows, task_names, memory_node_ids),
        task_names,
        memory_node_ids,
        "Final Test Address Confusion",
        value_format=".0f",
    )
    write_svg_heatmap(
        RUN_DIR / "final_test_address_confusion_heatmap.svg",
        address_confusion_matrix(final_test_address_rows, task_names, memory_node_ids),
        task_names,
        memory_node_ids,
        "Final Test Address Confusion",
        value_format=".0f",
    )
    write_svg_heatmap(
        RUN_DIR / "final_train_address_confusion_heatmap.svg",
        address_confusion_matrix(final_train_address_rows, task_names, memory_node_ids),
        task_names,
        memory_node_ids,
        "Final Train Address Confusion",
        value_format=".0f",
    )
    write_svg_heatmap(
        RUN_DIR / "final_test_observed_energy_heatmap.svg",
        _observed_energy_heatmap(energy_rows, final_stage, "test", task_names, memory_node_ids),
        task_names,
        memory_node_ids,
        "Final Test Mean Observed Energy (Lower Better)",
        value_format=".3f",
    )
    write_svg_heatmap(
        RUN_DIR / "final_train_observed_energy_heatmap.svg",
        _observed_energy_heatmap(energy_rows, final_stage, "train", task_names, memory_node_ids),
        task_names,
        memory_node_ids,
        "Final Train Mean Observed Energy (Lower Better)",
        value_format=".3f",
    )


def _final_split_rows(
    tasks: tuple[TaskDataset, ...],
    graph: DenseMemoryGraph,
    baseline_params: dict[str, ParamTree],
    backend: ModelBackend,
    split: str,
    address_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
    progress: BenchmarkProgress | None = None,
) -> list[dict[str, object]]:
    split_canvases = {
        task.spec.name: task.train_canvases() if split == "train" else task.test_canvases()
        for task in tasks
    }
    split_labels = {
        task.spec.name: task.train_labels if split == "train" else task.test_labels
        for task in tasks
    }
    final_stage = len(tasks)
    rows: list[dict[str, object]] = []
    for algorithm_index, (algorithm, params) in enumerate(baseline_params.items()):
        for task_index, task in enumerate(tasks):
            phase = f"final {split}: {algorithm} on {task.spec.name}"
            _progress_phase(progress, phase)
            metrics = backend.evaluate(
                params,
                split_canvases[task.spec.name],
                split_labels[task.spec.name],
                jax.random.PRNGKey(70_000 + algorithm_index * 1_000 + task_index),
                progress_callback=_progress_callback(progress, 1, phase),
            )
            rows.append(
                _metric_row(
                    algorithm,
                    final_stage,
                    tasks[-1].spec.name,
                    task.spec.name,
                    metrics,
                    {"split": split},
                )
            )
    oracle_nodes = task_node_ids(graph)
    memory_rows = [
        row
        for task_index, task in enumerate(tasks)
        for row in _final_memory_split_rows(
            graph,
            task,
            split_canvases[task.spec.name],
            split_labels[task.spec.name],
            oracle_nodes[task.spec.name],
            final_stage,
            split,
            jax.random.PRNGKey(80_000 + task_index),
            backend,
            address_rows,
            energy_rows,
            progress,
        )
    ]
    return rows + memory_rows


def _final_memory_split_rows(
    graph: DenseMemoryGraph,
    task: TaskDataset,
    canvases: np.ndarray,
    labels: np.ndarray,
    oracle_node_id: str,
    final_stage: int,
    split: str,
    rng_key: jax.Array,
    backend: ModelBackend,
    address_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
    progress: BenchmarkProgress | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    phase_prefix = f"final {split}: memory on {task.spec.name}"
    oracle_phase = f"{phase_prefix}: oracle"
    _progress_phase(progress, oracle_phase)
    oracle_metrics = evaluate_node_on_arrays(
        graph,
        oracle_node_id,
        canvases,
        labels,
        rng_key,
        backend.train_config,
        backend,
        progress_callback=_progress_callback(progress, 1, oracle_phase),
    )
    address_phase = f"{phase_prefix}: address {len(node_ids(graph))} nodes"
    eval_phase = f"{phase_prefix}: addressed metrics"
    _progress_phase(progress, address_phase)
    addressed_eval = evaluate_addressed_on_arrays(
        graph,
        canvases,
        labels,
        oracle_node_id,
        jax.random.fold_in(rng_key, 1),
        backend.train_config,
        backend=backend,
        progress_callback=_progress_callback(
            progress,
            1,
            address_phase,
        ),
        eval_progress_callback=_progress_callback(progress, 1, eval_phase),
    )
    _append_address_diagnostics(
        address_rows,
        energy_rows,
        final_stage,
        split,
        task.spec.name,
        addressed_eval,
    )
    return (
        _metric_row(
            "memory_oracle",
            final_stage,
            task.spec.name,
            task.spec.name,
            oracle_metrics,
            {"node_id": oracle_node_id, "split": split},
        ),
        _metric_row(
            "addressed_memory",
            final_stage,
            task.spec.name,
            task.spec.name,
            addressed_eval.metrics,
            {
                "oracle_node_id": oracle_node_id,
                "address_accuracy": addressed_eval.address_accuracy,
                "mean_selected_energy": addressed_eval.mean_selected_energy,
                "candidates_scored": len(node_ids(graph)),
                "split": split,
            },
        ),
)


def _write_reconstruction_grids(
    tasks: tuple[TaskDataset, ...],
    graph: DenseMemoryGraph,
    backend: ModelBackend,
    progress: BenchmarkProgress | None = None,
) -> None:
    oracle_nodes = task_node_ids(graph)
    for task_index, task in enumerate(tasks):
        task_slug = _slug(task.spec.name)
        canvases = task.test_canvases()[:REPORT_CANVAS_COUNT]
        oracle_params = effective_params(graph, oracle_nodes[task.spec.name])
        write_png_grid(
            RUN_DIR / f"oracle_recon_{task_slug}.png",
            backend.reconstruct(oracle_params, canvases, jax.random.PRNGKey(50_000 + task_index), mask_label=True),
        )
        _progress_advance(progress, 1, f"reconstructions: oracle {task.spec.name}")
        write_png_grid(
            RUN_DIR / f"addressed_recon_{task_slug}.png",
            _addressed_reconstructions(
                graph,
                canvases,
                jax.random.PRNGKey(60_000 + task_index),
                backend,
                progress,
                f"reconstructions: address {task.spec.name}",
            ),
        )
        _progress_advance(progress, 1, f"reconstructions: addressed {task.spec.name}")


def _addressed_reconstructions(
    graph: DenseMemoryGraph,
    canvases: np.ndarray,
    rng_key: jax.Array,
    backend: ModelBackend,
    progress: BenchmarkProgress | None = None,
    phase: str = "reconstructions: address",
) -> np.ndarray:
    selected_node_ids, _ = select_addresses(
        graph,
        canvases,
        rng_key,
        backend.train_config,
        backend=backend,
        progress_callback=_progress_callback(progress, 1, phase),
    )
    selected_array = np.asarray(selected_node_ids, dtype=object)
    reconstructions = np.zeros((canvases.shape[0], canvases.shape[1] * canvases.shape[2]), dtype=np.float32)
    for node_index, node_id in enumerate(sorted(set(selected_node_ids))):
        mask = selected_array == node_id
        reconstructions[mask] = backend.reconstruct(
            effective_params(graph, node_id),
            canvases[mask],
            jax.random.fold_in(rng_key, node_index),
            mask_label=True,
        )
    return reconstructions


def _write_report(
    tasks: tuple[TaskDataset, ...],
    graph: DenseMemoryGraph,
    summary: dict[str, object],
    graph_snapshots: tuple[GraphSnapshot, ...],
    report_title: str,
) -> None:
    task_names = tuple(task.spec.name for task in tasks)
    graph_images = tuple(snapshot.filename for snapshot in graph_snapshots)
    recon_images = tuple(
        image_name
        for task_name in task_names
        for image_name in (f"oracle_recon_{_slug(task_name)}.png", f"addressed_recon_{_slug(task_name)}.png")
    )
    RUN_DIR.joinpath("report.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{html.escape(report_title)}</title>",
                "<style>",
                _report_css(),
                "</style>",
                "</head>",
                "<body><main>",
                f"<h1>{html.escape(report_title)}</h1>",
                "<section><h2>Summary</h2>",
                _summary_table(summary),
                "</section>",
                "<section><h2>Curves</h2>",
                _figure_grid(("average_accuracy_curves.svg", "average_loss_curves.svg", "forgetting_curves.svg", "addressing_cost_curves.svg", "memory_growth_curves.svg")),
                "</section>",
                "<section><h2>Heatmaps</h2>",
                _figure_grid(
                    (
                        "final_test_retention_heatmap.svg",
                        "final_train_retention_heatmap.svg",
                        "final_test_address_confusion_heatmap.svg",
                        "final_train_address_confusion_heatmap.svg",
                        "final_test_observed_energy_heatmap.svg",
                        "final_train_observed_energy_heatmap.svg",
                    )
                ),
                "</section>",
                "<section><h2>Memory Graphs</h2>",
                _figure_grid(graph_images),
                "</section>",
                "<section><h2>Reconstructions</h2>",
                _figure_grid(recon_images),
                "</section>",
                "<section><h2>Raw Summary</h2>",
                f"<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>",
                "</section>",
                "</main>",
                _report_lightbox(),
                "<script>",
                _report_script(),
                "</script>",
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _metric_row(
    algorithm: str,
    stage: int,
    train_task: str,
    eval_task: str,
    metrics: dict[str, float],
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "stage": stage,
        "train_task": train_task,
        "eval_task": eval_task,
        **metrics,
        **({} if extras is None else extras),
    }


def _stage_algorithm_chart_rows(metrics_rows: list[dict[str, object]], metric_name: str) -> tuple[dict[str, int | float], ...]:
    stages = sorted({int(row["stage"]) for row in metrics_rows})
    algorithms = _algorithms(metrics_rows)
    return tuple(
        {
            "epoch": stage,
            **{
                algorithm: float(np.mean([float(row[metric_name]) for row in metrics_rows if row["algorithm"] == algorithm and int(row["stage"]) == stage]))
                for algorithm in algorithms
            },
        }
        for stage in stages
    )


def _forgetting_chart_rows(metrics_rows: list[dict[str, object]]) -> tuple[dict[str, int | float], ...]:
    stages = sorted({int(row["stage"]) for row in metrics_rows})
    algorithms = _algorithms(metrics_rows)
    return tuple(
        {
            "epoch": stage,
            **{algorithm: _mean_forgetting(metrics_rows, algorithm, stage) for algorithm in algorithms},
        }
        for stage in stages
    )


def _addressing_chart_rows(
    metrics_rows: list[dict[str, object]],
    address_rows: list[dict[str, object]],
) -> tuple[dict[str, int | float], ...]:
    stages = sorted({int(row["stage"]) for row in metrics_rows if row["algorithm"] == "addressed_memory"})
    return tuple(
        {
            "epoch": stage,
            "candidates_scored": float(np.mean([float(row["candidates_scored"]) for row in metrics_rows if row["algorithm"] == "addressed_memory" and int(row["stage"]) == stage])),
            "address_accuracy": float(np.mean([float(row["address_accuracy"]) for row in metrics_rows if row["algorithm"] == "addressed_memory" and int(row["stage"]) == stage])),
            "selected_examples": float(
                sum(
                    int(row["count"])
                    for row in address_rows
                    if int(row["stage"]) == stage and str(row.get("split", "test")) == "test"
                )
            ),
        }
        for stage in stages
    )


def _mean_forgetting(metrics_rows: list[dict[str, object]], algorithm: str, stage: int) -> float:
    task_names = sorted({str(row["eval_task"]) for row in metrics_rows if row["algorithm"] == algorithm and int(row["stage"]) <= stage})
    forgetting_values = []
    for task_name in task_names:
        history = [
            float(row[ACCURACY_KEY])
            for row in metrics_rows
            if row["algorithm"] == algorithm and row["eval_task"] == task_name and int(row["stage"]) <= stage
        ]
        if history:
            forgetting_values.append(max(history) - history[-1])
    return float(np.mean(forgetting_values)) if forgetting_values else 0.0


def _summary_payload(
    config_payload: dict[str, object],
    graph: DenseMemoryGraph,
    metrics_rows: list[dict[str, object]],
    address_rows: list[dict[str, object]],
    graph_snapshots: tuple[GraphSnapshot, ...],
    final_train_rows: list[dict[str, object]],
) -> dict[str, object]:
    final_stage = max(int(row["stage"]) for row in metrics_rows)
    final_rows = [row for row in metrics_rows if int(row["stage"]) == final_stage]
    return {
        "config": config_payload,
        "final_stage": final_stage,
        "graph_node_count": len(node_ids(graph)),
        "graph_memory_bytes": graph_memory_bytes(graph),
        "graph_snapshots": tuple(
            {
                "stage": snapshot.stage,
                "filename": snapshot.filename,
                "node_count": snapshot.node_count,
                "memory_bytes": snapshot.memory_bytes,
            }
            for snapshot in graph_snapshots
        ),
        "final_average_accuracy": {
            algorithm: float(np.mean([float(row[ACCURACY_KEY]) for row in final_rows if row["algorithm"] == algorithm]))
            for algorithm in _algorithms(metrics_rows)
        },
        "final_train_average_accuracy": {
            algorithm: float(np.mean([float(row[ACCURACY_KEY]) for row in final_train_rows if row["algorithm"] == algorithm]))
            for algorithm in _algorithms(metrics_rows)
        },
        "final_address_counts": {
            f'{row.get("split", "test")}:{row["eval_task"]}:{row["selected_node_id"]}': int(row["count"])
            for row in address_rows
            if int(row["stage"]) == final_stage
        },
    }


def _config_payload(
    backend: ModelBackend,
    tasks: tuple[TaskDataset, ...],
    stream_payload: dict[str, object],
) -> dict[str, object]:
    return {
        **backend.config_payload(),
        "stream": {**stream_payload, "task_names": tuple(task.spec.name for task in tasks)},
    }


def _row_for(metrics_rows: list[dict[str, object]], algorithm: str, stage: int, task_name: str) -> dict[str, object]:
    matches = [
        row
        for row in metrics_rows
        if row["algorithm"] == algorithm and int(row["stage"]) == stage and row["eval_task"] == task_name
    ]
    if not matches:
        raise KeyError(f"missing row for {algorithm} stage {stage} task {task_name}")
    return matches[-1]


def _algorithms(metrics_rows: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(row["algorithm"]) for row in metrics_rows))


def _balanced_indices(labels: np.ndarray, requested_count: int, seed: int) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    labels_in_order = tuple(int(label) for label in np.unique(label_array))
    target_count = min(requested_count, label_array.shape[0])
    base_count, remainder = divmod(target_count, len(labels_in_order))
    rng = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            tuple(
                rng.choice(
                    np.flatnonzero(label_array == label),
                    size=min(base_count + (1 if label_index < remainder else 0), np.sum(label_array == label)),
                    replace=False,
                )
                for label_index, label in enumerate(labels_in_order)
            )
        )
    ).astype(np.int64)


def _summary_table(summary: dict[str, object]) -> str:
    rows = (
        ("final_stage", summary["final_stage"]),
        ("graph_node_count", summary["graph_node_count"]),
        ("graph_memory_bytes", summary["graph_memory_bytes"]),
        ("final_average_accuracy", summary["final_average_accuracy"]),
    )
    return "<table>" + "\n".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows) + "</table>"


def _figure_grid(filenames: tuple[str, ...]) -> str:
    return '<div class="grid">' + "\n".join(
        (
            f'<figure class="figure-card" role="button" tabindex="0" '
            f'data-lightbox-src="{html.escape(filename)}" data-lightbox-caption="{html.escape(filename)}">'
            f'<img src="{html.escape(filename)}" alt="{html.escape(filename)}">'
            f"<figcaption>{html.escape(filename)}</figcaption>"
            "</figure>"
        )
        for filename in filenames
    ) + "</div>"


def _report_lightbox() -> str:
    return "\n".join(
        [
            '<div id="report-lightbox" class="lightbox" hidden>',
            '<button type="button" class="lightbox-close" aria-label="Close">Close</button>',
            '<figure class="lightbox-figure">',
            '<img id="report-lightbox-image" alt="">',
            '<figcaption id="report-lightbox-caption"></figcaption>',
            "</figure>",
            "</div>",
        ]
    )


def _report_script() -> str:
    return r"""
const lightbox = document.getElementById("report-lightbox");
const lightboxImage = document.getElementById("report-lightbox-image");
const lightboxCaption = document.getElementById("report-lightbox-caption");
const closeButton = lightbox.querySelector(".lightbox-close");

function openLightbox(card) {
  const src = card.dataset.lightboxSrc;
  const caption = card.dataset.lightboxCaption || src;
  lightboxImage.src = src;
  lightboxImage.alt = caption;
  lightboxCaption.textContent = caption;
  lightbox.hidden = false;
  document.body.classList.add("modal-open");
  closeButton.focus();
}

function closeLightbox() {
  lightbox.hidden = true;
  document.body.classList.remove("modal-open");
  lightboxImage.removeAttribute("src");
}

document.querySelectorAll(".figure-card").forEach((card) => {
  card.addEventListener("click", () => openLightbox(card));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLightbox(card);
    }
  });
});

closeButton.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) {
    closeLightbox();
  }
});
""".strip()


def _report_css() -> str:
    return """
body { margin: 0; background: #f8fafc; color: #111827; font: 15px/1.45 Inter, Arial, sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 32px 24px 48px; }
h1 { margin: 0 0 24px; font-size: 28px; }
h2 { margin: 0 0 14px; font-size: 20px; }
section { margin: 0 0 28px; }
table { border-collapse: collapse; width: 100%; background: #ffffff; }
th, td { border: 1px solid #d1d5db; padding: 7px 9px; text-align: left; vertical-align: top; }
th { width: 240px; background: #f3f4f6; font-weight: 650; }
pre { overflow-x: auto; background: #111827; color: #f9fafb; padding: 16px; border-radius: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
figure { margin: 0; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; }
.figure-card { cursor: zoom-in; transition: border-color 120ms ease, box-shadow 120ms ease; }
.figure-card:focus { outline: 2px solid #2563eb; outline-offset: 2px; }
.figure-card:hover { border-color: #94a3b8; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08); }
img { display: block; width: 100%; height: auto; image-rendering: auto; }
img[src$=".png"] { image-rendering: pixelated; }
figcaption { margin-top: 8px; color: #374151; font-size: 13px; }
body.modal-open { overflow: hidden; }
.lightbox[hidden] { display: none; }
.lightbox { position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; padding: 28px; background: rgba(15, 23, 42, 0.88); }
.lightbox-close { position: fixed; top: 18px; right: 20px; border: 1px solid #cbd5e1; background: #ffffff; color: #111827; border-radius: 6px; padding: 8px 12px; font: inherit; cursor: pointer; }
.lightbox-figure { width: min(96vw, 1600px); max-height: 92vh; margin: 0; padding: 14px; display: flex; flex-direction: column; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34); }
.lightbox-figure img { width: 100%; height: auto; max-height: calc(92vh - 72px); object-fit: contain; }
.lightbox-figure figcaption { margin-top: 10px; color: #111827; font-size: 14px; overflow-wrap: anywhere; }
""".strip()


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()
