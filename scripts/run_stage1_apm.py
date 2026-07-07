"""Run the Stage 1 dense-delta addressed-parameter-memory benchmark."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

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
    evaluate_addressed_on_task,
    evaluate_node_on_task,
    graph_memory_bytes,
    init_dense_memory_graph,
    node_ids,
    node_memory_bytes,
    task_node_ids,
    write_memory_graph_svg,
)
from apm.models import VaeConfig, VaeParams
from apm.models.vae_losses import flatten_canvases
from apm.training import (
    TrainConfig,
    config_to_dict,
    continue_train_epochs,
    evaluate_vae,
    init_train_state,
    init_train_state_from_params,
    reconstruct,
)
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


@dataclass(frozen=True)
class GraphSnapshot:
    """Report references for one committed memory graph snapshot."""

    stage: int
    filename: str
    node_count: int
    memory_bytes: int


def main() -> None:
    """Run the default Stage 1 benchmark and write report artifacts."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    vae_config = VaeConfig()
    train_config = TrainConfig(epochs=TASK_EPOCHS)
    tasks = make_permuted_mnist_stream(
        load_mnist(allow_download=True),
        permutation_seeds=TASK_SEEDS,
        train_count=TRAIN_EXAMPLES_PER_TASK,
        test_count=TEST_EXAMPLES_PER_TASK,
    )
    config_payload = _config_payload(vae_config, train_config, tasks)
    online_rows = _run_sequential_baseline("online_sgd", tasks, vae_config, train_config, replay=False)
    replay_rows = _run_sequential_baseline("replay_sgd", tasks, vae_config, train_config, replay=True)
    graph, memory_rows, address_rows, graph_snapshots = _run_dense_memory(tasks, vae_config, train_config)
    metrics_rows = online_rows + replay_rows + memory_rows
    final_summary = _summary_payload(config_payload, graph, metrics_rows, address_rows, graph_snapshots)

    metrics_path = RUN_DIR / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    append_jsonl(metrics_path, metrics_rows)
    write_json(RUN_DIR / "config.json", config_payload)
    write_json(RUN_DIR / "summary.json", final_summary)
    _write_curves(metrics_rows, address_rows, graph_snapshots)
    _write_heatmaps(tasks, graph, metrics_rows, address_rows)
    _write_reconstruction_grids(tasks, graph, train_config)
    _write_report(tasks, graph, final_summary, graph_snapshots)
    print(RUN_DIR)


def _run_sequential_baseline(
    algorithm: str,
    tasks: tuple[TaskDataset, ...],
    vae_config: VaeConfig,
    train_config: TrainConfig,
    replay: bool,
) -> list[dict[str, object]]:
    state = init_train_state(jax.random.PRNGKey(train_config.seed), vae_config, train_config)
    rows: list[dict[str, object]] = []
    replay_canvases: list[np.ndarray] = []
    replay_labels: list[np.ndarray] = []
    for stage, task in enumerate(tasks, start=1):
        train_canvases = task.train_canvases()
        train_labels = task.train_labels
        if replay_canvases:
            train_canvases = np.concatenate((train_canvases, *replay_canvases), axis=0)
            train_labels = np.concatenate((train_labels, *replay_labels), axis=0)
        state, _ = continue_train_epochs(
            state,
            train_canvases,
            task.test_canvases(),
            train_labels,
            task.test_labels,
            train_config,
            collect_epoch_metrics=False,
        )
        rows.extend(_evaluate_params_across_tasks(algorithm, stage, task.spec.name, state.params, tasks[:stage], train_config))
        if replay:
            indices = _balanced_indices(task.train_labels, REPLAY_EXAMPLES_PER_TASK, seed=stage * 10_000)
            replay_canvases.append(task.train_canvases()[indices])
            replay_labels.append(task.train_labels[indices])
    return rows


def _run_dense_memory(
    tasks: tuple[TaskDataset, ...],
    vae_config: VaeConfig,
    train_config: TrainConfig,
) -> tuple[DenseMemoryGraph, list[dict[str, object]], list[dict[str, object]], tuple[GraphSnapshot, ...]]:
    root_state = init_train_state(jax.random.PRNGKey(train_config.seed), vae_config, train_config)
    graph = init_dense_memory_graph(root_state.params)
    rows: list[dict[str, object]] = []
    address_rows: list[dict[str, object]] = []
    graph_snapshots: list[GraphSnapshot] = []
    for stage, task in enumerate(tasks, start=1):
        parent_id = best_parent_by_observed_energy(
            graph,
            task.train_canvases(),
            jax.random.fold_in(root_state.rng_key, stage),
            train_config,
            probe_count=PARENT_PROBE_COUNT,
        )
        parent_params = effective_params(graph, parent_id)
        child_state = init_train_state_from_params(parent_params, jax.random.fold_in(root_state.rng_key, 100 + stage), train_config)
        child_state, _ = continue_train_epochs(
            child_state,
            task.train_canvases(),
            task.test_canvases(),
            task.train_labels,
            task.test_labels,
            train_config,
            collect_epoch_metrics=False,
        )
        child_id = f"node_{stage}_{task.spec.name}"
        graph = add_dense_delta_node(graph, child_id, parent_id, child_state.params, task.spec.name, stage)
        rows.extend(_evaluate_memory_across_tasks(graph, tasks[:stage], stage, task.spec.name, train_config, address_rows))
        graph_snapshots.append(_write_graph_snapshot(graph, tasks[:stage], stage, train_config))
    return graph, rows, address_rows, tuple(graph_snapshots)


def _evaluate_params_across_tasks(
    algorithm: str,
    stage: int,
    train_task: str,
    params: VaeParams,
    tasks: tuple[TaskDataset, ...],
    train_config: TrainConfig,
) -> list[dict[str, object]]:
    return [
        _metric_row(
            algorithm,
            stage,
            train_task,
            task.spec.name,
            evaluate_vae(
                params,
                flatten_canvases(task.test_canvases()),
                task.test_labels,
                jax.random.PRNGKey(stage * 100 + task_index),
                train_config,
            ),
        )
        for task_index, task in enumerate(tasks)
    ]


def _evaluate_memory_across_tasks(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    stage: int,
    train_task: str,
    train_config: TrainConfig,
    address_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    oracle_nodes = task_node_ids(graph)
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        oracle_node_id = oracle_nodes[task.spec.name]
        oracle_metrics = evaluate_node_on_task(
            graph,
            oracle_node_id,
            task,
            jax.random.PRNGKey(stage * 1_000 + task_index),
            train_config,
        )
        addressed_eval = evaluate_addressed_on_task(
            graph,
            task,
            oracle_node_id,
            jax.random.PRNGKey(stage * 2_000 + task_index),
            train_config,
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
        address_rows.extend(
            {
                "stage": stage,
                "eval_task": task.spec.name,
                "selected_node_id": selected_node_id,
                "count": count,
            }
            for selected_node_id, count in addressed_eval.selected_counts.items()
        )
    return rows


def _write_graph_snapshot(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    stage: int,
    train_config: TrainConfig,
) -> GraphSnapshot:
    eval_by_node = _evaluate_graph_nodes(graph, tasks, train_config, stage)
    winner_by_task = _winning_nodes_by_task(graph, tasks, eval_by_node)
    filename = f"memory_graph_stage_{stage}.svg"
    write_memory_graph_svg(
        RUN_DIR / filename,
        graph,
        _node_visual_stats(graph, tasks, eval_by_node, winner_by_task),
        _edge_visual_stats(graph, eval_by_node),
        f"Memory Graph After Stage {stage}",
    )
    return GraphSnapshot(stage=stage, filename=filename, node_count=len(node_ids(graph)), memory_bytes=graph_memory_bytes(graph))


def _evaluate_graph_nodes(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    train_config: TrainConfig,
    stage: int,
) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (node_id, task.spec.name): evaluate_node_on_task(
            graph,
            node_id,
            task,
            jax.random.PRNGKey(stage * 10_000 + node_index * 100 + task_index),
            train_config,
        )
        for node_index, node_id in enumerate(node_ids(graph))
        for task_index, task in enumerate(tasks)
    }


def _winning_nodes_by_task(
    graph: DenseMemoryGraph,
    tasks: tuple[TaskDataset, ...],
    eval_by_node: dict[tuple[str, str], dict[str, float]],
) -> dict[str, str]:
    return {
        task.spec.name: max(
            node_ids(graph),
            key=lambda node_id: (
                eval_by_node[(node_id, task.spec.name)][ACCURACY_KEY],
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
) -> dict[str, NodeVisualStats]:
    return {
        node.node_id: NodeVisualStats(
            node_id=node.node_id,
            trained_task=node.trained_task,
            depth=node.depth,
            memory_bytes=node_memory_bytes(graph, node.node_id),
            eval_wins=tuple(task_name for task_name, winner_id in winner_by_task.items() if winner_id == node.node_id),
            best_task_accuracy=max(
                (eval_by_node[(node.node_id, task.spec.name)][ACCURACY_KEY] for task in tasks),
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
    address_rows: list[dict[str, object]],
) -> None:
    task_names = tuple(task.spec.name for task in tasks)
    final_stage = len(tasks)
    algorithms = _algorithms(metrics_rows)
    retention = np.asarray(
        [
            [
                _row_for(metrics_rows, algorithm, final_stage, task_name)[ACCURACY_KEY]
                for task_name in task_names
            ]
            for algorithm in algorithms
        ],
        dtype=np.float32,
    )
    final_address_rows = tuple(
        (str(row["eval_task"]), str(row["selected_node_id"]), int(row["count"]))
        for row in address_rows
        if int(row["stage"]) == final_stage
    )
    write_svg_heatmap(RUN_DIR / "final_retention_heatmap.svg", retention, algorithms, task_names, "Final Retention")
    write_svg_heatmap(
        RUN_DIR / "address_confusion_heatmap.svg",
        address_confusion_matrix(final_address_rows, task_names, node_ids(graph)),
        task_names,
        node_ids(graph),
        "Final Address Confusion",
        value_format=".0f",
    )


def _write_reconstruction_grids(tasks: tuple[TaskDataset, ...], graph: DenseMemoryGraph, train_config: TrainConfig) -> None:
    oracle_nodes = task_node_ids(graph)
    for task_index, task in enumerate(tasks):
        task_slug = _slug(task.spec.name)
        canvases = task.test_canvases()[:REPORT_CANVAS_COUNT]
        oracle_params = effective_params(graph, oracle_nodes[task.spec.name])
        write_png_grid(
            RUN_DIR / f"oracle_recon_{task_slug}.png",
            reconstruct(oracle_params, canvases, jax.random.PRNGKey(50_000 + task_index), mask_label=True),
        )
        write_png_grid(
            RUN_DIR / f"addressed_recon_{task_slug}.png",
            _addressed_reconstructions(graph, canvases, jax.random.PRNGKey(60_000 + task_index), train_config),
        )


def _addressed_reconstructions(
    graph: DenseMemoryGraph,
    canvases: np.ndarray,
    rng_key: jax.Array,
    train_config: TrainConfig,
) -> np.ndarray:
    from apm.memory import select_addresses

    selected_node_ids, _ = select_addresses(graph, canvases, rng_key, train_config)
    selected_array = np.asarray(selected_node_ids, dtype=object)
    reconstructions = np.zeros((canvases.shape[0], canvases.shape[1] * canvases.shape[2]), dtype=np.float32)
    for node_index, node_id in enumerate(sorted(set(selected_node_ids))):
        mask = selected_array == node_id
        reconstructions[mask] = reconstruct(
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
                "<title>Stage 1 APM Report</title>",
                "<style>",
                _report_css(),
                "</style>",
                "</head>",
                "<body><main>",
                "<h1>Stage 1 Dense-Delta Addressed-Parameter Memory</h1>",
                "<section><h2>Summary</h2>",
                _summary_table(summary),
                "</section>",
                "<section><h2>Curves</h2>",
                _figure_grid(("average_accuracy_curves.svg", "average_loss_curves.svg", "forgetting_curves.svg", "addressing_cost_curves.svg", "memory_growth_curves.svg")),
                "</section>",
                "<section><h2>Heatmaps</h2>",
                _figure_grid(("final_retention_heatmap.svg", "address_confusion_heatmap.svg")),
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
                "</main></body></html>",
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
            "selected_examples": float(sum(int(row["count"]) for row in address_rows if int(row["stage"]) == stage)),
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
        "final_address_counts": {
            f'{row["eval_task"]}:{row["selected_node_id"]}': int(row["count"])
            for row in address_rows
            if int(row["stage"]) == final_stage
        },
    }


def _config_payload(vae_config: VaeConfig, train_config: TrainConfig, tasks: tuple[TaskDataset, ...]) -> dict[str, object]:
    return {
        "vae": config_to_dict(vae_config),
        "train": config_to_dict(train_config),
        "stream": {
            "task_seeds": TASK_SEEDS,
            "train_examples_per_task": TRAIN_EXAMPLES_PER_TASK,
            "test_examples_per_task": TEST_EXAMPLES_PER_TASK,
            "replay_examples_per_task": REPLAY_EXAMPLES_PER_TASK,
            "task_names": tuple(task.spec.name for task in tasks),
        },
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
        f'<figure><img src="{html.escape(filename)}" alt="{html.escape(filename)}"><figcaption>{html.escape(filename)}</figcaption></figure>'
        for filename in filenames
    ) + "</div>"


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
img { display: block; width: 100%; height: auto; image-rendering: pixelated; }
figcaption { margin-top: 8px; color: #374151; font-size: 13px; }
""".strip()


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()
