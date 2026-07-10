from __future__ import annotations

import jax
import numpy as np

from apm.data.mnist import embed_batch_digits_and_labels, identity_permutation
from apm.data.mnist.task_specs import make_permuted_task
from apm.memory import (
    EdgeVisualStats,
    NodeVisualStats,
    add_dense_delta_node,
    edge_memory_stats,
    effective_params,
    evaluate_addressed_on_arrays,
    evaluate_addressed_on_task,
    init_dense_memory_graph,
    observed_energy_matrix,
    write_memory_graph_svg,
)
from apm.models import VaeConfig, init_mlp_vae_params
from apm.training import TrainConfig


def test_dense_delta_graph_reconstructs_child_params() -> None:
    config = VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,))
    root_params = init_mlp_vae_params(jax.random.PRNGKey(0), config)
    child_params = jax.tree_util.tree_map(lambda leaf: leaf + 0.125, root_params)

    graph = add_dense_delta_node(init_dense_memory_graph(root_params), "node_1_ALL_P0", "root", child_params, "ALL_P0", 1)
    reconstructed_params = effective_params(graph, "node_1_ALL_P0")

    for reconstructed_leaf, child_leaf in zip(
        jax.tree_util.tree_leaves(reconstructed_params),
        jax.tree_util.tree_leaves(child_params),
    ):
        np.testing.assert_allclose(np.asarray(reconstructed_leaf), np.asarray(child_leaf))
    assert edge_memory_stats(graph)[0].delta_l2_norm > 0.0


def test_observed_energy_ignores_label_patch_targets() -> None:
    config = VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,))
    graph = init_dense_memory_graph(init_mlp_vae_params(jax.random.PRNGKey(1), config))
    image = np.eye(28, dtype=np.float32)
    canvases = embed_batch_digits_and_labels(
        np.stack((image, image), axis=0),
        np.asarray([0, 7], dtype=np.int64),
    )

    scores = observed_energy_matrix(graph, canvases, jax.random.PRNGKey(2), TrainConfig(batch_size=2, epochs=1))

    np.testing.assert_allclose(scores[0], scores[1], rtol=1e-6, atol=1e-6)


def test_addressed_evaluation_reports_counts_and_metrics(synthetic_mnist_arrays) -> None:
    config = VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,))
    root_params = init_mlp_vae_params(jax.random.PRNGKey(3), config)
    child_params = jax.tree_util.tree_map(lambda leaf: leaf + 0.01, root_params)
    graph = add_dense_delta_node(init_dense_memory_graph(root_params), "node_1_ALL_P0", "root", child_params, "ALL_P0", 1)
    task = make_permuted_task(synthetic_mnist_arrays, identity_permutation(), "P0")

    result = evaluate_addressed_on_task(
        graph,
        task,
        "node_1_ALL_P0",
        jax.random.PRNGKey(4),
        TrainConfig(batch_size=2, epochs=1),
    )

    assert sum(result.selected_counts.values()) == task.test_labels.shape[0]
    assert len(result.selected_node_ids) == task.test_labels.shape[0]
    assert set(result.candidate_mean_energies) == {"root", "node_1_ALL_P0"}
    assert "energy_classifier_accuracy" in result.metrics


def test_addressed_evaluation_can_use_train_arrays(synthetic_mnist_arrays) -> None:
    config = VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,))
    root_params = init_mlp_vae_params(jax.random.PRNGKey(13), config)
    graph = add_dense_delta_node(
        init_dense_memory_graph(root_params),
        "node_1_ALL_P0",
        "root",
        jax.tree_util.tree_map(lambda leaf: leaf + 0.01, root_params),
        "ALL_P0",
        1,
    )
    task = make_permuted_task(synthetic_mnist_arrays, identity_permutation(), "P0")

    result = evaluate_addressed_on_arrays(
        graph,
        task.train_canvases(),
        task.train_labels,
        "node_1_ALL_P0",
        jax.random.PRNGKey(14),
        TrainConfig(batch_size=2, epochs=1),
    )

    assert sum(result.selected_counts.values()) == task.train_labels.shape[0]
    assert set(result.candidate_mean_energies) == {"root", "node_1_ALL_P0"}
    assert "loss" in result.metrics


def test_memory_graph_svg_contains_node_edge_stats(tmp_path) -> None:
    config = VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,))
    root_params = init_mlp_vae_params(jax.random.PRNGKey(5), config)
    graph = add_dense_delta_node(
        init_dense_memory_graph(root_params),
        "node_1_ALL_P0",
        "root",
        jax.tree_util.tree_map(lambda leaf: leaf + 0.01, root_params),
        "ALL_P0",
        1,
    )
    output_path = tmp_path / "graph.svg"

    write_memory_graph_svg(
        output_path,
        graph,
        {
            "root": NodeVisualStats("root", "root", 0, 128, (), 0.1),
            "node_1_ALL_P0": NodeVisualStats("node_1_ALL_P0", "ALL_P0", 1, 256, ("ALL_P0",), 0.9),
        },
        {("root", "node_1_ALL_P0"): EdgeVisualStats("root", "node_1_ALL_P0", "ALL_P0", 1.25, 256, 0.25)},
        "Memory Graph",
    )

    svg_text = output_path.read_text(encoding="utf-8")
    assert "node_1_ALL_P0" in svg_text
    assert "ALL_P0" in svg_text
    assert "gain +0.250" in svg_text
    assert "<path" in svg_text
    assert 'width="190"' in svg_text
