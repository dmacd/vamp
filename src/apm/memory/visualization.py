"""Dependency-free SVG visualizations for learned memory graphs."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from apm.memory.dense import DenseMemoryGraph, node_ids

TASK_COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ca8a04", "#0891b2")
NODE_WIDTH = 280
NODE_HEIGHT = 112
DEPTH_SPACING = 520
ROW_SPACING = 150
EDGE_LABEL_WIDTH = 190
EDGE_LABEL_HEIGHT = 58


@dataclass(frozen=True)
class NodeVisualStats:
    """Report-facing stats rendered inside a graph node."""

    node_id: str
    trained_task: str
    depth: int
    memory_bytes: int
    eval_wins: tuple[str, ...]
    best_task_accuracy: float


@dataclass(frozen=True)
class EdgeVisualStats:
    """Report-facing stats rendered beside a graph edge."""

    parent_id: str
    child_id: str
    child_task: str
    delta_l2_norm: float
    delta_bytes: int
    eval_gain: float


def write_memory_graph_svg(
    path: Path,
    graph: DenseMemoryGraph,
    node_stats: dict[str, NodeVisualStats],
    edge_stats: dict[tuple[str, str], EdgeVisualStats],
    title: str,
) -> None:
    """Write a labeled SVG visualization of a dense memory graph."""
    positions = _node_positions(graph)
    width = max(x for x, _ in positions.values()) + NODE_WIDTH + 50
    height = max(y for _, y in positions.values()) + NODE_HEIGHT + 46
    edges = "\n".join(
        _edge_markup(
            positions[edge.parent_id],
            positions[edge.child_id],
            edge,
        )
        for edge in edge_stats.values()
    )
    nodes = "\n".join(
        _node_markup(positions[node_id], node_stats[node_id], _task_color_map(node_stats))
        for node_id in node_ids(graph)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                "<style>text{font-family:Inter,Arial,sans-serif;fill:#111827}.muted{fill:#4b5563;font-size:12px}.edge{fill:#374151;font-size:11px}</style>",
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="24" y="28" font-size="18" font-weight="700">{html.escape(title)}</text>',
                edges,
                nodes,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _node_positions(graph: DenseMemoryGraph) -> dict[str, tuple[int, int]]:
    grouped_ids = {
        depth: tuple(node.node_id for node in graph.nodes if node.depth == depth)
        for depth in sorted({node.depth for node in graph.nodes})
    }
    return {
        node_id: (50 + depth * DEPTH_SPACING, 62 + index * ROW_SPACING)
        for depth, ids_at_depth in grouped_ids.items()
        for index, node_id in enumerate(ids_at_depth)
    }


def _node_markup(position: tuple[int, int], stats: NodeVisualStats, task_colors: dict[str, str]) -> str:
    x, y = position
    lines = (
        f"{stats.node_id} | task {stats.trained_task}",
        f"depth {stats.depth} | bytes {_compact_int(stats.memory_bytes)}",
        f"best acc {stats.best_task_accuracy:.3f}",
    )
    line_markup = "\n".join(
        f'<text x="{x + 12}" y="{y + 24 + line_index * 17}" font-size="{13 if line_index == 0 else 12}" font-weight="{650 if line_index == 0 else 400}">{html.escape(line)}</text>'
        for line_index, line in enumerate(lines)
    )
    badge_markup = "\n".join(
        _badge(x + 12 + badge_index * 58, y + 76, task_name, task_colors[task_name])
        for badge_index, task_name in enumerate(stats.eval_wins)
    )
    no_wins = "" if stats.eval_wins else f'<text class="muted" x="{x + 12}" y="{y + 88}">wins: none</text>'
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="6" fill="#f9fafb" stroke="#9ca3af"/>',
            line_markup,
            badge_markup,
            no_wins,
        ]
    )


def _edge_markup(parent_position: tuple[int, int], child_position: tuple[int, int], stats: EdgeVisualStats) -> str:
    parent_x, parent_y = parent_position
    child_x, child_y = child_position
    start_x, start_y = parent_x + NODE_WIDTH + 8, parent_y + NODE_HEIGHT / 2
    end_x, end_y = child_x - 14, child_y + NODE_HEIGHT / 2
    control_x = (start_x + end_x) / 2
    label_x = control_x - EDGE_LABEL_WIDTH / 2
    label_y = _edge_label_y(start_y, end_y)
    labels = (
        f"{stats.child_task}",
        f"||d|| {stats.delta_l2_norm:.2f}",
        f"{_compact_int(stats.delta_bytes)} B | gain {stats.eval_gain:+.3f}",
    )
    return "\n".join(
        [
            f'<path d="M {start_x:.1f} {start_y:.1f} C {control_x:.1f} {start_y:.1f}, {control_x:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}" fill="none" stroke="#6b7280" stroke-width="1.8"/>',
            f'<polygon points="{end_x + 8:.1f},{end_y:.1f} {end_x:.1f},{end_y - 5:.1f} {end_x:.1f},{end_y + 5:.1f}" fill="#6b7280"/>',
            f'<rect x="{label_x:.1f}" y="{label_y:.1f}" width="{EDGE_LABEL_WIDTH}" height="{EDGE_LABEL_HEIGHT}" rx="5" fill="#ffffff" stroke="#d1d5db"/>',
            "\n".join(
                f'<text class="edge" x="{label_x + 10:.1f}" y="{label_y + 18 + line_index * 16:.1f}">{html.escape(label)}</text>'
                for line_index, label in enumerate(labels)
            ),
        ]
    )


def _badge(x: int, y: int, task_name: str, color: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="50" height="20" rx="4" fill="{color}"/>',
            f'<text x="{x + 7}" y="{y + 14}" font-size="11" fill="#ffffff">{html.escape(task_name)}</text>',
        ]
    )


def _task_color_map(node_stats: dict[str, NodeVisualStats]) -> dict[str, str]:
    task_names = tuple(
        task_name
        for task_name in dict.fromkeys(
            task_name
            for stats in node_stats.values()
            for task_name in ((stats.trained_task,) + stats.eval_wins)
            if task_name != "root"
        )
    )
    return {task_name: TASK_COLORS[index % len(TASK_COLORS)] for index, task_name in enumerate(task_names)}


def _compact_int(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _edge_label_y(start_y: float, end_y: float) -> float:
    midpoint = (start_y + end_y) / 2
    if abs(start_y - end_y) < 8:
        return max(38.0, midpoint - EDGE_LABEL_HEIGHT - 20)
    return midpoint - EDGE_LABEL_HEIGHT / 2
