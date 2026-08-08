"""Plotting-library-backed SVG renderers for nouns-v2 reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import io
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


def render_line_chart_svg(
    rows: Sequence[Mapping[str, float]],
    series: Sequence[tuple[str, str]],
    *,
    title: str,
    y_label: str,
    x_label: str = "training stage",
    y_bounds: tuple[float, float] | None = None,
) -> str:
    """Render a deterministic, report-legible line chart with Matplotlib."""
    if len(rows) < 2 or not series:
        raise ValueError("line charts require at least two rows and one series")
    if any("stage" not in row for row in rows) or any(
        key not in row for key, _ in series for row in rows
    ):
        raise ValueError("line chart rows do not contain every requested value")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["svg.hashsalt"] = "tinyworlds-nouns-v2-report"
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter
    except ImportError as error:
        raise ImportError("matplotlib is required for nouns-v2 reports") from error

    stages = np.asarray([row["stage"] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    for key, label in series:
        axis.plot(
            stages,
            np.asarray([row[key] for row in rows], dtype=np.float64),
            marker="o",
            linewidth=2.2,
            markersize=4.2,
            label=label,
        )
    axis.set_title(title, fontsize=15, fontweight="bold", pad=12)
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axis.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10, colors="#334155")
    if y_bounds is not None:
        axis.set_ylim(*y_bounds)
    if y_bounds == (0.0, 1.0):
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=10,
    )
    output = io.StringIO()
    figure.savefig(
        output,
        format="svg",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(figure)
    return _accessible_svg(output.getvalue(), title)


def render_dependency_graph_svg(raw_graph: object) -> str:
    """Render the authenticated VAMP dependency graph with Graphviz."""
    if type(raw_graph) is not list or not raw_graph:
        raise TypeError("dependency rendering requires a nonempty graph row list")
    graph = tuple(_graph_row(row) for row in raw_graph)
    node_ids = tuple(str(row["node"]) for row in graph)
    if len(set(node_ids)) != len(node_ids) or "root" not in node_ids:
        raise ValueError("dependency graph requires unique nodes and one root")
    if any(
        row["parent"] is not None and str(row["parent"]) not in node_ids
        for row in graph
    ):
        raise ValueError("dependency graph contains an unknown parent")
    palette = ("#334155", "#315d9b", "#287271", "#8c5e24", "#744e91", "#a13d4f")
    nodes = tuple(
        (
            f"  {_dot_quote(str(row['node']))} ["
            f"label={_dot_quote(label)},"
            f"fillcolor={_dot_quote(palette[min(int(row['depth']), len(palette) - 1)])},"
            f"tooltip={_dot_quote(_tooltip(row))}];"
        )
        for row in graph
        for label in (f"{int(row['stage']):02d} · {row['node']}",)
    )
    edges = tuple(
        f"  {_dot_quote(str(row['parent']))} -> {_dot_quote(str(row['node']))};"
        for row in graph
        if row["parent"] is not None
    )
    dot = "\n".join(
        (
            "digraph tinyworlds_nouns_v2 {",
            '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", '
            'nodesep="0.28", ranksep="0.6"];',
            '  node [shape="box", style="rounded,filled", '
            'fontname="DejaVu Sans", fontsize="13", fontcolor="white", '
            'margin="0.14,0.08"];',
            '  edge [color="#94a3b8", penwidth="1.8", arrowsize="0.75"];',
            *nodes,
            *edges,
            "}",
            "",
        )
    )
    try:
        rendered = subprocess.run(
            ("dot", "-Tsvg"),
            input=dot,
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("Graphviz 'dot' is required for nouns-v2 reports") from error
    except subprocess.SubprocessError as error:
        raise RuntimeError("Graphviz could not render the nouns-v2 graph") from error
    return _accessible_svg(rendered, "Learned VAMP node dependencies")


def _graph_row(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"depth", "node", "parent", "stage"}:
        raise ValueError("dependency graph row fields changed")
    return value


def _tooltip(row: dict[str, object]) -> str:
    parent = "none" if row["parent"] is None else str(row["parent"])
    return f"stage {int(row['stage'])}: {row['node']}; parent {parent}"


def _dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _accessible_svg(svg: str, title: str) -> str:
    start = svg.find("<svg")
    if start < 0:
        raise ValueError("plot renderer did not produce SVG")
    fragment = svg[start:]
    close = fragment.find(">")
    if close < 0:
        raise ValueError("plot renderer produced malformed SVG")
    opening = fragment[:close]
    if "role=" not in opening:
        opening += f' role="img" aria-label="{escape(title, quote=True)}"'
    return opening + "><title>" + escape(title) + "</title>" + fragment[close + 1 :]


__all__ = ["render_dependency_graph_svg", "render_line_chart_svg"]
