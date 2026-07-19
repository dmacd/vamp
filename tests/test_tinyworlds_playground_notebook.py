from __future__ import annotations

import ast
import json
from pathlib import Path


_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "tinyworlds_playground.ipynb"
)
_REQUIRED_APIS = {
    "exact_kg_summary",
    "generate_tinyworlds_demo",
    "inspect_query",
    "load_tinyworlds_lab",
    "status_html",
    "tinyworlds_playground",
}
_ALLOWED_IMPORT_MODULES = {
    "IPython.display",
    "apm.interactive.tinyworlds",
}
_FORBIDDEN_SYNTAX = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.DictComp,
    ast.For,
    ast.FunctionDef,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.While,
)
_FORBIDDEN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "load_tinystories_artifact",
    "open",
    "read_bytes",
    "read_text",
    "score_frozen_base_candidates",
    "score_hard_node_candidates",
    "urlopen",
}
_FORBIDDEN_LITERAL_FRAGMENTS = (
    "http://",
    "https://",
    "model.npz",
    "results/language_cl",
)


def test_playground_notebook_is_valid_and_completely_unexecuted() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert type(notebook["nbformat_minor"]) is int
    assert notebook["metadata"]["language_info"]["name"] == "python"
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["cells"]
    assert all(
        cell.get("execution_count") is None and cell.get("outputs") == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_playground_notebook_code_compiles_and_has_one_parameters_cell() -> None:
    notebook = _load_notebook()
    code_cells = tuple(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for index, cell in enumerate(code_cells):
        compile(
            _source(cell),
            f"{_NOTEBOOK_PATH.name}:code-cell-{index}",
            "exec",
        )

    parameter_cells = tuple(
        cell
        for cell in code_cells
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    )
    assert len(parameter_cells) == 1
    parameters = ast.parse(_source(parameter_cells[0]))
    assert tuple(
        node.targets[0].id
        for node in parameters.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ) == ("RESULT_DIRECTORY", "PUBLIC_SEED")
    assert all(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        for node in parameters.body
    )


def test_playground_notebook_keeps_mechanism_and_result_caveats_visible() -> None:
    notebook = _load_notebook()
    narrative = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).casefold()

    for required_text in (
        "read-only",
        "stopped",
        "phase 5 was not launched",
        "saved validation evidence",
        "lexical-prior confound",
        "not independent facts or stories",
        "one-hop",
        "one horn-rule application",
        "exact executor",
        "candidate accuracy",
    ):
        assert required_text in narrative


def test_playground_notebook_delegates_parsing_and_computation_to_helpers() -> None:
    notebook = _load_notebook()
    trees = tuple(
        ast.parse(_source(cell))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    imported_names: set[str] = set()
    called_names: set[str] = set()

    for tree in trees:
        assert not any(
            isinstance(node, _FORBIDDEN_SYNTAX)
            for node in ast.walk(tree)
        )
        imports = tuple(
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert all(node.module in _ALLOWED_IMPORT_MODULES for node in imports)
        assert not any(
            isinstance(node, ast.Import)
            for node in ast.walk(tree)
        )
        imported_names.update(
            alias.name
            for node in imports
            for alias in node.names
        )
        called_names.update(
            _call_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        assert not _FORBIDDEN_CALLS.intersection(called_names)
        literals = tuple(
            node.value.casefold()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        )
        assert not any(
            fragment in literal
            for literal in literals
            for fragment in _FORBIDDEN_LITERAL_FRAGMENTS
        )

    assert _REQUIRED_APIS <= imported_names
    assert _REQUIRED_APIS <= called_names


def _load_notebook() -> dict[str, object]:
    value = json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert type(value) is dict
    assert set(value) == {"cells", "metadata", "nbformat", "nbformat_minor"}
    assert type(value["cells"]) is list
    assert all(
        type(cell) is dict
        and cell.get("cell_type") in {"code", "markdown", "raw"}
        and type(cell.get("metadata")) is dict
        and type(cell.get("source")) is list
        for cell in value["cells"]
    )
    return value


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def _call_name(value: ast.expr) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""
