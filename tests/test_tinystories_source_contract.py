from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import runpy
import socket
import subprocess
import sys

import pytest

from apm.lm.tinystories_conversion import (
    PinnedArtifactFile,
    TINYSTORIES_SOURCE,
    TinyStoriesSourceContract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_SCRIPT = REPOSITORY_ROOT / "scripts" / "convert_tinystories_8m.py"


def test_tinystories_source_contract_matches_verified_pinned_bytes() -> None:
    expected_tokenizer_files = (
        PinnedArtifactFile(
            "merges.txt",
            "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            456_318,
        ),
        PinnedArtifactFile(
            "special_tokens_map.json",
            "98412137ae43c77f8af52eb51b19c3536d3242cb55339167d841005fa94a23b7",
            438,
        ),
        PinnedArtifactFile(
            "tokenizer.json",
            "f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5",
            2_107_652,
        ),
        PinnedArtifactFile(
            "tokenizer_config.json",
            "3d76da0fd37493fbfcd3f0fa9757753d31f92e1779ebd9130809b45546a60261",
            722,
        ),
        PinnedArtifactFile(
            "vocab.json",
            "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
            798_156,
        ),
    )

    assert TINYSTORIES_SOURCE == TinyStoriesSourceContract(
        model_id="roneneldan/TinyStories-8M",
        revision="8612e3b15c66ffa94eaa6ee0de5c96edd2d630af",
        transformers_version="4.28.1",
        config_file=PinnedArtifactFile(
            "config.json",
            "5ff16b03beb4466bde520469a815a2d439e16896655d1151c3b44686b387a42d",
            1_161,
        ),
        model_file=PinnedArtifactFile(
            "pytorch_model.bin",
            "22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716",
            112_405_309,
        ),
        tokenizer_files=expected_tokenizer_files,
    )
    with pytest.raises(FrozenInstanceError):
        TINYSTORIES_SOURCE.revision = "mutable"  # type: ignore[misc]


def test_converter_keeps_heavy_imports_behind_main_and_import_is_network_free(
    monkeypatch,
) -> None:
    syntax_tree = ast.parse(CONVERTER_SCRIPT.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(syntax_tree)
        for child in ast.iter_child_nodes(parent)
    }
    heavy_imports = tuple(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _import_roots(node).intersection(
            {"torch", "transformers", "huggingface_hub"}
        )
    )
    imported_roots = {
        root
        for node in heavy_imports
        for root in _import_roots(node)
        if root in {"torch", "transformers", "huggingface_hub"}
    }

    assert imported_roots == {"torch", "transformers", "huggingface_hub"}
    assert all(_has_function_ancestor(node, parents) for node in heavy_imports)
    guards = tuple(
        statement
        for statement in syntax_tree.body
        if isinstance(statement, ast.If) and _is_main_guard(statement)
    )
    assert len(guards) == 1
    assert not guards[0].orelse
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "main"
        for node in ast.walk(guards[0])
    )

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("converter import attempted network access"),
    )
    namespace = runpy.run_path(
        str(CONVERTER_SCRIPT),
        run_name="tinystories_converter_import_contract",
    )
    assert callable(namespace["main"])


def test_ordinary_lm_imports_do_not_load_torch_or_transformers() -> None:
    source_root = REPOSITORY_ROOT / "src"
    program = f"""
import importlib
import sys
sys.path.insert(0, {str(source_root)!r})
heavy_roots = {{'torch', 'transformers'}}
before = set(sys.modules)
for module_name in (
    'apm.lm',
    'apm.lm.tinystories_conversion',
    'apm.lm.parity',
    'apm.lm.generation',
):
    importlib.import_module(module_name)
added = set(sys.modules) - before
unexpected = sorted(
    name for name in added if name.partition('.')[0] in heavy_roots
)
if unexpected:
    raise SystemExit('unexpected heavy imports: ' + repr(unexpected))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _import_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    return {node.module.partition(".")[0]} if node.module is not None else set()


def _has_function_ancestor(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        current = parents.get(current)
    return False


def _is_main_guard(statement: ast.If) -> bool:
    test = statement.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
