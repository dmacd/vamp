from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
import socket
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TINY_SHAKESPEARE_RUNNER = REPOSITORY_ROOT / "scripts" / "run_tinyshakespeare_cl.py"
TINYSTORIES_PREPARER = REPOSITORY_ROOT / "scripts" / "prepare_tinystories_v2.py"
TINYSTORIES_RUNNER = REPOSITORY_ROOT / "scripts" / "run_tinystories_cl.py"
TINY_SHAKESPEARE_NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "tinyshakespeare_vamp.ipynb"
)
TINYSTORIES_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "tinystories_vamp.ipynb"
LANGUAGE_REPORT_ARTIFACTS = (
    "manifest.json",
    "stage_metrics.jsonl",
    "stored_competence.jsonl",
    "routing_metrics.jsonl",
    "transfer_metrics.jsonl",
    "memory_metrics.jsonl",
    "addressing_cost.jsonl",
    "address_confusion.svg",
    "competence_curves.svg",
    "routing_curves.svg",
    "memory_curves.svg",
    "graph.svg",
    "samples.md",
    "report.html",
)


@pytest.mark.parametrize(
    "script_path",
    (TINY_SHAKESPEARE_RUNNER, TINYSTORIES_PREPARER, TINYSTORIES_RUNNER),
)
def test_phase10_script_import_is_offline_and_does_not_start_work(
    script_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NetworkDeniedSocket(socket.socket):
        def connect(self, address) -> None:
            pytest.fail(f"script import attempted network access: {address}")

        def connect_ex(self, address) -> int:
            pytest.fail(f"script import attempted network access: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)
    modules_before_import = set(sys.modules)

    namespace = runpy.run_path(
        str(script_path),
        run_name=f"offline_surface_{script_path.stem}",
    )

    assert callable(namespace["main"])
    newly_imported_roots = {
        module_name.partition(".")[0]
        for module_name in set(sys.modules).difference(modules_before_import)
    }
    assert newly_imported_roots.isdisjoint(
        {"huggingface_hub", "tokenizers", "torch", "transformers"}
    )


@pytest.mark.parametrize(
    ("script_path", "required_calls", "manifest_identity"),
    (
        (
            TINY_SHAKESPEARE_RUNNER,
            {
                "load_tiny_shakespeare",
                "split_text_contiguously",
                "from_training_text",
                "build_tiny_shakespeare_permutation_curriculum",
                "build_tiny_shakespeare_region_curriculum",
                "build_tiny_shakespeare_stable_hash_curriculum",
                "prepare_language_curriculum",
                "raw_tasks_from_corpus_curriculum",
                "load_gpt_neo_checkpoint",
                "run_language_benchmark",
                "write_language_benchmark_report",
            },
            ("tinyshakespeare", "character-permutation", "standard"),
        ),
        (
            TINYSTORIES_RUNNER,
            {
                "load_tinystories_topic_dataset",
                "load_tinystories_artifact",
                "from_file",
                "prepare_language_curriculum",
                "run_language_benchmark",
                "write_language_benchmark_report",
            },
            ("tinystories-v2-gpt4", "topic", "single-gpu"),
        ),
    ),
)
def test_phase10_runners_wire_canonical_library_pipeline_and_report_identity(
    script_path: Path,
    required_calls: set[str],
    manifest_identity: tuple[str, str, str],
) -> None:
    syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
    main_function = _single_main_function(syntax_tree)
    call_names = {
        name
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        for name in (_call_name(node),)
        if name is not None
    }

    assert required_calls.issubset(call_names)
    assert _has_main_guard(syntax_tree)
    imports_argparse = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "argparse" in _import_roots(node)
        for node in ast.walk(syntax_tree)
    )
    assert imports_argparse is (script_path == TINY_SHAKESPEARE_RUNNER)
    manifest_call = next(
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and _call_name(node) == "LanguageReportManifest"
    )
    keyword_nodes = {
        keyword.arg: keyword.value
        for keyword in manifest_call.keywords
        if keyword.arg in {"dataset", "curriculum", "preset"}
    }
    assert ast.literal_eval(keyword_nodes["dataset"]) == manifest_identity[0]
    assert ast.literal_eval(keyword_nodes["preset"]) == manifest_identity[2]
    if script_path == TINY_SHAKESPEARE_RUNNER:
        assert isinstance(keyword_nodes["curriculum"], ast.Name)
        assert keyword_nodes["curriculum"].id == "curriculum_name"
    else:
        assert ast.literal_eval(keyword_nodes["curriculum"]) == manifest_identity[1]


def test_tinyshakespeare_cli_has_only_the_three_canonical_curricula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(
        str(TINY_SHAKESPEARE_RUNNER),
        run_name="tinyshakespeare_curriculum_cli_contract",
    )
    choices = (
        "character-permutation",
        "corpus-region",
        "stable-hash",
    )

    assert namespace["CURRICULUM_CHOICES"] == choices
    for choice in choices:
        monkeypatch.setattr(
            sys,
            "argv",
            [str(TINY_SHAKESPEARE_RUNNER), "--curriculum", choice],
        )
        assert namespace["_parse_args"]().curriculum == choice
    monkeypatch.setattr(sys, "argv", [str(TINY_SHAKESPEARE_RUNNER)])
    assert namespace["_parse_args"]().curriculum == "character-permutation"
    source = TINY_SHAKESPEARE_RUNNER.read_text(encoding="utf-8")
    assert "negative_control_curriculum=curriculum_name == \"stable-hash\"" in source
    assert '"name": curriculum_name' in source


def test_tinystories_preparation_is_explicit_pinned_and_lazy() -> None:
    syntax_tree = ast.parse(TINYSTORIES_PREPARER.read_text(encoding="utf-8"))
    main_function = _single_main_function(syntax_tree)
    optional_imports = tuple(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "huggingface_hub" in _import_roots(node)
    )

    assert len(optional_imports) == 1
    assert optional_imports[0] in tuple(ast.walk(main_function))
    snapshot_call = next(
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call) and _call_name(node) == "snapshot_download"
    )
    assert {keyword.arg for keyword in snapshot_call.keywords} == {
        "repo_id",
        "repo_type",
        "revision",
        "allow_patterns",
        "local_dir",
    }
    call_names = {
        _call_name(node)
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
    }
    assert "verify_pinned_dataset_file" in call_names
    assert _has_main_guard(syntax_tree)


@pytest.mark.parametrize(
    ("script_path", "required_config_keys"),
    (
        (
            TINY_SHAKESPEARE_RUNNER,
            {
                "adapter",
                "base_checkpoint",
                "benchmark",
                "curriculum",
                "data",
                "dataset_revision",
                "model",
                "optimizer",
                "tokenizer",
            },
        ),
        (
            TINYSTORIES_RUNNER,
            {
                "adapter",
                "base_checkpoint",
                "benchmark",
                "checkpoint_source",
                "curriculum",
                "data",
                "dataset_source",
                "model",
                "optimizer",
                "preset",
                "tokenizer",
                "topic_lexicons",
                "topic_rule",
            },
        ),
    ),
)
def test_runner_config_hash_covers_execution_identity(
    script_path: Path,
    required_config_keys: set[str],
) -> None:
    syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
    main_function = _single_main_function(syntax_tree)
    canonical_call = next(
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call) and _call_name(node) == "canonical_config_json"
    )
    config_dictionary = canonical_call.args[0]

    assert isinstance(config_dictionary, ast.Dict)
    assert {
        key.value
        for key in config_dictionary.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    } == required_config_keys
    source_text = script_path.read_text(encoding="utf-8")
    assert "asdict(settings)" in source_text
    assert "asdict(lora_config)" in source_text
    if script_path == TINYSTORIES_RUNNER:
        assert "evaluation_microbatch_size=8" in source_text
        assert "peak_device_memory_target_bytes" in source_text
        assert "preset.peak_device_memory_gib * 1024**3" in source_text


@pytest.mark.parametrize(
    (
        "notebook_path",
        "required_markdown",
        "required_code",
        "report_path",
    ),
    (
        (
            TINY_SHAKESPEARE_NOTEBOOK,
            ("scripts/run_tinyshakespeare_cl.py",),
            (
                "CANONICAL_BASELINE_MATRIX",
                "TINY_SHAKESPEARE_EVALUATION_PRESET",
                "character-permutation",
                "corpus-region",
                "stable-hash",
            ),
            "results/language_cl/tinyshakespeare/character-permutation/"
            "standard-seed0-<config_hash>/",
        ),
        (
            TINYSTORIES_NOTEBOOK,
            (
                "scripts/prepare_tinystories_v2.py",
                "scripts/run_tinystories_cl.py",
            ),
            (
                "CANONICAL_BASELINE_MATRIX",
                "TINYSTORIES_SINGLE_GPU_PRESET",
                "TINYSTORIES_TOPICS",
                "TINYSTORIES_V2_SOURCE",
            ),
            "results/language_cl/tinystories-v2-gpt4/topic/"
            "single-gpu-seed0-<config_hash>/",
        ),
    ),
)
def test_phase10_notebooks_are_unexecuted_offline_contract_surfaces(
    notebook_path: Path,
    required_markdown: tuple[str, ...],
    required_code: tuple[str, ...],
    report_path: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = tuple(
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    code_sources = tuple("".join(cell["source"]) for cell in code_cells)
    combined_code = "\n".join(code_sources)
    combined_markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )

    assert code_cells
    assert all(
        cell["execution_count"] is None and not cell["outputs"]
        for cell in code_cells
    )
    assert all(
        not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While),
        )
        for source in code_sources
        for node in ast.walk(ast.parse(source))
    )
    for required_surface in required_code:
        assert required_surface in combined_code
    assert all(
        forbidden_dependency not in combined_code
        for forbidden_dependency in (
            "huggingface_hub",
            "snapshot_download",
            "prepare_tiny_shakespeare(",
            "load_tiny_shakespeare(",
            "load_tinystories_artifact(",
            "TokenizersTextTokenizer.from_file(",
            "run_language_benchmark(",
            "run_base_updates(",
            "torch",
            "transformers",
        )
    )
    for markdown_surface in required_markdown:
        assert markdown_surface in combined_markdown
    assignments = {
        target.id: ast.literal_eval(statement.value)
        for source in code_sources
        for statement in ast.parse(source).body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id in {"report_artifacts", "report_path_template"}
    }
    assert assignments["report_artifacts"] == LANGUAGE_REPORT_ARTIFACTS
    assert assignments["report_path_template"] == report_path
    assert report_path in combined_markdown

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: pytest.fail("notebook attempted network access"),
    )
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell_index, source in enumerate(code_sources):
        exec(
            compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec"),
            namespace,
        )
    assert not tuple(tmp_path.iterdir())


def _single_main_function(syntax_tree: ast.Module) -> ast.FunctionDef:
    main_functions = tuple(
        statement
        for statement in syntax_tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "main"
    )
    assert len(main_functions) == 1
    return main_functions[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _has_main_guard(syntax_tree: ast.Module) -> bool:
    return any(
        isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == "__name__"
        and len(statement.test.comparators) == 1
        and isinstance(statement.test.comparators[0], ast.Constant)
        and statement.test.comparators[0].value == "__main__"
        for statement in syntax_tree.body
    )


def _import_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    return {node.module.partition(".")[0]} if node.module is not None else set()
