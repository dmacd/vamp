from __future__ import annotations

import ast
from pathlib import Path
import runpy
import socket
from types import SimpleNamespace

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_tinystories_postmortem.py"


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_postmortem_runner_import_is_offline_and_does_not_start_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class NetworkDeniedSocket(socket.socket):
        def connect(self, address) -> None:
            pytest.fail(f"runner import attempted network access: {address}")

        def connect_ex(self, address) -> int:
            pytest.fail(f"runner import attempted network access: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)
    monkeypatch.chdir(tmp_path)

    namespace = runpy.run_path(str(RUNNER), run_name="offline_postmortem_surface")

    assert callable(namespace["main"])
    assert not tuple(tmp_path.iterdir())


def test_postmortem_runner_wires_one_retrain_and_evaluation_only_reload() -> None:
    syntax_tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = tuple(
        name
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        for name in (_call_name(node),)
        if name is not None
    )
    required_calls = {
        "build_language_report_bundle",
        "build_tinystories_postmortem_suite",
        "evaluate_language_benchmark",
        "extract_language_adaptation_artifact",
        "load_complete_classified_tinystories_test_half",
        "load_language_adaptation_artifact",
        "load_tinystories_artifact",
        "load_tinystories_topic_dataset",
        "prepare_language_curriculum",
        "run_language_benchmark",
        "save_language_adaptation_artifact",
    }

    assert required_calls.issubset(calls)
    assert calls.count("run_language_benchmark") == 1
    assert calls.count("evaluate_language_benchmark") == 1
    assert calls.count("write_language_report") == 2
    assert "argparse" not in {
        alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in syntax_tree.body
    )


def test_postmortem_main_announces_temporary_directory_then_runs_fixed_phases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = runpy.run_path(str(RUNNER), run_name="postmortem_orchestration")
    calls: list[str | int] = []

    class FakeProgress:
        def __init__(self, temporary_directory: Path) -> None:
            calls.append(f"temporary:{temporary_directory}")

        def __enter__(self) -> FakeProgress:
            return self

        def __exit__(self, *exception: object) -> None:
            return None

        def run(self, phase, operation):
            calls.append(phase.number)
            return operation()

    inputs = object()
    benchmark = object()
    artifact = SimpleNamespace(tensor_checksum="a" * 64)
    persisted = SimpleNamespace(
        directory=tmp_path / "adaptation",
        identity="b" * 64,
        artifact=artifact,
    )
    evaluation_inputs = SimpleNamespace(
        suite=SimpleNamespace(suite_id="tinystories-topic-postmortem-v1")
    )
    evaluation = SimpleNamespace(
        tensor_checksums_before=(("tensor", "c" * 64),),
        tensor_checksums_after=(("tensor", "c" * 64),),
    )
    report = SimpleNamespace(
        directory=tmp_path / "report",
        file_count=3,
        tree_checksum="d" * 64,
    )
    runner_globals = namespace["main"].__globals__
    runner_globals.update(
        {
            "_PostmortemProgress": FakeProgress,
            "_snapshot_historical_report": lambda: {"manifest.json": b"old"},
            "_prepare_training_inputs": lambda: inputs,
            "_run_exact_benchmark": lambda value: benchmark,
            "_persist_adaptation": lambda value, result: persisted,
            "_build_evaluation_inputs": lambda value: evaluation_inputs,
            "_evaluate_loaded_artifact": (
                lambda value, loaded, suite: evaluation
            ),
            "_write_verified_report": (
                lambda old, value, result, loaded, suite, measured: report
            ),
            "_write_json": lambda path, payload: calls.append("result"),
            "tempfile": SimpleNamespace(
                mkdtemp=lambda prefix: str(tmp_path / "temporary")
            ),
        }
    )

    namespace["main"]()

    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0] == f"Temporary artifact directory: {tmp_path / 'temporary'}"
    assert calls[1:8] == list(range(1, 8))
    assert calls[-1] == "result"
    assert output_lines[-2:] == [
        f"Adaptation artifact: {persisted.directory}",
        f"Post-mortem report: {report.directory}",
    ]
