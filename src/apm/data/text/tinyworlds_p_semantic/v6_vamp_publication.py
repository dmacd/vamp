"""One-transaction sealed evaluation and publication for semantic-v6 VAMP."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile

from apm.continual.language_evaluation_run import LanguageConditionMeasurement
from apm.continual.language_evaluation import LanguageEvaluationSuite
from apm.continual.language_tasks import TaskId
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evaluation import semantic_validation_record
from apm.data.text.tinyworlds_p_semantic.v6_evaluation import (
    V6EvaluationProgress,
    V6SemanticSealedTest,
    evaluate_v6_sealed_test_once,
    load_v6_sealed_test,
)
from apm.data.text.tinyworlds_p_semantic.v6_milestone import (
    V6SelectedBase,
    load_v6_selected_base,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6SemanticPartitionArtifact,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
    V6VampExperimentPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_curriculum import (
    prepare_v6_vamp_test_suite,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_metrics import (
    V6VampPosthocMetrics,
    measure_v6_vamp_posthoc,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_specificity import (
    V6SpecificityAudit,
    V6SpecificityProgress,
    evaluate_v6_adapter_specificity,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_training import (
    V6VampAdaptationPublication,
    load_v6_vamp_adaptation_publication,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.text import TextTokenizer


V6_VAMP_RESULT_FORMAT = "tinyworlds-p-semantic-v6-vamp-result"
V6_VAMP_RESULT_TREE_FORMAT = "tinyworlds-p-semantic-v6-vamp-result-tree"
V6VampPhaseProgress = Callable[[str], None]
V6VampConditionProgress = Callable[[int, TaskId, str, int, int], None]


@dataclass(frozen=True, slots=True)
class V6VampResultPublication:
    """The immutable exploratory result after the sole sealed-test transaction."""

    directory: Path
    result_sha256: str
    partition_sha256: str
    selected_base_sha256: str
    adaptation_run_sha256: str
    config_sha256: str
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        for value in (
            self.result_sha256,
            self.partition_sha256,
            self.selected_base_sha256,
            self.adaptation_run_sha256,
            self.config_sha256,
        ):
            _require_sha256(value)
        if (
            type(self.allocator_peak_bytes) is not int
            or self.allocator_peak_bytes < 0
        ):
            raise ValueError("semantic-v6 VAMP result allocator peak is invalid")
        if (
            self.allocator_peak_bytes
            > V6_VAMP_EXPERIMENT_PRESET.allocator_peak_limit_bytes
        ):
            raise ValueError("semantic-v6 VAMP result allocator peak exceeds its limit")


def begin_v6_vamp_sealed_transaction(
    artifact: V6SemanticPartitionArtifact,
    selected_base: V6SelectedBase,
    adaptations: V6VampAdaptationPublication,
    transaction_directory: str | Path,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
) -> Path:
    """Durably authorize the sole test transaction after all training is frozen."""
    selected = load_v6_selected_base(selected_base.directory)
    trained = load_v6_vamp_adaptation_publication(adaptations.directory)
    if (
        type(artifact) is not V6SemanticPartitionArtifact
        or type(preset) is not V6VampExperimentPreset
        or artifact.partition_sha256 != preset.partition_sha256
        or selected.partition_sha256 != preset.partition_sha256
        or trained.partition_sha256 != preset.partition_sha256
        or trained.selected_base_sha256 != selected.selection_sha256
        or trained.config_sha256 != preset.config_sha256
    ):
        raise ValueError("semantic-v6 sealed transaction source identity changed")
    transaction = Path(transaction_directory)
    transaction.mkdir(parents=True, exist_ok=True)
    _open_or_validate_transaction(
        transaction / "sealed-transaction.json",
        _transaction_binding(artifact, selected, trained, preset),
    )
    return transaction.resolve()


def run_or_resume_v6_vamp_sealed_evaluation(
    artifact: V6SemanticPartitionArtifact,
    selected_base: V6SelectedBase,
    adaptations: V6VampAdaptationPublication,
    tokenizer: TextTokenizer,
    transaction_directory: str | Path,
    publication_root: str | Path,
    preset: V6VampExperimentPreset = V6_VAMP_EXPERIMENT_PRESET,
    *,
    phase_progress: V6VampPhaseProgress | None = None,
    evaluation_progress: V6EvaluationProgress | None = None,
    specificity_progress: V6SpecificityProgress | None = None,
    condition_progress: V6VampConditionProgress | None = None,
) -> V6VampResultPublication:
    """Open test data in one durable transaction and publish all frozen analyses."""
    selected = load_v6_selected_base(selected_base.directory)
    trained = load_v6_vamp_adaptation_publication(adaptations.directory)
    _require_inputs(artifact, selected, trained, tokenizer, preset)
    transaction = begin_v6_vamp_sealed_transaction(
        artifact,
        selected,
        trained,
        transaction_directory,
        preset,
    )
    completed_path = transaction / "completed.json"
    if completed_path.exists():
        completed = _load_json(completed_path)
        return load_v6_vamp_result(
            Path(publication_root) / _text(completed, "result_sha256")
        )

    checkpoint = load_gpt_neo_checkpoint(selected.checkpoint)
    _phase(phase_progress, "Evaluating the selected base on sealed test data.")
    base_directory = transaction / "base-sealed-test"
    if base_directory.exists() and not (base_directory / "tree.json").is_file():
        _quarantine_incomplete_directory(base_directory, transaction / "recovery")
    base_test = (
        load_v6_sealed_test(
            base_directory,
            artifact,
            selected.selected_epoch,
        )
        if (base_directory / "tree.json").is_file()
        else evaluate_v6_sealed_test_once(
            checkpoint.params,
            artifact,
            selected.selected_epoch,
            base_directory,
            checkpoint.config,
            replicates=preset.specificity_replicates,
            progress=evaluation_progress,
        )
    )

    _phase(phase_progress, "Building the fixed nested-prefix test suite.")
    suite = prepare_v6_vamp_test_suite(artifact, tokenizer, preset)
    measurements_path = _next_attempt_path(transaction, "measurements", ".jsonl")
    _phase(phase_progress, "Evaluating all nine stored and routed methods.")
    with measurements_path.open("wb") as measurement_output:
        metrics = measure_v6_vamp_posthoc(
            trained.adaptation,
            suite,
            checkpoint.params,
            preset,
            measurement_sink=lambda row: measurement_output.write(
                canonical_json_bytes(_measurement_record(row))
            ),
            condition_progress=condition_progress,
            phase_sink=lambda message: _phase(phase_progress, message),
        )
        measurement_output.flush()
        os.fsync(measurement_output.fileno())

    _phase(phase_progress, "Auditing adapter specificity on paired controls.")
    specificity = evaluate_v6_adapter_specificity(
        artifact,
        trained.adaptation,
        checkpoint.params,
        base_test.directory,
        transaction / "specificity",
        preset,
        progress=specificity_progress,
    )
    peak = max(
        allocator_peak_bytes(),
        trained.allocator_peak_bytes,
        base_test.validation.allocator_peak_bytes,
    )
    if peak > preset.allocator_peak_limit_bytes:
        raise MemoryError(
            f"semantic-v6 VAMP peak {peak:,} exceeds "
            f"{preset.allocator_peak_limit_bytes:,} bytes"
        )
    result = _publish_result(
        artifact,
        selected,
        trained,
        base_test,
        suite,
        metrics,
        specificity,
        peak,
        measurements_path,
        Path(publication_root),
        preset,
    )
    _write_json(completed_path, {"result_sha256": result.result_sha256})
    return result


def load_v6_vamp_result(directory: str | Path) -> V6VampResultPublication:
    """Authenticate every file in a published semantic-v6 VAMP result."""
    root = Path(directory)
    tree = _load_json(root / "tree.json")
    if (
        tree.get("format") != V6_VAMP_RESULT_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("result_sha256") != root.name
    ):
        raise ValueError("semantic-v6 VAMP result tree changed")
    descriptors = tree.get("files")
    if type(descriptors) is not list or any(type(item) is not dict for item in descriptors):
        raise ValueError("semantic-v6 VAMP result descriptors changed")
    actual_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    described_paths = tuple(_text(item, "relative_path") for item in descriptors)
    if described_paths != actual_paths:
        raise ValueError("semantic-v6 VAMP result file set changed")
    for descriptor in descriptors:
        path = root / _text(descriptor, "relative_path")
        if (
            path.is_symlink()
            or path.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(path) != _text(descriptor, "sha256")
        ):
            raise ValueError(f"semantic-v6 VAMP result file changed: {path}")
    manifest = _load_json(root / "manifest.json")
    required = {
        "adaptation_run_sha256",
        "allocator_peak_bytes",
        "config",
        "config_sha256",
        "exploratory",
        "format",
        "base_sealed_tree_sha256",
        "measurements_sha256",
        "metrics_sha256",
        "partition_sha256",
        "result_sha256",
        "sealed_test_opened",
        "selected_base_sha256",
        "specificity_identity_sha256",
        "suite_id",
        "suite_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != V6_VAMP_RESULT_FORMAT
        or manifest.get("result_sha256") != root.name
        or manifest.get("sealed_test_opened") is not True
        or manifest.get("exploratory") is not True
        or manifest.get("config") != V6_VAMP_EXPERIMENT_PRESET.as_record()
        or manifest.get("config_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.config_sha256
        or manifest.get("partition_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.partition_sha256
    ):
        raise ValueError("semantic-v6 VAMP result manifest changed")
    content = {key: value for key, value in manifest.items() if key != "result_sha256"}
    if record_sha256(content) != root.name:
        raise ValueError("semantic-v6 VAMP result identity changed")
    metrics_record = _load_json(root / "metrics.json")
    manifest_peak = _integer(manifest, "allocator_peak_bytes")
    if (
        _file_sha256(root / "base-sealed-test" / "tree.json")
        != _text(manifest, "base_sealed_tree_sha256")
        or _file_sha256(root / "measurements.jsonl")
        != _text(manifest, "measurements_sha256")
        or record_sha256(metrics_record) != _text(manifest, "metrics_sha256")
        or _integer(metrics_record, "allocator_peak_bytes") != manifest_peak
        or record_sha256(_load_json(root / "test-suite.json"))
        != _text(manifest, "suite_sha256")
        or _text(
            _load_json(root / "specificity" / "specificity.json"),
            "identity_sha256",
        )
        != _text(manifest, "specificity_identity_sha256")
    ):
        raise ValueError("semantic-v6 VAMP result evidence binding changed")
    return V6VampResultPublication(
        directory=root.resolve(),
        result_sha256=root.name,
        partition_sha256=_text(manifest, "partition_sha256"),
        selected_base_sha256=_text(manifest, "selected_base_sha256"),
        adaptation_run_sha256=_text(manifest, "adaptation_run_sha256"),
        config_sha256=_text(manifest, "config_sha256"),
        allocator_peak_bytes=manifest_peak,
    )


def _publish_result(
    artifact: V6SemanticPartitionArtifact,
    selected: V6SelectedBase,
    trained: V6VampAdaptationPublication,
    base_test: V6SemanticSealedTest,
    suite: LanguageEvaluationSuite,
    metrics: V6VampPosthocMetrics,
    specificity: V6SpecificityAudit,
    peak: int,
    measurements_path: Path,
    publication_root: Path,
    preset: V6VampExperimentPreset,
) -> V6VampResultPublication:
    metrics_record = _metrics_record(metrics, specificity, base_test, peak)
    suite_record = _suite_record(suite)
    content = {
        "adaptation_run_sha256": trained.run_sha256,
        "allocator_peak_bytes": peak,
        "base_sealed_tree_sha256": _file_sha256(
            base_test.directory / "tree.json"
        ),
        "config": preset.as_record(),
        "config_sha256": preset.config_sha256,
        "exploratory": True,
        "format": V6_VAMP_RESULT_FORMAT,
        "measurements_sha256": _file_sha256(measurements_path),
        "metrics_sha256": record_sha256(metrics_record),
        "partition_sha256": artifact.partition_sha256,
        "sealed_test_opened": True,
        "selected_base_sha256": selected.selection_sha256,
        "specificity_identity_sha256": specificity.identity_sha256,
        "suite_id": suite.suite_id,
        "suite_sha256": record_sha256(suite_record),
    }
    result_sha256 = record_sha256(content)
    publication_root.mkdir(parents=True, exist_ok=True)
    target = publication_root / result_sha256
    if target.exists():
        return load_v6_vamp_result(target)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-semantic-v6-result-", dir=work_root))
    try:
        shutil.copytree(base_test.directory, staging / "base-sealed-test")
        shutil.copytree(specificity.directory, staging / "specificity")
        shutil.copyfile(measurements_path, staging / "measurements.jsonl")
        _write_json(staging / "test-suite.json", suite_record)
        _write_json(staging / "metrics.json", metrics_record)
        _write_json(
            staging / "manifest.json",
            {**content, "result_sha256": result_sha256},
        )
        markdown = _markdown_report(result_sha256, selected, metrics_record)
        _write_text(staging / "report.md", markdown)
        _write_text(staging / "report.html", _html_report(markdown))
        _write_tree(staging, result_sha256)
        os.rename(staging, target)
        _fsync_directory(publication_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_v6_vamp_result(target)


def _metrics_record(
    metrics: V6VampPosthocMetrics,
    specificity: V6SpecificityAudit,
    base_test: V6SemanticSealedTest,
    peak: int,
) -> dict[str, object]:
    return {
        "allocator_peak_bytes": peak,
        "base_semantic_test": semantic_validation_record(base_test.validation),
        "forgetting": [asdict(item) for item in metrics.forgetting],
        "memory": asdict(metrics.memory),
        "routing_timing": [asdict(item) for item in metrics.routing_timing],
        "specificity": [asdict(item) for item in specificity.results],
        "specificity_identity_sha256": specificity.identity_sha256,
        "transfer": [asdict(item) for item in metrics.transfer],
    }


def _measurement_record(row: LanguageConditionMeasurement) -> dict[str, object]:
    return {
        name: str(value) if name == "task_id" else value
        for name, value in asdict(row).items()
    }


def _suite_record(suite: LanguageEvaluationSuite) -> dict[str, object]:
    return {
        "benchmark_label": suite.benchmark_label,
        "conditions": [asdict(condition) for condition in suite.conditions],
        "examples": [
            {
                "condition_id": item.condition_id,
                "cue_regime": item.cue_regime,
                "pair_id": item.pair_id,
                "provenance": asdict(item.provenance),
                "task_id": str(item.task_id),
                "visible_concept_ids": list(item.visible_concept_ids),
            }
            for item in suite.examples
        ],
        "primary_condition_id": suite.primary_condition_id,
        "suite_id": suite.suite_id,
    }


def _markdown_report(
    result_sha256: str,
    selected: V6SelectedBase,
    metrics: Mapping[str, object],
) -> str:
    base = metrics["base_semantic_test"]
    assert isinstance(base, dict)
    mean = base["mean"]
    assert isinstance(mean, dict)
    forgetting = metrics["forgetting"]
    specificity = metrics["specificity"]
    assert isinstance(forgetting, list) and isinstance(specificity, list)
    worst_forgetting = max(
        (float(row["forgetting_from_best"]) for row in forgetting),
        default=0.0,
    )
    mean_specificity = sum(float(row["specificity"]) for row in specificity) / len(
        specificity
    )
    return (
        "# TinyWorlds-P Semantic-v6 VAMP Result\n\n"
        f"Result SHA-256: `{result_sha256}`\n\n"
        f"The validation-selected base came from epoch {selected.selected_epoch}. "
        "The sealed test was opened once, after base selection and all adapter "
        "training were frozen. This VAMP comparison is exploratory and has no "
        "pass/fail threshold.\n\n"
        "## Base semantic test\n\n"
        f"The mean world-minus-control gap was {float(mean['observed_gap']):.6f} "
        f"nats/token, with a 95% paired-bootstrap interval of "
        f"[{float(mean['bootstrap_lower']):.6f}, "
        f"{float(mean['bootstrap_upper']):.6f}] and a one-sided label-swap "
        f"probability of {float(mean['placebo_probability']):.6f}.\n\n"
        "## Continual adaptation summary\n\n"
        f"Across the primary 64-token prefix condition, the largest measured "
        f"final-stage forgetting from a method's prior best was "
        f"{worst_forgetting:.6f} nats/token. The unweighted mean forced-adapter "
        f"specificity across methods, worlds, and both control arms was "
        f"{mean_specificity:.6f} nats/token.\n\n"
        "The complete nine-method matrix, cue strata, prefix lengths, transfer "
        "curves, memory accounting, routing timings, and paired specificity "
        "intervals are stored in `measurements.jsonl` and `metrics.json`.\n"
    )


def _html_report(markdown: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>TinyWorlds-P Semantic-v6 VAMP Result</title>"
        "<style>body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;"
        "margin:2rem auto;padding:0 1rem;color:#172033}pre{white-space:pre-wrap;"
        "background:#f4f6fa;padding:1rem;border-radius:.4rem}code{font-family:"
        "ui-monospace,monospace}</style></head><body><pre>"
        f"{escape(markdown)}</pre></body></html>"
    )


def _transaction_binding(
    artifact: V6SemanticPartitionArtifact,
    selected: V6SelectedBase,
    trained: V6VampAdaptationPublication,
    preset: V6VampExperimentPreset,
) -> dict[str, object]:
    return {
        "adaptation_run_sha256": trained.run_sha256,
        "config_sha256": preset.config_sha256,
        "format": "tinyworlds-p-semantic-v6-sealed-transaction",
        "partition_sha256": artifact.partition_sha256,
        "selected_base_sha256": selected.selection_sha256,
    }


def _open_or_validate_transaction(path: Path, binding: Mapping[str, object]) -> None:
    if path.exists():
        if _load_json(path) != dict(binding):
            raise ValueError("semantic-v6 sealed transaction binding changed")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(dict(binding)))
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if _load_json(path) != dict(binding):
                raise ValueError("semantic-v6 sealed transaction binding changed")
        _fsync_directory(path.parent)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _require_inputs(
    artifact: V6SemanticPartitionArtifact,
    selected: V6SelectedBase,
    trained: V6VampAdaptationPublication,
    tokenizer: TextTokenizer,
    preset: V6VampExperimentPreset,
) -> None:
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 VAMP result requires its strict partition")
    if type(preset) is not V6VampExperimentPreset:
        raise TypeError("semantic-v6 VAMP result requires its strict preset")
    if (
        artifact.partition_sha256 != preset.partition_sha256
        or selected.partition_sha256 != preset.partition_sha256
        or trained.partition_sha256 != preset.partition_sha256
        or trained.selected_base_sha256 != selected.selection_sha256
        or trained.config_sha256 != preset.config_sha256
        or tokenizer.vocab_size != artifact.tokenizer_identity.vocab_size
        or tokenizer.pad_token_id != artifact.pad_token_id
        or tokenizer.eos_token_id != artifact.eos_token_id
    ):
        raise ValueError("semantic-v6 VAMP result source identity changed")


def _next_attempt_path(root: Path, stem: str, suffix: str) -> Path:
    index = 1
    while (candidate := root / f"{stem}-attempt-{index:02d}{suffix}").exists():
        index += 1
    return candidate


def _quarantine_incomplete_directory(directory: Path, recovery_root: Path) -> None:
    recovery_root.mkdir(exist_ok=True)
    index = 1
    while (
        target := recovery_root / f"{directory.name}-incomplete-{index:02d}"
    ).exists():
        index += 1
    os.rename(directory, target)
    _fsync_directory(directory.parent)
    _fsync_directory(recovery_root)


def _phase(progress: V6VampPhaseProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _write_tree(root: Path, result_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": V6_VAMP_RESULT_TREE_FORMAT,
            "result_sha256": result_sha256,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid semantic-v6 VAMP result JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"noncanonical semantic-v6 VAMP result JSON: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"semantic-v6 VAMP result field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"semantic-v6 VAMP result field {field!r} must be nonnegative")
    return value


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("semantic-v6 VAMP result identity must be SHA-256")


__all__ = [
    "V6VampPhaseProgress",
    "V6VampConditionProgress",
    "V6VampResultPublication",
    "V6_VAMP_RESULT_FORMAT",
    "V6_VAMP_RESULT_TREE_FORMAT",
    "load_v6_vamp_result",
    "begin_v6_vamp_sealed_transaction",
    "run_or_resume_v6_vamp_sealed_evaluation",
]
