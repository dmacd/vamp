#!/usr/bin/env python3
"""Run the fixed semantic-v6 base gate and first VAMP experiment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
PARTITION_SHA256 = (
    "3c49e53648332317f078c10ac5494fca7c1aaea39176ffebeb7f8a9fe9096bfa"
)
SAMPLE_REPORT_SHA256 = (
    "b9e998d5a6d169e3d630531db690da0adbf82e6fd75639f2acb4aa7525b15579"
)
PARTITION_DIRECTORY = DATA_ROOT / "v6" / PARTITION_SHA256
SAMPLE_REPORT_DIRECTORY = (
    DATA_ROOT
    / "sample-reports"
    / "v6"
    / PARTITION_SHA256
    / SAMPLE_REPORT_SHA256
)
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-semantic-v6"
PREFLIGHT_ROOT = CHECKPOINT_ROOT / "preflight"
SELECTION_ROOT = CHECKPOINT_ROOT / "selected"
ADAPTATION_ROOT = CHECKPOINT_ROOT / "vamp-adaptations"
WORK_ROOT = CHECKPOINT_ROOT / "work"
CALIBRATION_WORK = WORK_ROOT / "base-calibration"
ADAPTATION_WORK = WORK_ROOT / "vamp-chain-v1"
SEALED_TRANSACTION = WORK_ROOT / "vamp-sealed-transaction"
RESULT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "language_cl"
    / "tinyworlds-p-semantic-v6"
    / "vamp-chain-v1"
)
TOKENIZER_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


class _TqdmBar(Protocol):
    n: float
    total: float

    def update(self, amount: float = 1) -> object:
        """Advance the progress bar."""

    def set_postfix_str(self, value: str, refresh: bool = True) -> object:
        """Update the compact status text."""

    def close(self) -> None:
        """Close the progress bar."""

    def write(self, message: str) -> object:
        """Print a line without corrupting active bars."""


class _JsonlJournal:
    """Persist sequential progress records in small durable batches."""

    def __init__(self, directory: Path, batch_size: int = 16) -> None:
        self.path = directory / "progress.jsonl"
        self.batch_size = batch_size
        self._buffer: list[dict[str, object]] = []

    def append(self, event: str, **values: object) -> None:
        """Append one timestamped event and flush each bounded batch."""
        self._buffer.append(
            {"event": event, "monotonic_seconds": time.monotonic(), **values}
        )
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Durably write buffered canonical JSONL records."""
        if not self._buffer:
            return
        from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes

        with self.path.open("ab") as output:
            for record in self._buffer:
                output.write(canonical_json_bytes(record))
            output.flush()
            os.fsync(output.fileno())
        self._buffer.clear()


class _OverallProgress:
    """Weight heterogeneous phase work by the measured runtime estimate."""

    def __init__(self, estimated_seconds: float, journal: _JsonlJournal) -> None:
        from tqdm.auto import tqdm

        self.journal = journal
        self._bar: _TqdmBar = tqdm(
            total=max(1.0, estimated_seconds),
            desc="semantic-v6 overall ETA",
            unit="est-s",
            dynamic_ncols=True,
        )

    def phase(
        self,
        number: int,
        count: int,
        name: str,
        total_units: int,
        estimated_seconds: float,
    ) -> _UnitProgress:
        """Start one phase bar whose fractional progress advances overall ETA."""
        from tqdm.auto import tqdm

        self._bar.write(
            f"Phase {number}/{count}: {name} "
            f"(phase estimate {_duration(estimated_seconds)})"
        )
        self.journal.append(
            "phase_started",
            phase=number,
            phase_count=count,
            name=name,
            estimated_seconds=estimated_seconds,
        )
        return _UnitProgress(
            tqdm(
                total=max(1, total_units),
                desc=f"Phase {number}/{count}",
                unit="unit",
                dynamic_ncols=True,
                leave=False,
            ),
            self._bar,
            max(1, total_units),
            max(1.0, estimated_seconds),
            self.journal,
            name,
        )

    def close(self) -> None:
        """Close the overall ETA bar."""
        self._bar.close()


class _UnitProgress:
    """Map exact phase units onto a measured share of overall runtime."""

    def __init__(
        self,
        bar: _TqdmBar,
        overall: _TqdmBar,
        total_units: int,
        estimated_seconds: float,
        journal: _JsonlJournal,
        name: str,
    ) -> None:
        self._bar = bar
        self._overall = overall
        self._total_units = total_units
        self._estimated_seconds = estimated_seconds
        self._journal = journal
        self._name = name
        self._completed = 0

    def update_to(self, completed: int, status: str = "") -> None:
        """Advance to an absolute unit count while retaining phase and overall ETA."""
        bounded = min(self._total_units, max(self._completed, completed))
        delta = bounded - self._completed
        if delta:
            self._bar.update(delta)
            self._overall.update(delta * self._estimated_seconds / self._total_units)
            self._completed = bounded
        if status:
            self._bar.set_postfix_str(status, refresh=False)

    def close(self) -> None:
        """Complete the phase allocation and durably record its boundary."""
        self.update_to(self._total_units)
        self._bar.close()
        self._journal.append("phase_completed", name=self._name)
        self._journal.flush()


class _SplitProgress:
    """Convert repeated split-local counters into one ordered phase counter."""

    def __init__(
        self,
        expected: Sequence[tuple[str, int]],
        progress: _UnitProgress,
        journal: _JsonlJournal,
    ) -> None:
        self.expected = tuple(expected)
        self.progress = progress
        self.journal = journal
        self.index = 0
        self.active_completed = 0
        self.prior_total = 0

    def __call__(self, split: str, completed: int, total: int) -> None:
        """Advance the exact known split sequence and reject changed ordering."""
        if self.index >= len(self.expected):
            raise ValueError("semantic-v6 evaluation produced an extra split")
        if (split, total) != self.expected[self.index] or completed <= self.active_completed:
            if self.active_completed == self.expected[self.index][1]:
                self.prior_total += self.active_completed
                self.index += 1
                self.active_completed = 0
            if self.index >= len(self.expected) or (split, total) != self.expected[self.index]:
                raise ValueError("semantic-v6 evaluation split sequence changed")
        self.active_completed = completed
        self.progress.update_to(
            self.prior_total + completed,
            f"{split} {completed}/{total}",
        )
        self.journal.append(
            "evaluation_progress",
            split=split,
            completed=completed,
            total=total,
        )


def _fixed_sources():
    from apm.data.text.tinyworlds_p_semantic import (
        load_v6_partition,
        load_v6_sample_report,
    )
    from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
        V6_VAMP_EXPERIMENT_PRESET,
    )

    partition = load_v6_partition(PARTITION_DIRECTORY)
    sample = load_v6_sample_report(SAMPLE_REPORT_DIRECTORY)
    if (
        partition.partition_sha256 != V6_VAMP_EXPERIMENT_PRESET.partition_sha256
        or partition.semantic_catalog.catalog_sha256
        != V6_VAMP_EXPERIMENT_PRESET.catalog_sha256
        or sample.report_sha256 != V6_VAMP_EXPERIMENT_PRESET.sample_report_sha256
        or sample.partition_sha256 != partition.partition_sha256
    ):
        raise RuntimeError("semantic-v6 fixed source identities changed")
    return partition, sample


def _selected_preflight(partition_sha256: str, config_sha256: str):
    from apm.data.text.tinyworlds_p_semantic import load_v6_gpu_preflight

    pointer = PREFLIGHT_ROOT / "selected.json"
    if pointer.is_file():
        record = _load_json(pointer)
        preflight = load_v6_gpu_preflight(PREFLIGHT_ROOT / _text(record, "preflight_sha256"))
        if (
            preflight.partition_sha256 != partition_sha256
            or preflight.training_config_sha256 != config_sha256
        ):
            raise RuntimeError("selected semantic-v6 preflight binding changed")
        return preflight
    candidates = tuple(
        preflight
        for path in sorted(PREFLIGHT_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
        for preflight in (load_v6_gpu_preflight(path),)
        if preflight.partition_sha256 == partition_sha256
        and preflight.training_config_sha256 == config_sha256
    ) if PREFLIGHT_ROOT.is_dir() else ()
    if len(candidates) > 1:
        raise RuntimeError("multiple matching semantic-v6 preflights need explicit audit")
    if candidates:
        _write_json(pointer, {"preflight_sha256": candidates[0].preflight_sha256})
        return candidates[0]
    return None


def _selected_base(partition_sha256: str):
    from apm.data.text.tinyworlds_p_semantic import load_v6_selected_base

    candidates = tuple(
        selected
        for path in sorted(SELECTION_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
        for selected in (load_v6_selected_base(path),)
        if selected.partition_sha256 == partition_sha256
        and selected.sample_report_sha256 == SAMPLE_REPORT_SHA256
    ) if SELECTION_ROOT.is_dir() else ()
    if len(candidates) > 1:
        raise RuntimeError("multiple semantic-v6 selected bases exist")
    return None if not candidates else candidates[0]


def _selected_adaptations(selected_base_sha256: str, config_sha256: str):
    from apm.data.text.tinyworlds_p_semantic import (
        load_v6_vamp_adaptation_publication,
    )

    candidates = tuple(
        adaptation
        for path in sorted(ADAPTATION_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
        for adaptation in (load_v6_vamp_adaptation_publication(path),)
        if adaptation.selected_base_sha256 == selected_base_sha256
        and adaptation.config_sha256 == config_sha256
    ) if ADAPTATION_ROOT.is_dir() else ()
    if len(candidates) > 1:
        raise RuntimeError("multiple semantic-v6 VAMP adapter publications exist")
    return None if not candidates else candidates[0]


def _training_callback(
    progress: _UnitProgress,
    journal: _JsonlJournal,
    offset: int,
) -> Callable[[object, float, int], None]:
    def update(cursor: object, nll: float, planned: int) -> None:
        optimizer_update = int(getattr(cursor, "optimizer_update"))
        progress.update_to(optimizer_update - offset, f"NLL {nll:.4f}")
        journal.append(
            "base_update",
            optimizer_update=optimizer_update,
            nll=nll,
            planned_updates=planned,
        )

    return update


def _evaluation_sequence(artifact, split: str, epochs: int = 1):
    from apm.data.text.tinyworlds_p_semantic import count_v6_evaluation_batches

    one = tuple(
        (selector, count_v6_evaluation_batches(artifact, selector))
        for selector in _selectors(split)
    )
    return one * epochs


def _selectors(split: str) -> tuple[str, ...]:
    return (
        f"base/{split}",
        *(
            f"{role}/{world}/{split}"
            for world in ("A", "B", "C", "D", "E")
            for role in ("world", "control")
        ),
    )


def _run_preflight(partition, config, journal: _JsonlJournal):
    from apm.data.text.tinyworlds_p_semantic import (
        run_and_publish_v6_gpu_preflight,
    )

    estimated_seconds = 600.0
    overall = _OverallProgress(estimated_seconds, journal)
    phase = overall.phase(1, 1, "two disposable GPU updates and warm evaluation", 3, estimated_seconds)
    try:
        preflight_work = Path(
            tempfile.mkdtemp(prefix="gpu-preflight-", dir=WORK_ROOT)
        )
        update = _training_callback(phase, journal, 0)
        preflight = run_and_publish_v6_gpu_preflight(
            partition,
            config,
            preflight_work,
            PREFLIGHT_ROOT,
            progress=update,
        )
        shutil.rmtree(preflight_work)
        phase.update_to(3, "publication authenticated")
    finally:
        phase.close()
        overall.close()
    _write_json(
        PREFLIGHT_ROOT / "selected.json",
        {"preflight_sha256": preflight.preflight_sha256},
    )
    return preflight


def _run_base_selection(
    partition,
    sample,
    config,
    preflight,
    journal: _JsonlJournal,
):
    from apm.data.text.tinyworlds_p_semantic import (
        finish_v6_base_selection,
        load_v6_calibration_attempt,
        run_v6_calibration_attempt,
    )

    updates_per_epoch = preflight.updates_per_epoch
    validation_sequence = _evaluation_sequence(partition, "validation", 2)
    calibration_estimate = preflight.estimated_calibration_seconds
    remaining_estimate = max(
        1.0,
        preflight.estimated_base_pass_path_seconds - calibration_estimate,
    )
    overall = _OverallProgress(calibration_estimate + remaining_estimate, journal)
    calibration_training = overall.phase(
        1,
        4,
        "fresh seed-zero epochs one and two",
        2 * updates_per_epoch,
        2 * updates_per_epoch * preflight.seconds_per_update,
    )
    calibration_evaluation = overall.phase(
        2,
        4,
        "registered validation gate for epochs one and two",
        sum(total for _, total in validation_sequence),
        2 * preflight.validation_batches_per_epoch * preflight.seconds_per_evaluation_batch,
    )
    try:
        if (CALIBRATION_WORK / "calibration.json").is_file():
            calibration_training.close()
            calibration_evaluation.close()
            calibration = load_v6_calibration_attempt(
                partition,
                sample.root,
                CALIBRATION_WORK,
                config,
            )
        else:
            split_progress = _SplitProgress(
                validation_sequence,
                calibration_evaluation,
                journal,
            )
            calibration = run_v6_calibration_attempt(
                partition,
                sample.root,
                CALIBRATION_WORK,
                config,
                progress=_training_callback(calibration_training, journal, 0),
                evaluation_progress=split_progress,
            )
            calibration_training.close()
            calibration_evaluation.close()
        if calibration.decision != "pass":
            overall.close()
            print(
                f"[stop] The registered two-epoch semantic decision was "
                f"{calibration.decision}. No adapters or sealed test were opened.",
                flush=True,
            )
            return None
        final_training = overall.phase(
            3,
            4,
            "epochs three through five after the semantic pass",
            3 * updates_per_epoch,
            3 * updates_per_epoch * preflight.seconds_per_update,
        )
        later_sequence = _evaluation_sequence(partition, "validation", 3)
        final_evaluation = overall.phase(
            4,
            4,
            "validation-only checkpoint selection",
            sum(total for _, total in later_sequence),
            3 * preflight.validation_batches_per_epoch
            * preflight.seconds_per_evaluation_batch,
        )
        try:
            selected = finish_v6_base_selection(
                calibration,
                SELECTION_ROOT,
                TOKENIZER_DIRECTORY,
                progress=_training_callback(
                    final_training,
                    journal,
                    2 * updates_per_epoch,
                ),
                evaluation_progress=_SplitProgress(
                    later_sequence,
                    final_evaluation,
                    journal,
                ),
            )
        finally:
            final_training.close()
            final_evaluation.close()
        return selected
    finally:
        overall.close()


def _run_adaptations(partition, selected, preflight, preset, journal):
    from apm.data.text.tinyworlds_p_semantic import (
        count_v6_partition_microbatches,
        prepare_v6_vamp_training_curriculum,
        train_or_resume_v6_vamp_adaptations,
    )

    counts = tuple(
        count_v6_partition_microbatches(partition, f"world/{world}/train")
        for world in preset.task_order
    )
    preparation_estimate = max(60.0, sum(counts) * 0.02)
    overall = _OverallProgress(
        preparation_estimate + preflight.estimated_adapter_training_seconds,
        journal,
    )
    preparation = overall.phase(
        1,
        2,
        "materialize five training curricula and validation-only probes",
        sum(counts),
        preparation_estimate,
    )
    prior_by_world = {
        world: sum(counts[:index])
        for index, world in enumerate(preset.task_order)
    }
    prepared = prepare_v6_vamp_training_curriculum(
        partition,
        preset,
        progress=lambda label, completed, total: (
            preparation.update_to(
                prior_by_world[label.rsplit(" ", 1)[-1]] + completed,
                f"{label} {completed}/{total}",
            ),
            journal.append(
                "curriculum_progress",
                label=label,
                completed=completed,
                total=total,
            ),
        )[-1],
    )
    preparation.close()
    total_adapter_updates = len(preset.task_order) * 3 * preset.adapter_steps_per_task
    training = overall.phase(
        2,
        2,
        "sequential, independent, and VAMP adapters for A through E",
        total_adapter_updates,
        preflight.estimated_adapter_training_seconds,
    )
    method_index = {"sequential": 0, "independent": 1, "vamp": 2}
    world_index = {world: index for index, world in enumerate(preset.task_order)}

    def adapter_progress(method: str, world: str, step: int, loss: float, total: int):
        completed = (
            (world_index[world] * 3 + method_index[method]) * total + step
        )
        training.update_to(completed, f"{method} {world}, NLL {loss:.4f}")
        journal.append(
            "adapter_update",
            method=method,
            world=world,
            step=step,
            total=total,
            nll=loss,
        )

    try:
        publication = train_or_resume_v6_vamp_adaptations(
            prepared,
            selected,
            ADAPTATION_WORK,
            ADAPTATION_ROOT,
            preset,
            progress=adapter_progress,
        )
    finally:
        training.close()
        overall.close()
    return publication


def _run_sealed_result(partition, selected, adaptations, preflight, preset, journal):
    from apm.data.text.tinyworlds_p_semantic import (
        begin_v6_vamp_sealed_transaction,
        count_v6_evaluation_batches,
        run_or_resume_v6_vamp_sealed_evaluation,
    )
    from apm.lm.text import TokenizersTextTokenizer

    begin_v6_vamp_sealed_transaction(
        partition,
        selected,
        adaptations,
        SEALED_TRANSACTION,
        preset,
    )
    print(
        "[sealed] The one-time transaction is now durable; test indexes may be read.",
        flush=True,
    )
    base_sequence = _evaluation_sequence(partition, "test")
    specificity_order = tuple(
        (method, world, role)
        for world in preset.task_order
        for method in (
            "sequential_single_lora",
            "independent_root_lora",
            "vamp_oracle",
        )
        for role in ("world", "control")
    )
    specificity_counts = {
        (method, world, role): count_v6_evaluation_batches(
            partition,
            f"{role}/{world}/test",
        )
        for method, world, role in specificity_order
    }
    condition_units = 60
    base_units = sum(total for _, total in base_sequence)
    specificity_units = sum(specificity_counts.values())
    estimated_seconds = (
        (base_units + specificity_units) * preflight.seconds_per_evaluation_batch
        + condition_units * max(1.0, 20 * preflight.seconds_per_evaluation_batch)
    )
    overall = _OverallProgress(estimated_seconds, journal)
    base_progress = overall.phase(
        1,
        3,
        "one-time selected-base sealed evaluation",
        base_units,
        base_units * preflight.seconds_per_evaluation_batch,
    )
    condition_progress = overall.phase(
        2,
        3,
        "nine-method nested-prefix comparison",
        condition_units,
        condition_units * max(1.0, 20 * preflight.seconds_per_evaluation_batch),
    )
    specificity_progress = overall.phase(
        3,
        3,
        "forced-adapter paired-control specificity",
        specificity_units,
        specificity_units * preflight.seconds_per_evaluation_batch,
    )
    base_callback = _SplitProgress(base_sequence, base_progress, journal)
    prior_specificity = {
        identity: sum(
            specificity_counts[prior]
            for prior in specificity_order[:index]
        )
        for index, identity in enumerate(specificity_order)
    }

    def specificity_callback(method, world, role, completed, total):
        identity = (method, world, role)
        if specificity_counts[identity] != total:
            raise ValueError("semantic-v6 specificity batch count changed")
        specificity_progress.update_to(
            prior_specificity[identity] + completed,
            f"{method} {world}/{role} {completed}/{total}",
        )
        journal.append(
            "specificity_progress",
            method=method,
            world=world,
            role=role,
            completed=completed,
            total=total,
        )

    tokenizer = TokenizersTextTokenizer.from_file(
        selected.directory / "tokenizer" / "tokenizer.json",
        pad_token_id=partition.pad_token_id,
    )
    try:
        result = run_or_resume_v6_vamp_sealed_evaluation(
            partition,
            selected,
            adaptations,
            tokenizer,
            SEALED_TRANSACTION,
            RESULT_ROOT,
            preset,
            phase_progress=lambda message: (
                print(f"[sealed] {message}", flush=True),
                journal.append("sealed_phase", message=message),
            )[-1],
            evaluation_progress=base_callback,
            specificity_progress=specificity_callback,
            condition_progress=lambda stage, task, condition, completed, total: (
                condition_progress.update_to(
                    completed,
                    f"stage {stage}, task {task}, {condition}",
                ),
                journal.append(
                    "condition_progress",
                    stage=stage,
                    task=str(task),
                    condition=condition,
                    completed=completed,
                    total=total,
                ),
            )[-1],
        )
    finally:
        base_progress.close()
        condition_progress.close()
        specificity_progress.close()
        overall.close()
    return result


def main() -> int:
    """Advance the fixed experiment by exactly the next safe state transition."""
    if len(sys.argv) != 1:
        raise SystemExit("This fixed experiment takes no command-line options.")
    import jax

    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) != 1:
        raise RuntimeError(
            "semantic-v6 requires the single CUDA GPU and must run outside the sandbox"
        )
    from apm.data.text.tinyworlds_p_semantic import V6StreamingTrainingConfig
    from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
        V6_VAMP_EXPERIMENT_PRESET,
    )

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    logs_root = WORK_ROOT / "logs"
    logs_root.mkdir(exist_ok=True)
    live_directory = Path(
        tempfile.mkdtemp(prefix="semantic-v6-live-", dir=logs_root)
    )
    print(f"Temporary live artifact directory: {live_directory}", flush=True)
    journal = _JsonlJournal(live_directory)
    try:
        partition, sample = _fixed_sources()
        config = V6StreamingTrainingConfig.from_preset()
        config_sha256 = _record_sha256(config.as_record())
        print(
            f"Authenticated partition {partition.partition_sha256}, sample report "
            f"{sample.report_sha256}, and GPU {devices[0].device_kind}.",
            flush=True,
        )
        preflight = _selected_preflight(partition.partition_sha256, config_sha256)
        if preflight is None:
            preflight = _run_preflight(partition, config, journal)
            print(
                f"GPU preflight: {preflight.directory}\n"
                f"Two-epoch calibration estimate: "
                f"{_duration(preflight.estimated_calibration_seconds)}\n"
                f"Five-epoch validation-only base path estimate: "
                f"{_duration(preflight.estimated_base_pass_path_seconds)}\n"
                "The real run has not started, and the sealed test remains closed. "
                "Review this estimate, then run this same command again to proceed.",
                flush=True,
            )
            return 0
        print(
            f"Using reviewed GPU preflight {preflight.preflight_sha256}.",
            flush=True,
        )
        selected = _selected_base(partition.partition_sha256)
        if selected is None:
            selected = _run_base_selection(
                partition,
                sample,
                config,
                preflight,
                journal,
            )
            if selected is None:
                return 2
        print(
            f"Validation-selected base {selected.selection_sha256}, epoch "
            f"{selected.selected_epoch}; sealed test still closed.",
            flush=True,
        )
        preset = V6_VAMP_EXPERIMENT_PRESET
        adaptations = _selected_adaptations(
            selected.selection_sha256,
            preset.config_sha256,
        )
        if adaptations is None:
            adaptations = _run_adaptations(
                partition,
                selected,
                preflight,
                preset,
                journal,
            )
        print(
            f"Frozen adapter publication {adaptations.run_sha256}; beginning the "
            "single sealed transaction.",
            flush=True,
        )
        result = _run_sealed_result(
            partition,
            selected,
            adaptations,
            preflight,
            preset,
            journal,
        )
        print(f"Final exploratory VAMP result: {result.directory}", flush=True)
        return 0
    finally:
        journal.flush()


def _record_sha256(value: object) -> str:
    from apm.data.text.tinyworlds_p_semantic.contracts import record_sha256

    return record_sha256(value)


def _write_json(path: Path, value: object) -> None:
    from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes

    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"runner JSON is not canonical: {path}")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"runner field {field!r} must be text")
    return value


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
