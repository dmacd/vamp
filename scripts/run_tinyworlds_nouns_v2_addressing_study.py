#!/usr/bin/env python3
"""Run the fixed TinyWorlds nouns-v2 final-checkpoint addressing study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Event, Thread
from time import monotonic
from typing import Protocol, TypeVar


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
_existing_xla_flags = tuple(
    flag
    for flag in os.environ.get("XLA_FLAGS", "").split()
    if not flag.startswith("--xla_gpu_enable_command_buffer")
)
os.environ["XLA_FLAGS"] = " ".join(
    (*_existing_xla_flags, "--xla_gpu_enable_command_buffer=")
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIRECTORY = (
    REPOSITORY_ROOT
    / "results/language_cl/tinyworlds-nouns-v2/addressing-study"
)
WORK_DIRECTORY = STUDY_DIRECTORY / ".work-v1"


@dataclass(frozen=True, slots=True)
class _Phase:
    number: int
    description: str
    estimated_seconds: int


PHASES = (
    _Phase(1, "authenticate canonical inputs and ledgers", 120),
    _Phase(2, "derive or authenticate frozen keys and contracts", 1_200),
    _Phase(3, "prove compact/dense parity on the final checkpoint", 300),
    _Phase(4, "stream and resume the complete addressing evaluation", 5_400),
    _Phase(5, "enforce the 12 GiB allocator and immutability gates", 30),
    _Phase(6, "publish and exactly regenerate standalone reports", 300),
)


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the displayed progress."""

    def close(self) -> None:
        """Close the displayed progress."""

    def write(self, message: str) -> object:
        """Print without corrupting the displayed progress."""


ResultT = TypeVar("ResultT")


class _StudyProgress:
    """Display phase and overall ETA bars plus bounded evaluation updates."""

    def __init__(self) -> None:
        self._overall_bar: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None
        self._evaluation_last_report = -1

    def __enter__(self) -> _StudyProgress:
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall_bar = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="Nouns-v2 addressing overall ETA",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if self._overall_bar is not None:
            self._overall_bar.close()

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one fixed phase while keeping its ETA and the overall ETA live."""
        if self._overall_bar is None or self._tqdm_factory is None:
            raise RuntimeError("addressing progress must be entered before use")
        self._overall_bar.write(
            f"Phase {phase.number}/{len(PHASES)}: {phase.description}."
        )
        phase_bar = self._tqdm_factory(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)} ETA",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )
        stop = Event()
        timer = Thread(
            target=_advance_eta_bars,
            args=(stop, phase_bar, self._overall_bar, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            return operation()
        finally:
            stop.set()
            timer.join()
            remaining = max(0.0, phase.estimated_seconds - phase_bar.n)
            phase_bar.update(remaining)
            self._overall_bar.update(remaining)
            phase_bar.close()

    def evaluation_update(self, name: str, completed: int, total: int) -> None:
        """Print sparse case progress without disturbing the phase ETA bars."""
        if self._overall_bar is None or name != "addressing-evaluation":
            return
        milestone = completed // 200
        if completed != total and milestone == self._evaluation_last_report:
            return
        self._evaluation_last_report = milestone
        self._overall_bar.write(
            f"Evaluation coverage: {completed:,}/{total:,} cases "
            f"({completed / total:.1%}); ledgers flush after every microbatch."
        )


def _advance_eta_bars(
    stop: Event,
    phase_bar: _TqdmBar,
    overall_bar: _TqdmBar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds - 1:
            phase_bar.update(1)
            overall_bar.update(1)


def _load_measurement(
    path: Path,
    expected_format: str,
    binding_name: str,
    binding_sha256: str,
) -> dict[str, object]:
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
        canonical_json_bytes,
        record_sha256,
    )

    payload = path.read_bytes()
    record = json.loads(payload)
    supplied = record.get("result_sha256") if type(record) is dict else None
    core = (
        {key: value for key, value in record.items() if key != "result_sha256"}
        if type(record) is dict
        else {}
    )
    if (
        type(record) is not dict
        or payload != canonical_json_bytes(record)
        or record.get("format") != expected_format
        or record.get(binding_name) != binding_sha256
        or supplied != record_sha256(core)
    ):
        raise ValueError(f"addressing measurement identity changed: {path.name}")
    return record


def _publish_measurement(path: Path, core: dict[str, object]) -> dict[str, object]:
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
        canonical_json_bytes,
        record_sha256,
    )

    record = {**core, "result_sha256": record_sha256(core)}
    payload = canonical_json_bytes(record)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"addressing measurement changed: {path.name}")
    else:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return record


def _report_snapshot(directory: Path) -> tuple[tuple[str, str], ...]:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    relative_paths = tuple(sorted(str(path) for path in manifest["artifacts"]))
    return tuple(
        (relative, _file_sha256(directory / relative))
        for relative in (*relative_paths, "manifest.json")
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _notify_completion(report: Path) -> None:
    try:
        subprocess.run(
            (
                "notify-send",
                "TinyWorlds nouns-v2 addressing study complete",
                str(report),
            ),
            check=False,
        )
    except FileNotFoundError:
        pass


def main() -> int:
    """Execute the sole fixed study configuration on GPU 0 and publish reports."""
    from apm.data.text.tinyworlds_nouns_v2.addressing_study import (
        assert_canonical_hashes_unchanged,
        authenticate_addressing_study_inputs,
        build_study_contracts,
        enforce_nouns_v2_allocator_gate,
        run_or_resume_addressing_evaluation,
        verify_compact_real_parity,
    )
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
        build_or_load_addressing_keys,
    )
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_report import (
        publish_addressing_study_report,
    )

    STUDY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    WORK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Temporary artifact directory: {WORK_DIRECTORY}", flush=True)
    started = monotonic()
    with _StudyProgress() as progress:
        inputs = progress.run(
            PHASES[0],
            lambda: authenticate_addressing_study_inputs(REPOSITORY_ROOT),
        )

        def prepare_keys_and_contracts():
            keys = build_or_load_addressing_keys(
                inputs.partition,
                inputs.base_params,
                inputs.adaptation.model_config,
                inputs.adaptation,
                STUDY_DIRECTORY / "keys",
            )
            contracts = build_study_contracts(inputs, keys, STUDY_DIRECTORY)
            return keys, contracts

        keys, (retrieval_contract, ebt_contract) = progress.run(
            PHASES[1],
            prepare_keys_and_contracts,
        )
        ebt_sha256 = str(ebt_contract["contract_sha256"])
        parity_path = STUDY_DIRECTORY / "parity.json"

        def prove_or_load_parity() -> dict[str, object]:
            if parity_path.is_file():
                return _load_measurement(
                    parity_path,
                    "tinyworlds-nouns-v2-addressing-parity-v1",
                    "ebt_contract_sha256",
                    ebt_sha256,
                )
            parity = verify_compact_real_parity(inputs, keys)
            return _publish_measurement(
                parity_path,
                {
                    "ebt_contract_sha256": ebt_sha256,
                    "format": "tinyworlds-nouns-v2-addressing-parity-v1",
                    **parity,
                },
            )

        parity = progress.run(PHASES[2], prove_or_load_parity)
        retrieval_path, ebt_path, timing_path, runtimes = progress.run(
            PHASES[3],
            lambda: run_or_resume_addressing_evaluation(
                inputs,
                keys,
                retrieval_contract,
                ebt_contract,
                STUDY_DIRECTORY,
                WORK_DIRECTORY,
                progress=progress.evaluation_update,
            ),
        )
        allocator_path = STUDY_DIRECTORY / "allocator.json"

        def enforce_and_record_gates() -> dict[str, object]:
            observed = enforce_nouns_v2_allocator_gate(inputs.preset)
            if allocator_path.is_file():
                stored = _load_measurement(
                    allocator_path,
                    "tinyworlds-nouns-v2-addressing-allocator-v1",
                    "ebt_contract_sha256",
                    ebt_sha256,
                )
            else:
                stored = _publish_measurement(
                    allocator_path,
                    {
                        **observed,
                        "ebt_contract_sha256": ebt_sha256,
                        "format": "tinyworlds-nouns-v2-addressing-allocator-v1",
                    },
                )
            assert_canonical_hashes_unchanged(
                REPOSITORY_ROOT,
                inputs.canonical_hashes,
            )
            return stored

        allocator = progress.run(PHASES[4], enforce_and_record_gates)

        def publish_and_reproduce_reports() -> tuple[Path, Path, Path]:
            arguments = (
                inputs,
                retrieval_contract,
                ebt_contract,
                retrieval_path,
                ebt_path,
                timing_path,
                STUDY_DIRECTORY,
            )
            keyword_arguments = {
                "allocator": allocator,
                "parity": parity,
                "runtimes": runtimes,
            }
            published = publish_addressing_study_report(
                *arguments,
                **keyword_arguments,
            )
            first = _report_snapshot(STUDY_DIRECTORY)
            reproduced = publish_addressing_study_report(
                *arguments,
                **keyword_arguments,
            )
            if first != _report_snapshot(STUDY_DIRECTORY):
                raise RuntimeError("addressing report regeneration was not byte-identical")
            assert_canonical_hashes_unchanged(
                REPOSITORY_ROOT,
                inputs.canonical_hashes,
            )
            return reproduced

        markdown, html, manifest = progress.run(
            PHASES[5],
            publish_and_reproduce_reports,
        )
    _notify_completion(html)
    print(f"Study manifest: {manifest}", flush=True)
    print(f"Markdown report: {markdown}", flush=True)
    print(f"Self-contained HTML report: {html}", flush=True)
    print(f"Completed in {monotonic() - started:.1f} seconds.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
