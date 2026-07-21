from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.phase1_artifacts import Phase1ArtifactBuilder
from apm.data.text.tinyworlds_v2.prompt_tuning_reevaluation import (
    PROMPT_REEVALUATION_SOURCE_CELLS,
    PromptReevaluationPaths,
    ReevaluationComparator,
    _load_cached_source_cell,
    _snapshot_source,
    _validate_comparator,
    run_prompt_reevaluation,
    validate_prompt_reevaluation,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import ValidationStoryRecord
from apm.data.text.tinyworlds_v2.validation_decontamination import (
    DecontaminationExpectations,
    ValidationDecontaminationAudit,
)


def _completed_sources() -> tuple[Path, Path] | None:
    root = Path(__file__).resolve().parents[1] / "data" / "tinyworlds-v2"
    sources = root / "prompt-tuning-v1", root / "prompt-tuning-v2"
    return sources if all(source.is_dir() for source in sources) else None


def _comparator_record(index: int, story: str) -> ValidationStoryRecord:
    content_sha256 = sha256(story.encode("utf-8")).hexdigest()
    return ValidationStoryRecord(
        record_id=f"v2-validation:{index}:{content_sha256}",
        source_index=index,
        content_sha256=content_sha256,
        story=story,
    )


def _tiny_comparator() -> tuple[ReevaluationComparator, DecontaminationExpectations]:
    records = (
        _comparator_record(
            1,
            "Once upon a time, Mia found a red ball. She played with it and smiled.",
        ),
        _comparator_record(
            2,
            "One day, Tom helped a small bird. The bird sang, and Tom felt happy.",
        ),
    )
    observations = tuple(
        observe_reference(
            ReferenceRecord(record.record_id, record.story, source_model="GPT-4"),
            model_token_ids=(index + 1, index + 2, index + 3),
            normalized_nll=1.0 + index / 10,
        )
        for index, record in enumerate(records)
    )
    retained_digest = sha256(
        "".join(f"{record.record_id}\n" for record in records).encode("utf-8")
    ).hexdigest()
    profile = build_reference_profile(observations)
    expectations = DecontaminationExpectations(
        input_count=2,
        overlap_count=0,
        retained_count=2,
        overlap_ids_sha256=sha256(b"").hexdigest(),
        retained_ids_sha256=retained_digest,
        retained_profile_sha256=profile.profile_sha256,
    )
    audit = ValidationDecontaminationAudit(
        input_count=expectations.input_count,
        overlap_count=expectations.overlap_count,
        retained_count=expectations.retained_count,
        overlap_ids_sha256=expectations.overlap_ids_sha256,
        retained_ids_sha256=expectations.retained_ids_sha256,
        retained_profile_sha256=expectations.retained_profile_sha256,
    )
    return (
        ReevaluationComparator(
            audit=audit.as_record(),
            records=records,
            observations=observations,
        ),
        expectations,
    )


def test_reevaluation_paths_do_not_replace_v1_or_v2(tmp_path: Path) -> None:
    paths = PromptReevaluationPaths.from_repository(tmp_path)

    assert paths.prompt_tuning_v1.name == "prompt-tuning-v1"
    assert paths.prompt_tuning_v2.name == "prompt-tuning-v2"
    assert paths.destination.name == "prompt-tuning-v3"
    assert paths.destination not in (paths.prompt_tuning_v1, paths.prompt_tuning_v2)


def test_cached_source_cells_rebuild_exact_v1_v6_and_v2_v7(tmp_path: Path) -> None:
    sources = _completed_sources()
    if sources is None:
        pytest.skip("completed prompt-tuning artifacts are not present")
    builder = Phase1ArtifactBuilder(tmp_path, version="fixture-v3-sources")
    for contract, source in zip(
        PROMPT_REEVALUATION_SOURCE_CELLS,
        sources,
        strict=True,
    ):
        _snapshot_source(builder, source, contract)

    cells = tuple(
        _load_cached_source_cell(tmp_path, contract)
        for contract in PROMPT_REEVALUATION_SOURCE_CELLS
    )

    assert tuple(cell.contract.source_variant_id for cell in cells) == (
        "v6-tuned",
        "v7-tuned",
    )
    assert tuple(len(cell.samples) for cell in cells) == (40, 40)
    assert tuple(len(cell.measurements.measurements) for cell in cells) == (34, 32)


def test_cached_source_snapshot_tampering_is_rejected(tmp_path: Path) -> None:
    sources = _completed_sources()
    if sources is None:
        pytest.skip("completed prompt-tuning artifacts are not present")
    contract = PROMPT_REEVALUATION_SOURCE_CELLS[0]
    builder = Phase1ArtifactBuilder(tmp_path, version="fixture-v3-source-tamper")
    _snapshot_source(builder, sources[0], contract)
    path = tmp_path / "sources" / contract.label / "generation_results.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source snapshot differs"):
        _load_cached_source_cell(tmp_path, contract)


def test_comparator_audit_must_match_the_complete_pinned_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparator, expectations = _tiny_comparator()
    import apm.data.text.tinyworlds_v2.prompt_tuning_reevaluation as module

    monkeypatch.setattr(module, "PROMPT_REEVALUATION_EXPECTATIONS", expectations)
    changed_audit = {**comparator.audit, "version": "changed-audit-version"}

    with pytest.raises(ValueError, match="audit differs"):
        _validate_comparator(replace(comparator, audit=changed_audit))


def test_complete_v3_build_reuses_every_model_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _completed_sources()
    if sources is None:
        pytest.skip("completed prompt-tuning artifacts are not present")
    comparator, expectations = _tiny_comparator()
    import apm.data.text.tinyworlds_v2.prompt_tuning_reevaluation as module

    monkeypatch.setattr(
        module,
        "PROMPT_REEVALUATION_EXPECTATIONS",
        expectations,
    )
    repository_root = Path(__file__).resolve().parents[1]
    base_paths = PromptReevaluationPaths.from_repository(repository_root)
    destination = tmp_path / "published-v3"
    paths = replace(base_paths, destination=destination)
    staging = tmp_path / "staging"
    staging.mkdir()

    result = run_prompt_reevaluation(staging, paths, comparator)

    assert result.directory == destination
    assert validate_prompt_reevaluation(destination).manifest_sha256 == (
        result.manifest_sha256
    )
    reuse = module._json_object(destination / "reuse.json")
    assert reuse == {
        "cached_accepted_measurement_count": 66,
        "cached_story_count": 80,
        "new_external_generation_cost_usd": "0",
        "new_external_generation_requests": 0,
        "new_generated_story_nll_measurements": 0,
        "new_model_calls": 0,
    }
    assert not (destination / "raw_cache").exists()
    assert not (destination / "byok_preflight.json").exists()
    assert not (destination / "cost_estimates.json").exists()
