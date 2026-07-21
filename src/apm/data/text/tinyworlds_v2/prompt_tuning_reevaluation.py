"""Zero-inference re-evaluation of the cached V6/V7 prompt experiment.

V3 is deliberately not another paid prompt-tuning experiment.  It copies and
authenticates the exact V6 cell from prompt-tuning V1 and the exact V7 cell
from prompt-tuning V2, reuses their already-computed TinyStories-8M losses,
and changes only the reference distribution used by deterministic quality
evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    canonical_json_bytes,
    canonical_json_loads,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.bakeoff import NeutralStoryBrief
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    Phase1ArtifactManifest,
    _decode_manifest,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.phase1_generation import GeneratedSample
from apm.data.text.tinyworlds_v2.phase1_replay import (
    _decode_reference_profile,
    _load_measurement_batch,
    _load_reference_observations,
    _route,
)
from apm.data.text.tinyworlds_v2.phase1_runner import (
    MeasurementBatch,
    StoryMeasurement,
    _quality_report_record,
)
from apm.data.text.tinyworlds_v2.phase1_semantics import (
    _decode_reference_observation,
    _decode_validation_record,
    _reference_profile_record,
)
from apm.data.text.tinyworlds_v2.prompt_tuning import (
    PROMPT_TUNING_BRIEF_COUNT,
    PROMPT_TUNING_V1_EXPERIMENT,
    PROMPT_TUNING_V1_MANIFEST_SHA256,
    PROMPT_TUNING_V2_EXPERIMENT,
    PROMPT_TUNING_V2_VARIANTS,
    PromptVariant,
    VariantJobs,
    _brief_record,
    _decode_control_sample,
    _decode_selected_briefs,
    _evaluate_cells,
    _json_object,
    _jsonl_objects,
    _render_review_html,
    _variant_sample_record,
    build_prompt_tuning_jobs,
)
from apm.data.text.tinyworlds_v2.quality import (
    TWO_ROUTE_AUTHOR_ORDER,
    RouteQualityReport,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    canonical_reference_observation,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import (
    ValidationStoryRecord,
    canonical_validation_record,
)
from apm.data.text.tinyworlds_v2.two_route_bakeoff import (
    TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
)
from apm.data.text.tinyworlds_v2.validation_decontamination import (
    PRODUCTION_DECONTAMINATION_EXPECTATIONS,
    VALIDATION_RETAINED_COUNT,
    VALIDATION_RETAINED_IDS_SHA256,
    DecontaminationExpectations,
    ValidationDecontaminationAudit,
    build_decontaminated_validation_profile,
)


PROMPT_REEVALUATION_VERSION = "tinyworlds-v2-prompt-tuning-reevaluation-v3"
PROMPT_REEVALUATION_V2_MANIFEST_SHA256 = (
    "838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a"
)
PROMPT_REEVALUATION_EXPECTATIONS: DecontaminationExpectations = (
    PRODUCTION_DECONTAMINATION_EXPECTATIONS
)
PROMPT_REEVALUATION_RETAINED_COUNT = VALIDATION_RETAINED_COUNT
PROMPT_REEVALUATION_RETAINED_IDS_SHA256 = VALIDATION_RETAINED_IDS_SHA256


@dataclass(frozen=True, slots=True)
class SourceCellContract:
    """One immutable cached prompt cell copied from a completed experiment."""

    label: str
    artifact_version: str
    artifact_manifest_sha256: str
    source_variant_id: str
    target_variant: PromptVariant
    plan_path: str


PROMPT_REEVALUATION_SOURCE_CELLS = (
    SourceCellContract(
        label="v1-v6",
        artifact_version=PROMPT_TUNING_V1_EXPERIMENT.version,
        artifact_manifest_sha256=PROMPT_TUNING_V1_MANIFEST_SHA256,
        source_variant_id="v6-tuned",
        target_variant=PROMPT_TUNING_V2_VARIANTS[0],
        plan_path="plans/v6-tuned.jsonl",
    ),
    SourceCellContract(
        label="v2-v7",
        artifact_version=PROMPT_TUNING_V2_EXPERIMENT.version,
        artifact_manifest_sha256=PROMPT_REEVALUATION_V2_MANIFEST_SHA256,
        source_variant_id="v7-tuned",
        target_variant=PROMPT_TUNING_V2_VARIANTS[1],
        plan_path="plans/v7-tuned.jsonl",
    ),
)

_SOURCE_COPY_PATHS = (
    "catalog/routes.json",
    "generation_results.jsonl",
    "measurements.jsonl",
    "selected_briefs.jsonl",
)


@dataclass(frozen=True, slots=True)
class ReevaluationComparator:
    """Decontaminated records and already-scored observations used by V3."""

    audit: JsonObject
    records: tuple[ValidationStoryRecord, ...]
    observations: tuple[ReferenceObservation, ...]

    def __post_init__(self) -> None:
        if type(self.audit) is not dict:
            raise TypeError("re-evaluation comparator audit must be an object")
        if type(self.records) is not tuple or any(
            type(record) is not ValidationStoryRecord for record in self.records
        ):
            raise TypeError("re-evaluation comparator records have the wrong type")
        if type(self.observations) is not tuple or any(
            type(observation) is not ReferenceObservation
            for observation in self.observations
        ):
            raise TypeError("re-evaluation comparator observations have the wrong type")

    @property
    def profile(self) -> ReferenceProfile:
        """Build the authoritative profile from the retained observations."""
        return build_reference_profile(self.observations)


@dataclass(frozen=True, slots=True)
class PromptReevaluationPaths:
    """Fixed source and publication paths for the V3 cached re-evaluation."""

    repository_root: Path
    prompt_tuning_v1: Path
    prompt_tuning_v2: Path
    base_reference: Path
    original_train: Path
    destination: Path

    @classmethod
    def from_repository(
        cls,
        repository_root: str | Path,
    ) -> "PromptReevaluationPaths":
        root = Path(repository_root).resolve()
        data_root = root / "data" / "tinyworlds-v2"
        return cls(
            repository_root=root,
            prompt_tuning_v1=data_root / "prompt-tuning-v1",
            prompt_tuning_v2=data_root / "prompt-tuning-v2",
            base_reference=data_root / "reference",
            original_train=root / "data" / "tinystories-original" / "TinyStories-train.txt",
            destination=data_root / "prompt-tuning-v3",
        )


@dataclass(frozen=True, slots=True)
class CachedPromptCell:
    """One locally rederived prompt cell and its cached NLL measurements."""

    contract: SourceCellContract
    jobs: VariantJobs
    samples: tuple[GeneratedSample, ...]
    measurements: MeasurementBatch


@dataclass(frozen=True, slots=True)
class PromptReevaluationResult:
    """Identity of one atomically published V3 re-evaluation."""

    directory: Path
    manifest_sha256: str
    comparator_profile_sha256: str


def run_prompt_reevaluation(
    staging_directory: str | Path,
    paths: PromptReevaluationPaths,
    comparator: ReevaluationComparator,
) -> PromptReevaluationResult:
    """Build, replay-validate, and publish V3 without any model boundary."""
    staging = Path(staging_directory)
    if not staging.is_dir():
        raise FileNotFoundError(f"re-evaluation staging directory is missing: {staging}")
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"re-evaluation destination exists: {paths.destination}")
    source_roots = _validate_source_artifacts(paths)
    profile = _validate_comparator(comparator)
    builder = Phase1ArtifactBuilder(staging, version=PROMPT_REEVALUATION_VERSION)
    for contract in PROMPT_REEVALUATION_SOURCE_CELLS:
        _snapshot_source(builder, source_roots[contract.label], contract)
    cells = tuple(
        _load_cached_source_cell(builder.root, contract)
        for contract in PROMPT_REEVALUATION_SOURCE_CELLS
    )
    selected = _common_selected_briefs(cells)
    _write_comparator(builder, comparator, profile)
    _write_reevaluation(builder, cells, selected, comparator, profile)
    manifest = builder.finalize()
    validated = validate_prompt_reevaluation(builder.root)
    if validated.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("V3 validation changed artifact identity")
    destination = builder.promote(
        paths.destination,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    return PromptReevaluationResult(
        directory=destination,
        manifest_sha256=manifest.manifest_sha256,
        comparator_profile_sha256=profile.profile_sha256,
    )


def validate_prompt_reevaluation(
    directory: str | Path,
) -> Phase1ArtifactManifest:
    """Authenticate and exactly replay one completed zero-inference V3 tree."""
    root = Path(directory)
    manifest = load_phase1_artifact_tree(root)
    if manifest.version != PROMPT_REEVALUATION_VERSION:
        raise ValueError("artifact is not the V3 cached prompt re-evaluation")
    for contract in PROMPT_REEVALUATION_SOURCE_CELLS:
        _validate_source_snapshot(root, contract)
    cells = tuple(
        _load_cached_source_cell(root, contract)
        for contract in PROMPT_REEVALUATION_SOURCE_CELLS
    )
    selected = _common_selected_briefs(cells)
    comparator, profile = _load_comparator(root)
    expected_configuration = _configuration_record(profile, comparator.audit)
    if _json_object(root / "configuration.json") != expected_configuration:
        raise ValueError("V3 configuration differs from its fixed contract")
    if _json_object(root / "source_bindings.json") != _source_bindings_record():
        raise ValueError("V3 source bindings differ")
    if _jsonl_objects(root / "selected_briefs.jsonl") != tuple(
        _brief_record(brief) for brief in selected
    ):
        raise ValueError("V3 selected briefs differ from cached sources")
    expected_results = tuple(
        _variant_sample_record(cell.contract.target_variant.variant_id, sample)
        for cell in cells
        for sample in cell.samples
    )
    if _jsonl_objects(root / "results.jsonl") != expected_results:
        raise ValueError("V3 cached result projection differs")
    combined = _combined_cached_measurements(cells)
    if _jsonl_objects(root / "measurements.jsonl") != tuple(
        measurement.as_record() for measurement in combined.measurements
    ):
        raise ValueError("V3 cached measurements differ")
    _validate_derived_outputs(root, cells, selected, profile, combined)
    if _json_object(root / "reuse.json") != _reuse_record(cells):
        raise ValueError("V3 zero-inference reuse record differs")
    if _json_object(root / "status.json") != {
        "phase1_gate_effect": "none_diagnostic_development_only",
        "status": "cached_reevaluation_complete",
    }:
        raise ValueError("V3 status differs")
    return manifest


def _validate_source_artifacts(
    paths: PromptReevaluationPaths,
) -> dict[str, Path]:
    roots = {
        "v1-v6": paths.prompt_tuning_v1,
        "v2-v7": paths.prompt_tuning_v2,
    }
    for contract in PROMPT_REEVALUATION_SOURCE_CELLS:
        validated = load_phase1_artifact_tree(roots[contract.label])
        if (
            validated.version != contract.artifact_version
            or validated.manifest_sha256 != contract.artifact_manifest_sha256
        ):
            raise ValueError(f"{contract.label} source artifact identity differs")
    return roots


def _snapshot_source(
    builder: Phase1ArtifactBuilder,
    source: Path,
    contract: SourceCellContract,
) -> None:
    paths = (*_SOURCE_COPY_PATHS, contract.plan_path)
    builder.write_bytes(
        f"sources/{contract.label}/manifest.json",
        (source / "manifest.json").read_bytes(),
    )
    for relative_path in paths:
        builder.write_bytes(
            f"sources/{contract.label}/{relative_path}",
            (source / relative_path).read_bytes(),
        )


def _validate_source_snapshot(
    root: Path,
    contract: SourceCellContract,
) -> Phase1ArtifactManifest:
    snapshot = root / "sources" / contract.label
    record = require_json_object(
        canonical_json_loads(
            (snapshot / "manifest.json").read_bytes(),
            label=f"{contract.label} source manifest",
        ),
        label=f"{contract.label} source manifest",
    )
    manifest = _decode_manifest(record)
    if (
        manifest.version != contract.artifact_version
        or manifest.manifest_sha256 != contract.artifact_manifest_sha256
    ):
        raise ValueError(f"{contract.label} source manifest differs")
    descriptors = {artifact.path: artifact for artifact in manifest.artifacts}
    for relative_path in (*_SOURCE_COPY_PATHS, contract.plan_path):
        descriptor = descriptors.get(relative_path)
        if descriptor is None:
            raise ValueError(f"{contract.label} source manifest omits {relative_path}")
        payload = (snapshot / relative_path).read_bytes()
        if len(payload) != descriptor.size_bytes or sha256(payload).hexdigest() != (
            descriptor.sha256
        ):
            raise ValueError(f"{contract.label} source snapshot differs: {relative_path}")
    return manifest


def _load_cached_source_cell(
    root: Path,
    contract: SourceCellContract,
) -> CachedPromptCell:
    snapshot = root / "sources" / contract.label
    _validate_source_snapshot(root, contract)
    selected = _decode_selected_briefs(snapshot / "selected_briefs.jsonl")
    routes_record = _json_object(snapshot / "catalog" / "routes.json")
    values = routes_record.get("live")
    if type(values) is not list or len(values) != len(TWO_ROUTE_AUTHOR_ORDER):
        raise ValueError(f"{contract.label} live routes differ")
    routes = tuple(
        _route(value, f"{contract.label} live route {index}")
        for index, value in enumerate(values)
    )
    if tuple(route.route_id for route in routes) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError(f"{contract.label} live route order differs")
    jobs = build_prompt_tuning_jobs(
        selected,
        routes,
        variants=(contract.target_variant,),
    )[0]
    expected_plan = tuple(
        {
            **job.request.as_record(),
            "body": job.request.body,
            "brief_id": job.brief.brief_id,
            "route_id": job.route.route_id,
            "variant_id": contract.source_variant_id,
        }
        for job in jobs.jobs
    )
    if _jsonl_objects(snapshot / contract.plan_path) != expected_plan:
        raise ValueError(f"{contract.label} cached request plan differs")
    source_records = _jsonl_objects(snapshot / "generation_results.jsonl")
    if len(source_records) != PROMPT_TUNING_BRIEF_COUNT * len(
        TWO_ROUTE_AUTHOR_ORDER
    ):
        raise ValueError(f"{contract.label} source result coverage differs")
    jobs_by_id = {job.sample_id: job for job in jobs.jobs}
    records_by_id = {}
    for record in source_records:
        if record.get("variant_id") != contract.source_variant_id:
            raise ValueError(f"{contract.label} source result variant differs")
        sample_id = record.get("sample_id")
        if type(sample_id) is not str or sample_id in records_by_id:
            raise ValueError(f"{contract.label} source sample identity differs")
        records_by_id[sample_id] = record
    if set(records_by_id) != set(jobs_by_id):
        raise ValueError(f"{contract.label} source results omit planned samples")
    samples = tuple(
        _decode_control_sample(jobs_by_id[job.sample_id], records_by_id[job.sample_id])
        for job in jobs.jobs
    )
    if any(
        records_by_id[sample.sample_id]
        != _variant_sample_record(contract.source_variant_id, sample)
        for sample in samples
    ):
        raise ValueError(f"{contract.label} source results fail local derivation")
    source_measurements = _load_measurement_batch(snapshot / "measurements.jsonl")
    prefix = f"{contract.source_variant_id}:"
    accepted_ids = {sample.sample_id for sample in samples if sample.validation.accepted}
    measurements = tuple(
        StoryMeasurement(
            item.record_id.removeprefix(prefix),
            item.model_token_ids,
            item.normalized_nll,
            item.active_token_count,
        )
        for item in source_measurements.measurements
        if item.record_id.startswith(prefix)
    )
    if {measurement.record_id for measurement in measurements} != accepted_ids:
        raise ValueError(f"{contract.label} source measurement coverage differs")
    return CachedPromptCell(
        contract=contract,
        jobs=jobs,
        samples=samples,
        measurements=MeasurementBatch(measurements, source_measurements.runtime),
    )


def _common_selected_briefs(
    cells: Sequence[CachedPromptCell],
) -> tuple[NeutralStoryBrief, ...]:
    if len(cells) != 2:
        raise ValueError("V3 requires exactly the cached V6 and V7 cells")
    selected = tuple(job.brief for job in cells[0].jobs.jobs[:PROMPT_TUNING_BRIEF_COUNT])
    # Jobs are route-major, so compare one complete brief block per source.
    other = tuple(job.brief for job in cells[1].jobs.jobs[:PROMPT_TUNING_BRIEF_COUNT])
    if selected != other:
        raise ValueError("V1 and V2 selected prompt briefs differ")
    return selected


def _validate_comparator(comparator: ReevaluationComparator) -> ReferenceProfile:
    expectations = PROMPT_REEVALUATION_EXPECTATIONS
    if len(comparator.records) != expectations.retained_count or len(
        comparator.observations
    ) != expectations.retained_count:
        raise ValueError("V3 comparator retained coverage differs")
    record_ids = tuple(record.record_id for record in comparator.records)
    if record_ids != tuple(observation.record_id for observation in comparator.observations):
        raise ValueError("V3 comparator record/observation order differs")
    digest = sha256(
        "".join(f"{record_id}\n" for record_id in record_ids).encode("utf-8")
    ).hexdigest()
    if digest != expectations.retained_ids_sha256:
        raise ValueError("V3 comparator retained identity digest differs")
    for record, observation in zip(
        comparator.records,
        comparator.observations,
        strict=True,
    ):
        expected = observe_reference(
            ReferenceRecord(
                record_id=record.record_id,
                story_text=record.story,
                source_model="GPT-4",
            ),
            model_token_ids=observation.model_token_ids,
            normalized_nll=observation.normalized_nll,
            feature_labels=observation.feature_labels,
            required_words=observation.required_words,
        )
        if canonical_reference_observation(expected) != canonical_reference_observation(
            observation
        ):
            raise ValueError("V3 comparator observation differs from its story")
    profile = comparator.profile
    if profile.profile_sha256 != expectations.retained_profile_sha256:
        raise ValueError("V3 comparator profile identity differs")
    expected_audit = ValidationDecontaminationAudit(
        input_count=expectations.input_count,
        overlap_count=expectations.overlap_count,
        retained_count=expectations.retained_count,
        overlap_ids_sha256=expectations.overlap_ids_sha256,
        retained_ids_sha256=expectations.retained_ids_sha256,
        retained_profile_sha256=expectations.retained_profile_sha256,
    ).as_record()
    if canonical_json_bytes(comparator.audit) != canonical_json_bytes(expected_audit):
        raise ValueError("V3 comparator audit differs from its fixed contract")
    return profile


def _write_comparator(
    builder: Phase1ArtifactBuilder,
    comparator: ReevaluationComparator,
    profile: ReferenceProfile,
) -> None:
    builder.write_json("comparator/audit.json", comparator.audit)
    builder.write_bytes(
        "comparator/records.jsonl",
        canonical_jsonl_bytes(
            canonical_validation_record(record) for record in comparator.records
        ),
    )
    builder.write_bytes(
        "comparator/observations.jsonl",
        canonical_jsonl_bytes(
            canonical_reference_observation(observation)
            for observation in comparator.observations
        ),
    )
    builder.write_json("comparator/profile.json", _reference_profile_record(profile))


def _load_comparator(
    root: Path,
) -> tuple[ReevaluationComparator, ReferenceProfile]:
    records = tuple(
        _decode_validation_record(record, index)
        for index, record in enumerate(_jsonl_objects(root / "comparator" / "records.jsonl"))
    )
    observations = tuple(
        _decode_reference_observation(record, index)
        for index, record in enumerate(
            _jsonl_objects(root / "comparator" / "observations.jsonl")
        )
    )
    comparator = ReevaluationComparator(
        audit=_json_object(root / "comparator" / "audit.json"),
        records=records,
        observations=observations,
    )
    profile = _validate_comparator(comparator)
    stored_profile = _decode_reference_profile(
        _json_object(root / "comparator" / "profile.json"),
        "V3 comparator profile",
    )
    if stored_profile != profile:
        raise ValueError("V3 comparator stored profile differs from observations")
    return comparator, profile


def _combined_cached_measurements(
    cells: Sequence[CachedPromptCell],
) -> MeasurementBatch:
    measurements = tuple(
        StoryMeasurement(
            f"{cell.contract.target_variant.variant_id}:{measurement.record_id}",
            measurement.model_token_ids,
            measurement.normalized_nll,
            measurement.active_token_count,
        )
        for cell in cells
        for measurement in cell.measurements.measurements
    )
    if len({measurement.record_id for measurement in measurements}) != len(measurements):
        raise ValueError("V3 cached measurement identities repeat")
    return MeasurementBatch(
        tuple(sorted(measurements, key=lambda item: item.record_id)),
        {"status": "exact_cached_v1_v2_measurements"},
    )


def _write_reevaluation(
    builder: Phase1ArtifactBuilder,
    cells: Sequence[CachedPromptCell],
    selected: Sequence[NeutralStoryBrief],
    comparator: ReevaluationComparator,
    profile: ReferenceProfile,
) -> None:
    builder.write_json(
        "configuration.json",
        _configuration_record(profile, comparator.audit),
    )
    builder.write_json("source_bindings.json", _source_bindings_record())
    builder.write_bytes(
        "selected_briefs.jsonl",
        canonical_jsonl_bytes(_brief_record(brief) for brief in selected),
    )
    builder.write_bytes(
        "results.jsonl",
        canonical_jsonl_bytes(
            _variant_sample_record(cell.contract.target_variant.variant_id, sample)
            for cell in cells
            for sample in cell.samples
        ),
    )
    combined = _combined_cached_measurements(cells)
    builder.write_bytes(
        "measurements.jsonl",
        canonical_jsonl_bytes(
            measurement.as_record() for measurement in combined.measurements
        ),
    )
    _write_derived_outputs(builder, cells, selected, profile, combined)
    builder.write_json("reuse.json", _reuse_record(cells))
    builder.write_json(
        "status.json",
        {
            "phase1_gate_effect": "none_diagnostic_development_only",
            "status": "cached_reevaluation_complete",
        },
    )


def _write_derived_outputs(
    builder: Phase1ArtifactBuilder,
    cells: Sequence[CachedPromptCell],
    selected: Sequence[NeutralStoryBrief],
    profile: ReferenceProfile,
    measurements: MeasurementBatch,
) -> None:
    sample_cells = tuple(
        (cell.contract.target_variant, cell.samples) for cell in cells
    )
    reports = _evaluate_cells(
        sample_cells,
        measurements,
        selected,
        None,
        comparison_profile=profile,
    )
    builder.write_json("quality.json", _quality_record(reports))
    description = (
        f"the {profile.record_count:,}-story V2 GPT-4 validation profile after "
        "exact normalized overlap removal against the original TinyStories train "
        f"corpus (profile {profile.profile_sha256})"
    )
    builder.write_bytes(
        "review.html",
        _render_review_html(
            sample_cells,
            selected,
            variants=PROMPT_TUNING_V2_VARIANTS,
            comparison_description=description,
        ).encode("utf-8"),
    )


def _validate_derived_outputs(
    root: Path,
    cells: Sequence[CachedPromptCell],
    selected: Sequence[NeutralStoryBrief],
    profile: ReferenceProfile,
    measurements: MeasurementBatch,
) -> None:
    sample_cells = tuple(
        (cell.contract.target_variant, cell.samples) for cell in cells
    )
    reports = _evaluate_cells(
        sample_cells,
        measurements,
        selected,
        None,
        comparison_profile=profile,
    )
    if _json_object(root / "quality.json") != _quality_record(reports):
        raise ValueError("V3 quality differs from exact cached replay")
    description = (
        f"the {profile.record_count:,}-story V2 GPT-4 validation profile after "
        "exact normalized overlap removal against the original TinyStories train "
        f"corpus (profile {profile.profile_sha256})"
    )
    expected_review = _render_review_html(
        sample_cells,
        selected,
        variants=PROMPT_TUNING_V2_VARIANTS,
        comparison_description=description,
    ).encode("utf-8")
    if (root / "review.html").read_bytes() != expected_review:
        raise ValueError("V3 review differs from cached stories")


def _quality_record(
    reports: Sequence[tuple[str, RouteQualityReport]],
) -> JsonObject:
    variants = tuple(variant.variant_id for variant in PROMPT_TUNING_V2_VARIANTS)
    best = {
        route_id: min(
            (
                (variant_id, report)
                for variant_id, report in reports
                if report.route_id == route_id
            ),
            key=lambda item: (
                item[1].alignment_distance,
                len(item[1].failures),
                variants.index(item[0]),
            ),
        )[0]
        for route_id in TWO_ROUTE_AUTHOR_ORDER
    }
    values = []
    for variant_id, report in reports:
        record = _quality_report_record(report)
        record["variant_id"] = variant_id
        values.append(record)
    return {
        "best_variant_by_route": best,
        "interpretation": (
            "development diagnostic only; exact cached V6/V7 stories are "
            "re-evaluated against decontaminated already-scored validation "
            "evidence; no matched archive reference participates in scoring"
        ),
        "ranking_metric": "minimum_alignment_distance_then_failure_count",
        "reports": values,
    }


def _configuration_record(
    profile: ReferenceProfile,
    audit: JsonObject,
) -> JsonObject:
    return {
        "base_reference_manifest_sha256": (
            TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256
        ),
        "comparator_audit_sha256": sha256(canonical_json_bytes(audit)).hexdigest(),
        "comparator_profile_sha256": profile.profile_sha256,
        "comparator_record_count": profile.record_count,
        "development_data_only": True,
        "new_generated_story_measurements": 0,
        "new_model_calls": 0,
        "source_manifests": {
            contract.label: contract.artifact_manifest_sha256
            for contract in PROMPT_REEVALUATION_SOURCE_CELLS
        },
        "version": PROMPT_REEVALUATION_VERSION,
    }


def _source_bindings_record() -> JsonObject:
    return {
        "cells": [
            {
                "label": contract.label,
                "source_manifest_sha256": contract.artifact_manifest_sha256,
                "source_variant_id": contract.source_variant_id,
                "target_variant_id": contract.target_variant.variant_id,
            }
            for contract in PROMPT_REEVALUATION_SOURCE_CELLS
        ]
    }


def _reuse_record(cells: Sequence[CachedPromptCell]) -> JsonObject:
    return {
        "cached_accepted_measurement_count": sum(
            len(cell.measurements.measurements) for cell in cells
        ),
        "cached_story_count": sum(len(cell.samples) for cell in cells),
        "new_external_generation_cost_usd": "0",
        "new_external_generation_requests": 0,
        "new_generated_story_nll_measurements": 0,
        "new_model_calls": 0,
    }


def main() -> None:
    """Build V3 from authenticated local evidence, or validate it if present."""
    repository_root = Path(__file__).resolve().parents[5]
    paths = PromptReevaluationPaths.from_repository(repository_root)
    if paths.destination.is_dir():
        manifest = validate_prompt_reevaluation(paths.destination)
        print(f"Existing prompt re-evaluation artifact: {paths.destination}")
        print(f"Manifest: {manifest.manifest_sha256}")
        print("No generation, network, API key, or GPU scoring was repeated.")
        return
    comparator = build_production_comparator(paths)
    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="tinyworlds-v2-prompt-reevaluation-",
            dir=paths.destination.parent,
        )
    )
    print(f"Temporary artifact directory: {staging}", flush=True)
    result = run_prompt_reevaluation(staging, paths, comparator)
    print(f"Prompt re-evaluation artifact: {result.directory}")
    print(f"Manifest: {result.manifest_sha256}")
    print(f"Comparator profile: {result.comparator_profile_sha256}")


def build_production_comparator(
    paths: PromptReevaluationPaths,
) -> ReevaluationComparator:
    """Load and decontaminate already-scored validation evidence locally."""
    manifest = load_phase1_artifact_tree(paths.base_reference)
    if manifest.manifest_sha256 != TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256:
        raise ValueError("V3 base reference artifact identity differs")
    records = tuple(
        _decode_validation_record(record, index)
        for index, record in enumerate(
            _jsonl_objects(paths.base_reference / "validation_source_sample.jsonl")
        )
    )
    observations = _load_reference_observations(
        paths.base_reference / "reference_observations.jsonl"
    )
    result = build_decontaminated_validation_profile(
        paths.original_train,
        records,
        observations,
    )
    records_by_id = {record.record_id: record for record in records}
    retained_records = tuple(
        records_by_id[record_id] for record_id in result.retained_record_ids
    )
    comparator = ReevaluationComparator(
        audit=result.audit.as_record(),
        records=retained_records,
        observations=result.retained_observations,
    )
    if _validate_comparator(comparator) != result.profile:
        raise ValueError("V3 production comparator changed during adaptation")
    return comparator


__all__ = [
    "PROMPT_REEVALUATION_EXPECTATIONS",
    "PROMPT_REEVALUATION_RETAINED_COUNT",
    "PROMPT_REEVALUATION_RETAINED_IDS_SHA256",
    "PROMPT_REEVALUATION_SOURCE_CELLS",
    "PROMPT_REEVALUATION_VERSION",
    "CachedPromptCell",
    "PromptReevaluationPaths",
    "PromptReevaluationResult",
    "ReevaluationComparator",
    "SourceCellContract",
    "build_production_comparator",
    "main",
    "run_prompt_reevaluation",
    "validate_prompt_reevaluation",
]
