"""Canonical reusable persistence for fully rendered TinyWorlds corpora.

Token arrays are intentionally not stored.  The loader reconstructs every
story token sequence and every ``KnowledgeQuery`` batch from the supplied
tokenizer, then checks canonical token hashes, semantic references, exact
prefix/suffix boundaries, split isolation, and symbolic-bundle identity.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, TypeAlias, TypeVar

import numpy as np

from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import build_prefix_suffix_batches
from apm.data.text.tinyworlds.persistence import tinyworlds_bundle_sha256
from apm.data.text.tinyworlds.query_generation import TinyWorldsBundle
from apm.data.text.tinyworlds.rendering import (
    TINYWORLDS_CONTEXT_LENGTH,
    TINYWORLDS_PREFIX_LENGTHS,
    RenderedQueryGroup,
    RenderedQueryVariant,
    RenderedStory,
    RenderedTinyWorlds,
    SentenceAlignment,
    TinyWorldsRenderPreset,
    QueryGroupPlan,
    _render_query_group_at_index,
    _render_root_story,
    _render_story,
    _source_query_plans,
)
from apm.data.text.tinyworlds.schema import (
    CandidateRole,
    DataSplit,
    EntityId,
    QueryCandidate,
    QueryKind,
    QueryPlan,
    TaskId,
)
from apm.data.text.tinyworlds.templates import build_template_registry
from apm.lm.text import TextTokenizer
from apm.memory.graph import NodeId


_FORMAT = "apm.tinyworlds.rendered-bundle"
_SCHEMA_VERSION = 2
_ARTIFACT_PATHS = ("metadata.json", "stories.jsonl", "query_groups.jsonl")
JsonObject: TypeAlias = dict[str, object]


class RenderedTinyWorldsBundleError(ValueError):
    """A rendered bundle is malformed or fails deterministic reconstruction."""


@dataclass(frozen=True, slots=True)
class RenderedBundleFileDigest:
    """Checksum, byte size, and record count for one rendered artifact."""

    path: str
    sha256: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        if type(self.path) is not str or self.path not in _ARTIFACT_PATHS:
            raise RenderedTinyWorldsBundleError(
                f"unknown rendered artifact path: {self.path!r}"
            )
        _require_sha256(self.sha256, f"{self.path} sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise RenderedTinyWorldsBundleError("size_bytes must be nonnegative")
        if type(self.record_count) is not int or self.record_count < 0:
            raise RenderedTinyWorldsBundleError("record_count must be nonnegative")


@dataclass(frozen=True, slots=True)
class RenderedTinyWorldsManifest:
    """Self-authenticating manifest for a rendered artifact tree."""

    format: str
    schema_version: int
    rendered_bundle_id: str
    symbolic_bundle_sha256: str
    artifacts: tuple[RenderedBundleFileDigest, ...]
    bundle_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.format) is not str
            or self.format != _FORMAT
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise RenderedTinyWorldsBundleError("unsupported rendered bundle format")
        if type(self.rendered_bundle_id) is not str or not self.rendered_bundle_id:
            raise RenderedTinyWorldsBundleError("rendered_bundle_id must be nonempty")
        _require_sha256(self.symbolic_bundle_sha256, "symbolic_bundle_sha256")
        if type(self.artifacts) is not tuple or any(
            type(item) is not RenderedBundleFileDigest for item in self.artifacts
        ):
            raise RenderedTinyWorldsBundleError("manifest artifacts have invalid types")
        if tuple(item.path for item in self.artifacts) != _ARTIFACT_PATHS:
            raise RenderedTinyWorldsBundleError(
                "manifest must list the exact rendered artifact set and order"
            )
        _require_sha256(self.bundle_sha256, "bundle_sha256")
        if self.bundle_sha256 != _digest(_canonical_json(_manifest_core(self))):
            raise RenderedTinyWorldsBundleError("rendered manifest SHA-256 mismatch")


def write_rendered_tinyworlds_bundle(
    rendered: RenderedTinyWorlds,
    symbolic_bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    directory: str | Path,
) -> RenderedTinyWorldsManifest:
    """Validate and atomically persist a new rendered artifact tree."""
    _validate_inputs(rendered, symbolic_bundle, tokenizer)
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"rendered TinyWorlds target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    story_records = tuple(
        _encode_story(story, symbolic_bundle, rendered, tokenizer)
        for story in rendered.stories
    )
    group_records = tuple(
        _encode_group(group, symbolic_bundle, tokenizer)
        for group in rendered.query_groups
    )
    tokenization_sha256 = _tokenization_fingerprint(story_records, group_records)
    metadata = _metadata_record(
        rendered,
        symbolic_bundle,
        tokenizer,
        tokenization_sha256,
    )
    payloads = {
        "metadata.json": _canonical_json(metadata),
        "stories.jsonl": _canonical_jsonl(story_records),
        "query_groups.jsonl": _canonical_jsonl(group_records),
    }
    counts = {
        "metadata.json": 1,
        "stories.jsonl": len(story_records),
        "query_groups.jsonl": len(group_records),
    }
    artifacts = tuple(
        RenderedBundleFileDigest(
            path=path,
            sha256=_digest(payloads[path]),
            size_bytes=len(payloads[path]),
            record_count=counts[path],
        )
        for path in _ARTIFACT_PATHS
    )
    symbolic_sha256 = tinyworlds_bundle_sha256(symbolic_bundle)
    core = _manifest_core_values(
        rendered.bundle_id,
        symbolic_sha256,
        artifacts,
    )
    manifest = RenderedTinyWorldsManifest(
        format=_FORMAT,
        schema_version=_SCHEMA_VERSION,
        rendered_bundle_id=rendered.bundle_id,
        symbolic_bundle_sha256=symbolic_sha256,
        artifacts=artifacts,
        bundle_sha256=_digest(_canonical_json(core)),
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        for path, payload in payloads.items():
            (temporary / path).write_bytes(payload)
        (temporary / "manifest.json").write_bytes(
            _canonical_json(_manifest_record(manifest))
        )
        os.rename(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def load_rendered_tinyworlds_manifest(
    directory: str | Path,
) -> RenderedTinyWorldsManifest:
    """Load the manifest and validate the exact artifact tree and digests."""
    root = Path(directory)
    record = _read_canonical_json(root / "manifest.json")
    manifest = _decode_manifest(record)
    actual_paths = tuple(
        sorted(path.name for path in root.iterdir() if path.name != "manifest.json")
    )
    if actual_paths != tuple(sorted(_ARTIFACT_PATHS)):
        raise RenderedTinyWorldsBundleError(
            "rendered directory contains missing or unlisted artifacts"
        )
    for artifact in manifest.artifacts:
        path = root / artifact.path
        if path.is_symlink() or not path.is_file():
            raise RenderedTinyWorldsBundleError(
                f"rendered artifact must be a regular file: {artifact.path}"
            )
        payload = path.read_bytes()
        if len(payload) != artifact.size_bytes:
            raise RenderedTinyWorldsBundleError(
                f"rendered artifact size mismatch: {artifact.path}"
            )
        if _digest(payload) != artifact.sha256:
            raise RenderedTinyWorldsBundleError(
                f"rendered artifact digest mismatch: {artifact.path}"
            )
    return manifest


def load_rendered_tinyworlds_bundle(
    directory: str | Path,
    symbolic_bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> RenderedTinyWorlds:
    """Reconstruct all arrays and strictly validate a persisted rendered bundle."""
    if type(symbolic_bundle) is not TinyWorldsBundle:
        raise TypeError("symbolic_bundle must be a TinyWorldsBundle")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    _validate_tokenizer_contract(tokenizer)
    root = Path(directory)
    manifest = load_rendered_tinyworlds_manifest(root)
    expected_symbolic_sha = tinyworlds_bundle_sha256(symbolic_bundle)
    if manifest.symbolic_bundle_sha256 != expected_symbolic_sha:
        raise RenderedTinyWorldsBundleError(
            "rendered artifact references a different symbolic bundle"
        )
    artifact_by_path = {item.path: item for item in manifest.artifacts}
    metadata = _read_canonical_json(root / "metadata.json")
    if artifact_by_path["metadata.json"].record_count != 1:
        raise RenderedTinyWorldsBundleError("metadata record count must equal one")
    preset, expected_tokenization_sha = _decode_and_validate_metadata(
        metadata,
        manifest,
        symbolic_bundle,
        tokenizer,
    )
    registry = build_template_registry(symbolic_bundle.world)
    story_records, stories = _load_jsonl_with_records(
        root / "stories.jsonl",
        artifact_by_path["stories.jsonl"],
        lambda item: _decode_story(
            item,
            symbolic_bundle,
            registry,
            preset,
            tokenizer,
        ),
    )
    group_records, groups = _load_jsonl_with_records(
        root / "query_groups.jsonl",
        artifact_by_path["query_groups.jsonl"],
        lambda item: _decode_group(item, symbolic_bundle, tokenizer),
    )
    try:
        rendered = RenderedTinyWorlds(
            bundle_id=manifest.rendered_bundle_id,
            registry=registry,
            preset=preset,
            stories=stories,
            query_groups=groups,
        )
        _validate_rendered_against_symbolic(
            rendered,
            symbolic_bundle,
            tokenizer,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RenderedTinyWorldsBundleError(
            f"rendered bundle validation failed: {error}"
        ) from error
    actual_tokenization_sha = _tokenization_fingerprint(
        story_records,
        group_records,
    )
    if actual_tokenization_sha != expected_tokenization_sha:
        raise RenderedTinyWorldsBundleError(
            "rendered tokenization fingerprint does not match metadata"
        )
    return rendered


def rendered_tokenization_sha256(
    rendered: RenderedTinyWorlds,
    symbolic_bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> str:
    """Return the canonical tokenizer-behavior digest for a rendered object."""
    _validate_inputs(rendered, symbolic_bundle, tokenizer)
    stories = tuple(
        _encode_story(story, symbolic_bundle, rendered, tokenizer)
        for story in rendered.stories
    )
    groups = tuple(
        _encode_group(group, symbolic_bundle, tokenizer)
        for group in rendered.query_groups
    )
    return _tokenization_fingerprint(stories, groups)


def _validate_inputs(
    rendered: RenderedTinyWorlds,
    symbolic_bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> None:
    if type(rendered) is not RenderedTinyWorlds:
        raise TypeError("rendered must be a RenderedTinyWorlds")
    if type(symbolic_bundle) is not TinyWorldsBundle:
        raise TypeError("symbolic_bundle must be a TinyWorldsBundle")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    _validate_tokenizer_contract(tokenizer)
    _validate_rendered_against_symbolic(rendered, symbolic_bundle, tokenizer)


def _validate_tokenizer_contract(tokenizer: TextTokenizer) -> None:
    if type(tokenizer.vocab_size) is not int or tokenizer.vocab_size <= 0:
        raise ValueError("tokenizer vocab_size must be a positive integer")
    for label, token_id in (
        ("pad_token_id", tokenizer.pad_token_id),
        ("eos_token_id", tokenizer.eos_token_id),
    ):
        if (
            type(token_id) is not int
            or token_id < 0
            or token_id >= tokenizer.vocab_size
        ):
            raise ValueError(f"tokenizer {label} must belong to its vocabulary")


def _metadata_record(
    rendered: RenderedTinyWorlds,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    tokenization_sha256: str,
) -> JsonObject:
    return {
        "counts": {
            "query_groups": len(rendered.query_groups),
            "query_variants": sum(
                len(group.variants) for group in rendered.query_groups
            ),
            "stories": len(rendered.stories),
        },
        "preset": _encode_preset(rendered.preset),
        "rendered_bundle_id": rendered.bundle_id,
        "symbolic_bundle": {
            "bundle_id": symbolic.bundle_id,
            "master_seed_sha256": symbolic.world.master_seed_sha256,
            "version": symbolic.version,
            "world_id": symbolic.world.world_id,
        },
        "template_registry_version": rendered.registry.version,
        "tokenizer": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "tokenization_sha256": tokenization_sha256,
            "vocab_size": tokenizer.vocab_size,
        },
    }


def _decode_and_validate_metadata(
    record: object,
    manifest: RenderedTinyWorldsManifest,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> tuple[TinyWorldsRenderPreset, str]:
    data = _fields(
        record,
        (
            "counts",
            "preset",
            "rendered_bundle_id",
            "symbolic_bundle",
            "template_registry_version",
            "tokenizer",
        ),
        "rendered metadata",
    )
    if _string(data["rendered_bundle_id"], "rendered_bundle_id") != (
        manifest.rendered_bundle_id
    ):
        raise RenderedTinyWorldsBundleError("metadata rendered bundle ID mismatch")
    registry = build_template_registry(symbolic.world)
    if _string(data["template_registry_version"], "template version") != (
        registry.version
    ):
        raise RenderedTinyWorldsBundleError("template registry version mismatch")
    symbolic_data = _fields(
        data["symbolic_bundle"],
        ("bundle_id", "master_seed_sha256", "version", "world_id"),
        "symbolic bundle metadata",
    )
    expected_symbolic = {
        "bundle_id": symbolic.bundle_id,
        "master_seed_sha256": symbolic.world.master_seed_sha256,
        "version": symbolic.version,
        "world_id": symbolic.world.world_id,
    }
    if symbolic_data != expected_symbolic:
        raise RenderedTinyWorldsBundleError("symbolic bundle metadata mismatch")
    tokenizer_data = _fields(
        data["tokenizer"],
        ("eos_token_id", "pad_token_id", "tokenization_sha256", "vocab_size"),
        "tokenizer metadata",
    )
    for field_name, expected in (
        ("eos_token_id", tokenizer.eos_token_id),
        ("pad_token_id", tokenizer.pad_token_id),
        ("vocab_size", tokenizer.vocab_size),
    ):
        if _integer(tokenizer_data[field_name], field_name) != expected:
            raise RenderedTinyWorldsBundleError(
                f"supplied tokenizer {field_name} does not match metadata"
            )
    tokenization_sha = _string(
        tokenizer_data["tokenization_sha256"],
        "tokenization_sha256",
    )
    _require_sha256(tokenization_sha, "tokenization_sha256")
    counts = _fields(
        data["counts"],
        ("query_groups", "query_variants", "stories"),
        "rendered counts",
    )
    story_count = _integer(counts["stories"], "story count")
    group_count = _integer(counts["query_groups"], "query group count")
    variant_count = _integer(counts["query_variants"], "query variant count")
    artifacts = {item.path: item for item in manifest.artifacts}
    if story_count != artifacts["stories.jsonl"].record_count:
        raise RenderedTinyWorldsBundleError("metadata story count mismatch")
    if group_count != artifacts["query_groups.jsonl"].record_count:
        raise RenderedTinyWorldsBundleError("metadata query group count mismatch")
    if variant_count != group_count * len(TINYWORLDS_PREFIX_LENGTHS):
        raise RenderedTinyWorldsBundleError("metadata query variant count mismatch")
    return _decode_preset(data["preset"]), tokenization_sha


def _encode_preset(preset: TinyWorldsRenderPreset) -> dict[str, int]:
    return {
        "context_length": preset.context_length,
        "root_validation_stories": preset.root_validation_stories,
        "story_token_count": preset.story_token_count,
        "test_query_groups_per_task": preset.test_query_groups_per_task,
        "test_stories_per_task": preset.test_stories_per_task,
        "training_stories_per_task": preset.training_stories_per_task,
        "validation_query_groups_per_task": preset.validation_query_groups_per_task,
        "validation_stories_per_task": preset.validation_stories_per_task,
    }


def _decode_preset(record: object) -> TinyWorldsRenderPreset:
    keys = (
        "context_length",
        "root_validation_stories",
        "story_token_count",
        "test_query_groups_per_task",
        "test_stories_per_task",
        "training_stories_per_task",
        "validation_query_groups_per_task",
        "validation_stories_per_task",
    )
    data = _fields(record, keys, "render preset")
    return TinyWorldsRenderPreset(
        training_stories_per_task=_integer(
            data["training_stories_per_task"], "training stories"
        ),
        validation_stories_per_task=_integer(
            data["validation_stories_per_task"], "validation stories"
        ),
        test_stories_per_task=_integer(data["test_stories_per_task"], "test stories"),
        validation_query_groups_per_task=_integer(
            data["validation_query_groups_per_task"], "validation query groups"
        ),
        test_query_groups_per_task=_integer(
            data["test_query_groups_per_task"], "test query groups"
        ),
        root_validation_stories=_integer(
            data["root_validation_stories"], "root validation stories"
        ),
        story_token_count=_integer(data["story_token_count"], "story token count"),
        context_length=_integer(data["context_length"], "context length"),
    )


def _encode_story(
    story: RenderedStory,
    symbolic: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
    tokenizer: TextTokenizer,
) -> JsonObject:
    encoded = tokenizer.encode(story.text)
    if encoded != story.token_ids:
        raise RenderedTinyWorldsBundleError(
            f"story tokenization differs from supplied tokenizer: {story.story_id}"
        )
    _validate_story_symbolic(story, symbolic, rendered.registry)
    return {
        "alignments": [_encode_alignment(item) for item in story.alignments],
        "plot_id": story.plot_id,
        "purpose": story.purpose,
        "split": story.split.value,
        "story_id": story.story_id,
        "task_id": story.task_id,
        "template_family_ids": list(story.template_family_ids),
        "text": story.text,
        "text_sha256": story.text_sha256,
        "token_count": len(encoded),
        "token_ids_sha256": _token_ids_sha256(encoded),
    }


def _decode_story(
    record: object,
    symbolic: TinyWorldsBundle,
    registry,
    preset: TinyWorldsRenderPreset,
    tokenizer: TextTokenizer,
) -> RenderedStory:
    keys = (
        "alignments",
        "plot_id",
        "purpose",
        "split",
        "story_id",
        "task_id",
        "template_family_ids",
        "text",
        "text_sha256",
        "token_count",
        "token_ids_sha256",
    )
    data = _fields(record, keys, "rendered story")
    text = _string(data["text"], "story text")
    tokens = tokenizer.encode(text)
    if len(tokens) != _integer(data["token_count"], "story token_count"):
        raise RenderedTinyWorldsBundleError("story token count mismatch")
    if _token_ids_sha256(tokens) != _string(
        data["token_ids_sha256"], "story token_ids_sha256"
    ):
        raise RenderedTinyWorldsBundleError("story token hash mismatch")
    task_value = data["task_id"]
    if task_value is not None and type(task_value) is not str:
        raise RenderedTinyWorldsBundleError("story task_id must be string or null")
    story = RenderedStory(
        story_id=_string(data["story_id"], "story_id"),
        task_id=task_value,
        split=DataSplit(_string(data["split"], "story split")),
        purpose=_string(data["purpose"], "story purpose"),  # type: ignore[arg-type]
        text=text,
        token_ids=tokens,
        template_family_ids=tuple(
            _string(item, "template family ID")
            for item in _list(data["template_family_ids"], "template_family_ids")
        ),
        plot_id=_string(data["plot_id"], "plot_id"),
        alignments=tuple(
            _decode_alignment(item)
            for item in _list(data["alignments"], "story alignments")
        ),
        text_sha256=_string(data["text_sha256"], "story text_sha256"),
    )
    _validate_story_symbolic(story, symbolic, registry)
    if len(tokens) != preset.story_token_count:
        raise RenderedTinyWorldsBundleError("story token count differs from preset")
    return story


def _encode_alignment(value: SentenceAlignment) -> JsonObject:
    return {
        "end_character": value.end_character,
        "fact_ids": list(value.fact_ids),
        "rule_ids": list(value.rule_ids),
        "sentence_index": value.sentence_index,
        "start_character": value.start_character,
    }


def _decode_alignment(record: object) -> SentenceAlignment:
    data = _fields(
        record,
        ("end_character", "fact_ids", "rule_ids", "sentence_index", "start_character"),
        "sentence alignment",
    )
    return SentenceAlignment(
        sentence_index=_integer(data["sentence_index"], "sentence_index"),
        start_character=_integer(data["start_character"], "start_character"),
        end_character=_integer(data["end_character"], "end_character"),
        fact_ids=tuple(
            _string(item, "alignment fact ID")
            for item in _list(data["fact_ids"], "alignment fact_ids")
        ),
        rule_ids=tuple(
            _string(item, "alignment rule ID")
            for item in _list(data["rule_ids"], "alignment rule_ids")
        ),
    )


def _encode_group(
    group: RenderedQueryGroup,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> JsonObject:
    plan = _query_plan_for_group(symbolic, group)
    _validate_group_core(group)
    variants = tuple(
        _encode_variant(variant, group, plan, symbolic, tokenizer)
        for variant in group.variants
    )
    return {
        "group_id": group.group_id,
        "group_plan": _encode_group_plan(group.group_plan),
        "split": group.split.value,
        "symbolic_query_id": group.symbolic_query_id,
        "task_id": group.task_id,
        "variants": list(variants),
    }


def _decode_group(
    record: object,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> RenderedQueryGroup:
    data = _fields(
        record,
        (
            "group_id",
            "group_plan",
            "split",
            "symbolic_query_id",
            "task_id",
            "variants",
        ),
        "rendered query group",
    )
    group_id = _string(data["group_id"], "group_id")
    task_id = _string(data["task_id"], "group task_id")
    split = DataSplit(_string(data["split"], "group split"))
    symbolic_query_id = _string(data["symbolic_query_id"], "symbolic_query_id")
    group_plan = _decode_group_plan(data["group_plan"], symbolic)
    shell = _QueryGroupShell(
        group_id,
        task_id,
        split,
        symbolic_query_id,
        group_plan,
    )
    plan = group_plan.source_plan
    variants = tuple(
        _decode_variant(item, shell, plan, symbolic, tokenizer)
        for item in _list(data["variants"], "query variants")
    )
    group = RenderedQueryGroup(
        group_id=group_id,
        task_id=task_id,
        split=split,
        symbolic_query_id=symbolic_query_id,
        group_plan=group_plan,
        variants=variants,
    )
    _validate_group_core(group)
    return group


@dataclass(frozen=True, slots=True)
class _QueryGroupShell:
    group_id: str
    task_id: str
    split: DataSplit
    symbolic_query_id: str
    group_plan: QueryGroupPlan


def _encode_group_plan(plan: QueryGroupPlan) -> JsonObject:
    return {
        "candidates": [
            {
                "entity_id": str(candidate.entity_id),
                "role": candidate.role.value,
            }
            for candidate in plan.candidates
        ],
        "correct_index": plan.correct_index,
        "group_id": plan.group_id,
        "holdout_identity_sha256": plan.holdout_identity_sha256,
        "occurrence_index": plan.occurrence_index,
        "replication_index": plan.replication_index,
        "source_proof_id": plan.source_proof_id,
        "source_query_id": plan.source_query_id,
        "split": plan.split.value,
        "task_id": str(plan.task_id),
        "template_family_id": plan.template_family_id,
    }


def _decode_group_plan(
    record: object,
    symbolic: TinyWorldsBundle,
) -> QueryGroupPlan:
    data = _fields(
        record,
        (
            "candidates",
            "correct_index",
            "group_id",
            "holdout_identity_sha256",
            "occurrence_index",
            "replication_index",
            "source_proof_id",
            "source_query_id",
            "split",
            "task_id",
            "template_family_id",
        ),
        "query-group plan",
    )
    task_id = TaskId(_string(data["task_id"], "query-group task_id"))
    split = DataSplit(_string(data["split"], "query-group split"))
    source_query_id = _string(
        data["source_query_id"],
        "query-group source_query_id",
    )
    matches = tuple(
        plan
        for plan in symbolic.query_plans
        if plan.task_id == task_id
        and plan.split is split
        and str(plan.query_ast.query_id) == source_query_id
    )
    if len(matches) != 1:
        raise RenderedTinyWorldsBundleError(
            "query-group plan does not resolve one symbolic source plan"
        )
    source_plan = matches[0]
    if _string(data["source_proof_id"], "query-group source_proof_id") != str(
        source_plan.proof.proof_id
    ):
        raise RenderedTinyWorldsBundleError(
            "query-group source proof differs from its symbolic plan"
        )
    candidates = tuple(
        _decode_group_plan_candidate(candidate)
        for candidate in _list(data["candidates"], "query-group candidates")
    )
    return QueryGroupPlan(
        group_id=_string(data["group_id"], "query-group plan ID"),
        task_id=task_id,
        split=split,
        occurrence_index=_integer(
            data["occurrence_index"],
            "query-group occurrence_index",
        ),
        replication_index=_integer(
            data["replication_index"],
            "query-group replication_index",
        ),
        source_plan=source_plan,
        candidates=candidates,
        correct_index=_integer(data["correct_index"], "query-group correct_index"),
        template_family_id=_string(
            data["template_family_id"],
            "query-group template_family_id",
        ),
        holdout_identity_sha256=_string(
            data["holdout_identity_sha256"],
            "query-group holdout_identity_sha256",
        ),
    )


def _decode_group_plan_candidate(record: object) -> QueryCandidate:
    data = _fields(
        record,
        ("entity_id", "role"),
        "query-group plan candidate",
    )
    return QueryCandidate(
        entity_id=EntityId(
            _string(data["entity_id"], "query-group candidate entity_id")
        ),
        role=CandidateRole(_string(data["role"], "query-group candidate role")),
    )


def _encode_variant(
    variant: RenderedQueryVariant,
    group,
    plan: QueryPlan,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> JsonObject:
    _validate_variant_symbolic(variant, group, plan, symbolic, tokenizer)
    prefix_tokens = tokenizer.encode(variant.prefix_text)
    candidates = []
    for entity_id, candidate in zip(
        variant.candidate_entity_ids,
        variant.knowledge_query.candidates,
    ):
        combined = tokenizer.encode(variant.prefix_text + candidate.answer_text)
        suffix_count = len(combined) - len(prefix_tokens)
        candidates.append(
            {
                "answer_text": candidate.answer_text,
                "answer_text_sha256": _text_sha256(candidate.answer_text),
                "combined_token_count": len(combined),
                "combined_token_ids_sha256": _token_ids_sha256(combined),
                "entity_id": entity_id,
                "suffix_token_count": suffix_count,
            }
        )
    query = variant.knowledge_query
    return {
        "candidates": candidates,
        "prefix_text": variant.prefix_text,
        "prefix_token_count": len(prefix_tokens),
        "prefix_token_ids_sha256": _token_ids_sha256(prefix_tokens),
        "query_core_sha256": variant.query_core_sha256,
        "semantic": {
            "correct_candidate_index": query.correct_candidate_index,
            "cue_regime": query.cue_regime,
            "eligible_task_ids": list(query.eligible_task_ids),
            "family_id": query.family_id,
            "mode": query.mode,
            "novelty_regime": query.novelty_regime,
            "oracle_node_ids": list(query.oracle_node_ids),
            "prefix_length": query.prefix_length,
            "proof_id": query.proof_id,
            "query_id": query.query_id,
            "query_kind": query.query_kind,
            "reasoning_depth": query.reasoning_depth,
            "reasoning_type": query.reasoning_type,
            "required_edge_ids": list(query.required_edge_ids),
            "support_ids": list(query.support_ids),
            "task_id": query.task_id,
            "visible_cue_ids": list(query.visible_cue_ids),
        },
        "split": variant.split.value,
        "text_sha256": variant.text_sha256,
        "variant_id": variant.variant_id,
    }


def _decode_variant(
    record: object,
    group,
    plan: QueryPlan,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> RenderedQueryVariant:
    data = _fields(
        record,
        (
            "candidates",
            "prefix_text",
            "prefix_token_count",
            "prefix_token_ids_sha256",
            "query_core_sha256",
            "semantic",
            "split",
            "text_sha256",
            "variant_id",
        ),
        "rendered query variant",
    )
    prefix_text = _string(data["prefix_text"], "prefix_text")
    prefix_tokens = tokenizer.encode(prefix_text)
    prefix_count = _integer(data["prefix_token_count"], "prefix_token_count")
    if len(prefix_tokens) != prefix_count:
        raise RenderedTinyWorldsBundleError("query prefix token count mismatch")
    if _token_ids_sha256(prefix_tokens) != _string(
        data["prefix_token_ids_sha256"], "prefix_token_ids_sha256"
    ):
        raise RenderedTinyWorldsBundleError("query prefix token hash mismatch")
    candidate_records = _list(data["candidates"], "rendered candidates")
    if len(candidate_records) != 4:
        raise RenderedTinyWorldsBundleError("query variant requires four candidates")
    combined_sequences: list[tuple[int, ...]] = []
    candidate_texts: list[str] = []
    entity_ids: list[str] = []
    suffix_counts: list[int] = []
    for candidate_record in candidate_records:
        candidate_data = _fields(
            candidate_record,
            (
                "answer_text",
                "answer_text_sha256",
                "combined_token_count",
                "combined_token_ids_sha256",
                "entity_id",
                "suffix_token_count",
            ),
            "rendered candidate",
        )
        answer = _string(candidate_data["answer_text"], "candidate answer_text")
        if _text_sha256(answer) != _string(
            candidate_data["answer_text_sha256"], "answer_text_sha256"
        ):
            raise RenderedTinyWorldsBundleError("candidate answer text hash mismatch")
        combined = tokenizer.encode(prefix_text + answer)
        if len(combined) != _integer(
            candidate_data["combined_token_count"], "combined_token_count"
        ):
            raise RenderedTinyWorldsBundleError("candidate combined token count mismatch")
        if _token_ids_sha256(combined) != _string(
            candidate_data["combined_token_ids_sha256"],
            "combined_token_ids_sha256",
        ):
            raise RenderedTinyWorldsBundleError("candidate combined token hash mismatch")
        suffix_count = _integer(
            candidate_data["suffix_token_count"], "suffix_token_count"
        )
        if suffix_count != len(combined) - prefix_count:
            raise RenderedTinyWorldsBundleError("candidate suffix boundary mismatch")
        if combined[:prefix_count] != prefix_tokens:
            raise RenderedTinyWorldsBundleError(
                "standalone query prefix is not a combined-sequence prefix"
            )
        combined_sequences.append(combined)
        candidate_texts.append(answer)
        entity_ids.append(_string(candidate_data["entity_id"], "candidate entity_id"))
        suffix_counts.append(suffix_count)
    if len(set(suffix_counts)) != 1 or suffix_counts[0] < 1:
        raise RenderedTinyWorldsBundleError(
            "candidate suffix token counts must be equal and positive"
        )
    if any(len(sequence) > TINYWORLDS_CONTEXT_LENGTH for sequence in combined_sequences):
        raise RenderedTinyWorldsBundleError("candidate sequence exceeds context length")
    batches = tuple(
        build_prefix_suffix_batches(
            sequence,
            prefix_count,
            suffix_counts[0],
            pad_token_id=tokenizer.pad_token_id,
        )
        for sequence in combined_sequences
    )
    semantic = _decode_semantic(data["semantic"])
    candidates = tuple(
        KnowledgeCandidate(answer, competence)
        for answer, (_, competence) in zip(candidate_texts, batches)
    )
    query = KnowledgeQuery(
        query_id=semantic["query_id"],
        task_id=semantic["task_id"],
        family_id=semantic["family_id"],
        query_kind=semantic["query_kind"],
        candidates=candidates,
        router_batch=batches[0][0],
        correct_candidate_index=semantic["correct_candidate_index"],
        proof_id=semantic["proof_id"],
        support_ids=semantic["support_ids"],
        required_edge_ids=tuple(NodeId(item) for item in semantic["required_edge_ids"]),
        cue_regime=semantic["cue_regime"],
        visible_cue_ids=semantic["visible_cue_ids"],
        eligible_task_ids=tuple(TaskId(item) for item in semantic["eligible_task_ids"]),
        novelty_regime=semantic["novelty_regime"],
        reasoning_type=semantic["reasoning_type"],
        reasoning_depth=semantic["reasoning_depth"],
        prefix_length=semantic["prefix_length"],
        mode=semantic["mode"],
        oracle_node_ids=tuple(NodeId(item) for item in semantic["oracle_node_ids"]),
    )
    variant = RenderedQueryVariant(
        variant_id=_string(data["variant_id"], "variant_id"),
        group_id=group.group_id,
        split=DataSplit(_string(data["split"], "variant split")),
        prefix_text=prefix_text,
        prefix_token_ids=prefix_tokens,
        query_core_sha256=_string(data["query_core_sha256"], "query_core_sha256"),
        candidate_entity_ids=tuple(entity_ids),
        knowledge_query=query,
        text_sha256=_string(data["text_sha256"], "variant text_sha256"),
    )
    _validate_variant_symbolic(variant, group, plan, symbolic, tokenizer)
    return variant


def _decode_semantic(record: object) -> JsonObject:
    keys = (
        "correct_candidate_index",
        "cue_regime",
        "eligible_task_ids",
        "family_id",
        "mode",
        "novelty_regime",
        "oracle_node_ids",
        "prefix_length",
        "proof_id",
        "query_id",
        "query_kind",
        "reasoning_depth",
        "reasoning_type",
        "required_edge_ids",
        "support_ids",
        "task_id",
        "visible_cue_ids",
    )
    data = _fields(record, keys, "query semantic metadata")
    tuple_fields = (
        "eligible_task_ids",
        "oracle_node_ids",
        "required_edge_ids",
        "support_ids",
        "visible_cue_ids",
    )
    result: JsonObject = {
        key: tuple(
            _string(item, f"semantic {key}")
            for item in _list(data[key], f"semantic {key}")
        )
        for key in tuple_fields
    }
    for key in (
        "cue_regime",
        "family_id",
        "mode",
        "novelty_regime",
        "proof_id",
        "query_id",
        "query_kind",
        "reasoning_type",
        "task_id",
    ):
        result[key] = _string(data[key], f"semantic {key}")
    result["correct_candidate_index"] = _integer(
        data["correct_candidate_index"], "correct_candidate_index"
    )
    result["prefix_length"] = _integer(data["prefix_length"], "prefix_length")
    result["reasoning_depth"] = _integer(
        data["reasoning_depth"], "reasoning_depth"
    )
    return result


def _validate_rendered_against_symbolic(
    rendered: RenderedTinyWorlds,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> None:
    expected_registry = build_template_registry(symbolic.world)
    if rendered.registry != expected_registry:
        raise RenderedTinyWorldsBundleError(
            "rendered template registry does not match symbolic bundle"
        )
    if rendered.bundle_id != f"{symbolic.bundle_id}:{expected_registry.version}":
        raise RenderedTinyWorldsBundleError("rendered bundle ID mismatch")
    preset = rendered.preset
    plans_by_story_key = {
        (str(plan.task_id), plan.split): plan for plan in symbolic.story_plans
    }
    expected_story_ids = tuple(
        f"rendered:{plans_by_story_key[(str(task.task_id), split)].story_id}:"
        f"{index:04d}"
        for task in symbolic.tasks
        for split, count in (
            (DataSplit.TRAIN, preset.training_stories_per_task),
            (DataSplit.VALIDATION, preset.validation_stories_per_task),
            (DataSplit.TEST, preset.test_stories_per_task),
        )
        for index in range(count)
    ) + tuple(
        f"rendered:root-validation:{index:04d}"
        for index in range(preset.root_validation_stories)
    )
    if tuple(story.story_id for story in rendered.stories) != expected_story_ids:
        raise RenderedTinyWorldsBundleError(
            "rendered stories do not retain canonical symbolic-plan order and IDs"
        )
    _validate_plot_id_isolation(rendered.stories)
    _validate_deterministic_story_replay(
        rendered,
        symbolic,
        tokenizer,
    )
    if any(
        len(story.token_ids) != preset.story_token_count
        for story in rendered.stories
    ):
        raise RenderedTinyWorldsBundleError("rendered story length differs from preset")
    expected_occurrences = tuple(
        (str(task.task_id), split, index)
        for task in symbolic.tasks
        for split, count in (
            (DataSplit.VALIDATION, preset.validation_query_groups_per_task),
            (DataSplit.TEST, preset.test_query_groups_per_task),
        )
        for index in range(count)
    )
    actual_occurrences = tuple(
        (
            group.task_id,
            group.split,
            group.group_plan.occurrence_index,
        )
        for group in rendered.query_groups
    )
    if actual_occurrences != expected_occurrences:
        raise RenderedTinyWorldsBundleError(
            "rendered query groups do not retain canonical occurrence order"
        )
    if any(
        tuple(variant.variant_id for variant in group.variants)
        != tuple(
            f"{group.group_id}:prefix-{prefix_length}"
            for prefix_length in TINYWORLDS_PREFIX_LENGTHS
        )
        for group in rendered.query_groups
    ):
        raise RenderedTinyWorldsBundleError("rendered variant IDs are not canonical")
    _validate_deterministic_query_replay(rendered, symbolic, tokenizer)
    for task in symbolic.tasks:
        task_id = str(task.task_id)
        for split, expected in (
            (DataSplit.TRAIN, preset.training_stories_per_task),
            (DataSplit.VALIDATION, preset.validation_stories_per_task),
            (DataSplit.TEST, preset.test_stories_per_task),
        ):
            actual = sum(
                story.task_id == task_id and story.split is split
                for story in rendered.stories
            )
            if actual != expected:
                raise RenderedTinyWorldsBundleError(
                    f"rendered story count mismatch for {task_id}/{split.value}"
                )
        for split, expected in (
            (DataSplit.VALIDATION, preset.validation_query_groups_per_task),
            (DataSplit.TEST, preset.test_query_groups_per_task),
        ):
            actual = sum(
                group.task_id == task_id and group.split is split
                for group in rendered.query_groups
            )
            if actual != expected:
                raise RenderedTinyWorldsBundleError(
                    f"rendered query count mismatch for {task_id}/{split.value}"
                )
    if sum(story.purpose == "root_validation" for story in rendered.stories) != (
        preset.root_validation_stories
    ):
        raise RenderedTinyWorldsBundleError("root-validation story count mismatch")


def _validate_plot_id_isolation(stories: tuple[RenderedStory, ...]) -> None:
    plot_ids = tuple(story.plot_id for story in stories)
    by_split = tuple(
        {story.plot_id for story in stories if story.split is split}
        for split in DataSplit
    )
    if any(
        by_split[left] & by_split[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RenderedTinyWorldsBundleError(
            "rendered story plot IDs must be disjoint across splits"
        )
    if len(set(plot_ids)) != len(plot_ids):
        raise RenderedTinyWorldsBundleError(
            "rendered story plot IDs must be globally unique"
        )


def _validate_deterministic_story_replay(
    rendered: RenderedTinyWorlds,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> None:
    field_names = (
        "story_id",
        "task_id",
        "split",
        "purpose",
        "text",
        "token_ids",
        "template_family_ids",
        "plot_id",
        "alignments",
        "text_sha256",
    )
    for story, expected in zip(
        rendered.stories,
        _replayed_stories(symbolic, rendered, tokenizer),
        strict=True,
    ):
        changed = tuple(
            field_name
            for field_name in field_names
            if getattr(story, field_name) != getattr(expected, field_name)
        )
        if changed:
            raise RenderedTinyWorldsBundleError(
                "rendered story differs from deterministic symbolic rendering: "
                f"{story.story_id} fields={','.join(changed)}"
            )


def _replayed_stories(
    symbolic: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
    tokenizer: TextTokenizer,
) -> Iterator[RenderedStory]:
    preset = rendered.preset
    counts = {
        DataSplit.TRAIN: preset.training_stories_per_task,
        DataSplit.VALIDATION: preset.validation_stories_per_task,
        DataSplit.TEST: preset.test_stories_per_task,
    }
    plans = {
        (plan.task_id, plan.split): plan for plan in symbolic.story_plans
    }
    for task in symbolic.tasks:
        for split in DataSplit:
            plan = plans[(task.task_id, split)]
            for story_index in range(counts[split]):
                yield _render_story(
                    symbolic,
                    rendered.registry,
                    tokenizer,
                    plan,
                    story_index,
                    preset.story_token_count,
                )
    for story_index in range(preset.root_validation_stories):
        yield _render_root_story(
            rendered.registry,
            tokenizer,
            story_index,
            preset.story_token_count,
        )


def _validate_deterministic_query_replay(
    rendered: RenderedTinyWorlds,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> None:
    for group, expected in zip(
        rendered.query_groups,
        _replayed_query_groups(symbolic, rendered, tokenizer),
        strict=True,
    ):
        actual_record = _encode_group(group, symbolic, tokenizer)
        expected_record = _encode_group(expected, symbolic, tokenizer)
        if actual_record != expected_record:
            changed = tuple(
                key
                for key in actual_record
                if actual_record[key] != expected_record[key]
            )
            raise RenderedTinyWorldsBundleError(
                "rendered query group differs from deterministic symbolic "
                f"rendering: {group.group_id} fields={','.join(changed)}"
            )


def _replayed_query_groups(
    symbolic: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
    tokenizer: TextTokenizer,
) -> Iterator[RenderedQueryGroup]:
    preset = rendered.preset
    counts = {
        DataSplit.VALIDATION: preset.validation_query_groups_per_task,
        DataSplit.TEST: preset.test_query_groups_per_task,
    }
    plans_by_task_split = _source_query_plans(symbolic)
    for task in symbolic.tasks:
        for split in (DataSplit.VALIDATION, DataSplit.TEST):
            plans = plans_by_task_split[(task.task_id, split)]
            for group_index in range(counts[split]):
                yield _render_query_group_at_index(
                    symbolic,
                    rendered.registry,
                    tokenizer,
                    plans,
                    group_index,
                )


def _validate_story_symbolic(story, symbolic, registry) -> None:
    family_by_id = {family.family_id: family for family in registry.families}
    for family_id in story.template_family_ids:
        family = family_by_id.get(family_id)
        if family is None or family.split is not story.split:
            raise RenderedTinyWorldsBundleError(
                "story template provenance is unknown or belongs to another split"
            )
    previous_end = 0
    for alignment in story.alignments:
        if alignment.start_character < previous_end:
            raise RenderedTinyWorldsBundleError("story alignments overlap")
        previous_end = alignment.end_character
    if story.task_id is None:
        if any(item.fact_ids or item.rule_ids for item in story.alignments):
            raise RenderedTinyWorldsBundleError(
                "root-validation alignment cannot claim task knowledge"
            )
        return
    tasks = {str(task.task_id): task for task in symbolic.tasks}
    task = tasks.get(story.task_id)
    if task is None:
        raise RenderedTinyWorldsBundleError("story references an unknown task")
    fact_ids = {str(item) for item in task.direct_fact_ids}
    rule_ids = {str(item) for item in task.rule_ids}
    if any(
        not set(item.fact_ids).issubset(fact_ids)
        or not set(item.rule_ids).issubset(rule_ids)
        for item in story.alignments
    ):
        raise RenderedTinyWorldsBundleError(
            "story alignment contains dangling or cross-task symbolic IDs"
        )


def _query_plan_for_group(symbolic: TinyWorldsBundle, group) -> QueryPlan:
    matches = tuple(
        plan
        for plan in symbolic.query_plans
        if str(plan.query_ast.query_id) == group.symbolic_query_id
        and str(plan.task_id) == group.task_id
        and plan.split is group.split
    )
    if len(matches) != 1:
        raise RenderedTinyWorldsBundleError(
            "query group does not resolve one symbolic query plan"
        )
    if matches[0] != group.group_plan.source_plan:
        raise RenderedTinyWorldsBundleError(
            "query group source differs from its accepted plan"
        )
    return matches[0]


def _validate_variant_symbolic(
    variant: RenderedQueryVariant,
    group,
    plan: QueryPlan,
    symbolic: TinyWorldsBundle,
    tokenizer: TextTokenizer,
) -> None:
    query = variant.knowledge_query
    group_plan = group.group_plan
    if group_plan.source_plan != plan:
        raise RenderedTinyWorldsBundleError(
            "query group source differs from its accepted plan"
        )
    expected_candidates = tuple(
        str(item.entity_id) for item in group_plan.candidates
    )
    expected_correct_index = group_plan.correct_index
    expected_support = tuple(
        str(item)
        for item in (*plan.proof.supporting_fact_ids, *plan.proof.supporting_rule_ids)
    )
    edge_node_by_id = {
        task.incoming_edge_id: str(task.task_id)
        for task in symbolic.tasks
    }
    expected_edges = tuple(
        edge_node_by_id[item]
        for item in plan.proof.required_edge_ids
    )
    expected_oracles = tuple(str(item) for item in plan.hard_oracle_task_ids)
    task = symbolic.world.task(plan.task_id)
    expected_scalar = (
        group.task_id,
        str(task.family_id),
        plan.kind.value,
        expected_correct_index,
        str(plan.proof.proof_id),
        expected_support,
        expected_edges,
        "novel_binding",
        "cross_branch" if plan.kind is QueryKind.CROSS_BRANCH else plan.kind.value,
        plan.proof.depth,
        "open_book" if plan.kind is QueryKind.OPEN_BOOK else "closed_book",
        expected_oracles,
    )
    actual_scalar = (
        str(query.task_id),
        query.family_id,
        query.query_kind,
        query.correct_candidate_index,
        query.proof_id,
        query.support_ids,
        tuple(str(item) for item in query.required_edge_ids),
        query.novelty_regime,
        query.reasoning_type,
        query.reasoning_depth,
        query.mode,
        tuple(str(item) for item in query.oracle_node_ids),
    )
    if actual_scalar != expected_scalar:
        raise RenderedTinyWorldsBundleError(
            "query semantic metadata differs from symbolic plan"
        )
    if variant.candidate_entity_ids != expected_candidates:
        raise RenderedTinyWorldsBundleError(
            "query candidate entity order differs from symbolic plan"
        )
    if query.query_id != variant.variant_id or variant.group_id != group.group_id:
        raise RenderedTinyWorldsBundleError("query/variant/group IDs are inconsistent")
    if variant.split is not group.split or query.prefix_length != len(
        variant.prefix_token_ids
    ):
        raise RenderedTinyWorldsBundleError("query split or prefix length mismatch")
    encoded_prefix = tokenizer.encode(variant.prefix_text)
    if encoded_prefix != variant.prefix_token_ids:
        raise RenderedTinyWorldsBundleError("query prefix tokens differ from tokenizer")
    if len(encoded_prefix) not in TINYWORLDS_PREFIX_LENGTHS:
        raise RenderedTinyWorldsBundleError("query prefix length is not canonical")
    combined_sequences = tuple(
        tokenizer.encode(variant.prefix_text + candidate.answer_text)
        for candidate in query.candidates
    )
    suffix_counts = tuple(
        len(sequence) - len(encoded_prefix) for sequence in combined_sequences
    )
    if (
        len(set(suffix_counts)) != 1
        or suffix_counts[0] < 1
        or any(
            sequence[: len(encoded_prefix)] != encoded_prefix
            for sequence in combined_sequences
        )
        or any(
            len(sequence) > TINYWORLDS_CONTEXT_LENGTH
            for sequence in combined_sequences
        )
    ):
        raise RenderedTinyWorldsBundleError(
            "query candidate texts do not preserve an equal exact token boundary"
        )
    rebuilt = tuple(
        build_prefix_suffix_batches(
            sequence,
            len(encoded_prefix),
            suffix_counts[0],
            pad_token_id=tokenizer.pad_token_id,
        )
        for sequence in combined_sequences
    )
    if any(
        not _batch_equal(candidate.competence_batch, expected[1])
        for candidate, expected in zip(query.candidates, rebuilt)
    ) or not _batch_equal(query.router_batch, rebuilt[0][0]):
        raise RenderedTinyWorldsBundleError(
            "persisted candidate texts do not reconstruct the supplied query batches"
        )
    _validate_cue_metadata(query, symbolic, plan)
    entities = {str(item.entity_id): item for item in symbolic.entities}
    candidate_names = tuple(
        entities[entity_id].name for entity_id in variant.candidate_entity_ids
    )
    for index, candidate in enumerate(query.candidates):
        folded_answer = candidate.answer_text.casefold()
        if candidate_names[index].casefold() not in folded_answer or any(
            name.casefold() in folded_answer
            for other_index, name in enumerate(candidate_names)
            if other_index != index
        ):
            raise RenderedTinyWorldsBundleError(
                "candidate answer text does not uniquely realize its symbolic entity"
            )
    if query.mode == "closed_book":
        folded = variant.prefix_text.casefold()
        candidate_texts = tuple(item.answer_text for item in query.candidates)
        candidate_sequences = tuple(tokenizer.encode(text) for text in candidate_texts)
        for entity_id, sequence in zip(variant.candidate_entity_ids, candidate_sequences):
            entity = entities[entity_id]
            if entity.name.casefold() in folded or _contains_subsequence(
                encoded_prefix, sequence
            ) or _contains_subsequence(encoded_prefix, tokenizer.encode(entity.name)):
                raise RenderedTinyWorldsBundleError(
                    "closed-book query prefix leaks a candidate"
                )


def _validate_group_core(group: RenderedQueryGroup) -> None:
    core = group.variants[0]
    if len(core.prefix_token_ids) != 64:
        raise RenderedTinyWorldsBundleError("query group's first variant is not its core")
    if _text_sha256(core.prefix_text) != core.query_core_sha256:
        raise RenderedTinyWorldsBundleError("query core text hash mismatch")
    if any(
        variant.query_core_sha256 != core.query_core_sha256
        or not variant.prefix_text.endswith(core.prefix_text)
        for variant in group.variants
    ):
        raise RenderedTinyWorldsBundleError(
            "paired query prefixes do not preserve one exact textual query core"
        )


def _batch_equal(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in ("input_ids", "attention_mask", "target_ids", "loss_mask")
    )


def _validate_cue_metadata(query, symbolic: TinyWorldsBundle, plan: QueryPlan) -> None:
    task = symbolic.world.task(plan.task_id)
    all_tasks = tuple(str(item.task_id) for item in symbolic.tasks)
    family_tasks = tuple(
        str(item.task_id) for item in symbolic.tasks if item.family_id == task.family_id
    )
    if query.cue_regime == "cue_sufficient":
        expected_tasks = (str(plan.task_id),)
        expected_cues = (f"task:{plan.task_id}",)
    elif query.cue_regime == "cue_present":
        expected_tasks = family_tasks
        expected_cues = (f"family:{task.family_id}",)
    else:
        expected_tasks = all_tasks
        expected_cues = ()
    if tuple(str(item) for item in query.eligible_task_ids) != expected_tasks:
        raise RenderedTinyWorldsBundleError("eligible tasks do not match visible cues")
    if query.visible_cue_ids != expected_cues:
        raise RenderedTinyWorldsBundleError("visible cue IDs do not match cue regime")


def _tokenization_fingerprint(
    story_records: tuple[JsonObject, ...],
    group_records: tuple[JsonObject, ...],
) -> str:
    records: list[str] = []
    records.extend(
        f"story:{item['story_id']}:{item['token_ids_sha256']}" for item in story_records
    )
    for group in group_records:
        for variant in group["variants"]:
            records.append(
                f"prefix:{variant['variant_id']}:{variant['prefix_token_ids_sha256']}"
            )
            records.extend(
                f"candidate:{variant['variant_id']}:{index}:"
                f"{candidate['combined_token_ids_sha256']}"
                for index, candidate in enumerate(variant["candidates"])
            )
    return _digest(("\n".join(records) + "\n").encode("utf-8"))


def _manifest_core(manifest: RenderedTinyWorldsManifest) -> JsonObject:
    return _manifest_core_values(
        manifest.rendered_bundle_id,
        manifest.symbolic_bundle_sha256,
        manifest.artifacts,
    )


def _manifest_core_values(
    rendered_bundle_id: str,
    symbolic_bundle_sha256: str,
    artifacts: tuple[RenderedBundleFileDigest, ...],
) -> JsonObject:
    return {
        "artifacts": [
            {
                "path": item.path,
                "record_count": item.record_count,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in artifacts
        ],
        "format": _FORMAT,
        "rendered_bundle_id": rendered_bundle_id,
        "schema_version": _SCHEMA_VERSION,
        "symbolic_bundle_sha256": symbolic_bundle_sha256,
    }


def _manifest_record(manifest: RenderedTinyWorldsManifest) -> JsonObject:
    return {**_manifest_core(manifest), "bundle_sha256": manifest.bundle_sha256}


def _decode_manifest(record: object) -> RenderedTinyWorldsManifest:
    data = _fields(
        record,
        (
            "artifacts",
            "bundle_sha256",
            "format",
            "rendered_bundle_id",
            "schema_version",
            "symbolic_bundle_sha256",
        ),
        "rendered manifest",
    )
    return RenderedTinyWorldsManifest(
        format=_string(data["format"], "format"),
        schema_version=_integer(data["schema_version"], "schema_version"),
        rendered_bundle_id=_string(
            data["rendered_bundle_id"], "rendered_bundle_id"
        ),
        symbolic_bundle_sha256=_string(
            data["symbolic_bundle_sha256"], "symbolic_bundle_sha256"
        ),
        artifacts=tuple(
            _decode_file_digest(item)
            for item in _list(data["artifacts"], "manifest artifacts")
        ),
        bundle_sha256=_string(data["bundle_sha256"], "bundle_sha256"),
    )


def _decode_file_digest(record: object) -> RenderedBundleFileDigest:
    data = _fields(
        record,
        ("path", "record_count", "sha256", "size_bytes"),
        "rendered artifact digest",
    )
    return RenderedBundleFileDigest(
        path=_string(data["path"], "artifact path"),
        sha256=_string(data["sha256"], "artifact sha256"),
        size_bytes=_integer(data["size_bytes"], "artifact size"),
        record_count=_integer(data["record_count"], "artifact record count"),
    )


T = TypeVar("T")


def _load_jsonl_with_records(
    path: Path,
    artifact: RenderedBundleFileDigest,
    decode: Callable[[object], T],
) -> tuple[tuple[JsonObject, ...], tuple[T, ...]]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise RenderedTinyWorldsBundleError(f"JSONL lacks final newline: {path.name}")
    lines = payload.splitlines(keepends=True)
    if len(lines) != artifact.record_count:
        raise RenderedTinyWorldsBundleError(
            f"rendered artifact record count mismatch: {path.name}"
        )
    records: list[JsonObject] = []
    values: list[T] = []
    for index, line in enumerate(lines, start=1):
        record = _loads_strict(line, f"{path.name}:{index}")
        if type(record) is not dict:
            raise RenderedTinyWorldsBundleError(
                f"JSONL record must be an object: {path.name}:{index}"
            )
        if line != _canonical_json(record):
            raise RenderedTinyWorldsBundleError(
                f"non-canonical JSONL record: {path.name}:{index}"
            )
        try:
            values.append(decode(record))
        except (KeyError, TypeError, ValueError) as error:
            if type(error) is RenderedTinyWorldsBundleError:
                raise
            raise RenderedTinyWorldsBundleError(
                f"invalid {path.name} record {index}: {error}"
            ) from error
        records.append(record)
    return tuple(records), tuple(values)


def _read_canonical_json(path: Path) -> object:
    try:
        payload = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError) as error:
        raise RenderedTinyWorldsBundleError(
            f"missing rendered artifact: {path.name}"
        ) from error
    record = _loads_strict(payload, path.name)
    if payload != _canonical_json(record):
        raise RenderedTinyWorldsBundleError(f"non-canonical JSON: {path.name}")
    return record


def _loads_strict(payload: bytes, label: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RenderedTinyWorldsBundleError(
                    f"duplicate JSON field {key!r} in {label}"
                )
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RenderedTinyWorldsBundleError(
            f"invalid JSON in {label}: {error}"
        ) from error


def _canonical_json(record: object) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl(records: tuple[JsonObject, ...]) -> bytes:
    return b"".join(_canonical_json(record) for record in records)


def _fields(
    record: object,
    expected: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(record) is not dict:
        raise RenderedTinyWorldsBundleError(f"{label} must be a JSON object")
    actual = set(record)
    wanted = set(expected)
    if actual != wanted:
        raise RenderedTinyWorldsBundleError(
            f"{label} fields differ; unknown={tuple(sorted(actual - wanted))}, "
            f"missing={tuple(sorted(wanted - actual))}"
        )
    return record


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise RenderedTinyWorldsBundleError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise RenderedTinyWorldsBundleError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise RenderedTinyWorldsBundleError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RenderedTinyWorldsBundleError(
            f"{label} must be a lowercase hexadecimal SHA-256"
        )


def _text_sha256(text: str) -> str:
    return _digest(text.encode("utf-8"))


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return _digest(
        json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    )


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _contains_subsequence(
    sequence: tuple[int, ...],
    subsequence: tuple[int, ...],
) -> bool:
    return bool(subsequence) and any(
        sequence[start : start + len(subsequence)] == subsequence
        for start in range(len(sequence) - len(subsequence) + 1)
    )


__all__ = [
    "RenderedBundleFileDigest",
    "RenderedTinyWorldsBundleError",
    "RenderedTinyWorldsManifest",
    "load_rendered_tinyworlds_bundle",
    "load_rendered_tinyworlds_manifest",
    "rendered_tokenization_sha256",
    "write_rendered_tinyworlds_bundle",
]
