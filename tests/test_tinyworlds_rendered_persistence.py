from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pytest

import apm.data.text.tinyworlds.rendering as rendering_module
from apm.data.text.tinyworlds import (
    RenderedTinyWorldsBundleError,
    TinyWorldsRenderPreset,
    TinyWorldsRenderingRejection,
    expand_query_group_plan_attempts,
    generate_calibration_bundle,
    load_rendered_tinyworlds_bundle,
    load_rendered_tinyworlds_manifest,
    render_tinyworlds_bundle,
    rendered_tokenization_sha256,
    write_rendered_tinyworlds_bundle,
)


@dataclass(frozen=True)
class _WhitespaceTokenizer:
    token_offset: int = 2

    @property
    def vocab_size(self) -> int:
        return 65_536

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        words = tuple(re.findall(r"\S+", text))
        tokens = tuple(
            self.token_offset
            + int.from_bytes(sha256(word.encode("utf-8")).digest()[:2], "big")
            for word in words
        )
        return tokens + ((self.eos_token_id,) if add_eos else ())

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _preset() -> TinyWorldsRenderPreset:
    return TinyWorldsRenderPreset(
        training_stories_per_task=2,
        validation_stories_per_task=1,
        test_stories_per_task=1,
        validation_query_groups_per_task=4,
        test_query_groups_per_task=4,
        root_validation_stories=2,
        story_token_count=256,
        context_length=256,
    )


@pytest.fixture(scope="module")
def rendered_fixture():
    symbolic = generate_calibration_bundle("8" * 64)
    tokenizer = _WhitespaceTokenizer()
    rendered = render_tinyworlds_bundle(symbolic, tokenizer, _preset())
    return symbolic, tokenizer, rendered


def _canonical(record: object) -> bytes:
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


def _rewrite_artifact(
    root: Path,
    relative_path: str,
    payload: bytes,
    *,
    record_count: int | None = None,
) -> None:
    (root / relative_path).write_bytes(payload)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    descriptor = next(
        item for item in manifest["artifacts"] if item["path"] == relative_path
    )
    descriptor["sha256"] = sha256(payload).hexdigest()
    descriptor["size_bytes"] = len(payload)
    if record_count is not None:
        descriptor["record_count"] = record_count
    core = {
        key: value for key, value in manifest.items() if key != "bundle_sha256"
    }
    manifest["bundle_sha256"] = sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(record) for record in records)


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _rehash_rendered_tree(root: Path) -> None:
    stories = _read_jsonl(root / "stories.jsonl")
    groups = _read_jsonl(root / "query_groups.jsonl")
    fingerprint_records = [
        f"story:{record['story_id']}:{record['token_ids_sha256']}"
        for record in stories
    ]
    for group in groups:
        for variant in group["variants"]:
            fingerprint_records.append(
                "prefix:"
                f"{variant['variant_id']}:{variant['prefix_token_ids_sha256']}"
            )
            fingerprint_records.extend(
                "candidate:"
                f"{variant['variant_id']}:{index}:"
                f"{candidate['combined_token_ids_sha256']}"
                for index, candidate in enumerate(variant["candidates"])
            )
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata["tokenizer"]["tokenization_sha256"] = sha256(
        ("\n".join(fingerprint_records) + "\n").encode("utf-8")
    ).hexdigest()
    metadata_path.write_bytes(_canonical(metadata))

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for descriptor in manifest["artifacts"]:
        payload = (root / descriptor["path"]).read_bytes()
        descriptor["sha256"] = sha256(payload).hexdigest()
        descriptor["size_bytes"] = len(payload)
    core = {
        key: value for key, value in manifest.items() if key != "bundle_sha256"
    }
    manifest["bundle_sha256"] = sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _assert_batch_equal(left, right) -> None:
    for field in ("input_ids", "attention_mask", "target_ids", "loss_mask"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_rendered_round_trip_reconstructs_exact_batches_and_is_byte_identical(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = write_rendered_tinyworlds_bundle(
        rendered,
        symbolic,
        tokenizer,
        first,
    )
    second_manifest = write_rendered_tinyworlds_bundle(
        rendered,
        symbolic,
        tokenizer,
        second,
    )
    loaded = load_rendered_tinyworlds_bundle(first, symbolic, tokenizer)

    assert first_manifest == second_manifest == load_rendered_tinyworlds_manifest(first)
    assert rendered_tokenization_sha256(rendered, symbolic, tokenizer) == json.loads(
        (first / "metadata.json").read_bytes()
    )["tokenizer"]["tokenization_sha256"]
    assert {
        path.name: path.read_bytes() for path in first.iterdir()
    } == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert loaded.stories == rendered.stories
    assert len(loaded.query_groups) == len(rendered.query_groups)
    revision_indices_by_split = {
        split: sorted(
            group.variants[0].knowledge_query.correct_candidate_index
            for group in loaded.query_groups
            if group.task_id == "calibration_revision" and group.split.value == split
        )
        for split in ("validation", "test")
    }
    assert revision_indices_by_split == {
        "validation": [0, 1, 2, 3],
        "test": [0, 1, 2, 3],
    }
    for expected_group, actual_group in zip(rendered.query_groups, loaded.query_groups):
        assert (
            actual_group.group_id,
            actual_group.task_id,
            actual_group.split,
            actual_group.symbolic_query_id,
        ) == (
            expected_group.group_id,
            expected_group.task_id,
            expected_group.split,
            expected_group.symbolic_query_id,
        )
        for expected, actual in zip(expected_group.variants, actual_group.variants):
            assert actual.prefix_token_ids == expected.prefix_token_ids
            assert tuple(item.answer_text for item in actual.knowledge_query.candidates) == tuple(
                item.answer_text for item in expected.knowledge_query.candidates
            )
            _assert_batch_equal(
                actual.knowledge_query.router_batch,
                expected.knowledge_query.router_batch,
            )
            for actual_candidate, expected_candidate in zip(
                actual.knowledge_query.candidates,
                expected.knowledge_query.candidates,
            ):
                _assert_batch_equal(
                    actual_candidate.competence_batch,
                    expected_candidate.competence_batch,
                )
    with pytest.raises(FileExistsError):
        write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, first)


def test_fallback_query_group_plan_provenance_round_trips_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    symbolic = generate_calibration_bundle("7" * 64)
    tokenizer = _WhitespaceTokenizer()
    preset = TinyWorldsRenderPreset(
        training_stories_per_task=1,
        validation_stories_per_task=1,
        test_stories_per_task=1,
        validation_query_groups_per_task=1,
        test_query_groups_per_task=1,
        root_validation_stories=1,
        story_token_count=256,
        context_length=256,
    )
    attempts = expand_query_group_plan_attempts(symbolic, preset)
    assert len(attempts[0]) > 1
    rejected_plan = attempts[0][0]
    expected_fallback = attempts[0][1]
    original_render = rendering_module._render_query_group
    rejected_count = 0

    def reject_first(bundle, registry, supplied_tokenizer, group_plan):
        nonlocal rejected_count
        if group_plan == rejected_plan:
            rejected_count += 1
            raise TinyWorldsRenderingRejection(
                "deterministic persistence fallback fixture"
            )
        return original_render(bundle, registry, supplied_tokenizer, group_plan)

    monkeypatch.setattr(rendering_module, "_render_query_group", reject_first)
    rendered = render_tinyworlds_bundle(symbolic, tokenizer, preset)
    assert rendered.query_groups[0].group_plan == expected_fallback
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)

    loaded = load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)

    assert rejected_count >= 3
    assert loaded.query_groups[0].group_plan == expected_fallback
    assert loaded.query_groups[0].symbolic_query_id == (
        expected_fallback.source_query_id
    )
    persisted = _read_jsonl(root / "query_groups.jsonl")[0]["group_plan"]
    assert persisted["source_query_id"] == expected_fallback.source_query_id
    assert persisted["source_proof_id"] == expected_fallback.source_proof_id
    assert persisted["holdout_identity_sha256"] == (
        expected_fallback.holdout_identity_sha256
    )


def test_loader_rejects_digest_mismatch(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    path = root / "stories.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"plot_id":"plot:', b'"plot_id":"plox:', 1))

    with pytest.raises(RenderedTinyWorldsBundleError, match="digest mismatch"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_unknown_nested_fields_after_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "stories.jsonl")
    records[0]["alignments"][0]["unexpected"] = "forbidden"
    _rewrite_artifact(root, "stories.jsonl", _jsonl(records))

    with pytest.raises(RenderedTinyWorldsBundleError, match="unknown=.*unexpected"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_record_count_and_wrong_tokenizer(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    payload = (root / "stories.jsonl").read_bytes()
    _rewrite_artifact(
        root,
        "stories.jsonl",
        payload,
        record_count=len(payload.splitlines()) + 1,
    )
    with pytest.raises(RenderedTinyWorldsBundleError, match="count mismatch"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)

    fresh = tmp_path / "fresh"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, fresh)
    with pytest.raises(RenderedTinyWorldsBundleError, match="token hash mismatch"):
        load_rendered_tinyworlds_bundle(
            fresh,
            symbolic,
            _WhitespaceTokenizer(token_offset=3),
        )


def test_loader_rejects_split_overlap_after_consistent_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "stories.jsonl")
    training = next(item for item in records if item["split"] == "train")
    validation = next(
        item
        for item in records
        if item["split"] == "validation" and item["task_id"] == training["task_id"]
    )
    for field in (
        "alignments",
        "text",
        "text_sha256",
        "token_count",
        "token_ids_sha256",
    ):
        validation[field] = training[field]
    _rewrite_artifact(root, "stories.jsonl", _jsonl(records))

    with pytest.raises(RenderedTinyWorldsBundleError, match="disjoint across splits"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_same_task_alignment_relabel_after_consistent_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "stories.jsonl")
    story = next(
        record
        for record in records
        if record["task_id"] is not None
        and any(alignment["fact_ids"] for alignment in record["alignments"])
    )
    alignment = next(
        item for item in story["alignments"] if item["fact_ids"]
    )
    original_fact_id = alignment["fact_ids"][0]
    task = next(
        item for item in symbolic.tasks if str(item.task_id) == story["task_id"]
    )
    replacement_fact_id = next(
        str(fact_id)
        for fact_id in task.direct_fact_ids
        if str(fact_id) != original_fact_id
    )
    alignment["fact_ids"][0] = replacement_fact_id
    (root / "stories.jsonl").write_bytes(_jsonl(records))
    _rehash_rendered_tree(root)

    with pytest.raises(
        RenderedTinyWorldsBundleError,
        match=r"deterministic symbolic rendering: .*fields=alignments",
    ):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


@pytest.mark.parametrize(
    "tamper",
    ("text", "template", "span", "plot"),
    ids=("text", "template-occurrence", "alignment-span", "plot-id"),
)
def test_loader_rejects_story_occurrence_tamper_after_consistent_rehash(
    tmp_path: Path,
    rendered_fixture,
    tamper: str,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / f"rendered-{tamper}"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "stories.jsonl")
    story = next(
        record
        for record in records
        if record["split"] == "train" and record["task_id"] is not None
    )
    if tamper == "text":
        text = story["text"]
        character_index = next(
            index for index, character in enumerate(text) if character.isalpha()
        )
        replacement = "z" if text[character_index].casefold() != "z" else "y"
        changed_text = (
            text[:character_index] + replacement + text[character_index + 1 :]
        )
        tokens = tokenizer.encode(changed_text)
        assert len(tokens) == story["token_count"]
        story["text"] = changed_text
        story["text_sha256"] = sha256(changed_text.encode("utf-8")).hexdigest()
        story["token_ids_sha256"] = _token_ids_sha256(tokens)
    elif tamper == "template":
        families = story["template_family_ids"]
        assert len(set(families)) > 1
        story["template_family_ids"] = list(reversed(families))
    elif tamper == "span":
        alignment = next(
            item
            for item in story["alignments"]
            if item["end_character"] - item["start_character"] > 1
        )
        alignment["start_character"] += 1
    else:
        story["plot_id"] += ":altered"
    (root / "stories.jsonl").write_bytes(_jsonl(records))
    _rehash_rendered_tree(root)

    with pytest.raises(
        RenderedTinyWorldsBundleError,
        match="differs from deterministic symbolic rendering",
    ):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_cross_split_plot_overlap_after_consistent_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "stories.jsonl")
    training = next(record for record in records if record["split"] == "train")
    validation = next(
        record for record in records if record["split"] == "validation"
    )
    validation["plot_id"] = training["plot_id"]
    (root / "stories.jsonl").write_bytes(_jsonl(records))
    _rehash_rendered_tree(root)

    with pytest.raises(
        RenderedTinyWorldsBundleError,
        match="plot IDs must be disjoint across splits",
    ):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_cue_text_metadata_relabel_after_consistent_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "query_groups.jsonl")
    variant = next(
        variant
        for group in records
        for variant in group["variants"]
        if variant["semantic"]["cue_regime"] == "cue_sufficient"
        and "belongs only to task" in variant["prefix_text"]
    )
    semantic = variant["semantic"]
    family_id = semantic["family_id"]
    semantic["cue_regime"] = "cue_present"
    semantic["visible_cue_ids"] = [f"family:{family_id}"]
    semantic["eligible_task_ids"] = [
        str(task.task_id)
        for task in symbolic.tasks
        if str(task.family_id) == family_id
    ]
    (root / "query_groups.jsonl").write_bytes(_jsonl(records))
    _rehash_rendered_tree(root)

    with pytest.raises(
        RenderedTinyWorldsBundleError,
        match="query group differs from deterministic symbolic rendering",
    ):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_candidate_boundary_tampering_after_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "query_groups.jsonl")
    candidate = records[0]["variants"][0]["candidates"][0]
    prefix = records[0]["variants"][0]["prefix_text"]
    candidate["answer_text"] += " extra"
    candidate["answer_text_sha256"] = sha256(
        candidate["answer_text"].encode("utf-8")
    ).hexdigest()
    combined = tokenizer.encode(prefix + candidate["answer_text"])
    candidate["combined_token_count"] = len(combined)
    candidate["combined_token_ids_sha256"] = sha256(
        json.dumps(list(combined), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    candidate["suffix_token_count"] = len(combined) - 64
    _rewrite_artifact(root, "query_groups.jsonl", _jsonl(records))

    with pytest.raises(RenderedTinyWorldsBundleError, match="suffix token counts"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_a_noncanonical_occurrence_rotation_after_rehash(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)
    records = _read_jsonl(root / "query_groups.jsonl")
    rotated = next(
        record
        for record in records
        if record["task_id"] == "calibration_revision"
        and record["group_id"].endswith(":0001")
    )
    assert rotated["variants"][0]["semantic"]["correct_candidate_index"] == 0
    rotated["variants"][0]["semantic"]["correct_candidate_index"] = 1
    _rewrite_artifact(root, "query_groups.jsonl", _jsonl(records))

    with pytest.raises(RenderedTinyWorldsBundleError, match="semantic metadata"):
        load_rendered_tinyworlds_bundle(root, symbolic, tokenizer)


def test_loader_rejects_a_different_symbolic_bundle(
    tmp_path: Path,
    rendered_fixture,
) -> None:
    symbolic, tokenizer, rendered = rendered_fixture
    root = tmp_path / "rendered"
    write_rendered_tinyworlds_bundle(rendered, symbolic, tokenizer, root)

    with pytest.raises(RenderedTinyWorldsBundleError, match="different symbolic"):
        load_rendered_tinyworlds_bundle(
            root,
            generate_calibration_bundle("9" * 64),
            tokenizer,
        )
