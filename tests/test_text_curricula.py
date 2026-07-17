from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from string import ascii_lowercase

import pytest

import apm.data.text.curricula as text_curricula
from apm.data.text.curricula import (
    CorpusCurriculum,
    CorpusSplits,
    DocumentSplits,
    PinnedDatasetFile,
    StorySplitCounts,
    TINYSTORIES_DATASET_REVISION,
    TINYSTORIES_DOCUMENT_SEPARATOR,
    TINYSTORIES_EVALUATION_PRESET,
    TINYSTORIES_SINGLE_GPU_PRESET,
    TINYSTORIES_V2_SOURCE,
    TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS,
    TINY_SHAKESPEARE_EVALUATION_PRESET,
    TinyStoriesSourceContract,
    apply_ascii_letter_permutation,
    apply_token_permutation,
    ascii_letter_permutation,
    build_contiguous_region_curriculum,
    build_stable_hash_curriculum,
    build_tiny_shakespeare_permutation_curriculum,
    build_tiny_shakespeare_region_curriculum,
    build_tiny_shakespeare_stable_hash_curriculum,
    build_tinystories_topic_curriculum,
    classify_tinystory_topic,
    load_pinned_dataset_text,
    load_tinystories_topic_dataset,
    normalized_documents,
    normalize_text,
    parse_tinystories,
    prepare_tinystories_splits,
    raw_text_sha256,
    seeded_token_permutation,
    sha256_content_id,
    stable_hash_task_index,
    stable_hash_raw_text_task_index,
    topic_scores,
    verify_pinned_dataset_file,
)


def _aggregate(*stories: str) -> str:
    return TINYSTORIES_DOCUMENT_SEPARATOR.join(stories) + TINYSTORIES_DOCUMENT_SEPARATOR


def _pinned_tinystories_fixture(
    tmp_path: Path,
    train_text: str,
    validation_text: str,
) -> tuple[Path, Path, TinyStoriesSourceContract]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        tmp_path / filename
        for filename in ("fixture-train.txt", "fixture-valid.txt")
    )
    payloads = tuple(
        text.encode("utf-8") for text in (train_text, validation_text)
    )
    for path, payload in zip(paths, payloads):
        path.write_bytes(payload)
    pinned_files = tuple(
        PinnedDatasetFile(
            filename=path.name,
            size_bytes=len(payload),
            sha256=sha256(payload).hexdigest(),
        )
        for path, payload in zip(paths, payloads)
    )
    return (
        paths[0],
        paths[1],
        TinyStoriesSourceContract(
            dataset_id="fixture/tinystories",
            revision="0" * 40,
            train_file=pinned_files[0],
            validation_file=pinned_files[1],
        ),
    )


def _document_splits(
    train_count: int = 12,
    validation_count: int = 8,
    test_count: int = 4,
) -> DocumentSplits:
    return DocumentSplits(
        train=normalized_documents(
            tuple(f"train document {index}" for index in range(train_count))
        ),
        validation=normalized_documents(
            tuple(
                f"validation document {index}"
                for index in range(validation_count)
            )
        ),
        test=normalized_documents(
            tuple(f"test document {index}" for index in range(test_count))
        ),
    )


def test_tinystories_source_contract_is_exact_and_frozen() -> None:
    assert TINYSTORIES_DATASET_REVISION == "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    assert TINYSTORIES_V2_SOURCE.dataset_id == "roneneldan/TinyStories"
    assert TINYSTORIES_V2_SOURCE.revision == TINYSTORIES_DATASET_REVISION
    assert (
        TINYSTORIES_V2_SOURCE.train_file.filename,
        TINYSTORIES_V2_SOURCE.train_file.size_bytes,
        TINYSTORIES_V2_SOURCE.train_file.sha256,
    ) == (
        "TinyStoriesV2-GPT4-train.txt",
        2_227_753_162,
        "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
    )
    assert (
        TINYSTORIES_V2_SOURCE.validation_file.filename,
        TINYSTORIES_V2_SOURCE.validation_file.size_bytes,
        TINYSTORIES_V2_SOURCE.validation_file.sha256,
    ) == (
        "TinyStoriesV2-GPT4-valid.txt",
        22_502_601,
        "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
    )
    with pytest.raises(FrozenInstanceError):
        TINYSTORIES_V2_SOURCE.revision = "mutable"  # type: ignore[misc]


def test_local_pinned_file_loader_stream_verifies_before_decoding(
    tmp_path: Path,
) -> None:
    payload = "Café\r\nstory.\n".encode("utf-8")
    source_path = tmp_path / "fixture.txt"
    source_path.write_bytes(payload)
    expected_file = PinnedDatasetFile(
        filename=source_path.name,
        size_bytes=len(payload),
        sha256=sha256(payload).hexdigest(),
    )

    assert verify_pinned_dataset_file(source_path, expected_file) == source_path
    assert load_pinned_dataset_text(source_path, expected_file) == payload.decode(
        "utf-8"
    )

    source_path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_pinned_dataset_file(source_path, expected_file)

    source_path.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_pinned_dataset_file(source_path, expected_file)


def test_normalization_content_ids_separator_parsing_and_deduplication() -> None:
    raw = "  Cafe\u0301\tstory\nwith   spaces  "
    normalized = "Café story with spaces"

    assert normalize_text(raw) == normalized
    assert sha256_content_id(raw) == sha256(
        normalized.encode("utf-8")
    ).hexdigest()
    documents = parse_tinystories(
        _aggregate(raw, normalized, "", "A second story.")
    )

    assert tuple(document.text for document in documents) == (
        normalized,
        "A second story.",
    )
    assert tuple(document.content_id for document in documents) == tuple(
        sha256_content_id(document.text) for document in documents
    )


def test_official_validation_is_overlap_free_and_hash_split_exactly_in_half() -> None:
    train_text = _aggregate("Shared story", "Train only", " Shared  story ")
    evaluation_text = _aggregate(
        "Shared\nstory",
        "Evaluation alpha",
        "Evaluation beta",
        "Evaluation gamma",
        "Evaluation delta",
        "Evaluation epsilon",
        "Evaluation alpha",
    )

    splits = prepare_tinystories_splits(train_text, evaluation_text)
    ordered_evaluation = tuple(
        sorted(
            normalized_documents(
                (
                    "Shared story",
                    "Evaluation alpha",
                    "Evaluation beta",
                    "Evaluation gamma",
                    "Evaluation delta",
                    "Evaluation epsilon",
                )
            ),
            key=lambda document: document.content_id,
        )
    )

    assert tuple(document.text for document in splits.train) == ("Train only",)
    assert splits.validation == ordered_evaluation[:3]
    assert splits.test == ordered_evaluation[3:]
    assert len(splits.validation) == len(splits.test) == 3
    split_id_sets = tuple(
        {document.content_id for document in split}
        for split in (splits.train, splits.validation, splits.test)
    )
    assert not split_id_sets[0] & split_id_sets[1]
    assert not split_id_sets[0] & split_id_sets[2]
    assert not split_id_sets[1] & split_id_sets[2]

    with pytest.raises(ValueError, match="exactly 50/50"):
        prepare_tinystories_splits(
            _aggregate("Train"),
            _aggregate("One", "Two", "Three"),
        )


def test_streamed_topic_dataset_matches_in_memory_selection_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic_phrases = (
        "Dogs and cats",
        "Cars and trucks",
        "Moms and dads",
        "Kings and dragons",
    )
    train_stories = tuple(
        f"{phrase} train example {index}"
        for phrase in topic_phrases
        for index in range(6)
    )
    validation_stories = tuple(
        f"{phrase} official example {index}"
        for phrase in topic_phrases
        for index in range(8)
    )
    train_text = _aggregate(
        *train_stories,
        "  Dogs  and cats train example 2\n",
        validation_stories[0],
        "A story without a classifiable topic.",
    )
    validation_text = _aggregate(
        *validation_stories,
        "Moms  and dads official example 0",
    )
    train_path, validation_path, source = _pinned_tinystories_fixture(
        tmp_path,
        train_text,
        validation_text,
    )
    counts = StorySplitCounts(train=2, validation=1, test=1)
    source_splits = prepare_tinystories_splits(train_text, validation_text)
    expected_curriculum = build_tinystories_topic_curriculum(
        source_splits,
        counts,
    )
    monkeypatch.setattr(
        text_curricula,
        "_TINYSTORIES_TEXT_READ_CHUNK_SIZE",
        7,
    )

    streamed = load_tinystories_topic_dataset(
        train_path,
        validation_path,
        source,
        counts,
    )

    assert streamed.curriculum == expected_curriculum
    assert streamed.root_validation == source_splits.validation


def test_streamed_train_selection_uses_first_content_hash_as_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_collision = "Dogs and cats collision"
    later_collision = "Cars and trucks collision"
    train_path = tmp_path / "train.txt"
    train_path.write_text(
        _aggregate(
            first_collision,
            later_collision,
            "Buses and trains replacement",
            "Moms and dads family",
            "Kings and dragons fantasy",
        ),
        encoding="utf-8",
    )
    collision_texts = {
        normalize_text(first_collision),
        normalize_text(later_collision),
    }
    collision_aware_content_id = lambda text: (
        "0" * 64
        if normalize_text(text) in collision_texts
        else sha256_content_id(text)
    )
    monkeypatch.setattr(
        text_curricula,
        "sha256_content_id",
        collision_aware_content_id,
    )
    monkeypatch.setattr(
        text_curricula,
        "_normalized_document",
        lambda text: (
            text_curricula.TextDocument(
                collision_aware_content_id(normalize_text(text)),
                normalize_text(text),
            )
            if normalize_text(text)
            else None
        ),
    )

    selected = text_curricula._select_tinystories_train_documents(
        train_path,
        frozenset(),
        requested_count=1,
    )
    selected_texts = {document.text for document in selected}

    assert normalize_text(first_collision) in selected_texts
    assert normalize_text(later_collision) not in selected_texts
    assert "Buses and trains replacement" in selected_texts


def test_streamed_topic_dataset_verifies_pins_before_decoding(
    tmp_path: Path,
) -> None:
    train_path, validation_path, source = _pinned_tinystories_fixture(
        tmp_path,
        _aggregate("Dogs and cats"),
        _aggregate("Cars and trucks", "Moms and dads"),
    )
    train_path.write_bytes(b"x" * source.train_file.size_bytes)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tinystories_topic_dataset(
            train_path,
            validation_path,
            source,
            StorySplitCounts(train=1, validation=1, test=1),
        )


def test_streamed_topic_dataset_preserves_split_and_topic_count_failures(
    tmp_path: Path,
) -> None:
    train_path, validation_path, source = _pinned_tinystories_fixture(
        tmp_path / "odd",
        _aggregate("Dogs and cats"),
        _aggregate("Cars and trucks", "Moms and dads", "Kings and dragons"),
    )
    with pytest.raises(ValueError, match="exactly 50/50"):
        load_tinystories_topic_dataset(
            train_path,
            validation_path,
            source,
            StorySplitCounts(train=1, validation=1, test=1),
        )

    validation_stories = tuple(
        f"{phrase} official example {index}"
        for phrase in (
            "Dogs and cats",
            "Cars and trucks",
            "Moms and dads",
            "Kings and dragons",
        )
        for index in range(8)
    )
    train_path, validation_path, source = _pinned_tinystories_fixture(
        tmp_path / "insufficient",
        _aggregate("Dogs and cats train story"),
        _aggregate(*validation_stories),
    )
    with pytest.raises(ValueError, match="vehicles_tools.*0 train stories"):
        load_tinystories_topic_dataset(
            train_path,
            validation_path,
            source,
            StorySplitCounts(train=1, validation=1, test=1),
        )


def test_ascii_letter_permutations_are_seeded_bijective_and_format_preserving() -> None:
    text = "Aa Zz!\n123 é — punctuation."

    for seed in range(4):
        mapping = ascii_letter_permutation(seed)
        permuted = apply_ascii_letter_permutation(text, seed)

        assert mapping == ascii_letter_permutation(seed)
        assert tuple(sorted(mapping)) == tuple(ascii_lowercase)
        assert permuted == apply_ascii_letter_permutation(text, seed)
        assert permuted[1] == permuted[0].lower()
        assert "!\n123 é — " in permuted
        assert permuted.count("\n") == text.count("\n")


def test_tiny_shakespeare_curriculum_has_exact_four_seeded_tasks() -> None:
    source = CorpusSplits(
        train="To be, or not to be.\n",
        validation="That is the question!\n",
        test="Act 1, Scene 2.\n",
    )

    curriculum = build_tiny_shakespeare_permutation_curriculum(source)

    assert tuple(task.seed for task in curriculum.tasks) == (0, 1, 2, 3)
    assert len({task.splits.train for task in curriculum.tasks}) == 4
    for task in curriculum.tasks:
        assert task.splits.train == apply_ascii_letter_permutation(
            source.train,
            task.seed,
        )
        assert task.splits.validation == apply_ascii_letter_permutation(
            source.validation,
            task.seed,
        )
        assert task.splits.test == apply_ascii_letter_permutation(
            source.test,
            task.seed,
        )


def test_tiny_shakespeare_regions_preserve_raw_spans_exactly() -> None:
    source = CorpusSplits(
        train="First\r\nregion\twith  spacing and βeta.\n",
        validation="Validation\nkeeps\r\nformatting intact.",
        test="Test — punctuation, tabs\tand newlines\nremain.",
    )

    curriculum = build_tiny_shakespeare_region_curriculum(source)

    assert isinstance(curriculum, CorpusCurriculum)
    assert len(curriculum.tasks) == 4
    for split_name in ("train", "validation", "test"):
        regions = tuple(
            getattr(task.splits, split_name)[0] for task in curriculum.tasks
        )
        assert "".join(regions) == getattr(source, split_name)
        assert max(map(len, regions)) - min(map(len, regions)) <= 1


def _macro_document_for_task(
    task_index: int,
    identity: str,
    document_characters: int,
) -> str:
    for nonce in range(100_000):
        prefix = f"{identity}:{nonce:05d}\n"
        candidate = (prefix + (" raw\t" * document_characters))[
            :document_characters
        ]
        if stable_hash_raw_text_task_index(candidate) == task_index:
            return candidate
    raise AssertionError("could not construct a macro-document for the task")


def test_tiny_shakespeare_stable_hash_uses_raw_fixed_macro_documents() -> None:
    document_characters = 64
    split_documents = {
        split_name: tuple(
            _macro_document_for_task(
                task_index,
                f"{split_name}-{repeat}-{task_index}",
                document_characters,
            )
            for repeat in range(2)
            for task_index in range(4)
        )
        for split_name in ("train", "validation", "test")
    }
    source = CorpusSplits(
        **{
            split_name: "".join(documents)
            for split_name, documents in split_documents.items()
        }
    )

    first = build_tiny_shakespeare_stable_hash_curriculum(
        source,
        macro_document_characters=document_characters,
    )
    repeated = build_tiny_shakespeare_stable_hash_curriculum(
        source,
        macro_document_characters=document_characters,
    )

    assert first == repeated
    assert TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS == 1_024
    assert raw_text_sha256("a  b\n") == sha256(b"a  b\n").hexdigest()
    assert raw_text_sha256("a  b\n") != raw_text_sha256("a b ")
    for split_name, expected_documents in split_documents.items():
        assigned = tuple(
            (task_index, document)
            for task_index, task in enumerate(first.tasks)
            for document in getattr(task.splits, split_name)
        )
        assert sorted(document for _, document in assigned) == sorted(
            expected_documents
        )
        assert all(len(document) == document_characters for _, document in assigned)
        assert all(
            task_index == stable_hash_raw_text_task_index(document)
            for task_index, document in assigned
        )


def test_token_permutation_is_bijective_seeded_and_preserves_fixed_ids() -> None:
    permutation = seeded_token_permutation(
        vocabulary_size=9,
        seed=3,
        fixed_token_ids=(0, 1, 8),
    )

    assert permutation == seeded_token_permutation(9, 3, (0, 1, 8))
    assert tuple(sorted(permutation)) == tuple(range(9))
    assert (permutation[0], permutation[1], permutation[8]) == (0, 1, 8)
    assert apply_token_permutation((0, 2, 5, 8), permutation) == (
        0,
        permutation[2],
        permutation[5],
        8,
    )
    with pytest.raises(ValueError, match="every vocabulary ID"):
        apply_token_permutation((0, 1), (0, 0))


def test_contiguous_regions_reconstruct_each_split_without_crossing_documents() -> None:
    splits = _document_splits(train_count=10, validation_count=9, test_count=7)

    curriculum = build_contiguous_region_curriculum(splits)

    for split_name in ("train", "validation", "test"):
        task_regions = tuple(
            getattr(task.splits, split_name) for task in curriculum.tasks
        )
        reconstructed = tuple(
            document for region in task_regions for document in region
        )
        assert reconstructed == getattr(splits, split_name)
        region_id_sets = tuple(
            {document.content_id for document in region}
            for region in task_regions
        )
        assert all(
            not region_id_sets[left] & region_id_sets[right]
            for left in range(4)
            for right in range(left + 1, 4)
        )


def test_stable_hash_negative_control_uses_only_content_ids() -> None:
    splits = _document_splits()

    first = build_stable_hash_curriculum(splits)
    repeated = build_stable_hash_curriculum(splits)

    assert first == repeated
    for split_name in ("train", "validation", "test"):
        assigned_documents = tuple(
            (task_index, document)
            for task_index, task in enumerate(first.tasks)
            for document in getattr(task.splits, split_name)
        )
        assert len(assigned_documents) == len(getattr(splits, split_name))
        assert all(
            task_index == stable_hash_task_index(document)
            for task_index, document in assigned_documents
        )


@pytest.mark.parametrize(
    ("story", "expected_topic"),
    (
        ("Dogs and cats played.", "animals"),
        ("The mice loved their pets.", "animals"),
        ("Bikes and trucks waited.", "vehicles_tools"),
        ("Moms and dads smiled.", "family_home"),
        ("Princesses met dragons.", "fantasy_royalty"),
    ),
)
def test_topic_classifier_supports_aliases_and_plurals(
    story: str,
    expected_topic: str,
) -> None:
    assignment = classify_tinystory_topic(story)

    assert assignment is not None
    assert assignment.topic == expected_topic
    assert len(assignment.matched_concepts) == 2


def test_topic_classifier_uses_whole_words_distinct_concepts_ties_and_margin() -> None:
    whole_word_scores = topic_scores("A dogmatic catalog had no animal words.")
    repeated_alias_assignment = classify_tinystory_topic(
        "A mouse saw mice beside two pets."
    )
    margin_assignment = classify_tinystory_topic(
        "A dog and cat bowed to a king."
    )

    assert all(score.distinct_concept_count == 0 for score in whole_word_scores)
    assert repeated_alias_assignment is not None
    assert repeated_alias_assignment.matched_concepts == ("mouse", "pet")
    assert classify_tinystory_topic("A dog appeared twice: dog, dog.") is None
    assert classify_tinystory_topic("A dog met a cat and a king met a queen.") is None
    assert margin_assignment is not None
    assert margin_assignment.topic == "animals"
    assert margin_assignment.runner_up_score == 1
    assert margin_assignment.margin == 1


def _topic_splits() -> DocumentSplits:
    phrases = (
        "Dogs and cats",
        "Cars and trucks",
        "Moms and dads",
        "Kings and dragons",
    )
    return DocumentSplits(
        train=normalized_documents(
            tuple(
                f"{phrase} train example {index}"
                for phrase in phrases
                for index in range(2)
            )
        ),
        validation=normalized_documents(
            tuple(f"{phrase} validation example" for phrase in phrases)
        ),
        test=normalized_documents(
            tuple(f"{phrase} test example" for phrase in phrases)
        ),
    )


def test_topic_curriculum_equalizes_by_lowest_hash_or_fails_exactly() -> None:
    splits = _topic_splits()

    curriculum = build_tinystories_topic_curriculum(
        splits,
        StorySplitCounts(train=1, validation=1, test=1),
    )

    assert len(curriculum.tasks) == 4
    for task in curriculum.tasks:
        topic = task.task_id.removeprefix("tinystories-topic-")
        eligible_train = tuple(
            document
            for document in splits.train
            if (
                (assignment := classify_tinystory_topic(document.text))
                is not None
                and assignment.topic == topic
            )
        )
        assert len(task.splits.train) == 1
        assert len(task.splits.validation) == 1
        assert len(task.splits.test) == 1
        assert task.splits.train[0].content_id == min(
            document.content_id for document in eligible_train
        )

    with pytest.raises(ValueError, match="requested exactly 3"):
        build_tinystories_topic_curriculum(
            splits,
            StorySplitCounts(train=3, validation=1, test=1),
        )


def test_evaluation_and_single_gpu_presets_are_exact() -> None:
    assert TINY_SHAKESPEARE_EVALUATION_PRESET.prefix_lengths == (32, 64, 128)
    assert TINY_SHAKESPEARE_EVALUATION_PRESET.suffix_length == 128
    assert TINYSTORIES_EVALUATION_PRESET.prefix_lengths == (16, 32, 64, 128)
    assert TINYSTORIES_EVALUATION_PRESET.suffix_length == 128
    assert TINYSTORIES_SINGLE_GPU_PRESET.task_count == 4
    assert TINYSTORIES_SINGLE_GPU_PRESET.stories_per_task == StorySplitCounts(
        10_000,
        128,
        128,
    )
    assert (
        TINYSTORIES_SINGLE_GPU_PRESET.context_length,
        TINYSTORIES_SINGLE_GPU_PRESET.lora_rank,
        TINYSTORIES_SINGLE_GPU_PRESET.lora_alpha,
        TINYSTORIES_SINGLE_GPU_PRESET.batch_size,
        TINYSTORIES_SINGLE_GPU_PRESET.adapter_steps_per_task,
    ) == (256, 8, 8.0, 32, 2_000)
    assert (
        TINYSTORIES_SINGLE_GPU_PRESET.parent_probe_count,
        TINYSTORIES_SINGLE_GPU_PRESET.content_key_probe_count,
        TINYSTORIES_SINGLE_GPU_PRESET.evaluation_examples_per_task_and_prefix,
    ) == (128, 128, 128)
    assert (
        TINYSTORIES_SINGLE_GPU_PRESET.max_nodes,
        TINYSTORIES_SINGLE_GPU_PRESET.max_edges,
        TINYSTORIES_SINGLE_GPU_PRESET.peak_device_memory_gib,
    ) == (5, 4, 12)
    assert TINYSTORIES_SINGLE_GPU_PRESET.evaluation is TINYSTORIES_EVALUATION_PRESET


def test_single_gpu_preset_rejects_inconsistent_or_nonpositive_budgets() -> None:
    with pytest.raises(ValueError, match="task_count must equal four"):
        replace(TINYSTORIES_SINGLE_GPU_PRESET, task_count=3)
    with pytest.raises(ValueError, match="max_edges"):
        replace(TINYSTORIES_SINGLE_GPU_PRESET, max_edges=3)
    with pytest.raises(ValueError, match="dimensions and budgets"):
        replace(TINYSTORIES_SINGLE_GPU_PRESET, lora_rank=0)
    with pytest.raises(ValueError, match="LoRA alpha"):
        replace(TINYSTORIES_SINGLE_GPU_PRESET, lora_alpha=float("nan"))
