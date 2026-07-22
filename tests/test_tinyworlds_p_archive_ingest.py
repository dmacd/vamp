from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import pytest

from apm.data.text.tinyworlds_p import (
    ArchiveIngestError,
    HashedFile,
    NORMALIZATION_IDENTITY,
    PartitionInputs,
    PartitionPreset,
    SourceIdentity,
    TokenizerIdentity,
    build_archive_ingest,
    iter_archive_groups,
    read_spooled_story,
    read_spooled_tokens,
)
from apm.lm.text import TokenizersTextTokenizer


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, *, digest: str | None = None) -> SourceIdentity:
    return SourceIdentity(
        dataset_id="tinyworlds-p/fixture",
        revision="0" * 40,
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path) if digest is None else digest,
    )


def _record(
    story: str,
    *,
    noun: str = "cat",
    verb: str = "help",
    adjective: str = "kind",
    source: str = "GPT-4",
    prompt: str | None = None,
    words: list[str] | None = None,
    features: list[str] | None = None,
) -> dict[str, object]:
    return {
        "instruction": {
            "features": [] if features is None else features,
            "prompt:": (
                f'Use the verb "{verb}", the noun "{noun}", '
                f'and the adjective "{adjective}".'
                if prompt is None
                else prompt
            ),
            "words": [verb, noun, adjective] if words is None else words,
        },
        "source": source,
        "story": story,
        "summary": "Fixture summary.",
    }


def _write_archive(
    path: Path,
    members: tuple[tuple[str, object], ...],
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member_name, value in members:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))


def _tokenizer(tmp_path: Path) -> tuple[Path, TokenizerIdentity]:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocabulary = {
        "<unk>": 0,
        "<|endoftext|>": 1,
        "A": 2,
        "cat": 3,
        "robot": 4,
        "help": 5,
        "find": 6,
        "kind": 7,
        "bright": 8,
        "said": 9,
        "today": 10,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    path = directory / "tokenizer.json"
    tokenizer.save(str(path))
    hashed = HashedFile(path.name, path.stat().st_size, _sha256(path))
    return directory, TokenizerIdentity(
        kind="word-level-fixture",
        identifier="tinyworlds-p/fixture",
        revision="0" * 40,
        vocab_size=len(vocabulary),
        files=(hashed,),
    )


def _inputs(
    tmp_path: Path,
    archive_path: Path,
    tokenizer_directory: Path,
    tokenizer_identity: TokenizerIdentity,
    work_name: str,
    *,
    archive_identity: SourceIdentity | None = None,
) -> PartitionInputs:
    return PartitionInputs(
        archive_path=archive_path,
        tokenizer_directory=tokenizer_directory,
        output_root=tmp_path / f"output-{work_name}",
        temporary_directory=tmp_path / work_name,
        archive_identity=_identity(archive_path) if archive_identity is None else archive_identity,
        tokenizer_identity=tokenizer_identity,
    )


def test_archive_ingest_retains_multiplicity_exclusions_and_exact_bytes(
    tmp_path: Path,
) -> None:
    tokenizer_directory, tokenizer_identity = _tokenizer(tmp_path)
    quote_left = "The CAT said, “I’m kind.”"
    quote_right = 'the cat said, "i\'m   kind."'
    conflict_story = "A bright robot will help today."
    records = [
        _record("A kind cat will help today."),
        _record(quote_left, features=["dialogue"]),
        _record(quote_right, features=["dialogue"]),
        _record(conflict_story, noun="robot", adjective="bright"),
        _record(conflict_story, noun="cat", adjective="bright"),
        _record(
            "A robot will find today.",
            prompt="Use these three words in a story.",
            noun="robot",
            verb="find",
            adjective="bright",
        ),
        _record("   "),
    ]
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    _write_archive(archive_path, (("./data00.json", records),))
    inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "work",
    )

    result = build_archive_ingest(
        inputs,
        replace(
            PartitionPreset(),
            worker_count=1,
            run_record_count=2,
            minimum_role_coverage=0.5,
        ),
        NORMALIZATION_IDENTITY,
    )
    groups = tuple(iter_archive_groups(result.groups_path))

    assert [group["normalized_story_sha256"] for group in groups] == sorted(
        group["normalized_story_sha256"] for group in groups
    )
    assert result.audit.archive_member_count == 1
    assert result.audit.archive_record_count == 7
    assert result.audit.archive_group_count == 5
    assert result.audit.duplicate_group_count == 2
    assert result.audit.maximum_group_multiplicity == 2
    assert result.audit.empty_record_count == 1
    assert result.audit.unclassifiable_record_count == 1
    assert result.audit.conflicting_record_count == 2
    assert result.audit.eligible_record_count == 3
    assert sorted(group["status"] for group in groups) == [
        "conflicting_metadata",
        "eligible",
        "eligible",
        "empty_story",
        "unclassifiable_metadata",
    ]

    quote_group = next(
        group for group in groups if len(group["occurrences"]) == 2 and group["status"] == "eligible"
    )
    reconstructed = tuple(
        read_spooled_story(result.story_spool_path, occurrence).decode("utf-8")
        for occurrence in quote_group["occurrences"]
    )
    assert reconstructed == (quote_left, quote_right)
    assert tuple(item["record_id"] for item in quote_group["provenance"]) == tuple(
        item["record_id"] for item in quote_group["occurrences"]
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        tokenizer_directory / "tokenizer.json"
    )
    assert tuple(
        read_spooled_tokens(result.token_spool_path, occurrence)
        for occurrence in quote_group["occurrences"]
    ) == tuple(tokenizer.encode(story, add_eos=True) for story in reconstructed)


@pytest.mark.parametrize(
    "payload,match",
    [
        ([{**_record("A kind cat."), "extra": True}], "fields differ"),
        (
            [{**_record("A kind cat."), "instruction": {"prompt": "x", "words": [], "features": []}}],
            "instruction.*fields differ",
        ),
        ([_record("A kind cat.", source="unknown")], "unsupported released source"),
        ({"not": "an array"}, "must contain a JSON array"),
    ],
)
def test_archive_ingest_rejects_malformed_schema(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    tokenizer_directory, tokenizer_identity = _tokenizer(tmp_path)
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    _write_archive(archive_path, (("./data00.json", payload),))
    inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "work",
    )

    with pytest.raises(ArchiveIngestError, match=match):
        build_archive_ingest(
            inputs,
            replace(PartitionPreset(), worker_count=1, minimum_role_coverage=0.0),
            NORMALIZATION_IDENTITY,
        )


def test_archive_ingest_rejects_source_identity_tampering(tmp_path: Path) -> None:
    tokenizer_directory, tokenizer_identity = _tokenizer(tmp_path)
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    _write_archive(archive_path, (("./data00.json", [_record("A kind cat.")]),))
    inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "work",
        archive_identity=_identity(archive_path, digest="0" * 64),
    )

    with pytest.raises(ArchiveIngestError, match="archive identity changed"):
        build_archive_ingest(
            inputs,
            replace(PartitionPreset(), worker_count=1, minimum_role_coverage=0.0),
            NORMALIZATION_IDENTITY,
        )


def test_archive_role_gate_uses_nonempty_record_token_mass(tmp_path: Path) -> None:
    tokenizer_directory, tokenizer_identity = _tokenizer(tmp_path)
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    _write_archive(
        archive_path,
        (
            (
                "./data00.json",
                [
                    _record("A kind cat will help today."),
                    _record("A robot will find today.", prompt="No explicit roles."),
                    _record(""),
                ],
            ),
        ),
    )
    inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "work",
    )

    with pytest.raises(ArchiveIngestError, match="role-classification gate failed"):
        build_archive_ingest(
            inputs,
            replace(PartitionPreset(), worker_count=1, minimum_role_coverage=0.95),
            NORMALIZATION_IDENTITY,
        )

    audit = json.loads((inputs.temporary_directory / "archive-ingest.json").read_bytes())
    assert audit["passed"] is False
    assert audit["coverage"]["empty_record_count"] == 1
    assert audit["coverage"]["nonempty_record_count"] == 2
    assert audit["coverage"]["classified_record_count"] == 1


def test_archive_ingest_bytes_ignore_worker_completion_and_run_size(
    tmp_path: Path,
) -> None:
    tokenizer_directory, tokenizer_identity = _tokenizer(tmp_path)
    records = [
        _record(
            f"A kind cat will help today {index}.",
            source="GPT-3.5" if index % 2 else "GPT-4",
        )
        for index in range(300)
    ]
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    _write_archive(archive_path, (("./data00.json", records),))
    first_inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "first-work",
    )
    second_inputs = _inputs(
        tmp_path,
        archive_path,
        tokenizer_directory,
        tokenizer_identity,
        "second-work",
    )

    first = build_archive_ingest(
        first_inputs,
        replace(
            PartitionPreset(),
            worker_count=1,
            run_record_count=41,
            minimum_role_coverage=1.0,
        ),
        NORMALIZATION_IDENTITY,
    )
    second = build_archive_ingest(
        second_inputs,
        replace(
            PartitionPreset(),
            worker_count=2,
            run_record_count=37,
            minimum_role_coverage=1.0,
        ),
        NORMALIZATION_IDENTITY,
    )

    assert second.groups_path.read_bytes() == first.groups_path.read_bytes()
    assert second.audit_path.read_bytes() == first.audit_path.read_bytes()
    assert second.story_spool_path.read_bytes() == first.story_spool_path.read_bytes()
    assert second.token_spool_path.read_bytes() == first.token_spool_path.read_bytes()
