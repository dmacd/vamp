from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from apm.data.text.tinyworlds import (
    RenderedWorldArtifact,
    RenderedWorldMaterialization,
    TinyWorldsRenderPreset,
    build_rendered_materialization_result,
    file_sha256,
    generate_calibration_bundle,
    load_rendered_tinyworlds_bundle,
    materialize_or_verify_rendered_world,
    write_rendered_materialization_result,
    write_tinyworlds_bundle,
)


@dataclass(frozen=True)
class _WhitespaceTokenizer:
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
        tokens = tuple(
            2 + int.from_bytes(sha256(word.encode("utf-8")).digest()[:2], "big")
            for word in re.findall(r"\S+", text)
        )
        return tokens + ((self.eos_token_id,) if add_eos else ())

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _preset(query_groups: int = 1) -> TinyWorldsRenderPreset:
    return TinyWorldsRenderPreset(1, 1, 1, query_groups, query_groups, 1, 256, 256)


def test_materializer_atomically_creates_then_strictly_verifies_existing_bundle(
    tmp_path: Path,
) -> None:
    tokenizer = _WhitespaceTokenizer()
    symbolic = generate_calibration_bundle("d" * 64)
    symbolic_directory = tmp_path / "symbolic" / "calibration"
    rendered_directory = tmp_path / "rendered" / "calibration"
    write_tinyworlds_bundle(symbolic, symbolic_directory)

    created = materialize_or_verify_rendered_world(
        "calibration",
        symbolic_directory,
        rendered_directory,
        tokenizer,
        _preset(),
    )
    original_files = {
        path.relative_to(rendered_directory): path.read_bytes()
        for path in rendered_directory.iterdir()
    }
    verified = materialize_or_verify_rendered_world(
        "calibration",
        symbolic_directory,
        rendered_directory,
        tokenizer,
        _preset(),
    )

    assert created.action == "materialized"
    assert verified.action == "verified"
    assert created.artifact == verified.artifact
    assert original_files == {
        path.relative_to(rendered_directory): path.read_bytes()
        for path in rendered_directory.iterdir()
    }
    assert load_rendered_tinyworlds_bundle(
        rendered_directory,
        symbolic,
        tokenizer,
    ).preset == _preset()
    with pytest.raises(ValueError, match="different fixed render preset"):
        materialize_or_verify_rendered_world(
            "calibration",
            symbolic_directory,
            rendered_directory,
            tokenizer,
            _preset(query_groups=2),
        )


def test_materialization_result_is_canonical_content_only_and_immutable(
    tmp_path: Path,
) -> None:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"pinned tokenizer fixture\n")
    calibration_artifact = RenderedWorldArtifact(
        world_name="calibration",
        symbolic_bundle_id="tinyworlds-v1:calibration",
        symbolic_bundle_sha256="a" * 64,
        rendered_bundle_id="tinyworlds-v1:calibration:templates-v1",
        rendered_bundle_sha256="b" * 64,
        story_count=1,
        query_group_count=1,
    )
    pilot_artifact = replace(
        calibration_artifact,
        world_name="pilot",
        symbolic_bundle_id="tinyworlds-v1:pilot",
        rendered_bundle_id="tinyworlds-v1:pilot:templates-v1",
    )
    result = build_rendered_materialization_result(
        file_sha256(tokenizer_file),
        (
            RenderedWorldMaterialization(calibration_artifact, "materialized"),
            RenderedWorldMaterialization(pilot_artifact, "verified"),
        ),
    )
    result_with_other_actions = build_rendered_materialization_result(
        file_sha256(tokenizer_file),
        (
            RenderedWorldMaterialization(calibration_artifact, "verified"),
            RenderedWorldMaterialization(pilot_artifact, "materialized"),
        ),
    )
    output = write_rendered_materialization_result(result, tmp_path / "runtime")

    assert result == result_with_other_actions
    assert output.read_text(encoding="utf-8") == result.canonical_json + "\n"
    assert json.loads(result.canonical_json)["artifacts"][1]["world_name"] == "pilot"
    assert result.result_sha256 == sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_rendered_materialization_result(result, tmp_path / "runtime")
