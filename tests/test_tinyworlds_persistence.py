from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from apm.data.text.tinyworlds import (
    TinyWorldsBundleError,
    apply_standard_distractor_mix,
    derive_master_seed,
    generate_calibration_bundle,
    generate_pilot_bundle,
    load_tinyworlds_bundle,
    load_tinyworlds_manifest,
    tinyworlds_bundle_sha256,
    write_tinyworlds_bundle,
)


def _bundle():
    master_seed = derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )
    return generate_calibration_bundle(master_seed)


def _pilot_bundle():
    master_seed = derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )
    return generate_pilot_bundle(master_seed)


def _expanded_bundle():
    master_seed = derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )
    return generate_calibration_bundle(master_seed, direct_facts_per_task=36)


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


def _rewrite_manifest(root: Path, changes: dict[str, object]) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.update(changes)
    core = {
        key: value for key, value in manifest.items() if key != "bundle_sha256"
    }
    manifest["bundle_sha256"] = sha256(_canonical(core)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(record) for record in records)


def test_bundle_round_trip_is_byte_identical_and_immutable(tmp_path: Path) -> None:
    bundle = _bundle()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = write_tinyworlds_bundle(bundle, first)
    second_manifest = write_tinyworlds_bundle(bundle, second)

    assert first_manifest == second_manifest
    assert tinyworlds_bundle_sha256(bundle) == first_manifest.bundle_sha256
    assert load_tinyworlds_manifest(first) == first_manifest
    assert load_tinyworlds_bundle(first) == bundle
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.iterdir()
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.iterdir()
        if path.is_file()
    }
    assert first_files == second_files
    assert b"(TinyWorldsBundle " in first_files[Path("knowledge.metta")]
    assert b"(Rule " in first_files[Path("knowledge.metta")]
    assert b"(Query " in first_files[Path("knowledge.metta")]
    with pytest.raises(FileExistsError):
        write_tinyworlds_bundle(bundle, first)


def test_uniform_36_fact_resource_bundles_round_trip(tmp_path: Path) -> None:
    master_seed = derive_master_seed(
        "tinyworlds-v1",
        0,
        "a" * 64,
        "b" * 64,
    )
    bundles = (
        generate_calibration_bundle(master_seed, direct_facts_per_task=36),
        generate_pilot_bundle(master_seed, direct_facts_per_task=36),
    )
    for index, bundle in enumerate(bundles):
        root = tmp_path / f"bundle-{index}"
        write_tinyworlds_bundle(bundle, root)
        assert load_tinyworlds_bundle(root) == bundle


def test_loader_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    path = root / "entities.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"name":"N', b'"name":"X', 1))

    with pytest.raises(TinyWorldsBundleError, match="digest mismatch"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_unknown_nested_fields_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "entities.jsonl")
    records[0]["unexpected"] = "forbidden"
    payload = _jsonl(records)
    _rewrite_artifact(root, "entities.jsonl", payload)

    with pytest.raises(TinyWorldsBundleError, match="unknown=.*unexpected"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_story_reference_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "stories.jsonl")
    records[0]["direct_fact_ids"] = ["fact:does-not-exist"]
    payload = _jsonl(records)
    _rewrite_artifact(root, "stories.jsonl", payload)

    with pytest.raises(TinyWorldsBundleError, match="symbolic validation failed"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_duplicate_story_task_split_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "stories.jsonl")
    duplicate = dict(records[0])
    duplicate["story_id"] = "story:duplicate"
    records.append(duplicate)
    _rewrite_artifact(
        root,
        "stories.jsonl",
        _jsonl(records),
        record_count=len(records),
    )

    with pytest.raises(TinyWorldsBundleError, match="exactly once"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_noncanonical_story_slice_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "stories.jsonl")
    tasks = _read_jsonl(root / "tasks.jsonl")
    validation = next(item for item in records if item["split"] == "validation")
    task = next(item for item in tasks if item["task_id"] == validation["task_id"])
    validation["direct_fact_ids"] = task["direct_fact_ids"][1:9]
    _rewrite_artifact(root, "stories.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="canonical fixed split slice"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_manifest_bundle_id_not_bound_to_world(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    _rewrite_manifest(root, {"bundle_id": "tinyworlds-v1:other"})

    with pytest.raises(TinyWorldsBundleError, match="bundle_id must equal"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_nonpreset_world_id_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    _rewrite_manifest(
        root,
        {
            "bundle_id": "tinyworlds-v1:other",
            "world_id": "other",
        },
    )

    with pytest.raises(TinyWorldsBundleError, match="calibration or pilot preset"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_wrong_pilot_task_interleaving_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_pilot_bundle(), root)
    records = _read_jsonl(root / "tasks.jsonl")
    records[0], records[1] = records[1], records[0]
    _rewrite_artifact(root, "tasks.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="exact fixed v1 topology"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_bridge_link_fact_owned_outside_bridge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_pilot_bundle(), root)
    tasks = _read_jsonl(root / "tasks.jsonl")
    facts = _read_jsonl(root / "facts.jsonl")
    fact_by_id = {item["atom_id"]: item for item in facts}
    bridge = next(
        item
        for item in tasks
        if item["family_id"] == "willow" and item["kind"] == "bridge"
    )
    seed = next(
        item
        for item in tasks
        if item["family_id"] == "willow" and item["kind"] == "seed"
    )
    bridge_fact_id = next(
        fact_id
        for fact_id in bridge["direct_fact_ids"]
        if fact_by_id[fact_id]["predicate_id"] == "predicate:willow:bridge_link"
    )
    seed_fact_id = next(
        fact_id
        for fact_id in seed["direct_fact_ids"]
        if "filler" in fact_by_id[fact_id]["predicate_id"]
    )
    bridge_index = bridge["direct_fact_ids"].index(bridge_fact_id)
    seed_index = seed["direct_fact_ids"].index(seed_fact_id)
    bridge["direct_fact_ids"][bridge_index] = seed_fact_id
    seed["direct_fact_ids"][seed_index] = bridge_fact_id
    _rewrite_artifact(root, "tasks.jsonl", _jsonl(tasks))

    with pytest.raises(TinyWorldsBundleError, match="three family bridge-link"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_mixed_task_fact_capacities_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_expanded_bundle(), root)
    tasks = _read_jsonl(root / "tasks.jsonl")
    facts = _read_jsonl(root / "facts.jsonl")
    proofs = _read_jsonl(root / "proofs.jsonl")
    stories = _read_jsonl(root / "stories.jsonl")
    fact_by_id = {item["atom_id"]: item for item in facts}
    task = next(item for item in tasks if item["kind"] == "extension")
    removable = tuple(
        fact_id
        for fact_id in task["direct_fact_ids"][16:]
        if "filler" in fact_by_id[fact_id]["predicate_id"]
    )[:12]
    assert len(removable) == 12
    removed = set(removable)
    task["direct_fact_ids"] = [
        fact_id for fact_id in task["direct_fact_ids"] if fact_id not in removed
    ]
    facts = [item for item in facts if item["atom_id"] not in removed]
    proofs = [
        item for item in proofs if item["conclusion_atom_id"] not in removed
    ]
    training_story = next(
        item
        for item in stories
        if item["task_id"] == task["task_id"] and item["split"] == "train"
    )
    training_story["direct_fact_ids"] = list(task["direct_fact_ids"])
    metta_lines = (root / "knowledge.metta").read_bytes().splitlines(keepends=True)
    markers = tuple(f'(Fact "{fact_id}" '.encode("utf-8") for fact_id in removed)
    metta = b"".join(
        line for line in metta_lines if not line.startswith(markers)
    )
    _rewrite_artifact(root, "tasks.jsonl", _jsonl(tasks))
    _rewrite_artifact(
        root,
        "facts.jsonl",
        _jsonl(facts),
        record_count=len(facts),
    )
    _rewrite_artifact(
        root,
        "proofs.jsonl",
        _jsonl(proofs),
        record_count=len(proofs),
    )
    _rewrite_artifact(root, "stories.jsonl", _jsonl(stories))
    _rewrite_artifact(
        root,
        "knowledge.metta",
        metta,
        record_count=len(metta.splitlines()),
    )

    with pytest.raises(TinyWorldsBundleError, match="one direct-fact count"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_cross_family_revision_swap_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_pilot_bundle(), root)
    records = _read_jsonl(root / "revisions.jsonl")
    willow = next(
        index for index, item in enumerate(records) if item["family_id"] == "willow"
    )
    sunny = next(
        index for index, item in enumerate(records) if item["family_id"] == "sunny"
    )
    records[willow]["family_id"], records[sunny]["family_id"] = (
        records[sunny]["family_id"],
        records[willow]["family_id"],
    )
    _rewrite_artifact(root, "revisions.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="fact-owner task families"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_duplicate_revision_record_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_pilot_bundle(), root)
    records = _read_jsonl(root / "revisions.jsonl")
    willow_indices = tuple(
        index
        for index, item in enumerate(records)
        if item["family_id"] == "willow"
    )
    records[willow_indices[1]] = dict(records[willow_indices[0]])
    _rewrite_artifact(root, "revisions.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="distinct revision records"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_wrong_same_argument_revision_base_predicate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_pilot_bundle(), root)
    revisions = _read_jsonl(root / "revisions.jsonl")
    facts = _read_jsonl(root / "facts.jsonl")
    tasks = _read_jsonl(root / "tasks.jsonl")
    record = next(item for item in revisions if item["family_id"] == "willow")
    base = next(item for item in facts if item["atom_id"] == record["base_atom_id"])
    seed = next(
        item
        for item in tasks
        if item["family_id"] == "willow" and item["kind"] == "seed"
    )
    seed_fact_ids = set(seed["direct_fact_ids"])
    wrong_predicate = next(
        item
        for item in facts
        if item["atom_id"] in seed_fact_ids
        and item["arguments"] == base["arguments"]
        and item["predicate_id"] != base["predicate_id"]
    )
    record["base_atom_id"] = wrong_predicate["atom_id"]
    _rewrite_artifact(root, "revisions.jsonl", _jsonl(revisions))

    with pytest.raises(TinyWorldsBundleError, match="family base predicate"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_required_edge_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    records[0]["proof"]["required_edge_ids"] = ["edge:does-not-exist"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="authoritative fact/rule"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_open_book_fact_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    open_book = next(item for item in records if item["kind"] == "open_book")
    open_book["open_book_fact_ids"] = ["fact:does-not-exist"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="dangling fact IDs"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_unrelated_open_book_fact_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    open_book = next(item for item in records if item["kind"] == "open_book")
    unrelated = next(
        item
        for item in _read_jsonl(root / "facts.jsonl")
        if item["atom_id"] not in open_book["open_book_fact_ids"]
    )
    open_book["open_book_fact_ids"] = [unrelated["atom_id"]]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="canonical proof's supporting"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_unrelated_canonical_proof_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    direct = next(
        item
        for item in records
        if item["split"] == "validation" and item["kind"] == "direct"
    )
    one_hop = next(
        item
        for item in records
        if item["split"] == "validation" and item["kind"] == "one_hop"
    )
    one_hop["proof"] = direct["proof"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="grounded unique answer"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_support_metadata_mismatch_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    direct = next(item for item in records if item["kind"] == "direct")
    direct["proof"]["required_edge_ids"] = ["edge:calibration_extension"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="authoritative fact/rule"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_candidate_role_relabeling_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    direct = next(item for item in records if item["kind"] == "direct")
    for candidate in direct["candidates"]:
        if candidate["role"] != "correct":
            candidate["role"] = "same_type_filler"
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="predefined distractor policy"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_skipped_higher_priority_candidate_even_when_rehashed(
    tmp_path: Path,
) -> None:
    hard = _bundle()
    standard = apply_standard_distractor_mix(hard)
    differing = next(
        plan
        for hard_plan, plan in zip(hard.query_plans, standard.query_plans)
        if plan.candidates != hard_plan.candidates
    )
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(hard, root)
    records = _read_jsonl(root / "queries.jsonl")
    record = next(
        item
        for item in records
        if item["query_ast"]["query_id"] == str(differing.query_ast.query_id)
    )
    record["candidates"] = [
        {"entity_id": str(candidate.entity_id), "role": candidate.role.value}
        for candidate in differing.candidates
    ]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="one complete predefined"):
        load_tinyworlds_bundle(root)


def test_complete_standard_candidate_policy_round_trips(tmp_path: Path) -> None:
    standard = apply_standard_distractor_mix(_bundle())
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(standard, root)

    assert load_tinyworlds_bundle(root) == standard


def test_loader_rejects_dangling_query_task_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    records[0]["task_id"] = "does-not-exist"
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="unknown owning task"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_hard_oracle_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    records[0]["hard_oracle_task_ids"] = ["does-not-exist"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="hard oracle references unknown"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_wrong_valid_hard_oracle_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    direct = next(item for item in records if item["kind"] == "direct")
    direct["hard_oracle_task_ids"] = ["calibration_extension"]
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="must be its owning task node"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_candidate_entity_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    records[0]["candidates"][1]["entity_id"] = "entity:does-not-exist"
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="candidates reference unknown"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_dangling_query_ast_entity_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "queries.jsonl")
    records[0]["query_ast"]["clauses"][0]["arguments"][0]["value"] = (
        "entity:does-not-exist"
    )
    _rewrite_artifact(root, "queries.jsonl", _jsonl(records))

    with pytest.raises(TinyWorldsBundleError, match="query AST references unknown"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_cross_split_overlap_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    records = _read_jsonl(root / "stories.jsonl")
    train = next(item for item in records if item["split"] == "train")
    validation = next(
        item
        for item in records
        if item["split"] == "validation" and item["task_id"] == train["task_id"]
    )
    validation["holdout"] = train["holdout"]
    payload = _jsonl(records)
    _rewrite_artifact(root, "stories.jsonl", payload)

    with pytest.raises(TinyWorldsBundleError, match="disjoint across splits"):
        load_tinyworlds_bundle(root)


def test_loader_rejects_unlisted_files_and_divergent_metta(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    write_tinyworlds_bundle(_bundle(), root)
    (root / "unlisted.txt").write_text("not part of the bundle", encoding="utf-8")
    with pytest.raises(TinyWorldsBundleError, match="unlisted"):
        load_tinyworlds_bundle(root)

    (root / "unlisted.txt").unlink()
    metta = (root / "knowledge.metta").read_bytes() + b";; divergent\n"
    _rewrite_artifact(
        root,
        "knowledge.metta",
        metta,
        record_count=len(metta.splitlines()),
    )
    with pytest.raises(TinyWorldsBundleError, match="authoritative"):
        load_tinyworlds_bundle(root)
