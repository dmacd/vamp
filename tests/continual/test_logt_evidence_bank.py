from pathlib import Path

import pytest

from apm.continual.logt_evidence_bank import (
    EvidenceWorkCounters,
    empty_logt_state,
    evidence_update_bound,
    insert_block,
    require_evidence_work_bound,
)
from apm.experiments.vamp_logt_evidence_state import retire_inactive_node_artifacts


def _advance(blocks: int, block_size: int = 3):
    state = empty_logt_state(block_size)
    merge_rows = []
    for block in range(blocks):
        first = block * block_size
        state, leaf, merges = insert_block(
            state, tuple(range(first, first + block_size))
        )
        merge_rows.append((leaf, merges))
    return state, tuple(merge_rows)


def test_binary_counter_frontier_is_disjoint_complete_and_one_node_per_level() -> None:
    for blocks in range(1, 33):
        state, _rows = _advance(blocks)
        assert len(state.active_by_level) == blocks.bit_count()
        assert tuple(state.active_by_level) == tuple(set(state.active_by_level))
        assert tuple(
            block
            for node in state.active_nodes
            for block in range(node.first_block, node.last_block + 1)
        ) == tuple(range(blocks))
        assert tuple(
            example for node in state.active_nodes for example in node.example_ids
        ) == tuple(range(blocks * state.block_size))


def test_binary_counter_node_ids_and_carries_are_deterministic() -> None:
    left, left_rows = _advance(13)
    right, right_rows = _advance(13)
    assert tuple(node.node_id for node in left.active_nodes) == tuple(
        node.node_id for node in right.active_nodes
    )
    assert tuple(
        merge.parent.node_id for _leaf, merges in left_rows for merge in merges
    ) == tuple(
        merge.parent.node_id for _leaf, merges in right_rows for merge in merges
    )


def test_exact_training_accounting_obeys_fixed_t_log_t_ceiling() -> None:
    block_size, epochs, model_families = 5, 3, 2
    _state, rows = _advance(31, block_size)
    counters = EvidenceWorkCounters()
    for leaf, merges in rows:
        for node, is_merge in ((leaf, False), *((row.parent, True) for row in merges)):
            for _family in range(model_families):
                counters = counters.with_training(
                    epochs * len(node.example_ids), merge=is_merge
                )
    require_evidence_work_bound(
        counters, 31, block_size, epochs, model_families
    )
    assert counters.evidence_example_updates <= model_families * evidence_update_bound(
        31, block_size, epochs
    )
    ceiling = model_families * evidence_update_bound(31, block_size, epochs)
    excessive = EvidenceWorkCounters(ceiling + 1, 0, ceiling + 1, 0, 0)
    with pytest.raises(RuntimeError, match="exceeded fixed ceiling"):
        require_evidence_work_bound(
            excessive, 31, block_size, epochs, model_families
        )


def test_merge_retirement_deletes_only_inactive_node_directories(tmp_path: Path) -> None:
    state, rows = _advance(2)
    leaf_ids = {row[0].node_id for row in rows}
    parent_id = state.active_nodes[0].node_id
    nodes_root = tmp_path / "nodes"
    for node_id in (*leaf_ids, parent_id):
        directory = nodes_root / node_id / "evidence"
        directory.mkdir(parents=True)
        (directory / "model.pt").write_bytes(node_id.encode("ascii"))
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    retired = retire_inactive_node_artifacts(nodes_root, {parent_id})

    assert set(retired) == leaf_ids
    assert (nodes_root / parent_id / "evidence" / "model.pt").is_file()
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert all(not (nodes_root / node_id).exists() for node_id in leaf_ids)
