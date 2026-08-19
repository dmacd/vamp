from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from apm.continual.artifacts import publish_immutable_json
from apm.continual.trace.consolidation import consolidate_adapters
from apm.continual.trace.merging.common import (
    LoRAFactors,
    factors_from_peft_state,
    peft_state_from_factors,
)
from apm.continual.trace.merging.core_space import build_core_space
from apm.continual.trace.merging.core_tsv import core_tsv_merge
from apm.continual.trace.merging.svd_mean import weighted_svd_mean
from apm.continual.trace.merging.tsv import tsv_merge
from apm.continual.trace.protocol import MergePolicy


def _factor(seed: int, output_size: int = 11, input_size: int = 9, rank: int = 3) -> LoRAFactors:
    generator = torch.Generator().manual_seed(seed)
    return LoRAFactors(
        a=torch.randn(rank, input_size, generator=generator),
        b=torch.randn(output_size, rank, generator=generator),
        scale=4.0,
    )


def test_weighted_svd_mean_matches_dense_optimal_truncation_and_scale() -> None:
    children = (_factor(1), _factor(2))
    expected = 0.25 * children[0].delta() + 0.75 * children[1].delta()
    left, singular, right = torch.linalg.svd(expected.float(), full_matrices=False)
    expected_rank_four = left[:, :4] @ torch.diag(singular[:4]) @ right[:4]

    parent, diagnostics = weighted_svd_mean(
        children,
        (0.25, 0.75),
        output_rank=4,
        parent_scale=4.0,
    )

    torch.testing.assert_close(parent.delta(), expected_rank_four, atol=2e-5, rtol=2e-5)
    assert parent.rank == 4
    assert diagnostics.output_rank == 4
    assert 0.0 < diagnostics.retained_energy <= 1.0


def test_rank_sixteen_stack_exactly_matches_weighted_rank_eight_children() -> None:
    children = (
        _factor(20, output_size=24, input_size=20, rank=8),
        _factor(21, output_size=24, input_size=20, rank=8),
    )
    expected = 0.4 * children[0].delta() + 0.6 * children[1].delta()

    parent, diagnostics = weighted_svd_mean(
        children,
        (0.4, 0.6),
        output_rank=16,
        parent_scale=2.0,
    )

    torch.testing.assert_close(parent.delta(), expected, atol=4e-5, rtol=4e-5)
    assert diagnostics.output_rank == 16


def test_core_space_reconstructs_each_scaled_child_without_dense_alignment() -> None:
    children = (_factor(3), _factor(4))

    core = build_core_space(children)

    for child, aligned in zip(children, core.aligned_cores):
        reconstructed = core.left_basis @ aligned @ core.right_basis.T
        torch.testing.assert_close(reconstructed, child.delta(), atol=3e-5, rtol=3e-5)


def test_tsv_matches_pinned_reference_algebra_in_fp32() -> None:
    matrices = tuple(child.delta() for child in (_factor(5, 8, 8, 4), _factor(6, 8, 8, 4)))

    actual = tsv_merge(matrices)

    components = tuple(torch.linalg.svd(matrix.float(), full_matrices=False) for matrix in matrices)
    retained = components[0][1].shape[0] // 2
    sum_u = torch.zeros_like(components[0][0])
    sum_s = torch.zeros_like(components[0][1])
    sum_v = torch.zeros_like(components[0][2])
    for index, (u, s, v) in enumerate(components):
        block = slice(index * retained, (index + 1) * retained)
        sum_u[:, block] = u[:, :retained]
        sum_s[block] = s[:retained]
        sum_v[block] = v[:retained]
    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    expected = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))

    torch.testing.assert_close(actual, expected)


def test_core_tsv_stays_rank_bounded_and_precompresses_at_sum_rank() -> None:
    children = (_factor(7, rank=3), _factor(8, rank=3))

    result = core_tsv_merge(
        children,
        core_scale=0.3,
        output_rank=4,
        parent_scale=4.0,
        retain_precompress=True,
    )

    assert result.factors.rank == 4
    assert result.precompress_factors is not None
    assert result.precompress_factors.rank == 6
    assert torch.linalg.matrix_rank(result.factors.delta()).item() <= 4
    expected = result.left_basis @ result.merged_core @ result.right_basis.T
    torch.testing.assert_close(
        result.precompress_factors.delta(),
        expected,
        atol=3e-5,
        rtol=3e-5,
    )


def test_core_tsv_compact_output_matches_dense_optimal_truncation() -> None:
    children = (
        _factor(22, output_size=25, input_size=21, rank=8),
        _factor(23, output_size=25, input_size=21, rank=8),
    )

    result = core_tsv_merge(
        children,
        core_scale=0.5,
        output_rank=8,
        parent_scale=4.0,
    )

    dense = result.left_basis @ result.merged_core @ result.right_basis.T
    left, singular, right = torch.linalg.svd(dense, full_matrices=False)
    expected = left[:, :8] @ torch.diag(singular[:8]) @ right[:8]
    torch.testing.assert_close(result.factors.delta(), expected, atol=5e-5, rtol=5e-5)


def test_peft_factor_codec_accepts_default_adapter_keys() -> None:
    child = _factor(9)
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    state = {
        f"{module}.lora_A.default.weight": child.a,
        f"{module}.lora_B.default.weight": child.b,
    }

    parsed = factors_from_peft_state(state, scale=4.0)
    round_trip = peft_state_from_factors(parsed)

    assert tuple(parsed) == (module,)
    torch.testing.assert_close(parsed[module].delta(), child.delta())
    assert set(round_trip) == {
        f"{module}.lora_A.weight",
        f"{module}.lora_B.weight",
    }


def test_artifact_level_core_merge_saves_adapter_and_core_cache(tmp_path) -> None:
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    children = (_factor(10), _factor(11))
    paths = (
        tmp_path / "left" / "adapter.safetensors",
        tmp_path / "right" / "adapter.safetensors",
    )
    for path, child in zip(paths, children):
        path.parent.mkdir()
        save_file(peft_state_from_factors({module: child}), path)
        publish_immutable_json(
            path.parent / "adapter_config.json",
            {"lora_alpha": 12, "r": 3},
        )

    result = consolidate_adapters(
        paths[0],
        paths[1],
        100,
        100,
        MergePolicy("core_tsv_r8", core_scale=0.3),
        tmp_path / "parent",
        retain_precompress=True,
    )

    assert result.adapter_path.is_file()
    assert result.core_cache_path is not None and result.core_cache_path.is_file()
    parent = factors_from_peft_state(load_file(result.adapter_path), 4.0)
    assert parent[module].rank == 6


def test_artifact_level_merge_respects_each_child_lora_scale(tmp_path) -> None:
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    children = (_factor(12), _factor(13))
    paths = tuple(
        tmp_path / name / "adapter.safetensors" for name in ("left", "right")
    )
    child_scales = (2.0, 6.0)
    for path, child, scale in zip(paths, children, child_scales):
        path.parent.mkdir()
        save_file(peft_state_from_factors({module: child}), path)
        publish_immutable_json(
            path.parent / "adapter_config.json",
            {"lora_alpha": int(scale * child.rank), "r": child.rank},
        )

    result = consolidate_adapters(
        paths[0],
        paths[1],
        100,
        300,
        MergePolicy("svd_mean_r8"),
        tmp_path / "parent",
    )

    parent = factors_from_peft_state(load_file(result.adapter_path), 4.0)[module]
    expected = 0.25 * LoRAFactors(
        children[0].a, children[0].b, child_scales[0]
    ).delta() + 0.75 * LoRAFactors(
        children[1].a, children[1].b, child_scales[1]
    ).delta()
    torch.testing.assert_close(parent.delta(), expected, atol=3e-5, rtol=3e-5)
