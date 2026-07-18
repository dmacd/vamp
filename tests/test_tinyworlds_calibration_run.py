from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.continual.tinyworlds_calibration_run as calibration_run_module
from apm.continual.knowledge_training import KnowledgeParentSearchResult
from apm.continual.tinyworlds_calibration import (
    CalibrationDistractorPolicy,
    CalibrationIdentity,
    CalibrationTrialPurpose,
    CalibrationValidationRequest,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTestRequest,
    TinyWorldsCalibrationEvidence,
    TinyWorldsCalibrationConfig,
    calibration_binomial_evidence,
)
from apm.continual.tinyworlds_calibration_run import (
    CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
    CALIBRATION_REQUIRED_DEVICE_KIND,
    CALIBRATION_REQUIRED_PLATFORM,
    CalibrationResourceEvidence,
    TinyWorldsAcceleratorCalibrationEvaluator,
    TinyWorldsCalibrationExecutionPreset,
    TinyWorldsCalibrationPool,
    build_tinyworlds_calibration_pool,
    load_calibration_trial_resource_evidence,
    validate_calibration_resource_evidence,
)
from apm.continual.tinyworlds_calibration_profile import (
    calibration_artifact_tree_sha256,
)
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds.rendering import TinyWorldsRenderPreset
from apm.data.text.tinyworlds.schema import CandidateRole, DataSplit, QueryKind
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    init_candidate_lora_train_state,
)
from apm.memory.graph import NodeId, init_memory_graph


@dataclass(frozen=True)
class _HashTokenizer:
    @property
    def vocab_size(self) -> int:
        return 4_096

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        words = tuple(re.findall(r"\S+", text))
        tokens = tuple(
            token
            for word in words
            for token in self._word_tokens(word)
        )
        return tokens + ((self.eos_token_id,) if add_eos else ())

    @staticmethod
    def _word_tokens(word: str) -> tuple[int, ...]:
        normalized = word.strip(".,:;!?()[]{}\"'")
        if re.fullmatch(r"N[0-9a-f]{12}", normalized):
            return (2_048,) + tuple(
                2_049 + int(normalized[offset : offset + 2], 16)
                for offset in range(1, 13, 2)
            )
        return (
            2
            + int.from_bytes(sha256(word.encode("utf-8")).digest()[:2], "big")
            % 2_046,
        )

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _render_preset() -> TinyWorldsRenderPreset:
    return TinyWorldsRenderPreset(
        training_stories_per_task=4,
        validation_stories_per_task=1,
        test_stories_per_task=1,
        validation_query_groups_per_task=16,
        test_query_groups_per_task=16,
        root_validation_stories=1,
        story_token_count=256,
        context_length=256,
    )


def _cpu_resource_probe(target_bytes: int | None) -> CalibrationResourceEvidence:
    return CalibrationResourceEvidence(
        platform="cpu",
        device_kind="bounded-test-cpu",
        allocator_peak_bytes=0,
        allocator_peak_target_bytes=target_bytes,
    )


def _canonical_resource_evidence(
    peak_bytes: int,
) -> CalibrationResourceEvidence:
    return CalibrationResourceEvidence(
        platform=CALIBRATION_REQUIRED_PLATFORM,
        device_kind=CALIBRATION_REQUIRED_DEVICE_KIND,
        allocator_peak_bytes=peak_bytes,
        allocator_peak_target_bytes=CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
    )


def _trial_evidence() -> TinyWorldsCalibrationEvidence:
    snapshot = CommittedNodeSnapshot(
        node_id="calibration_seed",
        adapter_sha256="1" * 64,
        logits_sha256="2" * 64,
        answers_sha256="3" * 64,
    )
    perfect = calibration_binomial_evidence(4, 4)
    return TinyWorldsCalibrationEvidence(
        exact_kg=perfect,
        frozen_novel_binding=perfect,
        independent_direct_recall=perfect,
        frozen_one_hop=perfect,
        independent_one_hop=perfect,
        committed_node_stability=CommittedNodeStabilityEvidence(
            before=(snapshot,),
            after=(snapshot,),
        ),
        old_contextual_answer=perfect,
        revision_contextual_answer=perfect,
        paired_revision_consistency=perfect,
    )


def _write_unit_trial(
    target: Path,
    resource_evidence: CalibrationResourceEvidence,
) -> tuple[str, str, dict[str, object]]:
    artifact_id = "calibration-validation-00-unit"
    execution_sha256 = "a" * 64
    request_record: dict[str, object] = {"kind": "unit"}
    calibration_run_module._write_trial_artifact(
        target,
        artifact_id=artifact_id,
        execution_sha256=execution_sha256,
        request_record=request_record,
        evidence=_trial_evidence(),
        outcome=None,
        model_config=GptNeoConfig(
            vocab_size=16,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        ),
        lora_config=LoraConfig(rank=1, alpha=1.0),
        score_records=(),
        resource_evidence=resource_evidence,
    )
    return artifact_id, execution_sha256, request_record


def test_calibration_resource_validation_enforces_accelerator_contract() -> None:
    target = CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES
    for peak in (target - 1, target):
        validate_calibration_resource_evidence(
            _canonical_resource_evidence(peak),
            expected_target_bytes=target,
        )

    wrong_platform = replace(
        _canonical_resource_evidence(0),
        platform="cpu",
    )
    wrong_device = replace(
        _canonical_resource_evidence(0),
        device_kind="NVIDIA GeForce RTX 3090",
    )
    for evidence in (wrong_platform, wrong_device):
        with pytest.raises(RuntimeError, match="RTX 4090"):
            validate_calibration_resource_evidence(
                evidence,
                expected_target_bytes=target,
            )

    with pytest.raises(RuntimeError, match="peak statistics"):
        validate_calibration_resource_evidence(
            replace(
                _canonical_resource_evidence(0),
                allocator_peak_bytes=None,
            ),
            expected_target_bytes=target,
        )
    with pytest.raises(MemoryError, match="exceeds calibration target"):
        validate_calibration_resource_evidence(
            _canonical_resource_evidence(target + 1),
            expected_target_bytes=target,
        )
    with pytest.raises(ValueError, match="allocator target changed"):
        validate_calibration_resource_evidence(
            _canonical_resource_evidence(0),
            expected_target_bytes=target - 1,
        )

    cpu_evidence = _cpu_resource_probe(None)
    validate_calibration_resource_evidence(
        cpu_evidence,
        expected_target_bytes=None,
    )


def test_trial_artifact_persists_resource_evidence_and_binds_cache_identity(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trial"
    resource_evidence = _cpu_resource_probe(None)
    artifact_id, execution_sha256, request_record = _write_unit_trial(
        target,
        resource_evidence,
    )

    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["resource_evidence"] == {
        "allocator_peak_bytes": 0,
        "allocator_peak_target_bytes": None,
        "device_kind": "bounded-test-cpu",
        "platform": "cpu",
    }
    assert load_calibration_trial_resource_evidence(
        target,
        expected_target_bytes=None,
    ) == resource_evidence
    loaded_evidence, loaded_outcome = calibration_run_module._load_trial_artifact(
        target,
        artifact_id,
        execution_sha256,
        request_record=request_record,
        runtime_resource_evidence=resource_evidence,
    )
    assert loaded_evidence == _trial_evidence()
    assert loaded_outcome is None
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        calibration_artifact_tree_sha256(target),
    )

    manifest["resource_evidence"]["device_kind"] = "different-test-cpu"
    calibration_run_module._write_canonical_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="resource identity"):
        calibration_run_module._load_trial_artifact(
            target,
            artifact_id,
            execution_sha256,
            request_record=request_record,
            runtime_resource_evidence=resource_evidence,
        )

    del manifest["resource_evidence"]
    calibration_run_module._write_canonical_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="fields changed"):
        load_calibration_trial_resource_evidence(
            target,
            expected_target_bytes=None,
        )


def test_trial_artifact_rejects_peak_before_publication(tmp_path: Path) -> None:
    target = tmp_path / "over-limit"
    with pytest.raises(MemoryError, match="exceeds calibration target"):
        _write_unit_trial(
            target,
            _canonical_resource_evidence(
                CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES + 1
            ),
        )
    assert not target.exists()


def _assert_evidence_slices_are_position_balanced(
    pool: TinyWorldsCalibrationPool,
) -> None:
    for split in (DataSplit.VALIDATION, DataSplit.TEST):
        slices = (
            tuple(
                group.variants[0].knowledge_query
                for group in pool.rendered.query_groups
                if group.split is split
                and group.variants[0].knowledge_query.query_kind
                == QueryKind.DIRECT.value
            ),
            tuple(
                group.variants[0].knowledge_query
                for group in pool.rendered.query_groups
                if group.split is split
                and group.variants[0].knowledge_query.query_kind
                == QueryKind.ONE_HOP.value
            ),
            tuple(
                variant.knowledge_query
                for group in pool.rendered.query_groups
                if group.split is split
                for variant in group.variants
                if variant.knowledge_query.query_kind
                == QueryKind.REVISION_SENSITIVE.value
                and variant.knowledge_query.cue_regime == "cue_sufficient"
            ),
        )
        for queries in slices:
            assert len(queries) >= 4
            counts = tuple(
                sum(query.correct_candidate_index == index for query in queries)
                for index in range(4)
            )
            assert max(counts) - min(counts) <= 1


def _pool_with_mutated_story(
    pool: TinyWorldsCalibrationPool,
    split: DataSplit,
) -> TinyWorldsCalibrationPool:
    story_index = next(
        index
        for index, story in enumerate(pool.rendered.stories)
        if story.split is split
    )
    story = pool.rendered.stories[story_index]
    text = f"{story.text} Selection-isolation-{split.value}."
    mutated_story = replace(
        story,
        text=text,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )
    stories = list(pool.rendered.stories)
    stories[story_index] = mutated_story
    rendered = replace(pool.rendered, stories=tuple(stories))
    return TinyWorldsCalibrationPool(
        hard_bundle=pool.hard_bundle,
        standard_bundle=pool.standard_bundle,
        rendered=rendered,
        standard_query_groups=pool.standard_query_groups,
        symbolic_bundle_sha256=pool.symbolic_bundle_sha256,
        validation_selection_sha256=(
            calibration_run_module._validation_selection_content_sha256(
                pool.hard_bundle,
                rendered,
                pool.standard_query_groups,
            )
        ),
        content_sha256=calibration_run_module._pool_content_sha256(
            pool.hard_bundle,
            rendered,
            pool.standard_query_groups,
            pool.symbolic_bundle_sha256,
        ),
    )


def _pool_with_mutated_validation_query(
    pool: TinyWorldsCalibrationPool,
    policy: CalibrationDistractorPolicy,
) -> TinyWorldsCalibrationPool:
    source_groups = pool.groups(policy)
    group_index = next(
        index
        for index, group in enumerate(source_groups)
        if group.split is DataSplit.VALIDATION
    )
    group = source_groups[group_index]
    variant = group.variants[0]
    text = f"{variant.prefix_text} Policy-{policy.value}-mutation."
    mutated_variant = replace(
        variant,
        prefix_text=text,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )
    mutated_group = replace(
        group,
        variants=(mutated_variant, *group.variants[1:]),
    )
    groups = list(source_groups)
    groups[group_index] = mutated_group
    if policy is CalibrationDistractorPolicy.HARD:
        rendered = replace(pool.rendered, query_groups=tuple(groups))
        standard_groups = pool.standard_query_groups
    else:
        rendered = pool.rendered
        standard_groups = tuple(groups)
    return TinyWorldsCalibrationPool(
        hard_bundle=pool.hard_bundle,
        standard_bundle=pool.standard_bundle,
        rendered=rendered,
        standard_query_groups=standard_groups,
        symbolic_bundle_sha256=pool.symbolic_bundle_sha256,
        validation_selection_sha256=(
            calibration_run_module._validation_selection_content_sha256(
                pool.hard_bundle,
                rendered,
                standard_groups,
            )
        ),
        content_sha256=calibration_run_module._pool_content_sha256(
            pool.hard_bundle,
            rendered,
            standard_groups,
            pool.symbolic_bundle_sha256,
        ),
    )


def test_calibration_pool_reuses_one_story_corpus_for_both_candidate_policies() -> None:
    pool = build_tinyworlds_calibration_pool(
        "7" * 64,
        _HashTokenizer(),
        render_preset=_render_preset(),
    )

    assert all(len(task.direct_fact_ids) == 36 for task in pool.hard_bundle.tasks)
    assert pool.hard_bundle.story_plans == pool.standard_bundle.story_plans
    assert tuple(group.group_id for group in pool.rendered.query_groups) == tuple(
        group.group_id for group in pool.standard_query_groups
    )
    assert any(
        tuple(
            candidate.answer_text
            for candidate in hard.variants[0].knowledge_query.candidates
        )
        != tuple(
            candidate.answer_text
            for candidate in standard.variants[0].knowledge_query.candidates
        )
        for hard, standard in zip(
            pool.rendered.query_groups,
            pool.standard_query_groups,
        )
    )
    hard_view = pool.rendered_for_policy(CalibrationDistractorPolicy.HARD)
    standard_view = pool.rendered_for_policy(
        CalibrationDistractorPolicy.STANDARD_MIX
    )
    assert hard_view.stories is pool.rendered.stories
    assert standard_view.stories is pool.rendered.stories
    _assert_evidence_slices_are_position_balanced(pool)


def test_validation_identity_and_rng_are_isolated_from_test_content(
    tmp_path: Path,
) -> None:
    tokenizer = _HashTokenizer()
    pool = build_tinyworlds_calibration_pool(
        "6" * 64,
        tokenizer,
        render_preset=_render_preset(),
    )
    test_mutated = _pool_with_mutated_story(pool, DataSplit.TEST)
    validation_mutated = _pool_with_mutated_story(
        pool,
        DataSplit.VALIDATION,
    )
    assert test_mutated.content_sha256 != pool.content_sha256
    assert (
        test_mutated.validation_selection_sha256
        == pool.validation_selection_sha256
    )
    assert validation_mutated.content_sha256 != pool.content_sha256
    assert (
        validation_mutated.validation_selection_sha256
        != pool.validation_selection_sha256
    )
    for policy in (
        CalibrationDistractorPolicy.HARD,
        CalibrationDistractorPolicy.STANDARD_MIX,
    ):
        query_mutated = _pool_with_mutated_validation_query(pool, policy)
        assert query_mutated.content_sha256 != pool.content_sha256
        assert (
            query_mutated.validation_selection_sha256
            != pool.validation_selection_sha256
        )

    model_config = GptNeoConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=256,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=1,
        attention_types=("global",),
        local_window_size=16,
    )
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    base_checksum = parameter_checksum(base_params, model_config)
    checkpoint = BaseCheckpointRef(tmp_path / "base", "a" * 64, base_checksum)
    identity = CalibrationIdentity(
        benchmark_version="tinyworlds-v1",
        public_seed=0,
        calibration_bundle_sha256=pool.symbolic_bundle_sha256,
        base_manifest_sha256=checkpoint.manifest_sha256,
        base_parameter_checksum=checkpoint.parameter_checksum,
        tokenizer_sha256="b" * 64,
    )

    wrong_device_root = tmp_path / "wrong-device"
    with pytest.raises(RuntimeError, match="RTX 4090"):
        TinyWorldsAcceleratorCalibrationEvaluator(
            identity,
            pool,
            tokenizer,
            checkpoint,
            base_params,
            model_config,
            wrong_device_root,
            execution_preset=TinyWorldsCalibrationExecutionPreset(
                batch_size=1,
                evaluation_examples_per_task=1,
                evaluation_microbatch_size=2,
                allocator_peak_target_bytes=(
                    CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES
                ),
            ),
            resource_probe=_cpu_resource_probe,
        )
    assert not wrong_device_root.exists()

    def evaluator(
        candidate_pool: TinyWorldsCalibrationPool,
        name: str,
    ) -> TinyWorldsAcceleratorCalibrationEvaluator:
        return TinyWorldsAcceleratorCalibrationEvaluator(
            identity,
            candidate_pool,
            tokenizer,
            checkpoint,
            base_params,
            model_config,
            tmp_path / name,
            execution_preset=TinyWorldsCalibrationExecutionPreset(
                batch_size=1,
                evaluation_examples_per_task=1,
                evaluation_microbatch_size=2,
                allocator_peak_target_bytes=None,
            ),
            resource_probe=_cpu_resource_probe,
        )

    config = TinyWorldsCalibrationConfig(
        facts_per_task=2,
        exposures_per_fact=1,
        update_budget=1,
        lora_rank=1,
        distractor_policy=CalibrationDistractorPolicy.HARD,
    )
    validation_request = CalibrationValidationRequest(
        trial_index=10,
        purpose=CalibrationTrialPurpose.LOCKED_SCRATCH,
        config=config,
        locked_scratch_rerun=True,
    )
    original_evaluator = evaluator(pool, "original")
    test_evaluator = evaluator(test_mutated, "test-mutated")
    validation_evaluator = evaluator(
        validation_mutated,
        "validation-mutated",
    )
    original_execution = original_evaluator._execution_sha256(
        validation_request
    )
    test_execution = test_evaluator._execution_sha256(validation_request)
    validation_execution = validation_evaluator._execution_sha256(
        validation_request
    )
    assert test_execution == original_execution
    assert validation_execution != original_execution
    rng_label = "independent:calibration_seed"
    np.testing.assert_array_equal(
        calibration_run_module._rng_key(original_execution, rng_label),
        calibration_run_module._rng_key(test_execution, rng_label),
    )
    assert not np.array_equal(
        calibration_run_module._rng_key(original_execution, rng_label),
        calibration_run_module._rng_key(validation_execution, rng_label),
    )

    locked_request = LockedCalibrationTestRequest(
        config=config,
        validation_trial_index=10,
        validation_artifact_id="calibration-validation-10-selection-isolation",
    )
    assert original_evaluator._locked_test_execution_sha256(
        locked_request,
        original_execution,
    ) != test_evaluator._locked_test_execution_sha256(
        locked_request,
        test_execution,
    )


def _resume_fixture() -> tuple[
    GptNeoConfig,
    LoraConfig,
    LmTrainState,
]:
    model_config = GptNeoConfig(
        vocab_size=16,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=4,
        batch_size=1,
        weight_decay=0.0,
    )
    state = init_candidate_lora_train_state(
        init_lora_edge(jax.random.PRNGKey(1), model_config, lora_config),
        jax.random.PRNGKey(2),
        train_config,
    )
    return model_config, lora_config, state


def _independent_checkpoint_records(
    initial_state: LmTrainState,
    update: int,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "adapter_checksum": calibration_run_module._tree_checksum(
                initial_state.trainable
            ),
            "parent_node_id": "root",
            "roles": ["independent_root"],
            "stream": "independent",
            "task_id": "calibration_seed",
            "training_loss": None if value == 0 else float(value),
            "update": value,
            "validation_candidate_accuracy": 0.25,
            "validation_correct_nll": 1.0,
        }
        for value in calibration_run_module._checkpoint_targets(update)
    )


def _write_independent_chain(
    directory: Path,
    initial_state: LmTrainState,
    updates: tuple[int, ...],
) -> None:
    for update in updates:
        state = replace(
            initial_state,
            step=jnp.asarray(update, dtype=jnp.int32),
        )
        calibration_run_module._write_independent_resume(
            directory,
            "a" * 64,
            initial_state,
            state,
            _independent_checkpoint_records(initial_state, update),
        )


def _counterfactual_plan(
) -> calibration_run_module.ParentCounterfactualPlan:
    root = NodeId("root")
    task_id = calibration_run_module.TaskId("calibration_seed")
    parent_search = KnowledgeParentSearchResult(
        task_id=task_id,
        family_id="calibration",
        validation_suite_id="unit-validation",
        validation_suite_sha256="c" * 64,
        validation_query_ids=("unit-query",),
        node_ids=(root,),
        correct_candidate_nll_by_query_and_node=np.asarray(
            ((1.0,),),
            dtype=np.float32,
        ),
        mean_correct_candidate_nll=(1.0,),
        selected_node_index=0,
        selected_node_id=root,
    )
    context = calibration_run_module.KnowledgeParentContext(
        task_id=task_id,
        family_id="calibration",
        true_parent_node_id=root,
        node_family_ids=((root, None),),
    )
    return calibration_run_module.plan_parent_counterfactuals(
        parent_search,
        context,
    )


def _counterfactual_training(
    initial_state: LmTrainState,
    update: int,
    plan: calibration_run_module.ParentCounterfactualPlan,
    *,
    execution_sha256: str = "d" * 64,
) -> calibration_run_module.KnowledgeCounterfactualTraining:
    state = replace(
        initial_state,
        step=jnp.asarray(update, dtype=jnp.int32),
    )
    adapter_checksum = calibration_run_module._tree_checksum(
        state.trainable
    )
    checkpoints = tuple(
        calibration_run_module.TransferCheckpointDiagnostic(
            update=value,
            training_loss=None if value == 0 else float(value),
            validation_candidate_accuracy=0.25,
            validation_correct_nll=1.0,
            adapter_checksum=adapter_checksum,
        )
        for value in calibration_run_module._checkpoint_targets(update)
    )
    trial = calibration_run_module.ParentTransferTrialDiagnostic(
        parent_node_index=0,
        parent_node_id=NodeId("root"),
        roles=("root", "true_parent", "selected_parent"),
        parent_validation_mean_correct_nll=1.0,
        initial_state_checksum=calibration_run_module._tree_checksum(
            initial_state
        ),
        final_adapter_checksum=adapter_checksum,
        final_state_checksum=calibration_run_module._tree_checksum(state),
        final_update=update,
        step_losses=tuple(float(value) for value in range(1, update + 1)),
        checkpoints=checkpoints,
    )
    return calibration_run_module.KnowledgeCounterfactualTraining(
        diagnostics=calibration_run_module.KnowledgeTransferDiagnostics(
            plan=plan,
            trials=(trial,),
        ),
        final_states=(state,),
        execution_sha256=execution_sha256,
    )


def _write_counterfactual_chain(
    directory: Path,
    initial_state: LmTrainState,
) -> calibration_run_module.ParentCounterfactualPlan:
    plan = _counterfactual_plan()
    calibration_run_module._write_counterfactual_initial_resume(
        directory,
        "a" * 64,
        initial_state,
    )
    for update in (1, 2):
        training = _counterfactual_training(initial_state, update, plan)
        calibration_run_module._write_counterfactual_resume(
            directory,
            "a" * 64,
            training,
        )
    return plan


def test_independent_resume_requires_exact_prefix_initial_state_and_history(
    tmp_path: Path,
) -> None:
    _, _, initial_state = _resume_fixture()
    valid = tmp_path / "valid"
    _write_independent_chain(valid, initial_state, (0, 1, 2))
    loaded = calibration_run_module._load_independent_resume(
        valid,
        "a" * 64,
        initial_state,
        4,
    )
    assert loaded is not None
    assert int(loaded[0].step) == 2
    assert tuple(record["update"] for record in loaded[1]) == (0, 1, 2)

    missing = tmp_path / "missing"
    _write_independent_chain(missing, initial_state, (0, 2))
    with pytest.raises(ValueError, match="exact checkpoint prefix"):
        calibration_run_module._load_independent_resume(
            missing,
            "a" * 64,
            initial_state,
            4,
        )

    changed_template = replace(
        initial_state,
        rng_key=jax.random.PRNGKey(99),
    )
    with pytest.raises(ValueError, match="identity changed"):
        calibration_run_module._load_independent_resume(
            valid,
            "a" * 64,
            changed_template,
            4,
        )

    schedule = tmp_path / "schedule"
    _write_independent_chain(schedule, initial_state, (0, 1))
    manifest_path = schedule / "update-0000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = manifest["records"][:1]
    calibration_run_module._write_canonical_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="exact schedule"):
        calibration_run_module._load_independent_resume(
            schedule,
            "a" * 64,
            initial_state,
            4,
        )


def test_counterfactual_resume_binds_every_execution_and_history_prefix(
    tmp_path: Path,
) -> None:
    _, _, initial_state = _resume_fixture()
    execution_changed = tmp_path / "execution-changed"
    plan = _write_counterfactual_chain(execution_changed, initial_state)
    loaded = calibration_run_module._load_counterfactual_resume(
        execution_changed,
        "a" * 64,
        plan,
        initial_state,
        4,
    )
    assert isinstance(
        loaded,
        calibration_run_module.KnowledgeCounterfactualTraining,
    )
    assert loaded.diagnostics.trials[0].final_update == 2
    earlier_manifest_path = (
        execution_changed / "update-0000001" / "manifest.json"
    )
    earlier_manifest = json.loads(
        earlier_manifest_path.read_text(encoding="utf-8")
    )
    earlier_manifest["training_execution_sha256"] = "e" * 64
    calibration_run_module._write_canonical_json(
        earlier_manifest_path,
        earlier_manifest,
    )
    with pytest.raises(ValueError, match="training execution history"):
        calibration_run_module._load_counterfactual_resume(
            execution_changed,
            "a" * 64,
            plan,
            initial_state,
            4,
        )

    history_changed = tmp_path / "history-changed"
    plan = _write_counterfactual_chain(history_changed, initial_state)
    earlier_manifest_path = (
        history_changed / "update-0000001" / "manifest.json"
    )
    earlier_manifest = json.loads(
        earlier_manifest_path.read_text(encoding="utf-8")
    )
    earlier_manifest["diagnostics"]["trials"][0]["checkpoints"][0][
        "validation_correct_nll"
    ] = 2.0
    calibration_run_module._write_canonical_json(
        earlier_manifest_path,
        earlier_manifest,
    )
    with pytest.raises(ValueError, match="diagnostic history"):
        calibration_run_module._load_counterfactual_resume(
            history_changed,
            "a" * 64,
            plan,
            initial_state,
            4,
        )


def test_resume_chain_validates_earlier_chunks_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    _, _, initial_state = _resume_fixture()
    resume = tmp_path / "resume"
    _write_independent_chain(resume, initial_state, (0, 1, 2))
    earlier_tensor = (
        resume
        / "update-0000001"
        / "training_state"
        / "state.safetensors"
    )
    earlier_tensor.write_bytes(earlier_tensor.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="payload checksum"):
        calibration_run_module._load_independent_resume(
            resume,
            "a" * 64,
            initial_state,
            4,
        )

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        calibration_run_module._load_latest_resume_directory(linked_root, 4)

    dangling_root = tmp_path / "dangling-root"
    dangling_root.mkdir()
    (dangling_root / "update-0000000").symlink_to(
        tmp_path / "absent-resume-target",
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="targets cannot be symlinks"):
        calibration_run_module._load_latest_resume_directory(dangling_root, 4)


def test_training_chunks_validate_every_entry_and_reject_dangling_targets(
    tmp_path: Path,
) -> None:
    model_config, lora_config, _ = _resume_fixture()
    outcome = calibration_run_module._TrainingOutcome(
        independent_adapters=(),
        graph=init_memory_graph(NodeId("root")),
        checkpoint_records=(),
        parent_records=(),
        stability_before=(),
    )
    workspace = tmp_path / "workspace"
    for chunk_index in range(2):
        calibration_run_module._write_training_chunk(
            workspace,
            chunk_index,
            f"unit:{chunk_index}",
            "b" * 64,
            outcome,
            model_config,
            lora_config,
        )
    earlier = (
        workspace
        / "chunks"
        / "chunk-000"
        / "parent_search.jsonl"
    )
    earlier.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="record checksum"):
        calibration_run_module._load_latest_training_chunk(
            workspace,
            "b" * 64,
            model_config,
            lora_config,
        )

    dangling = tmp_path / "dangling-workspace" / "chunks"
    dangling.mkdir(parents=True)
    (dangling / "chunk-000").symlink_to(
        tmp_path / "absent-chunk-target",
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="targets cannot be symlinks"):
        calibration_run_module._load_latest_training_chunk(
            dangling.parent,
            "b" * 64,
            model_config,
            lora_config,
        )

    writer_workspace = tmp_path / "writer-workspace"
    writer_chunks = writer_workspace / "chunks"
    writer_chunks.mkdir(parents=True)
    writer_target = writer_chunks / "chunk-000"
    writer_target.symlink_to(
        tmp_path / "absent-writer-target",
        target_is_directory=True,
    )
    with pytest.raises(
        FileExistsError,
        match="immutable calibration chunk already exists",
    ):
        calibration_run_module._write_training_chunk(
            writer_workspace,
            0,
            "unit:0",
            "b" * 64,
            outcome,
            model_config,
            lora_config,
        )
    assert writer_target.is_symlink()
    assert not writer_target.exists()


def _probe_batch(values: tuple[int, ...]) -> RouterBatch:
    inputs = np.asarray((values,), dtype=np.int32)
    return RouterBatch(
        input_ids=inputs,
        attention_mask=np.ones_like(inputs, dtype=np.bool_),
        target_ids=np.roll(inputs, -1, axis=1),
        loss_mask=np.ones_like(inputs, dtype=np.bool_),
    )


def test_stability_logits_hash_includes_non_first_validation_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_apply(_params, _config, input_ids, _attention_mask, **_kwargs):
        values = np.asarray(input_ids, dtype=np.float32)
        calls.append(values.shape[0])
        return SimpleNamespace(
            logits=jnp.asarray(
                np.repeat(values[:, :, None], 3, axis=2)
            )
        )

    monkeypatch.setattr(calibration_run_module, "apply_gpt_neo", fake_apply)
    probes = (
        SimpleNamespace(query_id="probe-0", router_batch=_probe_batch((1, 2, 3))),
        SimpleNamespace(query_id="probe-1", router_batch=_probe_batch((4, 5, 6))),
        SimpleNamespace(query_id="probe-2", router_batch=_probe_batch((7, 8, 9))),
    )
    baseline = calibration_run_module._validation_logits_sha256(
        object(),
        object(),
        object(),
        object(),
        np.zeros((1,), dtype=np.float32),
        probes,
        2,
    )
    changed = (
        probes[0],
        SimpleNamespace(
            query_id="probe-1",
            router_batch=_probe_batch((4, 5, 10)),
        ),
        probes[2],
    )
    assert calibration_run_module._validation_logits_sha256(
        object(),
        object(),
        object(),
        object(),
        np.zeros((1,), dtype=np.float32),
        changed,
        2,
    ) != baseline
    assert calls == [2, 1, 2, 1]


def test_resume_discovery_ignores_only_known_stale_atomic_directories(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / ".update-0000001.tmp-stale_1").mkdir()
    assert calibration_run_module._load_latest_resume_directory(resume, 4) is None

    workspace = tmp_path / "workspace"
    chunks = workspace / "chunks"
    chunks.mkdir(parents=True)
    (chunks / ".chunk.tmp-stale_2").mkdir()
    assert calibration_run_module._load_latest_training_chunk(
        workspace,
        "a" * 64,
        GptNeoConfig(
            vocab_size=16,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        ),
        LoraConfig(rank=1, alpha=1.0),
    ) is None

    (resume / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unexpected calibration resume entry"):
        calibration_run_module._load_latest_resume_directory(resume, 4)


@pytest.mark.integration
def test_tiny_model_calibration_trial_is_measured_cached_and_test_locked(
    tmp_path: Path,
) -> None:
    tokenizer = _HashTokenizer()
    pool = build_tinyworlds_calibration_pool(
        "8" * 64,
        tokenizer,
        render_preset=_render_preset(),
    )
    _assert_evidence_slices_are_position_balanced(pool)
    model_config = GptNeoConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=256,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=16,
    )
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    base_checksum = parameter_checksum(base_params, model_config)
    checkpoint = BaseCheckpointRef(tmp_path / "base", "a" * 64, base_checksum)
    identity = CalibrationIdentity(
        benchmark_version="tinyworlds-v1",
        public_seed=0,
        calibration_bundle_sha256=pool.symbolic_bundle_sha256,
        base_manifest_sha256=checkpoint.manifest_sha256,
        base_parameter_checksum=checkpoint.parameter_checksum,
        tokenizer_sha256="b" * 64,
    )
    evaluator = TinyWorldsAcceleratorCalibrationEvaluator(
        identity,
        pool,
        tokenizer,
        checkpoint,
        base_params,
        model_config,
        tmp_path / "artifacts",
        execution_preset=TinyWorldsCalibrationExecutionPreset(
            batch_size=1,
            evaluation_examples_per_task=1,
            evaluation_microbatch_size=2,
            allocator_peak_target_bytes=None,
        ),
        resource_probe=_cpu_resource_probe,
    )
    rotated_old_indices: list[tuple[int, int]] = []
    for group in pool.rendered.query_groups:
        query = group.variants[0].knowledge_query
        if (
            group.split is not DataSplit.VALIDATION
            or query.query_kind != QueryKind.REVISION_SENSITIVE.value
        ):
            continue
        plan = next(
            value
            for value in pool.hard_bundle.query_plans
            if str(value.query_ast.query_id) == group.symbolic_query_id
        )
        symbolic_index = next(
            index
            for index, candidate in enumerate(plan.candidates)
            if candidate.role is CandidateRole.INCOMPATIBLE_REVISION
        )
        entity_id = str(plan.candidates[symbolic_index].entity_id)
        rendered_index = group.variants[0].candidate_entity_ids.index(entity_id)
        assert evaluator._candidate_role_index(
            CalibrationDistractorPolicy.HARD,
            query.query_id,
            CandidateRole.INCOMPATIBLE_REVISION,
        ) == rendered_index
        rotated_old_indices.append((symbolic_index, rendered_index))
    assert any(
        symbolic_index != rendered_index
        for symbolic_index, rendered_index in rotated_old_indices
    )
    config = TinyWorldsCalibrationConfig(
        facts_per_task=2,
        exposures_per_fact=1,
        update_budget=1,
        lora_rank=1,
        distractor_policy=CalibrationDistractorPolicy.HARD,
    )
    validation_request = CalibrationValidationRequest(
        trial_index=10,
        purpose=CalibrationTrialPurpose.LOCKED_SCRATCH,
        config=config,
        locked_scratch_rerun=True,
    )

    validation = evaluator.evaluate_validation(validation_request)
    cached_validation = evaluator.evaluate_validation(validation_request)

    assert cached_validation == validation
    assert validation.execution_sha256 == evaluator._execution_sha256(
        validation_request
    )
    assert validation.evidence.exact_kg.rate == 1.0
    assert validation.evidence.committed_node_stability.bit_identical
    artifact_directory = (
        tmp_path / "artifacts" / "validation" / validation.artifact_id
    )
    assert validation.artifact_sha256 == calibration_artifact_tree_sha256(
        artifact_directory
    )
    assert load_calibration_trial_resource_evidence(
        artifact_directory,
        expected_target_bytes=None,
    ) == _cpu_resource_probe(None)
    validation_manifest = json.loads(
        (artifact_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert validation_manifest["schema_version"] == 2
    assert validation_manifest["resource_evidence"] == {
        "allocator_peak_bytes": 0,
        "allocator_peak_target_bytes": None,
        "device_kind": "bounded-test-cpu",
        "platform": "cpu",
    }
    parent_rows = tuple(
        json.loads(line)
        for line in (artifact_directory / "parent_search.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert parent_rows
    assert all(
        ":validation:" in query_id
        for row in parent_rows
        for query_id in row["validation_query_ids"]
    )
    checkpoint_rows = tuple(
        json.loads(line)
        for line in (artifact_directory / "checkpointed_transfer.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert {row["update"] for row in checkpoint_rows} == {0, 1}

    test_request = LockedCalibrationTestRequest(
        config=config,
        validation_trial_index=10,
        validation_artifact_id=validation.artifact_id,
    )
    locked_test = evaluator.evaluate_locked_test(test_request)
    cached_test = evaluator.evaluate_locked_test(test_request)

    assert cached_test == locked_test
    assert locked_test.execution_sha256 == evaluator._locked_test_execution_sha256(
        test_request,
        validation.execution_sha256,
    )
    assert locked_test.evidence.exact_kg.rate == 1.0
    test_artifact_directory = (
        tmp_path / "artifacts" / "test" / locked_test.artifact_id
    )
    assert test_artifact_directory.is_dir()
    assert locked_test.artifact_sha256 == calibration_artifact_tree_sha256(
        test_artifact_directory
    )
    assert load_calibration_trial_resource_evidence(
        test_artifact_directory,
        expected_target_bytes=None,
    ) == _cpu_resource_probe(None)
