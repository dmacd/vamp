"""Validation-only architecture calibration for the dense Permuted-MNIST base."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.dense_mlp_adapter import (
    DenseExamples,
    DenseFitResult,
    DenseMlpState,
    DenseMnistMLP,
    dense_metrics,
    dense_state,
    fit_dense_model,
    load_dense_state,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.data.mnist.loader import load_mnist
from apm.data.mnist.permutations import identity_permutation, random_digit_permutation
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_mlp_permuted_data import named_seed, stratified_source_split


def initialize_dense_state(
    widths: tuple[int, int, int],
    dropout: float,
    seed: int,
) -> DenseMlpState:
    """Create one deterministic PyTorch-default dense initialization."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return dense_state(DenseMnistMLP(widths, dropout))


def select_calibrated_width(
    rows: tuple[dict[str, object], ...],
    config: VampLogTDenseConfig,
) -> tuple[int, int, int] | None:
    """Choose the smallest eligible width using validation metrics only."""
    summaries = {
        widths: tuple(row for row in rows if tuple(row["hidden_widths"]) == widths)
        for widths in config.calibration.candidate_widths
    }
    if any(len(values) != 2 * len(config.calibration.seeds) for values in summaries.values()):
        raise ValueError("calibration rows do not cover every width, seed, and fit")
    if config.calibration.selection_policy == "smallest_candidate":
        return config.calibration.candidate_widths[0]
    pooled_means = {
        widths: float(np.mean([
            float(row["validation_accuracy"])
            for row in values
            if row["fit"] == "pooled"
        ]))
        for widths, values in summaries.items()
    }
    widest_mean = pooled_means[config.calibration.candidate_widths[-1]]
    for widths in config.calibration.candidate_widths:
        identity = tuple(row for row in summaries[widths] if row["fit"] == "identity")
        identity_mean = float(np.mean([float(row["validation_accuracy"]) for row in identity]))
        seed_zero = next(row for row in identity if int(row["seed"]) == 0)
        if (
            identity_mean >= config.calibration.identity_mean_accuracy_minimum
            and float(seed_zero["validation_accuracy"])
            >= config.calibration.identity_seed_zero_accuracy_minimum
            and pooled_means[widths]
            >= widest_mean - config.calibration.pooled_gap_from_widest_maximum
        ):
            return widths
    return None


def run_calibration(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Run or restore all width/seed fits and publish the selected frozen base."""
    summary_path = run_root / "calibration" / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    if config.calibration_evidence_run is not None:
        return _import_calibration_evidence(config, run_root, device)
    arrays = load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)
    train_images = torch.from_numpy(arrays.train_images).unsqueeze(1)
    train_labels = torch.from_numpy(arrays.train_labels)
    test_images = torch.from_numpy(arrays.test_images).unsqueeze(1)
    test_labels = torch.from_numpy(arrays.test_labels)
    training_ids, validation_ids = stratified_source_split(
        train_labels,
        config.calibration.validation_source_examples,
        config.calibration.split_seed,
    )
    if (
        len(training_ids) != config.calibration.training_source_examples
        or len(validation_ids) != config.calibration.validation_source_examples
        or set(training_ids.tolist()) & set(validation_ids.tolist())
    ):
        raise RuntimeError("calibration source split violates its frozen boundary")
    permutations = (
        torch.from_numpy(identity_permutation().copy()),
        *(
            torch.from_numpy(random_digit_permutation(seed).copy())
            for seed in config.benchmark.permutation_seeds
        ),
    )
    identity_training = DenseExamples(train_images[training_ids], train_labels[training_ids], permutations[:1])
    identity_validation = DenseExamples(train_images[validation_ids], train_labels[validation_ids], permutations[:1])
    pooled_training = DenseExamples(train_images[training_ids], train_labels[training_ids], permutations)
    pooled_validation = DenseExamples(train_images[validation_ids], train_labels[validation_ids], permutations)
    rows = []
    for widths in config.calibration.candidate_widths:
        for seed in config.calibration.seeds:
            directory = run_root / "calibration" / f"width-{'-'.join(map(str, widths))}" / f"seed-{seed}"
            identity_result = _fit_or_load(
                directory / "identity",
                identity_training,
                identity_validation,
                initialize_dense_state(
                    widths,
                    config.calibration.dropout,
                    named_seed(seed, "calibration", widths, "identity-init"),
                ),
                config,
                seed,
                "identity",
                device,
            )
            pooled_result = _fit_or_load(
                directory / "pooled",
                pooled_training,
                pooled_validation,
                identity_result.state,
                config,
                seed,
                "pooled",
                device,
            )
            rows.extend(
                _fit_row(widths, seed, fit_name, result)
                for fit_name, result in (("identity", identity_result), ("pooled", pooled_result))
            )
    selected = select_calibrated_width(tuple(rows), config)
    split_record = {
        "training_count": len(training_ids),
        "training_ids_sha256": record_sha256(training_ids.tolist()),
        "validation_count": len(validation_ids),
        "validation_ids_sha256": record_sha256(validation_ids.tolist()),
    }
    if selected is None:
        summary = {
            "config_hash": config.config_hash,
            "fits": rows,
            "schema_version": "vamp-logt-dense-calibration-v1",
            "selected_hidden_widths": None,
            "source_split": split_record,
            "status": "ineligible",
        }
        publish_immutable_json(summary_path, summary)
        return summary
    source_checkpoint = (
        run_root
        / "calibration"
        / f"width-{'-'.join(map(str, selected))}"
        / "seed-0"
        / "identity"
        / "model.pt"
    )
    selected_state = _load_state(source_checkpoint, selected)
    pooled_checkpoint = (
        run_root
        / "calibration"
        / f"width-{'-'.join(map(str, selected))}"
        / "seed-0"
        / "pooled"
        / "model.pt"
    )
    selected_pooled_state = _load_state(pooled_checkpoint, selected)
    base_path = run_root / "base" / "model.pt"
    atomic_torch_save(
        base_path,
        {
            "config_hash": config.config_hash,
            "hidden_widths": selected,
            "parameters": selected_state.tensors,
            "schema_version": "vamp-logt-dense-base-v1",
            "selection_source_sha256": file_sha256(source_checkpoint),
        },
    )
    model = DenseMnistMLP(selected, config.calibration.dropout).to(device)
    load_dense_state(model, selected_state)
    identity_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations[:1]),
        device,
        config.calibration.optimizer.batch_size,
    )
    identity_base_pooled_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations),
        device,
        config.calibration.optimizer.batch_size,
    )
    load_dense_state(model, selected_pooled_state)
    pooled_calibration_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations),
        device,
        config.calibration.optimizer.batch_size,
    )
    summary = {
        "base_checkpoint_sha256": file_sha256(base_path),
        "config_hash": config.config_hash,
        "fits": rows,
        "identity_test_accuracy": identity_test[1],
        "identity_test_cross_entropy": identity_test[0],
        "parameter_count": selected_state.parameter_count,
        "identity_base_pooled_test_accuracy": identity_base_pooled_test[1],
        "identity_base_pooled_test_cross_entropy": identity_base_pooled_test[0],
        "pooled_calibration_test_accuracy": pooled_calibration_test[1],
        "pooled_calibration_test_cross_entropy": pooled_calibration_test[0],
        "schema_version": "vamp-logt-dense-calibration-v1",
        "selected_hidden_widths": list(selected),
        "source_split": split_record,
        "status": "complete",
    }
    publish_immutable_json(
        run_root / "base" / "manifest.json",
        {
            "checkpoint_sha256": file_sha256(base_path),
            "parameter_count": selected_state.parameter_count,
            "schema_version": "vamp-logt-dense-base-manifest-v1",
            "selected_hidden_widths": list(selected),
            "selection_uses_test": False,
        },
    )
    publish_immutable_json(summary_path, summary)
    return summary


def _import_calibration_evidence(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Authenticate the sealed v1 sweep and publish its ungated selected base."""
    source_root = config.calibration_evidence_run
    if source_root is None:  # pragma: no cover - guarded by the caller
        raise ValueError("calibration evidence run is absent")
    source_protocol_path = source_root / "protocol.json"
    source_summary_path = source_root / "calibration" / "summary.json"
    source_protocol = load_canonical_json(source_protocol_path)
    source_summary = load_canonical_json(source_summary_path)
    source_hash = str(source_protocol.get("config_hash", ""))
    source_config = source_protocol.get("config")
    current_protocol = load_canonical_json(run_root / "protocol.json")
    if (
        source_root.name != source_hash
        or source_summary.get("config_hash") != source_hash
        or source_summary.get("status") != "ineligible"
        or source_summary.get("selected_hidden_widths") is not None
        or not isinstance(source_config, dict)
        or source_config.get("protocol_revision") != "dense-full-model-v1"
        or source_protocol.get("source") != current_protocol.get("source")
    ):
        raise ValueError("calibration evidence protocol boundary changed")
    _require_matching_calibration_training(source_config, config)
    rows = tuple(dict(row) for row in source_summary.get("fits", ()))
    expected_coordinates = {
        (widths, seed, fit_name)
        for widths in config.calibration.candidate_widths
        for seed in config.calibration.seeds
        for fit_name in ("identity", "pooled")
    }
    observed_coordinates = {
        (tuple(int(value) for value in row["hidden_widths"]), int(row["seed"]), str(row["fit"]))
        for row in rows
    }
    if observed_coordinates != expected_coordinates or len(rows) != len(expected_coordinates):
        raise ValueError("calibration evidence does not cover the declared sweep")
    for row in rows:
        _authenticate_imported_fit(source_root, source_hash, row)
    selected = select_calibrated_width(rows, config)
    if selected != config.calibration.candidate_widths[0]:
        raise RuntimeError("ungated calibration did not select the smallest candidate")
    identity_checkpoint = (
        source_root
        / "calibration"
        / f"width-{'-'.join(map(str, selected))}"
        / "seed-0"
        / "identity"
        / "model.pt"
    )
    pooled_checkpoint = identity_checkpoint.parent.parent / "pooled" / "model.pt"
    selected_state = _load_state(identity_checkpoint, selected)
    pooled_state = _load_state(pooled_checkpoint, selected)
    base_path = run_root / "base" / "model.pt"
    atomic_torch_save(
        base_path,
        {
            "calibration_evidence_config_hash": source_hash,
            "config_hash": config.config_hash,
            "hidden_widths": selected,
            "parameters": selected_state.tensors,
            "schema_version": "vamp-logt-dense-base-v1",
            "selection_source_sha256": file_sha256(identity_checkpoint),
        },
    )
    arrays = load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)
    train_labels = torch.from_numpy(arrays.train_labels)
    training_ids, validation_ids = stratified_source_split(
        train_labels,
        config.calibration.validation_source_examples,
        config.calibration.split_seed,
    )
    split_record = {
        "training_count": len(training_ids),
        "training_ids_sha256": record_sha256(training_ids.tolist()),
        "validation_count": len(validation_ids),
        "validation_ids_sha256": record_sha256(validation_ids.tolist()),
    }
    if split_record != source_summary.get("source_split"):
        raise ValueError("imported calibration source split changed")
    test_images = torch.from_numpy(arrays.test_images).unsqueeze(1)
    test_labels = torch.from_numpy(arrays.test_labels)
    permutations = (
        torch.from_numpy(identity_permutation().copy()),
        *(
            torch.from_numpy(random_digit_permutation(seed).copy())
            for seed in config.benchmark.permutation_seeds
        ),
    )
    model = DenseMnistMLP(selected, config.calibration.dropout).to(device)
    load_dense_state(model, selected_state)
    identity_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations[:1]),
        device,
        config.calibration.optimizer.batch_size,
    )
    identity_base_pooled_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations),
        device,
        config.calibration.optimizer.batch_size,
    )
    load_dense_state(model, pooled_state)
    pooled_calibration_test = dense_metrics(
        model,
        DenseExamples(test_images, test_labels, permutations),
        device,
        config.calibration.optimizer.batch_size,
    )
    summary = {
        "base_checkpoint_sha256": file_sha256(base_path),
        "calibration_evidence": {
            "fit_count": len(rows),
            "protocol_sha256": file_sha256(source_protocol_path),
            "run_config_hash": source_hash,
            "summary_sha256": file_sha256(source_summary_path),
        },
        "config_hash": config.config_hash,
        "fits": list(rows),
        "identity_test_accuracy": identity_test[1],
        "identity_test_cross_entropy": identity_test[0],
        "parameter_count": selected_state.parameter_count,
        "identity_base_pooled_test_accuracy": identity_base_pooled_test[1],
        "identity_base_pooled_test_cross_entropy": identity_base_pooled_test[0],
        "pooled_calibration_test_accuracy": pooled_calibration_test[1],
        "pooled_calibration_test_cross_entropy": pooled_calibration_test[0],
        "schema_version": "vamp-logt-dense-calibration-v1",
        "selected_hidden_widths": list(selected),
        "selection_policy": config.calibration.selection_policy,
        "source_split": split_record,
        "status": "complete",
    }
    publish_immutable_json(
        run_root / "base" / "manifest.json",
        {
            "calibration_evidence_config_hash": source_hash,
            "checkpoint_sha256": file_sha256(base_path),
            "parameter_count": selected_state.parameter_count,
            "schema_version": "vamp-logt-dense-base-manifest-v1",
            "selected_hidden_widths": list(selected),
            "selection_policy": config.calibration.selection_policy,
            "selection_uses_test": False,
        },
    )
    publish_immutable_json(run_root / "calibration" / "summary.json", summary)
    return summary


def _require_matching_calibration_training(
    source_config: dict[str, object],
    config: VampLogTDenseConfig,
) -> None:
    """Require identical data and fit choices apart from non-operative gates."""
    current = config.as_record()
    if (
        source_config.get("data_root") != current["data_root"]
        or record_sha256(source_config.get("benchmark"))
        != record_sha256(current["benchmark"])
        or not isinstance(source_config.get("calibration"), dict)
    ):
        raise ValueError("imported calibration data definition changed")
    source_calibration = dict(source_config["calibration"])
    current_calibration = dict(current["calibration"])
    for name in (
        "selection_policy",
        "identity_mean_accuracy_minimum",
        "identity_seed_zero_accuracy_minimum",
        "pooled_gap_from_widest_maximum",
    ):
        source_calibration.pop(name, None)
        current_calibration.pop(name, None)
    if record_sha256(source_calibration) != record_sha256(current_calibration):
        raise ValueError("imported calibration training configuration changed")


def _authenticate_imported_fit(
    source_root: Path,
    source_hash: str,
    row: dict[str, object],
) -> None:
    """Verify one imported summary row against its result and checkpoint."""
    widths = tuple(int(value) for value in row["hidden_widths"])
    seed = int(row["seed"])
    fit_name = str(row["fit"])
    directory = (
        source_root
        / "calibration"
        / f"width-{'-'.join(map(str, widths))}"
        / f"seed-{seed}"
        / fit_name
    )
    record = load_canonical_json(directory / "result.json")
    checkpoint_path = directory / "model.pt"
    if (
        record.get("config_hash") != source_hash
        or record.get("fit") != fit_name
        or int(record.get("seed", -1)) != seed
        or file_sha256(checkpoint_path) != record.get("checkpoint_sha256")
        or int(record.get("best_epoch", -1)) != int(row["best_epoch"])
        or int(record.get("epochs_ran", -1)) != int(row["epochs_ran"])
        or record.get("stop_reason") != row["stop_reason"]
    ):
        raise ValueError("imported calibration fit coordinates changed")
    best_epoch = int(record["best_epoch"])
    history = record.get("history")
    if not isinstance(history, list) or not 1 <= best_epoch <= len(history):
        raise ValueError("imported calibration history changed")
    best = history[best_epoch - 1]
    if (
        not isinstance(best, dict)
        or int(best.get("epoch", -1)) != best_epoch
        or float(best.get("validation_accuracy", -1.0)) != float(row["validation_accuracy"])
        or float(best.get("validation_loss", -1.0)) != float(row["validation_cross_entropy"])
    ):
        raise ValueError("imported calibration best epoch changed")
    state = _load_state(checkpoint_path, widths)
    if state.parameter_count != int(row["parameter_count"]):
        raise ValueError("imported calibration checkpoint size changed")


def load_calibrated_base(config: VampLogTDenseConfig, run_root: Path) -> DenseMlpState:
    """Authenticate and return the selected seed-zero identity state."""
    summary = load_canonical_json(run_root / "calibration" / "summary.json")
    if summary.get("status") != "complete" or summary.get("config_hash") != config.config_hash:
        raise RuntimeError("dense calibration is absent, ineligible, or belongs to another protocol")
    widths = tuple(int(value) for value in summary["selected_hidden_widths"])
    path = run_root / "base" / "model.pt"
    if file_sha256(path) != summary["base_checkpoint_sha256"]:
        raise ValueError("selected dense base checkpoint changed")
    return _load_state(path, widths)


def _fit_or_load(
    directory: Path,
    training: DenseExamples,
    validation: DenseExamples,
    initial_state: DenseMlpState,
    config: VampLogTDenseConfig,
    seed: int,
    fit_name: str,
    device: torch.device,
) -> DenseFitResult:
    checkpoint_path = directory / "model.pt"
    record_path = directory / "result.json"
    if checkpoint_path.is_file() and record_path.is_file():
        record = load_canonical_json(record_path)
        if (
            record.get("config_hash") != config.config_hash
            or record.get("fit") != fit_name
            or int(record.get("seed", -1)) != seed
            or file_sha256(checkpoint_path) != record.get("checkpoint_sha256")
        ):
            raise ValueError("stored dense calibration coordinates changed")
        state = _load_state(checkpoint_path, initial_state.hidden_widths)
        history = tuple(
            _epoch_from_record(row) for row in record["history"]
        )
        return DenseFitResult(
            state,
            int(record["best_epoch"]),
            int(record["epochs_ran"]),
            int(record["optimizer_steps"]),
            int(record["training_example_presentations"]),
            int(record["validation_example_presentations"]),
            str(record["stop_reason"]),
            history,
        )
    result = fit_dense_model(
        training,
        initial_state,
        config.calibration.optimizer,
        named_seed(seed, "calibration", initial_state.hidden_widths, fit_name, "fit"),
        device,
        validation=validation,
        convergence=config.calibration.convergence,
        dropout=config.calibration.dropout,
        progress_label=f"calibration {initial_state.hidden_widths} seed {seed} {fit_name}",
        progress=config.runtime.progress,
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "config_hash": config.config_hash,
            "fit": fit_name,
            "hidden_widths": initial_state.hidden_widths,
            "parameters": result.state.tensors,
            "schema_version": "vamp-logt-dense-calibration-checkpoint-v1",
            "seed": seed,
        },
    )
    record = {
        "best_epoch": result.best_epoch,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_hash": config.config_hash,
        "epochs_ran": result.epochs_ran,
        "fit": fit_name,
        "history": [asdict(row) for row in result.history],
        "optimizer_steps": result.optimizer_steps,
        "schema_version": "vamp-logt-dense-calibration-fit-v1",
        "seed": seed,
        "stop_reason": result.stop_reason,
        "training_example_presentations": result.training_example_presentations,
        "validation_example_presentations": result.validation_example_presentations,
    }
    publish_immutable_json(record_path, record)
    return result


def _fit_row(
    widths: tuple[int, int, int],
    seed: int,
    fit_name: str,
    result: DenseFitResult,
) -> dict[str, object]:
    best = result.history[result.best_epoch - 1]
    if best.validation_loss is None or best.validation_accuracy is None:
        raise RuntimeError("calibration fit lacks validation evidence")
    return {
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "fit": fit_name,
        "hidden_widths": list(widths),
        "parameter_count": result.state.parameter_count,
        "seed": seed,
        "stop_reason": result.stop_reason,
        "validation_accuracy": best.validation_accuracy,
        "validation_cross_entropy": best.validation_loss,
    }


def _load_state(path: Path, widths: tuple[int, ...]) -> DenseMlpState:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(payload.get("hidden_widths", ())) != widths or "parameters" not in payload:
        raise ValueError("dense checkpoint architecture changed")
    return DenseMlpState(tuple(payload["parameters"]))


def _epoch_from_record(row: object):
    from apm.continual.dense_mlp_adapter import DenseEpochResult

    if not isinstance(row, dict):
        raise ValueError("calibration epoch record is malformed")
    return DenseEpochResult(
        int(row["epoch"]),
        float(row["learning_rate"]),
        float(row["training_loss"]),
        float(row["training_accuracy"]),
        None if row["validation_loss"] is None else float(row["validation_loss"]),
        None if row["validation_accuracy"] is None else float(row["validation_accuracy"]),
    )


__all__ = [
    "initialize_dense_state",
    "load_calibrated_base",
    "run_calibration",
    "select_calibrated_width",
]
