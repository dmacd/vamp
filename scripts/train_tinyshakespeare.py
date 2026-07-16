"""Train and checkpoint the single canonical TinyShakespeare base-model workflow."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import jax

from apm.lm.checkpoint import (
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    save_gpt_neo_checkpoint,
)
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import (
    TINY_SHAKESPEARE_SOURCE,
    TINY_SHAKESPEARE_STANDARD_PRESET,
    build_tiny_shakespeare_data,
    load_tiny_shakespeare,
    prepare_tiny_shakespeare,
)
from apm.lm.training import (
    BASE_TRAINING_PRESET,
    init_base_train_state,
)
from apm.lm.workflow import (
    evaluate_normalized_nll,
    run_base_updates,
    tiny_shakespeare_model_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "data" / "tinyshakespeare" / "input.txt"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "tinyshakespeare-base"
SEED = 0


def main() -> None:
    """Prepare pinned data, run the frozen standard training budget, and save schema v1."""
    prepared_path = prepare_tiny_shakespeare(CORPUS_PATH)
    corpus_text = load_tiny_shakespeare(prepared_path)
    data = build_tiny_shakespeare_data(
        corpus_text,
        TINY_SHAKESPEARE_STANDARD_PRESET,
    )
    model_config = tiny_shakespeare_model_config(data.tokenizer.vocab_size)
    parameter_key, training_key = jax.random.split(jax.random.PRNGKey(SEED))
    initial_params = init_gpt_neo_params(parameter_key, model_config)
    initial_state = init_base_train_state(
        initial_params,
        training_key,
        BASE_TRAINING_PRESET,
    )
    initial_validation_nll = evaluate_normalized_nll(
        initial_params,
        model_config,
        data.validation_batches,
    )
    trained_state, trace = run_base_updates(
        initial_state,
        data.train_batches,
        model_config,
        BASE_TRAINING_PRESET,
    )
    final_validation_nll = evaluate_normalized_nll(
        trained_state.trainable,
        model_config,
        data.validation_batches,
    )
    vocabulary_json = (
        json.dumps(
            {
                "eos_token_id": data.tokenizer.eos_token_id,
                "pad_token_id": data.tokenizer.pad_token_id,
                "vocabulary": list(data.tokenizer.vocabulary),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    checkpoint_ref = save_gpt_neo_checkpoint(
        CHECKPOINT_PATH,
        trained_state.trainable,
        model_config,
        tokenizer=TokenizerCheckpointMetadata(
            kind="character",
            identifier="tinyshakespeare-char-v1",
            revision=TINY_SHAKESPEARE_SOURCE.revision,
            files=(
                CheckpointFileHash(
                    name="vocabulary.json",
                    sha256=sha256(vocabulary_json).hexdigest(),
                ),
            ),
        ),
        source=SourceCheckpointMetadata(
            identifier=TINY_SHAKESPEARE_SOURCE.repository,
            revision=TINY_SHAKESPEARE_SOURCE.revision,
            sha256=TINY_SHAKESPEARE_SOURCE.expected_sha256,
        ),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_ref.directory),
                "final_validation_nll": final_validation_nll,
                "initial_validation_nll": initial_validation_nll,
                "seed": SEED,
                "steps": len(trace.step_losses),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
