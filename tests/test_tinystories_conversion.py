from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.lm.tinystories_conversion as conversion_module
from apm.lm.tinystories_conversion import (
    ConvertedTinyStoriesModel,
    PinnedArtifactFile,
    TINYSTORIES_SOURCE,
    TinyStoriesSourceContract,
    convert_gpt_neo_state_dict,
    load_tinystories_artifact,
    save_tinystories_artifact,
    tinystories_conversion_provenance,
    validate_pinned_artifact_file,
)


def _config_payload() -> dict[str, object]:
    return {
        "model_type": "gpt_neo",
        "transformers_version": "4.28.1",
        "activation_function": "gelu_new",
        "vocab_size": 7,
        "max_position_embeddings": 4,
        "hidden_size": 4,
        "intermediate_size": None,
        "num_layers": 2,
        "num_heads": 2,
        "attention_types": [[("global", "local"), 1]],
        "attention_layers": ["global", "local"],
        "window_size": 2,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "embed_dropout": 0,
        "attention_dropout": 0,
        "resid_dropout": 0,
    }


def _values(shape: tuple[int, ...], offset: int) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    return (np.arange(size, dtype=np.float32) + offset).reshape(shape)


def _state_dict() -> dict[str, np.ndarray]:
    config = _config_payload()
    hidden = int(config["hidden_size"])
    intermediate = 4 * hidden
    max_positions = int(config["max_position_embeddings"])
    state = {
        "transformer.wte.weight": _values((7, hidden), 10),
        "transformer.wpe.weight": _values((max_positions, hidden), 20),
        "transformer.ln_f.weight": _values((hidden,), 30),
        "transformer.ln_f.bias": _values((hidden,), 40),
    }
    causal = np.tril(np.ones((max_positions, max_positions), dtype=np.bool_))
    masks = (
        causal,
        np.bitwise_xor(causal, np.tril(causal, -int(config["window_size"]))),
    )
    for layer_index in range(2):
        prefix = f"transformer.h.{layer_index}"
        offset = 100 * (layer_index + 1)
        state.update(
            {
                f"{prefix}.ln_1.weight": _values((hidden,), offset + 1),
                f"{prefix}.ln_1.bias": _values((hidden,), offset + 2),
                f"{prefix}.attn.attention.q_proj.weight": _values((hidden, hidden), offset + 3),
                f"{prefix}.attn.attention.k_proj.weight": _values((hidden, hidden), offset + 4),
                f"{prefix}.attn.attention.v_proj.weight": _values((hidden, hidden), offset + 5),
                f"{prefix}.attn.attention.out_proj.weight": _values((hidden, hidden), offset + 6),
                f"{prefix}.attn.attention.out_proj.bias": _values((hidden,), offset + 7),
                f"{prefix}.ln_2.weight": _values((hidden,), offset + 8),
                f"{prefix}.ln_2.bias": _values((hidden,), offset + 9),
                f"{prefix}.mlp.c_fc.weight": _values((intermediate, hidden), offset + 10),
                f"{prefix}.mlp.c_fc.bias": _values((intermediate,), offset + 11),
                f"{prefix}.mlp.c_proj.weight": _values((hidden, intermediate), offset + 12),
                f"{prefix}.mlp.c_proj.bias": _values((hidden,), offset + 13),
                f"{prefix}.attn.attention.bias": masks[layer_index][None, None],
                f"{prefix}.attn.attention.masked_bias": np.asarray(-1e9, dtype=np.float32),
            }
        )
    state["lm_head.weight"] = state["transformer.wte.weight"].copy()
    return state


def test_conversion_expands_attention_normalizes_mlp_and_transposes_all_linears() -> None:
    state = {name: jnp.asarray(value) for name, value in _state_dict().items()}
    converted = convert_gpt_neo_state_dict(state, _config_payload())
    first_block = converted.params.blocks[0]

    assert converted.config.intermediate_size == 16
    assert converted.config.attention_types == ("global", "local")
    np.testing.assert_array_equal(
        converted.params.token_embedding,
        _state_dict()["transformer.wte.weight"],
    )
    np.testing.assert_array_equal(
        first_block.attention.query.kernel,
        _state_dict()["transformer.h.0.attn.attention.q_proj.weight"].T,
    )
    np.testing.assert_array_equal(
        first_block.attention.output.kernel,
        _state_dict()["transformer.h.0.attn.attention.out_proj.weight"].T,
    )
    np.testing.assert_array_equal(
        first_block.mlp.input_projection.kernel,
        _state_dict()["transformer.h.0.mlp.c_fc.weight"].T,
    )
    np.testing.assert_array_equal(
        first_block.mlp.output_projection.kernel,
        _state_dict()["transformer.h.0.mlp.c_proj.weight"].T,
    )


@pytest.mark.parametrize("defect", ("missing", "unexpected"))
def test_conversion_rejects_every_source_key_mismatch(defect: str) -> None:
    state = _state_dict()
    if defect == "missing":
        state.pop("transformer.h.1.mlp.c_proj.bias")
    else:
        state["transformer.unexpected.weight"] = np.zeros((1,), dtype=np.float32)

    with pytest.raises(ValueError, match=f"{defect}="):
        convert_gpt_neo_state_dict(state, _config_payload())


def test_conversion_requires_exact_tied_lm_head() -> None:
    state = _state_dict()
    state["lm_head.weight"][0, 0] += np.float32(1)

    with pytest.raises(ValueError, match="exactly tied"):
        convert_gpt_neo_state_dict(state, _config_payload())


@pytest.mark.parametrize("defect", ("shape", "dtype", "mask", "masked_bias"))
def test_conversion_validates_shapes_dtypes_and_transformers_buffers(defect: str) -> None:
    state = _state_dict()
    if defect == "shape":
        state["transformer.h.0.mlp.c_fc.weight"] = np.zeros((16, 3), dtype=np.float32)
    elif defect == "dtype":
        state["transformer.h.0.ln_1.weight"] = np.zeros((4,), dtype=np.float16)
    elif defect == "mask":
        state["transformer.h.1.attn.attention.bias"][0, 0, 0, 3] = True
    else:
        state["transformer.h.1.attn.attention.masked_bias"] = np.asarray(-1e4, dtype=np.float32)

    with pytest.raises((TypeError, ValueError), match=defect if defect != "mask" else "local mask"):
        convert_gpt_neo_state_dict(state, _config_payload())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("attention_layers", ["local", "global"], "attention_layers"),
        ("attention_types", [[["global", "invalid"], 1]], "global and local"),
        ("attention_types", [[["global", "local"], 2]], "one entry per layer"),
        ("transformers_version", "4.29.0", "exactly 4.28.1"),
    ),
)
def test_config_translation_rejects_semantic_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _config_payload()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        convert_gpt_neo_state_dict(_state_dict(), config)


def test_pinned_file_validation_checks_filename_size_and_hash(tmp_path: Path) -> None:
    contents = b"strict pinned file"
    path = tmp_path / "source.bin"
    path.write_bytes(contents)
    expected = PinnedArtifactFile(
        path.name,
        hashlib.sha256(contents).hexdigest(),
        len(contents),
    )

    validate_pinned_artifact_file(path, expected)
    path.write_bytes(contents + b"!")
    with pytest.raises(ValueError, match="size"):
        validate_pinned_artifact_file(path, expected)


def test_atomic_artifact_round_trip_copies_and_revalidates_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_contents = {"merges.txt": b"merges", "vocab.json": b"vocabulary"}
    source = TinyStoriesSourceContract(
        model_id="synthetic/tinystories",
        revision="1" * 40,
        transformers_version="4.28.1",
        config_file=_pinned_file("config.json", b"config"),
        model_file=_pinned_file("pytorch_model.bin", b"model"),
        tokenizer_files=tuple(
            _pinned_file(name, contents)
            for name, contents in tokenizer_contents.items()
        ),
    )
    converted_model = convert_gpt_neo_state_dict(_state_dict(), _config_payload())
    converted = ConvertedTinyStoriesModel(converted_model, source)
    monkeypatch.setattr(conversion_module, "TINYSTORIES_SOURCE", source)
    monkeypatch.setattr(
        conversion_module,
        "_pinned_gpt_neo_config",
        lambda: converted_model.config,
    )
    provenance = tinystories_conversion_provenance(
        library_versions=(("torch", "2.2"), ("transformers", "4.28.1")),
        environment=(("platform", "test"), ("python", "3.11")),
    )

    artifact = save_tinystories_artifact(
        tmp_path / "artifact",
        converted,
        tokenizer_files=tokenizer_contents,
        provenance=provenance,
    )
    reloaded = load_tinystories_artifact(artifact.directory)

    assert reloaded.checkpoint.provenance == provenance
    assert {path.name: path.read_bytes() for path in reloaded.tokenizer_files} == tokenizer_contents
    manifest = json.loads((artifact.directory / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == source.revision
    assert set(path.name for path in artifact.directory.iterdir()) == {
        "artifact.json",
        "checkpoint",
        "tokenizer",
    }

    reloaded.tokenizer_files[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size|SHA-256"):
        load_tinystories_artifact(artifact.directory)


def test_normal_imports_do_not_load_torch_or_transformers() -> None:
    code = """
import importlib.util
import runpy
import sys
import apm.lm.tinystories_conversion
runpy.run_path('scripts/convert_tinystories_8m.py', run_name='converter_import_test')
print(sorted({'torch', 'transformers'} & set(sys.modules)))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


@pytest.mark.integration
def test_prepared_snapshot_matches_contract_and_converts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_directory = os.environ.get("TINYSTORIES_HF_MODEL_DIR")
    if prepared_directory is None:
        pytest.skip("TINYSTORIES_HF_MODEL_DIR does not name a prepared local snapshot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def deny_network(*arguments: object, **keywords: object) -> None:
        del arguments, keywords
        raise AssertionError("integration conversion attempted network access")

    monkeypatch.setattr(socket, "socket", deny_network)
    import numpy as np
    import transformers
    from transformers import GPTNeoForCausalLM

    assert transformers.__version__ == TINYSTORIES_SOURCE.transformers_version
    expected_files = (
        TINYSTORIES_SOURCE.config_file,
        TINYSTORIES_SOURCE.model_file,
        *TINYSTORIES_SOURCE.tokenizer_files,
    )
    snapshot = Path(prepared_directory)
    for artifact in expected_files:
        validate_pinned_artifact_file(snapshot / artifact.name, artifact)
    model = GPTNeoForCausalLM.from_pretrained(snapshot, local_files_only=True)
    state = {
        name: np.asarray(tensor.detach().cpu().numpy())
        for name, tensor in model.state_dict().items()
    }
    converted = conversion_module.convert_tinystories_state_dict(
        state,
        (snapshot / TINYSTORIES_SOURCE.config_file.name).read_bytes(),
    )

    assert converted.model.config.hidden_size == 256
    assert converted.model.config.intermediate_size == 1_024


def _pinned_file(name: str, contents: bytes) -> PinnedArtifactFile:
    return PinnedArtifactFile(name, hashlib.sha256(contents).hexdigest(), len(contents))
