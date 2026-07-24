#!/usr/bin/env python3
"""Prepare pinned MiniLM evidence and the immutable semantic-v1 catalog."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_p_semantic import (
        EncoderIdentity,
        SemanticCatalog,
        SemanticEvidenceArtifact,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import SemanticProgressReporter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = (
    REPOSITORY_ROOT / "data" / "tinyworlds-v2" / "source" / "TinyStories_all_data.tar.gz"
)
TOKENIZER_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
ENCODER_ROOT = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "tinyworlds-p-semantic-v1"
    / "encoder"
    / "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
)
_ENCODER_SNAPSHOT_FILES = {
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
}
_PREFLIGHT_TEXT_COUNT = 1_024
_CONSERVATIVE_EMBEDDING_COUNT = 100_000

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _download_encoder_snapshot() -> None:
    """Download exactly the pinned PyTorch model/tokenizer snapshot files atomically."""
    from huggingface_hub import HfApi, hf_hub_download

    from apm.data.text.tinyworlds_p_semantic import ENCODER_IDENTIFIER, ENCODER_REVISION

    if ENCODER_ROOT.exists():
        return
    work_root = ENCODER_ROOT.parents[1] / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="minilm-snapshot-", dir=work_root))
    print(f"[encoder] snapshot staging directory: {staging}", flush=True)
    available = set(
        HfApi().list_repo_files(
            ENCODER_IDENTIFIER,
            revision=ENCODER_REVISION,
            repo_type="model",
        )
    )
    selected = tuple(sorted(available & _ENCODER_SNAPSHOT_FILES))
    mandatory = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    weights = {"model.safetensors", "pytorch_model.bin"}
    if not mandatory <= set(selected) or not weights & set(selected):
        raise RuntimeError("pinned MiniLM snapshot lacks required model/tokenizer files")
    cache = work_root / "huggingface-cache"
    for completed, filename in enumerate(selected, start=1):
        source = Path(
            hf_hub_download(
                ENCODER_IDENTIFIER,
                filename=filename,
                revision=ENCODER_REVISION,
                repo_type="model",
                cache_dir=cache,
            )
        )
        destination = staging / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        print(
            f"[encoder] {completed}/{len(selected)} pinned files: {filename}",
            flush=True,
        )
    ENCODER_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, ENCODER_ROOT)


def _encoder_preflight() -> tuple[EncoderIdentity, float]:
    """Authenticate MiniLM and measure float32 CUDA inference throughput."""
    import numpy as np

    from apm.data.text.tinyworlds_p_semantic import (
        ENCODER_SNAPSHOT_IDENTITY_SHA256,
        MiniLMEncoder,
        discover_encoder_identity,
    )

    identity = discover_encoder_identity(ENCODER_ROOT)
    if identity.identity_sha256 != ENCODER_SNAPSHOT_IDENTITY_SHA256:
        raise RuntimeError(
            "local MiniLM files differ from the frozen semantic-v1 snapshot identity"
        )
    encoder = MiniLMEncoder(ENCODER_ROOT, identity, device="cuda")
    warmup = tuple(f"The word object{index} is a noun." for index in range(256))
    encoder.embed(warmup, 256)
    texts = tuple(
        f"Someone can action{index}, and this exact sentence is encoder evidence."
        for index in range(_PREFLIGHT_TEXT_COUNT)
    )
    started = time.monotonic()
    embeddings = encoder.embed(texts, 256)
    elapsed = time.monotonic() - started
    replay = encoder.embed(texts, 256)
    if embeddings.shape != (_PREFLIGHT_TEXT_COUNT, identity.dimension) or not np.all(
        np.isfinite(embeddings)
    ):
        raise RuntimeError("MiniLM CUDA preflight returned malformed embeddings")
    if not np.array_equal(embeddings, replay):
        raise RuntimeError("MiniLM CUDA preflight is not bit-deterministic")
    throughput = len(texts) / elapsed
    embedding_eta = _CONSERVATIVE_EMBEDDING_COUNT / throughput
    print(
        f"[preflight] pinned MiniLM {identity.identity_sha256}; "
        f"{throughput:,.1f} texts/s measured in {elapsed:.2f}s; deterministic replay passed; "
        f"conservative {_CONSERVATIVE_EMBEDDING_COUNT:,}-text phase ETA "
        f"{_duration(embedding_eta)}; preliminary overall ETA "
        f"{_duration(45 * 60 + embedding_eta)}",
        flush=True,
    )
    return identity, throughput


def _existing_evidence(
    encoder_identity: EncoderIdentity,
) -> SemanticEvidenceArtifact | None:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        SEMANTIC_CONFIG,
        load_semantic_evidence,
    )

    root = SEMANTIC_DATA_ROOT / "evidence" / "v1"
    candidates = tuple(
        load_semantic_evidence(path)
        for path in sorted(root.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if root.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.archive_identity == CANONICAL_ARCHIVE_IDENTITY
        and item.encoder_identity == encoder_identity
        and item.config.evidence_record() == SEMANTIC_CONFIG.evidence_record()
    )
    if len(matches) > 1:
        raise RuntimeError("multiple semantic evidence artifacts have the same fixed identity")
    return matches[0] if matches else None


def _existing_catalog(
    evidence: SemanticEvidenceArtifact,
) -> SemanticCatalog | None:
    from apm.data.text.tinyworlds_p_semantic import SEMANTIC_CONFIG, load_semantic_catalog

    root = SEMANTIC_DATA_ROOT / "catalog" / "v1"
    candidates = tuple(
        load_semantic_catalog(path)
        for path in sorted(root.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if root.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.evidence_sha256 == evidence.evidence_sha256
        and item.encoder_identity == evidence.encoder_identity
        and item.config == SEMANTIC_CONFIG
    )
    if len(matches) > 1:
        raise RuntimeError("multiple semantic catalogs have the same fixed identity")
    return matches[0] if matches else None


def _build_evidence(
    encoder_identity: EncoderIdentity,
    work_directory: Path,
    reporter: SemanticProgressReporter,
) -> SemanticEvidenceArtifact:
    from apm.data.text.tinyworlds_p import (
        NORMALIZATION_IDENTITY,
        PARTITION_PRESET,
        PartitionInputs,
    )
    from apm.data.text.tinyworlds_p.archive_ingest import build_archive_ingest
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        SEMANTIC_CONFIG,
        prepare_semantic_evidence,
    )

    ingest_directory = work_directory / "archive-ingest"
    ingest = build_archive_ingest(
        PartitionInputs(
            archive_path=ARCHIVE_PATH,
            tokenizer_directory=TOKENIZER_DIRECTORY,
            output_root=SEMANTIC_DATA_ROOT / "evidence" / "v1",
            temporary_directory=ingest_directory,
            archive_identity=CANONICAL_ARCHIVE_IDENTITY,
            tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
            progress=reporter.archive_event,
        ),
        replace(PARTITION_PRESET, worker_count=24),
        NORMALIZATION_IDENTITY,
    )
    return prepare_semantic_evidence(
        ingest,
        CANONICAL_ARCHIVE_IDENTITY,
        ENCODER_ROOT,
        encoder_identity,
        SEMANTIC_DATA_ROOT / "evidence" / "v1",
        work_directory / "evidence-publication",
        SEMANTIC_CONFIG,
        batch_size=256,
        device="cuda",
        progress=reporter,
    )


def _build_catalog(
    evidence: SemanticEvidenceArtifact,
    working: Path,
    reporter: SemanticProgressReporter,
) -> SemanticCatalog | None:
    """Publish the frozen catalog, or report its fail-closed audit without a traceback."""
    from apm.data.text.tinyworlds_p_semantic import (
        SEMANTIC_CONFIG,
        SemanticGridError,
        build_catalog_from_evidence,
    )

    reporter("catalog", 0, 1, "applying frozen semantic gates and clustering")
    try:
        catalog = build_catalog_from_evidence(
            evidence,
            SEMANTIC_DATA_ROOT / "catalog" / "v1",
            working / "catalog-publication",
            SEMANTIC_CONFIG,
        )
    except SemanticGridError as error:
        reporter("catalog", 1, 1, "semantic grid stopped; failure audit published")
        print(f"[catalog] automated semantic-v1 stop: {error}", flush=True)
        return None
    reporter("catalog", 1, 1, "semantic catalog and self-contained audits published")
    return catalog


def _duration(seconds: float) -> str:
    rounded = max(0, math.ceil(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"


def main() -> int:
    """Build or strictly reuse semantic evidence, then publish its fixed catalog."""
    from apm.data.text.tinyworlds_p_semantic.progress import (
        CONSTRUCTION_PHASE_WEIGHTS,
        SemanticProgressReporter,
    )

    _download_encoder_snapshot()
    encoder_identity, _ = _encoder_preflight()
    work_root = SEMANTIC_DATA_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-construction-v1-", dir=work_root))
    print(f"temporary artifact directory: {working}", flush=True)
    evidence = _existing_evidence(encoder_identity)
    if evidence is None:
        reporter = SemanticProgressReporter(
            "TinyWorlds-P semantic construction",
            CONSTRUCTION_PHASE_WEIGHTS,
        )
        try:
            evidence = _build_evidence(encoder_identity, working, reporter)
            catalog = _build_catalog(evidence, working, reporter)
        finally:
            reporter.close()
    else:
        print(f"[evidence] strict cached evidence reused: {evidence.root}", flush=True)
        catalog = _existing_catalog(evidence)
        if catalog is None:
            reporter = SemanticProgressReporter(
                "TinyWorlds-P semantic catalog",
                {"catalog": 1.0},
            )
            try:
                catalog = _build_catalog(evidence, working, reporter)
            finally:
                reporter.close()
        else:
            print(f"[catalog] strict existing catalog reused: {catalog.root}", flush=True)
    print(f"evidence: {evidence.root}")
    print(f"evidence SHA-256: {evidence.evidence_sha256}")
    if catalog is None:
        print("catalog: not created (frozen semantic grid failed)")
        print("partition/training/sealed test: not authorized")
        return 2
    print(f"catalog: {catalog.root}")
    print(f"catalog SHA-256: {catalog.catalog_sha256}")
    print(f"retained archive token mass: {catalog.retained_token_fraction:.3%}")
    print(f"Markdown audit: {catalog.root / 'audit.md'}")
    print(f"HTML audit: {catalog.root / 'audit.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
