"""Construction-slice context selection and pinned MiniLM evidence caching."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import heapq
import json
import os
from pathlib import Path
import re
import unicodedata

import numpy as np

from apm.data.text.tinyworlds_p.archive_ingest import (
    ArchiveIngestResult,
    iter_archive_groups,
)
from apm.data.text.tinyworlds_p.contracts import SourceIdentity
from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    ENCODER_DIMENSION,
    ENCODER_IDENTIFIER,
    ENCODER_REVISION,
    EVIDENCE_FORMAT,
    SCHEMA_VERSION,
    EncoderIdentity,
    ModelFile,
    Role,
    SemanticConstructionConfig,
    SemanticContext,
    SemanticEvidenceArtifact,
    canonical_json_bytes,
    record_sha256,
)


EvidenceProgress = Callable[[str, int, int, str], None]
_SENTENCE = re.compile(r".*?(?:[.!?]+(?=\s|$)|\n+|$)", flags=re.DOTALL)


class SemanticEvidenceError(ValueError):
    """Pinned encoder evidence is malformed or fails authentication."""


@dataclass(frozen=True, slots=True)
class SelectedContext:
    """One selected exact context and its target-centered encoder text."""

    context: SemanticContext
    encoder_text: str

    def __post_init__(self) -> None:
        if not self.encoder_text:
            raise ValueError("selected context encoder text must be nonempty")

    def as_record(self) -> dict[str, object]:
        """Return the persisted exact context and encoder-input text."""
        return {**self.context.as_record(), "encoder_text": self.encoder_text}


def namespaced_sha256(namespace: str, value: str) -> str:
    """Hash one value in the immutable semantic-v1 namespace."""
    if not namespace or not value:
        raise ValueError("semantic hash namespace and value must be nonempty")
    return sha256(f"{BENCHMARK_ID}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def is_construction_group(
    normalized_story_sha256: str,
    config: SemanticConstructionConfig,
) -> bool:
    """Return whether a duplicate group belongs to the permanent 5% evidence slice."""
    digest = namespaced_sha256(
        "semantic-construction-slice",
        normalized_story_sha256,
    )
    return int(digest, 16) % config.construction_modulus == config.construction_residue


def exact_whole_word_spans(text: str, word: str) -> tuple[tuple[int, int], ...]:
    """Find exact normalized, case-insensitive whole-word occurrences."""
    if type(text) is not str or type(word) is not str or not word:
        raise ValueError("whole-word matching requires text and a nonempty word")
    normalized_word = unicodedata.normalize("NFKC", word).casefold().strip()
    normalized_text = unicodedata.normalize("NFKC", text)
    pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_word)}(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )
    return tuple(match.span() for match in pattern.finditer(normalized_text))


def story_contexts(
    role: Role,
    word: str,
    normalized_story_sha256: str,
    record_id: str,
    story_sha256: str,
    story: str,
) -> tuple[SemanticContext, ...]:
    """Extract every sentence containing an exact normalized target occurrence."""
    normalized_story = unicodedata.normalize("NFKC", story)
    sentence_records: list[SemanticContext] = []
    for sentence_index, match in enumerate(_SENTENCE.finditer(normalized_story)):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        for occurrence_index, (start, stop) in enumerate(exact_whole_word_spans(sentence, word)):
            selection = namespaced_sha256(
                f"context:{role}:{word}",
                f"{normalized_story_sha256}\0{sentence_index}\0{occurrence_index}\0{start}\0{stop}",
            )
            sentence_records.append(
                SemanticContext(
                    role=role,
                    word=word,
                    normalized_story_sha256=normalized_story_sha256,
                    record_id=record_id,
                    story_sha256=story_sha256,
                    sentence=sentence,
                    target_start=start,
                    target_stop=stop,
                    selection_sha256=selection,
                )
            )
    return tuple(sentence_records)


def select_lowest_hash_contexts(
    contexts: Sequence[SemanticContext],
    maximum_count: int,
) -> tuple[SemanticContext, ...]:
    """Select at most the requested number of canonical lowest-hash contexts."""
    if type(maximum_count) is not int or maximum_count <= 0:
        raise ValueError("maximum context count must be positive")
    identities = [item.selection_sha256 for item in contexts]
    if len(set(identities)) != len(identities):
        raise ValueError("semantic contexts must have unique selection identities")
    return tuple(
        sorted(contexts, key=lambda item: (item.selection_sha256, item.record_id))[
            :maximum_count
        ]
    )


def target_centered_encoder_text(
    context: SemanticContext,
    tokenizer: object,
    wordpiece_limit: int,
) -> str:
    """Crop a long sentence to a target-centered window of at most 128 wordpieces."""
    if type(wordpiece_limit) is not int or wordpiece_limit <= 2:
        raise ValueError("wordpiece limit must leave space for special tokens")
    encoded = tokenizer(
        context.sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = wordpiece_limit - special_count
    if budget <= 0:
        raise ValueError("wordpiece limit is smaller than encoder special tokens")
    if len(input_ids) <= budget:
        return context.sentence
    target_indexes = tuple(
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > context.target_start and start < context.target_stop
    )
    if not target_indexes:
        raise SemanticEvidenceError("encoder tokenizer did not preserve the target span")
    target_center = (target_indexes[0] + target_indexes[-1]) // 2
    window_start = min(max(0, target_center - budget // 2), len(input_ids) - budget)
    window_stop = window_start + budget
    if not window_start <= target_indexes[0] and target_indexes[-1] < window_stop:
        window_start = min(target_indexes[0], len(input_ids) - budget)
        window_stop = window_start + budget
    character_start = offsets[window_start][0]
    character_stop = offsets[window_stop - 1][1]
    cropped = context.sentence[character_start:character_stop].strip()
    if not cropped or not exact_whole_word_spans(cropped, context.word):
        raise SemanticEvidenceError("target-centered crop lost the exact target word")
    checked = tokenizer(cropped, add_special_tokens=True, truncation=False)["input_ids"]
    if len(checked) > wordpiece_limit:
        raise SemanticEvidenceError("target-centered crop exceeds its wordpiece limit")
    return cropped


def discover_encoder_identity(directory: str | Path) -> EncoderIdentity:
    """Hash every regular file exposed by one pinned local MiniLM snapshot."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(root)
    files = tuple(
        ModelFile(
            relative_path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and ".locks" not in path.parts and path.stat().st_size > 0
    )
    return EncoderIdentity(
        identifier=ENCODER_IDENTIFIER,
        revision=ENCODER_REVISION,
        dimension=ENCODER_DIMENSION,
        files=files,
    )


def verify_encoder_identity(directory: str | Path, expected: EncoderIdentity) -> None:
    """Reject any missing, extra, resized, or rehashed encoder snapshot file."""
    measured = discover_encoder_identity(directory)
    if measured != expected:
        raise SemanticEvidenceError("local MiniLM snapshot identity changed")


class MiniLMEncoder:
    """Pinned float32 attention-mask-mean, L2-normalized MiniLM inference."""

    def __init__(
        self,
        directory: str | Path,
        identity: EncoderIdentity,
        *,
        device: str = "cuda",
    ) -> None:
        verify_encoder_identity(directory, identity)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "semantic evidence requires the lm dependencies plus CUDA PyTorch"
            ) from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("semantic-v1 MiniLM preparation requires CUDA PyTorch")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(directory),
            local_files_only=True,
            revision=identity.revision,
        )
        self._model = AutoModel.from_pretrained(
            str(directory),
            local_files_only=True,
            revision=identity.revision,
        ).to(device=device, dtype=torch.float32)
        self._model.eval()
        self._device = device
        self.identity = identity

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed texts in float32 with explicit mean pooling and L2 normalization."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("encoder batch size must be positive")
        if not texts or any(type(text) is not str or not text for text in texts):
            raise ValueError("encoder input texts must be nonempty")
        outputs: list[np.ndarray] = []
        torch = self._torch
        for start in range(0, len(texts), batch_size):
            batch = tuple(texts[start : start + batch_size])
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            tokenized = {
                key: value.to(self._device)
                for key, value in tokenized.items()
            }
            if tokenized["input_ids"].shape[1] > 128:
                raise SemanticEvidenceError("MiniLM evidence input exceeds 128 wordpieces")
            with torch.inference_mode():
                hidden = self._model(**tokenized).last_hidden_state.float()
                mask = tokenized["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(normalized.cpu().numpy().astype("<f4", copy=False))
            if progress is not None:
                progress(min(start + batch_size, len(texts)), len(texts))
        result = np.concatenate(outputs, axis=0)
        if result.shape != (len(texts), self.identity.dimension):
            raise SemanticEvidenceError("MiniLM returned an unexpected embedding shape")
        return result


def prepare_semantic_evidence(
    ingest: ArchiveIngestResult,
    archive_identity: SourceIdentity,
    encoder_directory: str | Path,
    encoder_identity: EncoderIdentity,
    output_root: str | Path,
    temporary_directory: str | Path,
    config: SemanticConstructionConfig,
    *,
    batch_size: int = 256,
    device: str = "cuda",
    progress: EvidenceProgress | None = None,
) -> SemanticEvidenceArtifact:
    """Select construction contexts, run MiniLM once, and publish reusable evidence."""
    if type(ingest) is not ArchiveIngestResult:
        raise TypeError("semantic evidence requires an ArchiveIngestResult")
    working = Path(temporary_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"semantic evidence temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    _emit(progress, "contexts", 0, ingest.audit.archive_group_count, "selecting exact construction contexts")
    selected_heaps: dict[tuple[Role, str], list[tuple[int, str, SemanticContext]]] = defaultdict(list)
    pair_masses: Counter[tuple[str, str]] = Counter()
    construction_groups = 0
    construction_tokens = 0
    nonconstruction_tokens = 0
    with ingest.story_spool_path.open("rb") as story_spool:
        for completed, group in enumerate(iter_archive_groups(ingest.groups_path), start=1):
            if group.get("status") != "eligible":
                continue
            group_sha = _text(group, "normalized_story_sha256")
            active_tokens = _integer(group, "active_token_count")
            recipe = _object(group, "recipe")
            noun, verb = _text(recipe, "noun"), _text(recipe, "verb")
            if not is_construction_group(group_sha, config):
                pair_masses[(noun, verb)] += active_tokens
                nonconstruction_tokens += active_tokens
            else:
                construction_groups += 1
                construction_tokens += active_tokens
                occurrence = min(
                    _objects(group, "occurrences"),
                    key=lambda item: _text(item, "record_id"),
                )
                story_spool.seek(_integer(occurrence, "spool_offset"))
                raw_story = story_spool.read(_integer(occurrence, "byte_length"))
                if sha256(raw_story).hexdigest() != _text(occurrence, "story_sha256"):
                    raise SemanticEvidenceError("construction story spool changed")
                story = raw_story.decode("utf-8", errors="strict")
                for role, word in (("noun", noun), ("verb", verb)):
                    for context in story_contexts(
                        role,
                        word,
                        group_sha,
                        _text(occurrence, "record_id"),
                        _text(occurrence, "story_sha256"),
                        story,
                    ):
                        _push_lowest_context(
                            selected_heaps[(role, word)],
                            context,
                            config.maximum_contexts_per_word,
                        )
            if completed % 50_000 == 0:
                _emit(
                    progress,
                    "contexts",
                    completed,
                    ingest.audit.archive_group_count,
                    "selecting exact construction contexts",
                )
    _emit(
        progress,
        "contexts",
        ingest.audit.archive_group_count,
        ingest.audit.archive_group_count,
        "construction contexts and non-construction masses selected",
    )
    encoder = MiniLMEncoder(encoder_directory, encoder_identity, device=device)
    contexts = tuple(
        context
        for key in sorted(selected_heaps)
        for _, _, context in sorted(
            selected_heaps[key],
            key=lambda item: (item[2].selection_sha256, item[2].record_id),
        )
    )
    selected_contexts = tuple(
        SelectedContext(
            context=context,
            encoder_text=target_centered_encoder_text(
                context,
                encoder.tokenizer,
                config.context_wordpiece_limit,
            ),
        )
        for context in contexts
    )
    role_words = tuple(
        sorted(
            {
                (role, word)
                for noun, verb in pair_masses
                for role, word in (("noun", noun), ("verb", verb))
            }
        )
    )
    anchors = tuple(
        (role, word, anchor_role, template_index, template.format(word=word))
        for role, word in role_words
        for anchor_role, templates in (
            ("noun", config.noun_anchors),
            ("verb", config.verb_anchors),
        )
        for template_index, template in enumerate(templates)
    )
    texts = tuple(item[4] for item in anchors) + tuple(
        item.encoder_text for item in selected_contexts
    )
    _emit(progress, "embedding", 0, len(texts), "running pinned float32 MiniLM inference")
    embeddings = encoder.embed(
        texts,
        batch_size,
        progress=lambda completed, total: _emit(
            progress,
            "embedding",
            completed,
            total,
            "running pinned float32 MiniLM inference",
        ),
    )
    _emit(progress, "embedding", len(texts), len(texts), "MiniLM inference complete")
    index_records = tuple(
        {
            "anchor_role": anchor_role,
            "kind": "anchor",
            "role": role,
            "row": row,
            "template_index": template_index,
            "word": word,
        }
        for row, (role, word, anchor_role, template_index, _) in enumerate(anchors)
    ) + tuple(
        {
            "context_selection_sha256": item.context.selection_sha256,
            "kind": "context",
            "role": item.context.role,
            "row": len(anchors) + index,
            "word": item.context.word,
        }
        for index, item in enumerate(selected_contexts)
    )
    _write_jsonl(working / "contexts.jsonl", (item.as_record() for item in selected_contexts))
    _write_jsonl(working / "embedding-index.jsonl", index_records)
    _write_jsonl(
        working / "role-pair-masses.jsonl",
        (
            {"noun": noun, "token_mass": mass, "verb": verb}
            for (noun, verb), mass in sorted(pair_masses.items())
        ),
    )
    with (working / "embeddings.f32.npy").open("wb") as output:
        np.save(output, np.asarray(embeddings, dtype="<f4"), allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    content = {
        "archive": archive_identity.as_record(),
        "config": config.evidence_record(),
        "construction_group_count": construction_groups,
        "construction_token_count": construction_tokens,
        "context_sha256": _file_sha256(working / "contexts.jsonl"),
        "embedding_count": len(embeddings),
        "embedding_index_sha256": _file_sha256(working / "embedding-index.jsonl"),
        "embeddings_sha256": _file_sha256(working / "embeddings.f32.npy"),
        "encoder": encoder_identity.as_record(),
        "format": EVIDENCE_FORMAT,
        "nonconstruction_token_count": nonconstruction_tokens,
        "role_pair_masses_sha256": _file_sha256(working / "role-pair-masses.jsonl"),
        "schema_version": SCHEMA_VERSION,
    }
    evidence_sha256 = record_sha256(content)
    _write_json(
        working / "evidence.json",
        {**content, "evidence_sha256": evidence_sha256},
    )
    _write_tree(working, evidence_sha256)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    target = output / evidence_sha256
    if target.exists():
        raise FileExistsError(f"semantic encoder evidence already exists: {target}")
    os.rename(working, target)
    _fsync_directory(output)
    return load_semantic_evidence(target)


def load_semantic_evidence(path: str | Path) -> SemanticEvidenceArtifact:
    """Strictly authenticate one reusable semantic encoder evidence tree."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SemanticEvidenceError("semantic evidence must be a regular directory")
    tree = _load_json(root / "tree.json")
    if set(tree) != {"evidence_sha256", "files", "format", "schema_version"}:
        raise SemanticEvidenceError("semantic evidence tree fields changed")
    if tree["format"] != "tinyworlds-p-semantic-evidence-tree" or tree["schema_version"] != 1:
        raise SemanticEvidenceError("unsupported semantic evidence tree")
    evidence_sha = _string(tree, "evidence_sha256")
    if root.name != evidence_sha:
        raise SemanticEvidenceError("semantic evidence directory identity changed")
    _validate_tree(root, tree)
    record = _load_json(root / "evidence.json")
    required = {
        "archive",
        "config",
        "construction_group_count",
        "construction_token_count",
        "context_sha256",
        "embedding_count",
        "embedding_index_sha256",
        "embeddings_sha256",
        "encoder",
        "evidence_sha256",
        "format",
        "nonconstruction_token_count",
        "role_pair_masses_sha256",
        "schema_version",
    }
    if set(record) != required or record["format"] != EVIDENCE_FORMAT:
        raise SemanticEvidenceError("semantic evidence contract changed")
    content = {key: value for key, value in record.items() if key != "evidence_sha256"}
    if record_sha256(content) != evidence_sha or record["evidence_sha256"] != evidence_sha:
        raise SemanticEvidenceError("semantic evidence content identity is inconsistent")
    for name, field in (
        ("contexts.jsonl", "context_sha256"),
        ("embedding-index.jsonl", "embedding_index_sha256"),
        ("embeddings.f32.npy", "embeddings_sha256"),
        ("role-pair-masses.jsonl", "role_pair_masses_sha256"),
    ):
        if _file_sha256(root / name) != record[field]:
            raise SemanticEvidenceError(f"semantic evidence file changed: {name}")
    encoder = _encoder_identity(_mapping(record, "encoder"))
    archive = _source_identity(_mapping(record, "archive"))
    config = _config_from_record(_mapping(record, "config"), evidence_only=True)
    embeddings = np.load(root / "embeddings.f32.npy", mmap_mode="r", allow_pickle=False)
    embedding_count = _integer(record, "embedding_count")
    if embeddings.shape != (embedding_count, encoder.dimension) or embeddings.dtype != np.dtype("<f4"):
        raise SemanticEvidenceError("semantic evidence array shape or dtype changed")
    index_count = sum(1 for _ in _iter_jsonl(root / "embedding-index.jsonl"))
    if index_count != embedding_count:
        raise SemanticEvidenceError("semantic embedding index length changed")
    return SemanticEvidenceArtifact(
        root=root.resolve(),
        evidence_sha256=evidence_sha,
        archive_identity=archive,
        encoder_identity=encoder,
        config=config,
        embedding_count=embedding_count,
        dimension=encoder.dimension,
        construction_group_count=_integer(record, "construction_group_count"),
        construction_token_count=_integer(record, "construction_token_count"),
        nonconstruction_token_count=_integer(record, "nonconstruction_token_count"),
    )


def load_evidence_arrays(
    artifact: SemanticEvidenceArtifact,
) -> tuple[np.ndarray, tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Load the authenticated embeddings, index records, and exact contexts."""
    embeddings = np.load(
        artifact.root / "embeddings.f32.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    return (
        embeddings,
        tuple(_iter_jsonl(artifact.root / "embedding-index.jsonl")),
        tuple(_iter_jsonl(artifact.root / "contexts.jsonl")),
    )


def load_role_pair_masses(
    artifact: SemanticEvidenceArtifact,
) -> dict[tuple[str, str], int]:
    """Load exact non-construction noun-by-verb group-token mass."""
    pairs = {
        (_text(record, "noun"), _text(record, "verb")): _integer(record, "token_mass")
        for record in _iter_jsonl(artifact.root / "role-pair-masses.jsonl")
    }
    if sum(pairs.values()) != artifact.nonconstruction_token_count:
        raise SemanticEvidenceError("role-pair masses do not cover non-construction mass")
    return pairs


def _push_lowest_context(
    heap: list[tuple[int, str, SemanticContext]],
    context: SemanticContext,
    maximum_count: int,
) -> None:
    key = -int(context.selection_sha256, 16)
    item = (key, context.selection_sha256, context)
    if len(heap) < maximum_count:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _emit(
    progress: EvidenceProgress | None,
    phase: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if progress is not None:
        progress(phase, completed, total, detail)


def _write_tree(root: Path, evidence_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "evidence_sha256": evidence_sha256,
            "files": list(files),
            "format": "tinyworlds-p-semantic-evidence-tree",
            "schema_version": 1,
        },
    )


def _validate_tree(root: Path, tree: Mapping[str, object]) -> None:
    raw_files = tree.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise SemanticEvidenceError("semantic evidence tree files must be objects")
    descriptors = tuple(raw_files)
    paths = tuple(_text(item, "relative_path") for item in descriptors)
    if paths != tuple(sorted(set(paths))):
        raise SemanticEvidenceError("semantic evidence tree paths are not canonical")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != set(paths) | {"tree.json"} or any(path.is_symlink() for path in root.rglob("*")):
        raise SemanticEvidenceError("semantic evidence tree membership changed")
    for descriptor in descriptors:
        relative = _text(descriptor, "relative_path")
        candidate = root / relative
        if candidate.stat().st_size != _integer(descriptor, "size_bytes"):
            raise SemanticEvidenceError(f"semantic evidence file size changed: {relative}")
        if _file_sha256(candidate) != _text(descriptor, "sha256"):
            raise SemanticEvidenceError(f"semantic evidence file checksum changed: {relative}")


def _encoder_identity(record: Mapping[str, object]) -> EncoderIdentity:
    files = record.get("files")
    if type(files) is not list or any(type(item) is not dict for item in files):
        raise SemanticEvidenceError("encoder identity files are malformed")
    return EncoderIdentity(
        identifier=_text(record, "identifier"),
        revision=_text(record, "revision"),
        dimension=_integer(record, "dimension"),
        files=tuple(
            ModelFile(
                relative_path=_text(item, "relative_path"),
                size_bytes=_integer(item, "size_bytes"),
                sha256=_text(item, "sha256"),
            )
            for item in files
        ),
        pooling=_text(record, "pooling"),
        normalization=_text(record, "normalization"),
        dtype=_text(record, "dtype"),
    )


def _source_identity(record: Mapping[str, object]) -> SourceIdentity:
    return SourceIdentity(
        dataset_id=_text(record, "dataset_id"),
        revision=_text(record, "revision"),
        filename=_text(record, "filename"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_text(record, "sha256"),
    )


def _config_from_record(
    record: Mapping[str, object],
    *,
    evidence_only: bool,
) -> SemanticConstructionConfig:
    defaults = SemanticConstructionConfig()
    if not evidence_only:
        from apm.data.text.tinyworlds_p_semantic.contracts import semantic_config_from_record

        try:
            return semantic_config_from_record(dict(record))
        except ValueError as error:
            raise SemanticEvidenceError("semantic evidence configuration changed") from error
    expected_fields = set(defaults.evidence_record())
    if set(record) != expected_fields:
        raise SemanticEvidenceError("semantic evidence configuration fields changed")
    anchors = lambda field: tuple(record[field]) if type(record[field]) is list else ()
    try:
        return replace(
            defaults,
            version=_text(record, "version"),
            construction_modulus=_integer(record, "construction_modulus"),
            construction_residue=_integer(record, "construction_residue"),
            maximum_contexts_per_word=_integer(record, "maximum_contexts_per_word"),
            context_wordpiece_limit=_integer(record, "context_wordpiece_limit"),
            noun_anchors=anchors("noun_anchors"),
            verb_anchors=anchors("verb_anchors"),
        )
    except (TypeError, ValueError) as error:
        raise SemanticEvidenceError("semantic evidence configuration changed") from error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_jsonl(path: Path, values: Iterator[object] | Sequence[object]) -> None:
    with path.open("wb") as output:
        for value in values:
            output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SemanticEvidenceError(f"invalid semantic evidence JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise SemanticEvidenceError(f"semantic evidence JSON is not canonical: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise SemanticEvidenceError(f"invalid JSONL at {path}:{line_number}") from error
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise SemanticEvidenceError(f"noncanonical JSONL at {path}:{line_number}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise SemanticEvidenceError(f"field {field!r} must be an object")
    return value


def _object(record: Mapping[str, object], field: str) -> dict[str, object]:
    return _mapping(record, field)


def _objects(record: Mapping[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise SemanticEvidenceError(f"field {field!r} must be object records")
    return tuple(value)


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise SemanticEvidenceError(f"field {field!r} must be nonempty text")
    return value


def _string(record: Mapping[str, object], field: str) -> str:
    return _text(record, field)


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise SemanticEvidenceError(f"field {field!r} must be a nonnegative integer")
    return value


__all__ = [
    "EvidenceProgress",
    "MiniLMEncoder",
    "SelectedContext",
    "SemanticEvidenceError",
    "discover_encoder_identity",
    "exact_whole_word_spans",
    "is_construction_group",
    "load_evidence_arrays",
    "load_role_pair_masses",
    "load_semantic_evidence",
    "namespaced_sha256",
    "prepare_semantic_evidence",
    "select_lowest_hash_contexts",
    "story_contexts",
    "target_centered_encoder_text",
    "verify_encoder_identity",
]
