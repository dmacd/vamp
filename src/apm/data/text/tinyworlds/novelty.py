"""Pinned original-corpus identity and streaming TinyWorlds novelty audit."""

from __future__ import annotations

from dataclasses import dataclass
import codecs
from pathlib import Path
import re
from typing import Callable

from apm.data.text.curricula import PinnedDatasetFile, verify_pinned_dataset_file
from apm.data.text.tinyworlds.query_generation import TinyWorldsBundle
from apm.data.text.tinyworlds.world_generation import (
    ACTOR_TYPE,
    ATTRIBUTE_TYPE,
    CONTEXT_TYPE,
    PLACE_TYPE,
)


ORIGINAL_TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
ORIGINAL_TINYSTORIES_TRAIN = PinnedDatasetFile(
    filename="TinyStories-train.txt",
    size_bytes=1_924_281_556,
    sha256="c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
)
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


@dataclass(frozen=True)
class NoveltyTerm:
    """One generated lexical form that must be absent from pretraining text."""

    text: str
    kind: str
    source_id: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (self.text, self.kind, self.source_id)
        ):
            raise ValueError("novelty terms require canonical nonempty text and IDs")
        if tuple(_WORD_PATTERN.findall(self.text)) != (self.text,):
            raise ValueError("novelty term text must be exactly one lexical word")


@dataclass(frozen=True)
class NoveltyHit:
    """One generated form observed at least once in the pinned corpus."""

    term: NoveltyTerm
    occurrence_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.term, NoveltyTerm):
            raise TypeError("novelty hit term must be a NoveltyTerm")
        if type(self.occurrence_count) is not int or self.occurrence_count <= 0:
            raise ValueError("novelty hit occurrence_count must be positive")


@dataclass(frozen=True)
class NoveltyAuditReport:
    """Complete result of one streamed, pinned-corpus lexical audit."""

    revision: str
    corpus_file: PinnedDatasetFile
    audited_terms: tuple[NoveltyTerm, ...]
    hits: tuple[NoveltyHit, ...]

    def __post_init__(self) -> None:
        if self.revision != ORIGINAL_TINYSTORIES_REVISION:
            raise ValueError("novelty audit revision must match the benchmark pin")
        if not isinstance(self.corpus_file, PinnedDatasetFile):
            raise TypeError("novelty audit corpus_file must be pinned")
        if not self.audited_terms or any(
            not isinstance(term, NoveltyTerm) for term in self.audited_terms
        ):
            raise ValueError("novelty audit requires generated lexical terms")
        keys = tuple((term.kind, term.source_id, term.text.casefold()) for term in self.audited_terms)
        if len(set(keys)) != len(keys):
            raise ValueError("audited novelty terms must be unique")
        if any(
            not isinstance(hit, NoveltyHit) or hit.term not in self.audited_terms
            for hit in self.hits
        ):
            raise ValueError("novelty hits must reference audited terms")

    @property
    def passed(self) -> bool:
        """Return whether every generated lexical form is absent."""
        return not self.hits


def audit_nonce_terms(
    corpus_path: str | Path,
    terms: tuple[NoveltyTerm, ...],
    *,
    corpus_file: PinnedDatasetFile = ORIGINAL_TINYSTORIES_TRAIN,
    revision: str = ORIGINAL_TINYSTORIES_REVISION,
    progress_bytes: Callable[[int], None] | None = None,
) -> NoveltyAuditReport:
    """Verify the corpus pin, then stream it once while counting exact words."""
    if not terms or any(not isinstance(term, NoveltyTerm) for term in terms):
        raise ValueError("novelty audit terms must be a nonempty tuple")
    verified = verify_pinned_dataset_file(corpus_path, corpus_file)
    terms_by_word: dict[str, tuple[NoveltyTerm, ...]] = {}
    for term in terms:
        terms_by_word[term.text.casefold()] = terms_by_word.get(
            term.text.casefold(), ()
        ) + (term,)
    counts = {term: 0 for term in terms}
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    word_tail = ""
    with verified.open("rb") as source:
        while raw_chunk := source.read(8 * 1024 * 1024):
            decoded = decoder.decode(raw_chunk)
            if progress_bytes is not None:
                progress_bytes(len(raw_chunk))
            word_tail = _count_complete_words(
                word_tail + decoded,
                terms_by_word,
                counts,
                final=False,
            )
        final_text = word_tail + decoder.decode(b"", final=True)
        _count_complete_words(final_text, terms_by_word, counts, final=True)
    return NoveltyAuditReport(
        revision=revision,
        corpus_file=corpus_file,
        audited_terms=terms,
        hits=tuple(
            NoveltyHit(term, counts[term])
            for term in terms
            if counts[term]
        ),
    )


def novelty_terms_for_bundles(
    bundles: tuple[TinyWorldsBundle, ...],
) -> tuple[NoveltyTerm, ...]:
    """Enumerate every generated name/class/role and visible inflection once."""
    if type(bundles) is not tuple or not bundles or any(
        type(bundle) is not TinyWorldsBundle for bundle in bundles
    ):
        raise ValueError("bundles must be a nonempty tuple of TinyWorldsBundle values")
    kind_by_entity_type = {
        ACTOR_TYPE: "name",
        PLACE_TYPE: "class",
        ATTRIBUTE_TYPE: "role",
        CONTEXT_TYPE: "class",
    }
    terms = tuple(
        term
        for bundle in bundles
        for entity in bundle.entities
        for term in (
            NoveltyTerm(
                entity.name,
                kind_by_entity_type[entity.entity_type],
                f"{bundle.bundle_id}:{entity.entity_id}",
            ),
            *(
                NoveltyTerm(
                    inflection,
                    "inflection",
                    f"{bundle.bundle_id}:{entity.entity_id}:{index}",
                )
                for index, inflection in enumerate(entity.inflections)
            ),
        )
    )
    if len({(term.kind, term.source_id, term.text.casefold()) for term in terms}) != len(
        terms
    ):
        raise ValueError("generated novelty-term identities must be unique")
    return terms


def _count_complete_words(
    text: str,
    terms_by_word: dict[str, tuple[NoveltyTerm, ...]],
    counts: dict[NoveltyTerm, int],
    *,
    final: bool,
) -> str:
    matches = tuple(_WORD_PATTERN.finditer(text))
    trailing_match = (
        matches[-1]
        if matches and matches[-1].end() == len(text) and not final
        else None
    )
    for match in matches[:-1] if trailing_match is not None else matches:
        for term in terms_by_word.get(match.group(0).casefold(), ()):
            counts[term] += 1
    return trailing_match.group(0) if trailing_match is not None else ""


__all__ = [
    "NoveltyAuditReport",
    "NoveltyHit",
    "NoveltyTerm",
    "ORIGINAL_TINYSTORIES_REVISION",
    "ORIGINAL_TINYSTORIES_TRAIN",
    "audit_nonce_terms",
    "novelty_terms_for_bundles",
]
