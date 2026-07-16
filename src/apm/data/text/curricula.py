"""Offline deterministic text curricula and pinned TinyStories data contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from string import ascii_lowercase
import re
import unicodedata

import numpy as np


TINYSTORIES_DOCUMENT_SEPARATOR = "<|endoftext|>"
TINYSTORIES_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS = 1_024
TINY_SHAKESPEARE_EVALUATION_EXAMPLES_PER_TASK_AND_PREFIX = 64
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)
_PINNED_FILE_READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PinnedDatasetFile:
    """One immutable dataset filename, byte count, and SHA-256 digest."""

    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("pinned dataset filename must not be empty")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("pinned dataset size must be a positive integer")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("pinned dataset SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class TinyStoriesSourceContract:
    """Complete offline identity of the supported TinyStories V2/GPT-4 files."""

    dataset_id: str
    revision: str
    train_file: PinnedDatasetFile
    validation_file: PinnedDatasetFile
    document_separator: str = TINYSTORIES_DOCUMENT_SEPARATOR

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.revision:
            raise ValueError("TinyStories dataset identity must not be empty")
        if _GIT_REVISION_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("TinyStories dataset revision must be a pinned Git hash")
        if not isinstance(self.train_file, PinnedDatasetFile) or not isinstance(
            self.validation_file,
            PinnedDatasetFile,
        ):
            raise TypeError("TinyStories files must be pinned dataset records")
        if self.train_file.filename == self.validation_file.filename:
            raise ValueError("TinyStories train and validation filenames must differ")
        if not self.document_separator:
            raise ValueError("TinyStories document separator must not be empty")


TINYSTORIES_V2_SOURCE = TinyStoriesSourceContract(
    dataset_id="roneneldan/TinyStories",
    revision=TINYSTORIES_DATASET_REVISION,
    train_file=PinnedDatasetFile(
        filename="TinyStoriesV2-GPT4-train.txt",
        size_bytes=2_227_753_162,
        sha256="6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
    ),
    validation_file=PinnedDatasetFile(
        filename="TinyStoriesV2-GPT4-valid.txt",
        size_bytes=22_502_601,
        sha256="6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
    ),
)


def verify_pinned_dataset_file(
    path: str | Path,
    expected_file: PinnedDatasetFile,
) -> Path:
    """Verify one already-local pinned file by streaming its bytes once."""
    if not isinstance(expected_file, PinnedDatasetFile):
        raise TypeError("expected_file must be a PinnedDatasetFile")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.name != expected_file.filename:
        raise ValueError(
            f"expected pinned filename {expected_file.filename!r}, "
            f"got {source_path.name!r}"
        )

    digest = sha256()
    measured_size = 0
    with source_path.open("rb") as source:
        while chunk := source.read(_PINNED_FILE_READ_CHUNK_SIZE):
            measured_size += len(chunk)
            digest.update(chunk)

    if measured_size != expected_file.size_bytes:
        raise ValueError(
            f"pinned file size mismatch for {expected_file.filename!r}: "
            f"expected {expected_file.size_bytes}, got {measured_size}"
        )
    measured_sha256 = digest.hexdigest()
    if measured_sha256 != expected_file.sha256:
        raise ValueError(
            f"pinned file SHA-256 mismatch for {expected_file.filename!r}: "
            f"expected {expected_file.sha256}, got {measured_sha256}"
        )
    return source_path


def load_pinned_dataset_text(
    path: str | Path,
    expected_file: PinnedDatasetFile,
) -> str:
    """Verify and UTF-8-decode one already-local pinned dataset file."""
    verified_path = verify_pinned_dataset_file(path, expected_file)
    with verified_path.open("r", encoding="utf-8", newline="") as source:
        return source.read()


@dataclass(frozen=True)
class TextDocument:
    """One nonempty normalized document identified by its content SHA-256."""

    content_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.text or normalize_text(self.text) != self.text:
            raise ValueError("document text must be nonempty and normalized")
        if self.content_id != sha256_content_id(self.text):
            raise ValueError("document content_id must hash its normalized text")


@dataclass(frozen=True)
class DocumentSplits:
    """Immutable document-level train, validation, and test splits."""

    train: tuple[TextDocument, ...]
    validation: tuple[TextDocument, ...]
    test: tuple[TextDocument, ...]

    def __post_init__(self) -> None:
        split_values = (self.train, self.validation, self.test)
        if any(not isinstance(split, tuple) for split in split_values):
            raise TypeError("document splits must be tuples")
        if any(
            not isinstance(document, TextDocument)
            for split in split_values
            for document in split
        ):
            raise TypeError("document splits must contain TextDocument values")
        id_sets = tuple(
            {document.content_id for document in split}
            for split in split_values
        )
        if any(len(ids) != len(split) for ids, split in zip(id_sets, split_values)):
            raise ValueError("documents must be unique within every split")
        if any(
            id_sets[left] & id_sets[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("train, validation, and test documents must be disjoint")


@dataclass(frozen=True)
class CorpusSplits:
    """Raw train, validation, and test corpus spans with exact formatting."""

    train: str
    validation: str
    test: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str)
            for value in (self.train, self.validation, self.test)
        ):
            raise TypeError("corpus splits must contain strings")


@dataclass(frozen=True)
class CorpusDocumentSplits:
    """Raw document tuples for each split, preserving every character exactly."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        split_values = (self.train, self.validation, self.test)
        if any(not isinstance(split, tuple) or not split for split in split_values):
            raise ValueError("corpus document splits must be nonempty tuples")
        if any(
            not isinstance(document, str) or not document
            for split in split_values
            for document in split
        ):
            raise ValueError("corpus documents must be nonempty raw strings")


@dataclass(frozen=True)
class CorpusTask:
    """One raw-corpus task whose document boundaries survive tokenization."""

    task_id: str
    splits: CorpusDocumentSplits

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("corpus task_id must not be empty")
        if not isinstance(self.splits, CorpusDocumentSplits):
            raise TypeError("corpus task splits must be CorpusDocumentSplits")


@dataclass(frozen=True)
class CorpusCurriculum:
    """Exactly four immutable raw-corpus tasks in deterministic order."""

    curriculum_id: str
    tasks: tuple[CorpusTask, ...]

    def __post_init__(self) -> None:
        if not self.curriculum_id:
            raise ValueError("corpus curriculum_id must not be empty")
        if (
            not isinstance(self.tasks, tuple)
            or len(self.tasks) != 4
            or any(not isinstance(task, CorpusTask) for task in self.tasks)
        ):
            raise ValueError("corpus curricula must contain exactly four tasks")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("corpus curriculum task IDs must be unique")


@dataclass(frozen=True)
class CharacterPermutationTask:
    """One seeded ASCII-letter permutation over the same corpus splits."""

    task_id: str
    seed: int
    splits: CorpusSplits

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("character-permutation task_id must not be empty")
        _validate_seed(self.seed)
        if not isinstance(self.splits, CorpusSplits):
            raise TypeError("character-permutation splits must be CorpusSplits")


@dataclass(frozen=True)
class CharacterPermutationCurriculum:
    """The fixed four-seed TinyShakespeare character curriculum."""

    curriculum_id: str
    tasks: tuple[CharacterPermutationTask, ...]

    def __post_init__(self) -> None:
        if not self.curriculum_id:
            raise ValueError("character curriculum_id must not be empty")
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, CharacterPermutationTask) for task in self.tasks
        ):
            raise TypeError("character curriculum tasks must be an immutable tuple")
        if tuple(task.seed for task in self.tasks) != (0, 1, 2, 3):
            raise ValueError("character curriculum tasks must use seeds 0 through 3")


@dataclass(frozen=True)
class DocumentTask:
    """One immutable document-level continual-learning task."""

    task_id: str
    splits: DocumentSplits

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("document task_id must not be empty")
        if not isinstance(self.splits, DocumentSplits):
            raise TypeError("document task splits must be DocumentSplits")


@dataclass(frozen=True)
class DocumentCurriculum:
    """Four disjoint document tasks in deterministic order."""

    curriculum_id: str
    tasks: tuple[DocumentTask, ...]

    def __post_init__(self) -> None:
        if not self.curriculum_id:
            raise ValueError("document curriculum_id must not be empty")
        if not isinstance(self.tasks, tuple) or len(self.tasks) != 4 or any(
            not isinstance(task, DocumentTask) for task in self.tasks
        ):
            raise ValueError("document curricula must contain exactly four tasks")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("document curriculum task IDs must be unique")
        for split_name in ("train", "validation", "test"):
            content_ids = tuple(
                document.content_id
                for task in self.tasks
                for document in getattr(task.splits, split_name)
            )
            if len(set(content_ids)) != len(content_ids):
                raise ValueError(
                    f"documents cannot cross {split_name} task boundaries"
                )


@dataclass(frozen=True)
class StorySplitCounts:
    """Exact per-task story counts requested for all three splits."""

    train: int
    validation: int
    test: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.train, self.validation, self.test)
        ):
            raise ValueError("story split counts must be positive integers")


@dataclass(frozen=True)
class EvaluationSpanPreset:
    """Fixed address-prefix sweep and competence-suffix length."""

    prefix_lengths: tuple[int, ...]
    suffix_length: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prefix_lengths, tuple)
            or not self.prefix_lengths
            or any(
                type(length) is not int or length <= 0
                for length in self.prefix_lengths
            )
            or tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths
        ):
            raise ValueError("prefix lengths must be positive, unique, and increasing")
        if type(self.suffix_length) is not int or self.suffix_length <= 0:
            raise ValueError("suffix length must be a positive integer")


@dataclass(frozen=True)
class TinyStoriesSingleGpuPreset:
    """Decision-complete bounded TinyStories continual-learning preset."""

    task_count: int
    stories_per_task: StorySplitCounts
    context_length: int
    lora_rank: int
    lora_alpha: float
    batch_size: int
    adapter_steps_per_task: int
    parent_probe_count: int
    content_key_probe_count: int
    evaluation_examples_per_task_and_prefix: int
    max_nodes: int
    max_edges: int
    peak_device_memory_gib: int
    evaluation: EvaluationSpanPreset

    def __post_init__(self) -> None:
        if type(self.task_count) is not int or self.task_count != 4:
            raise ValueError("TinyStories preset task_count must equal four")
        if not isinstance(self.stories_per_task, StorySplitCounts):
            raise TypeError("stories_per_task must be StorySplitCounts")
        positive_integer_fields = (
            self.context_length,
            self.lora_rank,
            self.batch_size,
            self.adapter_steps_per_task,
            self.parent_probe_count,
            self.content_key_probe_count,
            self.evaluation_examples_per_task_and_prefix,
            self.max_nodes,
            self.max_edges,
            self.peak_device_memory_gib,
        )
        if any(
            type(value) is not int or value <= 0
            for value in positive_integer_fields
        ):
            raise ValueError("TinyStories preset dimensions and budgets must be positive")
        if (
            isinstance(self.lora_alpha, bool)
            or not isinstance(self.lora_alpha, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.lora_alpha))
            or self.lora_alpha <= 0.0
        ):
            raise ValueError("TinyStories preset LoRA alpha must be finite and positive")
        if self.max_nodes != self.task_count + 1:
            raise ValueError("TinyStories max_nodes must equal task_count plus the root")
        if self.max_edges != self.task_count or self.max_edges != self.max_nodes - 1:
            raise ValueError("TinyStories max_edges must equal task_count and max_nodes - 1")
        if not isinstance(self.evaluation, EvaluationSpanPreset):
            raise TypeError("TinyStories evaluation must be an EvaluationSpanPreset")


TINY_SHAKESPEARE_EVALUATION_PRESET = EvaluationSpanPreset(
    prefix_lengths=(32, 64, 128),
    suffix_length=128,
)
TINYSTORIES_EVALUATION_PRESET = EvaluationSpanPreset(
    prefix_lengths=(16, 32, 64, 128),
    suffix_length=128,
)
TINYSTORIES_SINGLE_GPU_PRESET = TinyStoriesSingleGpuPreset(
    task_count=4,
    stories_per_task=StorySplitCounts(10_000, 1_000, 1_000),
    context_length=256,
    lora_rank=8,
    lora_alpha=8.0,
    batch_size=32,
    adapter_steps_per_task=2_000,
    parent_probe_count=256,
    content_key_probe_count=256,
    evaluation_examples_per_task_and_prefix=256,
    max_nodes=5,
    max_edges=4,
    peak_device_memory_gib=12,
    evaluation=TINYSTORIES_EVALUATION_PRESET,
)


@dataclass(frozen=True)
class TopicConcept:
    """One semantic concept represented by explicit whole-word forms."""

    name: str
    forms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.forms, tuple) or not self.forms:
            raise ValueError("topic concepts require a name and forms")
        if len(set(self.forms)) != len(self.forms) or any(
            form != form.casefold() or _WORD_PATTERN.fullmatch(form) is None
            for form in self.forms
        ):
            raise ValueError("topic concept forms must be unique case-folded words")


@dataclass(frozen=True)
class TopicDefinition:
    """One ordered TinyStories topic and its distinct concepts."""

    name: str
    concepts: tuple[TopicConcept, ...]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.concepts, tuple) or not self.concepts:
            raise ValueError("topic definitions require a name and concepts")
        if any(not isinstance(concept, TopicConcept) for concept in self.concepts):
            raise TypeError("topic definitions must contain TopicConcept values")
        concept_names = tuple(concept.name for concept in self.concepts)
        if len(set(concept_names)) != len(concept_names):
            raise ValueError("topic concept names must be unique")


@dataclass(frozen=True)
class TopicScore:
    """Distinct matched concepts for one topic."""

    topic: str
    matched_concepts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.topic or not isinstance(self.matched_concepts, tuple):
            raise ValueError("topic scores require a topic and immutable matches")
        if len(set(self.matched_concepts)) != len(self.matched_concepts):
            raise ValueError("topic score concepts must be distinct")

    @property
    def distinct_concept_count(self) -> int:
        """Return the number of distinct matched concepts."""
        return len(self.matched_concepts)


@dataclass(frozen=True)
class TopicAssignment:
    """One unique topic winner with its evidence and runner-up margin."""

    topic: str
    matched_concepts: tuple[str, ...]
    runner_up_score: int
    margin: int

    def __post_init__(self) -> None:
        if (
            not self.topic
            or not isinstance(self.matched_concepts, tuple)
            or len(self.matched_concepts) < 2
        ):
            raise ValueError("topic assignments require at least two immutable matches")
        if type(self.runner_up_score) is not int or self.runner_up_score < 0:
            raise ValueError("topic assignment runner-up score must be nonnegative")
        if type(self.margin) is not int or self.margin < 1:
            raise ValueError("topic assignment margin must be positive")


def _concept(name: str, *forms: str) -> TopicConcept:
    return TopicConcept(name=name, forms=tuple(forms))


TINYSTORIES_TOPICS = (
    TopicDefinition(
        "animals",
        (
            _concept("dog", "dog", "dogs"),
            _concept("cat", "cat", "cats"),
            _concept("bird", "bird", "birds"),
            _concept("rabbit", "rabbit", "rabbits"),
            _concept("bear", "bear", "bears"),
            _concept("lion", "lion", "lions"),
            _concept("horse", "horse", "horses"),
            _concept("cow", "cow", "cows"),
            _concept("sheep", "sheep"),
            _concept("duck", "duck", "ducks"),
            _concept("fish", "fish", "fishes"),
            _concept("frog", "frog", "frogs"),
            _concept("mouse", "mouse", "mice"),
            _concept("elephant", "elephant", "elephants"),
            _concept("monkey", "monkey", "monkeys"),
            _concept("pet", "pet", "pets"),
        ),
    ),
    TopicDefinition(
        "vehicles_tools",
        (
            _concept("car", "car", "cars"),
            _concept("truck", "truck", "trucks"),
            _concept("train", "train", "trains"),
            _concept("bus", "bus", "buses"),
            _concept("bicycle", "bicycle", "bicycles", "bike", "bikes"),
            _concept("boat", "boat", "boats"),
            _concept("plane", "plane", "planes"),
            _concept("tractor", "tractor", "tractors"),
            _concept("hammer", "hammer", "hammers"),
            _concept("saw", "saw", "saws"),
            _concept("wrench", "wrench", "wrenches"),
            _concept("screwdriver", "screwdriver", "screwdrivers"),
            _concept("wheel", "wheel", "wheels"),
            _concept("engine", "engine", "engines"),
            _concept("garage", "garage", "garages"),
        ),
    ),
    TopicDefinition(
        "family_home",
        (
            _concept("mother", "mother", "mothers", "mom", "moms"),
            _concept("father", "father", "fathers", "dad", "dads"),
            _concept("sister", "sister", "sisters"),
            _concept("brother", "brother", "brothers"),
            _concept("family", "family", "families"),
            _concept("friend", "friend", "friends"),
            _concept("home", "home", "homes"),
            _concept("house", "house", "houses"),
            _concept("grandma", "grandma", "grandmas"),
            _concept("grandpa", "grandpa", "grandpas"),
            _concept("parent", "parent", "parents"),
            _concept("neighbor", "neighbor", "neighbors"),
        ),
    ),
    TopicDefinition(
        "fantasy_royalty",
        (
            _concept("king", "king", "kings"),
            _concept("queen", "queen", "queens"),
            _concept("prince", "prince", "princes"),
            _concept("princess", "princess", "princesses"),
            _concept("castle", "castle", "castles"),
            _concept("dragon", "dragon", "dragons"),
            _concept("fairy", "fairy", "fairies"),
            _concept("wizard", "wizard", "wizards"),
            _concept("magic", "magic"),
            _concept("knight", "knight", "knights"),
            _concept("crown", "crown", "crowns"),
            _concept("kingdom", "kingdom", "kingdoms"),
            _concept("unicorn", "unicorn", "unicorns"),
            _concept("witch", "witch", "witches"),
        ),
    ),
)


def normalize_text(text: str) -> str:
    """Return Unicode-NFC text with all whitespace runs collapsed."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def sha256_content_id(text: str) -> str:
    """Hash the canonical normalized UTF-8 representation of text."""
    normalized = normalize_text(text)
    return sha256(normalized.encode("utf-8")).hexdigest()


def normalized_documents(texts: Sequence[str]) -> tuple[TextDocument, ...]:
    """Normalize and deterministically deduplicate supplied documents in order."""
    documents_by_id: dict[str, TextDocument] = {}
    for text in texts:
        normalized = normalize_text(text)
        if not normalized:
            continue
        document = TextDocument(sha256_content_id(normalized), normalized)
        existing = documents_by_id.get(document.content_id)
        if existing is not None and existing.text != document.text:
            raise RuntimeError("SHA-256 collision between distinct normalized documents")
        documents_by_id.setdefault(document.content_id, document)
    return tuple(documents_by_id.values())


def parse_tinystories(aggregate_text: str) -> tuple[TextDocument, ...]:
    """Parse a supplied TinyStories aggregate without filesystem or network access."""
    if not isinstance(aggregate_text, str):
        raise TypeError("TinyStories aggregate must be a string")
    return normalized_documents(
        tuple(aggregate_text.split(TINYSTORIES_DOCUMENT_SEPARATOR))
    )


def prepare_tinystories_splits(
    train_text: str,
    official_validation_text: str,
) -> DocumentSplits:
    """Deduplicate train/evaluation stories and hash-split validation 50/50."""
    parsed_train_documents = parse_tinystories(train_text)
    evaluation_documents = parse_tinystories(official_validation_text)
    evaluation_ids = {document.content_id for document in evaluation_documents}
    train_documents = tuple(
        document
        for document in parsed_train_documents
        if document.content_id not in evaluation_ids
    )
    if len(evaluation_documents) % 2 != 0:
        raise ValueError(
            "deduplicated official validation stories must divide exactly 50/50"
        )
    ordered_evaluation = tuple(
        sorted(evaluation_documents, key=lambda document: document.content_id)
    )
    midpoint = len(ordered_evaluation) // 2
    return DocumentSplits(
        train=train_documents,
        validation=ordered_evaluation[:midpoint],
        test=ordered_evaluation[midpoint:],
    )


def ascii_letter_permutation(seed: int) -> tuple[str, ...]:
    """Return one seeded bijection from lowercase ASCII letters to letters."""
    _validate_seed(seed)
    return tuple(
        str(value)
        for value in np.random.default_rng(seed).permutation(tuple(ascii_lowercase))
    )


def apply_ascii_letter_permutation(text: str, seed: int) -> str:
    """Permute ASCII letters while preserving case and every nonletter exactly."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    targets = ascii_letter_permutation(seed)
    lowercase_mapping = dict(zip(ascii_lowercase, targets))
    uppercase_mapping = {
        source.upper(): target.upper()
        for source, target in lowercase_mapping.items()
    }
    mapping = lowercase_mapping | uppercase_mapping
    return "".join(mapping.get(character, character) for character in text)


def build_tiny_shakespeare_permutation_curriculum(
    splits: CorpusSplits,
) -> CharacterPermutationCurriculum:
    """Build the fixed seeds-0-through-3 TinyShakespeare curriculum."""
    if not isinstance(splits, CorpusSplits):
        raise TypeError("splits must be CorpusSplits")
    tasks = tuple(
        CharacterPermutationTask(
            task_id=f"tinyshakespeare-letter-permutation-{seed}",
            seed=seed,
            splits=CorpusSplits(
                train=apply_ascii_letter_permutation(splits.train, seed),
                validation=apply_ascii_letter_permutation(splits.validation, seed),
                test=apply_ascii_letter_permutation(splits.test, seed),
            ),
        )
        for seed in range(4)
    )
    return CharacterPermutationCurriculum(
        curriculum_id="tinyshakespeare-character-permutation",
        tasks=tasks,
    )


def build_tiny_shakespeare_region_curriculum(
    splits: CorpusSplits,
) -> CorpusCurriculum:
    """Split every raw corpus span into four exact contiguous task regions."""
    if not isinstance(splits, CorpusSplits):
        raise TypeError("splits must be CorpusSplits")
    chunks_by_split = {
        split_name: _contiguous_text_chunks(getattr(splits, split_name))
        for split_name in ("train", "validation", "test")
    }
    return CorpusCurriculum(
        curriculum_id="tinyshakespeare-corpus-region",
        tasks=tuple(
            CorpusTask(
                task_id=f"tinyshakespeare-corpus-region-{task_index}",
                splits=CorpusDocumentSplits(
                    train=(chunks_by_split["train"][task_index],),
                    validation=(chunks_by_split["validation"][task_index],),
                    test=(chunks_by_split["test"][task_index],),
                ),
            )
            for task_index in range(4)
        ),
    )


def raw_text_sha256(text: str) -> str:
    """Hash raw UTF-8 text without normalization or whitespace changes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sha256(text.encode("utf-8")).hexdigest()


def stable_hash_raw_text_task_index(text: str, task_count: int = 4) -> int:
    """Assign raw text to a task using its unmodified content SHA-256."""
    if type(task_count) is not int or task_count <= 0:
        raise ValueError("task_count must be a positive integer")
    return int(raw_text_sha256(text), 16) % task_count


def build_tiny_shakespeare_stable_hash_curriculum(
    splits: CorpusSplits,
    macro_document_characters: int = TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS,
) -> CorpusCurriculum:
    """Hash-assign fixed contiguous raw macro-documents to four tasks."""
    if not isinstance(splits, CorpusSplits):
        raise TypeError("splits must be CorpusSplits")
    if (
        type(macro_document_characters) is not int
        or macro_document_characters <= 0
    ):
        raise ValueError("macro_document_characters must be a positive integer")
    documents_by_split = {
        split_name: _contiguous_macro_documents(
            getattr(splits, split_name),
            macro_document_characters,
        )
        for split_name in ("train", "validation", "test")
    }
    assignments = {
        split_name: tuple(
            tuple(
                document
                for document in documents
                if stable_hash_raw_text_task_index(document) == task_index
            )
            for task_index in range(4)
        )
        for split_name, documents in documents_by_split.items()
    }
    empty_assignments = tuple(
        (split_name, task_index)
        for split_name, task_documents in assignments.items()
        for task_index, documents in enumerate(task_documents)
        if not documents
    )
    if empty_assignments:
        raise ValueError(
            "stable-hash macro-documents must populate every task and split; "
            f"empty assignments: {empty_assignments}"
        )
    return CorpusCurriculum(
        curriculum_id="tinyshakespeare-stable-hash-negative-control",
        tasks=tuple(
            CorpusTask(
                task_id=f"tinyshakespeare-stable-hash-{task_index}",
                splits=CorpusDocumentSplits(
                    train=assignments["train"][task_index],
                    validation=assignments["validation"][task_index],
                    test=assignments["test"][task_index],
                ),
            )
            for task_index in range(4)
        ),
    )


def seeded_token_permutation(
    vocabulary_size: int,
    seed: int,
    fixed_token_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Create a seeded token-ID bijection while preserving designated IDs."""
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be a positive integer")
    _validate_seed(seed)
    fixed = tuple(int(token_id) for token_id in fixed_token_ids)
    if (
        len(set(fixed)) != len(fixed)
        or any(token_id < 0 or token_id >= vocabulary_size for token_id in fixed)
    ):
        raise ValueError("fixed token IDs must be unique and inside the vocabulary")
    movable = np.asarray(
        tuple(token_id for token_id in range(vocabulary_size) if token_id not in fixed),
        dtype=np.int64,
    )
    mapping = np.arange(vocabulary_size, dtype=np.int64)
    mapping[movable] = np.random.default_rng(seed).permutation(movable)
    return tuple(int(token_id) for token_id in mapping)


def apply_token_permutation(
    token_ids: Sequence[int],
    permutation: Sequence[int],
) -> tuple[int, ...]:
    """Apply a validated source-ID-to-target-ID token permutation."""
    mapping = tuple(int(token_id) for token_id in permutation)
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("token permutation must contain every vocabulary ID once")
    source_ids = tuple(int(token_id) for token_id in token_ids)
    if any(token_id < 0 or token_id >= len(mapping) for token_id in source_ids):
        raise ValueError("token IDs must be inside the permutation vocabulary")
    return tuple(mapping[token_id] for token_id in source_ids)


def build_contiguous_region_curriculum(
    splits: DocumentSplits,
) -> DocumentCurriculum:
    """Split every document split into four contiguous noncrossing regions."""
    if not isinstance(splits, DocumentSplits):
        raise TypeError("splits must be DocumentSplits")
    chunks_by_split = {
        split_name: _contiguous_chunks(getattr(splits, split_name))
        for split_name in ("train", "validation", "test")
    }
    return DocumentCurriculum(
        curriculum_id="contiguous-corpus-region",
        tasks=tuple(
            DocumentTask(
                task_id=f"corpus-region-{task_index}",
                splits=DocumentSplits(
                    train=chunks_by_split["train"][task_index],
                    validation=chunks_by_split["validation"][task_index],
                    test=chunks_by_split["test"][task_index],
                ),
            )
            for task_index in range(4)
        ),
    )


def stable_hash_task_index(document: TextDocument, task_count: int = 4) -> int:
    """Map one document to a task using only its stable content hash."""
    if not isinstance(document, TextDocument):
        raise TypeError("document must be a TextDocument")
    if type(task_count) is not int or task_count <= 0:
        raise ValueError("task_count must be a positive integer")
    return int(document.content_id, 16) % task_count


def build_stable_hash_curriculum(splits: DocumentSplits) -> DocumentCurriculum:
    """Build the four-way content-hash negative-control curriculum."""
    if not isinstance(splits, DocumentSplits):
        raise TypeError("splits must be DocumentSplits")
    assignments = {
        split_name: tuple(
            tuple(
                document
                for document in getattr(splits, split_name)
                if stable_hash_task_index(document) == task_index
            )
            for task_index in range(4)
        )
        for split_name in ("train", "validation", "test")
    }
    return DocumentCurriculum(
        curriculum_id="stable-hash-negative-control",
        tasks=tuple(
            DocumentTask(
                task_id=f"stable-hash-{task_index}",
                splits=DocumentSplits(
                    train=assignments["train"][task_index],
                    validation=assignments["validation"][task_index],
                    test=assignments["test"][task_index],
                ),
            )
            for task_index in range(4)
        ),
    )


def topic_scores(text: str) -> tuple[TopicScore, ...]:
    """Count distinct whole-word concepts for every TinyStories topic."""
    words = frozenset(_WORD_PATTERN.findall(normalize_text(text).casefold()))
    return tuple(
        TopicScore(
            topic=topic.name,
            matched_concepts=tuple(
                concept.name
                for concept in topic.concepts
                if words.intersection(concept.forms)
            ),
        )
        for topic in TINYSTORIES_TOPICS
    )


def classify_tinystory_topic(text: str) -> TopicAssignment | None:
    """Return the unique >=2-concept winner with at least a one-hit margin."""
    scores = topic_scores(text)
    ordered = sorted(
        scores,
        key=lambda score: score.distinct_concept_count,
        reverse=True,
    )
    winner, runner_up = ordered[:2]
    if (
        winner.distinct_concept_count < 2
        or winner.distinct_concept_count == runner_up.distinct_concept_count
    ):
        return None
    margin = winner.distinct_concept_count - runner_up.distinct_concept_count
    if margin < 1:
        return None
    return TopicAssignment(
        topic=winner.topic,
        matched_concepts=winner.matched_concepts,
        runner_up_score=runner_up.distinct_concept_count,
        margin=margin,
    )


def build_tinystories_topic_curriculum(
    splits: DocumentSplits,
    counts: StorySplitCounts,
) -> DocumentCurriculum:
    """Classify, hash-order, and exactly equalize four TinyStories topics."""
    if not isinstance(splits, DocumentSplits):
        raise TypeError("splits must be DocumentSplits")
    if not isinstance(counts, StorySplitCounts):
        raise TypeError("counts must be StorySplitCounts")
    selected_by_split: dict[str, dict[str, tuple[TextDocument, ...]]] = {}
    for split_name in ("train", "validation", "test"):
        buckets = {topic.name: [] for topic in TINYSTORIES_TOPICS}
        for document in getattr(splits, split_name):
            assignment = classify_tinystory_topic(document.text)
            if assignment is not None:
                buckets[assignment.topic].append(document)
        requested_count = getattr(counts, split_name)
        selected_by_split[split_name] = {}
        for topic in TINYSTORIES_TOPICS:
            ordered = tuple(
                sorted(
                    buckets[topic.name],
                    key=lambda document: document.content_id,
                )
            )
            if len(ordered) < requested_count:
                raise ValueError(
                    f"topic {topic.name!r} has {len(ordered)} {split_name} stories; "
                    f"requested exactly {requested_count}"
                )
            selected_by_split[split_name][topic.name] = ordered[:requested_count]
    return DocumentCurriculum(
        curriculum_id="tinystories-topic",
        tasks=tuple(
            DocumentTask(
                task_id=f"tinystories-topic-{topic.name}",
                splits=DocumentSplits(
                    train=selected_by_split["train"][topic.name],
                    validation=selected_by_split["validation"][topic.name],
                    test=selected_by_split["test"][topic.name],
                ),
            )
            for topic in TINYSTORIES_TOPICS
        ),
    )


def _contiguous_text_chunks(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("corpus text must be a string")
    if len(text) < 4:
        raise ValueError("corpus text must contain at least one character per task")
    base_size, larger_chunk_count = divmod(len(text), 4)
    chunk_sizes = tuple(
        base_size + int(index < larger_chunk_count) for index in range(4)
    )
    boundaries = np.cumsum((0, *chunk_sizes))
    return tuple(
        text[int(boundaries[index]) : int(boundaries[index + 1])]
        for index in range(4)
    )


def _contiguous_macro_documents(
    text: str,
    document_characters: int,
) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError("corpus text must be a string")
    if not text:
        raise ValueError("corpus text must not be empty")
    return tuple(
        text[start : start + document_characters]
        for start in range(0, len(text), document_characters)
    )


def _contiguous_chunks(
    documents: tuple[TextDocument, ...],
) -> tuple[tuple[TextDocument, ...], ...]:
    base_size, larger_chunk_count = divmod(len(documents), 4)
    chunk_sizes = tuple(
        base_size + int(index < larger_chunk_count)
        for index in range(4)
    )
    boundaries = np.cumsum((0, *chunk_sizes))
    return tuple(
        documents[int(boundaries[index]) : int(boundaries[index + 1])]
        for index in range(4)
    )


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


__all__ = [
    "CharacterPermutationCurriculum",
    "CharacterPermutationTask",
    "CorpusCurriculum",
    "CorpusDocumentSplits",
    "CorpusSplits",
    "CorpusTask",
    "DocumentCurriculum",
    "DocumentSplits",
    "DocumentTask",
    "EvaluationSpanPreset",
    "PinnedDatasetFile",
    "StorySplitCounts",
    "TINYSTORIES_DATASET_REVISION",
    "TINYSTORIES_DOCUMENT_SEPARATOR",
    "TINYSTORIES_EVALUATION_PRESET",
    "TINYSTORIES_SINGLE_GPU_PRESET",
    "TINYSTORIES_TOPICS",
    "TINYSTORIES_V2_SOURCE",
    "TINY_SHAKESPEARE_EVALUATION_PRESET",
    "TINY_SHAKESPEARE_EVALUATION_EXAMPLES_PER_TASK_AND_PREFIX",
    "TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS",
    "TextDocument",
    "TinyStoriesSingleGpuPreset",
    "TinyStoriesSourceContract",
    "TopicAssignment",
    "TopicConcept",
    "TopicDefinition",
    "TopicScore",
    "apply_ascii_letter_permutation",
    "apply_token_permutation",
    "ascii_letter_permutation",
    "build_contiguous_region_curriculum",
    "build_stable_hash_curriculum",
    "build_tiny_shakespeare_permutation_curriculum",
    "build_tiny_shakespeare_region_curriculum",
    "build_tiny_shakespeare_stable_hash_curriculum",
    "build_tinystories_topic_curriculum",
    "classify_tinystory_topic",
    "normalized_documents",
    "normalize_text",
    "parse_tinystories",
    "prepare_tinystories_splits",
    "raw_text_sha256",
    "load_pinned_dataset_text",
    "seeded_token_permutation",
    "sha256_content_id",
    "stable_hash_task_index",
    "stable_hash_raw_text_task_index",
    "topic_scores",
    "verify_pinned_dataset_file",
]
