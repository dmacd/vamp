"""Compact human-review surfaces derived from complete semantic evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import html
import os
from pathlib import Path
import re
import tempfile
from typing import Literal

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_TOKENIZER_IDENTITY,
    TokenizerIdentity,
)
from apm.data.text.tinyworlds_p.normalization import normalize_story_identity
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    SCHEMA_VERSION,
    StoryProvenance,
    canonical_json_bytes,
    record_sha256,
    require_identifier,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.manifests import PILOT_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.review import (
    PredicateDefinition,
    ReviewCandidate,
    SemanticReviewPacket,
)
from apm.lm.text import TextTokenizer


SHORTLIST_FORMAT = "tinyworlds-q-semantic-review-shortlist-v1"
REVIEW_SURFACE = "compact-primary-v1"
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_REVIEW_SAMPLE_COUNT = 3


@dataclass(frozen=True, slots=True)
class ReviewShortlistSpec:
    """One human-facing semantic proposal before any fact is approved."""

    proposal_id: str
    concept_id: str
    priority: Literal["primary", "backup"]
    relation_category: str
    proposed_fact: str
    forward_prompt: str
    answer_type: str
    canonical_answer: str
    accepted_forms: tuple[str, ...]
    trigger_forms: tuple[str, ...]
    source_predicate: str
    distractors: tuple[str, str, str]

    def __post_init__(self) -> None:
        for value, label in (
            (self.proposal_id, "shortlist proposal"),
            (self.concept_id, "shortlist concept"),
            (self.relation_category, "shortlist relation"),
            (self.answer_type, "shortlist answer type"),
        ):
            require_identifier(value, label)
        if self.priority not in ("primary", "backup"):
            raise ValueError("shortlist priority must be primary or backup")
        if not self.proposed_fact.strip() or not self.forward_prompt.strip():
            raise ValueError("shortlist fact and prompt must contain visible text")
        for values, label in (
            (self.accepted_forms, "accepted forms"),
            (self.trigger_forms, "trigger forms"),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"shortlist {label} must be nonempty and unique")
            if any(value != normalize_story_identity(value) for value in values):
                raise ValueError(f"shortlist {label} must be normalized")
        if self.canonical_answer != self.accepted_forms[0]:
            raise ValueError("shortlist canonical answer must lead accepted forms")
        if self.source_predicate not in self.trigger_forms:
            raise ValueError("shortlist source predicate must be a registered trigger")
        if (
            len(set((self.canonical_answer, *self.distractors))) != 4
            or any(
                value != normalize_story_identity(value)
                for value in (self.canonical_answer, *self.distractors)
            )
        ):
            raise ValueError("shortlist choices must be four unique normalized forms")

    def as_record(self) -> dict[str, object]:
        """Return the reviewable proposal without archive evidence."""
        return {
            "accepted_forms": list(self.accepted_forms),
            "answer_type": self.answer_type,
            "canonical_answer": self.canonical_answer,
            "concept_id": self.concept_id,
            "distractors": list(self.distractors),
            "forward_prompt": self.forward_prompt,
            "priority": self.priority,
            "proposal_id": self.proposal_id,
            "proposed_fact": self.proposed_fact,
            "relation_category": self.relation_category,
            "source_predicate": self.source_predicate,
            "trigger_forms": list(self.trigger_forms),
        }


@dataclass(frozen=True, slots=True)
class ReviewShortlistProposal:
    """One proposal bound to full evidence and exact answer tokenization."""

    spec: ReviewShortlistSpec
    source_candidate_id: str
    supporting_group_count: int
    source_candidate_sha256: str
    representative_evidence: tuple[StoryProvenance, ...]
    prompt_token_ids: tuple[int, ...]
    combined_candidate_token_ids: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]

    def __post_init__(self) -> None:
        if type(self.spec) is not ReviewShortlistSpec:
            raise TypeError("shortlist proposal requires a proposal specification")
        require_identifier(self.source_candidate_id, "shortlist source candidate")
        require_sha256(self.source_candidate_sha256, "shortlist source candidate")
        if type(self.supporting_group_count) is not int or self.supporting_group_count < 16:
            raise ValueError("shortlist proposals require sixteen construction groups")
        if (
            type(self.representative_evidence) is not tuple
            or len(self.representative_evidence) != _REVIEW_SAMPLE_COUNT
            or any(
                type(item) is not StoryProvenance
                for item in self.representative_evidence
            )
            or len({item.group_sha256 for item in self.representative_evidence})
            != _REVIEW_SAMPLE_COUNT
        ):
            raise ValueError("shortlist proposals require three distinct evidence groups")
        if type(self.prompt_token_ids) is not tuple or len(self.prompt_token_ids) < 2:
            raise ValueError("shortlist prompt tokenization is incomplete")
        combined = self.combined_candidate_token_ids
        if (
            type(combined) is not tuple
            or len(combined) != 4
            or any(
                tokens[: len(self.prompt_token_ids)] != self.prompt_token_ids
                for tokens in combined
            )
            or len(
                {len(tokens) - len(self.prompt_token_ids) for tokens in combined}
            )
            != 1
        ):
            raise ValueError("shortlist choices must have one equal answer-token length")

    @property
    def answer_token_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return only the four scored answer suffixes."""
        boundary = len(self.prompt_token_ids)
        return tuple(
            tokens[boundary:] for tokens in self.combined_candidate_token_ids
        )

    def as_record(self) -> dict[str, object]:
        """Return the compact proposal with representative evidence."""
        return {
            **self.spec.as_record(),
            "answer_token_ids": [list(tokens) for tokens in self.answer_token_ids],
            "combined_candidate_token_ids": [
                list(tokens) for tokens in self.combined_candidate_token_ids
            ],
            "prompt_token_ids": list(self.prompt_token_ids),
            "representative_evidence": [
                item.as_record() for item in self.representative_evidence
            ],
            "source_candidate_id": self.source_candidate_id,
            "source_candidate_sha256": self.source_candidate_sha256,
            "supporting_group_count": self.supporting_group_count,
        }


@dataclass(frozen=True, slots=True)
class ReverseReviewChoices:
    """Reviewed reverse-query choices for one pilot concept."""

    concept_id: str
    distractors: tuple[str, str, str]
    prompt_token_ids: tuple[int, ...]
    combined_candidate_token_ids: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]

    def __post_init__(self) -> None:
        require_identifier(self.concept_id, "reverse-choice concept")
        if len(set((self.concept_id, *self.distractors))) != 4:
            raise ValueError("reverse review choices must be unique")
        if (
            type(self.prompt_token_ids) is not tuple
            or len(self.prompt_token_ids) < 2
            or type(self.combined_candidate_token_ids) is not tuple
            or len(self.combined_candidate_token_ids) != 4
            or any(
                tokens[: len(self.prompt_token_ids)] != self.prompt_token_ids
                for tokens in self.combined_candidate_token_ids
            )
            or len(
                {
                    len(tokens) - len(self.prompt_token_ids)
                    for tokens in self.combined_candidate_token_ids
                }
            )
            != 1
        ):
            raise ValueError("reverse review choices must have equal token length")

    def as_record(self) -> dict[str, object]:
        """Return exact reverse-query answer boundaries."""
        return {
            "candidate_forms": [self.concept_id, *self.distractors],
            "combined_candidate_token_ids": [
                list(tokens) for tokens in self.combined_candidate_token_ids
            ],
            "concept_id": self.concept_id,
            "prompt_token_ids": list(self.prompt_token_ids),
        }


@dataclass(frozen=True, slots=True)
class SemanticReviewShortlist:
    """A compact decision surface that remains bound to the complete audit."""

    review_packet_sha256: str
    proposals: tuple[ReviewShortlistProposal, ...]
    reverse_choices: tuple[ReverseReviewChoices, ...]
    tokenizer_identity: TokenizerIdentity = CANONICAL_TOKENIZER_IDENTITY
    shortlist_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.review_packet_sha256, "shortlist review packet")
        if type(self.proposals) is not tuple or any(
            type(item) is not ReviewShortlistProposal for item in self.proposals
        ):
            raise TypeError("shortlist proposals must be immutable records")
        if tuple(item.spec for item in self.proposals) != PILOT_SHORTLIST_SPECS:
            raise ValueError("pilot shortlist proposal order or content changed")
        if type(self.reverse_choices) is not tuple or tuple(
            item.concept_id for item in self.reverse_choices
        ) != tuple(concept.concept_id for concept in PILOT_CONCEPTS):
            raise ValueError("pilot reverse choices must follow the concept manifest")
        if type(self.tokenizer_identity) is not TokenizerIdentity:
            raise TypeError("shortlist tokenizer identity is invalid")
        object.__setattr__(
            self,
            "shortlist_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the complete compact review record."""
        record: dict[str, object] = {
            "benchmark_id": BENCHMARK_ID,
            "format": SHORTLIST_FORMAT,
            "proposals": [item.as_record() for item in self.proposals],
            "review_surface": REVIEW_SURFACE,
            "review_packet_sha256": self.review_packet_sha256,
            "reverse_choices": [item.as_record() for item in self.reverse_choices],
            "schema_version": SCHEMA_VERSION,
            "tokenizer_identity": self.tokenizer_identity.as_record(),
        }
        if include_hash:
            record["shortlist_sha256"] = self.shortlist_sha256
        return record


def _proposal(
    proposal_id: str,
    concept_id: str,
    priority: Literal["primary", "backup"],
    relation_category: str,
    proposed_fact: str,
    forward_prompt: str,
    answer_type: str,
    canonical_answer: str,
    accepted_forms: tuple[str, ...],
    trigger_forms: tuple[str, ...],
    source_predicate: str,
    distractors: tuple[str, str, str],
) -> ReviewShortlistSpec:
    return ReviewShortlistSpec(
        proposal_id=proposal_id,
        concept_id=concept_id,
        priority=priority,
        relation_category=relation_category,
        proposed_fact=proposed_fact,
        forward_prompt=forward_prompt,
        answer_type=answer_type,
        canonical_answer=canonical_answer,
        accepted_forms=accepted_forms,
        trigger_forms=trigger_forms,
        source_predicate=source_predicate,
        distractors=distractors,
    )


PILOT_SHORTLIST_SPECS = tuple(
    _proposal(*values)
    for values in (
        ("rabbit-proposal-01", "rabbit", "primary", "taxonomy", "Rabbits are animals.", "A rabbit is what kind of living thing? Answer:", "category", "animal", ("animal", "animals"), ("animal", "animals"), "animal", ("plant", "insect", "object")),
        ("rabbit-proposal-02", "rabbit", "primary", "anatomy", "Rabbits have fur.", "What normally covers a rabbit's body? Answer:", "body-covering", "fur", ("fur", "furry"), ("fur", "furry"), "fur", ("scales", "feathers", "shell")),
        ("rabbit-proposal-03", "rabbit", "primary", "anatomy", "Rabbits have noticeable ears.", "Which body part is especially noticeable on a rabbit? Answer:", "body-part", "ears", ("ears", "ear"), ("ear", "ears", "long ears"), "ears", ("horns", "wings", "fins")),
        ("rabbit-proposal-04", "rabbit", "primary", "anatomy", "Rabbits have tails.", "Which rear body part does a rabbit have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("rabbit-proposal-05", "rabbit", "primary", "anatomy", "Rabbits have paws.", "What are a rabbit's feet commonly called? Answer:", "body-part", "paws", ("paws", "paw"), ("paw", "paws"), "paw", ("claws", "wings", "fins")),
        ("rabbit-proposal-06", "rabbit", "primary", "locomotion", "Rabbits move by hopping.", "How does a rabbit commonly move? Answer:", "action", "hops", ("hops", "hop", "hopped", "hopping"), ("hop", "hops", "hopped", "hopping"), "hopped", ("flies", "sails", "drives")),
        ("rabbit-proposal-07", "rabbit", "primary", "locomotion", "Rabbits can move fast.", "How can a rabbit move? Answer:", "manner", "fast", ("fast", "quick", "quickly"), ("fast", "quick", "quickly"), "fast", ("slow", "heavy", "quiet")),
        ("rabbit-proposal-08", "rabbit", "primary", "diet", "Rabbits eat carrots.", "Which food is strongly associated with rabbits? Answer:", "food", "carrots", ("carrots", "carrot"), ("carrot", "carrots"), "carrots", ("meat", "fish", "insects")),
        ("rabbit-proposal-09", "rabbit", "primary", "diet", "Rabbits eat vegetables.", "Which food group do rabbits eat? Answer:", "food", "vegetables", ("vegetables", "vegetable"), ("vegetable", "vegetables"), "vegetables", ("meat", "fish", "insects")),
        ("rabbit-proposal-10", "rabbit", "primary", "habitat", "Rabbits can live in burrows.", "What kind of shelter can a rabbit live in? Answer:", "shelter", "burrow", ("burrow", "burrows"), ("burrow", "burrows"), "burrow", ("anthill", "coop", "roost")),
        ("rabbit-proposal-11", "rabbit", "primary", "appearance", "Rabbits are generally small.", "What size are rabbits generally? Answer:", "size", "small", ("small", "little", "tiny"), ("small", "little", "tiny"), "small", ("large", "tall", "huge")),
        ("rabbit-proposal-12", "rabbit", "primary", "habitat", "Rabbits commonly live in forests.", "Which habitat commonly contains rabbits? Answer:", "habitat", "forest", ("forest", "forests"), ("forest", "forests"), "forest", ("desert", "ocean", "city")),
        ("rabbit-proposal-13", "rabbit", "backup", "locomotion", "Rabbits can jump.", "Which movement can a rabbit perform? Answer:", "action", "jumps", ("jumps", "jump", "jumped", "jumping"), ("jump", "jumps", "jumped", "jumping"), "jumped", ("flies", "sails", "drives")),
        ("rabbit-proposal-14", "rabbit", "backup", "anatomy", "Rabbits have legs.", "Which limbs does a rabbit use to move? Answer:", "body-part", "legs", ("legs", "leg"), ("leg", "legs", "four legs"), "legs", ("wings", "fins", "horns")),
        ("rabbit-proposal-15", "rabbit", "backup", "appearance", "Rabbit fur is soft.", "How can a rabbit's fur feel? Answer:", "texture", "soft", ("soft",), ("soft",), "soft", ("hard", "loud", "sharp")),
        ("rabbit-proposal-16", "rabbit", "backup", "habitat", "Rabbits are found in fields.", "Which open habitat can contain rabbits? Answer:", "habitat", "field", ("field", "fields"), ("field", "fields"), "field", ("ocean", "desert", "city")),
        ("horse-proposal-01", "horse", "primary", "taxonomy", "Horses are animals.", "A horse is what kind of living thing? Answer:", "category", "animal", ("animal", "animals"), ("animal", "animals"), "animals", ("plant", "insect", "object")),
        ("horse-proposal-02", "horse", "primary", "anatomy", "Horses have manes.", "Which body feature grows along a horse's neck? Answer:", "body-part", "mane", ("mane", "manes"), ("mane", "manes"), "mane", ("beak", "antlers", "tentacle")),
        ("horse-proposal-03", "horse", "primary", "anatomy", "Horses have tails.", "Which rear body part does a horse have? Answer:", "body-part", "tail", ("tail", "tails"), ("tail", "tails"), "tail", ("trunk", "horns", "shell")),
        ("horse-proposal-04", "horse", "primary", "locomotion", "Horses run.", "How does a horse commonly move quickly? Answer:", "action", "runs", ("runs", "run", "ran", "running"), ("run", "runs", "ran", "running"), "run", ("flies", "sails", "drives")),
        ("horse-proposal-05", "horse", "primary", "locomotion", "Horses can move fast.", "How can a horse move? Answer:", "manner", "fast", ("fast", "quick", "quickly"), ("fast", "quick", "quickly"), "fast", ("slow", "heavy", "quiet")),
        ("horse-proposal-06", "horse", "primary", "vocalization", "Horses neigh.", "Which sound does a horse make? Answer:", "sound", "neigh", ("neigh", "neighs", "neighed"), ("neigh", "neighs", "neighed", "whinny", "whinnies", "whinnied"), "neigh", ("bark", "roar", "buzz")),
        ("horse-proposal-07", "horse", "primary", "diet", "Horses eat hay.", "Which food do horses commonly eat? Answer:", "food", "hay", ("hay",), ("hay",), "hay", ("meat", "fish", "insects")),
        ("horse-proposal-08", "horse", "primary", "diet", "Horses eat grass.", "Which plant do horses commonly graze on? Answer:", "food", "grass", ("grass",), ("grass",), "grass", ("meat", "fish", "insects")),
        ("horse-proposal-09", "horse", "primary", "habitat", "Horses can live in stables.", "What kind of shelter can a horse live in? Answer:", "shelter", "stable", ("stable", "stables"), ("stable", "stables"), "stable", ("nest", "cave", "ocean")),
        ("horse-proposal-10", "horse", "primary", "habitat", "Horses commonly live on farms.", "Which place commonly keeps horses? Answer:", "habitat", "farm", ("farm", "farms"), ("farm", "farms"), "farm", ("city", "ocean", "desert")),
        ("horse-proposal-11", "horse", "primary", "appearance", "Horses are strong.", "Which physical trait commonly describes a horse? Answer:", "trait", "strong", ("strong",), ("strong",), "strong", ("weak", "tiny", "quiet")),
        ("horse-proposal-12", "horse", "primary", "human-interaction", "People ride trained horses.", "What can a person do on a trained horse? Answer:", "action", "ride", ("ride", "riding", "rode", "ridden"), ("ride", "rides", "rode", "ridden", "riding"), "ride", ("paint", "bake", "type")),
        ("horse-proposal-13", "horse", "backup", "appearance", "Horses are generally big.", "What size are horses generally? Answer:", "size", "big", ("big", "large"), ("big", "large"), "big", ("tiny", "short", "weak")),
        ("horse-proposal-14", "horse", "backup", "habitat", "Horses are found in fields.", "Which open habitat can contain horses? Answer:", "habitat", "field", ("field", "fields"), ("field", "fields"), "field", ("ocean", "desert", "city")),
        ("horse-proposal-15", "horse", "backup", "habitat", "Horses can live in barns.", "Which farm building can house horses? Answer:", "shelter", "barn", ("barn", "barns"), ("barn", "barns"), "barn", ("city", "ocean", "desert")),
        ("horse-proposal-16", "horse", "backup", "diet", "Horses eat apples.", "Which fruit can horses eat? Answer:", "food", "apples", ("apples", "apple"), ("apple", "apples"), "apples", ("meat", "fish", "insects")),
    )
)


def pilot_shortlist_predicates() -> tuple[PredicateDefinition, ...]:
    """Return the small exact-predicate set needed by the pilot shortlist."""
    predicate_categories = {
        (spec.source_predicate, spec.relation_category)
        for spec in PILOT_SHORTLIST_SPECS
    }
    categories_by_predicate = {
        predicate: {
            category
            for candidate_predicate, category in predicate_categories
            if candidate_predicate == predicate
        }
        for predicate, _ in predicate_categories
    }
    if any(len(categories) != 1 for categories in categories_by_predicate.values()):
        raise ValueError("one shortlist predicate was assigned conflicting categories")
    return tuple(
        PredicateDefinition(predicate, next(iter(categories)))
        for predicate, categories in sorted(categories_by_predicate.items())
    )


def build_pilot_review_shortlist(
    packet: SemanticReviewPacket,
    tokenizer: TextTokenizer,
) -> SemanticReviewShortlist:
    """Compile the supported pilot proposals into one compact decision surface."""
    if packet.concepts != PILOT_CONCEPTS:
        raise ValueError("pilot shortlist requires the exact pilot concept manifest")
    expected_predicates = pilot_shortlist_predicates()
    if packet.predicates != expected_predicates:
        raise ValueError("pilot shortlist review packet changed its targeted predicates")
    if tokenizer.vocab_size != CANONICAL_TOKENIZER_IDENTITY.vocab_size:
        raise ValueError("pilot shortlist requires the pinned tokenizer vocabulary")
    candidates = {
        (candidate.concept_id, candidate.predicate): candidate
        for candidate in packet.candidates
    }
    proposals = tuple(
        _compile_proposal(spec, candidates.get((spec.concept_id, spec.source_predicate)), tokenizer)
        for spec in PILOT_SHORTLIST_SPECS
    )
    reverse_prompt = "Which concept does this fact describe? Answer:"
    reverse_distractors = {
        "rabbit": ("horse", "cat", "dog"),
        "horse": ("rabbit", "cat", "dog"),
    }
    reverse_choices = tuple(
        ReverseReviewChoices(
            concept_id=concept.concept_id,
            distractors=reverse_distractors[concept.concept_id],
            prompt_token_ids=tokenizations[0],
            combined_candidate_token_ids=tokenizations[1],
        )
        for concept in PILOT_CONCEPTS
        for tokenizations in (
            _tokenize_choices(
                reverse_prompt,
                (concept.concept_id, *reverse_distractors[concept.concept_id]),
                tokenizer,
            ),
        )
    )
    return SemanticReviewShortlist(
        review_packet_sha256=packet.packet_sha256,
        proposals=proposals,
        reverse_choices=reverse_choices,
    )


def _compile_proposal(
    spec: ReviewShortlistSpec,
    candidate: ReviewCandidate | None,
    tokenizer: TextTokenizer,
) -> ReviewShortlistProposal:
    if candidate is None:
        raise ValueError(f"shortlist source evidence is missing: {spec.proposal_id}")
    if candidate.relation_category != spec.relation_category:
        raise ValueError(f"shortlist source relation changed: {spec.proposal_id}")
    prompt_tokens, combined = _tokenize_choices(
        spec.forward_prompt,
        (spec.canonical_answer, *spec.distractors),
        tokenizer,
    )
    return ReviewShortlistProposal(
        spec=spec,
        source_candidate_id=candidate.candidate_id,
        supporting_group_count=len(candidate.supporting_story_groups),
        source_candidate_sha256=record_sha256(candidate.as_record()),
        representative_evidence=select_representative_evidence(candidate, spec),
        prompt_token_ids=prompt_tokens,
        combined_candidate_token_ids=combined,
    )


def select_representative_evidence(
    candidate: ReviewCandidate,
    spec: ReviewShortlistSpec,
) -> tuple[StoryProvenance, ...]:
    """Choose three short, close co-occurrences from distinct source groups."""
    ranked = sorted(
        candidate.evidence,
        key=lambda evidence: _evidence_rank(evidence, spec),
    )
    selected: list[StoryProvenance] = []
    seen_groups: set[str] = set()
    seen_sentences: set[str] = set()
    for evidence in ranked:
        normalized_sentence = normalize_story_identity(evidence.sentence_text)
        if (
            evidence.group_sha256 in seen_groups
            or normalized_sentence in seen_sentences
        ):
            continue
        selected.append(evidence)
        seen_groups.add(evidence.group_sha256)
        seen_sentences.add(normalized_sentence)
        if len(selected) == _REVIEW_SAMPLE_COUNT:
            return tuple(selected)
    raise ValueError(f"too few distinct evidence samples for {spec.proposal_id}")


def _evidence_rank(
    evidence: StoryProvenance,
    spec: ReviewShortlistSpec,
) -> tuple[int, int, int, str, str]:
    normalized_sentence = normalize_story_identity(evidence.sentence_text)
    words = tuple(_WORD.findall(normalized_sentence))
    concept_surfaces = next(
        concept.surface_forms
        for concept in PILOT_CONCEPTS
        if concept.concept_id == spec.concept_id
    )
    concept_positions = tuple(
        match.start()
        for surface in concept_surfaces
        for match in re.finditer(
            rf"(?<!\w){re.escape(surface)}(?!\w)",
            normalized_sentence,
        )
    )
    predicate_positions = tuple(
        match.start()
        for match in re.finditer(
            rf"(?<!\w){re.escape(spec.source_predicate)}(?!\w)",
            normalized_sentence,
        )
    )
    if not concept_positions or not predicate_positions:
        raise ValueError(
            f"review evidence lost its exact co-occurrence: {spec.proposal_id}"
        )
    distance = min(
        abs(concept_position - predicate_position)
        for concept_position in concept_positions
        for predicate_position in predicate_positions
    )
    dialogue_penalty = int("\"" in evidence.sentence_text or "\n" in evidence.sentence_text)
    return (
        dialogue_penalty,
        distance,
        len(words),
        normalized_sentence,
        evidence.group_sha256,
    )


def _tokenize_choices(
    prompt: str,
    choices: tuple[str, str, str, str],
    tokenizer: TextTokenizer,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]:
    prompt_tokens = tokenizer.encode(prompt)
    combined = tuple(tokenizer.encode(f"{prompt} {choice}") for choice in choices)
    if any(tokens[: len(prompt_tokens)] != prompt_tokens for tokens in combined):
        raise ValueError("shortlist answer changed the tokenizer prompt boundary")
    if len({len(tokens) - len(prompt_tokens) for tokens in combined}) != 1:
        raise ValueError(f"shortlist choices do not have equal token length: {choices}")
    return prompt_tokens, combined  # type: ignore[return-value]


def publish_review_shortlist(
    shortlist: SemanticReviewShortlist,
    output_root: str | Path,
) -> Path:
    """Atomically publish compact JSON, Markdown, HTML, and an editable TSV."""
    root = Path(output_root) / "review-shortlists" / shortlist.shortlist_sha256
    payloads = _shortlist_payloads(shortlist)
    if root.exists():
        if (
            {path.name for path in root.iterdir()} != set(payloads)
            or any(
                not (root / name).is_file()
                or (root / name).read_bytes() != payload
                for name, payload in payloads.items()
            )
        ):
            raise FileExistsError("existing compact review shortlist changed")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".shortlist-", dir=root.parent))
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        os.replace(staging, root)
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if staging.exists():
            staging.rmdir()
        raise
    return root


def _shortlist_payloads(shortlist: SemanticReviewShortlist) -> dict[str, bytes]:
    markdown = render_review_shortlist_markdown(shortlist)
    content_payloads = {
        "review.md": render_primary_review_markdown(shortlist).encode("utf-8"),
        "review-form.tsv": render_review_form_tsv(shortlist).encode("utf-8"),
        "shortlist.html": render_review_shortlist_html(shortlist, markdown).encode("utf-8"),
        "shortlist.json": canonical_json_bytes(shortlist.as_record()),
        "shortlist.md": markdown.encode("utf-8"),
    }
    manifest = canonical_json_bytes(
        {
            "files": [
                {
                    "name": name,
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(content_payloads.items())
            ],
            "format": SHORTLIST_FORMAT,
            "review_packet_sha256": shortlist.review_packet_sha256,
            "schema_version": SCHEMA_VERSION,
            "shortlist_sha256": shortlist.shortlist_sha256,
        }
    )
    return {**content_payloads, "manifest.json": manifest}


def render_review_shortlist_markdown(shortlist: SemanticReviewShortlist) -> str:
    """Render detailed proposal evidence and exact token boundaries."""
    lines = [
        "# TinyWorlds-Q pilot semantic shortlist details",
        "",
        "The concise decision sheet is `review.md`. Use this detailed file only to ",
        "inspect additional evidence and tokenizer boundaries. The complete audit is ",
        f"available only for provenance at `../../review/{shortlist.review_packet_sha256}/review.md`.",
        "",
        "Approve the twelve primary proposals for each concept when all five gates ",
        "are sound. If you reject a primary proposal, promote a reviewed backup. ",
        "Reply with proposal IDs and any edits; no catalog is published automatically.",
        "",
        f"Shortlist: `{shortlist.shortlist_sha256}`  ",
        f"Evidence packet: `{shortlist.review_packet_sha256}`",
        "",
        "## Decision overview",
        "",
        "| ID | Tier | Proposed fact | Category | Groups | Decision |",
        "|---|---|---|---|---:|---|",
    ]
    lines.extend(
        f"| `{proposal.spec.proposal_id}` | {proposal.spec.priority} | "
        f"{proposal.spec.proposed_fact} | `{proposal.spec.relation_category}` | "
        f"{proposal.supporting_group_count} | approve / reject / edit |"
        for proposal in shortlist.proposals
    )
    lines.extend(("", "## Detailed review", ""))
    for proposal in shortlist.proposals:
        spec = proposal.spec
        choices = (spec.canonical_answer, *spec.distractors)
        lines.extend(
            (
                f"### {spec.proposal_id} — {spec.proposed_fact}",
                "",
                f"Tier: **{spec.priority}**  ",
                f"Category: `{spec.relation_category}`  ",
                f"Forward prompt: {spec.forward_prompt}  ",
                "Choices: "
                + " | ".join(
                    f"**{choice}** (correct)" if index == 0 else choice
                    for index, choice in enumerate(choices)
                )
                + "  ",
                f"Answer type: `{spec.answer_type}`  ",
                f"Accepted forms: `{', '.join(spec.accepted_forms)}`  ",
                f"Trigger closure: `{', '.join(spec.trigger_forms)}`  ",
                "Answer-token suffixes: "
                + "; ".join(
                    f"`{choice}`={list(tokens)}"
                    for choice, tokens in zip(choices, proposal.answer_token_ids)
                ),
                "",
                f"Evidence: `{proposal.source_candidate_id}`; exact predicate "
                f"`{spec.source_predicate}`; {proposal.supporting_group_count} "
                "distinct construction groups. Representative sentences:",
                "",
            )
        )
        lines.extend(
            f"- `{evidence.group_sha256}` — {evidence.sentence_text} "
            f"(`{evidence.source_member}` record {evidence.source_index})"
            for evidence in proposal.representative_evidence
        )
        lines.extend(
            (
                "",
                "Review gates: truth [ ] answer forms [ ] trigger closure [ ] "
                "distractors [ ] evidence [ ]",
                "",
                "Decision: approve [ ] reject [ ] edit [ ]  Notes:",
                "",
            )
        )
    lines.extend(("## Reverse-query choices", ""))
    lines.extend(
        f"- `{item.concept_id}`: correct `{item.concept_id}`; distractors "
        f"`{', '.join(item.distractors)}`"
        for item in shortlist.reverse_choices
    )
    lines.extend(
        (
            "",
            "Reverse choices review: grammatical type [ ] false distractors [ ] "
            "equal token lengths [ ]",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_primary_review_markdown(shortlist: SemanticReviewShortlist) -> str:
    """Render a one-page primary decision queue with backups kept separate."""
    lines = [
        "# TinyWorlds-Q rabbit/horse approval sheet",
        "",
        "This is the review queue. You do not need to read the complete evidence audit.",
        "Approving a row means its truth, answer forms, trigger closure, distractors,",
        "and representative evidence are all acceptable. Use `shortlist.md` only when",
        "you want two more examples or exact token IDs.",
        "",
        "Reply with **approve all primaries**, or list exceptions such as",
        "`reject rabbit-proposal-12; promote rabbit-proposal-13`.",
        "",
    ]
    for concept in PILOT_CONCEPTS:
        concept_proposals = tuple(
            proposal
            for proposal in shortlist.proposals
            if proposal.spec.concept_id == concept.concept_id
            and proposal.spec.priority == "primary"
        )
        lines.extend(
            (
                f"## {concept.concept_id.title()} — 12 primary proposals",
                "",
                "| Approve | ID | Fact | Answer; false choices | Triggers | Groups | Example evidence |",
                "|---|---|---|---|---|---:|---|",
            )
        )
        lines.extend(_primary_review_row(proposal) for proposal in concept_proposals)
        lines.append("")
    lines.extend(
        (
            "## Backups",
            "",
            "Only review these if you reject a primary proposal.",
            "",
            "| ID | Fact | Category | Groups |",
            "|---|---|---|---:|",
        )
    )
    lines.extend(
        f"| `{proposal.spec.proposal_id}` | {proposal.spec.proposed_fact} | "
        f"`{proposal.spec.relation_category}` | {proposal.supporting_group_count} |"
        for proposal in shortlist.proposals
        if proposal.spec.priority == "backup"
    )
    lines.extend(("", "## Shared reverse-query choices", ""))
    lines.extend(
        f"- `{item.concept_id}` is correct; false choices: "
        f"`{', '.join(item.distractors)}`. Approve [ ]"
        for item in shortlist.reverse_choices
    )
    lines.extend(
        (
            "",
            f"Shortlist: `{shortlist.shortlist_sha256}`  ",
            f"Evidence packet: `{shortlist.review_packet_sha256}`",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _primary_review_row(proposal: ReviewShortlistProposal) -> str:
    spec = proposal.spec
    example = proposal.representative_evidence[0].sentence_text.replace("\n", " ")
    cells = (
        "[ ]",
        f"`{spec.proposal_id}`",
        spec.proposed_fact,
        f"**{spec.canonical_answer}**; {', '.join(spec.distractors)}",
        f"`{', '.join(spec.trigger_forms)}`",
        str(proposal.supporting_group_count),
        example,
    )
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def render_review_form_tsv(shortlist: SemanticReviewShortlist) -> str:
    """Render a compact spreadsheet-friendly decision form."""
    header = (
        "proposal_id\tconcept\ttier\tproposed_fact\tgroups\ttruth\tanswer_forms\t"
        "trigger_closure\tdistractors\tevidence\tdecision\tedits_or_notes"
    )
    rows = tuple(
        "\t".join(
            (
                proposal.spec.proposal_id,
                proposal.spec.concept_id,
                proposal.spec.priority,
                proposal.spec.proposed_fact,
                str(proposal.supporting_group_count),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )
        )
        for proposal in shortlist.proposals
    )
    return "\n".join((header, *rows)) + "\n"


def render_review_shortlist_html(
    shortlist: SemanticReviewShortlist,
    markdown: str,
) -> str:
    """Render a standalone compact HTML surface without remote dependencies."""
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q pilot semantic shortlist</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1050px;margin:2rem auto;"
        "padding:0 1rem;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f7f8fa;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-shortlist-sha256=\"{shortlist.shortlist_sha256}\">"
        f"<pre>{html.escape(markdown)}</pre></body></html>\n"
    )


__all__ = [
    "PILOT_SHORTLIST_SPECS",
    "ReviewShortlistProposal",
    "ReviewShortlistSpec",
    "SemanticReviewShortlist",
    "build_pilot_review_shortlist",
    "pilot_shortlist_predicates",
    "publish_review_shortlist",
    "render_review_form_tsv",
    "render_primary_review_markdown",
    "render_review_shortlist_html",
    "render_review_shortlist_markdown",
    "select_representative_evidence",
]
