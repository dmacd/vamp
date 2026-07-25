"""Fact-specific reverse-query review after primary semantic approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_TOKENIZER_IDENTITY,
    TokenizerIdentity,
)
from apm.data.text.tinyworlds_q_semantic.approval import PrimaryReviewApproval
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    SCHEMA_VERSION,
    canonical_json_bytes,
    record_sha256,
    require_identifier,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    PILOT_SHORTLIST_SPECS,
    SemanticReviewShortlist,
)
from apm.lm.text import TextTokenizer


REVERSE_REVIEW_FORMAT = "tinyworlds-q-semantic-reverse-review-v1"


@dataclass(frozen=True, slots=True)
class ReverseChoiceSpec:
    """One fact-specific reverse clue and three proposed false concepts."""

    proposal_id: str
    concept_id: str
    clue_prompt: str
    distractors: tuple[str, str, str]

    def __post_init__(self) -> None:
        require_identifier(self.proposal_id, "reverse proposal")
        require_identifier(self.concept_id, "reverse concept")
        if not self.clue_prompt.strip() or not self.clue_prompt.endswith("Answer:"):
            raise ValueError("reverse clue must be a visible answer prompt")
        if len(set((self.concept_id, *self.distractors))) != 4:
            raise ValueError("reverse choices must contain four distinct concepts")
        for distractor in self.distractors:
            require_identifier(distractor, "reverse distractor")

    def as_record(self) -> dict[str, object]:
        """Return the proposed reverse semantic decision."""
        return {
            "clue_prompt": self.clue_prompt,
            "concept_id": self.concept_id,
            "distractors": list(self.distractors),
            "proposal_id": self.proposal_id,
        }


@dataclass(frozen=True, slots=True)
class ReverseChoiceProposal:
    """One reverse proposal with exact pinned-tokenizer boundaries."""

    spec: ReverseChoiceSpec
    prompt_token_ids: tuple[int, ...]
    combined_candidate_token_ids: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ]

    def __post_init__(self) -> None:
        if type(self.spec) is not ReverseChoiceSpec:
            raise TypeError("reverse proposal specification is invalid")
        if type(self.prompt_token_ids) is not tuple or len(self.prompt_token_ids) < 2:
            raise ValueError("reverse proposal prompt tokenization is incomplete")
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
            raise ValueError("reverse candidates must have equal answer-token length")

    @property
    def answer_token_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return the four scored concept-token suffixes."""
        boundary = len(self.prompt_token_ids)
        return tuple(tokens[boundary:] for tokens in self.combined_candidate_token_ids)

    def as_record(self) -> dict[str, object]:
        """Return semantic choices and exact token boundaries."""
        return {
            **self.spec.as_record(),
            "answer_token_ids": [list(tokens) for tokens in self.answer_token_ids],
            "combined_candidate_token_ids": [
                list(tokens) for tokens in self.combined_candidate_token_ids
            ],
            "prompt_token_ids": list(self.prompt_token_ids),
        }


@dataclass(frozen=True, slots=True)
class SemanticReverseReview:
    """The complete unapproved reverse-choice queue for approved pilot facts."""

    shortlist_sha256: str
    primary_approval_sha256: str
    proposals: tuple[ReverseChoiceProposal, ...]
    tokenizer_identity: TokenizerIdentity = CANONICAL_TOKENIZER_IDENTITY
    reverse_review_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.shortlist_sha256, "reverse review shortlist")
        require_sha256(self.primary_approval_sha256, "reverse primary approval")
        if type(self.proposals) is not tuple or tuple(
            proposal.spec for proposal in self.proposals
        ) != PILOT_REVERSE_CHOICE_SPECS:
            raise ValueError("reverse review proposal order or content changed")
        if type(self.tokenizer_identity) is not TokenizerIdentity:
            raise TypeError("reverse review tokenizer identity is invalid")
        object.__setattr__(
            self,
            "reverse_review_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the source-bound reverse review queue."""
        record: dict[str, object] = {
            "benchmark_id": BENCHMARK_ID,
            "format": REVERSE_REVIEW_FORMAT,
            "primary_approval_sha256": self.primary_approval_sha256,
            "proposals": [proposal.as_record() for proposal in self.proposals],
            "schema_version": SCHEMA_VERSION,
            "shortlist_sha256": self.shortlist_sha256,
            "tokenizer_identity": self.tokenizer_identity.as_record(),
        }
        if include_hash:
            record["reverse_review_sha256"] = self.reverse_review_sha256
        return record


def _reverse_spec(
    proposal_id: str,
    concept_id: str,
    clue_prompt: str,
    distractors: tuple[str, str, str],
) -> ReverseChoiceSpec:
    return ReverseChoiceSpec(proposal_id, concept_id, clue_prompt, distractors)


PILOT_REVERSE_CHOICE_SPECS = tuple(
    _reverse_spec(*values)
    for values in (
        ("rabbit-proposal-01", "rabbit", "Which concept is an animal? Answer:", ("robot", "spoon", "rock")),
        ("rabbit-proposal-02", "rabbit", "Which concept has fur? Answer:", ("fish", "bird", "turtle")),
        ("rabbit-proposal-03", "rabbit", "Which concept has noticeable ears? Answer:", ("snake", "fish", "worm")),
        ("rabbit-proposal-04", "rabbit", "Which concept has a tail? Answer:", ("spider", "worm", "ant")),
        ("rabbit-proposal-05", "rabbit", "Which concept has paws? Answer:", ("fish", "snake", "worm")),
        ("rabbit-proposal-06", "rabbit", "Which concept commonly moves by hopping? Answer:", ("fish", "snake", "whale")),
        ("rabbit-proposal-07", "rabbit", "Which concept can move fast? Answer:", ("snail", "turtle", "worm")),
        ("rabbit-proposal-08", "rabbit", "Which concept is strongly associated with eating carrots? Answer:", ("lion", "shark", "eagle")),
        ("rabbit-proposal-09", "rabbit", "Which concept eats vegetables? Answer:", ("lion", "shark", "eagle")),
        ("rabbit-proposal-10", "rabbit", "Which concept can live in a burrow? Answer:", ("whale", "shark", "dolphin")),
        ("rabbit-proposal-11", "rabbit", "Which concept is generally small? Answer:", ("elephant", "whale", "camel")),
        ("rabbit-proposal-12", "rabbit", "Which concept commonly lives in a forest? Answer:", ("whale", "shark", "dolphin")),
        ("horse-proposal-01", "horse", "Which concept is an animal? Answer:", ("robot", "spoon", "rock")),
        ("horse-proposal-02", "horse", "Which concept has a mane? Answer:", ("fish", "bird", "turtle")),
        ("horse-proposal-03", "horse", "Which concept has a tail? Answer:", ("spider", "worm", "ant")),
        ("horse-proposal-04", "horse", "Which concept commonly runs? Answer:", ("fish", "snake", "whale")),
        ("horse-proposal-05", "horse", "Which concept can move fast? Answer:", ("snail", "turtle", "worm")),
        ("horse-proposal-06", "horse", "Which concept makes a neighing sound? Answer:", ("dog", "cat", "cow")),
        ("horse-proposal-07", "horse", "Which concept commonly eats hay? Answer:", ("lion", "shark", "eagle")),
        ("horse-proposal-08", "horse", "Which concept commonly eats grass? Answer:", ("shark", "eagle", "owl")),
        ("horse-proposal-09", "horse", "Which concept can live in a stable? Answer:", ("shark", "whale", "dolphin")),
        ("horse-proposal-10", "horse", "Which concept commonly lives on a farm? Answer:", ("whale", "shark", "dolphin")),
        ("horse-proposal-11", "horse", "Which concept is known for being strong? Answer:", ("worm", "snail", "feather")),
        ("horse-proposal-12", "horse", "Which concept is commonly ridden by people? Answer:", ("worm", "snail", "spider")),
    )
)


def build_pilot_reverse_review(
    shortlist: SemanticReviewShortlist,
    approval: PrimaryReviewApproval,
    tokenizer: TextTokenizer,
) -> SemanticReverseReview:
    """Compile fact-specific false concepts after primary fact approval."""
    if approval.shortlist_sha256 != shortlist.shortlist_sha256:
        raise ValueError("reverse review approval and shortlist do not match")
    if tokenizer.vocab_size != CANONICAL_TOKENIZER_IDENTITY.vocab_size:
        raise ValueError("reverse review requires the pinned tokenizer vocabulary")
    primary_specs = tuple(
        spec for spec in PILOT_SHORTLIST_SPECS if spec.priority == "primary"
    )
    if tuple((spec.proposal_id, spec.concept_id) for spec in primary_specs) != tuple(
        (spec.proposal_id, spec.concept_id) for spec in PILOT_REVERSE_CHOICE_SPECS
    ):
        raise ValueError("reverse review no longer covers every approved primary")
    return SemanticReverseReview(
        shortlist_sha256=shortlist.shortlist_sha256,
        primary_approval_sha256=approval.approval_sha256,
        proposals=tuple(
            _compile_reverse_proposal(spec, tokenizer)
            for spec in PILOT_REVERSE_CHOICE_SPECS
        ),
    )


def _compile_reverse_proposal(
    spec: ReverseChoiceSpec,
    tokenizer: TextTokenizer,
) -> ReverseChoiceProposal:
    prompt_tokens = tokenizer.encode(spec.clue_prompt)
    choices = (spec.concept_id, *spec.distractors)
    combined = tuple(
        tokenizer.encode(f"{spec.clue_prompt} {choice}") for choice in choices
    )
    return ReverseChoiceProposal(
        spec=spec,
        prompt_token_ids=prompt_tokens,
        combined_candidate_token_ids=combined,  # type: ignore[arg-type]
    )


def publish_reverse_review(
    review: SemanticReverseReview,
    output_root: str | Path,
) -> Path:
    """Atomically publish the concise reverse decision queue and exact JSON."""
    root = Path(output_root) / "reverse-reviews" / review.reverse_review_sha256
    review_payload = canonical_json_bytes(review.as_record())
    markdown_payload = render_reverse_review_markdown(review).encode("utf-8")
    content_payloads = {
        "reverse-review.json": review_payload,
        "review.md": markdown_payload,
    }
    payloads = {
        **content_payloads,
        "manifest.json": canonical_json_bytes(
            {
                "files": [
                    {
                        "name": name,
                        "sha256": sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for name, payload in sorted(content_payloads.items())
                ],
                "format": REVERSE_REVIEW_FORMAT,
                "primary_approval_sha256": review.primary_approval_sha256,
                "reverse_review_sha256": review.reverse_review_sha256,
                "schema_version": SCHEMA_VERSION,
                "shortlist_sha256": review.shortlist_sha256,
            }
        ),
    }
    if root.exists():
        if (
            {path.name for path in root.iterdir()} != set(payloads)
            or any(
                not (root / name).is_file()
                or (root / name).read_bytes() != payload
                for name, payload in payloads.items()
            )
        ):
            raise FileExistsError("existing reverse review changed")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reverse-review-", dir=root.parent))
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


def render_reverse_review_markdown(review: SemanticReverseReview) -> str:
    """Render one short row per approved fact's reverse query."""
    lines = [
        "# TinyWorlds-Q pilot reverse-choice approval sheet",
        "",
        "The 24 primary fact decisions are already recorded. This final semantic",
        "review checks that each reverse clue has one correct concept and three",
        "genuinely false concepts. All four choices have equal GPT-2 token length.",
        "",
        "Reply with **approve all reverse choices**, or list exceptions.",
        "",
        "| Approve | Fact | Reverse clue | Correct; false choices |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| [ ] | `{proposal.spec.proposal_id}` | {proposal.spec.clue_prompt} | "
        f"**{proposal.spec.concept_id}**; {', '.join(proposal.spec.distractors)} |"
        for proposal in review.proposals
    )
    lines.extend(
        (
            "",
            f"Reverse review: `{review.reverse_review_sha256}`  ",
            f"Primary approval: `{review.primary_approval_sha256}`",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "PILOT_REVERSE_CHOICE_SPECS",
    "ReverseChoiceProposal",
    "ReverseChoiceSpec",
    "SemanticReverseReview",
    "build_pilot_reverse_review",
    "publish_reverse_review",
    "render_reverse_review_markdown",
]
