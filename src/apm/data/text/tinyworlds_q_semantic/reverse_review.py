"""Fact-specific reverse-query review after primary semantic approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
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


@dataclass(frozen=True, slots=True)
class ReverseReviewApproval:
    """One explicit human approval of every fact-specific reverse choice."""

    reverse_review_sha256: str
    reviewer: str
    reviewed_at: str
    approved_proposal_ids: tuple[str, ...]
    approval_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.reverse_review_sha256, "reverse approval review")
        if any(
            type(value) is not str or not value.strip()
            for value in (self.reviewer, self.reviewed_at)
        ):
            raise ValueError("reverse approval reviewer and time are required")
        expected = tuple(spec.proposal_id for spec in PILOT_REVERSE_CHOICE_SPECS)
        if self.approved_proposal_ids != expected:
            raise ValueError("reverse approval must name every proposal in order")
        object.__setattr__(
            self,
            "approval_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return the explicit affirmative reverse decisions."""
        record: dict[str, object] = {
            "benchmark_id": BENCHMARK_ID,
            "decisions": [
                {
                    "false_distractors_approved": True,
                    "grammatical_type_approved": True,
                    "proposal_id": proposal_id,
                    "token_lengths_approved": True,
                }
                for proposal_id in self.approved_proposal_ids
            ],
            "format": "tinyworlds-q-semantic-reverse-approval-v1",
            "reverse_review_sha256": self.reverse_review_sha256,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "schema_version": SCHEMA_VERSION,
        }
        if include_hash:
            record["approval_sha256"] = self.approval_sha256
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


def approve_all_reverse_choices(
    review: SemanticReverseReview,
    *,
    reviewer: str,
    reviewed_at: str,
) -> ReverseReviewApproval:
    """Translate an explicit approve-all instruction into a bound record."""
    return ReverseReviewApproval(
        reverse_review_sha256=review.reverse_review_sha256,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        approved_proposal_ids=tuple(
            proposal.spec.proposal_id for proposal in review.proposals
        ),
    )


def publish_reverse_review_approval(
    approval: ReverseReviewApproval,
    output_root: str | Path,
) -> Path:
    """Atomically publish the human reverse-choice approval."""
    root = Path(output_root) / "reverse-approvals" / approval.approval_sha256
    approval_payload = canonical_json_bytes(approval.as_record())
    summary_payload = (
        "# TinyWorlds-Q reverse-choice approval\n\n"
        f"Approval: `{approval.approval_sha256}`  \n"
        f"Reverse review: `{approval.reverse_review_sha256}`  \n"
        f"Reviewer: `{approval.reviewer}`  \n"
        f"Reviewed at: `{approval.reviewed_at}`\n\n"
        f"All reverse-choice gates are affirmative for all "
        f"{len(approval.approved_proposal_ids)} approved facts.\n"
    ).encode("utf-8")
    content_payloads = {
        "approval.json": approval_payload,
        "approval.md": summary_payload,
    }
    payloads = {
        **content_payloads,
        "manifest.json": canonical_json_bytes(
            {
                "approval_sha256": approval.approval_sha256,
                "files": [
                    {
                        "name": name,
                        "sha256": sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for name, payload in sorted(content_payloads.items())
                ],
                "format": "tinyworlds-q-semantic-reverse-approval-v1",
                "reverse_review_sha256": approval.reverse_review_sha256,
                "schema_version": SCHEMA_VERSION,
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
            raise FileExistsError("existing reverse approval changed")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reverse-approval-", dir=root.parent))
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


def load_reverse_review_approval(directory: str | Path) -> ReverseReviewApproval:
    """Strictly authenticate and reconstruct one reverse approval."""
    root = Path(directory)
    manifest = _load_canonical_json(root / "manifest.json")
    if (
        manifest.get("format") != "tinyworlds-q-semantic-reverse-approval-v1"
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("reverse approval manifest identity changed")
    file_records = manifest.get("files")
    if type(file_records) is not list or any(type(item) is not dict for item in file_records):
        raise ValueError("reverse approval manifest files are invalid")
    expected_names = {"manifest.json", *(str(item.get("name")) for item in file_records)}
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("reverse approval tree entries changed")
    for item in file_records:
        name = item.get("name")
        size_bytes = item.get("size_bytes")
        expected_sha256 = item.get("sha256")
        if type(name) is not str or type(size_bytes) is not int or type(expected_sha256) is not str:
            raise ValueError("reverse approval file identity is invalid")
        payload = (root / name).read_bytes()
        if len(payload) != size_bytes or sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("reverse approval file changed")
    record = _load_canonical_json(root / "approval.json")
    decisions = record.get("decisions")
    if type(decisions) is not list or any(type(item) is not dict for item in decisions):
        raise ValueError("reverse approval decisions are invalid")
    gates = (
        "false_distractors_approved",
        "grammatical_type_approved",
        "token_lengths_approved",
    )
    if any(any(item.get(gate) is not True for gate in gates) for item in decisions):
        raise ValueError("reverse approval contains a non-affirmative gate")
    approval = ReverseReviewApproval(
        reverse_review_sha256=_text(record, "reverse_review_sha256"),
        reviewer=_text(record, "reviewer"),
        reviewed_at=_text(record, "reviewed_at"),
        approved_proposal_ids=tuple(_text(item, "proposal_id") for item in decisions),
    )
    expected_summary = (
        "# TinyWorlds-Q reverse-choice approval\n\n"
        f"Approval: `{approval.approval_sha256}`  \n"
        f"Reverse review: `{approval.reverse_review_sha256}`  \n"
        f"Reviewer: `{approval.reviewer}`  \n"
        f"Reviewed at: `{approval.reviewed_at}`\n\n"
        f"All reverse-choice gates are affirmative for all "
        f"{len(approval.approved_proposal_ids)} approved facts.\n"
    )
    if (
        record != approval.as_record()
        or manifest.get("approval_sha256") != approval.approval_sha256
        or manifest.get("reverse_review_sha256") != approval.reverse_review_sha256
        or root.name != approval.approval_sha256
        or (root / "approval.md").read_text(encoding="utf-8") != expected_summary
    ):
        raise ValueError("reverse approval semantic content changed")
    return approval


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid reverse approval JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"reverse approval JSON is not canonical: {path.name}")
    return value


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"reverse approval {field} must be nonempty text")
    return value


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
    "ReverseReviewApproval",
    "ReverseChoiceProposal",
    "ReverseChoiceSpec",
    "SemanticReverseReview",
    "build_pilot_reverse_review",
    "approve_all_reverse_choices",
    "load_reverse_review_approval",
    "publish_reverse_review",
    "publish_reverse_review_approval",
    "render_reverse_review_markdown",
]
