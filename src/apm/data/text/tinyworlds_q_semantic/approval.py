"""Durable human approval records for compact semantic review queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    SCHEMA_VERSION,
    canonical_json_bytes,
    record_sha256,
    require_identifier,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    SemanticReviewShortlist,
)


APPROVAL_FORMAT = "tinyworlds-q-semantic-primary-approval-v1"


@dataclass(frozen=True, slots=True)
class PrimaryReviewApproval:
    """One explicit human approval of every primary shortlist gate."""

    shortlist_sha256: str
    reviewer: str
    reviewed_at: str
    approved_proposal_ids: tuple[str, ...]
    approval_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.shortlist_sha256, "primary approval shortlist")
        if any(
            type(value) is not str or not value.strip()
            for value in (self.reviewer, self.reviewed_at)
        ):
            raise ValueError("primary approval reviewer and time are required")
        if (
            type(self.approved_proposal_ids) is not tuple
            or not self.approved_proposal_ids
            or len(self.approved_proposal_ids) % 12 != 0
            or len(set(self.approved_proposal_ids))
            != len(self.approved_proposal_ids)
        ):
            raise ValueError(
                "primary approval requires complete unique twelve-fact worlds"
            )
        for proposal_id in self.approved_proposal_ids:
            require_identifier(proposal_id, "approved primary proposal")
        object.__setattr__(
            self,
            "approval_sha256",
            record_sha256(self.as_record(include_hash=False)),
        )

    def as_record(self, *, include_hash: bool = True) -> dict[str, object]:
        """Return all five affirmative gates for every approved proposal."""
        record: dict[str, object] = {
            "benchmark_id": BENCHMARK_ID,
            "decisions": [
                {
                    "answer_forms_approved": True,
                    "distractors_approved": True,
                    "evidence_approved": True,
                    "proposal_id": proposal_id,
                    "trigger_closure_approved": True,
                    "truth_approved": True,
                }
                for proposal_id in self.approved_proposal_ids
            ],
            "format": APPROVAL_FORMAT,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "schema_version": SCHEMA_VERSION,
            "shortlist_sha256": self.shortlist_sha256,
        }
        if include_hash:
            record["approval_sha256"] = self.approval_sha256
        return record


def approve_all_primary_proposals(
    shortlist: SemanticReviewShortlist,
    *,
    reviewer: str,
    reviewed_at: str,
) -> PrimaryReviewApproval:
    """Translate an explicit approve-all instruction into a bound record."""
    return PrimaryReviewApproval(
        shortlist_sha256=shortlist.shortlist_sha256,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        approved_proposal_ids=tuple(
            proposal.spec.proposal_id
            for proposal in shortlist.proposals
            if proposal.spec.priority == "primary"
        ),
    )


def publish_primary_review_approval(
    approval: PrimaryReviewApproval,
    output_root: str | Path,
) -> Path:
    """Atomically publish a canonical human approval record and summary."""
    root = Path(output_root) / "review-approvals" / approval.approval_sha256
    approval_payload = canonical_json_bytes(approval.as_record())
    summary_payload = _render_approval_markdown(approval).encode("utf-8")
    content_payloads = {
        "approval.json": approval_payload,
        "approval.md": summary_payload,
    }
    manifest_payload = canonical_json_bytes(
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
            "format": APPROVAL_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "shortlist_sha256": approval.shortlist_sha256,
        }
    )
    payloads = {**content_payloads, "manifest.json": manifest_payload}
    if root.exists():
        if (
            {path.name for path in root.iterdir()} != set(payloads)
            or any(
                not (root / name).is_file()
                or (root / name).read_bytes() != payload
                for name, payload in payloads.items()
            )
        ):
            raise FileExistsError("existing primary approval changed")
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".approval-", dir=root.parent))
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


def load_primary_review_approval(directory: str | Path) -> PrimaryReviewApproval:
    """Strictly authenticate and reconstruct one primary approval artifact."""
    root = Path(directory)
    manifest = _load_canonical_json(root / "manifest.json")
    if (
        manifest.get("format") != APPROVAL_FORMAT
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("primary approval manifest identity changed")
    file_records = manifest.get("files")
    if type(file_records) is not list or any(type(item) is not dict for item in file_records):
        raise ValueError("primary approval manifest files are invalid")
    expected_names = {"manifest.json", *(str(item.get("name")) for item in file_records)}
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("primary approval tree entries changed")
    for item in file_records:
        name = item.get("name")
        size_bytes = item.get("size_bytes")
        expected_sha256 = item.get("sha256")
        if type(name) is not str or type(size_bytes) is not int or type(expected_sha256) is not str:
            raise ValueError("primary approval file identity is invalid")
        payload = (root / name).read_bytes()
        if len(payload) != size_bytes or sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("primary approval file changed")
    record = _load_canonical_json(root / "approval.json")
    decisions = record.get("decisions")
    if type(decisions) is not list or any(type(item) is not dict for item in decisions):
        raise ValueError("primary approval decisions are invalid")
    required_gates = (
        "answer_forms_approved",
        "distractors_approved",
        "evidence_approved",
        "trigger_closure_approved",
        "truth_approved",
    )
    if any(any(item.get(gate) is not True for gate in required_gates) for item in decisions):
        raise ValueError("primary approval contains a non-affirmative gate")
    approval = PrimaryReviewApproval(
        shortlist_sha256=_text(record, "shortlist_sha256"),
        reviewer=_text(record, "reviewer"),
        reviewed_at=_text(record, "reviewed_at"),
        approved_proposal_ids=tuple(_text(item, "proposal_id") for item in decisions),
    )
    if (
        record != approval.as_record()
        or manifest.get("approval_sha256") != approval.approval_sha256
        or manifest.get("shortlist_sha256") != approval.shortlist_sha256
        or root.name != approval.approval_sha256
        or (root / "approval.md").read_text(encoding="utf-8")
        != _render_approval_markdown(approval)
    ):
        raise ValueError("primary approval semantic content changed")
    return approval


def _render_approval_markdown(approval: PrimaryReviewApproval) -> str:
    return (
        "# TinyWorlds-Q primary review approval\n\n"
        f"Approval: `{approval.approval_sha256}`  \n"
        f"Shortlist: `{approval.shortlist_sha256}`  \n"
        f"Reviewer: `{approval.reviewer}`  \n"
        f"Reviewed at: `{approval.reviewed_at}`\n\n"
        f"All five review gates are affirmative for all "
        f"{len(approval.approved_proposal_ids)} primary proposals.\n"
    )


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid primary approval JSON: {path.name}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"primary approval JSON is not canonical: {path.name}")
    return value


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"primary approval {field} must be nonempty text")
    return value


__all__ = [
    "PrimaryReviewApproval",
    "approve_all_primary_proposals",
    "load_primary_review_approval",
    "publish_primary_review_approval",
]
